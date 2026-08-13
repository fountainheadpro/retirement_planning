from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class StrategyContext:
    """Context object passed to strategy methods containing current simulation state."""
    current_cash: np.ndarray
    current_equity: np.ndarray
    current_bonds: np.ndarray
    panic_mask: np.ndarray
    desired_withdrawal: np.ndarray
    # Market context
    market_index: np.ndarray
    market_peak: np.ndarray
    # Configuration
    target_cash_level: float  # For replenishment targets
    bond_allocation_pct: float = 0.0
    floor_spend: float = 0.0

def apply_cash_equity_transfer(
    cash: np.ndarray,
    equity: np.ndarray,
    transfer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a signed equity<->cash transfer, capped by available balances.

    Positive transfer moves equity to cash. Negative moves cash to equity.
    Returns the updated cash, equity, and realized transfer.
    """
    realized = np.zeros_like(transfer)
    to_equity = transfer < 0
    to_cash = transfer > 0
    if np.any(to_equity):
        amount = np.minimum(-transfer[to_equity], cash[to_equity])
        realized[to_equity] = -amount
    if np.any(to_cash):
        amount = np.minimum(transfer[to_cash], equity[to_cash])
        realized[to_cash] = amount
    cash = cash + realized
    equity = equity - realized
    return cash, equity, realized

class CashStrategy(ABC):
    """Abstract base class for cash management strategies."""

    @abstractmethod
    def pre_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        """
        Calculates transfers between Equity and Cash BEFORE withdrawals occur.
        
        Returns:
            np.ndarray: Amount to transfer from Equity to Cash. 
                        Positive = Equity -> Cash (Sell Stocks)
                        Negative = Cash -> Equity (Buy Stocks)
                        Zero = No action
        """
        pass

    @abstractmethod
    def determine_withdrawal_source(
        self, ctx: StrategyContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split the desired withdrawal across cash, bonds, and equity."""
        pass

    @abstractmethod
    def post_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        """
        Calculates transfers between Equity and Cash AFTER withdrawals occur (e.g., Replenishment).
        
        Returns:
            np.ndarray: Amount to transfer from Equity to Cash.
        """
        pass

    def post_withdrawal_bond_transfer(self, ctx: StrategyContext) -> np.ndarray:
        """Amount to move from bonds to cash after the equity replenishment step."""
        return np.zeros_like(ctx.current_cash)

    def rebalance_invested_assets(
        self, ctx: StrategyContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (equity, bonds) after any target-allocation rebalance."""
        return ctx.current_equity, ctx.current_bonds


class ConservativeStrategy(CashStrategy):
    """
    Standard Strategy:
    1. Protect Withdrawals: Use Cash first during Panic/Drawdown.
    2. Replenish: Refill cash buffer from Equity only when market recovers (High Water Mark).
    """
    def pre_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        # No action before withdrawal
        return np.zeros_like(ctx.current_cash)

    def determine_withdrawal_source(
        self, ctx: StrategyContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_paths = len(ctx.current_cash)
        remaining = ctx.desired_withdrawal.copy()
        from_cash = np.zeros(n_paths)
        from_bonds = np.zeros(n_paths)
        from_equity = np.zeros(n_paths)

        panic_with_cash = ctx.panic_mask & (ctx.current_cash > 0)
        if np.any(panic_with_cash):
            from_cash[panic_with_cash] = np.minimum(
                remaining[panic_with_cash],
                ctx.current_cash[panic_with_cash],
            )
            remaining -= from_cash

        panic_with_bonds = ctx.panic_mask & (remaining > 0)
        if np.any(panic_with_bonds):
            from_bonds[panic_with_bonds] = np.minimum(
                remaining[panic_with_bonds],
                ctx.current_bonds[panic_with_bonds],
            )
            remaining -= from_bonds

        use_equity = remaining > 0
        if np.any(use_equity):
            from_equity[use_equity] = np.minimum(
                remaining[use_equity],
                ctx.current_equity[use_equity],
            )
            remaining -= from_equity

        use_bonds_normal = (~ctx.panic_mask) & (remaining > 0)
        if np.any(use_bonds_normal):
            take = np.minimum(
                remaining[use_bonds_normal],
                ctx.current_bonds[use_bonds_normal],
            )
            from_bonds[use_bonds_normal] += take
            remaining[use_bonds_normal] -= take

        use_cash_normal = (~ctx.panic_mask) & (remaining > 0)
        if np.any(use_cash_normal):
            from_cash[use_cash_normal] += np.minimum(
                remaining[use_cash_normal],
                ctx.current_cash[use_cash_normal],
            )

        return from_cash, from_bonds, from_equity

    def post_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        n_paths = len(ctx.current_cash)
        transfers = np.zeros(n_paths)
        at_peak = ctx.market_index >= (ctx.market_peak * 0.999)
        replenish_mask = at_peak & (ctx.current_cash < ctx.target_cash_level)
        if np.any(replenish_mask):
            shortfall = ctx.target_cash_level - ctx.current_cash[replenish_mask]
            transfers[replenish_mask] = np.minimum(shortfall, ctx.current_equity[replenish_mask])
        return transfers

    def post_withdrawal_bond_transfer(self, ctx: StrategyContext) -> np.ndarray:
        n_paths = len(ctx.current_cash)
        transfers = np.zeros(n_paths)
        at_peak = ctx.market_index >= (ctx.market_peak * 0.999)
        replenish_mask = at_peak & (ctx.current_cash < ctx.target_cash_level)
        if np.any(replenish_mask):
            shortfall = ctx.target_cash_level - ctx.current_cash[replenish_mask]
            transfers[replenish_mask] = np.minimum(shortfall, ctx.current_bonds[replenish_mask])
        return transfers

    def rebalance_invested_assets(
        self, ctx: StrategyContext
    ) -> tuple[np.ndarray, np.ndarray]:
        if ctx.bond_allocation_pct <= 0.0:
            return ctx.current_equity, ctx.current_bonds
        equity = ctx.current_equity.copy()
        bonds = ctx.current_bonds.copy()
        invested = equity + bonds
        target_bonds = invested * ctx.bond_allocation_pct
        bond_shortfall = target_bonds - bonds
        buy = bond_shortfall > 0
        if np.any(buy):
            transfer = np.minimum(bond_shortfall[buy], equity[buy])
            bonds[buy] += transfer
            equity[buy] -= transfer
        sell = bond_shortfall < 0
        if np.any(sell):
            transfer = np.minimum(-bond_shortfall[sell], bonds[sell])
            bonds[sell] -= transfer
            equity[sell] += transfer
        return equity, bonds


class AggressiveStrategy(CashStrategy):
    """
    Buy The Dip Strategy:
    1. Buy Dip: If Panic, move ALL Cash -> Equity immediately.
    2. Withdraw: Always from Equity (since cash is deployed or prioritized for buying).
    3. Replenish: Refill cash buffer from Equity when market recovers.
    """
    def pre_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        n_paths = len(ctx.current_cash)
        transfers = np.zeros(n_paths)
        
        # Panic & Has Cash -> Buy Equity (Negative Transfer)
        buy_mask = ctx.panic_mask & (ctx.current_cash > 0)
        if np.any(buy_mask):
            # Move all available cash to equity
            # Transfer is Equity->Cash, so moving Cash->Equity is negative
            transfers[buy_mask] = -ctx.current_cash[buy_mask]
            
        return transfers

    def determine_withdrawal_source(
        self, ctx: StrategyContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from_equity = np.minimum(ctx.desired_withdrawal, ctx.current_equity)
        remaining = ctx.desired_withdrawal - from_equity
        from_cash = np.minimum(remaining, ctx.current_cash)
        remaining = remaining - from_cash
        from_bonds = np.minimum(remaining, ctx.current_bonds)
        return from_cash, from_bonds, from_equity

    def post_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        # Identical replenishment logic to Conservative
        n_paths = len(ctx.current_cash)
        transfers = np.zeros(n_paths)
        
        at_peak = ctx.market_index >= (ctx.market_peak * 0.999)
        replenish_mask = at_peak & (ctx.current_cash < ctx.target_cash_level)
        
        if np.any(replenish_mask):
            shortfall = ctx.target_cash_level - ctx.current_cash[replenish_mask]
            transfers[replenish_mask] = np.minimum(shortfall, ctx.current_equity[replenish_mask])
            
        return transfers


class NoCashBufferStrategy(CashStrategy):
    """
    Fully Invested Strategy:
    1. No Buffer: Target cash is 0.
    2. Withdraw: Always from Equity.
    3. Replenish: Never.
    """
    def pre_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        return np.zeros_like(ctx.current_cash)

    def determine_withdrawal_source(
        self, ctx: StrategyContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from_equity = np.minimum(ctx.desired_withdrawal, ctx.current_equity)
        remaining = ctx.desired_withdrawal - from_equity
        from_cash = np.minimum(remaining, ctx.current_cash)
        remaining = remaining - from_cash
        from_bonds = np.minimum(remaining, ctx.current_bonds)
        return from_cash, from_bonds, from_equity

    def post_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        return np.zeros_like(ctx.current_cash)


class ProRataBondStrategy(ConservativeStrategy):
    """Spend invested assets pro-rata, then cash. Rebalance like Conservative."""

    def determine_withdrawal_source(
        self, ctx: StrategyContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_paths = len(ctx.current_cash)
        remaining = ctx.desired_withdrawal.copy()
        from_cash = np.zeros(n_paths)
        invested = ctx.current_equity + ctx.current_bonds
        invest_take = np.minimum(remaining, invested)
        equity_share = np.divide(
            ctx.current_equity,
            invested,
            out=np.zeros(n_paths),
            where=invested > 0,
        )
        from_equity = np.minimum(invest_take * equity_share, ctx.current_equity)
        from_bonds = np.minimum(invest_take - from_equity, ctx.current_bonds)
        remaining = remaining - from_equity - from_bonds
        from_cash = np.minimum(remaining, ctx.current_cash)
        return from_cash, from_bonds, from_equity


class FloorFundingStrategy(CashStrategy):
    """Pay the floor from a 0% real bucket that is allowed to run down.

    The initial sleeve size is set by bond_allocation_pct. This strategy does
    not rebalance that sleeve back to a percent of remaining wealth.
    """

    def pre_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        return np.zeros_like(ctx.current_cash)

    def determine_withdrawal_source(
        self, ctx: StrategyContext
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_paths = len(ctx.current_cash)
        remaining = ctx.desired_withdrawal.copy()
        floor_need = np.minimum(remaining, float(ctx.floor_spend)) if ctx.floor_spend > 0 else remaining
        from_bonds = np.minimum(floor_need, ctx.current_bonds)
        remaining = remaining - from_bonds
        from_equity = np.minimum(remaining, ctx.current_equity)
        remaining = remaining - from_equity
        leftover = np.minimum(remaining, ctx.current_bonds - from_bonds)
        from_bonds = from_bonds + leftover
        remaining = remaining - leftover
        from_cash = np.minimum(remaining, ctx.current_cash)
        return from_cash, from_bonds, from_equity

    def post_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        return np.zeros_like(ctx.current_cash)
