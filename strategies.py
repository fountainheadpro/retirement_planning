from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class StrategyContext:
    """Context object passed to strategy methods containing current simulation state."""
    current_cash: np.ndarray
    current_equity: np.ndarray
    panic_mask: np.ndarray
    desired_withdrawal: np.ndarray
    # Market context
    market_index: np.ndarray
    market_peak: np.ndarray
    # Configuration
    target_cash_level: float  # For replenishment targets

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
    def determine_withdrawal_source(self, ctx: StrategyContext) -> tuple[np.ndarray, np.ndarray]:
        """
        Determines how to split the desired withdrawal between Cash and Equity.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: (amount_from_cash, amount_from_equity)
        """
        pass

    @abstractmethod
    def post_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        """
        Calculates transfers between Equity and Cash AFTER withdrawals occur (e.g., Replenishment).
        
        Returns:
            np.ndarray: Amount to transfer from Equity to Cash.
        """
        pass


class ConservativeStrategy(CashStrategy):
    """
    Standard Strategy:
    1. Protect Withdrawals: Use Cash first during Panic/Drawdown.
    2. Replenish: Refill cash buffer from Equity only when market recovers (High Water Mark).
    """
    def pre_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        # No action before withdrawal
        return np.zeros_like(ctx.current_cash)

    def determine_withdrawal_source(self, ctx: StrategyContext) -> tuple[np.ndarray, np.ndarray]:
        n_paths = len(ctx.current_cash)
        from_cash = np.zeros(n_paths)
        from_equity = np.zeros(n_paths)
        
        has_cash = ctx.current_cash > 0
        # Panic & Has Cash -> Use Cash First
        use_cash_mask = ctx.panic_mask & has_cash
        
        # 1. Panic Case
        if np.any(use_cash_mask):
            from_cash[use_cash_mask] = np.minimum(ctx.desired_withdrawal[use_cash_mask], ctx.current_cash[use_cash_mask])
            from_equity[use_cash_mask] = ctx.desired_withdrawal[use_cash_mask] - from_cash[use_cash_mask]
            
        # 2. Normal Case (or Cash Depleted)
        use_equity_mask = ~use_cash_mask
        if np.any(use_equity_mask):
            from_equity[use_equity_mask] = np.minimum(ctx.desired_withdrawal[use_equity_mask], ctx.current_equity[use_equity_mask])
            from_cash[use_equity_mask] = ctx.desired_withdrawal[use_equity_mask] - from_equity[use_equity_mask]
            
        return from_cash, from_equity

    def post_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        # Replenish Logic
        # Rule: Only replenish if we are at or above the Market High Water Mark
        n_paths = len(ctx.current_cash)
        transfers = np.zeros(n_paths)
        
        at_peak = ctx.market_index >= (ctx.market_peak * 0.999)
        replenish_mask = at_peak & (ctx.current_cash < ctx.target_cash_level)
        
        if np.any(replenish_mask):
            shortfall = ctx.target_cash_level - ctx.current_cash[replenish_mask]
            transfers[replenish_mask] = np.minimum(shortfall, ctx.current_equity[replenish_mask])
            
        return transfers


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

    def determine_withdrawal_source(self, ctx: StrategyContext) -> tuple[np.ndarray, np.ndarray]:
        # Always prioritize Equity First
        # (If we just bought the dip, cash is 0 anyway)
        n_paths = len(ctx.current_cash)
        from_cash = np.zeros(n_paths)
        from_equity = np.zeros(n_paths)
        
        from_equity = np.minimum(ctx.desired_withdrawal, ctx.current_equity)
        from_cash = ctx.desired_withdrawal - from_equity # Remainder from cash
        
        return from_cash, from_equity

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

    def determine_withdrawal_source(self, ctx: StrategyContext) -> tuple[np.ndarray, np.ndarray]:
        n_paths = len(ctx.current_cash)
        from_cash = np.zeros(n_paths)
        from_equity = np.zeros(n_paths)
        
        # Equity First
        from_equity = np.minimum(ctx.desired_withdrawal, ctx.current_equity)
        from_cash = ctx.desired_withdrawal - from_equity
        
        return from_cash, from_equity

    def post_withdrawal_rebalance(self, ctx: StrategyContext) -> np.ndarray:
        return np.zeros_like(ctx.current_cash)
