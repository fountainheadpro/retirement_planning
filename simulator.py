import numpy as np
import pandas as pd
# statsmodels for AR(p) mean-reversion market model calibration
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. MARKET MODELS (AR(p) on Residuals)
# ==========================================

def resolve_rng(rng: np.random.Generator | int | None = None) -> np.random.Generator:
    """Return a numpy Generator from an existing generator, integer seed, or entropy."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng()
    return np.random.default_rng(int(rng))


def block_indices(
    n_history: int,
    block_size: int,
    years: int,
    n_paths: int,
    rng: np.random.Generator,
    mode: str = "overlapping",
) -> np.ndarray:
    """Build (years, n_paths) historical indices for a block bootstrap."""
    if n_history < block_size:
        raise ValueError(f"History length {n_history} is shorter than block size {block_size}")
    n_blocks = int(np.ceil(years / block_size))
    offsets = np.arange(block_size)[:, None]
    if mode == "overlapping":
        max_start = n_history - block_size
        starts = rng.integers(0, max_start + 1, size=(n_blocks, n_paths))
        raw = np.empty((n_blocks * block_size, n_paths), dtype=np.int32)
        for block_idx in range(n_blocks):
            begin = block_idx * block_size
            raw[begin:begin + block_size, :] = starts[block_idx, :] + offsets
        return raw[:years, :]
    if mode == "circular":
        starts = rng.integers(0, n_history, size=(n_blocks, n_paths))
        raw = np.empty((n_blocks * block_size, n_paths), dtype=np.int32)
        for block_idx in range(n_blocks):
            begin = block_idx * block_size
            raw[begin:begin + block_size, :] = (starts[block_idx, :] + offsets) % n_history
        return raw[:years, :]
    if mode == "nonoverlapping":
        grid = np.arange(0, n_history - block_size + 1, block_size)
        if len(grid) == 0:
            grid = np.array([0])
        starts = rng.choice(grid, size=(n_blocks, n_paths))
        raw = np.empty((n_blocks * block_size, n_paths), dtype=np.int32)
        for block_idx in range(n_blocks):
            begin = block_idx * block_size
            raw[begin:begin + block_size, :] = starts[block_idx, :] + offsets
        return raw[:years, :]
    raise ValueError(f"Unknown block mode: {mode}")


def apply_erp_haircut(
    stock_returns: np.ndarray,
    reference_returns: np.ndarray,
    haircut: float,
) -> np.ndarray:
    """Keep excess-return sequence and volatility; scale only the mean premium.

    e_t = r_stock - r_bill
    e'_t = (e_t - mean(e)) + haircut * mean(e)
    r'_stock = r_bill + e'_t
    """
    stock = np.asarray(stock_returns, dtype=float)
    reference = np.asarray(reference_returns, dtype=float)
    excess = stock - reference
    mean_excess = float(np.mean(excess))
    adjusted = (excess - mean_excess) + float(haircut) * mean_excess
    return reference + adjusted


def tips_proxy_returns(inflation_rates: np.ndarray) -> np.ndarray:
    """Nominal returns that are 0% real after CPI deflation."""
    return np.asarray(inflation_rates, dtype=float)


class RandomWalkMarket:
    """
    Models the market using simple random sampling from historical residuals (Random Walk).
    """
    def __init__(self, mu, residuals):
        self.mu = mu
        self.residuals = np.asarray(residuals).ravel()
        
    def simulate_matrix(self, years, n_paths, rng=None):
        """
        Generates a matrix of market returns using random sampling.
        Returns: array of shape (years, n_paths)
        """
        rng = resolve_rng(rng)
        random_returns = rng.choice(self.residuals, size=(years, n_paths))
        return self.mu + random_returns

class BlockBootstrapMarket:
    """
    Models the market by resampling blocks of historical returns to preserve
    correlation structure (autocorrelation, volatility clustering).
    """
    def __init__(
        self,
        history_returns,
        block_size=5,
        inflation_rates=None,
        cash_returns=None,
    ):
        self.block_size = block_size
        self.history = np.asarray(history_returns).ravel()
        self.inflation_history = (
            np.asarray(inflation_rates).ravel()
            if inflation_rates is not None
            else None
        )
        self.cash_history = (
            np.asarray(cash_returns).ravel()
            if cash_returns is not None
            else None
        )
        self.block_mode = "overlapping"
        if len(self.history) < self.block_size:
            raise ValueError(f"History length {len(self.history)} is shorter than block size {self.block_size}")
        if self.inflation_history is not None and len(self.inflation_history) != len(self.history):
            raise ValueError("Inflation history must have the same length as return history")
        if self.cash_history is not None and len(self.cash_history) != len(self.history):
            raise ValueError("Cash return history must have the same length as return history")
        
    def simulate_matrix(self, years, n_paths, rng=None, block_mode: str | None = None):
        """
        Generates a matrix of market returns using block bootstrapping.
        Returns: array of shape (years, n_paths)
        """
        rng = resolve_rng(rng)
        mode = block_mode or self.block_mode
        indices = block_indices(
            len(self.history), self.block_size, years, n_paths, rng, mode=mode
        )
        market_matrix = self.history[indices]
        if self.inflation_history is None and self.cash_history is None:
            return market_matrix
        result = {"stock": market_matrix}
        if self.inflation_history is not None:
            result["inflation"] = self.inflation_history[indices]
        if self.cash_history is not None:
            result["cash"] = self.cash_history[indices]
        return result

class PairedBlockBootstrapMarket:
    """
    Models paired stock, bond, cash, and inflation records by resampling matching historical blocks.
    """
    def __init__(
        self,
        stock_returns,
        bond_returns,
        block_size=5,
        inflation_rates=None,
        cash_returns=None,
    ):
        self.block_size = block_size
        self.stock_history = np.asarray(stock_returns).ravel()
        self.bond_history = np.asarray(bond_returns).ravel()
        self.inflation_history = (
            np.asarray(inflation_rates).ravel()
            if inflation_rates is not None
            else None
        )
        self.cash_history = (
            np.asarray(cash_returns).ravel()
            if cash_returns is not None
            else None
        )

        if len(self.stock_history) != len(self.bond_history):
            raise ValueError("Stock and bond histories must have the same length")
        if self.inflation_history is not None and len(self.inflation_history) != len(self.stock_history):
            raise ValueError("Inflation history must have the same length as stock and bond histories")
        if self.cash_history is not None and len(self.cash_history) != len(self.stock_history):
            raise ValueError("Cash return history must have the same length as stock and bond histories")
        if len(self.stock_history) < self.block_size:
            raise ValueError(f"History length {len(self.stock_history)} is shorter than block size {self.block_size}")
        self.block_mode = "overlapping"

    def simulate_matrix(self, years, n_paths, rng=None, block_mode: str | None = None):
        """
        Generates paired stock, bond, cash, and inflation matrices using block bootstrapping.
        Returns: dict with arrays of shape (years, n_paths)
        """
        rng = resolve_rng(rng)
        mode = block_mode or self.block_mode
        indices = block_indices(
            len(self.stock_history), self.block_size, years, n_paths, rng, mode=mode
        )
        result = {
            "stock": self.stock_history[indices],
            "bond": self.bond_history[indices],
        }
        if self.inflation_history is not None:
            result["inflation"] = self.inflation_history[indices]
        if self.cash_history is not None:
            result["cash"] = self.cash_history[indices]
        return result

class MeanRevertingMarket:
    """
    Models the market using an AR(p) process on Returns.
    """
    def __init__(self, ar_order=1):
        self.ar_order = ar_order
        self.ar_coeffs = None 
        self.intercept = None
        self.residual_std = None
        self.history_window = np.zeros(ar_order) 
        self.full_history = None 

    def calibrate_from_history(self, historical_returns):
        # ... (keep existing implementation) ...
        # FIX: Ensure data is 1D flat array. yfinance sometimes returns (N, 1) which breaks matrix math.
        data = np.array(historical_returns).ravel()
        self.full_history = data
        p = self.ar_order
        
        if len(data) < p + 10:
            raise ValueError(f"Not enough data for AR({p}). Need at least {p+10} years.")

        try:
            model = ARIMA(data, order=(p, 0, 0), trend='c')
            res = model.fit()
            
            self.intercept = res.params[0]
            self.ar_coeffs = res.arparams 
            # statsmodels ARIMA (p,0,0) with trend='c' places the innovation
            # variance (sigma²) in the last parameter. We store the std dev.
            self.residual_std = np.sqrt(res.params[-1]) 
            
            # Set State (The most recent 'p' years from history)
            self.history_window = data[-p:][::-1] 
            
            denom = (1 - np.sum(self.ar_coeffs))
            long_term_mean = self.intercept / denom if abs(denom) > 1e-5 else 0.0
            
            return {
                "ar_coeffs": self.ar_coeffs,
                "intercept": self.intercept,
                "mean_return": long_term_mean,
                "volatility": self.residual_std
            }
            
        except Exception as e:
            # Pure function: propagate a clear error. Caller (app or notebook)
            # decides how to present it to the user.
            raise RuntimeError(
                f"ARIMA({p}) calibration failed on {len(data)} observations: {e}"
            ) from e

    def simulate_year(self, history_window, simulations=1, rng=None):
        if self.ar_coeffs is None:
            raise ValueError("Model not calibrated.")

        deterministic_part = self.intercept + np.dot(history_window, self.ar_coeffs)
        noise = resolve_rng(rng).normal(0, self.residual_std, simulations)
        return deterministic_part + noise

    def simulate_matrix(self, years, n_paths, rng=None):
        """
        Generates a matrix of market returns using AR(p) simulation.
        Returns: array of shape (years, n_paths)
        """
        if self.ar_coeffs is None:
            raise ValueError("Model not calibrated.")

        rng = resolve_rng(rng)
        market_matrix = np.zeros((years, n_paths))
        start_window = self.history_window
        if self.full_history is not None:
            p = self.ar_order
            if len(self.full_history) >= p:
                start_window = self.full_history[-p:][::-1]

        current_history_windows = np.tile(start_window.reshape(1, -1), (n_paths, 1))
        for t in range(years):
            market_return = self.simulate_year(
                current_history_windows, simulations=n_paths, rng=rng
            )
            market_matrix[t, :] = market_return
            current_history_windows = np.roll(current_history_windows, shift=1, axis=1)
            current_history_windows[:, 0] = market_return
        return market_matrix

def get_sp500_data(history_years: int = 60) -> np.ndarray:
    """
    Load historical S&P 500 total returns (decimal) from local CSV.
    
    Pure function with no UI dependencies. Raises on failure so callers
    (Streamlit app, notebooks, scripts) can handle errors appropriately.
    """
    path = "s_and_p_500_with_dividends.csv"
    try:
        df = pd.read_csv(path, header=None, names=["Year", "Return"])
        df["Return"] = df["Return"] / 100.0
        df = df.sort_values("Year")
        if len(df) > history_years:
            df = df.tail(history_years)
        arr = df["Return"].to_numpy(dtype=float)
        if len(arr) == 0:
            raise ValueError("No data loaded after filtering.")
        return arr
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required data file not found: {path}. "
            "Place it in the project root for AR model calibration."
        ) from None
    except Exception as e:
        raise RuntimeError(f"Failed to load S&P 500 data from {path}: {e}") from e

def get_stock_bond_data(
    history_years: int = 75, bond_column: str = "TreasuryBondReturn"
) -> dict:
    """
    Load paired annual stock, bond, T-bill, and inflation records from local CSVs.
    
    Pure function, no UI dependencies or caching. Raises on missing columns
    or data problems. Callers (e.g. the Streamlit app) add @st.cache_data.
    """
    returns = pd.read_csv("historical_asset_returns.csv")
    inflation = pd.read_csv("historical_inflation.csv")
    df = returns.merge(inflation, on="Year", how="inner", validate="one_to_one")
    df = df.sort_values("Year")

    if bond_column not in df.columns:
        raise ValueError(f"Unknown bond return column: {bond_column}")

    if len(df) > history_years:
        df = df.tail(history_years)

    return {
        "years": df["Year"].to_numpy(dtype=int),
        "stock_returns": df["StockReturn"].to_numpy(dtype=float),
        "bond_returns": df[bond_column].to_numpy(dtype=float),
        "inflation_rates": df["InflationRate"].to_numpy(dtype=float),
        "tbill_returns": df["TBillReturn"].to_numpy(dtype=float),
        "treasury_bond_returns": df["TreasuryBondReturn"].to_numpy(dtype=float),
        "corporate_bond_returns": df["CorporateBondReturn"].to_numpy(dtype=float),
    }

def create_ar_model(history_years: int = 50, ar_order: int = 1) -> tuple[MeanRevertingMarket, dict]:
    """
    Create and calibrate an AR(p) MeanRevertingMarket using statsmodels.
    
    Pure function (no Streamlit, no caching). Returns (model, stats_dict) on success.
    Raises RuntimeError or ValueError on data load or calibration failure.
    
    The caller (e.g. Streamlit app) is responsible for formatting stats for display
    and for any caching / error presentation.
    """
    model = MeanRevertingMarket(ar_order=ar_order)
    
    # Use full available history for best calibration of long cycles
    hist_returns = get_sp500_data(history_years=history_years)
    
    stats = model.calibrate_from_history(hist_returns)
    # On success calibrate always returns a dict; it raises on failure.
    
    return model, stats

def get_sp500_residuals(history_years: int):
    """
    Compute mean and residuals for Random Walk / residual sampling.
    Pure function; raises if underlying data load fails.
    """
    hist = get_sp500_data(history_years)
    mu = float(np.mean(hist))
    residuals = hist - mu
    return mu, residuals, hist

from strategies import (
    CashStrategy,
    ConservativeStrategy,
    StrategyContext,
    apply_cash_equity_transfer,
)
from spending import (
    apply_target_cap_floor,
    build_spending_reference_table,
    derive_spending_targets,
)
from metrics import calculate_statistics

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================


def run_simulation(
    initial_net_worth: float,
    annual_spend: float,
    buffer_years: int,
    years: int,
    panic_threshold: float,
    inflation_rate: float,
    n_paths: int,
    market_model,
    spending_cap_pct: float = 0.04,
    cash_interest_rate: float | None = None,
    strategy: CashStrategy | None = None,
    minimum_annual_spend: float = 0.0,
    bond_allocation_pct: float = 0.0,
    random_seed: int | None = None,
    withdraw_before_returns: bool = False,
    block_mode: str | None = None,
) -> dict:
    """
    Run the full Monte Carlo retirement simulation.

    If random_seed is provided, a numpy Generator is created and passed into
    the market model. This does not touch global numpy state.
    """
    rng = resolve_rng(random_seed) if random_seed is not None else None
    if strategy is None:
        strategy = ConservativeStrategy()

    use_sampled_cash_returns = cash_interest_rate is None

    simulate_kwargs = {"years": years, "n_paths": n_paths}
    if rng is not None:
        simulate_kwargs["rng"] = rng
    if block_mode is not None:
        simulate_kwargs["block_mode"] = block_mode
    try:
        market_returns_matrix = market_model.simulate_matrix(**simulate_kwargs)
    except TypeError:
        simulate_kwargs.pop("block_mode", None)
        try:
            market_returns_matrix = market_model.simulate_matrix(**simulate_kwargs)
        except TypeError:
            market_returns_matrix = market_model.simulate_matrix(years, n_paths)
    if isinstance(market_returns_matrix, dict):
        stock_returns_matrix = market_returns_matrix["stock"]
        bond_returns_matrix = market_returns_matrix.get(
            "bond",
            np.zeros_like(stock_returns_matrix),
        )
        inflation_matrix = market_returns_matrix.get(
            "inflation",
            np.full_like(stock_returns_matrix, float(inflation_rate)),
        )
        cash_returns_matrix = market_returns_matrix.get("cash")
    else:
        stock_returns_matrix = market_returns_matrix
        bond_returns_matrix = np.zeros_like(stock_returns_matrix)
        inflation_matrix = np.full_like(stock_returns_matrix, float(inflation_rate))
        cash_returns_matrix = None

    bond_allocation_pct = min(max(bond_allocation_pct, 0.0), 1.0)

    # Initial Allocation
    # Note: Strategy-specific overrides (like No Buffer forcing 0) should be handled
    # by the caller (app.py) setting buffer_years=0, or we could ask strategy,
    # but keeping it simple: Caller configures buffer_years appropriately.
        
    initial_cash_target = annual_spend * buffer_years
    initial_cash = min(initial_cash_target, initial_net_worth)
    initial_investable = initial_net_worth - initial_cash
    initial_bonds = initial_investable * bond_allocation_pct
    initial_equity = initial_investable - initial_bonds
    
    # State Arrays (All in Real Dollars)
    portfolio_values = np.zeros((years + 1, n_paths))
    cash_values = np.zeros((years + 1, n_paths))
    equity_values = np.zeros((years + 1, n_paths))
    bond_values = np.zeros((years + 1, n_paths))
    
    portfolio_values[0, :] = initial_net_worth
    cash_values[0, :] = initial_cash
    equity_values[0, :] = initial_equity
    bond_values[0, :] = initial_bonds
    
    # Detailed tracking
    withdrawals = np.zeros((years, n_paths))
    market_returns = np.zeros((years, n_paths))
    panic_flags = np.zeros((years, n_paths), dtype=bool)
    withdrawals_from_cash = np.zeros((years, n_paths))
    withdrawals_from_equity = np.zeros((years, n_paths))
    withdrawals_from_bonds = np.zeros((years, n_paths))
    replenishments = np.zeros((years, n_paths))
    inflation_rates = np.zeros((years, n_paths))
    cash_returns = np.zeros((years, n_paths))
    
    # Reset Arrays
    current_equity = np.full(n_paths, float(initial_equity))
    current_bonds = np.full(n_paths, float(initial_bonds))
    current_cash = np.full(n_paths, float(initial_cash))
    
    # Track Market High Water Mark
    market_index = np.ones(n_paths)
    market_peak = np.ones(n_paths)
    previous_market_return = np.zeros(n_paths)

    def apply_year_returns(year_idx: int) -> None:
        nonlocal current_equity, current_bonds, current_cash, market_index, market_peak
        market_return_nominal = stock_returns_matrix[year_idx, :]
        bond_return_nominal = bond_returns_matrix[year_idx, :]
        inflation_nominal = inflation_matrix[year_idx, :]
        inflation_rates[year_idx, :] = inflation_nominal
        if use_sampled_cash_returns and cash_returns_matrix is not None:
            cash_return_nominal = cash_returns_matrix[year_idx, :]
        elif use_sampled_cash_returns:
            cash_return_nominal = inflation_nominal
        else:
            cash_return_nominal = cash_interest_rate
        cash_returns[year_idx, :] = cash_return_nominal
        market_returns[year_idx, :] = market_return_nominal

        market_index *= (1 + market_return_nominal)
        market_peak = np.maximum(market_peak, market_index)

        real_market_return = (1 + market_return_nominal) / (1 + inflation_nominal) - 1
        real_bond_return = (1 + bond_return_nominal) / (1 + inflation_nominal) - 1
        real_cash_return = (1.0 + cash_return_nominal) / (1.0 + inflation_nominal) - 1.0
        current_equity = np.maximum(0.0, current_equity * (1 + real_market_return))
        current_bonds = np.maximum(0.0, current_bonds * (1 + real_bond_return))
        current_cash = np.maximum(0.0, current_cash * (1 + real_cash_return))

    def apply_strategy_step(year_idx: int, panic_return: np.ndarray) -> None:
        nonlocal current_equity, current_bonds, current_cash
        panic_mask = (panic_return < panic_threshold) | (
            market_index < (market_peak * 0.999)
        )
        panic_flags[year_idx, :] = panic_mask
        total_liquid_assets = current_equity + current_bonds + current_cash
        desired_withdrawal = apply_target_cap_floor(
            total_liquid_assets,
            annual_spend,
            spending_cap_pct,
            minimum_annual_spend,
        )
        target_cash_level = annual_spend * buffer_years
        ctx = StrategyContext(
            current_cash=current_cash,
            current_equity=current_equity,
            current_bonds=current_bonds,
            panic_mask=panic_mask,
            desired_withdrawal=desired_withdrawal,
            market_index=market_index,
            market_peak=market_peak,
            target_cash_level=target_cash_level,
            bond_allocation_pct=bond_allocation_pct,
            floor_spend=float(minimum_annual_spend),
        )
        pre_transfer = strategy.pre_withdrawal_rebalance(ctx)
        current_cash, current_equity, _ = apply_cash_equity_transfer(
            current_cash, current_equity, pre_transfer
        )
        ctx.current_cash = current_cash
        ctx.current_equity = current_equity
        from_cash, from_bonds, from_equity = strategy.determine_withdrawal_source(ctx)
        current_cash = current_cash - from_cash
        current_bonds = current_bonds - from_bonds
        current_equity = current_equity - from_equity
        ctx.current_cash = current_cash
        ctx.current_bonds = current_bonds
        ctx.current_equity = current_equity
        withdrawals_from_cash[year_idx, :] = from_cash
        withdrawals_from_bonds[year_idx, :] = from_bonds
        withdrawals_from_equity[year_idx, :] = from_equity
        withdrawals[year_idx, :] = from_cash + from_bonds + from_equity

        post_transfer = strategy.post_withdrawal_rebalance(ctx)
        current_cash, current_equity, realized_post = apply_cash_equity_transfer(
            current_cash, current_equity, post_transfer
        )
        ctx.current_cash = current_cash
        ctx.current_equity = current_equity
        bond_to_cash = strategy.post_withdrawal_bond_transfer(ctx)
        bond_to_cash = np.minimum(np.maximum(bond_to_cash, 0.0), current_bonds)
        current_cash = current_cash + bond_to_cash
        current_bonds = current_bonds - bond_to_cash
        ctx.current_cash = current_cash
        ctx.current_bonds = current_bonds
        replenishments[year_idx, :] = realized_post + bond_to_cash

        current_equity, current_bonds = strategy.rebalance_invested_assets(ctx)

    for t in range(1, years + 1):
        year_idx = t - 1
        if withdraw_before_returns:
            apply_strategy_step(year_idx, previous_market_return)
            apply_year_returns(year_idx)
        else:
            apply_year_returns(year_idx)
            apply_strategy_step(year_idx, market_returns[year_idx, :])
        previous_market_return = market_returns[year_idx, :]

        portfolio_values[t, :] = current_equity + current_bonds + current_cash
        cash_values[t, :] = current_cash
        equity_values[t, :] = current_equity
        bond_values[t, :] = current_bonds

    return {
        'portfolio_values': portfolio_values,
        'withdrawal_values': withdrawals,
        'cash_values': cash_values,
        'equity_values': equity_values,
        'bond_values': bond_values,
        'market_returns': market_returns,
        'inflation_rates': inflation_rates,
        'cash_returns': cash_returns,
        'panic_flags': panic_flags,
        'withdrawals_from_cash': withdrawals_from_cash,
        'withdrawals_from_bonds': withdrawals_from_bonds,
        'withdrawals_from_equity': withdrawals_from_equity,
        'replenishments': replenishments
    }
