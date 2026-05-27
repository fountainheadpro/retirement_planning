import numpy as np
import pandas as pd
# statsmodels for AR(p) mean-reversion market model calibration
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# 1. MARKET MODELS (AR(p) on Residuals)
# ==========================================

class RandomWalkMarket:
    """
    Models the market using simple random sampling from historical residuals (Random Walk).
    """
    def __init__(self, mu, residuals):
        self.mu = mu
        self.residuals = np.asarray(residuals).ravel()
        
    def simulate_matrix(self, years, n_paths):
        """
        Generates a matrix of market returns using random sampling.
        Returns: array of shape (years, n_paths)
        """
        # Generate all random returns at once
        random_returns = np.random.choice(self.residuals, size=(years, n_paths))
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
        if len(self.history) < self.block_size:
            raise ValueError(f"History length {len(self.history)} is shorter than block size {self.block_size}")
        if self.inflation_history is not None and len(self.inflation_history) != len(self.history):
            raise ValueError("Inflation history must have the same length as return history")
        if self.cash_history is not None and len(self.cash_history) != len(self.history):
            raise ValueError("Cash return history must have the same length as return history")
        
    def simulate_matrix(self, years, n_paths):
        """
        Generates a matrix of market returns using block bootstrapping.
        Returns: array of shape (years, n_paths)
        """
        n_history = len(self.history)
        n_blocks = int(np.ceil(years / self.block_size))
        
        market_matrix = np.zeros((years, n_paths))
        inflation_matrix = (
            np.zeros((years, n_paths))
            if self.inflation_history is not None
            else None
        )
        cash_matrix = (
            np.zeros((years, n_paths))
            if self.cash_history is not None
            else None
        )
        
        for i in range(n_paths):
            path = []
            inflation_path = []
            cash_path = []
            for _ in range(n_blocks):
                # Pick a random start index
                start_idx = np.random.randint(0, n_history - self.block_size + 1)
                end_idx = start_idx + self.block_size
                block = self.history[start_idx:end_idx]
                path.extend(block)
                if self.inflation_history is not None:
                    inflation_path.extend(self.inflation_history[start_idx:end_idx])
                if self.cash_history is not None:
                    cash_path.extend(self.cash_history[start_idx:end_idx])
            
            # Trim to exact number of years and assign
            market_matrix[:, i] = path[:years]
            if inflation_matrix is not None:
                inflation_matrix[:, i] = inflation_path[:years]
            if cash_matrix is not None:
                cash_matrix[:, i] = cash_path[:years]
            
        if inflation_matrix is not None or cash_matrix is not None:
            result = {"stock": market_matrix}
            if inflation_matrix is not None:
                result["inflation"] = inflation_matrix
            if cash_matrix is not None:
                result["cash"] = cash_matrix
            return result

        return market_matrix

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

    def simulate_matrix(self, years, n_paths):
        """
        Generates paired stock, bond, cash, and inflation matrices using block bootstrapping.
        Returns: dict with arrays of shape (years, n_paths)
        """
        n_history = len(self.stock_history)
        n_blocks = int(np.ceil(years / self.block_size))

        stock_matrix = np.zeros((years, n_paths))
        bond_matrix = np.zeros((years, n_paths))
        inflation_matrix = (
            np.zeros((years, n_paths))
            if self.inflation_history is not None
            else None
        )
        cash_matrix = (
            np.zeros((years, n_paths))
            if self.cash_history is not None
            else None
        )

        for i in range(n_paths):
            stock_path = []
            bond_path = []
            inflation_path = []
            cash_path = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, n_history - self.block_size + 1)
                end_idx = start_idx + self.block_size
                stock_path.extend(self.stock_history[start_idx:end_idx])
                bond_path.extend(self.bond_history[start_idx:end_idx])
                if self.inflation_history is not None:
                    inflation_path.extend(self.inflation_history[start_idx:end_idx])
                if self.cash_history is not None:
                    cash_path.extend(self.cash_history[start_idx:end_idx])

            stock_matrix[:, i] = stock_path[:years]
            bond_matrix[:, i] = bond_path[:years]
            if inflation_matrix is not None:
                inflation_matrix[:, i] = inflation_path[:years]
            if cash_matrix is not None:
                cash_matrix[:, i] = cash_path[:years]

        result = {"stock": stock_matrix, "bond": bond_matrix}
        if inflation_matrix is not None:
            result["inflation"] = inflation_matrix
        if cash_matrix is not None:
            result["cash"] = cash_matrix
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

    def simulate_year(self, history_window, simulations=1):
        # ... (keep existing implementation) ...
        if self.ar_coeffs is None:
            raise ValueError("Model not calibrated.")

        deterministic_part = self.intercept + np.dot(history_window, self.ar_coeffs)
        noise = np.random.normal(0, self.residual_std, simulations)
        return deterministic_part + noise

    def simulate_matrix(self, years, n_paths):
        """
        Generates a matrix of market returns using AR(p) simulation.
        Returns: array of shape (years, n_paths)
        """
        if self.ar_coeffs is None:
            raise ValueError("Model not calibrated.")
            
        market_matrix = np.zeros((years, n_paths))
        
        # Determine start window
        # Default to the calibrated window (most recent)
        start_window = self.history_window

        # User request: Use the final 'p' values from full history (most recent data)
        if self.full_history is not None:
            p = self.ar_order
            if len(self.full_history) >= p:
                start_window = self.full_history[-p:][::-1]

        # Initialize windows: shape (n_paths, p)
        current_history_windows = np.tile(start_window.reshape(1, -1), (n_paths, 1))
        
        for t in range(years):
            # Simulate one step
            market_return = self.simulate_year(current_history_windows, simulations=n_paths)
            market_matrix[t, :] = market_return
            
            # Update History
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

from strategies import CashStrategy, ConservativeStrategy, StrategyContext

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================

def derive_spending_targets(
    initial_net_worth: float,
    spending_cap_pct: float,
    floor_ratio: float = 0.5,
) -> tuple[float, float]:
    """Derive real target and floor spending from initial wealth and cap."""
    target_spend = max(0.0, initial_net_worth * spending_cap_pct)
    floor_ratio = min(max(floor_ratio, 0.0), 1.0)
    return target_spend, target_spend * floor_ratio


def build_spending_reference_table(
    spending_cap_pct: float = 0.05,
    portfolio_values: tuple[float, ...] = (
        2_000_000,
        3_000_000,
        4_000_000,
        5_000_000,
        6_000_000,
        10_000_000,
    ),
    floor_ratio: float = 0.5,
) -> list[dict[str, str]]:
    """Build display-ready target/floor rows for common portfolio sizes."""
    target_label = f"Target ({spending_cap_pct:.0%})"
    floor_label = f"Floor ({spending_cap_pct * floor_ratio:.1%})"
    rows = []

    for portfolio_value in portfolio_values:
        target_spend, floor_spend = derive_spending_targets(
            initial_net_worth=portfolio_value,
            spending_cap_pct=spending_cap_pct,
            floor_ratio=floor_ratio,
        )
        rows.append(
            {
                "Portfolio": f"${portfolio_value / 1_000_000:g}M",
                target_label: f"${target_spend:,.0f}",
                floor_label: f"${floor_spend:,.0f}",
            }
        )

    return rows


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
) -> dict:
    """
    Run the full Monte Carlo retirement simulation.
    
    If random_seed is provided, np.random.seed() is called at the start for
    reproducibility. Note: this affects global numpy state and is not
    thread-safe. For advanced use, pass a seeded Generator into custom models.
    """
    if random_seed is not None:
        np.random.seed(int(random_seed))
    # Default strategy
    if strategy is None:
        strategy = ConservativeStrategy()

    # If unspecified, use sampled cash returns when available; otherwise cash
    # falls back to inflation matching (0% real return).
    use_sampled_cash_returns = cash_interest_rate is None

    # Pre-calculate Market Scenarios (Matrix of shape: years x n_paths)
    # This separates market generation from portfolio logic
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
    has_bond_sleeve = bond_allocation_pct > 0.0

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
    
    for t in range(1, years + 1):
        # 1. Market Movement (Nominal)
        # Retrieve pre-calculated return for this year
        market_return_nominal = stock_returns_matrix[t-1, :]
        bond_return_nominal = bond_returns_matrix[t-1, :]
        inflation_nominal = inflation_matrix[t-1, :]
        inflation_rates[t-1, :] = inflation_nominal
        if use_sampled_cash_returns and cash_returns_matrix is not None:
            cash_return_nominal = cash_returns_matrix[t-1, :]
        elif use_sampled_cash_returns:
            cash_return_nominal = inflation_nominal
        else:
            cash_return_nominal = cash_interest_rate
        cash_returns[t-1, :] = cash_return_nominal
            
        # Store NOMINAL market return for analysis/display if needed, 
        # but use REAL return for portfolio growth
        market_returns[t-1, :] = market_return_nominal
        
        # Update Market Index and Peak (High Water Mark)
        market_index *= (1 + market_return_nominal)
        market_peak = np.maximum(market_peak, market_index)
        
        # Convert to REAL Return: (1 + r_nom) / (1 + i) - 1
        real_market_return = (1 + market_return_nominal) / (1 + inflation_nominal) - 1
        real_bond_return = (1 + bond_return_nominal) / (1 + inflation_nominal) - 1
        
        # Real Cash Return
        real_cash_return = (1.0 + cash_return_nominal) / (1.0 + inflation_nominal) - 1.0

        # 2. Update Asset Values (Real Terms)
        current_equity = np.maximum(0.0, current_equity * (1 + real_market_return))
        current_bonds = np.maximum(0.0, current_bonds * (1 + real_bond_return))
        current_cash = np.maximum(0.0, current_cash * (1 + real_cash_return))
        
        # 3. Strategy Execution
        #
        # NOTE (tech debt): When bond_allocation_pct > 0 we take a separate inline
        # path for withdrawal sourcing + bond rebalancing + replenishment.
        # The non-bond path uses the Strategy protocol (pre/withdraw/post hooks).
        # This duplication exists for historical reasons. Preferred future:
        # extend StrategyContext with bond fields and implement bond-aware
        # strategies (or composition) so there is a single source of truth.
        # See strategies.py and the credit-line research for the direction.
        
        # Common Signals
        panic_mask = (market_return_nominal < panic_threshold) | (market_index < (market_peak * 0.999))
        panic_flags[t-1, :] = panic_mask
        
        # Annual Spend (Real)
        target_spend_real = annual_spend
        total_liquid_assets = current_equity + current_bonds + current_cash
        spending_cap_amount = total_liquid_assets * spending_cap_pct
        capped_target_spend = np.minimum(target_spend_real, spending_cap_amount)
        minimum_spend_real = min(max(minimum_annual_spend, 0.0), target_spend_real)
        desired_withdrawal = np.minimum(
            np.maximum(capped_target_spend, minimum_spend_real),
            total_liquid_assets,
        )
        target_cash_level = target_spend_real * buffer_years

        if has_bond_sleeve:
            remaining_withdrawal = desired_withdrawal.copy()
            from_cash = np.zeros(n_paths)
            from_bonds = np.zeros(n_paths)
            from_equity = np.zeros(n_paths)

            panic_with_cash = panic_mask & (current_cash > 0)
            if np.any(panic_with_cash):
                from_cash[panic_with_cash] = np.minimum(
                    remaining_withdrawal[panic_with_cash],
                    current_cash[panic_with_cash],
                )
                remaining_withdrawal -= from_cash

            panic_with_bonds = panic_mask & (remaining_withdrawal > 0)
            if np.any(panic_with_bonds):
                from_bonds[panic_with_bonds] = np.minimum(
                    remaining_withdrawal[panic_with_bonds],
                    current_bonds[panic_with_bonds],
                )
                remaining_withdrawal -= from_bonds

            use_equity = remaining_withdrawal > 0
            if np.any(use_equity):
                from_equity[use_equity] = np.minimum(
                    remaining_withdrawal[use_equity],
                    current_equity[use_equity],
                )
                remaining_withdrawal -= from_equity

            use_bonds_normal = (~panic_mask) & (remaining_withdrawal > 0)
            if np.any(use_bonds_normal):
                bond_withdrawal = np.minimum(
                    remaining_withdrawal[use_bonds_normal],
                    current_bonds[use_bonds_normal],
                )
                from_bonds[use_bonds_normal] += bond_withdrawal
                remaining_withdrawal[use_bonds_normal] -= bond_withdrawal

            use_cash_normal = (~panic_mask) & (remaining_withdrawal > 0)
            if np.any(use_cash_normal):
                from_cash[use_cash_normal] += np.minimum(
                    remaining_withdrawal[use_cash_normal],
                    current_cash[use_cash_normal],
                )

            current_cash -= from_cash
            current_bonds -= from_bonds
            current_equity -= from_equity

            at_peak = market_index >= (market_peak * 0.999)
            replenish_mask = at_peak & (current_cash < target_cash_level)
            if np.any(replenish_mask):
                cash_shortfall = np.zeros(n_paths)
                cash_shortfall[replenish_mask] = target_cash_level - current_cash[replenish_mask]

                from_equity_replenish = np.minimum(cash_shortfall, current_equity)
                current_cash += from_equity_replenish
                current_equity -= from_equity_replenish
                cash_shortfall -= from_equity_replenish

                from_bonds_replenish = np.minimum(cash_shortfall, current_bonds)
                current_cash += from_bonds_replenish
                current_bonds -= from_bonds_replenish

                replenishments[t-1, :] = from_equity_replenish + from_bonds_replenish

            invested_assets = current_equity + current_bonds
            target_bonds = invested_assets * bond_allocation_pct
            bond_shortfall = target_bonds - current_bonds

            buy_bonds_mask = bond_shortfall > 0
            if np.any(buy_bonds_mask):
                transfer = np.minimum(
                    bond_shortfall[buy_bonds_mask],
                    current_equity[buy_bonds_mask],
                )
                current_bonds[buy_bonds_mask] += transfer
                current_equity[buy_bonds_mask] -= transfer

            sell_bonds_mask = bond_shortfall < 0
            if np.any(sell_bonds_mask):
                transfer = np.minimum(
                    -bond_shortfall[sell_bonds_mask],
                    current_bonds[sell_bonds_mask],
                )
                current_bonds[sell_bonds_mask] -= transfer
                current_equity[sell_bonds_mask] += transfer

            withdrawals_from_cash[t-1, :] = from_cash
            withdrawals_from_bonds[t-1, :] = from_bonds
            withdrawals_from_equity[t-1, :] = from_equity
            withdrawals[t-1, :] = from_cash + from_bonds + from_equity

            portfolio_values[t, :] = current_equity + current_bonds + current_cash
            cash_values[t, :] = current_cash
            equity_values[t, :] = current_equity
            bond_values[t, :] = current_bonds
            continue

        # Create Context
        ctx = StrategyContext(
            current_cash=current_cash,
            current_equity=current_equity,
            panic_mask=panic_mask,
            desired_withdrawal=desired_withdrawal, # Not used in pre-rebalance but useful
            market_index=market_index,
            market_peak=market_peak,
            target_cash_level=target_cash_level
        )

        # A. Pre-Withdrawal Rebalance (e.g. Buy Dip)
        # Returns: Positive = Equity->Cash, Negative = Cash->Equity
        pre_transfer = strategy.pre_withdrawal_rebalance(ctx)
        
        # Apply Pre-Transfer
        # Ensure we don't transfer more than available
        # If negative (Cash->Equity), capped by available cash
        # If positive (Equity->Cash), capped by available equity
        
        # Logic to safely apply transfer:
        # 1. Separate into Cash->Equity (negative) and Equity->Cash (positive)
        to_equity_mask = pre_transfer < 0
        to_cash_mask = pre_transfer > 0
        
        realized_transfer = np.zeros_like(pre_transfer)
        
        if np.any(to_equity_mask):
            # Want to move X from Cash to Equity. Max is current_cash.
            # pre_transfer is negative, so use abs or negate
            amount = -pre_transfer[to_equity_mask]
            available = current_cash[to_equity_mask]
            actual = np.minimum(amount, available)
            realized_transfer[to_equity_mask] = -actual # Keep sign
            
        if np.any(to_cash_mask):
            amount = pre_transfer[to_cash_mask]
            available = current_equity[to_cash_mask]
            actual = np.minimum(amount, available)
            realized_transfer[to_cash_mask] = actual

        # Apply realized transfer
        current_cash += realized_transfer
        current_equity -= realized_transfer
        
        # Update context with new balances for withdrawal phase
        # (Important if we just moved all cash to equity!)
        # Note: 'ctx' holds references to arrays, but we just modified the arrays in place?
        # Numpy arrays are mutable. current_cash += ... modifies in place.
        # So ctx.current_cash IS updated.
        
        # B. Withdrawals
        from_cash, from_equity = strategy.determine_withdrawal_source(ctx)

        # --- EXECUTE WITHDRAWALS ---
        current_cash -= from_cash
        current_equity -= from_equity
        
        withdrawals_from_cash[t-1, :] = from_cash
        withdrawals_from_bonds[t-1, :] = 0.0
        withdrawals_from_equity[t-1, :] = from_equity
        withdrawals[t-1, :] = from_cash + from_equity
        
        # C. Post-Withdrawal Rebalance (e.g. Replenish)
        post_transfer = strategy.post_withdrawal_rebalance(ctx)
        
        # Safely apply (Assume mostly Equity->Cash replenishment here)
        to_equity_mask = post_transfer < 0
        to_cash_mask = post_transfer > 0
        realized_post_transfer = np.zeros_like(post_transfer)
        
        if np.any(to_cash_mask):
            amount = post_transfer[to_cash_mask]
            available = current_equity[to_cash_mask]
            actual = np.minimum(amount, available)
            realized_post_transfer[to_cash_mask] = actual
            
        if np.any(to_equity_mask):
             # Rare case for post-rebalance but support it
            amount = -post_transfer[to_equity_mask]
            available = current_cash[to_equity_mask]
            actual = np.minimum(amount, available)
            realized_post_transfer[to_equity_mask] = -actual

        current_cash += realized_post_transfer
        current_equity -= realized_post_transfer
        
        # Record net flow for "replenishments" metric (Equity -> Cash is positive)
        # We combine both transfers for the metric? Or just the replenishment one?
        # Original code tracked only replenishment. Let's track post_transfer.
        replenishments[t-1, :] = realized_post_transfer
        
        # Store
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


def calculate_statistics(
    portfolio_values: np.ndarray,
    withdrawal_values: np.ndarray,
    confidence: float
) -> dict:
    """
    Calculate percentile statistics for visualization.
    
    Args:
        portfolio_values: Array of portfolio values [years+1, n_paths]
        withdrawal_values: Array of withdrawals [years, n_paths]
        confidence: Confidence level (e.g., 0.90 for 90%)
        
    Returns:
        Dictionary with statistical summaries
    """
    alpha = (1 - confidence) / 2
    
    return {
        'portfolio': {
            'lower': np.percentile(portfolio_values, alpha * 100, axis=1),
            'upper': np.percentile(portfolio_values, (1 - alpha) * 100, axis=1),
            'median': np.median(portfolio_values, axis=1),
        },
        'withdrawal': {
            'lower': np.percentile(withdrawal_values, alpha * 100, axis=1),
            'median': np.median(withdrawal_values, axis=1),
        }
    }
