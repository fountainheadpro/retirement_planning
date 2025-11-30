import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime
import requests
# Added statsmodels for robust ARIMA fitting
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
    def __init__(self, history_returns, block_size=5):
        self.block_size = block_size
        self.history = np.asarray(history_returns).ravel()
        if len(self.history) < self.block_size:
            raise ValueError(f"History length {len(self.history)} is shorter than block size {self.block_size}")
        
    def simulate_matrix(self, years, n_paths):
        """
        Generates a matrix of market returns using block bootstrapping.
        Returns: array of shape (years, n_paths)
        """
        n_history = len(self.history)
        n_blocks = int(np.ceil(years / self.block_size))
        
        # Output matrix
        market_matrix = np.zeros((years, n_paths))
        
        for i in range(n_paths):
            path = []
            for _ in range(n_blocks):
                # Pick a random start index
                start_idx = np.random.randint(0, n_history - self.block_size + 1)
                block = self.history[start_idx : start_idx + self.block_size]
                path.extend(block)
            
            # Trim to exact number of years and assign
            market_matrix[:, i] = path[:years]
            
        return market_matrix

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
            st.warning(f"ARIMA fitting failed: {e}. Falling back to synthetic.")
            return None

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

@st.cache_data
def get_sp500_data(history_years=60):
    """Fetches S&P 500 Annual Returns from local CSV (Total Return)."""
    try:
        # CSV format: Year, Return% (e.g., 2024, 25.02)
        # Assume file is in the same directory or root
        df = pd.read_csv("s_and_p_500_with_dividends.csv", header=None, names=["Year", "Return"])
        
        # Convert percent to decimal
        df["Return"] = df["Return"] / 100.0
        
        # Sort by Year just in case
        df = df.sort_values("Year")
        
        # Take the last 'history_years'
        if len(df) > history_years:
            df = df.tail(history_years)
            
        return df["Return"].values
    except Exception as e:
        st.error(f"Error loading S&P 500 data: {e}")
        return None

def create_ar_model(history_years=50, ar_order=1):
    """
    Creates and calibrates an AR(p) model using statsmodels.
    """
    model = MeanRevertingMarket(ar_order=ar_order)
    stats_msg = "Error fetching data."
    
    # Use full available history for best calibration of long cycles
    hist_returns = get_sp500_data(history_years=history_years) 
    
    stats = model.calibrate_from_history(hist_returns)
    
    if stats:
        # Format coefficients for display
        coeffs_str = ", ".join([f"{c:.2f}" for c in stats['ar_coeffs']])
        stats_msg = (f"Calibrated AR({ar_order}) on {len(hist_returns)}y history.\n"
                        f"Coeffs: [{coeffs_str}], "
                        f"Vol: {stats['volatility']:.1%}, "
                        f"Mean: {stats['mean_return']:.1%}")
        
    return model, stats

@st.cache_data
def get_sp500_residuals(history_years):
    hist = get_sp500_data(history_years)
    if hist is None: return None, None, None # Return None for hist if no data
    mu = np.mean(hist)
    residuals = hist - mu
    return mu, residuals, hist

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================

def run_simulation(
    initial_net_worth, annual_spend, buffer_years, years, 
    panic_threshold, inflation_rate, n_paths,
    market_model,
    spending_cap_pct=0.04,
    cash_interest_rate=None
):
    # Default cash interest to inflation if not specified (Real return = 0%)
    if cash_interest_rate is None:
        cash_interest_rate = inflation_rate

    # Pre-calculate Market Scenarios (Matrix of shape: years x n_paths)
    # This separates market generation from portfolio logic
    market_returns_matrix = market_model.simulate_matrix(years, n_paths)

    # Initial Allocation
    initial_cash_target = annual_spend * buffer_years
    initial_cash = min(initial_cash_target, initial_net_worth)
    initial_equity = initial_net_worth - initial_cash
    
    # State Arrays (All in Real Dollars)
    portfolio_values = np.zeros((years + 1, n_paths))
    cash_values = np.zeros((years + 1, n_paths))
    equity_values = np.zeros((years + 1, n_paths))
    
    portfolio_values[0, :] = initial_net_worth
    cash_values[0, :] = initial_cash
    equity_values[0, :] = initial_equity
    
    # Detailed tracking
    withdrawals = np.zeros((years, n_paths))
    market_returns = np.zeros((years, n_paths))
    panic_flags = np.zeros((years, n_paths), dtype=bool)
    withdrawals_from_cash = np.zeros((years, n_paths))
    withdrawals_from_equity = np.zeros((years, n_paths))
    replenishments = np.zeros((years, n_paths))
    
    # Reset Arrays
    current_equity = np.full(n_paths, float(initial_equity))
    current_cash = np.full(n_paths, float(initial_cash))
    
    # Track Market High Water Mark
    market_index = np.ones(n_paths)
    market_peak = np.ones(n_paths)
    
    for t in range(1, years + 1):
        # 1. Market Movement (Nominal)
        # Retrieve pre-calculated return for this year
        market_return_nominal = market_returns_matrix[t-1, :]
            
        # Store NOMINAL market return for analysis/display if needed, 
        # but use REAL return for portfolio growth
        market_returns[t-1, :] = market_return_nominal
        
        # Update Market Index and Peak (High Water Mark)
        market_index *= (1 + market_return_nominal)
        market_peak = np.maximum(market_peak, market_index)
        
        # Convert to REAL Return: (1 + r_nom) / (1 + i) - 1
        real_market_return = (1 + market_return_nominal) / (1 + inflation_rate) - 1
        
        # Real Cash Return
        real_cash_return = (1.0 + cash_interest_rate) / (1.0 + inflation_rate) - 1.0

        # 2. Update Asset Values (Real Terms)
        current_equity = np.maximum(0.0, current_equity * (1 + real_market_return))
        current_cash = np.maximum(0.0, current_cash * (1 + real_cash_return))
        
        # 3. Withdrawals (Real Terms)
        # Annual spend is constant in real terms
        target_spend_real = annual_spend
        
        total_liquid_assets = current_equity + current_cash 
        
        # Spending Cap (Solvency based) - 4% of current portfolio value (Real)
        desired_withdrawal = np.minimum(target_spend_real, total_liquid_assets * spending_cap_pct)
        
        # Smart Hedged Logic (Panic vs Normal) - Vectorized
        # Panic is triggered by NOMINAL drops (psychological)
        panic_mask = market_return_nominal < panic_threshold
        panic_flags[t-1, :] = panic_mask
        
        has_cash_mask = current_cash > 0
        
        # Allocation Arrays
        from_cash = np.zeros(n_paths).astype(float)
        from_equity = np.zeros(n_paths).astype(float)
        
        # CASE 1: Panic & Has Cash -> Use Cash First
        mask1 = panic_mask & has_cash_mask
        if np.any(mask1):
            from_cash[mask1] = np.minimum(desired_withdrawal[mask1], current_cash[mask1])
            from_equity[mask1] = desired_withdrawal[mask1] - from_cash[mask1]
            
        # CASE 2: Normal OR No Cash -> Use Equity First
        mask2 = ~mask1
        if np.any(mask2):
            from_equity[mask2] = np.minimum(desired_withdrawal[mask2], current_equity[mask2])
            from_cash[mask2] = desired_withdrawal[mask2] - from_equity[mask2]
            
        # Update Balances
        current_cash -= from_cash
        current_equity -= from_equity
        
        # Store withdrawal details
        withdrawals_from_cash[t-1, :] = from_cash
        withdrawals_from_equity[t-1, :] = from_equity
        withdrawals[t-1, :] = from_cash + from_equity
        
        # 4. Cash Buffer Replenishment
        # Rule: Only replenish if we are at or above the Market High Water Mark (Recovery)
        # This prevents selling equity during a drawdown to fill cash.
        
        target_cash_level = target_spend_real * buffer_years
        
        # Identify paths that:
        # 1. Are at High Water Mark (market_index >= market_peak)
        # 2. Have less cash than target
        # Note: We use a small epsilon tolerance for float comparison
        at_peak_mask = market_index >= (market_peak * 0.999)
        replenish_mask = at_peak_mask & (current_cash < target_cash_level)
        
        if np.any(replenish_mask):
            # Calculate shortfall
            shortfall = target_cash_level - current_cash[replenish_mask]
            
            # Move funds from equity to cash, capped by available equity
            to_transfer = np.minimum(shortfall, current_equity[replenish_mask])
            
            current_cash[replenish_mask] += to_transfer
            current_equity[replenish_mask] -= to_transfer
            replenishments[t-1, replenish_mask] = to_transfer
        
        # Store
        portfolio_values[t, :] = current_equity + current_cash
        cash_values[t, :] = current_cash
        equity_values[t, :] = current_equity
            
    return {
        'portfolio_values': portfolio_values,
        'withdrawal_values': withdrawals,
        'cash_values': cash_values,
        'equity_values': equity_values,
        'market_returns': market_returns,
        'panic_flags': panic_flags,
        'withdrawals_from_cash': withdrawals_from_cash,
        'withdrawals_from_equity': withdrawals_from_equity,
        'replenishments': replenishments
    }

def _source_funds(
    desired: float,
    market_return: float,
    panic_threshold: float,
    equity: float,
    cash: float
) -> float:
    """Calculate actual withdrawal amount based on strategy."""
    allocation = _allocate_withdrawal(
        desired,
        market_return,
        panic_threshold,
        equity,
        cash,
    )
    return allocation[0]


def _allocate_withdrawal(
    desired: float,
    market_return: float,
    panic_threshold: float,
    equity: float,
    cash: float,
) -> tuple[float, float, float]:
    """Allocate withdrawals between cash and equity without overdrawing.

    Returns a tuple of (total_withdrawal, from_cash, from_equity).
    """
    available_equity = max(0.0, equity)
    available_cash = max(0.0, cash)
    available_total = available_equity + available_cash

    actual_withdrawal = min(desired, available_total)

    if market_return < panic_threshold and available_cash > 0:
        from_cash = min(actual_withdrawal, available_cash)
        from_equity = min(actual_withdrawal - from_cash, available_equity)
    else:
        from_equity = min(actual_withdrawal, available_equity)
        from_cash = min(actual_withdrawal - from_equity, available_cash)

    return actual_withdrawal, from_cash, from_equity


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
