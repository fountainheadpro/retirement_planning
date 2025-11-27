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

class MeanRevertingMarket:
    """
    Models the market using an AR(p) process on Returns.
    R_t = Intercept + (Phi_1 * R_{t-1}) + ... + (Phi_p * R_{t-p}) + Noise
    
    This captures multi-year market memory (cycles), allowing the model to learn 
    patterns like "Deep crashes are often followed by multi-year recoveries."
    """
    def __init__(self, ar_order=1):
        self.ar_order = ar_order
        self.ar_coeffs = None # Array of shape (p,)
        self.intercept = None
        self.residual_std = None
        # Tracks the last 'p' years of returns to seed the simulation
        self.history_window = np.zeros(ar_order) 

    def calibrate_from_history(self, historical_returns):
        """
        Calibrates AR(p) parameters using statsmodels ARIMA.
        Uses ARIMA(p, 0, 0) which is equivalent to an AR(p) model on stationary data.
        """
        # FIX: Ensure data is 1D flat array. yfinance sometimes returns (N, 1) which breaks matrix math.
        data = np.array(historical_returns).ravel()
        p = self.ar_order
        
        # We need enough data to fit the model confidently
        if len(data) < p + 10:
            raise ValueError(f"Not enough data for AR({p}). Need at least {p+10} years.")

        try:
            # Fit ARIMA(p, 0, 0) with a constant term (trend='c')
            # This models: y_t = const + ar_1*y_{t-1} + ... + error_t
            model = ARIMA(data, order=(p, 0, 0), trend='c')
            res = model.fit()
            
            # Extract Parameters
            # params contains [const, ar.L1, ar.L2, ... sigma2]
            self.intercept = res.params[0]
            self.ar_coeffs = res.arparams # This helper gives just the AR coefficients
            self.residual_std = np.sqrt(res.params[-1]) # Last param is sigma2
            
            # Set State (The most recent 'p' years from history)
            # We need this to start the simulation "from today"
            # History needs to be [t-1, t-2, ... t-p]
            # data[-p:] gives [t-p ... t-1], so we reverse it
            self.history_window = data[-p:][::-1] 
            
            # Calculate Long Term Mean (Analytical)
            # Mean = Intercept / (1 - sum(coeffs))
            denom = (1 - np.sum(self.ar_coeffs))
            long_term_mean = self.intercept / denom if abs(denom) > 1e-5 else 0.0
            
            return {
                "ar_coeffs": self.ar_coeffs,
                "intercept": self.intercept,
                "mean_return": long_term_mean,
                "volatility": self.residual_std
            }
            
        except Exception as e:
            # Fallback if convergence fails (rare on simple AR)
            st.warning(f"ARIMA fitting failed: {e}. Falling back to synthetic.")
            return None

    def simulate_year(self, history_window, simulations=1):
        """
        Simulates exactly one year forward based on the history window.
        Args:
            history_window: Array of shape (simulations, p) containing [R_{t-1}, ..., R_{t-p}]
        Returns: 
            next_return: Array of shape (simulations,)
        """
        if self.ar_coeffs is None:
            raise ValueError("Model not calibrated.")

        # AR(p) Equation: Next = Intercept + Sum(Coeff_i * Lag_i) + Noise
        # history_window shape: (n_paths, p)
        # ar_coeffs shape: (p,)
        
        # Dot product sums the lagged effects (Result is 1D array of length n_paths)
        deterministic_part = self.intercept + np.dot(history_window, self.ar_coeffs)
        
        # FIX: Noise should be 1D (one random shock per simulation path)
        # Old code (wrong): np.random.normal(..., (simulations, self.ar_order))
        noise = np.random.normal(0, self.residual_std, simulations)
        
        next_returns = deterministic_part + noise
        
        return next_returns

@st.cache_data
def get_sp500_data(history_years=60):
    """Fetches S&P 500 Annual Returns."""
    ticker = "^GSPC"
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=history_years * 365)
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if len(data) < 250: return None
        annual_returns = data['Close'].resample('YE').last().pct_change().dropna()
        return annual_returns.values
    except Exception:
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
        
    return model, stats_msg

@st.cache_data
def get_sp500_residuals(history_years):
    hist = get_sp500_data(history_years)
    if hist is None: return None, None, "No Data"
    mu = np.mean(hist)
    residuals = hist - mu
    return mu, residuals, None

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================

def run_simulation(
    initial_net_worth, annual_spend, buffer_years, years, 
    panic_threshold, inflation_rate, n_paths,
    mu, residuals, 
    use_ar_model=True, ar_model=None, 
    spending_cap_pct=0.04
):
    # Initial Allocation
    initial_cash_target = annual_spend * buffer_years
    initial_cash = min(initial_cash_target, initial_net_worth)
    initial_equity = initial_net_worth - initial_cash
    
    # State Arrays
    portfolio_values = np.zeros((years + 1, n_paths))
    portfolio_values[0, :] = initial_net_worth
    withdrawals_real = np.zeros((years, n_paths))
    
    # Reset Arrays
    current_equity = np.full(n_paths, float(initial_equity))
    current_cash = np.full(n_paths, float(initial_cash))
    inflation_index = 1.0
    
    # Initialize Histories for AR Model
    if use_ar_model and ar_model:
        # FIX: Tile correctly for (n_paths, p). 
        # Explicit reshape guarantees we tile (1, p) into (n_paths, p)
        # This handles deeper AR orders safely and prevents (N*p, 1) shaping if input was vertical
        current_history_windows = np.tile(ar_model.history_window.reshape(1, -1), (n_paths, 1))

    for t in range(1, years + 1):
        # 1. Market Movement
        if use_ar_model and ar_model:
            market_return = ar_model.simulate_year(current_history_windows, simulations=n_paths)
            
            # Update History (Slide window)
            # Shift right: [New, t-1, t-2 ...]
            current_history_windows = np.roll(current_history_windows, shift=1, axis=1)
            
            # FIX: market_return is now 1D (n_paths,), so no slicing needed
            current_history_windows[:, 0] = market_return
        else:
            market_return = mu + np.random.choice(residuals, n_paths)

        # 2. Update Equity Value
        # FIX: Use np.maximum for vectorized max operation. standard max() collapses the array.
        current_equity = np.maximum(0.0, current_equity * (1 + market_return))
        
        # 3. Withdrawals
        inflation_index *= (1 + inflation_rate)
        target_spend_nominal = annual_spend * inflation_index
        
        # FIX: Use np.maximum/np.minimum for vectorized operations to preserve path data
        # Note: current_equity is already floored at 0.0 above, so total_liquid_assets is safe
        total_liquid_assets = current_equity + current_cash 
        
        # Spending Cap (Solvency based)
        desired_withdrawal = np.minimum(target_spend_nominal, total_liquid_assets * spending_cap_pct)
        
        # Smart Hedged Logic (Panic vs Normal) - Vectorized
        panic_mask = market_return < panic_threshold
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
        
        # Store
        portfolio_values[t, :] = current_equity + current_cash
        withdrawals_real[t-1, :] = (from_cash + from_equity) / inflation_index
            
    return portfolio_values, withdrawals_real

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
