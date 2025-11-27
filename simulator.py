"""
Core simulation engine for the Conformal Retirement Portfolio Simulator.
Uses historical residuals for non-parametric Monte Carlo simulation.
"""
import datetime
import numpy as np
import pandas as pd
import yfinance as yf


def get_sp500_residuals(history_years: int) -> tuple[float, np.ndarray, pd.Series]:
    """
    Fetches S&P 500 history and calculates residuals (historical errors).
    
    Args:
        history_years: Number of years to look back for historical data
        
    Returns:
        Tuple of (mean_return, residuals_array, annual_returns_series)
    """
    ticker = "^GSPC"
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=history_years * 365)
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if len(data) == 0:
        raise ValueError("No data fetched. Check internet connection or reduce HISTORY_YEARS.")
    
    # Resample to Annual Returns
    annual_returns = data['Close'].resample('YE').last().pct_change().dropna()
    returns_array = annual_returns.values.ravel()
    
    # Calculate Mean and Residuals
    mu = np.mean(returns_array)
    residuals = returns_array - mu
    
    return mu, residuals, annual_returns


def run_simulation(
    initial_net_worth: float,
    annual_spend: float,
    buffer_years: int,
    years: int,
    panic_threshold: float,
    inflation_rate: float,
    n_paths: int,
    mu: float,
    residuals: np.ndarray,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs Monte Carlo simulation with Smart Hedged withdrawal strategy.
    
    Args:
        initial_net_worth: Total starting assets
        annual_spend: Target annual spending in Year 0 dollars
        buffer_years: Size of cash buffer in years of spending
        years: Duration of simulation
        panic_threshold: Market drop % that triggers cash usage
        inflation_rate: Annual inflation assumption
        n_paths: Number of Monte Carlo iterations
        mu: Mean historical return
        residuals: Historical residuals array
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (portfolio_values, withdrawals_real) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initial allocation
    initial_cash_target = annual_spend * buffer_years
    initial_cash = min(initial_cash_target, initial_net_worth)
    initial_equity = initial_net_worth - initial_cash
    
    # Output arrays
    total_portfolio_values = np.zeros((years + 1, n_paths))
    total_portfolio_values[0, :] = initial_net_worth
    withdrawals_real = np.zeros((years, n_paths))
    
    for path in range(n_paths):
        current_equity = initial_equity
        current_cash = initial_cash
        inflation_index = 1.0
        
        for t in range(1, years + 1):
            # Market movement (Conformal Prediction)
            sampled_residual = np.random.choice(residuals)
            market_return = mu + sampled_residual
            
            # Update equity with market return
            current_equity = current_equity * (1 + market_return)
            
            # Determine needs
            inflation_index *= (1 + inflation_rate)
            target_spend_nominal = annual_spend * inflation_index
            
            # Safety cap: 4% of total remaining assets
            total_assets = max(0, current_equity) + current_cash
            safety_cap = 0.04 * total_assets
            desired_withdrawal = min(target_spend_nominal, safety_cap)
            
            # Source funds based on market conditions
            actual_withdrawal_nominal = _source_funds(
                desired_withdrawal, 
                market_return, 
                panic_threshold,
                current_equity, 
                current_cash
            )
            
            # Update balances
            if market_return < panic_threshold and current_cash > 0:
                from_cash = min(desired_withdrawal, current_cash)
                current_cash -= from_cash
                remainder = desired_withdrawal - from_cash
                if current_equity > 0:
                    from_equity = min(remainder, current_equity)
                    current_equity -= from_equity
            else:
                if current_equity > 0:
                    from_equity = min(desired_withdrawal, current_equity)
                    current_equity -= from_equity
                    remainder = desired_withdrawal - from_equity
                    from_cash = min(remainder, current_cash)
                    current_cash -= from_cash
                else:
                    from_cash = min(desired_withdrawal, current_cash)
                    current_cash -= from_cash
            
            # Store results
            total_portfolio_values[t, path] = max(0, current_equity) + current_cash
            withdrawals_real[t - 1, path] = actual_withdrawal_nominal / inflation_index
    
    return total_portfolio_values, withdrawals_real


def _source_funds(
    desired: float, 
    market_return: float, 
    panic_threshold: float,
    equity: float, 
    cash: float
) -> float:
    """Calculate actual withdrawal amount based on strategy."""
    if market_return < panic_threshold and cash > 0:
        # Bad market: use cash first
        from_cash = min(desired, cash)
        remainder = desired - from_cash
        from_equity = min(remainder, max(0, equity)) if equity > 0 else 0
        return from_cash + from_equity
    else:
        # Normal market: use equity first
        if equity > 0:
            from_equity = min(desired, equity)
            remainder = desired - from_equity
            from_cash = min(remainder, cash)
            return from_equity + from_cash
        else:
            return min(desired, cash)


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
