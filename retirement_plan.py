import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime

# ==========================================
# 1. CENTRALIZED CONFIGURATION
# ==========================================
SIM_CONFIG = {
    'INITIAL_NET_WORTH': 5_000_000, # Total Assets (Equity + Cash)
    'ANNUAL_SPEND':      250_000,   # Target annual spending (Year 0 dollars)
    'BUFFER_YEARS':      2,         # Size of cash buffer in years of spending
    'YEARS':             30,        # Duration of retirement
    'PANIC_THRESHOLD':   -0.15,     # If market drops > 15%, switch to cash
    'INFLATION_RATE':    0.03,      # Assumed inflation
    'N_PATHS':           5000,      # Number of Monte Carlo simulations
    'CONFIDENCE':        0.95,      # 90% Confidence Interval (5th to 95th percentile)
    
    # NEW: Configure how far back to look for historical data
    'HISTORY_YEARS':     60         # e.g., 60 years = 1965 to Present
}

# ==========================================
# 2. DATA & MODELING ENGINE
# ==========================================

def get_sp500_residuals(history_years):
    """
    Fetches S&P 500 history based on the configured number of years
    and calculates residuals (historical errors).
    """
    ticker = "^GSPC"
    end_date = datetime.date.today()
    # Calculate start date based on config
    start_date = end_date - datetime.timedelta(days=history_years*365)
    
    print(f"Fetching S&P 500 data from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    # Resample to Annual Returns ('YE' = Year End)
    # We use 'YE' to handle future pandas deprecation warnings
    if len(data) == 0:
        raise ValueError("No data fetched. Try reducing HISTORY_YEARS or checking internet connection.")
        
    annual_returns = data['Close'].resample('YE').last().pct_change().dropna().values.ravel()
    
    # Calculate Mean and Residuals
    mu = np.mean(annual_returns)
    residuals = annual_returns - mu
    
    print(f"Data calibrated. Mean Return: {mu:.2%}. Years of samples: {len(annual_returns)}")
    return mu, residuals

def run_simulation(config, mu, residuals):
    """
    Runs the simulation separating Equity and Cash Buffer logic.
    """
    years = config['YEARS']
    n_paths = config['N_PATHS']
    
    # --- A. Initial Setup ---
    # Calculate initial allocation
    initial_cash_target = config['ANNUAL_SPEND'] * config['BUFFER_YEARS']
    
    # Ensure we don't start with more cash than we have money
    initial_cash = min(initial_cash_target, config['INITIAL_NET_WORTH'])
    initial_equity = config['INITIAL_NET_WORTH'] - initial_cash
    
    # Arrays to track state over time [Years, Paths]
    # We track Equity and Cash separately to model the "Bucket" strategy accurately
    equity_values = np.zeros((years + 1, n_paths))
    cash_values   = np.zeros((years + 1, n_paths))
    
    equity_values[0, :] = initial_equity
    cash_values[0, :]   = initial_cash
    
    # Output Arrays
    total_portfolio_values = np.zeros((years + 1, n_paths))
    total_portfolio_values[0, :] = config['INITIAL_NET_WORTH']
    
    # Withdrawals (Real = Adjusted for inflation to show purchasing power)
    withdrawals_real = np.zeros((years, n_paths))
    
    # --- B. Simulation Loop ---
    for path in range(n_paths):
        current_equity = initial_equity
        current_cash = initial_cash
        inflation_index = 1.0
        
        for t in range(1, years + 1):
            # 1. Market Movement (Conformal Prediction)
            sampled_residual = np.random.choice(residuals)
            market_return = mu + sampled_residual
            
            # Update Equity 
            current_equity = current_equity * (1 + market_return)
            
            # 2. Determine Needs
            inflation_index *= (1 + config['INFLATION_RATE'])
            target_spend_nominal = config['ANNUAL_SPEND'] * inflation_index
            
            # 3. Apply Withdrawal Logic
            actual_withdrawal_nominal = 0
            
            # Rule: Cap withdrawal at 4% of TOTAL remaining assets (Self-preservation)
            # Use current_equity + current_cash to check solvency
            total_assets = current_equity + current_cash
            safety_cap = 0.04 * total_assets
            
            # We want to spend Target, but limited by Safety Cap
            desired_withdrawal = min(target_spend_nominal, safety_cap)
            
            # 4. Sourcing the Funds (The Strategy)
            if market_return < config['PANIC_THRESHOLD'] and current_cash > 0:
                # BAD MARKET: Use Cash Buffer first
                from_cash = min(desired_withdrawal, current_cash)
                current_cash -= from_cash
                
                # If cash wasn't enough, take remainder from equity
                remainder = desired_withdrawal - from_cash
                if current_equity > 0:
                    from_equity = min(remainder, current_equity)
                    current_equity -= from_equity
                    actual_withdrawal_nominal = from_cash + from_equity
                else:
                    actual_withdrawal_nominal = from_cash
            else:
                # NORMAL MARKET: Use Equity first
                if current_equity > 0:
                    from_equity = min(desired_withdrawal, current_equity)
                    current_equity -= from_equity
                    
                    # If equity ran out (rare), use cash
                    remainder = desired_withdrawal - from_equity
                    from_cash = min(remainder, current_cash)
                    current_cash -= from_cash
                    actual_withdrawal_nominal = from_equity + from_cash
                else:
                    # Equity is 0, burn cash
                    from_cash = min(desired_withdrawal, current_cash)
                    current_cash -= from_cash
                    actual_withdrawal_nominal = from_cash

            # 5. Store Results
            total_portfolio_values[t, path] = current_equity + current_cash
            withdrawals_real[t-1, path] = actual_withdrawal_nominal / inflation_index
            
    return total_portfolio_values, withdrawals_real

# ==========================================
# 3. PLOTTING ENGINE
# ==========================================

def plot_results(portfolio_vals, withdrawal_vals, config):
    years_range = range(config['YEARS'] + 1)
    years_withdraw = range(1, config['YEARS'] + 1)
    alpha = (1 - config['CONFIDENCE']) / 2 # e.g., 0.05 for 90% conf
    
    # Calculate Stats
    p_lower = np.percentile(portfolio_vals, alpha*100, axis=1)
    p_upper = np.percentile(portfolio_vals, (1-alpha)*100, axis=1)
    p_median = np.median(portfolio_vals, axis=1)
    
    w_lower = np.percentile(withdrawal_vals, alpha*100, axis=1)
    w_median = np.median(withdrawal_vals, axis=1)

    # --- Plot 1: Portfolio Value ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Portfolio Fan
    ax1.fill_between(years_range, p_lower, p_upper, color='blue', alpha=0.2, label=f'{int(config["CONFIDENCE"]*100)}% Confidence Interval')
    ax1.plot(years_range, p_median, color='blue', linewidth=2, label='Median Portfolio Value')
    ax1.plot(years_range, p_lower, color='red', linestyle='--', linewidth=1.5, label='5th Percentile (Risk Boundary)')
    
    # Zero Line Visualization
    ax1.axhline(0, color='black', linewidth=1)
    
    # Check for Ruin
    ruin_indices = np.where(p_lower <= 0)[0]
    if len(ruin_indices) > 0:
        ruin_year = ruin_indices[0]
        ax1.axvline(ruin_year, color='red', alpha=0.5, linestyle=':')
        ax1.text(ruin_year + 0.5, config['INITIAL_NET_WORTH']*0.1, 
                 f'Risk Boundary\nHits $0 at Year {ruin_year}', color='red', fontweight='bold')

    ax1.set_title(f"Projected Portfolio Value (Start: ${config['INITIAL_NET_WORTH']:,} | History: Last {config['HISTORY_YEARS']} Years)", fontsize=14)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1e6:,.1f}M'))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # --- Plot 2: Withdrawal Ability ---
    # Target Line
    ax2.axhline(config['ANNUAL_SPEND'], color='green', linestyle='-', linewidth=1, label='Target Annual Spend (Real $)')
    
    # Withdrawal Fan
    ax2.plot(years_withdraw, w_median, color='blue', linewidth=2, label='Median Real Withdrawal')
    ax2.plot(years_withdraw, w_lower, color='red', linestyle='--', linewidth=1.5, label='5th Percentile Withdrawal')
    
    # Highlight failure to sustain lifestyle
    ax2.fill_between(years_withdraw, 0, w_lower, where=(w_lower < config['ANNUAL_SPEND']), 
                     color='red', alpha=0.1, label='Lifestyle Gap (Risk Scenario)')

    ax2.set_title("Annual Withdrawal Capacity (Inflation Adjusted)", fontsize=14)
    ax2.set_ylabel("Annual Spend ($ Real)")
    ax2.set_xlabel("Years into Retirement")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x/1000:,.0f}k'))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. EXECUTION
# ==========================================

# A. Get Data
mu, residuals = get_sp500_residuals(SIM_CONFIG['HISTORY_YEARS'])

# B. Run Simulation
print(f"Running {SIM_CONFIG['N_PATHS']} simulations...")
port_vals, withdraw_vals = run_simulation(SIM_CONFIG, mu, residuals)

# C. Plot
plot_results(port_vals, withdraw_vals, SIM_CONFIG)