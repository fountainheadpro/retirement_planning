import streamlit as st
import plotly.graph_objects as go
import numpy as np

from simulator import get_sp500_residuals, run_simulation, calculate_statistics, create_ar_model, RandomWalkMarket, BlockBootstrapMarket, MeanRevertingMarket
from strategies import ConservativeStrategy, AggressiveStrategy, NoCashBufferStrategy

def ordinal(n):
    """Return number with ordinal suffix (1st, 2nd, 3rd, 4th)."""
    n = int(n)
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(4 if 10 <= n % 100 < 20 else n % 10, "th")
    return f"{n}{suffix}"

st.set_page_config(
    page_title="Retirement Portfolio Simulator",
    page_icon="📈",
    layout="wide"
)

st.markdown(
    """
    <style>
        .title-row { display:flex; align-items:center; gap:8px; }
        .tooltip { position:relative; display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; background:#f0f2f6; color:#0f1116; font-weight:600; cursor:help; }
        .tooltip .tooltiptext { visibility:hidden; opacity:0; position:absolute; left:28px; top:50%; transform:translateY(-50%); background:#0f1116; color:white; padding:10px 12px; border-radius:8px; width:320px; box-shadow:0 8px 20px rgba(0,0,0,0.15); font-size:0.9rem; line-height:1.35; z-index:10; transition:opacity 0.15s ease; }
        .tooltip .tooltiptext::after { content:""; position:absolute; left:-6px; top:50%; transform:translateY(-50%); border-width:6px; border-style:solid; border-color:transparent #0f1116 transparent transparent; }
        .tooltip:hover .tooltiptext { visibility:visible; opacity:1; }
    </style>
    <div class="title-row">
        <h1 style="margin:0;">📈 Retirement Portfolio Simulator</h1>
        <div class="tooltip" aria-label="How this simulator works">
            ℹ️
            <div class="tooltiptext">
                Models a retirement portfolio in real (inflation-adjusted) dollars using S&amp;P 500 total-return history. Configure net worth, annual spending, cash buffer and spending cap, panic threshold, inflation and cash rate, then pick Random Walk, Mean Reversion (AR), or Block Bootstrap to run Monte Carlo paths and visualize risk bands.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("All monetary values are displayed in Today's Dollars (Real Purchasing Power).")
st.markdown("""
This tool uses statistical models to simulate future market behavior,
ensuring fat-tail events (2000, 2008) are represented in risk projections.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

with st.sidebar.form("config_form"):
    submitted = st.form_submit_button("🚀 Run Simulation", type="primary")
    
    with st.expander("Model Configuration", expanded=True):
        model_options = ["Random Walk", "Mean Reversion (AR-1)", "Mean Reversion (AR-2)", "Mean Reversion (AR-3)", "Mean Reversion (AR-4)", "Mean Reversion (AR-5)", "Block Bootstrap"]
        selected_model = st.selectbox(
            "Market Model",
            options=model_options,
            index=model_options.index("Mean Reversion (AR-3)"), # Default to AR-3
            help="""
Select the statistical model for simulating market returns:

1. **Random Walk (Optimistic):** Assumes future returns are independent and follow the historical average distribution. Often ignores valuation risks (e.g., high P/E ratios) and assumes the "good times" will roll on average.

2. **Mean Reversion (AR-n) (Pessimistic/Conservative):** Assumes that periods of high returns are followed by lower returns (and vice versa) to return to a long-term mean. This is generally more conservative when starting from high market valuations, as it predicts a "cooling off" period.

3. **Block Bootstrap (Balanced/Realistic):** Resamples actual historical blocks of data (e.g., 5-year chunks). This preserves real-world market shocks (volatility clustering) and "fat tails" (crashes like 2000 or 2008) exactly as they happened, offering a realistic "what if history repeats" scenario.
"""
        )
        
        block_size = 5 # Default value
        if selected_model == "Block Bootstrap":
            block_size = st.slider("Block Size (Years)", 1, 10, 5, help="Length of historical blocks to resample. Preserves historical correlations.")
            
        history_years = st.slider(
            "Historical Data (Years)",
            min_value=20,
            max_value=100,
            value=50,
            help="Look-back window for calibration. Affects all models."
        )

    with st.expander("Portfolio Settings", expanded=True):
        initial_net_worth = st.number_input(
            "Initial Net Worth ($)",
            min_value=100_000,
            max_value=50_000_000,
            value=2_000_000,
            step=100_000,
            format="%d"
        )

        annual_spend = st.number_input(
            "Annual Spending ($)",
            min_value=10_000,
            max_value=1_000_000,
            value=80_000,
            step=10_000,
            format="%d"
        )

        # Strategy Selector
        strategy_display = st.selectbox(
            "Cash Strategy",
            options=[
                "Conservative (Protect Withdrawals)", 
                "Aggressive (Buy the Dip)", 
                "Fully Invested (No Cash Buffer)"
            ],
            index=0,
            help="""
**Conservative:** Uses cash buffer to fund withdrawals during market downturns (Panic/Drawdown) to avoid selling equity at a loss. Replenishes cash only when market recovers (High Water Mark).

**Aggressive (Buy the Dip):** Uses cash buffer to BUY equity during market downturns. Withdrawals come from Equity. Replenishes cash when market recovers.

**Fully Invested:** Holds 0% cash. All funds in equity. Withdrawals always sold from equity.
"""
        )
        
        # Map display name to internal name
        strategy_map = {
            "Conservative (Protect Withdrawals)": "Conservative",
            "Aggressive (Buy the Dip)": "Aggressive",
            "Fully Invested (No Cash Buffer)": "No Cash Buffer"
        }
        selected_strategy = strategy_map[strategy_display]

        # Conditional Inputs
        if selected_strategy != "No Cash Buffer":
            buffer_years = st.slider(
                "Cash Buffer (Years)",
                min_value=0,
                max_value=5,
                value=2,
                help="Years of expenses to keep in cash buffer"
            )
            
            cash_interest_rate = st.slider(
                "Cash Interest Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Nominal interest rate earned on cash buffer. Defaults to matching inflation if not set."
            ) / 100
        else:
            buffer_years = 0
            cash_interest_rate = 0.0

        spending_cap_pct = st.slider(
            "Spending Cap (% of Portfolio)",
            min_value=1.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help="Maximum annual withdrawal as percentage of total portfolio value"
        ) / 100

    with st.expander("Simulation Settings", expanded=True):
        years = st.slider(
            "Simulation Duration (Years)",
            min_value=10,
            max_value=50,
            value=30
        )

        panic_threshold = st.slider(
            "Panic Threshold (%)",
            min_value=-50,
            max_value=0,
            value=-15,
            help="Market drop that triggers cash usage"
        ) / 100

        inflation_rate = st.slider(
            "Inflation Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.5
        ) / 100

        n_paths = st.select_slider(
            "Monte Carlo Paths",
            options=[500, 1000, 2000, 5000, 10000],
            value=5000
        )

        confidence = st.slider(
            "Confidence Level (%)",
            min_value=80,
            max_value=99,
            value=90
        ) / 100


# Cache data fetching
@st.cache_data(ttl=3600)
def fetch_market_data(history_years: int):
    """Fetch and cache S&P 500 data."""
    return get_sp500_residuals(history_years)


if submitted or 'results' not in st.session_state:
    with st.spinner("Fetching market data..."):
        try:
            # Fetch data once for all models
            mu, residuals, history = fetch_market_data(history_years)
            
            if history is None: # Check if market data fetching failed
                st.error("Failed to fetch market data. Please check your internet connection or try again.")
                st.stop()

            # Instantiate Market Model based on selection
            market_model = None
            model_info_msg = "" # For sidebar info display

            if selected_model == "Random Walk":
                market_model = RandomWalkMarket(mu, residuals)
                model_info_msg = f"Random Walk (Mean: {mu:.1%}, Std Dev of Residuals: {np.std(residuals):.1%})"
                
            elif selected_model == "Block Bootstrap":
                market_model = BlockBootstrapMarket(history, block_size=block_size)
                model_info_msg = f"Block Bootstrap (Block Size: {block_size}y, History: {len(history)}y)"
                
            elif "Mean Reversion (AR-" in selected_model:
                ar_p = int(selected_model.split("AR-")[1][:-1]) # Extract AR order from string
                ar_model_calibrated, stats = create_ar_model(history_years, ar_order=ar_p)
                if ar_model_calibrated:
                    market_model = ar_model_calibrated
                    coeffs_str = ", ".join([f"{c:.2f}" for c in stats['ar_coeffs']])
                    model_info_msg = (f"Calibrated AR({ar_p}) (Coeffs: [{coeffs_str}], "
                                        f"Vol: {stats['volatility']:.1%}, "
                                        f"Mean: {stats['mean_return']:.1%})")
                else:
                    st.error("AR model calibration failed. Falling back to Random Walk.")
                    market_model = RandomWalkMarket(mu, residuals)
                    model_info_msg = "Random Walk (AR model calibration failed)"
                    
            if market_model is None: # Fallback if model selection logic somehow fails
                st.error("Failed to initialize market model. Defaulting to Random Walk.")
                market_model = RandomWalkMarket(mu, residuals)
                model_info_msg = "Random Walk (Default fallback)"

            # Instantiate Strategy Object
            strategy_obj = None
            if selected_strategy == "Conservative":
                strategy_obj = ConservativeStrategy()
            elif selected_strategy == "Aggressive":
                strategy_obj = AggressiveStrategy()
            elif selected_strategy == "No Cash Buffer":
                strategy_obj = NoCashBufferStrategy()
            else:
                strategy_obj = ConservativeStrategy()

        except Exception as e:
            st.error(f"Error initializing market model: {e}")
            st.stop()

    # Display calibration info in sidebar
    st.sidebar.success(f"Model Ready: {selected_model}")
    if model_info_msg:
        st.sidebar.info(model_info_msg)

    with st.spinner(f"Running {n_paths:,} simulations..."):
        sim_results = run_simulation(
            initial_net_worth=initial_net_worth,
            annual_spend=annual_spend,
            buffer_years=buffer_years,
            years=years,
            panic_threshold=panic_threshold,
            inflation_rate=inflation_rate,
            n_paths=n_paths,
            market_model=market_model, # Pass the instantiated model object
            spending_cap_pct=spending_cap_pct,
            cash_interest_rate=cash_interest_rate,
            strategy=strategy_obj
        )
        # Use REAL (Inflation-Adjusted) values for all visualizations
        portfolio_vals = sim_results['portfolio_values']
        withdrawal_vals = sim_results['withdrawal_values']
        cash_vals = sim_results['cash_values']
        equity_vals = sim_results['equity_values']
    
    stats = calculate_statistics(portfolio_vals, withdrawal_vals, confidence)
    st.session_state['results'] = {
        'portfolio_vals': portfolio_vals,
        'withdrawal_vals': withdrawal_vals,
        'cash_vals': cash_vals,
        'equity_vals': equity_vals,
        'stats': stats,
        'params': {
            'years': years,
            'initial_net_worth': initial_net_worth,
            'annual_spend': annual_spend,
            'confidence': confidence,
            'history_years': history_years,
            'cash_interest_rate': cash_interest_rate,
            'panic_threshold': panic_threshold, 
            'buffer_years': buffer_years,
            'inflation_rate': inflation_rate,
            'strategy': selected_strategy # Store string name for display logic
        }
    }

# Display Results
if 'results' in st.session_state:
    results = st.session_state['results']
    stats = results['stats']
    params = results['params']
    years_range = list(range(params['years'] + 1))
    years_withdraw = list(range(1, params['years'] + 1))
    
    # Summary Statistics (Moved to Top)
    st.subheader("📋 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    final_median = stats['portfolio']['median'][-1]
    final_lower = stats['portfolio']['lower'][-1]
    
    with col1:
        st.metric(
            "Final Portfolio (Median)",
            f"${final_median:,.0f}",
            delta=f"{(final_median/params['initial_net_worth']-1)*100:.1f}%",
            help="Expected portfolio value in the median scenario."
        )
    
    with col2:
        st.metric(
            "Final Portfolio (Risk)",
            f"${max(0, final_lower):,.0f}",
            delta=f"{(final_lower/params['initial_net_worth']-1)*100:.1f}%" if final_lower > 0 else "Depleted",
            help=f"Portfolio value in the {ordinal((1-params['confidence'])/2*100)} percentile (bad outcome) scenario."
        )
    
    with col3:
        ruin_prob = np.mean(results['portfolio_vals'][-1, :] <= 0) * 100
        st.metric(
            "Ruin Probability", 
            f"{ruin_prob:.1f}%",
            help="Probability of running out of money before the end of the simulation."
        )
    
    with col4:
        withdrawal_shortfall = np.mean(results['withdrawal_vals'] < params['annual_spend']) * 100
        st.metric(
            "Withdrawal Shortfall Risk", 
            f"{withdrawal_shortfall:.1f}%",
            help="Probability of having to reduce spending below your target."
        )
    
    # Portfolio Value Chart
    st.subheader("📊 Portfolio Value Projection (Real Dollars)")
    
    fig1 = go.Figure()
    
    # Confidence interval band
    fig1.add_trace(go.Scatter(
        x=years_range + years_range[::-1],
        y=list(stats['portfolio']['upper']) + list(stats['portfolio']['lower'][::-1]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f"{int(params['confidence']*100)}% Confidence Interval",
        hoverinfo='skip'
    ))
    
    # Median line
    fig1.add_trace(go.Scatter(
        x=years_range,
        y=stats['portfolio']['median'],
        mode='lines',
        name='Median Portfolio',
        line=dict(color='blue', width=2),
        hovertemplate='Year %{x}<br>$%{y:,.0f}<extra>Median</extra>'
    ))
    
    # Risk boundary (lower percentile)
    risk_percentile_val = int((1-params["confidence"])/2*100)
    fig1.add_trace(go.Scatter(
        x=years_range,
        y=stats['portfolio']['lower'],
        mode='lines',
        name=f'{ordinal(risk_percentile_val)} Percentile (Risk)',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='Year %{x}<br>$%{y:,.0f}<extra>Risk Boundary</extra>'
    ))
    
    # Zero line
    fig1.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    # Check for ruin
    ruin_indices = np.where(stats['portfolio']['lower'] <= 0)[0]
    if len(ruin_indices) > 0:
        ruin_year = ruin_indices[0]
        fig1.add_vline(x=ruin_year, line_dash="dot", line_color="red", opacity=0.5)
        fig1.add_annotation(
            x=ruin_year, y=params['initial_net_worth'] * 0.5,
            text=f"⚠️ Risk boundary hits $0<br>at Year {ruin_year}",
            showarrow=True, arrowhead=2, arrowcolor="red",
            font=dict(color="red", size=12)
        )
    
    fig1.update_layout(
        title=f"Projected Portfolio Value (Start: ${params['initial_net_worth']:,} | History: {params['history_years']} Years) - Real Dollars",
        xaxis_title="Years into Retirement",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450
    )
    
    st.plotly_chart(fig1, width="stretch")
    
    # Withdrawal Capacity Chart
    st.subheader("💰 Withdrawal Capacity (Inflation Adjusted)")
    
    fig2 = go.Figure()
    
    # Target spending line
    fig2.add_hline(
        y=params['annual_spend'],
        line_dash="solid",
        line_color="green",
        annotation_text=f"Target: ${params['annual_spend']:,}",
        annotation_position="top right"
    )
    
    # Lifestyle gap shading (where lower < target): fill area between target and lower percentile
    lower_vals = np.array(stats['withdrawal']['lower'])
    target_line = np.where(lower_vals < params['annual_spend'], params['annual_spend'], None)
    gap_floor = np.where(lower_vals < params['annual_spend'], lower_vals, None)

    if np.any(lower_vals < params['annual_spend']):
        fig2.add_trace(go.Scatter(
            x=years_withdraw,
            y=target_line,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig2.add_trace(go.Scatter(
            x=years_withdraw,
            y=gap_floor,
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.15)',
            line=dict(color='rgba(255,0,0,0.4)', dash='dot'),
            name='Lifestyle Gap Risk',
            hovertemplate='Year %{x}<br>Gap: $%{customdata:,.0f}<extra></extra>',
            customdata=np.maximum(0, params['annual_spend'] - lower_vals)
        ))
    
    # Median withdrawal
    fig2.add_trace(go.Scatter(
        x=years_withdraw,
        y=stats['withdrawal']['median'],
        mode='lines',
        name='Median Withdrawal',
        line=dict(color='blue', width=2),
        hovertemplate='Year %{x}<br>$%{y:,.0f}<extra>Median</extra>'
    ))
    
    # Lower percentile withdrawal
    risk_percentile_val = int((1-params["confidence"])/2*100)
    fig2.add_trace(go.Scatter(
        x=years_withdraw,
        y=stats['withdrawal']['lower'],
        mode='lines',
        name=f'{ordinal(risk_percentile_val)} Percentile',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='Year %{x}<br>$%{y:,.0f}<extra>Risk Scenario</extra>'
    ))
    
    fig2.update_layout(
        title="Annual Withdrawal Capacity (Real Dollars)",
        xaxis_title="Years into Retirement",
        yaxis_title="Annual Withdrawal ($)",
        yaxis_tickformat="$,.0f",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450
    )
    
    st.plotly_chart(fig2, width="stretch")

    # Asset Allocation Chart
    # Define alpha for the title
    alpha = (1 - params['confidence']) / 2
    risk_percentile_val = int(alpha * 100)
    
    # Calculate real cash return for display in tooltip
    # Use variables directly from widgets to avoid KeyError on first run/stale state
    real_cash_return_for_display = (1 + cash_interest_rate) / (1 + inflation_rate) - 1

    # Dynamic Description based on Strategy
    strat_desc = ""
    active_strat = params.get('strategy', 'Conservative') # Default to conservative if missing
    
    if active_strat == "Conservative":
        strat_desc = f"""
**Strategy: Conservative**
-   **Target:** A cash buffer is maintained at {buffer_years} years of annual spending.
-   **Withdrawals:** During normal market conditions, withdrawals come primarily from equity. If the market experiences a significant drop (return below {panic_threshold:.0%}) or if the overall market is in a drawdown, withdrawals will prioritize using cash from the buffer.
-   **Replenishment:** The cash buffer is only replenished from equity when the overall market has recovered to its previous peak (High Water Mark).
"""
    elif active_strat == "Aggressive":
        strat_desc = f"""
**Strategy: Aggressive (Buy the Dip)**
-   **Target:** A cash buffer is maintained at {buffer_years} years of annual spending.
-   **Investment:** When the market experiences a significant drop (Panic/Drawdown), available cash is **moved into equity** to buy the dip.
-   **Withdrawals:** Withdrawals are taken from Equity (maximizing time in market/exposure).
-   **Replenishment:** The cash buffer is replenished from equity only when the market recovers to its previous peak.
"""
    elif active_strat == "No Cash Buffer":
        strat_desc = """
**Strategy: Fully Invested**
-   **Target:** No cash buffer. 100% Equity allocation.
-   **Withdrawals:** All withdrawals come from selling equity, regardless of market conditions.
"""

    st.subheader("📊 Asset Allocation (Risk Scenario) - Real Dollars", help=f"""
This chart shows the composition of your portfolio for a specific "risk scenario" path, selected as the closest trajectory to the {ordinal(risk_percentile_val)} percentile outcome from the Portfolio Projection.

{strat_desc}

**Cash Growth:** Cash in the buffer grows at the nominal interest rate of {cash_interest_rate:.1%} (equivalent to {real_cash_return_for_display:.1%} real return given {inflation_rate:.1%} inflation).
""")
    
    # Improved Risk Path Selection: Nearest Neighbor to the Risk Boundary Curve
    # 1. Get the calculated risk boundary (lower percentile curve)
    risk_boundary_curve = stats['portfolio']['lower']
    
    # 2. Calculate Euclidean distance (sum of squared differences) for every path
    # portfolio_vals is shape (years+1, n_paths), risk_boundary is shape (years+1,)
    # We broadcast subtraction across paths
    differences = results['portfolio_vals'] - risk_boundary_curve[:, np.newaxis]
    distances = np.sum(differences**2, axis=0)
    
    # 3. Find the index of the path with the minimum distance
    risk_path_idx = np.argmin(distances)
    
    # Extract cash and equity for this representative risk path
    risk_cash = results['cash_vals'][:, risk_path_idx]
    risk_equity = results['equity_vals'][:, risk_path_idx]
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=years_range,
        y=risk_cash,
        mode='lines',
        line=dict(width=0.5, color='rgb(131, 90, 241)'),
        stackgroup='one',
        name='Cash Buffer'
    ))
    
    fig3.add_trace(go.Scatter(
        x=years_range,
        y=risk_equity,
        mode='lines',
        line=dict(width=0.5, color='rgb(0, 200, 100)'),
        stackgroup='one',
        name='Equity (S&P 500)'
    ))
    
    fig3.update_layout(
        title=f"Portfolio Composition ({ordinal(risk_percentile_val)} Percentile Outcome) - Real Dollars",
        xaxis_title="Years into Retirement",
        yaxis_title="Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450
    )
    
    st.plotly_chart(fig3, width="stretch")
