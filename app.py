"""
Retirement Portfolio Simulator - Streamlit App
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from simulator import get_sp500_residuals, run_simulation, calculate_statistics, create_ar_model

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

st.title("📈 Retirement Portfolio Simulator")
st.caption("All monetary values are displayed in Today's Dollars (Real Purchasing Power).")
st.markdown("""
This tool uses **Conformal Prediction via Residual Sampling** or **Mean Reversion via Ornstein-Uhlenbeck** to model future market behavior,
ensuring fat-tail events (2000, 2008) are represented in risk projections.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

with st.sidebar.form("config_form"):
    
    with st.expander("Model Configuration", expanded=True):
        use_mean_reversion = st.checkbox(
            "Use Mean Reversion Model",
            value=True,
            help="Use Ornstein-Uhlenbeck process instead of historical residuals"
        )

        # Mean reversion model parameters
        if use_mean_reversion:
            ar_order = st.selectbox(
                "Autoregressive Order",
                options=[1, 2, 3, 4, 5],
                index=0,
                help="Number of years to use in autoregression (AR order)"
            )
        else:
            ar_order = 1
            
        history_years = st.slider(
            "Historical Data (Years)",
            min_value=20,
            max_value=100,
            value=50,
            help="Look-back window for calibration"
        )

    with st.expander("Portfolio Settings", expanded=True):
        initial_net_worth = st.number_input(
            "Initial Net Worth ($)",
            min_value=100_000,
            max_value=50_000_000,
            value=6_000_000,
            step=100_000,
            format="%d"
        )

        annual_spend = st.number_input(
            "Annual Spending ($)",
            min_value=10_000,
            max_value=1_000_000,
            value=250_000,
            step=10_000,
            format="%d"
        )

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

    # Submit Button within the form
    submitted = st.form_submit_button("🚀 Run Simulation", type="primary")


# Cache data fetching
@st.cache_data(ttl=3600)
def fetch_market_data(history_years: int):
    """Fetch and cache S&P 500 data."""
    return get_sp500_residuals(history_years)


if submitted or 'results' not in st.session_state:
    with st.spinner("Fetching market data..."):
        try:
            if use_mean_reversion:
                mu, residuals = 0.0, np.array([])
                ar_model = None
                stats_msg = ""
                
                mean_reversion_model, stats_msg = create_ar_model(history_years=history_years, ar_order=ar_order)
            else:
                mu, residuals, annual_returns = fetch_market_data(history_years)
                mean_reversion_model = None
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    # Display calibration info
    if use_mean_reversion:
        st.sidebar.success(f"Mean Reversion Model Calibrated")
        st.sidebar.info(f"AR Order: {ar_order}")
    else:
        st.sidebar.success(f"Mean Return: {mu:.2%}")
        st.sidebar.info(f"Samples: {len(residuals)} years")

    with st.spinner(f"Running {n_paths:,} simulations..."):
        sim_results = run_simulation(
            initial_net_worth=initial_net_worth,
            annual_spend=annual_spend,
            buffer_years=buffer_years,
            years=years,
            panic_threshold=panic_threshold,
            inflation_rate=inflation_rate,
            n_paths=n_paths,
            mu=mu,
            residuals=residuals,
            use_ar_model=use_mean_reversion,
            ar_model=mean_reversion_model,
            spending_cap_pct=spending_cap_pct,
            cash_interest_rate=cash_interest_rate
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
            'cash_interest_rate': cash_interest_rate
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
    
    st.plotly_chart(fig1, use_container_width=True)
    
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
    
    st.plotly_chart(fig2, use_container_width=True)

    # Asset Allocation Chart
    # Define alpha for the title
    alpha = (1 - params['confidence']) / 2
    risk_percentile_val = int(alpha * 100)
    
    # Calculate real cash return for display in tooltip
    real_cash_return_for_display = (1 + params['cash_interest_rate']) / (1 + params['inflation_rate']) - 1

    st.subheader("📊 Asset Allocation (Risk Scenario) - Real Dollars", help=f"""
This chart shows the composition of your portfolio for a specific "risk scenario" path, selected as the closest trajectory to the {ordinal(risk_percentile_val)} percentile outcome from the Portfolio Projection.

**Cash Buffer Management:**
-   **Target:** A cash buffer is maintained at {params['buffer_years']} years of annual spending.
-   **Withdrawals:** During normal market conditions, withdrawals come primarily from equity. If the market experiences a significant drop (below {params['panic_threshold']:.0%}) or if the overall market is in a drawdown, withdrawals will prioritize using cash from the buffer.
-   **Replenishment:** The cash buffer is only replenished from equity when the overall market has recovered to its previous peak (High Water Mark). This strategy avoids selling equities at a loss to refill cash during drawdowns.
-   **Cash Growth:** Cash in the buffer grows at the nominal interest rate of {params['cash_interest_rate']:.1%} (equivalent to {real_cash_return_for_display:.1%} real return given {params['inflation_rate']:.1%} inflation). This aims to prevent it from losing purchasing power if `cash_interest_rate` matches `inflation_rate`.
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
    
    st.plotly_chart(fig3, use_container_width=True)