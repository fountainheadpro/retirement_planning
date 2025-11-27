"""
Conformal Retirement Portfolio Simulator - Streamlit App
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from simulator import get_sp500_residuals, run_simulation, calculate_statistics

st.set_page_config(
    page_title="Retirement Portfolio Simulator",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Conformal Retirement Portfolio Simulator")
st.markdown("""
This tool uses **Conformal Prediction via Residual Sampling** to model future market volatility,
ensuring fat-tail events (1929, 2000, 2008) are represented in risk projections.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Portfolio Settings")
initial_net_worth = st.sidebar.number_input(
    "Initial Net Worth ($)",
    min_value=100_000,
    max_value=50_000_000,
    value=5_000_000,
    step=100_000,
    format="%d"
)

annual_spend = st.sidebar.number_input(
    "Annual Spending ($)",
    min_value=10_000,
    max_value=1_000_000,
    value=200_000,
    step=10_000,
    format="%d"
)

buffer_years = st.sidebar.slider(
    "Cash Buffer (Years)",
    min_value=0,
    max_value=5,
    value=2,
    help="Years of expenses to keep in cash buffer"
)

st.sidebar.subheader("Simulation Settings")
years = st.sidebar.slider(
    "Simulation Duration (Years)",
    min_value=10,
    max_value=50,
    value=30
)

panic_threshold = st.sidebar.slider(
    "Panic Threshold (%)",
    min_value=-50,
    max_value=0,
    value=-15,
    help="Market drop that triggers cash usage"
) / 100

inflation_rate = st.sidebar.slider(
    "Inflation Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.5
) / 100

n_paths = st.sidebar.select_slider(
    "Monte Carlo Paths",
    options=[500, 1000, 2000, 5000, 10000],
    value=5000
)

confidence = st.sidebar.slider(
    "Confidence Level (%)",
    min_value=80,
    max_value=99,
    value=90
) / 100

history_years = st.sidebar.slider(
    "Historical Data (Years)",
    min_value=20,
    max_value=100,
    value=60,
    help="Look-back window for calibration"
)

# Run Simulation Button
run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)

# Cache data fetching
@st.cache_data(ttl=3600)
def fetch_market_data(history_years: int):
    """Fetch and cache S&P 500 data."""
    return get_sp500_residuals(history_years)


if run_button or 'results' not in st.session_state:
    with st.spinner("Fetching market data..."):
        try:
            mu, residuals, annual_returns = fetch_market_data(history_years)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()
    
    # Display calibration info
    st.sidebar.success(f"Mean Return: {mu:.2%}")
    st.sidebar.info(f"Samples: {len(residuals)} years")
    
    with st.spinner(f"Running {n_paths:,} simulations..."):
        portfolio_vals, withdrawal_vals = run_simulation(
            initial_net_worth=initial_net_worth,
            annual_spend=annual_spend,
            buffer_years=buffer_years,
            years=years,
            panic_threshold=panic_threshold,
            inflation_rate=inflation_rate,
            n_paths=n_paths,
            mu=mu,
            residuals=residuals
        )
    
    stats = calculate_statistics(portfolio_vals, withdrawal_vals, confidence)
    st.session_state['results'] = {
        'portfolio_vals': portfolio_vals,
        'withdrawal_vals': withdrawal_vals,
        'stats': stats,
        'params': {
            'years': years,
            'initial_net_worth': initial_net_worth,
            'annual_spend': annual_spend,
            'confidence': confidence,
            'history_years': history_years
        }
    }

# Display Results
if 'results' in st.session_state:
    results = st.session_state['results']
    stats = results['stats']
    params = results['params']
    years_range = list(range(params['years'] + 1))
    years_withdraw = list(range(1, params['years'] + 1))
    
    # Portfolio Value Chart
    st.subheader("📊 Portfolio Value Projection")
    
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
    fig1.add_trace(go.Scatter(
        x=years_range,
        y=stats['portfolio']['lower'],
        mode='lines',
        name=f'{int((1-params["confidence"])/2*100)}th Percentile (Risk)',
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
        title=f"Projected Portfolio Value (Start: ${params['initial_net_worth']:,} | History: {params['history_years']} Years)",
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
    
    # Lifestyle gap shading (where lower < target)
    lower_vals = stats['withdrawal']['lower']
    gap_x = []
    gap_y = []
    for i, (yr, val) in enumerate(zip(years_withdraw, lower_vals)):
        if val < params['annual_spend']:
            gap_x.extend([yr, yr])
            gap_y.extend([0, val])
    
    if gap_x:
        fig2.add_trace(go.Scatter(
            x=years_withdraw,
            y=[min(v, params['annual_spend']) for v in lower_vals],
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Lifestyle Gap Risk',
            hoverinfo='skip'
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
    fig2.add_trace(go.Scatter(
        x=years_withdraw,
        y=stats['withdrawal']['lower'],
        mode='lines',
        name=f'{int((1-params["confidence"])/2*100)}th Percentile',
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
    
    # Summary Statistics
    st.subheader("📋 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    final_median = stats['portfolio']['median'][-1]
    final_lower = stats['portfolio']['lower'][-1]
    
    with col1:
        st.metric(
            "Final Portfolio (Median)",
            f"${final_median:,.0f}",
            delta=f"{(final_median/params['initial_net_worth']-1)*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "Final Portfolio (Risk)",
            f"${max(0, final_lower):,.0f}",
            delta=f"{(final_lower/params['initial_net_worth']-1)*100:.1f}%" if final_lower > 0 else "Depleted"
        )
    
    with col3:
        ruin_prob = np.mean(results['portfolio_vals'][-1, :] <= 0) * 100
        st.metric("Ruin Probability", f"{ruin_prob:.1f}%")
    
    with col4:
        withdrawal_shortfall = np.mean(results['withdrawal_vals'] < params['annual_spend']) * 100
        st.metric("Withdrawal Shortfall Risk", f"{withdrawal_shortfall:.1f}%")
