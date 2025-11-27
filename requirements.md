Product Requirements Document

Conformal Retirement Portfolio Simulator

1. Overview

The goal is to build a robust interactive Streamlit web application for retirement planning that avoids the pitfalls of parametric (Normal Distribution) assumptions. Instead of assuming a theoretical distribution, the tool uses Conformal Prediction via Residual Sampling—drawing directly from historical market errors (residuals) to model future volatility. This ensures that "Fat Tail" events (e.g., 1929, 2000, 2008) are natively represented in the risk model.

The application specifically models a "Smart Hedged" Withdrawal Strategy, where the retiree maintains a cash buffer to avoid selling equities during market downturns, mitigating Sequence of Returns Risk.

2. Core Methodology

Simulation Method: Non-Parametric Monte Carlo.

Prediction Engine: Conformal Prediction using Historical Residuals.

Predictor: Long-term historical mean of S&P 500 returns.

Calibration Set: Historical annual return residuals ($Actual - Mean$) calculated from a configurable look-back window (e.g., last 60 years).

Simulation: Future Year Return = $Mean + Randomly Sampled Residual$.

Inflation: Constant rate applied annually to purchasing power and withdrawal targets.

3. Asset Allocation & Strategy Logic

3.1. Initial State

Total Net Worth: Split into two buckets at $T=0$:

Cash Buffer: Funded immediately to cover $N$ years of annual spending (e.g., 2 years).

Equity Portfolio: The remainder of the net worth is invested in the market (S&P 500).

3.2. "Smart Hedged" Withdrawal Rules

For every year $t$ in the simulation:

Calculate Needs: Determine inflation-adjusted target spending.

Safety Cap: Calculate maximum allowable withdrawal as 4% of Total Remaining Assets (Equity + Cash).

Actual Withdrawal = min(Target Spend, Safety Cap).

Sourcing Funds (The Decision Logic):

Scenario A: Panic Market (Market Return < PANIC_THRESHOLD):

Withdraw funds from Cash Buffer first.

If Cash Buffer is empty, withdraw remainder from Equity.

Scenario B: Normal Market (Market Return ≥ PANIC_THRESHOLD):

Withdraw funds from Equity first.

If Equity is empty, withdraw remainder from Cash Buffer.

4. Configuration Parameters (UI Inputs)

The app must provide a sidebar or input form allowing users to adjust parameters dynamically:

Parameter

UI Element

Description

Example

INITIAL_NET_WORTH

Number Input

Total starting assets ($)

$5,000,000

ANNUAL_SPEND

Number Input

Target annual spending in Year 0 dollars

$200,000

BUFFER_YEARS

Slider/Input

Size of cash bucket in years of expenses

2

YEARS

Slider

Duration of simulation

30

PANIC_THRESHOLD

Slider

Market drop % that triggers cash usage

-0.15 (-15%)

INFLATION_RATE

Slider

Annual inflation assumption

0.03 (3%)

N_PATHS

Select/Input

Number of Monte Carlo iterations

5,000

CONFIDENCE

Slider

Statistical guarantee level

0.90 (90%)

HISTORY_YEARS

Slider

Look-back window for calibration data

60

5. Data Requirements

Source: Yahoo Finance API (yfinance).

Ticker: ^GSPC (S&P 500 Index).

Frequency: Annual Returns.

Processing:

Fetch historical close prices.

Resample to annual percentage change.

Calculate residuals ($Return_i - Mean$).

Caching: Data fetching should be cached via Streamlit (@st.cache_data) to prevent redundant API calls.

6. Visualization & Reporting

The tool must generate two synchronized interactive time-series plots.

Plot A: Portfolio Value (The "Survival" Chart)

Fan Chart: Display the confidence interval (e.g., 5th to 95th percentile) as a shaded region.

Median Line: The most probable outcome.

Risk Boundary: A distinct line highlighting the lower bound (e.g., 5th percentile).

Interactivity: Mouseover tooltips showing the exact Portfolio Value ($) for any specific year on the median and risk lines.

Ruin Visualization:

Clearly mark the $0 line.

Annotate the Year of Ruin (if applicable) where the Risk Boundary hits $0.

Plot B: Withdrawal Capacity (The "Lifestyle" Chart)

Target Line: A horizontal line showing the Desired Annual Spend (Real $).

Capacity Curves: Plot the Median and 5th Percentile withdrawal amounts over time.

Lifestyle Gap: Use shading (e.g., red) to highlight years where the 5th Percentile Capacity < Target Spend. This visually represents the risk of being "solvents but poor".

Interactivity: Tooltips showing the Real Withdrawal Amount ($) vs Target for specific years.

7. Software Stack

Language: Python 3.x

Libraries:

streamlit: Web application framework and UI.

yfinance: Data fetching.

numpy: Matrix operations and vectorised simulation.

plotly: Interactive plotting (replacing static matplotlib charts) for tooltips and zooming.

pandas: Time series handling.