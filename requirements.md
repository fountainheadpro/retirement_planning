Product Requirements Document

Conformal Retirement Portfolio Simulator

1. Overview

The goal is to build a robust interactive Streamlit web application for retirement planning that avoids the pitfalls of simplistic parametric assumptions while offering advanced modeling capabilities. The tool provides two distinct simulation engines:

Non-Parametric: Conformal Prediction via Residual Sampling (Fat Tail aware).

Parametric: Autoregressive (AR) Mean Reversion (Cycle/Momentum aware).

The application specifically models a "Smart Hedged" Withdrawal Strategy, where the retiree maintains a cash buffer to avoid selling equities during market downturns, mitigating Sequence of Returns Risk.

2. Core Methodology

The simulator must support two user-selectable modes:

Mode A: Historical Residuals (Conformal)

Concept: Assumes the market is random but follows the shape of historical errors (Fat Tails).

Logic: $Return_{t} = \mu + \text{RandomSample}(Residuals)$

Calibration: Calculates residuals ($Return_i - Mean$) from a configurable historical window (e.g., last 60 years).

Mode B: Autoregressive (AR) Mean Reversion

Concept: Assumes the market has "memory." High returns are often followed by mean reversion, and crashes often exhibit momentum or drag.

Logic: AR(p) process. $Return_{t} = \text{Intercept} + \sum_{i=1}^{p} (\phi_i \cdot Return_{t-i}) + \epsilon$

Calibration: Uses statsmodels to fit an ARIMA(p,0,0) model to historical S&P 500 annual returns.

State Tracking: The simulation must track a rolling history window of $p$ years for every simulation path to correctly model the autocorrelation structure.

3. Asset Allocation & Strategy Logic

3.1. Initial State

Total Net Worth: Split into two buckets at $T=0$:

Cash Buffer: Funded immediately to cover $N$ years of annual spending.

Equity Portfolio: The remainder is invested in the market (S&P 500).

3.2. "Smart Hedged" Withdrawal Rules

For every year $t$:

Calculate Needs: Determine inflation-adjusted target spending.

Solvency Check: Calculate Total Liquid Assets (Equity + Cash).

Desired Withdrawal: min(Target Spend, Total Liquid Assets). (Prioritizes lifestyle maintenance unless completely broke).

Sourcing Funds:

Panic Market ($Return < \text{Threshold}$): Withdraw from Cash Buffer first. If empty, sell Equity.

Normal Market ($Return \ge \text{Threshold}$): Withdraw from Equity first. If empty, use Cash.

4. Configuration Parameters (UI Inputs)

The app must provide a sidebar allowing users to adjust parameters dynamically:

Parameter

UI Element

Description

Financials





INITIAL_NET_WORTH

Number Input

Total starting assets ($).

ANNUAL_SPEND

Number Input

Target annual spending in Year 0 dollars.

Strategy





BUFFER_YEARS

Slider

Size of cash bucket in years of expenses (e.g., 2).

PANIC_THRESHOLD

Slider

Market drop % that triggers cash usage (e.g., -15%).

Simulation Mode

Radio Select

"Historical Residuals" vs "Autoregressive (AR)".

AR_ORDER

Slider

(AR Mode Only) Number of lags ($p$) to use (1-5).

HISTORY_YEARS

Slider

Look-back window for calibration data (e.g., 60).

Settings





YEARS

Slider

Duration of simulation (e.g., 30).

INFLATION_RATE

Slider

Annual inflation assumption.

N_PATHS

Select

Number of Monte Carlo iterations (e.g., 2500).

CONFIDENCE

Slider

Statistical guarantee level (e.g., 90%).

5. Data Requirements

Source: Yahoo Finance API (yfinance).

Ticker: ^GSPC (S&P 500 Index).

Frequency: Annual Returns (resampled from Daily/Monthly).

Processing:

Fetch historical data based on HISTORY_YEARS.

For Residuals Mode: Calculate simple mean and residuals.

For AR Mode: Fit ARIMA model to extract coefficients ($\phi$), intercept, and volatility ($\sigma$).

Caching: Critical for performance; cache data fetching and model fitting.

6. Visualization & Reporting

The tool must generate two synchronized interactive Plotly charts:

Plot A: Portfolio Value (The "Survival" Chart)

Fan Chart: 5th–95th percentile shaded area.

Lines: Median Portfolio Value and 5th Percentile (Risk Boundary).

Tooltips: Show exact dollar values on hover.

Ruin Marker: If the Risk Boundary hits $0, visually indicate the year of ruin.

Plot B: Withdrawal Capacity (The "Lifestyle" Chart)

Target Line: Horizontal line showing Desired Real Spending.

Actual Spending: Plot Median and Risk (5th %ile) capacity.

Gap Analysis: Visually demonstrate when the portfolio cannot support the target lifestyle (Real Withdrawal < Target).

7. Software Stack

Language: Python 3.x

UI Framework: streamlit

Data Analysis: numpy (vectorized simulation), pandas, statsmodels (ARIMA calibration).

Data Feed: yfinance

Visualization: plotly.graph_objects