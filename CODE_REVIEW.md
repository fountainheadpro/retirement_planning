# Code Review: Conformal Retirement Portfolio Simulator

## Scope
Review against `requirements.md` for the Streamlit-based Conformal Retirement Portfolio Simulator.

## Findings

1. **Negative-equity withdrawals can increase the equity balance instead of depleting it**
   - In `run_simulation`, withdrawal sourcing subtracts funds directly from `current_equity` without clamping negative balances. If a severe drawdown makes `current_equity` negative, `min(remainder, current_equity)` can be negative and the subsequent subtraction adds money back into equity (e.g., `current_equity -= -10`), inflating portfolio value and underestimating ruin probability. Withdrawals should be limited to available non-negative equity, consistent with the “Smart Hedged” strategy in the requirements. 【F:simulator.py†L115-L136】

2. **“Lifestyle gap” shading does not visualize shortfall relative to the target spend**
   - The requirements call for shading years where the 5th percentile withdrawal falls below the target spend. The current Plotly trace fills the area between the lower percentile and zero, not between the lower percentile and the target spending line, and the `gap_x/gap_y` detection logic is unused. This under-represents shortfall severity and does not match the specified visualization. 【F:app.py†L220-L269】

## Recommendations
- Clamp equity to zero (or otherwise disallow negative withdrawals) before any deductions, and use the resulting actual withdrawal amount for reporting, so simulations respect insolvency rather than creating equity from negative balances.
- Rework the lifestyle-gap shading to fill the area between the target spend line and the lower-percentile curve wherever the curve is below the target; remove unused helper variables.
