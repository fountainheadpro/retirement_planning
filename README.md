# Retirement Planning Simulator

This project models a retirement portfolio with Monte Carlo returns and
safety-buffer withdrawal rules. It is centered around `simulator.py`,
which applies a spending cap and a cash buffer to show how disciplined
spending interacts with market drawdowns.

## What the safety buffers do
- **Spending cap**: Annual withdrawals are limited to 4% of current
  liquid assets, so spending scales down automatically when markets
  fall. This protects solvency even before the cash reserve is touched.
- **Cash buffer**: The model prefers to fund withdrawals from cash
  whenever the market is in a drawdown or hits the panic threshold, and
  it only refills cash after prices return to their prior high-water
  mark.

## Why it matters
The two buffers work together to illustrate sequence-risk management:

- The spending cap enforces restraint during selloffs, making the
  portfolio less likely to deplete from oversized withdrawals.
- Cash-first drawdown years avoid selling equity at depressed prices,
  slowing the damage of a bad sequence even when the panic threshold is
  not breached.
- Delayed replenishment shows the trade-off: the buffer can stay low for
  years, revealing that conservative spending does most of the risk
  reduction while cash mainly smooths withdrawals during recoveries.

Use the simulator via `uv run streamlit run app.py` to visualize how the
portfolio evolves under different market assumptions and buffer sizes.
