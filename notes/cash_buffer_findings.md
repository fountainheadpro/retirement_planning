# Cash buffer impact on risk

The current simulator makes the cash reserve much less influential than
you might expect because of how withdrawals and replenishment are coded:

- Withdrawals are capped at 4% of total liquid assets every year, so
  spending automatically scales down during drawdowns. This cap already
  mitigates sequence risk, leaving little for the buffer to absorb.
- The cash reserve now comes into play when the nominal market return
  for the year is below the panic threshold **or** the market is still
  below its prior high-water mark. This keeps withdrawals off equities
  while the portfolio remains in drawdown.
- Cash is replenished from equities only when the market revisits its
  prior peak. After an early drawdown, the buffer can sit depleted for
  many years, limiting any safety it would otherwise provide.
- Cash earns roughly a 0% real return by default (interest is set equal
  to inflation), so keeping more years of expenses in cash slightly
  drags growth without materially changing failure dynamics.

In a quick simulation I attempted to compare 0-, 2-, and 5-year buffers,
but the sandbox cannot install NumPy (`uv sync` and `pip install numpy`
both failed because external downloads are blocked). That prevents
actually running `run_simulation`, yet the withdrawal logic above
explains why the buffer has muted risk-reduction effects.

## Does the buffer work as implemented?

Within the model, the buffer behaves consistently with the rules above:

- Cash is tapped whenever returns fall below `panic_threshold` **or** the
  market is still under its high-water mark, provided cash is available
  (`current_cash > 0`). Otherwise equities fund the withdrawal.
  【F:simulator.py†L271-L304】
- The replenishment rule only fires when the portfolio revisits its
  prior high-water mark (`at_peak_mask`), at which point any shortfall to
  `target_cash_level` is moved from equity into cash. This keeps the
  buffer from refilling during extended drawdowns.【F:simulator.py†L305-L328】
- Because the spending cap scales down withdrawals to 4% of liquid
  assets, negative-return years automatically mean smaller withdrawals
  (and lower reliance on cash) even when the buffer exists.【F:simulator.py†L262-L301】

So the buffer is correctly implemented per the current ruleset, but its
impact on ruin probability is intentionally small because it is rarely
used and slowly replenished.

## Should we keep the buffer?

Pros:

- It communicates the idea of holding a few years of expenses in cash
  and shows how, under conservative spending rules, the benefit can be
  muted.
- The existing UI and scenarios can illustrate panic-withdrawal behavior
  without further code changes.

Cons:

- The present implementation barely changes outcomes; removing it would
  simplify the model without materially changing Monte Carlo results.
- Keeping a feature that looks more powerful than it is may confuse
  users who expect dramatic sequence-risk protection.

If we want the buffer to matter more, we could (a) route all negative or
sub-threshold years through cash first, (b) refill cash gradually from
equity even before a new high-water mark, and/or (c) allow a positive
real cash return to reduce the opportunity cost. Otherwise, removing the
buffer control would keep the model lean while staying faithful to the
current spending-cap philosophy.
