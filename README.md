# Retirement Planning Simulator

A high-quality, reproducible Monte Carlo retirement simulator focused on **honest sequence-of-returns risk management**.

The core insight (validated through block-bootstrap Monte Carlo experiments and the published credit-line research): a **flexible spending rule** (target / cap / floor as % of *current* portfolio) already provides the majority of downside protection. Cash buffers and permanent bond allocations have surprisingly small (or even negative) impact on the "maintain target lifestyle" objective once the spending cap is in place.

## Published Research

The project ships with one focused, self-critical research report:

- **[Can A Hybrid Credit Line Replace Cash And Bonds?](docs/credit-line.html)** — The flagship report. Tests whether a securities-backed credit line that respects the spending cap/floor can meaningfully replace cash buffers or bonds. Monthly S&P total-return block simulation, explicit debt and margin dynamics, and pragmatic conclusions. The generator lives at `notes/generate_credit_line_report.py`.

This report (and the interactive simulator that implements the same spending rule + cash strategy mechanics) is the primary deliverable. The tool lets you explore the exact same models interactively.

## Quick Start

```bash
# 1. Install uv (https://docs.astral.sh/uv/) — recommended
# 2. Clone and sync the locked environment
uv sync

# 3. Run the interactive simulator
uv run streamlit run app.py
```

The UI defaults to the "Stock/Bond Block Bootstrap" model + fully-invested strategy (consistent with the research finding that large cash/bond sleeves add limited protection under a proper spending cap).

## Key Features

- **Multiple market models**: Random Walk (residual sampling), AR(p) mean reversion, Block Bootstrap (single-asset and paired stock/bond/inflation/cash), all with proper historical dependence preservation.
- **Pluggable cash strategies** via the `CashStrategy` ABC (`Conservative`, `Aggressive (buy-the-dip)`, `NoCashBuffer`).
- **Flexible spending rule**: Target derived from initial portfolio × cap %, automatic scaling via current-portfolio cap, hard floor, and insolvency protection.
- **Reproducibility**: Optional fixed random seed in the UI so you can share or publish exact scenario results.
- **Pure core**: `simulator.py` has zero Streamlit dependency and can be imported from scripts, other research code, or notebooks.
- Strong test coverage (>50 tests) including strategy behavior, paired bootstrap invariance, AR alignment, and reproducibility.

## Reproducibility & Publishing

When you enable "Fixed random seed" in the sidebar and run a scenario, the exact same parameters + seed will produce identical paths. This makes it practical to publish specific "scary path" examples or exact statistics from the tool.

The credit-line report in `docs/credit-line.html` was generated with fixed seeds and documented data sources (Yahoo Finance ^SP500TR total return + FRED CPI). The annual simulator uses the local CSV files for its block-bootstrap and AR paths.

## Project Structure

- `simulator.py` — Market models (`RandomWalkMarket`, `BlockBootstrapMarket`, `PairedBlockBootstrapMarket`, `MeanRevertingMarket`) + `run_simulation` engine + strategy context.
- `strategies.py` — The `CashStrategy` protocol and the three built-in implementations.
- `app.py` — Streamlit UI (reactive, well-documented controls, fan charts, risk metrics).
- `test_simulator.py` + `test_credit_line_report.py` — Executable specification.
- `notes/generate_credit_line_report.py` — The generator for the published credit-line HTML report.
- `docs/` — GitHub Pages assets (run the generator to refresh).

See [AGENTS.md](AGENTS.md) for engineering guidelines and [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose changes.

## License

MIT — see [LICENSE](LICENSE).

## Citation / Attribution

If you use the reports or simulator in your own writing or decisions, a link back to the repository + credit to the specific report is appreciated. The methodology (flexible target/cap/floor + block bootstrap + explicit cash/buffer/replenishment rules) is the intellectual core.

---

**Why this exists**: Most retirement tools either (a) hide behind a single "safe withdrawal rate" or (b) use overly optimistic parametric assumptions. This project tries to be explicit about the trade-offs and let the data (historical blocks) speak.
