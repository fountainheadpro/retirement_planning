# Retirement Planning Simulator

A high-quality, reproducible Monte Carlo retirement simulator focused on **honest sequence-of-returns risk management**.

The core insight (validated through block-bootstrap Monte Carlo experiments): a **flexible spending rule** (target / cap / floor as % of *current* portfolio) already provides most of the downside protection. Permanent cash and 10-year Treasury sleeves then trade a small cut in already-low ruin risk for more years below starting wealth.

## Published Research

Two sibling reports ship in `docs/` and are published at [actions-im.github.io/retirement_planning](https://actions-im.github.io/retirement_planning/). They share the target/cap/floor rule but use different engines and data windows, so the ruin and shortfall rates are **not** interchangeable.

- **[The Measurable Tradeoffs Behind The 4% Rule](docs/index.html)** — Flagship annual report. Damodaran S&P 500 / 10-year Treasury / T-bill blocks paired with FRED CPI-U, 1951–2025, 20,000 paths, seed `20260513`. Tests cash buffers, history windows, withdrawal rates, block size, and permanent bond sleeves. Generators: `notes/generate_report_results.py` and `notes/generate_report_assets.py`.
- **[Can A Hybrid Credit Line Reduce Forced Selling?](docs/credit-line.html)** — Monthly sibling. Yahoo `^SP500TR` + FRED CPI from 1988 onward, 20,000 paths, seed `20260516`. Tests a cap-respecting securities-backed credit line versus sell-only. It does not simulate cash or 60/40. Generator: `notes/generate_credit_line_report.py`.

The Streamlit app implements the same annual spending rule and cash/bond mechanics as the allocation report. It does not reproduce the monthly credit-line engine.

## Regenerating the reports

```bash
# Allocation paper numbers and charts (slow: 20,000-path Monte Carlo)
uv run python notes/generate_report_results.py
uv run python notes/generate_report_assets.py
uv run python notes/generate_report_html.py

# Credit-line HTML from existing JSON (no Monte Carlo)
uv run python notes/generate_credit_line_report.py --render-only

# Credit-line paper from scratch (downloads monthly Yahoo + FRED data; also slow)
uv run python notes/generate_credit_line_report.py
```

Do not put these 20,000-path runs in CI. GitHub Pages deploys the static `docs/` folder only.

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

The allocation report uses local CSVs (`historical_asset_returns.csv`, `historical_inflation.csv`) and a fixed seed. The credit-line report downloads Yahoo `^SP500TR` and FRED CPI and uses a different seed. Enable "Fixed random seed" in the Streamlit sidebar to share a specific annual scenario; that is not sufficient to reproduce either published HTML file by itself.

## Project Structure

- `simulator.py` — Market models (`RandomWalkMarket`, `BlockBootstrapMarket`, `PairedBlockBootstrapMarket`, `MeanRevertingMarket`) + `run_simulation` engine + strategy context.
- `spending.py` — Shared target/cap/floor withdrawal rule.
- `metrics.py` — Shared ruin, shortfall, and fan-chart statistics.
- `strategies.py` — The `CashStrategy` protocol and the three built-in implementations.
- `app.py` — Streamlit UI (reactive, well-documented controls, fan charts, risk metrics).
- `test_simulator.py` + `test_credit_line_report.py` — Executable specification.
- `notes/generate_report_results.py` / `notes/generate_report_assets.py` / `notes/generate_report_html.py` — Allocation-report Monte Carlo, charts, and HTML.
- `notes/generate_credit_line_report.py` — Monthly credit-line report generator.
- `notes/generate_followup_experiments.py` — Optional TIPS / pro-rata / BOY / ERP / circular-block cases. Use `--n-paths 200` locally; do not run 20,000 paths in CI.
- `docs/` — GitHub Pages assets. `docs/index.html` is the allocation report; `docs/credit-line.html` is the sibling.

See [AGENTS.md](AGENTS.md) for engineering guidelines and [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose changes.

## License

MIT — see [LICENSE](LICENSE).

## Citation / Attribution

If you use the reports or simulator in your own writing or decisions, a link back to the repository + credit to the specific report is appreciated. The methodology (flexible target/cap/floor + block bootstrap + explicit cash/buffer/replenishment rules) is the intellectual core.

---

**Why this exists**: Most retirement tools either (a) hide behind a single "safe withdrawal rate" or (b) use overly optimistic parametric assumptions. This project tries to be explicit about the trade-offs and let the data (historical blocks) speak.
