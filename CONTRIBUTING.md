# Contributing to Retirement Planning Simulator

Thank you for your interest! This project combines careful financial modeling with clean, testable engineering. Contributions that improve either (or both) are welcome.

## Development Setup

1. Install [uv](https://docs.astral.sh/uv/) (recommended) or use `pip`.
2. `uv sync` — installs the exact Python 3.12 environment from `pyproject.toml` / `uv.lock`.
3. `uv run pytest` — all 50+ tests should pass before you submit.
4. `uv run streamlit run app.py` — run the interactive UI locally.

## Code Style & Quality (enforced)

Follow the rules in [AGENTS.md](AGENTS.md):

- PEP 8, 4-space indentation.
- `snake_case` for functions/variables; `PascalCase` for market model classes.
- Type hints + docstrings on all public functions (see `run_simulation`, `ordinal`, `derive_spending_targets`).
- **Add scenario tests for every new withdrawal rule, market model, or strategy** *before* wiring it into `app.py`.
- Keep notebooks output-light; put executable checks in `test_simulator.py`.
- Finish every change with a clean `uv run pytest` run (paste the output in your PR).

## Pull Request Process

- Keep PRs focused. One logical change per PR when possible.
- Imperative commit subjects, ~50-60 characters (e.g. "Add RNG seed for reproducibility").
- In the PR description:
  - Motivation (why this matters for retirement modeling or usability).
  - Summary of functional changes.
  - Before/after screenshots if the Streamlit UI changed.
  - Exact `uv run pytest` output.
  - Any new configuration steps or data requirements.
- Reference related reports or issues.

## Research & Reports

The published reports in `docs/` are a core deliverable:

- `docs/index.html` — allocation / 4% report (annual engine).
- `docs/credit-line.html` — credit-line report (monthly engine).

Changes that would alter the numbers in those reports require updated report text **and** regenerated assets. After changing `simulator.py`, spending rules, cash/bond logic, or a report generator:

```bash
uv run python notes/generate_report_results.py
uv run python notes/generate_report_assets.py
uv run python notes/generate_report_html.py
uv run python notes/generate_credit_line_report.py --render-only
```

Do not run the 20,000-path generators in CI. `pages.yml` only deploys the static `docs/` folder.

## Questions?

Open an issue or start a discussion. We're particularly interested in:

- Better visualizations of sequence risk
- Additional defensive strategies expressed through the `CashStrategy` protocol
- Real-world calibration improvements while preserving the "target / cap / floor" philosophy

Thanks for helping make retirement planning tools more honest and useful!
