# Repository Guidelines

## Project Structure & Module Organization
Source code lives at the repository root: `app.py` handles the Streamlit UI while `simulator.py` hosts market models, withdrawal rules, and data utilities. Keep notebooks (`retirement_analysis*.ipynb`) output-light and lean on `test_simulator.py` for executable checks. `pyproject.toml`/`uv.lock` define the reproducible environment, and only edit `requirements.md` when the methodology truly changes.

## Build, Test, and Development Commands
- `uv sync` — install the locked Python 3.12 toolchain defined in `pyproject.toml`/`uv.lock`.
- `uv run streamlit run app.py` — launch the simulator UI locally (append `--server.headless true` for CI pipelines).
- `uv run pytest` or `uv run pytest test_simulator.py -k run_simulation` — execute the Monte Carlo and sourcing tests; finish each branch with a green run.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, `snake_case` for functions/variables, and `PascalCase` for market model classes (`RandomWalkMarket`, `MeanRevertingMarket`, etc.). Keep functions small and deterministic so they can be reused in both the UI and test harness, and prefer type hints plus docstrings similar to `ordinal` or `run_simulation`. Any new helper module should expose one clear entry point back to the UI.

## Testing Guidelines
Pytest with NumPy assertions is the canonical testing stack. Mirror existing naming such as `test_simulation_shape` and seed NumPy RNGs whenever randomness is involved so regressions are reproducible. Add scenario tests for every new withdrawal rule or market model before wiring it into `app.py`. When adding notebooks, capture validation snippets inside tests rather than relying solely on ad-hoc notebook checks.

## Commit & Pull Request Guidelines
Recent history (`Implement Block Bootstrap and UI Model Selection`, `added regime change option`) shows imperative, 50–60 character commit subjects; follow that pattern and group related file changes into a single commit. Pull requests must describe the motivation, summarize functional changes, and include before/after screenshots whenever the Streamlit UI shifts. Reference linked issues, list any new configuration steps, and paste the exact `uv run pytest` output so reviewers can trust the change.

## Data & Configuration Tips
Market data arrives via `yfinance` and is cached with `st.cache_data`; avoid tight download loops and refresh the cache interval when necessary. Keep secrets and proprietary datasets out of version control, and document any new environment variables inside the corresponding pull request so deployers can reproduce your setup safely.
