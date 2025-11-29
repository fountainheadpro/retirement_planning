"""Rolling decade backtests to compare market models."""

from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from simulator import BlockBootstrapMarket, MeanRevertingMarket, RandomWalkMarket


def fetch_sp500(start_year: int) -> pd.DataFrame:
    """Download S&P 500 closes and compute annual simple returns."""
    end = dt.date.today()
    start = dt.date(start_year, 1, 1)
    data = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    if data.empty or len(data) < 250:
        raise RuntimeError("Insufficient data fetched")
    annual = data["Close"].resample("YE").last().pct_change().dropna()
    returns = np.asarray(annual, dtype=float).ravel()
    df = pd.DataFrame({"ret": returns}, index=annual.index)
    df["year"] = df.index.year
    return df


def longest_negative_streak(returns: np.ndarray) -> int:
    """Length of the longest run of negative returns."""
    streak = longest = 0
    for r in returns:
        if r < 0:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return longest


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown given a return series."""
    index = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(index)
    drawdowns = 1 - index / peak
    return float(np.max(drawdowns))


def metric_bundle(ret_seq: np.ndarray) -> Dict[str, float]:
    """Compute core return metrics."""
    return {
        "cum_return": float(np.prod(1 + ret_seq) - 1),
        "mean": float(np.mean(ret_seq)),
        "vol": float(np.std(ret_seq)),
        "longest_neg": float(longest_negative_streak(ret_seq)),
        "max_dd": float(max_drawdown(ret_seq)),
    }


def percentile_of_value(array: np.ndarray, value: float) -> float:
    """Percentile position of value within array."""
    return float(np.mean(array <= value) * 100)


def simulate_model(model, horizon: int, n_paths: int, seed: int) -> Dict[str, np.ndarray]:
    """Run Monte Carlo on returns-only models and collect metric distributions."""
    np.random.seed(seed)
    sim = model.simulate_matrix(horizon, n_paths)  # shape: years x n_paths
    metrics: Dict[str, list] = {k: [] for k in ["cum_return", "mean", "vol", "longest_neg", "max_dd"]}
    for col in sim.T:
        mb = metric_bundle(col)
        for k, v in mb.items():
            metrics[k].append(v)
    return {k: np.asarray(v, dtype=float) for k, v in metrics.items()}


def score_against_actual(sim_metrics: Dict[str, np.ndarray], actual: Dict[str, float]) -> float:
    """Aggregate z-score distance of actual metrics from simulated distributions."""
    scores = []
    for key, arr in sim_metrics.items():
        std = max(np.std(arr), 1e-8)  # avoid divide by zero
        scores.append(abs((actual[key] - float(np.mean(arr))) / std))
    return float(sum(scores))


def build_models(train_returns: np.ndarray, block_size: int) -> Dict[str, object]:
    """Instantiate calibrated models from training returns."""
    mu = float(np.mean(train_returns))
    resid = train_returns - mu
    models: Dict[str, object] = {
        "Random Walk": RandomWalkMarket(mu, resid),
        f"Block Bootstrap ({block_size}y)": BlockBootstrapMarket(train_returns, block_size=block_size),
    }

    ar1 = MeanRevertingMarket(ar_order=1)
    ar1.calibrate_from_history(train_returns)
    models["AR(1)"] = ar1

    ar5 = MeanRevertingMarket(ar_order=5)
    ar5.calibrate_from_history(train_returns)
    models["AR(5)"] = ar5
    return models


def evaluate_window(train: np.ndarray, test: np.ndarray, n_paths: int, block_size: int, seed: int) -> pd.DataFrame:
    """Evaluate all models for one test window."""
    actual_metrics = metric_bundle(test)
    models = build_models(train, block_size=block_size)

    rows = []
    for name, model in models.items():
        sim_metrics = simulate_model(model, horizon=len(test), n_paths=n_paths, seed=seed)
        total_score = score_against_actual(sim_metrics, actual_metrics)
        rows.append(
            {
                "model": name,
                "score": total_score,
                "cum_return_pct": percentile_of_value(sim_metrics["cum_return"], actual_metrics["cum_return"]),
                "max_dd_pct": percentile_of_value(sim_metrics["max_dd"], actual_metrics["max_dd"]),
                "longest_neg_pct": percentile_of_value(sim_metrics["longest_neg"], actual_metrics["longest_neg"]),
                "vol_pct": percentile_of_value(sim_metrics["vol"], actual_metrics["vol"]),
            }
        )
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def run_backtest(
    start_year: int,
    horizon: int,
    block_size: int,
    n_paths: int,
    min_train_years: int,
    seed: int,
    decade_step: int,
) -> None:
    """Run rolling backtests and print winners per window."""
    df = fetch_sp500(start_year=start_year)
    first_year = int(df["year"].min())
    last_year = int(df["year"].max())
    starts = []
    start = first_year + min_train_years
    while start + horizon - 1 <= last_year:
        starts.append(start)
        start += decade_step
    if not starts:
        raise RuntimeError("No valid windows found")

    print(f"Using S&P data {first_year}-{last_year}, horizon={horizon}y, paths={n_paths}, block={block_size}y")
    formatters = {
        "score": lambda v: f"{v:0.2f}",
        "cum_return_pct": lambda v: f"{v:0.1f}%",
        "max_dd_pct": lambda v: f"{v:0.1f}%",
        "longest_neg_pct": lambda v: f"{v:0.1f}%",
        "vol_pct": lambda v: f"{v:0.1f}%",
    }

    for start_year in starts:
        end_year = start_year + horizon - 1
        train = df[df["year"] < start_year]["ret"].values
        test = df[(df["year"] >= start_year) & (df["year"] <= end_year)]["ret"].values
        if len(test) < horizon:
            continue
        summary = evaluate_window(train, test, n_paths=n_paths, block_size=block_size, seed=seed)
        winner = summary.iloc[0]
        print(f"\nWindow {start_year}-{end_year}: best={winner['model']} (score {winner['score']:.2f})")
        print(summary.to_string(index=False, formatters=formatters))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest market models on rolling 10-year windows.")
    parser.add_argument("--start-year", type=int, default=1950, help="First year to download data from (default: 1950)")
    parser.add_argument("--horizon", type=int, default=10, help="Test window in years (default: 10)")
    parser.add_argument("--block-size", type=int, default=5, help="Block size for bootstrap model (default: 5)")
    parser.add_argument("--paths", type=int, default=5000, help="Monte Carlo paths per model (default: 5000)")
    parser.add_argument("--min-train-years", type=int, default=20, help="Minimum training years before a window (default: 20)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility (default: 123)")
    parser.add_argument("--decade-step", type=int, default=10, help="Step between window starts (default: 10)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_backtest(
        start_year=args.start_year,
        horizon=args.horizon,
        block_size=args.block_size,
        n_paths=args.paths,
        min_train_years=args.min_train_years,
        seed=args.seed,
        decade_step=args.decade_step,
    )


if __name__ == "__main__":
    main()
