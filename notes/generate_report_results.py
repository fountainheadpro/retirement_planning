"""Run report simulations and save normalized result tables as JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator import (
    BlockBootstrapMarket,
    PairedBlockBootstrapMarket,
    build_spending_reference_table,
    derive_spending_targets,
    get_stock_bond_data,
    run_simulation,
)
from strategies import ConservativeStrategy


NOTES_DIR = Path(__file__).resolve().parent
ASSET_DIR = NOTES_DIR / "assets"
RESULTS_PATH = ASSET_DIR / "bond_report_results.json"

BASE = 1_000_000
YEARS = 30
PANIC_THRESHOLD = -0.15
FALLBACK_INFLATION_RATE = 0.03
CASH_RATE = None
N_PATHS = 20_000
BLOCK_SIZE = 5
SEED = 20260513
BASELINE_SPENDING_CAP_PCT = 0.05
FLOOR_RATIO = 0.5


def summarize(results: dict, target_spend: float, floor_spend: float) -> dict:
    """Summarize portfolio and spending outcomes in normalized terms."""
    final = results["portfolio_values"][-1]
    withdrawals = results["withdrawal_values"]

    target_mask = withdrawals < (target_spend - 1e-6)
    floor_mask = withdrawals < (floor_spend - 1e-6)
    shortfall_counts = target_mask.sum(axis=0)
    affected_counts = shortfall_counts[shortfall_counts > 0]

    if np.any(target_mask):
        shortfall_withdrawals = withdrawals[target_mask]
        avg_shortfall_spend_pct_initial = float(np.mean(shortfall_withdrawals) / BASE)
        avg_shortfall_depth_pct_target = float(
            np.mean((target_spend - shortfall_withdrawals) / target_spend)
        )
    else:
        avg_shortfall_spend_pct_initial = float(target_spend / BASE)
        avg_shortfall_depth_pct_target = 0.0

    return {
        "ruin_pct": float(np.mean(final <= 1e-6) * 100),
        "target_shortfall_pct": float(np.mean(target_mask) * 100),
        "target_shortfall_ever_pct": float(np.mean(np.any(target_mask, axis=0)) * 100),
        "target_shortfall_median_years_if_any": (
            float(np.median(affected_counts)) if len(affected_counts) else 0.0
        ),
        "target_shortfall_avg_spend_pct_initial": avg_shortfall_spend_pct_initial,
        "target_shortfall_avg_depth_pct_target": avg_shortfall_depth_pct_target,
        "floor_breach_pct": float(np.mean(floor_mask) * 100),
        "floor_breach_ever_pct": float(np.mean(np.any(floor_mask, axis=0)) * 100),
        "final_p5_multiple": float(np.percentile(final / BASE, 5)),
        "final_p10_multiple": float(np.percentile(final / BASE, 10)),
        "final_median_multiple": float(np.median(final / BASE)),
        "median_withdrawal_y1_pct_initial": float(np.median(withdrawals[0]) / BASE),
        "median_withdrawal_y30_pct_initial": float(np.median(withdrawals[-1]) / BASE),
    }


def run_stock_simulation(
    stock_returns: np.ndarray,
    inflation_rates: np.ndarray,
    spending_cap_pct: float,
    buffer_years: int,
    block_size: int = BLOCK_SIZE,
) -> dict:
    target_spend, floor_spend = derive_spending_targets(BASE, spending_cap_pct, FLOOR_RATIO)
    market = BlockBootstrapMarket(
        stock_returns,
        block_size=block_size,
        inflation_rates=inflation_rates,
    )
    np.random.seed(SEED)
    results = run_simulation(
        initial_net_worth=BASE,
        annual_spend=target_spend,
        minimum_annual_spend=floor_spend,
        buffer_years=buffer_years,
        years=YEARS,
        panic_threshold=PANIC_THRESHOLD,
        inflation_rate=FALLBACK_INFLATION_RATE,
        n_paths=N_PATHS,
        market_model=market,
        spending_cap_pct=spending_cap_pct,
        cash_interest_rate=CASH_RATE,
        strategy=ConservativeStrategy(),
        bond_allocation_pct=0.0,
    )
    return summarize(results, target_spend=target_spend, floor_spend=floor_spend)


def run_bond_simulation(
    asset_history: dict,
    bond_pct: float,
    spending_cap_pct: float = BASELINE_SPENDING_CAP_PCT,
) -> dict:
    target_spend, floor_spend = derive_spending_targets(
        BASE,
        spending_cap_pct,
        FLOOR_RATIO,
    )
    market = PairedBlockBootstrapMarket(
        stock_returns=asset_history["stock_returns"],
        bond_returns=asset_history["bond_returns"],
        inflation_rates=asset_history["inflation_rates"],
        block_size=BLOCK_SIZE,
    )
    np.random.seed(SEED)
    results = run_simulation(
        initial_net_worth=BASE,
        annual_spend=target_spend,
        minimum_annual_spend=floor_spend,
        buffer_years=0,
        years=YEARS,
        panic_threshold=PANIC_THRESHOLD,
        inflation_rate=FALLBACK_INFLATION_RATE,
        n_paths=N_PATHS,
        market_model=market,
        spending_cap_pct=spending_cap_pct,
        cash_interest_rate=CASH_RATE,
        strategy=ConservativeStrategy(),
        bond_allocation_pct=bond_pct,
    )
    return summarize(results, target_spend=target_spend, floor_spend=floor_spend)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    baseline_assets = get_stock_bond_data(history_years=75)
    stock_returns = baseline_assets["stock_returns"]
    inflation_rates = baseline_assets["inflation_rates"]

    safe_withdrawal_rows = []
    for spending_cap_pct in [pct / 100 for pct in range(4, 11)]:
        target_spend, floor_spend = derive_spending_targets(BASE, spending_cap_pct, FLOOR_RATIO)
        row = {
            "spending_cap_pct": spending_cap_pct,
            "target_spending_pct_initial": spending_cap_pct,
            "floor_spending_pct_initial": floor_spend / BASE,
        }
        row.update(
            run_stock_simulation(
                stock_returns,
                inflation_rates,
                spending_cap_pct,
                buffer_years=0,
            )
        )
        safe_withdrawal_rows.append(row)

    cash_rows = []
    for buffer_years in [0, 1, 2, 3, 5]:
        row = {"buffer_years": buffer_years}
        row.update(
            run_stock_simulation(
                stock_returns,
                inflation_rates,
                BASELINE_SPENDING_CAP_PCT,
                buffer_years,
            )
        )
        cash_rows.append(row)

    history_rows = []
    for history_years in [50, 75, 98]:
        asset_history = get_stock_bond_data(history_years=history_years)
        row = {
            "history_years": history_years,
            "start_year": int(asset_history["years"][0]),
            "end_year": int(asset_history["years"][-1]),
        }
        row.update(
            run_stock_simulation(
                asset_history["stock_returns"],
                asset_history["inflation_rates"],
                BASELINE_SPENDING_CAP_PCT,
                buffer_years=0,
            )
        )
        history_rows.append(row)

    block_size_rows = []
    for block_size in [1, 3, 5, 10]:
        row = {"block_size_years": block_size}
        row.update(
            run_stock_simulation(
                stock_returns,
                inflation_rates,
                BASELINE_SPENDING_CAP_PCT,
                buffer_years=0,
                block_size=block_size,
            )
        )
        block_size_rows.append(row)

    bond_rows = []
    for bond_pct in [0.0, 0.1, 0.2, 0.4, 0.6]:
        row = {"bond_pct": bond_pct}
        row.update(run_bond_simulation(baseline_assets, bond_pct))
        bond_rows.append(row)

    traditional_benchmark_rows = []
    for spending_cap_pct, bond_pct, label in [
        (0.04, 0.0, "4% stock-only"),
        (0.04, 0.4, "4% 60/40"),
        (0.05, 0.0, "5% stock-only"),
        (0.05, 0.4, "5% 60/40"),
    ]:
        target_spend, floor_spend = derive_spending_targets(
            BASE,
            spending_cap_pct,
            FLOOR_RATIO,
        )
        row = {
            "label": label,
            "spending_cap_pct": spending_cap_pct,
            "target_spending_pct_initial": spending_cap_pct,
            "floor_spending_pct_initial": floor_spend / BASE,
            "bond_pct": bond_pct,
        }
        row.update(
            run_bond_simulation(
                baseline_assets,
                bond_pct=bond_pct,
                spending_cap_pct=spending_cap_pct,
            )
        )
        traditional_benchmark_rows.append(row)

    results = {
        "settings": {
            "base_initial_net_worth": BASE,
            "target_spending_pct_initial": BASELINE_SPENDING_CAP_PCT,
            "floor_spending_pct_initial": BASELINE_SPENDING_CAP_PCT * FLOOR_RATIO,
            "floor_ratio": FLOOR_RATIO,
            "years": YEARS,
            "panic_threshold": PANIC_THRESHOLD,
            "fallback_inflation_rate": FALLBACK_INFLATION_RATE,
            "cash_interest_rate": CASH_RATE,
            "cash_return_source": "Cash return matches sampled inflation, so cash earns 0% real return.",
            "n_paths": N_PATHS,
            "block_size_years": BLOCK_SIZE,
            "seed": SEED,
            "return_data_source": "historical_asset_returns.csv",
            "inflation_data_source": "historical_inflation.csv",
            "stock_return_source": "Damodaran S&P 500 total return (StockReturn; dividends reinvested)",
            "bond_return_source": "Damodaran TreasuryBondReturn",
            "inflation_source": "FRED CPIAUCNS, December-to-December CPI-U inflation",
            "baseline_start_year": int(baseline_assets["years"][0]),
            "baseline_end_year": int(baseline_assets["years"][-1]),
        },
        "spending_reference_rows": build_spending_reference_table(),
        "safe_withdrawal_rows": safe_withdrawal_rows,
        "cash_rows": cash_rows,
        "history_rows": history_rows,
        "block_size_rows": block_size_rows,
        "traditional_benchmark_rows": traditional_benchmark_rows,
        "bond_rows": bond_rows,
        "stock_bond_years": [
            int(baseline_assets["years"][0]),
            int(baseline_assets["years"][-1]),
        ],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
