"""Run allocation-paper follow-up experiments (not 20k in CI)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics import summarize_withdrawals
from simulator import (
    PairedBlockBootstrapMarket,
    apply_erp_haircut,
    derive_spending_targets,
    get_stock_bond_data,
    run_simulation,
    tips_proxy_returns,
)
from strategies import ConservativeStrategy, FloorFundingStrategy, ProRataBondStrategy

SEED = 20260513
BASE = 1_000_000
FLOOR_RATIO = 0.5
WEALTH_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
ERP_LEVELS = [1.0, 0.75, 0.5, 0.25, 0.0]
BOND_PCTS = [0.0, 0.2, 0.4, 0.6]
FLOOR_YEARS = [10, 20, 25]


def wealth_percentiles(final_wealth: object, initial_wealth: float = BASE) -> dict[str, float]:
    import numpy as np

    multiples = np.asarray(final_wealth, dtype=float) / initial_wealth
    return {
        f"p{int(p)}": float(np.percentile(multiples, p)) for p in WEALTH_PERCENTILES
    }


def run_case(
    asset_history: dict,
    *,
    n_paths: int,
    years: int,
    spending_cap_pct: float,
    bond_pct: float,
    strategy,
    withdraw_before_returns: bool = False,
    block_mode: str = "overlapping",
    bond_returns=None,
    stock_returns=None,
) -> dict:
    target, floor = derive_spending_targets(BASE, spending_cap_pct, FLOOR_RATIO)
    market = PairedBlockBootstrapMarket(
        stock_returns=stock_returns if stock_returns is not None else asset_history["stock_returns"],
        bond_returns=bond_returns if bond_returns is not None else asset_history["bond_returns"],
        inflation_rates=asset_history["inflation_rates"],
        cash_returns=asset_history["tbill_returns"],
        block_size=5,
    )
    results = run_simulation(
        initial_net_worth=BASE,
        annual_spend=target,
        minimum_annual_spend=floor,
        buffer_years=0,
        years=years,
        panic_threshold=-0.15,
        inflation_rate=0.03,
        n_paths=n_paths,
        market_model=market,
        spending_cap_pct=spending_cap_pct,
        cash_interest_rate=None,
        strategy=strategy,
        bond_allocation_pct=bond_pct,
        random_seed=SEED,
        withdraw_before_returns=withdraw_before_returns,
        block_mode=block_mode,
    )
    summary = summarize_withdrawals(
        results["withdrawal_values"],
        target,
        floor,
        final_wealth=results["portfolio_values"][-1],
        initial_wealth=BASE,
    )
    summary.update(
        {
            "spending_cap_pct": spending_cap_pct,
            "bond_pct": bond_pct,
            "years": years,
            "n_paths": n_paths,
            "wealth_percentiles": wealth_percentiles(results["portfolio_values"][-1]),
        }
    )
    return summary


def build_robustness_rows(assets: dict, n_paths: int) -> list[dict]:
    tips = tips_proxy_returns(assets["inflation_rates"])
    cases = [
        ("4% crash-sell-bonds 60/40", dict(spending_cap_pct=0.04, bond_pct=0.4, strategy=ConservativeStrategy())),
        ("4% pro-rata 60/40", dict(spending_cap_pct=0.04, bond_pct=0.4, strategy=ProRataBondStrategy())),
        (
            "50% zero-real safe sleeve (rebalanced)",
            dict(
                spending_cap_pct=0.04,
                bond_pct=0.5,
                strategy=ConservativeStrategy(),
                bond_returns=tips,
            ),
        ),
        ("4% stock-only beginning-of-year", dict(spending_cap_pct=0.04, bond_pct=0.0, strategy=ConservativeStrategy(), withdraw_before_returns=True)),
        ("4% stock-only 40-year", dict(spending_cap_pct=0.04, bond_pct=0.0, strategy=ConservativeStrategy(), years=40)),
        ("4% stock-only circular blocks", dict(spending_cap_pct=0.04, bond_pct=0.0, strategy=ConservativeStrategy(), block_mode="circular")),
    ]
    rows = []
    for label, kwargs in cases:
        kwargs.setdefault("years", 30)
        row = run_case(assets, n_paths=n_paths, **kwargs)
        row["label"] = label
        rows.append(row)
    return rows


def build_erp_grid(assets: dict, n_paths: int) -> list[dict]:
    rows = []
    for haircut in ERP_LEVELS:
        stock = apply_erp_haircut(
            assets["stock_returns"], assets["tbill_returns"], haircut
        )
        for bond_pct in BOND_PCTS:
            row = run_case(
                assets,
                n_paths=n_paths,
                years=30,
                spending_cap_pct=0.04,
                bond_pct=bond_pct,
                strategy=ConservativeStrategy(),
                stock_returns=stock,
            )
            row["label"] = f"ERP {haircut:.0%} / {bond_pct:.0%} bonds"
            row["erp_kept"] = haircut
            rows.append(row)
    return rows


def build_floor_rows(assets: dict, n_paths: int) -> list[dict]:
    tips = tips_proxy_returns(assets["inflation_rates"])
    rows = []
    for years_of_floor in FLOOR_YEARS:
        bond_pct = years_of_floor * 0.04 * FLOOR_RATIO
        row = run_case(
            assets,
            n_paths=n_paths,
            years=30,
            spending_cap_pct=0.04,
            bond_pct=bond_pct,
            strategy=FloorFundingStrategy(),
            bond_returns=tips,
        )
        row["label"] = f"4% run-down floor bucket ({years_of_floor}y)"
        row["floor_years"] = years_of_floor
        rows.append(row)
    return rows


def build_followup_rows(n_paths: int = 200) -> list[dict]:
    """Backward-compatible robustness rows used by existing tests."""
    assets = get_stock_bond_data(history_years=75)
    return build_robustness_rows(assets, n_paths)


def build_followup_payload(n_paths: int = 200) -> dict:
    assets = get_stock_bond_data(history_years=75)
    rows = build_robustness_rows(assets, n_paths)
    return {
        "n_paths": n_paths,
        "seed": SEED,
        "rows": rows,
        "erp_grid": build_erp_grid(assets, n_paths),
        "floor_rows": build_floor_rows(assets, n_paths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "notes" / "assets" / "followup_results.json",
    )
    args = parser.parse_args()
    payload = build_followup_payload(n_paths=args.n_paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    docs_copy = ROOT / "docs" / "assets" / "followup_results.json"
    docs_copy.parent.mkdir(parents=True, exist_ok=True)
    docs_copy.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Wrote {args.out} "
        f"({len(payload['rows'])} robustness, {len(payload['erp_grid'])} ERP, "
        f"{len(payload['floor_rows'])} floor; {args.n_paths} paths)"
    )


if __name__ == "__main__":
    main()
