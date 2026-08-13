"""Target / cap / floor spending rule used by the annual and monthly engines."""

from __future__ import annotations

import numpy as np


def derive_spending_targets(
    initial_net_worth: float,
    spending_cap_pct: float,
    floor_ratio: float = 0.5,
) -> tuple[float, float]:
    """Derive real target and floor spending from initial wealth and cap."""
    target_spend = max(0.0, initial_net_worth * spending_cap_pct)
    floor_ratio = min(max(floor_ratio, 0.0), 1.0)
    return target_spend, target_spend * floor_ratio


def apply_target_cap_floor(
    wealth: np.ndarray | float,
    target: float,
    cap_pct: float,
    floor: float,
) -> np.ndarray:
    """Apply withdrawal = min(wealth, max(floor, min(target, wealth * cap_pct))).

    When ``target == start * cap_pct``, a year is a target shortfall if and only
    if real wealth is below that starting wealth. If the target rate and cap
    rate differ, that identity does not hold.
    """
    wealth_array = np.maximum(np.asarray(wealth, dtype=float), 0.0)
    floor_spend = min(max(float(floor), 0.0), float(target))
    cap_amount = wealth_array * float(cap_pct)
    desired = np.maximum(np.minimum(float(target), cap_amount), floor_spend)
    return np.minimum(desired, wealth_array)


def build_spending_reference_table(
    spending_cap_pct: float = 0.05,
    portfolio_values: tuple[float, ...] = (
        2_000_000,
        3_000_000,
        4_000_000,
        5_000_000,
        6_000_000,
        10_000_000,
    ),
    floor_ratio: float = 0.5,
) -> list[dict[str, str]]:
    """Build display-ready target/floor rows for common portfolio sizes."""
    target_label = f"Target ({spending_cap_pct:.0%})"
    floor_label = f"Floor ({spending_cap_pct * floor_ratio:.1%})"
    rows = []

    for portfolio_value in portfolio_values:
        target_spend, floor_spend = derive_spending_targets(
            initial_net_worth=portfolio_value,
            spending_cap_pct=spending_cap_pct,
            floor_ratio=floor_ratio,
        )
        rows.append(
            {
                "Portfolio": f"${portfolio_value / 1_000_000:g}M",
                target_label: f"${target_spend:,.0f}",
                floor_label: f"${floor_spend:,.0f}",
            }
        )

    return rows
