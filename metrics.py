"""Shared ruin, shortfall, and fan-chart statistics."""

from __future__ import annotations

import numpy as np


def ruin_terminal(final_wealth: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Boolean mask of paths whose terminal wealth is at or below the threshold."""
    return np.asarray(final_wealth) <= threshold


def ruin_ever(wealth_values: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """Boolean mask of paths that hit the threshold at any recorded date."""
    return np.any(np.asarray(wealth_values) <= threshold, axis=0)


def path_period_rate(mask: np.ndarray) -> float:
    """Fraction of all path-periods that are True, as a percent."""
    return float(np.mean(mask) * 100)


def summarize_withdrawals(
    withdrawals: np.ndarray,
    target: float,
    floor: float,
    *,
    final_wealth: np.ndarray | None = None,
    initial_wealth: float = 1.0,
    tolerance: float = 1e-6,
) -> dict:
    """Summarize spending outcomes in the allocation-report field names.

    ``target_shortfall_pct`` is the percent of all path-periods with
    withdrawal below target. When the spending rule uses the same rate for
    target and cap, that equals the share of periods real wealth is below
    start.
    """
    withdrawals = np.asarray(withdrawals, dtype=float)
    target_mask = withdrawals < (target - tolerance)
    floor_mask = withdrawals < (floor - tolerance)
    shortfall_counts = target_mask.sum(axis=0)
    affected_counts = shortfall_counts[shortfall_counts > 0]
    shortfall_loss = np.maximum(target - withdrawals, 0.0) / target if target > 0 else np.zeros_like(withdrawals)

    if np.any(target_mask):
        shortfall_withdrawals = withdrawals[target_mask]
        avg_shortfall_spend_pct_initial = float(np.mean(shortfall_withdrawals) / initial_wealth)
        avg_shortfall_depth_pct_target = float(
            np.mean((target - shortfall_withdrawals) / target)
        )
    else:
        avg_shortfall_spend_pct_initial = float(target / initial_wealth) if initial_wealth else 0.0
        avg_shortfall_depth_pct_target = 0.0

    if final_wealth is None:
        ruin_mask = np.zeros(withdrawals.shape[1], dtype=bool)
    else:
        ruin_mask = ruin_terminal(final_wealth, threshold=tolerance)

    return {
        "ruin_pct": float(np.mean(ruin_mask) * 100),
        "ruin_count": int(np.sum(ruin_mask)),
        "target_shortfall_pct": path_period_rate(target_mask),
        "target_shortfall_path_years": int(np.sum(target_mask)),
        "target_shortfall_ever_pct": float(np.mean(np.any(target_mask, axis=0)) * 100),
        "target_shortfall_median_years_if_any": (
            float(np.median(affected_counts)) if len(affected_counts) else 0.0
        ),
        "target_shortfall_avg_spend_pct_initial": avg_shortfall_spend_pct_initial,
        "target_shortfall_avg_depth_pct_target": avg_shortfall_depth_pct_target,
        "target_shortfall_integrated_loss_pct_target": float(np.mean(shortfall_loss) * 100),
        "target_shortfall_integrated_loss_years": float(np.mean(shortfall_loss.sum(axis=0))),
        "target_spend_delivered_pct": float((1.0 - np.mean(shortfall_loss)) * 100),
        "floor_breach_pct": path_period_rate(floor_mask),
        "floor_breach_path_years": int(np.sum(floor_mask)),
        "floor_breach_ever_pct": float(np.mean(np.any(floor_mask, axis=0)) * 100),
        "final_p5_multiple": (
            float(np.percentile(np.asarray(final_wealth) / initial_wealth, 5))
            if final_wealth is not None
            else float("nan")
        ),
        "final_p10_multiple": (
            float(np.percentile(np.asarray(final_wealth) / initial_wealth, 10))
            if final_wealth is not None
            else float("nan")
        ),
        "final_median_multiple": (
            float(np.median(np.asarray(final_wealth) / initial_wealth))
            if final_wealth is not None
            else float("nan")
        ),
        "median_withdrawal_y1_pct_initial": float(np.median(withdrawals[0]) / initial_wealth),
        "median_withdrawal_y30_pct_initial": float(np.median(withdrawals[-1]) / initial_wealth),
    }


def calculate_statistics(
    portfolio_values: np.ndarray,
    withdrawal_values: np.ndarray,
    confidence: float,
) -> dict:
    """Calculate percentile statistics for visualization."""
    alpha = (1 - confidence) / 2

    return {
        "portfolio": {
            "lower": np.percentile(portfolio_values, alpha * 100, axis=1),
            "upper": np.percentile(portfolio_values, (1 - alpha) * 100, axis=1),
            "median": np.median(portfolio_values, axis=1),
        },
        "withdrawal": {
            "lower": np.percentile(withdrawal_values, alpha * 100, axis=1),
            "median": np.median(withdrawal_values, axis=1),
        },
    }
