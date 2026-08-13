"""Tests for shared ruin and shortfall statistics."""

import numpy as np

from metrics import ruin_ever, ruin_terminal, summarize_withdrawals


def test_summarize_withdrawals_on_synthetic_paths():
    """Known shortfall, floor, and ruin counts should match the masks."""
    withdrawals = np.array(
        [
            [1.00, 1.00],
            [0.80, 1.00],
            [0.00, 1.00],
        ]
    )
    final = np.array([0.0, 2.0])

    summary = summarize_withdrawals(
        withdrawals,
        target=1.0,
        floor=0.5,
        final_wealth=final,
        initial_wealth=1.0,
    )

    assert summary["ruin_count"] == 1
    assert summary["ruin_pct"] == 50.0
    assert summary["target_shortfall_path_years"] == 2
    assert abs(summary["target_shortfall_pct"] - (2 / 6) * 100) < 1e-9
    assert summary["target_shortfall_ever_pct"] == 50.0
    assert summary["floor_breach_path_years"] == 1
    assert summary["final_median_multiple"] == 1.0


def test_ruin_ever_catches_mid_horizon_zero():
    values = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.8],
            [0.4, 0.9],
        ]
    )
    np.testing.assert_array_equal(ruin_ever(values), [True, False])
    np.testing.assert_array_equal(ruin_terminal(values[-1]), [False, False])
