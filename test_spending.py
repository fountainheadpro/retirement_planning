"""Tests for the shared target/cap/floor spending rule."""

import numpy as np

from spending import apply_target_cap_floor, derive_spending_targets


def test_shortfall_equals_wealth_below_start_when_rates_match():
    """When target rate equals cap rate, shortfall years are years below start."""
    start = 1.0
    cap_pct = 0.05
    target, floor = derive_spending_targets(start, cap_pct)
    wealth = np.array([0.0, 0.01, 0.4, 0.5, 0.99, 1.0, 1.2, 2.0])

    desired = apply_target_cap_floor(wealth, target, cap_pct, floor)

    np.testing.assert_array_equal(desired < target, wealth < start)


def test_identity_breaks_when_target_and_cap_rates_differ():
    """A lower cap than the target rate can cut spending even above start."""
    target = 0.05
    floor = 0.025
    cap_pct = 0.04
    wealth = np.array([1.1])

    desired = apply_target_cap_floor(wealth, target, cap_pct, floor)

    assert wealth[0] >= 1.0
    assert desired[0] < target
    np.testing.assert_allclose(desired[0], 1.1 * cap_pct)


def test_floor_overrides_cap_until_wealth_is_exhausted():
    """The floor can exceed the cap; remaining wealth is the last constraint."""
    desired = apply_target_cap_floor(
        wealth=np.array([0.4, 0.02]),
        target=0.05,
        cap_pct=0.05,
        floor=0.025,
    )

    np.testing.assert_allclose(desired, [0.025, 0.02])


def test_derive_spending_targets_clamps_floor_ratio():
    target, floor = derive_spending_targets(100.0, 0.05, floor_ratio=2.0)
    assert target == 5.0
    assert floor == 5.0
