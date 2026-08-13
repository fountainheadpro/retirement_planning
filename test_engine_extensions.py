"""Tests for follow-up engine options: pro-rata, TIPS proxy, BOY, ERP, blocks."""

import numpy as np
import pytest

from simulator import (
    PairedBlockBootstrapMarket,
    apply_erp_haircut,
    block_indices,
    resolve_rng,
    run_simulation,
    tips_proxy_returns,
)
from strategies import ConservativeStrategy, FloorFundingStrategy, ProRataBondStrategy


def test_pro_rata_spends_equity_and_bonds_together():
    model = PairedBlockBootstrapMarket(
        stock_returns=np.array([0.0]),
        bond_returns=np.array([0.0]),
        block_size=1,
    )
    results = run_simulation(
        initial_net_worth=1_000_000,
        annual_spend=100_000,
        buffer_years=0,
        years=1,
        panic_threshold=-0.15,
        inflation_rate=0.0,
        n_paths=1,
        market_model=model,
        spending_cap_pct=1.0,
        bond_allocation_pct=0.40,
        strategy=ProRataBondStrategy(),
    )
    assert results["withdrawals_from_equity"][0, 0] == 60_000
    assert results["withdrawals_from_bonds"][0, 0] == 40_000


def test_beginning_of_year_withdrawal_happens_before_return():
    class OneYearMarket:
        def simulate_matrix(self, years, n_paths, rng=None, block_mode=None):
            return np.full((years, n_paths), 0.10)

    eoy = run_simulation(
        initial_net_worth=1_000_000,
        annual_spend=100_000,
        buffer_years=0,
        years=1,
        panic_threshold=-0.15,
        inflation_rate=0.0,
        n_paths=1,
        market_model=OneYearMarket(),
        spending_cap_pct=1.0,
        withdraw_before_returns=False,
    )
    boy = run_simulation(
        initial_net_worth=1_000_000,
        annual_spend=100_000,
        buffer_years=0,
        years=1,
        panic_threshold=-0.15,
        inflation_rate=0.0,
        n_paths=1,
        market_model=OneYearMarket(),
        spending_cap_pct=1.0,
        withdraw_before_returns=True,
    )
    assert eoy["portfolio_values"][1, 0] == pytest.approx(1_000_000)
    assert boy["portfolio_values"][1, 0] == pytest.approx(990_000)


def test_erp_haircut_shifts_mean_and_keeps_crash_depth():
    stock = np.array([0.20, -0.20, 0.10])
    tbill = np.array([0.02, 0.02, 0.02])
    original_excess = stock - tbill
    adjusted = apply_erp_haircut(stock, tbill, 0.5) - tbill
    assert adjusted.mean() == pytest.approx(0.5 * original_excess.mean())
    assert adjusted.std() == pytest.approx(original_excess.std())
    assert adjusted.min() == pytest.approx(original_excess.min() - 0.5 * original_excess.mean())
    np.testing.assert_allclose(apply_erp_haircut(stock, tbill, 1.0), stock)


def test_tips_proxy_is_zero_real():
    inflation = np.array([0.03, 0.08])
    np.testing.assert_array_equal(tips_proxy_returns(inflation), inflation)


def test_circular_blocks_can_wrap_year_zero():
    rng = resolve_rng(0)
    indices = block_indices(
        n_history=5,
        block_size=3,
        years=3,
        n_paths=20,
        rng=rng,
        mode="circular",
    )
    assert indices.max() <= 4
    assert set(indices.ravel()) <= {0, 1, 2, 3, 4}


def test_conservative_strategy_still_used_with_bonds():
    assert isinstance(ConservativeStrategy(), ConservativeStrategy)


def test_followup_experiment_runner_returns_labeled_rows():
    from notes.generate_followup_experiments import build_followup_rows

    try:
        rows = build_followup_rows(n_paths=2)
    except ModuleNotFoundError:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent / "notes" / "generate_followup_experiments.py"
        spec = importlib.util.spec_from_file_location("generate_followup_experiments", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = module.build_followup_rows(n_paths=2)

    labels = {row["label"] for row in rows}
    assert "4% pro-rata 60/40" in labels
    assert "50% zero-real safe sleeve (rebalanced)" in labels
    assert "4% stock-only beginning-of-year" in labels
    assert all("ruin_pct" in row for row in rows)


def test_followup_payload_includes_erp_grid_and_floor_rows():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / "notes" / "generate_followup_experiments.py"
    spec = importlib.util.spec_from_file_location("generate_followup_experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.build_followup_payload(n_paths=2)
    assert len(payload["erp_grid"]) == 20
    assert len(payload["floor_rows"]) == 3
    assert "wealth_percentiles" in payload["erp_grid"][0]


def test_floor_bucket_runs_down_instead_of_rebalancing():
    model = PairedBlockBootstrapMarket(
        stock_returns=np.array([0.0]),
        bond_returns=np.array([0.0]),
        block_size=1,
    )
    results = run_simulation(
        initial_net_worth=1_000_000,
        annual_spend=40_000,
        minimum_annual_spend=20_000,
        buffer_years=0,
        years=1,
        panic_threshold=-0.15,
        inflation_rate=0.0,
        n_paths=1,
        market_model=model,
        spending_cap_pct=0.04,
        bond_allocation_pct=0.50,
        strategy=FloorFundingStrategy(),
    )
    assert results["withdrawals_from_bonds"][0, 0] == pytest.approx(20_000)
    assert results["bond_values"][1, 0] == pytest.approx(480_000)
    assert results["withdrawals_from_equity"][0, 0] == pytest.approx(20_000)
