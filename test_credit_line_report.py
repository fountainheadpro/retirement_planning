"""Tests for the standalone credit-line report generator."""

import importlib.util
from pathlib import Path

import numpy as np


def load_credit_line_module():
    path = Path(__file__).resolve().parent / "notes" / "generate_credit_line_report.py"
    spec = importlib.util.spec_from_file_location("generate_credit_line_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spending_amount_uses_supplied_target_rate():
    """Credit-line report spending should support both 4% and 5% targets."""
    module = load_credit_line_module()
    wealth = np.array([1.0])

    four_pct = module.spending_amount(wealth, target_rate=0.04)
    five_pct = module.spending_amount(wealth, target_rate=0.05)

    assert four_pct[0] == 0.04 / 12.0
    assert five_pct[0] == 0.05 / 12.0


def test_run_strategy_rows_labels_target_rate():
    """Generated rows should carry the spending-rate scenario label."""
    module = load_credit_line_module()
    module.MONTHS = 2
    module.N_PATHS = 2
    module.TRIGGERS = [0.10]
    paths = {
        "stock_nominal": np.zeros((module.MONTHS, module.N_PATHS)),
        "stock_real": np.zeros((module.MONTHS, module.N_PATHS)),
        "inflation": np.zeros((module.MONTHS, module.N_PATHS)),
    }

    rows = module.run_strategy_rows(paths, target_rate=0.04)

    assert rows[0]["scenario"] == "4% target / 2% floor"
    assert rows[0]["target_rate"] == 0.04
