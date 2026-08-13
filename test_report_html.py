"""Lock published HTML headline cells to the saved JSON."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allocation_html_matches_json_headlines():
    html = (ROOT / "docs" / "index.html").read_text()
    results = json.loads((ROOT / "docs" / "assets" / "bond_report_results.json").read_text())
    gen = load_module("generate_report_html", "notes/generate_report_html.py")

    flex4_stock = gen.find_label(results, "flex4_stock")
    flex4_6040 = gen.find_label(results, "flex4_6040")
    fixed4 = gen.find_label(results, "fixed4_6040")
    n_paths = results["settings"]["n_paths"]

    assert gen.format_pct(fixed4["ruin_pct"]) in html
    assert gen.format_pct(flex4_stock["ruin_pct"]) in html
    assert gen.format_shortfall(flex4_6040) in html
    assert gen.format_x(flex4_stock["final_median_multiple"]) in html
    assert gen.format_pct(flex4_stock["target_shortfall_ever_pct"]) in html
    assert gen.format_ruin(flex4_6040, n_paths) in html
    assert "probability of ruin" in html
    assert "30 paths" not in html
    assert "years real wealth below start" in html
    assert "docs/assets/bond_report_results.json" in html
    assert 'href="credit-line.html"' in html
    assert 'aria-current="page">Allocation report</a>' in html


def test_allocation_html_is_generated_from_json():
    gen = load_module("generate_report_html", "notes/generate_report_html.py")
    rendered = gen.render(gen.load_results())
    current = (ROOT / "docs" / "index.html").read_text()
    assert current == rendered


def test_credit_line_html_matches_json_headlines():
    html = (ROOT / "docs" / "credit-line.html").read_text()
    results = json.loads((ROOT / "docs" / "assets" / "credit_line_results.json").read_text())
    baseline = results["scenarios"][0]["rows"][0]
    assert f"{baseline['target_shortfall_pct']:.2f}%" in html
    assert f"{baseline['ruin_pct']:.2f}%" in html
    assert 'href="index.html"' in html
    assert 'aria-current="page">Credit-line report</a>' in html
