"""Render docs/index.html from bond_report_results.json."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES_ASSETS = ROOT / "notes" / "assets"
DOCS_DIR = ROOT / "docs"
DOCS_ASSETS = DOCS_DIR / "assets"
RESULTS_NAME = "bond_report_results.json"
FOLLOWUP_NAME = "followup_results.json"
REPORT_PATH = DOCS_DIR / "index.html"

LABELS = {
    "flex4_stock": "4% target / 2% floor, stock-only flexible rule",
    "flex4_6040": "4% target / 2% floor, 60/40 flexible rule",
    "fixed4_6040": "Fixed real 4% withdrawal, 60/40, no spending flexibility",
    "flex5_stock": "5% target / 2.5% floor, stock-only flexible rule",
    "flex5_6040": "5% target / 2.5% floor, 60/40 flexible rule",
}


def load_results() -> dict:
    """Load results from docs/assets, copying from notes/assets if needed."""
    docs_path = DOCS_ASSETS / RESULTS_NAME
    notes_path = NOTES_ASSETS / RESULTS_NAME
    if not docs_path.exists() and notes_path.exists():
        DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notes_path, docs_path)
    if notes_path.exists():
        for png in NOTES_ASSETS.glob("bond_report_*.png"):
            dest = DOCS_ASSETS / png.name
            if not dest.exists() or png.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(png, dest)
    results = json.loads(docs_path.read_text())
    followup = None
    for candidate in (NOTES_ASSETS / FOLLOWUP_NAME, DOCS_ASSETS / FOLLOWUP_NAME):
        if candidate.exists():
            followup = json.loads(candidate.read_text())
            break
    if followup is not None:
        results["followup"] = followup
    return results


def find_label(results: dict, key: str) -> dict:
    label = LABELS[key]
    for row in results["traditional_benchmark_rows"]:
        if row["label"] == label:
            return row
    raise KeyError(label)


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def format_x(value: float) -> str:
    return f"{value:.2f}x"


def format_int(value: float) -> str:
    return f"{int(round(value))}"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval on a path-level rate, returned in percent."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half) * 100.0, min(1.0, center + half) * 100.0


def format_ruin_pct(row: dict) -> str:
    """Probability of ruin as a percent. Tiny rates keep an extra decimal."""
    p = float(row["ruin_pct"])
    if p <= 0.0:
        return "0.00%"
    if p < 0.05:
        return f"{p:.3f}%"
    return f"{p:.2f}%"


def format_ruin(row: dict, n_paths: int) -> str:
    low, high = wilson_interval(int(row["ruin_count"]), n_paths)
    return f"{format_ruin_pct(row)} (95% CI {low:.2f}–{high:.2f}%)"


def format_shortfall(row: dict) -> str:
    return format_pct(row["target_shortfall_pct"])


def spend_delivered(row: dict) -> float:
    if "target_spend_delivered_pct" in row:
        return float(row["target_spend_delivered_pct"])
    return 100.0 - float(row["target_shortfall_integrated_loss_pct_target"])


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{c}</td>" for c in row)
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def spend_table_rows(results: dict) -> list[list[str]]:
    rows = []
    for row in results["spending_reference_rows"]:
        values = list(row.values())
        rows.append(values)
    return rows


def render(results: dict) -> str:
    settings = results["settings"]
    n_paths = int(settings["n_paths"])
    flex4_stock = find_label(results, "flex4_stock")
    flex4_6040 = find_label(results, "flex4_6040")
    fixed4 = find_label(results, "fixed4_6040")
    flex5_stock = find_label(results, "flex5_stock")
    flex5_6040 = find_label(results, "flex5_6040")
    cash0 = results["cash_rows"][0]
    cash5 = results["cash_rows"][-1]
    start_year = settings["baseline_start_year"]
    end_year = settings["baseline_end_year"]

    benchmark = table(
        [
            "Benchmark",
            "Spending rule",
            "Bonds",
            "Ruin",
            "Years wealth below start",
            "Ever miss target",
            "Spend delivered",
            "Integrated target loss",
            "Real final median",
        ],
        [
            [
                "4% stock-only",
                "4% target / 2% floor, flexible",
                "0%",
                format_ruin(flex4_stock, n_paths),
                format_shortfall(flex4_stock),
                format_pct(flex4_stock["target_shortfall_ever_pct"]),
                format_pct(spend_delivered(flex4_stock)),
                format_pct(flex4_stock["target_shortfall_integrated_loss_pct_target"]),
                format_x(flex4_stock["final_median_multiple"]),
            ],
            [
                "4% 60/40",
                "4% target / 2% floor, flexible",
                "40%",
                format_ruin(flex4_6040, n_paths),
                format_shortfall(flex4_6040),
                format_pct(flex4_6040["target_shortfall_ever_pct"]),
                format_pct(spend_delivered(flex4_6040)),
                format_pct(flex4_6040["target_shortfall_integrated_loss_pct_target"]),
                format_x(flex4_6040["final_median_multiple"]),
            ],
            [
                "Classic fixed-real 4% 60/40",
                "Fixed 4%, no spending flexibility",
                "40%",
                format_ruin(fixed4, n_paths),
                format_shortfall(fixed4),
                format_pct(fixed4["target_shortfall_ever_pct"]),
                format_pct(spend_delivered(fixed4)),
                format_pct(fixed4["target_shortfall_integrated_loss_pct_target"]),
                format_x(fixed4["final_median_multiple"]),
            ],
            [
                "5% stock-only",
                "5% target / 2.5% floor, flexible",
                "0%",
                format_ruin(flex5_stock, n_paths),
                format_shortfall(flex5_stock),
                format_pct(flex5_stock["target_shortfall_ever_pct"]),
                format_pct(spend_delivered(flex5_stock)),
                format_pct(flex5_stock["target_shortfall_integrated_loss_pct_target"]),
                format_x(flex5_stock["final_median_multiple"]),
            ],
            [
                "5% 60/40",
                "5% target / 2.5% floor, flexible",
                "40%",
                format_ruin(flex5_6040, n_paths),
                format_shortfall(flex5_6040),
                format_pct(flex5_6040["target_shortfall_ever_pct"]),
                format_pct(spend_delivered(flex5_6040)),
                format_pct(flex5_6040["target_shortfall_integrated_loss_pct_target"]),
                format_x(flex5_6040["final_median_multiple"]),
            ],
        ],
    )

    ref_headers = list(results["spending_reference_rows"][0].keys())
    reference = table(ref_headers, spend_table_rows(results))

    cash = table(
        ["Cash buffer", "Ruin", "Years wealth below start", "Floor breach", "Real final p10", "Real final median"],
        [
            [
                f"{int(row['buffer_years'])} years",
                format_pct(row["ruin_pct"]),
                format_shortfall(row),
                format_pct(row["floor_breach_pct"]),
                format_x(row["final_p10_multiple"]),
                format_x(row["final_median_multiple"]),
            ]
            for row in results["cash_rows"]
        ],
    )
    history = table(
        ["History window", "Ruin", "Years wealth below start", "Floor breach", "Real final p10", "Real final median"],
        [
            [
                f"{row['history_years']} years ({row['start_year']}-{row['end_year']})",
                format_pct(row["ruin_pct"]),
                format_shortfall(row),
                format_pct(row["floor_breach_pct"]),
                format_x(row["final_p10_multiple"]),
                format_x(row["final_median_multiple"]),
            ]
            for row in results["history_rows"]
        ],
    )
    withdrawal = table(
        ["Target spending rate", "Floor", "Ruin", "Years wealth below start", "Floor breach", "Real final p10", "Real final median"],
        [
            [
                f"{row['spending_cap_pct']*100:.0f}%",
                f"{row['floor_spending_pct_initial']*100:.1f}%",
                format_pct(row["ruin_pct"]),
                format_shortfall(row),
                format_pct(row["floor_breach_pct"]),
                format_x(row["final_p10_multiple"]),
                format_x(row["final_median_multiple"]),
            ]
            for row in results["safe_withdrawal_rows"]
        ],
    )
    withdrawal_depth = table(
        [
            "Target spending rate",
            "Years wealth below start",
            "Ever miss target",
            "Median shortfall years if any",
            "Avg shortfall-year spending",
            "Avg target gap",
            "Integrated target loss",
        ],
        [
            [
                f"{row['spending_cap_pct']*100:.0f}%",
                format_shortfall(row),
                format_pct(row["target_shortfall_ever_pct"]),
                format_int(row["target_shortfall_median_years_if_any"]),
                format_pct(row["target_shortfall_avg_spend_pct_initial"] * 100),
                format_pct(row["target_shortfall_avg_depth_pct_target"] * 100),
                format_pct(row["target_shortfall_integrated_loss_pct_target"]),
            ]
            for row in results["safe_withdrawal_rows"]
        ],
    )
    blocks = table(
        ["Block size", "Ruin", "Years wealth below start", "Floor breach", "Real final p10", "Real final median"],
        [
            [
                f"{int(row['block_size_years'])} year{'s' if row['block_size_years'] != 1 else ''}",
                format_pct(row["ruin_pct"]),
                format_shortfall(row),
                format_pct(row["floor_breach_pct"]),
                format_x(row["final_p10_multiple"]),
                format_x(row["final_median_multiple"]),
            ]
            for row in results["block_size_rows"]
        ],
    )

    def bond_table(rows: list[dict]) -> str:
        return table(
            ["Bond allocation", "Ruin", "Years wealth below start", "Floor breach", "Real final p10", "Real final median"],
            [
                [
                    f"{row['bond_pct']*100:.0f}% bonds",
                    format_ruin(row, n_paths),
                    format_shortfall(row),
                    (
                        f"{format_pct(row['floor_breach_pct'])} ({row['floor_breach_path_years']} path-years)"
                        if row["floor_breach_path_years"] < 10
                        else format_pct(row["floor_breach_pct"])
                    ),
                    format_x(row["final_p10_multiple"]),
                    format_x(row["final_median_multiple"]),
                ]
                for row in rows
            ],
        )

    bond_depth = table(
        [
            "Bond allocation",
            "Years wealth below start",
            "Ever miss target",
            "Median shortfall years if any",
            "Avg shortfall-year spending",
            "Avg target gap",
            "Integrated target loss",
        ],
        [
            [
                f"{row['bond_pct']*100:.0f}% bonds",
                format_shortfall(row),
                format_pct(row["target_shortfall_ever_pct"]),
                format_int(row["target_shortfall_median_years_if_any"]),
                format_pct(row["target_shortfall_avg_spend_pct_initial"] * 100),
                format_pct(row["target_shortfall_avg_depth_pct_target"] * 100),
                format_pct(row["target_shortfall_integrated_loss_pct_target"]),
            ]
            for row in results["bond_rows"]
        ],
    )
    bonds5_0 = results["bond_rows"][0]
    bonds5_40 = next(row for row in results["bond_rows"] if abs(row["bond_pct"] - 0.4) < 1e-9)
    bonds5_60 = results["bond_rows"][-1]

    followup = results.get("followup")
    if followup:
        followup_n = int(followup.get("n_paths", n_paths))
        robustness_table = table(
            [
                "Experiment",
                "Ruin",
                "Years wealth below start",
                "Spend delivered",
                "Real final p10",
                "Real final median",
            ],
            [
                [
                    row["label"],
                    format_ruin(row, int(row.get("n_paths", followup_n))),
                    format_shortfall(row),
                    format_pct(spend_delivered(row)),
                    format_x(row["final_p10_multiple"]),
                    format_x(row["final_median_multiple"]),
                ]
                for row in followup["rows"]
            ],
        )
        erp_grid = followup.get("erp_grid") or []
        erp_levels = sorted({row.get("erp_kept", 1.0) for row in erp_grid}, reverse=True)
        bond_levels = sorted({row.get("bond_pct", 0.0) for row in erp_grid})
        erp_lookup = {
            (round(row["erp_kept"], 4), round(row["bond_pct"], 4)): row for row in erp_grid
        }
        erp_table = ""
        if erp_grid:
            erp_headers = ["Kept mean ERP"] + [f"{pct:.0%} bonds" for pct in bond_levels]
            erp_body = []
            for haircut in erp_levels:
                cells = [f"{haircut:.0%}"]
                for bond_pct in bond_levels:
                    cell = erp_lookup.get((round(haircut, 4), round(bond_pct, 4)))
                    if cell is None:
                        cells.append("—")
                    else:
                        cells.append(
                            f"{format_pct(spend_delivered(cell))} spend / "
                            f"{format_x(cell['final_median_multiple'])} / "
                            f"{format_ruin(cell, int(cell.get('n_paths', followup_n)))}"
                        )
                erp_body.append(cells)
            erp_table = table(erp_headers, erp_body)
        floor_rows = followup.get("floor_rows") or []
        floor_table = ""
        if floor_rows:
            floor_table = table(
                [
                    "Policy",
                    "Initial safe share",
                    "Ruin",
                    "Years wealth below start",
                    "Spend delivered",
                    "Real final median",
                ],
                [
                    [
                        row["label"],
                        f"{row['bond_pct']:.0%}",
                        format_ruin(row, int(row.get("n_paths", followup_n))),
                        format_shortfall(row),
                        format_pct(spend_delivered(row)),
                        format_x(row["final_median_multiple"]),
                    ]
                    for row in floor_rows
                ],
            )
        by_label = {row["label"]: row for row in followup["rows"]}
        prorata = by_label.get("4% pro-rata 60/40")
        crash = by_label.get("4% crash-sell-bonds 60/40")
        sleeve = by_label.get("50% zero-real safe sleeve (rebalanced)")
        boy = by_label.get("4% stock-only beginning-of-year")
        notes = []
        hist_0 = erp_lookup.get((1.0, 0.0)) if erp_grid else None
        hist_40 = erp_lookup.get((1.0, 0.4)) if erp_grid else None
        if erp_grid:
            half = erp_lookup.get((0.5, 0.0))
            half_40 = erp_lookup.get((0.5, 0.4))
            if half and half_40:
                if hist_0 and hist_40:
                    notes.append(
                        f"Once the mean US premium is cut in half, 40% Treasuries beat stock-only on both ruin probability and spending: "
                        f"{format_ruin_pct(half_40)} versus {format_ruin_pct(half)} ruin, and "
                        f"{format_pct(spend_delivered(half_40))} versus {format_pct(spend_delivered(half))} of target spending delivered. "
                        f"At the historical premium the ranking is the reverse on wealth ({format_x(hist_0['final_median_multiple'])} versus {format_x(hist_40['final_median_multiple'])}) while spending stays close."
                    )
                else:
                    notes.append(
                        f"At a 50% mean premium, 40% Treasuries have {format_ruin_pct(half_40)} ruin versus {format_ruin_pct(half)} for stock-only."
                    )
        floor10 = next((row for row in floor_rows if row.get("floor_years") == 10), None)
        if floor10 and hist_0 and hist_40:
            notes.insert(
                0,
                f"The 10-year run-down floor bucket is the most actionable sleeve: {format_ruin_pct(floor10)} ruin, "
                f"{format_pct(spend_delivered(floor10))} of target spending delivered, and "
                f"{format_x(floor10['final_median_multiple'])} median wealth — versus "
                f"{format_x(hist_40['final_median_multiple'])} for a permanent 40% Treasury allocation.",
            )
        elif floor10:
            notes.insert(
                0,
                f"The 10-year run-down floor bucket keeps {format_pct(spend_delivered(floor10))} of target spending "
                f"and {format_x(floor10['final_median_multiple'])} median wealth.",
            )
        if crash and prorata:
            notes.append(
                "Under annual rebalancing and frictionless trading, withdrawal source does not explain the 60/40 result. "
                "Crash-sell-bonds and pro-rata restore the same bond share after the withdrawal, so the rows are identical."
            )
        if sleeve:
            notes.append(
                f"The rebalanced 50% zero-real sleeve spends {format_shortfall(sleeve)} of years below start and ends at "
                f"{format_x(sleeve['final_median_multiple'])}. That is the cost of keeping half the portfolio in a 0% real asset, not of matching a floor liability."
            )
        if boy:
            notes.append(
                f"Beginning-of-year withdrawals are slightly harsher ({format_pct(boy['ruin_pct'])} ruin, "
                f"{format_shortfall(boy)} years below start, {format_x(boy['final_median_multiple'])} median) and do not overturn stock-only."
            )
        erp_fig = ""
        if erp_grid:
            erp_fig = """
          <figure class="figure">
            <img src="assets/bond_report_erp_grid.png" alt="Spend delivered and median wealth versus bond allocation under several mean equity premiums.">
            <figcaption>Mean equity premium is scaled; year-to-year excess-return sequencing is kept. Bond and T-bill series are not haircut.</figcaption>
          </figure>
          <figure class="figure">
            <img src="assets/bond_report_terminal_cdf.png" alt="Terminal wealth CDF by bond allocation at the historical equity premium.">
            <figcaption>At the historical premium, a 40% Treasury sleeve is above stock-only only in the far left tail, then sits below it.</figcaption>
          </figure>
"""
        followup_html = f"""
        <section id="followup">
          <h2>Experiment 6: When The Bond Tradeoff Changes</h2>
          <p>These rows use the same seed family and 5-year blocks as the main tables, with {followup_n:,} paths. Ruin is the <b>probability</b> a simulated path ends at or near zero — the share of the {followup_n:,} histories that go broke, not a count of personal failures.</p>
          {("<h3>Run-down floor bucket</h3><p>A 10-year 0% real floor reserve is the most practical sleeve in this set: it keeps almost all of stock-only spending and most of the upside, while cutting ruin relative to 100% equities. Unlike a permanent 40% Treasury allocation, the bucket is allowed to run down instead of being refilled from stocks.</p>" + floor_table) if floor_table else ""}
          <figure class="figure">
            <img src="assets/bond_report_insurance_frontier.png" alt="Median ending wealth versus probability of ruin for stock-only, permanent Treasury sleeves, and run-down floor buckets.">
            <figcaption>Each point is a 4% flexible plan. Higher is richer; left is safer. Marker size scales with lifetime target spending delivered.</figcaption>
          </figure>
          <h3>Bond allocation under a lower equity premium</h3>
          <p>The equity-premium transform shifts only the <b>mean</b> excess return versus T-bills. Crash years stay crash years. “Worthwhile” still depends on how much wealth a household will trade for a smaller left tail.</p>
          {erp_table}
          {erp_fig}
          <h3>Robustness</h3>
          {robustness_table}
          <p class="callout"><b>Read:</b> {" ".join(notes)}</p>
        </section>
"""
        followup_nav = '<a href="#followup">6. Bond Tradeoff</a>'
        followup_limits = (
            "Experiment 6 now includes a mean-only ERP × allocation grid, a run-down 0% real floor bucket, "
            "beginning-of-year withdrawals, a 40-year horizon, and circular blocks."
        )
        tips_sentence = (
            "A run-down 0% real floor bucket and a mean-only ERP grid are in Experiment 6. "
            "They separate “stocks did well in this sample” from “a permanent Treasury sleeve is the right insurance.”"
        )
    else:
        followup_html = ""
        followup_nav = ""
        followup_limits = ""
        tips_sentence = (
            "A household that values floor certainty or a smaller left tail can still prefer a bond sleeve."
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>The Measurable Tradeoffs Behind The 4% Rule</title>
  <meta name="description" content="A retirement allocation report stress-testing 60/40 and the 4% rule against mandatory and discretionary spending needs.">
  <meta property="og:title" content="The Measurable Tradeoffs Behind The 4% Rule">
  <meta property="og:description" content="Under a flexible target/cap/floor rule, classic 4% 60/40 ruin mostly disappears; extra permanent bonds then buy little extra safety and more years below starting wealth.">
  <meta property="og:type" content="article">
  <meta property="og:image" content="assets/bond_report_objective_tradeoff.png">
  <link rel="icon" href="favicon.svg" type="image/svg+xml">
  <style>
    @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,650;8..60,750&display=swap");
    :root {{
      --paper: #fbfaf5; --panel: #ffffff; --ink: #16212c; --muted: #697381;
      --line: #d8dedf; --soft: #f0eee5; --red: #b23a48; --green: #2f7d64;
      --blue: #274c77; --gold: #d2872c; --shadow: 0 24px 70px rgba(22, 33, 44, 0.12); --radius: 6px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background:
        linear-gradient(90deg, rgba(22, 33, 44, 0.035) 1px, transparent 1px) 0 0 / 44px 44px,
        linear-gradient(180deg, #f7f3e9 0%, var(--paper) 36%, #f4f0e6 100%);
      color: var(--ink);
      font-family: "IBM Plex Sans", ui-sans-serif, sans-serif;
      line-height: 1.58;
    }}
    a {{ color: var(--blue); text-decoration-color: rgba(39, 76, 119, 0.35); text-underline-offset: 3px; }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 0 24px 72px; }}
    .site-nav {{ border-bottom: 1px solid var(--line); background: rgba(251, 250, 245, 0.92); backdrop-filter: blur(12px); }}
    .site-nav-inner {{ max-width: 1180px; margin: 0 auto; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .site-nav-brand {{ color: var(--ink); font-weight: 600; text-decoration: none; letter-spacing: 0.02em; }}
    .site-nav-links {{ display: flex; flex-wrap: wrap; gap: 8px 18px; }}
    .site-nav a {{ color: #344250; text-decoration: none; font-size: 0.92rem; }}
    .site-nav a:hover, .site-nav a[aria-current="page"] {{ color: var(--red); }}
    .site-nav a[aria-current="page"] {{ font-weight: 600; }}
    .hero {{ min-height: 78vh; display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr); gap: 52px; align-items: center; padding: 56px 0 42px; }}
    .eyebrow {{ color: var(--red); font-size: 0.76rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin: 0 0 18px; }}
    h1, h2, h3 {{ font-family: "Source Serif 4", Georgia, serif; line-height: 1.03; letter-spacing: 0; margin: 0; }}
    h1 {{ font-size: clamp(3rem, 8vw, 7.4rem); max-width: 900px; }}
    h2 {{ font-size: clamp(2.1rem, 4vw, 4.5rem); margin-bottom: 20px; }}
    h3 {{ font-size: clamp(1.45rem, 2vw, 2.2rem); margin: 20px 0 10px; }}
    .dek {{ font-size: clamp(1.15rem, 2vw, 1.55rem); max-width: 760px; color: #33414f; margin: 24px 0 0; }}
    .hero-panel {{ background: rgba(255, 255, 255, 0.78); border: 1px solid rgba(22, 33, 44, 0.12); border-radius: var(--radius); box-shadow: var(--shadow); padding: 22px; backdrop-filter: blur(18px); }}
    .hero-panel img {{ width: 100%; display: block; border-radius: var(--radius); border: 1px solid var(--line); background: white; }}
    .caption {{ color: var(--muted); font-size: 0.92rem; margin: 12px 0 0; }}
    .metric-strip {{ display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); margin: 20px 0 44px; background: rgba(255, 255, 255, 0.55); }}
    .metric {{ padding: 22px 18px; border-right: 1px solid var(--line); }}
    .metric:last-child {{ border-right: 0; }}
    .metric strong {{ display: block; font-family: "Source Serif 4", Georgia, serif; font-size: clamp(1.7rem, 3vw, 3rem); line-height: 1; margin-bottom: 8px; }}
    .metric span {{ color: var(--muted); font-size: 0.92rem; }}
    .layout {{ display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 44px; align-items: start; }}
    nav.toc {{ position: sticky; top: 18px; padding: 18px 0; border-top: 2px solid var(--ink); }}
    nav.toc a {{ display: block; padding: 8px 0; color: #344250; text-decoration: none; font-size: 0.95rem; border-bottom: 1px solid rgba(22, 33, 44, 0.1); }}
    nav.toc a:hover {{ color: var(--red); }}
    main {{ min-width: 0; }}
    section {{ padding: 58px 0; border-top: 1px solid var(--line); }}
    section:first-child {{ border-top: 0; padding-top: 0; }}
    .lead {{ font-size: 1.17rem; color: #2e3c49; max-width: 860px; }}
    .callout {{ background: #fff8ea; border-left: 5px solid var(--gold); padding: 18px 20px; border-radius: var(--radius); margin: 24px 0; }}
    .formula {{ background: #16212c; color: #f7f3e9; border-radius: var(--radius); padding: 20px; overflow-x: auto; font: 500 0.96rem/1.7 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .figure {{ background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: 0 14px 40px rgba(22, 33, 44, 0.08); padding: 14px; margin: 28px 0; }}
    .figure img {{ width: 100%; display: block; border-radius: 4px; }}
    .figure figcaption {{ color: var(--muted); font-size: 0.95rem; padding: 10px 4px 2px; }}
    .grid-two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 22px 0; }}
    .story-grid {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 22px; margin: 26px 0; }}
    .note-card {{ background: rgba(255, 255, 255, 0.72); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px; }}
    .note-card .label {{ color: var(--muted); display: block; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 8px; text-transform: uppercase; }}
    .table-wrap {{ overflow-x: auto; margin: 22px 0 30px; border: 1px solid var(--line); border-radius: var(--radius); background: white; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 720px; font-size: 0.95rem; }}
    th, td {{ padding: 11px 13px; text-align: right; border-bottom: 1px solid #e9edee; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; font-weight: 600; }}
    th {{ background: #f0f3ef; color: #32404d; font-size: 0.79rem; letter-spacing: 0.06em; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    blockquote {{ margin: 34px 0; padding: 4px 0 4px 24px; border-left: 5px solid var(--red); font-family: "Source Serif 4", Georgia, serif; font-size: clamp(1.35rem, 2.2vw, 2rem); line-height: 1.22; }}
    ul, ol {{ padding-left: 22px; }}
    li {{ margin: 7px 0; }}
    footer {{ color: var(--muted); border-top: 1px solid var(--line); padding-top: 28px; margin-top: 40px; font-size: 0.95rem; }}
    @media (max-width: 920px) {{
      .hero, .layout, .grid-two, .story-grid {{ grid-template-columns: 1fr; }}
      .hero {{ min-height: auto; }}
      nav.toc {{ position: static; display: grid; grid-template-columns: repeat(2, 1fr); gap: 0 18px; }}
      .metric-strip {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 560px) {{
      .page {{ padding: 0 16px 48px; }}
      .metric-strip {{ grid-template-columns: 1fr; }}
      .metric {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .metric:last-child {{ border-bottom: 0; }}
      nav.toc {{ grid-template-columns: 1fr; }}
      .site-nav-inner {{ padding: 12px 16px; flex-direction: column; align-items: flex-start; }}
    }}
  </style>
</head>
<body>
  <nav class="site-nav" aria-label="Site">
    <div class="site-nav-inner">
      <a class="site-nav-brand" href="index.html">Retirement Planning</a>
      <div class="site-nav-links">
        <a href="index.html" aria-current="page">Allocation report</a>
        <a href="credit-line.html">Credit-line report</a>
        <a href="https://github.com/actions-im/retirement_planning">GitHub</a>
      </div>
    </div>
  </nav>
  <div class="page">
    <header class="hero">
      <div>
        <p class="eyebrow">Retirement Planning Report · May 14, 2026</p>
        <h1>The Measurable Tradeoffs Behind The 4% Rule</h1>
        <p class="dek">At a flexible 4% target, a permanent 40% 10-year Treasury sleeve mostly insures the extreme left tail. Average spending barely changes; the long-term wealth cushion is cut roughly in half.</p>
      </div>
      <aside class="hero-panel">
        <img src="assets/bond_report_objective_tradeoff.png" alt="Objective tradeoff chart showing higher bond allocations moving toward more years below starting wealth and lower median wealth.">
        <p class="caption">When the target rate equals the cap rate, “target shortfall” is the share of years real wealth is below start. Ruin and floor-breach reductions are in the tables.</p>
      </aside>
    </header>

    <div class="metric-strip" aria-label="Key findings">
      <div class="metric">
        <strong>{format_ruin_pct(fixed4)}</strong>
        <span>probability of ruin, classic fixed-real 4% 60/40</span>
      </div>
      <div class="metric">
        <strong>{format_ruin_pct(flex4_stock)}</strong>
        <span>probability of ruin, flexible 4% stock-only</span>
      </div>
      <div class="metric">
        <strong>{format_shortfall(flex4_6040)}</strong>
        <span>years real wealth below start for flexible 4% 60/40</span>
      </div>
      <div class="metric">
        <strong>{format_x(flex4_stock["final_median_multiple"])}</strong>
        <span>real median ending wealth for flexible 4% stock-only</span>
      </div>
    </div>

    <div class="layout">
      <nav class="toc" aria-label="Report sections">
        <a href="#summary">Summary</a>
        <a href="#setup">Scenario</a>
        <a href="#cash">1. Cash Buffer</a>
        <a href="#history">2. History Window</a>
        <a href="#withdrawal">3. Withdrawal Search</a>
        <a href="#block-size">4. Block Size</a>
        <a href="#bonds">5. Bonds</a>
        {followup_nav}
        <a href="#interpretation">Interpretation</a>
        <a href="#limits">Limitations</a>
      </nav>

      <main>
        <section id="summary">
          <h2>Executive Summary</h2>
          <p class="lead">Imagine someone retiring with a portfolio of a few million dollars. Some spending is mandatory: food, basic housing, utilities, insurance, taxes, and health care. Other spending is discretionary: travel, nicer housing, gifts, upgrades, and the parts of life that make retirement feel abundant.</p>
          <p>That budget naturally creates two retirement objectives. The <b>target</b> is the preferred lifestyle. The <b>floor</b> is the minimum acceptable lifestyle. This report replaces labels like conservative and aggressive with measurable tradeoffs: ruin, years below starting wealth, floor-breach risk, shortfall depth, and real ending wealth.</p>
          <p>The useful comparison is at one spending rate. Classic fixed-real 4% 60/40 has a {format_ruin_pct(fixed4)} probability of ruin in this model — numerically close to the historical failure rate associated with the original Trinity results, though the methodology differs (block bootstrap, 10-year Treasuries, and annual year-end withdrawals versus Trinity’s historical payout periods, corporate bonds, and monthly inflation-adjusted withdrawals). The same 4% target with a flexible cap/floor rule cuts the stock-only probability of ruin to {format_ruin_pct(flex4_stock)}. Adding a permanent 40% 10-year Treasury sleeve then cuts that probability to {format_ruin_pct(flex4_6040)}. Years below starting wealth rise from {format_shortfall(flex4_stock)} to {format_shortfall(flex4_6040)}, but lifetime target spending delivered only moves from {format_pct(spend_delivered(flex4_stock))} to {format_pct(spend_delivered(flex4_6040))}. Real median ending wealth falls from {format_x(flex4_stock["final_median_multiple"])} to {format_x(flex4_6040["final_median_multiple"])}.</p>
          <div class="callout">
            <b>What the rule is doing:</b> when the target rate equals the cap rate, a target-shortfall year is a year the real portfolio is below its start. Stock-only “maintains lifestyle” here mostly because US equities in {start_year}–{end_year} spend fewer years underwater than a 10-year Treasury sleeve. A household that weights a funded floor or a smaller left tail more than median terminal wealth can still prefer bonds.
          </div>
          {benchmark}
          <p>Flexible 4% stock-only still misses the target on {format_pct(flex4_stock["target_shortfall_ever_pct"])} of paths. The binary “years below start” gap is large because 60/40 spends more years slightly underwater. Integrated target loss, the depth-weighted companion, is {format_pct(flex4_stock["target_shortfall_integrated_loss_pct_target"])} versus {format_pct(flex4_6040["target_shortfall_integrated_loss_pct_target"])} — that is {format_pct(spend_delivered(flex4_stock))} versus {format_pct(spend_delivered(flex4_6040))} of desired lifetime spending delivered. The main economic cost of 60/40 in this sample is the longevity cushion, not current lifestyle. Ruin is a probability: the share of the {n_paths:,} simulated histories that end at or near zero, shown with a 95% Wilson interval.</p>
        </section>

        <section id="setup">
          <h2>Scenario And Spending Rule</h2>
          <div class="story-grid">
            <div class="note-card">
              <span class="label">Mandatory spending</span>
              <p>Food, basic housing, utilities, insurance, taxes, health care, and other expenses that cannot be cut without changing the retiree's minimum acceptable life.</p>
            </div>
            <div class="note-card">
              <span class="label">Discretionary spending</span>
              <p>Vacations, nicer housing, upgrades, gifts, and other spending that can be reduced when markets are bad without immediately creating ruin.</p>
            </div>
          </div>
          <p>The report converts that budget into percentage rules. Experiments 1–4 use a 5% target and 2.5% floor. Experiment 5 and the summary table also show the 4% target that matches the traditional withdrawal-rate baseline.</p>
          {reference}
          <p>For the 5% baseline, the withdrawal rule is:</p>
          <pre class="formula">target = 5.0% of initial portfolio, inflation adjusted
floor = 2.5% of initial portfolio, inflation adjusted
cap = 5.0% of current portfolio after that year's market return

withdrawal = min(current portfolio, max(floor, min(target, cap)))</pre>
          <p>The floor can override the cap. If the remaining portfolio cannot fund the floor, the withdrawal is limited to the remaining portfolio, and that year is counted as a floor breach.</p>
          <div class="note-card">
            <h3>Metric definitions</h3>
            <ul>
              <li><b>Probability of ruin</b> is the share of simulated 30-year histories whose terminal wealth is at or below zero. Intervals are 95% Wilson intervals on that probability.</li>
              <li><b>Target shortfall / years wealth below start</b> is the percent of all simulated path-years in which withdrawal is below the real target. When the target rate equals the cap rate, that event is exactly “real wealth after the year’s return is below the starting portfolio.”</li>
              <li><b>Ever miss target</b> is the percent of paths with at least one shortfall year.</li>
              <li><b>Integrated target loss</b> is the mean, over all path-years, of the shortfall as a fraction of target. It weights depth, not just a binary miss.</li>
              <li><b>Floor breach</b> is the percent of path-years in which the portfolio cannot fund the real floor.</li>
              <li><b>Real final p10 / median</b> is ending wealth in starting-year purchasing power, divided by starting wealth.</li>
            </ul>
          </div>
          <figure class="figure">
            <img src="assets/bond_report_spending_cap.png" alt="Spending rule chart showing floor breach, floor funded, target shortfall, and target funded regimes.">
            <figcaption>The left threshold, {settings["floor_spending_pct_initial"]}x, means one year of floor spending remains.</figcaption>
          </figure>
          <h3>Model</h3>
          <p>The engine is a historical block-bootstrap Monte Carlo. It generates {n_paths:,} random {settings["years"]}-year retirement paths by sampling contiguous {settings["block_size_years"]}-year historical blocks with replacement, not independent normal draws. Seed {settings["seed"]}.</p>
          <figure class="figure">
            <img src="assets/bond_report_bootstrap_method.png" alt="Diagram explaining historical block bootstrap sampling.">
            <figcaption>The model samples actual {settings["block_size_years"]}-year historical regimes and chains them into {settings["years"]}-year retirement paths.</figcaption>
          </figure>
          <p>Each sampled calendar year keeps its observed stock return, bond return, T-bill return, and CPI inflation rate. That preserves local sequences such as crashes followed by recoveries, inflation clusters, and stock/bond/inflation co-movement. Interior years appear in more overlapping blocks than the sample endpoints.</p>
          <p>The main lookback window is {start_year}–{end_year} from Aswath Damodaran’s annual return dataset, using the S&amp;P 500 total-return series with dividends reinvested. Inflation comes from <a href="https://fred.stlouisfed.org/series/CPIAUCNS">FRED CPI-U</a>, calculated December-to-December. Cash is a T-bill-like reserve sampled from the same year, not an assumed 0% real asset.</p>
          <div class="note-card">
            <h3>Annual chronology</h3>
            <ol>
              <li>Sample the next historical year from the current {settings["block_size_years"]}-year block.</li>
              <li>Apply sampled stock, bond, and T-bill returns, then deflate by that year's CPI inflation.</li>
              <li>Compute the spending cap from the post-return real portfolio value.</li>
              <li>Withdraw at year-end under the target/cap/floor rule.</li>
              <li>Replenish cash when the strategy calls for it, rebalance bonds annually, and record outcomes.</li>
            </ol>
          </div>
          <p>Because the model deflates returns each year, all ending-wealth multiples are real, CPI-adjusted multiples of the starting portfolio.</p>
        </section>

        <section id="cash">
          <h2>Experiment 1: Cash Buffer</h2>
          <p>This experiment varies the cash buffer from 0 to 5 years of target spending on the 5% baseline, with the remaining portfolio in the Damodaran S&amp;P 500 total-return series.</p>
          {cash}
          <figure class="figure">
            <img src="assets/bond_report_cash_buffer.png" alt="Cash buffer tradeoff chart.">
            <figcaption>The plot separates years below starting wealth from tiny ruin/floor probabilities so both sides of the tradeoff are visible.</figcaption>
          </figure>
          <p class="callout"><b>Conclusion:</b> T-bill cash slightly lowers ruin and floor-breach risk, but it is expensive. A 5-year buffer lowers ruin from {format_pct(cash0["ruin_pct"])} to {format_pct(cash5["ruin_pct"])}, while years below start rise from {format_shortfall(cash0)} to {format_shortfall(cash5)} and median ending wealth falls from {format_x(cash0["final_median_multiple"])} to {format_x(cash5["final_median_multiple"])}. The rest of the report uses zero cash as the baseline.</p>
        </section>

        <section id="history">
          <h2>Experiment 2: Historical Window Sensitivity</h2>
          <p>This experiment keeps the 5% target, 2.5% floor, and no cash buffer. It changes only the lookback window.</p>
          {history}
          <figure class="figure">
            <img src="assets/bond_report_history_sensitivity.png" alt="Historical window sensitivity chart.">
            <figcaption>The Depression-inclusive 98-year case is a useful stress test, but it pulls in a different economic regime.</figcaption>
          </figure>
          <p class="callout"><b>Conclusion:</b> the 75-year window remains the main baseline because it keeps modern severe sequences, including the 1970s inflation problem, while excluding the Great Depression and World War II regime from the central case. The 98-year result should still be read as a legitimate stress test, not dismissed. All of these windows are still US large-cap history.</p>
        </section>

        <section id="withdrawal">
          <h2>Experiment 3: Safe Withdrawal Search</h2>
          <p>Zero cash, zero bonds. Target spending rates from 4% to 10% of the initial portfolio. The floor is always 50% of the target.</p>
          {withdrawal}
          <figure class="figure">
            <img src="assets/bond_report_safe_withdrawal_search.png" alt="Safe withdrawal search chart.">
            <figcaption>Zone labels are judgmental summaries based on rising ruin, floor breach, years below start, and declining real p10 wealth.</figcaption>
          </figure>
          {withdrawal_depth}
          <p class="callout"><b>Conclusion:</b> the withdrawal rate is a risk choice, not a magic number. Both 4% and 5% have low ruin here, but 4% is much lower: {format_pct(flex4_stock["ruin_pct"])} versus {format_pct(flex5_stock["ruin_pct"])}. Even at 4%, {format_pct(flex4_stock["target_shortfall_ever_pct"])} of paths go below starting wealth at least once. Above 6%, the plan starts depending too heavily on favorable sequences.</p>
        </section>

        <section id="block-size">
          <h2>Experiment 4: Block Size Sensitivity</h2>
          <p>5% target, 2.5% floor, zero cash, zero bonds, 75-year Damodaran history. Only the bootstrap block size changes.</p>
          {blocks}
          <p class="callout"><b>Conclusion:</b> block size is a modeling assumption, not something to optimize until it best fits the past. Very short blocks break historical sequencing; very long blocks replay old macro regimes too literally. A 5-year block is the working compromise.</p>
        </section>

        <section id="bonds">
          <h2>Experiment 5: Bonds And 60/40</h2>
          <p>Zero cash. The bond sleeve is Damodaran 10-year Treasury total returns, rebalanced annually, with crash withdrawals going cash → bonds → equity. That is a permanent duration sleeve, not TIPS, not a ladder, and not pro-rata spending from a balanced fund.</p>
          <h3>4% Target / 2% Floor</h3>
          {bond_table(results["bond_rows_4pct"])}
          <h3>5% Target / 2.5% Floor</h3>
          {bond_table(results["bond_rows"])}
          <figure class="figure">
            <img src="assets/bond_report_bond_allocation.png" alt="Bond allocation tradeoff chart.">
            <figcaption>This plot shows the lifestyle/wealth cost side and the small ruin/floor-risk benefit of 10-year Treasuries on separate scales.</figcaption>
          </figure>
          <div class="grid-two">
            <div class="note-card">
              <h3>0% bonds</h3>
              <p>{format_shortfall(bonds5_0)} of years below starting wealth and {format_x(bonds5_0["final_median_multiple"])} median ending wealth.</p>
            </div>
            <div class="note-card">
              <h3>60% bonds</h3>
              <p>{format_shortfall(bonds5_60)} of years below starting wealth and {format_x(bonds5_60["final_median_multiple"])} median ending wealth.</p>
            </div>
          </div>
          {bond_depth}
          <figure class="figure">
            <img src="assets/bond_report_objective_tradeoff_4_5.png" alt="Objective tradeoff chart for 4% and 5% spending targets.">
            <figcaption>Both the 4% and 5% target cases show the same movement: more 10-year Treasuries mean more years below start and lower real median ending wealth.</figcaption>
          </figure>
          <p>Ending wealth is a longevity cushion. A 30-year simulation is a modeling horizon, not a known lifespan. The difference between ending with {format_x(bonds5_0["final_median_multiple"])} and {format_x(bonds5_40["final_median_multiple"])} is the reserve that protects extra years, late-life care, and bad post-year-30 returns. It is also the median of a right-skewed equity terminal-wealth distribution.</p>
          <p class="callout"><b>Conclusion:</b> at a flexible 4% target, a permanent 40% Treasury sleeve mostly insures the extreme left tail. It changes average spending only modestly ({format_pct(spend_delivered(flex4_stock))} vs {format_pct(spend_delivered(flex4_6040))} of target spending delivered) while cutting median ending wealth from {format_x(flex4_stock["final_median_multiple"])} to {format_x(flex4_6040["final_median_multiple"])}. Whether cutting the probability of ruin from {format_ruin_pct(flex4_stock)} to {format_ruin_pct(flex4_6040)} is worth that cushion is a preference, not a theorem.</p>
        </section>
        {followup_html}
        <section id="interpretation">
          <h2>Interpretation</h2>
          <p>The traditional allocation argument says bonds reduce sequence risk. That is true, but incomplete. This spending rule already cuts discretionary spending before the portfolio is exhausted. In this sample the 60/40 lifestyle cost is mostly cosmetic: more years slightly below start, almost the same lifetime spending, and a much smaller terminal reserve.</p>
          <blockquote>At a flexible 4% target, a permanent 40% Treasury sleeve mostly insures the extreme left tail. Average spending barely changes; the long-term wealth cushion is cut roughly in half.</blockquote>
          <p>Beginning-of-year withdrawals are slightly harsher and do not overturn stock-only. That is a scoped result, not a falsification of 60/40. The sample is US large-cap history from {start_year} to {end_year}. Bonds here are 10-year Treasury total returns. A household that values floor certainty, a smaller left tail, or not sitting through a 50% equity drawdown can still prefer a bond sleeve. {tips_sentence}</p>
        </section>

        <section id="limits">
          <h2>Limitations And Next Tests</h2>
          <div class="grid-two">
            <div class="note-card">
              <h3>Important limitations</h3>
              <ul>
                <li>Taxes and fees are ignored.</li>
                <li>Returns are annual, not monthly.</li>
                <li>Bonds are 10-year Treasury total returns, not a real TIPS curve or coupon ladder.</li>
                <li>The ERP grid shifts only the mean US equity premium; it is not an international history.</li>
                <li>Overlapping 5-year blocks overweight interior years.</li>
                <li>No Social Security, pension, mortgage, or health shock modeling.</li>
              </ul>
            </div>
            <div class="note-card">
              <h3>Remaining tests</h3>
              <ul>
                <li>Monthly withdrawals.</li>
                <li>A real TIPS / laddered floor, not a 0% real proxy.</li>
                <li>Developed ex-US / international return history.</li>
                <li>Log-return ERP robustness and non-overlapping blocks.</li>
                <li>Threshold or delayed rebalancing, so “sell bonds in crashes” can persist.</li>
                <li>Social Security or a pension covering part of the floor.</li>
                <li>Taxes and fees.</li>
              </ul>
            </div>
          </div>
          <p>The result should be framed narrowly: under this specific flexible spending rule, with annual returns, sampled historical CPI inflation, and a permanent 10-year Treasury sleeve, extra defensive allocation did not improve the selected “years below start plus median wealth” score. {followup_limits}</p>
          <p>A sibling monthly report, <a href="credit-line.html">Can A Hybrid Credit Line Reduce Forced Selling?</a>, tests a cap-respecting credit line on S&amp;P 500 total-return data from 1988 onward. Those ruin and shortfall rates are not comparable to the annual tables above.</p>
          <footer>Generated from <code>docs/assets/bond_report_results.json</code>, seed {settings["seed"]}, {n_paths:,} paths. Regenerating the HTML does not rerun the Monte Carlo.</footer>
        </section>
      </main>
    </div>
  </div>
</body>
</html>
"""


def write_report(results: dict | None = None) -> Path:
    """Write docs/index.html from JSON."""
    if results is None:
        results = load_results()
    REPORT_PATH.write_text(render(results))
    return REPORT_PATH


def main() -> None:
    path = write_report()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
