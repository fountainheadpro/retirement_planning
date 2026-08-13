"""Generate report charts from saved retirement experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
RESULTS_PATH = ASSET_DIR / "bond_report_results.json"

COLORS = {
    "ink": "#17212b",
    "muted": "#64707d",
    "grid": "#d8dee5",
    "target": "#b23a48",
    "floor": "#2f7d64",
    "ruin": "#68717d",
    "wealth": "#274c77",
    "wealth_light": "#8fb3d9",
    "accent": "#d2872c",
    "accent_light": "#f6d9ad",
    "panel": "#f8faf8",
    "paper": "#ffffff",
}


def load_results() -> dict:
    with RESULTS_PATH.open() as f:
        return json.load(f)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["panel"],
            "axes.edgecolor": "#c7cfd8",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(
        title,
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(0.06, 0.918, subtitle, ha="left", fontsize=10.5, color=COLORS["muted"])


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(ASSET_DIR / filename, dpi=180, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)


def pct_label(value: float) -> str:
    return f"{value:.0f}%"


def annotate_endpoint(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9.5,
        color=color,
        fontweight="bold",
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, alpha=0.8, linewidth=0.8)
    ax.tick_params(length=0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c7cfd8")


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + 0.025,
        xy[1] + height - 0.055,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["ink"],
        fontsize=11.5,
        fontweight="bold",
    )
    ax.text(
        xy[0] + 0.025,
        xy[1] + height - 0.13,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["muted"],
        fontsize=9.6,
        linespacing=1.35,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["muted"],
            "linewidth": 1.5,
            "shrinkA": 5,
            "shrinkB": 5,
        },
    )


def plot_bootstrap_method(results: dict) -> None:
    settings = results["settings"]
    start_year = settings["baseline_start_year"]
    end_year = settings["baseline_end_year"]
    block_size = settings["block_size_years"]
    n_paths = settings["n_paths"]

    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ax.axis("off")
    title_block(
        fig,
        "Monte Carlo Engine: Historical Block Bootstrap",
        "The simulation is random, but each draw is an actual multi-year market regime sampled with replacement.",
    )

    add_box(
        ax,
        (0.035, 0.60),
        0.25,
        0.34,
        f"1. Historical records\n{start_year}-{end_year}",
        "Each calendar year keeps its\nstock return, bond return,\nT-bill return, and CPI inflation.",
        "#eef4f7",
        "#9ab0bf",
    )
    add_box(
        ax,
        (0.375, 0.60),
        0.25,
        0.34,
        f"2. Sample {block_size}-year blocks",
        "Blocks are sampled with replacement.\nEach block keeps local sequence:\ncrash, recovery, inflation,\nand rates stay together.",
        "#fff0d6",
        "#d8a24b",
    )
    add_box(
        ax,
        (0.715, 0.60),
        0.25,
        0.34,
        "3. Chain six blocks",
        f"30 years = six {block_size}-year blocks.\nAppend blocks to build one path.\nRepeat {n_paths:,} times.",
        "#e7f1ee",
        "#7aa892",
    )
    add_arrow(ax, (0.29, 0.77), (0.37, 0.77))
    add_arrow(ax, (0.63, 0.77), (0.71, 0.77))

    year_rows = [
        ("1973", "stocks", "bonds", "T-bills", "CPI"),
        ("1974", "stocks", "bonds", "T-bills", "CPI"),
        ("1975", "stocks", "bonds", "T-bills", "CPI"),
        ("...", "...", "...", "..."),
    ]
    for idx, row in enumerate(year_rows):
        y = 0.53 - idx * 0.045
        ax.text(
            0.055,
            y,
            "  ".join(row),
            transform=ax.transAxes,
            family="monospace",
            fontsize=8.6,
            color=COLORS["muted"],
        )

    block_labels = ["1973-1977", "2000-2004", "2021-2025"]
    for idx, label in enumerate(block_labels):
        x = 0.395 + idx * 0.075
        chip = patches.FancyBboxPatch(
            (x, 0.47),
            0.065,
            0.055,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            transform=ax.transAxes,
            facecolor="#f7dca8",
            edgecolor="#d8a24b",
            linewidth=0.8,
        )
        ax.add_patch(chip)
        ax.text(
            x + 0.0325,
            0.497,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            color="#8d5a12",
            fontweight="bold",
        )

    path_x = np.linspace(0.735, 0.945, 7)
    path_y = [0.48, 0.53, 0.50, 0.57, 0.54, 0.61, 0.58]
    ax.plot(path_x, path_y, transform=ax.transAxes, color=COLORS["floor"], linewidth=2.4)
    ax.scatter(path_x, path_y, transform=ax.transAxes, s=18, color=COLORS["floor"], zorder=3)
    ax.text(
        0.84,
        0.43,
        "one simulated retirement path",
        transform=ax.transAxes,
        ha="center",
        color=COLORS["muted"],
        fontsize=9.2,
    )

    random_walk_panel = patches.FancyBboxPatch(
        (0.035, 0.08),
        0.93,
        0.15,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor="#f2f4f7",
        edgecolor="#c7cfd8",
        linewidth=1.1,
    )
    ax.add_patch(random_walk_panel)
    ax.text(
        0.06,
        0.18,
        "Why not independent annual sampling?",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=COLORS["ink"],
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.06,
        0.12,
        "Independent annual draws can match average return and volatility, but they break sequence risk and macro co-movement.\n"
        "The block bootstrap keeps stocks, bonds, T-bills, and inflation tied to the same historical regime.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=COLORS["muted"],
        fontsize=9.4,
        linespacing=1.35,
    )

    fig.subplots_adjust(top=0.82, left=0.04, right=0.98, bottom=0.06)
    save(fig, "bond_report_bootstrap_method.png")


def plot_spending_rule(results: dict) -> None:
    settings = results["settings"]
    target_pct = settings["target_spending_pct_initial"]
    floor_pct = settings["floor_spending_pct_initial"]
    floor_threshold = floor_pct
    floor_to_cap_threshold = floor_pct / target_pct
    target_threshold = 1.0

    x = np.linspace(0, 1.65, 500)
    cap_spend = x * target_pct
    desired = np.minimum(np.maximum(np.minimum(target_pct, cap_spend), floor_pct), x)

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    title_block(
        fig,
        "Spending Rule Example: 5% Target / 2.5% Floor",
        "The floor protects mandatory spending; the cap cuts discretionary spending before ruin.",
    )

    regimes = [
        (0.0, floor_threshold, "#f8d7da", "floor breach", "assets cannot fund minimum"),
        (floor_threshold, floor_to_cap_threshold, "#e7f1ee", "floor funded", "mandatory spending only"),
        (floor_to_cap_threshold, target_threshold, "#fff0d6", "target shortfall", "discretionary lifestyle cut"),
        (target_threshold, 1.65, "#dff0e6", "target funded", "full preferred lifestyle"),
    ]
    for start, end, color, _, _ in regimes:
        ax.axvspan(start, end, color=color, alpha=0.92, zorder=0)

    ax.plot(
        x,
        desired * 100,
        color=COLORS["target"],
        linewidth=3.4,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        x,
        cap_spend * 100,
        color=COLORS["muted"],
        linewidth=1.4,
        linestyle=(0, (2, 3)),
        alpha=0.58,
        zorder=2,
    )

    ax.axhline(target_pct * 100, color=COLORS["target"], linestyle=(0, (5, 3)), linewidth=1.7)
    ax.axhline(floor_pct * 100, color=COLORS["floor"], linestyle=(0, (5, 3)), linewidth=1.7)
    ax.axvline(floor_to_cap_threshold, color=COLORS["muted"], linestyle=":", linewidth=1.4)
    ax.axvline(target_threshold, color=COLORS["muted"], linestyle=":", linewidth=1.4)

    ax.scatter(
        [floor_to_cap_threshold, target_threshold],
        [floor_pct * 100, target_pct * 100],
        s=54,
        color=COLORS["paper"],
        edgecolor=COLORS["target"],
        linewidth=1.8,
        zorder=5,
    )

    label_y = 5.55
    ax.text(
        0.20,
        label_y,
        "floor funded\nmandatory only",
        ha="center",
        va="center",
        color=COLORS["floor"],
        fontweight="bold",
        fontsize=10,
    )
    ax.text(
        0.75,
        label_y,
        "target shortfall\ndiscretionary cut",
        ha="center",
        va="center",
        color="#9a5a15",
        fontweight="bold",
        fontsize=10,
    )
    ax.text(
        1.30,
        label_y,
        "target funded\npreferred lifestyle",
        ha="center",
        va="center",
        color=COLORS["floor"],
        fontweight="bold",
        fontsize=10,
    )

    ax.annotate(
        "floor breach below\none floor-spending year",
        xy=(floor_threshold * 0.55, floor_threshold * 55),
        xytext=(0.13, 1.05),
        arrowprops=dict(arrowstyle="->", color=COLORS["ruin"], linewidth=1.1),
        color=COLORS["ruin"],
        fontsize=9.5,
        ha="left",
        va="center",
    )
    ax.annotate(
        "cap = floor",
        xy=(floor_to_cap_threshold, floor_pct * 100),
        xytext=(0.47, 1.55),
        arrowprops=dict(arrowstyle="->", color=COLORS["muted"], linewidth=1.0),
        color=COLORS["muted"],
        fontsize=9.5,
        ha="right",
    )
    ax.annotate(
        "cap = target",
        xy=(target_threshold, target_pct * 100),
        xytext=(1.03, 4.35),
        arrowprops=dict(arrowstyle="->", color=COLORS["muted"], linewidth=1.0),
        color=COLORS["muted"],
        fontsize=9.5,
        ha="left",
    )
    ax.text(
        0.72,
        4.05,
        "uncapped line:\n5% of current portfolio",
        color=COLORS["muted"],
        fontsize=8.8,
        ha="center",
    )
    ax.text(1.42, target_pct * 100 + 0.13, "target 5%", color=COLORS["target"], fontweight="bold")
    ax.text(1.42, floor_pct * 100 + 0.13, "floor 2.5%", color=COLORS["floor"], fontweight="bold")

    ax.set_xlabel("Current portfolio value (multiple of starting portfolio)")
    ax.set_ylabel("Withdrawal (% of starting portfolio)")
    ax.set_xlim(0, 1.65)
    ax.set_ylim(0, 6.05)
    ax.set_xticks([floor_threshold, 0.5, 1.0, 1.5])
    ax.set_xticklabels(["0.025x\n1 floor year", "0.50x", "1.00x", "1.50x"])
    style_axis(ax)
    fig.subplots_adjust(top=0.82, left=0.08, right=0.97, bottom=0.14)
    save(fig, "bond_report_spending_cap.png")


def plot_safe_withdrawal_search(results: dict) -> None:
    rows = results["safe_withdrawal_rows"]
    xs = np.array([row["spending_cap_pct"] * 100 for row in rows])

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25], "hspace": 0.16},
    )
    title_block(
        fig,
        "Safe Withdrawal Search",
        "Zero cash and zero bonds; 30-year horizon, 20,000 paths, 5-year historical blocks; floor is 50% of target.",
    )

    ax1.plot(xs, [r["target_shortfall_pct"] for r in rows], marker="o", color=COLORS["target"], linewidth=2.8)
    ax1.plot(xs, [r["floor_breach_pct"] for r in rows], marker="o", color=COLORS["floor"], linewidth=2.5)
    ax1.plot(xs, [r["ruin_pct"] for r in rows], marker="o", color=COLORS["ruin"], linewidth=2.2)
    ax1.axvspan(4, 5.05, color="#d9eadf", alpha=0.55)
    ax1.axvspan(6.0, 7.0, color=COLORS["accent_light"], alpha=0.45)
    ax1.axvspan(8.0, 10.0, color="#f2c8ce", alpha=0.45)
    ax1.text(4.15, 61, "durable", color=COLORS["floor"], fontweight="bold")
    ax1.text(6.05, 61, "aggressive", color=COLORS["accent"], fontweight="bold")
    ax1.text(8.25, 61, "fragile", color=COLORS["target"], fontweight="bold")
    annotate_endpoint(ax1, xs[-1], rows[-1]["target_shortfall_pct"], "target shortfall", COLORS["target"])
    annotate_endpoint(ax1, xs[-1], rows[-1]["floor_breach_pct"], "floor breach", COLORS["floor"])
    annotate_endpoint(ax1, xs[-1], rows[-1]["ruin_pct"], "ruin", COLORS["ruin"])
    ax1.set_ylabel("Risk / shortfall rate (%)")
    ax1.set_ylim(0, 70)
    style_axis(ax1)

    ax2.bar(xs, [r["final_p10_multiple"] for r in rows], width=0.52, color=COLORS["wealth_light"], label="10th percentile")
    ax2.plot(xs, [r["final_median_multiple"] for r in rows], marker="s", color=COLORS["wealth"], linewidth=2.6, label="median")
    annotate_endpoint(ax2, xs[-1], rows[-1]["final_median_multiple"], "median wealth", COLORS["wealth"])
    ax2.set_ylabel("Ending wealth (x)")
    ax2.set_xlabel("Target spending rate (% of initial portfolio)")
    ax2.set_xticks(xs, [pct_label(x) for x in xs])
    ax2.legend(frameon=False, loc="upper right")
    style_axis(ax2)
    fig.subplots_adjust(top=0.85, left=0.08, right=0.88, bottom=0.1)
    save(fig, "bond_report_safe_withdrawal_search.png")


def plot_tradeoff_rows(
    rows: list[dict],
    x_labels: list[str],
    title: str,
    subtitle: str,
    filename: str,
    xlabel: str,
    x_values: list[float] | None = None,
) -> None:
    xs = np.array(x_values if x_values is not None else list(range(len(rows))), dtype=float)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(10.8, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.15, 1.25], "hspace": 0.16},
    )
    title_block(fig, title, subtitle)

    ax1.plot(xs, [r["target_shortfall_pct"] for r in rows], marker="o", color=COLORS["target"], linewidth=2.8)
    ax1.set_ylabel("Target shortfall\npath-years (%)")
    ax1.set_ylim(0, max(r["target_shortfall_pct"] for r in rows) * 1.22)
    style_axis(ax1)

    ax2.plot(xs, [r["ruin_pct"] for r in rows], marker="o", color=COLORS["ruin"], linewidth=2.4, label="Ruin")
    ax2.plot(xs, [r["floor_breach_pct"] for r in rows], marker="o", color=COLORS["floor"], linewidth=2.4, label="Floor breach")
    small_risk_max = max(max(r["ruin_pct"], r["floor_breach_pct"]) for r in rows)
    ax2.set_ylim(0, max(1.5, small_risk_max * 1.35))
    ax2.set_ylabel("Ruin / floor\nbreach (%)")
    ax2.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax2)

    bar_width = 0.55 if x_values is None else max(0.22, min(0.6, (xs[1] - xs[0]) * 0.45 if len(xs) > 1 else 0.55))
    ax3.bar(xs, [r["final_p10_multiple"] for r in rows], color=COLORS["wealth_light"], width=bar_width, label="10th percentile")
    ax3.plot(xs, [r["final_median_multiple"] for r in rows], marker="s", color=COLORS["wealth"], linewidth=2.6, label="median")
    ax3.set_ylabel("Real ending\nwealth (x)")
    ax3.set_xlabel(xlabel)
    ax3.set_xticks(xs, x_labels)
    ax3.legend(frameon=False, loc="upper right")
    style_axis(ax3)

    fig.subplots_adjust(top=0.84, left=0.1, right=0.96, bottom=0.11)
    save(fig, filename)


def plot_cash_buffer(results: dict) -> None:
    rows = results["cash_rows"]
    plot_tradeoff_rows(
        rows=rows,
        x_labels=[f"{row['buffer_years']}y" for row in rows],
        title="T-Bill Cash Buffer Tradeoff",
        subtitle="5% target / 2.5% floor, 30 years, 20,000 paths; x-axis spacing reflects actual buffer years.",
        filename="bond_report_cash_buffer.png",
        xlabel="Cash buffer measured in years of target spending",
        x_values=[row["buffer_years"] for row in rows],
    )


def plot_history_sensitivity(results: dict) -> None:
    rows = results["history_rows"]
    plot_tradeoff_rows(
        rows=rows,
        x_labels=[f"{row['history_years']}y" for row in rows],
        title="Historical Window Sensitivity",
        subtitle="50y starts after 1973-74; 75y includes 1970s inflation; 98y includes Depression/WWII regimes.",
        filename="bond_report_history_sensitivity.png",
        xlabel="Historical lookback window",
    )


def plot_bond_allocation(results: dict) -> None:
    rows = results["bond_rows"]
    xs = np.array([row["bond_pct"] * 100 for row in rows])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(10.8, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.15, 1.25], "hspace": 0.16},
    )
    title_block(
        fig,
        "Bond Allocation Tradeoff",
        "5% target / 2.5% floor, 30 years, 20,000 matched historical-block paths.",
    )

    ax1.axvspan(40, 60, color="#f2c8ce", alpha=0.4)
    ax1.plot(xs, [r["target_shortfall_pct"] for r in rows], marker="o", color=COLORS["target"], linewidth=2.8)
    ax1.text(41, 45, "high-bond region:\nmore shortfall,\nlower ending wealth", color=COLORS["target"], fontweight="bold")
    annotate_endpoint(ax1, xs[-1], rows[-1]["target_shortfall_pct"], "target shortfall", COLORS["target"])
    ax1.set_ylabel("Target shortfall\npath-years (%)")
    ax1.set_ylim(0, 54)
    style_axis(ax1)

    ax2.plot(xs, [r["ruin_pct"] for r in rows], marker="o", color=COLORS["ruin"], linewidth=2.4, label="Ruin")
    ax2.plot(xs, [r["floor_breach_pct"] for r in rows], marker="o", color=COLORS["floor"], linewidth=2.4, label="Floor breach")
    annotate_endpoint(ax2, xs[-1], rows[-1]["ruin_pct"], f"ruin {rows[-1]['ruin_pct']:.2f}%", COLORS["ruin"])
    annotate_endpoint(ax2, xs[-1], rows[-1]["floor_breach_pct"], f"floor {rows[-1]['floor_breach_pct']:.2f}%", COLORS["floor"])
    ax2.set_ylabel("Ruin / floor\nbreach (%)")
    ax2.set_ylim(0, 1.35)
    ax2.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax2)

    ax3.bar(xs, [r["final_p10_multiple"] for r in rows], width=6, color=COLORS["wealth_light"], label="10th percentile")
    ax3.plot(xs, [r["final_median_multiple"] for r in rows], marker="s", color=COLORS["wealth"], linewidth=2.6, label="median")
    annotate_endpoint(ax3, xs[-1], rows[-1]["final_median_multiple"], "median wealth", COLORS["wealth"])
    ax3.set_ylabel("Real ending\nwealth (x)")
    ax3.set_xlabel("Bond allocation (% of non-cash portfolio)")
    ax3.set_xticks(xs, [pct_label(x) for x in xs])
    ax3.legend(frameon=False, loc="upper right")
    style_axis(ax3)
    fig.subplots_adjust(top=0.84, left=0.1, right=0.88, bottom=0.1)
    save(fig, "bond_report_bond_allocation.png")


def plot_objective_tradeoff(results: dict) -> None:
    rows = results["bond_rows"]

    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    title_block(
        fig,
        "Objective Tradeoff",
        "5% target / 2.5% floor, 30-year horizon, 20,000 matched historical-block paths.",
    )

    x = [row["target_shortfall_pct"] for row in rows]
    y = [row["final_median_multiple"] for row in rows]
    sizes = [130 + row["bond_pct"] * 380 for row in rows]
    colors = [COLORS["wealth"] if row["bond_pct"] == 0 else COLORS["accent"] for row in rows]

    ax.plot(x, y, color="#c9ced6", linewidth=2.0, zorder=1)
    ax.scatter(x, y, s=sizes, color=colors, edgecolor="white", linewidth=1.6, zorder=3)

    for row in rows:
        label = f"{row['bond_pct']:.0%} bonds"
        ax.annotate(
            label,
            xy=(row["target_shortfall_pct"], row["final_median_multiple"]),
            xytext=(9, 5),
            textcoords="offset points",
            fontsize=9.5,
            color=COLORS["ink"],
            fontweight="bold" if row["bond_pct"] in [0.0, 0.6] else "normal",
        )

    ax.annotate(
        "direction of higher bond allocation",
        xy=(47.8, 0.96),
        xytext=(27.5, 3.3),
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "linewidth": 1.4},
        color=COLORS["muted"],
        fontsize=10,
    )
    ax.set_xlabel("Target shortfall (% of simulated path-years)")
    ax.set_ylabel("Real median ending wealth (x starting portfolio)")
    ax.set_xlim(17, 52)
    ax.set_ylim(0.4, 4.7)
    style_axis(ax, grid_axis="both")
    fig.subplots_adjust(top=0.82, left=0.1, right=0.96, bottom=0.13)
    save(fig, "bond_report_objective_tradeoff.png")


def plot_objective_tradeoff_combined(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 6.7))
    title_block(
        fig,
        "Objective Tradeoff: 4% And 5% Targets",
        "Both lines use 2:1 target/floor spending, zero cash, 30 years, 20,000 matched historical-block paths.",
    )

    series = [
        ("4% target", results["bond_rows_4pct"], COLORS["wealth"], "o"),
        ("5% target", results["bond_rows"], COLORS["accent"], "s"),
    ]
    for label, rows, color, marker in series:
        x = [row["target_shortfall_pct"] for row in rows]
        y = [row["final_median_multiple"] for row in rows]
        ax.plot(x, y, color=color, linewidth=2.5, marker=marker, markersize=7, label=label)
        for row in rows:
            if row["bond_pct"] in [0.0, 0.4, 0.6]:
                ax.annotate(
                    f"{row['bond_pct']:.0%} bonds",
                    xy=(row["target_shortfall_pct"], row["final_median_multiple"]),
                    xytext=(8, 5),
                    textcoords="offset points",
                    fontsize=8.8,
                    color=COLORS["ink"],
                )

    ax.annotate(
        "higher bond allocation",
        xy=(51.7, 0.83),
        xytext=(27.5, 3.35),
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "linewidth": 1.4},
        color=COLORS["muted"],
        fontsize=10,
    )
    ax.set_xlabel("Target shortfall (% of simulated path-years)")
    ax.set_ylabel("Real median ending wealth (x starting portfolio)")
    ax.set_xlim(17, 54)
    ax.set_ylim(0.4, 4.7)
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax, grid_axis="both")
    fig.subplots_adjust(top=0.82, left=0.1, right=0.96, bottom=0.13)
    save(fig, "bond_report_objective_tradeoff_4_5.png")


def load_followup() -> dict | None:
    for path in (
        ASSET_DIR / "followup_results.json",
        ROOT.parent / "docs" / "assets" / "followup_results.json",
    ):
        if path.exists():
            return json.loads(path.read_text())
    return None


def plot_erp_grid(followup: dict) -> None:
    grid = followup.get("erp_grid") or []
    if not grid:
        return
    levels = sorted({row["erp_kept"] for row in grid}, reverse=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.4))
    title_block(
        fig,
        "When Does A Permanent Treasury Sleeve Pay?",
        "4% target / 2% floor. Mean equity premium is scaled; crash sequencing is kept.",
    )
    palette = [COLORS["wealth"], COLORS["floor"], COLORS["accent"], COLORS["target"], COLORS["ruin"]]
    for color, haircut in zip(palette, levels):
        rows = sorted(
            [row for row in grid if row["erp_kept"] == haircut],
            key=lambda row: row["bond_pct"],
        )
        xs = [row["bond_pct"] * 100 for row in rows]
        delivered = [
            row.get(
                "target_spend_delivered_pct",
                100 - row["target_shortfall_integrated_loss_pct_target"],
            )
            for row in rows
        ]
        medians = [row["final_median_multiple"] for row in rows]
        ax1.plot(xs, delivered, marker="o", color=color, linewidth=2.2, label=f"{haircut:.0%} ERP")
        ax2.plot(xs, medians, marker="o", color=color, linewidth=2.2, label=f"{haircut:.0%} ERP")
    ax1.set_xlabel("Bond allocation (%)")
    ax1.set_ylabel("Target spending delivered (%)")
    ax2.set_xlabel("Bond allocation (%)")
    ax2.set_ylabel("Real median ending wealth (x)")
    ax1.legend(frameon=False, fontsize=8.5)
    ax2.legend(frameon=False, fontsize=8.5)
    style_axis(ax1)
    style_axis(ax2)
    fig.subplots_adjust(top=0.82, left=0.08, right=0.98, bottom=0.14, wspace=0.28)
    save(fig, "bond_report_erp_grid.png")


def _spend_delivered(row: dict) -> float:
    if "target_spend_delivered_pct" in row:
        return float(row["target_spend_delivered_pct"])
    return 100.0 - float(row["target_shortfall_integrated_loss_pct_target"])


def plot_insurance_frontier(followup: dict) -> None:
    """Median wealth vs ruin probability for stock-only, Treasuries, and floor buckets."""
    points = []
    grid = followup.get("erp_grid") or []
    historical = {
        round(row.get("bond_pct", -1), 4): row
        for row in grid
        if abs(row.get("erp_kept", -1) - 1.0) < 1e-9
    }
    for bond_pct, label in ((0.0, "Stock-only"), (0.2, "Permanent 20% Treasuries"), (0.4, "Permanent 40% Treasuries")):
        row = historical.get(bond_pct)
        if row:
            points.append((label, row, COLORS["wealth"] if bond_pct == 0 else COLORS["accent"]))
    for row in followup.get("floor_rows") or []:
        years = int(row.get("floor_years", 0))
        color = {10: COLORS["floor"], 20: COLORS["ink"], 25: COLORS["target"]}.get(years, COLORS["muted"])
        points.append((f"{years}-year floor bucket", row, color))
    if len(points) < 2:
        return

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    title_block(
        fig,
        "How Much Wealth For Each Increment Of Insurance?",
        "4% flexible target. Left is safer; higher is richer. Marker size is lifetime spending delivered.",
    )
    xs = [row["ruin_pct"] for _, row, _ in points]
    ys = [row["final_median_multiple"] for _, row, _ in points]
    delivered = [_spend_delivered(row) for _, row, _ in points]
    sizes = [180 + (d - min(delivered)) / max(max(delivered) - min(delivered), 0.01) * 420 for d in delivered]
    for (label, row, color), x, y, size, d in zip(points, xs, ys, sizes, delivered):
        ax.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(
            f"{label}\n{d:.1f}% spend",
            xy=(x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.6,
            color=COLORS["ink"],
        )
    ax.set_xlabel("Probability of ruin (%)")
    ax.set_ylabel("Real median ending wealth (x)")
    style_axis(ax, grid_axis="both")
    fig.subplots_adjust(top=0.82, left=0.1, right=0.96, bottom=0.13)
    save(fig, "bond_report_insurance_frontier.png")


def plot_terminal_wealth_cdf(followup: dict) -> None:
    grid = followup.get("erp_grid") or []
    historical = [
        row for row in grid if abs(row.get("erp_kept", -1) - 1.0) < 1e-9 and "wealth_percentiles" in row
    ]
    if not historical:
        return
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    title_block(
        fig,
        "Terminal Wealth CDF: 4% Flexible Rule",
        "Historical equity premium. Bonds insure the far left tail, then sit below stocks.",
    )
    palette = {
        0.0: COLORS["wealth"],
        0.2: COLORS["floor"],
        0.4: COLORS["accent"],
        0.6: COLORS["target"],
    }
    for row in sorted(historical, key=lambda item: item["bond_pct"]):
        pcts = row["wealth_percentiles"]
        xs = [int(key[1:]) for key in pcts]
        ys = [pcts[f"p{p}"] for p in xs]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.3,
            color=palette.get(row["bond_pct"], COLORS["ink"]),
            label=f"{row['bond_pct']:.0%} bonds",
        )
    ax.set_xlabel("Terminal-wealth percentile")
    ax.set_ylabel("Real ending wealth (x starting portfolio)")
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="both")
    fig.subplots_adjust(top=0.82, left=0.1, right=0.96, bottom=0.13)
    save(fig, "bond_report_terminal_cdf.png")


def main() -> None:
    setup_style()
    results = load_results()
    plot_bootstrap_method(results)
    plot_spending_rule(results)
    plot_safe_withdrawal_search(results)
    plot_cash_buffer(results)
    plot_history_sensitivity(results)
    plot_bond_allocation(results)
    plot_objective_tradeoff(results)
    plot_objective_tradeoff_combined(results)
    followup = load_followup()
    if followup:
        plot_erp_grid(followup)
        plot_terminal_wealth_cdf(followup)
        plot_insurance_frontier(followup)


if __name__ == "__main__":
    main()
