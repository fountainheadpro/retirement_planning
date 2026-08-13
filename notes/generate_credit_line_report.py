"""Generate a standalone monthly credit-line retirement report."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics import ruin_ever, summarize_withdrawals
from spending import apply_target_cap_floor

DOCS_DIR = ROOT / "docs"
ASSET_DIR = DOCS_DIR / "assets"
REPORT_PATH = DOCS_DIR / "credit-line.html"
RESULTS_PATH = ASSET_DIR / "credit_line_results.json"

BASE = 1.0
YEARS = 30
MONTHS = YEARS * 12
N_PATHS = 20_000
BLOCK_MONTHS = 60
SEED = 20260516
TARGET_RATES = [0.05, 0.04]
FLOOR_RATIO = 0.5
REAL_CREDIT_SPREAD = 0.02
MAX_LTV = 0.25
TRIGGERS = [0.10, 0.20, 0.30, 0.40]
MAIN_BORROW_RULE = "cap"
COMPARISON_BORROW_RULE = "target"

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
    "paper": "#fbfaf5",
    "panel": "#ffffff",
}


@dataclass(frozen=True)
class MonthlyHistory:
    """Aligned monthly total-return, CPI, and real-return records."""

    data: pd.DataFrame

    @property
    def start_month(self) -> str:
        return self.data.index.min().strftime("%b %Y")

    @property
    def end_month(self) -> str:
        return self.data.index.max().strftime("%b %Y")

    @property
    def n_months(self) -> int:
        return len(self.data)


def fetch_monthly_history() -> MonthlyHistory:
    """Download S&P 500 total return and CPI, then align completed months."""
    sp = yf.download(
        "^SP500TR",
        period="max",
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    if sp.empty:
        raise RuntimeError("Yahoo returned no data for ^SP500TR")

    close = sp["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().rename("sp500tr")
    close.index = pd.to_datetime(close.index).to_period("M").to_timestamp()

    cpi = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL")
    cpi["observation_date"] = pd.to_datetime(cpi["observation_date"])
    cpi = cpi.rename(columns={"observation_date": "date", "CPIAUCSL": "cpi"})
    cpi = cpi.set_index("date")["cpi"].dropna()
    cpi.index = cpi.index.to_period("M").to_timestamp()

    current_month = pd.Timestamp.today().to_period("M").to_timestamp()
    combined = pd.concat([close, cpi], axis=1).dropna()
    combined = combined[combined.index < current_month]
    combined["stock_return_nominal"] = combined["sp500tr"].pct_change()
    combined["inflation"] = combined["cpi"].pct_change()
    combined = combined.dropna()
    combined["stock_return_real"] = (
        (1.0 + combined["stock_return_nominal"])
        / (1.0 + combined["inflation"])
        - 1.0
    )
    return MonthlyHistory(combined)


def sample_monthly_paths(history: MonthlyHistory) -> dict[str, np.ndarray]:
    """Sample paired monthly stock/inflation records in 5-year blocks."""
    rng = np.random.default_rng(SEED)
    n_blocks = int(np.ceil(MONTHS / BLOCK_MONTHS))
    max_start = history.n_months - BLOCK_MONTHS
    if max_start < 0:
        raise ValueError("Monthly history is shorter than one bootstrap block")

    starts = rng.integers(0, max_start + 1, size=(n_blocks, N_PATHS))
    offsets = np.arange(BLOCK_MONTHS)[:, None]
    indices = np.empty((n_blocks * BLOCK_MONTHS, N_PATHS), dtype=np.int32)
    for block_idx in range(n_blocks):
        begin = block_idx * BLOCK_MONTHS
        end = begin + BLOCK_MONTHS
        indices[begin:end, :] = starts[block_idx, :] + offsets
    indices = indices[:MONTHS, :]

    return {
        "stock_nominal": history.data["stock_return_nominal"].to_numpy()[indices],
        "stock_real": history.data["stock_return_real"].to_numpy()[indices],
        "inflation": history.data["inflation"].to_numpy()[indices],
    }


def scenario_label(target_rate: float) -> str:
    """Human-readable target/floor scenario label."""
    target = f"{target_rate * 100:g}%"
    floor = f"{target_rate * FLOOR_RATIO * 100:g}%"
    return f"{target} target / {floor} floor"


def borrow_rule_label(borrow_rule: str) -> str:
    """Human-readable label for how much the credit line can fund."""
    labels = {
        "target": "Target-protecting credit",
        "cap": "Hybrid cap-respecting credit",
    }
    if borrow_rule not in labels:
        raise ValueError(f"Unknown borrow rule: {borrow_rule}")
    return labels[borrow_rule]


def spending_amount(wealth: np.ndarray, target_rate: float) -> np.ndarray:
    """Monthly target/cap/floor spending rule in real dollars."""
    target_monthly = target_rate / 12.0 * BASE
    floor_monthly = target_monthly * FLOOR_RATIO
    return apply_target_cap_floor(
        wealth,
        target_monthly,
        target_rate / 12.0,
        floor_monthly,
    )


def run_sell_baseline(paths: dict[str, np.ndarray], target_rate: float) -> dict[str, np.ndarray]:
    """Run stock-only flexible spending with no borrowing."""
    assets = np.full(N_PATHS, BASE, dtype=float)
    spending = np.zeros((MONTHS, N_PATHS), dtype=np.float32)
    net_values = np.zeros((MONTHS + 1, N_PATHS), dtype=np.float32)
    net_values[0, :] = assets

    for month in range(MONTHS):
        alive = assets > 0.0
        assets[alive] *= 1.0 + paths["stock_real"][month, alive]
        assets = np.maximum(assets, 0.0)

        withdrawal = spending_amount(assets, target_rate)
        assets -= withdrawal
        assets = np.maximum(assets, 0.0)
        spending[month, :] = withdrawal
        net_values[month + 1, :] = assets

    return {
        "spending": spending,
        "net_values": net_values,
        "assets": net_values,
        "debt": np.zeros_like(net_values),
        "credit_used": np.zeros_like(spending, dtype=bool),
        "margin_event": np.zeros_like(spending, dtype=bool),
    }


def run_credit_line(
    paths: dict[str, np.ndarray],
    trigger: float,
    target_rate: float,
    borrow_rule: str = MAIN_BORROW_RULE,
) -> dict[str, np.ndarray]:
    """Run credit-line strategy through drawdown cycles."""
    if borrow_rule not in {"target", "cap"}:
        raise ValueError(f"Unknown borrow rule: {borrow_rule}")

    assets = np.full(N_PATHS, BASE, dtype=float)
    debt = np.zeros(N_PATHS, dtype=float)
    market_index = np.ones(N_PATHS, dtype=float)
    market_peak = np.ones(N_PATHS, dtype=float)
    cycle_peak = np.ones(N_PATHS, dtype=float)
    in_credit_cycle = np.zeros(N_PATHS, dtype=bool)

    spending = np.zeros((MONTHS, N_PATHS), dtype=np.float32)
    net_values = np.zeros((MONTHS + 1, N_PATHS), dtype=np.float32)
    asset_values = np.zeros((MONTHS + 1, N_PATHS), dtype=np.float32)
    debt_values = np.zeros((MONTHS + 1, N_PATHS), dtype=np.float32)
    credit_used = np.zeros((MONTHS, N_PATHS), dtype=bool)
    margin_event = np.zeros((MONTHS, N_PATHS), dtype=bool)

    real_credit_rate = (1.0 + REAL_CREDIT_SPREAD) ** (1.0 / 12.0) - 1.0
    target_monthly = target_rate / 12.0 * BASE

    net_values[0, :] = assets
    asset_values[0, :] = assets

    for month in range(MONTHS):
        assets *= 1.0 + paths["stock_real"][month, :]
        assets = np.maximum(assets, 0.0)
        debt *= 1.0 + real_credit_rate

        market_index *= 1.0 + paths["stock_nominal"][month, :]
        prior_peak = market_peak.copy()
        market_peak = np.maximum(market_peak, market_index)
        drawdown = market_index / market_peak - 1.0

        newly_triggered = (~in_credit_cycle) & (drawdown <= -trigger) & (assets > 0.0)
        cycle_peak[newly_triggered] = prior_peak[newly_triggered]
        in_credit_cycle |= newly_triggered

        over_ltv = (debt > 0.0) & (assets > 0.0) & (debt > MAX_LTV * assets)
        if np.any(over_ltv):
            required_repay = (debt[over_ltv] - MAX_LTV * assets[over_ltv]) / (1.0 - MAX_LTV)
            repay = np.minimum(required_repay, np.minimum(assets[over_ltv], debt[over_ltv]))
            assets[over_ltv] -= repay
            debt[over_ltv] -= repay
            margin_event[month, over_ltv] = True

        recovered = in_credit_cycle & (market_index >= cycle_peak * 0.999)
        if np.any(recovered):
            has_debt = recovered & (debt > 0.0)
            if np.any(has_debt):
                repay = np.minimum(debt[has_debt], assets[has_debt])
                assets[has_debt] -= repay
                debt[has_debt] -= repay
            in_credit_cycle[recovered] = False

        in_credit_cycle &= assets > 0.0
        normal_desired = spending_amount(np.maximum(assets - debt, 0.0), target_rate)
        capacity = np.maximum(MAX_LTV * assets - debt, 0.0)
        can_borrow = in_credit_cycle & (capacity > 0.0) & (assets > debt)

        borrowed = np.zeros(N_PATHS, dtype=float)
        if np.any(can_borrow):
            if borrow_rule == "target":
                borrow_need = np.full_like(assets, target_monthly)
            else:
                borrow_need = normal_desired
            borrowed[can_borrow] = np.minimum(
                borrow_need[can_borrow],
                capacity[can_borrow],
            )
            debt[can_borrow] += borrowed[can_borrow]
            credit_used[month, can_borrow & (borrowed > 0.0)] = True

        sale_needed = np.maximum(normal_desired - borrowed, 0.0)
        sale = np.minimum(sale_needed, assets)
        assets -= sale
        actual_spend = borrowed + sale

        insolvent = (assets - debt) <= 0.0
        if np.any(insolvent):
            assets[insolvent] = 0.0
            debt[insolvent] = 0.0
            in_credit_cycle[insolvent] = False

        spending[month, :] = actual_spend
        net_values[month + 1, :] = np.maximum(assets - debt, 0.0)
        asset_values[month + 1, :] = assets
        debt_values[month + 1, :] = debt

    return {
        "spending": spending,
        "net_values": net_values,
        "assets": asset_values,
        "debt": debt_values,
        "credit_used": credit_used,
        "margin_event": margin_event,
    }


def summarize(
    label: str,
    simulation: dict[str, np.ndarray],
    target_rate: float,
    borrow_rule: str = "none",
) -> dict:
    """Summarize monthly path outcomes in report-ready metrics."""
    spending = simulation["spending"]
    net_values = simulation["net_values"]
    debt = simulation["debt"]
    credit_used = simulation["credit_used"]
    margin_event = simulation["margin_event"]

    target_monthly = target_rate / 12.0 * BASE
    floor_monthly = target_monthly * FLOOR_RATIO
    final = net_values[-1, :]
    common = summarize_withdrawals(
        spending,
        target_monthly,
        floor_monthly,
        final_wealth=final,
        initial_wealth=BASE,
        tolerance=1e-8,
    )
    ruin_path = ruin_ever(net_values)
    debt_months = debt[1:, :] > 1e-8
    debt_month_counts = debt_months.sum(axis=0)
    debt_month_counts = debt_month_counts[debt_month_counts > 0]

    return {
        "label": label,
        "scenario": scenario_label(target_rate),
        "target_rate": target_rate,
        "floor_rate": target_rate * FLOOR_RATIO,
        "borrow_rule": borrow_rule,
        "borrow_rule_label": (
            "Sell only" if borrow_rule == "none" else borrow_rule_label(borrow_rule)
        ),
        "ruin_pct": float(np.mean(ruin_path) * 100),
        "ruin_count": int(np.sum(ruin_path)),
        "target_shortfall_pct": common["target_shortfall_pct"],
        "target_shortfall_ever_pct": common["target_shortfall_ever_pct"],
        "target_shortfall_median_months_if_any": common[
            "target_shortfall_median_years_if_any"
        ],
        "floor_breach_pct": common["floor_breach_pct"],
        "floor_breach_ever_pct": common["floor_breach_ever_pct"],
        "avg_shortfall_gap_pct": common["target_shortfall_avg_depth_pct_target"] * 100,
        "integrated_target_loss_pct": common["target_shortfall_integrated_loss_pct_target"],
        "final_p10_multiple": common["final_p10_multiple"],
        "final_median_multiple": common["final_median_multiple"],
        "final_p90_multiple": float(np.percentile(final, 90)),
        "ever_credit_used_pct": float(np.mean(np.any(credit_used, axis=0)) * 100),
        "median_debt_months_if_any": (
            float(np.median(debt_month_counts)) if len(debt_month_counts) else 0.0
        ),
        "peak_debt_p95_pct_initial": float(np.percentile(np.max(debt, axis=0), 95) * 100),
        "margin_event_pct": float(np.mean(np.any(margin_event, axis=0)) * 100),
    }


def run_strategy_rows(
    paths: dict[str, np.ndarray],
    target_rate: float,
    borrow_rule: str = MAIN_BORROW_RULE,
) -> list[dict]:
    """Run sell-only and credit-line trigger rows for one spending target."""
    rows: list[dict] = []
    baseline = summarize("Sell only", run_sell_baseline(paths, target_rate), target_rate)
    baseline["trigger_pct"] = None
    rows.append(baseline)

    for trigger in TRIGGERS:
        label = f"{trigger:.0%} trigger"
        row = summarize(
            label,
            run_credit_line(paths, trigger, target_rate, borrow_rule=borrow_rule),
            target_rate,
            borrow_rule=borrow_rule,
        )
        row["trigger_pct"] = trigger * 100
        rows.append(row)

    return rows


def build_scenarios(paths: dict[str, np.ndarray], borrow_rule: str) -> list[dict]:
    """Run all spending scenarios for a borrow rule."""
    return [
        {
            "label": scenario_label(target_rate),
            "target_rate": target_rate,
            "floor_rate": target_rate * FLOOR_RATIO,
            "borrow_rule": borrow_rule,
            "borrow_rule_label": borrow_rule_label(borrow_rule),
            "rows": run_strategy_rows(paths, target_rate, borrow_rule=borrow_rule),
        }
        for target_rate in TARGET_RATES
    ]


def find_trigger_row(rows: list[dict], trigger_pct: float) -> dict:
    """Find the row for a trigger percentage."""
    for row in rows:
        if row.get("trigger_pct") is not None and abs(row["trigger_pct"] - trigger_pct) < 1e-9:
            return row
    raise ValueError(f"Missing trigger row: {trigger_pct}%")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": "#f8faf8",
            "axes.edgecolor": "#c7cfd8",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "grid.color": COLORS["grid"],
            "font.size": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(
        title,
        x=0.06,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(0.06, 0.915, subtitle, ha="left", fontsize=10.5, color=COLORS["muted"])


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, alpha=0.85, linewidth=0.8)
    ax.tick_params(length=0)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c7cfd8")


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(ASSET_DIR / filename, dpi=180, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)


def plot_mechanics() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.axis("off")
    title_block(
        fig,
        "Hybrid Credit-Line Mechanics",
        "The loan replaces forced stock sales during drawdowns; it does not override the spending cap.",
    )

    boxes = [
        (
            0.04,
            "1. Stay invested",
            "Portfolio remains in S&P 500\nTotal Return. Spending normally\ncomes from selling shares.",
            "#eef4f7",
        ),
        (
            0.285,
            "2. Drawdown trigger",
            "If the total-return index falls\nbelow the selected threshold,\nenter a credit cycle.",
            "#fff0d6",
        ),
        (
            0.53,
            "3. Borrow less",
            "Borrow only the normal\ncap/floor spending amount.\nDebt costs CPI + 2%.",
            "#f7e2e6",
        ),
        (
            0.765,
            "4. Recover and repay",
            "At the pre-trigger high,\nsell shares to repay the\noutstanding balance.",
            "#e7f1ee",
        ),
    ]

    for x, title, body, color in boxes:
        patch = patches.FancyBboxPatch(
            (x, 0.46),
            0.20,
            0.32,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="#c7cfd8",
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x + 0.018, 0.72, title, transform=ax.transAxes, fontweight="bold", color=COLORS["ink"])
        ax.text(x + 0.018, 0.65, body, transform=ax.transAxes, fontsize=8.9, color=COLORS["muted"], va="top")

    for x in [0.245, 0.49, 0.735]:
        ax.annotate(
            "",
            xy=(x + 0.035, 0.62),
            xytext=(x, 0.62),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "linewidth": 1.5},
        )

    ax.text(
        0.07,
        0.24,
        "Guardrails: max loan-to-value is 25% of portfolio assets. If falling markets push debt above that limit,\n"
        "the model forces a repayment by selling shares and records a margin event.",
        transform=ax.transAxes,
        fontsize=10.2,
        color=COLORS["ink"],
        bbox={"facecolor": "#ffffff", "edgecolor": "#d8dee5", "boxstyle": "round,pad=0.55"},
    )
    fig.subplots_adjust(top=0.82, left=0.04, right=0.98, bottom=0.06)
    save(fig, "credit_line_mechanics.png")


def plot_tradeoff(scenarios: list[dict]) -> None:
    colors = [COLORS["target"], COLORS["floor"]]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(10.8, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.2, 1.25], "hspace": 0.17},
    )
    title_block(
        fig,
        "Credit-Line Trigger Search",
        "Hybrid cap-respecting credit, monthly S&P 500 Total Return blocks, CPI + 2% credit cost, 25% max LTV.",
    )

    for idx, scenario in enumerate(scenarios):
        rows = scenario["rows"]
        baseline = rows[0]
        credit_rows = rows[1:]
        xs = np.array([row["trigger_pct"] for row in credit_rows])
        color = colors[idx % len(colors)]

        ax1.axhline(
            baseline["target_shortfall_pct"],
            color=color,
            linestyle=":",
            linewidth=1.6,
            alpha=0.7,
        )
        ax1.plot(
            xs,
            [r["target_shortfall_pct"] for r in credit_rows],
            marker="o",
            color=color,
            linewidth=2.6,
            label=f"{scenario['label']} hybrid credit",
        )
        ax1.text(
            xs[0] - 0.5,
            baseline["target_shortfall_pct"] + 0.15,
            f"{scenario['target_rate']:.0%} sell-only",
            color=color,
            fontsize=8.9,
        )
    ax1.set_ylabel("Target shortfall\npath-months (%)")
    ax1.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax1)

    for idx, scenario in enumerate(scenarios):
        rows = scenario["rows"]
        credit_rows = rows[1:]
        xs = np.array([row["trigger_pct"] for row in credit_rows])
        color = colors[idx % len(colors)]
        ax2.plot(
            xs,
            [r["ruin_pct"] for r in credit_rows],
            marker="o",
            color=color,
            linewidth=2.3,
            label=f"{scenario['target_rate']:.0%} ruin",
        )
        ax2.plot(
            xs,
            [r["margin_event_pct"] for r in credit_rows],
            marker="s",
            color=color,
            linestyle="--",
            linewidth=2.0,
            alpha=0.72,
            label=f"{scenario['target_rate']:.0%} margin",
        )
    ax2.set_ylabel("Path risk (%)")
    ax2.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax2)

    for idx, scenario in enumerate(scenarios):
        rows = scenario["rows"]
        baseline = rows[0]
        credit_rows = rows[1:]
        xs = np.array([row["trigger_pct"] for row in credit_rows])
        color = colors[idx % len(colors)]
        ax3.axhline(
            baseline["final_p10_multiple"],
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.65,
        )
        ax3.plot(
            xs,
            [r["final_p10_multiple"] for r in credit_rows],
            marker="o",
            color=color,
            linestyle="--",
            linewidth=2.0,
            label=f"{scenario['target_rate']:.0%} p10",
        )
        ax3.plot(
            xs,
            [r["final_median_multiple"] for r in credit_rows],
            marker="s",
            color=color,
            linewidth=2.5,
            label=f"{scenario['target_rate']:.0%} median",
        )
    ax3.set_ylabel("Real ending\nnet wealth (x)")
    ax3.set_xlabel("Drawdown trigger for credit use")
    ax3.set_xticks(xs, [f"{x:.0f}%" for x in xs])
    ax3.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax3)

    fig.subplots_adjust(top=0.84, left=0.1, right=0.9, bottom=0.1)
    save(fig, "credit_line_trigger_search.png")


def plot_objective(scenarios: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    title_block(
        fig,
        "Objective Tradeoff",
        "The hybrid rule borrows less, so the test is whether reduced forced selling is worth the debt risk.",
    )

    colors = [COLORS["target"], COLORS["floor"]]
    markers = ["o", "s"]
    for idx, scenario in enumerate(scenarios):
        rows = scenario["rows"]
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        x = [row["target_shortfall_pct"] for row in rows]
        y = [row["final_p10_multiple"] for row in rows]
        sizes = [135] + [105 + row["margin_event_pct"] * 8 for row in rows[1:]]
        ax.plot(x, y, color=color, linewidth=2.0, alpha=0.5, zorder=1)
        ax.scatter(
            x,
            y,
            s=sizes,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.6,
            label=scenario["label"],
            zorder=3,
        )

        for row in rows:
            if row["label"] not in ["Sell only", "20% trigger", "40% trigger"]:
                continue
            label = f"{scenario['target_rate']:.0%} {row['label']}"
            ax.annotate(
                label,
                xy=(row["target_shortfall_pct"], row["final_p10_multiple"]),
                xytext=(8, 5),
                textcoords="offset points",
                fontsize=8.4,
                color=COLORS["ink"],
                fontweight="bold" if row["label"] == "Sell only" else "normal",
                bbox={"facecolor": COLORS["paper"], "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
                zorder=6,
            )

    ax.set_xlabel("Target shortfall (% of simulated path-months)")
    ax.set_ylabel("Real 10th percentile ending net wealth (x starting portfolio)")
    ax.text(
        0.03,
        0.06,
        "Bubble size = margin-event risk",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=9.2,
    )
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax, grid_axis="both")
    fig.subplots_adjust(top=0.82, left=0.1, right=0.96, bottom=0.13)
    save(fig, "credit_line_objective_tradeoff.png")


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def multiple(value: float) -> str:
    return f"{value:.2f}x"


def result_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        trigger = "n/a" if row.get("trigger_pct") is None else f"{row['trigger_pct']:.0f}%"
        body.append(
            "<tr>"
            f"<td>{row['label']}</td>"
            f"<td>{trigger}</td>"
            f"<td>{pct(row['ruin_pct'])}</td>"
            f"<td>{pct(row['target_shortfall_pct'])}</td>"
            f"<td>{pct(row['floor_breach_pct'])}</td>"
            f"<td>{multiple(row['final_p10_multiple'])}</td>"
            f"<td>{multiple(row['final_median_multiple'])}</td>"
            f"<td>{pct(row['ever_credit_used_pct'])}</td>"
            f"<td>{pct(row['margin_event_pct'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def debt_table(rows: list[dict]) -> str:
    body = []
    for row in rows[1:]:
        body.append(
            "<tr>"
            f"<td>{row['label']}</td>"
            f"<td>{pct(row['ever_credit_used_pct'])}</td>"
            f"<td>{row['median_debt_months_if_any']:.0f}</td>"
            f"<td>{pct(row['peak_debt_p95_pct_initial'])}</td>"
            f"<td>{pct(row['target_shortfall_ever_pct'])}</td>"
            f"<td>{pct(row['avg_shortfall_gap_pct'], 1)}</td>"
            f"<td>{pct(row['integrated_target_loss_pct'])}</td>"
            "</tr>"
        )
    return "\n".join(body)


def scenario_result_sections(scenarios: list[dict]) -> str:
    """Render result tables for each spending scenario."""
    sections = []
    for scenario in scenarios:
        sections.append(
            f"""
          <h3>{scenario['label']} · {scenario['borrow_rule_label']}</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Strategy</th><th>Trigger</th><th>Ruin</th><th>Target shortfall</th><th>Floor breach</th><th>Real final p10</th><th>Real final median</th><th>Ever used credit</th><th>Margin event</th></tr>
              </thead>
              <tbody>
                {result_table(scenario['rows'])}
              </tbody>
            </table>
          </div>"""
        )
    return "\n".join(sections)


def scenario_debt_sections(scenarios: list[dict]) -> str:
    """Render debt-risk tables for each spending scenario."""
    sections = []
    for scenario in scenarios:
        sections.append(
            f"""
          <h3>{scenario['label']}</h3>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Strategy</th><th>Ever used credit</th><th>Median debt months if used</th><th>95th pct peak debt</th><th>Ever miss target</th><th>Avg shortfall gap</th><th>Integrated target loss</th></tr>
              </thead>
              <tbody>
                {debt_table(scenario['rows'])}
              </tbody>
            </table>
          </div>"""
        )
    return "\n".join(sections)


def borrow_rule_comparison_table(
    hybrid_scenarios: list[dict],
    target_protecting_scenarios: list[dict],
    trigger_pct: float = 20.0,
) -> str:
    """Render old versus new credit-line behavior at one drawdown trigger."""
    body = []
    for hybrid, target_protecting in zip(hybrid_scenarios, target_protecting_scenarios):
        baseline = hybrid["rows"][0]
        target_row = find_trigger_row(target_protecting["rows"], trigger_pct)
        hybrid_row = find_trigger_row(hybrid["rows"], trigger_pct)
        rows = [
            ("Sell only", "No borrowing", baseline),
            ("Old credit model", "Borrow full target spending", target_row),
            ("Hybrid credit model", "Borrow target/cap/floor spending", hybrid_row),
        ]
        for strategy, funding_rule, row in rows:
            body.append(
                "<tr>"
                f"<td>{hybrid['label']}</td>"
                f"<td>{strategy}</td>"
                f"<td>{funding_rule}</td>"
                f"<td>{pct(row['ruin_pct'])}</td>"
                f"<td>{pct(row['target_shortfall_pct'])}</td>"
                f"<td>{pct(row['floor_breach_pct'])}</td>"
                f"<td>{multiple(row['final_p10_multiple'])}</td>"
                f"<td>{multiple(row['final_median_multiple'])}</td>"
                f"<td>{pct(row['margin_event_pct'])}</td>"
                "</tr>"
            )
    return "\n".join(body)


def write_report(results: dict | None = None) -> None:
    if results is None:
        results = json.loads(RESULTS_PATH.read_text())
    scenarios = results["scenarios"]
    target_protecting_scenarios = results["target_protecting_scenarios"]
    primary_rows = scenarios[0]["rows"]
    lower_spend_rows = scenarios[1]["rows"]
    baseline = primary_rows[0]
    lower_spend_baseline = lower_spend_rows[0]
    best_shortfall = min(primary_rows[1:], key=lambda row: row["target_shortfall_pct"])
    lower_spend_best_shortfall = min(lower_spend_rows[1:], key=lambda row: row["target_shortfall_pct"])
    hybrid_20 = find_trigger_row(primary_rows, 20.0)
    target_protecting_20 = find_trigger_row(target_protecting_scenarios[0]["rows"], 20.0)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Can A Hybrid Credit Line Reduce Forced Selling?</title>
  <meta name="description" content="A monthly S&P 500 total-return simulation testing a cap-respecting asset-backed credit line as a retirement drawdown buffer.">
  <meta property="og:title" content="Can A Hybrid Credit Line Reduce Forced Selling?">
  <meta property="og:description" content="A separate retirement report testing whether a smaller, cap-respecting credit line can reduce forced selling without overriding flexible spending.">
  <meta property="og:type" content="article">
  <meta property="og:image" content="assets/credit_line_objective_tradeoff.png">
  <link rel="icon" href="favicon.svg" type="image/svg+xml">
  <style>
    @import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,500;8..60,650;8..60,750&display=swap");
    :root {{
      --paper: #fbfaf5;
      --panel: #ffffff;
      --ink: #16212c;
      --muted: #697381;
      --line: #d8dedf;
      --red: #b23a48;
      --green: #2f7d64;
      --blue: #274c77;
      --gold: #d2872c;
      --radius: 6px;
      --shadow: 0 24px 70px rgba(22, 33, 44, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background:
        linear-gradient(90deg, rgba(22, 33, 44, 0.035) 1px, transparent 1px) 0 0 / 44px 44px,
        linear-gradient(180deg, #f4efe3 0%, var(--paper) 34%, #f4f0e6 100%);
      color: var(--ink);
      font-family: "IBM Plex Sans", ui-sans-serif, sans-serif;
      line-height: 1.58;
    }}
    a {{ color: var(--blue); text-underline-offset: 3px; }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 0 24px 72px; }}
    .hero {{
      min-height: 76vh;
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      gap: 52px;
      align-items: center;
      padding: 56px 0 42px;
    }}
    .eyebrow {{
      color: var(--red);
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin: 0 0 18px;
    }}
    h1, h2, h3 {{ font-family: "Source Serif 4", Georgia, serif; line-height: 1.03; letter-spacing: 0; margin: 0; }}
    h1 {{ font-size: clamp(3rem, 7vw, 6.8rem); max-width: 920px; }}
    h2 {{ font-size: clamp(2rem, 4vw, 4.2rem); margin-bottom: 20px; }}
    h3 {{ font-size: clamp(1.35rem, 2vw, 2.05rem); margin: 20px 0 10px; }}
    .dek {{ font-size: clamp(1.15rem, 2vw, 1.48rem); max-width: 760px; color: #33414f; margin: 24px 0 0; }}
    .hero-panel {{
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(22, 33, 44, 0.12);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 22px;
      backdrop-filter: blur(18px);
    }}
    .hero-panel img, .figure img {{ width: 100%; display: block; border-radius: 4px; border: 1px solid var(--line); background: white; }}
    .caption, figcaption {{ color: var(--muted); font-size: 0.94rem; margin: 12px 0 0; }}
    .metric-strip {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      margin: 20px 0 44px;
      background: rgba(255, 255, 255, 0.55);
    }}
    .metric {{ padding: 22px 18px; border-right: 1px solid var(--line); }}
    .metric:last-child {{ border-right: 0; }}
    .metric strong {{ display: block; font-family: "Source Serif 4", Georgia, serif; font-size: clamp(1.65rem, 3vw, 2.8rem); line-height: 1; margin-bottom: 8px; }}
    .metric span {{ color: var(--muted); font-size: 0.92rem; }}
    .layout {{ display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 44px; align-items: start; }}
    .site-nav {{ border-bottom: 1px solid var(--line); background: rgba(251, 250, 245, 0.92); backdrop-filter: blur(12px); }}
    .site-nav-inner {{ max-width: 1180px; margin: 0 auto; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .site-nav-brand {{ color: var(--ink); font-weight: 600; text-decoration: none; letter-spacing: 0.02em; }}
    .site-nav-links {{ display: flex; flex-wrap: wrap; gap: 8px 18px; }}
    .site-nav a {{ color: #344250; text-decoration: none; font-size: 0.92rem; }}
    .site-nav a:hover, .site-nav a[aria-current="page"] {{ color: var(--red); }}
    .site-nav a[aria-current="page"] {{ font-weight: 600; }}
    nav.toc {{ position: sticky; top: 18px; padding: 18px 0; border-top: 2px solid var(--ink); }}
    nav.toc a {{ display: block; padding: 8px 0; color: #344250; text-decoration: none; font-size: 0.95rem; border-bottom: 1px solid rgba(22, 33, 44, 0.1); }}
    nav.toc a:hover {{ color: var(--red); }}
    section {{ padding: 58px 0; border-top: 1px solid var(--line); }}
    section:first-child {{ border-top: 0; padding-top: 0; }}
    .lead {{ font-size: 1.17rem; color: #2e3c49; max-width: 860px; }}
    .callout {{ background: #fff8ea; border-left: 5px solid var(--gold); padding: 18px 20px; border-radius: var(--radius); margin: 24px 0; }}
    .formula {{ background: #16212c; color: #f7f3e9; border-radius: var(--radius); padding: 20px; overflow-x: auto; font: 500 0.96rem/1.7 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .figure {{ background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: 0 14px 40px rgba(22, 33, 44, 0.08); padding: 14px; margin: 28px 0; }}
    .grid-two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin: 22px 0; }}
    .note-card {{ background: rgba(255, 255, 255, 0.72); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px; }}
    .table-wrap {{ overflow-x: auto; margin: 22px 0 30px; border: 1px solid var(--line); border-radius: var(--radius); background: white; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 860px; font-size: 0.94rem; }}
    th, td {{ padding: 11px 13px; text-align: right; border-bottom: 1px solid #e9edee; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; font-weight: 600; }}
    th {{ background: #f0f3ef; color: #32404d; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; }}
    tr:last-child td {{ border-bottom: 0; }}
    blockquote {{ margin: 34px 0; padding: 4px 0 4px 24px; border-left: 5px solid var(--red); font-family: "Source Serif 4", Georgia, serif; font-size: clamp(1.35rem, 2.2vw, 2rem); line-height: 1.22; }}
    ul, ol {{ padding-left: 22px; }}
    li {{ margin: 7px 0; }}
    footer {{ color: var(--muted); border-top: 1px solid var(--line); padding-top: 28px; margin-top: 40px; font-size: 0.95rem; }}
    @media (max-width: 920px) {{
      .hero, .layout, .grid-two {{ grid-template-columns: 1fr; }}
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
        <a href="index.html">Allocation report</a>
        <a href="credit-line.html" aria-current="page">Credit-line report</a>
        <a href="https://github.com/actions-im/retirement_planning">GitHub</a>
      </div>
    </div>
  </nav>
  <div class="page">
    <header class="hero">
      <div>
        <p class="eyebrow">Retirement Planning Report · May 16, 2026</p>
        <h1>Can A Hybrid Credit Line Reduce Forced Selling?</h1>
        <p class="dek">A monthly S&amp;P 500 total-return simulation testing whether an asset-backed credit line can reduce forced selling during drawdowns while still letting discretionary spending flex down.</p>
      </div>
      <aside class="hero-panel">
        <img src="assets/credit_line_objective_tradeoff.png" alt="Objective tradeoff chart for sell-only and credit-line strategies.">
        <p class="caption">The rebuilt model borrows less: the loan funds the normal target/cap/floor withdrawal instead of forcing full target spending through every drawdown.</p>
      </aside>
    </header>

    <div class="metric-strip">
      <div class="metric"><strong>{pct(baseline['target_shortfall_pct'])}</strong><span>target-shortfall path-months for 5% sell-only stock exposure</span></div>
      <div class="metric"><strong>{pct(best_shortfall['target_shortfall_pct'])}</strong><span>best 5% hybrid-credit target-shortfall result</span></div>
      <div class="metric"><strong>{pct(lower_spend_baseline['target_shortfall_pct'])}</strong><span>target-shortfall path-months for 4% sell-only stock exposure</span></div>
      <div class="metric"><strong>{pct(lower_spend_best_shortfall['target_shortfall_pct'])}</strong><span>best 4% hybrid-credit target-shortfall result</span></div>
    </div>

    <div class="layout">
      <nav class="toc" aria-label="Report sections">
        <a href="#summary">Summary</a>
        <a href="#setup">Setup</a>
        <a href="#hybrid">Hybrid Rule</a>
        <a href="#results">Results</a>
        <a href="#debt">Debt Risk</a>
        <a href="#interpretation">Interpretation</a>
        <a href="#limits">Limitations</a>
      </nav>
      <main>
        <section id="summary">
          <h2>Executive Summary</h2>
          <p class="lead">The credit-line strategy is a substitute for a cash buffer, not a free source of retirement income. It tries to solve one specific problem: do not sell equities after the market has already fallen.</p>
          <p>The first version of this experiment was too aggressive: once the market crossed the drawdown trigger, it borrowed full target spending. That protected lifestyle on paper, but it also ignored the core rule of the retirement plan: discretionary spending should fall when wealth falls.</p>
          <p>This rebuild uses a hybrid rule. The retiree still follows the target/cap/floor spending rule. If the market is down enough, the credit line funds that reduced withdrawal instead of selling shares. When the index recovers to the pre-trigger high, the loan is repaid by selling shares.</p>
          <p>That produces the better version of the idea. At the 5% target and 20% trigger, the old target-protecting model has {pct(target_protecting_20['ruin_pct'])} ruin and {pct(target_protecting_20['margin_event_pct'])} margin-event risk. The hybrid model has {pct(hybrid_20['ruin_pct'])} ruin and {pct(hybrid_20['margin_event_pct'])} margin-event risk because it borrows less.</p>
          <blockquote>The credit line should not be used to pretend the drawdown did not happen. It should be used, if at all, to avoid selling shares while the spending rule already does its job.</blockquote>
        </section>

        <section id="setup">
          <h2>Setup</h2>
          <p>This report uses monthly data because credit-line triggers and paybacks are path-dependent. Annual data is too coarse for this specific question. It does not simulate a T-bill cash buffer or a 60/40 sleeve, so it cannot answer whether the line replaces cash or bonds. The 5% sell-only ruin here is also not the {1.21:.2f}% figure from the annual allocation report: the data window starts in 1988, the steps are monthly, and ruin is an ever-hit-zero rate rather than terminal wealth.</p>
          <div class="grid-two">
            <div class="note-card">
              <h3>Data</h3>
              <ul>
                <li>S&amp;P 500 Total Return: Yahoo/yfinance <code>^SP500TR</code>.</li>
                <li>Inflation: FRED monthly CPI-U, <code>CPIAUCSL</code>.</li>
                <li>Aligned completed monthly window: {results['settings']['data_start']} to {results['settings']['data_end']}.</li>
                <li>{results['settings']['n_months']} paired monthly records.</li>
              </ul>
            </div>
            <div class="note-card">
              <h3>Simulation</h3>
              <ul>
                <li>{N_PATHS:,} random 30-year paths.</li>
                <li>5-year monthly blocks sampled with replacement.</li>
                <li>Two spending paths: 5% target / 2.5% floor and 4% target / 2% floor, all real.</li>
                <li>Credit cost: CPI + 2%, modeled as 2% real.</li>
                <li>Maximum loan-to-value: 25% of portfolio assets.</li>
              </ul>
            </div>
          </div>
          <pre class="formula">target = 5.0% or 4.0% of initial portfolio / 12 each month
floor = 50% of target spending
cap = target rate / 12 of current real net wealth

normal spending = min(net wealth, max(floor, min(target, cap)))

if drawdown trigger is active:
    borrow normal spending when collateral capacity is available
    sell shares only for any unfunded remainder
    repay debt after the total-return index recovers to the pre-trigger high</pre>
          <figure class="figure">
            <img src="assets/credit_line_mechanics.png" alt="Credit-line mechanics diagram.">
            <figcaption>The model treats borrowing as a temporary bridge, then explicitly repays the balance after recovery.</figcaption>
          </figure>
        </section>

        <section id="hybrid">
          <h2>The Hybrid Rule</h2>
          <p>The key change is that borrowing no longer overrides the lifestyle cut. If the portfolio is below its starting value, the cap already reduces discretionary spending. The loan can fund that smaller withdrawal, but it cannot force spending back to the full target.</p>
          <p>This matters because the old target-protecting model mixed two ideas together: avoid selling after a drawdown and maintain full lifestyle spending after a drawdown. The hybrid model tests only the first idea.</p>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Scenario</th><th>Strategy</th><th>Funding rule at 20% trigger</th><th>Ruin</th><th>Target shortfall</th><th>Floor breach</th><th>Real final p10</th><th>Real final median</th><th>Margin event</th></tr>
              </thead>
              <tbody>
                {borrow_rule_comparison_table(scenarios, target_protecting_scenarios)}
              </tbody>
            </table>
          </div>
          <p>The comparison makes the tradeoff explicit. The old credit model can show fewer target-shortfall months because it borrows enough to keep spending near target. The hybrid model gives up some of that lifestyle smoothing, but it reduces debt pressure and keeps the retirement rule internally consistent.</p>
        </section>

        <section id="results">
          <h2>Results</h2>
          <p>The tables compare the sell-only baseline against the hybrid cap-respecting credit line triggered at 10%, 20%, 30%, and 40% drawdowns from the S&amp;P 500 Total Return high-water mark. The 4% path is the same strategy with lower spending pressure.</p>
          {scenario_result_sections(scenarios)}
          <figure class="figure">
            <img src="assets/credit_line_trigger_search.png" alt="Credit-line trigger search chart.">
            <figcaption>The dotted lines show the sell-only baseline. Credit-line triggers are only useful if their improvement in shortfall is worth the added debt and forced-sale risk.</figcaption>
          </figure>
          <figure class="figure">
            <img src="assets/credit_line_objective_tradeoff.png" alt="Objective tradeoff chart.">
            <figcaption>The objective is left-and-up: lower target shortfall and higher real ending net wealth.</figcaption>
          </figure>
        </section>

        <section id="debt">
          <h2>Debt Risk</h2>
          <p>Borrowing improves lifestyle only when the line is actually used. But more use also means more months with debt outstanding and more exposure to collateral limits.</p>
          {scenario_debt_sections(scenarios)}
          <p>The 95th percentile peak debt is shown as a percentage of the starting portfolio. With a $5M starting portfolio, 10% peak debt means a $500,000 real loan balance.</p>
        </section>

        <section id="interpretation">
          <h2>Interpretation</h2>
          <p>The hybrid credit line is most attractive when it is used rarely, at moderate drawdowns, and repaid after a strong recovery. It is least attractive when markets continue falling after borrowing begins. That is exactly when an asset-backed lender can reduce available credit, raise collateral requirements, or force sales.</p>
          <p>Compared with cash and bonds, the hybrid line preserves upside because the portfolio stays invested until a drawdown actually happens. Compared with the old target-protecting loan rule, it is less fragile because it borrows a smaller amount and lets discretionary spending fall.</p>
          <div class="callout"><b>Bottom line:</b> a securities-backed credit line can be a reasonable tactical liquidity tool when it respects the spending rule. It is not a clean replacement for cash or bonds, but the hybrid version is the right version to test: lower expected drag than a permanent buffer, less leverage risk than borrowing full target spending.</div>
        </section>

        <section id="limits">
          <h2>Limitations And Next Tests</h2>
          <ul>
            <li>The direct total-return data begins in 1988, so the bootstrap window is much shorter than the annual 75-year report.</li>
            <li>Five-year blocks on a 1988-present monthly sample can overfit modern market history.</li>
            <li>Taxes, asset-backed lending fees, variable spreads, and lender discretion are ignored.</li>
            <li>The model assumes credit remains available until the LTV limit is hit; real lenders can change terms earlier.</li>
            <li>The model uses S&amp;P 500 Total Return as the whole portfolio, not a diversified taxable account with concentrated-position haircuts.</li>
            <li>The next useful tests are different LTV caps, 1%-4% real credit spreads, 35- and 40-year horizons, and a more defensive rule that borrows only floor spending.</li>
          </ul>
          <footer>
            Data sources: Yahoo/yfinance <code>^SP500TR</code> and FRED <code>CPIAUCSL</code>.
            The sibling annual report, <a href="index.html">The Measurable Tradeoffs Behind The 4% Rule</a>, uses Damodaran 1951–2025 stock/bond/T-bill blocks and is not numerically comparable to these monthly paths.
          </footer>
        </section>
      </main>
    </div>
  </div>
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    history = fetch_monthly_history()
    paths = sample_monthly_paths(history)

    scenarios = build_scenarios(paths, MAIN_BORROW_RULE)
    target_protecting_scenarios = build_scenarios(paths, COMPARISON_BORROW_RULE)

    results = {
        "settings": {
            "base_initial_net_worth": BASE,
            "years": YEARS,
            "months": MONTHS,
            "n_paths": N_PATHS,
            "block_months": BLOCK_MONTHS,
            "seed": SEED,
            "target_rates": TARGET_RATES,
            "floor_rates": [target_rate * FLOOR_RATIO for target_rate in TARGET_RATES],
            "floor_ratio": FLOOR_RATIO,
            "real_credit_spread": REAL_CREDIT_SPREAD,
            "max_ltv": MAX_LTV,
            "main_borrow_rule": MAIN_BORROW_RULE,
            "comparison_borrow_rule": COMPARISON_BORROW_RULE,
            "data_start": history.start_month,
            "data_end": history.end_month,
            "n_months": history.n_months,
            "stock_source": "Yahoo/yfinance ^SP500TR, S&P 500 Total Return Index",
            "inflation_source": "FRED CPIAUCSL monthly CPI-U",
        },
        "scenarios": scenarios,
        "target_protecting_scenarios": target_protecting_scenarios,
        "rows": scenarios[0]["rows"],
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")
    plot_mechanics()
    plot_tradeoff(scenarios)
    plot_objective(scenarios)
    write_report()


if __name__ == "__main__":
    if "--render-only" in sys.argv:
        write_report()
    else:
        main()
