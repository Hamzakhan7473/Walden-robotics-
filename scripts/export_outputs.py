"""Write the v2 review tables and figures.

Run from the repo root:

    python scripts/export_outputs.py

The solver is PuLP's bundled CBC (``PULP_CBC_CMD``), the same binary the
package ships on macOS and Linux. A system CBC on ``PATH`` is not used.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pulp

from src.allocate import allocate, load_capacity
from src.scoring import V2_CRITERIA, load_requests, score_requests
from src.sensitivity import ROBUST_HIGH, ROBUST_LOW, run_sensitivity

OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "reports" / "figures"
REQUESTS_PATH = ROOT / "data" / "requests.csv"
CAPACITY_PATH = ROOT / "data" / "capacity.csv"

INK = "#1C1917"
RUST = "#9A3412"
CLAY = "#C2410C"
STONE = "#A8A29E"
SLATE = "#57534E"
PAPER = "#FFFFFF"
CONTRIB_COLORS = {
    "commercial": RUST,
    "learning": CLAY,
    "readiness": STONE,
    "strategic": SLATE,
    "success": "#78716C",
    "cost": "#44403C",
}
STABILITY_COLORS = {
    "robust_yes": RUST,
    "contested": "#C4B8A5",
    "robust_no": "#D6D3D1",
}


def main() -> None:
    _require_bundled_cbc()
    if not REQUESTS_PATH.exists() or not CAPACITY_PATH.exists():
        raise FileNotFoundError("data/requests.csv and data/capacity.csv are required. Run make data.")

    requests = load_requests(REQUESTS_PATH)
    capacity = load_capacity(CAPACITY_PATH)
    scored = score_requests(requests, version="v2")
    result = allocate(scored, capacity, exploration_floor=0.20, version="v2")
    scored_v1 = score_requests(requests, version="v1")
    result_v1 = allocate(scored_v1, capacity, exploration_floor=0.20, version="v1")
    sensitivity = run_sensitivity(
        requests,
        capacity,
        n_sims=200,
        sigma=0.06,
        seed=42,
        exploration_floor=0.20,
        version="v2",
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    scored_out = _scored_table(scored)
    allocation_out = _allocation_table(result.decisions)
    utilization_out = _utilization_table(result.weekly)
    sensitivity_out = _sensitivity_table(sensitivity.by_request)
    comparison_out = _comparison_table(result_v1.decisions, result.decisions)

    scored_out.to_csv(OUTPUTS / "scored_requests.csv", index=False)
    allocation_out.to_csv(OUTPUTS / "allocation.csv", index=False)
    utilization_out.to_csv(OUTPUTS / "utilization_by_week.csv", index=False)
    sensitivity_out.to_csv(OUTPUTS / "sensitivity.csv", index=False)
    comparison_out.to_csv(OUTPUTS / "v1_vs_v2.csv", index=False)

    _plot_utilization(utilization_out, FIGURES / "weekly_utilization.png")
    _plot_sensitivity(sensitivity_out, FIGURES / "sensitivity_bars.png")
    _plot_funded_diff(comparison_out, FIGURES / "v1_vs_v2_funded.png")
    _plot_contributions(scored, FIGURES / "score_contributions.png")

    print(f"wrote {OUTPUTS}")
    print(f"wrote {FIGURES}")


def _require_bundled_cbc() -> None:
    available = pulp.listSolvers(onlyAvailable=True)
    if "PULP_CBC_CMD" not in available:
        raise RuntimeError(
            "PuLP's bundled CBC solver is not available. "
            "Install pulp==2.9.0 so PULP_CBC_CMD ships with the package."
        )


def _scored_table(scored: pd.DataFrame) -> pd.DataFrame:
    columns = ["request_id", "customer_name", "gate_pass", "gate_reason", "score", "score_per_fde_hour"]
    for name in V2_CRITERIA:
        columns.append(f"criterion_{name}")
        columns.append(f"contrib_{name}")
    return scored[columns].sort_values(["gate_pass", "score"], ascending=[False, False]).reset_index(drop=True)


def _allocation_table(decisions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "request_id",
        "customer_name",
        "decision",
        "decision_reason",
        "in_horizon_robot_weeks",
        "score",
    ]
    return decisions[columns].reset_index(drop=True)


def _utilization_table(weekly: pd.DataFrame) -> pd.DataFrame:
    frame = weekly.copy()
    frame["robots_pct"] = (frame["robots_utilization"] * 100).round(1)
    frame["fde_pct"] = (frame["fde_utilization"] * 100).round(1)
    return frame[
        [
            "week",
            "robots_used",
            "robots_available",
            "robots_pct",
            "fde_hours_used",
            "fde_hours_available",
            "fde_pct",
        ]
    ].reset_index(drop=True)


def _sensitivity_table(by_request: pd.DataFrame) -> pd.DataFrame:
    frame = by_request.copy()
    frame["stability"] = frame["funded_rate"].map(_stability)
    columns = ["request_id", "customer_name", "funded_rate", "stability"]
    if "baseline_decision" in frame.columns:
        columns.append("baseline_decision")
    return frame[columns].sort_values(
        ["funded_rate", "request_id"], ascending=[False, True]
    ).reset_index(drop=True)


def _stability(rate: float) -> str:
    if float(rate) >= ROBUST_HIGH:
        return "robust_yes"
    if float(rate) <= ROBUST_LOW:
        return "robust_no"
    return "contested"


def _comparison_table(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    left = v1.set_index("request_id")
    right = v2.set_index("request_id")
    rows = []
    for request_id in left.index:
        in_v1 = left.loc[request_id, "decision"] == "funded"
        in_v2 = right.loc[request_id, "decision"] == "funded"
        if in_v1 and in_v2:
            status = "both"
        elif in_v1:
            status = "v1_only"
        elif in_v2:
            status = "v2_only"
        else:
            status = "neither"
        rows.append(
            {
                "request_id": request_id,
                "customer_name": left.loc[request_id, "customer_name"],
                "in_v1": bool(in_v1),
                "in_v2": bool(in_v2),
                "set_status": status,
                "v1_in_horizon_robot_weeks": float(left.loc[request_id, "in_horizon_robot_weeks"]) if in_v1 else 0.0,
                "v2_in_horizon_robot_weeks": float(right.loc[request_id, "in_horizon_robot_weeks"]) if in_v2 else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    order = {"v1_only": 0, "v2_only": 1, "both": 2, "neither": 3}
    frame["_ord"] = frame["set_status"].map(order)
    return frame.sort_values(["_ord", "request_id"]).drop(columns="_ord").reset_index(drop=True)


def _plot_utilization(weekly: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=PAPER)
    weeks = weekly["week"].astype(int)
    ax.plot(weeks, weekly["robots_pct"], color=RUST, marker="o", linewidth=1.8, label="Robots")
    ax.plot(weeks, weekly["fde_pct"], color=SLATE, marker="o", linewidth=1.8, label="FDE hours")
    ax.set_xlabel("Week")
    ax.set_ylabel("Capacity used (%)")
    ax.set_title("Weekly utilization, v2 plan")
    ax.set_xticks(list(weeks))
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right")
    _clean(ax)
    _save(fig, path)


def _plot_sensitivity(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("funded_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 9.2), facecolor=PAPER)
    colors = [STABILITY_COLORS[label] for label in ordered["stability"]]
    ax.barh(ordered["request_id"], ordered["funded_rate"] * 100, color=colors, height=0.72)
    ax.set_xlabel("Funded share of weight draws (%)")
    ax.set_ylabel("Request")
    ax.set_title("Funding frequency under weight noise")
    ax.set_xlim(0, 100)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STABILITY_COLORS["robust_yes"], label="Robust yes (≥ 80%)"),
        plt.Rectangle((0, 0), 1, 1, color=STABILITY_COLORS["contested"], label="Contested"),
        plt.Rectangle((0, 0), 1, 1, color=STABILITY_COLORS["robust_no"], label="Robust no (≤ 20%)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _clean(ax)
    _save(fig, path)


def _plot_funded_diff(comparison: pd.DataFrame, path: Path) -> None:
    funded = comparison[comparison["set_status"] != "neither"].copy()
    funded = funded.sort_values("request_id")
    labels = funded["request_id"].tolist()
    y = range(len(funded))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 6.2), facecolor=PAPER)
    ax.barh(
        [value + height / 2 for value in y],
        funded["v1_in_horizon_robot_weeks"],
        height=height,
        color=STONE,
        label="v1",
    )
    ax.barh(
        [value - height / 2 for value in y],
        funded["v2_in_horizon_robot_weeks"],
        height=height,
        color=RUST,
        label="v2",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("In-quarter robot-weeks")
    ax.set_ylabel("Funded in v1, v2, or both")
    ax.set_title("Funded set, in-quarter robot-weeks")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _clean(ax)
    _save(fig, path)


def _plot_contributions(scored: pd.DataFrame, path: Path) -> None:
    eligible = scored[scored["gate_pass"]].nlargest(15, "score").sort_values("score", ascending=False)
    labels = eligible["request_id"].tolist()
    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor=PAPER)
    bottom = [0.0] * len(eligible)
    x = range(len(eligible))
    for name in V2_CRITERIA:
        heights = eligible[f"contrib_{name}"].astype(float).tolist()
        ax.bar(x, heights, bottom=bottom, color=CONTRIB_COLORS[name], width=0.72, label=name.capitalize())
        bottom = [base + height for base, height in zip(bottom, heights)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Contribution to score")
    ax.set_xlabel("Request")
    ax.set_title("Score contributions, top 15 eligible requests")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    _clean(ax)
    _save(fig, path)


def _clean(ax) -> None:
    ax.set_facecolor(PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, length=0)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
