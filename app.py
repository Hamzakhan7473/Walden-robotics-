"""Review the deployment portfolio, the binding weeks, and weight stability.

Run from the repo root:

    streamlit run app.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.allocate import allocate, load_capacity
from src.scoring import (
    VALUE_BLEND,
    EXPANSION_BLEND,
    criteria_for,
    default_weights,
    explain_score,
    load_requests,
    score_requests,
    strategic_map,
)
from src.sensitivity import run_sensitivity

ROOT = Path(__file__).resolve().parent
REQUESTS_PATH = ROOT / "data" / "requests.csv"
CAPACITY_PATH = ROOT / "data" / "capacity.csv"
GITHUB_URL = "https://github.com/Hamzakhan7473/Walden-robotics-"

SEGMENT_LABEL = {
    "auto_oem": "Auto OEM",
    "tier1_supplier": "Tier-1 supplier",
    "3pl_warehouse": "3PL warehouse",
    "electronics": "Electronics",
    "food_bev": "Food & beverage",
}
TASK_LABEL = {
    "machine_tending": "Machine tending",
    "kitting": "Kitting",
    "palletizing": "Palletizing",
    "bin_picking": "Bin picking",
    "inspection": "Inspection",
    "packing": "Packing",
    "assembly_assist": "Assembly assist",
}
TIER_LABEL = {
    "anchor": "Anchor",
    "investor_linked": "Investor-linked",
    "reference_logo": "Reference logo",
    "standard": "Standard",
}
WEIGHT_LABEL = {
    "commercial": "Commercial",
    "learning": "Learning",
    "success": "Success",
    "readiness": "Readiness",
    "strategic": "Strategic",
    "cost": "Cost",
}
WEIGHT_HELP = {
    "v1": {
        "commercial": "Contract dollars per robot-week, blended with expansion potential.",
        "learning": "Min-max of task novelty inside this eligible book. Not the raw novelty used by the floor.",
        "success": "Min-max of success probability.",
        "readiness": "Min-max of site readiness.",
        "strategic": "Fixed tier map. Not min-maxed.",
        "cost": "Penalty on FDE hours per robot-week.",
    },
    "v2": {
        "commercial": "Normalized value density times success probability, blended with capped log expansion.",
        "learning": "Raw task novelty. Not min-maxed.",
        "readiness": "(site readiness − 1) / 4.",
        "strategic": "Fixed tier map. Standard is zero.",
    },
}
WEIGHT_BOUNDS = {
    "commercial": (0.0, 0.60, 0.01),
    "learning": (0.0, 0.60, 0.01),
    "success": (0.0, 0.60, 0.01),
    "readiness": (0.0, 0.60, 0.01),
    "strategic": (0.0, 0.60, 0.01),
    "cost": (-0.30, 0.0, 0.01),
}

PORTFOLIO_COLUMNS = [
    "request_id",
    "customer_name",
    "segment_label",
    "task_label",
    "tier_label",
    "robots_requested",
    "duration_weeks",
    "requested_start_week",
    "task_novelty",
    "contract_value_usd",
    "score",
    "score_per_fde_hour",
    "in_horizon_robot_weeks",
    "decision",
    "decision_reason",
]


def _requested_version() -> str:
    """Read ``--version`` from the process args.

    Streamlit keeps its own flags in front of ``--`` and passes the rest
    through: ``streamlit run app.py -- --version v1``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    known, _ = parser.parse_known_args()
    return known.version


def main() -> None:
    st.set_page_config(page_title="Walden allocation", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    _ensure_data()

    if "policy_version" not in st.session_state:
        st.session_state["policy_version"] = _requested_version()

    requests, capacity = _load(str(REQUESTS_PATH), str(CAPACITY_PATH), REQUESTS_PATH.stat().st_mtime, CAPACITY_PATH.stat().st_mtime)
    version, weights = _sidebar()
    floor = st.session_state["exploration_floor"]
    threshold = st.session_state.get("novelty_threshold", 0.60)

    scored = score_requests(requests, weights, version=version)
    result = allocate(
        scored,
        capacity,
        exploration_floor=floor,
        novelty_threshold=threshold,
        version=version,
    )
    decisions = _labeled(result.decisions)

    st.title("Deployment allocation")
    st.caption(
        f"{version} · {len(requests)} synthetic requests · seed 42 · fictional customers. "
        "Each deployment pays and adds to the skill library. The fleet and FDE hours both run out."
    )

    about, portfolio, requests_tab, capacity_tab, sensitivity_tab, method_tab = st.tabs(
        ["About", "Portfolio", "Score audit", "Capacity", "Sensitivity", "Method"]
    )
    with about:
        _render_about()
    with portfolio:
        _render_portfolio(decisions, result)
    with requests_tab:
        _render_audit(decisions, weights, version)
    with capacity_tab:
        _render_capacity(result)
    with sensitivity_tab:
        _render_sensitivity(requests, capacity, weights, floor, threshold, version)
    with method_tab:
        _render_method(version)


def _ensure_data() -> None:
    """Write the seed-42 book when the CSVs are not already in the repo."""
    if REQUESTS_PATH.exists() and CAPACITY_PATH.exists():
        return
    from data.generate_data import SEED, build_capacity, build_requests

    REQUESTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_requests(seed=SEED).to_csv(REQUESTS_PATH, index=False)
    build_capacity().to_csv(CAPACITY_PATH, index=False)


def _render_about() -> None:
    st.markdown(
        "\n\n".join(
            [
                "This framework ranks deployment requests, then funds a subset when two weekly capacities cannot cover the book.",
                "The scarce resources are deployable robot units and Field Deployment Engineer hours. A deployment returns contract revenue and training data, so the choice is an explore/exploit portfolio.",
                "Safety review, a signed letter of intent, and site readiness of at least 2 are hard gates. Every other request gets a weighted score that keeps each criterion's contribution.",
                "An integer program then picks the funded set for the next 12 weeks. Only robot-weeks inside that horizon count, and their novelty-weighted average has a floor.",
                f"The request book is synthetic, seed 42. Source: [Walden Robotics allocation]({GITHUB_URL}).",
            ]
        )
    )


def _sidebar() -> tuple[str, dict[str, float]]:
    st.sidebar.header("Policy")
    version = st.sidebar.selectbox(
        "Version",
        options=["v2", "v1"],
        key="policy_version",
        help="v2 is the revised score and in-quarter floor. v1 is the original min-max score and the 0.6 novelty cut. The flag --version sets this on first load.",
    )
    defaults = default_weights(version)
    names = criteria_for(version)
    if st.sidebar.button("Reset to defaults", width="stretch"):
        for name, value in defaults.items():
            st.session_state[f"weight_{version}_{name}"] = float(value)
        st.session_state["exploration_floor"] = 0.20
        st.session_state["novelty_threshold"] = 0.60

    st.sidebar.subheader("Weights")
    weights = {}
    for name in names:
        low, high, step = WEIGHT_BOUNDS[name]
        key = f"weight_{version}_{name}"
        if key not in st.session_state:
            st.session_state[key] = float(defaults[name])
        weights[name] = st.sidebar.slider(
            WEIGHT_LABEL[name],
            min_value=low,
            max_value=high,
            step=step,
            key=key,
            help=WEIGHT_HELP[version][name],
        )
    if version == "v1":
        positive = sum(weights[name] for name in names if name != "cost")
        st.sidebar.caption(
            f"Positive weights sum to {positive:.2f}. Cost weight is {weights['cost']:.2f}. "
            "They are applied as entered and are not rescaled."
        )
    else:
        st.sidebar.caption(
            f"Weights sum to {sum(weights.values()):.2f}. "
            "They are applied as entered and are not rescaled."
        )

    st.sidebar.subheader("Exploration floor")
    if "exploration_floor" not in st.session_state:
        st.session_state["exploration_floor"] = 0.20
    if "novelty_threshold" not in st.session_state:
        st.session_state["novelty_threshold"] = 0.60
    floor_help = (
        "Minimum share of funded robot-weeks that must come from tasks at or above the novelty threshold."
        if version == "v1"
        else "Minimum novelty-weighted average of funded in-quarter robot-weeks."
    )
    st.sidebar.slider(
        "Floor",
        min_value=0.0,
        max_value=0.80,
        step=0.05,
        key="exploration_floor",
        help=floor_help,
    )
    if version == "v1":
        st.sidebar.slider(
            "Novelty threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="novelty_threshold",
            help="Raw task novelty. This is not the min-maxed learning score.",
        )
    return version, weights


def _render_portfolio(decisions: pd.DataFrame, result) -> None:
    if result.exploration_floor_relaxed:
        st.warning(result.floor_note)
    elif result.total_robot_weeks <= 0:
        st.info("Nothing was funded.")
    else:
        if result.version == "v1":
            st.caption(
                f"Exploration floor of {result.exploration_floor:.0%} is met: "
                f"{result.exploration_share:.0%} of funded robot-weeks have task novelty "
                f"at or above {result.novelty_threshold:.2f}."
            )
        else:
            st.caption(
                f"In-quarter weighted task novelty is {result.exploration_share:.0%}, "
                f"against a floor of {result.exploration_floor:.0%}. "
                "Weeks after this plan are not credited."
            )

    funded = decisions[decisions["decision"] == "funded"]
    waitlisted = decisions[decisions["decision"] == "waitlisted"]
    peak_robots = float(result.weekly["robots_utilization"].max())
    peak_fde = float(result.weekly["fde_utilization"].max())
    booked = float(funded["contract_value_usd"].sum()) if len(funded) else 0.0

    top = st.columns(3)
    top[0].metric("Funded", f"{len(funded)} of {len(decisions)}")
    weeks_label = "Robot-weeks" if result.version == "v1" else "In-quarter robot-weeks"
    top[1].metric(weeks_label, f"{result.total_robot_weeks:.0f}")
    top[2].metric("Exploration share", f"{result.exploration_share:.0%}")
    bottom = st.columns(3)
    bottom[0].metric("Booked value", _money(booked), help="Sum of contract value on funded requests. Not the objective.")
    bottom[1].metric("Peak robot use", f"{peak_robots:.0%}")
    bottom[2].metric("Peak FDE use", f"{peak_fde:.0%}")

    if len(waitlisted):
        nxt = waitlisted.iloc[0]
        st.info(
            f"Highest-scoring request left out: {nxt['customer_name']} ({nxt['request_id']}), "
            f"score {nxt['score']:.3f}. {nxt['decision_reason']}"
        )

    view = decisions[PORTFOLIO_COLUMNS].copy()
    view["score"] = view["score"].round(3)
    view["task_novelty"] = view["task_novelty"].round(2)
    view["score_per_fde_hour"] = view["score_per_fde_hour"].round(5)
    view["in_horizon_robot_weeks"] = view["in_horizon_robot_weeks"].round(0)
    st.dataframe(
        view,
        column_config={
            "request_id": "Request",
            "customer_name": "Customer",
            "segment_label": "Segment",
            "task_label": "Task",
            "tier_label": "Tier",
            "robots_requested": st.column_config.NumberColumn("Robots", format="%d"),
            "duration_weeks": st.column_config.NumberColumn("Weeks", format="%d"),
            "requested_start_week": st.column_config.NumberColumn("Start", format="%d"),
            "task_novelty": st.column_config.NumberColumn("Novelty", format="%.2f"),
            "contract_value_usd": st.column_config.NumberColumn("Contract", format="$%d"),
            "score": st.column_config.NumberColumn("Score", format="%.3f"),
            "score_per_fde_hour": st.column_config.NumberColumn("Score / FDE hour", format="%.5f"),
            "in_horizon_robot_weeks": st.column_config.NumberColumn("In-quarter robot-weeks", format="%.0f"),
            "decision": "Decision",
            "decision_reason": st.column_config.TextColumn("Reason", width="large"),
        },
        hide_index=True,
        width="stretch",
        height=460,
    )
    st.download_button(
        "Download decisions",
        data=result_csv(decisions),
        file_name="decisions.csv",
        mime="text/csv",
    )


def _render_audit(decisions: pd.DataFrame, weights: dict[str, float], version: str) -> None:
    if version == "v1":
        st.caption(
            "The score is the sum of the contributions. Min-max is fit on the eligible book only. "
            "A failed gate has no score."
        )
    else:
        st.caption(
            "The score is the sum of the contributions. Learning and readiness use fixed scales. "
            "Value density and expansion are still scaled on the eligible book. "
            "Score per FDE hour is a diagnostic and is not in the objective."
        )
    names = dict(zip(decisions["request_id"], decisions["customer_name"]))
    options = decisions["request_id"].tolist()
    selected = st.selectbox(
        "Request",
        options,
        format_func=lambda request_id: f"{request_id} · {names[request_id]}",
    )
    row = decisions.set_index("request_id").loc[selected]
    meta = st.columns(4)
    meta[0].markdown(f"**{row['decision'].title()}**")
    meta[1].markdown(f"{row['segment_label']} · {row['task_label']}")
    meta[2].markdown(f"{int(row['robots_requested'])} robots · {int(row['duration_weeks'])} weeks")
    meta[3].markdown(f"Starts week {int(row['requested_start_week'])}")
    st.write(row["decision_reason"])

    if not bool(row["gate_pass"]):
        st.warning(explain_score(row, weights))
        return

    st.write(explain_score(row, weights, version=version))
    criteria_names = criteria_for(version)
    parts = pd.DataFrame(
        [
            {
                "criterion": name,
                "weight": weights[name],
                "normalized": float(row[f"criterion_{name}"]),
                "contribution": float(row[f"contrib_{name}"]),
            }
            for name in criteria_names
        ]
    )
    left, right = st.columns([1.1, 1])
    with left:
        chart = (
            alt.Chart(parts)
            .mark_bar()
            .encode(
                x=alt.X("contribution:Q", title="Contribution to score"),
                y=alt.Y("criterion:N", sort=list(criteria_names), title=None),
                color=alt.condition(
                    alt.datum.contribution >= 0,
                    alt.value("#9A3412"),
                    alt.value("#44403C"),
                ),
                tooltip=[
                    "criterion",
                    alt.Tooltip("weight:Q", format="+.2f"),
                    alt.Tooltip("normalized:Q", format=".2f"),
                    alt.Tooltip("contribution:Q", format="+.3f"),
                ],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")
    with right:
        raw = pd.DataFrame(
            [
                {"input": "Value density ($ / robot-week)", "value": f"{row['value_density']:,.0f}"},
                {"input": "Expansion potential (units)", "value": f"{int(row['expansion_units_potential'])}"},
                {"input": "Task novelty (raw)", "value": f"{row['task_novelty']:.2f}"},
                {"input": "Success probability", "value": f"{row['success_prob']:.2f}"},
                {"input": "Site readiness", "value": f"{int(row['site_readiness'])}"},
                {"input": "Strategic tier", "value": row["tier_label"]},
                {"input": "FDE hours / robot-week", "value": f"{row['fde_per_robot_week']:.1f}"},
                {"input": "Score per FDE hour", "value": f"{row['score_per_fde_hour']:.5f}"},
                {"input": "In-quarter robot-weeks", "value": f"{row['in_horizon_robot_weeks']:.0f}"},
            ]
        )
        st.dataframe(raw, hide_index=True, width="stretch")

    st.dataframe(
        parts,
        column_config={
            "criterion": "Criterion",
            "weight": st.column_config.NumberColumn("Weight", format="+.2f"),
            "normalized": st.column_config.NumberColumn("Normalized", format="%.2f"),
            "contribution": st.column_config.NumberColumn("Contribution", format="%+.3f"),
        },
        hide_index=True,
        width="stretch",
    )


def _render_capacity(result) -> None:
    weekly = result.weekly
    st.caption("Bars are hours or robots in use. The dashed line is what is available that week.")
    left, right = st.columns(2)
    with left:
        st.altair_chart(
            _utilization_chart(weekly, "robots_used", "robots_available", "Robots", "Robots"),
            width="stretch",
        )
    with right:
        st.altair_chart(
            _utilization_chart(weekly, "fde_hours_used", "fde_hours_available", "FDE hours", "Hours"),
            width="stretch",
        )
    show = weekly.copy()
    show["robots_used"] = show["robots_used"].round(1)
    show["fde_hours_used"] = show["fde_hours_used"].round(1)
    show["robots_utilization"] = show["robots_utilization"].map(lambda value: f"{value:.0%}")
    show["fde_utilization"] = show["fde_utilization"].map(lambda value: f"{value:.0%}")
    st.dataframe(
        show[
            [
                "week",
                "robots_used",
                "robots_available",
                "robots_utilization",
                "fde_hours_used",
                "fde_hours_available",
                "fde_utilization",
            ]
        ],
        column_config={
            "week": "Week",
            "robots_used": st.column_config.NumberColumn("Robots used", format="%.1f"),
            "robots_available": st.column_config.NumberColumn("Robots available", format="%.0f"),
            "robots_utilization": "Robot use",
            "fde_hours_used": st.column_config.NumberColumn("FDE hours used", format="%.1f"),
            "fde_hours_available": st.column_config.NumberColumn("FDE hours available", format="%.0f"),
            "fde_utilization": "FDE use",
        },
        hide_index=True,
        width="stretch",
    )


def _render_sensitivity(requests, capacity, weights, floor, threshold, version: str) -> None:
    if version == "v1":
        st.caption(
            "Each draw adds noise to the weights, puts the positive weights back on their original total, "
            "keeps cost non-positive, then re-scores and re-solves. A request is robust when it is funded "
            "in at most 20% of draws or at least 80%, and contested in between."
        )
    else:
        st.caption(
            "Each draw adds noise to the weights, puts them back on their original total, "
            "then re-scores and re-solves. A request is robust when it is funded in at most 20% of draws "
            "or at least 80%, and contested in between."
        )
    controls = st.columns(3)
    n_sims = int(
        controls[0].number_input("Simulations", min_value=10, max_value=400, value=80, step=10)
    )
    sigma = float(
        controls[1].slider("Weight noise (std. dev.)", min_value=0.01, max_value=0.15, value=0.06, step=0.01)
    )
    seed = int(controls[2].number_input("Seed", min_value=0, max_value=10_000, value=42, step=1))

    signature = (
        version,
        tuple((name, round(weights[name], 4)) for name in criteria_for(version)),
        round(floor, 4),
        round(threshold, 4),
        n_sims,
        round(sigma, 4),
        seed,
    )
    if st.button("Run sensitivity", type="primary"):
        with st.spinner("Re-solving the portfolio…"):
            st.session_state["sensitivity"] = run_sensitivity(
                requests,
                capacity,
                base_weights=weights,
                exploration_floor=floor,
                novelty_threshold=threshold,
                n_sims=n_sims,
                sigma=sigma,
                seed=seed,
                version=version,
            )
            st.session_state["sensitivity_key"] = signature

    stored = st.session_state.get("sensitivity")
    if stored is None:
        st.caption("Not run yet.")
        return
    if st.session_state.get("sensitivity_key") != signature:
        st.warning("These results were computed for different weights or settings. Run again to refresh.")

    band = st.columns(4)
    band[0].metric("Mean overlap with baseline", f"{stored.jaccard_mean:.0%}")
    band[1].metric(
        "Objective, 10th–90th",
        f"{stored.objective_p10:.1f} – {stored.objective_p90:.1f}",
        help="Score times the robot-weeks that version credits, across the draws.",
    )
    band[2].metric("Robust requests", f"{stored.robust_count}")
    band[3].metric("Contested requests", f"{stored.contested_count}")

    histogram = (
        alt.Chart(stored.runs)
        .mark_bar(color="#9A3412")
        .encode(
            x=alt.X("objective:Q", bin=alt.Bin(maxbins=18), title="Objective"),
            y=alt.Y("count()", title="Draws"),
        )
        .properties(height=220)
    )
    st.altair_chart(histogram, width="stretch")

    table = stored.by_request.copy()
    if "segment" in table.columns:
        table["segment"] = table["segment"].map(lambda value: SEGMENT_LABEL.get(value, value))
    if "task_type" in table.columns:
        table["task_type"] = table["task_type"].map(lambda value: TASK_LABEL.get(value, value))
    st.dataframe(
        table,
        column_config={
            "request_id": "Request",
            "customer_name": "Customer",
            "segment": "Segment",
            "task_type": "Task",
            "task_novelty": st.column_config.NumberColumn("Novelty", format="%.2f"),
            "strategic_tier": "Tier",
            "baseline_score": st.column_config.NumberColumn("Baseline score", format="%.3f"),
            "baseline_decision": "Baseline",
            "funded_rate": st.column_config.ProgressColumn(
                "Funded in draws",
                min_value=0,
                max_value=1,
                format="percent",
            ),
        },
        hide_index=True,
        width="stretch",
        height=480,
    )


def _render_method(version: str) -> None:
    tiers = strategic_map(version)
    weights = default_weights(version)
    tier_line = ", ".join(f"{TIER_LABEL[name].lower()} {score:g}" for name, score in tiers.items())
    weight_line = ", ".join(
        f"{WEIGHT_LABEL[name].lower()} {weights[name]:.2f}" for name in criteria_for(version)
    )
    if version == "v1":
        body = f"""
**Gates.** A request is eligible only if the safety review has passed, an LOI is signed,
and site readiness is at least 2. Anything else is rejected and does not take a score.

**Score (v1).** Eligible requests are scored inside this book. Commercial value is
{VALUE_BLEND:.0%} value density (contract dollars per robot-week) and {EXPANSION_BLEND:.0%}
expansion potential, each min-maxed before the blend. Learning, success, and readiness
are min-maxed. Cost is FDE hours per robot-week, min-maxed, and enters with a negative weight.
Strategic tier uses a fixed map: {tier_line}. Default weights: {weight_line}.
The score is the sum of weight times criterion.

A criterion with no spread in the eligible book contributes nothing. The learning score
is not the raw novelty number. The floor below uses the raw number.

**Allocation (v1).** One binary variable per eligible request. The objective maximizes
score times robot-weeks for the whole engagement, including weeks past this file.
Each week in the capacity file, the selected requests must fit the robot count and
the FDE hours. FDE hours are spread evenly across the engagement; weeks outside
this file are not charged here, but they still count in the objective.

**Exploration floor (v1).** At least 20% of funded robot-weeks must come from requests
with raw task novelty at or above 0.6, unless you move those controls. The floor is
a mix constraint. Funding nothing satisfies it, so it is not a quota of exploratory
volume. If no eligible novel request fits the calendar, the floor would rule out
every non-empty portfolio and it is not applied.
"""
    else:
        body = f"""
**Gates.** A request is eligible only if the safety review has passed, an LOI is signed,
and site readiness is at least 2. Anything else is rejected and does not take a score.

**Score (v2).** Commercial value is {VALUE_BLEND:.0%} normalized value density times the
raw success probability, and {EXPANSION_BLEND:.0%} log1p of expansion after capping
expansion at the eligible 90th percentile and normalizing. Learning is raw task novelty.
Readiness is (site readiness − 1) / 4. Strategic tier uses a fixed map: {tier_line}.
Default weights: {weight_line}. Success is not a separate criterion. FDE hours are not
in the score. Score per FDE hour is shown so a cheap cell and an expensive cell with
the same score can be told apart.

**Allocation (v2).** One binary variable per eligible request. The objective maximizes
score times in-quarter robot-weeks: robots times the weeks of the engagement that fall
inside this capacity file. Duration past week 12 is not credited. Each of those weeks,
the selected requests must fit the robot count and the FDE hours. FDE hours are spread
evenly across the whole engagement; only the in-quarter slice is charged here.

**Exploration floor (v2).** The novelty-weighted average of funded in-quarter robot-weeks
is at least 20%, unless you move the floor. There is no 0.6 cutoff. A request is labeled
as exploration when it is not in the top half of eligible scores and its novelty is at
or above the eligible median. A funded request that is neither is labeled as filling
residual capacity. If no request novel enough to meet the floor fits the calendar,
the floor is not applied.
"""
    st.markdown(
        body
        + """
**What this does not know.** Margin, travel, robot variants, and whether two sites
teach the same skill. Those gaps are left open in `data/DATA_NOTES.md`.
"""
    )


def _utilization_chart(weekly: pd.DataFrame, used: str, available: str, title: str, y_title: str):
    ymax = float(max(weekly[used].max(), weekly[available].max()) * 1.08)
    scale = alt.Scale(domain=[0, ymax])
    base = alt.Chart(weekly).encode(
        x=alt.X("week:O", title="Week", axis=alt.Axis(labelAngle=0))
    )
    bars = base.mark_bar(color="#C2410C", size=18).encode(
        y=alt.Y(f"{used}:Q", title=y_title, scale=scale),
        tooltip=[
            alt.Tooltip("week:O", title="Week"),
            alt.Tooltip(f"{used}:Q", title="Used", format=".1f"),
            alt.Tooltip(f"{available}:Q", title="Available", format=".1f"),
        ],
    )
    line = base.mark_line(color="#1C1917", strokeDash=[4, 3]).encode(
        y=alt.Y(f"{available}:Q", scale=scale)
    )
    points = base.mark_point(color="#1C1917", size=28, filled=True).encode(
        y=alt.Y(f"{available}:Q", scale=scale)
    )
    return (bars + line + points).resolve_scale(y="shared").properties(height=260, title=title)


def _labeled(decisions: pd.DataFrame) -> pd.DataFrame:
    frame = decisions.copy()
    frame["segment_label"] = frame["segment"].map(SEGMENT_LABEL).fillna(frame["segment"])
    frame["task_label"] = frame["task_type"].map(TASK_LABEL).fillna(frame["task_type"])
    frame["tier_label"] = frame["strategic_tier"].map(TIER_LABEL).fillna(frame["strategic_tier"])
    return frame


def result_csv(decisions: pd.DataFrame) -> bytes:
    return decisions.to_csv(index=False).encode("utf-8")


def _money(value: float) -> str:
    return f"${value:,.0f}"


@st.cache_data
def _load(_requests_path: str, _capacity_path: str, _requests_mtime: float, _capacity_mtime: float):
    return load_requests(_requests_path), load_capacity(_capacity_path)


_CSS = """
<style>
.block-container { padding-top: 1.4rem; max-width: 1180px; }
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E7E1D8;
    border-radius: 10px;
    padding: 0.65rem 0.8rem;
}
</style>
"""


main()
