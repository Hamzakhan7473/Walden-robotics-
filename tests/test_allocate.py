"""The integer program: capacity, the exploration floor, and reasons."""

from pathlib import Path

import pandas as pd
import pytest

from src.allocate import allocate, load_capacity
from src.scoring import load_requests, score_requests

ROOT = Path(__file__).resolve().parents[1]


def row(**overrides):
    base = {
        "request_id": "R001",
        "customer_name": "Calder Logistics",
        "segment": "3pl_warehouse",
        "task_type": "palletizing",
        "robots_requested": 2,
        "duration_weeks": 4,
        "fde_hours_required": 40,
        "contract_value_usd": 100_000,
        "expansion_units_potential": 4,
        "success_prob": 0.8,
        "task_novelty": 0.2,
        "site_readiness": 4,
        "strategic_tier": "standard",
        "safety_review_passed": True,
        "loi_signed": True,
        "requested_start_week": 1,
        "gate_pass": True,
        "gate_reason": "",
        "score": 0.5,
    }
    base.update(overrides)
    return base


def capacity(n_weeks=4, robots=4, fde=1000):
    return pd.DataFrame(
        {
            "week": list(range(1, n_weeks + 1)),
            "robots_available": robots,
            "fde_hours_available": fde,
        }
    )


def funded_ids(result):
    chosen = result.decisions.loc[result.decisions["decision"] == "funded", "request_id"]
    return set(chosen)


def test_without_a_floor_the_higher_score_wins_the_slot():
    frame = pd.DataFrame(
        [
            row(request_id="A", score=0.9, task_novelty=0.1, robots_requested=2),
            row(request_id="B", score=0.8, task_novelty=0.9, robots_requested=2),
        ]
    )
    result = allocate(frame, capacity(robots=2), exploration_floor=0.0)
    assert funded_ids(result) == {"A"}
    assert result.objective == pytest.approx(0.9 * 8)


def test_floor_funds_the_novel_request_when_they_cannot_share():
    frame = pd.DataFrame(
        [
            row(request_id="A", score=0.9, task_novelty=0.1, robots_requested=2),
            row(request_id="B", score=0.8, task_novelty=0.9, robots_requested=2),
        ]
    )
    result = allocate(frame, capacity(robots=2), exploration_floor=0.20)
    assert funded_ids(result) == {"B"}
    # v2 share is the novelty-weighted average of in-horizon robot-weeks.
    assert result.exploration_share == pytest.approx(0.9)
    reason = result.decisions.set_index("request_id").loc["B", "decision_reason"]
    assert reason.startswith("Funded for exploration")


def test_v1_floor_uses_the_novelty_threshold_and_full_duration():
    frame = pd.DataFrame(
        [
            row(request_id="A", score=0.9, task_novelty=0.1, robots_requested=2),
            row(request_id="B", score=0.8, task_novelty=0.9, robots_requested=2),
        ]
    )
    result = allocate(frame, capacity(robots=2), exploration_floor=0.20, version="v1")
    assert funded_ids(result) == {"B"}
    assert result.exploration_share == pytest.approx(1.0)
    assert result.objective == pytest.approx(0.8 * 8)
    reason = result.decisions.set_index("request_id").loc["B", "decision_reason"]
    assert "exploration floor" in reason


def test_fde_shortfall_is_named_on_the_waitlist():
    frame = pd.DataFrame(
        [
            row(request_id="HEAVY", score=0.9, robots_requested=1, fde_hours_required=80, task_novelty=0.1),
            row(request_id="LIGHT", score=0.5, robots_requested=1, fde_hours_required=20, task_novelty=0.1),
        ]
    )
    result = allocate(frame, capacity(robots=10, fde=10), exploration_floor=0.0)
    assert funded_ids(result) == {"LIGHT"}
    reason = result.decisions.set_index("request_id").loc["HEAVY", "decision_reason"]
    assert reason.startswith("Waitlisted")
    # 80 hours over 4 weeks is 20h/week. LIGHT leaves 5h and 9 robots.
    assert "needs 20h FDE / 1 robots in week 1; 5h FDE and 9 robots free." in reason


def test_v1_waitlist_still_names_the_residual_shortfall():
    frame = pd.DataFrame(
        [
            row(request_id="HEAVY", score=0.9, robots_requested=1, fde_hours_required=80, task_novelty=0.1),
            row(request_id="LIGHT", score=0.5, robots_requested=1, fde_hours_required=20, task_novelty=0.1),
        ]
    )
    result = allocate(frame, capacity(robots=10, fde=10), exploration_floor=0.0, version="v1")
    reason = result.decisions.set_index("request_id").loc["HEAVY", "decision_reason"]
    assert "FDE hours short in week" in reason


def test_waitlist_names_the_floor_when_residual_capacity_remains():
    # A+B keeps weighted novelty above 0.20. C fits in the leftover robots
    # but would pull the weighted novelty under the floor.
    frame = pd.DataFrame(
        [
            row(request_id="A", score=1.0, task_novelty=0.1, robots_requested=2, fde_hours_required=4),
            row(request_id="B", score=0.6, task_novelty=0.9, robots_requested=1, fde_hours_required=4),
            row(request_id="C", score=0.5, task_novelty=0.0, robots_requested=3, fde_hours_required=4),
        ]
    )
    result = allocate(frame, capacity(robots=6), exploration_floor=0.20)
    assert funded_ids(result) == {"A", "B"}
    reason = result.decisions.set_index("request_id").loc["C", "decision_reason"]
    assert "needs 1h FDE / 3 robots in week 1" in reason
    assert "below the 20% floor" in reason


def test_waitlist_names_higher_value_work_on_the_binding_week():
    frame = pd.DataFrame(
        [
            row(request_id="A", customer_name="Anchor Plant", score=0.9, task_novelty=0.4, robots_requested=2),
            row(request_id="B", customer_name="Side Cell", score=0.4, task_novelty=0.4, robots_requested=2),
        ]
    )
    result = allocate(frame, capacity(robots=2), exploration_floor=0.0)
    assert funded_ids(result) == {"A"}
    reason = result.decisions.set_index("request_id").loc["B", "decision_reason"]
    assert "needs 10h FDE / 2 robots in week 1; 990h FDE and 0 robots free." in reason
    assert "Funding it would displace Anchor Plant (A)" in reason


def test_binding_week_is_the_week_with_the_shortfall():
    cap = pd.DataFrame(
        {
            "week": [1, 2, 3, 4],
            "robots_available": [10, 10, 10, 10],
            "fde_hours_available": [100, 100, 100, 30],
        }
    )
    frame = pd.DataFrame(
        [
            row(
                request_id="LATE",
                score=0.8,
                task_novelty=0.4,
                robots_requested=1,
                duration_weeks=4,
                fde_hours_required=160,
            )
        ]
    )
    result = allocate(frame, cap, exploration_floor=0.0)
    reason = result.decisions.set_index("request_id").loc["LATE", "decision_reason"]
    # 160 hours over 4 weeks is 40h/week. Only week 4 is short.
    assert "needs 40h FDE / 1 robots in week 4; 30h FDE and 10 robots free." in reason


def test_failed_gate_is_never_funded():
    frame = pd.DataFrame(
        [
            row(request_id="BAD", score=0.99, gate_pass=False, gate_reason="LOI not signed", robots_requested=1),
            row(request_id="OK", score=0.2, robots_requested=1),
        ]
    )
    result = allocate(frame, capacity(robots=4), exploration_floor=0.0)
    bad = result.decisions.set_index("request_id").loc["BAD"]
    assert bad.decision == "rejected"
    assert bad.decision_reason == "Rejected. LOI not signed."
    assert funded_ids(result) == {"OK"}


def test_engagements_that_do_not_overlap_both_fund():
    frame = pd.DataFrame(
        [
            row(request_id="EARLY", requested_start_week=1, duration_weeks=4, robots_requested=2, score=0.4),
            row(request_id="LATE", requested_start_week=5, duration_weeks=4, robots_requested=2, score=0.4),
        ]
    )
    result = allocate(frame, capacity(n_weeks=8, robots=2), exploration_floor=0.0)
    assert funded_ids(result) == {"EARLY", "LATE"}
    weekly = result.weekly.set_index("week")
    assert weekly.loc[1, "robots_used"] == 2
    assert weekly.loc[5, "robots_used"] == 2
    assert (weekly["robots_used"] <= weekly["robots_available"]).all()


def test_weeks_past_the_horizon_are_not_charged():
    cap = pd.DataFrame(
        {
            "week": [1, 2, 3, 4],
            "robots_available": [0, 0, 2, 2],
            "fde_hours_available": [0, 0, 100, 100],
        }
    )
    frame = pd.DataFrame(
        [
            row(
                request_id="LATE",
                requested_start_week=3,
                duration_weeks=4,
                robots_requested=2,
                fde_hours_required=40,
                score=0.5,
            )
        ]
    )
    result = allocate(frame, cap, exploration_floor=0.0)
    assert funded_ids(result) == {"LATE"}
    weekly = result.weekly.set_index("week")
    assert weekly.loc[1, "robots_used"] == 0
    assert weekly.loc[3, "robots_used"] == 2
    assert weekly.loc[4, "fde_hours_used"] == pytest.approx(10.0)
    # 2 robots × 2 in-horizon weeks. The other two weeks are not credited.
    assert result.objective == pytest.approx(0.5 * 2 * 2)
    funded = result.decisions.set_index("request_id").loc["LATE"]
    assert funded.in_horizon_robot_weeks == pytest.approx(4)
    assert funded.portfolio_value == pytest.approx(result.objective)
    reason = funded.decision_reason
    assert "2 of 4 weeks fall inside this plan" in reason
    assert "not credited" in reason

    v1 = allocate(frame, cap, exploration_floor=0.0, version="v1")
    assert v1.objective == pytest.approx(0.5 * 2 * 4)
    v1_reason = v1.decisions.set_index("request_id").loc["LATE", "decision_reason"]
    assert "later quarter" in v1_reason


def test_floor_is_relaxed_when_no_novel_request_fits():
    frame = pd.DataFrame(
        [
            row(request_id="NOVEL", score=0.9, task_novelty=0.9, robots_requested=10),
            row(request_id="PLAIN", score=0.4, task_novelty=0.1, robots_requested=1),
        ]
    )
    result = allocate(frame, capacity(robots=2), exploration_floor=0.20)
    assert result.exploration_floor_relaxed
    assert "not applied" in result.floor_note
    assert funded_ids(result) == {"PLAIN"}


def test_out_of_horizon_novelty_does_not_satisfy_the_floor():
    # PLAIN is the valuable, familiar cell and occupies the whole horizon.
    # LATE is novel but only week 4 of a long engagement is inside the file.
    # Crediting the later weeks would let both fund and would clear a 0.50 floor.
    cap = capacity(n_weeks=4, robots=4, fde=1000)
    frame = pd.DataFrame(
        [
            row(
                request_id="PLAIN",
                score=0.8,
                task_novelty=0.05,
                robots_requested=2,
                duration_weeks=4,
                requested_start_week=1,
            ),
            row(
                request_id="LATE",
                score=0.4,
                task_novelty=0.95,
                robots_requested=2,
                duration_weeks=10,
                requested_start_week=4,
            ),
        ]
    )
    result = allocate(frame, cap, exploration_floor=0.50)
    assert funded_ids(result) == {"LATE"}
    assert result.exploration_share == pytest.approx(0.95)
    assert result.objective == pytest.approx(0.4 * 2 * 1)

    v1 = allocate(frame, cap, exploration_floor=0.50, version="v1")
    assert funded_ids(v1) == {"PLAIN", "LATE"}


def test_low_score_familiar_request_is_labeled_residual_capacity():
    # Three eligible scores. The funded familiar cell is below the median
    # and below the median novelty, so the reason is residual capacity.
    frame = pd.DataFrame(
        [
            row(request_id="TOP", score=0.9, task_novelty=0.2, robots_requested=1, requested_start_week=1),
            row(request_id="FILL", score=0.2, task_novelty=0.1, robots_requested=1, requested_start_week=5),
            row(request_id="OTHER", score=0.5, task_novelty=0.8, robots_requested=1, requested_start_week=1),
        ]
    )
    result = allocate(frame, capacity(n_weeks=8, robots=1), exploration_floor=0.0)
    assert "FILL" in funded_ids(result)
    reason = result.decisions.set_index("request_id").loc["FILL", "decision_reason"]
    assert "Fills residual capacity" in reason


def test_book_credits_only_in_horizon_weeks_and_states_the_shortfall():
    requests = load_requests(ROOT / "data" / "requests.csv")
    cap = load_capacity(ROOT / "data" / "capacity.csv")
    scored = score_requests(requests)
    held = allocate(scored, cap, exploration_floor=0.20)
    free = allocate(scored, cap, exploration_floor=0.0)

    assert held.version == "v2"
    assert held.solver_status == "Optimal"
    assert not held.exploration_floor_relaxed
    assert held.exploration_share >= 0.20 - 1e-6
    assert free.solver_status == "Optimal"

    weekly = held.weekly
    assert (weekly["robots_used"] <= weekly["robots_available"] + 1e-6).all()
    assert (weekly["fde_hours_used"] <= weekly["fde_hours_available"] + 1e-6).all()

    decisions = held.decisions
    assert decisions["request_id"].is_unique
    assert len(decisions) == len(requests)
    rejected = decisions[decisions["decision"] == "rejected"]
    funded = decisions[decisions["decision"] == "funded"]
    waitlisted = decisions[decisions["decision"] == "waitlisted"]
    assert not rejected["gate_pass"].any()
    assert rejected["decision_reason"].str.startswith("Rejected").all()
    assert funded["gate_pass"].all()
    assert funded["decision_reason"].str.startswith("Funded").all()
    assert held.objective == pytest.approx(float(funded["portfolio_value"].sum()))
    expected = funded["score"] * funded["in_horizon_robot_weeks"]
    assert held.objective == pytest.approx(float(expected.sum()))
    overhang = funded[funded["requested_start_week"] + funded["duration_weeks"] - 1 > 12]
    if len(overhang):
        assert (overhang["in_horizon_weeks"] < overhang["duration_weeks"]).all()
        full_credit = overhang["score"] * overhang["robots_requested"] * overhang["duration_weeks"]
        assert (overhang["portfolio_value"] < full_credit - 1e-6).all()
    assert waitlisted["decision_reason"].str.contains("needs ").all()
    assert waitlisted["decision_reason"].str.contains("free").all()


def test_v1_book_still_binds_the_threshold_floor():
    requests = load_requests(ROOT / "data" / "requests.csv")
    cap = load_capacity(ROOT / "data" / "capacity.csv")
    scored = score_requests(requests, version="v1")
    held = allocate(scored, cap, exploration_floor=0.20, version="v1")
    free = allocate(scored, cap, exploration_floor=0.0, version="v1")

    assert held.version == "v1"
    assert held.solver_status == "Optimal"
    assert not held.exploration_floor_relaxed
    assert held.exploration_share >= 0.20 - 1e-6
    assert free.exploration_share < 0.20 - 1e-6
    assert funded_ids(held) != funded_ids(free)

    weekly = held.weekly
    assert weekly["fde_utilization"].max() >= 0.95
    assert weekly["robots_utilization"].max() >= 0.90
    funded = held.decisions[held.decisions["decision"] == "funded"]
    assert held.objective == pytest.approx(float(funded["portfolio_value"].sum()))
