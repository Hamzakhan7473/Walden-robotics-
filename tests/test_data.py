"""The seeded generator and the files it writes."""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from data.generate_data import (
    FDE_HEADCOUNT,
    FDE_HOURS_PER_WEEK,
    FDE_UTILIZATION,
    REQUEST_COLUMNS,
    build_capacity,
    build_requests,
)
from src.scoring import score_requests

ROOT = Path(__file__).resolve().parents[1]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def test_seed_is_reproducible_and_not_decorative():
    first = build_requests(seed=42)
    second = build_requests(seed=42)
    other = build_requests(seed=1)
    assert_frame_equal(first, second)
    assert not first.equals(other)


def test_request_columns_and_bounds():
    frame = build_requests()
    assert list(frame.columns) == REQUEST_COLUMNS
    assert len(frame) == 40
    assert frame["request_id"].is_unique
    assert frame["customer_name"].is_unique
    assert set(frame["segment"]) == {
        "auto_oem",
        "tier1_supplier",
        "3pl_warehouse",
        "electronics",
        "food_bev",
    }
    assert frame["robots_requested"].between(1, 6).all()
    assert frame["duration_weeks"].isin([4, 6, 8, 10, 12]).all()
    assert frame["success_prob"].between(0, 1).all()
    assert frame["task_novelty"].between(0, 1).all()
    assert frame["site_readiness"].between(1, 5).all()
    assert frame["requested_start_week"].between(1, 12).all()


def test_novel_tasks_are_less_likely_to_succeed_and_oem_pays_more():
    frame = build_requests()
    corr = frame["task_novelty"].corr(frame["success_prob"])
    assert corr < -0.6
    oem = frame.loc[frame["segment"] == "auto_oem", "contract_value_usd"].mean()
    warehouse = frame.loc[frame["segment"] == "3pl_warehouse", "contract_value_usd"].mean()
    assert oem > warehouse


def test_about_fifteen_percent_fail_a_hard_gate():
    scored = score_requests(build_requests())
    rate = float((~scored["gate_pass"]).mean())
    assert 0.10 <= rate <= 0.20


def test_capacity_ramps_robots_and_holds_fde_hours_fixed():
    capacity = build_capacity()
    assert list(capacity["week"]) == list(range(1, 13))
    assert capacity.loc[capacity["week"] == 1, "robots_available"].item() == 24
    assert capacity.loc[capacity["week"] == 4, "robots_available"].item() == 24
    assert capacity.loc[capacity["week"] == 5, "robots_available"].item() == 26
    assert capacity.loc[capacity["week"] == 9, "robots_available"].item() == 28
    assert capacity.loc[capacity["week"] == 12, "robots_available"].item() == 28
    expected_hours = FDE_HEADCOUNT * FDE_HOURS_PER_WEEK * FDE_UTILIZATION
    assert (capacity["fde_hours_available"] == expected_hours).all()


def test_committed_files_match_the_generator():
    requests = pd.read_csv(ROOT / "data" / "requests.csv")
    capacity = pd.read_csv(ROOT / "data" / "capacity.csv")
    built_requests = build_requests()
    built_capacity = build_capacity()
    for column in ("safety_review_passed", "loi_signed"):
        requests[column] = _as_bool(requests[column])
        built_requests[column] = _as_bool(built_requests[column])
    assert_frame_equal(
        requests[REQUEST_COLUMNS].reset_index(drop=True),
        built_requests[REQUEST_COLUMNS].reset_index(drop=True),
        check_dtype=False,
        rtol=1e-4,
        atol=1e-4,
    )
    assert_frame_equal(
        capacity.reset_index(drop=True),
        built_capacity.reset_index(drop=True),
        check_dtype=False,
    )
