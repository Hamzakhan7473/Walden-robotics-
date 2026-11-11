"""Gates, min-max, the commercial blend, and weight application."""

import numpy as np
import pandas as pd
import pytest

from src.scoring import (
    CRITERIA,
    DEFAULT_WEIGHTS,
    STRATEGIC_TIER_SCORE,
    V1_CRITERIA,
    V1_STRATEGIC_TIER_SCORE,
    V1_WEIGHTS,
    V2_CRITERIA,
    explain_score,
    score_requests,
)


def request(**overrides):
    base = {
        "request_id": "R001",
        "customer_name": "Calder Logistics",
        "segment": "3pl_warehouse",
        "task_type": "palletizing",
        "robots_requested": 1,
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
    }
    base.update(overrides)
    return base


def test_default_weights_match_the_spec():
    assert CRITERIA == V2_CRITERIA
    assert DEFAULT_WEIGHTS == {
        "commercial": 0.40,
        "learning": 0.30,
        "readiness": 0.15,
        "strategic": 0.15,
    }
    assert STRATEGIC_TIER_SCORE == {
        "anchor": 1.0,
        "investor_linked": 0.7,
        "reference_logo": 0.5,
        "standard": 0.0,
    }


def test_v1_weights_and_tier_map_stay_available():
    assert V1_CRITERIA == (
        "commercial",
        "learning",
        "success",
        "readiness",
        "strategic",
        "cost",
    )
    assert V1_WEIGHTS == {
        "commercial": 0.25,
        "learning": 0.25,
        "success": 0.20,
        "readiness": 0.15,
        "strategic": 0.10,
        "cost": -0.05,
    }
    assert V1_STRATEGIC_TIER_SCORE["standard"] == 0.3
    assert V1_STRATEGIC_TIER_SCORE["reference_logo"] == 0.6


def test_each_gate_and_the_readiness_boundary():
    frame = pd.DataFrame(
        [
            request(request_id="OK"),
            request(request_id="SAFETY", safety_review_passed=False),
            request(request_id="LOI", loi_signed=False),
            request(request_id="SITE", site_readiness=1),
            request(request_id="EDGE", site_readiness=2),
        ]
    )
    scored = score_requests(frame).set_index("request_id")

    assert scored.loc["OK", "gate_pass"]
    assert scored.loc["OK", "gate_reason"] == ""
    assert scored.loc["EDGE", "gate_pass"]
    assert not scored.loc["SAFETY", "gate_pass"]
    assert scored.loc["SAFETY", "gate_reason"] == "safety review not passed"
    assert scored.loc["LOI", "gate_reason"] == "LOI not signed"
    assert "site readiness 1 < 2" in scored.loc["SITE", "gate_reason"]
    assert pd.isna(scored.loc["SAFETY", "score"])


def test_multiple_gate_failures_are_listed_together():
    frame = pd.DataFrame(
        [
            request(request_id="KEEP"),
            request(
                request_id="ALL",
                safety_review_passed=False,
                loi_signed=False,
                site_readiness=1,
            ),
        ]
    )
    reason = score_requests(frame).set_index("request_id").loc["ALL", "gate_reason"]
    assert reason == "safety review not passed; LOI not signed; site readiness 1 < 2"


def test_v1_commercial_blend_is_70_30_and_ignores_rejected_outliers():
    frame = pd.DataFrame(
        [
            request(
                request_id="DENSE",
                contract_value_usd=100_000,
                expansion_units_potential=0,
            ),
            request(
                request_id="EXPANDS",
                contract_value_usd=4_000,
                expansion_units_potential=10,
            ),
            request(
                request_id="OUTLIER",
                contract_value_usd=9_999_999,
                expansion_units_potential=100,
                safety_review_passed=False,
                loi_signed=False,
                site_readiness=1,
            ),
        ]
    )
    scored = score_requests(frame, version="v1").set_index("request_id")
    # Value density 25000 vs 1000, expansion 0 vs 10. Rejected outlier is wider
    # on both and must not set the scale.
    assert scored.loc["DENSE", "criterion_commercial"] == pytest.approx(0.70)
    assert scored.loc["EXPANDS", "criterion_commercial"] == pytest.approx(0.30)
    assert pd.isna(scored.loc["OUTLIER", "score"])


def test_v2_commercial_multiplies_density_by_success_and_caps_expansion():
    frame = pd.DataFrame(
        [
            request(
                request_id="DENSE",
                contract_value_usd=100_000,
                expansion_units_potential=0,
                success_prob=0.5,
            ),
            request(
                request_id="EXPANDS",
                contract_value_usd=4_000,
                expansion_units_potential=10,
                success_prob=1.0,
            ),
            request(
                request_id="OUTLIER",
                contract_value_usd=9_999_999,
                expansion_units_potential=100,
                safety_review_passed=False,
            ),
        ]
    )
    scored = score_requests(frame, version="v2").set_index("request_id")
    # Density min-max is 1 and 0. p90 of expansion {0, 10} is 9, so the
    # larger cell caps, log1p-minmaxes to 1, and the zero stays 0.
    # 0.70 * 1 * 0.50 + 0.30 * 0 = 0.35. 0.70 * 0 * 1 + 0.30 * 1 = 0.30.
    assert scored.loc["DENSE", "criterion_commercial"] == pytest.approx(0.35)
    assert scored.loc["EXPANDS", "criterion_commercial"] == pytest.approx(0.30)
    assert pd.isna(scored.loc["OUTLIER", "criterion_commercial"])

    wide = pd.DataFrame(
        [
            request(request_id=f"E{i}", expansion_units_potential=value, contract_value_usd=10_000)
            for i, value in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 100])
        ]
    )
    wide_scored = score_requests(wide, version="v2")
    expansion_score = wide_scored["criterion_commercial"] / 0.30
    raw = np.log1p(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 100], dtype=float))
    uncapped = (raw - raw.min()) / (raw.max() - raw.min())
    assert expansion_score.iloc[-1] == pytest.approx(1.0)
    assert float(expansion_score.iloc[8]) > uncapped[8] + 0.05


def test_v1_strategic_tier_keeps_the_fixed_map():
    frame = pd.DataFrame(
        [
            request(request_id="ANCHOR", strategic_tier="anchor"),
            request(request_id="STD", strategic_tier="standard"),
        ]
    )
    scored = score_requests(frame, version="v1").set_index("request_id")
    assert scored.loc["ANCHOR", "criterion_strategic"] == 1.0
    assert scored.loc["STD", "criterion_strategic"] == 0.3
    # Every other field is identical, so min-max criteria have no spread.
    for name in ("commercial", "learning", "success", "readiness", "cost"):
        assert scored.loc["ANCHOR", f"criterion_{name}"] == pytest.approx(0.0)
    assert scored.loc["ANCHOR", "score"] == pytest.approx(0.10)
    assert scored.loc["STD", "score"] == pytest.approx(0.03)


def test_v2_uses_absolute_learning_readiness_and_strategic_scales():
    frame = pd.DataFrame(
        [
            request(request_id="ANCHOR", strategic_tier="anchor", task_novelty=0.25, site_readiness=5),
            request(request_id="STD", strategic_tier="standard", task_novelty=0.25, site_readiness=2),
        ]
    )
    scored = score_requests(frame, version="v2").set_index("request_id")
    assert scored.loc["ANCHOR", "criterion_learning"] == pytest.approx(0.25)
    assert scored.loc["STD", "criterion_learning"] == pytest.approx(0.25)
    assert scored.loc["ANCHOR", "criterion_readiness"] == pytest.approx(1.0)
    assert scored.loc["STD", "criterion_readiness"] == pytest.approx(0.25)
    assert scored.loc["ANCHOR", "criterion_strategic"] == pytest.approx(1.0)
    assert scored.loc["STD", "criterion_strategic"] == pytest.approx(0.0)
    assert "criterion_success" not in scored.columns
    assert "criterion_cost" not in scored.columns


def test_contributions_sum_to_the_score():
    frame = pd.DataFrame(
        [
            request(request_id="A", task_novelty=0.2, success_prob=0.9, contract_value_usd=80_000),
            request(request_id="B", task_novelty=0.8, success_prob=0.4, contract_value_usd=200_000, strategic_tier="anchor", site_readiness=5, fde_hours_required=200),
            request(request_id="C", task_novelty=0.5, success_prob=0.7, contract_value_usd=50_000, site_readiness=2, expansion_units_potential=12),
        ]
    )
    scored = score_requests(frame)
    eligible = scored[scored["gate_pass"]]
    summed = eligible[[f"contrib_{name}" for name in CRITERIA]].sum(axis=1)
    assert np.allclose(summed, eligible["score"])


def test_weights_are_configurable_and_do_not_mutate_the_defaults():
    frozen = dict(DEFAULT_WEIGHTS)
    weights = {name: 0.0 for name in CRITERIA}
    weights["learning"] = 1.0
    frame = pd.DataFrame(
        [
            request(request_id="LOW", task_novelty=0.1),
            request(request_id="HIGH", task_novelty=0.9),
        ]
    )
    scored = score_requests(frame, weights).set_index("request_id")
    # v2 learning is the raw novelty, not a min-max of it.
    assert scored.loc["HIGH", "score"] == pytest.approx(0.9)
    assert scored.loc["LOW", "score"] == pytest.approx(0.1)
    assert scored.loc["HIGH", "contrib_learning"] == pytest.approx(0.9)
    assert scored.loc["HIGH", "contrib_commercial"] == pytest.approx(0.0)
    assert DEFAULT_WEIGHTS == frozen


def test_v1_cost_enters_as_a_penalty():
    weights = {name: 0.0 for name in V1_CRITERIA}
    weights["cost"] = -0.05
    frame = pd.DataFrame(
        [
            request(request_id="CHEAP", fde_hours_required=10),
            request(request_id="DEAR", fde_hours_required=100),
        ]
    )
    scored = score_requests(frame, weights, version="v1").set_index("request_id")
    assert scored.loc["CHEAP", "contrib_cost"] == pytest.approx(0.0)
    assert scored.loc["DEAR", "contrib_cost"] == pytest.approx(-0.05)
    assert scored.loc["DEAR", "score"] == pytest.approx(-0.05)


def test_v2_score_per_fde_hour_is_diagnostic_and_not_a_criterion():
    frame = pd.DataFrame(
        [
            request(request_id="CHEAP", fde_hours_required=10),
            request(request_id="DEAR", fde_hours_required=100),
        ]
    )
    scored = score_requests(frame, version="v2").set_index("request_id")
    assert scored.loc["CHEAP", "score"] == pytest.approx(scored.loc["DEAR", "score"])
    assert scored.loc["CHEAP", "score_per_fde_hour"] == pytest.approx(scored.loc["CHEAP", "score"] / 10)
    assert scored.loc["DEAR", "score_per_fde_hour"] == pytest.approx(scored.loc["DEAR", "score"] / 100)
    assert scored.loc["CHEAP", "score_per_fde_hour"] > scored.loc["DEAR", "score_per_fde_hour"]


def test_string_booleans_are_read_as_booleans():
    frame = pd.DataFrame(
        [
            request(request_id="BAD", safety_review_passed="False", loi_signed="yes"),
            request(request_id="GOOD", safety_review_passed="True", loi_signed=1),
        ]
    )
    scored = score_requests(frame).set_index("request_id")
    assert not scored.loc["BAD", "gate_pass"]
    assert scored.loc["GOOD", "gate_pass"]


def test_unknown_tier_and_missing_weight_are_errors():
    with pytest.raises(ValueError, match="strategic_tier"):
        score_requests(pd.DataFrame([request(strategic_tier="logo")]))
    with pytest.raises(KeyError, match="learning"):
        score_requests(pd.DataFrame([request()]), {"commercial": 1.0})


def test_explain_score_adds_up_or_names_the_gate():
    frame = pd.DataFrame(
        [
            request(request_id="IN"),
            request(request_id="OUT", safety_review_passed=False),
        ]
    )
    scored = score_requests(frame).set_index("request_id")
    text = explain_score(scored.loc["IN"])
    assert text.startswith("Score")
    assert "commercial" in text
    assert explain_score(scored.loc["OUT"]).startswith("Excluded. safety review not passed")
