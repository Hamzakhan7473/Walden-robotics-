"""Weight noise stays a penalty on cost and does not rescale the score."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.scoring import V1_CRITERIA, V1_WEIGHTS, V2_CRITERIA, default_weights
from src.sensitivity import POSITIVE, perturb_weights, run_sensitivity


def request(**overrides):
    base = {
        "request_id": "R001",
        "customer_name": "Calder Logistics",
        "segment": "3pl_warehouse",
        "task_type": "palletizing",
        "robots_requested": 1,
        "duration_weeks": 4,
        "fde_hours_required": 40,
        "contract_value_usd": 80_000,
        "expansion_units_potential": 2,
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


def test_positive_weights_keep_their_mass_and_cost_stays_negative():
    rng = np.random.default_rng(0)
    mass = sum(V1_WEIGHTS[name] for name in POSITIVE)
    for _ in range(25):
        weights = perturb_weights(V1_WEIGHTS, rng, sigma=0.06, version="v1")
        assert set(weights) == set(V1_CRITERIA)
        assert weights["cost"] <= 0
        assert sum(weights[name] for name in POSITIVE) == pytest.approx(mass)


def test_v2_weights_keep_their_mass_and_have_no_cost_term():
    rng = np.random.default_rng(0)
    base = default_weights("v2")
    mass = sum(base.values())
    for _ in range(25):
        weights = perturb_weights(base, rng, sigma=0.06, version="v2")
        assert set(weights) == set(V2_CRITERIA)
        assert "cost" not in weights
        assert sum(weights.values()) == pytest.approx(mass)
        assert all(value > 0 for value in weights.values())


def test_sensitivity_is_deterministic():
    frame = pd.DataFrame(
        [
            request(request_id="A", task_novelty=0.2, contract_value_usd=120_000, strategic_tier="anchor"),
            request(request_id="B", task_novelty=0.85, contract_value_usd=40_000, success_prob=0.45, fde_hours_required=160),
            request(request_id="C", task_novelty=0.3, contract_value_usd=90_000, robots_requested=2, requested_start_week=1),
            request(request_id="D", task_novelty=0.7, contract_value_usd=70_000, robots_requested=2, requested_start_week=1, success_prob=0.55),
        ]
    )
    capacity = pd.DataFrame(
        {"week": [1, 2, 3, 4], "robots_available": 2, "fde_hours_available": 80}
    )
    kwargs = dict(n_sims=6, sigma=0.08, seed=7, exploration_floor=0.2)
    first = run_sensitivity(frame, capacity, **kwargs)
    second = run_sensitivity(frame, capacity, **kwargs)
    assert_frame_equal(first.runs, second.runs)
    assert first.by_request["funded_rate"].between(0, 1).all()
    assert 0 <= first.jaccard_mean <= 1
    assert len(first.runs) == 6
