"""Monte Carlo perturbation of the score weights.

Positive weights are noised, clipped above zero, and rescaled so they keep
the baseline positive mass. In v1, cost is noised separately and kept
non-positive. v2 has no cost term. Each draw is re-scored and re-solved.
What moves is who gets funded, not the units of the score.

A request is robust when its funded rate is at or below 20% or at or above
80%. It is contested when the rate sits strictly between those cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.allocate import AllocationResult, allocate
from src.scoring import V1, criteria_for, normalize_version, resolve_weights, score_requests

# v1 positive criteria. v2 uses every criterion it has; none of them is a penalty.
POSITIVE = ("commercial", "learning", "success", "readiness", "strategic")
COST = "cost"
ROBUST_LOW = 0.20
ROBUST_HIGH = 0.80


@dataclass
class SensitivityResult:
    baseline: AllocationResult
    runs: pd.DataFrame
    by_request: pd.DataFrame
    jaccard_mean: float
    objective_mean: float
    objective_p10: float
    objective_p90: float
    floor_relaxed_rate: float
    n_sims: int
    sigma: float
    seed: int
    version: str
    robust_count: int
    contested_count: int


def perturb_weights(
    base: Mapping[str, float],
    rng: np.random.Generator,
    sigma: float,
    version: str | None = None,
) -> dict[str, float]:
    """Return one perturbed weight dict.

    ``sigma`` is the standard deviation of the Gaussian noise added to each
    weight before the positive weights are rescaled. v1 keeps the cost
    weight non-positive. v2 rescales its four positive weights only.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    chosen_version = normalize_version(version)
    resolved = resolve_weights(base, chosen_version)
    positive = [name for name in criteria_for(chosen_version) if name != COST]
    noised = {
        name: max(resolved[name] + float(rng.normal(0.0, sigma)), 0.01) for name in positive
    }
    mass = sum(resolved[name] for name in positive)
    noised_sum = sum(noised.values())
    if mass > 0 and noised_sum > 0:
        weights = {name: noised[name] * mass / noised_sum for name in positive}
    else:
        weights = noised
    if chosen_version == V1:
        cost = resolved[COST] + float(rng.normal(0.0, sigma))
        lower = min(-0.25, resolved[COST])
        weights[COST] = float(min(0.0, max(cost, lower)))
    return weights


def stability_counts(by_request: pd.DataFrame) -> tuple[int, int]:
    """Return ``(robust_count, contested_count)`` from a funded-rate column."""
    rate = by_request["funded_rate"].astype(float)
    contested = int(((rate > ROBUST_LOW) & (rate < ROBUST_HIGH)).sum())
    robust = int(len(by_request) - contested)
    return robust, contested


def run_sensitivity(
    requests: pd.DataFrame,
    capacity: pd.DataFrame,
    base_weights: Mapping[str, float] | None = None,
    exploration_floor: float = 0.20,
    novelty_threshold: float = 0.6,
    n_sims: int = 200,
    sigma: float = 0.06,
    seed: int = 42,
    version: str | None = None,
) -> SensitivityResult:
    """Re-solve the portfolio under perturbed weights.

    The baseline allocation uses ``base_weights`` with no noise. Jaccard
    similarity compares each draw's funded set to that baseline.
    ``version`` selects the scorer and the allocator together.
    """
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    chosen_version = normalize_version(version)
    resolved = resolve_weights(base_weights, chosen_version)
    criteria_names = criteria_for(chosen_version)
    rng = np.random.default_rng(seed)
    baseline = allocate(
        score_requests(requests, resolved, version=chosen_version),
        capacity,
        exploration_floor=exploration_floor,
        novelty_threshold=novelty_threshold,
        version=chosen_version,
    )
    base_funded = set(
        baseline.decisions.loc[baseline.decisions["decision"] == "funded", "request_id"]
    )

    funded_hits = {request_id: 0 for request_id in baseline.decisions["request_id"]}
    runs = []
    jaccards = []
    for sim in range(n_sims):
        weights = perturb_weights(resolved, rng, sigma, version=chosen_version)
        result = allocate(
            score_requests(requests, weights, version=chosen_version),
            capacity,
            exploration_floor=exploration_floor,
            novelty_threshold=novelty_threshold,
            version=chosen_version,
        )
        funded = set(result.decisions.loc[result.decisions["decision"] == "funded", "request_id"])
        for request_id in funded:
            funded_hits[request_id] = funded_hits.get(request_id, 0) + 1
        union = base_funded | funded
        jaccards.append((len(base_funded & funded) / len(union)) if union else 1.0)
        runs.append(
            {
                "sim": sim,
                "objective": result.objective,
                "exploration_share": result.exploration_share,
                "floor_relaxed": result.exploration_floor_relaxed,
                "funded_count": len(funded),
                **{f"weight_{name}": weights[name] for name in criteria_names},
            }
        )

    run_frame = pd.DataFrame(runs)
    keep = [
        column
        for column in (
            "request_id",
            "customer_name",
            "segment",
            "task_type",
            "task_novelty",
            "strategic_tier",
            "score",
            "decision",
        )
        if column in baseline.decisions.columns
    ]
    by_request = baseline.decisions[keep].copy()
    by_request = by_request.rename(columns={"score": "baseline_score", "decision": "baseline_decision"})
    by_request["funded_rate"] = by_request["request_id"].map(
        lambda request_id: funded_hits.get(request_id, 0) / n_sims
    )
    by_request = by_request.sort_values(
        ["funded_rate", "baseline_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    objectives = run_frame["objective"].to_numpy()
    robust_count, contested_count = stability_counts(by_request)
    return SensitivityResult(
        baseline=baseline,
        runs=run_frame,
        by_request=by_request,
        jaccard_mean=float(np.mean(jaccards)),
        objective_mean=float(np.mean(objectives)),
        objective_p10=float(np.percentile(objectives, 10)),
        objective_p90=float(np.percentile(objectives, 90)),
        floor_relaxed_rate=float(run_frame["floor_relaxed"].mean()),
        n_sims=n_sims,
        sigma=sigma,
        seed=seed,
        version=chosen_version,
        robust_count=robust_count,
        contested_count=contested_count,
    )
