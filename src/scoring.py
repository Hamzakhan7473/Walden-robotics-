"""Hard gates and a weighted score for deployment requests.

A request that fails a gate is excluded and does not receive a score.
A request that passes receives a score equal to the sum of its criterion
contributions, so the number can be taken apart.

``v1`` min-maxes each criterion on the eligible book. Strategic tier is the
exception: a fixed map, so the gaps in that map survive. Learning in v1 is
a min-max of task novelty. The v1 exploration floor uses raw novelty instead.

``v2`` does not min-max a criterion that already has a bounded scale.
Learning is raw task novelty. Readiness is ``(site_readiness - 1) / 4``.
Success is inside the commercial term, as a multiplier on value density.
There is no FDE penalty in the score. ``score_per_fde_hour`` is reported
beside the score and is not a criterion.

Pass ``version="v1"`` to score the original way.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

V1 = "v1"
V2 = "v2"
DEFAULT_VERSION = V2
VERSIONS = (V1, V2)

V1_WEIGHTS: dict[str, float] = {
    "commercial": 0.25,
    "learning": 0.25,
    "success": 0.20,
    "readiness": 0.15,
    "strategic": 0.10,
    "cost": -0.05,
}

V2_WEIGHTS: dict[str, float] = {
    "commercial": 0.40,
    "learning": 0.30,
    "readiness": 0.15,
    "strategic": 0.15,
}

# The running default is v2. v1 stays available for the before/after.
DEFAULT_WEIGHTS: dict[str, float] = dict(V2_WEIGHTS)

V1_CRITERIA: tuple[str, ...] = (
    "commercial",
    "learning",
    "success",
    "readiness",
    "strategic",
    "cost",
)
V2_CRITERIA: tuple[str, ...] = (
    "commercial",
    "learning",
    "readiness",
    "strategic",
)
CRITERIA: tuple[str, ...] = V2_CRITERIA

# v1 keeps a floor under "standard" so the tier is never a zero.
# v2 lets standard contribute nothing, and reference logos sit at half.
V1_STRATEGIC_TIER_SCORE: dict[str, float] = {
    "anchor": 1.0,
    "investor_linked": 0.7,
    "reference_logo": 0.6,
    "standard": 0.3,
}
V2_STRATEGIC_TIER_SCORE: dict[str, float] = {
    "anchor": 1.0,
    "investor_linked": 0.7,
    "reference_logo": 0.5,
    "standard": 0.0,
}
STRATEGIC_TIER_SCORE: dict[str, float] = dict(V2_STRATEGIC_TIER_SCORE)

# Commercial is 70% of the value-density piece and 30% of the expansion
# piece, blended after each piece is scaled. Blending before scaling would
# mix dollars with counts.
VALUE_BLEND = 0.70
EXPANSION_BLEND = 0.30

MIN_SITE_READINESS = 2

_REQUIRED = (
    "safety_review_passed",
    "loi_signed",
    "site_readiness",
    "contract_value_usd",
    "robots_requested",
    "duration_weeks",
    "expansion_units_potential",
    "task_novelty",
    "success_prob",
    "strategic_tier",
    "fde_hours_required",
)


def normalize_version(version: str | None) -> str:
    """Return ``v1`` or ``v2``. ``None`` is the current default, v2."""
    if version is None:
        return DEFAULT_VERSION
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, got {version!r}")
    return version


def criteria_for(version: str | None = None) -> tuple[str, ...]:
    return V1_CRITERIA if normalize_version(version) == V1 else V2_CRITERIA


def default_weights(version: str | None = None) -> dict[str, float]:
    source = V1_WEIGHTS if normalize_version(version) == V1 else V2_WEIGHTS
    return dict(source)


def strategic_map(version: str | None = None) -> dict[str, float]:
    source = V1_STRATEGIC_TIER_SCORE if normalize_version(version) == V1 else V2_STRATEGIC_TIER_SCORE
    return dict(source)


def resolve_weights(
    weights: Mapping[str, float] | None = None,
    version: str | None = None,
) -> dict[str, float]:
    """Return a complete weight dict for ``version``.

    Omitted calls use that version's defaults. A dict must contain every
    criterion for the version; extra keys are ignored.
    """
    chosen_version = normalize_version(version)
    criteria = criteria_for(chosen_version)
    if weights is None:
        return default_weights(chosen_version)
    missing = [name for name in criteria if name not in weights]
    if missing:
        raise KeyError(f"weights missing {missing}")
    return {name: float(weights[name]) for name in criteria}


def load_requests(path) -> pd.DataFrame:
    """Read requests.csv, including boolean columns written as True/False."""
    frame = pd.read_csv(path)
    frame["safety_review_passed"] = _as_bool(frame["safety_review_passed"])
    frame["loi_signed"] = _as_bool(frame["loi_signed"])
    return frame


def score_requests(
    requests: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    version: str | None = None,
) -> pd.DataFrame:
    """Score eligible requests and exclude the rest.

    Parameters
    ----------
    requests:
        One row per deployment request.
    weights:
        Optional map with every criterion name for ``version``. Pass a
        complete dict, or None for that version's defaults.
    version:
        ``"v1"`` or ``"v2"``. None means v2.

    Returns
    -------
    A copy of ``requests`` with gate columns, raw intensities, normalized
    criteria, contributions, ``score``, and ``score_per_fde_hour``. Failed
    gates have a reason and a null score. The diagnostic is score divided
    by FDE hours. It is not part of the score.
    """
    chosen_version = normalize_version(version)
    chosen = resolve_weights(weights, chosen_version)
    criteria_names = criteria_for(chosen_version)
    tiers = strategic_map(chosen_version)

    frame = requests.copy()
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise KeyError(f"requests missing columns: {missing}")

    frame["safety_review_passed"] = _as_bool(frame["safety_review_passed"])
    frame["loi_signed"] = _as_bool(frame["loi_signed"])
    frame["site_readiness"] = frame["site_readiness"].astype(int)

    robots = frame["robots_requested"].astype(float)
    duration = frame["duration_weeks"].astype(float)
    if (robots <= 0).any() or (duration <= 0).any():
        raise ValueError("robots_requested and duration_weeks must be positive")

    frame["robot_weeks"] = robots * duration
    frame["value_density"] = frame["contract_value_usd"].astype(float) / frame["robot_weeks"]
    frame["fde_per_robot_week"] = frame["fde_hours_required"].astype(float) / frame["robot_weeks"]

    gate_pass, gate_reason = evaluate_gates(frame)
    frame["gate_pass"] = gate_pass
    frame["gate_reason"] = gate_reason

    unknown = sorted(set(frame["strategic_tier"].astype(str)) - set(tiers))
    if unknown:
        raise ValueError(f"unknown strategic_tier values: {unknown}")

    criteria = _criteria(frame, gate_pass, chosen_version, tiers)

    for name in criteria_names:
        frame[f"criterion_{name}"] = criteria[name]
        frame[f"contrib_{name}"] = chosen[name] * criteria[name]

    contrib_cols = [f"contrib_{name}" for name in criteria_names]
    frame["score"] = frame[contrib_cols].sum(axis=1)
    hours = frame["fde_hours_required"].astype(float)
    frame["score_per_fde_hour"] = frame["score"] / hours.where(hours > 0)

    excluded = ~frame["gate_pass"]
    for name in criteria_names:
        frame.loc[excluded, f"criterion_{name}"] = np.nan
        frame.loc[excluded, f"contrib_{name}"] = np.nan
    frame.loc[excluded, "score"] = np.nan
    frame.loc[excluded, "score_per_fde_hour"] = np.nan
    frame["policy_version"] = chosen_version

    frame.attrs["weights"] = chosen
    frame.attrs["version"] = chosen_version
    return frame


def explain_score(
    row: pd.Series,
    weights: Mapping[str, float] | None = None,
    version: str | None = None,
) -> str:
    """One sentence that adds up to the score, or the gate that excluded it."""
    if not bool(row["gate_pass"]):
        return f"Excluded. {row['gate_reason']}."
    chosen_version = normalize_version(version or _row_version(row))
    chosen = resolve_weights(weights, chosen_version)
    parts = []
    for name in criteria_for(chosen_version):
        criterion = float(row[f"criterion_{name}"])
        contribution = float(row[f"contrib_{name}"])
        parts.append(f"{name} {chosen[name]:+.2f} × {criterion:.2f} = {contribution:+.3f}")
    return f"Score {float(row['score']):.3f} = " + "; ".join(parts)


def evaluate_gates(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return ``(passes, reason)``.

    The gates are safety review passed, LOI signed, and site readiness
    at least ``MIN_SITE_READINESS``. Reasons are joined with ``"; "``.
    """
    safety_ok = frame["safety_review_passed"].astype(bool)
    loi_ok = frame["loi_signed"].astype(bool)
    readiness = frame["site_readiness"].astype(int)
    ready_ok = readiness >= MIN_SITE_READINESS

    reasons = []
    for safety, loi, ready, level in zip(safety_ok, loi_ok, ready_ok, readiness):
        failed = []
        if not safety:
            failed.append("safety review not passed")
        if not loi:
            failed.append("LOI not signed")
        if not ready:
            failed.append(f"site readiness {int(level)} < {MIN_SITE_READINESS}")
        reasons.append("; ".join(failed))

    passes = pd.Series(safety_ok & loi_ok & ready_ok, index=frame.index, name="gate_pass")
    reason = pd.Series(reasons, index=frame.index, name="gate_reason")
    return passes, reason


def _criteria(
    frame: pd.DataFrame,
    gate_pass: pd.Series,
    version: str,
    tiers: Mapping[str, float],
) -> dict[str, pd.Series]:
    strategic = frame["strategic_tier"].astype(str).map(tiers).astype(float)
    if version == V1:
        value_score = _minmax(frame["value_density"], gate_pass)
        expansion_score = _minmax(frame["expansion_units_potential"].astype(float), gate_pass)
        return {
            "commercial": VALUE_BLEND * value_score + EXPANSION_BLEND * expansion_score,
            "learning": _minmax(frame["task_novelty"].astype(float), gate_pass),
            "success": _minmax(frame["success_prob"].astype(float), gate_pass),
            "readiness": _minmax(frame["site_readiness"].astype(float), gate_pass),
            "strategic": strategic,
            "cost": _minmax(frame["fde_per_robot_week"], gate_pass),
        }

    # v2. Value density is still min-maxed: it is a dollar rate with no
    # natural ceiling. It is then multiplied by raw success probability,
    # so a dense contract that is unlikely to land does not score as if
    # it were certain. Expansion is log1p, capped at the eligible 90th
    # percentile before the log, then min-maxed. The cap stops one large
    # follow-on from compressing everyone else.
    value_score = _minmax(frame["value_density"], gate_pass)
    success = frame["success_prob"].astype(float).clip(lower=0.0, upper=1.0)
    expansion_score = _capped_log_expansion(frame["expansion_units_potential"].astype(float), gate_pass)
    readiness = (frame["site_readiness"].astype(float) - 1.0) / 4.0
    return {
        "commercial": VALUE_BLEND * value_score * success + EXPANSION_BLEND * expansion_score,
        "learning": frame["task_novelty"].astype(float).clip(lower=0.0, upper=1.0),
        "readiness": readiness.clip(lower=0.0, upper=1.0),
        "strategic": strategic,
    }


def _capped_log_expansion(expansion: pd.Series, fit_mask: pd.Series) -> pd.Series:
    """log1p of expansion after capping eligible values at their 90th percentile."""
    reference = expansion.loc[fit_mask].astype(float)
    if reference.empty:
        return pd.Series(np.nan, index=expansion.index, dtype=float)
    cap = float(np.quantile(reference.to_numpy(dtype=float), 0.9))
    capped = expansion.astype(float).clip(lower=0.0, upper=cap)
    return _minmax(np.log1p(capped), fit_mask)


def _minmax(series: pd.Series, fit_mask: pd.Series) -> pd.Series:
    """Min-max against rows where ``fit_mask`` is true.

    No eligible rows → NaN. No spread → 0, so a constant criterion adds
    nothing instead of a fabricated 1. Values outside the eligible range
    clip to [0, 1]; excluded rows are cleared by the caller anyway.
    """
    reference = series.loc[fit_mask]
    if reference.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    lo = float(reference.min())
    hi = float(reference.max())
    if hi == lo:
        return pd.Series(0.0, index=series.index, dtype=float)
    scaled = (series.astype(float) - lo) / (hi - lo)
    return scaled.clip(lower=0.0, upper=1.0)


def _row_version(row: pd.Series) -> str:
    if "policy_version" in row.index and isinstance(row["policy_version"], str):
        return row["policy_version"]
    return DEFAULT_VERSION


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
