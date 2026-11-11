"""Integer program for the deployment book.

Decision: fund a request or not.

``v1`` maximizes ``score × robot-weeks`` over the whole engagement, including
weeks past the capacity file. Its exploration floor requires at least
``exploration_floor`` of those robot-weeks to come from requests whose raw
task novelty is at least ``novelty_threshold`` (default 0.6).

``v2`` maximizes ``score × in_horizon_robot_weeks``. In-horizon robot-weeks
are robots times the weeks of the engagement that fall inside the capacity
file (weeks 1–12 in the take-home book). Duration past that file is not
credited in the objective or in the floor. The floor is continuous:
the novelty-weighted average of funded in-horizon robot-weeks is at least
``exploration_floor``. There is no novelty threshold.

The empty portfolio satisfies a share constraint, so the floor is not a
minimum volume of exploration. If no eligible request can pull the mix up
to the floor, the floor would ban every non-empty portfolio and it is not
applied.

FDE hours are spread evenly across the engagement. Weeks outside the file
do not consume this plan.
"""

from __future__ import annotations

from dataclasses import dataclass
import uuid

import pandas as pd
import pulp

from src.scoring import V1, V2, normalize_version

FUNDED = "funded"
WAITLISTED = "waitlisted"
REJECTED = "rejected"

_REQUIRED = (
    "request_id",
    "gate_pass",
    "gate_reason",
    "score",
    "robots_requested",
    "duration_weeks",
    "requested_start_week",
    "fde_hours_required",
    "task_novelty",
)


@dataclass
class AllocationResult:
    decisions: pd.DataFrame
    weekly: pd.DataFrame
    objective: float
    exploration_share: float
    exploration_robot_weeks: float
    total_robot_weeks: float
    exploration_floor: float
    novelty_threshold: float
    exploration_floor_relaxed: bool
    floor_note: str
    solver_status: str
    version: str


def load_capacity(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"week", "robots_available", "fde_hours_available"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"capacity file missing columns: {sorted(missing)}")
    out = frame.copy()
    out["week"] = out["week"].astype(int)
    out["robots_available"] = out["robots_available"].astype(float)
    out["fde_hours_available"] = out["fde_hours_available"].astype(float)
    return out.sort_values("week").reset_index(drop=True)


def in_horizon_week_count(start: int, duration: int, weeks: list[int]) -> int:
    """How many weeks of an engagement land in ``weeks``."""
    week_set = set(int(week) for week in weeks)
    return sum(1 for offset in range(int(duration)) if int(start) + offset in week_set)


def engagement_profile(
    robots: float,
    fde_hours: float,
    duration: int,
    start: int,
    weeks: list[int],
) -> tuple[dict[int, float], dict[int, float]]:
    """Robot load and even FDE load on each week of the capacity file.

    Assumption: ``fde_hours`` is the total for the engagement and is spread
    evenly across ``duration``. A week outside ``weeks`` is a later quarter
    and is not charged here. The robot stays on site for every in-horizon
    week of the engagement.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    per_week_fde = float(fde_hours) / float(duration)
    week_set = set(weeks)
    robots_load = {week: 0.0 for week in weeks}
    fde_load = {week: 0.0 for week in weeks}
    for offset in range(int(duration)):
        week = int(start) + offset
        if week in week_set:
            robots_load[week] = float(robots)
            fde_load[week] = per_week_fde
    return robots_load, fde_load


def allocate(
    scored: pd.DataFrame,
    capacity: pd.DataFrame,
    exploration_floor: float = 0.20,
    novelty_threshold: float = 0.6,
    version: str | None = None,
) -> AllocationResult:
    """Fund, waitlist, or reject every request.

    Rejected means a hard gate failed. Waitlisted means the request was
    eligible and was not in the optimal set. Funded means it was selected.
    Each row carries ``decision_reason``.

    ``version="v1"`` credits full duration and uses the 0.6 novelty cut.
    ``version="v2"`` (the default) credits only in-horizon robot-weeks and
    uses a continuous novelty floor.
    """
    chosen_version = normalize_version(version)
    if not 0.0 <= exploration_floor <= 1.0:
        raise ValueError("exploration_floor must be between 0 and 1")
    if chosen_version == V1 and not 0.0 <= novelty_threshold <= 1.0:
        raise ValueError("novelty_threshold must be between 0 and 1")

    frame = scored.copy()
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise KeyError(f"scored requests missing columns: {missing}")

    frame["request_id"] = frame["request_id"].astype(str)
    if frame["request_id"].duplicated().any():
        raise ValueError("request_id values must be unique")

    frame["robot_weeks"] = (
        frame["robots_requested"].astype(float) * frame["duration_weeks"].astype(float)
    )
    frame["gate_pass"] = frame["gate_pass"].astype(bool)

    weeks, robots_avail, fde_avail = _capacity_maps(capacity)
    frame["in_horizon_weeks"] = [
        in_horizon_week_count(int(start), int(duration), weeks)
        for start, duration in zip(frame["requested_start_week"], frame["duration_weeks"])
    ]
    frame["in_horizon_robot_weeks"] = (
        frame["robots_requested"].astype(float) * frame["in_horizon_weeks"].astype(float)
    )
    credit_column = "robot_weeks" if chosen_version == V1 else "in_horizon_robot_weeks"
    frame["portfolio_value"] = frame["score"] * frame[credit_column]

    passed = frame["gate_pass"]
    if passed.any() and frame.loc[passed, "score"].isna().any():
        raise ValueError("eligible requests must have a score")

    profiles: dict[str, dict[str, dict[int, float]]] = {}
    for row in frame[passed].itertuples(index=False):
        robot_load, fde_load = engagement_profile(
            row.robots_requested,
            row.fde_hours_required,
            int(row.duration_weeks),
            int(row.requested_start_week),
            weeks,
        )
        profiles[row.request_id] = {"robots": robot_load, "fde": fde_load}

    # v2 does not offer the solver a request that occupies no week in the
    # file. A zero coefficient would let CBC fund it for free, and those
    # weeks must not count toward the floor either.
    solver_rows = frame[passed].copy()
    outside = pd.Series(False, index=solver_rows.index)
    if chosen_version == V2:
        outside = solver_rows["in_horizon_robot_weeks"] <= 0
        solver_rows = solver_rows.loc[~outside].copy()

    floor_cut = novelty_threshold if chosen_version == V1 else exploration_floor
    novel_fits = _any_floor_request_fits(
        solver_rows, profiles, robots_avail, fde_avail, floor_cut, chosen_version
    )
    relaxed = False
    floor_note = ""
    use_floor = exploration_floor > 0 and novel_fits
    if exploration_floor > 0 and not novel_fits:
        relaxed = True
        if chosen_version == V1:
            floor_note = (
                "No eligible request with task novelty at or above "
                f"{novelty_threshold:.2f} fits inside weekly capacity, "
                "so the exploration floor was not applied."
            )
        else:
            floor_note = (
                "No eligible request with task novelty at or above "
                f"{exploration_floor:.2f} fits inside the horizon, "
                "so the exploration floor was not applied."
            )

    if solver_rows.empty:
        status, chosen, pulp_objective = "Optimal", {}, 0.0
    else:
        status, chosen, pulp_objective = _solve(
            solver_rows,
            profiles,
            weeks,
            robots_avail,
            fde_avail,
            use_floor,
            exploration_floor,
            novelty_threshold,
            chosen_version,
        )
        if status != "Optimal":
            failed_status = status
            status, chosen, pulp_objective = _solve(
                solver_rows,
                profiles,
                weeks,
                robots_avail,
                fde_avail,
                False,
                exploration_floor,
                novelty_threshold,
                chosen_version,
            )
            relaxed = True
            floor_note = (
                f"The exploration-floor model returned {failed_status}, "
                "so the floor was not applied."
            )
            if status != "Optimal":
                raise RuntimeError(f"allocation did not solve: {status}")

    weekly, slack_robots, slack_fde = _weekly_usage(
        capacity, weeks, profiles, chosen, robots_avail, fde_avail
    )
    _reject_infeasible_usage(weekly)

    funded_ids = {request_id for request_id, flag in chosen.items() if flag}
    funded_rows = frame[frame["request_id"].isin(funded_ids)]
    total_rw, novel_rw, share = _exploration_totals(
        funded_rows, chosen_version, novelty_threshold
    )

    objective = float(funded_rows["portfolio_value"].sum()) if len(funded_rows) else 0.0
    scale = max(abs(objective), 1.0)
    if abs(objective - pulp_objective) > 0.01 * scale:
        raise RuntimeError(
            f"solver objective {pulp_objective:.4f} does not match "
            f"the funded rows {objective:.4f}"
        )

    if (
        not relaxed
        and exploration_floor > 0
        and total_rw > 1e-6
        and novel_rw + 1e-4 < exploration_floor * total_rw
    ):
        raise RuntimeError(
            f"exploration floor missed: {novel_rw:.2f} of {total_rw:.2f} "
            f"is below {exploration_floor:.2f}"
        )

    eligible_scored = frame[passed]
    median_score = float(eligible_scored["score"].median()) if len(eligible_scored) else 0.0
    if chosen_version == V1:
        exploration_cut = float(novelty_threshold)
    else:
        exploration_cut = (
            float(eligible_scored["task_novelty"].median()) if len(eligible_scored) else 0.0
        )

    funded_lookup = {
        row.request_id: row for row in funded_rows.itertuples(index=False)
    }
    outside_ids = set(frame.loc[passed & (frame["in_horizon_robot_weeks"] <= 0), "request_id"]) if chosen_version == V2 else set()

    decisions = []
    reasons = []
    for row in frame.itertuples(index=False):
        if not row.gate_pass:
            decisions.append(REJECTED)
            reasons.append(f"Rejected. {row.gate_reason}.")
            continue
        if row.request_id in funded_ids:
            decisions.append(FUNDED)
            reasons.append(
                _funded_reason(
                    row,
                    chosen_version,
                    exploration_floor,
                    novelty_threshold,
                    relaxed,
                    median_score,
                    exploration_cut,
                    weeks_inside=int(row.in_horizon_weeks),
                )
            )
            continue
        decisions.append(WAITLISTED)
        if chosen_version == V2 and row.request_id in outside_ids:
            reasons.append(
                "Waitlisted. No weeks of this deployment fall inside this plan, "
                "so it is not credited."
            )
            continue
        reasons.append(
            _waitlist_reason(
                row,
                profiles[row.request_id],
                slack_robots,
                slack_fde,
                novel_rw,
                total_rw,
                exploration_floor,
                novelty_threshold,
                relaxed,
                chosen_version,
                funded_lookup,
                profiles,
            )
        )

    frame["decision"] = decisions
    frame["decision_reason"] = reasons
    frame["policy_version"] = chosen_version
    frame["_ord"] = frame["decision"].map({FUNDED: 0, WAITLISTED: 1, REJECTED: 2})
    frame = (
        frame.sort_values(
            ["_ord", "score", "request_id"],
            ascending=[True, False, True],
            na_position="last",
        )
        .drop(columns="_ord")
        .reset_index(drop=True)
    )

    return AllocationResult(
        decisions=frame,
        weekly=weekly,
        objective=objective,
        exploration_share=share,
        exploration_robot_weeks=novel_rw,
        total_robot_weeks=total_rw,
        exploration_floor=exploration_floor,
        novelty_threshold=float(novelty_threshold),
        exploration_floor_relaxed=relaxed,
        floor_note=floor_note,
        solver_status=status,
        version=chosen_version,
    )


def _exploration_totals(
    funded_rows: pd.DataFrame,
    version: str,
    novelty_threshold: float,
) -> tuple[float, float, float]:
    """Return ``(total, novel_mass, share)``.

    v1 share is the fraction of full-duration robot-weeks with novelty at
    or above the threshold. v2 share is the novelty-weighted average of
    in-horizon robot-weeks. ``novel_mass`` is the numerator of that share.
    """
    if funded_rows.empty:
        return 0.0, 0.0, 0.0
    if version == V1:
        total = float(funded_rows["robot_weeks"].sum())
        novel = funded_rows["task_novelty"].astype(float) >= novelty_threshold
        mass = float(funded_rows.loc[novel, "robot_weeks"].sum())
    else:
        total = float(funded_rows["in_horizon_robot_weeks"].sum())
        mass = float(
            (funded_rows["in_horizon_robot_weeks"] * funded_rows["task_novelty"].astype(float)).sum()
        )
    share = (mass / total) if total > 1e-9 else 0.0
    return total, mass, share


def _capacity_maps(
    capacity: pd.DataFrame,
) -> tuple[list[int], dict[int, float], dict[int, float]]:
    if capacity.empty:
        raise ValueError("capacity is empty")
    weeks = [int(week) for week in capacity["week"].tolist()]
    if len(weeks) != len(set(weeks)):
        raise ValueError("capacity weeks must be unique")
    robots = {
        int(row.week): float(row.robots_available) for row in capacity.itertuples(index=False)
    }
    fde = {
        int(row.week): float(row.fde_hours_available)
        for row in capacity.itertuples(index=False)
    }
    if any(value < 0 for value in robots.values()) or any(value < 0 for value in fde.values()):
        raise ValueError("capacity cannot be negative")
    return weeks, robots, fde


def _any_floor_request_fits(
    eligible, profiles, robots_avail, fde_avail, floor_cut: float, version: str
) -> bool:
    """True when some request that can lift the floor also fits alone.

    A convex mix of novelties cannot exceed the best novelty in the mix.
    v2 therefore needs one fitting request whose novelty is at least the
    floor. v1 needs one fitting request at or above the threshold. A v1
    request past the horizon still counts: that version credits those weeks.
    """
    for row in eligible.itertuples(index=False):
        if float(row.task_novelty) < floor_cut:
            continue
        if version == V2 and float(row.in_horizon_robot_weeks) <= 0:
            continue
        if _fits(profiles[row.request_id], robots_avail, fde_avail):
            return True
    return False


def _fits(profile, robots_avail, fde_avail) -> bool:
    for week, load in profile["robots"].items():
        if load > robots_avail[week] + 1e-6:
            return False
    for week, load in profile["fde"].items():
        if load > fde_avail[week] + 1e-6:
            return False
    return True


def _solve(
    eligible: pd.DataFrame,
    profiles: dict,
    weeks: list[int],
    robots_avail: dict[int, float],
    fde_avail: dict[int, float],
    use_floor: bool,
    exploration_floor: float,
    novelty_threshold: float,
    version: str,
) -> tuple[str, dict[str, bool], float]:
    suffix = uuid.uuid4().hex[:8]
    problem = pulp.LpProblem(f"deployment_{suffix}", pulp.LpMaximize)
    variables = {
        row.request_id: pulp.LpVariable(f"fund_{_safe(row.request_id)}_{suffix}", cat="Binary")
        for row in eligible.itertuples(index=False)
    }

    problem += pulp.lpSum(
        float(row.portfolio_value) * variables[row.request_id]
        for row in eligible.itertuples(index=False)
    )

    for week in weeks:
        problem += (
            pulp.lpSum(profiles[request_id]["robots"][week] * var for request_id, var in variables.items())
            <= robots_avail[week]
        )
        problem += (
            pulp.lpSum(profiles[request_id]["fde"][week] * var for request_id, var in variables.items())
            <= fde_avail[week]
        )

    if use_floor:
        terms = []
        for row in eligible.itertuples(index=False):
            if version == V1:
                robot_weeks = float(row.robot_weeks)
                if float(row.task_novelty) >= novelty_threshold:
                    terms.append(robot_weeks * (1.0 - exploration_floor) * variables[row.request_id])
                else:
                    terms.append(-robot_weeks * exploration_floor * variables[row.request_id])
            else:
                robot_weeks = float(row.in_horizon_robot_weeks)
                terms.append(
                    robot_weeks * (float(row.task_novelty) - exploration_floor) * variables[row.request_id]
                )
        problem += pulp.lpSum(terms) >= 0

    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(status_code, str(status_code))
    if status != "Optimal":
        return status, {request_id: False for request_id in variables}, 0.0

    chosen = {}
    for request_id, variable in variables.items():
        value = pulp.value(variable)
        chosen[request_id] = value is not None and value >= 0.5
    objective = pulp.value(problem.objective)
    return status, chosen, float(objective or 0.0)


def _weekly_usage(capacity, weeks, profiles, chosen, robots_avail, fde_avail):
    used_robots = {week: 0.0 for week in weeks}
    used_fde = {week: 0.0 for week in weeks}
    for request_id, flag in chosen.items():
        if not flag:
            continue
        for week, load in profiles[request_id]["robots"].items():
            used_robots[week] += load
        for week, load in profiles[request_id]["fde"].items():
            used_fde[week] += load

    weekly = capacity.copy()
    weekly["week"] = weekly["week"].astype(int)
    weekly["robots_used"] = weekly["week"].map(used_robots).astype(float)
    weekly["fde_hours_used"] = weekly["week"].map(used_fde).astype(float)
    weekly["robots_slack"] = weekly["robots_available"] - weekly["robots_used"]
    weekly["fde_hours_slack"] = weekly["fde_hours_available"] - weekly["fde_hours_used"]
    weekly["robots_utilization"] = weekly["robots_used"] / weekly["robots_available"].where(
        weekly["robots_available"] > 0
    )
    weekly["fde_utilization"] = weekly["fde_hours_used"] / weekly["fde_hours_available"].where(
        weekly["fde_hours_available"] > 0
    )
    slack_robots = {week: robots_avail[week] - used_robots[week] for week in weeks}
    slack_fde = {week: fde_avail[week] - used_fde[week] for week in weeks}
    return weekly, slack_robots, slack_fde


def _reject_infeasible_usage(weekly: pd.DataFrame) -> None:
    if (weekly["robots_used"] - weekly["robots_available"]).max() > 1e-4:
        raise RuntimeError("selected plan exceeds robot capacity")
    if (weekly["fde_hours_used"] - weekly["fde_hours_available"]).max() > 1e-4:
        raise RuntimeError("selected plan exceeds FDE capacity")


def _funded_reason(
    row,
    version: str,
    floor: float,
    threshold: float,
    relaxed: bool,
    median_score: float,
    exploration_cut: float,
    weeks_inside: int,
) -> str:
    duration = int(row.duration_weeks)
    if version == V1:
        credit = float(row.robot_weeks)
        text = (
            f"Funded. Score {float(row.score):.3f} × {credit:.0f} robot-weeks "
            f"adds {float(row.portfolio_value):.2f} to the objective."
        )
        if weeks_inside < duration:
            text += (
                f" {weeks_inside} of {duration} weeks fall inside this plan; "
                "the rest sit in a later quarter."
            )
        if not relaxed and floor > 0 and float(row.task_novelty) >= threshold:
            text += (
                f" Task novelty {float(row.task_novelty):.2f} counts toward "
                f"the exploration floor of {floor:.0%}."
            )
        return text

    credit = float(row.in_horizon_robot_weeks)
    detail = (
        f"Score {float(row.score):.3f} × {credit:.0f} in-horizon robot-weeks "
        f"adds {float(row.portfolio_value):.2f} to the objective."
    )
    if weeks_inside < duration:
        detail += (
            f" {weeks_inside} of {duration} weeks fall inside this plan; "
            "weeks after the horizon are not credited."
        )
    top_scored = float(row.score) + 1e-12 >= median_score
    exploratory = float(row.task_novelty) + 1e-12 >= exploration_cut
    if top_scored:
        text = f"Funded. {detail}"
        if exploratory and not relaxed and floor > 0:
            text += (
                f" Task novelty {float(row.task_novelty):.2f} is at or above the "
                f"eligible median of {exploration_cut:.2f}."
            )
        return text
    if exploratory:
        return (
            f"Funded for exploration. Task novelty {float(row.task_novelty):.2f} "
            f"is at or above the eligible median of {exploration_cut:.2f}. {detail}"
        )
    return f"Funded. Fills residual capacity. {detail}"


def _waitlist_reason(
    row,
    profile,
    slack_robots,
    slack_fde,
    novel_rw: float,
    total_rw: float,
    floor: float,
    threshold: float,
    relaxed: bool,
    version: str,
    funded_lookup: dict,
    profiles: dict,
) -> str:
    if version == V1:
        return _waitlist_reason_v1(
            row, profile, slack_robots, slack_fde, novel_rw, total_rw, floor, threshold, relaxed
        )
    return _waitlist_reason_v2(
        row,
        profile,
        slack_robots,
        slack_fde,
        novel_rw,
        total_rw,
        floor,
        relaxed,
        funded_lookup,
        profiles,
    )


def _waitlist_reason_v1(
    row,
    profile,
    slack_robots,
    slack_fde,
    novel_rw: float,
    total_rw: float,
    floor: float,
    threshold: float,
    relaxed: bool,
) -> str:
    robot_hits = []
    fde_hits = []
    for week, need in profile["robots"].items():
        if need > slack_robots[week] + 1e-6:
            robot_hits.append((need - slack_robots[week], week, need, slack_robots[week]))
    for week, need in profile["fde"].items():
        if need > slack_fde[week] + 1e-6:
            fde_hits.append((need - slack_fde[week], week, need, slack_fde[week]))

    blocks = []
    if robot_hits:
        _, week, need, slack = max(robot_hits)
        blocks.append(
            f"robots short in week {week} (needs {need:.0f}, slack {max(slack, 0):.0f})"
        )
    if fde_hits:
        _, week, need, slack = max(fde_hits)
        blocks.append(
            f"FDE hours short in week {week} (needs {need:.0f}h, slack {max(slack, 0):.0f}h)"
        )
    if blocks:
        return (
            f"Waitlisted. Score {float(row.score):.3f}. Does not fit the residual plan: "
            + "; ".join(blocks)
            + "."
        )

    robot_weeks = float(row.robot_weeks)
    if not relaxed and floor > 0 and robot_weeks > 0 and total_rw > 0:
        added_novel = robot_weeks if float(row.task_novelty) >= threshold else 0.0
        new_share = (novel_rw + added_novel) / (total_rw + robot_weeks)
        if new_share + 1e-9 < floor:
            current = novel_rw / total_rw
            return (
                f"Waitlisted. Score {float(row.score):.3f}. Residual capacity can fit it, "
                f"but adding it would move the exploration share from {current:.1%} "
                f"to {new_share:.1%}, below the {floor:.0%} floor."
            )

    return (
        f"Waitlisted. Score {float(row.score):.3f}. Residual capacity can fit it, "
        "but the selected portfolio scores higher."
    )


def _waitlist_reason_v2(
    row,
    profile,
    slack_robots,
    slack_fde,
    novel_mass: float,
    total_rw: float,
    floor: float,
    relaxed: bool,
    funded_lookup: dict,
    profiles: dict,
) -> str:
    week = _binding_week(profile, slack_robots, slack_fde)
    if week is None:
        return (
            f"Waitlisted. Score {float(row.score):.3f}. "
            "No weeks of this deployment fall inside this plan."
        )
    fde_need = float(profile["fde"][week])
    robot_need = float(profile["robots"][week])
    fde_free = float(slack_fde[week])
    robot_free = float(slack_robots[week])
    short = fde_need > fde_free + 1e-6 or robot_need > robot_free + 1e-6
    needs = (
        f"needs {fde_need:.0f}h FDE / {robot_need:.0f} robots in week {week}; "
        f"{max(fde_free, 0):.0f}h FDE and {max(robot_free, 0):.0f} robots free."
    )
    if short:
        displace = _displacement_clause(row, week, funded_lookup, profiles)
        return f"Waitlisted. {needs}{displace}"

    added = float(row.in_horizon_robot_weeks)
    if not relaxed and floor > 0 and added > 0 and total_rw > 0:
        new_mass = novel_mass + added * float(row.task_novelty)
        new_share = new_mass / (total_rw + added)
        if new_share + 1e-9 < floor:
            current = novel_mass / total_rw
            return (
                f"Waitlisted. {needs} Residual capacity can fit it, but adding it would "
                f"move weighted task novelty from {current:.1%} to {new_share:.1%}, "
                f"below the {floor:.0%} floor."
            )

    return (
        f"Waitlisted. {needs} Residual capacity can fit it, "
        "but the selected portfolio scores higher."
    )


def _binding_week(profile, slack_robots, slack_fde) -> int | None:
    """Week where this request is furthest past the residual slack.

    Ties go to the earlier week. A request that fits everywhere still
    returns the week it comes closest to filling, so the reason can quote
    a need and a free amount.
    """
    active = [
        week
        for week, load in profile["robots"].items()
        if load > 0 or profile["fde"].get(week, 0) > 0
    ]
    if not active:
        return None

    def tightness(week: int) -> tuple[float, float, int]:
        robots_need = float(profile["robots"][week])
        fde_need = float(profile["fde"][week])
        robot_short = max(0.0, robots_need - float(slack_robots[week]))
        fde_short = max(0.0, fde_need - float(slack_fde[week]))
        robot_ratio = robot_short / robots_need if robots_need > 1e-9 else 0.0
        fde_ratio = fde_short / fde_need if fde_need > 1e-9 else 0.0
        return (max(robot_ratio, fde_ratio), robot_short + fde_short, -week)

    return max(active, key=tightness)


def _displacement_clause(row, week: int, funded_lookup: dict, profiles: dict) -> str:
    """Name higher-value funded requests that occupy the binding week."""
    higher = []
    for request_id, funded in funded_lookup.items():
        if float(funded.portfolio_value) <= float(row.portfolio_value) + 1e-9:
            continue
        if profiles[request_id]["robots"].get(week, 0) <= 0 and profiles[request_id]["fde"].get(week, 0) <= 0:
            continue
        higher.append(funded)
    if not higher:
        return ""
    higher.sort(key=lambda funded: (-float(funded.portfolio_value), funded.request_id))
    shown = higher[:3]
    names = []
    for funded in shown:
        customer = getattr(funded, "customer_name", "")
        label = f"{customer} ({funded.request_id})" if customer else funded.request_id
        names.append(label)
    listed = ", ".join(names)
    extra = len(higher) - len(shown)
    if extra > 0:
        listed += f", and {extra} others"
    return f" Funding it would displace {listed}."


def _safe(request_id: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in str(request_id))
    return cleaned or "request"
