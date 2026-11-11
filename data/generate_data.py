"""Seeded synthetic book of deployment requests.

Seed 42 fixes the draw. The order of random draws inside ``build_requests``
is part of that contract: do not reorder them if you want the same file.

This is not Walden operational data. Every distribution choice is commented
where it is made. Narrative sources, and which assumptions you will defend,
belong in DATA_NOTES.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_REQUESTS = 40
HORIZON_WEEKS = 12

# Assumption: deployable fleet starts at 24 units. Two more units clear
# and become deployable every 4 weeks (weeks 5 and 9). No maintenance
# downtime is modeled inside the quarter.
FLEET_START = 24
FLEET_INCREMENT = 2
FLEET_STEP_WEEKS = 4

# Assumption: 6 field deployment engineers, 40-hour weeks, 0.75 utilization
# after internal work, travel overhead, and time off. Headcount does not
# ramp inside the quarter; hiring an FDE is slower than shipping a robot.
# 6 * 40 * 0.75 = 180 hours every week.
FDE_HEADCOUNT = 6
FDE_HOURS_PER_WEEK = 40
FDE_UTILIZATION = 0.75

# Assumption: fictional compounds, not sampled from a real account list.
# A collision with a real firm's name would be accidental.
STEMS = [
    "Calder",
    "Northfen",
    "Millshaw",
    "Redoak",
    "Harlow",
    "Bexley",
    "Quarry",
    "Lantern",
    "Westmere",
    "Cinder",
    "Lowridge",
    "Pember",
    "Kittredge",
    "Ironwharf",
    "Dovetail",
    "Bracken",
    "Yarrow",
    "Holt",
    "Merrick",
    "Dunlin",
    "Plover",
    "Ashcombe",
    "Rookery",
    "Fielding",
    "Tarn",
    "Hale",
    "Oster",
    "Vesper",
    "Keel",
    "Norden",
    "Elmsworth",
    "Cairn",
    "Whitlow",
    "Soren",
    "Marlow",
    "Pellin",
    "Greave",
    "Anvil",
    "Bromley",
    "Lumenfold",
]

TAILS = {
    "auto_oem": ["Motor Works", "Drive Systems", "Vehicle Assembly", "Chassis"],
    "tier1_supplier": ["Components", "Stamping", "Precision Parts", "Formed Metal"],
    "3pl_warehouse": ["Logistics", "Cartage", "Fulfillment", "Warehousing"],
    "electronics": ["Electronics", "Circuit Assembly", "Device Works", "Microassembly"],
    "food_bev": ["Foods", "Beverage Pack", "Provisions", "Cold Chain"],
}

# Assumption: the inbound book is logistics-heavy. Auto OEM programs are
# fewer and, further down, larger. Manufacturing and logistics are both
# in the mix; 3PL volume is the thick part of the book.
SEGMENT_P = {
    "auto_oem": 0.15,
    "tier1_supplier": 0.22,
    "3pl_warehouse": 0.30,
    "electronics": 0.18,
    "food_bev": 0.15,
}

# Assumption: tier mix of a company still building reference deployments.
# Anchors are scarce. Most of the book is standard.
TIER_P = {
    "anchor": 0.08,
    "investor_linked": 0.12,
    "reference_logo": 0.22,
    "standard": 0.58,
}

# Assumption: task mix follows the work those segments actually buy.
# Auto and tier-1 cells tend machines and assist assembly. 3PLs palletize,
# kit, and pack. Electronics adds bin picking. Food is pack and pallet heavy.
TASK_P = {
    "auto_oem": {
        "machine_tending": 0.45,
        "assembly_assist": 0.25,
        "inspection": 0.20,
        "kitting": 0.10,
    },
    "tier1_supplier": {
        "machine_tending": 0.35,
        "kitting": 0.25,
        "inspection": 0.20,
        "assembly_assist": 0.15,
        "packing": 0.05,
    },
    "3pl_warehouse": {
        "palletizing": 0.35,
        "kitting": 0.25,
        "packing": 0.20,
        "bin_picking": 0.15,
        "inspection": 0.05,
    },
    "electronics": {
        "bin_picking": 0.30,
        "kitting": 0.25,
        "inspection": 0.20,
        "assembly_assist": 0.15,
        "packing": 0.10,
    },
    "food_bev": {
        "packing": 0.35,
        "palletizing": 0.25,
        "kitting": 0.20,
        "inspection": 0.15,
        "machine_tending": 0.05,
    },
}

# Assumption: distance from a skill library that is already strong on
# palletizing, machine tending, and packing, and weak on bin picking and
# assembly assist. This is a distance, not a judgment of task value.
# Inspection sits under the allocator's default novelty threshold of 0.6
# so the exploration floor has something to pull against.
TASK_NOVELTY_CENTER = {
    "palletizing": 0.16,
    "machine_tending": 0.20,
    "packing": 0.28,
    "kitting": 0.40,
    "inspection": 0.48,
    "assembly_assist": 0.64,
    "bin_picking": 0.76,
}

# Assumption: food and beverage adds novelty (hygiene, deformable packs,
# changeover). Electronics adds a little (small parts). Auto OEM cells
# sit closer to the library.
SEGMENT_NOVELTY_SHIFT = {
    "auto_oem": -0.06,
    "tier1_supplier": -0.02,
    "3pl_warehouse": 0.00,
    "electronics": 0.05,
    "food_bev": 0.08,
}

# Assumption: Gaussian noise on novelty, sd 0.06, then clip to [0.02, 0.98]
# so no task is recorded as perfectly known or perfectly new.
NOVELTY_NOISE_SD = 0.06
NOVELTY_CLIP = (0.02, 0.98)

# Assumption: most asks are 1-3 units. Anchor accounts and auto OEM
# programs ask for larger cells. Categorical on 1..6.
ROBOT_LEVELS = [1, 2, 3, 4, 5, 6]
BASE_ROBOT_P = [0.16, 0.34, 0.26, 0.14, 0.07, 0.03]
LARGE_ROBOT_P = [0.04, 0.12, 0.26, 0.28, 0.18, 0.12]

# Assumption: durations are quoted in even week blocks. Novel tasks
# (novelty >= 0.55, a generation cutoff, not the policy threshold) are
# more often 4-8 week pilots. Known tasks are more often 8-12 week cells.
DURATION_LEVELS = [4, 6, 8, 10, 12]
KNOWN_DURATION_P = [0.06, 0.14, 0.32, 0.28, 0.20]
NOVEL_DURATION_P = [0.30, 0.34, 0.22, 0.10, 0.04]
DURATION_NOVELTY_SPLIT = 0.55

# Assumption: booking value per robot-week, not margin. Auto OEM is higher
# (line-rate penalties and process rigor). 3PL is lower. A lognormal
# multiplier (sigma 0.12) adds spread without a runaway tail. Novel work
# is discounted as a pilot. Anchor programs carry a premium.
USD_PER_ROBOT_WEEK = {
    "auto_oem": 18000,
    "tier1_supplier": 12500,
    "electronics": 13500,
    "3pl_warehouse": 8500,
    "food_bev": 9500,
}
PRICE_NOISE_SIGMA = 0.12
NOVELTY_PRICE_DISCOUNT = 0.18
TIER_PRICE = {
    "anchor": 1.20,
    "investor_linked": 1.08,
    "reference_logo": 1.05,
    "standard": 1.00,
}

# Assumption: expansion is extra robots the customer could add within a
# year if the first cell works. It scales with segment footprint and tier.
# Standard-tier requests are pilots with no stated follow-on 35% of the time.
# Capped at 40 so one logo cannot own the expansion term by itself.
EXPANSION_SEGMENT = {
    "auto_oem": 3.5,
    "tier1_supplier": 2.2,
    "3pl_warehouse": 2.8,
    "electronics": 1.8,
    "food_bev": 1.4,
}
EXPANSION_TIER = {
    "anchor": 2.2,
    "investor_linked": 1.6,
    "reference_logo": 1.3,
    "standard": 0.7,
}
STANDARD_ZERO_EXPANSION_P = 0.35
EXPANSION_CAP = 40

# Assumption: a known task at a typical site succeeds about 88% of the time
# before noise. A fully novel task succeeds about 46% before the readiness
# adjustment. Readiness moves the probability about 3 points per step away
# from 3. Noise sd is 0.04. Clip to [0.30, 0.95]: nothing is certain, and
# the hard gates, not this draw, remove the truly unready.
SUCCESS_INTERCEPT = 0.90
SUCCESS_NOVELTY_SLOPE = 0.42
SUCCESS_READINESS_SLOPE = 0.03
SUCCESS_NOISE_SD = 0.04
SUCCESS_CLIP = (0.30, 0.95)

# Assumption: about 15% of inbound requests should fail at least one hard
# gate, as a union, not 15% per gate. Anchor and investor-linked work is
# later in the sales cycle and rarely fails. Novel tasks are slightly more
# likely to still be in safety review. The realized rate under seed 42 is
# checked in tests; it is not forced by rejecting and redrawing.
TIER_GATE_MULT = {
    "anchor": 0.20,
    "investor_linked": 0.35,
    "reference_logo": 0.85,
    "standard": 1.20,
}
BASE_SAFETY_FAIL = 0.05
SAFETY_NOVELTY_LIFT = 0.80
BASE_LOI_FAIL = 0.07

# Assumption: readiness is ordinal 1-5, mode at 4. A 1 fails the gate.
# A 2 passes and then scores poorly. Unsigned LOIs are redrawn from a
# worse distribution so commercial immaturity and site immaturity travel
# together.
READINESS_LEVELS = [1, 2, 3, 4, 5]
READINESS_P = [0.04, 0.11, 0.24, 0.40, 0.21]
READINESS_NO_LOI_P = [0.18, 0.32, 0.28, 0.16, 0.06]
READINESS_FACTOR = {1: 1.45, 2: 1.25, 3: 1.05, 4: 0.90, 5: 0.75}

# Assumption: FDE hours are a mobilization plus a per-robot commissioning
# cost. The per-robot term rises steeply with novelty and falls when the
# site is ready. Rounded to the nearest hour.
FDE_MOBILIZATION = (36.0, 72.0)
FDE_PER_ROBOT = (28.0, 48.0)
FDE_NOVELTY_LIFT = 1.6

# Assumption: requested starts are front-loaded. Weights follow 0.80 ** (week-1),
# renormalized. Readiness 2 slips 1-3 weeks; readiness 1 slips 2-4. Clipped
# to week 12 so every request touches this horizon.
START_DECAY = 0.80

REQUEST_COLUMNS = [
    "request_id",
    "customer_name",
    "segment",
    "task_type",
    "robots_requested",
    "duration_weeks",
    "fde_hours_required",
    "contract_value_usd",
    "expansion_units_potential",
    "success_prob",
    "task_novelty",
    "site_readiness",
    "strategic_tier",
    "safety_review_passed",
    "loi_signed",
    "requested_start_week",
]


def _check_probabilities() -> None:
    tables = {
        "SEGMENT_P": SEGMENT_P,
        "TIER_P": TIER_P,
        "BASE_ROBOT_P": dict(zip(ROBOT_LEVELS, BASE_ROBOT_P)),
        "LARGE_ROBOT_P": dict(zip(ROBOT_LEVELS, LARGE_ROBOT_P)),
        "KNOWN_DURATION_P": dict(zip(DURATION_LEVELS, KNOWN_DURATION_P)),
        "NOVEL_DURATION_P": dict(zip(DURATION_LEVELS, NOVEL_DURATION_P)),
        "READINESS_P": dict(zip(READINESS_LEVELS, READINESS_P)),
        "READINESS_NO_LOI_P": dict(zip(READINESS_LEVELS, READINESS_NO_LOI_P)),
    }
    for segment, tasks in TASK_P.items():
        tables[f"TASK_P[{segment}]"] = tasks
    for name, table in tables.items():
        total = float(sum(table.values()))
        if abs(total - 1.0) > 1e-9:
            raise RuntimeError(f"{name} probabilities sum to {total}")


_check_probabilities()


def _pick(rng: np.random.Generator, labels: list, probs: list[float]):
    weights = np.asarray(probs, dtype=float)
    weights = weights / weights.sum()
    return rng.choice(np.asarray(labels), p=weights)


def build_capacity(horizon: int = HORIZON_WEEKS) -> pd.DataFrame:
    """Weekly robot and FDE capacity for the planning horizon."""
    weeks = np.arange(1, horizon + 1)
    robots = FLEET_START + FLEET_INCREMENT * ((weeks - 1) // FLEET_STEP_WEEKS)
    fde_hours = int(FDE_HEADCOUNT * FDE_HOURS_PER_WEEK * FDE_UTILIZATION)
    return pd.DataFrame(
        {
            "week": weeks.astype(int),
            "robots_available": robots.astype(int),
            "fde_hours_available": np.full(horizon, fde_hours, dtype=int),
        }
    )


def build_requests(seed: int = SEED, n: int = N_REQUESTS) -> pd.DataFrame:
    """Build ``n`` deployment requests from a single numpy Generator.

    Draw order per row, which the seed depends on:
    segment, tier, name suffix, task, novelty, safety, LOI, readiness,
    robots, duration, success, price, expansion, FDE hours, start week.
    """
    if n > len(STEMS):
        raise ValueError(f"n={n} exceeds the {len(STEMS)} fictional name stems")
    if n < 1:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    start_weeks = np.arange(1, HORIZON_WEEKS + 1)
    start_p = START_DECAY ** (start_weeks - 1)

    rows: list[dict] = []
    for i in range(n):
        segment = str(_pick(rng, list(SEGMENT_P), list(SEGMENT_P.values())))
        tier = str(_pick(rng, list(TIER_P), list(TIER_P.values())))
        tail = str(rng.choice(np.asarray(TAILS[segment])))
        customer = f"{STEMS[i]} {tail}"

        task_table = TASK_P[segment]
        task = str(_pick(rng, list(task_table), list(task_table.values())))

        novelty = (
            TASK_NOVELTY_CENTER[task]
            + SEGMENT_NOVELTY_SHIFT[segment]
            + float(rng.normal(0.0, NOVELTY_NOISE_SD))
        )
        novelty = float(np.clip(novelty, *NOVELTY_CLIP))
        novelty = round(novelty, 4)

        # Safety and LOI are drawn before readiness so an unsigned LOI can
        # pull the site score down.
        p_safety_fail = min(
            0.85,
            BASE_SAFETY_FAIL
            * TIER_GATE_MULT[tier]
            * (1.0 + SAFETY_NOVELTY_LIFT * novelty),
        )
        safety_passed = bool(rng.random() >= p_safety_fail)
        p_loi_fail = min(0.85, BASE_LOI_FAIL * TIER_GATE_MULT[tier])
        loi_signed = bool(rng.random() >= p_loi_fail)

        if loi_signed:
            readiness = int(_pick(rng, READINESS_LEVELS, READINESS_P))
        else:
            readiness = int(_pick(rng, READINESS_LEVELS, READINESS_NO_LOI_P))

        robot_p = LARGE_ROBOT_P if tier == "anchor" or segment == "auto_oem" else BASE_ROBOT_P
        robots = int(_pick(rng, ROBOT_LEVELS, robot_p))

        duration_p = (
            NOVEL_DURATION_P if novelty >= DURATION_NOVELTY_SPLIT else KNOWN_DURATION_P
        )
        duration = int(_pick(rng, DURATION_LEVELS, duration_p))

        success = (
            SUCCESS_INTERCEPT
            - SUCCESS_NOVELTY_SLOPE * novelty
            + SUCCESS_READINESS_SLOPE * (readiness - 3)
            + float(rng.normal(0.0, SUCCESS_NOISE_SD))
        )
        success = float(np.clip(success, *SUCCESS_CLIP))
        success = round(success, 4)

        price_mult = float(np.exp(rng.normal(0.0, PRICE_NOISE_SIGMA)))
        price_mult *= 1.0 - NOVELTY_PRICE_DISCOUNT * novelty
        price_mult *= TIER_PRICE[tier]
        contract = int(round(robots * duration * USD_PER_ROBOT_WEEK[segment] * price_mult))
        contract = int(max(1000, round(contract / 1000) * 1000))

        expansion = robots * EXPANSION_SEGMENT[segment] * EXPANSION_TIER[tier]
        expansion *= float(rng.uniform(0.7, 1.3))
        expansion = int(np.clip(round(expansion), 0, EXPANSION_CAP))
        if tier == "standard" and float(rng.random()) < STANDARD_ZERO_EXPANSION_P:
            expansion = 0

        mobilization = float(rng.uniform(*FDE_MOBILIZATION))
        per_robot = float(rng.uniform(*FDE_PER_ROBOT))
        fde_hours = mobilization + per_robot * robots * (
            1.0 + FDE_NOVELTY_LIFT * novelty
        ) * READINESS_FACTOR[readiness]
        fde_hours = int(max(24, round(fde_hours)))

        start = int(_pick(rng, list(start_weeks), list(start_p)))
        if readiness == 2:
            start = min(HORIZON_WEEKS, start + int(rng.integers(1, 4)))
        elif readiness == 1:
            start = min(HORIZON_WEEKS, start + int(rng.integers(2, 5)))

        rows.append(
            {
                "request_id": f"R{i + 1:03d}",
                "customer_name": customer,
                "segment": segment,
                "task_type": task,
                "robots_requested": robots,
                "duration_weeks": duration,
                "fde_hours_required": fde_hours,
                "contract_value_usd": contract,
                "expansion_units_potential": expansion,
                "success_prob": success,
                "task_novelty": novelty,
                "site_readiness": readiness,
                "strategic_tier": tier,
                "safety_review_passed": safety_passed,
                "loi_signed": loi_signed,
                "requested_start_week": start,
            }
        )

    frame = pd.DataFrame(rows, columns=REQUEST_COLUMNS)
    _validate(frame)
    return frame


def _validate(frame: pd.DataFrame) -> None:
    tasks = {task for table in TASK_P.values() for task in table}
    if frame["request_id"].duplicated().any():
        raise RuntimeError("duplicate request_id")
    if frame["customer_name"].duplicated().any():
        raise RuntimeError("duplicate customer_name")
    if not set(frame["segment"]).issubset(SEGMENT_P):
        raise RuntimeError("unexpected segment")
    if not set(frame["task_type"]).issubset(tasks):
        raise RuntimeError("unexpected task_type")
    if not set(frame["strategic_tier"]).issubset(TIER_P):
        raise RuntimeError("unexpected strategic_tier")
    if not frame["robots_requested"].between(1, 6).all():
        raise RuntimeError("robots_requested out of range")
    if not frame["duration_weeks"].isin(DURATION_LEVELS).all():
        raise RuntimeError("duration_weeks out of range")
    if not frame["success_prob"].between(0, 1).all():
        raise RuntimeError("success_prob out of range")
    if not frame["task_novelty"].between(0, 1).all():
        raise RuntimeError("task_novelty out of range")
    if not frame["site_readiness"].between(1, 5).all():
        raise RuntimeError("site_readiness out of range")
    if not frame["requested_start_week"].between(1, HORIZON_WEEKS).all():
        raise RuntimeError("requested_start_week out of range")
    if (frame["fde_hours_required"] <= 0).any() or (frame["contract_value_usd"] <= 0).any():
        raise RuntimeError("non-positive hours or contract value")
    if (frame["expansion_units_potential"] < 0).any():
        raise RuntimeError("negative expansion")


def _print_summary(requests: pd.DataFrame, capacity: pd.DataFrame) -> None:
    """Print a check of the book and of the default allocation."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.allocate import allocate
    from src.scoring import score_requests

    scored = score_requests(requests)
    fail_rate = float((~scored["gate_pass"]).mean())
    eligible = scored[scored["gate_pass"]].copy()
    corr = float(requests["task_novelty"].corr(requests["success_prob"]))
    print(f"requests: {len(requests)}")
    print(f"gate failure rate: {fail_rate:.1%} ({int((~scored['gate_pass']).sum())} requests)")
    print(f"corr(task_novelty, success_prob): {corr:.3f}")
    print("mean contract value by segment:")
    means = requests.groupby("segment")["contract_value_usd"].mean().sort_values(ascending=False)
    for segment, value in means.items():
        print(f"  {segment}: ${value:,.0f}")
    novel = eligible["task_novelty"] >= 0.6
    eligible_rw = float(eligible["robot_weeks"].sum())
    novel_rw = float(eligible.loc[novel, "robot_weeks"].sum())
    supply_rw = float(
        (capacity["robots_available"]).sum()
    )  # robot-weeks if the whole fleet is busy all quarter
    print(f"eligible robot-weeks: {eligible_rw:.0f}  |  fleet-weeks of supply: {supply_rw:.0f}")
    print(f"eligible robot-weeks with novelty >= 0.6: {novel_rw:.0f} ({novel_rw / eligible_rw:.1%})")

    held = allocate(scored, capacity, exploration_floor=0.20)
    free = allocate(scored, capacity, exploration_floor=0.0)
    print(
        f"with floor 20%: funded {int((held.decisions.decision == 'funded').sum())}, "
        f"exploration share {held.exploration_share:.1%}, "
        f"peak robots {held.weekly['robots_utilization'].max():.0%}, "
        f"peak FDE {held.weekly['fde_utilization'].max():.0%}"
    )
    print(
        f"with floor 0%:  funded {int((free.decisions.decision == 'funded').sum())}, "
        f"exploration share {free.exploration_share:.1%}"
    )
    funded_ids = set(held.decisions.loc[held.decisions.decision == "funded", "request_id"])
    free_ids = set(free.decisions.loc[free.decisions.decision == "funded", "request_id"])
    print(f"floor changes the funded set: {funded_ids != free_ids}")


def main() -> None:
    requests = build_requests()
    capacity = build_capacity()
    out = Path(__file__).resolve().parent
    requests.to_csv(out / "requests.csv", index=False)
    capacity.to_csv(out / "capacity.csv", index=False)
    print(f"wrote {out / 'requests.csv'}")
    print(f"wrote {out / 'capacity.csv'}")
    _print_summary(requests, capacity)


if __name__ == "__main__":
    main()
