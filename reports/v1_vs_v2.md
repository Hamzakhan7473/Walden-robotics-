# v1 vs v2

Same request book and the same 12-week capacity file. v1 is the original scorer and allocator. v2 is the revision: fixed scales, in-quarter robot-weeks only, and a continuous novelty floor. Both runs use an exploration floor of 0.20.

The two objectives are different units. v1 multiplies a min-max score by full-duration robot-weeks. v2 multiplies a different score by in-quarter robot-weeks. The levels should not be subtracted.

## Funded set

v1 funds 12. v2 funds 12. 11 requests are in both.

Only v1: R031 Elmsworth Fulfillment.

Only v2: R012 Pember Provisions.

| Request | In v1 | In v2 | v1 objective | v2 objective | In-quarter robot-weeks |
|---|---|---|---:|---:|---:|
| R007 Quarry Drive Systems | yes | yes | 22.8 | 23.8 | 40 |
| R011 Lowridge Components | yes | yes | 13.9 | 1.9 | 3 |
| R012 Pember Provisions |  | yes | 0.0 | 7.2 | 18 |
| R014 Ironwharf Device Works | yes | yes | 2.9 | 0.7 | 2 |
| R020 Dunlin Stamping | yes | yes | 28.1 | 15.4 | 30 |
| R023 Rookery Chassis | yes | yes | 20.4 | 22.7 | 36 |
| R025 Tarn Circuit Assembly | yes | yes | 19.0 | 10.8 | 18 |
| R029 Keel Microassembly | yes | yes | 14.1 | 15.2 | 30 |
| R031 Elmsworth Fulfillment | yes |  | 9.3 | 0.0 | 18 |
| R033 Whitlow Fulfillment | yes | yes | 7.9 | 9.7 | 18 |
| R035 Marlow Components | yes | yes | 15.1 | 14.7 | 30 |
| R036 Pellin Circuit Assembly | yes | yes | 10.4 | 12.0 | 24 |
| R038 Anvil Beverage Pack | yes | yes | 25.0 | 3.1 | 6 |

v1 still pays Anvil Beverage Pack and Lowridge Components for duration past week 12 (objective credit 25.0 and 13.9). v2 credits the one week of each that falls inside the plan (3.1 and 1.9). Elmsworth Fulfillment is in the v1 plan because the 0.6 floor swaps it in, and it is absent from v2. Pember Provisions is in the v2 plan on its in-quarter score, not because the floor pulled it in.

## Objective

| | v1 | v2 |
|---|---:|---:|
| Objective | 189.05 | 137.16 |
| What the objective multiplies | score × full robot-weeks | score × in-quarter robot-weeks |
| Robot-weeks in the objective | 386 | 255 |
| Of which fall in weeks 1–12 | 255 | 255 |

Under v1, 131 of the 386 credited robot-weeks sit after week 12. Under v2 that tail is zero: every credited robot-week is inside the capacity file.

## Exploration share, in-quarter

In-quarter share here is the novelty-weighted average of funded robot-weeks that fall in weeks 1–12. That is the v2 floor. v1's own floor is a different number: the share of full-duration robot-weeks with task novelty at least 0.6.

| | v1 | v2 |
|---|---:|---:|
| In-quarter weighted novelty, floor 0.20 | 43.2% | 44.7% |
| Same metric with the floor off | 35.4% | 44.7% |
| Version's own floor metric, floor 0.20 | 21.2% of full robot-weeks have novelty ≥ 0.6 | 44.7% weighted novelty |
| Own metric with the floor off | 12.2% | 44.7% |
| Floor changes the funded set | yes | no |

v1's 0.6 cutoff binds: turning it off drops the threshold share from 21.2% to 12.2% and swaps three requests. v2's continuous floor at 0.20 does not bind. The in-quarter weighted novelty is already 44.7% with the floor off, so the constraint is slack. Raising the v2 floor above that level is what would force a different set.

## Utilization per week

Percent of that week's robots and FDE hours used by the funded plan.

| Week | v1 robots | v1 FDE | v2 robots | v2 FDE |
|---:|---:|---:|---:|---:|
| 1 | 42% | 58% | 42% | 58% |
| 2 | 88% | 99% | 88% | 99% |
| 3 | 88% | 99% | 88% | 99% |
| 4 | 88% | 99% | 88% | 99% |
| 5 | 81% | 99% | 81% | 99% |
| 6 | 81% | 99% | 81% | 99% |
| 7 | 96% | 90% | 96% | 95% |
| 8 | 96% | 90% | 96% | 95% |
| 9 | 89% | 90% | 89% | 95% |
| 10 | 71% | 71% | 71% | 77% |
| 11 | 75% | 80% | 75% | 86% |
| 12 | 86% | 93% | 86% | 99% |

Peak robot use is 96% in v1 and 96% in v2. Peak FDE use is 99% in v1 and 99% in v2.

## Sensitivity

200 draws, Gaussian weight noise with standard deviation 0.06, seed 42, floor 0.20. Positive weights are rescaled to their baseline total. v1 keeps the cost weight non-positive. v2 has no cost weight.

A request is **robust** when its funded rate is at most 20% or at least 80%. It is **contested** when the rate is strictly between those cuts. Counts are out of all 40 requests, including requests a gate rejects. Those stay at a funded rate of zero and count as robust.

| | v1 | v2 |
|---|---:|---:|
| Robust | 40 | 40 |
| Contested | 0 | 0 |
| Mean overlap with the baseline funded set | 99.8% | 98.2% |
| Objective, 10th–90th percentile | 180.1–197.6 | 132.2–142.0 |
| Draws that relaxed the floor | 0% | 0% |

Contested under v1:

None.

Contested under v2:

None.

Nothing lands in the contested band. The overlap below 100% is a few requests that move in under 5% of draws: in v1, Quarry, Whitlow, and Elmsworth drop to 99.5% funded, and Lantern, Fielding, and Millshaw appear in 0.5% of draws. In v2, Pellin, Pember, and Ironwharf drop to 95.5% funded, and Cinder, Soren, and Millshaw appear in 4.5% of draws. Under the 20/80 cut those six on each side stay robust.
