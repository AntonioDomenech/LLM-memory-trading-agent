# Binary-regime union selector v1: strict rejection

Status: **rejected at the frozen 2019-2023 selector validation**. The branch
must not open or evaluate 2024 or later data.

The selector was frozen, audited, committed, and pushed before its first
2019-2023 row was loaded. Its 2000-2018 calibration passed, but calibration is
training evidence. The inherited expert family is also not globally pristine:
earlier research had already used later periods to select parts of that family.
This validation therefore tests the new selector, not the entire end-to-end
research history.

## What happened on 2019-2023

| Cost per changing leg | Log edge vs AAPL | Wealth vs AAPL | Log edge vs exact union | Wealth vs exact union | Positive years vs AAPL |
|---|---:|---:|---:|---:|---:|
| 5 bps | +0.074905437683 | +7.778223% | +0.003081945017 | +0.308670% | 4/5 |
| 10 bps | +0.038905416683 | +3.967214% | +0.006081946767 | +0.610048% | 4/5 |

The selector therefore did beat same-ledger buy-and-hold in aggregate at both
cost assumptions. It also beat buy-and-hold in the negative AAPL year 2022,
executed 36 complete cash episodes, and passed the episode mean, median,
concentration, best-year-removal, no-leverage, and accounting gates.

It nevertheless failed three pre-registered robustness gates:

- At both costs, selector-minus-union edge was positive in only one of five
  years, not the required two. It helped in 2020, hurt in 2021, and was exactly
  unchanged in 2019, 2022, and 2023.
- At 10 bps, the largest helpful veto supplied 85.20% of all positive veto
  benefit, above the frozen 50% limit.

Only three union opportunities were vetoed. Two helped, but the incremental
benefit was too sparse and concentrated to establish that the new regime model
reliably improves the simpler union.

## The learning component did not change behavior

The causal learner did continue admitting newly matured lessons: risk-on grew
from 25 to 28 lessons and not-risk-on from 138 to 174. Their final discounted
means remained on the same sides of the frozen thresholds, so neither latch
switched. The online and frozen-2018 action streams and metrics are therefore
byte-identical.

In plain language: the system kept learning, but the new information did not
change a single trade. Most of the validation gain came from the inherited
fixed union, not from adaptive prediction.

## Long-run diagnostic retained as a research lead

The continuous 2005-2023 account is strong enough to preserve as a long-run
lead, but not to override the strict rejection:

| Cost | Total log edge vs AAPL | Relative wealth vs AAPL | Positive / negative / tied years | Negative-AAPL years beaten |
|---|---:|---:|---:|---:|
| 5 bps | +1.276733935090 | +258.491203% | 16 / 2 / 1 | 4/4 |
| 10 bps | +1.139733855173 | +212.593630% | 14 / 4 / 1 | 4/4 |

This long-run figure includes the in-sample 2005-2018 calibration period and
inherits the expert-family selection risk. It is evidence that the underlying
long/cash union remains worth studying, not proof of a reliable autonomous
money-making agent.

## Integrity and cost

- One account starts in 2005; reporting years never reset it.
- Targets and realized exposure stay between 0 and 1.
- There is no shorting, leverage, borrowing, negative cash, margin interest,
  network access, API call, LLM call, or external cost.
- The validation finished in 30.52 seconds.
- All 28 sealed artifact hashes match `checksums.json`.
- No post-2023 row was loaded or returned.

The immutable validation run is
`binary-regime-union-selector-validation-v1`. Its stage manifest self-hash is
`sha256:0354355ac460042f96663d1a45cf5e9a8cf4fe873ddf5cc87afefd9c4b5d81dc`.
