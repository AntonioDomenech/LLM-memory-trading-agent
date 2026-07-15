# Binary-regime selector long-run audit v1: strict rejection, long-run pass

Status: **rejected for the main 2024/2025/2026-YTD goal** because it
underperformed same-ledger AAPL in 2024 at both cost assumptions. The separate
post-hoc long-run robustness test passed every gate, so the approach remains a
strong historical long-run research lead rather than a successful recent-period
system.

The frozen 2019-2023 validation rejection remains final. This audit did not
refit the selector, union, checkpoint, thresholds, or gates. It extended the
same continuous account and causal learner through 2026-07-09 exactly once.

## Exact recent-period result

| Cost per changing leg | Period | Strategy return | AAPL buy-and-hold | Excess return | Active log edge |
|---|---|---:|---:|---:|---:|
| 5 bps | 2024 | +27.4184% | +30.8286% | **-3.4102 pp** | -0.0264118383 |
| 5 bps | 2025 | +24.8517% | +8.6575% | **+16.1942 pp** | +0.1389256227 |
| 5 bps | 2026 YTD | +15.8327% | +13.9262% | **+1.9065 pp** | +0.0165956272 |
| 10 bps | 2024 | +26.7829% | +30.8286% | **-4.0457 pp** | -0.0314118412 |
| 10 bps | 2025 | +23.7330% | +8.6575% | **+15.0755 pp** | +0.1299256174 |
| 10 bps | 2026 YTD | +15.3703% | +13.9262% | **+1.4440 pp** | +0.0125956249 |

The strict report passed 21 of 23 gates. Its only failures were the frozen
2024 active-edge-above-0.001 gate at 5 and 10 bps. This is substantive
underperformance, not rounding noise. The 2025 and 2026-YTD performance gates,
all accounting controls, and all no-leverage controls passed.

Across the combined 2024-2026-YTD window, the strategy still finished ahead:

| Cost | Strategy compounded return | AAPL compounded return | Relative wealth vs AAPL |
|---|---:|---:|---:|
| 5 bps | +84.2713% | +61.9519% | +13.7815% |
| 10 bps | +80.9841% | +61.9519% | +11.7517% |

That aggregate result cannot rescue the strict failure because the goal
requires a material win in each named period, including 2024.

## Long-run robustness result

The continuous account has an administrative start of 2005-01-01, reaches its
first executable fill/session on 2005-01-03, and never resets. All 41 post-hoc
long-run gates passed:

| Measure, 2005 through 2026-07-09 | 5 bps | 10 bps |
|---|---:|---:|
| Strategy ending value from $1,000 | $1,305,605.48 | $1,117,581.74 |
| AAPL ending value from $1,000 | $320,082.52 | $319,922.64 |
| Strategy total return | +130,460.55% | +111,658.17% |
| AAPL total return | +31,908.25% | +31,892.26% |
| Ending wealth / AAPL ending wealth | 4.078965x | 3.493287x |
| Active log edge | +1.4058433467 | +1.2508432563 |
| Positive reporting periods | 18 / 22 | 16 / 22 |
| Complete cash episodes | 155 | 155 |
| Winning episodes | 97 | 93 |
| Episode win rate | 62.58% | 60.00% |
| Mean episode log edge | +0.00906996 | +0.00806996 |
| Median episode log edge | +0.00810048 | +0.00710048 |
| Strategy maximum drawdown | -42.99% | -43.22% |
| AAPL maximum drawdown | -60.42% | -60.42% |
| Edge after removing the best period | +0.904124 | +0.772124 |
| Edge after removing the five best episodes | +0.897881 | +0.747881 |
| Selector-minus-fixed-union log edge | +0.117965 | +0.143965 |

The negative-AAPL years were 2008, 2015, 2018, and 2022. Aggregate active log
edge across those years was +0.841144 at 5 bps and +0.784143 at 10 bps:

| Year | AAPL | Strategy, 5 bps | Strategy, 10 bps |
|---|---:|---:|---:|
| 2008 | -56.91% | -28.83% | -30.45% |
| 2015 | -3.53% | +8.18% | +7.21% |
| 2018 | -5.64% | +16.34% | +14.72% |
| 2022 | -27.48% | -26.34% | -27.15% |

This is the clearest chronological long-run lead so far. It is still
post-hoc, includes 2005-2018 training evidence, and inherits selection risk
from the underlying expert family. It is not prospective proof of reliable
profit.

## Continued learning did not affect a trade

The online arm admitted 21 new lessons after 2023, but they caused:

- 0 state-threshold crossings;
- 0 action differences across 631 post-2023 forecasts;
- 0 divergence runs or differing episodes; and
- exactly 0 incremental edge versus the frozen-2023 arm.

The online and frozen metrics are byte-identical. In plain language, internal
statistics kept updating, but the updates never changed a decision. The strong
historical result therefore supports the already-learned fixed selector, not a
claim that continual learning added value after 2023.

## Integrity, cost, and the one-attempt record

- Targets, holding exposure, and post-fill exposure stayed in `[0, 1]`.
- Cash and shares never became negative; there was no shorting, leverage,
  borrowing, or margin interest.
- Runtime before sealing was 68.27 seconds, below the one-hour limit.
- Network, news, LLM, and API calls were all zero; external cost was $0.
- All 47 sealed artifact checksums and the complete seal contract independently
  recomputed successfully.
- The final input contains exactly 6,875 sessions from 1999-03-10 through
  2026-07-09 and has canonical result hash
  `sha256:c01447f975d4a90e49c315f23177f357966363b1ec4790632fa54c0dee250b21`.

The final bounded input identity was corrected and pushed at
`9bab9e023563d7ccaa4586d59dab464c0297da49` before the consumed audit. The
sealed evidence proves that the one consumed audit attempt is bound to that
commit, that its persistent lock was created before any raw or post-2023 value
was read, and that the tracked input matched its committed blob.

The persistent attempt-lock hash is
`sha256:45fe7ed8ca4949a87428cb1557c62bcee541e01d690cf76c0b14ab74fcd07c65`.
The sealed manifest self-hash is
`sha256:ad59b77b27ead2012c8ae73e422ba85b26bb58121fbefedaecfa5640c7f87cdb`.

Historical results do not authorize real capital. Under the frozen contract,
`stage_pass=false`, `fixed_policy_candidate_for_prospective_paper=false`, and
`online_learning_candidate_for_paper=false`.
