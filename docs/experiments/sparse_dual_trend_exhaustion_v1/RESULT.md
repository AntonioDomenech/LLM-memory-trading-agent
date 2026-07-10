# Sparse dual-trend exhaustion v1 — rejected

Run: `aapl-unleveraged-sparse_dual_trend_exhaustion_v1-20260710T143927Z-8847c04c`

This candidate was frozen by a separate pre-2024-only search. It briefly exits
after a top-2.5% AAPL intraday move only when AAPL is below both its 63- and
126-session averages. It is a provisional finalist, not a strict pre-2024 gate
winner, because its sparse inactive years create many exact ties.

## Required final periods at 5 bps

| Period | Strategy | AAPL buy-and-hold | Excess | Terminal-close excess |
|---|---:|---:|---:|---:|
| 2024 | 34.4951% | 35.4795% | -0.9844 pp | -0.9765 pp |
| 2025 | 15.2240% | 10.1345% | +5.0895 pp | +5.0671 pp |
| 2026 YTD | 14.2039% | 14.2039% | 0.0000 pp | 0.0000 pp |

It is closer in 2024 than the earlier exhaustion rules, but still loses; it
makes no 2026 timing decision and exactly ties. The continuous account finishes
3.8610% ahead in relative wealth, but both required promotion gates fail.

## Negative buy-and-hold years

The rule beat buy-and-hold in all six losing AAPL years from 2000–2023, though
only 2015 ended with a positive absolute return.

| Year | Strategy | AAPL buy-and-hold | Excess |
|---|---:|---:|---:|
| 2000 | -71.1204% | -72.0044% | +0.8841 pp |
| 2002 | -34.5425% | -36.5397% | +1.9972 pp |
| 2008 | -36.6404% | -56.8791% | +20.2387 pp |
| 2015 | +7.9447% | -2.3441% | +10.2888 pp |
| 2018 | -5.1632% | -5.4833% | +0.3201 pp |
| 2022 | -24.9340% | -27.4078% | +2.4738 pp |

Across all 24 years it won 11 and tied many inactive years. Mean annual excess
was +3.1135 percentage points, median zero, and worst -1.3229 points.

## Integrity

- Runtime: 6.4260 seconds.
- Holdout reveal index: 9. Index 8 was conservatively consumed by a run stopped
  during the pre-2024 audit because its warm-up behavior was unspecified; that
  failed run never reached the final outcomes.
- Warm-up policy: remain fully invested until all 252 observations mature.
- Approved price snapshot and complete session sequence: passed.
- Model/API calls and external cost: zero / `$0.00`.
- Maximum exposure: 1.0; minimum cash and shares: 0.0.
- Margin interest and integrity errors: zero / none.
- Candidate SHA-256: `a6b0f869ac34e8345d25d58f62f568334cec4a33efc2183addd9a3030b8bbebb`.

This is a useful defensive rule, but it does not satisfy the all-period goal.
