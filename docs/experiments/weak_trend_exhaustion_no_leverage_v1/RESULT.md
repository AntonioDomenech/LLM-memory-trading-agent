# Weak-trend exhaustion v1 — rejected

Run: `aapl-unleveraged-weak_trend_exhaustion_v1-20260710T143346Z-d6ffe1f3`

This explicitly adaptive family was proposed after diagnosing the first
contextual-exhaustion failure. A 675-rule search was then ranked using only
2000–2023 outcomes. The frozen rule exits for one next-open interval when AAPL
intraday exhaustion occurs while AAPL is below its 20-day average and both SPY
and QQQ have negative 20-session momentum.

## Required final periods at 5 bps

| Period | Strategy | AAPL buy-and-hold | Excess | Terminal-close excess |
|---|---:|---:|---:|---:|
| 2024 | 31.9575% | 35.4795% | -3.5220 pp | -3.4938 pp |
| 2025 | 27.6713% | 10.1345% | +17.5368 pp | +17.4597 pp |
| 2026 YTD | 15.7902% | 14.2039% | +1.5863 pp | +1.6155 pp |

The filter strengthens 2025 substantially and the continuous 2024–2026
account finishes 14.4778% ahead in relative wealth. It still loses materially
in 2024, so base and 10 bps promotion both fail.

## Pre-2024 and negative-year behavior

Across all 24 calendar years from 2000–2023 the rule won 16, with +4.6839
percentage points mean annual excess, +2.8706 points median, and -3.3111 points
worst. It beat buy-and-hold in all six negative AAPL years:

| Year | Strategy | AAPL buy-and-hold | Excess |
|---|---:|---:|---:|
| 2000 | -68.1777% | -72.0044% | +3.8267 pp |
| 2002 | -35.1842% | -36.5397% | +1.3555 pp |
| 2008 | -34.7249% | -56.8791% | +22.1542 pp |
| 2015 | +4.3371% | -2.3441% | +6.6812 pp |
| 2018 | +12.5737% | -5.4833% | +18.0570 pp |
| 2022 | -26.3005% | -27.4078% | +1.1073 pp |

## Integrity

- Runtime: 6.3631 seconds.
- Holdout reveal index: 7, assigned by the locked registry.
- Approved price snapshot and complete session sequence: passed.
- 172 repository tests passed before the candidate was frozen.
- Model/API calls and external cost: zero / `$0.00`.
- Maximum exposure: 1.0; minimum cash and shares: 0.0.
- Margin interest and integrity errors: zero / none.
- Data SHA-256: `0c460bde5bbca9b237f8ce14d276d86d709264fff88bb4fc0c9da48ba7fc3de1`.
- Candidate SHA-256: `49650400347112a40ffe46dd309818b719c86140c9456ee15407964b89359aa3`.

This is useful evidence that a sparse learned condition can protect negative
years and add value in 2025/2026, but it does not meet the original goal.
