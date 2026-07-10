# AAPL downside-rebound v1 — rejected

This branch preserves a failed, pre-2024-selected challenger instead of
overwriting it with the later winner.

## Rule

At each completed AAPL session, calculate its intraday return and the 5th
percentile of the prior 756 intraday returns, excluding the current session. If
the current return is below both that percentile and -2.5%, target 1.10x AAPL at
the next adjusted open for one open-to-open session. Otherwise target 1.00x.

Costs are 5 bps per order and 8% annual interest on negative cash. The run used
the same margin ledger, AAPL buy-and-hold benchmark, terminal-close sensitivity,
and stress cases as the winning branch.

## Result

| Period | Strategy | AAPL B&H | Excess | Terminal-close excess | Max DD |
|---|---:|---:|---:|---:|---:|
| 2024 | 34.0636% | 35.4795% | **-1.4159 pp** | -1.4046 pp | -16.6076% |
| 2025 | 10.5652% | 10.1345% | +0.4307 pp | +0.4288 pp | -30.9685% |
| 2026 YTD | 14.5898% | 14.2039% | +0.3859 pp | +0.3930 pp | -12.3967% |

The approach failed because its 2024 rebound bets were followed by sufficiently
weak next-open returns that the leverage and costs lost 1.42 percentage points
relative to ordinary buy-and-hold. Both stress suites also fail the all-period
gate because 2024 remains negative.

Its earlier diagnostic evidence was promising but did not generalize perfectly:
it beat buy-and-hold in 16 of 19 calendar years from 2005–2023, with mean annual
excess +0.6002 points, median +0.3572 points, and worst year -0.7871 points.

- Run: `aapl-downside_rebound_v1-20260710T121517Z-04197338`
- Runtime: 4.26 seconds
- Model/API calls: 0
- External cost: `$0.00`
- Integrity errors: none

See the immutable [report](aapl-downside_rebound_v1-20260710T121517Z-04197338/report.json)
and exact saved market snapshot in the same run directory.
