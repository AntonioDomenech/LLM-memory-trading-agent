# Contextual exhaustion v1 — rejected

Run: `aapl-unleveraged-contextual_exhaustion_v1-20260710T140841Z-cce43197`

This frozen long/cash rule exits AAPL for one next-open interval after an
intraday return above the preceding 126-session 90th percentile, but only while
both SPY and QQQ have negative 10-session momentum. It never borrows, shorts,
or requests more than 100% exposure.

## Required final periods at 5 bps

| Period | Strategy | AAPL buy-and-hold | Excess | Terminal-close excess |
|---|---:|---:|---:|---:|
| 2024 | 31.9481% | 35.4795% | -3.5314 pp | -3.5032 pp |
| 2025 | 22.9627% | 10.1345% | +12.8282 pp | +12.7718 pp |
| 2026 YTD | 16.6832% | 14.2039% | +2.4794 pp | +2.5250 pp |

The continuous 2024–2026 account finished with 11.0982% more relative wealth
than buy-and-hold, but the rule fails the non-negotiable 2024 fresh-account
gate. Both base and 10 bps promotion gates are false.

## Negative buy-and-hold years

The objective diagnostic found six losing AAPL calendar years from 2000–2023.
The rule beat buy-and-hold in all six, although it produced a positive absolute
return only in 2015.

| Year | Strategy | AAPL buy-and-hold | Excess |
|---|---:|---:|---:|
| 2000 | -64.7796% | -72.0044% | +7.2248 pp |
| 2002 | -23.1887% | -36.5397% | +13.3510 pp |
| 2008 | -22.6875% | -56.8791% | +34.1916 pp |
| 2015 | +9.3110% | -2.3441% | +11.6551 pp |
| 2018 | -1.0267% | -5.4833% | +4.4566 pp |
| 2022 | -23.8694% | -27.4078% | +3.5384 pp |

Across all 24 calendar years from 2000–2023, it beat buy-and-hold in 21,
averaged +6.1212 percentage points of annual excess, had a +4.5875-point median,
and a -13.6076-point worst year. These are retrospective, heavily searched
results and do not repair the failed 2024 audit.

## Integrity

- Runtime: 6.2430 seconds.
- Model/API calls and external cost: zero / `$0.00`.
- Maximum target, post-fill, and holding exposure: 1.0.
- Minimum cash and shares: 0.0.
- Margin interest: 0.0.
- Integrity errors: none.
- Exact data SHA-256: `0c460bde5bbca9b237f8ce14d276d86d709264fff88bb4fc0c9da48ba7fc3de1`.
- Candidate SHA-256: `5426e5dda13fb8096e76ed0f364283f74535c0c8d28cb7cf3180d86672ce3b38`.

The run directory contains the exact market snapshot, all fresh and continuous
daily ledgers at 5/10/20 bps, a selection declaration, the full JSON report,
and independent checksums.
