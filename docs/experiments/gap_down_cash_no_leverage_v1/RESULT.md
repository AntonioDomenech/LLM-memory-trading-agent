# Gap-down cash v1 — rejected

Run: `aapl-unleveraged-gap_down_cash_v1-20260710T140948Z-3e82dd97`

This frozen rule holds cash for one next-open interval after an AAPL opening gap
below -4%. It uses only binary 0%/100% exposure and never borrows or shorts.

## Required final periods at 5 bps

| Period | Strategy | AAPL buy-and-hold | Excess | Terminal-close excess |
|---|---:|---:|---:|---:|
| 2024 | 34.2975% | 35.4795% | -1.1820 pp | -1.1726 pp |
| 2025 | 9.0856% | 10.1345% | -1.0489 pp | -1.0443 pp |
| 2026 YTD | 14.2039% | 14.2039% | 0.0000 pp | 0.0000 pp |

The continuous 2024–2026 account finished 1.8166% behind buy-and-hold in
relative wealth. The rule fails both the base and 10 bps promotion gates.

## Negative buy-and-hold years

It beat buy-and-hold in five of six losing AAPL years from 2000–2023, but only
produced a positive absolute return in 2015. It made the already severe 2000
loss worse.

| Year | Strategy | AAPL buy-and-hold | Excess |
|---|---:|---:|---:|
| 2000 | -76.0318% | -72.0044% | -4.0274 pp |
| 2002 | -32.0796% | -36.5397% | +4.4601 pp |
| 2008 | -46.5901% | -56.8791% | +10.2890 pp |
| 2015 | +1.8293% | -2.3441% | +4.1734 pp |
| 2018 | -1.9471% | -5.4833% | +3.5362 pp |
| 2022 | -27.1335% | -27.4078% | +0.2743 pp |

Across all 24 calendar years from 2000–2023, it won 15, with +3.0121 percentage
points of mean annual excess, +1.8787 points median, and -5.8937 points worst.

## Integrity

- Runtime: 6.2170 seconds.
- Model/API calls and external cost: zero / `$0.00`.
- Maximum target, post-fill, and holding exposure: 1.0.
- Minimum cash and shares: 0.0.
- Margin interest: 0.0.
- Integrity errors: none.
- Exact data SHA-256: `0c460bde5bbca9b237f8ce14d276d86d709264fff88bb4fc0c9da48ba7fc3de1`.
- Candidate SHA-256: `20287853ce96ba9af0ccc0b3e0cfa1bbbee039d25560a599db1fb61c5c4d4010`.

The run directory contains the market snapshot, all daily ledgers at 5/10/20
bps, continuous-account ledgers, the full report, and independent checksums.
