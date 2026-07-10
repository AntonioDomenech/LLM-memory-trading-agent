# Exhaustion-or-gap v1 — rejected

Run: `aapl-unleveraged-exhaustion_or_gap_v1-20260710T142433Z-a2938e83`

This predeclared rule combines contextual exhaustion with the -4% gap-down
cash exit. It uses only 0% or 100% AAPL exposure, with no borrowing or shorts.

## Required final periods at 5 bps

| Period | Strategy | AAPL buy-and-hold | Excess | Terminal-close excess |
|---|---:|---:|---:|---:|
| 2024 | 31.9481% | 35.4795% | -3.5314 pp | -3.5032 pp |
| 2025 | 12.5066% | 10.1345% | +2.3721 pp | +2.3617 pp |
| 2026 YTD | 16.6832% | 14.2039% | +2.4794 pp | +2.5250 pp |

The continuous account finished 1.6510% ahead in relative wealth, but the rule
fails 2024 under both terminal marks. The 10 bps replay also loses 4.1874
percentage points in 2024. Promotion is false.

## Negative buy-and-hold years

The combined rule beat buy-and-hold in all six losing AAPL years from
2000–2023. It produced positive absolute returns in 2015 and 2018.

| Year | Strategy | AAPL buy-and-hold | Excess |
|---|---:|---:|---:|
| 2000 | -69.8162% | -72.0044% | +2.1882 pp |
| 2002 | -17.7902% | -36.5397% | +18.7494 pp |
| 2008 | -27.4903% | -56.8791% | +29.3888 pp |
| 2015 | +9.9686% | -2.3441% | +12.3127 pp |
| 2018 | +2.7789% | -5.4833% | +8.2622 pp |
| 2022 | -23.8694% | -27.4078% | +3.5384 pp |

Across all 24 pre-2024 calendar years it won 20, with +8.2032 percentage
points mean annual excess, +6.0184 points median, and -9.3859 points worst.
These attractive retrospective figures do not override the required failure.

## Integrity

- Runtime: 6.4917 seconds.
- Holdout reveal index: 6; assigned automatically by the locked registry. The
  earlier index 5 was consumed by a discarded run whose registry checksum
  exposed a Windows newline mismatch before its artifacts were committed.
- Approved price snapshot and complete session-sequence checks: passed.
- Model/API calls and external cost: zero / `$0.00`.
- Maximum target, post-fill, and holding exposure: 1.0.
- Minimum cash and shares: 0.0.
- Margin interest: 0.0.
- Integrity errors: none.
- Exact data SHA-256: `0c460bde5bbca9b237f8ce14d276d86d709264fff88bb4fc0c9da48ba7fc3de1`.
- Candidate SHA-256: `3ff207600ccfe24f9ae6ab65cff08d2f1b80cb8eef3953382a20829664213334`.

The run directory contains the authenticated market snapshot, all ledgers at
5/10/20 bps, the exact holdout-registry snapshot, the report, and checksums.
