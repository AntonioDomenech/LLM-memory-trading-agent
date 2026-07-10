# Rejected: CFTC COT sentiment v1

> Final evidence: `cftc-cot-development-20260710T175155Z-9ad7307f`, generated
> from committed source `e83341f` with round-trip replay and the `1e-12`
> win-tolerance fix. The earlier complete run remains preserved as attempt 2.

## Decision

Reject this approach. Do not open its one-shot 2019-2023 confirmation block and
do not use any of its four variants for the 2024+ frozen audit.

The final complete development run is
`cftc-cot-development-20260710T175155Z-9ad7307f`. It used only the 1999-12-31
price warm-up, 2000-2018 AAPL/SPY/QQQ prices, and CFTC reports through
2018-12-31. It finished in 46.24 seconds with zero model calls and $0.00
estimated external cost. `development_pass=false`, `selected_variant_id=null`,
and the report contains no integrity errors.

## Economic result

All values below start from $1,000 and use the same next-adjusted-open ledger.

| Variant | Final wealth at 5 bps | Return | Active-log edge | Wealth versus buy-and-hold |
| --- | ---: | ---: | ---: | ---: |
| AAPL buy-and-hold | $47,899.86 | +4,689.99% | - | 100.00% |
| 26 weeks, z=0.75 | $16,567.19 | +1,556.72% | -1.0617 | 34.59% |
| 26 weeks, z=1.25 | $24,659.87 | +2,365.99% | -0.6639 | 51.48% |
| 52 weeks, z=0.75 | $11,793.25 | +1,079.33% | -1.4016 | 24.62% |
| 52 weeks, z=1.25 | $12,062.36 | +1,106.24% | -1.3790 | 25.18% |

At the stricter 10-bps cost, the least-bad 26-week/z=1.25 variant ended at
$23,633.79 versus $47,875.93 for buy-and-hold, retaining only 49.36% of the
benchmark wealth. Every development fold was negative for every variant, and
the best rolling 252-session and 756-session win rates were only 28.57% and
35.42%, versus required 60% and 70%.

The signal occasionally helped in selloffs: 26-week/z=1.25 reduced AAPL's 2008
loss from -56.91% to -44.75%, and 26-week/z=0.75 turned AAPL's -5.64% 2018 into
+17.61%. These isolated wins were overwhelmed by incorrectly timed cash exits,
including severe underperformance in 2015. Cash exposure ranged from 458 to
996 sessions across 27 to 50 episodes, so the failure was timing quality rather
than insufficient activity.

## Safety and evidence

- All eight 5/10-bps ledgers passed the no-leverage proof.
- Maximum requested, post-fill, and holding exposure was exactly 1.0.
- Minimum cash and shares were 0.0; borrowing, margin interest, and shorting
  were absent.
- The official raw CFTC payload and independent count query both contained
  2,838 rows. After documented anomaly exclusions, 2,824 were usable and 14
  were retained as explicit exclusions.
- The artifact directory contains the raw inputs, count proof, provenance,
  exclusions, four decision ledgers, eight strategy/benchmark ledgers,
  manifest, report, and exact checksum map.
- 2019-2023 confirmation was not accessed. No 2024+ observation or outcome was
  accessed.

## Reporting correction

Attempt 2 used strict `> 0` for annual/rolling wins. That counted two
near-zero values (`1.24e-15` and `6.62e-17`) as wins. This only flattered an
already rejected result: corrected annual win rates are 21.05% rather than
26.32% for 26-week/z=1.25 at 10 bps, and 5.26% rather than 10.53% for
52-week/z=1.25 at 5 bps. Subsequent code requires active-log edge greater than
`1e-12` and uses round-trip parsing for sealed price replay. The final evidence
includes those fixes and reproduces every policy/ledger byte. No trading result,
gate decision, or rejection changed.

The earlier pre-scoring data-contract stop is preserved separately under
`development-attempt-1-data-contract-failure/`.
