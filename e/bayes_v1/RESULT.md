# Hierarchical empirical-Bayes predictor v1 — rejected

Run: `aapl-unleveraged-hierarchical_empirical_bayes_irrm_h1_v1-20260710T145708Z-46f8b26a`

This is the first genuine chronological predictor in the no-leverage series. It
learns a probability that cash will beat AAPL over the next adjusted-open
interval, adds each lesson only when that exit open exists, and continues
learning through 2024–2026. It makes no LLM calls.

## Required final periods at 5 bps

| Period | Strategy | AAPL buy-and-hold | Excess | Terminal-close excess |
|---|---:|---:|---:|---:|
| 2024 | 34.2975% | 35.4795% | -1.1820 pp | -1.1726 pp |
| 2025 | 18.8570% | 10.1345% | +8.7224 pp | +8.6841 pp |
| 2026 YTD | 14.3557% | 14.2039% | +0.1518 pp | +0.1546 pp |

At 10 bps it loses 1.3156 points in 2024 and 0.0766 points in 2026 YTD. Both
promotion gates therefore fail, despite a +7.1204% continuous relative-wealth
advantage across the full final span.

## Forecast quality

| Period | Eligible forecasts | Cash signals | Cash-signal win rate | Mean realized cash edge |
|---|---:|---:|---:|---:|
| Pre-2024 | 5,584 | 56 | 44.64% | -0.1103% |
| 2024 | 252 | 1 | 0.00% | -0.8793% |
| 2025 | 250 | 3 | 100.00% | +2.4964% |
| 2026 YTD | 127 | 2 | 50.00% | +0.0661% |

The learner's selected 2005–2023 continuous replay had +11.6761% relative
wealth, but the broader 2000–2023 annual audit has only seven winning years,
zero median excess, and -0.2916 points mean annual excess. The apparent edge is
concentrated rather than reliable “most of the time.”

## Negative buy-and-hold years

It beat buy-and-hold in only three of six negative AAPL years: 2015 (+3.5175
points), 2018 (+14.8664), and 2022 (+0.2015). It tied 2000 and 2008 and was 2.6746
points worse in 2002.

## Integrity

- Runtime: 9.2804 seconds.
- Holdout reveal index: 10; the exact retry retained the same candidate hash.
- 174 repository tests passed after completed audit processes released DuckDB.
- Approved price snapshot and complete session sequence: passed.
- Future labels used: false; final-period online learning: true.
- Maximum exposure: 1.0; no shorts, borrowing, negative cash, or margin interest.
- Model/API calls and external cost: zero / `$0.00`.
- Candidate SHA-256: `2796b55294e748f4fcb562c1faa6bcd377c2f267defb4645bd47cf71ec75c32a`.

This approach is preserved because it is a real predictor, but its calibration
and generalization are inadequate for the requested system.
