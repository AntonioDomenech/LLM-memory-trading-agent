# Rejected: AAPL sector-breadth residual edge v1

## Decision

Rejected at the frozen 2005-2018 chronological development gate. No candidate
was promoted, so the 2019-2023 intermediate validation and the 2024, 2025, and
2026-YTD final periods were not opened.

This is not a close failure. The market-breadth model made too many cash calls,
was less accurate than its historical-average baseline, and lost substantial
wealth relative to continuously holding AAPL.

## Least-bad candidate

`sector_breadth_p50_e0` had the highest total active edge among the four full
models, but still failed decisively:

| Metric | 5 bps | 10 bps |
|---|---:|---:|
| Total active log edge vs AAPL buy-and-hold | -0.8951 | -0.9881 |
| Relative wealth vs AAPL buy-and-hold | -59.15% | -62.77% |
| Calendar years beating buy-and-hold | 4 / 14 | 4 / 14 |
| Positive two-year folds | 3 / 7 | 3 / 7 |
| Weakest fold active log edge | -0.4353 | -0.4473 |
| Cash decision rows | 1,005 / 3,523 (28.53%) | same |
| Cash episodes | 93 | same |

The episode-start evidence was also negative: only 40.30% of matured cash
episodes beat holding AAPL after 10-bps-per-leg costs, and their mean realized
active log edge was `-0.005991`.

Predictive accuracy deteriorated when breadth features were added:

| Predictive metric | Sector breadth | Price-only ablation | Causal training-mean baseline |
|---|---:|---:|---:|
| Brier score (lower is better) | 0.295290 | 0.276634 | 0.241907 |
| Expected-edge MAE (lower is better) | 0.053974 | 0.044500 | 0.034238 |

On identical breadth-ready rows the full model was economically less bad than
the price-only model, but it still had negative total edge and failed both
predictive ablation checks. It beat buy-and-hold in the negative AAPL years
2008 and 2015, but lost in 2018. That limited bear-market behavior does not
justify a longer run because the full 14-year evidence is strongly negative.

## Integrity and runtime

- Long/cash only; no leverage, shorting, borrowing, or negative cash.
- Same adjusted-open ledger and adverse costs as AAPL buy-and-hold.
- Seven purged two-year folds; each fit used only labels matured before its
  prediction block.
- Post-2023 market data accessed: false.
- Runtime: 23.16 seconds, below the 3,600-second ceiling.
- LLM/API calls: 0. Estimated external cost: $0.00.
- Immutable selection manifest:
  `sha256:75093b811a62888bb1abe15b6a863adeb86ba4847c8a203e9203e551136c2e09`.

The complete predictions, ledgers, model/training hashes, provenance, named
gate failures, and checksums are preserved in
`sector-breadth-development-v1/`.
