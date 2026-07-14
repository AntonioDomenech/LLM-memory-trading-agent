# AAPL sector-breadth residual-edge experiment v1

## Question and status

Can completed-close cross-sectional market breadth and risk appetite identify
five-session intervals when holding cash should beat holding AAPL after costs?

This document preregisters the approach before any 2024, 2025, or 2026 outcome
is loaded. Development and intermediate validation may use only information
whose labels matured by 2023-12-31. The final periods may be opened only after
the exact selected specification and its evidence are committed.

This is a zero-cost statistical predictor. It does not call an LLM. The
separate SEC/Gemma approach remains preserved on its own branch; this branch
tests whether verified market-sentiment metrics contain an economic signal
before more LLM infrastructure is justified.

## Non-negotiable trading contract

- Decisions use completed information through AAPL close `t` and fill at the
  adjusted AAPL open `t+1`.
- Exposure is exactly `1.0` AAPL or `0.0` AAPL. Shorting, leverage, borrowing,
  fractional target exposure, margin interest, and negative cash are forbidden.
- A triggered cash episode lasts five decision rows. Triggers during an active
  episode do not extend it.
- Strategy and buy-and-hold use the same adjusted-open ledger, fractional
  shares, initial capital, sessions, terminal marks, and adverse transaction
  costs.
- Both 5-bps and 10-bps costs per changing trade leg are scored. A 20-bps
  scenario is a severe diagnostic only.
- Cash earns zero unless a vintage-safe rate is separately proven; none is
  authorized in v1.
- The complete approach run has a 3,600-second wall-clock ceiling and may use
  no paid API.

## Frozen information set

The cross-sectional universe is fixed to the nine original US sector ETFs:

`XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY`.

`IWM` measures small-cap risk appetite and `^VIX` measures option-implied
market fear. AAPL, SPY, and QQQ prices come from the existing exact
long/cash-ledger snapshot. No present-day stock constituent panel is allowed,
which avoids a survivorship-biased current-member universe.

Features are deterministic completed-close transformations only:

- fraction of sectors with positive 1-, 5-, 20-, and 60-session returns;
- median and cross-sectional dispersion of sector returns;
- cyclical-minus-defensive participation;
- IWM-minus-SPY risk-appetite momentum;
- AAPL residual momentum versus XLK and QQQ;
- VIX log level, changes, and trailing z-score; and
- the existing frozen AAPL/market price feature set as common controls.

Missing context is never forward-filled, interpolated, or replaced with zero.
Rows without a complete feature vector cannot generate a full-model forecast.

## Frozen learner and candidates

The learner is the repository's deterministic regularized two-head GAM. One
head estimates the probability that cash wins after 10-bps-per-leg costs; the
other estimates the continuous cash active log edge. Every fold fits:

1. the full sector-breadth model; and
2. the existing price-only feature set as a core ablation.

The only permitted policy gates are frozen in advance:

| Candidate | Cash-win probability | Expected active edge |
|---|---:|---:|
| `p50_e0` | 0.50 | 0.0000 |
| `p55_e0` | 0.55 | 0.0000 |
| `p50_e25` | 0.50 | 0.0025 |
| `p55_e25` | 0.55 | 0.0025 |

No threshold, feature, holding period, or model hyperparameter may be changed
after intermediate or final results are visible. A materially different choice
requires a new branch and preregistration.

## Chronological selection

Development uses seven expanding, purged, frozen two-year out-of-fold blocks
covering 2005-2018. Each model may train only on labels whose five-session
open-to-open outcome matured strictly before its block began. Model and
preprocessing state remain fixed throughout each block.

A candidate must pass the existing direct-edge predictive and economic gates
at both 5 and 10 bps. The full model must also beat its price-only ablation on
identical sentiment-ready rows: lower Brier score, lower edge MAE, higher total
active log edge, and no worse weakest-fold active edge. Ranking is frozen:

1. highest 10-bps weakest-fold active log edge;
2. highest 10-bps total active log edge;
3. fewest 10-bps cash episodes; and
4. lexical candidate ID.

If nothing passes, selection is null and no later period is opened.

## Intermediate validation and final test

After development selection, the chosen policy is refit once using only labels
matured by 2018-12-31 and frozen for 2019-2023. Intermediate validation cannot
change the candidate. Promotion requires positive 10-bps active log edge
overall, positive active edge in at least three of five calendar years,
positive active edge in every negative-buy-and-hold year, and no failure of
the predictive, cost, chronology, or no-leverage proofs.

Only after passing may the same specification be refit using labels matured by
2023-12-31, hashed, committed, and evaluated once on fresh-start 2024, 2025,
and 2026-YTD accounts plus one continuous account. It must materially beat the
same-ledger benchmark in every requested segment at 5 and 10 bps, with all
integrity and no-leverage checks true. A failure is preserved, not tuned.

If the completed evidence wins more independent periods than it loses, the
same frozen chronological procedure may be extended to an approximately
ten-year assessment. This extension cannot retroactively redefine the final
test.

## Evidence

Every run must save canonical input manifests, feature and label hashes,
fold-specific training/model hashes, all out-of-fold predictions, every
strategy and benchmark ledger, predictive/economic diagnostics, named gate
results, runtime and zero-cost proof, and exact checksums. The cross-approach
comparison table is updated even when selection is null.
