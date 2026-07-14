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

On the first execution attempt, before any candidate score or performance
artifact was produced, the deterministic Huber solver reached its inherited
50-iteration ceiling on one development fit. An outcome-blind convergence
diagnostic showed that the frozen fits converged in at most 51 iterations.
The only numerical correction therefore raises the Huber iteration ceiling to
75; all features, targets, gates, regularization values, tolerances, periods,
and ranking rules remain unchanged. This correction is committed before the
first scored development run.

## Non-negotiable trading contract

- Decisions are timestamped only after the completed Cboe VIX daily value for
  session `t`, approximately 4:15 p.m. ET. AAPL and ETF closes for `t` are
  already complete by then. The strategy cannot trade at those closes and
  fills only at the adjusted AAPL open `t+1`.
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

The complete cross-sectional panel begins on `2000-05-26`, IWM's first listed
session in the frozen source. Context before that date is forbidden. This
still leaves more than 252 sessions of warm-up before the first 2005
development prediction.

Features are deterministic completed-close transformations only:

- sector log returns are `log(close_t / close_t-w)` for fixed windows
  `w in {1, 5, 20, 60}`; every cross-section requires all nine ETFs;
- participation is the fraction of the nine returns strictly above zero,
  the median is the ordinary cross-sectional median, and dispersion is the
  population standard deviation (`ddof=0`);
- the defensive group is exactly `XLP, XLU, XLV`; the cyclical group is exactly
  `XLB, XLE, XLF, XLI, XLK, XLY`; both defensive-minus-cyclical participation
  and defensive-minus-cyclical median log return are included at all four
  fixed windows;
- IWM-minus-SPY log-return momentum is included at all four fixed windows;
- AAPL-minus-XLK log-return momentum is included at all four fixed windows;
  AAPL-minus-QQQ at 5 and 20 sessions is already present in the common frozen
  price controls, so the incremental sector vector adds only its nonduplicated
  1- and 60-session versions;
- VIX uses its completed 4:15 p.m. ET daily value: log level, 1-, 5-, 20-, and
  60-session log changes, plus an inclusive trailing 252-session z-score of
  log(VIX) using sample standard deviation (`ddof=1`); and
- the existing frozen AAPL/market price feature set as common controls.

Missing context is never forward-filled, interpolated, or replaced with zero.
Rows without a complete feature vector cannot generate a full-model forecast.
Before feature construction, the feature builder rejects context dates outside
the explicitly authorized price-session index. The bounded Parquet loader may
validate values while forming that panel; the experiment runner then requires
its session index to equal the canonical price-session index exactly.

Each run's input manifest must bind the resolved context artifact identity,
its SHA-256 checksum, its UTC acquisition timestamp, the inclusive query
bounds, the exact bounded query text, and the bounded-result checksum. A
future/live extension must append new dated vintages; it may not silently
refresh historical adjusted closes in place. A corrected historical source is
a new explicit input version and cannot replace the bytes used by an earlier
run.

## Frozen learner and candidates

The learner is the repository's deterministic regularized two-head GAM. One
head estimates the probability that cash wins after 10-bps-per-leg costs; the
other estimates the continuous cash active log edge. Every fold fits:

1. the full sector-breadth model; and
2. the existing price-only feature set as a core ablation.

Both fits use the identical training rows on which the complete breadth vector
is ready; the price-only ablation differs only by omitting the 36 incremental
features. Paired ablation triggers are likewise permitted only on identical
breadth-ready out-of-fold rows.

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
change the candidate. The same frozen policy must pass each of these gates at
both 5 and 10 bps:

- positive total active log edge;
- positive active log edge in at least two of the three fixed validation
  blocks `2019-2020`, `2021-2022`, and `2023`;
- positive active log edge in at least three of the five calendar years;
- positive active log edge in every calendar year in which the same-ledger
  AAPL buy-and-hold return is negative;
- Brier score strictly below the causal training-mean probability baseline
  and expected-edge MAE strictly below the causal training-mean edge baseline;
- strictly positive mean realized 10-bps edge and a win rate strictly above
  50% at accepted cash-episode starts;
- at least four accepted cash episodes and no more than 20% cash decision
  rows; and
- all cost binding, chronology, exact five-row block, binary exposure, and
  no-leverage proofs true.

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
