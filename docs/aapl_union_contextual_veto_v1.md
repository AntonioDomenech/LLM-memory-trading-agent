# AAPL union contextual veto v1

This document freezes one experiment before its outcome is scored. It tests a
continually learning, long/cash-only veto on top of the already published
contextual-exhaustion and weak-trend-exhaustion union. It is not a claim that
the strategy works.

## Evidence classification

The two fixed expert rules are not globally pristine. They were selected in
earlier research that inspected later historical periods, and the immediately
preceding experiment exposed their strong 2005-2018 union result. This branch
is therefore a diagnostic successor chosen from development evidence. Its
2019-2023 stage remains an untouched chronological confirmation for this exact
veto design; 2024 onward remains a repeated historical audit for the expert
family. Only a locked paper/live period can establish prospective evidence.

“Continually learning” in this branch means that causal full-history replay
keeps admitting each newly matured lesson as later rows are appended. This is
the statistical experiment, not yet a broker-connected or daily paper-trading
command. If every historical gate passes, an append-only `as-of` paper command
will be built and verified on a separate live branch; a rejected policy will
not be operationalized.

## Fixed opportunity stream

The fixed contextual and weak-trend signals, their arithmetic, session joins,
and one-session union cooldown are exactly those in
`aapl_chronological_exhaustion_expert_v1.md`. There is no new signal search or
parameter grid.

At completed close `t`, the raw union candidate is true when either expert's
independently canonicalized virtual signal is true. Before evaluating an
account period, candidates before that period's administrative start are
removed and the union's one-session cooldown is recomputed. This produces the
stage-authorized canonical union opportunity stream.

The learner is a pure veto:

`learner_cash[t] = canonical_union_cash[t] AND NOT veto[t]`.

It can never create a cash trade where the canonical union remains long.
Vetoing a union opportunity does not resurrect a raw candidate on `t+1` that
the union cooldown suppressed. During warm-up and whenever the fixed veto
condition is false, the learner takes the union trade. A candidate whose
`t+1` and `t+2` opens are outside the physical stage snapshot is unresolved
and forced long in that bounded backtest.

## Causal shadow lesson

Every canonical union opportunity creates exactly one shadow lesson,
regardless of whether one or both experts signalled and regardless of whether
the learner vetoed it. A signal at close `t` receives the exact 10-bps-per-leg
net cash edge

`y[t] = log(adjusted_open[t+1] / adjusted_open[t+2]) + log(0.999 / 1.001)`.

The lesson first enters memory at completed close `t+2`, after the re-entry
open is known and before a candidate at that same close is scored. It cannot
change the action that generated it. Lessons whose signal close precedes
2000-01-01 are never stored. The action stream learned from these fixed
10-bps labels is replayed unchanged in the 5-bps and 10-bps ledgers.

Vetoed opportunities continue to mature as counterfactual lessons. Thus the
system keeps learning indefinitely without requiring it to execute every
trade. Frozen diagnostic mode stops admitting lessons after its inclusive
cutoff; causal-online mode admits every later lesson only at its maturity
close.

## Frozen close-time features

The model is scored only on a raw union candidate. All quantities are known at
the completed close. There is no full-stage normalization. The feature order
and transforms are fixed:

| Order | Feature | Frozen transform |
|---:|---|---|
| 1 | Intercept | `1` |
| 2 | Weak only | weak-trend virtual signal is true and contextual is false |
| 3 | Expert overlap | both virtual expert signals are true |
| 4 | Tail strength | `clip((rank126(AAPL intraday return) - 0.90) / 0.10, 0, 1)` |
| 5 | Market 10-session sentiment | `clip(mean(SPY return 10, QQQ return 10) / 0.10, -1, 1)` |
| 6 | Market 20-session sentiment | `clip(mean(SPY return 20, QQQ return 20) / 0.15, -1, 1)` |
| 7 | AAPL trend | `clip((adjusted close / completed SMA20 - 1) / 0.10, -1, 1)` |

`rank126` is the number of the prior 126 completed AAPL intraday returns less
than or equal to the current completed intraday return, divided by 126. The
current session is not in that reference window. Contextual-only is the expert
identity reference category. A union candidate with a missing or nonfinite
required feature aborts the run; it is never silently converted into a veto or
a pass.

## Frozen discounted Bayesian veto

The learner stores only causal sufficient state over matured union lessons:
raw count `n_raw`, effective discounted count `n_eff`, wins, precision matrix
`A`, vector `b`, label sum, and squared-label sum. The state includes shadow
lessons from vetoed opportunities and is checkpointed with the exact feature
order and the two unresolved trailing sessions.

There is one model and no grid. Its immutable constants are:

| Constant | Value |
|---|---:|
| Coefficient prior standard deviation | `0.02` |
| Fixed observation standard deviation | `0.04` |
| Per-lesson discount `rho` | `0.995` |
| Minimum raw lessons | `40` |
| Minimum effective lessons | `30` |
| One-sided confidence multiplier | `1.282` |
| Minimum predicted harm | `0.001` |

Let `A0 = I / 0.02^2`. Initial state is `A = A0`, `b = 0`, `n_raw = 0`,
and `n_eff = 0`. For each newly matured design/label pair `(x, y)`, before the
same close's prediction:

`A = A0 + 0.995 * (A - A0) + x x' / 0.04^2`

`b = 0.995 * b + x y / 0.04^2`

`n_raw = n_raw + 1`

`n_eff = 0.995 * n_eff + 1`.

Prediction uses

`beta = solve(A, b)`

`V = inverse(A)`

`mu = x' beta`

`se = sqrt(x' V x)`

`upper = mu + 1.282 * se`.

The learner vetoes only when `n_raw >= 40`, `n_eff >= 30`, and
`upper < -0.001`. Equality takes the union trade. The discount ensures recent
patterns retain influence indefinitely instead of the learner becoming a
permanent one-time switch. Nonfinite or non-positive-definite model state
aborts. The model, features, prior, discount, warm-up, confidence level, harm
margin, and strict inequality are one fixed candidate; there is no retry or
threshold search on this branch.

## Execution and integrity

- The only targets are 0% or 100% AAPL exposure.
- Shorting, leverage, borrowing, negative cash, interest on cash, paid APIs,
  network calls, and LLM calls are forbidden.
- Both learner and benchmark make the same initial all-in AAPL purchase.
- Cash trades execute from adjusted open `t+1` to adjusted open `t+2` with
  adverse slippage on both changing legs. There is no forced terminal sale.
- The always-long control must equal the same-ledger AAPL benchmark exactly.
- Total active log edge must reconcile to executed episode edges. Learner cash
  dates must be a subset of the exact union cash dates.
- Every stage must finish within 3,600 seconds and report zero external cost.
  The deadline is checked before atomic artifact promotion.

Each command accepts only a clean, Git-tracked, physically bounded CSV with
exactly these columns:
`date,aapl_open,aapl_close,aapl_adj_close,spy_adj_close,qqq_adj_close`.
Missing, extra, invalid, duplicate, revised, or later rows fail closed before
forecast construction. Dates, canonical price-content hashes, runtime/library
versions, source lineage, implementation dependencies, checkpoints, reports,
gates, manifests, and payload checksums are sealed. A later stage cannot load
its first new market row until the committed parent manifest and gate report
are independently verified as passing.

## Development: 2005-2018

The model learns causally from lessons starting in 2000 and is evaluated
continuously from 2005 through 2018. The exact union comparator must reproduce
the already sealed reference: 121 episodes, active log edge
`1.092826209473698` at 5 bps and `0.9718261388903109` at 10 bps.

At both 5 and 10 bps, the learner must have positive total active edge and
relative wealth, at least 8 executed cash episodes, at least 8 of 14 positive
calendar years, at least 4 of 7 positive fixed two-year folds, positive edge
after removing its best year, no year above 50% of positive annual edge, and
positive aggregate edge in the negative-AAPL years 2008, 2015, and 2018 with
at least two of those three positive. At 10 bps its executed episodes must
have at least a 55% win rate and positive mean and median edge.

Learning adds value only if all of these additional gates pass:

- learner total active edge exceeds union by more than `0.0001` at both costs;
- at least 8 canonical union episodes are vetoed;
- at 10 bps, at least 55% of vetoes are beneficial, and veto benefit has
  positive mean and median;
- no veto supplies more than 50% of all positive veto benefit;
- incremental learner-minus-union edge is positive in at least 4 of the 7
  fixed folds and remains positive after removing the best fold; and
- no fold supplies more than 50% of all positive incremental fold edge.

The folds are 2005-2006, 2007-2008, 2009-2010, 2011-2012, 2013-2014,
2015-2016, and 2017-2018. A failed development stage is preserved and cannot
authorize a later input.

## Untouched online confirmation: 2019-2023

If development passes, the primary confirmation begins from the exact
through-2018 checkpoint and keeps adding lessons chronologically. Frozen-2018
predictions are diagnostic and cannot rescue a failed causal-online replay.

At both costs the online learner must have positive total edge versus AAPL,
at least 3 of 5 positive calendar years, positive edge after removing the best
validation year, at least 3 cash episodes, positive mean and median episode
edge, and no episode above 50% of all positive episode edge. Its aggregate
edge across any mechanically identified negative-AAPL years must be positive.
It must also exceed exact-union total edge by more than `0.0001` at both costs,
have positive incremental edge in at least 2 of 5 years, and make at least 3
vetoes. At 10 bps, at least 50% of vetoes must be beneficial, veto benefit must
have positive mean and median, and no veto may supply over 50% of all positive
veto benefit.

Failure rejects the branch before any 2024+ input can be opened.

## Final repeated audit: 2024-2026 YTD

The primary final policy begins from the exact through-2023 online checkpoint
and continues learning one matured lesson at a time. At both costs it must
beat same-ledger AAPL by active log edge strictly greater than `0.001` in each
of 2024, 2025, and 2026 YTD through 2026-07-09. Across the continuous final
window it must also beat union by more than `0.0001` at both costs, with non-negative
incremental edge in at least two of the three requested periods.

A separate relaxed lifetime diagnostic reports continuous 2005-2026-YTD
relative wealth, positive versus negative years, negative-AAPL-year behavior,
episode statistics, and drawdown. It cannot override a failed strict gate.
