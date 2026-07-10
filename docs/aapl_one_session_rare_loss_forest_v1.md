# Frozen pre-2019 AAPL one-session rare-loss forest v1

This document is the immutable pre-run contract for the final price/market-only
experiment. The implementation branch is
`codex/aapl-one-session-rare-loss-forest-v1`. No market data may be opened and
no model code may be tuned on the basis of results until this contract has
been committed.

This is chronological internal development and model-selection evidence, not
an unseen test. It may use only data through 2018-12-31. No 2019 or later row,
including a row with empty values, may be loaded by the development feature,
fit, prediction, scoring, selection, or verification path.

## Decision, execution, and labels

The decision is made after the completed close on session `t`. A CASH decision
sells AAPL at the adjusted open on `t+1` and returns to 100% AAPL at the
adjusted open on `t+2`. A LONG decision remains 100% invested. There is no
shorting, leverage, borrowing, margin interest, cash interest, or partial
exposure.

The one-session AAPL simple return is:

`aapl_return_1 = adjusted_open[t+2] / adjusted_open[t+1] - 1`.

The three frozen learning targets are:

- Severe-loss classification: `aapl_return_1 <= -0.025`.
- Ordinary CASH-win classification: the 10-bps CASH active log edge is
  strictly greater than zero.
- CASH-edge regression: the 10-bps CASH active log edge, clipped to the closed
  interval `[-0.08, 0.08]`.

The 10-bps edge includes 10 basis points on each executed sell and re-entry
leg. A row becomes eligible for learning only after its `t+2` open has occurred.
At every fold boundary, the forest, preprocessing, global priors, OOB tail
thresholds, and causal prevalence baselines must use only labels whose maturity
date is strictly earlier than the first decision in that fold.

Strategy and buy-and-hold are simulated on the same adjusted-open ledger at
both 5 and 10 basis points per executed leg. The first and final development
fills must be LONG. Consecutive CASH signals merge into one continuous CASH
episode and incur no artificial intermediate re-entry.

## Frozen feature sets

The core model has exactly these 15 completed-close features, in this order:

1. `aapl_lr_1`
2. `aapl_lr_5`
3. `aapl_lr_20`
4. `aapl_intraday_lr`
5. `aapl_gap_lr`
6. `aapl_downside_rv_20`
7. `aapl_drawdown_60`
8. `aapl_drawdown_252`
9. `aapl_minus_qqq_lr_5`
10. `aapl_minus_qqq_lr_20`
11. `qqq_lr_5`
12. `qqq_lr_20`
13. `qqq_lr_60`
14. `qqq_rv_20`
15. `qqq_minus_spy_lr_20`

The full model appends exactly these five market-stress features:

16. `iwm_lr_5`
17. `iwm_lr_20`
18. `qqq_minus_iwm_lr_20`
19. `vix_z_252`
20. `vix_lr_5`

The formulas and completed-close timing of shared features must be reused
unchanged from the existing audited price/context feature implementation.
There are no date, calendar, ticker-identity, or future-return features.

Core and full forests must train, predict, and be evaluated on the identical
full-feature-ready rows. A missing or nonfinite full input forces LONG; partial
head/model averaging and imputation are forbidden. TNX, CFTC positioning,
macro placeholders, HMM states, GDELT rows, news, and all other text/event data
are excluded from this experiment.

## Deterministic forest

Core and full are separate deterministic multi-output forests. Each forest has
one shared tree structure; every terminal leaf emits a severe-loss probability,
an ordinary CASH-win probability, and an expected clipped CASH edge. The frozen
configuration is:

- 127 trees.
- Maximum depth 3.
- Fixed random seed 2741; no random restarts.
- A 20-session moving-block bootstrap for every tree.
- The bootstrap target is `ceil(0.75 * N)` in-bag row occurrences, where `N`
  is the causally eligible training-sample size. Contiguous 20-row block starts
  are sampled uniformly with replacement, blocks are concatenated, and the
  final block is truncated to the exact target. Duplicate row indices are
  allowed. A row is OOB for a tree only when its index was never sampled by
  that tree.
- Exactly five candidate features are sampled without replacement at every
  splittable node.
- Candidate split thresholds are the in-node empirical quantiles
  `0.10, 0.25, 0.50, 0.75, 0.90`, calculated with
  `numpy.quantile(..., method="lower")`. Duplicate thresholds and thresholds
  equal to the node minimum or maximum are discarded.
- Both children must contain at least 64 in-bag row occurrences.
- Splits maximize class-balanced weighted Gini gain on the severe-loss label
  only. When both classes are present, each class has total weight 0.5. A
  single-class node cannot split.
- The maximum gain wins. Gains equal within `1e-15` are tied and resolve first
  to the lower feature index and then to the lower numeric threshold.

Leaf shrinkage is 64 and applies independently to all three outputs. The
shrinkage targets are computed from the complete causally eligible model
training sample: severe-loss prevalence, ordinary CASH-win prevalence, and
mean clipped 10-bps edge. For a leaf with `n` in-bag row occurrences, each
output is:

`(leaf_sum + 64 * global_value) / (n + 64)`.

Forest predictions are the arithmetic mean of the 127 corresponding leaf
outputs. The ordinary and edge targets never influence split selection.

## OOB threshold contract and candidates

Every eligible training row must receive predictions from at least 16 trees
for which it is OOB. If any required row has fewer than 16 OOB trees, that
forest/fold fails closed; the seed, tree count, or sample contract may not be
changed. OOB predictions must be generated without using the row in the
corresponding tree.

Each variant and fold calculates its own severe-probability tail threshold
from training-only OOB severe probabilities. Tail quantiles use
`numpy.quantile(..., method="higher")`. Validation rows and their outcomes
cannot affect these thresholds.

Exactly two full-model candidates are selectable:

- `tail975`: severe probability at least the fold's 97.5th-percentile OOB
  threshold.
- `tail990`: severe probability at least the fold's 99th-percentile OOB
  threshold.

Both candidates additionally require all of the following on the decision row:

- all 20 inputs are ready and finite;
- predicted severe-loss probability is at least two times the fold's causal
  severe-loss prevalence;
- predicted ordinary CASH-win probability is at least `0.55`; and
- predicted clipped 10-bps CASH edge is at least `0.001`.

The core forest applies the same two candidate definitions using its own
training-only OOB tail thresholds, but it is an ablation only and can never be
selected.

## Chronological development

There are seven fixed, expanding, purged folds, assigned by entry-fill year:

1. 2005-2006
2. 2007-2008
3. 2009-2010
4. 2011-2012
5. 2013-2014
6. 2015-2016
7. 2017-2018

Each fold fits exactly one core and one full forest on all causally eligible
earlier rows, for 14 development fits in total. Forest parameters and OOB
thresholds remain frozen throughout the fold. There is no within-fold or
post-outcome online update. All prediction rows, model states, OOB coverage,
fold thresholds, causal priors, and label-maturity proofs must be preserved.

The candidate grid, features, costs, thresholds, tree configuration, gates,
tie-breaks, and fold boundaries cannot change after any result is observed.

## Frozen promotion gates

A selectable full-model candidate must pass every inherited economic gate at
both 5 and 10 basis points:

- total active log edge at least `0.02`;
- median annual active log edge at least `0.0005`;
- active log edge excluding the best year at least `0.005`;
- 252-session month-end rolling win rate at least `0.60`;
- 756-session month-end rolling win rate at least `0.70`;
- annual win rate at least `0.55`;
- at least five positive folds;
- at least 30 CASH decision days;
- at least 12 CASH episodes;
- CASH decision-day rate no greater than `0.20`;
- largest positive year's share of total positive edge no greater than `0.45`;
- active log edge in negative buy-and-hold years at least `0.01`; and
- win rate across negative buy-and-hold years at least `0.60`.

It must also pass all rare-loss-specific gates on mature, identical OOF
support:

- ordinary CASH-win Brier relative improvement over the causal prevalence
  baseline at least `0.02`;
- severe-event Brier relative improvement over the causal severe-prevalence
  baseline at least `0.05`;
- clipped-edge MAE relative improvement over the causal training-mean baseline
  at least `0.01`;
- severe-event precision among CASH decision rows at least `0.25` and at least
  two times the causal severe-event prevalence on the corresponding support;
- at least 20 completed CASH episodes;
- completed-episode win rate at least `0.55`; and
- completed-episode mean active log edge strictly greater than zero.

The full model must also beat its exact core ablation on identical OOF support.
Both Brier differences must exceed `1e-12`: full severe-event Brier must be
strictly lower than core and full ordinary CASH-win Brier must be strictly
lower than core. Each Brier score is computed once on the shared OOF support
and bound into both cost-scenario results. At both 5 and 10 basis points, full
total active log edge and full minimum-fold active log edge must each exceed
core by more than `1e-12`.
All six ablation checks must pass.

Only passing full-model candidates enter selection. Ranking is frozen as:

1. descending 10-bps minimum-fold active log edge;
2. descending 10-bps total active log edge;
3. descending severe-event Brier relative improvement;
4. ascending completed CASH episodes;
5. ascending CASH decision days; and
6. ascending lexical candidate ID.

If one candidate is selected, only its full forest and OOB-derived tail rule
may advance. The through-2018 final full forest, OOB threshold, and associated
priors must be fitted twice and required to serialize byte-identically. A null
selection must produce no final model state.

## Offline runner, evidence, and runtime

The public runner must require a clean committed branch and record the source
commit plus Git-blob hashes for every transitive runtime source. It may replay
only the committed checksum-verified AAPL/SPY/QQQ through-2018 price artifact
and a bounded local context query returning only IWM and VIX rows dated no
later than 2018-12-31. It performs no network, paid API, free API, or LLM call.

The sealed artifact must contain the bounded input CSVs, source and feature
provenance, all 14 development forest states, OOB counts and thresholds, exact
OOF predictions, candidate targets, every full/core strategy ledger and the
same-ledger buy-and-hold benchmarks at both costs, predictive and economic
metrics, all gates and ablations, deterministic selection, and any permitted
final state. Declared costs must be proved from reference prices, fills, share
deltas, and slippage; ledger cash, holdings, equity, returns, and completed
CASH episodes must reconcile exactly.

The verifier must rebuild features, labels, fits, OOB predictions and
thresholds, signals, merged episodes, ledgers, metrics, gates, ablations,
selection, and any final refit from the sealed bounded inputs and require exact
payload bytes. The bundle is written in a temporary flat directory, protected
by exact checksums and a byte-preservation `.gitattributes`, and promoted
atomically only after complete replay verification. Timeout, checksum,
replay, chronology, or semantic failure removes the temporary bundle.

The complete public workflow, including bounded input loading, all fits,
simulation, sealing, and full replay verification, has a hard limit of 3,600
wall-clock seconds. Expected runtime is 8-20 minutes. Exceeding the limit is a
failed experiment, not permission to reduce the forest, skip verification, or
change a gate.

## Mandatory stop and pivot rule

If neither `tail975` nor `tail990` passes every full-model gate at both costs
and all six core-ablation checks, selection is `null`, no final forest is
fitted, and this price/market-only research family stops. The failed result is
preserved on its branch; it may not be inverted, threshold-tuned, or followed
by another price/market-only architecture.

The only permitted next research branch is a point-in-time information audit
for genuinely timestamp-safe SEC filings and/or real news text using zero paid
APIs. That audit must prove document content, publication availability,
coverage, duplicates, and timestamps before any LLM, lesson memory, or trading
model is allowed to use the data.
