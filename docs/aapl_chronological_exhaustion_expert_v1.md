# AAPL chronological exhaustion expert v1

This document freezes a single close-out experiment before its combined
learner has been scored. It is a long/cash-only causal test, not a claim that
the strategy works.

## Evidence limitation

No globally pristine historical AAPL period remains for this rule family.
The contextual-exhaustion rule was selected with 2000-2023 data and the
weak-trend rule was selected from 675 pre-2024 configurations after the first
rule's 2024 failure was known. Both have already been audited through 2026.

This experiment can answer a narrower useful question: if those two published
rules had been treated as fixed experts, could a learner using only their
already-matured past results have decided when to trust them? A positive result
would be chronological supporting evidence, not a new unbiased proof. Only a
locked prospective paper/live period can provide that proof.

## Fixed experts

Both experts observe a completed session close `t` and can move the account to
cash only from adjusted AAPL open `t+1` to adjusted AAPL open `t+2`.

### Contextual exhaustion

- AAPL's unadjusted intraday return at `t` is above the 90th percentile of the
  prior 126 completed intraday returns.
- SPY and QQQ adjusted-close 10-session returns at `t` are both negative.

### Weak-trend exhaustion

- AAPL's intraday return at `t` is above the 92.5th percentile of the prior
  126 completed intraday returns.
- SPY and QQQ adjusted-close 20-session returns at `t` are both negative.
- AAPL adjusted close at `t` is below its completed 20-session simple moving
  average.

There is no parameter grid, new feature search, LLM prompt, or learned signal
definition. These are the exact two previously published rules. Consecutive
raw signals are canonicalized into non-overlapping one-session virtual
episodes: after accepting a signal at `t`, that expert cannot accept another
at `t+1`.

The arithmetic is exact and fixed. Intraday return is unadjusted
`close[t] / open[t] - 1`. Rolling quantiles use pandas' linear interpolation
over exactly the prior 126 sessions (`shift(1)`). An N-session market return is
`adjusted_close[t] / adjusted_close[t-N] - 1`. The AAPL SMA contains `t` and
the prior 19 completed adjusted closes. AAPL, SPY, and QQQ must share the exact
session; inputs are never forward-filled. Any missing, nonfinite, or nonpositive
source price aborts the stage as an integrity failure. A derived lookback that
is unavailable only because the required history has not accumulated makes
that expert unavailable and therefore LONG.

## Causal lesson

For an accepted expert signal at close index `t`, define the raw cash advantage
for the next open-to-open interval as:

`log(adjusted_open[t+1] / adjusted_open[t+2])`.

The primary learner subtracts exact 10-bps-per-side friction:

`net_edge = raw_edge + log((1 - 0.001) / (1 + 0.001))`.

That result may first enter memory at close `t+2`, after the re-entry open has
occurred. It can never alter the prediction that created it. Virtual expert
lessons are observable even when the combined account did not follow that
expert, because both relevant historical opens are then known.

Matured lessons are added before signals are scored at the same completed
close. Each expert canonicalizes virtual signals independently. A combined-
policy cooldown suppresses only the account's next trade; it never suppresses
another expert's virtual lesson. Simultaneous expert signals create two
expert-specific lessons but at most one combined cash episode.

Each expert keeps only four sufficient statistics over its matured canonical
episodes: count, wins, sum of net edges, and sum of squared net edges. The
fixed prior is:

- Beta(1, 1) for cash-win probability;
- zero edge with prior strength 8; and
- 4% one-session edge scale with prior strength 8.

All trust statistics use the 10-bps net label, and `win` means
`net_edge_10bps > 0`. The resulting single action stream is replayed unchanged
under both 5-bps and 10-bps ledger costs. For `n` matured lessons, define
`alpha = wins + 1` and `beta = n - wins + 1`; the calculations are frozen as:

- `p = (wins + 1) / (n + 2)`;
- `p_lower = p - 0.842 * sqrt(alpha*beta / ((alpha+beta)^2*(alpha+beta+1)))`;
- `edge_mean = sum_edge / (n + 8)`;
- `edge_second = (sum_squared_edge + 8 * 0.04^2) / (n + 8)`;
- `edge_se = sqrt(edge_second / (n + 8))`; and
- `edge_lower_score = edge_mean - 0.842 * edge_se`.

`edge_lower_score` is deliberately a conservative heuristic based on a
shrunk uncentered second moment; it is not described as a posterior credible
bound. The 4% one-session prior scale is intentional, not a units conversion.

An expert is trusted at a new signal only when all fixed gates pass:

- at least 12 matured canonical episodes;
- posterior cash-win probability at least 0.55;
- 80% one-sided probability lower bound above 0.50; and
- conservative expected net-edge lower score above zero.

The combined policy moves to cash if at least one currently signaling expert
is trusted. Otherwise it stays 100% in AAPL. If it accepts a cash signal at
`t`, it ignores all signals at `t+1`, guaranteeing a one-session episode with
no stacking or extension.

Virtual lesson memory begins on 2000-01-01; 1999 prices are feature warm-up
only. A signal without both `t+1` and `t+2` inside the currently authorized
stage cannot become a scored trade in that stage and is forced LONG for stage
accounting. It may remain pending for a later causal-online stage, but it can
never enter a frozen state whose cutoff precedes its maturity.

At a newly authorized stage boundary, account-level cooldown is recomputed
after all earlier decision rows are forced LONG. Therefore a signal that was
not executable in the prior stage cannot suppress the first executable signal
of the new stage. The continuous expert-specific virtual lesson streams are
not reset or rewritten.

## Chronological stages

| Stage | Outcomes available to learner | Evaluation |
|---|---|---|
| Development | Each result only after it matures; never later than 2018-12-31 | Annual prequential replay, 2005-2018 |
| Frozen confirmation | State learned through 2018 only; no 2019-2023 update | 2019-2023 |
| Online confirmation diagnostic | Each 2019-2023 result after it matures | 2019-2023, reported separately |
| Frozen final audit | Refit only with results matured by 2023-12-31 | 2024, 2025, 2026 YTD |
| Online final diagnostic | Later result only after it matures | Same final periods, reported separately |

The runner must not load 2019+ data unless development passes. It must not load
2024+ data unless frozen 2019-2023 confirmation passes and its exact evidence
has been committed on this branch. Later stages use separate commands and
self-hashed manifests.

Each command accepts only a physically truncated, stage-specific CSV committed
on this branch: through 2018, through 2023, or through 2026-07-09. The runner
loads that complete file and rejects the command unless its final date, row
count, full session-date sequence, and canonical price hash match the
preregistered stage. These three price hashes are frozen in the runner before
development is scored. The development process therefore has no 2019+ bytes
in its input. A later stage
proves continuity by re-hashing the already-authorized historical prefix of
its longer committed snapshot. The full future-filled source is never passed
to an experiment command.

The snapshots live under
`e/chronological_exhaustion_expert_v1/authorized_inputs/` and are derived from
the already-approved 2026-07-10 Yahoo AAPL/SPY/QQQ snapshot whose canonical
hash is `0c460bde5bbca9b237f8ce14d276d86d709264fff88bb4fc0c9da48ba7fc3de1`.
Only the through-2018 file is created before development; a longer file is
created and committed only after its parent gate has passed.

The expert, runner, ledger, no-leverage wrapper, and this contract are tracked
dependencies. Their hashes must match the parent stage before a later snapshot
can be loaded. Exact Python, NumPy, pandas, and DuckDB versions are bound too.
The regenerated sufficient statistics, learning mode, cutoff session, and
pending expert-signal context at the parent cutoff must also equal the
committed checkpoint; changing code or runtime after a result cannot open the
next period.

Development is seeded by all virtual lessons beginning 2000-01-01 that mature
before each 2005-2018 decision. Frozen and online confirmation clone the exact
same through-2018 state. Frozen confirmation never changes it; online
confirmation adds later virtual lessons only after maturity. If confirmation
passes, the frozen final state incorporates every virtual lesson matured by
2023-12-31 and then stops. Online final starts from that same through-2023
state and carries continuously across 2024, 2025, and 2026 YTD. The frozen
final stream, not the online diagnostic, must satisfy the strict goal. The
exact YTD endpoint is 2026-07-09.

## Execution and comparison

- Initial capital: $1,000.
- Position: exactly 100% AAPL or 100% cash.
- Maximum requested and realized exposure: 1.0.
- Shorting, borrowing, margin interest, leverage, and negative cash: forbidden.
- Signal: completed close; fill: following adjusted open.
- Same price rows, fills, and 5/10-bps-per-changing-leg costs for strategy and
  AAPL buy-and-hold.
- Always-long, unfiltered contextual, unfiltered weak-trend, and unfiltered
  union policies are reported as ablations on the same ledger.
- Runtime ceiling for each stage: 3,600 seconds; network/API/LLM calls: zero.

The deadline is checked after all bundle files and checksums are written but
before the temporary directory is atomically promoted. An over-limit run may
not leave a completed or passing artifact directory.

The ledger allows fractional shares and pays zero interest on cash. Strategy
and benchmark make the same initial all-in AAPL purchase when their target is
LONG. A cash transition sells at the adjusted open with adverse slippage and
the re-entry buys at the next adjusted open with adverse slippage. Remaining
shares are valued, but not forcibly liquidated, at the final authorized
adjusted open. An unresolved final signal is forced LONG as stated above.
Always-long must match the same-ledger AAPL benchmark exactly.

## Development gates

The one fixed combined policy passes 2005-2018 only if, at both 5 and 10 bps:

- total active log edge and relative wealth versus AAPL are positive;
- at least 8 combined cash episodes occur;
- at least 8 of 14 calendar years have positive active edge;
- at least 4 of 7 fixed two-year folds have positive active edge;
- total active edge remains positive after removing the best calendar year;
- no calendar year supplies more than half of all positive annual edge; and
- aggregate active edge is positive across the negative AAPL years 2008, 2015,
  and 2018, with at least two of those three years positive.

At 10 bps, combined cash episodes must also have at least 55% wins and positive
mean and median realized edge. The learner must outperform the unfiltered
union by a strictly greater total 10-bps active log edge over identical
evaluation rows; otherwise learning added no value. The unfiltered contextual,
weak-trend, and union ablations trade from their first eligible signal. The
union uses its own one-session cooldown.

The seven aggregation folds are exactly 2005-2006, 2007-2008, 2009-2010,
2011-2012, 2013-2014, 2015-2016, and 2017-2018. They never reset learner state.
Calendar performance is attributed by fill-session year; a cross-year episode
row is attributed to its entry-open year. Zero active-edge years are neither
wins nor losses. Best-year removal and annual concentration are calculated
separately on each cost ledger.

Both strategy and benchmark are forced LONG before the first evaluation fill,
so each makes the same initial AAPL purchase and no strategy can begin in cash
without paying the sell leg. Total active log edge must reconcile to the exact
sum of executed cash-episode edges. Calendar and fold gate edges use the entry-
open attribution above; ledger-boundary returns are retained separately for
audit.

## Frozen 2019-2023 confirmation gates

The primary frozen replay passes only if, at both 5 and 10 bps:

- total active log edge is positive;
- at least 3 of 5 calendar years are positive;
- 2022 active edge is non-negative.

Additionally, using the fixed 10-bps episode labels, at least three cash
episodes must occur, mean and median episode edge must be positive, and no
episode may supply more than half of all positive episode edge.

The causal-online replay is diagnostic and cannot rescue a failed frozen
confirmation. A failure rejects the branch before the 2024+ command can open
its data.

## Final interpretation

The original strict goal requires material excess in 2024, 2025, and 2026 YTD
at both base and stress costs. Before any result is exposed, `material` is
fixed as active log edge strictly above `0.001` (10 basis points) in every
requested period; a merely positive floating-point difference cannot pass.
Separately, the relaxed long-run goal requires positive continuous relative
wealth and more positive than negative calendar years. Because the underlying
expert family has already seen these years, any final result is explicitly a
repeated historical audit.
