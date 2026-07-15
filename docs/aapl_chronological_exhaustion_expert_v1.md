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

Each expert keeps only four sufficient statistics over its matured canonical
episodes: count, wins, sum of net edges, and sum of squared net edges. The
fixed prior is:

- Beta(1, 1) for cash-win probability;
- zero edge with prior strength 8; and
- 4% one-session edge scale with prior strength 8.

For `n` matured lessons, the posterior calculations are frozen as:

- `p = (wins + 1) / (n + 2)`;
- `p_lower = p - 0.842 * sqrt(alpha*beta / ((alpha+beta)^2*(alpha+beta+1)))`;
- `edge_mean = sum_edge / (n + 8)`;
- `edge_second = (sum_squared_edge + 8 * 0.04^2) / (n + 8)`;
- `edge_se = sqrt(edge_second / (n + 8))`; and
- `edge_lower = edge_mean - 0.842 * edge_se`.

An expert is trusted at a new signal only when all fixed gates pass:

- at least 12 matured canonical episodes;
- posterior cash-win probability at least 0.55;
- 80% one-sided probability lower bound above 0.50; and
- 80% one-sided expected net-edge lower bound above zero.

The combined policy moves to cash if at least one currently signaling expert
is trusted. Otherwise it stays 100% in AAPL. If it accepts a cash signal at
`t`, it ignores all signals at `t+1`, guaranteeing a one-session episode with
no stacking or extension.

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
union at 10 bps; otherwise learning added no value.

## Frozen 2019-2023 confirmation gates

The primary frozen replay passes only if:

- total active log edge is positive at 5 and 10 bps;
- at least 3 of 5 calendar years are positive;
- at least 3 cash episodes occur;
- mean and median 10-bps episode edge are positive;
- 2022 active edge is non-negative; and
- no episode supplies more than half of all positive episode edge.

The causal-online replay is diagnostic and cannot rescue a failed frozen
confirmation. A failure rejects the branch before the 2024+ command can open
its data.

## Final interpretation

The original strict goal still requires positive excess in 2024, 2025, and
2026 YTD at both base and stress costs. Separately, the relaxed long-run goal
requires positive continuous relative wealth and more positive than negative
calendar years. Because the underlying expert family has already seen these
years, any final result is explicitly a repeated historical audit.
