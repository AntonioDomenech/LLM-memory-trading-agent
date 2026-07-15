# AAPL causal union duration learner v1: preflight rejection

Status: rejected before preregistration or implementation on 2026-07-15.

## Decision

Do not implement a learner that chooses one-, two-, or three-session CASH
duration after the fixed exhaustion-union entry. A bounded through-2018
training diagnostic showed that extending the already strong one-session
trade is substantially and chronologically worse. The few positive context
cells are sparse or unstable across time.

A conservative learner would keep choosing one session and exactly tie its
best fixed comparator, adding no learning value. A learner that actively
extends would bet against the dominant evidence. The required overlap,
pending-label, scheduler, checkpoint, and terminal-episode machinery is not
justified by the information available.

No learner contract, production module, development stage, attempt lock,
checkpoint, confirmation input, or 2019+ value was opened or created.

## Fixed-duration diagnostic

The diagnostic used only the already sealed v2 development price frame and
accepted union-opportunity stream ending 2018-12-31. It reused the exact
continuous adjusted-open ledger. For an accepted signal at completed close
`t`:

- duration 1 sells at open `t+1` and buys at open `t+2`;
- duration 2 sells at open `t+1` and buys at open `t+3`; and
- duration 3 sells at open `t+1` and buys at open `t+4`.

Targets from overlapping signals were combined as the union of their CASH
windows. There was no forced terminal liquidation. Duration 1 reproduced the
sealed exact-union result to floating-point representation.

| Metric | Duration 1 | Duration 2 | Duration 3 |
|---|---:|---:|---:|
| 5-bps active log edge vs AAPL | 1.092826 | 0.566302 | 0.632323 |
| 5-bps final equity from $1,000 | $115,649.44 | $68,308.85 | $72,970.84 |
| 5-bps complete episodes | 121 | 101 | 98 |
| 5-bps winning episodes | 79 | 57 | 52 |
| 5-bps positive years | 12/14 | 8/14 | 7/14 |
| 5-bps positive two-year folds | 7/7 | 5/7 | 5/7 |
| 10-bps active log edge vs AAPL | 0.971826 | 0.465302 | 0.533822 |
| 10-bps final equity from $1,000 | $102,418.14 | $61,715.78 | $66,092.80 |
| 10-bps complete episodes | 121 | 101 | 98 |
| 10-bps winning episodes | 76 | 54 | 52 |
| 10-bps positive years | 11/14 | 8/14 | 7/14 |
| 10-bps positive two-year folds | 7/7 | 4/7 | 4/7 |
| Unresolved terminal episodes | 0 | 0 | 1 |

At 10 bps, duration 2 trails duration 1 by `0.5065242808198568`
active log edge and duration 3 trails it by `0.4380040144926322`.
Both longer policies still beat AAPL buy-and-hold in this historical window,
but they discard a large part of the one-session union's edge.

## Extension-label stability

The isolated incremental reward of extending beyond duration 1 was also
negative across the full scored period and independently in both halves:

| Period | Complete opportunities | Duration 2 minus 1 | Duration 3 minus 1 |
|---|---:|---:|---:|
| 2005-2018 | 120 | -0.540006 | -0.545507 |
| 2005-2011 | 66 | -0.407010 | -0.421270 |
| 2012-2018 | 54 | -0.132996 | -0.124237 |

The 2000-2004 shadow period had positive extension sums, so using it as
warm-up would initially teach the wrong behavior for the scored era. The
largest 2005-2018 context, broad fast/slow trends both off with both
exhaustion experts firing, was strongly negative for both extensions across
46 opportunities. The only clearly positive source cell contained seven
weak-trend-only opportunities. A contextual-only three-session cell changed
from negative in 2005-2011 to positive in 2012-2018. These are not stable
grounds for a causal duration selector.

## Execution complexity found during preflight

The accepted union stream contains 166 opportunities through 2018. Its
minimum separation is two sessions: 24 adjacent accepted opportunities are
two sessions apart and another five are three sessions apart. Longer CASH
windows therefore require explicit merge or blocking semantics and cannot
truthfully claim that every accepted opportunity executes a fresh entry.

Two independent design reviews concluded that this machinery should not be
built given the negative and unstable extension evidence. Any future revisit
would require a genuinely new information source or prediction target, not a
different warm-up, threshold, context split, or duration-learner constant.

## Evidence and restrictions

- Branch: `codex/aapl-causal-union-duration-learner-v1`.
- Source bundle:
  `e/aapl_causal_contextual_expert_aggregation_v2/contextual-expert-aggregation-development-v2/`.
- Source manifest identity:
  `sha256:6cb657d1d9c323bd96a84b598b9b993e2444f756a65521aa1bf28207f7a63539`.
- Physical data end: 2018-12-31.
- Same-ledger AAPL final equity at 10 bps: $38,754.15.
- Minimum exposure: 0.0. Maximum exposure: 1.0. Minimum cash: 0.0.
- Network, news, LLM, Ollama, and API calls: 0. External cost: $0.00.
- The complete source bundle remained checksum-clean after the diagnostic.

This is a training preflight, not unseen test evidence and not authorization
for real capital.
