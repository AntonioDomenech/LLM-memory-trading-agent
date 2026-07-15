# AAPL causal contextual expert aggregation v1

Status: preregistered design only. This document freezes one candidate before
its implementation or any historical score is produced. It does not authorize
a trading run, a 2024+ audit, paper trading, broker execution, or real capital.

## Question and evidence classification

This experiment asks whether a small causal online aggregator can decide when
to trust the already fixed contextual-exhaustion and weak-trend-exhaustion
cash signals. It is long/cash only and keeps learning from each opportunity
after that opportunity's result becomes knowable.

The underlying expert family is not globally pristine. The contextual expert
was selected using 2000-2023 research, and the weak-trend expert was selected
after 2024 had already been inspected. Consequently:

- 2000-2004 is shadow-only learning warm-up;
- 2005-2018 is development, not test evidence;
- 2019-2023 is a one-shot, locked selector-level confirmation for this exact
  aggregation rule, but it is reused historical screening for the end-to-end
  expert family; and
- 2024 onward may be opened only on a separate branch, after confirmation
  passes and is committed and pushed, and can only be called a repeated
  historical audit.

Only predictions durably locked before future market opens in a later
append-only paper period can provide prospective evidence. No historical pass
would establish reliable autonomous profit.

## Explicit exclusions: no news, Gemma, API, or new acquisition

This candidate uses only the already tracked AAPL, SPY, and QQQ price
artifacts identified below. Existing local news, headline, article, sentiment,
embedding, filing-text, Gemma, Ollama, and other LLM artifacts are outside the
dependency set and must not be opened by the experiment.

There is no yfinance or other market-data acquisition, network request, paid
service, API call, news call, LLM call, prompt, or model inference in any
stage. No new snapshot or acquisition utility is part of this version. Market
sentiment means only the completed-close SPY/QQQ trend state defined below.
The sealed report must record zero network, news, LLM, and API calls and zero
external cost.

## Authorized physically bounded price artifacts

Development and confirmation must read the already tracked, physically
bounded authorized inputs, not an unbounded warehouse, a later stage output,
or a newly downloaded source:

| Stage | Exact tracked artifact | Rows | First/last session | SHA-256 |
|---|---|---:|---|---|
| Development | `e/chronological_exhaustion_expert_v1/authorized_inputs/aapl_spy_qqq_through_2018.csv` | 4,986 | 1999-03-10 / 2018-12-31 | `9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1` |
| Confirmation | `e/binary_regime_union_selector_v1/authorized_inputs/aapl_spy_qqq_through_2023.csv` | 6,244 | 1999-03-10 / 2023-12-29 | `c5189db9796f25ae69d14814b22ac4a852449aef8b86d3615a289b9fcb8029e9` |

Both files must have exactly these columns in this order:

`date,aapl_open,aapl_close,aapl_adj_close,spy_adj_close,qqq_adj_close`.

Every date and value must reproduce the corresponding payload hash in its
existing sealed provenance and preregistered identity. Dates must be unique,
strictly increasing exact AAPL sessions. Every numeric value must be finite
and strictly positive. There is no forward fill, backward fill, date
substitution, inner-join row loss, or fallback source. Adjusted open is derived
in memory on every row, with no persisted alternative value:

`aapl_adj_open = aapl_open * aapl_adj_close / aapl_close`.

The canonical seven-field frames after that deterministic derivation must
reproduce bounded-result SHA-256
`31b56551b8d1b837f2e69178ab7f206b6bf3c19d11d9d47be0e33cf501db3f45`
through 2018 and
`3b5e02acaa69a56fa47a0fd34275472d226b62c0f61741c13d3680b239b82535`
through 2023. The complete canonical through-2018 frame must be the exact
prefix of the through-2023 frame.

The loader must verify the tracked source bytes, the existing source
provenance, the complete applicable parent bundle, its manifest self-hash,
and payload checksum inventory before returning a price row. Confirmation
must additionally prove that its complete through-2018
canonical price prefix, derived features, accepted opportunity stream,
learner replay, pending lessons, and checkpoint equal the sealed development
versions before it may return its first 2019 value.

This branch has no authorized post-2023 input. A future audit input may use a
different sealed schema and reconstruct adjusted open, but that rule belongs
to the separate audit contract and must not be inferred or exercised here.

## Fixed opportunity stream

The fixed contextual and weak-trend expert arithmetic is reused byte for byte
from `aapl_chronological_exhaustion_expert_v1.md` and its implementation. At
completed close `t`:

- contextual exhaustion requires AAPL's unadjusted intraday return to exceed
  the 90th percentile of the prior 126 completed intraday returns and both
  SPY and QQQ 10-session adjusted-close returns to be negative; and
- weak-trend exhaustion requires the intraday return to exceed the 92.5th
  percentile of the prior 126 completed returns, both SPY and QQQ 20-session
  adjusted-close returns to be negative, and AAPL adjusted close to be below
  its completed 20-session simple moving average.

Each expert independently canonicalizes consecutive raw signals into
non-overlapping one-session virtual signals. Their raw union is true when
either canonical expert signal is true. The existing one-session union
cooldown then accepts a canonical union opportunity and suppresses the next
raw union candidate. The cooldown is applied before aggregation and is never
recomputed because the aggregator chooses LONG. A rejected or skipped
opportunity cannot resurrect a suppressed `t+1` candidate.

The shadow opportunity stream is independent of the learner's actions. Every
accepted canonical union opportunity creates exactly one aggregation lesson,
including opportunities on which the learner stays LONG.

## Fixed experts and advice

The expert order is immutable:

1. `always_long`
2. `union_cash`
3. `contextual_only`
4. `weak_trend_only`

At an accepted union opportunity, let `a[e,t]` be one for CASH advice and zero
for LONG advice:

- `a[always_long,t] = 0`;
- `a[union_cash,t] = 1`;
- `a[contextual_only,t] = 1` exactly when the canonical contextual expert
  contributed at `t`; and
- `a[weak_trend_only,t] = 1` exactly when the canonical weak-trend expert
  contributed at `t`.

The latter two advisers also vote CASH on an overlap opportunity. They vote
LONG when only the other fixed expert contributed. No expert can advise or
create an action outside the accepted union opportunity stream.

## Completed-close market state

For symbol `X` in `{SPY, QQQ}` and lookback `k` in `{10, 20}`, define only from
completed adjusted closes:

`R[X,k,t] = X_adj_close[t] / X_adj_close[t-k] - 1`.

Then:

`fast_on[t] = (R[SPY,10,t] > 0) AND (R[QQQ,10,t] > 0)`

`slow_on[t] = (R[SPY,20,t] > 0) AND (R[QQQ,20,t] > 0)`.

Equality is false. The four states are the fixed lexicographic Boolean cross
of `(fast_on, slow_on)`:

1. `(false, false)`
2. `(false, true)`
3. `(true, false)`
4. `(true, true)`

If any required lookback is missing or nonfinite, the state is `unknown` and
only the global pool may influence the decision. There is no VIX, IWM,
252-session moving average, fitted regime, full-stage normalization, or
future-dependent transform.

## Causal outcome and update order

For an accepted opportunity at close `t`, the fixed conservative label is its
10-bps-per-changing-leg cash advantage:

`y[t] = log(aapl_adj_open[t+1] / aapl_adj_open[t+2]) + log(0.999 / 1.001)`.

At signal time the system must store the exact market state and all four
expert advice values. For expert `e`, the full-information reward is

`r[e,t] = y[t]` if `a[e,t] = 1`, otherwise `r[e,t] = 0`.

Define the reward range

`b[t] = max_e(r[e,t]) - min_e(r[e,t])`.

The label and rewards first become admissible at completed close `t+2`, after
the re-entry open is known. All lessons maturing at a close are applied before
an opportunity at that same close is scored. A lesson cannot affect its own
action or an earlier action. An unresolved opportunity remains pending with
its signal-time state and advice; it is never silently dropped or relabelled.

## Fixed multiscale contextual aggregation

The half-life order is immutable:

`H = (8 opportunities, 32 opportunities, lifetime)`.

For finite half-life `h`, `lambda[h] = 2^(-1/h)`; for lifetime,
`lambda[lifetime] = 1`.

At each scale there is one global pool and one pool for each of the four known
states. A pool contains:

- a four-element discounted reward vector `G` in the fixed expert order;
- a scalar discounted reward-range square `V`; and
- a scalar discounted effective count `N`.

Every value starts at zero. When one lesson matures, every pool at that scale
is first decayed:

`G <- lambda[h] * G`

`V <- lambda[h] * V`

`N <- lambda[h] * N`.

The global pool then receives:

`G <- G + r[t]`, `V <- V + b[t]^2`, `N <- N + 1`.

If the stored signal-time state is known, that one matching state pool receives
the same additions. No state pool is updated for an `unknown` lesson. All
nonmatching state pools receive decay only.

For any pool, expert probabilities are uniform when `V = 0`. Otherwise:

`eta = sqrt(2 * log(4) / V)`

`p[e] = exp(eta * G[e]) / sum_j(exp(eta * G[j]))`.

The implementation must use a numerically stable softmax but may not alter the
formula. Nonfinite state, negative `V` or `N`, or probabilities that are
nonfinite, negative, or fail to sum to one within tolerance abort the stage.

For a known current state `s`, at scale `h` define

`rho[h,s] = N[h,s] / (N[h,s] + 4)`

and

`p_context[h] = (1 - rho[h,s]) * p_global[h] + rho[h,s] * p_state[h,s]`.

For an unknown state, `rho = 0` and `p_context = p_global`. Average the three
scales equally:

`p_bar[e] = (p_context[8,e] + p_context[32,e] + p_context[lifetime,e]) / 3`.

At a current accepted opportunity, the CASH vote mass is

`Q[t] = sum_e(p_bar[e] * a[e,t])`.

The learner chooses CASH if and only if

`Q[t] > 0.5 + 1e-12`.

Every tie, cold-start uncertainty, or lower score chooses LONG. Outside an
accepted union opportunity the target is always LONG. There is no parameter
grid, randomization, retry, threshold tuning, or alternate candidate.

## Learning modes and serialized state

`causal_online` admits every eligible lesson only when it matures.
`frozen_cutoff` still records maturity and the realized counterfactual label,
but does not update any sufficient state when the maturity close is later than
its inclusive cutoff. A maturity on the cutoff close is admitted.

The checkpoint schema must bind the contract version; exact expert, state, and
half-life order; all constants; every `G`, `V`, and `N`; the learning mode and
inclusive cutoff; every strict pending record; the last union-cooldown rows;
and the administrative account state. Restore must reject extra, missing,
reordered, nonfinite, or inconsistent content. Full-history deterministic
replay is the source of truth; checkpoint resume must reproduce every later
state, probability, action, and pending lesson exactly.

## Shadow warm-up and continuous account

Signal closes from 2000-01-01 through 2004-12-31 are shadow-only warm-up.
Their lessons are admitted only when `t+2` is knowable, but they do not create
a scored account trade. Learning continues causally during every later stage.

There is one administrative account inception on 2005-01-01. The account,
union cooldown, aggregator state, and pending lessons continue without reset
through 2018 and, if authorized, through 2023. A stage, calendar-year, fold,
or reporting boundary is not a new account. Both learner and benchmark make
the same initial all-in AAPL purchase.

Targets are exactly 0% or 100% AAPL. A CASH decision at completed close `t`
sells at adjusted open `t+1` and returns to AAPL at adjusted open `t+2`.
The same deterministic action stream is replayed at 5 and 10 bps of adverse
slippage per changing leg. Fractional shares are allowed. Shorting, leverage,
borrowing, negative cash, negative shares, margin interest, cash interest, and
a forced terminal sale are forbidden.

## Development: 2005-2018

The primary development replay uses `causal_online`, seeded only by causally
matured shadow lessons. It evaluates one continuous account over entry-fill
years 2005-2018. The fixed two-year folds are 2005-2006, 2007-2008,
2009-2010, 2011-2012, 2013-2014, 2015-2016, and 2017-2018; folds do not reset
state or the account.

At both 5 and 10 bps, development passes only if all integrity gates pass and:

- active log edge versus same-ledger AAPL is strictly greater than `0.001`;
- at least 5 of the 7 fixed folds have strictly positive active edge;
- total edge remains strictly positive after removing the single best fold;
- at least 30 complete CASH episodes execute;
- at least 55% of episodes have strictly positive edge;
- mean and median episode edge are strictly positive;
- no episode supplies more than 50% of all positive episode edge; and
- aggregate edge is strictly positive across every mechanically selected
  calendar year whose same-ledger AAPL return is strictly negative.

At 10 bps, the learner's full-account active edge must additionally exceed the
best same-stream fixed comparator by strictly more than `0.0001`. The fixed
comparators are always LONG, exact union CASH, contextual advice, and
weak-trend advice, using the same opportunity stream, cooldown, account,
fills, and costs.

Development results are training diagnostics. A failed gate seals a rejection,
ends version 1, and cannot authorize confirmation or a retry.

## Locked confirmation: 2019-2023

Confirmation is a single attempt beginning from the exact passing through-2018
checkpoint. The primary arm remains `causal_online`; it may learn from a
2019-2023 outcome only after that outcome matures. The frozen-2018 arm begins
from the identical checkpoint but admits no maturity after 2018-12-31.

At both 5 and 10 bps, the primary online arm passes only if all integrity gates
pass and:

- 2019-2023 active log edge versus same-ledger AAPL is strictly greater than
  `0.001`;
- at least 3 of the 5 calendar years have strictly positive active edge;
- aggregate edge remains strictly positive after removing the best year;
- at least 5 complete CASH episodes execute;
- mean and median episode edge are strictly positive;
- no episode supplies more than 50% of all positive episode edge;
- aggregate edge in mechanically selected negative-AAPL years is strictly
  positive;
- full-account edge exceeds every same-stream fixed comparator by strictly
  more than `0.0001`; and
- learner-minus-exact-union edge is strictly positive in at least 2 years.

Three prespecified ablation comparisons must also pass. The `global_only`
ablation forces `rho = 0` at all scales. The `lifetime_only` ablation uses only
the lifetime scale while retaining the same contextual pooling. The
`frozen_2018` ablation is the full model with post-2018 state updates disabled.

For each of online minus frozen-2018, full contextual minus global-only, and
full multiscale minus lifetime-only, the 10-bps comparison must have:

- at least 3 complete XOR action differences;
- aggregate signed incremental edge strictly greater than `0.0001`;
- strictly positive incremental edge in at least 2 entry years;
- at least 50% strictly beneficial XOR differences;
- strictly positive mean and median signed XOR edge; and
- no one difference above 50% of all positive signed difference edge.

Online minus frozen-2018 must additionally have aggregate incremental edge
strictly greater than `0.0001` at 5 bps. Difference rows must preserve the
orientation `primary_cash_comparator_long` or
`primary_long_comparator_cash`, use both changing-leg costs, and be attributed
to the year of the entry open.

Any confirmation failure is final: seal `REJECTED`, preserve the attempt, and
do not open or derive any 2024+ market value. Passing confirmation authorizes
only preparation of a separate repeated-audit branch, not a success claim.

## 2024+ is a separate repeated audit

This contract and branch must never load, query, inspect, score, or return a
post-2023 market value. Only after every confirmation gate passes and the
complete confirmation bundle is independently verified, committed, and
pushed may a new branch named
`codex/aapl-causal-contextual-expert-aggregation-audit-v1` be created.

That branch must preregister its exact endpoint, input identity, gates, runner,
parent hash, and one-attempt lock before any post-2023 value is opened. It must
start from the exact through-2023 online checkpoint and continue causal
learning without reset. Its evidence label is always repeated historical
audit, never untouched test. A failed or unexercised adaptive result cannot be
retried by changing this model under the same version.

## Integrity and reconciliation gates

Every development and confirmation bundle must prove all of the following:

- the authorized tracked price artifact, parent manifest, parent checksums,
  schema, dates, row count, raw and canonical content hashes, and deterministic
  adjusted-open derivation are exact;
- fixed expert signals and the accepted canonical union opportunity stream
  reproduce their sealed parent prefix exactly;
- there is exactly one action-independent lesson per accepted opportunity,
  with exact signal-time advice/state, `t+2` maturity, label, reward, and
  update-before-same-close ordering;
- online and frozen forecasts are identical through their cutoff, and no
  frozen post-cutoff maturity updates sufficient state;
- all pools, probabilities, mixing weights, CASH vote mass, actions,
  checkpoints, pending lessons, and cooldown rows replay exactly;
- the learner chooses CASH only on accepted union dates, all targets are
  exactly zero or one, and the account never resets after inception;
- every ledger has requested, realized, and held exposure in `[0,1]`,
  nonnegative finite cash and shares, zero margin interest, no shorting or
  borrowing, and identical action dates at both costs;
- always LONG equals the same-ledger AAPL benchmark exactly;
- full-account and reporting-period active edge reconcile to complete
  entry-attributed episode edges, and every learner/comparator incremental
  edge reconciles to its signed XOR differences within `1e-10`;
- every gate input is finite, gate report and top-level status agree, and gate
  results can be recomputed from sealed forecasts, ledgers, and episodes; and
- network, news, LLM, API, and external cost counters are all zero.

Missing, extra, duplicate, revised, nonfinite, inconsistent, or unauthorized
data fails closed. Integrity failure is not a reason to substitute a source,
relax a threshold, repair an artifact after exposure, or rerun the stage.

## Runtime and immutable sealing

Each stage has a strict wall-clock limit of 3,600 seconds measured with a
monotonic clock from runner entry. The deadline must be checked at major
phases and at the last instant before final artifact promotion. All model
work, ledgers, diagnostics, gate computation, payload construction, checksum
construction, and internal seal verification must finish before that check.

Artifacts are first written to a private same-filesystem temporary directory.
The runner must fail if the immutable final run directory, a prior attempt
lock, or a stale sealing directory already exists. It may atomically rename
the temporary directory to the final run ID only after all gates, consistency
checks, and the runtime check complete. Timeout or failure leaves no completed
run directory. A confirmation attempt lock, once created immediately before
the first 2019 value is returned, is never removed and makes a crash an
auditable consumed attempt.

The sealed bundle must include at least input and parent provenance; flattened
online, frozen, global-only, lifetime-only, and fixed-comparator forecasts;
all 5/10-bps ledgers; entry-attributed episodes and XOR differences; state and
weight diagnostics; checkpoints and pending lessons; metrics; gate reports;
runtime/cost evidence; an overall report; a self-hashed stage manifest; and an
exact checksum inventory covering every payload other than the checksum file
itself.

The manifest binds the contract version, run ID, stage, pass/fail status,
exact Git commit and upstream, implementation and test dependency hashes,
parent manifest and payload hashes, input identity, runtime/library versions,
all model constants and ordering, and every payload hash. A separate verifier
must independently recompute the manifest self-hash, payload inventory,
checkpoint continuity, report/gate consistency, causal diagnostics, ledger
reconciliations, and no-leverage proofs.

Both the stage and verifier must be entered through
`python -I -B agent_benchmark/contextual_expert_aggregation_bootstrap.py`,
using operation `stage` or `verify` and exactly one allowed stage. Before any
project package import, this bootstrap requires isolated Python with bytecode
writes disabled and rejects redirected, ignored, untracked, staged-divergent,
or working-tree-divergent Python/native import candidates. Ordinary Git status
is recorded honestly as Git-visible cleanliness; the bootstrap separately
attests the ignored import surface and local Git controls.

## Branch and commit-before-run discipline

Version 1 lives on `codex/aapl-causal-contextual-expert-aggregation-v1`.
This contract must be reviewed, committed, and pushed before model or runner
implementation begins. The implementation, tests, exact dependency list, and
frozen stage command must then be committed and pushed before development is
run. Immediately before every run, `HEAD` must equal its upstream, all tracked
dependencies and inputs must equal their HEAD/index bytes, and the worktree
and index must be clean except for a predeclared durable attempt lock created
by the runner at its authorized point.

The sealed development bundle must be independently verified, committed, and
pushed before confirmation can create its attempt lock or return a 2019 row.
The sealed confirmation bundle must likewise be verified, committed, and
pushed before a separate audit branch or contract is created. A failure is
preserved on its branch. Any formula, constant, expert, feature, threshold,
gate, input, or repair change creates a new version and branch; it is never a
retry of version 1.
