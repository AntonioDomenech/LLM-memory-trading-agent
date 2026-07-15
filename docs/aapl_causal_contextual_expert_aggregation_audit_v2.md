# AAPL causal contextual expert aggregation: 2019-2023 audit v2

Status: preregistered one-shot post-rejection historical continuation audit.
No 2019-2023 v2 score has been opened. This contract does not authorize any
2024-or-later market value, paper trading, broker execution, or real capital.

## Question and evidence label

The exact v2 learner was rejected in development because it trailed the
simpler fixed union, even though it beat same-ledger AAPL buy-and-hold by a
large amount across 2005-2018. This branch asks the user's separate relaxed
question: if the already-frozen learner continues chronologically and keeps
learning only from outcomes after they mature, does it preserve a broad
long-run advantage over AAPL through 2023?

This is a `post_rejection_reused_historical_continuation_audit`, never a
confirmation, retry, refit, new development attempt, or pristine holdout. The
v2 development rejection remains final. The inherited expert family was
influenced by earlier research that had inspected later history, so even a
passing result is descriptive historical evidence rather than proof of a
reliable money-making system. Only a separately locked prospective paper
record can provide genuinely new forward evidence.

The report must encode `is_confirmation=false`, `is_holdout_test=false`,
`is_prospective=false`, `parent_rejection_remains_final=true`, and
`historical_results_authorize_real_capital=false`. The evidence periods are:

- 2000-2004: reused causal shadow-learning warm-up before account inception;
- 2005-2018: rejected development/training diagnostic;
- 2019-2023: one-shot post-rejection historical continuation audit; and
- 2005-2023: post-hoc continuous long-run research diagnostic.

No result may be described as untouched, out of sample, validated, confirmed,
or proof of reliable profit.

## Frozen identity

- Contract version:
  `aapl-causal-contextual-expert-aggregation-2019-2023-audit-v2`
- Branch: `codex/aapl-causal-contextual-expert-aggregation-audit-v2`
- Stage: `audit`
- Run ID:
  `contextual-expert-aggregation-post-rejection-2019-2023-audit-v2`
- Output parent: `e/aapl_causal_contextual_expert_aggregation_audit_v2`
- Durable lock:
  `e/aapl_causal_contextual_expert_aggregation_audit_v2/AUDIT_ATTEMPT_LOCK.json`
- Verifier ID:
  `contextual-expert-aggregation-post-rejection-audit-verifier-v2`
- Attempt count: exactly one
- Maximum stage runtime: 3,600 seconds
- External cost: exactly $0.00

The contract must be committed and pushed before audit implementation begins.
The completed implementation and tests must then be committed and pushed
before the single audit attempt begins. The branch, pushed commit, source
dependencies, rejected parent, input identity, lock, output directory, and
runtime are fail-closed.

## Exact rejected parent

The audit is authorized only for the immutable development bundle at
`e/aapl_causal_contextual_expert_aggregation_v2/`
`contextual-expert-aggregation-development-v2/`. Before the audit lock is
created, an audit-specific verifier must prove all of the following without
requiring a passing development result:

- contract version `aapl-causal-contextual-expert-aggregation-v2`;
- stage `development`, run ID
  `contextual-expert-aggregation-development-v2`, and status `REJECTED`;
- implementation commit
  `25609a514a9f36bbd09597a1b98ddd2fc753f1cd`;
- rejection-preservation commit
  `7bd9e6e992e47cd34cb4e4312661a767cd15fce5`;
- manifest self-hash
  `sha256:6cb657d1d9c323bd96a84b598b9b993e2444f756a65521aa1bf28207f7a63539`;
- manifest file hash
  `sha256:fbc55137d612699f794af4eedaf1e4600b4e0e916f954b962ced34f030752499`;
- checksum-inventory file hash
  `sha256:e621c845ef17d2f9e4e87efaae083e036200043fbf5ba1619800bde8df04c6a6`;
- composite checkpoint self-hash
  `sha256:e195356c0540a4a05ea635927fa466c1c85948beb7de0191e80f5bc10d60fd03`;
- checkpoint payload hash
  `sha256:abe07112ce2bb72292d65b357dc006ab3be268f8b0d02077563c59fa71a22538`;
- report payload hash
  `sha256:d08e3b650738dad248ce633c4e8077def9c89a614203dfcfc4110b787287e815`;
- gate-report payload hash
  `sha256:b28b06d28f749efe10b2f853f55386bab957dc4a68480cbaa7ce815ffa6f4e39`;
- independently regenerated semantic evidence
  `sha256:964ad44206d3a1cb481a330317217a2d910101b34e35240aa1186c8a6069e944`;
- parent causal-proof self-hash
  `sha256:dc7715dc5ad21f6f86cba0aee6488c9a24859dc19aa761115d48e2a2f199b350`;
- exactly the 48 checksummed payloads and no extras or omissions;
- `stage_pass=false`, report and gate equality, 35 of 36 checks passing,
  and exactly
  `stress_10bps.full_account_beats_best_fixed_by_gt_0_0001` failing; and
- an exact independent regeneration of every forecast, matured lesson,
  ledger, cash episode, XOR comparison, checkpoint, metric, report, manifest,
  checksum, source-lineage proof, and parent causal-prefix proof.

The ordinary v2 passing-parent confirmation verifier is forbidden because it
requires `stage_pass=true`. The audit verifier must intentionally accept only
this exact rejected parent. It must reject a passing parent, a differently
rejected parent, or any changed byte.

## Frozen input and unopened boundary

The only audit price input is the tracked, physically bounded file
`e/binary_regime_union_selector_v1/authorized_inputs/`
`aapl_spy_qqq_through_2023.csv`, with:

- raw SHA-256
  `sha256:c5189db9796f25ae69d14814b22ac4a852449aef8b86d3615a289b9fcb8029e9`;
- Git blob `9b47e6596294025bb22ec872051dcc5f6320962a`;
- six physical columns fixed by the v2 input contract;
- 6,244 unique increasing sessions from 1999-03-10 through 2023-12-29;
- date-sequence SHA-256
  `77098a2d35b6cee78ccb100514e599ee4ef6dac55738e7084b02d0e0dd0b63c1`;
- canonical bounded-result SHA-256
  `sha256:3b5e02acaa69a56fa47a0fd34275472d226b62c0f61741c13d3680b239b82535`;
- source validation manifest self-hash
  `sha256:0354355ac460042f96663d1a45cf5e9a8cf4fe873ddf5cc87afefd9c4b5d81dc`;
  and
- source-provenance SHA-256
  `sha256:2a13e0bc4a5b5a9dc7ecff2fbbc0c02c1b37e7383bc9752e608ec7affadfaf6c`.

No 2024-or-later path, byte, date, or market value is authorized. The loader
must reproduce the exact through-2018 prefix and development replay before it
returns the first 2019 value to model or evaluation code.

## One-way attempt lock

Before the lock, the bootstrap may inspect Git metadata and committed code,
contracts, tests, rejected-parent artifacts, and the through-2018 development
input. It may prove the audit input's HEAD and index blob identity without
reading its worktree contents. It may not read or hash the audit input's
worktree bytes, parse its dates, or expose any 2019 value.

Any broad pre-lock clean-tree check must explicitly exclude the through-2023
worktree path so Git does not hash or otherwise inspect those bytes. That path
receives its full status and byte check only after the lock is durable.

The pre-lock phase must prove:

1. the expected branch and origin;
2. HEAD is pushed and the complete frozen dependency inventory is clean;
3. the audit input's HEAD and index objects both equal the frozen Git blob;
4. neither output directory nor lock already exists;
5. the exact rejected parent passes the audit-specific byte and semantic
   verification above; and
6. its through-2018 model checkpoints, administrative account checkpoints,
   ledgers, causal prefix, and source lineage reproduce exactly.

The runner then atomically creates the canonical lock with exclusive-create
semantics, fsyncs its parent where supported, reads it back byte for byte, and
never removes or rewrites it. Only after the durable lock exists may the runner
read the local through-2023 input, require its raw and canonical identities,
require path-specific cleanliness, and reproduce the exact through-2018
prefix. Any failure, exception, timeout, crash, dirty input, or partial output
after lock creation consumes the sole attempt. There is no repair-and-retry.

The lock binds at least the audit contract, run ID, expected branch, pushed
commit, input HEAD/index blob, rejected-parent identities, verifier evidence,
pre-lock dependency identity, creation timestamp, and output path. Its exact
bytes and hash are copied into the sealed bundle and independently verified.

## Exact frozen model and chronological learning

The audit reuses the v2 model, feature construction, market-state encoding,
expert opportunities, advice, priors, thresholds, discounting, horizons,
pending lessons, admission timing, cooldowns, targets, account logic, and
serialization byte for byte. No formula, constant, threshold, feature,
expert, fallback, tie-break, or decision rule may change.

All four model arms begin from the identical sealed through-2018 model state
and economically identical through-2018 accounts:

- `online_full`: continues normal v2 learning. A 2019-2023 result is admitted
  only after the contract's causal maturity delay, never before the action it
  evaluates could have completed.
- `frozen_2018`: makes decisions from the exact through-2018 state and admits
  no later lesson.
- `global_only`: continues causal learning with the exact v2 global-only
  ablation.
- `lifetime_only`: continues causal learning with the exact v2 lifetime-only
  ablation.

The fixed `always_long`, `exact_union_cash`, `contextual_only`, and
`weak_trend_only` comparators and same-ledger `aapl_buy_hold` are continued as
well. Model state, pending lessons, union cooldown, target state, account cash,
shares, and cost basis never reset at 2019, a year boundary, or a reporting
boundary. The continuous account begins in 2005 exactly as in development.

This design directly separates two questions:

- Does the already-learned policy keep beating AAPL? Compare `online_full`
  with buy-and-hold and the fixed policies.
- Do newly matured 2019-2023 lessons improve decisions? Compare `online_full`
  with `frozen_2018` on their exact XOR decisions.

Learning happens from every eligible matured counterfactual outcome whether
the online account traded or skipped it. No action may learn its own unknown
future result.

## Execution and ledgers

All policies target exactly 0% or 100% AAPL. Orders decided from completed
information execute at the next adjusted open under the inherited v2 ledger.
The audit runs 5 and 10 basis points of adverse cost on every changing leg.
Cash earns zero interest. The strategy may not short, use leverage, borrow,
hold negative cash, pay or receive margin interest, or request exposure
outside `[0,1]`.

For every arm, fixed comparator, and cost, the audit must produce one
continuous 2005-2023 ledger and exact entry-attributed complete cash episodes.
It must report:

- every calendar year 2005-2023;
- the uninterrupted 2005-2023 account;
- the unopened v2 suffix 2019-2023;
- fixed blocks 2019-2020, 2021-2022, and 2023; and
- every complete online-versus-ablation XOR episode.

All full-account, period, episode, comparator, and XOR edges must reconcile to
absolute tolerance `1e-10`. Always-long must equal same-ledger AAPL
buy-and-hold. Cross-cost decisions must be identical, with costs affecting
only fills and returns. Every requested exposure, realized exposure, minimum
cash, changing leg, execution date, pending lesson, and terminal checkpoint
must pass the inherited integrity controls.

## Preregistered 2019-2023 policy criterion

For each cost separately, `post_rejection_2019_2023_pass` requires the exact
`online_full` suffix to satisfy all of the following:

- aggregate active log edge versus AAPL is strictly greater than `0.001`;
- at least three of the five calendar years have strictly positive edge;
- aggregate edge remains strictly positive after removing the best year;
- at least two of the three fixed reporting blocks have positive edge;
- at least five complete cash episodes enter during the suffix;
- strictly beneficial-episode rate is at least 50%;
- mean and median complete-episode edge are strictly positive;
- no episode supplies more than 50% of total positive episode edge; and
- if AAPL has any negative-return suffix years, aggregate edge across those
  years is strictly positive; otherwise this diagnostic is explicitly `N/A`
  and is not silently treated as observed support.

The combined suffix flag passes only if every applicable condition passes at
both 5 and 10 bps. Empty, nonfinite, nonreconciling, or insufficient evidence
fails closed. A `1e-12` tolerance may classify floating-point zero only; it
may not soften the declared `0.001` materiality threshold.

## Preregistered continuous 2005-2023 criterion

For each cost separately, `continuous_2005_2023_robustness_pass` requires the
continuous `online_full` account to satisfy all of the following:

- total active log edge versus AAPL is strictly greater than `0.001`;
- at least 11 of 19 calendar years have strictly positive edge;
- total edge remains strictly positive after removing the best year;
- aggregate edge across all negative-AAPL years is strictly positive;
- at least 125 complete cash episodes;
- strictly beneficial-episode rate is at least 55%;
- mean and median complete-episode edge are strictly positive;
- total edge remains strictly positive after removing the five largest
  complete episode edges;
- no episode supplies more than 25% of total positive episode edge; and
- strategy maximum drawdown is no worse than AAPL maximum drawdown, with
  drawdowns represented as negative values.

The continuous flag passes only at both costs. These thresholds were chosen
after the 2005-2018 development result was known, so this is a deliberately
relaxed descriptive robustness criterion, not new confirmation evidence.

## Fixed comparators and adaptive-value statuses

The report always gives `online_full` edge versus `exact_union_cash` for the
suffix and continuous account at both costs. It also reports the best fixed
comparator. These comparisons are not weakened, rounded away, or hidden, but
the fixed union is not part of the relaxed policy-versus-AAPL pass because the
user's stated target is buy-and-hold. Instead, comparator superiority controls
the separate learning claim below.

For each comparison—`online_full` minus `frozen_2018`, `online_full` minus
`global_only`, and `online_full` minus `lifetime_only`—the audit reports state
threshold transitions, complete XOR episodes, distinct entry years, and
signed incremental edge at both costs for the suffix and continuous account.
The 10-bps suffix classification is:

- `unexercised` when there is no causal state difference or no action XOR;
- `exercised_insufficient_evidence` with fewer than five complete XOR
  episodes or fewer than two distinct entry years;
- `exercised_positive` with sufficient exposure and incremental edge above
  `1e-12`;
- `exercised_negative` with sufficient exposure and incremental edge below
  `-1e-12`; or
- `exercised_flat` otherwise.

An unexercised or insufficient result must say plainly that useful continual
learning was not demonstrated. It cannot be converted into a positive result
because the online policy itself performed well.

## Decision statuses and next boundary

`historical_policy_candidate_for_2024_audit` is true only if both the
2019-2023 policy criterion and continuous 2005-2023 robustness criterion pass.
This status means only that a separate, newly preregistered 2024+ audit may be
proposed. It does not authorize opening any later data.

`learning_candidate_for_2024_audit` additionally requires, at both costs:

- `online_full` has positive 2019-2023 incremental edge over
  `frozen_2018`;
- the online-versus-frozen adaptive status is `exercised_positive`; and
- `online_full` has positive 2019-2023 incremental edge over
  `exact_union_cash`.

If the policy flag passes but the learning flag does not, the model may be
retained only as a fixed-policy historical lead with an online shadow arm.
The result must not claim that continuing to learn added value.

The sealed report preserves the original v2 rejection prominently regardless
of audit outcome. No audit result may relabel v2 development as passed.

## Integrity, sealing, and independent verification

The runner may make no network, news, LLM, Ollama, paid-API, or external-model
call. It uses no randomness and records zero API calls, zero LLM calls, false
network/news access, and zero external cost. Tests before dispatch use only
synthetic fixtures and sealed through-2018 artifacts.

The run builds in a private same-filesystem directory. Before promotion it
must independently regenerate the full semantic result, validate the exact
payload inventory and hashes, verify the audit lock and rejected parent,
verify clean pushed Git identity, and enforce elapsed time below 3,600 seconds.
It then atomically promotes the private directory to the final run path.
Failure leaves the durable lock and no reusable partial result.

The sealed bundle includes the exact attempt lock, rejected-parent and input
provenance, canonical bounded prices, every arm and fixed-policy forecast,
5/10-bps continuous ledgers, complete episodes, all XOR differences, matured
and pending lessons, diagnostics, terminal checkpoints, metrics, criteria,
integrity evidence, runtime/cost evidence, report, self-hashed manifest, and
checksum inventory.

The standalone verifier independently reloads and regenerates the result from
the exact rejected parent and audit input; it does not trust report claims or
the runner's pass flags. It verifies every model action, lesson admission,
checkpoint, account continuation, ledger, episode, XOR, metric, criterion,
manifest field, payload hash, lock byte, input identity, branch, commit,
dependency, safety invariant, and runtime/cost claim.

The measured v2 development stage took about 400 seconds and its verifier
about 394 seconds. With the longer continuation suffix, the practical estimate
for this audit is roughly 8-10 minutes for staging plus 8-10 minutes for
independent verification, or about 16-20 minutes end to end. This is an
estimate, not a relaxed limit: each phase must still finish before its strict
3,600-second deadline.

The only authorized commands are:

```text
python -I -B agent_benchmark/contextual_expert_aggregation_audit_bootstrap.py stage audit
python -I -B agent_benchmark/contextual_expert_aggregation_audit_bootstrap.py verify audit
```
