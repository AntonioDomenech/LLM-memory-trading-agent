# AAPL SEC/Gemma lean science v3.21 preregistration

## Status and plain-language purpose

This document is the complete immutable specification for one fresh
development-only AAPL-or-cash successor attempt. Creating and pushing this
document performs no qualification, preflight reservation, source replay, SEC
request, Yahoo request, Ollama request, Gemma generation, market-value read,
prediction, action, return or performance calculation, confirmation or
live-style data open, paid API call, broker action, or real-money action.

V3.21 exists for one narrow reason. V3.20's trading science was not tested and
did not fail. Its locally measured preflight work could not fit inside its
inherited 1,500-second process deadline, so V3.20 was rejected before its
implementation was committed and before its once-only official reservation was
consumed. V3.21 changes only that process deadline. In simple terms, the same
inspection needs a correctly sized stopwatch.

The V3.21 whole-preflight deadline is exactly 4,800 seconds. It is one shared
monotonic clock created after the once-only reservation and used through every
qualification phase, runtime-authority step, source/projection/request step,
privacy check, and durable finalization step. It cannot be restarted, retried,
split into per-phase allowances, lengthened dynamically, or extended after a
slow operation.

V3.21 carries forward unchanged the corrected V3.20 working behavior:

1. deterministic standard-library `_strptime` preload before both runtime
   dependency captures;
2. direct binding of publication recovery to its dedicated recovery authority,
   with no intermediate wrapper or broad execution-authentication fallback;
   and
3. exact read-only authentication and zero-effect publication recovery for the
   replay-proven candidate-less `rejected/runtime_binding_mismatch` state.

It changes no source, filing cohort, sanitizer, request, model, prompt, market
series, feature, learner, policy, threshold, transaction cost, chronological
stage, scientific gate, or effect budget.

## Immutable base and predecessor evidence

The V3.21 branch is `codex/aapl-sec-gemma-lean-science-v3-21`. Its exact base is
the pushed V3.20 pre-implementation rejection commit
`ecb5c809370e15b27f8eb67cfe644438d47c890f`, tree
`f8171d8e116f7db8d6b717d95a263c9e9360a1fc`, whose sole parent is the pushed
P320 preregistration commit
`0eb50606e9cac3d4069a70680a6f05dac11ba973`, tree
`393a3b5698e6a8263585e16771998d75b84978db`.

At that base:

- `docs/aapl_sec_gemma_lean_science_v3_20.md` is Git blob
  `7814ecdfd20afa927bd5167f59aa5b42cdf58035`, 19,333 bytes, with literal
  SHA-256
  `bbba46970db5a41e4ec2f3c30fcd8bbad0703f0171ff634ced0f54958ee45ef9`;
- `docs/aapl_sec_gemma_lean_science_v3_20_rejection.md` is Git blob
  `56c920e4ae847814daa06a0dbd0c7439f89feffb`, 8,753 bytes, with literal
  SHA-256
  `4a11aa49eb1c7c4394db2d04f6d4d6e44cf37d0897a58d0a199059337d535434`;
  and
- `e/APPROACH_COMPARISON.md` is Git blob
  `eb5ecbd82451991cab45c0cd41365da4d427e8ea`, 39,168 bytes, with literal
  SHA-256
  `fd64eb442b02c3aa3b50d474166b6f21949253a4879ea37470836a2ae81b5cc0`;
  and
- the timing-source F319 artifact at commit
  `dcdb4a78beb1bb954701ed012c829d5040918166`, path
  `e/aapl_sec_gemma_lean_science_v3_19/DEVELOPMENT_PREFLIGHT.json`, is Git
  blob `2aa77b3b8b8f9cf5d70f24ac936882d3213c4b80`, 9,014 bytes, with literal
  SHA-256
  `3f7e228458c88f3dd9f9fb9d3c7a9e793fb0b950219583ca054ac23a090e31c4`.

The V3.20 rejection is process evidence only. V3.20 has no implementation
commit, official preflight receipt, private preflight namespace, attempt
authority, strategy result, performance metric, pause, continuation, or later
stage authority. No such V3.20 object may be invented, copied, resumed,
reinterpreted, or used as V3.21 authority.

The document-only V3.21 preregistration commit is named `P321`. It must have
the exact V3.20 rejection commit above as its sole parent, add only this
document, and be pushed and live-remote authenticated before any V3.21
implementation path is created. Once pushed, P321 is immutable.

## Exact inheritance and sole semantic replacement

V3.21 incorporates every normative P320 provision after the exact eight-form
mechanical mapping `V3.20 -> V3.21`, `v3.20 -> v3.21`,
`V3_20 -> V3_21`, `v3_20 -> v3_21`, `V3-20 -> V3-21`,
`v3-20 -> v3-21`, `V320 -> V321`, and `v320 -> v321`, plus the topology
mapping `P320/I320/F320/R320/S320/C320/X320 ->
P321/I321/F321/R321/S321/C321/X321`, except for the explicit replacements in
this document.

The mapping applies to branch, module, class, function, schema, namespace,
attempt, artifact, pending suffix, phase, command, safe-code, comparison-row,
and Git-topology identities. It never rewrites the quoted immutable base
identities, historical commits, trees, blobs, hashes, byte counts, measured
timings, observed V3.19 or V3.20 facts, candidate-source rows below, or the two
exact case-distinct shared Phase-2 node IDs.

Only these provisions are replaced:

1. successor base, branch, preregistration, implementation, namespace,
   attempt, artifact, schema, and topology identities;
2. the six production and six test path names;
3. Phase-1 minimum collection and the required-node list, by adding the one
   literal deadline-contract test specified below; and
4. `QUALIFICATION_TIMEOUT_SECONDS` and every exact expected serialization of
   it, from 1,500 seconds to 4,800 seconds.

The only semantic production-code change from the corrected V3.20 candidate
is item 4. The new test in item 3 proves that change; it does not change the
runtime behavior. There are no other replacements. All P320 `_strptime`,
direct recovery, exact zero-effect recovery, runtime-binding, source,
sanitizer, request, privacy, model, market, strategy, chronology, cost, gate,
effect-accounting, publication, and crash-safety requirements remain unchanged.

## Frozen measured deadline

The exact public F319 qualification durations are:

| Work | Seconds |
|---|---:|
| Three inherited qualification phases, collection plus execution | 697.108 |
| Measured V3.20 required source/runtime node 34 | 1,153.830 |
| Conservative repeat of that runtime/source chain after qualification | 1,153.830 |
| **Measured planning baseline** | **3,004.768** |

The deadline is frozen by this exact formula:

```text
300 * ceil(1.5 * (697.108 + 2 * 1153.83) / 300)
= 300 * ceil(4507.152 / 300)
= 4800 seconds
```

The 3,004.768-second baseline is approximately 50 minutes 5 seconds. The
1.5 multiplier provides margin for the other required nodes, Git and
filesystem inspection, durable writes, and ordinary machine variation. The
result is rounded upward to one of sixteen complete five-minute blocks. Thus
the honest pre-run estimate is approximately 50 minutes, while 80 minutes is
the hard failure ceiling, not a promised or target duration.

The deadline must be formed from one successful `time.monotonic()` read after
the private reservation becomes consumed:

```text
whole_preflight_deadline = reservation_start_monotonic + 4800.0
```

That exact absolute value must be placed in the inherited active preflight
context and passed through or read unchanged by the frozen qualification
suite. The outer official child wait and termination timeout must be bounded by
the positive time remaining on that same deadline. Any inherited narrower
local subprocess timeout remains only a local upper bound and cannot authorize
continued work or a passing artifact at or after the shared deadline.
Qualification completion, runtime authority, source authentication,
projection, request commitments, privacy scans, private manifest, public
artifact, completion records, self-hashes, and final durable markers must all
finish strictly before it. Reaching or crossing the deadline fails closed
under the inherited deadline-failure topology.

No code path may add another 4,800 seconds, reset the active value between
collection and execution, reset it between phases, create separate budgets for
post-qualification work, retry a timed-out child, infer a new deadline from
elapsed work, or use wall-clock time in place of monotonic time. Offline unit
tests may supply a synthetic fixed absolute deadline, but the official
top-level preflight may allocate exactly one real deadline and may run exactly
once.

## Hash-preserved V3.20 candidate inputs

V3.20 was rejected before `I320`, so its corrected twelve candidate files are
not Git authority. They are nevertheless the exact inspected source inputs for
the V3.21 implementation. Together they total 1,885,007 bytes. Their canonical
sorted-row manifest is 2,002 bytes with literal SHA-256
`2db9ce9e2c9b4768a08bba7eb6efaf753c77d2aabbaad8970bea6cf00bc5a889`.

I321 must derive each V3.21 destination below from exactly the corresponding
source bytes by applying the eight-form version mapping above. It must then
apply only the explicitly authorized successor-authority substitutions, the
frozen timeout replacement, and the one direct deadline-test addition. The
successor-authority substitutions are `BASE_COMMIT =
ecb5c809370e15b27f8eb67cfe644438d47c890f`, `BASE_TREE =
f8171d8e116f7db8d6b717d95a263c9e9360a1fc`, `BASE_PARENT =
0eb50606e9cac3d4069a70680a6f05dac11ba973`, and the exact P321 commit, tree,
document blob, literal SHA-256, and byte count measured only after P321 is
committed and pushed. No other authority substitution is allowed.

| Hash-preserved source -> I321 destination | Source bytes | Source literal SHA-256 |
|---|---:|---|
| `agent_benchmark/sec_gemma_lean_science_v320_bridge.py` -> `agent_benchmark/sec_gemma_lean_science_v321_bridge.py` | 93,155 | `9a6ae8d5ee40f2db9efbc37c6da0e598e158fbdd14438a5d60a5ac291758a1bf` |
| `agent_benchmark/sec_gemma_lean_science_v320_contract.py` -> `agent_benchmark/sec_gemma_lean_science_v321_contract.py` | 168,944 | `d3462f44cdc578f79b027317353dc5b9a6a32a9b46980a032c7f40aead682b45` |
| `agent_benchmark/sec_gemma_lean_science_v320_journal.py` -> `agent_benchmark/sec_gemma_lean_science_v321_journal.py` | 21,869 | `05a7654073835c5354bb78b748e12aea9b16f065c3c46a4cf5ce6956fcaca1d5` |
| `agent_benchmark/sec_gemma_lean_science_v320_preflight.py` -> `agent_benchmark/sec_gemma_lean_science_v321_preflight.py` | 557,637 | `fa92ee981c7fa6c5647807e4e415ba8d8b33cb454aec0cc5b905196d527ef7ee` |
| `agent_benchmark/sec_gemma_lean_science_v320_runner.py` -> `agent_benchmark/sec_gemma_lean_science_v321_runner.py` | 227,239 | `44e80d0138d108f05b84e02443bc441126f0beba8d382b0435447684f13d30ca` |
| `agent_benchmark/sec_gemma_lean_science_v320_store.py` -> `agent_benchmark/sec_gemma_lean_science_v321_store.py` | 192,331 | `fc2e198a6cfefc8ed9b3b1587d8721df4aeb4a5c60a6bd006608ef33b3695cf8` |
| `tests/test_sec_gemma_lean_science_v320_bridge.py` -> `tests/test_sec_gemma_lean_science_v321_bridge.py` | 69,745 | `a15ce9f990761ec5eefff868d420f55a0b2c3d6c71fbf0d2cf3ffbb20df16dc3` |
| `tests/test_sec_gemma_lean_science_v320_contract.py` -> `tests/test_sec_gemma_lean_science_v321_contract.py` | 58,380 | `1ce8e7c5fc458e70bf75d14dd997373ed77d276240e4f28061891a6de304d95d` |
| `tests/test_sec_gemma_lean_science_v320_journal.py` -> `tests/test_sec_gemma_lean_science_v321_journal.py` | 8,463 | `72ab0787b11e695463bb355132146ef86c44176ed8334551b5c6d9592037180f` |
| `tests/test_sec_gemma_lean_science_v320_preflight.py` -> `tests/test_sec_gemma_lean_science_v321_preflight.py` | 225,684 | `95b0ec6ebbbe61606da3c14d2fa341e6881a6fd03b5f4ddcc24e26deea74c813` |
| `tests/test_sec_gemma_lean_science_v320_runner.py` -> `tests/test_sec_gemma_lean_science_v321_runner.py` | 198,964 | `1f0601644b13724a1df269c2f26ce8f41d90eecd0a25a37de2d2848666fe4178` |
| `tests/test_sec_gemma_lean_science_v320_store.py` -> `tests/test_sec_gemma_lean_science_v321_store.py` | 62,596 | `57b6601c0389e5f2dc97ff9d6fc447c96233d77b978c5b2ca6ea7ddab7897999` |

The candidate manifest serialization is the compact UTF-8 JSON array of twelve
objects with keys `byte_count`, `literal_sha256`, and `path`; object keys are
sorted by their UTF-8 bytes, rows are sorted by the slash-normalized path's
UTF-8 bytes, separators are exactly `,` and `:`, non-ASCII is not escaped, and
there is no trailing newline. Before derivation, all twelve source paths, byte
counts, individual hashes, total bytes, canonical row count/order, manifest
bytes, and manifest hash must match exactly. A mismatch blocks I321. After the
V3.21 files are created, the untracked V3.20 candidates must be removed before
the implementation is committed. Their preservation is the table and
aggregate above plus their exact mapped V3.21 descendants; no V3.20 path may
enter P321, I321, or any later V3.21 commit.

## Implementation topology

`I321` has P321 as its sole parent and adds exactly these twelve paths:

```text
agent_benchmark/sec_gemma_lean_science_v321_contract.py
agent_benchmark/sec_gemma_lean_science_v321_bridge.py
agent_benchmark/sec_gemma_lean_science_v321_journal.py
agent_benchmark/sec_gemma_lean_science_v321_store.py
agent_benchmark/sec_gemma_lean_science_v321_preflight.py
agent_benchmark/sec_gemma_lean_science_v321_runner.py
tests/test_sec_gemma_lean_science_v321_contract.py
tests/test_sec_gemma_lean_science_v321_bridge.py
tests/test_sec_gemma_lean_science_v321_journal.py
tests/test_sec_gemma_lean_science_v321_store.py
tests/test_sec_gemma_lean_science_v321_preflight.py
tests/test_sec_gemma_lean_science_v321_runner.py
```

No tracked predecessor path may change in I321. The six V3.21 production
modules may not import or execute V3.20, V3.19, or older production/test
modules or historical strategy entry points. The inherited 37-module shared
closure, package initializer, standard library, and exact qualified
third-party runtime remain the only allowed imports outside the six successor
production modules.

The production recovery dependency must remain the corrected direct object
binding:

```text
authenticate_publication_recovery = load_publication_recovery_authority
```

It must not be replaced by a nested callable, lambda, partial, adapter,
execution-authentication call, or exception fallback. The `_strptime` preload
and the exact candidate-less zero-effect recovery predicate remain byte-for-
byte mechanically mapped behavior except for version identities. Their direct
tests remain required.

I321 must derive and freeze its own exact twelve-path byte counts, Git blob
identities, literal SHA-256 values, aggregate canonical manifest, repository
tree, protected source/callable manifests, and runtime authority from its
pushed bytes. It may not copy a V3.19 or hypothetical V3.20 runtime-manifest
hash.

Before changing the successor `BASE_*` constants, I321 must resolve the two
V3.20 convenience aliases that would otherwise silently corrupt historical
V3.19 evidence. `V319_PREFLIGHT_COMMIT` must become the literal
`dcdb4a78beb1bb954701ed012c829d5040918166`;
`V319_EXCEPTIONAL_REJECTION_COMMIT` and
`V319_EXCEPTIONAL_REJECTION_TREE` must become the literals
`1e1f323fa6b50797874d8b0fff7ad8ae992cb128` and
`3ec2fce9596506cee09945287c5bb7397ac71e05`. They may not remain aliases of
the new `BASE_PARENT`, `BASE_COMMIT`, or `BASE_TREE`. Mapped contract tests
must compare the V3.19 preservation commit, tree, and parent to those literal
historical identities rather than to any successor `BASE_*` alias. The mapped
scientific bootstrap's literal authority values, branch/path/role identities,
byte count, and SHA-256 must likewise be regenerated and tested from the final
I321 text; stale P320 or V3.20 successor authority is forbidden, while quoted
historical evidence remains unchanged.

## Required direct tests and qualification

Every mechanically mapped V3.20 candidate test remains mandatory. Phase 1
runs every collected node in the six V3.21 test files. It must collect at
least 342 case-sensitive unique nodes with zero duplicates, execute the exact
ordered collected list once, and pass every node with collection/execution
parity and zero skip, xfail, or xpass.

The exact 40 P320 required literal node IDs remain required after their
mechanical V3.21 mapping. One new literal, unparametrized required node is
added, for exactly 41 required IDs:

41. `tests/test_sec_gemma_lean_science_v321_contract.py::test_v321_single_deadline_is_measured_4800_seconds_and_never_resets`

Required node 41 must prove all of the following using real V3.21 production
objects plus controlled monotonic-clock and child-runner doubles:

- `QUALIFICATION_TIMEOUT_SECONDS` is exactly `80 * 60 == 4800`;
- the frozen decimal inputs produce exactly the documented 4,800-second
  rounded ceiling;
- official preflight allocates one absolute monotonic deadline only after its
  once-only reservation is consumed;
- collection, execution, all three phases, runtime-authority construction,
  source/projection/request construction, privacy checks, and durable
  finalization observe that same unchanged absolute value;
- the outer official child wait/termination bound can only shrink with the
  remaining duration, while any inherited narrower local subprocess bound
  cannot authorize work or a passing artifact at or after the shared deadline;
- no retry, reset, dynamic extension, per-phase clock, wall-clock substitute,
  or second official allocation is reachable; and
- an operation at or beyond the absolute deadline rejects and cannot publish a
  passing artifact.

Phase 2 remains the exact 14 ordered shared selectors: exactly 655 ordered
nodes, 655 case-sensitive unique IDs, zero duplicates, and ordered node-list
SHA-256
`1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c`.
Phase 3 remains the one exact Requests-identity sentinel. The controlled
environment, exact collection/execution parity, zero skip/xfail/xpass policy,
subprocess completion records, private/public validation, and once-only
reservation remain otherwise unchanged.

Local tests before I321 and before reservation are diagnostics only. They may
not create the official private namespace, public artifact, launch proof,
attempt identity, market/model response, or performance evidence, and they may
not be described as an official pass.

## Mandatory pre-push and pre-reservation evidence

Before I321 is pushed and again immediately before official preflight is
reserved, the inherited redacted real-source proof must authenticate the
preserved V3.8 source and report exactly:

- 75 canonical requests;
- seven requests with direction removals;
- eight removed sentences, partitioned as four current-period and four
  prior-period sentences;
- zero empty required partitions;
- zero V3.21 contract failures;
- zero inherited-production failures; and
- zero external effects.

The runtime proof must preload `_strptime` and use the real repository,
dependency, and loaded-code builders. It must report exactly 44 repository
module rows, 366 dependency module-file rows, and 3,242 protected callable
rows, with equality before and after the actual preserved-source
authentication, streaming projection, request commitments, and model-plan
construction. It must expose only redacted aggregates and exact hashes derived
from the exact local I321 commit bytes before push and the exact authenticated
live-remote I321 bytes before reservation, never private paths or source
content. Any count, identity, hash, or before/after mismatch blocks I321 or the
official reservation.

Immediately before reservation, all effect counters must still be zero; no
Yahoo/Ollama/Gemma/SEC transport or market/model open may have occurred; the
V3.21 private namespace and public artifact must be absent; HEAD must be the
exact pushed I321 commit and tree; local and remote must agree; and tracked,
staged, and untracked state must be exactly clean. The official command and
environment must match the inherited scientific bootstrap exactly.

The operator must print, before consumption, a short human-readable line with
the measured planning baseline of 3,004.768 seconds (about 50 minutes 5
seconds), the 4,800-second hard ceiling (80 minutes), the fact that this is the
single official attempt, and the current zero-effect/clean-state result. That
line is informational only and cannot change the frozen clock. During the run,
progress may report completed phase, monotonic elapsed time, and remaining
hard-ceiling time from the one deadline; it must not expose private source or
claim that a stage passed before its durable receipt exists.

If the machine cannot safely provide the full 4,800-second window, any
mandatory precheck fails, or the estimate is no longer honest, the reservation
must not be consumed. The issue must instead be preserved under the inherited
pre-implementation rejection topology.

## One official attempt and authority isolation

V3.21 authorizes exactly one official preflight reservation. A consumed
reservation can never be retried, even after interruption, restart, code
change, branch change, timeout, or apparent environmental repair. A failed
official preflight is preserved through mapped X321 topology; it never becomes
a second attempt.

The official V3.21 preflight creates a fresh V3.21 namespace, identity,
one-shot reservation, runtime authority, and public artifact. It must not read,
copy, manufacture, or accept a V3.20 private directory, launch proof, recovery
authority, attempt identity, journal, candidate, terminal receipt, result,
pause, continuation, or public artifact. V3.20 candidate hashes authenticate
only the pre-implementation source derivation described above. They do not
authenticate any execution state.

Publication recovery remains restricted to a V3.21 store created by the exact
pushed F321 authority and the exact mapped P320 predicates. It cannot create a
new development attempt, turn a V3.20 rejection into V3.21 state, or introduce
an external effect.

## Unchanged science and chronological gates

The preserved source and filing cohort, direction sanitizer and shared final
predicate, seven-request/eight-sentence aggregate, 75 canonical requests,
fixed local `gemma4:12b` model, prompt, context, output schema, Yahoo series,
features, learner, AAPL-or-cash policy, thresholds, transaction costs,
development gates, confirmation closure, live-style closure, runtime-v2
identities, Requests sentinel, and terminal construction are unchanged from
P320 after version mapping.

Development remains 2000-2018, confirmation remains 2019-2023, and live-style
evaluation remains 2024 onward. Confirmation and live-style data remain
closed unless all 22 inherited development gates pass and the separately
pushed result gate authenticates the exact result. Buy-and-hold remains
evaluated at both 5 bps and 10 bps. No shorting, leverage, borrowing, negative
cash, paid API, broker action, or real-money action is authorized.

Changing the clock does not make the strategy more likely to win, change a
return, relax a gate, select a better period, or reveal later data. It only
allows the already frozen zero-effect inspection enough measured time to
finish. A passing preflight is permission to begin the development stage, not
evidence that the system beats buy-and-hold.

## Terminal Git topology and honest outcomes

P321 adds only this preregistration. I321 adds only the twelve implementation
and test paths listed above. Once-only F321 adds only its validated public
preflight artifact. R321 adds only an authenticated terminal result and the
comparison-table change. S321 adds only a pause artifact. C321 adds only a
continuation document. A consumed preflight failure uses mapped X321 topology.
Every failure, pause, and successful result must be committed and pushed as
immutable evidence before a successor can rely on it.

If official preflight misses the 4,800-second deadline or any frozen check,
V3.21 is permanently rejected and no development or later stage opens. If
preflight passes, development may run once under the inherited rules. If any
development gate fails, V3.21 is rejected and confirmation remains closed. If
development passes, the pushed-result gate must authenticate exact
private/public evidence before confirmation can be considered. A later
success claim requires the frozen buy-and-hold comparisons and every
chronological gate; absence of a result is never a win.

Nothing in this preregistration authorizes real-money trading.
