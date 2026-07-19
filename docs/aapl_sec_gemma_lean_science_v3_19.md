# AAPL SEC/Gemma lean science v3.19 preregistration

## Status and zero-effect boundary

This document is the complete immutable specification for one development-only
AAPL-or-cash successor attempt. Creating and pushing this document performs no
qualification, preflight reservation, source replay, SEC request, Yahoo
request, Ollama request, Gemma generation, market-value read, prediction,
action, return or performance calculation, confirmation or live-style data
open, paid API call, broker action, or real-money action.

V3.19 is a preprocessing-boundary successor to the implementation-free V3.18
rejection. V3.18 proved that its exact final privacy grammar is internally
complete, but the mandatory redacted all-request proof found seven of 75
requests containing sentences selected by that exact grammar. The final gate
was correct to reject them; the mistake was allowing those sentences into the
canonical request and then expecting the final gate to accept the request.

V3.19 does not weaken the V3.18 security-direction regex and does not add
another word exception. It adds one deterministic, fail-closed sanitizer
between the inherited preprocessed event and canonical model-request
construction. The sanitizer removes every sentence selected by the V3.19
security-direction predicate: the exact V3.18 regex with only the market-share
exemption narrowed to the exact span-local rule below. It renumbers retained
current and prior sentences, rebuilds the canonical payload, and binds the
source and derived hashes. No sentence not selected by that V3.19 predicate is
changed or removed.

Before this preregistration was created, a read-only in-memory study
authenticated the preserved V3.8 source and applied this exact sanitizer to all
75 development requests. It reported only aggregate values: 75 requests,
seven requests with removals, eight removed sentences, zero empty current
partitions, zero empty prior partitions, zero V3.19-style contract failures,
zero inherited-production-validator failures, and zero external effects. It
published no row identity, request ordinal, accession, CIK, filename, URL,
date, form, filing hash, sentence, matched lexical value, contact, or
per-request length.

## Immutable base and predecessor authority

The V3.19 branch is
`codex/aapl-sec-gemma-lean-science-v3-19`. Its immutable successor base is the
pushed V3.18 pre-implementation rejection commit
`503826b5ff0f14b0f1c1a11432bdbd7f41712fc2`, tree
`ffa89ed5b2fcd41e4930fa9854ebf25b99308d6b`, whose sole parent is the
document-only V3.18 preregistration commit
`f0080b056634866955e7ee73845d509f2bfcf0f2`, tree
`999181ff11f60596dbeeb17466e9c660472252b9`.

The V3.18 preregistration document is Git blob
`507746c8a9a6c07093fe0aee4a19ab1a5e0e74f5`, 24,821 UTF-8 bytes, with literal
SHA-256
`1873ab7a0b42c0b347344d4c7c58d252f5f29ab4570018b6cfa138274fe03de9`.
The V3.18 rejection document is Git blob
`ed59c67a9cdcab165cc1440716dd15fbd43e617b`, 3,740 UTF-8 bytes, with literal
SHA-256
`21bff88a7da5faa0e9907c2908884a46dd28998db77365e1013270ae8a1e6e95`.
At the successor base, `e/APPROACH_COMPARISON.md` is Git blob
`6aac9a11428447266a8d64c4c68ca6b626cd3725`, 37,132 UTF-8 bytes, with literal
SHA-256
`769dec11a0c12f1a454cd0a99a71c63f3addc12e49b222b6201e5a4eb90c21cb`.

V3.18 has no implementation commit, official preflight, private authority,
model response, market evidence, or strategy result. Its working implementation
must never be represented as committed evidence. V3.19 must separately retain
the immutable V3.17 document-only P/X authority and V3.16 P/I/F/X scientific
donor authority exactly as pinned in the V3.18 specification and rejection
history.

The V3.19 document-only preregistration commit is named `P319`. It must have
the V3.18 rejection commit as its sole parent, add only this document, and be
pushed and live-remote authenticated before any V3.19 implementation path is
created. Once pushed, `P319` is immutable.

## Exact inheritance and replacements

V3.19 incorporates every normative V3.18 preregistration provision after the
exact eight-form mechanical mapping `V3.18 -> V3.19`, `v3.18 -> v3.19`,
`V3_18 -> V3_19`, `v3_18 -> v3_19`, `V3-18 -> V3-19`,
`v3-18 -> v3-19`, `V318 -> V319`, and `v318 -> v319`, plus the topology
mapping `P318/I318/F318/R318/S318/C318/X318 ->
P319/I319/F319/R319/S319/C319/X319`, except for the explicit replacements in
this document.

The mapping applies to branch, module, class, function, schema, namespace,
attempt, artifact, pending suffix, phase, command, safe-code, comparison-row,
and Git-topology identities. It never rewrites quoted V3.16, V3.17, or V3.18
commits, trees, blobs, hashes, byte counts, failure facts, preserved V3.8/V3.9
authority, or the two exact case-distinct Phase-2 node IDs.

Only these provisions are replaced:

1. successor base, branch, ancestry, preregistration, implementation,
   namespace, attempt, artifact, schema, and topology identities;
2. the six production and six test path names;
3. the Phase-1 selector paths, minimum node count, and required-node list;
4. canonical request construction, to insert the exact sanitizer below;
5. preprocessing provenance, request commitments, and private aggregates, to
   bind the exact source-to-sanitized derivation below; and
6. the sentence-global V3.18 `market share(s)` exemption, replacing it with the
   exact span-local punctuation barrier below; and
7. revision-owned repository, dependency, loaded-code, runtime-authority, and
   public/private hashes derived afresh from pushed V3.19 bytes.

There are no other replacements. The V3.18 exact security-direction regex,
`are` linker, punctuation behavior, zero-to-two modifier bound, undirected
accounting correction, 20,000-byte joined-sentence cap, 131,072-byte canonical
request cap, 121,077-byte tight-wrapper proof, source bytes, nullable-source
logic, prompt, fixed `gemma4:12b` model, context, output schema, Yahoo series,
features after request construction, learner, policy, thresholds, transaction
costs, chronological stages, development gates, confirmation closure, effect
budgets, controlled qualification environment, 5,482-byte bootstrap, literal
44-module preload, 45 literal import calls, runtime-v2 authority, exact JUnit
parity, long-duration grammar, duplicate-summary rejection, and Requests
identity sentinel remain unchanged.

## Exact direction sanitizer

The V3.19 contract adds a pure function named
`build_direction_sanitized_preprocessed_event`. It performs no filesystem,
Git, clock, network, SEC, Yahoo, Ollama, model, market, broker, or real-money
I/O. Its sole input is the exact mapping returned by the inherited
`preprocess_filing_event` for one current filing and its optional immediate
prior same-form filing.

Before filtering, the function must require:

1. the exact inherited preprocessed-event key set and schema;
2. a valid lowercase 64-character `preprocessed_event_sha256` equal to the
   canonical SHA-256 of the mapping with that field removed;
3. `sentences_sha256` and `model_payload_sha256` equal to their canonical
   values, and `model_payload` byte-for-byte reproducible by
   `build_extractor_model_payload(sentences)`;
4. one or more consecutive `C0001...` current sentences followed by zero or
   more consecutive `P0001...` prior sentences, with no duplicate, gap,
   interleaving, or other prefix;
5. a prior partition if and only if
   `prior_same_form_filing_sha256` is non-null; and
6. every inherited ASCII, trim, sentence-count, sentence-length, joined UTF-8
   byte, and residual invariant.

Any failure in these source checks rejects with exactly
`v319_direction_sanitizer_source_invalid`, except the two empty-partition codes
defined below.

These are structural and dependent-hash checks; the pure sanitizer cannot
prove that a fully self-consistent source mapping came from particular raw
text. Production provenance is instead closed by the runner's non-injectable
direct call from the authenticated normalized current/prior texts, as required
below. The sanitizer must not claim stronger source authentication than that.

V3.19 defines exactly one span regex:

```python
_BLINDED_TEXT_MARKET_SHARE_SPAN_RE = re.compile(
    r"(?<![a-z0-9])market[ -]+shares?(?![a-z0-9])"
)
```

It also defines one helper named `_blinded_text_has_security_direction`.
For each sentence, in original order, the sanitizer and final request validator
must call that same helper. The helper computes exactly:

```python
folded = sentence["text"].casefold()
masked = _BLINDED_TEXT_MARKET_SHARE_SPAN_RE.sub(";", folded)
remove = _BLINDED_TEXT_SECURITY_DIRECTION_RE.search(masked) is not None
return remove
```

`_BLINDED_TEXT_SECURITY_DIRECTION_RE` is byte-for-byte the exact V3.18 regex,
including all seven subjects, optional price/value metric, the exact eleven
single linkers, zero-to-two exact modifiers, all fifteen directions, the
reverse direction-subject branch, punctuation-preserving folded-text search,
and word boundaries. The final privacy validator must call the same helper
predicate; a second approximate grammar or independently maintained word list
is forbidden. The semicolon is an internal non-connector barrier and never
enters retained text. Only each exact `market share` or `market shares` span is
masked. Its presence may never exempt another security-direction match in the
same sentence. Both `market share ... shares rose` and
`shares rose ... market share`, including the reverse direction-subject branch,
must still return `True`. A sentence whose only otherwise matching phrase is
the exact space/hyphen-connected `market share` or `market shares` span must
return `False`; punctuation such as `market; shares rose` is not one exempt
span and must return `True`.

Every sentence with `remove is True` is omitted. Every other sentence is
retained byte-for-byte and in its original relative order. Retained current
sentences are renumbered consecutively from `C0001`; retained prior sentences
are renumbered consecutively from `P0001`. No text may be edited, replaced,
masked, summarized, invented, copied between partitions, or backfilled from an
unselected filing sentence.

If no current sentence remains, the function rejects with the fixed redacted
code `v319_direction_sanitizer_empty_current`. If the source has a prior
partition and no prior sentence remains, it rejects with
`v319_direction_sanitizer_empty_prior`. These rejections occur before request
construction or any external effect. The inherited empty-sentence placeholders
are ordinary source sentences only when produced by the inherited preprocessor;
the sanitizer may never invent a new placeholder.

Within each sanitizer build invocation, the retained sentences are passed
exactly once to `build_extractor_model_payload`; validation may independently
replay the pure builder. The returned private sanitized event has exactly these
fields before its self-hash:

```text
schema_version
sanitizer_schema_version
source_preprocessed_event_sha256
source_sentence_count
removed_current_sentence_count
removed_prior_sentence_count
retained_current_sentence_count
retained_prior_sentence_count
sentences
sentences_sha256
model_payload
model_payload_sha256
```

`schema_version` is exactly
`aapl-sec-gemma-lean-science-v3-19-direction-sanitized-event-v1` and
`sanitizer_schema_version` is exactly
`aapl-sec-gemma-lean-science-v3-19-security-direction-filter-v1`.
`source_preprocessed_event_sha256` equals the exact validated
`source["preprocessed_event_sha256"]`; it is never caller-selected or derived
from any other value.
`sentences_sha256` and `model_payload_sha256` are canonical SHA-256 values.
All five count fields must have exact type `int`, never `bool`, and be
nonnegative. `source_sentence_count` equals the source sentence length and the
sum of all four removed/retained partition counts. For each partition,
`removed + retained` equals its source length; each removed count equals the
number of source sentences for which the shared helper returned `True`; and
each retained count equals the corresponding sanitized partition length. The
final field
`preprocessed_event_sha256` is the canonical SHA-256 of the complete mapping
above before that field is added.

The contract also adds
`validate_direction_sanitized_preprocessed_event(source, candidate, *,
expected_source_preprocessed_event_sha256)`. It requires the expected source
hash to be one exact lowercase 64-character SHA-256, equal to the source
self-hash, recomputes the expected candidate exclusively by calling
`build_direction_sanitized_preprocessed_event(source)`, and requires exact
canonical equality with `candidate`. It then independently rechecks the
candidate key set, types, five count equations, sentence order and IDs,
sentence hash, model-payload reproduction/hash, and final self-hash. It returns
a detached validated candidate or one fixed redacted contract code; it never
repairs a candidate.
Every mismatch in this validator rejects with exactly
`v319_direction_sanitized_event_invalid`.

The V3.19 `validate_blinded_model_request` requires a keyword-only
`sanitized_event` argument. Before its inherited checks, it independently
validates the candidate's exact shape and self-hash and requires all of:

```text
request.preprocessed_event_sha256 == sanitized_event.preprocessed_event_sha256
request.supplied_sentence_ids == ordered IDs from sanitized_event.sentences
request.request_bytes == canonical_json_bytes(sanitized_event.model_payload)
request.request_sha256 == SHA-256(request.request_bytes)
```

Any mismatch in this binding rejects with exactly
`v319_contract_blinded_request_sanitized_event` before the inherited privacy
and shape checks.

The inherited production validator then receives the unchanged strict request
shape. Both validators remain hard gates. The sanitizer is not permission to
skip, monkeypatch, weaken, catch-and-continue, or reinterpret either validator.

The V3.19 model-slice builder takes four equal-length ordered sequences:
requests, universe proofs, source preprocessed events, and validated sanitized
events. It binds all four sequences, calls the sanitized-event validator and
request validator for every row, and stores both event sequences only in the
private model slice. Its private self-hashed index contains only the ordered
source-event hashes, sanitized-event hashes, partition-removal counts, and the
existing request/proof commitments. Model-slice validation rebuilds this exact
four-sequence derivation. No later evaluator path may validate a request
without its bound sanitized event.

The source event, sanitized event, and model slice remain private. Public
artifacts may contain only fixed labels, aggregate integer counts, Boolean
gates, and self-hashes; they may not expose a request ordinal, row identity,
accession, CIK, filename, URL, date, form, filing hash, sentence, removed
lexical value, contact, or per-request length.

The runner must call the inherited `preprocess_filing_event` directly on the
already authenticated normalized current/prior texts. Before sanitation, it
must call inherited `validate_preprocessed_event` on that returned mapping with
the same exact normalized texts and canonical identity lexicon. The returned
trusted source hash becomes
`expected_source_preprocessed_event_sha256` for the V3.19 sanitized-event
validator. Only its returned validated candidate may construct a request. No
dependency injection, caller-supplied preprocessed mapping, cached alternative,
or fallback construction is allowed on the production path. The runner
collects one private sanitizer row per
event containing only the source-event hash, sanitized-event hash, and the two
removed-partition counts. It reduces those 75 rows to one exact aggregate with:

```text
sanitizer_schema_version
event_count
events_with_removals
removed_current_sentence_count
removed_prior_sentence_count
empty_current_count
empty_prior_count
source_events_sha256
sanitized_events_sha256
removal_counts_sha256
```

The three hashes are canonical hashes of the ordered source hashes, ordered
sanitized hashes, and ordered two-count rows respectively. `event_count` is 75;
the empty counts are zero; every other count is a nonnegative integer with
exact row/total parity. The complete aggregate and its three ordered hashes
remain private and are bound into the model-plan manifest, request commitments,
and private preflight manifest. Public `F319` contains only the aggregate
integer counts, inherited private-manifest commitment, Boolean gates, and its
own self-hash. Neither per-event rows nor ordered source, sanitized, request,
or removal-count hashes enter a public artifact.

## Exact implementation boundary

`I319` has `P319` as its sole parent and adds exactly these twelve paths:

```text
agent_benchmark/sec_gemma_lean_science_v319_contract.py
agent_benchmark/sec_gemma_lean_science_v319_bridge.py
agent_benchmark/sec_gemma_lean_science_v319_journal.py
agent_benchmark/sec_gemma_lean_science_v319_store.py
agent_benchmark/sec_gemma_lean_science_v319_preflight.py
agent_benchmark/sec_gemma_lean_science_v319_runner.py
tests/test_sec_gemma_lean_science_v319_contract.py
tests/test_sec_gemma_lean_science_v319_bridge.py
tests/test_sec_gemma_lean_science_v319_journal.py
tests/test_sec_gemma_lean_science_v319_store.py
tests/test_sec_gemma_lean_science_v319_preflight.py
tests/test_sec_gemma_lean_science_v319_runner.py
```

No tracked predecessor path may change in `I319`. Because no committed V3.18
implementation exists, the implementation must use immutable I316 only as the
mechanical source donor, apply every frozen V3.18 correction, then add only the
V3.19 replacements in this document. Uncommitted V3.18 working bytes are not
Git authority and may not be cited as such.

Before official preflight, branch, upstream, cached and live remote head,
commit, tree, sole parent, exact twelve additions, Git blobs, working hashes
and byte counts, predecessor equality, and clean worktree must authenticate.
No V3.19 production or Phase-1 test module may import or execute a V3.18,
V3.17, V3.16, V3.15, V3.14, V3.13, V3.12, V3.11, or V3.10 production/test
module or historical strategy entry point. The six V3.19 production modules
own successor behavior and may import only one another, the exact inherited
37-module shared closure, the package initializer, standard-library modules,
and the exact qualified third-party runtime.

## Required direct tests and qualification

All mapped V3.18 tests remain mandatory. Phase 1 runs every collected node in
the six V3.19 test files. It must collect at least 333 case-sensitive unique
nodes, zero duplicates, and every node must pass. It includes the mapped 27
V3.18 required nodes plus exactly:

28. `tests/test_sec_gemma_lean_science_v319_contract.py::test_direction_sanitizer_filters_complete_security_direction_grammar_and_preserves_exclusions`
29. `tests/test_sec_gemma_lean_science_v319_contract.py::test_direction_sanitizer_renumbers_current_and_prior_deterministically`
30. `tests/test_sec_gemma_lean_science_v319_contract.py::test_direction_sanitizer_fails_closed_when_required_partition_would_be_empty`
31. `tests/test_sec_gemma_lean_science_v319_contract.py::test_direction_sanitizer_rebuilds_payload_and_binds_source_and_output_hashes`
32. `tests/test_sec_gemma_lean_science_v319_runner.py::test_runner_builds_only_sanitized_requests_through_both_validators`

The grammar test must include the complete V3.18 forbidden matrix, safe
punctuation/nonlocal/accounting controls, and `market share(s)` exclusions.
It must also include space/hyphen-only span controls and dangerous
co-occurrences on either side of the span, including punctuation-separated
`market; shares rose`, without adding a parametrized node.
The renumbering test must remove first, middle, and last sentences from both
partitions and prove byte-preserving stable order. The empty-partition test
must prove both redacted failure codes and zero request/external effects. The
hash test must independently rebuild the payload and every source/output hash,
then reject stale outer hashes and outer-rehashed mappings whose dependent
schema, payload, count, order, or nested-hash commitments are inconsistent.
It must not claim that the pure helper alone can distinguish an otherwise
fully self-consistent forged source mapping from the real preprocessor output;
that provenance belongs to the runner integration test.
The runner integration test must use real V3.19 functions without monkeypatching
either validator. The hash or runner test must also install fail-fast traps or
perform an equivalent source/call audit proving that the successful pure
sanitizer path cannot touch filesystem, Git, clock, network, SEC, Yahoo,
Ollama, model, market, broker, or real-money surfaces.

Phase 2 remains the exact 14 ordered shared selectors: 655 ordered nodes, 655
case-sensitive unique IDs, zero duplicates, and ordered node-list SHA-256
`1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c`.
Phase 3 remains the one exact Requests-identity sentinel. The controlled
Windows environment, timeouts, JUnit authentication, exact collection/execution
parity, zero skip/xfail/xpass policy, and once-only reservation remain mapped
V3.18 behavior.

Before `I319` is pushed and again immediately before official preflight is
reserved, the redacted all-request proof must authenticate the preserved V3.8
source, reconstruct all 75 inherited preprocessed events, derive all 75
sanitized events, and pass every canonical request through the real V3.19
contract validator and inherited production validator. It must report request
count 75, exactly seven requests with removals, exactly eight total removed
sentences across the current/prior aggregate counts, zero empty required
partitions, zero contract failures, zero inherited-production failures, and
zero external effects. Only aggregate fixed labels, integer counts, Boolean
outcomes, and public bounds may be printed. Any other removal total is a hard
failure, including a zero-failure result produced by over-filtering. Failure
blocks `I319` push and official preflight.

## Chronology, gates, effects, and terminal topology

Development remains 2000-2018, confirmation remains 2019-2023, and live-style
evaluation remains 2024 onward. Confirmation and live-style data stay closed
unless all 22 frozen development gates pass and the separately pushed-result
gate authenticates the exact result. Buy-and-hold comparison remains evaluated
at both 5 bps and 10 bps. No shorting, leverage, borrowing, negative cash,
paid API, broker action, or real-money action is authorized.

Official preflight is once-only and zero-effect. It may authenticate local
source and construct sanitized requests after all qualification phases pass,
but may not make a model, Yahoo, market, paid, broker, or real-money call. A
failure is preserved and consumed. Development may begin only from a separately
pushed and authenticated passing `F319` authority.

`F319` adds only the public preflight artifact. `R319` adds only the terminal
result and changes the comparison table. `S319` adds only the pause artifact.
`C319` adds only the continuation document. After a consumed official failure,
`X319` changes only the comparison table because the failure artifact already
exists. A rejection before `I319` or before official reservation instead adds
exactly one V3.19 rejection document and changes only the comparison table.
Every `X319` creates no development result and authorizes no execution. Every
terminal row preserves all earlier rows and appends exactly one V3.19 outcome
after V3.18.

If development fails any frozen gate, V3.19 is permanently rejected and no
later stage opens. If development passes, the pushed-result gate must
authenticate exact private/public evidence before confirmation can be
considered. All failures and successful results are committed and pushed as
immutable evidence. Nothing in this preregistration authorizes real-money
trading.
