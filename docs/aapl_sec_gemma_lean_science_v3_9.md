# AAPL SEC/Gemma lean science bridge v3.9

## Status

This document preregisters one development-only bridge from the preserved v3.8
SEC source authority into the already-frozen AAPL/Gemma science. It is written
before any v3.9 implementation, private projection, Yahoo request, Ollama
runtime request, Gemma generation, market-value read, prediction, action,
performance calculation, broker interaction, or real-money action.

The branch is `codex/aapl-sec-gemma-lean-science-v3-9`. It starts at the exact
v3.8 evidence commit below. This document must be the only change in its first
commit and must be pushed before implementation begins. Once pushed, this
document is immutable. A correction requires a new version, branch, and
preregistration; it may not be silently amended.

V3.9 is not another SEC acquisition attempt. It may authenticate and read the
existing private v3.8 evidence, but it may not call a v3.8 preflight or
acquisition entrypoint and it may make no SEC request. Its only scientific
question is whether the already-acquired development filings can be projected
without loss into the frozen experiment and whether the frozen system passes
its development gates.

## Immutable source authority

The exact public base is:

| Item | Frozen value |
|---|---|
| base commit | `0c4b01cf5f1ef77548658d9bfa38e76fb1b70635` |
| base tree | `68a9176ed22edeb2edc0f00ae1179c8724603539` |
| base parent | `a0f971184ad26630478182be97f01c46c80498e1` |
| v3.8 terminal path | `e/aapl_sec_gemma_lean_evidence_v3_8/DEVELOPMENT_ACQUISITION.json` |
| v3.8 terminal Git blob | `28075db4e7cb85bfcdeb92e7b1aecce28f3a49fa` |
| v3.8 terminal literal SHA-256 | `c176c6fb5e200dd363be656562d72bc40f722912221fe7f5cd52d8e42cc9723a` |
| v3.8 terminal internal SHA-256 | `d72059f39eb4b1019ce83799682eadc7eecf3fd77310f43bee8537433a20cb20` |
| v3.8 source-authority path | `e/aapl_sec_gemma_lean_evidence_v3_8/development/source-authority-d7f87ac22e58ac4b50033af95e6f768692f1e31a6890e9f50e49f20d59ff0d0a.json` |
| v3.8 source-authority Git blob | `d2d17dc2e1159e79bbcd2ca3f63cb733289b404f` |
| v3.8 source-authority literal SHA-256 | `d7f87ac22e58ac4b50033af95e6f768692f1e31a6890e9f50e49f20d59ff0d0a` |
| v3.8 source module literal SHA-256 | `8f624df2b28425b8e31db81997aeb0387d3c19418c0781e3b04fca66830d0f04` |
| v3.8 source module Git blob | `b04263d5b3a8bdd74362fd02bc637d773d76ff9a` |
| v3.8 acquisition module literal SHA-256 | `8ba5b3dee285a5e12e631c5d5e8fbafce1a66d1c81b4c5736371ac8ceffbb3f6` |
| v3.8 acquisition module Git blob | `b5fbd94ecaff681ac698e47dd9ea0448d7886ee3` |

The authenticated private v3.8 pins are:

| Item | Frozen value |
|---|---|
| checkpoint file SHA-256 | `cd2896d627d0694c9efc202c2c8579569265fb4f61f5f74bb9d7db875fbc981e` |
| logical checkpoint SHA-256 | `704a4a9554444201ec9468e74f8c77abe82a3e3320acce29cac0c7ace11dd6fc` |
| stage source seal SHA-256 | `0eb7c6a83de59d44b3a5ceeca5918b03070ff1f508cc6a5411cd81871368f413` |
| compact replay SHA-256 | `64a7b7776206b008c0dffe26d4a14be9a092281a7834d878332d4f741db05822` |
| role-manifest inventory SHA-256 | `b6c66fba0d7482ee9b1b1892351b0d68dcb43f4ce2a2916e3ea96a316fbc2ad7` |
| role-plan SHA-256 | `6a03e4f8c2a444dd0612d7b1ff74cc605e026ace54941696a5ae180dd226110a` |
| authenticated role counts | `I=97`, `U=75`, `D=75` |
| v3.8 SEC request count | exactly `199` |
| complete experiment-family SEC request count | exactly `964` |
| authenticated v3.8 private inventory files | exactly `1,410` |
| authenticated v3.8 private inventory bytes | exactly `612,601,642` |
| authenticated v3.8 private inventory SHA-256 | `850f3a022fcaca5b56156d7843f35b05d3bc5732bf480634bd9872bfac7194f2` |

The v3.9 bridge must independently reproduce all of those pins from the pushed
public files and the read-only private v3.8 store before it creates a model
request or performs any external effect. A self-consistent mapping supplied by
a caller is not authority. Any mismatch is a terminal preregistration failure.

V3.8 remains immutable. The bridge must snapshot its authenticated inventory,
file count, byte count, and run-lock identity before and after every v3.9
private-source pass and prove that nothing changed. It may never construct
`DiskBackedSecAcquisition`, acquire a source, create a v3.8 directory, take a
v3.8 write lock, repair a v3.8 file, or adopt predecessor state.

## Immutable scientific authority

The scientific design remains the one preserved by:

| Item | Frozen value |
|---|---|
| scientific parent commit | `efbc481c57e480d48303763163676e64e87df49d` |
| scientific parent tree | `bc91d619a4d04680d9600b1acd74f0a29662d0f9` |
| current contract path | `agent_benchmark/sec_gemma_online_risk_overlay_contract.py` |
| current contract Git blob | `8a1c18728eeb7a31864397bb0c8e992b85d2320b` |
| current contract literal SHA-256 | `fccb45098f505a2f272f970762f928fbe8b75a23380292924b67d2bdd4696d2e` |
| internal contract-manifest SHA-256 | `64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d` |

V3.9 freezes the canonical JSON projection of these exact manifest keys:

1. `objective`
2. `event_availability`
3. `chronology`
4. `data`
5. `features`
6. `gemma`
7. `learner`
8. `policy`
9. `ledger`
10. `metric_definitions`
11. `gates`
12. `evidence_classification`

The projection is encoded with UTF-8, `ensure_ascii=False`, keys sorted, and
JSON separators `,` and `:` with no insignificant whitespace. It is exactly
38,320 bytes and has SHA-256
`1ee05d2916752752bbef3710d70c7dab664fb82b9ac22829dac8897401058609`.

This projection excludes old operational plumbing: version labels, branch
names, old acquisition runtime, old stage-access machinery, old artifact
paths, old publication machinery, and the obsolete one-hour execution cap.
Exclusion does not permit any scientific change. The projected values above
are the numerical and semantic authority.

## Hypotheses and interpretation

The primary frozen hypothesis is:

> Fixed local Gemma extraction from all 75 authoritative development filings
> adds after-cost chronological value over the inherited fixed exhaustion
> baseline and over the quality-preserving no-filing-meaning control.

The bridge hypothesis is:

> The authenticated v3.8 development source can be transformed into the exact
> frozen scientific input without SEC reacquisition, row selection, omission,
> duplication, reordering, byte substitution, or a change to the science.

The system is long AAPL or cash only. It cannot short, borrow, use leverage,
hold negative cash, earn cash interest, trade another security, call a paid
API, or interact with a broker. It begins with USD 1,000 and uses the same
action stream at 5 and 10 basis points per changing leg.

This is a retrospective development audit, not pristine unseen evidence. The
local model may have learned facts after the historical filing dates during
pretraining. Prompts contain only authenticated decision-time filing material,
but that cannot remove pretrained knowledge. Therefore a historical pass is
only permission to continue the staged audit. Reliable performance language is
reserved for a later prospective paper-trading track whose decisions are
sealed before outcomes exist.

## Development-only boundary

V3.9 accepts only the literal command `development`. It has no stage argument
that can select confirmation or final data. The only market window is
1998-01-01 through 2019-01-01 exclusive; the scored development corpus ends on
2018-12-31. The 2000-2004 interval is warm-up and learning only. Qualification
uses the five frozen blocks:

| Block | First session | Last session |
|---|---|---|
| 1 | 2005-01-03 | 2007-12-31 |
| 2 | 2008-01-02 | 2010-12-31 |
| 3 | 2011-01-03 | 2013-12-31 |
| 4 | 2014-01-02 | 2016-12-30 |
| 5 | 2017-01-03 | 2018-12-31 |

No confirmation or final path, file, count, timestamp, source record, market
value, label, action, return, or diagnostic may be opened. Even a development
pass leaves both later stages closed until the result is committed, pushed,
and authenticated by a separate read-only pushed-result gate, followed by a
new preregistration.

## Exact streaming source bridge

The compact v3.8 replay exposes metadata-only `StageOutput`; it does not expose
the raw source bundle required by `build_legacy_science_projection`. V3.9 may
therefore add one narrowly-scoped read-only streaming adapter.

The adapter must:

1. authenticate the exact base commit, public artifacts, private hash-chained
   journal, receipts, manifests, checkpoint, stage seal, detached replay,
   content-addressed blobs, and inventory;
2. rehydrate the exact compact development stage and require the frozen
   logical checkpoint, role plan, role counts, and `D` membership;
3. sort all and only the 75 `D` rows by `(availability_session, accession)`;
4. load one authenticated complete-submission blob at a time;
5. validate that blob against its content-addressed manifest and exact response
   metadata;
6. strictly parse the complete submission and select the one sequence-1
   embedded document whose exact form matches the authenticated target;
7. slice the exact embedded `<TEXT>` bytes using authenticated offsets;
8. reproduce the raw selected-document and normalized-text SHA-256 values;
9. create the existing `LegacyScienceDocument` value with those exact bytes
   and authenticated identities;
10. release the complete-submission blob before loading the next blob; and
11. reproduce the existing non-streaming legacy projection manifest exactly in
    a synthetic parity test.

`raw_primary_document` means the selected embedded `<TEXT>` bytes. It never
means the full SEC complete-submission response. Normalization is inherited
unchanged. Caller-supplied accessions, dates, forms, filenames, URLs, offsets,
hashes, text, or order are never accepted as evidence.

The bridge must recreate the frozen legacy typed-row identity from the
authenticated v3.8 typed values. Before Yahoo or Gemma, it must require exact
one-to-one parity between the derived frozen development universe and the 75
v3.8 `D` rows for accession, form, filing date, acceptance time, availability
session, selected filename, source identity, normalized identity, membership,
and order. It must prove set equality and sequence equality. Any difference
rejects v3.9 before an external effect; no mapping table may repair it.

For each row, the prior comparison document is only the immediately preceding
same-form row already encountered in this 75-row chronological projection. The
first 10-K and first 10-Q therefore have no prior same-form filing: their
prior-change fields are `not_comparable` and encode as zero, while their
current-impact and quality features remain live. No pre-2000 or non-`D` row is
silently admitted.

The private projection manifest must bind at least:

- all source-authority pins above;
- exact count 75 and exact ordered membership;
- each source, selected-text, normalized-text, typed-row, legacy-record,
  preprocessed-event, canonical-request, and prior-link hash;
- ordered aggregate hashes for documents, records, events, requests, and
  pilot order;
- the science-projection hash;
- exact implementation commit and tree; and
- pre/post v3.8 inventory and run-lock snapshots.

Public evidence may release only the count, aggregate hashes, authority pins,
and pass/fail diagnostics. It may not release an accession, URL, filename,
offset, source body, normalized body, canonical prompt body, private path, or
readable SEC contact.

## Frozen model and request semantics

The following values cannot change:

| Item | Frozen value |
|---|---|
| model | `gemma4:12b` |
| model manifest SHA-256 | `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c` |
| runtime fingerprint SHA-256 | `816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77` |
| prompt SHA-256 | `9ed8496ed101c138cdbee162bdf6dfd53434f6f1e0c64fc93d844405d6eae9f7` |
| schema SHA-256 | `1707ae581abb1a256dfb1ee8f51efd9dd67df5d9b3e8f7c22ee2d92b67b6f82b` |
| temperature | `0` |
| seed | `0` |
| context | `6144` tokens |
| output cap | `512` tokens |
| preprocessed input cap | `20,000` UTF-8 bytes, 72 sentences, 220 characters per sentence |
| response transport cap | `256 KiB` per response |
| retry, repair, pull, fallback, alternate model | all exactly `0` |

The prompt is a filing reader, never a trader. It must not receive prices,
future records, labels, returns, actions, gate values, stage results, or the
pause threshold. Exactly one canonical request is made for each of the 75
unique development filings. A successful HTTP response that is invalid under
the frozen schema is sealed once, is not retried or repaired, and becomes the
frozen neutral semantic vector with quality risk set to one.

A missing, unauthenticated, transport-truncated, unparseable-envelope,
transport-failed, or runtime-unguarded response makes the consumed development
attempt terminally fail. It is not converted into a neutral row and is never
resent. By contrast, a bounded authenticated Ollama envelope that is
structurally valid but reports an abnormal `done_reason` is sealed once as an
invalid extraction and follows the frozen neutral-semantic, quality-risk-one
rule. It is not a transport truncation and is not retried.

## Runtime identity guard

One runtime probe is exactly two loopback HTTP requests in this order:

1. `GET http://127.0.0.1:11434/api/version`;
2. `POST http://127.0.0.1:11434/api/show` for `gemma4:12b`.

Each request uses the inherited hardened loopback transport with no proxy,
redirect, retry, streaming, pull, fallback, or alternate endpoint. Raw bounded
responses are sealed privately. The exact semantic identity, active model
digest, manifest, config, ordered layer digests, and runtime fingerprint must
match the frozen values.

The complete model batch is guarded immediately before and immediately after.
Normally, one probe precedes the first generation and one probe follows the
75th sealed generation, for exactly four identity HTTP requests. Each boundary
must use the v2.2-compatible `verify_installed_pinned_runtime` semantics, or an
exact v3.9 equivalent: Gemma 4 has two active `FROM` blobs and four ordered
layer contents. The older v1 runtime identity type is incompatible and may not
be reused. Each boundary also rehashes the local manifest, config, and all
ordered layer contents. Model outputs remain opaque and unparsed until the
post-batch boundary independently passes all candidate pins.

The guard compares candidate-authoritative semantic fields. It requires equal
model manifest, config digest, ordered layer content digests, exact version
pin, canonical show-semantic hash, model-info identity, active `FROM` blobs,
and frozen runtime fingerprint. It does not require raw `/api/show` bytes or
whole receipt hashes to be equal. The exact `modified_at` value, whitespace,
JSON key order, and raw-show hash are diagnostic only; only `modified_at` is
excluded before the strict canonical semantic comparison.

If the mandatory pilot pause described below occurs, the pilot becomes one
closed guarded segment: its pre-probe is the original batch pre-probe, and a
post-probe is made immediately after the fifth sealed generation before the
pause is released. After fresh authorized continuation, a new pre-probe and
post-probe guard the remaining 70 generations. Thus a completed paused-and-
resumed attempt has exactly eight identity HTTP requests. The two segment
guards bind their own exact ordered generation receipt hashes, and one v3.9
aggregate receipt binds both guards and the combined exact 75-row execution
order. A single guard may never be claimed to span the pause. No identity
request is repeated after an open intent, and no generation is repeated.

The inherited v2 semantic-row receipt assumes one shared runtime-probe hash.
That receipt cannot falsely bind both paused segments to one probe. V3.9 must
add a segment-aware provenance envelope: each row binds its segment and
pre-probe, each segment binds its before/after receipts and ordered call
receipts, and the aggregate stage receipt binds both segments. This changes
only operational provenance, never the semantic row or deterministic science.

Any candidate-authoritative identity mismatch, semantic pre/post difference,
unexpected response, open probe intent, or successful-path identity effect
count other than four normally or eight after the mandated pause terminally
rejects the attempt. An attempted request that fails before the full successful
count is journaled honestly and terminates; the successful-path count is not a
claim that a failed no-retry transport somehow completed.

## Exact Yahoo market batch

V3.9 uses the inherited Yahoo Chart v8 acquisition unchanged. The six logical
symbols and provider symbols are requested once in this exact order:

| Order | Logical symbol | Provider symbol |
|---:|---|---|
| 1 | AAPL | `AAPL` |
| 2 | SPY | `SPY` |
| 3 | QQQ | `QQQ` |
| 4 | IWM | `IWM` |
| 5 | VIX | `^VIX` |
| 6 | TNX | `^TNX` |

Every request uses only
`https://query1.finance.yahoo.com/v8/finance/chart/{percent-encoded-symbol}`
with ordered query items:

1. `period1=883612800`;
2. `period2=1546300800`;
3. `interval=1d`;
4. `includePrePost=false`;
5. `includeAdjustedClose=true`; and
6. `events=div,splits`.

This is the exact 1998-01-01 through 2019-01-01-exclusive development request.
Transport uses normal TLS certificate and hostname verification, the inherited
fixed user agent, no proxy, cookie, authentication, redirect, compression,
alternate host, provider, or fallback, and no retry. The bounds remain 30
seconds and 64 MiB for each response, 210 seconds and 128 MiB for the batch.

There are exactly six Yahoo requests. A durable intent precedes each one and a
committed response plus marker-last checkpoint follows it. A crash after both
`response_committed` and `checkpoint_committed` may continue at the next Yahoo
request. A crash after an intent without `response_committed` is terminally
indeterminate and that request is not resent. No market value,
metadata, row count, date, price, dividend, split, return, label, or diagnostic
is opened until all six exact responses are sealed and the complete inherited
market validator passes. Raw Yahoo bodies and transport-only metadata remain
private.

## Fixed pilot and 12-hour rule

The old 2,160-second Gemma cap and 3,600-second complete-run cap are operational
limits from an obsolete execution path. They do not change the science and do
not reject v3.9. V3.9 replaces only those caps with this preregistered rule.

Before any generation, the zero-effect preflight fixes the five pilot filings:

1. sort all 75 canonical request byte strings by byte length descending;
2. break equal-length ties by accession ascending;
3. select the first five; and
4. execute those five in that order.

The pilots are ordinary members of the 75-row science batch. They are never
duplicated. If no pause occurs, the remaining 70 are executed in canonical
`(availability_session, acceptance_datetime, accession)` event order with the
five pilot members removed. The final semantic rows are restored to that same
canonical science-event order before feature construction. This downstream
event order is deliberately distinct from the source-projection and prior-link
order `(availability_session, accession)` frozen above.

Each pilot request intent, exact request hash, response, elapsed monotonic
duration, and completion is individually durable. For each pilot, the runner
records `start_ns = time.monotonic_ns()` immediately after the request intent is
durable and immediately before invoking the HTTP transport. It records
`end_ns = time.monotonic_ns()` immediately after the bounded response body has
been completely read, the declared framing and byte cap have been checked, and
the response has been closed, but before response persistence or semantic
parsing. `duration_ns = end_ns - start_ns` must be a positive integer and is
stored as a canonical decimal integer. After all five are sealed, the runner
may read only those integer durations and the count and computes exactly:

`projected_ns = sum(pilot_duration_ns) + 70 * max(pilot_duration_ns)`

No average, percentile, adjustment, manual estimate, or later observation may
replace that formula. The decision uses no output text, schema validity,
semantic value, market value, price, label, action, return, performance, or
gate result.

If `projected_ns` is less than or equal to `43_200_000_000_000`, the runner
continues without asking. If it is strictly greater than that integer
(43,200 seconds), it:

1. makes the pilot post-probe and requires the guarded prefix to pass;
2. appends and fsyncs `paused_for_justification`;
3. closes with an authenticated checkpoint after exactly five generations;
4. publishes only pilot count, finite durations, the formula, projected
   duration, threshold, hashes, and effect counts;
5. does not parse or expose model responses or market values; and
6. makes no sixth generation until the user gives fresh explicit permission.

The redacted pause artifact path is
`e/aapl_sec_gemma_lean_science_v3_9/DEVELOPMENT_PAUSE.json`. Let `I` be the
pushed implementation commit and let `F` be the later pushed preflight-evidence
commit whose parent is `I`; the development attempt starts only when local
`HEAD`, the remote branch ref, and `F` are identical. If the pause occurs, the
pause commit `S` must have parent `F` and exactly one added path: the pause
artifact. `S` is pushed and remote-verified. A short continuation
preregistration and justification is then added at
`docs/aapl_sec_gemma_lean_science_v3_9_continuation.md`. It may contain only the
authenticated timing/count evidence, the fixed remaining order hash, the two
segment design, the expected remaining compute, and the unchanged safety
rules. Its commit `C` must have parent `S` and exactly one added path: that
document. Relative to `F`, `C` therefore has exactly two added paths and no
other tracked delta. `C` is pushed and remote-verified. Only after both pushes
does the user receive the simple timing explanation and decide whether to
continue.

Continuation requires fresh explicit user permission after those pushed
artifacts, exact authentication of the five pilot receipts and pilot guard, a
clean worktree, unchanged executable source hashes and preflight pins, and a
new pre-probe. Immediately before resume, local `HEAD`, the remote branch ref,
and `C` must be identical. The runner may accept only this two-commit descendant
chain and must reprove that every executable, test, contract, preflight, and
earlier tracked blob equals `F`; no other tracked change is allowed. The user
permission authorizes only the remaining local compute; it does not authorize
a science change or any new data source. Refusal or no reply leaves the attempt
safely paused. It is not silently scored as failure.

An interruption after a pilot generation but before the successfully persisted
pilot post-probe is not a clean pause. The word "immediately" cannot be restored
later, so that attempt is indeterminate and may neither resume nor regenerate
the prefix.

## Frozen deterministic science

The bridge may reuse the already-tested pure functions, but it may not use the
old v2 operational runner, old store, old acquisition entrypoint, old stage
authorization, or old publication path. Numerical behavior remains exactly the
frozen science projection.

In particular, v3.9 keeps unchanged:

- the 12 ordered features and their exact formulas;
- exact event availability and first-same-form behavior;
- 20-session label maturity and same-session admission order;
- continuous expanding causal refits with at least 20 rows and four rows per
  class before the learner is ready;
- robust-scaled ridge logistic and Huber heads and every solver constant;
- the probability threshold 0.55 and expected 10-bps edge threshold 0.0025;
- the fixed contextual-plus-weak-trend baseline;
- the 20-session non-overlapping SEC cash overlay;
- the semantic, quality-preserving no-meaning, all-zero, and block-frozen
  controls;
- independent control accounts and continuous primary account state;
- next-open fills, t+21 exit open, fractional shares, and exact 5/10-bps costs;
- AAPL buy-and-hold on the same price basis and ledger;
- no reset at a block, year, or later stage boundary; and
- every metric definition and undefined/nonfinite failure rule.

Every output float used for durable identity is canonicalized with the inherited
binary64 hexadecimal rule. The deterministic result must be reproducible from
the sealed market and semantic batches without another external effect.

## Development success gates

Development passes only if every inherited gate passes. The exact 22 numerical
and Boolean conditions are:

| Gate | Required value |
|---|---:|
| combined total active log edge at 10 bps | at least `0.02` |
| combined edge without best block at 10 bps | at least `0.005` |
| positive combined blocks at 10 bps | at least `4` of `5` |
| annual win rate at 10 bps | at least `0.55` |
| negative-AAPL-year win rate at 10 bps | at least `0.60` |
| complete SEC overlay episodes | at least `12` |
| overlay episode win rate at 10 bps | at least `0.55` |
| overlay median edge at 10 bps | strictly positive |
| largest positive episode share at 10 bps | at most `0.35` |
| schema-valid extraction rate | at least `0.90` |
| nonzero filing-meaning rows | at least `24` |
| incremental versus baseline at 10 bps | at least `0.005` |
| incremental versus baseline without best block | strictly positive |
| online versus block-frozen action differences | at least `5` |
| online versus block-frozen difference blocks | at least `3` |
| online versus block-frozen edge at 10 bps | strictly positive |
| semantic versus no-meaning action differences | at least `5` |
| semantic versus no-meaning difference blocks | at least `3` |
| semantic versus no-meaning complete XOR intervals | at least `4` |
| semantic versus no-meaning edge at 10 bps | at least `0.005` |
| semantic edge without best XOR interval | strictly positive |
| semantic Brier relative improvement | at least `0.01` |

Zero action difference is rejection. Any undefined, missing, nonfinite, or
unreconciled dependent value fails its gate. Failure of any one gate blocks a
later stage. No subset, score, narrative judgment, p-hacking adjustment, or
post-result threshold change can turn a failure into a pass.

## One-shot state and crash rules

The fixed attempt identity is
`aapl-sec-gemma-lean-science-v3-9-development-001`.

V3.9 uses a new ignored private namespace under
`data/aapl_sec_gemma_lean_science_v3_9`. The zero-effect preflight and the
development attempt have disjoint children. No predecessor v1, v2, or v3.x
state is copied, adopted, repaired, upgraded, or translated.

The attempt store must use one exclusive lock, canonical JSON, content-addressed
payloads, atomic write/replace, directory fsync where supported, an append-only
hash-chained journal, and marker-last completion. It must bind the exact plan,
attempt, implementation, preflight, source, science, effect budget, request
order, pilot order, and previous event hash.

A durable attempt intent is committed before the first external effect. A
durable per-request intent is committed before every Yahoo, identity, and Gemma
request. The response transition is exact:

1. fully read and close the bounded response;
2. write its canonical content-addressed payload to a temporary file, fsync the
   file, atomically rename it, and fsync the parent directory where supported;
3. append one hash-chained `response_committed` event containing the request
   intent and payload hashes and fsync the journal; this event is the sole
   authoritative response-completion marker;
4. write and fsync the derived checkpoint as a content-addressed immutable
   payload; and
5. append and fsync a marker-last `checkpoint_committed` event binding the
   checkpoint hash before any next external intent.

There is no separate ambiguous response marker. Deterministic projections,
aggregate guards, checkpoints, and evaluation may be recomputed only from
authoritative committed events and only if their hashes reproduce exactly.

Crash behavior is fixed:

- before an intent, the unit has not happened and may be executed once;
- after an intent but before `response_committed`, the attempt is terminally
  indeterminate and the unit is never retried or resent; a payload file without
  that journal event is an orphan and is never adopted;
- after `response_committed` but before `checkpoint_committed`, an uninterrupted
  process must finish the checkpoint before another effect; after a restart,
  an allowed non-model boundary deterministically rebuilds the exact checkpoint
  bytes and hash from the committed journal event and payload; an existing
  content-addressed checkpoint payload is usable only after exact byte/hash
  equality with that independent reconstruction, while an absent payload is
  written once under its derived hash;
- a `checkpoint_committed` event with a missing or mismatched checkpoint is
  terminal corruption, while any checkpoint payload without its event remains
  inert and non-authoritative; appending the independently rebuilt exact marker
  at an allowed non-model boundary is deterministic reconstruction, not
  adoption or repair of external-effect evidence;
- after a complete Yahoo-unit checkpoint, restart continues at the next Yahoo
  request without repeating the committed request;
- process termination anywhere after the model-segment pre-probe begins and
  before that segment's post-probe `response_committed` event is durable makes
  the attempt terminally indeterminate, even if one or more generation
  responses and checkpoints are already committed; no model guard may span
  unplanned downtime and no generation is retried;
- after a segment post-probe is authoritatively committed, its aggregate guard
  and checkpoint may be rebuilt locally from the exact committed evidence;
- an exact authenticated pilot pause may resume only under its special rule;
- an extra, duplicate, reordered, orphaned, malformed, or foreign event,
  payload, marker, lock, or request poisons the attempt; and
- terminal evidence is written only after all required seals and independent
  replays pass.

There is no overwrite, truncate, delete, reset, force, retry, repair, adoption,
or choose-best path. A failed, rejected, indeterminate, or paused attempt is
preserved honestly.

## Exact effect budget and order

A normal completed development attempt has:

| Effect | Exact count |
|---|---:|
| SEC requests | `0` |
| Yahoo requests | `6` |
| Ollama identity HTTP requests | `4` |
| Ollama `/api/chat` generations | `75` |
| retries, repairs, pulls, fallbacks, paid calls | `0` |
| confirmation/final data opens | `0` |
| broker or real-money effects | `0` |

A successfully completed attempt that crossed the strict pilot pause has eight
identity HTTP requests instead of four; every other count is identical. A
failed request may stop at a smaller journaled attempted count and remains a
failure, never a successful completed effect budget. The complete family SEC
count remains exactly 964 in either successful path.

The normal high-level order is:

1. authenticate pushed preregistration and implementation;
2. authenticate v3.8 public and private source authority read-only;
3. rebuild exact 75-row legacy parity and request/pilot commitments;
4. consume the one-shot attempt before its first external effect;
5. acquire and seal the six Yahoo responses without opening market values;
6. run the model-batch pre-probe;
7. execute and seal the five fixed pilots;
8. apply the timing-only pause rule;
9. if not paused, execute the remaining 70 generations;
10. run the model-batch post-probe and require exact identity continuity;
11. open and validate the complete market and semantic batches;
12. construct features, causal learners, policies, controls, ledgers, metrics,
    diagnostics, and gates deterministically;
13. independently replay all evidence and no-leverage proofs; and
14. seal redacted terminal evidence before releasing a result.

The paused path inserts a pilot post-probe and durable pause after step 8, then
requires a pushed justification, a new pre-probe, the remaining 70 calls, and a
new post-probe. It never repeats steps 5 or 7.

## Implementation allowlist

After this document is committed and pushed alone, implementation may add only
these six production files:

1. `agent_benchmark/sec_gemma_lean_science_v39_contract.py`
2. `agent_benchmark/sec_gemma_lean_science_v39_bridge.py`
3. `agent_benchmark/sec_gemma_lean_science_v39_journal.py`
4. `agent_benchmark/sec_gemma_lean_science_v39_store.py`
5. `agent_benchmark/sec_gemma_lean_science_v39_preflight.py`
6. `agent_benchmark/sec_gemma_lean_science_v39_runner.py`

It may add only these six test files:

1. `tests/test_sec_gemma_lean_science_v39_contract.py`
2. `tests/test_sec_gemma_lean_science_v39_bridge.py`
3. `tests/test_sec_gemma_lean_science_v39_journal.py`
4. `tests/test_sec_gemma_lean_science_v39_store.py`
5. `tests/test_sec_gemma_lean_science_v39_preflight.py`
6. `tests/test_sec_gemma_lean_science_v39_runner.py`

No existing tracked source, test, document, evidence file, configuration, or
ignore rule may change in the implementation commit. Imports from existing
modules are permitted only through explicit v3.9 adapters whose behavior and
source hashes are tested. Production must not import a test helper.

Offline tests must cover at least:

- every frozen constant and canonical projection hash;
- exact implementation allowlist and unchanged predecessor blobs;
- synthetic streaming/non-streaming projection byte parity;
- 75-row set/order/prior-link parity and every mismatch rejection;
- one-blob-at-a-time retention;
- no SEC constructor, transport, preflight, or acquisition reachability;
- Yahoo URLs, order, bounds, no retry, and crash states;
- model requests, pilot selection, remaining order, response bounds, and
  invalid-schema neutral behavior;
- normal four-request and paused eight-request runtime guards;
- strict `> 43,200` pause boundary and timing-only decision;
- no output opening before complete guards and seals;
- every intent/completion/restart/indeterminate transition;
- deterministic science parity and all 22 gate boundaries;
- no-leverage, no-borrowing, no-negative-cash, and same-ledger reconciliation;
- privacy scans over public artifacts and the complete v3.9 private namespace;
  and
- confirmation/final/paid/broker/real-money path rejection.

All applicable repository tests must pass. Implementation then receives one
commit and push. Its remote branch ref, local HEAD, tree, exact 12-file delta,
source hashes, tests, and clean worktree must be verified before preflight.

## Zero-effect v3.9 preflight

After the implementation is pushed, one and only one v3.9 preflight may run.
It is a private read-and-hash pass with no SEC, Yahoo, Ollama, generation,
market-value, prediction, action, performance, broker, or paid effect.

It must:

1. authenticate this preregistration and the exact pushed implementation;
2. prove the implementation delta matches the allowlist and all predecessor
   tracked blobs are unchanged;
3. authenticate the full v3.8 source authority and unchanged inventory;
4. build the exact streaming 75-row projection;
5. prove exact frozen-legacy universe, record, prior-link, preprocessing,
   canonical-request, and pilot-order parity;
6. prove confirmation and final remain unreachable;
7. prove exact effect budgets and journal state transitions using fakes only;
8. run the complete offline test suite and privacy scans; and
9. seal the private preflight manifest before emitting its redacted public
   artifact.

The public artifact path is
`e/aapl_sec_gemma_lean_science_v3_9/DEVELOPMENT_PREFLIGHT.json`. It may contain
only public pins, counts, aggregate hashes, test results, privacy results,
effect budgets, and Boolean gates. It may not contain raw source or response
bytes, accessions, URLs, filenames, canonical requests, private paths, readable
contact data, prices, labels, actions, returns, or performance.

The preflight artifact is committed and pushed alone. A separate read-only
pushed-preflight gate must authenticate its remote commit, tree, parent, exact
one-path delta, Git blob, literal and internal hashes, implementation pins,
private preflight replay, zero-effect counts, and clean worktree. Only that
gate may authorize the one-shot development command.

A failed preflight is terminal for v3.9. It is preserved and pushed. It is not
rerun, repaired, or converted into permission to proceed.

## Result preservation and later-stage rule

The development attempt must preserve one redacted terminal artifact whether
it passes, fails, or becomes indeterminate. A clean explicit pilot pause uses
the separately frozen pause path and is resumable, not terminal. For an
ordinary terminal outcome the path is
`e/aapl_sec_gemma_lean_science_v3_9/DEVELOPMENT_RESULT.json`; the comparison is
updated at `e/APPROACH_COMPARISON.md`. Only those exact public result paths may
change in the result commit.

The terminal artifact must report source/science/implementation/preflight
pins, effect counts, chronological diagnostics, all gate values, all pass/fail
Booleans, no-leverage proofs, privacy checks, evidence limitations, and its own
canonical hash. It must not contain any forbidden private material.

The terminal result and comparison are committed and pushed even on failure.
A normal-path result commit must have parent `F`. A paused-and-resumed result
commit must have parent `C`. In either case its exact delta is the added result
artifact plus the modified comparison and no other path. An indeterminate
outcome before a clean pause uses the same parent that was checked when that
invocation began.

A separate read-only pushed-result gate then authenticates the remote commit,
tree, exact delta, Git blobs, public self-hashes, private terminal replay,
effect counts, deterministic reproduction, gate calculations, privacy, and
unchanged v3.8 source. The public terminal artifact honestly describes the
state when committed; the later read-only gate does not recursively rewrite it.

Only a development pass plus a passing pushed-result gate can justify a new
confirmation preregistration. Confirmation remains 2019-2023 and final remains
2024 onward, but neither is authorized here. A development failure ends this
approach honestly. A development pass is promising retrospective evidence,
not proof of live profitability. No real-money execution is authorized under
any outcome.

## Privacy boundary

The readable SEC contact is private. If exact replay of the old journal needs
it, v3.9 may load it only from the ignored local configuration to reproduce the
already-sealed cryptographic fingerprint. It may not print it, prompt with it,
copy it, store it in v3.9, expose its length or parts, or include it in an error.

The six exact planned Yahoo request URLs are public preregistered constants in
this document. The following remain private: all raw SEC and Yahoo bodies;
accessions; realized response URLs and redirect history; SEC URLs; filenames;
offsets; headers; normalized filing bodies; canonical Gemma requests; raw Gemma
responses; private paths; readable contact data; provider response metadata;
and pre-release market, semantic, label, action, return, and performance values.

Every exception crossing the v3.9 public boundary is mapped to a fixed redacted
error code. Public artifacts and normal console output use only allowlisted
fields. Before publication, the implementation must scan both serialized
public bytes and all v3.9 private serialized bytes for readable contact echo,
forbidden paths, and forbidden public fields. A privacy uncertainty fails
closed.

## Hard falsifiers

Any one of the following terminally rejects v3.9:

- any SEC request or a family SEC count other than exactly 964;
- any v3.8 mutation, acquisition construction, preflight invocation, repair,
  lock acquisition, or predecessor-state adoption;
- any mismatch in a pinned commit, tree, blob, literal hash, internal hash,
  checkpoint, journal, manifest, receipt, replay, seal, plan, inventory, or
  run-lock snapshot;
- a projection count other than 75;
- any omitted, duplicated, reordered, selected, sampled, substituted, or
  non-`D` filing;
- any mismatch between v3.8 and the frozen legacy development universe;
- more than one complete-submission blob retained by the streaming adapter;
- any prior document outside the preceding same-form row in the 75-row
  projection;
- any change to source normalization, prompt, schema, model, digest, feature,
  label, learner, threshold, policy, cost, ledger, control, chronology, metric,
  gate, or evidence classification;
- any Yahoo URL, query, order, request, redirect, retry, proxy, cookie,
  alternate host/provider, or fallback outside the exact six-request batch;
- any successful normal identity count other than four, successful
  paused/resumed identity count other than eight, candidate-authoritative
  pre/post mismatch, incompatible v1 runtime identity, or identity request
  retry;
- treating diagnostic raw-show bytes, whitespace, JSON key order, or
  `modified_at` alone as a semantic runtime mismatch;
- claiming one runtime guard spans a pause, or binding both paused segments to
  one inherited runtime-probe hash;
- any completed generation count other than exactly 75 unique filings;
- any duplicate generation, retry, repair, pull, fallback, streaming response,
  alternate model, changed request bytes, or response above the cap;
- bypassing the strict `projected_ns > 43_200_000_000_000` pause;
- using anything except the five frozen durations and count in that decision;
- continuing after a required pause without fresh explicit permission and an
  authenticated committed-and-pushed pause artifact and continuation
  justification;
- resuming a pilot prefix whose immediate post-probe did not complete and
  persist successfully;
- parsing or releasing semantic outputs before the full applicable runtime
  guard passes;
- opening market values before all six Yahoo responses are sealed and the
  complete batch passes;
- any open external-effect intent being adopted, retried, or resent;
- resuming after an unplanned process interruption inside an open model-guard
  segment, even when its last generation response was committed;
- any result released before terminal sealing and independent replay;
- any confirmation/final data access or later-stage effect;
- any paid API, broker, real-money, short, leverage, borrowed-money,
  negative-cash, cash-interest, or exposure-above-one action;
- any privacy leak or unredacted error; or
- any uncommitted, unpushed, dirty, or out-of-allowlist tracked change at an
  authority gate.

Failures are evidence. They must be preserved on their own branch and may not
be erased, relabeled, or replaced by a more favorable run.
