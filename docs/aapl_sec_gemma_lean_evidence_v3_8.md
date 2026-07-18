# AAPL SEC/Gemma lean evidence v3.8

## Status and immutable predecessor

This is a new, separately preregistered checkpoint-binding successor on
`codex/aapl-sec-gemma-lean-evidence-v3-8`. This document must be committed and
pushed by itself before any v3.8 implementation commit, loopback identity
call, SEC request, Yahoo value, Gemma generation, performance calculation,
confirmation source, or final source is opened.

V3.7 is terminally rejected and preserved at commit
`8d84ca6b1d8a5ef50d790beb9b4b53e51812a542`, tree
`1287094442f4ceecdaf0ca29b9bdb4980d65b7e7`. Its public terminal artifact is
`e/aapl_sec_gemma_lean_evidence_v3_7/DEVELOPMENT_ACQUISITION.json`, Git blob
`5d93d8cacd314732d2cff504bf5ce015b0e34a1c`, literal file SHA-256
`271a036422861199a0e13236107e3664249d4cdab1b96d4ff411292afe52787d`, and
internal acquisition SHA-256
`31d0d7587a7263b294bb156d81ceaf6f06c073d66562baacf8330361747054a5`.
The comparison update in that terminal commit has Git blob
`2a79cccc44afc7097428d2dda7f81cc75d6e70b8` and literal SHA-256
`c8ff4da3c939f92aac73739bde870c4fd8696394c06ecd4db2eef2e5f37599b7`.
V3.8 never resumes, reruns, repairs in place, or reclassifies that attempt.

The v3.7 preregistration is pinned at commit
`fcebe9b6de9d2de81e7cc9c79b011b2e0ca1051a`, tree
`c2e141d3546da5d11d0ab9d2e120b0685798eac6`, document blob
`ec49b15630eac71132d321c88e8e02f7cfa5a488`, and document SHA-256
`645b4adafc56e71bc71d41780501db85963b5a485d3841230f5a2075b13143a7`.
Its implementation is pinned at commit
`6cd3d46ed86bfa8553e465741ab6950e871140c9`, tree
`29e7df70f20eefa12dca645fbfb83f2975ba83d1`, with committed-blob delta
manifest SHA-256
`8fe9db80215dbdf2facfbd6c1cda63c14ee42278931f8cb488327b136f516480`.
Its one-shot zero-source-effect preflight is pinned at evidence commit
`dfc4e34a4cdcd4188a1946db534c1b22b3c2eb74`, tree
`4feae6923949ad7fbd26df792334471e18072a8e`, public artifact blob
`13878558b12f7fa96f8a623bc6fd5db13a31be29`, preflight SHA-256
`7621da90259156ad069b5d9bac8e061c8d707ff58b0b339f86ab6927402e648d`, and
literal file SHA-256
`cd303e4a3c7715d6f99f6cd9eb0922407cae7d67bf9cc8af56d0962ad01c9e44`.

Every v3.8 counterpart is derived from the same-name committed v3.7 path at
the terminal commit `8d84ca6b1d8a5ef50d790beb9b4b53e51812a542`, never from a worktree,
private artifact, older implementation commit, or preflight commit. Useful
public code pins at that terminal tree are:

| Committed v3.7 path | Git blob | Literal SHA-256 |
|---|---|---|
| `agent_benchmark/sec_gemma_lean_v37_acquisition.py` | `4323686362ff3d690cfe27358d4d3c07635d14f4` | `082238efe69b29048a9951d6caf576c12e1a96c2e93138bbb55611989024d308` |
| `agent_benchmark/sec_gemma_lean_v37_source.py` | `f2882c8c5c8d82e1f32392dc275d01bbb280fb94` | `d063d3a95d4ea2b3eb21105c3e7c03dc269479570f1df43d858f8ab42b20ca46` |
| `agent_benchmark/sec_gemma_lean_v37_delta.py` | `2277bb78631de0369edb78f9c6f81dad6e21ddae` | `bada9dc67d168c721a9f48af3505e48e3059897faae86ec9e179741cd7d7afdf` |
| `agent_benchmark/sec_gemma_lean_v37_journal.py` | `05a841d51060caf1c3267a56c56b7048778e0abe` | `f9b3a6de35c20bdb6896a03854cfdbd12a60138b93d5fcaa4b6c0df1f56fc5ee` |
| `agent_benchmark/sec_gemma_lean_v37_preflight.py` | `0d39f8bec5cd1850817f928524a22c5338216a83` | `f192f4bab33c5980428e436457429e88cf5de069dda18887c5f93ef915fd51c8` |
| `agent_benchmark/sec_gemma_lean_v37_transport.py` | `3c9b5bab0d3a8acb7c10e8456246a35b3b060080` | `bc0de296851ec6fd853a261ca2bc4e5cf04907f12080144479b4559fe905a28a` |
| `tests/test_sec_gemma_lean_v37_acquisition.py` | `867336123607f2d27773ad17bfe324f2f0313a9a` | `d93ea7c7b65dc1f3c7f127b4855acac70e4722d117728fbe121384c1e74b4ab1` |
| `tests/test_sec_gemma_lean_v37_delta.py` | `43df7176d5d62b074f8191e1a74d604099d9cbb8` | `c2c635c0443cccfb5fdebbb03834b493fe528093bba5607cb001a5cf293f48ca` |
| `tests/test_sec_gemma_lean_v37_journal.py` | `2fe98d500a51c0f7c41da0cbc8676a2917f77cd8` | `18b93d0306185bea668e4424068e67e9449f5014d7258f1eb1cd7251c521466f` |
| `tests/test_sec_gemma_lean_v37_preflight.py` | `ef1db3a25f973444a7079b50fbe62cf76b30fa46` | `01f6b0ee9d7648a42c62a6c584bc51053f4a8859fee34605b44867ba049a65b7` |
| `tests/test_sec_gemma_lean_v37_source.py` | `7834a92a9f8ec2aa8d7a4dde708ec29008da37d8` | `2df46ca80507aaa06e26d67327fa13a524281e821cc8348ace8cab61e8d478d8` |
| `tests/test_sec_gemma_lean_v37_transport.py` | `7fe09b3ee3ee1f2cc8afd83d081646323f56da5b` | `ed519f5f0ec056f73d12be1e284a209b630db4f7f4b37c73cf7386f1f3b6bf5c` |

The original scientific parent remains
`efbc481c57e480d48303763163676e64e87df49d`.

## Public v3.7 result and evidence still closed

V3.7 made exactly 199 fresh official SEC requests, all returning HTTP 200, and
sealed all 199 planned roles: two Apple Submissions roles, 100 quarterly
master roles from 1994-Q1 through 2018-Q4, and 97 complete-submission roles.
It ended with no failed role, no open intent, and no unsealed response. Its
four authenticated acquisition invocations contain 609 journal events and a
clean terminal rejection. The executed v3.7 production-source inventory
SHA-256 is
`26afcdccb877a4ff796e8906dfbf639f8783ea2a9d461681d4067e2928937bd6`;
it is predecessor evidence only.

Deterministic source finalization passed inside the failed invocation with
`I=97`, `P=97`, `U=75`, and `D=75`. All 75 admitted development rows met exact
acceptance; their forms were 18 Forms 10-K and 57 Forms 10-Q. Fifteen older
rows were unsupported prehistory with null availability and null stage, and
seven rows were ordinary 1998/1999 pre-stage rows. Those values show that the
v3.7 left-boundary correction behaved as preregistered, but they are public
diagnostic evidence only. They are not a checkpoint, a complete-source replay
authority, or a public source authority, and they may not be used as a v3.8
target, fixture, row allowlist, request-count assumption, or acceptance claim.

The exact public failure is mechanical. The committed v3.7 acquisition caller
invoked the required `build_compact_checkpoint(...)` source builder without
supplying its required keyword `main_parse_receipt_sha256`. Python therefore
raised a `TypeError`; the source-only entrypoint correctly failed closed as
`source_rejected` before checkpoint persistence. No source gate, calendar
gate, count gate, acceptance gate, detached replay, market, model, learner,
performance, or trading rule produced that failure.

From the start of v3.8 work, no production path, test, diagnosis, or manual
step may open, read, mount, import, copy, compare, parse, replay, reseal, or
otherwise reuse any private v3.7 response, journal, role blob, parse receipt,
manifest, checkpoint path, marker, or diagnostic. V3.8 may use only committed
code, committed tests, committed documentation, and the redacted public v3.7
terminal evidence pinned above.

Before v3.8, this approach family has made exactly 765 official SEC requests:
v3 made 4, v3.1 made 2, v3.2 made 15, v3.3 made 103, v3.4 made 109, v3.5 made
134, v3.6 made 199, and v3.7 made 199. Every v3.8 preservation must report
v3.8 effects separately and the exact cumulative family total as
`765 + v3.8 actual official SEC requests`. No v3.8 request count is assumed in
advance. Every durable v3.8 SEC role intent counts once, including a request
that fails or produces no usable response. The two permitted loopback identity
calls are not SEC requests and must be reported separately.

## Frozen hypothesis, source rules, and science

The scientific question remains literal and unchanged:

> Can a fixed local Gemma reader extract point-in-time deterioration evidence
> from Apple 10-K and 10-Q filings that helps a chronological long-or-cash
> learner avoid enough harmful AAPL exposure to add after-cost value over the
> frozen exhaustion baseline and a no-filing-meaning control?

V3.8 inherits every scientific and source rule from the pinned v3.7
preregistration and terminal implementation:

- model identity, model fingerprint, prompt, schema, seed, temperature,
  context, output caps, and the development model-call cap of 80;
- the twelve filing values, market features, and missingness flags;
- learner, fitting, threshold, interval, non-overlap, and next-open rules;
- inherited exhaustion baseline, semantic control, and no-meaning control;
- development, confirmation, final, and frozen-learning gates;
- 5-bps and 10-bps same-ledger AAPL buy-and-hold comparisons; and
- AAPL-or-cash only, no shorting, no leverage, no borrowing, no negative cash,
  no paid API, no real-money action, and no future leakage.

`H`, `I`, `P`, `Q`, `U`, `D`, every cap, the 95% acceptance requirement,
admitted counts, chronological stage windows, stage ordering, complete-source
reconciliation, and every performance threshold remain unchanged. No filing
deletion, observed-row exception, prompt rewrite, feature change, class
rebalance, model swap, threshold search, alternate horizon, or
performance-informed repair is allowed.

The sole v3.8 operational hypothesis is:

> The public v3.7 terminal artifact reports that diagnostic source
> finalization passed, but no checkpoint or source authority was created
> because the acquisition orchestrator omitted one already-required checkpoint
> argument. V3.8 tests whether explicitly passing the first authenticated role
> manifest's parse-receipt hash to the unchanged checkpoint builder removes
> only that call-site failure. A fresh v3.8 checkpoint and authority may exist
> only if every frozen source, checkpoint, persistence, and detached-replay gate
> independently passes.

This hypothesis does not assert that fresh official v3.8 data will reproduce
`D=75`, use 199 requests, pass finalization, or yield identical manifests,
hashes, accessions, dates, rows, forms, or counts. It changes no source
builder, validator, parser, calendar, stage, model, or science behavior.

## Frozen calendars, stage configurations, and caps

V3.8 keeps the exact v3.7 two-calendar causal rule. The caller, bundle replay
state, and science use the frozen experiment calendar:

- calendar ID `nyse_trading_session_dates_2000_01_01_2026_07_10_v2`;
- covered interval `2000-01-01` through `2026-07-10`;
- 6,669 sessions, first `2000-01-03`, last `2026-07-10`;
- newline SHA-256
  `f3ea99a9fcbe99187701acb030cd6fd0f6dc2c7be53528ef7f5adb13a1c5fdd3`;
  and
- canonical-JSON SHA-256
  `e0550f12f98d7e0cf38d6797ae0d0e410bb3f9006793830ed75e0f753ccc5cf9`.

Availability search uses the frozen market-history calendar:

- calendar ID `nyse_trading_session_dates_1998_01_01_2026_07_10_v1`;
- covered interval `1998-01-01` through `2026-07-10`;
- 7,173 sessions, first `1998-01-02`, last `2026-07-10`;
- newline SHA-256
  `df335e3d907a517e4072bbbc005433d7ff4cb930631740849b412700989d7ae6`;
  and
- canonical-JSON SHA-256
  `e9d37d63d158f8a3b6de58ef81970b27bcac9edb6a9c7b9e2f3ffe477d2f3032`.

The unchanged shared calendar module is Git blob
`7242b803e1d3b909abf40133fb2de2534189cabf`, literal SHA-256
`9a463fa0453440dc777a850e7af934dfc0f2740243e7e8e05ff6ebc430262f71`.

The full ordered tuple relation remains
`EXPECTED_MARKET_HISTORY_SESSIONS[504:] == EXPECTED_SESSIONS`. Availability
uses the unchanged maximum of filing date, exact acceptance date when present,
and authenticated `DATE AS OF CHANGE` when present, then selects the first
frozen history session strictly greater than that boundary. Boundaries earlier
than `1997-12-31` remain unsupported prehistory with null availability and
null stage. Genuine 1998/1999 availability remains pre-stage. A boundary of
`1999-12-31` maps to `2000-01-03` and enters development. A boundary on or
after the final frozen session remains null. The v3.7 composite calendar
receipt, seal validation, and detached-replay recomputation remain
byte-mechanically unchanged after namespace normalization.

No stage, model-call, request, byte, or time bound changes:

| Stage | Availability window | Master coverage | `Q` | `I` cap | `P` cap | Admitted `D` | Maximum requests |
|---|---|---:|---:|---:|---:|---:|---:|
| Development | 2000-01-01 through 2018-12-31 | 1994 Q1 through 2018 Q4 | 100 | 128 | 128 | 72-80 | 245 |
| Intermediate | 2019-01-01 through 2023-12-31 | 1994 Q1 through 2023 Q4 | 120 | 160 | 24 | 19-20 | 161 |
| Final | 2024-01-01 through 2026-07-09 | 1994 Q1 through 2026 Q3, filtered at the cutoff | 131 | 176 | 16 | at most 12 plus every inherited completed-year rule | 164 |

A passing stage still has exactly `1 + H + Q + P` successful official SEC
requests. `H <= 16`; lifetime intents remain capped at 256 per stage. The
individual, aggregate successful-response, aggregate decompressed-master,
lifetime received-byte, `L + 1` reservation, TLS, privacy, timeout,
pause/resume, one-intent, no-retry, and crash-open rules remain exactly those
of v3.7. The fixed final cutoff does not advance with the calendar. V3.7's
observed counts and 199-request formula are diagnostic predecessor evidence
only; v3.8 must rebuild `H`, `I`, `P`, and its request formula from fresh
responses.

The frozen shared science contract remains Git blob
`897286a72be5af2df3e5d59655838ef92364b086`, literal SHA-256
`a51a96a9b763fd3f16cb6461b7dfca58d88cafd5862cff7cc5d18d990994a68e`.
It and every other model, prompt, feature, market, learner, ledger,
performance, stage-access, and authorization path remain byte-unchanged.

## Exact checkpoint-binding correction

After the ordered v3.7-to-v3.8 namespace replacements, the only production
semantic change outside the new delta verifier is in
`DiskBackedSecAcquisition._run_locked`. The existing
`build_compact_checkpoint(...)` call receives exactly this keyword immediately
after `stage_output` and before the phase-receipt hashes:

```python
main_parse_receipt_sha256=records[0].manifest[
    "parse_receipt_sha256"
],
```

There is no `str(...)`, `.get(...)`, fallback, dynamic `**kwargs`, helper,
literal digest, public v3.7 digest, phase-receipt substitution, negative index,
or alternate record. The unchanged `_load_role_records` path has already
authenticated the complete ordered role sequence. The unchanged checkpoint
builder independently requires sequence zero to be the `submissions/main`
role, validates the supplied value as a canonical receipt hash, and requires
exact equality with the first authenticated manifest's
`parse_receipt_sha256`. Passing the raw manifest value preserves that
independent type and equality validation.

The only corresponding non-delta test semantic change is in
`_FakeSource.build_compact_checkpoint`. The fake must stop synthesizing the
missing value from the manifests and instead require and verify the caller's
explicit binding:

```python
main_parse_receipt_sha256 = kwargs["main_parse_receipt_sha256"]
assert main_parse_receipt_sha256 == manifests[0]["parse_receipt_sha256"]
```

The fake checkpoint stores `main_parse_receipt_sha256`, not a separately
derived value. Existing complete fake-acquisition and full-size three-stage
orchestrator tests must therefore fail if the production caller omits or
substitutes the keyword.

The v3.8 source module and source tests must be exact mechanical counterparts
of v3.7. The checkpoint parameter remains required; its type, canonical-hash
validation, first-role validation, equality check, schema, receipt, checkpoint
seal, source authority, and detached replay may not be weakened or inferred.

## Canonical v3.8 sequence

After all Git and preflight gates below pass, development runs in this order:

1. reacquire the main Apple Submissions JSON once;
2. reacquire every referenced historical Submissions file once in canonical
   filename order;
3. build rejection-only pre-master `I` and `P` upper bounds;
4. fetch every master from 1994-Q1 through 2018-Q4 in canonical order;
5. retain every structurally valid row under the unchanged master rule;
6. exact-set reconcile the complete Apple Submissions/master target set;
7. fetch every required complete submission in `(filing_date, accession)`
   order from the exact master filename;
8. parse each original body under the unchanged source, envelope, raw-form,
   acceptance, topology, and byte-identity rules;
9. derive the unchanged maximum source-date boundary, apply the exact frozen
   history-calendar/censor rule, and apply the unchanged stage windows;
10. finalize the complete development source seal;
11. pass the first authenticated role manifest's parse-receipt hash explicitly
    to the unchanged compact-checkpoint builder;
12. persist the checkpoint and prove exact detached replay; and
13. only then preserve and push public source authority before a separate gate
    may allow inherited development Yahoo values, fixed Gemma requests,
    features, labels, ledgers, or performance calculations.

Intermediate remains forbidden unless development passes its unchanged
scientific gate. Final remains forbidden unless intermediate passes. V3.8
reacquires its own response-time evidence; all predecessor bytes remain
diagnostic public evidence only.

## Implementation and committed-blob proof

The mandatory sequence is:

1. commit and push this document alone;
2. add the exact v3.8 implementation and focused offline tests;
3. commit and push that implementation;
4. run one fresh zero-source-effect v3.8 preflight;
5. commit and push the preflight receipt alone;
6. only then make the first v3.8 SEC request;
7. preserve and push either the terminal rejection or complete source seal;
8. authenticate a passing pushed authority with a separate read-only gate
   before any scientific stage can open.

The implementation commit may add exactly these twelve regular `100644`
files:

- `agent_benchmark/sec_gemma_lean_v38_acquisition.py`;
- `agent_benchmark/sec_gemma_lean_v38_delta.py`;
- `agent_benchmark/sec_gemma_lean_v38_journal.py`;
- `agent_benchmark/sec_gemma_lean_v38_preflight.py`;
- `agent_benchmark/sec_gemma_lean_v38_source.py`;
- `agent_benchmark/sec_gemma_lean_v38_transport.py`;
- `tests/test_sec_gemma_lean_v38_acquisition.py`;
- `tests/test_sec_gemma_lean_v38_delta.py`;
- `tests/test_sec_gemma_lean_v38_journal.py`;
- `tests/test_sec_gemma_lean_v38_preflight.py`;
- `tests/test_sec_gemma_lean_v38_source.py`; and
- `tests/test_sec_gemma_lean_v38_transport.py`.

No existing tracked path may change. Relative to the preregistration commit,
the implementation delta must contain exactly twelve `A` records and zero `M`
records, with no rename, copy, deletion, type change, submodule, or nonregular
mode.

Each added file is derived from its same-name v3.7 counterpart at the exact
terminal commit. The ordered raw-byte replacements are:

1. `sec_gemma_lean_v37` to `sec_gemma_lean_v38`;
2. `SecGemmaLeanV37` to `SecGemmaLeanV38`;
3. `V37` to `V38`;
4. `v37` to `v38`;
5. `v3_7` to `v3_8`;
6. `v3-7` to `v3-8`; and
7. `v3.7` to `v3.8`.

The exact replacement counts, in that order, are:

| V3.8 path | Counts |
|---|---|
| `agent_benchmark/sec_gemma_lean_v38_acquisition.py` | `12, 271, 0, 1, 2, 7, 2` |
| `agent_benchmark/sec_gemma_lean_v38_delta.py` | `43, 1, 11, 11, 2, 1, 7` |
| `agent_benchmark/sec_gemma_lean_v38_journal.py` | `0, 105, 0, 0, 0, 3, 1` |
| `agent_benchmark/sec_gemma_lean_v38_preflight.py` | `4, 142, 0, 0, 2, 12, 3` |
| `agent_benchmark/sec_gemma_lean_v38_source.py` | `0, 318, 0, 16, 0, 0, 9` |
| `agent_benchmark/sec_gemma_lean_v38_transport.py` | `5, 140, 0, 0, 0, 2, 3` |
| `tests/test_sec_gemma_lean_v38_acquisition.py` | `11, 38, 0, 0, 1, 1, 0` |
| `tests/test_sec_gemma_lean_v38_delta.py` | `19, 0, 8, 1, 1, 0, 0` |
| `tests/test_sec_gemma_lean_v38_journal.py` | `2, 43, 0, 0, 0, 0, 0` |
| `tests/test_sec_gemma_lean_v38_preflight.py` | `2, 28, 0, 0, 0, 2, 0` |
| `tests/test_sec_gemma_lean_v38_source.py` | `2, 56, 0, 1, 0, 0, 2` |
| `tests/test_sec_gemma_lean_v38_transport.py` | `1, 15, 0, 0, 0, 0, 0` |

After normalization, the acquisition production change has exactly these
physical symbol identifiers and no changed import:

- `class|DiskBackedSecAcquisition|1`; and
- `function|DiskBackedSecAcquisition._run_locked|1`.

With the exact formatting preregistered above, its candidate raw SHA-256 is
`8ba5b3dee285a5e12e631c5d5e8fbafce1a66d1c81b4c5736371ac8ceffbb3f6`
and its changed-symbol evidence SHA-256 is
`3b7a76502cda4ab6da4e88585c4ee7d234b7d86fe80096384126236b88ed04d3`.

The acquisition-test change has exactly these physical symbol identifiers and
no changed import:

- `class|_FakeSource|1`; and
- `function|_FakeSource.build_compact_checkpoint|1`.

With the exact formatting preregistered above, its candidate raw SHA-256 is
`48f247952c7db6eede09b5b32b918b419e6c2e8f641028872907f29df6e9c2c8`
and its changed-symbol evidence SHA-256 is
`e81a3685164f185209b59251d7b9dc8b79246d6f4c3a37170aadc6948c6550be`.

The eight journal, preflight, source, and transport production/test files have
empty changed-symbol and changed-import sets. Their exact mechanical candidate
raw SHA-256 values are:

| V3.8 path | Candidate raw SHA-256 |
|---|---|
| `agent_benchmark/sec_gemma_lean_v38_journal.py` | `c433fcfa0b201546ed22bb7fed550c21eaf780170fc55336a82839df74d654b6` |
| `agent_benchmark/sec_gemma_lean_v38_preflight.py` | `07b1610a3c2eabfab1bc4edcc4c5a3b198a44cf5a4c311f006401ae8a63a5ad5` |
| `agent_benchmark/sec_gemma_lean_v38_source.py` | `8f624df2b28425b8e31db81997aeb0387d3c19418c0781e3b04fca66830d0f04` |
| `agent_benchmark/sec_gemma_lean_v38_transport.py` | `ec60f3868b1f5f3f4f638215e888d35bd3cae97ca442dedb3de4d28249252a26` |
| `tests/test_sec_gemma_lean_v38_journal.py` | `e1e5fd68d0e125624cc7edbd891578f3afef1c5cb88288e4b0c103839bb4b16e` |
| `tests/test_sec_gemma_lean_v38_preflight.py` | `ad805271af364b835ed59dca93cdc322bb825bf42d51551aad9f2512ea14929b` |
| `tests/test_sec_gemma_lean_v38_source.py` | `5c39dac99e2b8f73d2411bac18dff0cfe47218809dfebc11df32943e64c73e93` |
| `tests/test_sec_gemma_lean_v38_transport.py` | `965eba624ddcabc73dbc822ea970d8123cf79225d4c73dcec2bc83bced00b68b` |

The new delta verifier may change only the literal v3.8
preregistration/base pins, predecessor replacement data, exact
path/raw/symbol/import/evidence metadata, predecessor labels, and cross-anchor
metadata needed to enforce this document. Excluding its self-referential
`assignment|$module._COUNTERPART_RULE_DATA|1`, its admitted normalized
assignment changes are exactly:

- `assignment|$module.IMPLEMENTATION_BASE_COMMIT|1`;
- `assignment|$module.IMPLEMENTATION_BASE_TREE|1`;
- `assignment|$module.PREREG_COMMIT|1`;
- `assignment|$module.PREREG_DOC_BLOB|1`;
- `assignment|$module.PREREG_DOC_SHA256|1`;
- `assignment|$module.PREREG_TREE|1`;
- `assignment|$module._COUNTERPART_MECHANICAL_REPLACEMENTS|1`;
- `assignment|$module._DEFAULT_EXISTING_PATH_RULES|1`;
- `assignment|$module._V36_CLONE_PATH_RULES|1`; and
- `assignment|$module._V37_CLONE_PATH_RULES|1`.

No delta-checker function, class, import, or `build_delta_manifest` body may
change. The exact twelve-path inventory, strict additions, direct-parent
checks, worktree blindness, raw framing, full position-free symbol evidence,
import inventories, physical hashes, semantic hashes, replacement counts, and
counterpart hashes remain enforced.

The cloned delta test may update only exact v3.8 predecessor,
preregistration, inventory, counterpart, pin, and cross-anchor fixtures and
add the two focused hostile functions:

- `test_default_checkpoint_binding_counterparts_are_exact_and_only_expected_symbols_change`;
- `test_checker_rejects_checkpoint_main_receipt_omission_substitution_and_indirection`.

The hostile proof must reject an omitted keyword, public v3.7 digest, literal
digest, `records[-1]`, `records[1]`, phase receipt, `.get(...)`, fallback,
`str(...)`, dynamic `**kwargs`, new helper, new import, optional source
parameter, inferred source value, weakened source equality, the old fake's
silent derivation, or any second production change. Granting a hostile
candidate's raw SHA-256 must still fail independent changed-symbol evidence.

The delta module and test use the inherited one-way cross-anchor. The test
stores the raw delta-module SHA-256 in `V38_DELTA_MODULE_SHA256`; the delta
module binds the test blob with only those 64 digest characters masked; and
canonical syntax plus every unmasked byte remains exact. The legacy manifest
field name `committed_v32_counterpart_comparisons` remains unchanged solely
for schema and inherited-test compatibility; its entries must truthfully bind
the v3.7 terminal counterparts.

The preregistration commit must have exactly one direct parent, the pinned
v3.7 terminal commit, and exactly one added document. The implementation
commit must have exactly one direct parent, the preregistration commit, and
exactly the twelve strict additions above. Neither may be a merge. Worktree
bytes never establish committed implementation authority.

## Fresh namespace and zero-effect gate

The fresh private checkpoint namespace is
`data/aapl_sec_gemma_lean_evidence_v3_8`; public evidence is under
`e/aapl_sec_gemma_lean_evidence_v3_8`. No v3.8 run artifact, marker, ledger,
journal, receipt, blob, manifest, parse receipt, checkpoint, or public
authority may exist before its defined step. Symlinks, hardlinks, reparse
points, copied predecessor schemas, renamed artifacts, and cross-version
resealing cannot satisfy freshness. The private SEC contact remains readable
only through the inherited ignored local configuration; public evidence may
contain at most its cryptographic fingerprint.

The v3.8 preflight requires a clean branch equal to
`origin/codex/aapl-sec-gemma-lean-evidence-v3-8`, this preregistration as an
ancestor, the exact pushed implementation commit/tree, raw worktree bytes
equal to committed source blobs, no active run, no prior v3.8 probe marker, a
safe writable ignored checkpoint root, and the unchanged private contact. It
must bind its inherited loaded-source closure and the committed delta
manifest. The later acquisition gate additionally binds every raw
production-module blob in its unchanged production closure.

The preflight may make exactly two loopback identity calls in order:
`GET /api/tags`, then `POST /api/show` for `gemma4:12b`. It makes no
`/api/version`, chat, generation, SEC, Yahoo, market, performance, broker, or
trade call. Intent, completion, and publish markers are durable and
non-retryable. Its public receipt must be the only tracked change in its own
commit and must be pushed before acquisition. The v3.8 identity probe may not
be rerun after any durable identity intent, including a probe failure, or
after success. A zero-effect local repository or private-contact gate failure
before any durable identity intent may be corrected and retried.

The production acquisition gate must revalidate that committed receipt,
implementation commit/tree, delta manifest, source inventory,
branch/upstream, clean worktree, and a v3.8-only private namespace containing
only its own authenticated state. It must fail closed without an SEC request
on any mismatch. A v3.8 role may be satisfied only by a fresh v3.8 intent,
response, transport receipt, blob, parse receipt, manifest, and seal.

V3.8 has one fresh namespace and one nonretryable acquisition attempt; this
does not require one process invocation. Clean continuation is allowed only
between fully sealed v3.8 roles after an authenticated clean close. An open
intent, unsealed response, failed role, terminal rejection, or crash debris
can never be adopted, resealed, resent, or resumed. No private v3.7 namespace
may be opened or referenced by a v3.8 production, test, or manual path.

The production entrypoint remains development-only; intermediate and final
calls reject. The modules expose no Yahoo, Gemma generation,
scientific-feature, portfolio/performance-ledger, broker, or trading path; the
frozen SEC dispatch ledger remains unchanged. A passing parser role or
in-memory finalization is not source authority.

After a complete development source pass, the generated single
`e/aapl_sec_gemma_lean_evidence_v3_8/development/source-authority-<sha256>.json`,
the redacted `DEVELOPMENT_ACQUISITION.json`, and the comparison update must be
committed and pushed without implementation or scientific-code changes. A
separate read-only gate must authenticate that pushed preservation commit,
allowlisted evidence-only delta, public authority, checkpoint, terminal pass,
and detached replay before inherited development science can open. A rejection
cannot yield authenticated source authority and is preserved instead. No
preflight or acquisition is rerun after terminal success or failure; a new
repair would require a separately preregistered successor.

## Required offline tests before implementation publication

Focused tests must prove at least:

- the exact committed v3.7 terminal base, document-only preregistration child,
  twelve-addition implementation child, counterpart mapping, mechanical
  replacement counts, path modes, and no existing-path modification;
- the production caller passes exactly
  `records[0].manifest["parse_receipt_sha256"]` under the exact call-site and
  symbol budget;
- the fake source requires the keyword, checks equality with the first
  manifest, and stores the supplied value rather than deriving it silently;
- complete fake acquisition, clean sealed-prefix continuation, full-size
  three-stage orchestrator rehearsal, checkpoint sealing, and detached replay
  all exercise the explicit binding;
- omission, wrong index, phase hash, literal, predecessor hash, `.get`,
  fallback, cast, helper, dynamic keyword expansion, optional source
  parameter, validator weakening, fake derivation, second production change,
  import change, or raw-byte drift fails committed-blob proof;
- source, calendar, parser, acceptance, form, envelope, byte, topology,
  missing-filename, caps, journal, privacy, transport, crash, timeout,
  checkpoint schema, replay, and compatibility behavior remains the exact
  mechanical v3.7 behavior;
- both frozen calendars, ordered 504-session suffix, censor boundary, strict
  next-session rule, stage assignments, development `D=72-80`, intermediate
  `D=19-20`, final cap, completed-year coverage, and source projection remain
  unchanged;
- no source parameter, equality check, first-role check, checkpoint field,
  source seal, authority schema, stage gate, science contract, model, market,
  learner, ledger, performance, or authorization path is weakened;
- no test or implementation reads or parses any saved private v3.7 response,
  receipt, manifest, journal, checkpoint, or diagnostic; and
- the fresh two-call preflight cannot call version, chat, generation, SEC,
  Yahoo, market, performance, broker, or trade paths.

Synthetic fixtures establish the general rule without adapting to unpublished
official rows. Public v3.7 `D=75`, forms, counts, and hashes are diagnosis only
and may not become test constants except for explicit immutable predecessor
artifact/code pins and hostile rejection inputs.

## Hard falsifiers and scientific boundary

Before a complete development source seal, v3.8 rejects on any inherited v3.7
source falsifier plus any of these:

- the sole production correction is not the exact raw first-manifest binding
  at the existing checkpoint-builder call site;
- the caller uses a cast, fallback, helper, dynamic keyword mapping, phase
  hash, wrong record, literal, predecessor digest, or inferred value;
- the source builder, required parameter, first-role rule, canonical-hash
  rule, equality validation, checkpoint schema, source seal, authority, or
  detached replay changes or weakens;
- the fake source still derives the value independently or fails to require
  and verify the explicit caller binding;
- `d_min`, `d_max`, a model-call cap, stage window, global availability start,
  request cap, source-set cap, calendar rule, source rule, or science rule
  changes;
- a filing is sampled, trimmed, dropped, reordered, or reclassified using the
  public v3.7 `D=75`, count, form, date, accession, response hash, ordinal, or
  other predecessor-specific property;
- any private predecessor byte, receipt, journal, manifest, parse receipt,
  checkpoint, source authority, marker, or diagnostic is opened or reused;
- the preregistration is not one document-only child of the exact v3.7
  terminal commit, or the implementation is not one exact twelve-addition
  child of the preregistration;
- any implementation change falls outside the exact twelve-path,
  changed-symbol, changed-import, raw-hash, and cross-anchor budget;
- request effects are not reported as exact v3.8 counts plus the inherited 765
  family requests, or any request, byte, time, role, or intent cap is exceeded;
- the preflight is rerun after a durable identity intent, a role is retried,
  an unsealed response is adopted, or a terminal attempt is resumed; or
- any unplanned source, model, market, performance, paid-API, broker, or
  trading effect occurs.

If fresh official v3.8 data fails any source or checkpoint gate, v3.8 is
terminally rejected and preserved. It is not repaired or rerun during the
attempt. Any successor requires another separately committed and pushed
preregistration based only on public evidence, with science and later stages
still closed.

If the development source seal passes, source evidence is committed and
pushed first, then authenticated by a separate read-only gate. Only after that
gate may inherited development science run without modification. Confirmation
and final data remain closed until their unchanged gates pass. No historical
result can authorize real capital. Any historically promising survivor must
make append-only prospective paper-trading decisions before outcomes are
known; real-money trading remains outside this goal without fresh explicit
user authorization.
