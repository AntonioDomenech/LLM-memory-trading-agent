# AAPL SEC/Gemma lean science v3.22 pre-implementation rejection

## Verdict

**Engineering: fail. Learning value after costs: insufficient evidence.**

V3.22 is permanently rejected before an I322 implementation commit, official
one-shot preflight reservation, development attempt, or market/model access.
The mandatory pre-commit diagnostic batch executed 341 tests: 339 passed and
two failed. Both failures are `preflight_local_closure_hash_invalid`.

This is an implementation derivation omission. The V3.22 candidate retained
the V3.21 literal fingerprint for the shared-source closure manifest even
though the manifest's base commit, base tree, and schema version changed.
The fingerprint must authenticate those identity fields as well as the files.
All 37 shared-source rows are unchanged. This is not evidence of damaged SEC
filings, an installed-library mismatch, a failed trading strategy, or a
scientific result.

P322 explicitly requires all final identities to be regenerated. It also
requires a failed mandatory precheck to be preserved under the inherited
pre-implementation rejection topology. The missing regeneration was discovered
by that check after candidate fingerprints had been frozen. It is recorded as
a failure, not silently repaired and rerun as a passing V3.22 attempt. The
official reservation remains unused and grants no permission to revive this
rejected candidate.

## Immutable authority and source preservation

P322 is commit `42513ccca8afeb8a76d35b7f0e952d27a0a8e7ee`, tree
`2dae735a3b4f6cf50dcfda9d5c86853e4434a409`, sole parent
`8ca0d781f266def7ce6adced5d45431617817078`. Its only added path is
`docs/aapl_sec_gemma_lean_science_v3_22.md`, Git blob
`36f07b86b7b2bec13aae993be9f698f9286cd7e4`,
27,129 literal bytes, SHA-256
`eaa686102b0488f342f25a08549b1788ddd8644fee222e08f7856fbcc0a102c7`. P322 was authenticated against the live
GitHub branch before any V3.22 implementation path was created.

The twelve preserved V3.21 input files still total 1,898,729 bytes and retain
canonical manifest SHA-256
`fd49940c63bffd350aeb0ee6db5aa8934b34ac0c973cc43f7e12c3da88763a6b`.
Their original paths remain unchanged and untracked. V3.22 candidate paths
also remain untracked: this rejection does not invent an I321 or I322 commit.
The rejection commit adds only this document and appends one comparison row;
every tracked predecessor path other than that comparison remains unchanged.

## Exact failure

The failed nodes were:

- `tests/test_sec_gemma_lean_science_v322_preflight.py::test_local_production_closure_reproduces_exact_frozen_manifest`
- `tests/test_sec_gemma_lean_science_v322_preflight.py::test_pushed_execution_authority_replays_public_private_and_source_inventory`

Both reach the shared-source closure validator. The second failure occurs
while authenticating a synthetic pushed implementation; it stops before a
synthetic passing preflight can be created. It is the same underlying failure,
not a separate live execution or a second official attempt.

| Closure-manifest fact | V3.21 preserved input | V3.22 failed candidate |
|---|---|---|
| Shared-source rows | 37 | 37, byte-identical |
| Canonical manifest bytes | 8,756 | 8,756 |
| Base commit | `ecb5c809370e15b27f8eb67cfe644438d47c890f` | `8ca0d781f266def7ce6adced5d45431617817078` |
| Base tree | `f8171d8e116f7db8d6b717d95a263c9e9360a1fc` | `0a54c21c6793857f9a976dce62c5cc06bd5e6a28` |
| Candidate's expected SHA-256 | `54804244804b74c2fa185782c18b6e628f59849ae4bef4dc5d4e4e119084286d` | `54804244804b74c2fa185782c18b6e628f59849ae4bef4dc5d4e4e119084286d` |
| Independently recomputed SHA-256 | `54804244804b74c2fa185782c18b6e628f59849ae4bef4dc5d4e4e119084286d` | `8fabacc1b3529e14c6c5cde15509b50777ec6cc00b51b0f8789ffeb110e5e441` |

The independent read-only diagnosis parsed both contracts as data, read the
37 committed shared files at each pinned base, reconstructed the canonical
manifests, and compared every field. Exactly `base_commit`, `base_tree`, and
`schema` differ. It did not import predecessor production, patch validators,
change either candidate, open private filing source, or consume a reservation.
Its report SHA-256 is `750bef7f8d4499c13a768aef5f82ed19a3c629aab50c739b572f5b179bc5d3ec`.

## Checks completed and limits

- Source derivation: all twelve V3.21 input sizes and hashes, the aggregate
  manifest, P322 topology, and live-remote P322 identity matched.
- Runtime identities: exact Python 3.12.2 and pytest 9.0.3 executable/module
  identities passed the production qualification identity check.
- Collection: 343 unique nodes, no duplicates, all 42 required literal nodes
  present. Ordered collection SHA-256:
  `64f097a9cbb13a196344fad985129ae9f0441da0d3fb38f960f7c29bf0a72e31`.
- Diagnostic execution: 341 selected nodes, 339 passed, two failed, no skip,
  xfail, xpass, or collection error; pytest duration 456.56 seconds and measured
  enclosing child duration 457.032 seconds. These are diagnostic measurements,
  not an official qualification receipt or the full preflight duration.
- The real-source runtime node was deliberately scheduled separately and did
  not run after the mandatory failure. The real-HEAD import-closure node was
  omitted as P322 requires before I322 exists. Neither omission is a pass.
- Required node 42 passed. The isolated real-object probe observed 44
  repository modules, 366 dependency module-file rows, and 3,243 protected
  callable rows. The complete normalized V3.19 multiset after removal of the
  sole helper has 3,242 rows, 541,613 bytes, and SHA-256
  `8324901df84c3cfc6fb03e790f57bcd80baaffadf70f974a68c28961a0f913b7`.
  There is exactly one added helper and no removed callable. All six
  normalized production AST callable inventories match the preserved V3.21
  candidate inventories. The entire normalized recovery builder also matches.
- Direct recovery binding and the inherited synthetic recovery/crash tests
  passed. `authenticate_publication_recovery` is the exact dedicated recovery
  authority, with no recovery wrapper or new rejection helper.
- The short isolated probe proved repository, dependency, loaded-code, and
  process-object equality without running the real filing chain. It completed
  in 17.516 seconds. **It does not prove equality after filing authentication,
  projection, request construction, or model-plan construction.**

An initial diagnostic probe failed because newly written test instrumentation
looked for the locally imported recovery function as a runner module attribute.
Only that instrumentation was corrected before required node 42 was frozen;
both probe receipts are retained. No production behavior or frozen assertion
was changed. That instrumentation error is distinct from the later mandatory
closure-manifest failure, which remains unrepaired and terminal for V3.22.

The final production-bound helper row is frozen in required node 42. Its
owner-file SHA-256 is
`33f29d9ea7b7656c63b671dc91d4ab0e9f150b22b01d22d832a1da855f673456`;
owner-slots SHA-256 is
`2836a6cef7856ae54cf525472555d41703147404a7fa8df33447be03c7350ddd`;
code SHA-256 is
`913771c37a31ceb7322ec098f3212902098b27f069f92d63d448a982824e9b96`.
These are measured V3.22 values, not relabeled V3.21 values.

The full filing parser, acceptance timestamps, chronological production
source chain, real-HEAD/remote I322 proof, shared Phase 2, Requests Phase 3,
and actual whole-preflight runtime were not completed. The old approximately
50-minute estimate therefore remains an unverified planning estimate for this
candidate. The 4,800-second deadline was not allocated or consumed. There is
no F322, R322 trading result, pause, continuation, or later-stage authority.

## Research verdict and data restrictions

| Requested evidence | V3.22 observation |
|---|---|
| Net returns after 5/10 bps costs | Not evaluated |
| Buy-and-hold AAPL, fixed numerical baseline, frozen learning, filings without meaning | Required controls preserved; comparisons not evaluated |
| Maximum drawdown, worst trade/year, largest losses | Not evaluated |
| Simulated trades and independent decisions | 0 |
| New lessons admitted or decision changes attributed to lessons/meaning | 0 evaluated |
| Realized trading costs | No trades; performance cost comparison unavailable |
| Return uncertainty and dependence on individual trades/years | Not estimable with zero evaluated decisions |
| Paid API spending | $0.00 |
| New SEC/Yahoo/Ollama/Gemma transport or generation effects | 0 |
| Confirmation/live-style data opens, broker and real-money effects | 0 |

The existing local verified SEC archive is retained and was not opened or
modified by this attempt's real-source work, because that work never began.
The inherited 75-request cohort, sanitizer, source restrictions, fixed local
`gemma4:12b`, numerical baseline, causal lesson timing, four comparisons,
transaction costs, and all 22 development gates remain unchanged. The exact
38,320-byte scientific projection test passed.

Development remains 2000-2018; confirmation 2019-2023 and live-style 2024 onward
remain closed. Previously examined historical dates remain reused evidence,
never fresh holdout evidence. No future paper evaluation is preregistered or
opened because no qualifying historical result exists. Passing code checks
cannot establish investment value. This result makes no profitability claim
and authorizes no real-money trading.

## Reproducible local evidence

The failed twelve-file V3.22 candidate totals 1,910,330
literal bytes. Its canonical sorted compact JSON row manifest has 2,002
bytes and SHA-256 `319544b142aa61d87c94e6c23b824b2da24b4c1614caca5279351c3778f0cfb2`. Each row contains exactly
`byte_count`, `literal_sha256`, and `path`, with sorted keys, UTF-8, compact
comma/colon separators, and no trailing newline.

| Candidate path | Bytes | Literal SHA-256 |
|---|---:|---|
| `agent_benchmark/sec_gemma_lean_science_v322_bridge.py` | 93,155 | `3fcc9077d0d00e70d24bb6a76307d747ff7dc6510764d1f15e6f539086acee75` |
| `agent_benchmark/sec_gemma_lean_science_v322_contract.py` | 170,130 | `33f29d9ea7b7656c63b671dc91d4ab0e9f150b22b01d22d832a1da855f673456` |
| `agent_benchmark/sec_gemma_lean_science_v322_journal.py` | 21,869 | `f8ea5bc25e0f8ddb426c73157fd736572f8c8ee5e16f53d0a030c59b6f2b9e3c` |
| `agent_benchmark/sec_gemma_lean_science_v322_preflight.py` | 557,907 | `299ddb95ce0c855be609b2217b571302cf176e8e59885431b6ca500b78baaa08` |
| `agent_benchmark/sec_gemma_lean_science_v322_runner.py` | 227,239 | `4ff325d24230d35d0665c782d57a8d87ccda88c102b698d1ca2df4bdc97ced47` |
| `agent_benchmark/sec_gemma_lean_science_v322_store.py` | 192,331 | `072b6a7e2b0e037e6fcb638da39513ef49fe14a04073a7311674cfab5b8a656c` |
| `tests/test_sec_gemma_lean_science_v322_bridge.py` | 69,745 | `6da7112c21ba21bcd5417be65ad3f1f007d529ed776d9c17a13d1f32c66a3745` |
| `tests/test_sec_gemma_lean_science_v322_contract.py` | 71,856 | `afeeefeb76fcac34f7f23acd53355403ce81865acb3441e9a4dec9274358dd21` |
| `tests/test_sec_gemma_lean_science_v322_journal.py` | 8,463 | `7848f3829e1186437894ffdb384e82c1e592e7cff5957bad5c6ce07191f46f0a` |
| `tests/test_sec_gemma_lean_science_v322_preflight.py` | 236,075 | `a75d64779864686440a54f891cb3e21fe2630f8ede482a2234cc86accd0cde42` |
| `tests/test_sec_gemma_lean_science_v322_runner.py` | 198,964 | `485e428db6567ab03c3d89b56d580d4dd2b32dabe4379be74608ed1a850b03b8` |
| `tests/test_sec_gemma_lean_science_v322_store.py` | 62,596 | `e70ecd727a14a65c1097af3e7acc4b939d8f2a5fcb95dbb975f2742534118853` |

The ignored local diagnostic directory is `data/tmp/v322_diagnostics`.
It contains the exact scripts, original stdout/stderr, JUnit result,
collection, independent mismatch report, and immutable per-run receipts.
Raw diagnostic logs remain local; they are not public filing evidence.
The execution stdout has 4,829 bytes and SHA-256
`50caf9864de950f50241e549fb3ae4e4bd550503dbfd44c587b40250671915da`. Its JUnit SHA-256 is
`451c995aee8f6546c601b9386ea2c11cb43e5aea06043222b355424e35c90f0a`.

`V322_REJECTED_CANDIDATE.zip` in that directory contains the exact twelve
V3.21 sources, twelve V3.22 candidates, diagnostic scripts, and safe aggregate
receipts. It contains no filing archive, contact settings, market series,
model output, credentials, or raw test logs. The archive is 728,419
bytes, SHA-256 `789fe1c9b297804e1ba6d95361e6c88c8ca3e71c91bb4779060101423a9a3ba8`, and its entries were verified
against the candidate manifest. This local package is preservation, not an
implementation commit or execution authority.

Read-only diagnosis can be reproduced from this preserved workspace with:

```powershell
python -B data/tmp/v322_diagnostics/closure_mismatch_probe.py
```

The diagnostic test command used the exact 5,482-byte frozen qualification
bootstrap, cleared 16-name launch environment and six controlled pytest
environment additions. The retained `diagnose.py` uses those production
environment builders and the pinned interpreter; it does not call the
official reservation entry point. The failed node can be reproduced as a
local diagnostic with a new log name:

```powershell
python -B data/tmp/v322_diagnostics/diagnose.py nodes reproduce-closure-failure tests/test_sec_gemma_lean_science_v322_preflight.py::test_local_production_closure_reproduces_exact_frozen_manifest
```

Expected outcome: `preflight_local_closure_hash_invalid`. Diagnostic replay
does not revive V3.22 or permit its official preflight or development command.

## Bounded successor proposal; not authorization

A separately preregistered V3.23 may address only the incomplete derivation of
version-dependent authority fingerprints. It must start from this pushed
rejection, preserve both failed nodes and all rejected evidence, use its own
namespace and once-only attempt, and bind the same 37 shared files.

Before freezing its final production and test pins, derive the closure
manifest from its own exact base commit, tree, schema and committed file
rows; freeze its byte count and actual hash. Audit every other derived
identity for stale predecessor values, then regenerate the public contract,
scientific bootstrap, helper row and twelve-file inventory from final bytes.
The V3.22 observed hash above must not be relabeled as a V3.23 hash.

The successor must retain exactly 3,243 protected callables, the sole historical
helper, no removed callable, the direct recovery binding, `_strptime` preload,
all existing tests, source and timestamp restrictions, local-only inference,
zero paid spending, the one shared 4,800-second deadline, and every scientific
gate and comparison. Define any new engineering derivation checks in that new
preregistration before implementation or results. Complete the actual
filing/parser/source-chain and whole-runtime proofs before reserving its
official attempt. No change to dates, costs, model, policy, learner, control,
sample thresholds, or success gates is proposed. No V3.23 path or attempt is
created by this rejection.
