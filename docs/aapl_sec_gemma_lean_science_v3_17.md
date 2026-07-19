# AAPL SEC/Gemma lean science v3.17 preregistration

## Status and decision boundary

This document preregisters V3.17 before any V3.17 implementation commit,
qualification, one-shot preflight reservation, source replay under V3.17,
Yahoo request, Ollama request, Gemma generation, market-value read, prediction,
action, return, performance calculation, confirmation or live-style data open,
paid API, broker action, or real-money action.

V3.17 is a narrow validation-compatibility successor to the consumed
zero-effect V3.16 preflight rejection. It changes no source row, filing byte,
nullable-source rule, prompt, model, model context, feature, market series,
learner, policy, threshold, score, transaction cost, stage boundary,
development gate, or external-effect budget. It corrects only three proven
request-validation defects:

1. the hard privacy gate treats every undirected `stock/share/security/market
   value` phrase as a market outcome, even when it is ordinary accounting
   language;
2. the same gate treats a security word and a direction word anywhere in one
   sentence as related, even when they belong to different clauses; and
3. the outer canonical JSON request incorrectly reuses the 20,000-byte joined
   sentence-text limit even though JSON necessarily adds a fixed prompt,
   schema, field names, identifiers, delimiters, and escaping.

The official V3.16 failure occurred after all qualification phases, runtime
authority, authenticated source replay, and canonical preprocessing, but before
any model call. Privacy-safe replay assigned the failure only to a predeclared
plain-text market-context category. The raw-token, encoded-context,
structured-key, canonical-shape, sentence-order, prior-parity, date-residual,
and inherited-preprocessor-residual category counts were all zero.

The same zero-effect replay reconstructed all 75 frozen development requests.
It found exactly six V3.16 plain-text privacy false positives: four in the
predeclared `undirected_security_value_pair` category and two in the
predeclared `nonlocal_security_direction_pair` category. It also found 46
later-valid canonical wrappers above 20,000 bytes, with no request sent. These
aggregate labels and counts define the general correction; no request ordinal,
accession, CIK, filename, URL, filing text, lexical match value, date, form,
hash, or contact value is admitted to public V3.17 evidence.

V3.16 is permanently rejected. It may not be retried, resumed, rewritten,
force-replaced, or reinterpreted as a strategy result.

## Immutable base and predecessor authority

The V3.17 branch is
`codex/aapl-sec-gemma-lean-science-v3-17`. Its immutable successor base is the
pushed V3.16 comparison-only rejection commit
`7ff2c5a6915ad1684d803be909647f0d040f121f`, tree
`0ba867ba80f71d722e0e1182c30c8a84812706d9`, whose sole parent is pushed
`F316` commit `475084cb48ef081a3b10ac35a554d5e8ca408f7b`, tree
`910fa694aede2b27167af6adbbaa2e9c2522bf2c`.

The V3.16 public failure artifact is
`e/aapl_sec_gemma_lean_science_v3_16/DEVELOPMENT_PREFLIGHT.json`, Git blob
`c58a04277da9b35581de08086bb0c75df927595c`, 622 UTF-8 bytes, with literal
SHA-256
`8485edee1943b54f97ff263db0f6318ea8767b79eee12ae3e096dd5316a5d3fc`
and embedded self-hash
`e770a83bfcf4ebb2afa477e63e67812fe6d1480c08ce258bbf0b5e1c73e5e94e`.
It fixes `preflight_consumed=true`, `rerun_authorized=false`,
`development_authorized=false`, confirmation/final closed, and real money
unauthorized.

The immutable V3.16 implementation is commit
`14eea6f536e1fc93ea54dbe8fa6b436a3f462cf4`, tree
`8b0b6783aeeaa5d6ee7ec0264f79e29b2bc71ca5`, whose sole parent is V3.16
preregistration commit `2cb4abf35b13502953d4f0b502ab6fc5c638be58`, tree
`bac40934de7bd299590ac18ffd6a912e125b356f`. The V3.16 preregistration
document is Git blob `fac61696f74b15108d43ff0518ec7d3fbb5dce4c`, 30,557 bytes,
with literal SHA-256
`e4609a7063784323e94637008a2d034c5904f3f2acf113d629f684ac147c93ba`.

At the successor base, `e/APPROACH_COMPARISON.md` is Git blob
`bbe16d8fe516ed1d86dd5d2af027afd80145b595`, 35,283 bytes, with literal
SHA-256
`b34df02d6d9eeadaaf77133a966463cc8418266631d11e970ef04145611b9250`.

The V3.17 document-only preregistration commit is named `P317`. It must have
the V3.16 comparison-only rejection commit as its sole parent, add only this
document, and be pushed and live-remote authenticated before any V3.17
implementation path is created. Once pushed, `P317` is immutable.

## Exact inheritance and replacements

V3.17 incorporates every normative V3.16 preregistration provision after the
exact eight-form mechanical mapping `V3.16 -> V3.17`, `v3.16 -> v3.17`,
`V3_16 -> V3_17`, `v3_16 -> v3_17`, `V3-16 -> V3-17`,
`v3-16 -> v3-17`, `V316 -> V317`, and `v316 -> v317`, plus the topology
mapping `P316/I316/F316/R316/S316/C316/X316 ->
P317/I317/F317/R317/S317/C317/X317`, except for the explicit replacements in
this document.

The mapping applies to branch, module, class, function, schema, namespace,
attempt, artifact, pending suffix, phase, command, safe-code, comparison-row,
and Git-topology identities. It never rewrites quoted V3.16 commits, trees,
blobs, hashes, byte counts, failure facts, scientific authority, preserved
V3.8/V3.9 evidence, or the two exact case-distinct Phase 2 node IDs.

Only these V3.16 provisions are replaced:

1. successor base, branch, ancestry, preregistration, implementation,
   namespace, attempt, artifact, schema, and topology identities;
2. the six production and six test path names;
3. the Phase 1 selector paths, minimum node count, and required node list;
4. the one undirected-value alternative in the first hard market-context
   regex, exactly as specified below;
5. the clause-insensitive `security_direction` calculation, exactly as
   specified below;
6. the outer canonical-request byte cap and its manifest disclosure, exactly
   as specified below; and
7. revision-owned repository, dependency, loaded-code, runtime-authority, and
   public/private hashes derived afresh from pushed V3.17 bytes.

There are no other replacements. The exact controlled Windows qualification
environment, 5,482-byte bootstrap, literal 44-module preload, 45 literal import
calls, real-production-source self-audit, strict local-import scanner, runtime-
v2 authority, JUnit case mapping, exact count parity, long-duration grammar,
duplicate-summary rejection, and Requests-identity sentinel remain unchanged.
If this document is silent, the mapped V3.16 rule remains exact.

## Exact privacy correction

The first alternative of `_BLINDED_TEXT_MARKET_CONTEXT_RES` currently matches:

```text
(stock|share|security|market) + (price|prices|return|returns|performance|value|values)
```

V3.17 removes only `value|values` from that second group. The alternative must
continue to reject `price`, `prices`, `return`, `returns`, and `performance`.
Every other market-context regex, issuer/source/date/missingness rule, keyed
label, model-action rule, raw proof-token scan, encoded-context scan, canonical
sentence reconstruction, nested-JSON inspection, and inherited preprocessor
residual check remains byte-for-byte mapped V3.16 behavior.

The clause-insensitive bag-of-words `security_direction` Boolean is replaced
by one exact punctuation-preserving folded-text regex named
`_BLINDED_TEXT_SECURITY_DIRECTION_RE`. Its source is the concatenation of
these raw literals:

```python
r"(?<![a-z0-9])(?:"
r"(?:stock|stocks|security|securities|share|shares|market)"
r"(?:[ -]+(?:price|prices|value|values))?"
r"[ -]+"
r"(?:(?:closed|had|has|have|is|moved|moving|traded|was|were)[ -]+)?"
r"(?:(?:considerably|dramatically|far|markedly|materially|moderately|modestly|much|notably|quite|sharply|significantly|slightly|somewhat|steeply|substantially|very)[ -]+){0,2}"
r"(?:advanced|appreciated|down|dropped|fell|gained|higher|lost|lower|outperformed|rallied|rose|surged|underperformed|up)"
r"|"
r"(?:advanced|appreciated|down|dropped|fell|gained|higher|lost|lower|outperformed|rallied|rose|surged|underperformed|up)"
r"[ -]+(?:stock|stocks|security|securities|share|shares|market)"
r"(?:[ -]+(?:price|prices|value|values))?"
r")(?![a-z0-9])"
```

`security_direction` becomes exactly
`_BLINDED_TEXT_SECURITY_DIRECTION_RE.search(folded) is not None`, with the
existing `market share` and `market shares` exclusions retained. Only spaces
and lexical hyphens may connect the matched pieces; comma, semicolon, colon,
period, question mark, exclamation mark, parentheses, and other punctuation
cannot be erased into adjacency. The exact modifier allowlist permits up to two
local modifiers after an optional movement/linking verb. It does not permit an
arbitrary intervening word.

All examples in this section are synthetic tests, not preserved filing text.
The regex must reject at least `shares rose`, `shares moved very sharply lower`,
`market rose`, `security gained`, `stock was up`, `stock value rose`, `stock
prices were materially lower`, `higher share price`, and `lower market value`.
It must allow an undirected accounting use of `stock value`, a sentence where
`lower` describes a nonlocal business noun while `stock` occurs later, and the
punctuation-separated synthetic case `stock; lower business costs`.

The direct required test
`test_blinded_request_allows_undirected_accounting_value_and_rejects_local_market_direction`
must exercise both allowed classes and every listed forbidden control through
the real V3.17 validator. It may not monkeypatch a privacy helper.

## Exact canonical-request byte cap

`MODEL_INPUT_MAX_BYTES` remains exactly 20,000 and continues to cap the joined
canonical sentence text before the model payload is built. The 72-sentence
limit, 36-current/36-prior split, 220-character per-sentence limit, 6,144-token
model context, and 512-token output limit remain unchanged.

V3.17 adds `MODEL_REQUEST_MAX_BYTES = 128 * 1024`, exactly 131,072 bytes. Only
the outer `request_bytes` length check in `validate_blinded_model_request` uses
this new constant. The preregistration manifest retains the 20,000-byte
sentence-text cap and additionally publishes the 131,072-byte canonical-
request cap. No response, Yahoo, SEC, journal, database, worker, or model
context limit changes.

This is a general serializer bound, not a value selected from the observed
dataset. The frozen shape permits at most 72 five-character IDs and 72 texts
of 220 ASCII characters, so total text characters are at most 15,840 and the
joined text including 71 newlines is 15,911 bytes. Each allowed ASCII
character occupies at most seven bytes after the inner sentence JSON and outer
request JSON serializations. The exact fixed 72-row outer overhead with empty
text slots is 10,197 bytes. Therefore every exact canonical request is at most
`10,197 + 15,840 * 7 = 121,077` bytes. The 128-KiB cap provides exactly 9,995
bytes of headroom over that tight maximum and remains far below the inherited
64-MiB worker bound.

The direct required test
`test_canonical_request_cap_proves_tight_wrapper_maximum` must prove:

1. no ASCII character has a two-layer serialized contribution above seven;
2. a 36-current/36-prior payload containing 220 copies per sentence of a
   maximally escaped allowed ASCII control is exactly 121,077 bytes;
3. that payload passes exact sentence reconstruction, inherited residual and
   privacy validation, and the 131,072-byte request cap; and
4. a byte string of 131,073 bytes rejects before any request can be sent.

## Redacted all-request compatibility proof

Before `I317` is pushed and again before the official one-shot preflight is
reserved, a read-only local replay must authenticate the preserved V3.8 source,
reconstruct all 75 frozen development proofs and model requests, and pass each
request through both the real V3.17 contract validator and the inherited
production validator. It must report only fixed aggregate labels, integer
counts, Boolean outcomes, request-count 75, and public bounds. It must report
zero contract failures and zero inherited-production failures.

The replay may not print or persist request bytes, sentence text, identity
tokens, row IDs, accessions, CIKs, filenames, URLs, dates, forms, filing hashes,
contacts, or per-row lengths. It makes no SEC, Yahoo, Ollama, Gemma, market,
paid, broker, or real-money call and creates no execution authority. Failure
blocks `I317` push and official preflight.

## Frozen science and chronological boundary

The authenticated scientific authority remains the exact V3.16 authority and
the exact V3.8 preserved source: all 75 development filings, including exactly
73 real filename/official-primary-URL pairs and two honest null/null pairs. The
frozen 12-key scientific projection remains exactly 38,320 bytes with SHA-256
`1ee05d2916752752bbef3710d70c7dab664fb82b9ac22829dac8897401058609`.

Only the literal `development` experiment is authorized. The requested market
window remains 1998-01-01 through 2019-01-01 exclusive. Years 2000-2004 remain
warm-up/learning only. The five scored blocks remain 2005-2007, 2008-2010,
2011-2013, 2014-2016, and 2017-2018. Confirmation 2019-2023 and live-style
2024 onward remain unreachable unless every earlier gate passes and a new
later-stage preregistration is committed, pushed, and authenticated.

Every policy remains long AAPL or cash only. Exposure remains in `[0,1]` and
cash may never be negative. Shorting, leverage, borrowing, margin, cash
interest, paid APIs, broker calls, and real-money execution remain forbidden.
V3.17 and AAPL buy-and-hold use the same starting cash, dates, adjusted-price
source, next-open fills, corporate-action treatment, ledger, valuation date,
and full precision. Costs remain 5 basis points per changing leg with a
10-basis-point stress result. Every inherited development gate must pass.

No historical strategy version is executed. Older-named Phase 2 and Phase 3
test files are exact shared offline component checks for the latest V3.17
system; they do not run or reinterpret an older experiment.

## Exact twelve-file implementation boundary

After pushed `P317` exists, its implementation child `I317` must have `P317`
as its sole parent and add exactly these twelve paths:

1. `agent_benchmark/sec_gemma_lean_science_v317_contract.py`
2. `agent_benchmark/sec_gemma_lean_science_v317_bridge.py`
3. `agent_benchmark/sec_gemma_lean_science_v317_journal.py`
4. `agent_benchmark/sec_gemma_lean_science_v317_store.py`
5. `agent_benchmark/sec_gemma_lean_science_v317_preflight.py`
6. `agent_benchmark/sec_gemma_lean_science_v317_runner.py`
7. `tests/test_sec_gemma_lean_science_v317_contract.py`
8. `tests/test_sec_gemma_lean_science_v317_bridge.py`
9. `tests/test_sec_gemma_lean_science_v317_journal.py`
10. `tests/test_sec_gemma_lean_science_v317_store.py`
11. `tests/test_sec_gemma_lean_science_v317_preflight.py`
12. `tests/test_sec_gemma_lean_science_v317_runner.py`

No tracked predecessor path may change in `I317`. Before official preflight,
the local branch, upstream, cached and live remote branch, commit, tree, sole
parent, exact twelve additions, Git blobs, working literal hashes and byte
counts, predecessor equality, and clean worktree must authenticate.

No V3.17 production or Phase 1 test module may import or execute a V3.16,
V3.15, V3.14, V3.13, V3.12, V3.11, or V3.10 production/test module or a
historical strategy entry point. The six V3.17 production modules own successor
behavior and may import only one another, the exact inherited 37-module shared
closure, the package initializer, standard-library modules, and the exact
qualified third-party runtime.

The mapped literal preload contains the six V3.17 production modules followed
by the exact 37 shared modules and package initializer, for exactly 44 keys and
45 recognized literal dynamic-import calls including exact literal `numpy`.
The mapped real-source audit and exact required node
`test_real_v317_production_import_calls_are_literal_and_upper_bound_closes`
remain mandatory. No import alias, variable argument, computed string, scanner
exception, or scanner weakening is permitted.

## Exact qualification phases

The isolated launcher remains CPython 3.12.2 with `-s -S -B`, pytest 9.0.3,
the exact 5,482-byte bootstrap with SHA-256
`59aa7f29b7200e5a915b15a840da050e7803c825bf40ad42720dc0a4092e4407`,
the exact reduced 16-name launch environment followed by the controlled
six-name pytest additions, JUnit receipts, collection/execution parity,
marker-last durability, and 25-minute total deadline. The latest phase name is
exactly `v317`.

Phase 1 runs every collected node in the six exact V3.17 test files. It must
collect at least 327 case-sensitive unique nodes, zero duplicates, and every
node must pass. It includes the mapped 24 V3.16 required nodes plus exactly:

25. `tests/test_sec_gemma_lean_science_v317_bridge.py::test_blinded_request_allows_undirected_accounting_value_and_rejects_local_market_direction`
26. `tests/test_sec_gemma_lean_science_v317_bridge.py::test_canonical_request_cap_proves_tight_wrapper_maximum`

Phase 2 remains the exact 14 ordered shared selectors: 655 ordered nodes, 655
case-sensitive unique IDs, zero duplicates, and ordered node-list SHA-256
`1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c`.
Phase 3 remains only
`tests/test_sec_gemma_lean_runner.py::test_runtime_modules_share_exact_verified_requests_identity`,
one unique passing node with ordered node-list SHA-256
`465bbb7fb1bd0633502006db2b84f6adab6ca7542d8c4e3d75b13d6cb7e73229`.

All phases require exit zero, exact collection/execution parity, zero failures,
errors, skips, xfails, xpasses, duplicates, missing required nodes, and zero
SEC, Yahoo, Ollama, Gemma, paid, broker, market, performance, confirmation,
live-style, or real-money effects.

## Runtime authority and Git topology

At clean pushed `I317`, preflight creation authenticates exactly
`7ff2c5a6915ad1684d803be909647f0d040f121f -> P317 -> I317`, runs the three
offline qualification phases, creates fresh V3.17 repository/dependency/
loaded-code/runtime authority under fakes, proves zero external effects, replays
the preserved source and all-request compatibility proof, and writes private
authority marker-last. It must not require a nonexistent `F317` or consume
newly created authority.

The private namespace is `data/aapl_sec_gemma_lean_science_v3_17`. The fixed
attempt identity is `aapl-sec-gemma-lean-science-v3-17-development-001`.
Public paths are:

- `e/aapl_sec_gemma_lean_science_v3_17/DEVELOPMENT_PREFLIGHT.json`;
- `e/aapl_sec_gemma_lean_science_v3_17/DEVELOPMENT_PAUSE.json`;
- `e/aapl_sec_gemma_lean_science_v3_17/DEVELOPMENT_RESULT.json`;
- `e/APPROACH_COMPARISON.md`; and
- `docs/aapl_sec_gemma_lean_science_v3_17_continuation.md` only if required.

Every arrow is single-parent and every named commit is pushed before it
authorizes the next step:

- normal result: `base -> P317 -> I317 -> F317 -> R317`;
- planned pause and result: `F317 -> S317 -> C317 -> R317`;
- consumed preflight rejection: `F317 -> X317`.

`F317` adds only the public preflight artifact. `R317` adds only the terminal
result and changes the comparison table. `S317` adds only the pause artifact.
`C317` adds only the continuation document. `X317` changes only the comparison
table, creates no development result, and authorizes no execution. The
comparison table preserves every earlier row and appends exactly one V3.17 safe
outcome after V3.16.

After any public result or rejection commit, the separate pushed-result gate
authenticates branch/upstream, local/cached/live remote commit, tree, parent,
exact path delta, Git blobs, working bytes, private replay, public hashes,
comparison transition, effect counts, privacy, and clean worktree.

## One V3.17 development experiment only

Only after pushed `P317`, pushed and self-audited `I317`, passing pushed
`F317`, and every offline and pushed-result gate pass may the literal
`development` command start. Before it starts, report a simple runtime
estimate, prove offline that the six Yahoo requests can be constructed, and
state clearly that only V3.17 is running.

The one attempt reuses authenticated SEC evidence and therefore makes zero SEC
requests. A normal completed attempt has exactly six Yahoo requests, four local
Ollama identity requests, and 75 no-retry local Gemma generations. A correctly
paused and continued attempt has exactly eight identity requests. The inherited
30-second Yahoo, 60-second identity, and 900-second generation monotonic
deadlines remain exact. Every logical request has one physical request; no
retry, repair, pull, fallback, paid service, broker, or real-money effect is
allowed.

If any development gate fails, V3.17 is rejected and later stages remain
closed. Only if every development gate passes may a separately preregistered
confirmation stage open 2019-2023. Only if confirmation passes may a separately
preregistered live-style stage open 2024 onward. Historical inspection is never
prospective proof; reliable advantage claims require later prospective paper
trading.
