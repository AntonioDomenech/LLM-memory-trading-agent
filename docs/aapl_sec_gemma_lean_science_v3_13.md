# AAPL SEC/Gemma lean science v3.13

## Status and exact ancestry

This document preregisters V3.13 before any V3.13 implementation commit,
official qualification, one-shot preflight, SEC request, Yahoo request, Ollama
request, Gemma generation, market-value read, prediction, action, return, or
performance result.

The branch is `codex/aapl-sec-gemma-lean-science-v3-13`. Its immutable starting
authority is:

| Item | Frozen value |
|---|---|
| successor base commit | `342f8086bc9d9763d09a38c11ed6850da395ead5` |
| successor base tree | `4228ff17daca87b1d261c4e02b628607cba9a801` |
| successor base parent | `f9122a4db5b4cbe27516cd03a8a86933fe9c2472` |
| required preregistration parent | the successor base commit above |
| preregistration path | `docs/aapl_sec_gemma_lean_science_v3_13.md` |

This document must be the only changed path in the first V3.13 commit. That
document-only commit, `P313`, must have the successor base commit as its only
parent and must be pushed before any V3.13 implementation begins. Once pushed,
this document is immutable. A later correction requires another version,
branch, and preregistration. The pushed `P313` commit defines the exact
preregistration commit, tree, Git blob, literal SHA-256, and byte count. The
implementation must pin and authenticate those identities.

## Why V3.13 exists

V3.12 was permanently rejected before an implementation commit or official
one-shot preflight. Its public authorities are:

| Item | Frozen value |
|---|---|
| V3.12 preregistration commit | `f9122a4db5b4cbe27516cd03a8a86933fe9c2472` |
| V3.12 preregistration tree | `2136f781d804a9579ed78551d25d5d86c07a74d6` |
| V3.12 preregistration document | `docs/aapl_sec_gemma_lean_science_v3_12.md` |
| V3.12 preregistration document blob | `7195936f87b66e34fb7e4119d5b9d4f6f949df74` |
| V3.12 preregistration literal SHA-256 | `2bf92ba7f29836094c3204dd4a33f897eec316fa50272afab0ee8b98f15168ba` |
| V3.12 preregistration bytes | `39050` |
| V3.12 rejection commit | `342f8086bc9d9763d09a38c11ed6850da395ead5` |
| V3.12 rejection tree | `4228ff17daca87b1d261c4e02b628607cba9a801` |
| V3.12 rejection document | `docs/aapl_sec_gemma_lean_science_v3_12_rejection.md` |
| V3.12 rejection document blob | `34e11f81e512cd2b8fa7641ef03c767c11c683f5` |
| V3.12 rejection literal SHA-256 | `fd06a7a1bcc9753207c6d74fd9cdd4984a5e1a6b683fc1d2f833a7360956eff0` |
| V3.12 rejection bytes | `4849` |

The immutable V3.12 loaded-code rules cannot represent their own exact Python
3.12 process. In particular, they permit process identity sentinels and locks
only when held directly by module globals, yet one frozen shared module owns two
sentinels, two weak registries, and one lock through closure cells. A frozen
slots dataclass also keeps an original class and an exported replacement class
with the same natural module and qualified name. Accepting an unpreregistered
adapter after seeing those facts would have changed V3.12 after the fact, so
V3.12 was correctly rejected.

Read-only, network-trapped successor diagnostics found that the same problem is
general: generated dataclass methods, decorators, and weak-registry callbacks
can also produce distinct process identities with the same natural callable
name. V3.13 therefore uses one rooted owner-slot
algorithm for every reached callable. It does not use a list of callable names
as exceptions. The exact collision count is deliberately not a frozen gate;
the fixed graph and rules below determine the result.

The same diagnostics found four unique relative `Path` objects under five
module bindings, two exact immutable `typing` aliases under thirteen bindings,
and four exact pseudo-module aliases inserted by the Python standard library.
Those values are named explicitly below because silently accepting the broad
uncommitted V3.12 draft adapters is forbidden.

Twelve uncommitted V3.12 draft paths may exist locally while `P313` is created.
They are not an implementation, qualification, preflight, or scientific
result. They may be used as donor text only after `P313` is pushed, when the
exact V3.13 implementation paths are added in the one allowed implementation
commit.

## Exact inherited authority and narrow replacements

Except for the process replacements explicitly written in this document,
V3.13 incorporates every normative byte of the immutable V3.12
preregistration identified above. Through V3.12, it also incorporates the exact
V3.11, V3.10, and V3.9 authorities named there. Every scientific, source,
nullable-universe, body, content, proof, request, prompt, model, market,
feature, learner, policy, ledger, cost, gate, effect-budget, privacy,
publication, failure, recovery, and later-stage-lock rule remains normative.

The mechanical successor mapping is `V3.12` to `V3.13`, `v3.12` to `v3.13`,
`V3_12` to `V3_13`, `v3_12` to `v3_13`, `V3-12` to `V3-13`, `v3-12` to
`v3-13`, `V312` to `V313`, and `v312` to `v313`. It applies only to normative
successor branch, path, module, class, schema, attempt, artifact, command, and
Git-topology identities. It never rewrites quoted commits, trees, blobs,
literal hashes, byte counts, historical evidence, V3.12 rejection facts,
V3.8/V3.9 authority, or the two exact case-distinct Phase 2 node IDs inherited
from V3.12.

The following V3.12 provisions are replaced rather than inherited:

1. V3.12 branch, ancestry, preregistration, implementation, namespace,
   attempt, artifact, schema, command, and public/private successor identities;
2. V3.12 implementation paths, Phase 1 selector paths, minimum count, and
   required regression-node list;
3. the loaded-code manifest schema, callable-row schema, callable-reference
   key, callable discovery and qualification algorithm, callable ordering, and
   the old `python312_recursive_callable_namespace_v1` identity;
4. the module-global-only treatment of identity sentinels and synchronization
   objects, but only for the exact five closure-owned objects below;
5. unsupported-value handling only for the exact weak-registry self-reference,
   relative-Path, and typing-alias adapters below;
6. the execution-dependency manifest schema only to record the exact four-row
   pseudo-module normalization before the ordinary module snapshot;
7. runtime manifests and authority hashes, which must be derived afresh from
   the pushed V3.13 implementation and may never reuse a V3.12 draft value;
8. V3.12 Git topology, pending suffixes, comparison row, continuation path,
   and versioned safe-code identities; and
9. the literal qualification phase name `v312`, which becomes `v313` while the
   exact inherited 3,999-byte bootstrap remains unchanged.

There are no other replacements. If this document is silent, the exact V3.12
rule controls after the mechanical mapping. If an explicit replacement
conflicts with incorporated text, only that narrow replacement controls; all
unaffected surrounding clauses remain in force. Weakening an inherited rule by
omission, inference, a donor implementation, or a diagnostic observation is
forbidden.

## Frozen science and chronological boundary

V3.13 changes no scientific hypothesis, data row, filing byte, body, prompt,
model, feature, market value, threshold, action rule, cost, gate, stage, or
effect budget.

The authenticated source remains all 75 V3.8 development filings and the exact
selected embedded TEXT bytes. The nullable source boundary remains exactly 73
real filename/official-primary-URL pairs and two honest null/null pairs. The
frozen 12-key scientific-contract projection remains exactly 38,320 bytes with
SHA-256
`1ee05d2916752752bbef3710d70c7dab664fb82b9ac22829dac8897401058609`.
The source-derived 75-row private projection remains separate and its hash is
reproduced only after `P313` is pushed.

Only the literal `development` experiment is authorized. The market window is
1998-01-01 through 2019-01-01 exclusive, with 2000-2004 used only for warm-up
and learning and the five scored blocks 2005-2007, 2008-2010, 2011-2013,
2014-2016, and 2017-2018. Confirmation 2019-2023 and live-style 2024 onward
remain unreachable unless every prior frozen gate passes and a new later-stage
preregistration is committed, pushed, and authenticated.

Every policy remains long AAPL or cash only. Requested and realized exposure
must remain in `[0,1]`; cash may never be negative. Shorting, leverage,
borrowing, margin, cash interest, paid APIs, broker calls, and real-money
execution remain forbidden. V3.13 and AAPL buy-and-hold use the same starting
cash, dates, adjusted-price source, next-open fills, corporate-action treatment,
ledger, valuation date, and full precision. Costs remain 5 basis points per
changing leg with a 10-basis-point stress result. Every inherited development
gate must pass.

No historical strategy version is executed. Phase 2 and Phase 3 use older-named
test files only as exact offline component checks required by the latest V3.13
system. They do not run V3.8, V3.9, V3.10, V3.11, or V3.12 experiments or
reinterpret their outcomes.

## Exact twelve-file implementation boundary

After pushed `P313` exists, its implementation child `I313` must have `P313` as
its only parent and add exactly these twelve paths:

1. `agent_benchmark/sec_gemma_lean_science_v313_contract.py`
2. `agent_benchmark/sec_gemma_lean_science_v313_bridge.py`
3. `agent_benchmark/sec_gemma_lean_science_v313_journal.py`
4. `agent_benchmark/sec_gemma_lean_science_v313_store.py`
5. `agent_benchmark/sec_gemma_lean_science_v313_preflight.py`
6. `agent_benchmark/sec_gemma_lean_science_v313_runner.py`
7. `tests/test_sec_gemma_lean_science_v313_contract.py`
8. `tests/test_sec_gemma_lean_science_v313_bridge.py`
9. `tests/test_sec_gemma_lean_science_v313_journal.py`
10. `tests/test_sec_gemma_lean_science_v313_store.py`
11. `tests/test_sec_gemma_lean_science_v313_preflight.py`
12. `tests/test_sec_gemma_lean_science_v313_runner.py`

No tracked predecessor path may change in `I313`. Before official preflight,
the local branch, upstream, remote branch, commit, tree, sole parent, exact
twelve added paths, Git blobs, working-file literal hashes and byte counts,
predecessor equality, and clean worktree must authenticate. No V3.13 production
or Phase 1 test module may import or execute a V3.12/V3.11/V3.10 production
module, test module, or historical strategy entry point. The six V3.13 modules
must own successor behavior and may import only one another, the exact inherited
37-module shared closure, standard-library modules, and the exact qualified
third-party runtime.

## Exact qualification launcher and latest-only phases

Every collection and execution child uses the exact V3.12 qualification-only
bootstrap bytes and rules after the V3.13 mapping. The frozen bootstrap remains
3,999 UTF-8 bytes with literal SHA-256
`5fa1720bdf18e3f32be0ccb87997441d76b91e2381efb3b47a9598acd3263833`.
The authenticated Python remains CPython 3.12.2 with `-s -S -B`; the exact
qualification environment, pytest 9.0.3 identity, `sys.path` construction,
outer argv grammar, JUnit partial paths, collection/execution parity, marker-last
receipts, and 25-minute total deadline remain unchanged. The qualification
phase directory name is exactly `v313`, not `v312`.

Qualification has exactly three fresh-process phases. No repository-wide suite,
historical CLI, old experiment, or external effect is permitted.

### Phase 1: V3.13 only

Run every collected test in the six exact V3.13 test files. Phase 1 must collect
at least 310 case-sensitive unique nodes and every node must pass. It inherits
all V3.12 production-shaped coverage and must additionally collect and pass
these twenty-one exact nodes:

1. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_phase2_case_sensitive_multiplicity_is_655_unique_zero_duplicates`
2. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_phase2_rejects_case_insensitive_node_id_collapsing`
3. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_isolated_qualifier_bootstrap_matches_frozen_literal_and_imports_pinned_pytest`
4. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_authority_creation_precedes_authority_consumption`
5. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_authority_creation_never_requires_f_or_pushed_authority`
6. `tests/test_sec_gemma_lean_science_v313_runner.py::test_development_consumes_only_pushed_gated_authority`
7. `tests/test_sec_gemma_lean_science_v313_runner.py::test_continuation_authorized_precedes_attempt_open_binding`
8. `tests/test_sec_gemma_lean_science_v313_runner.py::test_continuation_attempt_open_precedes_model_continuation_pre`
9. `tests/test_sec_gemma_lean_science_v313_runner.py::test_model_continuation_pre_precedes_first_ollama_intent`
10. `tests/test_sec_gemma_lean_science_v313_store.py::test_recovery_api_is_reachable_end_to_end_with_ordered_demotion_and_verification_only`
11. `tests/test_sec_gemma_lean_science_v313_store.py::test_nested_mappingproxytype_evidence_is_recursively_thawed_and_canonicalized`
12. `tests/test_sec_gemma_lean_science_v313_runner.py::test_forbidden_old_validator_sentinel_is_retained_and_triggered`
13. `tests/test_sec_gemma_lean_science_v313_runner.py::test_pushed_result_gate_uses_candidate_material_schema_with_plan_equals_plan`
14. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_runtime_owner_graph_has_no_discovered_pseudo_slots_or_unrooted_callables`
15. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_runtime_owner_slots_disambiguate_every_same_natural_callable_identity`
16. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_runtime_closure_state_allowlist_binds_exact_five_objects_and_slots`
17. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_runtime_weak_registries_are_empty_callback_bound_and_identity_continuous`
18. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_runtime_relative_path_allowlist_is_exact_and_alias_bound`
19. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_runtime_typing_alias_allowlist_is_exact_and_identity_bound`
20. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_runtime_pseudo_module_normalization_requires_all_four_once`
21. `tests/test_sec_gemma_lean_science_v313_preflight.py::test_loaded_code_v2_callable_rows_bind_owner_slot_hashes`

Nodes 14-21 must exercise the production runtime graph/encoders, not detached
look-alike helpers. Phase 1 node IDs must be unique. A renamed, missing, skipped,
xfailed, xpassed, deselected, duplicated, failed, errored, or timed-out required
node rejects V3.13.

### Phase 2: exact shared dependencies

Run the same exact 14 ordered selectors inherited from V3.12. They must collect
and pass exactly 655 ordered nodes, 655 case-sensitive unique node IDs, zero
duplicates, and ordered node-list SHA-256
`1de8cc183af68c089aa5616f5e51405abda34382ff15e73123f5cd3c0dfea83c`.
The two case-distinct transport node IDs remain distinct and each occurs once.
The exact inherited 37-path shared closure and its ordered 2,023-byte path-list
SHA-256
`0a1e7374b35677596418cfa82575bd6614df631eba34f1978ce9c9017a74aae9`
remain unchanged. Its revision-owned manifest is recomputed for V3.13 and may
not adopt a predecessor manifest hash.

### Phase 3: isolated Requests identity

Run exactly
`tests/test_sec_gemma_lean_runner.py::test_runtime_modules_share_exact_verified_requests_identity`.
It must collect and pass exactly one unique node with ordered node-list SHA-256
`465bbb7fb1bd0633502006db2b84f6adab6ca7542d8c4e3d75b13d6cb7e73229`.

All phases require exit zero, exact collection/execution parity, marker-last
private receipts, and zero SEC, Yahoo, Ollama, Gemma, paid, broker, market,
performance, confirmation, live-style, or real-money effects.

## Loaded-code v2 rooted owner graph

The V3.13 loaded-code manifest schema is
`aapl-sec-gemma-lean-science-v3-13-loaded-code-manifest-v2`. Its top-level
fields remain the inherited fields. Each callable row has the thirteen
inherited fields plus exactly `owner_slots_sha256`. In this schema,
`qualified_name` is the callable's exact natural Python `__qualname__`; it is
not rewritten with an address, ordinal, or synthetic suffix. A callable
reference and row key is exactly `(owner_module, qualified_name, kind,
owner_slots_sha256)`. Callable rows sort by those four UTF-8 fields in that
order, and no two distinct identities may share the same four-field key.

The fingerprint algorithm identity is exactly
`python312_recursive_callable_namespace_v2`. It retains the inherited tagged
value and code encodings except where this document explicitly replaces them.
It uses a complete rooted owner graph before it serializes any callable row or
reference.

### Roots, edges, paths, and owner slots

Ordinary roots are the exact retained namespace bindings produced by the
inherited 44-module repository namespace scan: every repository-defined
module-global function/class plus every defining-module global named by the
recursively nested code objects that the scan retains. The only additional
roots are the exact relative-Path and typing module-global adapter roots listed
below. Root names sort by canonical UTF-8 JSON bytes and all ordinary and
adapter roots are seeded before traversal. A value that is not reachable from
this complete root set or an exact adapter edge below cannot enter a row.
Every adapter root also becomes an exact `semantic_value` entry in its owning
repository module's namespace array, so its binding hash contributes to that
module's `namespace_sha256`; adapter roots are never graph-only side data.

Traversal retains a strong reference to every reached object. `id` and `is`
may be used only transiently to recognize aliases and to reject ID reuse while
the strong reference exists. No ID, address, `repr`, discovery ordinal, or
`discovered:` pseudo-slot is serialized.

Every non-root graph edge is one exact `[EDGE_KIND, EDGE_NAME]` pair from this
table. Numeric indices are base-10 ASCII with no leading zero except `0`.

| Reached relationship | `EDGE_KIND` | Exact `EDGE_NAME` |
|---|---|---|
| repository class base | `class_base` | base index |
| ordinary method | `class_method` | exact class attribute name |
| static method | `class_staticmethod` | exact class attribute name |
| class method | `class_classmethod` | exact class attribute name |
| property getter | `class_property_fget` | exact property attribute name |
| property setter | `class_property_fset` | exact property attribute name |
| property deleter | `class_property_fdel` | exact property attribute name |
| nested repository class | `class_nested_class` | exact class attribute name |
| repository class annotation | `class_annotation` | exact annotation name |
| positional default | `function_default` | tuple index |
| keyword default | `function_kwdefault` | exact keyword name |
| annotation | `function_annotation` | exact parameter name or `return` |
| closure cell | `function_closure` | exact `co_freevars` name |
| repository referenced global | `function_referenced_global` | exact global name |
| list item | `list_item` | list index |
| tuple item | `tuple_item` | tuple index |
| mapping value | `mapping_value` | owner-free key SHA-256 |
| dataclass instance field | `dataclass_field` | exact field name |
| dataclass instance type | `dataclass_type` | literal `type` |
| enum instance type | `enum_type` | literal `type` |
| weak registry callback | `weak_registry_remove` | literal `_remove` |
| callback self weakref | `weak_registry_self_ref` | literal `referent` |
| relative Path concrete type | `relative_path_type` | literal `pathlib.WindowsPath` |
| typing alias concrete type | `typing_alias_type` | literal `typing._SpecialGenericAlias` |
| typing alias origin | `typing_alias_origin` | exact `_name`, `Mapping` or `Sequence` |

The callback's sole weakref is itself reached through the ordinary
`function_default` edge at index `0`; `weak_registry_self_ref` is the one
permitted back-edge from that weakref to its owning registry. Any adapter or
ordinary edge kind/name not defined in this table is terminal.

When an identity-bearing mapping value needs an owner edge, its edge token uses
the bare SHA-256 of an owner-free tagged key encoding, not the callable
fingerprint encoder. The owner-free encoder permits only exact null, Boolean,
integer, canonical float/complex, string, bytes, range, slice, and the inherited
acyclic date/time scalar tags; it rejects every callable, module, stateful
object, weak reference, Path, typing value, and container. A mapping whose keys
and values contain no graph identity retains the inherited ordinary value
encoding and needs no owner edge.
An identity-bearing mapping key or identity-bearing set/frozenset element is
terminal because it has no position independent of its own unfinished owner
identity. This prevents a circular owner token. All such tokens remain private
hashes and never expose a readable body, path, secret, or object representation.
External callable global graphs remain untraversed exactly as in V3.12. Their
own code, defaults, keyword defaults, annotations, and closure values remain
fingerprinted.

A root owner path is the canonical JSON edge array
`[["module_global", MODULE_NAME, GLOBAL_NAME]]`. A child path appends its exact
`[EDGE_KIND, EDGE_NAME]` pair. All candidate paths are processed by one priority
queue ordered first by edge count and then by canonical UTF-8 JSON bytes. All
roots enter the queue before the first pop. The first path that settles a
strong-held object identity is its canonical owner path; later aliases record
incoming edges but do not change that settled path. After every reachable
identity is settled, all callable-to-callable cycle and incoming edges are
recorded through the settled holder-path hash and are never recursively
inlined. An unresolved edge, unsupported cycle, or reached object without a
settled root path is terminal.

After canonical paths are fixed, a second pass records every incoming slot. A
root slot is exactly `["root", MODULE_NAME, "module_global", GLOBAL_NAME]`. A
non-root slot is exactly `["edge", HOLDER_CANONICAL_PATH_SHA256, EDGE_KIND,
EDGE_NAME]`. An object's owner-slot array is duplicate-free and sorted by each
slot's canonical UTF-8 JSON bytes. `owner_slots_sha256` is the bare SHA-256 of
that canonical JSON array and must be a non-null lowercase 64-hex value on every
callable row and reference. The complete slot arrays remain only in the strong
nonserialized in-process identity baseline; the durable manifest stores their
hashes. Each checkpoint regenerates the arrays and hashes before comparing the
loaded-code manifest and baseline. A missing slot, extra slot, alias change, two
identities with the same four-field callable key, changed root, unstable owner
set, unresolved reference, or replacement object is terminal.

The graph is built to a fixed point before any callable encoding. A callable
encountered later by value encoding that has no already-sealed owner graph row
is terminal; the implementation may not invent a self-described or
discovery-order slot for it.

### Exact closure-owned process state

All inherited module-global sentinel, lock, and RLock rules remain unchanged.
The only additional closure-owned stateful values are the following exact five
unique identities in
`agent_benchmark.sec_filing_gemma_market_acquirer`:

| Value | Exact closure owner slots |
|---|---|
| `issuer`, exact `builtins.object` | `issue:issuer`; `OwnedDevelopmentMarketAcquisition.__init__:issuer`; `OwnedMarketTransportCapability.__init__:issuer` |
| `missing`, exact `builtins.object` | `unwrap:missing`; `bind_to_claim:missing`; `consume:missing` |
| `acquisition_results`, exact `weakref.WeakKeyDictionary` | `issue:acquisition_results`; `unwrap:acquisition_results` |
| `capability_bindings`, exact `weakref.WeakKeyDictionary` | `issue:capability_bindings`; `bind_to_claim:capability_bindings`; `consume:capability_bindings` |
| `registry_lock`, exact `_thread.lock` | `issue:registry_lock`; `unwrap:registry_lock`; `bind_to_claim:registry_lock`; `consume:registry_lock` |

The short holder names in the table mean the exact natural qualified names
under `_build_owned_transport_capability_boundary.<locals>`. No other short-name
matching is allowed. The two sentinels are distinct from each other; the two
registries are distinct from each other; every alias in the table must point to
the one named identity. All five exact objects, complete owner-slot arrays, and
process identities enter a strong nonserialized baseline. Any additional
closure sentinel, registry, lock/RLock, cross-module alias, owner-set change,
replacement, cycle, or unsupported stateful value is terminal.

The closure lock must be exact `_thread.lock` and satisfy `locked() is False` at
authority creation and every checkpoint. The check occurs only on the sole main
thread before external/body work at that checkpoint. The inherited module-global
RLock probe is not broadened to closure state.

### Exact weak-registry representation

Both permitted registries are code-bound capability registries, not scientific
data. At authority creation and every V3.13 checkpoint each must satisfy all of:

- exact type `weakref.WeakKeyDictionary` and `len(value) == 0`;
- exact `vars` key set `data`, `_pending_removals`, `_iterating`, `_dirty_len`,
  and `_remove`;
- `data == {}`, `_pending_removals == []`, `_iterating == set()`, and
  `_dirty_len is False`;
- `_remove` is an exact distinct Python function with module `weakref`, natural
  qualified name `WeakKeyDictionary.__init__.<locals>.remove`, no closure,
  null keyword defaults, empty annotations, and exactly one positional default;
  and
- that default is an exact live `weakref.ReferenceType` whose referent is the
  owning registry and whose `__callback__` is null.

The registry adapter encodes exact type, owner-slot hash, weakref module origin,
callback code/default identity, and the sole tagged
`weak_registry_self_ref` cycle. Both callbacks and self-reference objects enter
the strong process-local baseline. Registry entries are never serialized,
ignored, or treated as scientific state. Any entry, dead or foreign weakref,
extra default, closure, pending removal, iteration state, attribute change,
callback replacement, owner change, or identity change is terminal. Every
other weak reference or weak-registry cycle is unsupported and terminal.

### Exact relative-Path allowlist

The only permitted relative `Path` bindings are these five bindings, which
represent four unique objects of exact concrete type `pathlib.WindowsPath`:

| Module and global | Exact POSIX value |
|---|---|
| `agent_benchmark.sec_gemma_online_risk_overlay_runtime.MANIFEST_RELATIVE_PATH` | `manifests/registry.ollama.ai/library/gemma4/12b` |
| `agent_benchmark.sec_gemma_online_risk_overlay_store.STATE_RELATIVE_DIRECTORY` | `data/sec_gemma_online_risk_overlay_v2_2/64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d` |
| `agent_benchmark.sec_gemma_online_risk_overlay_store.ANCHOR_RELATIVE_DIRECTORY` | `data/sec_gemma_online_risk_overlay_v2_2_anchors` |
| `agent_benchmark.sec_gemma_online_risk_overlay_vault.PRODUCTION_VAULT_RELATIVE_PATH` | `data/sec_gemma_online_risk_overlay_v2_2/64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d/quarantine.sqlite3` |
| `agent_benchmark.sec_gemma_online_risk_overlay_vault.STATE_RELATIVE_DIRECTORY` | exact identity alias of the store binding above |

They encode exact concrete type, canonical binding/alias set, and the POSIX
component array. Drive, root, and anchor must be empty; parts must be nonempty;
and no part may be empty, `.` or `..`. No resolve, current directory,
existence check, filesystem read, or absolute private-path token participates.
Every absolute `Path` retains the inherited strict existing private-path hash
adapter. Any other relative `Path`, subclass, value, binding, or alias change is
terminal.

### Exact typing-alias allowlist

The only permitted `typing._SpecialGenericAlias` identities are exact
`typing.Mapping` and exact `typing.Sequence`. The adapter explicitly seeds these
thirteen module-global roots even when an individual name is not retained by
the ordinary `co_names` namespace scan:

- Mapping in exact modules
  `agent_benchmark.sec_audit_transport`,
  `agent_benchmark.sec_filing_content`,
  `agent_benchmark.sec_gemma_lean_v38_source`,
  `agent_benchmark.sec_gemma_online_risk_overlay_attempt`,
  `agent_benchmark.sec_gemma_online_risk_overlay_publisher`,
  `agent_benchmark.sec_gemma_online_risk_overlay_registry`,
  `agent_benchmark.sec_gemma_online_risk_overlay_source_verifier`, and
  `agent_benchmark.sec_point_in_time`; and
- Sequence in exact modules
  `agent_benchmark.sec_filing_content`,
  `agent_benchmark.sec_gemma_lean_v38_source`,
  `agent_benchmark.sec_gemma_online_risk_overlay_attempt`,
  `agent_benchmark.sec_point_in_time`, and
  `agent_benchmark.sec_session_calendar`.

All eight Mapping bindings must be the exact one process identity
`typing.Mapping`; all five Sequence bindings must be exact one process identity
`typing.Sequence`. For each, `type(value) is typing._SpecialGenericAlias` and
`vars(value)` has exactly `_inst`, `_name`, `__origin__`, `__slots__`,
`_nparams`, and `__doc__`. Mapping has exact `_inst is True`, `_name ==
"Mapping"`, `_nparams == 2`, `__slots__ is None`, and `__origin__ is
collections.abc.Mapping`. Sequence has exact `_inst is True`, `_name ==
"Sequence"`, `_nparams == 1`, `__slots__ is None`, and `__origin__ is
collections.abc.Sequence`. Each `__doc__` must be exact `str` and is bound by
its SHA-256 rather than copied into a public artifact. The adapter binds every
exact field, complete owner-slot set, and exact origin callable reference. Both
identities enter the strong process baseline. Any other typing alias, type,
state, root, binding count, owner set, replacement, or identity is terminal.

### Exact pseudo-module normalization

The V3.13 execution-dependency manifest schema is
`aapl-sec-gemma-lean-science-v3-13-execution-dependency-manifest-v2`. It adds
exactly one top-level field, `normalized_pseudo_module_aliases`, to the inherited
schema. The inherited five-field `counts` object is unchanged. The new value is
a UTF-8-`alias_name`-sorted four-row array. Each row has exactly
`alias_name`, `owner_module`, `owner_attribute`, `object_kind`, `object_type`,
`metadata_sha256`, `owner_origin_sha256`, and `row_sha256`. `row_sha256` is the
bare SHA-256 of the canonical JSON of the first seven fields.
`owner_origin_sha256` is the bare canonical SHA-256 of the owner module's exact
execution-dependency origin row. `metadata_sha256` is the bare canonical
SHA-256 of exactly `alias_name`, `object_name`, `spec_is_null`,
`package_is_null`, `loader_is_null`, and `file_attribute_present`; the last
four values must be `true`, `true`, `true`, and `false` respectively, and
`object_name` must equal `alias_name`.

Immediately after canonical preload and before the normative module, binary,
and identity snapshot, all four aliases must be present in `sys.modules`:

| Alias | Owner module | Attribute | Exact type |
|---|---|---|---|
| `pyexpat.errors` | `pyexpat` | `errors` | `object_kind="module"`, `object_type="builtins.module"` |
| `pyexpat.model` | `pyexpat` | `model` | `object_kind="module"`, `object_type="builtins.module"` |
| `typing.io` | `typing` | `io` | `object_kind="deprecated_type"`, `object_type="typing._DeprecatedType"` |
| `typing.re` | `typing` | `re` | `object_kind="deprecated_type"`, `object_type="typing._DeprecatedType"` |

For each row, `sys.modules[alias_name]` must be identical to
`getattr(sys.modules[owner_module], owner_attribute)`. Each object must have its
exact alias `__name__`, null `__spec__`, `__package__`, and `__loader__`, and no
`__file__`. After sealing the four rows and strong identities, remove the
aliases in UTF-8 name order; each `pop` must return the sealed identity. All
normative module snapshots occur after this one-time normalization, so
`__main__` remains the sole ordinary module-inventory exclusion. At every
checkpoint, the four `sys.modules` names remain absent and the four owner
attributes retain their sealed identities. Missing-at-first, partially present,
already absent, substituted, reappearing, or additional pseudo aliases are
terminal.

### Structural and identity checkpoint rules

Authority creation stores strong nonserialized identity baselines for all 44
repository modules, every retained namespace binding, every strongly retained
object reached in the complete rooted owner graph, every callable row, the five
closure-owned objects, both registry callbacks and self weakrefs, the four
relative Paths, the two typing aliases, the four pseudo-alias objects, and all
inherited critical transport objects.

Every checkpoint rebuilds the structural manifests from pushed authority and
compares every baseline identity. Same bytes with a replacement object still
fail. All callable owner-slot arrays, semantic adapter state, module names,
binary tokens, pseudo-alias absence, lock state, and registry emptiness must be
reproduced. A later import, binary expansion, object replacement, owner-slot
change, weak state, or structural mismatch follows the inherited terminal rule
for that checkpoint.

## Runtime-authority lifecycle and one-shot preflight

The exact V3.12 separation between authority creation and authority consumption
is inherited after mapping. At clean pushed `I313`, preflight creation
authenticates exactly
`342f8086bc9d9763d09a38c11ed6850da395ead5 -> P313 -> I313`, creates fresh V3.13
runtime/dependency/loaded-code-v2 manifests under fakes, runs the three offline
qualification phases, proves zero external effects, and writes private authority
marker-last. It must not require nonexistent `F313` or call the newly created
authority pushed.

The private namespace is `data/aapl_sec_gemma_lean_science_v3_13`. The fixed
attempt identity is `aapl-sec-gemma-lean-science-v3-13-development-001`. Public
paths are:

- `e/aapl_sec_gemma_lean_science_v3_13/DEVELOPMENT_PREFLIGHT.json`;
- `e/aapl_sec_gemma_lean_science_v3_13/DEVELOPMENT_PAUSE.json`;
- `e/aapl_sec_gemma_lean_science_v3_13/DEVELOPMENT_RESULT.json`;
- `e/APPROACH_COMPARISON.md`; and
- `docs/aapl_sec_gemma_lean_science_v3_13_continuation.md` only if a
  continuation is required.

The one-shot preflight consumption boundary, pre-trust retry rule,
qualification order, source replay, nullable projection, request sealing,
privacy scans, zero-effect proof, safe failure route, and marker-last public
artifact are exactly inherited. Its public artifact is the only changed path in
`F313`, whose sole parent is `I313`; `F313` must be pushed. Only a separate
read-only pushed-result gate may turn development authorization true.

Authority consumption is impossible until pushed `F313` and the separate gate
authenticate `P313 -> I313 -> F313`, the cached remote ref, public bytes,
private replay, authority hash, qualification receipts, zero effects, and clean
worktree. Development, continuation, and publication recovery reproduce and
compare pushed expectations; they never regenerate current values and adopt
them as authority.

## Continuation, recovery, and Git topology

All inherited continuation ordering, candidate, terminal publication, pause,
crash recovery, binding receipt, privacy, and failure invariants remain exact
after mapping. A continuation still requires a pushed pause commit, a separate
pushed continuation preregistration, and fresh explicit user permission before
`continuation_authorized -> attempt_open -> model_continuation_pre -> first
continuation intent`.

Let `P313` be the pushed document-only preregistration commit, `I313` its exact
twelve-addition implementation child, and `F313` its one-path preflight child.
Every arrow below is single-parent and every named commit must be pushed before
it authorizes the next step:

- normal result: `base -> P313 -> I313 -> F313 -> R313`;
- planned pause and result: `F313 -> S313 -> C313 -> R313`;
- preflight rejection: `F313 -> X313`.

`R313` adds only the terminal result and changes the comparison table. `S313`
adds only the pause artifact. `C313` adds only the continuation
preregistration. `X313` changes only the comparison table, creates no
development result, and authorizes no execution. The comparison table preserves
all earlier rows and appends exactly one V3.13 safe outcome row. V3.12 remains a
zero-effect preregistration rejection, never a strategy result. V3.13 can have
`X313` or `R313`, never both.

After any public result or rejection commit, the separate pushed-result gate
must authenticate branch/upstream, local and cached remote commit, tree, parent,
exact path delta, Git blobs, working bytes, private journal and recovery replay,
public hashes, comparison transition, effect counts, privacy scan, and clean
worktree. Local-only or unpushed evidence unlocks nothing.

## One V3.13 development experiment only

Only after pushed `P313`, pushed `I313`, passing pushed `F313`, and every
offline and pushed-result gate pass may the literal `development` command
start. Before it starts, report a simple runtime estimate, prove offline that
the six Yahoo requests can be constructed, and state clearly that only V3.13 is
running.

The one attempt reuses authenticated SEC evidence and therefore makes zero SEC
requests. A normal completed attempt has exactly six Yahoo requests, four local
Ollama identity requests, and 75 no-retry local Gemma generations. A correctly
paused and continued attempt has exactly eight identity requests; all other
successful counts are unchanged. The inherited 30-second Yahoo, 60-second
identity, and 900-second generation monotonic deadlines remain exact. Every
logical request has exactly one physical request; no retry, repair, model pull,
fallback model, paid service, broker, or real-money effect is allowed.

The five fixed pilots run first. The inherited 12-hour pause formula and strict
comparison remain unchanged. Every terminal path preserves exact available
private evidence and one redacted public outcome. Evaluation-complete paths
preserve predictions, actions, long-or-cash ledgers, 5/10-basis-point
buy-and-hold comparisons, diagnostics, no-leverage proofs, controls, and all
development gates. Earlier rejection or indeterminate paths never fabricate
science fields or claim performance.

A development pass only permits a separate confirmation preregistration. It is
not proof of reliable future profit. Confirmation 2019-2023, live-style 2024+,
and prospective paper trading remain separately gated. Real-money trading is
outside this goal.
