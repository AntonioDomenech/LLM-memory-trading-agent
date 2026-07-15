# AAPL causal contextual expert aggregation: 2024 audit v2

## Purpose and evidence boundary

This contract authorizes one chronological replay of calendar 2024 for the
fixed policy that earned historical-policy-candidate status in the locked
2019-2023 continuation audit. It does not reopen development, tune a threshold,
or promote the continual-learning mechanism that the parent audit did not
validate.

The lead arm is the exact verified `online_full` terminal state at
2023-12-29, frozen before 2024. A shadow arm starts from the identical model and
account state but continues to admit lessons causally during 2024. Shadow
learning can be measured, but it cannot rescue or redefine the lead policy's
result.

This is a post-hoc frozen-policy replication, not a new unseen-year test or
prospective proof. Other approaches in this repository have already been
scored on 2024, and their results and action artifacts are known outside this
audit. Those artifacts are forbidden inputs here. Credibility comes only from
forking an exact policy/model/account frozen at 2023, preregistering all gates,
and permitting no tuning or implementation oracle. The result authorizes no
real capital, broker execution, later value, news, LLM, Ollama, external model,
network request, or paid API.

## Frozen identity

- Contract version:
  `aapl-causal-contextual-expert-aggregation-2024-audit-v2`
- Branch:
  `codex/aapl-causal-contextual-expert-aggregation-2024-audit-v2`
- Stage: `audit_2024`
- Run ID:
  `contextual-expert-aggregation-frozen-policy-2024-audit-v2`
- Control root:
  `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2`
- Output parent:
  `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/runs`
- Durable lock:
  `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/AUDIT_2024_ATTEMPT_LOCK.json`
- Durable success marker:
  `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/AUDIT_2024_STAGE_SUCCESS.json`
- Pending success marker:
  `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/AUDIT_2024_STAGE_SUCCESS.pending`
- Final output directory:
  `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/runs/contextual-expert-aggregation-frozen-policy-2024-audit-v2`
- Private seal directory:
  `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/runs/.sealing-contextual-expert-aggregation-frozen-policy-2024-audit-v2`
- Consumed-attempt failure directory:
  `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/runs/.failed-contextual-expert-aggregation-frozen-policy-2024-audit-v2`
- Verifier ID:
  `contextual-expert-aggregation-2024-audit-verifier-v2`
- Stage attempt count: exactly one
- Strict stage internal final-commit sample: less than 1,800 seconds
- Strict externally measured complete stage-command wall time: less than 1,800
  seconds for the result to count toward the approach
- Strict standalone-verifier command wall time: less than 1,800 seconds
- Finalization reserve before promotion: 5 seconds
- Combined externally measured command wall times: less than 3,600 seconds,
  defined as the sum of the two independent command durations rather than
  elapsed wall time between invocations
- External cost: exactly $0.00

This contract, sanitized receipt, physically bounded input, and quarantined
preparation record must be committed and pushed before audit implementation
begins. Completed implementation and tests must be committed and pushed before
the one-shot stage command begins. Each monotonic runtime clock starts at
bootstrap entry and includes code attestation, imports, parent/input
validation, computation, sealing, and private semantic verification. Runtime
locks the exact pushed implementation commit and complete explicit dependency
identity. The internal stage metric ends at the explicitly named
`final_commit_sample`; the external command-wall requirement includes the
subsequent atomic marker tail and normal command return.

The next commit must be a data/control-only preregistration commit with parent
`e4783a93391f8b6ba632bcbff8192c554dd374e7` and exactly the five new paths
described in this preregistration. Implementation freezes that resulting commit
as `PREREGISTRATION_COMMIT`; stage proves it is an ancestor of pushed runtime
HEAD using commit metadata only and binds it in the attempt lock/manifest.

Only this contract is a runtime control document. The human-readable input
snapshot note and the full preparation record are documentary-only and must be
excluded from `FROZEN_DEPENDENCY_PATHS`, preimport attestation, stage, and
standalone verification.

## Exact parent trust root

The only authorized policy parent is the already standalone-verified bundle:

`e/aapl_causal_contextual_expert_aggregation_audit_v2/contextual-expert-aggregation-post-rejection-2019-2023-audit-v2`

Frozen parent identities are:

- preservation commit:
  `e4783a93391f8b6ba632bcbff8192c554dd374e7`;
- original run commit:
  `f2085fcf776d32a29918727ad8d9b64932cb17d9`;
- contract:
  `aapl-causal-contextual-expert-aggregation-2019-2023-audit-v2`;
- manifest file SHA-256:
  `fd4c7427bdc9ec2a5463dcc2781310aeab5c498836db0617c0c41ebf81267bbd`;
- manifest self SHA-256:
  `c71e201a8a3d1ef2b0ebe914d46abfdd0f5a9154d4f106ce723168a9bce009e8`;
- checksums file SHA-256:
  `4d7fd305fb0ae6c8584b993528366af433c5b3401455745446fe2e36c36085b6`;
- report file SHA-256:
  `5877434e4decb79f35a765793c9408638434d7ab1a590f29fe2f03da46522502`;
- gate-report file SHA-256:
  `2caf58663fbe6b5fdc255708e6cbb4c3b5ed8383fa487c7ae602e868c741d7a2`;
- metrics file SHA-256:
  `415b709a44a177706227162b5778ccfa80161ba5f4154bfd7d9579ad12359280`;
- integrity-evidence file SHA-256:
  `a4442f5d45bdcecf5f2ec32f356643593aed57ef91b554d61f97f05df45009f3`;
- checkpoint payload SHA-256:
  `560571500f580f9e0fd4319a93905b4ad1e498f1c1f27de9d647da455b6d5477`;
- checkpoint self SHA-256:
  `4aaf549c9352ac322f36c46cb8dcdd03f672a1eba580813ba8a1947d323a85eb`.

Within that composite checkpoint, the only selectable replay member is
`online_full`, with these frozen facts:

- replay-checkpoint digest:
  `349ca26a1694970a1b24762fc3c9dfe289c837128780503f07dbda126848f1b8`;
- checkpoint date and model-state last date: `2023-12-29`;
- source sessions and processed sessions: `6244`;
- runtime: learning mode `causal_online`, frozen cutoff `null`, ablation mode
  `full`;
- admitted/matured/pending lessons: `202/205/0`;
- model-state SHA-256:
  `781d9ae7f10335c7167b43b57f42f7c483052839b9f32da99e4a68fe286f502b`;
- pending-state SHA-256:
  `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`;
- 5-bps online account: 4,781 ledger rows, ledger tip
  `sha256:8c7b1c17ff2ea17086a09df78e342235ad1e2bd57fcbe05816350f317f01549a`,
  account-state SHA-256
  `sha256:ed9c7d4cb720ddf27c7db0429e66c16a9656dbc2561fca8a3ee41101eb056970`;
  and
- 10-bps online account: 4,781 ledger rows, ledger tip
  `sha256:05f10e984ae3f3044c36078fb4ea6dd1bee2f78e7db393ce2a478913e3c1b167`,
  account-state SHA-256
  `sha256:6ca2a8f11ab4ce875cfd10057e978de83d91f0d6401fbe1bf62f4d0bba962683`.

Every required comparator must continue its own distinct sealed account; none
may start from the lead, cash, fresh capital, or a reconstructed 2024 boundary.
All have last session `2023-12-29`, held target `1`, pending target `1`, and
4,781 ledger rows. Their exact cost-specific identities are:

| Cost | Policy | Account-state SHA-256 | Ledger-tip SHA-256 |
|---|---|---|---|
| 5 bps | `aapl_buy_hold` | `sha256:ae06a2f96d286e8a9a0c33db9aba22c58a9ef2fc15247498f71832d91c8e4cc6` | `sha256:fd6441546a54da6696ac9b553d9b8369878fb8e4abaf55ea67d024eff5ce5f3e` |
| 5 bps | `always_long` | `sha256:618ee307f83735f810ab47f547772ad7993bc2b621f9fc407fda7fa96f7fd5b5` | `sha256:a26e056036be39fca6b50267c25dc161490edfb3f0d1c2d57a317ed999164a1f` |
| 5 bps | `exact_union_cash` | `sha256:121e91e39e8ee9821455b2e31d9e079c604376914efb42e13dea1d95a394bc50` | `sha256:5b8d4d25dccbf9275be1308ab60a340d5925576df718156114c4a07e4285a5f4` |
| 5 bps | `contextual_only` | `sha256:d50ccb3be50d80310169eac244760057b3b77a3762fc6382676fd6db6f60fa30` | `sha256:cf293ad8bdfe5038e936c83b5a55bd5ef8d21191f0ff65d67a1ab1edc4aa1e89` |
| 10 bps | `aapl_buy_hold` | `sha256:35ccf533717a511fac464f6778d0a75cef64285f9d10c3cad2c54d5c14b98ba9` | `sha256:66b0f2dffc131c640f6f43780adecb79027f786608330c3db07616878bbaef75` |
| 10 bps | `always_long` | `sha256:0da8604bacdfceffab5a1d10255df15472512b51901e1ec536548ccc85cf82f1` | `sha256:f301f7b239e7d7e76f2ec94ac30e28897252a00100ecd6a886c7429228260266` |
| 10 bps | `exact_union_cash` | `sha256:92882101074b5e149c5023332ae250419e6516efb8c9893099ff069cbea50293` | `sha256:cadcf7afccefa03edb1aa8aa65a7c7f6d5a00b87d61586d1a63f6db8fae7c5f3` |
| 10 bps | `contextual_only` | `sha256:4030d8a3099b3319420e2b927a6d8281536eb1deb9cfe8d988f07259d5ecc3db` | `sha256:b4453107598e3c00452f0fee402665e3fbbb0712c23b46aff1e42f5dfd651da7` |

Before the stage attempt's postlock read or release of any 2024 market byte,
the parent validator must prove the exact payload inventory and hashes, parse
the checkpoint canonically, bind the terminal online model and every required
cost-specific administrative account, and require all of the following parent
facts:

- `stage_pass=true`;
- `status=POST_REJECTION_AUDIT_POLICY_CANDIDATE`;
- historical policy candidate true;
- learning candidate false;
- 64/64 gates passed;
- original v2 development rejection remains final;
- no post-2023 market value was accessed;
- no open cash episode is silently closed or reset at the boundary; and
- the terminal account is continued, never restarted from cash or new capital.

The exact pinned and already independently verified parent bundle is the trust
root. The 2024 stage revalidates its bytes, schemas, checkpoint, model state,
account state, and cutoff, but does not rerun the entire 1999-2023 experiment.

## Physically bounded 2024 input

The audit may name and open exactly one market input:

`e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/authorized_inputs/aapl_spy_qqq_through_2024.csv`

Its frozen identity is:

- physical columns, in order:
  `date,aapl_open,aapl_close,aapl_adj_close,spy_adj_close,qqq_adj_close`;
- raw SHA-256:
  `abf8115e61e7a7724ed816db7dbd0fe053b098123b2e1e43886eb0074b575be7`;
- Git blob:
  `3cdb68aac6a3b7b1cda4f64ee81193690f28a1d7`;
- first session: `1999-03-10`;
- last session: `2024-12-31`;
- rows: `6496`;
- date-sequence SHA-256:
  `6a3a247806d34797029831a945c15458e15eabc2bd36b021d2a815668c7d5e4c`;
- canonical SHA-256 after adjusted-open reconstruction:
  `5b3df584b4ccedb6f6871e6cbd43095126378485aec4f3a55f4846d2fb74f071`;
- exact canonical through-2023 prefix SHA-256:
  `3b5e02acaa69a56fa47a0fd34275472d226b62c0f61741c13d3680b239b82535`;
- physically later rows: false; and
- rows returned after the bound: false.

The audit-safe receipt is frozen at:

`e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/input_snapshot_receipt.json`

with raw SHA-256
`b2f541541f132f6c22257c18b36ec2f785260b105d44e35a8f0da0c35ac5fae9`
and Git blob `cd69e5d15c3ed122c0082ae5d7f87bc84c2311f7`. Its schema contains only
the bounded destination, through-2023 prefix authority, opaque source/bundle
hashes, and data-only preparation attestations. It contains no later path or
later date.

A separate full preparation record is committed outside the audit runtime
surface as documentary evidence. Its path and contents are deliberately absent
from this contract's executable requirements. The sanitized receipt preserves
only opaque expected hashes. Stage and verifier must never name, locate,
inspect, bind, open, parse, copy, hash, enumerate through, or print the full
record itself.

The input was prepared mechanically before any 2024 policy implementation or
scoring, then independently checked using bounded-data integrity functions
only. No standalone cutter implementation or execution commit was
preregistered; this is transparently self-attested mechanical preparation plus
independent destination/prefix/checkpoint validation, not a claim of a sealed
cutter run. The preregistration review invoked no policy or scoring function
and printed no market value. The stage attempt's own first receipt/input
worktree read remains postlock. Policy, evaluation, runner, bootstrap, and
verifier accept only the frozen bounded path and sanitized receipt above.

## 2025-and-later firewall

No later path, date, byte, market value, outcome, forecast, ledger, action, or
metric is authorized. No prior 2024 action, forecast, ledger, score, return, or
expected action digest is authorized either. Tests use synthetic temporary
paths only. Source scans, Git scans, broad filesystem enumeration, and error
messages must not expose a later market artifact or a prior 2024 result to
audit code.

The bootstrap scans and attests executable code before importing the audit
package, while blocking the repository data tree as an import surface. Every
Git command is restricted to a frozen, exact pathspec allowlist. Full-index,
full-tree, repository-wide, or broad `e/` enumeration is forbidden. Before the
durable lock, Git checks may prove only the unopened input and receipt's HEAD
and stage-zero index object IDs. Input/receipt local status, worktree equality,
and literal bytes are deferred until after lock. The quarantined preparation
record is outside the runtime allowlist entirely.

Only after the lock is durable may the runner verify the receipt's local raw
hash and schema, then the input's path status, literal raw hash, physical end,
canonical identity, receipt agreement, and exact through-2023 canonical
prefix. No 2024 row may be returned to model or evaluation code until that
prefix and the sealed parent checkpoint agree.

## One-way attempt lock

Immediately after bootstrap entry, and before any audit-package import or
repository path access, bootstrap must harden every path. Interpreter loading
of the explicitly invoked bootstrap file and Python standard library is the
only unavoidable exemption. Repository root, control root,
`authorized_inputs/`, input, receipt, lock, pending/success markers, `runs/`,
private seal, failure, and final paths must be normalized absolute paths with no `..`;
every child must be a
lexical descendant of the frozen repository/control parent. Every existing
path component must be an ordinary file or directory of the expected type and
must not be a symbolic link, junction, mount point, or Windows reparse point.
Input and receipt must be ordinary non-reparse files. Nonexistent output paths
must have an already hardened ordinary parent. These checks repeat immediately
after lock readback and immediately before each receipt/input read, private
directory creation, and promotion.

Prelock work may inspect only explicitly named committed code/control files,
pathspec-limited Git metadata, and the sealed through-2023 parent. The
implementation freezes a literal dependency inventory; dependency discovery,
full-repository scans, and directory-glob expansion are forbidden. Prelock
must prove, in order:

1. exact branch, pushed HEAD/upstream/origin identity, and frozen
   preregistration-commit ancestry;
2. exact HEAD/index/worktree bytes and clean status for every literal
   executable/control dependency, explicitly excluding input and receipt;
3. exact parent inventory, bytes, schemas, selected `online_full` checkpoint,
   model state, and every required cost-specific administrative account state;
4. exact unopened input and sanitized-receipt HEAD/index Git blobs, without
   consulting their status or worktree bytes;
5. the control root inventory is exactly the one bounded CSV under
   `authorized_inputs/` plus `input_snapshot_receipt.json`, with no lock,
   pending/success marker, `runs/`, temporary, final, quarantine, or
   failed-seal entry; and
6. the monotonic elapsed time remains below the strict stage deadline.

The runner then exclusive-creates the canonical lock, flushes it, fsyncs the
control root where supported, and verifies canonical byte readback. The lock
binds the contract, run ID, stage, branch, preregistration commit, pushed
runtime commit, literal dependency inventory/hashes, parent inventory/hashes,
selected checkpoint/account hashes,
input HEAD/index Git blob plus expected raw/canonical identities, receipt
HEAD/index Git blob plus expected raw identity and its opaque preparation
lineage hashes, exact prelock control-root inventory, output path, creation
time, expected-absent pending/success-marker paths, and attempt number one. It
is never removed, replaced, or rewritten.

After the lock, and before any 2024 row is returned, the runner must perform
this exact one-way sequence:

1. read back the durable lock and compare its canonical bytes;
2. recheck branch, HEAD, upstream, origin, literal dependency identities, and
   sealed-parent inventory;
3. require the control root to equal its frozen prelock inventory plus the
   lock, with `runs/` still absent;
4. recheck receipt HEAD/index identity, require exact-path clean status and
   literal local bytes equal HEAD/index, repeat anti-reparse checks, then prove
   its raw hash and every field with a dedicated strict verifier for this
   custom nested receipt schema; inherited `verify_authorized_price_lineage`
   must not be used or weakened;
5. recheck input HEAD/index identity, require exact-path clean status and
   literal local bytes equal HEAD/index, repeat ordinary-file/anti-reparse
   checks, then prove local raw hash, physical schema/bounds,
   date/canonical hashes, receipt agreement, and exact canonical through-2023
   prefix/checkpoint continuity; and
6. only after all prior checks pass, release the 2024 suffix to replay and
   evaluation code.

Before marker preparation, the control root may add only `runs/`, containing
exactly the private directory and then, after promotion, exactly the final run
directory. During marker preparation, the exact root inventory is bounded
input, sanitized receipt, immutable lock, `runs/<run-id>`, and the pending
marker; the final success marker is absent. Between pending-to-final marker
rename and control-root fsync, the same inventory contains the final marker
instead, but that marker is explicitly uncommitted. Commit completes only when
the rename and control-root fsync both succeed. After success, the inventory is
unchanged, the final marker is committed, and the pending marker is absent.

Cleanup is limited to the current attempt's private directory, uncommitted
pending marker, and an uncommitted final marker whose post-rename directory
fsync failed. It may never remove, rewrite, move, or broadly hash the input,
receipt, lock, committed success marker, or control root.

After a private directory exists, any caught failure before marker-commit
completion must preserve the bundle by atomically renaming that exact private
directory -- or a just-promoted final directory -- to the one frozen failure
directory, then fsyncing `runs/`. Any pending marker, or final-path marker still
uncommitted because its control-root fsync failed, is explicitly removed first.
The final run path must be absent and `runs/` must contain only the failure
directory; pending and success markers must be absent. No second failure name,
suffix, retry, deletion, or reuse is allowed.

Any failure after lock creation consumes the only stage attempt. A partial
result is never reusable.

## Exact sealed output inventory and schemas

The promoted directory contains exactly 43 ordinary non-reparse files, no
subdirectory, and no other entry. The filenames are:

1. `.gitattributes`
2. `AUDIT_2024_ATTEMPT_LOCK.json`
3. `input_snapshot_receipt.json`
4. `audit_prices_through_2024.csv`
5. `parent_stage_manifest.json`
6. `parent_checksums.json`
7. `parent_verification.json`
8. `prefix_continuity_proof.json`
9. `source_bundle_provenance.json`
10. `known_result_isolation_evidence.json`
11. `audit_fixed_features_2024.table.json`
12. `audit_forecast__frozen_2023_lead.table.json`
13. `audit_forecast__online_2024_shadow.table.json`
14. `audit_forecast__fixed_comparators.table.json`
15. `audit_matured_lessons__frozen_2023_lead.table.json`
16. `audit_matured_lessons__online_2024_shadow.table.json`
17. `audit_state_weight_diagnostics_2024.table.json`
18. `audit_continuation_checkpoint_through_2024.json`
19. `audit_pending_lessons.json`
20. `audit_replay_diagnostics.json`
21. `audit_ledger__base_5bps__frozen_2023_lead.table.json`
22. `audit_ledger__base_5bps__online_2024_shadow.table.json`
23. `audit_ledger__base_5bps__aapl_buy_hold.table.json`
24. `audit_ledger__base_5bps__always_long.table.json`
25. `audit_ledger__base_5bps__exact_union_cash.table.json`
26. `audit_ledger__base_5bps__contextual_only.table.json`
27. `audit_ledger__stress_10bps__frozen_2023_lead.table.json`
28. `audit_ledger__stress_10bps__online_2024_shadow.table.json`
29. `audit_ledger__stress_10bps__aapl_buy_hold.table.json`
30. `audit_ledger__stress_10bps__always_long.table.json`
31. `audit_ledger__stress_10bps__exact_union_cash.table.json`
32. `audit_ledger__stress_10bps__contextual_only.table.json`
33. `audit_episodes__base_5bps.json`
34. `audit_episodes__stress_10bps.json`
35. `audit_xor__base_5bps.json`
36. `audit_xor__stress_10bps.json`
37. `audit_metrics.json`
38. `audit_gate_report.json`
39. `audit_integrity_evidence.json`
40. `audit_runtime_cost_evidence.json`
41. `report.json`
42. `checksums.json`
43. `stage_manifest.json`

`.gitattributes` has exact bytes `* -text\n`, SHA-256
`705fd4d6451a31d36b3df7de96f83f30ac976c9b4a6d1e51671d8e2f33e2d0da`.
The lock, sanitized receipt, bounded CSV, parent manifest, and parent checksums
are byte-exact copies of their pinned sources. The bounded CSV therefore keeps
the frozen input raw hash. All other JSON is canonical UTF-8 with finite
numbers, sorted object keys, compact separators, and one terminal LF; unknown
or missing keys are fatal.

Every `*.table.json` uses table schema version 1 and the exact key set
`column_types,columns,index_name,index_type,index_values,rows,table_schema_version`.
The canonical column-list hash domain is SHA-256 of the ordered UTF-8 compact
JSON array. Schemas are frozen as follows:

- 2024 fixed features: 252 increasing date-indexed rows, column-list hash
  `8282e3a6cd1ef04f26bc2131d9ec9827a3f0aaa706f5f68fd00cae7ff8c42a20`;
- each lead/shadow forecast: the same 252-date index, 61-column hash
  `d08661dc9ea8526c9697750143fa061cba2c197641f6c8480fa25c7a512ea9b1`;
- fixed-comparator forecasts: the ordered columns
  `fixed_always_long_target_exposure,fixed_union_cash_target_exposure,fixed_contextual_only_target_exposure`,
  hash
  `0c6815db9e6e2e7e550e909611129b75ed2329783d2609ed35b561cc241615a2`,
  with the same 252 increasing 2024-date index;
- each 2024-matured-lessons table: only events processed during the suffix,
  ordered by maturity and signal date, 16-column hash
  `10cf64eea698fa0761acbaa56793bb92ddcf3b50098b21832cae06b43e8afb4c`;
- state/weight diagnostics: exactly two scenario labels times 252 sessions,
  22-column hash
  `558403da0eccf05db0313c8668840f9726ea2ad1652ff0695a24eafa052ba4e0`;
  and
- every ledger: the exact continuous 5,033 rows from 2005 through 2024 and
  34-column hash
  `9f1f1b34d3f0db6cedf9b7cbd59ba9c75b8e2be2055e761568397e60fd386971`.

The checkpoint is a strict schema-version-1 object containing the two terminal
model scenarios, all six policy account states at both costs, prefix/source
identities, pending/cooldown state, and its own canonical self-hash. Episode
files contain every full-ledger cash episode per policy plus explicit open
status. XOR files use the fill-based schema below. `audit_metrics.json` contains
the exact annual, quarter, full-period, removed-best-quarter, negative-quarter,
drawdown, turnover, trade, episode, XOR, and continuous diagnostics.
`audit_gate_report.json` contains every primitive unrounded value, Boolean,
tolerance, and final status. The parent/prefix/provenance/isolation/replay,
pending, integrity, runtime, and human report files each use schema version 1
and their exact declared evidence class. Their literal nested key/type schemas
are frozen in a single implementation-time `ARTIFACT_SCHEMA_REGISTRY` before
the one-shot lock, committed/pushed with the runtime, hashed in dependency
identity, and copied into verifier constants. This registry may only make the
semantic requirements above stricter; it may not omit, rename, weaken, or make
optional any contracted evidence or gate. Unknown or missing keys relative to
that frozen registry are fatal. These non-economic serialization key sets are
therefore implementation-frozen, not falsely claimed to be fully enumerated by
this preregistration. The verifier independently reconstructs every payload
from sealed primitives rather than trusting report summaries.

`stage_manifest.json` has the exact fields
`manifest_schema_version,manifest_sha256,contract_version,verifier_id,run_id,stage,status,stage_pass,evidence_classification,preregistration_commit,git_identity,dependency_identity_sha256,artifact_schema_registry_sha256,parent_manifest_file_sha256,parent_manifest_self_sha256,parent_checkpoint_file_sha256,parent_checkpoint_self_sha256,receipt_file_sha256,input_raw_sha256,input_canonical_sha256,attempt_lock_file_sha256,payload_sha256,runtime_cost_evidence_file_sha256,gate_report_file_sha256,learning_classification,later_market_data_accessed,prior_2024_artifact_accessed,external_cost_usd`.
Its `payload_sha256` is the exact sorted 41-entry map for files 1-41.
`manifest_sha256` reuses the proven self-hash domain: SHA-256 of canonical
manifest JSON with the `manifest_sha256` field omitted, not null.
`checksums.json` is the exact sorted 42-entry raw-file SHA-256 map for files
1-41 plus `stage_manifest.json`; it excludes only itself. Any extra, missing,
renamed, nested, linked, or nonordinary entry is fatal.

Every manifest hash domain is explicit:

- `parent_manifest_file_sha256` is the raw-file hash
  `fd4c7427bdc9ec2a5463dcc2781310aeab5c498836db0617c0c41ebf81267bbd`;
- `parent_manifest_self_sha256` is its internal omitted-field self-hash
  `c71e201a8a3d1ef2b0ebe914d46abfdd0f5a9154d4f106ce723168a9bce009e8`;
- `parent_checkpoint_file_sha256` is raw checkpoint payload hash
  `560571500f580f9e0fd4319a93905b4ad1e498f1c1f27de9d647da455b6d5477`;
- `parent_checkpoint_self_sha256` is its internal self-hash
  `4aaf549c9352ac322f36c46cb8dcdd03f672a1eba580813ba8a1947d323a85eb`;
- receipt, attempt-lock, runtime-evidence, gate-report, checksums entries, and
  every `payload_sha256` value are raw file-byte SHA-256 values;
- `input_raw_sha256` hashes the literal bounded CSV bytes, while
  `input_canonical_sha256` uses the frozen adjusted-open canonical frame domain;
- `dependency_identity_sha256` hashes canonical compact JSON
  `{"schema_version":1,"files":{path:{"head_blob":...,"index_blob":...,"worktree_raw_sha256":...}}}`
  over the exact sorted `FROZEN_DEPENDENCY_PATHS`; and
- `artifact_schema_registry_sha256` hashes canonical compact JSON of the exact
  literal nested registry committed in the artifacts module.

## Exact arms and causal continuation

Both arms fork the exact parent `online_full` model and online administrative
account at the 2023 terminal checkpoint.

Implementation must use a dedicated two-scenario fork. The inherited
four-arm `fork_confirmation_arms` path is forbidden because it selects
different parent arms. Both restored model payloads and both duplicated
cost-specific policy accounts retain their internal `policy_name=online_full`
for checkpoint and hash-chain continuity. `frozen_2023_lead` and
`online_2024_shadow` are external scenario labels only and must never rewrite
the inherited ledger policy name.

### `frozen_2023_lead`

- Processes 2024 market features chronologically.
- Restores the exact terminal online model state learned through 2023 with the
  sole runtime override `learning_mode=frozen_cutoff`,
  `frozen_cutoff=2023-12-29`, `ablation_mode=full`.
- Admits zero lessons whose maturity occurs after 2023-12-29.
- Preserves the parent's decision threshold and every frozen hyperparameter.
- Supplies the only policy result eligible for top-level stage success.

### `online_2024_shadow`

- Starts from exactly the same model, pending lessons, cooldown, target,
  portfolio, cash, shares, and cost basis as the lead.
- Restores runtime `learning_mode=causal_online`, `frozen_cutoff=null`,
  `ablation_mode=full`.
- Continues the existing causal maturity/update order.
- A lesson may affect only decisions after its outcome has matured.
- Its result is diagnostic and cannot rescue a failed lead.

Required fixed comparators are:

- `aapl_buy_hold`;
- `always_long`, whose economics must equal AAPL exactly under the identity
  exclusions below;
- `exact_union_cash`, the best complete-period parent comparator; and
- `contextual_only`, the best 2019-2023 parent comparator.

All decisions are binary 0% or 100% AAPL. Target and realized exposure must
stay in `[0,1]`; shares and cash must never be negative; margin, leverage,
shorting, borrowing, and interest are forbidden. The two costs are 5 bps and
10 bps per changing leg, with identical action streams across costs.

`always_long` and `aapl_buy_hold` retain distinct policy names and independent
hash chains, so their raw ledgers/accounts cannot be byte-identical. Fatal
economic equality compares every ledger field except `policy_name`,
`previous_row_sha256`, and `row_sha256`, and every terminal account field
except `policy_name` and `ledger_tip_sha256`. Each excluded identity/hash field
must still validate independently. Their equity, cash, shares, exposure,
returns, costs, drawdown, active edge, and all other fields must agree within
`1e-10` (categorical/integer fields exactly).

## Known-result isolation

No prior 2024 forecast, action stream, action count, action digest, ledger,
metric, return, gate, or report may enter stage, verifier, tests, expected
fixtures, dependency constants, source comments, or error messages. Policy and
replay code must not import any reporting-only diagnostic module. Static tests
must prove the literal dependency inventory contains no prior-2024 artifact
path and that policy/replay modules contain no expected 2024 action/result
constant.

The audit generates, canonicalizes, hashes, and freezes both new forecast/action
streams in memory from the parent before ledger construction or return
interpretation. Stage and standalone verification do not compare them with
another branch. A human may document a cross-approach comparison only after
the immutable result is sealed, and that later observation cannot change this
contract, implementation, gates, or status.

## Preregistered lead-policy gates

Calendar quarters are fixed as Jan-Mar, Apr-Jun, Jul-Sep, and Oct-Dec. All
edges are same-ledger boundary active log edges; no terminal cash or XOR
episode is force-closed.

All gate arithmetic uses unrounded binary64 values. Absolute reconciliation
tolerance is `1e-10`; zero-only classification tolerance is `1e-12`. Neither
tolerance may soften, offset, or turn equality into a pass for a strict
`>0.001`, `>0`, or count threshold. Display rounding occurs only after every
gate and status is fixed. The 2024 edge is the increment in cumulative active
log edge from the sealed 2023 account boundary through the final 2024 ledger
row. A quarter edge is the additive increment over that calendar quarter.
Removing the best quarter means `full_year_edge - max(quarter_edges)`; it does
not rerun or splice the ledger. Negative-AAPL-quarter aggregate edge is the sum
of lead quarter edges for quarters whose same-boundary AAPL log return is less
than `-1e-12`; if none exist, the gate is explicitly not applicable.

At each of 5 bps and 10 bps, `aapl_policy_pass` requires:

- full-2024 frozen-lead active log edge versus AAPL strictly greater than
  `0.001`;
- at least two of four calendar quarters with strictly positive edge;
- full-year edge after removing the best quarter strictly positive; and
- if AAPL has any negative-return quarters, positive aggregate lead edge in
  those quarters; otherwise an explicit not-applicable status.

At stress cost, `simple_policy_superiority_pass` additionally requires the
frozen lead's full-2024 same-boundary log return minus each of
`contextual_only` and `exact_union_cash` to be strictly greater than `0.001`.

Top-level `stage_pass` is true only when `aapl_policy_pass` passes at both
costs, `simple_policy_superiority_pass` passes, and every fatal integrity check
passes. A passing status is
`POST_HOC_FROZEN_POLICY_2024_REPLICATION_PASS`, never a claim of new unseen
evidence. With all integrity checks passing,
`AAPL_BEATEN_BUT_MODEL_NOT_SELECTED` is selected if and only if
`aapl_policy_pass` is true at both costs and
`simple_policy_superiority_pass=false`; it is not triggered by the full-year
edge alone. If any fatal integrity check fails, status is
`REJECTED_2024_INTEGRITY`; every other nonpass status is `REJECTED_2024`.

The report also preserves annual/quarter returns, episode statistics, turnover,
trade counts, maximum drawdown, continuous 2005-2024 diagnostics, and results
with the best quarter removed. Those diagnostics cannot replace a failed
preregistered gate.

## Separate learning classification

XOR economics are fill-based, not decision-date based. The two full ledgers are
aligned by `fill_date`. Canonical exposure after each fill is
`requested_target_exposure`; `post_fill_exposure` must exist and equal it
exactly on every row. An XOR episode begins on the first fill row whose
canonical post-fill exposures differ and includes that divergence fill and its
costs. It ends on and includes the first later fill row whose post-fill
exposures are equal, thereby including convergence costs. Start/end decision
dates are preserved only as diagnostics; entry quarter is determined by the
start fill date. A differing terminal decision with no authorized next fill
creates no economic XOR. If exposures remain different on the last authorized
fill, the episode is reported open, is not complete, and is never force-closed.

Orientation is `shadow_cash_lead_long` for `(shadow,lead)=(0,1)` and
`shadow_long_lead_cash` for `(1,0)`. A same-fill orientation flip is split into
two episodes exactly as the inherited signed-XOR extractor: the old episode
receives that fill's raw market component plus the changing-leg cost that
exits its orientation and closes at the flip fill; the new episode opens at
the same fill with zero raw component plus the other changing-leg entry cost.
Each policy's changing-leg cost is assigned exactly once. The new episode gets
the fill observation; the closing old episode does not. Continuous mixed-
orientation episodes are forbidden.

Episode incremental edge is the sum, from divergence through convergence
inclusive, of shadow minus lead active-log increments. Full-period incremental
edge is the shadow same-boundary log return minus lead same-boundary log return.
Complete plus open XOR contributions must reconcile that full edge within
`1e-10`. Removing the best complete XOR episode means subtracting its additive
incremental log edge from the full-period incremental edge; no ledger is
rerun.

`learning_candidate_for_2025_shadow` is true only if online versus frozen has:

- at least three complete XOR episodes at both costs;
- complete-XOR fill entries in at least two distinct calendar quarters at both
  costs;
- incremental active log edge strictly greater than `0.001` at both costs; and
- 10-bps incremental edge after removing the best complete XOR episode
  strictly greater than zero.

The mutually exclusive classification is exact:

- `unexercised`: no economic XOR fill row, even if lessons were admitted,
  decision targets briefly differed without an authorized fill, or internal
  scores/model state changed;
- `exercised_insufficient_evidence`: at least one economic XOR fill row but
  fewer than three complete episodes or entries in fewer than two quarters;
- otherwise `exercised_positive` when 10-bps incremental edge is greater than
  `1e-12`, `exercised_negative` when it is less than `-1e-12`, and
  `exercised_flat` when its absolute value is at most `1e-12`.

The classification sign does not imply candidate status; all stricter
conditions above must pass. Learning status never changes `stage_pass`.

## Fatal integrity requirements

Every integrity condition below is fatal:

- exact contract, branch, pushed commit, literal dependency inventory, parent,
  selected checkpoint/accounts, sanitized receipt, input, lock, control-root
  inventory, and output identity;
- exact parent model/account fork with no reset or added capital;
- exact through-2023 canonical prefix and checkpoint continuity;
- frozen lead admits zero post-cutoff lessons;
- online shadow obeys causal maturity and update order;
- one lesson per eligible opportunity and exact pending/cooldown state;
- binary actions and cross-cost action identity;
- no short, leverage, borrowing, margin interest, negative shares, negative
  cash, or exposure outside `[0,1]`;
- exact always-long/AAPL economic equality under the explicitly frozen
  policy/hash-chain field exclusions, with both chains independently valid;
- exact unrounded ledger, episode, XOR, quarter, removal, and gate
  reconciliation under the frozen tolerances;
- no later-data access and no prior-2024 result/action artifact access;
- zero network, news, LLM, API, external-model, and monetary use; and
- strict stage and verifier runtime at every named checkpoint, including the
  reserved stage prepromotion/final-commit sample and standalone bootstrap-exit
  check.

## Sealing and independent verification

The stage builds a private same-filesystem bundle, writes the exact payload
inventory, flushes every file, writes canonical checksums and a self-hashed
manifest, fsyncs directories where supported, and invokes an independent
semantic verifier before one atomic promotion.

Bootstrap records one monotonic start before path attestation. Named live
checks are `preseal`, `post_private_verify`, `prepromotion`, and
`final_commit_sample`. `prepromotion` must be strictly below 1,795 seconds,
reserving five seconds. Stage atomically renames the private bundle, fsyncs
`runs/`, validates the exact final inventory, and constructs the canonical
success-marker bytes.

Those bytes are exclusive-written to the frozen pending-marker path, flushed,
fsynced, read back, and self-validated; the control root is fsynced while the
marker is still pending. The marker is canonical schema version 1 with exact
fields
`marker_schema_version,marker_sha256,contract_version,run_id,stage,preregistration_commit,git_commit,attempt_lock_file_sha256,final_relative_path,final_inventory_sha256,stage_manifest_file_sha256,stage_manifest_self_sha256,checksums_file_sha256,marker_preparation_elapsed_seconds,stage_deadline_seconds,marker_preparation_deadline_pass,external_cost_usd`.
`final_inventory_sha256` hashes the canonical sorted 43-filename JSON array;
all file hashes are raw-byte hashes except the explicitly named manifest self
hash. Marker self-hashing uses canonical JSON with `marker_sha256` omitted.

After pending-marker validation, `final_commit_sample` must be strictly below
1,800 seconds. This sample is the explicit endpoint of the internal stage
runtime metric. The immediately following atomic rename from pending to final
marker plus a successful control-root directory fsync together form the sole
stage-success commit protocol. That small atomic/fsync tail is outside the
internal metric and is instead covered by the separately required external
complete-command wall time. A caught fsync failure leaves the renamed marker
uncommitted and triggers its explicitly authorized cleanup plus the frozen
failure transition. No integrity/deadline condition may change success after
both commit operations succeed. A process loss after commit completion is
therefore a completed stage even if stdout was not observed.

A crash after final-directory promotion but before a complete valid marker is
distinguishable and is never success: standalone verification requires both.
Any caught failure before marker commit completion must attempt the frozen final/private
to failure-directory transition. That transition and its fsync are explicitly
exempt from the normal postpromotion operation allowlist. Because filesystem
operations cannot be guaranteed to succeed after a hardware/process failure,
if the mandatory failure transition itself fails, the terminal state is a hard
forensic failure with no valid marker; any residual path is preserved and
standalone verification rejects it. The attempt remains consumed and is never
retried.

The private verifier reconstructs parent, receipt, input, checkpoint, action,
ledger, comparator, learning, gate, and manifest evidence and requires
byte-exact agreement before promotion. Monotonic timestamps and durations in
`audit_runtime_cost_evidence.json` are explicitly process-local attestations:
their schema, finiteness, ordering, clock identity, command start, and recorded
deadline pass are verifiable, but their numeric values are not regenerated by
another process. The in-process private verifier starts a clock at invocation;
the standalone verifier starts at bootstrap entry. Each checks its own named
deadlines and independently rebuilds all reproducible economic/integrity
evidence. The invoking shell's measured stage wall time, live final-commit
sample, and marker-preparation attestation are later copied verbatim into the
preservation result; none is falsely described as independently reproducible
by another process.

The separately invoked standalone verifier has its own `preverify`,
`postreconstruction`, and `bootstrap_exit` checks, each strictly below 1,800
seconds. It requires the exact durable success marker and final bundle. It is
strictly read-only: success or failure never removes, rewrites, renames, or
quarantines evidence.

The only authorized commands are:

```text
python -I -B agent_benchmark/contextual_expert_aggregation_2024_audit_bootstrap.py stage audit_2024
python -I -B agent_benchmark/contextual_expert_aggregation_2024_audit_bootstrap.py verify audit_2024
```

The standalone verifier is repeatable and read-only; the stage is not.

## Later decision

A 2024 pass authorizes only proposing and preregistering a separate 2025 audit.
It does not authorize opening 2025 under this contract. If the lead fails,
2025 and 2026 remain closed for this approach. If the lead passes but learning
does not, the next lead remains fixed and the learner remains a shadow. Even a
full pass is one historical year and does not authorize real capital.
