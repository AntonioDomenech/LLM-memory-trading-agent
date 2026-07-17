# AAPL SEC/Gemma online risk overlay v2.2

## Status

Rejected at the complete-lifecycle runtime preflight on
`codex/aapl-sec-gemma-online-risk-overlay-v2-2`. The implementation remains
unrun: no official SEC acquisition, Yahoo market acquisition, Gemma
generation, prediction, trade, score, or performance result occurred. See
`docs/aapl_sec_gemma_online_risk_overlay_v2_2_preflight_rejection.md`.

The rejected v2 branch and the v2.1 rejected no-effect preflight branch at
commit `9c2fbb0` remain preserved separately. No v2.2 SEC acquisition, Gemma
extraction, market-value read, prediction, action, or score has occurred.

v2.2 keeps the same trading thesis, data windows, chronology, model, features,
learner, policy, transaction costs, no-leverage rules, and success gates as
v2.1. It changes only execution-integrity controls for crash-safe ordinary
report publication. Any later change to the trading rule or success thresholds
requires another branch and preregistration.

The complete machine-readable contract is
`agent_benchmark/sec_gemma_online_risk_overlay_contract.py`, whose literal
manifest SHA-256 is
`64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d`.

## Trading idea retained unchanged

Gemma is a fixed local filing reader, not the trader. It reads redacted Apple
10-K and 10-Q evidence and extracts deterioration signals. A chronological
online learner combines those signals with market context. The account may be
long one unit of AAPL or in cash only. It never shorts, borrows, uses leverage,
earns cash interest, or holds negative cash.

Learning is causal. Before each filing decision, the learner may use only
lessons whose complete 20-session outcomes have already matured. It keeps
learning during later unseen periods exactly as a live system would, but it
never sees an outcome early. The same action stream is charged at both 5 and
10 basis points and is compared with AAPL buy-and-hold on the same ledger.

The stages remain:

1. development on 2000-2018, with 2000-2004 used only as warm-up;
2. untouched confirmation on 2019-2023; and
3. untouched live-style final evaluation on 2024, 2025, and 2026 year to date.

No later stage can be opened unless its predecessor terminally passes.

## Local Gemma identity retained unchanged

The model, manifest, config digest, ordered four layer digests, Ollama version,
prompt, schema, temperature, seed, context, and output limits remain unchanged.

The inherited runtime identity requires strict JSON and the exact `/api/show`
top-level key set. It requires `modified_at` to exist as a string, removes only
that field, then canonicalizes and hashes the entire remaining object. The
semantic show hash remains
`5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6`.
Whitespace, JSON key order, and the timestamp value alone may vary. Every other
semantic mutation fails. The raw response hash is recorded as diagnostic
evidence but is not an identity gate.

The complete inherited runtime fingerprint remains
`816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77`.

## Acquisition cannot leak future information

Official SEC and Yahoo bytes live in a durable opaque quarantine vault. The
runner and scoring code never receive a public raw bundle, mapping, or object
attribute. A current SEC submissions response may contain later records, and a
Yahoo response may contain current quote metadata or a transport-only row.
Those values remain inside the vault. After a scored attempt is consumed, it
receives only its canonical stage-cutoff slice.

The validator rebuilds the official SEC catalogue, complete eligible universe,
selected primary-document artifact, and receipts from the exact raw bytes by
using the existing detached catalogue and stage-content replay validators.
Every accession, form, URL, acceptance timestamp, and availability session must
come from that replay. Caller-supplied dates are not evidence.

Market acquisition must contain the complete frozen history, not merely the
latest 253 rows. AAPL must contain every expected market session through the
stage boundary. Context symbols must contain every expected session from their
frozen inception except only the literal allowlisted TNX absences. Weekends,
unexplained gaps, a missing last usable session, a foreign URL, or a truncated
tail fail acquisition.

Only exact source-bound production transports and authorities can consume an
attempt. Test doubles can exercise tests but cannot authorize a production
registration or consumption.

## Exact pass and failure evidence

An acquisition may pass only through an opaque verified acquisition report
whose seven exact digest-valued checks replay raw bytes, receipts, attempt
scope, private identity handling, market-prefix continuity, blinded Gemma
requests, and all request/byte/retry/redirect caps. An arbitrary all-true
mapping cannot pass.

The contract freezes the exact fields for acquisition validation, acquisition
terminal evidence, scored terminal evidence, publication intent, external
publication receipt, final-registry successor, and final-registry
authorization. Bare hashes, strings, mappings, and older artifacts cannot
substitute for those opaque v2.2 objects.

A scored stage may pass only after deterministic reconstruction of the complete
chronological replay, stage metrics, exact literal gate key set, independent
no-leverage proofs, current store record commitment, and joint report. The
terminal report binds an artifact receipt and payload hash that actually exist
in the append-only store.

If a valid run fails a performance gate, its complete evaluation, metrics, gate
report, no-leverage proofs, and joint report are stored and externally pinned.
Only then does the attempt become terminal-fail and release its sealed
diagnostic. It cannot be hidden, retried, converted to pass, or used to open
the next stage.

## Crash-safe publication intent

Before the first remote push for an ordinary acquisition-pass, scored-pass, or
scored-failed-gate report, the publisher prepares and verifies the complete
canonical annotated-tag message, the one stable per-attempt terminal tag ref,
exact local tag-object SHA-1, peeled implementation commit, frozen remote name,
and frozen remote URL. The ordinary ref is independent of terminal status,
report kind, and artifact hash, so copied or forked stores cannot publish
contradictory terminal claims under different refs.

The store then appends a self-hashed publication intent that fixes the intended
terminal status, report kind, terminal artifact and its store receipt, record
counts and record commitment, terminal-reconstruction material hash, store
receipt for that reconstruction material, store identity, predecessor
publication, complete canonical tag identity, and expected publication hash.
Its journal sequence and tip identify the exact committed store state
immediately before the intent append.

Intent durability is explicitly two phase: fsync
`publication_intent_prepare`, append the exact intent journal row, then fsync
`publication_intent_committed`. Before the first anchor, the normal parent's
elapsed time is encoded as an exact lowercase finite nonnegative binary64
hexadecimal value below 3,600 seconds and placed inside the self-hashed intent.
The prepare anchor stores the complete canonical intent row bytes, including
that elapsed hex. Therefore a crash after the database row but before the
committed anchor can reproduce the exact committed state without reading a new
clock after reboot. Each partial database/anchor state has one deterministic
reconciliation, and exact recommit is idempotent. A different or unexpected row
or anchor poisons the state. Only the committed state can issue the opaque
intent store receipt or permit a remote push.

The intent store receipt is deliberately outside the self-hashed intent body,
avoiding a circular hash. Later acquisition or scored terminal evidence and the
terminal anchor bind both `publication_intent_sha256` and
`publication_intent_store_receipt_sha256`. This proves the intent was durable
before this implementation was permitted to push; it does not claim to prove
that an independently preexisting exact remote tag was created later. Intent
and publication-receipt governance records remain journaled and externally
anchored but are excluded from the semantic record counts and record
commitment. No live monotonic clock or deadline is stored in the intent; only
the exact elapsed snapshot frozen before the prepare anchor survives reboot.

Issuing that intent invalidates every research and effect capability. SEC,
Yahoo, Ollama, Gemma, model, market-value, feature, learner, replay, metric,
gate, no-leverage, scoring, and artifact-mutation work can never resume. The
store may issue only a fresh narrow publication capability, bound to the exact
store instance and a new store-session nonce. Every capability from a prior
process or session is stale.

The verified intent is also the durable no-semantic publication-pending
receipt. Publication-pending releases no extraction, prediction, action,
metric, gate, return, diagnostic direction, or terminal artifact. It blocks
result release and the next stage.

On restart, the store reconciles an exact durable publication intent before its
generic consumed-attempt recovery. A consumed attempt interrupted before any
valid intent remains terminal-indeterminate under the existing rule. Once the
intent exists, later interruption cannot convert it to indeterminate; it stays
publication-pending until exact recovery completes or a conflict poisons it.

Before intent preparation, the store must also commit exactly one self-hashed
terminal-reconstruction-material row under the attempt and report kind. It has
separate exact acquisition and scored schemas, a two-phase
`terminal_reconstruction_prepare` / journal row /
`terminal_reconstruction_committed` protocol, and an opaque store receipt. The
intent binds both its row hash and receipt hash.

The acquisition row names the exact terminal artifact and receipt, validation,
bundle, manifest, private index, check set, record commitment, sealed
acquisition evidence, and vault commitments. The scored row names the exact
joint artifact and receipt, stage input, deterministic evaluation, metric
inputs and outputs, gate report, no-leverage proofs, record commitment, frozen
gate checks, and failed-gate names. Recovery validates that exact row and
receipt, then reads only those already-bound projections. It may rehydrate a
fresh opaque acquisition report or terminal evidence, but performs no new SEC,
Yahoo, replay, metric, gate, ledger, no-leverage, or scoring work.

## Exact publish-or-recover

The exact publish-or-recover operation may continue after a process restart,
operating-system restart, new boot, or arbitrary downtime. Recovery reconstructs
only the tag fixed by the intent. If its remote ref is absent, it may push that
precomputed object once without force. If the ref exists, recovery performs
readback only.

Before any remote readback, each recovery invocation commits and externally
anchors a separate self-hashed start record. It fixes the store-session nonce,
ordinal, prior completion hash and cumulative time, `started` status, exact
300-second cap, and requirement for durable pre-push authorization. Only after
that start is fully committed may the invocation inspect the remote. The
pre-push marker key is then derived from the committed start hash and the other
frozen invocation identities; it is deliberately not part of the start record.

Every normal or recovery Git/SSH worker tree runs inside a dedicated Windows
Job Object with kill-on-close enabled. Children are created suspended, assigned
before resume, cannot break away, and the source-bound parent alone retains the
controlling handle. A deadline terminates the whole job; unexpected parent exit
closes the sole handle and makes the operating system kill every descendant.
Before a fresh post-restart capability is issued, the store must prove the
prior owner PID and creation identity dead, its exclusive mutex unowned, the
job absent or empty, and every recorded Git/SSH process dead. Thus an old
worker cannot overlap a new publisher.

After the committed start, the invocation performs a fresh exact remote
readback and first commits it as a self-hashed durable remote-observation row.
Its prepare anchor contains both the exact readback evidence bytes and exact
canonical row bytes. No readback may authorize a push, receipt, completion
field, or poison until that observation and both anchors are durable. The
readback evidence has a frozen field list covering the exact command profile,
command-sequence hash, process exit, stdout and stderr byte counts and hashes,
transport status, lookup status, raw normalized values, operation, phase,
ordinal, and prior push command. The profile is one exact raw-byte
`git ls-remote --tags` read of the stable ref and its peeled ref, with no shell
or alternate command ordering. Literal `REF_ABSENT`, `VALUE_MISSING`, and
`VALUE_MALFORMED` sentinels distinguish absence from transport failure. An
exhaustive evidence matrix permits absence only for exit status `exited`, exit
code zero, completed transport, empty stdout and stderr, and all three
`REF_ABSENT` values. Nonzero exit, deadline, interruption, spawn failure,
protocol error, or unknown lookup cannot create an observation or push marker.

Readback and push never use a mutable Git remote name. Each operation first
anchors an exact manifest for a fresh isolated bare Git directory with no
remotes, URL rewrites, includes, push URLs, proxies, or extra headers. System
and global Git config are disabled; the isolated config is allowlist-only and
contains frozen core, TLS, no-redirect, and absolute credential-helper entries
plus one exact object-directory alternate. `core.hooksPath` is fixed to a
manifest-bound empty directory inside the isolated transport root. Its empty
listing, identity, parent chain, and lack of files, links, junctions, reparse
points, alternate data streams, or executable entries are checked immediately
before every Git process. The push also uses frozen `--no-verify`, so a
`pre-push` hook is independently disabled even if the empty-directory check
were raced. The child process receives a newly constructed exact environment
mapping and inherits nothing from the parent.
All unlisted Git, SSH, askpass, SSL-override, and proxy variables are absent.
Proxy names are omitted instead of duplicated in upper and lower case because
Windows environment names are case-insensitive. This includes
`GIT_CONFIG_PARAMETERS`, `GIT_DIR`, `GIT_COMMON_DIR`, `GIT_WORK_TREE`,
`GIT_OBJECT_DIRECTORY`, and `GIT_ALTERNATE_OBJECT_DIRECTORIES`.

Both commands use a manifest-bound absolute Git executable and place the
intent's one allowed literal GitHub HTTPS `remote_url` directly in their argv.
The Git executable, exact HTTPS helper, exec-path directory, absolute
credential helper, command interpreter if used, complete
transport-executable closure, and exact child environment are hashed into the
durable isolation manifest and checked again before process creation. Path
lookup is forbidden for these executables. The credential helper config is an
exact shell snippet with its whitespace-containing absolute path quoted, and
the pinned command interpreter is checked before it can run. The top-level Git
process is always launched directly with shell disabled; that exact
credential-helper snippet is the only permitted internal shell use. HTTPS is
the only allowed protocol, and redirects are disabled. A changed executable,
helper, environment, `.git/config`, URL rewrite, or different effective
endpoint therefore cannot be reported as the frozen origin.

Each publication operation has exactly one of seven observation sequences: an
empty sequence when no completed readback permits an observation; pre-push
absent, exact, or conflict; or pre-push absent followed by post-push absent,
exact, or conflict. Pre-push is ordinal one and post-push is ordinal two. There
are no gaps, duplicates, extra observations, or other transitions. A post-push
observation binds the exact preceding push command.

If the durable pre-push observation proves the ref absent, the store must
commit and anchor a separate one-push authorization marker before issuing the
push. The same generic marker covers the normal publication operation and
recovery: normal binds its frozen operation sentinel and ordinal zero; recovery
binds its committed start hash and invocation ordinal. Both bind the exact
worker owner and observation. No normal or recovery push is legal without this
marker. No marker is created for an existing exact ref, conflict, unavailable
remote, or unanchored observation. A later recovery invocation may push the
same precomputed object only after its own new start, durable observation, and
marker.

Only an exact match of the tag object, peeled target, message, ref, remote,
contract, implementation, attempt, terminal status, report kind, artifact,
predecessor, and zero cost is accepted. Any conflicting or malformed tag
poisons the publication state and fails closed. It is never deleted,
force-updated, replaced, or reused as evidence, and the poison is irreversible.
The poison identifies whether the proof came from the normal publication
operation or a recovery invocation. A normal proof binds the frozen normal
operation sentinel and its committed worker-ownership record; a recovery proof
binds its exact committed start hash and matching committed worker-ownership
record. Thus a conflict proven during normal publication is preserved
immediately and never depends on a later recovery invocation.
“Proven” means the conflicting remote-observation row and both anchors are
already durable. If the process dies before the observation prepare anchor,
there is no durable proof to preserve. If it dies after the observation commits
but before poison preparation, restart reconstructs the poison from those
frozen bytes before any new remote action, even if the foreign ref disappeared.
Network, timeout, process, restart, or remote-unavailable failures without a
proven conflict preserve publication-pending for a later bounded invocation.

On every restart, each partial transport-isolation manifest is reconciled and
validated before partial observations, any fresh capability, or any remote
read. The exact sequence is then validated and extras poison. The first valid
terminal observation is authoritative. A committed recovery terminal
observation reconstructs its unique matching completion before any receipt,
with a conflict poison committed before a conflict completion; an exact or
conflicting normal observation reconstructs the matching receipt or poison
locally. A lone old pre-push absence is not reused by a new operation. Thus an
exact publication observed before a crash cannot be discarded or replaced by
a later read of a changed or deleted ref.

After remote work ends, a separate self-hashed completion record freezes the
outcome, final remote observation, push-command-count upper bound, elapsed
seconds, prior completion hash, and cumulative recovery time. Start and
completion are never conflated.
Its finite outcome table covers: unavailable before any observation; exact
existing publication; conflict without a push; absent ref without a committed
authorization; committed authorization followed by no push; one unconfirmed
push; confirmed post-push absence; exact publication after one push; conflict
after one push; and
interruption with or without authorization. The table fixes the authorization
hash or no-authorization sentinel, exact observation hash or no-observation
sentinel, push-command-count upper bound, conflict-poison requirement, and
receipt eligibility for every path. A second table binds every clean outcome to its exact observation
phase, ordinal, state, and complete sequence, so pre-push evidence can never be
misrepresented as post-push confirmation. No other outcome spelling or
cross-field combination is valid.
If an old-session start has no completion prepare anchor, reconciliation writes
one canonical completion with outcome `interrupted_before_completion`, the
frozen no-observation sentinel, exact binary64 hexadecimal elapsed value
`0x1.2c00000000000p+8` (300 seconds), and the correctly rounded exact cumulative
hex value. Normal recovery elapsed and cumulative values use the same lossless
lowercase binary64 hexadecimal encoding. Its push-command-count upper bound is
one if and only if
the exact durable pre-push marker exists; otherwise it is zero and uses the
frozen no-authorization sentinel. This interruption field is explicitly a
conservative upper bound, not a claim that the command certainly ran: a crash
can happen after marker commit but before command issuance. If
a completion prepare anchor exists, recovery uses its already-fixed exact row
instead of inventing interruption values. A conflict writes a separate
two-phase self-hashed poison record whose prepare anchor contains the exact
canonical poison row bytes. Once committed, poison remains authoritative even
if the foreign tag is later deleted.

After an exact-expected remote observation is durable, the external publication
is persisted through its own two-phase `publication_receipt_prepare`, exact
journal row, and
`publication_receipt_committed` sequence. Exact partial states reconcile
deterministically and idempotently because the receipt prepare anchor contains
the exact canonical receipt row bytes. The receipt binds the durable
remote-observation hash. A recovery receipt additionally binds and follows its
committed receipt-eligible completion; a normal receipt uses a frozen
no-recovery-completion sentinel. Pre-push exact confirmation binds the
no-authorization sentinel, while post-push confirmation binds the exact marker.
Terminal evidence cannot be issued until that durable receipt and its store
receipt exist. The publication capability is then invalidated and a fresh
store-nonce-bound terminalization capability is issued.

Terminalization reconstructs only the already sealed evidence. If a crash
leaves `terminal_intent` but no database transition, recovery appends that exact
predeclared transition and then `terminal_committed`. If the database transition
exists but `terminal_committed` does not, recovery validates it and appends only
that exact anchor.

The terminalization capability is single-use. Before the first
`terminal_intent`, the store atomically commits a unique self-hashed
terminalization claim that fixes the capability nonce, evidence, and terminal
status; committing the claim invalidates the capability. Concurrent or repeated
entry sees the existing claim and must use exact terminal recovery, never append
a second `terminal_intent`. If a crash leaves the committed claim but no first
`terminal_intent`, the explicit `claim_without_terminal_intent` recovery window
validates the frozen claim and evidence, then appends that one predeclared
intent exactly once without another capability or claim. Once terminal commit
is complete, restart may return the identical sealed result only through
read-only validation and rehydration from the terminal anchor and row, bound
artifact and receipt, reconstruction row and receipt, terminal evidence,
intent, and publication receipt. It performs no new publication, transition,
anchor, or research computation. Once publication intent exists, none of these
windows may become terminal-indeterminate.

Each non-effectful recovery invocation is supervised, killable, and limited to
300 seconds. It performs zero new SEC, Yahoo, model, Gemma, market acquisition,
feature, learner, replay, metric, gate, or scoring effects. Recovery elapsed
time and cumulative recovery time are reported separately and cannot be
described as part of the sub-hour effectful attempt.

## External pins and final registry

Every terminal pass and valid failed-gate diagnostic is externally pinned
through one non-force annotated Git tag at
`refs/tags/sec-gemma-online-risk-overlay-v2-2/attempts/{attempt_id}/terminal`.
The remote tag object and peeled implementation target are read back before the
terminal transition. Tags cannot be reused, deleted, or force-updated.

Before final registration or consumption, an opaque registry verifier validates
the frozen predecessor registry, appends exactly this final attempt, publishes
that successor through its existing exact mechanism, and returns its verified
remote receipt under
`refs/tags/sec-gemma-online-risk-overlay-v2-2/registry/{attempt_id}/successor`.
A caller-provided hexadecimal string is not authorization.

## Runtime and cost limits

Each normal acquisition or scored attempt remains independently below one hour.
The frozen maximum allocation is 720 seconds for SEC plus market acquisition,
2,160 seconds for Gemma, 480 seconds for deterministic work, and 239 seconds of
normal governance contingency, totalling 3,599 seconds. Paid APIs, model pulls,
research retries, fallbacks, leverage, and shorting remain forbidden.

The 239 seconds are partitioned without borrowing:

- 89 seconds for local canonical-tag and durable intent preparation;
- 90 seconds for supervised normal publication; and
- 60 seconds for terminal finalization or durable pending handoff.

All three partitions belong to the same original normal invocation. Its clock
continues after intent commit until the parent returns either the final sealed
result or the explicit durable no-semantic publication-pending receipt, always
strictly below 3,600 seconds. Only a later invocation is separately timed
recovery. If the whole host crashes after the intent prepare anchor, its exact
canonical row bytes and elapsed hex define deterministic reconciliation and the
pending boundary; the first post-boot work is reconciliation, prior worker-death
verification, and only then a fresh bounded recovery invocation.

No effectful attempt may be registered or consumed until the complete
production path, source inventory, clean pushed implementation, durable vault,
external publisher, final-registry authority, and parent deadline guard all
pass local preflight.
