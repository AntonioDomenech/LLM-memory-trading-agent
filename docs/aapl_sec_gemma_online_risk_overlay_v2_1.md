# AAPL SEC/Gemma online risk overlay v2.1

## Status

This is an unrun preregistration on
`codex/aapl-sec-gemma-online-risk-overlay-v2-1`.

The rejected v2 branch remains preserved separately. Its local Gemma content
matched, but its raw `/api/show` byte pin changed only because Ollama changed
the `modified_at` value. No SEC acquisition, Gemma extraction, market-value
read, prediction, or score occurred on v2.

v2.1 keeps the same trading thesis, chronology, features, learner, policy,
costs, no-leverage rules, and success gates. It changes only runtime identity
and execution-integrity controls. Any later change to the trading rule or
success thresholds requires another branch and preregistration.

The complete machine-readable contract is
`agent_benchmark/sec_gemma_online_risk_overlay_contract.py`, whose literal
manifest SHA-256 is
`913a743495d4c92025110cf8af036bfce3a73dfa67f324de93e1cee0be3e9dfa`.

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

## Corrected local Gemma identity

The model, manifest, config digest, ordered four layer digests, Ollama version,
prompt, schema, temperature, seed, context, and output limits remain unchanged.

v2.1 requires strict JSON and the exact `/api/show` top-level key set. It
requires `modified_at` to exist as a string, removes only that field, then
canonicalizes and hashes the entire remaining object. The semantic show hash is
`5ccdf8b9a40bb762ea998dc9f5691ab2855d96833d60940b1d93686d08eb47e6`.
Whitespace, JSON key order, and the timestamp value alone may vary. Every other
semantic mutation fails. The raw response hash is recorded as diagnostic
evidence but is not an identity gate.

The complete runtime fingerprint is
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

The contract freezes the exact role-to-path inventory for acquisition, attempt
control, features, learner, ledger, metrics, no-leverage verification, replay,
runner, runtime, source verification, store, production executor, publisher,
registry, and vault. It also pins the inherited SEC transport and point-in-time
parser sources. Production imports must be dependency-closed; any allowed
external HTTP distribution is identified and hashed before registration.

## Exact pass and failure evidence

An acquisition may pass only through an opaque verified acquisition report
whose seven exact digest-valued checks replay raw bytes, receipts, attempt
scope, private identity handling, market-prefix continuity, blinded Gemma
requests, and all request/byte/retry/redirect caps. An arbitrary all-true
mapping cannot pass.

The contract freezes the exact fields for the acquisition validation,
acquisition terminal evidence, scored terminal evidence, external publication
receipt, final-registry successor, and final-registry authorization. Bare
hashes, strings, mappings, and old v2 artifacts cannot substitute for those
opaque v2.1 objects.

A scored stage may pass only after deterministic reconstruction of the complete
chronological replay, stage metrics, the exact literal gate key set,
independent no-leverage proofs, the current store record commitment, and the
joint report. The terminal report must bind an artifact receipt and payload
hash that actually exist in the append-only store.

If a valid run fails a performance gate, that is still an important result.
The complete evaluation, metrics, gate report, no-leverage proofs, and joint
report are stored and externally pinned. Only then does the attempt become
terminal-fail and release its sealed diagnostic. It cannot be hidden, retried,
converted to pass, or used to open the next stage.

## External pins and final registry

Every terminal pass and every valid failed-gate diagnostic is pinned through
one non-force annotated Git tag pushed to the frozen origin under
`refs/tags/sec-gemma-online-risk-overlay-v2-1/{attempt_id}/{report_kind}/{artifact_sha256}`.
The canonical tag message
binds the contract, clean pushed implementation commit, attempt, terminal
status, report kind, report SHA-256, predecessor pin, and zero external cost.
The remote tag object and peeled implementation target are read back before
the terminal transition. Tags cannot be reused, deleted, or force-updated by
the experiment.

Before final registration or consumption, an opaque registry verifier must
validate the frozen predecessor registry, append exactly this final attempt,
publish that successor through the same external mechanism, and return its
verified remote receipt. A caller-provided hexadecimal string is not
authorization.

## Runtime and cost limits

Each acquisition or scored attempt remains independently below one hour. The
frozen maximum allocation is 720 seconds for SEC plus market acquisition,
2,160 seconds for Gemma, 480 seconds for deterministic work, and 239 seconds of
contingency, totalling 3,599 seconds. Paid APIs, model pulls, retries,
fallbacks, leverage, and shorting remain forbidden.

The parent clock starts before final-registry authorization or attempt
registration, whichever is earlier. It continues through source checks,
consumption, every worker phase, replay, metrics, gates, no-leverage proofs,
store writes, Git publication and remote readback, terminal anchoring, and
sealed-result construction. Publication and governance therefore cannot escape
the one-hour limit.

No effectful attempt may be registered or consumed until the complete
production path, source inventory, clean pushed implementation, durable vault,
external publisher, final-registry authority, and parent deadline guard all
pass local preflight.
