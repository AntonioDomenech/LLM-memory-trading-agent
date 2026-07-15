# AAPL contextual aggregation: physically bounded through-2024 input

This record freezes the deterministic data-only preparation performed before
any 2024 policy implementation or scoring.

The source was the already tracked six-column AAPL/SPY/QQQ snapshot used by the
independently sealed binary-regime long-run audit. Its exact Git blob and full
raw SHA-256 were verified before copying. The preparation process wrote only
the header and first 6,496 increasing sessions, ending on 2024-12-31, to a new
UTF-8 no-BOM file with LF line endings. It did not run the trading policy,
calculate a return, change a threshold, call a model or API, or print a market
value.

The new snapshot is physically bounded: it contains no row after 2024. Its
canonical first 6,244 sessions reproduce the sealed through-2023 parent hash
exactly. An independent preregistration review loaded only this bounded file,
rechecked its raw, canonical, date-sequence, schema, physical-bound, and Git
identities, and regenerated the exact through-2023 prefix and checkpoint. It
did not invoke policy or scoring code and printed no market value.

The audit-safe machine-readable receipt is
`e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/input_snapshot_receipt.json`.
It contains no later path or later date. The complete preparation-only lineage
is quarantined at
`docs/aapl_causal_contextual_expert_aggregation_2024_input_preparation_quarantine.json`.
That full record is documentary evidence only and is not an input to stage or
verification code.

The 2024 audit implementation must reference only the bounded path and the
sanitized receipt. Policy, evaluation, bootstrap, runner, and verifier code
must not name, enumerate, hash, import, or open the longer source, the full
preparation record, or any prior 2024 forecast, action, ledger, or result.
Before the durable one-shot attempt lock, stage code may inspect only the
bounded input and receipt's committed HEAD/index object identities. The stage
attempt's first worktree-byte read of either is permitted only after its lock
is durable; the data-only preregistration review above is explicitly outside
that future attempt.
