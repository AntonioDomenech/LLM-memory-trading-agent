# AAPL SEC/Gemma online risk overlay v2.1 preflight rejection

## Status

The preregistered `v2.1` branch is rejected during local execution-integrity
review. No official SEC document or Yahoo market snapshot was acquired for
this approach, Gemma performed no filing extraction, no return was opened, and
no stage was scored.

This is not a trading result. It is a fail-closed protocol finding that must
remain separate from any successor approach.

## What was repaired successfully

The local implementation work closed several earlier preflight gaps:

- acquisition and deterministic work run under killable parent deadlines;
- the private SEC identity remains memory-only and never enters command-line
  arguments, environment variables, logs, or files;
- Requests is isolated from unpinned optional dependencies;
- market, SEC, model, deterministic, store, and source identities are checked
  before effectful access;
- chronological learning and stage boundaries remain causal;
- the exact inherited NumPy learner is retained; and
- long/cash-only accounting and no-leverage proofs remain mandatory.

These repairs are transferable implementation work, but they do not make the
v2.1 terminal protocol safe.

## What failed

The frozen protocol requires a pass or valid failed-gate report to be pushed
as one immutable annotated Git tag and read back before the local terminal
transition. It also requires the complete attempt, including publication,
local terminal anchoring, and sealed-result construction, to finish before the
strict deadline.

That ordering leaves a non-atomic interval:

1. the remote tag can be created successfully;
2. the process can then exceed its deadline, crash, or lose its publisher
   receipt before the local terminal anchor is committed; and
3. reopening the consumed attempt must classify it as
   `terminal_indeterminate`.

The result is contradictory durable evidence: an immutable remote tag claims
`terminal_pass` or `terminal_fail`, while the local one-shot state claims
`terminal_indeterminate`. The tag cannot be deleted, force-updated, or safely
reused under the frozen contract.

A shorter publication deadline and a reconciliation reserve reduce this
window but do not remove it. A crash-safe repair needs an explicitly
preregistered terminal-commit authority and exact restart-recovery rule. That
changes the frozen v2.1 treatment of interrupted or indeterminate execution,
so it cannot be introduced silently as an implementation detail.

## Consequence

1. The v2.1 branch remains preserved as a local preflight rejection.
2. None of its four one-shot effectful attempts has been consumed.
3. No 2019-2023 confirmation data or 2024-2026 final result has been opened.
4. A successor approach must preregister the publication-intent, authoritative
   terminal-commit, exact recovery, deadline-reservation, and fail-closed
   split-brain rules before any SEC, market-value, or Gemma effect.

The successor may reuse the verified acquisition, dependency, chronology,
learner, ledger, and source-closure work, but it requires its own branch,
contract hash, attempt identities, state namespace, tag namespace, and pushed
implementation binding.
