# AAPL causal contextual expert aggregation v2

Status: preregistered parent-proof repair only. No v2 historical score has
been produced. This document does not authorize a 2024+ audit, paper trading,
broker execution, or real capital.

## Incorporated contract and reason for v2

Except for the explicit overrides below, v2 incorporates
`docs/aapl_causal_contextual_expert_aggregation_v1.md` in full. The v1 model,
data boundaries, causal timing, learning equations, advice, actions, costs,
ledgers, development and confirmation gates, runtime limit, sealing rules,
and evidence labels remain unchanged.

V1 made one development attempt at pushed commit `d78c4e5`. It stopped before
ledger construction or performance scoring because its parent-prefix proof
incorrectly required the v1 union-gated contextual-only and weak-trend-only
comparator targets to equal legacy parent columns whose individual policies
could act outside the accepted union-opportunity stream. Across 4,986 rows
and 23 projected fields, every date and all 19 causal-signal fields matched;
only three individual-comparator targets differed. V1 produced no score,
checkpoint, manifest, completed output, confirmation access, or 2019+ data
access. Its permanent tombstone is
`e/aapl_causal_contextual_expert_aggregation_v1/PREFLIGHT_REJECTED.md`.

Changing an integrity proof after that attempt is a new version even though
no return was observed. V1 is permanently closed and must never be rerun.

## Frozen v2 identity

- Contract version: `aapl-causal-contextual-expert-aggregation-v2`
- Branch: `codex/aapl-causal-contextual-expert-aggregation-v2`
- Development run ID: `contextual-expert-aggregation-development-v2`
- Confirmation run ID: `contextual-expert-aggregation-confirmation-v2`
- Output parent: `e/aapl_causal_contextual_expert_aggregation_v2`
- Any later repeated-audit branch, if authorized, must be
  `codex/aapl-causal-contextual-expert-aggregation-audit-v2`

The inherited chronological-exhaustion and binary-regime parent bundle
identities remain their exact v1 identities. The model method, replay hash
domains, ledger genesis, and isolated bootstrap protocol also remain v1
identities because their formulas and protocols do not change.

## The only methodology change: parent-proof semantics

The sealed parent forecast remains fully checksum-verified. V2 then projects
its exact dates and the 19 fields in
`FIXED_CAUSAL_SIGNAL_COLUMNS`, including both experts' raw and canonical
virtual signals plus the accepted union-cooldown stream. The regenerated v2
projection must equal that parent causal projection byte for byte. Any causal
field, date, row-count, order, type, finite-value, or serialization difference
aborts the stage.

Legacy parent comparator target columns are not current-model truth and are
excluded from exact pass/fail equality. They may be summarized only as
diagnostic provenance; they must never overwrite, seed, tune, or select a v2
action, gate, threshold, or model state.

V2 derives its four comparator targets solely from the already-proved causal
frame, exactly as v1 was preregistered:

- always-long is always 100% AAPL;
- union-cash is CASH if and only if the accepted union signal is true;
- contextual-only is CASH if and only if both the accepted union signal and
  the contextual virtual signal are true; and
- weak-trend-only is CASH if and only if both the accepted union signal and
  the weak-trend virtual signal are true.

The proof must byte-check the generated comparator table against that direct
derivation and record explicit invariants proving that neither individual
comparator can act outside the accepted union stream. Copying the legacy
independent-expert targets would change the policy and is forbidden.

The independent verifier must reconstruct the same causal projection,
derivation, hashes, and invariants from the sealed inputs. The proof schema
and verifier identity are v2. No other model or evaluation behavior may
change.

## Attempt and data discipline

The v2 contract must be committed and pushed before its implementation is
changed. The completed v2 implementation and tests must then be committed and
pushed before the one v2 development attempt begins. That attempt may open
only the already authorized physically bounded through-2018 input and its
development parent bundle.

A technical or integrity failure after dispatch permanently ends v2. A clean
development gate rejection is sealed and permanently ends v2. Confirmation
may open the first 2019 value only if the exact passing development bundle has
been independently verified, committed, and pushed and the one canonical v2
confirmation lock has been consumed. Any confirmation result is final.

No v2 command may use news, Gemma, Ollama, a network request, a paid API, an
external model, leverage, shorting, borrowing, negative cash, or exposure
outside `[0,1]`. All claims remain subject to the evidence limitations and
prospective-paper-trading requirement in the incorporated v1 contract.
