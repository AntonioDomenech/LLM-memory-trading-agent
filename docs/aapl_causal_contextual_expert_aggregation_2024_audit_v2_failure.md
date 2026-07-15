# Frozen-policy 2024 audit v2: failed closed

Date: 2026-07-16

Branch: `codex/aapl-causal-contextual-expert-aggregation-2024-audit-v2`

Preregistration commit: `2f4537440eccb41c62ab553d2ffb76460296cc9d`

Execution commit: `b0aff9ebd6e07f6ec38937a8c38f227d8911df74`

## Decision

This one-shot audit is permanently closed and is not a certified result. The
attempt lock was consumed, the private bundle was moved to the exact failed
directory, no final bundle was promoted, and no success marker exists. The
contract must never be rerun against 2024.

The forensic bundle also contains an independently checked provisional
economic rejection. Even if the runtime-sealing defect had not occurred, the
frozen policy would have failed the 2024 gate and the online learner would not
have qualified for a 2025 shadow. This approach does not authorize real
capital or access to a later audit period.

## Execution failure

The authorized command was:

`python -I -B agent_benchmark/contextual_expert_aggregation_2024_audit_bootstrap.py stage audit_2024`

It exited 1 after 570.742 seconds. The provisional private bundle had already
passed `verify_private_audit_bundle`. Immediately afterward the runner sampled
`post_private_verify` and `prepromotion` back-to-back. Its strict sequence check
raised:

`ContextualExpertAggregation2024AuditError: runtime samples are not strictly increasing`

This machine's Python reports `time.monotonic` as Windows `GetTickCount64()`
with 0.015625-second resolution. Two immediate calls were observed to return
the same value. The runner incorrectly treated equality from that valid,
coarse monotonic clock as a fatal ordering failure.

The required standalone command was then run read-only:

`python -I -B agent_benchmark/contextual_expert_aggregation_2024_audit_bootstrap.py verify audit_2024`

It exited 1 after 7.483 seconds because the final bundle path is absent. This
correctly prevents the failed forensic directory from being mistaken for a
certified result.

## Forensic preservation

- Attempt-lock SHA-256: `e0c559d04fd456c49cfb3fd2dede2cb0a0d63ff6b1478624c38c769e11e9556c`
- The root and failed-bundle copies of the attempt lock are byte-identical.
- Failed directory: `e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/runs/.failed-contextual-expert-aggregation-frozen-policy-2024-audit-v2`
- It contains 43 files totaling 29,548,890 bytes.
- All 42 entries in `checksums.json` match their current file bytes; the 43rd
  file is `checksums.json` itself.
- No final directory, pending marker, or success marker exists.

## Provisional economic diagnostic

These numbers are preserved diagnostics from the failed bundle, not a sealed
success. The provisional bundle passed all 58 economic/integrity checks that
precede the runtime-sealing protocol, and its own gate status is
`REJECTED_2024`.

| Cost assumption | Frozen policy | Online learner | AAPL buy-and-hold | Frozen minus buy-and-hold |
|---|---:|---:|---:|---:|
| 5 bps per changing leg | +27.4184% | +27.4184% | +30.8286% | -3.4102 pp |
| 10 bps per changing leg | +26.7829% | +26.7829% | +30.8286% | -4.0457 pp |

At 5 bps the active log edge was `-0.026411838298386864`; at 10 bps it
was `-0.031411841215055664`. The policy made five complete one-session cash
episodes, or ten changing-leg trades. One episode helped and four hurt. Their
5-bps net active log edges were:

| Decision date | Net active log edge |
|---|---:|
| 2024-04-24 | -0.0030623494 |
| 2024-08-05 | -0.0087631969 |
| 2024-08-07 | +0.0037505702 |
| 2024-08-09 | -0.0145148820 |
| 2024-12-20 | -0.0038219802 |

The causal online shadow admitted all five newly matured 2024 lessons, but its
action-stream hash and both cost-level ledger hashes were exactly identical to
the frozen-2023 policy. Therefore the new lessons changed no trade, generated
zero incremental edge, and were classified as `unexercised`. The learning
candidate gate failed 0/4 criteria.

## What this means

The historical contextual/exhaustion union remains an interesting long-run
diagnostic, but it again failed the known 2024 test. More importantly, the
current online aggregation mechanism updates internal state without moving a
decision threshold. A future approach must demonstrate, on chronological
pre-test folds, that learning changes some actions and improves the decision
to enter cash. It should not advance merely because its inherited fixed rule
has strong long-run historical returns.

Any future one-shot contract must also replace the invalid adjacent-sample
assumption: either use a genuinely high-resolution monotonic clock such as
`time.perf_counter`, wait for a new coarse-clock tick before recording a new
named phase, or accept nondecreasing timestamps while separately proving phase
order. That repair cannot retroactively certify or reopen this 2024 attempt.
