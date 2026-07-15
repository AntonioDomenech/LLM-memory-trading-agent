# AAPL causal contextual expert aggregation v1: preflight rejection

Status: permanently rejected before performance scoring on 2026-07-15.

## Attempt identity

- Branch: `codex/aapl-causal-contextual-expert-aggregation-v1`
- Commit and upstream: `d78c4e51c2eaeb13a22a6cc4925484a48cb30164`
- Command: `python -I -B agent_benchmark/contextual_expert_aggregation_bootstrap.py stage development`
- Exit code: `1`
- Wall time: `13.110` seconds
- Last completed phase: causal replay of the authorized through-2018 input
- Failing phase: exact fixed-parent prefix proof, before ledger construction or performance gates

The run opened only the authorized 4,986-session development input ending
2018-12-31 and the sealed through-2018 parent development bundle. It did not
open the through-2023 confirmation input or any 2024+ value. It made no
network, news, LLM, Ollama, or API call.

No final development directory, sealing directory, manifest, checkpoint,
ledger, score, or performance result was produced. The only directory created
was the ordinary parent directory containing this tombstone.

## Exact failure evidence

The regenerated and sealed projections compared 4,986 rows across 23 fields,
or 114,678 cells. All 19 causal-signal fields, dates, the always-long target,
and the union target matched exactly. Three individual-comparator targets did
not:

| Date | Zero-based row | Field | Generated | Sealed parent |
|---|---:|---|---:|---:|
| 2006-02-15 | 1745 | `fixed_contextual_only_target_exposure` | 1.0 | 0.0 |
| 2010-06-11 | 2832 | `fixed_contextual_only_target_exposure` | 1.0 | 0.0 |
| 2014-09-29 | 3914 | `fixed_weak_trend_only_target_exposure` | 1.0 | 0.0 |

Evidence hashes:

- Sealed parent forecast payload: `sha256:83e64c07082b241f717b7e4bab923f6f69150e2a6ac4c2a2e9e54386e037e345`
- Sealed parent projection: `sha256:dae3c22a49cdb701873081252e6718b6026e53185e141ea271c3ffb871fd5b40`
- Regenerated v1 projection: `sha256:83359c0690655869baac857b7c94990faa5f0d683f2200df9142311ea877fd08`

The cause is a contract mismatch in the proof, not price drift, look-ahead,
rounding, or changed causal signals. The v1 trading contract requires the
contextual-only and weak-trend-only comparators to act only on the accepted
union-opportunity stream. The legacy parent target columns instead allow each
individual comparator to act independently when the union cooldown suppresses
that date. The v1 implementation correctly retained the preregistered
union-gated policy semantics, but its prefix proof incorrectly required those
generated targets to equal the semantically different legacy columns.

## Disposition

V1 is closed and must never be rerun. It has no performance result and cannot
authorize confirmation. A v2 branch may repair only the proof contract:
preserve exact parent causal-signal and accepted-union evidence, derive the
unchanged union-gated comparator targets from those signals, and prove their
invariants separately. Copying the legacy independent targets into the model
would change the preregistered policy and is not an acceptable repair.
