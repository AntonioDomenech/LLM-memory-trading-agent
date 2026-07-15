# Rejected: AAPL causal contextual expert aggregation v2

## Decision

Rejected at the frozen 2005-2018 chronological development gate. The
2019-2023 confirmation and the 2024, 2025, and 2026-YTD final periods were not
opened, and confirmation is permanently unauthorized for this version.

This is a strong result against AAPL buy-and-hold but a failed learning
overlay. The causal online aggregator passed 35 of 36 gates, beat buy-and-hold
in nine of 14 years at stress cost, and beat it in all seven two-year folds and
all three negative-AAPL years. It nevertheless finished behind the simpler
fixed exhaustion-union comparator at both costs. The stress-cost deficit was
`0.07128550512648013` active log edge, far larger than rounding noise.

## Development result

All figures cover one continuous adjusted-open account from 2005 through 2018
and charge adverse transaction cost on every changing leg.

| Metric | Learner, 5 bps | Fixed union, 5 bps | Learner, 10 bps | Fixed union, 10 bps |
|---|---:|---:|---:|---:|
| Active log edge vs AAPL buy-and-hold | 1.010541 | 1.092826 | 0.900541 | 0.971826 |
| Wealth above AAPL buy-and-hold | +174.71% | +198.27% | +146.09% | +164.28% |
| Final equity from $1,000 | $106,514.17 | $115,649.44 | $95,371.36 | $102,418.14 |
| AAPL buy-and-hold final equity | $38,773.51 | $38,773.51 | $38,754.15 | $38,754.15 |
| Cash episodes | 110 | 121 | 110 | 121 |
| Winning cash episodes | 71 | 79 | 68 | 76 |
| Episode win rate | 64.55% | 65.29% | 61.82% | 62.81% |

At 5 bps, the learner had positive edge in 12 of 14 years; at 10 bps it
had positive edge in nine of 14. All seven fixed two-year folds were positive
at both costs. Its aggregate 10-bps edge in the negative-AAPL years 2008,
2015, and 2018 was `0.6700607309970282`. Removing its best year still left
`0.43690360981982007` active log edge at 10 bps. Its maximum drawdown was
43.95% at 10 bps versus 60.42% for buy-and-hold.

These diagnostics make the inherited exhaustion family a long-horizon
research lead. They do not prove a reliable money-making system: this version
failed its preregistered comparator gate, and the underlying experts were
identified during research that had already inspected later history.

## Exact failed comparison

The learner took 110 of the fixed union's 121 cash episodes. The 11 omitted
union episodes fully explain the stress-cost gap:

- Three omissions were beneficial and added `0.07014008245156149` learner
  edge.
- Eight omissions were harmful and removed `0.14142558757804058` learner
  edge.
- Net learner-minus-union edge was therefore
  `-0.0712855051264791`, matching the sealed gate result apart from floating
  representation.

The aggregator did learn causally from matured outcomes during development:
it processed 166 accepted opportunities, admitted 163 learning events, and
ended with no pending lesson. The problem was not a lack of action or a
look-ahead error. Its learned filtering skipped more profitable union trades
than losing ones. A successor must solve a materially different prediction
problem rather than retune this rejected selector.

## Integrity, verification, and runtime

- Branch: `codex/aapl-causal-contextual-expert-aggregation-v2`.
- Immutable implementation commit:
  `25609a514a9f36bbd09597a1b98ddd2fc753f1cd`.
- Stage command:
  `python -I -B agent_benchmark/contextual_expert_aggregation_bootstrap.py stage development`.
- Stage wall time: 400.374 seconds; sealed pre-finalization time: 390.922
  seconds, below the 3,600-second limit.
- Independent verifier command:
  `python -I -B agent_benchmark/contextual_expert_aggregation_bootstrap.py verify development`.
- Verifier wall time: 393.682 seconds; exit code 0; `verified: true`,
  `stage_pass: false`, and `status: REJECTED`.
- Physical development input: 4,986 sessions ending 2018-12-31. No 2019+
  market value was accessed.
- Long/cash only; minimum exposure 0.0, maximum exposure 1.0, minimum cash
  0.0; no leverage, shorting, borrowing, negative cash, or margin interest.
- Network access: false. News access: false. LLM calls: 0. API calls: 0.
  External cost: $0.00.
- Passed gates: 35 of 36. Sole failure:
  `stress_10bps.full_account_beats_best_fixed_by_gt_0_0001`.
- Manifest identity:
  `sha256:6cb657d1d9c323bd96a84b598b9b993e2444f756a65521aa1bf28207f7a63539`.
- Checkpoint identity:
  `sha256:e195356c0540a4a05ea635927fa466c1c85948beb7de0191e80f5bc10d60fd03`.
- Independent semantic evidence:
  `sha256:964ad44206d3a1cb481a330317217a2d910101b34e35240aa1186c8a6069e944`.
- All 48 checksummed payloads match their recorded file hashes.

The complete 49-file forecast, checkpoint, ledgers, episodes, diagnostics,
metrics, provenance, gate report, manifest, and checksums are preserved in
`contextual-expert-aggregation-development-v2/`.
