# Rejected: chronological exhaustion expert v1

## Decision

Rejected at the frozen 2005-2018 chronological development gate. The 2019-2023
validation and the 2024, 2025, and 2026-YTD final periods were not opened, and
no later-stage input snapshot was created.

The underlying exhaustion signals were strong in this historical window, but
the online learning layer failed its essential ablation test. It removed two
profitable early weak-trend trades and never added a compensating advantage
over taking the unfiltered union of both signals. This is underperformance,
not a rounding tie.

## Development result

All figures cover the continuous 2005-2018 adjusted-open ledger and include
the stated adverse transaction cost on every changing leg.

| Metric | Learner, 5 bps | Union, 5 bps | Learner, 10 bps | Union, 10 bps |
|---|---:|---:|---:|---:|
| Active log edge vs AAPL buy-and-hold | 1.055469 | 1.092826 | 0.936469 | 0.971826 |
| Wealth above AAPL buy-and-hold | +187.33% | +198.27% | +155.10% | +164.28% |
| Final equity from $1,000 | $111,408.82 | $115,649.44 | $98,860.21 | $102,418.14 |
| AAPL buy-and-hold final equity | $38,773.51 | $38,773.51 | $38,754.15 | $38,754.15 |
| Cash episodes | 119 | 121 | 119 | 121 |
| Winning cash episodes | 77 | 79 | 74 | 76 |
| Episode win rate | 64.71% | 65.29% | 62.18% | 62.81% |

At 10 bps, the learner beat buy-and-hold in 10 of 14 calendar years and all
seven fixed two-year folds. Its edge was positive in all three negative-AAPL
years: 2008, 2015, and 2018. Removing the best year, 2008, still left 0.457750
active log edge. These results make the underlying signal combination a
research lead, but do not rescue a learning layer that was preregistered to
beat that simpler combination.

## Exact failed ablation

Every learner cash episode was also a union episode. The union made exactly
two additional weak-trend trades, both profitable:

| Decision close | Cash interval | Net edge, 5 bps | Net edge, 10 bps | Why the learner stayed long |
|---|---|---:|---:|---|
| 2005-05-04 | 2005-05-05 to 2005-05-06 | 0.008711 | 0.007711 | Lower confidence and lower edge-score gates had not matured |
| 2006-06-01 | 2006-06-02 to 2006-06-05 | 0.028646 | 0.027646 | Lower confidence and lower edge-score gates had not matured |

Their combined 10-bps edge is `0.03535704896217844`, which explains the full
gap between union edge `0.9718261388903109` and learner edge
`0.9364690899281323`, apart from floating-point representation. The learner
therefore filtered out two winners and otherwise reproduced the union.

## Interpretation

This result does not show that an adaptive system is impossible. It shows that
this particular posterior trust gate was redundant after warm-up and harmful
during warm-up. A successor should learn a genuinely different decision, such
as which expert is preferable in a predeclared market regime, and must beat
both the union and each individual expert on untouched chronological blocks.

The contextual and weak-trend experts were discovered during earlier research
that had already inspected later historical periods. Therefore even their very
strong 2005-2018 result is supporting historical evidence, not pristine
prospective proof.

## Integrity and runtime

- Branch: `codex/aapl-chronological-exhaustion-expert-v1`.
- Immutable implementation commit: `ed93d44cc0180b71a4f4f07801c207ba566934f1`.
- Physical input: 4,986 rows from 1999-03-10 through 2018-12-31; no later rows.
- Long/cash only; maximum exposure 1.0; no leverage, shorting, borrowing, or
  negative cash.
- LLM calls: 0. API calls: 0. Estimated external cost: $0.00.
- Sealed runtime: 6.27 seconds, below the 3,600-second limit.
- Development gates passed: 22 of 23. Sole failure:
  `learner_strictly_beats_union_at_10bps`.
- Manifest identity:
  `sha256:d6352a96629008a97b54d56462d9549c5b59cf0d43bbbe621d602c2d9a7b4d94`.
- Canonical bounded-price hash:
  `sha256:31b56551b8d1b837f2e69178ab7f206b6bf3c19d11d9d47be0e33cf501db3f45`.
- All 13 checksummed payloads verified after the run.

The complete forecast, checkpoint, ledgers, episodes, metrics, provenance,
gate report, manifest, and checksums are preserved in
`exhaustion-expert-development-v1/`.
