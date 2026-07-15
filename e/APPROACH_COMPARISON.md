# AAPL long/cash approach comparison

Updated through the SEC numeric event-drift preflight on 2026-07-15.
All eligible approaches use 0% or 100% AAPL exposure, no shorting, no leverage,
no negative cash, the same adjusted-open ledger as AAPL buy-and-hold, and no
paid API. Published 2024-2026 figures are repeated historical audits, not a
fresh prospective track record.

## Approaches with published 2024-2026 audits

Excess figures below are percentage points versus same-ledger AAPL
buy-and-hold at 5 bps per changing leg.

| Approach | Learning/model | Pre-2024 evidence | 2024 excess | 2025 excess | 2026 YTD excess | Continuous relative wealth | Decision |
|---|---|---|---:|---:|---:|---:|---|
| Contextual exhaustion v1 | Fixed sparse rule | Won 21/24 years; selected retrospectively | -3.5314 | +12.8282 | +2.4794 | +11.0982% | Strictly rejected; promising but not cleanly selected |
| Gap-down cash v1 | Fixed sparse rule | Won 15/24 years | -1.1820 | -1.0489 | 0.0000 | -1.8166% | Rejected |
| Exhaustion-or-gap v1 | Fixed combined rule | Won 20/24 years; selected retrospectively | -3.5314 | +2.3721 | +2.4794 | +1.6510% | Strictly rejected; modest long-run lead |
| Weak-trend exhaustion v1 | 675-rule retrospective search | Won 16/24 years; +4.6839 pp mean annual excess | -3.5220 | +17.5368 | +1.5863 | +14.4778% | Strictly rejected; strongest lead, high selection-risk |
| Sparse dual-trend exhaustion v1 | Sparse pre-2024 search | Won 11/24 years with many ties | -0.9844 | +5.0895 | 0.0000 | +3.8610% | Rejected; too inactive |
| Hierarchical empirical-Bayes v1 | Genuine causal online learner | Broader 2000-2023 audit won 7 years; -0.2916 pp mean annual excess | -1.1820 | +8.7224 | +0.1518 | +7.1204% | Rejected; real 2025 edge but poor general reliability |

The contextual and weak-trend rules have more historical winning years than
losing years and positive continuous 2024-2026 relative wealth. They are
research leads under the relaxed long-run-average criterion, but neither is a
reliable money-making proof: their parameters were chosen after extensive
pre-2024 searching, and both still lost the already-revealed 2024 audit.

## Chronological development approaches that did not reach later periods

| Approach | Model/information | Development result | Later data opened? | Decision |
|---|---|---|---|---|
| CFTC COT sentiment v1 | CFTC positioning z-scores | Best variant retained 51.48% of buy-and-hold wealth; every fold lost | No | Rejected |
| Downside ensemble v1 | Deterministic downside forest | Best candidate gained +25.96% relative wealth, but only 3 episodes, 2/14 winning years and 2/7 positive folds | No | Rejected; positive but extremely concentrated |
| Direct CASH-edge GAM v1 | Price plus IWM/VIX/TNX | Least-bad 10-bps edge -1.0646; 39.88% episode win rate | No | Rejected |
| One-session regime consensus v1 | Local HMM/state consensus | Zero cash trades; exact tie | No | Rejected |
| One-session rare-loss forest v1 | Rare-event forest with IWM/VIX | Zero cash trades; exact tie; sentiment worsened accuracy | No | Rejected |
| Sector-breadth residual edge v1 | Sector ETFs, IWM, VIX and residual momentum GAM | Full candidate: -0.8951 edge and -59.15% relative wealth. Post-rejection p99 tail: +9.82% at 10 bps, but only 3/7 folds and 4/14 years won | No | Rejected; tail retained only as an ensemble lead |
| SEC numeric event drift v1 | AAPL 10-Q/10-K changes plus QQQ/VIX sentiment | Preflight found only 38 events from 2009-2018; after warm-up, 24 predictions and five helpful cash episodes | No | Rejected before implementation; too small for a reliable claim |

## Outside the current no-leverage contract

The earlier `trend_regime_tilt_v1` beat buy-and-hold in 2024, 2025 and 2026
YTD in its base audit, but requested 1.10 exposure, borrowed cash, and paid
margin interest. It is disqualified and is not evidence that the current
long/cash goal has been achieved. The local Gemma trinary runs also allowed
short behavior and are excluded from the eligible table.

## Evidence locations

- `codex/aapl-contextual-exhaustion-no-leverage-v1:docs/experiments/contextual_exhaustion_no_leverage_v1/RESULT.md`
- `codex/aapl-gap-down-cash-no-leverage-v1:docs/experiments/gap_down_cash_no_leverage_v1/RESULT.md`
- `codex/aapl-exhaustion-or-gap-no-leverage-v1:docs/experiments/exhaustion_or_gap_no_leverage_v1/RESULT.md`
- `codex/aapl-weak-trend-exhaustion-no-leverage-v1:docs/experiments/weak_trend_exhaustion_no_leverage_v1/RESULT.md`
- `codex/aapl-sparse-dual-trend-exhaustion-v1:docs/experiments/sparse_dual_trend_exhaustion_v1/RESULT.md`
- `e/bayes_v1/RESULT.md`
- `e/cftc_cot_v1/REJECTED.md`
- `e/downside_ensemble_v1/REJECTED.md`
- `e/direct_edge_v1/REJECTED.md`
- `e/regime_consensus_v1/REJECTED.md`
- `e/rare_loss_forest_v1/REJECTED.md`
- `e/aapl_sector_breadth_residual_edge_v1/REJECTED.md`
- `e/aapl_sector_breadth_residual_edge_v1/SPARSE_TAIL_DIAGNOSTIC.md`
- `e/sec_numeric_event_drift_v1/PREFLIGHT_REJECTED.md`
