# Frozen direct CASH-edge v1: rejected in development

This branch is preserved as a complete rejected approach. It must not be
retuned from these results and it must not advance to 2019-2023 or the frozen
2024+ test.

## Evidence identity

- Branch: `codex/aapl-frozen-direct-edge-v1`
- Frozen source commit: `ad8a1027e75598ba68435c0d87cde6304ec1627a`
- Artifact: `direct-edge-development-20260710T214648Z-642bf330`
- Development decisions: 2005-2018 only
- Selected candidate: `null`
- Post-2018 market data accessed: `false`
- Model/API calls: `0`
- External cost: `$0.00`
- Runtime before sealing: `16.879951299983077` seconds
- Complete command wall time, including replay verification: about 34 seconds
- Artifact files: 45
- Checksum entries: 44
- Checksum manifest SHA-256:
  `ff902561436274431bbd303948e940ce93d4bd19eb1ac099277390061beafe32`

The verifier rebuilds features, all 14 chronological model fits, candidate
triggers, five-row CASH blocks, all 34 ledgers, costs, metrics, gates,
sentiment ablations, and the null selection from the sealed bounded inputs.

## Result

All eight frozen candidates failed at both 5 and 10 bps. The least-bad
10-bps candidate was `price_only_p50_e25`, and it was still decisively worse
than same-ledger AAPL buy-and-hold:

- total active log edge: `-1.0645503154283888`;
- terminal wealth relative to buy-and-hold: `-65.51170915750805%`;
- winning years: `4/14`;
- positive frozen folds: `2/7`;
- CASH days: `865`;
- continuous CASH episodes: `69`;
- separately scored five-session predictions: `173`;
- prediction-block mean realized 10-bps edge: `-0.7355782570877191%`;
- prediction-block win rate: `39.88439306358382%`;
- 252-session rolling win rate: `21.019108280254778%`;
- 756-session rolling win rate: `14.393939393939395%`.

Its probability forecast was 12.70% worse than the causal training-only
baseline by relative Brier score, and its expected-edge forecast was 21.08%
worse than the causal mean forecast. The market-sentiment family was worse
again: roughly 19.09% worse on probability and 58.84% worse on edge error,
and every sentiment candidate failed the paired common-support ablation.

The least-bad model did add positive aggregate edge in AAPL's negative
buy-and-hold years, but it went to CASH far too often and missed much more
wealth in normal and rising periods. That is not a general market predictor
and cannot satisfy the goal.

## Decision

Reject the five-session direct GAM hypothesis. Do not invert its predictions,
change thresholds, or reinterpret the best failure after seeing these data.
The next branch must test the predeclared one-session regime-consensus design,
which is materially different and limits the damage of a wrong CASH forecast.
