# One-session regime consensus v1: rejected in development

This branch preserves the complete rejected regime-consensus approach. It
must not be retuned from these results and it must not advance to 2019-2023 or
the frozen 2024+ test.

## Evidence identity

- Branch: `codex/aapl-one-session-regime-consensus-v1`
- Frozen source commit: `9514103b3b39117b6c3f8c93991aa62fea33df12`
- Artifact: `regime-consensus-development-20260710T223112Z-f3d4f8da`
- Entry-fill years: 2005-2018 only
- Frozen HMM fits: 21 local numeric fits
- Selected candidate: `null`
- Post-2018 market data accessed: `false`
- LLM/API/network calls: `0`
- External cost: `$0.00`
- Runtime before sealing: `89.84803559991997` seconds
- Complete command wall time, including independent replay: about 179 seconds
- Artifact files: 22
- Checksum entries: 21
- Checksum manifest SHA-256:
  `ffaaa2167c87a4de620d1dbd7bb3008dba4ac30ed286e2ca58383c44bc449deb`

The verifier rebuilds the features, 21 HMM fits, predictions, candidates,
every strategy and benchmark ledger, exact costs, metrics, gates, AAPL-only
ablation, and null selection from the sealed bounded inputs.

## Result

Both frozen candidates (`p55_e5` and `p60_e5`) made zero CASH trades. They
therefore tied buy-and-hold exactly rather than beating it:

- total active log edge at 5 bps: `0.0`;
- total active log edge at 10 bps: `0.0`;
- relative terminal wealth versus buy-and-hold: `0.0%`;
- CASH days and completed episodes: `0`;
- positive years and folds: `0/14` and `0/7` under the strict positive-edge
  definition.

The probability forecast was slightly better than its causal training-only
baseline (`+0.42%` relative Brier improvement), and the expected-edge forecast
was only `+0.11%` better than the causal mean. Both are far below the frozen
`2%` and `1%` predictive gates.

This was not a near miss caused by the two thresholds. Across all 3,523
all-head-ready decisions:

- consensus CASH-win probability ranged from `0.405521019404584` to
  `0.481581212445249`;
- predicted CASH edge ranged from `-0.00401983918719` to
  `-0.000766211542490691`;
- neither the market-state nor sentiment-state head predicted positive CASH
  edge on any row;
- zero rows satisfied the fixed economic-consensus conditions even before the
  `0.55` or `0.60` probability gate.

In plain terms, the model learned that moving out of AAPL for one day was
usually a losing choice after costs. Staying invested was the internally
consistent action, but it cannot beat buy-and-hold. Lowering a threshold or
removing the confirming heads after seeing this result would contradict the
model's own negative expected-edge estimate and would be post-hoc tuning.

## Decision

Reject the one-session two-state regime-consensus hypothesis. Do not reveal
2019+ data for it. The next approach must change the information or objective
materially rather than forcing trades from these same weak state forecasts.
