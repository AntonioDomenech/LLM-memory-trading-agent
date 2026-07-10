# Rejected: one-session rare-loss forest v1

The frozen pre-2019 development experiment was rejected. No 2019 or later
market row was loaded, no candidate was selected, and no final model was
fitted.

## Immutable run

- Source branch: `codex/aapl-one-session-rare-loss-forest-v1`
- Frozen source commit: `cf903cd`
- Run ID: `rare-loss-development-20260710T230138Z-cb841be8`
- Artifact: `rare-loss-development-20260710T230138Z-cb841be8/`
- Complete wall time, including sealed replay verification: 46.8 seconds
- Runtime recorded before sealing: 23.7508 seconds
- External/API/LLM/network calls: 0
- Estimated external cost: $0.00
- Development fits: 14 deterministic local forests, plus the verifier's exact replay
- Mature common-support OOF rows: 3,522

## Result

Both `tail975` and `tail990` produced zero CASH decisions. Their strategy
ledgers therefore exactly tied same-ledger AAPL buy-and-hold at both 5 and 10
basis points, with zero active log edge, zero positive folds, and zero
completed CASH episodes. A tie does not pass.

The result was not caused by one nearly missed tail threshold. Across all
3,523 full-feature-ready prediction rows:

- Only 10 rows cleared the 97.5th-percentile OOB severe-probability threshold;
  only 3 cleared the 99th-percentile threshold.
- No row reached twice its fold's causal severe-loss prevalence. The maximum
  predicted severe probability was 0.204116, while the required fold-specific
  levels ranged from 0.207805 to 0.295409.
- No row reached the 0.55 ordinary CASH-win probability gate; the maximum was
  0.470125.
- No row reached the +0.001 expected 10-bps CASH-edge gate. Every prediction
  was negative; the range was -0.005343 to -0.000763.

The full model also missed every predictive promotion threshold:

- Ordinary CASH-win Brier improvement: 0.3519%, versus 2% required.
- Severe-event Brier improvement: 4.6176%, versus 5% required.
- Clipped-edge MAE improvement: 0.0885%, versus 1% required.

Adding IWM/VIX sentiment made predictive quality slightly worse than the exact
core-price ablation:

- Ordinary Brier: 0.240972 full versus 0.240866 core.
- Severe Brier: 0.074396 full versus 0.074196 core.
- Edge MAE: 0.014812 full versus 0.014810 core.

## Decision

This failure activates the predeclared stop rule: do not try another daily
price/market-only AAPL timing architecture and do not tune these thresholds
after seeing the result. The only permitted research pivot is a cheap
point-in-time data-quality audit for genuine SEC filings and/or historical
news text with provable availability timestamps. Gemma may be considered only
after that audit proves the information existed at each decision time.
