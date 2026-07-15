# Binary-regime union selector v1: calibration pass

Status: **through-2018 calibration/training passed**. This is not holdout
evidence and is not evidence that the selector predicts unseen years.

The frozen implementation was committed and pushed at
`1d9359c989694adf8d9690df980bd2e8946b8f3f` before this stage ran. The input
is a Git-tracked, physically bounded local snapshot ending on 2018-12-31.
The report records no network, API, LLM, or later-outcome access.

## Training diagnostics

| Cost per changing leg | Selector log edge vs AAPL | Selector wealth vs AAPL | Exact-union log edge vs AAPL | Selector minus union |
|---|---:|---:|---:|---:|
| 5 bps | 1.201828497406 | +232.619330% | 1.092826209474 | 0.109002287933 |
| 10 bps | 1.100828438490 | +200.665582% | 0.971826138890 | 0.129002299599 |

- One continuous account starts in 2005 and is never reset.
- The exact union has 121 complete cash episodes; the selector takes 101 and
  vetoes exactly 20.
- At 10 bps, 55% of the vetoes help; their mean and median benefits are
  positive, and the largest positive veto supplies 21.61% of positive veto
  benefit.
- The risk-on state ends LONG after 25 matured lessons, with effective count
  23.555951 and discounted mean -0.003904.
- The not-risk-on state ends CASH after 138 matured lessons, with effective
  count 99.858259 and discounted mean +0.010431.
- Neither state changes its structural default during this training window.

## Integrity result

All frozen gates passed, including exact causal maturity and label arithmetic,
continuous cooldown/accounting, selector-as-union-subset, episode and veto
reconciliation, identical action streams at both cost assumptions, same-ledger
buy-and-hold, and no leverage, shorting, borrowing, or negative cash. The run
finished in 12.29 seconds, below the 3,600-second cap. All 18 sealed artifact
hashes independently matched `checksums.json`.

The immutable run is
`binary-regime-union-selector-development-v1`. Its stage manifest self-hash is
`sha256:3dfd21609d8851157d7062e5f27cad9e50119fb45a2cd5e8cff8cba9a3d28cd6`.

## Next authorization

This pass authorizes one untouched selector-level validation on 2019-2023.
The calibration artifacts and this status note must be committed and pushed
before a physically through-2023 snapshot is created or opened. A failed
validation gate ends this branch without access to 2024 or later outcomes.
