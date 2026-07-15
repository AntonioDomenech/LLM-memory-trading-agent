# AAPL additive residual shock v1: preflight rejection

Status: exact candidate rejected before preregistration or implementation on
2026-07-15.

## Decision

Do not build an online learner around this residual-shock opportunity stream.
A single conventional, causal specification added many trades outside the
strong exhaustion union but reduced edge at both transaction costs. At 10
bps, the addition improved only two of 14 years and one of seven fixed
two-year folds. The result is broad underperformance, not a rounding issue.

No threshold grid, alternate lookback, feature search, or post-result repair
was attempted. Searching variants after this result would turn a disciplined
preflight into retrospective rule mining. A future residual family would need
a genuinely new information source or prediction target, not a tuned version
of this candidate.

No learner contract, production module, stage, attempt lock, checkpoint,
confirmation input, or 2019+ value was opened or created.

## Exact causal candidate

Only the sealed through-2018 AAPL, SPY, and QQQ frame was read. At completed
close `t`:

1. Compute adjusted-close log returns for AAPL, SPY, and QQQ.
2. Fit ordinary least squares with an intercept over the previous 126
   completed sessions, predicting AAPL return from SPY and QQQ returns.
3. Compute the current AAPL residual using that prior-only fit.
4. Divide by the prior fit's residual standard deviation with three fitted
   degrees of freedom.
5. Emit a raw positive residual shock only when the z-score is strictly above
   `2.0`.
6. Apply the existing one-session canonicalization rule and retain the signal
   as an addition only when the fixed exhaustion union is not already CASH.

The fixed policy sells AAPL at adjusted open `t+1` and repurchases at adjusted
open `t+2`. The union-plus-residual policy is the binary union of both CASH
target streams. It uses the same continuous adjusted-open ledger, account
inception, costs, and no-leverage rules as the sealed v2 result.

This preflight tests one fixed candidate. It does not claim that all possible
residual models have been disproved.

## Signal count

- Raw residual shocks through 2018: 137.
- Canonical residual shocks through 2018: 132.
- Canonical shocks outside the exhaustion union: 107.
- Outside-union shocks during 2000-2004 shadow history: 20.
- Outside-union shocks during the scored 2005-2018 period: 85.
- Fixed union opportunities during 2005-2018: 121.

## Same-ledger result

| Metric | Fixed union | Residual only | Union plus residual |
|---|---:|---:|---:|
| 5-bps active log edge vs AAPL | 1.092826 | -0.176002 | 0.775635 |
| 5-bps final equity from $1,000 | $115,649.44 | $32,516.10 | $84,214.97 |
| 5-bps complete episodes | 121 | 104 | 198 |
| 5-bps winning episodes | 79 | 48 | 112 |
| 10-bps active log edge vs AAPL | 0.971826 | -0.280002 | 0.577635 |
| 10-bps final equity from $1,000 | $102,418.14 | $29,289.69 | $69,052.91 |
| 10-bps complete episodes | 121 | 104 | 198 |
| 10-bps winning episodes | 76 | 44 | 105 |

At 5 bps, adding the residual signal reduced active log edge by
`0.31719082744210664`. At 10 bps, it reduced edge by
`0.3941908723588032` and left 32.58% less final wealth than the union.

The 10-bps incremental result was positive only in 2008
(`+0.10781436635798369`) and 2012 (`+0.02393849930960684`). It was negative
in the other 12 years. Only the 2007-2008 fold was positive; the remaining
six folds were negative. The strongest benefit was therefore concentrated in
the financial crisis and did not persist.

The residual-only policy also lost to AAPL buy-and-hold after costs. An online
gate trained on this stream would either learn to make no additions and become
inert, or select sparse subgroups with a high risk of overfitting. Neither
outcome justifies a full implementation.

## Evidence and restrictions

- Branch: `codex/aapl-additive-residual-shock-v1`.
- Source bundle:
  `e/aapl_causal_contextual_expert_aggregation_v2/contextual-expert-aggregation-development-v2/`.
- Source manifest identity:
  `sha256:6cb657d1d9c323bd96a84b598b9b993e2444f756a65521aa1bf28207f7a63539`.
- Physical data end: 2018-12-31.
- Same-ledger AAPL final equity at 10 bps: $38,754.15.
- Exposure remained binary in `[0,1]`; minimum cash was 0.0; there was no
  leverage, shorting, borrowing, negative cash, or margin interest.
- Network, news, LLM, Ollama, and API calls: 0. External cost: $0.00.
- The complete source bundle remained checksum-clean after the diagnostic.

This is a through-2018 training preflight, not unseen test evidence and not
authorization for real capital.
