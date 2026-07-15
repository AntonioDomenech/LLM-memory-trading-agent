# AAPL persistent market stress v1: preflight rejection

Status: exact candidate rejected before preregistration or implementation on
2026-07-15.

## Decision

Do not build an online learner around this persistent risk-off state. The
fixed state protected the account during 2008 and 2018 but sacrificed far
more edge during ordinary markets and recoveries. When added to the strong
one-session exhaustion union, it finished below AAPL buy-and-hold at 10 bps
and improved only two of seven fixed two-year folds.

No moving-average grid, hysteresis variant, weekly timing variant, or
post-result repair was tested. The result rejects this exact conventional
candidate. A future slow-regime approach would require genuinely different
information or a new decision target, not another threshold search over the
same through-2018 history.

No learner contract, production module, stage, attempt lock, checkpoint,
confirmation input, or 2019+ value was opened or created.

## Exact causal candidate

At each completed close `t`, calculate the trailing 200-session simple moving
average, including `t`, of SPY and QQQ adjusted closes. The market is risk-off
if and only if both current closes are strictly below their respective
averages. Missing warm-up history is LONG.

The persistent policy is CASH at adjusted AAPL open `t+1` while that state is
true and LONG otherwise. The combined policy is CASH whenever either the
fixed one-session exhaustion union or the risk-off state requests CASH. Both
use the exact continuous adjusted-open ledger, binary exposure, transaction
costs, and account inception from the sealed v2 development bundle.

During scored 2005-2018 history, the state was risk-off on 531 sessions and
changed state 83 times. The exhaustion union requested 121 one-session CASH
intervals.

## Same-ledger result

| Metric | Fixed union | Risk-off only | Union plus risk-off |
|---|---:|---:|---:|
| 5-bps active log edge vs AAPL | 1.092826 | -0.451488 | 0.013531 |
| 5-bps final equity from $1,000 | $115,649.44 | $24,686.31 | $39,301.73 |
| 5-bps complete CASH episodes | 121 | 41 | 104 |
| 5-bps winning episodes | 79 | 14 | 52 |
| 5-bps positive years | 12/14 | 2/14 | 7/14 |
| 5-bps positive two-year folds | 7/7 | 2/7 | 3/7 |
| 10-bps active log edge vs AAPL | 0.971826 | -0.492989 | -0.090969 |
| 10-bps final equity from $1,000 | $102,418.14 | $23,670.96 | $35,384.31 |
| 10-bps complete CASH episodes | 121 | 41 | 104 |
| 10-bps winning episodes | 76 | 14 | 50 |
| 10-bps positive years | 11/14 | 2/14 | 6/14 |
| 10-bps positive two-year folds | 7/7 | 2/7 | 3/7 |
| Unresolved terminal episodes | 0 | 1 | 1 |

At 10 bps, adding the risk-off state reduced union edge by
`1.062795330188261` and left 65.45% less final wealth than the fixed union.
Its incremental edge was positive only in 2008
(`+0.28542679998972076`) and 2018 (`+0.1203646758655894`), zero in 2007,
2013, 2014, and 2017, and negative in every other scored year. Only the
2007-2008 and 2017-2018 folds were incrementally positive.

The risk-off-only policy's large 2008 edge was offset by losses in 2005,
2006, 2009-2012, 2015, and 2016. This is exactly the crisis-concentration
failure the preflight was designed to detect. An online overlay would either
learn to stay inactive or risk fitting a small number of exceptional market
episodes; neither justifies full staged implementation.

## Evidence and restrictions

- Branch: `codex/aapl-persistent-market-stress-v1`.
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
