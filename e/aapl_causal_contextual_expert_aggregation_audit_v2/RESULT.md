# Causal contextual expert aggregation v2: 2019-2023 continuation audit

## Verdict

**Verified post-rejection historical-policy candidate; not a learning
candidate. The original v2 development rejection remains final.**

The one-shot 2019-2023 continuation audit passed all 64 declared policy and
integrity checks at both cost levels. It supports the already learned policy as
a historical long/cash lead, but it does not show that continuing to learn new
lessons improved a single trade.

This was a post-rejection reuse of historical data. It was not a confirmation
test, a pristine holdout, a prospective result, or authorization to use real
capital. The input ended on 2023-12-29, no post-2023 market value was accessed,
and the current contract does not authorize opening 2024 or later.

## Locked 2019-2023 result

Relative wealth is the terminal wealth ratio versus the same-ledger AAPL
buy-and-hold benchmark, minus one.

| Metric | 5 bps | 10 bps stress |
|---|---:|---:|
| Relative wealth versus AAPL | +7.7782% | +3.9672% |
| Positive calendar years | 4/5 | 4/5 |
| Positive fixed blocks | 3/3 | 2/3 |
| Complete cash episodes | 36 | 36 |
| Beneficial episodes | 50.00% | 50.00% |
| Episode median log edge | +0.001357334 | +0.000357334 |
| Log edge after removing the best year | +0.043228314 | +0.011228296 |

Annual same-ledger returns were:

| Year | AAPL buy-and-hold | Policy at 5 bps | Excess | Policy at 10 bps | Excess |
|---|---:|---:|---:|---:|---:|
| 2019 | +85.6344% | +84.8974% | -0.7370 pp | +83.7914% | -1.8430 pp |
| 2020 | +86.5801% | +90.4662% | +3.8861 pp | +88.1942% | +1.6141 pp |
| 2021 | +33.6425% | +37.9437% | +4.3012 pp | +37.3930% | +3.7505 pp |
| 2022 | -27.4775% | -26.3435% | +1.1340 pp | -27.1493% | +0.3282 pp |
| 2023 | +51.8447% | +53.5361% | +1.6913 pp | +53.0761% | +1.2314 pp |

The stress-cost block edges were negative for 2019-2020 but positive for
2021-2022 and 2023. The result therefore did not depend on one isolated year,
and the edge remained positive after removing the best year.

## Continuous 2005-2023 diagnostic

| Metric | 5 bps | 10 bps stress |
|---|---:|---:|
| Relative wealth versus AAPL | +196.0760% | +155.8564% |
| Positive calendar years | 16/19 | 13/19 |
| Complete cash episodes | 146 | 146 |
| Beneficial episodes | 60.96% | 58.90% |
| Log edge after removing the best year | +0.600809099 | +0.475809027 |
| Log edge after removing five largest episodes | +0.579301899 | +0.438301817 |
| Policy maximum drawdown | -43.79% | -43.95% |
| AAPL maximum drawdown | -60.42% | -60.42% |

All four negative-AAPL years in the continuous window (2008, 2015, 2018, and
2022) had positive aggregate policy edge at both cost levels.

The exact fixed union remained better over the complete 2005-2023 account. The
online policy trailed it by 0.079204 log edge at 5 bps and 0.065204 at 10 bps,
equivalent to 7.61% and 6.31% less relative wealth. Over the new 2019-2023
suffix alone, the online policy was modestly better than the union by 0.003082
and 0.006082 log edge. However, the simpler contextual-only rule was the best
fixed comparator in 2019-2023 and beat online by 0.000990 and 0.001990 log edge.

## What the learning controls proved

The online arm admitted all 39 newly matured 2019-2023 lessons, while the
frozen arm admitted none. Despite that, they made exactly the same decisions:

- 38 cash-score differences;
- zero threshold crossings;
- zero action differences;
- zero incremental edge at either cost; and
- zero complete online-versus-frozen comparison episodes.

Therefore `online_minus_frozen_adaptive_status` is `unexercised`, and useful
continual learning was not demonstrated.

The full policy differed from the global-only control on three completed
episodes in 2020-2021 and from the lifetime-only control on one episode in
2022. Those comparisons were positive but too sparse, so both are classified
as `exercised_insufficient_evidence`.

## Integrity, runtime, and cost

- Standalone verifier: passed.
- Declared gates: 64/64 passed, with no fatal or ordinary failure.
- No shorting, leverage, borrowing, negative cash, or exposure above 1.0.
- One uninterrupted account from 2005 through 2023; no year-boundary reset.
- Network, news, LLM, model, and API calls: zero.
- External cost: $0.00.
- Time to private seal: 1,503.422 seconds.
- Observed complete locked stage command, including its internal independent
  verifier: 3,003.7 seconds, below the 3,600-second limit.
- Observed separate read-only verifier command: 1,489.7 seconds. These two
  wall-clock observations come from the command runner rather than the sealed
  runtime field.

## Frozen identity

- Branch: `codex/aapl-causal-contextual-expert-aggregation-audit-v2`
- Frozen run commit: `f2085fcf776d32a29918727ad8d9b64932cb17d9`
- Run ID:
  `contextual-expert-aggregation-post-rejection-2019-2023-audit-v2`
- Attempt-lock SHA-256:
  `c929ca38a3bc10161630d85ab58d3bbcf916739fd81088b55779639ffcd21664`
- Checkpoint SHA-256:
  `4aaf549c9352ac322f36c46cb8dcdd03f672a1eba580813ba8a1947d323a85eb`
- Report SHA-256:
  `01c58f963566517c5abe88ea293e34fa84cad22d7a8fd42d79bde36d6dba940e`
- Manifest SHA-256:
  `c71e201a8a3d1ef2b0ebe914d46abfdd0f5a9154d4f106ce723168a9bce009e8`

## Decision

Preserve this approach as a **fixed-policy historical lead with an online
shadow arm**, not as evidence that a self-improving trading agent works. A
separate 2024-or-later audit may now be proposed and preregistered, but this
result itself authorizes no later-data access and no real-money trading. No
LLM, news, or sentiment input was exercised, so this audit provides no evidence
for those proposed components.
