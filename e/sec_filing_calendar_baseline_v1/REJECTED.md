# SEC filing-event cash baseline v1: rejected

Status: **rejected because the trading rule performed badly**. The run itself
completed normally in 2.20 seconds.

This test used the 75 authenticated Apple 10-K/10-Q rows available from 2000
through 2018. After each safely available filing, the rule sold AAPL at the
next open, held cash for 20 sessions, and bought AAPL again at open `t+21`.
The one duplicate filing day did not extend the episode, leaving 74 complete
cash episodes. No filing contents, LLM, parameter search, 2019-or-later data,
network call, paid API, broker, or real-money action was used.

## 2000-2018 result

| Cost per changing leg | Strategy return | AAPL buy-and-hold | Excess return | Relative ending wealth vs AAPL | Winning / losing / tied years |
|---|---:|---:|---:|---:|---:|
| 5 bps | +524.5767% | +4,689.9859% | -4,165.4092 pp | **-86.9608%** | 5 / 14 / 0 |
| 10 bps | +479.7370% | +4,687.5933% | -4,207.8564 pp | **-87.8908%** | 5 / 14 / 0 |

The strategy made money in absolute terms because AAPL rose enormously over
the period. But it ended with only about 12.1% of buy-and-hold's wealth at the
10-bps stress cost. The reason is simple: being in cash after every filing
missed too many strong AAPL rises.

At 10 bps:

- mean annual excess was -17.7683 percentage points;
- median annual excess was -12.2792 percentage points;
- active log edge was -2.111209;
- edge after removing the best year was still -2.435702;
- the 74 episodes helped only 35.14% of the time;
- mean episode edge was -0.028530 and median episode edge was -0.039334; and
- removing the best episode still left -2.373491 total episode edge.

## Negative-AAPL years

The rule did not provide reliable protection when AAPL itself lost money.

| Year | AAPL return | Strategy return at 10 bps | Excess return |
|---|---:|---:|---:|
| 2000 | -72.0184% | -78.2187% | -6.2003 pp |
| 2002 | -37.8054% | -13.9643% | +23.8410 pp |
| 2008 | -56.9073% | -59.1189% | -2.2116 pp |
| 2015 | -3.5337% | -1.6746% | +1.8590 pp |
| 2018 | -5.6357% | -17.6153% | -11.9796 pp |

It helped in only two of those five years. Aggregated stress-cost relative
wealth across negative-AAPL years was -9.0958%.

## Risk and trading activity

| Measure | 5 bps | 10 bps |
|---|---:|---:|
| Maximum drawdown | -80.5510% | -80.6480% |
| AAPL maximum drawdown | -81.7604% | -81.7604% |
| Time fully in cash | 30.9688% | 30.9688% |
| Executed orders, including the initial buy | 149 | 149 |
| Total turnover | 148.9625 | 148.9251 |
| Nominal trading costs on the evolving EUR 1,000 account | EUR 261.35 | EUR 496.08 |

The small drawdown improvement did not compensate for the enormous loss of
upside.

## Exact failed trading gates

All seven preregistered performance gates failed:

- cumulative relative wealth was not positive at 5 bps;
- cumulative relative wealth was not positive at 10 bps;
- winning years did not outnumber losing years at 10 bps;
- mean annual excess was not positive at 10 bps;
- median annual excess was not positive at 10 bps;
- edge was not positive after removing the best year at 10 bps; and
- edge across negative-AAPL years was not positive at 10 bps.

Every scientific safety check passed. Targets and realized exposure stayed in
`[0, 1]`; cash and shares never became negative; margin interest was zero; the
always-long control exactly matched same-ledger AAPL; all 10 artifact hashes
recomputed; all 75 filing rows were preserved; and every price, decision, fill,
and exit ended no later than 2018-12-31.

## Decision

Do not open 2019-2023 or 2024+ for this rule. Filing timing by itself is not a
useful AAPL trading signal in this experiment, and this exact unconditional
20-session rule should not receive a cosmetic successor.

The result does **not** say filing contents are useless. A materially different
next test may ask whether a small local reader can distinguish the few filings
that deserve a cash response from the many filings after which staying in AAPL
was better.

Complete manifest, schedule, target history, annual metrics, episode results,
same-ledger ledgers, gates, report, and checksums are preserved under
`sec-filing-calendar-development-v1/`.

Nothing here authorizes real-money trading.
