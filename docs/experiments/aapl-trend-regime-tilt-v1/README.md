# AAPL trend-regime tilt v1

## Outcome

This deterministic, zero-API-cost strategy beat the same-ledger AAPL
buy-and-hold benchmark in calendar 2024, calendar 2025, and 2026 year-to-date
through July 9. It also passed the predeclared 10 bps slippage / 12% margin-rate
stress case and both final-open and terminal-close cutoffs. The complete run took
5.14 seconds and recorded `$0.00` of external API/model cost.

This is a **retrospective historical-fit result**, not proof of reliable future
profit. The code was created after all evaluated dates occurred, and the result
uses modest leverage for much of each period.

## Frozen rule

At each completed close:

1. Compare AAPL adjusted close with its 150-session simple moving average.
2. Compare SPY adjusted close with its 200-session simple moving average.
3. Set the next AAPL adjusted-open target:

   - both above trend: `1.10x` AAPL;
   - both below trend: `0.975x` AAPL;
   - mixed: `1.00x` AAPL.

The account rebalances daily to the selected exposure. Exposure above 1.0 is
funded with explicit negative cash charged 8% annual interest over calendar
days. Orders pay 5 bps of adverse slippage. The benchmark uses the identical
ledger and enters 1.0x AAPL once.

## Base-case results

| Period | Strategy | AAPL B&H | Excess | Terminal-close excess | Strategy max DD | B&H max DD |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 37.8106% | 35.4795% | **+2.3311 pp** | +2.2021 pp | -17.3448% | -15.8111% |
| 2025 | 10.5611% | 10.1345% | **+0.4266 pp** | +0.3761 pp | -31.4850% | -30.8830% |
| 2026 YTD | 14.5684% | 14.2039% | **+0.3645 pp** | +0.5819 pp | -13.6996% | -12.4706% |

The return hurdle passed, but drawdown was worse than buy-and-hold in every
period.

## Risk, trading, and cost metrics

| Period | Volatility | Sharpe | Sortino | Worst day | Best day | Mean exposure | Max held exposure | Orders | Turnover | Slippage | Margin interest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 26.7064% | 1.3347 | 2.0264 | -10.0760% | +8.1966% | 1.0770x | 1.1113x | 199 | 2.3242x | $1.17 | $7.17 |
| 2025 | 34.6452% | 0.4633 | 0.6966 | -8.3929% | +13.2758% | 1.0510x | 1.1084x | 186 | 2.3898x | $1.14 | $4.43 |
| 2026 YTD | 28.5100% | 1.0736 | 1.6433 | -5.4228% | +4.9427% | 1.0798x | 1.1058x | 120 | 1.5050x | $0.75 | $3.45 |

Worst days were August 5, 2024; April 7, 2025; and February 13, 2026. Best days
were May 3, 2024; April 14, 2025; and July 6, 2026.

## Robustness checks

| Scenario | 2024 excess | 2025 excess | 2026 YTD excess | All periods pass? |
|---|---:|---:|---:|---:|
| Base: 5 bps / 8% margin | +2.3311 pp | +0.4266 pp | +0.3645 pp | Yes |
| Stress: 10 bps / 12% margin | +1.8171 pp | +0.1079 pp | +0.1426 pp | Yes |
| Severe: 20 bps / 12% margin | +1.6339 pp | **-0.0451 pp** | +0.0847 pp | No |

The pre-2024 diagnostic window (2005–2023) beat buy-and-hold in 16 of 19
calendar years. Mean annual excess was +2.5803 percentage points, median excess
was +1.7715 points, and the worst year was -2.2938 points.

## Extra-beta check

Static 1.10x AAPL returned more than this strategy in 2024 and 2026 YTD:

| Period | Strategy minus static 1.10x AAPL |
|---|---:|
| 2024 | -0.2449 pp |
| 2025 | +0.9456 pp |
| 2026 YTD | -0.4563 pp |

That means most of the raw outperformance over ordinary buy-and-hold comes from
additional AAPL exposure, not a demonstrated forecasting edge. The regime rule
did add value relative to constant leverage in the turbulent 2025 period.

## Integrity and reproducibility

- Run: `aapl-trend_regime_tilt_v1-20260710T121306Z-02a84016`
- Clean source commit at run start: `904e9614a4b46d2c50703c5bc8d319c234d2dc6d`
- Market observations: 6,920 joint AAPL/SPY sessions, 1999-01-04 through 2026-07-09
- Market-data SHA-256: `562545b0b0b492fdd172deb43551b866739ac634d2b9a89f26bd32987a43e51e`
- Model calls: `0`
- External monetary cost: `$0.00`
- Integrity errors: none

The run directory contains the exact market snapshot, complete JSON report, and
aligned daily strategy/buy-and-hold/static-leverage ledgers for every period and
cost scenario. See [report.json](aapl-trend_regime_tilt_v1-20260710T121306Z-02a84016/report.json).

## Failure points before real money

- The three headline windows were already historical when the system was built;
  they are not untouched forward tests.
- Leverage increased drawdown and daily loss severity. Real brokers also impose
  maintenance requirements that this research ledger does not simulate.
- Results depend on financing cost: the severe cost case fails 2025.
- Taxes, whole-share constraints, order rejection, market impact, intraday
  margin calls, and broker-specific margin rates are not modeled.
- Yahoo can revise historical adjusted prices. This branch therefore preserves
  the exact tested snapshot and its hash.
- The strategy does not establish that an LLM adds value; no LLM was needed.

The historical benchmark goal is satisfied. Real-capital readiness is not.
