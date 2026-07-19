# Regime-conditioned expert disagreement v1: rejected

Status: **rejected as an improvement over the fixed union**. This is a real
trading-result rejection, not a runtime, module-count, timeout or bookkeeping
failure.

The repeated historical development diagnostic used only the physically
bounded 4,986-session file ending 2018-12-31. The inherited expert family had
already been influenced by later historical research, so this is not a
globally unseen holdout.

## Frozen calibration choice

Using only completed one-session outcomes through 2011 at 10 bps per changing
leg:

| Regime | Exclusive expert | Episodes | Total cash log edge | Mean edge | Positive years | Eligible |
|---|---|---:|---:|---:|---:|---|
| Risk-on | Contextual-only | 8 | -0.136208 | -0.017026 | 1 | No |
| Risk-on | Weak-trend-only | 0 | 0.000000 | n/a | 0 | No |
| Not-risk-on | Contextual-only | 20 | +0.190428 | +0.009521 | 3 | Yes |
| Not-risk-on | Weak-trend-only | 11 | +0.009927 | +0.000902 | 3 | Yes |

The frozen selector therefore chose contextual-only in `not_risk_on` and no
exclusive expert in `risk_on`. Opportunities where both experts agreed always
remained cash.

## 2012-2018 result

| Cost per changing leg | Selector return | AAPL buy-and-hold | Excess return | Relative wealth vs AAPL | Fixed-union return | Selector minus union log edge |
|---|---:|---:|---:|---:|---:|---:|
| 5 bps | +374.1257% | +211.3932% | +162.7325 pp | +52.2595% | +422.9935% | **-0.098097** |
| 10 bps | +355.9907% | +211.3932% | +144.5975 pp | +46.4357% | +395.0056% | **-0.082097** |

The selector still beat buy-and-hold strongly, but it removed useful trades
from an even better fixed union. Incremental selector-versus-union edge was
positive in only 2 of 7 years at both costs. At 10 bps, incremental edge after
removing the best incremental year was -0.098794.

The selector made 39 complete evaluation cash episodes. At 10 bps, 66.67% of
those episodes helped, mean episode edge was +0.009780 and median episode edge
was +0.011564. It beat AAPL in 6/7 years at 5 bps and 4/7 at 10 bps. Mean and
median annual excess were +6.7290 pp and +4.5267 pp at 5 bps, and +6.0533 pp
and +4.0638 pp at 10 bps.

During the two negative-AAPL years in this evaluation, the selector behaved
usefully:

| Year | AAPL | Selector at 5 bps | Selector at 10 bps |
|---|---:|---:|---:|
| 2015 | -3.5337% | +8.1781% | +7.2089% |
| 2018 | -5.6357% | +10.0511% | +8.7384% |

Continuous-account maximum drawdown was -42.99% at 5 bps and -43.22% at 10
bps. Continuous-account cash-day rate was 2.98%. The continuous 2005-2018
account made 211 executed orders; total turnover was 210.95 at 5 bps and
210.89 at 10 bps.

## Exact failed gates

Four performance gates failed:

- selector-minus-union edge was not above 0.0001 at 5 bps;
- selector-minus-union edge was not above 0.0001 at 10 bps;
- 10-bps incremental edge was positive in only 2/7 years, not at least 4/7;
  and
- 10-bps incremental edge was not positive after removing its best year.

Every safety and accounting gate passed: exposure stayed in `[0, 1]`, cash and
shares never became negative, there was no shorting, leverage, borrowing or
margin interest, the selector was an exact subset of the fixed union, and the
always-long comparison matched the same-ledger AAPL benchmark.

## Evidence and decision

The computation finished in 3.39 seconds with zero network, API, LLM, broker
or real-money actions. All nine payload hashes in `checksums.json` recomputed
exactly, and the forecast ends on 2018-12-31. Complete metrics, ledgers,
episodes, calibration rows, choices, forecast, gates and checksums are stored
under `regime-expert-disagreement-development-v1/`.

No 2019 or later row was opened. This exact selector is rejected; it will not
receive a cosmetic successor. The fixed union and the previously preserved
binary-regime selector remain the stronger historical leads.

Nothing here authorizes real-money trading.
