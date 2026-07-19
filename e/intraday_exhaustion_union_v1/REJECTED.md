# Intraday-only exhaustion union v1: rejected as an improvement

Status: **rejected at the 2005-2018 development gate**. This is a real trading
result, not a timeout, import, test-count or bookkeeping failure. No
2019-or-later price was opened.

## Result

The candidate used the exact 121 fixed contextual-plus-weak-trend union signal
dates. It sold AAPL at the next adjusted open exactly like the existing union,
but bought back at that same session's adjusted close instead of waiting for
the following adjusted open.

| Cost per changing leg | Candidate return | AAPL buy-and-hold | Relative wealth vs AAPL | Fixed union relative wealth | Candidate minus union log edge |
|---|---:|---:|---:|---:|---:|
| 5 bps | +11,428.73% | +3,777.35% | +197.34% | +198.27% | **-0.003136** |
| 10 bps | +10,109.75% | +3,775.41% | +163.45% | +164.28% | **-0.003136** |

The candidate remained an excellent historical AAPL-or-cash policy. At 5 bps
it beat AAPL in 11 of 14 years and all seven two-year folds. At 10 bps it beat
AAPL in 11 of 14 years and six of seven folds. Mean/median annual excess return
was +7.74/+5.90 percentage points at 5 bps and +6.64/+4.81 points at 10 bps.

That strength does not make this successor an improvement. The existing fixed
open-to-open union ended with slightly more wealth at both costs. The new
timing improved on the union in only 2 of 7 two-year folds, and its incremental
edge was negative after removing its best incremental fold. Six frozen gates
therefore failed: union improvement, at least four positive incremental folds,
and positive incremental edge without the best fold, each at both costs.

## What the test learned

The only return difference between the two policies is the overnight interval
from the signal session's close-buy to the following open. Across all 121
episodes, keeping that overnight exposure subtracted `0.0031360693` log edge.
The number is identical at 5 and 10 bps because both policies execute the same
two changing legs; only their buyback time differs.

In plain language, buying back at the close almost tied the stronger union and
slightly reduced maximum drawdown, but the retained overnight moves were
slightly harmful overall. There is no justification to replace the existing
union or to open later periods for this exact timing rule.

## Diagnostics and integrity

- Complete cash episodes: 121; winning episodes: 64.46% at both costs.
- Mean/median episode edge at 10 bps: +0.008006 / +0.006055.
- Maximum drawdown: -42.52% at 5 bps and -42.75% at 10 bps, versus -60.42%
  for AAPL buy-and-hold.
- Negative-AAPL years 2008, 2015 and 2018 all had positive candidate edge at
  both costs.
- Executed orders: 243, including every same-day closing buy.
- Open-ledger cash-day rate: 3.43%; no shorting, leverage, borrowing, negative
  cash or margin interest.
- The exact decision-date SHA-256 is
  `b4bec71ac4159086edb1faa3151630bb524b6f2e8b7fdaebd2ebf5dab68dbb13`.
- Candidate events, fixed-union ledger, buy-and-hold ledger and independent
  buy-and-hold closed form all reconstructed at both costs.
- All 16 checksummed run payloads recomputed exactly.
- Runtime was 7.422 seconds. Network, paid API, LLM, broker and real-money
  actions were zero.
- The run was bound to clean pushed commit
  `31d1e63bc6ddc748567d51528beac17442060c15`.

The close fill is an idealized cash-notional fractional market-on-close
assumption. A specific zero-cost broker path was not verified, which would have
been required before prospective execution even if the historical gates had
passed.

Complete metrics, gates, forecasts, event/open ledgers, episodes and checksums
are preserved under `intraday-exhaustion-union-development-v1/`.

Nothing here authorizes real-money trading.
