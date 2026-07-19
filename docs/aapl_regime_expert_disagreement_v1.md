# AAPL regime-conditioned expert disagreement v1

## Purpose

This is a small trading experiment, not another infrastructure version. It
tests whether the already-strong contextual-exhaustion plus weak-trend union
can improve when the two experts disagree.

The hypothesis is simple: a contextual-only cash signal and a weak-trend-only
cash signal may have different value in rising versus non-rising broad-market
regimes. A fixed selector learned only from 2005-2011 may therefore remove bad
union cash episodes during 2012-2018 without using later outcomes in the new
selector.

The inherited expert family was selected during earlier research that already
inspected later historical periods. Therefore neither 2012-2018 nor any later
period is a globally unseen holdout. This run is a repeated historical
development diagnostic. It may identify a paper-trading candidate, but it may
not call 2019-2023 clean confirmation or call 2024 onward pristine evidence.

## Frozen input and inherited signals

The only market input is the tracked file
`e/chronological_exhaustion_expert_v1/authorized_inputs/aapl_spy_qqq_through_2018.csv`:

- Git blob `71b9aa69478a9a591200833131e930064859721f`;
- literal SHA-256
  `9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1`;
- 4,986 joint sessions from 1999-03-10 through 2018-12-31; and
- AAPL open, close and adjusted close plus SPY and QQQ adjusted close.

The contextual, weak-trend and canonical one-session union signals are reused
unchanged from `agent_benchmark/chronological_exhaustion_expert.py` at Git blob
`3d01affe636f2917306358a7d80f285b5a10f1d4`. No signal threshold or lookback
may be changed in this experiment.

## Chronological split

- Warm-up: all rows before 2005; no scored trade.
- Calibration diagnostic: 2005-01-01 through 2011-12-31.
- Development evaluation diagnostic: 2012-01-01 through 2018-12-31.
- The already-inspected 2019-2023 rows remain closed until the development
  result is committed and pushed and every development gate below passes. If
  opened, they are a repeated historical continuation audit, not clean
  confirmation.
- The already-inspected 2024 onward rows remain closed until that continuation
  audit passes. If opened, they remain repeated historical evidence.

The data file physically ends in 2018, so this development run cannot inspect
2019 or later prices.

## Exact selector

The market regime at decision close `t` is `risk_on` only when both
`SPY_adj_close[t] / SPY_adj_close[t-20] - 1` and
`QQQ_adj_close[t] / QQQ_adj_close[t-20] - 1` are finite and strictly positive.
All other rows with two finite 20-session returns are `not_risk_on`; a union
opportunity without both finite returns is invalid.

At each row where `unfiltered_union_signal` is true, membership uses exactly
`contextual_virtual_signal` and `weak_trend_virtual_signal` and is classified
as:

- `both`: contextual and weak-trend both signal;
- `contextual_only`: contextual signals and weak-trend does not; or
- `weak_trend_only`: weak-trend signals and contextual does not.

The one-session net cash edge used for calibration is
`log(AAPL adjusted open[t+1] / AAPL adjusted open[t+2])` plus the exact
round-trip friction for 10 basis points per changing leg. The outcome becomes
available only on session `t+2`. Calibration admits an episode only when its
entry open `t+1` and completed exit/maturity open `t+2` are both no later than
2011-12-31.

For each of the two regimes, calibration independently evaluates the
`contextual_only` and `weak_trend_only` cells. A cell is eligible only when it
has at least five completed calibration episodes, strictly positive total net
cash edge, and positive net cash edge in at least three distinct calibration
calendar years. The selector chooses the eligible expert with the larger mean
net cash edge. If only one is eligible, it chooses that expert. If neither is
eligible, it chooses neither. An exact mean tie chooses `contextual_only`.

During 2012-2018, `both` opportunities always remain cash. An exclusive
opportunity remains cash only when its expert is the frozen choice for its
current regime; otherwise the selector stays in AAPL. The choices never change
after 2011 and no 2012-2018 outcome can alter them. An episode belongs to a
stage, calendar year and yearly gate by its entry/fill date `t+1`; the outcome
still must be complete at `t+2`.

## Trading and comparison rules

- Target exposure is exactly 0% or 100% AAPL.
- No shorting, leverage, borrowing, negative cash or interest on cash.
- A close-`t` decision fills at the next adjusted open.
- Cash episodes last one session, using the inherited canonical union
  cooldown.
- Strategy and AAPL buy-and-hold use the same starting money, dates, adjusted
  opens, ledger and final valuation.
- Results are calculated at 5 bps and 10 bps per changing leg.
- Network calls, paid APIs, news calls, LLM calls and broker actions are zero.

## Cheap preliminary diagnostics

Before building any larger system, the run reports all four calibration cell
counts, yearly edges, total/mean/median edge, eligibility and frozen choice.
It also reports the selector, fixed union and buy-and-hold ledgers and complete
cash-episode rows. Expected runtime is under one minute; comparable preserved
runs took roughly 6-12 seconds. There is no artificial hard runtime rejection.

## Development gates

The approach advances only if all of these are true:

1. Every ledger passes the no-leverage and nonnegative-cash checks.
2. At 5 bps, the 2012-2018 selector active log edge versus AAPL is positive.
3. At 10 bps, the 2012-2018 selector active log edge versus AAPL is positive.
4. At both costs, selector-minus-fixed-union incremental active log edge is
   strictly greater than `0.0001`.
5. At 10 bps, incremental edge is positive in at least four of the seven
   calendar years 2012-2018.
6. At 10 bps, total incremental edge remains positive after subtracting the
   single best incremental calendar year.
7. At 10 bps, the selector has at least 25 complete evaluation cash episodes,
   a strictly positive mean episode edge and a strictly positive median episode
   edge.
8. The selector cash stream is an exact subset of the fixed union stream and
   the AAPL benchmark is identical across policy comparisons.

Failure rejects this approach immediately; no later row is opened and no
cosmetic successor is created. Passing permits one separately recorded
2019-2023 repeated historical continuation audit with the exact frozen choices.

## Evidence and scope

The branch is `codex/aapl-regime-expert-disagreement-v1`. This preregistration
must be committed and pushed before the development computation. The final
result must preserve the code, cell diagnostics, ledgers, metrics, checksums,
runtime, gate failures and comparison-table update, then be committed and
pushed.

This historical experiment cannot prove reliable future profit and never
authorizes real-money trading.
