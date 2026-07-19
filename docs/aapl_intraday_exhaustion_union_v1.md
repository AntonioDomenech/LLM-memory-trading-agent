# AAPL intraday-only exhaustion union v1

## Purpose

This experiment changes the trade, not the signal or the research machinery.
The existing fixed contextual-plus-weak-trend exhaustion union sells AAPL at
the next adjusted open and buys it back at the following adjusted open. That
cash interval discards both the next session's intraday return and the
following overnight return.

The new hypothesis is that the useful reversal is concentrated inside the next
trading session, while AAPL's overnight exposure is still valuable. On every
unchanged union signal, the candidate will therefore sell at the next adjusted
open and buy back at that same session's adjusted close. Both orders are fixed
by the signal close; the close price cannot be inspected before deciding to buy.

This is a repeated historical mechanism test. The underlying exhaustion family
was selected during earlier research that had already inspected later history,
so this run is not a pristine holdout and cannot prove reliable future profit.

## Frozen input and signal

The only input is the physically bounded file
`e/chronological_exhaustion_expert_v1/authorized_inputs/aapl_spy_qqq_through_2018.csv`:

- Git blob `71b9aa69478a9a591200833131e930064859721f`;
- literal SHA-256
  `9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1`;
- 4,986 joint sessions from 1999-03-10 through 2018-12-31; and
- AAPL open, close and adjusted close plus SPY and QQQ adjusted close.

The contextual, weak-trend and canonical union signals are reused unchanged
from `agent_benchmark/chronological_exhaustion_expert.py`. Their thresholds,
lookbacks, market filters and cooldown may not change. The development run must
reproduce exactly 121 complete fixed-union opportunities from 2005 through
2018. It must use the inherited `stage_outcome_available` mask, even though the
candidate buyback finishes earlier, so the candidate and the open-to-open union
receive exactly the same opportunities.

The inherited signal module is additionally bound to literal SHA-256
`a30224763c9858aed905b76215c2c5a66eddd58f107182d751d8eb6a32688c6e`.
The exact 121 accepted decision dates, serialized as ascending ISO dates with
one `\n` after every date including the last, must hash to
`b4bec71ac4159086edb1faa3151630bb524b6f2e8b7fdaebd2ebf5dab68dbb13`.
Count equality without this date identity is not enough.

## Frozen trading rule

For a union decision made after completed close `t`:

1. Commit at close `t` to both the sell and unconditional same-day buyback
   policy without knowing any `t+1` value.
2. Sell all AAPL at adjusted open `t+1`, including adverse per-leg cost.
3. Once the sale proceeds are mechanically known, submit a broker-supported
   notional market-on-close order for the full realized cash amount before the
   closing-auction cutoff. The order size may depend only on those sale
   proceeds, never on a later `t+1` price or signal.
4. Remain 100% in non-interest-bearing cash during session `t+1` and fill the
   unconditional order at adjusted close `t+1`, including the same adverse
   per-leg cost.
5. Hold AAPL overnight and thereafter until another canonical union signal.

Adjusted close is the historical proxy for the official closing auction. The
policy is precommitted; only its notional size is mechanically determined from
the realized open-sale cash. It is not a decision made after observing the
close. The experiment may not cancel, delay or condition the buyback on any
other `t+1` observation. Auction rejection or a missing/non-positive close must
fail the experiment; the ledger may not invent another fill.

For this historical development test, cash-notional fractional MOC is an
explicit idealized execution assumption, not a claim about support at the
user's broker. Even a historical pass cannot enter prospective paper trading
until a separate preregistration verifies a specific zero-cost broker/order
path, its cutoff, fractional/notional support and rejection handling. This
feasibility limitation does not permit changing the historical fill formula.

Target exposure is exactly 0% or 100% AAPL. Shorting, leverage, borrowing,
negative cash and interest on cash are forbidden. Fractional shares are
allowed. Adjusted open is `raw open * adjusted close / raw close`; adjusted
close is used for the close fill. Every changing leg pays either 5 basis points
or 10 basis points of adverse execution cost.

For cost fraction `c`, exact fractional-share accounting is:

- initial shares = starting cash / (`initial adjusted open * (1 + c)`);
- open-sale cash = prior shares * `adjusted open * (1 - c)`;
- close-buy shares = all sale cash / (`adjusted close * (1 + c)`);
- cash after each all-in buy is exactly zero, apart from floating-point dust
  smaller than `1e-10`, which is set to zero; and
- there is no terminal liquidation or terminal fee because both strategy and
  benchmark are valued while holding their positions.

## Period and controls

- Warm-up: all rows before 2005; no scored trade.
- Development: 2005-01-01 through 2018-12-31.
- Final valuation: the last adjusted open in 2018, with every admitted
  candidate buyback already completed by the preceding close.
- Control 1: same-ledger AAPL buy-and-hold.
- Control 2: the unchanged fixed union's open-`t+1` to open-`t+2` cash rule.

The candidate, fixed union and buy-and-hold must use the same starting money,
sessions, corporate-action adjustments, initial entry cost, final valuation
and per-leg cost assumption. Development may receive no byte containing a
2019-or-later market row.

## Development evidence and gates

The run must save full event ledgers, daily/open valuation ledgers, cash
episodes, annual results, seven fixed two-year folds, drawdowns, turnover,
costs, checksums, runtime and all safety checks. An episode belongs to the year
and fold of its `t+1` sell-open date.

The candidate advances only if every gate below passes at both 5 and 10 basis
points unless a gate explicitly says 10 basis points:

1. All exposure, shares and cash safety checks pass, and the candidate has no
   network, API, LLM, broker or real-money action.
2. Candidate active log edge and relative ending wealth versus AAPL are
   strictly positive.
3. Candidate active log edge exceeds the fixed open-to-open union by more than
   `0.0001`.
4. Candidate edge versus AAPL is positive in at least 8 of 14 calendar years
   and at least 4 of 7 fixed two-year folds.
5. Candidate-minus-union incremental edge is positive in at least 4 of 7 folds
   and remains strictly positive after removing its best incremental fold.
6. Candidate edge versus AAPL remains positive after removing its best year,
   and no one year supplies more than 50% of total positive annual edge.
7. At 10 basis points, aggregate edge is positive across negative-AAPL years
   2008, 2015 and 2018, with at least two of those three years positive.
8. Exactly 121 complete candidate episodes are present. At 10 basis points at
   least 55% win, and mean and median episode edge are strictly positive. No
   single episode may supply more than 50% of total positive episode edge.
9. The candidate and fixed union signal dates match exactly, both cost ledgers
   reconstruct from their events, and the always-long ledger matches the
   same-ledger AAPL benchmark.

Failure rejects this exact hypothesis immediately. No 2019-or-later row may be
opened. Passing permits a separately committed, unchanged 2019-2023 repeated
historical continuation audit; only another pass may permit a repeated
2024-onward audit.

This document authorizes development only. It does not choose how a pending
late-2018 signal, account state or cooldown crosses into 2019, nor how a
late-2023 state crosses into 2024. Before any later-stage byte is opened, a
separate pushed continuation preregistration must bind those boundary and
carryover rules. Until then, the final two development rows remain permanently
ineligible for this development score.

## Runtime and evidence scope

The data already exists and the expected development runtime is under one
minute. There is no artificial timeout gate: a slow import or harmless count
message is logged and does not invalidate a trading result. Only wrong data,
future leakage, wrong fills/costs, corrupted evidence or a failed trading gate
can stop the experiment.

The branch is `codex/aapl-intraday-exhaustion-union-v1`. This preregistration
must be committed and pushed before any candidate return is calculated. The
final result, including rejection, must also be committed and pushed.

Nothing in this experiment authorizes real-money trading.
