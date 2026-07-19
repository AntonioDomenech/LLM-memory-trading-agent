# AAPL binary-regime union: prospective paper track

## Purpose and promotion basis

This is not another historical strategy version. It moves the strongest
already-preserved long/cash policy into an append-only prospective paper
record, where each decision is saved before its outcome exists.

The source policy is the frozen binary-regime union selector. Its post-hoc
continuous 2005 through 2026-07-09 audit passed all 41 declared long-run gates:

- 18/22 positive reporting periods at 5 bps and 16/22 at 10 bps;
- total active log edge `1.4058433467` and `1.2508432563`;
- 155 complete episodes with positive mean and median edge at both costs;
- positive aggregate edge in negative-AAPL years;
- positive edge after removing the best period and the five best episodes;
- maximum drawdown near -43%, versus about -60% for AAPL; and
- ending wealth about 4.08 times AAPL at 5 bps and 3.49 times AAPL at 10 bps.

This passed the goal's separate long-term-success standard. It did not pass the
stricter recent-year standard because it lost 2024, and all historical evidence
is retrospective. The paper track exists to obtain evidence that is genuinely
recorded before the market outcome.

## Exact frozen policy

The paper arm uses the **frozen-through-2023** policy, not continued learning.
The historical online and frozen arms made identical post-2023 actions, so the
unexercised adaptive layer adds no demonstrated value and is excluded from the
promoted arm.

At each completed market close `t`:

1. Recompute the unchanged contextual and weak-trend exhaustion candidates
   from `agent_benchmark/chronological_exhaustion_expert.py`.
2. Remove candidates before 2005-01-01 and apply the one-session account-union
   cooldown exactly once over the continuous history. Never reset it at a
   calendar or paper boundary.
3. Compute strict 20-session SPY and QQQ adjusted-close returns using only
   completed closes through `t`.
4. `risk_on[t]` is true only when both returns are strictly positive.
5. The frozen selector stays LONG for a risk-on union signal and selects CASH
   for a not-risk-on union signal. With no accepted union signal it stays LONG.

The close-`t` target fills at the next actual AAPL adjusted open. A CASH episode
sells all AAPL at open `t+1` and buys all AAPL at open `t+2`, with 5 bps per
changing leg in the main paper ledger and 10 bps in the stress ledger. Exposure
is exactly 0% or 100%. There is no shorting, leverage, borrowing, negative cash,
margin interest or interest on cash.

## First prospective decision

This preregistration is being pushed while U.S. markets are closed on Sunday,
2026-07-19. The intended first as-of close is Friday, 2026-07-17, and the
intended first possible fill is the next actual session open. The runner must
not assume those dates: it must receive a complete Yahoo session through
2026-07-17 and must save the decision with a trustworthy UTC creation time
before any next-session open value is downloaded or read.

Both paper ledgers inherit an identical hypothetical `$1,000` AAPL position
valued at the as-of adjusted close. This common starting position has no entry
cost because no pre-paper return is scored. At the next open both arms first
receive the same overnight mark; only then may the strategy execute the saved
target and pay its changing-leg cost. This prevents a first CASH decision from
receiving a free or missing initial trade.

## Free data and continuity

The runner may make one zero-cost Yahoo Finance download through `yfinance` for
AAPL, SPY and QQQ, ending at the preregistered as-of session. It must save the
exact bounded CSV and its SHA-256 beside the decision.

Before producing a decision, the downloaded history through 2026-07-09 must
reproduce the preserved audited input and frozen action stream. A real value,
date, corporate-action or action-stream mismatch stops this first decision;
the runner may report the mismatch but may not silently splice incompatible
series or change thresholds. This is a trading-data integrity issue, not a
runtime-count gate.

Yahoo may revise historical adjusted data. Every prospective decision therefore
binds its own exact snapshot. A later outcome evaluation uses the decision's
saved snapshot plus newly observed sessions and never rewrites the old input or
decision.

## Append-only decision and outcome records

The first run must create, without overwriting:

- a bounded market snapshot;
- a decision JSON containing creation time, as-of close, intended next action,
  reason, signal/regime values, next unknown fill role, exact policy/code/data
  identities and an explicit `outcome_known=false`;
- a hash manifest covering both files; and
- an initial paper-state JSON with the common AAPL starting position.

The decision file is immutable. After the relevant future opens occur, a
separate command may append an outcome JSON and updated state. It must not edit
the original decision. Every later close repeats the same order: save the
decision first, then wait for future data.

The paper report must distinguish:

- frozen strategy and same-ledger AAPL returns at 5/10 bps;
- decisions, changing actions, cash episodes, costs, turnover and drawdown;
- wins, losses and ties by completed episode and reporting period; and
- any shadow online-learning difference, if evaluated separately later.

No result is promoted from one decision. The record should normally accumulate
at least 12 months and 20 complete cash episodes before reliability is judged,
unless strong negative evidence rejects it sooner.

## Execution and authority

The branch is `codex/aapl-binary-regime-prospective-paper`. The preregistration
and implementation must be committed and pushed before the first current-data
download. The runner must start from that clean pushed branch, bind its relevant
files and dependency hashes, and fail if the first decision path already
exists. No retry may replace a consumed decision; a transport failure may be
resumed only if no market snapshot or decision was published.

This is paper trading only. Broker calls and real-money actions are forbidden.
