# Unleveraged AAPL long/cash harness

This runner is the hard reset after rejecting leverage as a way to beat AAPL
buy-and-hold. It permits only two actions: hold 100% AAPL or hold 100% cash.
It makes no paid API or model calls.

## Non-negotiable execution contract

- The signal uses completed information through close `t` and fills at AAPL's
  adjusted open `t+1`.
- Requested, post-fill, and holding exposure must remain in `[0, 1]`.
- Cash and AAPL shares must never be negative. Shorting, margin, borrowing, and
  margin interest are forbidden.
- The strategy and benchmark use the same adjusted-open ledger, fractional
  shares, and adverse slippage.
- Cash earns zero because the warehouse has no verified vintage-safe cash-rate
  series. Splits and dividends are implicit in adjusted prices.

Every saved period includes a machine-checked `no_leverage_proof`. A single
invariant violation invalidates the run.

## Selection and final audit

Rules are selected using data ending no later than 2023-12-31. The requested
final audit is then run without modifying the rule on fresh-start 2024, 2025,
and 2026-YTD accounts, plus one continuous account spanning all three periods.
The report records how many times final-period outcomes have been examined.

Each candidate also receives a downside-behavior audit: every full 2000–2023
calendar year in which the same-ledger AAPL buy-and-hold account lost money,
plus fixed dot-com, 2008, Q4 2018, COVID-crash, and 2022 stress windows. The
report shows absolute return, excess return, drawdown, and time in cash. These
are diagnostics and cannot compensate for failing a required final year.

Because development takes place after all these dates, none is a truly unseen
holdout. Passing is retrospective evidence only; it is not a profit guarantee.
Prospective paper trading is required before risking capital.

Promotion requires all of the following:

- more than one basis point of excess return in every fresh period at 5 bps,
  under both final-open and terminal-close marks;
- material positive active log return in every segment of the continuous
  2024-to-2026 account (a deliberately stricter 10-basis-point log-return
  threshold);
- positive excess in every fresh segment under both terminal marks and every
  continuous segment at 10 bps;
- zero integrity or no-leverage errors;
- total runtime under one hour and `$0.00` external/model cost.

The 20 bps scenario is reported as a severe diagnostic. The leveraged 1.10x
comparator and margin-rate stresses from the older harness are intentionally
absent.

## Frozen first families

- `contextual_exhaustion_v1`: after an AAPL intraday return above the prior
  126-session 90th percentile, hold cash for the next session only when both
  SPY and QQQ have negative 10-session momentum.
- `gap_down_cash_v1`: hold cash for one next-open interval after an AAPL opening
  gap below -4%.
- `exhaustion_or_gap_v1`: the predeclared OR combination of those signals.

These materially different families must be evaluated and preserved on
separate Git branches. A failed final audit is saved, not tuned in place.

The frozen search manifest is explicitly labeled self-attested: the exploratory
1,508-rule screen was not saved in full. Its hash proves that the declaration
did not change during a particular run; it does not prove the original search
history or create a pristine holdout. Reports also hash every ledger CSV, the
selection manifest, and the final report via a separate `checksums.json` file.

## Commands

```powershell
python -m pytest tests/test_unleveraged_aapl.py -q
python -m agent_benchmark.unleveraged_aapl --strategy contextual_exhaustion_v1 --refresh-data
```

Ignored local cache and scratch runs default to `data/unleveraged_aapl`. A
candidate branch should point `--output-dir` at a versioned evidence directory
so its immutable report, daily ledgers, and exact market snapshot are committed.
