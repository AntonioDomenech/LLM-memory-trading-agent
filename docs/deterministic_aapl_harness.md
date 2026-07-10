# Deterministic AAPL strategy harness

This harness tests zero-API-cost AAPL strategies without routing them through
the LLM decision path.  It exists to answer the economic question quickly and
audibly: did a fixed strategy beat the same-window AAPL buy-and-hold account
after modeled costs?

## Execution contract

- A signal uses completed information through decision-date close.
- The target fills at the next AAPL trading session's adjusted open.
- `adjusted_open = raw_open * adjusted_close / raw_close`.
- Splits and dividends are implicit in the adjusted series and are not credited
  a second time.
- Strategy and buy-and-hold start with fresh cash in every evaluation period.
- Both use the same margin-aware ledger, fractional shares, slippage, and
  commission assumptions.
- Long targets above 1.0 use explicit negative cash and accrue margin interest
  over calendar days.  Shorting is not supported.
- Missing warm-up data, missing fills, misaligned benchmark dates, non-positive
  equity, or exposure outside the declared bound invalidates the run.

The primary curve uses the repository's existing adjusted-open contract and is
valued at the final requested session's adjusted open. Because that cutoff does
not include the final session's intraday move, promotion separately marks the
unchanged final holdings at that session's adjusted close and requires a
material win under both cutoffs.

The default cost case is 5 bps of adverse slippage on every order, zero
commission, and 8% annual interest on negative cash.  The default suite also
tests 10 bps with 12% margin interest and a severe 20 bps / 12% scenario.

## Fixed evaluation hierarchy

The declared strategy-selection audit stops on 2023-12-31. A fixed strategy
hash is then evaluated, without parameter changes, on:

- calendar 2024;
- calendar 2025;
- 2026 year-to-date through the last requested completed AAPL session.

Because this code was developed after all three periods occurred, these windows
are retrospective historical-fit evidence, not provably untouched holdouts.
Every report states that limitation explicitly; only later prospective paper
trading can create genuinely unseen evidence.

The literal requested success gate is positive net excess return versus AAPL
buy-and-hold in every period. A second material gate requires more than one
basis point in every period and under both terminal cutoffs. Capital promotion
also requires the 10 bps / 12% financing stress case, a clean committed
worktree, reproducible source/data hashes, and no integrity errors. Passing
these historical windows is not a profit guarantee or authorization for real
capital.

## Metrics and artifacts

Each run saves a report plus one aligned daily CSV for every cost scenario and
period.  Reports include:

- return, final and minimum equity, annualized return;
- max drawdown with peak, trough, recovery, and underwater duration;
- best/worst daily percentage and monetary result;
- volatility, Sharpe, Sortino, downside deviation, and Calmar;
- exposure distribution, order count, turnover, slippage, fees, and margin
  interest;
- same-ledger AAPL buy-and-hold metrics, relative wealth, tracking error,
  information ratio, and return correlation;
- comparison with static 1.10x AAPL, which exposes whether a result is mainly
  additional market beta rather than timing skill;
- zero-friction replay of the frozen target sequence;
- data and strategy hashes, Git state, runtime, model-call count, and proof of
  `$0.00` external API cost.

Runtime measurement begins before data loading and ends after a complete report
and latest-run pointer have been serialized.  The run fails the time requirement
if it exceeds 3,600 seconds.

## Commands

Run the focused tests:

```powershell
python -m pytest tests/test_deterministic_aapl.py -q
```

Run the default fixed suite with a fresh free Yahoo Finance snapshot:

```powershell
python -m agent_benchmark.deterministic_aapl --refresh-data
```

Subsequent runs reuse the ignored local market-data cache and still record its
content hash:

```powershell
python -m agent_benchmark.deterministic_aapl
```

The default ignored output location is `data/deterministic_aapl/runs`.  An
experiment branch can set `--output-dir` to a versioned documentation directory
so its report and daily evidence are committed with that approach.
