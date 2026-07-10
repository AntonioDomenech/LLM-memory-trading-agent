# CFTC COT AAPL sentiment experiment

This branch tests whether public futures-positioning sentiment adds a robust
risk-off signal to an unleveraged AAPL long/cash account. It makes no LLM or
paid API calls. It is a bounded candidate experiment, not permission to tune on
2024 onward.

## Frozen signal family

The complete family contains four variants only:

- 26 released weeks with a 0.75 z-score threshold;
- 26 released weeks with a 1.25 z-score threshold;
- 52 released weeks with a 0.75 z-score threshold;
- 52 released weeks with a 1.25 z-score threshold.

Each decision uses the newest complete CFTC Legacy Futures-Only release that
was conservatively available by that date. Availability is the report date plus
eight calendar days. A release older than 14 days is neutral. The current value
is standardized against earlier released observations only.

The policy moves to 100% cash only when at least two conditions hold:

1. Nasdaq-100 mini noncommercial net positioning is unusually low.
2. Nasdaq positioning relative to E-mini S&P 500 positioning is unusually low.
3. VIX-futures noncommercial net positioning is unusually high.

Otherwise it holds 100% AAPL. Partial exposure, leverage, shorting, borrowing,
negative cash, and outcome-driven parameter updates are absent.

The official sources are the [CFTC Legacy Futures-Only dataset](https://publicreporting.cftc.gov/Commitments-of-Traders/Legacy-Futures-Only/6dca-aqww),
the [CFTC explanation and release timing](https://www.cftc.gov/MarketReports/CommitmentsofTraders/AbouttheCOTReports/index.htm),
and the [CFTC historical special announcements](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm).
CFTC government information is addressed by its [Web Policy](https://www.cftc.gov/WebPolicy/index.htm).

## Physical evidence separation

Development downloads end on 2018-12-31. The development runner cannot accept
another end date and rejects a response containing an out-of-range row,
incorrect hash, nonofficial source, missing contract, or materially truncated
history. Every data download is paired with a separately hashed official
Socrata `count(*)` query using the identical fixed contracts and date bounds.
The payload must contain exactly that many rows; both raw responses and their
exact query URLs are sealed, so scattered or common missing weeks cannot hide
inside a percentage-based coverage tolerance.

Structural availability is checked separately from download completeness. The
two index contracts must remain at least 99% weekly, with maximum gaps of 28
days for E-mini S&P and 14 days for Nasdaq. The official early VIX series is
sparser (711 of roughly 753 pre-2019 weeks, including a 168-day 2008-09 gap),
so its frozen floor is 94% with a maximum 175-day gap. This does not fill or
invent observations: after 14 stale days the policy is forced back to long
AAPL. The first attempted run stopped before scoring under the former generic
95%/28-day assumption; its immutable diagnostic is saved under
`e/cftc_cot_v1/development-attempt-1-data-contract-failure/`.

The price snapshot includes 1999-12-31 solely as the causal pre-fill warm-up;
the simulated account and scored outcomes begin on the first 2000 trading
session. Complete VIX
futures positioning is unavailable until July 2004, so the frozen policy stays
100% long AAPL during 2000-2003 and the early-2004 warm-up rather than inventing
sentiment. Those years remain in the annual and downside diagnostics; signal
selection folds begin only when all three CFTC markets can be observed.

Only if at least one variant passes every development gate is one winner frozen
using the declared tie-break. Before confirmation, the exact development raw
CFTC bytes, price bytes, anomaly exclusions, candidate results, source hashes,
the CFTC count-proof bytes, Git ancestry, manifest, and artifact checksums are
verified. Each run directory contains a local `.gitattributes` rule disabling
text conversion, so Windows line-ending settings cannot change committed
evidence bytes. Confirmation enumerates and replays every development artifact
directly from the single Git commit captured at startup rather than reopening
mutable worktree files.

Confirmation downloads only the 2019-2023 suffix. It reuses the saved 1997-2018
CFTC prefix rather than downloading revised history. It evaluates only the
frozen winner. Its outcomes cannot switch to another variant, but its gates do
accept or reject the entire approach. It is therefore pre-2024 development
validation, never final test evidence. A one-shot access record is atomically created under Git's common
directory before either confirmation download, so alternate worktrees share
the same reservation. That reservation is never overwritten; completion uses a
second exclusive record and first verifies the exact reservation bytes and
access id. Completion is written only after the complete confirmation report,
ledger set, and checksum manifest have been read back successfully. The
completion record binds their checksum-manifest hash.

Neither phase can request or parse post-2023 market observations. The static
anomaly calendar contains later administrative metadata so future live handling
can be audited; this metadata is not a market observation or outcome.

## Development gates

Both the 5-basis-point and 10-basis-point same-ledger scenarios must pass:

- total active-log edge of at least 2%;
- monthly-end rolling 252-session win rate of at least 60%;
- monthly-end rolling 756-session win rate of at least 70%;
- annual win rate of at least 55% and median annual active-log edge of at least 5 bps;
- positive edge in at least three of four temporal folds;
- at least 30 deliberate cash days and 12 cash episodes, but no more than 20% cash days;
- at least 0.5% active-log edge after removing the best year;
- the largest positive year supplies no more than 45% of all positive annual edge;
- a complete no-leverage proof.

Passing variants are ranked by the weakest 10-bps fold, then total 10-bps edge,
then fewer cash days, then lexical variant id. If none pass, the experiment
stops without opening 2019-2023.

## Confirmation gates

The selected policy must pass both 5 and 10 bps on the continuous 2019-2023
account:

- at least 1% total active-log edge;
- at least 60% monthly-end 252-session wins;
- at least three of five annual wins and at least 5 bps median annual edge;
- at least 10 deliberate cash days and four episodes, with no more than 20% cash days;
- at least 1 bp active-log edge during 2022;
- a complete no-leverage proof.

Absolute returns and behavior in negative AAPL years, the 2008 crisis, Q4 2018,
the COVID crash, and 2022 are saved as diagnostics. They cannot compensate for
a failed gate.

## Commands

Run tests and commit the clean implementation before accessing development
data:

```powershell
python -m pytest tests/test_cftc_cot.py tests/test_cftc_cot_policy.py tests/test_cftc_cot_experiment.py -q
python -m agent_benchmark.cftc_cot_experiment development --output-dir e/cftc_cot_v1
```

If and only if development passes, commit its immutable artifacts before the
one-shot confirmation command:

```powershell
python -m agent_benchmark.cftc_cot_experiment confirmation --output-dir e/cftc_cot_v1 --development-report <development-run-directory>
```

Commit the complete confirmation directory without changing its bytes, then
create the final external verification record. A confirmation result is not
authoritative until this command succeeds against clean committed Git blobs:

```powershell
python -m agent_benchmark.cftc_cot_experiment verify-confirmation --confirmation-report <confirmation-run-directory>
```

Even a confirmation pass remains retrospective evidence because these years
have already occurred. It would authorize one separately frozen 2024-2026
historical audit, not real-capital deployment. Locked future paper trading is
still required before risking money.
