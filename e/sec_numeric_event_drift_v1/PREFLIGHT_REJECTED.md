# Preflight rejection: AAPL SEC numeric event drift v1

## Decision

Do not implement or score this version. The local SEC facts are technically
usable for a lean event study, but the AAPL-only development sample is too
small to support a credible predictive trading claim. This decision was made
without loading any 2019+ outcome.

The proposed policy would have combined point-in-time changes in Apple's
10-Q/10-K numbers with AAPL/QQQ/VIX market sentiment, then learned whether a
20-session move to cash should replace continuously holding AAPL. It would
have been 100% AAPL or 100% cash, with no leverage, borrowing, or shorting.

## What is locally available

The bounded source is `data/warehouse/parquet/sec_facts.parquet`, SHA-256
`E6A1397C13ECFBFD229EFA103D026048864E6C7632B1C962AACAF44210FEAD67`.
For exact AAPL forms `10-Q` and `10-K` through 2018 it contains:

- 1,237 fact rows;
- 38 pseudo-events from 2009-07-22 through 2018-11-05;
- 28 quarterly reports and 10 annual reports; and
- no AAPL SEC numeric coverage for 2000-2008.

All 38 events have complete pre-2019 market history and a mature 20-session
outcome. Their minimum fill-to-fill separation is 57 sessions, so fixed
20-session episodes cannot overlap.

Core same-filing comparisons are complete:

| Causal feature | Usable events |
|---|---:|
| Revenue year-over-year change | 38 / 38 |
| Diluted-EPS year-over-year change | 38 / 38 |
| Net-income year-over-year change | 38 / 38 |
| Operating-margin year-over-year change | 38 / 38 |
| Cash/assets and liabilities/assets | 38 / 38 |
| Operating-cash-flow margin change | 26 / 38 |
| Capex/revenue change | 16 / 38 |
| Long-term debt/assets | 14 / 38 |

The optional fields are missing in calendar blocks. Using their missingness as
a model input could accidentally encode time rather than business quality.

## Required causal handling

The retained `fiscal_year` and `fiscal_period` describe the filing, not each
fact's economic period. A 10-Q can contain both quarter-only and year-to-date
values, and a 10-K can contain annual plus quarterly values. There are 206
same-filing concept/end groups with different durations across 28 of the 38
events.

Consequently, the existing `canonicalize_fundamentals` helper is not suitable:
its latest-value selection ignores duration and can combine, for example,
annual operating income with quarterly revenue. Any future implementation
must instead:

1. filter exact non-amended forms `10-Q` and `10-K`;
2. group facts by `(filed_date, form)`;
3. use only versions disclosed in that exact filing event;
4. match current and prior-year quarter durations for 10-Q income features;
5. match annual durations for 10-K features; and
6. match concept and unit, allowing a revenue alias only when it contains both
   the current and comparative value.

This matters because 39 historical fact contexts change value in later
filings, including split-adjusted EPS and accounting recasts. Selecting the
latest version would introduce hindsight.

The data has `filed_date` but no accession number or acceptance timestamp.
The only defensible timing rule is conservative:

1. treat the first NYSE session strictly after the filing date as the
   availability/decision session;
2. use market information only through that completed close;
3. fill at the following adjusted AAPL open;
4. remain in cash for exactly 20 open-to-open intervals; and
5. let the outcome become a lesson only after the exit open has occurred and
   that session has closed.

## Why the sample is inadequate

Using 2009-2012 as a minimal warm-up provides only 14 matured lessons. The
honest 2013-2018 prequential stream then contains:

- 24 decisions;
- six calendar test years; and
- only five episodes in which cash beat AAPL after 10-bps-per-side costs.

That is not enough for a multivariable model, nested threshold selection, or a
reliable long-run conclusion. A strategy could appear excellent by identifying
only a few events by chance. Relaxing the evidence gates would hide that
problem rather than solve it.

The repository's 49 other stocks cannot safely provide outcome labels,
feature scaling, priors, or thresholds: they are a handpicked present-day
large-cap universe, so pooled historical learning would contain survivorship
bias.

## Frozen pilot that was not run

If this limited dataset is ever used only as a falsification pilot, freeze one
policy rather than search alternatives:

- three complete SEC inputs: revenue growth, operating-margin change, and a
  symmetric diluted-EPS change;
- two market-sentiment inputs: AAPL filing reaction relative to QQQ and the
  trailing 20-session VIX change;
- one equal-weight fundamental-risk score and one equal-weight market-risk
  score, normalized only with expanding AAPL history;
- one strongly regularized logistic cash-win head and one regularized
  expected-edge head;
- one fixed action gate and no feature, interaction, model, or threshold grid;
- 2009-2012 warm-up followed by causal annual replay; and
- SEC-only, market-only, filing-calendar, and always-long ablations on the
  exact same events and ledger.

It should still require positive 5- and 10-bps total edge, positive edge after
removing the best year, at least eight cash episodes, at least four of six
winning forward years, at least 55% episode wins, positive mean and median
episode edge, non-concentrated gains, and full-model superiority to both
information ablations. With only five helpful opportunities, an honest pass is
unlikely.

## Worthwhile next version

Acquire and seal Apple's official 2000-2008 filings before revisiting this
family. A useful corpus should preserve accession number, SEC acceptance time,
raw bytes, primary-document identity, and hashes, bringing the pre-2019 AAPL
event count to roughly 72-76. Those documents can support a separate local
Gemma filing-reader approach; the current numeric parquet cannot.

This preflight rejection cost no model/API calls and opened no later outcome.
