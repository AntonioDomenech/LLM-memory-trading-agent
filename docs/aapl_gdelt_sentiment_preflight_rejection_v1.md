# AAPL GDELT sentiment preflight rejection v1

## Status

Rejected before model fitting or outcome evaluation. No 2024, 2025, or 2026
return was opened for this approach.

## Intended approach

The proposed experiment would have aggregated GDELT `AvgTone`,
`GoldsteinScale`, event volume, and conflict indicators into causal daily
sentiment features, then trained a chronological long/cash AAPL predictor.
The strategy would have remained unleveraged: either 100% AAPL or 100% cash.

## Blocking data-quality evidence

The authoritative audit in
`docs/aapl_point_in_time_text_data_audit_v1.md` establishes that the local
`data/warehouse/parquet/news_articles.parquet` file is not an admissible AAPL
news or sentiment source:

- all 182,899,080 titles are synthetic `GDELT event` labels rather than
  publisher headlines;
- only 2.17% of the 1,369,228 nominal AAPL rows contain the matching term in
  the synthetic actor title, while 97.83% are URL-only matches;
- at least 658,577 nominal AAPL rows are demonstrable false positives caused
  by strings such as `pineapple`, `grapple`, `littleapple`, `applevalley`, or
  `rappler`;
- `published_at` is derived from the event day at midnight and has no timezone;
  the ingestion does not preserve GDELT `DATEADDED`, so historical
  availability at a decision time cannot be proved; and
- the file has no real headline or article body that Gemma could interpret.

Lagging the rows by one day would not repair false entity association or the
missing observed-availability timestamp. Using the data would therefore create
an apparently sophisticated sentiment model whose input identity and
chronology are not trustworthy.

## Decision

Do not fit, tune, or test this GDELT AAPL sentiment family. Preserve this
preflight rejection so the same 1.37-million-row count is not mistaken for
usable AAPL news coverage in a later branch.

The next permitted sentiment-oriented approach must use independently
timestamped market information. Cross-sectional sector breadth, IWM risk
appetite, and VIX are eligible because their completed-close values can be
aligned exactly to each AAPL decision session.
