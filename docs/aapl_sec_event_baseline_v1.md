# AAPL SEC filing-event cash baseline v1

Status: **preregistered; no SEC-event return has been calculated**.

Branch: `codex/aapl-sec-event-baseline-v1`

Parent evidence commit: `3e248208bd6d653102c652c719e451731ac991be`

## Question

Does the date of an Apple 10-K or 10-Q filing, without reading or classifying
its contents, identify a repeatable 20-session period when holding cash beats
holding AAPL?

This is the simplest SEC control. It must be measured before asking an LLM to
read filings. A failure means filing timing alone is not useful; it does not
prove that filing contents are useless.

## Frozen evidence

Only the already authenticated SEC development receipts are eligible. The
source is the ignored private directory
`data/aapl_sec_gemma_lean_evidence_v3_8/development/parse_receipts`; no new SEC
request is allowed for this experiment and no private contact identity may be
published.

A row is eligible only when all of these are true:

- `source_evidence.seal_row` exists;
- the receipt's top-level `stage == "development"`;
- `acquisition_stage == "development"`;
- `stage_assignment == "development"`;
- `form` is exactly `10-K` or `10-Q`;
- `exact_acceptance` is exactly `true`; and
- `availability_session` is from `2000-01-01` through `2018-12-31`, inclusive.

Before any price join, this produces exactly 75 filings: 18 10-K rows and 57
10-Q rows, sequences 124 through 198, with availability sessions from
`2000-02-02` through `2018-11-06`. There are 74 unique availability sessions;
two filings share `2007-01-03`.

The public-safe rows are sorted by `(availability_session,
accession_number)`. Each row contains only `accession_number`,
`acquisition_stage`, `availability_session`, `exact_acceptance`, `filing_date`,
`form`, `frozen_prefix_sha256`, `sequence`, `source_evidence_sha256`, and
`stage_assignment`. Their canonical UTF-8 representation is sorted-key,
compact JSON followed by one line feed:

- bytes: `30677`
- SHA-256: `eea1ff57f6f9f2db31ee341fb48494d81b7c56e9e6bf889278b955b5ff15dabb`

The physically bounded market input is
`e/chronological_exhaustion_expert_v1/authorized_inputs/aapl_spy_qqq_through_2018.csv`:

- literal bytes: `590761`
- literal SHA-256: `9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1`
- first session: `1999-03-10`
- last session: `2018-12-31`
- rows: `4986`
- canonical bounded-result SHA-256:
  `31b56551b8d1b837f2e69178ab7f206b6bf3c19d11d9d47be0e33cf501db3f45`

No 2019-or-later filing, price, prediction, or performance is admissible in
development.

## Frozen trading rule

The account starts with EUR 1,000 and is long AAPL by default. For every
eligible filing whose conservative `availability_session` is market session
`t`:

1. make the CASH decision after the close of `t`;
2. sell AAPL at adjusted open `t+1`;
3. remain in cash for exactly 20 open-to-open return intervals; and
4. buy AAPL again at adjusted open `t+21`.

Equivalently, target exposure is zero on decision rows `t` through `t+19` and
returns to one on decision row `t+20`. Multiple same-day or overlapping filing
windows are combined by union. A later filing may therefore extend an active
cash interval. Costs occur only when the combined 0/1 target changes.

The 20-session horizon is inherited unchanged from the already preregistered
SEC overlay contract. It is not selected from this experiment's returns.

There is no model, threshold search, parameter search, filing-text inspection,
online learning, interest on cash, shorting, leverage, borrowing, or negative
cash.

## Fair comparison

Run one continuous account from `2000-01-03` through `2018-12-31`. Compare the
rule with always-long AAPL using the same starting money, adjusted-open fills,
price source, corporate-action adjustment, dates, final valuation, and ledger.

Run both:

- base cost: 5 basis points on every changing leg;
- stress cost: 10 basis points on every changing leg.

Report full-precision and readable results for total return, AAPL return,
percentage-point excess, relative ending wealth, annual results, wins/losses/
ties, mean and median annual excess, maximum drawdown, trades, cash time,
turnover, costs, negative-AAPL years, and every unleveraged-account proof.

## Development decision

The idea advances to 2019-2023 confirmation only if **all** of these trading
evidence gates pass:

1. cumulative relative wealth versus AAPL is positive at both 5 and 10 bps;
2. at 10 bps, winning calendar years outnumber losing calendar years;
3. at 10 bps, both mean and median annual excess are positive;
4. at 10 bps, cumulative relative wealth remains positive after removing the
   strategy's single best active year; and
5. at 10 bps, relative wealth aggregated across negative-AAPL calendar years
   is positive.

Ties are reported separately. The always-long control must match the
same-ledger AAPL benchmark, and all chronology/no-leverage checks must pass.
Leakage, a bad source hash, incorrect fills, leverage, or corrupted accounting
invalidate the run. Runtime estimates, harmless count-message differences,
extra standard-library imports, and similar bookkeeping details are warnings;
they do not stop or reject the trading test.

If any trading-performance gate fails, preserve the rejection and do not open
2019 or later. Do not create a cosmetic v1.1. The next experiment, if justified,
must test a materially different source of edge, such as filing content rather
than filing timing.

## Runtime and outputs

Expected runtime is under one minute on the current machine. There is no hard
runtime cutoff. Save the public-safe filing manifest, target history, ledgers,
metrics, gate report, checksums, and a plain-language terminal decision. Then
update the central approach comparison and commit and push the complete result.
