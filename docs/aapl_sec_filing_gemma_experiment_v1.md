# AAPL SEC-filing Gemma experiment v1

This document freezes the next approach before any predictive filing corpus,
Gemma extraction, post-2018 outcome, or final-period result is opened. The
machine-readable version is
`agent_benchmark/sec_filing_gemma_contract.py`.

This is a design and safety contract, not a claim that the approach works.
At the time this contract is committed, no SEC/Gemma trading result exists.

## Plain-language idea

The system will use two different kinds of information:

1. Gemma reads Apple's quarterly and annual SEC reports and converts their
   language into a small set of factual signals, such as whether demand is
   improving, cost pressure is worsening, or management is expressing more
   uncertainty.
2. A conventional numerical model combines those filing signals with market
   context that was already known, such as recent SPY, QQQ, IWM, VIX, and
   Treasury-yield behavior. That numerical model, not Gemma, decides whether
   the strategy is LONG Apple or in CASH.

Gemma never receives Apple's later return, the correct trade, buy-and-hold
performance, or a request to predict the stock. It is a constrained document
reader rather than the trader.

This is broader than the rejected price-only safety rules. It attempts to
learn whether changes in the business and the surrounding market predict a
period in which temporarily holding cash is preferable to holding Apple.

## Chronological experiment

The system simulates trading throughout the historical learning period, but a
result may become a lesson only after its complete 20-session outcome is
known.

| Phase | Availability sessions | Permitted use |
| --- | --- | --- |
| Development | 2000-01-01 through 2018-12-31 | Build the fixed design and select one candidate from the predeclared four-candidate grid. Every training label must also have matured by 2018-12-31. |
| Intermediate | 2019-01-01 through 2023-12-31 | Open once to confirm the frozen development winner. No prompt, feature, threshold, horizon, or hyperparameter change is allowed. If it passes, perform the already declared refit using only labels matured by 2023-12-31. |
| Final | 2024-01-01 through 2026-07-09 | Score 2024, 2025, and 2026 YTD together. The model state is immutable and none of these outcomes can train it. Partial yearly results cannot be exposed. |

This prevents 2024 onward from entering this candidate's downstream numerical
learner. It does not make those years globally untouched: earlier repository
branches have already inspected them, and Gemma's original pretrained weights
may also know 2024. The 2019-2023 period stops being a test after its single
confirmation and becomes part of the final pre-2024 training set. A failed
development or intermediate phase rejects the branch and prevents the next
phase from being opened.

Every phase must eventually have a checksum-bound receipt. The next phase must
consume the exact previous model state, candidate identity, audit identity,
calendar, sealed predictions, score ledgers, no-leverage proof, runtime proof,
and implementation hashes. Changing even one prompt character or threshold
requires a new approach on a new branch.

The current contract deliberately does not authorize any stage transition.
Its access function fails closed until the authoritative runner can perform a
sequential as-of replay: each prediction and action is sealed before its own
future return, while later completed AAPL and market closes remain available
as legitimate inputs to later decisions. Labels, aggregate scores, and gates
stay quarantined from the learner and user until the joint report is released.
The verifier must independently recompute all gates from the bound strategy
and buy-and-hold ledgers. A caller-supplied `passed=true` value is never
sufficient.

## Learning after deployment

The frozen final test and the forever-running system have different jobs:

- The primary 2024-2026 test does not learn from its own test outcomes. This is
  the honest measurement of whether the pre-2024 pattern generalizes.
- After that evaluation is sealed, a live version may keep learning. A
  20-session outcome can be appended as a new lesson only after it has
  completely matured, and the lesson may first affect a later decision.

The live lesson ledger is append-only and chronological. A result that is
partly or wholly in the future can never be used. This preserves continuous
learning without retroactively improving an earlier decision.

Each live lesson also binds its parent lesson, input model state, and output
model state. The numerical edge must agree with the lesson's win/loss label,
and the calendar must prove that the full 20-session horizon matured. The
downstream model may refit before the next filing decision; Gemma's weights,
prompt, schema, and action thresholds remain fixed.

When the system runs past the current calendar, a new calendar may only be an
append-only extension whose entire historical session prefix matches the
previously sealed calendar hash.

The live replay is reported separately from the frozen final score. It cannot
be substituted for a failed frozen test.

## Sentiment inputs

Company sentiment comes from the wording of Apple's official filings. Market
sentiment comes from point-in-time SPY, QQQ, IWM, VIX, and TNX measurements
from the previous completed close.

The large local `news_articles.parquet` file is not used in this version. The
completed audit found that it contains synthetic GDELT event labels rather
than article bodies, has weak timestamps, and has extensive false-positive
Apple matches. Treating it as genuine historical news would create the
appearance of sentiment without reliable evidence.

Real news remains a separate future approach and branch. It must first pass
its own point-in-time source, article-text, timestamp, entity-resolution,
coverage, cost, and runtime audit.

## SEC prerequisite

The existing 24-accession SEC audit is a source/parser audit only. It is not
the predictive corpus. No corpus or Gemma run is authorized until a production
audit:

- has `overall_pass=true`;
- was produced from the clean committed audit implementation;
- is verified against an externally retained `checksums.json` hash and source
  commit;
- used fresh official SEC evidence rather than mock or cache-supported
  production evidence; and
- made zero model and paid-API calls.

The production audit needs a private, valid SEC user-agent contact in local
configuration. That contact must never be committed or printed.

The currently frozen session calendar ends in January 2025. Before corpus
acquisition, a new checksum-bound calendar must cover at least through
2026-07-10. Scored prices still end on 2026-07-09; the extra session is needed
to assign conservative availability around the cutoff.

## Corpus

The v1 corpus contains the complete metadata-eligible universe for Apple CIK
`0000320193`:

- exact form `10-K` or `10-Q`;
- non-amended reports only;
- the sole primary document named by SEC Submissions metadata;
- conservative point-in-time availability successfully reconciled across the
  official SEC sources; and
- no semantic, price, or return-based document selection.

The corpus cannot take the first N reports, drop slow reports, keep only
interesting language, or substitute another filing. If the complete universe
exceeds a resource ceiling, v1 fails rather than samples it.

Minimum adequacy is:

- at least 72 development filings;
- at least 19 intermediate filings;
- for every completed year from 2000 through 2025, at least one 10-K and two
  10-Q filings assigned by conservative availability session; and
- every eligible 2026 filing through the cutoff, without inventing a
  full-year minimum.

Development, intermediate, and final text live in physically separate sealed
artifacts. The model-call ceilings are respectively 80, 20, and 12.

The narrow periodic-report scope is intentional. Earnings-release exhibits
and 8-Ks could add useful information, but the audited primary 8-K can be
mostly a cover page and deterministic exhibit selection needs a separate
contract. That will be a separate branch, so its value can be measured rather
than hidden inside this result.

## Point-in-time availability

For each filing, the system reconciles SEC Submissions metadata, the quarterly
`master.idx`, the raw SGML acceptance timestamp, the accession
`index.json`, and the primary document.

Availability is the first complete AAPL session strictly after the latest
defensible acceptance date, filing date, or filing-date-change date. The
filing may enter the decision only after that session closes. A trade then
fills at the next adjusted open.

A missing exact acceptance timestamp never permits same-day use. If the SEC
filing date remains defensible, the filing falls back conservatively to the
first complete session strictly after that filing/change date; the full
corpus must still have at least 95% exact acceptance timestamps. Missing or
conflicting evidence with no defensible filing date excludes the filing.
Report date or fiscal period never substitutes for public availability.

## Filing blinding

The frozen preprocessor is
`issuer-relative-period-grounded-sentences-v1`. It:

- removes issuer, ticker, CIK, executive, product, and exact-date identity;
- replaces absolute currency, share, percentage, and accounting values with
  typed redaction tokens;
- preserves directional language such as improved, declined, withdrawn, and
  uncertain;
- labels current-filing sentences `C####` and prior same-form sentences
  `P####`;
- supplies the prior same-form filing in the same request when one exists;
- permits at most 72 sentences, 220 characters per sentence, and 20,000 UTF-8
  bytes; and
- contains no prices, returns, labels, actions, forecasts, or benchmark
  results.

The exact prompt, schema, preprocessor, model digest, generation options,
current and prior filing hashes, candidate-wide redacted-input manifest,
calendar, audit, and implementation sources are part of the candidate
identity. The identity also binds a checksum-bound contract-specific lexicon
of known issuer/product/executive/location identities, the Ollama runtime
template/system/parameter fingerprint, the clean Git source commit, the
source-tree hash, the sealed corpus universe, and every
decision/security-critical dependency.

Filing identity, form, stage, availability, accession, and source hashes live
only in a validation envelope. They are never serialized to Gemma. The
dedicated client may serialize only the validated model payload: the fixed
anonymous system instruction, canonical C/P sentence JSON, exact output
schema, model name, and frozen generation options. Its byte-equivalent hash
must match the candidate-bound redacted-input manifest.

## Gemma output

The exact schema version is `sec-filing-extractor-v1`. It allows only:

- document quality: usable, thin, or unusable;
- ten categorical dimensions: demand, pricing power, gross margin, operating
  cost pressure, capital allocation, liquidity, forward guidance, supply
  chain, legal/regulatory conditions, and management uncertainty; and
- six Boolean flags: new material risk, withdrawn guidance, liquidity stress,
  restructuring/impairment, internal-control weakness, and management
  transition.

Each dimension contains a fixed current-impact enum, a fixed
change-versus-prior enum, and evidence sentence IDs. Each flag contains a
Boolean and evidence sentence IDs.

A current claim needs current-filing evidence. A comparative claim needs both
current and prior evidence. A true flag needs current evidence; a false flag
must cite nothing. No extra fields or free text are allowed. An invalid output
makes the filing unavailable: there is no repair call, retry, or discretionary
imputation.

## Foundation-model contamination caveat

The installed `gemma4:12b` reportedly has training knowledge through January
2025. Candidate-level chronology can prove that this downstream learner did
not consume 2024 outcomes, but it cannot prove that Gemma's original weights
never encountered a 2024 Apple filing.

For that reason, every result must say
`parametric_contamination_risk = unresolved_but_bounded`. Identity and
absolute-value redaction, evidence grounding, no market outcomes in the
prompt, and no direct trading authority reduce the risk; they do not erase
it. The result is a frozen retrospective test of this approach, not globally
pristine evidence that the foundation model knew nothing about 2024.

The strongest clean evidence will eventually be prospective paper trading.
Alternatively, a later branch can use a model with a documented pre-2024
training cutoff.

The 2024-2026 result is also a reused historical holdout at the repository
level because earlier approaches have already reported those years. Before
stage access is enabled, this candidate must register one immutable attempt ID
in a repository-wide reveal registry that declares every earlier final
attempt. Selecting the best branch after repeatedly opening the same final
years is forbidden. Results are therefore approach-specific retrospective
evidence; only locked prospective paper trading can be called globally
pristine.

## Numerical learner and ablation

The downstream model has two frozen heads:

- probability that a 20-session CASH episode beats holding AAPL after
  10-basis-point costs; and
- expected active log edge of that CASH episode.

At each filing event it receives the exact frozen AAPL price-regime fields,
including returns, volatility, drawdown, moving-average distance, gap, and
relative-strength history, plus the previous-close SPY/QQQ/IWM/VIX/TNX market
sentiment fields. Gemma enums use one fixed signed/stated encoding. The
calendar-only ablation receives the same AAPL and market history.

To control overfitting with roughly quarterly observations, the ten filing
dimensions are reduced to frozen commercial, financial, and risk/outlook
aggregates plus flag, coverage, and document-quality fields. Both prediction
heads are fixed ridge-regularized linear models with training-only robust
scaling. There is no feature selection, interaction search, or hyperparameter
tuning after this contract.

The four possible action gates are frozen before development:

| Candidate | Probability gate | Expected-edge gate |
| --- | ---: | ---: |
| `p50_e0` | 0.50 | 0 |
| `p55_e0` | 0.55 | 0 |
| `p50_e25` | 0.50 | 0.0025 |
| `p55_e25` | 0.55 | 0.0025 |

An accepted signal creates one fixed 20-session CASH episode; otherwise the
strategy remains LONG. A signal during an active episode cannot extend it.

The full semantic predictor is compared with an otherwise identical
filing-calendar-only ablation. The ablation receives the same filing dates,
forms, missingness, market context, learner, folds, and thresholds, but no
Gemma meaning. The full approach must beat this ablation, not merely beat
buy-and-hold by chance.

Development uses five expanding chronological folds:

1. train through 2004, test 2005-2007;
2. train through 2007, test 2008-2010;
3. train through 2010, test 2011-2013;
4. train through 2013, test 2014-2016; and
5. train through 2016, test 2017-2018.

Candidate selection is deterministic: keep only candidates passing every
development gate at both 5 and 10 bps, then rank by lower 10-bps Brier score,
higher 10-bps active edge, and finally the frozen candidate order.

## Success gates

The machine-readable contract contains the exact thresholds. Important
development requirements at both 5 and 10 bps include:

- total active log edge of at least 0.02;
- edge excluding the best year of at least 0.005;
- positive performance in at least four of five folds;
- annual win rate of at least 55%;
- negative-buy-and-hold-year win rate of at least 60%;
- enough distinct CASH episodes to avoid a one-trade result;
- performance that is not dominated by one year; and
- better probability accuracy and active edge than causal climatology and the
  filing-calendar-only ablation.

The single 2019-2023 intermediate reveal must pass its own material 5-bps and
positive 10-bps gates, including the ablation comparison. Failure rejects the
branch; it does not authorize tuning.

Final success requires all of the following:

- in each of 2024, 2025, and 2026 YTD, at least 0.5% active edge at 5 bps;
- at least 2% continuous active edge across the combined final ledger at
  5 bps;
- positive active edge in every period and continuously at 10 bps;
- the result passes using both adjusted-open and terminal-adjusted-close
  valuation;
- at least one CASH episode in every period and at least six in total;
- episode win rate of at least 55%;
- positive mean episode edge at 10 bps;
- no single episode supplies more than half of positive edge; and
- maximum drawdown is no more than one percentage point worse than
  buy-and-hold; and
- the full semantic model beats the frozen no-semantics ablation continuously
  and in at least two of the three final periods.

These gates deliberately include periods in which buy-and-hold was negative.
A strategy that only looks good while Apple rises is insufficient.

## Trading and benchmark ledger

The strategy and buy-and-hold start with the same USD 1,000 and use the same
AAPL sessions and adjusted prices.

The only permitted target exposures are exactly 0 and 1. There is no
leverage, shorting, borrowing, negative cash, fractional exposure, or margin
interest. Decisions are made after a completed close and fill at the next
adjusted open.

A decision made after the 2026-07-09 close would fill outside the scored
cutoff and is recorded but not scored. An episode already open at a reporting
cutoff is valued at the cutoff under both terminal conventions and can never
become a training lesson inside that frozen score. Period results are sums of
daily active log increments from one continuous ledger; Brier statistics use
only predictions whose complete horizon matured by the relevant cutoff.

Every saved ledger must pass the existing independent no-leverage proof and
reconcile daily strategy-versus-buy-and-hold log increments. A declaration in
a manifest is not sufficient.

## Local model and zero-cost rule

Gemma may be called only through
`http://127.0.0.1:11434/api/chat` using the exact locally installed
`gemma4:12b` digest. Environment proxies, redirects, model pulls, streaming,
thinking mode, retries, and repair attempts are disabled. Generation uses
temperature 0, seed 0, context 6144, and output limit 512.

Official SEC HTTPS requests are permitted for the free filing corpus. Model
traffic is loopback-only. Paid API calls and estimated external cost must
remain exactly zero.

## Time and resource ceiling

The complete experiment has a hard one-hour limit:

- SEC acquisition: at most 720 seconds, 1,000 requests, 1.5 GiB, and two
  requests per second;
- local Gemma extraction: at most 2,160 seconds;
- deterministic fitting, simulation, verification, and sealing: at most 480
  seconds; and
- total wall-clock time: at most 3,600 seconds.

There are 240 seconds of unallocated contingency. A five-filing
development-only latency preflight must project the complete run inside the
remaining budget before 2019 or later material can be opened. Documents
cannot be dropped to make the estimate pass.

These are ceilings, not a promise that a first run will take exactly one
hour. The run fails cleanly if a phase exceeds its budget; partial output can
never pass.

## Branch and artifact discipline

This approach lives on `codex/aapl-sec-filing-gemma-v1`. The frozen contract
must be committed and pushed before semantic corpus acquisition or model
work.

Later milestones are also committed before the next reveal:

1. production SEC audit and extended calendar;
2. metadata-only universe, corpus selector, extractor, and model client;
3. sealed development result and selected candidate;
4. sealed intermediate result and pre-final refit state; and
5. one sealed final result containing 2024, 2025, and 2026 YTD.

A materially different source family, prompt, learner, threshold set, or
filing universe is a new approach on a new branch. Rejected results remain
preserved rather than overwritten.

## What has and has not happened

At contract-freeze time:

- the SEC audit/parser implementation exists and its offline tests pass;
- this experiment contract and its mutation tests exist;
- stage access remains intentionally disabled until the authoritative sealed
  prediction/ledger/gate verifier is implemented;
- the final all-stage runtime summary is diagnostic only; stage-specific and
  cumulative receipts plus the five-development-filing latency preflight must
  be implemented before it can unlock anything;
- the future verifier must derive the eligible universe from the sealed SEC
  catalogue, bind exact AAPL/SPY/QQQ/IWM/VIX/TNX input hashes, and prove that
  every training row is one exact matured filing event tied to its extraction
  and price-label ledger;
- live extensions must be anchored to the externally sealed prior lesson
  count, tip, and model state and contain all and only newly matured bound
  execution receipts;
- no production SEC audit has run because a valid private SEC contact is not
  configured;
- no predictive filing corpus has been downloaded;
- Gemma has not processed a filing for this experiment;
- no 2019-2023 confirmation result has been opened; and
- no 2024, 2025, or 2026 performance has been calculated.

The next implementation milestone is to bind the authoritative SEC audit,
extend and seal the session calendar, implement the deterministic corpus and
redaction pipeline, implement stage-specific runtime receipts and the global
reveal registry, and implement the sequential sealed prediction/ledger
verifier. That machinery must be committed before opening filing meaning or
any later-stage outcome.
