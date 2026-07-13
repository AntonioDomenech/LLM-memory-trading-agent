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

The cumulative live binding ledger retains every lesson ID, Apple accession,
decision session, feature hash, prediction receipt, market frame, and both
ledger slices. A later batch cannot reuse any prior lesson, accession, or
prediction receipt under a new name.

When the system runs past the current calendar, a new calendar may only be an
append-only extension whose entire historical session prefix matches the
previously sealed calendar hash. The present implementation recognizes the
exact NYSE trading-session-date sequence only through 2026-07-10 and fails
closed beyond it; a later live extension requires updated authoritative
calendar code and newly sealed official-source evidence, not merely a list of
weekdays.

The live replay is reported separately from the frozen final score. It cannot
be substituted for a failed frozen test.

## Sentiment inputs

Company sentiment comes from the wording of Apple's official filings. Market
sentiment comes from point-in-time SPY, QQQ, IWM, VIX, and TNX measurements
through the completed filing-decision session close. The decision is made only
after that close and fills at the next adjusted open, so the same-session close
is known information rather than a future value. Market history is requested
from 1998-01-01 to support the frozen 252-session lookbacks; where a required
series does not yet have enough history, that filing's prediction is marked
unavailable rather than backfilled.

The exchange calendar is versioned. The immutable v1 prefix ends 2025-01-10;
v2 appends the exact NYSE trading-session dates through 2026-07-10 while
preserving all 6,295 earlier sessions byte-for-byte. Early-close dates remain
sessions because this artifact freezes dates, not trading hours. The v2
calendar and its official NYSE source evidence must be sealed before the SEC
audit can become a candidate binding.

A separate, equally exact market-feature calendar begins 1998-01-02 and
contains 504 pre-2000 sessions. It exists only to make the declared
252-session lookbacks auditable; it does not move the filing-universe or score
start before 2000.

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

The versioned v2 session-date calendar now covers through 2026-07-10 and
preserves the complete v1 prefix ending 2025-01-10. It is code-verified but
does not become a candidate binding until its official NYSE source bytes and
external checksum are sealed. Scored prices still end on 2026-07-09; the extra
session is needed to assign conservative availability around the cutoff.

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

The 24-slot production prerequisite audit reconciles its sampled filings
across SEC Submissions, quarterly `master.idx`, raw SGML acceptance data,
accession `index.json`, and the primary document. It is parser evidence, not
an exhaustive proof for the predictive corpus. The complete predictive
universe is instead authenticated from the exact current SEC Submissions file
plus every historical Submissions file it references; each selected primary
document is then fetched from its internally derived official SEC URL and
bound by exact raw and normalized byte hashes. This v1 approach makes no
master-index or SGML-reconciliation claim for 2025-2026.

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
- requires canonical ASCII and removes calendar-month and weekday language so
  Unicode digits, currency marks, or identity homoglyphs cannot bypass the
  scanners;
- preserves directional language such as improved, declined, withdrawn, and
  uncertain;
- labels current-filing sentences `C####` and prior same-form sentences
  `P####`;
- supplies the prior same-form filing in the same request when one exists;
- permits at most 72 sentences, 220 characters per sentence, and 20,000 UTF-8
  bytes; and
- contains no prices, returns, labels, actions, forecasts, or benchmark
  results.

Executive-name removal combines a frozen historical/current leadership roster
with bounded deterministic context rules for honorifics, leadership titles,
title appositions, appointment transitions, and speech attribution. Those
generic rules cover one-to-three capitalized tokens and a bounded set of
one-or-two lowercase name particles. They are not general named-entity
recognition: names with other casing or scripts, longer structures, unlisted
particles/titles/verbs/transitions, or unrelated prose contexts are not
guaranteed to be removed. Every accepted sentence must still pass the exact
roster and bounded-context residual checks.

The exact prompt, schema, preprocessor, model digest, generation options,
calendar, audit, and implementation sources are part of the candidate
identity. The identity also binds the exact contract-owned canonical lexicon
and hash for known issuer/product/executive/location identities, while source
identity binds the bounded generic context grammar, the Ollama runtime
template/system/parameter fingerprint, the clean Git source commit, the
source-tree hash, official calendar-source evidence, the exact session hash,
both the reproducibility and semantic hashes of the sealed corpus universe,
and every
decision/security-critical dependency. Current/prior filing hashes and each
exact model-payload hash are bound by a separate per-event redacted-input
receipt. Version 3 also binds the exact owned preprocessing receipt,
preprocessed-event hash, SEC reader receipt, and, outside development, carry-in
reader receipt. A fresh preprocessing worker receives only that current filing
and its immediate prior same-form filing; it never receives the rest of the
stage.
Development cannot read or preprocess intermediate or final filing text.

Filing identity, form, stage, availability, accession, and source hashes live
only in a validation envelope. They are never serialized to Gemma. The
dedicated client may serialize only the validated model payload: the fixed
anonymous system instruction, canonical C/P sentence JSON, exact output
schema, model name, and frozen generation options. Its byte-equivalent hash
must match the externally pinned receipt for that exact filing event.

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
makes Gemma meaning unavailable: there is no repair call, retry, or
discretionary imputation.

A sealed invalid model output is encoded with neutral semantic content and
`semantic_output_unavailable = 1`; the identical missingness indicator is
given to the ablation. A valid `unusable` output is also semantically neutral,
but remains a validated output and therefore does not set that indicator.
Missing or unauthenticated extraction evidence is different: it is an
integrity failure and makes the complete prediction row unavailable.

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
in a repository-wide reveal registry. The recoverable history is explicitly a
lower bound of ten earlier reveals: six entries retain candidate hashes and
four older reveals are counted but cannot be attributed completely. It is not
presented as exhaustive. The candidate binds the externally pinned predecessor
registry snapshot; the appended registry entry then binds the candidate, after
which the new tip and count must be pinned separately. Selecting the best
branch after repeatedly opening the same final years is forbidden. Results are
therefore approach-specific retrospective evidence; only locked prospective
paper trading can be called globally pristine.

Registration is not itself a reveal and does not increment the historical
final-reveal count. The pure registry can create only a non-authorizing,
stage-bound request. A later effectful gate must independently load the latest
external pin, validate the prerequisite stage evidence, atomically consume the
request once, and record the actual final-period touch in a separate ledger.

## Numerical learner and ablation

The downstream model has two frozen heads:

- probability that a 20-session CASH episode beats holding AAPL after
  10-basis-point costs; and
- expected active log edge of that CASH episode.

At each filing event it receives the exact frozen AAPL price-regime fields,
including returns, volatility, drawdown, moving-average distance, gap, and
relative-strength history, plus SPY/QQQ/IWM/VIX/TNX market-sentiment fields
through that completed decision-session close. Gemma enums use one fixed
signed/stated encoding. The
calendar-only ablation receives the same AAPL and market history.

To control overfitting with roughly quarterly observations, the ten filing
dimensions are reduced to frozen commercial, financial, and risk/outlook
aggregates plus flag, coverage, and document-quality fields. Both prediction
heads are fixed ridge-regularized linear models with training-only robust
scaling. There is no feature selection, interaction search, or hyperparameter
tuning after this contract.

All semantic reductions are exact. Group means use their fixed declared
denominators, treating `not_stated` as zero. The adverse count covers the five
adverse flags and excludes `management_transition`, which has its own feature.
The stated fraction counts current-impact values other than `not_stated`; the
comparable fraction counts changes other than `not_stated` and
`not_comparable`. `sessions_since_prior_same_form` is the difference between
the two zero-based positions in the frozen 2000-onward NYSE session sequence;
the first filing of a form receives zero.

A feature row requires complete authenticated AAPL/SPY/QQQ/IWM/VIX/TNX
market support through the completed decision-session close, including the
exact 253-row slice needed for 252-session calculations. Labels compare the
next adjusted open with the adjusted open 20 held sessions later. The declared
5- or 10-basis-point rate is charged on each position-changing fill, at both
entry and exit.

Prediction and training receipts use event-local causal identities. The market
chain identity ends at the decision-session row, and the market-feature
identity covers only that event, complete-support/missingness state, and exact
feature values. The extraction identity covers only the current and immediate
prior filing identities plus the authenticated extraction status, evidence,
output, and supplied-sentence identities for that event. Full stage, source,
corpus, and proof hashes remain separately preserved for audit, but are not
part of these causal identities. Appending future-only market rows, filings,
or stage provenance therefore cannot change an earlier event identity,
feature value, probability, expected edge, or action.

The numerical recipe is part of the frozen candidate, not an implementation
choice. Each raw feature is centered on its training median, divided by
`max(1.4826 * MAD, 1e-6)`, and clipped to `[-4, 4]`. The edge target is first
clipped to `[-0.5, 0.5]`, then centered and scaled by the same training-only
median/MAD rule. The intercept is not regularized. The probability head uses
Newton updates with a frozen Armijo line search; the edge head uses frozen
Huber IRLS. Their initialization, iteration caps, tolerances, and all remaining
numerical constants are recorded in the contract manifest. Model-state floats
are sealed as canonical hexadecimal strings so a replay cannot silently change
precision or serialization.

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

Each fold's semantic model, no-semantics ablation, robust scaler, and smoothed
climatology are fitted exactly once from all and only labels whose maturity
session strictly precedes that fold's first test session. Those states must
remain byte-identical throughout the complete test window: outcomes from an
earlier prediction in a fold cannot update a later prediction in the same
fold. After candidate selection there is one separate refit using all and only
development labels matured by 2018-12-31.

The training audit retains every matured eligible filing event, including an
explicit unavailable row when evidence is incomplete. Learner support is the
strict subset whose feature row says `prediction_available = true`; it is
never inferred by silently dropping rows. A sealed invalid Gemma output may
still be fit-eligible through neutral semantics plus its missingness indicator,
whereas missing/unauthenticated extraction evidence or incomplete required
market history is not fit-eligible.

The probability head and every Brier comparison use one binary target: whether
a 20-session CASH episode beats holding AAPL after 10-basis-point costs by more
than `1e-12` active log edge. The same tolerance defines episode wins and live
lesson labels. Candidate actions use exact greater-than-or-equal comparisons
against the frozen probability and expected-edge thresholds.
Each prediction also binds the exact fold training-set size and positive-label
count. The Beta(1,1) causal climatology is derived from those complete frozen
counts, including the 2000-2004 baseline for fold 1, rather than reconstructed
from later prediction rows.

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

Development is the single portfolio genesis. At the 2019 and 2024 stage
boundaries, both ledgers inherit their exact cumulative positions: buy-and-hold
is already LONG, and the strategy is LONG or CASH according to any episode
already in progress. Neither side receives a synthetic boundary trade or an
extra entry cost. A position-changing fill that genuinely occurs on the first
session of a new stage is still charged normally. The inherited positions also
earn or avoid the move from the immediately preceding adjusted open to that
first stage-session adjusted open before the opening fill, so a cross-stage
CASH episode cannot silently lose one return interval.

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
Episode concentration uses every within-window ledger contribution, including
episodes carried in from a prior stage and episodes still open at the cutoff.
Those partial contributions do not enter full-horizon episode win, mean-edge,
Brier, or training statistics.

Every saved ledger must pass the existing independent no-leverage proof and
reconcile daily strategy-versus-buy-and-hold log increments. A declaration in
a manifest is not sufficient.

## Local model and zero-cost rule

Gemma may be called only through
`http://127.0.0.1:11434/api/chat` using the exact locally installed
`gemma4:12b` digest. Environment proxies, redirects, model pulls, streaming,
thinking mode, retries, and repair attempts are disabled. Generation uses
temperature 0, seed 0, context 6144, and output limit 512.

The per-call client receipt is byte-level evidence, not a self-issued claim
that production transport or model identity is trustworthy. The stage runner
must independently compare the actual runtime evidence, model digest, and
runtime fingerprint with the candidate immediately before and after the
complete extraction batch. Until that guard passes, even the internally
created hardened loopback session remains explicitly unattested.

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

The 1.5-GiB SEC number is an aggregate network-transport ceiling, not an
in-memory evidence allowance. The owned stage-document runner binds a stricter
64-MiB aggregate raw-response ceiling into each execution claim; the original
access-manifest allowance is retained separately as evidence and cannot widen
that effective cap. HTML parsing and NFKC can expand text by more than two
times, so the normalizer computes a conservative per-character NFKC UTF-8
upper bound and rejects the batch before materializing normalized output if it
would exceed the remaining 128-MiB aggregate allowance. The exact final UTF-8
length is checked again, and every durable file is independently limited to
128 MiB before the first file is written. The complete caller bundle,
including source evidence and recursively embedded parent evidence, is capped
at 192 MiB of decoded Base64 and an estimated 256 MiB of canonical JSON. A
size-only evidence preflight must pass after corpus acquisition and before the
first Gemma call or any 2019+ stage is opened. If the complete corpus does not
fit, this version fails; it may not drop documents or silently raise the cap.

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

1. production SEC audit and sealed official-source calendar evidence;
2. metadata-only universe, corpus selector, extractor, and model client;
3. sealed development result and selected candidate;
4. sealed intermediate result and pre-final refit state; and
5. one sealed final result containing 2024, 2025, and 2026 YTD.

A materially different source family, prompt, learner, threshold set, or
filing universe is a new approach on a new branch. Rejected results remain
preserved rather than overwritten.

## What has and has not happened

At the current implementation checkpoint:

- the SEC audit/parser implementation exists and its offline tests pass;
- this experiment contract and its mutation tests exist;
- the exact market-source snapshot parser now replays every available OHLCV
  value from detached bytes and rejects future rows, noncanonical containers,
  and one-ULP substitutions;
- a completed Gemma call now emits one exact attempt receipt even when the
  extractor payload is invalid, with no retry or repair and no invalid
  semantic output exposed downstream;
- non-authorizing stage-access plans bind the exact candidate, registry,
  verifier source, SEC URL set, presealed market sources, model, budgets, and
  prohibited stages without containing a result or pass field. Version 2 also
  pins the prerequisite stage's content manifest, stage artifact, and external
  seal receipt; its only prior-stage text scope is the exact read-only
  normalized 10-K/10-Q carry-in required for the first same-form comparison;
- the prediction artifact sealer now persists exact pre-label prefix bytes
  through an append-only external-pin compare-and-swap receipt rather than
  accepting caller-supplied checksum strings;
- prediction evidence v2 binds the always-present extraction identity, the
  independent market-feature-row identity, and a causal market-prefix-chain
  identity even when a prediction is unavailable; each fixed fold also binds
  its exact training row and positive-label counts;
- an artifact seal proves exact structural ancestry only; the authoritative
  stage verifier must additionally replay the full prediction-prefix semantics
  against the same bytes and all candidate, calendar, event, and market pins;
- the SEC/Gemma scorer now deterministically replays the cumulative LONG/CASH
  state, same-ledger buy-and-hold, both cost levels, terminal conventions,
  stage-boundary positions, and open/carry-in episode concentration;
- a separate zero-tolerance adapter independently replays every score-ledger
  return, fill cost, binary exposure, cash/share identity, and debt value and
  also runs the repository's pre-existing unleveraged proof;
- the reveal store directly invokes one fixed verifier and exposes no
  caller-supplied validator parameter. Reveal, registry-CAS, and downstream
  authorization inputs must be exact built-in JSON within fixed depth,
  element, text, integer, estimated-JSON, and Base64 budgets; reveal inputs
  share one no-copy budget and every bound is rechecked while detaching.
  Verifier exceptions, invalid results, state mutation, or path redirection
  normally restore the exact authenticated prior bytes before the error
  escapes; an interrupted restore is resumed from its separate recovery record
  on the next locked load;
- the reveal store now persists a separate monotonic current-tip/CAS anchor,
  uses a write-ahead pending transaction for state-plus-grant changes, and can
  return the exact persisted grant bundle on an identical crash retry without
  rerunning the verifier or consuming a request twice. Interrupted genesis
  creation is also recoverable. Current-tip schema v10 stores append-only
  request-keyed trusted-content pins, consumed-stage first-recorded-evidence
  receipts, bounded SEC, market, and model execution claims/reader receipts/
  terminal aborts, and final-stage prior-same-form carry-in reader receipts.
  An active SEC claim blocks every registry, consumption, output, or competing
  execution transition until the store independently re-reads the exact granted
  layout and semantically replays every raw document, deterministic normalized
  document, request receipt, byte-manifest row, transport budget, contact hash,
  and self-hash, or records a non-retriable indeterminate abort. The owned
  runner does not accept or return an already committed receipt until it calls
  the store finalizer and the same durable replay succeeds again. A recovered
  complete marker is finalized without another network call;
  a recovered claim without such a marker is never retried. The receipt sets
  `fresh_network_provenance_claimed=false`: this proves internal consistency and
  grant binding, not external attestation of a fresh SEC response.
  A failed verifier retains its non-authorizing content pin, and an exact retry
  reuses it without another revision. After a grant is issued and its SEC
  claim has an exact terminal reader receipt, the store's private stage-output
  finalizer accepts only the request hash. It replays the SEC batch again and
  then reads exactly
  `stage_outputs/<claim_sha256>/stage_evidence/stage_evidence.json` plus its
  `complete.json` marker. The evidence must be the exact compact canonical v3
  envelope with an explicit parent, self-hash, candidate and stage matching the
  grant; the marker must be pretty-canonical and bind the request, claim, SEC
  reader receipt, fixed component/path, byte count, document hash and semantic
  evidence hash. Output-receipt v2 derives the SEC claim and reader hashes from
  their validated mappings and records the physical marker-file hash. Exact
  finalization re-reads both files after the marker and source-closure checks,
  which detects mutations between the first and closure reads. This is still a
  snapshot attestation: the reads are sequential, so the same-user mutable path
  namespace cannot prove simultaneous immutability through the later receipt
  commit. Any later retry or final-stage parent lookup replays the files again
  and fails closed if they no longer match. Exact retries of unchanged bytes
  create no new revision; missing,
  extra, linked, noncanonical, mutated or coherently substituted files are
  rejected. The receipt deliberately sets
  `fresh_stage_evidence_provenance_claimed=false`: the store proves the durable
  bytes and SEC ancestry, but no owned model/market assembler yet proves who
  produced the stage-evidence file. A
  final-stage transition must find the exact persisted parent-output receipt
  before its own trusted-content pin is committed. The anchor is a second file
  in the same store directory: it detects state-only rollback, but it is not an
  external trust domain and cannot by itself defeat coordinated replacement of
  both files;
- the private owned carry-in finalizer accepts only a consumed final-stage
  request hash. It requires that request to be the current ledger tip and its
  intermediate request to be the immediately preceding entry with the same
  attempt, candidate, design, registry entry, registry and registry tip. It
  replays both terminal SEC batches and the exact durable intermediate stage-
  evidence document, rederives the carry-in scope from that document's complete
  corpus universe and content manifest, and permits exactly the latest
  intermediate 10-K and 10-Q required by the first final-stage filings. It maps
  those accessions through the parent grant's SEC plan, reads only their
  normalized UTF-8 bytes, and copies them create-new to
  `stage_outputs/<final_claim_sha256>/prior_same_form_carry_in/`. The canonical
  marker binds the child claim and reader, parent request/claim/reader/output
  receipt, parent stage-evidence document and marker, content manifest, exact
  records, copied byte index and total bytes. Current-tip receipt v1 binds the
  same ancestry and can only be appended in a dedicated state-preserving CAS
  transition. A retry replays both source and copied bytes; missing, extra,
  linked, case-colliding, reordered or changed files fail closed. A partial
  local file or incomplete marker may be repaired only before any valid marker
  or persisted receipt exists; after either commitment, mismatches are never
  repaired. That method remains final-only. A separate request-only
  development-root finalizer now replays the terminal development root and
  intermediate child SEC readers, proves that the root was claimed against the
  exact ledger prefix before the child was consumed, and copies exactly the
  latest development 10-K and 10-Q required by the intermediate request into
  `stage_outputs/<intermediate_claim_sha256>/development_root_prior_same_form_carry_in/`.
  Its canonical marker and dedicated current-tip receipt bind both claims and
  readers, the complete development content manifest, the trusted child pin,
  the two selected records, copied byte index and total bytes. The first carry
  receipt cannot be retrofitted after a child stage-output receipt; carry first,
  output later, and an unchanged carry replay remains valid and idempotent.
  Missing, extra, linked, case-colliding, reordered, changed, reverse-
  chronological, or post-receipt-deleted artifacts fail closed without network
  access. Both carry receipt families set
  `fresh_carry_in_provenance_claimed=false`, are bound into the owned
  preprocessor and model attempt, but not yet into a downstream later-stage
  feature, label, learner, ledger, or stage-evidence assembler, and do not
  enable promotion. The request-free development feature projection is
  root-scoped and requires no carry;
  Like the stage-evidence receipt, each attests a sequence of
  store-observed snapshots rather than making the same-user Windows namespace
  immutable; a later exact retry detects post-closure mutation;
- the request-free development-content root plan derives all and only the
  2000-2018 development filings from the complete candidate-bound corpus
  universe. It embeds that exact universe for recovery, fixes a 64 MiB raw-byte
  ceiling, and grants no reveal-request, outcome, market, model, future-stage,
  or consumption-ledger authority. Current-tip anchor version 10 includes
  separate append-only development claim, reader, and abort maps keyed by the root-scope
  hash plus the dedicated development-root carry receipt map keyed by the
  intermediate request. It permits only one globally active SEC effect across
  ordinary stage requests and this root. Claiming the root leaves the registry, state, and
  consumption ledger unchanged. The owned root runner derives URLs and budgets
  only from the persisted claim, performs one strict no-cache/no-retry SEC
  acquisition, and seals raw and normalized filings, request receipts, the byte
  manifest, complete corpus universe, and an actual-byte-derived development
  content manifest under `stage_outputs/<claim_sha256>/sec/`. The distinct
  complete marker and store reader receipt bind the candidate, universe, plan,
  source closure, contact hash, content manifest, and every durable byte. A
  separate root-scope execution lock prevents a concurrent invocation from
  treating a live owner as an abandoned claim. An unchanged completed retry
  only rehashes local bytes; an abandoned active claim with no valid marker is
  terminally aborted without another SEC request, while a transient receipt
  failure after a valid marker leaves that marker recoverable without
  refetching. Together with the separate carry finalizer, this proves a durable
  pre-reveal training-corpus root and its exact intermediate carry-in. The
  request-free development feature projection now consumes the terminal root
  only; the carry remains reserved for the future intermediate path, and label,
  prediction, learner, ledger, and stage-evidence ancestry remain absent;
- current-tip version 10 also defines two separate append-only model-effect
  lifecycles. Development model claims are keyed by the request-free root scope,
  bind its terminal SEC root reader, and require no carry. Intermediate/final
  model claims are keyed by consumed request, bind the terminal child SEC reader
  and exactly the correct current-tip carry receipt. Both bind the SEC
  acquisition order separately from a chronological event plan sorted by
  availability session then accession. Every row binds its event ordinal,
  accession, form, availability session, and mapped SEC-document ordinal.
  Development derives this plan from its authenticated root universe;
  intermediate/final additionally require exactly one terminal matching
  development-root SEC claim and reader and derive their stage rows from that
  candidate/registry-bound full universe. Both bind candidate
  model/runtime/source identities, the canonical identity-lexicon hash, and
  fixed limits, grant no
  caller path, filing text, market, outcome, future-stage, paid-API, or external-
  network authority, and share a single globally active model-effect exclusion.
  Dedicated reader or terminal abort transitions are the only allowed successors
  to an active model claim. The reveal store and bounded runner now expose those
  owned claim/finalizer paths, including marker-last sealing, semantic replay,
  and terminal orphan handling. Offline tests use synthetic runtime output; no
  real model call has occurred;
- the fixed verifier now produces a version-6 canonical non-authorizing audit
  that replays candidate/source pins, calendar and universe manifests, exact
  Ollama attempt receipts, market-stage snapshots, prediction-prefix ancestry,
  learner arithmetic, raw scores, gates, ranking, no-leverage proof, runtime
  structure, registry, request, and stage-access bindings. It bounds all
  untrusted envelopes before decoding or copying, replays the complete parent
  evidence and receipt recursively, and binds final-stage lineage to the exact
  authenticated intermediate consumption-ledger tip, semantic audit, persisted
  trusted-content pin, authorization bundle, grant, consumed first-recorded
  receipt, exact parent membership in the reveal-store-supplied receipt map,
  and current store tip. The verifier recomputes
  the parent evidence's canonical document hash and byte count and cross-binds
  them to that exact receipt. Final-stage lookup now replaces the caller's
  parent mapping with the store-replayed durable parent document before the
  verifier runs, and the compact parent-tip proof carries the exact SEC claim
  and reader maps and their hashes. It rejects altered-and-rehashed evidence,
  access, context, audit, pin, entry, bundle, grant, parent output receipt,
  parent map membership or declared map hash, tip, and child identities. It
  does not independently authenticate unrelated entries in that supplied map;
- source-identity receipt version 5 now checks the current regular files at the canonical
  paths of modules that were already loaded; the audit refuses to import an
  absent module and accepts no caller-supplied root, path, or runtime bytes. Eleven
  previously omitted local dependencies are now separately pinned, and an AST
  closure check rejects any future static local import that is not in the
  declared source tree. At claim time the store re-reads every resolved source-
  role file at its canonical repository path, rejects a loaded module whose
  path differs, compares the complete role map with the candidate pins, and
  binds its map hash and count into both the claim and reader receipt. It checks
  the disk-source map again immediately before SEC I/O and receipt finalization.
  The new bounded stage runner is a distinct resolved and candidate-pinned
  source owner. The extractor prompt and JSON schema now also have distinct
  frozen repository owners instead of sharing an unresolved contract alias, and
  the preprocessor exports the exact immutable production identity lexicon plus
  its canonical hash and applies the bounded executive-context grammar above.
  This still cannot prove that current disk bytes created
  every already-running Python code object or exclude monkeypatching, so a fresh
  owned startup/import attestation remains blocked. One conceptual owner remains
  unresolved: ledger. The fixed-provider development market acquirer now has a
  distinct candidate-pinned source owner and a local reveal-store-owned claim,
  reader, abort, and replay lifecycle. This does not externally attest the
  network effect or a fresh process. A request-free development feature
  assembler now privately replays this reader into exact causal prefixes, but
  no owned label or later-stage market assembler exists yet;
- the pure preprocessor now builds and independently replays a canonical owned
  preprocessing receipt. It binds the development-root or stage-request scope,
  candidate and model claim, exact SEC/carry reader ancestry, event identity,
  current and optional prior normalized-byte descriptors, prior provenance,
  frozen lexicon, bounded-context implementation source, preprocessed event,
  and model payload.
  Filing chronology remains independent of SEC acquisition-file ordinal. This
  receipt performs no I/O by itself. The owned model runner now loads the exact
  current/prior/carry bytes only through the reveal store, and the store
  independently rebuilds this receipt while finalizing the complete model
  component;
- the Ollama client now has a strict two-request local runtime probe for the
  fixed `/api/version` and `/api/show` endpoints. It seals and replays bounded
  raw responses, derives the exact active model blob digest from the generated
  Modelfile, and fingerprints canonical version/show semantics. Its hardened
  transport inherits no proxy, redirect, retry, or pull behavior. All tests use
  injected fake loopback responses; no real runtime probe or model call has run.
  The probe is now connected to reveal-store-owned development and stage model
  claims. A global model lock, pre-call durable intent, exactly one attempt per
  filing, marker-last completion, semantic store replay, and terminal orphan
  handling prevent retries or caller-selected text, model, path, or transport.
  Intermediate and final loaders bind the exact development root, parent-stage
  evidence, current SEC reader, and first-same-form carry ancestry. Pending-tip
  and state-replace crash tests also replay development model claims from the
  authenticated store snapshot;
- supplied learner matrices and targets must match the fold-bound feature,
  target, membership, count, and maturity identities before any deterministic
  refit. Intermediate stage identity remains explicitly blocked: recursive
  parent evidence is now authenticated against the prior consumed reveal-store
  ledger entry and grant, but the development winner/output model state is not
  yet bound to the intermediate learner input state;
- stage access remains intentionally disabled twice: the authorizing verifier
  entry point raises while any end-to-end check is unsupported, and the reveal
  store has a separate false-by-default promotion gate, so substituting only a
  successful verifier cannot consume a request. The reveal store derives and
  persists the trusted stage-content pin itself, authenticates its current-tip
  membership, and requires the audit receipt to return the exact pin and store
  context hashes. It also persists the exact first-output receipt and
  recursively replays the fixed durable evidence associated with a consumed
  grant; the evidence document itself remains in the fixed component
  directory rather than inside the current-tip anchor. The bounded
  owned runner now claims the exact current grant before the SEC document batch,
  derives the document plan and budgets only from the persisted access manifest,
  binds the validated private-contact hash into that pre-effect claim,
  seals the actual raw/normalized bytes and canonical receipts into a fixed
  create-new directory, and has the reveal store independently replay the exact
  files and semantics before committing the SEC reader receipt. Together with
  the request-free development-root function, these are the only exported
  production SEC-document network entry points; the older universe-derived fetch helper is
  private and test-only. A noncanonical or invalid private SEC contact is rejected
  before the claim transition, so its hash and transmitted header cannot diverge
  and a typo cannot consume a grant. Tests use synthetic transports only;
  no SEC request was made. `stage_access_identity` remains `BLOCKED` because
  owned label assembly, prediction sealing, learner output, ledger, and final
  stage-evidence generation are not yet forced through owned components. The
  SEC, local market acquisition/replay, model, carry-in, and request-free
  development feature paths are now owned and cross-bound. The store no longer
  accepts a caller mapping at the output-receipt boundary, but a same-user
  process can
  still place coherently formed bytes in the fixed directory, so this milestone is
  durable replay rather than fresh end-to-end provenance. The owned SEC batch
  runner's final component-directory creation rejects even a pre-existing empty
  directory; the stage-evidence finalizer instead requires its fixed directory
  and exact two files to exist. The same-user mutable Windows path namespace is
  still not an external trust domain. The verifier also does not independently
  load the store files, attest its executing Python code object, or turn the
  same mutable directory into an external trust domain;
- stage-specific runtime receipts are structurally reconciled, but the final
  all-stage summary remains diagnostic until the canonical-market assembler,
  prediction/learner/ledger evidence, externally anchored process and
  monotonic-time attestations, and five-development-filing latency preflight
  exist;
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
- no 2024, 2025, or 2026 holdout run or performance calculation has occurred.

Market reconciliation now has a fixed development-only Yahoo Chart-v8
raw-response-to-canonical-snapshot implementation in addition to the existing
snapshot-to-stage replay. It uses six fixed unauthenticated zero-cost requests,
no fallback, retry, redirect, proxy, cookie, cache, credential, compression, or
paid call, enforces per-request and batch deadlines during bounded body reads,
rejects duplicate, conflicting, or unsupported HTTP body framing, and preserves
exact raw bytes plus deterministic normalization evidence. The request-free
development market claim/reader/abort lifecycle is
now implemented. It persists an exact 23-file, create-new, marker-last component;
the reveal store independently reparses every provider response, snapshot,
manifest, and reconciliation receipt before appending the reader receipt.
A first reader receipt also requires a non-persisted, process-local, one-use
sequencing witness created by the exact owned Yahoo transport path and bound to
the claim plus all seven acquisition identities. This blocks accidental public-
runner injection and recovery promotion, but normal Python introspection means
it is not a security boundary against hostile same-process code and cannot
authorize production. A synthetic bundle or prewritten completion marker cannot
be promoted by the supported recovery path; an indeterminate first execution is
terminally aborted. An already committed receipt remains replayable without
another network call. Public runner output contains only the claim and reader
receipt, never the provider body.

Offline tests patch the exact owned transport's fetch method with reviewed
synthetic responses; no real price request has run. Provider bodies stay behind
the verifier/store API contract, but the same-user directory is not OS/ACL
isolation or an external trust boundary. The Gemma event-input contract accepts
only normalized SEC text and fixed ancestry hashes, has
`market_access_permitted = false`, and rejects extra input fields, so provider
metadata is not serialized to Gemma. The request-free development feature worker
now receives only exact 253-session canonical prefixes from a store-owned
accessor, never the raw-response directory. A future prediction worker must keep
that same boundary. The acquirer deliberately stops at 2018 and does not
claim that a later Yahoo download can extend the prefix byte-for-byte, because
adjusted-close history can be revised. Before live extension, the development
dataset must be globally versioned and sealed so different candidates cannot
silently reacquire different adjusted histories.

The current `urllib` timeout is an inactivity timeout for DNS, connection, TLS,
and response-header phases; only response-body reads enforce the shrinking
absolute deadline directly. Therefore `trusted_production_transport=true` means
that the owned transport path was selected inside this process, not that an
external network occurrence or hard wall-clock deadline was independently
attested. Any authorizing live version requires a cancellable fresh worker or
process/IPC boundary plus external receipt and tip anchoring.

Development, intermediate, and final Gemma execution claims now require the
same-root successful, non-aborted, store-attested market reader and bind its
claim, receipt, acquisition, manifest, reconciliation, and byte-index hashes.
This freezes the experiment ancestry without granting Gemma market access.
Promotion remains false: the detached stage verifier can still accept
caller-supplied market snapshots. The new store-owned development feature
assembler consumes only the terminal SEC, Gemma, and market receipts and emits
self-hashed development feature rows with explicit flags denying labels,
outcomes, training membership, learner fit, promotion, and production use. A
separate store-owned development label assembler now derives only outcomes
whose `t+21` session is on or before 2018-12-31 and emits compact, self-hashed
label evidence while still denying training membership, learner fit,
prediction, holdout, ledger, promotion, and production use. No owned
training-membership/prediction/learner/ledger assembler exists yet. The ledger
source role is the sole unresolved source role.

The completed feature-only checkpoint is deliberately non-authorizing. Under one
store lock it replays terminal non-aborted same-root SEC, market, and development
Gemma readers, requires zero consumed reveals and no active effect, reconstructs
each valid or sealed-invalid extraction, isolates exactly sessions `t-252..t`,
and returns only compact proofs to the runner. The public runner emits derived
feature rows and hashes; it cannot request paths, text, market rows, model calls,
labels, outcomes, holdouts, training membership, predictions, ledger mutation,
or stage promotion. The complete corpus-universe metadata is revalidated
internally only to prove the fixed split identity; the operation emits none of
the 2019-2026 filing text, market observations, outcomes, labels, learner inputs,
or later-stage metadata.

The completed label checkpoint keeps one audit row for every development event,
labels every event whose result is mature by the cutoff regardless of feature
availability, and gives chronologically immature events a null label hash. It
never opens a target path for those immature events. Any missing, nonpositive,
or corrupt AAPL adjusted-open value in a mature `t+1..t+21` path fails the whole
projection with event and session context. The public runner can receive only
the owned store plus the development-root scope and exposes compact adjusted-open
paths rather than full market rows. Frozen-source integration tests replayed the
feature projection in 459.77 seconds and the new label projection in 609.29
seconds, both without network or live-model calls.

The next implementation milestone is the separate request-free development
training-membership assembler, followed by owned learner state, predictions,
and the continuous no-leverage ledger. Those later components must
remove caller-supplied market snapshots from every authorizing path, parse
official calendar semantics, chain every artifact from genesis, and bind the
development winner/output state to the intermediate learner input. A production
trust domain must retain the current
store tip outside the mutable store directory, and a fresh owned process must
attest the executing verifier rather than only the current source files. That
machinery must be committed and pass the five-filing preflight before any
filing meaning or later-stage outcome is opened.
