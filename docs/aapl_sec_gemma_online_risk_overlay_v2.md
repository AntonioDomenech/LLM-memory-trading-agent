# AAPL SEC/Gemma online risk overlay v2

## Status

This approach is **preregistered but unrun**. No real filing has been acquired
for it, Gemma has not read a filing, and no SEC/Gemma return has been scored.
The machine-readable contract is
`agent_benchmark/sec_gemma_online_risk_overlay_contract.py`.

It lives on `codex/aapl-sec-gemma-online-risk-overlay-v2`. The older
`codex/aapl-sec-filing-gemma-v1` remains preserved and is not overwritten.

## Plain-language idea

The best historical lead stays invested in Apple most of the time and uses a
small number of one-session cash exits. It has encouraging long-run behavior,
but its learner stopped changing trades and it failed 2024.

This approach keeps the existing fixed exhaustion union as a baseline and adds
one genuinely different source of information: Apple's official 10-K and 10-Q
filings. Gemma is only a document reader. It translates changes in demand,
margins, costs, liquidity, guidance, supply risk, legal risk, and management
uncertainty into four compact business-risk measurements. It never sees market
returns or chooses a trade.

A small numerical model combines those four filing measurements, one extraction
quality measurement, one form indicator, and six already-known market
measurements. When both of its fixed risk tests pass, it adds a 20-session cash
period. Otherwise the strategy follows the inherited baseline. The SEC overlay
must add value over the baseline; inherited performance cannot rescue a useless
filing model.

The inherited weak-trend expert was itself selected after extensive historical
search and after final-year information had been inspected. Its exact source
bytes are frozen here so it cannot be improved after this preregistration, but
its contribution is not new evidence. Only SEC-overlay-minus-baseline results
measure the new idea.

## Why this is a new approach

The rejected price/regime families repeatedly changed scores without changing
actions, or inherited almost all of their apparent success from a fixed rule.
The downloaded GDELT warehouse is not a usable news corpus: its titles are
synthetic event labels, it has no article bodies, and its Apple matching has
severe false positives. This branch therefore does not call those rows news or
use them as sentiment.

Compared with the unfinished SEC/Gemma v1, this version deliberately reduces
selection and small-sample risk:

- 12 model inputs instead of roughly 54;
- one action threshold instead of selecting among four;
- continuous causal learning instead of freezing the model for years at a
  time;
- an explicit frozen-state control proving whether new lessons change actions;
- an explicit no-filing-meaning ablation proving whether Gemma-derived values
  add anything; and
- incremental gates versus the inherited baseline.

## Inputs and timing

The filing universe is every metadata-eligible, non-amended Apple 10-K and
10-Q for CIK `0000320193`. Selection cannot depend on document language,
runtime, price, or outcome. Availability is the first complete NYSE session
strictly after the latest defensible SEC acceptance, filing, or filing-change
date. The decision is made after that session closes and fills at the next
adjusted open.

Gemma receives only blinded sentences from the current and previous same-form
filing. Issuer identity, ticker, exact date, market values, returns, labels,
actions, and benchmark results are forbidden. Its exact existing prompt and
schema are hash-bound. Invalid output is not repaired or retried; it becomes
neutral semantics with a quality-risk indicator.

The exact installed model is already pinned: the local `gemma4:12b` manifest
hash is `4eb23ef...b2b05c`, its config and four layer digests are fixed, the
Ollama version is `0.32.0`, and the raw version/show responses and derived
runtime fingerprint are in the machine contract. Gemma 4 exposes two active
model blobs, so v2 must use a new exact-manifest verifier rather than the old
v1 verifier that assumed a single blob. The existing preprocessor, identity
lexicon, request builder, prompt, schema, response validator, corpus parser,
calendar, market parser, learner, and baseline source bytes are also pinned.

Market evidence uses one frozen zero-cost source family: unauthenticated Yahoo
Finance Chart v8 at `query1.finance.yahoo.com`. Each stage makes exactly one
owned HTTPS request per symbol, in the fixed order AAPL, SPY, QQQ, IWM, VIX,
and TNX, with
the exact daily query, symbol mapping, User-Agent, provider timezone, and Unix
boundaries in the machine contract. Development requests 1998 through the end
of 2018; confirmation requests the same continuous prefix through 2023; final
transport requests through 2026-07-10, but keeps that final row quarantined.
Market values, performance, and terminal valuation end on 2026-07-09. Only a
decision through July 8 can create a scored next-open fill; a July 9 decision is
sealed only as a pending live action. There is no proxy, redirect, cookie,
alternate provider, or fallback.

Every later canonical snapshot must reproduce every earlier session,
explicit-absence marker, and floating-point bit pattern. A Yahoo
back-adjustment or other prefix revision terminally fails the branch rather
than quietly changing its history. Exact raw responses stay in a private
quarantine because provider metadata can contain current quote fields. The
development raw snapshot may be acquired before its scoring lock, but no price
value is exposed then; confirmation and final acquisition occur only after
their one-shot stage locks.

The twelve ordered numerical inputs are:

1. AAPL minus QQQ 20-session log return;
2. AAPL 63-session drawdown;
3. AAPL 20-session realized volatility;
4. SPY 20-session log return;
5. IWM 20-session log return;
6. VIX 20-session log change;
7. commercial deterioration;
8. financial deterioration;
9. risk/outlook deterioration;
10. adverse-flag fraction;
11. semantic quality risk; and
12. 10-K form indicator.

Every market value is adjusted close. A 20-session return is
`log(close[t]/close[t-20])`; AAPL relative return subtracts QQQ's result. The
63-session drawdown is `close[t]/max(close[t-62:t])-1`. Realized volatility is
the sample standard deviation (`ddof=1`) of the 20 log returns ending at `t`,
annualized by `sqrt(252)`. VIX uses the same 20-session log-change formula.
Any missing, duplicate, nonfinite, or nonpositive required value makes that
filing unavailable; nothing is imputed.

Ledger pricing is stricter: every exposed AAPL session must have exactly one
finite, positive raw open, raw close, and adjusted close, and their
`raw_open * adjusted_close / raw_close` adjusted open must also be finite and
positive. A violation fails the entire stage because neither the strategy nor
buy-and-hold can be valued fairly.

SPY and IWM trend plus VIX movement are the approach's quantitative
market-sentiment proxies. They measure how optimistic, broad, and fearful the
market was using only information already known at the decision close. This is
not article/news sentiment; company-specific outlook comes from Apple's
official filing text.

The primary no-filing-meaning ablation uses identical events, market history,
labels, learner, 10-K form control, and observed extraction-quality risk. Only
the four actual filing-meaning values are zero. Consequently, invalid Gemma
output cannot create a fake "meaning" advantage merely because its quality
flag differs. A secondary all-five-zero no-Gemma-channel arm is reported only
as a diagnostic and cannot satisfy a filing-meaning gate.

Every eligible filing remains in a chronological audit row. A first filing with
no prior same-form document is still available: its changes are
`not_comparable`, while current filing evidence remains usable. An authenticated
but schema-invalid Gemma response is not retried; its four meaning features are
zero, quality risk is one, and it remains trainable if market inputs are
complete. The full and quality-preserving no-meaning arms are identical for
that row. A missing or unauthenticated model response, provenance record, or
required market value instead makes the event unavailable: it produces no SEC
action or fitted prediction and enters neither training, Brier support,
episode counts, nor action-difference counts. A later label may be preserved as
audit-only but cannot train without its immutable decision-time features.

Before learner readiness, an otherwise available event can become a future
lesson but emits no fitted prediction and no overlay. A filing arriving during
an active overlay is still evaluated and learned from, but non-overlap forces
its effective scheduling action to false.

## Learner and action

There are two deterministic, ridge-regularized heads:

- the probability that activating the overlay beats the exact fixed
  baseline-only ledger over the next 20 held sessions after 10-basis-point
  costs; and
- the expected incremental 10-basis-point log edge versus that baseline.

The label is computed from two counterfactual ledgers after the horizon
matures: fixed baseline only, and the same baseline with a 20-session overlay
activated at this filing. This includes any baseline cash signals and exact
transaction-cost interactions inside the horizon. It is defined whether or not
the live policy activated the overlay, so the learner cannot hide rejected
signals.

Both counterfactuals fork the exact baseline-only account immediately before
open `t+1`. One continues the fixed baseline; the other forces cash until open
`t+21`. Other SEC overlays are absent from both arms, while future baseline
signals continue identically. Thus the label measures the new overlay, not
whether cash happened to beat an oversimplified always-long comparator.

Before each filing decision, both heads are refitted on all and only earlier
filing outcomes whose complete `t+21` exit-open result is already known. A label
maturing at the current session's open may train the decision after that
session's close; formally, `maturity_session <= decision_session`. A signal's
own result can never affect that signal. Counterfactual lessons mature even if
the strategy did not take the signal, preventing its action history from hiding
mistakes.

The SEC overlay is CASH only when both conditions hold:

`probability >= 0.55 AND expected_incremental_10bps_log_edge >= 0.0025`.

There is no threshold search. A passing signal sells at adjusted open `t+1`
and returns to AAPL at adjusted open `t+21`, after exactly 20 open-to-open
return intervals. A signal
during an active overlay cannot extend or overlap it, but its eventual result
still becomes a lesson. The combined strategy is in cash whenever either the
existing one-session exhaustion union or the SEC overlay is in cash.

The baseline is specifically the prefix-invariant
`unfiltered_union_signal` from the fixed contextual and weak-trend experts, not
the binary-regime selector and not the existing end-of-dataset actionable mask.
A baseline signal at close `t` schedules cash from open `t+1` to open `t+2`
even when those fills are still pending beyond the current data prefix. Later
rows may complete that action but may not rewrite it. Baseline cooldown evolves
independently of the SEC overlay. When the two policies overlap, a cost is paid
only when their combined 0/1 exposure actually changes.

Until at least 20 trainable filing lessons and at least four examples of each
binary class exist, the SEC overlay is inactive and the inherited baseline
continues normally.

## Chronological experiment

The portfolio starts on 2000-01-03. The system is long by default and begins
accumulating causal filing lessons as they mature. The declared periods are:

| Period | Dates | Use |
| --- | --- | --- |
| Development-corpus warm-up | 2000-01-03 to 2004-12-31 | Part of the 2000-2018 development corpus and its 72-filing minimum; trade the baseline and build only matured lessons |
| Development qualification | 2005-01-03 to 2018-12-31 | Continue the same online replay; qualify the one frozen design over five reporting blocks |
| Confirmation | 2019-01-01 to 2023-12-31 | One untouched approach-specific confirmation; no design change |
| Final live-style replay | 2024-01-01 to 2026-07-09 | Continue learning only after outcomes mature; report performance and terminal value through July 9 jointly, with any July 9 close decision preserved as pending |

Immediately before every 2025 filing decision, the model admits every earlier
eligible result whose exit open has occurred by that decision session, even if
it matured after January 1. There is no artificial annual freeze. The same
live-style rule continues into 2026.

If multiple filings share a decision session, all labels maturing on that
session are admitted once, then filings are processed by exact SEC acceptance
timestamp and accession. The first passing filing may reserve the next-open
overlay. That pending overlay blocks a later same-close filing from scheduling
or extending another one, although every filing still receives an audit
prediction and later lesson.

The full semantic arm, the quality-preserving no-meaning arm, and the
all-five-zero diagnostic each start at portfolio genesis with their own causal
learner, account, baseline cooldown, pending actions, and non-overlap state.
They carry that state continuously through every block and stage and receive
the same counterfactual labels; only their declared feature transform differs.

Three controls accompany the online strategy. Each forks exact cash, shares,
pending fills, baseline cooldown, active overlay, fitted coefficients, scaler,
and training membership. A frozen control may see later causal inputs but never
admit another label after its fork:

- a model frozen at each development block's first session;
- a through-2018 model frozen throughout confirmation; and
- a through-2023 model frozen throughout the final replay.

Each frozen control forks at the start of its boundary session, after labels
maturing by the preceding session have been admitted but before any label
maturing on the boundary session or any boundary-session filing is processed.
Reporting windows never create or reset an account; they only normalize the
continuously carried wealth for comparison.

New lessons must change actions and add net edge versus these controls. Merely
changing probabilities is not learning success.

## Exact comparison arithmetic

An action difference is one eligible filing where the effective
`schedule_overlay` booleans differ after readiness, thresholds, and each arm's
non-overlap state. An XOR interval is one maximal contiguous open-to-open ledger
interval where two compared target exposures differ. Its contribution is the
sum of their net log-return difference, including actual fill costs. Only XOR
intervals and overlay episodes with both boundaries inside available data enter
counts, win rates, medians, and concentration; carried/open contributions still
enter aggregate ledger edge.

Each open-to-open return and fill cost belongs to its destination-open session.
For an adjusted-open report, carried wealth is normalized immediately before
the first assigned interval and run through the final session open. The
terminal-adjusted-close version appends exactly one factor on that report's
final session: `adjusted_close/adjusted_open` if the arm is long after the open
fill, or one if it is cash. This diagnostic close value is never carried into
the next window and never incurs a fictional liquidation cost.

Maximum drawdown starts from normalized wealth one at the report boundary and
observes wealth after every destination-open return and fill; the close variant
adds only its one final-close observation. At each observation, the running
peak includes the starting one, and
`MDD = min(wealth/running_peak - 1)`, so it is nonpositive. At both costs and
under both terminal variants, the strategy must satisfy
`strategy_MDD >= AAPL_MDD - 0.01` over the continuous final window.

Positive means log edge strictly above `1e-12`; absolute edge at or below that
is a tie. A negative-AAPL year has same-ledger AAPL log return below
`-1e-12`. Removing the best block means total edge minus the greatest block
edge. Removing the best episode/XOR means total incremental edge minus its
largest strictly positive complete contribution. Brier scores use identical
mature, available filing decisions; the target is incremental overlay edge
above `1e-12`, and relative improvement is
`(ablation - semantic)/max(ablation, 1e-12)`.
An episode win rate counts strictly positive complete contributions divided by
all complete contributions; ties and losses remain in the denominator. A
difference block or year is counted once when it contains at least one effective
action difference assigned by the filing decision session. Empty or nonfinite
metrics fail every gate that depends on them.

## Gates before later periods

Development fails unless the complete strategy has at least 0.02 total
10-basis-point active log edge, remains at least 0.005 ahead after removing its
best block, wins at least four of five blocks and 55% of years, behaves
positively in at least 60% of negative-AAPL years, and has at least 12 complete
SEC-overlay episodes with a 55% win rate, positive median edge, and no episode
above 35% of positive edge. Every statistic in this sentence is evaluated at
10 basis points per changing leg.

The SEC component must independently:

- produce authenticated schema-valid output for at least 90% of eligible calls
  and at least 24 rows with a nonzero actual filing-meaning feature;
- add at least 0.005 10-basis-point log edge versus the inherited baseline and
  stay positive after removing its best block;
- differ from the block-frozen learner on at least five decisions across three
  blocks and add positive stress-cost edge;
- differ from the no-filing-meaning ablation on at least five decisions across
  three blocks and at least four complete XOR intervals;
- add at least 0.005 stress-cost edge over that ablation, remain positive after
  its best XOR episode is removed, and improve Brier error by at least 1%.

If development passes, confirmation is opened once. It requires at least 90%
schema-valid extraction, at least six nonzero meaning rows, positive strategy
edge and at least three winning years separately at both costs, material
10-basis-point incremental edge versus the baseline and ablation, and
action-changing, profitable online learning versus the frozen-through-2018
control. Failure permanently rejects this branch before the final replay.

Final success requires at least `0.005` active log edge (approximately 0.50%
relative wealth) at 5 basis points in each of 2024, 2025, and 2026 YTD; at
least `0.02` continuous active log edge (approximately 2.02% relative wealth);
positive stress-cost edge in every period; positive incremental edge versus the
inherited baseline in aggregate and in at least two of the three final periods,
plus positive incremental edge versus the no-filing-meaning ablation and
frozen-through-2023 control;
at least 90% schema-valid extraction and three nonzero meaning rows; at least
two semantic-versus-no-meaning and three post-2023 online-versus-frozen action
differences; at least six complete overlay episodes; and the declared
10-basis-point episode win-rate and concentration limits plus the drawdown
limit at both costs.
Every return, edge, and drawdown gate must pass under both declared terminal
valuations (adjusted open and terminal adjusted close), and any undefined or
nonfinite metric fails the stage.

Zero action differences is an explicit rejection at every learning/semantic
gate. A failed stage cannot be rescued by opening a later stage.

## Trading, cost, and artifact invariants

Target and realized exposure must always be exactly 0 or 1. Shorting, leverage,
borrowing, negative cash, fractional exposure, margin, and cash interest are
forbidden. Strategy and buy-and-hold start with the same cash, use the same
adjusted-open ledger, and pay 5 or 10 basis points on every changing leg. The
action stream is identical at both cost levels.

Both begin with USD 1,000 cash immediately before the 2000-01-03 open, target
LONG, and pay the same initial buy cost. Adjusted open is
`raw_open * adjusted_close / raw_close`. A buy multiplies wealth by
`1/(1+cost)` and a sale by `1-cost`; while long, the prior exposure first earns
the current-open/prior-open return, and the new target then fills. Fractional
shares are allowed, but fractional exposure and debt are not. Stage boundaries
carry exact cash, shares, prior open, pending fills, and active overlays without
a reset or synthetic trade. Final valuation is checked at both adjusted open
and terminal adjusted close.

Predictions and actions are sealed before their outcomes are opened. Lessons
are append-only. Every stage saves complete ledgers, predictions, controls,
episode and XOR attribution, costs, drawdown, yearly/block results, runtime,
source identities, no-leverage proof, and checksums. Failed attempts remain
immutable.

Development acquisition, development scoring, confirmation, and final each
have one fixed attempt ID. The acquisition lock is consumed before its first
official SEC or market request and may emit only quarantined bytes, receipts,
hashes, counts, and blinded deterministic requests; never a price, semantic
output, label, action, or score. Its failure is terminal.

After that prerequisite succeeds, the development scoring lock is consumed
before either the first real Gemma call (including the five latency calls) or the
first canonical market value is read. Confirmation and final locks are consumed
before their first stage acquisition, feature, or outcome access. Every
indeterminate execution is terminal rather than retried. Final access also
appends and externally pins the repository-wide reveal registry whose current
lower bound is ten earlier final-period reveals. Semantic responses and partial
metrics remain hidden until the joint stage report is sealed. The implementation
and every new v2 source hash must be committed and bound to this
preregistration before any effectful attempt.

## Runtime and zero-cost limit

Every effectful acquisition attempt and every scored stage attempt must
individually finish in less than 3,600 seconds. Development acquisition and
development scoring are separate fixed attempts; confirmation and final each
include their own newly unlocked acquisition. No test may be split merely to
evade the ceiling, and the cumulative multi-stage lifecycle time is reported
separately rather than described as a sub-hour run.

- combined official SEC and fixed Yahoo acquisition: at most 720 seconds, with
  Yahoo itself at most 210 seconds and exactly six requests;
- local Gemma extraction: at most 2,160 seconds;
- fitting, ledgers, verification, and sealing: at most 480 seconds; and
- 239 seconds reserved as contingency.

The largest phase caps plus contingency therefore total 3,599 seconds, leaving
one full second below the strict 3,600-second boundary.

SEC is additionally capped at 1,000 requests, 1.5 GiB, and two requests per
second. Gemma calls are capped at 80 in development, 20 in confirmation, and 12
in final. After deterministic preprocessing, the five development events with
the largest canonical UTF-8 request bodies are selected, breaking ties by
accession. They execute first, largest then accession; those calls become
sealed members of the real batch after the development lock. Projected model
time is their actual sum plus
the remaining call count times the slowest of the five and must not exceed
2,160 seconds. Remaining calls execute in chronological availability order.
An external parent process enforces shrinking phase and per-attempt total
deadlines. There are no paid APIs, model pulls, fallbacks, or retries.

## Known limitation and current blocker

The repository has already inspected the final years in other approaches, and
the installed Gemma model reportedly has a January 2025 knowledge cutoff.
Therefore this experiment can provide candidate-specific chronological
evidence, not a globally pristine 2024 result. Only locked prospective paper
trading, or a model demonstrably trained before the evaluated filings, can supply
that stronger evidence.

No real SEC text is currently downloaded. Official SEC access requires a
private User-Agent containing the user's chosen identity and a reachable email.
The ignored local configuration currently has no such value. It must never be
committed or printed. Until it is supplied, production acquisition and the
five-filing Gemma preflight remain blocked, while offline implementation and
tests may continue.

The old v1 stage runners are intentionally not accepted as v2 runners: they
bind a different 54-feature, frozen-fold contract and assume one active model
blob. V2 therefore still needs committed feature, runtime, online-refit,
policy, ledger, store, and verifier adapters. They may reuse the pinned corpus,
preprocessor, extractor, local client, and numerical learner primitives, but
their new source hashes must be bound before any real SEC or model effect.
