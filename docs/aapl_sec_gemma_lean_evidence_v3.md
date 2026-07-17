# AAPL SEC/Gemma lean evidence v3

## Status

This approach is preregistered on
`codex/aapl-sec-gemma-lean-evidence-v3` before any effectful attempt.

No official SEC or Yahoo request, Gemma generation, filing prediction,
trade, return, or stage result has been opened for v3. The rejected v2, v2.1,
and v2.2 branches remain immutable historical records. In particular, v2.2
remains rejected under its own obsolete complete-lifecycle one-hour contract;
v3 does not rewrite that result.

The exact scientific question is still untested:

> Can a fixed local Gemma reader extract point-in-time deterioration evidence
> from Apple 10-K and 10-Q filings that helps a chronological long-or-cash
> learner avoid enough harmful AAPL exposure to add after-cost value over the
> frozen exhaustion baseline and a no-filing-meaning control?

V3 changes the operational contract only. It removes the former one-hour
pass/fail rule and replaces the elaborate mandatory remote-publication
lifecycle with small, resumable, checksummed stage checkpoints. It does not
change the trading hypothesis after seeing a result.

## Frozen scientific parent

The scientific parent is the v2.2 implementation commit
`efbc481c57e480d48303763163676e64e87df49d`. The v2.2 preregistration commit is
`a849b9d704ffd98547e570735a221b2b75f7db86`, and its literal contract-manifest
SHA-256 is
`64c0139fec91fd82a1c5c38050f0076989054c2b2c866eda0abdf436573f4e1d`.

V3 inherits unchanged from that parent:

- the local `gemma4:12b` model identity, semantic runtime fingerprint, prompt,
  schema, temperature, seed, context length, and output limit;
- the eligible Apple 10-K/10-Q universe and point-in-time availability rules;
- the twelve filing-derived values, market-context values, missingness flags,
  learner, fitting rules, candidate thresholds, and deterministic ordering;
- the rule that the overlay enters cash only when
  `probability >= 0.55` and
  `expected_incremental_10bps_log_edge >= 0.0025`;
- the 20-session cash interval, non-overlap rule, next-open fills, and
  outcome-maturity rule;
- the exact inherited one-session exhaustion union baseline;
- the semantic model, no-filing-meaning ablation, no-Gemma-channel diagnostic,
  frozen-learning controls, and always-long AAPL comparison;
- the development, confirmation, and final scientific gates described in
  `docs/aapl_sec_gemma_online_risk_overlay_v2.md`; and
- the no-short, no-leverage, no-borrowing, nonnegative-cash, zero-cash-interest,
  and same-ledger accounting rules.

There is no threshold search, alternate horizon, model swap, prompt rewrite,
feature addition, row deletion, class rebalancing, or failed-result retry in
this branch. Any such scientific change requires a new branch and a new
preregistration.

## Chronology and evidence labels

The stages remain:

1. development: 2000-01-01 through 2018-12-31, with 2000-2004 used as warm-up
   and 2005-2018 used for qualification;
2. confirmation: 2019-01-01 through 2023-12-31, opened only after a complete
   development pass; and
3. final live-style replay: 2024-01-01 through the frozen 2026-07-09 cutoff,
   opened only after a complete confirmation pass.

At a decision close, the learner may use only outcomes whose full 20-session
result has matured by then. The final stage must report both the continually
learning arm and the arm frozen after 2023, including lesson availability,
prediction differences, action differences, and the signed contribution of
changed actions.

The repository has already examined 2019 onward in other approaches, and the
installed model may contain information through January 2025. Confirmation and
final results are therefore candidate-specific chronological evidence and the
2024-2026 result is a repeated historical audit, not globally pristine proof.
Only decisions saved before future market outcomes can become prospective
evidence.

## Data and privacy contract

Only official Apple SEC filing evidence with defensible publication or
acceptance timestamps is admissible. The existing synthetic GDELT collection
is forbidden. Market inputs must be the exact point-in-time AAPL and declared
context series used by the frozen parent.

Official SEC requests require a real name or organization and reachable email
explicitly authorized by the user. The readable User-Agent exists only in the
ignored `data/local_config.json`. It must never be committed, printed, copied
into a prompt, or written into an artifact. Public evidence may contain only
its SHA-256 fingerprint.

Paid APIs, paid services, model pulls during an attempt, broker connections,
shorting, leverage, and real-money execution are forbidden.

## Cheap checks before substantial work

The following sequence is mandatory:

1. A zero-effect preflight verifies the clean pushed branch, frozen scientific
   parent, dependency and source identities, private-contact presence without
   revealing it, installed local model identity, writable checkpoint area, and
   absence of a conflicting active run.
2. Acquisition checks the complete development filing universe and market
   coverage before any performance result is calculated. Too few filings,
   missing timestamps, truncated market history, unexplained session gaps, or
   unusable text rejects or blocks the attempt before a long model run.
3. After deterministic preprocessing, the five development requests with the
   largest canonical UTF-8 byte length run first, in the exact v2.2 order.
   These are real members of the development batch, not disposable probes.
4. All five pilot calls must complete, authenticate the pinned model, and
   produce schema-valid sealed outputs. Their elapsed time and peak resource
   use are recorded. The conservative full-development estimate is
   `sum(first five elapsed) + remaining calls * slowest pilot call`.
5. A projected run above twelve hours is not a scientific rejection, but it
   pauses before the remaining calls until a new written justification is
   committed. A broken model identity, invalid pilot output, or obviously
   unsuitable corpus rejects the attempt without spending the longer runtime.

No confirmation or final value may be opened by these checks.

## Runtime, checkpoints, and clean resume

There is no one-hour success gate. Initial planning estimates are:

- zero-effect preflight and focused tests: 5-30 minutes;
- official-source development acquisition and validation: 15-60 minutes,
  depending on SEC response time;
- five largest-request Gemma pilot: 5-30 minutes;
- remaining development extraction: estimated from the five-call pilot and
  expected to take roughly 1-8 hours on the installed GPU; and
- deterministic development scoring and artifact verification: 5-30 minutes.

These are planning ranges, not claims. Measured pilot timing replaces them
before the long extraction begins.

Each completed unit is written once beneath
`e/aapl_sec_gemma_lean_evidence_v3/` with canonical hashes and a stage manifest.
The resumable units are source acquisition, each sealed Gemma call, the complete
development extraction batch, deterministic scoring, and each later stage.
Resume must validate all earlier bytes and identities and continue from the
first incomplete unit. It may not silently regenerate or replace a committed
model response.

A stop request saves the current checkpoint, ends the worker cleanly, verifies
that no experiment process remains, and unloads the model only after the worker
has exited. No computer power action is part of this workflow.

## Stage gates and decisions

Development, confirmation, and final use the exact frozen v2.2 scientific
gates. In summary:

- development must show broad, non-concentrated 10-bps edge, sufficient valid
  semantic outputs and overlay episodes, material improvement over both the
  inherited baseline and no-meaning ablation, and action-changing value versus
  the frozen learner;
- confirmation must independently show useful results at both cost levels,
  at least three winning years, semantic value over controls, and profitable
  action-changing online learning; and
- strict final success requires at least 0.5 percentage points of 5-bps excess
  return in each of 2024, 2025, and 2026 YTD through 2026-07-09, positive edge
  in each period at 10 bps, useful aggregate semantic and online-learning
  differences, and every safety and accounting check.

Zero changed actions fails every semantic or learning-value claim. A failed
stage is preserved and stops later-stage access. A completed or preflight-
rejected result must update `e/APPROACH_COMPARISON.md`, preserve all diagnostics
and checksums, and be committed and pushed.

Historical outperformance cannot authorize real capital. A historically
promising survivor must next make append-only prospective paper decisions
before its outcomes are known. Broker execution requires separate fresh user
authorization outside this approach.
