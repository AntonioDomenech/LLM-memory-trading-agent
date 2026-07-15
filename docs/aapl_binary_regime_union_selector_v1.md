# AAPL binary regime union selector v1

This document freezes one materially different experiment before any row after
2018 is opened for it. The selector is long/cash only, has no paid service or
LLM dependency, and continues learning from each outcome only when that
outcome becomes knowable.

## Evidence classification and train/test boundary

Rows through 2018 are calibration and training, not test evidence. The binary
regime, structural defaults, discount, readiness count, thresholds, and gates
were selected after examining only through-2018 outcomes. Attractive
2005-2018 results are therefore in-sample diagnostics and must never be called
proof of generalization.

The first selector-level untouched test is 2019-2023. The exact implementation
and through-2018 checkpoint must be committed and pushed before a snapshot
containing a 2019 row can be created or opened. Any failed validation gate
rejects this branch without a retry and without access to 2024 or later data.

This is not a globally pristine end-to-end holdout. The inherited contextual
expert was selected using earlier 2000-2023 research and the inherited
weak-trend expert was selected after 2024 had already been inspected. Thus
2019-2023 can validate the newly frozen selector only; 2024 onward remains a
repeated historical audit for the expert family. Only a newly locked forward
paper/live period can supply prospective evidence.

## Fixed opportunity stream and account continuity

The raw contextual and weak-trend expert signals and their arithmetic are
exactly those frozen in `aapl_chronological_exhaustion_expert_v1.md`. Their raw
union is true when either virtual expert signal is true.

There is one account inception on 2005-01-01. Raw candidates before that date
are removed from the trading account and the one-session union cooldown is
then applied once. This exact account union must reproduce the known 121
complete 2005-2018 episodes. It is never reset at 2019, 2024, a calendar-year
boundary, or a reporting boundary. A prior accepted union signal therefore
continues to suppress the next raw candidate even if the two dates straddle a
stage or year.

The selector is a pure union filter:

`selector_cash[t] = account_union_cash[t] AND cash_for_regime[t]`.

It can never create a cash action where the fixed union stays long. A selector
veto does not resurrect a `t+1` raw candidate suppressed by the union
cooldown. A bounded historical ledger masks a decision whose `t+1` or `t+2`
open lies outside the physical snapshot only after cooldown, and never
re-canonicalizes the selector. The live prediction and its pending lesson
remain in the checkpoint even when that unresolved action is excluded from a
historical score.

The account is simulated continuously from 2005 through the current physical
end. Validation and final dates are reporting windows, not fresh accounts.
Both selector and AAPL buy-and-hold make the same initial all-in AAPL purchase.
Cash episodes are attributed to the period containing their entry open.

## Action-independent causal lesson

Learning uses the continuous canonical union shadow stream from 2000 onward,
including opportunities before account inception and opportunities the
selector vetoes. A signal at completed close `t` receives the exact
10-bps-per-changing-leg cash edge

`y[t] = log(adjusted_open[t+1] / adjusted_open[t+2]) + log(0.999 / 1.001)`.

Its regime is stored at signal close. The lesson enters memory only at
completed close `t+2`, after the exit open is known and before a candidate at
that same close is scored. It cannot affect its own action or any earlier
action. Signal closes before 2000-01-01 are warm-up only. The action stream
learned from this one conservative label is identical in the 5-bps and 10-bps
ledgers.

## Frozen binary market-sentiment regime

At completed close `t`:

`spy_return_20[t] = SPY_adjusted_close[t] / SPY_adjusted_close[t-20] - 1`

`qqq_return_20[t] = QQQ_adjusted_close[t] / QQQ_adjusted_close[t-20] - 1`

`risk_on[t] = (spy_return_20[t] > 0) AND (qqq_return_20[t] > 0)`.

Equality is not risk-on. Both values use only the current and earlier completed
closes. There is no normalization over a full stage. A raw union candidate
with a nonfinite required return aborts the run.

This discrete conjunction is intentionally different from the failed
seven-feature linear veto. It asks whether union cash trades behave differently
when both broad-market trends are positive, rather than fitting correlated
continuous coefficients to a sparse sample.

## Frozen two-state online learner

The learner has one independent state for `risk_on` and one for
`not_risk_on`. Each contains:

- raw lesson count `n_raw`;
- effective count `n_eff`;
- discounted label sum `S`;
- discounted squared-label sum;
- the regime's current CASH/LONG latch.

Only the state belonging to the lesson's signal-time regime changes. For a
newly matured label `y`:

`n_raw = n_raw + 1`

`n_eff = 0.995 * n_eff + 1`

`S = 0.995 * S + y`

`Q = 0.995 * Q + y^2`

`mean = S / n_eff`.

The fixed structural defaults are LONG in `risk_on` and CASH in
`not_risk_on`. Before a regime reaches `n_eff >= 12`, its default is mandatory.
After readiness:

- `mean > +0.001` sets that regime's latch to CASH;
- `mean < -0.001` sets that regime's latch to LONG;
- equality or a mean inside `[-0.001, +0.001]` retains the prior latch.

The update happens before a same-close prediction. There is no grid, retry,
Bayesian confidence multiplier, or threshold search on later data. Every
vetoed opportunity still matures, so either regime can switch in the future
as its own outcomes change.

The through-2018 checkpoint must serialize both states, both latches, the
feature order, the last two union cooldown rows, and all unresolved shadow
lessons. Deterministic full-history replay is the source of truth. A later
checkpoint-resume optimization must reproduce it exactly before being trusted.

## Calibration diagnostics through 2018

The frozen implementation must reproduce these training facts within numeric
tolerance before a later snapshot is authorized:

| Regime | Matured lessons | Effective count | Discounted mean | End latch |
|---|---:|---:|---:|---|
| Risk-on | 25 | 23.555951 | -0.003904 | LONG |
| Not risk-on | 138 | 99.858259 | +0.010431 | CASH |

The chronological cold-start replay and the fitted end-state mapping happen to
make the same 2005-2018 actions: 20 risk-on union opportunities are vetoed and
101 not-risk-on opportunities are taken. The online state never changes either
structural default in this training window. This validates the static regime
association, not adaptive value.

At 5 bps the selector training edge must reproduce approximately
`1.20182849740637`, versus exact-union `1.092826209473698`. At 10 bps it must
reproduce approximately `1.10082843848966`, versus exact-union
`0.9718261388903109`.

Calibration authorizes validation only if all integrity checks pass and:

- both regime states have `n_eff >= 12` and the stated opposite-sign end
  means/latches;
- the selector improves exact union by more than `0.0001` at both costs;
- at least 8 union opportunities are vetoed;
- 10-bps veto benefit has at least a 50% beneficial rate, positive mean and
  median, and no veto above 50% of all positive veto benefit;
- incremental selector-minus-union edge is positive in at least 4 of the 7
  fixed two-year folds, remains positive after its best fold is removed, and
  has no fold above 50% of all positive incremental fold edge; and
- the same-ledger, no-leverage, shadow-lesson, checkpoint, source, cost, and
  runtime integrity gates below all pass.

These are training sanity gates, not holdout-success gates. A separate causal
cold-start report is also required but cannot improve the evidence label.

## Untouched selector validation: 2019-2023

The primary selector starts from the exact through-2018 online checkpoint and
continues adding one matured lesson at a time. It is not reset or refit.
Frozen-2018 actions are secondary diagnostics and cannot rescue online failure.

At both 5 and 10 bps, the online selector must:

- have aggregate active log edge versus same-ledger AAPL greater than `0.001`;
- have at least 3 of 5 positive calendar-year edges and remain positive after
  its best validation year is removed;
- execute at least 3 complete cash episodes with positive mean and median edge
  and no episode above 50% of all positive episode edge;
- have positive aggregate edge in any mechanically identified negative-AAPL
  years;
- exceed the exact continuous union by more than `0.0001` in aggregate;
- have positive incremental edge in at least 2 of 5 years; and
- veto at least 3 union opportunities.

At 10 bps, at least 50% of vetoes must be beneficial, veto benefit must have
positive mean and median, and no veto may provide over 50% of all positive
veto benefit. Any failure rejects the branch before a 2024 row is opened.

## Final repeated audit: 2024-2026 YTD

Only an exact passing through-2023 online checkpoint may authorize the final
snapshot. The same continuous account and learner proceed unchanged. Outcomes
from 2024 may influence 2025 only after each `t+2` maturity, exactly as they
would in live use; the same rule applies within 2025 and 2026.

At both costs, active log edge versus same-ledger AAPL must be strictly greater
than `0.001` in each of 2024, 2025, and 2026 YTD through 2026-07-09. Across the
combined final window the selector must also beat continuous union by more
than `0.0001`, with nonnegative incremental edge in at least two of the three
periods.

A relaxed continuous 2005-2026-YTD diagnostic reports total relative wealth,
positive versus negative years, negative-AAPL-year behavior, episode quality,
state switches, and drawdown. It may classify a strict failure as a promising
long-run lead, but cannot turn it into success.

## Execution, integrity, cost, and runtime

- Targets are exactly 0% or 100% AAPL.
- Shorting, leverage, borrowing, negative cash, margin interest, interest on
  cash, paid APIs, network calls, and LLM calls are forbidden.
- Cash trades execute from adjusted open `t+1` to adjusted open `t+2` with
  adverse slippage on both changing legs. There is no forced terminal sale.
- The always-long control must match the same continuous AAPL benchmark.
- Full-account active edge must reconcile to every complete episode. Each
  reporting-window edge must reconcile to episodes attributed by entry open,
  and selector-minus-union edge must reconcile to veto benefits.
- Learner cash dates must be a subset of exact account-union dates. Cooldown is
  applied before the selector and never reset after 2005.
- Each stage must finish within 3,600 seconds and report zero external cost.
- Inputs must be clean, Git-tracked, physically bounded CSV snapshots with the
  exact frozen columns, row coverage, hashes, and source lineage. Missing,
  extra, duplicate, revised, or later rows fail closed.
- A later-stage loader cannot return its first new market row until the exact
  committed parent manifest, pass report, implementation hashes, input prefix,
  and checkpoint have been independently verified.

If all historical gates pass, a separate
`codex/aapl-binary-regime-union-selector-live-v1` branch will add an append-only
`as-of` paper command. Broker execution remains out of scope until a locked
paper period has been observed prospectively.
