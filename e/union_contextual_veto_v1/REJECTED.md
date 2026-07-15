# Rejected: union contextual veto v1

## Decision

Rejected at the frozen 2005-2018 chronological development gate. The
2019-2023 validation and the 2024, 2025, and 2026-YTD final periods were not
opened, and no later-stage input snapshot was created.

The fixed contextual-plus-weak-trend union remained strong, but the new
discounted Bayesian veto made no action change at all. The learner and union
therefore have exactly the same ledgers, episodes, returns, and active edge.
This is a genuine zero incremental result, not a rounding tie.

## Development result

All figures cover the continuous 2005-2018 adjusted-open ledger and include
adverse transaction cost on every changing leg.

| Metric | Learner = union, 5 bps | Learner = union, 10 bps |
|---|---:|---:|
| Active log edge vs AAPL buy-and-hold | 1.092826 | 0.971826 |
| Wealth above AAPL buy-and-hold | +198.27% | +164.28% |
| Final equity from $1,000 | $115,649.44 | $102,418.14 |
| AAPL buy-and-hold final equity | $38,773.51 | $38,754.15 |
| Cash episodes | 121 | 121 |
| Winning cash episodes | 79 | 76 |
| Episode win rate | 65.29% | 62.81% |
| Incremental active log edge vs union | 0.000000 | 0.000000 |
| Vetoes | 0 | 0 |

At 5 bps the shared policy had positive edge in 12 of 14 years; at 10 bps it
had positive edge in 11 of 14. All seven fixed two-year folds were positive at
both costs. It also had positive edge in each negative-AAPL year—2008, 2015,
and 2018—with aggregate 10-bps edge `0.703551115469572`. These are useful
properties of the inherited union, but they do not demonstrate that the new
learner added value.

## Why the learner was inert

After the 2005 administrative start there were 124 raw union candidates and
121 canonical union episodes. Every stage candidate was past the model's
warm-up, yet `model_veto_prediction` and the executed veto count were both
zero.

The frozen rule required the posterior upper bound to be strictly below
`-0.001`. The lowest bound was still positive: `0.00873844137971393` on
2012-05-30. The lowest posterior mean was `-0.005338089071169744`, but its
standard error was `0.01402718168954954`, leaving an upper bound of
`0.012644757854832767`. The model never found a context that it could identify
as harmful with the preregistered confidence.

This is not a case where simply deleting the confidence safeguard would have
worked. In a labeled post-rejection development diagnostic, eight canonical
trades had a negative posterior mean. Only three would have been beneficial
vetoes; vetoing all eight would have reduced 10-bps active log edge by
`0.0753981647795239`. The features and linear model did not reliably separate
losing union trades from winners.

## Gate outcome

The stage passed 37 of 47 recorded gates and failed 10. All baseline,
chronology, shadow-lesson, exact-union, same-ledger, accounting, and
no-leverage checks passed. The failures were the intended learning-value
checks: no minimum veto count, no learner improvement over union, no positive
incremental folds, and no measurable veto quality or diversification.

The underlying union remains a long-horizon research lead under the relaxed
average-performance criterion. This particular adaptive overlay does not
merit later-period access because it adds no decisions and no edge. A
successor should solve a different prediction problem rather than tune this
confidence threshold.

## Integrity and runtime

- Branch: `codex/aapl-union-contextual-veto-v1`.
- Immutable implementation commit:
  `08a4d0e9a03f4c632b83cd1e674899cacfc49148`.
- Physical input: 4,986 rows from 1999-03-10 through 2018-12-31; no later rows.
- Long/cash only; maximum exposure 1.0; no leverage, shorting, borrowing,
  negative cash, or cash interest.
- LLM calls: 0. API calls: 0. Estimated external cost: $0.00.
- Sealed runtime: 4.40 seconds; complete command runtime: 5.65 seconds, both
  below the 3,600-second limit.
- Manifest identity:
  `sha256:d9af3a9caff52a0cbadc5e162751f42644d10d7c19ab01edf56037e3fa7cf973`.
- Canonical bounded-price hash:
  `sha256:31b56551b8d1b837f2e69178ab7f206b6bf3c19d11d9d47be0e33cf501db3f45`.
- All 17 recorded file checksums, the manifest self-hash, the committed Git
  blobs, and all episode/ledger identities were independently verified.

The complete forecast, continuation checkpoint, ledgers, episodes, metrics,
veto diagnostics, provenance, gate report, manifest, and checksums are
preserved in `union-contextual-veto-development-v1/`.
