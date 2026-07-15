# AAPL binary-regime selector long-run audit v1

This branch answers the user's separate relaxed question: can the already
frozen long/cash selector beat AAPL on average over a long continuous history,
even though it failed its stricter 2019-2023 incremental-validation contract?

It is a post-rejection diagnostic, not a retry, refit, or new holdout. The
rejection on `codex/aapl-binary-regime-union-selector-v1` remains final. This
branch may not alter the selector, its expert union, its 2018 checkpoint, its
2023 checkpoint, or any failed validation gate.

## Evidence classification

- 2000-2018 is calibration/training.
- 2019-2023 is the already-opened selector-level validation. It beat AAPL in
  aggregate but failed three frozen selector-versus-union robustness gates.
- 2024-2026 YTD is a repeated historical audit. The exact selector has not
  used these rows to choose its parameters, but the inherited expert family
  was influenced by earlier later-period research. This is therefore not a
  globally pristine test.
- Only decisions locked before future market opens in a separate append-only
  paper branch can provide prospective evidence.

No row after 2023 may be opened for this branch until this contract and its
runner are committed and pushed. There is one run and no threshold retry.

## Frozen policy and continuous learning

The implementation in `binary_regime_union_selector.py` is reused byte for
byte. Its rules remain:

- the exact contextual-plus-weak-trend canonical union opportunity stream;
- a two-state SPY/QQQ 20-session market regime;
- one action-independent 10-bps cash-edge lesson per canonical opportunity;
- signal-time regime storage and admission only at close `t+2`;
- exponential per-opportunity state updates with the existing discount,
  readiness, thresholds, defaults, and hysteresis;
- learning from skipped opportunities as well as traded ones; and
- targets of exactly 0% or 100% AAPL.

The online arm continues admitting each newly matured lesson through 2026 YTD.
The frozen-2023 arm starts from the same through-2023 state but admits no later
lessons. The exact fixed union and always-long AAPL are additional comparators.
SPY and QQQ are price-based market-regime proxies. This policy does not use
news, article text, or LLM sentiment.

There is one account inception on 2005-01-01. Accounts, union cooldown, learner
state, and pending lessons never reset at 2019, 2024, a calendar year, or a
reporting boundary. Both strategy and benchmark make the same initial all-in
AAPL purchase. Cash episodes are attributed to the period containing their
entry open.

## Parent and input authorization

The runner must fail closed unless the committed validation bundle at
`e/binary_regime_union_selector_v1/binary-regime-union-selector-validation-v1`
has:

- the exact self-hash
  `sha256:0354355ac460042f96663d1a45cf5e9a8cf4fe873ddf5cc87afefd9c4b5d81dc`;
- `stage_pass=false`, gate `passed=false`, report/gate equality, and exactly
  `base_5bps_minimum_two_positive_incremental_years`,
  `stress_10bps_minimum_two_positive_incremental_years`, and
  `stress_10bps_veto_benefit_not_concentrated` as the three failures;
- a passing embedded development parent;
- intact payload hashes, implementation hashes, runtime versions, source
  prefix, and causal through-2023 checkpoint; and
- no row after 2023.

This is an intentional verifier for one rejected parent. The passing-parent
verifier from earlier staged runners cannot be reused unchanged.

The final input must be an exact clean Git-tracked blob with the frozen
six-column schema, unique increasing sessions, and finite positive values. Its
canonical identity is 6,875 rows from 1999-03-10 through 2026-07-09, date
sequence SHA-256
`b88df14b4ec60534ace68645ee19c8a0b7d03d0c2c1829a3ad48f8a0a24c9299`,
and bounded-result SHA-256
`sha256:c01447f975d4a90e49c315f23177f357966363b1ec4790632fa54c0dee250b21`.
The manifest records the raw tracked-blob hash. Before returning its first
2024 row, the loader must independently reproduce the exact committed
through-2023 source prefix and checkpoint.

Git authorization is deliberately two phase. Before the attempt lock, the
runner proves that HEAD is pushed, all code and contract dependencies are
clean, and the final-input HEAD and index object IDs match, while excluding
the final-input worktree path from the global status scan so Git cannot hash
its later contents. After the lock, it requires the HEAD blob, index blob, and
local input bytes to be identical and the path-specific status to be clean.
A dirty or changed local input discovered in that second phase consumes the
attempt; it cannot be fixed and retried.

Parent manifest and payload verification occurs before any final-input access.
After that verification, a dates-only physical maximum-date scan is allowed to
prove the file bound, but no 2024+ price/value row may be returned to Python
before the exact through-2023 prefix and checkpoint replay pass.

Immediately after that replay and the through-2023 gate mirror, the runner
must atomically create `e/binary_regime_longrun_audit_v1/AUDIT_ATTEMPT_LOCK.json`
before reading the raw final-input blob or returning any 2024+ value. The lock
is never removed after either success or failure, blocks every later attempt,
and is copied byte for byte into the sealed bundle. Its exact content and hash
must also be bound into the report and manifest. This makes a crash after later
history is opened an auditable consumed attempt rather than an invisible retry
opportunity.

## Reporting periods and controls

The audit reports every calendar year from 2005 through 2025 plus 2026 YTD,
the combined 2005-2026-YTD account, and the repeated-audit periods 2024, 2025,
and 2026 YTD. It runs both 5 and 10 bps of adverse slippage per changing leg.

Every result must reconcile within absolute tolerance `1e-10`:

- full-account active edge to complete entry-attributed cash episodes;
- each reporting-period edge to its entry-attributed episodes;
- selector-minus-union edge to veto benefit; and
- online-minus-frozen edge to XOR one-session divergence episodes, with
  orientation (`online_cash_frozen_long` or `online_long_frozen_cash`), both
  changing-leg costs, and entry-open-period attribution, for the full account
  and every reporting period at both costs.

The always-long control must equal same-ledger AAPL buy-and-hold.

## Strict recent-history diagnostic

`strict_recent_history_pass` is true only if online active log edge versus AAPL
is strictly greater than `0.001` at both costs in each of 2024, 2025, and 2026
YTD. Passing is reported only as a repeated historical target pass and cannot
erase the earlier validation rejection or create a prospective claim.

## Relaxed long-run criterion

`post_hoc_long_run_robustness_pass` is true only if, at both costs, the
continuous online 2005-2026-YTD account:

- has total active log edge versus AAPL greater than `0.001`;
- has strictly positive edge in at least 12 of the 22 annual/YTD periods;
- remains strictly positive after subtracting the single largest annual/YTD
  active edge;
- has positive aggregate edge across periods whose same-ledger AAPL benchmark
  return is strictly negative;
- executes at least 150 complete cash episodes;
- has at least a 50% strictly beneficial-episode rate and strictly positive
  mean and median episode edge;
- remains strictly positive after subtracting its five largest complete
  episode edges;
- has no episode above 25% of all positive episode edge; and
- has a strategy maximum drawdown greater than or equal to the same-ledger
  AAPL maximum drawdown, since drawdowns are represented as negative values.

Total selector-minus-fixed-union edge must also remain strictly positive at
both costs. Empty/nonfinite inputs fail closed; comparisons use `1e-12` only
as a floating-point zero tolerance.

These thresholds were selected after the pre-2024 history was already known,
so they are post-hoc and non-confirmatory. The report must emit the same gate
inputs and pass/fail statuses using only rows through 2023. In particular, it
must disclose that before the new audit the selector already had 16/14
positive years, 137 episodes, 64.2%/62.0% win rates, and total edges
1.2767/1.1397 at 5/10 bps; almost every proposed robustness condition was
already satisfied except the 150-episode count. A final pass is descriptive
robustness, not new validation evidence.

`fixed_policy_candidate_for_prospective_paper` is true only when both
`post_hoc_long_run_robustness_pass` and `strict_recent_history_pass` are true.
Neither status means reliable autonomous money-making proof.

## Adaptive-value status

The online and frozen-2023 arms have identical features, opportunity stream,
and starting state; only post-2023 lesson admission differs. A threshold
crossing is an actual regime-latch transition, recorded with its date, regime,
and pre/post state. The report must count those transitions, XOR differing
complete episodes, their distinct entry years, and their signed 10-bps
incremental edge.

- With no causal state-threshold crossing or no action difference, adaptive
  value is `unexercised`.
- Any exercised case with fewer than 10 complete differing episodes or fewer
  than two distinct entry years is `exercised_insufficient_evidence`.
- With sufficient exposure, 10-bps incremental edge above `1e-12` is
  `exercised_positive`, below `-1e-12` is `exercised_negative`, and otherwise
  is `exercised_flat`. None is reliable prospective evidence.

Adaptive status cannot rescue or invalidate the descriptive fixed-policy
robustness score. It does control the separate learning claim:

- `online_learning_historical_value_demonstrated` is true only for
  `exercised_positive`;
- `online_learning_candidate_for_paper` requires both the fixed-policy
  candidate flag and `exercised_positive`; and
- `unexercised` or `exercised_insufficient_evidence` must state plainly that
  useful continual learning has not been demonstrated. The online arm may
  still be retained as a shadow challenger, not as a proven learning system.

## Safety, cost, runtime, and next step

Shorting, leverage, borrowing, negative cash, margin interest, interest on
cash, paid APIs, network calls, news calls, and LLM calls are forbidden.
Runtime must remain below 3,600 seconds and external cost must be zero. The
output is an immutable checksummed bundle bound to the exact Git commit and
parent artifacts.

The strict runtime gate is checked at the last instant before the private
sealed directory is atomically promoted to its final name. Successful atomic
promotion is completion of the run; elapsed time observed after the rename is
reported only operationally and is not a second gate that could contradict an
already-promoted bundle.

If `fixed_policy_candidate_for_prospective_paper` is true, a separate
prospective paper branch will lock four append-only arms: AAPL buy-and-hold,
fixed union, frozen selector, and online selector. The report must identify
whether the online arm is a learning candidate or only a shadow challenger.
Historical performance alone cannot authorize broker execution or real
capital.

All YTD labels must say `through 2026-07-09`. The sealed output must include a
final online continuation checkpoint with both regime states, all pending
lessons, and trailing account-union cooldown context for the prospective
branch.
