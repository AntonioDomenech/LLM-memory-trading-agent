# AAPL Frozen-Holdout and Online-Learning System

## Objective and evidence rule

The economic target is to beat same-ledger AAPL buy-and-hold after costs without leverage or shorting. The evidence roles are deliberately separate:

- `training_diagnostics`: 2000-01-01 through 2023-12-31. Outcomes may be used to discover patterns, select a model, and fit its final pre-test state. These results are not test evidence.
- `frozen_holdout`: post-2023 evaluation. Current completed market inputs are visible, but model parameters, thresholds, normalization, memories, and lessons cannot change. This is the only retrospective pass/fail score.
- `causal_online_replay`: starts from the same pre-2024 base, then admits a new lesson only after its outcome matures. It tests operational adaptation but cannot count as the frozen holdout.
- `live_learning`: the durable paper-trading continuation of the causal replay.

The repository has already inspected 2024 onward during earlier research. The report records `globally_pristine=false` and a conservative reveal-count lower bound (currently 10). The split prevents a new candidate from training on those outcomes, but it cannot make those dates globally pristine again. Only locked future paper trading can provide genuinely untouched evidence.

## Training and frozen model

The base case library is generated deterministically from AAPL and market history through 2023. A case contains close-time features and only those 1/5/20/60-session labels whose outcome date is no later than 2023-12-31. Building it makes no Gemma calls.

The base snapshot id is `aapl-2000-2023-adjusted-v2`. Its content hash and exact cutoff/provenance metadata are registered locally. Reusing that id with changed content or metadata fails closed. The run also records a system-manifest hash over the public configuration, base content, model digest, prompt source, implementation source, and Git commit.

`local_gemma_aapl_online_config()` is retained as the primary CLI-compatible preset, but it now means a frozen evaluation:

- `train_end = selection_cutoff = 2023-12-31`
- `evaluation_mode = frozen_holdout`
- `online_test_learning = false`
- no durable online stream

The engine snapshots both the estimator and scoped learning memory before and after a frozen run. A changed digest invalidates the run. The queue and maturity functions also reject direct calls from a frozen test.

Training does not mean Gemma trades every day from 2000 onward. The affordable implementation builds numerical, labeled historical cases and fits/retrieves from them; Gemma is used only for scheduled test/live decisions. Historical model quality can be reported for diagnosis and selection, but never presented as held-out success.

## Separate causal replay and live learning

`local_gemma_aapl_causal_replay_config()` uses a different namespace and enables `online_test_learning`. Every eligible executed decision creates a pending experience. A 20-session outcome becomes a lesson only when its exit price is knowable; it can influence later decisions but never the decision that created it or any earlier one.

`local_gemma_aapl_live_config(stream_id=...)` requires one durable stream id. A non-dry live snapshot also requires the exact Ollama model digest and committed implementation identity; both are part of the memory compatibility fingerprint, so changed code or model weights cannot silently reuse the learned state. It restores portfolio and schedule state after restarts, reconciles dividends and splits, and accumulates matured lessons indefinitely. If a crash leaves a pending lesson newer than the saved portfolio—or any learning artifact without portfolio state—the next snapshot fails closed instead of silently restarting from cash. A policy, feature, price-basis, or cost change requires a new stream.

Historical replay fills at the next adjusted open. Live paper execution runs once between 09:30 and 10:00 New York time: Gemma sees only the last completed daily bar, and a separate current-minute quote is fetched after the decision. This is an operational approximation, not a guaranteed broker fill.

## Decision system

The numerical policy compares the current point-in-time state with causally available historical cases and estimates whether cash is likely to outperform AAPL after costs. Gemma receives that support on a weekly schedule and on configured stress events. The gate can block an unsupported cash decision; it never forces leverage or a short.

The action space is:

- `BUY_ALL`: hold AAPL at no more than 100% exposure.
- `CASH_ALL`: liquidate AAPL when the frozen/causal evidence permits it.
- `HOLD`: preserve the current valid position.

News is disabled because the local warehouse does not contain a trustworthy point-in-time headline archive. Synthetic GDELT event labels are not shown to Gemma as news.

## Verification and run commands

Run implementation tests without calling Gemma or opening the holdout:

```powershell
python -m pytest tests/test_aapl_online_preset.py tests/test_aapl_online_engine.py tests/test_online_policy.py tests/test_online_memory.py tests/test_deterministic_online_memory.py -q
```

Check configuration and warehouse coverage without checking or calling Ollama:

```powershell
python -m agent_benchmark.local_gemma_loop --preset aapl-online --preflight-only
```

Do not use the first few holdout days as a smoke test. Frozen mode rejects `--max-test-days`; pipeline smoke testing must use unit tests, synthetic data, or a pre-2024 segment.

After the candidate is selected and committed, run the complete frozen window once:

```powershell
python -m agent_benchmark.local_gemma_loop --preset aapl-online --max-train-days 0 --max-test-days 0 --no-pull
```

Run the learning-forward comparison separately:

```powershell
python -m agent_benchmark.local_gemma_loop --preset aapl-causal-replay --max-train-days 0 --max-test-days 0 --no-pull
```

A causal replay is never eligible for frozen-test success, even if it beats buy-and-hold. A frozen report succeeds only if it beats same-window AAPL buy-and-hold after costs, has no invalid allocations, proves zero paid-API cost, and proves unchanged learning state.
