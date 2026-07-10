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

Training does not mean Gemma trades every day from 2000 onward. The affordable implementation builds numerical, labeled historical cases and fits/retrieves from them. Historical model quality can be reported for diagnosis and selection, but never presented as held-out success.

Development inside 2000-2023 uses chronological walk-forward splits: fit on the past, predict the next development block, inspect the result, and only then advance. Those predictions help select or reject a pattern, but they remain practice evidence because we are allowed to tune from them. After the design is fixed, it may be refit on all causally mature information through 2023 and then frozen. The 44.6% pre-2024 cash-call figure from the rejected predecessor is therefore not part of the test score; it was a development warning that the apparent pattern was weak.

During the frozen 2024-onward exam, the policy may observe information that would have been available by each decision date, including earlier prices within 2024. It may not fit, recalibrate, change thresholds, add outcome lessons, or use any later result. Causal replay is reported separately because it deliberately learns an outcome after that outcome matures, which is useful for simulating an everlasting live system but is not the same fixed-model test.

## Gemma's own knowledge cutoff

Warehouse chronology is not the only possible source of future leakage. The installed `gemma4:12b` model was pretrained on data through January 2025, according to Google's Gemma 4 model card. An ordinary prompt containing `AAPL`, `Apple`, an exact 2024 date, and recognizable prices could therefore invite the model to recall the historical outcome from its weights.

Prompt blinding alone cannot prove that a high-dimensional market vector was not recognized from pretraining. Therefore Gemma has **no decision authority and receives no market prompt in frozen or causal historical replay**. Those trades must come from the manifest-bound `precutoff_quantitative_policy`, fitted only from outcomes that matured by the selection cutoff. Gemma becomes decisional only in genuine live operation after its January 2025 knowledge cutoff.

As defense in depth for any explicitly diagnostic historical call, the harness uses `historical_prompt_blinding=true` and the versioned `identity_relative_time_scale_free_v2` contract. Immediately before such a model call, the exact prompt is transformed as follows:

- AAPL, Apple, and context ticker identities become stable pseudonyms such as `ASSET_1`, `MARKET_1`, and `SENTIMENT_1`.
- Exact calendar dates become offsets relative to an undisclosed `T0`.
- Only explicitly allowlisted scale-free numerical fields survive. Raw prices, volumes, share counts, cash/equity amounts, unknown numeric fields, free-text numbers, and raw SEC statement amounts are removed.
- Scale-free returns, volatility, drawdowns, moving-average distances, ranks, probabilities, portfolio weights, and matured pre-cutoff outcomes remain.
- Parsed pseudonymous output is mapped back to engine symbols only after generation; the raw response remains pseudonymous for audit.

Frozen certification requires zero historical market-decision calls to Gemma, zero allocation repairs, the cutoff-safe quantitative authority, the declared model cutoff, and the current blinding contract. Historical LLM caches and news text are forbidden. This removes Gemma's parametric memory from the retrospective trading result; it still does not make the already inspected 2024 onward period globally pristine. Locked future paper trading remains the strongest evidence.

Source: [Google Gemma 4 model card - Training Dataset](https://huggingface.co/google/gemma-4-12B#training-dataset).

## Separate causal replay and live learning

`local_gemma_aapl_causal_replay_config()` uses a different namespace and enables `online_test_learning`. Every eligible executed decision creates a pending experience. A 20-session outcome becomes a lesson only when its exit price is knowable; it can influence later decisions but never the decision that created it or any earlier one.

`local_gemma_aapl_live_config(stream_id=...)` requires one durable stream id. A non-dry live snapshot also requires the exact Ollama model digest and committed implementation identity; both are part of the memory compatibility fingerprint, so changed code or model weights cannot silently reuse the learned state. It restores portfolio and schedule state after restarts, reconciles dividends and splits, and accumulates matured lessons indefinitely. If a crash leaves a pending lesson newer than the saved portfolio—or any learning artifact without portfolio state—the next snapshot fails closed instead of silently restarting from cash. A policy, feature, price-basis, or cost change requires a new stream.

Historical replay fills at the next adjusted open. Live paper execution runs once between 09:30 and 10:00 New York time: Gemma sees only the last completed daily bar, and a separate current-minute quote is fetched after the decision. This is an operational approximation, not a guaranteed broker fill.

## Decision system

The numerical policy compares the current point-in-time state with causally available historical cases and estimates whether cash is likely to outperform AAPL after costs. In frozen and causal historical replay, that policy alone owns the weekly/stress-event action. In live operation, Gemma receives the same support and can propose an action; the gate can block unsupported cash, leverage, or a short.

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
