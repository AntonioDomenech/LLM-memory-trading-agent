# AAPL Chronological Online-Learning System

## Objective and honest success criterion

The system has one economic target: after trading costs, its test-period return must exceed an AAPL buy-and-hold benchmark measured on the same adjusted-price execution contract. If it does not, holding AAPL was the better result for that period.

This is a research benchmark, not a profit guarantee. A model can beat one historical window through luck or overfitting and then lose money live. Because the 2025 result has already been inspected, it should be treated as an engineering and walk-forward test, not as permanently untouched evidence. Promotion to real capital needs later unseen periods, repeated walk-forward results, and explicit loss limits.

## Architecture

### 1. Immutable 2000-2024 case library

The base memory is generated deterministically from AAPL and market history through 2024. Each case contains only features that were available at its decision timestamp, plus outcome labels that are attached only after their horizon elapsed. Prices and returns use the adjusted historical execution basis so splits and dividends do not create a different contract from the benchmark.

Building this library makes no Gemma calls. It replaces the old multi-year daily LLM replay with a reproducible case bank that can be rebuilt and audited quickly. The base snapshot id is `aapl-2000-2024-adjusted-v1`; its first deterministic content hash is registered locally, and a later rebuild with different content is rejected under the same id. Changing warehouse history, features, or execution contract therefore requires a new snapshot id.

### 2. Chronological lessons that continue to mature

Every eligible decision creates a pending experience. The experience is not evidence yet. After the configured 20-trading-day horizon, the system computes after-cost counterfactual results for long, cash, and short exposure and promotes the record to a matured structured lesson. The estimator uses the cash-versus-long comparison; the deployed policy cannot select the short counterfactual.

During a 2025 replay, lessons from January can therefore influence later dates only after their outcomes became knowable. December outcomes cannot travel backward into January. The same mechanism continues indefinitely in live operation: new experiences remain pending, mature when their horizon arrives, and join the usable stream. Compact Stage 1 and Stage 2 payloads reserve explicit `recent_online_lessons` slots, so newly matured experience reaches Gemma instead of being truncated behind the larger historical case bank; the same lessons also update the numerical policy.

### 3. Replay isolation and a persistent live stream

The base snapshot is reusable, but learned outcomes have two different lifecycles:

- Historical replay: `memory_online_stream_id` is blank. The runtime binds the stream to the unique run id, so every replay begins from the same clean pre-2025 base and learns forward within that replay only. A previous replay's 2025 lessons are never preloaded.
- Live operation: snapshots must use one explicit durable stream id, for example `aapl-live-v1`. That stream survives process restarts and keeps accumulating matured lessons. A policy or feature-schema change must start a new compatible stream instead of silently mixing old lessons into a different policy.

This distinction preserves causal testing while still allowing the deployed system to learn forever.

For live snapshots, use `local_gemma_aapl_live_config()` (or set an explicit stable stream id). A blank live stream is rejected. Every later snapshot must reuse that exact id. The live path restores its portfolio, recent decision state, and event re-arm state after process restarts; reconciles dividends and splits; matures due lessons from the local warehouse; and can backfill a missing matured price from the same free Yahoo source used for live quotes. It fails closed if durable portfolio state is malformed and will not silently restart the account from cash. Do not reuse the stream id after changing the policy, feature schema, price basis, or cost assumptions.

The live decision runs once between 09:30 and 10:00 New York time. Gemma sees only the last completed daily bar, and the portfolio shown in its prompt is marked at that completed close. After every Gemma and repair call is finished, the paper executor fetches a separate one-minute quote. It fails closed unless that quote belongs to the current New York session, falls inside the opening window, and is from the current minute. The post-fetch observation timestamp and raw entry price become the pending lesson's entry point; neither is retroactively inserted into the model input.

This is causal, but it is deliberately described as an operational approximation rather than an exact market-on-open fill. Historical replay fills at the next adjusted open; live paper execution fills at the validated post-decision observation and learns from that actual entry. Before real capital, either connect a broker-supported market-on-open workflow decided before the open or validate the observed opening-delay/slippage contract separately. When a live lesson spans a later split, its stored raw quote is divided by the cumulative split factor and put on the same adjusted basis as its exit before a return is learned.

### 4. Risk-off estimator plus scheduled Gemma decisions

The structured policy estimates whether cash is likely to outperform AAPL after costs by comparing the current point-in-time state with compatible matured cases. It reports expected active return, uncertainty, effective sample size, evidence strength, and a recommended action. Pending outcomes are excluded, and overlapping outcome windows cannot both count as independent neighbors. The empirical-Bayes frequency and interval are conservative decision support, not a claim of held-out probability calibration. Promotion beyond paper trading requires pre-2025 walk-forward reliability, Brier-score, and interval-coverage reporting.

Gemma receives this decision support and the auditable case evidence. It is called on a weekly schedule and on configured market-stress events, rather than being asked to reinterpret nearly identical inputs every day. A minimum holding period and two-confirmation hysteresis reduce one-day reversals.

The numerical gate is deliberately one-sided: strong empirical risk-off evidence gives Gemma permission to choose cash, but never forces a trade. Weak evidence blocks both entering cash and remaining there. Returning from unsupported cash to the long baseline bypasses discretionary hysteresis and also applies on no-Gemma cadence days. This keeps every risk-off move explainable while making AAPL ownership the mechanical default.

The action space is deliberately small:

- `BUY_ALL`: hold AAPL up to the permitted gross exposure.
- `CASH_ALL`: liquidate AAPL only when the evidence supports a positive after-cost active advantage for cash.
- `HOLD`: preserve the current position unless a mechanical gross-exposure correction is required.

Short selling is disabled. News is also disabled in this preset because the local data did not provide reliable point-in-time headlines; synthetic GDELT event labels are not presented as news.

## Configuration contract

Use `local_gemma_aapl_online_config()` for the new system. The older `local_gemma_aapl_config()` remains available under the `legacy` CLI preset for reproducibility.

The online preset uses local Ollama `gemma4:12b`, local embeddings, no paid API, adjusted historical prices, a reset portfolio at the test boundary, structured counterfactual lessons, and the long/cash/hold action contract. It runs one benchmark iteration and never auto-patches itself after seeing the result.

`dry_run=True` may write ordinary run/audit rows, but it does not mature or queue lessons, register a base snapshot, or save the durable live portfolio/schedule state.

## Commands

Run the focused implementation tests (no benchmark or model call):

```powershell
python -m pytest tests/test_aapl_online_preset.py tests/test_aapl_online_engine.py tests/test_online_policy.py tests/test_online_memory.py tests/test_deterministic_online_memory.py -q
```

Check configuration and local warehouse coverage only. This does not pull or call Gemma and does not start a benchmark:

```powershell
python -m agent_benchmark.local_gemma_loop --preset aapl-online --preflight-only
```

When ready, run a bounded five-day pipeline smoke test. This does run Gemma and the benchmark, so it is intentionally not part of repository verification:

```powershell
python -m agent_benchmark.local_gemma_loop --preset aapl-online --max-test-days 5 --no-pull --no-commit-before-run
```

Run the full 2025 benchmark exactly once after the smoke test passes:

```powershell
python -m agent_benchmark.local_gemma_loop --preset aapl-online --max-train-days 0 --max-test-days 0 --no-pull --no-commit-before-run
```

If a monitored run is safely interrupted, resume its existing memory stream and checkpoint rather than starting a new replay:

```powershell
python -m agent_benchmark.local_gemma_loop --resume-run-id <run-id> --no-pull --no-commit-before-run
```

The report must be rejected as a success unless it beats the same-window AAPL buy-and-hold result after costs, has no invalid allocations, and proves local-only estimated API cost of `$0.00`.
