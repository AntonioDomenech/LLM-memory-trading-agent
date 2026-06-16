# Local Gemma AAPL Benchmark Goal

## Compact Goal Prompt

Execute `docs/local_gemma_aapl_benchmark_goal.md` end-to-end. Implement and run the local-only Ollama `gemma4:12b` AAPL benchmark loop with no paid APIs, API cost `$0.00`, point-in-time LLM memory, monitoring, tests, and allowlisted autonomous patches. Work on `codex/local-gemma-aapl-loop`. Commit current code before each train/test process. Repeat train/test/diagnose/patch until AI beats AAPL buy-and-hold in 2025 with zero invalid decisions and local-only proof, or until the user stops/hard safety blocker.

## Summary

Run the Apple benchmark entirely locally with Ollama and Gemma 4 12B, with hard no-paid-API safeguards. The loop trains on 2000-2024 using LLM-generated memory, tests on full-year 2025, then autonomously analyzes and patches the agent until it beats AAPL buy-and-hold or the user stops it.

Local hardware reports an RTX 3080 with 10GB VRAM, not a 3060. Gemma 4 12B Q4 is expected to fit around 6.7GB, so it is the right first target.

## Key Changes

- Add a local-only model provider path.
- Use Ollama with `gemma4:12b`.
- Configure the OpenAI-compatible local endpoint as `http://127.0.0.1:11434/v1`.
- Use a dummy local key only, never a paid API key.
- Block non-loopback model URLs when `no_paid_api_mode=true`.
- Force estimated API cost to `$0.00` for local runs.
- Add a full local benchmark preset:
  - `mode="single_stock"`.
  - `symbol="AAPL"`.
  - Train from `2000-01-01` to `2024-12-31`.
  - Test from `2025-01-01` to `2025-12-31`.
  - `memory_mode="model_specific_cases_and_lessons"`.
  - `outcome_learning_mode="llm_reflection_lessons"`.
  - `use_cached_llm=false` for real loop iterations.
- Implement true LLM-generated training memory:
  - During training, run local Stage 1, critic, and Stage 2 decisions.
  - After outcomes are knowable, generate compact local-model lessons.
  - Store lessons with `knowledge_timestamp`.
  - Retrieve only lessons where `knowledge_timestamp <= decision_date`.
- Add an autonomous improvement loop:
  - Commit the current code state before starting each train/test process so the code is saved and recoverable.
  - Run full train/test.
  - Stop only if AI return beats AAPL buy-and-hold, with zero invalid decisions and local-only cost proof.
  - If it fails, analyze diagnostics and apply only allowlisted autonomous patches.
  - Allowlisted patch categories: prompts, exposure policy, critic framing, memory lesson formatting, diagnostics, and benchmark config.
  - Run backend tests before rerunning the benchmark.
  - Commit each successful loop patch separately if implementation occurs under repository update rules.
- Add monitoring:
  - Capture GPU utilization, VRAM, temperature, power, system RAM, call latency, tokens/sec or chars/sec, JSON repair rate, invalid decision rate, and total run time.
  - Abort or pause if GPU temperature, VRAM, RAM, or Ollama errors indicate instability.
  - Save monitoring logs with each benchmark run.

## Implementation Details

- Start on a dedicated branch such as `codex/local-gemma-aapl-loop`.
- Install Ollama on Windows if missing.
- Pull `gemma4:12b`.
- Run a local JSON smoke test before any benchmark.
- If 12B cannot run reliably, stop and report the failure rather than silently using a weaker model.
- Enforce no-cost execution:
  - Reject `api.openai.com`.
  - Reject hosted model endpoints.
  - Reject paid news APIs.
  - Reject non-loopback inference URLs.
  - Disable paid embeddings and paid enrichment sources.
  - Permit only free/local data sources already used by the app.
- One full loop iteration means:
  - Preflight.
  - Commit the current code state before starting the train/test process.
  - Train on 2000-2024.
  - Test on 2025.
  - Collect diagnostics.
  - Judge success.
  - Analyze failure if needed.
  - Patch if needed.
  - Test.
  - Rerun.
- Keep user-visible pause/cancel support.
- Record every iteration's config, commit hash, model tag, benchmark metrics, and monitoring metrics.

## Success Definition

- Primary: AI 2025 return must beat AAPL buy-and-hold.
- Required cleanliness: zero invalid allocations or decisions.
- Required proof: local model endpoint only and estimated API cost exactly `$0.00`.

## Test Plan

### Unit Tests

- Local provider works without a real OpenAI key.
- Non-loopback URLs are rejected in no-paid mode.
- Local runs report zero API cost.
- `gemma4:12b` config selects chat completions against Ollama.
- LLM-generated lessons are stored and retrieved point-in-time safely.
- Success evaluator requires beating AAPL buy-and-hold.
- Monitoring parser handles `nvidia-smi` output.
- Abort thresholds trigger on simulated resource pressure.

### Integration Tests

- Fake local OpenAI-compatible server returns JSON decisions.
- Full single-stock train/test pipeline runs with local-provider config.
- Autonomous loop performs analysis and applies only allowlisted patch categories.

### Manual Acceptance

- Ollama health check passes.
- Full AAPL 2025 benchmark completes.
- Results page shows zero API cost and local model metadata.
- Monitoring log is attached to the run.
- Loop continues until success or user stop.

## Assumptions

- "Good" means beating AAPL buy-and-hold in 2025.
- Ollama Gemma 4 12B is the default model because the reported RTX 3080 10GB VRAM should fit the quantized 12B model.
- Internet downloads for free model/data files are allowed.
- Paid API usage is not allowed.
- Electricity and bandwidth are outside the app's cost accounting.
- No GitHub release is created unless explicitly requested later.
