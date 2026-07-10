# Benchmark Definition

## Purpose

This project is a benchmark for testing whether an AI model can act as the
portfolio manager of a market simulation. The goal is not to encode human
investment expertise into a deterministic trading strategy. The goal is to give
each model rich point-in-time market information, deterministic historical
memory, and then measure whether its own decisions make money, beat market
baselines, or perform poorly.

The core rule is:

> The model decides. The simulator only simulates market mechanics.

That means the model is responsible for direction, sizing, timing, confidence,
risk reasoning, and portfolio construction. The simulator applies prices, cash,
shorting limits, slippage, fees, fills, and constraint logging. It must not
rewrite a bad model decision into a better one.

## Benchmark Modes

The system supports two benchmark modes.

### Single-stock diagnostic mode

Single-stock mode tests one symbol at a time with cash and one position. It is
used for debugging data quality, prompts, memory retrieval, and per-stock
behavior.

The model can buy, short, reduce exposure, close a position, or hold. The same
no-lookahead, memory, and simulation rules apply.

### Balanced 50 portfolio mode

Balanced 50 portfolio mode is the official leaderboard benchmark. The model
manages one fake portfolio across the selected 50-stock universe, cash, and
optional short exposure.

This mode tests whether the model can compare opportunities, allocate capital,
manage portfolio-level risk, and learn relationships across stocks, indexes,
macro variables, fundamentals, news, sentiment, and prior outcomes.

## Benchmark Lifecycle

### 1. Warehouse phase

The warehouse stores raw point-in-time data before the benchmark transforms it.
The warehouse must preserve enough source detail to audit what the model could
have known at each decision timestamp.

Stored data includes:

- Stock OHLCV, adjusted close, volume, returns, volatility, and technical
  context.
- Index, ETF, sector, volatility, yield, and market-regime context.
- SEC fundamentals and filings, joined only after their filing or availability
  date.
- Macro series when available, with point-in-time release discipline.
- GDELT/news metadata, URLs, domains, event fields, source country, tone, and
  short article extracts when available.
- Data-source freshness, quality flags, missing-data flags, and download logs.
- Calendar rows for trading days, weekends, holidays, pre-IPO periods, and
  unavailable data.

Full article text is not required for the first benchmark contract. News starts
with metadata plus short extracts where available.

### 2. Deterministic training memory

Training is where the system builds a historical case library from warehouse
data. Dates are user-configurable. The default training period is:

- `train_start`: `2000-01-01`
- `train_end`: `2023-12-31`

Training memory is deterministic and shared across models. The system computes
historical market cases directly from prices, index context, volatility,
fundamentals, news density, and known later outcomes. The model does not write
the memory and no paid LLM calls are made during this phase.

This is the budget benchmark's definition of learning: the model enters the test
phase with retrieved historical examples, but the expensive daily LLM replay is
not required.

### 3. Frozen test phase

The test phase evaluates the model on a future or hidden period. Dates are
user-configurable. The default test period is:

- `test_start`: `2025-01-01`
- `test_end`: `2025-12-31`

During the primary frozen test, the model can use:

- Current point-in-time input data for the decision timestamp.
- Deterministic memory items generated from the training period.
- Historical outcomes whose outcome date is known before or equal to the memory
  cutoff.

It cannot add test-period outcomes to training memory, refit parameters, change
thresholds, or update normalization. A separate `causal_online_replay` may add
an experience only after its outcome matures, but that replay is operational
evidence and is not the frozen test score.

The model must not receive future prices, future news, future filings, future
macro releases, or future outcome labels before they would have been known.

### 4. Live phase

Live mode is a present-day paper benchmark. It should eventually run hourly
during US market hours and skip non-market hours and exchange holidays.

Each live run uses data that is fresh within a few minutes when available. The
model receives the latest point-in-time bundle, retrieves eligible memories,
and produces a paper-trading decision. The live benchmark is not connected to a
real brokerage account.

All transformations used in historical testing should be designed so they can
later run in live mode within a few minutes.

## Memory System

The benchmark uses deterministic market-case memory, not only same-stock memory.

Memory is built without LLM calls. It is shared across models so model
comparisons use the same historical evidence. Models still own the final
investment decisions, but they do not author or edit the memory.

Memory contains:

- Historical cases: point-in-time market state, news density, fundamentals,
  macro/index context, volatility, and data-quality state.
- Deterministic similarity features such as trailing returns, volatility, index
  regime, and VIX context.
- Known later stock outcomes after `1d`, `5d`, `20d`, and `60d` when those
  outcomes are available inside the training cutoff.

Every memory item must have at least:

- `mode`
- `symbol` or `portfolio_scope`
- `decision_timestamp`
- `knowledge_timestamp`
- `source_run_id`
- `memory_type`
- `content`
- `outcome_horizon`
- `outcome_available_at`

Retrieval must be point-in-time. A decision at time `T` can only retrieve memory
where `knowledge_timestamp <= T`, and a frozen post-2023 test uses only memory
whose labels matured by the 2023-12-31 selection cutoff.

For a historical LLM test, point-in-time retrieval is necessary but not sufficient.
If the base model's own training corpus overlaps the evaluation window, exact
issuer identities, calendar dates, raw prices, and raw financial-statement amounts
must be blinded before inference. The parsed response may be mapped back to engine
identifiers only after generation. The applied blinding contract and model training
cutoff must be recorded in the immutable run manifest.

The retrieval system should favor:

- Similar portfolio states.
- Similar market regimes.
- Similar stock, sector, index, volatility, macro, valuation, and news patterns.
- Prior market cases with known outcomes.
- Cross-stock relationships when they are relevant to the portfolio decision.

## Two-stage LLM Decision Process

Balanced 50 portfolio mode uses a two-stage LLM process. Both stages use the
same model being benchmarked.

### Stage 1: analyst and scoring pass

The model reviews compact evidence for stocks. If the universe is too large for
one prompt, stocks may be processed in deterministic chunks.

For each stock reviewed, Stage 1 returns:

- `symbol`
- `stance`: `bullish`, `bearish`, `neutral`, or `uncertain`
- `confidence`: `0.0` to `1.0`
- `expected_return_bps`
- `horizon_days`
- `key_evidence`
- `memory_refs`
- `uncertainty`
- `proposed_target_weight`

Stage 1 should identify opportunities, risks, and weak inputs. It does not
execute trades.

### Stage 2: portfolio manager pass

The model receives:

- Stage 1 outputs.
- Current portfolio state.
- Cash, exposure, and shorting constraints.
- Market/index/macro context.
- Important retrieved deterministic memories.
- Data quality and freshness flags.

Stage 2 returns the final portfolio allocation:

- `target_weights`: object mapping symbols to target portfolio weights.
- `cash_weight`
- `gross_exposure`
- `net_exposure`
- `confidence`
- `portfolio_thesis`
- `major_risks`
- `uncertainty`
- `expected_return_bps`
- `horizon_days`

The Stage 2 output is the model's final decision. The simulator only applies
mechanics and records failures.

## Trading Rules

Default rules:

- Long and short positions are allowed.
- No leverage is allowed.
- Portfolio gross exposure must be less than or equal to `1.0`.
- Cash target plus absolute position weights must respect the exposure limit.
- Fractional shares are allowed unless a specific experiment disables them.
- Slippage and fees are applied by the simulator.
- Invalid, missing, or impossible allocations are logged as benchmark failures.

The simulator may clip or reject orders only to enforce explicit mechanical
constraints. It must log each constraint event and count it against the run. It
must not add investment judgement after the model speaks.

## Information Bundle

Each decision should receive a structured input bundle with:

- `portfolio_state`
- `candidate_universe`
- `market_snapshots`
- `index_context`
- `sector_context`
- `macro_context`
- `fundamentals`
- `news_and_events`
- `article_extracts`
- `sentiment_and_tone`
- `data_quality`
- `memory`
- `benchmark_rules`
- `information_cutoff`

All fields must be dated or traceable to a source availability timestamp. Missing
or low-quality data is part of the decision problem and should be shown to the
model rather than hidden.

## Public Configuration Contract

The benchmark configuration should include:

```json
{
  "mode": "single_stock | balanced_50_portfolio",
  "train_start": "2000-01-01",
  "train_end": "2023-12-31",
  "test_start": "2024-01-01",
  "test_end": "2026-07-09",
  "live_frequency": "hourly",
  "initial_cash": 1000.0,
  "allow_short": true,
  "max_gross_exposure": 1.0,
  "memory_mode": "deterministic_market_cases",
  "memory_retrieval": "deterministic_similarity",
  "decision_process": "two_stage_llm",
  "evaluation_mode": "frozen_holdout",
  "historical_prompt_blinding": true,
  "historical_prompt_blinding_contract": "identity_relative_time_scale_free_v2",
  "model_training_data_cutoff": "2025-01-31",
  "historical_decision_authority": "precutoff_quantitative_policy",
  "online_test_learning": false
}
```

Users may change train and test dates, but a frozen evaluation must start after
its selection cutoff. The AAPL contract learns through 2023 and begins
evaluation in 2024; causal online replay is reported separately.

## Reporting

The benchmark reports multiple metrics instead of one magic score.

Reports should include:

- Final equity.
- Total return.
- Alpha versus SPY.
- Alpha versus QQQ.
- Alpha versus an equal-weight Balanced 50 baseline.
- Max drawdown.
- Volatility.
- Sharpe-like risk-adjusted return.
- Hit rate by decision horizon.
- Turnover.
- Long P&L and short P&L.
- Fees and slippage.
- Invalid decisions and constraint events.
- Model failures and parse failures.
- Memory coverage and retrieval quality.
- Data coverage and source freshness.

Single-stock reports should include equivalent per-symbol metrics and compare
against buy-and-hold for the same stock.

## Cost Constraint

The data and infrastructure budget target is under `10 EUR/month`, excluding
model tokens.

The default data stack should remain no-cost where possible:

- Local DuckDB and Parquet storage.
- yfinance and Stooq for daily prices.
- GDELT for historical news/event metadata.
- SEC EDGAR for fundamentals.
- FRED with a free key for macro data.

Optional paid or free-tier providers may be added only when the benchmark
records provider status, cost profile, and data coverage.

## Non-negotiable Fairness Rules

- No future data in training decisions, test decisions, or live decisions.
- No future outcome labels in memory before the outcome horizon has elapsed.
- No simulator-side investment intelligence.
- No silent correction of bad model allocations.
- Same cached input bundle and deterministic memory for each model when
  comparing models on the same decision timestamp.
- All prompts, inputs, decisions, executions, memories, and outcomes must be
  logged for audit.
