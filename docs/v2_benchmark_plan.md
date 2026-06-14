# V2 AI Fund Manager Benchmark Plan

## Goal

Build a local benchmark where the model is the portfolio manager. The model
decides action, target exposure, horizon, sizing, and risk plan from a dated
market-information bundle. The simulator only applies market mechanics and logs
all constraint events.

Monthly infrastructure/data budget target: under 10 EUR/month, excluding model
tokens. The default stack is 0 EUR/month.

## Local Tech Stack

- Backend: FastAPI, SQLite, DuckDB, Parquet, pandas, yfinance/Stooq, requests.
- Frontend: Vite, React, lucide-react, Recharts.
- Storage: `data/local_config.json` for local settings/secrets and
  `data/benchmark.db` for runs and decisions.
- Runtime: local only, with backend on `127.0.0.1:8000` and frontend on
  `127.0.0.1:5173`.

## LLM Input Bundle

Each decision receives a JSON object with:

- `portfolio_state`: cash, shares, equity.
- `market`: OHLCV, 1-day/5-day/20-day returns, SMA context, realized volatility.
- `index_context`: SPY, QQQ, IWM, VIX, 10-year yield proxy by default.
- `fundamentals`: SEC companyfacts values available before the decision date.
- `macro`: FRED series when a free FRED key is configured.
- `news`: dated headlines/articles from the configured provider chain.
- `memory`: previous local decisions and executions for the same symbol.
- `manifest`: provider names, cost profile, and benchmark rule statement.

The prompt contract requires the model to return JSON:

```json
{
  "action": "BUY",
  "target_exposure": 0.65,
  "confidence": 0.72,
  "horizon_days": 20,
  "expected_return_bps": 250,
  "risk_plan": {
    "max_loss_pct": 0.08,
    "stop_loss_price": 184.5,
    "take_profit_price": 230,
    "invalidation": "..."
  },
  "reasoning_summary": "...",
  "used_information": ["market", "news", "fundamentals"],
  "uncertainty": ["..."]
}
```

## Data Source Strategy

### Default Free Stack

- Prices: yfinance first, Stooq fallback for daily OHLCV. Cost: 0 EUR/month.
- News: GDELT DOC API. Cost: 0 EUR/month. Strength: broad global coverage and
  historical news discovery. Weakness: not finance-specialized and full text is
  not guaranteed.
- Fundamentals: SEC EDGAR companyfacts. Cost: 0 EUR/month. Strength: official
  US filings and XBRL facts. Weakness: US-listed focus and slower reporting.
- Macro: FRED with a free API key. Cost: 0 EUR/month. Strength: official macro
  time series. Weakness: key required and release-date alignment needs care.
- Local memory: SQLite. Cost: 0 EUR/month.

### Optional Free-Tier News Providers

- Marketaux free tier: 100 requests/day and 3 articles/request at the time this
  plan was written. Good finance-specific headline metadata.
- NewsAPI developer tier: 100 requests/day, development/testing use, 24-hour
  article delay and up to one-month search window at the time this plan was
  written.
- Finnhub free tier: useful for company-news endpoint when available on the
  user's account.
- RSS: user-provided feeds for extra public sources. These should be cached and
  filtered by ticker/company name.

The frontend supports API keys for these optional sources without requiring
them. The benchmark records source status for each date so weak or missing data
is visible in the run artifact.

## Execution Rules

- The model owns the investment decision.
- The broker simulator applies cash, share, shorting, leverage, fee, slippage,
  and price mechanics.
- Invalid actions, missing sizing, cash limits, short blocks, and leverage caps
  are not hidden. They are stored as constraint events and count against the run.
- The simulator does not add market expertise after the model speaks.

## Next Milestones

1. Add provider health checks and per-source freshness diagnostics.
2. Add benchmark suites by regime: bull, bear, sideways, high-rate, crisis.
3. Add multi-model batch runs over identical cached inputs.
4. Add stricter point-in-time release calendars for macro and filings.
5. Add vector retrieval for memory once baseline SQLite memory is validated.

## Historical Warehouse

The first warehouse target is the Balanced 50 stock universe plus market/index
context from `2000-01-01` to `2025-12-31`. It stores every calendar day,
including weekends and pre-IPO periods, under `data/warehouse/`.

Primary commands:

- `python -m agent_benchmark.warehouse bootstrap`
- `python -m agent_benchmark.warehouse download-prices`
- `python -m agent_benchmark.warehouse download-sec`
- `python -m agent_benchmark.warehouse download-macro`
- `python -m agent_benchmark.warehouse download-news`
- `python -m agent_benchmark.warehouse validate`
- `python -m agent_benchmark.warehouse status`

The GDELT news command is intentionally resumable and rate-limited. A full
50-stock 2000-2025 pull can take many hours because free access is the priority.
