from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from .fundamentals import sec_fundamentals_snapshot
from .macro import fred_macro_snapshot
from .market_data import index_context, market_snapshot, price_window_start
from .news import fetch_news_bundle
from .schemas import BenchmarkConfig, PortfolioState, SecretConfig, model_to_dict
from .storage import BenchmarkStore


def information_manifest(config: BenchmarkConfig) -> Dict[str, Any]:
    sources = [
        {
            "name": "Primary market data",
            "provider": "yfinance with Stooq fallback",
            "fields": ["OHLCV", "returns", "moving averages", "realized volatility"],
            "cost": "0 EUR/month",
        },
        {
            "name": "News discovery",
            "provider": ", ".join(config.data_sources.news_sources),
            "fields": ["title", "url", "publisher/domain", "published timestamp", "short summary when available"],
            "cost": "0 EUR/month by default; optional free keyed tiers can be added",
        },
        {
            "name": "Company fundamentals",
            "provider": "SEC EDGAR companyfacts",
            "fields": ["revenue", "net income", "assets", "liabilities", "equity", "EPS", "shares"],
            "cost": "0 EUR/month",
        },
        {
            "name": "Macro context",
            "provider": "FRED when a free API key is configured",
            "fields": ["policy rate", "10-year yield", "unemployment", "CPI"],
            "cost": "0 EUR/month",
        },
        {
            "name": "Benchmark memory",
            "provider": "local SQLite run history",
            "fields": ["prior decisions", "prior executions", "constraint events", "realized result context"],
            "cost": "0 EUR/month",
        },
    ]
    return {
        "monthly_data_budget_eur": 10,
        "default_expected_monthly_cost_eur": 0,
        "sources": sources,
        "llm_decision_rule": "The model is the portfolio manager. The simulator only applies market mechanics and records failures.",
    }


def build_information_bundle(
    config: BenchmarkConfig,
    secrets: SecretConfig,
    as_of_date: Optional[str] = None,
    portfolio: Optional[PortfolioState] = None,
    store: Optional[BenchmarkStore] = None,
) -> Dict[str, Any]:
    as_of_date = as_of_date or config.start_date
    lookback_start = price_window_start(config.start_date)
    primary_market = market_snapshot(config.symbol, as_of_date, lookback_start)

    indexes = []
    if config.data_sources.include_index_context:
        indexes = index_context(config.data_sources.index_symbols, as_of_date, lookback_start)

    news = fetch_news_bundle(config, secrets, as_of_date)

    fundamentals: Dict[str, Any] = {"status": "disabled"}
    if config.data_sources.include_sec_fundamentals:
        fundamentals = sec_fundamentals_snapshot(config.symbol, secrets.sec_user_agent, as_of_date)

    macro: Dict[str, Any] = {"status": "disabled"}
    if config.data_sources.include_fred_macro:
        macro = fred_macro_snapshot(secrets.fred_api_key, as_of_date)

    memory = []
    if store is not None:
        memory = store.recent_memory(config.symbol.upper(), as_of_date, limit=5)

    portfolio_payload = model_to_dict(portfolio) if portfolio else {
        "cash": config.initial_cash,
        "position_shares": 0.0,
        "equity": config.initial_cash,
    }

    return {
        "schema_version": "benchmark-input-v1",
        "symbol": config.symbol.upper(),
        "company_name": config.company_name,
        "as_of_date": as_of_date,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "information_cutoff": f"All data must be dated on or before {as_of_date}.",
        "portfolio_state": portfolio_payload,
        "market": primary_market,
        "index_context": indexes,
        "fundamentals": fundamentals,
        "macro": macro,
        "news": news,
        "memory": memory,
        "manifest": information_manifest(config),
    }
