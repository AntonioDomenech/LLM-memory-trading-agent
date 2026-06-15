from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SecretConfig(BaseModel):
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    marketaux_key: str = ""
    newsapi_key: str = ""
    finnhub_key: str = ""
    fred_api_key: str = ""
    sec_user_agent: str = ""


class DataSourceConfig(BaseModel):
    news_sources: List[str] = Field(default_factory=lambda: ["gdelt"])
    max_news_per_day: int = 8
    include_sec_fundamentals: bool = True
    include_fred_macro: bool = False
    include_index_context: bool = True
    index_symbols: List[str] = Field(default_factory=lambda: ["SPY", "QQQ", "IWM", "^VIX", "^TNX"])
    rss_feeds: List[str] = Field(default_factory=list)


class BenchmarkConfig(BaseModel):
    mode: Literal["single_stock", "balanced_50_portfolio"] = "balanced_50_portfolio"
    run_preset: Literal["balanced_50_mini", "single_stock_diagnostic", "budget_official", "full_official"] = "balanced_50_mini"
    symbol: str = "AAPL"
    company_name: str = "Apple"
    selected_symbols: List[str] = Field(default_factory=list)
    start_date: str = "2025-01-02"
    end_date: str = "2025-03-31"
    train_start: str = "2024-12-02"
    train_end: str = "2024-12-09"
    test_start: str = "2025-01-02"
    test_end: str = "2025-01-08"
    max_train_days: int = 5
    max_test_days: int = 5
    historical_cadence: Literal["daily"] = "daily"
    fill_timing: Literal["next_open"] = "next_open"
    live_frequency: Literal["hourly"] = "hourly"
    model: str = ""
    endpoint: str = "responses"
    initial_cash: float = 1000.0
    max_days: int = 20
    allow_short: bool = True
    max_leverage: float = 1.0
    max_gross_exposure: float = 1.0
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0
    temperature: float = 0.0
    max_output_tokens: int = 900
    use_cached_llm: bool = True
    stage1_chunk_size: int = 10
    max_news_per_symbol: int = 2
    memory_mode: Literal["deterministic_market_cases", "model_specific_cases_and_lessons"] = "deterministic_market_cases"
    memory_retrieval: Literal["deterministic_similarity", "hybrid", "structured"] = "deterministic_similarity"
    deterministic_memory_per_symbol: int = 1
    deterministic_memory_max_items: int = 50
    memory_k_neighbors: int = 50
    memory_examples_per_symbol: int = 2
    prompt_detail_level: Literal["compact", "full"] = "compact"
    embedding_provider: Literal["local", "openai"] = "local"
    decision_process: Literal["two_stage_llm"] = "two_stage_llm"
    strict_preflight: bool = True
    require_paid_micro_pilot: bool = True
    max_nonzero_positions: int = 12
    max_daily_turnover: float = 0.20
    turnover_edge_multiplier: float = 3.0
    invalid_run_abort_count: int = 3
    invalid_run_abort_rate: float = 0.05
    macro_policy: Literal["omit_if_missing", "include_status_rows"] = "omit_if_missing"
    news_policy: Literal["real_titles_or_aggregate_events", "raw_titles"] = "real_titles_or_aggregate_events"
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)


class LocalConfig(BaseModel):
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    secrets: SecretConfig = Field(default_factory=SecretConfig)


class PortfolioState(BaseModel):
    cash: float
    position_shares: float = 0.0
    equity: float


class PortfolioBook(BaseModel):
    cash: float
    positions: Dict[str, float] = Field(default_factory=dict)
    equity: float
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0


class PreviewRequest(BaseModel):
    config: Optional[BenchmarkConfig] = None
    as_of_date: Optional[str] = None


class RunRequest(BaseModel):
    config: Optional[BenchmarkConfig] = None
    dry_run: bool = False


class BenchmarkRunRequest(BaseModel):
    config: Optional[BenchmarkConfig] = None
    dry_run: bool = False


class PreviewRequestV2(BaseModel):
    config: Optional[BenchmarkConfig] = None
    phase: Literal["train", "test", "live"] = "test"
    decision_date: Optional[str] = None


class LiveSnapshotRequest(BaseModel):
    config: Optional[BenchmarkConfig] = None
    dry_run: bool = False
    force: bool = False


class SaveConfigRequest(BaseModel):
    benchmark: BenchmarkConfig
    secrets: SecretConfig = Field(default_factory=SecretConfig)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
