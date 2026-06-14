from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SecretConfig(BaseModel):
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
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
    symbol: str = "AAPL"
    company_name: str = "Apple"
    start_date: str = "2025-01-02"
    end_date: str = "2025-03-31"
    model: str = ""
    endpoint: str = "responses"
    initial_cash: float = 100000.0
    max_days: int = 20
    allow_short: bool = False
    max_leverage: float = 1.0
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0
    temperature: float = 0.0
    max_output_tokens: int = 900
    use_cached_llm: bool = True
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)


class LocalConfig(BaseModel):
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    secrets: SecretConfig = Field(default_factory=SecretConfig)


class PortfolioState(BaseModel):
    cash: float
    position_shares: float = 0.0
    equity: float


class PreviewRequest(BaseModel):
    config: Optional[BenchmarkConfig] = None
    as_of_date: Optional[str] = None


class RunRequest(BaseModel):
    config: Optional[BenchmarkConfig] = None
    dry_run: bool = False


class SaveConfigRequest(BaseModel):
    benchmark: BenchmarkConfig
    secrets: SecretConfig = Field(default_factory=SecretConfig)


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
