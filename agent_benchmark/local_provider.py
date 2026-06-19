from __future__ import annotations

import ipaddress
from typing import Any, Dict
from urllib.parse import urlparse

from .schemas import BenchmarkConfig, DataSourceConfig, SecretConfig

LOCAL_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
LOCAL_OLLAMA_MODEL = "gemma4:12b"
LOCAL_DUMMY_API_KEY = "ollama-local-dummy-key"
PAID_NEWS_SOURCES = {"marketaux", "newsapi", "finnhub"}


def is_loopback_url(url: str) -> bool:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_local_model_run(config: BenchmarkConfig | None, secrets: SecretConfig | None = None) -> bool:
    if not config:
        return False
    base_url = model_base_url(config, secrets or SecretConfig())
    return bool(config.no_paid_api_mode or config.model_provider == "ollama_local" or is_loopback_url(base_url))


def model_base_url(config: BenchmarkConfig | None, secrets: SecretConfig) -> str:
    if config and config.local_model_base_url:
        return config.local_model_base_url.rstrip("/")
    return (secrets.openai_base_url or LOCAL_OLLAMA_BASE_URL).rstrip("/")


def local_auth_headers(config: BenchmarkConfig | None, secrets: SecretConfig) -> Dict[str, str]:
    key = secrets.openai_api_key or ""
    if config and config.no_paid_api_mode:
        key = LOCAL_DUMMY_API_KEY
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def validate_no_paid_api_mode(config: BenchmarkConfig, secrets: SecretConfig) -> None:
    if not config.no_paid_api_mode:
        return

    base_url = model_base_url(config, secrets)
    if not is_loopback_url(base_url):
        raise ValueError(f"no_paid_api_mode requires a loopback model URL, got {base_url!r}.")
    if "api.openai.com" in base_url.lower():
        raise ValueError("no_paid_api_mode rejects api.openai.com.")
    if config.model_provider != "ollama_local":
        raise ValueError("no_paid_api_mode requires model_provider='ollama_local'.")
    if config.endpoint != "chat_completions":
        raise ValueError("Ollama local no-paid runs must use chat_completions.")
    if config.embedding_provider != "local":
        raise ValueError("no_paid_api_mode requires local embeddings.")

    sources = {str(item).strip().lower() for item in (config.data_sources.news_sources or [])}
    paid_sources = sorted(sources & PAID_NEWS_SOURCES)
    if paid_sources:
        raise ValueError(f"no_paid_api_mode rejects paid news sources: {', '.join(paid_sources)}.")


def local_gemma_aapl_config(**overrides: Any) -> BenchmarkConfig:
    data_sources = DataSourceConfig(
        news_sources=["gdelt"],
        max_news_per_day=8,
        include_sec_fundamentals=True,
        include_fred_macro=False,
        include_index_context=True,
    )
    payload: Dict[str, Any] = {
        "mode": "single_stock",
        "run_preset": "local_gemma_aapl_full",
        "symbol": "AAPL",
        "company_name": "Apple",
        "train_start": "2000-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
        "max_train_days": 0,
        "max_test_days": 0,
        "model": LOCAL_OLLAMA_MODEL,
        "model_provider": "ollama_local",
        "endpoint": "chat_completions",
        "no_paid_api_mode": True,
        "local_model_base_url": LOCAL_OLLAMA_BASE_URL,
        "local_ollama_num_ctx": 6144,
        "allow_short": True,
        "max_gross_exposure": 1.0,
        "max_daily_turnover": 2.0,
        "turnover_prompt_buffer": 0.0,
        "turnover_edge_multiplier": 0.0,
        "single_stock_action_space": "trinary_all_in",
        "temperature": 0.0,
        "max_output_tokens": 520,
        "use_cached_llm": False,
        "stage1_chunk_size": 1,
        "memory_mode": "model_specific_cases_and_lessons",
        "memory_retrieval": "hybrid",
        "outcome_learning_mode": "llm_reflection_lessons",
        "embedding_provider": "local",
        "exposure_critic_enabled": True,
        "strict_preflight": True,
        "require_paid_micro_pilot": False,
        "macro_policy": "omit_if_missing",
        "news_policy": "real_titles_or_aggregate_events",
        "monitoring_enabled": True,
        "data_sources": data_sources,
    }
    payload.update(overrides)
    return BenchmarkConfig(**payload)


def local_gemma_secret_config(**overrides: Any) -> SecretConfig:
    payload: Dict[str, Any] = {
        "openai_api_key": LOCAL_DUMMY_API_KEY,
        "openai_base_url": LOCAL_OLLAMA_BASE_URL,
        "openai_embedding_model": "",
        "marketaux_key": "",
        "newsapi_key": "",
        "finnhub_key": "",
        "fred_api_key": "",
    }
    payload.update(overrides)
    return SecretConfig(**payload)
