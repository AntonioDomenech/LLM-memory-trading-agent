from __future__ import annotations

import ipaddress
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import requests

from .schemas import BenchmarkConfig, DataSourceConfig, SecretConfig
from .historical_blinding import HISTORICAL_BLINDING_CONTRACT

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


def resolve_local_model_digest(config: BenchmarkConfig) -> str:
    """Read the exact digest currently served by the local Ollama runtime."""

    parsed = urlparse(config.local_model_base_url or LOCAL_OLLAMA_BASE_URL)
    tags_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"
    response = requests.get(tags_url, timeout=10)
    response.raise_for_status()
    models = (response.json() or {}).get("models") or []
    record = next(
        (
            item
            for item in models
            if config.model in {item.get("name"), item.get("model")}
        ),
        None,
    )
    digest = str((record or {}).get("digest") or "")
    if not digest:
        raise RuntimeError(f"Ollama did not report a digest for {config.model!r}")
    return digest


def resolve_repository_identity(repo_root: Path | None = None) -> Dict[str, Any]:
    """Return the current Git commit and cleanliness of the local implementation."""

    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty, "repo_root": str(root)}


def with_verified_local_runtime_identity(config: BenchmarkConfig) -> BenchmarkConfig:
    """Stamp a config with the model/code identity verified at this moment."""

    repository = resolve_repository_identity()
    if repository["dirty"]:
        raise RuntimeError("Live learning requires a clean committed worktree")
    payload = config.model_dump() if hasattr(config, "model_dump") else config.dict()
    payload["local_model_digest"] = resolve_local_model_digest(config)
    payload["implementation_commit"] = repository["commit"]
    return BenchmarkConfig(**payload)


def validate_no_paid_api_mode(config: BenchmarkConfig, secrets: SecretConfig) -> None:
    if config.evaluation_mode != "legacy":
        try:
            train_end = date.fromisoformat(config.train_end)
            test_start = date.fromisoformat(config.test_start)
            test_end = date.fromisoformat(config.test_end)
            selection_cutoff = date.fromisoformat(config.selection_cutoff)
            fixed_cutoff = (
                date.fromisoformat(config.fixed_evaluation_cutoff)
                if config.fixed_evaluation_cutoff
                else None
            )
        except ValueError as exc:
            raise ValueError("Non-legacy evaluation contracts require ISO date cutoffs") from exc
        if train_end >= test_start:
            raise ValueError("Training must end strictly before the evaluation window starts")
        if selection_cutoff != train_end:
            raise ValueError("selection_cutoff must exactly equal train_end")
        if fixed_cutoff is not None and test_end > fixed_cutoff:
            raise ValueError("test_end cannot exceed fixed_evaluation_cutoff")
        if config.evaluation_mode == "frozen_holdout":
            if config.online_test_learning:
                raise ValueError("frozen_holdout forbids learning from evaluation outcomes")
            if str(config.memory_online_stream_id or "").strip():
                raise ValueError("frozen_holdout forbids a durable online learning stream")
            if int(config.max_test_days or 0) != 0:
                raise ValueError(
                    "frozen_holdout forbids partial test-day smoke runs; use pre-2024 or synthetic data"
                )
        elif config.evaluation_mode == "causal_online_replay":
            if not config.online_test_learning:
                raise ValueError("causal_online_replay requires matured test learning")
        elif config.evaluation_mode == "live_learning":
            if not config.online_test_learning:
                raise ValueError("live_learning requires matured online learning")
            if not str(config.memory_online_stream_id or "").strip():
                raise ValueError("live_learning requires a durable memory_online_stream_id")
        if config.evaluation_mode in {"frozen_holdout", "causal_online_replay"}:
            if config.memory_mode != "deterministic_market_cases":
                raise ValueError(
                    f"{config.evaluation_mode} requires deterministic_market_cases so historical "
                    "execution cannot enter an LLM training phase"
                )
            if str(config.symbol or "").upper() != "AAPL" or str(
                config.company_name or ""
            ).strip().casefold() not in {"apple", "apple inc", "apple inc."}:
                raise ValueError(
                    f"{config.evaluation_mode} AAPL contract requires symbol='AAPL' and company_name='Apple'"
                )
            if not config.historical_prompt_blinding:
                raise ValueError(
                    f"{config.evaluation_mode} requires historical_prompt_blinding because model "
                    "weights may contain post-selection market facts"
                )
            if config.historical_prompt_blinding_contract != HISTORICAL_BLINDING_CONTRACT:
                raise ValueError(
                    f"{config.evaluation_mode} requires blinding contract "
                    f"{HISTORICAL_BLINDING_CONTRACT!r}"
                )
            if not str(config.model_training_data_cutoff or "").strip():
                raise ValueError(
                    f"{config.evaluation_mode} requires a declared model_training_data_cutoff"
                )
            try:
                date.fromisoformat(config.model_training_data_cutoff)
            except ValueError as exc:
                raise ValueError("model_training_data_cutoff must be an ISO date") from exc
            if config.data_sources.news_sources or int(config.max_news_per_symbol or 0) > 0:
                raise ValueError(
                    f"{config.evaluation_mode} identity blinding currently forbids news text; use "
                    "scale-free market sentiment features or a separately audited semantic "
                    "redaction contract"
                )
            if config.use_cached_llm:
                raise ValueError(
                    f"{config.evaluation_mode} forbids cached LLM decisions; every model response "
                    "must be generated by the manifest-bound runtime"
                )
            if config.historical_decision_authority != "precutoff_quantitative_policy":
                raise ValueError(
                    f"{config.evaluation_mode} requires precutoff_quantitative_policy decision "
                    "authority because Gemma's own training cutoff overlaps historical evaluation"
                )
    if config.outcome_learning_mode == "counterfactual_online" and (
        float(config.commission_per_trade or 0.0) != 0.0
        or float(config.commission_per_share or 0.0) != 0.0
    ):
        raise ValueError(
            "counterfactual_online currently supports slippage costs only; "
            "commission_per_trade and commission_per_share must both be zero."
        )
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
        "max_daily_turnover": 0.0,
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
        "llm_reflection_cadence": "weekly",
        "embedding_provider": "local",
        "exposure_critic_enabled": False,
        "strict_preflight": True,
        "require_paid_micro_pilot": False,
        "macro_policy": "omit_if_missing",
        "news_policy": "real_titles_or_aggregate_events",
        "monitoring_enabled": True,
        "warehouse_recycle_interval_days": 250,
        "data_sources": data_sources,
    }
    payload.update(overrides)
    return BenchmarkConfig(**payload)


def local_gemma_aapl_online_config(**overrides: Any) -> BenchmarkConfig:
    """Return the primary AAPL preset with a frozen post-2023 evaluation model.

    A blank ``memory_online_stream_id`` is deliberate: HybridMemory binds it to
    the benchmark run id so every replay starts from the same clean historical
    snapshot. Evaluation outcomes cannot enter memory. A separate causal replay
    or live deployment may resume learning after outcomes mature.
    """

    data_sources = DataSourceConfig(
        news_sources=[],
        max_news_per_day=0,
        include_sec_fundamentals=True,
        include_fred_macro=False,
        include_index_context=True,
    )
    payload: Dict[str, Any] = {
        "mode": "single_stock",
        "run_preset": "local_gemma_aapl_online",
        "symbol": "AAPL",
        "company_name": "Apple",
        "train_start": "2000-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2026-07-09",
        "max_train_days": 0,
        "max_test_days": 0,
        "historical_price_basis": "adjusted",
        "live_frequency": "daily_open",
        "model": LOCAL_OLLAMA_MODEL,
        "model_provider": "ollama_local",
        "endpoint": "chat_completions",
        "no_paid_api_mode": True,
        "local_model_base_url": LOCAL_OLLAMA_BASE_URL,
        "local_ollama_num_ctx": 6144,
        "allow_short": False,
        "max_gross_exposure": 1.0,
        "max_daily_turnover": 0.0,
        "turnover_prompt_buffer": 0.0,
        "turnover_edge_multiplier": 0.0,
        "single_stock_action_space": "long_cash_hold",
        "temperature": 0.0,
        "max_output_tokens": 520,
        "use_cached_llm": False,
        "stage1_chunk_size": 1,
        "max_news_per_symbol": 0,
        "memory_mode": "deterministic_market_cases",
        "memory_retrieval": "structured",
        "deterministic_memory_per_symbol": 8,
        "deterministic_memory_max_items": 100,
        "memory_k_neighbors": 75,
        "memory_examples_per_symbol": 6,
        "memory_namespace": "aapl-frozen-v2",
        "memory_base_snapshot_id": "aapl-2000-2023-adjusted-v2",
        "memory_online_stream_id": "",
        "memory_policy_version": "long-cash-counterfactual-v2",
        "memory_feature_schema_version": "aapl-market-state-v1",
        "outcome_learning_mode": "counterfactual_online",
        "evaluation_mode": "frozen_holdout",
        "selection_cutoff": "2023-12-31",
        "fixed_evaluation_cutoff": "2026-07-09",
        "globally_pristine": False,
        "historical_holdout_reveal_count_lower_bound": 10,
        "historical_prompt_blinding": True,
        "historical_prompt_blinding_contract": HISTORICAL_BLINDING_CONTRACT,
        "model_training_data_cutoff": "2025-01-31",
        "historical_decision_authority": "precutoff_quantitative_policy",
        "online_test_learning": False,
        "online_learning_horizon_days": 20,
        "online_policy_enabled": True,
        "online_policy_min_samples": 40,
        "online_policy_max_neighbors": 75,
        "online_policy_min_neighbor_separation_days": 21,
        "online_policy_min_feature_overlap": 0.90,
        "online_policy_risk_off_probability": 0.62,
        "online_policy_min_confidence": 0.55,
        "online_policy_min_active_return": 0.002,
        "decision_cadence": "weekly_event",
        "minimum_holding_days": 5,
        "action_hysteresis_confirmations": 2,
        "event_drawdown_trigger": -0.08,
        "event_volatility_trigger": 0.45,
        "reset_book_at_test_start": True,
        "benchmark_contract_version": "aapl-frozen-holdout-v2",
        "embedding_provider": "local",
        "exposure_critic_enabled": False,
        "strict_preflight": True,
        "require_paid_micro_pilot": False,
        "macro_policy": "omit_if_missing",
        "news_policy": "real_titles_or_aggregate_events",
        "monitoring_enabled": True,
        "warehouse_recycle_interval_days": 250,
        "data_sources": data_sources,
    }
    payload.update(overrides)
    return BenchmarkConfig(**payload)


def local_gemma_aapl_causal_replay_config(**overrides: Any) -> BenchmarkConfig:
    """Replay post-2023 data while learning only after each outcome matures."""

    payload: Dict[str, Any] = {
        "evaluation_mode": "causal_online_replay",
        "online_test_learning": True,
        "memory_namespace": "aapl-causal-replay-v2",
        "benchmark_contract_version": "aapl-causal-online-v2",
    }
    payload.update(overrides)
    return local_gemma_aapl_online_config(**payload)


def local_gemma_aapl_live_config(
    *,
    stream_id: str = "aapl-live-v1",
    **overrides: Any,
) -> BenchmarkConfig:
    """Return the online preset with an explicit durable live-learning stream."""

    stream_id = str(stream_id or "").strip()
    if not stream_id:
        raise ValueError("A durable live stream_id is required.")
    payload: Dict[str, Any] = {
        "memory_online_stream_id": stream_id,
        "evaluation_mode": "live_learning",
        "online_test_learning": True,
        "historical_decision_authority": "llm",
        "memory_namespace": "aapl-live-v2",
        "benchmark_contract_version": "aapl-live-learning-v2",
    }
    payload.update(overrides)
    return local_gemma_aapl_online_config(**payload)


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
