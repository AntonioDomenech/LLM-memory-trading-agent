from __future__ import annotations

import pytest

from agent_benchmark import local_gemma_loop, quality
from agent_benchmark.local_provider import (
    local_gemma_aapl_config,
    local_gemma_aapl_live_config,
    local_gemma_aapl_online_config,
    local_gemma_secret_config,
    validate_no_paid_api_mode,
)


def test_online_preset_is_local_chronological_and_replay_isolated():
    config = local_gemma_aapl_online_config()

    validate_no_paid_api_mode(config, local_gemma_secret_config())
    assert config.run_preset == "local_gemma_aapl_online"
    assert config.memory_mode == "deterministic_market_cases"
    assert config.memory_retrieval == "structured"
    assert config.memory_base_snapshot_id == "aapl-2000-2024-adjusted-v1"
    assert config.memory_online_stream_id == ""
    assert config.outcome_learning_mode == "counterfactual_online"
    assert config.online_test_learning is True
    assert config.online_learning_horizon_days == 20
    assert config.historical_price_basis == "adjusted"
    assert config.live_frequency == "daily_open"
    assert config.single_stock_action_space == "long_cash_hold"
    assert config.allow_short is False
    assert config.decision_cadence == "weekly_event"
    assert config.minimum_holding_days == 5
    assert config.action_hysteresis_confirmations == 2
    assert config.online_policy_min_neighbor_separation_days == 21
    assert config.reset_book_at_test_start is True
    assert config.max_news_per_symbol == 0
    assert config.data_sources.news_sources == []


def test_legacy_preset_remains_available_unchanged():
    config = local_gemma_aapl_config()

    assert config.run_preset == "local_gemma_aapl_full"
    assert config.single_stock_action_space == "trinary_all_in"
    assert config.outcome_learning_mode == "llm_reflection_lessons"
    assert config.allow_short is True


def test_online_preset_rejects_unmodelled_commissions():
    config = local_gemma_aapl_online_config(commission_per_trade=1.0)

    with pytest.raises(ValueError, match="supports slippage costs only"):
        validate_no_paid_api_mode(config, local_gemma_secret_config())


def test_day_limit_override_is_explicit_and_non_mutating():
    original = local_gemma_aapl_online_config()
    bounded = local_gemma_loop._with_day_limits(original, max_train_days=10, max_test_days=5)

    assert original.max_train_days == 0
    assert original.max_test_days == 0
    assert bounded.max_train_days == 10
    assert bounded.max_test_days == 5
    with pytest.raises(ValueError, match="cannot be negative"):
        local_gemma_loop._with_day_limits(original, max_train_days=None, max_test_days=-1)


def test_live_preset_requires_and_sets_a_durable_stream():
    assert local_gemma_aapl_live_config().memory_online_stream_id == "aapl-live-v1"
    assert local_gemma_aapl_live_config(stream_id="paper-a").memory_online_stream_id == "paper-a"
    with pytest.raises(ValueError, match="stream_id"):
        local_gemma_aapl_live_config(stream_id="")


def test_preflight_only_never_checks_or_calls_ollama(monkeypatch):
    monkeypatch.setattr(local_gemma_loop, "BenchmarkStore", lambda: object())
    monkeypatch.setattr(local_gemma_loop, "_preflight", lambda config, secrets, store: {"status": "pass", "checks": []})
    monkeypatch.setattr(
        local_gemma_loop,
        "ensure_ollama_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Ollama must not be checked")),
    )
    monkeypatch.setattr(
        local_gemma_loop,
        "run_local_json_smoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Gemma must not be called")),
    )

    result = local_gemma_loop.run_loop(preflight_only=True, preset="aapl-online")

    assert result["status"] == "preflight_pass"
    assert result["model_status"] == "not_checked"
    assert result["smoke"] == "not_run"


def test_online_loop_runs_once_and_never_auto_patches(monkeypatch):
    monkeypatch.setattr(local_gemma_loop, "BenchmarkStore", lambda: object())
    monkeypatch.setattr(local_gemma_loop, "ensure_ollama_model", lambda *args, **kwargs: {"status": "present"})
    monkeypatch.setattr(local_gemma_loop, "run_local_json_smoke", lambda *args, **kwargs: {"ok": True})
    calls = []

    def fake_iteration(**kwargs):
        calls.append(kwargs["iteration"])
        return {"evaluation": {"success": False}, "diagnostics": {}}

    monkeypatch.setattr(local_gemma_loop, "run_iteration", fake_iteration)
    monkeypatch.setattr(
        local_gemma_loop,
        "propose_allowlisted_patch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("online preset must not auto-patch")),
    )

    result = local_gemma_loop.run_loop(max_iterations=9, preset="aapl-online")

    assert result["status"] == "target_not_met"
    assert calls == [1]


def test_preflight_estimate_accounts_for_weekly_base_cadence(monkeypatch):
    config = local_gemma_aapl_online_config()
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    monkeypatch.setattr(quality, "_trading_dates", lambda *args, **kwargs: dates)
    for name in (
        "_price_coverage_check",
        "_context_coverage_check",
        "_macro_check",
        "_news_check",
        "_fundamental_check",
        "_memory_check",
    ):
        monkeypatch.setattr(quality, name, lambda *args, **kwargs: None)

    report = quality.build_preflight_report(config, local_gemma_secret_config(), object())

    assert config.run_preset in quality.OFFICIAL_PRESETS
    assert report["estimate"]["decision_cadence"] == "weekly_event"
    assert report["estimate"]["scheduled_decision_days"] == 2
    assert report["estimate"]["estimated_model_calls"] == 4
    assert report["estimate"]["estimated_model_calls_upper_bound"] == 8
    assert report["estimate"]["online_lessons_require_model_calls"] is False
