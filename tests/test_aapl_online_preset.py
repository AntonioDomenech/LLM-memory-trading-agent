from __future__ import annotations

import pytest

from agent_benchmark import local_gemma_loop, quality
from agent_benchmark.benchmark_engine import BenchmarkEngine
from agent_benchmark.local_provider import (
    local_gemma_aapl_config,
    local_gemma_aapl_causal_replay_config,
    local_gemma_aapl_live_config,
    local_gemma_aapl_online_config,
    local_gemma_secret_config,
    validate_no_paid_api_mode,
)


def test_primary_preset_is_local_chronological_and_frozen_after_2023():
    config = local_gemma_aapl_online_config()

    validate_no_paid_api_mode(config, local_gemma_secret_config())
    assert config.run_preset == "local_gemma_aapl_online"
    assert config.memory_mode == "deterministic_market_cases"
    assert config.memory_retrieval == "structured"
    assert config.train_end == "2023-12-31"
    assert config.selection_cutoff == "2023-12-31"
    assert config.test_start == "2024-01-01"
    assert config.test_end == "2026-07-09"
    assert config.memory_base_snapshot_id == "aapl-2000-2023-adjusted-v2"
    assert config.memory_online_stream_id == ""
    assert config.outcome_learning_mode == "counterfactual_online"
    assert config.evaluation_mode == "frozen_holdout"
    assert config.globally_pristine is False
    assert config.historical_holdout_reveal_count_lower_bound >= 10
    assert config.online_test_learning is False
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


def test_causal_replay_is_separate_and_cannot_share_frozen_online_state():
    frozen = local_gemma_aapl_online_config()
    replay = local_gemma_aapl_causal_replay_config()

    validate_no_paid_api_mode(replay, local_gemma_secret_config())
    assert replay.train_end == frozen.train_end == "2023-12-31"
    assert replay.memory_base_snapshot_id == frozen.memory_base_snapshot_id
    assert replay.memory_namespace != frozen.memory_namespace
    assert replay.evaluation_mode == "causal_online_replay"
    assert replay.online_test_learning is True


def test_frozen_contract_rejects_overlap_learning_and_partial_holdout():
    secrets = local_gemma_secret_config()
    with pytest.raises(ValueError, match="strictly before"):
        validate_no_paid_api_mode(
            local_gemma_aapl_online_config(train_end="2025-01-02", selection_cutoff="2025-01-02"),
            secrets,
        )
    with pytest.raises(ValueError, match="forbids learning"):
        validate_no_paid_api_mode(
            local_gemma_aapl_online_config(online_test_learning=True),
            secrets,
        )
    with pytest.raises(ValueError, match="partial test-day smoke"):
        validate_no_paid_api_mode(
            local_gemma_aapl_online_config(max_test_days=5),
            secrets,
        )
    with pytest.raises(ValueError, match="requires deterministic_market_cases"):
        validate_no_paid_api_mode(
            local_gemma_aapl_online_config(memory_mode="model_specific_cases_and_lessons"),
            secrets,
        )


def test_resume_call_count_excludes_quantitative_and_cadence_decisions():
    engine = object.__new__(BenchmarkEngine)
    run = {
        "decisions": [
            {"output": {"_api_status": "quantitative_policy"}},
            {"output": {"_api_status": "cadence_hold"}},
            {"output": {"_api_status": "ok"}},
        ]
    }

    assert engine._resume_model_call_count(object(), "run-a", run) == 1


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
    assert local_gemma_aapl_live_config().evaluation_mode == "live_learning"
    assert local_gemma_aapl_live_config().online_test_learning is True
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


@pytest.mark.parametrize("preset", ["aapl-online", "aapl-causal-replay"])
def test_historical_quantitative_loop_runs_once_without_model_smoke_or_auto_patch(monkeypatch, preset):
    monkeypatch.setattr(local_gemma_loop, "BenchmarkStore", lambda: object())
    monkeypatch.setattr(local_gemma_loop, "ensure_ollama_model", lambda *args, **kwargs: {"status": "present"})
    monkeypatch.setattr(
        local_gemma_loop,
        "run_local_json_smoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("historical Gemma smoke must not run")),
    )
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

    result = local_gemma_loop.run_loop(max_iterations=9, preset=preset)

    assert result["status"] == "target_not_met"
    assert result["smoke"] == {
        "status": "skipped",
        "reason": "no_historical_llm_decision_authority",
    }
    assert calls == [1]


def test_live_learning_still_requires_local_model_smoke():
    assert local_gemma_loop._requires_local_model_smoke(local_gemma_aapl_live_config()) is True


def test_live_loop_still_runs_local_model_smoke(monkeypatch):
    live_config = local_gemma_aapl_live_config()
    monkeypatch.setattr(local_gemma_loop, "_config_for_preset", lambda preset: live_config)
    monkeypatch.setattr(local_gemma_loop, "BenchmarkStore", lambda: object())
    monkeypatch.setattr(
        local_gemma_loop,
        "ensure_ollama_model",
        lambda *args, **kwargs: {"status": "present", "digest": "sha256:live"},
    )
    smoke_calls = []

    def fake_smoke(config, secrets):
        smoke_calls.append(config.evaluation_mode)
        return {"ok": True}

    monkeypatch.setattr(local_gemma_loop, "run_local_json_smoke", fake_smoke)
    monkeypatch.setattr(
        local_gemma_loop,
        "run_iteration",
        lambda **kwargs: {"evaluation": {"success": False}, "diagnostics": {}},
    )

    result = local_gemma_loop.run_loop(max_iterations=1, preset="aapl-live")

    assert result["status"] == "target_not_met"
    assert result["smoke"] == {"ok": True}
    assert smoke_calls == ["live_learning"]


def test_frozen_resume_rejects_a_changed_ollama_digest(monkeypatch):
    config = local_gemma_aapl_online_config(
        local_model_digest="sha256:original",
        implementation_commit="a" * 40,
    )
    existing = {
        "id": "resume-frozen",
        "status": "running",
        "config": config.model_dump() if hasattr(config, "model_dump") else config.dict(),
    }

    class FakeStore:
        def get_benchmark_run(self, run_id):
            return existing if run_id == "resume-frozen" else None

    monkeypatch.setattr(local_gemma_loop, "BenchmarkStore", FakeStore)
    monkeypatch.setattr(
        local_gemma_loop,
        "ensure_ollama_model",
        lambda *args, **kwargs: {"status": "present", "digest": "sha256:changed"},
    )

    with pytest.raises(RuntimeError, match="Ollama model digest differs"):
        local_gemma_loop.run_loop(
            resume_run_id="resume-frozen",
            pull_model=False,
            commit_before_run=False,
        )


def test_preflight_estimate_accounts_for_weekly_base_cadence(monkeypatch):
    config = local_gemma_aapl_online_config()
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    monkeypatch.setattr(quality, "_trading_dates", lambda *args, **kwargs: dates)
    monkeypatch.setattr(quality, "_calendar_coverage_end", lambda *args, **kwargs: config.test_end)
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
    assert report["estimate"]["decision_calls_per_day"] == 0
    assert report["estimate"]["estimated_model_calls"] == 0
    assert report["estimate"]["estimated_model_calls_upper_bound"] == 0
    assert report["estimate"]["historical_decision_authority"] == "precutoff_quantitative_policy"
    assert report["estimate"]["event_trigger_calls_in_estimate"] == "not_applicable_no_historical_llm_authority"
    assert report["estimate"]["online_lessons_require_model_calls"] is False


def test_engine_estimate_reports_zero_historical_calls_but_not_live(monkeypatch):
    engine = object.__new__(BenchmarkEngine)
    pairs = [
        {"decision_date": "2025-01-03", "fill_date": "2025-01-06"},
        {"decision_date": "2025-01-06", "fill_date": "2025-01-07"},
    ]
    monkeypatch.setattr(engine, "_symbols", lambda config: ["AAPL"])
    monkeypatch.setattr(engine, "_trading_pairs", lambda *args, **kwargs: pairs)
    monkeypatch.setattr(engine, "_should_run_exposure_critic", lambda config: False)

    frozen = engine.estimate(local_gemma_aapl_online_config())
    replay = engine.estimate(local_gemma_aapl_causal_replay_config())
    live = engine.estimate(local_gemma_aapl_live_config())

    for historical in (frozen, replay):
        assert historical["decision_calls_per_day"] == 0
        assert historical["estimated_decision_calls"] == 0
        assert historical["estimated_decision_calls_upper_bound"] == 0
        assert historical["estimated_lesson_calls_upper_bound"] == 0
    assert live["decision_calls_per_day"] == 2
    assert live["estimated_decision_calls"] == 4
