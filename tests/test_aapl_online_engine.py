from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

from agent_benchmark.benchmark_engine import BenchmarkEngine
from agent_benchmark.llm_client import _ollama_schema_for_namespace
from agent_benchmark.local_provider import (
    local_gemma_aapl_causal_replay_config,
    local_gemma_aapl_live_config,
    local_gemma_aapl_online_config,
)
from agent_benchmark.memory import HybridMemory
from agent_benchmark.online_policy import (
    CalibratedOnlineRiskOffEstimator,
    PointInTimeSnapshot,
    RiskOffEstimatorConfig,
)
from agent_benchmark.prompting import build_stage1_prompt, build_stage2_prompt
from agent_benchmark.schemas import PortfolioBook, SecretConfig
from agent_benchmark.storage import BenchmarkStore
from agent_benchmark.warehouse.store import Warehouse


def _manager(current: float = 1.0, recommendation: str = "HOLD"):
    return {
        "mode": "single_stock",
        "decision_date": "2025-01-16",
        "target_exposure_symbol": "AAPL",
        "current_position_weights": {"AAPL": current} if current else {},
        "online_policy": {
            "recommended_action": recommendation,
            "cash_outperformance_probability": 0.50 if recommendation == "HOLD" else 0.75,
            "expected_active_return": 0.0 if recommendation == "HOLD" else 0.03,
            "lower_bound": -0.01 if recommendation == "HOLD" else 0.01,
            "confidence": 0.2 if recommendation == "HOLD" else 0.8,
        },
        "single_stock_contract": {"action_space": "long_cash_hold"},
        "valid_target_exposure_range": {
            "allowed_target_exposures": {"CASH_ALL": 0.0, "HOLD": current, "BUY_ALL": 1.0}
        },
    }


def test_long_cash_prompt_and_ollama_contract_exclude_shorts():
    config = local_gemma_aapl_online_config()
    system, _ = build_stage2_prompt(_manager(), [])
    schema = _ollama_schema_for_namespace(config, "stage2")
    live_schema = _ollama_schema_for_namespace(config, "live-stage2-repair")
    live_stage1_schema = _ollama_schema_for_namespace(config, "live-stage1")

    assert "CASH_ALL" in system
    assert "BUY_ALL" in system
    assert "HOLD" in system
    assert "Shorting and partial sizing are not allowed" in system
    assert schema["properties"]["action"]["enum"] == ["CASH_ALL", "HOLD", "BUY_ALL"]
    assert live_schema["properties"]["action"]["enum"] == ["CASH_ALL", "HOLD", "BUY_ALL"]
    assert live_stage1_schema["properties"]["analyses"]["type"] == "array"
    assert "SHORT_ALL" not in schema["properties"]["action"]["enum"]


def test_compact_gemma_prompts_reserve_mature_online_lessons():
    engine = BenchmarkEngine()
    config = local_gemma_aapl_online_config()
    historical = [
        {
            "id": f"det-{index}",
            "memory_type": "deterministic_market_case",
            "symbol": "AAPL",
            "content": f"historical case {index}",
        }
        for index in range(8)
    ]
    online = {
        "id": "online-1",
        "memory_type": "counterfactual_online_lesson",
        "symbol": "AAPL",
        "decision_timestamp": "2025-01-02",
        "knowledge_timestamp": "2025-01-31",
        "content": "mature online lesson: cash active return was positive",
        "state_features": {"return_20d": -0.12, "volatility_20d": 0.42},
        "counterfactual_outcomes": {
            "cash_active_return": 0.08,
            "LONG": {"gross_return": -0.08, "net_return": -0.08, "total_cost": 0.0},
            "CASH": {"gross_return": 0.0, "net_return": -0.001, "total_cost": 0.001},
        },
    }
    bundle = {
        "schema_version": "benchmark-input-v2",
        "mode": "single_stock",
        "run_id": "prompt-test",
        "phase": "test",
        "decision_date": "2025-02-03",
        "fill_date": "2025-02-04",
        "candidate_universe": [{"symbol": "AAPL", "sector": "Technology"}],
        "portfolio_state": {
            "cash": 0.0,
            "positions": {"AAPL": 10.0},
            "equity": 1000.0,
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
        },
        "market_snapshots": {"AAPL": {"close": 100.0, "return_20d": 0.01}},
        "memory": [*historical, online],
        "recent_online_lessons": [online],
        "online_policy": {"recommended_action": "HOLD"},
        "decision_support": {},
        "benchmark_rules": {},
    }
    try:
        stage1 = engine._stage1_bundle(bundle, ["AAPL"])
        stage2 = engine._stage2_bundle(bundle, config)
        stage1_system, stage1_user = build_stage1_prompt(stage1, ["AAPL"])
        stage2_system, stage2_user = build_stage2_prompt(stage2, [])
    finally:
        engine.close()

    assert [item["id"] for item in stage1["recent_online_lessons"]] == ["online-1"]
    assert [item["id"] for item in stage2["recent_online_lessons"]] == ["online-1"]
    assert stage1["recent_online_lessons"][0]["state_features"]["return_20d"] == pytest.approx(-0.12)
    assert stage2["recent_online_lessons"][0]["state_features"]["volatility_20d"] == pytest.approx(0.42)
    assert stage2["recent_online_lessons"][0]["counterfactual_outcomes"]["cash_active_return"] == pytest.approx(0.08)
    assert all(item["memory_type"] == "deterministic_market_case" for item in stage1["memory"])
    assert "recent_online_lessons" in stage1_system
    assert '"recent_online_lessons"' in stage1_user
    assert '"return_20d": -0.12' in stage1_user
    assert "recent_online_lessons" in stage2_system
    assert '"recent_online_lessons"' in stage2_user
    assert '"volatility_20d": 0.42' in stage2_user


def test_live_online_run_rejects_an_ephemeral_stream_before_market_access(tmp_path):
    engine = BenchmarkEngine(Warehouse(tmp_path / "warehouse.duckdb"))
    try:
        with pytest.raises(ValueError, match="stable memory_online_stream_id"):
            engine.run_live_snapshot(
                run_id="snapshot-1",
                config=local_gemma_aapl_online_config(),
                secrets=SecretConfig(),
                store=BenchmarkStore(tmp_path / "benchmark.db"),
                dry_run=True,
            )
    finally:
        engine.close()


def test_live_learning_requires_frozen_code_and_model_identity(tmp_path):
    engine = BenchmarkEngine(Warehouse(tmp_path / "warehouse.duckdb"))
    try:
        with pytest.raises(ValueError, match="local_model_digest and implementation_commit"):
            engine.run_live_snapshot(
                run_id="snapshot-identity",
                config=local_gemma_aapl_live_config(stream_id="live-identity"),
                secrets=SecretConfig(),
                store=BenchmarkStore(tmp_path / "benchmark.db"),
                dry_run=False,
            )
    finally:
        engine.close()


def test_live_learning_verifies_identity_against_current_runtime(tmp_path, monkeypatch):
    engine = BenchmarkEngine(Warehouse(tmp_path / "warehouse.duckdb"))
    config = local_gemma_aapl_live_config(
        stream_id="live-identity",
        local_model_digest="sha256:expected-model",
        implementation_commit="a" * 40,
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_repository_identity",
        lambda: {"commit": "b" * 40, "dirty": False},
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_local_model_digest",
        lambda current: current.local_model_digest,
    )
    try:
        with pytest.raises(ValueError, match="current clean Git HEAD"):
            engine.run_live_snapshot(
                run_id="snapshot-identity-mismatch",
                config=config,
                secrets=SecretConfig(),
                store=BenchmarkStore(tmp_path / "benchmark.db"),
                dry_run=False,
            )
    finally:
        engine.close()


def test_live_stream_rejects_state_from_different_code_or_model_contract(tmp_path, monkeypatch):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    old_config = local_gemma_aapl_live_config(
        stream_id="live-stable-name",
        local_model_digest="sha256:old-model",
        implementation_commit="a" * 40,
    )
    old_memory = HybridMemory(store, old_config, SecretConfig(), run_id="old-snapshot")
    old_memory.save_live_state(
        portfolio={"cash": 500.0, "positions": {"AAPL": 5.0}, "equity": 1000.0},
        schedule_state={},
        last_snapshot_at="2026-07-09T09:36:00-04:00",
    )
    new_config = local_gemma_aapl_live_config(
        stream_id="live-stable-name",
        local_model_digest="sha256:new-model",
        implementation_commit="b" * 40,
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_repository_identity",
        lambda: {"commit": new_config.implementation_commit, "dirty": False},
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_local_model_digest",
        lambda current: current.local_model_digest,
    )
    try:
        with pytest.raises(RuntimeError, match="incompatible model/code contract"):
            engine.run_live_snapshot(
                run_id="new-snapshot",
                config=new_config,
                secrets=SecretConfig(),
                store=store,
                dry_run=False,
                timestamp=datetime(2026, 7, 10, 9, 40, tzinfo=ZoneInfo("America/New_York")),
            )
    finally:
        engine.close()
        warehouse.close()


@pytest.mark.parametrize("save_stale_state", [False, True])
def test_live_stream_fails_closed_after_partial_learning_checkpoint(
    tmp_path,
    monkeypatch,
    save_stale_state,
):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    config = local_gemma_aapl_live_config(
        stream_id="live-partial",
        local_model_digest="sha256:model",
        implementation_commit="a" * 40,
    )
    memory = HybridMemory(store, config, SecretConfig(), run_id="partial-snapshot")
    if save_stale_state:
        memory.save_live_state(
            portfolio={"cash": 1000.0, "positions": {}, "equity": 1000.0},
            schedule_state={},
            last_snapshot_at="2026-07-09T09:36:00-04:00",
        )
    memory.add_pending_experience(
        source_run_id="partial-snapshot",
        portfolio_scope="single_stock",
        symbol="AAPL",
        decision_timestamp="2026-07-10T09:35:00-04:00",
        outcome_available_at="2026-08-07",
        outcome_horizon="20d",
        chosen_action="BUY_ALL",
        state_features={"return_20d": 0.01},
        metadata={
            "entry_timestamp": "2026-07-10T09:36:00-04:00",
            "entry_price": 100.0,
            "horizon_days": 20,
        },
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_repository_identity",
        lambda: {"commit": config.implementation_commit, "dirty": False},
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_local_model_digest",
        lambda current: current.local_model_digest,
    )
    expected = "newer executed experience" if save_stale_state else "no portfolio state"
    try:
        with pytest.raises(RuntimeError, match=expected):
            engine.run_live_snapshot(
                run_id="resume-after-partial",
                config=config,
                secrets=SecretConfig(),
                store=store,
                dry_run=False,
                timestamp=datetime(2026, 7, 10, 9, 40, tzinfo=ZoneInfo("America/New_York")),
            )
    finally:
        engine.close()
        warehouse.close()


def test_live_online_run_rejects_incompatible_intraday_timing(tmp_path):
    engine = BenchmarkEngine(Warehouse(tmp_path / "warehouse.duckdb"))
    try:
        with pytest.raises(ValueError, match="09:30-10:00"):
            engine.run_live_snapshot(
                run_id="snapshot-1",
                config=local_gemma_aapl_live_config(stream_id="live-test"),
                secrets=SecretConfig(),
                store=BenchmarkStore(tmp_path / "benchmark.db"),
                dry_run=True,
                timestamp=datetime(2026, 7, 10, 12, 0, tzinfo=ZoneInfo("America/New_York")),
            )
    finally:
        engine.close()


def test_durable_live_stream_rejects_duplicate_session_snapshot(tmp_path, monkeypatch):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    config = local_gemma_aapl_live_config(
        stream_id="live-once",
        local_model_digest="sha256:test-model",
        implementation_commit="a" * 40,
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_repository_identity",
        lambda: {"commit": config.implementation_commit, "dirty": False},
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_local_model_digest",
        lambda current: current.local_model_digest,
    )
    memory = HybridMemory(store, config, SecretConfig(), run_id="snapshot-1")
    memory.save_live_state(
        portfolio={"cash": 1000.0, "positions": {}, "equity": 1000.0},
        schedule_state={},
        last_snapshot_at="2026-07-10T09:36:00-04:00",
    )
    try:
        with pytest.raises(ValueError, match="already completed"):
            engine.run_live_snapshot(
                run_id="snapshot-2",
                config=config,
                secrets=SecretConfig(),
                store=store,
                dry_run=False,
                timestamp=datetime(2026, 7, 10, 9, 40, tzinfo=ZoneInfo("America/New_York")),
            )
    finally:
        engine.close()
        warehouse.close()


def test_durable_live_stream_fails_closed_on_malformed_portfolio_state(tmp_path, monkeypatch):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    config = local_gemma_aapl_live_config(
        stream_id="live-corrupt",
        local_model_digest="sha256:test-model",
        implementation_commit="a" * 40,
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_repository_identity",
        lambda: {"commit": config.implementation_commit, "dirty": False},
    )
    monkeypatch.setattr(
        "agent_benchmark.benchmark_engine.resolve_local_model_digest",
        lambda current: current.local_model_digest,
    )
    memory = HybridMemory(store, config, SecretConfig(), run_id="snapshot-1")
    memory.save_live_state(
        portfolio={"cash": "not-a-number", "positions": {"AAPL": "bad"}, "equity": 1000.0},
        schedule_state={},
        last_snapshot_at="2026-07-09T09:36:00-04:00",
    )
    try:
        with pytest.raises(RuntimeError, match="refusing to restart from cash"):
            engine.run_live_snapshot(
                run_id="snapshot-2",
                config=config,
                secrets=SecretConfig(),
                store=store,
                dry_run=False,
                timestamp=datetime(2026, 7, 10, 9, 40, tzinfo=ZoneInfo("America/New_York")),
            )
    finally:
        engine.close()
        warehouse.close()


def test_cash_gate_blocks_weak_risk_off_but_allows_calibrated_cash():
    engine = BenchmarkEngine()
    config = local_gemma_aapl_online_config()
    try:
        weak = engine._apply_online_policy_gate(config, {"action": "CASH_ALL"}, _manager())
        strong = engine._apply_online_policy_gate(
            config,
            {"action": "CASH_ALL"},
            _manager(recommendation="CASH_ALL"),
        )
        weak_hold_in_cash = engine._apply_online_policy_gate(
            config,
            {"action": "HOLD"},
            _manager(current=0.0),
        )
        weak_cash_in_cash = engine._apply_online_policy_gate(
            config,
            {"action": "CASH_ALL"},
            _manager(current=0.0),
        )
    finally:
        engine.close()

    assert weak["requested_action"] == "CASH_ALL"
    assert weak["action"] == "HOLD"
    assert weak["online_policy_gate"]["blocked_action"] == "CASH_ALL"
    assert strong["action"] == "CASH_ALL"
    assert weak_hold_in_cash["action"] == "BUY_ALL"
    assert weak_hold_in_cash["online_policy_gate"]["blocked_action"] == "HOLD_IN_CASH"
    assert weak_cash_in_cash["action"] == "BUY_ALL"
    assert weak_cash_in_cash["online_policy_gate"]["blocked_action"] == "CASH_ALL"


def test_policy_gate_forced_long_baseline_bypasses_hysteresis_on_cadence_hold():
    engine = BenchmarkEngine()
    config = local_gemma_aapl_online_config()
    manager = _manager(current=0.0)
    schedule = {"call_model": False, "reason": "cadence_hold"}
    prior_cash_trade = {
        "phase": "test",
        "decision_date": "2025-01-02",
        "fill_date": "2025-01-03",
        "decision_kind": "model_decision",
        "stage2_output": {"policy_candidate_action": "CASH_ALL"},
        "execution": {"trades": [{"side": "SELL"}]},
    }
    try:
        cadence = engine._cadence_hold_output(["AAPL"], manager, schedule)
        gated = engine._apply_online_policy_gate(config, cadence, manager)
        resolved = engine._apply_action_hysteresis(
            config,
            "test",
            gated,
            manager,
            [prior_cash_trade],
        )
    finally:
        engine.close()

    assert gated["_policy_gate_forced_baseline"] is True
    assert resolved["action"] == "BUY_ALL"
    assert resolved["executed_action"] == "BUY_ALL"
    assert resolved["hysteresis"]["status"] == "policy_gate_forced_baseline"


def test_weekly_event_schedule_calls_on_phase_start_week_change_and_threshold_crossing():
    engine = BenchmarkEngine()
    config = local_gemma_aapl_online_config()
    try:
        first_bundle = {
            "decision_date": "2025-01-06",
            "market_snapshots": {"AAPL": {"drawdown_60d": -0.03, "volatility_20d": 0.20}},
        }
        first = engine._decision_schedule(config, "test", first_bundle, [])
        history = [
            {
                "phase": "test",
                "decision_date": "2025-01-06",
                "decision_kind": "model_decision",
                "stage2_output": {"_schedule_state": first},
            }
        ]
        ordinary = engine._decision_schedule(
            config,
            "test",
            {
                "decision_date": "2025-01-07",
                "market_snapshots": {"AAPL": {"drawdown_60d": -0.04, "volatility_20d": 0.22}},
            },
            history,
        )
        event = engine._decision_schedule(
            config,
            "test",
            {
                "decision_date": "2025-01-08",
                "market_snapshots": {"AAPL": {"drawdown_60d": -0.09, "volatility_20d": 0.22}},
            },
            [
                *history,
                {
                    "phase": "test",
                    "decision_date": "2025-01-07",
                    "decision_kind": "cadence_hold",
                    "stage2_output": {"_schedule_state": ordinary},
                },
            ],
        )
        next_week = engine._decision_schedule(
            config,
            "test",
            {
                "decision_date": "2025-01-13",
                "market_snapshots": {"AAPL": {"drawdown_60d": -0.04, "volatility_20d": 0.22}},
            },
            history,
        )
    finally:
        engine.close()

    assert first["call_model"] and first["reason"] == "phase_start"
    assert not ordinary["call_model"] and ordinary["reason"] == "cadence_hold"
    assert event["call_model"] and event["event_reasons"] == ["drawdown_threshold_crossed"]
    assert next_week["call_model"] and next_week["reason"] == "weekly_boundary"


def test_hysteresis_requires_second_confirmation_after_minimum_hold():
    engine = BenchmarkEngine()
    config = local_gemma_aapl_online_config()
    initial_buy = {
        "phase": "test",
        "decision_date": "2025-01-02",
        "fill_date": "2025-01-03",
        "decision_kind": "model_decision",
        "stage2_output": {"policy_candidate_action": "BUY_ALL"},
        "execution": {"trades": [{"side": "BUY"}]},
    }
    first_cash_request = {
        "phase": "test",
        "decision_date": "2025-01-09",
        "fill_date": "2025-01-10",
        "decision_kind": "model_decision",
        "stage2_output": {"policy_candidate_action": "CASH_ALL", "executed_action": "HOLD"},
        "execution": {"trades": []},
    }
    try:
        deferred = engine._apply_action_hysteresis(
            config,
            "test",
            {"action": "CASH_ALL", "requested_action": "CASH_ALL"},
            _manager(),
            [initial_buy],
        )
        confirmed = engine._apply_action_hysteresis(
            config,
            "test",
            {"action": "CASH_ALL", "requested_action": "CASH_ALL"},
            _manager(),
            [initial_buy, first_cash_request],
        )
    finally:
        engine.close()

    assert deferred["action"] == "HOLD"
    assert deferred["hysteresis"]["status"] == "deferred"
    assert confirmed["action"] == "CASH_ALL"
    assert confirmed["executed_action"] == "CASH_ALL"
    assert confirmed["hysteresis"]["confirmations"] == 2


def test_pending_2025_experience_becomes_usable_only_after_maturity(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    dates = pd.bdate_range("2025-01-02", periods=25).date.astype(str).tolist()
    for index, day in enumerate(dates):
        price = 100.0 if index < 20 else 90.0
        warehouse.conn.execute(
            """
            INSERT INTO context_daily VALUES
                (CAST(? AS DATE), 'SPY', true, true, true, 100, 100, 100, 100, 100, 1000, 0, 'test', '')
            """,
            [day],
        )
        warehouse.conn.execute(
            """
            INSERT INTO asset_daily VALUES
                (CAST(? AS DATE), 'AAPL', true, true, true, ?, ?, ?, ?, ?, 1000, 0, 'test', '')
            """,
            [day, price, price, price, price, price],
        )
    config = local_gemma_aapl_causal_replay_config(
        test_start=dates[0],
        test_end=dates[-1],
        online_policy_min_samples=1,
        online_policy_max_neighbors=5,
    )
    memory = HybridMemory(store, config, SecretConfig(), run_id="replay-a")
    estimator = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(symbol="AAPL", min_samples=1, max_neighbors=5)
    )
    snapshot = PointInTimeSnapshot(
        symbol="AAPL",
        decision_timestamp=dates[0],
        as_of_timestamp=dates[0],
        features={"return_20d": -0.02, "volatility_20d": 0.30},
    )
    try:
        pending_id = engine._queue_online_experience(
            memory,
            config,
            "replay-a",
            "test",
            snapshot,
            dates[0],
            {"AAPL": 100.0},
            {"action": "BUY_ALL", "requested_action": "BUY_ALL", "executed_action": "BUY_ALL"},
            {"trades": [{"side": "BUY"}]},
        )
        assert pending_id is not None
        assert memory.retrieve(decision_timestamp=dates[19]) == []
        matured = engine._mature_due_online_experiences(
            memory,
            config,
            "replay-a",
            dates[20],
            estimator,
        )
        learned = memory.retrieve(decision_timestamp=dates[20])
    finally:
        engine.close()
        warehouse.close()

    assert matured == 1
    assert len(estimator.lessons) == 1
    assert learned[0]["memory_type"] == "counterfactual_online_lesson"
    assert learned[0]["knowledge_timestamp"] == dates[20]
    assert learned[0]["counterfactual_outcomes"]["cash_active_return"] > 0
    assert memory.due_pending_experiences(as_of=dates[-1]) == []


def test_frozen_holdout_fails_closed_if_learning_path_is_called(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(Warehouse(tmp_path / "warehouse.duckdb"))
    config = local_gemma_aapl_online_config()
    memory = HybridMemory(store, config, SecretConfig(), run_id="frozen-a")
    estimator = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(symbol="AAPL", min_samples=1, max_neighbors=5)
    )
    snapshot = PointInTimeSnapshot(
        symbol="AAPL",
        decision_timestamp="2025-01-02",
        as_of_timestamp="2025-01-02",
        features={"return_20d": 0.0},
    )
    try:
        with pytest.raises(RuntimeError, match="forbids queuing"):
            engine._queue_online_experience(
                memory,
                config,
                "frozen-a",
                "test",
                snapshot,
                "2025-01-03",
                {"AAPL": 100.0},
                {"action": "BUY_ALL"},
            )
        with pytest.raises(RuntimeError, match="forbids maturing"):
            engine._mature_due_online_experiences(
                memory,
                config,
                "frozen-a",
                "2025-02-03",
                estimator,
                phase="test",
            )
    finally:
        engine.close()


def test_adjusted_open_buy_hold_and_phase_start_equity_use_same_contract(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    engine = BenchmarkEngine(warehouse)
    try:
        warehouse.conn.execute(
            """
            INSERT INTO asset_daily VALUES
                ('2025-01-02', 'AAPL', true, true, true, 100, 100, 100, 100, 90, 1000, 0, 'test', ''),
                ('2025-01-31', 'AAPL', true, true, true, 100, 100, 100, 100, 100, 1000, 0, 'test', '')
            """
        )
        benchmark = engine._instrument_buy_hold(
            table="asset_daily",
            symbol="AAPL",
            label="AAPL buy & hold",
            benchmark_id="single_stock",
            start_date="2025-01-02",
            end_date="2025-01-31",
            initial=1000.0,
            ai_return=0.0,
            price_basis="adjusted",
        )
        window = engine._window_metrics(
            [
                {"date": "2025-01-02", "equity": 999.5, "phase_start_equity": 1000.0},
                {"date": "2025-01-31", "equity": 1100.0},
            ]
        )
    finally:
        engine.close()
        warehouse.close()

    assert benchmark is not None
    assert benchmark["price_basis"] == "adjusted"
    assert benchmark["total_return"] == pytest.approx(100.0 / 90.0 - 1.0)
    assert window["initial_equity"] == 1000.0
    assert window["total_return"] == pytest.approx(0.10)


def test_frozen_holdout_calendar_reports_use_one_continuous_account(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    engine = BenchmarkEngine(warehouse)
    rows = [
        ("2024-01-02", 100.0),
        ("2024-12-31", 110.0),
        ("2025-01-02", 111.0),
        ("2025-12-31", 121.0),
        ("2026-01-02", 122.0),
        ("2026-07-09", 127.0),
    ]
    try:
        for day, price in rows:
            warehouse.conn.execute(
                """
                INSERT INTO asset_daily VALUES
                    (CAST(? AS DATE), 'AAPL', true, true, true, ?, ?, ?, ?, ?, 1000, 0, 'test', '')
                """,
                [day, price, price, price, price, price],
            )
        reports = engine._continuous_single_stock_period_reports(
            local_gemma_aapl_online_config(slippage_bps=0.0),
            ["AAPL"],
            [
                {"date": day, "phase": "test", "equity": 1000.0 * price / 100.0, **({"phase_start_equity": 1000.0} if index == 0 else {})}
                for index, (day, price) in enumerate(rows)
            ],
        )
    finally:
        engine.close()
        warehouse.close()

    assert list(reports) == ["2024", "2025", "2026_ytd"]
    assert reports["2024"]["strategy_return"] == pytest.approx(0.10)
    assert reports["2025"]["strategy_return"] == pytest.approx(0.10)
    assert reports["2026_ytd"]["strategy_return"] == pytest.approx(127.0 / 121.0 - 1.0)
    assert all(item["accounting"] == "continuous_account_no_calendar_reset" for item in reports.values())


def test_live_lesson_rebases_raw_entry_and_adjusted_exit_together(tmp_path, monkeypatch):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    dates = pd.bdate_range("2025-01-02", periods=22).date.astype(str).tolist()
    for index, day in enumerate(dates):
        raw = 100.0 if index < 20 else 60.0
        adjusted = 50.0 if index < 20 else 60.0
        warehouse.conn.execute(
            """
            INSERT INTO asset_daily VALUES
                (CAST(? AS DATE), 'AAPL', true, true, true, ?, ?, ?, ?, ?, 1000, 0, 'test', '')
            """,
            [day, raw, raw, raw, raw, adjusted],
        )
    config = local_gemma_aapl_live_config(
        stream_id="live-test",
        test_start=dates[0],
        test_end=dates[-1],
        online_policy_min_samples=1,
        online_policy_max_neighbors=5,
    )
    memory = HybridMemory(store, config, SecretConfig(), run_id="snapshot-1")
    estimator = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(symbol="AAPL", min_samples=1, max_neighbors=5)
    )
    snapshot = PointInTimeSnapshot(
        symbol="AAPL",
        decision_timestamp=dates[0],
        as_of_timestamp=dates[0],
        features={"return_20d": 0.0},
    )
    yahoo = pd.DataFrame(
        {
            "Open": [100.0] * 20 + [60.0, 60.0],
            "High": [100.0] * 20 + [60.0, 60.0],
            "Low": [100.0] * 20 + [60.0, 60.0],
            "Close": [100.0] * 20 + [60.0, 60.0],
            "Adj Close": [50.0] * 20 + [60.0, 60.0],
            "Stock Splits": [0.0] * 22,
            "Dividends": [0.0] * 22,
        },
        index=pd.to_datetime(dates),
    )
    download_kwargs = {}

    def fake_download(*args, **kwargs):
        download_kwargs.update(kwargs)
        return yahoo

    monkeypatch.setattr("yfinance.download", fake_download)
    try:
        memory.add_pending_experience(
            source_run_id="snapshot-1",
            portfolio_scope="single_stock",
            symbol="AAPL",
            decision_timestamp=dates[0],
            outcome_available_at=dates[20],
            outcome_horizon="20d",
            chosen_action="BUY_ALL",
            state_features=dict(snapshot.features),
            metadata={
                "entry_timestamp": dates[0],
                "entry_price": 100.0,
                "entry_price_basis": "raw_live_quote",
                "horizon_days": 20,
            },
        )
        assert engine._mature_due_online_experiences(
            memory,
            config,
            "snapshot-2",
            f"{dates[20]}T15:00:00Z",
            estimator,
        ) == 1
        assert memory.retrieve(decision_timestamp=f"{dates[20]}T14:59:59Z") == []
        assert memory.retrieve(decision_timestamp=f"{dates[20]}T15:00:00Z")
    finally:
        engine.close()
        warehouse.close()

    lesson = estimator.lessons[0]
    assert lesson.entry_price == pytest.approx(50.0)
    assert lesson.exit_price == pytest.approx(60.0)
    assert lesson.outcomes.long.gross_return == pytest.approx(0.20)
    assert download_kwargs["actions"] is True


def test_live_lesson_rebases_raw_entry_across_intervening_split(tmp_path, monkeypatch):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    dates = pd.bdate_range("2025-01-02", periods=22).date.astype(str).tolist()
    yahoo = pd.DataFrame(
        {
            "Open": [100.0] * 20 + [110.0, 111.0],
            "High": [100.0] * 20 + [110.0, 111.0],
            "Low": [100.0] * 20 + [110.0, 111.0],
            "Close": [100.0] * 20 + [110.0, 111.0],
            "Adj Close": [100.0] * 20 + [110.0, 111.0],
            "Stock Splits": [0.0, 4.0] + [0.0] * 20,
            "Dividends": [0.0] * 22,
        },
        index=pd.to_datetime(dates),
    )
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: yahoo)
    config = local_gemma_aapl_live_config(
        stream_id="live-split-test",
        online_policy_min_samples=1,
        online_policy_max_neighbors=5,
    )
    memory = HybridMemory(store, config, SecretConfig(), run_id="snapshot-1")
    estimator = CalibratedOnlineRiskOffEstimator(
        RiskOffEstimatorConfig(symbol="AAPL", min_samples=1, max_neighbors=5)
    )
    try:
        memory.add_pending_experience(
            source_run_id="snapshot-1",
            portfolio_scope="single_stock",
            symbol="AAPL",
            decision_timestamp=dates[0],
            outcome_available_at=dates[20],
            outcome_horizon="20d",
            chosen_action="BUY_ALL",
            state_features={"return_20d": 0.0},
            metadata={
                "entry_timestamp": f"{dates[0]}T09:35:00-05:00",
                "entry_price": 400.0,
                "entry_price_basis": "raw_live_quote",
                "horizon_days": 20,
            },
        )
        matured = engine._mature_due_online_experiences(
            memory,
            config,
            "snapshot-2",
            f"{dates[20]}T15:00:00Z",
            estimator,
        )
        learned = memory.retrieve(decision_timestamp=f"{dates[20]}T15:00:00Z")
    finally:
        engine.close()
        warehouse.close()

    assert matured == 1
    assert estimator.lessons[0].entry_price == pytest.approx(100.0)
    assert estimator.lessons[0].exit_price == pytest.approx(110.0)
    assert estimator.lessons[0].outcomes.long.gross_return == pytest.approx(0.10)
    assert learned[0]["metadata"]["cumulative_split_factor"] == pytest.approx(4.0)
    assert learned[0]["metadata"]["entry_rebase_factor"] == pytest.approx(0.25)


def test_live_book_credits_dividends_and_splits_once(monkeypatch):
    class FakeTicker:
        def history(self, **kwargs):
            return pd.DataFrame(
                {"Dividends": [1.0], "Stock Splits": [2.0]},
                index=pd.to_datetime(["2025-01-02"]),
            )

    monkeypatch.setattr("yfinance.Ticker", lambda symbol: FakeTicker())
    engine = BenchmarkEngine()
    book = PortfolioBook(cash=0.0, positions={"AAPL": 2.0}, equity=200.0)
    try:
        updated, events = engine._apply_live_corporate_actions(
            book,
            {"last_snapshot_at": "2025-01-01T14:35:00Z"},
            "2025-01-03T14:35:00Z",
        )
    finally:
        engine.close()

    assert updated.cash == pytest.approx(2.0)
    assert updated.positions["AAPL"] == pytest.approx(4.0)
    assert {item["status"] for item in events} == {"dividend_credited", "split_applied"}


def test_live_policy_features_use_last_completed_adjusted_daily_state():
    engine = BenchmarkEngine()
    index = pd.bdate_range("2024-01-02", periods=300)
    adjusted = pd.Series(range(100, 400), index=index, dtype=float)
    daily = pd.DataFrame(
        {
            "open": adjusted * 4,
            "high": adjusted * 4,
            "low": adjusted * 4,
            "close": adjusted * 4,
            "adj_close": adjusted,
            "volume": range(1000, 1300),
        },
        index=index,
    )
    intraday = pd.DataFrame(
        {"open": [1600.0], "high": [1601.0], "low": [1599.0], "close": [1600.0], "volume": [10.0]},
        index=pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=14)]),
    )
    try:
        snapshot = engine._snapshot_from_live_frames("AAPL", daily, intraday, adjusted=True)
    finally:
        engine.close()

    assert snapshot["feature_timing"] == "last_completed_daily_close"
    assert snapshot["history_points"] == 299
    assert snapshot["return_252d"] is not None
    assert snapshot["sma200_distance"] is not None
    assert snapshot["drawdown_60d"] == pytest.approx(0.0)
    assert snapshot["close"] == pytest.approx(1592.0)
    assert snapshot["open"] == pytest.approx(1592.0)
    assert snapshot["high"] == pytest.approx(1592.0)
    assert snapshot["low"] == pytest.approx(1592.0)
    assert snapshot["volume"] == pytest.approx(1298.0)
    assert snapshot["as_of_timestamp"] == snapshot["feature_as_of_timestamp"]


def test_live_snapshot_excludes_current_partial_daily_row_without_intraday_data():
    engine = BenchmarkEngine()
    daily = pd.DataFrame(
        {
            "open": [98.0, 100.0, 999.0],
            "high": [101.0, 103.0, 1001.0],
            "low": [97.0, 99.0, 998.0],
            "close": [100.0, 102.0, 1000.0],
            "adj_close": [100.0, 102.0, 1000.0],
            "volume": [10.0, 11.0, 999.0],
        },
        index=pd.to_datetime(["2026-07-08", "2026-07-09", "2026-07-10"]),
    )
    try:
        snapshot = engine._snapshot_from_live_frames(
            "AAPL",
            daily,
            pd.DataFrame(),
            adjusted=True,
            decision_time=datetime(2026, 7, 10, 9, 35, tzinfo=ZoneInfo("America/New_York")),
        )
    finally:
        engine.close()

    assert snapshot["date"] == "2026-07-09"
    assert snapshot["close"] == pytest.approx(102.0)
    assert snapshot["volume"] == pytest.approx(11.0)
    assert snapshot["history_points"] == 2


@pytest.mark.parametrize(
    ("bar_timestamp", "accepted"),
    [
        ("2026-07-10 09:35:00-04:00", True),
        ("2026-07-10 09:34:00-04:00", False),
        ("2026-07-10 09:20:00-04:00", False),
        ("2026-07-09 09:34:00-04:00", False),
        ("2026-07-10 09:37:00-04:00", False),
        ("2026-07-10 00:00:00-04:00", False),
    ],
)
def test_live_execution_quote_must_be_fresh_current_open_session(monkeypatch, bar_timestamp, accepted):
    frame = pd.DataFrame(
        {"Close": [101.25]},
        index=pd.DatetimeIndex([pd.Timestamp(bar_timestamp)]),
    )
    monkeypatch.setattr("yfinance.download", lambda *args, **kwargs: frame)
    engine = BenchmarkEngine()
    try:
        prices, status = engine._live_execution_prices(
            ["AAPL"],
            decision_time=datetime(2026, 7, 10, 9, 35, tzinfo=ZoneInfo("America/New_York")),
            observed_at=datetime(2026, 7, 10, 9, 35, tzinfo=ZoneInfo("America/New_York")),
        )
    finally:
        engine.close()

    assert ("AAPL" in prices) is accepted
    assert status[-1]["status"] == ("ok" if accepted else "stale_or_future")


def test_live_dry_run_orders_quote_after_model_and_does_not_mutate_learning_state(tmp_path, monkeypatch):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    config = local_gemma_aapl_live_config(stream_id="dry-run-live")
    config.data_sources.include_index_context = False
    run_id = "dry-run-snapshot"
    store.create_benchmark_run(run_id, config.model_dump() if hasattr(config, "model_dump") else config.dict())
    events = []
    market = {
        "AAPL": {
            "symbol": "AAPL",
            "date": "2026-07-09",
            "as_of_date": "2026-07-09",
            "as_of_timestamp": "2026-07-09T16:00:00-04:00",
            "feature_as_of_timestamp": "2026-07-09T16:00:00-04:00",
            "source": "test_completed_daily",
            "open": 99.0,
            "high": 101.0,
            "low": 98.0,
            "close": 100.0,
            "adj_close": 100.0,
            "volume": 1000.0,
            "return_20d": 0.01,
            "return_60d": 0.02,
            "volatility_20d": 0.20,
            "volatility_60d": 0.22,
            "sma20_distance": 0.01,
            "sma50_distance": 0.02,
            "sma200_distance": 0.03,
            "drawdown_60d": -0.01,
            "drawdown_252d": -0.02,
            "volume_z20": 0.0,
            "feature_timing": "last_completed_daily_close",
        }
    }

    def fake_model(*args, **kwargs):
        events.append("model")
        output = dict(kwargs.get("fallback") or {})
        output["_api_status"] = "dry_run"
        return output

    def fake_execution_prices(*args, **kwargs):
        events.append("quote")
        return {"AAPL": 101.0}, [
            {
                "symbol": "AAPL",
                "source": "test_intraday_execution",
                "status": "ok",
                "quote_timestamp": "2026-07-10T09:36:00-04:00",
                "observed_at": "2026-07-10T09:36:00-04:00",
            }
        ]

    monkeypatch.setattr("agent_benchmark.benchmark_engine.call_json_model", fake_model)
    monkeypatch.setattr(
        engine,
        "_live_market_snapshots",
        lambda *args, **kwargs: (market, {"AAPL": 100.0}, []),
    )
    monkeypatch.setattr(engine, "_live_execution_prices", fake_execution_prices)
    monkeypatch.setattr(
        engine,
        "_current_ny_time",
        lambda: datetime(2026, 7, 10, 9, 36, tzinfo=ZoneInfo("America/New_York")),
    )
    try:
        result = engine.run_live_snapshot(
            run_id=run_id,
            config=config,
            secrets=SecretConfig(),
            store=store,
            dry_run=True,
            timestamp=datetime(2026, 7, 10, 9, 35, tzinfo=ZoneInfo("America/New_York")),
        )
        with store._connect() as conn:
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "benchmark_memory",
                    "benchmark_pending_experiences",
                    "benchmark_live_state",
                    "benchmark_base_snapshots",
                )
            }
    finally:
        engine.close()
        warehouse.close()

    assert result["summary"]["dry_run"] is True
    assert events[-1] == "quote"
    assert "model" in events
    assert counts == {
        "benchmark_memory": 0,
        "benchmark_pending_experiences": 0,
        "benchmark_live_state": 0,
        "benchmark_base_snapshots": 0,
    }


def test_replay_dry_run_does_not_mutate_learning_state(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = BenchmarkEngine(warehouse)
    config = local_gemma_aapl_online_config(
        test_start="2025-01-01",
        test_end="2025-01-31",
    )
    run_id = "dry-run-replay"
    store.create_benchmark_run(run_id, config.model_dump() if hasattr(config, "model_dump") else config.dict())
    try:
        result = engine.run(
            run_id=run_id,
            config=config,
            secrets=SecretConfig(),
            store=store,
            dry_run=True,
        )
        with store._connect() as conn:
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in (
                    "benchmark_memory",
                    "benchmark_pending_experiences",
                    "benchmark_live_state",
                    "benchmark_base_snapshots",
                )
            }
    finally:
        engine.close()
        warehouse.close()

    assert result["summary"]["dry_run"] is True
    assert result["summary"]["frozen_learning_state_proof"]["unchanged"] is True
    assert result["summary"]["frozen_learning_state_proof"]["test_outcomes_used_for_learning"] is False
    assert counts == {
        "benchmark_memory": 0,
        "benchmark_pending_experiences": 0,
        "benchmark_live_state": 0,
        "benchmark_base_snapshots": 0,
    }
