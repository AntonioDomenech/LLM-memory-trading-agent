from agent_benchmark.memory import HybridMemory
from agent_benchmark.api_usage import estimate_run_api_usage
from agent_benchmark.benchmark_engine import BenchmarkEngine
from agent_benchmark.deterministic_memory import DeterministicMarketMemory
from agent_benchmark.llm_client import _ollama_schema_for_namespace, _post_json, _retry_after_seconds, call_json_model
from agent_benchmark.portfolio import execute_target_weights, initial_book
from agent_benchmark.prompting import build_stage1_prompt, build_stage2_prompt
from agent_benchmark.quality import build_run_diagnostics, canonicalize_fundamentals
from agent_benchmark.schemas import BenchmarkConfig, PortfolioBook, SecretConfig
from agent_benchmark.storage import BenchmarkStore
from agent_benchmark.jobs import BenchmarkJobManager, LiveScheduler
from agent_benchmark.warehouse.store import Warehouse

from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pytest
import pandas as pd


def test_config_defaults_follow_benchmark_contract():
    config = BenchmarkConfig()

    assert config.mode == "single_stock"
    assert config.run_preset == "single_stock_diagnostic"
    assert config.symbol == "AAPL"
    assert config.allow_short is True
    assert config.max_gross_exposure == 1.0
    assert config.decision_process == "two_stage_llm"
    assert config.opportunity_cost_policy == "soft"
    assert config.exposure_critic_enabled is True
    assert config.outcome_learning_mode == "off"
    assert config.turnover_prompt_buffer == 0.02
    assert config.memory_mode == "deterministic_market_cases"
    assert config.memory_retrieval == "deterministic_similarity"
    assert config.max_output_tokens == 900
    assert config.initial_cash == 1000.0
    assert config.strict_preflight is True
    assert config.require_paid_micro_pilot is True
    assert config.max_nonzero_positions == 12
    assert config.max_daily_turnover == 0.20
    assert config.turnover_edge_multiplier == 3.0
    assert config.memory_k_neighbors == 50
    assert config.news_policy == "real_titles_or_aggregate_events"
    assert config.macro_policy == "omit_if_missing"


def test_portfolio_rejects_invalid_gross_exposure_without_improving_model_decision():
    config = BenchmarkConfig(max_gross_exposure=1.0, allow_short=True, slippage_bps=0)
    book = initial_book(1000)

    next_book, execution = execute_target_weights(
        book,
        {"AAPL": 1.0, "MSFT": -1.0},
        {"AAPL": 100.0, "MSFT": 100.0},
        config,
    )

    event_types = [event["type"] for event in execution["events"]]
    assert "invalid_gross_exposure" in event_types
    assert next_book == book
    assert execution["target_weights"]["AAPL"] == 1.0
    assert execution["target_weights"]["MSFT"] == -1.0
    assert execution["executed_target_weights"] == {}
    assert execution["trades"] == []
    assert execution["model_failure"] is True


def test_memory_retrieval_is_point_in_time(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.db")
    config = BenchmarkConfig(model="test-model", embedding_provider="local")
    memory = HybridMemory(store, config, SecretConfig())

    memory.add(
        portfolio_scope="balanced_50_portfolio",
        decision_timestamp="2024-01-02",
        knowledge_timestamp="2024-01-05",
        source_run_id="run",
        memory_type="lesson",
        content="AAPL strength during falling volatility worked.",
    )
    memory.add(
        portfolio_scope="balanced_50_portfolio",
        decision_timestamp="2024-01-02",
        knowledge_timestamp="2024-02-01",
        source_run_id="run",
        memory_type="lesson",
        content="Future lesson that must not leak.",
    )

    retrieved = memory.retrieve(decision_timestamp="2024-01-10", query="AAPL volatility", limit=10)

    assert len(retrieved) == 1
    assert "Future lesson" not in retrieved[0]["content"]


def test_stage_prompts_preserve_model_owned_decision_rule():
    bundle = {
        "decision_date": "2025-01-02",
        "fill_date": "2025-01-03",
        "portfolio_state": {"cash": 1000, "positions": {}, "equity": 1000},
        "benchmark_rules": {"simulator_role": "mechanics_only"},
    }

    stage1_system, _ = build_stage1_prompt(bundle, ["AAPL"])
    stage2_system, _ = build_stage2_prompt(bundle, [{"symbol": "AAPL", "stance": "neutral"}])

    assert "point-in-time" in stage1_system
    assert "You own the final investment decision" in stage2_system
    assert "mechanical constraints" in stage2_system
    assert "sum(abs(target_weights.values()))" in stage2_system
    assert "cash_weight" in stage2_system
    assert "estimated_turnover" in stage2_system
    assert "current_position_weights" in stage2_system
    assert "copy those weights" in stage2_system
    assert "rejects the allocation" in stage2_system
    assert "sparse portfolio" in stage2_system
    assert "decision_support" in stage1_system
    assert "decision_support" in stage2_system
    assert "max_daily_turnover" in stage2_system


def test_stage_prompts_include_json_word_for_responses_json_mode():
    bundle = {
        "decision_date": "2025-01-02",
        "fill_date": "2025-01-03",
        "portfolio_state": {"cash": 1000, "positions": {}, "equity": 1000},
        "benchmark_rules": {"simulator_role": "mechanics_only"},
    }

    _, stage1_user = build_stage1_prompt(bundle, ["AAPL"])
    _, stage2_user = build_stage2_prompt(bundle, [{"symbol": "AAPL", "stance": "neutral"}])

    assert "json" in stage1_user.lower()
    assert "json" in stage2_user.lower()


def test_single_stock_stage2_prompt_uses_target_exposure_contract():
    bundle = {
        "mode": "single_stock",
        "decision_date": "2025-01-02",
        "fill_date": "2025-01-03",
        "target_exposure_symbol": "AAPL",
        "valid_target_exposure_range": {"symbol": "AAPL", "min": -0.08, "max": 0.28},
        "single_stock_opportunity_cost": {
            "policy": "soft",
            "benchmark_hurdle": {
                "comparison": "same_stock_buy_and_hold",
                "test_window": {"start_date": "2025-01-01", "end_date": "2025-12-31"},
            },
        },
        "single_stock_contract": {"action_space": "trinary_all_in", "allowed_actions": ["SHORT_ALL", "HOLD", "BUY_ALL"]},
        "exposure_critic": {"recommended_exposure_band": [0.2, 0.5], "cash_drag_risk": "cash can lag"},
    }

    system, user = build_stage2_prompt(bundle, [{"analyses": [{"symbol": "AAPL", "stance": "bullish"}]}])

    assert "SHORT_ALL" in system
    assert "BUY_ALL" in system
    assert "No partial sizing" in system
    assert "cash_drag_justification" in system
    assert "why_not_buy_hold" in system
    assert "stage1_alignment" in system
    assert "HOLD while short remains short" in system
    assert "Shorts are tactical and high hurdle" in system
    assert "Avoid reactionary shorts" in system
    assert "already happened is not enough" in system
    assert "valid_target_exposure_range" in user
    assert "exposure_critic" in user
    assert "buy-and-hold" in system
    assert "benchmark_hurdle" in user
    assert "single-stock action" in user


def test_single_stock_bundle_carries_official_2025_buy_hold_hurdle(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    engine = BenchmarkEngine(warehouse)
    config = BenchmarkConfig(
        mode="single_stock",
        symbol="AAPL",
        train_start="2000-01-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-31",
        memory_mode="model_specific_cases_and_lessons",
        single_stock_action_space="trinary_all_in",
        allow_short=True,
        max_daily_turnover=2.0,
        turnover_prompt_buffer=0.0,
    )
    store = BenchmarkStore(tmp_path / "benchmark.db")
    memory = HybridMemory(store, config, SecretConfig())
    try:
        bundle = engine._build_bundle(
            config,
            memory,
            "run",
            "test",
            "2025-01-02",
            "2025-01-03",
            ["AAPL"],
            initial_book(config.initial_cash),
        )
        stage2 = engine._stage2_bundle(bundle, config)
    finally:
        engine.close()
        warehouse.close()

    hurdle = stage2["single_stock_opportunity_cost"]["benchmark_hurdle"]
    assert stage2["single_stock_contract"]["model_returns"] == "action"
    assert stage2["single_stock_contract"]["allowed_actions"] == ["SHORT_ALL", "HOLD", "BUY_ALL"]
    assert stage2["trinary_short_hurdle"]["hold_semantics"].startswith("HOLD means no trade")
    assert "high volatility by itself" in stage2["trinary_short_hurdle"]["ordinary_signals_not_enough"]
    assert "already sold off" in stage2["trinary_short_hurdle"]["no_reactionary_shorts"]
    assert stage2["single_stock_shock_guard"]["symbol"] == "AAPL"
    assert "short_open_rule" in stage2["single_stock_shock_guard"]
    assert hurdle["comparison"] == "same_stock_buy_and_hold"
    assert hurdle["train_window"] == {
        "start_date": "2000-01-01",
        "end_date": "2024-12-31",
        "purpose": "point_in_time_memory_only",
    }
    assert hurdle["test_window"] == {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "purpose": "official_success_score",
    }
    assert "buy-and-hold" in hurdle["requirement"]


def test_ollama_schema_requires_action_for_trinary_single_stock_stage2():
    config = BenchmarkConfig(
        mode="single_stock",
        allow_short=True,
        single_stock_action_space="trinary_all_in",
        max_gross_exposure=1.0,
    )

    schema = _ollama_schema_for_namespace(config, "stage2")

    assert "action" in schema["required"]
    assert "target_exposure" not in schema["required"]
    assert schema["properties"]["action"]["enum"] == ["SHORT_ALL", "HOLD", "BUY_ALL"]


def test_stage2_allocation_repair_uses_same_model_before_simulator_rejection(monkeypatch):
    calls = []

    def fake_call_json_model(config, secrets, system, user, **kwargs):
        calls.append({"system": system, "user": user, "kwargs": kwargs})
        return {
            "target_weights": {"AAPL": 0.6, "MSFT": -0.4},
            "cash_weight": 0.0,
            "gross_exposure": 1.0,
            "net_exposure": 0.2,
            "confidence": 0.4,
            "portfolio_thesis": "Repaired by model.",
            "major_risks": [],
            "uncertainty": [],
            "expected_return_bps": 100,
            "horizon_days": 20,
            "expected_holding_days": 20,
            "estimated_turnover": 1.0,
            "estimated_slippage_cost_bps": 5.0,
            "rebalance_reason": "test repair",
            "input_evidence_refs": ["test"],
            "data_quality_warnings_used": [],
            "_api_status": "ok",
        }

    monkeypatch.setattr("agent_benchmark.benchmark_engine.call_json_model", fake_call_json_model)
    engine = BenchmarkEngine()
    try:
        repaired, repair_calls = engine._repair_stage2_allocation_if_needed(
            BenchmarkConfig(mode="balanced_50_portfolio", run_preset="balanced_50_mini", model="test-model", max_gross_exposure=1.0, max_daily_turnover=1.0),
            SecretConfig(openai_api_key="sk-test"),
            {"portfolio_state": {}, "symbol_summary_table": [], "benchmark_rules": {"max_gross_exposure": 1.0}},
            {
                "target_weights": {"AAPL": 1.0, "MSFT": -1.0},
                "cash_weight": -1.0,
                "gross_exposure": 1.0,
                "net_exposure": 0.0,
            },
            ["AAPL", "MSFT"],
            book=initial_book(1000),
            prices={"AAPL": 100.0, "MSFT": 100.0},
            dry_run=False,
            cache_namespace="test-repair",
        )
    finally:
        engine.close()

    assert repair_calls == 1
    assert len(calls) == 1
    assert "validation_errors" in calls[0]["user"]
    assert repaired["target_weights"] == {"AAPL": 0.6, "MSFT": -0.4}
    assert repaired["_allocation_repair"]["attempted"] is True
    assert repaired["_allocation_repair"]["remaining_errors"] == []


def test_stage2_allocation_contract_rejects_incoherent_math_and_costs():
    engine = BenchmarkEngine()
    try:
        errors = engine._allocation_errors(
            BenchmarkConfig(mode="balanced_50_portfolio", run_preset="balanced_50_mini", max_gross_exposure=1.0, max_daily_turnover=0.2, slippage_bps=10, turnover_edge_multiplier=3),
            {
                "target_weights": {"AAPL": 0.7, "MSFT": -0.2},
                "cash_weight": 0.5,
                "gross_exposure": 0.5,
                "net_exposure": 0.0,
                "expected_return_bps": 1,
                "expected_holding_days": 5,
                "estimated_turnover": 0.1,
                "input_evidence_refs": [],
                "data_quality_warnings_used": [],
            },
            ["AAPL", "MSFT"],
            book=initial_book(1000),
            prices={"AAPL": 100.0, "MSFT": 100.0},
        )
    finally:
        engine.close()

    error_types = {item["type"] for item in errors}
    assert "gross_exposure_mismatch" in error_types
    assert "net_exposure_mismatch" in error_types
    assert "cash_weight_mismatch" in error_types
    assert "estimated_turnover_mismatch" in error_types
    assert "max_daily_turnover_exceeded" in error_types
    assert "turnover_cost_hurdle_failed" in error_types


def test_stage2_no_model_fallback_keeps_current_weights():
    engine = BenchmarkEngine()
    try:
        fallback = engine._stage2_fallback(
            ["AAPL", "MSFT"],
            {"current_position_weights": {"AAPL": 0.2, "MSFT": -0.1}},
        )
    finally:
        engine.close()

    assert fallback["target_weights"] == {"AAPL": 0.2, "MSFT": -0.1}
    assert fallback["gross_exposure"] == pytest.approx(0.3)
    assert fallback["net_exposure"] == pytest.approx(0.1)
    assert fallback["cash_weight"] == pytest.approx(0.7)
    assert fallback["estimated_turnover"] == 0.0


def test_single_stock_valid_target_exposure_range_clips_turnover_and_short_rules():
    engine = BenchmarkEngine()
    try:
        config = BenchmarkConfig(max_daily_turnover=0.2, turnover_prompt_buffer=0.02, allow_short=True, max_gross_exposure=1.0)
        ranged = engine._valid_target_exposure_range(config, {"AAPL": 0.10}, ["AAPL"])
        long_only = engine._valid_target_exposure_range(
            BenchmarkConfig(max_daily_turnover=0.2, turnover_prompt_buffer=0.02, allow_short=False, max_gross_exposure=1.0),
            {"AAPL": 0.10},
            ["AAPL"],
        )
        near_limit = engine._valid_target_exposure_range(config, {"AAPL": 0.95}, ["AAPL"])
        trinary = engine._valid_target_exposure_range(
            BenchmarkConfig(
                max_daily_turnover=2.0,
                turnover_prompt_buffer=0.0,
                allow_short=True,
                max_gross_exposure=1.0,
                single_stock_action_space="trinary_all_in",
            ),
            {"AAPL": 0.0},
            ["AAPL"],
        )
    finally:
        engine.close()

    assert ranged["min"] == pytest.approx(-0.08)
    assert ranged["max"] == pytest.approx(0.28)
    assert long_only["min"] == 0.0
    assert near_limit["max"] == 1.0
    assert trinary["allowed_actions"] == ["SHORT_ALL", "HOLD", "BUY_ALL"]
    assert trinary["allowed_target_exposures"] == {"SHORT_ALL": -1.0, "HOLD": 0.0, "BUY_ALL": 1.0}


def test_trinary_single_stock_actions_are_normalized_and_partial_targets_rejected():
    engine = BenchmarkEngine()
    try:
        config = BenchmarkConfig(
            mode="single_stock",
            allow_short=True,
            max_gross_exposure=1.0,
            max_daily_turnover=0.0,
            turnover_prompt_buffer=0.0,
            single_stock_action_space="trinary_all_in",
        )
        manager_bundle = {
            "current_position_weights": {"AAPL": 0.25},
            "valid_target_exposure_range": engine._valid_target_exposure_range(config, {"AAPL": 0.25}, ["AAPL"]),
        }
        normalized = engine._normalize_stage2_output(
            config,
            {
                "action": "SHORT_ALL",
                "expected_holding_days": 20,
                "rebalance_reason": "test",
                "input_evidence_refs": [],
                "data_quality_warnings_used": [],
                "cash_drag_justification": "short avoids drawdown",
                "why_not_buy_hold": "bearish evidence",
                "stage1_alignment": "veto",
                "stage1_veto_reason": "bearish evidence",
            },
            ["AAPL"],
            manager_bundle,
        )
        hold = engine._normalize_stage2_output(config, {"action": "HOLD"}, ["AAPL"], manager_bundle)
        errors = engine._allocation_errors(
            config,
            {
                "target_exposure": 0.4,
                "expected_holding_days": 20,
                "rebalance_reason": "test",
                "input_evidence_refs": [],
                "data_quality_warnings_used": [],
                "cash_drag_justification": "test",
                "why_not_buy_hold": "test",
                "stage1_alignment": "partial",
                "stage1_veto_reason": "",
            },
            ["AAPL"],
            manager_bundle=manager_bundle,
        )
    finally:
        engine.close()

    assert normalized["target_exposure"] == -1.0
    assert normalized["target_weights"] == {"AAPL": -1.0}
    assert normalized["cash_weight"] == 0.0
    assert normalized["estimated_turnover"] == pytest.approx(1.25)
    assert hold["target_exposure"] == pytest.approx(0.25)
    error_types = {item["type"] for item in errors}
    assert "invalid_trinary_action" in error_types
    assert "invalid_trinary_target_exposure" in error_types


def test_trinary_full_flips_and_hold_mechanically_corrects_gross_drift():
    engine = BenchmarkEngine()
    try:
        config = BenchmarkConfig(
            mode="single_stock",
            allow_short=True,
            max_gross_exposure=1.0,
            max_daily_turnover=0.0,
            turnover_prompt_buffer=0.0,
            single_stock_action_space="trinary_all_in",
        )
        manager_bundle = {
            "current_position_weights": {"AAPL": -1.12},
            "valid_target_exposure_range": engine._valid_target_exposure_range(config, {"AAPL": -1.12}, ["AAPL"]),
        }
        buy_all = engine._normalize_stage2_output(
            config,
            {
                "action": "BUY_ALL",
                "expected_holding_days": 20,
                "rebalance_reason": "flip after bullish evidence",
                "input_evidence_refs": [],
                "data_quality_warnings_used": [],
                "cash_drag_justification": "full long avoids missed upside",
                "why_not_buy_hold": "same as buy-hold for this step",
                "stage1_alignment": "follow",
                "stage1_veto_reason": "",
            },
            ["AAPL"],
            manager_bundle,
        )
        hold = engine._normalize_stage2_output(
            config,
            {
                "action": "HOLD",
                "expected_holding_days": 20,
                "rebalance_reason": "no new evidence",
                "input_evidence_refs": [],
                "data_quality_warnings_used": [],
                "cash_drag_justification": "no cash change",
                "why_not_buy_hold": "holding the existing position",
                "stage1_alignment": "partial",
                "stage1_veto_reason": "",
            },
            ["AAPL"],
            manager_bundle,
        )
        buy_errors = engine._allocation_errors(config, buy_all, ["AAPL"], manager_bundle=manager_bundle)
        hold_errors = engine._allocation_errors(config, hold, ["AAPL"], manager_bundle=manager_bundle)
        drifted_book = PortfolioBook(
            cash=212.0,
            positions={"AAPL": -1.0},
            equity=100.0,
            long_exposure=0.0,
            short_exposure=1.12,
            gross_exposure=1.12,
            net_exposure=-1.12,
        )
        next_book, execution = engine._execute_stage2_output(
            config,
            hold,
            hold["target_weights"],
            drifted_book,
            {"AAPL": 112.0},
            hold_errors,
        )
    finally:
        engine.close()

    assert manager_bundle["valid_target_exposure_range"]["allowed_actions"] == ["SHORT_ALL", "HOLD", "BUY_ALL"]
    assert buy_all["target_exposure"] == 1.0
    assert buy_all["estimated_turnover"] == pytest.approx(2.12)
    assert buy_errors == []
    assert -1.0 < hold["target_exposure"] < -0.99
    assert hold["gross_exposure"] < 1.0
    assert hold["_mechanical_deleverage"] is True
    assert hold_errors == []
    assert next_book.gross_exposure <= 1.0 + 1e-9
    assert execution["trades"]
    assert execution["events"][-1]["type"] == "mechanical_gross_deleverage"
    assert execution["model_failure"] is False


def test_single_stock_target_exposure_is_normalized_to_executable_weights():
    engine = BenchmarkEngine()
    try:
        config = BenchmarkConfig(slippage_bps=5, max_daily_turnover=0.2)
        normalized = engine._normalize_stage2_output(
            config,
            {
                "target_exposure": 0.28,
                "cash_weight": 0.0,
                "gross_exposure": 0.0,
                "net_exposure": 0.0,
                "expected_holding_days": 20,
                "rebalance_reason": "test",
                "input_evidence_refs": [],
                "data_quality_warnings_used": [],
                "cash_drag_justification": "A small cash reserve is justified by volatility.",
                "why_not_buy_hold": "Do not use full buy-and-hold because the setup is mixed.",
                "stage1_alignment": "partial",
                "stage1_veto_reason": "",
            },
            ["AAPL"],
            {"current_position_weights": {"AAPL": 0.10}},
        )
        errors = engine._allocation_errors(config, normalized, ["AAPL"])
    finally:
        engine.close()

    assert normalized["target_weights"] == {"AAPL": 0.28}
    assert normalized["cash_weight"] == pytest.approx(0.72)
    assert normalized["gross_exposure"] == pytest.approx(0.28)
    assert normalized["net_exposure"] == pytest.approx(0.28)
    assert normalized["estimated_turnover"] == pytest.approx(0.18)
    assert errors == []


def test_missing_macro_is_omitted_from_prompts(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    engine = BenchmarkEngine(warehouse)
    try:
        warehouse.conn.execute(
            """
            INSERT INTO macro_daily (date, series_id, label, observation_date, value, source, source_status, vintage_safe)
            VALUES (DATE '2025-01-02', 'DGS10', '10Y Treasury', DATE '2025-01-02', NULL, 'fred', 'missing_key', true)
            """
        )

        macro = engine._macro(BenchmarkConfig(macro_policy="omit_if_missing"), "2025-01-02")
    finally:
        engine.close()

    assert macro == []


def test_synthetic_gdelt_titles_are_aggregated_not_passed_as_headlines(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    engine = BenchmarkEngine(warehouse)
    try:
        warehouse.conn.execute(
            """
            INSERT INTO news_articles
                (article_id, symbol, bucket_start, bucket_end, published_at, title, url, domain, language, source_country, source, query, raw_json)
            VALUES
                ('n1', 'AAPL', DATE '2025-01-02', DATE '2025-01-02', '2025-01-02T12:00:00Z',
                 'GDELT event 010 nan', '', '', 'eng', 'US', 'gdelt_events', 'AAPL',
                 '{"AvgTone": -1.5, "NumMentions": 8, "EventCode": "010"}')
            """
        )

        news, quality = engine._news(["AAPL"], "2025-01-03", 5, BenchmarkConfig())
    finally:
        engine.close()

    assert quality["status"] == "event_aggregate_only"
    assert news["AAPL"][0]["type"] == "event_summary"
    assert all("title" not in item or not str(item["title"]).startswith("GDELT event") for item in news["AAPL"])


def test_stale_sec_facts_are_excluded_from_canonical_fundamentals():
    canonical, quality = canonicalize_fundamentals(
        [
            {"concept": "Revenues", "value": 1000, "filed_date": "2020-01-01", "period_end": "2019-12-31"},
            {"concept": "Revenues", "value": 1500, "filed_date": "2024-03-01", "period_end": "2023-12-31"},
            {"concept": "OperatingIncomeLoss", "value": 300, "filed_date": "2024-03-01", "period_end": "2023-12-31"},
        ],
        "2025-01-02",
    )

    assert canonical["revenue_ttm"] == 1500
    assert canonical["operating_margin"] == pytest.approx(0.2)
    assert quality["stale_facts_excluded"] == 1
    assert quality["status"] == "ok"


def test_legacy_run_missing_stage2_contract_is_classified_diagnostic():
    run = {
        "decisions": [
            {
                "stage": "stage1",
                "input": {"news_and_events": {"AAPL": [{"title": "GDELT event 010 nan"}]}},
                "output": {},
                "execution": {},
            },
            {
                "stage": "stage2",
                "decision_date": "2025-01-02",
                "fill_date": "2025-01-03",
                "input": {},
                "output": {"target_weights": {"AAPL": 0.0}, "gross_exposure": 0.0, "net_exposure": 0.0},
                "execution": {
                    "portfolio_before": {"equity": 1000},
                    "portfolio_after": {"equity": 1000, "gross_exposure": 0.0, "net_exposure": 0.0, "short_exposure": 0.0, "positions": {}},
                    "target_weights": {"AAPL": 0.0},
                    "trades": [],
                    "model_failure": False,
                },
            },
        ]
    }

    diagnostics = build_run_diagnostics(run, BenchmarkConfig(mode="single_stock", selected_symbols=["AAPL"], initial_cash=1000))

    assert diagnostics["official_status"] == "diagnostic"
    assert "legacy_stage2_schema" in diagnostics["diagnostic_reasons"]
    assert "synthetic_news_titles_in_prompt" in diagnostics["diagnostic_reasons"]
    assert diagnostics["metrics"]["legacy_stage2_schema_days"] == 1


def test_single_stock_diagnostics_report_participation_and_cash_drag(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    try:
        warehouse.conn.execute(
            """
            INSERT INTO asset_daily
                (date, symbol, market_open, listed, ohlcv_available, open, high, low, close, adj_close, volume, return_1d, source, source_error)
            VALUES
                (DATE '2025-01-03', 'AAPL', true, true, true, 100, 100, 100, 100, 100, 1000, NULL, 'test', ''),
                (DATE '2025-01-06', 'AAPL', true, true, true, 110, 110, 110, 110, 110, 1000, 0.10, 'test', '')
            """
        )
        run = {
            "decisions": [
                {
                    "stage": "stage1",
                    "decision_date": "2025-01-02",
                    "output": {"analyses": [{"symbol": "AAPL", "stance": "bullish"}]},
                    "input": {},
                    "execution": {},
                },
                {
                    "stage": "stage2",
                    "decision_date": "2025-01-02",
                    "fill_date": "2025-01-03",
                    "input": {},
                    "output": {
                        "target_exposure": 0.10,
                        "target_weights": {"AAPL": 0.10},
                        "cash_weight": 0.90,
                        "gross_exposure": 0.10,
                        "net_exposure": 0.10,
                        "expected_holding_days": 20,
                        "estimated_turnover": 0.10,
                        "estimated_slippage_cost_bps": 0.5,
                        "rebalance_reason": "test",
                        "input_evidence_refs": [],
                        "data_quality_warnings_used": [],
                        "cash_drag_justification": "test",
                        "why_not_buy_hold": "test",
                        "stage1_alignment": "veto",
                        "stage1_veto_reason": "test",
                    },
                    "execution": {
                        "portfolio_before": {"equity": 1000},
                        "portfolio_after": {"equity": 1000, "gross_exposure": 0.10, "net_exposure": 0.10, "short_exposure": 0.0, "positions": {}},
                        "target_weights": {"AAPL": 0.10},
                        "trades": [],
                        "model_failure": False,
                        "slippage_cost": 0.0,
                    },
                },
                {
                    "stage": "stage1",
                    "decision_date": "2025-01-03",
                    "output": {"analyses": [{"symbol": "AAPL", "stance": "bullish"}]},
                    "input": {},
                    "execution": {},
                },
                {
                    "stage": "stage2",
                    "decision_date": "2025-01-03",
                    "fill_date": "2025-01-06",
                    "input": {},
                    "output": {
                        "target_exposure": 0.80,
                        "target_weights": {"AAPL": 0.80},
                        "cash_weight": 0.20,
                        "gross_exposure": 0.80,
                        "net_exposure": 0.80,
                        "expected_holding_days": 20,
                        "estimated_turnover": 0.70,
                        "estimated_slippage_cost_bps": 3.5,
                        "rebalance_reason": "test",
                        "input_evidence_refs": [],
                        "data_quality_warnings_used": [],
                        "cash_drag_justification": "test",
                        "why_not_buy_hold": "test",
                        "stage1_alignment": "follow",
                        "stage1_veto_reason": "",
                    },
                    "execution": {
                        "portfolio_before": {"equity": 1000},
                        "portfolio_after": {"equity": 1010, "gross_exposure": 0.80, "net_exposure": 0.80, "short_exposure": 0.0, "positions": {}},
                        "target_weights": {"AAPL": 0.80},
                        "trades": [],
                        "model_failure": False,
                        "slippage_cost": 0.0,
                    },
                },
            ]
        }

        diagnostics = build_run_diagnostics(run, BenchmarkConfig(mode="single_stock", symbol="AAPL", initial_cash=1000), warehouse)
    finally:
        warehouse.close()

    metrics = diagnostics["metrics"]
    assert metrics["avg_target_exposure"] == pytest.approx(0.45)
    assert metrics["participation_ratio"] == pytest.approx(0.45)
    assert metrics["missed_upside_days"] == 1
    assert metrics["cash_drag_proxy"] == pytest.approx(0.09)
    assert metrics["bullish_but_underexposed_days"] == 1
    assert metrics["stage1_follow_rate"] == pytest.approx(0.5)
    assert metrics["stage1_veto_rate"] == pytest.approx(0.5)


def test_deterministic_memory_exposes_suggested_exposure_metadata():
    memory = DeterministicMarketMemory.__new__(DeterministicMarketMemory)
    rows = pd.DataFrame(
        [
            {"symbol": "AAPL", "date": "2024-01-02", "knowledge_timestamp": "2024-02-01", "_distance": 0.1, "outcome_20d": 0.08},
            {"symbol": "AAPL", "date": "2024-01-03", "knowledge_timestamp": "2024-02-02", "_distance": 0.2, "outcome_20d": 0.04},
            {"symbol": "AAPL", "date": "2024-01-04", "knowledge_timestamp": "2024-02-03", "_distance": 0.3, "outcome_20d": -0.01},
        ]
    )

    aggregate = memory._aggregate_to_memory("AAPL", rows, examples_per_symbol=0)

    suggested = aggregate["metadata"]["suggested_exposure"]
    assert suggested["base_rate_return"] == pytest.approx((0.08 + 0.04 - 0.01) / 3)
    assert suggested["hit_rate"] == pytest.approx(2 / 3)
    assert suggested["downside_rate"] == pytest.approx(1 / 3)
    assert "band" in suggested
    assert "Suggested exposure evidence" in aggregate["content"]


def test_diagnostic_lesson_memory_is_point_in_time_safe(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.db")
    config = BenchmarkConfig(model="test-model", outcome_learning_mode="diagnostic_lessons", memory_retrieval="structured")
    memory = HybridMemory(store, config, SecretConfig())
    memory.add(
        portfolio_scope="single_stock",
        symbol="AAPL",
        decision_timestamp="2025-01-02",
        knowledge_timestamp="2025-01-05",
        source_run_id="run",
        memory_type="diagnostic_lesson",
        content="AAPL due lesson.",
    )
    memory.add(
        portfolio_scope="single_stock",
        symbol="AAPL",
        decision_timestamp="2025-01-02",
        knowledge_timestamp="2025-01-20",
        source_run_id="run",
        memory_type="diagnostic_lesson",
        content="AAPL future lesson.",
    )
    engine = BenchmarkEngine()
    try:
        retrieved = engine._diagnostic_lesson_memories(memory, config, "2025-01-10", "AAPL", limit=10)
    finally:
        engine.close()

    assert [item["content"] for item in retrieved] == ["AAPL due lesson."]


def test_responses_calls_request_json_mode(monkeypatch):
    calls = []

    def fake_post_json(url, headers, payload):
        calls.append(payload)
        return {"output_text": '{"ok": true}'}

    monkeypatch.setattr("agent_benchmark.llm_client._post_json", fake_post_json)

    result = call_json_model(
        BenchmarkConfig(model="test-model", use_cached_llm=False),
        SecretConfig(openai_api_key="sk-test"),
        "Return JSON.",
        '{"task": "test"}',
    )

    assert result["ok"] is True
    assert calls[0]["text"]["format"]["type"] == "json_object"


def test_model_usage_metadata_is_saved_from_provider_response(monkeypatch):
    def fake_post_json(url, headers, payload):
        return {
            "output_text": '{"ok": true}',
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
                "input_tokens_details": {"cached_tokens": 20},
            },
        }

    monkeypatch.setattr("agent_benchmark.llm_client._post_json", fake_post_json)

    result = call_json_model(
        BenchmarkConfig(model="gpt-5.4-mini-2026-03-17", use_cached_llm=False),
        SecretConfig(openai_api_key="sk-test"),
        "Return JSON.",
        '{"task": "test"}',
    )

    assert result["_api_usage"] == {
        "input_tokens": 120,
        "cached_input_tokens": 20,
        "output_tokens": 30,
        "total_tokens": 150,
    }
    assert result["_api_usage_source"] == "provider_usage"


def test_chat_completions_use_new_token_limit_field(monkeypatch):
    calls = []

    def fake_post_json(url, headers, payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setattr("agent_benchmark.llm_client._post_json", fake_post_json)

    result = call_json_model(
        BenchmarkConfig(model="test-model", endpoint="chat_completions", use_cached_llm=False),
        SecretConfig(openai_api_key="sk-test"),
        "Return JSON.",
        '{"task": "test"}',
    )

    assert result["ok"] is True
    assert "max_completion_tokens" in calls[0]
    assert "max_tokens" not in calls[0]


def test_malformed_model_json_retries_once(monkeypatch):
    calls = []

    def fake_post_json(url, headers, payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"output_text": '{"analyses": [{"symbol": "AAPL" "stance": "neutral"}]}'}
        return {"output_text": '{"analyses": [{"symbol": "AAPL", "stance": "neutral"}]}'}

    monkeypatch.setattr("agent_benchmark.llm_client._post_json", fake_post_json)
    monkeypatch.setattr("agent_benchmark.llm_client._write_malformed_response", lambda *args, **kwargs: None)

    result = call_json_model(
        BenchmarkConfig(model="test-model", use_cached_llm=False),
        SecretConfig(openai_api_key="sk-test"),
        "Return JSON.",
        '{"task": "test"}',
        cache_namespace="stage1",
    )

    assert result["analyses"][0]["symbol"] == "AAPL"
    assert result["_api_retry_count"] == 1
    assert len(calls) == 2
    assert "previous_response" in calls[1]["input"]


def test_truncated_model_json_retries_with_larger_output_cap(monkeypatch):
    calls = []

    def fake_post_json(url, headers, payload):
        calls.append(payload)
        if len(calls) == 1:
            return {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": '{"analyses": [{"symbol": "AAPL", "stance": "neutral", "proposed_target_weight": -0',
            }
        return {"output_text": '{"analyses": [{"symbol": "AAPL", "stance": "neutral", "proposed_target_weight": -0.01}]}'}

    monkeypatch.setattr("agent_benchmark.llm_client._post_json", fake_post_json)
    monkeypatch.setattr("agent_benchmark.llm_client._write_malformed_response", lambda *args, **kwargs: None)

    result = call_json_model(
        BenchmarkConfig(model="test-model", max_output_tokens=900, use_cached_llm=False),
        SecretConfig(openai_api_key="sk-test"),
        "Return JSON.",
        '{"task": "test"}',
        cache_namespace="stage1",
    )

    assert result["analyses"][0]["proposed_target_weight"] == -0.01
    assert result["_api_retry_count"] == 1
    assert calls[0]["max_output_tokens"] == 900
    assert calls[1]["max_output_tokens"] == 1800


def test_post_json_retries_rate_limits(monkeypatch):
    calls = []
    sleeps = []

    class FakeResponse:
        def __init__(self, status_code, text, payload=None):
            self.status_code = status_code
            self.text = text
            self.headers = {}
            self._payload = payload or {}

        def json(self):
            return self._payload

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        if len(calls) == 1:
            return FakeResponse(429, "Rate limit reached. Please try again in 1.268s.")
        return FakeResponse(200, "ok", {"ok": True})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("agent_benchmark.llm_client.time.sleep", lambda seconds: sleeps.append(seconds))

    result = _post_json("https://example.test", {}, {"hello": "world"})

    assert result == {"ok": True}
    assert len(calls) == 2
    assert sleeps == [1.768]


def test_retry_after_prefers_header():
    response = requests.Response()
    response.status_code = 429
    response._content = b"Please try again in 1.268s."
    response.headers["Retry-After"] = "3.5"

    assert _retry_after_seconds(response, default=1.0) == 3.5


def test_compact_stage_bundles_remove_unrelated_symbol_bulk():
    engine = BenchmarkEngine()
    try:
        bundle = {
            "schema_version": "test",
            "mode": "balanced_50_portfolio",
            "decision_date": "2025-01-02",
            "fill_date": "2025-01-03",
            "candidate_universe": [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
            "portfolio_state": {"cash": 800, "positions": {"AAPL": 2}, "equity": 1000},
            "market_snapshots": {"AAPL": {"close": 100, "return_20d": 0.1}, "MSFT": {"close": 50, "return_20d": -0.1}},
            "fundamentals": {"AAPL": {"Revenue": 1}, "MSFT": {"Revenue": 2}},
            "news_and_events": {"AAPL": [{"title": "a"}], "MSFT": [{"title": "m"}]},
            "data_quality": {"checked_symbols": 2, "missing_ohlcv": ["MSFT"], "not_listed": []},
            "memory": [{"symbol": "AAPL", "content": "a"}, {"symbol": "MSFT", "content": "m"}],
        }

        stage1 = engine._stage1_bundle(bundle, ["AAPL"])
        stage2 = engine._stage2_bundle(bundle)

        assert list(stage1["market_snapshots"]) == ["AAPL"]
        assert list(stage1["fundamentals"]) == ["AAPL"]
        assert list(stage1["news_and_events"]) == ["AAPL"]
        assert stage1["data_quality"]["missing_ohlcv"] == []
        assert "market_snapshots" not in stage2
        assert "fundamentals" not in stage2
        assert "news_and_events" not in stage2
        assert stage2["current_position_weights"]["AAPL"] == pytest.approx(0.2)
        assert stage2["symbol_summary_table"][0]["current_weight"] == pytest.approx(0.2)
    finally:
        engine.close()


def test_decision_support_ranks_symbols_and_flows_into_stage_bundles():
    engine = BenchmarkEngine()
    try:
        bundle = {
            "schema_version": "test",
            "mode": "balanced_50_portfolio",
            "decision_date": "2025-01-02",
            "fill_date": "2025-01-03",
            "candidate_universe": [{"symbol": "AAPL", "sector": "Technology"}, {"symbol": "MSFT", "sector": "Technology"}],
            "portfolio_state": {"cash": 1000, "positions": {}, "equity": 1000},
            "market_snapshots": {
                "AAPL": {"close": 100, "return_5d": 0.03, "return_20d": 0.12, "return_60d": 0.2, "volatility_20d": 0.2},
                "MSFT": {"close": 100, "return_5d": -0.03, "return_20d": -0.08, "return_60d": -0.1, "volatility_20d": 0.3},
            },
            "news_and_events": {"AAPL": [{"type": "event_summary", "avg_tone": 1.0, "event_rows": 2}], "MSFT": []},
            "memory": [
                {
                    "symbol": "AAPL",
                    "memory_type": "deterministic_market_aggregate",
                    "retrieval_score": 0.9,
                    "metadata": {
                        "confidence": "moderate",
                        "aggregate_stats": {"20d": {"cases": 30, "mean_return": 0.08, "median_return": 0.04, "hit_rate": 0.7, "downside_rate": 0.3}},
                    },
                },
                {
                    "symbol": "MSFT",
                    "memory_type": "deterministic_market_aggregate",
                    "retrieval_score": 0.9,
                    "metadata": {
                        "confidence": "moderate",
                        "aggregate_stats": {"20d": {"cases": 30, "mean_return": -0.04, "median_return": -0.03, "hit_rate": 0.35, "downside_rate": 0.65}},
                    },
                },
            ],
        }
        bundle["decision_support"] = engine._decision_support(bundle)
        stage1 = engine._stage1_bundle(bundle, ["AAPL"])
        stage2 = engine._stage2_bundle(bundle)
    finally:
        engine.close()

    assert bundle["decision_support"]["ranked_symbols"][0]["symbol"] == "AAPL"
    assert bundle["decision_support"]["ranked_symbols"][0]["stance_hint"] == "favorable"
    assert [item["symbol"] for item in stage1["decision_support"]["ranked_symbols"]] == ["AAPL"]
    assert stage2["decision_support"]["ranked_symbols"][0]["symbol"] == "AAPL"
    table_by_symbol = {item["symbol"]: item for item in stage2["symbol_summary_table"]}
    assert table_by_symbol["AAPL"]["signal_rank"] == 1
    assert table_by_symbol["AAPL"]["memory_cases"] == 30


def test_live_scheduler_explains_closed_weekend_market():
    scheduler = LiveScheduler(BenchmarkJobManager())
    ny = ZoneInfo("America/New_York")

    session = scheduler._market_session(datetime(2026, 6, 14, 12, 0, tzinfo=ny))

    assert session["market_open"] is False
    assert session["reason"] == "weekend"
    assert "weekend" in session["message"]


def test_daily_open_scheduler_waits_for_the_compatible_execution_window():
    scheduler = LiveScheduler(BenchmarkJobManager())
    ny = ZoneInfo("America/New_York")

    assert scheduler._seconds_until_daily_open_snapshot(
        datetime(2026, 6, 15, 9, 40, tzinfo=ny)
    ) == 0.0
    delay = scheduler._seconds_until_daily_open_snapshot(
        datetime(2026, 6, 12, 11, 0, tzinfo=ny)
    )

    assert delay > 2 * 24 * 60 * 60


def test_summary_includes_buy_hold_market_comparison(tmp_path):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    engine = None
    try:
        for table in ("context_daily", "asset_daily"):
            for symbol, start_price, end_price in (
                ("SPY", 100.0, 110.0),
                ("QQQ", 100.0, 120.0),
                ("AAPL", 100.0, 130.0),
                ("MSFT", 100.0, 90.0),
            ):
                if table == "context_daily" and symbol not in {"SPY", "QQQ"}:
                    continue
                if table == "asset_daily" and symbol in {"SPY", "QQQ"}:
                    continue
                warehouse.conn.execute(
                    f"""
                    INSERT INTO {table}
                        (date, symbol, market_open, listed, ohlcv_available, open, high, low, close, adj_close, volume, return_1d, source, source_error)
                    VALUES
                        (DATE '2025-01-02', ?, true, true, true, ?, ?, ?, ?, ?, 1000, NULL, 'test', ''),
                        (DATE '2025-01-03', ?, true, true, true, ?, ?, ?, ?, ?, 1000, ?, 'test', '')
                    """,
                    [
                        symbol,
                        start_price,
                        start_price,
                        start_price,
                        start_price,
                        start_price,
                        symbol,
                        end_price,
                        end_price,
                        end_price,
                        end_price,
                        end_price,
                        end_price / start_price - 1.0,
                    ],
                )

        engine = BenchmarkEngine(warehouse)
        summary = engine._summary(
            BenchmarkConfig(selected_symbols=["AAPL", "MSFT"], initial_cash=1000),
            ["AAPL", "MSFT"],
            [
                {"date": "2025-01-02", "phase": "test", "equity": 1000},
                {"date": "2025-01-03", "phase": "test", "equity": 1050},
            ],
            [],
            model_calls=0,
            dry_run=False,
        )
    finally:
        if engine is not None:
            engine.close()

    comparison = summary["buy_hold_comparison"]
    by_id = {item["id"]: item for item in comparison["benchmarks"]}

    assert summary["metrics"]["total_return"] == pytest.approx(0.05)
    assert by_id["spy"]["total_return"] == pytest.approx(0.1)
    assert by_id["qqq"]["total_return"] == pytest.approx(0.2)
    assert by_id["selected_equal_weight"]["total_return"] == pytest.approx(0.1)
    assert by_id["spy"]["excess_return"] == pytest.approx(-0.05)
    assert summary["metrics"]["alpha_spy"] == pytest.approx(-0.05)
    assert summary["success_evaluation_window"]["name"] == "2025_test"
    assert summary["test_metrics"]["total_return"] == pytest.approx(0.05)
    assert summary["test_buy_hold_comparison"]["start_date"] == "2025-01-02"
    assert summary["test_buy_hold_comparison"]["end_date"] == "2025-01-03"
    test_by_id = {item["id"]: item for item in summary["test_buy_hold_comparison"]["benchmarks"]}
    assert test_by_id["selected_equal_weight"]["total_return"] == pytest.approx(0.1)


def test_api_usage_estimate_prefers_exact_provider_usage():
    config = BenchmarkConfig(model="gpt-5.4-mini-2026-03-17", initial_cash=1000)
    run = {
        "id": "usage-run",
        "model": config.model,
        "config": config.model_dump() if hasattr(config, "model_dump") else config.dict(),
        "summary": {"model": config.model, "model_calls": 2},
        "progress": {"model_calls": 2},
        "decisions": [
            {
                "stage": "stage1",
                "decision_date": "2025-01-02",
                "symbol": "AAPL",
                "input": {"data_quality": {}},
                "output": {
                    "_api_status": "ok",
                    "_api_usage": {"input_tokens": 100, "cached_input_tokens": 10, "output_tokens": 20, "total_tokens": 120},
                    "analyses": [],
                },
                "execution": {},
            },
            {
                "stage": "stage2",
                "decision_date": "2025-01-02",
                "symbol": "PORTFOLIO",
                "input": {"portfolio_state": {}},
                "output": {
                    "_api_status": "ok",
                    "_api_usage": {"input_tokens": 200, "cached_input_tokens": 0, "output_tokens": 40, "total_tokens": 240},
                    "target_weights": {},
                },
                "execution": {},
            },
        ],
    }

    usage = estimate_run_api_usage(run, config)

    assert usage["is_estimate"] is False
    assert usage["input_tokens"] == 300
    assert usage["cached_input_tokens"] == 10
    assert usage["output_tokens"] == 60
    assert usage["estimated_cost_usd"] == pytest.approx(round((290 * 0.75 + 10 * 0.075 + 60 * 4.5) / 1_000_000, 6))


def test_api_usage_estimate_can_reconstruct_old_runs_without_usage():
    config = BenchmarkConfig(model="gpt-5.4-mini-2026-03-17", initial_cash=1000)
    run = {
        "id": "old-run",
        "model": config.model,
        "config": config.model_dump() if hasattr(config, "model_dump") else config.dict(),
        "summary": {"model": config.model, "model_calls": 3},
        "progress": {"model_calls": 3},
        "decisions": [
            {
                "stage": "stage1",
                "decision_date": "2025-01-02",
                "symbol": "AAPL",
                "input": {"market_snapshots": {"AAPL": {"close": 100}}},
                "output": {"_api_status": "ok", "_raw_text": '{"analyses":[]}', "analyses": []},
                "execution": {},
            },
            {
                "stage": "stage2",
                "decision_date": "2025-01-02",
                "symbol": "PORTFOLIO",
                "input": {"portfolio_state": {}, "current_position_weights": {"AAPL": 0.0}},
                "output": {"_api_status": "ok", "_raw_text": '{"target_weights":{}}', "target_weights": {}},
                "execution": {"repair_count": 1},
            },
        ],
    }

    usage = estimate_run_api_usage(run, config)

    assert usage["is_estimate"] is True
    assert usage["stage1_calls"] == 1
    assert usage["stage2_calls"] == 1
    assert usage["repair_calls"] == 1
    assert usage["billable_model_calls"] == 3
    assert usage["estimated_cost_usd"] > 0


def test_api_usage_ignores_cadence_holds_without_model_calls():
    config = BenchmarkConfig(model="local-test", model_provider="ollama_local", no_paid_api_mode=True)
    run = {
        "config": config.model_dump() if hasattr(config, "model_dump") else config.dict(),
        "summary": {"model": config.model},
        "decisions": [
            {
                "stage": "stage2",
                "decision_date": "2025-01-03",
                "input": {},
                "output": {"_api_status": "cadence_hold", "action": "HOLD"},
                "execution": {},
            }
        ],
    }

    usage = estimate_run_api_usage(run, config)

    assert usage["stage2_calls"] == 0
    assert usage["total_tokens"] == 0
