from agent_benchmark.memory import HybridMemory
from agent_benchmark.api_usage import estimate_run_api_usage
from agent_benchmark.benchmark_engine import BenchmarkEngine
from agent_benchmark.llm_client import _post_json, _retry_after_seconds, call_json_model
from agent_benchmark.portfolio import execute_target_weights, initial_book
from agent_benchmark.prompting import build_stage1_prompt, build_stage2_prompt
from agent_benchmark.quality import build_run_diagnostics, canonicalize_fundamentals
from agent_benchmark.schemas import BenchmarkConfig, SecretConfig
from agent_benchmark.storage import BenchmarkStore
from agent_benchmark.jobs import BenchmarkJobManager, LiveScheduler
from agent_benchmark.warehouse.store import Warehouse

from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pytest


def test_config_defaults_follow_benchmark_contract():
    config = BenchmarkConfig()

    assert config.mode == "balanced_50_portfolio"
    assert config.run_preset == "balanced_50_mini"
    assert config.allow_short is True
    assert config.max_gross_exposure == 1.0
    assert config.decision_process == "two_stage_llm"
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
            BenchmarkConfig(model="test-model", max_gross_exposure=1.0, max_daily_turnover=1.0),
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
            BenchmarkConfig(max_gross_exposure=1.0, max_daily_turnover=0.2, slippage_bps=10, turnover_edge_multiplier=3),
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
    assert "turnover_cost_hurdle_failed" in error_types


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


def test_live_scheduler_explains_closed_weekend_market():
    scheduler = LiveScheduler(BenchmarkJobManager())
    ny = ZoneInfo("America/New_York")

    session = scheduler._market_session(datetime(2026, 6, 14, 12, 0, tzinfo=ny))

    assert session["market_open"] is False
    assert session["reason"] == "weekend"
    assert "weekend" in session["message"]


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
                {"date": "2025-01-02", "equity": 1000},
                {"date": "2025-01-03", "equity": 1050},
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
