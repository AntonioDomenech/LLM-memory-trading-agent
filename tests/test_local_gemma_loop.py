import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_benchmark.api_usage import estimate_run_api_usage
from agent_benchmark.benchmark_engine import BenchmarkEngine
from agent_benchmark.llm_client import call_json_model
from agent_benchmark.local_gemma_loop import (
    ALLOWLISTED_PATCH_CATEGORIES,
    LocalPatch,
    apply_allowlisted_patch,
    evaluate_success,
    propose_allowlisted_patch,
)
from agent_benchmark.local_provider import (
    LOCAL_DUMMY_API_KEY,
    is_loopback_url,
    local_gemma_aapl_config,
    local_gemma_secret_config,
    validate_no_paid_api_mode,
)
from agent_benchmark.memory import HybridMemory
from agent_benchmark.monitoring import AbortThresholds, evaluate_abort, parse_nvidia_smi_csv
from agent_benchmark.schemas import BenchmarkConfig, SecretConfig
from agent_benchmark.storage import BenchmarkStore
from agent_benchmark.warehouse.store import Warehouse


def test_local_gemma_config_is_chat_completions_and_no_paid_safe():
    config = local_gemma_aapl_config()
    secrets = local_gemma_secret_config()

    validate_no_paid_api_mode(config, secrets)

    assert config.model == "gemma4:12b"
    assert config.endpoint == "chat_completions"
    assert config.no_paid_api_mode is True
    assert config.memory_mode == "model_specific_cases_and_lessons"
    assert config.outcome_learning_mode == "llm_reflection_lessons"
    assert config.use_cached_llm is False
    assert config.allow_short is True
    assert config.single_stock_action_space == "trinary_all_in"
    assert config.max_daily_turnover == 2.0


def test_no_paid_mode_rejects_non_loopback_url_and_paid_sources():
    config = local_gemma_aapl_config(local_model_base_url="https://api.openai.com/v1")
    with pytest.raises(ValueError, match="loopback"):
        validate_no_paid_api_mode(config, local_gemma_secret_config(openai_base_url="https://api.openai.com/v1"))

    config = local_gemma_aapl_config()
    config.data_sources.news_sources = ["gdelt", "newsapi"]
    with pytest.raises(ValueError, match="paid news"):
        validate_no_paid_api_mode(config, local_gemma_secret_config())


def test_local_json_call_uses_dummy_key_without_real_openai_key(monkeypatch):
    calls = []

    def fake_post_json(url, headers, payload):
        calls.append({"url": url, "headers": headers, "payload": payload})
        return {"choices": [{"message": {"content": '{"ok": true}'}}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}

    monkeypatch.setattr("agent_benchmark.llm_client._post_json", fake_post_json)
    config = local_gemma_aapl_config(use_cached_llm=False)
    secrets = local_gemma_secret_config(openai_api_key="")

    result = call_json_model(config, secrets, "Return JSON.", '{"task":"test"}')

    assert result["ok"] is True
    assert calls[0]["url"] == "http://127.0.0.1:11434/api/chat"
    assert calls[0]["headers"]["Authorization"] == f"Bearer {LOCAL_DUMMY_API_KEY}"
    assert calls[0]["payload"]["think"] is False
    assert calls[0]["payload"]["stream"] is False
    assert calls[0]["payload"]["options"]["num_ctx"] == 6144
    assert result["_api_provider"] == "ollama_local"


def test_local_api_usage_cost_is_forced_to_zero():
    config = local_gemma_aapl_config()
    run = {
        "model": config.model,
        "config": config.model_dump() if hasattr(config, "model_dump") else config.dict(),
        "summary": {"model": config.model, "model_calls": 1},
        "decisions": [
            {
                "stage": "stage2",
                "decision_date": "2025-01-02",
                "input": {},
                "output": {"_api_status": "ok", "_api_usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}},
                "execution": {},
            }
        ],
    }

    usage = estimate_run_api_usage(run, config)

    assert usage["local_only"] is True
    assert usage["estimated_cost_usd"] == 0.0
    assert usage["estimated_cost_display"] == "$0.00"
    assert usage["billable_model_calls"] == 0


def test_llm_reflection_lessons_are_stored_point_in_time(tmp_path, monkeypatch):
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    try:
        _insert_market_rows(warehouse, ["2025-01-03", "2025-01-06", "2025-01-07"], [100, 100, 110])
        engine = BenchmarkEngine(warehouse)
        config = local_gemma_aapl_config(
            train_start="2025-01-06",
            train_end="2025-01-06",
            test_start="2025-01-07",
            test_end="2025-01-07",
            local_model_base_url="http://127.0.0.1:11434/v1",
        )
        memory = HybridMemory(store, config, local_gemma_secret_config())

        monkeypatch.setattr(
            "agent_benchmark.benchmark_engine.call_json_model",
            lambda *args, **kwargs: {
                "summary_lesson": "Full participation helped after a strong setup.",
                "lesson_tags": ["participation_helped"],
                "use_in_future_if": "trend is strong",
                "avoid_if": "setup changes",
                "confidence": 0.8,
                "_api_status": "ok",
            },
        )
        decisions = [
            {
                "phase": "training",
                "decision_date": "2025-01-03",
                "fill_date": "2025-01-06",
                "stage1_outputs": [],
                "stage2_output": {
                    "target_exposure": 1.0,
                    "expected_holding_days": 1,
                    "input_evidence_refs": [],
                    "data_quality_warnings_used": [],
                },
                "execution": {},
            }
        ]

        calls = engine._record_due_llm_reflection_lessons(memory, config, local_gemma_secret_config(), "run", decisions, "2025-01-07", dry_run=False)
        early = memory.retrieve(decision_timestamp="2025-01-06", query="participation", limit=5)
        due = memory.retrieve(decision_timestamp="2025-01-07", query="participation", limit=5)
    finally:
        engine.close()
        warehouse.close()

    assert calls == 1
    assert early == []
    assert len(due) == 1
    assert due[0]["memory_type"] == "llm_reflection_lesson"
    assert "Full participation helped" in due[0]["content"]


def test_success_evaluator_requires_buy_hold_zero_invalid_and_local_cost():
    config = local_gemma_aapl_config()
    run = {
        "summary": {
            "model": config.model,
            "model_provider": config.model_provider,
            "no_paid_api_mode": True,
            "local_model_base_url": config.local_model_base_url,
            "metrics": {"total_return": 0.31, "invalid_allocation_count": 0},
            "api_usage_estimate": {"local_only": True, "estimated_cost_usd": 0.0, "estimated_cost_display": "$0.00"},
            "buy_hold_comparison": {"benchmarks": [{"id": "single_stock", "total_return": 0.30}]},
        }
    }

    assert evaluate_success(run, config)["success"] is True
    run["summary"]["metrics"]["invalid_allocation_count"] = 1
    assert evaluate_success(run, config)["success"] is False


def test_success_evaluator_prefers_2025_test_window():
    config = local_gemma_aapl_config()
    run = {
        "summary": {
            "model": config.model,
            "model_provider": config.model_provider,
            "no_paid_api_mode": True,
            "local_model_base_url": config.local_model_base_url,
            "metrics": {"total_return": 0.05, "invalid_allocation_count": 0},
            "test_metrics": {"start_date": "2025-01-02", "end_date": "2025-12-31", "total_return": 0.12},
            "api_usage_estimate": {"local_only": True, "estimated_cost_usd": 0.0, "estimated_cost_display": "$0.00"},
            "buy_hold_comparison": {"benchmarks": [{"id": "single_stock", "total_return": 0.50}]},
            "test_buy_hold_comparison": {
                "start_date": "2025-01-02",
                "end_date": "2025-12-31",
                "benchmarks": [{"id": "single_stock", "total_return": 0.10}],
            },
            "success_evaluation_window": {"name": "2025_test", "phase": "test"},
        }
    }

    evaluation = evaluate_success(run, config)

    assert evaluation["success"] is True
    assert evaluation["beat_buy_hold"] is True
    assert evaluation["ai_return"] == pytest.approx(0.12)
    assert evaluation["buy_hold_return"] == pytest.approx(0.10)
    assert evaluation["evaluation_window"]["name"] == "2025_test"


def test_monitoring_parser_and_abort_thresholds():
    rows = parse_nvidia_smi_csv("92, 9790, 10000, 87, 280.5\n")

    assert rows[0]["gpu_utilization_pct"] == 92
    assert rows[0]["vram_used_mb"] == 9790
    reason = evaluate_abort({"gpu": {"gpus": rows}, "system_ram": {"ram_used_fraction": 0.5}}, AbortThresholds(gpu_temp_c=86, vram_fraction=0.98))
    assert "temperature" in reason

    reason = evaluate_abort({"gpu": {"gpus": [{"temperature_c": 60, "vram_used_mb": 9900, "vram_total_mb": 10000}]}, "system_ram": {}}, AbortThresholds(vram_fraction=0.98))
    assert "VRAM" in reason


def test_allowlisted_patch_policy_only_applies_known_categories():
    config = local_gemma_aapl_config()
    patch = propose_allowlisted_patch({"metrics": {"cash_drag_proxy": 0.1}}, {"beat_buy_hold": False, "invalid_decisions": 0}, config)

    assert patch.category in ALLOWLISTED_PATCH_CATEGORIES
    updated = apply_allowlisted_patch(config, patch)
    assert updated.allow_short is True
    assert updated.single_stock_action_space == "trinary_all_in"
    assert updated.max_daily_turnover == 2.0

    with pytest.raises(ValueError):
        apply_allowlisted_patch(config, LocalPatch(category="arbitrary_code", reason="bad", config_updates={}))


def test_fake_loopback_chat_server_works_with_local_client():
    server, base_url = _start_fake_chat_server()
    try:
        config = local_gemma_aapl_config(local_model_base_url=f"{base_url}/v1", use_cached_llm=False)
        result = call_json_model(config, local_gemma_secret_config(openai_api_key="sk-real-key-that-must-not-be-sent"), "Return JSON.", '{"task":"smoke"}')
    finally:
        server.shutdown()

    assert result["ok"] is True
    assert is_loopback_url(config.local_model_base_url)


def test_full_single_stock_train_test_pipeline_with_fake_local_server(tmp_path):
    server, base_url = _start_fake_chat_server()
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = None
    try:
        _insert_market_rows(
            warehouse,
            ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"],
            [100, 100, 110, 120],
        )
        config = local_gemma_aapl_config(
            train_start="2025-01-06",
            train_end="2025-01-06",
            test_start="2025-01-08",
            test_end="2025-01-08",
            max_train_days=1,
            max_test_days=1,
            local_model_base_url=f"{base_url}/v1",
            max_output_tokens=500,
            monitoring_enabled=False,
            use_cached_llm=False,
        )
        secrets = local_gemma_secret_config(openai_base_url=f"{base_url}/v1", openai_api_key="")
        run_id = "fake-local-run"
        store.create_benchmark_run(run_id, config.model_dump() if hasattr(config, "model_dump") else config.dict())
        engine = BenchmarkEngine(warehouse)
        engine.run(run_id=run_id, config=config, secrets=secrets, store=store, dry_run=False)
        run = store.get_benchmark_run(run_id)
    finally:
        if engine:
            engine.close()
        warehouse.close()
        server.shutdown()

    assert run["status"] == "completed"
    assert run["summary"]["api_usage_estimate"]["estimated_cost_usd"] == 0.0
    assert any(item["stage"] == "exposure_critic" for item in run["decisions"])
    assert any(item["stage"] == "stage2" for item in run["decisions"])
    assert store.list_memory(model="gemma4:12b", limit=5)


def test_engine_resumes_paused_run_without_duplicate_completed_day(tmp_path):
    server, base_url = _start_fake_chat_server()
    warehouse = Warehouse(tmp_path / "warehouse.duckdb")
    store = BenchmarkStore(tmp_path / "benchmark.db")
    engine = None
    try:
        _insert_market_rows(
            warehouse,
            ["2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09"],
            [100, 100, 105, 110, 120],
        )
        config = local_gemma_aapl_config(
            train_start="2025-01-06",
            train_end="2025-01-07",
            test_start="2025-01-08",
            test_end="2025-01-08",
            local_model_base_url=f"{base_url}/v1",
            max_output_tokens=500,
            monitoring_enabled=False,
            use_cached_llm=False,
        )
        secrets = local_gemma_secret_config(openai_base_url=f"{base_url}/v1", openai_api_key="")
        run_id = "paused-local-run"
        store.create_benchmark_run(run_id, config.model_dump() if hasattr(config, "model_dump") else config.dict())
        store.save_benchmark_decision(
            run_id=run_id,
            phase="training",
            decision_date="2025-01-03",
            fill_date="2025-01-06",
            stage="stage1",
            symbol="AAPL",
            input_payload={},
            output_payload={"analyses": [{"symbol": "AAPL", "stance": "neutral"}], "_api_status": "ok"},
            execution_payload={},
        )
        store.save_benchmark_decision(
            run_id=run_id,
            phase="training",
            decision_date="2025-01-03",
            fill_date="2025-01-06",
            stage="exposure_critic",
            symbol="AAPL",
            input_payload={},
            output_payload={"recommended_exposure_band": [0.0, 0.0], "_api_status": "ok"},
            execution_payload={},
        )
        book = {"cash": 1000.0, "positions": {}, "equity": 1000.0, "long_exposure": 0.0, "short_exposure": 0.0, "gross_exposure": 0.0, "net_exposure": 0.0}
        store.save_benchmark_decision(
            run_id=run_id,
            phase="training",
            decision_date="2025-01-03",
            fill_date="2025-01-06",
            stage="stage2",
            symbol="PORTFOLIO",
            input_payload={},
            output_payload={"target_exposure": 0.0, "expected_holding_days": 1, "target_weights": {"AAPL": 0.0}, "_api_status": "ok"},
            execution_payload={
                "portfolio_before": book,
                "portfolio_after": book,
                "target_weights": {"AAPL": 0.0},
                "executed_target_weights": {"AAPL": 0.0},
                "trades": [],
                "events": [],
                "fees": 0.0,
                "slippage_cost": 0.0,
                "model_failure": False,
                "repair_attempted": False,
                "repair_count": 0,
            },
        )
        store.update_benchmark_run(
            run_id,
            status="paused",
            phase="training",
            started=True,
            progress={"completed_days": 1, "total_days": 3, "model_calls": 3, "message": "Paused by test"},
        )

        engine = BenchmarkEngine(warehouse)
        engine.run(run_id=run_id, config=config, secrets=secrets, store=store, dry_run=False, resume=True)
        run = store.get_benchmark_run(run_id)
    finally:
        if engine:
            engine.close()
        warehouse.close()
        server.shutdown()

    stage2_dates = [item["decision_date"] for item in run["decisions"] if item["stage"] == "stage2"]
    assert run["status"] == "completed"
    assert stage2_dates.count("2025-01-03") == 1
    assert stage2_dates == ["2025-01-03", "2025-01-06", "2025-01-07"]
    assert run["summary"]["model_calls"] >= 10


def _insert_market_rows(warehouse, dates, aapl_prices):
    for index, day in enumerate(dates):
        aapl_price = aapl_prices[index]
        for table, symbol, price in (("context_daily", "SPY", 100 + index), ("asset_daily", "AAPL", aapl_price)):
            warehouse.conn.execute(
                f"""
                INSERT OR REPLACE INTO {table}
                    (date, symbol, market_open, listed, ohlcv_available, open, high, low, close, adj_close, volume, return_1d, source, source_error)
                VALUES
                    (CAST(? AS DATE), ?, true, true, true, ?, ?, ?, ?, ?, 1000, ?, 'test', '')
                """,
                [day, symbol, price, price, price, price, price, 0.01 if index else None],
            )


def _start_fake_chat_server():
    class Handler(BaseHTTPRequestHandler):
        seen_authorization = []

        def do_GET(self):
            if self.path == "/api/tags":
                self._send({"models": [{"name": "gemma4:12b"}]})
                return
            self._send({"data": [{"id": "gemma4:12b"}]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.seen_authorization.append(self.headers.get("Authorization", ""))
            system = (payload.get("messages") or [{}])[0].get("content", "")
            if "analyst stage" in system:
                content = {
                    "analyses": [
                        {
                            "symbol": "AAPL",
                            "stance": "bullish",
                            "confidence": 0.8,
                            "expected_return_bps": 100,
                            "horizon_days": 1,
                            "key_evidence": ["trend"],
                            "memory_refs": [],
                            "uncertainty": [],
                            "proposed_target_weight": 1.0,
                        }
                    ],
                    "market_regime_notes": "firm",
                    "data_quality_notes": [],
                }
            elif "exposure critic" in system:
                content = {
                    "bull_exposure_case": "participate",
                    "defensive_case": "limited",
                    "cash_drag_risk": "cash can lag",
                    "recommended_exposure_band": [1.0, 1.0],
                    "key_disagreement": "none",
                }
            elif "trading memory" in system:
                content = {
                    "summary_lesson": "Full exposure helped during this trend setup.",
                    "lesson_tags": ["participation_helped"],
                    "use_in_future_if": "trend setup repeats",
                    "avoid_if": "trend breaks",
                    "confidence": 0.8,
                }
            elif "single-stock" in system:
                content = {
                    "target_exposure": 1.0,
                    "expected_holding_days": 1,
                    "rebalance_reason": "trend",
                    "input_evidence_refs": ["stage1:AAPL"],
                    "data_quality_warnings_used": [],
                    "confidence": 0.8,
                    "portfolio_thesis": "participate",
                    "major_risks": [],
                    "uncertainty": [],
                    "expected_return_bps": 100,
                    "horizon_days": 1,
                    "cash_drag_justification": "cash can lag",
                    "why_not_buy_hold": "use full exposure",
                    "stage1_alignment": "follow",
                    "stage1_veto_reason": "",
                }
            else:
                content = {"ok": True}
            if self.path == "/api/chat":
                self._send(
                    {
                        "model": "gemma4:12b",
                        "message": {"role": "assistant", "content": json.dumps(content)},
                        "done": True,
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    }
                )
            else:
                self._send({"choices": [{"message": {"content": json.dumps(content)}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

        def log_message(self, format, *args):
            return

        def _send(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"
