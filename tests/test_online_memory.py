import sqlite3

import pytest

from agent_benchmark.memory import HybridMemory, build_policy_fingerprint
from agent_benchmark.schemas import BenchmarkConfig, SecretConfig
from agent_benchmark.storage import BenchmarkStore


def _scoped_config(**updates):
    payload = {
        "model": "gemma-test",
        "mode": "single_stock",
        "symbol": "AAPL",
        "memory_mode": "model_specific_cases_and_lessons",
        "memory_retrieval": "structured",
        "memory_namespace": "aapl-online-v1",
        "memory_base_snapshot_id": "aapl-through-2024",
        "memory_policy_version": "long-cash-v1",
        "memory_feature_schema_version": "market-state-v1",
        "allow_short": False,
    }
    payload.update(updates)
    return BenchmarkConfig(**payload)


def _add_lesson(memory, *, content, knowledge_timestamp, layer=None, features=None):
    return memory.add(
        portfolio_scope="single_stock",
        symbol="AAPL",
        decision_timestamp="2024-12-02",
        knowledge_timestamp=knowledge_timestamp,
        source_run_id="snapshot-builder" if layer == "base" else "online-run",
        memory_type="counterfactual_lesson",
        content=content,
        outcome_horizon="20d",
        outcome_available_at=knowledge_timestamp,
        state_features=features or {"momentum": {"return_20d": 0.05}},
        counterfactual_outcomes={
            "long": {"net_return": 0.08},
            "cash": {"net_return": 0.0},
            "short": {"net_return": -0.08},
        },
        memory_layer=layer,
    )


def test_base_snapshot_is_shared_but_backtest_overlays_are_run_isolated(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.db")
    config = _scoped_config(memory_online_stream_id="")
    base = HybridMemory(store, config, SecretConfig())
    registered = base.register_base_snapshot(content_hash="sha256:abc", metadata={"cases": 10})
    assert registered["content_hash"] == "sha256:abc"
    assert base.register_base_snapshot(content_hash="sha256:abc", metadata={})["metadata"] == {"cases": 10}
    with pytest.raises(ValueError, match="changed content"):
        base.register_base_snapshot(content_hash="sha256:different", metadata={})
    base.add_base(
        portfolio_scope="single_stock",
        symbol="AAPL",
        decision_timestamp="2024-11-01",
        knowledge_timestamp="2024-12-31",
        source_run_id="snapshot-builder",
        memory_type="counterfactual_lesson",
        content="Reusable pre-2025 base record.",
        state_features={"momentum": {"return_20d": 0.03}},
        counterfactual_outcomes={"long": {"net_return": 0.04}, "cash": {"net_return": 0.0}},
    )

    run_a = HybridMemory(store, config, SecretConfig(), run_id="replay-a")
    run_b = HybridMemory(store, config, SecretConfig(), run_id="replay-b")
    _add_lesson(run_a, content="Run A matured lesson.", knowledge_timestamp="2025-01-20")

    assert run_a.context["online_stream_id"] == "replay-a"
    assert [item["content"] for item in run_a.retrieve(decision_timestamp="2025-01-19")] == [
        "Reusable pre-2025 base record."
    ]
    assert {item["content"] for item in run_a.retrieve(decision_timestamp="2025-01-20")} == {
        "Reusable pre-2025 base record.",
        "Run A matured lesson.",
    }
    assert [item["content"] for item in run_b.retrieve(decision_timestamp="2025-02-01")] == [
        "Reusable pre-2025 base record."
    ]

    incompatible = _scoped_config(memory_policy_version="long-cash-v2")
    assert build_policy_fingerprint(incompatible) != build_policy_fingerprint(config)
    other_policy = HybridMemory(store, incompatible, SecretConfig(), run_id="replay-a")
    assert other_policy.retrieve(decision_timestamp="2025-02-01") == []


def test_structured_retrieval_uses_state_and_preserves_counterfactuals(tmp_path, monkeypatch):
    def fail_if_embedded(*args, **kwargs):
        raise AssertionError("structured memory must not depend on prose embeddings")

    monkeypatch.setattr("agent_benchmark.memory.embed_text", fail_if_embedded)
    store = BenchmarkStore(tmp_path / "benchmark.db")
    config = _scoped_config()
    memory = HybridMemory(store, config, SecretConfig())
    _add_lesson(
        memory,
        content="Text claims this is the closest case, but its numbers are distant.",
        knowledge_timestamp="2024-12-20",
        layer="base",
        features={"momentum": {"return_20d": -0.20}, "regime": "risk_off"},
    )
    _add_lesson(
        memory,
        content="Unrelated prose.",
        knowledge_timestamp="2024-12-10",
        layer="base",
        features={"momentum": {"return_20d": 0.051}, "regime": "risk_on"},
    )
    # The matching historical regime must remain discoverable even when more
    # than the old 200-row candidate cap arrived after it.
    for index in range(205):
        _add_lesson(
            memory,
            content=f"Recent numerical decoy {index}.",
            knowledge_timestamp="2024-12-30",
            layer="base",
            features={"momentum": {"return_20d": -0.30}, "regime": "risk_off"},
        )

    retrieved = memory.retrieve(
        decision_timestamp="2025-01-02",
        query="closest exact prose",
        structured_query={"momentum": {"return_20d": 0.05}, "regime": "risk_on"},
        limit=2,
    )

    assert retrieved[0]["content"] == "Unrelated prose."
    assert retrieved[0]["retrieval_score"] > retrieved[1]["retrieval_score"]
    assert retrieved[0]["state_features"]["momentum"]["return_20d"] == pytest.approx(0.051)
    assert retrieved[0]["counterfactual_outcomes"]["long"]["net_return"] == pytest.approx(0.08)
    assert "embedding" not in retrieved[0]


def test_pending_experience_matures_across_live_snapshot_run_ids(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmark.db")
    config = _scoped_config(memory_online_stream_id="aapl-live")
    first_snapshot = HybridMemory(store, config, SecretConfig(), run_id="live-snapshot-1")
    first_snapshot.save_live_state(
        portfolio={"cash": 0.0, "positions": {"AAPL": 5.0}, "equity": 1000.0},
        schedule_state={"drawdown_60d": -0.03, "call_model": True},
        last_snapshot_at="2026-07-01T13:35:00Z",
    )
    experience_id = first_snapshot.add_pending_experience(
        source_run_id="live-snapshot-1",
        portfolio_scope="single_stock",
        symbol="AAPL",
        decision_timestamp="2026-07-01",
        outcome_available_at="2026-07-29",
        outcome_horizon="20d",
        chosen_action="LONG",
        state_features={"momentum": {"return_20d": 0.04}, "volatility_20d": 0.21},
        metadata={"fill_price": 210.0},
    )
    duplicate_id = first_snapshot.add_pending_experience(
        source_run_id="live-snapshot-resume",
        portfolio_scope="single_stock",
        symbol="AAPL",
        decision_timestamp="2026-07-01",
        outcome_available_at="2026-07-29",
        outcome_horizon="20d",
        chosen_action="LONG",
        state_features={"momentum": {"return_20d": 0.04}, "volatility_20d": 0.21},
    )
    assert duplicate_id == experience_id

    later_snapshot = HybridMemory(store, config, SecretConfig(), run_id="live-snapshot-2")
    assert later_snapshot.context["online_stream_id"] == "aapl-live"
    restored_state = later_snapshot.load_live_state()
    assert restored_state["portfolio"]["positions"] == {"AAPL": 5.0}
    assert restored_state["schedule_state"]["drawdown_60d"] == pytest.approx(-0.03)
    assert later_snapshot.due_pending_experiences(as_of="2026-07-28") == []
    due = later_snapshot.due_pending_experiences(as_of="2026-07-29")
    assert [item["id"] for item in due] == [experience_id]
    assert [item["id"] for item in later_snapshot.recent_experiences()] == [experience_id]
    assert due[0]["source_run_id"] == "live-snapshot-1"
    assert due[0]["chosen_action"] == "LONG"

    counterfactuals = {
        "long": {"gross_return": -0.06, "net_return": -0.061},
        "cash": {"gross_return": 0.0, "net_return": 0.0},
        "short": {"gross_return": 0.06, "net_return": 0.057},
    }
    memory_id = later_snapshot.add(
        portfolio_scope=due[0]["portfolio_scope"],
        symbol=due[0]["symbol"],
        decision_timestamp=due[0]["decision_timestamp"],
        knowledge_timestamp="2026-07-29",
        source_run_id="live-snapshot-2",
        memory_type="counterfactual_lesson",
        content="",
        outcome_horizon=due[0]["outcome_horizon"],
        outcome_available_at="2026-07-29",
        state_features=due[0]["state_features"],
        counterfactual_outcomes=counterfactuals,
        metadata={"chosen_action": due[0]["chosen_action"], "pending_experience_id": experience_id},
        pending_experience_id=experience_id,
    )
    retry_memory_id = later_snapshot.add(
        portfolio_scope=due[0]["portfolio_scope"],
        symbol=due[0]["symbol"],
        decision_timestamp=due[0]["decision_timestamp"],
        knowledge_timestamp="2026-07-29",
        source_run_id="live-snapshot-retry",
        memory_type="counterfactual_lesson",
        outcome_horizon=due[0]["outcome_horizon"],
        outcome_available_at="2026-07-29",
        state_features=due[0]["state_features"],
        counterfactual_outcomes=counterfactuals,
        pending_experience_id=experience_id,
    )
    assert retry_memory_id == memory_id
    assert later_snapshot.mark_experience_matured(
        experience_id,
        matured_at="2026-07-29",
        counterfactual_outcomes=counterfactuals,
        matured_memory_id=memory_id,
    )
    assert later_snapshot.due_pending_experiences(as_of="2026-07-30") == []

    third_snapshot = HybridMemory(store, config, SecretConfig(), run_id="live-snapshot-3")
    learned = third_snapshot.retrieve(
        decision_timestamp="2026-07-30",
        structured_query={"momentum": {"return_20d": 0.04}, "volatility_20d": 0.21},
    )
    assert [item["id"] for item in learned] == [memory_id]
    assert learned[0]["counterfactual_outcomes"] == counterfactuals

    other_stream = HybridMemory(
        store,
        config,
        SecretConfig(),
        run_id="live-snapshot-4",
        online_stream_id="aapl-paper",
    )
    assert other_stream.retrieve(decision_timestamp="2026-07-30") == []


def test_existing_sqlite_memory_table_is_migrated_as_legacy(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE benchmark_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                mode TEXT NOT NULL,
                portfolio_scope TEXT NOT NULL,
                symbol TEXT,
                decision_timestamp TEXT NOT NULL,
                knowledge_timestamp TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT,
                outcome_horizon TEXT,
                outcome_available_at TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO benchmark_memory
                (model, mode, portfolio_scope, symbol, decision_timestamp,
                 knowledge_timestamp, source_run_id, memory_type, content,
                 embedding_json, outcome_horizon, outcome_available_at,
                 metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-model",
                "single_stock",
                "single_stock",
                "AAPL",
                "2024-01-02",
                "2024-01-20",
                "old-run",
                "lesson",
                "Old record remains readable.",
                "",
                "20d",
                "2024-01-20",
                "{}",
                "2024-01-20T00:00:00+00:00",
            ),
        )

    store = BenchmarkStore(path)
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(benchmark_memory)")}
    assert {
        "memory_namespace",
        "policy_fingerprint",
        "base_snapshot_id",
        "memory_layer",
        "online_stream_id",
        "state_features_json",
        "counterfactual_outcomes_json",
        "pending_experience_id",
    } <= columns

    legacy = HybridMemory(
        store,
        BenchmarkConfig(model="legacy-model", memory_retrieval="structured"),
        SecretConfig(),
        run_id="ignored-for-legacy-compatibility",
    )
    retrieved = legacy.retrieve(decision_timestamp="2024-02-01")
    assert legacy.context["online_stream_id"] == ""
    assert [item["content"] for item in retrieved] == ["Old record remains readable."]
    assert retrieved[0]["memory_namespace"] == "legacy"
    assert retrieved[0]["policy_fingerprint"] == "legacy"
    assert retrieved[0]["memory_layer"] == "legacy"
    assert retrieved[0]["state_features"] == {}
    assert retrieved[0]["counterfactual_outcomes"] == {}
