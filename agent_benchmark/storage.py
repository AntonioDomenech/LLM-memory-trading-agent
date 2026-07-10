from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_store import DATA_DIR

DB_PATH = DATA_DIR / "benchmark.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BenchmarkStore:
    def __init__(self, path: Path | str = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    execution_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    preset TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    decision_date TEXT NOT NULL,
                    fill_date TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    symbol TEXT,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    execution_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    memory_namespace TEXT NOT NULL DEFAULT 'legacy',
                    policy_fingerprint TEXT NOT NULL DEFAULT 'legacy',
                    base_snapshot_id TEXT NOT NULL DEFAULT '',
                    memory_layer TEXT NOT NULL DEFAULT 'legacy',
                    online_stream_id TEXT NOT NULL DEFAULT '',
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
                    state_features_json TEXT NOT NULL DEFAULT '{}',
                    counterfactual_outcomes_json TEXT NOT NULL DEFAULT '{}',
                    pending_experience_id INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            # SQLite's CREATE TABLE IF NOT EXISTS does not evolve an existing
            # table.  Additive migration keeps historical benchmark databases
            # readable while marking their rows as explicitly legacy-scoped.
            self._ensure_columns(
                conn,
                "benchmark_memory",
                {
                    "memory_namespace": "TEXT NOT NULL DEFAULT 'legacy'",
                    "policy_fingerprint": "TEXT NOT NULL DEFAULT 'legacy'",
                    "base_snapshot_id": "TEXT NOT NULL DEFAULT ''",
                    "memory_layer": "TEXT NOT NULL DEFAULT 'legacy'",
                    "online_stream_id": "TEXT NOT NULL DEFAULT ''",
                    "state_features_json": "TEXT NOT NULL DEFAULT '{}'",
                    "counterfactual_outcomes_json": "TEXT NOT NULL DEFAULT '{}'",
                    "pending_experience_id": "INTEGER",
                },
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_benchmark_memory_scope_time
                ON benchmark_memory (
                    model, mode, memory_namespace, policy_fingerprint,
                    base_snapshot_id, memory_layer, online_stream_id,
                    knowledge_timestamp
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_pending_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_namespace TEXT NOT NULL,
                    policy_fingerprint TEXT NOT NULL,
                    base_snapshot_id TEXT NOT NULL,
                    online_stream_id TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    portfolio_scope TEXT NOT NULL,
                    symbol TEXT,
                    decision_timestamp TEXT NOT NULL,
                    outcome_available_at TEXT NOT NULL,
                    outcome_horizon TEXT NOT NULL,
                    chosen_action TEXT NOT NULL,
                    state_features_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    matured_at TEXT,
                    matured_memory_id INTEGER,
                    counterfactual_outcomes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_experience_identity
                ON benchmark_pending_experiences (
                    memory_namespace, policy_fingerprint, base_snapshot_id,
                    online_stream_id, symbol, decision_timestamp, outcome_horizon
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pending_experience_due
                ON benchmark_pending_experiences (
                    memory_namespace, policy_fingerprint, base_snapshot_id,
                    online_stream_id, status, outcome_available_at
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_pending_experience
                ON benchmark_memory (pending_experience_id)
                WHERE pending_experience_id IS NOT NULL
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_live_state (
                    memory_namespace TEXT NOT NULL,
                    policy_fingerprint TEXT NOT NULL,
                    base_snapshot_id TEXT NOT NULL,
                    online_stream_id TEXT NOT NULL,
                    portfolio_json TEXT NOT NULL,
                    schedule_state_json TEXT NOT NULL,
                    last_snapshot_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        memory_namespace, policy_fingerprint,
                        base_snapshot_id, online_stream_id
                    )
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_base_snapshots (
                    memory_namespace TEXT NOT NULL,
                    base_snapshot_id TEXT NOT NULL,
                    feature_schema_version TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_namespace, base_snapshot_id, feature_schema_version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
        existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for name, declaration in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    def save_run(self, run_id: str, config: Dict[str, Any], summary: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (id, created_at, config_json, summary_json) VALUES (?, ?, ?, ?)",
                (run_id, utc_now(), json.dumps(config), json.dumps(summary)),
            )

    def save_decision(
        self,
        *,
        run_id: str,
        date: str,
        symbol: str,
        model: str,
        input_bundle: Dict[str, Any],
        decision: Dict[str, Any],
        execution: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions
                    (run_id, date, symbol, model, input_json, decision_json, execution_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    date,
                    symbol,
                    model,
                    json.dumps(input_bundle, default=str),
                    json.dumps(decision, default=str),
                    json.dumps(execution, default=str),
                ),
            )

    def list_runs(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, config_json, summary_json FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "config": json.loads(row["config_json"]),
                "summary": json.loads(row["summary_json"]),
            }
            for row in rows
        ]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            run = conn.execute(
                "SELECT id, created_at, config_json, summary_json FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            rows = conn.execute(
                """
                SELECT date, symbol, model, input_json, decision_json, execution_json
                FROM decisions
                WHERE run_id = ?
                ORDER BY date ASC, id ASC
                """,
                (run_id,),
            ).fetchall()
        if not run:
            return None
        return {
            "id": run["id"],
            "created_at": run["created_at"],
            "config": json.loads(run["config_json"]),
            "summary": json.loads(run["summary_json"]),
            "decisions": [
                {
                    "date": row["date"],
                    "symbol": row["symbol"],
                    "model": row["model"],
                    "input": json.loads(row["input_json"]),
                    "decision": json.loads(row["decision_json"]),
                    "execution": json.loads(row["execution_json"]),
                }
                for row in rows
            ],
        }

    def recent_memory(self, symbol: str, before_date: str, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT date, model, decision_json, execution_json
                FROM decisions
                WHERE symbol = ? AND date < ?
                ORDER BY date DESC, id DESC
                LIMIT ?
                """,
                (symbol.upper(), before_date, limit),
            ).fetchall()
        return [
            {
                "date": row["date"],
                "model": row["model"],
                "decision": json.loads(row["decision_json"]),
                "execution": json.loads(row["execution_json"]),
            }
            for row in rows
        ]

    def create_benchmark_run(self, run_id: str, config: Dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_runs
                    (id, created_at, updated_at, status, phase, model, mode, preset, config_json, progress_json, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    now,
                    now,
                    "queued",
                    "queued",
                    config.get("model") or "unselected-model",
                    config.get("mode") or "single_stock",
                    config.get("run_preset") or "",
                    json.dumps(config),
                    json.dumps({"percent": 0, "message": "Queued"}),
                    json.dumps({}),
                ),
            )

    def update_benchmark_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        progress: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        fields = ["updated_at = ?"]
        values: List[Any] = [utc_now()]
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if phase is not None:
            fields.append("phase = ?")
            values.append(phase)
        if progress is not None:
            fields.append("progress_json = ?")
            values.append(json.dumps(progress, default=str))
        if summary is not None:
            fields.append("summary_json = ?")
            values.append(json.dumps(summary, default=str))
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if started:
            fields.append("started_at = COALESCE(started_at, ?)")
            values.append(utc_now())
        if finished:
            fields.append("finished_at = ?")
            values.append(utc_now())
        values.append(run_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE benchmark_runs SET {', '.join(fields)} WHERE id = ?", values)

    def append_benchmark_event(self, run_id: str, phase: str, event_type: str, payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_events (run_id, created_at, phase, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, utc_now(), phase, event_type, json.dumps(payload, default=str)),
            )

    def save_benchmark_report(self, run_id: str, report_type: str, report: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_reports (run_id, report_type, report_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, report_type, json.dumps(report, default=str), utc_now()),
            )

    def latest_benchmark_report(self, run_id: str, report_type: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT report_json, created_at
                FROM benchmark_reports
                WHERE run_id = ? AND report_type = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (run_id, report_type),
            ).fetchone()
        if not row:
            return None
        report = json.loads(row["report_json"] or "{}")
        report.setdefault("created_at", row["created_at"])
        return report

    def mark_benchmark_run_diagnostic(self, run_id: str, reason: str) -> Optional[Dict[str, Any]]:
        run = self.get_benchmark_run(run_id)
        if not run:
            return None
        summary = dict(run.get("summary") or {})
        summary["official_status"] = "diagnostic"
        summary["diagnostic_reason"] = reason
        self.update_benchmark_run(run_id, summary=summary)
        self.append_benchmark_event(run_id, "diagnostic", "marked_diagnostic", {"reason": reason})
        return self.get_benchmark_run(run_id)

    def save_benchmark_decision(
        self,
        *,
        run_id: str,
        phase: str,
        decision_date: str,
        fill_date: str,
        stage: str,
        symbol: str = "",
        input_payload: Dict[str, Any],
        output_payload: Dict[str, Any],
        execution_payload: Dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_decisions
                    (run_id, phase, decision_date, fill_date, stage, symbol, input_json, output_json, execution_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    phase,
                    decision_date,
                    fill_date,
                    stage,
                    symbol,
                    json.dumps(input_payload, default=str),
                    json.dumps(output_payload, default=str),
                    json.dumps(execution_payload, default=str),
                    utc_now(),
                ),
            )

    def add_memory(
        self,
        *,
        model: str,
        mode: str,
        portfolio_scope: str,
        symbol: str = "",
        decision_timestamp: str,
        knowledge_timestamp: str,
        source_run_id: str,
        memory_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
        outcome_horizon: str = "",
        outcome_available_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        memory_namespace: str = "legacy",
        policy_fingerprint: str = "legacy",
        base_snapshot_id: str = "",
        memory_layer: str = "legacy",
        online_stream_id: str = "",
        state_features: Optional[Dict[str, Any]] = None,
        counterfactual_outcomes: Optional[Dict[str, Any]] = None,
        pending_experience_id: Optional[int] = None,
    ) -> int:
        if memory_layer not in {"legacy", "base", "online"}:
            raise ValueError(f"Unsupported memory layer: {memory_layer}")
        if memory_layer == "online" and not online_stream_id:
            raise ValueError("Online memory requires an online_stream_id")
        if memory_layer != "online" and online_stream_id:
            raise ValueError("Only online memory may set online_stream_id")
        with self._connect() as conn:
            values = (
                model,
                mode,
                memory_namespace,
                policy_fingerprint,
                base_snapshot_id,
                memory_layer,
                online_stream_id,
                portfolio_scope,
                symbol,
                decision_timestamp,
                knowledge_timestamp,
                source_run_id,
                memory_type,
                content,
                json.dumps(embedding) if embedding else "",
                outcome_horizon,
                outcome_available_at,
                json.dumps(metadata or {}, default=str),
                json.dumps(state_features or {}, default=str),
                json.dumps(counterfactual_outcomes or {}, default=str),
                pending_experience_id,
                utc_now(),
            )
            try:
                cur = conn.execute(
                    """
                    INSERT INTO benchmark_memory
                        (model, mode, memory_namespace, policy_fingerprint, base_snapshot_id,
                         memory_layer, online_stream_id, portfolio_scope, symbol, decision_timestamp,
                         knowledge_timestamp, source_run_id, memory_type, content, embedding_json,
                         outcome_horizon, outcome_available_at, metadata_json, state_features_json,
                         counterfactual_outcomes_json, pending_experience_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            except sqlite3.IntegrityError:
                if pending_experience_id is None:
                    raise
                existing = conn.execute(
                    "SELECT id FROM benchmark_memory WHERE pending_experience_id = ?",
                    (pending_experience_id,),
                ).fetchone()
                if existing is None:
                    raise
                return int(existing["id"])
            return int(cur.lastrowid)

    def eligible_memory(
        self,
        *,
        model: str,
        mode: str,
        before_or_at: str,
        limit: Optional[int] = 200,
        memory_namespace: Optional[str] = None,
        policy_fingerprint: Optional[str] = None,
        base_snapshot_id: Optional[str] = None,
        online_stream_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where = [
            "model = ?",
            "mode = ?",
            "datetime(knowledge_timestamp) <= datetime(?)",
            "(COALESCE(outcome_available_at, '') = '' OR datetime(outcome_available_at) <= datetime(?))",
        ]
        params: List[Any] = [model, mode, before_or_at, before_or_at]
        if memory_namespace is not None:
            where.append("memory_namespace = ?")
            params.append(memory_namespace)
        if policy_fingerprint is not None:
            where.append("policy_fingerprint = ?")
            params.append(policy_fingerprint)
        if base_snapshot_id is not None:
            where.append("base_snapshot_id = ?")
            params.append(base_snapshot_id)
            if online_stream_id:
                where.append("(memory_layer = 'base' OR (memory_layer = 'online' AND online_stream_id = ?))")
                params.append(online_stream_id)
            else:
                where.append("memory_layer = 'base'")
        elif online_stream_id is not None:
            where.append("memory_layer = 'online' AND online_stream_id = ?")
            params.append(online_stream_id)
        limit_sql = ""
        if limit is not None:
            if limit <= 0:
                return []
            limit_sql = "LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, model, mode, memory_namespace, policy_fingerprint, base_snapshot_id,
                       memory_layer, online_stream_id, portfolio_scope, symbol, decision_timestamp,
                       knowledge_timestamp, source_run_id, memory_type, content, embedding_json,
                       outcome_horizon, outcome_available_at, metadata_json, state_features_json,
                       counterfactual_outcomes_json, pending_experience_id, created_at
                FROM benchmark_memory
                WHERE {' AND '.join(where)}
                ORDER BY knowledge_timestamp DESC, id DESC
                {limit_sql}
                """,
                params,
            ).fetchall()
        return [self._memory_row_to_dict(row, include_embedding=True) for row in rows]

    def list_memory(self, *, model: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        params: List[Any] = []
        where = ""
        if model:
            where = "WHERE model = ?"
            params.append(model)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, model, mode, memory_namespace, policy_fingerprint, base_snapshot_id,
                       memory_layer, online_stream_id, portfolio_scope, symbol, decision_timestamp,
                       knowledge_timestamp, source_run_id, memory_type, content, outcome_horizon,
                       outcome_available_at, metadata_json, state_features_json,
                       counterfactual_outcomes_json, pending_experience_id, created_at
                FROM benchmark_memory
                {where}
                ORDER BY knowledge_timestamp DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._memory_row_to_dict(row) for row in rows]

    @staticmethod
    def _memory_row_to_dict(row: sqlite3.Row, *, include_embedding: bool = False) -> Dict[str, Any]:
        item = {
            "id": row["id"],
            "model": row["model"],
            "mode": row["mode"],
            "memory_namespace": row["memory_namespace"],
            "policy_fingerprint": row["policy_fingerprint"],
            "base_snapshot_id": row["base_snapshot_id"],
            "memory_layer": row["memory_layer"],
            "online_stream_id": row["online_stream_id"],
            "portfolio_scope": row["portfolio_scope"],
            "symbol": row["symbol"],
            "decision_timestamp": row["decision_timestamp"],
            "knowledge_timestamp": row["knowledge_timestamp"],
            "source_run_id": row["source_run_id"],
            "memory_type": row["memory_type"],
            "content": row["content"],
            "outcome_horizon": row["outcome_horizon"],
            "outcome_available_at": row["outcome_available_at"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "state_features": json.loads(row["state_features_json"] or "{}"),
            "counterfactual_outcomes": json.loads(row["counterfactual_outcomes_json"] or "{}"),
            "pending_experience_id": row["pending_experience_id"],
            "created_at": row["created_at"],
        }
        if include_embedding:
            item["embedding"] = json.loads(row["embedding_json"]) if row["embedding_json"] else []
        return item

    def add_pending_experience(
        self,
        *,
        memory_namespace: str,
        policy_fingerprint: str,
        base_snapshot_id: str,
        online_stream_id: str,
        source_run_id: str,
        portfolio_scope: str,
        symbol: str,
        decision_timestamp: str,
        outcome_available_at: str,
        outcome_horizon: str,
        chosen_action: str,
        state_features: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Durably queue an outcome once; replay/resume calls are idempotent."""

        if not online_stream_id:
            raise ValueError("Pending experience requires an online_stream_id")
        now = utc_now()
        identity = (
            memory_namespace,
            policy_fingerprint,
            base_snapshot_id,
            online_stream_id,
            symbol,
            decision_timestamp,
            outcome_horizon,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO benchmark_pending_experiences
                    (memory_namespace, policy_fingerprint, base_snapshot_id, online_stream_id,
                     source_run_id, portfolio_scope, symbol, decision_timestamp,
                     outcome_available_at, outcome_horizon, chosen_action, state_features_json,
                     metadata_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    *identity[:4],
                    source_run_id,
                    portfolio_scope,
                    symbol,
                    decision_timestamp,
                    outcome_available_at,
                    outcome_horizon,
                    chosen_action,
                    json.dumps(state_features or {}, default=str),
                    json.dumps(metadata or {}, default=str),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT id
                FROM benchmark_pending_experiences
                WHERE memory_namespace = ? AND policy_fingerprint = ? AND base_snapshot_id = ?
                  AND online_stream_id = ? AND symbol = ? AND decision_timestamp = ?
                  AND outcome_horizon = ?
                """,
                identity,
            ).fetchone()
        if row is None:  # Defensive: the insert/select identity must agree.
            raise RuntimeError("Unable to persist pending memory experience")
        return int(row["id"])

    def list_due_pending_experiences(
        self,
        *,
        memory_namespace: str,
        policy_fingerprint: str,
        base_snapshot_id: str,
        online_stream_id: str,
        as_of: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, memory_namespace, policy_fingerprint, base_snapshot_id,
                       online_stream_id, source_run_id, portfolio_scope, symbol,
                       decision_timestamp, outcome_available_at, outcome_horizon,
                       chosen_action, state_features_json, metadata_json, status,
                       matured_at, matured_memory_id, counterfactual_outcomes_json,
                       created_at, updated_at
                FROM benchmark_pending_experiences
                WHERE memory_namespace = ? AND policy_fingerprint = ? AND base_snapshot_id = ?
                  AND online_stream_id = ? AND status = 'pending'
                  AND datetime(outcome_available_at) <= datetime(?)
                ORDER BY outcome_available_at ASC, id ASC
                LIMIT ?
                """,
                (
                    memory_namespace,
                    policy_fingerprint,
                    base_snapshot_id,
                    online_stream_id,
                    as_of,
                    limit,
                ),
            ).fetchall()
        return [self._pending_experience_row_to_dict(row) for row in rows]

    def list_recent_pending_experiences(
        self,
        *,
        memory_namespace: str,
        policy_fingerprint: str,
        base_snapshot_id: str,
        online_stream_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return recent scheduled decisions for durable cadence/hysteresis state."""

        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, memory_namespace, policy_fingerprint, base_snapshot_id,
                       online_stream_id, source_run_id, portfolio_scope, symbol,
                       decision_timestamp, outcome_available_at, outcome_horizon,
                       chosen_action, state_features_json, metadata_json, status,
                       matured_at, matured_memory_id, counterfactual_outcomes_json,
                       created_at, updated_at
                FROM benchmark_pending_experiences
                WHERE memory_namespace = ? AND policy_fingerprint = ? AND base_snapshot_id = ?
                  AND online_stream_id = ?
                ORDER BY decision_timestamp DESC, id DESC
                LIMIT ?
                """,
                (
                    memory_namespace,
                    policy_fingerprint,
                    base_snapshot_id,
                    online_stream_id,
                    limit,
                ),
            ).fetchall()
        return [self._pending_experience_row_to_dict(row) for row in rows]

    def mark_pending_experience_matured(
        self,
        experience_id: int,
        *,
        memory_namespace: str,
        policy_fingerprint: str,
        base_snapshot_id: str,
        online_stream_id: str,
        matured_at: str,
        counterfactual_outcomes: Dict[str, Any],
        matured_memory_id: Optional[int] = None,
    ) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE benchmark_pending_experiences
                SET status = 'matured', matured_at = ?, matured_memory_id = ?,
                    counterfactual_outcomes_json = ?, updated_at = ?
                WHERE id = ? AND memory_namespace = ? AND policy_fingerprint = ?
                  AND base_snapshot_id = ? AND online_stream_id = ?
                  AND status = 'pending' AND datetime(outcome_available_at) <= datetime(?)
                """,
                (
                    matured_at,
                    matured_memory_id,
                    json.dumps(counterfactual_outcomes or {}, default=str),
                    utc_now(),
                    experience_id,
                    memory_namespace,
                    policy_fingerprint,
                    base_snapshot_id,
                    online_stream_id,
                    matured_at,
                ),
            )
            return cursor.rowcount == 1

    def get_live_state(
        self,
        *,
        memory_namespace: str,
        policy_fingerprint: str,
        base_snapshot_id: str,
        online_stream_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT portfolio_json, schedule_state_json, last_snapshot_at, updated_at
                FROM benchmark_live_state
                WHERE memory_namespace = ? AND policy_fingerprint = ?
                  AND base_snapshot_id = ? AND online_stream_id = ?
                """,
                (
                    memory_namespace,
                    policy_fingerprint,
                    base_snapshot_id,
                    online_stream_id,
                ),
            ).fetchone()
        if row is None:
            return None
        return {
            "portfolio": json.loads(row["portfolio_json"] or "{}"),
            "schedule_state": json.loads(row["schedule_state_json"] or "{}"),
            "last_snapshot_at": row["last_snapshot_at"],
            "updated_at": row["updated_at"],
        }

    def incompatible_online_stream_fingerprints(
        self,
        *,
        memory_namespace: str,
        base_snapshot_id: str,
        online_stream_id: str,
        policy_fingerprint: str,
    ) -> List[str]:
        """Find state under the same human stream id but a different contract."""

        parameters = (memory_namespace, base_snapshot_id, online_stream_id, policy_fingerprint)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT policy_fingerprint
                FROM (
                    SELECT policy_fingerprint
                    FROM benchmark_live_state
                    WHERE memory_namespace = ? AND base_snapshot_id = ?
                      AND online_stream_id = ? AND policy_fingerprint <> ?
                    UNION
                    SELECT policy_fingerprint
                    FROM benchmark_pending_experiences
                    WHERE memory_namespace = ? AND base_snapshot_id = ?
                      AND online_stream_id = ? AND policy_fingerprint <> ?
                    UNION
                    SELECT policy_fingerprint
                    FROM benchmark_memory
                    WHERE memory_namespace = ? AND base_snapshot_id = ?
                      AND online_stream_id = ? AND policy_fingerprint <> ?
                )
                ORDER BY policy_fingerprint
                """,
                (*parameters, *parameters, *parameters),
            ).fetchall()
        return [str(row[0]) for row in rows if row[0]]

    def online_stream_artifact_counts(
        self,
        *,
        memory_namespace: str,
        base_snapshot_id: str,
        online_stream_id: str,
        policy_fingerprint: str,
    ) -> Dict[str, int]:
        parameters = (
            memory_namespace,
            policy_fingerprint,
            base_snapshot_id,
            online_stream_id,
        )
        with self._connect() as conn:
            pending = conn.execute(
                """
                SELECT COUNT(*) FROM benchmark_pending_experiences
                WHERE memory_namespace = ? AND policy_fingerprint = ?
                  AND base_snapshot_id = ? AND online_stream_id = ?
                """,
                parameters,
            ).fetchone()[0]
            memories = conn.execute(
                """
                SELECT COUNT(*) FROM benchmark_memory
                WHERE memory_namespace = ? AND policy_fingerprint = ?
                  AND base_snapshot_id = ? AND online_stream_id = ?
                """,
                parameters,
            ).fetchone()[0]
        return {"pending_experiences": int(pending), "online_memories": int(memories)}

    def register_base_snapshot(
        self,
        *,
        memory_namespace: str,
        base_snapshot_id: str,
        feature_schema_version: str,
        content_hash: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Freeze a named deterministic base snapshot to its first observed hash."""

        identity = (memory_namespace, base_snapshot_id, feature_schema_version)
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT content_hash, metadata_json, created_at
                FROM benchmark_base_snapshots
                WHERE memory_namespace = ? AND base_snapshot_id = ?
                  AND feature_schema_version = ?
                """,
                identity,
            ).fetchone()
            if existing is None:
                created_at = utc_now()
                conn.execute(
                    """
                    INSERT INTO benchmark_base_snapshots
                        (memory_namespace, base_snapshot_id, feature_schema_version,
                         content_hash, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (*identity, content_hash, json.dumps(metadata or {}, default=str), created_at),
                )
                return {"content_hash": content_hash, "metadata": metadata or {}, "created_at": created_at}
        if str(existing["content_hash"]) != str(content_hash):
            raise ValueError(
                f"Base snapshot {base_snapshot_id!r} changed content: "
                f"registered {existing['content_hash']}, rebuilt {content_hash}. "
                "Bump memory_base_snapshot_id or restore the frozen warehouse."
            )
        existing_metadata = json.loads(existing["metadata_json"] or "{}")
        requested_metadata = metadata or {}
        existing_canonical = json.dumps(existing_metadata, sort_keys=True, separators=(",", ":"), default=str)
        requested_canonical = json.dumps(requested_metadata, sort_keys=True, separators=(",", ":"), default=str)
        if existing_canonical != requested_canonical:
            raise ValueError(
                f"Base snapshot {base_snapshot_id!r} changed cutoff or provenance metadata. "
                "Bump memory_base_snapshot_id or restore the original contract."
            )
        return {
            "content_hash": existing["content_hash"],
            "metadata": existing_metadata,
            "created_at": existing["created_at"],
        }

    def save_live_state(
        self,
        *,
        memory_namespace: str,
        policy_fingerprint: str,
        base_snapshot_id: str,
        online_stream_id: str,
        portfolio: Dict[str, Any],
        schedule_state: Dict[str, Any],
        last_snapshot_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_live_state
                    (memory_namespace, policy_fingerprint, base_snapshot_id,
                     online_stream_id, portfolio_json, schedule_state_json,
                     last_snapshot_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_namespace, policy_fingerprint, base_snapshot_id, online_stream_id)
                DO UPDATE SET
                    portfolio_json = excluded.portfolio_json,
                    schedule_state_json = excluded.schedule_state_json,
                    last_snapshot_at = excluded.last_snapshot_at,
                    updated_at = excluded.updated_at
                """,
                (
                    memory_namespace,
                    policy_fingerprint,
                    base_snapshot_id,
                    online_stream_id,
                    json.dumps(portfolio or {}, default=str),
                    json.dumps(schedule_state or {}, default=str),
                    last_snapshot_at,
                    utc_now(),
                ),
            )

    @staticmethod
    def _pending_experience_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "memory_namespace": row["memory_namespace"],
            "policy_fingerprint": row["policy_fingerprint"],
            "base_snapshot_id": row["base_snapshot_id"],
            "online_stream_id": row["online_stream_id"],
            "source_run_id": row["source_run_id"],
            "portfolio_scope": row["portfolio_scope"],
            "symbol": row["symbol"],
            "decision_timestamp": row["decision_timestamp"],
            "outcome_available_at": row["outcome_available_at"],
            "outcome_horizon": row["outcome_horizon"],
            "chosen_action": row["chosen_action"],
            "state_features": json.loads(row["state_features_json"] or "{}"),
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "status": row["status"],
            "matured_at": row["matured_at"],
            "matured_memory_id": row["matured_memory_id"],
            "counterfactual_outcomes": json.loads(row["counterfactual_outcomes_json"] or "{}"),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_benchmark_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, updated_at, started_at, finished_at, status, phase, model,
                       mode, preset, config_json, progress_json, summary_json, error
                FROM benchmark_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._run_row_to_dict(row) for row in rows]

    def get_benchmark_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            run = conn.execute(
                """
                SELECT id, created_at, updated_at, started_at, finished_at, status, phase, model,
                       mode, preset, config_json, progress_json, summary_json, error
                FROM benchmark_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if not run:
                return None
            decisions = conn.execute(
                """
                SELECT phase, decision_date, fill_date, stage, symbol, input_json, output_json, execution_json, created_at
                FROM benchmark_decisions
                WHERE run_id = ?
                ORDER BY decision_date ASC, id ASC
                """,
                (run_id,),
            ).fetchall()
            events = conn.execute(
                """
                SELECT created_at, phase, event_type, payload_json
                FROM benchmark_events
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        payload = self._run_row_to_dict(run)
        payload["decisions"] = [
            {
                "phase": row["phase"],
                "decision_date": row["decision_date"],
                "fill_date": row["fill_date"],
                "stage": row["stage"],
                "symbol": row["symbol"],
                "input": json.loads(row["input_json"]),
                "output": json.loads(row["output_json"]),
                "execution": json.loads(row["execution_json"]),
                "created_at": row["created_at"],
            }
            for row in decisions
        ]
        payload["events"] = [
            {
                "created_at": row["created_at"],
                "phase": row["phase"],
                "type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in events
        ]
        return payload

    def _run_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "status": row["status"],
            "phase": row["phase"],
            "model": row["model"],
            "mode": row["mode"],
            "preset": row["preset"],
            "config": json.loads(row["config_json"]),
            "progress": json.loads(row["progress_json"] or "{}"),
            "summary": json.loads(row["summary_json"] or "{}"),
            "error": row["error"],
        }
