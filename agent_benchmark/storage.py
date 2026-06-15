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
                CREATE TABLE IF NOT EXISTS benchmark_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

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
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO benchmark_memory
                    (model, mode, portfolio_scope, symbol, decision_timestamp, knowledge_timestamp,
                     source_run_id, memory_type, content, embedding_json, outcome_horizon,
                     outcome_available_at, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model,
                    mode,
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
                    utc_now(),
                ),
            )
            return int(cur.lastrowid)

    def eligible_memory(self, *, model: str, mode: str, before_or_at: str, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, model, mode, portfolio_scope, symbol, decision_timestamp, knowledge_timestamp,
                       source_run_id, memory_type, content, embedding_json, outcome_horizon,
                       outcome_available_at, metadata_json, created_at
                FROM benchmark_memory
                WHERE model = ? AND mode = ? AND knowledge_timestamp <= ?
                ORDER BY knowledge_timestamp DESC, id DESC
                LIMIT ?
                """,
                (model, mode, before_or_at, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "model": row["model"],
                "mode": row["mode"],
                "portfolio_scope": row["portfolio_scope"],
                "symbol": row["symbol"],
                "decision_timestamp": row["decision_timestamp"],
                "knowledge_timestamp": row["knowledge_timestamp"],
                "source_run_id": row["source_run_id"],
                "memory_type": row["memory_type"],
                "content": row["content"],
                "embedding": json.loads(row["embedding_json"]) if row["embedding_json"] else [],
                "outcome_horizon": row["outcome_horizon"],
                "outcome_available_at": row["outcome_available_at"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

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
                SELECT id, model, mode, portfolio_scope, symbol, decision_timestamp, knowledge_timestamp,
                       source_run_id, memory_type, content, outcome_horizon, outcome_available_at,
                       metadata_json, created_at
                FROM benchmark_memory
                {where}
                ORDER BY knowledge_timestamp DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            {
                "id": row["id"],
                "model": row["model"],
                "mode": row["mode"],
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
                "created_at": row["created_at"],
            }
            for row in rows
        ]

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
