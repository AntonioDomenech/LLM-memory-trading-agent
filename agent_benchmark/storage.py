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
