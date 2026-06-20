from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import duckdb
import pandas as pd

from ..config_store import DATA_DIR
from .universe import END_DATE, START_DATE, SymbolMeta, all_symbols

WAREHOUSE_DIR = DATA_DIR / "warehouse"
DB_PATH = WAREHOUSE_DIR / "warehouse.duckdb"
PARQUET_DIR = WAREHOUSE_DIR / "parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Warehouse:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.root = self.db_path.parent
        self.parquet_dir = self.root / "parquet"
        self.root.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self.configure_connection()
        self.create_schema()

    def close(self) -> None:
        self.conn.close()

    def configure_connection(self) -> None:
        memory_limit = os.environ.get("BENCHMARK_DUCKDB_MEMORY_LIMIT", "8GB")
        temp_dir = self.root / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        for pragma in (
            f"PRAGMA memory_limit='{memory_limit}'",
            f"PRAGMA temp_directory='{temp_dir.as_posix()}'",
        ):
            try:
                self.conn.execute(pragma)
            except Exception:
                pass

    def reopen(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = duckdb.connect(str(self.db_path))
        self.configure_connection()
        self.create_schema()

    def create_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar_daily (
                date DATE PRIMARY KEY,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                weekday INTEGER,
                is_weekend BOOLEAN
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                kind VARCHAR,
                sector VARCHAR,
                aliases_json VARCHAR,
                yahoo_symbol VARCHAR,
                cik VARCHAR,
                first_price_date DATE,
                last_price_date DATE,
                active BOOLEAN
            )
            """
        )
        for table in ("asset_daily", "context_daily"):
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    date DATE,
                    symbol VARCHAR,
                    market_open BOOLEAN,
                    listed BOOLEAN,
                    ohlcv_available BOOLEAN,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume DOUBLE,
                    return_1d DOUBLE,
                    source VARCHAR,
                    source_error VARCHAR,
                    PRIMARY KEY(date, symbol)
                )
                """
            )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_articles (
                article_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                bucket_start DATE,
                bucket_end DATE,
                interval_start VARCHAR,
                interval_end VARCHAR,
                interval_seconds INTEGER,
                published_at VARCHAR,
                title VARCHAR,
                url VARCHAR,
                domain VARCHAR,
                language VARCHAR,
                source_country VARCHAR,
                source VARCHAR,
                query VARCHAR,
                raw_json VARCHAR
            )
            """
        )
        for column_sql in (
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS interval_start VARCHAR",
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS interval_end VARCHAR",
            "ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS interval_seconds INTEGER",
        ):
            self.conn.execute(column_sql)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_download_state (
                source VARCHAR,
                symbol VARCHAR,
                interval_start VARCHAR,
                interval_end VARCHAR,
                status VARCHAR,
                article_count INTEGER,
                request_count INTEGER,
                split_count INTEGER,
                retry_count INTEGER,
                last_error VARCHAR,
                updated_at VARCHAR,
                PRIMARY KEY(source, symbol, interval_start, interval_end)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sec_facts (
                fact_id VARCHAR PRIMARY KEY,
                symbol VARCHAR,
                cik VARCHAR,
                concept VARCHAR,
                unit VARCHAR,
                value DOUBLE,
                period_start DATE,
                period_end DATE,
                filed_date DATE,
                fiscal_year INTEGER,
                fiscal_period VARCHAR,
                form VARCHAR,
                source VARCHAR
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS macro_daily (
                date DATE,
                series_id VARCHAR,
                label VARCHAR,
                observation_date DATE,
                value DOUBLE,
                source VARCHAR,
                source_status VARCHAR,
                vintage_safe BOOLEAN,
                PRIMARY KEY(date, series_id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS download_log (
                log_id VARCHAR PRIMARY KEY,
                task VARCHAR,
                source VARCHAR,
                symbol VARCHAR,
                bucket_start DATE,
                bucket_end DATE,
                status VARCHAR,
                retry_count INTEGER,
                message VARCHAR,
                created_at VARCHAR
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS coverage_report (
                report_id VARCHAR PRIMARY KEY,
                created_at VARCHAR,
                symbol VARCHAR,
                kind VARCHAR,
                year INTEGER,
                calendar_rows INTEGER,
                available_rows INTEGER,
                market_open_rows INTEGER,
                listed_rows INTEGER,
                missing_listed_market_rows INTEGER,
                suspicious_move_rows INTEGER,
                zero_volume_rows INTEGER,
                status VARCHAR,
                message VARCHAR
            )
            """
        )

    def bootstrap(self, start: str = START_DATE, end: str = END_DATE, symbols: Optional[List[SymbolMeta]] = None) -> Dict[str, Any]:
        dates = pd.date_range(start=start, end=end, freq="D")
        calendar = pd.DataFrame({"date": dates})
        calendar["year"] = calendar["date"].dt.year
        calendar["month"] = calendar["date"].dt.month
        calendar["day"] = calendar["date"].dt.day
        calendar["weekday"] = calendar["date"].dt.weekday
        calendar["is_weekend"] = calendar["weekday"] >= 5
        self.replace_table("calendar_daily", calendar)

        symbol_rows = []
        for item in symbols or all_symbols():
            symbol_rows.append(
                {
                    "symbol": item.symbol,
                    "name": item.name,
                    "kind": item.kind,
                    "sector": item.sector,
                    "aliases_json": json.dumps(item.aliases),
                    "yahoo_symbol": item.yahoo_symbol,
                    "cik": None,
                    "first_price_date": None,
                    "last_price_date": None,
                    "active": True,
                }
            )
        self.replace_table("symbols", pd.DataFrame(symbol_rows))
        self.export_parquet()
        return {"calendar_rows": len(calendar), "symbols": len(symbol_rows)}

    def replace_table(self, table: str, df: pd.DataFrame) -> None:
        self.conn.register("_incoming_df", df)
        self.conn.execute(f"DELETE FROM {table}")
        self.conn.execute(f"INSERT INTO {table} SELECT * FROM _incoming_df")
        self.conn.unregister("_incoming_df")

    def upsert_frame(self, table: str, df: pd.DataFrame, key_cols: Iterable[str]) -> int:
        if df.empty:
            return 0
        columns = [
            row[1]
            for row in self.conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        ]
        for column in columns:
            if column not in df.columns:
                df[column] = None
        df = df[columns]
        tmp = f"_incoming_{table}"
        self.conn.register(tmp, df)
        conditions = " AND ".join([f"{table}.{col} = {tmp}.{col}" for col in key_cols])
        self.conn.execute(f"DELETE FROM {table} USING {tmp} WHERE {conditions}")
        column_sql = ", ".join(columns)
        self.conn.execute(f"INSERT INTO {table} ({column_sql}) SELECT {column_sql} FROM {tmp}")
        self.conn.unregister(tmp)
        return len(df)

    def _parquet_source_sql(self, parquet_paths: Iterable[str | Path]) -> tuple[str, List[Path]]:
        paths = [Path(path) for path in parquet_paths if Path(path).exists()]
        if not paths:
            return "", []
        escaped = [str(path).replace("'", "''") for path in paths]
        if len(escaped) == 1:
            return f"read_parquet('{escaped[0]}')", paths
        path_list = ", ".join(f"'{path}'" for path in escaped)
        return f"read_parquet([{path_list}])", paths

    def insert_or_ignore_parquet(self, table: str, parquet_path: str | Path) -> int:
        return self.insert_or_ignore_parquets(table, [parquet_path])

    def insert_or_ignore_parquets(
        self,
        table: str,
        parquet_paths: Iterable[str | Path],
        *,
        known_count: int | None = None,
    ) -> int:
        source_sql, paths = self._parquet_source_sql(parquet_paths)
        if not paths:
            return 0
        columns = [
            row[1]
            for row in self.conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        ]
        source_columns = {
            row[0]
            for row in self.conn.execute(
                f"DESCRIBE SELECT * FROM {source_sql}"
            ).fetchall()
        }
        select_sql = ", ".join(
            column if column in source_columns else f"NULL AS {column}"
            for column in columns
        )
        column_sql = ", ".join(columns)
        self.conn.execute(
            f"""
            INSERT OR IGNORE INTO {table} ({column_sql})
            SELECT {select_sql}
            FROM {source_sql}
            """
        )
        if known_count is not None:
            return known_count
        return self.conn.execute(f"SELECT COUNT(*) FROM {source_sql}").fetchone()[0]

    def log(self, task: str, source: str, status: str, *, symbol: str = "", bucket_start: str | None = None, bucket_end: str | None = None, message: str = "", retry_count: int = 0) -> None:
        raw = f"{task}|{source}|{symbol}|{bucket_start}|{bucket_end}|{status}|{utc_now()}"
        import hashlib

        row = pd.DataFrame(
            [
                {
                    "log_id": hashlib.sha1(raw.encode("utf-8")).hexdigest(),
                    "task": task,
                    "source": source,
                    "symbol": symbol,
                    "bucket_start": bucket_start,
                    "bucket_end": bucket_end,
                    "status": status,
                    "retry_count": retry_count,
                    "message": message[:2000],
                    "created_at": utc_now(),
                }
            ]
        )
        self.upsert_frame("download_log", row, ["log_id"])

    def already_done(self, task: str, source: str, symbol: str, bucket_start: str, bucket_end: str) -> bool:
        result = self.conn.execute(
            """
            SELECT COUNT(*) FROM download_log
            WHERE task = ? AND source = ? AND symbol = ? AND bucket_start = ? AND bucket_end = ? AND status = 'ok'
            """,
            [task, source, symbol, bucket_start, bucket_end],
        ).fetchone()[0]
        return bool(result)

    def update_symbol_price_dates(self, symbol: str, first_date: str | None, last_date: str | None, cik: str | None = None) -> None:
        self.conn.execute(
            """
            UPDATE symbols
            SET first_price_date = COALESCE(CAST(? AS DATE), first_price_date),
                last_price_date = COALESCE(CAST(? AS DATE), last_price_date),
                cik = COALESCE(?, cik)
            WHERE symbol = ?
            """,
            [first_date, last_date, cik, symbol],
        )

    def export_parquet(self, tables: Optional[Iterable[str]] = None) -> None:
        for table in tables or self.table_names():
            path = self.parquet_dir / f"{table}.parquet"
            self.conn.execute(f"COPY {table} TO ? (FORMAT PARQUET)", [str(path)])

    def table_names(self) -> List[str]:
        rows = self.conn.execute("SHOW TABLES").fetchall()
        return [row[0] for row in rows]

    def status(self) -> Dict[str, Any]:
        tables = {}
        for table in self.table_names():
            tables[table] = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        latest_logs = self.conn.execute(
            """
            SELECT task, source, symbol, status, message, created_at
            FROM download_log
            ORDER BY created_at DESC
            LIMIT 20
            """
        ).fetchdf().to_dict(orient="records")
        return {
            "db_path": str(self.db_path),
            "parquet_dir": str(self.parquet_dir),
            "tables": tables,
            "latest_logs": latest_logs,
        }
