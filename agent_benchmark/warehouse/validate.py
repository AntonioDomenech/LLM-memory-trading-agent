from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import pandas as pd

from .store import Warehouse, utc_now


def _report_id(symbol: str, kind: str, year: int) -> str:
    return hashlib.sha1(f"{symbol}|{kind}|{year}|{utc_now()}".encode("utf-8")).hexdigest()


def validate_warehouse(warehouse: Warehouse) -> Dict[str, Any]:
    warehouse.conn.execute("DELETE FROM coverage_report")
    calendar_count = warehouse.conn.execute("SELECT COUNT(*) FROM calendar_daily").fetchone()[0]
    symbols = warehouse.conn.execute("SELECT symbol, kind FROM symbols ORDER BY kind, symbol").fetchdf()
    report_rows: List[Dict[str, Any]] = []
    duplicate_rows = 0

    for _, row in symbols.iterrows():
        symbol = row["symbol"]
        kind = row["kind"]
        table = "asset_daily" if kind == "stock" else "context_daily"
        duplicate_rows += warehouse.conn.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT date, symbol, COUNT(*) AS n
                FROM {table}
                WHERE symbol = ?
                GROUP BY date, symbol
                HAVING COUNT(*) > 1
            )
            """,
            [symbol],
        ).fetchone()[0]
        df = warehouse.conn.execute(
            f"""
            SELECT
                d.year,
                COUNT(*) AS calendar_rows,
                SUM(CASE WHEN p.ohlcv_available THEN 1 ELSE 0 END) AS available_rows,
                SUM(CASE WHEN p.market_open THEN 1 ELSE 0 END) AS market_open_rows,
                SUM(CASE WHEN p.listed THEN 1 ELSE 0 END) AS listed_rows,
                SUM(CASE WHEN spy.market_open AND p.listed AND NOT p.ohlcv_available THEN 1 ELSE 0 END) AS missing_listed_market_rows,
                SUM(CASE WHEN ABS(p.return_1d) > 0.35 THEN 1 ELSE 0 END) AS suspicious_move_rows,
                SUM(CASE WHEN p.ohlcv_available AND COALESCE(p.volume, 0) = 0 THEN 1 ELSE 0 END) AS zero_volume_rows
            FROM calendar_daily d
            LEFT JOIN {table} p ON p.date = d.date AND p.symbol = ?
            LEFT JOIN context_daily spy ON spy.date = d.date AND spy.symbol = 'SPY'
            GROUP BY d.year
            ORDER BY d.year
            """,
            [symbol],
        ).fetchdf()
        for _, yr in df.iterrows():
            status = "ok"
            messages = []
            if int(yr["calendar_rows"]) != 366 and int(yr["calendar_rows"]) != 365:
                status = "error"
                messages.append("calendar row count is not 365/366")
            if int(yr["missing_listed_market_rows"] or 0) > 5:
                status = "warn"
                messages.append("listed symbol missing data on market-open days")
            if int(yr["suspicious_move_rows"] or 0) > 0:
                status = "warn"
                messages.append("suspicious daily moves")
            report_rows.append(
                {
                    "report_id": _report_id(symbol, kind, int(yr["year"])),
                    "created_at": utc_now(),
                    "symbol": symbol,
                    "kind": kind,
                    "year": int(yr["year"]),
                    "calendar_rows": int(yr["calendar_rows"]),
                    "available_rows": int(yr["available_rows"] or 0),
                    "market_open_rows": int(yr["market_open_rows"] or 0),
                    "listed_rows": int(yr["listed_rows"] or 0),
                    "missing_listed_market_rows": int(yr["missing_listed_market_rows"] or 0),
                    "suspicious_move_rows": int(yr["suspicious_move_rows"] or 0),
                    "zero_volume_rows": int(yr["zero_volume_rows"] or 0),
                    "status": status,
                    "message": "; ".join(messages),
                }
            )

    if report_rows:
        warehouse.upsert_frame("coverage_report", pd.DataFrame(report_rows), ["report_id"])
    warehouse.log("validate", "warehouse", "ok" if duplicate_rows == 0 else "error", message=f"calendar_rows={calendar_count}; duplicate_price_groups={duplicate_rows}")
    warehouse.export_parquet(["coverage_report", "download_log"])
    status_counts = warehouse.conn.execute("SELECT status, COUNT(*) FROM coverage_report GROUP BY status").fetchall()
    return {
        "calendar_rows": calendar_count,
        "symbols": len(symbols),
        "duplicate_price_groups": int(duplicate_rows),
        "coverage_status": {status: count for status, count in status_counts},
    }
