from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests

from .store import Warehouse
from .universe import END_DATE, START_DATE, SymbolMeta, all_symbols

PRICE_COLUMNS = [
    "date",
    "symbol",
    "market_open",
    "listed",
    "ohlcv_available",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "return_1d",
    "source",
    "source_error",
]


def _download_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    end_plus = (datetime.fromisoformat(end) + timedelta(days=1)).date().isoformat()
    raw = yf.download(symbol, start=start, end=end_plus, progress=False, auto_adjust=False, actions=False, threads=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no rows for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(col[0]).lower().replace(" ", "_") for col in raw.columns]
    else:
        raw.columns = [str(col).lower().replace(" ", "_") for col in raw.columns]
    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else "date"
    raw = raw.rename(columns={date_col: "date", "adj_close": "adj_close"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.date.astype(str)
    if "adj_close" not in raw.columns:
        raw["adj_close"] = raw.get("close")
    return raw[["date", "open", "high", "low", "close", "adj_close", "volume"]].dropna(subset=["close"])


def _stooq_symbol(symbol: str) -> str:
    if symbol.startswith("^"):
        mapping = {
            "^GSPC": "^spx",
            "^IXIC": "^ndq",
            "^DJI": "^dji",
            "^RUT": "^rut",
            "^VIX": "^vix",
            "^TNX": "10usy.b",
        }
        return mapping.get(symbol.upper(), symbol.lower())
    return f"{symbol.lower().replace('-', '.').replace('/', '.')}.us"


def _download_stooq(symbol: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "s": _stooq_symbol(symbol),
        "d1": start.replace("-", ""),
        "d2": end.replace("-", ""),
        "i": "d",
    }
    resp = requests.get("https://stooq.com/q/d/l/", params=params, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"Stooq returned no rows for {symbol}")
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["adj_close"] = df["close"]
    return df[["date", "open", "high", "low", "close", "adj_close", "volume"]].dropna(subset=["close"])


def download_price_history(symbol: str, start: str = START_DATE, end: str = END_DATE) -> tuple[pd.DataFrame, str, str]:
    errors: List[str] = []
    try:
        return _download_yfinance(symbol, start, end), "yfinance", ""
    except Exception as exc:
        errors.append(f"yfinance: {exc}")
    try:
        return _download_stooq(symbol, start, end), "stooq", "; ".join(errors)
    except Exception as exc:
        errors.append(f"stooq: {exc}")
    raise RuntimeError("; ".join(errors))


def _calendar_frame(warehouse: Warehouse) -> pd.DataFrame:
    df = warehouse.conn.execute("SELECT date FROM calendar_daily ORDER BY date").fetchdf()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _build_daily_panel(warehouse: Warehouse, meta: SymbolMeta, raw: pd.DataFrame, source: str, source_error: str = "") -> pd.DataFrame:
    calendar = _calendar_frame(warehouse)
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    raw = raw.sort_values("date").drop_duplicates("date", keep="last")
    raw["return_1d"] = raw["close"].astype(float).pct_change()

    first_date = raw["date"].min() if not raw.empty else None
    last_date = raw["date"].max() if not raw.empty else None
    panel = calendar.merge(raw, on="date", how="left")
    panel["symbol"] = meta.symbol
    panel["market_open"] = panel["close"].notna()
    panel["ohlcv_available"] = panel["close"].notna()
    panel["listed"] = False
    if first_date is not None and last_date is not None:
        panel["listed"] = (panel["date"] >= first_date) & (panel["date"] <= last_date)
    panel["source"] = source
    panel["source_error"] = source_error
    return panel[PRICE_COLUMNS]


def _empty_panel(warehouse: Warehouse, meta: SymbolMeta, error: str) -> pd.DataFrame:
    panel = _calendar_frame(warehouse)
    panel["symbol"] = meta.symbol
    panel["market_open"] = False
    panel["listed"] = False
    panel["ohlcv_available"] = False
    for col in ("open", "high", "low", "close", "adj_close", "volume", "return_1d"):
        panel[col] = None
    panel["source"] = ""
    panel["source_error"] = error
    return panel[PRICE_COLUMNS]


def download_prices(warehouse: Warehouse, symbols: Iterable[SymbolMeta] | None = None, *, start: str = START_DATE, end: str = END_DATE, force: bool = False) -> Dict[str, Any]:
    symbols = list(symbols or all_symbols())
    summary = {"symbols": len(symbols), "ok": 0, "error": 0, "rows": 0, "errors": []}
    for meta in symbols:
        table = "asset_daily" if meta.kind == "stock" else "context_daily"
        existing = warehouse.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE symbol = ?", [meta.symbol]).fetchone()[0]
        if existing and not force:
            summary["ok"] += 1
            summary["rows"] += int(existing)
            warehouse.log("prices", "cache", "ok", symbol=meta.symbol, bucket_start=start, bucket_end=end, message=f"Skipped existing {existing} rows")
            continue
        try:
            raw, source, source_error = download_price_history(meta.yahoo_symbol, start, end)
            panel = _build_daily_panel(warehouse, meta, raw, source, source_error)
            first_available = raw["date"].min() if not raw.empty else None
            last_available = raw["date"].max() if not raw.empty else None
            warehouse.upsert_frame(table, panel, ["date", "symbol"])
            warehouse.update_symbol_price_dates(meta.symbol, first_available, last_available)
            warehouse.log("prices", source, "ok", symbol=meta.symbol, bucket_start=start, bucket_end=end, message=f"{len(raw)} trading rows; {len(panel)} calendar rows")
            summary["ok"] += 1
            summary["rows"] += len(panel)
        except Exception as exc:
            panel = _empty_panel(warehouse, meta, str(exc))
            warehouse.upsert_frame(table, panel, ["date", "symbol"])
            warehouse.log("prices", "all", "error", symbol=meta.symbol, bucket_start=start, bucket_end=end, message=str(exc))
            summary["error"] += 1
            summary["rows"] += len(panel)
            summary["errors"].append({"symbol": meta.symbol, "message": str(exc)})
    warehouse.export_parquet(["symbols", "asset_daily", "context_daily", "download_log"])
    return summary
