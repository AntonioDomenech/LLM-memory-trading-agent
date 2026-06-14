from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests

from .config_store import DATA_DIR

CACHE_DIR = DATA_DIR / "cache" / "market"


def _cache_path(symbol: str, start: str, end: str) -> Path:
    safe = symbol.replace("^", "idx_").replace("/", "_").upper()
    return CACHE_DIR / f"{safe}_{start}_{end}.csv"


def _download_with_stooq(symbol: str, start: str, end: str) -> pd.DataFrame:
    stooq_symbol = symbol.lower()
    if "." not in stooq_symbol and not stooq_symbol.startswith("^"):
        stooq_symbol = f"{stooq_symbol}.us"
    url = "https://stooq.com/q/d/l/"
    params = {
        "s": stooq_symbol,
        "d1": start.replace("-", ""),
        "d2": end.replace("-", ""),
        "i": "d",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"Stooq returned no daily prices for {symbol}")
    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    return df[["date", "open", "high", "low", "close", "volume"]]


def _download_with_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    end_plus = (datetime.fromisoformat(end) + timedelta(days=1)).date().isoformat()
    raw = yf.download(symbol, start=start, end=end_plus, progress=False, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"yfinance returned no daily prices for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [str(col).lower() for col in raw.columns]
    raw = raw.reset_index().rename(columns={"Date": "date", "adj close": "adj_close"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.date.astype(str)
    columns = ["date", "open", "high", "low", "close", "volume"]
    return raw[columns].dropna(subset=["close"])


def load_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, start, end)
    if path.exists():
        return pd.read_csv(path)
    try:
        df = _download_with_yfinance(symbol, start, end)
        source = "yfinance"
    except Exception:
        df = _download_with_stooq(symbol, start, end)
        source = "stooq"
    df = df.copy()
    df["source"] = source
    df.to_csv(path, index=False)
    return df


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def price_window_start(start: str, extra_days: int = 420) -> str:
    return (datetime.fromisoformat(start) - timedelta(days=extra_days)).date().isoformat()


def trading_dates(symbol: str, start: str, end: str, max_days: int) -> List[str]:
    df = load_prices(symbol, start, end)
    dates = df["date"].astype(str).tolist()
    return dates[: max(1, max_days)]


def market_snapshot(symbol: str, as_of_date: str, start: str) -> Dict[str, Any]:
    df = load_prices(symbol, start, as_of_date)
    df["date"] = df["date"].astype(str)
    df = df[df["date"] <= as_of_date].sort_values("date")
    if df.empty:
        raise ValueError(f"No market data for {symbol} on or before {as_of_date}")
    closes = df["close"].astype(float)
    returns = closes.pct_change()
    row = df.iloc[-1]
    close = float(row["close"])
    previous_close = float(closes.iloc[-2]) if len(closes) > 1 else close

    def trailing_return(days: int) -> float | None:
        if len(closes) <= days:
            return None
        base = float(closes.iloc[-days - 1])
        if base == 0:
            return None
        return close / base - 1.0

    snapshot = {
        "symbol": symbol.upper(),
        "as_of_date": as_of_date,
        "source": row.get("source", "unknown"),
        "open": _safe_float(row.get("open")),
        "high": _safe_float(row.get("high")),
        "low": _safe_float(row.get("low")),
        "close": close,
        "volume": _safe_float(row.get("volume")),
        "previous_close": previous_close,
        "return_1d": close / previous_close - 1.0 if previous_close else None,
        "return_5d": trailing_return(5),
        "return_20d": trailing_return(20),
        "sma_20": _safe_float(closes.tail(20).mean()) if len(closes) >= 5 else None,
        "sma_50": _safe_float(closes.tail(50).mean()) if len(closes) >= 20 else None,
        "volatility_20d_annualized": _safe_float(returns.tail(20).std() * (252 ** 0.5)) if len(returns.dropna()) >= 5 else None,
        "history_points": int(len(df)),
    }
    if snapshot["sma_20"]:
        snapshot["close_vs_sma_20"] = close / snapshot["sma_20"] - 1.0
    if snapshot["sma_50"]:
        snapshot["close_vs_sma_50"] = close / snapshot["sma_50"] - 1.0
    return snapshot


def index_context(symbols: Iterable[str], as_of_date: str, start: str) -> List[Dict[str, Any]]:
    context = []
    for symbol in symbols:
        try:
            context.append(market_snapshot(symbol, as_of_date, start))
        except Exception as exc:
            context.append({"symbol": symbol, "as_of_date": as_of_date, "error": str(exc)})
    return context


def write_json_artifact(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
