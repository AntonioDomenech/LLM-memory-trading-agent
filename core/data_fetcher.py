
import os
import pandas as pd
import hashlib
from datetime import datetime
from typing import Dict, Iterable, Tuple

from .logger import get_logger, warn_once
from .config import get_alpaca_keys

log = get_logger()

# ----------------- helpers -----------------

def _cache_path(symbol, start, end):
    key = f"{symbol}_{start}_{end}".encode()
    h = hashlib.md5(key).hexdigest()
    return os.path.join("data", f"cache_{h}.parquet")

def _ensure_parent(path: str):
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)

def _parse_date_any(x: str):
    try:
        return pd.to_datetime(x).to_pydatetime()
    except Exception:
        return None

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make all column labels simple, lower-case strings. Flattens MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(p) for p in tup if p is not None]).strip() for tup in df.columns]
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _standardize_ohlc_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map a variety of vendor column names to open/high/low/close/volume."""
    mapping = {}
    for c in list(df.columns):
        cl = str(c).strip().lower()
        if cl in {"open", "opening", "open price", "apertura"}: mapping[c] = "open"
        elif cl in {"high", "max", "high price", "máximo", "maximo"}: mapping[c] = "high"
        elif cl in {"low", "min", "low price", "mínimo", "minimo"}: mapping[c] = "low"
        elif cl in {"close", "close*", "closing", "last", "cierre"}: mapping[c] = "close"
        elif cl in {"adj close", "adjusted close", "close_adj"} and "close" not in [str(x).lower() for x in df.columns]:
            # use adjusted close if plain close is absent
            mapping[c] = "close"
        elif cl in {"volume", "vol", "volumen"}: mapping[c] = "volume"
        elif cl in {"timestamp", "datetime", "date"}: mapping[c] = cl  # keep as is
    if mapping:
        df = df.rename(columns=mapping)
    return df

def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_cols(df)
    df = _standardize_ohlc_names(df)
    # remove duplicated columns that sometimes appear after rename
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df

def _from_any_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a 'date' column in date (not datetime) dtype."""
    df = df.copy()
    # If there is DatetimeIndex and no explicit date col, pull from index
    if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})
    # Common names already normalized to 'date' or present as 'datetime'/'timestamp'
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None).dt.date
    else:
        # last try: if there is any column that looks like a date
        for cand in df.columns:
            if "date" in str(cand):
                df["date"] = pd.to_datetime(df[cand], errors="coerce").dt.date
                break
    return df

def _clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize arbitrary vendor bar formats to columns:
        date, open, high, low, close, volume
    Rolls intraday to daily if a timestamp column is present.
    """
    df = _dedup_columns(df.copy())

    # If we have per-tick or intraday records, group by date
    if "timestamp" in df.columns or "datetime" in df.columns:
        df = _from_any_date_col(df)
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            # certain vendors give OHLC as uppercase even after our mapping
            uc_map = {c: c.lower() for c in df.columns if str(c).upper() in {"OPEN","HIGH","LOW","CLOSE","VOLUME"}}
            if uc_map:
                df = df.rename(columns=uc_map)
        group_cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in group_cols: agg["volume"] = "sum"
        df = df.groupby("date", as_index=False).agg(agg)

    # If we already have daily rows with some kind of date-like column
    df = _from_any_date_col(df)

    # Final column presence check and coercion
    needed = {"open","high","low","close"}
    if not needed.issubset(df.columns):
        cols = ", ".join(df.columns)
        raise ValueError(f"Unknown bar format (cols=[{cols}])")

    # numeric coercion
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open","high","low","close"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    return _clean_bars(df)

# ----------------- public API -----------------

def get_daily_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    cpath = _cache_path(symbol, start, end)
    if os.path.exists(cpath):
        try:
            return pd.read_parquet(cpath)
        except Exception:
            pass

    key, secret = get_alpaca_keys()
    s_dt = _parse_date_any(start); e_dt = _parse_date_any(end)

    # Try Alpaca first if keys are provided
    if key and secret and s_dt and e_dt:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            client = StockHistoricalDataClient(key, secret)
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                                   start=s_dt, end=e_dt, limit=10000, adjustment='raw')
            bars = client.get_stock_bars(req).df
            if bars is None or (hasattr(bars, "empty") and bars.empty):
                warn_once(log, "alpaca_empty", f"Alpaca returned 0 bars for {symbol} {start}→{end}. Falling back to yfinance.")
                raise RuntimeError("empty")
            df = bars.reset_index()
            df = _normalize(df)
            _ensure_parent(cpath)
            df.to_parquet(cpath, index=False)
            return df
        except Exception as e:
            warn_once(log, "alpaca_fail", f"Alpaca bars failed: {e}. Falling back to yfinance.")

    # yfinance fallback
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance empty")
    # If yfinance returns MultiIndex columns (rare even for single tickers), flatten
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.droplevel(1, axis=1)
        except Exception:
            df.columns = ["_".join([str(p) for p in tup if p is not None]).strip() for tup in df.columns]
    df = df.reset_index()
    df = _normalize(df)
    _ensure_parent(cpath)
    df.to_parquet(cpath, index=False)
    return df
