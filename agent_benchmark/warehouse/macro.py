from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests

from .store import Warehouse
from .universe import END_DATE, START_DATE

FRED_SERIES = {
    "FEDFUNDS": "Effective federal funds rate",
    "DGS10": "10-year treasury yield",
    "DGS2": "2-year treasury yield",
    "T10Y2Y": "10-year minus 2-year treasury spread",
    "CPIAUCSL": "Consumer price index",
    "UNRATE": "Unemployment rate",
    "GDP": "Gross domestic product",
    "INDPRO": "Industrial production",
    "PAYEMS": "Nonfarm payrolls",
    "BAA10Y": "Moody's BAA minus 10-year treasury spread",
    "DCOILWTICO": "WTI crude oil",
    "DTWEXBGS": "Trade weighted US dollar index",
}


def _calendar(warehouse: Warehouse) -> pd.DataFrame:
    return warehouse.conn.execute("SELECT date FROM calendar_daily ORDER BY date").fetchdf()


def _missing_key_frame(warehouse: Warehouse) -> pd.DataFrame:
    calendar = _calendar(warehouse)
    frames = []
    for series_id, label in FRED_SERIES.items():
        df = calendar.copy()
        df["series_id"] = series_id
        df["label"] = label
        df["observation_date"] = pd.NaT
        df["value"] = None
        df["source"] = "fred"
        df["source_status"] = "missing_key"
        df["vintage_safe"] = False
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _fetch_fred(series_id: str, api_key: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
        "limit": 100000,
    }
    resp = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for obs in data.get("observations", []):
        value = obs.get("value")
        try:
            value = float(value)
        except Exception:
            value = None
        if value is None:
            continue
        rows.append({"observation_date": pd.to_datetime(obs.get("date")).date(), "value": value})
    return pd.DataFrame(rows)


def download_macro(warehouse: Warehouse, *, api_key: str = "", start: str = START_DATE, end: str = END_DATE) -> Dict[str, Any]:
    if not api_key:
        frame = _missing_key_frame(warehouse)
        warehouse.upsert_frame("macro_daily", frame, ["date", "series_id"])
        warehouse.log("macro", "fred", "missing_key", bucket_start=start, bucket_end=end, message="No FRED API key configured")
        warehouse.export_parquet(["macro_daily", "download_log"])
        return {"series": len(FRED_SERIES), "status": "missing_key", "rows": len(frame)}

    calendar = _calendar(warehouse)
    frames: List[pd.DataFrame] = []
    summary = {"series": len(FRED_SERIES), "ok": 0, "error": 0, "rows": 0, "errors": []}
    for series_id, label in FRED_SERIES.items():
        try:
            obs = _fetch_fred(series_id, api_key, start, end)
            if obs.empty:
                raise ValueError("No observations returned")
            df = calendar.copy()
            obs = obs.sort_values("observation_date")
            df = pd.merge_asof(
                df.sort_values("date"),
                obs.sort_values("observation_date"),
                left_on="date",
                right_on="observation_date",
                direction="backward",
            )
            df["series_id"] = series_id
            df["label"] = label
            df["source"] = "fred"
            df["source_status"] = df["value"].notna().map(lambda ok: "ok" if ok else "no_prior_observation")
            df["vintage_safe"] = False
            frames.append(df[["date", "series_id", "label", "observation_date", "value", "source", "source_status", "vintage_safe"]])
            warehouse.log("macro", "fred", "ok", symbol=series_id, bucket_start=start, bucket_end=end, message=f"{len(obs)} observations")
            summary["ok"] += 1
        except Exception as exc:
            summary["error"] += 1
            summary["errors"].append({"series_id": series_id, "message": str(exc)})
            warehouse.log("macro", "fred", "error", symbol=series_id, bucket_start=start, bucket_end=end, message=str(exc))
    if frames:
        frame = pd.concat(frames, ignore_index=True)
        warehouse.upsert_frame("macro_daily", frame, ["date", "series_id"])
        summary["rows"] = len(frame)
    warehouse.export_parquet(["macro_daily", "download_log"])
    return summary
