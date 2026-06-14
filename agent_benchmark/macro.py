from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests

DEFAULT_FRED_SERIES = {
    "FEDFUNDS": "Effective federal funds rate",
    "DGS10": "10-year treasury yield",
    "UNRATE": "US unemployment rate",
    "CPIAUCSL": "US CPI all urban consumers",
}


def fetch_fred_series(series_id: str, api_key: str, as_of_date: str) -> Dict[str, Any]:
    start = (datetime.fromisoformat(as_of_date) - timedelta(days=430)).date().isoformat()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": as_of_date,
        "sort_order": "desc",
        "limit": 1,
    }
    resp = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    observations = data.get("observations") or []
    if not observations:
        return {"series_id": series_id, "status": "empty"}
    obs = observations[0]
    value = obs.get("value")
    try:
        value = float(value)
    except Exception:
        pass
    return {"series_id": series_id, "date": obs.get("date"), "value": value, "status": "ok"}


def fred_macro_snapshot(api_key: str, as_of_date: str, series: List[str] | None = None) -> Dict[str, Any]:
    if not api_key:
        return {"source": "fred", "status": "disabled", "message": "FRED requires a free API key."}
    out = []
    for series_id in series or list(DEFAULT_FRED_SERIES):
        try:
            item = fetch_fred_series(series_id, api_key, as_of_date)
            item["label"] = DEFAULT_FRED_SERIES.get(series_id, series_id)
            out.append(item)
        except Exception as exc:
            out.append({"series_id": series_id, "status": "error", "message": str(exc)})
    return {"source": "fred", "status": "ok", "as_of_date": as_of_date, "series": out}
