from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

from .store import Warehouse
from .universe import END_DATE, START_DATE, STOCKS, SymbolMeta


COMMON_SHORT_SYMBOLS = {"V", "MA", "MS", "SO", "HD", "KO", "PG"}
BROAD_SINGLE_WORD_TERMS = {
    "apple",
    "amazon",
    "google",
    "meta",
    "facebook",
    "visa",
    "goldman",
    "berkshire",
    "southern",
}


class GdeltRequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code


def _parse_boundary(value: str, *, inclusive_end: bool = False) -> datetime:
    parsed = datetime.fromisoformat(value)
    if len(value) <= 10:
        parsed = datetime.combine(parsed.date(), datetime.min.time())
        if inclusive_end:
            parsed += timedelta(days=1)
    return parsed


def _month_intervals(start: str, end: str) -> List[Tuple[datetime, datetime]]:
    start_dt = _parse_boundary(start)
    end_dt = _parse_boundary(end, inclusive_end=True)
    current = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    buckets: List[Tuple[datetime, datetime]] = []
    while current < end_dt:
        next_month = (current.date().replace(day=28) + timedelta(days=4)).replace(day=1)
        next_month_dt = datetime.combine(next_month, datetime.min.time())
        bucket_start = max(current, start_dt)
        bucket_end = min(next_month_dt, end_dt)
        if bucket_start < bucket_end:
            buckets.append((bucket_start, bucket_end))
        current = next_month_dt
    return buckets


def _gdelt_datetime(moment: datetime) -> str:
    return moment.strftime("%Y%m%d%H%M%S")


def _interval_key(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat()


def _query(meta: SymbolMeta) -> str:
    terms: List[str] = []
    if len(meta.symbol) >= 3 and meta.symbol not in COMMON_SHORT_SYMBOLS:
        terms.append(meta.symbol)
    candidates = [meta.name, *meta.aliases]
    has_phrase = any(" " in term.strip() or "." in term.strip() for term in candidates)
    for term in candidates:
        term = term.strip()
        if not term:
            continue
        is_single_word = " " not in term and "." not in term
        if is_single_word and has_phrase and term.lower() in BROAD_SINGLE_WORD_TERMS:
            continue
        terms.append(term)
    clean = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        clean.append(f'"{term}"' if " " in term or "-" in term else term)
    unique = list(dict.fromkeys(clean))
    if not unique:
        unique = [f'"{meta.name}"' if " " in meta.name else meta.name]
    if len(unique) == 1:
        return unique[0]
    return "(" + " OR ".join(unique) + ")"


def _article_id(symbol: str, item: Dict[str, Any]) -> str:
    raw = "|".join(
        [
            symbol,
            item.get("url", ""),
            item.get("title", ""),
            item.get("seendate", ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalize(symbol: str, interval_start: datetime, interval_end: datetime, query: str, item: Dict[str, Any]) -> Dict[str, Any]:
    url = item.get("url", "") or ""
    return {
        "article_id": _article_id(symbol, item),
        "symbol": symbol,
        "bucket_start": interval_start.date(),
        "bucket_end": (interval_end - timedelta(seconds=1)).date(),
        "interval_start": _interval_key(interval_start),
        "interval_end": _interval_key(interval_end),
        "interval_seconds": int((interval_end - interval_start).total_seconds()),
        "published_at": item.get("seendate", ""),
        "title": item.get("title", ""),
        "url": url,
        "domain": urlparse(url).netloc,
        "language": item.get("language", ""),
        "source_country": item.get("sourcecountry", ""),
        "source": "gdelt",
        "query": query,
        "raw_json": json.dumps(item, ensure_ascii=False),
    }


def _fetch_gdelt(meta: SymbolMeta, interval_start: datetime, interval_end: datetime, *, max_records: int, sleep_seconds: float) -> List[Dict[str, Any]]:
    time.sleep(max(0.0, sleep_seconds))
    query = _query(meta)
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max(1, min(max_records, 250)),
        "sort": "hybridrel",
        "startdatetime": _gdelt_datetime(interval_start),
        "enddatetime": _gdelt_datetime(interval_end),
    }
    try:
        resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=60)
    except requests.RequestException as exc:
        raise GdeltRequestError(str(exc), retryable=True) from exc
    if resp.status_code == 429 or resp.status_code >= 500:
        raise GdeltRequestError(
            f"{resp.status_code} {resp.reason}: {(resp.text or '').strip()[:300]}",
            retryable=True,
            status_code=resp.status_code,
        )
    if resp.status_code >= 400:
        raise GdeltRequestError(
            f"{resp.status_code} {resp.reason}: {(resp.text or '').strip()[:300]}",
            retryable=False,
            status_code=resp.status_code,
        )
    try:
        payload = resp.json()
    except ValueError as exc:
        raise GdeltRequestError((resp.text or "").strip()[:300], retryable=True) from exc
    return [_normalize(meta.symbol, interval_start, interval_end, query, item) for item in payload.get("articles", [])]


def _split_interval(interval_start: datetime, interval_end: datetime) -> List[Tuple[datetime, datetime]]:
    seconds = int((interval_end - interval_start).total_seconds())
    if seconds <= 1:
        return [(interval_start, interval_end)]
    mid = interval_start + timedelta(seconds=seconds // 2)
    return [(interval_start, mid), (mid, interval_end)]


def _state_done(warehouse: Warehouse, meta: SymbolMeta, interval_start: datetime, interval_end: datetime) -> bool:
    result = warehouse.conn.execute(
        """
        SELECT COUNT(*) FROM news_download_state
        WHERE source = 'gdelt' AND symbol = ? AND interval_start = ? AND interval_end = ? AND status = 'ok'
        """,
        [meta.symbol, _interval_key(interval_start), _interval_key(interval_end)],
    ).fetchone()[0]
    return bool(result)


def _record_state(
    warehouse: Warehouse,
    meta: SymbolMeta,
    interval_start: datetime,
    interval_end: datetime,
    *,
    status: str,
    article_count: int = 0,
    request_count: int = 0,
    split_count: int = 0,
    retry_count: int = 0,
    last_error: str = "",
) -> None:
    row = pd.DataFrame(
        [
            {
                "source": "gdelt",
                "symbol": meta.symbol,
                "interval_start": _interval_key(interval_start),
                "interval_end": _interval_key(interval_end),
                "status": status,
                "article_count": article_count,
                "request_count": request_count,
                "split_count": split_count,
                "retry_count": retry_count,
                "last_error": last_error[:2000],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    warehouse.upsert_frame(
        "news_download_state",
        row,
        ["source", "symbol", "interval_start", "interval_end"],
    )


def _download_interval(
    warehouse: Warehouse,
    meta: SymbolMeta,
    interval_start: datetime,
    interval_end: datetime,
    *,
    max_records: int,
    sleep_seconds: float,
    min_interval_seconds: int,
    retry_until_success: bool,
    max_retries: int,
    rate_limit_cooldown_seconds: float,
    max_backoff_seconds: float,
    force: bool,
    depth: int = 0,
) -> Dict[str, Any]:
    start_iso = _interval_key(interval_start)
    end_iso = _interval_key(interval_end)
    start_date = interval_start.date().isoformat()
    end_date = (interval_end - timedelta(seconds=1)).date().isoformat()
    if not force and _state_done(warehouse, meta, interval_start, interval_end):
        return {"status": "skipped", "articles": 0, "requests": 0}

    rows: List[Dict[str, Any]] = []
    retry_count = 0
    cooldown = max(rate_limit_cooldown_seconds, sleep_seconds)
    try:
        while True:
            try:
                rows = _fetch_gdelt(meta, interval_start, interval_end, max_records=max_records, sleep_seconds=sleep_seconds)
                break
            except GdeltRequestError as exc:
                retry_count += 1
                retryable = exc.retryable or retry_until_success
                if not retryable or (not retry_until_success and retry_count > max_retries):
                    raise
                message = f"{exc}; retrying in {int(cooldown)}s"
                warehouse.log("news", "gdelt", "retry", symbol=meta.symbol, bucket_start=start_date, bucket_end=end_date, message=message, retry_count=retry_count)
                _record_state(
                    warehouse,
                    meta,
                    interval_start,
                    interval_end,
                    status="retry",
                    retry_count=retry_count,
                    last_error=str(exc),
                )
                print(
                    json.dumps(
                        {
                            "event": "gdelt_doc_retry",
                            "symbol": meta.symbol,
                            "start": start_iso,
                            "end": end_iso,
                            "retry_count": retry_count,
                            "sleep_seconds": int(cooldown),
                            "message": str(exc),
                        }
                    ),
                    flush=True,
                )
                time.sleep(cooldown)
                cooldown = min(max_backoff_seconds, max(cooldown * 1.7, cooldown + 1.0))

        interval_seconds = int((interval_end - interval_start).total_seconds())
        if len(rows) >= max_records and interval_seconds > min_interval_seconds:
            print(
                json.dumps(
                    {
                        "event": "gdelt_doc_split",
                        "symbol": meta.symbol,
                        "start": start_iso,
                        "end": end_iso,
                        "articles": len(rows),
                        "max_records": max_records,
                        "interval_seconds": interval_seconds,
                        "depth": depth,
                    }
                ),
                flush=True,
            )
            total = {"status": "ok", "articles": 0, "requests": 1, "splits": 1}
            for left, right in _split_interval(interval_start, interval_end):
                result = _download_interval(
                    warehouse,
                    meta,
                    left,
                    right,
                    max_records=max_records,
                    sleep_seconds=sleep_seconds,
                    min_interval_seconds=min_interval_seconds,
                    retry_until_success=retry_until_success,
                    max_retries=max_retries,
                    rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                    force=force,
                    depth=depth + 1,
                )
                if result.get("status") == "error":
                    return result
                total["articles"] += result.get("articles", 0)
                total["requests"] += result.get("requests", 0)
                total["splits"] += result.get("splits", 0)
            _record_state(
                warehouse,
                meta,
                interval_start,
                interval_end,
                status="ok",
                article_count=total["articles"],
                request_count=total["requests"],
                split_count=total["splits"],
                retry_count=retry_count,
            )
            warehouse.log("news", "gdelt", "ok", symbol=meta.symbol, bucket_start=start_date, bucket_end=end_date, message=f"Split saturated interval {start_iso}..{end_iso}; articles={total['articles']}; requests={total['requests']}")
            return total

        if len(rows) >= max_records and interval_seconds <= min_interval_seconds:
            message = f"GDELT cap still reached at minimum interval {start_iso}..{end_iso}; stored capped page only"
            if rows:
                warehouse.upsert_frame("news_articles", pd.DataFrame(rows), ["article_id"])
            _record_state(
                warehouse,
                meta,
                interval_start,
                interval_end,
                status="capped",
                article_count=len(rows),
                request_count=1,
                retry_count=retry_count,
                last_error=message,
            )
            warehouse.log("news", "gdelt", "error", symbol=meta.symbol, bucket_start=start_date, bucket_end=end_date, message=message, retry_count=retry_count)
            return {"status": "error", "articles": len(rows), "requests": 1, "message": message}

        if rows:
            warehouse.upsert_frame("news_articles", pd.DataFrame(rows), ["article_id"])
        _record_state(
            warehouse,
            meta,
            interval_start,
            interval_end,
            status="ok",
            article_count=len(rows),
            request_count=1,
            retry_count=retry_count,
        )
        warehouse.log("news", "gdelt", "ok", symbol=meta.symbol, bucket_start=start_date, bucket_end=end_date, message=f"{len(rows)} articles for {start_iso}..{end_iso}", retry_count=retry_count)
        return {"status": "ok", "articles": len(rows), "requests": 1, "splits": 0}
    except Exception as exc:
        _record_state(
            warehouse,
            meta,
            interval_start,
            interval_end,
            status="error",
            request_count=1,
            retry_count=retry_count,
            last_error=str(exc),
        )
        warehouse.log("news", "gdelt", "error", symbol=meta.symbol, bucket_start=start_date, bucket_end=end_date, message=f"{start_iso}..{end_iso}: {exc}", retry_count=retry_count)
        return {"status": "error", "articles": 0, "requests": 1, "message": str(exc)}


def download_news(
    warehouse: Warehouse,
    symbols: Iterable[SymbolMeta] | None = None,
    *,
    start: str = START_DATE,
    end: str = END_DATE,
    max_months: int | None = None,
    max_records: int = 100,
    sleep_seconds: float = 5.2,
    min_interval_seconds: int = 1,
    retry_until_success: bool = False,
    max_retries: int = 5,
    rate_limit_cooldown_seconds: float = 300.0,
    max_backoff_seconds: float = 3600.0,
    checkpoint_every: int = 25,
    force: bool = False,
) -> Dict[str, Any]:
    symbols = [item for item in list(symbols or STOCKS) if item.kind == "stock"]
    buckets = _month_intervals(start, end)
    summary = {"symbols": len(symbols), "buckets": len(buckets), "ok": 0, "error": 0, "skipped": 0, "articles": 0, "requests": 0, "splits": 0}
    completed_months = 0
    for meta in symbols:
        for interval_start, interval_end in buckets:
            if max_months is not None and completed_months >= max_months:
                warehouse.conn.execute("CHECKPOINT")
                warehouse.export_parquet(["news_download_state", "download_log"])
                return summary
            result = _download_interval(
                warehouse,
                meta,
                interval_start,
                interval_end,
                max_records=max_records,
                sleep_seconds=sleep_seconds,
                min_interval_seconds=min_interval_seconds,
                retry_until_success=retry_until_success,
                max_retries=max_retries,
                rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
                max_backoff_seconds=max_backoff_seconds,
                force=force,
            )
            status = result.get("status")
            if status == "skipped":
                summary["skipped"] += 1
            elif status == "error":
                summary["error"] += 1
            else:
                summary["ok"] += 1
            summary["articles"] += result.get("articles", 0)
            summary["requests"] += result.get("requests", 0)
            summary["splits"] += result.get("splits", 0)
            completed_months += 1
            print(
                json.dumps(
                    {
                        "event": "gdelt_doc_interval",
                        "symbol": meta.symbol,
                        "start": _interval_key(interval_start),
                        "end": _interval_key(interval_end),
                        "status": status,
                        "articles": result.get("articles", 0),
                        "requests": result.get("requests", 0),
                        "completed": completed_months,
                        "total": len(symbols) * len(buckets),
                    }
                ),
                flush=True,
            )
            if checkpoint_every > 0 and completed_months % checkpoint_every == 0:
                warehouse.conn.execute("CHECKPOINT")
                warehouse.export_parquet(["news_download_state", "download_log"])
    warehouse.export_parquet(["news_articles", "news_download_state", "download_log"])
    return summary
