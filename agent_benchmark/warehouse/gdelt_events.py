from __future__ import annotations

import hashlib
import gc
import json
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

from .store import Warehouse
from .universe import STOCKS, SymbolMeta

EVENT_SOURCE = "gdelt_events"
EVENT_BASE_URL = "http://data.gdeltproject.org/events"

EVENT_COLUMNS = [
    "GlobalEventID",
    "Day",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]

TEXT_COLUMNS = ["Actor1Name", "Actor2Name", "SOURCEURL"]
COMMON_SHORT_SYMBOLS = {"V", "MA", "MS", "SO", "HD", "KO"}


class GdeltArchiveError(RuntimeError):
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


def _interval_key(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat()


def _overlaps(start: datetime, end: datetime, window_start: datetime, window_end: datetime) -> bool:
    return max(start, window_start) < min(end, window_end)


def iter_event_archives(start: str, end: str) -> List[Tuple[str, datetime, datetime, str]]:
    start_dt = _parse_boundary(start)
    end_dt = _parse_boundary(end, inclusive_end=True)
    archives: List[Tuple[str, datetime, datetime, str]] = []

    for year in range(2000, 2006):
        interval_start = datetime(year, 1, 1)
        interval_end = datetime(year + 1, 1, 1)
        if _overlaps(start_dt, end_dt, interval_start, interval_end):
            name = f"{year}.zip"
            archives.append((name, interval_start, interval_end, f"{EVENT_BASE_URL}/{name}"))

    current = datetime(2006, 1, 1)
    while current < datetime(2013, 4, 1):
        next_month = (current.date().replace(day=28) + timedelta(days=4)).replace(day=1)
        interval_end = datetime.combine(next_month, datetime.min.time())
        if _overlaps(start_dt, end_dt, current, interval_end):
            name = f"{current:%Y%m}.zip"
            archives.append((name, current, interval_end, f"{EVENT_BASE_URL}/{name}"))
        current = interval_end

    current = max(datetime(2013, 4, 1), start_dt.replace(hour=0, minute=0, second=0, microsecond=0))
    final = end_dt
    while current < final:
        interval_end = current + timedelta(days=1)
        name = f"{current:%Y%m%d}.export.CSV.zip"
        archives.append((name, current, interval_end, f"{EVENT_BASE_URL}/{name}"))
        current = interval_end

    return archives


def _terms(meta: SymbolMeta) -> List[str]:
    terms = [meta.name, *meta.aliases]
    if len(meta.symbol) >= 3 and meta.symbol not in COMMON_SHORT_SYMBOLS:
        terms.append(meta.symbol)
    return [term.strip().lower() for term in dict.fromkeys(terms) if term.strip()]


def _patterns(symbols: Iterable[SymbolMeta]) -> Dict[str, re.Pattern[str]]:
    patterns = {}
    for meta in symbols:
        escaped = [re.escape(term) for term in _terms(meta)]
        if escaped:
            patterns[meta.symbol] = re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)
    return patterns


def _term_map(symbols: Iterable[SymbolMeta]) -> Dict[str, List[str]]:
    return {meta.symbol: _terms(meta) for meta in symbols if _terms(meta)}


def _patterns_from_terms(term_map: Dict[str, Sequence[str]]) -> Dict[str, re.Pattern[str]]:
    patterns = {}
    for symbol, terms in term_map.items():
        escaped = [re.escape(term) for term in terms if term]
        if escaped:
            patterns[symbol] = re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)
    return patterns


def _archive_state_symbol(symbols: Iterable[str]) -> str:
    raw = ",".join(sorted(symbols))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"__archive__:{digest}"


def _article_id(symbol: str, row: pd.Series) -> str:
    raw = "|".join(
        [
            EVENT_SOURCE,
            symbol,
            str(row.get("GlobalEventID", "")),
            str(row.get("SOURCEURL", "")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _published_at(day: Any) -> str:
    text = str(day or "")
    if len(text) >= 8 and text[:8].isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}T00:00:00"
    return ""


def _event_row(symbol: str, interval_start: datetime, interval_end: datetime, row: pd.Series) -> Dict[str, Any]:
    url = str(row.get("SOURCEURL") or "")
    title = f"GDELT event {row.get('EventCode') or ''}: {row.get('Actor1Name') or ''} / {row.get('Actor2Name') or ''}".strip()
    raw = {
        key: row.get(key)
        for key in [
            "GlobalEventID",
            "Day",
            "Actor1Name",
            "Actor2Name",
            "EventCode",
            "EventBaseCode",
            "EventRootCode",
            "GoldsteinScale",
            "NumMentions",
            "NumSources",
            "NumArticles",
            "AvgTone",
            "SOURCEURL",
        ]
    }
    return {
        "article_id": _article_id(symbol, row),
        "symbol": symbol,
        "bucket_start": interval_start.date(),
        "bucket_end": (interval_end - timedelta(seconds=1)).date(),
        "interval_start": _interval_key(interval_start),
        "interval_end": _interval_key(interval_end),
        "interval_seconds": int((interval_end - interval_start).total_seconds()),
        "published_at": _published_at(row.get("Day")),
        "title": title,
        "url": url,
        "domain": urlparse(url).netloc,
        "language": "",
        "source_country": str(row.get("ActionGeo_CountryCode") or ""),
        "source": EVENT_SOURCE,
        "query": "local archive filter",
        "raw_json": json.dumps(raw, ensure_ascii=False, default=str),
    }


def _record_state(
    warehouse: Warehouse,
    symbol: str,
    interval_start: datetime,
    interval_end: datetime,
    *,
    status: str,
    article_count: int = 0,
    request_count: int = 0,
    retry_count: int = 0,
    last_error: str = "",
) -> None:
    row = pd.DataFrame(
        [
            {
                "source": EVENT_SOURCE,
                "symbol": symbol,
                "interval_start": _interval_key(interval_start),
                "interval_end": _interval_key(interval_end),
                "status": status,
                "article_count": article_count,
                "request_count": request_count,
                "split_count": 0,
                "retry_count": retry_count,
                "last_error": last_error[:2000],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    warehouse.upsert_frame("news_download_state", row, ["source", "symbol", "interval_start", "interval_end"])


def _archive_done(warehouse: Warehouse, state_symbol: str, interval_start: datetime, interval_end: datetime) -> bool:
    result = warehouse.conn.execute(
        """
        SELECT COUNT(*) FROM news_download_state
        WHERE source = ? AND symbol = ? AND interval_start = ? AND interval_end = ?
          AND status IN ('ok', 'missing_file')
        """,
        [EVENT_SOURCE, state_symbol, _interval_key(interval_start), _interval_key(interval_end)],
    ).fetchone()[0]
    return bool(result)


def _download_archive(url: str, path: Path, *, sleep_seconds: float) -> None:
    time.sleep(max(0.0, sleep_seconds))
    tmp_path = path.with_suffix(path.suffix + ".part")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            if response.status_code == 404:
                raise GdeltArchiveError("404 Not Found", retryable=False, status_code=404)
            if response.status_code == 429 or response.status_code >= 500:
                raise GdeltArchiveError(f"{response.status_code} {response.reason}", retryable=True, status_code=response.status_code)
            if response.status_code >= 400:
                raise GdeltArchiveError(f"{response.status_code} {response.reason}", retryable=False, status_code=response.status_code)
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        tmp_path.replace(path)
    except requests.RequestException as exc:
        raise GdeltArchiveError(str(exc), retryable=True) from exc
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _process_archive(
    warehouse: Warehouse,
    archive_path: Path,
    interval_start: datetime,
    interval_end: datetime,
    patterns: Dict[str, re.Pattern[str]],
    *,
    chunksize: int,
) -> int:
    total = 0
    with zipfile.ZipFile(archive_path) as zipped:
        members = [name for name in zipped.namelist() if not name.endswith("/")]
        if not members:
            return 0
        with zipped.open(members[0]) as raw:
            reader = pd.read_csv(
                raw,
                sep="\t",
                header=None,
                names=EVENT_COLUMNS,
                dtype=str,
                chunksize=chunksize,
                on_bad_lines="skip",
                encoding="latin1",
            )
            for chunk in reader:
                text = chunk[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)
                rows: List[Dict[str, Any]] = []
                for symbol, pattern in patterns.items():
                    mask = text.str.contains(pattern, regex=True, na=False)
                    if not mask.any():
                        continue
                    for _, row in chunk.loc[mask].iterrows():
                        rows.append(_event_row(symbol, interval_start, interval_end, row))
                if rows:
                    frame = pd.DataFrame(rows).drop_duplicates(subset=["article_id"])
                    total += warehouse.upsert_frame("news_articles", frame, ["article_id"])
                    del frame
                del rows, text, chunk
                gc.collect()
    return total


def _extract_archive_frame(
    archive_path: Path,
    interval_start: datetime,
    interval_end: datetime,
    term_map: Dict[str, Sequence[str]],
    *,
    chunksize: int,
) -> pd.DataFrame:
    patterns = _patterns_from_terms(term_map)
    rows: List[Dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as zipped:
        members = [name for name in zipped.namelist() if not name.endswith("/")]
        if not members:
            return pd.DataFrame()
        with zipped.open(members[0]) as raw:
            reader = pd.read_csv(
                raw,
                sep="\t",
                header=None,
                names=EVENT_COLUMNS,
                dtype=str,
                chunksize=chunksize,
                on_bad_lines="skip",
                encoding="latin1",
            )
            for chunk in reader:
                text = chunk[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)
                for symbol, pattern in patterns.items():
                    mask = text.str.contains(pattern, regex=True, na=False)
                    if not mask.any():
                        continue
                    for _, row in chunk.loc[mask].iterrows():
                        rows.append(_event_row(symbol, interval_start, interval_end, row))
                del text, chunk
                gc.collect()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["article_id"])


def _parallel_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload["name"]
    interval_start = datetime.fromisoformat(payload["interval_start"])
    interval_end = datetime.fromisoformat(payload["interval_end"])
    raw_dir = Path(payload["raw_dir"])
    stage_dir = Path(payload["stage_dir"])
    archive_path = raw_dir / name
    stage_path = stage_dir / f"{name}.parquet"
    retry_until_success = payload["retry_until_success"]
    max_retries = payload["max_retries"]
    sleep_seconds = payload["sleep_seconds"]
    cooldown = max(payload["rate_limit_cooldown_seconds"], sleep_seconds)
    max_backoff_seconds = payload["max_backoff_seconds"]
    retry_count = 0
    stage_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            if not stage_path.exists():
                if not archive_path.exists():
                    _download_archive(payload["url"], archive_path, sleep_seconds=sleep_seconds)
                frame = _extract_archive_frame(
                    archive_path,
                    interval_start,
                    interval_end,
                    payload["term_map"],
                    chunksize=payload["chunksize"],
                )
                frame.to_parquet(stage_path, index=False)
                article_count = len(frame)
                del frame
                gc.collect()
            else:
                article_count = len(pd.read_parquet(stage_path, columns=["article_id"]))
            if archive_path.exists() and not payload["keep_raw"]:
                archive_path.unlink(missing_ok=True)
            return {
                "status": "ok",
                "archive": name,
                "stage_path": str(stage_path),
                "articles": article_count,
                "retry_count": retry_count,
                "interval_start": payload["interval_start"],
                "interval_end": payload["interval_end"],
            }
        except GdeltArchiveError as exc:
            retry_count += 1
            if exc.status_code == 404:
                return {
                    "status": "missing_file",
                    "archive": name,
                    "articles": 0,
                    "retry_count": retry_count,
                    "message": str(exc),
                    "interval_start": payload["interval_start"],
                    "interval_end": payload["interval_end"],
                }
            if not (exc.retryable or retry_until_success) or (not retry_until_success and retry_count > max_retries):
                return {
                    "status": "error",
                    "archive": name,
                    "articles": 0,
                    "retry_count": retry_count,
                    "message": str(exc),
                    "interval_start": payload["interval_start"],
                    "interval_end": payload["interval_end"],
                }
            time.sleep(cooldown)
            cooldown = min(max_backoff_seconds, max(cooldown * 1.7, cooldown + 1.0))
        except Exception as exc:
            retry_count += 1
            if not retry_until_success and retry_count > max_retries:
                return {
                    "status": "error",
                    "archive": name,
                    "articles": 0,
                    "retry_count": retry_count,
                    "message": str(exc),
                    "interval_start": payload["interval_start"],
                    "interval_end": payload["interval_end"],
                }
            time.sleep(cooldown)
            cooldown = min(max_backoff_seconds, max(cooldown * 1.7, cooldown + 1.0))


def _archive_payload(
    name: str,
    interval_start: datetime,
    interval_end: datetime,
    url: str,
    raw_dir: Path,
    stage_dir: Path,
    term_map: Dict[str, Sequence[str]],
    *,
    sleep_seconds: float,
    retry_until_success: bool,
    max_retries: int,
    rate_limit_cooldown_seconds: float,
    max_backoff_seconds: float,
    chunksize: int,
    keep_raw: bool,
) -> Dict[str, Any]:
    return {
        "name": name,
        "interval_start": _interval_key(interval_start),
        "interval_end": _interval_key(interval_end),
        "url": url,
        "raw_dir": str(raw_dir),
        "stage_dir": str(stage_dir),
        "term_map": term_map,
        "sleep_seconds": sleep_seconds,
        "retry_until_success": retry_until_success,
        "max_retries": max_retries,
        "rate_limit_cooldown_seconds": rate_limit_cooldown_seconds,
        "max_backoff_seconds": max_backoff_seconds,
        "chunksize": chunksize,
        "keep_raw": keep_raw,
    }


def download_gdelt_events(
    warehouse: Warehouse,
    symbols: Iterable[SymbolMeta] | None = None,
    *,
    start: str = "2000-01-01",
    end: str = "2016-12-31",
    sleep_seconds: float = 1.0,
    retry_until_success: bool = False,
    max_retries: int = 5,
    rate_limit_cooldown_seconds: float = 120.0,
    max_backoff_seconds: float = 1800.0,
    checkpoint_every: int = 25,
    chunksize: int = 50_000,
    keep_raw: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    selected = [item for item in list(symbols or STOCKS) if item.kind == "stock"]
    patterns = _patterns(selected)
    state_symbol = _archive_state_symbol(patterns.keys())
    archives = iter_event_archives(start, end)
    raw_dir = warehouse.root / "raw" / EVENT_SOURCE
    summary = {"archives": len(archives), "ok": 0, "missing": 0, "error": 0, "articles": 0, "requests": 0}

    for index, (name, interval_start, interval_end, url) in enumerate(archives, start=1):
        if not force and _archive_done(warehouse, state_symbol, interval_start, interval_end):
            summary["ok"] += 1
            print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "skipped", "index": index, "total": len(archives)}), flush=True)
            continue

        archive_path = raw_dir / name
        retry_count = 0
        cooldown = max(rate_limit_cooldown_seconds, sleep_seconds)
        while True:
            try:
                if not archive_path.exists():
                    _download_archive(url, archive_path, sleep_seconds=sleep_seconds)
                    summary["requests"] += 1
                articles = _process_archive(warehouse, archive_path, interval_start, interval_end, patterns, chunksize=chunksize)
                _record_state(warehouse, state_symbol, interval_start, interval_end, status="ok", article_count=articles, request_count=1, retry_count=retry_count)
                warehouse.log("news", EVENT_SOURCE, "ok", bucket_start=interval_start.date().isoformat(), bucket_end=(interval_end - timedelta(seconds=1)).date().isoformat(), message=f"{name}: {articles} filtered rows", retry_count=retry_count)
                warehouse.conn.execute("CHECKPOINT")
                gc.collect()
                summary["ok"] += 1
                summary["articles"] += articles
                print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "ok", "articles": articles, "index": index, "total": len(archives)}), flush=True)
                break
            except GdeltArchiveError as exc:
                retry_count += 1
                if exc.status_code == 404:
                    _record_state(warehouse, state_symbol, interval_start, interval_end, status="missing_file", request_count=1, retry_count=retry_count, last_error=str(exc))
                    warehouse.log("news", EVENT_SOURCE, "missing_file", bucket_start=interval_start.date().isoformat(), bucket_end=(interval_end - timedelta(seconds=1)).date().isoformat(), message=f"{name}: {exc}", retry_count=retry_count)
                    summary["missing"] += 1
                    print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "missing_file", "index": index, "total": len(archives), "message": str(exc)}), flush=True)
                    break
                if not (exc.retryable or retry_until_success) or (not retry_until_success and retry_count > max_retries):
                    _record_state(warehouse, state_symbol, interval_start, interval_end, status="error", request_count=1, retry_count=retry_count, last_error=str(exc))
                    warehouse.log("news", EVENT_SOURCE, "error", bucket_start=interval_start.date().isoformat(), bucket_end=(interval_end - timedelta(seconds=1)).date().isoformat(), message=f"{name}: {exc}", retry_count=retry_count)
                    summary["error"] += 1
                    print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "error", "index": index, "total": len(archives), "message": str(exc)}), flush=True)
                    break
                warehouse.log("news", EVENT_SOURCE, "retry", bucket_start=interval_start.date().isoformat(), bucket_end=(interval_end - timedelta(seconds=1)).date().isoformat(), message=f"{name}: {exc}; retrying in {int(cooldown)}s", retry_count=retry_count)
                print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "retry", "index": index, "total": len(archives), "retry_count": retry_count, "sleep_seconds": int(cooldown), "message": str(exc)}), flush=True)
                time.sleep(cooldown)
                cooldown = min(max_backoff_seconds, max(cooldown * 1.7, cooldown + 1.0))
            except Exception as exc:
                retry_count += 1
                if not retry_until_success and retry_count > max_retries:
                    _record_state(warehouse, state_symbol, interval_start, interval_end, status="error", request_count=1, retry_count=retry_count, last_error=str(exc))
                    warehouse.log("news", EVENT_SOURCE, "error", bucket_start=interval_start.date().isoformat(), bucket_end=(interval_end - timedelta(seconds=1)).date().isoformat(), message=f"{name}: {exc}", retry_count=retry_count)
                    summary["error"] += 1
                    print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "error", "index": index, "total": len(archives), "message": str(exc)}), flush=True)
                    break
                warehouse.log("news", EVENT_SOURCE, "retry", bucket_start=interval_start.date().isoformat(), bucket_end=(interval_end - timedelta(seconds=1)).date().isoformat(), message=f"{name}: {exc}; retrying in {int(cooldown)}s", retry_count=retry_count)
                print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "retry", "index": index, "total": len(archives), "retry_count": retry_count, "sleep_seconds": int(cooldown), "message": str(exc)}), flush=True)
                time.sleep(cooldown)
                cooldown = min(max_backoff_seconds, max(cooldown * 1.7, cooldown + 1.0))
            finally:
                if archive_path.exists() and not keep_raw:
                    archive_path.unlink(missing_ok=True)

        if checkpoint_every > 0 and index % checkpoint_every == 0:
            warehouse.conn.execute("CHECKPOINT")
            warehouse.export_parquet(["news_download_state", "download_log"])

    warehouse.export_parquet(["news_articles", "news_download_state", "download_log"])
    return summary


def download_gdelt_events_parallel(
    warehouse: Warehouse,
    symbols: Iterable[SymbolMeta] | None = None,
    *,
    start: str = "2000-01-01",
    end: str = "2016-12-31",
    workers: int = 4,
    sleep_seconds: float = 0.5,
    retry_until_success: bool = False,
    max_retries: int = 5,
    rate_limit_cooldown_seconds: float = 120.0,
    max_backoff_seconds: float = 1800.0,
    checkpoint_every: int = 25,
    chunksize: int = 100_000,
    import_batch_size: int = 16,
    keep_raw: bool = False,
    keep_stage: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    selected = [item for item in list(symbols or STOCKS) if item.kind == "stock"]
    terms = _term_map(selected)
    state_symbol = _archive_state_symbol(terms.keys())
    archives = iter_event_archives(start, end)
    raw_dir = warehouse.root / "raw" / EVENT_SOURCE
    stage_dir = warehouse.root / "stage" / EVENT_SOURCE
    summary = {
        "archives": len(archives),
        "scheduled": 0,
        "skipped": 0,
        "ok": 0,
        "missing": 0,
        "error": 0,
        "articles": 0,
        "requests": 0,
        "workers": workers,
    }

    payloads = []
    archive_lookup: Dict[str, Tuple[int, datetime, datetime]] = {}
    for index, (name, interval_start, interval_end, url) in enumerate(archives, start=1):
        archive_lookup[name] = (index, interval_start, interval_end)
        if not force and _archive_done(warehouse, state_symbol, interval_start, interval_end):
            summary["skipped"] += 1
            print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "skipped", "index": index, "total": len(archives)}), flush=True)
            continue
        payloads.append(
            _archive_payload(
                name,
                interval_start,
                interval_end,
                url,
                raw_dir,
                stage_dir,
                terms,
                sleep_seconds=sleep_seconds,
                retry_until_success=retry_until_success,
                max_retries=max_retries,
                rate_limit_cooldown_seconds=rate_limit_cooldown_seconds,
                max_backoff_seconds=max_backoff_seconds,
                chunksize=chunksize,
                keep_raw=keep_raw,
            )
        )

    summary["scheduled"] = len(payloads)
    completed = summary["skipped"]
    import_batch_size = max(1, import_batch_size)
    stage_batch: List[Dict[str, Any]] = []

    def _archive_context(result: Dict[str, Any]) -> Tuple[str, int, datetime, datetime, str, str]:
        name = result["archive"]
        index, interval_start, interval_end = archive_lookup[name]
        start_date = interval_start.date().isoformat()
        end_date = (interval_end - timedelta(seconds=1)).date().isoformat()
        return name, index, interval_start, interval_end, start_date, end_date

    def _checkpoint_if_needed(before: int, after: int) -> None:
        if checkpoint_every > 0 and before // checkpoint_every != after // checkpoint_every:
            warehouse.conn.execute("CHECKPOINT")
            warehouse.export_parquet(["news_download_state", "download_log"])
            gc.collect()

    def _flush_stage_batch() -> None:
        nonlocal completed
        if not stage_batch:
            return
        batch = list(stage_batch)
        stage_batch.clear()
        before_completed = completed
        transaction_started = False
        try:
            parquet_paths: List[Path] = []
            total_article_count = 0
            for result in batch:
                stage_path = Path(result["stage_path"])
                article_count = int(result.get("articles", 0))
                if article_count > 0 and not stage_path.exists():
                    raise FileNotFoundError(f"Missing staged GDELT file: {stage_path}")
                if stage_path.exists() and article_count > 0:
                    parquet_paths.append(stage_path)
                    total_article_count += article_count

            warehouse.conn.execute("BEGIN TRANSACTION")
            transaction_started = True
            if parquet_paths:
                warehouse.insert_or_ignore_parquets("news_articles", parquet_paths, known_count=total_article_count)
            for result in batch:
                name, _, interval_start, interval_end, start_date, end_date = _archive_context(result)
                article_count = int(result.get("articles", 0))
                _record_state(
                    warehouse,
                    state_symbol,
                    interval_start,
                    interval_end,
                    status="ok",
                    article_count=article_count,
                    request_count=1,
                    retry_count=result.get("retry_count", 0),
                )
                warehouse.log(
                    "news",
                    EVENT_SOURCE,
                    "ok",
                    bucket_start=start_date,
                    bucket_end=end_date,
                    message=f"{name}: {article_count} filtered rows",
                    retry_count=result.get("retry_count", 0),
                )
            warehouse.conn.execute("COMMIT")
            transaction_started = False
        except Exception:
            if transaction_started:
                warehouse.conn.execute("ROLLBACK")
            if len(batch) > 1:
                for result in batch:
                    stage_batch.append(result)
                    _flush_stage_batch()
                return
            raise

        for result in batch:
            name, index, _, _, _, _ = _archive_context(result)
            stage_path = Path(result["stage_path"])
            article_count = int(result.get("articles", 0))
            if stage_path.exists() and not keep_stage:
                stage_path.unlink(missing_ok=True)
            summary["ok"] += 1
            summary["articles"] += article_count
            summary["requests"] += 1
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "gdelt_events_archive",
                        "archive": name,
                        "status": "ok",
                        "articles": article_count,
                        "index": index,
                        "completed": completed,
                        "total": len(archives),
                        "workers": workers,
                        "import_batch_size": len(batch),
                    }
                ),
                flush=True,
            )
        _checkpoint_if_needed(before_completed, completed)

    def _record_terminal_result(result: Dict[str, Any]) -> None:
        nonlocal completed
        before_completed = completed
        name, index, interval_start, interval_end, start_date, end_date = _archive_context(result)
        status = result["status"]
        completed += 1
        if status == "missing_file":
            _record_state(warehouse, state_symbol, interval_start, interval_end, status="missing_file", request_count=1, retry_count=result.get("retry_count", 0), last_error=result.get("message", ""))
            warehouse.log("news", EVENT_SOURCE, "missing_file", bucket_start=start_date, bucket_end=end_date, message=f"{name}: {result.get('message', '')}", retry_count=result.get("retry_count", 0))
            summary["missing"] += 1
            summary["requests"] += 1
            print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "missing_file", "index": index, "completed": completed, "total": len(archives), "message": result.get("message", "")}), flush=True)
        else:
            _record_state(warehouse, state_symbol, interval_start, interval_end, status="error", request_count=1, retry_count=result.get("retry_count", 0), last_error=result.get("message", ""))
            warehouse.log("news", EVENT_SOURCE, "error", bucket_start=start_date, bucket_end=end_date, message=f"{name}: {result.get('message', '')}", retry_count=result.get("retry_count", 0))
            summary["error"] += 1
            summary["requests"] += 1
            print(json.dumps({"event": "gdelt_events_archive", "archive": name, "status": "error", "index": index, "completed": completed, "total": len(archives), "message": result.get("message", "")}), flush=True)
        _checkpoint_if_needed(before_completed, completed)

    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_parallel_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            status = result["status"]

            if status == "ok":
                stage_batch.append(result)
                if len(stage_batch) >= import_batch_size:
                    _flush_stage_batch()
            else:
                _flush_stage_batch()
                _record_terminal_result(result)

    _flush_stage_batch()
    warehouse.conn.execute("CHECKPOINT")
    warehouse.export_parquet(["news_articles", "news_download_state", "download_log"])
    return summary
