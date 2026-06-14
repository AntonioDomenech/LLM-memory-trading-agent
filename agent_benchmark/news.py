from __future__ import annotations

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from .config_store import DATA_DIR
from .schemas import BenchmarkConfig, SecretConfig

CACHE_DIR = DATA_DIR / "cache" / "news"
_LAST_CALL: Dict[str, float] = {}


def _respect_rate_limit(source: str, min_seconds: float) -> None:
    now = time.monotonic()
    last = _LAST_CALL.get(source, 0.0)
    wait = min_seconds - (now - last)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[source] = time.monotonic()


def _day_bounds(date_iso: str) -> tuple[str, str]:
    start = datetime.fromisoformat(date_iso)
    end = start + timedelta(days=1)
    return start.isoformat()[:10], end.isoformat()[:10]


def _cache_key(source: str, symbol: str, date_iso: str, query: str = "") -> Path:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / source / f"{symbol.upper()}_{date_iso}_{digest}.json"


def _get_cached(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _set_cached(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _normalize_item(source: str, title: str, url: str = "", published_at: str = "", summary: str = "", raw=None) -> Dict[str, Any]:
    domain = urlparse(url).netloc if url else ""
    return {
        "source": source,
        "title": (title or "").strip(),
        "url": url or "",
        "domain": domain,
        "published_at": published_at or "",
        "summary": (summary or "").strip(),
        "raw": raw or {},
    }


def _dedupe(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        title_key = " ".join((item.get("title") or "").lower().split())
        key = item.get("url") or title_key
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def fetch_gdelt(symbol: str, company_name: str, date_iso: str, limit: int) -> List[Dict[str, Any]]:
    company_query = f'"{company_name}"' if " " in company_name.strip() else company_name.strip()
    query = f"({symbol} OR {company_query})"
    path = _cache_key("gdelt", symbol, date_iso, query)
    cached = _get_cached(path)
    if cached is not None:
        return cached
    start = datetime.fromisoformat(date_iso).strftime("%Y%m%d000000")
    end = (datetime.fromisoformat(date_iso) + timedelta(days=1)).strftime("%Y%m%d000000")
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max(1, min(limit, 50)),
        "sort": "hybridrel",
        "startdatetime": start,
        "enddatetime": end,
    }
    _respect_rate_limit("gdelt", 5.2)
    resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=20)
    if resp.status_code == 429:
        time.sleep(6.0)
        _respect_rate_limit("gdelt", 5.2)
        resp = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=20)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as exc:
        prefix = (resp.text or "").strip()[:240]
        raise RuntimeError(f"GDELT returned non-JSON response: {prefix}") from exc
    items = [
        _normalize_item(
            "gdelt",
            item.get("title", ""),
            item.get("url", ""),
            item.get("seendate", ""),
            item.get("sourcecountry", ""),
            item,
        )
        for item in data.get("articles", [])
    ]
    items = _dedupe(items, limit)
    _set_cached(path, items)
    return items


def fetch_marketaux(symbol: str, date_iso: str, key: str, limit: int) -> List[Dict[str, Any]]:
    if not key:
        return []
    start, end = _day_bounds(date_iso)
    query = f"{symbol}:{start}:{end}"
    path = _cache_key("marketaux", symbol, date_iso, query)
    cached = _get_cached(path)
    if cached is not None:
        return cached
    params = {
        "api_token": key,
        "symbols": symbol.upper(),
        "published_after": start,
        "published_before": end,
        "language": "en",
        "limit": max(1, min(limit, 50)),
    }
    resp = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    items = [
        _normalize_item(
            "marketaux",
            item.get("title", ""),
            item.get("url", ""),
            item.get("published_at", ""),
            item.get("description", ""),
            item,
        )
        for item in data.get("data", [])
    ]
    items = _dedupe(items, limit)
    _set_cached(path, items)
    return items


def fetch_newsapi(symbol: str, company_name: str, date_iso: str, key: str, limit: int) -> List[Dict[str, Any]]:
    if not key:
        return []
    start, end = _day_bounds(date_iso)
    query = f'({symbol} OR "{company_name}")'
    path = _cache_key("newsapi", symbol, date_iso, query)
    cached = _get_cached(path)
    if cached is not None:
        return cached
    params = {
        "apiKey": key,
        "q": query,
        "from": start,
        "to": end,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max(1, min(limit, 100)),
    }
    resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    items = [
        _normalize_item(
            "newsapi",
            item.get("title", ""),
            item.get("url", ""),
            item.get("publishedAt", ""),
            item.get("description", ""),
            item,
        )
        for item in data.get("articles", [])
    ]
    items = _dedupe(items, limit)
    _set_cached(path, items)
    return items


def fetch_finnhub(symbol: str, date_iso: str, key: str, limit: int) -> List[Dict[str, Any]]:
    if not key:
        return []
    path = _cache_key("finnhub", symbol, date_iso, symbol)
    cached = _get_cached(path)
    if cached is not None:
        return cached
    params = {"symbol": symbol.upper(), "from": date_iso, "to": date_iso, "token": key}
    resp = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    items = [
        _normalize_item(
            "finnhub",
            item.get("headline", ""),
            item.get("url", ""),
            datetime.fromtimestamp(item.get("datetime", 0)).isoformat() if item.get("datetime") else "",
            item.get("summary", ""),
            item,
        )
        for item in data
    ]
    items = _dedupe(items, limit)
    _set_cached(path, items)
    return items


def fetch_rss(symbol: str, company_name: str, feeds: List[str], date_iso: str, limit: int) -> List[Dict[str, Any]]:
    if not feeds:
        return []
    path = _cache_key("rss", symbol, date_iso, "|".join(feeds))
    cached = _get_cached(path)
    if cached is not None:
        return cached
    needle = f"{symbol} {company_name}".lower()
    items: List[Dict[str, Any]] = []
    for feed_url in feeds:
        try:
            resp = requests.get(feed_url, timeout=15)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for node in root.findall(".//item"):
                title = "".join(node.findtext("title") or "").strip()
                description = "".join(node.findtext("description") or "").strip()
                if not any(part and part.lower() in f"{title} {description}".lower() for part in needle.split()):
                    continue
                items.append(
                    _normalize_item(
                        "rss",
                        title,
                        node.findtext("link") or "",
                        node.findtext("pubDate") or "",
                        description,
                        {"feed": feed_url},
                    )
                )
        except Exception:
            continue
    items = _dedupe(items, limit)
    _set_cached(path, items)
    return items


def fetch_news_bundle(config: BenchmarkConfig, secrets: SecretConfig, date_iso: str) -> Dict[str, Any]:
    sources = [source.lower() for source in config.data_sources.news_sources]
    limit = max(0, config.data_sources.max_news_per_day)
    if limit <= 0:
        return {"items": [], "source_status": [{"source": "news", "status": "disabled"}]}

    collected: List[Dict[str, Any]] = []
    status: List[Dict[str, str]] = []
    source_limit = max(limit, 3)

    for source in sources:
        before = len(collected)
        try:
            if source == "gdelt":
                collected.extend(fetch_gdelt(config.symbol, config.company_name, date_iso, source_limit))
            elif source == "marketaux":
                collected.extend(fetch_marketaux(config.symbol, date_iso, secrets.marketaux_key, source_limit))
            elif source == "newsapi":
                collected.extend(fetch_newsapi(config.symbol, config.company_name, date_iso, secrets.newsapi_key, source_limit))
            elif source == "finnhub":
                collected.extend(fetch_finnhub(config.symbol, date_iso, secrets.finnhub_key, source_limit))
            elif source == "rss":
                collected.extend(fetch_rss(config.symbol, config.company_name, config.data_sources.rss_feeds, date_iso, source_limit))
            else:
                status.append({"source": source, "status": "unknown"})
                continue
            status.append({"source": source, "status": "ok", "items": str(len(collected) - before)})
        except Exception as exc:
            status.append({"source": source, "status": "error", "message": str(exc)})

    return {"items": _dedupe(collected, limit), "source_status": status}
