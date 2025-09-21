import os, json, time, requests
from datetime import datetime, timedelta

# Local store helpers
from .news_store import load_local_day, save_local_day, _is_synth

# Simple logger fallback
def _log(msg):
    """Log ``msg`` using the project logger when available."""

    try:
        from .logger import get_logger
        get_logger().info(msg)
    except Exception:
        print(msg)

CACHE_PATH = os.environ.get("NEWS_CACHE_PATH", "data/news_cache.jsonl")

def _write_cache(entry: dict):
    """Append a JSON encoded cache entry to the local cache file."""

    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _try_request_json(url, params=None, headers=None, timeout=20):
    """Execute a GET request and return ``(json, error_label)``."""

    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        ct = r.headers.get("content-type", "")
        if "json" not in ct:
            return None, "not_json"
        return r.json(), None
    except Exception as e:
        return None, f"error:{e}"

def _day_plus_one(day_iso: str) -> str:
    """Return the ISO date string for the day after ``day_iso``."""

    d = datetime.fromisoformat(day_iso).date()
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")

def _as_str(value):
    """Best-effort conversion of nested structures (dict/list) into readable strings."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("name", "title", "source", "publisher"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return ""
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            s = _as_str(item).strip()
            if s and s not in parts:
                parts.append(s)
        return ", ".join(parts)
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _map_articles(items, mapping_keys, K):
    """Map provider-specific payloads to the canonical article dict format."""

    out = []
    for it in items or []:
        try:
            title = None; source = None; url = None; content = None
            for path in mapping_keys[0]:  # title paths
                cur = it
                for key in path if isinstance(path, list) else [path]:
                    cur = cur.get(key) if isinstance(cur, dict) else None
                if cur: title = cur; break
            for path in mapping_keys[1]:  # source paths
                cur = it
                for key in path if isinstance(path, list) else [path]:
                    cur = cur.get(key) if isinstance(cur, dict) else None
                if cur: source = cur; break
            for path in mapping_keys[2]:  # url paths
                cur = it
                for key in path if isinstance(path, list) else [path]:
                    cur = cur.get(key) if isinstance(cur, dict) else None
                if cur: url = cur; break
            if len(mapping_keys) > 3:
                for path in mapping_keys[3]:  # content paths
                    cur = it
                    for key in path if isinstance(path, list) else [path]:
                        cur = cur.get(key) if isinstance(cur, dict) else None
                    if cur: content = cur; break
            if title and url:
                rec = {
                    "title": title,
                    "source": _as_str(source).strip() if source is not None else "",
                    "url": url,
                }
                if content: rec["content"] = content
                out.append(rec)
                if len(out) >= K:
                    break
        except Exception:
            continue
    return out

ALIASES = {
    "finnhub": "Finnhub",
    "polygon": "Polygon",
    "newsapi": "NewsAPI",
    "marketaux": "MarketAux",
    "fmp": "FMP",
    "bing": "Bing",
}

def get_provider_chain():
    """Return the normalized provider chain configured via environment variables."""

    raw = os.environ.get("NEWS_PROVIDER_CHAIN", "Finnhub,Polygon,NewsAPI")
    chain = []
    for token in raw.split(","):
        t = token.strip()
        if not t or t.startswith("#"):
            continue
        chain.append(ALIASES.get(t.lower(), t))
    return chain

def _newsapi(symbol, day_iso, K):
    """Fetch articles from NewsAPI for ``symbol`` on ``day_iso``."""

    k = os.environ.get("NEWSAPI_KEY") or os.environ.get("NEWS_API_KEY")
    if not k:
        return [], "missing_key"
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol, "from": day_iso, "to": _day_plus_one(day_iso),
        "language": "en", "pageSize": max(K, 50), "sortBy": "relevancy",
        "apiKey": k
    }
    data, err = _try_request_json(url, params=params)
    if err: return [], err
    arts = _map_articles(
        data.get("articles", []),
        (["title"], [["source","name"]], ["url"], ["content"]),
        K
    )
    return arts, "fetched" if arts else "empty"

def _polygon(symbol, day_iso, K):
    """Fetch articles from Polygon.io for ``symbol`` on ``day_iso``."""

    k = os.environ.get("POLYGON_KEY") or os.environ.get("POLYGON_API_KEY")
    if not k:
        return [], "missing_key"
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": symbol, "published_utc.gte": day_iso,
        "published_utc.lt": _day_plus_one(day_iso), "limit": max(50, K),
        "apiKey": k
    }
    data, err = _try_request_json(url, params=params)
    if err: return [], err
    arts = _map_articles(
        data.get("results", []),
        (["title"], [["publisher", "name"], "publisher"], ["article_url"], ["description"]),
        K
    )
    return arts, "fetched" if arts else "empty"

def _finnhub(symbol, day_iso, K):
    """Fetch articles from Finnhub for ``symbol`` on ``day_iso``."""

    k = os.environ.get("FINNHUB_KEY")
    if not k:
        return [], "missing_key"
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": symbol, "from": day_iso, "to": _day_plus_one(day_iso), "token": k}
    data, err = _try_request_json(url, params=params)
    if err: return [], err
    arts = _map_articles(
        data,
        (["headline"], ["source"], ["url"], ["summary"]),
        K
    )
    return arts, "fetched" if arts else "empty"

def _marketaux(symbol, day_iso, K):
    """Fetch articles from MarketAux for ``symbol`` on ``day_iso``."""

    k = os.environ.get("MARKETAUX_KEY") or os.environ.get("MARKET_AUX_KEY")
    if not k:
        return [], "missing_key"
    url = "https://api.marketaux.com/v1/news/all"
    params = {"symbols": symbol, "published_after": day_iso, "published_before": _day_plus_one(day_iso), "filter_entities": "true", "limit": max(50, K), "api_token": k}
    data, err = _try_request_json(url, params=params)
    if err: return [], err
    arts = _map_articles(
        data.get("data", []),
        (["title"], ["source"], ["url"], ["snippet"]),
        K
    )
    return arts, "fetched" if arts else "empty"

def _fmp(symbol, day_iso, K):
    """Fetch articles from Financial Modeling Prep for ``symbol`` on ``day_iso``."""

    k = os.environ.get("FMP_API_KEY")
    if not k:
        return [], "missing_key"
    # FMP has /stock_news and /press-releases; using /stock_news with tickers
    url = "https://financialmodelingprep.com/api/v3/stock_news"
    params = {"tickers": symbol, "from": day_iso, "to": _day_plus_one(day_iso), "limit": max(50, K), "apikey": k}
    data, err = _try_request_json(url, params=params)
    if err: return [], err
    arts = _map_articles(
        data,
        (["title"], ["site"], ["url"], ["text"]),
        K
    )
    return arts, "fetched" if arts else "empty"

def _bing(symbol, day_iso, K):
    """Fetch news results from Bing for ``symbol`` on ``day_iso``."""

    k = os.environ.get("BING_API_KEY") or os.environ.get("BING_KEY")
    if not k:
        return [], "missing_key"
    url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": k}
    params = {"q": symbol, "freshness": "Day", "count": max(50, K), "mkt": "en-US"}
    data, err = _try_request_json(url, params=params, headers=headers)
    if err: return [], err
    arts = _map_articles(
        data.get("value", []),
        (["name"], ["provider","name"], ["url"], ["description"]),
        K
    )
    return arts, "fetched" if arts else "empty"

_PROVIDERS = {
    "Finnhub": _finnhub,
    "Polygon": _polygon,
    "NewsAPI": _newsapi,
    "MarketAux": _marketaux,
    "FMP": _fmp,
    "Bing": _bing,
}

def fetch_day(symbol: str, day_iso: str, K: int):
    """Fetch up to ``K`` articles for ``symbol`` by iterating the provider chain."""
    chain = get_provider_chain()
    attempts = []
    for prov in chain:
        fn = _PROVIDERS.get(prov)
        if not fn:
            attempts.append(f"{prov}=unsupported")
            continue
        arts, reason = fn(symbol, day_iso, K)
        attempts.append(f"{prov}={reason}")
        if arts:
            _write_cache({"provider": prov, "symbol": symbol, "date": day_iso, "articles": arts, "reason": reason})
            return arts, f"{prov}:{reason}"
        # mild backoff on rate limits
        if "rate" in str(reason).lower():
            time.sleep(1.0)
    trace_str = ";".join(attempts) if attempts else "no_attempts"
    _write_cache({"provider": "none", "symbol": symbol, "date": day_iso, "articles": [], "reason": f"none:{trace_str}"})
    return [], f"none:{trace_str}"


def fetch_news_with_reason(symbol: str, day_iso: str, K: int, base_dir: str = None):
    """Return ``(articles, reason)`` while honoring cached local copies."""
    local_dir = base_dir or os.environ.get("NEWS_LOCAL_DIR")
    try:
        cached_arts, cached_provider = load_local_day(symbol, day_iso, base_dir=local_dir)
    except Exception:
        cached_arts, cached_provider = [], ""

    usable_cached = [a for a in (cached_arts or []) if not _is_synth(a)]
    if len(usable_cached) >= K:
        reason = f"local:{cached_provider or 'cached'}"
        return usable_cached[:K], reason

    arts, reason = fetch_day(symbol, day_iso, K)

    if arts:
        provider = reason.split(":", 1)[0] if reason else "unknown"
        try:
            save_local_day(symbol, day_iso, arts, provider, reason, base_dir=local_dir)
        except Exception:
            pass

    return arts, reason
