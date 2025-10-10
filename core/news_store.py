import os
import os
import json
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Callable, Any

DEFAULT_DIR = "data/news_local"


# ---- Paths & IO helpers -----------------------------------------------------

def _dir(base_dir: str = None) -> str:
    """Return the effective local news directory."""

    return base_dir or os.environ.get("NEWS_LOCAL_DIR") or DEFAULT_DIR


def local_day_path(symbol: str, day_iso: str, base_dir: str = None) -> str:
    """Compute the file path storing news for ``symbol`` on ``day_iso``."""

    d = _dir(base_dir)
    sym = (symbol or "UNKNOWN").upper()
    return os.path.join(d, sym, f"{day_iso}.json")


def ensure_dirs(path: str) -> None:
    """Ensure the parent directory for ``path`` exists."""

    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_local_day(symbol: str, day_iso: str, articles: List[Dict], provider: str, reason: str,
                   base_dir: str = None) -> str:
    """Persist downloaded news articles and return the saved path."""

    p = local_day_path(symbol, day_iso, base_dir)
    ensure_dirs(p)
    rec = {"symbol": symbol, "date": day_iso, "provider": provider, "reason": reason, "articles": articles}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    return p


def load_local_day(symbol: str, day_iso: str, base_dir: str = None) -> Tuple[List[Dict], str]:
    """Load cached news articles if available, returning ``(articles, provider)``."""

    p = local_day_path(symbol, day_iso, base_dir)
    if not os.path.exists(p):
        return [], ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d.get("articles", []), d.get("provider") or d.get("reason") or "local"
    except Exception as e:
        return [], f"local:error:{e}"


def daterange(start_iso: str, end_iso: str):
    """Yield ISO date strings from ``start_iso`` to ``end_iso`` inclusive."""

    d0 = datetime.fromisoformat(start_iso).date()
    d1 = datetime.fromisoformat(end_iso).date()
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


# ---- Article helpers ---------------------------------------------------------

def _has_content(a: dict) -> bool:
    """Return ``True`` when an article dictionary contains substantive text."""

    if not isinstance(a, dict):
        return False
    c = a.get("content") or a.get("text") or a.get("body")
    try:
        return bool(c and isinstance(c, str) and len(c.strip()) > 120)
    except Exception:
        return False


def _key(a: dict) -> str:
    """Return a stable deduplication key for an article."""

    if not isinstance(a, dict):
        return ""
    u = a.get("url")
    if isinstance(u, str):
        u = u.strip().lower()
    else:
        u = ""
    if u:
        return u
    t = a.get("title")
    if isinstance(t, str):
        t = t.strip().lower()
    else:
        t = ""
    return f"title:{t}" if t else ""


def _as_str(x):
    """Convert nested structures to a representative string."""

    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # common nested shapes like {"name": "..."} etc.
        for k in ("name", "title", "source", "publisher"):
            v = x.get(k)
            if isinstance(v, str):
                return v
    try:
        return str(x) if x is not None else ""
    except Exception:
        return ""


def _is_synth(a: dict) -> bool:
    """Detect placeholder articles that should not be counted as usable news."""
    if not isinstance(a, dict):
        return False
    src_val = _as_str(a.get("source")).strip().lower()
    url_val = _as_str(a.get("url")).strip().lower()
    title_val = _as_str(a.get("title")).strip().lower()
    return (src_val == "synthetic") or (not url_val and title_val.startswith("no reliable articles found"))


def _content_failed(a: dict) -> bool:
    """Return True when an article is explicitly marked as failed content extraction."""

    if not isinstance(a, dict):
        return False
    status = a.get("_content_status")
    if isinstance(status, str):
        return status.strip().lower() == "failed"
    return False


def _is_retryable_reason(reason: str) -> bool:
    """Heuristically decide whether a provider response is worth retrying later."""

    if not reason:
        return True
    lowered = str(reason).lower()
    non_retry_tokens = (
        "missing_key",
        "invalid",
        "unauthorized",
        "forbidden",
        "permission",
        "unsupported",
    )
    return not any(token in lowered for token in non_retry_tokens)


def _retry_delay_for_reason(reason: str, attempt: int) -> float:
    """Return a delay (in seconds) before retrying provider downloads."""

    attempt = max(1, int(attempt or 1))
    lowered = str(reason or "").lower()

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return default

    base_default = _env_float("NEWS_RETRY_BASE_SECONDS", 5.0)
    cap_default = _env_float("NEWS_RETRY_MAX_SECONDS", 300.0)

    if any(token in lowered for token in ("429", "rate", "too many", "quota", "limit")):
        base = base_default
    elif any(token in lowered for token in ("http_5", "timeout", "temporar", "gateway", "unavailable")):
        base = max(2.0, base_default / 2.0)
    else:
        base = max(1.0, base_default / 5.0)

    delay = base * (2 ** (attempt - 1))
    return max(1.0, min(delay, cap_default))



def _provider_mentions_synth(label: str) -> bool:
    """Check whether a provider label references synthetic content."""

    if not label:
        return False
    try:
        parts = str(label).split("+")
    except Exception:
        parts = [str(label)]
    for part in parts:
        if part and part.strip().lower().startswith("synthetic"):
            return True
    return False


def _clean_provider_label(label: str) -> str:
    """Remove synthetic markers and duplicates from a provider label."""

    if not label:
        return "local"
    try:
        parts = [p.strip() for p in str(label).split("+")]
    except Exception:
        parts = [str(label).strip()]
    clean_parts = []
    for part in parts:
        if not part or part.lower().startswith("synthetic"):
            continue
        if part not in clean_parts:
            clean_parts.append(part)
    return "+".join(clean_parts) if clean_parts else "local"


def _merge_articles_for_k(existing: List[Dict], new: List[Dict], K: int,
                          require_content: bool) -> List[Dict]:
    """Merge existing and new articles keeping at most ``K`` unique entries."""

    out: List[Dict] = []
    seen = set()

    def _add(a):
        """Add article ``a`` to ``out`` if it has not been seen."""

        if require_content and not _has_content(a):
            return False
        k = _key(a)
        if not k or k in seen:
            return False
        seen.add(k)
        out.append(a)
        return True

    for a in (existing or []):
        _add(a)
        if len(out) >= K:
            return out[:K]

    for a in (new or []):
        _add(a)
        if len(out) >= K:
            break

    return out[:K]


def _retry_delay_seconds(meta: Dict) -> float:
    """Determine how long to wait before retrying content extraction."""

    if not isinstance(meta, dict):
        return 300.0
    retry_after = meta.get("retry_after")
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        return float(retry_after)
    attempts = meta.get("attempts")
    if isinstance(attempts, int) and attempts > 0:
        # Exponential backoff capped at 15 minutes
        delay = 2 ** attempts
        return float(max(60.0, min(delay * 60.0, 900.0)))
    return 300.0


def _content_diag_string(diag: Any) -> str:
    """Serialise diagnostic objects for logging or counting."""

    if not diag:
        return ""
    if isinstance(diag, str):
        return diag
    try:
        return json.dumps(diag, ensure_ascii=False, sort_keys=True)
    except Exception:
        try:
            return str(diag)
        except Exception:
            return "unserializable_diagnostic"


def _register_content_error(stats: Dict, diag: Any) -> str:
    """Track a content extraction error and return its bucket label."""

    diag_str = _content_diag_string(diag) or "unknown"
    bucket = stats.setdefault("content_error_types", {})
    bucket[diag_str] = bucket.get(diag_str, 0) + 1
    return diag_str


def _apply_content_result(article: Dict, text: str, diag: Any, stats: Dict) -> None:
    """Update ``article`` in-place using the scraped text/diagnostics."""

    diag_dict = diag if isinstance(diag, dict) else ({"detail": diag} if diag else {})
    text_val = text.strip() if isinstance(text, str) else ""
    if text_val:
        if isinstance(article, dict):
            article["content"] = text_val
            article.pop("_content_status", None)
            article.pop("_content_retry_at", None)
            if isinstance(diag_dict, dict) and diag_dict.get("error"):
                article["_content_error"] = diag_dict.get("error")
            else:
                article.pop("_content_error", None)
        stats["content_ok"] = stats.get("content_ok", 0) + 1
        return

    stats["content_fail"] = stats.get("content_fail", 0) + 1
    diag_str = _register_content_error(stats, diag_dict)
    if isinstance(article, dict):
        err_label = diag_str or (diag_dict.get("error") if isinstance(diag_dict, dict) else "unknown")
        article["_content_error"] = err_label or "unknown"
        retryable = bool(diag_dict.get("retryable")) if isinstance(diag_dict, dict) else False
        if retryable:
            article["_content_status"] = "retry"
            delay = _retry_delay_seconds(diag_dict)
            article["_content_retry_at"] = time.time() + delay
        else:
            article["_content_status"] = "failed"
            article.pop("_content_retry_at", None)


def _enrich_content_only(arts: List[Dict], content_delay: float, stats: Dict) -> None:
    """Fetch missing article body text without contacting headline providers."""

    """Fill missing content for given articles in-place using article_scraper, updating stats.
    Does NOT fetch headlines; only enriches body text from each article's own URL.
    """
    try:
        from .article_scraper import fetch_fulltext
    except Exception:
        fetch_fulltext = None
    if not fetch_fulltext or not arts:
        return
    now_fn = time.time
    for it in arts:
        try:
            if isinstance(it, dict) and isinstance(it.get("content"), str) and it.get("content").strip():
                continue
            u = it.get("url", "")
            if not isinstance(u, str) or not u.strip():
                continue
            status_flag = it.get("_content_status")
            if status_flag == "failed":
                continue
            retry_at = it.get("_content_retry_at")
            if status_flag == "retry" and isinstance(retry_at, (int, float)):
                if retry_at > now_fn():
                    continue
            try:
                result = fetch_fulltext(u, return_meta=True)
            except TypeError:
                result = fetch_fulltext(u)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                text, meta = result
            else:
                text, meta = result, {}
            _apply_content_result(it, text, meta, stats)
            if content_delay and content_delay > 0:
                time.sleep(content_delay)
        except Exception as e:
            diag = {
                "phase": "unexpected",
                "error": type(e).__name__,
                "message": str(e),
                "retryable": True,
                "retry_after": 300.0,
            }
            _apply_content_result(it, "", diag, stats)
            continue


# ---- Pre-scan for diagnostics ------------------------------------------------

def prescan_days(symbol: str, start_iso: str, end_iso: str, K: int,
                 base_dir: str = None, full_content: bool = False):
    """Inspect cached files to plan which days need fetching or enrichment."""

    """Return a list of {date, path, exists, count, available, provider, decision} for diagnostics."""
    plan = []
    for day in daterange(start_iso, end_iso):
        p = local_day_path(symbol, day, base_dir)
        exists = os.path.exists(p)
        pre_arts, provider = ([], "")
        if exists:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                pre_arts = d.get("articles", []) or []
                provider = d.get("provider") or d.get("reason") or "local"
            except Exception as e:
                pre_arts, provider = [], f"local:error:{e}"
        usable_list = [a for a in pre_arts if not _is_synth(a)]
        usable_count = len(usable_list)
        enough = usable_count >= K
        need_cont = bool(full_content) and any((not _has_content(a)) for a in usable_list[:K])
        if exists and enough and not need_cont:
            decision = "skip_existing"
        elif exists and (not enough or need_cont):
            decision = "top_up"
        else:
            decision = "fetch_new"
        plan.append({
            "date": day, "path": p, "exists": bool(exists),
            "count": len(pre_arts), "available": int(usable_count),
            "provider": provider, "decision": decision
        })
    return plan


# ---- Main API ----------------------------------------------------------------

def download_range(symbol: str, start_iso: str, end_iso: str, K: int = 5, base_dir: str = None,
                   fetch_fn: Callable = None, on_event: Callable = None, full_content: bool = False,
                   content_delay: float = 0.2, skip_existing: bool = True,
                   should_stop: Callable[[], bool] = None):
    """Download and optionally enrich local news files across a date range."""

    """Download (and optionally content-enrich) daily news and save locally.
    Resumable and top-up aware.
    """
    stats: Dict = {
        "saved": 0,
        "content_ok": 0,
        "content_fail": 0,
        "days_with_news": 0,
        "skipped_existing": 0,
        "content_error_types": {},
        "retry_queue": [],
        "paused": False,
    }
    for _k in ("saved", "content_ok", "content_fail", "days_with_news", "skipped_existing"):
        if _k not in stats:
            stats[_k] = 0
    if "content_error_types" not in stats:
        stats["content_error_types"] = {}

    if fetch_fn is None:
        from .news_fetcher import fetch_day as fetch_fn

    days_list = list(daterange(start_iso, end_iso))
    total_days = len(days_list)

    if on_event:
        try:
            on_event({"type": "plan",
                      "plan": prescan_days(symbol, start_iso, end_iso, K, base_dir=base_dir,
                                           full_content=bool(full_content))})
        except Exception:
            pass

    day_queue = deque((day, 0.0) for day in days_list)
    retry_attempts: Dict[str, int] = {}
    retry_meta: Dict[str, Dict[str, Any]] = {}
    max_retries = 1

    while day_queue:
        day, not_before = day_queue.popleft()
        if should_stop and should_stop():
            stats["paused"] = True
            day_queue.appendleft((day, not_before))
            break
        now = time.time()
        if not_before and now < not_before:
            wait = not_before - now
            if wait > 0:
                time.sleep(wait)
            now = time.time()
        attempt = retry_attempts.get(day, 0)
        try:
            stats["last_date"] = day
            p = local_day_path(symbol, day, base_dir)

            # Load pre-existing (always inspect so we can avoid redundant fetches)
            pre_exists = os.path.exists(p)
            pre_arts: List[Dict] = []
            pre_provider = ""
            if pre_exists:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    pre_arts = d.get("articles", []) or []
                    pre_provider = d.get("provider") or d.get("reason") or "local"
                except Exception:
                    pre_arts, pre_provider = [], ""

            fails_removed = False
            if full_content and pre_arts:
                if any(_content_failed(a) for a in pre_arts):
                    fails_removed = True
                    pre_arts = [a for a in pre_arts if not _content_failed(a)]
                    if pre_exists and not pre_arts:
                        try:
                            os.remove(p)
                            pre_exists = False
                        except FileNotFoundError:
                            pre_exists = False
                        except Exception:
                            pass

            non_synth = [a for a in pre_arts if not _is_synth(a)]
            if full_content:
                missing_content = [a for a in non_synth if not _has_content(a)]
                usable_existing = [a for a in non_synth if _has_content(a)]
            else:
                missing_content = []
                usable_existing = list(non_synth)
            have_headlines = len(non_synth) >= K
            enough_count = len(usable_existing) >= K
            need_content = bool(full_content and missing_content)
            needs_replacement = bool(full_content and fails_removed)

            # 1) Full skip (optional via skip_existing flag)
            if skip_existing and pre_arts and enough_count and not need_content and not needs_replacement:
                retry_meta.pop(day, None)
                retry_attempts.pop(day, None)
                stats["skipped_existing"] += 1
                cleaned = usable_existing
                provider_label = pre_provider or "local"
                path = p
                if cleaned != pre_arts or _provider_mentions_synth(pre_provider):
                    provider_label = _clean_provider_label(pre_provider)
                    path = save_local_day(symbol, day, cleaned,
                                          provider_label, "local:normalized_cached_metadata", base_dir)
                    stats["saved"] += 1
                stats["days_with_news"] += 1
                if on_event:
                    on_event({"type": "progress", "date": day, "provider": provider_label,
                              "saved_path": path, "content_ok": stats["content_ok"],
                              "content_fail": stats["content_fail"],
                              "days_with_news": stats["days_with_news"],
                              "skipped_existing": stats["skipped_existing"],
                              "content_error_types": dict(stats.get("content_error_types", {}))})
                continue

            # 2) Content-only enrichment (no provider API calls)
            if pre_arts and have_headlines and need_content and not needs_replacement:
                work = missing_content[:K]
                if work:
                    _enrich_content_only(work, content_delay, stats)
                non_synth = [a for a in pre_arts if not _is_synth(a)]
                if full_content:
                    missing_content = [a for a in non_synth if not _has_content(a)]
                    usable_existing = [a for a in non_synth if _has_content(a)]
                else:
                    missing_content = []
                    usable_existing = list(non_synth)
                have_headlines = len(non_synth) >= K
                enough_count = len(usable_existing) >= K
                need_content = bool(full_content and missing_content)
                if have_headlines and enough_count and not need_content:
                    retry_meta.pop(day, None)
                    provider_to_save = _clean_provider_label(pre_provider or "local+content")
                    final_articles = usable_existing[:K]
                    path = save_local_day(symbol, day, final_articles,
                                          provider_to_save or "local+content",
                                          "content_enriched", base_dir)
                    stats["saved"] += 1
                    stats["days_with_news"] += 1
                    if on_event:
                        on_event({"type": "progress", "date": day,
                                  "provider": provider_to_save,
                                  "saved_path": path, "content_ok": stats["content_ok"],
                                  "content_fail": stats["content_fail"],
                                  "days_with_news": stats["days_with_news"],
                                  "skipped_existing": stats["skipped_existing"],
                                  "content_error_types": dict(stats.get("content_error_types", {}))})
                    continue
                # fallthrough to provider fetch to top-up with new headlines

            # 3) Fetch headlines (providers)
            arts, reason = fetch_fn(symbol, day, K)
            prov_label_raw = (reason or "").split(":", 1)[0].lower() if reason else ""
            prov_label = _clean_provider_label(prov_label_raw)

            # Filter out placeholder articles from providers
            arts = [a for a in (arts or []) if not _is_synth(a)]

            if not arts:
                retryable = _is_retryable_reason(reason)
                will_retry = retryable and attempt < max_retries
                delay = _retry_delay_for_reason(reason, attempt + 1) if will_retry else 0.0
                retry_meta[day] = {
                    "date": day,
                    "reason": str(reason or ""),
                    "attempt": attempt + 1,
                    "retryable": bool(retryable),
                    "scheduled": will_retry,
                    "retry_after": float(delay),
                }
                if will_retry:
                    retry_attempts[day] = attempt + 1
                    day_queue.append((day, time.time() + delay))
                else:
                    retry_attempts.pop(day, None)
                if pre_exists and (not pre_arts or all(_is_synth(a) for a in pre_arts)):
                    try:
                        os.remove(p)
                        pre_exists = False
                    except FileNotFoundError:
                        pre_exists = False
                    except Exception:
                        pass
                provider_display = prov_label or "none"
                if will_retry:
                    provider_display = f"{provider_display or 'none'}+retry"
                else:
                    provider_display = f"{provider_display or 'none'}+blocked"
                if on_event:
                    on_event({
                        "type": "progress",
                        "date": day,
                        "provider": provider_display,
                        "saved_path": p if pre_exists else "",
                        "content_ok": stats["content_ok"],
                        "content_fail": stats["content_fail"],
                        "days_with_news": stats["days_with_news"],
                        "skipped_existing": stats["skipped_existing"],
                        "content_error_types": dict(stats.get("content_error_types", {})),
                        "status_note": "queued_retry" if will_retry else "no_data",
                        "retry_reason": str(reason or ""),
                        "retry_attempt": attempt + 1,
                        "retry_after": float(delay),
                    })
                continue

            # Optional content enrichment on freshly fetched headlines
            if full_content:
                _enrich_content_only(arts, content_delay, stats)

            # Remove placeholder articles from cached data before merging
            pre_arts = [a for a in pre_arts if not _is_synth(a)]

            # Merge and save
            final_arts = _merge_articles_for_k(pre_arts, arts, K,
                                               require_content=bool(full_content))
            if not final_arts:
                retryable = _is_retryable_reason(reason)
                will_retry = retryable and attempt < max_retries
                delay = _retry_delay_for_reason(reason, attempt + 1) if will_retry else 0.0
                retry_meta[day] = {
                    "date": day,
                    "reason": str(reason or ""),
                    "attempt": attempt + 1,
                    "retryable": bool(retryable),
                    "scheduled": will_retry,
                    "retry_after": float(delay),
                }
                if will_retry:
                    retry_attempts[day] = attempt + 1
                    day_queue.append((day, time.time() + delay))
                else:
                    retry_attempts.pop(day, None)
                try:
                    if os.path.exists(p):
                        os.remove(p)
                        pre_exists = False
                except Exception:
                    pass
                provider_display = prov_label or "none"
                if will_retry:
                    provider_display = f"{provider_display or 'none'}+retry"
                else:
                    provider_display = f"{provider_display or 'none'}+blocked"
                if on_event:
                    on_event({
                        "type": "progress",
                        "date": day,
                        "provider": provider_display,
                        "saved_path": "",
                        "content_ok": stats["content_ok"],
                        "content_fail": stats["content_fail"],
                        "days_with_news": stats["days_with_news"],
                        "skipped_existing": stats["skipped_existing"],
                        "content_error_types": dict(stats.get("content_error_types", {})),
                        "status_note": "queued_retry" if will_retry else "no_data",
                        "retry_reason": str(reason or ""),
                        "retry_attempt": attempt + 1,
                        "retry_after": float(delay),
                    })
                continue

            retry_meta.pop(day, None)
            retry_attempts.pop(day, None)
            label_to_save = prov_label if not pre_arts or len(final_arts) == len(arts) else f"{prov_label}+local"
            label_to_save = _clean_provider_label(label_to_save)
            path = save_local_day(symbol, day, final_arts,
                                  label_to_save or "unknown", str(reason or ""), base_dir)
            stats["saved"] += 1
            stats["days_with_news"] += 1

            if on_event:
                on_event({"type": "progress", "date": day, "provider": label_to_save,
                          "saved_path": path, "content_ok": stats["content_ok"],
                          "content_fail": stats["content_fail"],
                          "days_with_news": stats["days_with_news"],
                          "skipped_existing": stats["skipped_existing"],
                          "content_error_types": dict(stats.get("content_error_types", {}))})

        except Exception as e:
            retry_meta[day] = {
                "date": day,
                "reason": f"exception:{type(e).__name__}",
                "attempt": attempt + 1,
                "retryable": False,
                "scheduled": False,
                "retry_after": 0.0,
            }
            retry_attempts.pop(day, None)
            if on_event:
                try:
                    on_event({"type": "error", "date": day, "error": f"{type(e).__name__}: {e}"})
                except Exception:
                    pass
            continue

    stats["retry_queue"] = list(retry_meta.values())
    stats["remaining_days"] = [day for day, _ in day_queue]
    return stats
