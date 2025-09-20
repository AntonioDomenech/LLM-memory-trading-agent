import os
import json
import time
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Callable

DEFAULT_DIR = "data/news_local"


# ---- Paths & IO helpers -----------------------------------------------------

def _dir(base_dir: str = None) -> str:
    return base_dir or os.environ.get("NEWS_LOCAL_DIR") or DEFAULT_DIR


def local_day_path(symbol: str, day_iso: str, base_dir: str = None) -> str:
    d = _dir(base_dir)
    sym = (symbol or "UNKNOWN").upper()
    return os.path.join(d, sym, f"{day_iso}.json")


def ensure_dirs(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_local_day(symbol: str, day_iso: str, articles: List[Dict], provider: str, reason: str,
                   base_dir: str = None) -> str:
    p = local_day_path(symbol, day_iso, base_dir)
    ensure_dirs(p)
    rec = {"symbol": symbol, "date": day_iso, "provider": provider, "reason": reason, "articles": articles}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    return p


def load_local_day(symbol: str, day_iso: str, base_dir: str = None) -> Tuple[List[Dict], str]:
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
    d0 = datetime.fromisoformat(start_iso).date()
    d1 = datetime.fromisoformat(end_iso).date()
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


# ---- Article helpers ---------------------------------------------------------

def _has_content(a: dict) -> bool:
    if not isinstance(a, dict):
        return False
    c = a.get("content") or a.get("text") or a.get("body")
    try:
        return bool(c and isinstance(c, str) and len(c.strip()) > 120)
    except Exception:
        return False


def _key(a: dict) -> str:
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


def _provider_mentions_synth(label: str) -> bool:
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
    out: List[Dict] = []
    seen = set()

    def _add(a):
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


def _enrich_content_only(arts: List[Dict], content_delay: float, stats: Dict) -> None:
    """Fill missing content for given articles in-place using article_scraper, updating stats.
    Does NOT fetch headlines; only enriches body text from each article's own URL.
    """
    try:
        from .article_scraper import fetch_fulltext
    except Exception:
        fetch_fulltext = None
    if not fetch_fulltext or not arts:
        return
    now = time.time
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
                if retry_at > now():
                    continue
            result = fetch_fulltext(u)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                text, meta = result
            else:
                text, meta = result, {}
            if isinstance(text, str) and text.strip():
                it["content"] = text
                it.pop("_content_status", None)
                it.pop("_content_retry_at", None)
                if meta.get("error"):
                    it["_content_error"] = meta.get("error")
                else:
                    it.pop("_content_error", None)
                stats["content_ok"] = stats.get("content_ok", 0) + 1
            else:
                meta = meta or {}
                retryable = bool(meta.get("retryable"))
                it["_content_error"] = meta.get("error") or "unknown"
                if retryable:
                    it["_content_status"] = "retry"
                    delay = _retry_delay_seconds(meta)
                    it["_content_retry_at"] = now() + delay
                else:
                    it["_content_status"] = "failed"
                    it.pop("_content_retry_at", None)
                stats["content_fail"] = stats.get("content_fail", 0) + 1
            if content_delay and content_delay > 0:
                time.sleep(content_delay)
        except Exception:
            stats["content_fail"] = stats.get("content_fail", 0) + 1
            if isinstance(it, dict):
                it["_content_status"] = "retry"
                it["_content_error"] = "exception"
                it["_content_retry_at"] = now() + 300.0
            continue


# ---- Pre-scan for diagnostics ------------------------------------------------

def prescan_days(symbol: str, start_iso: str, end_iso: str, K: int,
                 base_dir: str = None, full_content: bool = False):
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
                   content_delay: float = 0.2, skip_existing: bool = True):
    """Download (and optionally content-enrich) daily news and save locally.
    Resumable and top-up aware.
    """
    stats: Dict = {
        "saved": 0,
        "content_ok": 0,
        "content_fail": 0,
        "days_with_news": 0,
        "skipped_existing": 0,
    }
    for _k in ("saved", "content_ok", "content_fail", "days_with_news", "skipped_existing"):
        if _k not in stats:
            stats[_k] = 0

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

    for day in days_list:
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

            # Decide action with correct counting (all usable articles in the file)
            usable_existing = [a for a in pre_arts if not _is_synth(a)]
            usable_count = len(usable_existing)
            enough_count = usable_count >= K
            need_content = bool(full_content) and any((not _has_content(a)) for a in usable_existing[:K])

            # 1) Full skip
            if pre_arts and enough_count and not need_content:
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
                              "skipped_existing": stats["skipped_existing"]})
                continue

            # 2) Content-only enrichment (no provider API calls)
            if pre_arts and enough_count and need_content:
                work = usable_existing[:K]
                _enrich_content_only(work, content_delay, stats)
                # Write back enriched items
                by_key = {_key(a): a for a in work}
                new_pre = []
                for a in pre_arts:
                    k = _key(a)
                    new_pre.append(by_key.get(k, a))
                pre_arts = new_pre
                path = save_local_day(symbol, day, pre_arts[:K],
                                      pre_provider or "local+content", "content_enriched", base_dir)
                stats["saved"] += 1
                stats["days_with_news"] += 1
                if on_event:
                    on_event({"type": "progress", "date": day,
                              "provider": pre_provider or "local+content",
                              "saved_path": path, "content_ok": stats["content_ok"],
                              "content_fail": stats["content_fail"],
                              "days_with_news": stats["days_with_news"],
                              "skipped_existing": stats["skipped_existing"]})
                continue

            # 3) Fetch headlines (providers)
            arts, reason = fetch_fn(symbol, day, K)
            prov_label = (reason or "").split(":", 1)[0].lower() if reason else ""
            prov_label = _clean_provider_label(prov_label)

            # Filter out placeholder articles from providers
            arts = [a for a in (arts or []) if not _is_synth(a)]

            if not arts:
                # No new data from providers; keep existing file as-is and emit progress
                if pre_exists and (not pre_arts or all(_is_synth(a) for a in pre_arts)):
                    try:
                        os.remove(p)
                        pre_exists = False
                    except FileNotFoundError:
                        pre_exists = False
                    except Exception:
                        pass
                if on_event:
                    on_event({
                        "type": "progress",
                        "date": day,
                        "provider": prov_label or "none",
                        "saved_path": p if pre_exists else "",
                        "content_ok": stats["content_ok"],
                        "content_fail": stats["content_fail"],
                        "days_with_news": stats["days_with_news"],
                        "skipped_existing": stats["skipped_existing"],
                    })
                continue

            # Optional content enrichment
            if full_content:
                _enrich_content_only(arts, content_delay, stats)

            # Remove placeholder articles from cached data before merging
            pre_arts = [a for a in pre_arts if not _is_synth(a)]

            # Merge and save
            final_arts = _merge_articles_for_k(pre_arts, arts, K,
                                               require_content=bool(full_content))
            if not final_arts:
                # Nothing usable to save; remove empty placeholder file if needed
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
                if on_event:
                    on_event({"type": "progress", "date": day, "provider": prov_label or "none",
                              "saved_path": "", "content_ok": stats["content_ok"],
                              "content_fail": stats["content_fail"],
                              "days_with_news": stats["days_with_news"],
                              "skipped_existing": stats["skipped_existing"]})
                continue

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
                          "skipped_existing": stats["skipped_existing"]})

        except Exception as e:
            # Emit error and continue
            if on_event:
                try:
                    on_event({"type": "error", "date": day, "error": f"{type(e).__name__}: {e}"})
                except Exception:
                    pass
            continue

    return stats
