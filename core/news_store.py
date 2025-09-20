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
    """Detect synthetic placeholder articles robustly, handling dict fields and non-strings."""
    if not isinstance(a, dict):
        return False
    src_val = _as_str(a.get("source")).strip().lower()
    url_val = _as_str(a.get("url")).strip().lower()
    title_val = _as_str(a.get("title")).strip().lower()
    return (src_val == "synthetic") or (not url_val and title_val.startswith("no reliable articles found"))


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
    for it in arts:
        try:
            if isinstance(it, dict) and isinstance(it.get("content"), str) and it.get("content").strip():
                continue
            u = it.get("url", "")
            if not isinstance(u, str) or not u.strip():
                continue
            text = ""
            try:
                text = fetch_fulltext(u)
            except Exception:
                text = ""
            if isinstance(text, str) and text.strip():
                it["content"] = text
                stats["content_ok"] = stats.get("content_ok", 0) + 1
            else:
                stats["content_fail"] = stats.get("content_fail", 0) + 1
            if content_delay and content_delay > 0:
                time.sleep(content_delay)
        except Exception:
            stats["content_fail"] = stats.get("content_fail", 0) + 1
            continue


# ---- Pre-scan for diagnostics ------------------------------------------------

def prescan_days(symbol: str, start_iso: str, end_iso: str, K: int,
                 base_dir: str = None, full_content: bool = False):
    """Return a list of {date, path, exists, count, non_synth, provider, decision} for diagnostics."""
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
        non_synth_list = [a for a in pre_arts if not _is_synth(a)]
        non_synth = len(non_synth_list)
        enough = non_synth >= K
        need_cont = bool(full_content) and any((not _has_content(a)) for a in non_synth_list[:K])
        if exists and enough and not need_cont:
            decision = "skip_existing"
        elif exists and (not enough or need_cont):
            decision = "top_up"
        else:
            decision = "fetch_new"
        plan.append({
            "date": day, "path": p, "exists": bool(exists),
            "count": len(pre_arts), "non_synth": int(non_synth),
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
        "synthetic_days": 0,
        "real_days": 0,
        "skipped_existing": 0,
    }
    for _k in ("saved", "content_ok", "content_fail", "synthetic_days", "real_days", "skipped_existing"):
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

            # Load pre-existing
            pre_arts: List[Dict] = []
            pre_provider = ""
            if skip_existing and os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    pre_arts = d.get("articles", []) or []
                    pre_provider = d.get("provider") or d.get("reason") or "local"
                except Exception:
                    pre_arts, pre_provider = [], ""

            # Decide action with correct counting (ALL non-synthetic in the file)
            non_synth_list = [a for a in pre_arts if not _is_synth(a)]
            non_synth = len(non_synth_list)
            enough_count = non_synth >= K
            need_content = bool(full_content) and any((not _has_content(a)) for a in non_synth_list[:K])

            # 1) Full skip
            if pre_arts and enough_count and not need_content:
                stats["skipped_existing"] += 1
                if on_event:
                    on_event({"type": "progress", "date": day, "provider": "local",
                              "saved_path": p, "content_ok": stats["content_ok"],
                              "content_fail": stats["content_fail"],
                              "synthetic_days": stats["synthetic_days"],
                              "real_days": stats["real_days"],
                              "skipped_existing": stats["skipped_existing"]})
                continue

            # 2) Content-only enrichment (no provider API calls)
            if pre_arts and enough_count and need_content:
                work = non_synth_list[:K]
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
                if on_event:
                    on_event({"type": "progress", "date": day,
                              "provider": pre_provider or "local+content",
                              "saved_path": path, "content_ok": stats["content_ok"],
                              "content_fail": stats["content_fail"],
                              "synthetic_days": stats["synthetic_days"],
                              "real_days": stats["real_days"],
                              "skipped_existing": stats["skipped_existing"]})
                continue

            # 3) Fetch headlines (providers)
            arts, reason = fetch_fn(symbol, day, K)
            prov_label = (reason or "").split(":", 1)[0].lower() if reason else ""

            # Optional content enrichment
            if full_content:
                try:
                    from .article_scraper import fetch_fulltext
                except Exception:
                    fetch_fulltext = None
                if fetch_fulltext:
                    for it in arts:
                        try:
                            if isinstance(it.get("content"), str) and it.get("content").strip():
                                continue
                            u = it.get("url", "")
                            if not isinstance(u, str) or not u:
                                continue
                            text = ""
                            try:
                                text = fetch_fulltext(u)
                            except Exception:
                                text = ""
                            if isinstance(text, str) and text.strip():
                                it["content"] = text
                                stats["content_ok"] += 1
                            else:
                                stats["content_fail"] += 1
                            if content_delay and content_delay > 0:
                                time.sleep(content_delay)
                        except Exception:
                            stats["content_fail"] += 1

            # Remove synthetics if providers gave real data
            if prov_label != "synthetic":
                pre_arts = [a for a in pre_arts if not _is_synth(a)]

            # Merge and save
            final_arts = _merge_articles_for_k(pre_arts, arts, K,
                                               require_content=bool(full_content))
            label_to_save = prov_label if not pre_arts or len(final_arts) == len(arts) else f"{prov_label}+local"
            path = save_local_day(symbol, day, final_arts,
                                  label_to_save or "unknown", str(reason or ""), base_dir)
            stats["saved"] += 1
            if (label_to_save or "").lower().startswith("synthetic"):
                stats["synthetic_days"] += 1
            else:
                stats["real_days"] += 1

            if on_event:
                on_event({"type": "progress", "date": day, "provider": label_to_save,
                          "saved_path": path, "content_ok": stats["content_ok"],
                          "content_fail": stats["content_fail"],
                          "synthetic_days": stats["synthetic_days"],
                          "real_days": stats["real_days"],
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
