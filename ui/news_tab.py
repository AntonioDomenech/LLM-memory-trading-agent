import os
import time
import threading
import queue
from datetime import date, datetime
from typing import Dict, Any

import pandas as pd
import streamlit as st

from core.news_store import download_range, load_local_day, prescan_days


STATE_KEY = "news_tab_state"
PROGRESS_LIMIT = 400


def _initial_state() -> Dict[str, Any]:
    return {
        "base_dir": os.environ.get("NEWS_LOCAL_DIR", "data/news_local"),
        "thread": None,
        "event_queue": queue.Queue(),
        "pause_event": threading.Event(),
        "is_running": False,
        "is_paused": False,
        "progress": [],
        "plan": [],
        "history": [],
        "last_stats": None,
        "error_messages": [],
        "params": None,
        "last_update": None,
        "coverage_summary": None,
    }


def _get_state() -> Dict[str, Any]:
    state = st.session_state.get(STATE_KEY)
    if state is None:
        state = _initial_state()
        st.session_state[STATE_KEY] = state
    if state.get("event_queue") is None:
        state["event_queue"] = queue.Queue()
    if state.get("pause_event") is None:
        state["pause_event"] = threading.Event()
    return state


def _compute_total_days(start_iso: str, end_iso: str) -> int:
    try:
        d0 = datetime.fromisoformat(start_iso)
        d1 = datetime.fromisoformat(end_iso)
        return max(1, (d1 - d0).days + 1)
    except Exception:
        return 1


def _drain_events(state: Dict[str, Any]) -> None:
    q = state.get("event_queue")
    if not q:
        return
    while True:
        try:
            kind, payload = q.get_nowait()
        except queue.Empty:
            break

        if kind == "event":
            evt = payload or {}
            evt_type = evt.get("type")
            if evt_type == "plan":
                state["plan"] = evt.get("plan", [])
            elif evt_type == "progress":
                state["progress"].append(evt)
                if len(state["progress"]) > PROGRESS_LIMIT:
                    state["progress"] = state["progress"][-PROGRESS_LIMIT:]
                state["last_update"] = time.time()
            elif evt_type == "error":
                msg = evt.get("error") or str(evt)
                state["error_messages"].append(msg)
        elif kind == "done":
            stats = payload or {}
            state["last_stats"] = stats
            params = state.get("params") or {}
            state["history"].append({
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "symbol": params.get("symbol"),
                "start": params.get("start_iso"),
                "end": params.get("end_iso"),
                "saved": stats.get("saved", 0),
                "days_with_news": stats.get("days_with_news", 0),
                "content_ok": stats.get("content_ok", 0),
                "paused": bool(stats.get("paused")),
            })
            state["thread"] = None
            state["is_running"] = False
            state["is_paused"] = bool(stats.get("paused"))
            state["pause_event"] = threading.Event()
        elif kind == "fatal":
            state["error_messages"].append(str(payload))
            state["thread"] = None
            state["is_running"] = False
            state["is_paused"] = False
            state["pause_event"] = threading.Event()

    thread = state.get("thread")
    if thread and not thread.is_alive() and not state.get("is_running"):
        state["thread"] = None


def _start_download(state: Dict[str, Any], *, symbol: str, start_iso: str, end_iso: str,
                    K: int, base_dir: str, full_content: bool, skip_existing: bool) -> None:
    if state.get("is_running"):
        return

    event_queue = queue.Queue()
    pause_event = threading.Event()

    state["event_queue"] = event_queue
    state["pause_event"] = pause_event
    state["progress"] = []
    state["plan"] = []
    state["error_messages"] = []
    state["last_stats"] = None
    total_days = _compute_total_days(start_iso, end_iso)
    state["params"] = {
        "symbol": symbol,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "K": K,
        "base_dir": base_dir,
        "full_content": full_content,
        "skip_existing": skip_existing,
        "total_days": total_days,
    }
    state["is_running"] = True
    state["is_paused"] = False

    def worker():
        def on_evt(evt: Dict[str, Any]):
            event_queue.put(("event", evt))

        try:
            stats = download_range(
                symbol,
                start_iso,
                end_iso,
                K=K,
                base_dir=base_dir,
                on_event=on_evt,
                full_content=full_content,
                skip_existing=skip_existing,
                should_stop=lambda: pause_event.is_set(),
            )
            event_queue.put(("done", stats))
        except Exception as exc:  # pragma: no cover - defensive
            event_queue.put(("fatal", f"{type(exc).__name__}: {exc}"))

    thread = threading.Thread(target=worker, daemon=True)
    state["thread"] = thread
    thread.start()


def _render_config_summary(symbol: str, start_iso: str, end_iso: str, K: int,
                           full: bool, skip_existing: bool, base_dir: str) -> None:
    st.markdown("**Current selection**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Symbol", symbol or "—")
    c2.metric("Date range", f"{start_iso} → {end_iso}")
    c3.metric("Daily headlines", K)
    c4, c5, c6 = st.columns(3)
    c4.metric("Full content", "Yes" if full else "No")
    c5.metric("Skip existing", "Yes" if skip_existing else "No")
    c6.metric("Local folder", base_dir or "—")


def _render_progress(state: Dict[str, Any]) -> None:
    params = state.get("params") or {}
    total_days = params.get("total_days") or 1
    unique_days = {evt.get("date") for evt in state.get("progress", []) if evt.get("type") == "progress"}
    processed_days = len([day for day in unique_days if day])
    progress_ratio = min(1.0, max(0.0, processed_days / total_days)) if total_days else 0.0

    st.progress(progress_ratio, text=f"Processed {processed_days}/{total_days} days")

    if state.get("progress"):
        latest = state["progress"][-1]
        st.markdown(
            f"- **Last day:** `{latest.get('date','?')}` • "
            f"Provider: `{latest.get('provider','?')}` • "
            f"Saved path: `{latest.get('saved_path','') or '—'}` • "
            f"Content ok: {latest.get('content_ok',0)} / Fail: {latest.get('content_fail',0)} • "
            f"Days with news: {latest.get('days_with_news',0)}"
        )

        pending = [
            {
                "date": evt.get("date"),
                "provider": evt.get("provider"),
                "reason": evt.get("retry_reason"),
                "next_retry_s": evt.get("retry_after"),
                "attempt": evt.get("retry_attempt"),
            }
            for evt in state["progress"]
            if evt.get("status_note") == "queued_retry"
        ]
        if pending:
            df_pending = pd.DataFrame(pending).drop_duplicates(subset=["date", "provider"], keep="last")
            st.info("Days queued for retry")
            st.dataframe(df_pending, width="stretch")

    if state.get("plan"):
        with st.expander("Download plan (pre-scan)", expanded=False):
            df_plan = pd.DataFrame(state["plan"])
            if "available" in df_plan.columns and params.get("K"):
                df_plan["status"] = df_plan["available"].apply(
                    lambda val: "✅ complete" if int(val or 0) >= int(params["K"]) else "⚠️ missing"
                )
            st.dataframe(df_plan, width="stretch")

    if state.get("progress"):
        with st.expander("Recent progress events", expanded=False):
            cols = ["date", "provider", "saved_path", "status_note", "retry_reason", "retry_after"]
            df_events = pd.DataFrame(state["progress"][-100:])  # last 100 events
            st.dataframe(df_events[[c for c in cols if c in df_events.columns]], width="stretch")


def _render_errors(state: Dict[str, Any]) -> None:
    if not state.get("error_messages"):
        return
    for msg in state["error_messages"][-5:]:
        st.error(msg)


def _render_history(state: Dict[str, Any]) -> None:
    if not state.get("history"):
        return
    with st.expander("Download history", expanded=False):
        df = pd.DataFrame(state["history"])
        st.dataframe(df, width="stretch")


def render_news_tab():
    """Render the Streamlit interface for managing the news cache."""

    state = _get_state()
    _drain_events(state)

    st.subheader("📥 News cache manager")
    st.caption(
        "Download and inspect cached news articles with retry awareness, progress tracking, and pause/resume controls."
    )

    with st.expander("Settings", expanded=True):
        base_dir = st.text_input("Local folder", value=state.get("base_dir", "data/news_local"))
        if base_dir != state.get("base_dir"):
            state["base_dir"] = base_dir
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol", value=(state.get("params") or {}).get("symbol", "AAPL")).upper().strip()
        with col2:
            start_default = (
                datetime.fromisoformat((state.get("params") or {}).get("start_iso")).date()
                if state.get("params") and (state.get("params") or {}).get("start_iso")
                else date(2022, 1, 1)
            )
            start_d = st.date_input("Start date", value=start_default)
        with col3:
            end_default = (
                datetime.fromisoformat((state.get("params") or {}).get("end_iso")).date()
                if state.get("params") and (state.get("params") or {}).get("end_iso")
                else date(2022, 3, 31)
            )
            end_d = st.date_input("End date", value=end_default)

        col_k, col_full, col_skip = st.columns([1, 1, 1])
        with col_k:
            K = int(
                st.number_input(
                    "Headlines per day (K)",
                    min_value=1,
                    max_value=50,
                    value=int((state.get("params") or {}).get("K", 10)),
                    step=1,
                )
            )
        with col_full:
            full = st.checkbox(
                "Download full article bodies",
                value=bool((state.get("params") or {}).get("full_content", True)),
                help="Fetch and store readable text for each news URL when available.",
            )
        with col_skip:
            skip_existing = st.checkbox(
                "Skip already cached days",
                value=bool((state.get("params") or {}).get("skip_existing", True)),
                help="Avoid re-fetching days that already have enough local articles.",
            )

        if st.button("Use this folder now"):
            os.environ["NEWS_LOCAL_DIR"] = base_dir
            st.session_state["NEWS_LOCAL_DIR"] = base_dir
            st.success(f"Using folder: {base_dir}")

    start_iso, end_iso = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
    _render_config_summary(symbol, start_iso, end_iso, K, full, skip_existing, base_dir)

    controls = st.columns([1, 1, 1])
    start_label = "Resume download" if state.get("is_paused") else "Start download"
    with controls[0]:
        start_disabled = state.get("is_running", False)
        if st.button(start_label, type="primary", disabled=start_disabled):
            if state.get("is_paused") and state.get("params"):
                stored = state["params"]
                resume_params = {key: stored[key] for key in ("symbol", "start_iso", "end_iso", "K", "base_dir", "full_content", "skip_existing") if key in stored}
            else:
                resume_params = {
                    "symbol": symbol,
                    "start_iso": start_iso,
                    "end_iso": end_iso,
                    "K": K,
                    "base_dir": base_dir,
                    "full_content": full,
                    "skip_existing": skip_existing,
                }
            _start_download(state, **resume_params)

    with controls[1]:
        if st.button(
            "Pause after current day",
            disabled=not state.get("is_running") or state.get("pause_event").is_set(),
            help="Finishes the in-flight day and pauses before the next one.",
        ):
            state["pause_event"].set()

    with controls[2]:
        if st.button(
            "Refresh coverage snapshot",
            disabled=state.get("is_running", False),
        ):
            summary = prescan_days(
                symbol,
                start_iso,
                end_iso,
                K=K,
                base_dir=base_dir,
                full_content=full,
            )
            state["coverage_summary"] = summary

    if state.get("pause_event").is_set() and state.get("is_running"):
        st.info("Pause requested. The downloader will stop after the current day completes.")

    if state.get("is_running"):
        st.success("Download in progress…")
        _render_progress(state)
    elif state.get("is_paused"):
        st.warning("Download paused. Press **Resume download** to continue.")
    elif state.get("last_stats"):
        st.success("Most recent download finished.")
        _render_progress(state)

    _render_errors(state)

    if state.get("last_stats"):
        stats = state["last_stats"]
        st.markdown(
            f"**Latest run summary:** saved {stats.get('saved', 0)} days · "
            f"days with news {stats.get('days_with_news', 0)} · "
            f"content OK {stats.get('content_ok', 0)} · "
            f"content failed {stats.get('content_fail', 0)} · "
            f"skipped existing {stats.get('skipped_existing', 0)}"
        )
        if stats.get("retry_queue"):
            st.warning("Some days still require follow-up.")
            df_retry = pd.DataFrame(stats["retry_queue"])
            st.dataframe(df_retry, width="stretch")

    if state.get("coverage_summary"):
        st.caption("Local coverage snapshot")
        df_cover = pd.DataFrame(state["coverage_summary"])
        st.dataframe(df_cover, width="stretch")

    _render_history(state)

    st.divider()
    st.caption("Quick check: open one saved day (if exists)")
    preview_cols = st.columns([1, 1])
    with preview_cols[0]:
        test_day = st.text_input("Day YYYY-MM-DD", value=start_iso, key="preview_day")
    with preview_cols[1]:
        preview_symbol = st.text_input("Preview symbol", value=symbol, key="preview_symbol").upper().strip()
    if st.button("Open saved day", key="open_saved_day_button", help="Display cached articles if present."):
        arts, reason = load_local_day(preview_symbol, test_day, base_dir)
        if arts:
            st.success(f"{len(arts)} article(s) found — source `{reason}`")
            st.json({"symbol": preview_symbol, "date": test_day, "count": len(arts), "sample": arts[:2]})
        else:
            st.warning(f"No local file for {preview_symbol} {test_day} under {base_dir}")

    if state.get("is_running"):
        time.sleep(0.5)
        st.rerun()
