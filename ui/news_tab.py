import os
from datetime import date
import streamlit as st
from core.news_store import download_range, load_local_day

def render_news_tab():
    st.subheader("📥 News cache")

    with st.expander("Settings", expanded=True):
        col1, col2 = st.columns([2,1])
        with col1:
            base_dir = st.text_input("Local folder", value=os.environ.get("NEWS_LOCAL_DIR","data/news_local"))
        with col2:
            st.write(" ")
            if st.button("Use this folder now"):
                os.environ["NEWS_LOCAL_DIR"] = base_dir
                st.session_state["NEWS_LOCAL_DIR"] = base_dir
                st.success(f"Using folder: {base_dir}")

        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Symbol", value="AAPL").upper().strip()
        with c2:
            start_d = st.date_input("Start date", value=date(2022,1,1))
        with c3:
            end_d = st.date_input("End date", value=date(2022,3,31))

        K = st.number_input("Headlines per day (K)", min_value=1, max_value=50, value=10, step=1)
        full = st.checkbox("Also download full article content", value=True, help="Fetch and store readable text for each news URL.")
        skip_existing = st.checkbox("Skip days already downloaded (resume)", value=True, help="If a day exists locally with enough articles and (optionally) content, skip fetching.")

    if st.button("Download & save news", type="primary"):
        start_iso, end_iso = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
        total_days = max(1, (end_d - start_d).days + 1)

        progress = st.progress(0.0)
        status = st.empty()

        cnt = {"i": 0}

        plan_holder = st.expander("Plan (pre-scan)", expanded=False)

        def on_evt(evt: dict):
            if evt.get("type") == "plan":
                plan = evt.get("plan", [])
                if plan:
                    import pandas as pd
                    df = pd.DataFrame(plan)
                    try:
                        from caas_jupyter_tools import display_dataframe_to_user as _disp
                        _disp("Download plan", df)
                    except Exception:
                        plan_holder.write(df)
                return
            if evt.get("type") != "progress":
                return
            cnt["i"] += 1
            prov = evt.get("provider") or "unknown"
            day = evt.get("date", "")
            saved_path = evt.get("saved_path", "")
            ok = evt.get("content_ok", 0)
            fail = evt.get("content_fail", 0)
            synt = evt.get("synthetic_days", 0)
            real = evt.get("real_days", 0)
            skipped = evt.get("skipped_existing", 0)

            progress.progress(min(1.0, cnt["i"] / total_days))
            status.write(f"{day}: **{prov}** → `{saved_path}` · content ok={ok} fail={fail} · real={real} synthetic={synt} · skipped={skipped}")

        stats = download_range(
            symbol,
            start_iso,
            end_iso,
            K=int(K),
            base_dir=base_dir,
            on_event=on_evt,
            full_content=bool(full),
            skip_existing=bool(skip_existing),
        )

        st.success(
            f"{stats.get('saved',0)}/{total_days} saved — "
            f"{stats.get('last_date','')} last · "
            f"real={stats.get('real_days',0)} synthetic={stats.get('synthetic_days',0)} | "
            f"content ok={stats.get('content_ok',0)} fail={stats.get('content_fail',0)} · "
            f"skipped={stats.get('skipped_existing',0)}"
        )
        st.session_state["NEWS_LOCAL_DIR"] = base_dir
        os.environ["NEWS_LOCAL_DIR"] = base_dir

    st.divider()
    st.caption("Quick check: open one saved day (if exists)")
    test_day = st.text_input("Day YYYY-MM-DD", value="2022-01-03")
    if st.button("Open saved day"):
        arts, reason = load_local_day(symbol, test_day, base_dir)
        if arts:
            st.json({"symbol": symbol, "date": test_day, "provider_or_origin": reason, "count": len(arts), "sample": arts[:1]})
        else:
            st.warning(f"No local file for {symbol} {test_day} under {base_dir}")
