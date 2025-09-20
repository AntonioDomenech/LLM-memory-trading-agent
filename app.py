
import os
import time
from datetime import datetime
import streamlit as st

# --- begin: path+env bootstrap ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    # load .env placed at project root; do not override already-set variables (e.g., OS env or Streamlit secrets)
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
except Exception:
    pass
# --- end: path+env bootstrap ---


from core.config import load_config
from core.train import run_training
from core.backtest import run_backtest
from ui.news_tab import render_news_tab

st.set_page_config(page_title="FinMem Pro", layout="wide")
st.title("FinMem Pro")

with st.sidebar:
    st.subheader("Config")
    cfg_path = st.text_input("Config file", value="config.json")
    if "NEWS_LOCAL_DIR" in os.environ:
        st.caption(f"NEWS_LOCAL_DIR = {os.environ['NEWS_LOCAL_DIR']}")

tabs = st.tabs(["Entrenamiento", "Backtest", "News cache"])

# --- Training tab ---
with tabs[0]:
    st.subheader("Entrenamiento")
    spot = st.empty()
    prog = st.progress(0.0, text="Idle")
    log = st.container()
    def on_train_event(evt):
        t = evt.get("type")
        if t == "phase":
            spot.info(f"Fase: {evt.get('label')} → {evt.get('state')}")
        elif t == "progress":
            i, n = evt.get("i",0), evt.get("n",1)
            prog.progress(min(1.0, i/max(1,n)), text=f"{i}/{n}")
        elif t == "capsule":
            with log.expander(f"Cápsula {evt.get('date','')}"):
                st.json(evt.get("capsule"))
        elif t == "factor":
            with log.expander(f"Factor LLM {evt.get('date','')}"):
                st.json(evt.get("factor"))
        elif t == "info":
            st.info(evt.get("message",""))
        elif t == "warn":
            st.warning(evt.get("message",""))
        elif t == "done":
            st.success("Entrenamiento finalizado")
    if st.button("Ejecutar entrenamiento", type="primary"):
        res = run_training(cfg_path, on_event=on_train_event)
        st.subheader("Memory snapshot")
        st.json(res.get("memory_snapshot", {}))

# --- Backtest tab ---
with tabs[1]:
    st.subheader("Backtest")
    update_rate = st.number_input("Event rate (days)", min_value=1, max_value=50, value=10)
    spot2 = st.empty()
    prog2 = st.progress(0.0, text="Idle")
    decs = st.container()
    def on_test_event(evt):
        t = evt.get("type")
        if t == "phase":
            spot2.info(f"Fase: {evt.get('label')} → {evt.get('state')}")
        elif t == "progress":
            i, n = evt.get("i",0), evt.get("n",1)
            prog2.progress(min(1.0, i/max(1,n)), text=f"{i}/{n}")
        elif t == "capsule":
            with decs.expander(f"Cápsula {evt.get('date','')}"):
                st.json(evt.get("capsule"))
        elif t == "decision":
            with decs.expander(f"Decisión {evt.get('date','')}"):
                st.json(evt.get("decision"))
        elif t == "info":
            st.info(evt.get("message",""))
        elif t == "warn":
            st.warning(evt.get("message",""))
    if st.button("Ejecutar backtest", type="primary"):
        res = run_backtest(cfg_path, on_event=on_test_event, event_rate=int(update_rate))
        st.subheader("Rendimiento")
        st.json(res.get("metrics", {}))
        st.pyplot(res.get("fig_equity"))
        st.pyplot(res.get("fig_drawdown"))
        if "trades_tail" in res:
            st.subheader("Últimas operaciones")
            st.dataframe(res["trades_tail"])

# --- News cache tab ---
with tabs[2]:
    render_news_tab()
