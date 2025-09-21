
import os

import pandas as pd
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


from core.train import run_training
from core.backtest import run_backtest
from ui.news_tab import render_news_tab
from ui.config_editor import render_config_tab

st.set_page_config(page_title="FinMem Pro", layout="wide", page_icon="📈")
st.title("FinMem Pro")
st.caption(
    "Laboratorio interactivo para entrenar memorias con noticias financieras y evaluar estrategias LLM."
)

with st.sidebar:
    st.header("Panel de control")
    cfg_path = st.text_input(
        "Archivo de configuración",
        value="config.json",
        help="Ruta hacia el JSON principal con fechas, modelos y parámetros de riesgo.",
    )
    st.caption("Los cambios se guardan directamente sobre el archivo indicado.")
    if "NEWS_LOCAL_DIR" in os.environ:
        st.caption(f"📁 NEWS_LOCAL_DIR = {os.environ['NEWS_LOCAL_DIR']}")
    st.markdown("---")
    st.markdown(
        "<small>Consejo: actualiza la pestaña Configuración antes de entrenar o lanzar un backtest para"
        " mantener sincronizada la sesión.</small>",
        unsafe_allow_html=True,
    )

tabs = st.tabs(["⚙️ Configuración", "🧠 Entrenamiento", "📊 Backtest", "📰 News cache"])

# --- Config tab ---
with tabs[0]:
    render_config_tab(cfg_path)

# --- Training tab ---
with tabs[1]:
    st.header("🧠 Entrenamiento del banco de memoria")
    st.markdown(
        "Genera o actualiza recuerdos contextualizados a partir de noticias históricas y series de precios."
    )

    cfg_snapshot = st.session_state.get("CONFIG_LAST") or {}
    if cfg_snapshot:
        cols = st.columns(4)
        cols[0].metric("Símbolo", cfg_snapshot.get("symbol", "—"))
        cols[1].metric(
            "Periodo de entrenamiento",
            f"{cfg_snapshot.get('train_start', '—')} → {cfg_snapshot.get('train_end', '—')}",
        )
        cols[2].metric("Modelo de decisión", cfg_snapshot.get("decision_model", "—"))
        cols[3].metric("Titulares K/día", cfg_snapshot.get("K_news_per_day", "—"))
    else:
        st.info("Carga una configuración en la pestaña Configuración para habilitar el entrenamiento.")

    st.divider()

    status_placeholder = st.empty()
    progress_placeholder = st.progress(0.0, text="Listo para ejecutar")

    stream_tabs = st.tabs(["Eventos en vivo", "Cápsulas generadas"])
    with stream_tabs[0]:
        train_events = st.container()
    with stream_tabs[1]:
        train_capsules = st.container()

    def on_train_event(evt):
        t = evt.get("type")
        if t == "phase":
            status_placeholder.info(f"Fase: {evt.get('label')} → {evt.get('state')}")
        elif t == "progress":
            i, n = evt.get("i", 0), evt.get("n", 1)
            progress_placeholder.progress(min(1.0, i / max(1, n)), text=f"{i}/{n}")
        elif t == "capsule":
            with train_capsules.expander(f"Cápsula {evt.get('date', '')}", expanded=False):
                st.json(evt.get("capsule"))
        elif t == "factor":
            with train_events.expander(f"Factor LLM {evt.get('date', '')}", expanded=False):
                st.json(evt.get("factor"))
        elif t == "info":
            train_events.info(evt.get("message", ""))
        elif t == "warn":
            train_events.warning(evt.get("message", ""))
        elif t == "done":
            status_placeholder.success("Entrenamiento finalizado")
            progress_placeholder.progress(1.0, text="Completado")

    if st.button("Ejecutar entrenamiento", type="primary", use_container_width=True):
        train_events.empty()
        train_capsules.empty()
        progress_placeholder.progress(0.0, text="Inicializando")
        status_placeholder.info("Preparando datos…")
        result = run_training(cfg_path, on_event=on_train_event)
        status_placeholder.success("Memoria entrenada correctamente")
        st.success("Entrenamiento completado")
        with st.expander("Instantánea de memoria", expanded=False):
            st.json(result.get("memory_snapshot", {}))
        artifacts = result.get("artifacts") or {}
        if artifacts:
            st.caption("Artefactos generados durante el proceso")
            st.write(artifacts)

# --- Backtest tab ---
with tabs[2]:
    st.header("📊 Backtest y evaluación")
    st.markdown(
        "Simula el desempeño diario del agente utilizando la memoria entrenada y compara contra un benchmark buy & hold."
    )

    cfg_snapshot_bt = st.session_state.get("CONFIG_LAST") or {}
    if cfg_snapshot_bt:
        cols = st.columns(4)
        cols[0].metric("Símbolo", cfg_snapshot_bt.get("symbol", "—"))
        cols[1].metric(
            "Ventana de test",
            f"{cfg_snapshot_bt.get('test_start', '—')} → {cfg_snapshot_bt.get('test_end', '—')}",
        )
        cols[2].metric(
            "Capital inicial",
            f"${float(cfg_snapshot_bt.get('initial_cash', 100000.0)) :,.0f}",
        )
        cols[3].metric("Modelo de decisión", cfg_snapshot_bt.get("decision_model", "—"))
    else:
        st.info("Configura primero la pestaña Configuración para ejecutar el backtest.")

    st.divider()

    controls_col, rate_col = st.columns([3, 1])
    with controls_col:
        st.caption("Actualiza la frecuencia de eventos para controlar cuántos días se muestran en vivo.")
    with rate_col:
        update_rate = st.number_input(
            "Notificar cada (días)", min_value=1, max_value=50, value=10, step=1
        )

    bt_status = st.empty()
    bt_progress = st.progress(0.0, text="Listo para ejecutar")

    backtest_tabs = st.tabs(["Decisiones en vivo", "Contexto diario"])
    with backtest_tabs[0]:
        decision_stream = st.container()
    with backtest_tabs[1]:
        context_stream = st.container()

    def on_test_event(evt):
        t = evt.get("type")
        if t == "phase":
            bt_status.info(f"Fase: {evt.get('label')} → {evt.get('state')}")
        elif t == "progress":
            i, n = evt.get("i", 0), evt.get("n", 1)
            bt_progress.progress(min(1.0, i / max(1, n)), text=f"{i}/{n}")
        elif t == "capsule":
            with context_stream.expander(f"Cápsula {evt.get('date', '')}", expanded=False):
                st.json(evt.get("capsule"))
        elif t == "decision":
            payload = evt.get("decision", {})
            headline = (
                f"{evt.get('date', '')}: {payload.get('action', 'HOLD')}"
                f" · target={payload.get('target_exposure', 0.0):.2f}"
            )
            with decision_stream.expander(headline, expanded=False):
                st.json(payload)
        elif t == "info":
            decision_stream.info(evt.get("message", ""))
        elif t == "warn":
            decision_stream.warning(evt.get("message", ""))

    if st.button("Ejecutar backtest", type="primary", use_container_width=True):
        decision_stream.empty()
        context_stream.empty()
        bt_progress.progress(0.0, text="Inicializando")
        bt_status.info("Preparando simulación…")
        result = run_backtest(
            cfg_path,
            on_event=on_test_event,
            event_rate=int(update_rate),
        )
        bt_status.success("Backtest completado")
        st.success("Simulación finalizada")

        metrics = result.get("metrics") or {}
        if metrics:
            st.markdown("#### Indicadores clave")
            mcols = st.columns(3)

            def _pct(val: float) -> str:
                return f"{val * 100:.2f}%"

            mcols[0].metric(
                "CAGR estrategia",
                _pct(metrics.get("CAGR", 0.0)),
                delta=f"BH {_pct(metrics.get('BH_CAGR', 0.0))}",
            )
            mcols[1].metric(
                "Sharpe",
                f"{metrics.get('Sharpe', 0.0):.2f}",
                delta=f"Sortino {metrics.get('Sortino', 0.0):.2f}",
            )
            mcols[2].metric(
                "Máx. drawdown",
                _pct(metrics.get("MaxDrawdown", 0.0)),
                delta=f"Volatilidad {_pct(metrics.get('Volatilidad', 0.0))}",
            )

        equity_curve = result.get("equity_curve")
        if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
            st.markdown("#### Evolución del equity")
            equity_display = equity_curve.copy()
            equity_display = equity_display.rename(
                columns={
                    "strategy_equity": "Estrategia",
                    "benchmark_equity": "Buy & Hold",
                }
            )
            st.line_chart(equity_display, height=340)
            csv_equity = equity_display.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar curva (CSV)",
                data=csv_equity,
                file_name="equity_curve.csv",
                mime="text/csv",
            )

        drawdown_curve = result.get("drawdown_curve")
        if isinstance(drawdown_curve, pd.DataFrame) and not drawdown_curve.empty:
            st.markdown("#### Drawdown acumulado")
            st.area_chart(drawdown_curve, height=220)

        with st.expander("Visualizaciones clásicas", expanded=False):
            st.pyplot(result.get("fig_equity"))
            st.pyplot(result.get("fig_drawdown"))

        trades_tail = result.get("trades_tail")
        if isinstance(trades_tail, pd.DataFrame) and not trades_tail.empty:
            st.markdown("#### Últimas operaciones registradas")
            st.dataframe(trades_tail, use_container_width=True)
            st.download_button(
                "Descargar operaciones (CSV)",
                data=trades_tail.to_csv(index=False).encode("utf-8"),
                file_name="recent_trades.csv",
                mime="text/csv",
            )

# --- News cache tab ---
with tabs[3]:
    render_news_tab()
