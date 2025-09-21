
import json
import os
from datetime import datetime, date

import pandas as pd
import plotly.graph_objects as go
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
from ui.config_editor import render_config_tab


def render_metric_card(target, title: str, value: str, caption: str = "") -> None:
    """Render a stylised KPI card inside the provided Streamlit container."""

    target.markdown(
        f"""
        <div class=\"metric-card\">
            <div class=\"metric-label\">{title}</div>
            <div class=\"metric-value\">{value}</div>
            <div class=\"metric-caption\">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_currency(value: float) -> str:
    """Format ``value`` as a whole-dollar currency string."""

    try:
        return f"${value:,.0f}"
    except Exception:
        return str(value)


def fmt_percent(value: float) -> str:
    """Format a fraction as a percentage string."""

    try:
        return f"{value * 100:.2f}%"
    except Exception:
        return str(value)


def fmt_shares(value: float) -> str:
    """Format share quantities trimming trailing zeros."""

    try:
        rounded = round(float(value), 6)
    except Exception:
        return f"{value} sh"
    formatted = f"{rounded:.6f}".rstrip("0").rstrip(".")
    if not formatted or formatted == "-0":
        formatted = "0"
    return f"{formatted} sh"


def _json_default(obj):
    """Serialize complex objects (numpy, pandas, datetimes) for JSON export."""

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return obj.__dict__
        except Exception:
            pass
    return str(obj)

st.set_page_config(page_title="FinMem Pro", layout="wide")

st.markdown(
    """
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        padding: 28px 32px;
        border-radius: 20px;
        color: #f8fafc;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at top right, rgba(56,189,248,0.35), transparent 55%),
                    radial-gradient(circle at bottom left, rgba(129,140,248,0.25), transparent 60%);
        pointer-events: none;
    }
    .hero-left {
        position: relative;
        z-index: 1;
        max-width: 560px;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .hero-subtitle {
        margin: 0;
        font-size: 1.05rem;
        color: rgba(248,250,252,0.85);
    }
    .hero-tags {
        position: relative;
        z-index: 1;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .meta-pill {
        background: rgba(15,23,42,0.35);
        border: 1px solid rgba(148,163,184,0.35);
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(248,250,252,0.95);
    }
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 12px 32px rgba(15,23,42,0.08);
        border: 1px solid rgba(148,163,184,0.18);
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        color: #475569;
    }
    .metric-card .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #111827;
        margin-top: 0.2rem;
    }
    .metric-card .metric-caption {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-left">
            <h1 class="hero-title">FinMem Pro</h1>
            <p class="hero-subtitle">Suite interactiva para entrenar y evaluar un agente de trading con memoria a largo plazo.</p>
        </div>
        <div class="hero-tags">
            <span class="meta-pill">Live dashboards</span>
            <span class="meta-pill">LLM-driven</span>
            <span class="meta-pill">Backtesting asistido</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Panel de control")
    cfg_path = st.text_input("Archivo de configuración", value="config.json")
    st.caption("Selecciona el archivo JSON que se utilizará en la sesión.")
    if "NEWS_LOCAL_DIR" in os.environ:
        st.caption(f"NEWS_LOCAL_DIR = {os.environ['NEWS_LOCAL_DIR']}")
    cfg_snapshot_sidebar = st.session_state.get("CONFIG_LAST")
    if cfg_snapshot_sidebar:
        st.markdown("### Configuración activa")
        st.markdown(
            f"""
            **Símbolo:** `{cfg_snapshot_sidebar.get('symbol', '?')}`<br>
            **Entrenamiento:** {cfg_snapshot_sidebar.get('train_start', '?')} → {cfg_snapshot_sidebar.get('train_end', '?')}<br>
            **Backtest:** {cfg_snapshot_sidebar.get('test_start', '?')} → {cfg_snapshot_sidebar.get('test_end', '?')}<br>
            **Memoria:** `{cfg_snapshot_sidebar.get('memory_path', '-')}`
            """,
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Usa las pestañas para editar la configuración, entrenar la memoria y lanzar backtests interactivos.")

tabs = st.tabs(["Config", "Entrenamiento", "Backtest", "News cache"])

# --- Config tab ---
with tabs[0]:
    render_config_tab(cfg_path)

# --- Training tab ---
with tabs[1]:
    st.subheader("🧠 Entrenamiento con memoria activa")
    st.caption(f"Usando configuración: {cfg_path}")

    cfg_snapshot = st.session_state.get("CONFIG_LAST")
    if cfg_snapshot:
        info_cols = st.columns(3)
        render_metric_card(info_cols[0].empty(), "Símbolo", cfg_snapshot.get("symbol", "?"), "Ticker objetivo")
        render_metric_card(info_cols[1].empty(), "Inicio", cfg_snapshot.get("train_start", "?"), "Período de entrenamiento")
        render_metric_card(info_cols[2].empty(), "Fin", cfg_snapshot.get("train_end", "?"), "Período de entrenamiento")

    progress_col, chart_col = st.columns([1, 2])

    with progress_col:
        status_placeholder = st.empty()
        status_placeholder.info("Esperando ejecución…")
        prog = st.progress(0.0, text="En espera")
        metrics_box = st.container()
        row1 = metrics_box.columns(2)
        capsules_metric = row1[0].empty()
        factors_metric = row1[1].empty()
        row2 = metrics_box.columns(2)
        info_metric = row2[0].empty()
        warn_metric = row2[1].empty()
        render_metric_card(capsules_metric, "Cápsulas", "0", "Generadas")
        render_metric_card(factors_metric, "Factores", "0", "Evaluados")
        render_metric_card(info_metric, "Mensajes", "0", "Info")
        render_metric_card(warn_metric, "Alertas", "0", "Warnings")

    with chart_col:
        st.markdown("#### Evolución de factores LLM")
        factor_chart_placeholder = st.empty()
        factor_chart_placeholder.info("Los factores aparecerán automáticamente al procesar la primera fecha.")

    event_tabs = st.tabs(["Eventos", "Cápsulas generadas", "Factores detallados"])
    with event_tabs[0]:
        event_log_container = st.container()
    with event_tabs[1]:
        capsule_container = st.container()
    with event_tabs[2]:
        factor_table_placeholder = st.empty()
        factor_table_placeholder.info("Sin factores registrados todavía.")

    factor_history: list[dict] = []
    train_event_records: list[dict] = []
    stats = {"capsules": 0, "factors": 0, "infos": 0, "warnings": 0}

    def render_factor_chart():
        """Plot the evolution of AI-derived factor scores."""

        if not factor_history:
            return
        df = pd.DataFrame(factor_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        fig = go.Figure()
        series = [
            ("mood_score", "Sentimiento"),
            ("narrative_bias", "Sesgo narrativo"),
            ("novelty", "Novedad"),
            ("credibility", "Credibilidad"),
            ("confidence", "Confianza"),
            ("regime_alignment", "Alineación de régimen"),
        ]
        for key, label in series:
            if key in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df[key],
                        mode="lines+markers",
                        name=label,
                        line=dict(width=2),
                    )
                )
        fig.update_layout(
            margin=dict(l=16, r=16, t=32, b=16),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=360,
        )
        fig.update_xaxes(title="Fecha")
        fig.update_yaxes(title="Valor normalizado")
        factor_chart_placeholder.plotly_chart(fig, use_container_width=True)

    def render_factor_table():
        """Display the latest factor breakdown in a table."""

        if not factor_history:
            return
        df = pd.DataFrame(factor_history)
        cols = [
            col
            for col in [
                "date",
                "mood_score",
                "narrative_bias",
                "novelty",
                "credibility",
                "confidence",
                "regime_alignment",
            ]
            if col in df.columns
        ]
        if not cols:
            return
        tbl = df[cols].copy()
        tbl["date"] = pd.to_datetime(tbl["date"])
        tbl = tbl.sort_values("date").set_index("date")
        tbl = tbl.rename(
            columns={
                "mood_score": "sentimiento",
                "narrative_bias": "sesgo",
                "novelty": "novedad",
                "credibility": "credibilidad",
                "confidence": "confianza",
                "regime_alignment": "régimen",
            }
        )
        factor_table_placeholder.dataframe(tbl.tail(15), use_container_width=True)

    def on_train_event(evt):
        """Handle training callbacks and update the UI."""

        timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        record = {"timestamp": timestamp}
        if isinstance(evt, dict):
            try:
                record.update(json.loads(json.dumps(evt, default=_json_default)))
            except Exception:
                record.update({k: evt.get(k) for k in evt})
        else:
            try:
                record["event"] = json.loads(json.dumps(evt, default=_json_default))
            except Exception:
                record["event"] = str(evt)
        train_event_records.append(record)

        t = evt.get("type")
        if t == "phase":
            status_placeholder.info(f"Fase: {evt.get('label')} → {evt.get('state')}")
        elif t == "progress":
            i, n = evt.get("i", 0), evt.get("n", 1)
            prog.progress(min(1.0, i / max(1, n)), text=f"{i}/{n}")
        elif t == "capsule":
            stats["capsules"] += 1
            render_metric_card(capsules_metric, "Cápsulas", str(stats["capsules"]), "Generadas")
            with capsule_container.expander(f"Cápsula {evt.get('date', '')}"):
                st.json(evt.get("capsule"))
        elif t == "factor":
            stats["factors"] += 1
            render_metric_card(factors_metric, "Factores", str(stats["factors"]), "Evaluados")
            factor = evt.get("factor") or {}
            record = {"date": evt.get("date")}
            for key in [
                "mood_score",
                "narrative_bias",
                "novelty",
                "credibility",
                "confidence",
                "regime_alignment",
            ]:
                record[key] = factor.get(key)
            factor_history.append(record)
            render_factor_chart()
            render_factor_table()
        elif t == "info":
            stats["infos"] += 1
            render_metric_card(info_metric, "Mensajes", str(stats["infos"]), "Info")
            event_log_container.info(evt.get("message", ""))
        elif t == "warn":
            stats["warnings"] += 1
            render_metric_card(warn_metric, "Alertas", str(stats["warnings"]), "Warnings")
            event_log_container.warning(evt.get("message", ""))
        elif t == "done":
            status_placeholder.success("Entrenamiento finalizado")

    if st.button("Ejecutar entrenamiento", type="primary"):
        stats.update({"capsules": 0, "factors": 0, "infos": 0, "warnings": 0})
        factor_history.clear()
        train_event_records.clear()
        render_metric_card(capsules_metric, "Cápsulas", "0", "Generadas")
        render_metric_card(factors_metric, "Factores", "0", "Evaluados")
        render_metric_card(info_metric, "Mensajes", "0", "Info")
        render_metric_card(warn_metric, "Alertas", "0", "Warnings")
        factor_chart_placeholder.info("Los factores aparecerán automáticamente al procesar la primera fecha.")
        factor_table_placeholder.info("Sin factores registrados todavía.")
        res = run_training(cfg_path, on_event=on_train_event)
        st.success("Entrenamiento completado")
        st.subheader("📦 Snapshot de memoria")
        st.json(res.get("memory_snapshot", {}))

        stats_snapshot = dict(stats)
        log_payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "stage": "training",
            "config_path": cfg_path,
            "config_snapshot": cfg_snapshot,
            "stats": stats_snapshot,
            "factor_history": list(factor_history),
            "events": list(train_event_records),
            "result": res,
        }
        train_log_bytes = json.dumps(
            log_payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ).encode("utf-8")
        train_log_name = f"training_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.json"
        st.session_state["TRAIN_LOG_FILE"] = {
            "file_name": train_log_name,
            "data": train_log_bytes,
        }

    train_log_state = st.session_state.get("TRAIN_LOG_FILE")
    if train_log_state and train_log_state.get("data"):
        st.download_button(
            "⬇️ Descargar log de entrenamiento",
            data=train_log_state["data"],
            file_name=train_log_state.get("file_name", "training_log.json"),
            mime="application/json",
            key="download_training_log",
        )

# --- Backtest tab ---
with tabs[2]:
    st.subheader("📈 Backtest interactivo")
    st.caption(f"Usando configuración: {cfg_path}")

    cfg_snapshot_bt = st.session_state.get("CONFIG_LAST")
    if cfg_snapshot_bt:
        info_cols = st.columns(4)
        render_metric_card(info_cols[0].empty(), "Símbolo", cfg_snapshot_bt.get("symbol", "?"), "Activo evaluado")
        render_metric_card(info_cols[1].empty(), "Inicio", cfg_snapshot_bt.get("test_start", "?"), "Periodo de test")
        render_metric_card(info_cols[2].empty(), "Fin", cfg_snapshot_bt.get("test_end", "?"), "Periodo de test")
        render_metric_card(
            info_cols[3].empty(),
            "Capital inicial",
            fmt_currency(float(cfg_snapshot_bt.get("initial_cash", 0.0))),
            "Simulación",
        )

    update_rate = st.slider("Intervalo de eventos (días)", min_value=1, max_value=30, value=10)

    prog_col, charts_col = st.columns([1, 2])

    with prog_col:
        status_bt = st.empty()
        status_bt.info("Esperando ejecución…")
        prog2 = st.progress(0.0, text="En espera")
        metrics_box = st.container()
        row1 = metrics_box.columns(2)
        equity_metric = row1[0].empty()
        cash_metric = row1[1].empty()
        row2 = metrics_box.columns(2)
        position_metric = row2[0].empty()
        drawdown_metric = row2[1].empty()
        row3 = metrics_box.columns(2)
        decisions_metric = row3[0].empty()
        messages_metric = row3[1].empty()
        render_metric_card(equity_metric, "Equity", fmt_currency(0.0), "Valor de la cartera")
        render_metric_card(cash_metric, "Efectivo", fmt_currency(0.0), "Disponible")
        render_metric_card(position_metric, "Posición", fmt_shares(0.0), "Acciones netas")
        render_metric_card(drawdown_metric, "Drawdown", "0.00%", "Desde el máximo")
        render_metric_card(decisions_metric, "Señales", "0", "Decisiones emitidas")
        render_metric_card(messages_metric, "Mensajes", "0 info / 0 warn", "Logs recibidos")
        row4 = metrics_box.columns(2)
        pnl_metric = row4[0].empty()
        pnl_pct_metric = row4[1].empty()
        render_metric_card(pnl_metric, "Ganancia/Pérdida", fmt_currency(0.0), "Sobre capital invertido")
        render_metric_card(pnl_pct_metric, "Rentabilidad", fmt_percent(0.0), "Equity vs. aportes")

    with charts_col:
        st.markdown("#### Evolución del patrimonio")
        equity_chart_placeholder = st.empty()
        equity_chart_placeholder.info("La curva de equity se dibujará al recibir los primeros datos.")
        st.markdown("#### Drawdown acumulado")
        drawdown_chart_placeholder = st.empty()
        drawdown_chart_placeholder.info("El drawdown aparecerá en tiempo real durante el backtest.")

    bt_tabs = st.tabs(["Señales del LLM", "Contexto diario", "Mensajes"])
    with bt_tabs[0]:
        decision_container = st.container()
    with bt_tabs[1]:
        context_container = st.container()
    with bt_tabs[2]:
        message_container = st.container()

    equity_history: list[dict] = []
    test_event_records: list[dict] = []

    def render_equity_chart():
        """Render the backtest equity curve and benchmark chart."""

        if not equity_history:
            return
        df = pd.DataFrame(equity_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["equity"],
                name="Estrategia",
                mode="lines",
                line=dict(width=3, color="#38bdf8"),
            )
        )
        if "benchmark" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["benchmark"],
                    name="Buy & Hold",
                    mode="lines",
                    line=dict(width=2, dash="dash", color="#facc15"),
                )
            )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=16, r=16, t=32, b=16),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360,
        )
        fig.update_yaxes(title="Equity")
        fig.update_xaxes(title="Fecha")
        equity_chart_placeholder.plotly_chart(fig, use_container_width=True)

    def render_drawdown_chart():
        """Render the backtest drawdown chart."""

        if not equity_history:
            return
        df = pd.DataFrame(equity_history)
        if "drawdown" not in df.columns:
            return
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["drawdown"] * 100.0,
                mode="lines",
                name="Drawdown",
                line=dict(color="#f87171", width=2),
                fill="tozeroy",
                fillcolor="rgba(248,113,113,0.25)",
            )
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=16, r=16, t=32, b=16),
            height=320,
            yaxis=dict(ticksuffix="%"),
        )
        fig.update_xaxes(title="Fecha")
        drawdown_chart_placeholder.plotly_chart(fig, use_container_width=True)

    stats_bt = {"decisions": 0, "infos": 0, "warnings": 0}
    cfg_initial_cash = 0.0
    if cfg_snapshot_bt is not None:
        try:
            cfg_initial_cash = float(cfg_snapshot_bt.get("initial_cash", 0.0) or 0.0)
        except Exception:
            cfg_initial_cash = 0.0
    last_equity_point = None

    def update_messages_metric():
        """Update the UI metric that counts LLM calls."""

        render_metric_card(
            messages_metric,
            "Mensajes",
            f"{stats_bt['infos']} info / {stats_bt['warnings']} warn",
            "Logs recibidos",
        )

    def on_test_event(evt):
        """Handle backtest events and refresh visual elements."""

        nonlocal last_equity_point
        timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        record = {"timestamp": timestamp}
        if isinstance(evt, dict):
            try:
                record.update(json.loads(json.dumps(evt, default=_json_default)))
            except Exception:
                record.update({k: evt.get(k) for k in evt})
        else:
            try:
                record["event"] = json.loads(json.dumps(evt, default=_json_default))
            except Exception:
                record["event"] = str(evt)
        test_event_records.append(record)

        t = evt.get("type")
        if t == "phase":
            status_bt.info(f"Fase: {evt.get('label')} → {evt.get('state')}")
        elif t == "progress":
            i, n = evt.get("i", 0), evt.get("n", 1)
            prog2.progress(min(1.0, i / max(1, n)), text=f"{i}/{n}")
        elif t == "decision":
            stats_bt["decisions"] += 1
            render_metric_card(
                decisions_metric,
                "Señales",
                str(stats_bt["decisions"]),
                "Decisiones emitidas",
            )
            with decision_container.expander(f"Decisión {evt.get('date', '')}"):
                st.json(evt.get("decision"))
        elif t == "capsule":
            with context_container.expander(f"Contexto {evt.get('date', '')}"):
                st.json(evt.get("capsule"))
        elif t == "info":
            stats_bt["infos"] += 1
            message_container.info(evt.get("message", ""))
            update_messages_metric()
        elif t == "warn":
            stats_bt["warnings"] += 1
            message_container.warning(evt.get("message", ""))
            update_messages_metric()
        elif t == "equity_point":
            equity_history.append(evt)
            render_equity_chart()
            render_drawdown_chart()
            render_metric_card(equity_metric, "Equity", fmt_currency(evt.get("equity", 0.0)), "Valor de la cartera")
            render_metric_card(cash_metric, "Efectivo", fmt_currency(evt.get("cash", 0.0)), "Disponible")
            render_metric_card(
                drawdown_metric,
                "Drawdown",
                fmt_percent(evt.get("drawdown", 0.0)),
                "Desde el máximo",
            )
            render_metric_card(
                position_metric,
                "Posición",
                fmt_shares(evt.get("position", 0.0)),
                "Acciones netas",
            )
            total_contributions = float(evt.get("total_contributions", 0.0) or 0.0)
            invested_capital = cfg_initial_cash + total_contributions
            pnl_amount = float(evt.get("equity", 0.0) or 0.0) - invested_capital
            pnl_pct = (pnl_amount / invested_capital) if invested_capital > 0 else 0.0
            render_metric_card(
                pnl_metric,
                "Ganancia/Pérdida",
                fmt_currency(pnl_amount),
                f"Capital invertido: {fmt_currency(invested_capital)}",
            )
            render_metric_card(
                pnl_pct_metric,
                "Rentabilidad",
                fmt_percent(pnl_pct),
                "Equity vs. aportes",
            )
            last_equity_point = dict(evt)
            contribution = float(evt.get("contribution", 0.0) or 0.0)
            if contribution > 0:
                message_container.info(
                    f"{evt.get('date', '')}: aporte automático de {fmt_currency(contribution)}"
                )
        elif t == "done":
            status_bt.success("Backtest finalizado")

    if st.button("Ejecutar backtest", type="primary"):
        equity_history.clear()
        test_event_records.clear()
        render_metric_card(equity_metric, "Equity", fmt_currency(0.0), "Valor de la cartera")
        render_metric_card(cash_metric, "Efectivo", fmt_currency(0.0), "Disponible")
        render_metric_card(position_metric, "Posición", fmt_shares(0.0), "Acciones netas")
        render_metric_card(drawdown_metric, "Drawdown", "0.00%", "Desde el máximo")
        render_metric_card(decisions_metric, "Señales", "0", "Decisiones emitidas")
        render_metric_card(messages_metric, "Mensajes", "0 info / 0 warn", "Logs recibidos")
        render_metric_card(pnl_metric, "Ganancia/Pérdida", fmt_currency(0.0), "Sobre capital invertido")
        render_metric_card(pnl_pct_metric, "Rentabilidad", fmt_percent(0.0), "Equity vs. aportes")
        equity_chart_placeholder.info("La curva de equity se dibujará al recibir los primeros datos.")
        drawdown_chart_placeholder.info("El drawdown aparecerá en tiempo real durante el backtest.")
        stats_bt.update({"decisions": 0, "infos": 0, "warnings": 0})
        last_equity_point = None
        res = run_backtest(cfg_path, on_event=on_test_event, event_rate=int(update_rate))
        st.success("Backtest completado")

        metrics = res.get("metrics", {})
        if metrics:
            st.markdown("#### Indicadores de rendimiento")
            metric_cols = st.columns(4)
            render_metric_card(metric_cols[0].empty(), "CAGR", fmt_percent(metrics.get("CAGR", 0.0)), "Estrategia")
            render_metric_card(metric_cols[1].empty(), "Sharpe", f"{metrics.get('Sharpe', 0.0):.2f}", "Rendimiento ajustado")
            render_metric_card(metric_cols[2].empty(), "Sortino", f"{metrics.get('Sortino', 0.0):.2f}", "Riesgo a la baja")
            render_metric_card(
                metric_cols[3].empty(),
                "Max DD",
                fmt_percent(metrics.get("MaxDrawdown", 0.0)),
                "Caída máxima",
            )
            extra_cols = st.columns(3)
            render_metric_card(extra_cols[0].empty(), "Volatilidad", fmt_percent(metrics.get("Volatilidad", 0.0)), "Anualizada")
            render_metric_card(extra_cols[1].empty(), "BH CAGR", fmt_percent(metrics.get("BH_CAGR", 0.0)), "Buy & Hold")
            render_metric_card(extra_cols[2].empty(), "Active Return", f"{metrics.get('ActiveReturn', 0.0):.2f}", "vs. benchmark")

        contrib = res.get("contributions")
        if contrib:
            st.markdown(
                f"**Aportes periódicos:** {fmt_currency(contrib.get('amount', 0.0))} · "
                f"Frecuencia: {contrib.get('frequency', 'none')} · "
                f"Total inyectado: {fmt_currency(contrib.get('total_added', 0.0))}"
            )

        if last_equity_point:
            total_contributions = float(last_equity_point.get("total_contributions", 0.0) or 0.0)
            invested_capital = cfg_initial_cash + total_contributions
            pnl_amount = float(last_equity_point.get("equity", 0.0) or 0.0) - invested_capital
            pnl_pct = (pnl_amount / invested_capital) if invested_capital > 0 else 0.0
            st.markdown("#### Resultado acumulado")
            summary_cols = st.columns(2)
            render_metric_card(
                summary_cols[0].empty(),
                "Ganancia/Pérdida",
                fmt_currency(pnl_amount),
                f"Capital invertido: {fmt_currency(invested_capital)}",
            )
            render_metric_card(
                summary_cols[1].empty(),
                "Rentabilidad",
                fmt_percent(pnl_pct),
                "Equity vs. aportes",
            )

        st.markdown("#### Reportes finales")
        st.pyplot(res.get("fig_equity"))
        st.pyplot(res.get("fig_drawdown"))

        if "trades_tail" in res:
            st.markdown("#### Últimas operaciones ejecutadas")
            st.dataframe(res["trades_tail"], use_container_width=True)

        stats_bt_snapshot = dict(stats_bt)
        res_for_log = {k: v for k, v in res.items() if k not in {"fig_equity", "fig_drawdown"}}
        log_payload = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "stage": "backtest",
            "config_path": cfg_path,
            "config_snapshot": cfg_snapshot_bt,
            "event_rate": int(update_rate),
            "stats": stats_bt_snapshot,
            "equity_history": list(equity_history),
            "last_equity_point": last_equity_point,
            "events": list(test_event_records),
            "result": res_for_log,
        }
        backtest_log_bytes = json.dumps(
            log_payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ).encode("utf-8")
        backtest_log_name = f"backtest_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z.json"
        st.session_state["BACKTEST_LOG_FILE"] = {
            "file_name": backtest_log_name,
            "data": backtest_log_bytes,
        }

    backtest_log_state = st.session_state.get("BACKTEST_LOG_FILE")
    if backtest_log_state and backtest_log_state.get("data"):
        st.download_button(
            "⬇️ Descargar log de backtest",
            data=backtest_log_state["data"],
            file_name=backtest_log_state.get("file_name", "backtest_log.json"),
            mime="application/json",
            key="download_backtest_log",
        )

# --- News cache tab ---
with tabs[3]:
    render_news_tab()
