import json
import os
from datetime import date, datetime

import streamlit as st

from core.config import (
    Config,
    RetrievalCfg,
    RiskCfg,
    default_memory_path,
    load_config,
    resolve_memory_path,
    save_config,
)


def _parse_date(value: str, fallback: date) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return fallback


def _format_symbol(value: str) -> str:
    return (value or "").strip().upper()


def render_config_tab(cfg_path: str) -> None:
    st.subheader("⚙️ Editor de configuración")
    st.caption(
        "Ajusta el archivo JSON que controla el entrenamiento, la memoria y el backtest."
        " Cada control incluye una explicación sobre cómo impacta al agente." 
    )

    if not cfg_path:
        st.warning("Ingresa una ruta de configuración en la barra lateral para comenzar.")
        return

    try:
        cfg = load_config(cfg_path)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo `{cfg_path}`.")
        if st.button("Crear archivo con valores por defecto"):
            default_cfg = Config()
            save_config(cfg_path, default_cfg)
            st.success("Se creó un archivo de configuración nuevo.")
        return
    except json.JSONDecodeError as exc:
        st.error(f"El archivo de configuración contiene errores JSON: {exc}")
        return

    st.session_state["CONFIG_LAST"] = cfg.to_dict()

    memory_path = resolve_memory_path(cfg, prefer_existing=True)
    memory_exists = os.path.exists(memory_path)

    st.markdown(
        "Los rangos de fechas determinan qué datos alimentan al modelo,"
        " mientras que las opciones de memoria y riesgo definen cómo se generan"
        " y se utilizan las cápsulas de contexto durante la toma de decisiones."
    )

    if memory_exists:
        st.success(
            f"Memoria entrenada detectada en `{memory_path}` para {cfg.symbol}."
            " Se reutilizará automáticamente en los backtests."
        )
    else:
        st.info(
            "Todavía no existe un banco de memoria para este símbolo."
            " Se creará cuando ejecutes el entrenamiento."
        )

    retrieval = cfg.retrieval or RetrievalCfg()
    risk = cfg.risk or RiskCfg()

    with st.form("config_form"):
        st.markdown("### Conjunto de datos y modelos")
        col_symbol, col_news = st.columns([1, 1])
        symbol = col_symbol.text_input(
            "Símbolo a operar",
            value=cfg.symbol,
            help="Ticker bursátil que define qué series de precios y noticias se usarán para entrenar y evaluar el modelo.",
        )
        news_source = col_news.text_input(
            "Proveedor de noticias",
            value=cfg.news_source,
            help="Identificador del origen de titulares. Se registra en las cápsulas para entender el contexto de cada noticia.",
        )

        col_train_start, col_train_end = st.columns(2)
        train_start = col_train_start.date_input(
            "Inicio de entrenamiento",
            value=_parse_date(cfg.train_start, date(2022, 1, 1)),
            help="Primer día incluido para generar memoria histórica. Un tramo más largo aporta más experiencias al banco de memoria.",
        )
        train_end = col_train_end.date_input(
            "Fin de entrenamiento",
            value=_parse_date(cfg.train_end, date(2022, 12, 31)),
            help="Último día usado para construir recuerdos. Debe ser anterior a la fecha de inicio del backtest para evitar fuga de información.",
        )

        col_test_start, col_test_end = st.columns(2)
        test_start = col_test_start.date_input(
            "Inicio de backtest",
            value=_parse_date(cfg.test_start, date(2023, 1, 1)),
            help="Primer día evaluado en simulación. El modelo utilizará la memoria aprendida hasta esta fecha.",
        )
        test_end = col_test_end.date_input(
            "Fin de backtest",
            value=_parse_date(cfg.test_end, date(2023, 8, 31)),
            help="Último día del periodo de simulación. Fechas más largas permiten observar distintos regímenes de mercado.",
        )

        k_news = st.number_input(
            "Titulares por día (K)",
            min_value=1,
            max_value=50,
            value=int(cfg.K_news_per_day or 5),
            help="Cantidad de noticias más relevantes que se almacenan por jornada. Un valor mayor aumenta el contexto para el LLM, pero también el costo de procesamiento.",
        )

        embedding_model = st.text_input(
            "Modelo de embeddings",
            value=cfg.embedding_model,
            help="Modelo encargado de vectorizar los textos de memoria. Controla la calidad de las búsquedas semánticas en el banco de memoria.",
        )
        decision_model = st.text_input(
            "Modelo de decisiones",
            value=cfg.decision_model,
            help="LLM que sintetiza factores diarios y genera las señales de trading. Modelos más grandes suelen razonar mejor pero son más costosos.",
        )

        memory_path_input = st.text_input(
            "Archivo de memoria",
            value=cfg.memory_path,
            help="Ruta donde se guarda la memoria entrenada. Si la dejas vacía se utilizará una carpeta dedicada por símbolo para reutilizar sesiones anteriores.",
        )

        initial_cash_input = st.number_input(
            "Capital inicial",
            min_value=0.0,
            value=float(getattr(cfg, "initial_cash", 100000.0)),
            step=1000.0,
            help="Monto de efectivo disponible al inicio del backtest. Afecta el tamaño absoluto de las posiciones y el benchmark buy & hold.",
        )

        contrib_cols = st.columns(2)
        periodic_contribution_input = contrib_cols[0].number_input(
            "Aporte periódico",
            min_value=0.0,
            value=float(getattr(cfg, "periodic_contribution", 0.0)),
            step=100.0,
            help="Capital adicional que se ingresará automáticamente según la frecuencia elegida. Déjalo en 0 para desactivar los aportes.",
        )
        freq_options = {
            "Sin aportes": "none",
            "Mensual (primer día hábil)": "monthly",
        }
        freq_labels = list(freq_options.keys())
        freq_value_to_label = {v: k for k, v in freq_options.items()}
        current_freq_value = str(getattr(cfg, "contribution_frequency", "none") or "none").strip().lower()
        current_freq_label = freq_value_to_label.get(current_freq_value, "Sin aportes")
        try:
            freq_index = freq_labels.index(current_freq_label)
        except ValueError:
            freq_index = 0
        contribution_frequency_label = contrib_cols[1].selectbox(
            "Frecuencia de aportes",
            options=freq_labels,
            index=freq_index,
            help="Define cada cuánto se ejecuta el aporte programado. Escoge 'Sin aportes' o establece el monto en 0 para desactivarlos.",
        )
        contribution_frequency_value = freq_options[contribution_frequency_label]
        st.caption(
            "Los aportes se aplican al inicio del primer día de mercado de cada mes y quedan registrados en el historial de operaciones."
        )

        st.markdown("### Recuperación de memoria")
        k_cols = st.columns(3)
        k_shallow = k_cols[0].number_input(
            "k (capa superficial)",
            min_value=0,
            max_value=20,
            value=int(retrieval.k_shallow),
            help="Número de recuerdos recientes que se inyectan desde la capa superficial en cada prompt. Incrementarlo aporta más ejemplos inmediatos al modelo.",
        )
        k_intermediate = k_cols[1].number_input(
            "k (capa intermedia)",
            min_value=0,
            max_value=20,
            value=int(retrieval.k_intermediate),
            help="Cantidad de recuerdos con importancia media que se añaden al contexto. Útil para exponer patrones semanales o mensuales al LLM.",
        )
        k_deep = k_cols[2].number_input(
            "k (capa profunda)",
            min_value=0,
            max_value=20,
            value=int(retrieval.k_deep),
            help="Recuerdos estratégicos de largo plazo que se suman al prompt. Ayudan a conservar aprendizajes estructurales del modelo.",
        )

        q_cols = st.columns(3)
        Q_shallow = q_cols[0].number_input(
            "Q recencia superficial",
            min_value=1,
            max_value=90,
            value=int(retrieval.Q_shallow),
            help="Controla cuántos días tarda en desvanecerse la relevancia de un recuerdo superficial. Valores altos refuerzan experiencias antiguas.",
        )
        Q_intermediate = q_cols[1].number_input(
            "Q recencia intermedia",
            min_value=1,
            max_value=180,
            value=int(retrieval.Q_intermediate),
            help="Determina la velocidad de decaimiento para recuerdos de mediano plazo. Ajusta el equilibrio entre información fresca y patrones persistentes.",
        )
        Q_deep = q_cols[2].number_input(
            "Q recencia profunda",
            min_value=1,
            max_value=365,
            value=int(retrieval.Q_deep),
            help="Cantidad de días que se consideran aún relevantes en la capa profunda. Un valor alto conserva conocimientos de mercado muy antiguos.",
        )

        alpha_cols = st.columns(3)
        alpha_shallow = alpha_cols[0].number_input(
            "α importancia superficial",
            min_value=0.0,
            max_value=1.0,
            value=float(retrieval.alpha_shallow),
            step=0.05,
            help="Peso de la importancia manual frente a la similitud semántica en la capa superficial. Aumentarlo prioriza los recuerdos marcados como críticos.",
        )
        alpha_intermediate = alpha_cols[1].number_input(
            "α importancia intermedia",
            min_value=0.0,
            max_value=1.0,
            value=float(retrieval.alpha_intermediate),
            step=0.05,
            help="Peso relativo de la importancia en la capa intermedia. Permite resaltar experiencias que hayan rendido bien en el pasado.",
        )
        alpha_deep = alpha_cols[2].number_input(
            "α importancia profunda",
            min_value=0.0,
            max_value=1.0,
            value=float(retrieval.alpha_deep),
            step=0.05,
            help="Peso aplicado a la importancia en la capa profunda. Valores altos favorecen reglas de trading establecidas frente a nuevas observaciones.",
        )

        st.markdown("### Gestión de riesgo y ejecución")
        risk_cols1 = st.columns(3)
        risk_per_trade = risk_cols1[0].number_input(
            "Riesgo por operación",
            min_value=0.0,
            max_value=1.0,
            value=float(risk.risk_per_trade),
            step=0.01,
            help="Fracción del capital que el modelo intenta arriesgar en cada señal. Sirve como guía al diseñar stops y tamaños de posición.",
        )
        max_position = risk_cols1[1].number_input(
            "Máx. acciones",
            min_value=0,
            value=int(risk.max_position),
            help="Tope absoluto de acciones en cartera. Evita que el agente sobreapalancado exceda límites logísticos o regulatorios.",
        )
        allow_short = risk_cols1[2].checkbox(
            "Permitir cortos",
            value=bool(getattr(risk, "allow_short", False)),
            help="Activa la posibilidad de que el backtest abra posiciones short. Sin esta opción el agente solo podrá comprar o cerrar posiciones.",
        )

        risk_cols2 = st.columns(3)
        stop_loss_atr_mult = risk_cols2[0].number_input(
            "Stop loss · ATR",
            min_value=0.0,
            value=float(risk.stop_loss_atr_mult),
            step=0.5,
            help="Multiplicador del ATR usado para fijar stops de protección. Limita pérdidas cuando el precio se mueve en contra del escenario previsto.",
        )
        trailing_stop_atr_mult = risk_cols2[1].number_input(
            "Trailing stop · ATR",
            min_value=0.0,
            value=float(risk.trailing_stop_atr_mult),
            step=0.5,
            help="Multiplicador del ATR que sigue a la tendencia para asegurar ganancias. Un valor mayor da más holgura a operaciones con momentum.",
        )
        take_profit_atr_mult = risk_cols2[2].number_input(
            "Take profit · ATR",
            min_value=0.0,
            value=float(risk.take_profit_atr_mult),
            step=0.5,
            help="Multiplicador del ATR utilizado para cerrar posiciones ganadoras de forma discrecional. Cero desactiva el take profit automático.",
        )

        risk_cols3 = st.columns(3)
        commission_per_trade = risk_cols3[0].number_input(
            "Comisión fija",
            min_value=0.0,
            value=float(risk.commission_per_trade),
            step=0.1,
            help="Costo fijo aplicado a cada orden en el backtest. Refleja tarifas del broker por operación enviada.",
        )
        commission_per_share = risk_cols3[1].number_input(
            "Comisión por acción",
            min_value=0.0,
            value=float(risk.commission_per_share),
            step=0.001,
            help="Costo proporcional por acción ejecutada. Afecta el cálculo de cash después de cada trade.",
        )
        slippage_bps = risk_cols3[2].number_input(
            "Slippage (pbs)",
            min_value=0.0,
            value=float(risk.slippage_bps),
            step=0.5,
            help="Penalización en puntos básicos aplicada al precio de ejecución. Modela la fricción del mercado al llenar órdenes.",
        )

        risk_cols4 = st.columns(2)
        min_trade_value = risk_cols4[0].number_input(
            "Valor mínimo por trade",
            min_value=0.0,
            value=float(getattr(risk, "min_trade_value", 0.0)),
            step=100.0,
            help="Mínimo de capital que debe involucrar una operación. Útil para simular restricciones de ciertos brokers.",
        )
        min_trade_shares = risk_cols4[1].number_input(
            "Mínimo de acciones",
            min_value=0,
            value=int(getattr(risk, "min_trade_shares", 1)),
            help="Cantidad mínima de acciones por orden. Sirve para emular lotes predeterminados en mercados específicos.",
        )

        submitted = st.form_submit_button("Guardar configuración", type="primary")

    if not submitted:
        with st.expander("Vista previa del JSON actual", expanded=False):
            st.json(cfg.to_dict())
        return

    symbol_clean = _format_symbol(symbol)
    if not symbol_clean:
        symbol_clean = _format_symbol(cfg.symbol)
    memory_path_clean = memory_path_input.strip()
    if not memory_path_clean:
        memory_path_clean = default_memory_path(symbol_clean)

    updated_cfg = Config(
        symbol=symbol_clean or cfg.symbol,
        train_start=train_start.isoformat(),
        train_end=train_end.isoformat(),
        test_start=test_start.isoformat(),
        test_end=test_end.isoformat(),
        news_source=news_source.strip() or cfg.news_source,
        K_news_per_day=int(k_news),
        embedding_model=embedding_model.strip() or cfg.embedding_model,
        decision_model=decision_model.strip() or cfg.decision_model,
        memory_path=memory_path_clean,
        initial_cash=float(initial_cash_input),
        periodic_contribution=float(periodic_contribution_input),
        contribution_frequency=contribution_frequency_value,
        retrieval=RetrievalCfg(
            k_shallow=int(k_shallow),
            k_intermediate=int(k_intermediate),
            k_deep=int(k_deep),
            Q_shallow=int(Q_shallow),
            Q_intermediate=int(Q_intermediate),
            Q_deep=int(Q_deep),
            alpha_shallow=float(alpha_shallow),
            alpha_intermediate=float(alpha_intermediate),
            alpha_deep=float(alpha_deep),
        ),
        risk=RiskCfg(
            risk_per_trade=float(risk_per_trade),
            max_position=int(max_position),
            stop_loss_atr_mult=float(stop_loss_atr_mult),
            trailing_stop_atr_mult=float(trailing_stop_atr_mult),
            take_profit_atr_mult=float(take_profit_atr_mult),
            commission_per_trade=float(commission_per_trade),
            commission_per_share=float(commission_per_share),
            slippage_bps=float(slippage_bps),
            min_trade_value=float(min_trade_value),
            min_trade_shares=int(min_trade_shares),
            allow_short=bool(allow_short),
        ),
    )

    save_config(cfg_path, updated_cfg)
    st.session_state["CONFIG_LAST"] = updated_cfg.to_dict()
    st.success("Configuración guardada correctamente.")
    with st.expander("Vista previa del JSON actualizado", expanded=True):
        st.json(updated_cfg.to_dict())
