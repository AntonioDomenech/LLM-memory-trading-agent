import os
from datetime import date

import pandas as pd
import streamlit as st

from core.news_store import download_range, load_local_day, prescan_days


def _format_coverage_df(summary: list[dict], required: int) -> pd.DataFrame:
    df = pd.DataFrame(summary)
    if "available" in df.columns:
        df["status"] = df["available"].apply(
            lambda val: "✅ Completo" if int(val or 0) >= int(required) else "⚠️ Incompleto"
        )
    else:
        df["status"] = ""
    ordered_cols = [
        col
        for col in ["date", "available", "status", "exists", "provider", "decision"]
        if col in df.columns
    ]
    return df[ordered_cols]


def render_news_tab():
    st.header("📰 Gestor de noticias")
    st.markdown(
        "Centraliza las descargas de titulares y controla qué días del histórico ya cuentan con información local."
    )

    with st.container():
        st.subheader("Configuración de la caché")
        c_dir, c_actions = st.columns([3, 1])
        with c_dir:
            base_dir = st.text_input(
                "Carpeta local",
                value=os.environ.get("NEWS_LOCAL_DIR", "data/news_local"),
                help="Ruta donde se almacenarán los JSON de noticias descargados.",
            )
        with c_actions:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("Usar carpeta", use_container_width=True):
                os.environ["NEWS_LOCAL_DIR"] = base_dir
                st.session_state["NEWS_LOCAL_DIR"] = base_dir
                st.success(f"Carpeta activa: {base_dir}")

        c1, c2, c3 = st.columns(3)
        with c1:
            symbol = st.text_input("Símbolo", value="AAPL").upper().strip()
        with c2:
            start_d = st.date_input("Inicio", value=date(2022, 1, 1))
        with c3:
            end_d = st.date_input("Fin", value=date(2022, 3, 31))

        col_opts = st.columns(3)
        K = col_opts[0].number_input(
            "Titulares por día (K)", min_value=1, max_value=50, value=10, step=1
        )
        full = col_opts[1].checkbox(
            "Guardar contenido completo",
            value=True,
            help="Descarga también el texto legible de cada artículo cuando esté disponible.",
        )
        skip_existing = col_opts[2].checkbox(
            "Omitir días existentes",
            value=True,
            help="Permite reanudar descargas evitando volver a solicitar días completos.",
        )

    st.divider()

    st.subheader("Descarga de titulares")
    st.markdown(
        "Pulsa el botón para poblar o actualizar la caché en el rango seleccionado. Se mostrará el avance en tiempo real."
    )

    if st.button("Descargar y guardar noticias", type="primary", use_container_width=True):
        start_iso, end_iso = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
        total_days = max(1, (end_d - start_d).days + 1)

        progress = st.progress(0.0)
        status = st.empty()

        cnt = {"i": 0}
        plan_holder = st.expander("Planificación previa", expanded=False)

        def on_evt(evt: dict):
            if evt.get("type") == "plan":
                plan = evt.get("plan", [])
                if plan:
                    plan_df = pd.DataFrame(plan)
                    plan_holder.dataframe(plan_df, use_container_width=True)
                return
            if evt.get("type") != "progress":
                return
            cnt["i"] += 1
            prov = evt.get("provider") or "unknown"
            day = evt.get("date", "")
            saved_path = evt.get("saved_path", "")
            ok = evt.get("content_ok", 0)
            fail = evt.get("content_fail", 0)
            days_ready = evt.get("days_with_news", 0)
            skipped = evt.get("skipped_existing", 0)

            progress.progress(min(1.0, cnt["i"] / total_days))
            status.info(
                f"{day}: **{prov}** → `{saved_path}` · contenido ok={ok} fallos={fail} · días listos={days_ready} · omitidos={skipped}"
            )

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
            f"{stats.get('saved', 0)}/{total_days} días guardados · "
            f"último={stats.get('last_date', '')} · "
            f"con contenido={stats.get('content_ok', 0)} fallos={stats.get('content_fail', 0)} · "
            f"omitidos={stats.get('skipped_existing', 0)}"
        )

        summary = prescan_days(
            symbol,
            start_iso,
            end_iso,
            K=int(K),
            base_dir=base_dir,
            full_content=bool(full),
        )
        if summary:
            df = _format_coverage_df(summary, int(K))
            st.markdown("#### Cobertura local")
            st.dataframe(df, use_container_width=True, hide_index=True)
            missing_days = [row["date"] for row in summary if int(row.get("available", 0)) < int(K)]
            cols = st.columns(3)
            cols[0].metric("Días con noticias", stats.get("days_with_news", 0))
            cols[1].metric("Días pendientes", len(missing_days))
            cols[2].metric("Artículos por día objetivo", int(K))
            if missing_days:
                st.warning("Faltan titulares en: " + ", ".join(missing_days))
            else:
                st.info("Todos los días del rango tienen suficientes titulares guardados.")

        st.session_state["NEWS_LOCAL_DIR"] = base_dir
        os.environ["NEWS_LOCAL_DIR"] = base_dir

    st.divider()
    st.subheader("Verificación rápida")
    st.caption("Abre un día específico para revisar los artículos almacenados.")
    test_day = st.text_input("Día (YYYY-MM-DD)", value="2022-01-03")
    if st.button("Abrir día guardado", use_container_width=True):
        arts, reason = load_local_day(symbol, test_day, base_dir)
        if arts:
            st.json(
                {
                    "symbol": symbol,
                    "date": test_day,
                    "provider_or_origin": reason,
                    "count": len(arts),
                    "sample": arts[:1],
                }
            )
        else:
            st.warning(f"No hay archivos locales para {symbol} {test_day} en {base_dir}")
