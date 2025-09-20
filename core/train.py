
from datetime import datetime, timedelta
import os, json, pandas as pd
from .logger import get_logger
from .config import load_config
from .data_fetcher import get_daily_bars
from .indicators import add_indicators
from .news_fetcher import fetch_news_with_reason
from .memory import MemoryBank
from .capsules import build_capsule, capsule_path
from .llm import chat_json

log = get_logger()

def _to_float(x, default=0.0):
    try:
        if isinstance(x, str):
            xs = x.strip()
            if xs.endswith("%"):
                return float(xs[:-1]) / 100.0
            return float(xs)
        return float(x)
    except Exception:
        return float(default)

def _coerce_factor_numbers(factor: dict) -> dict:
    if not isinstance(factor, dict):
        return {}
    keys = ["mood_score","narrative_bias","novelty","credibility","regime_alignment","confidence"]
    for k in keys:
        factor[k] = _to_float(factor.get(k, 0.0), 0.0)
    return factor

def _prov_label(reason: str) -> str:
    parts = (reason or "").split(":")
    if not parts:
        return "unknown"
    if parts[0] == "cache" and len(parts) > 1:
        return f"cache->{parts[1]}"
    return parts[0]

def run_training(config_path="config.json", on_event=None):
    def emit(evt):
        if on_event:
            try: on_event(evt)
            except Exception: pass

    cfg = load_config(config_path)

    emit({"type":"phase","label":"Load prices","state":"running"})
    df = add_indicators(get_daily_bars(cfg.symbol, cfg.train_start, cfg.train_end))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    dates = list(df["date"].astype("datetime64[ns]").dt.date)

    bank = MemoryBank("data/memory_bank.json", emb_model=cfg.embedding_model)
    cap_path = capsule_path(cfg.symbol, cfg.train_start, cfg.train_end)

    emit({"type":"phase","label":"Training loop","state":"running"})
    emit({"type":"progress","i":0,"n":max(1,len(dates)-1)})

    for i, d in enumerate(dates[:-1]):
        d_iso = d.strftime("%Y-%m-%d")
        row = df.loc[df["date"] == d]
        if row.empty:
            emit({"type":"warn","message":f"No bar for {d_iso}, skipping"})
            continue
        pr = row.iloc[0].to_dict()

        arts, reason = ([], "K_news_per_day=0")
        provider_used = "Off"
        if cfg.K_news_per_day > 0 and cfg.news_source != "Off":
            arts, reason = fetch_news_with_reason(cfg.symbol, d_iso, cfg.K_news_per_day)
            provider_used = _prov_label(reason)

        if not arts:
            emit({"type":"info","message":f"[{d_iso}] Sin noticias. Motivo: {reason}"})
            bank.add_item("shallow", f"{cfg.symbol} daily capsule {d_iso}", {"date": d_iso}, base_importance=5.0, seen_date=d_iso)
        else:
            emit({"type":"info","message":f"[{d_iso}] {len(arts)} titulares de {provider_used} (reason={reason})"})

        regime = {
            "market": "up" if pr.get("trend_up",0)==1 else "down_or_sideways",
            "vol_bucket": "high" if float(pr.get("atr",0))/max(1.0,float(pr.get("price",1.0))) > 0.02 else "low"
        }

        cap = build_capsule(d_iso, cfg.symbol, pr, arts, [], regime)
        cap["headlines_source"] = provider_used  # embed provider in capsule for UI debugging
        emit({"type":"capsule","date":d_iso,"capsule":cap})

        sys = {"role":"system","content": open("prompts/factor_head.txt","r",encoding="utf-8").read()}
        usr = {"role":"user","content": json.dumps(cap, sort_keys=True)}
        factor = chat_json([sys, usr], model=cfg.decision_model, max_tokens=120)

        factor = _coerce_factor_numbers(factor)
        emit({"type":"factor","date":d_iso,"factor":factor})

        summary = (
            f"[{d_iso}] "
            f"mood={factor.get('mood_score',0.0):.2f} "
            f"bias={factor.get('narrative_bias',0.0):+.2f} "
            f"nov={factor.get('novelty',0.0):.2f} "
            f"cred={factor.get('credibility',0.0):.2f}"
        )
        bank.add_item("shallow", summary, {"date": d_iso, "factor": factor}, base_importance=10.0, seen_date=d_iso)

        if (i+1) % 7 == 0:
            bank.promote()

        with open(cap_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"date": d_iso, "capsule": cap, "factor": factor}) + "\\n")

        if i % 5 == 0:
            emit({"type":"progress","i":i,"n":max(1,len(dates)-1)})

    bank.save()
    emit({"type":"done"})
    return {"memory_snapshot": bank.snapshot(), "artifacts": {"capsules_path": cap_path}}
