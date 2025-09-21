
import json
from typing import Callable, Optional

import pandas as pd
from .logger import get_logger
from .config import load_config, resolve_memory_path
from .data_fetcher import get_daily_bars
from .indicators import add_indicators
from .memory import MemoryBank
from .capsules import capsule_path
from .llm import chat_json
from .pipeline import prepare_daily_context

log = get_logger()


def _to_float(x, default=0.0):
    """Convert ``x`` to ``float`` while tolerating percents and blanks."""

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
    """Ensure numeric keys in the factor dictionary are floats."""

    if not isinstance(factor, dict):
        return {}
    keys = [
        "mood_score",
        "narrative_bias",
        "novelty",
        "credibility",
        "regime_alignment",
        "confidence",
    ]
    for k in keys:
        factor[k] = _to_float(factor.get(k, 0.0), 0.0)
    return factor


def run_training(
    config_path="config.json",
    on_event=None,
    should_stop: Optional[Callable[[], bool]] = None,
):
    """Train the memory bank by generating daily capsules from configuration."""

    def emit(evt):
        """Send structured events to ``on_event`` while swallowing handler errors."""

        if on_event:
            try:
                on_event(evt)
            except Exception:
                pass

    def stop_requested() -> bool:
        """Check if the caller asked to halt the training loop."""

        if not should_stop:
            return False
        try:
            return bool(should_stop())
        except Exception:
            return False

    cfg = load_config(config_path)

    emit({"type":"phase","label":"Load prices","state":"running"})
    df = add_indicators(get_daily_bars(cfg.symbol, cfg.train_start, cfg.train_end))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    dates = list(df["date"].astype("datetime64[ns]").dt.date)

    memory_path = resolve_memory_path(cfg, prefer_existing=True, ensure_parent=True)
    bank = MemoryBank(memory_path, emb_model=cfg.embedding_model)
    cap_path = capsule_path(cfg.symbol, cfg.train_start, cfg.train_end)

    emit({"type":"phase","label":"Training loop","state":"running"})
    emit({"type":"progress","i":0,"n":max(1,len(dates)-1)})

    risk_cfg = getattr(cfg, "risk", None)
    risk_snapshot = {}
    if risk_cfg is not None:
        risk_snapshot = {key: value for key, value in vars(risk_cfg).items()}
    risk_snapshot["allow_short"] = bool(getattr(risk_cfg, "allow_short", False))
    default_portfolio_state = {
        "cash": 0.0,
        "position": 0.0,
        "equity": 0.0,
        "max_position": float(getattr(risk_cfg, "max_position", 0.0) or 0.0),
        "slippage_bps": float(getattr(risk_cfg, "slippage_bps", 0.0) or 0.0),
        "commission_per_trade": float(getattr(risk_cfg, "commission_per_trade", 0.0) or 0.0),
        "commission_per_share": float(getattr(risk_cfg, "commission_per_share", 0.0) or 0.0),
        "risk": risk_snapshot,
    }

    stopped = False

    for i, d in enumerate(dates[:-1]):
        if stop_requested():
            stopped = True
            break
        d_iso = d.strftime("%Y-%m-%d")
        row = df.loc[df["date"] == d]
        if row.empty:
            emit({"type":"warn","message":f"No bar for {d_iso}, skipping"})
            continue
        pr = row.iloc[0].to_dict()

        ctx = prepare_daily_context(
            cfg,
            d_iso,
            pr,
            memory_bank=bank,
            portfolio_state=default_portfolio_state,
        )

        arts = list(ctx.articles)
        if not arts:
            emit({"type":"info","message":f"[{d_iso}] Sin noticias. Motivo: {ctx.news_reason}"})
            bank.add_item("shallow", f"{cfg.symbol} daily capsule {d_iso}", {"date": d_iso}, base_importance=5.0, seen_date=d_iso)
        else:
            emit({"type":"info","message":f"[{d_iso}] {len(arts)} titulares de {ctx.provider_label} (reason={ctx.news_reason})"})

        cap = ctx.capsule
        emit({"type":"capsule","date":d_iso,"capsule":cap})

        factor = chat_json(ctx.factor_prompt.as_messages(), model=cfg.decision_model, max_tokens=120)

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
    emit({"type": "done", "status": "stopped" if stopped else "completed"})
    return {
        "memory_snapshot": bank.snapshot(),
        "artifacts": {"capsules_path": cap_path},
        "stopped": stopped,
    }
