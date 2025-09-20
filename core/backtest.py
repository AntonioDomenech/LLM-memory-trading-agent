
import numpy as np, pandas as pd
from .logger import get_logger
from .config import load_config
from .data_fetcher import get_daily_bars
from .indicators import add_indicators
from .llm import chat_json
from .metrics import compute_metrics
from .plots import plot_equity, plot_drawdown
from .memory import MemoryBank
from .pipeline import prepare_daily_context

log = get_logger()

def _get(cfg, *path, default=None):
    obj = cfg
    for key in path:
        if not hasattr(obj, key):
            return default
        obj = getattr(obj, key)
    return obj

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def run_backtest(config_path="config.json", on_event=None, event_rate=10):
    def emit(evt):
        if on_event:
            try: on_event(evt)
            except Exception: pass

    cfg = load_config(config_path)

    retrieval_cfg = getattr(cfg, "retrieval", None)
    use_memory = False
    if retrieval_cfg is not None:
        for key in ("k_shallow", "k_intermediate", "k_deep"):
            if getattr(retrieval_cfg, key, 0) > 0:
                use_memory = True
                break
    memory_bank = None
    if use_memory:
        memory_path = getattr(cfg, "memory_path", None) or "data/memory_bank.json"
        memory_bank = MemoryBank(path=memory_path, emb_model=cfg.embedding_model)

    max_pos_shares = float(_get(cfg, "risk", "max_position", default=0) or 0)
    slippage_bps   = float(_get(cfg, "risk", "slippage_bps", default=0.0) or 0.0)
    c_per_trade    = float(_get(cfg, "risk", "commission_per_trade", default=0.0) or 0.0)
    c_per_share    = float(_get(cfg, "risk", "commission_per_share", default=0.0) or 0.0)
    allow_short    = bool(_get(cfg, "risk", "allow_short", default=False))

    emit({"type":"phase","label":"Load prices","state":"running"})
    df = add_indicators(get_daily_bars(cfg.symbol, cfg.test_start, cfg.test_end))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    dates = list(df["date"].astype("datetime64[ns]").dt.date)

    cash = 100000.0
    position = 0
    eq_series, bh_series, trades = [], [], []

    emit({"type":"phase","label":"Backtest loop","state":"running"})
    for i, d in enumerate(dates):
        d_iso = d.strftime("%Y-%m-%d")
        row = df.loc[df["date"] == d]
        if row.empty:
            emit({"type":"warn","message":f"No bar for {d_iso}, skipping"})
            continue
        pr = row.iloc[0].to_dict()
        price = float(pr["close"])

        # Buy&Hold benchmark
        if i == 0:
            bh_shares = int(cash // price); bh_cash = cash - bh_shares * price
        equity_bh = bh_shares * price + bh_cash
        bh_series.append((d_iso, equity_bh))

        ctx = prepare_daily_context(cfg, d_iso, pr, memory_bank=memory_bank)
        cap = ctx.capsule

        raw = chat_json(ctx.policy_prompt.as_messages(), model=cfg.decision_model, max_tokens=120)
        action = str(raw.get("action","HOLD")).strip().upper()

        # --- Exposure normalization ---
        target_exposure = raw.get("target_exposure", None)
        normalized = None
        if target_exposure is None:
            target_exposure = 0.0 if action == "SELL" else 1.0 if action == "BUY" else (position * price) / max(1e-9, (cash + position*price))
            normalized = f"defaulted:{target_exposure:.2f}"
        else:
            try:
                target_exposure = float(target_exposure)
            except Exception:
                target_exposure = 0.0
                normalized = "non_numeric->0.0"
            # Interpret 5..100 as percentages
            if target_exposure > 1.0 and target_exposure <= 100.0:
                target_exposure = target_exposure / 100.0
                normalized = f"percent_to_frac:{target_exposure:.2f}"
            # Clamp exposure
            lo = 0.0 if not allow_short else -1.0
            te_before = target_exposure
            target_exposure = _clamp(target_exposure, lo, 1.0)
            if te_before != target_exposure:
                normalized = f"clamped:{te_before:.3f}->{target_exposure:.3f}"

        # --- Translate exposure to shares with guardrails ---
        equity = cash + position * price
        desired_value = target_exposure * equity
        target_shares = int(desired_value // max(price, 1e-9))
        if max_pos_shares > 0:
            target_shares = int(_clamp(target_shares, 0 if not allow_short else -max_pos_shares, max_pos_shares))
        delta = target_shares - position

        # Fill price and commissions
        slip = (slippage_bps / 10000.0)
        fill = price * (1 + (slip if delta>0 else -slip))

        # No shorting by default
        if not allow_short and delta < 0:
            delta = max(delta, -position)

        # No margin: limit buys to affordable shares
        if delta > 0:
            # account for per-trade + per-share commission in affordability
            per_share_total = fill + c_per_share
            if per_share_total <= 0:
                affordable = 0
            else:
                affordable = int(max(0.0, (cash - c_per_trade)) // per_share_total)
            if affordable < delta:
                delta = affordable
                normalized = (normalized + ";afford_limited" if normalized else "afford_limited")

        # Execute
        if delta != 0:
            if delta > 0:
                cash -= delta * fill + c_per_trade + delta * c_per_share
            else:
                cash += (-delta) * fill - (c_per_trade + (-delta) * c_per_share)
            position += delta

            trades.append({"date": d_iso, "action": "BUY" if delta>0 else "SELL",
                           "shares_delta": int(delta), "fill_price": float(fill),
                           "note": normalized or ""})

        # Record decision periodically
        if i % max(1, event_rate) == 0:
            dec = {**raw, "action": action, "target_exposure": float(target_exposure),
                   "provider": ctx.provider_label, "norm": normalized or ""}
            emit({"type":"decision","date":d_iso,"decision": dec})
            emit({"type":"progress","i":i+1,"n":len(dates)})

        equity = cash + position * price
        eq_series.append((d_iso, equity))

    m = pd.DataFrame(eq_series, columns=["date","equity"]).set_index("date")["equity"]
    bh = pd.DataFrame(bh_series, columns=["date","equity"]).set_index("date")["equity"]
    m = m.clip(lower=1e-6)  # avoid divide-by-zero weirdness

    metrics = compute_metrics(m, bh)
    fig_eq = plot_equity(m, bh)
    fig_dd = plot_drawdown(m)
    trades_df = pd.DataFrame(trades).tail(25)
    return {"metrics": metrics, "fig_equity": fig_eq, "fig_drawdown": fig_dd, "trades_tail": trades_df}
