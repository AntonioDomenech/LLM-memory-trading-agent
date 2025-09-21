
import numpy as np, pandas as pd
from .logger import get_logger
from .config import load_config, resolve_memory_path
from .data_fetcher import get_daily_bars
from .indicators import add_indicators
from .llm import chat_json
from .metrics import compute_metrics
from .plots import plot_equity, plot_drawdown
from .memory import MemoryBank
from .pipeline import prepare_daily_context

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


def _get(cfg, *path, default=None):
    obj = cfg
    for key in path:
        if not hasattr(obj, key):
            return default
        obj = getattr(obj, key)
    return obj

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _append_norm_label(current, label):
    return f"{current};{label}" if current else label

def run_backtest(config_path="config.json", on_event=None, event_rate=10):
    def emit(evt):
        if on_event:
            try: on_event(evt)
            except Exception: pass

    cfg = load_config(config_path)
    memory_path = resolve_memory_path(cfg, prefer_existing=True, ensure_parent=True)

    retrieval_cfg = getattr(cfg, "retrieval", None)
    use_memory = False
    if retrieval_cfg is not None:
        for key in ("k_shallow", "k_intermediate", "k_deep"):
            if getattr(retrieval_cfg, key, 0) > 0:
                use_memory = True
                break
    memory_bank = None
    if use_memory:
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
    first_trading_days = {}
    for dt in dates:
        key = (dt.year, dt.month)
        if key not in first_trading_days:
            first_trading_days[key] = dt

    initial_cash = _to_float(getattr(cfg, "initial_cash", 100000.0), 100000.0)
    cash = initial_cash
    position = 0
    bh_shares = 0
    bh_cash = initial_cash
    periodic_contribution = max(0.0, _to_float(getattr(cfg, "periodic_contribution", 0.0), 0.0))
    freq_raw = str(getattr(cfg, "contribution_frequency", "none") or "none").strip().lower()
    if freq_raw in {"monthly", "mensual", "mes", "month", "m"}:
        contribution_frequency = "monthly"
    else:
        contribution_frequency = "none"
    eq_series, bh_series, trades = [], [], []
    contributions_series = []
    total_contributions = 0.0
    pending_feedback = None

    emit({"type":"phase","label":"Backtest loop","state":"running"})
    for i, d in enumerate(dates):
        d_iso = d.strftime("%Y-%m-%d")
        row = df.loc[df["date"] == d]
        if row.empty:
            emit({"type":"warn","message":f"No bar for {d_iso}, skipping"})
            continue
        pr = row.iloc[0].to_dict()
        price = float(pr["close"])

        contribution_today = 0.0
        if periodic_contribution > 0.0 and contribution_frequency == "monthly":
            if first_trading_days.get((d.year, d.month)) == d:
                cash += periodic_contribution
                bh_cash += periodic_contribution
                contribution_today = periodic_contribution
                total_contributions += contribution_today
                trades.append(
                    {
                        "date": d_iso,
                        "action": "CONTRIBUTION",
                        "shares_delta": 0,
                        "fill_price": None,
                        "amount": float(contribution_today),
                        "note": "Aporte mensual automático",
                    }
                )

        # --- Evaluate feedback from the previous decision (if any) ---
        if pending_feedback is not None:
            entry_price = pending_feedback.get("entry_price")
            prev_position = pending_feedback.get("position_after", 0)
            realized_pnl = 0.0
            realized_return = 0.0
            if entry_price not in (None, 0) and prev_position != 0:
                realized_pnl = (price - entry_price) * prev_position
                realized_return = realized_pnl / (abs(prev_position) * entry_price)
            elif entry_price not in (None, 0):
                realized_pnl = 0.0
                realized_return = 0.0

            if use_memory and memory_bank is not None:
                prev_date = pending_feedback.get("date", "")
                action_prev = pending_feedback.get("action", "")
                shares_after = int(pending_feedback.get("position_after", 0))
                ret_pct = realized_return * 100.0
                summary = (
                    f"{prev_date}: {action_prev} {abs(shares_after)}"
                    f" → {ret_pct:+.2f}% next day"
                )
                importance = max(
                    1.0,
                    10.0 + 50.0 * realized_return + 20.0 * abs(realized_return),
                )
                feedback_meta = {
                    "date": f"{prev_date}-feedback",
                    "decision_date": prev_date,
                    "observed_on": d_iso,
                    "action": action_prev,
                    "target_exposure": float(
                        pending_feedback.get("target_exposure", 0.0)
                    ),
                    "position": shares_after,
                    "entry_price": float(entry_price) if entry_price is not None else None,
                    "realized_return": float(realized_return),
                    "realized_pnl": float(realized_pnl),
                    "shares_delta": int(pending_feedback.get("shares_delta", 0)),
                }
                memory_bank.add_item(
                    "shallow",
                    summary,
                    feedback_meta,
                    base_importance=importance,
                    seen_date=d_iso,
                )

            pending_feedback = None

        # Buy&Hold benchmark
        if i == 0:
            bh_shares = int(bh_cash // price)
            bh_cash = bh_cash - bh_shares * price
        equity_bh = bh_shares * price + bh_cash
        bh_series.append((d_iso, equity_bh))

        equity = cash + position * price
        risk_cfg = getattr(cfg, "risk", None)
        risk_snapshot = {}
        if risk_cfg is not None:
            risk_snapshot = {key: value for key, value in vars(risk_cfg).items()}
        risk_snapshot["allow_short"] = bool(allow_short)
        portfolio_state = {
            "cash": float(cash),
            "position": int(position),
            "equity": float(equity),
            "max_position": float(max_pos_shares),
            "slippage_bps": float(slippage_bps),
            "commission_per_trade": float(c_per_trade),
            "commission_per_share": float(c_per_share),
            "risk": risk_snapshot,
        }

        ctx = prepare_daily_context(
            cfg,
            d_iso,
            pr,
            memory_bank=memory_bank,
            portfolio_state=portfolio_state,
        )

        if use_memory and memory_bank is not None:
            factor_raw = chat_json(
                ctx.factor_prompt.as_messages(),
                model=cfg.decision_model,
                max_tokens=120,
            )
            factor = _coerce_factor_numbers(factor_raw)
            summary = (
                f"[{d_iso}] "
                f"mood={factor.get('mood_score', 0.0):.2f} "
                f"bias={factor.get('narrative_bias', 0.0):+.2f} "
                f"nov={factor.get('novelty', 0.0):.2f} "
                f"cred={factor.get('credibility', 0.0):.2f}"
            )
            memory_bank.add_item(
                "shallow",
                summary,
                {"date": d_iso, "factor": factor},
                base_importance=10.0,
                seen_date=d_iso,
            )

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
            raw_target = target_exposure
            target_exposure = _to_float(raw_target, default=0.0)
            parsed_from_percent = False
            parse_failed = False

            if isinstance(raw_target, str):
                stripped = raw_target.strip()
                if stripped.endswith("%"):
                    try:
                        float(stripped[:-1])
                    except Exception:
                        parse_failed = True
                    else:
                        parsed_from_percent = True
                        normalized = _append_norm_label(
                            normalized, f"percent_to_frac:{target_exposure:.2f}"
                        )
                elif stripped:
                    try:
                        float(stripped)
                    except Exception:
                        parse_failed = True
                else:
                    parse_failed = True
            elif not isinstance(raw_target, (int, float, np.number)):
                parse_failed = True

            if parse_failed:
                target_exposure = 0.0
                normalized = _append_norm_label(normalized, "non_numeric->0.0")

            if (
                not parsed_from_percent
                and target_exposure > 1.0
                and target_exposure <= 100.0
            ):
                target_exposure = target_exposure / 100.0
                normalized = _append_norm_label(
                    normalized, f"percent_to_frac:{target_exposure:.2f}"
                )

            lo = 0.0 if not allow_short else -1.0
            te_before = target_exposure
            target_exposure = _clamp(target_exposure, lo, 1.0)
            if te_before != target_exposure:
                normalized = _append_norm_label(
                    normalized, f"clamped:{te_before:.3f}->{target_exposure:.3f}"
                )

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
                normalized = _append_norm_label(normalized, "afford_limited")

        # Execute
        if delta != 0:
            if delta > 0:
                cash -= delta * fill + c_per_trade + delta * c_per_share
            else:
                cash += (-delta) * fill - (c_per_trade + (-delta) * c_per_share)
            position += delta

            trades.append({"date": d_iso, "action": "BUY" if delta>0 else "SELL",
                           "shares_delta": int(delta), "fill_price": float(fill),
                           "amount": None,
                           "note": normalized or ""})

        # Record decision periodically
        if i % max(1, event_rate) == 0:
            dec = {**raw, "action": action, "target_exposure": float(target_exposure),
                   "provider": ctx.provider_label, "norm": normalized or ""}
            emit({"type":"decision","date":d_iso,"decision": dec})
            emit({"type":"progress","i":i+1,"n":len(dates)})

        equity = cash + position * price
        eq_series.append((d_iso, equity))
        contributions_series.append((d_iso, contribution_today))

        pending_feedback = {
            "date": d_iso,
            "action": action,
            "target_exposure": float(target_exposure),
            "position_after": int(position),
            "entry_price": float(price),
            "shares_delta": int(delta),
        }

    m = pd.DataFrame(eq_series, columns=["date", "equity"]).set_index("date")["equity"]
    bh = pd.DataFrame(bh_series, columns=["date", "equity"]).set_index("date")["equity"]
    contrib = (
        pd.DataFrame(contributions_series, columns=["date", "contribution"])
        .set_index("date")["contribution"]
        .astype(float)
    )
    m = m.clip(lower=1e-6)  # avoid divide-by-zero weirdness

    metrics = compute_metrics(m, bh, contributions=contrib)
    fig_eq = plot_equity(m, bh)
    fig_dd = plot_drawdown(m)

    if eq_series:
        equity_curve = (
            pd.DataFrame(eq_series, columns=["date", "equity"])
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .set_index("date")
            .rename(columns={"equity": "strategy_equity"})
        )
        bh_curve = (
            pd.DataFrame(bh_series, columns=["date", "equity"])
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .set_index("date")
            .rename(columns={"equity": "benchmark_equity"})
        )
        equity_curve = equity_curve.join(bh_curve, how="left")
        drawdown_curve = (
            (equity_curve["strategy_equity"] / equity_curve["strategy_equity"].cummax() - 1.0)
            .to_frame(name="drawdown")
        )
    else:
        equity_curve = pd.DataFrame(columns=["strategy_equity", "benchmark_equity"])
        drawdown_curve = pd.DataFrame(columns=["drawdown"])

    trades_df = pd.DataFrame(trades).tail(25)
    return {
        "metrics": metrics,
        "fig_equity": fig_eq,
        "fig_drawdown": fig_dd,
        "trades_tail": trades_df,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "contributions": {
            "frequency": contribution_frequency,
            "amount": float(periodic_contribution),
            "total_added": float(total_contributions),
        },
    }
