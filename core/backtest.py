
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
    """Parse numeric inputs that may include percentage strings."""

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
    """Ensure factor dictionary fields are floats for downstream math."""

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
    """Traverse ``cfg`` following ``path`` attributes with a default."""

    obj = cfg
    for key in path:
        if not hasattr(obj, key):
            return default
        obj = getattr(obj, key)
    return obj

def _clamp(x, lo, hi):
    """Clamp ``x`` between ``lo`` and ``hi``."""

    return max(lo, min(hi, x))


def _append_norm_label(current, label):
    """Append a normalisation label keeping semi-colon separation."""

    return f"{current};{label}" if current else label


def _round_quantity(value, precision=8):
    """Return ``value`` rounded for stable reporting."""

    try:
        return float(round(float(value), precision))
    except Exception:
        return 0.0


def _format_quantity(value, precision=6):
    """Format share quantities removing trailing zeros."""

    try:
        rounded = round(float(value), precision)
    except Exception:
        return "0"
    formatted = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
    if not formatted or formatted == "-0":
        return "0"
    return formatted


def run_backtest(config_path="config.json", on_event=None, event_rate=10):
    """Simulate the trading agent using the stored configuration and memory."""

    def emit(evt):
        """Forward events to the optional callback without raising."""

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
    min_trade_value = max(0.0, float(_get(cfg, "risk", "min_trade_value", default=0.0) or 0.0))
    min_trade_shares = max(0.0, float(_get(cfg, "risk", "min_trade_shares", default=0.0) or 0.0))

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
    position = 0.0
    bh_shares = 0.0
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
    peak_equity = 0.0
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
                if price > 0:
                    shares_added = periodic_contribution / price
                    bh_shares += shares_added
                    bh_cash -= shares_added * price
                contribution_today = periodic_contribution
                total_contributions += contribution_today
                trades.append(
                    {
                        "date": d_iso,
                        "action": "CONTRIBUTION",
                        "shares_delta": 0.0,
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
                shares_after = float(pending_feedback.get("position_after", 0.0) or 0.0)
                shares_after_fmt = _format_quantity(abs(shares_after))
                ret_pct = realized_return * 100.0
                summary = (
                    f"{prev_date}: {action_prev} {shares_after_fmt}"
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
                    "position": _round_quantity(shares_after),
                    "entry_price": float(entry_price) if entry_price is not None else None,
                    "realized_return": float(realized_return),
                    "realized_pnl": float(realized_pnl),
                    "shares_delta": _round_quantity(
                        pending_feedback.get("shares_delta", 0.0)
                    ),
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
        if i == 0 and price > 0:
            initial_shares = int(bh_cash // price)
            if initial_shares > 0:
                bh_shares += float(initial_shares)
                bh_cash -= initial_shares * price
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
            "position": _round_quantity(position),
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
        if price <= 0.0:
            target_shares = 0.0
        else:
            target_shares = desired_value / price
        if max_pos_shares > 0:
            upper_bound = float(max_pos_shares)
            lower_bound = -upper_bound if allow_short else 0.0
            target_shares = _clamp(target_shares, lower_bound, upper_bound)
        delta = target_shares - position

        is_buy = delta > 0
        floor_applied = False
        floor_blocked = False
        share_floor_applied = False
        share_floor_blocked = False
        min_notional_shares = 0.0
        planned_delta = delta
        remaining_capacity = None
        if max_pos_shares > 0:
            remaining_capacity = max(0.0, float(max_pos_shares) - position)
        if is_buy:
            if min_trade_value > 0.0 and price > 0.0:
                min_notional_shares = min_trade_value / price
            if min_notional_shares > 0.0:
                desired_delta = planned_delta
                if remaining_capacity is not None and remaining_capacity < min_notional_shares:
                    planned_delta = 0.0
                else:
                    if planned_delta < min_notional_shares:
                        if min_notional_shares <= desired_delta + 1e-9:
                            planned_delta = min_notional_shares
                            floor_applied = True
                        else:
                            planned_delta = 0.0
                            floor_blocked = True
                    if planned_delta > 0 and remaining_capacity is not None:
                        planned_delta = min(planned_delta, remaining_capacity)

        planned_delta = float(planned_delta)

        # Fill price and commissions
        slip = (slippage_bps / 10000.0)
        fill = price * (1 + (slip if is_buy else -slip))

        # No shorting by default
        final_delta = planned_delta
        if not allow_short and final_delta < 0:
            final_delta = max(final_delta, -position)

        # No margin: limit buys to affordable shares
        affordability_limited = False
        if is_buy:
            if final_delta <= 0:
                affordable = 0.0
            else:
                per_share_total = fill + c_per_share
                available_cash = cash - c_per_trade
                if per_share_total <= 0 or available_cash <= 0:
                    affordable = 0.0
                else:
                    affordable = max(0.0, available_cash / per_share_total)
                if min_notional_shares > 0.0 and affordable < min_notional_shares:
                    affordable = 0.0
            if final_delta > 0:
                if affordable <= 0.0:
                    final_delta = 0.0
                    affordability_limited = True
                else:
                    if affordable < final_delta:
                        final_delta = affordable
                        affordability_limited = True
            else:
                final_delta = 0.0

            if floor_applied and final_delta > 0:
                normalized = _append_norm_label(
                    normalized,
                    f"min_trade_floor:{_format_quantity(max(min_notional_shares, 0.0))}",
                )
            if floor_blocked:
                label = "min_trade_floor_cancel"
                if min_notional_shares > 0.0:
                    label = (
                        f"{label}:{_format_quantity(max(min_notional_shares, 0.0))}"
                    )
                normalized = _append_norm_label(normalized, label)
        else:
            affordable = None

        if is_buy and final_delta > 0 and affordability_limited:
            normalized = _append_norm_label(normalized, "afford_limited")

        if min_trade_shares > 0.0:
            abs_final = abs(final_delta)
            if abs_final > 0.0 and abs_final + 1e-12 < min_trade_shares:
                sign = 1.0 if final_delta > 0 else -1.0
                candidate = sign * min_trade_shares
                can_round = True

                if sign > 0:
                    if remaining_capacity is not None and remaining_capacity + 1e-12 < min_trade_shares:
                        can_round = False
                    if affordable is not None and affordable + 1e-12 < min_trade_shares:
                        can_round = False
                else:
                    new_position = position + candidate
                    if not allow_short:
                        if position < min_trade_shares - 1e-12:
                            can_round = False
                        elif new_position < -1e-12:
                            can_round = False
                    if allow_short and max_pos_shares > 0:
                        lower_bound = -float(max_pos_shares)
                        if new_position < lower_bound - 1e-12:
                            can_round = False

                if can_round:
                    final_delta = candidate
                    share_floor_applied = True
                else:
                    final_delta = 0.0
                    share_floor_blocked = True

        if share_floor_applied:
            normalized = _append_norm_label(
                normalized,
                f"min_share_floor:{_format_quantity(max(min_trade_shares, 0.0))}",
            )
        if share_floor_blocked:
            label = "min_share_floor_cancel"
            if min_trade_shares > 0.0:
                label = f"{label}:{_format_quantity(max(min_trade_shares, 0.0))}"
            normalized = _append_norm_label(normalized, label)

        delta = _round_quantity(final_delta)

        # Execute
        if delta != 0:
            abs_delta = abs(delta)
            if delta > 0:
                cash -= delta * fill + c_per_trade + abs_delta * c_per_share
            else:
                cash += abs_delta * fill - (c_per_trade + abs_delta * c_per_share)
            position += delta
            position = _round_quantity(position)

            trades.append({"date": d_iso, "action": "BUY" if delta>0 else "SELL",
                           "shares_delta": _round_quantity(delta), "fill_price": float(fill),
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

        peak_equity = max(peak_equity, equity)
        drawdown = 0.0 if peak_equity <= 0 else (equity / peak_equity) - 1.0

        emit(
            {
                "type": "equity_point",
                "date": d_iso,
                "equity": float(equity),
                "benchmark": float(equity_bh),
                "cash": float(cash),
                "position": _round_quantity(position),
                "drawdown": float(drawdown),
                "contribution": float(contribution_today),
                "total_contributions": float(total_contributions),
            }
        )

        pending_feedback = {
            "date": d_iso,
            "action": action,
            "target_exposure": float(target_exposure),
            "position_after": _round_quantity(position),
            "entry_price": float(price),
            "shares_delta": _round_quantity(delta),
        }

    m = pd.DataFrame(eq_series, columns=["date","equity"]).set_index("date")["equity"]
    bh = pd.DataFrame(bh_series, columns=["date","equity"]).set_index("date")["equity"]
    contrib = pd.DataFrame(contributions_series, columns=["date", "contribution"]).set_index("date")["contribution"].astype(float)
    m = m.clip(lower=1e-6)  # avoid divide-by-zero weirdness

    metrics = compute_metrics(m, bh, contributions=contrib)
    fig_eq = plot_equity(m, bh)
    fig_dd = plot_drawdown(m)
    trades_df = pd.DataFrame(trades).tail(25)
    emit({"type": "done"})
    return {
        "metrics": metrics,
        "fig_equity": fig_eq,
        "fig_drawdown": fig_dd,
        "trades_tail": trades_df,
        "contributions": {
            "frequency": contribution_frequency,
            "amount": float(periodic_contribution),
            "total_added": float(total_contributions),
        },
    }
