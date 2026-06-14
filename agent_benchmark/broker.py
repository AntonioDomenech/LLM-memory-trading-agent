from __future__ import annotations

from typing import Any, Dict, Tuple

from .schemas import BenchmarkConfig, PortfolioState


def _float(value: Any, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _round(value: float) -> float:
    return float(round(value, 8))


def execute_decision(
    portfolio: PortfolioState,
    decision: Dict[str, Any],
    market_close: float,
    config: BenchmarkConfig,
) -> Tuple[PortfolioState, Dict[str, Any]]:
    events = []
    model_failure = False
    action = str(decision.get("action", "HOLD")).upper().strip()
    if action not in {"BUY", "SELL", "HOLD"}:
        events.append({"type": "invalid_action", "value": decision.get("action")})
        action = "HOLD"
        model_failure = True

    price = float(market_close)
    if price <= 0:
        events.append({"type": "invalid_price", "value": price})
        return portfolio, {"events": events, "model_failure": True, "trade": None}

    equity_before = portfolio.cash + portfolio.position_shares * price
    target = _float(decision.get("target_exposure"), None)
    position_size = _float(decision.get("position_size_shares"), None)

    if target is None and position_size is None:
        if action == "HOLD":
            target = (portfolio.position_shares * price) / max(1e-9, equity_before)
        else:
            events.append({"type": "missing_sizing", "message": "No target_exposure or position_size_shares"})
            model_failure = True
            target = (portfolio.position_shares * price) / max(1e-9, equity_before)

    if position_size is not None and target is None:
        desired_shares = position_size
    else:
        target = float(target)
        if not config.allow_short and target < 0:
            events.append({"type": "short_blocked", "requested_exposure": target})
            target = 0.0
        if abs(target) > config.max_leverage:
            events.append({"type": "leverage_capped", "requested_exposure": target, "max_leverage": config.max_leverage})
            target = config.max_leverage if target > 0 else -config.max_leverage
        desired_shares = (target * equity_before) / price

    if not config.allow_short and desired_shares < 0:
        events.append({"type": "short_position_blocked", "requested_shares": desired_shares})
        desired_shares = 0.0

    raw_delta = desired_shares - portfolio.position_shares
    if action == "BUY" and raw_delta < 0:
        events.append({"type": "action_size_conflict", "action": action, "computed_delta": raw_delta})
    if action == "SELL" and raw_delta > 0:
        events.append({"type": "action_size_conflict", "action": action, "computed_delta": raw_delta})

    delta = raw_delta
    if not config.allow_short and portfolio.position_shares + delta < 0:
        delta = -portfolio.position_shares
        events.append({"type": "sell_limited_to_position"})

    slip = config.slippage_bps / 10000.0
    is_buy = delta > 0
    fill_price = price * (1 + slip if is_buy else 1 - slip)
    commission = config.commission_per_trade + abs(delta) * config.commission_per_share if abs(delta) > 1e-12 else 0.0

    if delta > 0:
        cost = delta * fill_price + commission
        if cost > portfolio.cash:
            affordable = max(0.0, (portfolio.cash - config.commission_per_trade) / max(fill_price + config.commission_per_share, 1e-9))
            events.append({"type": "cash_limited", "requested_shares": delta, "affordable_shares": affordable})
            delta = affordable
            commission = config.commission_per_trade + abs(delta) * config.commission_per_share if delta > 1e-12 else 0.0
            cost = delta * fill_price + commission
        cash_after = portfolio.cash - cost
    elif delta < 0:
        proceeds = abs(delta) * fill_price - commission
        cash_after = portfolio.cash + proceeds
    else:
        cash_after = portfolio.cash

    position_after = portfolio.position_shares + delta
    equity_after = cash_after + position_after * price
    new_portfolio = PortfolioState(cash=_round(cash_after), position_shares=_round(position_after), equity=_round(equity_after))

    trade = None
    if abs(delta) > 1e-12:
        trade = {
            "side": "BUY" if delta > 0 else "SELL",
            "shares": _round(abs(delta)),
            "signed_delta": _round(delta),
            "fill_price": _round(fill_price),
            "commission": _round(commission),
        }

    return new_portfolio, {
        "action": action,
        "price": price,
        "equity_before": _round(equity_before),
        "portfolio_before": portfolio.model_dump() if hasattr(portfolio, "model_dump") else portfolio.dict(),
        "portfolio_after": new_portfolio.model_dump() if hasattr(new_portfolio, "model_dump") else new_portfolio.dict(),
        "raw_delta_shares": _round(raw_delta),
        "trade": trade,
        "events": events,
        "model_failure": model_failure,
    }
