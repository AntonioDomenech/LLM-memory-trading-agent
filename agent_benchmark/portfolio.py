from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .schemas import BenchmarkConfig, PortfolioBook


def _round(value: float) -> float:
    return float(round(value, 8))


def _book_from_positions(cash: float, positions: Dict[str, float], prices: Dict[str, float]) -> PortfolioBook:
    long_value = 0.0
    short_value = 0.0
    market_value = 0.0
    for symbol, shares in positions.items():
        price = float(prices.get(symbol, 0.0) or 0.0)
        value = shares * price
        market_value += value
        if value >= 0:
            long_value += value
        else:
            short_value += abs(value)
    equity = cash + market_value
    if equity <= 0:
        gross = 0.0
        net = 0.0
    else:
        gross = (long_value + short_value) / equity
        net = market_value / equity
    return PortfolioBook(
        cash=_round(cash),
        positions={symbol: _round(shares) for symbol, shares in positions.items() if abs(shares) > 1e-10},
        equity=_round(equity),
        long_exposure=_round(long_value / equity) if equity > 0 else 0.0,
        short_exposure=_round(short_value / equity) if equity > 0 else 0.0,
        gross_exposure=_round(gross),
        net_exposure=_round(net),
    )


def initial_book(initial_cash: float) -> PortfolioBook:
    return PortfolioBook(cash=float(initial_cash), positions={}, equity=float(initial_cash))


def mark_to_market(book: PortfolioBook, prices: Dict[str, float]) -> PortfolioBook:
    return _book_from_positions(book.cash, dict(book.positions), prices)


def estimate_target_turnover(book: PortfolioBook, target_weights: Dict[str, Any], prices: Dict[str, float]) -> float:
    current = mark_to_market(book, prices)
    equity = max(float(current.equity), 1e-9)
    current_weights = {
        symbol: float(shares) * float(prices.get(symbol, 0.0) or 0.0) / equity
        for symbol, shares in current.positions.items()
        if symbol in prices and float(prices.get(symbol, 0.0) or 0.0) > 0
    }
    symbols = set(current_weights)
    clean_targets: Dict[str, float] = {}
    for symbol, value in (target_weights or {}).items():
        try:
            weight = float(value)
        except Exception:
            continue
        symbol = str(symbol).upper().strip()
        clean_targets[symbol] = weight
        symbols.add(symbol)
    return _round(sum(abs(clean_targets.get(symbol, 0.0) - current_weights.get(symbol, 0.0)) for symbol in symbols))


def reject_target_weights(
    book: PortfolioBook,
    target_weights: Dict[str, Any],
    prices: Dict[str, float],
    validation_errors: List[Dict[str, Any]],
) -> Tuple[PortfolioBook, Dict[str, Any]]:
    current = mark_to_market(book, prices)
    clean_weights: Dict[str, float] = {}
    for symbol, raw_weight in (target_weights or {}).items():
        try:
            clean_weights[str(symbol).upper().strip()] = _round(float(raw_weight))
        except Exception:
            clean_weights[str(symbol).upper().strip()] = 0.0
    event = {
        "type": "allocation_validation_rejected",
        "action": "no_trades_keep_previous_portfolio",
        "validation_errors": validation_errors,
    }
    return current, {
        "portfolio_before": current.model_dump() if hasattr(current, "model_dump") else current.dict(),
        "portfolio_after": current.model_dump() if hasattr(current, "model_dump") else current.dict(),
        "target_weights": clean_weights,
        "executed_target_weights": {},
        "trades": [],
        "events": [event],
        "fees": 0.0,
        "slippage_cost": 0.0,
        "model_failure": True,
        "validation_errors": validation_errors,
    }


def execute_target_weights(
    book: PortfolioBook,
    target_weights: Dict[str, Any],
    prices: Dict[str, float],
    config: BenchmarkConfig,
) -> Tuple[PortfolioBook, Dict[str, Any]]:
    events = []
    trades = []
    clean_weights: Dict[str, float] = {}

    for symbol, raw_weight in (target_weights or {}).items():
        symbol = str(symbol).upper().strip()
        if symbol not in prices or float(prices.get(symbol) or 0.0) <= 0:
            events.append({"type": "invalid_symbol_or_price", "symbol": symbol})
            continue
        try:
            weight = float(raw_weight)
        except Exception:
            events.append({"type": "invalid_weight", "symbol": symbol, "value": raw_weight})
            continue
        if not config.allow_short and weight < 0:
            events.append({"type": "short_blocked", "symbol": symbol, "requested_weight": weight})
            weight = 0.0
        clean_weights[symbol] = weight

    current = mark_to_market(book, prices)
    gross = sum(abs(value) for value in clean_weights.values())
    if gross > config.max_gross_exposure + 1e-9:
        events.append(
            {
                "type": "invalid_gross_exposure",
                "requested": _round(gross),
                "max_gross_exposure": config.max_gross_exposure,
                "action": "allocation_rejected",
            }
        )
        return current, {
            "portfolio_before": current.model_dump() if hasattr(current, "model_dump") else current.dict(),
            "portfolio_after": current.model_dump() if hasattr(current, "model_dump") else current.dict(),
            "target_weights": {symbol: _round(value) for symbol, value in clean_weights.items()},
            "executed_target_weights": {},
            "trades": [],
            "events": events,
            "fees": 0.0,
            "slippage_cost": 0.0,
            "model_failure": True,
        }

    equity_before = max(float(current.equity), 1e-9)
    desired_positions: Dict[str, float] = {}
    for symbol, weight in clean_weights.items():
        desired_positions[symbol] = (weight * equity_before) / float(prices[symbol])

    for symbol in current.positions:
        desired_positions.setdefault(symbol, 0.0)

    cash = float(current.cash)
    positions = dict(current.positions)
    slip = config.slippage_bps / 10000.0
    total_fees = 0.0
    total_slippage = 0.0

    for symbol, desired_shares in sorted(desired_positions.items()):
        price = float(prices.get(symbol) or 0.0)
        if price <= 0:
            continue
        before = float(positions.get(symbol, 0.0))
        delta = desired_shares - before
        if abs(delta) < 1e-10:
            continue
        is_buy = delta > 0
        fill_price = price * (1 + slip if is_buy else 1 - slip)
        commission = config.commission_per_trade + abs(delta) * config.commission_per_share
        total_fees += commission
        total_slippage += abs(delta) * abs(fill_price - price)
        if is_buy:
            cost = delta * fill_price + commission
            if cost > cash:
                affordable = max(0.0, (cash - config.commission_per_trade) / max(fill_price + config.commission_per_share, 1e-9))
                events.append({"type": "cash_limited", "symbol": symbol, "requested_shares": delta, "affordable_shares": affordable})
                delta = affordable
                commission = config.commission_per_trade + abs(delta) * config.commission_per_share if delta > 1e-10 else 0.0
                cost = delta * fill_price + commission
            cash -= cost
        else:
            proceeds = abs(delta) * fill_price - commission
            cash += proceeds
        positions[symbol] = before + delta
        if abs(delta) > 1e-10:
            trades.append(
                {
                    "symbol": symbol,
                    "side": "BUY" if delta > 0 else "SELL",
                    "shares": _round(abs(delta)),
                    "signed_delta": _round(delta),
                    "reference_price": _round(price),
                    "fill_price": _round(fill_price),
                    "commission": _round(commission),
                }
            )

    next_book = _book_from_positions(cash, positions, prices)
    return next_book, {
        "portfolio_before": current.model_dump() if hasattr(current, "model_dump") else current.dict(),
        "portfolio_after": next_book.model_dump() if hasattr(next_book, "model_dump") else next_book.dict(),
        "target_weights": {symbol: _round(value) for symbol, value in clean_weights.items()},
        "executed_target_weights": {symbol: _round(value) for symbol, value in clean_weights.items()},
        "trades": trades,
        "events": events,
        "fees": _round(total_fees),
        "slippage_cost": _round(total_slippage),
        "model_failure": any(event["type"] in {"invalid_weight", "invalid_symbol_or_price", "invalid_gross_exposure"} for event in events),
    }
