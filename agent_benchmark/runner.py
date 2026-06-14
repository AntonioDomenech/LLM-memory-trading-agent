from __future__ import annotations

import uuid
from typing import Any, Dict, List

from .broker import execute_decision
from .information import build_information_bundle
from .llm_client import call_decision_model
from .market_data import load_prices
from .prompting import build_decision_prompt
from .schemas import BenchmarkConfig, PortfolioState, SecretConfig, model_to_dict
from .storage import BenchmarkStore


def _metrics(equity_curve: List[Dict[str, Any]], initial_cash: float) -> Dict[str, Any]:
    if not equity_curve:
        return {"total_return": 0.0, "max_drawdown": 0.0}
    peak = initial_cash
    max_drawdown = 0.0
    for point in equity_curve:
        equity = float(point["equity"])
        peak = max(peak, equity)
        if peak > 0:
            max_drawdown = min(max_drawdown, equity / peak - 1.0)
    final = float(equity_curve[-1]["equity"])
    return {
        "total_return": final / initial_cash - 1.0 if initial_cash else 0.0,
        "final_equity": final,
        "max_drawdown": max_drawdown,
    }


def run_benchmark(config: BenchmarkConfig, secrets: SecretConfig, dry_run: bool = False) -> Dict[str, Any]:
    store = BenchmarkStore()
    run_id = str(uuid.uuid4())
    model = config.model or "unselected-model"

    prices = load_prices(config.symbol, config.start_date, config.end_date)
    prices = prices[(prices["date"].astype(str) >= config.start_date) & (prices["date"].astype(str) <= config.end_date)]
    prices = prices.sort_values("date").head(max(1, config.max_days))
    if prices.empty:
        raise ValueError("No trading dates available for the requested benchmark window.")

    portfolio = PortfolioState(cash=config.initial_cash, position_shares=0.0, equity=config.initial_cash)
    equity_curve: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    event_count = 0
    model_failures = 0

    for _, row in prices.iterrows():
        date_iso = str(row["date"])
        bundle = build_information_bundle(config, secrets, date_iso, portfolio, store)
        system, user = build_decision_prompt(bundle)
        try:
            decision = call_decision_model(config, secrets, system, user, dry_run=dry_run)
        except Exception as exc:
            decision = {
                "action": "HOLD",
                "target_exposure": (portfolio.position_shares * float(row["close"])) / max(1e-9, portfolio.equity),
                "confidence": 0.0,
                "horizon_days": 1,
                "expected_return_bps": 0,
                "risk_plan": {"max_loss_pct": None, "stop_loss_price": None, "take_profit_price": None, "invalidation": str(exc)},
                "reasoning_summary": "Model call or JSON parsing failed.",
                "used_information": [],
                "uncertainty": [str(exc)],
                "_api_status": "error",
            }
        portfolio, execution = execute_decision(portfolio, decision, float(row["close"]), config)
        event_count += len(execution.get("events") or [])
        if execution.get("model_failure") or decision.get("_api_status") in {"missing_key", "error"}:
            model_failures += 1

        point = {
            "date": date_iso,
            "equity": portfolio.equity,
            "cash": portfolio.cash,
            "position_shares": portfolio.position_shares,
            "close": float(row["close"]),
        }
        equity_curve.append(point)
        record = {"date": date_iso, "decision": decision, "execution": execution, "equity": point}
        decisions.append(record)
        store.save_decision(
            run_id=run_id,
            date=date_iso,
            symbol=config.symbol.upper(),
            model=model,
            input_bundle=bundle,
            decision=decision,
            execution=execution,
        )

    summary = {
        "run_id": run_id,
        "symbol": config.symbol.upper(),
        "model": model,
        "days": len(equity_curve),
        "event_count": event_count,
        "model_failures": model_failures,
        "metrics": _metrics(equity_curve, config.initial_cash),
        "equity_curve": equity_curve,
        "dry_run": dry_run,
    }
    store.save_run(run_id, model_to_dict(config), summary)
    return {"summary": summary, "decisions": decisions}
