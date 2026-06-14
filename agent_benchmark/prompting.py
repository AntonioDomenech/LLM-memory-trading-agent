from __future__ import annotations

import json
from typing import Any, Dict, Tuple


SYSTEM_PROMPT = """You are the sole AI portfolio manager in a market benchmark.

You must make the final investment decision from the supplied market state.
There is no human risk engine after you. You choose the action, target exposure,
horizon, sizing, and risk plan. The simulator will only apply market mechanics:
cash, positions, fees, slippage, shorting rules, and fill prices.

Use only information in the input bundle. Treat missing or low-quality data as
part of the decision problem. Do not request more data. Do not mention that you
are an AI model. Return only a single JSON object.

Required JSON fields:
- action: BUY, SELL, or HOLD
- target_exposure: number, where 1.0 means 100% long, 0 means cash, -1.0 means 100% short
- confidence: number from 0 to 1
- horizon_days: integer
- expected_return_bps: number
- risk_plan: object with max_loss_pct, stop_loss_price, take_profit_price, and invalidation
- reasoning_summary: concise explanation of the decision
- used_information: array naming the most important input sections used
- uncertainty: array of important unknowns or weak inputs
"""


def build_decision_prompt(input_bundle: Dict[str, Any]) -> Tuple[str, str]:
    user_payload = {
        "task": "Make the next trading decision for the benchmark portfolio.",
        "decision_output_contract": {
            "action": "BUY | SELL | HOLD",
            "target_exposure": "float",
            "confidence": "float 0..1",
            "horizon_days": "integer",
            "expected_return_bps": "float",
            "risk_plan": {
                "max_loss_pct": "float or null",
                "stop_loss_price": "float or null",
                "take_profit_price": "float or null",
                "invalidation": "string",
            },
            "reasoning_summary": "short string",
            "used_information": ["market", "news", "fundamentals", "macro", "memory"],
            "uncertainty": ["string"],
        },
        "input_bundle": input_bundle,
    }
    return SYSTEM_PROMPT, json.dumps(user_payload, sort_keys=True, default=str)
