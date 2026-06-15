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


STAGE1_SYSTEM_PROMPT = """You are the analyst stage of an AI market benchmark.

Use only the compact point-in-time bundle. The memory items are deterministic
historical cases, not model-written lessons. Score each supplied symbol. Keep
every string short; evidence, memory, and uncertainty arrays should contain at
most 1 terse item each. Use minified JSON and do not include zero-weight filler.

Return only compact JSON:
{
  "analyses": [
    {
      "symbol": "AAPL",
      "stance": "bullish | bearish | neutral | uncertain",
      "confidence": 0.0,
      "expected_return_bps": 0,
      "horizon_days": 20,
      "key_evidence": ["..."],
      "memory_refs": ["..."],
      "uncertainty": ["..."],
      "proposed_target_weight": 0.0
    }
  ],
  "market_regime_notes": "...",
  "data_quality_notes": ["..."]
}
"""


STAGE2_SYSTEM_PROMPT = """You are the portfolio manager stage of an AI market benchmark.

You own the final investment decision. The simulator will only apply mechanical constraints:
cash, fills, fees, slippage, shorting, and gross exposure.
It will not add market expertise after you speak.

Use only the compact portfolio bundle, deterministic historical memory, and
Stage 1 outputs. Return final target weights. Weights may be negative only when
shorting is enabled. Keep strings short, omit zero target weights, and do not
repeat Stage 1 evidence. Prefer a sparse portfolio with 8 to 12 nonzero
positions; fewer is valid, including all cash.

Portfolio weight rule: sum(abs(target_weights.values())) must be <= max_gross_exposure.
gross_exposure must equal that sum, and net_exposure must equal sum(target_weights.values()).
If you exceed the limit, the simulator rejects the allocation as a model failure.

Return only compact JSON:
{
  "target_weights": {"AAPL": 0.05},
  "cash_target_weight": 0.35,
  "gross_exposure": 0.65,
  "net_exposure": 0.65,
  "confidence": 0.0,
  "portfolio_thesis": "...",
  "major_risks": ["..."],
  "uncertainty": ["..."],
  "expected_return_bps": 0,
  "horizon_days": 20
}
"""


def build_stage1_prompt(input_bundle: Dict[str, Any], symbols: list[str]) -> Tuple[str, str]:
    user_payload = {
        "task": "Score these candidate stocks for the benchmark portfolio and return valid JSON only.",
        "symbols_to_score": symbols,
        "input_bundle": input_bundle,
    }
    return STAGE1_SYSTEM_PROMPT, json.dumps(user_payload, sort_keys=True, default=str)


def build_stage2_prompt(input_bundle: Dict[str, Any], stage1_outputs: list[Dict[str, Any]]) -> Tuple[str, str]:
    user_payload = {
        "task": "Choose final portfolio target weights for the benchmark and return valid JSON only.",
        "stage1_outputs": stage1_outputs,
        "input_bundle": input_bundle,
    }
    return STAGE2_SYSTEM_PROMPT, json.dumps(user_payload, sort_keys=True, default=str)
