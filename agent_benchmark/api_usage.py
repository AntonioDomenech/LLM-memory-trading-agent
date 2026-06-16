from __future__ import annotations

import json
import math
from typing import Any, Dict, List

from .prompting import build_stage1_prompt, build_stage2_prompt
from .schemas import BenchmarkConfig


PRICING_SOURCE_URL = "https://openai.com/api/pricing/"

# USD per 1M tokens. Ordered most-specific first so dated mini model ids do not
# match their larger family before the mini price.
OPENAI_PRICING_USD_PER_1M = [
    {
        "match": "gpt-5.4-mini",
        "label": "GPT-5.4 mini",
        "input": 0.75,
        "cached_input": 0.075,
        "output": 4.50,
    },
    {
        "match": "gpt-5.5",
        "label": "GPT-5.5",
        "input": 5.00,
        "cached_input": 0.50,
        "output": 30.00,
    },
    {
        "match": "gpt-5.4",
        "label": "GPT-5.4",
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
    },
]


def estimate_text_tokens(text: str) -> int:
    """Cheap local approximation used when provider usage is unavailable."""
    return max(1, int(math.ceil(len(text or "") / 4)))


def normalize_api_usage(raw_usage: Dict[str, Any] | None) -> Dict[str, int]:
    if not isinstance(raw_usage, dict) or not raw_usage:
        return {}

    input_tokens = _int(raw_usage.get("input_tokens", raw_usage.get("prompt_tokens")))
    output_tokens = _int(raw_usage.get("output_tokens", raw_usage.get("completion_tokens")))
    total_tokens = _int(raw_usage.get("total_tokens"))
    input_details = raw_usage.get("input_tokens_details") or raw_usage.get("prompt_tokens_details") or {}
    cached_input_tokens = _int(raw_usage.get("cached_input_tokens", input_details.get("cached_tokens")))

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    if input_tokens == 0 and total_tokens and output_tokens:
        input_tokens = max(0, total_tokens - output_tokens)
    if output_tokens == 0 and total_tokens and input_tokens:
        output_tokens = max(0, total_tokens - input_tokens)

    if input_tokens == 0 and output_tokens == 0 and total_tokens == 0:
        return {}
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": min(cached_input_tokens, input_tokens),
        "output_tokens": output_tokens,
        "total_tokens": total_tokens or input_tokens + output_tokens,
    }


def attach_response_usage_metadata(decision: Dict[str, Any], response_data: Dict[str, Any]) -> None:
    usage = normalize_api_usage((response_data or {}).get("usage"))
    if usage:
        decision["_api_usage"] = usage
        decision["_api_usage_source"] = "provider_usage"


def pricing_for_model(model: str) -> Dict[str, Any] | None:
    model_lc = (model or "").lower()
    for pricing in OPENAI_PRICING_USD_PER_1M:
        if model_lc.startswith(pricing["match"]):
            return {**pricing, "currency": "USD", "per_tokens": 1_000_000, "source_url": PRICING_SOURCE_URL}
    return None


def estimate_run_api_usage(
    run: Dict[str, Any],
    config: BenchmarkConfig | None = None,
    *,
    summary_override: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    decisions = run.get("decisions") or []
    summary = summary_override or run.get("summary") or {}
    config = config or _config_from_run(run)
    model = (summary.get("model") or run.get("model") or getattr(config, "model", "") or "").strip()
    pricing = pricing_for_model(model)

    if not decisions:
        return {
            "status": "unavailable",
            "message": "No saved decisions are available to estimate API usage.",
            "model": model,
            "pricing": pricing,
            "pricing_source_url": PRICING_SOURCE_URL,
        }

    exact = _usage_from_provider_metadata(decisions)
    if exact["calls_with_usage"] and exact["missing_billable_call_estimate"] == 0:
        usage = {
            "token_source": "provider_usage",
            "is_estimate": False,
            "estimator": "OpenAI usage field saved with each response",
            **exact,
        }
    else:
        estimated = _usage_from_saved_prompts(decisions, config)
        usage = {
            "token_source": "saved_prompt_estimate",
            "is_estimate": True,
            "estimator": "characters divided by 4 from saved prompts and outputs",
            **estimated,
        }
        if exact["calls_with_usage"]:
            usage["provider_usage_partial"] = exact

    logical_model_calls = _int(summary.get("model_calls") or (run.get("progress") or {}).get("model_calls"))
    if logical_model_calls:
        usage["logical_model_calls"] = logical_model_calls
    usage["model"] = model
    usage["status"] = "ok"
    usage["pricing"] = pricing
    usage["pricing_source_url"] = PRICING_SOURCE_URL
    usage["currency"] = "USD"
    usage["estimated_cost_usd"] = _estimate_cost_usd(
        usage["input_tokens"],
        usage.get("cached_input_tokens", 0),
        usage["output_tokens"],
        pricing,
    )
    usage["price_available"] = pricing is not None
    usage["notes"] = _usage_notes(usage)
    return usage


def _usage_from_provider_metadata(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []
    local_cache_hits = 0
    logical_calls = 0
    for decision in decisions:
        output = decision.get("output") or {}
        if (output.get("_api_status") or "ok") in {"dry_run", "missing_key"}:
            continue
        if decision.get("stage") in {"stage1", "stage2"}:
            logical_calls += 1
        if decision.get("stage") == "stage2":
            repair = output.get("_allocation_repair") or {}
            if repair.get("original_api_usage") or repair.get("attempt_api_usages"):
                _append_usage_call(calls, repair.get("original_api_usage"), repair.get("original_cache_hit"))
                for item in repair.get("attempt_api_usages") or []:
                    _append_usage_call(calls, item.get("usage"), item.get("cache_hit"))
                    logical_calls += 1
                continue
            repair_attempts = _int(repair.get("attempts") or (decision.get("execution") or {}).get("repair_count"))
            logical_calls += repair_attempts
        if output.get("_api_cache_hit"):
            local_cache_hits += 1
            continue
        _append_usage_call(calls, output.get("_api_usage"), output.get("_api_cache_hit"))

    input_tokens = sum(call["input_tokens"] for call in calls)
    cached_input_tokens = sum(call.get("cached_input_tokens", 0) for call in calls)
    output_tokens = sum(call["output_tokens"] for call in calls)
    calls_with_usage = len(calls)
    missing = max(0, logical_calls - calls_with_usage - local_cache_hits)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "calls_with_usage": calls_with_usage,
        "local_cache_hits": local_cache_hits,
        "billable_model_calls": calls_with_usage,
        "missing_billable_call_estimate": missing,
    }


def _usage_from_saved_prompts(decisions: List[Dict[str, Any]], config: BenchmarkConfig) -> Dict[str, Any]:
    stage1_by_date: Dict[str, List[Dict[str, Any]]] = {}
    for decision in decisions:
        if decision.get("stage") == "stage1":
            stage1_by_date.setdefault(str(decision.get("decision_date")), []).append(decision.get("output") or {})

    stage1_inputs: List[int] = []
    stage1_outputs: List[int] = []
    stage2_inputs: List[int] = []
    stage2_outputs: List[int] = []
    repair_count = 0
    local_cache_hits = 0

    for decision in decisions:
        stage = decision.get("stage")
        output = decision.get("output") or {}
        if (output.get("_api_status") or "ok") in {"dry_run", "missing_key"}:
            continue
        if output.get("_api_cache_hit"):
            local_cache_hits += 1
            continue
        input_payload = decision.get("input") or {}
        raw_output = output.get("_raw_text") or json.dumps(output, sort_keys=True, default=str)
        if stage == "stage1":
            symbols = _symbols_from_stage1_decision(decision)
            system, user = build_stage1_prompt(input_payload, symbols)
            stage1_inputs.append(estimate_text_tokens(system) + estimate_text_tokens(user))
            stage1_outputs.append(estimate_text_tokens(raw_output))
        elif stage == "stage2":
            system, user = build_stage2_prompt(input_payload, stage1_by_date.get(str(decision.get("decision_date")), []))
            stage2_inputs.append(estimate_text_tokens(system) + estimate_text_tokens(user))
            stage2_outputs.append(estimate_text_tokens(raw_output))
            repair_count += _int((decision.get("execution") or {}).get("repair_count") or (output.get("_allocation_repair") or {}).get("attempts"))

    avg_stage2_input = sum(stage2_inputs) / len(stage2_inputs) if stage2_inputs else 0
    avg_stage2_output = sum(stage2_outputs) / len(stage2_outputs) if stage2_outputs else 0
    repair_input_tokens = int(round(repair_count * avg_stage2_input))
    repair_output_tokens = int(round(repair_count * avg_stage2_output))
    input_tokens = sum(stage1_inputs) + sum(stage2_inputs) + repair_input_tokens
    output_tokens = sum(stage1_outputs) + sum(stage2_outputs) + repair_output_tokens
    return {
        "input_tokens": int(round(input_tokens)),
        "cached_input_tokens": 0,
        "output_tokens": int(round(output_tokens)),
        "total_tokens": int(round(input_tokens + output_tokens)),
        "stage1_calls": len(stage1_inputs),
        "stage2_calls": len(stage2_inputs),
        "repair_calls": repair_count,
        "local_cache_hits": local_cache_hits,
        "billable_model_calls": len(stage1_inputs) + len(stage2_inputs) + repair_count,
        "estimated_repair_input_tokens": repair_input_tokens,
        "estimated_repair_output_tokens": repair_output_tokens,
    }


def _append_usage_call(calls: List[Dict[str, int]], usage: Any, cache_hit: Any = False) -> None:
    if cache_hit:
        return
    normalized = normalize_api_usage(usage)
    if normalized:
        calls.append(normalized)


def _estimate_cost_usd(input_tokens: int, cached_input_tokens: int, output_tokens: int, pricing: Dict[str, Any] | None) -> float | None:
    if not pricing:
        return None
    cached = min(max(0, cached_input_tokens), max(0, input_tokens))
    uncached = max(0, input_tokens - cached)
    cost = (
        uncached / 1_000_000 * float(pricing["input"])
        + cached / 1_000_000 * float(pricing["cached_input"])
        + max(0, output_tokens) / 1_000_000 * float(pricing["output"])
    )
    return round(cost, 6)


def _usage_notes(usage: Dict[str, Any]) -> List[str]:
    notes = []
    if usage.get("is_estimate"):
        notes.append("Token usage is estimated because exact provider usage was not saved for every call.")
    if not usage.get("price_available"):
        notes.append("No local price table entry exists for this model, so cost is unavailable.")
    if usage.get("local_cache_hits"):
        notes.append("Local LLM cache hits are counted separately and are not treated as new billable API calls.")
    if usage.get("repair_calls"):
        notes.append("Repair calls are included; old runs estimate their repair prompt size from average Stage 2 prompts.")
    return notes


def _symbols_from_stage1_decision(decision: Dict[str, Any]) -> List[str]:
    raw = decision.get("symbol") or ""
    if raw:
        return [item.strip() for item in str(raw).split(",") if item.strip()]
    input_payload = decision.get("input") or {}
    symbols = input_payload.get("symbols") or input_payload.get("symbols_to_score")
    if isinstance(symbols, list):
        return [str(item) for item in symbols]
    return []


def _config_from_run(run: Dict[str, Any]) -> BenchmarkConfig:
    try:
        return BenchmarkConfig(**(run.get("config") or {}))
    except Exception:
        return BenchmarkConfig(model=run.get("model") or "")


def _int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0
