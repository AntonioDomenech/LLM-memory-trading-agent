from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .api_usage import attach_response_usage_metadata
from .config_store import DATA_DIR
from .local_provider import is_local_model_run, local_auth_headers, model_base_url, validate_no_paid_api_mode
from .schemas import BenchmarkConfig, SecretConfig

LLM_CACHE = DATA_DIR / "cache" / "llm"
MALFORMED_CACHE = LLM_CACHE / "malformed"


def _base_url(secrets: SecretConfig, config: BenchmarkConfig | None = None) -> str:
    if config is not None:
        return model_base_url(config, secrets)
    return (secrets.openai_base_url or "https://api.openai.com/v1").rstrip("/")


def _auth_headers(secrets: SecretConfig, config: BenchmarkConfig | None = None) -> Dict[str, str]:
    return local_auth_headers(config, secrets)


def list_openai_models(secrets: SecretConfig) -> Dict[str, Any]:
    if not secrets.openai_api_key:
        return {"models": [], "error": "missing_openai_api_key"}
    resp = requests.get(f"{_base_url(secrets)}/models", headers=_auth_headers(secrets), timeout=20)
    resp.raise_for_status()
    data = resp.json()
    models = sorted(
        [
            {
                "id": item.get("id"),
                "created": item.get("created"),
                "owned_by": item.get("owned_by"),
            }
            for item in data.get("data", [])
            if item.get("id")
        ],
        key=lambda item: item["id"],
    )
    return {"models": models}


def _extract_output_text(data: Dict[str, Any]) -> str:
    if data.get("output_text"):
        return data["output_text"]
    message = data.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content") or ""
        if content:
            return content
        if message.get("reasoning") or message.get("thinking"):
            return message.get("reasoning") or message.get("thinking") or ""
    chunks: List[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(content["text"])
    if chunks:
        return "\n".join(chunks)
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if content:
            return content
        return message.get("reasoning") or message.get("thinking") or ""
    return json.dumps(data)


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            raise ValueError(f"Model response contained malformed JSON: {exc}") from exc
    raise ValueError("Model response did not contain a JSON object")


def _cache_path(model: str, system: str, user: str) -> Path:
    digest = hashlib.sha256(f"{model}\n{system}\n{user}".encode("utf-8")).hexdigest()
    return LLM_CACHE / f"{digest}.json"


def _named_cache_path(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return LLM_CACHE / namespace / f"{digest}.json"


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    last_error: requests.HTTPError | None = None
    for attempt in range(8):
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        if resp.status_code < 400:
            return resp.json()

        error = requests.HTTPError(resp.text, response=resp)
        last_error = error
        retryable = resp.status_code == 429 or resp.status_code in {500, 502, 503, 504}
        if not retryable or attempt >= 7:
            raise error

        retry_after = _retry_after_seconds(resp, default=min(60.0, 2.0 ** attempt))
        time.sleep(retry_after)

    if last_error:
        raise last_error
    raise RuntimeError("Request failed before a response was returned")


def _retry_after_seconds(resp: requests.Response, *, default: float) -> float:
    header = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if header:
        try:
            return max(0.5, min(120.0, float(header)))
        except Exception:
            pass
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", resp.text or "", re.IGNORECASE)
    if match:
        return max(0.5, min(120.0, float(match.group(1)) + 0.5))
    return max(0.5, min(120.0, float(default)))


def _write_malformed_response(namespace: str, key: str, text: str, response_data: Dict[str, Any], error: str) -> None:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    path = MALFORMED_CACHE / namespace / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "error": error,
                "raw_text": text,
                "response": response_data,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def _call_responses(config: BenchmarkConfig, secrets: SecretConfig, system: str, user: str) -> Dict[str, Any]:
    payload = {
        "model": config.model,
        "instructions": system,
        "input": user,
        "max_output_tokens": config.max_output_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    try:
        return _post_json(f"{_base_url(secrets, config)}/responses", _auth_headers(secrets, config), payload)
    except requests.HTTPError as exc:
        text = str(exc)
        if "temperature" in text:
            payload.pop("temperature", None)
            return _post_json(f"{_base_url(secrets, config)}/responses", _auth_headers(secrets, config), payload)
        raise


def _call_chat_completions(config: BenchmarkConfig, secrets: SecretConfig, system: str, user: str) -> Dict[str, Any]:
    base_payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }
    if config.temperature is not None:
        base_payload["temperature"] = config.temperature
    if is_local_model_run(config, secrets):
        base_payload["think"] = False

    url = f"{_base_url(secrets, config)}/chat/completions"
    for token_field in ("max_completion_tokens", "max_tokens"):
        payload = dict(base_payload)
        payload[token_field] = config.max_output_tokens
        try:
            return _post_json(url, _auth_headers(secrets, config), payload)
        except requests.HTTPError as exc:
            text = str(exc)
            if "temperature" in text and "temperature" in payload:
                payload.pop("temperature", None)
                return _post_json(url, _auth_headers(secrets, config), payload)
            if token_field == "max_completion_tokens" and "max_completion_tokens" in text:
                continue
            raise

    raise RuntimeError("Chat Completions request failed before a response was returned")


def _ollama_native_base_url(config: BenchmarkConfig, secrets: SecretConfig) -> str:
    base = _base_url(secrets, config)
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _call_ollama_native_chat(config: BenchmarkConfig, secrets: SecretConfig, system: str, user: str, *, cache_namespace: str) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "num_predict": config.max_output_tokens,
        "num_ctx": max(4096, int(getattr(config, "local_ollama_num_ctx", 4096) or 4096)),
    }
    if config.temperature is not None:
        options["temperature"] = config.temperature
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "format": _ollama_schema_for_namespace(config, cache_namespace),
        "options": options,
    }
    data = _post_json(f"{_ollama_native_base_url(config, secrets)}/api/chat", _auth_headers(secrets, config), payload)
    prompt_tokens = _int(data.get("prompt_eval_count"))
    output_tokens = _int(data.get("eval_count"))
    if prompt_tokens or output_tokens:
        data["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        }
    return data


def _ollama_schema_for_namespace(config: BenchmarkConfig, namespace: str) -> Dict[str, Any] | str:
    namespace = namespace or ""
    if namespace.startswith("stage1"):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "analyses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral", "uncertain"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "expected_return_bps": {"type": "number"},
                            "horizon_days": {"type": "integer", "minimum": 1},
                            "key_evidence": {"type": "array", "items": {"type": "string"}},
                            "memory_refs": {"type": "array", "items": {"type": "string"}},
                            "uncertainty": {"type": "array", "items": {"type": "string"}},
                            "proposed_target_weight": {"type": "number", "minimum": -1, "maximum": 1},
                        },
                        "required": ["symbol", "stance", "confidence", "expected_return_bps", "horizon_days", "key_evidence", "memory_refs", "uncertainty", "proposed_target_weight"],
                    },
                },
                "market_regime_notes": {"type": "string"},
                "data_quality_notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["analyses", "market_regime_notes", "data_quality_notes"],
        }
    if namespace.startswith("exposure-critic"):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "bull_exposure_case": {"type": "string"},
                "defensive_case": {"type": "string"},
                "cash_drag_risk": {"type": "string"},
                "recommended_exposure_band": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "key_disagreement": {"type": "string"},
            },
            "required": ["bull_exposure_case", "defensive_case", "cash_drag_risk", "recommended_exposure_band", "key_disagreement"],
        }
    if namespace.startswith("reflection-lesson"):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "summary_lesson": {"type": "string"},
                "lesson_tags": {"type": "array", "items": {"type": "string"}},
                "use_in_future_if": {"type": "string"},
                "avoid_if": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["summary_lesson", "lesson_tags", "use_in_future_if", "avoid_if", "confidence"],
        }
    if namespace.startswith("stage2") and config.mode == "single_stock":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_exposure": {"type": "number", "minimum": -float(config.max_gross_exposure or 1.0) if config.allow_short else 0.0, "maximum": float(config.max_gross_exposure or 1.0)},
                "expected_holding_days": {"type": "integer", "minimum": 1, "maximum": 60},
                "rebalance_reason": {"type": "string"},
                "input_evidence_refs": {"type": "array", "items": {"type": "string"}},
                "data_quality_warnings_used": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "portfolio_thesis": {"type": "string"},
                "major_risks": {"type": "array", "items": {"type": "string"}},
                "uncertainty": {"type": "array", "items": {"type": "string"}},
                "expected_return_bps": {"type": "number"},
                "horizon_days": {"type": "integer", "minimum": 1, "maximum": 60},
                "cash_drag_justification": {"type": "string"},
                "why_not_buy_hold": {"type": "string"},
                "stage1_alignment": {"type": "string", "enum": ["follow", "partial", "veto"]},
                "stage1_veto_reason": {"type": "string"},
            },
            "required": [
                "target_exposure",
                "expected_holding_days",
                "rebalance_reason",
                "input_evidence_refs",
                "data_quality_warnings_used",
                "confidence",
                "portfolio_thesis",
                "major_risks",
                "uncertainty",
                "expected_return_bps",
                "horizon_days",
                "cash_drag_justification",
                "why_not_buy_hold",
                "stage1_alignment",
                "stage1_veto_reason",
            ],
        }
    if namespace.startswith("stage2"):
        return {
            "type": "object",
            "properties": {
                "target_weights": {"type": "object"},
                "cash_weight": {"type": "number"},
                "gross_exposure": {"type": "number"},
                "net_exposure": {"type": "number"},
                "expected_holding_days": {"type": "integer", "minimum": 1},
                "estimated_turnover": {"type": "number"},
                "estimated_slippage_cost_bps": {"type": "number"},
                "rebalance_reason": {"type": "string"},
                "input_evidence_refs": {"type": "array", "items": {"type": "string"}},
                "data_quality_warnings_used": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "portfolio_thesis": {"type": "string"},
                "major_risks": {"type": "array", "items": {"type": "string"}},
                "uncertainty": {"type": "array", "items": {"type": "string"}},
                "expected_return_bps": {"type": "number"},
                "horizon_days": {"type": "integer", "minimum": 1},
            },
            "required": ["target_weights", "cash_weight", "gross_exposure", "net_exposure", "expected_holding_days", "estimated_turnover", "estimated_slippage_cost_bps", "rebalance_reason", "input_evidence_refs", "data_quality_warnings_used", "confidence", "portfolio_thesis", "major_risks", "uncertainty", "expected_return_bps", "horizon_days"],
        }
    return "json"


def _with_output_tokens(config: BenchmarkConfig, max_output_tokens: int) -> BenchmarkConfig:
    if max_output_tokens <= config.max_output_tokens:
        return config
    if hasattr(config, "model_copy"):
        return config.model_copy(update={"max_output_tokens": max_output_tokens})
    payload = config.dict()
    payload["max_output_tokens"] = max_output_tokens
    return BenchmarkConfig(**payload)


def _response_hit_output_limit(response_data: Dict[str, Any]) -> bool:
    if response_data.get("done_reason") == "length":
        return True
    incomplete = response_data.get("incomplete_details") or {}
    if response_data.get("status") == "incomplete" and incomplete.get("reason") == "max_output_tokens":
        return True
    for choice in response_data.get("choices") or []:
        if choice.get("finish_reason") == "length":
            return True
    return False


def call_json_model(
    config: BenchmarkConfig,
    secrets: SecretConfig,
    system: str,
    user: str,
    *,
    dry_run: bool = False,
    fallback: Optional[Dict[str, Any]] = None,
    cache_namespace: str = "json",
) -> Dict[str, Any]:
    fallback = dict(fallback or {})
    validate_no_paid_api_mode(config, secrets)
    if dry_run:
        fallback.setdefault("_raw_text", "")
        fallback["_api_status"] = "dry_run"
        return fallback

    local_run = is_local_model_run(config, secrets)
    if not secrets.openai_api_key and not local_run:
        fallback.setdefault("_raw_text", "")
        fallback.setdefault("uncertainty", [])
        fallback["uncertainty"] = [*fallback.get("uncertainty", []), "Missing OpenAI API key."]
        fallback["_api_status"] = "missing_key"
        return fallback

    if config.use_cached_llm:
        path = _named_cache_path(cache_namespace, f"{config.model}\n{system}\n{user}")
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            cached["_api_cache_hit"] = True
            cached.setdefault("_api_status", "ok")
            return cached

    parse_errors: List[str] = []
    request_key = f"{config.model}\n{system}\n{user}"
    response_data = {}
    text = ""
    active_system = system
    active_user = user
    active_config = config

    for attempt in range(3):
        api_errors = []
        started_at = time.time()
        try:
            if local_run and active_config.model_provider == "ollama_local":
                response_data = _call_ollama_native_chat(active_config, secrets, active_system, active_user, cache_namespace=cache_namespace)
            elif active_config.endpoint == "chat_completions":
                response_data = _call_chat_completions(active_config, secrets, active_system, active_user)
            else:
                response_data = _call_responses(active_config, secrets, active_system, active_user)
        except Exception as exc:
            api_errors.append(str(exc))
            if local_run and active_config.model_provider == "ollama_local":
                response_data = _call_ollama_native_chat(active_config, secrets, active_system, active_user, cache_namespace=cache_namespace)
            else:
                response_data = _call_chat_completions(active_config, secrets, active_system, active_user)
        latency_seconds = max(0.0, time.time() - started_at)

        text = _extract_output_text(response_data)
        try:
            decision = _extract_json(text)
            decision["_raw_text"] = text
            decision["_api_status"] = "ok"
            attach_response_usage_metadata(decision, response_data)
            decision["_api_provider"] = "ollama_local" if local_run else "openai"
            decision["_api_base_url"] = _base_url(secrets, config)
            decision["_api_latency_seconds"] = round(latency_seconds, 6)
            decision["_api_chars_per_second"] = round(len(text or "") / latency_seconds, 4) if latency_seconds else None
            usage = decision.get("_api_usage") or {}
            output_tokens = usage.get("output_tokens")
            if output_tokens and latency_seconds:
                decision["_api_output_tokens_per_second"] = round(float(output_tokens) / latency_seconds, 4)
            if attempt:
                decision["_api_retry_count"] = attempt
                decision["_api_retry_max_output_tokens"] = active_config.max_output_tokens
            if api_errors:
                decision["_api_fallback_errors"] = api_errors

            if config.use_cached_llm:
                path = _named_cache_path(cache_namespace, request_key)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
            return decision
        except Exception as exc:
            error = str(exc)
            parse_errors.append(error)
            _write_malformed_response(cache_namespace, f"{request_key}\n{attempt}", text, response_data, error)
            if _response_hit_output_limit(response_data):
                next_limit = min(max(active_config.max_output_tokens * 2, active_config.max_output_tokens + 600), 3200)
                active_config = _with_output_tokens(active_config, next_limit)
                active_system = (
                    f"{system}\n"
                    "Return exactly one compact valid JSON object. Use very short strings, "
                    "arrays with at most one item, and no markdown or trailing text."
                )
                active_user = user
            else:
                active_system = f"{system}\nReturn exactly one valid JSON object. Do not include markdown, prose, or trailing text."
                active_user = json.dumps(
                    {
                        "task": "Repair the previous response for the same benchmark request. Return only valid JSON.",
                        "parser_error": error,
                        "previous_response": text[:2000],
                        "original_request": user,
                    },
                    sort_keys=True,
                    default=str,
                )

    raise ValueError(f"Model response JSON parsing failed after {len(parse_errors)} attempts: {'; '.join(parse_errors)}")


def call_decision_model(config: BenchmarkConfig, secrets: SecretConfig, system: str, user: str, dry_run: bool = False) -> Dict[str, Any]:
    fallback = {
        "action": "HOLD",
        "target_exposure": 0.0,
        "confidence": 0.0,
        "horizon_days": 1,
        "expected_return_bps": 0,
        "risk_plan": {
            "max_loss_pct": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "invalidation": "No model call made.",
        },
        "reasoning_summary": "Fallback placeholder decision.",
        "used_information": [],
        "uncertainty": [],
    }
    if dry_run:
        fallback["risk_plan"]["invalidation"] = "Dry run: no model call made."
        fallback["reasoning_summary"] = "Dry run placeholder decision."
        fallback["uncertainty"] = ["No model was called."]
    if not secrets.openai_api_key:
        fallback["risk_plan"]["invalidation"] = "Missing OpenAI API key."
        fallback["reasoning_summary"] = "No OpenAI API key configured."
    return call_json_model(config, secrets, system, user, dry_run=dry_run, fallback=fallback, cache_namespace="decision")


def local_embedding(text: str, dimensions: int = 96) -> List[float]:
    import math
    import re

    vector = [0.0] * dimensions
    for token in re.findall(r"[a-zA-Z0-9_.$%-]+", (text or "").lower()):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 8) for value in vector]


def embed_text(text: str, secrets: SecretConfig, *, provider: str = "local") -> List[float]:
    if provider != "openai" or not secrets.openai_api_key:
        return local_embedding(text)
    key = f"{secrets.openai_embedding_model}\n{text}"
    path = _named_cache_path("embeddings", key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {"model": secrets.openai_embedding_model, "input": text or ""}
    data = _post_json(f"{_base_url(secrets)}/embeddings", _auth_headers(secrets), payload)
    embedding = data.get("data", [{}])[0].get("embedding")
    if not isinstance(embedding, list):
        return local_embedding(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(embedding), encoding="utf-8")
    return embedding


def _int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0
