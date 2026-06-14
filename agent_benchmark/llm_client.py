from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

from .config_store import DATA_DIR
from .schemas import BenchmarkConfig, SecretConfig

LLM_CACHE = DATA_DIR / "cache" / "llm"


def _base_url(secrets: SecretConfig) -> str:
    return (secrets.openai_base_url or "https://api.openai.com/v1").rstrip("/")


def _auth_headers(secrets: SecretConfig) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {secrets.openai_api_key}",
        "Content-Type": "application/json",
    }


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
        return message.get("content") or ""
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
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model response did not contain a JSON object")


def _cache_path(model: str, system: str, user: str) -> Path:
    digest = hashlib.sha256(f"{model}\n{system}\n{user}".encode("utf-8")).hexdigest()
    return LLM_CACHE / f"{digest}.json"


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if resp.status_code >= 400:
        raise requests.HTTPError(resp.text, response=resp)
    return resp.json()


def _call_responses(config: BenchmarkConfig, secrets: SecretConfig, system: str, user: str) -> Dict[str, Any]:
    payload = {
        "model": config.model,
        "instructions": system,
        "input": user,
        "max_output_tokens": config.max_output_tokens,
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    try:
        return _post_json(f"{_base_url(secrets)}/responses", _auth_headers(secrets), payload)
    except requests.HTTPError as exc:
        text = str(exc)
        if "temperature" in text:
            payload.pop("temperature", None)
            return _post_json(f"{_base_url(secrets)}/responses", _auth_headers(secrets), payload)
        raise


def _call_chat_completions(config: BenchmarkConfig, secrets: SecretConfig, system: str, user: str) -> Dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": config.max_output_tokens,
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    return _post_json(f"{_base_url(secrets)}/chat/completions", _auth_headers(secrets), payload)


def call_decision_model(config: BenchmarkConfig, secrets: SecretConfig, system: str, user: str, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {
            "action": "HOLD",
            "target_exposure": 0.0,
            "confidence": 0.0,
            "horizon_days": 1,
            "expected_return_bps": 0,
            "risk_plan": {
                "max_loss_pct": None,
                "stop_loss_price": None,
                "take_profit_price": None,
                "invalidation": "Dry run: no model call made.",
            },
            "reasoning_summary": "Dry run placeholder decision.",
            "used_information": [],
            "uncertainty": ["No model was called."],
            "_raw_text": "",
            "_api_status": "dry_run",
        }

    if not secrets.openai_api_key:
        return {
            "action": "HOLD",
            "target_exposure": 0.0,
            "confidence": 0.0,
            "horizon_days": 1,
            "expected_return_bps": 0,
            "risk_plan": {
                "max_loss_pct": None,
                "stop_loss_price": None,
                "take_profit_price": None,
                "invalidation": "Missing OpenAI API key.",
            },
            "reasoning_summary": "No OpenAI API key configured.",
            "used_information": [],
            "uncertainty": ["Missing API key."],
            "_raw_text": "",
            "_api_status": "missing_key",
        }

    if config.use_cached_llm:
        path = _cache_path(config.model, system, user)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    api_errors: List[str] = []
    response_data: Dict[str, Any]
    try:
        if config.endpoint == "chat_completions":
            response_data = _call_chat_completions(config, secrets, system, user)
        else:
            response_data = _call_responses(config, secrets, system, user)
    except Exception as exc:
        api_errors.append(str(exc))
        response_data = _call_chat_completions(config, secrets, system, user)

    text = _extract_output_text(response_data)
    decision = _extract_json(text)
    decision["_raw_text"] = text
    decision["_api_status"] = "ok"
    if api_errors:
        decision["_api_fallback_errors"] = api_errors

    if config.use_cached_llm:
        path = _cache_path(config.model, system, user)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    return decision
