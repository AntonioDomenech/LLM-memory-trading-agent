
import json
import os
from functools import lru_cache
from typing import Any, Dict, List

from .logger import get_logger

log = get_logger()


def _fallback_payload(reason: str = ""):
    """Return the safe default payload when the API is unavailable."""

    payload = {
        "mood_score": 0.5,
        "narrative_bias": 0.0,
        "novelty": 0.1,
        "credibility": 0.5,
        "regime_alignment": 0.5,
        "confidence": 0.5,
        "action": "HOLD",
        "target_exposure": 0.0,
        "horizon_days": 5,
        "expected_return_bps": 0,
    }
    if reason:
        payload["__warning__"] = reason
    return payload


@lru_cache(maxsize=1)
def _build_client(api_key: str):
    """Initialise and cache the OpenAI SDK client."""

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _is_gpt5(model_name: str) -> bool:
    """Detect whether the requested model belongs to the GPT-5 family."""

    if not model_name:
        return False
    name = model_name.lower()
    return name.startswith("gpt-5") or name.startswith("o5-")


def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert chat-style messages into Responses API format."""

    converted: List[Dict[str, Any]] = []
    for msg in messages or []:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content is None:
            content = ""
        converted.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": str(content),
                    }
                ],
            }
        )
    return converted


def chat_json(messages, model="gpt-4.1-mini", timeout=15, max_tokens=200):
    """Call the OpenAI API expecting a JSON object response (GPT‑4/5 compatible)."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:  # attempt lazy load from .env for non-Streamlit callers
            from dotenv import load_dotenv  # type: ignore

            load_dotenv(override=False)
            api_key = os.environ.get("OPENAI_API_KEY")
        except Exception:  # pragma: no cover - optional dependency
            api_key = None
    if not api_key:
        reason = "OPENAI_API_KEY not set; using fallback response."
        log.warning(reason)
        return _fallback_payload(reason)

    try:
        client = _build_client(api_key)
    except Exception as exc:  # pragma: no cover - defensive guard
        reason = f"Failed to initialise OpenAI client: {exc}"
        log.warning(reason)
        return _fallback_payload(reason)

    try:
        if _is_gpt5(model):
            # GPT-5 models require the Responses API and tend to emit long reasoning traces.
            max_out = max(int(max_tokens or 0), 2048)
            resp = client.responses.create(
                model=model,
                input=_to_responses_input(messages),
                text={"format": {"type": "json_object"}},
                max_output_tokens=max_out,
                timeout=timeout,
            )
            content = getattr(resp, "output_text", None)
            if not content:
                # Concatenate message text chunks when output_text is absent.
                parts = []
                for item in getattr(resp, "output", []) or []:
                    if getattr(item, "type", None) != "message":
                        continue
                    for block in getattr(item, "content", []) or []:
                        text = getattr(block, "text", None)
                        if text:
                            parts.append(text)
                content = "".join(parts)
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            choice = resp.choices[0]
            content = choice.message.content if choice and choice.message else ""

        if not content:
            raise ValueError("Empty response content")

        return json.loads(content)

    except Exception as exc:
        reason = f"OpenAI chat_json failed: {exc}"
        log.warning(reason)
        return _fallback_payload(reason)
