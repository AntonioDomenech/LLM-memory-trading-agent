
import json
import os
from functools import lru_cache

from .logger import get_logger

log = get_logger()


def _fallback_payload():
    """Return the safe default payload when the API is unavailable."""

    return {
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


@lru_cache(maxsize=1)
def _build_client(api_key: str):
    """Initialise and cache the OpenAI SDK client."""

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _is_gpt5(model_name: str) -> bool:
    """Detect whether the requested model belongs to the GPT‑5 family."""

    if not model_name:
        return False
    name = model_name.lower()
    return name.startswith("gpt-5") or name.startswith("o5-")


def chat_json(messages, model="gpt-4.1-mini", timeout=15, max_tokens=200):
    """Call the OpenAI API expecting a JSON object response (GPT‑4/5 compatible)."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.warning("OPENAI_API_KEY not set; using fallback response.")
        return _fallback_payload()

    try:
        client = _build_client(api_key)
    except Exception as exc:  # pragma: no cover - defensive guard
        log.warning(f"Failed to initialise OpenAI client: {exc}")
        return _fallback_payload()

    try:
        if _is_gpt5(model):
            # GPT‑5 models require the Responses API.
            resp = client.responses.create(
                model=model,
                input=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_output_tokens=max_tokens,
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
        log.warning(f"OpenAI chat_json failed: {exc}")
        return _fallback_payload()
