
import os, json

from .logger import get_logger

log = get_logger()


def chat_json(messages, model="gpt-4.1-mini", timeout=15, max_tokens=200):
    """Call OpenAI chat completions API expecting a JSON object response."""

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
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
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        content = resp.choices[0].message.content
        return json.loads(content)
    except Exception as exc:
        log.warning(f"OpenAI chat_json failed: {exc}")
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
