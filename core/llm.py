
import os, json
from .logger import get_logger
log = get_logger()
def chat_json(messages, model="gpt-4.1-mini", timeout=15, max_tokens=200):
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"mood_score":0.5,"narrative_bias":0.0,"novelty":0.1,"credibility":0.5,"regime_alignment":0.5,"confidence":0.5,"action":"HOLD","target_exposure":0.0,"horizon_days":5,"expected_return_bps":0}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model, messages=messages, response_format={"type":"json_object"},
            temperature=0, max_tokens=max_tokens, timeout=timeout,
        )
        c = resp.choices[0].message.content
        return json.loads(c)
    except Exception as e:
        log.warning(f"OpenAI chat_json failed: {e}")
        return {"mood_score":0.5,"narrative_bias":0.0,"novelty":0.1,"credibility":0.5,"regime_alignment":0.5,"confidence":0.5,"action":"HOLD","target_exposure":0.0,"horizon_days":5,"expected_return_bps":0}
