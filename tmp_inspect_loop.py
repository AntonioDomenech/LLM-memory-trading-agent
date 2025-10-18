import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from core.config import load_config
from core.indicators import add_indicators
from core.data_fetcher import get_daily_bars
from core.pipeline import prepare_daily_context

load_dotenv()
client = OpenAI()
cfg = load_config('config.json')
df = add_indicators(get_daily_bars(cfg.symbol, cfg.test_start, cfg.test_end))
df['date'] = pd.to_datetime(df['date']).dt.date
for idx, row in df.iloc[:40].iterrows():
    ctx = prepare_daily_context(cfg, row['date'].isoformat(), row.to_dict(), content_policy='auto')
    messages = ctx.policy_prompt.as_messages()
    responses_input = [
        {"role": m["role"], "content": [{"type": "input_text", "text": m["content"]}]}
        for m in messages
    ]
    resp = client.responses.create(
        model=cfg.decision_model,
        input=responses_input,
        text={"format": {"type": "json_object"}},
        max_output_tokens=2048,
    )
    content = resp.output_text
    if not content:
        print('empty output', row['date'], resp)
    else:
        try:
            data = json.loads(content)
            if data.get('action','HOLD') != 'HOLD' or data.get('target_exposure',0) != 0:
                print('non hold', row['date'], data)
        except Exception as exc:
            print('parse error', row['date'], exc, 'raw=', content)
