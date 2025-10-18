from dotenv import load_dotenv
load_dotenv()
from core.config import load_config
from core.indicators import add_indicators
from core.data_fetcher import get_daily_bars
from core.pipeline import prepare_daily_context
from core.llm import chat_json
import pandas as pd

cfg = load_config('config.json')
df = add_indicators(get_daily_bars(cfg.symbol, cfg.test_start, cfg.test_end))
df['date'] = pd.to_datetime(df['date']).dt.date
for idx, row in df.head(40).iterrows():
    ctx = prepare_daily_context(cfg, row['date'].isoformat(), row.to_dict(), content_policy='auto')
    try:
        decision = chat_json(ctx.policy_prompt.as_messages(), model=cfg.decision_model, max_tokens=120)
        if decision.get('action') != 'HOLD' or decision.get('target_exposure',0) != 0:
            print('non-hold', row['date'], decision)
    except Exception as exc:
        print('exception', row['date'], exc)
