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
first_row = df.iloc[0]
ctx = prepare_daily_context(cfg, first_row['date'].isoformat(), first_row.to_dict(), content_policy='auto')
print('articles', len(ctx.articles))
print('factor prompt call...')
factor = chat_json(ctx.factor_prompt.as_messages(), model=cfg.decision_model, max_tokens=120)
print('factor', factor)
print('policy call...')
decision = chat_json(ctx.policy_prompt.as_messages(), model=cfg.decision_model, max_tokens=120)
print('decision', decision)
