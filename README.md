# FinMem Narrative Pro (v4)

Run `pip install -r requirements.txt` then `streamlit run app.py`.
Set OPENAI_API_KEY and optionally NEWSAPI_KEY.

When tuning `config.json`, you can set `risk.min_trade_value` to enforce a minimum notional for BUY orders. The backtester will round small allocations up to at least that dollar amount (subject to max position limits and available cash).

The app now bundles [`readability-lxml`](https://github.com/buriy/python-readability) to provide a more robust HTML-to-text extraction fallback for news articles; make sure your environment can compile the underlying lxml dependency when installing.
