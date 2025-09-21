# FinMem Narrative Pro (v4)

Run `pip install -r requirements.txt` then `streamlit run app.py`.
Set OPENAI_API_KEY and optionally NEWSAPI_KEY.

When tuning `config.json`, you can set `risk.min_trade_value` to enforce a minimum notional for BUY orders and `risk.min_trade_shares` to require a minimum lot size (set it to 0 to allow fractional trades). The notional floor is evaluated first, potentially rounding buys up to the corresponding share count before cash checks occur. Afterwards the share floor applies symmetrically to buys and sells: if the post-notional size is below the threshold the engine either rounds the order up to the lot (when capacity and cash allow) or cancels it to respect both floors.

The app now bundles [`readability-lxml`](https://github.com/buriy/python-readability) to provide a more robust HTML-to-text extraction fallback for news articles; make sure your environment can compile the underlying lxml dependency when installing.
