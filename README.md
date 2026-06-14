# LLM Memory Trading Agent

## V2 local benchmark

The new benchmark is a local FastAPI + React app where the model is the
portfolio manager and the simulator only applies market mechanics.

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

Run the backend:

```bash
python run_backend.py
```

Run the frontend in another terminal:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`.

Local secrets are saved to `data/local_config.json`, which is ignored by git.
The frontend asks for OpenAI, SEC, FRED, Marketaux, Finnhub, and NewsAPI keys as
needed. Only the OpenAI key is required for real model runs.

See `docs/v2_benchmark_plan.md` for the information bundle, data-source plan,
and benchmark rules.

## Historical warehouse

The 2000-2025 local warehouse uses DuckDB plus Parquet files under
`data/warehouse/`.

Initialize the calendar and Balanced 50 symbol universe:

```bash
python -m agent_benchmark.warehouse bootstrap
```

Download the cheap core datasets:

```bash
python -m agent_benchmark.warehouse download-prices
python -m agent_benchmark.warehouse download-sec
python -m agent_benchmark.warehouse download-macro
python -m agent_benchmark.warehouse validate
```

Download historical GDELT news/event metadata resumably. The no-cost default
path uses raw daily GDELT Events archives for the full 2000-2025 range:

```bash
python -m agent_benchmark.warehouse download-gdelt-events --start 2000-01-01 --end 2025-12-31 --retry-until-success --workers 6 --import-batch-size 32
```

The DOC API downloader is also available for narrower experiments, but it can
rate-limit broad historical company queries:

```bash
python -m agent_benchmark.warehouse download-news
```

For a quick smoke download:

```bash
python -m agent_benchmark.warehouse download-news --symbols AAPL --start 2025-01-01 --end 2025-01-31 --max-months 1
```

Inspect status:

```bash
python -m agent_benchmark.warehouse status
```

## Legacy Streamlit prototype

Run `pip install -r requirements.txt` then `streamlit run app.py`.
Set OPENAI_API_KEY and optionally NEWSAPI_KEY.

When tuning `config.json`, you can set `risk.min_trade_value` to enforce a minimum notional for BUY orders and `risk.min_trade_shares` to require a minimum lot size (set it to 0 to allow fractional trades). The notional floor is evaluated first, potentially rounding buys up to the corresponding share count before cash checks occur. Afterwards the share floor applies symmetrically to buys and sells: if the post-notional size is below the threshold the engine either rounds the order up to the lot (when capacity and cash allow) or cancels it to respect both floors.

The app now bundles [`readability-lxml`](https://github.com/buriy/python-readability) to provide a more robust HTML-to-text extraction fallback for news articles; make sure your environment can compile the underlying lxml dependency when installing.
