# FinMem Narrative Pro (v4)

Run `pip install -r requirements.txt` then `streamlit run app.py`.
Set OPENAI_API_KEY and optionally NEWSAPI_KEY.

The app now bundles [`readability-lxml`](https://github.com/buriy/python-readability) to provide a more robust HTML-to-text extraction fallback for news articles; make sure your environment can compile the underlying lxml dependency when installing.
