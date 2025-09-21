import json

from core import news_store


def test_download_range_full_content_filters_missing(tmp_path):
    base_dir = tmp_path / "news"
    symbol = "TEST"
    day = "2024-01-02"

    existing_articles = [
        {
            "url": "http://existing-good",
            "title": "Existing Good",
            "content": "g" * 150,
        },
        {
            "url": "http://existing-bad",
            "title": "Existing Bad",
            "content": "short",
        },
    ]

    news_store.save_local_day(
        symbol,
        day,
        existing_articles,
        provider="local",
        reason="seed",
        base_dir=str(base_dir),
    )

    fetched_articles = [
        {
            "url": "http://new-good",
            "title": "New Good",
            "content": "n" * 150,
        },
        {
            "url": "http://new-bad",
            "title": "New Bad",
            "content": " ",
        },
    ]

    def fake_fetch(symbol_arg, day_arg, k_arg):
        assert symbol_arg == symbol
        assert day_arg == day
        assert k_arg == 3
        return fetched_articles, "mock:provider"

    stats = news_store.download_range(
        symbol,
        day,
        day,
        K=3,
        base_dir=str(base_dir),
        fetch_fn=fake_fetch,
        full_content=True,
    )

    saved_path = news_store.local_day_path(symbol, day, base_dir=str(base_dir))
    with open(saved_path, "r", encoding="utf-8") as handle:
        saved_payload = json.load(handle)

    saved_articles = saved_payload.get("articles", [])

    assert stats["saved"] == 1
    assert saved_articles, "expected at least one article to be saved"
    for article in saved_articles:
        content = article.get("content", "")
        assert isinstance(content, str) and content.strip()
        assert len(content.strip()) > 0

    urls = {article.get("url") for article in saved_articles}
    assert "http://existing-bad" not in urls
    assert "http://new-bad" not in urls
