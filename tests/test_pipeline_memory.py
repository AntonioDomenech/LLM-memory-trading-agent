import json

from core.config import Config, RetrievalCfg
from core.memory import MemoryBank
from core.pipeline import prepare_daily_context


def _base_price_row():
    return {"close": 100.0, "atr": 1.5, "trend_up": 1}


def test_prepare_daily_context_memory_section_empty(tmp_path):
    cfg = Config()
    cfg.K_news_per_day = 0
    cfg.news_source = "Off"
    cfg.memory_path = str(tmp_path / "bank.json")
    cfg.retrieval = RetrievalCfg(k_shallow=0, k_intermediate=0, k_deep=0)

    ctx = prepare_daily_context(cfg, "2024-01-02", _base_price_row())

    assert ctx.memory_highlights == []
    assert ctx.memory_retrieval == {"shallow": [], "intermediate": [], "deep": []}

    factor_payload = json.loads(ctx.factor_prompt.user["content"])
    assert "memory_highlights" in factor_payload
    assert factor_payload["memory_highlights"] == []

    policy_payload = json.loads(ctx.policy_prompt.user["content"])
    assert policy_payload["memory_highlights"] == []


def test_prepare_daily_context_with_retrieved_memory(tmp_path):
    cfg = Config()
    cfg.K_news_per_day = 0
    cfg.news_source = "Off"
    cfg.memory_path = str(tmp_path / "bank.json")
    cfg.retrieval = RetrievalCfg(k_shallow=2, k_intermediate=0, k_deep=0)

    bank = MemoryBank(cfg.memory_path, emb_model=cfg.embedding_model)
    bank.add_item(
        "shallow",
        "AAPL narrative: bullish momentum noted",
        {"date": "2024-01-01", "tag": "summary"},
        base_importance=12.0,
        seen_date="2024-01-01",
    )

    ctx = prepare_daily_context(
        cfg,
        "2024-01-02",
        _base_price_row(),
        memory_bank=bank,
    )

    assert ctx.memory_highlights, "Expected retrieved highlights"
    assert ctx.memory_retrieval["shallow"], "Shallow layer should include the stored item"
    assert ctx.memory_retrieval["intermediate"] == []
    assert ctx.memory_retrieval["deep"] == []

    highlight = ctx.memory_highlights[0]
    assert highlight["text"].startswith("AAPL narrative")
    assert highlight["meta"].get("tag") == "summary"
    assert "embedding" not in highlight

    factor_payload = json.loads(ctx.factor_prompt.user["content"])
    assert factor_payload["memory_highlights"][0]["text"].startswith("AAPL narrative")

    policy_payload = json.loads(ctx.policy_prompt.user["content"])
    assert policy_payload["memory_highlights"][0]["text"].startswith("AAPL narrative")
