import json

import pandas as pd

from core.backtest import run_backtest
from core.memory import MemoryBank


def _dummy_bars():
    dates = pd.to_datetime(["2023-01-02", "2023-01-03"])
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1100],
        }
    )


def _write_config(path, memory_path, retrieval):
    cfg = {
        "symbol": "AAPL",
        "train_start": "2022-01-01",
        "train_end": "2022-12-31",
        "test_start": "2023-01-02",
        "test_end": "2023-01-04",
        "news_source": "Off",
        "K_news_per_day": 0,
        "decision_model": "test-model",
        "embedding_model": "text-embedding-3-small",
        "memory_path": str(memory_path),
        "retrieval": retrieval,
    }
    path.write_text(json.dumps(cfg))


def test_run_backtest_records_factor_memory(tmp_path, monkeypatch):
    memory_path = tmp_path / "bank.json"
    config_path = tmp_path / "config.json"
    _write_config(
        config_path,
        memory_path,
        {"k_shallow": 2, "k_intermediate": 0, "k_deep": 0},
    )

    bars = _dummy_bars()

    monkeypatch.setattr("core.backtest.get_daily_bars", lambda *args, **kwargs: bars.copy())

    def fake_add_indicators(df):
        out = df.copy()
        out["atr"] = 1.0
        out["trend_up"] = 1
        out["sma200"] = out["close"]
        out["sma100"] = out["close"]
        return out

    monkeypatch.setattr("core.backtest.add_indicators", fake_add_indicators)
    monkeypatch.setattr("core.backtest.plot_equity", lambda *args, **kwargs: "fig_eq")
    monkeypatch.setattr("core.backtest.plot_drawdown", lambda *args, **kwargs: "fig_dd")

    factor_data = {
        "mood_score": 0.7,
        "narrative_bias": 0.1,
        "novelty": 0.3,
        "credibility": 0.6,
        "regime_alignment": 0.5,
        "confidence": 0.4,
    }
    policy_data = {
        "action": "BUY",
        "target_exposure": 0.5,
        "horizon_days": 5,
        "expected_return_bps": 25,
        "confidence": 0.4,
    }

    calls = {"factor": 0, "policy": 0}

    def fake_chat(messages, model=None, max_tokens=None):
        content = messages[0].get("content", "")
        if "equity narrative analyst" in content:
            calls["factor"] += 1
            return dict(factor_data)
        calls["policy"] += 1
        return dict(policy_data)

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    run_backtest(config_path=str(config_path))

    bank = MemoryBank(str(memory_path), emb_model="text-embedding-3-small")
    shallow_items = bank.layers.get("shallow", [])

    assert calls["factor"] == len(bars)
    assert calls["policy"] == len(bars)
    assert len(shallow_items) == len(bars)

    texts = [item.get("text", "") for item in shallow_items]
    assert any(text.startswith("[2023-01-02] mood=0.70 bias=+0.10 nov=0.30 cred=0.60") for text in texts)
    assert all("factor" in (item.get("meta") or {}) for item in shallow_items)


def test_run_backtest_skips_factor_memory_when_disabled(tmp_path, monkeypatch):
    memory_path = tmp_path / "bank_disabled.json"
    config_path = tmp_path / "config_disabled.json"
    _write_config(
        config_path,
        memory_path,
        {"k_shallow": 0, "k_intermediate": 0, "k_deep": 0},
    )

    bars = _dummy_bars()
    monkeypatch.setattr("core.backtest.get_daily_bars", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr("core.backtest.add_indicators", lambda df: df.copy())
    monkeypatch.setattr("core.backtest.plot_equity", lambda *args, **kwargs: "fig_eq")
    monkeypatch.setattr("core.backtest.plot_drawdown", lambda *args, **kwargs: "fig_dd")

    calls = {"factor": 0, "policy": 0}

    def fake_chat(messages, model=None, max_tokens=None):
        content = messages[0].get("content", "")
        if "equity narrative analyst" in content:
            calls["factor"] += 1
        else:
            calls["policy"] += 1
        return {"action": "HOLD", "target_exposure": 0.0}

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    run_backtest(config_path=str(config_path))

    bank = MemoryBank(str(memory_path), emb_model="text-embedding-3-small")
    shallow_items = bank.layers.get("shallow", [])

    assert calls["factor"] == 0
    assert calls["policy"] == len(bars)
    assert shallow_items == []
