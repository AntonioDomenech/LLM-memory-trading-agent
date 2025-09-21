import json
import math

import pandas as pd
import pytest

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


def _write_config(path, memory_path, retrieval, risk=None):
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
        "initial_cash": 100000.0,
        "retrieval": retrieval,
    }
    if risk is not None:
        cfg["risk"] = risk
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
    expected_items = len(bars) + max(len(bars) - 1, 0)
    assert len(shallow_items) == expected_items

    texts = [item.get("text", "") for item in shallow_items]
    assert any(text.startswith("[2023-01-02] mood=0.70 bias=+0.10 nov=0.30 cred=0.60") for text in texts)
    feedback_texts = [text for text in texts if "→" in text]
    assert len(feedback_texts) == max(len(bars) - 1, 0)
    factor_items = [item for item in shallow_items if "factor" in (item.get("meta") or {})]
    assert len(factor_items) == len(bars)


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


def test_run_backtest_writes_feedback_memory(tmp_path, monkeypatch):
    memory_path = tmp_path / "bank_feedback.json"
    config_path = tmp_path / "config_feedback.json"
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

    policy_sequence = [
        {"action": "BUY", "target_exposure": 0.2},
        {"action": "HOLD", "target_exposure": 0.2},
    ]

    calls = {"factor": 0, "policy": 0}

    def fake_chat(messages, model=None, max_tokens=None):
        content = messages[0].get("content", "")
        if "equity narrative analyst" in content:
            calls["factor"] += 1
            return dict(factor_data)
        idx = min(calls["policy"], len(policy_sequence) - 1)
        resp = dict(policy_sequence[idx])
        calls["policy"] += 1
        return resp

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    run_backtest(config_path=str(config_path))

    bank = MemoryBank(str(memory_path), emb_model="text-embedding-3-small")
    shallow_items = bank.layers.get("shallow", [])

    feedback_items = [item for item in shallow_items if "→" in (item.get("text") or "")]

    # Two factor summaries + one feedback entry expected
    assert len(shallow_items) == 3
    assert len(feedback_items) == 1

    feedback = feedback_items[0]
    meta = feedback.get("meta") or {}

    assert feedback["text"].startswith("2023-01-02: BUY 200")
    assert meta.get("action") == "BUY"
    assert meta.get("decision_date") == "2023-01-02"
    assert meta.get("observed_on") == "2023-01-03"
    assert meta.get("target_exposure") == pytest.approx(0.2)
    assert meta.get("position") == 200
    assert meta.get("realized_return") == pytest.approx(0.01, rel=1e-6)


def test_run_backtest_parses_percent_exposure(tmp_path, monkeypatch):
    memory_path = tmp_path / "bank_percent.json"
    config_path = tmp_path / "config_percent.json"
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

    policy_sequence = [
        {"action": "BUY", "target_exposure": "75%"},
        {"action": "HOLD", "target_exposure": "75%"},
    ]

    calls = {"policy": 0}

    def fake_chat(messages, model=None, max_tokens=None):
        idx = min(calls["policy"], len(policy_sequence) - 1)
        calls["policy"] += 1
        return dict(policy_sequence[idx])

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    result = run_backtest(config_path=str(config_path))

    trades_tail = result["trades_tail"]
    assert not trades_tail.empty

    first_trade = trades_tail.iloc[0]
    assert first_trade["action"] == "BUY"
    assert first_trade["shares_delta"] == 750
    assert str(first_trade.get("note", "")).startswith("percent_to_frac:0.75")

    assert calls["policy"] == len(bars)


def test_run_backtest_enforces_min_notional(tmp_path, monkeypatch):
    memory_path = tmp_path / "bank_floor.json"
    config_path = tmp_path / "config_floor.json"
    risk = {
        "max_position": 1000,
        "commission_per_trade": 0.0,
        "commission_per_share": 0.0,
        "slippage_bps": 0.0,
        "min_trade_value": 500.0,
        "allow_short": False,
    }
    _write_config(
        config_path,
        memory_path,
        {"k_shallow": 0, "k_intermediate": 0, "k_deep": 0},
        risk=risk,
    )

    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
            "open": [20.0, 21.0],
            "high": [21.0, 22.0],
            "low": [19.0, 20.0],
            "close": [20.0, 21.0],
            "volume": [1000, 1100],
        }
    )

    monkeypatch.setattr("core.backtest.get_daily_bars", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr("core.backtest.add_indicators", lambda df: df.copy())
    monkeypatch.setattr("core.backtest.plot_equity", lambda *args, **kwargs: "fig_eq")
    monkeypatch.setattr("core.backtest.plot_drawdown", lambda *args, **kwargs: "fig_dd")

    policy_sequence = [
        {"action": "BUY", "target_exposure": 0.001},
        {"action": "HOLD", "target_exposure": 0.0},
    ]
    calls = {"policy": 0}

    def fake_chat(messages, model=None, max_tokens=None):
        idx = min(calls["policy"], len(policy_sequence) - 1)
        resp = dict(policy_sequence[idx])
        calls["policy"] += 1
        return resp

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    result = run_backtest(config_path=str(config_path))

    trades = result["trades_tail"]
    buy_trades = trades[trades["action"] == "BUY"]
    assert not buy_trades.empty

    expected_floor = math.ceil(risk["min_trade_value"] / bars.iloc[0]["close"])
    last_buy = buy_trades.iloc[-1]
    assert int(last_buy["shares_delta"]) >= expected_floor
    assert "min_trade_floor" in (last_buy.get("note") or "")
