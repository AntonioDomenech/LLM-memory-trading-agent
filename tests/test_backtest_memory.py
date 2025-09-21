import json

import pandas as pd
import pytest

from core.backtest import run_backtest
from core.memory import MemoryBank


def _dummy_bars():
    """Return a minimal OHLCV DataFrame for backtest tests."""

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
    """Write a temporary configuration JSON file for tests."""

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
    """Backtest should log factors and persist them into the memory bank."""

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
        """Simplified indicator calculator for deterministic tests."""

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
        """Return scripted LLM responses for deterministic assertions."""

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
    """No factor entries should be saved when retrieval is turned off."""

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
        """Return scripted LLM responses for deterministic assertions."""

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
    """Feedback summaries must be saved alongside factor memories."""

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
        """Simplified indicator calculator for deterministic tests."""

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
        """Return scripted LLM responses for deterministic assertions."""

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
    """Percent strings should be normalised to fractional exposures."""

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
        """Return scripted LLM responses for deterministic assertions."""

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


def test_run_backtest_allows_fractional_positions(tmp_path, monkeypatch):
    """Backtest must support fractional quantities when max_position is small."""

    memory_path = tmp_path / "bank_fractional.json"
    config_path = tmp_path / "config_fractional.json"
    _write_config(
        config_path,
        memory_path,
        {"k_shallow": 0, "k_intermediate": 0, "k_deep": 0},
        risk={
            "max_position": 10,
            "commission_per_trade": 0.0,
            "commission_per_share": 0.0,
            "slippage_bps": 0.0,
            "allow_short": False,
        },
    )

    cfg_raw = json.loads(config_path.read_text())
    cfg_raw["initial_cash"] = 100.0
    config_path.write_text(json.dumps(cfg_raw))

    bars = _dummy_bars()
    monkeypatch.setattr("core.backtest.get_daily_bars", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr("core.backtest.add_indicators", lambda df: df.copy())
    monkeypatch.setattr("core.backtest.plot_equity", lambda *args, **kwargs: "fig_eq")
    monkeypatch.setattr("core.backtest.plot_drawdown", lambda *args, **kwargs: "fig_dd")

    policy_sequence = [
        {"action": "BUY", "target_exposure": 0.5},
        {"action": "HOLD", "target_exposure": 0.5},
    ]
    calls = {"policy": 0}

    def fake_chat(messages, model=None, max_tokens=None):
        """Return scripted LLM responses for deterministic assertions."""

        idx = min(calls["policy"], len(policy_sequence) - 1)
        calls["policy"] += 1
        return dict(policy_sequence[idx])

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    equity_events = []

    def on_event(evt):
        """Collect emitted equity events for verification."""

        if evt.get("type") == "equity_point":
            equity_events.append(evt)

    result = run_backtest(config_path=str(config_path), on_event=on_event, event_rate=1)

    trades = result["trades_tail"]
    buy_trades = trades[trades["action"] == "BUY"]
    assert not buy_trades.empty
    assert buy_trades.iloc[0]["shares_delta"] == pytest.approx(0.5)

    assert equity_events, "expected at least one equity event"
    first_equity = equity_events[0]
    assert first_equity.get("position") == pytest.approx(0.5)

    assert calls["policy"] == len(bars)

def test_run_backtest_enforces_min_notional(tmp_path, monkeypatch):
    """Minimum notional buys below the floor should be cancelled."""

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
        """Return scripted LLM responses for deterministic assertions."""

        idx = min(calls["policy"], len(policy_sequence) - 1)
        resp = dict(policy_sequence[idx])
        calls["policy"] += 1
        return resp

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    events = []

    result = run_backtest(
        config_path=str(config_path), on_event=events.append, event_rate=1
    )

    trades = result["trades_tail"]
    assert trades.empty

    decision_norms = [
        evt.get("decision", {}).get("norm", "")
        for evt in events
        if evt.get("type") == "decision"
    ]
    assert any("min_trade_floor_cancel" in norm for norm in decision_norms)


def test_min_trade_floor_does_not_flip_short_cover(tmp_path, monkeypatch):
    """Covering a short below the floor must not flip the position long."""

    memory_path = tmp_path / "bank_short.json"
    config_path = tmp_path / "config_short.json"
    risk = {
        "allow_short": True,
        "min_trade_value": 500.0,
    }
    _write_config(
        config_path,
        memory_path,
        {"k_shallow": 0, "k_intermediate": 0, "k_deep": 0},
        risk=risk,
    )

    bars = _dummy_bars()
    monkeypatch.setattr("core.backtest.get_daily_bars", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr("core.backtest.add_indicators", lambda df: df.copy())
    monkeypatch.setattr("core.backtest.plot_equity", lambda *args, **kwargs: "fig_eq")
    monkeypatch.setattr("core.backtest.plot_drawdown", lambda *args, **kwargs: "fig_dd")

    policy_sequence = [
        {"action": "SELL", "target_exposure": -0.002},
        {"action": "BUY", "target_exposure": 0.0},
    ]

    def fake_chat(messages, model=None, max_tokens=None):
        """Return scripted LLM responses for deterministic assertions."""

        if not policy_sequence:
            pytest.fail("unexpected chat_json call")
        return dict(policy_sequence.pop(0))

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    events = []
    result = run_backtest(
        config_path=str(config_path), on_event=events.append, event_rate=1
    )

    equity_positions = [
        evt.get("position")
        for evt in events
        if evt.get("type") == "equity_point"
    ]
    assert equity_positions, "expected equity events"
    assert equity_positions[-1] == pytest.approx(-2.0)

    trades = result["trades_tail"]
    assert (trades["action"] == "BUY").sum() == 0

    decision_norms = [
        evt.get("decision", {}).get("norm", "")
        for evt in events
        if evt.get("type") == "decision"
    ]
    assert any("min_trade_floor_cancel" in norm for norm in decision_norms)


def test_min_trade_shares_rounding_and_sell_floor(tmp_path, monkeypatch):
    """When min_trade_shares is set the simulator rounds up or blocks trades."""

    memory_path = tmp_path / "bank_shares.json"
    config_path = tmp_path / "config_shares.json"
    _write_config(
        config_path,
        memory_path,
        {"k_shallow": 0, "k_intermediate": 0, "k_deep": 0},
        risk={
            "max_position": 100,
            "slippage_bps": 0.0,
            "commission_per_trade": 0.0,
            "commission_per_share": 0.0,
            "min_trade_value": 0.0,
            "min_trade_shares": 5,
            "allow_short": False,
        },
    )

    bars = _dummy_bars()
    monkeypatch.setattr("core.backtest.get_daily_bars", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr("core.backtest.add_indicators", lambda df: df.copy())
    monkeypatch.setattr("core.backtest.plot_equity", lambda *args, **kwargs: "fig_eq")
    monkeypatch.setattr("core.backtest.plot_drawdown", lambda *args, **kwargs: "fig_dd")

    policy_responses = [
        {"action": "BUY", "target_exposure": 0.002},
        {"action": "SELL", "target_exposure": 0.00303},
    ]

    def fake_chat(messages, model=None, max_tokens=None):
        content = messages[0].get("content", "")
        if "equity narrative analyst" in content:
            return {}
        idx = fake_chat.calls
        fake_chat.calls = min(fake_chat.calls + 1, len(policy_responses) - 1)
        return dict(policy_responses[idx])

    fake_chat.calls = 0
    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    result = run_backtest(config_path=str(config_path))
    trades = result["trades_tail"]
    assert len(trades) == 2
    first_trade, second_trade = trades.iloc[0], trades.iloc[1]
    assert first_trade["action"] == "BUY"
    assert first_trade["shares_delta"] == 5.0
    assert "min_share_floor:5" in (first_trade.get("note") or "")
    assert second_trade["action"] == "SELL"
    assert second_trade["shares_delta"] == -5.0
    assert "min_share_floor:5" in (second_trade.get("note") or "")


def test_min_trade_shares_blocks_when_floor_unreachable(tmp_path, monkeypatch):
    """Trades under the share floor should be skipped when the floor is unaffordable."""

    memory_path = tmp_path / "bank_shares_block.json"
    config_path = tmp_path / "config_shares_block.json"
    _write_config(
        config_path,
        memory_path,
        {"k_shallow": 0, "k_intermediate": 0, "k_deep": 0},
        risk={
            "max_position": 100,
            "slippage_bps": 0.0,
            "commission_per_trade": 0.0,
            "commission_per_share": 0.0,
            "min_trade_value": 0.0,
            "min_trade_shares": 5,
            "allow_short": False,
        },
    )

    cfg_raw = json.loads(config_path.read_text())
    cfg_raw["initial_cash"] = 450.0
    config_path.write_text(json.dumps(cfg_raw))

    bars = _dummy_bars()
    monkeypatch.setattr("core.backtest.get_daily_bars", lambda *args, **kwargs: bars.copy())
    monkeypatch.setattr("core.backtest.add_indicators", lambda df: df.copy())
    monkeypatch.setattr("core.backtest.plot_equity", lambda *args, **kwargs: "fig_eq")
    monkeypatch.setattr("core.backtest.plot_drawdown", lambda *args, **kwargs: "fig_dd")

    policy_responses = [
        {"action": "BUY", "target_exposure": 0.8889},
        {"action": "HOLD", "target_exposure": 0.8889},
    ]

    def fake_chat(messages, model=None, max_tokens=None):
        content = messages[0].get("content", "")
        if "equity narrative analyst" in content:
            return {}
        idx = fake_chat.calls
        fake_chat.calls = min(fake_chat.calls + 1, len(policy_responses) - 1)
        return dict(policy_responses[idx])

    fake_chat.calls = 0
    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    result = run_backtest(config_path=str(config_path))
    trades = result["trades_tail"]
    assert trades.empty


def test_benchmark_reinvests_periodic_contributions(tmp_path, monkeypatch):
    """Benchmark equity should deploy contributions instead of holding cash."""

    memory_path = tmp_path / "bank_contrib.json"
    config_path = tmp_path / "config_contrib.json"
    _write_config(
        config_path,
        memory_path,
        {"k_shallow": 0, "k_intermediate": 0, "k_deep": 0},
    )

    cfg_raw = json.loads(config_path.read_text())
    cfg_raw["initial_cash"] = 100.0
    cfg_raw["periodic_contribution"] = 100.0
    cfg_raw["contribution_frequency"] = "monthly"
    config_path.write_text(json.dumps(cfg_raw))

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

    captured = {}

    def fake_plot_equity(equity, benchmark):
        captured["bh"] = benchmark.copy()
        return "fig_eq"

    monkeypatch.setattr("core.backtest.plot_equity", fake_plot_equity)
    monkeypatch.setattr("core.backtest.plot_drawdown", lambda *args, **kwargs: "fig_dd")

    def fake_chat(messages, model=None, max_tokens=None):
        return {"action": "HOLD", "target_exposure": 0.0}

    monkeypatch.setattr("core.backtest.chat_json", fake_chat)

    result = run_backtest(config_path=str(config_path))

    bh_series = captured.get("bh")
    assert bh_series is not None, "benchmark series was not captured"

    first_price = bars["close"].iloc[0]
    last_price = bars["close"].iloc[-1]
    initial_shares = int(cfg_raw["initial_cash"] // first_price)
    leftover_cash = cfg_raw["initial_cash"] - initial_shares * first_price
    expected_shares = initial_shares + cfg_raw["periodic_contribution"] / first_price
    expected_bh_equity = expected_shares * last_price + leftover_cash

    assert bh_series.iloc[-1] == pytest.approx(expected_bh_equity)
    assert result["contributions"]["total_added"] == pytest.approx(cfg_raw["periodic_contribution"])
