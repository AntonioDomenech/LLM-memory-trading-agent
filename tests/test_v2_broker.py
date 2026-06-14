import pytest

from agent_benchmark.broker import execute_decision
from agent_benchmark.schemas import BenchmarkConfig, PortfolioState


def test_broker_logs_cash_limit_without_hiding_decision():
    config = BenchmarkConfig(symbol="AAPL", allow_short=False, max_leverage=1.0, slippage_bps=0.0)
    portfolio = PortfolioState(cash=1000.0, position_shares=0.0, equity=1000.0)
    decision = {
        "action": "BUY",
        "target_exposure": 2.0,
        "confidence": 0.8,
    }

    new_portfolio, execution = execute_decision(portfolio, decision, market_close=100.0, config=config)

    event_types = [event["type"] for event in execution["events"]]
    assert "leverage_capped" in event_types
    assert new_portfolio.position_shares == pytest.approx(10.0)
    assert execution["trade"]["side"] == "BUY"


def test_broker_blocks_short_when_disabled():
    config = BenchmarkConfig(symbol="AAPL", allow_short=False, max_leverage=1.0, slippage_bps=0.0)
    portfolio = PortfolioState(cash=1000.0, position_shares=0.0, equity=1000.0)
    decision = {
        "action": "SELL",
        "target_exposure": -0.5,
        "confidence": 0.8,
    }

    new_portfolio, execution = execute_decision(portfolio, decision, market_close=100.0, config=config)

    event_types = [event["type"] for event in execution["events"]]
    assert "short_blocked" in event_types
    assert new_portfolio.position_shares == 0.0
    assert execution["trade"] is None


def test_invalid_action_is_model_failure():
    config = BenchmarkConfig(symbol="AAPL", allow_short=True, max_leverage=1.0)
    portfolio = PortfolioState(cash=1000.0, position_shares=0.0, equity=1000.0)
    decision = {"action": "WAIT", "target_exposure": 0.0}

    _, execution = execute_decision(portfolio, decision, market_close=100.0, config=config)

    assert execution["model_failure"] is True
    assert execution["events"][0]["type"] == "invalid_action"
