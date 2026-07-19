from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.binary_regime_prospective_paper import (
    FIRST_AS_OF,
    ProspectivePaperError,
    _account_action_stream,
    _publish,
    _states_match_checkpoint,
    build_decision,
)


def _cash_signal_frame() -> pd.DataFrame:
    index = pd.bdate_range(end=FIRST_AS_OF, periods=160)
    aapl_open = np.full(len(index), 100.0)
    aapl_close = np.full(len(index), 100.0)
    aapl_close[-1] = 120.0
    broad = np.linspace(130.0, 90.0, len(index))
    return pd.DataFrame(
        {
            "aapl_open": aapl_open,
            "aapl_close": aapl_close,
            "aapl_adj_close": aapl_close,
            "spy_adj_close": broad,
            "qqq_adj_close": broad * 1.1,
        },
        index=index,
    )


def test_latest_unresolved_signal_becomes_a_prospective_cash_decision():
    frame = _cash_signal_frame()
    forecast, stream = _account_action_stream(frame)
    assert bool(stream.iloc[-1]["union_candidate_signal"]) is True
    assert bool(stream.iloc[-1]["account_union_cash_signal"]) is True
    assert bool(stream.iloc[-1]["risk_on"]) is False
    assert bool(stream.iloc[-1]["selector_cash_prediction"]) is True
    assert float(stream.iloc[-1]["target_exposure"]) == 0.0
    assert bool(forecast.iloc[-1]["stage_outcome_available"]) is False


def test_decision_is_saved_before_outcome_and_starts_from_common_aapl():
    created = datetime(2026, 7, 19, 20, 0, tzinfo=timezone.utc)
    decision, state = build_decision(_cash_signal_frame(), created_at=created)
    assert decision["action"] == "SELL_ALL_AAPL_TO_CASH_AT_NEXT_ACTUAL_OPEN"
    assert decision["target_exposure_next_open"] == 0.0
    assert decision["fill_price"] is None
    assert decision["exit_price"] is None
    assert decision["outcome_known"] is False
    assert state["common_starting_cash"] == 0.0
    assert state["common_starting_shares"] == pytest.approx(1000.0 / 120.0)
    assert state["no_pre_paper_return_scored"] is True


def test_first_decision_rejects_a_timestamp_after_the_fill_deadline():
    late = datetime(2026, 7, 20, 13, 30, tzinfo=timezone.utc)
    with pytest.raises(ProspectivePaperError, match="deadline"):
        build_decision(_cash_signal_frame(), created_at=late)


def test_state_match_allows_only_tiny_replay_roundoff():
    expected = {
        "schema_version": "s",
        "regime_order": ["risk_on", "not_risk_on"],
        "regime_feature_order": ["a"],
        "lesson_discount": 0.995,
        "minimum_effective_lessons": 12.0,
        "positive_mean_threshold": 0.001,
        "negative_mean_threshold": -0.001,
        "structural_default_cash": {"risk_on": False, "not_risk_on": True},
        "states": {
            "risk_on": {
                "n_raw": 2,
                "n_eff": 1.5,
                "weighted_label_sum": -0.1,
                "weighted_squared_label_sum": 0.01,
                "cash_selected": False,
            },
            "not_risk_on": {
                "n_raw": 3,
                "n_eff": 2.5,
                "weighted_label_sum": 0.2,
                "weighted_squared_label_sum": 0.02,
                "cash_selected": True,
            },
        },
    }
    observed = {
        **expected,
        "states": {
            key: dict(value) for key, value in expected["states"].items()
        },
    }
    observed["states"]["risk_on"]["weighted_label_sum"] += 1e-13
    assert _states_match_checkpoint(observed, expected) is True
    observed["states"]["risk_on"]["weighted_label_sum"] += 1e-6
    assert _states_match_checkpoint(observed, expected) is False


def test_publish_is_append_only(tmp_path):
    destination = tmp_path / "decisions"
    first = _publish(destination=destination, payloads={"decision.json": b"{}\n"})
    assert (first / "decision.json").read_bytes() == b"{}\n"
    with pytest.raises(ProspectivePaperError, match="already exists"):
        _publish(destination=destination, payloads={"decision.json": b"changed"})

