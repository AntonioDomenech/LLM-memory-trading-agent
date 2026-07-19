from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.binary_regime_prospective_paper import (
    FIRST_AS_OF,
    ProspectivePaperError,
    _account_action_stream,
    _load_audited_input,
    _prefix_price_compatibility,
    _publish,
    _sealed_action_stream_sha256,
    _states_match_checkpoint,
    build_decision,
    settle_decision_outcome,
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
            "aapl_adj_open": aapl_open,
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
    assert state["common_starting_adjusted_units"] == pytest.approx(1000.0 / 120.0)
    assert state["no_pre_paper_return_scored"] is True


def test_preserved_input_reproduces_the_sealed_action_fingerprint():
    root = Path(__file__).resolve().parents[1]
    _, stream = _account_action_stream(_load_audited_input(root))
    assert _sealed_action_stream_sha256(stream) == (
        "f77c68462ced8158bca6bf5a0aec95b4161bacd811075048e7918ba1de4d15ed"
    )


def test_price_compatibility_accepts_rounding_but_rejects_trading_scale_change():
    audited = _cash_signal_frame()
    fresh = audited.copy()
    fresh.loc[fresh.index[0], "aapl_open"] += 1e-13
    fresh.loc[fresh.index[0], "qqq_adj_close"] *= 1.0 + 1e-6
    result = _prefix_price_compatibility(fresh=fresh, audited=audited)
    assert result["accepted_as_numerical_vendor_revision"] is True

    fresh.loc[fresh.index[0], "qqq_adj_close"] *= 1.0 + 2e-4
    with pytest.raises(ProspectivePaperError, match="trading scale"):
        _prefix_price_compatibility(fresh=fresh, audited=audited)


def test_cash_outcome_rebases_adjusted_units_and_applies_both_cost_legs():
    created = datetime(2026, 7, 19, 20, 0, tzinfo=timezone.utc)
    decision, state = build_decision(_cash_signal_frame(), created_at=created)
    observed = _cash_signal_frame().copy()
    price_columns = [
        "aapl_open",
        "aapl_close",
        "aapl_adj_open",
        "aapl_adj_close",
    ]
    observed.loc[:, price_columns] /= 2.0
    future = pd.DataFrame(
        {
            "aapl_open": [65.0, 64.0],
            "aapl_close": [64.0, 64.5],
            "aapl_adj_open": [65.0, 64.0],
            "aapl_adj_close": [64.0, 64.5],
            "spy_adj_close": [89.0, 88.0],
            "qqq_adj_close": [97.9, 96.8],
        },
        index=pd.to_datetime(["2026-07-20", "2026-07-21"]),
    )
    observed = pd.concat([observed, future])

    outcome, updated = settle_decision_outcome(
        decision=decision,
        state=state,
        observed=observed,
        cost_bps=5.0,
    )

    assert outcome["entry_open_date"] == "2026-07-20"
    assert outcome["exit_open_date"] == "2026-07-21"
    assert outcome["changing_legs"] == 2
    assert outcome["estimated_costs"] > 0.0
    assert outcome["benchmark_equity"] == pytest.approx((1000.0 / 60.0) * 64.0)
    assert outcome["strategy_equity"] > outcome["benchmark_equity"]
    ledger = updated["cost_ledgers"]["5bps"]
    assert ledger["strategy_cash"] == 0.0
    assert ledger["strategy_adjusted_units"] > 0.0
    assert ledger["changing_legs"] == 2


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
