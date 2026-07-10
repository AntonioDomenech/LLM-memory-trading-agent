from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from agent_benchmark.direct_edge_features import build_market_sentiment_features
from agent_benchmark.downside_features import build_downside_price_features
from agent_benchmark.regime_consensus_features import (
    ALL_HEADS_READY_COLUMN,
    AAPL_STATE_FEATURE_COLUMNS,
    HEAD_FEATURE_COLUMNS,
    HEAD_NAMES,
    HEAD_ORIENTATIONS,
    HEAD_READINESS_COLUMNS,
    MARKET_STATE_FEATURE_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    REGIME_LABEL_COLUMNS,
    SENTIMENT_STATE_FEATURE_COLUMNS,
    build_one_session_regime_labels,
    build_regime_consensus_feature_label_frame,
    build_regime_consensus_features,
    entry_fill_year_mask,
    entry_fill_years,
    regime_head_readiness,
    strict_pre_fold_head_training_mask,
    strict_pre_fold_label_mask,
    strict_pre_fold_training_mask,
)


def _price_frame(periods: int = 340, *, start: str = "2000-01-03") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    step = np.arange(periods, dtype=float)
    aapl_close = 100.0 * np.exp(0.0004 * step + 0.015 * np.sin(step / 8.0))
    aapl_open = aapl_close * np.exp(-0.0015 * np.cos(step / 6.0))
    spy_close = 90.0 * np.exp(0.00025 * step + 0.008 * np.sin(step / 10.0))
    qqq_close = 80.0 * np.exp(0.00035 * step + 0.011 * np.cos(step / 9.0))
    return pd.DataFrame(
        {
            "date": dates,
            "aapl_open": aapl_open,
            "aapl_close": aapl_close,
            "aapl_adj_close": aapl_close,
            "spy_adj_close": spy_close,
            "qqq_adj_close": qqq_close,
        }
    )


def _context(dates: pd.DatetimeIndex) -> pd.DataFrame:
    step = np.arange(len(dates), dtype=float)
    # TNX is deliberately absent: it is outside the frozen feature contract.
    return pd.DataFrame(
        {
            "date": dates,
            "IWM": 70.0 * np.exp(0.0002 * step + 0.014 * np.sin(step / 12.0)),
            "^VIX": 20.0 * np.exp(0.08 * np.sin(step / 15.0) + 0.001 * step),
        }
    )


def _open_path_frame(opens: list[float], *, start: str = "2010-01-04") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(opens))
    values = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "aapl_open": values,
            "aapl_close": values,
            "aapl_adj_close": values,
            "spy_adj_close": 100.0 + np.arange(len(values), dtype=float),
            "qqq_adj_close": 80.0 + np.arange(len(values), dtype=float),
        }
    )


def test_frozen_head_schema_orientations_and_reused_values_are_exact() -> None:
    prices = _price_frame()
    context = _context(pd.DatetimeIndex(prices["date"]))
    features = build_regime_consensus_features(prices, context)
    price_reference = build_downside_price_features(prices)
    sentiment_reference = build_market_sentiment_features(prices, context)

    assert HEAD_NAMES == ("aapl_state", "market_state", "sentiment_state")
    assert HEAD_FEATURE_COLUMNS == (
        ("aapl_lr_5", "aapl_lr_20", "aapl_drawdown_60", "aapl_downside_rv_20"),
        ("qqq_lr_20", "qqq_lr_60", "iwm_lr_20", "spy_lr_20"),
        ("vix_z_252", "vix_lr_5", "qqq_minus_iwm_lr_20", "qqq_rv_20"),
    )
    assert HEAD_FEATURE_COLUMNS == (
        AAPL_STATE_FEATURE_COLUMNS,
        MARKET_STATE_FEATURE_COLUMNS,
        SENTIMENT_STATE_FEATURE_COLUMNS,
    )
    assert HEAD_ORIENTATIONS == (
        (-1, -1, -1, +1),
        (-1, -1, -1, -1),
        (+1, +1, +1, +1),
    )
    assert tuple(features.columns) == REGIME_FEATURE_COLUMNS
    assert len(REGIME_FEATURE_COLUMNS) == 12
    assert len(set(REGIME_FEATURE_COLUMNS)) == 12
    assert all(
        not column.startswith(("tnx_", "cot_", "news_", "raw_"))
        for column in REGIME_FEATURE_COLUMNS
    )
    assert "date" not in REGIME_FEATURE_COLUMNS

    for column in REGIME_FEATURE_COLUMNS:
        source = (
            price_reference[column]
            if column in price_reference.columns
            else sentiment_reference[column]
        )
        pdt.assert_series_equal(features[column], source, check_names=True)


def test_readiness_requires_all_four_per_head_and_all_three_to_trade() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    context = _context(dates)
    missing_vix_position = 290
    context.loc[missing_vix_position, "^VIX"] = np.nan

    combined = build_regime_consensus_feature_label_frame(prices, context)
    readiness = regime_head_readiness(combined)
    row = combined.iloc[missing_vix_position]

    assert tuple(readiness.columns) == (
        *HEAD_READINESS_COLUMNS,
        ALL_HEADS_READY_COLUMN,
    )
    assert bool(row["aapl_state_ready"])
    assert bool(row["market_state_ready"])
    assert not bool(row["sentiment_state_ready"])
    assert not bool(row[ALL_HEADS_READY_COLUMN])
    assert pd.isna(row["vix_z_252"])
    assert pd.isna(row["vix_lr_5"])

    # Warm-up is head-specific, but a decision still requires all three.
    assert bool(combined.iloc[100]["aapl_state_ready"])
    assert bool(combined.iloc[100]["market_state_ready"])
    assert not bool(combined.iloc[100]["sentiment_state_ready"])
    assert not bool(combined.iloc[100][ALL_HEADS_READY_COLUMN])
    assert bool(combined.iloc[280][ALL_HEADS_READY_COLUMN])


def test_missing_iwm_is_not_forward_filled_and_breaks_only_dependent_heads() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    context = _context(dates)
    missing_position = 290
    context.loc[missing_position, "IWM"] = np.nan

    combined = build_regime_consensus_feature_label_frame(prices, context)
    row = combined.iloc[missing_position]

    assert pd.isna(row["iwm_lr_20"])
    assert pd.isna(row["qqq_minus_iwm_lr_20"])
    assert bool(row["aapl_state_ready"])
    assert not bool(row["market_state_ready"])
    assert not bool(row["sentiment_state_ready"])
    assert not bool(row[ALL_HEADS_READY_COLUMN])


def test_one_session_target_uses_open_t_plus_1_to_t_plus_2_and_exact_costs() -> None:
    # Row 0 sells at 100 on t+1 and buys back after a 15 bp decline at t+2.
    exit_open = 100.0 * math.exp(-0.0015)
    prices = _open_path_frame([90.0, 100.0, exit_open, 101.0, 100.0])
    labels = build_one_session_regime_labels(prices)
    dates = pd.DatetimeIndex(prices["date"])
    row = labels.iloc[0]

    assert tuple(labels.columns) == REGIME_LABEL_COLUMNS
    assert row["label_entry_date"] == dates[1]
    assert row["label_maturity_date"] == dates[2]
    assert row["label_entry_adjusted_open"] == pytest.approx(100.0)
    assert row["label_exit_adjusted_open"] == pytest.approx(exit_open)
    assert row["aapl_forward_log_return_1"] == pytest.approx(-0.0015)
    assert row["cash_active_log_edge_5bps"] == pytest.approx(
        math.log((1.0 - 0.0005) / (1.0 + 0.0005)) + 0.0015
    )
    assert row["cash_active_log_edge_10bps"] == pytest.approx(
        math.log((1.0 - 0.0010) / (1.0 + 0.0010)) + 0.0015
    )
    assert row["cash_beats_long_5bps"] == 1
    assert row["cash_beats_long_10bps"] == 0


def test_penultimate_decision_preserves_entry_fill_but_not_immature_outcome() -> None:
    prices = _open_path_frame([90.0, 100.0, 99.0, 98.0])
    labels = build_one_session_regime_labels(prices)
    dates = pd.DatetimeIndex(prices["date"])
    penultimate = labels.iloc[-2]

    assert penultimate["label_entry_date"] == dates[-1]
    assert penultimate["label_entry_adjusted_open"] == pytest.approx(98.0)
    assert pd.isna(penultimate["label_maturity_date"])
    assert pd.isna(penultimate["label_exit_adjusted_open"])
    assert pd.isna(penultimate["cash_active_log_edge_5bps"])
    assert pd.isna(penultimate["cash_active_log_edge_10bps"])
    assert pd.isna(penultimate["cash_beats_long_10bps"])
    assert labels.iloc[-1][["label_entry_date", "label_entry_adjusted_open"]].isna().all()


def test_missing_required_future_open_invalidates_only_affected_outcomes() -> None:
    prices = _open_path_frame([90.0, 100.0, 99.0, 98.0, 97.0])
    prices.loc[2, "aapl_open"] = np.nan
    labels = build_one_session_regime_labels(prices)

    # Decision 0 has an entry but no t+2 exit.  Decision 1 has no t+1 entry.
    assert not pd.isna(labels.iloc[0]["label_entry_date"])
    assert pd.isna(labels.iloc[0]["label_maturity_date"])
    assert pd.isna(labels.iloc[0]["cash_active_log_edge_10bps"])
    assert pd.isna(labels.iloc[1]["label_entry_date"])
    assert pd.isna(labels.iloc[1]["cash_active_log_edge_10bps"])


def test_strict_masks_exclude_maturity_on_boundary_and_train_each_head_separately() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    context = _context(dates)
    missing_vix_position = 270
    context.loc[missing_vix_position, "^VIX"] = np.nan
    combined = build_regime_consensus_feature_label_frame(prices, context)
    boundary_position = 300
    boundary = dates[boundary_position]

    label_mask = strict_pre_fold_label_mask(
        combined,
        fold_first_decision_date=boundary,
    )
    aapl_mask = strict_pre_fold_head_training_mask(
        combined,
        fold_first_decision_date=boundary,
        head_name="aapl_state",
    )
    market_mask = strict_pre_fold_training_mask(
        combined,
        fold_first_decision_date=boundary,
        head_name="market_state",
    )
    sentiment_mask = strict_pre_fold_head_training_mask(
        combined,
        fold_first_decision_date=boundary,
        head_name="sentiment_state",
    )

    # t+2 at the boundary is not known before the frozen fold starts.
    assert not label_mask.iloc[boundary_position - 2]
    assert label_mask.iloc[boundary_position - 3]
    assert not label_mask.iloc[boundary_position:].any()

    # Missing VIX does not remove a mature case from the AAPL/market heads or
    # from the predictive baseline, but it splits the sentiment-head sequence.
    assert label_mask.iloc[missing_vix_position]
    assert aapl_mask.iloc[missing_vix_position]
    assert market_mask.iloc[missing_vix_position]
    assert not sentiment_mask.iloc[missing_vix_position]


def test_training_mask_rejects_tampered_head_readiness() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    combined = build_regime_consensus_feature_label_frame(prices, _context(dates))
    combined.loc[dates[280], "aapl_state_ready"] = False

    with pytest.raises(ValueError, match="inconsistent"):
        strict_pre_fold_head_training_mask(
            combined,
            fold_first_decision_date=dates[300],
            head_name="aapl_state",
        )
    with pytest.raises(ValueError, match="head_name"):
        strict_pre_fold_head_training_mask(
            combined,
            fold_first_decision_date=dates[300],
            head_name="unknown",
        )


def test_entry_fill_year_helpers_use_fill_not_decision_calendar_year() -> None:
    prices = _open_path_frame(
        [90.0, 91.0, 92.0, 93.0, 94.0],
        start="2020-12-30",
    )
    labels = build_one_session_regime_labels(prices)
    years = entry_fill_years(labels)
    selected = entry_fill_year_mask(labels, first_year=2021, last_year=2021)

    assert years.iloc[0] == 2020
    assert years.iloc[1] == 2021
    assert years.iloc[-2] == 2021  # entry exists even though outcome is immature
    assert pd.isna(years.iloc[-1])
    assert not selected.iloc[0]
    assert selected.iloc[1]
    assert selected.iloc[-2]
    assert not selected.iloc[-1]
    with pytest.raises(ValueError, match="last_year"):
        entry_fill_year_mask(labels, first_year=2021, last_year=2020)


def test_completed_close_features_do_not_change_when_future_inputs_change() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    context = _context(dates)
    decision_position = 280
    baseline = build_regime_consensus_features(prices, context)

    changed_prices = prices.copy()
    changed_context = context.copy()
    changed_prices.loc[decision_position + 1 :, "aapl_adj_close"] *= 11.0
    changed_prices.loc[decision_position + 1 :, "spy_adj_close"] *= 7.0
    changed_prices.loc[decision_position + 1 :, "qqq_adj_close"] *= 5.0
    changed_context.loc[decision_position + 1 :, ["IWM", "^VIX"]] *= 13.0
    changed = build_regime_consensus_features(changed_prices, changed_context)

    pdt.assert_series_equal(
        baseline.iloc[decision_position],
        changed.iloc[decision_position],
        check_names=False,
    )
