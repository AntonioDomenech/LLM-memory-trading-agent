from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.rare_loss_features import (
    CASH_EDGE_SLIPPAGE_BPS,
    CORE_FEATURE_COLUMNS,
    RARE_LOSS_FEATURE_COLUMNS,
    RARE_LOSS_LABEL_COLUMNS,
    REGRESSION_EDGE_CLIP,
    SENTIMENT_FEATURE_COLUMNS,
    SEVERE_SIMPLE_RETURN_THRESHOLD,
    build_one_session_rare_loss_labels,
    build_rare_loss_feature_label_frame,
    build_rare_loss_features,
    entry_fill_year_mask,
    rare_loss_feature_readiness,
    strict_pre_fold_training_mask,
)


def _market_frames(rows: int = 340) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2010-01-04", periods=rows)
    position = np.arange(rows, dtype=float)
    aapl_return = 0.0004 + 0.0015 * np.sin(position / 9.0)
    qqq_return = 0.0003 + 0.0010 * np.sin(position / 13.0)
    spy_return = 0.0002 + 0.0008 * np.cos(position / 15.0)
    iwm_return = 0.00025 + 0.0012 * np.cos(position / 11.0)
    aapl = 50.0 * np.exp(np.cumsum(aapl_return))
    prices = pd.DataFrame(
        {
            "aapl_open": aapl * np.exp(0.0005 * np.sin(position / 5.0)),
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": 100.0 * np.exp(np.cumsum(spy_return)),
            "qqq_adj_close": 80.0 * np.exp(np.cumsum(qqq_return)),
        },
        index=index,
    )
    context = pd.DataFrame(
        {
            "iwm_adj_close": 70.0 * np.exp(np.cumsum(iwm_return)),
            "vix_close": 20.0 + 3.0 * np.sin(position / 17.0),
            "tnx_close": 25.0 + position * 0.01,
        },
        index=index,
    )
    return prices, context


def test_frozen_feature_schema_and_completed_close_formulas_are_exact() -> None:
    prices, context = _market_frames()
    features = build_rare_loss_features(prices, context)
    assert CORE_FEATURE_COLUMNS == (
        "aapl_lr_1",
        "aapl_lr_5",
        "aapl_lr_20",
        "aapl_intraday_lr",
        "aapl_gap_lr",
        "aapl_downside_rv_20",
        "aapl_drawdown_60",
        "aapl_drawdown_252",
        "aapl_minus_qqq_lr_5",
        "aapl_minus_qqq_lr_20",
        "qqq_lr_5",
        "qqq_lr_20",
        "qqq_lr_60",
        "qqq_rv_20",
        "qqq_minus_spy_lr_20",
    )
    assert SENTIMENT_FEATURE_COLUMNS == (
        "iwm_lr_5",
        "iwm_lr_20",
        "qqq_minus_iwm_lr_20",
        "vix_z_252",
        "vix_lr_5",
    )
    assert tuple(features.columns) == RARE_LOSS_FEATURE_COLUMNS
    assert len(RARE_LOSS_FEATURE_COLUMNS) == 20
    assert not any("tnx" in column for column in features.columns)

    row = 300
    date = prices.index[row]
    aapl = prices["aapl_adj_close"]
    expected_lr_5 = math.log(float(aapl.iloc[row] / aapl.iloc[row - 5]))
    expected_drawdown_60 = float(
        aapl.iloc[row] / aapl.iloc[row - 59 : row + 1].max() - 1.0
    )
    expected_gap = math.log(
        float(prices["aapl_open"].iloc[row] / aapl.iloc[row - 1])
    )
    assert features.loc[date, "aapl_lr_5"] == pytest.approx(expected_lr_5)
    assert features.loc[date, "aapl_drawdown_60"] == pytest.approx(
        expected_drawdown_60
    )
    assert features.loc[date, "aapl_gap_lr"] == pytest.approx(
        expected_gap
    )
    assert features.loc[date, "vix_lr_5"] == pytest.approx(
        math.log(
            float(
                context.loc[date, "vix_close"]
                / context.iloc[row - 5]["vix_close"]
            )
        )
    )
    assert features.loc[date].notna().all()

    changed_tnx = context.copy()
    changed_tnx["tnx_close"] *= 1000.0
    pd.testing.assert_frame_equal(
        features,
        build_rare_loss_features(prices, changed_tnx),
    )


def test_one_session_targets_use_t_plus_1_to_t_plus_2_and_clip_only_regression() -> None:
    prices, _ = _market_frames()
    # With close == adjusted close, canonical adjusted open equals raw open.
    loss_position = 280
    gain_position = 290
    threshold_position = 300
    prices.iloc[loss_position + 1, prices.columns.get_loc("aapl_open")] = 100.0
    prices.iloc[loss_position + 2, prices.columns.get_loc("aapl_open")] = 90.0
    prices.iloc[gain_position + 1, prices.columns.get_loc("aapl_open")] = 100.0
    prices.iloc[gain_position + 2, prices.columns.get_loc("aapl_open")] = 110.0
    prices.iloc[threshold_position + 1, prices.columns.get_loc("aapl_open")] = 100.0
    prices.iloc[threshold_position + 2, prices.columns.get_loc("aapl_open")] = 97.5

    labels = build_one_session_rare_loss_labels(prices)
    assert tuple(labels.columns) == RARE_LOSS_LABEL_COLUMNS
    cost = math.log1p(-CASH_EDGE_SLIPPAGE_BPS / 10_000.0) - math.log1p(
        CASH_EDGE_SLIPPAGE_BPS / 10_000.0
    )

    loss_date = prices.index[loss_position]
    expected_loss_edge = cost - math.log(0.9)
    assert labels.loc[loss_date, "label_entry_date"] == prices.index[
        loss_position + 1
    ]
    assert labels.loc[loss_date, "label_maturity_date"] == prices.index[
        loss_position + 2
    ]
    assert labels.loc[loss_date, "aapl_forward_simple_return_1"] == pytest.approx(
        -0.10
    )
    assert labels.loc[loss_date, "severe_loss_label_1session"] == 1
    assert labels.loc[loss_date, "cash_beats_long_10bps"] == 1
    assert labels.loc[loss_date, "cash_active_log_edge_10bps"] == pytest.approx(
        expected_loss_edge
    )
    assert labels.loc[
        loss_date, "cash_active_log_edge_10bps_clipped"
    ] == pytest.approx(REGRESSION_EDGE_CLIP)

    gain_date = prices.index[gain_position]
    expected_gain_edge = cost - math.log(1.1)
    assert labels.loc[gain_date, "aapl_forward_simple_return_1"] == pytest.approx(
        0.10
    )
    assert labels.loc[gain_date, "severe_loss_label_1session"] == 0
    assert labels.loc[gain_date, "cash_beats_long_10bps"] == 0
    assert labels.loc[gain_date, "cash_active_log_edge_10bps"] == pytest.approx(
        expected_gain_edge
    )
    assert labels.loc[
        gain_date, "cash_active_log_edge_10bps_clipped"
    ] == pytest.approx(-REGRESSION_EDGE_CLIP)

    threshold_date = prices.index[threshold_position]
    assert labels.loc[
        threshold_date, "aapl_forward_simple_return_1"
    ] == pytest.approx(SEVERE_SIMPLE_RETURN_THRESHOLD)
    assert labels.loc[threshold_date, "severe_loss_label_1session"] == 1
    assert pd.isna(labels.iloc[-2]["severe_loss_label_1session"])
    assert pd.isna(labels.iloc[-1]["cash_active_log_edge_10bps"])


def test_features_are_causal_while_future_opens_change_the_target() -> None:
    prices, context = _market_frames()
    position = 300
    date = prices.index[position]
    original_features = build_rare_loss_features(prices, context)
    original_labels = build_one_session_rare_loss_labels(prices)

    changed_prices = prices.copy()
    changed_context = context.copy()
    changed_prices.iloc[position + 1, :] *= 1.10
    changed_prices.iloc[position + 2, :] *= 1.25
    for future in (position + 1, position + 2):
        changed_context.iloc[future, :] *= 1.50
    changed_features = build_rare_loss_features(changed_prices, changed_context)
    changed_labels = build_one_session_rare_loss_labels(changed_prices)

    pd.testing.assert_series_equal(
        original_features.loc[date],
        changed_features.loc[date],
    )
    assert original_labels.loc[date, "cash_active_log_edge_10bps"] != pytest.approx(
        changed_labels.loc[date, "cash_active_log_edge_10bps"]
    )


def test_exact_context_alignment_has_no_fill_and_controls_readiness() -> None:
    prices, context = _market_frames()
    missing_date = prices.index[300]
    context.loc[missing_date, ["iwm_adj_close", "vix_close"]] = np.nan
    features = build_rare_loss_features(prices, context)
    assert features.loc[missing_date, list(SENTIMENT_FEATURE_COLUMNS)].isna().all()
    ready = rare_loss_feature_readiness(features)
    assert ready.loc[missing_date] is np.False_
    assert len(features) == len(prices)
    assert features.index.equals(prices.index)


def test_combined_frame_and_strict_fold_mask_require_mature_complete_rows() -> None:
    prices, context = _market_frames()
    missing_position = 280
    frame = build_rare_loss_feature_label_frame(prices, context)
    frame.loc[prices.index[missing_position], "vix_z_252"] = np.nan
    frame["features_ready"] = rare_loss_feature_readiness(frame)
    frame["training_ready"] = frame["features_ready"] & frame["label_available"]
    assert frame["features_ready"].dtype == bool
    assert frame["label_available"].dtype == bool
    assert frame["training_ready"].equals(
        frame["features_ready"] & frame["label_available"]
    )

    boundary_position = 310
    boundary = prices.index[boundary_position]
    selected = strict_pre_fold_training_mask(
        frame,
        fold_first_decision_date=boundary,
    )
    maturity = pd.to_datetime(frame.loc[selected, "label_maturity_date"])
    assert not selected.loc[prices.index[missing_position]]
    assert (frame.index[selected] < boundary).all()
    assert (maturity < boundary).all()
    # Decision at boundary-2 matures exactly at the boundary and is purged.
    assert not selected.loc[prices.index[boundary_position - 2]]
    assert selected.loc[prices.index[boundary_position - 3]]
    assert not selected.iloc[-2:].any()

    years = entry_fill_year_mask(
        frame,
        first_year=int(frame["entry_fill_year"].dropna().min()),
        last_year=int(frame["entry_fill_year"].dropna().min()),
    )
    assert years.dtype == bool
    assert years.any()

    tampered = frame.copy()
    tampered.loc[prices.index[300], "features_ready"] = False
    with pytest.raises(ValueError, match="inconsistent"):
        strict_pre_fold_training_mask(
            tampered,
            fold_first_decision_date=boundary,
        )

    tampered_label = frame.copy()
    tampered_label.loc[prices.index[300], "cash_beats_long_10bps"] = 1 - int(
        tampered_label.loc[prices.index[300], "cash_beats_long_10bps"]
    )
    with pytest.raises(ValueError, match="ordinary CASH target"):
        strict_pre_fold_training_mask(
            tampered_label,
            fold_first_decision_date=boundary,
        )
