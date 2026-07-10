from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from agent_benchmark.direct_edge_features import (
    CONTEXT_VALUE_COLUMNS,
    DIRECT_EDGE_LABEL_COLUMNS,
    MARKET_SENTIMENT_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    build_direct_edge_feature_label_frame,
    build_five_session_direct_edge_labels,
    build_market_sentiment_features,
    canonical_market_sentiment_context,
    price_only_chronological_training_mask,
    sentiment_chronological_training_mask,
)
from agent_benchmark.downside_features import build_downside_price_features


def _price_frame(periods: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2000-01-03", periods=periods)
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


def _wide_context(dates: pd.DatetimeIndex) -> pd.DataFrame:
    step = np.arange(len(dates), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "IWM": 70.0 * np.exp(0.0002 * step + 0.014 * np.sin(step / 12.0)),
            "^VIX": 20.0 * np.exp(0.08 * np.sin(step / 15.0) + 0.001 * step),
            "^TNX": 3.5 + 0.003 * step + 0.15 * np.cos(step / 17.0),
        }
    )


def _open_path_frame(opens: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2010-01-04", periods=len(opens))
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


def test_schema_and_price_features_are_exactly_reused() -> None:
    prices = _price_frame()
    context = _wide_context(pd.DatetimeIndex(prices["date"]))
    combined = build_direct_edge_feature_label_frame(prices, context)
    reference = build_downside_price_features(prices)

    assert len(PRICE_FEATURE_COLUMNS) == 24
    assert len(MARKET_SENTIMENT_FEATURE_COLUMNS) == 15
    assert tuple(combined.loc[:, list(PRICE_FEATURE_COLUMNS)].columns) == PRICE_FEATURE_COLUMNS
    assert (
        tuple(combined.loc[:, list(MARKET_SENTIMENT_FEATURE_COLUMNS)].columns)
        == MARKET_SENTIMENT_FEATURE_COLUMNS
    )
    pdt.assert_frame_equal(
        combined.loc[:, list(PRICE_FEATURE_COLUMNS)],
        reference,
    )


def test_wide_context_aligns_only_to_aapl_sessions_without_filling_or_dropping() -> None:
    prices = _price_frame(280)
    dates = pd.DatetimeIndex(prices["date"])
    context = _wide_context(dates)
    missing_date = dates[260]
    context = context.loc[context["date"] != missing_date].copy()
    context = pd.concat(
        [
            context,
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("1999-12-31")],
                    "IWM": [999.0],
                    "^VIX": [999.0],
                    "^TNX": [999.0],
                }
            ),
        ],
        ignore_index=True,
    )

    aligned = canonical_market_sentiment_context(context, dates)
    combined = build_direct_edge_feature_label_frame(prices, context)

    assert tuple(aligned.columns) == CONTEXT_VALUE_COLUMNS
    assert aligned.index.equals(dates)
    assert len(aligned) == len(prices)
    assert aligned.loc[missing_date].isna().all()
    assert not (aligned == 0.0).any().any()
    assert combined.loc[missing_date, "sentiment_price_only_fallback"]
    assert combined.loc[missing_date, "sentiment_status"] == "missing_context"
    assert combined.loc[missing_date, list(MARKET_SENTIMENT_FEATURE_COLUMNS)].isna().any()


def test_long_context_prefers_adjusted_iwm_and_raw_vix_tnx_closes() -> None:
    dates = pd.bdate_range("2020-01-02", periods=3)
    rows: list[dict[str, object]] = []
    for offset, current_date in enumerate(dates):
        for symbol, close, adjusted in (
            ("IWM", 100.0 + offset, 200.0 + offset),
            ("^VIX", 20.0 + offset, 120.0 + offset),
            ("TNX", 3.0 + offset, 103.0 + offset),
        ):
            rows.append(
                {
                    "date": current_date,
                    "symbol": symbol,
                    "close": close,
                    "adj_close": adjusted,
                }
            )
    rows.append(
        {
            "date": pd.Timestamp("2020-01-04"),
            "symbol": "IWM",
            "close": 999.0,
            "adj_close": 999.0,
        }
    )

    aligned = canonical_market_sentiment_context(pd.DataFrame(rows), dates)

    assert aligned["iwm_adj_close"].tolist() == [200.0, 201.0, 202.0]
    assert aligned["vix_close"].tolist() == [20.0, 21.0, 22.0]
    assert aligned["tnx_close"].tolist() == [3.0, 4.0, 5.0]


def test_market_sentiment_formulas_for_iwm_vix_tnx_and_broad_market() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    context = _wide_context(dates)
    features = build_market_sentiment_features(prices, context)
    aligned = canonical_market_sentiment_context(context, dates)
    row = 280

    spy = prices["spy_adj_close"].to_numpy(dtype=float)
    qqq = prices["qqq_adj_close"].to_numpy(dtype=float)
    iwm = aligned["iwm_adj_close"].to_numpy(dtype=float)
    vix = aligned["vix_close"].to_numpy(dtype=float)
    tnx = aligned["tnx_close"].to_numpy(dtype=float)
    vix_log_window = np.log(vix[row - 251 : row + 1])
    iwm_one_day = np.diff(np.log(iwm[row - 20 : row + 1]))

    assert features.iloc[row]["spy_lr_5"] == pytest.approx(math.log(spy[row] / spy[row - 5]))
    assert features.iloc[row]["spy_drawdown_60"] == pytest.approx(
        spy[row] / np.max(spy[row - 59 : row + 1]) - 1.0
    )
    assert features.iloc[row]["qqq_drawdown_60"] == pytest.approx(
        qqq[row] / np.max(qqq[row - 59 : row + 1]) - 1.0
    )
    assert features.iloc[row]["iwm_lr_20"] == pytest.approx(
        math.log(iwm[row] / iwm[row - 20])
    )
    assert features.iloc[row]["iwm_rv_20"] == pytest.approx(
        np.std(iwm_one_day, ddof=1) * math.sqrt(252.0)
    )
    assert features.iloc[row]["qqq_minus_iwm_lr_20"] == pytest.approx(
        math.log(qqq[row] / qqq[row - 20]) - math.log(iwm[row] / iwm[row - 20])
    )
    assert features.iloc[row]["vix_log_level"] == pytest.approx(math.log(vix[row]))
    assert features.iloc[row]["vix_lr_5"] == pytest.approx(math.log(vix[row] / vix[row - 5]))
    assert features.iloc[row]["vix_z_252"] == pytest.approx(
        (math.log(vix[row]) - np.mean(vix_log_window))
        / np.std(vix_log_window, ddof=1)
    )
    assert features.iloc[row]["tnx_level"] == pytest.approx(tnx[row])
    assert features.iloc[row]["tnx_delta_5"] == pytest.approx(tnx[row] - tnx[row - 5])
    assert features.iloc[row]["tnx_delta_20"] == pytest.approx(tnx[row] - tnx[row - 20])


def test_direct_target_uses_open_t_plus_1_to_open_t_plus_6_and_exact_costs() -> None:
    # Row 0 sells at 100 on t+1 and buys back at 99 on t+6.
    prices = _open_path_frame([90, 100, 101, 102, 103, 104, 99, 98, 97, 96])
    labels = build_five_session_direct_edge_labels(prices)
    dates = pd.DatetimeIndex(prices["date"])
    row = labels.iloc[0]
    forward = math.log(99.0 / 100.0)

    assert tuple(labels.columns) == DIRECT_EDGE_LABEL_COLUMNS
    assert row["label_entry_date"] == dates[1]
    assert row["label_maturity_date"] == dates[6]
    assert row["label_entry_adjusted_open"] == pytest.approx(100.0)
    assert row["label_exit_adjusted_open"] == pytest.approx(99.0)
    assert row["aapl_forward_log_return_5"] == pytest.approx(forward)
    assert row["cash_active_log_edge_5bps"] == pytest.approx(
        math.log((1.0 - 0.0005) / (1.0 + 0.0005)) - forward
    )
    assert row["cash_active_log_edge_10bps"] == pytest.approx(
        math.log((1.0 - 0.0010) / (1.0 + 0.0010)) - forward
    )
    assert row["cash_beats_long_10bps"] == 1
    assert labels.iloc[-6:]["label_maturity_date"].isna().all()
    assert labels.iloc[-6:]["cash_beats_long_10bps"].isna().all()


def test_missing_interior_open_invalidates_direct_target() -> None:
    prices = _open_path_frame([90, 100, 101, 102, 103, 104, 99, 98, 97])
    prices.loc[3, "aapl_open"] = np.nan
    labels = build_five_session_direct_edge_labels(prices)

    assert pd.isna(labels.iloc[0]["label_maturity_date"])
    assert pd.isna(labels.iloc[0]["cash_active_log_edge_10bps"])
    assert pd.isna(labels.iloc[0]["cash_beats_long_10bps"])


def test_strict_before_masks_exclude_label_maturing_on_fold_boundary() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    context = _wide_context(dates)
    combined = build_direct_edge_feature_label_frame(prices, context)
    boundary_position = 280
    boundary = dates[boundary_position]

    price_mask = price_only_chronological_training_mask(
        combined,
        as_of_date=boundary,
    )
    sentiment_mask = sentiment_chronological_training_mask(
        combined,
        as_of_date=boundary,
    )

    # boundary-6 matures exactly at the boundary open and is excluded.  The
    # previous case matured one AAPL session earlier and is eligible.
    assert not price_mask.iloc[boundary_position - 6]
    assert price_mask.iloc[boundary_position - 7]
    assert not sentiment_mask.iloc[boundary_position - 6]
    assert sentiment_mask.iloc[boundary_position - 7]
    assert not price_mask.iloc[boundary_position:].any()
    assert not sentiment_mask.iloc[boundary_position:].any()


def test_sentiment_mask_falls_back_while_price_only_case_remains_eligible() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    context = _wide_context(dates)
    missing_position = 270
    context.loc[missing_position, "^VIX"] = np.nan
    combined = build_direct_edge_feature_label_frame(prices, context)
    boundary = dates[300]

    price_mask = price_only_chronological_training_mask(combined, as_of_date=boundary)
    sentiment_mask = sentiment_chronological_training_mask(combined, as_of_date=boundary)

    assert combined.iloc[missing_position]["sentiment_price_only_fallback"]
    assert price_mask.iloc[missing_position]
    assert not sentiment_mask.iloc[missing_position]
    assert pd.isna(combined.iloc[missing_position]["vix_log_level"])


def test_completed_close_features_do_not_change_when_future_rows_change() -> None:
    prices = _price_frame()
    dates = pd.DatetimeIndex(prices["date"])
    context = _wide_context(dates)
    decision_position = 280
    baseline = build_direct_edge_feature_label_frame(prices, context)

    changed_prices = prices.copy()
    changed_context = context.copy()
    changed_prices.loc[decision_position + 1 :, "aapl_adj_close"] *= 11.0
    changed_prices.loc[decision_position + 1 :, "spy_adj_close"] *= 7.0
    changed_prices.loc[decision_position + 1 :, "qqq_adj_close"] *= 5.0
    changed_context.loc[decision_position + 1 :, ["IWM", "^VIX", "^TNX"]] *= 13.0
    changed = build_direct_edge_feature_label_frame(changed_prices, changed_context)

    all_features = list(PRICE_FEATURE_COLUMNS) + list(MARKET_SENTIMENT_FEATURE_COLUMNS)
    pdt.assert_series_equal(
        baseline.iloc[decision_position][all_features],
        changed.iloc[decision_position][all_features],
        check_names=False,
    )

