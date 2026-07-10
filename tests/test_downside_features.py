from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.cftc_cot_policy import COTWeeklyRecord
from agent_benchmark.downside_features import (
    CFTC_FEATURE_COLUMNS,
    LABEL_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    build_cftc_continuous_features,
    build_downside_feature_label_frame,
    build_downside_price_features,
    build_five_session_downside_labels,
    canonical_downside_price_frame,
    chronological_training_mask,
    chronologically_eligible_cases,
)


def _price_frame(periods: int = 280) -> pd.DataFrame:
    dates = pd.bdate_range("2000-01-03", periods=periods)
    step = np.arange(periods, dtype=float)
    aapl_close = 100.0 * np.exp(0.0005 * step + 0.012 * np.sin(step / 7.0))
    aapl_open = aapl_close * np.exp(-0.002 * np.cos(step / 5.0))
    spy_close = 90.0 * np.exp(0.00025 * step + 0.006 * np.sin(step / 11.0))
    qqq_close = 80.0 * np.exp(0.00035 * step + 0.009 * np.cos(step / 9.0))
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


def _open_path_frame(opens: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2010-01-04", periods=len(opens))
    values = np.asarray(opens, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "aapl_open": values,
            "aapl_close": values,
            "aapl_adj_close": values,
            "spy_adj_close": 100.0 + np.arange(len(opens), dtype=float),
            "qqq_adj_close": 80.0 + np.arange(len(opens), dtype=float),
        }
    )


def _cot_release_records(
    count: int,
    *,
    common_last_availability: bool = False,
    constant: bool = False,
) -> tuple[list[COTWeeklyRecord], list[dict[str, float | date]]]:
    first_report = date(2004, 1, 6)
    values: list[dict[str, float | date]] = []
    records: list[COTWeeklyRecord] = []
    shared_availability = first_report + timedelta(days=7 * (count - 1) + 8)
    for index in range(count):
        report_date = first_report + timedelta(days=7 * index)
        availability = report_date + timedelta(days=8)
        if common_last_availability and index >= count - 2:
            availability = shared_availability
        if constant:
            nasdaq, sp500, vix = 0.10, -0.05, 0.25
        else:
            nasdaq = -0.30 + 0.012 * index
            sp500 = -0.10 + 0.003 * index
            vix = 0.15 + 0.004 * index
        values.append(
            {
                "report_date": report_date,
                "availability_date": availability,
                "NASDAQ": nasdaq,
                "SP500": sp500,
                "VIX": vix,
            }
        )
        for market, value in (("NASDAQ", nasdaq), ("SP500", sp500), ("VIX", vix)):
            records.append(
                COTWeeklyRecord(
                    market=market,
                    report_date=report_date,
                    availability_date=availability,
                    net_share=value,
                )
            )
    # Input ordering must not affect the causal release ordering.
    return list(reversed(records)), values


def test_price_feature_schema_is_exact_and_uses_only_trailing_values() -> None:
    raw = _price_frame()
    data = canonical_downside_price_frame(raw)
    features = build_downside_price_features(raw)

    assert tuple(features.columns) == PRICE_FEATURE_COLUMNS
    assert len(PRICE_FEATURE_COLUMNS) == 24
    assert features.index.equals(data.index)
    assert not features.iloc[251].notna().all()
    assert features.iloc[252].notna().all()

    row = 270
    assert features.iloc[row]["aapl_lr_5"] == pytest.approx(
        math.log(
            data["aapl_adj_close"].iloc[row]
            / data["aapl_adj_close"].iloc[row - 5]
        )
    )
    assert features.iloc[row]["aapl_intraday_lr"] == pytest.approx(
        math.log(data["aapl_close"].iloc[row] / data["aapl_open"].iloc[row])
    )
    assert features.iloc[row]["aapl_minus_qqq_lr_20"] == pytest.approx(
        features.iloc[row]["aapl_lr_20"] - features.iloc[row]["qqq_lr_20"]
    )
    assert features.iloc[row]["qqq_minus_spy_lr_20"] == pytest.approx(
        features.iloc[row]["qqq_lr_20"] - features.iloc[row]["spy_lr_20"]
    )


def test_incomplete_price_session_is_preserved_and_cannot_become_ready() -> None:
    raw = _price_frame()
    missing_date = pd.Timestamp(raw.loc[260, "date"])
    raw.loc[260, "qqq_adj_close"] = np.nan

    canonical = canonical_downside_price_frame(raw)
    features = build_downside_price_features(raw)

    assert len(canonical) == len(raw)
    assert missing_date in canonical.index
    assert not features.loc[missing_date].notna().all()


def test_five_session_label_uses_open_t_plus_1_to_open_t_plus_6() -> None:
    # For row 0: enter at 100 on session +1 and mature/exit at 96 on +6.
    raw = _open_path_frame([90, 100, 101, 102, 103, 104, 96, 97, 98, 99, 100, 101])
    labels = build_five_session_downside_labels(raw)
    dates = pd.DatetimeIndex(raw["date"])

    assert tuple(labels.columns) == LABEL_COLUMNS
    row = labels.iloc[0]
    assert row["label_entry_date"] == dates[1]
    assert row["label_maturity_date"] == dates[6]
    assert row["label_entry_adjusted_open"] == pytest.approx(100.0)
    assert row["label_exit_adjusted_open"] == pytest.approx(96.0)
    assert row["aapl_forward_simple_return_5"] == pytest.approx(-0.04)
    assert row["aapl_forward_log_return_5"] == pytest.approx(math.log(0.96))
    # The crash threshold is inclusive at a simple five-session return of -4%.
    assert row["crash_label_5session"] == 1

    expected_5bps = math.log((1.0 - 0.0005) / (1.0 + 0.0005)) - math.log(0.96)
    expected_10bps = math.log((1.0 - 0.0010) / (1.0 + 0.0010)) - math.log(0.96)
    assert row["cash_active_log_edge_5bps"] == pytest.approx(expected_5bps)
    assert row["cash_active_log_edge_10bps"] == pytest.approx(expected_10bps)
    assert row["cash_beats_long_5bps"] == 1
    assert row["cash_beats_long_10bps"] == 1
    assert labels.iloc[-6:]["label_maturity_date"].isna().all()
    assert labels.iloc[-6:]["crash_label_5session"].isna().all()


def test_missing_interior_future_open_invalidates_the_whole_label() -> None:
    raw = _open_path_frame([90, 100, 101, 102, 103, 104, 96, 97, 98])
    # Session +3 lies inside row 0's five-session holding interval.
    raw.loc[3, "aapl_open"] = np.nan
    labels = build_five_session_downside_labels(raw)

    assert pd.isna(labels.iloc[0]["label_maturity_date"])
    assert pd.isna(labels.iloc[0]["crash_label_5session"])
    assert pd.isna(labels.iloc[0]["cash_active_log_edge_10bps"])


def test_chronological_mask_admits_label_only_at_t_plus_6_open() -> None:
    combined = build_downside_feature_label_frame(_price_frame(), cot_records=())
    dates = combined.index
    boundary = 260
    mask = chronological_training_mask(combined, as_of_date=dates[boundary])

    # Row boundary-6 matures at the boundary open and is knowable by its close.
    assert mask.iloc[boundary - 6]
    # The five immediately preceding decisions have not yet matured.
    assert not mask.iloc[boundary - 5 : boundary].any()
    assert not mask.iloc[boundary]

    cases = chronologically_eligible_cases(combined, as_of_date=dates[boundary])
    assert (cases["label_maturity_date"] <= dates[boundary]).all()
    assert cases.loc[:, list(PRICE_FEATURE_COLUMNS)].notna().all().all()

    frozen_fold_mask = chronological_training_mask(
        combined,
        as_of_date=dates[boundary],
        strictly_before_as_of=True,
    )
    # A model frozen before the fold starts cannot use a label first revealed
    # at that fold's opening print.
    assert not frozen_fold_mask.iloc[boundary - 6]
    assert frozen_fold_mask.iloc[boundary - 7]


def test_cftc_z52_excludes_current_and_same_availability_batch() -> None:
    # Releases 52 and 53 share one availability date.  Release 53 is current;
    # release 52 must not leak into its strictly-earlier reference window.
    records, values = _cot_release_records(54, common_last_availability=True)
    decision = values[-1]["availability_date"]
    assert isinstance(decision, date)
    frame = build_cftc_continuous_features([decision], records)
    row = frame.iloc[0]

    nasdaq_history = np.asarray([float(item["NASDAQ"]) for item in values[:52]])
    divergence_history = np.asarray(
        [float(item["NASDAQ"]) - float(item["SP500"]) for item in values[:52]]
    )
    vix_history = np.asarray([float(item["VIX"]) for item in values[:52]])
    expected = (
        (float(values[-1]["NASDAQ"]) - nasdaq_history.mean())
        / nasdaq_history.std(ddof=0),
        (
            float(values[-1]["NASDAQ"])
            - float(values[-1]["SP500"])
            - divergence_history.mean()
        )
        / divergence_history.std(ddof=0),
        (float(values[-1]["VIX"]) - vix_history.mean())
        / vix_history.std(ddof=0),
    )

    assert row["cftc_status"] == "ready"
    assert not row["cftc_price_only_fallback"]
    assert row["cftc_complete_releases_available"] == 54
    assert row["cftc_prior_complete_releases"] == 52
    assert tuple(row[list(CFTC_FEATURE_COLUMNS)]) == pytest.approx(expected)


def test_cftc_age_14_is_ready_and_age_15_forces_price_only_fallback() -> None:
    records, values = _cot_release_records(53)
    availability = values[-1]["availability_date"]
    assert isinstance(availability, date)
    dates = [availability, availability + timedelta(days=14), availability + timedelta(days=15)]
    frame = build_cftc_continuous_features(dates, records)

    assert frame["cftc_status"].tolist() == ["ready", "ready", "stale_release"]
    assert frame["cftc_price_only_fallback"].tolist() == [False, False, True]
    assert frame["cftc_release_age_days"].tolist() == [0, 14, 15]
    assert frame.iloc[2][list(CFTC_FEATURE_COLUMNS)].isna().all()


def test_cftc_missing_history_or_zero_variance_never_imputes_zero_zscores() -> None:
    short_records, short_values = _cot_release_records(52)
    short_date = short_values[-1]["availability_date"]
    assert isinstance(short_date, date)
    short = build_cftc_continuous_features([short_date], short_records).iloc[0]
    assert short["cftc_status"] == "insufficient_history"
    assert short["cftc_price_only_fallback"]
    assert short[list(CFTC_FEATURE_COLUMNS)].isna().all()

    constant_records, constant_values = _cot_release_records(53, constant=True)
    constant_date = constant_values[-1]["availability_date"]
    assert isinstance(constant_date, date)
    constant = build_cftc_continuous_features(
        [constant_date], constant_records
    ).iloc[0]
    assert constant["cftc_status"] == "zero_variance"
    assert constant["cftc_price_only_fallback"]
    assert constant[list(CFTC_FEATURE_COLUMNS)].isna().all()


def test_combined_frame_uses_explicit_price_only_fallback_without_cftc() -> None:
    combined = build_downside_feature_label_frame(_price_frame(), cot_records=())

    assert combined["cftc_price_only_fallback"].all()
    assert (combined["cftc_status"] == "no_complete_release").all()
    assert combined[list(CFTC_FEATURE_COLUMNS)].isna().all().all()
    assert not combined.iloc[251]["price_features_ready"]
    assert combined.iloc[252]["price_features_ready"]
