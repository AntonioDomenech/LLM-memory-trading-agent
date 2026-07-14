from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import agent_benchmark.sector_breadth_features as sector_breadth
from agent_benchmark.direct_edge_features import DIRECT_EDGE_LABEL_COLUMNS
from agent_benchmark.downside_features import PRICE_FEATURE_COLUMNS
from agent_benchmark.sector_breadth_features import (
    SECTOR_BREADTH_FEATURE_COLUMNS,
    build_sector_breadth_feature_label_frame,
    load_sector_context_parquet,
)


_SECTOR_RATES = {
    "xlb_adj_close": -0.009,
    "xle_adj_close": -0.007,
    "xlf_adj_close": -0.005,
    "xli_adj_close": -0.003,
    "xlk_adj_close": 0.001,
    "xlp_adj_close": 0.003,
    "xlu_adj_close": 0.005,
    "xlv_adj_close": 0.007,
    "xly_adj_close": 0.009,
}
_LOADER_SYMBOLS = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "IWM",
    "^VIX",
)


def _frames(rows: int = 320) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.bdate_range("2004-01-02", periods=rows)
    step = np.arange(rows, dtype=float)
    aapl_close = 100.0 * np.exp(0.0012 * step)
    aapl_open = aapl_close * np.exp(-0.0003 + 0.0001 * np.sin(step / 7.0))
    prices = pd.DataFrame(
        {
            "aapl_open": aapl_open,
            "aapl_close": aapl_close,
            "aapl_adj_close": aapl_close,
            "spy_adj_close": 90.0 * np.exp(0.0008 * step),
            "qqq_adj_close": 80.0 * np.exp(0.0011 * step),
        },
        index=index,
    )
    context = pd.DataFrame(index=index)
    for offset, (column, rate) in enumerate(_SECTOR_RATES.items(), start=1):
        context[column] = (70.0 + offset) * np.exp(rate * step)
    context["iwm_adj_close"] = 65.0 * np.exp(0.0015 * step)
    context["vix_close"] = 20.0 * np.exp(
        0.0002 * step + 0.03 * np.sin(step / 13.0)
    )
    return prices, context


def _long_context_rows(dates: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_number, session in enumerate(dates, start=1):
        for symbol_number, symbol in enumerate(_LOADER_SYMBOLS, start=1):
            value = 10.0 + date_number + symbol_number / 10.0
            rows.append(
                {
                    "date": pd.Timestamp(session),
                    "symbol": symbol,
                    "close": value if symbol == "^VIX" else np.nan,
                    "adj_close": np.nan if symbol == "^VIX" else value,
                    "market_open": True,
                    "listed": True,
                    "ohlcv_available": True,
                    "ignored_payload": f"ignored-{symbol}",
                }
            )
    return pd.DataFrame(rows)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


def test_public_schema_is_fixed_and_builder_orders_complete_output() -> None:
    prices, context = _frames()
    frame = build_sector_breadth_feature_label_frame(prices, context)

    assert sector_breadth.__all__ == [
        "SECTOR_BREADTH_FEATURE_COLUMNS",
        "build_sector_breadth_feature_label_frame",
        "load_sector_context_parquet",
    ]
    assert len(SECTOR_BREADTH_FEATURE_COLUMNS) == 36
    assert len(set(SECTOR_BREADTH_FEATURE_COLUMNS)) == 36
    assert tuple(frame.columns) == (
        *PRICE_FEATURE_COLUMNS,
        *SECTOR_BREADTH_FEATURE_COLUMNS,
        *DIRECT_EDGE_LABEL_COLUMNS,
        "price_features_ready",
        "sector_breadth_features_ready",
        "sector_breadth_price_only_fallback",
        "sector_breadth_status",
        "label_available",
    )
    assert frame.iloc[-1]["price_features_ready"]
    assert frame.iloc[-1]["sector_breadth_features_ready"]
    assert not frame.iloc[-1]["sector_breadth_price_only_fallback"]
    assert frame.iloc[-1]["sector_breadth_status"] == "ready"


@pytest.mark.parametrize("window", [1, 5, 20, 60])
def test_sector_cross_section_and_relative_momentum_are_exact(window: int) -> None:
    prices, context = _frames()
    row = build_sector_breadth_feature_label_frame(prices, context).iloc[-1]
    sector_returns = np.asarray(list(_SECTOR_RATES.values())) * window
    defensive = np.asarray([0.003, 0.005, 0.007]) * window
    cyclical = np.asarray([-0.009, -0.007, -0.005, -0.003, 0.001, 0.009]) * window

    assert row[f"sector_participation_{window}"] == pytest.approx(5.0 / 9.0)
    assert row[f"sector_median_lr_{window}"] == pytest.approx(
        float(np.median(sector_returns))
    )
    assert row[f"sector_dispersion_lr_{window}"] == pytest.approx(
        float(np.std(sector_returns, ddof=0))
    )
    assert row[f"defensive_minus_cyclical_participation_{window}"] == pytest.approx(
        1.0 - 2.0 / 6.0
    )
    assert row[f"defensive_minus_cyclical_median_lr_{window}"] == pytest.approx(
        float(np.median(defensive) - np.median(cyclical))
    )
    assert row[f"iwm_minus_spy_lr_{window}"] == pytest.approx(0.0007 * window)
    assert row[f"aapl_vs_xlk_residual_lr_{window}"] == pytest.approx(
        0.0002 * window
    )
    if window in {1, 60}:
        assert row[f"aapl_vs_qqq_residual_lr_{window}"] == pytest.approx(
            0.0001 * window
        )
    else:
        assert f"aapl_vs_qqq_residual_lr_{window}" not in row.index
        assert row[f"aapl_minus_qqq_lr_{window}"] == pytest.approx(
            0.0001 * window
        )


def test_vix_uses_completed_close_level_changes_and_trailing_z_score() -> None:
    prices, context = _frames()
    frame = build_sector_breadth_feature_label_frame(prices, context)
    vix_log = np.log(context["vix_close"])
    expected_z = (
        vix_log.iloc[-1] - vix_log.iloc[-252:].mean()
    ) / vix_log.iloc[-252:].std(ddof=1)

    assert frame.iloc[-1]["vix_log_level"] == pytest.approx(vix_log.iloc[-1])
    for window in (1, 5, 20, 60):
        assert frame.iloc[-1][f"vix_lr_{window}"] == pytest.approx(
            vix_log.iloc[-1] - vix_log.iloc[-1 - window]
        )
    assert frame.iloc[-1]["vix_z_252"] == pytest.approx(expected_z)


def test_features_are_point_in_time_and_ignore_later_rows() -> None:
    prices, context = _frames()
    cutoff_position = 275
    cutoff = prices.index[cutoff_position]
    baseline = build_sector_breadth_feature_label_frame(prices, context)

    changed_prices = prices.copy()
    changed_context = context.copy()
    changed_prices.loc[prices.index > cutoff, :] *= 17.0
    changed_context.loc[context.index > cutoff, :] *= 23.0
    changed = build_sector_breadth_feature_label_frame(changed_prices, changed_context)
    feature_columns = [*PRICE_FEATURE_COLUMNS, *SECTOR_BREADTH_FEATURE_COLUMNS]

    pdt.assert_frame_equal(
        baseline.loc[:cutoff, feature_columns],
        changed.loc[:cutoff, feature_columns],
        check_exact=True,
    )


def test_direct_edge_label_fills_t_plus_1_and_matures_t_plus_6() -> None:
    prices, context = _frames()
    frame = build_sector_breadth_feature_label_frame(prices, context)
    position = 270
    decision = prices.index[position]
    entry = prices.index[position + 1]
    maturity = prices.index[position + 6]
    entry_open = prices.iloc[position + 1]["aapl_open"]
    exit_open = prices.iloc[position + 6]["aapl_open"]
    expected_return = math.log(exit_open / entry_open)
    expected_edge = math.log((1.0 - 0.001) / (1.0 + 0.001)) - expected_return

    assert frame.loc[decision, "label_entry_date"] == entry
    assert frame.loc[decision, "label_maturity_date"] == maturity
    assert frame.loc[decision, "aapl_forward_log_return_5"] == pytest.approx(
        expected_return
    )
    assert frame.loc[decision, "cash_active_log_edge_10bps"] == pytest.approx(
        expected_edge
    )
    assert frame.loc[decision, "label_available"]
    assert frame.iloc[-1]["label_available"] is np.False_


def test_missing_context_is_not_filled_and_early_rows_report_history() -> None:
    prices, context = _frames()
    missing_session = context.index[280]
    context = context.drop(index=missing_session)
    frame = build_sector_breadth_feature_label_frame(prices, context)

    assert frame.loc[missing_session, "sector_breadth_status"] == "missing_context"
    assert not frame.loc[missing_session, "sector_breadth_features_ready"]
    assert frame.loc[missing_session, "sector_breadth_price_only_fallback"]
    assert np.isnan(frame.loc[missing_session, "vix_log_level"])
    assert frame.iloc[10]["sector_breadth_status"] == "insufficient_history"


def test_wide_context_rejects_duplicates_missing_columns_and_nonpositive_values() -> None:
    prices, context = _frames()
    duplicate = pd.concat([context, context.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate dates"):
        build_sector_breadth_feature_label_frame(prices, duplicate)
    with pytest.raises(ValueError, match="missing required columns"):
        build_sector_breadth_feature_label_frame(
            prices, context.drop(columns="xlb_adj_close")
        )

    invalid = context.copy()
    invalid.iloc[-1, invalid.columns.get_loc("xlb_adj_close")] = 0.0
    frame = build_sector_breadth_feature_label_frame(prices, invalid)
    assert frame.iloc[-1]["sector_breadth_status"] == "missing_context"
    assert not frame.iloc[-1]["sector_breadth_features_ready"]


def test_context_dates_outside_price_index_are_rejected_before_value_coercion() -> None:
    prices, context = _frames()
    unauthorized = context.iloc[[-1]].copy()
    unauthorized.index = pd.DatetimeIndex([prices.index[-1] + pd.Timedelta(days=7)])
    unauthorized["xlb_adj_close"] = unauthorized["xlb_adj_close"].astype(object)
    unauthorized.loc[:, "xlb_adj_close"] = "not-a-number-and-must-not-be-coerced"
    crossed = pd.concat([context, unauthorized])

    with pytest.raises(ValueError, match="outside the authorized price index"):
        build_sector_breadth_feature_label_frame(prices, crossed)


def test_bounded_parquet_loader_projects_allowlist_and_inclusive_dates(
    tmp_path: Path,
) -> None:
    rows = _long_context_rows(["2018-12-28", "2018-12-31", "2019-01-02"])
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2018-12-31"),
                        "symbol": "SPY",
                        "close": 999.0,
                        "adj_close": 999.0,
                        "market_open": True,
                        "listed": True,
                        "ohlcv_available": True,
                        "ignored_payload": "must-not-pass",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    path = tmp_path / "context.parquet"
    _write_parquet(path, rows)

    loaded = load_sector_context_parquet(
        path, start="2018-12-28", end="2018-12-31"
    )

    assert loaded.index.tolist() == [
        pd.Timestamp("2018-12-28"),
        pd.Timestamp("2018-12-31"),
    ]
    assert tuple(loaded.columns) == (
        "xlb_adj_close",
        "xle_adj_close",
        "xlf_adj_close",
        "xli_adj_close",
        "xlk_adj_close",
        "xlp_adj_close",
        "xlu_adj_close",
        "xlv_adj_close",
        "xly_adj_close",
        "iwm_adj_close",
        "vix_close",
    )
    assert np.isfinite(loaded.to_numpy(dtype=float)).all()
    assert loaded.loc[pd.Timestamp("2018-12-31"), "vix_close"] == pytest.approx(
        rows.loc[
            (rows["date"] == pd.Timestamp("2018-12-31"))
            & (rows["symbol"] == "^VIX"),
            "close",
        ].iloc[0]
    )


def test_bounded_parquet_loader_excludes_closed_unlisted_and_unavailable_placeholders(
    tmp_path: Path,
) -> None:
    eligible = _long_context_rows(["2000-05-26", "2000-05-30"])
    placeholder_sets: list[pd.DataFrame] = []
    for session, flag in (
        ("2000-05-27", "market_open"),
        ("2000-05-28", "listed"),
        ("2000-05-29", "ohlcv_available"),
    ):
        placeholders = _long_context_rows([session])
        placeholders.loc[:, ["close", "adj_close"]] = np.nan
        placeholders.loc[:, flag] = False
        placeholder_sets.append(placeholders)
    rows = pd.concat([eligible, *placeholder_sets], ignore_index=True)
    path = tmp_path / "placeholders.parquet"
    _write_parquet(path, rows)

    loaded = load_sector_context_parquet(
        path, start="2000-05-26", end="2000-05-30"
    )

    assert loaded.index.tolist() == [
        pd.Timestamp("2000-05-26"),
        pd.Timestamp("2000-05-30"),
    ]
    assert np.isfinite(loaded.to_numpy(dtype=float)).all()


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nonfinite"])
def test_bounded_parquet_loader_rejects_ineligible_panels(
    tmp_path: Path,
    corruption: str,
) -> None:
    rows = _long_context_rows(["2018-12-31"])
    if corruption == "missing":
        rows = rows.loc[rows["symbol"] != "XLB"].copy()
        match = "incomplete symbol panel"
    elif corruption == "duplicate":
        rows = pd.concat([rows, rows.loc[rows["symbol"] == "XLB"]], ignore_index=True)
        match = "duplicate date/symbol"
    else:
        rows.loc[rows["symbol"] == "XLB", "adj_close"] = np.inf
        match = "nonfinite"
    path = tmp_path / f"{corruption}.parquet"
    _write_parquet(path, rows)

    with pytest.raises(ValueError, match=match):
        load_sector_context_parquet(
            path, start="2018-12-31", end="2018-12-31"
        )


def test_bounded_parquet_loader_rejects_invalid_bounds_empty_and_missing_schema(
    tmp_path: Path,
) -> None:
    rows = _long_context_rows(["2018-12-31"])
    path = tmp_path / "valid.parquet"
    _write_parquet(path, rows)

    with pytest.raises(ValueError, match="start must be"):
        load_sector_context_parquet(path, start="2019-01-01", end="2018-12-31")
    with pytest.raises(ValueError, match="2000-05-26"):
        load_sector_context_parquet(path, start="2000-05-25", end="2018-12-31")
    with pytest.raises(ValueError, match="no eligible rows"):
        load_sector_context_parquet(path, start="2017-01-01", end="2017-01-02")

    missing_schema = tmp_path / "missing-schema.parquet"
    _write_parquet(missing_schema, rows.drop(columns="adj_close"))
    with pytest.raises(ValueError, match="Could not execute bounded"):
        load_sector_context_parquet(
            missing_schema, start="2018-12-31", end="2018-12-31"
        )
