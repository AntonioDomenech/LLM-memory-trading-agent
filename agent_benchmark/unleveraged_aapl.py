from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from agent_benchmark.deterministic_aapl import (
    CostAssumptions,
    EvaluationPeriod,
    _atomic_write_csv,
    _atomic_write_text,
    _flatten_download,
    _json_default,
    compare_ledgers,
    file_sha256,
    performance_metrics,
    simulate_period,
    terminal_close_sensitivity,
)


REQUIRED_CONTEXT_COLUMNS = (
    "aapl_open",
    "aapl_close",
    "aapl_adj_close",
    "spy_adj_close",
    "qqq_adj_close",
)

FINAL_PERIODS = (
    EvaluationPeriod("2024", "2024-01-01", "2024-12-31"),
    EvaluationPeriod("2025", "2025-01-01", "2025-12-31"),
    EvaluationPeriod("2026_ytd", "2026-01-01", "2026-07-09"),
)

# A date-only hash is stable across later price/dividend revisions and detects
# an omitted interior session even when a corrupted cache reaches the required
# final date.
FINAL_SESSION_COVERAGE = {
    "first_session": "1999-03-10",
    "last_session": "2026-07-09",
    "observations": 6875,
    "date_sequence_sha256": "b88df14b4ec60534ace68645ee19c8a0b7d03d0c2c1829a3ad48f8a0a24c9299",
}

# Four known outcome-reveal batches preceded the append-only local registry:
# the no-leverage trend replay, the frozen exhaustion-finalist batch,
# contextual exhaustion, and gap-down cash.
KNOWN_FINAL_REVEALS_BEFORE_REGISTRY = 4

BEAR_STRESS_PERIODS = (
    EvaluationPeriod("dotcom_2000_2002", "2000-01-01", "2002-12-31"),
    EvaluationPeriod("global_financial_crisis_2008", "2008-01-01", "2008-12-31"),
    EvaluationPeriod("q4_2018_selloff", "2018-10-01", "2018-12-31"),
    EvaluationPeriod("covid_crash", "2020-02-19", "2020-03-23"),
    EvaluationPeriod("rate_shock_2022", "2022-01-01", "2022-12-31"),
)


@dataclass(frozen=True)
class LongCashSpec:
    """A frozen, interpretable AAPL long/cash rule.

    Every supported rule returns exactly 0% or 100% AAPL exposure.  The
    decision is made from completed information at close t and the shared
    simulator executes it at adjusted open t+1.
    """

    name: str
    rule_type: str
    aapl_percentile_lookback: int = 126
    aapl_percentile: float = 0.90
    market_return_lookback: int = 10
    require_spy_negative: bool = True
    require_qqq_negative: bool = True
    cash_sessions: int = 1
    gap_threshold: float = -0.04
    selection_data_cutoff: str = "2023-12-31"

    def validate(self) -> None:
        supported = {"contextual_exhaustion", "gap_down", "exhaustion_or_gap"}
        if self.rule_type not in supported:
            raise ValueError(f"Unsupported long/cash rule_type: {self.rule_type}")
        if self.aapl_percentile_lookback < 20:
            raise ValueError("aapl_percentile_lookback must be at least 20 sessions")
        if not 0.5 < self.aapl_percentile < 1.0:
            raise ValueError("aapl_percentile must be strictly between 0.5 and 1.0")
        if self.market_return_lookback < 1:
            raise ValueError("market_return_lookback must be positive")
        if self.cash_sessions < 1:
            raise ValueError("cash_sessions must be positive")
        if not -0.5 < self.gap_threshold < 0.0:
            raise ValueError("gap_threshold must be a plausible negative return")
        if pd.Timestamp(self.selection_data_cutoff) >= pd.Timestamp("2024-01-01"):
            raise ValueError("Selection data cutoff must be before 2024")


CONTEXTUAL_EXHAUSTION_V1 = LongCashSpec(
    name="contextual_exhaustion_v1",
    rule_type="contextual_exhaustion",
    aapl_percentile_lookback=126,
    aapl_percentile=0.90,
    market_return_lookback=10,
    require_spy_negative=True,
    require_qqq_negative=True,
    cash_sessions=1,
)

GAP_DOWN_CASH_V1 = LongCashSpec(
    name="gap_down_cash_v1",
    rule_type="gap_down",
    gap_threshold=-0.04,
    cash_sessions=1,
)

EXHAUSTION_OR_GAP_V1 = LongCashSpec(
    name="exhaustion_or_gap_v1",
    rule_type="exhaustion_or_gap",
    aapl_percentile_lookback=126,
    aapl_percentile=0.90,
    market_return_lookback=10,
    require_spy_negative=True,
    require_qqq_negative=True,
    cash_sessions=1,
    gap_threshold=-0.04,
)

SPECS_BY_NAME = {
    spec.name: spec
    for spec in (CONTEXTUAL_EXHAUSTION_V1, GAP_DOWN_CASH_V1, EXHAUSTION_OR_GAP_V1)
}

SELECTION_PROTOCOL = {
    "protocol_name": "pre_2024_long_cash_v1",
    "selection_data_cutoff": "2023-12-31",
    "research_grid_candidates_screened_self_reported": 1508,
    "frozen_compact_grid_candidates": 36,
    "temporal_blocks": ["2004-2013", "2014-2018", "2019-2023"],
    "ranking": [
        "positive mean excess in every temporal block",
        "minimum wins of 6/10, 3/5, and 3/5",
        "positive median annual excess",
        "no more than 15 cash sessions and 30 orders per year",
        "maximize weakest block mean, then annual wins, then lower turnover",
        "require passing adjacent percentile/lookback configurations",
    ],
    "important_limitation": (
        "The 1,508-candidate research count is self-attested and has no complete saved table. "
        "This manifest was committed after the final years existed and is an auditable "
        "retrospective declaration, not cryptographic proof of a pristine holdout."
    ),
}


def canonical_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    if "date" in clean.columns:
        clean["date"] = pd.to_datetime(clean["date"], errors="raise")
        clean = clean.set_index("date")
    clean.index = pd.DatetimeIndex(pd.to_datetime(clean.index, errors="raise")).tz_localize(None)
    clean = clean.sort_index()
    if clean.index.has_duplicates:
        raise ValueError("Market data contains duplicate trading dates")
    missing = [column for column in REQUIRED_CONTEXT_COLUMNS if column not in clean.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")
    for column in REQUIRED_CONTEXT_COLUMNS:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=list(REQUIRED_CONTEXT_COLUMNS))
    if clean.empty:
        raise ValueError("No complete AAPL/SPY/QQQ observations remain")
    if (clean[list(REQUIRED_CONTEXT_COLUMNS)] <= 0).any().any():
        raise ValueError("Market prices must be strictly positive")
    clean["aapl_adj_open"] = (
        clean["aapl_open"] * clean["aapl_adj_close"] / clean["aapl_close"]
    )
    if (~np.isfinite(clean["aapl_adj_open"]) | (clean["aapl_adj_open"] <= 0)).any():
        raise ValueError("Adjusted AAPL open could not be constructed")
    return clean[
        [
            "aapl_open",
            "aapl_close",
            "aapl_adj_close",
            "aapl_adj_open",
            "spy_adj_close",
            "qqq_adj_close",
        ]
    ].astype(float)


def download_context_frame(start: str, end_inclusive: str) -> pd.DataFrame:
    end_exclusive = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).date().isoformat()
    downloaded: Dict[str, pd.DataFrame] = {}
    for symbol in ("AAPL", "SPY", "QQQ"):
        raw = yf.download(
            symbol,
            start=start,
            end=end_exclusive,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise RuntimeError(f"Yahoo Finance returned no data for {symbol}")
        downloaded[symbol] = _flatten_download(raw)
    aapl = downloaded["AAPL"][["open", "close", "adj_close"]].rename(
        columns={"open": "aapl_open", "close": "aapl_close", "adj_close": "aapl_adj_close"}
    )
    spy = downloaded["SPY"][["adj_close"]].rename(columns={"adj_close": "spy_adj_close"})
    qqq = downloaded["QQQ"][["adj_close"]].rename(columns={"adj_close": "qqq_adj_close"})
    return canonical_context_frame(aapl.join(spy, how="inner").join(qqq, how="inner"))


def load_or_download_context_frame(
    cache_path: Path,
    *,
    start: str,
    end_inclusive: str,
    refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end_inclusive)

    def requested_slice(source: pd.DataFrame) -> pd.DataFrame:
        result = source.loc[(source.index >= requested_start) & (source.index <= requested_end)]
        if result.empty:
            raise ValueError("Market snapshot does not overlap requested dates")
        return result

    if cache_path.exists() and not refresh:
        cached = canonical_context_frame(pd.read_csv(cache_path))
        if (
            cached.index.min() <= requested_start + pd.Timedelta(days=75)
            and cached.index.max() >= requested_end
        ):
            return requested_slice(cached), "cache"
    frame = download_context_frame(start, end_inclusive)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(
        frame.reset_index(names="date"),
        cache_path,
        index=False,
        float_format="%.17g",
    )
    return requested_slice(canonical_context_frame(pd.read_csv(cache_path))), "yfinance"


def context_data_sha256(frame: pd.DataFrame) -> str:
    payload = canonical_context_frame(frame).reset_index(names="date").to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.12g",
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def session_dates_sha256(frame: pd.DataFrame) -> str:
    dates = canonical_context_frame(frame).index
    payload = "".join(f"{value.date().isoformat()}\n" for value in dates)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_final_session_coverage(frame: pd.DataFrame) -> Dict[str, Any]:
    data = canonical_context_frame(frame)
    observed = {
        "first_session": data.index.min().date().isoformat(),
        "last_session": data.index.max().date().isoformat(),
        "observations": int(len(data)),
        "date_sequence_sha256": session_dates_sha256(data),
    }
    if observed != FINAL_SESSION_COVERAGE:
        raise ValueError(
            "Market data does not match the complete required AAPL/SPY/QQQ session sequence"
        )
    return {"passed": True, **observed}


def spec_sha256(spec: LongCashSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def selection_manifest(spec: LongCashSpec) -> Dict[str, Any]:
    manifest = {**SELECTION_PROTOCOL, "selected_spec": asdict(spec)}
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return {**manifest, "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def _rolling_cash_mask(trigger: pd.Series, sessions: int) -> pd.Series:
    return trigger.astype(float).rolling(sessions, min_periods=1).max().fillna(0.0).astype(bool)


def build_long_cash_target(frame: pd.DataFrame, spec: LongCashSpec) -> pd.Series:
    spec.validate()
    data = canonical_context_frame(frame)
    intraday_return = data["aapl_close"] / data["aapl_open"] - 1.0
    percentile = intraday_return.rolling(
        spec.aapl_percentile_lookback,
        min_periods=spec.aapl_percentile_lookback,
    ).quantile(spec.aapl_percentile).shift(1)
    spy_momentum = data["spy_adj_close"].pct_change(spec.market_return_lookback)
    qqq_momentum = data["qqq_adj_close"].pct_change(spec.market_return_lookback)
    exhaustion = intraday_return > percentile
    if spec.require_spy_negative:
        exhaustion &= spy_momentum < 0.0
    if spec.require_qqq_negative:
        exhaustion &= qqq_momentum < 0.0
    gap = data["aapl_adj_open"] / data["aapl_adj_close"].shift(1) - 1.0
    gap_down = gap < spec.gap_threshold
    if spec.rule_type == "contextual_exhaustion":
        trigger = exhaustion
        warmup_missing = percentile.isna() | spy_momentum.isna() | qqq_momentum.isna()
    elif spec.rule_type == "gap_down":
        trigger = gap_down
        warmup_missing = gap.isna()
    elif spec.rule_type == "exhaustion_or_gap":
        trigger = exhaustion | gap_down
        warmup_missing = percentile.isna() | spy_momentum.isna() | qqq_momentum.isna()
    else:  # pragma: no cover - validate is the public guard
        raise AssertionError(f"Unhandled rule type: {spec.rule_type}")
    cash = _rolling_cash_mask(trigger, spec.cash_sessions)
    target = pd.Series(np.where(cash, 0.0, 1.0), index=data.index, dtype=float)
    target.loc[warmup_missing] = np.nan
    return target.rename("target_exposure")


def assert_unleveraged_ledger(ledger: pd.DataFrame, *, tolerance: float = 1e-9) -> Dict[str, Any]:
    required = {
        "target_exposure",
        "new_exposure_after_fill",
        "holding_exposure_for_return",
        "cash",
        "shares",
        "margin_interest",
    }
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"Ledger lacks no-leverage proof columns: {missing}")
    targets = ledger["target_exposure"].astype(float)
    post_fill = ledger["new_exposure_after_fill"].astype(float)
    holding = ledger["holding_exposure_for_return"].astype(float)
    cash = ledger["cash"].astype(float)
    shares = ledger["shares"].astype(float)
    margin = ledger["margin_interest"].astype(float)
    violations: list[str] = []
    proof_columns = {
        "target_exposure": targets,
        "new_exposure_after_fill": post_fill,
        "holding_exposure_for_return": holding,
        "cash": cash,
        "shares": shares,
        "margin_interest": margin,
    }
    nonfinite = [
        name for name, values in proof_columns.items() if not np.isfinite(values.to_numpy()).all()
    ]
    if nonfinite:
        violations.append("nonfinite_values:" + ",".join(nonfinite))
    # Bounds checks below are meaningful only after excluding NaN/inf. Keep
    # collecting violations where possible, but never allow skipped-NaN pandas
    # reductions to turn corruption into a passing proof.
    if targets.min() < -tolerance or targets.max() > 1.0 + tolerance:
        violations.append("requested_target_outside_0_1")
    if post_fill.min() < -tolerance or post_fill.max() > 1.0 + tolerance:
        violations.append("post_fill_exposure_outside_0_1")
    if holding.min() < -tolerance or holding.max() > 1.0 + tolerance:
        violations.append("holding_exposure_outside_0_1")
    if cash.min() < -tolerance:
        violations.append("negative_cash")
    if shares.min() < -tolerance:
        violations.append("short_position")
    if margin.abs().max() > tolerance:
        violations.append("margin_interest_nonzero")
    if violations:
        raise RuntimeError("Unleveraged execution invariant failed: " + ", ".join(violations))
    return {
        "passed": True,
        "maximum_requested_target": float(targets.max()),
        "maximum_post_fill_exposure": float(post_fill.max()),
        "maximum_holding_exposure": float(holding.max()),
        "minimum_cash": float(cash.min()),
        "minimum_shares": float(shares.min()),
        "total_margin_interest": float(margin.sum()),
        "shorting": False,
        "borrowing": False,
    }


def simulate_unleveraged_period(
    frame: pd.DataFrame,
    target: pd.Series,
    period: EvaluationPeriod,
    costs: CostAssumptions,
    *,
    initial_cash: float = 1000.0,
) -> pd.DataFrame:
    if costs.annual_margin_rate != 0.0:
        raise ValueError("Unleveraged experiments require annual_margin_rate=0")
    numeric_target = pd.to_numeric(target, errors="coerce").dropna().to_numpy(dtype=float)
    if not np.all(np.isclose(numeric_target, 0.0, atol=1e-12) | np.isclose(numeric_target, 1.0, atol=1e-12)):
        raise ValueError("Unleveraged experiment targets must be binary LONG/CASH actions")
    ledger = simulate_period(
        frame,
        target,
        period,
        costs,
        initial_cash=initial_cash,
        max_exposure=1.0,
        rebalance="on_target_change",
    )
    assert_unleveraged_ledger(ledger)
    return ledger


def _period_report(
    frame: pd.DataFrame,
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    initial_cash: float,
) -> Dict[str, Any]:
    comparison = compare_ledgers(strategy, benchmark, initial_cash=initial_cash)
    strategy_terminal = terminal_close_sensitivity(strategy, frame, initial_cash=initial_cash)
    benchmark_terminal = terminal_close_sensitivity(benchmark, frame, initial_cash=initial_cash)
    terminal_excess = float(
        strategy_terminal["terminal_total_return"]
        - benchmark_terminal["terminal_total_return"]
    )
    comparison["terminal_close_sensitivity"] = {
        "strategy": strategy_terminal,
        "aapl_buy_hold": benchmark_terminal,
        "excess_return_vs_aapl_buy_hold": terminal_excess,
        "requested_success": bool(terminal_excess > 1e-8),
        "material_one_basis_point_success": bool(terminal_excess > 0.0001),
    }
    comparison["no_leverage_proof"] = assert_unleveraged_ledger(strategy)
    return comparison


def evaluate_fresh_periods(
    frame: pd.DataFrame,
    spec: LongCashSpec,
    periods: Sequence[EvaluationPeriod],
    costs: CostAssumptions,
    *,
    initial_cash: float = 1000.0,
) -> tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    target = build_long_cash_target(frame, spec)
    benchmark_target = pd.Series(1.0, index=canonical_context_frame(frame).index)
    reports: Dict[str, Any] = {}
    merged_ledgers: Dict[str, pd.DataFrame] = {}
    for period in periods:
        strategy = simulate_unleveraged_period(
            frame, target, period, costs, initial_cash=initial_cash
        )
        benchmark = simulate_unleveraged_period(
            frame, benchmark_target, period, costs, initial_cash=initial_cash
        )
        comparison = _period_report(
            frame, strategy, benchmark, initial_cash=initial_cash
        )
        merged = pd.concat(
            [strategy.add_prefix("strategy_"), benchmark.add_prefix("buy_hold_")],
            axis=1,
        )
        merged["active_daily_return"] = (
            merged["strategy_daily_return"] - merged["buy_hold_daily_return"]
        )
        reports[period.name] = {"period": asdict(period), **comparison}
        merged_ledgers[period.name] = merged
    return {
        "periods": reports,
        "all_periods_requested_success": all(r["requested_success"] for r in reports.values()),
        "all_periods_material_success": all(
            r["material_one_basis_point_success"] for r in reports.values()
        ),
        "all_periods_open_and_close_material_success": all(
            r["material_one_basis_point_success"]
            and r["terminal_close_sensitivity"]["material_one_basis_point_success"]
            for r in reports.values()
        ),
        "all_periods_open_and_close_requested_success": all(
            r["requested_success"]
            and r["terminal_close_sensitivity"]["requested_success"]
            for r in reports.values()
        ),
    }, merged_ledgers


def _continuous_period_returns(
    ledger: pd.DataFrame,
    periods: Sequence[EvaluationPeriod],
    *,
    initial_cash: float,
) -> Dict[str, float]:
    dates = pd.to_datetime(ledger["fill_date"])
    equity = ledger["equity"].astype(float).reset_index(drop=True)
    results: Dict[str, float] = {}
    for period in periods:
        indices = np.flatnonzero(
            (dates >= pd.Timestamp(period.start)) & (dates <= pd.Timestamp(period.end))
        )
        if not len(indices):
            raise ValueError(f"Continuous ledger lacks period {period.name}")
        first = int(indices[0])
        last = int(indices[-1])
        start_equity = float(initial_cash) if first == 0 else float(equity.iloc[first - 1])
        results[period.name] = float(equity.iloc[last] / start_equity - 1.0)
    return results


def evaluate_continuous_account(
    frame: pd.DataFrame,
    spec: LongCashSpec,
    periods: Sequence[EvaluationPeriod],
    costs: CostAssumptions,
    *,
    initial_cash: float = 1000.0,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    if not periods:
        raise ValueError("At least one continuous-account period is required")
    period = EvaluationPeriod("continuous", periods[0].start, periods[-1].end)
    data = canonical_context_frame(frame)
    target = build_long_cash_target(data, spec)
    benchmark_target = pd.Series(1.0, index=data.index)
    strategy = simulate_unleveraged_period(
        data, target, period, costs, initial_cash=initial_cash
    )
    benchmark = simulate_unleveraged_period(
        data, benchmark_target, period, costs, initial_cash=initial_cash
    )
    strategy_period_returns = _continuous_period_returns(
        strategy, periods, initial_cash=initial_cash
    )
    benchmark_period_returns = _continuous_period_returns(
        benchmark, periods, initial_cash=initial_cash
    )
    active_log_returns = {
        name: float(math.log1p(strategy_period_returns[name]) - math.log1p(benchmark_period_returns[name]))
        for name in strategy_period_returns
    }
    full = compare_ledgers(strategy, benchmark, initial_cash=initial_cash)
    merged = pd.concat(
        [strategy.add_prefix("strategy_"), benchmark.add_prefix("buy_hold_")], axis=1
    )
    merged["relative_wealth"] = merged["strategy_equity"] / merged["buy_hold_equity"] - 1.0
    merged["active_daily_return"] = (
        merged["strategy_daily_return"] - merged["buy_hold_daily_return"]
    )
    return {
        "period": asdict(period),
        "strategy_period_returns": strategy_period_returns,
        "buy_hold_period_returns": benchmark_period_returns,
        "period_active_log_returns": active_log_returns,
        "all_periods_positive_active_log_return": all(v > 0.0 for v in active_log_returns.values()),
        "all_periods_material_active_log_return": all(v > 0.001 for v in active_log_returns.values()),
        "full_span": full,
        "no_leverage_proof": assert_unleveraged_ledger(strategy),
    }, merged


def historical_rule_audit(
    frame: pd.DataFrame,
    spec: LongCashSpec,
    costs: CostAssumptions,
    *,
    first_year: int = 2000,
    last_year: int = 2023,
) -> Dict[str, Any]:
    if last_year >= 2024:
        raise ValueError("Historical selection audit must end before 2024")
    periods = tuple(
        EvaluationPeriod(str(year), f"{year}-01-01", f"{year}-12-31")
        for year in range(first_year, last_year + 1)
    )
    fresh, _ = evaluate_fresh_periods(frame, spec, periods, costs)
    annual = {
        name: float(report["excess_return_vs_aapl_buy_hold"])
        for name, report in fresh["periods"].items()
    }
    values = np.asarray(list(annual.values()), dtype=float)
    blocks = ((2000, 2007), (2008, 2015), (2016, 2023))
    block_reports: Dict[str, Any] = {}
    for start, end in blocks:
        block_values = np.asarray(
            [value for year, value in annual.items() if start <= int(year) <= end],
            dtype=float,
        )
        block_reports[f"{start}_{end}"] = {
            "years": int(len(block_values)),
            "wins": int(np.count_nonzero(block_values > 1e-8)),
            "mean_excess": float(block_values.mean()),
            "median_excess": float(np.median(block_values)),
        }
    continuous, _ = evaluate_continuous_account(frame, spec, periods, costs)
    return {
        "selection_data_cutoff": spec.selection_data_cutoff,
        "available_history_audit_window": {
            "first_year": first_year,
            "last_year": last_year,
            "note": (
                "All complete 2000-2023 years are reported. The self-attested original "
                "research selection blocks in the frozen manifest began in 2004."
            ),
        },
        "evidence_classification": "retrospective_pre_2024_selection",
        "annual_excess_returns": annual,
        "years_evaluated": int(len(values)),
        "years_beating_buy_hold": int(np.count_nonzero(values > 1e-8)),
        "mean_annual_excess": float(values.mean()),
        "median_annual_excess": float(np.median(values)),
        "worst_annual_excess": float(values.min()),
        "best_annual_excess": float(values.max()),
        "temporal_blocks": block_reports,
        "continuous_account": continuous,
    }


def negative_buy_hold_diagnostics(
    frame: pd.DataFrame,
    spec: LongCashSpec,
    costs: CostAssumptions,
    *,
    first_year: int = 2000,
    last_year: int = 2023,
) -> Dict[str, Any]:
    """Report all losing AAPL calendar years and fixed bear episodes.

    Calendar years are included by an objective rule rather than hand-picking
    only intervals flattering the strategy.  Named episodes add economically
    recognizable continuous stress windows.  These diagnostics never replace
    the declared 2024/2025/2026 promotion gate.
    """

    if last_year >= 2024:
        raise ValueError("Bear diagnostics used for selection must end before 2024")
    annual_periods = tuple(
        EvaluationPeriod(str(year), f"{year}-01-01", f"{year}-12-31")
        for year in range(first_year, last_year + 1)
    )
    annual, _ = evaluate_fresh_periods(frame, spec, annual_periods, costs)
    negative_years: Dict[str, Any] = {}
    for year, report in annual["periods"].items():
        benchmark_return = float(report["aapl_buy_hold"]["total_return"])
        if benchmark_return < 0.0:
            strategy_return = float(report["strategy"]["total_return"])
            negative_years[year] = {
                "strategy_return": strategy_return,
                "aapl_buy_hold_return": benchmark_return,
                "excess_return_vs_aapl_buy_hold": float(
                    report["excess_return_vs_aapl_buy_hold"]
                ),
                "strategy_positive_absolute_return": bool(strategy_return > 0.0),
                "beat_buy_hold": bool(report["requested_success"]),
                "strategy_max_drawdown": float(report["strategy"]["max_drawdown"]),
                "buy_hold_max_drawdown": float(report["aapl_buy_hold"]["max_drawdown"]),
                "cash_day_rate": float(report["strategy"]["cash_day_rate"]),
            }
    named, _ = evaluate_fresh_periods(frame, spec, BEAR_STRESS_PERIODS, costs)
    named_episodes: Dict[str, Any] = {}
    for name, report in named["periods"].items():
        strategy_return = float(report["strategy"]["total_return"])
        benchmark_return = float(report["aapl_buy_hold"]["total_return"])
        named_episodes[name] = {
            "period": report["period"],
            "strategy_return": strategy_return,
            "aapl_buy_hold_return": benchmark_return,
            "buy_hold_was_negative": bool(benchmark_return < 0.0),
            "excess_return_vs_aapl_buy_hold": float(
                report["excess_return_vs_aapl_buy_hold"]
            ),
            "strategy_positive_absolute_return": bool(strategy_return > 0.0),
            "beat_buy_hold": bool(report["requested_success"]),
            "strategy_max_drawdown": float(report["strategy"]["max_drawdown"]),
            "buy_hold_max_drawdown": float(report["aapl_buy_hold"]["max_drawdown"]),
            "cash_day_rate": float(report["strategy"]["cash_day_rate"]),
        }
    return {
        "role": "diagnostic_only_not_a_substitute_for_final_period_success",
        "calendar_year_inclusion_rule": (
            "every complete 2000-2023 calendar year with negative same-ledger "
            "AAPL buy-and-hold total return at the stated costs"
        ),
        "negative_calendar_years": negative_years,
        "negative_calendar_year_count": int(len(negative_years)),
        "years_strategy_beat_negative_buy_hold": int(
            sum(item["beat_buy_hold"] for item in negative_years.values())
        ),
        "years_strategy_earned_positive_absolute_return": int(
            sum(item["strategy_positive_absolute_return"] for item in negative_years.values())
        ),
        "named_stress_episodes": named_episodes,
    }


def _git_state(repo_root: Path) -> Dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = command("rev-parse", "HEAD")
        branch = command("branch", "--show-current")
        status = command("status", "--porcelain")
        return {"commit": commit, "branch": branch, "dirty": bool(status), "status": status}
    except Exception as exc:  # pragma: no cover - exercised only outside Git
        return {"commit": None, "branch": None, "dirty": None, "error": str(exc)}


def validate_source_repository(repo_root: Path, source_paths: Sequence[Path]) -> Dict[str, Any]:
    requested = repo_root.resolve()
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=source_paths[0].resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = Path(completed.stdout.strip()).resolve()
    if requested != actual:
        raise ValueError(
            "repo_root must be the actual Git repository containing the executing source"
        )
    tracked: list[str] = []
    for source in source_paths:
        resolved = source.resolve()
        try:
            relative = resolved.relative_to(actual)
        except ValueError as exc:
            raise ValueError("Executing source is outside repo_root") from exc
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
            cwd=actual,
            check=True,
            capture_output=True,
            text=True,
        )
        tracked.append(relative.as_posix())
    return {"repo_root": str(actual), "tracked_source_files": tracked}


def reserve_holdout_touch(
    registry_path: Path,
    *,
    candidate_hash: str,
    strategy_name: str,
    data_hash: str,
    git_commit: str | None,
) -> Dict[str, Any]:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {
            "schema_version": 1,
            "known_final_reveals_before_registry": KNOWN_FINAL_REVEALS_BEFORE_REGISTRY,
            "entries": [],
        }
    if registry.get("known_final_reveals_before_registry") != KNOWN_FINAL_REVEALS_BEFORE_REGISTRY:
        raise ValueError("Holdout registry baseline is inconsistent with this implementation")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Holdout registry entries must be a list")
    for entry in entries:
        if entry.get("candidate_hash") == candidate_hash:
            return {
                "touch_count": int(entry["touch_count"]),
                "registry_path": str(registry_path.resolve()),
                "registry_sha256": file_sha256(registry_path),
                "new_candidate_reveal": False,
                "entry": entry,
            }
    touch_count = KNOWN_FINAL_REVEALS_BEFORE_REGISTRY + len(entries) + 1
    entry = {
        "touch_count": touch_count,
        "candidate_hash": candidate_hash,
        "strategy_name": strategy_name,
        "data_sha256": data_hash,
        "git_commit": git_commit,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    entries.append(entry)
    _atomic_write_text(
        registry_path,
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
    )
    return {
        "touch_count": touch_count,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": file_sha256(registry_path),
        "new_candidate_reveal": True,
        "entry": entry,
    }


def run_unleveraged_experiment(
    *,
    repo_root: Path,
    output_dir: Path,
    cache_path: Path,
    spec: LongCashSpec,
    refresh_data: bool = False,
    data_start: str = "1999-01-01",
    data_end: str = "2026-07-09",
    periods: Sequence[EvaluationPeriod] = FINAL_PERIODS,
    selection_notes: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    created_at = datetime.now(timezone.utc)
    spec.validate()
    if tuple(periods) != FINAL_PERIODS:
        raise ValueError(
            "Promotion runs require the exact ordered 2024, 2025, and 2026-YTD periods"
        )
    implementation_path = Path(__file__).resolve()
    ledger_dependency_path = Path(simulate_period.__code__.co_filename).resolve()
    source_repository = validate_source_repository(
        repo_root, (implementation_path, ledger_dependency_path)
    )
    git_state = _git_state(repo_root)
    implementation_hash_at_start = file_sha256(implementation_path)
    ledger_dependency_hash_at_start = file_sha256(ledger_dependency_path)
    frame, origin = load_or_download_context_frame(
        cache_path,
        start=data_start,
        end_inclusive=data_end,
        refresh=refresh_data,
    )
    session_coverage = assert_final_session_coverage(frame)
    data_hash = context_data_sha256(frame)
    candidate_hash_at_start = hashlib.sha256(
        f"{spec_sha256(spec)}:{implementation_hash_at_start}:{ledger_dependency_hash_at_start}:{data_hash}".encode(
            "utf-8"
        )
    ).hexdigest()
    holdout_touch = reserve_holdout_touch(
        repo_root.resolve() / "data" / "unleveraged_aapl" / "holdout_registry.json",
        candidate_hash=candidate_hash_at_start,
        strategy_name=spec.name,
        data_hash=data_hash,
        git_commit=git_state.get("commit"),
    )
    base_costs = CostAssumptions(slippage_bps=5.0, annual_margin_rate=0.0)
    scenarios_costs = {
        "base_5bps": base_costs,
        "stress_10bps": CostAssumptions(slippage_bps=10.0, annual_margin_rate=0.0),
        "severe_20bps": CostAssumptions(slippage_bps=20.0, annual_margin_rate=0.0),
    }
    selection_audit = historical_rule_audit(frame, spec, base_costs)
    bear_diagnostics = negative_buy_hold_diagnostics(frame, spec, base_costs)
    scenarios: Dict[str, Any] = {}
    scenario_ledgers: Dict[str, Dict[str, pd.DataFrame]] = {}
    continuous_ledgers: Dict[str, pd.DataFrame] = {}
    for name, costs in scenarios_costs.items():
        fresh, ledgers = evaluate_fresh_periods(frame, spec, periods, costs)
        continuous, continuous_ledger = evaluate_continuous_account(
            frame, spec, periods, costs
        )
        scenarios[name] = {
            "costs": asdict(costs),
            "fresh_accounts": fresh,
            "continuous_account": continuous,
        }
        scenario_ledgers[name] = ledgers
        continuous_ledgers[name] = continuous_ledger

    run_id = f"aapl-unleveraged-{spec.name}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = run_dir / "market_snapshot.csv"
    _atomic_write_csv(
        canonical_context_frame(frame).reset_index(names="date"),
        snapshot_path,
        index=False,
        float_format="%.17g",
    )
    for scenario_name, ledgers in scenario_ledgers.items():
        for period_name, ledger in ledgers.items():
            _atomic_write_csv(
                ledger,
                run_dir / f"daily_{scenario_name}_{period_name}.csv",
                index=False,
                float_format="%.12g",
            )
        _atomic_write_csv(
            continuous_ledgers[scenario_name],
            run_dir / f"daily_{scenario_name}_continuous.csv",
            index=False,
            float_format="%.12g",
        )

    elapsed = time.perf_counter() - started
    integrity_errors: list[str] = []
    if git_state.get("dirty") is not False:
        integrity_errors.append("dirty_or_unverifiable_git_worktree_at_run_start")
    if context_data_sha256(pd.read_csv(snapshot_path)) != data_hash:
        integrity_errors.append("saved_market_snapshot_hash_mismatch")
    if elapsed > 3600.0:
        integrity_errors.append("runtime_exceeded_3600_seconds")
    base = scenarios["base_5bps"]
    stress = scenarios["stress_10bps"]
    base_success = bool(
        base["fresh_accounts"]["all_periods_open_and_close_material_success"]
        and base["continuous_account"]["all_periods_material_active_log_return"]
    )
    stress_success = bool(
        stress["fresh_accounts"]["all_periods_open_and_close_requested_success"]
        and stress["continuous_account"]["all_periods_positive_active_log_return"]
    )
    implementation_hash = file_sha256(implementation_path)
    ledger_dependency_hash = file_sha256(ledger_dependency_path)
    git_state_end = _git_state(repo_root)
    if implementation_hash != implementation_hash_at_start:
        integrity_errors.append("implementation_changed_during_run")
    if ledger_dependency_hash != ledger_dependency_hash_at_start:
        integrity_errors.append("ledger_dependency_changed_during_run")
    if git_state_end.get("commit") != git_state.get("commit"):
        integrity_errors.append("git_commit_changed_during_run")
    candidate_hash = hashlib.sha256(
        f"{spec_sha256(spec)}:{implementation_hash}:{ledger_dependency_hash}:{data_hash}".encode(
            "utf-8"
        )
    ).hexdigest()
    if candidate_hash != candidate_hash_at_start:
        integrity_errors.append("candidate_hash_changed_during_run")
    manifest = selection_manifest(spec)
    report = {
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "artifact_dir": str(run_dir.resolve()),
        "strategy": asdict(spec),
        "strategy_hash": spec_sha256(spec),
        "implementation_sha256": implementation_hash,
        "ledger_dependency_sha256": ledger_dependency_hash,
        "candidate_hash": candidate_hash,
        "data": {
            "source": "Yahoo Finance via yfinance; no paid API",
            "origin_this_run": origin,
            "cache_path": str(cache_path.resolve()),
            "saved_snapshot_path": str(snapshot_path.resolve()),
            "sha256": data_hash,
            "first_observation": canonical_context_frame(frame).index.min().date().isoformat(),
            "last_observation": canonical_context_frame(frame).index.max().date().isoformat(),
            "observations": int(len(frame)),
            "session_coverage_proof": session_coverage,
        },
        "execution_contract": {
            "asset": "AAPL only",
            "actions": ["LONG_100_PERCENT", "CASH_100_PERCENT"],
            "decision_information_cutoff": "completed close on decision_date",
            "fill": "next AAPL trading session adjusted open",
            "shorting": False,
            "leverage": False,
            "borrowing": False,
            "maximum_target_exposure": 1.0,
            "maximum_realized_exposure": 1.0,
            "cash_interest_rate": 0.0,
            "fractional_shares": True,
        },
        "selection": {
            "selection_data_cutoff": spec.selection_data_cutoff,
            "final_holdout_touch_count": int(holdout_touch["touch_count"]),
            "holdout_touch_registry": holdout_touch,
            "frozen_manifest": manifest,
            "provenance_classification": "self_attested_retrospective_manifest",
            "notes": dict(selection_notes or {}),
            "audit": selection_audit,
            "caveat": (
                "All final years are retrospective and have been observed during development; "
                "this is historical-fit evidence, not a pristine prospective holdout."
            ),
        },
        "bear_market_diagnostics": bear_diagnostics,
        "scenarios": scenarios,
        "promotion": {
            "base_success": base_success,
            "stress_10bps_success": stress_success,
            "capital_promotion_success": bool(base_success and stress_success and not integrity_errors),
            "integrity_errors": integrity_errors,
            "evidence_classification": "retrospective_historical_fit_not_profit_guarantee",
        },
        "reproducibility": {
            "git": git_state,
            "git_at_end": git_state_end,
            "source_repository": source_repository,
            "model": None,
            "model_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "api_cost_display": "$0.00",
            "runtime_seconds": elapsed,
            "completed_within_3600_seconds": bool(elapsed <= 3600.0),
        },
    }
    report_path = run_dir / "report.json"
    artifact_hashes = {
        str(path.relative_to(run_dir)).replace("\\", "/"): file_sha256(path)
        for path in sorted(run_dir.glob("*.csv"))
    }
    report["reproducibility"]["artifact_sha256"] = artifact_hashes
    manifest_path = run_dir / "selection_manifest.json"
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    report["reproducibility"]["selection_manifest_sha256"] = file_sha256(manifest_path)
    registry_snapshot_path = run_dir / "holdout_registry_snapshot.json"
    registry_source_path = Path(holdout_touch["registry_path"])
    _atomic_write_text(
        registry_snapshot_path,
        registry_source_path.read_text(encoding="utf-8"),
    )
    report["reproducibility"]["holdout_registry_snapshot_sha256"] = file_sha256(
        registry_snapshot_path
    )
    _atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    _atomic_write_text(
        output_dir / "latest.json",
        json.dumps({"run_id": run_id, "report": str(report_path.resolve())}, indent=2) + "\n",
    )
    final_elapsed = time.perf_counter() - started
    report["reproducibility"]["runtime_seconds"] = final_elapsed
    report["reproducibility"]["completed_within_3600_seconds"] = bool(
        final_elapsed <= 3600.0
    )
    if final_elapsed > 3600.0:
        report["promotion"]["capital_promotion_success"] = False
        if "runtime_exceeded_3600_seconds" not in report["promotion"]["integrity_errors"]:
            report["promotion"]["integrity_errors"].append(
                "runtime_exceeded_3600_seconds"
            )
    _atomic_write_text(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    checksums = {
        **report["reproducibility"]["artifact_sha256"],
        "selection_manifest.json": file_sha256(manifest_path),
        "holdout_registry_snapshot.json": file_sha256(registry_snapshot_path),
        "report.json": file_sha256(report_path),
    }
    _atomic_write_text(
        run_dir / "checksums.json",
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AAPL-only no-leverage long/cash audit")
    parser.add_argument("--strategy", choices=sorted(SPECS_BY_NAME), default=CONTEXTUAL_EXHAUSTION_V1.name)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/unleveraged_aapl/aapl_spy_qqq.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/unleveraged_aapl/runs"),
    )
    parser.add_argument("--refresh-data", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_unleveraged_experiment(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        cache_path=args.cache_path.resolve(),
        spec=SPECS_BY_NAME[args.strategy],
        refresh_data=args.refresh_data,
    )
    summary = {
        "run_id": report["run_id"],
        "strategy": report["strategy"]["name"],
        "promotion": report["promotion"],
        "runtime_seconds": report["reproducibility"]["runtime_seconds"],
        "api_cost_display": report["reproducibility"]["api_cost_display"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["promotion"]["capital_promotion_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
