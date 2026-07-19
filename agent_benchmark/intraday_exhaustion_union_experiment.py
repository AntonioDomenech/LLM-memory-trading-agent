"""Development-only test of the fixed exhaustion union's intraday cash window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .chronological_exhaustion_experiment import load_bounded_prices
from .chronological_exhaustion_expert import (
    build_fixed_expert_signals,
    canonicalize_one_session_signals,
)
from .deterministic_aapl import (
    CostAssumptions,
    EvaluationPeriod,
    compare_ledgers,
)
from .unleveraged_aapl import (
    assert_unleveraged_ledger,
    canonical_context_frame,
    simulate_unleveraged_period,
)


CONTRACT_VERSION = "aapl-intraday-exhaustion-union-v1"
INPUT_LITERAL_SHA256 = (
    "9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1"
)
INPUT_BYTE_COUNT = 590_761
INPUT_ROWS = 4_986
INPUT_FIRST = pd.Timestamp("1999-03-10")
INPUT_LAST = pd.Timestamp("2018-12-31")
EXPERT_SOURCE_GIT_BLOB = "3d01affe636f2917306358a7d80f285b5a10f1d4"
EXPERT_SOURCE_LITERAL_SHA256 = (
    "a30224763c9858aed905b76215c2c5a66eddd58f107182d751d8eb6a32688c6e"
)
EXPECTED_DECISION_DATES_SHA256 = (
    "b4bec71ac4159086edb1faa3151630bb524b6f2e8b7fdaebd2ebf5dab68dbb13"
)
ACCOUNT_START = pd.Timestamp("2005-01-01")
ACCOUNT_END = pd.Timestamp("2018-12-31")
INITIAL_CASH = 1_000.0
EXPECTED_EPISODES = 121
COST_SCENARIOS = (("base_5bps", 5.0), ("stress_10bps", 10.0))
NEGATIVE_AAPL_YEARS = (2008, 2015, 2018)
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")


class IntradayExhaustionUnionError(RuntimeError):
    """The frozen experiment contract could not be evaluated safely."""


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    text = json.dumps(
        value,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_oid(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object ID


def _friction_log_edge(cost_bps: float) -> float:
    cost = float(cost_bps) / 10_000.0
    return math.log((1.0 - cost) / (1.0 + cost))


def build_account_union_signal(frame: pd.DataFrame) -> pd.Series:
    """Return the inherited development-account union with its frozen cooldown."""

    signals = build_fixed_expert_signals(frame)
    candidate = (
        signals["unfiltered_union_candidate_signal"].astype(bool)
        & signals["stage_outcome_available"].astype(bool)
    )
    candidate.loc[candidate.index < ACCOUNT_START] = False
    accepted = canonicalize_one_session_signals(candidate)
    accepted.name = "union_cash_signal"
    return accepted


def _decision_dates_sha256(signal: pd.Series) -> str:
    dates = pd.DatetimeIndex(signal.index[signal.to_numpy(dtype=bool)])
    payload = "".join(f"{date.date().isoformat()}\n" for date in dates).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _finalize_open_ledger(rows: list[dict[str, Any]]) -> pd.DataFrame:
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise IntradayExhaustionUnionError("Intraday ledger is empty")
    ledger["daily_return"] = ledger["equity"].pct_change()
    ledger.loc[0, "daily_return"] = ledger.loc[0, "equity"] / INITIAL_CASH - 1.0
    ledger["monetary_pnl"] = ledger["equity"].diff()
    ledger.loc[0, "monetary_pnl"] = ledger.loc[0, "equity"] - INITIAL_CASH
    peaks = np.maximum.accumulate(
        np.r_[INITIAL_CASH, ledger["equity"].to_numpy(dtype=float)]
    )[1:]
    ledger["drawdown"] = ledger["equity"].to_numpy(dtype=float) / peaks - 1.0
    return ledger


def simulate_intraday_only(
    frame: pd.DataFrame,
    cash_signal: pd.Series,
    *,
    cost_bps: float,
    start: pd.Timestamp = ACCOUNT_START,
    end: pd.Timestamp = ACCOUNT_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sell at the next open and execute the precommitted buy at that close."""

    data = canonical_context_frame(frame)
    signal = cash_signal.reindex(data.index)
    if signal.isna().any() or not signal.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise IntradayExhaustionUnionError("Cash signal must be complete booleans")
    trade_today = signal.shift(1, fill_value=False).astype(bool)
    evaluation = data.loc[(data.index >= start) & (data.index <= end)]
    if evaluation.empty:
        raise IntradayExhaustionUnionError("No development rows are available")

    cost = float(cost_bps) / 10_000.0
    if not math.isfinite(cost) or cost < 0.0 or cost >= 1.0:
        raise IntradayExhaustionUnionError("Invalid per-leg execution cost")

    cash = float(INITIAL_CASH)
    shares = 0.0
    open_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    full_dates = data.index
    for row_number, (fill_date, market) in enumerate(evaluation.iterrows()):
        fill_date = pd.Timestamp(fill_date)
        position = int(full_dates.get_loc(fill_date))
        if position == 0:
            raise IntradayExhaustionUnionError("Development lacks a prior decision row")
        decision_date = pd.Timestamp(full_dates[position - 1])
        reference_open = float(market["aapl_adj_open"])
        equity_before = cash + shares * reference_open
        if not math.isfinite(equity_before) or equity_before <= 0.0:
            raise IntradayExhaustionUnionError("Portfolio equity became non-positive")
        holding_exposure = shares * reference_open / equity_before
        intraday_cash = bool(trade_today.loc[fill_date])

        delta = 0.0
        fill_price = reference_open
        slippage = 0.0
        target = 0.0 if intraday_cash else 1.0
        if row_number == 0:
            if intraday_cash:
                raise IntradayExhaustionUnionError(
                    "The development account cannot start in cash for free"
                )
            fill_price = reference_open * (1.0 + cost)
            delta = cash / fill_price
            slippage = delta * (fill_price - reference_open)
            cash = 0.0
            shares = delta
        elif intraday_cash:
            if shares <= 0.0 or cash < -1e-10:
                raise IntradayExhaustionUnionError("Open sell has no long position")
            delta = -shares
            fill_price = reference_open * (1.0 - cost)
            slippage = abs(delta) * (reference_open - fill_price)
            cash += shares * fill_price
            shares = 0.0
        elif shares <= 0.0:
            raise IntradayExhaustionUnionError("Unexpected open without AAPL holdings")

        equity_after = cash + shares * reference_open
        post_exposure = shares * reference_open / equity_after
        turnover = abs(delta) * reference_open / equity_before
        open_rows.append(
            {
                "decision_date": decision_date.date().isoformat(),
                "fill_date": fill_date.date().isoformat(),
                "adjusted_open": reference_open,
                "equity_before_fill": equity_before,
                "equity": equity_after,
                "cash": cash,
                "shares": shares,
                "holding_exposure_for_return": holding_exposure,
                "target_exposure": target,
                "new_exposure_after_fill": post_exposure,
                "signed_share_delta": delta,
                "reference_price": reference_open,
                "fill_price": fill_price,
                "turnover": turnover,
                "fees": 0.0,
                "slippage": slippage,
                "margin_interest": 0.0,
                "trade_executed": bool(abs(delta) > 1e-12),
            }
        )
        if abs(delta) > 1e-12:
            event_rows.append(
                {
                    "decision_date": decision_date.date().isoformat(),
                    "event_date": fill_date.date().isoformat(),
                    "event_time": "open",
                    "side": "buy" if delta > 0.0 else "sell",
                    "reference_price": reference_open,
                    "fill_price": fill_price,
                    "signed_share_delta": delta,
                    "cash_after": cash,
                    "shares_after": shares,
                    "equity_at_reference_after": equity_after,
                    "target_exposure_after": target,
                    "slippage": slippage,
                }
            )

        if intraday_cash:
            reference_close = float(market["aapl_adj_close"])
            close_equity_before = cash
            close_fill = reference_close * (1.0 + cost)
            close_delta = cash / close_fill
            close_slippage = close_delta * (close_fill - reference_close)
            cash = 0.0
            shares = close_delta
            close_equity_after = shares * reference_close
            event_rows.append(
                {
                    "decision_date": decision_date.date().isoformat(),
                    "event_date": fill_date.date().isoformat(),
                    "event_time": "close",
                    "side": "buy",
                    "reference_price": reference_close,
                    "fill_price": close_fill,
                    "signed_share_delta": close_delta,
                    "cash_after": cash,
                    "shares_after": shares,
                    "equity_at_reference_after": close_equity_after,
                    "target_exposure_after": 1.0,
                    "slippage": close_slippage,
                }
            )
            if close_equity_after <= 0.0 or close_equity_after > close_equity_before:
                raise IntradayExhaustionUnionError("Close buy cost is inconsistent")

    ledger = _finalize_open_ledger(open_rows)
    assert_unleveraged_ledger(ledger)
    events = pd.DataFrame(event_rows)
    if events.empty:
        raise IntradayExhaustionUnionError("No trading events were produced")
    if (events["cash_after"] < -1e-9).any() or (events["shares_after"] < -1e-12).any():
        raise IntradayExhaustionUnionError("Event ledger violates cash/share safety")
    return ledger, events


def _episode_rows(
    frame: pd.DataFrame,
    cash_signal: pd.Series,
    *,
    cost_bps: float,
    mode: str,
) -> pd.DataFrame:
    if mode not in {"intraday", "open_to_open"}:
        raise ValueError("Unknown episode mode")
    data = canonical_context_frame(frame)
    positions = np.flatnonzero(cash_signal.reindex(data.index).to_numpy(dtype=bool))
    friction = _friction_log_edge(cost_bps)
    rows: list[dict[str, Any]] = []
    for position in positions:
        if position + 2 >= len(data):
            continue
        decision = data.index[position]
        entry = data.index[position + 1]
        reference_exit = data.index[position + 2]
        entry_open = float(data.iloc[position + 1]["aapl_adj_open"])
        if mode == "intraday":
            buyback_price = float(data.iloc[position + 1]["aapl_adj_close"])
            buyback_date = entry
        else:
            buyback_price = float(data.iloc[position + 2]["aapl_adj_open"])
            buyback_date = reference_exit
        raw_edge = math.log(entry_open / buyback_price)
        net_edge = raw_edge + friction
        rows.append(
            {
                "decision_date": decision.date().isoformat(),
                "entry_date": entry.date().isoformat(),
                "buyback_date": buyback_date.date().isoformat(),
                "reference_exit_date": reference_exit.date().isoformat(),
                "mode": mode,
                "raw_active_log_edge": raw_edge,
                "net_active_log_edge": net_edge,
                "win": bool(net_edge > 0.0),
            }
        )
    return pd.DataFrame(rows)


def _period_return(
    ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> float:
    dates = pd.to_datetime(ledger["fill_date"])
    positions = np.flatnonzero((dates >= start) & (dates <= end))
    if not len(positions):
        raise IntradayExhaustionUnionError("Ledger lacks a reporting period")
    first = int(positions[0])
    last = int(positions[-1])
    start_equity = INITIAL_CASH if first == 0 else float(ledger.iloc[first - 1]["equity"])
    return float(ledger.iloc[last]["equity"] / start_equity - 1.0)


def _positive_share(values: list[float]) -> float | None:
    positive = np.asarray([value for value in values if value > 0.0], dtype=float)
    if not len(positive):
        return None
    return float(np.max(positive) / np.sum(positive))


def _policy_summary(
    ledger: pd.DataFrame,
    benchmark: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    comparison = compare_ledgers(ledger, benchmark, initial_cash=INITIAL_CASH)
    total_edge = float(
        math.log1p(comparison["strategy"]["total_return"])
        - math.log1p(comparison["aapl_buy_hold"]["total_return"])
    )
    attributed = float(episodes["net_active_log_edge"].sum())
    identity_error = total_edge - attributed
    if abs(identity_error) > 1e-10:
        raise IntradayExhaustionUnionError("Episode edges do not reconcile to ledger")
    annual: dict[str, Any] = {}
    for year in range(2005, 2019):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        edge = float(
            episodes.loc[
                pd.to_datetime(episodes["entry_date"]).dt.year == year,
                "net_active_log_edge",
            ].sum()
        )
        annual[str(year)] = {
            "strategy_return": _period_return(ledger, start, end),
            "aapl_buy_hold_return": _period_return(benchmark, start, end),
            "active_log_edge": edge,
        }
    edges = episodes["net_active_log_edge"].to_numpy(dtype=float)
    annual_values = [float(annual[str(year)]["active_log_edge"]) for year in range(2005, 2019)]
    folds = {
        f"{year}-{year + 1}": float(annual[str(year)]["active_log_edge"] + annual[str(year + 1)]["active_log_edge"])
        for year in range(2005, 2019, 2)
    }
    return {
        "comparison": comparison,
        "total_active_log_edge": total_edge,
        "attributed_episode_active_log_edge": attributed,
        "episode_ledger_identity_error": identity_error,
        "relative_ending_wealth": float(comparison["relative_wealth_vs_aapl_buy_hold"]),
        "annual": annual,
        "positive_year_count": int(sum(value > 0.0 for value in annual_values)),
        "edge_after_best_year_removed": float(total_edge - max(annual_values)),
        "maximum_positive_year_share": _positive_share(annual_values),
        "fold_edges": folds,
        "positive_fold_count": int(sum(value > 0.0 for value in folds.values())),
        "cash_episode_count": int(len(episodes)),
        "cash_episode_win_rate": float(np.mean(edges > 0.0)),
        "mean_cash_episode_edge": float(np.mean(edges)),
        "median_cash_episode_edge": float(np.median(edges)),
        "maximum_positive_episode_share": _positive_share(edges.tolist()),
        "negative_aapl_year_edges": {
            str(year): float(annual[str(year)]["active_log_edge"])
            for year in NEGATIVE_AAPL_YEARS
        },
        "negative_aapl_year_aggregate_edge": float(
            sum(annual[str(year)]["active_log_edge"] for year in NEGATIVE_AAPL_YEARS)
        ),
        "negative_aapl_year_positive_count": int(
            sum(annual[str(year)]["active_log_edge"] > 0.0 for year in NEGATIVE_AAPL_YEARS)
        ),
        "no_leverage_proof": assert_unleveraged_ledger(ledger),
    }


def evaluate_development(
    frame: pd.DataFrame, cash_signal: pd.Series
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, pd.DataFrame]],
    dict[str, dict[str, pd.DataFrame]],
]:
    data = canonical_context_frame(frame)
    union_target = pd.Series(
        np.where(cash_signal.reindex(data.index), 0.0, 1.0),
        index=data.index,
        dtype=float,
    )
    union_target.loc[union_target.index < ACCOUNT_START] = 1.0
    always_long = pd.Series(1.0, index=data.index, dtype=float)
    full = EvaluationPeriod("development", "2005-01-01", "2018-12-31")
    metrics: dict[str, Any] = {}
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    episodes: dict[str, dict[str, pd.DataFrame]] = {}
    for cost_name, cost_bps in COST_SCENARIOS:
        costs = CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0)
        candidate, events = simulate_intraday_only(
            data, cash_signal, cost_bps=cost_bps
        )
        union = simulate_unleveraged_period(
            data, union_target, full, costs, initial_cash=INITIAL_CASH
        )
        benchmark = simulate_unleveraged_period(
            data, always_long, full, costs, initial_cash=INITIAL_CASH
        )
        if candidate["fill_date"].tolist() != benchmark["fill_date"].tolist():
            raise IntradayExhaustionUnionError("Candidate and benchmark dates differ")
        candidate_episodes = _episode_rows(
            data, cash_signal, cost_bps=cost_bps, mode="intraday"
        )
        union_episodes = _episode_rows(
            data, cash_signal, cost_bps=cost_bps, mode="open_to_open"
        )
        if candidate_episodes["decision_date"].tolist() != union_episodes["decision_date"].tolist():
            raise IntradayExhaustionUnionError("Candidate and union signals differ")
        candidate_summary = _policy_summary(candidate, benchmark, candidate_episodes)
        union_summary = _policy_summary(union, benchmark, union_episodes)
        incremental_folds = {
            name: float(candidate_summary["fold_edges"][name] - union_summary["fold_edges"][name])
            for name in candidate_summary["fold_edges"]
        }
        incremental_total = float(
            candidate_summary["total_active_log_edge"]
            - union_summary["total_active_log_edge"]
        )
        if abs(incremental_total - sum(incremental_folds.values())) > 1e-10:
            raise IntradayExhaustionUnionError("Incremental fold edges do not reconcile")
        metrics[cost_name] = {
            "candidate": candidate_summary,
            "open_to_open_union": union_summary,
            "candidate_vs_union": {
                "total_incremental_active_log_edge": incremental_total,
                "incremental_fold_edges": incremental_folds,
                "positive_incremental_fold_count": int(
                    sum(value > 0.0 for value in incremental_folds.values())
                ),
                "incremental_after_best_fold_removed": float(
                    incremental_total - max(incremental_folds.values())
                ),
            },
        }
        ledgers[cost_name] = {
            "candidate_open": candidate,
            "candidate_events": events,
            "open_to_open_union": union,
            "aapl_buy_hold": benchmark,
        }
        episodes[cost_name] = {
            "candidate": candidate_episodes,
            "open_to_open_union": union_episodes,
        }
    metrics["integrity"] = {
        "candidate_and_union_signal_dates_identical": True,
        "candidate_and_benchmark_open_dates_identical": True,
        "candidate_episode_count": int(cash_signal.sum()),
        "physical_later_rows_opened": False,
        "network_calls": 0,
        "api_calls": 0,
        "llm_calls": 0,
        "broker_actions": 0,
        "real_money_actions": 0,
    }
    return metrics, ledgers, episodes


def apply_development_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    gates: dict[str, bool] = {}
    for cost_name, _ in COST_SCENARIOS:
        candidate = metrics[cost_name]["candidate"]
        incremental = metrics[cost_name]["candidate_vs_union"]
        suffix = "5bps" if cost_name == "base_5bps" else "10bps"
        gates[f"candidate_edge_positive_{suffix}"] = candidate["total_active_log_edge"] > 0.0
        gates[f"candidate_relative_wealth_positive_{suffix}"] = candidate["relative_ending_wealth"] > 0.0
        gates[f"candidate_beats_union_by_0001_{suffix}"] = incremental["total_incremental_active_log_edge"] > 0.0001
        gates[f"at_least_8_positive_years_{suffix}"] = candidate["positive_year_count"] >= 8
        gates[f"at_least_4_positive_folds_{suffix}"] = candidate["positive_fold_count"] >= 4
        gates[f"at_least_4_positive_incremental_folds_{suffix}"] = incremental["positive_incremental_fold_count"] >= 4
        gates[f"incremental_positive_without_best_fold_{suffix}"] = incremental["incremental_after_best_fold_removed"] > 0.0
        gates[f"edge_positive_without_best_year_{suffix}"] = candidate["edge_after_best_year_removed"] > 0.0
        gates[f"no_year_over_half_positive_edge_{suffix}"] = (
            candidate["maximum_positive_year_share"] is not None
            and candidate["maximum_positive_year_share"] <= 0.5
        )
        gates[f"unleveraged_and_nonnegative_{suffix}"] = candidate["no_leverage_proof"]["passed"] is True
    stress = metrics["stress_10bps"]["candidate"]
    gates.update(
        {
            "stress_negative_aapl_years_aggregate_positive": stress["negative_aapl_year_aggregate_edge"] > 0.0,
            "stress_at_least_2_of_3_negative_aapl_years_positive": stress["negative_aapl_year_positive_count"] >= 2,
            "exactly_121_candidate_episodes": stress["cash_episode_count"] == EXPECTED_EPISODES,
            "stress_episode_win_rate_at_least_55pct": stress["cash_episode_win_rate"] >= 0.55,
            "stress_mean_episode_edge_positive": stress["mean_cash_episode_edge"] > 0.0,
            "stress_median_episode_edge_positive": stress["median_cash_episode_edge"] > 0.0,
            "stress_no_episode_over_half_positive_edge": stress["maximum_positive_episode_share"] is not None
            and stress["maximum_positive_episode_share"] <= 0.5,
            "candidate_and_union_signal_dates_identical": metrics["integrity"]["candidate_and_union_signal_dates_identical"] is True,
            "no_later_data_or_external_actions": all(
                metrics["integrity"][name] in (False, 0)
                for name in (
                    "physical_later_rows_opened",
                    "network_calls",
                    "api_calls",
                    "llm_calls",
                    "broker_actions",
                    "real_money_actions",
                )
            ),
        }
    )
    failures = [name for name, passed in gates.items() if not passed]
    return {"passed": not failures, "failures": failures, "gates": gates}


def _publish_payloads(destination: Path, run_id: str, payloads: Mapping[str, bytes]) -> Path:
    final_dir = destination / run_id
    temporary = destination / f".{run_id}.pending"
    if final_dir.exists() or temporary.exists():
        raise IntradayExhaustionUnionError("Run destination already exists")
    destination.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, payload in payloads.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        temporary.rename(final_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_dir


def run_development(
    *, repo_root: Path, price_artifact: Path, output_dir: Path, run_id: str
) -> dict[str, Any]:
    started = time.monotonic()
    root = repo_root.resolve()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise IntradayExhaustionUnionError("Invalid run_id")
    source = price_artifact if price_artifact.is_absolute() else root / price_artifact
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    if source.stat().st_size != INPUT_BYTE_COUNT or _file_sha256(source) != INPUT_LITERAL_SHA256:
        raise IntradayExhaustionUnionError("Input bytes do not match preregistration")
    expert_source = root / "agent_benchmark/chronological_exhaustion_expert.py"
    if (
        _git_blob_oid(expert_source) != EXPERT_SOURCE_GIT_BLOB
        or _file_sha256(expert_source) != EXPERT_SOURCE_LITERAL_SHA256
    ):
        raise IntradayExhaustionUnionError("Inherited signal source changed")
    frame, provenance = load_bounded_prices(
        source, end=INPUT_LAST, required_last_session=INPUT_LAST
    )
    if len(frame) != INPUT_ROWS or frame.index.min() != INPUT_FIRST:
        raise IntradayExhaustionUnionError("Input coverage does not match preregistration")
    cash_signal = build_account_union_signal(frame)
    decision_dates_sha256 = _decision_dates_sha256(cash_signal)
    if (
        int(cash_signal.sum()) != EXPECTED_EPISODES
        or decision_dates_sha256 != EXPECTED_DECISION_DATES_SHA256
    ):
        raise IntradayExhaustionUnionError("Union opportunity identity changed")
    metrics, ledgers, episodes = evaluate_development(frame, cash_signal)
    gate_report = apply_development_gates(metrics)
    runtime_seconds = float(time.monotonic() - started)
    forecast = pd.DataFrame(
        {
            "decision_date": frame.index.date.astype(str),
            "union_cash_signal": cash_signal.to_numpy(dtype=bool),
            "intraday_target_at_next_open": np.where(cash_signal, 0.0, 1.0),
            "precommitted_same_day_close_target": np.where(cash_signal, 1.0, np.nan),
        }
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": "repeated_historical_development_diagnostic",
        "globally_unseen_holdout": False,
        "period": {"start": "2005-01-01", "end": "2018-12-31"},
        "input_provenance": provenance,
        "signal_identity": {
            "expert_source_git_blob": EXPERT_SOURCE_GIT_BLOB,
            "expert_source_literal_sha256": EXPERT_SOURCE_LITERAL_SHA256,
            "decision_count": int(cash_signal.sum()),
            "decision_dates_sha256": decision_dates_sha256,
        },
        "trading_rule": {
            "signal": "unchanged fixed contextual-plus-weak-trend union",
            "sell": "adjusted open t+1",
            "buyback": "precommitted adjusted close t+1",
            "decision_uses_t_plus_1_value": False,
            "exposure_values": [0.0, 1.0],
        },
        "metrics": metrics,
        "development_gate_report": gate_report,
        "runtime": {
            "seconds": runtime_seconds,
            "network_calls": 0,
            "paid_api_calls": 0,
            "llm_calls": 0,
            "broker_actions": 0,
        },
        "later_period_opened": False,
        "real_money_authorized": False,
    }
    payloads: dict[str, bytes] = {
        "report.json": _json_bytes(report, pretty=True),
        "metrics.json": _json_bytes(metrics, pretty=True),
        "gate_report.json": _json_bytes(gate_report, pretty=True),
        "development_forecast.csv": _frame_csv_bytes(forecast),
    }
    for cost_name, _ in COST_SCENARIOS:
        for name, ledger in ledgers[cost_name].items():
            payloads[f"{cost_name}_{name}_ledger.csv"] = _frame_csv_bytes(ledger)
        for name, episode_frame in episodes[cost_name].items():
            payloads[f"{cost_name}_{name}_episodes.csv"] = _frame_csv_bytes(episode_frame)
    checksums = {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in sorted(payloads.items())
    }
    payloads["checksums.json"] = _json_bytes(checksums, pretty=True)
    final_dir = _publish_payloads(destination.resolve(), run_id, payloads)
    report["artifact_dir"] = str(final_dir)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--price-artifact",
        type=Path,
        default=Path(
            "e/chronological_exhaustion_expert_v1/authorized_inputs/"
            "aapl_spy_qqq_through_2018.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("e/intraday_exhaustion_union_v1")
    )
    parser.add_argument(
        "--run-id", default="intraday-exhaustion-union-development-v1"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_development(
        repo_root=args.repo_root,
        price_artifact=args.price_artifact,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "passed": report["development_gate_report"]["passed"],
                "failures": report["development_gate_report"]["failures"],
                "runtime_seconds": report["runtime"]["seconds"],
                "later_period_opened": report["later_period_opened"],
                "artifact_dir": report["artifact_dir"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
