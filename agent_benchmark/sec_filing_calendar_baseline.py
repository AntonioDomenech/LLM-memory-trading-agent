"""Simple 2018-bounded AAPL cash rule after authenticated SEC filings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .chronological_exhaustion_experiment import load_bounded_prices
from .deterministic_aapl import CostAssumptions, EvaluationPeriod, compare_ledgers
from .unleveraged_aapl import (
    assert_unleveraged_ledger,
    canonical_context_frame,
    simulate_unleveraged_period,
)


CONTRACT_VERSION = "aapl-sec-event-baseline-v1"
DEVELOPMENT_START = pd.Timestamp("2000-01-03")
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
INPUT_FIRST = pd.Timestamp("1999-03-10")
INPUT_ROWS = 4_986
INPUT_BYTE_COUNT = 590_761
INPUT_LITERAL_SHA256 = (
    "9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1"
)
MANIFEST_ROWS = 75
MANIFEST_BYTE_COUNT = 30_677
MANIFEST_SHA256 = (
    "eea1ff57f6f9f2db31ee341fb48494d81b7c56e9e6bf889278b955b5ff15dabb"
)
CASH_SESSIONS = 20
INITIAL_CASH = 1_000.0
COST_SCENARIOS = (("base_5bps", 5.0), ("stress_10bps", 10.0))
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")


class SecFilingCalendarBaselineError(RuntimeError):
    """The frozen baseline contract could not be evaluated safely."""


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=separators,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def load_filing_manifest(receipt_dir: Path) -> tuple[list[dict[str, Any]], bytes]:
    """Read only public-safe timing fields from the authenticated receipts."""

    source = receipt_dir.resolve()
    if not source.is_dir():
        raise SecFilingCalendarBaselineError("Private SEC receipt directory is absent")
    rows: list[dict[str, Any]] = []
    try:
        paths = sorted(source.glob("*.json"))
        for path in paths:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            seal_row = receipt.get("source_evidence", {}).get("seal_row")
            if not isinstance(seal_row, dict):
                continue
            available = seal_row.get("availability_session")
            if not isinstance(available, str) or not (
                "2000-01-01" <= available <= "2018-12-31"
            ):
                continue
            if (
                receipt.get("stage") != "development"
                or seal_row.get("acquisition_stage") != "development"
                or seal_row.get("stage_assignment") != "development"
                or seal_row.get("form") not in {"10-K", "10-Q"}
                or seal_row.get("exact_acceptance") is not True
            ):
                continue
            row = {
                "accession_number": str(seal_row["accession_number"]),
                "acquisition_stage": str(seal_row["acquisition_stage"]),
                "availability_session": available,
                "exact_acceptance": True,
                "filing_date": str(seal_row["filing_date"]),
                "form": str(seal_row["form"]),
                "frozen_prefix_sha256": str(seal_row["frozen_prefix_sha256"]),
                "sequence": int(receipt["sequence"]),
                "source_evidence_sha256": str(receipt["source_evidence_sha256"]),
                "stage_assignment": str(seal_row["stage_assignment"]),
            }
            rows.append(row)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SecFilingCalendarBaselineError("SEC receipt metadata is malformed") from exc

    rows.sort(key=lambda row: (row["availability_session"], row["accession_number"]))
    payload = _json_bytes(rows)
    if (
        len(rows) != MANIFEST_ROWS
        or len(payload) != MANIFEST_BYTE_COUNT
        or _sha256(payload) != MANIFEST_SHA256
    ):
        raise SecFilingCalendarBaselineError(
            "SEC timing manifest does not match the pushed preregistration"
        )
    if sorted(row["sequence"] for row in rows) != list(range(124, 199)):
        raise SecFilingCalendarBaselineError("SEC receipt sequence is incomplete")
    if sum(row["form"] == "10-K" for row in rows) != 18:
        raise SecFilingCalendarBaselineError("SEC 10-K count changed")
    if sum(row["form"] == "10-Q" for row in rows) != 57:
        raise SecFilingCalendarBaselineError("SEC 10-Q count changed")
    if (
        rows[0]["availability_session"] != "2000-02-02"
        or rows[-1]["availability_session"] != "2018-11-06"
    ):
        raise SecFilingCalendarBaselineError("SEC availability coverage changed")
    for field in (
        "accession_number",
        "frozen_prefix_sha256",
        "source_evidence_sha256",
    ):
        values = [row[field] for row in rows]
        if len(values) != len(set(values)):
            raise SecFilingCalendarBaselineError(f"SEC {field} is not unique")
    if any(
        pd.Timestamp(row["availability_session"]) <= pd.Timestamp(row["filing_date"])
        for row in rows
    ):
        raise SecFilingCalendarBaselineError(
            "A filing is not strictly later-safe at availability_session"
        )
    return rows, payload


def build_filing_target(
    frame: pd.DataFrame, rows: Sequence[Mapping[str, Any]]
) -> tuple[pd.Series, pd.DataFrame]:
    """Create exact non-overlapping t+1 through t+21 cash episodes."""

    data = canonical_context_frame(frame)
    target = pd.Series(1.0, index=data.index, name="target_exposure", dtype=float)
    position_by_date = {day.date().isoformat(): index for index, day in enumerate(data.index)}
    blocked_through = -1
    audit: list[dict[str, Any]] = []
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["availability_session"]),
            str(row["accession_number"]),
        ),
    )
    for row in ordered:
        available = str(row["availability_session"])
        if available not in position_by_date:
            raise SecFilingCalendarBaselineError(
                f"SEC availability session is absent from market data: {available}"
            )
        position = position_by_date[available]
        scheduled = position > blocked_through
        reason = "scheduled" if scheduled else "active_or_pending_episode"
        entry_position: int | None = None
        exit_position: int | None = None
        if scheduled:
            if position + 1 >= len(data):
                raise SecFilingCalendarBaselineError("A filing has no causal next-open fill")
            entry_position = position + 1
            exit_position = position + CASH_SESSIONS + 1
            target.iloc[position : min(position + CASH_SESSIONS, len(target))] = 0.0
            # At close t+20 the account is still cash until its t+21 buy, so a
            # filing on that close cannot extend the inherited episode.
            blocked_through = position + CASH_SESSIONS
        audit.append(
            {
                "sequence": int(row["sequence"]),
                "accession_number": str(row["accession_number"]),
                "form": str(row["form"]),
                "filing_date": str(row["filing_date"]),
                "availability_session": available,
                "scheduled": bool(scheduled),
                "reason": reason,
                "entry_open": (
                    data.index[entry_position].date().isoformat()
                    if entry_position is not None
                    else None
                ),
                "exit_open": (
                    data.index[exit_position].date().isoformat()
                    if exit_position is not None and exit_position < len(data)
                    else None
                ),
                "complete_by_2018": bool(
                    exit_position is not None and exit_position < len(data)
                ),
            }
        )
    values = target.to_numpy(dtype=float)
    if not np.all((values == 0.0) | (values == 1.0)):
        raise SecFilingCalendarBaselineError("SEC target is not exact LONG/CASH")
    return target, pd.DataFrame(audit)


def _period_return(ledger: pd.DataFrame, start: str, end: str) -> float:
    dates = pd.to_datetime(ledger["fill_date"])
    positions = np.flatnonzero(
        (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    )
    if not len(positions):
        raise SecFilingCalendarBaselineError(f"Ledger lacks {start} through {end}")
    first, last = int(positions[0]), int(positions[-1])
    initial = INITIAL_CASH if first == 0 else float(ledger.iloc[first - 1]["equity"])
    return float(ledger.iloc[last]["equity"] / initial - 1.0)


def _annual_metrics(
    strategy: pd.DataFrame, benchmark: pd.DataFrame
) -> dict[str, Any]:
    years: dict[str, dict[str, float]] = {}
    for year in range(2000, 2019):
        strategy_return = _period_return(
            strategy, f"{year}-01-01", f"{year}-12-31"
        )
        benchmark_return = _period_return(
            benchmark, f"{year}-01-01", f"{year}-12-31"
        )
        years[str(year)] = {
            "strategy_return": strategy_return,
            "aapl_buy_hold_return": benchmark_return,
            "excess_return": float(strategy_return - benchmark_return),
            "active_log_edge": float(
                math.log1p(strategy_return) - math.log1p(benchmark_return)
            ),
        }
    edges = np.asarray(
        [row["active_log_edge"] for row in years.values()], dtype=float
    )
    excess = np.asarray([row["excess_return"] for row in years.values()], dtype=float)
    total_edge = float(np.sum(edges))
    negative_years = {
        year: row
        for year, row in years.items()
        if row["aapl_buy_hold_return"] < 0.0
    }
    return {
        "years": years,
        "winning_years": int(np.sum(excess > 1e-12)),
        "losing_years": int(np.sum(excess < -1e-12)),
        "tied_years": int(np.sum(np.abs(excess) <= 1e-12)),
        "mean_annual_excess": float(np.mean(excess)),
        "median_annual_excess": float(np.median(excess)),
        "total_active_log_edge": total_edge,
        "active_log_edge_after_best_year_removed": float(total_edge - np.max(edges)),
        "negative_aapl_years": negative_years,
        "negative_aapl_year_active_log_edge": float(
            np.sum([row["active_log_edge"] for row in negative_years.values()])
        ),
    }


def _episode_metrics(
    frame: pd.DataFrame, schedule: pd.DataFrame, cost_bps: float
) -> dict[str, Any]:
    data = canonical_context_frame(frame)
    cost = float(cost_bps) / 10_000.0
    friction = math.log((1.0 - cost) / (1.0 + cost))
    rows: list[dict[str, Any]] = []
    for event in schedule.to_dict(orient="records"):
        if not event["scheduled"] or not event["complete_by_2018"]:
            continue
        entry = pd.Timestamp(event["entry_open"])
        exit_day = pd.Timestamp(event["exit_open"])
        entry_open = float(data.loc[entry, "aapl_adj_open"])
        exit_open = float(data.loc[exit_day, "aapl_adj_open"])
        gross_aapl_log_return = float(math.log(exit_open / entry_open))
        rows.append(
            {
                "sequence": int(event["sequence"]),
                "entry_open": event["entry_open"],
                "exit_open": event["exit_open"],
                "gross_aapl_log_return": gross_aapl_log_return,
                "net_cash_active_log_edge": float(friction - gross_aapl_log_return),
            }
        )
    edges = np.asarray([row["net_cash_active_log_edge"] for row in rows], dtype=float)
    return {
        "rows": rows,
        "completed_episode_count": int(len(rows)),
        "win_rate": float(np.mean(edges > 0.0)) if len(edges) else None,
        "mean_active_log_edge": float(np.mean(edges)) if len(edges) else None,
        "median_active_log_edge": float(np.median(edges)) if len(edges) else None,
        "total_active_log_edge": float(np.sum(edges)),
        "active_log_edge_after_best_episode_removed": (
            float(np.sum(edges) - np.max(edges)) if len(edges) else None
        ),
    }


def evaluate_baseline(
    frame: pd.DataFrame, target: pd.Series, schedule: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    data = canonical_context_frame(frame)
    period = EvaluationPeriod(
        "development", DEVELOPMENT_START.date().isoformat(), DEVELOPMENT_END.date().isoformat()
    )
    benchmark_target = pd.Series(1.0, index=data.index, dtype=float)
    metrics: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    episode_rows: dict[str, Any] = {}
    for cost_name, cost_bps in COST_SCENARIOS:
        costs = CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0)
        strategy = simulate_unleveraged_period(
            data, target, period, costs, initial_cash=INITIAL_CASH
        )
        benchmark = simulate_unleveraged_period(
            data, benchmark_target, period, costs, initial_cash=INITIAL_CASH
        )
        always_long = simulate_unleveraged_period(
            data, benchmark_target, period, costs, initial_cash=INITIAL_CASH
        )
        if not always_long.equals(benchmark):
            raise SecFilingCalendarBaselineError(
                "Always-long control differs from same-ledger AAPL"
            )
        comparison = compare_ledgers(strategy, benchmark, initial_cash=INITIAL_CASH)
        annual = _annual_metrics(strategy, benchmark)
        episodes = _episode_metrics(data, schedule, cost_bps)
        continuous_edge = float(
            math.log1p(comparison["strategy"]["total_return"])
            - math.log1p(comparison["aapl_buy_hold"]["total_return"])
        )
        if abs(continuous_edge - annual["total_active_log_edge"]) > 1e-10:
            raise SecFilingCalendarBaselineError("Annual edges do not reconcile")
        if abs(continuous_edge - episodes["total_active_log_edge"]) > 1e-10:
            raise SecFilingCalendarBaselineError("Episode edges do not reconcile")
        metrics[cost_name] = {
            "cost_bps_per_changing_leg": cost_bps,
            "comparison": comparison,
            "annual": annual,
            "episodes": {key: value for key, value in episodes.items() if key != "rows"},
            "continuous_active_log_edge": continuous_edge,
            "no_leverage_proof": assert_unleveraged_ledger(strategy),
            "always_long_equals_aapl": True,
        }
        combined = pd.concat(
            [strategy.add_prefix("strategy_"), benchmark.add_prefix("aapl_")], axis=1
        )
        ledgers[cost_name] = combined
        episode_rows[cost_name] = episodes["rows"]
    return metrics, ledgers, episode_rows


def apply_development_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    base = metrics["base_5bps"]
    stress = metrics["stress_10bps"]
    gates = {
        "all_ledgers_unleveraged_and_nonnegative": all(
            metrics[name]["no_leverage_proof"]["passed"]
            for name, _ in COST_SCENARIOS
        ),
        "always_long_matches_same_ledger_aapl": all(
            metrics[name]["always_long_equals_aapl"] for name, _ in COST_SCENARIOS
        ),
        "base_cumulative_relative_wealth_positive": (
            base["comparison"]["relative_wealth_vs_aapl_buy_hold"] > 0.0
        ),
        "stress_cumulative_relative_wealth_positive": (
            stress["comparison"]["relative_wealth_vs_aapl_buy_hold"] > 0.0
        ),
        "stress_winning_years_outnumber_losing_years": (
            stress["annual"]["winning_years"] > stress["annual"]["losing_years"]
        ),
        "stress_mean_annual_excess_positive": (
            stress["annual"]["mean_annual_excess"] > 0.0
        ),
        "stress_median_annual_excess_positive": (
            stress["annual"]["median_annual_excess"] > 0.0
        ),
        "stress_edge_positive_after_best_year_removed": (
            stress["annual"]["active_log_edge_after_best_year_removed"] > 0.0
        ),
        "stress_negative_aapl_year_edge_positive": (
            stress["annual"]["negative_aapl_year_active_log_edge"] > 0.0
        ),
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {"passed": not failures, "failures": failures, "gates": gates}


def run_development(
    *,
    repo_root: Path,
    receipt_dir: Path,
    price_artifact: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    started = time.monotonic()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SecFilingCalendarBaselineError("Invalid run_id")
    root = repo_root.resolve()
    source = price_artifact if price_artifact.is_absolute() else root / price_artifact
    receipts = receipt_dir if receipt_dir.is_absolute() else root / receipt_dir
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    final_dir = destination.resolve() / run_id
    temporary = destination.resolve() / f".{run_id}.pending"
    if final_dir.exists() or temporary.exists():
        raise SecFilingCalendarBaselineError("Run destination already exists")
    if source.stat().st_size != INPUT_BYTE_COUNT or _file_sha256(source) != INPUT_LITERAL_SHA256:
        raise SecFilingCalendarBaselineError("Market bytes do not match preregistration")

    manifest, manifest_payload = load_filing_manifest(receipts)
    frame, provenance = load_bounded_prices(
        source, end=DEVELOPMENT_END, required_last_session=DEVELOPMENT_END
    )
    if len(frame) != INPUT_ROWS or frame.index.min() != INPUT_FIRST:
        raise SecFilingCalendarBaselineError("Market coverage does not match preregistration")
    target, schedule = build_filing_target(frame, manifest)
    metrics, ledgers, episode_rows = evaluate_baseline(frame, target, schedule)
    gate_report = apply_development_gates(metrics)
    runtime_seconds = float(time.monotonic() - started)
    report = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": "repeated_historical_development",
        "globally_unseen_holdout": False,
        "period": {
            "start": DEVELOPMENT_START.date().isoformat(),
            "end": DEVELOPMENT_END.date().isoformat(),
        },
        "rule": {
            "cash_sessions": CASH_SESSIONS,
            "entry": "next adjusted open after availability-session close",
            "exit": "adjusted open t+21",
            "overlap": "later filing audited but cannot extend active or pending episode",
        },
        "filing_manifest": {
            "rows": len(manifest),
            "unique_availability_sessions": len(
                {row["availability_session"] for row in manifest}
            ),
            "bytes": len(manifest_payload),
            "sha256": _sha256(manifest_payload),
            "private_contact_published": False,
        },
        "schedule": {
            "scheduled_episodes": int(schedule["scheduled"].sum()),
            "suppressed_rows": int((~schedule["scheduled"]).sum()),
            "completed_episodes": int(schedule["complete_by_2018"].sum()),
        },
        "input_provenance": provenance,
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
    target_frame = pd.DataFrame(
        {
            "date": frame.index.date.astype(str),
            "target_exposure": target.to_numpy(dtype=float),
        }
    )
    payloads: dict[str, bytes] = {
        "report.json": _json_bytes(report, pretty=True),
        "filing_manifest.json": manifest_payload,
        "filing_schedule.csv": _csv_bytes(schedule),
        "target_history.csv": _csv_bytes(target_frame),
        "metrics.json": _json_bytes(metrics, pretty=True),
        "gate_report.json": _json_bytes(gate_report, pretty=True),
    }
    for cost_name, _ in COST_SCENARIOS:
        payloads[f"{cost_name}_ledgers.csv"] = _csv_bytes(ledgers[cost_name])
        payloads[f"{cost_name}_episodes.json"] = _json_bytes(
            episode_rows[cost_name], pretty=True
        )
    checksums = {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in sorted(payloads.items())
    }
    payloads["checksums.json"] = _json_bytes(checksums, pretty=True)
    destination.resolve().mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for name, payload in payloads.items():
            (temporary / name).write_bytes(payload)
        temporary.rename(final_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "run_id": run_id,
        "artifact_dir": str(final_dir),
        "passed": bool(gate_report["passed"]),
        "failures": list(gate_report["failures"]),
        "runtime_seconds": runtime_seconds,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("develop",))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--price-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_development(
        repo_root=args.repo_root,
        receipt_dir=args.receipt_dir,
        price_artifact=args.price_artifact,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SecFilingCalendarBaselineError",
    "apply_development_gates",
    "build_filing_target",
    "evaluate_baseline",
    "load_filing_manifest",
    "main",
    "run_development",
]
