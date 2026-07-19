"""Cheap 2018-bounded AAPL regime/expert disagreement diagnostic."""

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

from .binary_regime_union_selector_experiment import _continuous_policy_result
from .chronological_exhaustion_experiment import load_bounded_prices
from .chronological_exhaustion_expert import (
    build_fixed_expert_signals,
    canonicalize_one_session_signals,
)
from .deterministic_aapl import EvaluationPeriod


CONTRACT_VERSION = "aapl-regime-expert-disagreement-v1"
INPUT_LITERAL_SHA256 = (
    "9e661722b9e474121654b348b44e646af6e5b2f94b1c613690113da2ad4ffdc1"
)
INPUT_BYTE_COUNT = 590_761
INPUT_ROWS = 4_986
INPUT_FIRST = pd.Timestamp("1999-03-10")
INPUT_LAST = pd.Timestamp("2018-12-31")
EXPERT_SOURCE_GIT_BLOB = "3d01affe636f2917306358a7d80f285b5a10f1d4"
ACCOUNT_START = pd.Timestamp("2005-01-01")
CALIBRATION_END = pd.Timestamp("2011-12-31")
EVALUATION_START = pd.Timestamp("2012-01-01")
EVALUATION_END = pd.Timestamp("2018-12-31")
CALIBRATION_COST_BPS = 10.0
COST_SCENARIOS = (("base_5bps", 5.0), ("stress_10bps", 10.0))
REGIMES = ("risk_on", "not_risk_on")
EXPERTS = ("contextual_only", "weak_trend_only")
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")


class RegimeExpertDisagreementError(RuntimeError):
    """A fail-closed experiment error."""


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    text = json.dumps(
        value,
        sort_keys=True,
        indent=2 if pretty else None,
        separators=separators,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


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
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git identity


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _risk_regime(frame: pd.DataFrame) -> pd.DataFrame:
    spy_return = frame["spy_adj_close"].pct_change(20, fill_method=None)
    qqq_return = frame["qqq_adj_close"].pct_change(20, fill_method=None)
    ready = spy_return.notna() & qqq_return.notna()
    ready &= np.isfinite(spy_return) & np.isfinite(qqq_return)
    risk_on = ready & (spy_return > 0.0) & (qqq_return > 0.0)
    return pd.DataFrame(
        {
            "spy_return_20": spy_return,
            "qqq_return_20": qqq_return,
            "regime_ready": ready.astype(bool),
            "risk_on": risk_on.astype(bool),
        },
        index=frame.index,
    )


def _event_membership(signals: pd.DataFrame) -> pd.DataFrame:
    union = signals["unfiltered_union_signal"].astype(bool)
    contextual = signals["contextual_virtual_signal"].astype(bool)
    weak = signals["weak_trend_virtual_signal"].astype(bool)
    if bool((union & ~(contextual | weak)).any()):
        raise RegimeExpertDisagreementError(
            "A canonical union event has no contributing expert"
        )
    return pd.DataFrame(
        {
            "union": union,
            "both": union & contextual & weak,
            "contextual_only": union & contextual & ~weak,
            "weak_trend_only": union & weak & ~contextual,
        },
        index=signals.index,
    )


def _edge_friction(cost_bps: float) -> float:
    cost = float(cost_bps) / 10_000.0
    return math.log((1.0 - cost) / (1.0 + cost))


def _calibration_rows(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    regime: pd.DataFrame,
) -> list[dict[str, Any]]:
    dates = frame.index
    opens = frame["aapl_adj_open"].to_numpy(dtype=float)
    union_positions = np.flatnonzero(membership["union"].to_numpy(dtype=bool))
    rows: list[dict[str, Any]] = []
    friction = _edge_friction(CALIBRATION_COST_BPS)
    for position in union_positions:
        if position + 2 >= len(frame):
            continue
        entry = dates[position + 1]
        maturity = dates[position + 2]
        if entry < ACCOUNT_START or maturity > CALIBRATION_END:
            continue
        if not bool(regime.iloc[position]["regime_ready"]):
            raise RegimeExpertDisagreementError(
                "A calibration union event lacks a finite 20-session regime"
            )
        if bool(membership.iloc[position]["both"]):
            expert = "both"
        elif bool(membership.iloc[position]["contextual_only"]):
            expert = "contextual_only"
        elif bool(membership.iloc[position]["weak_trend_only"]):
            expert = "weak_trend_only"
        else:  # pragma: no cover - guarded by _event_membership
            raise RegimeExpertDisagreementError("Unknown union membership")
        rows.append(
            {
                "decision_date": dates[position].date().isoformat(),
                "entry_date": entry.date().isoformat(),
                "maturity_date": maturity.date().isoformat(),
                "entry_year": int(entry.year),
                "regime": (
                    "risk_on"
                    if bool(regime.iloc[position]["risk_on"])
                    else "not_risk_on"
                ),
                "expert_membership": expert,
                "net_cash_log_edge_10bps": float(
                    math.log(opens[position + 1] / opens[position + 2]) + friction
                ),
            }
        )
    return rows


def build_calibration_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str | None]]:
    diagnostics: dict[str, Any] = {}
    choices: dict[str, str | None] = {}
    for regime_name in REGIMES:
        diagnostics[regime_name] = {}
        eligible: list[tuple[float, str]] = []
        for expert in EXPERTS:
            selected = [
                row
                for row in rows
                if row["regime"] == regime_name
                and row["expert_membership"] == expert
            ]
            values = np.asarray(
                [float(row["net_cash_log_edge_10bps"]) for row in selected],
                dtype=float,
            )
            yearly = {
                str(year): float(
                    np.sum(
                        [
                            float(row["net_cash_log_edge_10bps"])
                            for row in selected
                            if int(row["entry_year"]) == year
                        ]
                    )
                )
                for year in range(2005, 2012)
            }
            positive_years = sum(value > 0.0 for value in yearly.values())
            total = float(np.sum(values)) if len(values) else 0.0
            mean = float(np.mean(values)) if len(values) else None
            median = float(np.median(values)) if len(values) else None
            is_eligible = bool(
                len(values) >= 5 and total > 0.0 and positive_years >= 3
            )
            diagnostics[regime_name][expert] = {
                "episode_count": int(len(values)),
                "total_net_cash_log_edge_10bps": total,
                "mean_net_cash_log_edge_10bps": mean,
                "median_net_cash_log_edge_10bps": median,
                "positive_calendar_year_count": int(positive_years),
                "yearly_net_cash_log_edges_10bps": yearly,
                "eligible": is_eligible,
            }
            if is_eligible and mean is not None:
                eligible.append((mean, expert))
        if not eligible:
            choice = None
        else:
            best_mean = max(item[0] for item in eligible)
            tied = {expert for mean, expert in eligible if mean == best_mean}
            choice = (
                "contextual_only"
                if "contextual_only" in tied
                else "weak_trend_only"
            )
        choices[regime_name] = choice
        diagnostics[regime_name]["frozen_choice"] = choice
    return diagnostics, choices


def build_policy_targets(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
    regime: pd.DataFrame,
    choices: Mapping[str, str | None],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    dates = frame.index
    complete_entry_mask = pd.Series(False, index=dates, dtype=bool)
    for position in range(max(0, len(dates) - 2)):
        complete_entry_mask.iloc[position] = dates[position + 1] >= ACCOUNT_START
    account_union = canonicalize_one_session_signals(
        membership["union"] & complete_entry_mask
    ).astype(bool)
    selector_cash = account_union.copy()
    reason = pd.Series("no_union", index=dates, dtype=object)
    for position in np.flatnonzero(account_union.to_numpy(dtype=bool)):
        if position + 2 >= len(frame):
            selector_cash.iloc[position] = False
            reason.iloc[position] = "incomplete_outcome"
            continue
        entry = dates[position + 1]
        if entry < EVALUATION_START:
            reason.iloc[position] = "pre_2012_union"
            continue
        if not bool(regime.iloc[position]["regime_ready"]):
            raise RegimeExpertDisagreementError(
                "An evaluation union event lacks a finite 20-session regime"
            )
        regime_name = (
            "risk_on" if bool(regime.iloc[position]["risk_on"]) else "not_risk_on"
        )
        if bool(membership.iloc[position]["both"]):
            keep = True
            event_reason = "both_experts"
        elif bool(membership.iloc[position]["contextual_only"]):
            keep = choices.get(regime_name) == "contextual_only"
            event_reason = "contextual_only"
        elif bool(membership.iloc[position]["weak_trend_only"]):
            keep = choices.get(regime_name) == "weak_trend_only"
            event_reason = "weak_trend_only"
        else:  # pragma: no cover - guarded earlier
            raise RegimeExpertDisagreementError("Unknown evaluation membership")
        selector_cash.iloc[position] = bool(keep)
        reason.iloc[position] = (
            f"{event_reason}_{regime_name}_{'kept' if keep else 'vetoed'}"
        )
    if bool((selector_cash & ~account_union).any()):
        raise RegimeExpertDisagreementError("Selector cash is not a union subset")
    targets = {
        "selector": pd.Series(
            np.where(selector_cash, 0.0, 1.0), index=dates, dtype=float
        ),
        "union": pd.Series(
            np.where(account_union, 0.0, 1.0), index=dates, dtype=float
        ),
        "always_long": pd.Series(1.0, index=dates, dtype=float),
    }
    decisions = pd.DataFrame(
        {
            "date": dates,
            "union_cash_signal": account_union.to_numpy(dtype=bool),
            "selector_cash_signal": selector_cash.to_numpy(dtype=bool),
            "both": membership["both"].to_numpy(dtype=bool),
            "contextual_only": membership["contextual_only"].to_numpy(dtype=bool),
            "weak_trend_only": membership["weak_trend_only"].to_numpy(dtype=bool),
            "risk_on": regime["risk_on"].to_numpy(dtype=bool),
            "reason": reason.to_numpy(),
            "target_exposure": targets["selector"].to_numpy(dtype=float),
        }
    )
    return targets, decisions


def _evaluation_periods() -> tuple[EvaluationPeriod, ...]:
    return tuple(
        EvaluationPeriod(str(year), f"{year}-01-01", f"{year}-12-31")
        for year in range(2012, 2019)
    )


def evaluate_policies(
    frame: pd.DataFrame, targets: Mapping[str, pd.Series]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    periods = _evaluation_periods()
    metrics: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    episodes: dict[str, pd.DataFrame] = {}
    for cost_name, cost_bps in COST_SCENARIOS:
        metrics[cost_name] = {}
        combined_ledgers: list[pd.DataFrame] = []
        combined_episodes: list[pd.DataFrame] = []
        benchmark: pd.DataFrame | None = None
        for policy_name in ("selector", "union", "always_long"):
            result, ledger, policy_benchmark, policy_episodes = (
                _continuous_policy_result(
                    frame, targets[policy_name], periods=periods, cost_bps=cost_bps
                )
            )
            metrics[cost_name][policy_name] = result
            ledger_copy = ledger.copy()
            ledger_copy.insert(0, "policy", policy_name)
            combined_ledgers.append(ledger_copy)
            episode_copy = policy_episodes.copy()
            episode_copy.insert(0, "policy", policy_name)
            combined_episodes.append(episode_copy)
            if benchmark is None:
                benchmark = policy_benchmark
            elif not benchmark.equals(policy_benchmark):
                raise RegimeExpertDisagreementError(
                    "The buy-and-hold benchmark changed between policies"
                )
            if policy_name == "always_long" and not ledger.equals(policy_benchmark):
                raise RegimeExpertDisagreementError(
                    "The always-long control differs from buy-and-hold"
                )
        assert benchmark is not None
        benchmark_copy = benchmark.copy()
        benchmark_copy.insert(0, "policy", "aapl_buy_hold")
        combined_ledgers.append(benchmark_copy)
        selector = metrics[cost_name]["selector"]
        union = metrics[cost_name]["union"]
        yearly = {
            name: float(
                selector["periods"][name]["active_log_edge"]
                - union["periods"][name]["active_log_edge"]
            )
            for name in (str(year) for year in range(2012, 2019))
        }
        total = float(
            selector["total_active_log_edge"] - union["total_active_log_edge"]
        )
        if abs(total - float(np.sum(list(yearly.values())))) > 1e-10:
            raise RegimeExpertDisagreementError(
                "Incremental yearly edges do not reconcile to the total"
            )
        metrics[cost_name]["selector_vs_union"] = {
            "total_incremental_active_log_edge": total,
            "yearly_incremental_active_log_edges": yearly,
            "positive_incremental_year_count": int(
                sum(value > 0.0 for value in yearly.values())
            ),
            "incremental_after_best_year_removed": float(total - max(yearly.values())),
        }
        ledgers[cost_name] = pd.concat(combined_ledgers, ignore_index=True)
        episodes[cost_name] = pd.concat(combined_episodes, ignore_index=True)
    metrics["integrity"] = {
        "selector_cash_subset_of_union": True,
        "benchmark_identical_across_policies": True,
        "always_long_equals_buy_and_hold": True,
    }
    return metrics, ledgers, episodes


def apply_development_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    base = metrics["base_5bps"]
    stress = metrics["stress_10bps"]
    gates = {
        "all_ledgers_unleveraged_and_nonnegative": all(
            metrics[cost][policy]["no_leverage_proof"]["passed"]
            for cost, _ in COST_SCENARIOS
            for policy in ("selector", "union", "always_long")
        ),
        "base_selector_edge_positive": base["selector"]["total_active_log_edge"] > 0.0,
        "stress_selector_edge_positive": stress["selector"]["total_active_log_edge"] > 0.0,
        "base_incremental_above_0001": (
            base["selector_vs_union"]["total_incremental_active_log_edge"] > 0.0001
        ),
        "stress_incremental_above_0001": (
            stress["selector_vs_union"]["total_incremental_active_log_edge"] > 0.0001
        ),
        "stress_at_least_four_positive_incremental_years": (
            stress["selector_vs_union"]["positive_incremental_year_count"] >= 4
        ),
        "stress_incremental_positive_without_best_year": (
            stress["selector_vs_union"]["incremental_after_best_year_removed"] > 0.0
        ),
        "stress_at_least_25_selector_episodes": (
            stress["selector"]["cash_episode_count"] >= 25
        ),
        "stress_positive_mean_selector_episode": (
            stress["selector"]["mean_cash_episode_edge"] is not None
            and stress["selector"]["mean_cash_episode_edge"] > 0.0
        ),
        "stress_positive_median_selector_episode": (
            stress["selector"]["median_cash_episode_edge"] is not None
            and stress["selector"]["median_cash_episode_edge"] > 0.0
        ),
        "benchmark_comparison_present": all(
            isinstance(metrics[cost][policy].get("comparison"), dict)
            for cost, _ in COST_SCENARIOS
            for policy in ("selector", "union", "always_long")
        ),
        "selector_cash_is_exact_union_subset": (
            metrics["integrity"]["selector_cash_subset_of_union"] is True
        ),
        "benchmark_is_identical_across_policies": (
            metrics["integrity"]["benchmark_identical_across_policies"] is True
            and metrics["integrity"]["always_long_equals_buy_and_hold"] is True
        ),
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {"passed": not failures, "failures": failures, "gates": gates}


def run_development(
    *, repo_root: Path, price_artifact: Path, output_dir: Path, run_id: str
) -> dict[str, Any]:
    started = time.monotonic()
    root = repo_root.resolve()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RegimeExpertDisagreementError("Invalid run_id")
    source = price_artifact if price_artifact.is_absolute() else root / price_artifact
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    final_dir = destination.resolve() / run_id
    temporary = destination.resolve() / f".{run_id}.pending"
    if final_dir.exists() or temporary.exists():
        raise RegimeExpertDisagreementError("Run destination already exists")
    if source.stat().st_size != INPUT_BYTE_COUNT or _file_sha256(source) != INPUT_LITERAL_SHA256:
        raise RegimeExpertDisagreementError("Input bytes do not match preregistration")
    expert_source = root / "agent_benchmark/chronological_exhaustion_expert.py"
    if _git_blob_oid(expert_source) != EXPERT_SOURCE_GIT_BLOB:
        raise RegimeExpertDisagreementError(
            "Inherited expert source does not match preregistration"
        )
    frame, provenance = load_bounded_prices(
        source, end=INPUT_LAST, required_last_session=INPUT_LAST
    )
    if len(frame) != INPUT_ROWS or frame.index.min() != INPUT_FIRST:
        raise RegimeExpertDisagreementError("Input row coverage does not match")
    signals = build_fixed_expert_signals(frame)
    regime = _risk_regime(frame)
    membership = _event_membership(signals)
    if bool((membership["union"] & ~regime["regime_ready"]).any()):
        raise RegimeExpertDisagreementError("A union opportunity lacks regime data")
    calibration_rows = _calibration_rows(frame, membership, regime)
    diagnostics, choices = build_calibration_diagnostics(calibration_rows)
    targets, decisions = build_policy_targets(
        frame, membership, regime, choices
    )
    metrics, ledgers, episodes = evaluate_policies(frame, targets)
    gate_report = apply_development_gates(metrics)
    runtime_seconds = float(time.monotonic() - started)
    report = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": "repeated_historical_development_diagnostic",
        "globally_unseen_holdout": False,
        "calibration_period": {"start": "2005-01-01", "end": "2011-12-31"},
        "evaluation_period": {"start": "2012-01-01", "end": "2018-12-31"},
        "input_provenance": provenance,
        "calibration_choices": choices,
        "calibration_diagnostics": diagnostics,
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
        "calibration_cells.json": _json_bytes(
            {"rows": calibration_rows, "diagnostics": diagnostics, "choices": choices},
            pretty=True,
        ),
        "gate_report.json": _json_bytes(gate_report, pretty=True),
        "metrics.json": _json_bytes(metrics, pretty=True),
        "selector_forecast.csv": _frame_csv_bytes(decisions),
    }
    for cost_name, _ in COST_SCENARIOS:
        payloads[f"{cost_name}_ledgers.csv"] = _frame_csv_bytes(ledgers[cost_name])
        payloads[f"{cost_name}_episodes.csv"] = _frame_csv_bytes(episodes[cost_name])
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
        "choices": choices,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("develop",))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--price-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_development(
        repo_root=args.repo_root,
        price_artifact=args.price_artifact,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "RegimeExpertDisagreementError",
    "apply_development_gates",
    "build_calibration_diagnostics",
    "build_policy_targets",
    "evaluate_policies",
    "main",
    "run_development",
]
