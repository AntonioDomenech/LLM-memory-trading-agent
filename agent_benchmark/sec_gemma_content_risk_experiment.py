"""Evaluate the frozen corroborated-adverse SEC/Gemma AAPL cash rule."""

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

from . import sec_gemma_content_risk_inputs as content_inputs
from .chronological_exhaustion_experiment import load_bounded_prices
from .deterministic_aapl import CostAssumptions, EvaluationPeriod, compare_ledgers
from .sec_filing_calendar_baseline import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    INITIAL_CASH,
    INPUT_BYTE_COUNT,
    INPUT_FIRST,
    INPUT_LITERAL_SHA256,
    INPUT_ROWS,
    MANIFEST_ROWS,
    _annual_metrics,
    _episode_metrics,
    _file_sha256,
    _json_bytes,
    _period_return,
    build_filing_target,
    load_filing_manifest,
)
from .sec_filing_gemma_extractor_schema import ADVERSE_FLAG_NAMES, DIMENSION_NAMES
from .unleveraged_aapl import assert_unleveraged_ledger, canonical_context_frame, simulate_unleveraged_period


CONTRACT_VERSION = "aapl-sec-gemma-content-risk-v1"
COST_SCENARIOS = (("base_5bps", 5.0), ("stress_10bps", 10.0))
CASH_SESSIONS = 20
MIN_VALID_OUTPUTS = 68
MIN_EPISODES = 8
MIN_EPISODE_YEARS = 5
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,79}\Z")


class SecGemmaContentRiskError(RuntimeError):
    """The content-risk experiment could not be evaluated safely."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _validate_model_identity_guard(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"before", "after"}:
        raise SecGemmaContentRiskError("Model identity guard is malformed")
    expected_keys = {
        "schema_version",
        "ollama_version",
        "model_name",
        "model_manifest_sha256",
        "semantic_runtime_fingerprint_sha256",
        "version_response_sha256",
        "tags_response_sha256",
    }
    identities: dict[str, dict[str, Any]] = {}
    for position in ("before", "after"):
        raw = value[position]
        if not isinstance(raw, Mapping) or set(raw) != expected_keys:
            raise SecGemmaContentRiskError("Model identity fields changed")
        identity = dict(raw)
        if (
            identity["schema_version"] != content_inputs.MODEL_IDENTITY_SCHEMA_VERSION
            or identity["model_name"] != content_inputs.MODEL_NAME
            or identity["model_manifest_sha256"]
            != content_inputs.MODEL_MANIFEST_SHA256
            or identity["semantic_runtime_fingerprint_sha256"]
            != content_inputs.SEMANTIC_RUNTIME_FINGERPRINT_SHA256
            or not isinstance(identity["ollama_version"], str)
            or not identity["ollama_version"]
        ):
            raise SecGemmaContentRiskError("Frozen Gemma identity changed")
        for name in ("version_response_sha256", "tags_response_sha256"):
            if not isinstance(identity[name], str) or not re.fullmatch(
                r"[0-9a-f]{64}", identity[name]
            ):
                raise SecGemmaContentRiskError("Model identity response hash changed")
        identities[position] = identity
    if identities["before"] != identities["after"]:
        raise SecGemmaContentRiskError("Local Gemma identity changed during the batch")
    return {
        "passed": True,
        "before": identities["before"],
        "after": identities["after"],
    }


def validate_sealed_model_batch(
    manifest: Sequence[Mapping[str, Any]],
    prepared_requests: Sequence[content_inputs.PreparedRequest],
    model_results: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate the frozen prompts and outputs before any price is read."""

    requests = list(prepared_requests)
    if len(requests) != MANIFEST_ROWS:
        raise SecGemmaContentRiskError("Prepared batch must contain all 75 requests")
    commitment = content_inputs.input_commitment_bytes(requests)
    if (
        len(commitment) != content_inputs.EXPECTED_COMMITMENT_BYTES
        or _sha256(commitment) != content_inputs.EXPECTED_COMMITMENT_SHA256
        or content_inputs._payload_hash_commitment(requests)
        != content_inputs.EXPECTED_ORDERED_PAYLOAD_HASH_SHA256
    ):
        raise SecGemmaContentRiskError("Frozen request commitment changed")

    manifest_by_sequence = {int(row["sequence"]): row for row in manifest}
    if len(manifest_by_sequence) != MANIFEST_ROWS:
        raise SecGemmaContentRiskError("SEC manifest sequence identity changed")
    results = _normalize_results(manifest, model_results)
    request_index: list[dict[str, Any]] = []
    for ordinal, request in enumerate(requests, start=1):
        try:
            content_inputs._validate_prepared_request(request)
        except Exception as exc:
            raise SecGemmaContentRiskError("Prepared Gemma request changed") from exc
        manifest_row = manifest_by_sequence.get(request.sequence)
        if (
            request.ordinal != ordinal
            or manifest_row is None
            or request.accession_number != str(manifest_row["accession_number"])
            or request.form != str(manifest_row["form"])
            or request.availability_session
            != str(manifest_row["availability_session"])
        ):
            raise SecGemmaContentRiskError("Prepared request crossed its SEC filing")
        result = results[request.sequence]
        try:
            validated = content_inputs._validate_loaded_result(result, request)
        except Exception as exc:
            raise SecGemmaContentRiskError("Model checkpoint failed exact replay") from exc
        if (
            validated["status"] == "valid"
            and validated["extractor_output"]["document_quality"] == "unusable"
        ):
            raise SecGemmaContentRiskError("Unusable document was not mapped to LONG")
        public = request.public_commitment_record()
        public["supplied_sentence_ids"] = list(request.supplied_sentence_ids)
        request_index.append(public)

    identity_proof = _validate_model_identity_guard(model_identity)
    index_bytes = content_inputs.canonical_json_bytes(request_index)
    result_commitment_rows = [
        {
            "ordinal": int(row["ordinal"]),
            "sequence": int(row["sequence"]),
            "request_sha256": str(row["request_sha256"]),
            "status": str(row["status"]),
            "raw_output_sha256": row.get("raw_output_sha256"),
            "extractor_output_sha256": row.get("extractor_output_sha256"),
        }
        for row in sorted(model_results, key=lambda item: int(item["ordinal"]))
    ]
    return {
        "passed": True,
        "request_count": len(requests),
        "input_commitment_bytes": len(commitment),
        "input_commitment_sha256": _sha256(commitment),
        "ordered_model_payload_hashes_sha256": (
            content_inputs.EXPECTED_ORDERED_PAYLOAD_HASH_SHA256
        ),
        "public_request_index_sha256": _sha256(index_bytes),
        "model_results_commitment_sha256": _sha256(
            content_inputs.canonical_json_bytes(result_commitment_rows)
        ),
        "model_identity": identity_proof,
    }, request_index


def corroborated_adverse_signal(output: Mapping[str, Any] | None) -> bool:
    """Apply the preregistered conjunction without scores or fitted thresholds."""

    if not isinstance(output, Mapping):
        return False
    if output.get("document_quality") not in {"usable", "thin"}:
        return False
    dimensions = output.get("dimensions")
    flags = output.get("flags")
    if not isinstance(dimensions, Mapping) or not isinstance(flags, Mapping):
        return False
    adverse_flag = any(
        isinstance(flags.get(name), Mapping) and flags[name].get("present") is True
        for name in ADVERSE_FLAG_NAMES
    )
    adverse_dimension = any(
        isinstance(dimensions.get(name), Mapping)
        and (
            dimensions[name].get("current_impact") == "unfavorable"
            or dimensions[name].get("change_vs_prior") == "deteriorating"
        )
        for name in DIMENSION_NAMES
    )
    return bool(adverse_flag and adverse_dimension)


def _normalize_results(
    manifest: Sequence[Mapping[str, Any]],
    model_results: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not manifest or len(model_results) != len(manifest):
        raise SecGemmaContentRiskError("Model results must exactly cover the filing manifest")
    expected = {int(row["sequence"]): str(row["availability_session"]) for row in manifest}
    normalized: dict[int, dict[str, Any]] = {}
    for raw in model_results:
        if not isinstance(raw, Mapping):
            raise SecGemmaContentRiskError("Model result is not an object")
        try:
            sequence = int(raw["sequence"])
            available = str(raw["availability_session"])
            status = str(raw["status"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SecGemmaContentRiskError("Model result identity is malformed") from exc
        if sequence in normalized or expected.get(sequence) != available:
            raise SecGemmaContentRiskError("Model result identity changed")
        if status not in {"valid", "invalid", "transport_error"}:
            raise SecGemmaContentRiskError("Unknown model-result status")
        output = raw.get("extractor_output")
        if (status == "valid") != isinstance(output, Mapping):
            raise SecGemmaContentRiskError("Model output/status mismatch")
        normalized[sequence] = dict(raw)
    if set(normalized) != set(expected):
        raise SecGemmaContentRiskError("Model result sequence is incomplete")
    return normalized


def build_content_target(
    frame: pd.DataFrame,
    manifest: Sequence[Mapping[str, Any]],
    model_results: Sequence[Mapping[str, Any]],
) -> tuple[pd.Series, pd.DataFrame]:
    """Create non-overlapping episodes from schema-valid corroborated warnings."""

    data = canonical_context_frame(frame)
    results = _normalize_results(manifest, model_results)
    target = pd.Series(1.0, index=data.index, name="target_exposure", dtype=float)
    positions = {day.date().isoformat(): index for index, day in enumerate(data.index)}
    blocked_through = -1
    audit: list[dict[str, Any]] = []
    ordered = sorted(
        manifest,
        key=lambda row: (str(row["availability_session"]), str(row["accession_number"])),
    )
    for row in ordered:
        sequence = int(row["sequence"])
        available = str(row["availability_session"])
        if available not in positions:
            raise SecGemmaContentRiskError("Filing date is absent from market data")
        result = results[sequence]
        valid = result["status"] == "valid"
        output = result.get("extractor_output") if valid else None
        signal = corroborated_adverse_signal(output)
        position = positions[available]
        scheduled = bool(signal and position > blocked_through)
        if not signal:
            reason = "no_corroborated_adverse_signal"
        elif not scheduled:
            reason = "active_or_pending_episode"
        else:
            reason = "scheduled"
        entry_position: int | None = None
        exit_position: int | None = None
        if scheduled:
            if position + 1 >= len(data):
                raise SecGemmaContentRiskError("Signal has no next-open fill")
            entry_position = position + 1
            exit_position = position + CASH_SESSIONS + 1
            target.iloc[position : min(position + CASH_SESSIONS, len(target))] = 0.0
            blocked_through = position + CASH_SESSIONS
        audit.append(
            {
                "sequence": sequence,
                "form": str(row["form"]),
                "filing_date": str(row["filing_date"]),
                "availability_session": available,
                "model_status": str(result["status"]),
                "document_quality": (
                    str(output.get("document_quality"))
                    if isinstance(output, Mapping)
                    else None
                ),
                "corroborated_adverse_signal": bool(signal),
                "scheduled": scheduled,
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
        raise SecGemmaContentRiskError("Content target is not exact LONG/CASH")
    return target, pd.DataFrame(audit)


def extraction_summary(model_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {name: 0 for name in ("valid", "invalid", "transport_error")}
    qualities = {name: 0 for name in ("usable", "thin", "unusable")}
    flags = {name: 0 for name in (*ADVERSE_FLAG_NAMES, "management_transition")}
    unfavorable = {name: 0 for name in DIMENSION_NAMES}
    deteriorating = {name: 0 for name in DIMENSION_NAMES}
    elapsed: list[float] = []
    for row in model_results:
        status = str(row["status"])
        statuses[status] += 1
        value = row.get("elapsed_seconds")
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0:
            elapsed.append(float(value))
        if status != "valid":
            continue
        output = row["extractor_output"]
        quality = str(output["document_quality"])
        qualities[quality] += 1
        for name in flags:
            flags[name] += int(output["flags"][name]["present"] is True)
        for name in DIMENSION_NAMES:
            item = output["dimensions"][name]
            unfavorable[name] += int(item["current_impact"] == "unfavorable")
            deteriorating[name] += int(item["change_vs_prior"] == "deteriorating")
    return {
        "statuses": statuses,
        "valid_rate": statuses["valid"] / len(model_results),
        "document_qualities": qualities,
        "flag_counts": flags,
        "unfavorable_dimension_counts": unfavorable,
        "deteriorating_dimension_counts": deteriorating,
        "call_seconds": {
            "count": len(elapsed),
            "total": float(np.sum(elapsed)) if elapsed else 0.0,
            "mean": float(np.mean(elapsed)) if elapsed else None,
            "median": float(np.median(elapsed)) if elapsed else None,
            "maximum": float(np.max(elapsed)) if elapsed else None,
        },
    }


def _subperiod_edge(strategy: pd.DataFrame, benchmark: pd.DataFrame, start: str, end: str) -> float:
    strategy_return = _period_return(strategy, start, end)
    benchmark_return = _period_return(benchmark, start, end)
    return float(math.log1p(strategy_return) - math.log1p(benchmark_return))


def _chronology_proof(frame: pd.DataFrame, schedule: pd.DataFrame) -> dict[str, Any]:
    """Make the causal next-open and fixed-horizon facts explicit in evidence."""

    data = canonical_context_frame(frame)
    positions = {day.date().isoformat(): index for index, day in enumerate(data.index)}
    scheduled = schedule.loc[schedule["scheduled"]].copy()
    next_open = True
    fixed_horizon = True
    nonoverlapping = True
    previous_exit = -1
    for row in scheduled.to_dict(orient="records"):
        available_position = positions[str(row["availability_session"])]
        entry_position = positions[str(row["entry_open"])]
        exit_position = positions[str(row["exit_open"])]
        next_open &= entry_position == available_position + 1
        fixed_horizon &= exit_position == available_position + CASH_SESSIONS + 1
        nonoverlapping &= entry_position > previous_exit
        previous_exit = exit_position
    all_development = bool(
        (pd.to_datetime(schedule["availability_session"]) <= DEVELOPMENT_END).all()
    )
    return {
        "scheduled_episode_count": int(len(scheduled)),
        "decisions_use_next_open": bool(next_open),
        "cash_episode_is_exactly_20_open_intervals": bool(fixed_horizon),
        "scheduled_episodes_do_not_overlap_or_extend": bool(nonoverlapping),
        "all_filing_availability_is_development_only": all_development,
        "passed": bool(next_open and fixed_horizon and nonoverlapping and all_development),
    }


def evaluate_policies(
    frame: pd.DataFrame,
    content_target: pd.Series,
    content_schedule: pd.DataFrame,
    calendar_target: pd.Series,
    calendar_schedule: pd.DataFrame,
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
        content = simulate_unleveraged_period(data, content_target, period, costs, initial_cash=INITIAL_CASH)
        calendar = simulate_unleveraged_period(data, calendar_target, period, costs, initial_cash=INITIAL_CASH)
        benchmark = simulate_unleveraged_period(data, benchmark_target, period, costs, initial_cash=INITIAL_CASH)
        always_long = simulate_unleveraged_period(
            data, benchmark_target, period, costs, initial_cash=INITIAL_CASH
        )
        if not always_long.equals(benchmark):
            raise SecGemmaContentRiskError(
                "Always-long control differs from same-ledger AAPL"
            )
        content_comparison = compare_ledgers(content, benchmark, initial_cash=INITIAL_CASH)
        calendar_comparison = compare_ledgers(calendar, benchmark, initial_cash=INITIAL_CASH)
        annual = _annual_metrics(content, benchmark)
        episodes = _episode_metrics(data, content_schedule, cost_bps)
        content_edge = float(
            math.log1p(content_comparison["strategy"]["total_return"])
            - math.log1p(content_comparison["aapl_buy_hold"]["total_return"])
        )
        calendar_edge = float(
            math.log1p(calendar_comparison["strategy"]["total_return"])
            - math.log1p(calendar_comparison["aapl_buy_hold"]["total_return"])
        )
        if abs(content_edge - annual["total_active_log_edge"]) > 1e-10:
            raise SecGemmaContentRiskError("Annual content edges do not reconcile")
        if abs(content_edge - episodes["total_active_log_edge"]) > 1e-10:
            raise SecGemmaContentRiskError("Episode content edges do not reconcile")
        metrics[cost_name] = {
            "cost_bps_per_changing_leg": cost_bps,
            "content_rule": {
                "comparison": content_comparison,
                "annual": annual,
                "episodes": {key: value for key, value in episodes.items() if key != "rows"},
                "continuous_active_log_edge": content_edge,
                "subperiod_active_log_edges": {
                    "2000_2008": _subperiod_edge(content, benchmark, "2000-01-01", "2008-12-31"),
                    "2009_2018": _subperiod_edge(content, benchmark, "2009-01-01", "2018-12-31"),
                },
                "no_leverage_proof": assert_unleveraged_ledger(content),
            },
            "all_filings_calendar_control": {
                "comparison": calendar_comparison,
                "continuous_active_log_edge": calendar_edge,
                "no_leverage_proof": assert_unleveraged_ledger(calendar),
            },
            "aapl_buy_hold_no_leverage_proof": assert_unleveraged_ledger(benchmark),
            "content_minus_calendar_active_log_edge": float(content_edge - calendar_edge),
            "always_long_matches_same_ledger_aapl": True,
        }
        ledgers[cost_name] = pd.concat(
            [
                content.add_prefix("content_"),
                calendar.add_prefix("calendar_"),
                benchmark.add_prefix("aapl_"),
            ],
            axis=1,
        )
        episode_rows[cost_name] = episodes["rows"]
    return metrics, ledgers, episode_rows


def apply_development_gates(
    metrics: Mapping[str, Any],
    extraction: Mapping[str, Any],
    schedule: pd.DataFrame,
    scientific_proofs: Mapping[str, Any],
) -> dict[str, Any]:
    base = metrics["base_5bps"]
    stress = metrics["stress_10bps"]
    stress_rule = stress["content_rule"]
    completed = schedule.loc[schedule["scheduled"] & schedule["complete_by_2018"]]
    entry_years = {
        pd.Timestamp(value).year for value in completed["entry_open"].dropna().tolist()
    }
    gates = {
        "at_least_68_valid_outputs": extraction["statuses"]["valid"] >= MIN_VALID_OUTPUTS,
        "at_least_8_complete_cash_episodes": len(completed) >= MIN_EPISODES,
        "episodes_span_at_least_5_entry_years": len(entry_years) >= MIN_EPISODE_YEARS,
        "base_relative_wealth_vs_aapl_positive": (
            base["content_rule"]["comparison"]["relative_wealth_vs_aapl_buy_hold"] > 0.0
        ),
        "stress_relative_wealth_vs_aapl_positive": (
            stress_rule["comparison"]["relative_wealth_vs_aapl_buy_hold"] > 0.0
        ),
        "base_content_minus_calendar_edge_positive": (
            base["content_minus_calendar_active_log_edge"] > 0.0
        ),
        "stress_content_minus_calendar_edge_positive": (
            stress["content_minus_calendar_active_log_edge"] > 0.0
        ),
        "stress_winning_years_outnumber_losing_years": (
            stress_rule["annual"]["winning_years"] > stress_rule["annual"]["losing_years"]
        ),
        "stress_mean_annual_excess_positive": stress_rule["annual"]["mean_annual_excess"] > 0.0,
        "stress_median_annual_excess_positive": stress_rule["annual"]["median_annual_excess"] > 0.0,
        "stress_mean_episode_edge_positive": (
            stress_rule["episodes"]["mean_active_log_edge"] is not None
            and stress_rule["episodes"]["mean_active_log_edge"] > 0.0
        ),
        "stress_median_episode_edge_positive": (
            stress_rule["episodes"]["median_active_log_edge"] is not None
            and stress_rule["episodes"]["median_active_log_edge"] > 0.0
        ),
        "stress_2000_2008_edge_positive": (
            stress_rule["subperiod_active_log_edges"]["2000_2008"] > 0.0
        ),
        "stress_2009_2018_edge_positive": (
            stress_rule["subperiod_active_log_edges"]["2009_2018"] > 0.0
        ),
        "stress_edge_positive_after_best_year_removed": (
            stress_rule["annual"]["active_log_edge_after_best_year_removed"] > 0.0
        ),
        "stress_negative_aapl_year_edge_positive": (
            stress_rule["annual"]["negative_aapl_year_active_log_edge"] > 0.0
        ),
        "all_ledgers_unleveraged_and_nonnegative": all(
            proof["passed"]
            for cost, _ in COST_SCENARIOS
            for proof in (
                metrics[cost]["content_rule"]["no_leverage_proof"],
                metrics[cost]["all_filings_calendar_control"]["no_leverage_proof"],
                metrics[cost]["aapl_buy_hold_no_leverage_proof"],
            )
        ),
        "always_long_matches_same_ledger_aapl": all(
            metrics[cost]["always_long_matches_same_ledger_aapl"]
            for cost, _ in COST_SCENARIOS
        ),
        "input_hash_and_chronology_proofs_pass": bool(scientific_proofs["passed"]),
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {"passed": not failures, "failures": failures, "gates": gates}


def run_evaluation(
    *,
    repo_root: Path,
    receipt_dir: Path,
    price_artifact: Path,
    prepared_requests: Sequence[content_inputs.PreparedRequest],
    model_results: Sequence[Mapping[str, Any]],
    model_identity: Mapping[str, Any],
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    started = time.monotonic()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SecGemmaContentRiskError("Invalid run_id")
    root = repo_root.resolve()
    source = price_artifact if price_artifact.is_absolute() else root / price_artifact
    receipts = receipt_dir if receipt_dir.is_absolute() else root / receipt_dir
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    final_dir = destination.resolve() / run_id
    temporary = destination.resolve() / f".{run_id}.pending"
    if final_dir.exists() or temporary.exists():
        raise SecGemmaContentRiskError("Run destination already exists")
    manifest, manifest_payload = load_filing_manifest(receipts)
    if len(manifest) != MANIFEST_ROWS or len(model_results) != MANIFEST_ROWS:
        raise SecGemmaContentRiskError("Development evaluation requires all 75 filings")
    batch_proof, request_index = validate_sealed_model_batch(
        manifest, prepared_requests, model_results, model_identity
    )
    if source.stat().st_size != INPUT_BYTE_COUNT or _file_sha256(source) != INPUT_LITERAL_SHA256:
        raise SecGemmaContentRiskError("Market bytes do not match preregistration")
    frame, provenance = load_bounded_prices(source, end=DEVELOPMENT_END, required_last_session=DEVELOPMENT_END)
    if len(frame) != INPUT_ROWS or frame.index.min() != INPUT_FIRST:
        raise SecGemmaContentRiskError("Market coverage changed")
    content_target, content_schedule = build_content_target(frame, manifest, model_results)
    calendar_target, calendar_schedule = build_filing_target(frame, manifest)
    metrics, ledgers, episode_rows = evaluate_policies(
        frame, content_target, content_schedule, calendar_target, calendar_schedule
    )
    extraction = extraction_summary(model_results)
    content_chronology = _chronology_proof(frame, content_schedule)
    calendar_chronology = _chronology_proof(frame, calendar_schedule)
    scientific_proofs = {
        "market_literal_sha256_matches": _file_sha256(source) == INPUT_LITERAL_SHA256,
        "market_literal_byte_count_matches": source.stat().st_size == INPUT_BYTE_COUNT,
        "market_is_physically_bounded_through_2018": bool(
            provenance["physical_snapshot_has_later_rows"] is False
            and provenance["bounded_last_date"] == DEVELOPMENT_END.date().isoformat()
        ),
        "filing_manifest_row_count_matches": len(manifest) == MANIFEST_ROWS,
        "filing_manifest_is_development_only": bool(
            max(str(row["availability_session"]) for row in manifest)
            <= DEVELOPMENT_END.date().isoformat()
        ),
        "content_rule_chronology": content_chronology,
        "calendar_control_chronology": calendar_chronology,
    }
    scientific_proofs["passed"] = bool(
        scientific_proofs["market_literal_sha256_matches"]
        and scientific_proofs["market_literal_byte_count_matches"]
        and scientific_proofs["market_is_physically_bounded_through_2018"]
        and scientific_proofs["filing_manifest_row_count_matches"]
        and scientific_proofs["filing_manifest_is_development_only"]
        and content_chronology["passed"]
        and calendar_chronology["passed"]
    )
    gate_report = apply_development_gates(
        metrics, extraction, content_schedule, scientific_proofs
    )
    runtime_seconds = float(time.monotonic() - started)
    report = {
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "stage": "repeated_historical_development",
        "globally_unseen_holdout": False,
        "model_pretraining_contamination_risk": "unresolved",
        "period": {
            "start": DEVELOPMENT_START.date().isoformat(),
            "end": DEVELOPMENT_END.date().isoformat(),
        },
        "model_identity": dict(model_identity),
        "model_batch_proof": batch_proof,
        "filing_manifest": {
            "rows": len(manifest),
            "bytes": len(manifest_payload),
            "sha256": _sha256(manifest_payload),
            "unique_availability_sessions": len(
                {str(row["availability_session"]) for row in manifest}
            ),
        },
        "extraction_summary": extraction,
        "content_schedule": {
            "corroborated_signals": int(content_schedule["corroborated_adverse_signal"].sum()),
            "scheduled_episodes": int(content_schedule["scheduled"].sum()),
            "completed_episodes": int(content_schedule["complete_by_2018"].sum()),
        },
        "input_provenance": provenance,
        "scientific_proofs": scientific_proofs,
        "metrics": metrics,
        "development_gate_report": gate_report,
        "runtime": {
            "evaluation_seconds": runtime_seconds,
            "paid_api_calls": 0,
            "broker_actions": 0,
            "real_money_actions": 0,
        },
        "later_period_opened": False,
        "real_money_authorized": False,
    }
    public_results = []
    for row in sorted(model_results, key=lambda item: int(item["sequence"])):
        output = row.get("extractor_output")
        public_results.append(
            {
                "sequence": int(row["sequence"]),
                "availability_session": str(row["availability_session"]),
                "request_sha256": str(row.get("request_sha256", "")),
                "status": str(row["status"]),
                "supplied_sentence_ids": list(row.get("supplied_sentence_ids", [])),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "extractor_output": output if isinstance(output, Mapping) else None,
                "raw_output_sha256": row.get("raw_output_sha256"),
                "extractor_output_sha256": row.get("extractor_output_sha256"),
            }
        )
    target_frame = pd.DataFrame(
        {"date": frame.index.date.astype(str), "target_exposure": content_target.to_numpy(dtype=float)}
    )
    payloads: dict[str, bytes] = {
        "report.json": _json_bytes(report, pretty=True),
        "model_results.json": _json_bytes(public_results, pretty=True),
        "extraction_summary.json": _json_bytes(extraction, pretty=True),
        "filing_manifest.json": manifest_payload,
        "request_index.json": _json_bytes(request_index, pretty=True),
        "model_batch_proof.json": _json_bytes(batch_proof, pretty=True),
        "model_identity.json": _json_bytes(model_identity, pretty=True),
        "content_schedule.csv": _csv_bytes(content_schedule),
        "calendar_schedule.csv": _csv_bytes(calendar_schedule),
        "target_history.csv": _csv_bytes(target_frame),
        "metrics.json": _json_bytes(metrics, pretty=True),
        "gate_report.json": _json_bytes(gate_report, pretty=True),
    }
    for cost_name, _ in COST_SCENARIOS:
        payloads[f"{cost_name}_ledgers.csv"] = _csv_bytes(ledgers[cost_name])
        payloads[f"{cost_name}_episodes.json"] = _json_bytes(episode_rows[cost_name], pretty=True)
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
    parser.add_argument("command", choices=("evaluate",))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--price-artifact", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    prepared_requests = content_inputs.prepare_requests(args.private_root)
    model_results = content_inputs.load_model_results(
        args.checkpoint_dir, prepared_requests
    )
    if len(model_results) != MANIFEST_ROWS:
        raise SecGemmaContentRiskError("All 75 model checkpoints are required")
    model_identity = {
        "before": json.loads(
            (args.checkpoint_dir / "runtime-before.json").read_text(encoding="utf-8")
        ),
        "after": json.loads(
            (args.checkpoint_dir / "runtime-after.json").read_text(encoding="utf-8")
        ),
    }
    result = run_evaluation(
        repo_root=args.repo_root,
        receipt_dir=args.receipt_dir,
        price_artifact=args.price_artifact,
        prepared_requests=prepared_requests,
        model_results=model_results,
        model_identity=model_identity,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SecGemmaContentRiskError",
    "apply_development_gates",
    "build_content_target",
    "corroborated_adverse_signal",
    "evaluate_policies",
    "extraction_summary",
    "run_evaluation",
    "validate_sealed_model_batch",
]
