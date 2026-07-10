"""Sealed offline pre-2019 experiment for the rare-loss forest.

This public runner accepts the same immutable through-2018 price artifact and
bounded IWM/VIX context source as the prior one-session experiment.  It has no
entry point capable of evaluating 2019 or later data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .rare_loss_features import (
    CORE_FEATURE_COLUMNS,
    RARE_LOSS_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    build_rare_loss_feature_label_frame,
    strict_pre_fold_training_mask,
)
from .rare_loss_forest import MINIMUM_OOB_TREE_COUNT, RareLossForest
from .rare_loss_scoring import (
    evaluate_rare_loss_ablation,
    evaluate_rare_loss_development,
)
from .rare_loss_walkforward import (
    MODEL_VARIANTS,
    RARE_LOSS_CANDIDATES,
    WALK_FORWARD_FOLDS,
    RareLossCandidate,
    build_pre2019_rare_loss_walkforward,
    candidate_cash_target_column,
)
from .regime_consensus_experiment import (
    ARTIFACT_GIT_ATTRIBUTES,
    COST_SCENARIOS,
    DEVELOPMENT_END,
    INITIAL_CASH,
    RegimeConsensusExperimentError,
    SOURCE_FILES as REGIME_SOURCE_FILES,
    _Deadline,
    _artifact_snapshot,
    _bounded_inputs,
    _canonical_json_bytes,
    _frame_csv_bytes,
    _parse_json,
    _pretty_json_bytes,
    _seal_artifact_bundle,
    _sha256_hex,
    _sha256_tagged,
    _snapshot_csv,
    _verify_checksum_snapshot,
    load_public_development_inputs,
)
from .deterministic_aapl import CostAssumptions, EvaluationPeriod
from .unleveraged_aapl import (
    _git_state,
    simulate_unleveraged_period,
    validate_source_repository,
)


CONTRACT_VERSION = "aapl-one-session-rare-loss-forest-development-v1"
DEVELOPMENT_PERIOD = EvaluationPeriod(
    "rare_loss_development", "2005-01-01", "2018-12-31"
)
RUN_TIME_LIMIT_SECONDS = 3600.0

SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            *REGIME_SOURCE_FILES,
            "agent_benchmark/rare_loss_experiment.py",
            "agent_benchmark/rare_loss_features.py",
            "agent_benchmark/rare_loss_forest.py",
            "agent_benchmark/rare_loss_scoring.py",
            "agent_benchmark/rare_loss_walkforward.py",
            "docs/aapl_one_session_rare_loss_forest_v1.md",
        )
    )
)

SOURCE_EXCLUSION_AUDITS: Mapping[str, Mapping[str, Any]] = {
    "news": {
        "included": False,
        "finding": "available rows are not proven point-in-time historical text",
        "decision": "excluded from this final price/market-only experiment",
    },
    "cftc": {
        "included": False,
        "finding": "weekly positioning already failed its separate frozen approach",
        "decision": "no positioning value enters a feature, fit, or gate",
    },
    "macro_and_tnx": {
        "included": False,
        "finding": "stored macro values are placeholder or vintage-unsafe",
        "decision": "bounded context physically contains only IWM and VIX",
    },
    "llm": {
        "included": False,
        "finding": "an LLM cannot add information to the same numeric inputs",
        "decision": "zero LLM calls; the forest is deterministic local NumPy",
    },
}


class RareLossExperimentError(RuntimeError):
    """Raised when the sealed development contract is violated."""


class RareLossExperimentTimeout(RareLossExperimentError):
    """Raised when the complete public workflow exceeds one hour."""


class _RareLossDeadline(_Deadline):
    def check(self, step: str) -> None:
        elapsed = self.elapsed()
        if not math.isfinite(elapsed) or elapsed > RUN_TIME_LIMIT_SECONDS:
            raise RareLossExperimentTimeout(
                f"Development exceeded {RUN_TIME_LIMIT_SECONDS:.0f} seconds at {step}"
            )


@dataclass(frozen=True)
class CandidateSpec:
    candidate: RareLossCandidate

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def cash_target_column(self, variant: str) -> str:
        return candidate_cash_target_column(variant, self.candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "training_oob_severe_tail_quantile": self.candidate.oob_tail_quantile,
            "severe_probability_minimum_multiple_of_causal_prevalence": 2.0,
            "ordinary_cash_win_probability_gate": 0.55,
            "expected_clipped_edge_10bps_gate": 0.001,
            "cash_decision_rows": 1,
        }


CANDIDATE_SPECS = tuple(CandidateSpec(item) for item in RARE_LOSS_CANDIDATES)


def _source_hashes_at_commit(repo_root: Path, commit: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in SOURCE_FILES:
        blob = subprocess.run(
            ["git", "cat-file", "blob", f"{commit}:{path}"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if blob.returncode != 0:
            raise RareLossExperimentError(
                f"Recorded source commit lacks required file: {path}"
            )
        hashes[path] = _sha256_hex(bytes(blob.stdout))
    return hashes


def _verify_source_identity(
    snapshot: Mapping[str, bytes],
    *,
    repo_root: Path,
    artifact_commit: str,
) -> str:
    provenance = _parse_json(snapshot, "input_provenance.json")
    identity = provenance.get("source_identity") if isinstance(provenance, dict) else None
    if not isinstance(identity, dict):
        raise RareLossExperimentError("Artifact lacks source identity")
    git = identity.get("git")
    hashes = identity.get("source_hashes")
    if not isinstance(git, dict) or not isinstance(hashes, dict):
        raise RareLossExperimentError("Source identity is incomplete")
    source_commit = str(git.get("commit") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) or git.get("dirty") is not False:
        raise RareLossExperimentError("Recorded source was not a clean commit")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, artifact_commit],
        cwd=repo_root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RareLossExperimentError("Source commit is not an ancestor of artifact commit")
    if _source_hashes_at_commit(repo_root, source_commit) != hashes:
        raise RareLossExperimentError("Recorded source hashes do not replay")
    return source_commit


def _full_exposure_target(
    price_index: pd.DatetimeIndex,
    prediction_index: pd.DatetimeIndex,
    cash_target: Sequence[float],
) -> pd.Series:
    if not prediction_index.isin(price_index).all():
        raise RareLossExperimentError("Prediction dates are absent from prices")
    cash = np.asarray(cash_target, dtype=float)
    if cash.shape != (len(prediction_index),) or not np.isin(cash, (0.0, 1.0)).all():
        raise RareLossExperimentError("Candidate CASH target is not exact binary")
    target = pd.Series(1.0, index=price_index, name="target_exposure")
    target.loc[prediction_index] = 1.0 - cash
    return target


def _simulate(prices: pd.DataFrame, target: pd.Series, *, cost_bps: float) -> pd.DataFrame:
    return simulate_unleveraged_period(
        prices,
        target,
        DEVELOPMENT_PERIOD,
        CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0),
        initial_cash=INITIAL_CASH,
    )


def _float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _common_predictive_support(predictions: pd.DataFrame) -> np.ndarray:
    required = [
        "features_ready",
        "causal_severe_prevalence",
        "causal_ordinary_prevalence_10bps",
        "causal_training_mean_clipped_edge_10bps",
    ]
    for variant in MODEL_VARIANTS:
        required.extend(
            (
                f"{variant}_severe_probability",
                f"{variant}_ordinary_cash_win_probability_10bps",
                f"{variant}_expected_clipped_edge_10bps",
            )
        )
    missing = sorted(set(required).difference(predictions.columns))
    if missing:
        raise RareLossExperimentError(f"Walk-forward lacks support fields: {missing}")
    ready = predictions["features_ready"].to_numpy(dtype=bool)
    finite = np.isfinite(predictions.loc[:, required[1:]].to_numpy(dtype=float)).all(axis=1)
    support = ready & finite
    if not support.any():
        raise RareLossExperimentError("Common full/core OOF support is empty")
    return support


def _prediction_inputs(
    predictions: pd.DataFrame,
    *,
    variant: str,
    cash_target: Sequence[float],
    support: Sequence[bool],
) -> tuple[Any, ...]:
    selected = np.asarray(support, dtype=bool)
    cash = np.asarray(cash_target, dtype=float)
    if selected.shape != (len(predictions),) or cash.shape != selected.shape:
        raise RareLossExperimentError("Prediction inputs are not OOF-aligned")
    if bool((cash[~selected] != 0.0).any()):
        raise RareLossExperimentError("Unavailable feature rows must remain LONG")

    columns = {
        "ordinary_probability": f"{variant}_ordinary_cash_win_probability_10bps",
        "ordinary_label": "cash_beats_long_10bps",
        "ordinary_baseline": "causal_ordinary_prevalence_10bps",
        "predicted_edge": f"{variant}_expected_clipped_edge_10bps",
        "actual_edge": "cash_active_log_edge_10bps_clipped",
        "edge_baseline": "causal_training_mean_clipped_edge_10bps",
        "severe_probability": f"{variant}_severe_probability",
        "severe_label": "severe_loss_label_1session",
        "severe_baseline": "causal_severe_prevalence",
    }
    missing = sorted(set(columns.values()).difference(predictions.columns))
    if missing:
        raise RareLossExperimentError(f"Walk-forward predictions lack columns: {missing}")
    return tuple(
        _float_array(predictions[column])[selected]
        for column in columns.values()
    ) + (cash[selected], predictions.index[selected])


def _score_strategy(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    inputs: tuple[Any, ...],
    cost_bps: float,
) -> dict[str, Any]:
    (
        ordinary_probability,
        ordinary_label,
        ordinary_baseline,
        predicted_edge,
        actual_edge,
        edge_baseline,
        severe_probability,
        severe_label,
        severe_baseline,
        cash,
        dates,
    ) = inputs
    return evaluate_rare_loss_development(
        strategy,
        benchmark,
        strategy["target_exposure"].to_numpy(dtype=float),
        ordinary_probability,
        ordinary_label,
        ordinary_baseline,
        predicted_edge,
        actual_edge,
        edge_baseline,
        severe_probability,
        severe_label,
        severe_baseline,
        cash,
        dates,
        cost_bps=cost_bps,
    )


def select_development_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> str | None:
    passing = [item for item in candidate_results if bool(item.get("passed"))]
    if not passing:
        return None

    def rank(item: Mapping[str, Any]) -> tuple[float, float, float, int, int, str]:
        metrics = item["scenarios"]["stress_10bps"]["metrics"]
        return (
            -float(metrics["minimum_fold_active_log_edge"]),
            -float(metrics["total_active_log_edge"]),
            -float(metrics["severe_brier_relative_improvement"]),
            int(metrics["completed_cash_episodes"]),
            int(metrics["cash_days"]),
            str(item["candidate"]["candidate_id"]),
        )

    return str(sorted(passing, key=rank)[0]["candidate"]["candidate_id"])


def _feature_provenance() -> dict[str, Any]:
    return {
        "decision_timestamp": "after completed close t",
        "execution_timestamp": "adjusted AAPL open t+1",
        "label_maturity": "adjusted AAPL open t+2",
        "cash_horizon": "one open-to-open session",
        "core_feature_columns": list(CORE_FEATURE_COLUMNS),
        "sentiment_feature_columns": list(SENTIMENT_FEATURE_COLUMNS),
        "full_feature_columns": list(RARE_LOSS_FEATURE_COLUMNS),
        "context_alignment": "exact AAPL session date; no fill or interpolation",
        "missing_policy": "all twenty inputs required for both models; otherwise LONG",
        "training_policy": "seven fixed purged expanding folds; maturity strictly before fold",
        "excluded_sources": {
            key: dict(value) for key, value in SOURCE_EXCLUSION_AUDITS.items()
        },
    }


def _compute_development(
    prices: pd.DataFrame,
    context: pd.DataFrame,
    *,
    deadline: _RareLossDeadline,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, pd.DataFrame, list[dict[str, Any]], dict[str, pd.DataFrame]]:
    feature_label = build_rare_loss_feature_label_frame(prices, context)
    if feature_label.index.max() > DEVELOPMENT_END:
        raise RareLossExperimentError("Features contain post-2018 rows")
    deadline.check("feature and label construction")
    walkforward = build_pre2019_rare_loss_walkforward(feature_label)
    predictions = walkforward.predictions.copy()
    states = json.loads(_canonical_json_bytes(walkforward.model_states).decode("utf-8"))
    if predictions.index.max() > DEVELOPMENT_END or len(states) != 14:
        raise RareLossExperimentError("Walk-forward escaped its 14-fit pre-2019 contract")
    deadline.check("fourteen rare-loss forest fits")

    support = _common_predictive_support(predictions)
    target_audit = pd.DataFrame(
        {"common_predictive_support": support}, index=predictions.index
    )
    target_audit.index.name = "decision_date"
    cash_by_key: dict[tuple[str, str], np.ndarray] = {}
    targets: dict[tuple[str, str], pd.Series] = {}
    for spec in CANDIDATE_SPECS:
        for variant in MODEL_VARIANTS:
            column = spec.cash_target_column(variant)
            cash = _float_array(predictions[column])
            if not np.isfinite(cash).all() or not np.isin(cash, (0.0, 1.0)).all():
                raise RareLossExperimentError(f"Target is not binary: {column}")
            cash_by_key[(variant, spec.candidate_id)] = cash
            target_audit[column] = cash.astype(np.int8)
            targets[(variant, spec.candidate_id)] = _full_exposure_target(
                prices.index, predictions.index, cash
            )

    benchmark_target = pd.Series(1.0, index=prices.index, name="target_exposure")
    ledgers: dict[str, pd.DataFrame] = {}
    benchmarks: dict[str, pd.DataFrame] = {}
    for scenario, bps in COST_SCENARIOS:
        benchmark = _simulate(prices, benchmark_target, cost_bps=bps)
        benchmarks[scenario] = benchmark
        ledgers[f"buy_hold_{scenario}.csv"] = benchmark
        deadline.check(f"{scenario} benchmark simulation")

    variant_results: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in CANDIDATE_SPECS:
        for variant in MODEL_VARIANTS:
            key = (variant, spec.candidate_id)
            inputs = _prediction_inputs(
                predictions,
                variant=variant,
                cash_target=cash_by_key[key],
                support=support,
            )
            scenarios: dict[str, Any] = {}
            for scenario, bps in COST_SCENARIOS:
                strategy = _simulate(prices, targets[key], cost_bps=bps)
                scenarios[scenario] = _score_strategy(
                    strategy, benchmarks[scenario], inputs=inputs, cost_bps=bps
                )
                ledgers[f"{variant}_{spec.candidate_id}_{scenario}_strategy.csv"] = strategy
                deadline.check(f"{variant} {spec.candidate_id} {scenario} scoring")
            variant_results[key] = scenarios

    results: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        full = variant_results[("full", spec.candidate_id)]
        core = variant_results[("core", spec.candidate_id)]
        ablation = evaluate_rare_loss_ablation(
            full["base_5bps"],
            core["base_5bps"],
            full["stress_10bps"],
            core["stress_10bps"],
        )
        both_costs = bool(all(item["gates"]["passed"] for item in full.values()))
        results.append(
            {
                "candidate": spec.to_dict(),
                "stage": "development_only",
                "scenarios": full,
                "core_ablation_scenarios": core,
                "both_cost_gates_passed": both_costs,
                "paired_core_ablation": ablation,
                "passed": bool(both_costs and ablation["passed"]),
            }
        )
    return feature_label, predictions, states, target_audit, results, ledgers


def _fit_final_model(
    feature_label: pd.DataFrame,
    selected_id: str | None,
    *,
    deadline: _RareLossDeadline,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if selected_id is None:
        return {}, {"performed": False, "reason": "no_development_candidate_passed"}
    candidates = {item.candidate_id: item for item in RARE_LOSS_CANDIDATES}
    if selected_id not in candidates:
        raise RareLossExperimentError("Selected candidate is outside the frozen grid")
    boundary = pd.Timestamp("2019-01-01")
    mask = strict_pre_fold_training_mask(
        feature_label, fold_first_decision_date=boundary
    ).to_numpy(dtype=bool)
    training = feature_label.loc[mask]
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    metadata = {
        "variant": "full",
        "purpose": "selected_through_2018_refit",
        "training_start_date": training.index.min().date().isoformat(),
        "training_end_date": training.index.max().date().isoformat(),
        "maximum_label_maturity_date": maturity.max().date().isoformat(),
    }

    def fit() -> RareLossForest:
        return RareLossForest().fit(
            training.loc[:, list(RARE_LOSS_FEATURE_COLUMNS)],
            training["severe_loss_label_1session"].to_numpy(float),
            training["cash_beats_long_10bps"].to_numpy(float),
            training["cash_active_log_edge_10bps_clipped"].to_numpy(float),
            namespace="selected-through-2018|full",
            fit_metadata=metadata,
        )

    first = fit()
    deadline.check("first selected-model refit")
    second = fit()
    deadline.check("second selected-model refit")
    first_bytes = (first.canonical_json() + "\n").encode("utf-8")
    second_bytes = (second.canonical_json() + "\n").encode("utf-8")
    first_oob_counts = first.oob_components()["oob_tree_count"]
    second_oob_counts = second.oob_components()["oob_tree_count"]
    if (
        int(first_oob_counts.min()) < MINIMUM_OOB_TREE_COUNT
        or int(second_oob_counts.min()) < MINIMUM_OOB_TREE_COUNT
    ):
        raise RareLossExperimentError(
            "Selected refit failed the minimum 16-tree OOB coverage contract"
        )
    candidate = candidates[selected_id]
    first_threshold = first.oob_severe_threshold(candidate.oob_tail_quantile)
    second_threshold = second.oob_severe_threshold(candidate.oob_tail_quantile)
    if first_bytes != second_bytes or first_threshold != second_threshold:
        raise RareLossExperimentError("Selected-model refit was not byte-identical")
    payload_name = "final_full_rare_loss_forest_through_2018.json"
    return {payload_name: first_bytes}, {
        "performed": True,
        "selected_candidate_id": selected_id,
        "training_cutoff": "2018-12-31",
        "training_rows": int(len(training)),
        "training_latest_label_maturity": maturity.max().date().isoformat(),
        "model_sha256": first.model_sha256,
        "state_file_sha256": _sha256_tagged(first_bytes),
        "oob_tail_quantile": candidate.oob_tail_quantile,
        "oob_severe_threshold": first_threshold,
        "minimum_oob_tree_count": int(first_oob_counts.min()),
        "refit_repeated_byte_identically": True,
    }


def _rebuilt_payloads(
    snapshot: Mapping[str, bytes],
) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    prices = _snapshot_csv(
        snapshot, "development_prices_through_2018.csv", index_column="date"
    )
    context = _snapshot_csv(
        snapshot, "development_context_through_2018.csv", index_column="date"
    )
    prices, context, _ = _bounded_inputs(prices, context)
    deadline = _RareLossDeadline(lambda: 0.0)
    feature_label, predictions, states, targets, results, ledgers = _compute_development(
        prices, context, deadline=deadline
    )
    selected = select_development_candidate(results)
    model_payloads, refit = _fit_final_model(
        feature_label, selected, deadline=deadline
    )
    payloads: dict[str, bytes] = {
        "development_prices_through_2018.csv": _frame_csv_bytes(
            prices.reset_index(names="date")
        ),
        "development_context_through_2018.csv": _frame_csv_bytes(
            context.reset_index(names="date")
        ),
        "feature_provenance.json": _pretty_json_bytes(_feature_provenance()),
        "oof_predictions_2005_2018.csv": _frame_csv_bytes(predictions.reset_index()),
        "oof_forest_states.json": _pretty_json_bytes(states),
        "candidate_targets_2005_2018.csv": _frame_csv_bytes(targets.reset_index()),
        "candidate_results.json": _pretty_json_bytes(results),
        **model_payloads,
    }
    payloads.update(
        {
            name: _frame_csv_bytes(frame, float_format="%.17g")
            for name, frame in ledgers.items()
        }
    )
    return payloads, results, refit


def _verify_development_artifact(
    artifact: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    run_dir = artifact.resolve()
    if run_dir.is_file():
        if run_dir.name != "report.json":
            raise RareLossExperimentError("Verification file must be report.json")
        run_dir = run_dir.parent
    snapshot, artifact_commit = _artifact_snapshot(run_dir, repo_root=repo_root)
    checksums = _verify_checksum_snapshot(snapshot)
    if snapshot.get(".gitattributes") != ARTIFACT_GIT_ATTRIBUTES:
        raise RareLossExperimentError("Byte-preservation rule changed")
    source_commit: str | None = None
    if repo_root is not None:
        if artifact_commit is None:
            raise RareLossExperimentError("Missing artifact Git commit")
        source_commit = _verify_source_identity(
            snapshot,
            repo_root=repo_root.resolve(),
            artifact_commit=artifact_commit,
        )

    report = _parse_json(snapshot, "report.json")
    manifest = _parse_json(snapshot, "selection_manifest.json")
    results = _parse_json(snapshot, "candidate_results.json")
    provenance = _parse_json(snapshot, "input_provenance.json")
    if not all(isinstance(value, dict) for value in (report, manifest, provenance)):
        raise RareLossExperimentError("Report, manifest, and provenance must be objects")
    if (
        report.get("contract_version") != CONTRACT_VERSION
        or report.get("stage") != "development"
        or report.get("physical_data_end") != "2018-12-31"
        or report.get("post_2018_market_data_accessed") is not False
        or manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("post_2018_market_data_access_allowed") is not False
        or manifest.get("post_2018_market_data_accessed") is not False
        or provenance.get("contract_version") != CONTRACT_VERSION
        or provenance.get("physical_data_end") != "2018-12-31"
        or provenance.get("network_access") is not False
        or provenance.get("api_calls") != 0
        or provenance.get("llm_calls") != 0
    ):
        raise RareLossExperimentError("Artifact violates the offline pre-2019 identity")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha256_tagged(_canonical_json_bytes(unsigned)):
        raise RareLossExperimentError("Selection manifest hash is invalid")
    if report.get("selection_manifest") != manifest:
        raise RareLossExperimentError("Report and manifest differ")
    if not isinstance(results, list) or len(results) != len(CANDIDATE_SPECS):
        raise RareLossExperimentError("Candidate result set is not frozen")
    if manifest.get("candidate_results_sha256") != _sha256_tagged(
        _canonical_json_bytes(results)
    ) or report.get("candidate_results_sha256") != manifest.get(
        "candidate_results_sha256"
    ):
        raise RareLossExperimentError("Candidate results are not bound")
    if (
        manifest.get("candidate_family") != [item.to_dict() for item in CANDIDATE_SPECS]
        or manifest.get("model_variants") != list(MODEL_VARIANTS)
        or manifest.get("walk_forward_folds") != [asdict(item) for item in WALK_FORWARD_FOLDS]
        or manifest.get("development_forest_fit_count") != 14
        or manifest.get("cost_scenarios_bps") != dict(COST_SCENARIOS)
        or manifest.get("source_exclusion_audits")
        != {key: dict(value) for key, value in SOURCE_EXCLUSION_AUDITS.items()}
    ):
        raise RareLossExperimentError("Frozen candidate, fold, or source contract changed")
    if manifest.get("execution") != {
        "asset": "AAPL",
        "actions": ["LONG_100_PERCENT", "CASH_100_PERCENT"],
        "decision_information_cutoff": "completed close t",
        "fill": "next adjusted AAPL open",
        "shorting": False,
        "leverage": False,
        "maximum_target_exposure": 1.0,
        "cash_interest_rate": 0.0,
    }:
        raise RareLossExperimentError("Execution contract changed")
    payload_hashes = manifest.get("payload_sha256")
    expected_payloads = set(snapshot) - {
        ".gitattributes",
        "checksums.json",
        "selection_manifest.json",
        "report.json",
    }
    if not isinstance(payload_hashes, dict) or set(payload_hashes) != expected_payloads:
        raise RareLossExperimentError("Manifest payload map is not exact")
    for name, digest in payload_hashes.items():
        if _sha256_tagged(snapshot[name]) != digest:
            raise RareLossExperimentError(f"Manifest payload mismatch: {name}")

    rebuilt, rebuilt_results, rebuilt_refit = _rebuilt_payloads(snapshot)
    exact_payload_names = set(rebuilt) | {
        "development_prices_through_2018.csv",
        "development_context_through_2018.csv",
        "feature_provenance.json",
        "input_provenance.json",
    }
    if set(payload_hashes) != exact_payload_names:
        raise RareLossExperimentError("Artifact contains an undeclared semantic payload")
    if _parse_json(snapshot, "feature_provenance.json") != _feature_provenance():
        raise RareLossExperimentError("Feature provenance does not replay")
    for name, payload in rebuilt.items():
        if snapshot.get(name) != payload:
            raise RareLossExperimentError(f"Sealed payload does not replay: {name}")
    if _canonical_json_bytes(rebuilt_results) != _canonical_json_bytes(results):
        raise RareLossExperimentError("Candidate results do not replay")
    selected = select_development_candidate(rebuilt_results)
    if selected != manifest.get("selected_candidate_id"):
        raise RareLossExperimentError("Candidate selection does not replay")
    if bool(report.get("development_pass")) != (selected is not None):
        raise RareLossExperimentError("Development pass flag is inconsistent")
    if manifest.get("refit") != rebuilt_refit or report.get("refit") != rebuilt_refit:
        raise RareLossExperimentError("Final refit metadata does not replay")
    reproducibility = report.get("reproducibility", {})
    if (
        reproducibility.get("model_calls") != 0
        or reproducibility.get("model_calls_semantics")
        != "external_or_llm_model_calls_only"
        or reproducibility.get("local_numeric_forest_fits_before_seal")
        != 14 + (2 if selected is not None else 0)
        or reproducibility.get("llm_calls") != 0
        or reproducibility.get("network_calls") != 0
        or reproducibility.get("estimated_external_cost_usd") != 0.0
        or reproducibility.get("api_cost_display") != "$0.00"
    ):
        raise RareLossExperimentError("Zero-cost offline evidence changed")
    return {
        "verified": True,
        "run_id": report.get("run_id"),
        "development_pass": bool(report.get("development_pass")),
        "selected_candidate_id": selected,
        "artifact_dir": str(run_dir),
        "verified_git_commit": artifact_commit,
        "verified_source_commit": source_commit,
        "artifact_files": len(snapshot),
        "checksum_entries": len(checksums),
        "checksums_sha256": _sha256_tagged(snapshot["checksums.json"]),
    }


def verify_development_artifact(
    artifact: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a sealed bundle while preserving this runner's public error type."""

    try:
        return _verify_development_artifact(artifact, repo_root=repo_root)
    except RegimeConsensusExperimentError as exc:
        raise RareLossExperimentError(str(exc)) from exc


def run_development_from_inputs(
    *,
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    output_dir: Path,
    input_provenance: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    run_id: str | None = None,
    _deadline: _RareLossDeadline | None = None,
) -> dict[str, Any]:
    deadline = _deadline or _RareLossDeadline(clock)
    try:
        prices, context, boundary = _bounded_inputs(price_frame, context_frame)
    except RegimeConsensusExperimentError as exc:
        raise RareLossExperimentError(str(exc)) from exc
    deadline.check("bounded input validation")
    feature_label, predictions, states, targets, results, ledgers = _compute_development(
        prices, context, deadline=deadline
    )
    selected = select_development_candidate(results)
    model_payloads, refit = _fit_final_model(
        feature_label, selected, deadline=deadline
    )
    deadline.check("selected-model refit")

    created_at = datetime.now(timezone.utc)
    resolved_run_id = run_id or (
        f"rare-loss-development-{created_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resolved_run_id):
        raise RareLossExperimentError("run_id is not a safe directory name")
    root = output_dir.resolve()
    run_dir = root / resolved_run_id
    temporary = root / f".{resolved_run_id}.{uuid.uuid4().hex}.sealing"
    if run_dir.exists():
        raise RareLossExperimentError(f"Artifact already exists: {run_dir}")

    provenance = {
        "contract_version": CONTRACT_VERSION,
        "source": dict(input_provenance or {"source_type": "injected_pre2019_inputs"}),
        "source_identity": dict(source_identity or {}),
        "boundary_audit": boundary,
        "physical_data_end": "2018-12-31",
        "post_2018_market_data_accessed": False,
        "network_access": False,
        "api_calls": 0,
        "llm_calls": 0,
    }
    payloads: dict[str, bytes] = {
        "development_prices_through_2018.csv": _frame_csv_bytes(
            prices.reset_index(names="date")
        ),
        "development_context_through_2018.csv": _frame_csv_bytes(
            context.reset_index(names="date")
        ),
        "feature_provenance.json": _pretty_json_bytes(_feature_provenance()),
        "input_provenance.json": _pretty_json_bytes(provenance),
        "oof_predictions_2005_2018.csv": _frame_csv_bytes(predictions.reset_index()),
        "oof_forest_states.json": _pretty_json_bytes(states),
        "candidate_targets_2005_2018.csv": _frame_csv_bytes(targets.reset_index()),
        "candidate_results.json": _pretty_json_bytes(results),
        **model_payloads,
    }
    payloads.update(
        {
            name: _frame_csv_bytes(frame, float_format="%.17g")
            for name, frame in ledgers.items()
        }
    )
    payload_hashes = {
        name: _sha256_tagged(payload) for name, payload in sorted(payloads.items())
    }
    manifest_payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": "development",
        "evidence_classification": "chronological_internal_development_and_selection_not_unseen_test_evidence",
        "physical_data_end": "2018-12-31",
        "post_2018_market_data_access_allowed": False,
        "post_2018_market_data_accessed": False,
        "candidate_family": [item.to_dict() for item in CANDIDATE_SPECS],
        "model_variants": list(MODEL_VARIANTS),
        "walk_forward_folds": [asdict(item) for item in WALK_FORWARD_FOLDS],
        "development_forest_fit_count": 14,
        "cost_scenarios_bps": dict(COST_SCENARIOS),
        "selected_candidate_id": selected,
        "selection_tie_break": [
            "highest 10bps minimum-fold active-log edge",
            "highest 10bps total active-log edge",
            "highest severe-event Brier relative improvement",
            "fewest completed CASH episodes",
            "fewest CASH days",
            "lexical candidate id",
        ],
        "source_exclusion_audits": {
            key: dict(value) for key, value in SOURCE_EXCLUSION_AUDITS.items()
        },
        "candidate_results_sha256": _sha256_tagged(_canonical_json_bytes(results)),
        "payload_sha256": payload_hashes,
        "refit": refit,
        "execution": {
            "asset": "AAPL",
            "actions": ["LONG_100_PERCENT", "CASH_100_PERCENT"],
            "decision_information_cutoff": "completed close t",
            "fill": "next adjusted AAPL open",
            "shorting": False,
            "leverage": False,
            "maximum_target_exposure": 1.0,
            "cash_interest_rate": 0.0,
        },
        "later_stage_lock": {
            "only_selected_candidate_may_advance": True,
            "candidate_grid_may_change_after_this_run": False,
            "thresholds_may_change_after_this_run": False,
            "features_may_change_after_this_run": False,
            "forest_configuration_may_change_after_this_run": False,
            "failure_stops_price_market_only_family": True,
        },
    }
    manifest = {
        **manifest_payload,
        "manifest_sha256": _sha256_tagged(_canonical_json_bytes(manifest_payload)),
    }
    payloads["selection_manifest.json"] = _pretty_json_bytes(manifest)
    deadline.check("before artifact sealing")
    report = {
        "contract_version": CONTRACT_VERSION,
        "run_id": resolved_run_id,
        "stage": "development",
        "evidence_classification": manifest_payload["evidence_classification"],
        "artifact_dir": str(run_dir),
        "created_at_utc": created_at.isoformat(),
        "physical_data_end": "2018-12-31",
        "post_2018_market_data_accessed": False,
        "selected_candidate_id": selected,
        "development_pass": selected is not None,
        "candidate_count": len(CANDIDATE_SPECS),
        "candidate_results_sha256": manifest_payload["candidate_results_sha256"],
        "selection_manifest": manifest,
        "refit": refit,
        "reproducibility": {
            "runtime_seconds_before_seal": deadline.elapsed(),
            "runtime_limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "completed_before_seal_within_limit": deadline.elapsed()
            <= RUN_TIME_LIMIT_SECONDS,
            "model_calls": 0,
            "model_calls_semantics": "external_or_llm_model_calls_only",
            "local_numeric_forest_fits_before_seal": 14
            + (2 if selected is not None else 0),
            "llm_calls": 0,
            "network_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "api_cost_display": "$0.00",
        },
    }
    payloads["report.json"] = _pretty_json_bytes(report)
    promoted = False
    try:
        _seal_artifact_bundle(temporary, payloads)
        deadline.check("artifact checksum readback")
        verify_development_artifact(temporary)
        deadline.check("completed development replay verification")
        temporary.replace(run_dir)
        promoted = True
        deadline.check("atomic artifact promotion")
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        if promoted and run_dir.exists():
            shutil.rmtree(run_dir)
        if isinstance(exc, RegimeConsensusExperimentError):
            raise RareLossExperimentError(str(exc)) from exc
        raise
    return report


def run_development_experiment(
    *,
    repo_root: Path,
    price_artifact: Path,
    context_parquet: Path,
    output_dir: Path,
) -> dict[str, Any]:
    deadline = _RareLossDeadline(time.perf_counter)
    repo = repo_root.resolve()
    repository = validate_source_repository(repo, [repo / path for path in SOURCE_FILES])
    deadline.check("source repository validation")
    git = _git_state(repo)
    if git.get("dirty") is not False:
        raise RareLossExperimentError("Development requires committed source and a clean worktree")
    commit = str(git.get("commit") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise RareLossExperimentError("Could not capture source commit")
    source_hashes = _source_hashes_at_commit(repo, commit)
    deadline.check("source Git-blob hashing")
    loaded = load_public_development_inputs(
        price_artifact=price_artifact,
        context_parquet=context_parquet,
        repo_root=repo,
    )
    deadline.check("bounded public input loading")
    return run_development_from_inputs(
        price_frame=loaded.price_frame,
        context_frame=loaded.context_frame,
        output_dir=output_dir,
        input_provenance=loaded.provenance,
        source_identity={
            "repository": repository,
            "git": git,
            "source_hashes": source_hashes,
        },
        _deadline=deadline,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or verify the sealed through-2018 rare-loss experiment"
    )
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--price-artifact", type=Path)
    parser.add_argument("--context-parquet", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--development-artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        if args.development_artifact is None:
            raise SystemExit("verify requires --development-artifact")
        result = verify_development_artifact(
            args.development_artifact, repo_root=args.repo_root
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if any(
        value is None
        for value in (
            args.repo_root,
            args.price_artifact,
            args.context_parquet,
            args.output_dir,
        )
    ):
        raise SystemExit(
            "run requires --repo-root, --price-artifact, --context-parquet, and --output-dir"
        )
    report = run_development_experiment(
        repo_root=args.repo_root,
        price_artifact=args.price_artifact,
        context_parquet=args.context_parquet,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "development_pass": report["development_pass"],
                "selected_candidate_id": report["selected_candidate_id"],
                "artifact_dir": report["artifact_dir"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
