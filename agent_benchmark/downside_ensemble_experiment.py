"""Offline, physically pre-2019 development runner for the downside ensemble.

The module has one evidence role: choose at most one of eight predeclared
price/CFTC downside candidates using chronological 2005-2018 development
predictions.  It contains no later-period evaluation entry point.  The public
file-backed runner consumes a previously sealed CFTC development snapshot; the
in-memory entry point exists so the same contract can be tested without data
or network access.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cftc_cot import COTDownload, COTRetrievalMetadata
from .cftc_cot_experiment import (
    DEVELOPMENT_END as CFTC_DEVELOPMENT_END,
    DEVELOPMENT_INPUT_START,
    DEVELOPMENT_PRICE_START,
    _assert_stage_frame_bounds,
    _committed_artifact_snapshot,
    _parse_bounded_download,
    to_policy_records,
)
from .cftc_cot_policy import COTWeeklyRecord
from .deterministic_aapl import CostAssumptions, EvaluationPeriod, file_sha256
from .downside_features import (
    CFTC_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    build_downside_feature_label_frame,
    chronological_training_mask,
)
from .downside_forest import DownsideForest, ForestConfig
from .downside_scoring import (
    ACTIVE_EDGE_WIN_TOLERANCE,
    evaluate_downside_development,
)
from .downside_walkforward import (
    MODEL_FAMILIES,
    RISK_MULTIPLE_GATES,
    WALK_FORWARD_FOLDS,
    build_pre2019_walkforward_predictions,
    candidate_cash_target_column,
    five_session_cash_target,
)
from .unleveraged_aapl import (
    _git_state,
    canonical_context_frame,
    simulate_unleveraged_period,
    validate_source_repository,
)


CONTRACT_VERSION = "aapl-downside-ensemble-development-v1"
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
DEVELOPMENT_PERIOD = EvaluationPeriod(
    "downside_development", "2005-01-01", "2018-12-31"
)
BASE_COST_BPS = 5.0
STRESS_COST_BPS = 10.0
COST_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("base_5bps", BASE_COST_BPS),
    ("stress_10bps", STRESS_COST_BPS),
)
INITIAL_CASH = 1000.0
RUN_TIME_LIMIT_SECONDS = 3600.0
ARTIFACT_GIT_ATTRIBUTES = b"* -text\n"

SOURCE_FILES = (
    ".gitattributes",
    "agent_benchmark/cftc_cot.py",
    "agent_benchmark/cftc_cot_policy.py",
    "agent_benchmark/cftc_cot_experiment.py",
    "agent_benchmark/deterministic_aapl.py",
    "agent_benchmark/downside_features.py",
    "agent_benchmark/downside_forest.py",
    "agent_benchmark/downside_walkforward.py",
    "agent_benchmark/downside_scoring.py",
    "agent_benchmark/downside_ensemble_experiment.py",
    "agent_benchmark/unleveraged_aapl.py",
    "docs/downside_ensemble_experiment.md",
    "requirements.txt",
)


class DownsideEnsembleError(RuntimeError):
    """Raised when the sealed development contract is violated."""


class DownsideEnsembleTimeout(DownsideEnsembleError):
    """Raised when the fixed one-hour wall-clock budget is exhausted."""


@dataclass(frozen=True)
class CandidateSpec:
    model_family: str
    risk_multiple: float

    def __post_init__(self) -> None:
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(f"Unknown model family: {self.model_family!r}")
        if not any(float(self.risk_multiple) == gate for gate in RISK_MULTIPLE_GATES):
            raise ValueError("Candidate risk-multiple gate is not in the frozen grid")

    @property
    def candidate_id(self) -> str:
        return f"{self.model_family}_r{int(round(self.risk_multiple * 100)):03d}"

    @property
    def cash_target_column(self) -> str:
        return candidate_cash_target_column(
            self.model_family, self.risk_multiple
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "risk_multiple_gate": float(self.risk_multiple),
            "trigger": (
                "downside_probability >= risk_multiple_gate * "
                "causal_fold_training_prevalence"
            ),
            "predicted_mean_return_role": "diagnostic_only_not_a_trading_gate",
            "cash_decision_rows": 5,
        }


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(family, gate)
    for family in MODEL_FAMILIES
    for gate in RISK_MULTIPLE_GATES
)


REJECTED_ABSOLUTE_TRIGGER_PREFLIGHT: Mapping[str, Any] = {
    "evidence_stage": "pre2019_structural_development_preflight",
    "rejected_absolute_probability_gates": [0.25, 0.30, 0.35, 0.40],
    "rejected_predicted_mean_log_return_gate": -0.005,
    "observed_result": "zero_cash_targets_for_all_eight_candidates",
    "diagnosis": [
        "predicted mean-return floor remained above the rejected -0.005 gate",
        "maximum downside probability was approximately 0.26",
        "absolute probabilities were not comparable across changing fold prevalences",
    ],
    "permitted_response": (
        "replace the nonfunctional trigger before sealing with fixed causal "
        "relative-risk multiples; do not inspect any later period"
    ),
}


@dataclass(frozen=True)
class LoadedDevelopmentInputs:
    price_frame: pd.DataFrame
    cot_records: tuple[COTWeeklyRecord, ...]
    provenance: Mapping[str, Any]


class _Deadline:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.started = float(clock())

    def elapsed(self) -> float:
        return float(self._clock()) - self.started

    def check(self, step: str) -> None:
        elapsed = self.elapsed()
        if not math.isfinite(elapsed) or elapsed > RUN_TIME_LIMIT_SECONDS:
            raise DownsideEnsembleTimeout(
                f"Development exceeded {RUN_TIME_LIMIT_SECONDS:.0f} seconds at {step}"
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite numbers are forbidden in sealed JSON")
        return parsed
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_tagged(payload: bytes) -> str:
    return f"sha256:{_sha256_hex(payload)}"


def _frame_csv_bytes(
    frame: pd.DataFrame,
    *,
    float_format: str = "%.17g",
) -> bytes:
    return frame.to_csv(
        index=False,
        date_format="%Y-%m-%d",
        float_format=float_format,
        lineterminator="\n",
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _verify_checksum_snapshot(snapshot: Mapping[str, bytes]) -> dict[str, str]:
    try:
        raw_checksums = snapshot["checksums.json"]
        checksums = json.loads(raw_checksums.decode("utf-8", errors="strict"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownsideEnsembleError("Artifact has no valid checksum manifest") from exc
    expected_names = set(snapshot) - {"checksums.json"}
    if not isinstance(checksums, dict) or set(checksums) != expected_names:
        raise DownsideEnsembleError(
            "Checksum manifest does not cover the exact artifact file set"
        )
    for name, expected in checksums.items():
        if not isinstance(expected, str) or _sha256_hex(snapshot[name]) != expected:
            raise DownsideEnsembleError(f"Artifact checksum mismatch: {name}")
    return {str(name): str(value) for name, value in checksums.items()}


def _local_artifact_snapshot(run_dir: Path) -> dict[str, bytes]:
    if not run_dir.is_dir():
        raise DownsideEnsembleError(f"Artifact directory does not exist: {run_dir}")
    nested = [path for path in run_dir.rglob("*") if path.is_file() and path.parent != run_dir]
    if nested:
        raise DownsideEnsembleError("Artifact directory must not contain nested files")
    return {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}


def _artifact_snapshot(
    run_dir: Path,
    *,
    repo_root: Path | None,
) -> tuple[dict[str, bytes], str | None]:
    if repo_root is None:
        return _local_artifact_snapshot(run_dir), None
    git = _git_state(repo_root.resolve())
    head = str(git.get("commit") or "")
    if not head:
        raise DownsideEnsembleError("Could not capture Git HEAD for artifact verification")
    try:
        snapshot = _committed_artifact_snapshot(
            repo_root.resolve(), run_dir.resolve(), head_commit=head
        )
    except Exception as exc:
        raise DownsideEnsembleError(
            "Could not load the input artifact from immutable Git blobs"
        ) from exc
    return snapshot, head


def _parse_json_snapshot(snapshot: Mapping[str, bytes], name: str) -> Any:
    try:
        return json.loads(snapshot[name].decode("utf-8", errors="strict"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DownsideEnsembleError(f"Artifact contains invalid {name}") from exc


def load_cftc_development_artifact(
    artifact: Path,
    *,
    repo_root: Path | None = None,
) -> LoadedDevelopmentInputs:
    """Load only checksum-verified CFTC development inputs ending in 2018.

    The CFTC policy result is allowed to be rejected; this loader reuses only
    its independently bounded and audited source snapshot.
    """

    run_dir = artifact.resolve()
    if run_dir.is_file():
        if run_dir.name != "report.json":
            raise DownsideEnsembleError("Input artifact file must be report.json")
        run_dir = run_dir.parent
    snapshot, git_commit = _artifact_snapshot(run_dir, repo_root=repo_root)
    checksums = _verify_checksum_snapshot(snapshot)
    required = {
        "development_prices_through_2018.csv",
        "development_cftc_raw_through_2018.csv",
        "development_cftc_official_count_proof_through_2018.csv",
        "cftc_retrieval_metadata.json",
        "selection_manifest.json",
        "report.json",
    }
    missing = sorted(required.difference(snapshot))
    if missing:
        raise DownsideEnsembleError(f"Input artifact is missing required files: {missing}")

    manifest = _parse_json_snapshot(snapshot, "selection_manifest.json")
    report = _parse_json_snapshot(snapshot, "report.json")
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise DownsideEnsembleError("Input report and manifest must be JSON objects")
    if (
        manifest.get("stage") != "development"
        or manifest.get("physical_data_end") != "2018-12-31"
        or report.get("stage") != "development"
    ):
        raise DownsideEnsembleError(
            "Input artifact is not the physically bounded through-2018 development stage"
        )
    embedded_manifest_hash = manifest.get("manifest_sha256")
    manifest_without_hash = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if embedded_manifest_hash != _sha256_tagged(
        _canonical_json_bytes(manifest_without_hash)
    ):
        raise DownsideEnsembleError("Input selection manifest hash is invalid")
    if report.get("selection_manifest") != manifest:
        raise DownsideEnsembleError(
            "Input report does not contain the same sealed selection manifest"
        )

    price_bytes = snapshot["development_prices_through_2018.csv"]
    if manifest.get("price_sha256") != _sha256_tagged(price_bytes):
        raise DownsideEnsembleError("Input price bytes differ from their sealed manifest")
    try:
        raw_price = pd.read_csv(io.BytesIO(price_bytes), float_precision="round_trip")
    except Exception as exc:
        raise DownsideEnsembleError("Could not parse sealed development prices") from exc
    try:
        price_frame = _assert_stage_frame_bounds(
            raw_price,
            start=DEVELOPMENT_PRICE_START,
            end=CFTC_DEVELOPMENT_END,
            stage="downside_input_replay",
        )
    except Exception as exc:
        raise DownsideEnsembleError("Sealed price input failed its 2018 boundary") from exc

    metadata_payload = _parse_json_snapshot(snapshot, "cftc_retrieval_metadata.json")
    if not isinstance(metadata_payload, dict):
        raise DownsideEnsembleError("CFTC retrieval metadata must be a JSON object")
    metadata_payload = dict(metadata_payload)
    metadata_payload["contract_codes"] = tuple(metadata_payload["contract_codes"])
    try:
        download = COTDownload(
            raw_csv=snapshot["development_cftc_raw_through_2018.csv"],
            count_proof_csv=snapshot[
                "development_cftc_official_count_proof_through_2018.csv"
            ],
            metadata=COTRetrievalMetadata(**metadata_payload),
        )
        cleaned, coverage = _parse_bounded_download(
            download,
            expected_start=DEVELOPMENT_INPUT_START,
            expected_end=CFTC_DEVELOPMENT_END,
        )
        policy_records = to_policy_records(cleaned.records)
    except Exception as exc:
        raise DownsideEnsembleError("Sealed CFTC input failed bounded replay") from exc

    # A report dated in 2018 can have a mechanically derived release date in
    # 2019.  It is not usable by a 2018 decision and is removed before feature
    # construction, so the development runner never supplies it to a model.
    usable_records = tuple(
        item
        for item in policy_records
        if pd.Timestamp(item.availability_date) <= DEVELOPMENT_END
    )
    provenance = {
        "source_type": "checksum_verified_cftc_development_artifact",
        "source_artifact_dir": str(run_dir),
        "source_artifact_git_commit": git_commit,
        "source_report_run_id": report.get("run_id"),
        "source_manifest_sha256": _sha256_tagged(
            snapshot["selection_manifest.json"]
        ),
        "source_checksums_sha256": _sha256_tagged(snapshot["checksums.json"]),
        "source_checksums": checksums,
        "cftc_coverage_audit": coverage,
        "cftc_clean_records": len(policy_records),
        "cftc_records_available_by_2018_end": len(usable_records),
        "cftc_later_availability_rows_excluded": len(policy_records)
        - len(usable_records),
    }
    return LoadedDevelopmentInputs(price_frame, usable_records, provenance)


def _bounded_inputs(
    price_frame: pd.DataFrame,
    cot_records: Iterable[COTWeeklyRecord],
) -> tuple[pd.DataFrame, tuple[COTWeeklyRecord, ...], dict[str, Any]]:
    data = canonical_context_frame(price_frame)
    if data.index.max() > DEVELOPMENT_END:
        first = data.index[data.index > DEVELOPMENT_END][0]
        raise DownsideEnsembleError(
            "Development input contains a post-2018 price row: "
            f"{first.date().isoformat()}"
        )
    observed_years = set(int(value) for value in data.index.year)
    required_years = set(range(2005, 2019))
    if not required_years.issubset(observed_years):
        missing = sorted(required_years - observed_years)
        raise DownsideEnsembleError(
            f"Development prices do not cover every 2005-2018 year: {missing}"
        )
    if data.index.min().year > 2004:
        raise DownsideEnsembleError("Development prices lack pre-2005 training history")

    materialized = tuple(cot_records)
    for item in materialized:
        if pd.Timestamp(item.report_date) > DEVELOPMENT_END:
            raise DownsideEnsembleError("Development input contains a post-2018 CFTC report")
    usable = tuple(
        item
        for item in materialized
        if pd.Timestamp(item.availability_date) <= DEVELOPMENT_END
    )
    return data, usable, {
        "price_first_date": data.index.min().date().isoformat(),
        "price_last_date": data.index.max().date().isoformat(),
        "price_rows": int(len(data)),
        "cftc_supplied_records": len(materialized),
        "cftc_records_available_by_2018_end": len(usable),
        "cftc_later_availability_rows_excluded": len(materialized) - len(usable),
    }


def _cot_records_frame(records: Sequence[COTWeeklyRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market": item.market,
                "report_date": item.report_date,
                "availability_date": item.availability_date,
                "net_share": float(item.net_share),
            }
            for item in records
        ],
        columns=("market", "report_date", "availability_date", "net_share"),
    )


def _family_probability_columns(model_family: str) -> tuple[str, str, str]:
    if model_family == "price_only":
        return (
            "price_only_downside_probability",
            "price_only_baseline_downside_probability",
            "price_only_predicted_mean_clipped_return",
        )
    if model_family == "price_cftc":
        return (
            "price_cftc_downside_probability",
            "price_cftc_effective_baseline_downside_probability",
            "price_cftc_predicted_mean_clipped_return",
        )
    raise ValueError(f"Unknown model family: {model_family!r}")


def _brier_inputs(
    predictions: pd.DataFrame,
    model_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    probability_column, baseline_column, _ = _family_probability_columns(model_family)
    probability = pd.to_numeric(
        predictions[probability_column], errors="coerce"
    ).to_numpy(dtype=float)
    baseline = pd.to_numeric(
        predictions[baseline_column], errors="coerce"
    ).to_numpy(dtype=float)
    label = pd.to_numeric(
        predictions["crash_label_5session"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_prediction = np.isfinite(probability) & np.isfinite(baseline)
    if not finite_prediction.any():
        raise DownsideEnsembleError(
            f"No finite out-of-fold predictions exist for {model_family}"
        )
    return (
        probability[finite_prediction],
        label[finite_prediction],
        baseline[finite_prediction],
        {
            "oof_rows": int(len(probability)),
            "brier_prediction_rows": int(finite_prediction.sum()),
            "unavailable_prediction_rows": int((~finite_prediction).sum()),
        },
    )


def _full_exposure_target(
    price_index: pd.DatetimeIndex,
    prediction_index: pd.DatetimeIndex,
    cash_target: Sequence[int],
) -> pd.Series:
    if not prediction_index.isin(price_index).all():
        raise DownsideEnsembleError("Prediction dates are absent from the price frame")
    cash = np.asarray(cash_target, dtype=float)
    if cash.ndim != 1 or len(cash) != len(prediction_index):
        raise DownsideEnsembleError("Candidate CASH target is not prediction-aligned")
    if not np.isin(cash, (0.0, 1.0)).all():
        raise DownsideEnsembleError("Candidate CASH target must be binary")
    exposure = pd.Series(1.0, index=price_index, name="target_exposure")
    exposure.loc[prediction_index] = 1.0 - cash
    return exposure


def _common_support_cash_target(
    predictions: pd.DataFrame,
    spec: CandidateSpec,
) -> np.ndarray:
    probability_column, baseline_column, _ = _family_probability_columns(
        spec.model_family
    )
    probability = pd.to_numeric(
        predictions[probability_column], errors="coerce"
    ).to_numpy(dtype=float)
    baseline = pd.to_numeric(
        predictions[baseline_column], errors="coerce"
    ).to_numpy(dtype=float)
    support = (
        predictions["price_features_ready"].astype(bool).to_numpy()
        & predictions["cftc_features_finite"].astype(bool).to_numpy()
        & ~predictions["price_cftc_used_fallback"].astype(bool).to_numpy()
    )
    trigger = (
        support
        & np.isfinite(probability)
        & np.isfinite(baseline)
        & (baseline > 0.0)
        & (probability >= spec.risk_multiple * baseline)
    )
    return five_session_cash_target(trigger)


def _simulate(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    return simulate_unleveraged_period(
        frame,
        target,
        DEVELOPMENT_PERIOD,
        CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0),
        initial_cash=INITIAL_CASH,
    )


def _score_strategy(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    brier: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]],
    cost_bps: float,
) -> dict[str, Any]:
    probability, label, baseline, availability = brier
    result = evaluate_downside_development(
        strategy,
        benchmark,
        strategy["target_exposure"].to_numpy(dtype=float),
        probability,
        label,
        baseline,
        cost_bps=cost_bps,
    )
    return {**result, "prediction_availability": availability}


def common_support_ablation(
    common_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    risk_multiple: float,
) -> dict[str, Any]:
    """Compare augmented and price models on identical CFTC-ready opportunities."""

    gate_key = f"r{int(round(float(risk_multiple) * 100)):03d}"
    scenarios: dict[str, Any] = {}
    all_checks: list[bool] = []
    for scenario_name, _ in COST_SCENARIOS:
        try:
            price = common_results[f"price_only_{gate_key}"][scenario_name]["metrics"]
            augmented = common_results[f"price_cftc_{gate_key}"][scenario_name][
                "metrics"
            ]
        except KeyError as exc:
            raise DownsideEnsembleError(
                f"Common-support result is incomplete for {gate_key}"
            ) from exc
        checks = {
            "strict_total_active_log_improvement": (
                float(augmented["total_active_log_edge"])
                > float(price["total_active_log_edge"])
                + ACTIVE_EDGE_WIN_TOLERANCE
            ),
            "weakest_fold_not_worse": (
                float(augmented["minimum_fold_active_log_edge"])
                + ACTIVE_EDGE_WIN_TOLERANCE
                >= float(price["minimum_fold_active_log_edge"])
            ),
        }
        scenarios[scenario_name] = {
            "price_only_total_active_log_edge": float(
                price["total_active_log_edge"]
            ),
            "price_cftc_total_active_log_edge": float(
                augmented["total_active_log_edge"]
            ),
            "price_only_minimum_fold_active_log_edge": float(
                price["minimum_fold_active_log_edge"]
            ),
            "price_cftc_minimum_fold_active_log_edge": float(
                augmented["minimum_fold_active_log_edge"]
            ),
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
        all_checks.extend(checks.values())
    return {
        "applicable": True,
        "comparison": "identical CFTC-ready trigger support; active episodes continue",
        "active_edge_win_tolerance": ACTIVE_EDGE_WIN_TOLERANCE,
        "scenarios": scenarios,
        "passed": bool(all(all_checks)),
    }


def select_development_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> str | None:
    """Select only a passing candidate with the predeclared deterministic rank."""

    passing = [item for item in candidate_results if bool(item.get("passed"))]
    if not passing:
        return None

    def rank(item: Mapping[str, Any]) -> tuple[float, float, float, int, int, str]:
        stress = item["scenarios"]["stress_10bps"]["metrics"]
        family = str(item["candidate"]["model_family"])
        return (
            -float(stress["minimum_fold_active_log_edge"]),
            -float(stress["total_active_log_edge"]),
            -float(stress["brier_improvement"]),
            int(stress["cash_days"]),
            0 if family == "price_only" else 1,
            str(item["candidate"]["candidate_id"]),
        )

    return str(sorted(passing, key=rank)[0]["candidate"]["candidate_id"])


def _refit_one_model(
    feature_label_frame: pd.DataFrame,
    *,
    model_family: str,
    deadline: _Deadline,
) -> tuple[DownsideForest, dict[str, Any]]:
    require_cftc = model_family == "price_cftc"
    mask = chronological_training_mask(
        feature_label_frame,
        as_of_date=DEVELOPMENT_END,
        require_cftc=require_cftc,
        strictly_before_as_of=False,
    )
    future = pd.to_numeric(
        feature_label_frame["aapl_forward_log_return_5"], errors="coerce"
    )
    mask &= np.isfinite(future.to_numpy(dtype=float))
    training = feature_label_frame.loc[mask].copy()
    names = (
        PRICE_FEATURE_COLUMNS
        if model_family == "price_only"
        else PRICE_FEATURE_COLUMNS + CFTC_FEATURE_COLUMNS
    )
    namespace = f"downside-development-v1|{model_family}|refit-through-2018-12-31"
    first = DownsideForest(ForestConfig()).fit(
        training.loc[:, list(names)],
        training["aapl_forward_log_return_5"].to_numpy(dtype=float),
        namespace=namespace,
    )
    deadline.check(f"first deterministic {model_family} refit")
    second = DownsideForest(ForestConfig()).fit(
        training.loc[:, list(names)],
        training["aapl_forward_log_return_5"].to_numpy(dtype=float),
        namespace=namespace,
    )
    deadline.check(f"second deterministic {model_family} refit")
    first_bytes = (first.canonical_json() + "\n").encode("utf-8")
    second_bytes = (second.canonical_json() + "\n").encode("utf-8")
    if first_bytes != second_bytes:
        raise DownsideEnsembleError(
            f"Repeated {model_family} final refits were not byte-identical"
        )
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    return first, {
        "model_family": model_family,
        "model_sha256": first.model_sha256,
        "state_file_sha256": _sha256_tagged(first_bytes),
        "training_rows": int(len(training)),
        "training_first_decision": training.index.min().date().isoformat(),
        "training_last_decision": training.index.max().date().isoformat(),
        "training_latest_label_maturity": maturity.max().date().isoformat(),
        "refit_repeated_byte_identically": True,
    }


def _refit_selected_models(
    feature_label_frame: pd.DataFrame,
    selected: CandidateSpec | None,
    *,
    deadline: _Deadline,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    if selected is None:
        return {}, {"performed": False, "reason": "no_development_candidate_passed"}
    families = (
        ("price_only", "price_cftc")
        if selected.model_family == "price_cftc"
        else ("price_only",)
    )
    states: dict[str, bytes] = {}
    metadata: dict[str, Any] = {}
    for family in families:
        model, item = _refit_one_model(
            feature_label_frame, model_family=family, deadline=deadline
        )
        filename = f"final_{family}_model_through_2018.json"
        states[filename] = (model.canonical_json() + "\n").encode("utf-8")
        metadata[family] = item
    return states, {
        "performed": True,
        "selected_candidate_id": selected.candidate_id,
        "training_cutoff": "2018-12-31",
        "models": metadata,
    }


def _seal_artifact_bundle(
    run_dir: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, str]:
    if run_dir.exists():
        raise DownsideEnsembleError(f"Artifact directory already exists: {run_dir}")
    if any("/" in name or "\\" in name for name in payloads):
        raise DownsideEnsembleError("Artifact payload names must be flat")
    forbidden = {".gitattributes", "checksums.json"}.intersection(payloads)
    if forbidden:
        raise DownsideEnsembleError(f"Reserved artifact names supplied: {sorted(forbidden)}")
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_write_bytes(run_dir / ".gitattributes", ARTIFACT_GIT_ATTRIBUTES)
    for name in sorted(payloads):
        _atomic_write_bytes(run_dir / name, payloads[name])
    checksums = {
        path.name: _sha256_hex(path.read_bytes())
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    _atomic_write_bytes(run_dir / "checksums.json", _pretty_json_bytes(checksums))
    snapshot = _local_artifact_snapshot(run_dir)
    verified = _verify_checksum_snapshot(snapshot)
    if verified != checksums:
        raise DownsideEnsembleError("Artifact checksum readback changed unexpectedly")
    return checksums


def verify_development_artifact(
    artifact: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    run_dir = artifact.resolve()
    if run_dir.is_file():
        if run_dir.name != "report.json":
            raise DownsideEnsembleError("Verification file must be report.json")
        run_dir = run_dir.parent
    snapshot, git_commit = _artifact_snapshot(run_dir, repo_root=repo_root)
    checksums = _verify_checksum_snapshot(snapshot)
    report = _parse_json_snapshot(snapshot, "report.json")
    manifest = _parse_json_snapshot(snapshot, "selection_manifest.json")
    results = _parse_json_snapshot(snapshot, "candidate_results.json")
    if not isinstance(report, dict) or not isinstance(manifest, dict):
        raise DownsideEnsembleError("Development report and manifest must be objects")
    if (
        report.get("contract_version") != CONTRACT_VERSION
        or report.get("stage") != "development"
        or report.get("physical_data_end") != "2018-12-31"
        or report.get("post_2018_market_data_accessed") is not False
        or manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("stage") != "development"
    ):
        raise DownsideEnsembleError("Artifact violates the pre-2019 development identity")
    if snapshot.get(".gitattributes") != ARTIFACT_GIT_ATTRIBUTES:
        raise DownsideEnsembleError("Artifact byte-preservation rule is missing or altered")
    manifest_hash = manifest.get("manifest_sha256")
    manifest_payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest_hash != _sha256_tagged(_canonical_json_bytes(manifest_payload)):
        raise DownsideEnsembleError("Selection manifest hash is invalid")
    if manifest.get("candidate_results_sha256") != _sha256_tagged(
        _canonical_json_bytes(results)
    ):
        raise DownsideEnsembleError("Candidate results are not bound by the manifest")
    if report.get("selection_manifest") != manifest:
        raise DownsideEnsembleError(
            "Report and selection-manifest artifacts do not contain the same manifest"
        )
    if report.get("candidate_results_sha256") != manifest.get(
        "candidate_results_sha256"
    ):
        raise DownsideEnsembleError("Report does not bind the candidate results")
    if not isinstance(results, list) or len(results) != len(CANDIDATE_SPECS):
        raise DownsideEnsembleError("Candidate result set is not the frozen eight")
    expected_ids = [item.candidate_id for item in CANDIDATE_SPECS]
    observed_ids = [
        str((item.get("candidate") or {}).get("candidate_id"))
        for item in results
        if isinstance(item, dict)
    ]
    if observed_ids != expected_ids:
        raise DownsideEnsembleError("Candidate result order or identity changed")
    for item in results:
        if set((item.get("scenarios") or {})) != {
            name for name, _ in COST_SCENARIOS
        }:
            raise DownsideEnsembleError("A candidate does not contain both cost scenarios")
    payload_hashes = manifest.get("payload_sha256")
    if not isinstance(payload_hashes, dict):
        raise DownsideEnsembleError("Manifest has no payload hash map")
    expected_payload_names = set(snapshot) - {
        ".gitattributes",
        "checksums.json",
        "selection_manifest.json",
        "report.json",
    }
    if set(payload_hashes) != expected_payload_names:
        raise DownsideEnsembleError("Manifest payload map is not exact")
    for name, expected in payload_hashes.items():
        if _sha256_tagged(snapshot[name]) != expected:
            raise DownsideEnsembleError(f"Manifest payload mismatch: {name}")
    if report.get("selected_candidate_id") != manifest.get("selected_candidate_id"):
        raise DownsideEnsembleError("Selected candidate differs across sealed records")
    selected = select_development_candidate(results)
    if selected != manifest.get("selected_candidate_id"):
        raise DownsideEnsembleError("Sealed selection does not replay from candidate results")
    if bool(report.get("development_pass")) != (selected is not None):
        raise DownsideEnsembleError("Development pass flag disagrees with frozen selection")

    for name in (
        "development_prices_through_2018.csv",
        "oof_predictions_2005_2018.csv",
        "candidate_targets_2005_2018.csv",
    ):
        try:
            frame = pd.read_csv(io.BytesIO(snapshot[name]))
            dates = pd.to_datetime(
                frame["date" if name.startswith("development_prices") else "decision_date"],
                errors="raise",
            )
        except (KeyError, ValueError) as exc:
            raise DownsideEnsembleError(f"Could not verify bounded dates in {name}") from exc
        if bool((dates > DEVELOPMENT_END).any()):
            raise DownsideEnsembleError(f"Sealed artifact contains a post-2018 row: {name}")
    cot_frame = pd.read_csv(
        io.BytesIO(snapshot["development_cftc_records_available_through_2018.csv"])
    )
    for column in ("report_date", "availability_date"):
        try:
            dates = pd.to_datetime(cot_frame[column], errors="raise")
        except (KeyError, ValueError) as exc:
            raise DownsideEnsembleError(
                f"Could not verify bounded CFTC {column}"
            ) from exc
        if bool((dates > DEVELOPMENT_END).any()):
            raise DownsideEnsembleError(
                f"Sealed CFTC evidence contains a post-2018 {column}"
            )
    return {
        "verified": True,
        "run_id": report.get("run_id"),
        "development_pass": bool(report.get("development_pass")),
        "selected_candidate_id": report.get("selected_candidate_id"),
        "artifact_dir": str(run_dir),
        "verified_git_commit": git_commit,
        "artifact_files": len(snapshot),
        "checksums_sha256": _sha256_tagged(snapshot["checksums.json"]),
        "checksum_entries": len(checksums),
    }


def run_development_from_inputs(
    *,
    price_frame: pd.DataFrame,
    cot_records: Iterable[COTWeeklyRecord],
    output_dir: Path,
    input_provenance: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run and seal the complete through-2018 development experiment."""

    deadline = _Deadline(clock)
    data, bounded_records, boundary_audit = _bounded_inputs(price_frame, cot_records)
    deadline.check("bounded input validation")

    feature_label = build_downside_feature_label_frame(
        data, cot_records=bounded_records
    )
    if feature_label.index.max() > DEVELOPMENT_END:
        raise DownsideEnsembleError("Feature construction produced a post-2018 row")
    deadline.check("feature and label construction")

    predictions = build_pre2019_walkforward_predictions(feature_label)
    if predictions.index.max() > DEVELOPMENT_END:
        raise DownsideEnsembleError("Walk-forward produced a post-2018 prediction")
    deadline.check("fourteen frozen fold fits")

    brier_by_family = {
        family: _brier_inputs(predictions, family) for family in MODEL_FAMILIES
    }
    target_audit = pd.DataFrame(index=predictions.index)
    target_audit.index.name = "decision_date"
    full_targets: dict[str, pd.Series] = {}
    common_targets: dict[str, pd.Series] = {}
    for spec in CANDIDATE_SPECS:
        actual_cash = predictions[spec.cash_target_column].to_numpy(dtype=np.int8)
        common_cash = _common_support_cash_target(predictions, spec)
        target_audit[f"{spec.candidate_id}_cash"] = actual_cash
        target_audit[f"{spec.candidate_id}_common_support_cash"] = common_cash
        full_targets[spec.candidate_id] = _full_exposure_target(
            data.index, predictions.index, actual_cash
        )
        common_targets[spec.candidate_id] = _full_exposure_target(
            data.index, predictions.index, common_cash
        )

    benchmark_target = pd.Series(1.0, index=data.index, name="target_exposure")
    benchmarks: dict[str, pd.DataFrame] = {}
    ledger_frames: dict[str, pd.DataFrame] = {}
    for scenario_name, bps in COST_SCENARIOS:
        benchmark = _simulate(data, benchmark_target, cost_bps=bps)
        benchmarks[scenario_name] = benchmark
        ledger_frames[f"buy_hold_{scenario_name}.csv"] = benchmark
        deadline.check(f"{scenario_name} buy-and-hold simulation")

    common_results: dict[str, dict[str, Any]] = {}
    for spec in CANDIDATE_SPECS:
        scenario_results: dict[str, Any] = {}
        for scenario_name, bps in COST_SCENARIOS:
            strategy = _simulate(data, common_targets[spec.candidate_id], cost_bps=bps)
            scored = _score_strategy(
                strategy,
                benchmarks[scenario_name],
                brier=brier_by_family[spec.model_family],
                cost_bps=bps,
            )
            scenario_results[scenario_name] = scored
            ledger_frames[
                f"common_{spec.candidate_id}_{scenario_name}_strategy.csv"
            ] = strategy
            deadline.check(
                f"{spec.candidate_id} {scenario_name} common-support simulation"
            )
        common_results[spec.candidate_id] = scenario_results

    candidate_results: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        scenarios: dict[str, Any] = {}
        for scenario_name, bps in COST_SCENARIOS:
            strategy = _simulate(data, full_targets[spec.candidate_id], cost_bps=bps)
            scenarios[scenario_name] = _score_strategy(
                strategy,
                benchmarks[scenario_name],
                brier=brier_by_family[spec.model_family],
                cost_bps=bps,
            )
            ledger_frames[f"{spec.candidate_id}_{scenario_name}_strategy.csv"] = strategy
            deadline.check(f"{spec.candidate_id} {scenario_name} simulation")
        general_pass = bool(
            all(item["gates"]["passed"] for item in scenarios.values())
        )
        ablation = (
            common_support_ablation(
                common_results, risk_multiple=spec.risk_multiple
            )
            if spec.model_family == "price_cftc"
            else {
                "applicable": False,
                "reason": "price-only is the ablation baseline",
                "passed": True,
            }
        )
        candidate_results.append(
            {
                "candidate": spec.to_dict(),
                "stage": "development_only",
                "scenarios": scenarios,
                "both_cost_gates_passed": general_pass,
                "common_support_cftc_ablation": ablation,
                "passed": bool(general_pass and ablation["passed"]),
            }
        )

    selected_id = select_development_candidate(candidate_results)
    selected_spec = next(
        (item for item in CANDIDATE_SPECS if item.candidate_id == selected_id),
        None,
    )
    model_payloads, refit = _refit_selected_models(
        feature_label, selected_spec, deadline=deadline
    )
    deadline.check("selected-model deterministic refit")

    created_at = datetime.now(timezone.utc)
    resolved_run_id = run_id or (
        f"downside-development-{created_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resolved_run_id):
        raise DownsideEnsembleError("run_id is not a safe flat directory name")
    run_dir = output_dir.resolve() / resolved_run_id

    price_payload = _frame_csv_bytes(
        data.reset_index(names="date")
    )
    cot_payload = _frame_csv_bytes(_cot_records_frame(bounded_records))
    predictions_payload = _frame_csv_bytes(
        predictions.reset_index(), float_format="%.17g"
    )
    targets_payload = _frame_csv_bytes(
        target_audit.reset_index(), float_format="%.17g"
    )
    results_payload = _pretty_json_bytes(candidate_results)
    provenance = {
        "contract_version": CONTRACT_VERSION,
        "source": dict(input_provenance or {"source_type": "injected_pre2019_inputs"}),
        "source_identity": dict(source_identity or {}),
        "boundary_audit": boundary_audit,
        "physical_data_end": "2018-12-31",
        "post_2018_market_data_accessed": False,
    }

    payloads: dict[str, bytes] = {
        "development_prices_through_2018.csv": price_payload,
        "development_cftc_records_available_through_2018.csv": cot_payload,
        "input_provenance.json": _pretty_json_bytes(provenance),
        "oof_predictions_2005_2018.csv": predictions_payload,
        "candidate_targets_2005_2018.csv": targets_payload,
        "candidate_results.json": results_payload,
        **model_payloads,
    }
    for filename, ledger in sorted(ledger_frames.items()):
        payloads[filename] = _frame_csv_bytes(ledger, float_format="%.12g")

    payload_hashes = {
        name: _sha256_tagged(payload) for name, payload in sorted(payloads.items())
    }
    manifest_payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": "development",
        "evidence_classification": (
            "chronological_internal_development_and_selection_not_unseen_test_evidence"
        ),
        "physical_data_end": "2018-12-31",
        "post_2018_market_data_access_allowed": False,
        "post_2018_market_data_accessed": False,
        "candidate_family": [item.to_dict() for item in CANDIDATE_SPECS],
        "rejected_absolute_trigger_preflight": dict(
            REJECTED_ABSOLUTE_TRIGGER_PREFLIGHT
        ),
        "walk_forward_folds": [asdict(item) for item in WALK_FORWARD_FOLDS],
        "forest_config": asdict(ForestConfig()),
        "cost_scenarios_bps": {
            name: bps for name, bps in COST_SCENARIOS
        },
        "selected_candidate_id": selected_id,
        "selection_tie_break": [
            "highest 10bps minimum fold active-log edge",
            "highest 10bps total active-log edge",
            "highest causal Brier improvement",
            "fewer 10bps cash days",
            "price-only before price+CFTC",
            "lexical candidate id",
        ],
        "candidate_results_sha256": _sha256_tagged(
            _canonical_json_bytes(candidate_results)
        ),
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
        "selected_candidate_id": selected_id,
        "development_pass": selected_id is not None,
        "candidate_count": len(CANDIDATE_SPECS),
        "candidate_results_sha256": manifest_payload["candidate_results_sha256"],
        "selection_manifest": manifest,
        "rejected_absolute_trigger_preflight": dict(
            REJECTED_ABSOLUTE_TRIGGER_PREFLIGHT
        ),
        "refit": refit,
        "reproducibility": {
            "runtime_seconds_before_seal": deadline.elapsed(),
            "runtime_limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "completed_before_seal_within_limit": (
                deadline.elapsed() <= RUN_TIME_LIMIT_SECONDS
            ),
            "model_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "api_cost_display": "$0.00",
        },
    }
    payloads["report.json"] = _pretty_json_bytes(report)
    _seal_artifact_bundle(run_dir, payloads)
    deadline.check("artifact checksum readback")
    verify_development_artifact(run_dir)
    deadline.check("completed development verification")
    return report


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {path: file_sha256(repo_root / path) for path in SOURCE_FILES}


def run_development_experiment(
    *,
    repo_root: Path,
    input_artifact: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run from a committed, checksum-verified through-2018 input artifact."""

    repo = repo_root.resolve()
    source_paths = [repo / path for path in SOURCE_FILES]
    repository = validate_source_repository(repo, source_paths)
    git = _git_state(repo)
    if git.get("dirty") is not False:
        raise DownsideEnsembleError(
            "Development requires committed source and a clean worktree"
        )
    loaded = load_cftc_development_artifact(
        input_artifact, repo_root=repo
    )
    return run_development_from_inputs(
        price_frame=loaded.price_frame,
        cot_records=loaded.cot_records,
        output_dir=output_dir,
        input_provenance=loaded.provenance,
        source_identity={
            "repository": repository,
            "git": git,
            "source_hashes": _source_hashes(repo),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or verify the sealed through-2018 downside development experiment"
    )
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--input-artifact", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--development-artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        if args.development_artifact is None:
            raise SystemExit("verify requires --development-artifact")
        if args.input_artifact is not None or args.output_dir is not None:
            raise SystemExit("verify does not accept run inputs")
        verification = verify_development_artifact(
            args.development_artifact,
            repo_root=args.repo_root,
        )
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0
    if args.repo_root is None or args.input_artifact is None or args.output_dir is None:
        raise SystemExit("run requires --repo-root, --input-artifact, and --output-dir")
    if args.development_artifact is not None:
        raise SystemExit("run does not accept --development-artifact")
    report = run_development_experiment(
        repo_root=args.repo_root,
        input_artifact=args.input_artifact,
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
    return 0 if report["development_pass"] else 2


__all__ = [
    "BASE_COST_BPS",
    "CANDIDATE_SPECS",
    "CONTRACT_VERSION",
    "CandidateSpec",
    "DownsideEnsembleError",
    "DownsideEnsembleTimeout",
    "LoadedDevelopmentInputs",
    "RUN_TIME_LIMIT_SECONDS",
    "STRESS_COST_BPS",
    "common_support_ablation",
    "load_cftc_development_artifact",
    "run_development_experiment",
    "run_development_from_inputs",
    "select_development_candidate",
    "verify_development_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
