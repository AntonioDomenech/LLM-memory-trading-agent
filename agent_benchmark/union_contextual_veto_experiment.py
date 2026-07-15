"""Sealed staged runner for the preregistered union contextual-veto experiment."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import chronological_exhaustion_experiment as _sealed_base
from .chronological_exhaustion_expert import canonicalize_one_session_signals
from .deterministic_aapl import EvaluationPeriod
from .union_contextual_veto import (
    CAUSAL_ONLINE_MODE,
    FROZEN_CUTOFF_MODE,
    build_union_contextual_veto_forecast,
)
from .unleveraged_aapl import canonical_context_frame


CONTRACT_VERSION = "aapl-union-contextual-veto-v1"
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
VALIDATION_END = pd.Timestamp("2023-12-31")
FINAL_END = pd.Timestamp("2026-07-09")
DEVELOPMENT_START = pd.Timestamp("2005-01-01")
VALIDATION_START = pd.Timestamp("2019-01-01")
FINAL_START = pd.Timestamp("2024-01-01")
INITIAL_CASH = 1000.0
RUN_TIME_LIMIT_SECONDS = 3600.0
STRICT_UNION_IMPROVEMENT = 0.0001
FINAL_MATERIAL_ACTIVE_LOG_EDGE = 0.001
COST_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("base_5bps", 5.0),
    ("stress_10bps", 10.0),
)
CONTRACT_PATH = Path("docs/aapl_union_contextual_veto_v1.md")
IMPLEMENTATION_PATHS = (
    Path("agent_benchmark/union_contextual_veto.py"),
    Path("agent_benchmark/union_contextual_veto_experiment.py"),
    Path("agent_benchmark/chronological_exhaustion_expert.py"),
    # This runner deliberately reuses the already audited physical-input and
    # sealing primitives from this dependency, so its exact bytes are bound.
    Path("agent_benchmark/chronological_exhaustion_experiment.py"),
    Path("agent_benchmark/deterministic_aapl.py"),
    Path("agent_benchmark/unleveraged_aapl.py"),
)
FEATURE_COLUMNS = (
    "feature_intercept",
    "feature_weak_only",
    "feature_expert_overlap",
    "feature_tail_strength",
    "feature_market_sentiment_10",
    "feature_market_sentiment_20",
    "feature_aapl_trend",
)
STATE_SCALAR_COLUMNS = (
    "n_raw",
    "n_eff",
    "wins",
    "label_sum",
    "squared_label_sum",
)
UNION_REFERENCE = {
    "base_5bps": {
        "episodes": 121,
        "total_active_log_edge": 1.092826209473698,
    },
    "stress_10bps": {
        "episodes": 121,
        "total_active_log_edge": 0.9718261388903109,
    },
}

# Reuse the already committed physical-snapshot commitments without opening a
# later file during implementation or an unauthorized stage.
PHYSICAL_SNAPSHOT_COLUMNS = _sealed_base.PHYSICAL_SNAPSHOT_COLUMNS
STAGE_SESSION_COVERAGE = _sealed_base.STAGE_SESSION_COVERAGE
STAGE_BOUNDED_RESULT_SHA256 = _sealed_base.STAGE_BOUNDED_RESULT_SHA256
PRICE_QUERY = _sealed_base.PRICE_QUERY

UnionContextualVetoExperimentError = (
    _sealed_base.ChronologicalExhaustionExperimentError
)
_Deadline = _sealed_base._Deadline
_canonical_json_bytes = _sealed_base._canonical_json_bytes
_pretty_json_bytes = _sealed_base._pretty_json_bytes
_sha256 = _sealed_base._sha256
_frame_csv_bytes = _sealed_base._frame_csv_bytes
_safe_run_id = _sealed_base._safe_run_id
_manifest = _sealed_base._manifest
_seal_bundle = _sealed_base._seal_bundle
_git_bytes = _sealed_base._git_bytes
_git_text = _sealed_base._git_text
_tracked_input_identity = _sealed_base._tracked_input_identity
load_bounded_prices = _sealed_base.load_bounded_prices
_positive_concentration = _sealed_base._positive_concentration
_prefix_sha256 = _sealed_base._prefix_sha256


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnionContextualVetoExperimentError(
            f"{description} is unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise UnionContextualVetoExperimentError(
            f"{description} must be a JSON object"
        )
    return value


def _clean_git_identity(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    try:
        actual = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
        status = _git_text(root, "status", "--porcelain", "--untracked-files=all")
        branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
        commit = _git_text(root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise UnionContextualVetoExperimentError(
            "Stage requires a valid Git repository"
        ) from exc
    if actual != root:
        raise UnionContextualVetoExperimentError(
            "repo_root must be the actual Git root"
        )
    if status:
        raise UnionContextualVetoExperimentError(
            "Stage requires a completely clean worktree"
        )
    if not branch or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise UnionContextualVetoExperimentError(
            "Stage requires an attached valid Git commit"
        )
    tracked_hashes: dict[str, str] = {}
    for relative in (*IMPLEMENTATION_PATHS, CONTRACT_PATH):
        name = relative.as_posix()
        try:
            _git_bytes(root, "ls-files", "--error-unmatch", "--", name)
            committed = _git_bytes(root, "show", f"HEAD:{name}")
        except subprocess.CalledProcessError as exc:
            raise UnionContextualVetoExperimentError(
                f"Frozen dependency is not tracked: {name}"
            ) from exc
        # A clean index/worktree is the authority. Hash the committed blob,
        # not checkout bytes, because Git's clean/smudge and CRLF conversion
        # may legitimately make those byte streams differ on Windows.
        tracked_hashes[name] = _sha256(committed)
    return {
        "branch": branch,
        "commit": commit,
        "dirty": False,
        "tracked_dependency_sha256": tracked_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "duckdb": __import__("duckdb").__version__,
        },
    }


def _required_parent_payloads(stage: str) -> frozenset[str]:
    if stage == "development":
        prefixes = ("development",)
        fixed = {
            ".gitattributes",
            "report.json",
            "development_prices_through_2018.csv",
            "development_forecast_through_2018.csv",
            "development_metrics.json",
            "development_gate_report.json",
            "development_checkpoint_through_2018.json",
            "input_provenance.json",
        }
    elif stage == "validation":
        prefixes = ("validation_online", "validation_frozen")
        fixed = {
            ".gitattributes",
            "report.json",
            "authorized_development_manifest.json",
            "validation_prices_through_2023.csv",
            "validation_online_forecast.csv",
            "validation_frozen_forecast.csv",
            "validation_online_metrics.json",
            "validation_frozen_metrics.json",
            "validation_gate_report.json",
            "validation_checkpoint_through_2023.json",
            "input_provenance.json",
        }
    else:
        raise UnionContextualVetoExperimentError(
            f"Unsupported parent stage: {stage}"
        )
    for prefix in prefixes:
        for cost_name, _ in COST_SCENARIOS:
            fixed.update(
                {
                    f"{prefix}_{cost_name}_ledgers.csv",
                    f"{prefix}_{cost_name}_learner_episodes.csv",
                    f"{prefix}_{cost_name}_union_episodes.csv",
                    f"{prefix}_{cost_name}_veto_benefits.csv",
                }
            )
    return frozenset(fixed)


def _validated_payload_inventory(
    value: Any, *, expected_stage: str
) -> dict[str, str]:
    required = _required_parent_payloads(expected_stage)
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or any(
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(expected, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None
            for filename, expected in value.items()
        )
    ):
        raise UnionContextualVetoExperimentError(
            "Prior manifest has an incomplete or unsafe payload inventory"
        )
    return dict(value)


def _require_pass_evidence_consistency(
    manifest: Mapping[str, Any],
    gate: Any,
    report: Any,
    *,
    expected_stage: str,
) -> None:
    if (
        manifest.get("stage_pass") is not True
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
        or not isinstance(report, dict)
        or report.get("contract_version") != CONTRACT_VERSION
        or report.get("stage") != expected_stage
        or report.get("run_id") != manifest.get("run_id")
        or report.get("gate_report") != gate
    ):
        raise UnionContextualVetoExperimentError(
            "Prior manifest, report, and gate result do not agree on a pass"
        )


def _validated_prior_manifest(
    *, repo_root: Path, path: Path, expected_stage: str
) -> dict[str, Any]:
    root = repo_root.resolve()
    manifest_path = path.resolve()
    try:
        relative = manifest_path.relative_to(root).as_posix()
        _git_bytes(root, "ls-files", "--error-unmatch", "--", relative)
        committed = _git_bytes(root, "show", f"HEAD:{relative}")
        local = manifest_path.read_bytes()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise UnionContextualVetoExperimentError(
            "Prior manifest must be an exact tracked file at HEAD"
        ) from exc
    if committed != local:
        raise UnionContextualVetoExperimentError(
            "Prior manifest differs from its committed blob"
        )
    value = _json_object(manifest_path, description="Prior manifest")
    payload = dict(value)
    recorded = payload.pop("manifest_sha256", None)
    if recorded != _sha256(_canonical_json_bytes(payload)):
        raise UnionContextualVetoExperimentError(
            "Prior manifest self-hash is invalid"
        )
    if (
        value.get("contract_version") != CONTRACT_VERSION
        or value.get("stage") != expected_stage
        or value.get("stage_pass") is not True
    ):
        raise UnionContextualVetoExperimentError(
            "Prior manifest did not pass the required stage"
        )
    hashes = _validated_payload_inventory(
        value.get("payload_sha256"), expected_stage=expected_stage
    )
    verified: dict[str, bytes] = {}
    for filename, expected in hashes.items():
        artifact = manifest_path.parent / filename
        try:
            artifact_relative = artifact.resolve().relative_to(root).as_posix()
            _git_bytes(root, "ls-files", "--error-unmatch", "--", artifact_relative)
            artifact_committed = _git_bytes(root, "show", f"HEAD:{artifact_relative}")
            artifact_local = artifact.read_bytes()
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            raise UnionContextualVetoExperimentError(
                f"Prior artifact is not tracked: {filename}"
            ) from exc
        if artifact_committed != artifact_local or _sha256(artifact_local) != expected:
            raise UnionContextualVetoExperimentError(
                f"Prior artifact checksum changed: {filename}"
            )
        verified[filename] = artifact_local
    gate_name = f"{expected_stage}_gate_report.json"
    try:
        gate = json.loads(verified[gate_name].decode("utf-8"))
        report = json.loads(verified["report.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnionContextualVetoExperimentError(
            "Prior stage pass evidence is unreadable"
        ) from exc
    _require_pass_evidence_consistency(
        value, gate, report, expected_stage=expected_stage
    )
    return value


def _require_dependency_continuity(
    current_git_identity: Mapping[str, Any], parent: Mapping[str, Any]
) -> None:
    current_hashes = current_git_identity.get("tracked_dependency_sha256")
    parent_hashes = parent.get("git_identity", {}).get(
        "tracked_dependency_sha256"
    )
    current_runtime = current_git_identity.get("runtime_versions")
    parent_runtime = parent.get("git_identity", {}).get("runtime_versions")
    if (
        not isinstance(current_hashes, dict)
        or current_hashes != parent_hashes
        or not isinstance(current_runtime, dict)
        or current_runtime != parent_runtime
    ):
        raise UnionContextualVetoExperimentError(
            "Frozen implementation or runtime changed after the parent stage"
        )


def _require_source_continuity(
    frame: pd.DataFrame,
    provenance: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    parent_end: pd.Timestamp,
) -> None:
    if (
        provenance.get("source_type")
        != parent.get("source_provenance", {}).get("source_type")
    ):
        raise UnionContextualVetoExperimentError(
            "Physical price-snapshot lineage changed between stages"
        )
    if _prefix_sha256(frame, end=parent_end) != parent.get(
        "bounded_result_sha256"
    ):
        raise UnionContextualVetoExperimentError(
            "Authorized historical price prefix changed"
        )


def _required_forecast_columns() -> frozenset[str]:
    matrix = {
        f"state_A_{row}_{column}"
        for row in range(len(FEATURE_COLUMNS))
        for column in range(row, len(FEATURE_COLUMNS))
    }
    vector = {f"state_b_{index}" for index in range(len(FEATURE_COLUMNS))}
    return frozenset(
        {
            "contextual_virtual_signal",
            "weak_trend_virtual_signal",
            "stage_outcome_available",
            "union_candidate_signal",
            "canonical_union_cash_signal",
            "model_veto_prediction",
            "veto",
            "learner_cash_signal",
            "target_exposure",
            "shadow_matures_on_close",
            "shadow_matured_now",
            "shadow_signal_close",
            "shadow_label_10bps",
            "shadow_lesson_added_now",
            "shadow_pending",
            "model_mu",
            "model_se",
            "model_upper",
            "model_ready",
            *FEATURE_COLUMNS,
            *STATE_SCALAR_COLUMNS,
            *matrix,
            *vector,
        }
    )


def _canonical_forecast(frame: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    data = canonical_context_frame(frame)
    value = forecast.copy()
    value.index = pd.DatetimeIndex(
        pd.to_datetime(value.index, errors="raise")
    ).tz_localize(None)
    value = value.sort_index()
    if not value.index.equals(data.index):
        raise UnionContextualVetoExperimentError(
            "Forecast rows are not exactly aligned to physical market rows"
        )
    missing = sorted(_required_forecast_columns().difference(value.columns))
    if missing:
        raise UnionContextualVetoExperimentError(
            f"Core forecast lacks required columns: {missing}"
        )
    attrs = value.attrs
    expected_attr_values = {
        "feature_order": list(FEATURE_COLUMNS),
        "lesson_maturity": "close t+2 before same-close prediction",
        "lesson_memory_start": "2000-01-01",
        "coefficient_prior_standard_deviation": 0.02,
        "observation_standard_deviation": 0.04,
        "lesson_discount": 0.995,
        "minimum_raw_lessons": 40,
        "minimum_effective_lessons": 30.0,
        "confidence_multiplier": 1.282,
        "minimum_predicted_harm": 0.001,
    }
    if any(
        attrs.get(name) != expected
        for name, expected in expected_attr_values.items()
    ):
        raise UnionContextualVetoExperimentError(
            "Core forecast attributes differ from the frozen learning contract"
        )
    if not math.isclose(
        float(attrs.get("round_trip_log_friction", math.nan)),
        math.log(0.999 / 1.001),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise UnionContextualVetoExperimentError(
            "Core forecast uses the wrong shadow-lesson friction"
        )
    mode = attrs.get("learning_mode")
    if mode not in {CAUSAL_ONLINE_MODE, FROZEN_CUTOFF_MODE}:
        raise UnionContextualVetoExperimentError(
            "Core forecast has an invalid learning mode"
        )
    if mode == CAUSAL_ONLINE_MODE and (
        attrs.get("frozen_cutoff") is not None
        or attrs.get("post_cutoff_outcomes_used_for_learning") is not True
    ):
        raise UnionContextualVetoExperimentError(
            "Causal-online forecast attributes are inconsistent"
        )
    if mode == FROZEN_CUTOFF_MODE and (
        not isinstance(attrs.get("frozen_cutoff"), str)
        or attrs.get("post_cutoff_outcomes_used_for_learning") is not False
    ):
        raise UnionContextualVetoExperimentError(
            "Frozen diagnostic forecast attributes are inconsistent"
        )
    if not isinstance(attrs.get("final_state"), dict) or not isinstance(
        attrs.get("pending_lessons"), dict
    ):
        raise UnionContextualVetoExperimentError(
            "Core forecast lacks serialized continuation state"
        )
    return value


def _bool_column(frame: pd.DataFrame, name: str) -> pd.Series:
    values = frame[name]
    if values.isna().any():
        raise UnionContextualVetoExperimentError(
            f"Forecast boolean column contains missing values: {name}"
        )
    return values.astype(bool)


def _stage_targets(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    administrative_start: pd.Timestamp,
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    data = canonical_context_frame(frame)
    value = _canonical_forecast(data, forecast)
    raw = _bool_column(value, "union_candidate_signal")
    contextual = _bool_column(value, "contextual_virtual_signal")
    weak_trend = _bool_column(value, "weak_trend_virtual_signal")
    if not raw.equals(contextual | weak_trend):
        raise UnionContextualVetoExperimentError(
            "Raw union candidate differs from the exact fixed-expert union"
        )
    continuous_union = _bool_column(value, "canonical_union_cash_signal")
    expected_continuous = canonicalize_one_session_signals(raw)
    if not continuous_union.equals(expected_continuous):
        raise UnionContextualVetoExperimentError(
            "Core canonical union stream differs from the frozen union cooldown"
        )

    model_veto = _bool_column(value, "model_veto_prediction")
    if bool((model_veto & ~raw).any()):
        raise UnionContextualVetoExperimentError(
            "Model veto prediction was emitted outside a raw union candidate"
        )
    n_raw_series = pd.to_numeric(value["n_raw"], errors="raise")
    n_eff_series = pd.to_numeric(value["n_eff"], errors="raise")
    model_ready = _bool_column(value, "model_ready")
    expected_ready = (n_raw_series >= 40) & (n_eff_series >= 30.0)
    if not model_ready.equals(expected_ready):
        raise UnionContextualVetoExperimentError(
            "Model readiness differs from the frozen lesson-count gates"
        )
    model_se = pd.to_numeric(value["model_se"], errors="coerce")
    model_upper = pd.to_numeric(value["model_upper"], errors="coerce")
    prediction_values = value.loc[
        raw, ["model_mu", "model_se", "model_upper"]
    ].apply(pd.to_numeric, errors="raise")
    if (
        not np.isfinite(prediction_values.to_numpy(dtype=float)).all()
        or bool((model_se.loc[raw] <= 0.0).any())
    ):
        raise UnionContextualVetoExperimentError(
            "A raw candidate has an invalid Bayesian prediction"
        )
    expected_model_veto = raw & model_ready & (model_upper < -0.001)
    if not model_veto.equals(expected_model_veto):
        raise UnionContextualVetoExperimentError(
            "Model veto prediction differs from the frozen strict rule"
        )
    continuous_veto = _bool_column(value, "veto")
    if not continuous_veto.equals(continuous_union & model_veto):
        raise UnionContextualVetoExperimentError(
            "Continuous veto stream is not union opportunity AND prediction"
        )
    core_learner_cash = _bool_column(value, "learner_cash_signal")
    if not core_learner_cash.equals(continuous_union & ~model_veto):
        raise UnionContextualVetoExperimentError(
            "Core continuous learner violates the pure-veto identity"
        )
    core_target = pd.to_numeric(value["target_exposure"], errors="raise")
    expected_core_target = pd.Series(
        np.where(core_learner_cash, 0.0, 1.0), index=data.index, dtype=float
    )
    if not core_target.astype(float).equals(expected_core_target):
        raise UnionContextualVetoExperimentError(
            "Core target exposure differs from its continuous learner signal"
        )
    candidate_features = value.loc[raw, list(FEATURE_COLUMNS)].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(candidate_features.to_numpy(dtype=float)).all():
        raise UnionContextualVetoExperimentError(
            "A raw union candidate has a missing or nonfinite feature"
        )

    stage_outcome_available = _bool_column(value, "stage_outcome_available")
    expected_available = pd.Series(False, index=data.index, dtype=bool)
    if len(expected_available) > 2:
        expected_available.iloc[:-2] = True
    if not stage_outcome_available.equals(expected_available):
        raise UnionContextualVetoExperimentError(
            "Stage outcome availability does not exactly mask the unresolved tail"
        )
    allowed = raw & stage_outcome_available
    allowed.loc[allowed.index < administrative_start] = False
    stage_union_cash = canonicalize_one_session_signals(allowed)
    # The union consumes its cooldown before the veto. Never canonicalize this
    # learner stream a second time, because doing so could resurrect t+1.
    stage_learner_cash = stage_union_cash & ~model_veto
    vetoed = stage_union_cash & model_veto

    union_target = pd.Series(
        np.where(stage_union_cash, 0.0, 1.0),
        index=data.index,
        name="union_target_exposure",
        dtype=float,
    )
    learner_target = pd.Series(
        np.where(stage_learner_cash, 0.0, 1.0),
        index=data.index,
        name="learner_target_exposure",
        dtype=float,
    )
    always_long = pd.Series(
        1.0, index=data.index, name="always_long_target_exposure", dtype=float
    )

    maturity_dates = pd.to_datetime(
        value["shadow_matures_on_close"], errors="coerce"
    )
    matures = maturity_dates.notna()
    pending = _bool_column(value, "shadow_pending")
    matured_now = _bool_column(value, "shadow_matured_now")
    lesson_added_now = _bool_column(value, "shadow_lesson_added_now")
    expected_maturity_dates = pd.Series(
        pd.NaT, index=data.index, dtype="datetime64[ns]"
    )
    resolved_positions = np.flatnonzero(
        continuous_union.to_numpy(dtype=bool)
        & stage_outcome_available.to_numpy(dtype=bool)
    )
    for position in resolved_positions:
        expected_maturity_dates.iloc[position] = data.index[position + 2]
    if not maturity_dates.equals(expected_maturity_dates):
        raise UnionContextualVetoExperimentError(
            "Shadow maturity schedule differs from signal close plus two sessions"
        )
    expected_pending = continuous_union & ~stage_outcome_available
    if not pending.equals(expected_pending):
        raise UnionContextualVetoExperimentError(
            "Pending shadow lessons do not equal unresolved union opportunities"
        )
    expected_matured_now = continuous_union.shift(2, fill_value=False).astype(bool)
    if not matured_now.equals(expected_matured_now):
        raise UnionContextualVetoExperimentError(
            "Shadow lessons do not mature exactly two sessions after their signal"
        )
    signal_close = pd.to_datetime(value["shadow_signal_close"], errors="coerce")
    expected_signal_close = pd.Series(
        pd.NaT, index=data.index, dtype="datetime64[ns]"
    )
    matured_positions = np.flatnonzero(expected_matured_now.to_numpy(dtype=bool))
    for position in matured_positions:
        expected_signal_close.iloc[position] = data.index[position - 2]
    if not signal_close.equals(expected_signal_close):
        raise UnionContextualVetoExperimentError(
            "Matured shadow lessons identify the wrong signal close"
        )
    labels = pd.to_numeric(value["shadow_label_10bps"], errors="coerce")
    if not labels.notna().equals(expected_matured_now) or not np.isfinite(
        labels.loc[expected_matured_now].to_numpy(dtype=float)
    ).all():
        raise UnionContextualVetoExperimentError(
            "Shadow labels are not finite exactly on maturity closes"
        )
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    for position in matured_positions:
        expected_label = math.log(opens[position - 1] / opens[position]) + math.log(
            0.999 / 1.001
        )
        if not math.isclose(
            float(labels.iloc[position]),
            expected_label,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise UnionContextualVetoExperimentError(
                "Shadow lesson label differs from the frozen 10-bps formula"
            )
    expected_lesson_added = expected_matured_now & (
        signal_close >= pd.Timestamp("2000-01-01")
    )
    if value.attrs["learning_mode"] == FROZEN_CUTOFF_MODE:
        expected_lesson_added &= data.index <= pd.Timestamp(
            value.attrs["frozen_cutoff"]
        )
    if not lesson_added_now.equals(expected_lesson_added):
        raise UnionContextualVetoExperimentError(
            "Lesson admission violates memory start, maturity, or frozen cutoff"
        )
    raw_counts = pd.to_numeric(value["n_raw"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.array_equal(
        raw_counts,
        expected_lesson_added.astype(int).cumsum().to_numpy(dtype=float),
    ):
        raise UnionContextualVetoExperimentError(
            "Raw lesson count does not match causal lesson admissions"
        )
    shadow_count = int(continuous_union.sum())
    shadow_records = int(matures.sum() + pending.sum())
    shadow_maturity_count = int(matured_now.sum())
    if bool((matures & pending).any()) or shadow_records != shadow_count:
        raise UnionContextualVetoExperimentError(
            "Canonical union opportunities do not map one-to-one to shadow lessons"
        )
    if shadow_maturity_count != int(matures.sum()):
        raise UnionContextualVetoExperimentError(
            "Shadow maturity rows do not match scheduled shadow lessons"
        )
    if not stage_learner_cash[stage_learner_cash].index.isin(
        stage_union_cash[stage_union_cash].index
    ).all():
        raise UnionContextualVetoExperimentError(
            "Learner cash stream is not a subset of union cash"
        )
    if not stage_learner_cash.equals(stage_union_cash & ~model_veto):
        raise UnionContextualVetoExperimentError(
            "Learner stream violates the pure-veto identity"
        )

    action_frame = pd.DataFrame(
        {
            "decision_date": data.index,
            "learner_target": learner_target.to_numpy(dtype=float),
            "union_target": union_target.to_numpy(dtype=float),
        }
    )
    integrity = {
        "passed": True,
        "learner_cash_subset_of_union": True,
        "pure_veto_identity": True,
        "frozen_veto_rule_exact": True,
        "union_cooldown_applied_before_veto": True,
        "continuous_shadow_opportunity_count": shadow_count,
        "continuous_shadow_record_count": shadow_records,
        "continuous_shadow_maturity_count": shadow_maturity_count,
        "shadow_count_equality": True,
        "shadow_maturity_and_label_formula_exact": True,
        "causal_lesson_admission_exact": True,
        "stage_union_cash_count": int(stage_union_cash.sum()),
        "stage_learner_cash_count": int(stage_learner_cash.sum()),
        "stage_veto_count": int(vetoed.sum()),
        "action_stream_sha256": _sha256(_frame_csv_bytes(action_frame)),
        "same_action_stream_all_costs": True,
    }
    return {
        "learner": learner_target,
        "union": union_target,
        "always_long": always_long,
    }, integrity


def _veto_benefit_rows(
    frame: pd.DataFrame,
    union_target: pd.Series,
    learner_target: pd.Series,
    forecast: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
) -> list[dict[str, Any]]:
    data = canonical_context_frame(frame)
    value = _canonical_forecast(data, forecast)
    union = pd.to_numeric(union_target.reindex(data.index), errors="raise")
    learner = pd.to_numeric(learner_target.reindex(data.index), errors="raise")
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    dates = data.index
    cost = float(cost_bps) / 10_000.0
    friction = math.log((1.0 - cost) / (1.0 + cost))
    veto_positions = np.flatnonzero(
        (union.to_numpy(dtype=float) == 0.0)
        & (learner.to_numpy(dtype=float) == 1.0)
    )
    rows: list[dict[str, Any]] = []
    for position in veto_positions:
        if position + 2 >= len(data):
            continue
        entry = dates[position + 1]
        exit_date = dates[position + 2]
        if entry < start or exit_date > end:
            continue
        raw_cash_edge = math.log(opens[position + 1] / opens[position + 2])
        net_cash_edge = raw_cash_edge + friction
        benefit = -net_cash_edge
        row: dict[str, Any] = {
            "decision_date": dates[position].date().isoformat(),
            "entry_date": entry.date().isoformat(),
            "exit_date": exit_date.date().isoformat(),
            "raw_union_cash_edge": raw_cash_edge,
            "net_union_cash_edge": net_cash_edge,
            "veto_benefit": benefit,
            "beneficial_veto": bool(benefit > 0.0),
        }
        for name in (
            *FEATURE_COLUMNS,
            "n_raw",
            "n_eff",
            "model_mu",
            "model_se",
            "model_upper",
            "model_ready",
        ):
            raw_value = value.iloc[position][name]
            row[name] = (
                bool(raw_value)
                if name == "model_ready"
                else float(raw_value)
            )
        rows.append(row)
    return rows


def _period_benefit(
    rows: Sequence[Mapping[str, Any]], period: EvaluationPeriod
) -> float:
    start = pd.Timestamp(period.start)
    end = pd.Timestamp(period.end)
    return float(
        np.sum(
            [
                float(row["veto_benefit"])
                for row in rows
                if start <= pd.Timestamp(row["entry_date"]) <= end
            ]
        )
    )


def _benefit_summary(
    rows: Sequence[Mapping[str, Any]], periods: Sequence[EvaluationPeriod]
) -> dict[str, Any]:
    values = np.asarray([float(row["veto_benefit"]) for row in rows], dtype=float)
    positive = values[values > 0.0]
    return {
        "veto_count": int(len(values)),
        "beneficial_veto_rate": float(np.mean(values > 0.0))
        if len(values)
        else None,
        "mean_veto_benefit": float(np.mean(values)) if len(values) else None,
        "median_veto_benefit": float(np.median(values)) if len(values) else None,
        "maximum_positive_veto_share": float(np.max(positive) / np.sum(positive))
        if len(positive)
        else None,
        "total_veto_benefit": float(np.sum(values)),
        "periods": {
            period.name: _period_benefit(rows, period) for period in periods
        },
    }


def _evaluate_policy_set(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    periods: Sequence[EvaluationPeriod],
    administrative_start: pd.Timestamp,
) -> tuple[
    dict[str, Any],
    dict[str, pd.DataFrame],
    dict[str, dict[str, pd.DataFrame]],
    dict[str, pd.DataFrame],
    dict[str, Any],
]:
    targets, integrity = _stage_targets(
        frame, forecast, administrative_start=administrative_start
    )
    metrics: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    episode_frames: dict[str, dict[str, pd.DataFrame]] = {}
    benefit_frames: dict[str, pd.DataFrame] = {}
    for cost_name, cost_bps in COST_SCENARIOS:
        metrics[cost_name] = {}
        policy_ledgers: dict[str, pd.DataFrame] = {}
        policy_episodes: dict[str, pd.DataFrame] = {}
        ledger_rows: list[pd.DataFrame] = []
        benchmark: pd.DataFrame | None = None
        for policy_name, target in targets.items():
            result, strategy, policy_benchmark, episodes = (
                _sealed_base._evaluate_policy(
                    frame,
                    target,
                    periods=periods,
                    cost_bps=cost_bps,
                )
            )
            metrics[cost_name][policy_name] = result
            policy_ledgers[policy_name] = strategy
            policy_episodes[policy_name] = episodes
            copy = strategy.copy()
            copy.insert(0, "policy", policy_name)
            copy.insert(1, "ledger_role", "strategy")
            ledger_rows.append(copy)
            if benchmark is None:
                benchmark = policy_benchmark
        if benchmark is None:
            raise UnionContextualVetoExperimentError(
                "Policy evaluation did not produce a benchmark ledger"
            )
        benchmark_copy = benchmark.copy()
        benchmark_copy.insert(0, "policy", "aapl_buy_hold")
        benchmark_copy.insert(1, "ledger_role", "benchmark")
        ledger_rows.append(benchmark_copy)

        benefit_rows = _veto_benefit_rows(
            frame,
            targets["union"],
            targets["learner"],
            forecast,
            start=pd.Timestamp(periods[0].start),
            end=pd.Timestamp(periods[-1].end),
            cost_bps=cost_bps,
        )
        summary = _benefit_summary(benefit_rows, periods)
        learner = metrics[cost_name]["learner"]
        union = metrics[cost_name]["union"]
        total_incremental = float(
            learner["total_active_log_edge"] - union["total_active_log_edge"]
        )
        identity_error = total_incremental - float(summary["total_veto_benefit"])
        if abs(identity_error) > 1e-10:
            raise UnionContextualVetoExperimentError(
                "Learner-minus-union edge does not reconcile to veto benefits"
            )
        incremental_periods: dict[str, float] = {}
        for period in periods:
            ledger_difference = float(
                learner["periods"][period.name]["active_log_edge"]
                - union["periods"][period.name]["active_log_edge"]
            )
            attributed = float(summary["periods"][period.name])
            if abs(ledger_difference - attributed) > 1e-10:
                raise UnionContextualVetoExperimentError(
                    f"Incremental period edge does not reconcile: {period.name}"
                )
            incremental_periods[period.name] = attributed
        learner_ledger = policy_ledgers["learner"]
        union_ledger = policy_ledgers["union"]
        relative_wealth = float(
            learner_ledger["equity"].iloc[-1]
            / union_ledger["equity"].iloc[-1]
            - 1.0
        )
        metrics[cost_name]["learner_vs_union"] = {
            "total_active_log_edge": total_incremental,
            "relative_wealth": relative_wealth,
            "periods": incremental_periods,
            "veto_benefit": summary,
            "veto_benefit_identity_error": identity_error,
        }
        ledgers[cost_name] = pd.concat(ledger_rows, ignore_index=True)
        episode_frames[cost_name] = {
            "learner": policy_episodes["learner"],
            "union": policy_episodes["union"],
        }
        benefit_frames[cost_name] = pd.DataFrame(benefit_rows)

    integrity["all_policy_ledgers_unleveraged"] = all(
        metrics[cost_name][policy]["no_leverage_proof"]["passed"]
        for cost_name, _ in COST_SCENARIOS
        for policy in ("learner", "union", "always_long")
    )
    integrity["episode_and_veto_edge_identity"] = True
    integrity["passed"] = bool(
        integrity["learner_cash_subset_of_union"]
        and integrity["pure_veto_identity"]
        and integrity["frozen_veto_rule_exact"]
        and integrity["shadow_count_equality"]
        and integrity["shadow_maturity_and_label_formula_exact"]
        and integrity["causal_lesson_admission_exact"]
        and integrity["same_action_stream_all_costs"]
        and integrity["all_policy_ledgers_unleveraged"]
    )
    return metrics, ledgers, episode_frames, benefit_frames, integrity


def _periods_for_years(first: int, last: int) -> tuple[EvaluationPeriod, ...]:
    return tuple(
        EvaluationPeriod(str(year), f"{year}-01-01", f"{year}-12-31")
        for year in range(first, last + 1)
    )


def _development_periods() -> tuple[EvaluationPeriod, ...]:
    annual = _periods_for_years(2005, 2018)
    folds = tuple(
        EvaluationPeriod(
            f"{start}_{start + 1}",
            f"{start}-01-01",
            f"{start + 1}-12-31",
        )
        for start in range(2005, 2019, 2)
    )
    return (*annual, *folds)


def _validation_periods() -> tuple[EvaluationPeriod, ...]:
    return _periods_for_years(2019, 2023)


def _final_periods() -> tuple[EvaluationPeriod, ...]:
    return (
        EvaluationPeriod("2024", "2024-01-01", "2024-12-31"),
        EvaluationPeriod("2025", "2025-01-01", "2025-12-31"),
        EvaluationPeriod("2026_ytd", "2026-01-01", "2026-07-09"),
    )


def _lifetime_periods() -> tuple[EvaluationPeriod, ...]:
    return (
        *_periods_for_years(2005, 2025),
        EvaluationPeriod("2026_ytd", "2026-01-01", "2026-07-09"),
    )


def _annual_values(result: Mapping[str, Any], first: int, last: int) -> list[float]:
    return [
        float(result["periods"][str(year)]["active_log_edge"])
        for year in range(first, last + 1)
    ]


def _always_long_gate(metrics: Mapping[str, Any]) -> bool:
    for cost_name, _ in COST_SCENARIOS:
        result = metrics[cost_name]["always_long"]
        if (
            abs(float(result["total_active_log_edge"])) > 1e-12
            or abs(
                float(
                    result["comparison"]["relative_wealth_vs_aapl_buy_hold"]
                )
            )
            > 1e-12
        ):
            return False
    return True


def _integrity_gates(integrity: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "integrity_passed": integrity.get("passed") is True,
        "learner_cash_subset_of_union": (
            integrity.get("learner_cash_subset_of_union") is True
        ),
        "pure_veto_identity": integrity.get("pure_veto_identity") is True,
        "frozen_veto_rule_exact": (
            integrity.get("frozen_veto_rule_exact") is True
        ),
        "union_cooldown_applied_before_veto": (
            integrity.get("union_cooldown_applied_before_veto") is True
        ),
        "one_shadow_lesson_per_continuous_union_opportunity": (
            integrity.get("shadow_count_equality") is True
        ),
        "shadow_maturity_and_label_formula_exact": (
            integrity.get("shadow_maturity_and_label_formula_exact") is True
        ),
        "causal_lesson_admission_exact": (
            integrity.get("causal_lesson_admission_exact") is True
        ),
        "same_action_stream_at_both_costs": (
            integrity.get("same_action_stream_all_costs") is True
        ),
        "all_policy_ledgers_unleveraged": (
            integrity.get("all_policy_ledgers_unleveraged") is True
        ),
        "episode_and_veto_edge_identity": (
            integrity.get("episode_and_veto_edge_identity") is True
        ),
    }


def _gate_report(gates: Mapping[str, bool]) -> dict[str, Any]:
    failures = sorted(name for name, passed in gates.items() if not passed)
    return {"passed": not failures, "gates": dict(gates), "failures": failures}


def apply_development_gates(
    metrics: Mapping[str, Any], integrity: Mapping[str, Any]
) -> dict[str, Any]:
    gates: dict[str, bool] = {
        "always_long_matches_buy_hold": _always_long_gate(metrics),
        **_integrity_gates(integrity),
    }
    for cost_name, _ in COST_SCENARIOS:
        result = metrics[cost_name]["learner"]
        union = metrics[cost_name]["union"]
        incremental = metrics[cost_name]["learner_vs_union"]
        annual = _annual_values(result, 2005, 2018)
        folds = [
            float(result["periods"][f"{start}_{start + 1}"]["active_log_edge"])
            for start in range(2005, 2019, 2)
        ]
        negative_years = [
            float(result["periods"][str(year)]["active_log_edge"])
            for year in (2008, 2015, 2018)
        ]
        concentration = _positive_concentration(annual)
        reference = UNION_REFERENCE[cost_name]
        prefix = f"{cost_name}_"
        gates.update(
            {
                prefix + "union_reference_episode_count": int(
                    union["cash_episode_count"]
                )
                == int(reference["episodes"]),
                prefix + "union_reference_active_log_edge": abs(
                    float(union["total_active_log_edge"])
                    - float(reference["total_active_log_edge"])
                )
                <= 1e-12,
                prefix + "positive_total_active_log_edge": float(
                    result["total_active_log_edge"]
                )
                > 0.0,
                prefix + "positive_relative_wealth": float(
                    result["comparison"]["relative_wealth_vs_aapl_buy_hold"]
                )
                > 0.0,
                prefix + "minimum_eight_cash_episodes": int(
                    result["cash_episode_count"]
                )
                >= 8,
                prefix + "minimum_eight_positive_years": int(
                    np.count_nonzero(np.asarray(annual) > 0.0)
                )
                >= 8,
                prefix + "minimum_four_positive_folds": int(
                    np.count_nonzero(np.asarray(folds) > 0.0)
                )
                >= 4,
                prefix + "positive_after_removing_best_year": float(
                    np.sum(annual) - np.max(annual)
                )
                > 0.0,
                prefix + "annual_positive_edge_not_concentrated": (
                    concentration is not None and concentration <= 0.50
                ),
                prefix + "positive_negative_year_aggregate": float(
                    np.sum(negative_years)
                )
                > 0.0,
                prefix + "two_of_three_negative_years_positive": int(
                    np.count_nonzero(np.asarray(negative_years) > 0.0)
                )
                >= 2,
                prefix + "learner_beats_union_by_more_than_0001": float(
                    incremental["total_active_log_edge"]
                )
                > STRICT_UNION_IMPROVEMENT,
            }
        )

    stress = metrics["stress_10bps"]["learner"]
    stress_incremental = metrics["stress_10bps"]["learner_vs_union"]
    benefit = stress_incremental["veto_benefit"]
    incremental_folds = [
        float(stress_incremental["periods"][f"{start}_{start + 1}"])
        for start in range(2005, 2019, 2)
    ]
    fold_concentration = _positive_concentration(incremental_folds)
    gates.update(
        {
            "stress_10bps_episode_win_rate_at_least_55pct": (
                stress["cash_episode_win_rate"] is not None
                and float(stress["cash_episode_win_rate"]) >= 0.55
            ),
            "stress_10bps_positive_mean_episode_edge": (
                stress["mean_cash_episode_edge"] is not None
                and float(stress["mean_cash_episode_edge"]) > 0.0
            ),
            "stress_10bps_positive_median_episode_edge": (
                stress["median_cash_episode_edge"] is not None
                and float(stress["median_cash_episode_edge"]) > 0.0
            ),
            "minimum_eight_vetoes": int(benefit["veto_count"]) >= 8,
            "stress_10bps_beneficial_veto_rate_at_least_55pct": (
                benefit["beneficial_veto_rate"] is not None
                and float(benefit["beneficial_veto_rate"]) >= 0.55
            ),
            "stress_10bps_positive_mean_veto_benefit": (
                benefit["mean_veto_benefit"] is not None
                and float(benefit["mean_veto_benefit"]) > 0.0
            ),
            "stress_10bps_positive_median_veto_benefit": (
                benefit["median_veto_benefit"] is not None
                and float(benefit["median_veto_benefit"]) > 0.0
            ),
            "stress_10bps_veto_benefit_not_concentrated": (
                benefit["maximum_positive_veto_share"] is not None
                and float(benefit["maximum_positive_veto_share"]) <= 0.50
            ),
            "stress_10bps_minimum_four_positive_incremental_folds": int(
                np.count_nonzero(np.asarray(incremental_folds) > 0.0)
            )
            >= 4,
            "stress_10bps_positive_incremental_after_best_fold_removed": float(
                np.sum(incremental_folds) - np.max(incremental_folds)
            )
            > 0.0,
            "stress_10bps_incremental_fold_edge_not_concentrated": (
                fold_concentration is not None and fold_concentration <= 0.50
            ),
        }
    )
    return _gate_report(gates)


def apply_validation_gates(
    online_metrics: Mapping[str, Any], integrity: Mapping[str, Any]
) -> dict[str, Any]:
    gates: dict[str, bool] = {
        "online_always_long_matches_buy_hold": _always_long_gate(online_metrics),
        **_integrity_gates(integrity),
    }
    for cost_name, _ in COST_SCENARIOS:
        result = online_metrics[cost_name]["learner"]
        incremental = online_metrics[cost_name]["learner_vs_union"]
        annual = _annual_values(result, 2019, 2023)
        concentration = result["maximum_positive_episode_share"]
        negative_aapl_years = [
            str(year)
            for year in range(2019, 2024)
            if float(
                result["periods"][str(year)]["aapl_buy_hold_return"]
            )
            < 0.0
        ]
        negative_aapl_edge = float(
            np.sum(
                [
                    result["periods"][year]["active_log_edge"]
                    for year in negative_aapl_years
                ]
            )
        )
        incremental_years = [
            float(incremental["periods"][str(year)])
            for year in range(2019, 2024)
        ]
        prefix = f"{cost_name}_"
        gates.update(
            {
                prefix + "positive_total_active_log_edge": float(
                    result["total_active_log_edge"]
                )
                > 0.0,
                prefix + "minimum_three_positive_years": int(
                    np.count_nonzero(np.asarray(annual) > 0.0)
                )
                >= 3,
                prefix + "positive_after_removing_best_validation_year": float(
                    np.sum(annual) - np.max(annual)
                )
                > 0.0,
                prefix + "minimum_three_cash_episodes": int(
                    result["cash_episode_count"]
                )
                >= 3,
                prefix + "positive_mean_episode_edge": (
                    result["mean_cash_episode_edge"] is not None
                    and float(result["mean_cash_episode_edge"]) > 0.0
                ),
                prefix + "positive_median_episode_edge": (
                    result["median_cash_episode_edge"] is not None
                    and float(result["median_cash_episode_edge"]) > 0.0
                ),
                prefix + "episode_positive_edge_not_concentrated": (
                    concentration is not None and float(concentration) <= 0.50
                ),
                prefix + "positive_negative_aapl_year_aggregate": (
                    not negative_aapl_years or negative_aapl_edge > 0.0
                ),
                prefix + "learner_beats_union_by_more_than_0001": float(
                    incremental["total_active_log_edge"]
                )
                > STRICT_UNION_IMPROVEMENT,
                prefix + "minimum_two_positive_incremental_years": int(
                    np.count_nonzero(np.asarray(incremental_years) > 0.0)
                )
                >= 2,
                prefix + "minimum_three_vetoes": int(
                    incremental["veto_benefit"]["veto_count"]
                )
                >= 3,
            }
        )
    benefit = online_metrics["stress_10bps"]["learner_vs_union"][
        "veto_benefit"
    ]
    gates.update(
        {
            "stress_10bps_beneficial_veto_rate_at_least_50pct": (
                benefit["beneficial_veto_rate"] is not None
                and float(benefit["beneficial_veto_rate"]) >= 0.50
            ),
            "stress_10bps_positive_mean_veto_benefit": (
                benefit["mean_veto_benefit"] is not None
                and float(benefit["mean_veto_benefit"]) > 0.0
            ),
            "stress_10bps_positive_median_veto_benefit": (
                benefit["median_veto_benefit"] is not None
                and float(benefit["median_veto_benefit"]) > 0.0
            ),
            "stress_10bps_veto_benefit_not_concentrated": (
                benefit["maximum_positive_veto_share"] is not None
                and float(benefit["maximum_positive_veto_share"]) <= 0.50
            ),
        }
    )
    report = _gate_report(gates)
    report["primary_evidence"] = "causal_online"
    report["frozen_diagnostic_can_rescue"] = False
    return report


def apply_final_gates(
    online_metrics: Mapping[str, Any], integrity: Mapping[str, Any]
) -> dict[str, Any]:
    gates: dict[str, bool] = {
        "online_always_long_matches_buy_hold": _always_long_gate(online_metrics),
        **_integrity_gates(integrity),
    }
    for cost_name, _ in COST_SCENARIOS:
        learner = online_metrics[cost_name]["learner"]
        incremental = online_metrics[cost_name]["learner_vs_union"]
        for period_name in ("2024", "2025", "2026_ytd"):
            gates[
                f"{cost_name}_{period_name}_active_log_edge_above_001"
            ] = (
                float(learner["periods"][period_name]["active_log_edge"])
                > FINAL_MATERIAL_ACTIVE_LOG_EDGE
            )
        incremental_periods = [
            float(incremental["periods"][name])
            for name in ("2024", "2025", "2026_ytd")
        ]
        gates[
            f"{cost_name}_continuous_learner_beats_union_by_more_than_0001"
        ] = float(incremental["total_active_log_edge"]) > STRICT_UNION_IMPROVEMENT
        gates[
            f"{cost_name}_minimum_two_nonnegative_incremental_periods"
        ] = int(np.count_nonzero(np.asarray(incremental_periods) >= 0.0)) >= 2
    report = _gate_report(gates)
    report["primary_evidence"] = "causal_online"
    report["frozen_or_lifetime_diagnostic_can_rescue"] = False
    return report


def _lifetime_diagnostic(metrics: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {}
    period_names = [str(year) for year in range(2005, 2026)] + ["2026_ytd"]
    for cost_name, _ in COST_SCENARIOS:
        learner = metrics[cost_name]["learner"]
        annual_edges = [
            float(learner["periods"][name]["active_log_edge"])
            for name in period_names
        ]
        negative_aapl_periods = [
            name
            for name in period_names
            if float(learner["periods"][name]["aapl_buy_hold_return"]) < 0.0
        ]
        comparison = learner["comparison"]
        diagnostic[cost_name] = {
            "continuous_relative_wealth_vs_aapl_buy_hold": float(
                comparison["relative_wealth_vs_aapl_buy_hold"]
            ),
            "positive_periods": int(
                np.count_nonzero(np.asarray(annual_edges) > 0.0)
            ),
            "negative_periods": int(
                np.count_nonzero(np.asarray(annual_edges) < 0.0)
            ),
            "zero_periods": int(
                np.count_nonzero(np.asarray(annual_edges) == 0.0)
            ),
            "negative_aapl_periods": negative_aapl_periods,
            "negative_aapl_period_aggregate_active_log_edge": float(
                np.sum(
                    [
                        learner["periods"][name]["active_log_edge"]
                        for name in negative_aapl_periods
                    ]
                )
            ),
            "cash_episode_count": int(learner["cash_episode_count"]),
            "cash_episode_win_rate": learner["cash_episode_win_rate"],
            "mean_cash_episode_edge": learner["mean_cash_episode_edge"],
            "median_cash_episode_edge": learner["median_cash_episode_edge"],
            "maximum_positive_episode_share": learner[
                "maximum_positive_episode_share"
            ],
            "strategy_max_drawdown": float(
                comparison["strategy"]["max_drawdown"]
            ),
            "aapl_buy_hold_max_drawdown": float(
                comparison["aapl_buy_hold"]["max_drawdown"]
            ),
            "max_drawdown_difference": float(
                comparison["max_drawdown_difference"]
            ),
        }
    return diagnostic


MODEL_CONSTANTS = {
    "coefficient_prior_standard_deviation": 0.02,
    "fixed_observation_standard_deviation": 0.04,
    "per_lesson_discount_rho": 0.995,
    "minimum_raw_lessons": 40,
    "minimum_effective_lessons": 30.0,
    "one_sided_confidence_multiplier": 1.282,
    "minimum_predicted_harm": 0.001,
    "lesson_cost_bps_per_changing_leg": 10.0,
    "lesson_start": "2000-01-01",
}


def _checkpoint_from_forecast(
    forecast: pd.DataFrame, *, cutoff: pd.Timestamp, learning_mode: str
) -> dict[str, Any]:
    if pd.DatetimeIndex(forecast.index).max() > cutoff:
        raise UnionContextualVetoExperimentError(
            "Checkpoint forecast must be physically bounded at its cutoff"
        )
    eligible = forecast.loc[:cutoff]
    if eligible.empty:
        raise UnionContextualVetoExperimentError("Checkpoint has no rows")
    missing = sorted(_required_forecast_columns().difference(eligible.columns))
    if missing:
        raise UnionContextualVetoExperimentError(
            f"Checkpoint forecast lacks required columns: {missing}"
        )
    row = eligible.iloc[-1]
    scalars: dict[str, int | float] = {}
    for name in STATE_SCALAR_COLUMNS:
        raw = float(row[name])
        if not math.isfinite(raw):
            raise UnionContextualVetoExperimentError(
                f"Checkpoint state is nonfinite: {name}"
            )
        scalars[name] = int(round(raw)) if name in {"n_raw", "wins"} else raw

    upper_triangle: dict[str, float] = {}
    for matrix_row in range(len(FEATURE_COLUMNS)):
        for matrix_column in range(matrix_row, len(FEATURE_COLUMNS)):
            name = f"state_A_{matrix_row}_{matrix_column}"
            value = float(row[name])
            if not math.isfinite(value):
                raise UnionContextualVetoExperimentError(
                    f"Checkpoint precision matrix is nonfinite: {name}"
                )
            upper_triangle[name] = value
    vector: dict[str, float] = {}
    for index in range(len(FEATURE_COLUMNS)):
        name = f"state_b_{index}"
        value = float(row[name])
        if not math.isfinite(value):
            raise UnionContextualVetoExperimentError(
                f"Checkpoint state vector is nonfinite: {name}"
            )
        vector[name] = value

    trailing_rows: list[dict[str, Any]] = []
    for decision_date, trailing in eligible.tail(2).iterrows():
        raw_signal = bool(trailing["union_candidate_signal"])
        canonical_signal = bool(trailing["canonical_union_cash_signal"])
        record: dict[str, Any] = {
            "decision_date": decision_date.date().isoformat(),
            "union_candidate_signal": raw_signal,
            "canonical_union_cash_signal": canonical_signal,
        }
        if raw_signal:
            features = [float(trailing[name]) for name in FEATURE_COLUMNS]
            if not np.isfinite(np.asarray(features, dtype=float)).all():
                raise UnionContextualVetoExperimentError(
                    "Checkpoint trailing candidate features are nonfinite"
                )
            record["features"] = features
        else:
            record["features"] = None
        trailing_rows.append(record)
    pending = [
        {
            "signal_close": record["decision_date"],
            "features": record["features"],
        }
        for record in trailing_rows
        if record["canonical_union_cash_signal"]
    ]
    attrs = forecast.attrs
    if attrs.get("learning_mode") != learning_mode:
        raise UnionContextualVetoExperimentError(
            "Checkpoint learning mode differs from the forecast"
        )
    serialized_state = attrs.get("final_state")
    serialized_pending = attrs.get("pending_lessons")
    if not isinstance(serialized_state, dict) or not isinstance(
        serialized_pending, dict
    ):
        raise UnionContextualVetoExperimentError(
            "Checkpoint forecast lacks serialized continuation state"
        )
    if serialized_state.get("feature_order") != list(FEATURE_COLUMNS) or (
        serialized_pending.get("feature_order") != list(FEATURE_COLUMNS)
    ):
        raise UnionContextualVetoExperimentError(
            "Serialized checkpoint feature order changed"
        )
    serialized_pending_rows = serialized_pending.get("lessons")
    if not isinstance(serialized_pending_rows, list) or len(
        serialized_pending_rows
    ) != len(pending):
        raise UnionContextualVetoExperimentError(
            "Serialized pending lessons differ from unresolved shadow opportunities"
        )
    serialized_signal_dates = [
        pd.Timestamp(item["signal_close"]).date().isoformat()
        for item in serialized_pending_rows
        if isinstance(item, dict) and "signal_close" in item
    ]
    if serialized_signal_dates != [item["signal_close"] for item in pending]:
        raise UnionContextualVetoExperimentError(
            "Serialized pending lesson dates differ from the shadow stream"
        )
    try:
        serialized_A = np.asarray(serialized_state["A"], dtype=float)
        serialized_b = np.asarray(serialized_state["b"], dtype=float)
        reconstructed_A = np.empty(
            (len(FEATURE_COLUMNS), len(FEATURE_COLUMNS)), dtype=float
        )
        for matrix_row in range(len(FEATURE_COLUMNS)):
            for matrix_column in range(len(FEATURE_COLUMNS)):
                name = (
                    f"state_A_{matrix_row}_{matrix_column}"
                    if matrix_row <= matrix_column
                    else f"state_A_{matrix_column}_{matrix_row}"
                )
                reconstructed_A[matrix_row, matrix_column] = upper_triangle[name]
        reconstructed_b = np.asarray(
            [vector[f"state_b_{index}"] for index in range(len(FEATURE_COLUMNS))],
            dtype=float,
        )
        scalar_match = all(
            float(serialized_state[name]) == float(scalars[name])
            for name in STATE_SCALAR_COLUMNS
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise UnionContextualVetoExperimentError(
            "Serialized Bayesian checkpoint state is malformed"
        ) from exc
    if (
        serialized_A.shape != reconstructed_A.shape
        or serialized_b.shape != reconstructed_b.shape
        or not np.array_equal(serialized_A, reconstructed_A)
        or not np.array_equal(serialized_b, reconstructed_b)
        or not scalar_match
    ):
        raise UnionContextualVetoExperimentError(
            "Serialized Bayesian checkpoint disagrees with forecast state columns"
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "learning_mode": learning_mode,
        "checkpoint_cutoff": cutoff.date().isoformat(),
        "last_observed_session": eligible.index[-1].date().isoformat(),
        "feature_order": list(FEATURE_COLUMNS),
        "model_constants": dict(MODEL_CONSTANTS),
        "serialized_bayesian_state": serialized_state,
        "serialized_pending_shadow_lessons": serialized_pending,
        "sufficient_state": {
            **scalars,
            "precision_matrix_upper_triangle": upper_triangle,
            "state_vector": vector,
        },
        "trailing_cooldown_context": trailing_rows,
        "pending_shadow_opportunities": pending,
    }


def _require_checkpoint_continuity(
    forecast: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    parent_manifest_path: Path,
    checkpoint_filename: str,
) -> None:
    checkpoint_path = parent_manifest_path.resolve().parent / checkpoint_filename
    parent = _json_object(checkpoint_path, description="Committed parent checkpoint")
    if parent.get("learning_mode") != CAUSAL_ONLINE_MODE:
        raise UnionContextualVetoExperimentError(
            "Parent checkpoint was not produced by causal-online learning"
        )
    current = _checkpoint_from_forecast(
        forecast,
        cutoff=cutoff,
        learning_mode=CAUSAL_ONLINE_MODE,
    )
    if _canonical_json_bytes(parent) != _canonical_json_bytes(current):
        raise UnionContextualVetoExperimentError(
            "Regenerated learner state does not match the committed parent checkpoint"
        )


def _flatten_forecast(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.reset_index(names="decision_date")


def _ledger_payloads(
    ledgers: Mapping[str, pd.DataFrame], *, prefix: str
) -> dict[str, bytes]:
    return {
        f"{prefix}_{cost_name}_ledgers.csv": _frame_csv_bytes(frame)
        for cost_name, frame in ledgers.items()
    }


_EPISODE_COLUMNS = (
    "decision_date",
    "entry_date",
    "exit_date",
    "raw_active_log_edge",
    "net_active_log_edge",
    "win",
)
_VETO_BENEFIT_COLUMNS = (
    "decision_date",
    "entry_date",
    "exit_date",
    "raw_union_cash_edge",
    "net_union_cash_edge",
    "veto_benefit",
    "beneficial_veto",
    *FEATURE_COLUMNS,
    "n_raw",
    "n_eff",
    "model_mu",
    "model_se",
    "model_upper",
    "model_ready",
)


def _episode_payloads(
    episodes: Mapping[str, Mapping[str, pd.DataFrame]], *, prefix: str
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for cost_name, policies in episodes.items():
        for policy in ("learner", "union"):
            value = policies[policy].copy()
            if value.empty:
                value = pd.DataFrame(columns=_EPISODE_COLUMNS)
            payloads[
                f"{prefix}_{cost_name}_{policy}_episodes.csv"
            ] = _frame_csv_bytes(value)
    return payloads


def _benefit_payloads(
    benefits: Mapping[str, pd.DataFrame], *, prefix: str
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for cost_name, frame in benefits.items():
        value = frame.copy()
        if value.empty:
            value = pd.DataFrame(columns=_VETO_BENEFIT_COLUMNS)
        payloads[f"{prefix}_{cost_name}_veto_benefits.csv"] = (
            _frame_csv_bytes(value)
        )
    return payloads


def _stage_bundle(
    *,
    output_dir: Path,
    run_id: str,
    stage: str,
    stage_pass: bool,
    report: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    source_provenance: Mapping[str, Any],
    git_identity: Mapping[str, Any],
    parent_manifest: Mapping[str, Any] | None,
    deadline: _Deadline,
) -> dict[str, Any]:
    data_payloads = {".gitattributes": b"* -text\n", **dict(payloads)}
    data_payloads["report.json"] = _pretty_json_bytes(report)
    hashes = {name: _sha256(data) for name, data in sorted(data_payloads.items())}
    manifest_payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "stage_pass": bool(stage_pass),
        "run_id": run_id,
        "source_provenance": dict(source_provenance),
        "source_path": source_provenance["source_path"],
        "bounded_result_sha256": source_provenance["bounded_result_sha256"],
        "git_identity": dict(git_identity),
        "parent_manifest_sha256": (
            parent_manifest.get("manifest_sha256") if parent_manifest else None
        ),
        "payload_sha256": hashes,
        "execution": {
            "asset": "AAPL",
            "actions": ["LONG_100_PERCENT", "CASH_100_PERCENT"],
            "maximum_target_exposure": 1.0,
            "learner_is_pure_union_veto": True,
            "shorting": False,
            "leverage": False,
            "borrowing": False,
            "cash_interest": False,
            "network_access": False,
            "llm_calls": 0,
            "api_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "strict_union_improvement": STRICT_UNION_IMPROVEMENT,
            "final_material_active_log_edge": FINAL_MATERIAL_ACTIVE_LOG_EDGE,
            "runtime_limit_seconds": RUN_TIME_LIMIT_SECONDS,
        },
    }
    manifest = _manifest(manifest_payload)
    data_payloads["stage_manifest.json"] = _pretty_json_bytes(manifest)
    run_dir = output_dir.resolve() / run_id
    checksums = _seal_bundle(
        run_dir,
        data_payloads,
        before_promote=lambda: deadline.check("before artifact promotion"),
    )
    return {
        "stage": stage,
        "stage_pass": bool(stage_pass),
        "run_id": run_id,
        "artifact_dir": str(run_dir),
        "stage_manifest": str(run_dir / "stage_manifest.json"),
        "manifest_sha256": manifest["manifest_sha256"],
        "checksums": checksums,
    }


def _runtime_report(deadline: _Deadline) -> dict[str, Any]:
    elapsed = deadline.elapsed()
    return {
        "seconds_before_seal": elapsed,
        "limit_seconds": RUN_TIME_LIMIT_SECONDS,
        "within_limit": elapsed <= RUN_TIME_LIMIT_SECONDS,
        "network_access": False,
        "llm_calls": 0,
        "api_calls": 0,
        "external_cost_usd": 0.0,
    }


def run_development(
    *,
    repo_root: Path,
    price_artifact: Path,
    output_dir: Path,
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    deadline = _Deadline(clock)
    git_identity = _clean_git_identity(repo_root)
    input_identity = _tracked_input_identity(repo_root, price_artifact)
    deadline.check("development authorization")
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=DEVELOPMENT_END,
        required_last_session=pd.Timestamp("2018-12-31"),
    )
    provenance["tracked_input"] = input_identity
    deadline.check("bounded development load")
    forecast = build_union_contextual_veto_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    metrics, ledgers, episodes, benefits, integrity = _evaluate_policy_set(
        frame,
        forecast,
        periods=_development_periods(),
        administrative_start=DEVELOPMENT_START,
    )
    gates = apply_development_gates(metrics, integrity)
    deadline.check("development replay and gates")
    resolved_run_id = _safe_run_id(
        run_id, prefix="union-contextual-veto-development"
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "development",
        "run_id": resolved_run_id,
        "evidence_classification": (
            "causal_prequential_2005_2018_diagnostic_successor_not_globally_pristine"
        ),
        "physical_data_end": DEVELOPMENT_END.date().isoformat(),
        "later_outcomes_accessed": False,
        "primary_causal_online_metrics": metrics,
        "integrity": integrity,
        "gate_report": gates,
        "runtime": _runtime_report(deadline),
    }
    checkpoint = _checkpoint_from_forecast(
        forecast, cutoff=DEVELOPMENT_END, learning_mode=CAUSAL_ONLINE_MODE
    )
    payloads = {
        "development_prices_through_2018.csv": _frame_csv_bytes(
            frame.reset_index(names="date")
        ),
        "development_forecast_through_2018.csv": _frame_csv_bytes(
            _flatten_forecast(forecast)
        ),
        "development_metrics.json": _pretty_json_bytes(metrics),
        "development_gate_report.json": _pretty_json_bytes(gates),
        "development_checkpoint_through_2018.json": _pretty_json_bytes(checkpoint),
        "input_provenance.json": _pretty_json_bytes(provenance),
        **_ledger_payloads(ledgers, prefix="development"),
        **_episode_payloads(episodes, prefix="development"),
        **_benefit_payloads(benefits, prefix="development"),
    }
    deadline.check("before development seal")
    result = _stage_bundle(
        output_dir=output_dir,
        run_id=resolved_run_id,
        stage="development",
        stage_pass=bool(gates["passed"]),
        report=report,
        payloads=payloads,
        source_provenance=provenance,
        git_identity=git_identity,
        parent_manifest=None,
        deadline=deadline,
    )
    deadline.check("development seal")
    return {**result, "gate_report": gates, "runtime_seconds": deadline.elapsed()}


def run_validation(
    *,
    repo_root: Path,
    price_artifact: Path,
    development_manifest: Path,
    output_dir: Path,
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    deadline = _Deadline(clock)
    git_identity = _clean_git_identity(repo_root)
    parent = _validated_prior_manifest(
        repo_root=repo_root,
        path=development_manifest,
        expected_stage="development",
    )
    _require_dependency_continuity(git_identity, parent)
    input_identity = _tracked_input_identity(repo_root, price_artifact)
    deadline.check("validation authorization")
    # The committed development pass is verified above before the first
    # operation that is capable of returning a 2019 market row.
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=VALIDATION_END,
        required_last_session=pd.Timestamp("2023-12-29"),
    )
    provenance["tracked_input"] = input_identity
    _require_source_continuity(
        frame, provenance, parent, parent_end=DEVELOPMENT_END
    )
    deadline.check("authorized validation load")
    parent_prefix = frame.loc[:DEVELOPMENT_END].copy()
    parent_prefix_forecast = build_union_contextual_veto_forecast(
        parent_prefix, learning_mode=CAUSAL_ONLINE_MODE
    )
    _require_checkpoint_continuity(
        parent_prefix_forecast,
        cutoff=DEVELOPMENT_END,
        parent_manifest_path=development_manifest,
        checkpoint_filename="development_checkpoint_through_2018.json",
    )
    deadline.check("development checkpoint replay")
    # No forecast is constructed over a validation row until the independently
    # replayed bounded parent checkpoint above agrees byte-for-byte.
    online = build_union_contextual_veto_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    frozen = build_union_contextual_veto_forecast(
        frame,
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=DEVELOPMENT_END,
    )
    (
        online_metrics,
        online_ledgers,
        online_episodes,
        online_benefits,
        online_integrity,
    ) = _evaluate_policy_set(
        frame,
        online,
        periods=_validation_periods(),
        administrative_start=VALIDATION_START,
    )
    (
        frozen_metrics,
        frozen_ledgers,
        frozen_episodes,
        frozen_benefits,
        frozen_integrity,
    ) = _evaluate_policy_set(
        frame,
        frozen,
        periods=_validation_periods(),
        administrative_start=VALIDATION_START,
    )
    gates = apply_validation_gates(online_metrics, online_integrity)
    deadline.check("validation replay and gates")
    resolved_run_id = _safe_run_id(
        run_id, prefix="union-contextual-veto-validation"
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "validation",
        "run_id": resolved_run_id,
        "evidence_classification": (
            "untouched_causal_online_2019_2023_confirmation_for_exact_veto_design"
        ),
        "physical_data_end": VALIDATION_END.date().isoformat(),
        "post_2023_outcomes_accessed": False,
        "primary_causal_online_metrics": online_metrics,
        "primary_causal_online_integrity": online_integrity,
        "secondary_frozen_2018_diagnostic_metrics": frozen_metrics,
        "secondary_frozen_2018_integrity": frozen_integrity,
        "frozen_diagnostic_cannot_rescue_online_failure": True,
        "gate_report": gates,
        "runtime": _runtime_report(deadline),
    }
    checkpoint = _checkpoint_from_forecast(
        online, cutoff=VALIDATION_END, learning_mode=CAUSAL_ONLINE_MODE
    )
    payloads = {
        "authorized_development_manifest.json": _pretty_json_bytes(parent),
        "validation_prices_through_2023.csv": _frame_csv_bytes(
            frame.reset_index(names="date")
        ),
        "validation_online_forecast.csv": _frame_csv_bytes(
            _flatten_forecast(online.loc[VALIDATION_START:VALIDATION_END])
        ),
        "validation_frozen_forecast.csv": _frame_csv_bytes(
            _flatten_forecast(frozen.loc[VALIDATION_START:VALIDATION_END])
        ),
        "validation_online_metrics.json": _pretty_json_bytes(online_metrics),
        "validation_frozen_metrics.json": _pretty_json_bytes(frozen_metrics),
        "validation_gate_report.json": _pretty_json_bytes(gates),
        "validation_checkpoint_through_2023.json": _pretty_json_bytes(checkpoint),
        "input_provenance.json": _pretty_json_bytes(provenance),
        **_ledger_payloads(online_ledgers, prefix="validation_online"),
        **_ledger_payloads(frozen_ledgers, prefix="validation_frozen"),
        **_episode_payloads(online_episodes, prefix="validation_online"),
        **_episode_payloads(frozen_episodes, prefix="validation_frozen"),
        **_benefit_payloads(online_benefits, prefix="validation_online"),
        **_benefit_payloads(frozen_benefits, prefix="validation_frozen"),
    }
    deadline.check("before validation seal")
    result = _stage_bundle(
        output_dir=output_dir,
        run_id=resolved_run_id,
        stage="validation",
        stage_pass=bool(gates["passed"]),
        report=report,
        payloads=payloads,
        source_provenance=provenance,
        git_identity=git_identity,
        parent_manifest=parent,
        deadline=deadline,
    )
    deadline.check("validation seal")
    return {**result, "gate_report": gates, "runtime_seconds": deadline.elapsed()}


def run_final(
    *,
    repo_root: Path,
    price_artifact: Path,
    validation_manifest: Path,
    output_dir: Path,
    run_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    deadline = _Deadline(clock)
    git_identity = _clean_git_identity(repo_root)
    parent = _validated_prior_manifest(
        repo_root=repo_root,
        path=validation_manifest,
        expected_stage="validation",
    )
    _require_dependency_continuity(git_identity, parent)
    input_identity = _tracked_input_identity(repo_root, price_artifact)
    deadline.check("final authorization")
    # The committed validation pass is verified above before the first
    # operation that is capable of returning a 2024+ market row.
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=FINAL_END,
        required_last_session=FINAL_END,
    )
    provenance["tracked_input"] = input_identity
    _require_source_continuity(frame, provenance, parent, parent_end=VALIDATION_END)
    deadline.check("authorized final load")
    parent_prefix = frame.loc[:VALIDATION_END].copy()
    parent_prefix_forecast = build_union_contextual_veto_forecast(
        parent_prefix, learning_mode=CAUSAL_ONLINE_MODE
    )
    _require_checkpoint_continuity(
        parent_prefix_forecast,
        cutoff=VALIDATION_END,
        parent_manifest_path=validation_manifest,
        checkpoint_filename="validation_checkpoint_through_2023.json",
    )
    deadline.check("validation checkpoint replay")
    # No forecast is constructed over a final-audit row until the independently
    # replayed bounded parent checkpoint above agrees byte-for-byte.
    online = build_union_contextual_veto_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    frozen = build_union_contextual_veto_forecast(
        frame,
        learning_mode=FROZEN_CUTOFF_MODE,
        frozen_cutoff=VALIDATION_END,
    )
    (
        online_metrics,
        online_ledgers,
        online_episodes,
        online_benefits,
        online_integrity,
    ) = _evaluate_policy_set(
        frame,
        online,
        periods=_final_periods(),
        administrative_start=FINAL_START,
    )
    (
        frozen_metrics,
        frozen_ledgers,
        frozen_episodes,
        frozen_benefits,
        frozen_integrity,
    ) = _evaluate_policy_set(
        frame,
        frozen,
        periods=_final_periods(),
        administrative_start=FINAL_START,
    )
    (
        lifetime_metrics,
        lifetime_ledgers,
        lifetime_episodes,
        lifetime_benefits,
        lifetime_integrity,
    ) = _evaluate_policy_set(
        frame,
        online,
        periods=_lifetime_periods(),
        administrative_start=DEVELOPMENT_START,
    )
    lifetime_diagnostic = _lifetime_diagnostic(lifetime_metrics)
    gates = apply_final_gates(online_metrics, online_integrity)
    deadline.check("final replay and gates")
    resolved_run_id = _safe_run_id(run_id, prefix="union-contextual-veto-final")
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "final",
        "run_id": resolved_run_id,
        "evidence_classification": "repeated_2024_2026_ytd_historical_audit",
        "physical_data_end": FINAL_END.date().isoformat(),
        "primary_causal_online_final_metrics": online_metrics,
        "primary_causal_online_integrity": online_integrity,
        "secondary_frozen_2023_diagnostic_metrics": frozen_metrics,
        "secondary_frozen_2023_integrity": frozen_integrity,
        "lifetime_causal_online_2005_2026_ytd_metrics": lifetime_metrics,
        "lifetime_causal_online_integrity": lifetime_integrity,
        "lifetime_diagnostic": lifetime_diagnostic,
        "frozen_and_lifetime_diagnostics_cannot_rescue_strict_failure": True,
        "gate_report": gates,
        "runtime": _runtime_report(deadline),
    }
    checkpoint = _checkpoint_from_forecast(
        online, cutoff=FINAL_END, learning_mode=CAUSAL_ONLINE_MODE
    )
    payloads = {
        "authorized_validation_manifest.json": _pretty_json_bytes(parent),
        "final_prices_through_2026_ytd.csv": _frame_csv_bytes(
            frame.reset_index(names="date")
        ),
        "final_online_forecast.csv": _frame_csv_bytes(
            _flatten_forecast(online.loc[FINAL_START:FINAL_END])
        ),
        "final_frozen_forecast.csv": _frame_csv_bytes(
            _flatten_forecast(frozen.loc[FINAL_START:FINAL_END])
        ),
        "final_online_metrics.json": _pretty_json_bytes(online_metrics),
        "final_frozen_metrics.json": _pretty_json_bytes(frozen_metrics),
        "lifetime_online_metrics.json": _pretty_json_bytes(lifetime_metrics),
        "lifetime_diagnostic.json": _pretty_json_bytes(lifetime_diagnostic),
        "final_gate_report.json": _pretty_json_bytes(gates),
        "online_checkpoint_through_2026_ytd.json": _pretty_json_bytes(checkpoint),
        "input_provenance.json": _pretty_json_bytes(provenance),
        **_ledger_payloads(online_ledgers, prefix="final_online"),
        **_ledger_payloads(frozen_ledgers, prefix="final_frozen"),
        **_ledger_payloads(lifetime_ledgers, prefix="lifetime_online"),
        **_episode_payloads(online_episodes, prefix="final_online"),
        **_episode_payloads(frozen_episodes, prefix="final_frozen"),
        **_episode_payloads(lifetime_episodes, prefix="lifetime_online"),
        **_benefit_payloads(online_benefits, prefix="final_online"),
        **_benefit_payloads(frozen_benefits, prefix="final_frozen"),
        **_benefit_payloads(lifetime_benefits, prefix="lifetime_online"),
    }
    deadline.check("before final seal")
    result = _stage_bundle(
        output_dir=output_dir,
        run_id=resolved_run_id,
        stage="final",
        stage_pass=bool(gates["passed"]),
        report=report,
        payloads=payloads,
        source_provenance=provenance,
        git_identity=git_identity,
        parent_manifest=parent,
        deadline=deadline,
    )
    deadline.check("final seal")
    return {**result, "gate_report": gates, "runtime_seconds": deadline.elapsed()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("develop", "validate", "final"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--price-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    common = {
        "repo_root": args.repo_root,
        "price_artifact": args.price_artifact,
        "output_dir": args.output_dir,
        "run_id": args.run_id,
    }
    if args.command == "develop":
        if args.development_manifest or args.validation_manifest:
            raise SystemExit("develop does not accept a prior manifest")
        result = run_development(**common)
    elif args.command == "validate":
        if args.development_manifest is None or args.validation_manifest:
            raise SystemExit("validate requires only --development-manifest")
        result = run_validation(
            **common, development_manifest=args.development_manifest
        )
    else:
        if args.validation_manifest is None or args.development_manifest:
            raise SystemExit("final requires only --validation-manifest")
        result = run_final(**common, validation_manifest=args.validation_manifest)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "UnionContextualVetoExperimentError",
    "apply_development_gates",
    "apply_final_gates",
    "apply_validation_gates",
    "load_bounded_prices",
    "main",
    "run_development",
    "run_final",
    "run_validation",
]
