"""Sealed staged runner for the binary-regime union-selector experiment.

The account begins once in 2005.  Validation and final intervals are reporting
windows over that same ledger and learner, never fresh backtest accounts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import chronological_exhaustion_experiment as _sealed_base
from .binary_regime_union_selector import (
    CAUSAL_ONLINE_MODE,
    FROZEN_CUTOFF_MODE,
    LESSON_DISCOUNT,
    MIN_EFFECTIVE_LESSONS,
    NEGATIVE_MEAN_THRESHOLD,
    POSITIVE_MEAN_THRESHOLD,
    REGIME_FEATURE_COLUMNS,
    REGIME_NAMES,
    build_binary_regime_union_selector_forecast,
    restore_pending_regime_lessons,
    restore_regime_states,
)
from .chronological_exhaustion_expert import canonicalize_one_session_signals
from .deterministic_aapl import CostAssumptions, EvaluationPeriod, compare_ledgers
from .unleveraged_aapl import (
    assert_unleveraged_ledger,
    canonical_context_frame,
    simulate_unleveraged_period,
)


CONTRACT_VERSION = "aapl-binary-regime-union-selector-v1"
DEVELOPMENT_START = pd.Timestamp("2005-01-01")
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
VALIDATION_START = pd.Timestamp("2019-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")
FINAL_START = pd.Timestamp("2024-01-01")
FINAL_END = pd.Timestamp("2026-07-09")
INITIAL_CASH = 1000.0
RUN_TIME_LIMIT_SECONDS = 3600.0
STRICT_UNION_IMPROVEMENT = 0.0001
FINAL_MATERIAL_ACTIVE_LOG_EDGE = 0.001
COST_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("base_5bps", 5.0),
    ("stress_10bps", 10.0),
)
CONTRACT_PATH = Path("docs/aapl_binary_regime_union_selector_v1.md")
IMPLEMENTATION_PATHS = (
    Path("agent_benchmark/binary_regime_union_selector.py"),
    Path("agent_benchmark/binary_regime_union_selector_experiment.py"),
    Path("agent_benchmark/chronological_exhaustion_expert.py"),
    # Physical-input, ledger, and atomic-sealing primitives are inherited from
    # this exact tracked dependency and therefore bound into every manifest.
    Path("agent_benchmark/chronological_exhaustion_experiment.py"),
    Path("agent_benchmark/deterministic_aapl.py"),
    Path("agent_benchmark/unleveraged_aapl.py"),
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
CALIBRATION_REFERENCE = {
    "risk_on": {
        "n_raw": 25,
        "n_eff": 23.5559514102397,
        "mean": -0.0039042043622158,
        "latch": "LONG",
    },
    "not_risk_on": {
        "n_raw": 138,
        "n_eff": 99.8582587508293,
        "mean": 0.0104308978451598,
        "latch": "CASH",
    },
}
SELECTOR_REFERENCE = {
    "base_5bps": 1.20182849740637,
    "stress_10bps": 1.10082843848966,
}

PHYSICAL_SNAPSHOT_COLUMNS = _sealed_base.PHYSICAL_SNAPSHOT_COLUMNS
STAGE_SESSION_COVERAGE = _sealed_base.STAGE_SESSION_COVERAGE
STAGE_BOUNDED_RESULT_SHA256 = _sealed_base.STAGE_BOUNDED_RESULT_SHA256
PRICE_QUERY = _sealed_base.PRICE_QUERY

BinaryRegimeUnionSelectorExperimentError = (
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
_positive_concentration = _sealed_base._positive_concentration
_prefix_sha256 = _sealed_base._prefix_sha256
_period_return = _sealed_base._period_return
_episode_rows = _sealed_base._episode_rows


_PREFIX_PRICE_QUERY = """
SELECT
  CAST(date AS DATE) AS date,
  CAST(aapl_open AS DOUBLE) AS aapl_open,
  CAST(aapl_close AS DOUBLE) AS aapl_close,
  CAST(aapl_adj_close AS DOUBLE) AS aapl_adj_close,
  CAST(spy_adj_close AS DOUBLE) AS spy_adj_close,
  CAST(qqq_adj_close AS DOUBLE) AS qqq_adj_close
FROM read_csv_auto(?, header = true)
WHERE CAST(date AS DATE) <= CAST(? AS DATE)
ORDER BY CAST(date AS DATE)
""".strip()


def load_bounded_prices(
    path: Path, *, end: pd.Timestamp, required_last_session: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return only the authorized prefix, even when the tracked file is longer.

    The strict base loader remains the path for a physically stage-bounded
    file.  A parent-checkpoint preflight may point at the next stage's longer
    tracked snapshot; in that case DuckDB returns only rows through ``end``.
    No DataFrame containing a new-stage row exists before checkpoint replay.
    """

    source = path.resolve()
    if not source.is_file():
        raise BinaryRegimeUnionSelectorExperimentError(
            f"Price artifact does not exist: {source}"
        )
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = tuple(next(reader))
    except (OSError, UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Physical stage snapshot header is unreadable"
        ) from exc
    if header != PHYSICAL_SNAPSHOT_COLUMNS:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Physical stage snapshot has unexpected columns"
        )
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            physical_last_value = connection.execute(
                "SELECT MAX(CAST(date AS DATE)) FROM read_csv_auto(?, header=true)",
                [str(source)],
            ).fetchone()[0]
        finally:
            connection.close()
    except Exception as exc:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Physical stage snapshot bound inspection failed"
        ) from exc
    physical_last = pd.Timestamp(physical_last_value)
    required = pd.Timestamp(required_last_session)
    bound = pd.Timestamp(end)
    if physical_last < required:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Physical stage snapshot ends before the required session"
        )
    if physical_last == required:
        return _sealed_base.load_bounded_prices(
            source, end=bound, required_last_session=required
        )
    try:
        connection = duckdb.connect(":memory:")
        try:
            raw = connection.execute(
                _PREFIX_PRICE_QUERY,
                [str(source), bound.date().isoformat()],
            ).fetchdf()
        finally:
            connection.close()
        frame = canonical_context_frame(raw)
    except Exception as exc:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Bounded parent-prefix query failed"
        ) from exc
    if (
        frame.empty
        or frame.index.max() != required
        or frame.index.max() > bound
        or frame.index.min() > pd.Timestamp("1999-06-01")
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Bounded parent prefix has an unauthorized date range"
        )
    session_payload = "".join(
        f"{value.date().isoformat()}\n" for value in frame.index
    ).encode("ascii")
    observed_coverage = {
        "first_session": frame.index.min().date().isoformat(),
        "last_session": frame.index.max().date().isoformat(),
        "observations": int(len(frame)),
        "date_sequence_sha256": hashlib.sha256(session_payload).hexdigest(),
    }
    expected_coverage = STAGE_SESSION_COVERAGE.get(required.date().isoformat())
    if observed_coverage != expected_coverage:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Bounded parent prefix does not match the frozen session sequence"
        )
    bounded_result_sha256 = _sha256(
        _frame_csv_bytes(frame.reset_index(names="date"))
    )
    expected_result_sha256 = STAGE_BOUNDED_RESULT_SHA256.get(
        required.date().isoformat()
    )
    if bounded_result_sha256 != expected_result_sha256:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Bounded parent-prefix prices do not match the frozen source"
        )
    return frame, {
        "source_type": "physically_bounded_local_csv_duckdb_query",
        "source_path": str(source),
        "query": _PREFIX_PRICE_QUERY,
        "query_parameters": [str(source), bound.date().isoformat()],
        "bounded_first_date": frame.index.min().date().isoformat(),
        "bounded_last_date": frame.index.max().date().isoformat(),
        "bounded_rows": int(len(frame)),
        "bounded_result_sha256": bounded_result_sha256,
        "expected_bounded_result_sha256": expected_result_sha256,
        "session_coverage": observed_coverage,
        "physical_snapshot_has_later_rows": True,
        "physical_snapshot_last_session": physical_last.date().isoformat(),
        "rows_after_bound_returned": False,
        "network_access": False,
    }


def _require_exact_physical_stage_bound(
    provenance: Mapping[str, Any], *, stage: str
) -> None:
    """Reject a full-stage input whose physical file contains later rows."""

    if provenance.get("physical_snapshot_has_later_rows") is not False:
        raise BinaryRegimeUnionSelectorExperimentError(
            f"{stage} requires a physically exact stage-bounded snapshot"
        )
    if provenance.get("rows_after_bound_returned") is not False:
        raise BinaryRegimeUnionSelectorExperimentError(
            f"{stage} price loader returned rows after its authorized bound"
        )


def _json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryRegimeUnionSelectorExperimentError(
            f"{description} is unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise BinaryRegimeUnionSelectorExperimentError(
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
        upstream = _git_text(
            root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        upstream_commit = _git_text(root, "rev-parse", "@{upstream}")
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Stage requires a valid Git repository with a pushed upstream"
        ) from exc
    if actual != root:
        raise BinaryRegimeUnionSelectorExperimentError(
            "repo_root must be the actual Git root"
        )
    if status:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Stage requires a completely clean worktree"
        )
    if not branch or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Stage requires an attached valid Git commit"
        )
    if (
        not upstream
        or re.fullmatch(r"[0-9a-f]{40}", upstream_commit) is None
        or upstream_commit != commit
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Stage requires HEAD to equal its pushed upstream commit"
        )
    tracked_hashes: dict[str, str] = {}
    for relative in (*IMPLEMENTATION_PATHS, CONTRACT_PATH):
        name = relative.as_posix()
        try:
            _git_bytes(root, "ls-files", "--error-unmatch", "--", name)
            committed = _git_bytes(root, "show", f"HEAD:{name}")
        except subprocess.CalledProcessError as exc:
            raise BinaryRegimeUnionSelectorExperimentError(
                f"Frozen dependency is not tracked: {name}"
            ) from exc
        tracked_hashes[name] = _sha256(committed)
    return {
        "branch": branch,
        "commit": commit,
        "upstream": upstream,
        "upstream_commit": upstream_commit,
        "head_equals_upstream": True,
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
            "development_cold_start_report.json",
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
        raise BinaryRegimeUnionSelectorExperimentError(
            f"Unsupported parent stage: {stage}"
        )
    for prefix in prefixes:
        for cost_name, _ in COST_SCENARIOS:
            fixed.update(
                {
                    f"{prefix}_{cost_name}_ledgers.csv",
                    f"{prefix}_{cost_name}_selector_episodes.csv",
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
        raise BinaryRegimeUnionSelectorExperimentError(
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
        raise BinaryRegimeUnionSelectorExperimentError(
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
        raise BinaryRegimeUnionSelectorExperimentError(
            "Prior manifest must be an exact tracked file at HEAD"
        ) from exc
    if committed != local:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Prior manifest differs from its committed blob"
        )
    value = _json_object(manifest_path, description="Prior manifest")
    payload = dict(value)
    recorded = payload.pop("manifest_sha256", None)
    if recorded != _sha256(_canonical_json_bytes(payload)):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Prior manifest self-hash is invalid"
        )
    if (
        value.get("contract_version") != CONTRACT_VERSION
        or value.get("stage") != expected_stage
        or value.get("stage_pass") is not True
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
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
            raise BinaryRegimeUnionSelectorExperimentError(
                f"Prior artifact is not tracked: {filename}"
            ) from exc
        if artifact_committed != artifact_local or _sha256(artifact_local) != expected:
            raise BinaryRegimeUnionSelectorExperimentError(
                f"Prior artifact checksum changed: {filename}"
            )
        verified[filename] = artifact_local
    gate_name = f"{expected_stage}_gate_report.json"
    try:
        gate = json.loads(verified[gate_name].decode("utf-8"))
        report = json.loads(verified["report.json"].decode("utf-8"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinaryRegimeUnionSelectorExperimentError(
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
        raise BinaryRegimeUnionSelectorExperimentError(
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
        raise BinaryRegimeUnionSelectorExperimentError(
            "Physical price-snapshot lineage changed between stages"
        )
    if _prefix_sha256(frame, end=parent_end) != parent.get(
        "bounded_result_sha256"
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Authorized historical price prefix changed"
        )


_REGIME_STATE_FIELDS = (
    "n_raw",
    "n_eff",
    "weighted_label_sum",
    "weighted_squared_label_sum",
    "mean",
    "ready",
    "cash_selected",
)


def _state_column(regime: str, field: str) -> str:
    return f"{regime}_{field}"


def _required_forecast_columns() -> frozenset[str]:
    state_columns = {
        _state_column(regime, field)
        for regime in REGIME_NAMES
        for field in _REGIME_STATE_FIELDS
    }
    return frozenset(
        {
            "contextual_virtual_signal",
            "weak_trend_virtual_signal",
            "spy_return_20",
            "qqq_return_20",
            "risk_regime_ready",
            "risk_on",
            "union_candidate_signal",
            "shadow_canonical_union_signal",
            "canonical_union_cash_signal",
            "shadow_matures_on_close",
            "shadow_matured_now",
            "shadow_signal_close",
            "shadow_opportunity_risk_on",
            "shadow_matured_signal_risk_on",
            "shadow_label_10bps",
            "shadow_lesson_added_now",
            "shadow_pending",
            "selector_regime_n_eff",
            "selector_regime_mean",
            "selector_regime_ready",
            "selector_cash_prediction",
            "selector_skip_prediction",
            "selector_skip_signal",
            "learner_cash_signal",
            "target_exposure",
            *state_columns,
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
        raise BinaryRegimeUnionSelectorExperimentError(
            "Forecast rows are not exactly aligned to physical market rows"
        )
    missing = sorted(_required_forecast_columns().difference(value.columns))
    if missing:
        raise BinaryRegimeUnionSelectorExperimentError(
            f"Core forecast lacks required columns: {missing}"
        )
    attrs = value.attrs
    expected_attrs = {
        "regime_feature_order": list(REGIME_FEATURE_COLUMNS),
        "lesson_maturity": "close t+2 before same-close prediction",
        "lesson_memory_start": "2000-01-01",
        "lesson_discount": LESSON_DISCOUNT,
        "minimum_effective_lessons": MIN_EFFECTIVE_LESSONS,
        "positive_mean_threshold": POSITIVE_MEAN_THRESHOLD,
        "negative_mean_threshold": NEGATIVE_MEAN_THRESHOLD,
        "risk_on_structural_default_cash": False,
        "not_risk_on_structural_default_cash": True,
    }
    if any(attrs.get(name) != expected for name, expected in expected_attrs.items()):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Core forecast attributes differ from the frozen selector contract"
        )
    if not isinstance(attrs.get("risk_on_definition"), str) or not attrs[
        "risk_on_definition"
    ]:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Core forecast lacks the frozen risk-on definition"
        )
    if not math.isclose(
        float(attrs.get("round_trip_log_friction", math.nan)),
        math.log(0.999 / 1.001),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Core forecast uses the wrong shadow-lesson friction"
        )
    mode = attrs.get("learning_mode")
    if mode not in {CAUSAL_ONLINE_MODE, FROZEN_CUTOFF_MODE}:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Core forecast has an invalid learning mode"
        )
    if mode == CAUSAL_ONLINE_MODE and (
        attrs.get("frozen_cutoff") is not None
        or attrs.get("post_cutoff_outcomes_used_for_learning") is not True
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Causal-online forecast attributes are inconsistent"
        )
    if mode == FROZEN_CUTOFF_MODE and (
        not isinstance(attrs.get("frozen_cutoff"), str)
        or attrs.get("post_cutoff_outcomes_used_for_learning") is not False
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Frozen forecast attributes are inconsistent"
        )
    if not isinstance(attrs.get("final_states"), dict) or not isinstance(
        attrs.get("pending_lessons"), dict
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Core forecast lacks strict serialized continuation state"
        )
    if not isinstance(attrs.get("shadow_stream"), str) or not attrs[
        "shadow_stream"
    ]:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Core forecast lacks shadow-stream metadata"
        )
    return value


def _bool_column(frame: pd.DataFrame, name: str) -> pd.Series:
    values = frame[name]
    if values.isna().any():
        raise BinaryRegimeUnionSelectorExperimentError(
            f"Forecast boolean column contains missing values: {name}"
        )
    return values.astype(bool)


def _finite_numeric(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    value = frame.loc[:, list(names)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(value.to_numpy(dtype=float)).all():
        raise BinaryRegimeUnionSelectorExperimentError(
            f"Forecast has nonfinite numeric state: {list(names)}"
        )
    return value


def _validate_regime_state_replay(value: pd.DataFrame) -> dict[str, Any]:
    """Independently replay both EW states from the forecast's matured labels."""

    lesson_added = _bool_column(value, "shadow_lesson_added_now")
    matured_regime = value["shadow_matured_signal_risk_on"]
    labels = pd.to_numeric(value["shadow_label_10bps"], errors="coerce")
    defaults = {"risk_on": False, "not_risk_on": True}
    state = {
        regime: {
            "n_raw": 0,
            "n_eff": 0.0,
            "weighted_label_sum": 0.0,
            "weighted_squared_label_sum": 0.0,
            "cash_selected": defaults[regime],
        }
        for regime in REGIME_NAMES
    }
    switch_counts = {regime: 0 for regime in REGIME_NAMES}
    for position in range(len(value)):
        if bool(lesson_added.iloc[position]):
            if pd.isna(matured_regime.iloc[position]):
                raise BinaryRegimeUnionSelectorExperimentError(
                    "Admitted lesson lacks its signal-time regime"
                )
            regime = (
                "risk_on"
                if bool(matured_regime.iloc[position])
                else "not_risk_on"
            )
            y = float(labels.iloc[position])
            if not math.isfinite(y):
                raise BinaryRegimeUnionSelectorExperimentError(
                    "Admitted lesson label is nonfinite"
                )
            current = state[regime]
            current["n_raw"] += 1
            current["n_eff"] = LESSON_DISCOUNT * current["n_eff"] + 1.0
            current["weighted_label_sum"] = (
                LESSON_DISCOUNT * current["weighted_label_sum"] + y
            )
            current["weighted_squared_label_sum"] = (
                LESSON_DISCOUNT * current["weighted_squared_label_sum"] + y * y
            )
            mean = current["weighted_label_sum"] / current["n_eff"]
            prior_latch = bool(current["cash_selected"])
            if current["n_eff"] < MIN_EFFECTIVE_LESSONS:
                current["cash_selected"] = defaults[regime]
            elif mean > POSITIVE_MEAN_THRESHOLD:
                current["cash_selected"] = True
            elif mean < NEGATIVE_MEAN_THRESHOLD:
                current["cash_selected"] = False
            if bool(current["cash_selected"]) != prior_latch:
                switch_counts[regime] += 1

        for regime in REGIME_NAMES:
            current = state[regime]
            expected = {
                "n_raw": float(current["n_raw"]),
                "n_eff": float(current["n_eff"]),
                "weighted_label_sum": float(current["weighted_label_sum"]),
                "weighted_squared_label_sum": float(
                    current["weighted_squared_label_sum"]
                ),
            }
            for field, expected_value in expected.items():
                actual = float(value.iloc[position][_state_column(regime, field)])
                if not math.isclose(
                    actual, expected_value, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise BinaryRegimeUnionSelectorExperimentError(
                        f"{regime} state replay mismatch at {value.index[position]}: {field}"
                    )
            expected_mean = (
                float(current["weighted_label_sum"]) / float(current["n_eff"])
                if float(current["n_eff"]) > 0.0
                else math.nan
            )
            actual_mean = float(value.iloc[position][_state_column(regime, "mean")])
            if math.isnan(expected_mean):
                if not math.isnan(actual_mean):
                    raise BinaryRegimeUnionSelectorExperimentError(
                        f"{regime} empty-state mean is not missing"
                    )
            elif not math.isclose(
                actual_mean, expected_mean, rel_tol=0.0, abs_tol=1e-12
            ):
                raise BinaryRegimeUnionSelectorExperimentError(
                    f"{regime} discounted mean replay mismatch"
                )
            ready = float(current["n_eff"]) >= MIN_EFFECTIVE_LESSONS
            if bool(value.iloc[position][_state_column(regime, "ready")]) != ready:
                raise BinaryRegimeUnionSelectorExperimentError(
                    f"{regime} readiness replay mismatch"
                )
            if bool(
                value.iloc[position][_state_column(regime, "cash_selected")]
            ) != bool(current["cash_selected"]):
                raise BinaryRegimeUnionSelectorExperimentError(
                    f"{regime} latch replay mismatch"
                )
    return {
        "regime_state_replay_exact": True,
        "regime_switch_counts": switch_counts,
        "final_replayed_states": {
            regime: {
                **state[regime],
                "mean": (
                    float(state[regime]["weighted_label_sum"])
                    / float(state[regime]["n_eff"])
                    if float(state[regime]["n_eff"]) > 0.0
                    else None
                ),
            }
            for regime in REGIME_NAMES
        },
    }


def _stage_targets(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    administrative_start: pd.Timestamp,
) -> tuple[dict[str, pd.Series], dict[str, Any]]:
    """Build the one continuous 2005 account and validate causal core output."""

    if pd.Timestamp(administrative_start) != DEVELOPMENT_START:
        raise BinaryRegimeUnionSelectorExperimentError(
            "The selector account must begin exactly once on 2005-01-01"
        )
    data = canonical_context_frame(frame)
    value = _canonical_forecast(data, forecast)
    contextual = _bool_column(value, "contextual_virtual_signal")
    weak_trend = _bool_column(value, "weak_trend_virtual_signal")
    raw = _bool_column(value, "union_candidate_signal")
    if not raw.equals(contextual | weak_trend):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Raw union candidate differs from the exact fixed-expert union"
        )

    spy20 = pd.to_numeric(value["spy_return_20"], errors="coerce")
    qqq20 = pd.to_numeric(value["qqq_return_20"], errors="coerce")
    expected_spy20 = data["spy_adj_close"] / data["spy_adj_close"].shift(20) - 1.0
    expected_qqq20 = data["qqq_adj_close"] / data["qqq_adj_close"].shift(20) - 1.0
    for name, actual, expected in (
        ("SPY", spy20, expected_spy20),
        ("QQQ", qqq20, expected_qqq20),
    ):
        mask = expected.notna()
        if not np.allclose(
            actual.loc[mask].to_numpy(dtype=float),
            expected.loc[mask].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
        ) or not actual.isna().equals(expected.isna()):
            raise BinaryRegimeUnionSelectorExperimentError(
                f"{name} 20-session return is not point-in-time exact"
            )
    regime_ready = _bool_column(value, "risk_regime_ready")
    expected_ready = spy20.notna() & qqq20.notna()
    if not regime_ready.equals(expected_ready):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Risk-regime readiness differs from exact lookback availability"
        )
    risk_on = _bool_column(value, "risk_on")
    expected_risk_on = regime_ready & (spy20 > 0.0) & (qqq20 > 0.0)
    if not risk_on.equals(expected_risk_on):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Risk-on regime differs from the frozen strict conjunction"
        )
    if bool((raw & ~regime_ready).any()):
        raise BinaryRegimeUnionSelectorExperimentError(
            "A raw union candidate lacks finite regime features"
        )

    shadow_union = _bool_column(value, "shadow_canonical_union_signal")
    canonical_alias = _bool_column(value, "canonical_union_cash_signal")
    expected_shadow = canonicalize_one_session_signals(raw)
    if not shadow_union.equals(expected_shadow) or not canonical_alias.equals(
        shadow_union
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Continuous 2000+ shadow union differs from its exact cooldown"
        )

    replay_integrity = _validate_regime_state_replay(value)
    cash_prediction = _bool_column(value, "selector_cash_prediction")
    skip_prediction = _bool_column(value, "selector_skip_prediction")
    if bool((cash_prediction & ~raw).any()) or bool(
        (skip_prediction & ~raw).any()
    ) or not (cash_prediction | skip_prediction).equals(raw) or bool(
        (cash_prediction & skip_prediction).any()
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Selector predictions are not an exact partition of raw candidates"
        )
    selected_cash = pd.Series(False, index=data.index, dtype=bool)
    selected_n_eff = pd.Series(np.nan, index=data.index, dtype=float)
    selected_mean = pd.Series(np.nan, index=data.index, dtype=float)
    selected_ready = pd.Series(False, index=data.index, dtype=bool)
    for regime, regime_mask in (
        ("risk_on", raw & risk_on),
        ("not_risk_on", raw & ~risk_on),
    ):
        selected_cash.loc[regime_mask] = _bool_column(
            value, _state_column(regime, "cash_selected")
        ).loc[regime_mask]
        selected_n_eff.loc[regime_mask] = pd.to_numeric(
            value[_state_column(regime, "n_eff")], errors="raise"
        ).loc[regime_mask]
        selected_mean.loc[regime_mask] = pd.to_numeric(
            value[_state_column(regime, "mean")], errors="coerce"
        ).loc[regime_mask]
        selected_ready.loc[regime_mask] = _bool_column(
            value, _state_column(regime, "ready")
        ).loc[regime_mask]
    if not cash_prediction.equals(raw & selected_cash):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Selector cash prediction differs from its regime latch"
        )
    if not skip_prediction.equals(raw & ~selected_cash):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Selector skip prediction differs from its regime latch"
        )
    raw_n_eff = pd.to_numeric(
        value.loc[raw, "selector_regime_n_eff"], errors="raise"
    )
    raw_mean = pd.to_numeric(
        value.loc[raw, "selector_regime_mean"], errors="coerce"
    )
    if not np.isfinite(raw_n_eff.to_numpy(dtype=float)).all() or bool(
        raw_mean.loc[raw_n_eff > 0.0].isna().any()
    ) or bool(raw_mean.loc[raw_n_eff == 0.0].notna().any()):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Raw selector diagnostics violate empty-state mean semantics"
        )
    if not np.allclose(
        raw_n_eff.to_numpy(dtype=float),
        selected_n_eff.loc[raw].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        raw_mean.to_numpy(dtype=float),
        selected_mean.loc[raw].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Selector diagnostics differ from the selected regime state"
        )
    selector_ready = _bool_column(value, "selector_regime_ready")
    if not selector_ready.equals(raw & selected_ready):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Selector readiness differs from the selected regime state"
        )
    continuous_skip = _bool_column(value, "selector_skip_signal")
    continuous_learner = _bool_column(value, "learner_cash_signal")
    if not continuous_skip.equals(shadow_union & skip_prediction):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Continuous skip stream violates shadow-union-first ordering"
        )
    if not continuous_learner.equals(shadow_union & cash_prediction):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Continuous learner stream violates the pure-filter identity"
        )
    target = pd.to_numeric(value["target_exposure"], errors="raise").astype(float)
    expected_target = pd.Series(
        np.where(continuous_learner, 0.0, 1.0), index=data.index, dtype=float
    )
    if not target.equals(expected_target):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Core target differs from the continuous learner signal"
        )

    # Validate one shadow lesson per continuous canonical opportunity.
    maturity_dates = pd.to_datetime(value["shadow_matures_on_close"], errors="coerce")
    pending = _bool_column(value, "shadow_pending")
    matured_now = _bool_column(value, "shadow_matured_now")
    signal_close = pd.to_datetime(value["shadow_signal_close"], errors="coerce")
    labels = pd.to_numeric(value["shadow_label_10bps"], errors="coerce")
    lesson_added = _bool_column(value, "shadow_lesson_added_now")
    opportunity_regime = value["shadow_opportunity_risk_on"]
    matured_regime = value["shadow_matured_signal_risk_on"]
    expected_maturity_dates = pd.Series(
        pd.NaT, index=data.index, dtype="datetime64[ns]"
    )
    resolved_positions = np.flatnonzero(shadow_union.to_numpy(dtype=bool))
    for position in resolved_positions:
        if position + 2 < len(data):
            expected_maturity_dates.iloc[position] = data.index[position + 2]
    if not maturity_dates.equals(expected_maturity_dates):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Shadow maturity schedule differs from signal close plus two sessions"
        )
    available = pd.Series(False, index=data.index, dtype=bool)
    if len(data) > 2:
        available.iloc[:-2] = True
    expected_pending = shadow_union & ~available
    if not pending.equals(expected_pending):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Pending lessons differ from unresolved shadow opportunities"
        )
    expected_matured = shadow_union.shift(2, fill_value=False).astype(bool)
    if not matured_now.equals(expected_matured):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Shadow lessons do not mature exactly two sessions later"
        )
    expected_signal_close = pd.Series(
        pd.NaT, index=data.index, dtype="datetime64[ns]"
    )
    matured_positions = np.flatnonzero(expected_matured.to_numpy(dtype=bool))
    for position in matured_positions:
        expected_signal_close.iloc[position] = data.index[position - 2]
    if not signal_close.equals(expected_signal_close):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Matured lessons identify the wrong signal close"
        )
    expected_opportunity_regime = shadow_union & risk_on
    if not opportunity_regime.astype(bool).equals(expected_opportunity_regime):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Shadow opportunity stored the wrong signal-time regime"
        )
    expected_matured_regime = pd.Series(False, index=data.index, dtype=bool)
    for position in matured_positions:
        expected_matured_regime.iloc[position] = bool(risk_on.iloc[position - 2])
    if not matured_regime.astype(bool).equals(expected_matured_regime):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Matured lesson restored the wrong signal-time regime"
        )
    if not labels.notna().equals(expected_matured) or not np.isfinite(
        labels.loc[expected_matured].to_numpy(dtype=float)
    ).all():
        raise BinaryRegimeUnionSelectorExperimentError(
            "Shadow labels are not finite exactly on maturity closes"
        )
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    for position in matured_positions:
        expected_label = math.log(opens[position - 1] / opens[position]) + math.log(
            0.999 / 1.001
        )
        if not math.isclose(
            float(labels.iloc[position]), expected_label, rel_tol=0.0, abs_tol=1e-15
        ):
            raise BinaryRegimeUnionSelectorExperimentError(
                "Shadow label differs from the frozen 10-bps formula"
            )
    expected_added = expected_matured & (
        signal_close >= pd.Timestamp("2000-01-01")
    )
    if value.attrs["learning_mode"] == FROZEN_CUTOFF_MODE:
        expected_added &= data.index <= pd.Timestamp(value.attrs["frozen_cutoff"])
    if not lesson_added.equals(expected_added):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Lesson admission violates maturity, memory start, or frozen cutoff"
        )

    # The account removes all pre-2005 candidates, applies cooldown once, and
    # only then masks unresolved tail rows.  Never canonicalize after masking.
    account_candidates = raw.copy()
    account_candidates.loc[account_candidates.index < DEVELOPMENT_START] = False
    account_union_unmasked = canonicalize_one_session_signals(account_candidates)
    account_union = account_union_unmasked & available
    account_selector = account_union & cash_prediction
    account_veto = account_union & skip_prediction
    if not account_selector.equals(account_union & ~account_veto):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Account selector violates the pure union-filter identity"
        )
    union_target = pd.Series(
        np.where(account_union, 0.0, 1.0),
        index=data.index,
        name="union_target_exposure",
        dtype=float,
    )
    selector_target = pd.Series(
        np.where(account_selector, 0.0, 1.0),
        index=data.index,
        name="selector_target_exposure",
        dtype=float,
    )
    always_long = pd.Series(
        1.0, index=data.index, name="always_long_target_exposure", dtype=float
    )
    action_frame = pd.DataFrame(
        {
            "decision_date": data.index,
            "selector_target": selector_target,
            "union_target": union_target,
        }
    )
    shadow_count = int(shadow_union.sum())
    shadow_records = int(maturity_dates.notna().sum() + pending.sum())
    if shadow_records != shadow_count or bool((maturity_dates.notna() & pending).any()):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Shadow opportunities do not map one-to-one to records"
        )
    integrity = {
        "passed": True,
        "continuous_account_start": DEVELOPMENT_START.date().isoformat(),
        "account_cooldown_reset_count_after_inception": 0,
        "account_union_derived_from_raw_after_pre_inception_removal": True,
        "unresolved_tail_masked_after_cooldown": True,
        "selector_cash_subset_of_union": True,
        "pure_union_filter_identity": True,
        "frozen_binary_regime_rule_exact": True,
        "selector_state_replay_exact": replay_integrity[
            "regime_state_replay_exact"
        ],
        "regime_switch_counts": replay_integrity["regime_switch_counts"],
        "final_replayed_states": replay_integrity["final_replayed_states"],
        "continuous_shadow_opportunity_count": shadow_count,
        "continuous_shadow_record_count": shadow_records,
        "continuous_shadow_maturity_count": int(expected_matured.sum()),
        "shadow_count_equality": True,
        "shadow_maturity_and_label_formula_exact": True,
        "causal_lesson_admission_exact": True,
        "account_union_cash_count": int(account_union.sum()),
        "account_selector_cash_count": int(account_selector.sum()),
        "account_veto_count": int(account_veto.sum()),
        "action_stream_sha256": _sha256(_frame_csv_bytes(action_frame)),
        "same_action_stream_all_costs": True,
    }
    return {
        "selector": selector_target,
        "union": union_target,
        "always_long": always_long,
    }, integrity


def _continuous_policy_result(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    periods: Sequence[EvaluationPeriod],
    cost_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate from 2005 once and attribute report edges by entry open."""

    if not periods:
        raise BinaryRegimeUnionSelectorExperimentError("No evaluation periods")
    data = canonical_context_frame(frame)
    report_start = pd.Timestamp(periods[0].start)
    report_end = pd.Timestamp(periods[-1].end)
    if report_start < DEVELOPMENT_START or report_end < report_start:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Invalid continuous-account reporting window"
        )
    account_end = data.index.max()
    if report_end < account_end and report_end != DEVELOPMENT_END:
        # Development's nominal 2018-12-31 is its physical end. Other stage
        # aggregate windows are likewise expected to reach their snapshot end.
        raise BinaryRegimeUnionSelectorExperimentError(
            "Reporting periods do not reach the physical account end"
        )
    account_period = EvaluationPeriod(
        "continuous_account",
        DEVELOPMENT_START.date().isoformat(),
        account_end.date().isoformat(),
    )
    values = pd.to_numeric(target.reindex(data.index), errors="raise").astype(float)
    values.loc[values.index < DEVELOPMENT_START] = 1.0
    benchmark_target = pd.Series(1.0, index=data.index, dtype=float)
    costs = CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0)
    strategy = simulate_unleveraged_period(
        data, values, account_period, costs, initial_cash=INITIAL_CASH
    )
    benchmark = simulate_unleveraged_period(
        data, benchmark_target, account_period, costs, initial_cash=INITIAL_CASH
    )
    comparison = compare_ledgers(strategy, benchmark, initial_cash=INITIAL_CASH)
    all_episodes = _episode_rows(
        data,
        values,
        start=DEVELOPMENT_START,
        end=account_end,
        cost_bps=cost_bps,
    )
    full_episode_edges = np.asarray(
        [float(row["net_active_log_edge"]) for row in all_episodes], dtype=float
    )
    continuous_edge = float(
        math.log1p(comparison["strategy"]["total_return"])
        - math.log1p(comparison["aapl_buy_hold"]["total_return"])
    )
    full_identity_error = continuous_edge - float(np.sum(full_episode_edges))
    if abs(full_identity_error) > 1e-10:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Continuous-account ledger edge does not reconcile to all episodes"
        )
    report_episodes = [
        row
        for row in all_episodes
        if report_start <= pd.Timestamp(row["entry_date"]) <= report_end
    ]
    report_edges = np.asarray(
        [float(row["net_active_log_edge"]) for row in report_episodes], dtype=float
    )
    report_total = float(np.sum(report_edges))
    reporting_period = EvaluationPeriod(
        "reporting_window",
        report_start.date().isoformat(),
        report_end.date().isoformat(),
    )
    strategy_report_return = _period_return(
        strategy, reporting_period, initial_cash=INITIAL_CASH
    )
    benchmark_report_return = _period_return(
        benchmark, reporting_period, initial_cash=INITIAL_CASH
    )
    reporting_boundary_edge = float(
        math.log1p(strategy_report_return) - math.log1p(benchmark_report_return)
    )
    period_results: dict[str, Any] = {}
    for period in periods:
        strategy_return = _period_return(
            strategy, period, initial_cash=INITIAL_CASH
        )
        benchmark_return = _period_return(
            benchmark, period, initial_cash=INITIAL_CASH
        )
        boundary_edge = float(
            math.log1p(strategy_return) - math.log1p(benchmark_return)
        )
        attributed_edge = float(
            np.sum(
                [
                    float(row["net_active_log_edge"])
                    for row in all_episodes
                    if pd.Timestamp(period.start)
                    <= pd.Timestamp(row["entry_date"])
                    <= pd.Timestamp(period.end)
                ]
            )
        )
        period_results[period.name] = {
            "strategy_return": strategy_return,
            "aapl_buy_hold_return": benchmark_return,
            "active_log_edge": attributed_edge,
            "ledger_boundary_active_log_edge": boundary_edge,
            "active_edge_attribution": "cash_episode_entry_open_date",
        }
    result = {
        "cost_bps_per_changing_leg": float(cost_bps),
        "account_period": asdict(account_period),
        "reporting_period": asdict(reporting_period),
        "account_inception_reset_count": 0,
        "comparison": comparison,
        "continuous_account_active_log_edge": continuous_edge,
        "continuous_account_attributed_episode_active_log_edge": float(
            np.sum(full_episode_edges)
        ),
        "continuous_account_episode_ledger_identity_error": full_identity_error,
        # This is the gate-bearing stage-window measure. It is intentionally
        # entry-attributed, so a year-boundary episode belongs to one period.
        "total_active_log_edge": report_total,
        "attributed_episode_active_log_edge": report_total,
        "reporting_window_ledger_boundary_active_log_edge": reporting_boundary_edge,
        "reporting_window_strategy_return": strategy_report_return,
        "reporting_window_aapl_buy_hold_return": benchmark_report_return,
        "periods": period_results,
        "cash_episode_count": int(len(report_episodes)),
        "cash_episode_win_rate": (
            float(np.mean(report_edges > 0.0)) if len(report_edges) else None
        ),
        "mean_cash_episode_edge": (
            float(np.mean(report_edges)) if len(report_edges) else None
        ),
        "median_cash_episode_edge": (
            float(np.median(report_edges)) if len(report_edges) else None
        ),
        "maximum_positive_episode_share": (
            float(
                np.max(report_edges[report_edges > 0.0])
                / np.sum(report_edges[report_edges > 0.0])
            )
            if np.any(report_edges > 0.0)
            else None
        ),
        "no_leverage_proof": assert_unleveraged_ledger(strategy),
    }
    return result, strategy, benchmark, pd.DataFrame(report_episodes)


def _veto_benefit_rows(
    frame: pd.DataFrame,
    union_target: pd.Series,
    selector_target: pd.Series,
    forecast: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
) -> list[dict[str, Any]]:
    data = canonical_context_frame(frame)
    value = _canonical_forecast(data, forecast)
    union = pd.to_numeric(union_target.reindex(data.index), errors="raise")
    selector = pd.to_numeric(selector_target.reindex(data.index), errors="raise")
    opens = data["aapl_adj_open"].to_numpy(dtype=float)
    dates = data.index
    cost = float(cost_bps) / 10_000.0
    friction = math.log((1.0 - cost) / (1.0 + cost))
    veto_positions = np.flatnonzero(
        (union.to_numpy(dtype=float) == 0.0)
        & (selector.to_numpy(dtype=float) == 1.0)
    )
    rows: list[dict[str, Any]] = []
    for position in veto_positions:
        if position + 2 >= len(data):
            continue
        entry = dates[position + 1]
        exit_date = dates[position + 2]
        if entry < start or entry > end:
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
            "risk_on": bool(value.iloc[position]["risk_on"]),
            "spy_return_20": float(value.iloc[position]["spy_return_20"]),
            "qqq_return_20": float(value.iloc[position]["qqq_return_20"]),
            "selector_regime_n_eff": float(
                value.iloc[position]["selector_regime_n_eff"]
            ),
            "selector_regime_mean": float(
                value.iloc[position]["selector_regime_mean"]
            ),
            "selector_regime_ready": bool(
                value.iloc[position]["selector_regime_ready"]
            ),
            "selector_cash_prediction": bool(
                value.iloc[position]["selector_cash_prediction"]
            ),
            "selector_skip_prediction": bool(
                value.iloc[position]["selector_skip_prediction"]
            ),
        }
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
        "beneficial_veto_rate": (
            float(np.mean(values > 0.0)) if len(values) else None
        ),
        "mean_veto_benefit": float(np.mean(values)) if len(values) else None,
        "median_veto_benefit": float(np.median(values)) if len(values) else None,
        "maximum_positive_veto_share": (
            float(np.max(positive) / np.sum(positive)) if len(positive) else None
        ),
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
    data = canonical_context_frame(frame)
    report_start = pd.Timestamp(periods[0].start)
    report_end = pd.Timestamp(periods[-1].end)
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
            result, strategy, policy_benchmark, episodes = _continuous_policy_result(
                data, target, periods=periods, cost_bps=cost_bps
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
            elif not benchmark.equals(policy_benchmark):
                raise BinaryRegimeUnionSelectorExperimentError(
                    "Continuous benchmark ledger changed between policies"
                )
        if benchmark is None:
            raise BinaryRegimeUnionSelectorExperimentError(
                "Policy evaluation did not produce a benchmark ledger"
            )
        benchmark_copy = benchmark.copy()
        benchmark_copy.insert(0, "policy", "aapl_buy_hold")
        benchmark_copy.insert(1, "ledger_role", "benchmark")
        ledger_rows.append(benchmark_copy)

        benefit_rows = _veto_benefit_rows(
            data,
            targets["union"],
            targets["selector"],
            forecast,
            start=report_start,
            end=report_end,
            cost_bps=cost_bps,
        )
        summary = _benefit_summary(benefit_rows, periods)
        selector_result = metrics[cost_name]["selector"]
        union_result = metrics[cost_name]["union"]
        total_incremental = float(
            selector_result["total_active_log_edge"]
            - union_result["total_active_log_edge"]
        )
        identity_error = total_incremental - float(summary["total_veto_benefit"])
        if abs(identity_error) > 1e-10:
            raise BinaryRegimeUnionSelectorExperimentError(
                "Selector-minus-union report edge does not reconcile to vetoes"
            )
        incremental_periods: dict[str, float] = {}
        for period in periods:
            attributed = float(summary["periods"][period.name])
            difference = float(
                selector_result["periods"][period.name]["active_log_edge"]
                - union_result["periods"][period.name]["active_log_edge"]
            )
            if abs(difference - attributed) > 1e-10:
                raise BinaryRegimeUnionSelectorExperimentError(
                    f"Incremental period edge does not reconcile: {period.name}"
                )
            incremental_periods[period.name] = attributed

        full_benefits = _veto_benefit_rows(
            data,
            targets["union"],
            targets["selector"],
            forecast,
            start=DEVELOPMENT_START,
            end=data.index.max(),
            cost_bps=cost_bps,
        )
        full_benefit = float(
            np.sum([row["veto_benefit"] for row in full_benefits])
        )
        full_incremental = float(
            selector_result["continuous_account_active_log_edge"]
            - union_result["continuous_account_active_log_edge"]
        )
        full_identity_error = full_incremental - full_benefit
        if abs(full_identity_error) > 1e-10:
            raise BinaryRegimeUnionSelectorExperimentError(
                "Continuous selector-minus-union edge does not reconcile to vetoes"
            )
        selector_ledger = policy_ledgers["selector"]
        union_ledger = policy_ledgers["union"]
        metrics[cost_name]["selector_vs_union"] = {
            "total_active_log_edge": total_incremental,
            "continuous_account_total_active_log_edge": full_incremental,
            "continuous_account_relative_wealth": float(
                selector_ledger["equity"].iloc[-1]
                / union_ledger["equity"].iloc[-1]
                - 1.0
            ),
            "periods": incremental_periods,
            "veto_benefit": summary,
            "veto_benefit_identity_error": identity_error,
            "continuous_account_veto_benefit": full_benefit,
            "continuous_account_veto_benefit_identity_error": full_identity_error,
        }
        ledgers[cost_name] = pd.concat(ledger_rows, ignore_index=True)
        episode_frames[cost_name] = {
            "selector": policy_episodes["selector"],
            "union": policy_episodes["union"],
        }
        benefit_frames[cost_name] = pd.DataFrame(benefit_rows)

    integrity["all_policy_ledgers_unleveraged"] = all(
        metrics[cost_name][policy]["no_leverage_proof"]["passed"]
        for cost_name, _ in COST_SCENARIOS
        for policy in ("selector", "union", "always_long")
    )
    integrity["continuous_account_episode_identity"] = True
    integrity["reporting_episode_and_veto_edge_identity"] = True
    integrity["passed"] = bool(
        integrity["selector_cash_subset_of_union"]
        and integrity["pure_union_filter_identity"]
        and integrity["frozen_binary_regime_rule_exact"]
        and integrity["selector_state_replay_exact"]
        and integrity["shadow_count_equality"]
        and integrity["shadow_maturity_and_label_formula_exact"]
        and integrity["causal_lesson_admission_exact"]
        and integrity["same_action_stream_all_costs"]
        and integrity["all_policy_ledgers_unleveraged"]
        and integrity["continuous_account_episode_identity"]
        and integrity["reporting_episode_and_veto_edge_identity"]
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
            or abs(float(result["continuous_account_active_log_edge"])) > 1e-12
            or abs(
                float(result["comparison"]["relative_wealth_vs_aapl_buy_hold"])
            )
            > 1e-12
        ):
            return False
    return True


def _integrity_gates(integrity: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "integrity_passed": integrity.get("passed") is True,
        "continuous_account_starts_once_in_2005": (
            integrity.get("continuous_account_start") == "2005-01-01"
            and integrity.get("account_cooldown_reset_count_after_inception") == 0
        ),
        "account_union_cooldown_applied_once_before_selector": (
            integrity.get(
                "account_union_derived_from_raw_after_pre_inception_removal"
            )
            is True
        ),
        "unresolved_tail_masked_after_cooldown": (
            integrity.get("unresolved_tail_masked_after_cooldown") is True
        ),
        "selector_cash_subset_of_union": (
            integrity.get("selector_cash_subset_of_union") is True
        ),
        "pure_union_filter_identity": (
            integrity.get("pure_union_filter_identity") is True
        ),
        "frozen_binary_regime_rule_exact": (
            integrity.get("frozen_binary_regime_rule_exact") is True
        ),
        "selector_state_replay_exact": (
            integrity.get("selector_state_replay_exact") is True
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
        "continuous_account_episode_identity": (
            integrity.get("continuous_account_episode_identity") is True
        ),
        "reporting_episode_and_veto_edge_identity": (
            integrity.get("reporting_episode_and_veto_edge_identity") is True
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
    final_states = integrity.get("final_replayed_states", {})
    for regime, reference in CALIBRATION_REFERENCE.items():
        state = final_states.get(regime, {}) if isinstance(final_states, dict) else {}
        gates.update(
            {
                f"{regime}_matured_lesson_count_reference": int(
                    state.get("n_raw", -1)
                )
                == int(reference["n_raw"]),
                f"{regime}_effective_count_reference": math.isclose(
                    float(state.get("n_eff", math.nan)),
                    float(reference["n_eff"]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ),
                f"{regime}_discounted_mean_reference": math.isclose(
                    float(state.get("mean", math.nan)),
                    float(reference["mean"]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ),
                f"{regime}_effective_count_ready": float(
                    state.get("n_eff", -math.inf)
                )
                >= MIN_EFFECTIVE_LESSONS,
                f"{regime}_end_latch_reference": (
                    ("CASH" if state.get("cash_selected") is True else "LONG")
                    == reference["latch"]
                ),
            }
        )

    switch_counts = integrity.get("regime_switch_counts", {})
    if not isinstance(switch_counts, Mapping):
        switch_counts = {}
    gates.update(
        {
            "training_account_exactly_twenty_vetoes": int(
                integrity.get("account_veto_count", -1)
            )
            == 20,
            "training_account_exactly_101_selector_cash_actions": int(
                integrity.get("account_selector_cash_count", -1)
            )
            == 101,
            "risk_on_latch_never_switched_in_training": int(
                switch_counts.get("risk_on", -1)
            )
            == 0,
            "not_risk_on_latch_never_switched_in_training": int(
                switch_counts.get("not_risk_on", -1)
            )
            == 0,
        }
    )

    for cost_name, _ in COST_SCENARIOS:
        selector = metrics[cost_name]["selector"]
        union = metrics[cost_name]["union"]
        incremental = metrics[cost_name]["selector_vs_union"]
        reference = UNION_REFERENCE[cost_name]
        prefix = f"{cost_name}_"
        gates.update(
            {
                prefix + "union_reference_episode_count": int(
                    union["cash_episode_count"]
                )
                == int(reference["episodes"]),
                prefix + "union_reference_active_log_edge": math.isclose(
                    float(union["total_active_log_edge"]),
                    float(reference["total_active_log_edge"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                prefix + "selector_training_edge_reference": math.isclose(
                    float(selector["total_active_log_edge"]),
                    float(SELECTOR_REFERENCE[cost_name]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                prefix + "selector_beats_union_by_more_than_0001": float(
                    incremental["total_active_log_edge"]
                )
                > STRICT_UNION_IMPROVEMENT,
            }
        )

    stress_incremental = metrics["stress_10bps"]["selector_vs_union"]
    benefit = stress_incremental["veto_benefit"]
    incremental_folds = [
        float(stress_incremental["periods"][f"{start}_{start + 1}"])
        for start in range(2005, 2019, 2)
    ]
    fold_concentration = _positive_concentration(incremental_folds)
    gates.update(
        {
            "minimum_eight_vetoes": int(benefit["veto_count"]) >= 8,
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
    report = _gate_report(gates)
    report["evidence_role"] = "calibration_training_sanity_only"
    report["development_performance_is_holdout_evidence"] = False
    return report


def apply_validation_gates(
    online_metrics: Mapping[str, Any], integrity: Mapping[str, Any]
) -> dict[str, Any]:
    gates: dict[str, bool] = {
        "online_always_long_matches_buy_hold": _always_long_gate(online_metrics),
        **_integrity_gates(integrity),
    }
    for cost_name, _ in COST_SCENARIOS:
        selector = online_metrics[cost_name]["selector"]
        incremental = online_metrics[cost_name]["selector_vs_union"]
        annual = _annual_values(selector, 2019, 2023)
        negative_aapl_years = [
            str(year)
            for year in range(2019, 2024)
            if float(selector["periods"][str(year)]["aapl_buy_hold_return"]) < 0.0
        ]
        negative_aapl_edge = float(
            np.sum(
                [
                    selector["periods"][year]["active_log_edge"]
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
                prefix + "aggregate_active_log_edge_above_001": float(
                    selector["total_active_log_edge"]
                )
                > FINAL_MATERIAL_ACTIVE_LOG_EDGE,
                prefix + "minimum_three_positive_years": int(
                    np.count_nonzero(np.asarray(annual) > 0.0)
                )
                >= 3,
                prefix + "positive_after_removing_best_validation_year": float(
                    np.sum(annual) - np.max(annual)
                )
                > 0.0,
                prefix + "minimum_three_cash_episodes": int(
                    selector["cash_episode_count"]
                )
                >= 3,
                prefix + "positive_mean_episode_edge": (
                    selector["mean_cash_episode_edge"] is not None
                    and float(selector["mean_cash_episode_edge"]) > 0.0
                ),
                prefix + "positive_median_episode_edge": (
                    selector["median_cash_episode_edge"] is not None
                    and float(selector["median_cash_episode_edge"]) > 0.0
                ),
                prefix + "episode_positive_edge_not_concentrated": (
                    selector["maximum_positive_episode_share"] is not None
                    and float(selector["maximum_positive_episode_share"]) <= 0.50
                ),
                prefix + "positive_negative_aapl_year_aggregate": (
                    not negative_aapl_years or negative_aapl_edge > 0.0
                ),
                prefix + "selector_beats_union_by_more_than_0001": float(
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
    benefit = online_metrics["stress_10bps"]["selector_vs_union"]["veto_benefit"]
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
    report["primary_evidence"] = "causal_online_selector_level_holdout"
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
        selector = online_metrics[cost_name]["selector"]
        incremental = online_metrics[cost_name]["selector_vs_union"]
        for period_name in ("2024", "2025", "2026_ytd"):
            gates[f"{cost_name}_{period_name}_active_log_edge_above_001"] = (
                float(selector["periods"][period_name]["active_log_edge"])
                > FINAL_MATERIAL_ACTIVE_LOG_EDGE
            )
        incremental_periods = [
            float(incremental["periods"][name])
            for name in ("2024", "2025", "2026_ytd")
        ]
        gates[
            f"{cost_name}_combined_selector_beats_union_by_more_than_0001"
        ] = float(incremental["total_active_log_edge"]) > STRICT_UNION_IMPROVEMENT
        gates[
            f"{cost_name}_minimum_two_nonnegative_incremental_periods"
        ] = int(np.count_nonzero(np.asarray(incremental_periods) >= 0.0)) >= 2
    report = _gate_report(gates)
    report["primary_evidence"] = "causal_online_repeated_historical_audit"
    report["frozen_or_lifetime_diagnostic_can_rescue"] = False
    return report


def _lifetime_diagnostic(
    metrics: Mapping[str, Any], integrity: Mapping[str, Any]
) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "regime_switch_counts": dict(integrity.get("regime_switch_counts", {})),
        "strict_final_gate_can_be_rescued": False,
    }
    names = [str(year) for year in range(2005, 2026)] + ["2026_ytd"]
    for cost_name, _ in COST_SCENARIOS:
        selector = metrics[cost_name]["selector"]
        annual = [
            float(selector["periods"][name]["active_log_edge"]) for name in names
        ]
        negative_aapl = [
            name
            for name in names
            if float(selector["periods"][name]["aapl_buy_hold_return"]) < 0.0
        ]
        comparison = selector["comparison"]
        diagnostic[cost_name] = {
            "continuous_relative_wealth_vs_aapl_buy_hold": float(
                comparison["relative_wealth_vs_aapl_buy_hold"]
            ),
            "continuous_active_log_edge": float(
                selector["continuous_account_active_log_edge"]
            ),
            "positive_periods": int(np.count_nonzero(np.asarray(annual) > 0.0)),
            "negative_periods": int(np.count_nonzero(np.asarray(annual) < 0.0)),
            "zero_periods": int(np.count_nonzero(np.asarray(annual) == 0.0)),
            "negative_aapl_periods": negative_aapl,
            "negative_aapl_period_aggregate_active_log_edge": float(
                np.sum(
                    [
                        selector["periods"][name]["active_log_edge"]
                        for name in negative_aapl
                    ]
                )
            ),
            "cash_episode_count": int(selector["cash_episode_count"]),
            "cash_episode_win_rate": selector["cash_episode_win_rate"],
            "mean_cash_episode_edge": selector["mean_cash_episode_edge"],
            "median_cash_episode_edge": selector["median_cash_episode_edge"],
            "maximum_positive_episode_share": selector[
                "maximum_positive_episode_share"
            ],
            "strategy_max_drawdown": float(comparison["strategy"]["max_drawdown"]),
            "aapl_buy_hold_max_drawdown": float(
                comparison["aapl_buy_hold"]["max_drawdown"]
            ),
            "max_drawdown_difference": float(comparison["max_drawdown_difference"]),
        }
    return diagnostic


MODEL_CONSTANTS = {
    "per_lesson_discount_rho": LESSON_DISCOUNT,
    "minimum_effective_lessons": MIN_EFFECTIVE_LESSONS,
    "positive_mean_threshold": POSITIVE_MEAN_THRESHOLD,
    "negative_mean_threshold": NEGATIVE_MEAN_THRESHOLD,
    "risk_on_structural_default": "LONG",
    "not_risk_on_structural_default": "CASH",
    "lesson_cost_bps_per_changing_leg": 10.0,
    "lesson_start": "2000-01-01",
    "account_start": "2005-01-01",
}


def _checkpoint_from_forecast(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    learning_mode: str,
) -> dict[str, Any]:
    data = canonical_context_frame(frame)
    index = pd.DatetimeIndex(pd.to_datetime(forecast.index, errors="raise"))
    if index.tz is not None:
        index = index.tz_localize(None)
    if not index.equals(data.index):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Checkpoint forecast is not aligned to its physical market frame"
        )
    if index.max() > cutoff:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Checkpoint forecast must be physically bounded at its cutoff"
        )
    eligible = forecast.copy()
    eligible.index = index
    eligible = eligible.loc[:cutoff]
    if eligible.empty:
        raise BinaryRegimeUnionSelectorExperimentError("Checkpoint has no rows")
    missing = sorted(_required_forecast_columns().difference(eligible.columns))
    if missing:
        raise BinaryRegimeUnionSelectorExperimentError(
            f"Checkpoint forecast lacks required columns: {missing}"
        )
    attrs = forecast.attrs
    if attrs.get("learning_mode") != learning_mode:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Checkpoint learning mode differs from the forecast"
        )
    serialized_states = attrs.get("final_states")
    serialized_pending = attrs.get("pending_lessons")
    if not isinstance(serialized_states, dict) or not isinstance(
        serialized_pending, dict
    ):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Checkpoint lacks strict serialized state"
        )
    try:
        restored_states = restore_regime_states(serialized_states)
        restored_pending = restore_pending_regime_lessons(serialized_pending)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Serialized checkpoint state is malformed"
        ) from exc
    row = eligible.iloc[-1]
    sufficient_states: dict[str, dict[str, Any]] = {}
    for regime in REGIME_NAMES:
        state = restored_states[regime]
        expected = {
            "n_raw": int(state.n_raw),
            "n_eff": float(state.n_eff),
            "weighted_label_sum": float(state.weighted_label_sum),
            "weighted_squared_label_sum": float(
                state.weighted_squared_label_sum
            ),
            "mean": float(state.mean) if state.n_eff > 0.0 else None,
            "ready": bool(state.ready),
            "cash_selected": bool(state.cash_selected),
        }
        for field in (
            "n_raw",
            "n_eff",
            "weighted_label_sum",
            "weighted_squared_label_sum",
        ):
            actual = float(row[_state_column(regime, field)])
            if not math.isclose(
                actual, float(expected[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise BinaryRegimeUnionSelectorExperimentError(
                    f"Serialized {regime} state disagrees with final row: {field}"
                )
        actual_mean = float(row[_state_column(regime, "mean")])
        if expected["mean"] is None:
            if not math.isnan(actual_mean):
                raise BinaryRegimeUnionSelectorExperimentError(
                    f"Serialized {regime} empty mean disagrees with final row"
                )
        elif not math.isclose(
            actual_mean, float(expected["mean"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise BinaryRegimeUnionSelectorExperimentError(
                f"Serialized {regime} mean disagrees with final row"
            )
        if bool(row[_state_column(regime, "ready")]) != expected["ready"] or bool(
            row[_state_column(regime, "cash_selected")]
        ) != expected["cash_selected"]:
            raise BinaryRegimeUnionSelectorExperimentError(
                f"Serialized {regime} latch/readiness disagrees with final row"
            )
        sufficient_states[regime] = expected

    pending_rows = eligible.loc[_bool_column(eligible, "shadow_pending")]
    if len(restored_pending) != len(pending_rows):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Serialized pending lessons differ from unresolved shadow rows"
        )
    for lesson, (signal_date, pending_row) in zip(
        restored_pending, pending_rows.iterrows(), strict=True
    ):
        if pd.Timestamp(lesson.signal_close) != signal_date or bool(
            lesson.risk_on
        ) != bool(pending_row["shadow_opportunity_risk_on"]):
            raise BinaryRegimeUnionSelectorExperimentError(
                "Serialized pending lesson disagrees with its shadow row"
            )
        position = int(eligible.index.get_loc(signal_date))
        expected_remaining = int(position + 2 - (len(eligible) - 1))
        expected_entry = (
            float(data.iloc[position + 1]["aapl_adj_open"])
            if position + 1 < len(data)
            else None
        )
        expected_eligible = bool(signal_date >= pd.Timestamp("2000-01-01"))
        entry_matches = (
            lesson.entry_adjusted_open is None
            if expected_entry is None
            else lesson.entry_adjusted_open is not None
            and math.isclose(
                float(lesson.entry_adjusted_open),
                expected_entry,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
        if (
            int(lesson.sessions_until_maturity) != expected_remaining
            or not entry_matches
            or bool(lesson.learning_eligible) != expected_eligible
        ):
            raise BinaryRegimeUnionSelectorExperimentError(
                "Serialized pending lesson disagrees with its physical tail"
            )

    raw = _bool_column(eligible, "union_candidate_signal")
    account_candidates = raw.copy()
    account_candidates.loc[account_candidates.index < DEVELOPMENT_START] = False
    account_union = canonicalize_one_session_signals(account_candidates)
    trailing: list[dict[str, Any]] = []
    for decision_date, trailing_row in eligible.tail(2).iterrows():
        trailing.append(
            {
                "decision_date": decision_date.date().isoformat(),
                "union_candidate_signal": bool(
                    trailing_row["union_candidate_signal"]
                ),
                "account_union_cash_signal_before_tail_mask": bool(
                    account_union.loc[decision_date]
                ),
                "risk_on": bool(trailing_row["risk_on"]),
                "selector_cash_prediction": bool(
                    trailing_row["selector_cash_prediction"]
                ),
            }
        )
    pending_summary = [
        {
            "signal_close": pd.Timestamp(lesson.signal_close).date().isoformat(),
            "sessions_until_maturity": int(lesson.sessions_until_maturity),
            "risk_on": bool(lesson.risk_on),
            "entry_adjusted_open": lesson.entry_adjusted_open,
            "learning_eligible": bool(lesson.learning_eligible),
        }
        for lesson in restored_pending
    ]
    return {
        "contract_version": CONTRACT_VERSION,
        "learning_mode": learning_mode,
        "checkpoint_cutoff": cutoff.date().isoformat(),
        "last_observed_session": eligible.index[-1].date().isoformat(),
        "account_start": DEVELOPMENT_START.date().isoformat(),
        "account_reset_count_after_inception": 0,
        "regime_feature_order": list(REGIME_FEATURE_COLUMNS),
        "model_constants": dict(MODEL_CONSTANTS),
        "serialized_regime_states": serialized_states,
        "serialized_pending_shadow_lessons": serialized_pending,
        "sufficient_states": sufficient_states,
        "account_trailing_cooldown_context": trailing,
        "pending_shadow_opportunities": pending_summary,
    }


def _require_checkpoint_continuity(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    parent_manifest_path: Path,
    checkpoint_filename: str,
) -> None:
    checkpoint_path = parent_manifest_path.resolve().parent / checkpoint_filename
    parent = _json_object(checkpoint_path, description="Committed parent checkpoint")
    if parent.get("learning_mode") != CAUSAL_ONLINE_MODE:
        raise BinaryRegimeUnionSelectorExperimentError(
            "Parent checkpoint was not produced by causal-online learning"
        )
    current = _checkpoint_from_forecast(
        frame,
        forecast,
        cutoff=cutoff,
        learning_mode=CAUSAL_ONLINE_MODE,
    )
    if _canonical_json_bytes(parent) != _canonical_json_bytes(current):
        raise BinaryRegimeUnionSelectorExperimentError(
            "Regenerated selector/account state does not match parent checkpoint"
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
    "risk_on",
    "spy_return_20",
    "qqq_return_20",
    "selector_regime_n_eff",
    "selector_regime_mean",
    "selector_regime_ready",
    "selector_cash_prediction",
    "selector_skip_prediction",
)


def _episode_payloads(
    episodes: Mapping[str, Mapping[str, pd.DataFrame]], *, prefix: str
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for cost_name, policies in episodes.items():
        for policy in ("selector", "union"):
            value = policies[policy].copy()
            if value.empty:
                value = pd.DataFrame(columns=_EPISODE_COLUMNS)
            payloads[f"{prefix}_{cost_name}_{policy}_episodes.csv"] = (
                _frame_csv_bytes(value)
            )
    return payloads


def _benefit_payloads(
    benefits: Mapping[str, pd.DataFrame], *, prefix: str
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for cost_name, frame in benefits.items():
        value = frame.copy()
        if value.empty:
            value = pd.DataFrame(columns=_VETO_BENEFIT_COLUMNS)
        payloads[f"{prefix}_{cost_name}_veto_benefits.csv"] = _frame_csv_bytes(
            value
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
            "continuous_account_start": DEVELOPMENT_START.date().isoformat(),
            "account_resets_after_inception": 0,
            "maximum_target_exposure": 1.0,
            "selector_is_pure_union_filter": True,
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
    _require_exact_physical_stage_bound(provenance, stage="Development")
    provenance["tracked_input"] = input_identity
    deadline.check("bounded development load")
    forecast = build_binary_regime_union_selector_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    metrics, ledgers, episodes, benefits, integrity = _evaluate_policy_set(
        frame,
        forecast,
        periods=_development_periods(),
        administrative_start=DEVELOPMENT_START,
    )
    gates = apply_development_gates(metrics, integrity)
    cold_start_report = {
        "evidence_role": "causal_cold_start_training_diagnostic_only",
        "initialized_from_empty_states": True,
        "learning_mode": CAUSAL_ONLINE_MODE,
        "account_start": DEVELOPMENT_START.date().isoformat(),
        "account_reset_count_after_inception": 0,
        "account_union_cash_count": integrity["account_union_cash_count"],
        "account_selector_cash_count": integrity["account_selector_cash_count"],
        "account_veto_count": integrity["account_veto_count"],
        "regime_switch_counts": integrity["regime_switch_counts"],
        "final_replayed_states": integrity["final_replayed_states"],
        "performance_is_holdout_evidence": False,
    }
    deadline.check("development replay and gates")
    resolved_run_id = _safe_run_id(
        run_id, prefix="binary-regime-union-selector-development"
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "development",
        "run_id": resolved_run_id,
        "evidence_classification": (
            "through_2018_calibration_training_not_holdout_evidence"
        ),
        "physical_data_end": DEVELOPMENT_END.date().isoformat(),
        "later_outcomes_accessed": False,
        "continuous_account_start": DEVELOPMENT_START.date().isoformat(),
        "continuous_account_reset_count_after_inception": 0,
        "calibration_metrics": metrics,
        "cold_start_diagnostic": cold_start_report,
        "integrity": integrity,
        "gate_report": gates,
        "runtime": _runtime_report(deadline),
    }
    checkpoint = _checkpoint_from_forecast(
        frame,
        forecast,
        cutoff=DEVELOPMENT_END,
        learning_mode=CAUSAL_ONLINE_MODE,
    )
    payloads = {
        "development_prices_through_2018.csv": _frame_csv_bytes(
            frame.reset_index(names="date")
        ),
        "development_forecast_through_2018.csv": _frame_csv_bytes(
            _flatten_forecast(forecast)
        ),
        "development_metrics.json": _pretty_json_bytes(metrics),
        "development_cold_start_report.json": _pretty_json_bytes(
            cold_start_report
        ),
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
    # Parent pass is verified first.  The first market DataFrame returned from
    # the longer tracked artifact is still physically bounded at 2018.
    parent_prefix, prefix_provenance = load_bounded_prices(
        price_artifact,
        end=DEVELOPMENT_END,
        required_last_session=DEVELOPMENT_END,
    )
    prefix_provenance["tracked_input"] = input_identity
    _require_source_continuity(
        parent_prefix,
        prefix_provenance,
        parent,
        parent_end=DEVELOPMENT_END,
    )
    deadline.check("bounded development-prefix load from validation artifact")
    parent_prefix_forecast = build_binary_regime_union_selector_forecast(
        parent_prefix, learning_mode=CAUSAL_ONLINE_MODE
    )
    _require_checkpoint_continuity(
        parent_prefix,
        parent_prefix_forecast,
        cutoff=DEVELOPMENT_END,
        parent_manifest_path=development_manifest,
        checkpoint_filename="development_checkpoint_through_2018.json",
    )
    deadline.check("development checkpoint replay")
    # Only after exact prefix/checkpoint replay may a DataFrame contain 2019+.
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=VALIDATION_END,
        required_last_session=pd.Timestamp("2023-12-29"),
    )
    _require_exact_physical_stage_bound(provenance, stage="Validation")
    provenance["tracked_input"] = input_identity
    _require_source_continuity(
        frame, provenance, parent, parent_end=DEVELOPMENT_END
    )
    deadline.check("authorized validation-row load")
    online = build_binary_regime_union_selector_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    frozen = build_binary_regime_union_selector_forecast(
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
        administrative_start=DEVELOPMENT_START,
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
        administrative_start=DEVELOPMENT_START,
    )
    gates = apply_validation_gates(online_metrics, online_integrity)
    deadline.check("validation replay and gates")
    resolved_run_id = _safe_run_id(
        run_id, prefix="binary-regime-union-selector-validation"
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "validation",
        "run_id": resolved_run_id,
        "evidence_classification": (
            "untouched_2019_2023_selector_level_causal_online_validation_"
            "not_globally_pristine_expert_holdout"
        ),
        "physical_data_end": VALIDATION_END.date().isoformat(),
        "post_2023_outcomes_accessed": False,
        "continuous_account_start": DEVELOPMENT_START.date().isoformat(),
        "continuous_account_reset_count_after_inception": 0,
        "primary_causal_online_metrics": online_metrics,
        "primary_causal_online_integrity": online_integrity,
        "secondary_frozen_2018_diagnostic_metrics": frozen_metrics,
        "secondary_frozen_2018_integrity": frozen_integrity,
        "frozen_diagnostic_cannot_rescue_online_failure": True,
        "gate_report": gates,
        "runtime": _runtime_report(deadline),
    }
    checkpoint = _checkpoint_from_forecast(
        frame,
        online,
        cutoff=VALIDATION_END,
        learning_mode=CAUSAL_ONLINE_MODE,
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
    # The first DataFrame returned from the longer tracked artifact remains
    # bounded at the committed through-2023 parent checkpoint.
    parent_prefix, prefix_provenance = load_bounded_prices(
        price_artifact,
        end=VALIDATION_END,
        required_last_session=pd.Timestamp("2023-12-29"),
    )
    prefix_provenance["tracked_input"] = input_identity
    _require_source_continuity(
        parent_prefix,
        prefix_provenance,
        parent,
        parent_end=VALIDATION_END,
    )
    deadline.check("bounded validation-prefix load from final artifact")
    parent_prefix_forecast = build_binary_regime_union_selector_forecast(
        parent_prefix, learning_mode=CAUSAL_ONLINE_MODE
    )
    _require_checkpoint_continuity(
        parent_prefix,
        parent_prefix_forecast,
        cutoff=VALIDATION_END,
        parent_manifest_path=validation_manifest,
        checkpoint_filename="validation_checkpoint_through_2023.json",
    )
    deadline.check("validation checkpoint replay")
    # Only now may a DataFrame containing a 2024+ row be returned.
    frame, provenance = load_bounded_prices(
        price_artifact,
        end=FINAL_END,
        required_last_session=FINAL_END,
    )
    _require_exact_physical_stage_bound(provenance, stage="Final")
    provenance["tracked_input"] = input_identity
    _require_source_continuity(frame, provenance, parent, parent_end=VALIDATION_END)
    deadline.check("authorized final-row load")
    online = build_binary_regime_union_selector_forecast(
        frame, learning_mode=CAUSAL_ONLINE_MODE
    )
    frozen = build_binary_regime_union_selector_forecast(
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
        administrative_start=DEVELOPMENT_START,
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
        administrative_start=DEVELOPMENT_START,
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
    lifetime_diagnostic = _lifetime_diagnostic(
        lifetime_metrics, lifetime_integrity
    )
    gates = apply_final_gates(online_metrics, online_integrity)
    deadline.check("final replay and gates")
    resolved_run_id = _safe_run_id(
        run_id, prefix="binary-regime-union-selector-final"
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "stage": "final",
        "run_id": resolved_run_id,
        "evidence_classification": "repeated_2024_2026_ytd_historical_audit",
        "physical_data_end": FINAL_END.date().isoformat(),
        "continuous_account_start": DEVELOPMENT_START.date().isoformat(),
        "continuous_account_reset_count_after_inception": 0,
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
        frame,
        online,
        cutoff=FINAL_END,
        learning_mode=CAUSAL_ONLINE_MODE,
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
    "BinaryRegimeUnionSelectorExperimentError",
    "apply_development_gates",
    "apply_final_gates",
    "apply_validation_gates",
    "load_bounded_prices",
    "main",
    "run_development",
    "run_final",
    "run_validation",
]
