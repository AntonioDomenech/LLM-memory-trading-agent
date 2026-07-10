"""Sealed, offline pre-2019 one-session AAPL regime experiment.

The public entry point accepts one immutable verified price artifact and one
bounded local context file.  The in-memory entry point is intentionally useful
for tests, but applies the same physical-date refusal: even an empty post-2018
row makes a development run invalid.  There is no 2019+ evaluation entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cftc_cot_experiment import (
    _committed_artifact_snapshot,
    _git_commit_is_ancestor,
)
from .deterministic_aapl import CostAssumptions, EvaluationPeriod, file_sha256
from .direct_edge_features import (
    CONTEXT_VALUE_COLUMNS,
    canonical_market_sentiment_context,
)
from .downside_ensemble_experiment import (
    load_cftc_development_artifact as _load_verified_price_inputs,
)
from .regime_consensus_features import (
    HEAD_FEATURE_COLUMNS,
    HEAD_NAMES,
    HEAD_ORIENTATIONS,
    REGIME_FEATURE_COLUMNS,
    build_regime_consensus_feature_label_frame,
    strict_pre_fold_head_training_mask,
)
from .regime_hmm import RegimeHeadModel
from .regime_consensus_scoring import (
    evaluate_regime_consensus_ablation,
    evaluate_regime_consensus_development,
)
from .regime_consensus_walkforward import (
    MODEL_VARIANTS,
    REGIME_CANDIDATES,
    WALK_FORWARD_FOLDS,
    RegimeCandidate,
    build_pre2019_regime_consensus_walkforward,
    candidate_cash_target_column,
)
from .unleveraged_aapl import (
    _git_state,
    canonical_context_frame,
    simulate_unleveraged_period,
    validate_source_repository,
)


CONTRACT_VERSION = "aapl-one-session-regime-consensus-development-v1"
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
DEVELOPMENT_PERIOD = EvaluationPeriod(
    "regime_consensus_development", "2005-01-01", "2018-12-31"
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

# TNX is intentionally absent.  The fixed query cannot load it accidentally.
CONTEXT_PARQUET_QUERY = """
SELECT date, symbol, close, adj_close
FROM read_parquet(?)
WHERE CAST(date AS DATE) <= DATE '2018-12-31'
  AND upper(replace(trim(symbol), '^', '')) IN ('IWM', 'VIX')
ORDER BY CAST(date AS DATE), upper(replace(trim(symbol), '^', ''))
""".strip()

SOURCE_FILES = (
    ".gitattributes",
    "agent_benchmark/cftc_cot.py",
    "agent_benchmark/cftc_cot_experiment.py",
    "agent_benchmark/cftc_cot_policy.py",
    "agent_benchmark/deterministic_aapl.py",
    "agent_benchmark/direct_edge_features.py",
    "agent_benchmark/downside_ensemble_experiment.py",
    "agent_benchmark/downside_features.py",
    "agent_benchmark/downside_forest.py",
    "agent_benchmark/downside_scoring.py",
    "agent_benchmark/downside_walkforward.py",
    "agent_benchmark/regime_consensus_experiment.py",
    "agent_benchmark/regime_consensus_features.py",
    "agent_benchmark/regime_hmm.py",
    "agent_benchmark/regime_consensus_scoring.py",
    "agent_benchmark/regime_consensus_walkforward.py",
    "agent_benchmark/unleveraged_aapl.py",
    "docs/regime_consensus_experiment.md",
    "requirements.txt",
)

SOURCE_EXCLUSION_AUDITS: Mapping[str, Mapping[str, Any]] = {
    "macro": {
        "included": False,
        "finding": "stored macro values are placeholders and vintage timing is unsafe",
        "decision": "excluded before feature and candidate construction",
    },
    "news": {
        "included": False,
        "finding": "available event rows lack safe publication-availability timing",
        "decision": "excluded before feature and candidate construction",
    },
    "cftc": {
        "included": False,
        "finding": "weekly positioning is not part of the frozen one-session hypothesis",
        "decision": "no CFTC value enters a feature, fit, prediction, or gate",
    },
    "tnx": {
        "included": False,
        "finding": "the frozen three heads do not contain an interest-rate input",
        "decision": "TNX is excluded by the public query and canonical input",
    },
}


class RegimeConsensusExperimentError(RuntimeError):
    """Raised when the sealed development contract is violated."""


class RegimeConsensusExperimentTimeout(RegimeConsensusExperimentError):
    """Raised when the complete run exceeds one wall-clock hour."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate: RegimeCandidate

    def __post_init__(self) -> None:
        if self.candidate not in REGIME_CANDIDATES:
            raise ValueError("Candidate is outside the frozen two-gate grid")

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    def cash_target_column(self, variant: str) -> str:
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model variant: {variant!r}")
        return candidate_cash_target_column(variant, self.candidate_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "probability_gate": float(self.candidate.probability_gate),
            "expected_edge_gate_10bps": float(self.candidate.expected_edge_gate),
            "required_head_votes": 2,
            "cash_decision_rows": 1,
            "model_variant": "regime_consensus",
        }


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(candidate) for candidate in REGIME_CANDIDATES
)


@dataclass(frozen=True)
class LoadedDevelopmentInputs:
    price_frame: pd.DataFrame
    context_frame: pd.DataFrame
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
            raise RegimeConsensusExperimentTimeout(
                f"Development exceeded {RUN_TIME_LIMIT_SECONDS:.0f} seconds at {step}"
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite numbers are forbidden in sealed JSON")
        return parsed
    if isinstance(value, np.bool_):
        return bool(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
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


def _frame_csv_bytes(frame: pd.DataFrame, *, float_format: str = "%.17g") -> bytes:
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


def _local_artifact_snapshot(run_dir: Path) -> dict[str, bytes]:
    if not run_dir.is_dir():
        raise RegimeConsensusExperimentError(
            f"Artifact directory does not exist: {run_dir}"
        )
    if any(path.is_file() and path.parent != run_dir for path in run_dir.rglob("*")):
        raise RegimeConsensusExperimentError("Artifact must contain only flat files")
    return {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}


def _verify_checksum_snapshot(snapshot: Mapping[str, bytes]) -> dict[str, str]:
    try:
        checksums = json.loads(snapshot["checksums.json"].decode("utf-8", "strict"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegimeConsensusExperimentError("Artifact has no valid checksums") from exc
    expected = set(snapshot) - {"checksums.json"}
    if not isinstance(checksums, dict) or set(checksums) != expected:
        raise RegimeConsensusExperimentError("Checksum manifest file set is not exact")
    for name, digest in checksums.items():
        if not isinstance(digest, str) or _sha256_hex(snapshot[name]) != digest:
            raise RegimeConsensusExperimentError(f"Artifact checksum mismatch: {name}")
    return {str(key): str(value) for key, value in checksums.items()}


def _seal_artifact_bundle(run_dir: Path, payloads: Mapping[str, bytes]) -> None:
    if run_dir.exists():
        raise RegimeConsensusExperimentError(f"Artifact already exists: {run_dir}")
    if any("/" in name or "\\" in name for name in payloads):
        raise RegimeConsensusExperimentError("Artifact payload names must be flat")
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
    _verify_checksum_snapshot(_local_artifact_snapshot(run_dir))


def _normalize_input_dates(frame: pd.DataFrame, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    keyed = {
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): column
        for column in frame.columns
    }
    source = next(
        (keyed[item] for item in ("date", "datetime", "timestamp") if item in keyed),
        None,
    )
    raw = frame[source] if source is not None else frame.index
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(raw, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise RegimeConsensusExperimentError(f"{name} has invalid dates") from exc
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _bounded_inputs(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_price_dates = _normalize_input_dates(price_frame, name="price input")
    if len(raw_price_dates) and raw_price_dates.max() > DEVELOPMENT_END:
        first = raw_price_dates[raw_price_dates > DEVELOPMENT_END][0]
        raise RegimeConsensusExperimentError(
            f"Development input contains a post-2018 price row: {first.date()}"
        )
    prices = canonical_context_frame(price_frame)
    if prices.empty or prices.index.max() > DEVELOPMENT_END:
        raise RegimeConsensusExperimentError("Canonical prices are empty or post-2018")
    missing_years = set(range(2005, 2019)) - set(prices.index.year)
    if missing_years or prices.index.min().year > 2004:
        raise RegimeConsensusExperimentError(
            f"Development prices lack required history: {sorted(missing_years)}"
        )

    raw_context_dates = _normalize_input_dates(context_frame, name="context input")
    if len(raw_context_dates) and raw_context_dates.max() > DEVELOPMENT_END:
        first = raw_context_dates[raw_context_dates > DEVELOPMENT_END][0]
        raise RegimeConsensusExperimentError(
            f"Development input contains a post-2018 context row: {first.date()}"
        )
    context = canonical_market_sentiment_context(context_frame, prices.index)
    # Preserve the canonical schema while making the TNX exclusion physically exact.
    context["tnx_close"] = np.nan
    context = context.loc[:, list(CONTEXT_VALUE_COLUMNS)].astype(float)
    if context.index.max() > DEVELOPMENT_END:
        raise RegimeConsensusExperimentError("Context alignment produced post-2018 data")
    relevant = context.loc[:, ["iwm_adj_close", "vix_close"]].to_numpy(float)
    return prices, context, {
        "price_first_date": prices.index.min().date().isoformat(),
        "price_last_date": prices.index.max().date().isoformat(),
        "price_rows": int(len(prices)),
        "context_first_aligned_date": context.index.min().date().isoformat(),
        "context_last_aligned_date": context.index.max().date().isoformat(),
        "context_aligned_rows": int(len(context)),
        "iwm_vix_complete_rows": int(np.isfinite(relevant).all(axis=1).sum()),
        "iwm_vix_missing_cells": int((~np.isfinite(relevant)).sum()),
        "tnx_cells_used": 0,
    }


def load_bounded_context_parquet(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only IWM/VIX rows through 2018 from a local Parquet file."""

    source = path.resolve()
    if not source.is_file():
        raise RegimeConsensusExperimentError(f"Context Parquet does not exist: {source}")
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            frame = connection.execute(CONTEXT_PARQUET_QUERY, [str(source)]).fetchdf()
        finally:
            connection.close()
    except Exception as exc:
        raise RegimeConsensusExperimentError("Bounded context query failed") from exc
    dates = _normalize_input_dates(frame, name="bounded context result")
    if len(dates) and dates.max() > DEVELOPMENT_END:
        raise RegimeConsensusExperimentError("Bounded query returned post-2018 data")
    symbols = {
        str(value).strip().upper().replace("^", "") for value in frame["symbol"]
    }
    if not symbols.issubset({"IWM", "VIX"}):
        raise RegimeConsensusExperimentError("Bounded query returned a forbidden symbol")
    bounded_bytes = _frame_csv_bytes(frame)
    return frame, {
        "source_type": "local_context_daily_parquet_bounded_query",
        "source_path": str(source),
        "source_file_sha256": f"sha256:{file_sha256(source)}",
        "query": CONTEXT_PARQUET_QUERY,
        "query_parameters": [str(source)],
        "allowed_symbols": ["IWM", "VIX"],
        "physical_data_end": "2018-12-31",
        "bounded_result_rows": int(len(frame)),
        "bounded_result_sha256": _sha256_tagged(bounded_bytes),
        "network_access": False,
    }


def load_public_development_inputs(
    *, price_artifact: Path, context_parquet: Path, repo_root: Path
) -> LoadedDevelopmentInputs:
    try:
        price_inputs = _load_verified_price_inputs(
            price_artifact, repo_root=repo_root.resolve()
        )
    except Exception as exc:
        raise RegimeConsensusExperimentError(
            "Could not replay the committed checksum-verified price artifact"
        ) from exc
    context, provenance = load_bounded_context_parquet(context_parquet)
    return LoadedDevelopmentInputs(
        price_frame=price_inputs.price_frame,
        context_frame=context,
        provenance={"price": dict(price_inputs.provenance), "context": provenance},
    )


def _full_exposure_target(
    price_index: pd.DatetimeIndex,
    prediction_index: pd.DatetimeIndex,
    cash_target: Sequence[float],
) -> pd.Series:
    if not prediction_index.isin(price_index).all():
        raise RegimeConsensusExperimentError("Prediction dates are absent from prices")
    cash = np.asarray(cash_target, dtype=float)
    if cash.shape != (len(prediction_index),) or not np.isin(cash, (0.0, 1.0)).all():
        raise RegimeConsensusExperimentError("Candidate CASH target is not exact binary")
    target = pd.Series(1.0, index=price_index, name="target_exposure")
    target.loc[prediction_index] = 1.0 - cash
    return target


def _simulate(frame: pd.DataFrame, target: pd.Series, *, cost_bps: float) -> pd.DataFrame:
    return simulate_unleveraged_period(
        frame,
        target,
        DEVELOPMENT_PERIOD,
        CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0),
        initial_cash=INITIAL_CASH,
    )


def _prediction_inputs(
    predictions: pd.DataFrame,
    *,
    variant: str,
    cash_target: Sequence[float],
    support: Sequence[bool],
) -> tuple[np.ndarray, ...]:
    names = {
        "probability": f"{variant}_cash_win_probability_10bps",
        "edge": f"{variant}_expected_edge_10bps",
        "label": "cash_beats_long_10bps",
        "actual_edge": "cash_active_log_edge_10bps",
        "baseline_probability": "causal_prevalence_probability_10bps",
        "baseline_edge": "causal_training_mean_edge_10bps",
    }
    missing = sorted(set(names.values()) - set(predictions.columns))
    if missing:
        raise RegimeConsensusExperimentError(
            f"Walk-forward predictions lack required columns: {missing}"
        )
    selected = np.asarray(support, dtype=bool)
    cash = np.asarray(cash_target, dtype=float)
    if selected.shape != (len(predictions),) or cash.shape != selected.shape:
        raise RegimeConsensusExperimentError("Predictive support is not OOF-aligned")
    if bool((cash[~selected] != 0.0).any()):
        raise RegimeConsensusExperimentError(
            "Unavailable three-head rows must remain LONG for every variant"
        )
    return (
        pd.to_numeric(predictions[names["probability"]], errors="coerce").to_numpy(float)[selected],
        pd.to_numeric(predictions[names["label"]], errors="coerce").to_numpy(float)[selected],
        pd.to_numeric(predictions[names["baseline_probability"]], errors="coerce").to_numpy(float)[selected],
        pd.to_numeric(predictions[names["edge"]], errors="coerce").to_numpy(float)[selected],
        pd.to_numeric(predictions[names["actual_edge"]], errors="coerce").to_numpy(float)[selected],
        pd.to_numeric(predictions[names["baseline_edge"]], errors="coerce").to_numpy(float)[selected],
        cash[selected],
        predictions.index[selected],
    )


def _common_predictive_support(predictions: pd.DataFrame) -> np.ndarray:
    required = [
        "all_three_heads_ready",
        "causal_prevalence_probability_10bps",
        "causal_training_mean_edge_10bps",
    ]
    for variant in MODEL_VARIANTS:
        required.extend(
            (
                f"{variant}_cash_win_probability_10bps",
                f"{variant}_expected_edge_10bps",
            )
        )
    missing = sorted(set(required) - set(predictions.columns))
    if missing:
        raise RegimeConsensusExperimentError(
            f"Walk-forward lacks common-support columns: {missing}"
        )
    ready = predictions["all_three_heads_ready"].astype(bool).to_numpy()
    finite = np.isfinite(
        predictions.loc[:, required[1:]].to_numpy(dtype=float)
    ).all(axis=1)
    support = ready & finite
    if not support.any():
        raise RegimeConsensusExperimentError("Common predictive support is empty")
    return support


def _score_strategy(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    prediction_inputs: tuple[np.ndarray, ...],
    cost_bps: float,
) -> dict[str, Any]:
    probability, label, baseline_probability, edge, actual_edge, baseline_edge, cash, dates = prediction_inputs
    return evaluate_regime_consensus_development(
        strategy,
        benchmark,
        strategy["target_exposure"].to_numpy(dtype=float),
        probability,
        label,
        baseline_probability,
        edge,
        actual_edge,
        baseline_edge,
        cash,
        dates,
        cost_bps=cost_bps,
    )


def select_development_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> str | None:
    """Rank passing full candidates by the frozen five-key stress rule."""

    passing = [item for item in candidate_results if bool(item.get("passed"))]
    if not passing:
        return None

    def rank(item: Mapping[str, Any]) -> tuple[float, float, float, int, str]:
        metrics = item["scenarios"]["stress_10bps"]["metrics"]
        return (
            -float(metrics["minimum_fold_active_log_edge"]),
            -float(metrics["total_active_log_edge"]),
            -float(metrics["brier_relative_improvement"]),
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
        "head_names": list(HEAD_NAMES),
        "head_feature_columns": [list(item) for item in HEAD_FEATURE_COLUMNS],
        "feature_columns": list(REGIME_FEATURE_COLUMNS),
        "context_alignment": "exact AAPL session date; no fill or interpolation",
        "missing_policy": "all three heads must be ready; otherwise LONG",
        "training_policy": "seven fixed purged expanding folds; labels mature strictly before each fold",
        "excluded_sources": {
            key: dict(value) for key, value in SOURCE_EXCLUSION_AUDITS.items()
        },
    }


def _json_safe_states(value: Any) -> Any:
    # Round-trip now so a non-canonical state fails before any artifact exists.
    return json.loads(_canonical_json_bytes(value).decode("utf-8"))


def _sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("ascii")
    ).hexdigest()


def _fit_final_head(
    feature_label_frame: pd.DataFrame,
    *,
    head_name: str,
    feature_columns: Sequence[str],
    stress_orientations: Sequence[int],
    deadline: _Deadline,
) -> tuple[bytes, dict[str, Any]]:
    boundary = pd.Timestamp("2019-01-01")
    history_mask = np.asarray(feature_label_frame.index < boundary, dtype=bool)
    outcome_mask = strict_pre_fold_head_training_mask(
        feature_label_frame,
        fold_first_decision_date=boundary,
        head_name=head_name,
    ).to_numpy(dtype=bool)
    if not history_mask.any() or not outcome_mask.any():
        raise RegimeConsensusExperimentError(
            f"No causal through-2018 refit data for {head_name}"
        )
    if bool((outcome_mask & ~history_mask).any()):
        raise RegimeConsensusExperimentError("Refit outcome escaped history")
    history = feature_label_frame.loc[history_mask, list(feature_columns)]
    binary = np.full(len(history), np.nan, dtype=float)
    edge = np.full(len(history), np.nan, dtype=float)
    history_positions = np.flatnonzero(history_mask)
    eligible_positions = np.flatnonzero(outcome_mask)
    position_lookup = {
        global_position: local
        for local, global_position in enumerate(history_positions)
    }
    local = np.asarray(
        [position_lookup[int(position)] for position in eligible_positions],
        dtype=int,
    )
    binary[local] = feature_label_frame.iloc[eligible_positions][
        "cash_beats_long_10bps"
    ].to_numpy(dtype=float)
    edge[local] = feature_label_frame.iloc[eligible_positions][
        "cash_active_log_edge_10bps"
    ].to_numpy(dtype=float)
    outcomes = feature_label_frame.loc[outcome_mask]
    maturity = pd.to_datetime(outcomes["label_maturity_date"], errors="raise")

    def token(value: float) -> str:
        number = float(value)
        return number.hex() if math.isfinite(number) else "missing"

    metadata = {
        "training_start_date": history.index.min().date().isoformat(),
        "training_end_date": history.index.max().date().isoformat(),
        "maximum_label_maturity_date": maturity.max().date().isoformat(),
        "training_dates_sha256": _sha256_lines(
            [timestamp.date().isoformat() for timestamp in history.index]
        ),
        "binary_target_sha256": _sha256_lines([token(value) for value in binary]),
        "edge_target_sha256": _sha256_lines([token(value) for value in edge]),
    }

    def fit() -> RegimeHeadModel:
        return RegimeHeadModel().fit(
            history,
            binary,
            edge,
            head_name=head_name,
            feature_names=tuple(feature_columns),
            stress_orientations=tuple(stress_orientations),
            fit_metadata=metadata,
        )

    first = fit()
    deadline.check(f"first deterministic {head_name} refit")
    second = fit()
    deadline.check(f"second deterministic {head_name} refit")
    first_bytes = (first.canonical_json() + "\n").encode("utf-8")
    second_bytes = (second.canonical_json() + "\n").encode("utf-8")
    if first_bytes != second_bytes:
        raise RegimeConsensusExperimentError(
            f"Repeated {head_name} refits were not byte-identical"
        )
    return first_bytes, {
        "head_name": head_name,
        "model_sha256": first.model_sha256,
        "state_file_sha256": _sha256_tagged(first_bytes),
        "training_rows": int(len(history)),
        "outcome_training_rows": int(outcome_mask.sum()),
        "training_first_decision": history.index.min().date().isoformat(),
        "training_last_decision": history.index.max().date().isoformat(),
        "training_latest_label_maturity": maturity.max().date().isoformat(),
        "refit_repeated_byte_identically": True,
    }


def _refit_selected_models(
    feature_label_frame: pd.DataFrame,
    selected_id: str | None,
    *,
    deadline: _Deadline,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Double-fit all three frozen heads only after a candidate passes."""

    if selected_id is None:
        return {}, {"performed": False, "reason": "no_development_candidate_passed"}
    if selected_id not in {item.candidate_id for item in CANDIDATE_SPECS}:
        raise RegimeConsensusExperimentError("Selected candidate is not frozen")
    payloads: dict[str, bytes] = {}
    models: dict[str, Any] = {}
    for head_name, columns, orientations in zip(
        HEAD_NAMES, HEAD_FEATURE_COLUMNS, HEAD_ORIENTATIONS, strict=True
    ):
        payload, metadata = _fit_final_head(
            feature_label_frame,
            head_name=head_name,
            feature_columns=columns,
            stress_orientations=orientations,
            deadline=deadline,
        )
        payloads[f"final_{head_name}_model_through_2018.json"] = payload
        models[head_name] = metadata
    return payloads, {
        "performed": True,
        "selected_candidate_id": selected_id,
        "training_cutoff": "2018-12-31",
        "models": models,
    }


def _compute_development(
    prices: pd.DataFrame,
    context: pd.DataFrame,
    *,
    deadline: _Deadline,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    Any,
    pd.DataFrame,
    list[dict[str, Any]],
    dict[str, pd.DataFrame],
]:
    feature_label = build_regime_consensus_feature_label_frame(prices, context)
    if feature_label.index.max() > DEVELOPMENT_END:
        raise RegimeConsensusExperimentError("Features contain post-2018 rows")
    deadline.check("feature and label construction")
    walkforward = build_pre2019_regime_consensus_walkforward(feature_label)
    predictions = walkforward.predictions.copy()
    states = _json_safe_states(walkforward.model_states)
    if predictions.index.max() > DEVELOPMENT_END:
        raise RegimeConsensusExperimentError("Predictions contain post-2018 rows")
    if len(states) != 21:
        raise RegimeConsensusExperimentError(
            f"Expected exactly 21 frozen HMM fits, found {len(states)}"
        )
    deadline.check("twenty-one frozen HMM fits and predictions")

    target_audit = pd.DataFrame(index=predictions.index)
    target_audit.index.name = "decision_date"
    support = _common_predictive_support(predictions)
    target_audit["common_predictive_support"] = support
    targets: dict[tuple[str, str], pd.Series] = {}
    cash_by_variant: dict[tuple[str, str], np.ndarray] = {}
    for spec in CANDIDATE_SPECS:
        for variant in MODEL_VARIANTS:
            column = spec.cash_target_column(variant)
            if column not in predictions:
                raise RegimeConsensusExperimentError(
                    f"Walk-forward prediction lacks frozen target: {column}"
                )
            cash = pd.to_numeric(predictions[column], errors="coerce").to_numpy(float)
            if not np.isfinite(cash).all() or not np.isin(cash, (0.0, 1.0)).all():
                raise RegimeConsensusExperimentError(f"Target is not binary: {column}")
            cash_by_variant[(variant, spec.candidate_id)] = cash
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
                cash_target=cash_by_variant[key],
                support=support,
            )
            scenarios: dict[str, Any] = {}
            for scenario, bps in COST_SCENARIOS:
                strategy = _simulate(prices, targets[key], cost_bps=bps)
                scenarios[scenario] = _score_strategy(
                    strategy,
                    benchmarks[scenario],
                    prediction_inputs=inputs,
                    cost_bps=bps,
                )
                ledgers[f"{variant}_{spec.candidate_id}_{scenario}_strategy.csv"] = strategy
                deadline.check(f"{variant} {spec.candidate_id} {scenario} simulation")
            variant_results[key] = scenarios

    results: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        full = variant_results[("regime_consensus", spec.candidate_id)]
        aapl = variant_results[("aapl_only", spec.candidate_id)]
        ablation = evaluate_regime_consensus_ablation(
            full["base_5bps"],
            aapl["base_5bps"],
            full["stress_10bps"],
            aapl["stress_10bps"],
        )
        both_costs = bool(
            all(item["gates"]["passed"] for item in full.values())
        )
        results.append(
            {
                "candidate": spec.to_dict(),
                "stage": "development_only",
                "scenarios": full,
                "aapl_only_scenarios": aapl,
                "both_cost_gates_passed": both_costs,
                "paired_aapl_only_ablation": ablation,
                "passed": bool(both_costs and ablation["passed"]),
            }
        )
    return feature_label, predictions, states, target_audit, results, ledgers


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
            raise RegimeConsensusExperimentError(
                f"Recorded source commit lacks required file: {path}"
            )
        hashes[path] = _sha256_hex(bytes(blob.stdout))
    return hashes


def _artifact_snapshot(
    run_dir: Path, *, repo_root: Path | None
) -> tuple[dict[str, bytes], str | None]:
    if repo_root is None:
        return _local_artifact_snapshot(run_dir), None
    repo = repo_root.resolve()
    head = str(_git_state(repo).get("commit") or "")
    if not head:
        raise RegimeConsensusExperimentError("Could not capture verification Git HEAD")
    try:
        snapshot = _committed_artifact_snapshot(repo, run_dir.resolve(), head_commit=head)
    except Exception as exc:
        raise RegimeConsensusExperimentError(
            "Could not load artifact from immutable Git blobs"
        ) from exc
    return snapshot, head


def _parse_json(snapshot: Mapping[str, bytes], name: str) -> Any:
    try:
        return json.loads(snapshot[name].decode("utf-8", "strict"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegimeConsensusExperimentError(f"Artifact contains invalid {name}") from exc


def _snapshot_csv(
    snapshot: Mapping[str, bytes], name: str, *, index_column: str
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(snapshot[name]), float_precision="round_trip")
        index = pd.DatetimeIndex(pd.to_datetime(frame.pop(index_column), errors="raise"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RegimeConsensusExperimentError(f"Could not read sealed CSV: {name}") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RegimeConsensusExperimentError(f"Sealed dates are not chronological: {name}")
    frame.index = index
    frame.index.name = index_column
    return frame


def _verify_source_identity(
    snapshot: Mapping[str, bytes],
    *,
    repo_root: Path,
    artifact_commit: str,
) -> str:
    provenance = _parse_json(snapshot, "input_provenance.json")
    identity = provenance.get("source_identity") if isinstance(provenance, dict) else None
    if not isinstance(identity, dict):
        raise RegimeConsensusExperimentError("Artifact lacks source identity")
    git = identity.get("git")
    hashes = identity.get("source_hashes")
    if not isinstance(git, dict) or not isinstance(hashes, dict):
        raise RegimeConsensusExperimentError("Source identity is incomplete")
    source_commit = str(git.get("commit") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) or git.get("dirty") is not False:
        raise RegimeConsensusExperimentError("Recorded source was not a clean commit")
    if not _git_commit_is_ancestor(repo_root, source_commit, artifact_commit):
        raise RegimeConsensusExperimentError(
            "Source commit is not an ancestor of the artifact commit"
        )
    if _source_hashes_at_commit(repo_root, source_commit) != hashes:
        raise RegimeConsensusExperimentError("Recorded source hashes do not replay")
    return source_commit


def _rebuild_payloads_from_snapshot(
    snapshot: Mapping[str, bytes],
) -> tuple[dict[str, bytes], list[dict[str, Any]], dict[str, Any]]:
    prices = _snapshot_csv(
        snapshot, "development_prices_through_2018.csv", index_column="date"
    )
    context = _snapshot_csv(
        snapshot, "development_context_through_2018.csv", index_column="date"
    )
    prices, context, _ = _bounded_inputs(prices, context)
    deadline = _Deadline(lambda: 0.0)
    feature_label, predictions, states, targets, results, ledgers = _compute_development(
        prices, context, deadline=deadline
    )
    selected = select_development_candidate(results)
    model_payloads, refit = _refit_selected_models(
        feature_label, selected, deadline=deadline
    )
    payloads: dict[str, bytes] = {
        "oof_predictions_2005_2018.csv": _frame_csv_bytes(
            predictions.reset_index()
        ),
        "oof_hmm_states.json": _pretty_json_bytes(states),
        "candidate_targets_2005_2018.csv": _frame_csv_bytes(targets.reset_index()),
        "candidate_results.json": _pretty_json_bytes(results),
    }
    payloads.update(
        {
            name: _frame_csv_bytes(frame, float_format="%.17g")
            for name, frame in ledgers.items()
        }
    )
    payloads.update(model_payloads)
    return payloads, results, refit


def verify_development_artifact(
    artifact: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    run_dir = artifact.resolve()
    if run_dir.is_file():
        if run_dir.name != "report.json":
            raise RegimeConsensusExperimentError("Verification file must be report.json")
        run_dir = run_dir.parent
    snapshot, artifact_commit = _artifact_snapshot(run_dir, repo_root=repo_root)
    checksums = _verify_checksum_snapshot(snapshot)
    if snapshot.get(".gitattributes") != ARTIFACT_GIT_ATTRIBUTES:
        raise RegimeConsensusExperimentError("Byte-preservation rule changed")
    source_commit: str | None = None
    if repo_root is not None:
        if artifact_commit is None:  # pragma: no cover
            raise RegimeConsensusExperimentError("Missing artifact Git commit")
        source_commit = _verify_source_identity(
            snapshot,
            repo_root=repo_root.resolve(),
            artifact_commit=artifact_commit,
        )

    report = _parse_json(snapshot, "report.json")
    manifest = _parse_json(snapshot, "selection_manifest.json")
    results = _parse_json(snapshot, "candidate_results.json")
    provenance = _parse_json(snapshot, "input_provenance.json")
    if not isinstance(report, dict) or not isinstance(manifest, dict):
        raise RegimeConsensusExperimentError("Report and manifest must be objects")
    if (
        report.get("contract_version") != CONTRACT_VERSION
        or report.get("stage") != "development"
        or report.get("physical_data_end") != "2018-12-31"
        or report.get("post_2018_market_data_accessed") is not False
        or manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("stage") != "development"
        or manifest.get("physical_data_end") != "2018-12-31"
        or manifest.get("post_2018_market_data_access_allowed") is not False
        or manifest.get("post_2018_market_data_accessed") is not False
    ):
        raise RegimeConsensusExperimentError("Artifact violates pre-2019 identity")
    if (
        not isinstance(provenance, dict)
        or provenance.get("contract_version") != CONTRACT_VERSION
        or provenance.get("physical_data_end") != "2018-12-31"
        or provenance.get("post_2018_market_data_accessed") is not False
        or provenance.get("network_access") is not False
        or provenance.get("api_calls") != 0
        or provenance.get("llm_calls") != 0
    ):
        raise RegimeConsensusExperimentError("Input provenance violates offline boundary")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha256_tagged(_canonical_json_bytes(unsigned)):
        raise RegimeConsensusExperimentError("Selection manifest hash is invalid")
    if report.get("selection_manifest") != manifest:
        raise RegimeConsensusExperimentError("Report and manifest differ")
    if manifest.get("candidate_results_sha256") != _sha256_tagged(
        _canonical_json_bytes(results)
    ):
        raise RegimeConsensusExperimentError("Manifest does not bind candidate results")
    if report.get("candidate_results_sha256") != manifest.get(
        "candidate_results_sha256"
    ):
        raise RegimeConsensusExperimentError("Report does not bind candidate results")
    if manifest.get("source_exclusion_audits") != {
        key: dict(value) for key, value in SOURCE_EXCLUSION_AUDITS.items()
    }:
        raise RegimeConsensusExperimentError("Source exclusions changed")
    if (
        manifest.get("candidate_family")
        != [item.to_dict() for item in CANDIDATE_SPECS]
        or manifest.get("model_variants") != list(MODEL_VARIANTS)
        or manifest.get("walk_forward_folds")
        != [asdict(item) for item in WALK_FORWARD_FOLDS]
        or manifest.get("hmm_fit_count") != 21
        or manifest.get("cost_scenarios_bps")
        != {name: value for name, value in COST_SCENARIOS}
    ):
        raise RegimeConsensusExperimentError("Frozen candidate or fold contract changed")
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
        raise RegimeConsensusExperimentError("Execution contract changed")
    if not isinstance(results, list) or len(results) != len(CANDIDATE_SPECS):
        raise RegimeConsensusExperimentError("Candidate result set is not frozen")

    payload_hashes = manifest.get("payload_sha256")
    expected_payloads = set(snapshot) - {
        ".gitattributes",
        "checksums.json",
        "selection_manifest.json",
        "report.json",
    }
    if not isinstance(payload_hashes, dict) or set(payload_hashes) != expected_payloads:
        raise RegimeConsensusExperimentError("Manifest payload map is not exact")
    for name, digest in payload_hashes.items():
        if _sha256_tagged(snapshot[name]) != digest:
            raise RegimeConsensusExperimentError(f"Manifest payload mismatch: {name}")

    rebuilt, rebuilt_results, rebuilt_refit = _rebuild_payloads_from_snapshot(snapshot)
    for name, payload in rebuilt.items():
        if snapshot.get(name) != payload:
            raise RegimeConsensusExperimentError(
                f"Sealed payload does not exactly replay: {name}"
            )
    if _canonical_json_bytes(rebuilt_results) != _canonical_json_bytes(results):
        raise RegimeConsensusExperimentError("Candidate results do not replay")
    selected = select_development_candidate(rebuilt_results)
    if selected != manifest.get("selected_candidate_id"):
        raise RegimeConsensusExperimentError("Candidate selection does not replay")
    if bool(report.get("development_pass")) != (selected is not None):
        raise RegimeConsensusExperimentError("Development pass flag is inconsistent")
    reproducibility = report.get("reproducibility", {})
    if (
        reproducibility.get("model_calls") != 0
        or reproducibility.get("model_calls_semantics")
        != "external_or_llm_model_calls_only"
        or reproducibility.get("local_numeric_hmm_fits_before_seal")
        != 21 + (6 if selected is not None else 0)
        or reproducibility.get("llm_calls") != 0
        or reproducibility.get("network_calls") != 0
        or reproducibility.get("estimated_external_cost_usd") != 0.0
        or reproducibility.get("api_cost_display") != "$0.00"
    ):
        raise RegimeConsensusExperimentError("Zero-cost offline evidence changed")
    model_files = {name for name in snapshot if name.startswith("final_")}
    refit = manifest.get("refit", {})
    if refit != rebuilt_refit or report.get("refit") != rebuilt_refit:
        raise RegimeConsensusExperimentError("Final refit metadata does not replay")
    if selected is None:
        if model_files or refit.get("performed") is not False:
            raise RegimeConsensusExperimentError("Null selection must not have a refit")
    elif not model_files or refit.get("performed") is not True:
        raise RegimeConsensusExperimentError("Selected candidate lacks final refit")
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


def run_development_from_inputs(
    *,
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    output_dir: Path,
    input_provenance: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    run_id: str | None = None,
    _deadline: _Deadline | None = None,
) -> dict[str, Any]:
    """Run, replay-verify, and atomically seal the complete experiment."""

    deadline = _deadline or _Deadline(clock)
    prices, context, boundary = _bounded_inputs(price_frame, context_frame)
    deadline.check("bounded input validation")
    feature_label, predictions, states, targets, results, ledgers = _compute_development(
        prices, context, deadline=deadline
    )
    selected = select_development_candidate(results)
    model_payloads, refit = _refit_selected_models(
        feature_label, selected, deadline=deadline
    )
    deadline.check("selected-model refit")

    created_at = datetime.now(timezone.utc)
    resolved_run_id = run_id or (
        f"regime-consensus-development-{created_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resolved_run_id):
        raise RegimeConsensusExperimentError("run_id is not a safe directory name")
    run_dir = output_dir.resolve() / resolved_run_id
    temporary = output_dir.resolve() / f".{resolved_run_id}.{uuid.uuid4().hex}.sealing"
    if run_dir.exists():
        raise RegimeConsensusExperimentError(f"Artifact already exists: {run_dir}")

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
        "oof_hmm_states.json": _pretty_json_bytes(states),
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
        "hmm_fit_count": 21,
        "cost_scenarios_bps": {name: value for name, value in COST_SCENARIOS},
        "selected_candidate_id": selected,
        "selection_tie_break": [
            "highest 10bps minimum-fold active-log edge",
            "highest 10bps total active-log edge",
            "highest 10bps Brier relative improvement",
            "fewest 10bps cash days",
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
            "hmm_configuration_may_change_after_this_run": False,
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
            "completed_before_seal_within_limit": deadline.elapsed() <= RUN_TIME_LIMIT_SECONDS,
            "model_calls": 0,
            "model_calls_semantics": "external_or_llm_model_calls_only",
            "local_numeric_hmm_fits_before_seal": 21
            + (6 if selected is not None else 0),
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
        deadline.check("completed development verification")
        temporary.replace(run_dir)
        promoted = True
        deadline.check("atomic artifact promotion")
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if promoted and run_dir.exists():
            shutil.rmtree(run_dir)
        raise
    return report


def run_development_experiment(
    *, repo_root: Path, price_artifact: Path, context_parquet: Path, output_dir: Path
) -> dict[str, Any]:
    deadline = _Deadline(time.perf_counter)
    repo = repo_root.resolve()
    repository = validate_source_repository(repo, [repo / path for path in SOURCE_FILES])
    deadline.check("source repository validation")
    git = _git_state(repo)
    if git.get("dirty") is not False:
        raise RegimeConsensusExperimentError(
            "Development requires committed source and a clean worktree"
        )
    commit = str(git.get("commit") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise RegimeConsensusExperimentError("Could not capture source commit")
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
        description="Run or verify the sealed through-2018 regime experiment"
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
    if args.development_artifact is not None:
        raise SystemExit("run does not accept --development-artifact")
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
    return 0 if report["development_pass"] else 2


__all__ = [
    "BASE_COST_BPS",
    "CANDIDATE_SPECS",
    "CONTEXT_PARQUET_QUERY",
    "CONTRACT_VERSION",
    "COST_SCENARIOS",
    "CandidateSpec",
    "LoadedDevelopmentInputs",
    "RUN_TIME_LIMIT_SECONDS",
    "RegimeConsensusExperimentError",
    "RegimeConsensusExperimentTimeout",
    "SOURCE_EXCLUSION_AUDITS",
    "STRESS_COST_BPS",
    "load_bounded_context_parquet",
    "load_public_development_inputs",
    "run_development_experiment",
    "run_development_from_inputs",
    "select_development_candidate",
    "verify_development_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
