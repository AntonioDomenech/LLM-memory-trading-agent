"""Sealed, offline pre-2019 development runner for direct AAPL CASH edge.

This module has one deliberately narrow evidence role: evaluate the eight
predeclared direct-edge candidates on chronological 2005-2018 out-of-fold
predictions and select at most one.  It contains no 2019+ evaluation entry
point.  The public runner replays prices from a committed, checksum-verified
CFTC development artifact and reads IWM/VIX/TNX from one explicitly bounded
local Parquet query.  The in-memory entry point exists for deterministic tests
and never performs network I/O.
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
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .deterministic_aapl import CostAssumptions, EvaluationPeriod, file_sha256
from .direct_edge_features import (
    CONTEXT_VALUE_COLUMNS,
    MARKET_SENTIMENT_FEATURE_COLUMNS,
    PRICE_FEATURE_COLUMNS,
    build_direct_edge_feature_label_frame,
    canonical_market_sentiment_context,
    price_only_chronological_training_mask,
    sentiment_chronological_training_mask,
)
from .direct_edge_gam import DirectEdgeGAM, DirectEdgeGAMConfig
from .direct_edge_scoring import evaluate_direct_edge_development
from .direct_edge_walkforward import (
    DIRECT_EDGE_CANDIDATES,
    MODEL_FAMILIES,
    WALK_FORWARD_FOLDS,
    DirectEdgeCandidate,
    build_pre2019_direct_edge_predictions,
    candidate_cash_block_start_column,
    candidate_cash_target_column,
    five_session_cash_policy,
)
from .downside_ensemble_experiment import (
    load_cftc_development_artifact as _load_verified_cftc_inputs,
)
from .cftc_cot_experiment import (
    _committed_artifact_snapshot,
    _git_commit_is_ancestor,
)
from .downside_scoring import ACTIVE_EDGE_WIN_TOLERANCE
from .unleveraged_aapl import (
    _git_state,
    canonical_context_frame,
    simulate_unleveraged_period,
    validate_source_repository,
)


CONTRACT_VERSION = "aapl-direct-edge-development-v1"
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
DEVELOPMENT_PERIOD = EvaluationPeriod(
    "direct_edge_development", "2005-01-01", "2018-12-31"
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

# This is the only query used by the public context loader.  It is fixed,
# symbol-bounded, end-bounded, ordered, and contains no API or extension call.
CONTEXT_PARQUET_QUERY = """
SELECT date, symbol, close, adj_close
FROM read_parquet(?)
WHERE CAST(date AS DATE) <= DATE '2018-12-31'
  AND upper(replace(trim(symbol), '^', '')) IN ('IWM', 'VIX', 'TNX')
ORDER BY CAST(date AS DATE), upper(replace(trim(symbol), '^', ''))
""".strip()

SOURCE_FILES = (
    ".gitattributes",
    "agent_benchmark/deterministic_aapl.py",
    "agent_benchmark/direct_edge_features.py",
    "agent_benchmark/direct_edge_gam.py",
    "agent_benchmark/direct_edge_scoring.py",
    "agent_benchmark/direct_edge_walkforward.py",
    "agent_benchmark/direct_edge_experiment.py",
    "agent_benchmark/downside_ensemble_experiment.py",
    "agent_benchmark/downside_scoring.py",
    "agent_benchmark/unleveraged_aapl.py",
    "docs/direct_edge_experiment.md",
    "requirements.txt",
)


SOURCE_EXCLUSION_AUDITS: Mapping[str, Mapping[str, Any]] = {
    "macro": {
        "included": False,
        "review_scope": "available macro_daily design and stored fields",
        "finding": "macro placeholders are present and vintage/revision timing is unsafe",
        "decision": "excluded before candidate construction",
    },
    "news": {
        "included": False,
        "review_scope": "available news event rows and timestamps",
        "finding": "news event data and publication/availability timing are unsafe",
        "decision": "excluded before candidate construction",
    },
}


class DirectEdgeExperimentError(RuntimeError):
    """Raised when the sealed direct-edge development contract is violated."""


class DirectEdgeExperimentTimeout(DirectEdgeExperimentError):
    """Raised when the hard one-hour wall-clock budget is exhausted."""


@dataclass(frozen=True)
class CandidateSpec:
    model_family: str
    candidate: DirectEdgeCandidate

    def __post_init__(self) -> None:
        if self.model_family not in MODEL_FAMILIES:
            raise ValueError(f"Unknown direct-edge model family: {self.model_family!r}")
        if self.candidate not in DIRECT_EDGE_CANDIDATES:
            raise ValueError("Candidate is not in the frozen direct-edge grid")

    @property
    def candidate_id(self) -> str:
        return f"{self.model_family}_{self.candidate.candidate_id}"

    @property
    def cash_target_column(self) -> str:
        return candidate_cash_target_column(
            self.model_family, self.candidate.candidate_id
        )

    @property
    def cash_block_start_column(self) -> str:
        return candidate_cash_block_start_column(
            self.model_family, self.candidate.candidate_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "gate_id": self.candidate.candidate_id,
            "probability_gate": float(self.candidate.probability_gate),
            "expected_edge_gate_10bps": float(self.candidate.expected_edge_gate),
            "trigger": (
                "cash_win_probability >= probability_gate and "
                "expected_cash_active_log_edge_10bps >= expected_edge_gate_10bps"
            ),
            "cash_decision_rows": 5,
        }


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(family, candidate)
    for family in MODEL_FAMILIES
    for candidate in DIRECT_EDGE_CANDIDATES
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
            raise DirectEdgeExperimentTimeout(
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


def _verify_checksum_snapshot(snapshot: Mapping[str, bytes]) -> dict[str, str]:
    try:
        checksums = json.loads(snapshot["checksums.json"].decode("utf-8", "strict"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectEdgeExperimentError("Artifact has no valid checksum manifest") from exc
    expected = set(snapshot) - {"checksums.json"}
    if not isinstance(checksums, dict) or set(checksums) != expected:
        raise DirectEdgeExperimentError(
            "Checksum manifest does not cover the exact artifact file set"
        )
    for name, digest in checksums.items():
        if not isinstance(digest, str) or _sha256_hex(snapshot[name]) != digest:
            raise DirectEdgeExperimentError(f"Artifact checksum mismatch: {name}")
    return {str(name): str(value) for name, value in checksums.items()}


def _local_artifact_snapshot(run_dir: Path) -> dict[str, bytes]:
    if not run_dir.is_dir():
        raise DirectEdgeExperimentError(f"Artifact directory does not exist: {run_dir}")
    nested = [
        path for path in run_dir.rglob("*") if path.is_file() and path.parent != run_dir
    ]
    if nested:
        raise DirectEdgeExperimentError("Artifact directory must not contain nested files")
    return {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}


def _artifact_snapshot(
    run_dir: Path,
    *,
    repo_root: Path | None,
) -> tuple[dict[str, bytes], str | None]:
    if repo_root is None:
        return _local_artifact_snapshot(run_dir), None
    repo = repo_root.resolve()
    git = _git_state(repo)
    head = str(git.get("commit") or "")
    if not head:
        raise DirectEdgeExperimentError(
            "Could not capture Git HEAD for artifact verification"
        )
    try:
        snapshot = _committed_artifact_snapshot(
            repo, run_dir.resolve(), head_commit=head
        )
    except Exception as exc:
        raise DirectEdgeExperimentError(
            "Could not load the artifact from immutable Git blobs"
        ) from exc
    return snapshot, head


def _parse_json_snapshot(snapshot: Mapping[str, bytes], name: str) -> Any:
    try:
        return json.loads(snapshot[name].decode("utf-8", "strict"))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectEdgeExperimentError(f"Artifact contains invalid {name}") from exc


def _seal_artifact_bundle(run_dir: Path, payloads: Mapping[str, bytes]) -> dict[str, str]:
    if run_dir.exists():
        raise DirectEdgeExperimentError(f"Artifact directory already exists: {run_dir}")
    if any("/" in name or "\\" in name for name in payloads):
        raise DirectEdgeExperimentError("Artifact payload names must be flat")
    if {".gitattributes", "checksums.json"}.intersection(payloads):
        raise DirectEdgeExperimentError("Reserved artifact name supplied")
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
    if _verify_checksum_snapshot(_local_artifact_snapshot(run_dir)) != checksums:
        raise DirectEdgeExperimentError("Artifact checksum readback changed unexpectedly")
    return checksums


def _normalize_context_input_dates(context_frame: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(context_frame, pd.DataFrame):
        raise TypeError("context_frame must be a pandas DataFrame")
    keyed = {
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): column
        for column in context_frame.columns
    }
    source = next((keyed[name] for name in ("date", "datetime", "timestamp") if name in keyed), None)
    raw = context_frame[source] if source is not None else context_frame.index
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(raw, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise DirectEdgeExperimentError("Context input has invalid dates") from exc
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _normalize_price_input_dates(price_frame: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(price_frame, pd.DataFrame):
        raise TypeError("price_frame must be a pandas DataFrame")
    keyed = {
        re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_"): column
        for column in price_frame.columns
    }
    source = next(
        (keyed[name] for name in ("date", "datetime", "timestamp") if name in keyed),
        None,
    )
    raw = price_frame[source] if source is not None else price_frame.index
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(raw, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise DirectEdgeExperimentError("Price input has invalid dates") from exc
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _bounded_inputs(
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_price_dates = _normalize_price_input_dates(price_frame)
    if len(raw_price_dates) and raw_price_dates.max() > DEVELOPMENT_END:
        first = raw_price_dates[raw_price_dates > DEVELOPMENT_END][0]
        raise DirectEdgeExperimentError(
            f"Development input contains a post-2018 price row: {first.date().isoformat()}"
        )
    prices = canonical_context_frame(price_frame)
    if prices.index.max() > DEVELOPMENT_END:  # defensive after canonicalization
        raise DirectEdgeExperimentError("Canonical prices contain a post-2018 row")
    required_years = set(range(2005, 2019))
    observed_years = {int(value) for value in prices.index.year}
    if not required_years.issubset(observed_years):
        raise DirectEdgeExperimentError(
            "Development prices do not cover every 2005-2018 year: "
            f"{sorted(required_years - observed_years)}"
        )
    if prices.index.min().year > 2004:
        raise DirectEdgeExperimentError("Development prices lack pre-2005 history")

    raw_context_dates = _normalize_context_input_dates(context_frame)
    if len(raw_context_dates) and raw_context_dates.max() > DEVELOPMENT_END:
        first = raw_context_dates[raw_context_dates > DEVELOPMENT_END][0]
        raise DirectEdgeExperimentError(
            f"Development input contains a post-2018 context row: {first.date().isoformat()}"
        )
    context = canonical_market_sentiment_context(context_frame, prices.index)
    if context.index.max() > DEVELOPMENT_END:
        raise DirectEdgeExperimentError("Context alignment produced a post-2018 row")
    finite = np.isfinite(context.to_numpy(dtype=float))
    return prices, context, {
        "price_first_date": prices.index.min().date().isoformat(),
        "price_last_date": prices.index.max().date().isoformat(),
        "price_rows": int(len(prices)),
        "context_first_aligned_date": context.index.min().date().isoformat(),
        "context_last_aligned_date": context.index.max().date().isoformat(),
        "context_aligned_rows": int(len(context)),
        "context_complete_rows": int(finite.all(axis=1).sum()),
        "context_missing_cells": int((~finite).sum()),
    }


def load_bounded_context_parquet(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only IWM/VIX/TNX rows through 2018 from a local Parquet file."""

    source = path.resolve()
    if not source.is_file():
        raise DirectEdgeExperimentError(f"Context Parquet does not exist: {source}")
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            frame = connection.execute(CONTEXT_PARQUET_QUERY, [str(source)]).fetchdf()
        finally:
            connection.close()
    except Exception as exc:
        raise DirectEdgeExperimentError("Could not execute bounded context Parquet query") from exc
    dates = _normalize_context_input_dates(frame)
    if len(dates) and dates.max() > DEVELOPMENT_END:
        raise DirectEdgeExperimentError("Bounded context query returned a post-2018 row")
    bounded_bytes = _frame_csv_bytes(frame)
    return frame, {
        "source_type": "local_context_daily_parquet_bounded_query",
        "source_path": str(source),
        "source_file_sha256": f"sha256:{file_sha256(source)}",
        "query": CONTEXT_PARQUET_QUERY,
        "query_parameters": [str(source)],
        "allowed_symbols": ["IWM", "VIX", "TNX"],
        "physical_data_end": "2018-12-31",
        "bounded_result_rows": int(len(frame)),
        "bounded_result_sha256": _sha256_tagged(bounded_bytes),
        "network_access": False,
    }


def load_public_development_inputs(
    *,
    price_artifact: Path,
    context_parquet: Path,
    repo_root: Path,
) -> LoadedDevelopmentInputs:
    """Replay committed prices and pair them with the bounded local context query."""

    try:
        price_inputs = _load_verified_cftc_inputs(
            price_artifact, repo_root=repo_root.resolve()
        )
    except Exception as exc:
        raise DirectEdgeExperimentError(
            "Could not replay the committed checksum-verified price artifact"
        ) from exc
    context, context_provenance = load_bounded_context_parquet(context_parquet)
    return LoadedDevelopmentInputs(
        price_frame=price_inputs.price_frame,
        context_frame=context,
        provenance={
            "price": dict(price_inputs.provenance),
            "context": context_provenance,
        },
    )


def _family_prediction_columns(model_family: str) -> tuple[str, str, str, str]:
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {model_family!r}")
    return (
        f"{model_family}_cash_win_probability",
        f"{model_family}_expected_net_edge",
        f"{model_family}_baseline_cash_win_probability",
        f"{model_family}_baseline_mean_net_edge",
    )


def _prediction_inputs(
    predictions: pd.DataFrame,
    model_family: str,
    cash_target: Sequence[float],
    cash_block_start: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, dict[str, int]]:
    probability_name, edge_name, baseline_probability_name, baseline_edge_name = (
        _family_prediction_columns(model_family)
    )
    probability = pd.to_numeric(predictions[probability_name], errors="coerce").to_numpy(float)
    edge = pd.to_numeric(predictions[edge_name], errors="coerce").to_numpy(float)
    baseline_probability = pd.to_numeric(
        predictions[baseline_probability_name], errors="coerce"
    ).to_numpy(float)
    baseline_edge = pd.to_numeric(predictions[baseline_edge_name], errors="coerce").to_numpy(float)
    label = pd.to_numeric(predictions["cash_beats_long_10bps"], errors="coerce").to_numpy(float)
    actual_edge = pd.to_numeric(
        predictions["cash_active_log_edge_10bps"], errors="coerce"
    ).to_numpy(float)
    target = np.asarray(cash_target, dtype=float)
    block_start = np.asarray(cash_block_start, dtype=float)
    if target.shape != (len(predictions),) or block_start.shape != target.shape:
        raise DirectEdgeExperimentError(
            "OOF CASH target and block starts are not prediction-aligned"
        )
    finite = (
        np.isfinite(probability)
        & np.isfinite(edge)
        & np.isfinite(baseline_probability)
        & np.isfinite(baseline_edge)
    )
    if not finite.all():
        raise DirectEdgeExperimentError(
            f"Every OOF row must have finite effective predictions for {model_family}; "
            f"unavailable rows={int((~finite).sum())}"
        )
    return (
        probability,
        label,
        baseline_probability,
        edge,
        actual_edge,
        baseline_edge,
        target,
        block_start,
        predictions.index,
        {
            "oof_rows": int(len(predictions)),
            "scored_prediction_rows": int(len(predictions)),
            "unavailable_prediction_rows": 0,
        },
    )


def _full_exposure_target(
    price_index: pd.DatetimeIndex,
    prediction_index: pd.DatetimeIndex,
    cash_target: Sequence[float],
) -> pd.Series:
    if not prediction_index.isin(price_index).all():
        raise DirectEdgeExperimentError("Prediction dates are absent from prices")
    cash = np.asarray(cash_target, dtype=float)
    if cash.shape != (len(prediction_index),) or not np.isin(cash, (0.0, 1.0)).all():
        raise DirectEdgeExperimentError("Candidate CASH target must be aligned and binary")
    exposure = pd.Series(1.0, index=price_index, name="target_exposure")
    exposure.loc[prediction_index] = 1.0 - cash
    return exposure


def _common_support_mask(predictions: pd.DataFrame) -> np.ndarray:
    required = (
        predictions["price_features_ready"].astype(bool).to_numpy()
        & predictions["sentiment_features_ready"].astype(bool).to_numpy()
        & ~predictions["market_sentiment_used_price_fallback"].astype(bool).to_numpy()
    )
    columns: list[str] = []
    for family in MODEL_FAMILIES:
        columns.extend(_family_prediction_columns(family))
    finite = np.isfinite(predictions.loc[:, columns].to_numpy(dtype=float)).all(axis=1)
    return required & finite


def _common_support_cash_target(
    predictions: pd.DataFrame,
    spec: CandidateSpec,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability_name, edge_name, _, _ = _family_prediction_columns(spec.model_family)
    probability = pd.to_numeric(predictions[probability_name], errors="coerce").to_numpy(float)
    edge = pd.to_numeric(predictions[edge_name], errors="coerce").to_numpy(float)
    trigger = (
        support
        & np.isfinite(probability)
        & np.isfinite(edge)
        & (probability >= spec.candidate.probability_gate)
        & (edge >= spec.candidate.expected_edge_gate)
    )
    return five_session_cash_policy(trigger)


def _common_support_predictive_metrics(
    predictions: pd.DataFrame,
    model_family: str,
    support: np.ndarray,
) -> dict[str, Any]:
    """Score probability and edge heads only on identical mature support."""

    probability_name, edge_name, _, _ = _family_prediction_columns(model_family)
    probability = pd.to_numeric(
        predictions[probability_name], errors="coerce"
    ).to_numpy(float)
    predicted_edge = pd.to_numeric(
        predictions[edge_name], errors="coerce"
    ).to_numpy(float)
    label = pd.to_numeric(
        predictions["cash_beats_long_10bps"], errors="coerce"
    ).to_numpy(float)
    actual_edge = pd.to_numeric(
        predictions["cash_active_log_edge_10bps"], errors="coerce"
    ).to_numpy(float)
    eligible = (
        np.asarray(support, dtype=bool)
        & np.isfinite(probability)
        & np.isfinite(predicted_edge)
        & np.isfinite(label)
        & np.isfinite(actual_edge)
    )
    if not eligible.any():
        raise DirectEdgeExperimentError(
            "Common-support ablation has no mature sentiment-ready rows"
        )
    if not np.isin(label[eligible], (0.0, 1.0)).all():
        raise DirectEdgeExperimentError(
            "Common-support binary labels are not exact zero/one values"
        )
    return {
        "rows": int(eligible.sum()),
        "brier_score": float(
            np.mean(np.square(probability[eligible] - label[eligible]))
        ),
        "expected_edge_mae": float(
            np.mean(np.abs(predicted_edge[eligible] - actual_edge[eligible]))
        ),
    }


def _simulate(frame: pd.DataFrame, target: pd.Series, *, cost_bps: float) -> pd.DataFrame:
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
    prediction_inputs: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        pd.DatetimeIndex,
        dict[str, int],
    ],
    cost_bps: float,
) -> dict[str, Any]:
    probability, label, baseline_probability, edge, actual_edge, baseline_edge, cash, block_start, decision_dates, availability = prediction_inputs
    result = evaluate_direct_edge_development(
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
        block_start,
        decision_dates,
        cost_bps=cost_bps,
    )
    return {**result, "prediction_availability": availability}


def common_support_ablation(
    common_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    gate_id: str,
) -> dict[str, Any]:
    """Require sentiment to improve prediction and trading on identical rows."""

    scenarios: dict[str, Any] = {}
    all_checks: list[bool] = []
    for scenario_name, _ in COST_SCENARIOS:
        try:
            price_result = common_results[f"price_only_{gate_id}"][scenario_name]
            sentiment_result = common_results[f"market_sentiment_{gate_id}"][scenario_name]
            price = price_result["metrics"]
            sentiment = sentiment_result["metrics"]
        except KeyError as exc:
            raise DirectEdgeExperimentError(
                f"Common-support result is incomplete for {gate_id}"
            ) from exc
        price_predictive = price_result.get(
            "common_support_predictive_metrics", price
        )
        sentiment_predictive = sentiment_result.get(
            "common_support_predictive_metrics", sentiment
        )
        checks = {
            "strict_brier_improvement": (
                float(sentiment_predictive["brier_score"])
                + ACTIVE_EDGE_WIN_TOLERANCE
                < float(price_predictive["brier_score"])
            ),
            "strict_expected_edge_mae_improvement": (
                float(sentiment_predictive["expected_edge_mae"])
                + ACTIVE_EDGE_WIN_TOLERANCE
                < float(price_predictive["expected_edge_mae"])
            ),
            "strict_total_active_log_improvement": (
                float(sentiment["total_active_log_edge"])
                > float(price["total_active_log_edge"])
                + ACTIVE_EDGE_WIN_TOLERANCE
            ),
            "weakest_fold_not_worse": (
                float(sentiment["minimum_fold_active_log_edge"])
                + ACTIVE_EDGE_WIN_TOLERANCE
                >= float(price["minimum_fold_active_log_edge"])
            ),
        }
        scenarios[scenario_name] = {
            "price_only": {
                "brier_score": float(price_predictive["brier_score"]),
                "expected_edge_mae": float(
                    price_predictive["expected_edge_mae"]
                ),
                "total_active_log_edge": float(price["total_active_log_edge"]),
                "minimum_fold_active_log_edge": float(
                    price["minimum_fold_active_log_edge"]
                ),
            },
            "market_sentiment": {
                "brier_score": float(sentiment_predictive["brier_score"]),
                "expected_edge_mae": float(
                    sentiment_predictive["expected_edge_mae"]
                ),
                "total_active_log_edge": float(sentiment["total_active_log_edge"]),
                "minimum_fold_active_log_edge": float(
                    sentiment["minimum_fold_active_log_edge"]
                ),
            },
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
        all_checks.extend(checks.values())
    return {
        "applicable": True,
        "comparison": "identical sentiment-ready OOF decision rows",
        "active_edge_win_tolerance": ACTIVE_EDGE_WIN_TOLERANCE,
        "scenarios": scenarios,
        "passed": bool(all(all_checks)),
    }


def select_development_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> str | None:
    """Apply the frozen passing-only rank: worst fold, total, episodes, ID."""

    passing = [item for item in candidate_results if bool(item.get("passed"))]
    if not passing:
        return None

    def rank(item: Mapping[str, Any]) -> tuple[float, float, int, str]:
        stress = item["scenarios"]["stress_10bps"]["metrics"]
        return (
            -float(stress["minimum_fold_active_log_edge"]),
            -float(stress["total_active_log_edge"]),
            int(stress["cash_episodes"]),
            str(item["candidate"]["candidate_id"]),
        )

    return str(sorted(passing, key=rank)[0]["candidate"]["candidate_id"])


def _sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode("ascii")).hexdigest()


def _refit_metadata(training: pd.DataFrame) -> dict[str, str]:
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    return {
        "training_start_date": training.index.min().date().isoformat(),
        "training_end_date": training.index.max().date().isoformat(),
        "maximum_label_maturity_date": maturity.max().date().isoformat(),
        "training_dates_sha256": _sha256_lines(
            [value.date().isoformat() for value in training.index]
        ),
        "binary_target_sha256": _sha256_lines(
            [float(value).hex() for value in training["cash_beats_long_10bps"]]
        ),
        "edge_target_sha256": _sha256_lines(
            [float(value).hex() for value in training["cash_active_log_edge_10bps"]]
        ),
    }


def _refit_one_model(
    feature_label_frame: pd.DataFrame,
    *,
    model_family: str,
    deadline: _Deadline,
) -> tuple[DirectEdgeGAM, dict[str, Any]]:
    # A 2019-01-01 information boundary includes exactly labels that matured by
    # the final 2018 open while still excluding every post-2018 decision row.
    boundary = pd.Timestamp("2019-01-01")
    if model_family == "price_only":
        mask = price_only_chronological_training_mask(
            feature_label_frame, as_of_date=boundary
        )
        names = PRICE_FEATURE_COLUMNS
    elif model_family == "market_sentiment":
        mask = sentiment_chronological_training_mask(
            feature_label_frame, as_of_date=boundary
        )
        names = PRICE_FEATURE_COLUMNS + MARKET_SENTIMENT_FEATURE_COLUMNS
    else:
        raise DirectEdgeExperimentError(f"Unknown refit family: {model_family}")
    training = feature_label_frame.loc[mask].copy()
    if training.empty:
        raise DirectEdgeExperimentError(f"No eligible {model_family} refit rows")
    metadata = _refit_metadata(training)

    def fit() -> DirectEdgeGAM:
        return DirectEdgeGAM(DirectEdgeGAMConfig()).fit(
            training.loc[:, list(names)],
            training["cash_beats_long_10bps"].to_numpy(dtype=float),
            training["cash_active_log_edge_10bps"].to_numpy(dtype=float),
            feature_names=names,
            fit_metadata=metadata,
        )

    first = fit()
    deadline.check(f"first deterministic {model_family} refit")
    second = fit()
    deadline.check(f"second deterministic {model_family} refit")
    first_bytes = (first.canonical_json() + "\n").encode("utf-8")
    second_bytes = (second.canonical_json() + "\n").encode("utf-8")
    if first_bytes != second_bytes:
        raise DirectEdgeExperimentError(
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
        ("price_only", "market_sentiment")
        if selected.model_family == "market_sentiment"
        else ("price_only",)
    )
    payloads: dict[str, bytes] = {}
    metadata: dict[str, Any] = {}
    for family in families:
        model, item = _refit_one_model(
            feature_label_frame, model_family=family, deadline=deadline
        )
        filename = f"final_{family}_model_through_2018.json"
        payloads[filename] = (model.canonical_json() + "\n").encode("utf-8")
        metadata[family] = item
    return payloads, {
        "performed": True,
        "selected_candidate_id": selected.candidate_id,
        "training_cutoff": "2018-12-31",
        "models": metadata,
    }


def _feature_provenance() -> dict[str, Any]:
    return {
        "decision_timestamp": "after completed close t",
        "execution_timestamp": "next adjusted AAPL open t+1",
        "label_maturity": "adjusted AAPL open t+6",
        "price_features": list(PRICE_FEATURE_COLUMNS),
        "market_sentiment_features": list(MARKET_SENTIMENT_FEATURE_COLUMNS),
        "market_sentiment_raw_context": list(CONTEXT_VALUE_COLUMNS),
        "context_alignment": "exact AAPL session date; no forward fill, interpolation, or zero imputation",
        "sentiment_missing_policy": "exact price-only fallback",
        "training_policy": "fixed purged expanding folds; each outcome matured strictly before its fold",
        "excluded_sources": {key: dict(value) for key, value in SOURCE_EXCLUSION_AUDITS.items()},
    }


def _snapshot_csv(
    snapshot: Mapping[str, bytes],
    name: str,
    *,
    index_column: str | None = None,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            io.BytesIO(snapshot[name]),
            float_precision="round_trip",
        )
    except (KeyError, ValueError) as exc:
        raise DirectEdgeExperimentError(f"Could not read sealed CSV: {name}") from exc
    if index_column is None:
        return frame
    if index_column not in frame.columns:
        raise DirectEdgeExperimentError(
            f"Sealed CSV lacks {index_column}: {name}"
        )
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame.pop(index_column), errors="raise")
        )
    except (TypeError, ValueError) as exc:
        raise DirectEdgeExperimentError(
            f"Sealed CSV has invalid {index_column}: {name}"
        ) from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise DirectEdgeExperimentError(
            f"Sealed CSV index is not unique and chronological: {name}"
        )
    frame.index = index
    frame.index.name = index_column
    return frame


def _sealed_binary_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise DirectEdgeExperimentError(f"Sealed target column is absent: {column}")
    try:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise DirectEdgeExperimentError(
            f"Sealed target column is not numeric: {column}"
        ) from exc
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise DirectEdgeExperimentError(
            f"Sealed target column is not exactly binary: {column}"
        )
    return values


def _replay_sealed_candidate_results(
    snapshot: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    """Recompute every downstream target, ledger score, gate, and ablation."""

    predictions = _snapshot_csv(
        snapshot,
        "oof_predictions_2005_2018.csv",
        index_column="decision_date",
    )
    targets = _snapshot_csv(
        snapshot,
        "candidate_targets_2005_2018.csv",
        index_column="decision_date",
    )
    if not predictions.index.equals(targets.index):
        raise DirectEdgeExperimentError(
            "Sealed predictions and candidate targets use different decisions"
        )
    prices = _snapshot_csv(
        snapshot,
        "development_prices_through_2018.csv",
        index_column="date",
    )
    context = _snapshot_csv(
        snapshot,
        "development_context_through_2018.csv",
        index_column="date",
    )
    rebuilt_feature_label = build_direct_edge_feature_label_frame(prices, context)
    rebuilt_predictions = build_pre2019_direct_edge_predictions(
        rebuilt_feature_label
    )
    rebuilt_prediction_bytes = _frame_csv_bytes(
        rebuilt_predictions.reset_index(),
        float_format="%.17g",
    )
    if snapshot.get("oof_predictions_2005_2018.csv") != rebuilt_prediction_bytes:
        raise DirectEdgeExperimentError(
            "Sealed OOF predictions do not replay from bounded inputs and frozen models"
        )

    support = _common_support_mask(predictions)
    sealed_support = _sealed_binary_column(
        targets, "market_sentiment_common_support"
    ).astype(bool)
    if not np.array_equal(support, sealed_support):
        raise DirectEdgeExperimentError(
            "Sealed common-support mask does not replay from predictions"
        )

    actual_cash: dict[str, np.ndarray] = {}
    actual_block_start: dict[str, np.ndarray] = {}
    common_cash: dict[str, np.ndarray] = {}
    common_block_start: dict[str, np.ndarray] = {}
    for spec in CANDIDATE_SPECS:
        actual_cash[spec.candidate_id] = _sealed_binary_column(
            targets, f"{spec.candidate_id}_cash"
        )
        actual_block_start[spec.candidate_id] = _sealed_binary_column(
            targets, f"{spec.candidate_id}_cash_block_start"
        )
        common_cash[spec.candidate_id] = _sealed_binary_column(
            targets, f"{spec.candidate_id}_common_support_cash"
        )
        common_block_start[spec.candidate_id] = _sealed_binary_column(
            targets,
            f"{spec.candidate_id}_common_support_cash_block_start",
        )

        prediction_cash = _sealed_binary_column(
            predictions, spec.cash_target_column
        )
        prediction_start = _sealed_binary_column(
            predictions, spec.cash_block_start_column
        )
        expected_full, expected_full_start = _common_support_cash_target(
            predictions,
            spec,
            np.ones(len(predictions), dtype=bool),
        )
        if not np.array_equal(prediction_cash, expected_full):
            raise DirectEdgeExperimentError(
                f"Sealed prediction CASH target does not replay from frozen gates: {spec.candidate_id}"
            )
        if not np.array_equal(prediction_start, expected_full_start):
            raise DirectEdgeExperimentError(
                f"Sealed prediction block starts do not replay from frozen gates: {spec.candidate_id}"
            )
        if not np.array_equal(actual_cash[spec.candidate_id], prediction_cash):
            raise DirectEdgeExperimentError(
                f"Sealed full CASH target differs from prediction: {spec.candidate_id}"
            )
        if not np.array_equal(
            actual_block_start[spec.candidate_id], prediction_start
        ):
            raise DirectEdgeExperimentError(
                f"Sealed block starts differ from prediction: {spec.candidate_id}"
            )

        expected_common, expected_common_start = _common_support_cash_target(
            predictions, spec, support
        )
        if not np.array_equal(common_cash[spec.candidate_id], expected_common):
            raise DirectEdgeExperimentError(
                f"Sealed common-support CASH target does not replay: {spec.candidate_id}"
            )
        if not np.array_equal(
            common_block_start[spec.candidate_id], expected_common_start
        ):
            raise DirectEdgeExperimentError(
                f"Sealed common-support block starts do not replay: {spec.candidate_id}"
            )

    full_targets = {
        spec.candidate_id: _full_exposure_target(
            prices.index,
            predictions.index,
            actual_cash[spec.candidate_id],
        )
        for spec in CANDIDATE_SPECS
    }
    common_targets = {
        spec.candidate_id: _full_exposure_target(
            prices.index,
            predictions.index,
            common_cash[spec.candidate_id],
        )
        for spec in CANDIDATE_SPECS
    }

    def exact_simulated_ledger(
        filename: str,
        target: pd.Series,
        cost_bps: float,
    ) -> pd.DataFrame:
        expected = _simulate(prices, target, cost_bps=cost_bps)
        expected_bytes = _frame_csv_bytes(expected, float_format="%.17g")
        if snapshot.get(filename) != expected_bytes:
            raise DirectEdgeExperimentError(
                f"Sealed ledger does not replay from prices and targets: {filename}"
            )
        return expected

    benchmark_target = pd.Series(1.0, index=prices.index, name="target_exposure")
    benchmarks = {
        scenario_name: exact_simulated_ledger(
            f"buy_hold_{scenario_name}.csv",
            benchmark_target,
            bps,
        )
        for scenario_name, bps in COST_SCENARIOS
    }
    predictive_by_family = (
        {
            family: _common_support_predictive_metrics(
                predictions, family, support
            )
            for family in MODEL_FAMILIES
        }
        if support.any()
        else {}
    )

    common_results: dict[str, dict[str, Any]] = {}
    for spec in CANDIDATE_SPECS:
        inputs = _prediction_inputs(
            predictions,
            spec.model_family,
            common_cash[spec.candidate_id],
            common_block_start[spec.candidate_id],
        )
        scenarios: dict[str, Any] = {}
        for scenario_name, bps in COST_SCENARIOS:
            strategy = exact_simulated_ledger(
                f"common_{spec.candidate_id}_{scenario_name}_strategy.csv",
                common_targets[spec.candidate_id],
                bps,
            )
            scored = _score_strategy(
                strategy,
                benchmarks[scenario_name],
                prediction_inputs=inputs,
                cost_bps=bps,
            )
            if support.any():
                scored["common_support_predictive_metrics"] = dict(
                    predictive_by_family[spec.model_family]
                )
            scenarios[scenario_name] = scored
        common_results[spec.candidate_id] = scenarios

    replayed: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        inputs = _prediction_inputs(
            predictions,
            spec.model_family,
            actual_cash[spec.candidate_id],
            actual_block_start[spec.candidate_id],
        )
        scenarios: dict[str, Any] = {}
        for scenario_name, bps in COST_SCENARIOS:
            strategy = exact_simulated_ledger(
                f"{spec.candidate_id}_{scenario_name}_strategy.csv",
                full_targets[spec.candidate_id],
                bps,
            )
            scenarios[scenario_name] = _score_strategy(
                strategy,
                benchmarks[scenario_name],
                prediction_inputs=inputs,
                cost_bps=bps,
            )
        both_costs = bool(
            all(item["gates"]["passed"] for item in scenarios.values())
        )
        if spec.model_family == "market_sentiment" and support.any():
            ablation = common_support_ablation(
                common_results,
                gate_id=spec.candidate.candidate_id,
            )
        elif spec.model_family == "market_sentiment":
            ablation = {
                "applicable": False,
                "reason": "no_sentiment_ready_oof_rows",
                "passed": False,
            }
        else:
            ablation = {
                "applicable": False,
                "reason": "price-only is the frozen ablation baseline",
                "passed": True,
            }
        replayed.append(
            {
                "candidate": spec.to_dict(),
                "stage": "development_only",
                "scenarios": scenarios,
                "both_cost_gates_passed": both_costs,
                "common_support_sentiment_ablation": ablation,
                "passed": bool(both_costs and ablation["passed"]),
            }
        )
    return replayed


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
            raise DirectEdgeExperimentError(
                f"Recorded source commit lacks required file: {path}"
            )
        hashes[path] = hashlib.sha256(bytes(blob.stdout)).hexdigest()
    return hashes


def _verify_recorded_source_identity(
    snapshot: Mapping[str, bytes],
    *,
    repo_root: Path,
    artifact_commit: str,
) -> str:
    provenance = _parse_json_snapshot(snapshot, "input_provenance.json")
    identity = provenance.get("source_identity") if isinstance(provenance, dict) else None
    if not isinstance(identity, dict):
        raise DirectEdgeExperimentError("Artifact lacks recorded source identity")
    git_identity = identity.get("git")
    recorded_hashes = identity.get("source_hashes")
    if not isinstance(git_identity, dict) or not isinstance(recorded_hashes, dict):
        raise DirectEdgeExperimentError(
            "Artifact source identity lacks Git state or source hashes"
        )
    source_commit = str(git_identity.get("commit") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise DirectEdgeExperimentError("Artifact has an invalid source Git commit")
    if git_identity.get("dirty") is not False:
        raise DirectEdgeExperimentError("Artifact was not produced from clean source")
    if not _git_commit_is_ancestor(repo_root, source_commit, artifact_commit):
        raise DirectEdgeExperimentError(
            "Recorded source commit is not an ancestor of the artifact commit"
        )
    committed_hashes = _source_hashes_at_commit(repo_root, source_commit)
    if committed_hashes != recorded_hashes:
        raise DirectEdgeExperimentError(
            "Recorded source hashes do not match the recorded source commit"
        )
    return source_commit


def verify_development_artifact(
    artifact: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    run_dir = artifact.resolve()
    if run_dir.is_file():
        if run_dir.name != "report.json":
            raise DirectEdgeExperimentError("Verification file must be report.json")
        run_dir = run_dir.parent
    snapshot, verified_git_commit = _artifact_snapshot(
        run_dir, repo_root=repo_root
    )
    checksums = _verify_checksum_snapshot(snapshot)
    verified_source_commit: str | None = None
    if repo_root is not None:
        if not verified_git_commit:  # pragma: no cover - defensive
            raise DirectEdgeExperimentError("Committed verification lacks Git HEAD")
        verified_source_commit = _verify_recorded_source_identity(
            snapshot,
            repo_root=repo_root.resolve(),
            artifact_commit=verified_git_commit,
        )
    report = _parse_json_snapshot(snapshot, "report.json")
    manifest = _parse_json_snapshot(snapshot, "selection_manifest.json")
    results = _parse_json_snapshot(snapshot, "candidate_results.json")
    if not isinstance(report, dict) or not isinstance(manifest, dict):
        raise DirectEdgeExperimentError("Development report and manifest must be objects")
    if (
        report.get("contract_version") != CONTRACT_VERSION
        or report.get("stage") != "development"
        or report.get("physical_data_end") != "2018-12-31"
        or report.get("post_2018_market_data_accessed") is not False
        or manifest.get("contract_version") != CONTRACT_VERSION
        or manifest.get("stage") != "development"
    ):
        raise DirectEdgeExperimentError("Artifact violates the pre-2019 identity")
    if snapshot.get(".gitattributes") != ARTIFACT_GIT_ATTRIBUTES:
        raise DirectEdgeExperimentError("Artifact byte-preservation rule is altered")
    manifest_without_hash = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if manifest.get("manifest_sha256") != _sha256_tagged(
        _canonical_json_bytes(manifest_without_hash)
    ):
        raise DirectEdgeExperimentError("Selection manifest hash is invalid")
    if manifest.get("candidate_results_sha256") != _sha256_tagged(
        _canonical_json_bytes(results)
    ):
        raise DirectEdgeExperimentError("Candidate results are not bound by manifest")
    if report.get("selection_manifest") != manifest:
        raise DirectEdgeExperimentError("Report and manifest differ")
    if report.get("candidate_results_sha256") != manifest.get(
        "candidate_results_sha256"
    ):
        raise DirectEdgeExperimentError("Report does not bind the candidate results")
    if not isinstance(results, list) or len(results) != len(CANDIDATE_SPECS):
        raise DirectEdgeExperimentError("Candidate result set is not the frozen eight")
    expected_ids = [item.candidate_id for item in CANDIDATE_SPECS]
    observed_ids = [str(item.get("candidate", {}).get("candidate_id")) for item in results]
    if observed_ids != expected_ids:
        raise DirectEdgeExperimentError("Candidate result order or identity changed")
    for item in results:
        if set(item.get("scenarios", {})) != {name for name, _ in COST_SCENARIOS}:
            raise DirectEdgeExperimentError("A candidate lacks both cost scenarios")
        both_costs = bool(
            all(
                bool(scenario.get("gates", {}).get("passed"))
                for scenario in item["scenarios"].values()
            )
        )
        if bool(item.get("both_cost_gates_passed")) != both_costs:
            raise DirectEdgeExperimentError(
                "Candidate two-cost pass flag does not replay from its gates"
            )
        family = str(item.get("candidate", {}).get("model_family"))
        ablation = item.get("common_support_sentiment_ablation", {})
        ablation_pass = True if family == "price_only" else bool(ablation.get("passed"))
        if bool(item.get("passed")) != bool(both_costs and ablation_pass):
            raise DirectEdgeExperimentError(
                "Candidate pass flag disagrees with its cost and ablation gates"
            )
    selected = select_development_candidate(results)
    if selected != manifest.get("selected_candidate_id"):
        raise DirectEdgeExperimentError("Sealed selection does not replay")
    if bool(report.get("development_pass")) != (selected is not None):
        raise DirectEdgeExperimentError("Development pass disagrees with selection")
    if manifest.get("source_exclusion_audits") != {
        key: dict(value) for key, value in SOURCE_EXCLUSION_AUDITS.items()
    }:
        raise DirectEdgeExperimentError("Macro/news exclusion audits are absent or altered")

    payload_hashes = manifest.get("payload_sha256")
    expected_payload_names = set(snapshot) - {
        ".gitattributes",
        "checksums.json",
        "selection_manifest.json",
        "report.json",
    }
    if not isinstance(payload_hashes, dict) or set(payload_hashes) != expected_payload_names:
        raise DirectEdgeExperimentError("Manifest payload map is not exact")
    for name, expected in payload_hashes.items():
        if _sha256_tagged(snapshot[name]) != expected:
            raise DirectEdgeExperimentError(f"Manifest payload mismatch: {name}")

    for name, date_column in (
        ("development_prices_through_2018.csv", "date"),
        ("development_context_through_2018.csv", "date"),
        ("oof_predictions_2005_2018.csv", "decision_date"),
        ("candidate_targets_2005_2018.csv", "decision_date"),
    ):
        try:
            frame = pd.read_csv(io.BytesIO(snapshot[name]))
            dates = pd.to_datetime(frame[date_column], errors="raise")
        except (KeyError, ValueError) as exc:
            raise DirectEdgeExperimentError(f"Could not verify bounded dates in {name}") from exc
        if bool((dates > DEVELOPMENT_END).any()):
            raise DirectEdgeExperimentError(f"Sealed artifact has a post-2018 row: {name}")

    required_ledgers = {
        f"buy_hold_{scenario}.csv" for scenario, _ in COST_SCENARIOS
    } | {
        f"{spec.candidate_id}_{scenario}_strategy.csv"
        for spec in CANDIDATE_SPECS
        for scenario, _ in COST_SCENARIOS
    } | {
        f"common_{spec.candidate_id}_{scenario}_strategy.csv"
        for spec in CANDIDATE_SPECS
        for scenario, _ in COST_SCENARIOS
    }
    if not required_ledgers.issubset(snapshot):
        raise DirectEdgeExperimentError("Artifact does not contain every evaluated ledger")
    for name in required_ledgers:
        try:
            ledger = pd.read_csv(io.BytesIO(snapshot[name]))
            fill_dates = pd.to_datetime(ledger["fill_date"], errors="raise")
            decision_dates = pd.to_datetime(ledger["decision_date"], errors="raise")
        except (KeyError, ValueError) as exc:
            raise DirectEdgeExperimentError(
                f"Could not verify bounded ledger dates in {name}"
            ) from exc
        if bool((fill_dates > DEVELOPMENT_END).any()) or bool(
            (decision_dates > DEVELOPMENT_END).any()
        ):
            raise DirectEdgeExperimentError(
                f"Sealed artifact has a post-2018 ledger row: {name}"
            )

    replayed_results = _replay_sealed_candidate_results(snapshot)
    if _canonical_json_bytes(replayed_results) != _canonical_json_bytes(results):
        raise DirectEdgeExperimentError(
            "Sealed candidate results do not replay from targets and ledgers"
        )
    replayed_selected = select_development_candidate(replayed_results)
    if replayed_selected != selected:
        raise DirectEdgeExperimentError(
            "Replayed candidate selection differs from the sealed selection"
        )

    refit = manifest.get("refit", {})
    model_files = {name for name in snapshot if name.startswith("final_")}
    if selected is None:
        if model_files or refit.get("performed") is not False:
            raise DirectEdgeExperimentError("Null selection must not contain a final refit")
    elif refit.get("performed") is not True or not model_files:
        raise DirectEdgeExperimentError("Selected candidate lacks its deterministic refit")
    return {
        "verified": True,
        "run_id": report.get("run_id"),
        "development_pass": bool(report.get("development_pass")),
        "selected_candidate_id": selected,
        "artifact_dir": str(run_dir),
        "verified_git_commit": verified_git_commit,
        "verified_source_commit": verified_source_commit,
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
    """Run and seal the complete, physically pre-2019 development experiment."""

    deadline = _deadline or _Deadline(clock)
    data, context, boundary_audit = _bounded_inputs(price_frame, context_frame)
    deadline.check("bounded input validation")
    feature_label = build_direct_edge_feature_label_frame(data, context)
    if feature_label.index.max() > DEVELOPMENT_END:
        raise DirectEdgeExperimentError("Feature construction produced post-2018 rows")
    deadline.check("feature and label construction")
    predictions = build_pre2019_direct_edge_predictions(feature_label)
    if predictions.index.max() > DEVELOPMENT_END:
        raise DirectEdgeExperimentError("Walk-forward produced post-2018 predictions")
    deadline.check("fourteen frozen fold fits")

    support = _common_support_mask(predictions)
    target_audit = pd.DataFrame(index=predictions.index)
    target_audit.index.name = "decision_date"
    target_audit["market_sentiment_common_support"] = support
    actual_cash: dict[str, np.ndarray] = {}
    actual_block_start: dict[str, np.ndarray] = {}
    common_cash: dict[str, np.ndarray] = {}
    common_block_start: dict[str, np.ndarray] = {}
    full_targets: dict[str, pd.Series] = {}
    common_targets: dict[str, pd.Series] = {}
    for spec in CANDIDATE_SPECS:
        cash = predictions[spec.cash_target_column].to_numpy(dtype=np.int8)
        block_start = predictions[spec.cash_block_start_column].to_numpy(
            dtype=np.int8
        )
        common, common_start = _common_support_cash_target(
            predictions, spec, support
        )
        actual_cash[spec.candidate_id] = cash
        actual_block_start[spec.candidate_id] = block_start
        common_cash[spec.candidate_id] = common
        common_block_start[spec.candidate_id] = common_start
        target_audit[f"{spec.candidate_id}_cash"] = cash
        target_audit[f"{spec.candidate_id}_cash_block_start"] = block_start
        target_audit[f"{spec.candidate_id}_common_support_cash"] = common
        target_audit[
            f"{spec.candidate_id}_common_support_cash_block_start"
        ] = common_start
        full_targets[spec.candidate_id] = _full_exposure_target(
            data.index, predictions.index, cash
        )
        common_targets[spec.candidate_id] = _full_exposure_target(
            data.index, predictions.index, common
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
    common_support_error: str | None = None
    if support.any():
        predictive_by_family = {
            family: _common_support_predictive_metrics(
                predictions, family, support
            )
            for family in MODEL_FAMILIES
        }
        for spec in CANDIDATE_SPECS:
            inputs = _prediction_inputs(
                predictions,
                spec.model_family,
                common_cash[spec.candidate_id],
                common_block_start[spec.candidate_id],
            )
            scenarios: dict[str, Any] = {}
            for scenario_name, bps in COST_SCENARIOS:
                strategy = _simulate(
                    data, common_targets[spec.candidate_id], cost_bps=bps
                )
                scored = _score_strategy(
                    strategy,
                    benchmarks[scenario_name],
                    prediction_inputs=inputs,
                    cost_bps=bps,
                )
                scored["common_support_predictive_metrics"] = dict(
                    predictive_by_family[spec.model_family]
                )
                scenarios[scenario_name] = scored
                ledger_frames[
                    f"common_{spec.candidate_id}_{scenario_name}_strategy.csv"
                ] = strategy
                deadline.check(
                    f"{spec.candidate_id} {scenario_name} common-support simulation"
                )
            common_results[spec.candidate_id] = scenarios
    else:
        common_support_error = "no_sentiment_ready_oof_rows"
        # Seal explicit all-LONG common-support ledgers even when the common
        # population is empty; this preserves the exact attempted comparison.
        for spec in CANDIDATE_SPECS:
            for scenario_name, bps in COST_SCENARIOS:
                ledger_frames[
                    f"common_{spec.candidate_id}_{scenario_name}_strategy.csv"
                ] = _simulate(data, common_targets[spec.candidate_id], cost_bps=bps)

    candidate_results: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        inputs = _prediction_inputs(
            predictions,
            spec.model_family,
            actual_cash[spec.candidate_id],
            actual_block_start[spec.candidate_id],
        )
        scenarios: dict[str, Any] = {}
        for scenario_name, bps in COST_SCENARIOS:
            strategy = _simulate(data, full_targets[spec.candidate_id], cost_bps=bps)
            scenarios[scenario_name] = _score_strategy(
                strategy,
                benchmarks[scenario_name],
                prediction_inputs=inputs,
                cost_bps=bps,
            )
            ledger_frames[f"{spec.candidate_id}_{scenario_name}_strategy.csv"] = strategy
            deadline.check(f"{spec.candidate_id} {scenario_name} simulation")
        both_costs = bool(all(item["gates"]["passed"] for item in scenarios.values()))
        if spec.model_family == "market_sentiment" and support.any():
            ablation = common_support_ablation(
                common_results, gate_id=spec.candidate.candidate_id
            )
        elif spec.model_family == "market_sentiment":
            ablation = {
                "applicable": False,
                "reason": common_support_error,
                "passed": False,
            }
        else:
            ablation = {
                "applicable": False,
                "reason": "price-only is the frozen ablation baseline",
                "passed": True,
            }
        candidate_results.append(
            {
                "candidate": spec.to_dict(),
                "stage": "development_only",
                "scenarios": scenarios,
                "both_cost_gates_passed": both_costs,
                "common_support_sentiment_ablation": ablation,
                "passed": bool(both_costs and ablation["passed"]),
            }
        )

    selected_id = select_development_candidate(candidate_results)
    selected_spec = next(
        (item for item in CANDIDATE_SPECS if item.candidate_id == selected_id), None
    )
    model_payloads, refit = _refit_selected_models(
        feature_label, selected_spec, deadline=deadline
    )
    deadline.check("selected-model deterministic refit")

    created_at = datetime.now(timezone.utc)
    resolved_run_id = run_id or (
        f"direct-edge-development-{created_at:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resolved_run_id):
        raise DirectEdgeExperimentError("run_id is not a safe flat directory name")
    run_dir = output_dir.resolve() / resolved_run_id
    temporary_run_dir = output_dir.resolve() / (
        f".{resolved_run_id}.{uuid.uuid4().hex}.sealing"
    )
    if run_dir.exists():
        raise DirectEdgeExperimentError(
            f"Artifact directory already exists: {run_dir}"
        )

    price_payload = _frame_csv_bytes(data.reset_index(names="date"))
    context_payload = _frame_csv_bytes(context.reset_index(names="date"))
    predictions_payload = _frame_csv_bytes(predictions.reset_index())
    targets_payload = _frame_csv_bytes(target_audit.reset_index())
    results_payload = _pretty_json_bytes(candidate_results)
    provenance = {
        "contract_version": CONTRACT_VERSION,
        "source": dict(input_provenance or {"source_type": "injected_pre2019_inputs"}),
        "source_identity": dict(source_identity or {}),
        "boundary_audit": boundary_audit,
        "physical_data_end": "2018-12-31",
        "post_2018_market_data_accessed": False,
        "network_access": False,
    }
    payloads: dict[str, bytes] = {
        "development_prices_through_2018.csv": price_payload,
        "development_context_through_2018.csv": context_payload,
        "feature_provenance.json": _pretty_json_bytes(_feature_provenance()),
        "input_provenance.json": _pretty_json_bytes(provenance),
        "oof_predictions_2005_2018.csv": predictions_payload,
        "candidate_targets_2005_2018.csv": targets_payload,
        "candidate_results.json": results_payload,
        **model_payloads,
    }
    for filename, ledger in sorted(ledger_frames.items()):
        payloads[filename] = _frame_csv_bytes(ledger, float_format="%.17g")

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
        "walk_forward_folds": [asdict(item) for item in WALK_FORWARD_FOLDS],
        "gam_config": asdict(DirectEdgeGAMConfig()),
        "cost_scenarios_bps": {name: bps for name, bps in COST_SCENARIOS},
        "selected_candidate_id": selected_id,
        "selection_tie_break": [
            "highest 10bps minimum-fold active-log edge",
            "highest 10bps total active-log edge",
            "fewest 10bps cash episodes",
            "lexical candidate id",
        ],
        "market_sentiment_selection_requirement": (
            "on identical sentiment-ready rows: strictly better Brier, strictly lower edge MAE, "
            "strictly higher total active edge, and no worse weakest fold at 5 and 10 bps"
        ),
        "source_exclusion_audits": {
            key: dict(value) for key, value in SOURCE_EXCLUSION_AUDITS.items()
        },
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
            "gam_configuration_may_change_after_this_run": False,
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
        "refit": refit,
        "reproducibility": {
            "runtime_seconds_before_seal": deadline.elapsed(),
            "runtime_limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "completed_before_seal_within_limit": deadline.elapsed() <= RUN_TIME_LIMIT_SECONDS,
            "model_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "api_cost_display": "$0.00",
        },
    }
    payloads["report.json"] = _pretty_json_bytes(report)
    promoted = False
    try:
        _seal_artifact_bundle(temporary_run_dir, payloads)
        deadline.check("artifact checksum readback")
        verify_development_artifact(temporary_run_dir)
        deadline.check("completed development verification")
        temporary_run_dir.replace(run_dir)
        promoted = True
        deadline.check("atomic artifact promotion")
    except Exception:
        if temporary_run_dir.exists():
            shutil.rmtree(temporary_run_dir)
        if promoted and run_dir.exists():
            shutil.rmtree(run_dir)
        raise
    return report


def run_development_experiment(
    *,
    repo_root: Path,
    price_artifact: Path,
    context_parquet: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run from clean committed source, committed prices, and bounded context."""

    deadline = _Deadline(time.perf_counter)
    repo = repo_root.resolve()
    source_paths = [repo / path for path in SOURCE_FILES]
    repository = validate_source_repository(repo, source_paths)
    deadline.check("source repository validation")
    git = _git_state(repo)
    if git.get("dirty") is not False:
        raise DirectEdgeExperimentError(
            "Development requires committed source and a clean worktree"
        )
    source_commit = str(git.get("commit") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit):
        raise DirectEdgeExperimentError("Could not capture clean source Git commit")
    # Bind to immutable Git blobs, not checkout bytes: Windows line-ending
    # conversion can make a clean working tree byte-different from its blob.
    source_hashes = _source_hashes_at_commit(repo, source_commit)
    deadline.check("source hashing")
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
        description="Run or verify the sealed through-2018 direct-edge experiment"
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
        if any(value is not None for value in (args.price_artifact, args.context_parquet, args.output_dir)):
            raise SystemExit("verify does not accept run inputs")
        verification = verify_development_artifact(
            args.development_artifact,
            repo_root=args.repo_root,
        )
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 0
    if any(
        value is None
        for value in (args.repo_root, args.price_artifact, args.context_parquet, args.output_dir)
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
    "CandidateSpec",
    "DirectEdgeExperimentError",
    "DirectEdgeExperimentTimeout",
    "LoadedDevelopmentInputs",
    "RUN_TIME_LIMIT_SECONDS",
    "SOURCE_EXCLUSION_AUDITS",
    "STRESS_COST_BPS",
    "common_support_ablation",
    "load_bounded_context_parquet",
    "load_public_development_inputs",
    "run_development_experiment",
    "run_development_from_inputs",
    "select_development_candidate",
    "verify_development_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
