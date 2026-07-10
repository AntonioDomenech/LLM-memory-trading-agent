"""Frozen pre-2019 walk-forward predictions for regime consensus.

The policy owns chronology and decision rules, while :mod:`regime_hmm` owns
the deterministic two-state head.  Each fixed two-year entry-fill block fits
three independent heads once, using only outcomes that matured strictly
before the block's first decision.  Validation features are then consumed by
causal forward filtering; validation outcomes never enter the frozen models.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from .regime_consensus_features import (
    ALL_HEADS_READY_COLUMN,
    HEAD_FEATURE_COLUMNS,
    HEAD_NAMES,
    HEAD_ORIENTATIONS,
    HEAD_READINESS_COLUMNS,
    REGIME_FEATURE_COLUMNS,
    REGIME_LABEL_COLUMNS,
    entry_fill_year_mask,
    regime_head_readiness,
    strict_pre_fold_head_training_mask,
    strict_pre_fold_label_mask,
)
from .regime_hmm import RegimeHeadModel


DEVELOPMENT_FIRST_ENTRY_YEAR: Final[int] = 2005
DEVELOPMENT_LAST_ENTRY_YEAR: Final[int] = 2018
DEVELOPMENT_LAST_DECISION: Final[pd.Timestamp] = pd.Timestamp("2018-12-31")
EXPECTED_EDGE_GATE_10BPS: Final[float] = 0.0005
MODEL_VARIANTS: Final[tuple[str, str]] = ("regime_consensus", "aapl_only")


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    first_entry_year: int
    last_entry_year: int

    def decision_mask(self, frame: pd.DataFrame) -> pd.Series:
        return entry_fill_year_mask(
            frame,
            first_year=self.first_entry_year,
            last_year=self.last_entry_year,
        )


@dataclass(frozen=True)
class RegimeCandidate:
    candidate_id: str
    probability_gate: float
    expected_edge_gate: float = EXPECTED_EDGE_GATE_10BPS


@dataclass(frozen=True)
class RegimeWalkForwardResult:
    """Predictions plus all 21 exact fold/head model states."""

    predictions: pd.DataFrame
    model_states: tuple[Mapping[str, Any], ...]


WALK_FORWARD_FOLDS: Final[tuple[WalkForwardFold, ...]] = tuple(
    WalkForwardFold(f"{year}-{year + 1}", year, year + 1)
    for year in range(DEVELOPMENT_FIRST_ENTRY_YEAR, DEVELOPMENT_LAST_ENTRY_YEAR, 2)
)

REGIME_CANDIDATES: Final[tuple[RegimeCandidate, ...]] = (
    RegimeCandidate("p55_e5", 0.55),
    RegimeCandidate("p60_e5", 0.60),
)

_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *REGIME_FEATURE_COLUMNS,
        *REGIME_LABEL_COLUMNS,
        *HEAD_READINESS_COLUMNS,
        ALL_HEADS_READY_COLUMN,
    }
)


def candidate_cash_target_column(model_variant: str, candidate_id: str) -> str:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown regime model variant: {model_variant!r}")
    if candidate_id not in {candidate.candidate_id for candidate in REGIME_CANDIDATES}:
        raise ValueError(f"Unknown frozen regime candidate: {candidate_id!r}")
    return f"cash_target_{model_variant}_{candidate_id}"


def _canonical_frame(feature_label_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_label_frame, pd.DataFrame):
        raise TypeError("feature_label_frame must be a pandas DataFrame")
    if feature_label_frame.empty:
        raise ValueError("feature_label_frame cannot be empty")
    missing = sorted(_REQUIRED_COLUMNS.difference(feature_label_frame.columns))
    if missing:
        raise ValueError(f"feature_label_frame is missing required columns: {missing}")

    try:
        index = pd.DatetimeIndex(pd.to_datetime(feature_label_frame.index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("feature_label_frame index must contain valid dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise ValueError("feature_label_frame contains duplicate decision dates")
    if not index.is_monotonic_increasing:
        raise ValueError("feature_label_frame must be in chronological order")
    if bool((index > DEVELOPMENT_LAST_DECISION).any()):
        first = index[index > DEVELOPMENT_LAST_DECISION][0]
        raise ValueError(
            "Pre-2019 regime walk-forward refuses post-2018 decision rows; "
            f"first forbidden date is {first.date().isoformat()}"
        )

    result = feature_label_frame.copy()
    result.index = index
    result.index.name = "decision_date"
    expected_readiness = regime_head_readiness(result)
    for column in (*HEAD_READINESS_COLUMNS, ALL_HEADS_READY_COLUMN):
        actual = result[column]
        if actual.isna().any() or not pd.api.types.is_bool_dtype(actual.dtype):
            raise ValueError(f"{column} must contain non-missing booleans")
        if not actual.astype(bool).equals(expected_readiness[column]):
            raise ValueError(f"{column} is inconsistent with its feature inputs")
    return result


def _sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode("ascii")).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Regime model state must be canonical JSON-compatible") from exc


def _model_hash(model: RegimeHeadModel) -> str:
    value = model.model_sha256
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Regime head returned an invalid SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("Regime head returned an invalid SHA-256") from exc
    return value


def _canonical_state_record(
    model: RegimeHeadModel,
    *,
    fold_id: str,
    head_name: str,
) -> dict[str, Any]:
    raw = model.to_state()
    if not isinstance(raw, Mapping):
        raise ValueError("Regime head state must be a mapping")
    # Canonical JSON round-trip both detaches mutable model internals and proves
    # the state can be sealed without Python-specific objects.
    payload = json.loads(_canonical_json(raw))
    state_sha256 = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return {
        "fold_id": fold_id,
        "head_name": head_name,
        "model_sha256": _model_hash(model),
        "canonical_state_sha256": state_sha256,
        "state": payload,
    }


def _fit_metadata(
    frame: pd.DataFrame,
    history_mask: np.ndarray,
    outcome_mask: np.ndarray,
) -> dict[str, Any]:
    history = frame.loc[history_mask]
    outcomes = frame.loc[outcome_mask]
    maturity = pd.to_datetime(outcomes["label_maturity_date"], errors="raise")
    binary = np.full(len(history), np.nan, dtype=float)
    edge = np.full(len(history), np.nan, dtype=float)
    history_positions = np.flatnonzero(history_mask)
    eligible_positions = np.flatnonzero(outcome_mask)
    position_lookup = {
        global_position: local
        for local, global_position in enumerate(history_positions)
    }
    local_outcome = np.asarray(
        [position_lookup[int(position)] for position in eligible_positions],
        dtype=int,
    )
    binary[local_outcome] = outcomes["cash_beats_long_10bps"].to_numpy(dtype=float)
    edge[local_outcome] = outcomes["cash_active_log_edge_10bps"].to_numpy(dtype=float)

    def target_token(value: float) -> str:
        return float(value).hex() if math.isfinite(float(value)) else "missing"

    return {
        "training_start_date": history.index.min().date().isoformat(),
        "training_end_date": history.index.max().date().isoformat(),
        "maximum_label_maturity_date": maturity.max().date().isoformat(),
        "training_dates_sha256": _sha256_lines(
            [timestamp.date().isoformat() for timestamp in history.index]
        ),
        "binary_target_sha256": _sha256_lines(
            [target_token(value) for value in binary]
        ),
        "edge_target_sha256": _sha256_lines([target_token(value) for value in edge]),
    }


def _fit_head(
    frame: pd.DataFrame,
    *,
    fold: WalkForwardFold,
    first_decision: pd.Timestamp,
    head_name: str,
    feature_columns: Sequence[str],
    stress_orientations: Sequence[int],
) -> tuple[RegimeHeadModel, dict[str, Any], np.ndarray, np.ndarray]:
    history_mask = np.asarray(frame.index < first_decision, dtype=bool)
    if not history_mask.any():
        raise ValueError(f"No emission history is available for fold {fold.fold_id}")
    outcome_mask = strict_pre_fold_head_training_mask(
        frame,
        fold_first_decision_date=first_decision,
        head_name=head_name,
    ).to_numpy(dtype=bool)
    if not outcome_mask.any():
        raise ValueError(
            f"No mature {head_name} outcomes are available for fold {fold.fold_id}"
        )
    if bool((outcome_mask & ~history_mask).any()):  # pragma: no cover - defensive
        raise RuntimeError("Outcome mask escaped the strict pre-fold emission history")

    history = frame.loc[history_mask, list(feature_columns)]
    binary = np.full(len(history), np.nan, dtype=float)
    edge = np.full(len(history), np.nan, dtype=float)
    history_positions = np.flatnonzero(history_mask)
    eligible_positions = np.flatnonzero(outcome_mask)
    position_lookup = {global_position: local for local, global_position in enumerate(history_positions)}
    local_outcome = np.asarray(
        [position_lookup[int(position)] for position in eligible_positions], dtype=int
    )
    binary[local_outcome] = frame.iloc[eligible_positions][
        "cash_beats_long_10bps"
    ].to_numpy(dtype=float)
    edge[local_outcome] = frame.iloc[eligible_positions][
        "cash_active_log_edge_10bps"
    ].to_numpy(dtype=float)

    metadata = _fit_metadata(
        frame,
        history_mask,
        outcome_mask,
    )
    maturity = pd.to_datetime(
        frame.loc[outcome_mask, "label_maturity_date"], errors="raise"
    )
    audit = {
        "emission_start_date": history.index.min().date().isoformat(),
        "emission_end_date": history.index.max().date().isoformat(),
        "emission_row_count": int(len(history)),
        "complete_emission_count": int(
            np.isfinite(history.to_numpy(dtype=float)).all(axis=1).sum()
        ),
        "outcome_training_count": int(outcome_mask.sum()),
        "maximum_label_maturity_date": maturity.max().date().isoformat(),
    }
    model = RegimeHeadModel().fit(
        history,
        binary,
        edge,
        head_name=head_name,
        feature_names=tuple(feature_columns),
        stress_orientations=tuple(stress_orientations),
        fit_metadata=metadata,
    )
    _model_hash(model)
    return model, audit, history_mask, outcome_mask


def _predict_components(
    model: RegimeHeadModel,
    chronological_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = model.predict_components(
        chronological_features,
        continue_from_training=True,
    )
    if not isinstance(raw, Mapping):
        raise ValueError("Regime head prediction components must be a mapping")
    stress = np.asarray(raw["stress_probability"], dtype=float)
    probability = np.asarray(raw["cash_beats_long_probability"], dtype=float)
    edge = np.asarray(raw["expected_edge_10bps"], dtype=float)
    available = np.asarray(raw["observation_available"])
    expected_shape = (len(chronological_features),)
    if (
        stress.shape != expected_shape
        or probability.shape != expected_shape
        or edge.shape != expected_shape
        or available.shape != expected_shape
    ):
        raise ValueError("Regime head prediction components have invalid shapes")
    if any(not isinstance(value, (bool, np.bool_)) for value in available):
        raise ValueError("Regime head observation availability must be boolean")
    available = available.astype(bool)
    for name, values in (
        ("stress_probability", stress),
        ("cash_beats_long_probability", probability),
        ("expected_edge_10bps", edge),
    ):
        if not np.isfinite(values[available]).all() or not np.isnan(
            values[~available]
        ).all():
            raise ValueError(
                f"Regime head {name} must be finite exactly on available rows"
            )
    if (
        ((stress[available] < 0.0) | (stress[available] > 1.0)).any()
        or (
            (probability[available] < 0.0)
            | (probability[available] > 1.0)
        ).any()
    ):
        raise ValueError("Regime head probabilities must lie in [0, 1]")
    return stress, probability, edge, available


def _predict_fold(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
) -> tuple[pd.DataFrame, tuple[Mapping[str, Any], ...]]:
    fold_mask = fold.decision_mask(frame).to_numpy(dtype=bool)
    if not fold_mask.any():
        raise ValueError(f"No decision rows are available for fixed fold {fold.fold_id}")
    fill = frame.loc[fold_mask].copy()
    first_decision = fill.index[0]
    last_decision = fill.index[-1]

    baseline_mask = strict_pre_fold_label_mask(
        frame, fold_first_decision_date=first_decision
    ).to_numpy(dtype=bool)
    if not baseline_mask.any():
        raise ValueError(f"No mature baseline labels are available for fold {fold.fold_id}")
    baseline_binary = frame.loc[baseline_mask, "cash_beats_long_10bps"].to_numpy(dtype=float)
    baseline_edge = frame.loc[baseline_mask, "cash_active_log_edge_10bps"].to_numpy(dtype=float)
    baseline_probability = float((baseline_binary.sum() + 1.0) / (len(baseline_binary) + 2.0))
    baseline_mean_edge = float(baseline_edge.mean())

    output = pd.DataFrame(index=fill.index)
    output.index.name = "decision_date"
    output["fold_id"] = fold.fold_id
    output["fold_first_decision_date"] = first_decision
    output["fold_last_decision_date"] = last_decision
    output["fold_first_entry_year"] = fold.first_entry_year
    output["fold_last_entry_year"] = fold.last_entry_year
    output["causal_baseline_training_count"] = int(len(baseline_binary))
    output["causal_baseline_positive_count"] = int(baseline_binary.sum())
    output["causal_prevalence_probability_10bps"] = baseline_probability
    output["causal_training_mean_edge_10bps"] = baseline_mean_edge
    baseline_maturity = pd.to_datetime(
        frame.loc[baseline_mask, "label_maturity_date"], errors="raise"
    )
    output["causal_baseline_max_label_maturity_date"] = baseline_maturity.max()

    # The model persisted the terminal causal posterior from the exact
    # pre-fold history.  Filtering starts with the validation block so that
    # historical observations are not counted twice.
    states: list[Mapping[str, Any]] = []
    for head_name, feature_columns, orientations, readiness_column in zip(
        HEAD_NAMES,
        HEAD_FEATURE_COLUMNS,
        HEAD_ORIENTATIONS,
        HEAD_READINESS_COLUMNS,
        strict=True,
    ):
        model, metadata, _, _ = _fit_head(
            frame,
            fold=fold,
            first_decision=first_decision,
            head_name=head_name,
            feature_columns=feature_columns,
            stress_orientations=orientations,
        )
        model_hash = _model_hash(model)
        sequence = fill.loc[:, list(feature_columns)]
        stress, probability, edge, available = _predict_components(model, sequence)
        expected_available = fill[readiness_column].to_numpy(dtype=bool)
        if not np.array_equal(available, expected_available):
            raise ValueError(
                f"{head_name} observation availability disagrees with frozen readiness"
            )
        output[f"{head_name}_stress_probability"] = stress
        output[f"{head_name}_cash_win_probability_10bps"] = probability
        output[f"{head_name}_expected_edge_10bps"] = edge
        output[f"{head_name}_ready"] = expected_available
        output[f"{head_name}_model_sha256"] = model_hash
        output[f"{head_name}_emission_training_rows"] = int(
            metadata["emission_row_count"]
        )
        output[f"{head_name}_outcome_training_rows"] = int(
            metadata["outcome_training_count"]
        )
        output[f"{head_name}_training_max_label_maturity_date"] = pd.Timestamp(
            metadata["maximum_label_maturity_date"]
        )
        output[f"{head_name}_training_emission_end_date"] = pd.Timestamp(
            metadata["emission_end_date"]
        )
        states.append(
            _canonical_state_record(model, fold_id=fold.fold_id, head_name=head_name)
        )

    probability_columns = [
        f"{head_name}_cash_win_probability_10bps" for head_name in HEAD_NAMES
    ]
    edge_columns = [f"{head_name}_expected_edge_10bps" for head_name in HEAD_NAMES]
    output["regime_consensus_cash_win_probability_10bps"] = output.loc[
        :, probability_columns
    ].mean(axis=1, skipna=False)
    output["regime_consensus_expected_edge_10bps"] = output.loc[
        :, edge_columns
    ].mean(axis=1, skipna=False)
    output["aapl_only_cash_win_probability_10bps"] = output[
        "aapl_state_cash_win_probability_10bps"
    ]
    output["aapl_only_expected_edge_10bps"] = output[
        "aapl_state_expected_edge_10bps"
    ]
    output[ALL_HEADS_READY_COLUMN] = fill[ALL_HEADS_READY_COLUMN].to_numpy(dtype=bool)
    for name in REGIME_LABEL_COLUMNS:
        output[name] = fill[name]
    if "entry_fill_year" in fill.columns:
        output["entry_fill_year"] = fill["entry_fill_year"]
    return output, tuple(states)


def _add_frozen_targets(predictions: pd.DataFrame) -> None:
    support = predictions[ALL_HEADS_READY_COLUMN].to_numpy(dtype=bool)
    aapl_edge = predictions["aapl_state_expected_edge_10bps"].to_numpy(dtype=float)
    market_edge = predictions["market_state_expected_edge_10bps"].to_numpy(dtype=float)
    sentiment_edge = predictions["sentiment_state_expected_edge_10bps"].to_numpy(dtype=float)
    dates = pd.to_datetime(predictions["label_entry_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Every development decision must preserve its entry-fill date")
    first_fill_position = int(np.argmin(dates.to_numpy(dtype="datetime64[ns]")))
    final_fill_position = int(np.argmax(dates.to_numpy(dtype="datetime64[ns]")))

    for candidate in REGIME_CANDIDATES:
        full_probability = predictions[
            "regime_consensus_cash_win_probability_10bps"
        ].to_numpy(dtype=float)
        full_edge = predictions["regime_consensus_expected_edge_10bps"].to_numpy(dtype=float)
        full = (
            support
            & (full_probability >= candidate.probability_gate)
            & (full_edge >= candidate.expected_edge_gate)
            & (aapl_edge > 0.0)
            & ((market_edge > 0.0) | (sentiment_edge > 0.0))
        )
        aapl_probability = predictions[
            "aapl_only_cash_win_probability_10bps"
        ].to_numpy(dtype=float)
        aapl = (
            support
            & (aapl_probability >= candidate.probability_gate)
            & (aapl_edge >= candidate.expected_edge_gate)
        )
        for variant, values in (("regime_consensus", full), ("aapl_only", aapl)):
            target = values.astype(np.int8)
            target[first_fill_position] = 0
            target[final_fill_position] = 0
            predictions[candidate_cash_target_column(variant, candidate.candidate_id)] = target


def build_pre2019_regime_consensus_walkforward(
    feature_label_frame: pd.DataFrame,
) -> RegimeWalkForwardResult:
    """Fit 21 frozen heads and return causal 2005-2018 predictions."""

    frame = _canonical_frame(feature_label_frame)
    blocks: list[pd.DataFrame] = []
    states: list[Mapping[str, Any]] = []
    for fold in WALK_FORWARD_FOLDS:
        block, block_states = _predict_fold(frame, fold)
        blocks.append(block)
        states.extend(block_states)
    predictions = pd.concat(blocks, axis=0).sort_index(kind="mergesort")
    if predictions.index.has_duplicates:
        raise RuntimeError("Fixed regime walk-forward folds overlap")
    if len(states) != len(WALK_FORWARD_FOLDS) * len(HEAD_NAMES):
        raise RuntimeError("Frozen regime walk-forward did not produce 21 model states")
    _add_frozen_targets(predictions)
    return RegimeWalkForwardResult(predictions=predictions, model_states=tuple(states))


def build_pre2019_regime_consensus_predictions(
    feature_label_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience wrapper returning only the prediction ledger."""

    return build_pre2019_regime_consensus_walkforward(feature_label_frame).predictions


__all__ = [
    "DEVELOPMENT_FIRST_ENTRY_YEAR",
    "DEVELOPMENT_LAST_DECISION",
    "DEVELOPMENT_LAST_ENTRY_YEAR",
    "EXPECTED_EDGE_GATE_10BPS",
    "MODEL_VARIANTS",
    "REGIME_CANDIDATES",
    "WALK_FORWARD_FOLDS",
    "RegimeCandidate",
    "RegimeWalkForwardResult",
    "WalkForwardFold",
    "build_pre2019_regime_consensus_predictions",
    "build_pre2019_regime_consensus_walkforward",
    "candidate_cash_target_column",
]
