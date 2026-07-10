"""Frozen causal walk-forward for the one-session rare-loss forest."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd

from .rare_loss_features import (
    CORE_FEATURE_COLUMNS,
    RARE_LOSS_FEATURE_COLUMNS,
    entry_fill_year_mask,
    entry_fill_years,
    rare_loss_feature_readiness,
    strict_pre_fold_training_mask,
)
from .rare_loss_forest import (
    MINIMUM_OOB_TREE_COUNT,
    RareLossForest,
)


DEVELOPMENT_FIRST_ENTRY_YEAR: Final[int] = 2005
DEVELOPMENT_LAST_ENTRY_YEAR: Final[int] = 2018
DEVELOPMENT_LAST_DECISION: Final[pd.Timestamp] = pd.Timestamp("2018-12-31")
MODEL_VARIANTS: Final[tuple[str, str]] = ("full", "core")
ORDINARY_PROBABILITY_GATE: Final[float] = 0.55
EXPECTED_EDGE_GATE_10BPS: Final[float] = 0.001
SEVERE_PREVALENCE_MULTIPLIER: Final[float] = 2.0


@dataclass(frozen=True)
class RareLossFold:
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
class RareLossCandidate:
    candidate_id: str
    oob_tail_quantile: float


@dataclass(frozen=True)
class RareLossWalkForwardResult:
    predictions: pd.DataFrame
    model_states: tuple[Mapping[str, Any], ...]


WALK_FORWARD_FOLDS: Final[tuple[RareLossFold, ...]] = tuple(
    RareLossFold(f"{year}-{year + 1}", year, year + 1)
    for year in range(DEVELOPMENT_FIRST_ENTRY_YEAR, DEVELOPMENT_LAST_ENTRY_YEAR, 2)
)

RARE_LOSS_CANDIDATES: Final[tuple[RareLossCandidate, ...]] = (
    RareLossCandidate("tail975", 0.975),
    RareLossCandidate("tail990", 0.99),
)

_LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "label_entry_date",
    "label_maturity_date",
    "severe_loss_label_1session",
    "cash_beats_long_10bps",
    "cash_active_log_edge_10bps_clipped",
    "entry_fill_year",
    "features_ready",
)
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {*RARE_LOSS_FEATURE_COLUMNS, *_LABEL_COLUMNS, "label_available"}
)


def candidate_cash_target_column(model_variant: str, candidate_id: str) -> str:
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown rare-loss variant: {model_variant!r}")
    if candidate_id not in {item.candidate_id for item in RARE_LOSS_CANDIDATES}:
        raise ValueError(f"Unknown rare-loss candidate: {candidate_id!r}")
    return f"cash_target_{model_variant}_{candidate_id}"


def _canonical_frame(feature_label_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_label_frame, pd.DataFrame) or feature_label_frame.empty:
        raise ValueError("feature_label_frame must be a non-empty DataFrame")
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
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("feature_label_frame dates must be unique and chronological")
    if bool((index > DEVELOPMENT_LAST_DECISION).any()):
        first = index[index > DEVELOPMENT_LAST_DECISION][0]
        raise ValueError(
            "Pre-2019 rare-loss walk-forward refuses post-2018 rows; "
            f"first forbidden date is {first.date().isoformat()}"
        )
    result = feature_label_frame.copy()
    result.index = index
    result.index.name = "decision_date"
    for column in ("label_entry_date", "label_maturity_date"):
        dates = pd.to_datetime(result[column], errors="coerce")
        if bool((dates > DEVELOPMENT_LAST_DECISION).any()):
            first = dates.loc[dates > DEVELOPMENT_LAST_DECISION].iloc[0]
            raise ValueError(
                f"Pre-2019 rare-loss walk-forward refuses a post-2018 {column}: "
                f"{first.date().isoformat()}"
            )
    expected_years = entry_fill_years(result)
    try:
        actual_years = pd.to_numeric(
            result["entry_fill_year"], errors="raise"
        ).astype("Int16")
    except (TypeError, ValueError) as exc:
        raise ValueError("entry_fill_year must be a nullable integer") from exc
    actual_years.name = "entry_fill_year"
    if not actual_years.equals(expected_years):
        raise ValueError("entry_fill_year is inconsistent with label_entry_date")
    actual_ready = result["features_ready"]
    if actual_ready.isna().any() or not pd.api.types.is_bool_dtype(actual_ready.dtype):
        raise ValueError("features_ready must contain non-missing booleans")
    derived_ready = rare_loss_feature_readiness(result)
    if not actual_ready.astype(bool).equals(derived_ready):
        raise ValueError("features_ready is inconsistent with the frozen inputs")
    return result


def _sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in values).encode("ascii")).hexdigest()


def _target_token(value: float) -> str:
    number = float(value)
    return number.hex() if math.isfinite(number) else "missing"


def _fit_variant(
    frame: pd.DataFrame,
    *,
    training_mask: np.ndarray,
    fold: RareLossFold,
    variant: str,
) -> tuple[RareLossForest, dict[str, Any]]:
    columns = RARE_LOSS_FEATURE_COLUMNS if variant == "full" else CORE_FEATURE_COLUMNS
    training = frame.loc[training_mask]
    features = training.loc[:, list(columns)]
    severe = training["severe_loss_label_1session"].to_numpy(dtype=float)
    ordinary = training["cash_beats_long_10bps"].to_numpy(dtype=float)
    clipped_edge = training["cash_active_log_edge_10bps_clipped"].to_numpy(dtype=float)
    maturity = pd.to_datetime(training["label_maturity_date"], errors="raise")
    metadata = {
        "fold_id": fold.fold_id,
        "variant": variant,
        "training_start_date": training.index.min().date().isoformat(),
        "training_end_date": training.index.max().date().isoformat(),
        "maximum_label_maturity_date": maturity.max().date().isoformat(),
        "training_dates_sha256": _sha256_lines(
            [date.date().isoformat() for date in training.index]
        ),
        "severe_target_sha256": _sha256_lines([_target_token(value) for value in severe]),
        "ordinary_target_sha256": _sha256_lines([_target_token(value) for value in ordinary]),
        "clipped_edge_target_sha256": _sha256_lines(
            [_target_token(value) for value in clipped_edge]
        ),
    }
    model = RareLossForest().fit(
        features,
        severe,
        ordinary,
        clipped_edge,
        namespace=f"{fold.fold_id}|{variant}",
        fit_metadata=metadata,
    )
    oob = model.oob_components()
    counts = np.asarray(oob["oob_tree_count"], dtype=np.int64)
    if counts.shape != (len(training),) or int(counts.min()) < MINIMUM_OOB_TREE_COUNT:
        raise ValueError(
            f"{fold.fold_id} {variant} failed the minimum {MINIMUM_OOB_TREE_COUNT}-tree OOB contract"
        )
    thresholds = {
        candidate.candidate_id: model.oob_severe_threshold(candidate.oob_tail_quantile)
        for candidate in RARE_LOSS_CANDIDATES
    }
    state = {
        "fold_id": fold.fold_id,
        "variant": variant,
        "feature_columns": list(columns),
        "training_row_count": int(len(training)),
        "training_first_decision": training.index.min().date().isoformat(),
        "training_last_decision": training.index.max().date().isoformat(),
        "training_latest_label_maturity": maturity.max().date().isoformat(),
        "causal_severe_prevalence": float(severe.mean()),
        "causal_ordinary_prevalence": float(ordinary.mean()),
        "causal_training_mean_clipped_edge": float(clipped_edge.mean()),
        "oob_minimum_tree_count": int(counts.min()),
        "oob_maximum_tree_count": int(counts.max()),
        "oob_tree_counts": counts.tolist(),
        "oob_tail_thresholds": thresholds,
        "model_sha256": model.model_sha256,
        "model_state": model.to_state(),
    }
    # Prove now that the serialized state is canonical and behavior-preserving.
    restored = RareLossForest.from_state(state["model_state"])
    if restored.canonical_json() != model.canonical_json():
        raise RuntimeError("Rare-loss forest state did not round-trip byte-identically")
    return model, state


def _prediction_columns(variant: str) -> tuple[str, str, str]:
    return (
        f"{variant}_severe_probability",
        f"{variant}_ordinary_cash_win_probability_10bps",
        f"{variant}_expected_clipped_edge_10bps",
    )


def _build_fold_predictions(
    frame: pd.DataFrame,
    fold: RareLossFold,
) -> tuple[pd.DataFrame, tuple[Mapping[str, Any], ...]]:
    selected = fold.decision_mask(frame).to_numpy(dtype=bool)
    if not selected.any():
        raise ValueError(f"Fold {fold.fold_id} has no entry-fill decisions")
    first_decision = frame.index[selected].min()
    training_mask = strict_pre_fold_training_mask(
        frame,
        fold_first_decision_date=first_decision,
    ).to_numpy(dtype=bool)
    if not training_mask.any():
        raise ValueError(f"Fold {fold.fold_id} has no causal training rows")
    if bool((frame.index[training_mask] >= first_decision).any()):
        raise RuntimeError("Rare-loss training escaped the strict fold boundary")

    fold_frame = frame.loc[selected]
    result = fold_frame.loc[:, list(_LABEL_COLUMNS)].copy()
    severe_train = frame.loc[training_mask, "severe_loss_label_1session"].to_numpy(float)
    ordinary_train = frame.loc[training_mask, "cash_beats_long_10bps"].to_numpy(float)
    edge_train = frame.loc[training_mask, "cash_active_log_edge_10bps_clipped"].to_numpy(float)
    result["causal_severe_prevalence"] = float(severe_train.mean())
    result["causal_ordinary_prevalence_10bps"] = float(ordinary_train.mean())
    result["causal_training_mean_clipped_edge_10bps"] = float(edge_train.mean())
    ready = result["features_ready"].to_numpy(dtype=bool)
    states: list[Mapping[str, Any]] = []

    for variant in MODEL_VARIANTS:
        model, state = _fit_variant(
            frame,
            training_mask=training_mask,
            fold=fold,
            variant=variant,
        )
        states.append(state)
        columns = RARE_LOSS_FEATURE_COLUMNS if variant == "full" else CORE_FEATURE_COLUMNS
        severe_column, ordinary_column, edge_column = _prediction_columns(variant)
        result[severe_column] = np.nan
        result[ordinary_column] = np.nan
        result[edge_column] = np.nan
        if ready.any():
            components = model.predict_components(
                fold_frame.loc[ready, list(columns)]
            )
            available = np.asarray(components["observation_available"], dtype=bool)
            if not available.all():
                raise RuntimeError("A full-ready validation row was unavailable to a forest")
            result.loc[ready, severe_column] = np.asarray(
                components["severe_probability"], dtype=float
            )
            result.loc[ready, ordinary_column] = np.asarray(
                components["ordinary_cash_win_probability"], dtype=float
            )
            result.loc[ready, edge_column] = np.asarray(
                components["expected_clipped_edge"], dtype=float
            )
        for candidate in RARE_LOSS_CANDIDATES:
            threshold = float(state["oob_tail_thresholds"][candidate.candidate_id])
            threshold_column = f"{variant}_{candidate.candidate_id}_oob_severe_threshold"
            result[threshold_column] = threshold
            severe_probability = pd.to_numeric(result[severe_column], errors="coerce").to_numpy(float)
            ordinary_probability = pd.to_numeric(result[ordinary_column], errors="coerce").to_numpy(float)
            expected_edge = pd.to_numeric(result[edge_column], errors="coerce").to_numpy(float)
            cash = (
                ready
                & np.isfinite(severe_probability)
                & np.isfinite(ordinary_probability)
                & np.isfinite(expected_edge)
                & (severe_probability >= threshold)
                & (
                    severe_probability
                    >= SEVERE_PREVALENCE_MULTIPLIER * float(severe_train.mean())
                )
                & (ordinary_probability >= ORDINARY_PROBABILITY_GATE)
                & (expected_edge >= EXPECTED_EDGE_GATE_10BPS)
            )
            result[candidate_cash_target_column(variant, candidate.candidate_id)] = cash.astype(np.int8)
    result["fold_id"] = fold.fold_id
    return result, tuple(states)


def build_pre2019_rare_loss_walkforward(
    feature_label_frame: pd.DataFrame,
) -> RareLossWalkForwardResult:
    """Fit 14 frozen forests and return their causal 2005--2018 predictions."""

    frame = _canonical_frame(feature_label_frame)
    fold_predictions: list[pd.DataFrame] = []
    states: list[Mapping[str, Any]] = []
    for fold in WALK_FORWARD_FOLDS:
        predictions, fold_states = _build_fold_predictions(frame, fold)
        fold_predictions.append(predictions)
        states.extend(fold_states)
    combined = pd.concat(fold_predictions).sort_index(kind="stable")
    if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
        raise RuntimeError("Walk-forward predictions are not uniquely chronological")
    if len(states) != len(WALK_FORWARD_FOLDS) * len(MODEL_VARIANTS):
        raise RuntimeError("Walk-forward did not preserve exactly 14 forest states")

    # The simulation contract requires the opening and terminal fill to be
    # LONG.  This is frozen before scoring and applies equally to full/core.
    if combined.empty:
        raise RuntimeError("Walk-forward prediction frame is empty")
    for candidate in RARE_LOSS_CANDIDATES:
        for variant in MODEL_VARIANTS:
            column = candidate_cash_target_column(variant, candidate.candidate_id)
            combined.loc[combined.index[0], column] = np.int8(0)
            combined.loc[combined.index[-1], column] = np.int8(0)
            values = combined[column].to_numpy(dtype=float)
            if not np.isin(values, (0.0, 1.0)).all():
                raise RuntimeError(f"Frozen target is not binary: {column}")
    return RareLossWalkForwardResult(
        predictions=combined,
        model_states=tuple(states),
    )


def build_pre2019_rare_loss_predictions(
    feature_label_frame: pd.DataFrame,
) -> pd.DataFrame:
    return build_pre2019_rare_loss_walkforward(feature_label_frame).predictions


__all__ = [
    "DEVELOPMENT_FIRST_ENTRY_YEAR",
    "DEVELOPMENT_LAST_ENTRY_YEAR",
    "EXPECTED_EDGE_GATE_10BPS",
    "MODEL_VARIANTS",
    "ORDINARY_PROBABILITY_GATE",
    "RARE_LOSS_CANDIDATES",
    "SEVERE_PREVALENCE_MULTIPLIER",
    "WALK_FORWARD_FOLDS",
    "RareLossCandidate",
    "RareLossFold",
    "RareLossWalkForwardResult",
    "build_pre2019_rare_loss_predictions",
    "build_pre2019_rare_loss_walkforward",
    "candidate_cash_target_column",
]
