"""Sealed two-stage AAPL sector-breadth experiment.

``develop`` is the only research entry point.  It performs fixed 2005-2018
walk-forward selection, followed by one frozen 2019-2023 validation.  Its
public loaders are physically bounded at 2023-12-31 and the runner rejects an
injected post-cutoff row before feature construction.

``final`` is deliberately a separate command.  Before any holdout row may be
loaded it requires a self-hashed development selection manifest which is an
exact tracked blob at the clean current Git commit.  Missing, dirty,
unselected, or failed manifests stop the command before market-data loading.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
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

from .deterministic_aapl import (
    CostAssumptions,
    EvaluationPeriod,
    compare_ledgers,
    file_sha256,
)
from .direct_edge_gam import DirectEdgeGAMConfig
from .direct_edge_scoring import evaluate_direct_edge_development
from .direct_edge_walkforward import DIRECT_EDGE_CANDIDATES, five_session_cash_policy
from .sector_breadth_features import (
    PRICE_FEATURE_COLUMNS,
    SECTOR_BREADTH_FEATURE_COLUMNS,
    build_sector_breadth_feature_label_frame,
    load_sector_context_parquet,
)
from .sector_breadth_walkforward import (
    FrozenSectorBreadthPolicy,
    SECTOR_BREADTH_MODEL_FAMILIES,
    SECTOR_BREADTH_WALK_FORWARD_FOLDS,
    build_sector_breadth_walkforward_predictions,
    fit_intermediate_sector_breadth_policy,
    fit_final_sector_breadth_policy,
    predict_intermediate_sector_breadth_validation,
    sector_breadth_candidate_cash_block_start_column,
    sector_breadth_candidate_cash_target_column,
)
from .unleveraged_aapl import (
    BEAR_STRESS_PERIODS,
    FINAL_PERIODS,
    _continuous_period_returns,
    _git_state,
    _period_report,
    assert_unleveraged_ledger,
    canonical_context_frame,
    reserve_holdout_touch,
    simulate_unleveraged_period,
)


CONTRACT_VERSION = "aapl-sector-breadth-residual-edge-v1"
DEVELOPMENT_INPUT_START = pd.Timestamp("1999-01-01")
CONTEXT_INPUT_START = pd.Timestamp("2000-05-26")
DEVELOPMENT_DATA_END = pd.Timestamp("2023-12-31")
SELECTION_START = pd.Timestamp("2005-01-01")
SELECTION_END = pd.Timestamp("2018-12-31")
VALIDATION_START = pd.Timestamp("2019-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")
DEVELOPMENT_PERIOD = EvaluationPeriod(
    "sector_breadth_selection", SELECTION_START.date().isoformat(), SELECTION_END.date().isoformat()
)
VALIDATION_PERIOD = EvaluationPeriod(
    "sector_breadth_validation", VALIDATION_START.date().isoformat(), VALIDATION_END.date().isoformat()
)
VALIDATION_FOLDS: tuple[tuple[str, int, int], ...] = (
    ("2019_2020", 2019, 2020),
    ("2021_2022", 2021, 2022),
    ("2023", 2023, 2023),
)
BASE_COST_BPS = 5.0
STRESS_COST_BPS = 10.0
COST_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("base_5bps", BASE_COST_BPS),
    ("stress_10bps", STRESS_COST_BPS),
)
VALIDATION_GATE_CONTRACT: Mapping[str, float | int] = {
    "minimum_positive_folds": 2,
    "minimum_annual_win_rate": 0.60,
    "maximum_cash_day_rate": 0.20,
    "minimum_cash_episodes": 4,
}
INITIAL_CASH = 1000.0
RUN_TIME_LIMIT_SECONDS = 3600.0
FINAL_DATA_END = pd.Timestamp(FINAL_PERIODS[-1].end)

DEVELOPMENT_PRICE_QUERY = """
SELECT
  CAST(date AS DATE) AS date,
  CAST(aapl_open AS DOUBLE) AS aapl_open,
  CAST(aapl_close AS DOUBLE) AS aapl_close,
  CAST(aapl_adj_close AS DOUBLE) AS aapl_adj_close,
  CAST(spy_adj_close AS DOUBLE) AS spy_adj_close,
  CAST(qqq_adj_close AS DOUBLE) AS qqq_adj_close
FROM read_csv_auto(?, header = true)
WHERE CAST(date AS DATE) >= CAST(? AS DATE)
  AND CAST(date AS DATE) <= CAST(? AS DATE)
ORDER BY CAST(date AS DATE)
""".strip()
FINAL_PRICE_QUERY = DEVELOPMENT_PRICE_QUERY
CONTEXT_DOWNLOAD_LOG_SYMBOLS = (
    "IWM",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "^VIX",
)
CONTEXT_DOWNLOAD_LOG_QUERY = """
SELECT upper(trim(symbol)) AS symbol, created_at
FROM read_parquet(?)
WHERE task = 'prices'
  AND status = 'ok'
  AND upper(trim(symbol)) IN (
    'IWM', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK',
    'XLP', 'XLU', 'XLV', 'XLY', '^VIX'
  )
ORDER BY upper(trim(symbol)), created_at
""".strip()

PRICE_LABEL_COLUMNS = (
    "label_entry_date",
    "label_maturity_date",
    "cash_active_log_edge_5bps",
    "cash_active_log_edge_10bps",
    "cash_beats_long_10bps",
)


class SectorBreadthExperimentError(RuntimeError):
    """Raised when a sealed experiment contract is violated."""


class SectorBreadthExperimentTimeout(SectorBreadthExperimentError):
    """Raised when the fixed wall-clock budget expires."""


@dataclass(frozen=True)
class LoadedDevelopmentInputs:
    price_frame: pd.DataFrame
    context_frame: pd.DataFrame
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ValidatedFinalSelection:
    manifest_path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    git_commit: str
    git_branch: str


@dataclass(frozen=True)
class CandidateSpec:
    model_family: str
    candidate: Any

    @property
    def candidate_id(self) -> str:
        return f"{self.model_family}_{self.candidate.candidate_id}"

    @property
    def cash_target_column(self) -> str:
        return sector_breadth_candidate_cash_target_column(
            self.model_family, self.candidate.candidate_id
        )

    @property
    def cash_block_start_column(self) -> str:
        return sector_breadth_candidate_cash_block_start_column(
            self.model_family, self.candidate.candidate_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "gate_id": self.candidate.candidate_id,
            "probability_gate": float(self.candidate.probability_gate),
            "expected_edge_gate_10bps": float(self.candidate.expected_edge_gate),
            "cash_decision_rows": 5,
        }


CANDIDATE_SPECS: tuple[CandidateSpec, ...] = tuple(
    CandidateSpec(family, candidate)
    for family in SECTOR_BREADTH_MODEL_FAMILIES
    for candidate in DIRECT_EDGE_CANDIDATES
)


class _Deadline:
    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = float(clock())

    def elapsed(self) -> float:
        return float(self._clock()) - self._started

    def check(self, location: str) -> None:
        if self.elapsed() > RUN_TIME_LIMIT_SECONDS:
            raise SectorBreadthExperimentTimeout(
                f"Sector-breadth experiment exceeded {RUN_TIME_LIMIT_SECONDS:.0f}s at {location}"
            )


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot encode {type(value).__name__} as canonical JSON")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_tagged(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return buffer.getvalue().encode("utf-8")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _file_timestamp_provenance(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "observed_filesystem_creation_time_utc": datetime.fromtimestamp(
            stat.st_ctime, tz=timezone.utc
        ).isoformat(),
        "observed_filesystem_last_write_time_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "timestamp_semantics": (
            "filesystem observations bind this local artifact; the download log, when "
            "present, supplies the acquisition event timestamps"
        ),
    }


def _context_download_log_provenance(context_parquet: Path) -> dict[str, Any]:
    log_path = context_parquet.resolve().parent / "download_log.parquet"
    if not log_path.is_file():
        return {"present": False, "expected_path": str(log_path)}
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            rows = connection.execute(
                CONTEXT_DOWNLOAD_LOG_QUERY, [str(log_path)]
            ).fetchdf()
        finally:
            connection.close()
    except Exception as exc:
        raise SectorBreadthExperimentError(
            "Could not bind the local context download log"
        ) from exc
    observed_symbols = tuple(sorted(set(rows.get("symbol", pd.Series(dtype=str)).astype(str))))
    expected_symbols = tuple(sorted(CONTEXT_DOWNLOAD_LOG_SYMBOLS))
    created = pd.to_datetime(rows.get("created_at", pd.Series(dtype=str)), utc=True, errors="raise")
    return {
        "present": True,
        "path": str(log_path),
        "file_sha256": f"sha256:{file_sha256(log_path)}",
        **_file_timestamp_provenance(log_path),
        "query": CONTEXT_DOWNLOAD_LOG_QUERY,
        "bounded_rows": int(len(rows)),
        "bounded_result_sha256": _sha256_tagged(_frame_csv_bytes(rows)),
        "expected_symbols": list(expected_symbols),
        "observed_symbols": list(observed_symbols),
        "complete_fixed_symbol_coverage": observed_symbols == expected_symbols,
        "first_acquisition_event_utc": created.min().isoformat() if len(created) else None,
        "last_acquisition_event_utc": created.max().isoformat() if len(created) else None,
    }


def _seal_bundle(run_dir: Path, payloads: Mapping[str, bytes]) -> dict[str, str]:
    if run_dir.exists():
        raise SectorBreadthExperimentError(f"Artifact directory already exists: {run_dir}")
    temporary = run_dir.with_name(f".{run_dir.name}.{uuid.uuid4().hex}.sealing")
    if temporary.exists():
        raise SectorBreadthExperimentError(f"Temporary artifact directory exists: {temporary}")
    temporary.mkdir(parents=True)
    promoted = False
    try:
        checksums = {
            name: _sha256_tagged(payload) for name, payload in sorted(payloads.items())
        }
        for name, payload in payloads.items():
            destination = temporary / name
            if destination.parent != temporary:
                raise SectorBreadthExperimentError("Artifact names must be flat")
            _atomic_write_bytes(destination, payload)
        _atomic_write_bytes(temporary / "checksums.json", _pretty_json_bytes(checksums))
        for name, expected in checksums.items():
            observed = _sha256_tagged((temporary / name).read_bytes())
            if observed != expected:
                raise SectorBreadthExperimentError(f"Artifact checksum mismatch: {name}")
        temporary.replace(run_dir)
        promoted = True
        return checksums
    finally:
        if not promoted and temporary.exists():
            shutil.rmtree(temporary)


def _normalize_index(frame: pd.DataFrame, *, name: str) -> pd.DatetimeIndex:
    if "date" in frame.columns:
        raw = frame["date"]
    else:
        raw = frame.index
    try:
        index = pd.DatetimeIndex(pd.to_datetime(raw, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise SectorBreadthExperimentError(f"{name} dates are invalid") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise SectorBreadthExperimentError(f"{name} dates must be unique and chronological")
    return index


def load_development_price_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load only the physical price rows allowed by the development cutoff."""

    source = path.resolve()
    if not source.is_file():
        raise SectorBreadthExperimentError(f"Price CSV does not exist: {source}")
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            raw = connection.execute(
                DEVELOPMENT_PRICE_QUERY,
                [
                    str(source),
                    DEVELOPMENT_INPUT_START.date().isoformat(),
                    DEVELOPMENT_DATA_END.date().isoformat(),
                ],
            ).fetchdf()
        finally:
            connection.close()
    except Exception as exc:
        raise SectorBreadthExperimentError("Bounded development price query failed") from exc
    dates = _normalize_index(raw, name="development price query")
    if not len(dates) or dates.max() > DEVELOPMENT_DATA_END:
        raise SectorBreadthExperimentError("Bounded price query violated the pre-2024 boundary")
    try:
        frame = canonical_context_frame(raw)
    except (TypeError, ValueError) as exc:
        raise SectorBreadthExperimentError("Bounded price rows are not canonical") from exc
    bounded_payload = _frame_csv_bytes(frame.reset_index(names="date"))
    return frame, {
        "source_type": "local_csv_bounded_duckdb_query",
        "source_path": str(source),
        "source_file_sha256": f"sha256:{file_sha256(source)}",
        "query": DEVELOPMENT_PRICE_QUERY,
        "query_parameters": [
            str(source),
            DEVELOPMENT_INPUT_START.date().isoformat(),
            DEVELOPMENT_DATA_END.date().isoformat(),
        ],
        "bounded_first_date": frame.index.min().date().isoformat(),
        "bounded_last_date": frame.index.max().date().isoformat(),
        "bounded_rows": int(len(frame)),
        "bounded_result_sha256": _sha256_tagged(bounded_payload),
        "post_2023_rows_returned": False,
    }


def load_public_development_inputs(
    *, price_artifact: Path, context_parquet: Path
) -> LoadedDevelopmentInputs:
    prices, price_provenance = load_development_price_csv(price_artifact)
    try:
        context = load_sector_context_parquet(
            context_parquet,
            start=CONTEXT_INPUT_START,
            end=DEVELOPMENT_DATA_END,
        )
    except Exception as exc:
        raise SectorBreadthExperimentError("Bounded sector-context query failed") from exc
    context_dates = _normalize_index(context, name="development sector context")
    if not len(context_dates) or context_dates.max() > DEVELOPMENT_DATA_END:
        raise SectorBreadthExperimentError("Sector context violated the pre-2024 boundary")
    context_payload = _frame_csv_bytes(context.reset_index(names="date"))
    return LoadedDevelopmentInputs(
        price_frame=prices,
        context_frame=context,
        provenance={
            "price": price_provenance,
            "context": {
                "source_type": "local_sector_context_parquet_bounded_query",
                "source_path": str(context_parquet.resolve()),
                "source_file_sha256": f"sha256:{file_sha256(context_parquet.resolve())}",
                **_file_timestamp_provenance(context_parquet.resolve()),
                "download_log": _context_download_log_provenance(context_parquet),
                "bounded_start": CONTEXT_INPUT_START.date().isoformat(),
                "bounded_end": DEVELOPMENT_DATA_END.date().isoformat(),
                "bounded_rows": int(len(context)),
                "bounded_result_sha256": _sha256_tagged(context_payload),
                "post_2023_rows_returned": False,
                "network_access": False,
            },
        },
    )


def _bounded_development_inputs(
    price_frame: pd.DataFrame, context_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_price_dates = _normalize_index(price_frame, name="development prices")
    raw_context_dates = _normalize_index(context_frame, name="development context")
    if len(raw_price_dates) and raw_price_dates.max() > DEVELOPMENT_DATA_END:
        first = raw_price_dates[raw_price_dates > DEVELOPMENT_DATA_END][0]
        raise SectorBreadthExperimentError(
            f"Development input contains a post-2023 price row: {first.date().isoformat()}"
        )
    if len(raw_context_dates) and raw_context_dates.max() > DEVELOPMENT_DATA_END:
        first = raw_context_dates[raw_context_dates > DEVELOPMENT_DATA_END][0]
        raise SectorBreadthExperimentError(
            f"Development input contains a post-2023 context row: {first.date().isoformat()}"
        )
    try:
        prices = canonical_context_frame(price_frame)
    except (TypeError, ValueError) as exc:
        raise SectorBreadthExperimentError("Development prices are not canonical") from exc
    context = context_frame.copy()
    context.index = raw_context_dates
    if prices.index.min() > DEVELOPMENT_INPUT_START + pd.Timedelta(days=120):
        raise SectorBreadthExperimentError("Development prices lack required warm-up history")
    if prices.index.max() < pd.Timestamp("2023-12-20"):
        raise SectorBreadthExperimentError("Development prices do not reach the frozen cutoff")
    expected_context_index = prices.index[
        (prices.index >= CONTEXT_INPUT_START) & (prices.index <= DEVELOPMENT_DATA_END)
    ]
    if not context.index.equals(expected_context_index):
        missing = expected_context_index.difference(context.index)
        extra = context.index.difference(expected_context_index)
        raise SectorBreadthExperimentError(
            "Sector context must exactly equal every expected price session from "
            f"{CONTEXT_INPUT_START.date().isoformat()} through 2023-12-31; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return prices, context, {
        "price_first_date": prices.index.min().date().isoformat(),
        "price_last_date": prices.index.max().date().isoformat(),
        "price_rows": int(len(prices)),
        "context_first_date": context.index.min().date().isoformat(),
        "context_last_date": context.index.max().date().isoformat(),
        "context_rows": int(len(context)),
        "post_2023_rows_accessed": False,
    }


def _validated_feature_label_frame(
    price_frame: pd.DataFrame, context_frame: pd.DataFrame
) -> pd.DataFrame:
    try:
        frame = build_sector_breadth_feature_label_frame(price_frame, context_frame)
    except Exception as exc:
        raise SectorBreadthExperimentError("Sector-breadth feature construction failed") from exc
    index = _normalize_index(frame, name="sector-breadth feature-label frame")
    result = frame.copy()
    result.index = index
    result.index.name = "decision_date"
    if result.index.max() > DEVELOPMENT_DATA_END:
        raise SectorBreadthExperimentError("Feature construction produced a post-2023 row")
    required = {
        *PRICE_FEATURE_COLUMNS,
        *SECTOR_BREADTH_FEATURE_COLUMNS,
        *PRICE_LABEL_COLUMNS,
        "price_features_ready",
        "sector_breadth_features_ready",
        "sector_breadth_price_only_fallback",
        "label_available",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise SectorBreadthExperimentError(
            f"Feature-label frame is missing required columns: {missing}"
        )
    available = result["label_available"]
    if available.isna().any() or any(
        not isinstance(value, (bool, np.bool_)) for value in available.to_numpy()
    ):
        raise SectorBreadthExperimentError("label_available must contain non-missing booleans")
    maturity = pd.to_datetime(result["label_maturity_date"], errors="coerce")
    if bool((maturity.loc[available] > DEVELOPMENT_DATA_END).any()):
        raise SectorBreadthExperimentError(
            "A development label matures after the 2023-12-31 cutoff"
        )
    if bool(result.loc[available, "cash_beats_long_10bps"].isna().any()):
        raise SectorBreadthExperimentError("An available development label is missing")
    return result


def _full_exposure_target(
    price_index: pd.DatetimeIndex,
    decision_index: pd.DatetimeIndex,
    cash_target: Sequence[float],
) -> pd.Series:
    decision_index = pd.DatetimeIndex(decision_index)
    values = np.asarray(cash_target, dtype=float)
    if len(values) != len(decision_index) or not np.isin(values, (0.0, 1.0)).all():
        raise SectorBreadthExperimentError("CASH target must be aligned binary data")
    if not decision_index.isin(price_index).all():
        raise SectorBreadthExperimentError("A CASH decision is absent from the price index")
    result = pd.Series(1.0, index=price_index, name="target_exposure")
    result.loc[decision_index] = 1.0 - values
    return result


def _simulate(
    frame: pd.DataFrame,
    target: pd.Series,
    period: EvaluationPeriod,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    ledger = simulate_unleveraged_period(
        frame,
        target,
        period,
        CostAssumptions(slippage_bps=cost_bps, annual_margin_rate=0.0),
        initial_cash=INITIAL_CASH,
    )
    assert_unleveraged_ledger(ledger)
    return ledger


def _prediction_inputs(
    predictions: pd.DataFrame,
    family: str,
    cash_target: np.ndarray,
    cash_block_start: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
]:
    columns = {
        "probability": f"{family}_cash_win_probability",
        "edge": f"{family}_expected_net_edge",
        "baseline_probability": f"{family}_baseline_cash_win_probability",
        "baseline_edge": f"{family}_baseline_mean_net_edge",
        "label": "cash_beats_long_10bps",
        "actual_edge": "cash_active_log_edge_10bps",
    }
    missing = sorted(set(columns.values()).difference(predictions.columns))
    if missing:
        raise SectorBreadthExperimentError(
            f"Walk-forward predictions are missing scoring columns: {missing}"
        )
    arrays = [
        predictions[columns["probability"]].to_numpy(dtype=float),
        predictions[columns["label"]].to_numpy(dtype=float),
        predictions[columns["baseline_probability"]].to_numpy(dtype=float),
        predictions[columns["edge"]].to_numpy(dtype=float),
        predictions[columns["actual_edge"]].to_numpy(dtype=float),
        predictions[columns["baseline_edge"]].to_numpy(dtype=float),
        np.asarray(cash_target, dtype=float),
        np.asarray(cash_block_start, dtype=float),
    ]
    if len({len(value) for value in arrays}) != 1:
        raise SectorBreadthExperimentError("Walk-forward scoring vectors are misaligned")
    return (*arrays, pd.DatetimeIndex(predictions.index))


def _score_development_strategy(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
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
    ],
    *,
    cost_bps: float,
) -> dict[str, Any]:
    probability, label, baseline_probability, edge, actual_edge, baseline_edge, cash, block_start, decision_dates = prediction_inputs
    scored = evaluate_direct_edge_development(
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
    scored["no_leverage_proof"] = assert_unleveraged_ledger(strategy)
    return scored


def _common_support_mask(predictions: pd.DataFrame) -> np.ndarray:
    required = {
        "price_only_cash_win_probability",
        "price_only_expected_net_edge",
        "sector_breadth_cash_win_probability",
        "sector_breadth_expected_net_edge",
        "sector_breadth_features_ready",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise SectorBreadthExperimentError(
            f"Walk-forward predictions lack common-support columns: {missing}"
        )
    ready = predictions["sector_breadth_features_ready"].to_numpy(dtype=bool)
    fallback = (
        predictions["sector_breadth_used_price_fallback"].to_numpy(dtype=bool)
        if "sector_breadth_used_price_fallback" in predictions.columns
        else ~ready
    )
    finite = np.ones(len(predictions), dtype=bool)
    for family in SECTOR_BREADTH_MODEL_FAMILIES:
        finite &= np.isfinite(
            predictions[f"{family}_cash_win_probability"].to_numpy(dtype=float)
        )
        finite &= np.isfinite(
            predictions[f"{family}_expected_net_edge"].to_numpy(dtype=float)
        )
    return ready & ~fallback & finite


def _common_support_target(
    predictions: pd.DataFrame, spec: CandidateSpec, support: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probability = predictions[
        f"{spec.model_family}_cash_win_probability"
    ].to_numpy(dtype=float)
    edge = predictions[f"{spec.model_family}_expected_net_edge"].to_numpy(dtype=float)
    trigger = (
        support
        & np.isfinite(probability)
        & np.isfinite(edge)
        & (probability >= spec.candidate.probability_gate)
        & (edge >= spec.candidate.expected_edge_gate)
    )
    return five_session_cash_policy(trigger)


def _common_support_predictive_metrics(
    predictions: pd.DataFrame, family: str, support: np.ndarray
) -> dict[str, Any]:
    label = predictions["cash_beats_long_10bps"].to_numpy(dtype=float)
    actual_edge = predictions["cash_active_log_edge_10bps"].to_numpy(dtype=float)
    probability = predictions[f"{family}_cash_win_probability"].to_numpy(dtype=float)
    baseline_probability = predictions[
        f"{family}_baseline_cash_win_probability"
    ].to_numpy(dtype=float)
    expected_edge = predictions[f"{family}_expected_net_edge"].to_numpy(dtype=float)
    baseline_edge = predictions[f"{family}_baseline_mean_net_edge"].to_numpy(dtype=float)
    mask = support & np.isfinite(label) & np.isfinite(actual_edge)
    if not mask.any():
        raise SectorBreadthExperimentError("Common-support predictive population is empty")
    return {
        "rows": int(mask.sum()),
        "brier_score": float(np.mean(np.square(probability[mask] - label[mask]))),
        "causal_training_mean_brier": float(
            np.mean(np.square(baseline_probability[mask] - label[mask]))
        ),
        "expected_edge_mae": float(
            np.mean(np.abs(expected_edge[mask] - actual_edge[mask]))
        ),
        "causal_training_mean_edge_mae": float(
            np.mean(np.abs(baseline_edge[mask] - actual_edge[mask]))
        ),
    }


def _paired_ablation(
    common_results: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    gate_id: str,
) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    passed: list[bool] = []
    for scenario_name, _ in COST_SCENARIOS:
        price = common_results[f"price_only_{gate_id}"][scenario_name]
        breadth = common_results[f"sector_breadth_{gate_id}"][scenario_name]
        price_metrics = price["metrics"]
        breadth_metrics = breadth["metrics"]
        price_predictive = price["common_support_predictive_metrics"]
        breadth_predictive = breadth["common_support_predictive_metrics"]
        checks = {
            "strict_brier_improvement": float(breadth_predictive["brier_score"])
            < float(price_predictive["brier_score"]),
            "strict_expected_edge_mae_improvement": float(
                breadth_predictive["expected_edge_mae"]
            )
            < float(price_predictive["expected_edge_mae"]),
            "strict_total_active_log_improvement": float(
                breadth_metrics["total_active_log_edge"]
            )
            > float(price_metrics["total_active_log_edge"]),
            "weakest_fold_not_worse": float(
                breadth_metrics["minimum_fold_active_log_edge"]
            )
            >= float(price_metrics["minimum_fold_active_log_edge"]),
        }
        scenarios[scenario_name] = {
            "price_only": {
                "brier_score": float(price_predictive["brier_score"]),
                "expected_edge_mae": float(price_predictive["expected_edge_mae"]),
                "total_active_log_edge": float(price_metrics["total_active_log_edge"]),
                "minimum_fold_active_log_edge": float(
                    price_metrics["minimum_fold_active_log_edge"]
                ),
            },
            "sector_breadth": {
                "brier_score": float(breadth_predictive["brier_score"]),
                "expected_edge_mae": float(breadth_predictive["expected_edge_mae"]),
                "total_active_log_edge": float(
                    breadth_metrics["total_active_log_edge"]
                ),
                "minimum_fold_active_log_edge": float(
                    breadth_metrics["minimum_fold_active_log_edge"]
                ),
            },
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
        passed.extend(checks.values())
    return {
        "applicable": True,
        "comparison": "identical sector-breadth-ready OOF decision rows",
        "scenarios": scenarios,
        "passed": bool(all(passed)),
    }


def _select_breadth_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
) -> str | None:
    passing = [
        item
        for item in candidate_results
        if bool(item.get("passed"))
        and item.get("candidate", {}).get("model_family") == "sector_breadth"
    ]
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


def _score_selection_candidates(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    deadline: _Deadline,
) -> tuple[
    list[dict[str, Any]],
    str | None,
    dict[str, pd.DataFrame],
    dict[str, np.ndarray],
]:
    prediction_index = _normalize_index(predictions, name="selection predictions")
    if prediction_index.min() < SELECTION_START or prediction_index.max() > SELECTION_END:
        raise SectorBreadthExperimentError(
            "Selection predictions must contain only 2005-2018 decisions"
        )
    predictions = predictions.copy()
    predictions.index = prediction_index
    support = _common_support_mask(predictions)
    if not support.any():
        raise SectorBreadthExperimentError("No paired sector-breadth OOF support exists")

    benchmark_target = pd.Series(1.0, index=data.index, name="target_exposure")
    benchmarks: dict[str, pd.DataFrame] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for scenario_name, bps in COST_SCENARIOS:
        benchmark = _simulate(data, benchmark_target, DEVELOPMENT_PERIOD, cost_bps=bps)
        benchmarks[scenario_name] = benchmark
        ledgers[f"selection_buy_hold_{scenario_name}.csv"] = benchmark
        deadline.check(f"selection {scenario_name} benchmark")

    cash_by_candidate: dict[str, np.ndarray] = {}
    block_by_candidate: dict[str, np.ndarray] = {}
    common_cash: dict[str, np.ndarray] = {}
    common_block: dict[str, np.ndarray] = {}
    for spec in CANDIDATE_SPECS:
        try:
            cash = predictions[spec.cash_target_column].to_numpy(dtype=np.int8)
            block = predictions[spec.cash_block_start_column].to_numpy(dtype=np.int8)
        except KeyError as exc:
            raise SectorBreadthExperimentError(
                f"Walk-forward omitted candidate target for {spec.candidate_id}"
            ) from exc
        cash_by_candidate[spec.candidate_id] = cash
        block_by_candidate[spec.candidate_id] = block
        common_cash[spec.candidate_id], common_block[spec.candidate_id] = (
            _common_support_target(predictions, spec, support)
        )

    common_results: dict[str, dict[str, Any]] = {}
    predictive = {
        family: _common_support_predictive_metrics(predictions, family, support)
        for family in SECTOR_BREADTH_MODEL_FAMILIES
    }
    for spec in CANDIDATE_SPECS:
        target = _full_exposure_target(
            data.index, predictions.index, common_cash[spec.candidate_id]
        )
        inputs = _prediction_inputs(
            predictions,
            spec.model_family,
            common_cash[spec.candidate_id],
            common_block[spec.candidate_id],
        )
        scenarios: dict[str, Any] = {}
        for scenario_name, bps in COST_SCENARIOS:
            strategy = _simulate(data, target, DEVELOPMENT_PERIOD, cost_bps=bps)
            scored = _score_development_strategy(
                strategy, benchmarks[scenario_name], inputs, cost_bps=bps
            )
            scored["common_support_predictive_metrics"] = predictive[spec.model_family]
            scenarios[scenario_name] = scored
            ledgers[f"selection_common_{spec.candidate_id}_{scenario_name}.csv"] = strategy
            deadline.check(f"selection common {spec.candidate_id} {scenario_name}")
        common_results[spec.candidate_id] = scenarios

    results: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        target = _full_exposure_target(
            data.index, predictions.index, cash_by_candidate[spec.candidate_id]
        )
        inputs = _prediction_inputs(
            predictions,
            spec.model_family,
            cash_by_candidate[spec.candidate_id],
            block_by_candidate[spec.candidate_id],
        )
        scenarios: dict[str, Any] = {}
        for scenario_name, bps in COST_SCENARIOS:
            strategy = _simulate(data, target, DEVELOPMENT_PERIOD, cost_bps=bps)
            scenarios[scenario_name] = _score_development_strategy(
                strategy, benchmarks[scenario_name], inputs, cost_bps=bps
            )
            ledgers[f"selection_{spec.candidate_id}_{scenario_name}.csv"] = strategy
            deadline.check(f"selection {spec.candidate_id} {scenario_name}")
        both_costs = bool(all(item["gates"]["passed"] for item in scenarios.values()))
        if spec.model_family == "sector_breadth":
            ablation = _paired_ablation(
                common_results, gate_id=spec.candidate.candidate_id
            )
            eligible = True
        else:
            ablation = {
                "applicable": False,
                "reason": "price-only is the fixed ablation baseline and cannot advance",
                "passed": True,
            }
            eligible = False
        results.append(
            {
                "candidate": spec.to_dict(),
                "stage": "2005_2018_walk_forward_selection",
                "scenarios": scenarios,
                "both_cost_gates_passed": both_costs,
                "paired_full_vs_price_ablation": ablation,
                "eligible_for_selection": eligible,
                "passed": bool(eligible and both_costs and ablation["passed"]),
            }
        )
    return results, _select_breadth_candidate(results), ledgers, cash_by_candidate


def _candidate_by_id(candidate_id: str) -> CandidateSpec:
    selected = next((item for item in CANDIDATE_SPECS if item.candidate_id == candidate_id), None)
    if selected is None:
        raise SectorBreadthExperimentError(f"Unknown frozen candidate: {candidate_id}")
    return selected


def _canonical_validation_predictions(
    predictions: pd.DataFrame,
    feature_label: pd.DataFrame,
    selected: CandidateSpec,
) -> pd.DataFrame:
    result = predictions.copy()
    index = _normalize_index(result, name="intermediate validation predictions")
    result.index = index
    result.index.name = "decision_date"
    if result.index.min() < VALIDATION_START or result.index.max() > VALIDATION_END:
        raise SectorBreadthExperimentError(
            "Intermediate validation predictions must contain only 2019-2023 decisions"
        )
    required = {
        "cash_win_probability",
        "expected_net_edge",
        "baseline_cash_win_probability",
        "baseline_mean_net_edge",
        "cash_target",
        "cash_block_start",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise SectorBreadthExperimentError(
            f"Intermediate validation predictions lack columns: {missing}"
        )
    for label in PRICE_LABEL_COLUMNS:
        if label not in result.columns:
            result[label] = feature_label.reindex(result.index)[label]
    if "model_family" in result.columns:
        observed = set(result["model_family"].astype(str).unique())
        if observed != {selected.model_family}:
            raise SectorBreadthExperimentError("Validation model family drifted after selection")
    if "candidate_id" in result.columns:
        observed = set(result["candidate_id"].astype(str).unique())
        accepted = {selected.candidate.candidate_id, selected.candidate_id}
        if not observed.issubset(accepted):
            raise SectorBreadthExperimentError("Validation candidate gate drifted after selection")
    cash = result["cash_target"].to_numpy(dtype=float)
    starts = result["cash_block_start"].to_numpy(dtype=float)
    if not np.isin(cash, (0.0, 1.0)).all() or not np.isin(starts, (0.0, 1.0)).all():
        raise SectorBreadthExperimentError("Validation policy targets are not binary")
    rebuilt = np.zeros(len(result), dtype=np.int8)
    for position in np.flatnonzero(starts == 1.0):
        rebuilt[int(position) : min(int(position) + 5, len(rebuilt))] = 1
    if not np.array_equal(rebuilt, cash.astype(np.int8)):
        raise SectorBreadthExperimentError("Validation CASH blocks do not replay exactly")
    return result


def _active_log_series(
    strategy: pd.DataFrame, benchmark: pd.DataFrame
) -> tuple[pd.DatetimeIndex, pd.Series]:
    strategy_dates = pd.DatetimeIndex(pd.to_datetime(strategy["fill_date"], errors="raise"))
    benchmark_dates = pd.DatetimeIndex(pd.to_datetime(benchmark["fill_date"], errors="raise"))
    if not strategy_dates.equals(benchmark_dates):
        raise SectorBreadthExperimentError("Strategy and buy-and-hold ledgers are misaligned")
    strategy_return = strategy["daily_return"].to_numpy(dtype=float)
    benchmark_return = benchmark["daily_return"].to_numpy(dtype=float)
    active = np.log1p(strategy_return) - np.log1p(benchmark_return)
    return strategy_dates, pd.Series(active, index=strategy_dates, dtype=float)


def _negative_buy_hold_year_gate(
    annual_active_log_edges: Mapping[str, float],
    annual_buy_hold_returns: Mapping[str, float],
) -> dict[str, Any]:
    if set(annual_active_log_edges) != set(annual_buy_hold_returns):
        raise SectorBreadthExperimentError(
            "Annual active-edge and buy-and-hold years do not align"
        )
    negative = {
        year: float(annual_active_log_edges[year])
        for year, benchmark_return in annual_buy_hold_returns.items()
        if float(benchmark_return) < 0.0
    }
    return {
        "negative_buy_hold_years": sorted(negative),
        "negative_buy_hold_year_active_log_edges": negative,
        "all_negative_buy_hold_years_have_positive_active_log_edge": bool(
            all(value > 1e-12 for value in negative.values())
        ),
    }


def _score_validation_scenario(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    cost_bps: float,
) -> dict[str, Any]:
    assert_unleveraged_ledger(strategy)
    assert_unleveraged_ledger(benchmark)
    dates, active = _active_log_series(strategy, benchmark)
    years = tuple(sorted(set(int(value) for value in dates.year)))
    if years != tuple(range(2019, 2024)):
        raise SectorBreadthExperimentError(
            "Validation ledgers must contain exactly the 2019-2023 fill years"
        )
    annual = {
        str(year): float(active.loc[dates.year == year].sum()) for year in years
    }
    benchmark_daily = benchmark["daily_return"].to_numpy(dtype=float)
    annual_buy_hold = {
        str(year): float(np.prod(1.0 + benchmark_daily[dates.year == year]) - 1.0)
        for year in years
    }
    negative_year_gate = _negative_buy_hold_year_gate(annual, annual_buy_hold)
    folds: dict[str, float] = {}
    for name, first_year, last_year in VALIDATION_FOLDS:
        mask = (dates.year >= first_year) & (dates.year <= last_year)
        if not bool(mask.any()):
            raise SectorBreadthExperimentError(f"Validation fold has no fills: {name}")
        folds[name] = float(active.loc[mask].sum())

    probability = predictions["cash_win_probability"].to_numpy(dtype=float)
    baseline_probability = predictions[
        "baseline_cash_win_probability"
    ].to_numpy(dtype=float)
    expected_edge = predictions["expected_net_edge"].to_numpy(dtype=float)
    baseline_edge = predictions["baseline_mean_net_edge"].to_numpy(dtype=float)
    label = predictions["cash_beats_long_10bps"].to_numpy(dtype=float)
    actual_edge = predictions["cash_active_log_edge_10bps"].to_numpy(dtype=float)
    ready = predictions["sector_breadth_features_ready"].to_numpy(dtype=bool)
    forecast_finite = np.isfinite(probability) & np.isfinite(expected_edge)
    if bool((forecast_finite & ~ready).any()):
        raise SectorBreadthExperimentError(
            "Validation produced a breadth forecast on a breadth-unready row"
        )
    if bool((ready & ~forecast_finite).any()):
        raise SectorBreadthExperimentError(
            "Validation omitted a forecast on a breadth-ready row"
        )
    mature = ready & np.isfinite(label) & np.isfinite(actual_edge)
    if not mature.any():
        raise SectorBreadthExperimentError("Intermediate validation has no mature outcomes")
    for name, values in {
        "cash probability": probability,
        "expected edge": expected_edge,
    }.items():
        if not np.isfinite(values[ready]).all():
            raise SectorBreadthExperimentError(
                f"Validation {name} is not finite on breadth-ready rows"
            )
    for name, values in {
        "baseline probability": baseline_probability,
        "baseline edge": baseline_edge,
    }.items():
        if not np.isfinite(values[mature]).all():
            raise SectorBreadthExperimentError(
                f"Validation {name} is not finite on scored rows"
            )
    brier = float(np.mean(np.square(probability[mature] - label[mature])))
    baseline_brier = float(
        np.mean(np.square(baseline_probability[mature] - label[mature]))
    )
    edge_mae = float(np.mean(np.abs(expected_edge[mature] - actual_edge[mature])))
    baseline_edge_mae = float(
        np.mean(np.abs(baseline_edge[mature] - actual_edge[mature]))
    )
    cash = predictions["cash_target"].to_numpy(dtype=float)
    starts = predictions["cash_block_start"].to_numpy(dtype=float)
    mature_starts = (starts == 1.0) & mature
    episode_edges = actual_edge[mature_starts]
    episode_mean = float(episode_edges.mean()) if len(episode_edges) else None
    episode_win_rate = (
        float(np.mean(episode_edges > 1e-12)) if len(episode_edges) else None
    )
    comparison = compare_ledgers(strategy, benchmark, initial_cash=INITIAL_CASH)
    metrics = {
        "scoring_contract": "aapl-sector-breadth-intermediate-validation-v1",
        "cost_bps": float(cost_bps),
        "total_active_log_edge": float(active.sum()),
        "annual_active_log_edges": annual,
        "annual_buy_hold_returns": annual_buy_hold,
        **negative_year_gate,
        "annual_win_rate": float(np.mean(np.asarray(list(annual.values())) > 1e-12)),
        "fold_active_log_edges": folds,
        "positive_folds": int(sum(value > 1e-12 for value in folds.values())),
        "minimum_fold_active_log_edge": float(min(folds.values())),
        "brier_score": brier,
        "causal_training_mean_brier": baseline_brier,
        "expected_edge_mae": edge_mae,
        "causal_training_mean_edge_mae": baseline_edge_mae,
        "oof_cash_days": int(cash.sum()),
        "cash_day_rate": float(cash.mean()),
        "cash_episodes": int(starts.sum()),
        "episode_start_realized_edge_mean_10bps": episode_mean,
        "episode_start_realized_edge_win_rate_10bps": episode_win_rate,
        "same_ledger_comparison": comparison,
        "no_leverage_proof": assert_unleveraged_ledger(strategy),
    }
    checks = {
        "positive_total_active_log_edge": metrics["total_active_log_edge"] > 1e-12,
        "at_least_two_positive_fixed_folds": metrics["positive_folds"]
        >= VALIDATION_GATE_CONTRACT["minimum_positive_folds"],
        "annual_win_rate_at_least_60pct": metrics["annual_win_rate"]
        >= VALIDATION_GATE_CONTRACT["minimum_annual_win_rate"],
        "positive_active_edge_in_every_negative_buy_hold_year": bool(
            metrics["all_negative_buy_hold_years_have_positive_active_log_edge"]
        ),
        "brier_strictly_improves_causal_mean": brier < baseline_brier,
        "expected_edge_mae_strictly_improves_causal_mean": edge_mae
        < baseline_edge_mae,
        "episode_start_realized_edge_mean_positive": episode_mean is not None
        and episode_mean > 1e-12,
        "episode_start_realized_edge_win_rate_above_half": episode_win_rate
        is not None
        and episode_win_rate > 0.50,
        "minimum_four_cash_episodes": metrics["cash_episodes"]
        >= VALIDATION_GATE_CONTRACT["minimum_cash_episodes"],
        "cash_day_rate_at_most_20pct": metrics["cash_day_rate"]
        <= VALIDATION_GATE_CONTRACT["maximum_cash_day_rate"],
        "no_leverage_proof": bool(metrics["no_leverage_proof"]["passed"]),
    }
    return {
        "metrics": metrics,
        "gates": {
            "contract": dict(VALIDATION_GATE_CONTRACT),
            "checks": checks,
            "passed": bool(all(checks.values())),
        },
    }


def _score_intermediate_validation(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    deadline: _Deadline,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    target = _full_exposure_target(
        data.index,
        pd.DatetimeIndex(predictions.index),
        predictions["cash_target"].to_numpy(dtype=float),
    )
    benchmark_target = pd.Series(1.0, index=data.index, name="target_exposure")
    scenarios: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for scenario_name, bps in COST_SCENARIOS:
        strategy = _simulate(data, target, VALIDATION_PERIOD, cost_bps=bps)
        benchmark = _simulate(data, benchmark_target, VALIDATION_PERIOD, cost_bps=bps)
        scenarios[scenario_name] = _score_validation_scenario(
            strategy, benchmark, predictions, cost_bps=bps
        )
        ledgers[f"validation_strategy_{scenario_name}.csv"] = strategy
        ledgers[f"validation_buy_hold_{scenario_name}.csv"] = benchmark
        deadline.check(f"intermediate validation {scenario_name}")
    passed = bool(all(value["gates"]["passed"] for value in scenarios.values()))
    return {
        "stage": "frozen_2019_2023_intermediate_validation",
        "selection_or_tuning_performed": False,
        "cost_scenarios": scenarios,
        "passed": passed,
    }, ledgers


def _negative_and_stress_diagnostics(
    data: pd.DataFrame,
    target: pd.Series,
    *,
    deadline: _Deadline,
) -> dict[str, Any]:
    costs = CostAssumptions(slippage_bps=BASE_COST_BPS, annual_margin_rate=0.0)
    benchmark_target = pd.Series(1.0, index=data.index, name="target_exposure")
    negative_years: dict[str, Any] = {}
    for year in range(2005, 2024):
        period = EvaluationPeriod(str(year), f"{year}-01-01", f"{year}-12-31")
        strategy = simulate_unleveraged_period(data, target, period, costs)
        benchmark = simulate_unleveraged_period(data, benchmark_target, period, costs)
        report = _period_report(data, strategy, benchmark, initial_cash=INITIAL_CASH)
        benchmark_return = float(report["aapl_buy_hold"]["total_return"])
        if benchmark_return < 0.0:
            negative_years[str(year)] = {
                "strategy_return": float(report["strategy"]["total_return"]),
                "aapl_buy_hold_return": benchmark_return,
                "excess_return_vs_aapl_buy_hold": float(
                    report["excess_return_vs_aapl_buy_hold"]
                ),
                "beat_buy_hold": bool(report["requested_success"]),
                "strategy_positive_absolute_return": bool(
                    float(report["strategy"]["total_return"]) > 0.0
                ),
            }
        deadline.check(f"negative-year diagnostic {year}")

    named: dict[str, Any] = {}
    for period in BEAR_STRESS_PERIODS:
        if pd.Timestamp(period.start) < SELECTION_START or pd.Timestamp(period.end) > VALIDATION_END:
            named[period.name] = {
                "available": False,
                "reason": "outside the 2005-2023 chronological prediction span",
            }
            continue
        strategy = simulate_unleveraged_period(data, target, period, costs)
        benchmark = simulate_unleveraged_period(data, benchmark_target, period, costs)
        report = _period_report(data, strategy, benchmark, initial_cash=INITIAL_CASH)
        named[period.name] = {
            "available": True,
            "period": asdict(period),
            "strategy_return": float(report["strategy"]["total_return"]),
            "aapl_buy_hold_return": float(report["aapl_buy_hold"]["total_return"]),
            "excess_return_vs_aapl_buy_hold": float(
                report["excess_return_vs_aapl_buy_hold"]
            ),
            "beat_buy_hold": bool(report["requested_success"]),
        }
        deadline.check(f"stress diagnostic {period.name}")
    return {
        "role": "diagnostic_only_not_a_selection_or_validation_substitute",
        "calendar_year_inclusion_rule": (
            "every complete 2005-2023 year with negative same-ledger AAPL buy-and-hold"
        ),
        "negative_calendar_years": negative_years,
        "negative_calendar_year_count": int(len(negative_years)),
        "named_stress_episodes": named,
    }


def _policy_payload(policy: Any) -> bytes:
    if hasattr(policy, "canonical_json") and callable(policy.canonical_json):
        payload = policy.canonical_json()
        if not isinstance(payload, str):
            raise SectorBreadthExperimentError("Frozen policy canonical_json is not text")
        return (payload.rstrip("\n") + "\n").encode("utf-8")
    if hasattr(policy, "to_state") and callable(policy.to_state):
        return _pretty_json_bytes(policy.to_state())
    model = getattr(policy, "model", None)
    if model is not None and hasattr(model, "to_state"):
        metadata: dict[str, Any] = {}
        fields = getattr(policy, "__dataclass_fields__", {})
        for field_name in fields:
            if field_name != "model":
                metadata[field_name] = getattr(policy, field_name)
        return _pretty_json_bytes(
            {"policy_metadata": metadata, "model_state": model.to_state()}
        )
    raise SectorBreadthExperimentError("Frozen policy lacks canonical serialization")


def _safe_run_id(value: str | None, *, prefix: str) -> str:
    resolved = value or (
        f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resolved):
        raise SectorBreadthExperimentError("run_id is not a safe flat directory name")
    return resolved


def _manifest_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical = _canonical_json_bytes(payload)
    return {**dict(payload), "manifest_sha256": _sha256_tagged(canonical)}


def _combined_chronological_target(
    data: pd.DataFrame,
    selection_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    selected: CandidateSpec,
) -> pd.Series:
    target = pd.Series(1.0, index=data.index, name="target_exposure")
    selection_cash = selection_predictions[selected.cash_target_column].to_numpy(
        dtype=float
    )
    target.loc[selection_predictions.index] = 1.0 - selection_cash
    target.loc[validation_predictions.index] = 1.0 - validation_predictions[
        "cash_target"
    ].to_numpy(dtype=float)
    return target


def run_development_from_inputs(
    *,
    price_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    output_dir: Path,
    input_provenance: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run, gate, and seal development without accessing a 2024+ row."""

    deadline = _Deadline(clock)
    data, context, boundary_audit = _bounded_development_inputs(
        price_frame, context_frame
    )
    deadline.check("bounded input validation")
    feature_label = _validated_feature_label_frame(data, context)
    deadline.check("feature and label construction")
    try:
        selection_predictions = build_sector_breadth_walkforward_predictions(
            feature_label
        )
    except Exception as exc:
        raise SectorBreadthExperimentError("Fixed 2005-2018 walk-forward failed") from exc
    deadline.check("fixed 2005-2018 walk-forward fits")
    candidate_results, development_selected_id, selection_ledgers, _ = (
        _score_selection_candidates(data, selection_predictions, deadline=deadline)
    )
    deadline.check("selection candidate scoring")

    intermediate_policy: Any | None = None
    validation_predictions: pd.DataFrame | None = None
    validation_result: dict[str, Any] = {
        "stage": "frozen_2019_2023_intermediate_validation",
        "performed": False,
        "reason": "no_2005_2018_sector_breadth_candidate_passed",
        "passed": False,
    }
    validation_ledgers: dict[str, pd.DataFrame] = {}
    final_policy: Any | None = None
    diagnostics: dict[str, Any] = {
        "performed": False,
        "reason": "no_frozen_chronological_policy_available",
    }
    selected_spec: CandidateSpec | None = None

    if development_selected_id is not None:
        selected_spec = _candidate_by_id(development_selected_id)
        try:
            intermediate_policy = fit_intermediate_sector_breadth_policy(
                feature_label,
                selected_family=selected_spec.model_family,
                selected_candidate=selected_spec.candidate,
            )
            raw_validation = predict_intermediate_sector_breadth_validation(
                feature_label, intermediate_policy
            )
        except Exception as exc:
            raise SectorBreadthExperimentError(
                "Frozen 2019-2023 intermediate policy failed"
            ) from exc
        validation_predictions = _canonical_validation_predictions(
            raw_validation, feature_label, selected_spec
        )
        validation_result, validation_ledgers = _score_intermediate_validation(
            data, validation_predictions, deadline=deadline
        )
        combined_target = _combined_chronological_target(
            data, selection_predictions, validation_predictions, selected_spec
        )
        diagnostics = _negative_and_stress_diagnostics(
            data, combined_target, deadline=deadline
        )
        if validation_result["passed"]:
            try:
                final_policy = fit_final_sector_breadth_policy(
                    feature_label, selected_policy=intermediate_policy
                )
            except Exception as exc:
                raise SectorBreadthExperimentError(
                    "Final through-2023 frozen-policy refit failed"
                ) from exc
        deadline.check("intermediate validation and final refit")

    final_selected_id = development_selected_id if final_policy is not None else None
    if development_selected_id is None:
        no_winner_reason = "no_2005_2018_sector_breadth_candidate_passed"
    elif not validation_result["passed"]:
        no_winner_reason = "selected_candidate_failed_frozen_2019_2023_validation"
    elif final_policy is None:
        no_winner_reason = "final_policy_refit_was_not_sealed"
    else:
        no_winner_reason = None

    created_at = datetime.now(timezone.utc)
    resolved_run_id = _safe_run_id(run_id, prefix="sector-breadth-development")
    run_dir = output_dir.resolve() / resolved_run_id
    payloads: dict[str, bytes] = {
        ".gitattributes": b"* -text\n",
        "development_prices_through_2023.csv": _frame_csv_bytes(
            data.reset_index(names="date")
        ),
        "development_context_through_2023.csv": _frame_csv_bytes(
            context.reset_index(names="date")
        ),
        "feature_label_through_2023.csv": _frame_csv_bytes(
            feature_label.reset_index(names="decision_date")
        ),
        "selection_oof_predictions_2005_2018.csv": _frame_csv_bytes(
            selection_predictions.reset_index(names="decision_date")
        ),
        "selection_candidate_results.json": _pretty_json_bytes(candidate_results),
        "intermediate_validation_result.json": _pretty_json_bytes(validation_result),
        "negative_and_stress_diagnostics.json": _pretty_json_bytes(diagnostics),
        "input_provenance.json": _pretty_json_bytes(
            {
                "contract_version": CONTRACT_VERSION,
                "source": dict(input_provenance or {"source_type": "injected_inputs"}),
                "source_identity": dict(source_identity or {}),
                "boundary_audit": boundary_audit,
                "physical_data_end": DEVELOPMENT_DATA_END.date().isoformat(),
                "post_2023_market_data_accessed": False,
                "network_access": False,
            }
        ),
    }
    if validation_predictions is not None:
        payloads["intermediate_predictions_2019_2023.csv"] = _frame_csv_bytes(
            validation_predictions.reset_index(names="decision_date")
        )
    if intermediate_policy is not None:
        payloads["intermediate_frozen_policy.json"] = _policy_payload(
            intermediate_policy
        )
    if final_policy is not None:
        payloads["final_frozen_policy_through_2023.json"] = _policy_payload(final_policy)
    for filename, ledger in {**selection_ledgers, **validation_ledgers}.items():
        payloads[filename] = _frame_csv_bytes(ledger)

    payload_hashes = {
        name: _sha256_tagged(payload) for name, payload in sorted(payloads.items())
    }
    manifest_payload = {
        "contract_version": CONTRACT_VERSION,
        "stage": "development",
        "evidence_classification": (
            "chronological_internal_selection_2005_2018_then_frozen_intermediate_validation_2019_2023"
        ),
        "physical_data_end": DEVELOPMENT_DATA_END.date().isoformat(),
        "post_2023_market_data_access_allowed": False,
        "post_2023_market_data_accessed": False,
        "candidate_family": [item.to_dict() for item in CANDIDATE_SPECS],
        "selection_walk_forward_folds": [
            asdict(item) if is_dataclass(item) else str(item)
            for item in SECTOR_BREADTH_WALK_FORWARD_FOLDS
        ],
        "gam_config": asdict(DirectEdgeGAMConfig()),
        "cost_scenarios_bps": {name: value for name, value in COST_SCENARIOS},
        "development_selected_candidate_id": development_selected_id,
        "development_pass": development_selected_id is not None,
        "intermediate_validation_performed": bool(
            validation_result.get("performed", development_selected_id is not None)
        ),
        "intermediate_validation_pass": bool(validation_result.get("passed")),
        "selected_candidate_id": final_selected_id,
        "final_selection_frozen": final_selected_id is not None,
        "explicit_no_winner_gate": {
            "triggered": final_selected_id is None,
            "reason": no_winner_reason,
            "price_only_candidate_may_advance": False,
            "sector_breadth_must_pass_paired_price_ablation": True,
            "intermediate_validation_may_retune": False,
        },
        "selection_tie_break": [
            "highest 10bps minimum-fold active-log edge",
            "highest 10bps total active-log edge",
            "fewest 10bps cash episodes",
            "lexical candidate id",
        ],
        "frozen_policy": {
            "intermediate_filename": "intermediate_frozen_policy.json"
            if intermediate_policy is not None
            else None,
            "intermediate_sha256": payload_hashes.get(
                "intermediate_frozen_policy.json"
            ),
            "final_filename": "final_frozen_policy_through_2023.json"
            if final_policy is not None
            else None,
            "final_sha256": payload_hashes.get(
                "final_frozen_policy_through_2023.json"
            ),
            "maximum_training_label_maturity_date": (
                final_policy.training_max_label_maturity_date.date().isoformat()
                if final_policy is not None
                else None
            ),
        },
        "candidate_results_sha256": _sha256_tagged(
            _canonical_json_bytes(candidate_results)
        ),
        "intermediate_validation_result_sha256": _sha256_tagged(
            _canonical_json_bytes(validation_result)
        ),
        "payload_sha256": payload_hashes,
        "source_identity": dict(source_identity or {}),
        "execution": {
            "asset": "AAPL",
            "actions": ["LONG_100_PERCENT", "CASH_100_PERCENT"],
            "decision_information_cutoff": (
                "after the completed AAPL/ETF close and the completed 4:15pm ET "
                "VIX daily value on decision session t"
            ),
            "fill": "next adjusted AAPL open",
            "shorting": False,
            "leverage": False,
            "maximum_target_exposure": 1.0,
            "cash_interest_rate": 0.0,
        },
        "later_stage_lock": {
            "only_exact_committed_manifest_may_advance": True,
            "only_selected_candidate_may_advance": True,
            "candidate_grid_may_change": False,
            "thresholds_may_change": False,
            "features_may_change": False,
            "model_configuration_may_change": False,
        },
    }
    manifest = _manifest_with_hash(manifest_payload)
    payloads["selection_manifest.json"] = _pretty_json_bytes(manifest)
    report = {
        "contract_version": CONTRACT_VERSION,
        "run_id": resolved_run_id,
        "stage": "development",
        "artifact_dir": str(run_dir),
        "created_at_utc": created_at.isoformat(),
        "physical_data_end": DEVELOPMENT_DATA_END.date().isoformat(),
        "post_2023_market_data_accessed": False,
        "development_selected_candidate_id": development_selected_id,
        "intermediate_validation_pass": bool(validation_result.get("passed")),
        "selected_candidate_id": final_selected_id,
        "development_pass": final_selected_id is not None,
        "explicit_no_winner_gate": manifest["explicit_no_winner_gate"],
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "reproducibility": {
            "runtime_seconds_before_seal": deadline.elapsed(),
            "runtime_limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "completed_before_seal_within_limit": deadline.elapsed()
            <= RUN_TIME_LIMIT_SECONDS,
            "model_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "api_cost_display": "$0.00",
        },
    }
    payloads["report.json"] = _pretty_json_bytes(report)
    deadline.check("before artifact sealing")
    checksums = _seal_bundle(run_dir, payloads)
    deadline.check("artifact checksum readback")
    return {**report, "checksums": checksums}


def run_development_experiment(
    *,
    repo_root: Path,
    price_artifact: Path,
    context_parquet: Path,
    output_dir: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    loaded = load_public_development_inputs(
        price_artifact=price_artifact, context_parquet=context_parquet
    )
    return run_development_from_inputs(
        price_frame=loaded.price_frame,
        context_frame=loaded.context_frame,
        output_dir=output_dir,
        input_provenance=loaded.provenance,
        source_identity={"git": _git_state(repo_root.resolve())},
        run_id=run_id,
    )


def _git_bytes(repo_root: Path, *args: str, check: bool = True) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
    )
    return completed.stdout


def _git_text(repo_root: Path, *args: str, check: bool = True) -> str:
    return _git_bytes(repo_root, *args, check=check).decode("utf-8", errors="strict").strip()


def validate_final_selection_manifest(
    *, repo_root: Path, selection_manifest: Path
) -> ValidatedFinalSelection:
    """Validate final authority without reading any holdout market row."""

    root = repo_root.resolve()
    try:
        actual_root = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise SectorBreadthExperimentError("Final requires a Git repository") from exc
    if actual_root != root:
        raise SectorBreadthExperimentError("repo_root must be the actual Git repository root")
    status = _git_text(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise SectorBreadthExperimentError(
            "Final requires a completely clean branch, including no untracked files"
        )
    branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if not branch:
        raise SectorBreadthExperimentError("Final refuses a detached HEAD")
    commit = _git_text(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SectorBreadthExperimentError("Current Git commit is invalid")

    manifest_path = selection_manifest.resolve()
    try:
        relative = manifest_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise SectorBreadthExperimentError(
            "Final selection manifest must be inside repo_root"
        ) from exc
    try:
        _git_bytes(root, "ls-files", "--error-unmatch", "--", relative)
        committed_bytes = _git_bytes(root, "show", f"HEAD:{relative}")
        local_bytes = manifest_path.read_bytes()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SectorBreadthExperimentError(
            "Final selection manifest must be tracked at the current commit"
        ) from exc
    if committed_bytes != local_bytes:
        raise SectorBreadthExperimentError(
            "Final selection manifest differs from the exact committed Git blob"
        )
    try:
        manifest = json.loads(local_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SectorBreadthExperimentError("Selection manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise SectorBreadthExperimentError("Selection manifest must be a JSON object")
    recorded_hash = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    expected_hash = _sha256_tagged(_canonical_json_bytes(payload))
    if recorded_hash != expected_hash:
        raise SectorBreadthExperimentError("Selection manifest self-hash is invalid")
    if manifest.get("contract_version") != CONTRACT_VERSION or manifest.get("stage") != "development":
        raise SectorBreadthExperimentError("Selection manifest contract or stage is invalid")
    required_true = {
        "development_pass": manifest.get("development_pass"),
        "intermediate_validation_pass": manifest.get("intermediate_validation_pass"),
        "final_selection_frozen": manifest.get("final_selection_frozen"),
    }
    failed = sorted(name for name, value in required_true.items() if value is not True)
    if failed:
        raise SectorBreadthExperimentError(
            "Final refuses an unselected or failed manifest: " + ", ".join(failed)
        )
    selected_id = manifest.get("selected_candidate_id")
    if not isinstance(selected_id, str) or not selected_id.startswith("sector_breadth_"):
        raise SectorBreadthExperimentError(
            "Final requires an explicitly selected sector-breadth candidate"
        )
    no_winner = manifest.get("explicit_no_winner_gate")
    if not isinstance(no_winner, Mapping) or no_winner.get("triggered") is not False:
        raise SectorBreadthExperimentError("Final manifest still has the no-winner gate engaged")
    policy = manifest.get("frozen_policy")
    if not isinstance(policy, Mapping):
        raise SectorBreadthExperimentError("Final manifest lacks a frozen-policy binding")
    filename = policy.get("final_filename")
    expected_policy_hash = policy.get("final_sha256")
    if not isinstance(filename, str) or not filename or not isinstance(expected_policy_hash, str):
        raise SectorBreadthExperimentError("Final frozen-policy binding is incomplete")
    policy_path = manifest_path.parent / filename
    try:
        policy_relative = policy_path.resolve().relative_to(root).as_posix()
        _git_bytes(root, "ls-files", "--error-unmatch", "--", policy_relative)
        committed_policy = _git_bytes(root, "show", f"HEAD:{policy_relative}")
        local_policy = policy_path.read_bytes()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SectorBreadthExperimentError(
            "Frozen final policy must be an exact tracked file at the current commit"
        ) from exc
    if committed_policy != local_policy or _sha256_tagged(local_policy) != expected_policy_hash:
        raise SectorBreadthExperimentError("Frozen final policy does not match its manifest")
    return ValidatedFinalSelection(
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=expected_hash,
        git_commit=commit,
        git_branch=branch,
    )


def _load_final_price_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = path.resolve()
    if not source.is_file():
        raise SectorBreadthExperimentError(f"Final price CSV does not exist: {source}")
    try:
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            raw = connection.execute(
                FINAL_PRICE_QUERY,
                [
                    str(source),
                    DEVELOPMENT_INPUT_START.date().isoformat(),
                    FINAL_DATA_END.date().isoformat(),
                ],
            ).fetchdf()
        finally:
            connection.close()
    except Exception as exc:
        raise SectorBreadthExperimentError("Bounded final price query failed") from exc
    dates = _normalize_index(raw, name="final price query")
    if not len(dates) or dates.max() > FINAL_DATA_END or dates.max() < FINAL_DATA_END:
        raise SectorBreadthExperimentError("Final price query lacks the exact frozen end date")
    try:
        frame = canonical_context_frame(raw)
    except (TypeError, ValueError) as exc:
        raise SectorBreadthExperimentError("Final price rows are not canonical") from exc
    return frame, {
        "source_type": "local_csv_bounded_duckdb_query",
        "source_path": str(source),
        "source_file_sha256": f"sha256:{file_sha256(source)}",
        "bounded_first_date": frame.index.min().date().isoformat(),
        "bounded_last_date": frame.index.max().date().isoformat(),
        "bounded_rows": int(len(frame)),
    }


def _load_final_inputs(
    *, price_artifact: Path, context_parquet: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prices, price_provenance = _load_final_price_csv(price_artifact)
    try:
        context = load_sector_context_parquet(
            context_parquet, start=CONTEXT_INPUT_START, end=FINAL_DATA_END
        )
    except Exception as exc:
        raise SectorBreadthExperimentError("Bounded final context query failed") from exc
    index = _normalize_index(context, name="final context")
    expected = prices.index[
        (prices.index >= CONTEXT_INPUT_START) & (prices.index <= FINAL_DATA_END)
    ]
    if not index.equals(expected):
        raise SectorBreadthExperimentError(
            "Final context does not exactly cover every expected price session"
        )
    context = context.copy()
    context.index = index
    return prices, context, {
        "price": price_provenance,
        "context": {
            "source_path": str(context_parquet.resolve()),
            "source_file_sha256": f"sha256:{file_sha256(context_parquet.resolve())}",
            **_file_timestamp_provenance(context_parquet.resolve()),
            "download_log": _context_download_log_provenance(context_parquet),
            "bounded_first_date": index.min().date().isoformat(),
            "bounded_last_date": index.max().date().isoformat(),
            "bounded_rows": int(len(context)),
        },
    }


def _build_final_predictions(
    feature_label: pd.DataFrame, policy: FrozenSectorBreadthPolicy
) -> pd.DataFrame:
    if policy.training_boundary != pd.Timestamp("2024-01-01"):
        raise SectorBreadthExperimentError("Final policy was not frozen at the 2024 boundary")
    if policy.model_family != "sector_breadth":
        raise SectorBreadthExperimentError("Final policy is not the selected breadth family")
    period = feature_label.loc[
        (feature_label.index >= pd.Timestamp(FINAL_PERIODS[0].start))
        & (feature_label.index <= FINAL_DATA_END)
    ].copy()
    if not set(range(2024, FINAL_DATA_END.year + 1)).issubset(set(period.index.year)):
        raise SectorBreadthExperimentError("Final feature frame lacks a required holdout year")
    ready = (
        period["price_features_ready"].to_numpy(dtype=bool)
        & period["sector_breadth_features_ready"].to_numpy(dtype=bool)
    )
    features = period.loc[:, list(policy.feature_names)]
    finite = np.isfinite(features.to_numpy(dtype=float)).all(axis=1)
    ready &= finite
    probability = np.full(len(period), np.nan, dtype=float)
    edge = np.full(len(period), np.nan, dtype=float)
    if ready.any():
        components = policy.model.predict_components(features.loc[ready])
        probability[ready] = np.asarray(
            components["cash_beats_long_probability"], dtype=float
        )
        edge[ready] = np.asarray(components["expected_edge_10bps"], dtype=float)
    trigger = (
        ready
        & (probability >= policy.candidate.probability_gate)
        & (edge >= policy.candidate.expected_edge_gate)
    )
    cash, starts = five_session_cash_policy(trigger)
    output = pd.DataFrame(index=period.index)
    output.index.name = "decision_date"
    output["model_family"] = policy.model_family
    output["candidate_id"] = policy.candidate_id
    output["model_sha256"] = policy.model_sha256
    output["training_sha256"] = policy.training_sha256
    output["breadth_ready"] = ready
    output["cash_win_probability"] = probability
    output["expected_net_edge"] = edge
    output["cash_target"] = cash
    output["cash_block_start"] = starts
    return output


def _evaluate_final_scenario(
    data: pd.DataFrame,
    target: pd.Series,
    *,
    cost_bps: float,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    benchmark_target = pd.Series(1.0, index=data.index, name="target_exposure")
    periods: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for period in FINAL_PERIODS:
        strategy = _simulate(data, target, period, cost_bps=cost_bps)
        benchmark = _simulate(data, benchmark_target, period, cost_bps=cost_bps)
        report = _period_report(data, strategy, benchmark, initial_cash=INITIAL_CASH)
        report["period"] = asdict(period)
        report["no_leverage_proof"] = assert_unleveraged_ledger(strategy)
        periods[period.name] = report
        merged = pd.concat(
            [strategy.add_prefix("strategy_"), benchmark.add_prefix("buy_hold_")],
            axis=1,
        )
        merged["active_daily_return"] = (
            merged["strategy_daily_return"] - merged["buy_hold_daily_return"]
        )
        ledgers[f"final_{period.name}_{int(cost_bps)}bps.csv"] = merged

    continuous_period = EvaluationPeriod(
        "continuous", FINAL_PERIODS[0].start, FINAL_PERIODS[-1].end
    )
    continuous_strategy = _simulate(
        data, target, continuous_period, cost_bps=cost_bps
    )
    continuous_benchmark = _simulate(
        data, benchmark_target, continuous_period, cost_bps=cost_bps
    )
    strategy_returns = _continuous_period_returns(
        continuous_strategy, FINAL_PERIODS, initial_cash=INITIAL_CASH
    )
    benchmark_returns = _continuous_period_returns(
        continuous_benchmark, FINAL_PERIODS, initial_cash=INITIAL_CASH
    )
    active_log = {
        name: float(math.log1p(strategy_returns[name]) - math.log1p(benchmark_returns[name]))
        for name in strategy_returns
    }
    continuous = {
        "period": asdict(continuous_period),
        "strategy_period_returns": strategy_returns,
        "buy_hold_period_returns": benchmark_returns,
        "period_active_log_returns": active_log,
        "all_periods_positive_active_log_return": all(value > 0.0 for value in active_log.values()),
        "full_span": compare_ledgers(
            continuous_strategy, continuous_benchmark, initial_cash=INITIAL_CASH
        ),
        "no_leverage_proof": assert_unleveraged_ledger(continuous_strategy),
    }
    ledgers[f"final_continuous_{int(cost_bps)}bps.csv"] = pd.concat(
        [
            continuous_strategy.add_prefix("strategy_"),
            continuous_benchmark.add_prefix("buy_hold_"),
        ],
        axis=1,
    )
    return {
        "cost_bps": float(cost_bps),
        "fresh_accounts": periods,
        "continuous_account": continuous,
        "all_fresh_periods_beat_buy_hold": all(
            bool(item["requested_success"]) for item in periods.values()
        ),
        "all_continuous_periods_beat_buy_hold": bool(
            continuous["all_periods_positive_active_log_return"]
        ),
    }, ledgers


def run_final_experiment(
    *,
    repo_root: Path,
    selection_manifest: Path,
    price_artifact: Path,
    context_parquet: Path,
    output_dir: Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate the three holdouts once after exact committed authorization."""

    deadline = _Deadline()
    authority = validate_final_selection_manifest(
        repo_root=repo_root, selection_manifest=selection_manifest
    )
    policy_info = authority.manifest["frozen_policy"]
    policy_path = authority.manifest_path.parent / str(policy_info["final_filename"])
    try:
        policy = FrozenSectorBreadthPolicy.load(policy_path)
    except Exception as exc:
        raise SectorBreadthExperimentError("Committed frozen final policy cannot be loaded") from exc
    selected_id = str(authority.manifest["selected_candidate_id"])
    if policy.candidate_id != selected_id:
        raise SectorBreadthExperimentError("Frozen policy candidate differs from selection manifest")
    if not price_artifact.resolve().is_file() or not context_parquet.resolve().is_file():
        raise SectorBreadthExperimentError("Final local market inputs are missing")
    source_file_hashes = {
        "price": file_sha256(price_artifact.resolve()),
        "context": file_sha256(context_parquet.resolve()),
    }
    data_hash = hashlib.sha256(
        _canonical_json_bytes(source_file_hashes)
    ).hexdigest()
    holdout_touch = reserve_holdout_touch(
        repo_root.resolve() / "data" / "unleveraged_aapl" / "holdout_registry.json",
        candidate_hash=authority.manifest_sha256.removeprefix("sha256:"),
        strategy_name=selected_id,
        data_hash=data_hash,
        git_commit=authority.git_commit,
    )
    holdout_touch.pop("_reserved_registry_json", None)
    deadline.check("holdout reservation")
    data, context, input_provenance = _load_final_inputs(
        price_artifact=price_artifact, context_parquet=context_parquet
    )
    try:
        feature_label = build_sector_breadth_feature_label_frame(data, context)
    except Exception as exc:
        raise SectorBreadthExperimentError("Final feature construction failed") from exc
    predictions = _build_final_predictions(feature_label, policy)
    target = _full_exposure_target(
        data.index, predictions.index, predictions["cash_target"].to_numpy(dtype=float)
    )
    deadline.check("frozen holdout prediction")
    scenarios: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for scenario_name, bps in COST_SCENARIOS:
        scenario, scenario_ledgers = _evaluate_final_scenario(
            data, target, cost_bps=bps
        )
        scenarios[scenario_name] = scenario
        ledgers.update(scenario_ledgers)
        deadline.check(f"final {scenario_name} evaluation")
    promotion_pass = bool(
        all(
            value["all_fresh_periods_beat_buy_hold"]
            and value["all_continuous_periods_beat_buy_hold"]
            for value in scenarios.values()
        )
    )
    resolved_run_id = _safe_run_id(run_id, prefix="sector-breadth-final")
    run_dir = output_dir.resolve() / resolved_run_id
    report = {
        "contract_version": CONTRACT_VERSION,
        "run_id": resolved_run_id,
        "stage": "final",
        "artifact_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_manifest_sha256": authority.manifest_sha256,
        "selected_candidate_id": selected_id,
        "git": {"commit": authority.git_commit, "branch": authority.git_branch},
        "holdout_touch": holdout_touch,
        "periods": [asdict(item) for item in FINAL_PERIODS],
        "scenarios": scenarios,
        "promotion_pass": promotion_pass,
        "learning_or_refitting_after_2023": False,
        "reproducibility": {
            "runtime_seconds_before_seal": deadline.elapsed(),
            "runtime_limit_seconds": RUN_TIME_LIMIT_SECONDS,
            "model_calls": 0,
            "estimated_external_cost_usd": 0.0,
            "api_cost_display": "$0.00",
        },
    }
    payloads: dict[str, bytes] = {
        ".gitattributes": b"* -text\n",
        "report.json": _pretty_json_bytes(report),
        "authorized_selection_manifest.json": authority.manifest_path.read_bytes(),
        "final_predictions.csv": _frame_csv_bytes(
            predictions.reset_index(names="decision_date")
        ),
        "input_provenance.json": _pretty_json_bytes(input_provenance),
        "holdout_touch.json": _pretty_json_bytes(holdout_touch),
    }
    for filename, ledger in ledgers.items():
        payloads[filename] = _frame_csv_bytes(ledger)
    deadline.check("before final artifact sealing")
    checksums = _seal_bundle(run_dir, payloads)
    return {**report, "checksums": checksums}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sealed AAPL sector-breadth experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("develop", "final"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo-root", type=Path, required=True)
        child.add_argument("--price-artifact", type=Path, required=True)
        child.add_argument("--context-parquet", type=Path, required=True)
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--run-id")
        if command == "final":
            child.add_argument("--selection-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "develop":
        report = run_development_experiment(
            repo_root=args.repo_root.resolve(),
            price_artifact=args.price_artifact.resolve(),
            context_parquet=args.context_parquet.resolve(),
            output_dir=args.output_dir.resolve(),
            run_id=args.run_id,
        )
        summary = {
            "run_id": report["run_id"],
            "stage": "development",
            "selected_candidate_id": report["selected_candidate_id"],
            "development_pass": report["development_pass"],
            "artifact_dir": report["artifact_dir"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["development_pass"] else 2
    report = run_final_experiment(
        repo_root=args.repo_root.resolve(),
        selection_manifest=args.selection_manifest.resolve(),
        price_artifact=args.price_artifact.resolve(),
        context_parquet=args.context_parquet.resolve(),
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "stage": "final",
                "selected_candidate_id": report["selected_candidate_id"],
                "promotion_pass": report["promotion_pass"],
                "artifact_dir": report["artifact_dir"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["promotion_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
