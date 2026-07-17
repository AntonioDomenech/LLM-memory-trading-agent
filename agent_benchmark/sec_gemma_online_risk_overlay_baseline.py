"""Pure adapter for the frozen fixed exhaustion-union baseline.

Only the completed-close ``unfiltered_union_signal`` is projected.  The old
end-of-stage actionable target is deliberately forbidden because its final two
rows depend on whether future fills are physically present in that batch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import date
import hmac
import math
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    BASELINE_POLICY_ID,
    BASELINE_SOURCE_FILE,
    BASELINE_SOURCE_SHA256,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_ledger import (
    build_baseline_signal_row,
)


BASELINE_INPUT_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-baseline-input-row-v1"
)
BASELINE_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-baseline-signal-batch-v1"
)
BASELINE_INPUT_FIELDS: Final[tuple[str, ...]] = (
    "aapl_raw_open",
    "aapl_raw_close",
    "aapl_adjusted_close",
    "spy_adjusted_close",
    "qqq_adjusted_close",
)
_CONTEXTUAL_PERCENTILE: Final[float] = 0.90
_WEAK_TREND_PERCENTILE: Final[float] = 0.925
_INTRADAY_LOOKBACK: Final[int] = 126
_CONTEXTUAL_MARKET_LOOKBACK: Final[int] = 10
_WEAK_TREND_MARKET_LOOKBACK: Final[int] = 20
_WEAK_TREND_SMA_LOOKBACK: Final[int] = 20


class SecGemmaOnlineRiskOverlayBaselineError(ValueError):
    """Raised when baseline evidence or output differs from the frozen rule."""


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayBaselineError(
            f"{location} must be an ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayBaselineError(
            f"{location} must be an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecGemmaOnlineRiskOverlayBaselineError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _decode_positive_float_hex(value: Any, location: str) -> float:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayBaselineError(
            f"{location} must be canonical float.hex text"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayBaselineError(
            f"{location} must be canonical float.hex text"
        ) from exc
    if (
        not math.isfinite(number)
        or number <= 0.0
        or number.hex() != value
    ):
        raise SecGemmaOnlineRiskOverlayBaselineError(
            f"{location} must be a canonical positive finite float"
        )
    return number


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SecGemmaOnlineRiskOverlayBaselineError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def build_baseline_input_row(
    *,
    session: str,
    aapl_raw_open: float,
    aapl_raw_close: float,
    aapl_adjusted_close: float,
    spy_adjusted_close: float,
    qqq_adjusted_close: float,
    market_evidence_row_sha256: str,
) -> dict[str, Any]:
    """Build one exact row projected from authenticated market evidence."""

    session = _iso_date(session, "baseline input session")
    market_hash = _sha256(
        market_evidence_row_sha256, "market_evidence_row_sha256"
    )
    values = {
        "aapl_raw_open_hex": float(aapl_raw_open).hex(),
        "aapl_raw_close_hex": float(aapl_raw_close).hex(),
        "aapl_adjusted_close_hex": float(aapl_adjusted_close).hex(),
        "spy_adjusted_close_hex": float(spy_adjusted_close).hex(),
        "qqq_adjusted_close_hex": float(qqq_adjusted_close).hex(),
    }
    for name, encoded in values.items():
        _decode_positive_float_hex(encoded, name)
    body = {
        "schema_version": BASELINE_INPUT_ROW_SCHEMA_VERSION,
        "session": session,
        **values,
        "market_evidence_row_sha256": market_hash,
    }
    return {**body, "baseline_input_row_sha256": canonical_sha256(body)}


def _validated_input_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise SecGemmaOnlineRiskOverlayBaselineError(
            "baseline input rows must be a sequence"
        )
    expected = {
        "schema_version",
        "session",
        "aapl_raw_open_hex",
        "aapl_raw_close_hex",
        "aapl_adjusted_close_hex",
        "spy_adjusted_close_hex",
        "qqq_adjusted_close_hex",
        "market_evidence_row_sha256",
        "baseline_input_row_sha256",
    }
    result: list[dict[str, Any]] = []
    previous: str | None = None
    for ordinal, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise SecGemmaOnlineRiskOverlayBaselineError(
                f"baseline input row {ordinal} must be a mapping"
            )
        row = dict(raw)
        if set(row) != expected:
            raise SecGemmaOnlineRiskOverlayBaselineError(
                "baseline input row keys changed"
            )
        if row["schema_version"] != BASELINE_INPUT_ROW_SCHEMA_VERSION:
            raise SecGemmaOnlineRiskOverlayBaselineError(
                "baseline input row schema changed"
            )
        session = _iso_date(
            row["session"], f"baseline input row {ordinal}.session"
        )
        if previous is not None and session <= previous:
            raise SecGemmaOnlineRiskOverlayBaselineError(
                "baseline input sessions must be strictly increasing"
            )
        for field in BASELINE_INPUT_FIELDS:
            _decode_positive_float_hex(
                row[f"{field}_hex"],
                f"baseline input row {ordinal}.{field}_hex",
            )
        _sha256(
            row["market_evidence_row_sha256"],
            f"baseline input row {ordinal}.market_evidence_row_sha256",
        )
        observed = _sha256(
            row["baseline_input_row_sha256"],
            f"baseline input row {ordinal}.baseline_input_row_sha256",
        )
        body = {
            key: row[key]
            for key in row
            if key != "baseline_input_row_sha256"
        }
        if not hmac.compare_digest(observed, canonical_sha256(body)):
            raise SecGemmaOnlineRiskOverlayBaselineError(
                "baseline input row self-hash changed"
            )
        result.append(copy.deepcopy(row))
        previous = session
    if not result:
        raise SecGemmaOnlineRiskOverlayBaselineError(
            "baseline input rows must not be empty"
        )
    return result


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise SecGemmaOnlineRiskOverlayBaselineError(
            "baseline quantile window must not be empty"
        )
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def _canonicalize_one_session_signals(
    raw: Sequence[bool],
) -> list[bool]:
    accepted = [False] * len(raw)
    previous_accepted = -2
    for position, active in enumerate(raw):
        if active and position != previous_accepted + 1:
            accepted[position] = True
            previous_accepted = position
    return accepted


def _fixed_expert_signals(
    values: Sequence[Mapping[str, Any]],
) -> dict[str, list[bool]]:
    opens = [
        float.fromhex(row["aapl_raw_open_hex"]) for row in values
    ]
    closes = [
        float.fromhex(row["aapl_raw_close_hex"]) for row in values
    ]
    adjusted = [
        float.fromhex(row["aapl_adjusted_close_hex"]) for row in values
    ]
    spy = [
        float.fromhex(row["spy_adjusted_close_hex"]) for row in values
    ]
    qqq = [
        float.fromhex(row["qqq_adjusted_close_hex"]) for row in values
    ]
    intraday = [
        close / open_value - 1.0
        for open_value, close in zip(opens, closes, strict=True)
    ]
    contextual_raw: list[bool] = []
    weak_trend_raw: list[bool] = []
    for position in range(len(values)):
        contextual_ready = position >= _INTRADAY_LOOKBACK
        if contextual_ready:
            contextual_threshold = _linear_quantile(
                intraday[
                    position - _INTRADAY_LOOKBACK : position
                ],
                _CONTEXTUAL_PERCENTILE,
            )
            contextual_spy_return = (
                spy[position]
                / spy[position - _CONTEXTUAL_MARKET_LOOKBACK]
                - 1.0
            )
            contextual_qqq_return = (
                qqq[position]
                / qqq[position - _CONTEXTUAL_MARKET_LOOKBACK]
                - 1.0
            )
        else:
            contextual_threshold = 0.0
            contextual_spy_return = 0.0
            contextual_qqq_return = 0.0
        contextual_raw.append(
            bool(
                contextual_ready
                and intraday[position] > contextual_threshold
                and contextual_spy_return < 0.0
                and contextual_qqq_return < 0.0
            )
        )

        weak_ready = (
            position >= _INTRADAY_LOOKBACK
            and position >= _WEAK_TREND_MARKET_LOOKBACK
            and position + 1 >= _WEAK_TREND_SMA_LOOKBACK
        )
        if weak_ready:
            weak_threshold = _linear_quantile(
                intraday[
                    position - _INTRADAY_LOOKBACK : position
                ],
                _WEAK_TREND_PERCENTILE,
            )
            weak_spy_return = (
                spy[position]
                / spy[position - _WEAK_TREND_MARKET_LOOKBACK]
                - 1.0
            )
            weak_qqq_return = (
                qqq[position]
                / qqq[position - _WEAK_TREND_MARKET_LOOKBACK]
                - 1.0
            )
            aapl_sma = sum(
                adjusted[
                    position - _WEAK_TREND_SMA_LOOKBACK + 1 :
                    position + 1
                ]
            ) / _WEAK_TREND_SMA_LOOKBACK
        else:
            weak_threshold = 0.0
            weak_spy_return = 0.0
            weak_qqq_return = 0.0
            aapl_sma = 0.0
        weak_trend_raw.append(
            bool(
                weak_ready
                and intraday[position] > weak_threshold
                and weak_spy_return < 0.0
                and weak_qqq_return < 0.0
                and adjusted[position] < aapl_sma
            )
        )
    contextual_virtual = _canonicalize_one_session_signals(
        contextual_raw
    )
    weak_trend_virtual = _canonicalize_one_session_signals(
        weak_trend_raw
    )
    union_candidate = [
        contextual or weak
        for contextual, weak in zip(
            contextual_virtual,
            weak_trend_virtual,
            strict=True,
        )
    ]
    return {
        "contextual_raw_signal": contextual_raw,
        "contextual_virtual_signal": contextual_virtual,
        "weak_trend_raw_signal": weak_trend_raw,
        "weak_trend_virtual_signal": weak_trend_virtual,
        "unfiltered_union_candidate_signal": union_candidate,
        "unfiltered_union_signal": _canonicalize_one_session_signals(
            union_candidate
        ),
    }


def build_frozen_baseline_signal_batch(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay the pinned fixed baseline over one continuous market prefix."""

    values = _validated_input_rows(rows)
    projected = _fixed_expert_signals(values)
    signals = [
        build_baseline_signal_row(
            session=row["session"],
            unfiltered_union_signal=projected[
                "unfiltered_union_signal"
            ][position],
        )
        for position, row in enumerate(values)
    ]
    diagnostics = [
        {
            "session": row["session"],
            "contextual_raw_signal": projected[
                "contextual_raw_signal"
            ][position],
            "contextual_virtual_signal": projected[
                "contextual_virtual_signal"
            ][position],
            "weak_trend_raw_signal": projected[
                "weak_trend_raw_signal"
            ][position],
            "weak_trend_virtual_signal": projected[
                "weak_trend_virtual_signal"
            ][position],
            "unfiltered_union_candidate_signal": projected[
                "unfiltered_union_candidate_signal"
            ][position],
            "unfiltered_union_signal": signals[position][
                "unfiltered_union_signal"
            ],
        }
        for position, row in enumerate(values)
    ]
    body = {
        "schema_version": BASELINE_BATCH_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "baseline_policy_id": BASELINE_POLICY_ID,
        "baseline_source_file": BASELINE_SOURCE_FILE,
        "baseline_source_sha256": BASELINE_SOURCE_SHA256,
        "input_row_count": len(values),
        "input_rows_sha256": canonical_sha256(values),
        "signal_rows": signals,
        "signal_rows_sha256": canonical_sha256(signals),
        "diagnostic_rows": diagnostics,
        "diagnostic_rows_sha256": canonical_sha256(diagnostics),
        "forbidden_actionable_target_used": False,
    }
    return {**body, "baseline_batch_sha256": canonical_sha256(body)}


def validate_frozen_baseline_signal_batch(
    batch: Mapping[str, Any],
    *,
    expected_baseline_batch_sha256: str,
    input_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Rebuild and require exact prefix-invariant baseline evidence."""

    if not isinstance(batch, Mapping):
        raise SecGemmaOnlineRiskOverlayBaselineError(
            "baseline batch must be a mapping"
        )
    rebuilt = build_frozen_baseline_signal_batch(input_rows)
    if dict(batch) != rebuilt:
        raise SecGemmaOnlineRiskOverlayBaselineError(
            "baseline batch differs from deterministic replay"
        )
    observed = _sha256(
        batch.get("baseline_batch_sha256"), "baseline batch hash"
    )
    if not hmac.compare_digest(
        observed,
        _sha256(expected_baseline_batch_sha256, "expected baseline hash"),
    ):
        raise SecGemmaOnlineRiskOverlayBaselineError(
            "baseline batch is not externally pinned"
        )
    return observed


__all__ = [
    "BASELINE_BATCH_SCHEMA_VERSION",
    "BASELINE_INPUT_FIELDS",
    "BASELINE_INPUT_ROW_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayBaselineError",
    "build_baseline_input_row",
    "build_frozen_baseline_signal_batch",
    "validate_frozen_baseline_signal_batch",
]
