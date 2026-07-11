from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping as ABCMapping
import hashlib
import math
from typing import Any, Mapping

import pytest

from agent_benchmark.sec_filing_gemma_market_evidence import (
    MARKET_FIELDS,
    MARKET_SYMBOLS,
    build_market_source_manifest,
    build_market_stage_manifest,
    encode_float_hex,
)
from agent_benchmark.sec_filing_gemma_market_source_bytes import (
    MarketSourceBytesError,
    build_market_source_snapshot_bytes,
    validate_market_source_bytes_against_stage,
)
from tests import test_sec_filing_gemma_market_evidence as scaffold


def _snapshot_rows(
    raw_rows: list[dict[str, Any]], symbol: str
) -> list[dict[str, str]]:
    return [
        {
            "session": row["session"],
            **{
                f"{field}_hex": encode_float_hex(
                    row["observations"][symbol][field],
                    f"{symbol}.{field}",
                )
                for field in MARKET_FIELDS
            },
        }
        for row in raw_rows
        if row["observations"][symbol]["available"] is True
    ]


def _case(
    *,
    raw_rows: list[dict[str, Any]] | None = None,
    artifact_rows_by_symbol: Mapping[str, list[dict[str, str]]] | None = None,
    window_rows_by_symbol: Mapping[str, list[dict[str, str]]] | None = None,
) -> dict[str, Any]:
    rows = scaffold._raw_rows("development") if raw_rows is None else raw_rows
    default_rows = {symbol: _snapshot_rows(rows, symbol) for symbol in MARKET_SYMBOLS}
    artifact_rows = default_rows if artifact_rows_by_symbol is None else artifact_rows_by_symbol
    window_rows = default_rows if window_rows_by_symbol is None else window_rows_by_symbol
    artifacts = {
        symbol: build_market_source_snapshot_bytes(
            symbol=symbol,
            rows=artifact_rows[symbol],
        )
        for symbol in MARKET_SYMBOLS
    }
    windows = {
        symbol: build_market_source_snapshot_bytes(
            symbol=symbol,
            rows=window_rows[symbol],
        )
        for symbol in MARKET_SYMBOLS
    }
    specs = {
        symbol: {
            "artifact_sha256": hashlib.sha256(artifacts[symbol]).hexdigest(),
            "artifact_bytes": len(artifacts[symbol]),
            "window_sha256": hashlib.sha256(windows[symbol]).hexdigest(),
            "window_bytes": len(windows[symbol]),
            "available_sessions": [row["session"] for row in window_rows[symbol]],
        }
        for symbol in MARKET_SYMBOLS
    }
    source = build_market_source_manifest(
        artifact_stage="development",
        source_artifacts=specs,
    )
    stage = build_market_stage_manifest(
        artifact_stage="development",
        source_manifest=source,
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        rows=rows,
    )
    return {
        "rows": rows,
        "artifacts": artifacts,
        "windows": windows,
        "source": source,
        "stage": stage,
    }


@pytest.fixture(scope="module")
def case() -> dict[str, Any]:
    return _case()


def _validate(case: Mapping[str, Any]) -> dict[str, Any]:
    return validate_market_source_bytes_against_stage(
        artifact_bytes_by_symbol=case["artifacts"],
        window_bytes_by_symbol=case["windows"],
        source_manifest=case["source"],
        stage_manifest=case["stage"],
        expected_artifact_stage="development",
        expected_source_manifest_sha256=case["source"]["source_manifest_sha256"],
        expected_market_stage_manifest_sha256=case["stage"][
            "market_stage_manifest_sha256"
        ],
    )


def test_exact_source_bytes_reconcile_every_available_stage_value(case) -> None:
    receipt = _validate(case)
    assert receipt["artifact_stage"] == "development"
    assert receipt["value_reconciliation"] == (
        "all_available_ohlcv_fields_exact_float_hex"
    )
    assert receipt["reconciled_row_counts"]["AAPL"] == len(case["rows"])
    assert set(receipt["artifact_snapshot_sha256s"]) == set(MARKET_SYMBOLS)


def test_one_ulp_stage_mutation_fails_even_when_stage_is_rebuilt(case) -> None:
    changed_rows = copy.deepcopy(case["rows"])
    value = changed_rows[100]["observations"]["AAPL"]["open"]
    changed_rows[100]["observations"]["AAPL"]["open"] = math.nextafter(
        float(value), math.inf
    )
    changed = dict(case)
    changed["stage"] = build_market_stage_manifest(
        artifact_stage="development",
        source_manifest=case["source"],
        expected_source_manifest_sha256=case["source"]["source_manifest_sha256"],
        rows=changed_rows,
    )
    with pytest.raises(MarketSourceBytesError, match="do not reconcile"):
        _validate(changed)


def test_rehashed_window_substitution_cannot_detach_from_artifact(case) -> None:
    window_rows = {
        symbol: _snapshot_rows(case["rows"], symbol) for symbol in MARKET_SYMBOLS
    }
    changed = copy.deepcopy(window_rows["SPY"])
    changed[10]["close_hex"] = encode_float_hex(
        float.fromhex(changed[10]["close_hex"]) + 0.125,
        "changed SPY close",
    )
    window_rows["SPY"] = changed
    substituted = _case(window_rows_by_symbol=window_rows)
    with pytest.raises(MarketSourceBytesError, match="exact cumulative stage slice"):
        _validate(substituted)


def test_future_row_in_stage_artifact_is_rejected_before_reconciliation(case) -> None:
    artifact_rows = {
        symbol: _snapshot_rows(case["rows"], symbol) for symbol in MARKET_SYMBOLS
    }
    future_session = next(
        session for session in scaffold.EXPECTED_MARKET_HISTORY_SESSIONS
        if session > "2018-12-31"
    )
    future = copy.deepcopy(artifact_rows["AAPL"][-1])
    future["session"] = future_session
    artifact_rows["AAPL"].append(future)
    changed = _case(artifact_rows_by_symbol=artifact_rows)
    with pytest.raises(MarketSourceBytesError, match="future row"):
        _validate(changed)


class _SwitchingStage(ABCMapping[str, Any]):
    def __init__(self, first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
        self._first = first
        self._second = second
        self._reads = 0

    def __iter__(self) -> Iterator[str]:
        return iter(self._first)

    def __len__(self) -> int:
        return len(self._first)

    def __getitem__(self, key: str) -> Any:
        self._reads += 1
        source = self._first if self._reads == 1 else self._second
        return source[key]


def test_switching_stage_mapping_cannot_validate_one_view_and_receipt_another(case) -> None:
    mutated = copy.deepcopy(case["stage"])
    mutated["rows"][100]["observations"]["AAPL"]["open_hex"] = encode_float_hex(
        math.nextafter(
            float.fromhex(
                mutated["rows"][100]["observations"]["AAPL"]["open_hex"]
            ),
            math.inf,
        ),
        "mutated open",
    )
    switching = _SwitchingStage(mutated, case["stage"])
    with pytest.raises(MarketSourceBytesError, match="detached built-in"):
        validate_market_source_bytes_against_stage(
            artifact_bytes_by_symbol=case["artifacts"],
            window_bytes_by_symbol=case["windows"],
            source_manifest=case["source"],
            stage_manifest=switching,
            expected_artifact_stage="development",
            expected_source_manifest_sha256=case["source"]["source_manifest_sha256"],
            expected_market_stage_manifest_sha256=case["stage"][
                "market_stage_manifest_sha256"
            ],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.replace(b'"symbol":"AAPL"', b'"symbol":"SPY"', 1),
        lambda payload: payload + b"\n",
        lambda payload: payload.replace(b'"row_count":', b'"row_count":0,"row_count":', 1),
    ],
)
def test_noncanonical_duplicate_or_cross_symbol_bytes_fail(case, mutation) -> None:
    changed = dict(case)
    artifacts = dict(case["artifacts"])
    artifacts["AAPL"] = mutation(artifacts["AAPL"])
    changed["artifacts"] = artifacts
    with pytest.raises(MarketSourceBytesError):
        _validate(changed)
