from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import pytest

from agent_benchmark.sec_filing_gemma_contract import LABEL_MATURITY_OFFSET
from agent_benchmark.sec_filing_gemma_market_evidence import (
    ADJUSTED_OPEN_DERIVATION,
    CANONICAL_MARKET_FIELDS,
    FILL_SESSION_OFFSET,
    MARKET_CUTOFF_RULE,
    MARKET_FIELDS,
    MARKET_LOOKBACK_ROW_COUNT,
    MARKET_SYMBOLS,
    MarketEvidenceError,
    build_decision_market_prefix,
    build_market_source_manifest,
    build_market_stage_manifest,
    decode_float_hex,
    derive_adjusted_open_hex,
    encode_float_hex,
    validate_decision_market_prefix,
    validate_market_source_manifest,
    validate_market_stage_extension,
    validate_market_stage_manifest,
)
from agent_benchmark.sec_session_calendar import EXPECTED_MARKET_HISTORY_SESSIONS


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _stage_sessions(stage: str) -> tuple[str, ...]:
    ends = {
        "development": "2018-12-31",
        "intermediate": "2023-12-31",
        "final": "2026-07-09",
    }
    return tuple(
        session
        for session in EXPECTED_MARKET_HISTORY_SESSIONS
        if "1998-01-01" <= session <= ends[stage]
    )


def _available(symbol: str, index: int) -> bool:
    if symbol == "AAPL":
        return True
    # Deterministic genuine source gaps.  They remain explicit in every row
    # and in the separately pinned per-symbol source-session sequence.
    divisor = 89 + MARKET_SYMBOLS.index(symbol) * 11
    return index % divisor != 0


def _raw_observation(symbol: str, index: int) -> dict[str, Any]:
    if not _available(symbol, index):
        return {"available": False, **{name: None for name in MARKET_FIELDS}}
    base = 20.0 + MARKET_SYMBOLS.index(symbol) * 30.0 + index / 64.0
    return {
        "available": True,
        "open": base,
        "high": base + 2.0,
        "low": base - 2.0,
        "close": base + 0.25,
        "adjusted_close": base + 0.125,
        "volume": float(1_000_000 + index * 100 + MARKET_SYMBOLS.index(symbol)),
    }


def _raw_rows(stage: str) -> list[dict[str, Any]]:
    return [
        {
            "session": session,
            "observations": {
                symbol: _raw_observation(symbol, index)
                for symbol in MARKET_SYMBOLS
            },
        }
        for index, session in enumerate(_stage_sessions(stage))
    ]


def _source_artifacts(
    stage: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {
        symbol: {
            "artifact_sha256": _digest(f"{stage}:{symbol}:artifact"),
            "artifact_bytes": 50_000 + 100 * index,
            "window_sha256": _digest(f"{stage}:{symbol}:window"),
            "window_bytes": 40_000 + 100 * index,
            "available_sessions": [
                row["session"]
                for row in rows
                if row["observations"][symbol]["available"] is True
            ],
        }
        for index, symbol in enumerate(MARKET_SYMBOLS)
    }


def _external_hashes(source: Mapping[str, Any], field: str) -> dict[str, str]:
    return {item["symbol"]: item[field] for item in source["sources"]}


def _replace_stage_observation(
    stage: Mapping[str, Any],
    *,
    row_index: int,
    symbol: str,
    field: str | None = None,
    value: Any = None,
    drop_symbol: bool = False,
) -> dict[str, Any]:
    changed = dict(stage)
    rows = list(stage["rows"])
    row = dict(rows[row_index])
    observations = dict(row["observations"])
    if drop_symbol:
        del observations[symbol]
    else:
        observation = dict(observations[symbol])
        assert field is not None
        observation[field] = value
        observations[symbol] = observation
    row["observations"] = observations
    rows[row_index] = row
    changed["rows"] = rows
    return changed


def _replace_raw_observation(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_index: int,
    symbol: str,
    field: str,
    value: Any,
) -> list[Mapping[str, Any]]:
    changed = list(rows)
    row = dict(changed[row_index])
    observations = dict(row["observations"])
    observation = dict(observations[symbol])
    observation[field] = value
    observations[symbol] = observation
    row["observations"] = observations
    changed[row_index] = row
    return changed


@pytest.fixture(scope="module")
def development_evidence() -> dict[str, Any]:
    rows = _raw_rows("development")
    artifacts = _source_artifacts("development", rows)
    source = build_market_source_manifest(
        artifact_stage="development", source_artifacts=artifacts
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
        "source": source,
        "stage": stage,
    }


@pytest.fixture(scope="module")
def intermediate_evidence() -> dict[str, Any]:
    rows = _raw_rows("intermediate")
    artifacts = _source_artifacts("intermediate", rows)
    source = build_market_source_manifest(
        artifact_stage="intermediate", source_artifacts=artifacts
    )
    stage = build_market_stage_manifest(
        artifact_stage="intermediate",
        source_manifest=source,
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        rows=rows,
    )
    return {
        "rows": rows,
        "artifacts": artifacts,
        "source": source,
        "stage": stage,
    }


@pytest.fixture(scope="module")
def final_evidence() -> dict[str, Any]:
    rows = _raw_rows("final")
    artifacts = _source_artifacts("final", rows)
    source = build_market_source_manifest(
        artifact_stage="final", source_artifacts=artifacts
    )
    stage = build_market_stage_manifest(
        artifact_stage="final",
        source_manifest=source,
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        rows=rows,
    )
    return {"source": source, "stage": stage}


def _validate_source(evidence: Mapping[str, Any], stage: str = "development") -> str:
    source = evidence["source"]
    return validate_market_source_manifest(
        source,
        expected_artifact_stage=stage,
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        expected_source_artifact_sha256s=_external_hashes(
            source, "artifact_sha256"
        ),
        expected_source_window_sha256s=_external_hashes(source, "window_sha256"),
    )


def _validate_stage(evidence: Mapping[str, Any], stage: str = "development") -> str:
    return validate_market_stage_manifest(
        evidence["stage"],
        source_manifest=evidence["source"],
        expected_artifact_stage=stage,
        expected_source_manifest_sha256=evidence["source"][
            "source_manifest_sha256"
        ],
        expected_market_stage_manifest_sha256=evidence["stage"][
            "market_stage_manifest_sha256"
        ],
    )


def _walk(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk(child)


def test_complete_market_source_and_stage_manifests_validate(
    development_evidence: Mapping[str, Any],
) -> None:
    source = development_evidence["source"]
    stage = development_evidence["stage"]
    assert _validate_source(development_evidence) == source["source_manifest_sha256"]
    assert _validate_stage(development_evidence) == stage[
        "market_stage_manifest_sha256"
    ]
    assert stage["row_count"] == len(_stage_sessions("development"))
    assert [item["symbol"] for item in source["sources"]] == list(MARKET_SYMBOLS)
    assert stage["market_window_start"] == "1998-01-01"
    assert stage["market_window_end"] == "2018-12-31"


def test_numbers_use_only_exact_canonical_float_hex(
    development_evidence: Mapping[str, Any],
) -> None:
    stage = development_evidence["stage"]
    observation = stage["rows"][1]["observations"]["AAPL"]
    for field in CANONICAL_MARKET_FIELDS:
        encoded = observation[f"{field}_hex"]
        assert decode_float_hex(encoded).hex() == encoded
    assert observation["adjusted_open_hex"] == derive_adjusted_open_hex(
        open_hex=observation["open_hex"],
        close_hex=observation["close_hex"],
        adjusted_close_hex=observation["adjusted_close_hex"],
    )
    # There are no floating-point JSON numbers anywhere in the sealed output.
    assert not any(isinstance(value, float) for value in _walk(stage))


def test_adjusted_open_is_derived_with_exact_frozen_operation_order(
    development_evidence: Mapping[str, Any],
) -> None:
    observation = development_evidence["stage"]["rows"][123]["observations"][
        "AAPL"
    ]
    open_value = decode_float_hex(observation["open_hex"])
    close_value = decode_float_hex(observation["close_hex"])
    adjusted_close = decode_float_hex(observation["adjusted_close_hex"])
    assert observation["adjusted_open_hex"] == (
        open_value * adjusted_close / close_value
    ).hex()
    assert development_evidence["stage"]["adjusted_open_derivation"] == (
        ADJUSTED_OPEN_DERIVATION
    )
    assert development_evidence["stage"]["source_value_reconciliation"] == (
        "authoritative_stage_verifier_must_parse_exact_source_window_bytes"
    )


def test_caller_cannot_supply_or_override_adjusted_open(
    development_evidence: Mapping[str, Any],
) -> None:
    rows = list(development_evidence["rows"])
    row = dict(rows[0])
    observations = dict(row["observations"])
    aapl = dict(observations["AAPL"])
    aapl["adjusted_open"] = 123.0
    observations["AAPL"] = aapl
    row["observations"] = observations
    rows[0] = row
    with pytest.raises(MarketEvidenceError, match="Invalid .*AAPL keys"):
        build_market_stage_manifest(
            artifact_stage="development",
            source_manifest=development_evidence["source"],
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            rows=rows,
        )


def test_adjusted_open_mutation_is_rejected_by_exact_recomputation(
    development_evidence: Mapping[str, Any],
) -> None:
    stage = development_evidence["stage"]
    index = 50
    original = decode_float_hex(
        stage["rows"][index]["observations"]["AAPL"]["adjusted_open_hex"]
    )
    changed = _replace_stage_observation(
        stage,
        row_index=index,
        symbol="AAPL",
        field="adjusted_open_hex",
        value=math.nextafter(original, math.inf).hex(),
    )
    with pytest.raises(MarketEvidenceError, match="exact derived adjusted open"):
        validate_market_stage_manifest(
            changed,
            source_manifest=development_evidence["source"],
            expected_artifact_stage="development",
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_market_stage_manifest_sha256=stage[
                "market_stage_manifest_sha256"
            ],
        )


def test_recomputed_adjusted_open_cannot_hide_an_open_price_mutation(
    development_evidence: Mapping[str, Any],
) -> None:
    stage = development_evidence["stage"]
    index = 51
    observation = stage["rows"][index]["observations"]["AAPL"]
    changed_open = math.nextafter(decode_float_hex(observation["open_hex"]), math.inf)
    changed = _replace_stage_observation(
        stage,
        row_index=index,
        symbol="AAPL",
        field="open_hex",
        value=changed_open.hex(),
    )
    recomputed = derive_adjusted_open_hex(
        open_hex=changed_open.hex(),
        close_hex=observation["close_hex"],
        adjusted_close_hex=observation["adjusted_close_hex"],
    )
    changed = _replace_stage_observation(
        changed,
        row_index=index,
        symbol="AAPL",
        field="adjusted_open_hex",
        value=recomputed,
    )
    with pytest.raises(MarketEvidenceError, match="sub-decimal float mutation"):
        validate_market_stage_manifest(
            changed,
            source_manifest=development_evidence["source"],
            expected_artifact_stage="development",
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_market_stage_manifest_sha256=stage[
                "market_stage_manifest_sha256"
            ],
        )


@pytest.mark.parametrize("value", [True, False, math.nan, math.inf, -math.inf, 1])
def test_non_float_bool_nan_inf_and_integer_values_are_rejected(value: Any) -> None:
    with pytest.raises(MarketEvidenceError, match="float|finite"):
        encode_float_hex(value)


@pytest.mark.parametrize("value", ["0x1.0p+0", "nan", "inf", True, 1.0])
def test_noncanonical_or_nonfinite_float_hex_is_rejected(value: Any) -> None:
    with pytest.raises(MarketEvidenceError, match="float.hex|canonical finite"):
        decode_float_hex(value)


@pytest.mark.parametrize("bad_value", [True, math.nan, math.inf, -math.inf])
def test_stage_builder_rejects_bad_market_numbers(
    development_evidence: Mapping[str, Any], bad_value: Any
) -> None:
    rows = _replace_raw_observation(
        development_evidence["rows"],
        row_index=0,
        symbol="AAPL",
        field="open",
        value=bad_value,
    )
    with pytest.raises(MarketEvidenceError, match="float|finite"):
        build_market_stage_manifest(
            artifact_stage="development",
            source_manifest=development_evidence["source"],
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            rows=rows,
        )


def test_one_ulp_change_beyond_twelve_significant_digits_breaks_row_chain(
    development_evidence: Mapping[str, Any],
) -> None:
    stage = development_evidence["stage"]
    index = 12
    original_hex = stage["rows"][index]["observations"]["AAPL"]["high_hex"]
    original = decode_float_hex(original_hex)
    changed_value = math.nextafter(original, math.inf)
    assert changed_value != original
    assert f"{changed_value:.12g}" == f"{original:.12g}"
    changed = _replace_stage_observation(
        stage,
        row_index=index,
        symbol="AAPL",
        field="high_hex",
        value=changed_value.hex(),
    )
    with pytest.raises(MarketEvidenceError, match="sub-decimal float mutation"):
        validate_market_stage_manifest(
            changed,
            source_manifest=development_evidence["source"],
            expected_artifact_stage="development",
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_market_stage_manifest_sha256=stage[
                "market_stage_manifest_sha256"
            ],
        )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_missing_duplicate_or_reordered_aapl_session_is_rejected(
    development_evidence: Mapping[str, Any], mutation: str
) -> None:
    rows = list(development_evidence["rows"])
    if mutation == "missing":
        del rows[5]
    elif mutation == "duplicate":
        rows.insert(5, rows[5])
    else:
        rows[5], rows[6] = rows[6], rows[5]
    with pytest.raises(MarketEvidenceError, match="missing|duplicated|reordered|extra"):
        build_market_stage_manifest(
            artifact_stage="development",
            source_manifest=development_evidence["source"],
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            rows=rows,
        )


def test_aapl_unavailability_is_never_permitted(
    development_evidence: Mapping[str, Any],
) -> None:
    rows = list(development_evidence["rows"])
    row = dict(rows[4])
    observations = dict(row["observations"])
    observations["AAPL"] = {
        "available": False,
        **{field: None for field in MARKET_FIELDS},
    }
    row["observations"] = observations
    rows[4] = row
    with pytest.raises(MarketEvidenceError, match="AAPL must be available"):
        build_market_stage_manifest(
            artifact_stage="development",
            source_manifest=development_evidence["source"],
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            rows=rows,
        )


def test_missing_context_is_explicit_false_plus_all_nulls(
    development_evidence: Mapping[str, Any],
) -> None:
    stage = development_evidence["stage"]
    missing = stage["rows"][0]["observations"]["SPY"]
    assert missing["available"] is False
    assert all(
        missing[f"{field}_hex"] is None for field in CANONICAL_MARKET_FIELDS
    )
    assert _validate_stage(development_evidence) == stage[
        "market_stage_manifest_sha256"
    ]


def test_unavailable_context_cannot_retain_hidden_numeric_values(
    development_evidence: Mapping[str, Any],
) -> None:
    rows = _replace_raw_observation(
        development_evidence["rows"],
        row_index=0,
        symbol="SPY",
        field="close",
        value=100.0,
    )
    with pytest.raises(MarketEvidenceError, match="explicit nulls"):
        build_market_stage_manifest(
            artifact_stage="development",
            source_manifest=development_evidence["source"],
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            rows=rows,
        )


def test_context_symbol_cannot_be_silently_dropped_from_a_session(
    development_evidence: Mapping[str, Any],
) -> None:
    changed = _replace_stage_observation(
        development_evidence["stage"],
        row_index=1,
        symbol="QQQ",
        drop_symbol=True,
    )
    with pytest.raises(MarketEvidenceError, match="all context symbols"):
        validate_market_stage_manifest(
            changed,
            source_manifest=development_evidence["source"],
            expected_artifact_stage="development",
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_market_stage_manifest_sha256=development_evidence["stage"][
                "market_stage_manifest_sha256"
            ],
        )


def test_available_context_row_cannot_be_silently_converted_to_missing(
    development_evidence: Mapping[str, Any],
) -> None:
    rows = list(development_evidence["rows"])
    index = next(
        i for i, row in enumerate(rows) if row["observations"]["IWM"]["available"]
    )
    row = dict(rows[index])
    observations = dict(row["observations"])
    observations["IWM"] = {
        "available": False,
        **{field: None for field in MARKET_FIELDS},
    }
    row["observations"] = observations
    rows[index] = row
    with pytest.raises(MarketEvidenceError, match="silently dropped"):
        build_market_stage_manifest(
            artifact_stage="development",
            source_manifest=development_evidence["source"],
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            rows=rows,
        )


def test_source_artifact_and_window_hashes_are_independently_pinned(
    development_evidence: Mapping[str, Any],
) -> None:
    source = development_evidence["source"]
    artifacts = _external_hashes(source, "artifact_sha256")
    windows = _external_hashes(source, "window_sha256")
    artifacts["SPY"] = _digest("wrong-spy-artifact")
    with pytest.raises(MarketEvidenceError, match="artifact hashes"):
        validate_market_source_manifest(
            source,
            expected_artifact_stage="development",
            expected_source_manifest_sha256=source["source_manifest_sha256"],
            expected_source_artifact_sha256s=artifacts,
            expected_source_window_sha256s=windows,
        )
    artifacts = _external_hashes(source, "artifact_sha256")
    windows["VIX"] = _digest("wrong-vix-window")
    with pytest.raises(MarketEvidenceError, match="window hashes"):
        validate_market_source_manifest(
            source,
            expected_artifact_stage="development",
            expected_source_manifest_sha256=source["source_manifest_sha256"],
            expected_source_artifact_sha256s=artifacts,
            expected_source_window_sha256s=windows,
        )


def test_source_stage_and_window_substitution_are_rejected(
    development_evidence: Mapping[str, Any],
) -> None:
    source = development_evidence["source"]
    with pytest.raises(MarketEvidenceError, match="stage"):
        validate_market_source_manifest(
            source,
            expected_artifact_stage="intermediate",
            expected_source_manifest_sha256=source["source_manifest_sha256"],
            expected_source_artifact_sha256s=_external_hashes(
                source, "artifact_sha256"
            ),
            expected_source_window_sha256s=_external_hashes(source, "window_sha256"),
        )
    changed = dict(source)
    changed["window_end"] = "2018-12-28"
    with pytest.raises(MarketEvidenceError, match="window"):
        validate_market_source_manifest(
            changed,
            expected_artifact_stage="development",
            expected_source_manifest_sha256=source["source_manifest_sha256"],
            expected_source_artifact_sha256s=_external_hashes(
                source, "artifact_sha256"
            ),
            expected_source_window_sha256s=_external_hashes(source, "window_sha256"),
        )


def test_non_authoritative_market_calendar_is_rejected(
    development_evidence: Mapping[str, Any],
) -> None:
    shortened = list(EXPECTED_MARKET_HISTORY_SESSIONS)
    del shortened[100]
    with pytest.raises(MarketEvidenceError, match="authoritative"):
        build_market_source_manifest(
            artifact_stage="development",
            source_artifacts=development_evidence["artifacts"],
            session_dates=shortened,
        )


@pytest.fixture(scope="module")
def development_prefix(development_evidence: Mapping[str, Any]) -> dict[str, Any]:
    decision_session = "2018-10-01"
    stage = development_evidence["stage"]
    source = development_evidence["source"]
    return build_decision_market_prefix(
        stage_manifest=stage,
        source_manifest=source,
        expected_artifact_stage="development",
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        expected_market_stage_manifest_sha256=stage[
            "market_stage_manifest_sha256"
        ],
        decision_event_id="0000320193-18-000145",
        decision_session=decision_session,
    )


def _validate_prefix(
    prefix: Mapping[str, Any], development_evidence: Mapping[str, Any]
) -> str:
    return validate_decision_market_prefix(
        prefix,
        stage_manifest=development_evidence["stage"],
        source_manifest=development_evidence["source"],
        expected_artifact_stage="development",
        expected_source_manifest_sha256=development_evidence["source"][
            "source_manifest_sha256"
        ],
        expected_market_stage_manifest_sha256=development_evidence["stage"][
            "market_stage_manifest_sha256"
        ],
        expected_decision_event_id="0000320193-18-000145",
        expected_decision_session="2018-10-01",
        expected_market_prefix_sha256=prefix["market_prefix_sha256"],
    )


def test_decision_prefix_is_inclusive_only_through_completed_t_close_and_binds_offsets(
    development_evidence: Mapping[str, Any], development_prefix: Mapping[str, Any]
) -> None:
    prefix = development_prefix
    calendar = list(EXPECTED_MARKET_HISTORY_SESSIONS)
    decision_index = calendar.index("2018-10-01")
    assert prefix["market_cutoff_rule"] == MARKET_CUTOFF_RULE
    assert prefix["full_prefix_last_session"] == "2018-10-01"
    assert prefix["full_prefix_row_count"] == decision_index + 1
    assert prefix["lookback_last_session"] == "2018-10-01"
    assert prefix["lookback_rows"][-1]["session"] == "2018-10-01"
    assert prefix["lookback_row_count"] == MARKET_LOOKBACK_ROW_COUNT == 253
    assert prefix["lookback_first_session"] == calendar[decision_index - 252]
    assert prefix["lookback_start_row_index"] == decision_index - 252
    assert prefix["lookback_end_row_index"] == decision_index
    assert prefix["lookback_row_chain_tip_sha256"] == prefix[
        "full_prefix_row_chain_tip_sha256"
    ]
    assert prefix["fill_session_offset"] == FILL_SESSION_OFFSET == 1
    assert prefix["fill_session"] == calendar[decision_index + 1]
    assert prefix["label_maturity_session_offset"] == LABEL_MATURITY_OFFSET == 21
    assert prefix["label_maturity_session"] == calendar[
        decision_index + LABEL_MATURITY_OFFSET
    ]
    assert _validate_prefix(prefix, development_evidence) == prefix[
        "market_prefix_sha256"
    ]


def test_decision_prefix_has_a_strict_compact_size_bound(
    development_evidence: Mapping[str, Any], development_prefix: Mapping[str, Any]
) -> None:
    prefix_bytes = len(
        json.dumps(
            development_prefix,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    stage_bytes = len(
        json.dumps(
            development_evidence["stage"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    assert development_prefix["lookback_row_count"] == 253
    assert prefix_bytes < 1_000_000
    assert prefix_bytes < stage_bytes // 10


def test_earliest_development_decision_still_has_exact_frozen_lookback(
    development_evidence: Mapping[str, Any],
) -> None:
    stage = development_evidence["stage"]
    source = development_evidence["source"]
    prefix = build_decision_market_prefix(
        stage_manifest=stage,
        source_manifest=source,
        expected_artifact_stage="development",
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        expected_market_stage_manifest_sha256=stage[
            "market_stage_manifest_sha256"
        ],
        decision_event_id="earliest-development-event",
        decision_session="2000-01-03",
    )
    calendar = list(EXPECTED_MARKET_HISTORY_SESSIONS)
    decision_index = calendar.index("2000-01-03")
    assert prefix["full_prefix_row_count"] == decision_index + 1
    assert prefix["lookback_row_count"] == 253
    assert prefix["lookback_first_session"] == calendar[decision_index - 252]
    assert prefix["lookback_last_session"] == "2000-01-03"
    assert validate_decision_market_prefix(
        prefix,
        stage_manifest=stage,
        source_manifest=source,
        expected_artifact_stage="development",
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        expected_market_stage_manifest_sha256=stage[
            "market_stage_manifest_sha256"
        ],
        expected_decision_event_id="earliest-development-event",
        expected_decision_session="2000-01-03",
        expected_market_prefix_sha256=prefix["market_prefix_sha256"],
    ) == prefix["market_prefix_sha256"]


def test_later_market_row_is_rejected_even_if_attached_to_a_prefix(
    development_evidence: Mapping[str, Any], development_prefix: Mapping[str, Any]
) -> None:
    changed = dict(development_prefix)
    rows = list(changed["lookback_rows"])
    next_row = development_evidence["stage"]["rows"][
        development_prefix["full_prefix_row_count"]
    ]
    assert next_row["session"] > changed["decision_session"]
    rows.append(next_row)
    changed["lookback_rows"] = rows
    with pytest.raises(MarketEvidenceError, match="later row"):
        _validate_prefix(changed, development_evidence)


def test_decision_prefix_does_not_alias_or_mutate_the_full_stage(
    development_evidence: Mapping[str, Any], development_prefix: Mapping[str, Any]
) -> None:
    row_index = development_prefix["lookback_rows"][1]["row_index"]
    stage_value = development_evidence["stage"]["rows"][row_index]["observations"]["AAPL"][
        "close_hex"
    ]
    prefix_value = development_prefix["lookback_rows"][1]["observations"]["AAPL"][
        "close_hex"
    ]
    assert prefix_value == stage_value
    try:
        development_prefix["lookback_rows"][1]["observations"]["AAPL"]["close_hex"] = (
            math.nextafter(decode_float_hex(prefix_value), math.inf).hex()
        )
        assert development_evidence["stage"]["rows"][row_index]["observations"]["AAPL"][
            "close_hex"
        ] == stage_value
    finally:
        # Restore the module-scoped fixture so subsequent mutation tests start
        # from the externally pinned canonical prefix.
        development_prefix["lookback_rows"][1]["observations"]["AAPL"][
            "close_hex"
        ] = prefix_value


@pytest.mark.parametrize("mutation", ["missing", "reordered"])
def test_decision_prefix_cannot_omit_or_reorder_earlier_rows(
    development_evidence: Mapping[str, Any],
    development_prefix: Mapping[str, Any],
    mutation: str,
) -> None:
    changed = dict(development_prefix)
    rows = list(changed["lookback_rows"])
    if mutation == "missing":
        del rows[10]
    else:
        rows[10], rows[11] = rows[11], rows[10]
    changed["lookback_rows"] = rows
    with pytest.raises(MarketEvidenceError, match="incomplete|reordered"):
        _validate_prefix(changed, development_evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_prefix_row_count", 1),
        ("full_prefix_row_chain_tip_sha256", _digest("forged-full-tip")),
        ("full_prefix_chain_identity_sha256", _digest("forged-full-identity")),
    ],
)
def test_full_prefix_count_tip_and_identity_are_recomputed(
    development_evidence: Mapping[str, Any],
    development_prefix: Mapping[str, Any],
    field: str,
    value: Any,
) -> None:
    changed = dict(development_prefix)
    changed[field] = value
    with pytest.raises(MarketEvidenceError, match="incomplete|mutated"):
        _validate_prefix(changed, development_evidence)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fill_session", "2018-10-05"),
        ("label_maturity_session", "2018-11-15"),
        ("market_cutoff_session", "2018-10-02"),
    ],
)
def test_fill_maturity_and_cutoff_binding_cannot_be_changed(
    development_evidence: Mapping[str, Any],
    development_prefix: Mapping[str, Any],
    field: str,
    value: str,
) -> None:
    changed = dict(development_prefix)
    changed[field] = value
    with pytest.raises(MarketEvidenceError, match="incomplete|future rows"):
        _validate_prefix(changed, development_evidence)


def test_prefix_event_and_external_pin_cannot_be_substituted(
    development_evidence: Mapping[str, Any], development_prefix: Mapping[str, Any]
) -> None:
    with pytest.raises(MarketEvidenceError, match="event identity"):
        validate_decision_market_prefix(
            development_prefix,
            stage_manifest=development_evidence["stage"],
            source_manifest=development_evidence["source"],
            expected_artifact_stage="development",
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_market_stage_manifest_sha256=development_evidence["stage"][
                "market_stage_manifest_sha256"
            ],
            expected_decision_event_id="different-event",
            expected_decision_session="2018-10-01",
            expected_market_prefix_sha256=development_prefix[
                "market_prefix_sha256"
            ],
        )
    with pytest.raises(MarketEvidenceError, match="externally pinned"):
        validate_decision_market_prefix(
            development_prefix,
            stage_manifest=development_evidence["stage"],
            source_manifest=development_evidence["source"],
            expected_artifact_stage="development",
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_market_stage_manifest_sha256=development_evidence["stage"][
                "market_stage_manifest_sha256"
            ],
            expected_decision_event_id="0000320193-18-000145",
            expected_decision_session="2018-10-01",
            expected_market_prefix_sha256=_digest("wrong-prefix-pin"),
        )


def test_late_decision_without_exact_maturity_session_fails_closed(
    final_evidence: Mapping[str, Any],
) -> None:
    # 2026-07-10 is the exact t+1 fill, but the frozen calendar has no t+21.
    # The evidence layer must not invent a calendar date for an unmatured label.
    stage = final_evidence["stage"]
    source = final_evidence["source"]
    with pytest.raises(MarketEvidenceError, match=r"exact \+21"):
        build_decision_market_prefix(
            stage_manifest=stage,
            source_manifest=source,
            expected_artifact_stage="final",
            expected_source_manifest_sha256=source["source_manifest_sha256"],
            expected_market_stage_manifest_sha256=stage[
                "market_stage_manifest_sha256"
            ],
            decision_event_id="unmatured-final-event",
            decision_session="2026-07-09",
        )


def test_later_stage_is_an_exact_append_and_preserves_every_earlier_hash(
    development_evidence: Mapping[str, Any],
    intermediate_evidence: Mapping[str, Any],
    development_prefix: Mapping[str, Any],
) -> None:
    prior = development_evidence["stage"]
    extended = intermediate_evidence["stage"]
    assert len(extended["rows"]) > len(prior["rows"])
    assert extended["rows"][: len(prior["rows"])] == prior["rows"]
    assert [row["row_sha256"] for row in extended["rows"][: len(prior["rows"])]] == [
        row["row_sha256"] for row in prior["rows"]
    ]
    prefix_count = development_prefix["full_prefix_row_count"]
    assert extended["rows"][prefix_count - 1]["row_sha256"] == development_prefix[
        "full_prefix_row_chain_tip_sha256"
    ]
    assert (
        validate_market_stage_extension(
            prior_stage_manifest=prior,
            prior_source_manifest=development_evidence["source"],
            expected_prior_stage="development",
            expected_prior_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_prior_market_stage_manifest_sha256=prior[
                "market_stage_manifest_sha256"
            ],
            extended_stage_manifest=extended,
            extended_source_manifest=intermediate_evidence["source"],
            expected_extended_stage="intermediate",
            expected_extended_source_manifest_sha256=intermediate_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_extended_market_stage_manifest_sha256=extended[
                "market_stage_manifest_sha256"
            ],
        )
        == prior["row_chain_tip_sha256"]
    )


def test_later_stage_cannot_rewrite_an_earlier_low_order_float(
    development_evidence: Mapping[str, Any],
    intermediate_evidence: Mapping[str, Any],
) -> None:
    extended = intermediate_evidence["stage"]
    index = 100
    original_hex = extended["rows"][index]["observations"]["AAPL"]["high_hex"]
    changed = _replace_stage_observation(
        extended,
        row_index=index,
        symbol="AAPL",
        field="high_hex",
        value=math.nextafter(decode_float_hex(original_hex), math.inf).hex(),
    )
    with pytest.raises(MarketEvidenceError, match="sub-decimal float mutation"):
        validate_market_stage_extension(
            prior_stage_manifest=development_evidence["stage"],
            prior_source_manifest=development_evidence["source"],
            expected_prior_stage="development",
            expected_prior_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_prior_market_stage_manifest_sha256=development_evidence["stage"][
                "market_stage_manifest_sha256"
            ],
            extended_stage_manifest=changed,
            extended_source_manifest=intermediate_evidence["source"],
            expected_extended_stage="intermediate",
            expected_extended_source_manifest_sha256=intermediate_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_extended_market_stage_manifest_sha256=extended[
                "market_stage_manifest_sha256"
            ],
        )


def test_stage_manifest_pin_rejects_a_complete_alternate_source_snapshot(
    development_evidence: Mapping[str, Any],
) -> None:
    changed_artifacts = {
        symbol: dict(spec)
        for symbol, spec in development_evidence["artifacts"].items()
    }
    changed_artifacts["TNX"]["window_sha256"] = _digest("new-tnx-window")
    alternate = build_market_source_manifest(
        artifact_stage="development", source_artifacts=changed_artifacts
    )
    with pytest.raises(MarketEvidenceError, match="externally pinned"):
        validate_market_source_manifest(
            alternate,
            expected_artifact_stage="development",
            expected_source_manifest_sha256=development_evidence["source"][
                "source_manifest_sha256"
            ],
            expected_source_artifact_sha256s=_external_hashes(
                alternate, "artifact_sha256"
            ),
            expected_source_window_sha256s=_external_hashes(
                alternate, "window_sha256"
            ),
        )
