from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.direct_edge_features import (
    MARKET_SENTIMENT_FEATURE_COLUMNS,
    build_market_sentiment_features,
)
from agent_benchmark.downside_features import (
    PRICE_FEATURE_COLUMNS,
    build_downside_price_features,
)
from agent_benchmark.sec_filing_gemma_contract import (
    ACTIVE_EDGE_TOLERANCE,
    DIMENSION_NAMES,
    FLAG_NAMES,
    build_corpus_universe_manifest,
    build_stage_content_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_features import (
    ABLATION_FEATURE_COLUMNS,
    FILING_CALENDAR_FEATURE_COLUMNS,
    LABEL_MATURITY_OFFSET,
    SEMANTIC_AGGREGATE_FEATURE_COLUMNS,
    SEMANTIC_FEATURE_COLUMNS,
    SecFilingGemmaFeatureError,
    build_sec_filing_gemma_feature_row,
    build_twenty_session_label_evidence,
    build_validated_extraction_event_proof,
    build_validated_market_prefix_proof,
    build_validated_universe_event_proof,
    validate_extraction_event_proof,
    validate_sec_filing_gemma_feature_row,
    validate_twenty_session_label_evidence,
    validate_universe_event_proof,
)
from agent_benchmark.sec_filing_gemma_market_evidence import (
    ADJUSTED_OPEN_DERIVATION,
    CANONICAL_MARKET_FIELDS,
    MARKET_ROW_SCHEMA_VERSION,
    MARKET_STAGE_MANIFEST_SCHEMA_VERSION,
    MARKET_SYMBOLS,
    build_decision_market_prefix,
    build_market_source_manifest,
    derive_adjusted_open_hex,
    market_session_calendar_sha256,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
    MARKET_HISTORY_CALENDAR_ID,
)


ACCESSION = "0000320193-02-000001"
PRIOR_ACCESSION = "0000320193-01-000001"
DECISION_SESSION = "2002-02-01"
ROW_CHAIN_GENESIS = hashlib.sha256(
    b"aapl-sec-gemma-market-row-chain-v1\x00"
).hexdigest()
DEVELOPMENT_LAST_INDEX = EXPECTED_MARKET_HISTORY_SESSIONS.index("2018-12-31")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _observation(
    symbol: str,
    index: int,
    *,
    available: bool = True,
    adjusted_open_override: float | None = None,
    random_path: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if not available:
        return {
            "available": False,
            **{f"{name}_hex": None for name in CANONICAL_MARKET_FIELDS},
        }
    number = MARKET_SYMBOLS.index(symbol)
    if random_path is None:
        base = (
            25.0
            + 17.0 * number
            + 0.031 * index
            + 1.7 * math.sin(index / (13.0 + number))
            + 0.8 * math.cos(index / (29.0 + number))
        )
    else:
        base = float(random_path[symbol][index])
    open_value = base * (1.0 + 0.003 * math.sin(index / 7.0))
    close = base * (1.0 + 0.004 * math.cos(index / 11.0))
    adjusted_close = close * (0.91 + 0.00001 * index)
    if adjusted_open_override is not None:
        open_value = close = adjusted_close = adjusted_open_override
    high = max(open_value, close) * 1.01
    low = min(open_value, close) * 0.99
    result = {
        "available": True,
        "open_hex": float(open_value).hex(),
        "high_hex": float(high).hex(),
        "low_hex": float(low).hex(),
        "close_hex": float(close).hex(),
        "adjusted_close_hex": float(adjusted_close).hex(),
        "volume_hex": float(1_000_000 + index * 31 + number).hex(),
    }
    result["adjusted_open_hex"] = derive_adjusted_open_hex(
        open_hex=result["open_hex"],
        close_hex=result["close_hex"],
        adjusted_close_hex=result["adjusted_close_hex"],
    )
    return result


def _canonical_rows(
    *,
    missing: set[tuple[int, str]] | None = None,
    adjusted_open_overrides: Mapping[int, float] | None = None,
    random_seed: int | None = None,
) -> list[dict[str, Any]]:
    missing = missing or set()
    overrides = adjusted_open_overrides or {}
    random_path: dict[str, np.ndarray] | None = None
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        random_path = {}
        for number, symbol in enumerate(MARKET_SYMBOLS):
            innovations = rng.normal(
                0.0002 + number * 0.00001,
                0.007 + number * 0.0002,
                DEVELOPMENT_LAST_INDEX + 1,
            )
            random_path[symbol] = (30.0 + number * 15.0) * np.exp(
                np.cumsum(innovations)
            )
    rows: list[dict[str, Any]] = []
    parent = ROW_CHAIN_GENESIS
    for index, session in enumerate(
        EXPECTED_MARKET_HISTORY_SESSIONS[: DEVELOPMENT_LAST_INDEX + 1]
    ):
        observations = {
            symbol: _observation(
                symbol,
                index,
                available=(index, symbol) not in missing,
                adjusted_open_override=(
                    overrides.get(index) if symbol == "AAPL" else None
                ),
                random_path=random_path,
            )
            for symbol in MARKET_SYMBOLS
        }
        body = {
            "schema_version": MARKET_ROW_SCHEMA_VERSION,
            "row_index": index,
            "session": session,
            "observations": observations,
            "previous_row_sha256": parent,
        }
        row_hash = canonical_sha256(body)
        rows.append({**body, "row_sha256": row_hash})
        parent = row_hash
    return rows


def _market_evidence(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    sessions = [row["session"] for row in rows]
    source_artifacts = {
        symbol: {
            "artifact_sha256": _digest(f"{label}:{symbol}:artifact"),
            "artifact_bytes": 100_000 + number,
            "window_sha256": _digest(f"{label}:{symbol}:window"),
            "window_bytes": 90_000 + number,
            "available_sessions": [
                row["session"]
                for row in rows
                if row["observations"][symbol]["available"]
            ],
        }
        for number, symbol in enumerate(MARKET_SYMBOLS)
    }
    source = build_market_source_manifest(
        artifact_stage="development", source_artifacts=source_artifacts
    )
    frames = {
        symbol: [
            {"session": row["session"], "observation": row["observations"][symbol]}
            for row in rows
        ]
        for symbol in MARKET_SYMBOLS
    }
    row_hashes = [row["row_sha256"] for row in rows]
    stage_body = {
        "schema_version": MARKET_STAGE_MANIFEST_SCHEMA_VERSION,
        "artifact_stage": "development",
        "calendar_id": MARKET_HISTORY_CALENDAR_ID,
        "calendar_sha256": market_session_calendar_sha256(
            EXPECTED_MARKET_HISTORY_SESSIONS
        ),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "market_window_start": "1998-01-01",
        "market_window_end": "2018-12-31",
        "score_window_start": "2000-01-01",
        "score_window_end": "2018-12-31",
        "row_schema_version": MARKET_ROW_SCHEMA_VERSION,
        "numeric_encoding": "python_float_hex_exact_finite",
        "adjusted_open_derivation": ADJUSTED_OPEN_DERIVATION,
        "missing_context_encoding": "available_false_and_all_value_hex_null",
        "source_value_reconciliation": (
            "authoritative_stage_verifier_must_parse_exact_source_window_bytes"
        ),
        "row_chain_genesis_sha256": ROW_CHAIN_GENESIS,
        "row_count": len(rows),
        "first_session": sessions[0],
        "last_session": sessions[-1],
        "row_chain_tip_sha256": row_hashes[-1],
        "row_hashes_sha256": canonical_sha256(row_hashes),
        "symbol_frame_sha256s": {
            symbol: canonical_sha256(frames[symbol]) for symbol in MARKET_SYMBOLS
        },
        "rows": list(rows),
    }
    stage = {
        **stage_body,
        "market_stage_manifest_sha256": canonical_sha256(stage_body),
    }
    prefix = build_decision_market_prefix(
        stage_manifest=stage,
        source_manifest=source,
        expected_artifact_stage="development",
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        expected_market_stage_manifest_sha256=stage[
            "market_stage_manifest_sha256"
        ],
        decision_event_id=ACCESSION,
        decision_session=DECISION_SESSION,
    )
    proof = build_validated_market_prefix_proof(
        prefix=prefix,
        stage_manifest=stage,
        source_manifest=source,
        expected_artifact_stage="development",
        expected_source_manifest_sha256=source["source_manifest_sha256"],
        expected_market_stage_manifest_sha256=stage[
            "market_stage_manifest_sha256"
        ],
        expected_decision_event_id=ACCESSION,
        expected_decision_session=DECISION_SESSION,
        expected_market_prefix_sha256=prefix["market_prefix_sha256"],
    )
    return {"rows": rows, "source": source, "stage": stage, "prefix": prefix, "proof": proof}


def _source_record(accession: str, availability: str, form: str, label: str) -> dict[str, Any]:
    position = EXPECTED_SESSIONS.index(availability)
    filing = EXPECTED_SESSIONS[position - 1]
    compact_date = filing.replace("-", "")
    return {
        "accession_number": accession,
        "subject_cik": "0000320193",
        "form": form,
        "acceptance_datetime": f"{compact_date}163000",
        "filing_date": filing,
        "filing_date_change": None,
        "primary_document": f"{label}.htm",
        "source_record_sha256": _digest(f"{label}:source"),
    }


def _universe_evidence() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counter = 1
    for year in range(2000, 2027):
        month_days = ((1, 15), (4, 15), (7, 15), (10, 15))
        if year == 2026:
            month_days = month_days[:2]
        for event_index, (month, day) in enumerate(month_days):
            target = f"{year:04d}-{month:02d}-{day:02d}"
            availability = next(
                session for session in EXPECTED_SESSIONS if session > target
            )
            accession = f"0000320193-{year % 100:02d}-{counter:06d}"
            label = f"event-{counter:03d}"
            if year == 2001 and event_index == 0:
                accession = PRIOR_ACCESSION
                label = "prior"
            if year == 2002 and event_index == 0:
                accession = ACCESSION
                availability = DECISION_SESSION
                label = "current"
            records.append(
                _source_record(
                    accession,
                    availability,
                    "10-K" if event_index == 0 else "10-Q",
                    label,
                )
            )
            counter += 1
    universe = build_corpus_universe_manifest(
        catalog_artifact_sha256=_digest("catalog"),
        calendar_artifact_sha256=_digest("calendar evidence"),
        catalog_total_record_count=len(records),
        catalog_eligible_record_count=len(records),
        session_dates=EXPECTED_SESSIONS,
        records=records,
    )
    documents = [
        {
            "accession_number": record["accession_number"],
            "primary_document_sha256": _digest(
                f"{record['accession_number']}:primary"
            ),
            "normalized_text_sha256": _digest(
                f"{record['accession_number']}:normalized"
            ),
            "primary_document_bytes": 50_000,
            "normalized_text_bytes": 40_000,
        }
        for record in universe["records"]
        if record["artifact_stage"] == "development"
    ]
    content = build_stage_content_manifest(
        artifact_stage="development",
        corpus_universe_sha256=universe["universe_sha256"],
        documents=documents,
        universe_manifest=universe,
    )
    proof = build_validated_universe_event_proof(
        universe_manifest=universe,
        expected_corpus_universe_sha256=universe["universe_sha256"],
        current_accession_number=ACCESSION,
        content_manifests_by_stage={"development": content},
        expected_content_manifest_sha256s={
            "development": content["content_manifest_sha256"]
        },
    )
    return {"universe": universe, "content": content, "proof": proof}


def _extractor_output(quality: str = "usable") -> dict[str, Any]:
    impacts = [
        "favorable", "unfavorable", "neutral", "mixed", "not_stated",
        "favorable", "unfavorable", "neutral", "mixed", "favorable",
    ]
    changes = [
        "improving", "deteriorating", "stable", "mixed", "not_comparable",
        "not_stated", "improving", "deteriorating", "stable", "mixed",
    ]
    dimensions: dict[str, Any] = {}
    for index, name in enumerate(DIMENSION_NAMES):
        current = impacts[index]
        change = changes[index]
        evidence: list[str] = []
        if current != "not_stated":
            evidence.append("C0001")
        if change not in {"not_stated", "not_comparable"}:
            if "C0001" not in evidence:
                evidence.append("C0001")
            evidence.append("P0001")
        dimensions[name] = {
            "current_impact": current,
            "change_vs_prior": change,
            "evidence_sentence_ids": evidence,
        }
    flags = {
        name: {
            "present": name in {"new_material_risk", "management_transition"},
            "evidence_sentence_ids": (
                ["C0001"] if name in {"new_material_risk", "management_transition"} else []
            ),
        }
        for name in FLAG_NAMES
    }
    if quality == "unusable":
        dimensions = {
            name: {
                "current_impact": "not_stated",
                "change_vs_prior": "not_stated",
                "evidence_sentence_ids": [],
            }
            for name in DIMENSION_NAMES
        }
        flags = {
            name: {"present": False, "evidence_sentence_ids": []}
            for name in FLAG_NAMES
        }
    return {
        "schema_version": "sec-filing-extractor-v1",
        "document_quality": quality,
        "dimensions": dimensions,
        "flags": flags,
    }


def _extraction_proof(
    universe_proof: Mapping[str, Any],
    *,
    status: str = "valid",
    quality: str = "usable",
    authenticated: bool = True,
    evidence_label: str | None = None,
    supplied_sentence_ids: Sequence[str] = ("C0001", "P0001"),
) -> dict[str, Any]:
    output = _extractor_output(quality) if status == "valid" else None
    output_bytes = (
        None
        if output is None
        else json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    )
    evidence = _digest(evidence_label or f"extraction:{status}:{quality}")
    expected_evidence = evidence if authenticated else _digest("wrong evidence")
    return build_validated_extraction_event_proof(
        universe_event_proof=universe_proof,
        expected_universe_event_proof_sha256=universe_proof[
            "universe_event_proof_sha256"
        ],
        extraction_status=status,
        extraction_evidence_sha256=evidence,
        expected_extraction_evidence_sha256=expected_evidence,
        extractor_output=output,
        extractor_output_bytes=output_bytes,
        expected_extraction_output_sha256=(
            None if output_bytes is None else hashlib.sha256(output_bytes).hexdigest()
        ),
        expected_extraction_output_canonical_sha256=(
            None if output is None else canonical_sha256(output)
        ),
        supplied_sentence_ids=supplied_sentence_ids,
    )


def _rehash_universe_event_proof(proof: dict[str, Any]) -> dict[str, Any]:
    body = {
        key: proof[key]
        for key in proof
        if key != "universe_event_proof_sha256"
    }
    proof["universe_event_proof_sha256"] = canonical_sha256(body)
    return proof


def _feature_kwargs(
    market: Mapping[str, Any],
    universe_proof: Mapping[str, Any],
    extraction_proof: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "market_prefix": market["prefix"],
        "market_prefix_proof": market["proof"],
        "expected_market_prefix_proof_sha256": market["proof"][
            "market_prefix_proof_sha256"
        ],
        "universe_event_proof": universe_proof,
        "expected_universe_event_proof_sha256": universe_proof[
            "universe_event_proof_sha256"
        ],
        "extraction_event_proof": extraction_proof,
        "expected_extraction_event_proof_sha256": extraction_proof[
            "extraction_event_proof_sha256"
        ],
    }


def _feature_validate_kwargs(case: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = dict(case["feature_kwargs"])
    return {
        "market_prefix": kwargs["market_prefix"],
        "market_prefix_proof": kwargs["market_prefix_proof"],
        "expected_market_prefix_proof_sha256": kwargs[
            "expected_market_prefix_proof_sha256"
        ],
        "universe_event_proof": kwargs["universe_event_proof"],
        "expected_universe_event_proof_sha256": kwargs[
            "expected_universe_event_proof_sha256"
        ],
        "extraction_event_proof": kwargs["extraction_event_proof"],
        "expected_extraction_event_proof_sha256": kwargs[
            "expected_extraction_event_proof_sha256"
        ],
    }


def _label_kwargs(case: Mapping[str, Any]) -> dict[str, Any]:
    feature_kwargs = case["feature_kwargs"]
    market = case["market"]
    return {
        "market_prefix": market["prefix"],
        "market_prefix_proof": market["proof"],
        "expected_market_prefix_proof_sha256": market["proof"][
            "market_prefix_proof_sha256"
        ],
        "stage_manifest": market["stage"],
        "source_manifest": market["source"],
        "expected_artifact_stage": "development",
        "expected_source_manifest_sha256": market["source"][
            "source_manifest_sha256"
        ],
        "expected_market_stage_manifest_sha256": market["stage"][
            "market_stage_manifest_sha256"
        ],
        "feature_row": case["feature"],
        "expected_feature_row_sha256": case["feature"]["feature_row_sha256"],
        "universe_event_proof": feature_kwargs["universe_event_proof"],
        "expected_universe_event_proof_sha256": feature_kwargs[
            "expected_universe_event_proof_sha256"
        ],
        "extraction_event_proof": feature_kwargs["extraction_event_proof"],
        "expected_extraction_event_proof_sha256": feature_kwargs[
            "expected_extraction_event_proof_sha256"
        ],
    }


@pytest.fixture(scope="module")
def case() -> dict[str, Any]:
    decision_index = EXPECTED_MARKET_HISTORY_SESSIONS.index(DECISION_SESSION)
    cost_10 = math.log1p(-0.001) - math.log1p(0.001)
    entry = 100.0
    exit_value = entry * math.exp(cost_10 - 0.5e-12)
    rows = _canonical_rows(
        adjusted_open_overrides={
            decision_index + 1: entry,
            decision_index + 21: exit_value,
        }
    )
    market = _market_evidence(rows, "base")
    universe = _universe_evidence()
    extraction = _extraction_proof(universe["proof"])
    feature_kwargs = _feature_kwargs(market, universe["proof"], extraction)
    feature = build_sec_filing_gemma_feature_row(**feature_kwargs)
    return {
        "decision_index": decision_index,
        "market": market,
        "universe": universe,
        "extraction": extraction,
        "feature_kwargs": feature_kwargs,
        "feature": feature,
    }


def _reference_frames(prefix: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    price: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []

    def number(row: Mapping[str, Any], symbol: str, field: str) -> float:
        return float.fromhex(row["observations"][symbol][f"{field}_hex"])

    for row in prefix["lookback_rows"]:
        price.append(
            {
                "date": row["session"],
                "aapl_open": number(row, "AAPL", "open"),
                "aapl_close": number(row, "AAPL", "close"),
                "aapl_adj_close": number(row, "AAPL", "adjusted_close"),
                "spy_adj_close": number(row, "SPY", "adjusted_close"),
                "qqq_adj_close": number(row, "QQQ", "adjusted_close"),
            }
        )
        context.append(
            {
                "date": row["session"],
                "iwm_adj_close": number(row, "IWM", "adjusted_close"),
                "vix_close": number(row, "VIX", "close"),
                "tnx_close": number(row, "TNX", "close"),
            }
        )
    return pd.DataFrame(price), pd.DataFrame(context)


def test_exact_features_semantics_order_schema_and_independent_hash(
    case: Mapping[str, Any],
) -> None:
    row = case["feature"]
    assert row["market_available"] is True
    assert row["semantic_available"] is True
    assert row["prediction_available"] is True
    assert len(PRICE_FEATURE_COLUMNS) == 24
    assert len(MARKET_SENTIMENT_FEATURE_COLUMNS) == 15
    assert len(FILING_CALENDAR_FEATURE_COLUMNS) == 3
    assert len(SEMANTIC_AGGREGATE_FEATURE_COLUMNS) == 12
    assert len(SEMANTIC_FEATURE_COLUMNS) == 54
    assert row["semantic_feature_names"] == list(SEMANTIC_FEATURE_COLUMNS)
    assert row["ablation_feature_names"] == list(ABLATION_FEATURE_COLUMNS)
    assert row["semantic_feature_schema_sha256"] == canonical_sha256(
        list(SEMANTIC_FEATURE_COLUMNS)
    )
    price_frame, context_frame = _reference_frames(case["market"]["prefix"])
    price = build_downside_price_features(price_frame).iloc[-1]
    sentiment = build_market_sentiment_features(price_frame, context_frame).iloc[-1]
    assert row["price_regime_features_hex"] == {
        name: float(price[name]).hex() for name in PRICE_FEATURE_COLUMNS
    }
    assert row["market_sentiment_features_hex"] == {
        name: float(sentiment[name]).hex()
        for name in MARKET_SENTIMENT_FEATURE_COLUMNS
    }
    expected_market_identity = {
        "schema_version": "aapl-sec-gemma-market-feature-row-v1",
        "accession_number": ACCESSION,
        "decision_session": DECISION_SESSION,
        "market_prefix_chain_identity_sha256": row["bindings"][
            "market_prefix_chain_identity_sha256"
        ],
        "market_available": True,
        "missing_market_observations": [],
        "missing_market_observations_sha256": canonical_sha256([]),
        "price_regime_features_hex": row["price_regime_features_hex"],
        "market_sentiment_features_hex": row["market_sentiment_features_hex"],
    }
    assert row["market_feature_row_sha256"] == canonical_sha256(
        expected_market_identity
    )
    universe = case["universe"]["proof"]
    extraction = case["extraction"]
    expected_extraction_identity = {
        "identity_domain": "aapl-sec-gemma-extraction-causal-identity-v1",
        "accession_number": ACCESSION,
        "form": universe["current_record"]["form"],
        "decision_session": DECISION_SESSION,
        "current_universe_record_sha256": universe["current_record_sha256"],
        "current_source_record_sha256": universe["current_record"][
            "source_record_sha256"
        ],
        "current_content_record_sha256": universe[
            "current_content_record_sha256"
        ],
        "current_primary_document_sha256": universe["current_content_record"][
            "primary_document_sha256"
        ],
        "current_filing_sha256": universe["current_filing_sha256"],
        "prior_same_form_universe_record_sha256": universe[
            "prior_same_form_record_sha256"
        ],
        "prior_same_form_source_record_sha256": universe[
            "prior_same_form_record"
        ]["source_record_sha256"],
        "prior_same_form_content_record_sha256": universe[
            "prior_same_form_content_record_sha256"
        ],
        "prior_same_form_primary_document_sha256": universe[
            "prior_same_form_content_record"
        ]["primary_document_sha256"],
        "prior_same_form_filing_sha256": universe[
            "prior_same_form_filing_sha256"
        ],
        "extraction_status": extraction["extraction_status"],
        "extraction_evidence_authenticated": extraction[
            "extraction_evidence_authenticated"
        ],
        "extraction_request_and_receipt_evidence_sha256": extraction[
            "extraction_evidence_sha256"
        ],
        "extraction_output_sha256": extraction["extraction_output_sha256"],
        "extraction_output_canonical_sha256": extraction[
            "extraction_output_canonical_sha256"
        ],
        "supplied_sentence_ids_sha256": extraction[
            "supplied_sentence_ids_sha256"
        ],
        "document_quality": extraction["document_quality"],
    }
    assert row["bindings"]["extraction_identity_sha256"] == canonical_sha256(
        expected_extraction_identity
    )
    semantic = row["semantic_aggregate_features_hex"]
    assert float.fromhex(semantic["adverse_flag_count"]) == 1.0
    assert float.fromhex(semantic["management_transition_flag"]) == 1.0
    assert float.fromhex(semantic["stated_dimension_fraction"]) == 0.9
    assert float.fromhex(semantic["comparable_dimension_fraction"]) == 0.8
    assert row["semantic_feature_values_hex"][:42] == row[
        "ablation_feature_values_hex"
    ][:42]
    assert row["ablation_feature_values_hex"][42:] == [float(0.0).hex()] * 12
    assert row["market_feature_row_sha256"] != row["feature_row_sha256"]
    assert validate_sec_filing_gemma_feature_row(
        row,
        expected_feature_row_sha256=row["feature_row_sha256"],
        **_feature_validate_kwargs(case),
    ) == row["feature_row_sha256"]


def test_random_frame_matches_existing_pandas_implementations(
    case: Mapping[str, Any],
) -> None:
    market = _market_evidence(_canonical_rows(random_seed=41), "random")
    kwargs = _feature_kwargs(market, case["universe"]["proof"], case["extraction"])
    row = build_sec_filing_gemma_feature_row(**kwargs)
    price_frame, context_frame = _reference_frames(market["prefix"])
    price = build_downside_price_features(price_frame).iloc[-1]
    sentiment = build_market_sentiment_features(price_frame, context_frame).iloc[-1]
    assert row["price_regime_features_hex"] == {
        name: float(price[name]).hex() for name in PRICE_FEATURE_COLUMNS
    }
    assert row["market_sentiment_features_hex"] == {
        name: float(sentiment[name]).hex()
        for name in MARKET_SENTIMENT_FEATURE_COLUMNS
    }


def test_invalid_unusable_and_unauthenticated_extractions_are_distinct(
    case: Mapping[str, Any],
) -> None:
    universe = case["universe"]["proof"]
    invalid_proof = _extraction_proof(universe, status="invalid")
    invalid = build_sec_filing_gemma_feature_row(
        **_feature_kwargs(case["market"], universe, invalid_proof)
    )
    assert invalid["prediction_available"] is True
    assert invalid["semantic_available"] is False
    assert float.fromhex(
        invalid["filing_calendar_features_hex"]["semantic_output_unavailable"]
    ) == 1.0
    assert set(invalid["semantic_aggregate_features_hex"].values()) == {
        float(0.0).hex()
    }
    assert invalid["bindings"]["extraction_output_sha256"] is None
    assert invalid["bindings"]["extraction_identity_sha256"] != invalid[
        "bindings"
    ]["extraction_event_proof_sha256"]

    unusable_proof = _extraction_proof(universe, quality="unusable")
    unusable = build_sec_filing_gemma_feature_row(
        **_feature_kwargs(case["market"], universe, unusable_proof)
    )
    assert unusable["document_quality"] == "unusable"
    assert unusable["prediction_available"] is True
    assert float.fromhex(
        unusable["filing_calendar_features_hex"]["semantic_output_unavailable"]
    ) == 0.0
    assert set(unusable["semantic_aggregate_features_hex"].values()) == {
        float(0.0).hex()
    }

    unauth_proof = _extraction_proof(universe, authenticated=False)
    unauth = build_sec_filing_gemma_feature_row(
        **_feature_kwargs(case["market"], universe, unauth_proof)
    )
    assert unauth["extraction_evidence_authenticated"] is False
    assert unauth["prediction_available"] is False
    assert unauth["fit_eligible"] is False
    assert unauth["semantic_feature_values_hex"] is None
    assert unauth["unavailable_reasons"] == [
        "missing_or_unauthenticated_extraction_evidence"
    ]


@pytest.mark.parametrize(
    "content_field",
    ["current_content_record", "prior_same_form_content_record"],
)
def test_rehashed_universe_content_record_cannot_carry_extra_fields(
    case: Mapping[str, Any], content_field: str
) -> None:
    forged = copy.deepcopy(case["universe"]["proof"])
    forged[content_field]["future_outcome"] = "DOWN"
    hash_field = f"{content_field}_sha256"
    forged[hash_field] = canonical_sha256(forged[content_field])
    body = {
        key: forged[key]
        for key in forged
        if key != "universe_event_proof_sha256"
    }
    forged["universe_event_proof_sha256"] = canonical_sha256(body)
    with pytest.raises(SecFilingGemmaFeatureError, match="Invalid .* content record keys"):
        validate_universe_event_proof(
            forged,
            expected_universe_event_proof_sha256=forged[
                "universe_event_proof_sha256"
            ],
        )


def test_rehashed_invalid_extraction_cannot_carry_noncanonical_sentence_ids(
    case: Mapping[str, Any],
) -> None:
    universe = case["universe"]["proof"]
    forged = _extraction_proof(universe, status="invalid")
    forged["supplied_sentence_ids"] = ["FUTURE_OUTCOME_DOWN", "FUTURE_OUTCOME_DOWN"]
    forged["supplied_sentence_ids_sha256"] = canonical_sha256(
        forged["supplied_sentence_ids"]
    )
    body = {
        key: forged[key]
        for key in forged
        if key != "extraction_event_proof_sha256"
    }
    forged["extraction_event_proof_sha256"] = canonical_sha256(body)
    with pytest.raises(SecFilingGemmaFeatureError, match="invalid sentence ids"):
        validate_extraction_event_proof(
            forged,
            universe_event_proof=universe,
            expected_universe_event_proof_sha256=universe[
                "universe_event_proof_sha256"
            ],
            expected_extraction_event_proof_sha256=forged[
                "extraction_event_proof_sha256"
            ],
        )


def test_missing_any_of_exact_253_six_symbol_support_rows_is_unavailable(
    case: Mapping[str, Any],
) -> None:
    missing_index = case["decision_index"] - 200
    market = _market_evidence(
        _canonical_rows(missing={(missing_index, "TNX")}), "missing"
    )
    kwargs = _feature_kwargs(
        market, case["universe"]["proof"], case["extraction"]
    )
    row = build_sec_filing_gemma_feature_row(**kwargs)
    assert row["market_available"] is False
    assert row["prediction_available"] is False
    assert row["fit_eligible"] is False
    assert row["semantic_feature_values_hex"] is None
    assert row["unavailable_reasons"] == ["missing_required_market_history"]
    assert f"TNX@{market['rows'][missing_index]['session']}" in row[
        "missing_market_observations"
    ]
    # TNX t-200 is not used by a delta endpoint; the complete-support gate,
    # rather than accidental NaN propagation, is what rejects the row.
    assert all(value is not None for value in row["market_sentiment_features_hex"].values())


def test_feature_worker_has_no_future_market_input_and_values_are_invariant(
    case: Mapping[str, Any],
) -> None:
    import inspect

    parameters = inspect.signature(build_sec_filing_gemma_feature_row).parameters
    assert "stage_manifest" not in parameters
    assert "future_market_rows" not in parameters
    changed_rows = _canonical_rows(
        adjusted_open_overrides={case["decision_index"] + 21: 321.0}
    )
    changed = _market_evidence(changed_rows, "changed-future")
    changed_row = build_sec_filing_gemma_feature_row(
        **_feature_kwargs(
            changed, case["universe"]["proof"], case["extraction"]
        )
    )
    original = case["feature"]
    assert changed_row["price_regime_features_hex"] == original[
        "price_regime_features_hex"
    ]
    assert changed_row["market_sentiment_features_hex"] == original[
        "market_sentiment_features_hex"
    ]
    assert changed_row["semantic_feature_values_hex"] == original[
        "semantic_feature_values_hex"
    ]
    assert changed_row["bindings"][
        "market_prefix_chain_identity_sha256"
    ] == original["bindings"]["market_prefix_chain_identity_sha256"]
    assert changed_row["market_feature_row_sha256"] == original[
        "market_feature_row_sha256"
    ]
    # Full provenance still records the changed future source/stage artifacts.
    assert changed_row["bindings"]["market_prefix_sha256"] != original[
        "bindings"
    ]["market_prefix_sha256"]
    assert changed_row["bindings"]["market_prefix_proof_sha256"] != original[
        "bindings"
    ]["market_prefix_proof_sha256"]
    assert changed_row["feature_row_sha256"] != original["feature_row_sha256"]


def test_future_only_corpus_and_content_extension_keeps_extraction_identity(
    case: Mapping[str, Any],
) -> None:
    extended_universe_proof = copy.deepcopy(case["universe"]["proof"])
    extended_universe_proof["corpus_universe_sha256"] = _digest(
        "universe-after-future-only-extension"
    )
    extended_universe_proof["content_manifest_sha256s_by_stage"][
        "development"
    ] = _digest("content-stage-after-future-only-extension")
    _rehash_universe_event_proof(extended_universe_proof)
    extended_extraction_proof = _extraction_proof(extended_universe_proof)
    extended_row = build_sec_filing_gemma_feature_row(
        **_feature_kwargs(
            case["market"], extended_universe_proof, extended_extraction_proof
        )
    )
    original = case["feature"]

    assert extended_universe_proof["universe_event_proof_sha256"] != case[
        "universe"
    ]["proof"]["universe_event_proof_sha256"]
    assert extended_extraction_proof["extraction_event_proof_sha256"] != case[
        "extraction"
    ]["extraction_event_proof_sha256"]
    assert extended_row["bindings"]["extraction_identity_sha256"] == original[
        "bindings"
    ]["extraction_identity_sha256"]
    assert extended_row["semantic_feature_values_hex"] == original[
        "semantic_feature_values_hex"
    ]
    assert extended_row["ablation_feature_values_hex"] == original[
        "ablation_feature_values_hex"
    ]
    assert extended_row["feature_row_sha256"] != original["feature_row_sha256"]


def test_extraction_identity_binds_exact_output_request_and_receipt(
    case: Mapping[str, Any],
) -> None:
    universe = case["universe"]["proof"]

    def identity(proof: Mapping[str, Any]) -> str:
        row = build_sec_filing_gemma_feature_row(
            **_feature_kwargs(case["market"], universe, proof)
        )
        return row["bindings"]["extraction_identity_sha256"]

    base = _extraction_proof(universe, evidence_label="fixed-receipt")
    changed_output = _extraction_proof(
        universe,
        quality="thin",
        evidence_label="fixed-receipt",
    )
    changed_request = _extraction_proof(
        universe,
        evidence_label="fixed-receipt",
        supplied_sentence_ids=("C0001", "C0002", "P0001"),
    )
    changed_receipt = _extraction_proof(
        universe,
        evidence_label="replacement-receipt",
    )
    identities = {
        identity(base),
        identity(changed_output),
        identity(changed_request),
        identity(changed_receipt),
    }
    assert len(identities) == 4


@pytest.mark.parametrize(
    "identity_part",
    [
        "current_source",
        "prior_source",
        "current_filing",
        "prior_filing",
    ],
)
def test_extraction_identity_binds_current_and_prior_event_local_sources(
    case: Mapping[str, Any], identity_part: str
) -> None:
    changed_universe_proof = copy.deepcopy(case["universe"]["proof"])
    replacement = _digest(f"replacement:{identity_part}")
    if identity_part == "current_source":
        record = changed_universe_proof["current_record"]
        record["source_record_sha256"] = replacement
        changed_universe_proof["current_record_sha256"] = canonical_sha256(record)
    elif identity_part == "prior_source":
        record = changed_universe_proof["prior_same_form_record"]
        record["source_record_sha256"] = replacement
        changed_universe_proof["prior_same_form_record_sha256"] = canonical_sha256(
            record
        )
    elif identity_part == "current_filing":
        record = changed_universe_proof["current_content_record"]
        record["normalized_text_sha256"] = replacement
        changed_universe_proof["current_content_record_sha256"] = canonical_sha256(
            record
        )
        changed_universe_proof["current_filing_sha256"] = replacement
    else:
        record = changed_universe_proof["prior_same_form_content_record"]
        record["normalized_text_sha256"] = replacement
        changed_universe_proof[
            "prior_same_form_content_record_sha256"
        ] = canonical_sha256(record)
        changed_universe_proof["prior_same_form_filing_sha256"] = replacement
    _rehash_universe_event_proof(changed_universe_proof)
    changed_extraction_proof = _extraction_proof(changed_universe_proof)
    changed_row = build_sec_filing_gemma_feature_row(
        **_feature_kwargs(
            case["market"], changed_universe_proof, changed_extraction_proof
        )
    )
    assert changed_row["bindings"]["extraction_identity_sha256"] != case[
        "feature"
    ]["bindings"]["extraction_identity_sha256"]


def test_unsealed_event_local_source_tampering_is_rejected(
    case: Mapping[str, Any],
) -> None:
    unsealed_tamper = copy.deepcopy(case["universe"]["proof"])
    unsealed_tamper["current_record"]["source_record_sha256"] = _digest(
        "unsealed-source-replacement"
    )
    with pytest.raises(SecFilingGemmaFeatureError, match="record hash changed"):
        validate_universe_event_proof(
            unsealed_tamper,
            expected_universe_event_proof_sha256=unsealed_tamper[
                "universe_event_proof_sha256"
            ],
        )


def test_label_is_derived_from_stage_with_all_opens_costs_and_strict_tolerance(
    case: Mapping[str, Any],
) -> None:
    kwargs = _label_kwargs(case)
    label = build_twenty_session_label_evidence(**kwargs)
    assert label["future_market_row_count"] == 21
    assert len(label["adjusted_open_path"]) == 21
    assert label["entry_session"] == EXPECTED_MARKET_HISTORY_SESSIONS[
        case["decision_index"] + 1
    ]
    assert label["exit_session"] == EXPECTED_MARKET_HISTORY_SESSIONS[
        case["decision_index"] + 21
    ]
    entry = float.fromhex(label["entry_adjusted_open_hex"])
    exit_value = float.fromhex(label["exit_adjusted_open_hex"])
    forward = math.log(exit_value / entry)
    edge_5 = math.log1p(-0.0005) - math.log1p(0.0005) - forward
    edge_10 = math.log1p(-0.001) - math.log1p(0.001) - forward
    assert label["cash_active_log_edge_5bps_hex"] == edge_5.hex()
    assert label["cash_active_log_edge_10bps_hex"] == edge_10.hex()
    assert label["cash_beats_long_5bps"] is True
    assert edge_10 <= ACTIVE_EDGE_TOLERANCE
    assert label["cash_beats_long_10bps"] is False
    assert label["market_feature_row_sha256"] == case["feature"][
        "market_feature_row_sha256"
    ]
    assert label["extraction_identity_sha256"] == case["feature"]["bindings"][
        "extraction_identity_sha256"
    ]
    assert validate_twenty_session_label_evidence(
        label,
        expected_label_evidence_sha256=label["label_evidence_sha256"],
        **kwargs,
    ) == label["label_evidence_sha256"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("accession_number",), "nonsense"),
        (("cash_beats_long_10bps",), 0),
        (("future_market_rows_sha256",), "0" * 64),
        (("extraction_identity_sha256",), "1" * 64),
        (("adjusted_open_path", 10, "source_market_row_sha256"), "2" * 64),
    ],
)
def test_rehashed_label_forgery_fails_authoritative_replay(
    case: Mapping[str, Any], path: tuple[Any, ...], replacement: Any
) -> None:
    kwargs = _label_kwargs(case)
    forged = copy.deepcopy(build_twenty_session_label_evidence(**kwargs))
    target: Any = forged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    if path[0] == "adjusted_open_path":
        forged["adjusted_open_path_sha256"] = canonical_sha256(
            forged["adjusted_open_path"]
        )
    body = {key: forged[key] for key in forged if key != "label_evidence_sha256"}
    forged["label_evidence_sha256"] = canonical_sha256(body)
    with pytest.raises(SecFilingGemmaFeatureError, match="authoritative stage replay"):
        validate_twenty_session_label_evidence(
            forged,
            expected_label_evidence_sha256=forged["label_evidence_sha256"],
            **kwargs,
        )


def test_rehashed_feature_and_proof_mutations_fail_source_replay(
    case: Mapping[str, Any],
) -> None:
    forged = copy.deepcopy(case["feature"])
    forged["bindings"]["contract_sha256"] = "0" * 64
    forged["bindings_sha256"] = canonical_sha256(forged["bindings"])
    body = {key: forged[key] for key in forged if key != "feature_row_sha256"}
    forged["feature_row_sha256"] = canonical_sha256(body)
    with pytest.raises(SecFilingGemmaFeatureError, match="compact-proof replay"):
        validate_sec_filing_gemma_feature_row(
            forged,
            expected_feature_row_sha256=forged["feature_row_sha256"],
            **_feature_validate_kwargs(case),
        )

    forged = copy.deepcopy(case["feature"])
    forged["price_regime_features_hex"]["aapl_lr_1"] = None
    body = {key: forged[key] for key in forged if key != "feature_row_sha256"}
    forged["feature_row_sha256"] = canonical_sha256(body)
    with pytest.raises(SecFilingGemmaFeatureError, match="compact-proof replay"):
        validate_sec_filing_gemma_feature_row(
            forged,
            expected_feature_row_sha256=forged["feature_row_sha256"],
            **_feature_validate_kwargs(case),
        )

    bad_proof = copy.deepcopy(case["market"]["proof"])
    bad_proof["lookback_rows_sha256"] = "3" * 64
    proof_body = {
        key: bad_proof[key]
        for key in bad_proof
        if key != "market_prefix_proof_sha256"
    }
    bad_proof["market_prefix_proof_sha256"] = canonical_sha256(proof_body)
    kwargs = dict(case["feature_kwargs"])
    kwargs["market_prefix_proof"] = bad_proof
    kwargs["expected_market_prefix_proof_sha256"] = bad_proof[
        "market_prefix_proof_sha256"
    ]
    with pytest.raises(SecFilingGemmaFeatureError, match="differs from its prefix"):
        build_sec_filing_gemma_feature_row(**kwargs)


def test_prefix_must_have_exact_253_rows_and_source_hash_chain(case: Mapping[str, Any]) -> None:
    short = copy.deepcopy(case["market"]["prefix"])
    short["lookback_rows"] = short["lookback_rows"][1:]
    kwargs = dict(case["feature_kwargs"])
    kwargs["market_prefix"] = short
    with pytest.raises(SecFilingGemmaFeatureError, match="253 rows"):
        build_sec_filing_gemma_feature_row(**kwargs)

    mutated = copy.deepcopy(case["market"]["prefix"])
    mutated["lookback_rows"][-1]["observations"]["AAPL"]["close_hex"] = float(
        999.0
    ).hex()
    kwargs["market_prefix"] = mutated
    with pytest.raises(SecFilingGemmaFeatureError):
        build_sec_filing_gemma_feature_row(**kwargs)
