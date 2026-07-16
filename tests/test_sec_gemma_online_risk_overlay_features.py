from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from typing import Any, Mapping

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    DIMENSION_NAMES,
    FLAG_NAMES,
    build_corpus_universe_manifest,
    build_stage_content_manifest,
)
from agent_benchmark.sec_filing_gemma_features import (
    build_validated_extraction_event_proof,
    build_validated_market_prefix_proof,
    build_validated_universe_event_proof,
    validate_extraction_event_proof as validate_legacy_extraction_event_proof,
    validate_market_prefix_proof as validate_legacy_market_prefix_proof,
    validate_universe_event_proof as validate_legacy_universe_event_proof,
)
from agent_benchmark.sec_filing_gemma_market_evidence import (
    build_decision_market_prefix,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    FEATURES,
    MARKET_FEATURES,
    MEANING_FEATURES,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_features import (
    REQUIRED_PREFIX_ROWS,
    SecGemmaOnlineRiskOverlayFeatureError,
    build_validated_universe_event_proof as build_overlay_universe_event_proof,
    build_sec_gemma_online_risk_overlay_feature_row,
    build_sec_gemma_online_risk_overlay_unavailable_feature_row,
    calculate_market_feature_components,
    calculate_semantic_feature_components,
    validate_sec_gemma_online_risk_overlay_feature_row,
    validate_sec_gemma_online_risk_overlay_unavailable_feature_row,
    validate_extraction_event_proof as validate_overlay_extraction_event_proof,
    validate_market_prefix_proof as validate_overlay_market_prefix_proof,
    validate_universe_event_proof as validate_overlay_universe_event_proof,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
)
from tests import test_sec_filing_gemma_features as v1_helpers
from tests.test_sec_gemma_online_risk_overlay_market_verifier import (
    _market_anomaly_chain,
)


def _market_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slopes = {
        "AAPL": 0.0010,
        "QQQ": 0.0004,
        "SPY": 0.0003,
        "IWM": -0.0002,
        "VIX": 0.0015,
    }
    bases = {
        "AAPL": 30.0,
        "QQQ": 50.0,
        "SPY": 80.0,
        "IWM": 40.0,
        "VIX": 15.0,
    }
    for index in range(REQUIRED_PREFIX_ROWS):
        observations: dict[str, Any] = {}
        for symbol in slopes:
            wobble = (
                0.002 * math.sin(index / 3.0)
                if symbol == "AAPL"
                else 0.0
            )
            value = bases[symbol] * math.exp(
                slopes[symbol] * index + wobble
            )
            observations[symbol] = {
                "available": True,
                "adjusted_close_hex": value.hex(),
            }
        rows.append(
            {
                "session": f"s{index:04d}",
                "observations": observations,
            }
        )
    return rows


def _semantic_output(*, quality: str = "thin") -> dict[str, Any]:
    dimensions = {
        name: {
            "current_impact": "neutral",
            "change_vs_prior": "stable",
            "evidence_sentence_ids": ["C0001", "P0001"],
        }
        for name in DIMENSION_NAMES
    }
    dimensions["demand"]["current_impact"] = "unfavorable"
    dimensions["demand"]["change_vs_prior"] = "deteriorating"
    dimensions["supply_chain"]["current_impact"] = "unfavorable"
    dimensions["gross_margin"]["current_impact"] = "favorable"
    dimensions["gross_margin"]["change_vs_prior"] = "improving"
    dimensions["forward_guidance"]["change_vs_prior"] = "deteriorating"
    flags = {
        name: {
            "present": name in {"new_material_risk", "liquidity_stress"},
            "evidence_sentence_ids": (
                ["C0001"]
                if name in {"new_material_risk", "liquidity_stress"}
                else []
            ),
        }
        for name in FLAG_NAMES
    }
    return {
        "schema_version": "sec-filing-extractor-v1",
        "document_quality": quality,
        "dimensions": dimensions,
        "flags": flags,
    }


def _integration_case(
    *,
    status: str = "valid",
    quality: str = "usable",
    authenticated: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    market = v1_helpers._market_evidence(
        v1_helpers._canonical_rows(), "online-overlay-v2"
    )
    universe = v1_helpers._universe_evidence()
    extraction = v1_helpers._extraction_proof(
        universe["proof"],
        status=status,
        quality=quality,
        authenticated=authenticated,
    )
    kwargs = v1_helpers._feature_kwargs(
        market, universe["proof"], extraction
    )
    return kwargs, build_sec_gemma_online_risk_overlay_feature_row(**kwargs)


def _unavailable_owned_kwargs() -> dict[str, Any]:
    chain = _market_anomaly_chain()
    universe = v1_helpers._universe_evidence()
    extraction = v1_helpers._extraction_proof(
        universe["proof"],
        status="valid",
        quality="usable",
        authenticated=True,
    )
    return {
        "market_source_manifest": chain["source"],
        "expected_market_source_manifest_sha256": chain["source"][
            "source_manifest_sha256"
        ],
        "market_acquisition_receipt": chain["acquisition"],
        "expected_market_acquisition_receipt_sha256": chain[
            "acquisition"
        ]["acquisition_receipt_sha256"],
        "market_verifier_receipt": chain["verifier"],
        "expected_market_verifier_receipt_sha256": chain["verifier"][
            "verifier_receipt_sha256"
        ],
        "market_unavailable_event_proof": chain["proof"],
        "expected_market_unavailable_event_proof_sha256": chain["proof"][
            "market_unavailable_event_proof_sha256"
        ],
        "universe_event_proof": universe["proof"],
        "expected_universe_event_proof_sha256": universe["proof"][
            "universe_event_proof_sha256"
        ],
        "extraction_event_proof": extraction,
        "expected_extraction_event_proof_sha256": extraction[
            "extraction_event_proof_sha256"
        ],
    }


def _extraction_proof_from_output(
    universe_proof: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    output_bytes = json.dumps(
        output, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    evidence_sha256 = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return build_validated_extraction_event_proof(
        universe_event_proof=universe_proof,
        expected_universe_event_proof_sha256=universe_proof[
            "universe_event_proof_sha256"
        ],
        extraction_status="valid",
        extraction_evidence_sha256=evidence_sha256,
        expected_extraction_evidence_sha256=evidence_sha256,
        extractor_output=output,
        extractor_output_bytes=output_bytes,
        expected_extraction_output_sha256=hashlib.sha256(
            output_bytes
        ).hexdigest(),
        expected_extraction_output_canonical_sha256=canonical_sha256(
            output
        ),
        supplied_sentence_ids=("C0001",),
    )


def _first_same_form_case(
    *,
    change_value: str = "not_comparable",
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_universe = v1_helpers._universe_evidence()["universe"]
    raw_keys = (
        "accession_number",
        "subject_cik",
        "form",
        "acceptance_datetime",
        "filing_date",
        "filing_date_change",
        "primary_document",
        "source_record_sha256",
    )
    raw_records = [
        {key: record[key] for key in raw_keys}
        for record in base_universe["records"]
    ]
    raw_records[0]["acceptance_datetime"] = None
    universe = build_corpus_universe_manifest(
        catalog_artifact_sha256=base_universe[
            "catalog_artifact_sha256"
        ],
        calendar_artifact_sha256=base_universe[
            "calendar_artifact_sha256"
        ],
        catalog_total_record_count=len(raw_records),
        catalog_eligible_record_count=len(raw_records),
        session_dates=EXPECTED_SESSIONS,
        records=raw_records,
    )
    current = universe["records"][0]
    documents = [
        {
            "accession_number": record["accession_number"],
            "primary_document_sha256": v1_helpers._digest(
                f"{record['accession_number']}:primary"
            ),
            "normalized_text_sha256": v1_helpers._digest(
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
    universe_proof = build_validated_universe_event_proof(
        universe_manifest=universe,
        expected_corpus_universe_sha256=universe["universe_sha256"],
        current_accession_number=current["accession_number"],
        content_manifests_by_stage={"development": content},
        expected_content_manifest_sha256s={
            "development": content["content_manifest_sha256"]
        },
    )

    market = v1_helpers._market_evidence(
        v1_helpers._canonical_rows(), "first-same-form-overlay"
    )
    prefix = build_decision_market_prefix(
        stage_manifest=market["stage"],
        source_manifest=market["source"],
        expected_artifact_stage="development",
        expected_source_manifest_sha256=market["source"][
            "source_manifest_sha256"
        ],
        expected_market_stage_manifest_sha256=market["stage"][
            "market_stage_manifest_sha256"
        ],
        decision_event_id=current["accession_number"],
        decision_session=current["availability_session"],
    )
    market_proof = build_validated_market_prefix_proof(
        prefix=prefix,
        stage_manifest=market["stage"],
        source_manifest=market["source"],
        expected_artifact_stage="development",
        expected_source_manifest_sha256=market["source"][
            "source_manifest_sha256"
        ],
        expected_market_stage_manifest_sha256=market["stage"][
            "market_stage_manifest_sha256"
        ],
        expected_decision_event_id=current["accession_number"],
        expected_decision_session=current["availability_session"],
        expected_market_prefix_sha256=prefix["market_prefix_sha256"],
    )

    output = v1_helpers._extractor_output("usable")
    for dimension in output["dimensions"].values():
        dimension["change_vs_prior"] = change_value
        dimension["evidence_sentence_ids"] = (
            []
            if dimension["current_impact"] == "not_stated"
            else ["C0001"]
        )
    extraction = _extraction_proof_from_output(
        universe_proof,
        output,
        label=f"first-same-form:{change_value}",
    )
    kwargs = {
        "market_prefix": prefix,
        "market_prefix_proof": market_proof,
        "expected_market_prefix_proof_sha256": market_proof[
            "market_prefix_proof_sha256"
        ],
        "universe_event_proof": universe_proof,
        "expected_universe_event_proof_sha256": universe_proof[
            "universe_event_proof_sha256"
        ],
        "extraction_event_proof": extraction,
        "expected_extraction_event_proof_sha256": extraction[
            "extraction_event_proof_sha256"
        ],
    }
    return kwargs, output


def test_pure_stdlib_proof_replay_matches_pinned_v1_validators() -> None:
    market = v1_helpers._market_evidence(
        v1_helpers._canonical_rows(), "overlay-proof-parity"
    )
    universe = v1_helpers._universe_evidence()
    extraction = v1_helpers._extraction_proof(
        universe["proof"],
        status="valid",
        quality="usable",
        authenticated=True,
    )

    rebuilt_universe = build_overlay_universe_event_proof(
        universe_manifest=universe["universe"],
        expected_corpus_universe_sha256=universe["universe"][
            "universe_sha256"
        ],
        current_accession_number=universe["proof"]["current_record"][
            "accession_number"
        ],
        content_manifests_by_stage={
            "development": universe["content"]
        },
        expected_content_manifest_sha256s={
            "development": universe["content"][
                "content_manifest_sha256"
            ]
        },
    )
    assert rebuilt_universe == universe["proof"]
    assert validate_overlay_market_prefix_proof(
        market["proof"],
        prefix=market["prefix"],
        expected_market_prefix_proof_sha256=market["proof"][
            "market_prefix_proof_sha256"
        ],
    ) == validate_legacy_market_prefix_proof(
        market["proof"],
        prefix=market["prefix"],
        expected_market_prefix_proof_sha256=market["proof"][
            "market_prefix_proof_sha256"
        ],
    )
    assert validate_overlay_universe_event_proof(
        universe["proof"],
        expected_universe_event_proof_sha256=universe["proof"][
            "universe_event_proof_sha256"
        ],
    ) == validate_legacy_universe_event_proof(
        universe["proof"],
        expected_universe_event_proof_sha256=universe["proof"][
            "universe_event_proof_sha256"
        ],
    )
    assert validate_overlay_extraction_event_proof(
        extraction,
        universe_event_proof=universe["proof"],
        expected_universe_event_proof_sha256=universe["proof"][
            "universe_event_proof_sha256"
        ],
        expected_extraction_event_proof_sha256=extraction[
            "extraction_event_proof_sha256"
        ],
    ) == validate_legacy_extraction_event_proof(
        extraction,
        universe_event_proof=universe["proof"],
        expected_universe_event_proof_sha256=universe["proof"][
            "universe_event_proof_sha256"
        ],
        expected_extraction_event_proof_sha256=extraction[
            "extraction_event_proof_sha256"
        ],
    )


def test_market_formulas_use_only_the_exact_completed_prefix() -> None:
    rows = _market_rows()
    result = calculate_market_feature_components(rows)

    assert result["available"] is True
    assert tuple(result["values"]) == MARKET_FEATURES
    values = result["values"]
    t = REQUIRED_PREFIX_ROWS - 1
    t20 = t - 20

    def close(symbol: str, index: int) -> float:
        return float.fromhex(
            rows[index]["observations"][symbol]["adjusted_close_hex"]
        )

    assert values["aapl_minus_qqq_log_return_20"] == pytest.approx(
        math.log(close("AAPL", t) / close("AAPL", t20))
        - math.log(close("QQQ", t) / close("QQQ", t20)),
        abs=1e-15,
    )
    assert values["aapl_drawdown_63"] == pytest.approx(
        close("AAPL", t)
        / max(close("AAPL", index) for index in range(t - 62, t + 1))
        - 1.0,
        abs=1e-15,
    )
    returns = [
        math.log(close("AAPL", index) / close("AAPL", index - 1))
        for index in range(t - 19, t + 1)
    ]
    mean = sum(returns) / len(returns)
    expected_volatility = math.sqrt(
        sum((item - mean) ** 2 for item in returns)
        / (len(returns) - 1)
    ) * math.sqrt(252.0)
    assert values["aapl_realized_volatility_20"] == pytest.approx(
        expected_volatility, abs=1e-15
    )
    assert "schema_version" not in result
    assert not any(key.endswith("_sha256") for key in result)


@pytest.mark.parametrize(
    "mutation, expected_reason",
    [
        ("missing", "AAPL@s0252:unavailable"),
        ("nonpositive", "AAPL@s0252:invalid_adjusted_close"),
        ("noncanonical", "AAPL@s0252:invalid_adjusted_close"),
        ("duplicate", "duplicate_session:s0251"),
    ],
)
def test_market_missingness_is_explicit_and_never_imputed(
    mutation: str, expected_reason: str
) -> None:
    rows = _market_rows()
    if mutation == "missing":
        rows[-1]["observations"]["AAPL"]["available"] = False
    elif mutation == "nonpositive":
        rows[-1]["observations"]["AAPL"][
            "adjusted_close_hex"
        ] = 0.0.hex()
    elif mutation == "noncanonical":
        rows[-1]["observations"]["AAPL"][
            "adjusted_close_hex"
        ] = "0x1.0p+0 "
    elif mutation == "duplicate":
        rows[-1]["session"] = rows[-2]["session"]

    result = calculate_market_feature_components(rows)

    assert result["available"] is False
    assert all(value is None for value in result["values"].values())
    assert expected_reason in result["unavailable_reasons"]


def test_semantic_numeric_components_encode_meaning_and_quality() -> None:
    result = calculate_semantic_feature_components(
        extraction_status="valid",
        extraction_authenticated=True,
        document_quality="thin",
        validated_output=_semantic_output(),
        has_prior_same_form=True,
    )

    assert result["extraction_available"] is True
    assert result["schema_valid_extraction"] is True
    assert result["semantic_quality_risk"] == 0.5
    assert result["meaning_values"] == pytest.approx(
        {
            "commercial_deterioration": 0.5,
            "financial_deterioration": -0.25,
            "risk_outlook_deterioration": 1.0 / 6.0,
            "adverse_flag_fraction": 0.4,
        }
    )
    assert "schema_version" not in result
    assert not any(key.endswith("_sha256") for key in result)


def test_invalid_and_unauthenticated_numeric_states_do_not_fake_meaning() -> None:
    invalid = calculate_semantic_feature_components(
        extraction_status="invalid",
        extraction_authenticated=True,
        document_quality=None,
        validated_output=None,
        has_prior_same_form=True,
    )
    unavailable = calculate_semantic_feature_components(
        extraction_status="unavailable",
        extraction_authenticated=False,
        document_quality=None,
        validated_output=None,
        has_prior_same_form=True,
    )

    assert invalid["extraction_available"] is True
    assert invalid["schema_valid_extraction"] is False
    assert unavailable["extraction_available"] is False
    assert unavailable["schema_valid_extraction"] is False
    for result in (invalid, unavailable):
        assert set(result["meaning_values"].values()) == {0.0}
        assert result["semantic_quality_risk"] == 1.0


def test_irrelevant_observations_cannot_change_market_components() -> None:
    rows = _market_rows()
    baseline = calculate_market_feature_components(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        row["observations"]["TNX"] = {
            "available": True,
            "adjusted_close_hex": 999.0.hex(),
        }

    assert calculate_market_feature_components(changed) == baseline


def test_extreme_finite_prices_cannot_emit_nonfinite_features() -> None:
    rows = _market_rows()
    rows[-21]["observations"]["AAPL"][
        "adjusted_close_hex"
    ] = (1e-308).hex()
    rows[-1]["observations"]["AAPL"][
        "adjusted_close_hex"
    ] = (1e308).hex()

    with pytest.raises(
        SecGemmaOnlineRiskOverlayFeatureError,
        match="Market feature arithmetic is non-finite",
    ):
        calculate_market_feature_components(rows)


def test_owned_entry_point_authenticates_v1_compact_proofs() -> None:
    kwargs, row = _integration_case()

    assert row["contract_sha256"] == CONTRACT_SHA256
    assert row["accession_number"] == v1_helpers.ACCESSION
    assert row["decision_session"] == v1_helpers.DECISION_SESSION
    assert row["prediction_available"] is True
    assert len(row["semantic_values_hex"]) == len(FEATURES)
    assert validate_sec_gemma_online_risk_overlay_feature_row(
        row,
        expected_feature_row_sha256=row["feature_row_sha256"],
        **kwargs,
    ) == row["feature_row_sha256"]


def test_owned_ablation_preserves_quality_but_zeros_gemma_channels() -> None:
    _, row = _integration_case(quality="thin")

    semantic = [
        float.fromhex(value) for value in row["semantic_values_hex"]
    ]
    no_meaning = [
        float.fromhex(value)
        for value in row["no_filing_meaning_values_hex"]
    ]
    no_gemma = [
        float.fromhex(value)
        for value in row["no_gemma_channel_values_hex"]
    ]
    assert semantic[:6] == no_meaning[:6] == no_gemma[:6]
    assert any(abs(value) > 0.0 for value in semantic[6:10])
    assert no_meaning[6:10] == [0.0] * 4
    assert semantic[10] == no_meaning[10] == 0.5
    assert no_gemma[10] == 0.0


def test_owned_invalid_output_is_trainable_but_has_no_fake_meaning() -> None:
    _, row = _integration_case(status="invalid")

    assert row["prediction_available"] is True
    assert row["schema_valid_extraction"] is False
    assert row["meaning_nonzero"] is False
    assert row["semantic_values_hex"] == row[
        "no_filing_meaning_values_hex"
    ]
    assert float.fromhex(row["semantic_values_hex"][10]) == 1.0


def test_reachable_authenticated_unusable_output_preserves_quality_only() -> None:
    kwargs, row = _integration_case(status="valid", quality="unusable")

    assert kwargs["extraction_event_proof"][
        "extraction_evidence_authenticated"
    ] is True
    assert kwargs["extraction_event_proof"]["validated_output"] == (
        v1_helpers._extractor_output("unusable")
    )
    assert row["prediction_available"] is True
    assert row["schema_valid_extraction"] is True
    assert row["meaning_nonzero"] is False
    assert set(row["meaning_values_hex"].values()) == {0.0.hex()}
    assert float.fromhex(row["semantic_values_hex"][10]) == 1.0
    assert float.fromhex(row["no_gemma_channel_values_hex"][10]) == 0.0


def test_normal_entry_cannot_mint_unverified_market_unavailable_row() -> None:
    decision_index = EXPECTED_MARKET_HISTORY_SESSIONS.index(
        v1_helpers.DECISION_SESSION
    )
    market = v1_helpers._market_evidence(
        v1_helpers._canonical_rows(missing={(decision_index, "VIX")}),
        "missing-market-overlay",
    )
    universe = v1_helpers._universe_evidence()
    extraction = v1_helpers._extraction_proof(universe["proof"])

    with pytest.raises(
        SecGemmaOnlineRiskOverlayFeatureError,
        match="requires the unavailable evidence entry point",
    ):
        build_sec_gemma_online_risk_overlay_feature_row(
            **v1_helpers._feature_kwargs(
                market, universe["proof"], extraction
            )
        )


def test_owned_unavailable_path_replays_chain_and_has_row_validator() -> None:
    kwargs = _unavailable_owned_kwargs()
    row = build_sec_gemma_online_risk_overlay_unavailable_feature_row(
        **kwargs
    )

    assert row["prediction_available"] is False
    assert row["fit_eligible"] is False
    assert row["market_available"] is False
    assert row["market_result"]["unavailable_reasons"] == [
        (
            f"QQQ@{v1_helpers.DECISION_SESSION}:adjusted_close:"
            "nonpositive_required_adjusted_close"
        )
    ]
    assert row["semantic_values_hex"] is None
    assert row["upstream_bindings"][
        "market_anomalies_sha256"
    ] == kwargs["market_unavailable_event_proof"]["anomalies_sha256"]
    assert validate_sec_gemma_online_risk_overlay_unavailable_feature_row(
        row,
        expected_feature_row_sha256=row["feature_row_sha256"],
        **kwargs,
    ) == row["feature_row_sha256"]


def test_unavailable_row_validation_rejects_row_or_chain_mutation() -> None:
    kwargs = _unavailable_owned_kwargs()
    row = build_sec_gemma_online_risk_overlay_unavailable_feature_row(
        **kwargs
    )
    changed_row = copy.deepcopy(row)
    changed_row["fit_eligible"] = True

    with pytest.raises(
        SecGemmaOnlineRiskOverlayFeatureError,
        match="differs from authenticated proof replay",
    ):
        validate_sec_gemma_online_risk_overlay_unavailable_feature_row(
            changed_row,
            expected_feature_row_sha256=row["feature_row_sha256"],
            **kwargs,
        )

    wrong_pin = dict(kwargs)
    wrong_pin[
        "expected_market_unavailable_event_proof_sha256"
    ] = "f" * 64
    with pytest.raises(ValueError, match="not externally pinned"):
        build_sec_gemma_online_risk_overlay_unavailable_feature_row(
            **wrong_pin
        )


def test_first_same_form_uses_current_meaning_and_zero_comparison() -> None:
    kwargs, output = _first_same_form_case(
        change_value="not_comparable"
    )
    row = build_sec_gemma_online_risk_overlay_feature_row(**kwargs)
    expected = calculate_semantic_feature_components(
        extraction_status="valid",
        extraction_authenticated=True,
        document_quality="usable",
        validated_output=output,
        has_prior_same_form=False,
    )

    assert kwargs["universe_event_proof"][
        "prior_same_form_record"
    ] is None
    assert row["acceptance_datetime"] is None
    assert "acceptance_timestamp_exact" not in row
    assert "acceptance_order_key" not in row
    assert row["meaning_nonzero"] is True
    assert {
        name: float.fromhex(row["meaning_values_hex"][name])
        for name in MEANING_FEATURES
    } == pytest.approx(expected["meaning_values"])


def test_first_same_form_rejects_authenticated_noncanonical_change_claims() -> None:
    kwargs, _ = _first_same_form_case(change_value="not_stated")
    assert kwargs["extraction_event_proof"][
        "extraction_evidence_authenticated"
    ] is True

    with pytest.raises(
        SecGemmaOnlineRiskOverlayFeatureError,
        match="must mark every comparative change not_comparable",
    ):
        build_sec_gemma_online_risk_overlay_feature_row(**kwargs)


def test_feature_row_validation_rejects_any_mutation() -> None:
    kwargs, row = _integration_case()
    changed = copy.deepcopy(row)
    changed["meaning_nonzero"] = not changed["meaning_nonzero"]

    with pytest.raises(
        SecGemmaOnlineRiskOverlayFeatureError,
        match="differs from authenticated proof replay",
    ):
        validate_sec_gemma_online_risk_overlay_feature_row(
            changed,
            expected_feature_row_sha256=row["feature_row_sha256"],
            **kwargs,
        )


def test_only_owned_proof_authenticated_entry_points_mint_rows() -> None:
    from agent_benchmark import (
        sec_gemma_online_risk_overlay_features as module,
    )

    assert not hasattr(module, "_assemble_feature_row")
    assert not hasattr(
        module, "_build_feature_row_from_validated_inputs"
    )
    assert not hasattr(module, "build_market_feature_result")
    assert {
        name
        for name in module.__all__
        if name.startswith("build_") and "feature_row" in name
    } == {
        "build_sec_gemma_online_risk_overlay_feature_row",
        "build_sec_gemma_online_risk_overlay_unavailable_feature_row",
    }

    normal_parameters = inspect.signature(
        build_sec_gemma_online_risk_overlay_feature_row
    ).parameters
    assert set(normal_parameters) == {
        "market_prefix",
        "market_prefix_proof",
        "expected_market_prefix_proof_sha256",
        "universe_event_proof",
        "expected_universe_event_proof_sha256",
        "extraction_event_proof",
        "expected_extraction_event_proof_sha256",
    }
    unavailable_parameters = inspect.signature(
        build_sec_gemma_online_risk_overlay_unavailable_feature_row
    ).parameters
    assert {
        "market_source_manifest",
        "expected_market_source_manifest_sha256",
        "market_acquisition_receipt",
        "expected_market_acquisition_receipt_sha256",
        "market_verifier_receipt",
        "expected_market_verifier_receipt_sha256",
        "market_unavailable_event_proof",
        "expected_market_unavailable_event_proof_sha256",
    }.issubset(unavailable_parameters)
    for parameters in (normal_parameters, unavailable_parameters):
        assert not any(
            token in name
            for name in parameters
            for token in ("outcome", "label", "future", "return")
        )
