from __future__ import annotations

import copy
import hashlib
import inspect
from typing import Any
from urllib.parse import quote, urlencode

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_market_verifier import (
    MARKET_SYMBOLS,
    PROVIDER_ENDPOINT,
    PROVIDER_SYMBOLS,
    SecGemmaOnlineRiskOverlayMarketStageTerminalError,
    SecGemmaOnlineRiskOverlayMarketVerifierError,
    build_market_anomaly_acquisition_receipt,
    build_market_anomaly_source_manifest,
    build_market_anomaly_verifier_receipt,
    build_market_unavailable_event_proof,
    validate_market_anomaly_acquisition_receipt,
    validate_market_anomaly_source_manifest,
    validate_market_anomaly_verifier_receipt,
    validate_market_unavailable_event_proof,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
)
from tests import test_sec_filing_gemma_features as v1_helpers


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _request_receipts(source: dict[str, Any]) -> list[dict[str, Any]]:
    window = source["request_window"]
    query = urlencode(
        [
            ("period1", str(window["period1_utc"])),
            ("period2", str(window["period2_utc"])),
            ("interval", "1d"),
            ("includePrePost", "false"),
            ("includeAdjustedClose", "true"),
            ("events", "div,splits"),
        ]
    )
    responses = {
        response["symbol"]: response for response in source["responses"]
    }
    return [
        {
            "ordinal": ordinal,
            "symbol": symbol,
            "provider_symbol": PROVIDER_SYMBOLS[symbol],
            "request_url": (
                f"{PROVIDER_ENDPOINT}/"
                f"{quote(PROVIDER_SYMBOLS[symbol], safe='')}?{query}"
            ),
            "http_status": 200,
            "tls_authenticated": True,
            "redirected": False,
            "content_encoding": "identity",
            "raw_response_sha256": responses[symbol][
                "raw_response_sha256"
            ],
            "raw_response_bytes": responses[symbol][
                "raw_response_bytes"
            ],
        }
        for ordinal, symbol in enumerate(MARKET_SYMBOLS, start=1)
    ]


def _market_anomaly_chain(
    *,
    anomalies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    source = build_market_anomaly_source_manifest(
        artifact_stage="development",
        response_artifacts={
            symbol: {
                "raw_response_sha256": _digest(
                    f"market-anomaly:{symbol}"
                ),
                "raw_response_bytes": 100_000 + index,
            }
            for index, symbol in enumerate(MARKET_SYMBOLS)
        },
    )
    acquisition = build_market_anomaly_acquisition_receipt(
        source_manifest=source,
        expected_source_manifest_sha256=source[
            "source_manifest_sha256"
        ],
        request_receipts=_request_receipts(source),
    )
    if anomalies is None:
        anomalies = [
            {
                "symbol": "QQQ",
                "session": v1_helpers.DECISION_SESSION,
                "field": "adjusted_close",
                "reason": "nonpositive_required_adjusted_close",
                "ledger_exposed": False,
            }
        ]
    verifier = build_market_anomaly_verifier_receipt(
        source_manifest=source,
        expected_source_manifest_sha256=source[
            "source_manifest_sha256"
        ],
        acquisition_receipt=acquisition,
        expected_acquisition_receipt_sha256=acquisition[
            "acquisition_receipt_sha256"
        ],
        artifact_stage="development",
        accession_number=v1_helpers.ACCESSION,
        decision_session=v1_helpers.DECISION_SESSION,
        anomalies=anomalies,
    )
    proof = build_market_unavailable_event_proof(
        source_manifest=source,
        expected_source_manifest_sha256=source[
            "source_manifest_sha256"
        ],
        acquisition_receipt=acquisition,
        expected_acquisition_receipt_sha256=acquisition[
            "acquisition_receipt_sha256"
        ],
        verifier_receipt=verifier,
        expected_verifier_receipt_sha256=verifier[
            "verifier_receipt_sha256"
        ],
    )
    return {
        "source": source,
        "acquisition": acquisition,
        "verifier": verifier,
        "proof": proof,
    }


def _proof_validation_kwargs(
    chain: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_manifest": chain["source"],
        "expected_source_manifest_sha256": chain["source"][
            "source_manifest_sha256"
        ],
        "acquisition_receipt": chain["acquisition"],
        "expected_acquisition_receipt_sha256": chain["acquisition"][
            "acquisition_receipt_sha256"
        ],
        "verifier_receipt": chain["verifier"],
        "expected_verifier_receipt_sha256": chain["verifier"][
            "verifier_receipt_sha256"
        ],
        "expected_market_unavailable_event_proof_sha256": chain["proof"][
            "market_unavailable_event_proof_sha256"
        ],
    }


def test_exact_four_link_chain_round_trips_with_external_hash_pins() -> None:
    chain = _market_anomaly_chain()

    assert validate_market_anomaly_source_manifest(
        chain["source"],
        expected_source_manifest_sha256=chain["source"][
            "source_manifest_sha256"
        ],
    ) == chain["source"]
    assert validate_market_anomaly_acquisition_receipt(
        chain["acquisition"],
        source_manifest=chain["source"],
        expected_source_manifest_sha256=chain["source"][
            "source_manifest_sha256"
        ],
        expected_acquisition_receipt_sha256=chain["acquisition"][
            "acquisition_receipt_sha256"
        ],
    ) == chain["acquisition"]
    assert validate_market_anomaly_verifier_receipt(
        chain["verifier"],
        source_manifest=chain["source"],
        expected_source_manifest_sha256=chain["source"][
            "source_manifest_sha256"
        ],
        acquisition_receipt=chain["acquisition"],
        expected_acquisition_receipt_sha256=chain["acquisition"][
            "acquisition_receipt_sha256"
        ],
        expected_verifier_receipt_sha256=chain["verifier"][
            "verifier_receipt_sha256"
        ],
    ) == chain["verifier"]
    assert validate_market_unavailable_event_proof(
        chain["proof"],
        **_proof_validation_kwargs(chain),
    ) == chain["proof"]
    assert chain["proof"]["anomalies"] == [
        {
            "symbol": "QQQ",
            "session": v1_helpers.DECISION_SESSION,
            "field": "adjusted_close",
            "reason": "nonpositive_required_adjusted_close",
            "ledger_exposed": False,
        }
    ]


def test_every_chain_link_rejects_wrong_external_pin_or_mutation() -> None:
    chain = _market_anomaly_chain()

    with pytest.raises(
        SecGemmaOnlineRiskOverlayMarketVerifierError,
        match="not externally pinned",
    ):
        validate_market_anomaly_source_manifest(
            chain["source"],
            expected_source_manifest_sha256="f" * 64,
        )

    changed_request = copy.deepcopy(chain["acquisition"])
    changed_request["requests"][0]["http_status"] = 500
    with pytest.raises(
        SecGemmaOnlineRiskOverlayMarketVerifierError,
        match="identity, transport, or response changed",
    ):
        validate_market_anomaly_acquisition_receipt(
            changed_request,
            source_manifest=chain["source"],
            expected_source_manifest_sha256=chain["source"][
                "source_manifest_sha256"
            ],
            expected_acquisition_receipt_sha256=chain["acquisition"][
                "acquisition_receipt_sha256"
            ],
        )

    changed_verifier = copy.deepcopy(chain["verifier"])
    changed_verifier["anomalies"][0]["field"] = "session"
    with pytest.raises(
        SecGemmaOnlineRiskOverlayMarketVerifierError,
        match="field and reason are inconsistent",
    ):
        validate_market_anomaly_verifier_receipt(
            changed_verifier,
            source_manifest=chain["source"],
            expected_source_manifest_sha256=chain["source"][
                "source_manifest_sha256"
            ],
            acquisition_receipt=chain["acquisition"],
            expected_acquisition_receipt_sha256=chain["acquisition"][
                "acquisition_receipt_sha256"
            ],
            expected_verifier_receipt_sha256=chain["verifier"][
                "verifier_receipt_sha256"
            ],
        )

    changed_proof = copy.deepcopy(chain["proof"])
    changed_proof["outcomes_included"] = True
    with pytest.raises(
        SecGemmaOnlineRiskOverlayMarketVerifierError,
        match="differs from exact chain replay",
    ):
        validate_market_unavailable_event_proof(
            changed_proof,
            **_proof_validation_kwargs(chain),
        )

    wrong_proof_pin = _proof_validation_kwargs(chain)
    wrong_proof_pin[
        "expected_market_unavailable_event_proof_sha256"
    ] = "e" * 64
    with pytest.raises(
        SecGemmaOnlineRiskOverlayMarketVerifierError,
        match="not externally pinned",
    ):
        validate_market_unavailable_event_proof(
            chain["proof"],
            **wrong_proof_pin,
        )


@pytest.mark.parametrize("claimed_exposure", [False, True])
def test_aapl_anomaly_on_ledger_session_is_stage_terminal(
    claimed_exposure: bool,
) -> None:
    with pytest.raises(
        SecGemmaOnlineRiskOverlayMarketStageTerminalError,
        match="terminally fails the stage",
    ):
        _market_anomaly_chain(
            anomalies=[
                {
                    "symbol": "AAPL",
                    "session": v1_helpers.DECISION_SESSION,
                    "field": "adjusted_close",
                    "reason": "nonpositive_required_adjusted_close",
                    "ledger_exposed": claimed_exposure,
                }
            ]
        )


@pytest.mark.parametrize(
    "anomaly, message",
    [
        (
            {
                "symbol": "TNX",
                "session": v1_helpers.DECISION_SESSION,
                "field": "adjusted_close",
                "reason": "nonpositive_required_adjusted_close",
                "ledger_exposed": False,
            },
            "not used by the frozen feature formulas",
        ),
        (
            {
                "symbol": "QQQ",
                "session": v1_helpers.DECISION_SESSION,
                "field": "session",
                "reason": "nonpositive_required_adjusted_close",
                "ledger_exposed": False,
            },
            "field and reason are inconsistent",
        ),
        (
            {
                "symbol": "QQQ",
                "session": v1_helpers.DECISION_SESSION,
                "field": "adjusted_close",
                "reason": "nonpositive_required_adjusted_close",
                "ledger_exposed": True,
            },
            "ledger exposure classification changed",
        ),
    ],
)
def test_structured_anomalies_reject_irrelevant_or_inconsistent_claims(
    anomaly: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(
        SecGemmaOnlineRiskOverlayMarketVerifierError,
        match=message,
    ):
        _market_anomaly_chain(anomalies=[anomaly])


def test_anomaly_must_hit_a_formula_required_session() -> None:
    decision_index = EXPECTED_MARKET_HISTORY_SESSIONS.index(
        v1_helpers.DECISION_SESSION
    )
    irrelevant = EXPECTED_MARKET_HISTORY_SESSIONS[decision_index - 1]

    with pytest.raises(
        SecGemmaOnlineRiskOverlayMarketVerifierError,
        match="does not affect a required decision-time market value",
    ):
        _market_anomaly_chain(
            anomalies=[
                {
                    "symbol": "QQQ",
                    "session": irrelevant,
                    "field": "adjusted_close",
                    "reason": "missing_required_market_history",
                    "ledger_exposed": False,
                }
            ]
        )


def test_public_verifier_apis_accept_no_market_values_or_outcomes() -> None:
    public_functions = (
        build_market_anomaly_source_manifest,
        build_market_anomaly_acquisition_receipt,
        build_market_anomaly_verifier_receipt,
        build_market_unavailable_event_proof,
        validate_market_anomaly_source_manifest,
        validate_market_anomaly_acquisition_receipt,
        validate_market_anomaly_verifier_receipt,
        validate_market_unavailable_event_proof,
    )
    for function in public_functions:
        parameter_names = inspect.signature(function).parameters
        assert not any(
            token in parameter
            for parameter in parameter_names
            for token in (
                "market_values",
                "adjusted_close",
                "outcome",
                "label",
                "future",
                "return",
            )
        )
