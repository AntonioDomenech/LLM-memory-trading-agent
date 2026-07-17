"""Pure, replayable market-anomaly evidence for the SEC/Gemma v2 overlay.

An unavailable feature row must not be justified by a caller-supplied reason
and a self-hash.  This module therefore validates a four-link chain:

1. the exact six-response source manifest;
2. the exact owned acquisition receipt;
3. the verifier receipt containing structured, decision-time anomalies; and
4. the compact unavailable-event proof consumed by the feature worker.

Every link is canonical JSON, externally hash-pinned, and cross-bound to the
same contract, stage, provider requests, filing event, and anomaly records.
No market value, outcome, label, prediction, or trading action is accepted by
any public API in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import date
import hmac
import re
from typing import Any, Final
from urllib.parse import quote, urlencode

from agent_benchmark.sec_filing_gemma_contract import STAGE_WINDOWS
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
)


MARKET_ANOMALY_SOURCE_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-overlay-market-anomaly-source-v1"
)
MARKET_ANOMALY_ACQUISITION_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-overlay-market-anomaly-acquisition-v1"
)
MARKET_ANOMALY_VERIFIER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-overlay-market-anomaly-verifier-v1"
)
MARKET_UNAVAILABLE_PROOF_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-market-unavailable-proof-v2"
)

MARKET_SYMBOLS: Final[tuple[str, ...]] = (
    "AAPL",
    "SPY",
    "QQQ",
    "IWM",
    "VIX",
    "TNX",
)
FEATURE_MARKET_SYMBOLS: Final[frozenset[str]] = frozenset(
    {"AAPL", "QQQ", "SPY", "IWM", "VIX"}
)
PROVIDER_SYMBOLS: Final[dict[str, str]] = {
    "AAPL": "AAPL",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "VIX": "^VIX",
    "TNX": "^TNX",
}
PROVIDER_FAMILY: Final[str] = (
    "yahoo-finance-chart-v8-public-unauthenticated"
)
PROVIDER_ENDPOINT: Final[str] = (
    "https://query1.finance.yahoo.com/v8/finance/chart"
)
LEDGER_FIRST_SESSION: Final[str] = "2000-01-03"

MARKET_UNAVAILABLE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "missing_required_market_history",
        "duplicate_required_market_session",
        "nonfinite_required_adjusted_close",
        "nonpositive_required_adjusted_close",
        "malformed_required_adjusted_close",
    }
)
_REASON_FIELD: Final[dict[str, str]] = {
    "missing_required_market_history": "adjusted_close",
    "duplicate_required_market_session": "session",
    "nonfinite_required_adjusted_close": "adjusted_close",
    "nonpositive_required_adjusted_close": "adjusted_close",
    "malformed_required_adjusted_close": "adjusted_close",
}
_STAGE_WINDOW_KEY: Final[dict[str, str]] = {
    "development": "development",
    "intermediate": "confirmation",
    "final": "final",
}
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_ACCESSION_RE: Final[re.Pattern[str]] = re.compile(
    r"0000320193-[0-9]{2}-[0-9]{6}\Z"
)

_SOURCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "artifact_stage",
        "provider_family",
        "endpoint",
        "request_window",
        "request_order",
        "responses",
        "responses_sha256",
        "source_manifest_sha256",
    }
)
_RESPONSE_KEYS: Final[frozenset[str]] = frozenset(
    {"symbol", "provider_symbol", "raw_response_sha256", "raw_response_bytes"}
)
_ACQUISITION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "artifact_stage",
        "provider_family",
        "endpoint",
        "request_window",
        "source_manifest_sha256",
        "request_count",
        "request_order",
        "requests",
        "requests_sha256",
        "market_values_exposed",
        "outcomes_accessed",
        "acquisition_receipt_sha256",
    }
)
_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ordinal",
        "symbol",
        "provider_symbol",
        "request_url",
        "http_status",
        "tls_authenticated",
        "redirected",
        "content_encoding",
        "raw_response_sha256",
        "raw_response_bytes",
    }
)
_ANOMALY_KEYS: Final[frozenset[str]] = frozenset(
    {"symbol", "session", "field", "reason", "ledger_exposed"}
)
_VERIFIER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "artifact_stage",
        "accession_number",
        "decision_session",
        "source_manifest_sha256",
        "acquisition_receipt_sha256",
        "anomalies",
        "anomalies_sha256",
        "market_values_exposed",
        "outcomes_accessed",
        "feature_values_derived",
        "stage_terminal",
        "verifier_receipt_sha256",
    }
)
_PROOF_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "artifact_stage",
        "accession_number",
        "decision_session",
        "source_manifest_sha256",
        "acquisition_receipt_sha256",
        "verifier_receipt_sha256",
        "anomalies",
        "anomalies_sha256",
        "market_values_included",
        "outcomes_included",
        "market_unavailable_event_proof_sha256",
    }
)


class SecGemmaOnlineRiskOverlayMarketVerifierError(ValueError):
    """Raised when the anomaly evidence chain is noncanonical or unbound."""


class SecGemmaOnlineRiskOverlayMarketStageTerminalError(
    SecGemmaOnlineRiskOverlayMarketVerifierError
):
    """Raised when an anomaly makes the whole same-ledger stage invalid."""


def _plain_json(value: Any, location: str) -> Any:
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or key in result:
                raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                    f"{location} must use unique built-in string keys"
                )
            result[key] = _plain_json(child, f"{location}.{key}")
        return result
    if type(value) is list:
        return [
            _plain_json(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise SecGemmaOnlineRiskOverlayMarketVerifierError(
        f"{location} must contain only detached plain JSON values"
    )


def _mapping(value: Any, location: str) -> dict[str, Any]:
    detached = _plain_json(value, location)
    if type(detached) is not dict:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            f"{location} must be a mapping"
        )
    return detached


def _expect_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    location: str,
) -> None:
    if set(value) != set(expected):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            f"{location} keys changed"
        )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            f"{location} must be a lowercase SHA-256"
        )
    return value


def _positive_int(value: Any, location: str) -> int:
    if type(value) is not int or value < 1:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            f"{location} must be an integer >= 1"
        )
    return value


def _artifact_stage(value: Any) -> str:
    if type(value) is not str or value not in _STAGE_WINDOW_KEY:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "artifact_stage must be development, intermediate, or final"
        )
    return value


def _iso_session(value: Any, location: str) -> str:
    if type(value) is not str:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            f"{location} must be a canonical market session"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            f"{location} must be a canonical market session"
        ) from exc
    if (
        parsed.isoformat() != value
        or value not in EXPECTED_MARKET_HISTORY_SESSIONS
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            f"{location} is outside the frozen market calendar"
        )
    return value


def _request_window(stage: str) -> dict[str, Any]:
    source = build_contract_manifest()["data"]["market"]["source_acquisition"]
    window = source["request_windows"][_STAGE_WINDOW_KEY[stage]]
    return copy.deepcopy(window)


def _request_url(stage: str, symbol: str) -> str:
    window = _request_window(stage)
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
    provider = quote(PROVIDER_SYMBOLS[symbol], safe="")
    return f"{PROVIDER_ENDPOINT}/{provider}?{query}"


def _response_artifacts_from_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    responses = manifest.get("responses")
    if type(responses) is not list or len(responses) != len(MARKET_SYMBOLS):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market source responses must contain exactly six items"
        )
    result: dict[str, dict[str, Any]] = {}
    for index, response_value in enumerate(responses):
        response = _mapping(response_value, f"responses[{index}]")
        _expect_keys(response, _RESPONSE_KEYS, f"responses[{index}]")
        symbol = response["symbol"]
        if symbol != MARKET_SYMBOLS[index] or symbol in result:
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                "Market source responses are missing, duplicated, or reordered"
            )
        result[symbol] = {
            "raw_response_sha256": response["raw_response_sha256"],
            "raw_response_bytes": response["raw_response_bytes"],
        }
    return result


def build_market_anomaly_source_manifest(
    *,
    artifact_stage: str,
    response_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the exact six provider-response byte identities without values."""

    stage = _artifact_stage(artifact_stage)
    artifacts = _mapping(response_artifacts, "response_artifacts")
    if set(artifacts) != set(MARKET_SYMBOLS):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "response_artifacts must contain the exact six frozen symbols"
        )
    responses: list[dict[str, Any]] = []
    for symbol in MARKET_SYMBOLS:
        artifact = _mapping(
            artifacts[symbol], f"response_artifacts.{symbol}"
        )
        if set(artifact) != {"raw_response_sha256", "raw_response_bytes"}:
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                f"response_artifacts.{symbol} keys changed"
            )
        responses.append(
            {
                "symbol": symbol,
                "provider_symbol": PROVIDER_SYMBOLS[symbol],
                "raw_response_sha256": _sha256(
                    artifact["raw_response_sha256"],
                    f"{symbol}.raw_response_sha256",
                ),
                "raw_response_bytes": _positive_int(
                    artifact["raw_response_bytes"],
                    f"{symbol}.raw_response_bytes",
                ),
            }
        )
    body = {
        "schema_version": MARKET_ANOMALY_SOURCE_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "artifact_stage": stage,
        "provider_family": PROVIDER_FAMILY,
        "endpoint": PROVIDER_ENDPOINT,
        "request_window": _request_window(stage),
        "request_order": list(MARKET_SYMBOLS),
        "responses": responses,
        "responses_sha256": canonical_sha256(responses),
    }
    return {**body, "source_manifest_sha256": canonical_sha256(body)}


def validate_market_anomaly_source_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_source_manifest_sha256: str,
) -> dict[str, Any]:
    value = _mapping(manifest, "market anomaly source manifest")
    _expect_keys(value, _SOURCE_KEYS, "market anomaly source manifest")
    rebuilt = build_market_anomaly_source_manifest(
        artifact_stage=value["artifact_stage"],
        response_artifacts=_response_artifacts_from_manifest(value),
    )
    if value != rebuilt:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market anomaly source manifest differs from exact replay"
        )
    observed = _sha256(
        value["source_manifest_sha256"], "source_manifest_sha256"
    )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_source_manifest_sha256,
            "expected_source_manifest_sha256",
        ),
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market anomaly source manifest is not externally pinned"
        )
    return copy.deepcopy(value)


def _request_receipts_from_source(
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": index,
            "symbol": response["symbol"],
            "provider_symbol": response["provider_symbol"],
            "request_url": _request_url(
                source["artifact_stage"], response["symbol"]
            ),
            "http_status": 200,
            "tls_authenticated": True,
            "redirected": False,
            "content_encoding": "identity",
            "raw_response_sha256": response["raw_response_sha256"],
            "raw_response_bytes": response["raw_response_bytes"],
        }
        for index, response in enumerate(source["responses"], start=1)
    ]


def build_market_anomaly_acquisition_receipt(
    *,
    source_manifest: Mapping[str, Any],
    expected_source_manifest_sha256: str,
    request_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind exact successful owned requests to the externally pinned source."""

    source = validate_market_anomaly_source_manifest(
        source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    if isinstance(request_receipts, (str, bytes)) or not isinstance(
        request_receipts, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "request_receipts must be an ordered sequence"
        )
    requests = [
        _mapping(value, f"request_receipts[{index}]")
        for index, value in enumerate(request_receipts)
    ]
    expected_requests = _request_receipts_from_source(source)
    if len(requests) != len(expected_requests):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Acquisition receipt must contain exactly six requests"
        )
    for index, (observed, expected) in enumerate(
        zip(requests, expected_requests, strict=True)
    ):
        _expect_keys(observed, _REQUEST_KEYS, f"request_receipts[{index}]")
        if observed != expected:
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                "Acquisition request identity, transport, or response changed"
            )
    body = {
        "schema_version": (
            MARKET_ANOMALY_ACQUISITION_RECEIPT_SCHEMA_VERSION
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "artifact_stage": source["artifact_stage"],
        "provider_family": PROVIDER_FAMILY,
        "endpoint": PROVIDER_ENDPOINT,
        "request_window": copy.deepcopy(source["request_window"]),
        "source_manifest_sha256": source["source_manifest_sha256"],
        "request_count": len(requests),
        "request_order": list(MARKET_SYMBOLS),
        "requests": requests,
        "requests_sha256": canonical_sha256(requests),
        "market_values_exposed": False,
        "outcomes_accessed": False,
    }
    return {
        **body,
        "acquisition_receipt_sha256": canonical_sha256(body),
    }


def validate_market_anomaly_acquisition_receipt(
    receipt: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
    expected_source_manifest_sha256: str,
    expected_acquisition_receipt_sha256: str,
) -> dict[str, Any]:
    value = _mapping(receipt, "market anomaly acquisition receipt")
    _expect_keys(value, _ACQUISITION_KEYS, "market anomaly acquisition receipt")
    rebuilt = build_market_anomaly_acquisition_receipt(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        request_receipts=value["requests"],
    )
    if value != rebuilt:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market anomaly acquisition receipt differs from exact replay"
        )
    observed = _sha256(
        value["acquisition_receipt_sha256"],
        "acquisition_receipt_sha256",
    )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_acquisition_receipt_sha256,
            "expected_acquisition_receipt_sha256",
        ),
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market anomaly acquisition receipt is not externally pinned"
        )
    return copy.deepcopy(value)


def _event_identity(
    *, artifact_stage: Any, accession_number: Any, decision_session: Any
) -> tuple[str, str, str]:
    stage = _artifact_stage(artifact_stage)
    if (
        type(accession_number) is not str
        or _ACCESSION_RE.fullmatch(accession_number) is None
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "accession_number must be a canonical Apple accession"
        )
    decision = _iso_session(decision_session, "decision_session")
    first, last = STAGE_WINDOWS[stage]
    if not first <= decision <= last:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Decision session is outside its artifact stage"
        )
    return stage, accession_number, decision


def _required_sessions(
    *, symbol: str, decision_session: str
) -> frozenset[str]:
    decision_index = EXPECTED_MARKET_HISTORY_SESSIONS.index(decision_session)
    if decision_index < 252:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Decision lacks the frozen 253-session market prefix"
        )
    if symbol == "AAPL":
        start = decision_index - 62
        return frozenset(
            EXPECTED_MARKET_HISTORY_SESSIONS[start : decision_index + 1]
        )
    return frozenset(
        {
            EXPECTED_MARKET_HISTORY_SESSIONS[decision_index - 20],
            decision_session,
        }
    )


def _normalize_anomalies(
    anomalies: Sequence[Mapping[str, Any]],
    *,
    decision_session: str,
) -> list[dict[str, Any]]:
    if isinstance(anomalies, (str, bytes)) or not isinstance(
        anomalies, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "anomalies must be an ordered sequence"
        )
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(anomalies):
        anomaly = _mapping(raw, f"anomalies[{index}]")
        _expect_keys(anomaly, _ANOMALY_KEYS, f"anomalies[{index}]")
        symbol = anomaly["symbol"]
        if symbol not in FEATURE_MARKET_SYMBOLS:
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                "Anomaly symbol is not used by the frozen feature formulas"
            )
        session = _iso_session(
            anomaly["session"], f"anomalies[{index}].session"
        )
        if session not in _required_sessions(
            symbol=symbol, decision_session=decision_session
        ):
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                "Anomaly does not affect a required decision-time market value"
            )
        reason = anomaly["reason"]
        if reason not in MARKET_UNAVAILABLE_REASONS:
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                "Market anomaly reason is invalid"
            )
        field = anomaly["field"]
        if field != _REASON_FIELD[reason]:
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                "Market anomaly field and reason are inconsistent"
            )
        ledger_exposed = anomaly["ledger_exposed"]
        if type(ledger_exposed) is not bool:
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                "Market anomaly ledger_exposed must be boolean"
            )
        expected_ledger_exposed = (
            symbol == "AAPL" and session >= LEDGER_FIRST_SESSION
        )
        if expected_ledger_exposed:
            raise SecGemmaOnlineRiskOverlayMarketStageTerminalError(
                "AAPL anomaly on a ledger-exposed session terminally fails the stage"
            )
        if ledger_exposed is not expected_ledger_exposed:
            raise SecGemmaOnlineRiskOverlayMarketVerifierError(
                "Market anomaly ledger exposure classification changed"
            )
        normalized.append(
            {
                "symbol": symbol,
                "session": session,
                "field": field,
                "reason": reason,
                "ledger_exposed": ledger_exposed,
            }
        )
    if not normalized:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "At least one market anomaly is required"
        )
    ordered = sorted(
        normalized,
        key=lambda item: (
            item["session"],
            item["symbol"],
            item["field"],
            item["reason"],
        ),
    )
    if normalized != ordered or len(
        {
            (
                item["symbol"],
                item["session"],
                item["field"],
                item["reason"],
            )
            for item in normalized
        }
    ) != len(normalized):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market anomalies must be sorted and unique"
        )
    return normalized


def build_market_anomaly_verifier_receipt(
    *,
    source_manifest: Mapping[str, Any],
    expected_source_manifest_sha256: str,
    acquisition_receipt: Mapping[str, Any],
    expected_acquisition_receipt_sha256: str,
    artifact_stage: str,
    accession_number: str,
    decision_session: str,
    anomalies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify the exact chain and attest structured feature-time anomalies."""

    source = validate_market_anomaly_source_manifest(
        source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    acquisition = validate_market_anomaly_acquisition_receipt(
        acquisition_receipt,
        source_manifest=source,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_acquisition_receipt_sha256=(
            expected_acquisition_receipt_sha256
        ),
    )
    stage, accession, decision = _event_identity(
        artifact_stage=artifact_stage,
        accession_number=accession_number,
        decision_session=decision_session,
    )
    if (
        source["artifact_stage"] != stage
        or acquisition["artifact_stage"] != stage
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market evidence chain belongs to another stage"
        )
    normalized_anomalies = _normalize_anomalies(
        anomalies, decision_session=decision
    )
    body = {
        "schema_version": MARKET_ANOMALY_VERIFIER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "artifact_stage": stage,
        "accession_number": accession,
        "decision_session": decision,
        "source_manifest_sha256": source["source_manifest_sha256"],
        "acquisition_receipt_sha256": acquisition[
            "acquisition_receipt_sha256"
        ],
        "anomalies": normalized_anomalies,
        "anomalies_sha256": canonical_sha256(normalized_anomalies),
        "market_values_exposed": False,
        "outcomes_accessed": False,
        "feature_values_derived": False,
        "stage_terminal": False,
    }
    return {**body, "verifier_receipt_sha256": canonical_sha256(body)}


def validate_market_anomaly_verifier_receipt(
    receipt: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
    expected_source_manifest_sha256: str,
    acquisition_receipt: Mapping[str, Any],
    expected_acquisition_receipt_sha256: str,
    expected_verifier_receipt_sha256: str,
) -> dict[str, Any]:
    value = _mapping(receipt, "market anomaly verifier receipt")
    _expect_keys(value, _VERIFIER_KEYS, "market anomaly verifier receipt")
    rebuilt = build_market_anomaly_verifier_receipt(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        acquisition_receipt=acquisition_receipt,
        expected_acquisition_receipt_sha256=(
            expected_acquisition_receipt_sha256
        ),
        artifact_stage=value["artifact_stage"],
        accession_number=value["accession_number"],
        decision_session=value["decision_session"],
        anomalies=value["anomalies"],
    )
    if value != rebuilt:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market anomaly verifier receipt differs from exact replay"
        )
    observed = _sha256(
        value["verifier_receipt_sha256"], "verifier_receipt_sha256"
    )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_verifier_receipt_sha256,
            "expected_verifier_receipt_sha256",
        ),
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market anomaly verifier receipt is not externally pinned"
        )
    return copy.deepcopy(value)


def build_market_unavailable_event_proof(
    *,
    source_manifest: Mapping[str, Any],
    expected_source_manifest_sha256: str,
    acquisition_receipt: Mapping[str, Any],
    expected_acquisition_receipt_sha256: str,
    verifier_receipt: Mapping[str, Any],
    expected_verifier_receipt_sha256: str,
) -> dict[str, Any]:
    """Compact the fully replayed chain for the unavailable feature worker."""

    source = validate_market_anomaly_source_manifest(
        source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    acquisition = validate_market_anomaly_acquisition_receipt(
        acquisition_receipt,
        source_manifest=source,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_acquisition_receipt_sha256=(
            expected_acquisition_receipt_sha256
        ),
    )
    verifier = validate_market_anomaly_verifier_receipt(
        verifier_receipt,
        source_manifest=source,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        acquisition_receipt=acquisition,
        expected_acquisition_receipt_sha256=(
            expected_acquisition_receipt_sha256
        ),
        expected_verifier_receipt_sha256=expected_verifier_receipt_sha256,
    )
    body = {
        "schema_version": MARKET_UNAVAILABLE_PROOF_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "artifact_stage": verifier["artifact_stage"],
        "accession_number": verifier["accession_number"],
        "decision_session": verifier["decision_session"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "acquisition_receipt_sha256": acquisition[
            "acquisition_receipt_sha256"
        ],
        "verifier_receipt_sha256": verifier["verifier_receipt_sha256"],
        "anomalies": copy.deepcopy(verifier["anomalies"]),
        "anomalies_sha256": verifier["anomalies_sha256"],
        "market_values_included": False,
        "outcomes_included": False,
    }
    return {
        **body,
        "market_unavailable_event_proof_sha256": canonical_sha256(body),
    }


def validate_market_unavailable_event_proof(
    proof: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
    expected_source_manifest_sha256: str,
    acquisition_receipt: Mapping[str, Any],
    expected_acquisition_receipt_sha256: str,
    verifier_receipt: Mapping[str, Any],
    expected_verifier_receipt_sha256: str,
    expected_market_unavailable_event_proof_sha256: str,
) -> dict[str, Any]:
    value = _mapping(proof, "market unavailable event proof")
    _expect_keys(value, _PROOF_KEYS, "market unavailable event proof")
    rebuilt = build_market_unavailable_event_proof(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        acquisition_receipt=acquisition_receipt,
        expected_acquisition_receipt_sha256=(
            expected_acquisition_receipt_sha256
        ),
        verifier_receipt=verifier_receipt,
        expected_verifier_receipt_sha256=expected_verifier_receipt_sha256,
    )
    if value != rebuilt:
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market unavailable event proof differs from exact chain replay"
        )
    observed = _sha256(
        value["market_unavailable_event_proof_sha256"],
        "market_unavailable_event_proof_sha256",
    )
    if not hmac.compare_digest(
        observed,
        _sha256(
            expected_market_unavailable_event_proof_sha256,
            "expected_market_unavailable_event_proof_sha256",
        ),
    ):
        raise SecGemmaOnlineRiskOverlayMarketVerifierError(
            "Market unavailable event proof is not externally pinned"
        )
    return copy.deepcopy(value)


__all__ = [
    "FEATURE_MARKET_SYMBOLS",
    "LEDGER_FIRST_SESSION",
    "MARKET_ANOMALY_ACQUISITION_RECEIPT_SCHEMA_VERSION",
    "MARKET_ANOMALY_SOURCE_MANIFEST_SCHEMA_VERSION",
    "MARKET_ANOMALY_VERIFIER_RECEIPT_SCHEMA_VERSION",
    "MARKET_SYMBOLS",
    "MARKET_UNAVAILABLE_PROOF_SCHEMA_VERSION",
    "MARKET_UNAVAILABLE_REASONS",
    "PROVIDER_ENDPOINT",
    "PROVIDER_FAMILY",
    "PROVIDER_SYMBOLS",
    "SecGemmaOnlineRiskOverlayMarketStageTerminalError",
    "SecGemmaOnlineRiskOverlayMarketVerifierError",
    "build_market_anomaly_acquisition_receipt",
    "build_market_anomaly_source_manifest",
    "build_market_anomaly_verifier_receipt",
    "build_market_unavailable_event_proof",
    "validate_market_anomaly_acquisition_receipt",
    "validate_market_anomaly_source_manifest",
    "validate_market_anomaly_verifier_receipt",
    "validate_market_unavailable_event_proof",
]
