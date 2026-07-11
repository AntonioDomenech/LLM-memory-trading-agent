"""Pure, fail-closed stage-access plans for the SEC/Gemma experiment.

The manifest produced here is an authorization *plan*, not an outcome or a
successful-stage receipt.  It deliberately contains no result, pass, or score
field.  It binds one already registered candidate and one exact reveal
transition to the only SEC documents, presealed market bytes, local model, and
output namespace that an effectful runner may touch.

There is one unavoidable hash-ordering detail.  A reveal request contains the
stage-access-manifest hash, so the manifest cannot also contain that request's
final self-hash without creating a circular hash.  The manifest therefore
stores the complete reveal-request body *except* the manifest hash and request
self-hash.  :func:`validate_stage_access_manifest` then requires the completed,
externally pinned request and verifies both missing hashes exactly.

This module performs no filesystem, network, clock, market-data, SEC, or model
I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hmac
import json
import re
from typing import Any, Final
from urllib.parse import urlsplit

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    MAX_FIT_SECONDS,
    MAX_INPUT_BYTES,
    MAX_MODEL_SECONDS,
    MAX_RUNTIME_SECONDS,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    STAGE_MODEL_CALL_CAPS,
    STAGE_ORDER,
    SecFilingGemmaContractError,
    build_contract_manifest,
    canonical_sha256,
    validate_candidate_manifest,
    validate_corpus_universe_manifest,
    validate_stage_content_manifest,
)
from agent_benchmark.sec_filing_gemma_market_evidence import (
    MARKET_SOURCE_FAMILY,
    MARKET_SYMBOLS,
)
from agent_benchmark.sec_filing_gemma_ollama import MAX_RESPONSE_BYTES
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND,
    REVEAL_REQUEST_SCHEMA_VERSION,
    candidate_design_sha256,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


STAGE_ACCESS_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-access-manifest-v1"
)
MODEL_NAME: Final[str] = "gemma4:12b"
MODEL_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/chat"
MODEL_HOST: Final[str] = "127.0.0.1"

_STAGE_PREREQUISITES: Final[dict[str, str]] = {
    "intermediate": "development",
    "final": "intermediate",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ACCESSION_RE = re.compile(r"0000320193-[0-9]{2}-[0-9]{6}\Z")
_ATTEMPT_RE = re.compile(
    rf"{re.escape(CONTRACT_VERSION)}-attempt-(?P<sequence>[0-9]{{3}})\Z"
)
_PRIMARY_PATH_RE = re.compile(
    r"/Archives/edgar/data/320193/(?P<accession>[0-9]{18})/"
    r"(?P<filename>[A-Za-z0-9][A-Za-z0-9._~-]{0,255})\Z"
)
_OUTPUT_NAMESPACE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_FORBIDDEN_FIELD_TOKENS: Final[frozenset[str]] = frozenset(
    {"result", "results", "pass", "passed", "score", "scores"}
)

_REVEAL_REQUEST_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "registry_sha256",
        "registry_tip_sha256",
        "registered_entry_count",
        "historical_final_reveal_count_lower_bound",
        "stage",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_scope",
        "authorizes_outcome_access",
        "effectful_atomic_single_use_consumption_required",
        "cross_attempt_comparison_permitted",
        "cross_attempt_winner_selection_permitted",
        "globally_pristine_claim",
    }
)
_FULL_REVEAL_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        *_REVEAL_REQUEST_IDENTITY_KEYS,
        "stage_access_manifest_sha256",
        "request_sha256",
    }
)


class SecFilingGemmaStageAccessError(SecFilingGemmaContractError):
    """A stage-access plan is noncanonical, overbroad, or not externally pinned."""


def _json_snapshot(value: Any, location: str) -> Any:
    """Detach one finite JSON value from mutable or hostile caller objects."""

    def detach(item: Any, item_location: str) -> Any:
        if isinstance(item, Mapping):
            try:
                pairs = list(item.items())
            except Exception as exc:
                raise SecFilingGemmaStageAccessError(
                    f"{item_location} could not be detached"
                ) from exc
            result: dict[str, Any] = {}
            for key, child in pairs:
                if not isinstance(key, str) or key in result:
                    raise SecFilingGemmaStageAccessError(
                        f"{item_location} must have unique string keys"
                    )
                result[key] = detach(child, f"{item_location}.{key}")
            return result
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            try:
                values = list(item)
            except Exception as exc:
                raise SecFilingGemmaStageAccessError(
                    f"{item_location} could not be detached"
                ) from exc
            return [
                detach(child, f"{item_location}[{index}]")
                for index, child in enumerate(values)
            ]
        if item is None or type(item) in {str, bool, int, float}:
            return item
        raise SecFilingGemmaStageAccessError(
            f"{item_location} contains a non-JSON value"
        )

    try:
        detached = detach(value, location)
        encoded = json.dumps(
            detached,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SecFilingGemmaStageAccessError(
            f"{location} must be finite canonical JSON"
        ) from exc


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SecFilingGemmaStageAccessError(
            f"{location} must be a string-keyed mapping"
        )
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaStageAccessError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageAccessError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_int(
    value: Any,
    location: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SecFilingGemmaStageAccessError(
            f"{location} must be an integer >= {minimum}"
        )
    if maximum is not None and value > maximum:
        raise SecFilingGemmaStageAccessError(
            f"{location} exceeds its frozen maximum"
        )
    return value


def _reject_forbidden_fields(value: Any, location: str = "stage access manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise SecFilingGemmaStageAccessError(
                    f"{location} keys must be strings"
                )
            tokens = {
                token
                for token in re.split(r"[^A-Za-z0-9]+", key.lower())
                if token
            }
            if tokens & _FORBIDDEN_FIELD_TOKENS:
                raise SecFilingGemmaStageAccessError(
                    "Stage-access manifests cannot contain result, pass, or score fields"
                )
            _reject_forbidden_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{location}[{index}]")


def _transition(prerequisite_stage: Any, requested_stage: Any) -> tuple[str, str]:
    if (
        not isinstance(requested_stage, str)
        or requested_stage not in _STAGE_PREREQUISITES
        or not isinstance(prerequisite_stage, str)
        or _STAGE_PREREQUISITES[requested_stage] != prerequisite_stage
    ):
        raise SecFilingGemmaStageAccessError(
            "Only development-to-intermediate and intermediate-to-final access is permitted"
        )
    return prerequisite_stage, requested_stage


def _attempt_id(value: Any, *, registered_entry_count: int) -> str:
    match = _ATTEMPT_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None or int(match.group("sequence")) != registered_entry_count:
        raise SecFilingGemmaStageAccessError(
            "Attempt id is not the canonical current registry sequence"
        )
    return value


def _canonical_output_namespace(attempt_id: str, requested_stage: str) -> str:
    namespace = f"aapl-sec-gemma-{attempt_id}-{requested_stage}"
    if _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None:
        raise SecFilingGemmaStageAccessError("Derived output namespace is unsafe")
    return namespace


def _canonical_primary_document_url(url: Any, accession: str) -> str:
    if not isinstance(url, str):
        raise SecFilingGemmaStageAccessError(
            "Authorized SEC document URL must be a string"
        )
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise SecFilingGemmaStageAccessError(
            "Authorized SEC document URL is malformed"
        ) from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc != "www.sec.gov"
        or parsed.hostname != "www.sec.gov"
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SecFilingGemmaStageAccessError(
            "Authorized filing URLs must be canonical official SEC HTTPS URLs"
        )
    match = _PRIMARY_PATH_RE.fullmatch(parsed.path)
    if match is None or match.group("accession") != accession.replace("-", ""):
        raise SecFilingGemmaStageAccessError(
            "Authorized SEC URL is not the exact primary-document path for its accession"
        )
    return url


def _canonical_documents(value: Any) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SecFilingGemmaStageAccessError(
            "Authorized SEC documents must be an ordered sequence"
        )
    snapshot = _json_snapshot(list(value), "authorized SEC documents")
    if not isinstance(snapshot, list) or not snapshot:
        raise SecFilingGemmaStageAccessError(
            "Authorized SEC documents cannot be empty"
        )
    documents: list[dict[str, str]] = []
    for index, raw in enumerate(snapshot):
        document = _expect_mapping(raw, f"authorized_documents[{index}]")
        _expect_keys(
            document,
            {"accession_number", "official_url"},
            f"authorized_documents[{index}]",
        )
        accession = document["accession_number"]
        if not isinstance(accession, str) or _ACCESSION_RE.fullmatch(accession) is None:
            raise SecFilingGemmaStageAccessError(
                "Authorized accession is not a canonical Apple accession"
            )
        documents.append(
            {
                "accession_number": accession,
                "official_url": _canonical_primary_document_url(
                    document["official_url"], accession
                ),
            }
        )
    accessions = [item["accession_number"] for item in documents]
    urls = [item["official_url"] for item in documents]
    if (
        accessions != sorted(accessions)
        or urls != sorted(urls)
        or len(accessions) != len(set(accessions))
        or len(urls) != len(set(urls))
    ):
        raise SecFilingGemmaStageAccessError(
            "Authorized accessions and URLs must each be exactly sorted and deduplicated"
        )
    return documents


def _documents_from_complete_universe(
    universe_manifest: Mapping[str, Any], *, requested_stage: str
) -> list[dict[str, str]]:
    records = universe_manifest.get("records")
    if not isinstance(records, list):
        raise SecFilingGemmaStageAccessError(
            "Complete corpus universe records are unavailable"
        )
    documents = [
        {
            "accession_number": record["accession_number"],
            "official_url": (
                "https://www.sec.gov/Archives/edgar/data/320193/"
                f"{record['accession_number'].replace('-', '')}/"
                f"{record['primary_document']}"
            ),
        }
        for record in records
        if record["artifact_stage"] == requested_stage
    ]
    documents.sort(key=lambda item: item["accession_number"])
    return _canonical_documents(documents)


def _prior_same_form_carry_ins(
    universe_manifest: Mapping[str, Any],
    prerequisite_content_manifest: Mapping[str, Any],
    *,
    prerequisite_stage: str,
    requested_stage: str,
    prerequisite_content_manifest_sha256: str,
) -> list[dict[str, Any]]:
    """Derive the exact sealed prior text needed by first-stage comparisons."""

    records = universe_manifest.get("records")
    if not isinstance(records, list):
        raise SecFilingGemmaStageAccessError(
            "Complete corpus universe records are unavailable"
        )
    requested = sorted(
        (
            record
            for record in records
            if record["artifact_stage"] == requested_stage
        ),
        key=lambda record: (
            record["availability_session"],
            record["accession_number"],
        ),
    )
    content_documents = prerequisite_content_manifest.get("documents")
    if not isinstance(content_documents, list):
        raise SecFilingGemmaStageAccessError(
            "Prerequisite content manifest documents are unavailable"
        )
    content_by_accession = {
        document["accession_number"]: document for document in content_documents
    }
    carry_ins: list[dict[str, Any]] = []
    for form in sorted({record["form"] for record in requested}):
        first = next(record for record in requested if record["form"] == form)
        prior = [
            record
            for record in records
            if record["form"] == form
            and (
                record["availability_session"],
                record["accession_number"],
            )
            < (
                first["availability_session"],
                first["accession_number"],
            )
        ]
        if not prior:
            raise SecFilingGemmaStageAccessError(
                "Requested stage lacks its exact prior same-form filing"
            )
        selected = max(
            prior,
            key=lambda record: (
                record["availability_session"],
                record["accession_number"],
            ),
        )
        if selected["artifact_stage"] != prerequisite_stage:
            raise SecFilingGemmaStageAccessError(
                "Prior same-form carry-in is not from the prerequisite stage"
            )
        content_record = content_by_accession.get(selected["accession_number"])
        if not isinstance(content_record, Mapping):
            raise SecFilingGemmaStageAccessError(
                "Prior same-form carry-in is absent from prerequisite content evidence"
            )
        carry_ins.append(
            {
                "accession_number": selected["accession_number"],
                "form": selected["form"],
                "availability_session": selected["availability_session"],
                "artifact_stage": selected["artifact_stage"],
                "source_record_sha256": selected["source_record_sha256"],
                "normalized_text_sha256": content_record[
                    "normalized_text_sha256"
                ],
                "normalized_text_bytes": content_record[
                    "normalized_text_bytes"
                ],
                "content_record_sha256": canonical_sha256(content_record),
                "content_manifest_sha256": prerequisite_content_manifest_sha256,
            }
        )
    return carry_ins


def _canonical_market_sources(
    artifact_hashes: Any, window_hashes: Any
) -> list[dict[str, str]]:
    artifacts = _expect_mapping(
        _json_snapshot(artifact_hashes, "market artifact hashes"),
        "market artifact hashes",
    )
    windows = _expect_mapping(
        _json_snapshot(window_hashes, "market window hashes"),
        "market window hashes",
    )
    expected = set(MARKET_SYMBOLS)
    _expect_keys(artifacts, expected, "market artifact hashes")
    _expect_keys(windows, expected, "market window hashes")
    return [
        {
            "symbol": symbol,
            "artifact_sha256": _sha256(
                artifacts[symbol], f"market artifact {symbol}"
            ),
            "window_sha256": _sha256(
                windows[symbol], f"market window {symbol}"
            ),
        }
        for symbol in MARKET_SYMBOLS
    ]


def _canonical_reveal_request_identity(
    value: Any,
    *,
    prerequisite_stage: str,
    requested_stage: str,
    candidate_sha256: str,
    candidate_design_sha256_value: str,
    attempt_id: str,
    registry_sha256: str,
    registry_tip_sha256: str,
    registered_entry_count: int,
    registry_entry_sha256: str,
) -> dict[str, Any]:
    identity = _expect_mapping(
        _json_snapshot(value, "reveal request identity"),
        "reveal request identity",
    )
    _expect_keys(
        identity,
        set(_REVEAL_REQUEST_IDENTITY_KEYS),
        "reveal request identity",
    )
    _reject_forbidden_fields(identity, "reveal request identity")
    expected_bindings = {
        "schema_version": REVEAL_REQUEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "registry_sha256": registry_sha256,
        "registry_tip_sha256": registry_tip_sha256,
        "registered_entry_count": registered_entry_count,
        "historical_final_reveal_count_lower_bound": (
            HISTORICAL_FINAL_REVEAL_COUNT_LOWER_BOUND
        ),
        "stage": requested_stage,
        "prerequisite_stage": prerequisite_stage,
        "attempt_id": attempt_id,
        "candidate_sha256": candidate_sha256,
        "candidate_design_sha256": candidate_design_sha256_value,
        "registry_entry_sha256": registry_entry_sha256,
        "request_scope": "one_current_tip_candidate_and_one_stage_only",
        "authorizes_outcome_access": False,
        "effectful_atomic_single_use_consumption_required": True,
        "cross_attempt_comparison_permitted": False,
        "cross_attempt_winner_selection_permitted": False,
        "globally_pristine_claim": False,
    }
    for key, expected_value in expected_bindings.items():
        if identity[key] != expected_value or type(identity[key]) is not type(
            expected_value
        ):
            raise SecFilingGemmaStageAccessError(
                f"Reveal request identity is not bound to {key}"
            )
    _sha256(
        identity["prerequisite_stage_evidence_sha256"],
        "prerequisite_stage_evidence_sha256",
    )
    return dict(identity)


def _validated_completed_reveal_request(
    request: Any,
    *,
    expected_manifest_sha256: str,
    expected_request_sha256: str,
) -> dict[str, Any]:
    value = _expect_mapping(
        _json_snapshot(request, "completed reveal request"),
        "completed reveal request",
    )
    _expect_keys(value, set(_FULL_REVEAL_REQUEST_KEYS), "completed reveal request")
    _reject_forbidden_fields(value, "completed reveal request")
    access_hash = _sha256(
        value["stage_access_manifest_sha256"],
        "reveal request stage_access_manifest_sha256",
    )
    if not hmac.compare_digest(access_hash, expected_manifest_sha256):
        raise SecFilingGemmaStageAccessError(
            "Reveal request does not bind the exact stage-access manifest"
        )
    observed_request_hash = _sha256(
        value["request_sha256"], "reveal request request_sha256"
    )
    body = {key: value[key] for key in value if key != "request_sha256"}
    calculated_request_hash = canonical_sha256(body)
    if (
        not hmac.compare_digest(observed_request_hash, calculated_request_hash)
        or not hmac.compare_digest(
            observed_request_hash,
            _sha256(expected_request_sha256, "expected_reveal_request_sha256"),
        )
    ):
        raise SecFilingGemmaStageAccessError(
            "Reveal request body or external request pin changed"
        )
    return {
        key: value[key]
        for key in value
        if key not in {"stage_access_manifest_sha256", "request_sha256"}
    }


def build_stage_access_manifest(
    *,
    prerequisite_stage: str,
    requested_stage: str,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    expected_candidate_design_sha256: str,
    expected_attempt_id: str,
    expected_stage_verifier_source_sha256: str,
    reveal_request_identity: Mapping[str, Any],
    registry_sha256: str,
    registry_tip_sha256: str,
    registered_entry_count: int,
    registry_entry_sha256: str,
    base_corpus_universe_sha256: str,
    corpus_universe_manifest: Mapping[str, Any],
    prerequisite_content_manifest: Mapping[str, Any],
    expected_prerequisite_content_manifest_sha256: str,
    session_calendar_sha256: str,
    authorized_documents: Sequence[Mapping[str, Any]],
    market_source_manifest_sha256: str,
    market_source_artifact_sha256s: Mapping[str, str],
    market_source_window_sha256s: Mapping[str, str],
    max_sec_response_bytes: int,
    output_namespace: str,
) -> dict[str, Any]:
    """Build one exact, non-authorizing stage-access plan.

    ``reveal_request_identity`` is the canonical reveal-request body with only
    ``stage_access_manifest_sha256`` and ``request_sha256`` omitted.  The
    completed request is required by the validator after this manifest is
    externally pinned.
    """

    prerequisite, requested = _transition(prerequisite_stage, requested_stage)
    candidate = _expect_mapping(
        _json_snapshot(candidate_manifest, "candidate manifest"),
        "candidate manifest",
    )
    candidate_hash = validate_candidate_manifest(
        candidate,
        expected_candidate_sha256=_sha256(
            expected_candidate_sha256, "expected_candidate_sha256"
        ),
    )
    try:
        design_hash = candidate_design_sha256(candidate)
    except ValueError as exc:
        raise SecFilingGemmaStageAccessError(
            "Candidate design identity is invalid"
        ) from exc
    expected_design_hash = _sha256(
        expected_candidate_design_sha256, "expected_candidate_design_sha256"
    )
    if not hmac.compare_digest(design_hash, expected_design_hash):
        raise SecFilingGemmaStageAccessError(
            "Candidate design does not match its external pin"
        )
    count = _strict_int(
        registered_entry_count,
        "registered_entry_count",
        minimum=1,
        maximum=999,
    )
    attempt = _attempt_id(expected_attempt_id, registered_entry_count=count)
    if candidate["bindings"]["holdout_attempt_id"] != attempt:
        raise SecFilingGemmaStageAccessError(
            "Candidate manifest does not bind the expected attempt id"
        )
    verifier_source_hash = _sha256(
        expected_stage_verifier_source_sha256,
        "expected_stage_verifier_source_sha256",
    )
    if (
        candidate["bindings"]["source_hashes"]["stage_verifier"]
        != verifier_source_hash
    ):
        raise SecFilingGemmaStageAccessError(
            "Stage verifier source differs from the immutable candidate pin"
        )
    registry_hash = _sha256(registry_sha256, "registry_sha256")
    registry_tip = _sha256(registry_tip_sha256, "registry_tip_sha256")
    registry_entry_hash = _sha256(
        registry_entry_sha256, "registry_entry_sha256"
    )
    universe_hash = _sha256(
        base_corpus_universe_sha256, "base_corpus_universe_sha256"
    )
    calendar_hash = _sha256(session_calendar_sha256, "session_calendar_sha256")
    if (
        candidate["bindings"]["corpus_universe_sha256"] != universe_hash
        or candidate["bindings"]["calendar_sessions_sha256"] != calendar_hash
    ):
        raise SecFilingGemmaStageAccessError(
            "Stage-access immutable corpus or calendar differs from the candidate"
        )

    universe = _expect_mapping(
        _json_snapshot(corpus_universe_manifest, "complete corpus universe"),
        "complete corpus universe",
    )
    try:
        validated_universe_hash = validate_corpus_universe_manifest(
            universe,
            session_dates=EXPECTED_SESSIONS,
            expected_universe_sha256=universe_hash,
            require_complete_coverage=True,
        )
    except SecFilingGemmaContractError as exc:
        raise SecFilingGemmaStageAccessError(
            "Stage access requires the complete externally pinned corpus universe"
        ) from exc
    if (
        validated_universe_hash != universe_hash
        or universe["universe_semantic_sha256"]
        != candidate["bindings"]["corpus_universe_semantic_sha256"]
        or universe["catalog_artifact_sha256"]
        != candidate["bindings"]["sec_catalog_artifact_sha256"]
        or universe["calendar_artifact_sha256"]
        != candidate["bindings"]["calendar_source_evidence_sha256"]
        or universe["calendar_sessions_sha256"] != calendar_hash
    ):
        raise SecFilingGemmaStageAccessError(
            "Complete corpus evidence differs from the immutable candidate"
        )

    prerequisite_content = _expect_mapping(
        _json_snapshot(
            prerequisite_content_manifest,
            "prerequisite stage content manifest",
        ),
        "prerequisite stage content manifest",
    )
    expected_content_hash = _sha256(
        expected_prerequisite_content_manifest_sha256,
        "expected_prerequisite_content_manifest_sha256",
    )
    try:
        content_hash = validate_stage_content_manifest(
            prerequisite_content,
            universe_manifest=universe,
            expected_content_manifest_sha256=expected_content_hash,
        )
    except SecFilingGemmaContractError as exc:
        raise SecFilingGemmaStageAccessError(
            "Stage access requires validated prerequisite content evidence"
        ) from exc
    if (
        prerequisite_content["artifact_stage"] != prerequisite
        or prerequisite_content["corpus_universe_sha256"] != universe_hash
    ):
        raise SecFilingGemmaStageAccessError(
            "Prerequisite content evidence belongs to another stage or universe"
        )

    model = _expect_mapping(candidate["model"], "candidate model")
    if model["name"] != MODEL_NAME or model["endpoint"] != MODEL_ENDPOINT:
        raise SecFilingGemmaStageAccessError(
            "Only the frozen loopback Gemma model may be authorized"
        )
    model_digest = _sha256(model["digest"], "candidate model digest")
    runtime_fingerprint = _sha256(
        model["runtime_fingerprint_sha256"],
        "candidate runtime fingerprint",
    )

    request_identity = _canonical_reveal_request_identity(
        reveal_request_identity,
        prerequisite_stage=prerequisite,
        requested_stage=requested,
        candidate_sha256=candidate_hash,
        candidate_design_sha256_value=design_hash,
        attempt_id=attempt,
        registry_sha256=registry_hash,
        registry_tip_sha256=registry_tip,
        registered_entry_count=count,
        registry_entry_sha256=registry_entry_hash,
    )
    documents = _documents_from_complete_universe(
        universe, requested_stage=requested
    )
    caller_documents = _canonical_documents(authorized_documents)
    if caller_documents != documents:
        raise SecFilingGemmaStageAccessError(
            "Authorized SEC documents must be all and only the complete "
            "requested-stage universe primary documents"
        )
    document_count = len(documents)
    if (
        document_count > STAGE_MODEL_CALL_CAPS[requested]
        or document_count > MAX_SEC_REQUESTS
    ):
        raise SecFilingGemmaStageAccessError(
            "Authorized document count exceeds the stage call or request cap"
        )
    sec_byte_budget = _strict_int(
        max_sec_response_bytes,
        "max_sec_response_bytes",
        minimum=1,
        maximum=MAX_SEC_BYTES,
    )
    market_sources = _canonical_market_sources(
        market_source_artifact_sha256s,
        market_source_window_sha256s,
    )
    market_manifest_hash = _sha256(
        market_source_manifest_sha256, "market_source_manifest_sha256"
    )
    expected_namespace = _canonical_output_namespace(attempt, requested)
    if (
        not isinstance(output_namespace, str)
        or _OUTPUT_NAMESPACE_RE.fullmatch(output_namespace) is None
        or output_namespace != expected_namespace
    ):
        raise SecFilingGemmaStageAccessError(
            "Output namespace must be the exact candidate-and-stage namespace"
        )

    accessions = [item["accession_number"] for item in documents]
    urls = [item["official_url"] for item in documents]
    prior_same_form_carry_ins = _prior_same_form_carry_ins(
        universe,
        prerequisite_content,
        prerequisite_stage=prerequisite,
        requested_stage=requested,
        prerequisite_content_manifest_sha256=content_hash,
    )
    prohibited_stages = [stage for stage in STAGE_ORDER if stage != requested]
    body: dict[str, Any] = {
        "schema_version": STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "transition": {
            "prerequisite_stage": prerequisite,
            "requested_stage": requested,
            "transition_ordinal": STAGE_ORDER.index(requested),
            "single_use_consumption_required": True,
            "stage_reuse_permitted": False,
        },
        "candidate": {
            "candidate_sha256": candidate_hash,
            "candidate_design_sha256": design_hash,
            "attempt_id": attempt,
        },
        "verifier": {
            "stage_verifier_source_sha256": verifier_source_hash,
            "source_identity": "exact_candidate_bound_immutable_bytes",
            "runtime_source_substitution_permitted": False,
        },
        "reveal_request_binding": {
            "request_body_without_manifest_or_self_hash": request_identity,
            "request_body_identity_sha256": canonical_sha256(request_identity),
            "completed_request_self_hash_required": True,
            "completed_request_external_pin_required": True,
            "completed_request_must_bind_this_manifest": True,
        },
        "registry": {
            "registry_sha256": registry_hash,
            "registry_tip_sha256": registry_tip,
            "registered_entry_count": count,
            "registry_entry_sha256": registry_entry_hash,
        },
        "corpus_provenance": {
            "frozen_base_universe": {
                "corpus_universe_sha256": universe_hash,
                "corpus_universe_semantic_sha256": candidate["bindings"][
                    "corpus_universe_semantic_sha256"
                ],
                "sec_catalog_artifact_sha256": candidate["bindings"][
                    "sec_catalog_artifact_sha256"
                ],
                "calendar_source_evidence_sha256": candidate["bindings"][
                    "calendar_source_evidence_sha256"
                ],
                "session_calendar_sha256": calendar_hash,
                "identity_role": "immutable_holdout_base",
            },
            "live_append_only_extension": {
                "identity_sha256": None,
                "entry_count": 0,
                "included_in_base_universe_identity": False,
                "authorized_for_holdout_stage": False,
                "may_replace_or_rewrite_base_universe": False,
            },
        },
        "sec_access_plan": {
            "selection_policy": (
                "all_and_only_requested_stage_universe_primary_documents"
            ),
            "method": "GET",
            "network_scope": "official_sec_https_only",
            "redirects_permitted": False,
            "retries_permitted": False,
            "cache_substitution_permitted": False,
            "document_count": document_count,
            "accessions_sha256": canonical_sha256(accessions),
            "official_urls_sha256": canonical_sha256(urls),
            "documents": documents,
        },
        "prior_same_form_carry_in": {
            "selection_policy": (
                "latest_prerequisite_stage_filing_of_each_requested_stage_form"
            ),
            "artifact_scope": "sealed_normalized_text_only",
            "network_refetch_permitted": False,
            "write_permitted": False,
            "bound_by_prerequisite_stage_evidence_sha256": request_identity[
                "prerequisite_stage_evidence_sha256"
            ],
            "prerequisite_content_manifest_sha256": content_hash,
            "record_count": len(prior_same_form_carry_ins),
            "records_sha256": canonical_sha256(prior_same_form_carry_ins),
            "records": prior_same_form_carry_ins,
        },
        "market_access": {
            "artifact_stage": requested,
            "source_family": MARKET_SOURCE_FAMILY,
            "source_manifest_sha256": market_manifest_hash,
            "source_bytes_presealed_before_request": True,
            "network_reads_permitted": False,
            "sources": market_sources,
        },
        "model_access": {
            "name": MODEL_NAME,
            "digest": model_digest,
            "runtime_fingerprint_sha256": runtime_fingerprint,
            "endpoint": MODEL_ENDPOINT,
            "hosts": [MODEL_HOST],
            "network_scope": "ipv4_loopback_only",
            "pull_attempts": 0,
            "retries": 0,
            "repair_attempts": 0,
            "paid_api_calls": 0,
            "estimated_cost_usd": 0.0,
        },
        "budgets": {
            "max_total_runtime_seconds": MAX_RUNTIME_SECONDS,
            "max_sec_acquisition_seconds": MAX_SEC_SECONDS,
            "max_model_seconds": MAX_MODEL_SECONDS,
            "max_fit_simulation_sealing_seconds": MAX_FIT_SECONDS,
            "max_sec_requests": document_count,
            "max_sec_response_bytes": sec_byte_budget,
            "max_model_calls": document_count,
            "max_model_input_bytes_per_call": MAX_INPUT_BYTES,
            "max_model_response_bytes_per_call": MAX_RESPONSE_BYTES,
            "max_total_model_input_bytes": document_count * MAX_INPUT_BYTES,
            "max_total_model_response_bytes": document_count * MAX_RESPONSE_BYTES,
            "max_paid_api_calls": 0,
            "max_estimated_cost_usd": 0.0,
        },
        "output": {
            "namespace": output_namespace,
            "write_mode": "create_new_exclusive",
            "existing_namespace_reuse_permitted": False,
            "cross_stage_write_permitted": False,
        },
        "scope": {
            "authorized_stage": requested,
            "prohibited_stages": prohibited_stages,
            "prerequisite_stage_data_scope": (
                "sealed_evidence_identity_plus_exact_read_only_prior_same_form_"
                "normalized_text_carry_in"
            ),
            "general_cross_stage_access_permitted": False,
            "exact_prior_same_form_carry_in_read_permitted": True,
            "future_stage_access_permitted": False,
            "outcome_access_before_atomic_request_consumption_permitted": False,
        },
        "authorization_semantics": (
            "non_authorizing_plan_until_external_pin_request_validation_"
            "and_atomic_single_use_consumption"
        ),
    }
    _reject_forbidden_fields(body)
    return {**body, "stage_access_manifest_sha256": canonical_sha256(body)}


def validate_stage_access_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_stage_access_manifest_sha256: str,
    prerequisite_stage: str,
    requested_stage: str,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    expected_candidate_design_sha256: str,
    expected_attempt_id: str,
    expected_stage_verifier_source_sha256: str,
    reveal_request: Mapping[str, Any] | None = None,
    expected_reveal_request_sha256: str,
    registry_sha256: str,
    registry_tip_sha256: str,
    registered_entry_count: int,
    registry_entry_sha256: str,
    base_corpus_universe_sha256: str,
    corpus_universe_manifest: Mapping[str, Any],
    prerequisite_content_manifest: Mapping[str, Any],
    expected_prerequisite_content_manifest_sha256: str,
    session_calendar_sha256: str,
    authorized_documents: Sequence[Mapping[str, Any]],
    market_source_manifest_sha256: str,
    market_source_artifact_sha256s: Mapping[str, str],
    market_source_window_sha256s: Mapping[str, str],
    max_sec_response_bytes: int,
    output_namespace: str,
) -> str:
    """Validate exact construction, self-hash, request hash, and external pin."""

    observed = _expect_mapping(
        _json_snapshot(manifest, "stage access manifest"),
        "stage access manifest",
    )
    _reject_forbidden_fields(observed)
    expected_top_keys = {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "transition",
        "candidate",
        "verifier",
        "reveal_request_binding",
        "registry",
        "corpus_provenance",
        "sec_access_plan",
        "prior_same_form_carry_in",
        "market_access",
        "model_access",
        "budgets",
        "output",
        "scope",
        "authorization_semantics",
        "stage_access_manifest_sha256",
    }
    _expect_keys(observed, expected_top_keys, "stage access manifest")
    observed_hash = _sha256(
        observed["stage_access_manifest_sha256"],
        "stage_access_manifest_sha256",
    )
    body = {
        key: observed[key]
        for key in observed
        if key != "stage_access_manifest_sha256"
    }
    calculated_hash = canonical_sha256(body)
    external_hash = _sha256(
        expected_stage_access_manifest_sha256,
        "expected_stage_access_manifest_sha256",
    )
    if (
        not hmac.compare_digest(observed_hash, calculated_hash)
        or not hmac.compare_digest(observed_hash, external_hash)
    ):
        raise SecFilingGemmaStageAccessError(
            "Stage-access manifest is not canonical or externally pinned"
        )

    completed_request: Mapping[str, Any]
    if reveal_request is None:
        binding = _expect_mapping(
            observed["reveal_request_binding"],
            "manifest reveal request binding",
        )
        identity = _expect_mapping(
            binding.get("request_body_without_manifest_or_self_hash"),
            "manifest reveal request identity",
        )
        request_body = {
            **dict(identity),
            "stage_access_manifest_sha256": observed_hash,
        }
        completed_request = {
            **request_body,
            "request_sha256": canonical_sha256(request_body),
        }
    else:
        completed_request = reveal_request
    request_identity = _validated_completed_reveal_request(
        completed_request,
        expected_manifest_sha256=observed_hash,
        expected_request_sha256=expected_reveal_request_sha256,
    )
    expected = build_stage_access_manifest(
        prerequisite_stage=prerequisite_stage,
        requested_stage=requested_stage,
        candidate_manifest=candidate_manifest,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_candidate_design_sha256=expected_candidate_design_sha256,
        expected_attempt_id=expected_attempt_id,
        expected_stage_verifier_source_sha256=(
            expected_stage_verifier_source_sha256
        ),
        reveal_request_identity=request_identity,
        registry_sha256=registry_sha256,
        registry_tip_sha256=registry_tip_sha256,
        registered_entry_count=registered_entry_count,
        registry_entry_sha256=registry_entry_sha256,
        base_corpus_universe_sha256=base_corpus_universe_sha256,
        corpus_universe_manifest=corpus_universe_manifest,
        prerequisite_content_manifest=prerequisite_content_manifest,
        expected_prerequisite_content_manifest_sha256=(
            expected_prerequisite_content_manifest_sha256
        ),
        session_calendar_sha256=session_calendar_sha256,
        authorized_documents=authorized_documents,
        market_source_manifest_sha256=market_source_manifest_sha256,
        market_source_artifact_sha256s=market_source_artifact_sha256s,
        market_source_window_sha256s=market_source_window_sha256s,
        max_sec_response_bytes=max_sec_response_bytes,
        output_namespace=output_namespace,
    )
    if observed != expected or canonical_sha256(body) != expected[
        "stage_access_manifest_sha256"
    ]:
        raise SecFilingGemmaStageAccessError(
            "Stage-access manifest differs from its exact externally bound construction"
        )
    return observed_hash


__all__ = [
    "MODEL_ENDPOINT",
    "MODEL_HOST",
    "MODEL_NAME",
    "STAGE_ACCESS_MANIFEST_SCHEMA_VERSION",
    "SecFilingGemmaStageAccessError",
    "build_stage_access_manifest",
    "validate_stage_access_manifest",
]
