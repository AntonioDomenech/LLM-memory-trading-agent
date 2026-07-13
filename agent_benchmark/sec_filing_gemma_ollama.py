"""Fail-closed, zero-cost Ollama client for the SEC filing extractor.

The client has one fixed destination and one request shape.  It accepts only
the safe payload returned by ``validate_extractor_request`` and it never
performs discovery, pulls, retries, redirects, streaming, or output repair.
Runtime identity can be supplied by a separately sealed local-runtime artifact
or minted by the bounded, loopback-only probe in this module.  The probe reads
only Ollama's version and fixed-model metadata endpoints; it never invokes a
model or discovers, pulls, or mutates installed models.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
import re
import time
from typing import Any, Final, Protocol

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from agent_benchmark.sec_filing_gemma_contract import (
    SecFilingGemmaContractError,
    build_extractor_model_payload,
    canonical_sha256,
    validate_extractor_output,
)


OLLAMA_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/chat"
OLLAMA_VERSION_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/version"
OLLAMA_SHOW_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/show"
OLLAMA_MODEL: Final[str] = "gemma4:12b"
CONNECT_TIMEOUT_SECONDS: Final[float] = 2.0
READ_TIMEOUT_SECONDS: Final[float] = 30.0
MAX_RESPONSE_BYTES: Final[int] = 256 * 1024
RUNTIME_IDENTITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-ollama-runtime-identity-v1"
)
RUNTIME_PROBE_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-ollama-runtime-probe-receipt-v1"
)
RUNTIME_FINGERPRINT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-ollama-runtime-fingerprint-v1"
)
RECEIPT_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-ollama-call-receipt-v1"
ATTEMPT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-ollama-model-attempt-receipt-v1"
)
VALID_ATTEMPT_STATUS: Final[str] = "valid"
INVALID_ATTEMPT_STATUS: Final[str] = "invalid_extractor_output"
INVALID_ATTEMPT_REASON: Final[str] = (
    "extractor_output_schema_invalid_no_retry_no_repair"
)
ABNORMAL_ATTEMPT_REASON: Final[str] = (
    "abnormal_model_completion_no_retry_no_repair"
)
RUNTIME_GUARD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-ollama-runtime-guard-v1"
)
DIAGNOSTIC_TIME_ROLE: Final[str] = "diagnostic_only_not_chronology"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SENTENCE_ID_RE = re.compile(r"[CP][0-9]{4}\Z")
_RFC3339_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?Z\Z"
)
_OLLAMA_VERSION_RE = re.compile(
    r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z][0-9A-Za-z.-]{0,126})?\Z"
)
_MODEL_BLOB_FROM_RE = re.compile(
    r"(?:^|[/\\])sha256[:-]([0-9a-f]{64})\Z"
)
_RUNTIME_IDENTITY_KEYS = {
    "schema_version",
    "endpoint",
    "model_name",
    "model_digest",
    "runtime_fingerprint_sha256",
}
_VALIDATED_REQUEST_KEYS = {
    "candidate_sha256",
    "sentence_ids",
    "model_payload",
    "model_payload_sha256",
}
_OLLAMA_RESPONSE_KEYS = {
    "model",
    "created_at",
    "message",
    "done",
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
}
_OLLAMA_MESSAGE_KEYS = {"role", "content"}
_RUNTIME_PROBE_MANIFEST_KEYS = {
    "schema_version",
    "version_endpoint",
    "show_endpoint",
    "model_name",
    "show_request_bytes_base64",
    "show_request_sha256",
    "version_response_bytes_base64",
    "version_response_sha256",
    "show_response_bytes_base64",
    "show_response_sha256",
    "version_http_status",
    "version_response_url",
    "version_response_content_type",
    "version_response_content_length",
    "version_response_history_count",
    "show_http_status",
    "show_response_url",
    "show_response_content_type",
    "show_response_content_length",
    "show_response_history_count",
    "model_digest",
    "runtime_fingerprint_sha256",
    "runtime_evidence_bytes_base64",
    "runtime_evidence_sha256",
    "transport_mode",
    "trusted_production_transport",
    "network_requests",
    "redirects",
    "retries",
    "pull_attempts",
    "model_calls",
    "receipt_sha256",
}
_REQUEST_HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "identity",
    "Content-Type": "application/json",
}
_PROBE_GET_HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "identity",
}
_OWNED_PROBE_TRANSPORT_MODE = "owned_hardened_loopback_runtime_probe_unattested"
_INJECTED_PROBE_TRANSPORT_MODE = "untrusted_injected_test_runtime_probe_transport"


class SecFilingGemmaOllamaError(RuntimeError):
    """A request, response, or identity violated the local-model contract."""


class ResponseLike(Protocol):
    status_code: int
    headers: Mapping[str, Any]
    url: str
    history: Sequence[Any]

    def iter_content(self, chunk_size: int) -> Sequence[bytes]: ...
    def close(self) -> None: ...


class TransportLike(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> ResponseLike: ...


@dataclass(frozen=True, slots=True)
class PinnedOllamaRuntimeIdentity:
    """Validated identity copied from a separately pinned runtime artifact."""

    schema_version: str
    endpoint: str
    model_name: str
    model_digest: str
    runtime_fingerprint_sha256: str
    evidence_bytes: bytes
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class OllamaRuntimeProbeReceipt:
    """Replayable evidence from the two non-generative runtime observations."""

    schema_version: str
    version_endpoint: str
    show_endpoint: str
    model_name: str
    show_request_bytes: bytes
    show_request_sha256: str
    version_response_bytes: bytes
    version_response_sha256: str
    show_response_bytes: bytes
    show_response_sha256: str
    version_http_status: int
    version_response_url: str
    version_response_content_type: str
    version_response_content_length: int | None
    version_response_history_count: int
    show_http_status: int
    show_response_url: str
    show_response_content_type: str
    show_response_content_length: int | None
    show_response_history_count: int
    model_digest: str
    runtime_fingerprint_sha256: str
    runtime_evidence_bytes: bytes
    runtime_evidence_sha256: str
    transport_mode: str
    trusted_production_transport: bool
    network_requests: int
    redirects: int
    retries: int
    pull_attempts: int
    model_calls: int

    def runtime_evidence(self) -> dict[str, Any]:
        """Return a fresh decoded compact identity envelope."""

        value = _strict_json_bytes(
            self.runtime_evidence_bytes,
            location="sealed runtime probe evidence",
            maximum=MAX_RESPONSE_BYTES,
        )
        if not isinstance(value, dict):
            raise SecFilingGemmaOllamaError(
                "Sealed runtime probe evidence is not an object"
            )
        return value

    def pinned_runtime_identity(self) -> PinnedOllamaRuntimeIdentity:
        """Revalidate and expose the compact identity derived by the probe."""

        return validate_pinned_runtime_identity(
            self.runtime_evidence(),
            expected_evidence_sha256=self.runtime_evidence_sha256,
            expected_model_digest=self.model_digest,
            expected_runtime_fingerprint_sha256=(
                self.runtime_fingerprint_sha256
            ),
        )

    def _manifest_body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version_endpoint": self.version_endpoint,
            "show_endpoint": self.show_endpoint,
            "model_name": self.model_name,
            "show_request_bytes_base64": base64.b64encode(
                self.show_request_bytes
            ).decode("ascii"),
            "show_request_sha256": self.show_request_sha256,
            "version_response_bytes_base64": base64.b64encode(
                self.version_response_bytes
            ).decode("ascii"),
            "version_response_sha256": self.version_response_sha256,
            "show_response_bytes_base64": base64.b64encode(
                self.show_response_bytes
            ).decode("ascii"),
            "show_response_sha256": self.show_response_sha256,
            "version_http_status": self.version_http_status,
            "version_response_url": self.version_response_url,
            "version_response_content_type": self.version_response_content_type,
            "version_response_content_length": (
                self.version_response_content_length
            ),
            "version_response_history_count": self.version_response_history_count,
            "show_http_status": self.show_http_status,
            "show_response_url": self.show_response_url,
            "show_response_content_type": self.show_response_content_type,
            "show_response_content_length": self.show_response_content_length,
            "show_response_history_count": self.show_response_history_count,
            "model_digest": self.model_digest,
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "runtime_evidence_bytes_base64": base64.b64encode(
                self.runtime_evidence_bytes
            ).decode("ascii"),
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "transport_mode": self.transport_mode,
            "trusted_production_transport": self.trusted_production_transport,
            "network_requests": self.network_requests,
            "redirects": self.redirects,
            "retries": self.retries,
            "pull_attempts": self.pull_attempts,
            "model_calls": self.model_calls,
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self._manifest_body())

    def to_manifest(self) -> dict[str, Any]:
        body = self._manifest_body()
        return {**body, "receipt_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class OllamaExtractionReceipt:
    """Immutable byte-level record of one accepted local extraction call."""

    schema_version: str
    endpoint: str
    candidate_sha256: str
    sentence_ids: tuple[str, ...]
    request_bytes: bytes
    request_sha256: str
    response_bytes: bytes
    response_sha256: str
    extractor_output_bytes: bytes
    extractor_output_sha256: str
    extractor_output_canonical_sha256: str
    model_name: str
    model_digest: str
    runtime_fingerprint_sha256: str
    runtime_evidence_sha256: str
    http_status: int
    transport_mode: str
    trusted_production_transport: bool
    network_requests: int
    redirects: int
    retries: int
    pull_attempts: int
    repair_attempts: int
    model_streaming: bool
    model_thinking: bool
    elapsed_nanoseconds: int
    elapsed_role: str

    def extractor_output(self) -> dict[str, Any]:
        """Return a fresh decoded copy; no mutable object is stored in the receipt."""

        value = _strict_json_bytes(
            self.extractor_output_bytes,
            location="sealed extractor output",
            maximum=MAX_RESPONSE_BYTES,
        )
        if not isinstance(value, dict):  # guarded when the receipt is built
            raise SecFilingGemmaOllamaError("Sealed extractor output is not an object")
        return value

    def _manifest_body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "endpoint": self.endpoint,
            "candidate_sha256": self.candidate_sha256,
            "sentence_ids": list(self.sentence_ids),
            "request_bytes_base64": base64.b64encode(self.request_bytes).decode("ascii"),
            "request_sha256": self.request_sha256,
            "response_bytes_base64": base64.b64encode(self.response_bytes).decode("ascii"),
            "response_sha256": self.response_sha256,
            "extractor_output_bytes_base64": base64.b64encode(
                self.extractor_output_bytes
            ).decode("ascii"),
            "extractor_output_sha256": self.extractor_output_sha256,
            "extractor_output_canonical_sha256": (
                self.extractor_output_canonical_sha256
            ),
            "model_name": self.model_name,
            "model_digest": self.model_digest,
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "http_status": self.http_status,
            "transport_mode": self.transport_mode,
            "trusted_production_transport": self.trusted_production_transport,
            "network_requests": self.network_requests,
            "redirects": self.redirects,
            "retries": self.retries,
            "pull_attempts": self.pull_attempts,
            "repair_attempts": self.repair_attempts,
            "model_streaming": self.model_streaming,
            "model_thinking": self.model_thinking,
            "elapsed_nanoseconds": self.elapsed_nanoseconds,
            "elapsed_role": self.elapsed_role,
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self._manifest_body())

    def to_manifest(self) -> dict[str, Any]:
        body = self._manifest_body()
        return {**body, "receipt_sha256": canonical_sha256(body)}


@dataclass(frozen=True, slots=True)
class OllamaModelAttemptReceipt:
    """Byte-level evidence for one completed model call, valid or invalid.

    Transport, HTTP, and outer Ollama-envelope failures still raise because no
    trustworthy completed model attempt exists.  Once a complete assistant
    message is received, its exact bytes are sealed even when the extractor
    payload violates the frozen schema.  Invalid bytes are never exposed as
    semantic output and cannot trigger a retry or repair call.
    """

    schema_version: str
    attempt_status: str
    invalid_reason: str | None
    endpoint: str
    candidate_sha256: str
    sentence_ids: tuple[str, ...]
    request_bytes: bytes
    request_sha256: str
    response_bytes: bytes
    response_sha256: str
    extractor_output_bytes: bytes
    extractor_output_sha256: str
    extractor_output_canonical_sha256: str | None
    model_name: str
    model_digest: str
    runtime_fingerprint_sha256: str
    runtime_evidence_sha256: str
    http_status: int
    response_url: str
    response_content_type: str
    response_content_length: int | None
    response_history_count: int
    transport_mode: str
    trusted_production_transport: bool
    network_requests: int
    redirects: int
    retries: int
    pull_attempts: int
    repair_attempts: int
    model_streaming: bool
    model_thinking: bool
    elapsed_nanoseconds: int
    elapsed_role: str

    def validated_extractor_output(self) -> dict[str, Any]:
        if self.attempt_status != VALID_ATTEMPT_STATUS:
            raise SecFilingGemmaOllamaError(
                "Invalid model-attempt bytes cannot be exposed as semantic output"
            )
        value = _strict_json_bytes(
            self.extractor_output_bytes,
            location="sealed valid extractor output",
            maximum=MAX_RESPONSE_BYTES,
        )
        if not isinstance(value, dict):
            raise SecFilingGemmaOllamaError(
                "Sealed valid extractor output is not an object"
            )
        validate_extractor_output(value, supplied_sentence_ids=self.sentence_ids)
        return value

    def _manifest_body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_status": self.attempt_status,
            "invalid_reason": self.invalid_reason,
            "endpoint": self.endpoint,
            "candidate_sha256": self.candidate_sha256,
            "sentence_ids": list(self.sentence_ids),
            "request_bytes_base64": base64.b64encode(self.request_bytes).decode("ascii"),
            "request_sha256": self.request_sha256,
            "response_bytes_base64": base64.b64encode(self.response_bytes).decode("ascii"),
            "response_sha256": self.response_sha256,
            "extractor_output_bytes_base64": base64.b64encode(
                self.extractor_output_bytes
            ).decode("ascii"),
            "extractor_output_sha256": self.extractor_output_sha256,
            "extractor_output_canonical_sha256": (
                self.extractor_output_canonical_sha256
            ),
            "model_name": self.model_name,
            "model_digest": self.model_digest,
            "runtime_fingerprint_sha256": self.runtime_fingerprint_sha256,
            "runtime_evidence_sha256": self.runtime_evidence_sha256,
            "http_status": self.http_status,
            "response_url": self.response_url,
            "response_content_type": self.response_content_type,
            "response_content_length": self.response_content_length,
            "response_history_count": self.response_history_count,
            "transport_mode": self.transport_mode,
            "trusted_production_transport": self.trusted_production_transport,
            "network_requests": self.network_requests,
            "redirects": self.redirects,
            "retries": self.retries,
            "pull_attempts": self.pull_attempts,
            "repair_attempts": self.repair_attempts,
            "model_streaming": self.model_streaming,
            "model_thinking": self.model_thinking,
            "elapsed_nanoseconds": self.elapsed_nanoseconds,
            "elapsed_role": self.elapsed_role,
        }

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self._manifest_body())

    def to_manifest(self) -> dict[str, Any]:
        body = self._manifest_body()
        return {**body, "receipt_sha256": canonical_sha256(body)}


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaOllamaError(f"{location} must be a lowercase SHA-256")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    if not all(isinstance(key, str) for key in value):
        raise SecFilingGemmaOllamaError(f"{location} keys must be strings")
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaOllamaError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _canonical_json_bytes(value: Any, location: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaOllamaError(f"{location} is not finite JSON") from exc


def _reject_duplicate_pairs(location: str) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    def build(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecFilingGemmaOllamaError(
                    f"{location} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    return build


def _reject_nonfinite(token: str) -> None:
    raise SecFilingGemmaOllamaError(f"JSON contains non-finite token {token!r}")


def _strict_json_bytes(data: bytes, *, location: str, maximum: int) -> Any:
    if not isinstance(data, bytes) or not data:
        raise SecFilingGemmaOllamaError(f"{location} must be non-empty bytes")
    if len(data) > maximum:
        raise SecFilingGemmaOllamaError(f"{location} exceeds its byte ceiling")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SecFilingGemmaOllamaError(f"{location} is not strict UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs(location),
            parse_constant=_reject_nonfinite,
        )
    except SecFilingGemmaOllamaError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SecFilingGemmaOllamaError(f"{location} is malformed JSON") from exc


def validate_pinned_runtime_identity(
    evidence: Mapping[str, Any],
    *,
    expected_evidence_sha256: str,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
) -> PinnedOllamaRuntimeIdentity:
    """Validate an externally pinned identity envelope without doing runtime I/O."""

    if not isinstance(evidence, Mapping):
        raise SecFilingGemmaOllamaError("Runtime identity evidence must be a mapping")
    _expect_exact_keys(evidence, _RUNTIME_IDENTITY_KEYS, "runtime identity evidence")
    expected_evidence = _sha256(
        expected_evidence_sha256, "expected_evidence_sha256"
    )
    expected_digest = _sha256(expected_model_digest, "expected_model_digest")
    expected_fingerprint = _sha256(
        expected_runtime_fingerprint_sha256,
        "expected_runtime_fingerprint_sha256",
    )
    evidence_bytes = _canonical_json_bytes(evidence, "runtime identity evidence")
    evidence_hash = hashlib.sha256(evidence_bytes).hexdigest()
    if not hmac.compare_digest(evidence_hash, expected_evidence):
        raise SecFilingGemmaOllamaError("Runtime identity evidence is not externally pinned")
    if evidence["schema_version"] != RUNTIME_IDENTITY_SCHEMA_VERSION:
        raise SecFilingGemmaOllamaError("Runtime identity schema version changed")
    if evidence["endpoint"] != OLLAMA_ENDPOINT:
        raise SecFilingGemmaOllamaError("Runtime identity endpoint is not exact loopback")
    if evidence["model_name"] != OLLAMA_MODEL:
        raise SecFilingGemmaOllamaError("Runtime identity model name changed")
    digest = _sha256(evidence["model_digest"], "runtime model_digest")
    fingerprint = _sha256(
        evidence["runtime_fingerprint_sha256"],
        "runtime runtime_fingerprint_sha256",
    )
    if not hmac.compare_digest(digest, expected_digest):
        raise SecFilingGemmaOllamaError("Runtime model digest differs from the candidate")
    if not hmac.compare_digest(fingerprint, expected_fingerprint):
        raise SecFilingGemmaOllamaError(
            "Runtime fingerprint differs from the candidate"
        )
    return PinnedOllamaRuntimeIdentity(
        schema_version=RUNTIME_IDENTITY_SCHEMA_VERSION,
        endpoint=OLLAMA_ENDPOINT,
        model_name=OLLAMA_MODEL,
        model_digest=digest,
        runtime_fingerprint_sha256=fingerprint,
        evidence_bytes=evidence_bytes,
        evidence_sha256=evidence_hash,
    )


def build_runtime_identity_guard(
    *,
    before_evidence: Mapping[str, Any],
    after_evidence: Mapping[str, Any],
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
    stage: str,
    model_call_receipt_sha256s: Sequence[str],
) -> dict[str, Any]:
    """Bind identical candidate-authoritative runtime probes around a batch."""

    if stage not in {"development", "intermediate", "final"}:
        raise SecFilingGemmaOllamaError("Runtime guard stage is invalid")
    if not isinstance(before_evidence, Mapping) or not isinstance(
        after_evidence, Mapping
    ):
        raise SecFilingGemmaOllamaError("Runtime guard probes must be mappings")
    before_hash = canonical_sha256(before_evidence)
    after_hash = canonical_sha256(after_evidence)
    before = validate_pinned_runtime_identity(
        before_evidence,
        expected_evidence_sha256=before_hash,
        expected_model_digest=expected_model_digest,
        expected_runtime_fingerprint_sha256=(
            expected_runtime_fingerprint_sha256
        ),
    )
    after = validate_pinned_runtime_identity(
        after_evidence,
        expected_evidence_sha256=after_hash,
        expected_model_digest=expected_model_digest,
        expected_runtime_fingerprint_sha256=(
            expected_runtime_fingerprint_sha256
        ),
    )
    if before.evidence_bytes != after.evidence_bytes or not hmac.compare_digest(
        before.evidence_sha256, after.evidence_sha256
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime identity changed during the extraction batch"
        )
    if (
        isinstance(model_call_receipt_sha256s, (str, bytes))
        or not isinstance(model_call_receipt_sha256s, Sequence)
        or not model_call_receipt_sha256s
    ):
        raise SecFilingGemmaOllamaError(
            "Runtime guard requires the complete nonempty call-receipt sequence"
        )
    receipt_hashes = [
        _sha256(value, f"model_call_receipt_sha256s[{index}]")
        for index, value in enumerate(model_call_receipt_sha256s)
    ]
    if len(receipt_hashes) != len(set(receipt_hashes)):
        raise SecFilingGemmaOllamaError(
            "Runtime guard call-receipt sequence contains a duplicate"
        )
    body = {
        "schema_version": RUNTIME_GUARD_SCHEMA_VERSION,
        "stage": stage,
        "endpoint": OLLAMA_ENDPOINT,
        "model_name": OLLAMA_MODEL,
        "model_digest": before.model_digest,
        "runtime_fingerprint_sha256": before.runtime_fingerprint_sha256,
        "before_runtime_evidence_sha256": before.evidence_sha256,
        "after_runtime_evidence_sha256": after.evidence_sha256,
        "runtime_identity_unchanged": True,
        "probe_order": "immediately_before_then_immediately_after_complete_batch",
        "model_call_count": len(receipt_hashes),
        "model_call_receipt_sha256s": receipt_hashes,
        "model_call_receipt_sequence_sha256": canonical_sha256(receipt_hashes),
    }
    return {**body, "runtime_guard_sha256": canonical_sha256(body)}


def validate_runtime_identity_guard(
    value: Mapping[str, Any],
    *,
    before_evidence: Mapping[str, Any],
    after_evidence: Mapping[str, Any],
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
    stage: str,
    model_call_receipt_sha256s: Sequence[str],
    expected_runtime_guard_sha256: str | None = None,
) -> str:
    expected = build_runtime_identity_guard(
        before_evidence=before_evidence,
        after_evidence=after_evidence,
        expected_model_digest=expected_model_digest,
        expected_runtime_fingerprint_sha256=(
            expected_runtime_fingerprint_sha256
        ),
        stage=stage,
        model_call_receipt_sha256s=model_call_receipt_sha256s,
    )
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise SecFilingGemmaOllamaError("Runtime guard is not canonical")
    observed = expected["runtime_guard_sha256"]
    if expected_runtime_guard_sha256 is not None and not hmac.compare_digest(
        observed,
        _sha256(expected_runtime_guard_sha256, "expected_runtime_guard_sha256"),
    ):
        raise SecFilingGemmaOllamaError("Runtime guard is not externally pinned")
    return observed


def _validate_runtime_identity_object(
    identity: PinnedOllamaRuntimeIdentity,
    *,
    expected_runtime_evidence_sha256: str,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
) -> None:
    if not isinstance(identity, PinnedOllamaRuntimeIdentity):
        raise SecFilingGemmaOllamaError("Pinned runtime identity object is required")
    expected_evidence = {
        "schema_version": identity.schema_version,
        "endpoint": identity.endpoint,
        "model_name": identity.model_name,
        "model_digest": identity.model_digest,
        "runtime_fingerprint_sha256": identity.runtime_fingerprint_sha256,
    }
    expected_bytes = _canonical_json_bytes(expected_evidence, "runtime identity")
    if (
        identity.schema_version != RUNTIME_IDENTITY_SCHEMA_VERSION
        or identity.endpoint != OLLAMA_ENDPOINT
        or identity.model_name != OLLAMA_MODEL
        or identity.evidence_bytes != expected_bytes
        or not hmac.compare_digest(
            hashlib.sha256(expected_bytes).hexdigest(),
            _sha256(identity.evidence_sha256, "runtime evidence_sha256"),
        )
    ):
        raise SecFilingGemmaOllamaError("Pinned runtime identity object is inconsistent")
    _sha256(identity.model_digest, "runtime model_digest")
    _sha256(identity.runtime_fingerprint_sha256, "runtime fingerprint")
    if not hmac.compare_digest(
        identity.evidence_sha256,
        _sha256(
            expected_runtime_evidence_sha256,
            "expected_runtime_evidence_sha256",
        ),
    ):
        raise SecFilingGemmaOllamaError(
            "Runtime identity evidence differs from the candidate pin"
        )
    if not hmac.compare_digest(
        identity.model_digest,
        _sha256(expected_model_digest, "expected_model_digest"),
    ):
        raise SecFilingGemmaOllamaError(
            "Runtime model digest differs from the candidate pin"
        )
    if not hmac.compare_digest(
        identity.runtime_fingerprint_sha256,
        _sha256(
            expected_runtime_fingerprint_sha256,
            "expected_runtime_fingerprint_sha256",
        ),
    ):
        raise SecFilingGemmaOllamaError(
            "Runtime fingerprint differs from the candidate pin"
        )


def _validated_payload(
    validated_request: Mapping[str, Any],
    *,
    expected_candidate_sha256: str,
    expected_model_payload_sha256: str,
) -> tuple[str, tuple[str, ...], bytes, str]:
    if not isinstance(validated_request, Mapping):
        raise SecFilingGemmaOllamaError("Validated extractor request must be a mapping")
    _expect_exact_keys(
        validated_request, _VALIDATED_REQUEST_KEYS, "validated extractor request"
    )
    candidate_hash = _sha256(
        expected_candidate_sha256, "expected_candidate_sha256"
    )
    payload_hash = _sha256(
        expected_model_payload_sha256, "expected_model_payload_sha256"
    )
    if validated_request["candidate_sha256"] != candidate_hash:
        raise SecFilingGemmaOllamaError("Validated request candidate binding changed")
    supplied_payload_hash = _sha256(
        validated_request["model_payload_sha256"],
        "validated request model_payload_sha256",
    )
    if not hmac.compare_digest(supplied_payload_hash, payload_hash):
        raise SecFilingGemmaOllamaError("Validated request payload is not externally pinned")

    raw_ids = validated_request["sentence_ids"]
    # The contract validator returns a tuple.  Requiring that exact type keeps
    # a merely deserialized or hand-built lookalike from crossing this boundary
    # without first being revalidated by the contract.
    if not isinstance(raw_ids, tuple):
        raise SecFilingGemmaOllamaError("Validated request sentence_ids are invalid")
    sentence_ids = tuple(raw_ids)
    if (
        not sentence_ids
        or len(sentence_ids) != len(set(sentence_ids))
        or any(
            not isinstance(item, str) or _SENTENCE_ID_RE.fullmatch(item) is None
            for item in sentence_ids
        )
    ):
        raise SecFilingGemmaOllamaError("Validated request sentence_ids are invalid")

    model_payload = validated_request["model_payload"]
    if not isinstance(model_payload, Mapping):
        raise SecFilingGemmaOllamaError("Validated request model payload is invalid")
    try:
        messages = model_payload["messages"]
        user_content = messages[1]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SecFilingGemmaOllamaError("Validated model payload shape changed") from exc
    if not isinstance(user_content, str):
        raise SecFilingGemmaOllamaError("Validated model user content is not text")
    decoded_user = _strict_json_bytes(
        user_content.encode("utf-8"),
        location="validated model user content",
        maximum=MAX_RESPONSE_BYTES,
    )
    if not isinstance(decoded_user, dict) or set(decoded_user) != {"sentences"}:
        raise SecFilingGemmaOllamaError("Validated model user content shape changed")
    sentences = decoded_user["sentences"]
    if not isinstance(sentences, list):
        raise SecFilingGemmaOllamaError("Validated model sentences are not a list")
    try:
        expected_payload = build_extractor_model_payload(sentences)
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaOllamaError("Validated model payload cannot be rebuilt") from exc
    if model_payload != expected_payload:
        raise SecFilingGemmaOllamaError("Model payload differs from the frozen safe payload")
    sentence_ids_from_payload = tuple(
        sentence.get("id") if isinstance(sentence, Mapping) else None
        for sentence in sentences
    )
    if sentence_ids_from_payload != sentence_ids:
        raise SecFilingGemmaOllamaError(
            "Validated sentence ids do not match the exact model payload"
        )
    request_bytes = _canonical_json_bytes(expected_payload, "model request payload")
    observed_hash = hashlib.sha256(request_bytes).hexdigest()
    if not hmac.compare_digest(observed_hash, payload_hash):
        raise SecFilingGemmaOllamaError("Canonical model request bytes changed")
    return candidate_hash, sentence_ids, request_bytes, observed_hash


def _strict_nonnegative_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SecFilingGemmaOllamaError(f"{location} must be a nonnegative integer")
    if value > 2**63 - 1:
        raise SecFilingGemmaOllamaError(f"{location} exceeds the integer ceiling")
    return value


def _validate_created_at(value: Any) -> None:
    if not isinstance(value, str) or _RFC3339_UTC_RE.fullmatch(value) is None:
        raise SecFilingGemmaOllamaError("Ollama created_at is not canonical UTC")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SecFilingGemmaOllamaError("Ollama created_at is invalid") from exc


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    target = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == target:
            return str(value)
    return None


def _validate_http_envelope(
    response: ResponseLike,
    response_bytes: bytes,
    *,
    expected_endpoint: str = OLLAMA_ENDPOINT,
) -> dict[str, Any]:
    status = getattr(response, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        raise SecFilingGemmaOllamaError("Ollama status code is invalid")
    if 300 <= status < 400:
        raise SecFilingGemmaOllamaError("Ollama redirects are forbidden")
    if status != 200:
        raise SecFilingGemmaOllamaError("Ollama returned a non-success status")
    response_url = getattr(response, "url", None)
    if type(response_url) is not str or response_url != expected_endpoint:
        raise SecFilingGemmaOllamaError("Ollama response URL changed")
    history = getattr(response, "history", None)
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise SecFilingGemmaOllamaError("Ollama redirect history is invalid")
    history_count = len(history)
    if history_count != 0:
        raise SecFilingGemmaOllamaError("Ollama redirects are forbidden")
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        raise SecFilingGemmaOllamaError("Ollama response headers are invalid")
    content_type = (
        (_header(headers, "Content-Type") or "").split(";", 1)[0].strip().casefold()
    )
    if content_type != "application/json":
        raise SecFilingGemmaOllamaError("Ollama response is not application/json")
    content_length = _header(headers, "Content-Length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdecimal():
            raise SecFilingGemmaOllamaError("Ollama Content-Length is invalid")
        if int(content_length) != len(response_bytes):
            raise SecFilingGemmaOllamaError("Ollama Content-Length does not reconcile")
    return {
        "http_status": status,
        "response_url": response_url,
        "response_content_type": content_type,
        "response_content_length": (
            None if content_length is None else int(content_length)
        ),
        "response_history_count": history_count,
    }


def _read_bounded_response(response: ResponseLike) -> bytes:
    """Read the HTTP body without ever buffering beyond the frozen ceiling."""

    iterator = getattr(response, "iter_content", None)
    if not callable(iterator):
        raise SecFilingGemmaOllamaError(
            "Ollama response does not provide bounded streaming reads"
        )
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in iterator(chunk_size=64 * 1024):
            if not isinstance(chunk, bytes):
                raise SecFilingGemmaOllamaError(
                    "Ollama response yielded non-byte content"
                )
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise SecFilingGemmaOllamaError(
                    "Ollama response exceeds its byte ceiling"
                )
            chunks.append(chunk)
    except SecFilingGemmaOllamaError:
        raise
    except Exception:
        raise SecFilingGemmaOllamaError(
            "Ollama response bytes are unreadable"
        ) from None
    payload = b"".join(chunks)
    if not payload:
        raise SecFilingGemmaOllamaError("Ollama response body is empty")
    return payload


def _decode_manifest_bytes(value: Any, *, location: str) -> bytes:
    if not isinstance(value, str) or not value or not value.isascii():
        raise SecFilingGemmaOllamaError(f"{location} must be canonical base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        raise SecFilingGemmaOllamaError(
            f"{location} must be canonical base64 text"
        ) from None
    if (
        not decoded
        or len(decoded) > MAX_RESPONSE_BYTES
        or base64.b64encode(decoded).decode("ascii") != value
    ):
        raise SecFilingGemmaOllamaError(
            f"{location} is empty, oversized, or noncanonical"
        )
    return decoded


def _extract_ollama_attempt_output_bytes(
    response_bytes: bytes,
) -> tuple[bytes, bool]:
    """Return exact assistant bytes and whether completion was a normal stop."""

    response = _strict_json_bytes(
        response_bytes,
        location="Ollama response",
        maximum=MAX_RESPONSE_BYTES,
    )
    if not isinstance(response, Mapping):
        raise SecFilingGemmaOllamaError("Ollama response must be an object")
    _expect_exact_keys(response, _OLLAMA_RESPONSE_KEYS, "Ollama response")
    if response["model"] != OLLAMA_MODEL:
        raise SecFilingGemmaOllamaError("Ollama response model identity changed")
    _validate_created_at(response["created_at"])
    if response["done"] is not True:
        raise SecFilingGemmaOllamaError("Ollama response is not complete")
    done_reason = response["done_reason"]
    if type(done_reason) is not str or not done_reason:
        raise SecFilingGemmaOllamaError("Ollama done_reason is invalid")
    normal_completion = done_reason == "stop"
    for field in (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        _strict_nonnegative_int(response[field], f"Ollama response {field}")
    message = response["message"]
    if not isinstance(message, Mapping):
        raise SecFilingGemmaOllamaError("Ollama response message must be an object")
    _expect_exact_keys(message, _OLLAMA_MESSAGE_KEYS, "Ollama response message")
    if message["role"] != "assistant" or not isinstance(message["content"], str):
        raise SecFilingGemmaOllamaError("Ollama response message changed")
    try:
        output_bytes = message["content"].encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise SecFilingGemmaOllamaError(
            "Ollama extractor output is not valid Unicode text"
        ) from exc
    if not output_bytes or len(output_bytes) > MAX_RESPONSE_BYTES:
        raise SecFilingGemmaOllamaError(
            "Ollama extractor output is empty or oversized"
        )
    return output_bytes, normal_completion


def _extract_ollama_output_bytes(response_bytes: bytes) -> bytes:
    """Validate one normally completed response for the valid-only API."""

    output_bytes, normal_completion = _extract_ollama_attempt_output_bytes(
        response_bytes
    )
    if not normal_completion:
        raise SecFilingGemmaOllamaError(
            "Ollama response was truncated or stopped abnormally"
        )
    return output_bytes


def _validate_extractor_output_bytes(
    output_bytes: bytes, *, supplied_sentence_ids: tuple[str, ...]
) -> tuple[str, str]:
    output = _strict_json_bytes(
        output_bytes,
        location="Ollama extractor output",
        maximum=MAX_RESPONSE_BYTES,
    )
    if not isinstance(output, Mapping):
        raise SecFilingGemmaOllamaError("Ollama extractor output must be an object")
    try:
        validate_extractor_output(
            output, supplied_sentence_ids=supplied_sentence_ids
        )
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaOllamaError("Ollama extractor output violates its schema") from exc
    return hashlib.sha256(output_bytes).hexdigest(), canonical_sha256(output)


def _validate_ollama_response(
    response_bytes: bytes, *, supplied_sentence_ids: tuple[str, ...]
) -> tuple[bytes, str, str]:
    output_bytes = _extract_ollama_output_bytes(response_bytes)
    output_hash, canonical_hash = _validate_extractor_output_bytes(
        output_bytes,
        supplied_sentence_ids=supplied_sentence_ids,
    )
    return output_bytes, output_hash, canonical_hash


def validate_ollama_extraction_receipt(
    manifest: Mapping[str, Any],
    *,
    expected_candidate_sha256: str,
    expected_model_payload_sha256: str,
    expected_sentence_ids: Sequence[str],
    expected_runtime_evidence_sha256: str,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
    expected_transport_mode: str,
) -> OllamaExtractionReceipt:
    """Replay a persisted call receipt against candidate-authoritative pins."""

    if type(manifest) is not dict:
        raise SecFilingGemmaOllamaError("Ollama receipt manifest must be a mapping")
    manifest = dict(manifest)
    if type(manifest.get("sentence_ids")) is not list:
        raise SecFilingGemmaOllamaError(
            "Ollama receipt sentence IDs must be a detached list"
        )
    body_keys = set(OllamaExtractionReceipt.__dataclass_fields__) - {
        "request_bytes",
        "response_bytes",
        "extractor_output_bytes",
    }
    body_keys |= {
        "request_bytes_base64",
        "response_bytes_base64",
        "extractor_output_bytes_base64",
    }
    expected_keys = body_keys | {"receipt_sha256"}
    _expect_exact_keys(manifest, expected_keys, "Ollama receipt manifest")
    body = {key: manifest[key] for key in manifest if key != "receipt_sha256"}
    observed_receipt_hash = _sha256(
        manifest["receipt_sha256"], "receipt_sha256"
    )
    if not hmac.compare_digest(canonical_sha256(body), observed_receipt_hash):
        raise SecFilingGemmaOllamaError("Ollama receipt hash is not canonical")

    request_bytes = _decode_manifest_bytes(
        manifest["request_bytes_base64"], location="request_bytes_base64"
    )
    response_bytes = _decode_manifest_bytes(
        manifest["response_bytes_base64"], location="response_bytes_base64"
    )
    output_bytes = _decode_manifest_bytes(
        manifest["extractor_output_bytes_base64"],
        location="extractor_output_bytes_base64",
    )
    if manifest["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise SecFilingGemmaOllamaError("Ollama receipt schema changed")
    if manifest["endpoint"] != OLLAMA_ENDPOINT:
        raise SecFilingGemmaOllamaError("Ollama receipt endpoint changed")
    candidate_hash = _sha256(
        expected_candidate_sha256, "expected_candidate_sha256"
    )
    if manifest["candidate_sha256"] != candidate_hash:
        raise SecFilingGemmaOllamaError("Ollama receipt candidate binding changed")
    if (
        isinstance(expected_sentence_ids, (str, bytes))
        or not isinstance(expected_sentence_ids, Sequence)
    ):
        raise SecFilingGemmaOllamaError("Expected sentence IDs must be a sequence")
    sentence_ids = tuple(expected_sentence_ids)
    if (
        list(sentence_ids) != manifest["sentence_ids"]
        or not sentence_ids
        or len(sentence_ids) != len(set(sentence_ids))
        or any(
            not isinstance(item, str) or _SENTENCE_ID_RE.fullmatch(item) is None
            for item in sentence_ids
        )
    ):
        raise SecFilingGemmaOllamaError("Ollama receipt sentence IDs changed")

    request_hash = hashlib.sha256(request_bytes).hexdigest()
    expected_payload_hash = _sha256(
        expected_model_payload_sha256, "expected_model_payload_sha256"
    )
    if (
        manifest["request_sha256"] != request_hash
        or not hmac.compare_digest(request_hash, expected_payload_hash)
    ):
        raise SecFilingGemmaOllamaError("Ollama receipt request bytes changed")
    request_payload = _strict_json_bytes(
        request_bytes,
        location="sealed Ollama request",
        maximum=MAX_RESPONSE_BYTES,
    )
    if not isinstance(request_payload, Mapping):
        raise SecFilingGemmaOllamaError("Sealed Ollama request must be an object")
    try:
        user_content = request_payload["messages"][1]["content"]
    except (KeyError, IndexError, TypeError):
        raise SecFilingGemmaOllamaError("Sealed Ollama request shape changed") from None
    if not isinstance(user_content, str):
        raise SecFilingGemmaOllamaError("Sealed Ollama request content changed")
    user_payload = _strict_json_bytes(
        user_content.encode("utf-8"),
        location="sealed Ollama user content",
        maximum=MAX_RESPONSE_BYTES,
    )
    if not isinstance(user_payload, Mapping) or set(user_payload) != {"sentences"}:
        raise SecFilingGemmaOllamaError("Sealed Ollama user content shape changed")
    try:
        rebuilt_payload = build_extractor_model_payload(user_payload["sentences"])
    except (SecFilingGemmaContractError, TypeError, ValueError):
        raise SecFilingGemmaOllamaError("Sealed Ollama request cannot be rebuilt") from None
    if request_payload != rebuilt_payload or request_bytes != _canonical_json_bytes(
        rebuilt_payload, "sealed Ollama request"
    ):
        raise SecFilingGemmaOllamaError("Sealed Ollama request is not canonical")
    request_sentence_ids = tuple(
        item.get("id") if isinstance(item, Mapping) else None
        for item in user_payload["sentences"]
    )
    if request_sentence_ids != sentence_ids:
        raise SecFilingGemmaOllamaError("Sealed request sentence IDs changed")

    response_output, output_hash, output_canonical_hash = _validate_ollama_response(
        response_bytes, supplied_sentence_ids=sentence_ids
    )
    if (
        manifest["response_sha256"] != hashlib.sha256(response_bytes).hexdigest()
        or response_output != output_bytes
        or manifest["extractor_output_sha256"] != output_hash
        or manifest["extractor_output_canonical_sha256"]
        != output_canonical_hash
    ):
        raise SecFilingGemmaOllamaError("Ollama receipt response or output bytes changed")
    for field, expected in (
        ("model_name", OLLAMA_MODEL),
        ("model_digest", _sha256(expected_model_digest, "expected_model_digest")),
        (
            "runtime_fingerprint_sha256",
            _sha256(
                expected_runtime_fingerprint_sha256,
                "expected_runtime_fingerprint_sha256",
            ),
        ),
        (
            "runtime_evidence_sha256",
            _sha256(
                expected_runtime_evidence_sha256,
                "expected_runtime_evidence_sha256",
            ),
        ),
        ("elapsed_role", DIAGNOSTIC_TIME_ROLE),
    ):
        if manifest[field] != expected:
            raise SecFilingGemmaOllamaError(f"Ollama receipt {field} changed")
    for field, expected in (
        ("http_status", 200),
        ("network_requests", 1),
        ("redirects", 0),
        ("retries", 0),
        ("pull_attempts", 0),
        ("repair_attempts", 0),
    ):
        if _strict_nonnegative_int(manifest[field], field) != expected:
            raise SecFilingGemmaOllamaError(f"Ollama receipt {field} changed")
    for field in (
        "trusted_production_transport",
        "model_streaming",
        "model_thinking",
    ):
        if manifest[field] is not False:
            raise SecFilingGemmaOllamaError(f"Ollama receipt {field} changed")
    permitted_transport_modes = {
        "untrusted_injected_test_transport",
        "owned_hardened_loopback_session_requires_stage_attestation",
    }
    if (
        expected_transport_mode not in permitted_transport_modes
        or manifest["transport_mode"] != expected_transport_mode
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama receipt transport mode changed or is not externally pinned"
        )
    elapsed = _strict_nonnegative_int(
        manifest["elapsed_nanoseconds"], "elapsed_nanoseconds"
    )
    receipt = OllamaExtractionReceipt(
        schema_version=manifest["schema_version"],
        endpoint=manifest["endpoint"],
        candidate_sha256=manifest["candidate_sha256"],
        sentence_ids=sentence_ids,
        request_bytes=request_bytes,
        request_sha256=manifest["request_sha256"],
        response_bytes=response_bytes,
        response_sha256=manifest["response_sha256"],
        extractor_output_bytes=output_bytes,
        extractor_output_sha256=manifest["extractor_output_sha256"],
        extractor_output_canonical_sha256=manifest[
            "extractor_output_canonical_sha256"
        ],
        model_name=manifest["model_name"],
        model_digest=manifest["model_digest"],
        runtime_fingerprint_sha256=manifest["runtime_fingerprint_sha256"],
        runtime_evidence_sha256=manifest["runtime_evidence_sha256"],
        http_status=manifest["http_status"],
        transport_mode=manifest["transport_mode"],
        trusted_production_transport=False,
        network_requests=manifest["network_requests"],
        redirects=manifest["redirects"],
        retries=manifest["retries"],
        pull_attempts=manifest["pull_attempts"],
        repair_attempts=manifest["repair_attempts"],
        model_streaming=manifest["model_streaming"],
        model_thinking=manifest["model_thinking"],
        elapsed_nanoseconds=elapsed,
        elapsed_role=manifest["elapsed_role"],
    )
    if receipt.to_manifest() != dict(manifest):
        raise SecFilingGemmaOllamaError("Ollama receipt manifest is not canonical")
    return receipt


def validate_ollama_model_attempt_receipt(
    manifest: Mapping[str, Any],
    *,
    expected_candidate_sha256: str,
    expected_model_payload_sha256: str,
    expected_sentence_ids: Sequence[str],
    expected_runtime_evidence_sha256: str,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
    expected_transport_mode: str,
) -> OllamaModelAttemptReceipt:
    """Replay one persisted valid-or-invalid model attempt from exact bytes."""

    if type(manifest) is not dict:
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt receipt must be a mapping"
        )
    manifest = dict(manifest)
    if type(manifest.get("sentence_ids")) is not list:
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt sentence IDs must be a detached list"
        )
    body_keys = set(OllamaModelAttemptReceipt.__dataclass_fields__) - {
        "request_bytes",
        "response_bytes",
        "extractor_output_bytes",
    }
    body_keys |= {
        "request_bytes_base64",
        "response_bytes_base64",
        "extractor_output_bytes_base64",
    }
    _expect_exact_keys(
        manifest,
        body_keys | {"receipt_sha256"},
        "Ollama model-attempt receipt",
    )
    body = {key: manifest[key] for key in manifest if key != "receipt_sha256"}
    receipt_hash = _sha256(manifest["receipt_sha256"], "receipt_sha256")
    if not hmac.compare_digest(canonical_sha256(body), receipt_hash):
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt receipt hash is not canonical"
        )
    if manifest["schema_version"] != ATTEMPT_RECEIPT_SCHEMA_VERSION:
        raise SecFilingGemmaOllamaError("Ollama model-attempt schema changed")
    status = manifest["attempt_status"]
    if status not in {VALID_ATTEMPT_STATUS, INVALID_ATTEMPT_STATUS}:
        raise SecFilingGemmaOllamaError("Ollama model-attempt status changed")
    reason = manifest["invalid_reason"]
    if (
        status == VALID_ATTEMPT_STATUS
        and reason is not None
    ) or (
        status == INVALID_ATTEMPT_STATUS
        and reason not in {INVALID_ATTEMPT_REASON, ABNORMAL_ATTEMPT_REASON}
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt status and invalid reason disagree"
        )

    request_bytes = _decode_manifest_bytes(
        manifest["request_bytes_base64"], location="request_bytes_base64"
    )
    response_bytes = _decode_manifest_bytes(
        manifest["response_bytes_base64"], location="response_bytes_base64"
    )
    output_bytes = _decode_manifest_bytes(
        manifest["extractor_output_bytes_base64"],
        location="extractor_output_bytes_base64",
    )
    request_payload = _strict_json_bytes(
        request_bytes,
        location="sealed Ollama attempt request",
        maximum=MAX_RESPONSE_BYTES,
    )
    if not isinstance(request_payload, Mapping):
        raise SecFilingGemmaOllamaError(
            "Sealed Ollama attempt request must be an object"
        )
    if (
        isinstance(expected_sentence_ids, (str, bytes))
        or not isinstance(expected_sentence_ids, Sequence)
    ):
        raise SecFilingGemmaOllamaError("Expected sentence IDs must be a sequence")
    sentence_ids = tuple(expected_sentence_ids)
    candidate_hash, normalized_ids, rebuilt_request, request_hash = _validated_payload(
        {
            "candidate_sha256": expected_candidate_sha256,
            "sentence_ids": sentence_ids,
            "model_payload": request_payload,
            "model_payload_sha256": expected_model_payload_sha256,
        },
        expected_candidate_sha256=expected_candidate_sha256,
        expected_model_payload_sha256=expected_model_payload_sha256,
    )
    if (
        request_bytes != rebuilt_request
        or manifest["request_sha256"] != request_hash
        or manifest["candidate_sha256"] != candidate_hash
        or manifest["sentence_ids"] != list(normalized_ids)
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt request or event binding changed"
        )

    response_output, normal_completion = _extract_ollama_attempt_output_bytes(
        response_bytes
    )
    response_hash = hashlib.sha256(response_bytes).hexdigest()
    output_hash = hashlib.sha256(output_bytes).hexdigest()
    if (
        response_output != output_bytes
        or manifest["response_sha256"] != response_hash
        or manifest["extractor_output_sha256"] != output_hash
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt response or output bytes changed"
        )
    if not normal_completion:
        output_is_valid = False
        canonical_output_hash = None
        expected_reason = ABNORMAL_ATTEMPT_REASON
    else:
        try:
            validated_hash, canonical_output_hash = _validate_extractor_output_bytes(
                output_bytes,
                supplied_sentence_ids=normalized_ids,
            )
        except SecFilingGemmaOllamaError:
            output_is_valid = False
            canonical_output_hash = None
            expected_reason = INVALID_ATTEMPT_REASON
        else:
            output_is_valid = validated_hash == output_hash
            expected_reason = None
    if output_is_valid != (status == VALID_ATTEMPT_STATUS):
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt status does not match the exact output bytes"
        )
    if reason != expected_reason:
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt invalid reason does not match completion evidence"
        )
    observed_canonical_hash = manifest["extractor_output_canonical_sha256"]
    if observed_canonical_hash is not None:
        observed_canonical_hash = _sha256(
            observed_canonical_hash,
            "extractor_output_canonical_sha256",
        )
    if observed_canonical_hash != canonical_output_hash:
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt canonical output identity changed"
        )

    for field, expected in (
        ("endpoint", OLLAMA_ENDPOINT),
        ("response_url", OLLAMA_ENDPOINT),
        ("response_content_type", "application/json"),
        ("model_name", OLLAMA_MODEL),
        ("model_digest", _sha256(expected_model_digest, "expected_model_digest")),
        (
            "runtime_fingerprint_sha256",
            _sha256(
                expected_runtime_fingerprint_sha256,
                "expected_runtime_fingerprint_sha256",
            ),
        ),
        (
            "runtime_evidence_sha256",
            _sha256(
                expected_runtime_evidence_sha256,
                "expected_runtime_evidence_sha256",
            ),
        ),
        ("elapsed_role", DIAGNOSTIC_TIME_ROLE),
    ):
        if manifest[field] != expected:
            raise SecFilingGemmaOllamaError(
                f"Ollama model-attempt {field} changed"
            )
    for field, expected in (
        ("http_status", 200),
        ("response_history_count", 0),
        ("network_requests", 1),
        ("redirects", 0),
        ("retries", 0),
        ("pull_attempts", 0),
        ("repair_attempts", 0),
    ):
        if _strict_nonnegative_int(manifest[field], field) != expected:
            raise SecFilingGemmaOllamaError(
                f"Ollama model-attempt {field} changed"
            )
    response_content_length = manifest["response_content_length"]
    if response_content_length is not None:
        if (
            _strict_nonnegative_int(
                response_content_length, "response_content_length"
            )
            != len(response_bytes)
        ):
            raise SecFilingGemmaOllamaError(
                "Ollama model-attempt response_content_length changed"
            )
    for field in (
        "trusted_production_transport",
        "model_streaming",
        "model_thinking",
    ):
        if manifest[field] is not False:
            raise SecFilingGemmaOllamaError(
                f"Ollama model-attempt {field} changed"
            )
    permitted_transport_modes = {
        "untrusted_injected_test_transport",
        "owned_hardened_loopback_session_requires_stage_attestation",
    }
    if (
        expected_transport_mode not in permitted_transport_modes
        or manifest["transport_mode"] != expected_transport_mode
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt transport mode changed or is not externally pinned"
        )
    elapsed = _strict_nonnegative_int(
        manifest["elapsed_nanoseconds"], "elapsed_nanoseconds"
    )
    receipt = OllamaModelAttemptReceipt(
        schema_version=manifest["schema_version"],
        attempt_status=status,
        invalid_reason=reason,
        endpoint=manifest["endpoint"],
        candidate_sha256=manifest["candidate_sha256"],
        sentence_ids=normalized_ids,
        request_bytes=request_bytes,
        request_sha256=manifest["request_sha256"],
        response_bytes=response_bytes,
        response_sha256=manifest["response_sha256"],
        extractor_output_bytes=output_bytes,
        extractor_output_sha256=manifest["extractor_output_sha256"],
        extractor_output_canonical_sha256=observed_canonical_hash,
        model_name=manifest["model_name"],
        model_digest=manifest["model_digest"],
        runtime_fingerprint_sha256=manifest["runtime_fingerprint_sha256"],
        runtime_evidence_sha256=manifest["runtime_evidence_sha256"],
        http_status=manifest["http_status"],
        response_url=manifest["response_url"],
        response_content_type=manifest["response_content_type"],
        response_content_length=response_content_length,
        response_history_count=manifest["response_history_count"],
        transport_mode=manifest["transport_mode"],
        trusted_production_transport=False,
        network_requests=manifest["network_requests"],
        redirects=manifest["redirects"],
        retries=manifest["retries"],
        pull_attempts=manifest["pull_attempts"],
        repair_attempts=manifest["repair_attempts"],
        model_streaming=manifest["model_streaming"],
        model_thinking=manifest["model_thinking"],
        elapsed_nanoseconds=elapsed,
        elapsed_role=manifest["elapsed_role"],
    )
    if receipt.to_manifest() != dict(manifest):
        raise SecFilingGemmaOllamaError(
            "Ollama model-attempt receipt is not canonical"
        )
    return receipt


def _runtime_probe_show_request_bytes() -> bytes:
    return _canonical_json_bytes(
        {"model": OLLAMA_MODEL, "verbose": False},
        "Ollama show request",
    )


def _extract_runtime_probe_model_digest(show: Mapping[str, Any]) -> str:
    required = {
        "modelfile",
        "parameters",
        "template",
        "details",
        "model_info",
        "capabilities",
        "modified_at",
    }
    missing = required - set(show)
    if missing:
        raise SecFilingGemmaOllamaError(
            f"Ollama show response is missing required runtime fields: {sorted(missing)}"
        )
    for field in ("modelfile", "parameters", "template", "modified_at"):
        if not isinstance(show[field], str):
            raise SecFilingGemmaOllamaError(
                f"Ollama show response {field} must be text"
            )
    if not show["modelfile"]:
        raise SecFilingGemmaOllamaError("Ollama show response modelfile is empty")
    if not isinstance(show["details"], Mapping) or not isinstance(
        show["model_info"], Mapping
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama show response details and model_info must be objects"
        )
    capabilities = show["capabilities"]
    if (
        not isinstance(capabilities, list)
        or not all(isinstance(value, str) and value for value in capabilities)
        or len(capabilities) != len(set(capabilities))
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama show response capabilities are invalid"
        )

    lines = [line.strip() for line in show["modelfile"].splitlines()]
    expected_name_marker = f"# FROM {OLLAMA_MODEL}"
    if lines.count(expected_name_marker) != 1:
        raise SecFilingGemmaOllamaError(
            "Ollama show response does not identify the exact fixed model"
        )
    from_lines = [
        line[5:].strip()
        for line in lines
        if line.startswith("FROM ") and not line.startswith("#")
    ]
    if len(from_lines) != 1:
        raise SecFilingGemmaOllamaError(
            "Ollama show response must contain one active model FROM directive"
        )
    target = from_lines[0]
    if len(target) >= 2 and target[0] == target[-1] == '"':
        target = target[1:-1]
    match = _MODEL_BLOB_FROM_RE.search(target)
    if match is None:
        raise SecFilingGemmaOllamaError(
            "Ollama show response does not expose an exact model blob digest"
        )
    return match.group(1)


def _derive_runtime_probe_identity(
    *,
    version_response_bytes: bytes,
    show_response_bytes: bytes,
    expected_model_digest: str,
) -> PinnedOllamaRuntimeIdentity:
    expected_digest = _sha256(expected_model_digest, "expected_model_digest")
    version = _strict_json_bytes(
        version_response_bytes,
        location="Ollama version response",
        maximum=MAX_RESPONSE_BYTES,
    )
    if not isinstance(version, Mapping):
        raise SecFilingGemmaOllamaError("Ollama version response must be an object")
    _expect_exact_keys(version, {"version"}, "Ollama version response")
    if (
        not isinstance(version["version"], str)
        or _OLLAMA_VERSION_RE.fullmatch(version["version"]) is None
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama version response is not a bounded semantic version"
        )
    show = _strict_json_bytes(
        show_response_bytes,
        location="Ollama show response",
        maximum=MAX_RESPONSE_BYTES,
    )
    if not isinstance(show, Mapping):
        raise SecFilingGemmaOllamaError("Ollama show response must be an object")
    observed_digest = _extract_runtime_probe_model_digest(show)
    if not hmac.compare_digest(observed_digest, expected_digest):
        raise SecFilingGemmaOllamaError(
            "Ollama installed model digest differs from the candidate"
        )
    fingerprint_material = {
        "schema_version": RUNTIME_FINGERPRINT_SCHEMA_VERSION,
        "chat_endpoint": OLLAMA_ENDPOINT,
        "version_endpoint": OLLAMA_VERSION_ENDPOINT,
        "show_endpoint": OLLAMA_SHOW_ENDPOINT,
        "model_name": OLLAMA_MODEL,
        "model_digest": observed_digest,
        "version_response": dict(version),
        "show_response": dict(show),
    }
    fingerprint = canonical_sha256(fingerprint_material)
    evidence = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "endpoint": OLLAMA_ENDPOINT,
        "model_name": OLLAMA_MODEL,
        "model_digest": observed_digest,
        "runtime_fingerprint_sha256": fingerprint,
    }
    return validate_pinned_runtime_identity(
        evidence,
        expected_evidence_sha256=canonical_sha256(evidence),
        expected_model_digest=expected_digest,
        expected_runtime_fingerprint_sha256=fingerprint,
    )


def _perform_runtime_probe_request(
    transport: TransportLike,
    *,
    method: str,
    endpoint: str,
    headers: Mapping[str, str],
    request_bytes: bytes | None,
) -> tuple[bytes, dict[str, Any]]:
    response: ResponseLike | None = None
    try:
        kwargs: dict[str, Any] = {
            "headers": dict(headers),
            "timeout": (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            "allow_redirects": False,
            "stream": True,
        }
        if request_bytes is not None:
            kwargs["data"] = request_bytes
        try:
            response = transport.request(method, endpoint, **kwargs)
        except Exception:
            raise SecFilingGemmaOllamaError(
                "Ollama loopback runtime probe failed"
            ) from None
        response_bytes = _read_bounded_response(response)
        http_evidence = _validate_http_envelope(
            response,
            response_bytes,
            expected_endpoint=endpoint,
        )
        return response_bytes, http_evidence
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def probe_owned_ollama_runtime(
    *,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
    transport: TransportLike | None = None,
) -> OllamaRuntimeProbeReceipt:
    """Observe the fixed local runtime without model I/O or mutable authority.

    Omitting ``transport`` creates the hardened, environment-independent
    production session.  Injection exists only for deterministic offline tests
    and is explicitly sealed as untrusted in the receipt.
    """

    expected_digest = _sha256(expected_model_digest, "expected_model_digest")
    expected_fingerprint = _sha256(
        expected_runtime_fingerprint_sha256,
        "expected_runtime_fingerprint_sha256",
    )
    show_request_bytes = _runtime_probe_show_request_bytes()
    owned_transport = transport is None
    active_transport: TransportLike = (
        build_hardened_loopback_session() if transport is None else transport
    )
    try:
        if not hasattr(active_transport, "request"):
            raise SecFilingGemmaOllamaError(
                "Ollama runtime-probe transport must provide request()"
            )
        version_bytes, version_http = _perform_runtime_probe_request(
            active_transport,
            method="GET",
            endpoint=OLLAMA_VERSION_ENDPOINT,
            headers=_PROBE_GET_HEADERS,
            request_bytes=None,
        )
        show_bytes, show_http = _perform_runtime_probe_request(
            active_transport,
            method="POST",
            endpoint=OLLAMA_SHOW_ENDPOINT,
            headers=_REQUEST_HEADERS,
            request_bytes=show_request_bytes,
        )
        identity = _derive_runtime_probe_identity(
            version_response_bytes=version_bytes,
            show_response_bytes=show_bytes,
            expected_model_digest=expected_digest,
        )
        if not hmac.compare_digest(
            identity.runtime_fingerprint_sha256, expected_fingerprint
        ):
            raise SecFilingGemmaOllamaError(
                "Ollama runtime fingerprint differs from the candidate"
            )
    finally:
        if owned_transport:
            try:
                close = getattr(active_transport, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

    return OllamaRuntimeProbeReceipt(
        schema_version=RUNTIME_PROBE_RECEIPT_SCHEMA_VERSION,
        version_endpoint=OLLAMA_VERSION_ENDPOINT,
        show_endpoint=OLLAMA_SHOW_ENDPOINT,
        model_name=OLLAMA_MODEL,
        show_request_bytes=show_request_bytes,
        show_request_sha256=hashlib.sha256(show_request_bytes).hexdigest(),
        version_response_bytes=version_bytes,
        version_response_sha256=hashlib.sha256(version_bytes).hexdigest(),
        show_response_bytes=show_bytes,
        show_response_sha256=hashlib.sha256(show_bytes).hexdigest(),
        version_http_status=version_http["http_status"],
        version_response_url=version_http["response_url"],
        version_response_content_type=version_http["response_content_type"],
        version_response_content_length=version_http["response_content_length"],
        version_response_history_count=version_http["response_history_count"],
        show_http_status=show_http["http_status"],
        show_response_url=show_http["response_url"],
        show_response_content_type=show_http["response_content_type"],
        show_response_content_length=show_http["response_content_length"],
        show_response_history_count=show_http["response_history_count"],
        model_digest=identity.model_digest,
        runtime_fingerprint_sha256=identity.runtime_fingerprint_sha256,
        runtime_evidence_bytes=identity.evidence_bytes,
        runtime_evidence_sha256=identity.evidence_sha256,
        transport_mode=(
            _OWNED_PROBE_TRANSPORT_MODE
            if owned_transport
            else _INJECTED_PROBE_TRANSPORT_MODE
        ),
        trusted_production_transport=False,
        network_requests=2,
        redirects=0,
        retries=0,
        pull_attempts=0,
        model_calls=0,
    )


def _validate_persisted_probe_http_evidence(
    manifest: Mapping[str, Any],
    *,
    prefix: str,
    endpoint: str,
    response_bytes: bytes,
) -> None:
    status = manifest[f"{prefix}_http_status"]
    if isinstance(status, bool) or status != 200:
        raise SecFilingGemmaOllamaError(
            f"Ollama runtime-probe {prefix} HTTP status changed"
        )
    if manifest[f"{prefix}_response_url"] != endpoint:
        raise SecFilingGemmaOllamaError(
            f"Ollama runtime-probe {prefix} response URL changed"
        )
    if manifest[f"{prefix}_response_content_type"] != "application/json":
        raise SecFilingGemmaOllamaError(
            f"Ollama runtime-probe {prefix} response content type changed"
        )
    content_length = manifest[f"{prefix}_response_content_length"]
    if content_length is not None and (
        isinstance(content_length, bool)
        or not isinstance(content_length, int)
        or content_length != len(response_bytes)
    ):
        raise SecFilingGemmaOllamaError(
            f"Ollama runtime-probe {prefix} Content-Length changed"
        )
    history_count = manifest[f"{prefix}_response_history_count"]
    if isinstance(history_count, bool) or history_count != 0:
        raise SecFilingGemmaOllamaError(
            f"Ollama runtime-probe {prefix} redirect history changed"
        )


def validate_ollama_runtime_probe_receipt(
    manifest: Mapping[str, Any],
    *,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
    expected_transport_mode: str | None = None,
    expected_probe_receipt_sha256: str | None = None,
) -> OllamaRuntimeProbeReceipt:
    """Replay a persisted probe from bounded bytes against candidate pins."""

    if not isinstance(manifest, Mapping):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe receipt must be a mapping"
        )
    _expect_exact_keys(
        manifest,
        _RUNTIME_PROBE_MANIFEST_KEYS,
        "Ollama runtime-probe receipt",
    )
    if manifest["schema_version"] != RUNTIME_PROBE_RECEIPT_SCHEMA_VERSION:
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe receipt schema changed"
        )
    if (
        manifest["version_endpoint"] != OLLAMA_VERSION_ENDPOINT
        or manifest["show_endpoint"] != OLLAMA_SHOW_ENDPOINT
        or manifest["model_name"] != OLLAMA_MODEL
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe fixed destination or model changed"
        )
    expected_digest = _sha256(expected_model_digest, "expected_model_digest")
    expected_fingerprint = _sha256(
        expected_runtime_fingerprint_sha256,
        "expected_runtime_fingerprint_sha256",
    )
    show_request_bytes = _decode_manifest_bytes(
        manifest["show_request_bytes_base64"],
        location="show_request_bytes_base64",
    )
    if show_request_bytes != _runtime_probe_show_request_bytes():
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe show request changed"
        )
    version_bytes = _decode_manifest_bytes(
        manifest["version_response_bytes_base64"],
        location="version_response_bytes_base64",
    )
    show_bytes = _decode_manifest_bytes(
        manifest["show_response_bytes_base64"],
        location="show_response_bytes_base64",
    )
    for field, payload in (
        ("show_request_sha256", show_request_bytes),
        ("version_response_sha256", version_bytes),
        ("show_response_sha256", show_bytes),
    ):
        observed = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(_sha256(manifest[field], field), observed):
            raise SecFilingGemmaOllamaError(
                f"Ollama runtime-probe {field} does not match its bytes"
            )
    _validate_persisted_probe_http_evidence(
        manifest,
        prefix="version",
        endpoint=OLLAMA_VERSION_ENDPOINT,
        response_bytes=version_bytes,
    )
    _validate_persisted_probe_http_evidence(
        manifest,
        prefix="show",
        endpoint=OLLAMA_SHOW_ENDPOINT,
        response_bytes=show_bytes,
    )
    identity = _derive_runtime_probe_identity(
        version_response_bytes=version_bytes,
        show_response_bytes=show_bytes,
        expected_model_digest=expected_digest,
    )
    if (
        not hmac.compare_digest(identity.model_digest, expected_digest)
        or manifest["model_digest"] != identity.model_digest
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe model digest is not candidate-bound"
        )
    if (
        not hmac.compare_digest(
            identity.runtime_fingerprint_sha256, expected_fingerprint
        )
        or manifest["runtime_fingerprint_sha256"]
        != identity.runtime_fingerprint_sha256
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe fingerprint is not candidate-bound"
        )
    evidence_bytes = _decode_manifest_bytes(
        manifest["runtime_evidence_bytes_base64"],
        location="runtime_evidence_bytes_base64",
    )
    if evidence_bytes != identity.evidence_bytes:
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe compact evidence changed"
        )
    if (
        _sha256(manifest["runtime_evidence_sha256"], "runtime_evidence_sha256")
        != identity.evidence_sha256
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe evidence hash changed"
        )
    permitted_modes = {
        _OWNED_PROBE_TRANSPORT_MODE,
        _INJECTED_PROBE_TRANSPORT_MODE,
    }
    transport_mode = manifest["transport_mode"]
    if transport_mode not in permitted_modes or (
        expected_transport_mode is not None
        and transport_mode != expected_transport_mode
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe transport mode changed or is not pinned"
        )
    if manifest["trusted_production_transport"] is not False:
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe cannot self-attest transport trust"
        )
    for field, expected in (
        ("network_requests", 2),
        ("redirects", 0),
        ("retries", 0),
        ("pull_attempts", 0),
        ("model_calls", 0),
    ):
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise SecFilingGemmaOllamaError(
                f"Ollama runtime-probe {field} changed"
            )
    body = {key: manifest[key] for key in manifest if key != "receipt_sha256"}
    observed_receipt_hash = canonical_sha256(body)
    if not hmac.compare_digest(
        _sha256(manifest["receipt_sha256"], "receipt_sha256"),
        observed_receipt_hash,
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe receipt hash changed"
        )
    if expected_probe_receipt_sha256 is not None and not hmac.compare_digest(
        observed_receipt_hash,
        _sha256(
            expected_probe_receipt_sha256,
            "expected_probe_receipt_sha256",
        ),
    ):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe receipt is not externally pinned"
        )
    receipt = OllamaRuntimeProbeReceipt(
        schema_version=manifest["schema_version"],
        version_endpoint=manifest["version_endpoint"],
        show_endpoint=manifest["show_endpoint"],
        model_name=manifest["model_name"],
        show_request_bytes=show_request_bytes,
        show_request_sha256=manifest["show_request_sha256"],
        version_response_bytes=version_bytes,
        version_response_sha256=manifest["version_response_sha256"],
        show_response_bytes=show_bytes,
        show_response_sha256=manifest["show_response_sha256"],
        version_http_status=manifest["version_http_status"],
        version_response_url=manifest["version_response_url"],
        version_response_content_type=manifest["version_response_content_type"],
        version_response_content_length=manifest[
            "version_response_content_length"
        ],
        version_response_history_count=manifest[
            "version_response_history_count"
        ],
        show_http_status=manifest["show_http_status"],
        show_response_url=manifest["show_response_url"],
        show_response_content_type=manifest["show_response_content_type"],
        show_response_content_length=manifest["show_response_content_length"],
        show_response_history_count=manifest["show_response_history_count"],
        model_digest=manifest["model_digest"],
        runtime_fingerprint_sha256=manifest["runtime_fingerprint_sha256"],
        runtime_evidence_bytes=evidence_bytes,
        runtime_evidence_sha256=manifest["runtime_evidence_sha256"],
        transport_mode=transport_mode,
        trusted_production_transport=False,
        network_requests=2,
        redirects=0,
        retries=0,
        pull_attempts=0,
        model_calls=0,
    )
    if receipt.to_manifest() != dict(manifest):
        raise SecFilingGemmaOllamaError(
            "Ollama runtime-probe receipt is not canonical"
        )
    return receipt


def build_hardened_loopback_session() -> requests.Session:
    """Construct the production transport without proxy or retry inheritance."""

    retry = Retry(
        total=0,
        connect=0,
        read=0,
        redirect=0,
        status=0,
        other=0,
        allowed_methods=frozenset(),
        raise_on_redirect=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.trust_env = False
    session.proxies.clear()
    session.headers.clear()
    session.mount("http://", adapter)
    return session


def call_ollama_extractor_attempt(
    validated_request: Mapping[str, Any],
    *,
    expected_candidate_sha256: str,
    expected_model_payload_sha256: str,
    expected_runtime_evidence_sha256: str,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
    runtime_identity: PinnedOllamaRuntimeIdentity,
    transport: TransportLike | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> OllamaModelAttemptReceipt:
    """Perform exactly one bounded local call and seal valid or invalid output.

    The client never self-attests production trust: even an internally created
    hardened session remains untrusted until the stage runner verifies the
    actual runtime and transport before and after the complete batch.  An
    injected transport exists solely for deterministic offline verification.
    """

    _validate_runtime_identity_object(
        runtime_identity,
        expected_runtime_evidence_sha256=expected_runtime_evidence_sha256,
        expected_model_digest=expected_model_digest,
        expected_runtime_fingerprint_sha256=(
            expected_runtime_fingerprint_sha256
        ),
    )
    candidate_hash, sentence_ids, request_bytes, request_hash = _validated_payload(
        validated_request,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_model_payload_sha256=expected_model_payload_sha256,
    )
    if not callable(monotonic_ns):
        raise SecFilingGemmaOllamaError("A monotonic diagnostic clock is required")
    start = monotonic_ns()
    _strict_nonnegative_int(start, "diagnostic clock start")

    owned_transport = transport is None
    active_transport: TransportLike = (
        build_hardened_loopback_session() if transport is None else transport
    )
    if not hasattr(active_transport, "request"):
        raise SecFilingGemmaOllamaError("Ollama transport must provide request()")
    response: ResponseLike | None = None
    try:
        try:
            response = active_transport.request(
                "POST",
                OLLAMA_ENDPOINT,
                headers=dict(_REQUEST_HEADERS),
                data=request_bytes,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
                allow_redirects=False,
                stream=True,
            )
        except Exception:
            raise SecFilingGemmaOllamaError("Ollama loopback request failed") from None
        response_bytes = _read_bounded_response(response)
        http_evidence = _validate_http_envelope(response, response_bytes)
        output_bytes, normal_completion = _extract_ollama_attempt_output_bytes(
            response_bytes
        )
        output_hash = hashlib.sha256(output_bytes).hexdigest()
        if not normal_completion:
            attempt_status = INVALID_ATTEMPT_STATUS
            invalid_reason: str | None = ABNORMAL_ATTEMPT_REASON
            output_canonical_hash = None
        else:
            try:
                validated_output_hash, output_canonical_hash = (
                    _validate_extractor_output_bytes(
                        output_bytes,
                        supplied_sentence_ids=sentence_ids,
                    )
                )
            except SecFilingGemmaOllamaError:
                attempt_status = INVALID_ATTEMPT_STATUS
                invalid_reason = INVALID_ATTEMPT_REASON
                output_canonical_hash = None
            else:
                if validated_output_hash != output_hash:
                    raise SecFilingGemmaOllamaError(
                        "Validated extractor output hash changed unexpectedly"
                    )
                attempt_status = VALID_ATTEMPT_STATUS
                invalid_reason = None
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if owned_transport:
            try:
                close = getattr(active_transport, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

    end = monotonic_ns()
    _strict_nonnegative_int(end, "diagnostic clock end")
    if end < start:
        raise SecFilingGemmaOllamaError("Diagnostic monotonic clock moved backwards")
    elapsed = end - start
    return OllamaModelAttemptReceipt(
        schema_version=ATTEMPT_RECEIPT_SCHEMA_VERSION,
        attempt_status=attempt_status,
        invalid_reason=invalid_reason,
        endpoint=OLLAMA_ENDPOINT,
        candidate_sha256=candidate_hash,
        sentence_ids=sentence_ids,
        request_bytes=request_bytes,
        request_sha256=request_hash,
        response_bytes=response_bytes,
        response_sha256=hashlib.sha256(response_bytes).hexdigest(),
        extractor_output_bytes=output_bytes,
        extractor_output_sha256=output_hash,
        extractor_output_canonical_sha256=output_canonical_hash,
        model_name=OLLAMA_MODEL,
        model_digest=runtime_identity.model_digest,
        runtime_fingerprint_sha256=runtime_identity.runtime_fingerprint_sha256,
        runtime_evidence_sha256=runtime_identity.evidence_sha256,
        http_status=http_evidence["http_status"],
        response_url=http_evidence["response_url"],
        response_content_type=http_evidence["response_content_type"],
        response_content_length=http_evidence["response_content_length"],
        response_history_count=http_evidence["response_history_count"],
        transport_mode=(
            "owned_hardened_loopback_session_requires_stage_attestation"
            if owned_transport
            else "untrusted_injected_test_transport"
        ),
        trusted_production_transport=False,
        network_requests=1,
        redirects=0,
        retries=0,
        pull_attempts=0,
        repair_attempts=0,
        model_streaming=False,
        model_thinking=False,
        elapsed_nanoseconds=elapsed,
        elapsed_role=DIAGNOSTIC_TIME_ROLE,
    )


def call_ollama_extractor(
    validated_request: Mapping[str, Any],
    *,
    expected_candidate_sha256: str,
    expected_model_payload_sha256: str,
    expected_runtime_evidence_sha256: str,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
    runtime_identity: PinnedOllamaRuntimeIdentity,
    transport: TransportLike | None = None,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> OllamaExtractionReceipt:
    """Backward-compatible valid-only facade over the durable attempt API."""

    attempt = call_ollama_extractor_attempt(
        validated_request,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_model_payload_sha256=expected_model_payload_sha256,
        expected_runtime_evidence_sha256=expected_runtime_evidence_sha256,
        expected_model_digest=expected_model_digest,
        expected_runtime_fingerprint_sha256=expected_runtime_fingerprint_sha256,
        runtime_identity=runtime_identity,
        transport=transport,
        monotonic_ns=monotonic_ns,
    )
    if attempt.attempt_status != VALID_ATTEMPT_STATUS:
        if attempt.invalid_reason == ABNORMAL_ATTEMPT_REASON:
            raise SecFilingGemmaOllamaError(
                "Ollama response was truncated or stopped abnormally"
            )
        # Preserve the original valid-only API's precise failure category while
        # the durable attempt API remains available to the stage runner.
        _validate_extractor_output_bytes(
            attempt.extractor_output_bytes,
            supplied_sentence_ids=attempt.sentence_ids,
        )
        raise SecFilingGemmaOllamaError("Invalid extractor attempt was misclassified")
    canonical_hash = attempt.extractor_output_canonical_sha256
    if canonical_hash is None:
        raise SecFilingGemmaOllamaError(
            "Valid Ollama attempt lacks its canonical extractor output hash"
        )
    return OllamaExtractionReceipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        endpoint=attempt.endpoint,
        candidate_sha256=attempt.candidate_sha256,
        sentence_ids=attempt.sentence_ids,
        request_bytes=attempt.request_bytes,
        request_sha256=attempt.request_sha256,
        response_bytes=attempt.response_bytes,
        response_sha256=attempt.response_sha256,
        extractor_output_bytes=attempt.extractor_output_bytes,
        extractor_output_sha256=attempt.extractor_output_sha256,
        extractor_output_canonical_sha256=canonical_hash,
        model_name=attempt.model_name,
        model_digest=attempt.model_digest,
        runtime_fingerprint_sha256=attempt.runtime_fingerprint_sha256,
        runtime_evidence_sha256=attempt.runtime_evidence_sha256,
        http_status=attempt.http_status,
        transport_mode=attempt.transport_mode,
        trusted_production_transport=attempt.trusted_production_transport,
        network_requests=attempt.network_requests,
        redirects=attempt.redirects,
        retries=attempt.retries,
        pull_attempts=attempt.pull_attempts,
        repair_attempts=attempt.repair_attempts,
        model_streaming=attempt.model_streaming,
        model_thinking=attempt.model_thinking,
        elapsed_nanoseconds=attempt.elapsed_nanoseconds,
        elapsed_role=attempt.elapsed_role,
    )


__all__ = [
    "ABNORMAL_ATTEMPT_REASON",
    "ATTEMPT_RECEIPT_SCHEMA_VERSION",
    "CONNECT_TIMEOUT_SECONDS",
    "DIAGNOSTIC_TIME_ROLE",
    "INVALID_ATTEMPT_REASON",
    "INVALID_ATTEMPT_STATUS",
    "MAX_RESPONSE_BYTES",
    "OLLAMA_ENDPOINT",
    "OLLAMA_MODEL",
    "OLLAMA_SHOW_ENDPOINT",
    "OLLAMA_VERSION_ENDPOINT",
    "OllamaExtractionReceipt",
    "OllamaModelAttemptReceipt",
    "OllamaRuntimeProbeReceipt",
    "PinnedOllamaRuntimeIdentity",
    "READ_TIMEOUT_SECONDS",
    "RECEIPT_SCHEMA_VERSION",
    "RUNTIME_GUARD_SCHEMA_VERSION",
    "RUNTIME_FINGERPRINT_SCHEMA_VERSION",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "RUNTIME_PROBE_RECEIPT_SCHEMA_VERSION",
    "SecFilingGemmaOllamaError",
    "VALID_ATTEMPT_STATUS",
    "build_runtime_identity_guard",
    "build_hardened_loopback_session",
    "call_ollama_extractor",
    "call_ollama_extractor_attempt",
    "probe_owned_ollama_runtime",
    "validate_ollama_extraction_receipt",
    "validate_ollama_model_attempt_receipt",
    "validate_ollama_runtime_probe_receipt",
    "validate_pinned_runtime_identity",
    "validate_runtime_identity_guard",
]
