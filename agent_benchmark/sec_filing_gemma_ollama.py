"""Fail-closed, zero-cost Ollama client for the SEC filing extractor.

The client has one fixed destination and one request shape.  It accepts only
the safe payload returned by ``validate_extractor_request`` and it never
performs discovery, pulls, retries, redirects, streaming, or output repair.
Runtime identity is supplied by a separately sealed local-runtime artifact;
this module deliberately does not query a second endpoint to discover it.
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
OLLAMA_MODEL: Final[str] = "gemma4:12b"
CONNECT_TIMEOUT_SECONDS: Final[float] = 2.0
READ_TIMEOUT_SECONDS: Final[float] = 30.0
MAX_RESPONSE_BYTES: Final[int] = 256 * 1024
RUNTIME_IDENTITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-ollama-runtime-identity-v1"
)
RECEIPT_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-ollama-call-receipt-v1"
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
_REQUEST_HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "identity",
    "Content-Type": "application/json",
}


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


def _validate_http_envelope(response: ResponseLike, response_bytes: bytes) -> None:
    status = getattr(response, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        raise SecFilingGemmaOllamaError("Ollama status code is invalid")
    if 300 <= status < 400:
        raise SecFilingGemmaOllamaError("Ollama redirects are forbidden")
    if status != 200:
        raise SecFilingGemmaOllamaError("Ollama returned a non-success status")
    if getattr(response, "url", None) != OLLAMA_ENDPOINT:
        raise SecFilingGemmaOllamaError("Ollama response URL changed")
    history = getattr(response, "history", None)
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise SecFilingGemmaOllamaError("Ollama redirect history is invalid")
    if len(history) != 0:
        raise SecFilingGemmaOllamaError("Ollama redirects are forbidden")
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        raise SecFilingGemmaOllamaError("Ollama response headers are invalid")
    content_type = (_header(headers, "Content-Type") or "").split(";", 1)[0]
    if content_type.strip().casefold() != "application/json":
        raise SecFilingGemmaOllamaError("Ollama response is not application/json")
    content_length = _header(headers, "Content-Length")
    if content_length is not None:
        if not content_length.isascii() or not content_length.isdecimal():
            raise SecFilingGemmaOllamaError("Ollama Content-Length is invalid")
        if int(content_length) != len(response_bytes):
            raise SecFilingGemmaOllamaError("Ollama Content-Length does not reconcile")


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


def _validate_ollama_response(
    response_bytes: bytes, *, supplied_sentence_ids: tuple[str, ...]
) -> tuple[bytes, str, str]:
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
    if response["done_reason"] != "stop":
        raise SecFilingGemmaOllamaError("Ollama response was truncated or stopped abnormally")
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
    except SecFilingGemmaContractError as exc:
        raise SecFilingGemmaOllamaError("Ollama extractor output violates its schema") from exc
    return (
        output_bytes,
        hashlib.sha256(output_bytes).hexdigest(),
        canonical_sha256(output),
    )


def validate_ollama_extraction_receipt(
    manifest: Mapping[str, Any],
    *,
    expected_candidate_sha256: str,
    expected_model_payload_sha256: str,
    expected_sentence_ids: Sequence[str],
    expected_runtime_evidence_sha256: str,
    expected_model_digest: str,
    expected_runtime_fingerprint_sha256: str,
) -> OllamaExtractionReceipt:
    """Replay a persisted call receipt against candidate-authoritative pins."""

    if not isinstance(manifest, Mapping):
        raise SecFilingGemmaOllamaError("Ollama receipt manifest must be a mapping")
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
    if manifest["transport_mode"] not in {
        "untrusted_injected_test_transport",
        "owned_hardened_loopback_session_requires_stage_attestation",
    }:
        raise SecFilingGemmaOllamaError("Ollama receipt transport mode changed")
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
    """Perform exactly one bounded local extraction call and return its receipt.

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
        _validate_http_envelope(response, response_bytes)
        output_bytes, output_hash, output_canonical_hash = _validate_ollama_response(
            response_bytes,
            supplied_sentence_ids=sentence_ids,
        )
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
    return OllamaExtractionReceipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
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
        http_status=200,
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


__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "DIAGNOSTIC_TIME_ROLE",
    "MAX_RESPONSE_BYTES",
    "OLLAMA_ENDPOINT",
    "OLLAMA_MODEL",
    "OllamaExtractionReceipt",
    "PinnedOllamaRuntimeIdentity",
    "READ_TIMEOUT_SECONDS",
    "RECEIPT_SCHEMA_VERSION",
    "RUNTIME_GUARD_SCHEMA_VERSION",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "SecFilingGemmaOllamaError",
    "build_runtime_identity_guard",
    "build_hardened_loopback_session",
    "call_ollama_extractor",
    "validate_ollama_extraction_receipt",
    "validate_pinned_runtime_identity",
    "validate_runtime_identity_guard",
]
