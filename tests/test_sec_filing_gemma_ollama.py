from __future__ import annotations

import base64
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json
from types import MappingProxyType
from typing import Any, Callable

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    DIMENSION_NAMES,
    EXTRACTOR_SCHEMA_VERSION,
    FLAG_NAMES,
    build_extractor_model_payload,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_ollama import (
    ABNORMAL_ATTEMPT_REASON,
    ATTEMPT_RECEIPT_SCHEMA_VERSION,
    CONNECT_TIMEOUT_SECONDS,
    DIAGNOSTIC_TIME_ROLE,
    INVALID_ATTEMPT_REASON,
    INVALID_ATTEMPT_STATUS,
    MAX_RESPONSE_BYTES,
    OLLAMA_ENDPOINT,
    OLLAMA_MODEL,
    OLLAMA_SHOW_ENDPOINT,
    OLLAMA_VERSION_ENDPOINT,
    READ_TIMEOUT_SECONDS,
    RECEIPT_SCHEMA_VERSION,
    RUNTIME_FINGERPRINT_SCHEMA_VERSION,
    RUNTIME_IDENTITY_SCHEMA_VERSION,
    RUNTIME_PROBE_RECEIPT_SCHEMA_VERSION,
    SecFilingGemmaOllamaError,
    VALID_ATTEMPT_STATUS,
    build_hardened_loopback_session,
    build_runtime_identity_guard,
    call_ollama_extractor,
    call_ollama_extractor_attempt,
    probe_owned_ollama_runtime,
    validate_ollama_extraction_receipt,
    validate_ollama_model_attempt_receipt,
    validate_ollama_runtime_probe_receipt,
    validate_pinned_runtime_identity,
    validate_runtime_identity_guard,
)


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _output() -> dict[str, Any]:
    return {
        "schema_version": EXTRACTOR_SCHEMA_VERSION,
        "document_quality": "usable",
        "dimensions": {
            name: {
                "current_impact": "not_stated",
                "change_vs_prior": "not_stated",
                "evidence_sentence_ids": [],
            }
            for name in DIMENSION_NAMES
        },
        "flags": {
            name: {"present": False, "evidence_sentence_ids": []}
            for name in FLAG_NAMES
        },
    }


def _response_payload(*, output: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "model": OLLAMA_MODEL,
        "created_at": "2026-07-11T10:11:12.123456789Z",
        "message": {
            "role": "assistant",
            "content": json.dumps(
                _output() if output is None else output,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 100,
        "load_duration": 10,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 30,
        "eval_count": 40,
        "eval_duration": 50,
    }


def _response_bytes(payload: dict[str, Any] | None = None) -> bytes:
    return json.dumps(
        _response_payload() if payload is None else payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status_code: int = 200,
        url: str = OLLAMA_ENDPOINT,
        history: list[Any] | None = None,
        headers: dict[str, str] | None = None,
        content_error: Exception | None = None,
    ) -> None:
        self._body = body
        self._content_error = content_error
        self.status_code = status_code
        self.url = url
        self.history = [] if history is None else history
        self.headers = headers or {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
        }
        self.closed = False

    @property
    def content(self) -> bytes:
        if self._content_error is not None:
            raise self._content_error
        return self._body

    def iter_content(self, chunk_size: int):
        if self._content_error is not None:
            raise self._content_error
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(
        self,
        response: FakeResponse | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.response = response or FakeResponse(_response_bytes())
        self.failure = failure
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.failure is not None:
            raise self.failure
        return self.response

    def close(self) -> None:
        self.closed = True


class FakeProbeTransport:
    def __init__(
        self,
        responses: list[FakeResponse],
        *,
        failure_at: int | None = None,
    ) -> None:
        self.responses = list(responses)
        self.failure_at = failure_at
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if self.failure_at == len(self.calls):
            raise RuntimeError("private local transport detail")
        if not self.responses:
            raise AssertionError("Unexpected extra runtime-probe request")
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _probe_show_payload(
    *,
    digest: str | None = None,
    model_name: str = OLLAMA_MODEL,
) -> dict[str, Any]:
    resolved_digest = _digest("installed-model-blob") if digest is None else digest
    return {
        "license": "Gemma terms",
        "modelfile": (
            '# Modelfile generated by "ollama show"\n'
            "# To build a new Modelfile based on this one, replace the FROM line with:\n"
            f"# FROM {model_name}\n\n"
            f"FROM C:/Users/test/.ollama/models/blobs/sha256:{resolved_digest}\n"
        ),
        "parameters": "temperature 0.7\nnum_ctx 8192",
        "template": "{{ .System }}{{ .Prompt }}",
        "modified_at": "2026-07-13T10:11:12.123456789+02:00",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "gemma4",
            "families": ["gemma4"],
            "parameter_size": "12B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {
            "general.architecture": "gemma4",
            "general.parameter_count": 12_000_000_000,
        },
        "capabilities": ["completion"],
    }


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        return json.dumps(value, indent=2, ensure_ascii=True).encode("utf-8")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _probe_context(
    *,
    version_payload: dict[str, Any] | None = None,
    show_payload: dict[str, Any] | None = None,
    pretty: bool = False,
) -> tuple[str, str, bytes, bytes]:
    digest = _digest("installed-model-blob")
    version = {"version": "0.12.3"} if version_payload is None else version_payload
    show = _probe_show_payload(digest=digest) if show_payload is None else show_payload
    version_bytes = _json_bytes(version, pretty=pretty)
    show_bytes = _json_bytes(show, pretty=pretty)
    fingerprint = canonical_sha256(
        {
            "schema_version": RUNTIME_FINGERPRINT_SCHEMA_VERSION,
            "chat_endpoint": OLLAMA_ENDPOINT,
            "version_endpoint": OLLAMA_VERSION_ENDPOINT,
            "show_endpoint": OLLAMA_SHOW_ENDPOINT,
            "model_name": OLLAMA_MODEL,
            "model_digest": digest,
            "version_response": version,
            "show_response": show,
        }
    )
    return digest, fingerprint, version_bytes, show_bytes


def _probe_transport(
    *,
    version_response: FakeResponse | None = None,
    show_response: FakeResponse | None = None,
    pretty: bool = False,
) -> tuple[FakeProbeTransport, str, str]:
    digest, fingerprint, version_bytes, show_bytes = _probe_context(pretty=pretty)
    version = version_response or FakeResponse(
        version_bytes,
        url=OLLAMA_VERSION_ENDPOINT,
    )
    show = show_response or FakeResponse(show_bytes, url=OLLAMA_SHOW_ENDPOINT)
    return FakeProbeTransport([version, show]), digest, fingerprint


def _runtime_context() -> tuple[dict[str, Any], str, str, str]:
    digest = _digest("model")
    fingerprint = _digest("runtime")
    evidence = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "endpoint": OLLAMA_ENDPOINT,
        "model_name": OLLAMA_MODEL,
        "model_digest": digest,
        "runtime_fingerprint_sha256": fingerprint,
    }
    return evidence, canonical_sha256(evidence), digest, fingerprint


def _identity():
    evidence, evidence_hash, digest, fingerprint = _runtime_context()
    return validate_pinned_runtime_identity(
        evidence,
        expected_evidence_sha256=evidence_hash,
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
    )


def _identity_pins(identity) -> dict[str, str]:
    return {
        "expected_runtime_evidence_sha256": identity.evidence_sha256,
        "expected_model_digest": identity.model_digest,
        "expected_runtime_fingerprint_sha256": (
            identity.runtime_fingerprint_sha256
        ),
    }


def _validated_request() -> tuple[dict[str, Any], str, str]:
    sentences = [
        {"id": "C0001", "text": "Demand improved while costs remained uncertain."},
        {"id": "P0001", "text": "Demand was stable while costs were favorable."},
    ]
    payload = build_extractor_model_payload(sentences)
    payload_hash = canonical_sha256(payload)
    candidate_hash = _digest("candidate")
    return (
        {
            "candidate_sha256": candidate_hash,
            "sentence_ids": ("C0001", "P0001"),
            "model_payload": payload,
            "model_payload_sha256": payload_hash,
        },
        candidate_hash,
        payload_hash,
    )


def _clock(*values: int) -> Callable[[], int]:
    iterator = iter(values)
    return lambda: next(iterator)


def _call(
    *,
    response: FakeResponse | None = None,
    transport: FakeTransport | None = None,
    request: dict[str, Any] | None = None,
    identity=None,
):
    valid, candidate_hash, payload_hash = _validated_request()
    resolved_transport = transport or FakeTransport(response)
    resolved_identity = _identity() if identity is None else identity
    receipt = call_ollama_extractor(
        valid if request is None else request,
        expected_candidate_sha256=candidate_hash,
        expected_model_payload_sha256=payload_hash,
        expected_runtime_evidence_sha256=resolved_identity.evidence_sha256,
        expected_model_digest=resolved_identity.model_digest,
        expected_runtime_fingerprint_sha256=(
            resolved_identity.runtime_fingerprint_sha256
        ),
        runtime_identity=resolved_identity,
        transport=resolved_transport,
        monotonic_ns=_clock(100, 175),
    )
    return receipt, resolved_transport


def _call_attempt(
    *,
    response: FakeResponse | None = None,
    transport: FakeTransport | None = None,
):
    valid, candidate_hash, payload_hash = _validated_request()
    resolved_transport = transport or FakeTransport(response)
    identity = _identity()
    receipt = call_ollama_extractor_attempt(
        valid,
        expected_candidate_sha256=candidate_hash,
        expected_model_payload_sha256=payload_hash,
        expected_runtime_evidence_sha256=identity.evidence_sha256,
        expected_model_digest=identity.model_digest,
        expected_runtime_fingerprint_sha256=(
            identity.runtime_fingerprint_sha256
        ),
        runtime_identity=identity,
        transport=resolved_transport,
        monotonic_ns=_clock(100, 175),
    )
    return receipt, resolved_transport


def test_one_exact_loopback_call_and_immutable_byte_receipt() -> None:
    receipt, transport = _call()
    valid, candidate_hash, payload_hash = _validated_request()
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call == {
        "method": "POST",
        "url": "http://127.0.0.1:11434/api/chat",
        "headers": {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
        },
        "data": receipt.request_bytes,
        "timeout": (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        "allow_redirects": False,
        "stream": True,
    }
    assert "json" not in call and "proxies" not in call
    assert call["stream"] is True
    assert json.loads(receipt.request_bytes) == valid["model_payload"]
    assert receipt.request_sha256 == payload_hash
    assert receipt.candidate_sha256 == candidate_hash
    assert receipt.sentence_ids == ("C0001", "P0001")
    assert receipt.schema_version == RECEIPT_SCHEMA_VERSION
    assert receipt.endpoint == OLLAMA_ENDPOINT
    assert receipt.response_sha256 == hashlib.sha256(receipt.response_bytes).hexdigest()
    assert receipt.extractor_output() == _output()
    assert receipt.extractor_output() is not receipt.extractor_output()
    assert receipt.elapsed_nanoseconds == 75
    assert receipt.elapsed_role == DIAGNOSTIC_TIME_ROLE
    assert receipt.transport_mode == "untrusted_injected_test_transport"
    assert receipt.trusted_production_transport is False
    assert receipt.network_requests == 1
    assert receipt.redirects == receipt.retries == 0
    assert receipt.pull_attempts == receipt.repair_attempts == 0
    assert receipt.model_streaming is receipt.model_thinking is False
    assert transport.response.closed is True
    assert transport.closed is False
    with pytest.raises(FrozenInstanceError):
        receipt.model_name = "changed"  # type: ignore[misc]

    manifest = receipt.to_manifest()
    body = dict(manifest)
    assert body.pop("receipt_sha256") == receipt.receipt_sha256
    assert canonical_sha256(body) == receipt.receipt_sha256
    assert base64.b64decode(manifest["request_bytes_base64"]) == receipt.request_bytes
    assert base64.b64decode(manifest["response_bytes_base64"]) == receipt.response_bytes


def test_persisted_receipt_replays_against_candidate_authoritative_pins() -> None:
    receipt, _ = _call()
    replayed = validate_ollama_extraction_receipt(
        receipt.to_manifest(),
        expected_candidate_sha256=receipt.candidate_sha256,
        expected_model_payload_sha256=receipt.request_sha256,
        expected_sentence_ids=receipt.sentence_ids,
        expected_runtime_evidence_sha256=receipt.runtime_evidence_sha256,
        expected_model_digest=receipt.model_digest,
        expected_runtime_fingerprint_sha256=(
            receipt.runtime_fingerprint_sha256
        ),
        expected_transport_mode=receipt.transport_mode,
    )
    assert replayed == receipt
    assert replayed.to_manifest() == receipt.to_manifest()


def _replay_attempt(receipt):
    return validate_ollama_model_attempt_receipt(
        receipt.to_manifest(),
        expected_candidate_sha256=receipt.candidate_sha256,
        expected_model_payload_sha256=receipt.request_sha256,
        expected_sentence_ids=receipt.sentence_ids,
        expected_runtime_evidence_sha256=receipt.runtime_evidence_sha256,
        expected_model_digest=receipt.model_digest,
        expected_runtime_fingerprint_sha256=(
            receipt.runtime_fingerprint_sha256
        ),
        expected_transport_mode=receipt.transport_mode,
    )


def test_valid_attempt_receipt_roundtrips_and_preserves_valid_facade() -> None:
    attempt, transport = _call_attempt()
    assert attempt.schema_version == ATTEMPT_RECEIPT_SCHEMA_VERSION
    assert attempt.attempt_status == VALID_ATTEMPT_STATUS
    assert attempt.invalid_reason is None
    assert attempt.validated_extractor_output() == _output()
    assert len(transport.calls) == 1
    assert attempt.network_requests == 1
    assert attempt.retries == attempt.repair_attempts == 0
    assert attempt.trusted_production_transport is False
    assert _replay_attempt(attempt) == attempt


def test_invalid_schema_is_sealed_once_without_semantic_output_or_retry() -> None:
    invalid = _output()
    invalid["dimensions"]["demand"] = {
        "current_impact": "favorable",
        "change_vs_prior": "improving",
        "evidence_sentence_ids": ["C9999"],
    }
    transport = FakeTransport(
        FakeResponse(_response_bytes(_response_payload(output=invalid)))
    )
    attempt, _ = _call_attempt(transport=transport)
    assert attempt.attempt_status == INVALID_ATTEMPT_STATUS
    assert attempt.invalid_reason == INVALID_ATTEMPT_REASON
    assert attempt.extractor_output_canonical_sha256 is None
    assert len(transport.calls) == 1
    assert attempt.network_requests == 1
    assert attempt.retries == attempt.repair_attempts == 0
    with pytest.raises(SecFilingGemmaOllamaError, match="cannot be exposed"):
        attempt.validated_extractor_output()
    assert _replay_attempt(attempt) == attempt


def test_malformed_inner_json_is_an_auditable_invalid_attempt() -> None:
    payload = _response_payload()
    payload["message"]["content"] = "```json\n{}\n```"
    attempt, transport = _call_attempt(
        response=FakeResponse(_response_bytes(payload))
    )
    assert attempt.attempt_status == INVALID_ATTEMPT_STATUS
    assert attempt.invalid_reason == INVALID_ATTEMPT_REASON
    assert attempt.extractor_output_canonical_sha256 is None
    assert len(transport.calls) == 1
    assert _replay_attempt(attempt) == attempt


def test_abnormal_completed_response_is_sealed_once_as_invalid() -> None:
    payload = _response_payload()
    payload["done_reason"] = "length"
    attempt, transport = _call_attempt(
        response=FakeResponse(_response_bytes(payload))
    )
    assert attempt.attempt_status == INVALID_ATTEMPT_STATUS
    assert attempt.invalid_reason == ABNORMAL_ATTEMPT_REASON
    assert attempt.extractor_output_canonical_sha256 is None
    assert len(transport.calls) == 1
    assert attempt.retries == attempt.repair_attempts == 0
    assert _replay_attempt(attempt) == attempt


def test_http_envelope_is_captured_once_without_post_validation_reread() -> None:
    class SwitchingUrlResponse(FakeResponse):
        def __init__(self, body: bytes) -> None:
            self._url_reads = 0
            super().__init__(body)

        @property
        def url(self) -> str:
            self._url_reads += 1
            return OLLAMA_ENDPOINT if self._url_reads == 1 else "http://evil.invalid/"

        @url.setter
        def url(self, _value: str) -> None:
            pass

    response = SwitchingUrlResponse(_response_bytes())
    attempt, _ = _call_attempt(response=response)
    assert attempt.response_url == OLLAMA_ENDPOINT
    assert response._url_reads == 1
    assert _replay_attempt(attempt) == attempt


def test_unhashable_schema_value_is_sealed_as_invalid_instead_of_escaping() -> None:
    invalid = _output()
    invalid["dimensions"]["demand"]["evidence_sentence_ids"] = [{}]
    attempt, transport = _call_attempt(
        response=FakeResponse(_response_bytes(_response_payload(output=invalid)))
    )
    assert attempt.attempt_status == INVALID_ATTEMPT_STATUS
    assert attempt.invalid_reason == INVALID_ATTEMPT_REASON
    assert attempt.extractor_output_canonical_sha256 is None
    assert len(transport.calls) == 1
    assert _replay_attempt(attempt) == attempt


def test_attempt_validator_rejects_live_or_switching_mapping_views() -> None:
    attempt, _ = _call_attempt()
    with pytest.raises(SecFilingGemmaOllamaError, match="mapping"):
        validate_ollama_model_attempt_receipt(
            MappingProxyType(attempt.to_manifest()),
            expected_candidate_sha256=attempt.candidate_sha256,
            expected_model_payload_sha256=attempt.request_sha256,
            expected_sentence_ids=attempt.sentence_ids,
            expected_runtime_evidence_sha256=attempt.runtime_evidence_sha256,
            expected_model_digest=attempt.model_digest,
            expected_runtime_fingerprint_sha256=(
                attempt.runtime_fingerprint_sha256
            ),
            expected_transport_mode=attempt.transport_mode,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(attempt_status=VALID_ATTEMPT_STATUS),
        lambda value: value.update(invalid_reason=None),
        lambda value: value.update(extractor_output_canonical_sha256=_digest("fake")),
        lambda value: value.update(network_requests=2),
        lambda value: value.update(trusted_production_transport=True),
        lambda value: value.update(response_url="http://localhost:11434/api/chat"),
        lambda value: value.update(response_content_type="text/plain"),
        lambda value: value.update(response_content_length=0),
        lambda value: value.update(response_history_count=1),
        lambda value: value.update(
            transport_mode="owned_hardened_loopback_session_requires_stage_attestation"
        ),
    ],
)
def test_rehashed_invalid_attempt_forgery_is_rejected(mutation) -> None:
    payload = _response_payload()
    payload["message"]["content"] = "not-json"
    attempt, _ = _call_attempt(response=FakeResponse(_response_bytes(payload)))
    manifest = attempt.to_manifest()
    mutation(manifest)
    manifest["receipt_sha256"] = canonical_sha256(
        {key: manifest[key] for key in manifest if key != "receipt_sha256"}
    )
    with pytest.raises(SecFilingGemmaOllamaError):
        validate_ollama_model_attempt_receipt(
            manifest,
            expected_candidate_sha256=attempt.candidate_sha256,
            expected_model_payload_sha256=attempt.request_sha256,
            expected_sentence_ids=attempt.sentence_ids,
            expected_runtime_evidence_sha256=attempt.runtime_evidence_sha256,
            expected_model_digest=attempt.model_digest,
            expected_runtime_fingerprint_sha256=(
                attempt.runtime_fingerprint_sha256
            ),
            expected_transport_mode=attempt.transport_mode,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.update(trusted_production_transport=True),
            "trusted_production_transport",
        ),
        (
            lambda value: value.update(model_digest=_digest("other model")),
            "model_digest",
        ),
        (
            lambda value: value.update(request_bytes_base64="not-base64!"),
            "base64",
        ),
        (
            lambda value: value.update(network_requests=True),
            "network_requests",
        ),
        (
            lambda value: value.update(
                transport_mode=(
                    "owned_hardened_loopback_session_requires_stage_attestation"
                )
            ),
            "transport mode",
        ),
    ],
)
def test_rehashed_persisted_receipt_tampering_is_rejected(
    mutate: Callable[[dict[str, Any]], None], match: str
) -> None:
    receipt, _ = _call()
    manifest = receipt.to_manifest()
    mutate(manifest)
    manifest["receipt_sha256"] = canonical_sha256(
        {key: manifest[key] for key in manifest if key != "receipt_sha256"}
    )
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        validate_ollama_extraction_receipt(
            manifest,
            expected_candidate_sha256=receipt.candidate_sha256,
            expected_model_payload_sha256=receipt.request_sha256,
            expected_sentence_ids=receipt.sentence_ids,
            expected_runtime_evidence_sha256=receipt.runtime_evidence_sha256,
            expected_model_digest=receipt.model_digest,
            expected_runtime_fingerprint_sha256=(
                receipt.runtime_fingerprint_sha256
            ),
            expected_transport_mode=receipt.transport_mode,
        )


def test_payload_serializes_only_contract_validated_model_fields() -> None:
    receipt, _ = _call()
    decoded = json.loads(receipt.request_bytes)
    assert set(decoded) == {"model", "messages", "format", "stream", "think", "options"}
    assert decoded["model"] == OLLAMA_MODEL
    assert decoded["stream"] is decoded["think"] is False
    serialized = receipt.request_bytes.decode("utf-8")
    assert "candidate_sha256" not in serialized
    assert "model_payload_sha256" not in serialized
    assert "current_accession_number" not in serialized
    assert receipt.request_bytes == json.dumps(
        decoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def test_pure_runtime_identity_validator_binds_external_evidence() -> None:
    evidence, evidence_hash, digest, fingerprint = _runtime_context()
    identity = validate_pinned_runtime_identity(
        evidence,
        expected_evidence_sha256=evidence_hash,
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
    )
    assert identity.evidence_sha256 == evidence_hash
    assert identity.evidence_bytes == json.dumps(
        evidence, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert identity.model_digest == digest
    assert identity.runtime_fingerprint_sha256 == fingerprint


@pytest.mark.parametrize(
    ("mutation", "expected_overrides", "match"),
    [
        (lambda value: value.update(extra=True), {}, "Invalid runtime identity"),
        (
            lambda value: value.update(schema_version="changed"),
            {},
            "schema version",
        ),
        (lambda value: value.update(endpoint="http://localhost:11434/api/chat"), {}, "endpoint"),
        (lambda value: value.update(model_name="gemma4:latest"), {}, "model name"),
        (lambda value: value.update(model_digest=_digest("other")), {}, "model digest"),
        (
            lambda value: value.update(runtime_fingerprint_sha256=_digest("other")),
            {},
            "fingerprint",
        ),
    ],
)
def test_runtime_identity_rejects_mutations_even_with_rehashed_evidence(
    mutation: Callable[[dict[str, Any]], None],
    expected_overrides: dict[str, str],
    match: str,
) -> None:
    evidence, _, digest, fingerprint = _runtime_context()
    mutation(evidence)
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        validate_pinned_runtime_identity(
            evidence,
            expected_evidence_sha256=canonical_sha256(evidence),
            expected_model_digest=expected_overrides.get("digest", digest),
            expected_runtime_fingerprint_sha256=expected_overrides.get(
                "fingerprint", fingerprint
            ),
        )


def test_runtime_identity_rejects_wrong_external_pin_before_call() -> None:
    evidence, _, digest, fingerprint = _runtime_context()
    with pytest.raises(SecFilingGemmaOllamaError, match="externally pinned"):
        validate_pinned_runtime_identity(
            evidence,
            expected_evidence_sha256=_digest("wrong"),
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=fingerprint,
        )


def test_runtime_guard_binds_identical_pre_and_post_batch_identity() -> None:
    evidence, _, digest, fingerprint = _runtime_context()
    receipts = [_digest("call-1"), _digest("call-2")]
    guard = build_runtime_identity_guard(
        before_evidence=evidence,
        after_evidence=copy.deepcopy(evidence),
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        stage="development",
        model_call_receipt_sha256s=receipts,
    )
    assert guard["runtime_identity_unchanged"] is True
    assert guard["model_call_count"] == 2
    assert validate_runtime_identity_guard(
        guard,
        before_evidence=evidence,
        after_evidence=copy.deepcopy(evidence),
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        stage="development",
        model_call_receipt_sha256s=receipts,
        expected_runtime_guard_sha256=guard["runtime_guard_sha256"],
    ) == guard["runtime_guard_sha256"]


def test_runtime_guard_rejects_mid_batch_model_change() -> None:
    before, _, digest, fingerprint = _runtime_context()
    after = copy.deepcopy(before)
    after["model_digest"] = _digest("changed-mid-batch")
    with pytest.raises(SecFilingGemmaOllamaError, match="candidate"):
        build_runtime_identity_guard(
            before_evidence=before,
            after_evidence=after,
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=fingerprint,
            stage="development",
            model_call_receipt_sha256s=[_digest("call")],
        )


def test_runtime_guard_preserves_duplicate_receipt_multiplicity_and_order() -> None:
    evidence, _, digest, fingerprint = _runtime_context()
    repeated = _digest("same-call-bytes")
    other = _digest("other-call")
    receipts = [repeated, repeated, other]
    guard = build_runtime_identity_guard(
        before_evidence=evidence,
        after_evidence=copy.deepcopy(evidence),
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        stage="development",
        model_call_receipt_sha256s=receipts,
    )
    assert guard["model_call_count"] == 3
    assert guard["model_call_receipt_sha256s"] == receipts
    assert guard["model_call_receipt_sequence_sha256"] == canonical_sha256(
        receipts
    )
    assert validate_runtime_identity_guard(
        guard,
        before_evidence=evidence,
        after_evidence=copy.deepcopy(evidence),
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        stage="development",
        model_call_receipt_sha256s=receipts,
    ) == guard["runtime_guard_sha256"]

    reordered = build_runtime_identity_guard(
        before_evidence=evidence,
        after_evidence=copy.deepcopy(evidence),
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        stage="development",
        model_call_receipt_sha256s=[repeated, other, repeated],
    )
    assert reordered["runtime_guard_sha256"] != guard["runtime_guard_sha256"]


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update(extra=True), "Invalid validated extractor"),
        (lambda value: value.update(candidate_sha256=_digest("other")), "candidate binding"),
        (lambda value: value.update(model_payload_sha256=_digest("other")), "externally pinned"),
        (lambda value: value.update(sentence_ids=("P0001", "C0001")), "sentence ids"),
        (lambda value: value.update(sentence_ids=("C0001", "C0001")), "sentence_ids"),
        (lambda value: value["model_payload"]["options"].update(seed=1), "frozen safe payload"),
        (lambda value: value["model_payload"].update(model="gemma4:latest"), "frozen safe payload"),
        (lambda value: value["model_payload"].update(stream=True), "frozen safe payload"),
        (lambda value: value["model_payload"].update(think=True), "frozen safe payload"),
    ],
)
def test_request_mutations_fail_before_any_transport_call(
    mutator: Callable[[dict[str, Any]], None], match: str
) -> None:
    request, candidate_hash, payload_hash = _validated_request()
    request = copy.deepcopy(request)
    mutator(request)
    transport = FakeTransport()
    identity = _identity()
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        call_ollama_extractor(
            request,
            expected_candidate_sha256=candidate_hash,
            expected_model_payload_sha256=payload_hash,
            **_identity_pins(identity),
            runtime_identity=identity,
            transport=transport,
            monotonic_ns=_clock(1),
        )
    assert transport.calls == []


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update(model="gemma4:latest"), "model identity"),
        (lambda value: value.update(done=False), "not complete"),
        (lambda value: value.update(done_reason="length"), "truncated"),
        (lambda value: value.update(error="failure"), "Invalid Ollama response"),
        (lambda value: value.update(created_at="yesterday"), "created_at"),
        (lambda value: value.update(total_duration=-1), "nonnegative integer"),
        (lambda value: value.update(eval_count=1.5), "nonnegative integer"),
        (lambda value: value.update(load_duration=True), "nonnegative integer"),
        (lambda value: value["message"].update(role="tool"), "message changed"),
        (lambda value: value["message"].update(thinking="secret"), "Invalid Ollama response message"),
    ],
)
def test_strict_response_envelope_rejects_extra_or_changed_fields(
    mutator: Callable[[dict[str, Any]], None], match: str
) -> None:
    payload = _response_payload()
    mutator(payload)
    response = FakeResponse(_response_bytes(payload))
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        _call(response=response)
    assert response.closed is True


def test_extractor_output_is_schema_and_evidence_validated_without_repair() -> None:
    output = _output()
    output["dimensions"]["demand"] = {
        "current_impact": "favorable",
        "change_vs_prior": "not_stated",
        "evidence_sentence_ids": ["C9999"],
    }
    response = FakeResponse(_response_bytes(_response_payload(output=output)))
    transport = FakeTransport(response)
    with pytest.raises(SecFilingGemmaOllamaError, match="violates its schema"):
        _call(transport=transport)
    assert len(transport.calls) == 1
    assert response.closed is True


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (b"not-json", "malformed JSON"),
        (b'{"model":"gemma4:12b","model":"other"}', "duplicate JSON key"),
        (b'{"model":NaN}', "non-finite"),
        (b"\xff", "strict UTF-8"),
    ],
)
def test_malformed_duplicate_nonfinite_or_non_utf8_response_is_rejected(
    body: bytes, match: str
) -> None:
    response = FakeResponse(body)
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        _call(response=response)


def test_inner_model_content_must_be_plain_strict_json() -> None:
    payload = _response_payload()
    payload["message"]["content"] = "```json\n{}\n```"
    with pytest.raises(SecFilingGemmaOllamaError, match="malformed JSON"):
        _call(response=FakeResponse(_response_bytes(payload)))


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (FakeResponse(_response_bytes(), status_code=302), "redirects"),
        (FakeResponse(_response_bytes(), status_code=500), "non-success"),
        (
            FakeResponse(_response_bytes(), url="http://localhost:11434/api/chat"),
            "response URL",
        ),
        (FakeResponse(_response_bytes(), history=[object()]), "redirects"),
        (
            FakeResponse(_response_bytes(), headers={"Content-Type": "text/plain"}),
            "application/json",
        ),
        (
            FakeResponse(
                _response_bytes(),
                headers={"Content-Type": "application/json", "Content-Length": "1"},
            ),
            "does not reconcile",
        ),
    ],
)
def test_status_redirect_destination_and_headers_are_fail_closed(
    response: FakeResponse, match: str
) -> None:
    transport = FakeTransport(response)
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        _call(transport=transport)
    assert len(transport.calls) == 1
    assert response.closed is True


def test_oversized_response_is_rejected_and_closed() -> None:
    body = b"{" + b" " * MAX_RESPONSE_BYTES + b"}"
    response = FakeResponse(body)
    with pytest.raises(SecFilingGemmaOllamaError, match="byte ceiling"):
        _call(response=response)
    assert response.closed is True


def test_transport_failure_is_sanitized_and_never_retried() -> None:
    private = "secret redacted filing sentence"
    transport = FakeTransport(failure=RuntimeError(private))
    with pytest.raises(SecFilingGemmaOllamaError, match="loopback request failed") as captured:
        _call(transport=transport)
    assert private not in str(captured.value)
    assert captured.value.__cause__ is None
    assert len(transport.calls) == 1


def test_unreadable_response_is_closed_without_fallback() -> None:
    response = FakeResponse(
        b"",
        content_error=RuntimeError("private runtime failure"),
    )
    transport = FakeTransport(response)
    with pytest.raises(SecFilingGemmaOllamaError, match="bytes are unreadable"):
        _call(transport=transport)
    assert len(transport.calls) == 1
    assert response.closed is True


def test_forged_or_mutated_identity_object_fails_before_network() -> None:
    identity = replace(_identity(), model_digest=_digest("forged"))
    transport = FakeTransport()
    request, candidate_hash, payload_hash = _validated_request()
    with pytest.raises(SecFilingGemmaOllamaError, match="inconsistent"):
        call_ollama_extractor(
            request,
            expected_candidate_sha256=candidate_hash,
            expected_model_payload_sha256=payload_hash,
            **_identity_pins(identity),
            runtime_identity=identity,
            transport=transport,
            monotonic_ns=_clock(1),
        )
    assert transport.calls == []


def test_coherent_wrong_runtime_identity_fails_against_candidate_pins() -> None:
    expected_identity = _identity()
    forged_evidence = {
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "endpoint": OLLAMA_ENDPOINT,
        "model_name": OLLAMA_MODEL,
        "model_digest": _digest("coherent-but-wrong-model"),
        "runtime_fingerprint_sha256": _digest("coherent-but-wrong-runtime"),
    }
    forged_identity = validate_pinned_runtime_identity(
        forged_evidence,
        expected_evidence_sha256=canonical_sha256(forged_evidence),
        expected_model_digest=forged_evidence["model_digest"],
        expected_runtime_fingerprint_sha256=forged_evidence[
            "runtime_fingerprint_sha256"
        ],
    )
    request, candidate_hash, payload_hash = _validated_request()
    transport = FakeTransport()

    with pytest.raises(SecFilingGemmaOllamaError, match="candidate pin"):
        call_ollama_extractor(
            request,
            expected_candidate_sha256=candidate_hash,
            expected_model_payload_sha256=payload_hash,
            **_identity_pins(expected_identity),
            runtime_identity=forged_identity,
            transport=transport,
            monotonic_ns=_clock(1),
        )
    assert transport.calls == []


def test_hardened_requests_session_has_no_env_proxy_or_retry_inheritance() -> None:
    session = build_hardened_loopback_session()
    try:
        assert session.trust_env is False
        assert session.proxies == {}
        assert dict(session.headers) == {}
        retries = session.get_adapter(OLLAMA_ENDPOINT).max_retries
        assert retries.total == retries.connect == retries.read == 0
        assert retries.redirect == retries.status == retries.other == 0
    finally:
        session.close()


def test_owned_production_transport_is_distinctly_marked_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()
    monkeypatch.setattr(
        "agent_benchmark.sec_filing_gemma_ollama.build_hardened_loopback_session",
        lambda: transport,
    )
    request, candidate_hash, payload_hash = _validated_request()
    identity = _identity()
    receipt = call_ollama_extractor(
        request,
        expected_candidate_sha256=candidate_hash,
        expected_model_payload_sha256=payload_hash,
        **_identity_pins(identity),
        runtime_identity=identity,
        monotonic_ns=_clock(10, 20),
    )
    assert (
        receipt.transport_mode
        == "owned_hardened_loopback_session_requires_stage_attestation"
    )
    assert receipt.trusted_production_transport is False
    assert transport.closed is True
    assert len(transport.calls) == 1


def test_backward_or_noninteger_diagnostic_clock_cannot_create_receipt() -> None:
    request, candidate_hash, payload_hash = _validated_request()
    identity = _identity()
    with pytest.raises(SecFilingGemmaOllamaError, match="moved backwards"):
        call_ollama_extractor(
            request,
            expected_candidate_sha256=candidate_hash,
            expected_model_payload_sha256=payload_hash,
            **_identity_pins(identity),
            runtime_identity=identity,
            transport=FakeTransport(),
            monotonic_ns=_clock(20, 10),
        )
    transport = FakeTransport()
    with pytest.raises(SecFilingGemmaOllamaError, match="nonnegative integer"):
        call_ollama_extractor(
            request,
            expected_candidate_sha256=candidate_hash,
            expected_model_payload_sha256=payload_hash,
            **_identity_pins(identity),
            runtime_identity=identity,
            transport=transport,
            monotonic_ns=lambda: 1.5,  # type: ignore[return-value]
        )
    assert transport.calls == []
    VALID_ATTEMPT_STATUS,


def test_owned_runtime_probe_makes_two_exact_non_model_loopback_calls() -> None:
    transport, digest, fingerprint = _probe_transport()
    version_response, show_response = list(transport.responses)
    receipt = probe_owned_ollama_runtime(
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        transport=transport,
    )

    assert receipt.schema_version == RUNTIME_PROBE_RECEIPT_SCHEMA_VERSION
    assert receipt.model_digest == digest
    assert receipt.runtime_fingerprint_sha256 == fingerprint
    assert receipt.transport_mode == "untrusted_injected_test_runtime_probe_transport"
    assert receipt.trusted_production_transport is False
    assert receipt.network_requests == 2
    assert receipt.redirects == receipt.retries == receipt.pull_attempts == 0
    assert receipt.model_calls == 0
    assert [call["url"] for call in transport.calls] == [
        OLLAMA_VERSION_ENDPOINT,
        OLLAMA_SHOW_ENDPOINT,
    ]
    assert transport.calls[0] == {
        "method": "GET",
        "url": OLLAMA_VERSION_ENDPOINT,
        "headers": {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
        "timeout": (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        "allow_redirects": False,
        "stream": True,
    }
    assert transport.calls[1] == {
        "method": "POST",
        "url": OLLAMA_SHOW_ENDPOINT,
        "headers": {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Content-Type": "application/json",
        },
        "timeout": (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        "allow_redirects": False,
        "stream": True,
        "data": b'{"model":"gemma4:12b","verbose":false}',
    }
    assert all(call["url"] != OLLAMA_ENDPOINT for call in transport.calls)
    assert version_response.closed is show_response.closed is True

    identity = receipt.pinned_runtime_identity()
    assert identity.model_digest == digest
    assert identity.runtime_fingerprint_sha256 == fingerprint
    replayed = validate_ollama_runtime_probe_receipt(
        receipt.to_manifest(),
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        expected_transport_mode=receipt.transport_mode,
        expected_probe_receipt_sha256=receipt.receipt_sha256,
    )
    assert replayed == receipt


def test_runtime_probe_before_after_semantic_replay_is_byte_order_independent() -> None:
    before_transport, digest, fingerprint = _probe_transport(pretty=False)
    after_transport, after_digest, after_fingerprint = _probe_transport(pretty=True)
    assert (after_digest, after_fingerprint) == (digest, fingerprint)
    before = probe_owned_ollama_runtime(
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        transport=before_transport,
    )
    after = probe_owned_ollama_runtime(
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        transport=after_transport,
    )
    assert before.version_response_bytes != after.version_response_bytes
    assert before.show_response_bytes != after.show_response_bytes
    assert before.runtime_evidence_bytes == after.runtime_evidence_bytes
    assert before.runtime_evidence_sha256 == after.runtime_evidence_sha256
    guard = build_runtime_identity_guard(
        before_evidence=before.runtime_evidence(),
        after_evidence=after.runtime_evidence(),
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        stage="development",
        model_call_receipt_sha256s=[_digest("model-call")],
    )
    assert guard["runtime_identity_unchanged"] is True


def test_runtime_probe_owned_transport_is_hardened_marked_and_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport, digest, fingerprint = _probe_transport()
    monkeypatch.setattr(
        "agent_benchmark.sec_filing_gemma_ollama.build_hardened_loopback_session",
        lambda: transport,
    )
    receipt = probe_owned_ollama_runtime(
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
    )
    assert receipt.transport_mode == "owned_hardened_loopback_runtime_probe_unattested"
    assert receipt.trusted_production_transport is False
    assert transport.closed is True


def test_runtime_probe_public_api_exposes_no_url_or_model_authority() -> None:
    assert list(inspect.signature(probe_owned_ollama_runtime).parameters) == [
        "expected_model_digest",
        "expected_runtime_fingerprint_sha256",
        "transport",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in inspect.signature(
            probe_owned_ollama_runtime
        ).parameters.values()
    )


@pytest.mark.parametrize(
    ("wrong_model", "wrong_digest", "wrong_fingerprint", "match"),
    [
        ("gemma4:latest", None, None, "exact fixed model"),
        (None, _digest("different installed blob"), None, "installed model digest"),
        (None, None, _digest("different runtime"), "runtime fingerprint"),
    ],
)
def test_runtime_probe_rejects_wrong_model_digest_or_candidate_fingerprint(
    wrong_model: str | None,
    wrong_digest: str | None,
    wrong_fingerprint: str | None,
    match: str,
) -> None:
    digest, fingerprint, version_bytes, show_bytes = _probe_context()
    if wrong_model is not None or wrong_digest is not None:
        show_bytes = _json_bytes(
            _probe_show_payload(
                digest=digest if wrong_digest is None else wrong_digest,
                model_name=OLLAMA_MODEL if wrong_model is None else wrong_model,
            )
        )
    transport = FakeProbeTransport(
        [
            FakeResponse(version_bytes, url=OLLAMA_VERSION_ENDPOINT),
            FakeResponse(show_bytes, url=OLLAMA_SHOW_ENDPOINT),
        ]
    )
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        probe_owned_ollama_runtime(
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=(
                fingerprint if wrong_fingerprint is None else wrong_fingerprint
            ),
            transport=transport,
        )
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    ("target", "response_kwargs", "match"),
    [
        ("version", {"status_code": 302}, "redirects"),
        ("version", {"status_code": 500}, "non-success"),
        (
            "version",
            {"url": "http://localhost:11434/api/version"},
            "response URL",
        ),
        ("show", {"history": [object()]}, "redirects"),
        (
            "show",
            {"headers": {"Content-Type": "text/plain"}},
            "application/json",
        ),
        (
            "show",
            {
                "headers": {
                    "Content-Type": "application/json",
                    "Content-Length": "1",
                }
            },
            "does not reconcile",
        ),
    ],
)
def test_runtime_probe_http_envelopes_fail_closed(
    target: str,
    response_kwargs: dict[str, Any],
    match: str,
) -> None:
    digest, fingerprint, version_bytes, show_bytes = _probe_context()
    version_kwargs = (
        response_kwargs if target == "version" else {"url": OLLAMA_VERSION_ENDPOINT}
    )
    show_kwargs = (
        response_kwargs if target == "show" else {"url": OLLAMA_SHOW_ENDPOINT}
    )
    if "url" not in version_kwargs:
        version_kwargs = {**version_kwargs, "url": OLLAMA_VERSION_ENDPOINT}
    if "url" not in show_kwargs:
        show_kwargs = {**show_kwargs, "url": OLLAMA_SHOW_ENDPOINT}
    responses = [
        FakeResponse(version_bytes, **version_kwargs),
        FakeResponse(show_bytes, **show_kwargs),
    ]
    transport = FakeProbeTransport(responses)
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        probe_owned_ollama_runtime(
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=fingerprint,
            transport=transport,
        )
    assert 1 <= len(transport.calls) <= 2
    assert responses[len(transport.calls) - 1].closed is True


def test_runtime_probe_rejects_oversize_without_attempting_show() -> None:
    digest, fingerprint, _, show_bytes = _probe_context()
    response = FakeResponse(
        b"{" + b" " * MAX_RESPONSE_BYTES + b"}",
        url=OLLAMA_VERSION_ENDPOINT,
    )
    transport = FakeProbeTransport(
        [response, FakeResponse(show_bytes, url=OLLAMA_SHOW_ENDPOINT)]
    )
    with pytest.raises(SecFilingGemmaOllamaError, match="byte ceiling"):
        probe_owned_ollama_runtime(
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=fingerprint,
            transport=transport,
        )
    assert len(transport.calls) == 1
    assert response.closed is True


@pytest.mark.parametrize(
    ("version_bytes", "match"),
    [
        (b'{"version":"0.12.3","version":"0.12.4"}', "duplicate JSON key"),
        (b'{"version":NaN}', "non-finite"),
        (b'{"version":"dev"}', "semantic version"),
    ],
)
def test_runtime_probe_rejects_malformed_or_ambiguous_version_bytes(
    version_bytes: bytes,
    match: str,
) -> None:
    digest, fingerprint, _, show_bytes = _probe_context()
    transport = FakeProbeTransport(
        [
            FakeResponse(version_bytes, url=OLLAMA_VERSION_ENDPOINT),
            FakeResponse(show_bytes, url=OLLAMA_SHOW_ENDPOINT),
        ]
    )
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        probe_owned_ollama_runtime(
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=fingerprint,
            transport=transport,
        )


def test_runtime_probe_failure_is_sanitized_and_never_retried_or_pulled() -> None:
    digest, fingerprint, version_bytes, show_bytes = _probe_context()
    transport = FakeProbeTransport(
        [
            FakeResponse(version_bytes, url=OLLAMA_VERSION_ENDPOINT),
            FakeResponse(show_bytes, url=OLLAMA_SHOW_ENDPOINT),
        ],
        failure_at=2,
    )
    with pytest.raises(SecFilingGemmaOllamaError, match="runtime probe failed") as error:
        probe_owned_ollama_runtime(
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=fingerprint,
            transport=transport,
        )
    assert "private" not in str(error.value)
    assert error.value.__cause__ is None
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.update(model_calls=1),
            "model_calls",
        ),
        (
            lambda value: value.update(trusted_production_transport=True),
            "cannot self-attest",
        ),
        (
            lambda value: value.update(model_digest=_digest("forged")),
            "model digest",
        ),
        (
            lambda value: value.update(show_response_content_type="text/plain"),
            "content type",
        ),
        (
            lambda value: value.update(
                transport_mode="owned_hardened_loopback_runtime_probe_unattested"
            ),
            "transport mode",
        ),
    ],
)
def test_rehashed_runtime_probe_receipt_mutations_are_rejected(
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    transport, digest, fingerprint = _probe_transport()
    receipt = probe_owned_ollama_runtime(
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        transport=transport,
    )
    manifest = receipt.to_manifest()
    mutation(manifest)
    manifest["receipt_sha256"] = canonical_sha256(
        {key: manifest[key] for key in manifest if key != "receipt_sha256"}
    )
    with pytest.raises(SecFilingGemmaOllamaError, match=match):
        validate_ollama_runtime_probe_receipt(
            manifest,
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=fingerprint,
            expected_transport_mode=receipt.transport_mode,
        )


def test_runtime_probe_response_byte_mutation_cannot_replay_under_old_pin() -> None:
    transport, digest, fingerprint = _probe_transport()
    receipt = probe_owned_ollama_runtime(
        expected_model_digest=digest,
        expected_runtime_fingerprint_sha256=fingerprint,
        transport=transport,
    )
    manifest = receipt.to_manifest()
    changed = _json_bytes({"version": "0.12.4"})
    manifest["version_response_bytes_base64"] = base64.b64encode(changed).decode(
        "ascii"
    )
    manifest["version_response_sha256"] = hashlib.sha256(changed).hexdigest()
    manifest["version_response_content_length"] = len(changed)
    manifest["receipt_sha256"] = canonical_sha256(
        {key: manifest[key] for key in manifest if key != "receipt_sha256"}
    )
    with pytest.raises(SecFilingGemmaOllamaError, match="fingerprint"):
        validate_ollama_runtime_probe_receipt(
            manifest,
            expected_model_digest=digest,
            expected_runtime_fingerprint_sha256=fingerprint,
            expected_transport_mode=receipt.transport_mode,
        )
