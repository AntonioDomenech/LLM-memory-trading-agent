"""Frozen SEC/Gemma content-risk inputs and resumable local-model calls.

This module deliberately keeps the two worlds apart:

* :func:`prepare_requests` authenticates private SEC bytes and builds the
  frozen anonymous Gemma payloads in memory.
* the checkpoint helpers persist only public-safe hashes, metadata, timings,
  statuses, and schema-valid extractor JSON.  Prompt text, SEC text, local
  paths, and contact details are never written to those checkpoints.

There is no market-data dependency here and there is exactly one HTTP request
per selected model ordinal.  Failed calls and invalid outputs are permanent
rows; callers must interpret them as unavailable (stay LONG).
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
import time
from typing import Any, Final, Protocol

import requests

from agent_benchmark.sec_filing_content import normalize_filing_text
from agent_benchmark.sec_filing_gemma_contract import (
    CANONICAL_IDENTITY_LEXICON,
    EXTRACTOR_SYSTEM_PROMPT,
    build_extractor_json_schema,
    build_extractor_model_payload,
    validate_extractor_output,
)
from agent_benchmark.sec_filing_gemma_preprocessor import preprocess_filing_event
from agent_benchmark.sec_gemma_lean_science_v319_contract import (
    build_direction_sanitized_preprocessed_event,
    canonical_json_bytes,
    validate_direction_sanitized_preprocessed_event,
)


MODEL_NAME: Final[str] = "gemma4:12b"
MODEL_CONTEXT_TOKENS: Final[int] = 8_192
MODEL_OUTPUT_TOKENS: Final[int] = 1_024
MODEL_MANIFEST_SHA256: Final[str] = (
    "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
)
SEMANTIC_RUNTIME_FINGERPRINT_SHA256: Final[str] = (
    "816a7c1a6b1e87d083f8e0f85654f80ba0db124bf09fe8dedce960c8124e1c77"
)
SYSTEM_PROMPT_SHA256: Final[str] = (
    "9ed8496ed101c138cdbee162bdf6dfd53434f6f1e0c64fc93d844405d6eae9f7"
)
OUTPUT_SCHEMA_SHA256: Final[str] = (
    "1707ae581abb1a256dfb1ee8f51efd9dd67df5d9b3e8f7c22ee2d92b67b6f82b"
)

CONTEXT_SEQUENCES: Final[tuple[int, ...]] = (122, 123)
EVENT_SEQUENCES: Final[tuple[int, ...]] = tuple(range(124, 199))
PILOT_ORDINALS: Final[tuple[int, ...]] = (1, 15, 30, 45, 60, 75)
EXPECTED_DOCUMENT_COUNT: Final[int] = 77
EXPECTED_REQUEST_COUNT: Final[int] = 75
EXPECTED_REQUEST_SIZE_RANGE: Final[tuple[int, int]] = (14_577, 23_921)
EXPECTED_SENTENCE_COUNT_RANGE: Final[tuple[int, int]] = (70, 72)
EXPECTED_REMOVED_CURRENT: Final[int] = 4
EXPECTED_REMOVED_PRIOR: Final[int] = 4
EXPECTED_COMMITMENT_BYTES: Final[int] = 29_101
EXPECTED_COMMITMENT_SHA256: Final[str] = (
    "dda7adb2fbd5f662b9abadc8122d7ae03fb19128951351369ba40087a67672c8"
)
EXPECTED_ORDERED_PAYLOAD_HASH_SHA256: Final[str] = (
    "aed8bd8695622132a4c53a04c514c04a00f72835c69de2828256c99a0d3899dd"
)

OLLAMA_CHAT_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/chat"
OLLAMA_VERSION_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/version"
OLLAMA_TAGS_ENDPOINT: Final[str] = "http://127.0.0.1:11434/api/tags"
CONNECT_TIMEOUT_SECONDS: Final[float] = 2.0
READ_TIMEOUT_SECONDS: Final[float] = 600.0
MAX_RUNTIME_RESPONSE_BYTES: Final[int] = 2 * 1024 * 1024
MAX_MODEL_RESPONSE_BYTES: Final[int] = 256 * 1024

INPUT_COMMITMENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-content-risk-v1-input-commitment-v1"
)
MODEL_IDENTITY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-content-risk-v1-ollama-identity-v1"
)
CALL_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-content-risk-v1-model-call-v1"
)
ATTEMPT_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-content-risk-v1-model-call-attempt-v1"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")
_RECEIPT_NAME_RE = re.compile(r"([0-9]{6})-([0-9a-f]{64})\.json\Z")
_BLOB_NAME_RE = re.compile(
    r"([0-9]{6})-development-([0-9a-f]{64})\.blob\Z"
)
_OLLAMA_VERSION_RE = re.compile(
    r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z][0-9A-Za-z.-]{0,126})?\Z"
)
_ABSOLUTE_WINDOWS_PATH_RE = re.compile(r"(?i)(?:^|[^A-Za-z0-9])[A-Z]:[\\/]")
_ABSOLUTE_POSIX_PATH_RE = re.compile(r"(?:^|[\s\"'])(?:/Users/|/home/)")
_CHECKPOINT_FORBIDDEN_KEYS = frozenset(
    {
        "messages",
        "model_payload",
        "prompt",
        "prompt_text",
        "request_bytes",
        "sentences",
        "text",
        "private_path",
        "contact",
        "sec_contact",
        "user_agent",
    }
)
_OLLAMA_RESPONSE_KEYS = frozenset(
    {
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
)
_DANGEROUS_OUTPUT_EXTRA_KEYS = frozenset(
    {
        "audio",
        "content_parts",
        "error",
        "function_call",
        "images",
        "response",
        "thinking",
        "tool_call",
        "tool_calls",
    }
)
_MODEL_TIMING_KEYS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


class SecGemmaContentRiskInputError(RuntimeError):
    """Private input, local runtime, or checkpoint contract failed closed."""


class ResponseLike(Protocol):
    status_code: int
    url: str
    history: Sequence[Any]

    def iter_content(self, chunk_size: int) -> Any: ...

    def close(self) -> None: ...


class TransportLike(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> ResponseLike: ...


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """One authenticated request; prompt-bearing fields stay memory-only."""

    ordinal: int
    sequence: int
    accession_number: str
    form: str
    availability_session: str
    prior_same_form_sequence: int
    request_sha256: str
    model_payload_sha256: str
    request_byte_count: int
    supplied_sentence_ids: tuple[str, ...]
    source_preprocessed_event_sha256: str
    sanitized_preprocessed_event_sha256: str
    removed_current_sentence_count: int
    removed_prior_sentence_count: int
    retained_sentence_count: int
    current_complete_response_sha256: str
    current_selected_text_sha256: str
    current_normalized_text_sha256: str
    prior_normalized_text_sha256: str
    request_bytes: bytes = field(repr=False)

    def public_commitment_record(self) -> dict[str, Any]:
        """Return metadata only: never the payload or source text."""

        return {
            "ordinal": self.ordinal,
            "sequence": self.sequence,
            "accession_number": self.accession_number,
            "form": self.form,
            "availability_session": self.availability_session,
            "prior_same_form_sequence": self.prior_same_form_sequence,
            "request_sha256": self.request_sha256,
            "request_byte_count": self.request_byte_count,
            "model_payload_sha256": self.model_payload_sha256,
            "source_preprocessed_event_sha256": (
                self.source_preprocessed_event_sha256
            ),
            "sanitized_preprocessed_event_sha256": (
                self.sanitized_preprocessed_event_sha256
            ),
            "retained_sentence_count": self.retained_sentence_count,
            "removed_current_sentence_count": self.removed_current_sentence_count,
            "removed_prior_sentence_count": self.removed_prior_sentence_count,
            "current_complete_response_sha256": (
                self.current_complete_response_sha256
            ),
            "current_selected_text_sha256": self.current_selected_text_sha256,
            "current_normalized_text_sha256": self.current_normalized_text_sha256,
            "prior_normalized_text_sha256": self.prior_normalized_text_sha256,
        }


@dataclass(frozen=True, slots=True)
class _VerifiedDocument:
    sequence: int
    accession_number: str
    form: str
    availability_session: str
    complete_response_sha256: str
    selected_text_sha256: str
    normalized_text_sha256: str
    normalized_text: str = field(repr=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _bare_sha256(value: Any, location: str) -> str:
    if type(value) is not str:
        raise SecGemmaContentRiskInputError(f"{location} is not a SHA-256 digest")
    match = _TAGGED_SHA256_RE.fullmatch(value)
    digest = match.group(1) if match else value
    if _SHA256_RE.fullmatch(digest) is None:
        raise SecGemmaContentRiskInputError(f"{location} is not a SHA-256 digest")
    return digest


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecGemmaContentRiskInputError(f"{location} is not a valid integer")
    return value


def _strict_string(value: Any, location: str) -> str:
    if type(value) is not str or not value:
        raise SecGemmaContentRiskInputError(f"{location} is not a non-empty string")
    return value


def _reject_duplicate_pairs(location: str) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecGemmaContentRiskInputError(
                    f"{location} contains duplicate JSON keys"
                )
            result[key] = value
        return result

    return hook


def _reject_nonfinite(value: str) -> None:
    raise SecGemmaContentRiskInputError(f"non-finite JSON value {value!r} is forbidden")


def _strict_json_loads(value: bytes | str, *, location: str) -> Any:
    try:
        text = value.decode("utf-8", errors="strict") if isinstance(value, bytes) else value
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs(location),
            parse_constant=_reject_nonfinite,
        )
    except SecGemmaContentRiskInputError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise SecGemmaContentRiskInputError(f"{location} is not strict UTF-8 JSON") from exc


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(type(key) is str for key in value):
        raise SecGemmaContentRiskInputError(f"{location} is not a string-keyed object")
    return value


def _receipt_for_sequence(receipts_dir: Path, sequence: int) -> Path:
    matches = list(receipts_dir.glob(f"{sequence:06d}-*.json"))
    if len(matches) != 1:
        raise SecGemmaContentRiskInputError(
            f"sequence {sequence} must have exactly one parse receipt"
        )
    path = matches[0]
    name_match = _RECEIPT_NAME_RE.fullmatch(path.name)
    if name_match is None or int(name_match.group(1)) != sequence:
        raise SecGemmaContentRiskInputError("parse receipt filename is not canonical")
    if path.is_symlink() or not path.is_file():
        raise SecGemmaContentRiskInputError("parse receipt must be a regular file")
    return path


def _verify_document(private_root: Path, sequence: int) -> _VerifiedDocument:
    receipt_path = _receipt_for_sequence(private_root / "parse_receipts", sequence)
    receipt_bytes = receipt_path.read_bytes()
    name_hash = _RECEIPT_NAME_RE.fullmatch(receipt_path.name)
    assert name_hash is not None
    if _sha256_bytes(receipt_bytes) != name_hash.group(2):
        raise SecGemmaContentRiskInputError("parse receipt filename hash mismatch")
    receipt = _expect_mapping(
        _strict_json_loads(receipt_bytes, location="parse receipt"), "parse receipt"
    )
    if receipt.get("sequence") != sequence or receipt.get("stage") != "development":
        raise SecGemmaContentRiskInputError("parse receipt sequence/stage mismatch")

    evidence = _expect_mapping(receipt.get("source_evidence"), "source_evidence")
    seal = _expect_mapping(evidence.get("seal_row"), "source_evidence.seal_row")
    frozen_prefix = _expect_mapping(seal.get("frozen_prefix"), "seal_row.frozen_prefix")
    frozen_prefix_hash = _bare_sha256(
        evidence.get("seal_row", {}).get("frozen_prefix_sha256"),
        "seal_row.frozen_prefix_sha256",
    )
    if _sha256_bytes(canonical_json_bytes(frozen_prefix)) != frozen_prefix_hash:
        raise SecGemmaContentRiskInputError("frozen SEC prefix hash mismatch")

    blob_name = _strict_string(receipt.get("blob_name"), "blob_name")
    blob_match = _BLOB_NAME_RE.fullmatch(blob_name)
    if blob_match is None or int(blob_match.group(1)) != sequence:
        raise SecGemmaContentRiskInputError("blob filename is not canonical")
    blob_path = private_root / "blobs" / blob_name
    if blob_path.is_symlink() or not blob_path.is_file():
        raise SecGemmaContentRiskInputError("SEC blob must be a regular file")
    blob = blob_path.read_bytes()
    complete = _expect_mapping(seal.get("complete_response"), "complete_response")
    complete_hash = _bare_sha256(complete.get("sha256"), "complete_response.sha256")
    body_hash = _bare_sha256(receipt.get("body_sha256"), "body_sha256")
    body_length = _strict_int(receipt.get("body_bytes"), "body_bytes")
    complete_length = _strict_int(complete.get("length"), "complete_response.length")
    observed_blob_hash = _sha256_bytes(blob)
    if (
        len(blob) != body_length
        or len(blob) != complete_length
        or observed_blob_hash != body_hash
        or observed_blob_hash != complete_hash
        or observed_blob_hash != blob_match.group(2)
    ):
        raise SecGemmaContentRiskInputError("complete SEC response verification failed")

    selected = _expect_mapping(seal.get("extracted_text"), "extracted_text")
    start = _strict_int(selected.get("start_byte"), "extracted_text.start_byte")
    end = _strict_int(selected.get("end_byte"), "extracted_text.end_byte")
    length = _strict_int(selected.get("length"), "extracted_text.length")
    selected_hash = _bare_sha256(selected.get("sha256"), "extracted_text.sha256")
    if end < start or end > len(blob) or end - start != length:
        raise SecGemmaContentRiskInputError("selected SEC text bounds are invalid")
    selected_bytes = blob[start:end]
    if _sha256_bytes(selected_bytes) != selected_hash:
        raise SecGemmaContentRiskInputError("selected SEC text hash mismatch")
    if selected.get("encoding") != "latin-1 exact one-to-one response slice":
        raise SecGemmaContentRiskInputError("selected SEC text encoding changed")

    normalized = normalize_filing_text(selected_bytes.decode("latin-1"))
    normalized_seal = _expect_mapping(seal.get("normalized_text"), "normalized_text")
    normalized_hash = _bare_sha256(
        normalized_seal.get("sha256"), "normalized_text.sha256"
    )
    normalized_bytes = normalized.text.encode("utf-8")
    if (
        _bare_sha256(normalized.sha256, "reconstructed normalized SHA-256")
        != normalized_hash
        or len(normalized_bytes)
        != _strict_int(normalized_seal.get("length"), "normalized_text.length")
        or normalized.character_count
        != _strict_int(
            normalized_seal.get("character_count"),
            "normalized_text.character_count",
        )
    ):
        raise SecGemmaContentRiskInputError("normalized SEC text verification failed")
    if not normalized.usable:
        raise SecGemmaContentRiskInputError("authenticated SEC document is unusable")

    accession = _strict_string(seal.get("accession_number"), "accession_number")
    form = _strict_string(seal.get("form"), "form")
    availability = _strict_string(
        seal.get("availability_session"), "availability_session"
    )
    if form not in {"10-K", "10-Q"}:
        raise SecGemmaContentRiskInputError("filing form is outside the frozen set")
    return _VerifiedDocument(
        sequence=sequence,
        accession_number=accession,
        form=form,
        availability_session=availability,
        complete_response_sha256=complete_hash,
        selected_text_sha256=selected_hash,
        normalized_text_sha256=normalized_hash,
        normalized_text=normalized.text,
    )


def _payload_hash_commitment(requests: Sequence[PreparedRequest]) -> str:
    # The ordered list, rather than a set, binds every payload to its ordinal.
    return _sha256_bytes(
        canonical_json_bytes([item.model_payload_sha256 for item in requests])
    )


def build_content_risk_model_payload(
    sentences: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Build the frozen extractor payload with the corrected completion budget."""

    payload = build_extractor_model_payload(sentences)
    payload["options"]["num_ctx"] = MODEL_CONTEXT_TOKENS
    payload["options"]["num_predict"] = MODEL_OUTPUT_TOKENS
    return payload


def input_commitment_bytes(requests: Sequence[PreparedRequest]) -> bytes:
    """Replay the exact preregistered 29,101-byte public commitment layout."""

    rows = [
        {
            "sequence": item.sequence,
            "accession_sha256": _sha256_bytes(item.accession_number.encode("utf-8")),
            "availability_session": item.availability_session,
            "model_payload_sha256": item.model_payload_sha256,
            # This is a byte *count*.  It never contains request bytes.
            "request_bytes": item.request_byte_count,
            "request_sha256": item.request_sha256,
            "sentences": item.retained_sentence_count,
            "removed_current": item.removed_current_sentence_count,
            "removed_prior": item.removed_prior_sentence_count,
        }
        for item in requests
    ]
    return canonical_json_bytes(rows)


def prepare_requests(
    private_root: Path,
    *,
    verify_frozen: bool = True,
    context_sequences: Sequence[int] = CONTEXT_SEQUENCES,
    event_sequences: Sequence[int] = EVENT_SEQUENCES,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[PreparedRequest]:
    """Authenticate V3.8 bytes and build current-plus-prior anonymous payloads."""

    root = Path(private_root)
    if root.is_symlink() or not root.is_dir():
        raise SecGemmaContentRiskInputError("private V3.8 development root is missing")
    context = tuple(context_sequences)
    events = tuple(event_sequences)
    required = tuple(sorted(set(context) | set(events)))
    if not required or any(type(item) is not int or item < 0 for item in required):
        raise SecGemmaContentRiskInputError("input sequences are invalid")
    if len(events) != len(set(events)):
        raise SecGemmaContentRiskInputError("event sequences must be unique")

    documents: dict[int, _VerifiedDocument] = {}
    for completed, sequence in enumerate(required, start=1):
        documents[sequence] = _verify_document(root, sequence)
        if progress is not None:
            progress(
                {
                    "stage": "authenticate_input",
                    "sequence": sequence,
                    "completed": completed,
                    "total": len(required),
                }
            )

    ordered_documents = sorted(documents.values(), key=lambda item: item.sequence)
    requests_out: list[PreparedRequest] = []
    for ordinal, sequence in enumerate(events, start=1):
        current = documents.get(sequence)
        if current is None:
            raise SecGemmaContentRiskInputError("event sequence has no document")
        eligible_priors = [
            item
            for item in ordered_documents
            if item.form == current.form
            and item.sequence < current.sequence
        ]
        if not eligible_priors:
            raise SecGemmaContentRiskInputError(
                f"sequence {sequence} lacks a strictly earlier same-form filing"
            )
        prior = max(eligible_priors, key=lambda item: item.sequence)
        if prior.sequence >= current.sequence:
            raise SecGemmaContentRiskInputError("prior same-form chronology is invalid")

        source = preprocess_filing_event(
            current_normalized_text=current.normalized_text,
            prior_same_form_normalized_text=prior.normalized_text,
            identity_lexicon=CANONICAL_IDENTITY_LEXICON,
        )
        sanitized = build_direction_sanitized_preprocessed_event(source)
        sanitized = validate_direction_sanitized_preprocessed_event(
            source,
            sanitized,
            expected_source_preprocessed_event_sha256=source[
                "preprocessed_event_sha256"
            ],
        )
        model_payload = build_content_risk_model_payload(sanitized["sentences"])
        request_bytes = canonical_json_bytes(model_payload)
        request_hash = _sha256_bytes(request_bytes)
        payload_hash = request_hash
        if request_hash != payload_hash:
            raise SecGemmaContentRiskInputError("model payload hash mismatch")
        sentence_ids = tuple(sentence["id"] for sentence in sanitized["sentences"])
        prepared = PreparedRequest(
            ordinal=ordinal,
            sequence=sequence,
            accession_number=current.accession_number,
            form=current.form,
            availability_session=current.availability_session,
            prior_same_form_sequence=prior.sequence,
            request_sha256=request_hash,
            model_payload_sha256=payload_hash,
            request_byte_count=len(request_bytes),
            supplied_sentence_ids=sentence_ids,
            source_preprocessed_event_sha256=source["preprocessed_event_sha256"],
            sanitized_preprocessed_event_sha256=sanitized[
                "preprocessed_event_sha256"
            ],
            removed_current_sentence_count=sanitized[
                "removed_current_sentence_count"
            ],
            removed_prior_sentence_count=sanitized["removed_prior_sentence_count"],
            retained_sentence_count=len(sentence_ids),
            current_complete_response_sha256=current.complete_response_sha256,
            current_selected_text_sha256=current.selected_text_sha256,
            current_normalized_text_sha256=current.normalized_text_sha256,
            prior_normalized_text_sha256=prior.normalized_text_sha256,
            request_bytes=request_bytes,
        )
        _validate_prepared_request(prepared)
        requests_out.append(prepared)
        if progress is not None:
            progress(
                {
                    "stage": "build_request",
                    "ordinal": ordinal,
                    "sequence": sequence,
                    "completed": ordinal,
                    "total": len(events),
                }
            )

    if verify_frozen:
        if context != CONTEXT_SEQUENCES or events != EVENT_SEQUENCES:
            raise SecGemmaContentRiskInputError("frozen sequence set changed")
        commitment = input_commitment_bytes(requests_out)
        observed_sizes = [item.request_byte_count for item in requests_out]
        observed_counts = [item.retained_sentence_count for item in requests_out]
        frozen_checks = (
            len(documents) == EXPECTED_DOCUMENT_COUNT,
            len(requests_out) == EXPECTED_REQUEST_COUNT,
            (min(observed_sizes), max(observed_sizes))
            == EXPECTED_REQUEST_SIZE_RANGE,
            (min(observed_counts), max(observed_counts))
            == EXPECTED_SENTENCE_COUNT_RANGE,
            sum(item.removed_current_sentence_count for item in requests_out)
            == EXPECTED_REMOVED_CURRENT,
            sum(item.removed_prior_sentence_count for item in requests_out)
            == EXPECTED_REMOVED_PRIOR,
            len(commitment) == EXPECTED_COMMITMENT_BYTES,
            _sha256_bytes(commitment) == EXPECTED_COMMITMENT_SHA256,
            _payload_hash_commitment(requests_out)
            == EXPECTED_ORDERED_PAYLOAD_HASH_SHA256,
        )
        if not all(frozen_checks):
            raise SecGemmaContentRiskInputError(
                "reconstructed inputs differ from the frozen v1 commitment"
            )
    return requests_out


def _validate_prepared_request(item: PreparedRequest) -> None:
    if type(item) is not PreparedRequest:
        raise SecGemmaContentRiskInputError("request is not a PreparedRequest")
    if _sha256_bytes(item.request_bytes) != _bare_sha256(
        item.request_sha256, "request_sha256"
    ):
        raise SecGemmaContentRiskInputError("prepared request bytes changed")
    if item.request_byte_count != len(item.request_bytes):
        raise SecGemmaContentRiskInputError("prepared request byte count changed")
    payload = _expect_mapping(
        _strict_json_loads(item.request_bytes, location="prepared request"),
        "prepared request",
    )
    if _sha256_bytes(canonical_json_bytes(payload)) != _bare_sha256(
        item.model_payload_sha256, "model_payload_sha256"
    ):
        raise SecGemmaContentRiskInputError("prepared model payload changed")
    messages = payload.get("messages")
    if type(messages) is not list or len(messages) != 2:
        raise SecGemmaContentRiskInputError("prepared messages changed")
    system = _expect_mapping(messages[0], "system message")
    user = _expect_mapping(messages[1], "user message")
    if (
        payload.get("model") != MODEL_NAME
        or system != {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT}
        or user.get("role") != "user"
        or type(user.get("content")) is not str
    ):
        raise SecGemmaContentRiskInputError("prepared fixed model request changed")
    supplied = _expect_mapping(
        _strict_json_loads(user["content"], location="anonymous sentence payload"),
        "anonymous sentence payload",
    )
    sentences = supplied.get("sentences")
    if type(sentences) is not list:
        raise SecGemmaContentRiskInputError("anonymous sentences are missing")
    rebuilt = build_content_risk_model_payload(sentences)
    if canonical_json_bytes(rebuilt) != item.request_bytes:
        raise SecGemmaContentRiskInputError("prepared request is not canonical")
    sentence_ids = tuple(sentence.get("id") for sentence in sentences)
    if sentence_ids != item.supplied_sentence_ids:
        raise SecGemmaContentRiskInputError("prepared sentence IDs changed")
    if _sha256_bytes(EXTRACTOR_SYSTEM_PROMPT.encode("utf-8")) != SYSTEM_PROMPT_SHA256:
        raise SecGemmaContentRiskInputError("system-prompt pin changed")
    if _sha256_bytes(canonical_json_bytes(build_extractor_json_schema())) != OUTPUT_SCHEMA_SHA256:
        raise SecGemmaContentRiskInputError("extractor-schema pin changed")


class _RequestsTransport:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.trust_env = False

    def request(self, method: str, url: str, **kwargs: Any) -> ResponseLike:
        return self.session.request(method, url, **kwargs)

    def close(self) -> None:
        self.session.close()


def _response_bytes(response: ResponseLike, *, limit: int, expected_url: str) -> bytes:
    try:
        if getattr(response, "url", expected_url) != expected_url:
            raise SecGemmaContentRiskInputError("local response URL changed")
        if getattr(response, "history", ()):
            raise SecGemmaContentRiskInputError("local redirects are forbidden")
        if type(response.status_code) is not int or response.status_code != 200:
            raise SecGemmaContentRiskInputError("local endpoint returned non-200")
        chunks: list[bytes] = []
        size = 0
        if hasattr(response, "iter_content"):
            iterator = response.iter_content(chunk_size=16 * 1024)
        else:  # Small fake transports may expose only ``content``.
            iterator = (getattr(response, "content", b""),)
        for chunk in iterator:
            if not isinstance(chunk, bytes):
                raise SecGemmaContentRiskInputError("local response chunk is not bytes")
            size += len(chunk)
            if size > limit:
                raise SecGemmaContentRiskInputError("local response is too large")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        response.close()


def _runtime_get(transport: TransportLike, endpoint: str) -> tuple[Mapping[str, Any], str]:
    response = transport.request(
        "GET",
        endpoint,
        headers={"Accept": "application/json", "Accept-Encoding": "identity"},
        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        allow_redirects=False,
        stream=True,
    )
    raw = _response_bytes(
        response, limit=MAX_RUNTIME_RESPONSE_BYTES, expected_url=endpoint
    )
    value = _expect_mapping(
        _strict_json_loads(raw, location="local runtime response"),
        "local runtime response",
    )
    return value, _sha256_bytes(raw)


def inspect_model_identity(
    *, transport: TransportLike | None = None
) -> dict[str, Any]:
    """Record Ollama's version and require the frozen installed model digest."""

    owned = transport is None
    active: TransportLike = _RequestsTransport() if transport is None else transport
    try:
        version_value, version_response_sha = _runtime_get(
            active, OLLAMA_VERSION_ENDPOINT
        )
        tags_value, tags_response_sha = _runtime_get(active, OLLAMA_TAGS_ENDPOINT)
    finally:
        if owned:
            assert isinstance(active, _RequestsTransport)
            active.close()
    version = version_value.get("version")
    if type(version) is not str or _OLLAMA_VERSION_RE.fullmatch(version) is None:
        raise SecGemmaContentRiskInputError("Ollama version is invalid")
    models = tags_value.get("models")
    if type(models) is not list:
        raise SecGemmaContentRiskInputError("Ollama tags response has no model list")
    matches = [
        item
        for item in models
        if isinstance(item, Mapping)
        and (item.get("name") == MODEL_NAME or item.get("model") == MODEL_NAME)
    ]
    if len(matches) != 1:
        raise SecGemmaContentRiskInputError("frozen Ollama model tag is not unique")
    observed_digest = _bare_sha256(matches[0].get("digest"), "model digest")
    if observed_digest != MODEL_MANIFEST_SHA256:
        raise SecGemmaContentRiskInputError("installed model manifest digest changed")
    return {
        "schema_version": MODEL_IDENTITY_SCHEMA_VERSION,
        "ollama_version": version,
        "model_name": MODEL_NAME,
        "model_manifest_sha256": observed_digest,
        "semantic_runtime_fingerprint_sha256": (
            SEMANTIC_RUNTIME_FINGERPRINT_SHA256
        ),
        "version_response_sha256": version_response_sha,
        "tags_response_sha256": tags_response_sha,
    }


def _validate_identity_checkpoint(value: Any) -> dict[str, Any]:
    identity = dict(_expect_mapping(value, "runtime identity checkpoint"))
    _assert_public_safe(identity)
    expected_keys = {
        "schema_version",
        "ollama_version",
        "model_name",
        "model_manifest_sha256",
        "semantic_runtime_fingerprint_sha256",
        "version_response_sha256",
        "tags_response_sha256",
    }
    if (
        set(identity) != expected_keys
        or identity.get("schema_version") != MODEL_IDENTITY_SCHEMA_VERSION
        or type(identity.get("ollama_version")) is not str
        or _OLLAMA_VERSION_RE.fullmatch(identity["ollama_version"]) is None
        or identity.get("model_name") != MODEL_NAME
        or identity.get("model_manifest_sha256") != MODEL_MANIFEST_SHA256
        or identity.get("semantic_runtime_fingerprint_sha256")
        != SEMANTIC_RUNTIME_FINGERPRINT_SHA256
    ):
        raise SecGemmaContentRiskInputError("runtime identity checkpoint changed")
    _bare_sha256(identity.get("version_response_sha256"), "version response hash")
    _bare_sha256(identity.get("tags_response_sha256"), "tags response hash")
    return identity


def _bind_identity_checkpoint(path: Path, current: Mapping[str, Any]) -> None:
    validated_current = _validate_identity_checkpoint(current)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise SecGemmaContentRiskInputError("runtime identity checkpoint is invalid")
        stored = _validate_identity_checkpoint(
            _strict_json_loads(path.read_bytes(), location="runtime identity checkpoint")
        )
        semantic_fields = (
            "schema_version",
            "ollama_version",
            "model_name",
            "model_manifest_sha256",
            "semantic_runtime_fingerprint_sha256",
        )
        if any(stored[field] != validated_current[field] for field in semantic_fields):
            raise SecGemmaContentRiskInputError(
                "resumed batch runtime identity differs from its first observation"
            )
        return
    _atomic_write_json(path, validated_current)


def _validate_ollama_envelope(value: Any) -> tuple[str, dict[str, int]]:
    envelope = _expect_mapping(value, "Ollama model response")
    envelope_keys = set(envelope)
    if not _OLLAMA_RESPONSE_KEYS.issubset(envelope_keys) or (
        (envelope_keys - _OLLAMA_RESPONSE_KEYS) & _DANGEROUS_OUTPUT_EXTRA_KEYS
    ):
        raise SecGemmaContentRiskInputError("Ollama response envelope is unsafe")
    message = _expect_mapping(envelope.get("message"), "Ollama message")
    message_keys = set(message)
    if (
        envelope.get("model") != MODEL_NAME
        or envelope.get("done") is not True
        or envelope.get("done_reason") != "stop"
        or not {"role", "content"}.issubset(message_keys)
        or ((message_keys - {"role", "content"}) & _DANGEROUS_OUTPUT_EXTRA_KEYS)
        or message.get("role") != "assistant"
        or type(message.get("content")) is not str
    ):
        raise SecGemmaContentRiskInputError("Ollama completion is not final")
    timing: dict[str, int] = {}
    for key in _MODEL_TIMING_KEYS:
        timing[key] = _strict_int(envelope.get(key), f"Ollama {key}")
    return message["content"], timing


def _call_model_once(
    item: PreparedRequest,
    *,
    transport: TransportLike,
    clock: Callable[[], float],
) -> dict[str, Any]:
    _validate_prepared_request(item)
    started = clock()
    try:
        response = transport.request(
            "POST",
            OLLAMA_CHAT_ENDPOINT,
            data=item.request_bytes,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Content-Type": "application/json",
            },
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            allow_redirects=False,
            stream=True,
        )
        response_bytes = _response_bytes(
            response,
            limit=MAX_MODEL_RESPONSE_BYTES,
            expected_url=OLLAMA_CHAT_ENDPOINT,
        )
    except Exception:
        elapsed = max(0.0, clock() - started)
        return _base_result(
            item,
            status="transport_error",
            elapsed_seconds=elapsed,
            extractor_output=None,
            raw_output_sha256=None,
            extractor_output_sha256=None,
            output_byte_count=None,
            exact_output_json_utf8_base64=None,
            model_timing=None,
            reason="local_call_failed_no_retry",
        )

    elapsed = max(0.0, clock() - started)
    output_bytes: bytes | None = None
    timing: dict[str, int] | None = None
    try:
        envelope = _strict_json_loads(response_bytes, location="Ollama response")
        output_text, timing = _validate_ollama_envelope(envelope)
        output_bytes = output_text.encode("utf-8")
        output = _expect_mapping(
            _strict_json_loads(output_bytes, location="extractor output"),
            "extractor output",
        )
        validate_extractor_output(
            output, supplied_sentence_ids=item.supplied_sentence_ids
        )
        if output.get("document_quality") == "unusable":
            return _base_result(
                item,
                status="invalid",
                elapsed_seconds=elapsed,
                extractor_output=None,
                raw_output_sha256=_sha256_bytes(output_bytes),
                extractor_output_sha256=None,
                output_byte_count=len(output_bytes),
                exact_output_json_utf8_base64=None,
                model_timing=timing,
                reason="unusable_document_no_retry_no_repair",
            )
        detached_output = json.loads(json.dumps(output, allow_nan=False))
        encoded_exact = base64.b64encode(output_bytes).decode("ascii")
        result = _base_result(
            item,
            status="valid",
            elapsed_seconds=elapsed,
            extractor_output=detached_output,
            raw_output_sha256=_sha256_bytes(output_bytes),
            extractor_output_sha256=_sha256_bytes(canonical_json_bytes(output)),
            output_byte_count=len(output_bytes),
            exact_output_json_utf8_base64=encoded_exact,
            model_timing=timing,
            reason=None,
        )
        _assert_public_safe(result)
        return result
    except Exception:
        return _base_result(
            item,
            status="invalid",
            elapsed_seconds=elapsed,
            extractor_output=None,
            raw_output_sha256=(
                None if output_bytes is None else _sha256_bytes(output_bytes)
            ),
            extractor_output_sha256=None,
            output_byte_count=(None if output_bytes is None else len(output_bytes)),
            exact_output_json_utf8_base64=None,
            model_timing=timing,
            reason="invalid_json_schema_or_evidence_no_retry_no_repair",
        )


def _base_result(
    item: PreparedRequest,
    *,
    status: str,
    elapsed_seconds: float,
    extractor_output: Mapping[str, Any] | None,
    raw_output_sha256: str | None,
    extractor_output_sha256: str | None,
    output_byte_count: int | None,
    exact_output_json_utf8_base64: str | None,
    model_timing: Mapping[str, int] | None,
    reason: str | None,
) -> dict[str, Any]:
    if status not in {"valid", "invalid", "transport_error"}:
        raise SecGemmaContentRiskInputError("result status is invalid")
    return {
        "schema_version": CALL_CHECKPOINT_SCHEMA_VERSION,
        "ordinal": item.ordinal,
        "sequence": item.sequence,
        "accession_number": item.accession_number,
        "form": item.form,
        "availability_session": item.availability_session,
        "request_sha256": item.request_sha256,
        "supplied_sentence_ids": list(item.supplied_sentence_ids),
        "status": status,
        "extractor_output": (
            None if extractor_output is None else dict(extractor_output)
        ),
        "raw_output_sha256": raw_output_sha256,
        "extractor_output_sha256": extractor_output_sha256,
        "output_byte_count": output_byte_count,
        "exact_output_json_utf8_base64": exact_output_json_utf8_base64,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "model_timing": None if model_timing is None else dict(model_timing),
        "reason": reason,
    }


def _attempt_marker(item: PreparedRequest) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_CHECKPOINT_SCHEMA_VERSION,
        "ordinal": item.ordinal,
        "sequence": item.sequence,
        "accession_number": item.accession_number,
        "form": item.form,
        "availability_session": item.availability_session,
        "request_sha256": item.request_sha256,
        "supplied_sentence_ids": list(item.supplied_sentence_ids),
        "status": "in_progress",
    }


def _validate_attempt_marker(value: Any, item: PreparedRequest) -> dict[str, Any]:
    marker = dict(_expect_mapping(value, "model attempt checkpoint"))
    _assert_public_safe(marker)
    expected = _attempt_marker(item)
    if marker != expected:
        raise SecGemmaContentRiskInputError("model attempt checkpoint binding changed")
    return marker


def _assert_public_safe(value: Any, *, key: str | None = None) -> None:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if type(child_key) is not str:
                raise SecGemmaContentRiskInputError("public checkpoint key is not text")
            if child_key.casefold() in _CHECKPOINT_FORBIDDEN_KEYS:
                raise SecGemmaContentRiskInputError(
                    f"private field {child_key!r} is forbidden in public checkpoint"
                )
            _assert_public_safe(child, key=child_key)
    elif isinstance(value, list) or isinstance(value, tuple):
        for child in value:
            _assert_public_safe(child, key=key)
    elif isinstance(value, str):
        if _ABSOLUTE_WINDOWS_PATH_RE.search(value) or _ABSOLUTE_POSIX_PATH_RE.search(value):
            raise SecGemmaContentRiskInputError(
                "absolute private path is forbidden in public checkpoint"
            )
    elif value is not None and type(value) not in {bool, int, float}:
        raise SecGemmaContentRiskInputError("public checkpoint contains a non-JSON value")
    elif type(value) is float and not math.isfinite(value):
        raise SecGemmaContentRiskInputError("public checkpoint contains non-finite time")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _assert_public_safe(value)
    encoded = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise SecGemmaContentRiskInputError("checkpoint symlinks are forbidden")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _checkpoint_path(checkpoint_dir: Path, ordinal: int) -> Path:
    return checkpoint_dir / f"model-call-{ordinal:04d}.json"


def _validate_loaded_result(value: Any, item: PreparedRequest) -> dict[str, Any]:
    result = dict(_expect_mapping(value, "model checkpoint"))
    _assert_public_safe(result)
    expected_keys = set(
        _base_result(
            item,
            status="transport_error",
            elapsed_seconds=0.0,
            extractor_output=None,
            raw_output_sha256=None,
            extractor_output_sha256=None,
            output_byte_count=None,
            exact_output_json_utf8_base64=None,
            model_timing=None,
            reason="local_call_failed_no_retry",
        )
    )
    if set(result) != expected_keys:
        raise SecGemmaContentRiskInputError("model checkpoint fields changed")
    for key, expected in (
        ("schema_version", CALL_CHECKPOINT_SCHEMA_VERSION),
        ("ordinal", item.ordinal),
        ("sequence", item.sequence),
        ("accession_number", item.accession_number),
        ("form", item.form),
        ("availability_session", item.availability_session),
        ("request_sha256", item.request_sha256),
        ("supplied_sentence_ids", list(item.supplied_sentence_ids)),
    ):
        if result.get(key) != expected:
            raise SecGemmaContentRiskInputError("model checkpoint binding changed")
    status = result.get("status")
    if status not in {"valid", "invalid", "transport_error"}:
        raise SecGemmaContentRiskInputError("model checkpoint status changed")
    elapsed = result.get("elapsed_seconds")
    if type(elapsed) not in {int, float} or elapsed < 0 or not math.isfinite(elapsed):
        raise SecGemmaContentRiskInputError("model checkpoint timing is invalid")
    if status == "valid":
        output = _expect_mapping(result.get("extractor_output"), "extractor_output")
        validate_extractor_output(output, supplied_sentence_ids=item.supplied_sentence_ids)
        encoded = result.get("exact_output_json_utf8_base64")
        if type(encoded) is not str:
            raise SecGemmaContentRiskInputError("exact extractor JSON is missing")
        try:
            exact = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise SecGemmaContentRiskInputError("exact extractor JSON is invalid") from exc
        replay = _strict_json_loads(exact, location="checkpoint extractor output")
        if replay != output or _sha256_bytes(exact) != result.get("raw_output_sha256"):
            raise SecGemmaContentRiskInputError("exact extractor JSON hash changed")
        if _sha256_bytes(canonical_json_bytes(output)) != result.get(
            "extractor_output_sha256"
        ):
            raise SecGemmaContentRiskInputError("canonical extractor output hash changed")
        if len(exact) != result.get("output_byte_count"):
            raise SecGemmaContentRiskInputError("exact extractor JSON size changed")
    elif (
        result.get("extractor_output") is not None
        or result.get("exact_output_json_utf8_base64") is not None
        or result.get("extractor_output_sha256") is not None
    ):
        raise SecGemmaContentRiskInputError("invalid call checkpoint contains raw output")
    return result


def load_model_results(
    checkpoint_dir: Path, requests: Sequence[PreparedRequest]
) -> list[dict[str, Any]]:
    """Load and authenticate completed public-safe per-call checkpoints."""

    directory = Path(checkpoint_dir)
    by_ordinal = {item.ordinal: item for item in requests}
    if len(by_ordinal) != len(requests):
        raise SecGemmaContentRiskInputError("request ordinals must be unique")
    results: list[dict[str, Any]] = []
    if not directory.exists():
        return results
    if directory.is_symlink() or not directory.is_dir():
        raise SecGemmaContentRiskInputError("checkpoint directory is invalid")
    for ordinal, item in sorted(by_ordinal.items()):
        path = _checkpoint_path(directory, ordinal)
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise SecGemmaContentRiskInputError("model checkpoint is not a regular file")
        value = _strict_json_loads(path.read_bytes(), location="model checkpoint")
        if isinstance(value, Mapping) and value.get("schema_version") == (
            ATTEMPT_CHECKPOINT_SCHEMA_VERSION
        ):
            _validate_attempt_marker(value, item)
            recovered = _base_result(
                item,
                status="transport_error",
                elapsed_seconds=0.0,
                extractor_output=None,
                raw_output_sha256=None,
                extractor_output_sha256=None,
                output_byte_count=None,
                exact_output_json_utf8_base64=None,
                model_timing=None,
                reason="interrupted_attempt_no_retry",
            )
            _atomic_write_json(path, recovered)
            results.append(recovered)
        else:
            results.append(_validate_loaded_result(value, item))
    return results


def _emit_progress(
    event: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None
) -> None:
    if progress is not None:
        progress(dict(event))
        return
    eta = event["estimated_remaining_seconds"]
    eta_text = "unknown" if eta is None else f"{eta:.0f}s"
    print(
        "Gemma "
        f"{event['ordinal']}/75 {event['status']} in {event['elapsed_seconds']:.1f}s; "
        f"completed {event['completed_count']}; "
        f"median {event['rolling_median_seconds']:.1f}s; ETA {eta_text}",
        flush=True,
    )


def _pilot_failed_only_before_output_extraction(
    completed: Mapping[int, Mapping[str, Any]],
) -> bool:
    """Identify the sealed legacy-wrapper failure that cannot affect semantics."""

    return all(
        ordinal in completed
        and completed[ordinal]["status"] == "invalid"
        and completed[ordinal].get("reason")
        == "invalid_json_schema_or_evidence_no_retry_no_repair"
        and completed[ordinal].get("raw_output_sha256") is None
        and completed[ordinal].get("output_byte_count") is None
        and completed[ordinal].get("model_timing") is None
        for ordinal in PILOT_ORDINALS
    )


def run_model_batch(
    requests: Sequence[PreparedRequest],
    checkpoint_dir: Path,
    ordinals: Sequence[int] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    *,
    transport: TransportLike | None = None,
    clock: Callable[[], float] = time.perf_counter,
    allow_sealed_preoutput_pilot_continuation: bool = False,
) -> list[dict[str, Any]]:
    """Run missing ordinals once, atomically checkpointing every outcome.

    With ``ordinals=None`` the fixed six-call pilot is run first.  The remaining
    calls run only after at least five pilot outputs are valid.  Passing an
    explicit ordinal list performs exactly that bounded selection, which is
    useful for an operator-driven pilot invocation and for tests.
    """

    items = list(requests)
    if not items:
        return []
    for item in items:
        _validate_prepared_request(item)
    by_ordinal = {item.ordinal: item for item in items}
    if len(by_ordinal) != len(items) or set(by_ordinal) != set(range(1, len(items) + 1)):
        raise SecGemmaContentRiskInputError("request ordinals are not contiguous")
    selected = (
        tuple(range(1, len(items) + 1)) if ordinals is None else tuple(ordinals)
    )
    if len(selected) != len(set(selected)) or any(
        type(item) is not int or item not in by_ordinal for item in selected
    ):
        raise SecGemmaContentRiskInputError("selected model ordinals are invalid")

    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    existing = load_model_results(directory, items)
    completed = {result["ordinal"]: result for result in existing}
    parser_only_continuation = bool(
        allow_sealed_preoutput_pilot_continuation
        and len(items) == EXPECTED_REQUEST_COUNT
        and _pilot_failed_only_before_output_extraction(completed)
    )

    if ordinals is not None and len(items) == EXPECTED_REQUEST_COUNT:
        nonpilot_selected = set(selected) - set(PILOT_ORDINALS)
        if nonpilot_selected:
            if not parser_only_continuation and (
                not all(ordinal in completed for ordinal in PILOT_ORDINALS)
                or sum(
                    completed[ordinal]["status"] == "valid"
                    for ordinal in PILOT_ORDINALS
                )
                < 5
            ):
                raise SecGemmaContentRiskInputError(
                    "nonpilot ordinals require a completed healthy fixed pilot"
                )
    owned = transport is None
    active: TransportLike = _RequestsTransport() if transport is None else transport
    try:
        identity_before = inspect_model_identity(transport=active)
        _bind_identity_checkpoint(directory / "runtime-before.json", identity_before)

        if ordinals is None and len(items) == EXPECTED_REQUEST_COUNT:
            pilot = [ordinal for ordinal in PILOT_ORDINALS if ordinal in by_ordinal]
            missing_pilot = [ordinal for ordinal in pilot if ordinal not in completed]
            order = missing_pilot
            if not missing_pilot:
                valid_pilot = sum(completed[item]["status"] == "valid" for item in pilot)
                order = [] if valid_pilot < 5 and not parser_only_continuation else [
                    ordinal for ordinal in selected if ordinal not in pilot
                ]
        else:
            order = list(selected)

        for ordinal in order:
            if ordinal in completed:
                continue
            _atomic_write_json(
                _checkpoint_path(directory, ordinal),
                _attempt_marker(by_ordinal[ordinal]),
            )
            result = _call_model_once(by_ordinal[ordinal], transport=active, clock=clock)
            _atomic_write_json(_checkpoint_path(directory, ordinal), result)
            completed[ordinal] = result
            durations = [
                float(value["elapsed_seconds"])
                for value in completed.values()
                if value["ordinal"] in selected
            ]
            median = statistics.median(durations)
            remaining_count = sum(item not in completed for item in selected)
            _emit_progress(
                {
                    "ordinal": ordinal,
                    "sequence": result["sequence"],
                    "status": result["status"],
                    "elapsed_seconds": float(result["elapsed_seconds"]),
                    "completed_count": sum(
                        item in completed for item in selected
                    ),
                    "rolling_median_seconds": float(median),
                    "estimated_remaining_seconds": float(median * remaining_count),
                },
                progress,
            )

            # A default all-75 invocation evaluates the pilot immediately after
            # its sixth checkpoint.  Invalid is permanent and never retried.
            if (
                ordinals is None
                and len(items) == EXPECTED_REQUEST_COUNT
                and all(item in completed for item in PILOT_ORDINALS)
                and ordinal in PILOT_ORDINALS
            ):
                valid_pilot = sum(
                    completed[item]["status"] == "valid" for item in PILOT_ORDINALS
                )
                if valid_pilot < 5:
                    break
                for remainder in selected:
                    if remainder not in PILOT_ORDINALS and remainder not in order:
                        order.append(remainder)

        identity_after = inspect_model_identity(transport=active)
        _bind_identity_checkpoint(directory / "runtime-after.json", identity_after)
        if (
            identity_before["model_manifest_sha256"]
            != identity_after["model_manifest_sha256"]
        ):
            raise SecGemmaContentRiskInputError("model manifest changed during batch")
    finally:
        if owned:
            assert isinstance(active, _RequestsTransport)
            active.close()
    return load_model_results(directory, items)


__all__ = [
    "CALL_CHECKPOINT_SCHEMA_VERSION",
    "EVENT_SEQUENCES",
    "EXPECTED_COMMITMENT_BYTES",
    "EXPECTED_COMMITMENT_SHA256",
    "EXPECTED_ORDERED_PAYLOAD_HASH_SHA256",
    "MODEL_MANIFEST_SHA256",
    "MODEL_CONTEXT_TOKENS",
    "MODEL_NAME",
    "MODEL_OUTPUT_TOKENS",
    "PILOT_ORDINALS",
    "PreparedRequest",
    "SecGemmaContentRiskInputError",
    "build_content_risk_model_payload",
    "input_commitment_bytes",
    "inspect_model_identity",
    "load_model_results",
    "prepare_requests",
    "run_model_batch",
]
