"""Bounded official-SEC corpus acquisition for the filing/Gemma experiment.

This module deliberately has no concrete HTTP dependency.  A caller supplies a
transport, a private SEC User-Agent value, a clock, and a budget.  Only exact
official SEC URLs derived inside this module can be requested.  The returned
JSON artifacts contain hashes and counts, never the private contact value.

The historical 2000-2024, 24-slot SEC audit remains useful source/parser
evidence, but it is not an exhaustive catalogue.  This layer independently
binds the current Apple Submissions payload, every historical payload referenced
by it, and (in separate stage calls) every exact primary-document byte stream.
It does not claim master-index, index.json, or SGML reconciliation for the full
predictive corpus, including 2025-2026.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .sec_audit_transport import SecAuditTransport, canonical_sec_url
from .sec_filing_content import normalize_filing_text
from .sec_filing_gemma_contract import (
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    STAGE_ORDER,
    STAGE_WINDOWS,
    build_contract_manifest,
    build_stage_content_manifest,
    canonical_session_calendar,
    canonical_sha256,
    validate_corpus_universe_manifest,
)
from .sec_point_in_time import (
    AAPL_CIK,
    FilingRecord,
    SecAuditLimitError,
    SecPointInTimeError,
    content_sha256,
    parse_submissions_acceptance_datetime,
    parse_submissions_rows,
    validate_sec_user_agent,
)


CATALOG_SCHEMA_VERSION = "aapl-sec-gemma-official-catalog-v1"
STAGE_ARTIFACT_SCHEMA_VERSION = "aapl-sec-gemma-stage-primary-bytes-v1"
REQUEST_RECEIPT_SCHEMA_VERSION = "aapl-sec-gemma-sec-request-receipt-v1"
MAIN_SUBMISSIONS_NAME = f"CIK{AAPL_CIK}.json"
MAIN_SUBMISSIONS_URL = canonical_sec_url(
    f"https://data.sec.gov/submissions/{MAIN_SUBMISSIONS_NAME}"
)
_HISTORICAL_NAME_RE = re.compile(
    rf"CIK{AAPL_CIK}-submissions-[0-9]{{3}}\.json\Z"
)
_PRIMARY_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_ACCESSION_RE = re.compile(r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_EASTERN = ZoneInfo("America/New_York")

_RESPONSE_AUDIT_KEYS = {
    "url",
    "status_code",
    "content_type",
    "size_bytes",
    "content_sha256",
    "cache_hit",
    "user_agent_sha256",
    "network_requests",
    "retries",
    "redirects",
}
_TRANSPORT_SECURITY_KEYS = {
    "trust_env",
    "proxies",
    "follow_redirects",
    "max_retries",
    "max_redirects",
    "allow_cache_reads",
    "allow_cache_writes",
    "streaming_body",
    "content_length_preflight",
    "incremental_byte_budget",
    "transport_max_requests",
    "transport_max_bytes",
    "transport_max_seconds",
}
_STRICT_TRANSPORT_BEHAVIOR = {
    "trust_env": False,
    "proxies": False,
    "follow_redirects": False,
    "max_retries": 0,
    "max_redirects": 0,
    "allow_cache_reads": False,
    "allow_cache_writes": False,
    "streaming_body": True,
    "content_length_preflight": True,
    "incremental_byte_budget": True,
}


class SecFilingGemmaCorpusError(SecPointInTimeError):
    """Official corpus evidence violated the frozen acquisition contract."""


class SecCorpusTransport(Protocol):
    """Small injected-transport surface; implementations decide how to fetch."""

    def fetch(self, url: str) -> tuple[bytes, Any]: ...


@dataclass
class SecCorpusBudget:
    """Logical reconciliation ceiling for one or more acquisition calls.

    This outer counter cannot protect memory after a transport has materialized
    an oversized response.  Production safety therefore also requires an
    attested transport that checks Content-Length and enforces the same or
    tighter request, incremental-byte, and wall-clock ceilings while streaming.
    This counter independently reconciles the bytes that reached this layer.
    """

    clock: Callable[[], float] = field(repr=False)
    max_requests: int = MAX_SEC_REQUESTS
    max_bytes: int = MAX_SEC_BYTES
    max_seconds: float = float(MAX_SEC_SECONDS)
    requests: int = field(init=False, default=0)
    bytes_received: int = field(init=False, default=0)
    _started_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(self.clock):
            raise SecAuditLimitError("SEC corpus clock must be callable")
        if (
            isinstance(self.max_requests, bool)
            or not isinstance(self.max_requests, int)
            or not 1 <= self.max_requests <= MAX_SEC_REQUESTS
        ):
            raise SecAuditLimitError("SEC corpus request ceiling may only be tightened")
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or not 1 <= self.max_bytes <= MAX_SEC_BYTES
        ):
            raise SecAuditLimitError("SEC corpus byte ceiling may only be tightened")
        seconds = float(self.max_seconds)
        if not math.isfinite(seconds) or not 0.0 < seconds <= MAX_SEC_SECONDS:
            raise SecAuditLimitError("SEC corpus time ceiling may only be tightened")
        self.max_seconds = seconds
        self._started_at = self._now()
        self.check()

    def _now(self) -> float:
        try:
            value = float(self.clock())
        except Exception:
            raise SecAuditLimitError("SEC corpus clock failed") from None
        if not math.isfinite(value):
            raise SecAuditLimitError("SEC corpus clock must be finite")
        return value

    def check(self) -> None:
        elapsed = self._now() - self._started_at
        if elapsed < 0.0:
            raise SecAuditLimitError("SEC corpus clock moved backwards")
        if elapsed > self.max_seconds:
            raise SecAuditLimitError("SEC corpus wall-clock ceiling exceeded")

    def begin_request(self) -> None:
        self.check()
        if self.requests >= self.max_requests:
            raise SecAuditLimitError("SEC corpus request ceiling exceeded")
        self.requests += 1

    def add_bytes(self, count: int) -> None:
        self.check()
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SecAuditLimitError("SEC corpus byte count is invalid")
        if self.bytes_received + count > self.max_bytes:
            raise SecAuditLimitError("SEC corpus byte ceiling exceeded")
        self.bytes_received += count
        self.check()

    def snapshot(self) -> dict[str, int | float]:
        self.check()
        return {
            "requests": self.requests,
            "bytes_received": self.bytes_received,
            "max_requests": self.max_requests,
            "max_bytes": self.max_bytes,
            "max_seconds": self.max_seconds,
            "elapsed_seconds": self._now() - self._started_at,
        }


@dataclass(frozen=True)
class CatalogSourceBytes:
    name: str
    url: str
    payload: bytes = field(repr=False)
    payload_sha256: str
    raw_record_count: int
    unique_record_count: int
    request_receipt_sha256: str


@dataclass(frozen=True)
class CatalogAcquisition:
    """Raw official metadata bytes plus their deterministic catalogue view."""

    sources: tuple[CatalogSourceBytes, ...]
    universe_records: tuple[Mapping[str, Any], ...]
    request_receipts: tuple[Mapping[str, Any], ...]
    artifact: Mapping[str, Any]
    artifact_json: bytes = field(repr=False)
    universe_records_json: bytes = field(repr=False)
    request_receipts_json: bytes = field(repr=False)

    @property
    def catalog_artifact_sha256(self) -> str:
        return str(self.artifact["catalog_artifact_sha256"])

    @property
    def catalog_total_record_count(self) -> int:
        return int(self.artifact["catalog_total_record_count"])

    @property
    def catalog_eligible_record_count(self) -> int:
        return int(self.artifact["catalog_eligible_record_count"])


@dataclass(frozen=True)
class StageDocumentBytes:
    accession_number: str
    url: str
    raw_primary_document: bytes = field(repr=False)
    normalized_text: bytes = field(repr=False)
    primary_document_sha256: str
    normalized_text_sha256: str
    request_receipt_sha256: str


@dataclass(frozen=True)
class _ParsedCatalogRow:
    record: FilingRecord
    raw_identity_json: bytes = field(repr=False)
    raw_identity_sha256: str


@dataclass(frozen=True)
class StageContentAcquisition:
    """One stage's exact raw/normalized bytes and manifest-bound evidence."""

    artifact_stage: str
    documents: tuple[StageDocumentBytes, ...]
    request_receipts: tuple[Mapping[str, Any], ...]
    content_manifest: Mapping[str, Any]
    artifact: Mapping[str, Any]
    artifact_json: bytes = field(repr=False)
    content_manifest_json: bytes = field(repr=False)
    request_receipts_json: bytes = field(repr=False)

    @property
    def stage_artifact_sha256(self) -> str:
        return str(self.artifact["stage_artifact_sha256"])

    @property
    def raw_documents_by_accession(self) -> Mapping[str, bytes]:
        return MappingProxyType(
            {item.accession_number: item.raw_primary_document for item in self.documents}
        )

    @property
    def normalized_documents_by_accession(self) -> Mapping[str, bytes]:
        return MappingProxyType(
            {item.accession_number: item.normalized_text for item in self.documents}
        )


def _bare_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except Exception:
        raise SecFilingGemmaCorpusError(
            "SEC corpus evidence is not finite canonical JSON"
        ) from None


def _json_object_snapshot(value: Any, *, location: str) -> dict[str, Any]:
    """Materialize one immutable JSON view before validating caller mappings."""

    try:
        payload = _canonical_json_bytes(value)
        snapshot = json.loads(payload.decode("utf-8"))
    except Exception:
        raise SecFilingGemmaCorpusError(
            f"{location} could not be safely snapshotted"
        ) from None
    if type(snapshot) is not dict:
        raise SecFilingGemmaCorpusError(f"{location} must be a JSON object")
    return snapshot


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise SecFilingGemmaCorpusError(
                "SEC corpus evidence keys must remain strings"
            )
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _typed_json_identity(value: Any) -> Any:
    """Preserve exact JSON scalar/container types for raw-row comparison."""

    if value is None:
        return {"type": "null", "value": None}
    if type(value) is bool:
        return {"type": "boolean", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise SecFilingGemmaCorpusError(
                "Submissions row contains a non-finite number"
            )
        return {"type": "float", "value_hex": value.hex()}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) is list:
        return {
            "type": "array",
            "value": [_typed_json_identity(item) for item in value],
        }
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise SecFilingGemmaCorpusError(
                "Submissions row object keys must be strings"
            )
        return {
            "type": "object",
            "value": {
                key: _typed_json_identity(value[key]) for key in sorted(value)
            },
        }
    raise SecFilingGemmaCorpusError(
        "Submissions row contains a non-JSON value"
    )


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError(f"{label} must be bytes")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecFilingGemmaCorpusError(
                    f"{label} contains a duplicate JSON object key"
                )
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                SecFilingGemmaCorpusError(f"{label} contains a non-finite JSON value")
            ),
        )
    except SecFilingGemmaCorpusError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise SecFilingGemmaCorpusError(f"{label} is not strict UTF-8 JSON") from None
    if not isinstance(value, dict):
        raise SecFilingGemmaCorpusError(f"{label} must be a JSON object")
    return value


def _canonical_cik(value: Any, *, location: str) -> str:
    if isinstance(value, bool):
        raise SecFilingGemmaCorpusError(f"{location} is not a canonical SEC CIK")
    raw = str(value).strip()
    if not raw.isdigit() or len(raw) > 10:
        raise SecFilingGemmaCorpusError(f"{location} is not a canonical SEC CIK")
    return str(int(raw)).zfill(10)


def _canonical_date(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise SecFilingGemmaCorpusError(f"{location} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise SecFilingGemmaCorpusError(
            f"{location} must be a canonical ISO date"
        ) from None
    if parsed.isoformat() != value:
        raise SecFilingGemmaCorpusError(f"{location} must be a canonical ISO date")
    return value


def _transport_user_agent_sha256(transport: SecCorpusTransport) -> str:
    audit = getattr(transport, "user_agent_audit", None)
    if callable(audit):
        audit = audit()
    if is_dataclass(audit) and not isinstance(audit, type):
        value = asdict(audit)
    elif isinstance(audit, Mapping):
        value = dict(audit)
    else:
        safe_state = getattr(transport, "safe_state", None)
        state = safe_state() if callable(safe_state) else None
        value = dict(state.get("user_agent", {})) if isinstance(state, Mapping) else {}
    if set(value) != {"sha256", "real_contact_validated"}:
        raise SecFilingGemmaCorpusError(
            "Injected SEC transport lacks exact validated User-Agent evidence"
        )
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or _TAGGED_SHA256_RE.fullmatch(digest) is None
        or value["real_contact_validated"] is not True
    ):
        raise SecFilingGemmaCorpusError(
            "Injected SEC transport lacks validated private-contact evidence"
        )
    return digest


def _transport_security_state(
    transport: SecCorpusTransport,
    *,
    budget: SecCorpusBudget,
) -> dict[str, Any]:
    provider = getattr(transport, "acquisition_security_state", None)
    if callable(provider):
        raw = provider()
        if not isinstance(raw, Mapping) or set(raw) != _TRANSPORT_SECURITY_KEYS:
            raise SecFilingGemmaCorpusError(
                "Injected SEC transport security-state schema is not exact"
            )
        state = dict(raw)
    else:
        # Compatibility with SecAuditTransport.  Its response path always sends
        # allow_redirects=False; max_redirects=0 additionally rejects any manual
        # redirect.  Requests-like sessions must disable environment proxy use.
        if type(transport) is not SecAuditTransport:
            raise SecFilingGemmaCorpusError(
                "Only the reviewed SecAuditTransport may use implicit capability evidence"
            )
        safe_provider = getattr(transport, "safe_state", None)
        safe = safe_provider() if callable(safe_provider) else None
        session = getattr(transport, "_session", None)
        if not isinstance(safe, Mapping) or session is None:
            raise SecFilingGemmaCorpusError(
                "Injected SEC transport lacks security-state evidence"
            )
        proxies = getattr(session, "proxies", None)
        transport_budget = safe.get("budget")
        if not isinstance(transport_budget, Mapping):
            raise SecFilingGemmaCorpusError(
                "Injected SEC transport lacks streaming-budget evidence"
            )
        state = {
            "trust_env": getattr(session, "trust_env", None),
            "proxies": bool(proxies) if isinstance(proxies, Mapping) else None,
            "follow_redirects": False,
            "max_retries": safe.get("max_retries"),
            "max_redirects": safe.get("max_redirects"),
            "allow_cache_reads": safe.get("allow_cache_reads"),
            "allow_cache_writes": safe.get("allow_cache_writes"),
            # These capabilities are guaranteed by the reviewed
            # SecAuditTransport implementation: it prechecks Content-Length and
            # calls BudgetCounter.add_bytes for each streamed chunk.
            "streaming_body": True,
            "content_length_preflight": True,
            "incremental_byte_budget": True,
            "transport_max_requests": transport_budget.get("max_requests"),
            "transport_max_bytes": transport_budget.get("max_bytes"),
            "transport_max_seconds": transport_budget.get("max_seconds"),
        }
    for key, expected in _STRICT_TRANSPORT_BEHAVIOR.items():
        observed = state.get(key)
        matches = (
            observed is expected
            if type(expected) is bool
            else type(observed) is type(expected) and observed == expected
        )
        if not matches:
            raise SecFilingGemmaCorpusError(
                "Injected SEC transport must disable environment proxies, redirects, "
                "retries, and cache access and attest streaming enforcement"
            )
    request_cap = state.get("transport_max_requests")
    byte_cap = state.get("transport_max_bytes")
    second_cap = state.get("transport_max_seconds")
    if (
        type(request_cap) is not int
        or not 1 <= request_cap <= budget.max_requests
        or type(byte_cap) is not int
        or not 1 <= byte_cap <= budget.max_bytes
        or type(second_cap) not in {int, float}
        or not math.isfinite(float(second_cap))
        or not 0.0 < float(second_cap) <= budget.max_seconds
    ):
        raise SecFilingGemmaCorpusError(
            "Injected SEC transport ceilings must be positive and no wider than "
            "the acquisition budget"
        )
    return {
        **_STRICT_TRANSPORT_BEHAVIOR,
        "transport_max_requests": request_cap,
        "transport_max_bytes": byte_cap,
        "transport_max_seconds": float(second_cap),
    }


def _prepare_transport(
    transport: SecCorpusTransport,
    *,
    user_agent: str,
    budget: SecCorpusBudget,
) -> tuple[str, dict[str, Any]]:
    if transport is None or not callable(getattr(transport, "fetch", None)):
        raise SecFilingGemmaCorpusError("An injected SEC transport is required")
    # The raw value exists only in the caller and this stack frame.  No returned
    # object, error, receipt, or artifact contains it.
    expected = validate_sec_user_agent(user_agent).sha256
    try:
        observed = _transport_user_agent_sha256(transport)
        security = _transport_security_state(transport, budget=budget)
    except Exception:
        # A third-party property/state provider could include the private
        # contact in its exception.  Convert it to a fixed message.
        raise SecFilingGemmaCorpusError(
            "Injected SEC transport security evidence could not be validated"
        ) from None
    if observed != expected:
        raise SecFilingGemmaCorpusError(
            "Injected SEC transport User-Agent does not match the validated contact"
        )
    return observed, security


def _audit_mapping(audit: Any) -> dict[str, Any]:
    try:
        if is_dataclass(audit) and not isinstance(audit, type):
            value = asdict(audit)
        elif isinstance(audit, Mapping):
            value = dict(audit)
        else:
            value = None
    except Exception:
        raise SecFilingGemmaCorpusError(
            "SEC response audit mapping could not be safely read"
        ) from None
    if type(value) is not dict or set(value) != _RESPONSE_AUDIT_KEYS:
        raise SecFilingGemmaCorpusError("SEC response audit schema is not exact")
    return value


def _validated_response_audit(
    audit: Mapping[str, Any],
    *,
    requested: str,
    payload: bytes,
    user_agent_sha256: str,
    require_json: bool,
) -> tuple[str, str]:
    """Return final URL/content type after exception-sanitized primitive checks."""

    try:
        url_value = audit["url"]
        status = audit["status_code"]
        content_type = audit["content_type"]
        size = audit["size_bytes"]
        payload_hash = audit["content_sha256"]
        cache_hit = audit["cache_hit"]
        audit_user_agent = audit["user_agent_sha256"]
        network_requests = audit["network_requests"]
        retries = audit["retries"]
        redirects = audit["redirects"]
    except Exception:
        raise SecFilingGemmaCorpusError(
            "SEC response audit fields could not be safely read"
        ) from None
    if type(url_value) is not str:
        raise SecFilingGemmaCorpusError("SEC response final URL is invalid")
    final_url = canonical_sec_url(url_value)
    if (
        type(content_type) is not str
        or not content_type
        or len(content_type) > 160
        or "@" in content_type
        or any(ord(char) < 32 or ord(char) > 126 for char in content_type)
    ):
        raise SecFilingGemmaCorpusError("SEC response Content-Type is unsafe")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if require_json and media_type not in {
        "application/json",
        "application/edgar+json",
        "text/json",
    }:
        raise SecFilingGemmaCorpusError("SEC Submissions response is not JSON")
    if (
        type(status) is not int
        or status != 200
        or type(size) is not int
        or size != len(payload)
        or type(payload_hash) is not str
        or _TAGGED_SHA256_RE.fullmatch(payload_hash) is None
        or payload_hash != content_sha256(payload)
        or type(audit_user_agent) is not str
        or _TAGGED_SHA256_RE.fullmatch(audit_user_agent) is None
        or audit_user_agent != user_agent_sha256
        or type(cache_hit) is not bool
        or cache_hit is not False
        or type(network_requests) is not int
        or network_requests != 1
        or type(retries) is not int
        or retries != 0
        or type(redirects) is not int
        or redirects != 0
        or final_url != requested
    ):
        raise SecFilingGemmaCorpusError(
            "SEC response must be one fresh exact-URL request with no retry or redirect"
        )
    return final_url, content_type


def _fetch_exact(
    transport: SecCorpusTransport,
    *,
    url: str,
    purpose: str,
    user_agent_sha256: str,
    budget: SecCorpusBudget,
    sequence_number: int,
    require_json: bool,
) -> tuple[bytes, dict[str, Any]]:
    requested = canonical_sec_url(url)
    budget.begin_request()
    try:
        result = transport.fetch(requested)
    except Exception:
        # Third-party exceptions may embed request headers, including contact
        # information.  Never retain or interpolate them.
        raise SecFilingGemmaCorpusError("Injected SEC transport fetch failed") from None
    budget.check()
    if type(result) is not tuple or len(result) != 2 or type(result[0]) is not bytes:
        raise SecFilingGemmaCorpusError(
            "Injected SEC transport returned an invalid response tuple"
        )
    payload, raw_audit = result
    audit = _audit_mapping(raw_audit)
    budget.add_bytes(len(payload))
    final_url, content_type = _validated_response_audit(
        audit,
        requested=requested,
        payload=payload,
        user_agent_sha256=user_agent_sha256,
        require_json=require_json,
    )
    body = {
        "schema_version": REQUEST_RECEIPT_SCHEMA_VERSION,
        "sequence_number": sequence_number,
        "purpose": purpose,
        "requested_url": requested,
        "final_url": final_url,
        "status_code": 200,
        "content_type": content_type,
        "size_bytes": len(payload),
        "content_sha256": content_sha256(payload),
        "cache_hit": False,
        "network_requests": 1,
        "retries": 0,
        "redirects": 0,
        "user_agent_sha256": user_agent_sha256,
    }
    return payload, {**body, "request_receipt_sha256": canonical_sha256(body)}


def _reference_specs(main_payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if _canonical_cik(main_payload.get("cik", ""), location="Submissions CIK") != AAPL_CIK:
        raise SecFilingGemmaCorpusError("Submissions payload is not Apple CIK 0000320193")
    filings = main_payload.get("filings")
    if not isinstance(filings, Mapping) or not isinstance(filings.get("recent"), Mapping):
        raise SecFilingGemmaCorpusError("Main Submissions payload lacks filings.recent")
    if "files" not in filings:
        raise SecFilingGemmaCorpusError(
            "Main Apple Submissions payload must contain filings.files"
        )
    values = filings["files"]
    if type(values) is not list or not values:
        raise SecFilingGemmaCorpusError(
            "Main Apple Submissions filings.files must be a nonempty array"
        )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        if type(raw) is not dict or set(raw) != {
            "name",
            "filingCount",
            "filingFrom",
            "filingTo",
        }:
            raise SecFilingGemmaCorpusError(
                f"Historical Submissions reference {index} lacks the exact frozen metadata"
            )
        name = raw.get("name")
        if (
            not isinstance(name, str)
            or name != name.strip()
            or PurePosixPath(name).name != name
            or "\\" in name
            or _HISTORICAL_NAME_RE.fullmatch(name) is None
        ):
            raise SecFilingGemmaCorpusError(
                "Historical Submissions filename is unsafe or outside Apple scope"
            )
        if name in seen:
            raise SecFilingGemmaCorpusError(
                "Historical Submissions references contain a duplicate filename"
            )
        seen.add(name)
        url = canonical_sec_url(f"https://data.sec.gov/submissions/{name}")
        count = raw["filingCount"]
        if type(count) is not int or count < 1:
            raise SecFilingGemmaCorpusError(
                "Historical Submissions filingCount must be a positive integer"
            )
        item = {
            "name": name,
            "url": url,
            "filing_count": count,
            "filing_from": _canonical_date(
                raw["filingFrom"], location="filings.files.filingFrom"
            ),
            "filing_to": _canonical_date(
                raw["filingTo"], location="filings.files.filingTo"
            ),
        }
        if item["filing_from"] > item["filing_to"]:
            raise SecFilingGemmaCorpusError(
                "Historical Submissions filing range is reversed"
            )
        result.append(item)
    return tuple(sorted(result, key=lambda item: item["name"]))


def _deduplicate_columns(
    columns: Mapping[str, Any],
    *,
    source_name: str,
) -> tuple[
    dict[str, list[Any]],
    int,
    int,
    dict[str, tuple[bytes, str]],
]:
    accessions = columns.get("accessionNumber")
    if type(accessions) is not list:
        raise SecFilingGemmaCorpusError(
            f"{source_name} lacks an accessionNumber row array"
        )
    size = len(accessions)
    arrays: dict[str, list[Any]] = {}
    for key, value in columns.items():
        if (
            type(key) is not str
            or type(value) is not list
            or len(value) != size
        ):
            raise SecFilingGemmaCorpusError(
                f"{source_name} Submissions columns are not exact row-aligned arrays"
            )
        arrays[key] = list(value)
    by_accession: dict[str, tuple[bytes, str]] = {}
    retained: list[int] = []
    ordered_keys = tuple(sorted(arrays))
    for position, accession in enumerate(accessions):
        if not isinstance(accession, str):
            raise SecFilingGemmaCorpusError(
                f"{source_name} accession values must remain strings"
            )
        raw_row = {key: arrays[key][position] for key in ordered_keys}
        identity_json = _canonical_json_bytes(_typed_json_identity(raw_row))
        identity = (identity_json, _bare_sha256(identity_json))
        prior = by_accession.get(accession)
        if prior is None:
            by_accession[accession] = identity
            retained.append(position)
        elif prior[0] != identity_json:
            raise SecFilingGemmaCorpusError(
                "Conflicting metadata exists for a duplicate accession"
            )
    return (
        {key: [values[index] for index in retained] for key, values in arrays.items()},
        size,
        len(retained),
        by_accession,
    )


def _records_from_columns(
    columns: Mapping[str, Any],
    *,
    source_name: str,
) -> tuple[tuple[_ParsedCatalogRow, ...], int, int]:
    deduplicated, raw_count, unique_count, identities = _deduplicate_columns(
        columns, source_name=source_name
    )
    synthetic = {
        "cik": AAPL_CIK,
        "filings": {"recent": deduplicated, "files": []},
    }
    try:
        records = parse_submissions_rows(synthetic)
    except (SecPointInTimeError, TypeError, ValueError):
        raise SecFilingGemmaCorpusError(
            f"{source_name} Submissions rows failed strict validation"
        ) from None
    parsed = tuple(
        _ParsedCatalogRow(
            record=replace(record, source_name=source_name),
            raw_identity_json=identities[record.accession_number][0],
            raw_identity_sha256=identities[record.accession_number][1],
        )
        for record in records
    )
    return parsed, raw_count, unique_count


def _normalized_acceptance(record: FilingRecord) -> tuple[str, date]:
    parsed = parse_submissions_acceptance_datetime(record.acceptance_datetime)
    if parsed.microsecond != 0:
        raise SecFilingGemmaCorpusError(
            "Submissions acceptance timestamp has subsecond precision that the frozen universe cannot preserve"
        )
    eastern = parsed.astimezone(_EASTERN)
    return eastern.strftime("%Y%m%d%H%M%S"), eastern.date()


def _next_session_after(value: date, sessions: Sequence[str]) -> str | None:
    return next((session for session in sessions if session > value.isoformat()), None)


def _stage_for_availability(value: str) -> str | None:
    for stage in STAGE_ORDER:
        first, last = STAGE_WINDOWS[stage]
        if first <= value <= last:
            return stage
    return None


def _universe_record(
    record: FilingRecord,
    *,
    source_url: str,
    source_content_sha256: str,
    raw_row_identity_sha256: str,
    sessions: Sequence[str],
) -> tuple[dict[str, Any] | None, str]:
    if record.subject_cik != AAPL_CIK:
        raise SecFilingGemmaCorpusError("Catalog row subject CIK is not Apple")
    if (
        _ACCESSION_RE.fullmatch(record.accession_number) is None
        or not record.accession_number.startswith(f"{AAPL_CIK}-")
    ):
        raise SecFilingGemmaCorpusError(
            "Apple Submissions row has a non-Apple accession"
        )
    upper_form = record.form.upper()
    if upper_form in {"10-K", "10-Q", "10-K/A", "10-Q/A"} and record.form != upper_form:
        raise SecFilingGemmaCorpusError("Periodic form spelling is not canonical")
    if record.form in {"10-K/A", "10-Q/A"}:
        return None, "amendment"
    if record.form not in {"10-K", "10-Q"}:
        return None, "other_form"
    if _PRIMARY_NAME_RE.fullmatch(record.primary_document) is None:
        raise SecFilingGemmaCorpusError(
            "Eligible Apple periodic filing has an unsafe primary document"
        )
    acceptance, acceptance_date = _normalized_acceptance(record)
    filing_date = date.fromisoformat(record.filing_date)
    change_date = (
        date.fromisoformat(record.date_of_filing_date_change)
        if record.date_of_filing_date_change
        else None
    )
    evidence_dates = [acceptance_date, filing_date]
    if change_date is not None:
        evidence_dates.append(change_date)
    availability = _next_session_after(max(evidence_dates), sessions)
    if availability is None or _stage_for_availability(availability) is None:
        return None, "outside_contract_availability_window"
    source_body = {
        "source_url": source_url,
        "source_content_sha256": source_content_sha256,
        "raw_row_identity_sha256": raw_row_identity_sha256,
        "accession_number": record.accession_number,
        "subject_cik": AAPL_CIK,
        "form": record.form,
        "acceptance_datetime_source": record.acceptance_datetime,
        "acceptance_datetime_et": acceptance,
        "filing_date": record.filing_date,
        "filing_date_change": record.date_of_filing_date_change or None,
        "primary_document": record.primary_document,
    }
    return (
        {
            "accession_number": record.accession_number,
            "subject_cik": AAPL_CIK,
            "form": record.form,
            "acceptance_datetime": acceptance,
            "filing_date": record.filing_date,
            "filing_date_change": record.date_of_filing_date_change or None,
            "primary_document": record.primary_document,
            "source_record_sha256": canonical_sha256(source_body),
        },
        "eligible",
    )


def acquire_official_sec_catalog(
    *,
    transport: SecCorpusTransport,
    user_agent: str,
    budget: SecCorpusBudget,
    session_dates: Sequence[str],
) -> CatalogAcquisition:
    """Fetch Apple main+all referenced historical Submissions payloads.

    The function returns every exact raw source payload, fresh-request receipts,
    source hashes/counts, and all-and-only canonical Apple 10-K/10-Q
    non-amendment records whose conservative availability belongs to the frozen
    2000-2026 universe.  It never fetches filing text.
    """

    if not isinstance(budget, SecCorpusBudget):
        raise SecFilingGemmaCorpusError("A SecCorpusBudget must be injected")
    sessions = canonical_session_calendar(session_dates)
    user_agent_sha256, security = _prepare_transport(
        transport, user_agent=user_agent, budget=budget
    )
    start_requests = budget.requests
    start_bytes = budget.bytes_received
    receipts: list[dict[str, Any]] = []
    main_body, main_receipt = _fetch_exact(
        transport,
        url=MAIN_SUBMISSIONS_URL,
        purpose="apple_main_submissions",
        user_agent_sha256=user_agent_sha256,
        budget=budget,
        sequence_number=1,
        require_json=True,
    )
    receipts.append(main_receipt)
    main_payload = _strict_json_object(main_body, label="main Submissions payload")
    references = _reference_specs(main_payload)

    raw_sources: list[tuple[str, str, bytes, dict[str, Any], Mapping[str, Any]]] = [
        (MAIN_SUBMISSIONS_NAME, MAIN_SUBMISSIONS_URL, main_body, main_receipt, main_payload)
    ]
    for position, reference in enumerate(references, start=2):
        body, receipt = _fetch_exact(
            transport,
            url=reference["url"],
            purpose="apple_historical_submissions",
            user_agent_sha256=user_agent_sha256,
            budget=budget,
            sequence_number=position,
            require_json=True,
        )
        receipts.append(receipt)
        payload = _strict_json_object(
            body, label=f"historical Submissions payload {reference['name']}"
        )
        if "cik" in payload and _canonical_cik(
            payload["cik"], location="historical Submissions CIK"
        ) != AAPL_CIK:
            raise SecFilingGemmaCorpusError(
                "Historical Submissions payload claims a non-Apple CIK"
            )
        raw_sources.append((reference["name"], reference["url"], body, receipt, payload))

    sourced_records: list[tuple[_ParsedCatalogRow, str, str]] = []
    source_objects: list[CatalogSourceBytes] = []
    source_summaries: list[dict[str, Any]] = []
    reference_by_name = {item["name"]: item for item in references}
    for name, url, body, receipt, payload in raw_sources:
        if name == MAIN_SUBMISSIONS_NAME:
            filings = payload.get("filings")
            columns = filings.get("recent") if isinstance(filings, Mapping) else None
        else:
            # Official historical payloads are normally column-only.  If a
            # scalar CIK is present, authenticate then remove it consistently;
            # every remaining member must be a row-aligned column.
            columns = {key: value for key, value in payload.items() if key != "cik"}
        if not isinstance(columns, Mapping):
            raise SecFilingGemmaCorpusError(f"{name} lacks a columnar filing table")
        records, raw_count, unique_count = _records_from_columns(
            columns, source_name=name
        )
        if name != MAIN_SUBMISSIONS_NAME:
            reference = reference_by_name[name]
            if raw_count != reference["filing_count"]:
                raise SecFilingGemmaCorpusError(
                    "Historical Submissions row count differs from its main reference"
                )
            filing_dates = [item.record.filing_date for item in records]
            if not filing_dates or min(filing_dates) != reference["filing_from"]:
                raise SecFilingGemmaCorpusError(
                    "Historical Submissions minimum filing date differs from filingFrom"
                )
            if max(filing_dates) != reference["filing_to"]:
                raise SecFilingGemmaCorpusError(
                    "Historical Submissions maximum filing date differs from filingTo"
                )
        source_hash = _bare_sha256(body)
        for item in records:
            sourced_records.append((item, url, source_hash))
        row_set = [
            {
                "accession_number": item.record.accession_number,
                "raw_identity_sha256": item.raw_identity_sha256,
            }
            for item in sorted(records, key=lambda value: value.record.accession_number)
        ]
        source = CatalogSourceBytes(
            name=name,
            url=url,
            payload=body,
            payload_sha256=source_hash,
            raw_record_count=raw_count,
            unique_record_count=unique_count,
            request_receipt_sha256=receipt["request_receipt_sha256"],
        )
        source_objects.append(source)
        source_summaries.append(
            {
                "name": name,
                "url": url,
                "payload_sha256": source_hash,
                "payload_bytes": len(body),
                "raw_record_count": raw_count,
                "unique_record_count": unique_count,
                "raw_row_set_sha256": canonical_sha256(row_set),
                "request_receipt_sha256": receipt["request_receipt_sha256"],
            }
        )

    by_accession: dict[str, tuple[_ParsedCatalogRow, str, str]] = {}
    global_exact_duplicates = 0
    for item, source_url, source_hash in sourced_records:
        prior = by_accession.get(item.record.accession_number)
        if prior is None:
            by_accession[item.record.accession_number] = (item, source_url, source_hash)
        elif prior[0].raw_identity_json == item.raw_identity_json:
            global_exact_duplicates += 1
        else:
            raise SecFilingGemmaCorpusError(
                "Conflicting metadata exists for an accession across Submissions sources"
            )

    eligible: list[dict[str, Any]] = []
    exclusion_counts = {
        "amendment": 0,
        "other_form": 0,
        "outside_contract_availability_window": 0,
    }
    for item, source_url, source_hash in by_accession.values():
        canonical, reason = _universe_record(
            item.record,
            source_url=source_url,
            source_content_sha256=source_hash,
            raw_row_identity_sha256=item.raw_identity_sha256,
            sessions=sessions,
        )
        if canonical is None:
            exclusion_counts[reason] += 1
        else:
            eligible.append(canonical)
    eligible.sort(
        key=lambda record: (
            record["filing_date"],
            record["acceptance_datetime"],
            record["accession_number"],
        )
    )
    if not eligible:
        raise SecFilingGemmaCorpusError(
            "Official Submissions metadata produced no eligible periodic filings"
        )

    request_delta = budget.requests - start_requests
    byte_delta = budget.bytes_received - start_bytes
    if request_delta != len(source_objects) or byte_delta != sum(
        len(item.payload) for item in source_objects
    ):
        raise SecFilingGemmaCorpusError("SEC corpus budget does not reconcile to source bytes")
    body = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "subject_cik": AAPL_CIK,
        "main_submissions_url": MAIN_SUBMISSIONS_URL,
        "historical_reference_names": [item["name"] for item in references],
        "historical_reference_policy": (
            "experiment_strict_required_fields_and_exact_downloaded_min_max_not_sec_wide_claim"
        ),
        "source_count": len(source_objects),
        "sources": source_summaries,
        "raw_record_count": len(sourced_records)
        + sum(item.raw_record_count - item.unique_record_count for item in source_objects),
        "catalog_total_record_count": len(by_accession),
        "within_source_exact_duplicate_count": sum(
            item.raw_record_count - item.unique_record_count for item in source_objects
        ),
        "cross_source_exact_duplicate_count": global_exact_duplicates,
        "catalog_eligible_record_count": len(eligible),
        "exclusion_counts": exclusion_counts,
        "eligible_records": eligible,
        "eligible_records_sha256": canonical_sha256(eligible),
        "request_receipts_sha256": canonical_sha256(receipts),
        "acquisition_request_count": request_delta,
        "acquisition_bytes": byte_delta,
        "transport_security": security,
        "outer_budget_role": "post_transport_reconciliation_not_streaming_protection",
        "user_agent_sha256": user_agent_sha256,
        "evidence_boundary": {
            "complete_current_plus_every_referenced_historical_submissions": True,
            "contract_availability_window": [
                STAGE_WINDOWS["development"][0],
                STAGE_WINDOWS["final"][1],
            ],
            "legacy_24_slot_audit_scope_end": "2024-12-31",
            "legacy_24_slot_audit_is_exhaustive_catalog_proof": False,
            "catalog_authentication": "exact_official_sec_submissions_bytes",
            "primary_document_authentication": "separate_stage_artifact",
            "full_predictive_corpus_master_index_sgml_or_index_reconciled": False,
            "sgml_or_master_index_reconciliation_claimed_for_2025_2026": False,
        },
        "contains_outcomes_market_data_or_model_output": False,
    }
    artifact = {**body, "catalog_artifact_sha256": canonical_sha256(body)}
    artifact_json = _canonical_json_bytes(artifact)
    records_json = _canonical_json_bytes(eligible)
    receipts_json = _canonical_json_bytes(receipts)
    return CatalogAcquisition(
        sources=tuple(source_objects),
        universe_records=tuple(_deep_freeze(record) for record in eligible),
        request_receipts=tuple(_deep_freeze(receipt) for receipt in receipts),
        artifact=_deep_freeze(artifact),
        artifact_json=artifact_json,
        universe_records_json=records_json,
        request_receipts_json=receipts_json,
    )


def _primary_document_url(record: Mapping[str, Any]) -> str:
    accession = record.get("accession_number")
    filename = record.get("primary_document")
    if (
        not isinstance(accession, str)
        or _ACCESSION_RE.fullmatch(accession) is None
        or not accession.startswith(f"{AAPL_CIK}-")
        or not isinstance(filename, str)
        or _PRIMARY_NAME_RE.fullmatch(filename) is None
    ):
        raise SecFilingGemmaCorpusError(
            "Validated universe contains an unsafe primary-document identity"
        )
    base = (
        f"https://www.sec.gov/Archives/edgar/data/{int(AAPL_CIK)}/"
        f"{accession.replace('-', '')}"
    )
    return canonical_sec_url(f"{base}/{quote(filename, safe='-._~')}")


def acquire_authorized_stage_documents(
    *,
    transport: SecCorpusTransport,
    user_agent: str,
    budget: SecCorpusBudget,
    authorized_stage: str,
    universe_manifest: Mapping[str, Any],
    expected_universe_sha256: str,
    session_dates: Sequence[str],
) -> StageContentAcquisition:
    """Fetch all and only one already-authorized universe stage.

    Accessions and URLs are never accepted from the caller.  They are derived
    from the externally pinned, canonically rebuilt universe, so a development
    request cannot fetch an intermediate/final filing and no arbitrary URL can
    enter the request set.  Any missing, cached, redirected, retried, duplicate,
    substituted, or over-budget response aborts the entire call.
    """

    if not isinstance(budget, SecCorpusBudget):
        raise SecFilingGemmaCorpusError("A SecCorpusBudget must be injected")
    if authorized_stage not in STAGE_ORDER:
        raise SecFilingGemmaCorpusError("Authorized corpus stage is invalid")
    sessions = canonical_session_calendar(session_dates)
    universe_snapshot = _json_object_snapshot(
        universe_manifest, location="Corpus universe manifest"
    )
    universe_hash = validate_corpus_universe_manifest(
        universe_snapshot,
        session_dates=sessions,
        expected_universe_sha256=expected_universe_sha256,
        require_complete_coverage=True,
    )
    if universe_hash != expected_universe_sha256:
        raise SecFilingGemmaCorpusError("Validated universe external pin changed")
    user_agent_sha256, security = _prepare_transport(
        transport, user_agent=user_agent, budget=budget
    )
    selected = [
        record
        for record in universe_snapshot["records"]
        if record["artifact_stage"] == authorized_stage
    ]
    selected.sort(
        key=lambda record: (record["availability_session"], record["accession_number"])
    )
    if not selected:
        raise SecFilingGemmaCorpusError("Authorized stage contains no filing documents")
    expected_stage_count = universe_snapshot["stage_counts"].get(authorized_stage)
    if expected_stage_count != len(selected):
        raise SecFilingGemmaCorpusError(
            "Authorized stage records do not reconcile to universe stage count"
        )

    urls = [_primary_document_url(record) for record in selected]
    if len(urls) != len(set(urls)):
        raise SecFilingGemmaCorpusError(
            "Authorized stage derives duplicate primary-document URLs"
        )
    start_requests = budget.requests
    start_bytes = budget.bytes_received
    receipts: list[dict[str, Any]] = []
    documents: list[StageDocumentBytes] = []
    manifest_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for sequence_number, (record, url) in enumerate(zip(selected, urls), start=1):
        raw, receipt = _fetch_exact(
            transport,
            url=url,
            purpose=f"{authorized_stage}_primary_document",
            user_agent_sha256=user_agent_sha256,
            budget=budget,
            sequence_number=sequence_number,
            require_json=False,
        )
        if not raw:
            raise SecFilingGemmaCorpusError("SEC primary document is empty")
        try:
            normalized = normalize_filing_text(raw.decode("latin-1"))
        except (UnicodeDecodeError, SecPointInTimeError, TypeError, ValueError):
            raise SecFilingGemmaCorpusError(
                "SEC primary document could not be deterministically normalized"
            ) from None
        normalized_bytes = normalized.text.encode("utf-8")
        if not normalized_bytes:
            raise SecFilingGemmaCorpusError(
                "SEC primary document has no normalized visible text"
            )
        raw_hash = _bare_sha256(raw)
        normalized_hash = _bare_sha256(normalized_bytes)
        if normalized.sha256 != f"sha256:{normalized_hash}":
            raise SecFilingGemmaCorpusError("Normalized SEC text hash does not reconcile")
        receipt_hash = receipt["request_receipt_sha256"]
        document = StageDocumentBytes(
            accession_number=record["accession_number"],
            url=url,
            raw_primary_document=raw,
            normalized_text=normalized_bytes,
            primary_document_sha256=raw_hash,
            normalized_text_sha256=normalized_hash,
            request_receipt_sha256=receipt_hash,
        )
        documents.append(document)
        receipts.append(receipt)
        manifest_rows.append(
            {
                "accession_number": record["accession_number"],
                "primary_document_sha256": raw_hash,
                "normalized_text_sha256": normalized_hash,
                "primary_document_bytes": len(raw),
                "normalized_text_bytes": len(normalized_bytes),
            }
        )
        artifact_rows.append(
            {
                "accession_number": record["accession_number"],
                "availability_session": record["availability_session"],
                "primary_document": record["primary_document"],
                "url": url,
                "primary_document_sha256": raw_hash,
                "primary_document_bytes": len(raw),
                "normalized_text_sha256": normalized_hash,
                "normalized_text_bytes": len(normalized_bytes),
                "normalized_character_count": normalized.character_count,
                "normalized_text_usable": normalized.usable,
                "request_receipt_sha256": receipt_hash,
            }
        )

    content_manifest = build_stage_content_manifest(
        artifact_stage=authorized_stage,
        corpus_universe_sha256=universe_hash,
        documents=manifest_rows,
        universe_manifest=universe_snapshot,
    )
    request_delta = budget.requests - start_requests
    byte_delta = budget.bytes_received - start_bytes
    if request_delta != len(documents) or byte_delta != sum(
        len(item.raw_primary_document) for item in documents
    ):
        raise SecFilingGemmaCorpusError(
            "SEC stage budget does not reconcile to exact primary-document bytes"
        )
    body = {
        "schema_version": STAGE_ARTIFACT_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "artifact_stage": authorized_stage,
        "corpus_universe_sha256": universe_hash,
        "catalog_artifact_sha256": universe_snapshot["catalog_artifact_sha256"],
        "content_manifest_sha256": content_manifest["content_manifest_sha256"],
        "document_count": len(documents),
        "documents": artifact_rows,
        "request_receipts": receipts,
        "request_receipts_sha256": canonical_sha256(receipts),
        "acquisition_request_count": request_delta,
        "acquisition_bytes": byte_delta,
        "transport_security": security,
        "outer_budget_role": "post_transport_reconciliation_not_streaming_protection",
        "user_agent_sha256": user_agent_sha256,
        "selection_policy": "all_and_only_authorized_stage_universe_primary_documents",
        "arbitrary_urls_or_accessions_accepted": False,
        "sampling_dropping_cache_or_substitution_allowed": False,
        "evidence_boundary": {
            "metadata": "exact_current_plus_referenced_official_sec_submissions_bytes",
            "document": "exact_official_sec_primary_document_bytes",
            "legacy_24_slot_audit_is_exhaustive_catalog_proof": False,
            "full_predictive_corpus_master_index_sgml_or_index_reconciled": False,
            "sgml_or_master_index_reconciliation_claimed_for_2025_2026": False,
        },
        "contains_outcomes_market_data_or_model_output": False,
    }
    artifact = {**body, "stage_artifact_sha256": canonical_sha256(body)}
    artifact_json = _canonical_json_bytes(artifact)
    content_manifest_json = _canonical_json_bytes(content_manifest)
    receipts_json = _canonical_json_bytes(receipts)
    return StageContentAcquisition(
        artifact_stage=authorized_stage,
        documents=tuple(documents),
        request_receipts=tuple(_deep_freeze(item) for item in receipts),
        content_manifest=_deep_freeze(content_manifest),
        artifact=_deep_freeze(artifact),
        artifact_json=artifact_json,
        content_manifest_json=content_manifest_json,
        request_receipts_json=receipts_json,
    )


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "MAIN_SUBMISSIONS_NAME",
    "MAIN_SUBMISSIONS_URL",
    "REQUEST_RECEIPT_SCHEMA_VERSION",
    "STAGE_ARTIFACT_SCHEMA_VERSION",
    "CatalogAcquisition",
    "CatalogSourceBytes",
    "SecCorpusBudget",
    "SecCorpusTransport",
    "SecFilingGemmaCorpusError",
    "StageContentAcquisition",
    "StageDocumentBytes",
    "acquire_authorized_stage_documents",
    "acquire_official_sec_catalog",
]
