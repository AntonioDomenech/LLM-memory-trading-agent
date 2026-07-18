"""Strict one-shot HTTPS transport for lean SEC evidence v3.7.

The parent process owns the hard deadline and starts one isolated child for one
official SEC response.  The child uses a direct TLS socket and a deliberately
small HTTP/1.1 parser.  It has no proxy, redirect, retry, or cache path.  Raw
third-party data never crosses the child boundary: a successful child returns
only fixed-schema metadata after it has privacy-scanned and fsynced the body.

The framing helpers are public so the byte contract can be tested without any
network access.  Production dispatch remains restricted to canonical HTTPS
URLs on ``data.sec.gov`` and ``www.sec.gov``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import ssl
import stat
import subprocess
import sys
import time
from typing import Any, Final, Protocol
from urllib.parse import quote_from_bytes, quote_plus, urlsplit


TRANSPORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-7-transport-v1"
)
TRANSPORT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-7-transport-receipt-v1"
)
OFFICIAL_SEC_HOSTS: Final[frozenset[str]] = frozenset(
    {"data.sec.gov", "www.sec.gov"}
)
REQUEST_DEADLINE_SECONDS: Final[float] = 30.0
MIB: Final[int] = 1024 * 1024
MAIN_SUBMISSIONS_BODY_LIMIT: Final[int] = 32 * MIB
HISTORICAL_SUBMISSIONS_BODY_LIMIT: Final[int] = 64 * MIB
MASTER_COMPRESSED_BODY_LIMIT: Final[int] = 16 * MIB
COMPLETE_SUBMISSION_BODY_LIMIT: Final[int] = 128 * MIB
ALLOWED_ROLE_BODY_LIMITS: Final[frozenset[int]] = frozenset(
    {
        MAIN_SUBMISSIONS_BODY_LIMIT,
        HISTORICAL_SUBMISSIONS_BODY_LIMIT,
        MASTER_COMPRESSED_BODY_LIMIT,
        COMPLETE_SUBMISSION_BODY_LIMIT,
    }
)
MAX_RESPONSE_HEAD_BYTES: Final[int] = 128 * 1024
MAX_HEADER_FIELDS: Final[int] = 512
MAX_HEADER_LINE_BYTES: Final[int] = 16 * 1024
MAX_CHUNK_LINE_BYTES: Final[int] = 8 * 1024
BODY_READ_CHUNK_BYTES: Final[int] = 64 * 1024
MAX_WORKER_REQUEST_BYTES: Final[int] = 8 * 1024
MAX_WORKER_RESPONSE_BYTES: Final[int] = 8 * 1024
PRIVACY_SCAN_FORMS: Final[tuple[str, ...]] = (
    "utf8",
    "json_unicode",
    "json_ascii",
    "html_quote",
    "html_no_quote",
    "percent_upper",
    "percent_lower",
    "percent_plus_upper",
    "percent_plus_lower",
)

SAFE_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "blob_persistence_failed",
        "body_limit_exceeded",
        "certificate_verification_failed",
        "content_encoding_rejected",
        "content_length_rejected",
        "hard_timeout",
        "hostname_verification_failed",
        "http_framing_rejected",
        "http_status_rejected",
        "premature_eof",
        "privacy_echo",
        "response_length_mismatch",
        "tls_configuration_rejected",
        "trailers_rejected",
        "transfer_encoding_rejected",
        "transport_error",
        "url_mismatch",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HISTORICAL_URL_RE = re.compile(
    r"https://data\.sec\.gov/submissions/"
    r"(?P<name>CIK0000320193-submissions-[0-9]{3}\.json)\Z"
)
_MASTER_URL_RE = re.compile(
    r"https://www\.sec\.gov/Archives/edgar/full-index/"
    r"(?P<year>[0-9]{4})/QTR(?P<quarter>[1-4])/master\.gz\Z"
)
_COMPLETE_URL_RE = re.compile(
    r"https://www\.sec\.gov/Archives/edgar/data/320193/"
    r"(?P<accession>[0-9]{10}-[0-9]{2}-[0-9]{6})\.txt\Z"
)
_MAIN_SUBMISSIONS_URL: Final[str] = (
    "https://data.sec.gov/submissions/CIK0000320193.json"
)
_EMAIL_RE = re.compile(
    r"(?<![A-Z0-9._%+-])([A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,}))"
    r"(?![A-Z0-9._%+-])",
    re.IGNORECASE,
)
_CONTACT_PLACEHOLDERS: Final[tuple[str, ...]] = (
    "example",
    "sample company",
    "placeholder",
    "change me",
    "your email",
)
_HEADER_NAME_RE = re.compile(rb"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")
_DECIMAL_RE = re.compile(rb"[0-9]+\Z")
_HEX_RE = re.compile(rb"[0-9A-Fa-f]+\Z")
_TOKEN_BYTES = frozenset(
    b"!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
_WORKER_FLAG: Final[str] = "--strict-sec-transport-worker"
_WORKER_MODULE: Final[str] = "agent_benchmark.sec_gemma_lean_v37_transport"
_WORKER_BOOTSTRAP: Final[str] = r"""
import hashlib
import sys
from pathlib import Path

if (
    sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or sys.flags.ignore_environment != 1
    or sys.flags.safe_path is not True
    or sys.dont_write_bytecode is not True
    or len(sys.argv) != 6
):
    raise SystemExit(91)
root = Path(sys.argv[1])
module_name = sys.argv[2]
worker_flag = sys.argv[3]
expected_source_sha256 = sys.argv[4]
expected_interpreter_sha256 = sys.argv[5]
source = root / "agent_benchmark" / "sec_gemma_lean_v37_transport.py"
executable = Path(sys.executable)
source_bytes = None
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
if (
    not root.is_absolute()
    or root.resolve(strict=True) != root
    or Path.cwd().resolve(strict=True) != root
    or source.resolve(strict=True) != source
    or source.name != "sec_gemma_lean_v37_transport.py"
    or source.suffix != ".py"
    or executable.resolve(strict=True) != executable
    or module_name != "agent_benchmark.sec_gemma_lean_v37_transport"
    or worker_flag != "--strict-sec-transport-worker"
    or len(expected_source_sha256) != 64
    or len(expected_interpreter_sha256) != 64
    or sha256_file(executable) != expected_interpreter_sha256
):
    raise SystemExit(91)
try:
    source_bytes = source.read_bytes()
except OSError:
    raise SystemExit(91) from None
if hashlib.sha256(source_bytes).hexdigest() != expected_source_sha256:
    raise SystemExit(91)
sys.argv = [str(source), worker_flag]
worker_globals = {
    "__name__": "__main__",
    "__file__": str(source),
    "__package__": None,
    "__cached__": None,
}
exec(
    compile(source_bytes, str(source), "exec", dont_inherit=True),
    worker_globals,
    worker_globals,
)
"""


class SecGemmaLeanV37TransportError(RuntimeError):
    """A redacted, fixed-code transport rejection."""

    def __init__(self, code: str, *, observed_body_bytes: int = 0) -> None:
        if code not in SAFE_ERROR_CODES:
            code = "transport_error"
        if type(observed_body_bytes) is not int or observed_body_bytes < 0:
            observed_body_bytes = 0
        super().__init__(code)
        self.code = code
        self.observed_body_bytes = observed_body_bytes


class _PrivateSecContact:
    """In-memory contact closure plus the exact v3.7 privacy needles."""

    __slots__ = ("__text", "_fingerprint", "_needles")

    def __init__(self, contact: str) -> None:
        if type(contact) is not str or not 6 <= len(contact) <= 512:
            raise SecGemmaLeanV37TransportError("transport_error")
        try:
            utf8 = contact.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise SecGemmaLeanV37TransportError("transport_error") from None
        matches = list(_EMAIL_RE.finditer(contact))
        identity = contact[: matches[0].start()].strip(" /;:-") if matches else ""
        domain = matches[0].group(2).lower() if matches else ""
        lowered = contact.lower()
        if (
            contact != contact.strip()
            or len(matches) != 1
            or re.search(r"[A-Za-z]{2}", identity) is None
            or any(token in lowered for token in _CONTACT_PLACEHOLDERS)
            or domain in {"example.com", "example.org", "example.net", "localhost"}
            or domain.endswith((".invalid", ".test"))
            or any(byte < 0x20 or byte == 0x7F for byte in utf8)
            or any(character in contact for character in "<>")
        ):
            raise SecGemmaLeanV37TransportError("transport_error")
        json_unicode = json.dumps(contact, ensure_ascii=False)[1:-1].encode("utf-8")
        json_ascii = json.dumps(contact, ensure_ascii=True)[1:-1].encode("ascii")
        html_quote = html.escape(contact, quote=True).encode("utf-8")
        html_no_quote = html.escape(contact, quote=False).encode("utf-8")
        percent = quote_from_bytes(utf8, safe="").encode("ascii")
        percent_plus = quote_plus(contact, safe="").encode("ascii")
        needles = {
            utf8,
            json_unicode,
            json_ascii,
            html_quote,
            html_no_quote,
            percent,
            percent.lower(),
            percent_plus,
            percent_plus.lower(),
        }
        self.__text = contact
        self._fingerprint = hashlib.sha256(utf8).hexdigest()
        self._needles = tuple(sorted(needles, key=lambda item: (len(item), item)))

    @property
    def fingerprint_sha256(self) -> str:
        return self._fingerprint

    def request_header_closure(self) -> Callable[[], str]:
        contact = self.__text

        def read_header_value() -> str:
            return contact

        return read_header_value

    def serialized_echo_needles(self) -> tuple[bytes, ...]:
        return self._needles

    def contains_echo(self, value: bytes | bytearray | memoryview) -> bool:
        if type(value) is bytes:
            payload = value
        elif type(value) in {bytearray, memoryview}:
            payload = bytes(value)
        else:
            raise SecGemmaLeanV37TransportError("transport_error")
        return any(needle in payload for needle in self._needles)


def privacy_echo_needles(private_contact: str) -> tuple[bytes, ...]:
    """Return the exact raw/JSON/HTML/percent needles for offline validation."""

    return _PrivateSecContact(private_contact).serialized_echo_needles()


@dataclass(frozen=True)
class TransportCapability:
    schema_version: str
    https_only: bool
    official_hosts: tuple[str, ...]
    certificate_required: bool
    hostname_checking: bool
    trust_store: str
    custom_ca_allowed: bool
    proxy_allowed: bool
    redirects_allowed: bool
    retries_allowed: bool
    cache_reads_allowed: bool
    cache_writes_allowed: bool
    accept_encoding: str
    hard_deadline_seconds: float
    isolated_child: bool
    sanitized_child_environment: bool
    bytecode_writes_allowed: bool
    transport_source_sha256: str
    transport_source_bytes: int
    transport_source_path_sha256: str
    worker_bootstrap_sha256: str
    worker_bootstrap_bytes: int
    interpreter_executable_sha256: str
    interpreter_executable_bytes: int
    interpreter_path_sha256: str
    interpreter_implementation: str
    interpreter_version: str
    worker_python_flags: tuple[str, ...]


@dataclass(frozen=True)
class SecRoleBinding:
    role_class: str
    role_id: str
    body_limit_bytes: int


@dataclass(frozen=True)
class ResponseHead:
    """One parsed HTTP/1.1 response head with duplicate fields preserved."""

    status_code: int
    headers: tuple[tuple[bytes, bytes], ...]
    raw_head: bytes

    def values(self, name: bytes | str) -> tuple[bytes, ...]:
        target = name.encode("ascii") if type(name) is str else name
        if type(target) is not bytes:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        lowered = target.lower()
        return tuple(value for key, value in self.headers if key.lower() == lowered)


@dataclass(frozen=True)
class BodyFraming:
    kind: str
    declared_content_length: int | None
    content_encoding: str | None


@dataclass(frozen=True)
class ParsedResponse:
    status_code: int
    headers: tuple[tuple[bytes, bytes], ...]
    framing: str
    declared_content_length: int | None
    content_encoding: str | None
    raw_head: bytes
    body: bytes


@dataclass(frozen=True)
class TransportResult:
    schema_version: str
    request_url: str
    observed_url: str
    role_class: str
    role_id: str
    intent_event_sha256: str
    body_limit_bytes: int
    temporary_blob_path: Path
    status_code: int
    framing: str
    declared_content_length: int | None
    content_encoding: str | None
    body_bytes: int
    body_sha256: str
    raw_headers_sha256: str
    raw_headers_bytes: int
    contact_fingerprint_sha256: str
    execution_identity: Mapping[str, Any]
    transport_receipt: Mapping[str, Any]
    transport_receipt_sha256: str
    elapsed_milliseconds: int


class _ReadSome(Protocol):
    def read_some(self, maximum_bytes: int) -> bytes: ...


class _BytesReader:
    def __init__(self, payload: bytes) -> None:
        if type(payload) is not bytes:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        self._payload = payload
        self._position = 0

    def read_some(self, maximum_bytes: int) -> bytes:
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        end = min(len(self._payload), self._position + maximum_bytes)
        value = self._payload[self._position : end]
        self._position = end
        return value


class _DeadlineSocketReader:
    def __init__(self, connection: ssl.SSLSocket, deadline: float) -> None:
        self._connection = connection
        self._deadline = deadline

    def read_some(self, maximum_bytes: int) -> bytes:
        if type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        remaining = self._deadline - time.monotonic()
        if remaining <= 0.0:
            raise SecGemmaLeanV37TransportError("hard_timeout")
        try:
            self._connection.settimeout(remaining)
            value = self._connection.recv(maximum_bytes)
        except (TimeoutError, socket.timeout):
            raise SecGemmaLeanV37TransportError("hard_timeout") from None
        except (OSError, ssl.SSLError):
            raise SecGemmaLeanV37TransportError("transport_error") from None
        if type(value) is not bytes or len(value) > maximum_bytes:
            raise SecGemmaLeanV37TransportError("transport_error")
        if time.monotonic() >= self._deadline:
            raise SecGemmaLeanV37TransportError("hard_timeout")
        return value


def _sha256_file(path: Path) -> tuple[int, str]:
    try:
        details = path.stat()
        if (
            not stat.S_ISREG(details.st_mode)
            or path.is_symlink()
            or details.st_size < 1
        ):
            raise OSError
        digest = hashlib.sha256()
        counted = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(BODY_READ_CHUNK_BYTES)
                if not chunk:
                    break
                counted += len(chunk)
                digest.update(chunk)
        if counted != details.st_size:
            raise OSError
        return counted, digest.hexdigest()
    except OSError:
        raise SecGemmaLeanV37TransportError("tls_configuration_rejected") from None


def execution_identity() -> dict[str, Any]:
    """Hash the exact source, bootstrap, and interpreter used by the worker."""

    try:
        source = Path(__file__).resolve(strict=True)
        executable = Path(sys.executable).resolve(strict=True)
    except OSError:
        raise SecGemmaLeanV37TransportError("tls_configuration_rejected") from None
    if source.name != "sec_gemma_lean_v37_transport.py" or source.suffix != ".py":
        raise SecGemmaLeanV37TransportError("tls_configuration_rejected")
    source_bytes, source_hash = _sha256_file(source)
    executable_bytes, executable_hash = _sha256_file(executable)
    bootstrap = _WORKER_BOOTSTRAP.encode("utf-8", errors="strict")
    return {
        "interpreter_executable_bytes": executable_bytes,
        "interpreter_executable_sha256": executable_hash,
        "interpreter_implementation": sys.implementation.name,
        "interpreter_path_sha256": hashlib.sha256(
            str(executable).encode("utf-8", errors="strict")
        ).hexdigest(),
        "interpreter_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "transport_source_bytes": source_bytes,
        "transport_source_path_sha256": hashlib.sha256(
            str(source).encode("utf-8", errors="strict")
        ).hexdigest(),
        "transport_source_sha256": source_hash,
        "worker_bootstrap_bytes": len(bootstrap),
        "worker_bootstrap_sha256": hashlib.sha256(bootstrap).hexdigest(),
    }


def transport_capability() -> TransportCapability:
    """Return the immutable capability that must be sealed before dispatch."""

    identity = execution_identity()
    return TransportCapability(
        schema_version=TRANSPORT_SCHEMA_VERSION,
        https_only=True,
        official_hosts=tuple(sorted(OFFICIAL_SEC_HOSTS)),
        certificate_required=True,
        hostname_checking=True,
        trust_store="runtime_default",
        custom_ca_allowed=False,
        proxy_allowed=False,
        redirects_allowed=False,
        retries_allowed=False,
        cache_reads_allowed=False,
        cache_writes_allowed=False,
        accept_encoding="identity",
        hard_deadline_seconds=REQUEST_DEADLINE_SECONDS,
        isolated_child=True,
        sanitized_child_environment=True,
        bytecode_writes_allowed=False,
        worker_python_flags=("-I", "-S", "-B"),
        **identity,
    )


def derive_sec_role(url: str) -> SecRoleBinding:
    """Derive the only role ID/class/limit authorized by an exact frozen URL."""

    if type(url) is not str or not url or len(url) > 4096 or "\x00" in url:
        raise SecGemmaLeanV37TransportError("url_mismatch")
    if url == _MAIN_SUBMISSIONS_URL:
        return SecRoleBinding(
            "main_submissions", "submissions/main", MAIN_SUBMISSIONS_BODY_LIMIT
        )
    historical = _HISTORICAL_URL_RE.fullmatch(url)
    if historical is not None:
        name = historical.group("name")
        return SecRoleBinding(
            "historical_submissions",
            f"submissions/historical/{name}",
            HISTORICAL_SUBMISSIONS_BODY_LIMIT,
        )
    master = _MASTER_URL_RE.fullmatch(url)
    if master is not None:
        year = int(master.group("year"))
        quarter = int(master.group("quarter"))
        if (year, quarter) < (1994, 1) or (year, quarter) > (2026, 3):
            raise SecGemmaLeanV37TransportError("url_mismatch")
        return SecRoleBinding(
            "quarterly_master",
            f"master/{year}/QTR{quarter}",
            MASTER_COMPRESSED_BODY_LIMIT,
        )
    complete = _COMPLETE_URL_RE.fullmatch(url)
    if complete is not None:
        accession = complete.group("accession")
        return SecRoleBinding(
            "complete_submission",
            f"complete/{accession}",
            COMPLETE_SUBMISSION_BODY_LIMIT,
        )
    raise SecGemmaLeanV37TransportError("url_mismatch")


def canonical_sec_url(url: str) -> str:
    """Accept only one exact frozen v3.7 role URL, with no normalization."""

    derive_sec_role(url)
    return url


def parse_http_response_head(raw_head: bytes) -> ResponseHead:
    """Parse one exact HTTP/1.1 head without coalescing duplicate fields."""

    if (
        type(raw_head) is not bytes
        or not raw_head.endswith(b"\r\n\r\n")
        or len(raw_head) > MAX_RESPONSE_HEAD_BYTES
        or b"\x00" in raw_head
    ):
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    stripped = raw_head[:-4]
    if b"\r" in stripped.replace(b"\r\n", b"") or b"\n" in stripped.replace(
        b"\r\n", b""
    ):
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    lines = stripped.split(b"\r\n")
    if not lines or len(lines) - 1 > MAX_HEADER_FIELDS:
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    status = lines[0]
    if len(status) > MAX_HEADER_LINE_BYTES or not status.startswith(b"HTTP/1.1 "):
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    status_parts = status.split(b" ", 2)
    if (
        len(status_parts) < 2
        or status_parts[0] != b"HTTP/1.1"
        or len(status_parts[1]) != 3
        or not status_parts[1].isdigit()
        or (
            len(status_parts) == 3
            and any(byte < 0x20 or byte > 0x7E for byte in status_parts[2])
        )
    ):
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    headers: list[tuple[bytes, bytes]] = []
    for line in lines[1:]:
        if (
            not line
            or len(line) > MAX_HEADER_LINE_BYTES
            or line[:1] in {b" ", b"\t"}
            or b":" not in line
        ):
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        name, value = line.split(b":", 1)
        if _HEADER_NAME_RE.fullmatch(name) is None:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        if any(
            byte < 0x20 and byte != 0x09 or byte == 0x7F
            for byte in value
        ):
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        headers.append((name, value.strip(b" \t")))
    return ResponseHead(
        status_code=int(status_parts[1]),
        headers=tuple(headers),
        raw_head=raw_head,
    )


def response_body_framing(head: ResponseHead) -> BodyFraming:
    """Resolve the one permitted response framing and content encoding."""

    if type(head) is not ResponseHead:
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    content_encodings = head.values(b"Content-Encoding")
    if len(content_encodings) > 1:
        raise SecGemmaLeanV37TransportError("content_encoding_rejected")
    content_encoding: str | None = None
    if content_encodings:
        value = content_encodings[0]
        if b"," in value or value.lower() != b"identity":
            raise SecGemmaLeanV37TransportError("content_encoding_rejected")
        content_encoding = "identity"

    content_lengths = head.values(b"Content-Length")
    transfer_encodings = head.values(b"Transfer-Encoding")
    if len(content_lengths) > 1:
        raise SecGemmaLeanV37TransportError("content_length_rejected")
    if len(transfer_encodings) > 1:
        raise SecGemmaLeanV37TransportError("transfer_encoding_rejected")
    if content_lengths and transfer_encodings:
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    if transfer_encodings:
        value = transfer_encodings[0]
        if b"," in value or value.lower() != b"chunked":
            raise SecGemmaLeanV37TransportError("transfer_encoding_rejected")
        return BodyFraming("chunked", None, content_encoding)
    if content_lengths:
        value = content_lengths[0]
        if (
            not value
            or b"," in value
            or _DECIMAL_RE.fullmatch(value) is None
            or len(value) > 20
        ):
            raise SecGemmaLeanV37TransportError("content_length_rejected")
        return BodyFraming("content-length", int(value), content_encoding)
    return BodyFraming("eof", None, content_encoding)


def _read_line(
    reader: _ReadSome,
    *,
    maximum_bytes: int,
    eof_code: str,
) -> bytes:
    value = bytearray()
    while len(value) <= maximum_bytes:
        byte = reader.read_some(1)
        if not byte:
            raise SecGemmaLeanV37TransportError(eof_code)
        if len(byte) != 1:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        value.extend(byte)
        if value.endswith(b"\r\n"):
            return bytes(value[:-2])
        if value.endswith(b"\n") and not value.endswith(b"\r\n"):
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
    raise SecGemmaLeanV37TransportError("http_framing_rejected")


def _read_head(reader: _ReadSome) -> bytes:
    value = bytearray()
    while len(value) < MAX_RESPONSE_HEAD_BYTES:
        byte = reader.read_some(1)
        if not byte:
            raise SecGemmaLeanV37TransportError("premature_eof")
        if len(byte) != 1:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        value.extend(byte)
        if value.endswith(b"\r\n\r\n"):
            return bytes(value)
        if value.endswith(b"\n") and not value.endswith(b"\r\n"):
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
    raise SecGemmaLeanV37TransportError("http_framing_rejected")


def _read_exact(reader: _ReadSome, count: int, *, eof_code: str) -> bytes:
    if type(count) is not int or count < 0:
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    value = bytearray()
    while len(value) < count:
        requested = min(BODY_READ_CHUNK_BYTES, count - len(value))
        chunk = reader.read_some(requested)
        if not chunk:
            raise SecGemmaLeanV37TransportError(eof_code)
        if len(chunk) > requested:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        value.extend(chunk)
    return bytes(value)


def _expect_connection_eof(reader: _ReadSome, observed: int) -> None:
    extra = reader.read_some(1)
    if extra:
        raise SecGemmaLeanV37TransportError(
            "response_length_mismatch", observed_body_bytes=observed + 1
        )


def _validate_chunk_extensions(extension: bytes) -> bool:
    """Validate RFC 9112 chunk extensions without interpreting them."""

    position = 0
    length = len(extension)
    while position < length:
        while position < length and extension[position] in b" \t":
            position += 1
        if position >= length or extension[position] != 0x3B:
            return False
        position += 1
        while position < length and extension[position] in b" \t":
            position += 1
        start = position
        while position < length and extension[position] in _TOKEN_BYTES:
            position += 1
        if start == position:
            return False
        while position < length and extension[position] in b" \t":
            position += 1
        if position >= length or extension[position] == 0x3B:
            continue
        if extension[position] != 0x3D:
            return False
        position += 1
        while position < length and extension[position] in b" \t":
            position += 1
        if position >= length:
            return False
        if extension[position] == 0x22:
            position += 1
            closed = False
            while position < length:
                byte = extension[position]
                position += 1
                if byte == 0x22:
                    closed = True
                    break
                if byte == 0x5C:
                    if position >= length:
                        return False
                    escaped = extension[position]
                    position += 1
                    if not (escaped == 0x09 or 0x20 <= escaped <= 0x7E or escaped >= 0x80):
                        return False
                elif not (
                    byte == 0x09
                    or byte == 0x20
                    or byte == 0x21
                    or 0x23 <= byte <= 0x5B
                    or 0x5D <= byte <= 0x7E
                    or byte >= 0x80
                ):
                    return False
            if not closed:
                return False
        else:
            start = position
            while position < length and extension[position] in _TOKEN_BYTES:
                position += 1
            if start == position:
                return False
        while position < length and extension[position] in b" \t":
            position += 1
        if position < length and extension[position] != 0x3B:
            return False
    return True


def _chunk_size(line: bytes) -> int:
    size_text, separator, extension = line.partition(b";")
    if (
        not size_text
        or _HEX_RE.fullmatch(size_text) is None
        or len(size_text) > 16
        or (separator and not _validate_chunk_extensions(b";" + extension))
    ):
        raise SecGemmaLeanV37TransportError("http_framing_rejected")
    return int(size_text, 16)


def _read_content_length_body(
    reader: _ReadSome,
    *,
    declared: int,
    body_limit: int,
) -> bytes:
    if declared > body_limit:
        raise SecGemmaLeanV37TransportError("body_limit_exceeded")
    body = _read_exact(reader, declared, eof_code="premature_eof")
    _expect_connection_eof(reader, len(body))
    return body


def _read_eof_body(reader: _ReadSome, *, body_limit: int) -> bytes:
    body = bytearray()
    while True:
        remaining_with_sentinel = body_limit + 1 - len(body)
        requested = min(BODY_READ_CHUNK_BYTES, remaining_with_sentinel)
        chunk = reader.read_some(requested)
        if not chunk:
            return bytes(body)
        if len(chunk) > requested:
            raise SecGemmaLeanV37TransportError("http_framing_rejected")
        body.extend(chunk)
        if len(body) == body_limit + 1:
            raise SecGemmaLeanV37TransportError(
                "body_limit_exceeded", observed_body_bytes=len(body)
            )


def _read_chunked_body(reader: _ReadSome, *, body_limit: int) -> bytes:
    body = bytearray()
    while True:
        line = _read_line(
            reader,
            maximum_bytes=MAX_CHUNK_LINE_BYTES,
            eof_code="premature_eof",
        )
        size = _chunk_size(line)
        if size == 0:
            trailer_or_end = _read_line(
                reader,
                maximum_bytes=MAX_HEADER_LINE_BYTES,
                eof_code="premature_eof",
            )
            if trailer_or_end:
                raise SecGemmaLeanV37TransportError("trailers_rejected")
            _expect_connection_eof(reader, len(body))
            return bytes(body)
        remaining_with_sentinel = body_limit + 1 - len(body)
        to_read = min(size, remaining_with_sentinel)
        body.extend(_read_exact(reader, to_read, eof_code="premature_eof"))
        if len(body) == body_limit + 1:
            raise SecGemmaLeanV37TransportError(
                "body_limit_exceeded", observed_body_bytes=len(body)
            )
        if to_read != size:
            raise SecGemmaLeanV37TransportError(
                "body_limit_exceeded", observed_body_bytes=len(body)
            )
        terminator = _read_exact(reader, 2, eof_code="premature_eof")
        if terminator != b"\r\n":
            raise SecGemmaLeanV37TransportError("http_framing_rejected")


def _parse_from_reader(
    reader: _ReadSome,
    *,
    body_limit: int,
    contact: _PrivateSecContact | None = None,
) -> ParsedResponse:
    if (
        type(body_limit) is not int
        or type(body_limit) is bool
        or not 0 <= body_limit <= COMPLETE_SUBMISSION_BODY_LIMIT
    ):
        raise SecGemmaLeanV37TransportError("body_limit_exceeded")
    raw_head = _read_head(reader)
    if contact is not None and contact.contains_echo(raw_head):
        raise SecGemmaLeanV37TransportError("privacy_echo")
    head = parse_http_response_head(raw_head)
    if head.status_code != 200:
        raise SecGemmaLeanV37TransportError("http_status_rejected")
    framing = response_body_framing(head)
    if framing.kind == "content-length":
        assert framing.declared_content_length is not None
        body = _read_content_length_body(
            reader,
            declared=framing.declared_content_length,
            body_limit=body_limit,
        )
    elif framing.kind == "chunked":
        body = _read_chunked_body(reader, body_limit=body_limit)
    else:
        body = _read_eof_body(reader, body_limit=body_limit)
    if contact is not None and contact.contains_echo(body):
        raise SecGemmaLeanV37TransportError(
            "privacy_echo", observed_body_bytes=len(body)
        )
    return ParsedResponse(
        status_code=head.status_code,
        headers=head.headers,
        framing=framing.kind,
        declared_content_length=framing.declared_content_length,
        content_encoding=framing.content_encoding,
        raw_head=raw_head,
        body=body,
    )


def parse_raw_http_response(
    raw_response: bytes,
    *,
    body_limit: int,
    private_contact: str | None = None,
) -> ParsedResponse:
    """Deterministically parse a complete raw response for offline tests/replay."""

    contact: _PrivateSecContact | None = None
    if private_contact is not None:
        try:
            contact = _PrivateSecContact(private_contact)
        except Exception:
            raise SecGemmaLeanV37TransportError("transport_error") from None
    return _parse_from_reader(
        _BytesReader(raw_response), body_limit=body_limit, contact=contact
    )


def _remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if not math.isfinite(value) or value <= 0.0:
        raise SecGemmaLeanV37TransportError("hard_timeout")
    return value


def _default_tls_context() -> ssl.SSLContext:
    """Create exactly the runtime default, certificate-required TLS context."""

    try:
        context = ssl.create_default_context()
    except Exception:
        raise SecGemmaLeanV37TransportError("tls_configuration_rejected") from None
    if context.verify_mode != ssl.CERT_REQUIRED or context.check_hostname is not True:
        raise SecGemmaLeanV37TransportError("tls_configuration_rejected")
    return context


def _connect_default_tls(
    host: str,
    *,
    deadline: float,
) -> ssl.SSLSocket:
    """Resolve once, connect once, and perform one default-trust TLS handshake."""

    plain: socket.socket | None = None
    try:
        addresses = socket.getaddrinfo(
            host,
            443,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        if not addresses:
            raise SecGemmaLeanV37TransportError("transport_error")
        family, socktype, proto, _canonname, address = addresses[0]
        plain = socket.socket(family, socktype, proto)
        plain.settimeout(_remaining(deadline))
        plain.connect(address)
        context = _default_tls_context()
        plain.settimeout(_remaining(deadline))
        wrapped = context.wrap_socket(
            plain,
            server_hostname=host,
            suppress_ragged_eofs=False,
        )
        plain = None
        if wrapped.context.verify_mode != ssl.CERT_REQUIRED:
            wrapped.close()
            raise SecGemmaLeanV37TransportError("tls_configuration_rejected")
        if wrapped.context.check_hostname is not True:
            wrapped.close()
            raise SecGemmaLeanV37TransportError("tls_configuration_rejected")
        return wrapped
    except SecGemmaLeanV37TransportError:
        raise
    except ssl.SSLCertVerificationError:
        raise SecGemmaLeanV37TransportError(
            "certificate_verification_failed"
        ) from None
    except ssl.CertificateError:
        raise SecGemmaLeanV37TransportError(
            "hostname_verification_failed"
        ) from None
    except (TimeoutError, socket.timeout):
        raise SecGemmaLeanV37TransportError("hard_timeout") from None
    except (OSError, ssl.SSLError):
        raise SecGemmaLeanV37TransportError("transport_error") from None
    finally:
        if plain is not None:
            try:
                plain.close()
            except BaseException:
                pass


def _request_bytes(url: str, contact: _PrivateSecContact) -> tuple[str, bytes]:
    canonical = canonical_sec_url(url)
    parsed = urlsplit(canonical)
    host = parsed.hostname
    if host is None:
        raise SecGemmaLeanV37TransportError("url_mismatch")
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    contact_value = contact.request_header_closure()()
    try:
        request = (
            f"GET {target} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"User-Agent: {contact_value}\r\n"
            "Accept: application/json,text/plain,*/*\r\n"
            "Accept-Encoding: identity\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n"
            "\r\n"
        ).encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise SecGemmaLeanV37TransportError("transport_error") from None
    return host, request


def _send_all(connection: ssl.SSLSocket, payload: bytes, deadline: float) -> None:
    sent = 0
    while sent < len(payload):
        try:
            connection.settimeout(_remaining(deadline))
            count = connection.send(payload[sent:])
        except (TimeoutError, socket.timeout):
            raise SecGemmaLeanV37TransportError("hard_timeout") from None
        except (OSError, ssl.SSLError):
            raise SecGemmaLeanV37TransportError("transport_error") from None
        if type(count) is not int or count <= 0:
            raise SecGemmaLeanV37TransportError("transport_error")
        sent += count


def _fetch_one_response(
    url: str,
    *,
    body_limit: int,
    contact: _PrivateSecContact,
    deadline: float,
) -> ParsedResponse:
    host, request = _request_bytes(url, contact)
    connection = _connect_default_tls(host, deadline=deadline)
    try:
        _send_all(connection, request, deadline)
        return _parse_from_reader(
            _DeadlineSocketReader(connection, deadline),
            body_limit=body_limit,
            contact=contact,
        )
    finally:
        try:
            connection.close()
        except BaseException:
            pass


def _validate_worker_environment() -> None:
    keys = {key.casefold() for key in os.environ}
    forbidden = {
        "all_proxy",
        "ca_bundle",
        "curl_ca_bundle",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "pythonpath",
        "requests_ca_bundle",
        "ssl_cert_dir",
        "ssl_cert_file",
    }
    allowed = {"systemroot", "windir"} if sys.platform == "win32" else set()
    if keys & forbidden or not keys <= allowed:
        raise SecGemmaLeanV37TransportError("tls_configuration_rejected")


def _safe_output_path(value: str) -> Path:
    if type(value) is not str or not value:
        raise SecGemmaLeanV37TransportError("blob_persistence_failed")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.name != path.name.lower()
        or re.fullmatch(r"[0-9a-f]{32}\.response\.tmp", path.name) is None
        or path.exists()
        or path.is_symlink()
    ):
        raise SecGemmaLeanV37TransportError("blob_persistence_failed")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError:
        raise SecGemmaLeanV37TransportError("blob_persistence_failed") from None
    if not parent.is_dir() or parent != path.parent:
        raise SecGemmaLeanV37TransportError("blob_persistence_failed")
    return path


def _write_fsynced_blob(path: Path, body: bytes) -> None:
    descriptor: int | None = None
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        descriptor = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
        created = True
        view = memoryview(body)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
    except OSError:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if created:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise SecGemmaLeanV37TransportError("blob_persistence_failed") from None


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeEncodeError):
        raise SecGemmaLeanV37TransportError("transport_error") from None


def _canonical_equal(left: Any, right: Any) -> bool:
    return _canonical_json_bytes(left) == _canonical_json_bytes(right)


def _worker_success_payload(
    *,
    url: str,
    role: SecRoleBinding,
    intent_event_sha256: str,
    output_path: Path,
    parsed: ParsedResponse,
    contact: _PrivateSecContact,
    worker_execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "body_bytes": len(parsed.body),
        "body_sha256": hashlib.sha256(parsed.body).hexdigest(),
        "body_limit_bytes": role.body_limit_bytes,
        "contact_fingerprint_sha256": contact.fingerprint_sha256,
        "content_encoding": parsed.content_encoding,
        "declared_content_length": parsed.declared_content_length,
        "execution_identity": dict(worker_execution_identity),
        "framing": parsed.framing,
        "intent_event_sha256": intent_event_sha256,
        "observed_url": url,
        "raw_headers_bytes": len(parsed.raw_head),
        "raw_headers_sha256": hashlib.sha256(parsed.raw_head).hexdigest(),
        "request_url": url,
        "role_class": role.role_class,
        "role_id": role.role_id,
        "status_code": parsed.status_code,
        "temporary_blob_path": str(output_path),
    }


def _read_worker_request() -> dict[str, Any]:
    payload = sys.stdin.buffer.read(MAX_WORKER_REQUEST_BYTES + 1)
    if not payload or len(payload) > MAX_WORKER_REQUEST_BYTES:
        raise SecGemmaLeanV37TransportError("transport_error")
    try:
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanV37TransportError("transport_error") from None
    if (
        type(decoded) is not dict
        or set(decoded)
        != {
            "body_limit",
            "intent_event_sha256",
            "output_path",
            "private_contact",
            "request_url",
            "role_id",
        }
        or type(decoded["body_limit"]) is not int
        or type(decoded["body_limit"]) is bool
        or decoded["body_limit"] not in ALLOWED_ROLE_BODY_LIMITS
        or type(decoded["request_url"]) is not str
        or type(decoded["role_id"]) is not str
        or type(decoded["intent_event_sha256"]) is not str
        or _SHA256_RE.fullmatch(decoded["intent_event_sha256"]) is None
        or type(decoded["private_contact"]) is not str
        or type(decoded["output_path"]) is not str
    ):
        raise SecGemmaLeanV37TransportError("transport_error")
    return decoded


def _worker_main() -> int:
    output_path: Path | None = None
    try:
        _validate_worker_environment()
        request = _read_worker_request()
        output_path = _safe_output_path(request["output_path"])
        url = canonical_sec_url(request["request_url"])
        role = derive_sec_role(url)
        if (
            request["role_id"] != role.role_id
            or request["body_limit"] != role.body_limit_bytes
        ):
            raise SecGemmaLeanV37TransportError("url_mismatch")
        try:
            contact = _PrivateSecContact(request["private_contact"])
        except Exception:
            raise SecGemmaLeanV37TransportError("transport_error") from None
        worker_execution_identity = execution_identity()
        deadline = time.monotonic() + REQUEST_DEADLINE_SECONDS
        parsed = _fetch_one_response(
            url,
            body_limit=request["body_limit"],
            contact=contact,
            deadline=deadline,
        )
        _write_fsynced_blob(output_path, parsed.body)
        result = {
            "metadata": _worker_success_payload(
                url=url,
                role=role,
                intent_event_sha256=request["intent_event_sha256"],
                output_path=output_path,
                parsed=parsed,
                contact=contact,
                worker_execution_identity=worker_execution_identity,
            ),
            "schema_version": TRANSPORT_SCHEMA_VERSION,
            "status": "ok",
        }
        encoded = _canonical_json_bytes(result)
        if len(encoded) > MAX_WORKER_RESPONSE_BYTES:
            raise SecGemmaLeanV37TransportError("transport_error")
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
        return 0
    except SecGemmaLeanV37TransportError as error:
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
        encoded = _canonical_json_bytes(
            {
                "code": error.code,
                "observed_body_bytes": error.observed_body_bytes,
                "schema_version": TRANSPORT_SCHEMA_VERSION,
                "status": "error",
            }
        )
        sys.stdout.buffer.write(encoded)
        sys.stdout.buffer.flush()
        return 2
    except BaseException:
        if output_path is not None:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            sys.stdout.buffer.write(
                _canonical_json_bytes(
                    {
                        "code": "transport_error",
                        "observed_body_bytes": 0,
                        "schema_version": TRANSPORT_SCHEMA_VERSION,
                        "status": "error",
                    }
                )
            )
            sys.stdout.buffer.flush()
        except BaseException:
            pass
        return 2


def _sanitized_worker_environment() -> dict[str, str]:
    if sys.platform != "win32":
        return {}
    environment: dict[str, str] = {}
    for required in ("SYSTEMROOT", "WINDIR"):
        value = next(
            (
                candidate
                for key, candidate in os.environ.items()
                if key.casefold() == required.casefold()
            ),
            None,
        )
        if type(value) is not str or not value or "\x00" in value:
            raise SecGemmaLeanV37TransportError("tls_configuration_rejected")
        try:
            root = Path(value).resolve(strict=True)
        except OSError:
            raise SecGemmaLeanV37TransportError(
                "tls_configuration_rejected"
            ) from None
        if not root.is_dir():
            raise SecGemmaLeanV37TransportError("tls_configuration_rejected")
        environment[required] = str(root)
    if Path(environment["SYSTEMROOT"]) != Path(environment["WINDIR"]):
        raise SecGemmaLeanV37TransportError("tls_configuration_rejected")
    return environment


def _terminate_worker(process: Any) -> None:
    """Kill and reap the worker before control can leave the fetch call."""

    try:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            # A successful OS kill must be followed by an unconditional reap.
            process.wait()
        except BaseException:
            raise SecGemmaLeanV37TransportError("transport_error") from None
    except BaseException:
        raise SecGemmaLeanV37TransportError("transport_error") from None
    try:
        returncode = process.poll()
    except BaseException:
        raise SecGemmaLeanV37TransportError("transport_error") from None
    if returncode is None:
        raise SecGemmaLeanV37TransportError("transport_error")


def _validated_temporary_directory(directory: Path) -> Path:
    if not isinstance(directory, Path) or not directory.is_absolute():
        raise SecGemmaLeanV37TransportError("blob_persistence_failed")
    try:
        resolved = directory.resolve(strict=True)
    except OSError:
        raise SecGemmaLeanV37TransportError("blob_persistence_failed") from None
    if resolved != directory or not resolved.is_dir() or directory.is_symlink():
        raise SecGemmaLeanV37TransportError("blob_persistence_failed")
    return resolved


def _decode_worker_output(
    stdout: bytes,
    *,
    expected_url: str,
    expected_path: Path,
    expected_contact_fingerprint: str,
    expected_role: SecRoleBinding,
    expected_intent_event_sha256: str,
    expected_execution_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    if type(stdout) is not bytes or not stdout or len(stdout) > MAX_WORKER_RESPONSE_BYTES:
        raise SecGemmaLeanV37TransportError("transport_error")
    try:
        decoded = json.loads(stdout.decode("ascii", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanV37TransportError("transport_error") from None
    if type(decoded) is not dict or decoded.get("schema_version") != TRANSPORT_SCHEMA_VERSION:
        raise SecGemmaLeanV37TransportError("transport_error")
    if decoded.get("status") == "error":
        if set(decoded) != {
            "code",
            "observed_body_bytes",
            "schema_version",
            "status",
        }:
            raise SecGemmaLeanV37TransportError("transport_error")
        raise SecGemmaLeanV37TransportError(
            decoded.get("code", "transport_error"),
            observed_body_bytes=decoded.get("observed_body_bytes", 0),
        )
    if set(decoded) != {"metadata", "schema_version", "status"} or decoded.get(
        "status"
    ) != "ok":
        raise SecGemmaLeanV37TransportError("transport_error")
    metadata = decoded["metadata"]
    expected_fields = {
        "body_bytes",
        "body_limit_bytes",
        "body_sha256",
        "contact_fingerprint_sha256",
        "content_encoding",
        "declared_content_length",
        "execution_identity",
        "framing",
        "intent_event_sha256",
        "observed_url",
        "raw_headers_bytes",
        "raw_headers_sha256",
        "request_url",
        "role_class",
        "role_id",
        "status_code",
        "temporary_blob_path",
    }
    if (
        type(metadata) is not dict
        or set(metadata) != expected_fields
        or metadata["request_url"] != expected_url
        or metadata["temporary_blob_path"] != str(expected_path)
        or metadata["observed_url"] != expected_url
        or metadata["role_id"] != expected_role.role_id
        or metadata["role_class"] != expected_role.role_class
        or metadata["body_limit_bytes"] != expected_role.body_limit_bytes
        or metadata["intent_event_sha256"] != expected_intent_event_sha256
        or not _canonical_equal(
            metadata["execution_identity"], dict(expected_execution_identity)
        )
        or metadata["contact_fingerprint_sha256"]
        != expected_contact_fingerprint
        or metadata["status_code"] != 200
        or metadata["framing"] not in {"content-length", "chunked", "eof"}
        or metadata["content_encoding"] not in {None, "identity"}
        or type(metadata["raw_headers_bytes"]) is not int
        or type(metadata["raw_headers_bytes"]) is bool
        or not 4 <= metadata["raw_headers_bytes"] <= MAX_RESPONSE_HEAD_BYTES
        or type(metadata["body_bytes"]) is not int
        or type(metadata["body_bytes"]) is bool
        or metadata["body_bytes"] < 0
        or (
            metadata["declared_content_length"] is not None
            and (
                type(metadata["declared_content_length"]) is not int
                or type(metadata["declared_content_length"]) is bool
                or metadata["declared_content_length"] < 0
            )
        )
        or _SHA256_RE.fullmatch(str(metadata["body_sha256"])) is None
        or _SHA256_RE.fullmatch(str(metadata["raw_headers_sha256"])) is None
        or _SHA256_RE.fullmatch(str(metadata["contact_fingerprint_sha256"]))
        is None
        or (
            metadata["framing"] == "content-length"
            and metadata["declared_content_length"] != metadata["body_bytes"]
        )
        or (
            metadata["framing"] != "content-length"
            and metadata["declared_content_length"] is not None
        )
    ):
        raise SecGemmaLeanV37TransportError("transport_error")
    return metadata


def build_transport_receipt(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build the deterministic, redacted receipt bound by the journal."""

    if type(metadata) is not dict:
        raise SecGemmaLeanV37TransportError("transport_error")
    required = {
        "body_bytes",
        "body_limit_bytes",
        "body_sha256",
        "contact_fingerprint_sha256",
        "content_encoding",
        "declared_content_length",
        "execution_identity",
        "framing",
        "intent_event_sha256",
        "observed_url",
        "raw_headers_bytes",
        "raw_headers_sha256",
        "request_url",
        "role_class",
        "role_id",
        "status_code",
        "temporary_blob_path",
    }
    if set(metadata) != required:
        raise SecGemmaLeanV37TransportError("transport_error")
    try:
        request_url = canonical_sec_url(metadata["request_url"])
        role = derive_sec_role(request_url)
        identity = execution_identity()
    except Exception:
        raise SecGemmaLeanV37TransportError("transport_error") from None
    if (
        metadata["status_code"] != 200
        or metadata["observed_url"] != request_url
        or metadata["role_class"] != role.role_class
        or metadata["role_id"] != role.role_id
        or metadata["body_limit_bytes"] != role.body_limit_bytes
        or type(metadata["intent_event_sha256"]) is not str
        or _SHA256_RE.fullmatch(metadata["intent_event_sha256"]) is None
        or not _canonical_equal(metadata["execution_identity"], identity)
        or metadata["framing"] not in {"content-length", "chunked", "eof"}
        or metadata["content_encoding"] not in {None, "identity"}
        or type(metadata["body_bytes"]) is not int
        or type(metadata["body_bytes"]) is bool
        or not 0 <= metadata["body_bytes"] <= role.body_limit_bytes
        or type(metadata["raw_headers_bytes"]) is not int
        or type(metadata["raw_headers_bytes"]) is bool
        or not 4 <= metadata["raw_headers_bytes"] <= MAX_RESPONSE_HEAD_BYTES
        or type(metadata["body_sha256"]) is not str
        or _SHA256_RE.fullmatch(metadata["body_sha256"]) is None
        or type(metadata["raw_headers_sha256"]) is not str
        or _SHA256_RE.fullmatch(metadata["raw_headers_sha256"]) is None
        or type(metadata["contact_fingerprint_sha256"]) is not str
        or _SHA256_RE.fullmatch(metadata["contact_fingerprint_sha256"]) is None
        or (
            metadata["declared_content_length"] is not None
            and (
                type(metadata["declared_content_length"]) is not int
                or type(metadata["declared_content_length"]) is bool
                or metadata["declared_content_length"] < 0
            )
        )
        or type(metadata["temporary_blob_path"]) is not str
        or (
            metadata["framing"] == "content-length"
            and metadata["declared_content_length"] != metadata["body_bytes"]
        )
        or (
            metadata["framing"] != "content-length"
            and metadata["declared_content_length"] is not None
        )
    ):
        raise SecGemmaLeanV37TransportError("transport_error")
    capability = json.loads(
        _canonical_json_bytes(asdict(transport_capability())).decode("ascii")
    )
    unsigned = {
        "body": {
            "byte_length": metadata["body_bytes"],
            "sha256": metadata["body_sha256"],
        },
        "deadline": {
            "duration_milliseconds": int(REQUEST_DEADLINE_SECONDS * 1000),
            "enforcement": "parent_process_hard_monotonic",
            "scope": "dns_tcp_tls_headers_body",
        },
        "execution_identity": identity,
        "http": {
            "content_encoding": (
                "absent"
                if metadata["content_encoding"] is None
                else metadata["content_encoding"]
            ),
            "declared_content_length": metadata["declared_content_length"],
            "framing_mode": metadata["framing"],
            "observed_url": metadata["observed_url"],
            "requested_url": request_url,
            "status_code": metadata["status_code"],
        },
        "journal_binding": {
            "intent_event_sha256": metadata["intent_event_sha256"],
            "role_id": role.role_id,
        },
        "privacy_scan": {
            "complete_body_scanned": True,
            "contact_fingerprint_sha256": metadata[
                "contact_fingerprint_sha256"
            ],
            "forms": list(PRIVACY_SCAN_FORMS),
            "raw_response_headers_scanned": True,
        },
        "raw_response_headers": {
            "byte_length": metadata["raw_headers_bytes"],
            "sha256": metadata["raw_headers_sha256"],
        },
        "role": {
            "body_limit_bytes": role.body_limit_bytes,
            "role_class": role.role_class,
        },
        "schema_version": TRANSPORT_RECEIPT_SCHEMA_VERSION,
        "transport_capability": capability,
    }
    receipt_hash = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    return {**unsigned, "transport_receipt_sha256": receipt_hash}


def validate_transport_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any],
) -> None:
    """Validate every receipt key, type, value, binding, and self-hash."""

    root_keys = {
        "body",
        "deadline",
        "execution_identity",
        "http",
        "journal_binding",
        "privacy_scan",
        "raw_response_headers",
        "role",
        "schema_version",
        "transport_capability",
        "transport_receipt_sha256",
    }
    if type(receipt) is not dict or set(receipt) != root_keys:
        raise SecGemmaLeanV37TransportError("transport_error")
    nested_keys = {
        "body": {"byte_length", "sha256"},
        "deadline": {"duration_milliseconds", "enforcement", "scope"},
        "http": {
            "content_encoding",
            "declared_content_length",
            "framing_mode",
            "observed_url",
            "requested_url",
            "status_code",
        },
        "journal_binding": {"intent_event_sha256", "role_id"},
        "privacy_scan": {
            "complete_body_scanned",
            "contact_fingerprint_sha256",
            "forms",
            "raw_response_headers_scanned",
        },
        "raw_response_headers": {"byte_length", "sha256"},
        "role": {"body_limit_bytes", "role_class"},
    }
    if any(
        type(receipt[name]) is not dict
        or set(receipt[name]) != expected_keys
        for name, expected_keys in nested_keys.items()
    ):
        raise SecGemmaLeanV37TransportError("transport_error")
    observed_hash = receipt["transport_receipt_sha256"]
    if type(observed_hash) is not str or _SHA256_RE.fullmatch(observed_hash) is None:
        raise SecGemmaLeanV37TransportError("transport_error")
    unsigned = dict(receipt)
    del unsigned["transport_receipt_sha256"]
    if hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() != observed_hash:
        raise SecGemmaLeanV37TransportError("transport_error")
    try:
        identity = execution_identity()
        capability = json.loads(
            _canonical_json_bytes(asdict(transport_capability())).decode("ascii")
        )
        http = receipt["http"]
        requested_url = canonical_sec_url(http["requested_url"])
        role = derive_sec_role(requested_url)
    except Exception:
        raise SecGemmaLeanV37TransportError("transport_error") from None
    body = receipt["body"]
    header = receipt["raw_response_headers"]
    privacy = receipt["privacy_scan"]
    journal = receipt["journal_binding"]
    role_value = receipt["role"]
    declared = http["declared_content_length"]
    if (
        receipt["schema_version"] != TRANSPORT_RECEIPT_SCHEMA_VERSION
        or not _canonical_equal(receipt["execution_identity"], identity)
        or not _canonical_equal(receipt["transport_capability"], capability)
        or http["observed_url"] != requested_url
        or http["status_code"] != 200
        or type(http["status_code"]) is not int
        or http["content_encoding"] not in {"absent", "identity"}
        or http["framing_mode"] not in {"content-length", "chunked", "eof"}
        or (
            declared is not None
            and (
                type(declared) is not int
                or type(declared) is bool
                or declared < 0
            )
        )
        or (
            http["framing_mode"] == "content-length"
            and declared != body["byte_length"]
        )
        or (
            http["framing_mode"] != "content-length" and declared is not None
        )
        or journal["role_id"] != role.role_id
        or type(journal["intent_event_sha256"]) is not str
        or _SHA256_RE.fullmatch(journal["intent_event_sha256"]) is None
        or not _canonical_equal(
            role_value,
            {
                "body_limit_bytes": role.body_limit_bytes,
                "role_class": role.role_class,
            },
        )
        or type(body["byte_length"]) is not int
        or type(body["byte_length"]) is bool
        or not 0 <= body["byte_length"] <= role.body_limit_bytes
        or type(body["sha256"]) is not str
        or _SHA256_RE.fullmatch(body["sha256"]) is None
        or type(header["byte_length"]) is not int
        or type(header["byte_length"]) is bool
        or not 4 <= header["byte_length"] <= MAX_RESPONSE_HEAD_BYTES
        or type(header["sha256"]) is not str
        or _SHA256_RE.fullmatch(header["sha256"]) is None
        or type(privacy["contact_fingerprint_sha256"]) is not str
        or _SHA256_RE.fullmatch(privacy["contact_fingerprint_sha256"]) is None
        or not _canonical_equal(
            privacy,
            {
                "complete_body_scanned": True,
                "contact_fingerprint_sha256": privacy[
                    "contact_fingerprint_sha256"
                ],
                "forms": list(PRIVACY_SCAN_FORMS),
                "raw_response_headers_scanned": True,
            },
        )
        or not _canonical_equal(
            receipt["deadline"],
            {
                "duration_milliseconds": int(REQUEST_DEADLINE_SECONDS * 1000),
                "enforcement": "parent_process_hard_monotonic",
                "scope": "dns_tcp_tls_headers_body",
            },
        )
    ):
        raise SecGemmaLeanV37TransportError("transport_error")
    expected = build_transport_receipt(expected_metadata)
    if not _canonical_equal(receipt, expected):
        raise SecGemmaLeanV37TransportError("transport_error")


class StrictSecTransport:
    """One-role parent transport with a process-enforced total deadline."""

    __slots__ = ("_contact", "_temporary_directory")

    def __init__(self, *, private_contact: str, temporary_directory: Path) -> None:
        try:
            contact = _PrivateSecContact(private_contact)
        except Exception:
            raise SecGemmaLeanV37TransportError("transport_error") from None
        self._contact = contact
        self._temporary_directory = _validated_temporary_directory(
            temporary_directory
        )

    def __repr__(self) -> str:
        return (
            "StrictSecTransport("
            f"contact_fingerprint_sha256={self._contact.fingerprint_sha256!r}, "
            f"temporary_directory={str(self._temporary_directory)!r})"
        )

    def safe_state(self) -> dict[str, Any]:
        return {
            "capability": asdict(transport_capability()),
            "contact_fingerprint_sha256": self._contact.fingerprint_sha256,
            "temporary_directory": str(self._temporary_directory),
        }

    def fetch(
        self,
        url: str,
        *,
        role_id: str,
        intent_event_sha256: str,
        body_limit: int,
    ) -> TransportResult:
        canonical = canonical_sec_url(url)
        role = derive_sec_role(canonical)
        if (
            type(role_id) is not str
            or role_id != role.role_id
            or type(intent_event_sha256) is not str
            or _SHA256_RE.fullmatch(intent_event_sha256) is None
        ):
            raise SecGemmaLeanV37TransportError("url_mismatch")
        if type(body_limit) is bool or body_limit != role.body_limit_bytes:
            raise SecGemmaLeanV37TransportError("body_limit_exceeded")
        output_path = self._temporary_directory / (
            secrets.token_hex(16) + ".response.tmp"
        )
        if output_path.exists():
            raise SecGemmaLeanV37TransportError("blob_persistence_failed")
        try:
            executable = Path(sys.executable).resolve(strict=True)
            repo_root = Path(__file__).resolve(strict=True).parents[1]
            environment = _sanitized_worker_environment()
            expected_execution_identity = execution_identity()
        except SecGemmaLeanV37TransportError:
            raise
        except Exception:
            raise SecGemmaLeanV37TransportError("transport_error") from None
        command = (
            str(executable),
            "-I",
            "-S",
            "-B",
            "-c",
            _WORKER_BOOTSTRAP,
            str(repo_root),
            _WORKER_MODULE,
            _WORKER_FLAG,
            expected_execution_identity["transport_source_sha256"],
            expected_execution_identity["interpreter_executable_sha256"],
        )
        request = _canonical_json_bytes(
            {
                "body_limit": body_limit,
                "intent_event_sha256": intent_event_sha256,
                "output_path": str(output_path),
                "private_contact": self._contact.request_header_closure()(),
                "request_url": canonical,
                "role_id": role_id,
            }
        )
        if len(request) > MAX_WORKER_REQUEST_BYTES:
            raise SecGemmaLeanV37TransportError("transport_error")
        creationflags = (
            int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if sys.platform == "win32"
            else 0
        )
        started = time.monotonic()
        deadline = started + REQUEST_DEADLINE_SECONDS
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=creationflags,
                cwd=str(repo_root),
                env=environment,
            )
        except Exception:
            raise SecGemmaLeanV37TransportError("transport_error") from None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate_worker(process)
                raise SecGemmaLeanV37TransportError("hard_timeout")
            try:
                stdout, _stderr = process.communicate(input=request, timeout=remaining)
            except subprocess.TimeoutExpired:
                _terminate_worker(process)
                raise SecGemmaLeanV37TransportError("hard_timeout") from None
            except Exception:
                _terminate_worker(process)
                raise SecGemmaLeanV37TransportError("transport_error") from None
            if time.monotonic() >= deadline:
                _terminate_worker(process)
                raise SecGemmaLeanV37TransportError("hard_timeout")
            metadata = _decode_worker_output(
                stdout,
                expected_url=canonical,
                expected_path=output_path,
                expected_contact_fingerprint=self._contact.fingerprint_sha256,
                expected_role=role,
                expected_intent_event_sha256=intent_event_sha256,
                expected_execution_identity=expected_execution_identity,
            )
            if process.returncode != 0:
                raise SecGemmaLeanV37TransportError("transport_error")
            try:
                file_stat = output_path.stat()
                if (
                    output_path.is_symlink()
                    or not stat.S_ISREG(file_stat.st_mode)
                    or file_stat.st_size != metadata["body_bytes"]
                ):
                    raise SecGemmaLeanV37TransportError("blob_persistence_failed")
                digest = hashlib.sha256()
                counted = 0
                with output_path.open("rb") as handle:
                    while True:
                        chunk = handle.read(BODY_READ_CHUNK_BYTES)
                        if not chunk:
                            break
                        counted += len(chunk)
                        if counted > body_limit:
                            raise SecGemmaLeanV37TransportError(
                                "blob_persistence_failed"
                            )
                        digest.update(chunk)
                if counted != metadata["body_bytes"] or digest.hexdigest() != metadata[
                    "body_sha256"
                ]:
                    raise SecGemmaLeanV37TransportError("blob_persistence_failed")
            except SecGemmaLeanV37TransportError:
                raise
            except OSError:
                raise SecGemmaLeanV37TransportError(
                    "blob_persistence_failed"
                ) from None
            elapsed_ms = int(math.ceil((time.monotonic() - started) * 1000.0))
            receipt = build_transport_receipt(metadata)
            validate_transport_receipt(receipt, expected_metadata=metadata)
            return TransportResult(
                schema_version=TRANSPORT_SCHEMA_VERSION,
                request_url=canonical,
                observed_url=metadata["observed_url"],
                role_class=role.role_class,
                role_id=role.role_id,
                intent_event_sha256=intent_event_sha256,
                body_limit_bytes=role.body_limit_bytes,
                temporary_blob_path=output_path,
                status_code=metadata["status_code"],
                framing=metadata["framing"],
                declared_content_length=metadata["declared_content_length"],
                content_encoding=metadata["content_encoding"],
                body_bytes=metadata["body_bytes"],
                body_sha256=metadata["body_sha256"],
                raw_headers_sha256=metadata["raw_headers_sha256"],
                raw_headers_bytes=metadata["raw_headers_bytes"],
                contact_fingerprint_sha256=metadata[
                    "contact_fingerprint_sha256"
                ],
                execution_identity=expected_execution_identity,
                transport_receipt=receipt,
                transport_receipt_sha256=receipt[
                    "transport_receipt_sha256"
                ],
                elapsed_milliseconds=elapsed_ms,
            )
        except SecGemmaLeanV37TransportError:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            if getattr(process, "poll", lambda: 0)() is None:
                _terminate_worker(process)


def _main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments != (_WORKER_FLAG,):
        return 91
    return _worker_main()


__all__ = [
    "ALLOWED_ROLE_BODY_LIMITS",
    "BodyFraming",
    "COMPLETE_SUBMISSION_BODY_LIMIT",
    "HISTORICAL_SUBMISSIONS_BODY_LIMIT",
    "MAIN_SUBMISSIONS_BODY_LIMIT",
    "MASTER_COMPRESSED_BODY_LIMIT",
    "ParsedResponse",
    "PRIVACY_SCAN_FORMS",
    "REQUEST_DEADLINE_SECONDS",
    "ResponseHead",
    "SecRoleBinding",
    "SAFE_ERROR_CODES",
    "SecGemmaLeanV37TransportError",
    "StrictSecTransport",
    "TRANSPORT_SCHEMA_VERSION",
    "TRANSPORT_RECEIPT_SCHEMA_VERSION",
    "TransportCapability",
    "TransportResult",
    "build_transport_receipt",
    "canonical_sec_url",
    "derive_sec_role",
    "execution_identity",
    "parse_http_response_head",
    "parse_raw_http_response",
    "privacy_echo_needles",
    "response_body_framing",
    "transport_capability",
    "validate_transport_receipt",
]


if __name__ == "__main__":
    raise SystemExit(_main())
