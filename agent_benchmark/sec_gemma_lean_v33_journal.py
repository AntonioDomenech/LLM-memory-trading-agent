"""Dynamic, append-only SEC request journal for lean evidence v3.3.

The journal starts with only the exact main Apple Submissions request.  Three
hash-chained ``phase_plan`` events then grow the request queue: historical
Submissions files (H), fixed quarterly masters (Q), and new complete submissions
(P).  This module is offline and never opens a socket or adopts response bytes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Final
from urllib.parse import quote_from_bytes, quote_plus

from agent_benchmark.sec_point_in_time import (
    SecPointInTimeError,
    validate_sec_user_agent,
)


JOURNAL_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-3-dynamic-request-journal-v1"
)
SEALED_PREFIX_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-3-sealed-prefix-v1"
)
PHASE_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-3-phase-plan-v1"
)

MAIN_SUBMISSIONS_URL: Final[str] = (
    "https://data.sec.gov/submissions/CIK0000320193.json"
)
HISTORICAL_SUBMISSIONS_URL_PREFIX: Final[str] = (
    "https://data.sec.gov/submissions/"
)
MASTER_URL_TEMPLATE: Final[str] = (
    "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.gz"
)
COMPLETE_SUBMISSION_URL_PREFIX: Final[str] = (
    "https://www.sec.gov/Archives/edgar/data/320193/"
)

MIB: Final[int] = 1024 * 1024
MAIN_SUBMISSIONS_BODY_LIMIT: Final[int] = 32 * MIB
HISTORICAL_SUBMISSIONS_BODY_LIMIT: Final[int] = 64 * MIB
MASTER_COMPRESSED_BODY_LIMIT: Final[int] = 16 * MIB
MASTER_DECOMPRESSED_BODY_LIMIT: Final[int] = 128 * MIB
COMPLETE_SUBMISSION_BODY_LIMIT: Final[int] = 128 * MIB
SUCCESSFUL_RESPONSE_BYTE_CAP: Final[int] = 19_964_887_040
LIFETIME_RECEIVED_BYTE_CAP: Final[int] = 21_474_836_480
MASTER_DECOMPRESSED_BYTE_CAP: Final[int] = 17_582_522_368
LIFETIME_INTENT_CAP: Final[int] = 256
HISTORICAL_REFERENCE_CAP: Final[int] = 16

FIRST_DISPATCH_WAIT_MS: Final[int] = 1_000
GLOBAL_DISPATCH_GAP_MS: Final[int] = 500
REQUEST_DEADLINE_MS: Final[int] = 30_000
ROLE_DEADLINE_MS: Final[int] = 600_000
INVOCATION_DEADLINE_MS: Final[int] = 14_400_000
STAGE_ACTIVE_TIME_CAP_MS: Final[int] = 43_200_000

PHASE_BODY_LIMITS: Final[dict[str, int]] = {
    "main_submissions": MAIN_SUBMISSIONS_BODY_LIMIT,
    "historical_submissions": HISTORICAL_SUBMISSIONS_BODY_LIMIT,
    "quarterly_master": MASTER_COMPRESSED_BODY_LIMIT,
    "complete_submission": COMPLETE_SUBMISSION_BODY_LIMIT,
}
PHASE_EXPANSION_ORDER: Final[tuple[str, ...]] = (
    "historical_submissions",
    "quarterly_master",
    "complete_submission",
)
PHASE_BASIS_KINDS: Final[dict[str, str]] = {
    "historical_submissions": "main_parse_receipt",
    "quarterly_master": "complete_submissions_snapshot_receipt",
    "complete_submission": "master_reconciliation_and_prior_carry_receipt",
}
EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "invocation_open",
        "phase_plan",
        "role_intent",
        "response_complete",
        "role_seal",
        "role_failure",
        "invocation_clean_close",
        "terminal",
    }
)
ROLE_FAILURE_CODES: Final[frozenset[str]] = frozenset(
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
        "interrupted_after_intent",
        "parse_rejected",
        "premature_eof",
        "privacy_echo",
        "redirect_rejected",
        "response_length_mismatch",
        "role_deadline_exceeded",
        "tls_configuration_rejected",
        "trailers_rejected",
        "transfer_encoding_rejected",
        "transport_error",
        "url_mismatch",
    }
)
SOURCE_DIAGNOSTIC_CODES: Final[frozenset[str]] = frozenset(
    {"master_boundary_incomplete"}
)
TERMINAL_REJECTION_CODES: Final[frozenset[str]] = frozenset(
    set(ROLE_FAILURE_CODES)
    | set(SOURCE_DIAGNOSTIC_CODES)
    | {
        "cumulative_source_deadline_exceeded",
        "integrity_rejected",
        "invocation_deadline_exceeded",
        "open_intent",
        "orphan_blob",
        "source_rejected",
        "unclosed_invocation",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HISTORICAL_NAME_RE = re.compile(
    r"CIK0000320193-submissions-[0-9]{3}\.json\Z"
)
_ACCESSION_RE = re.compile(r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z")
_ROLE_ID_RE = re.compile(r"[A-Za-z0-9._/-]{1,200}\Z")
_EVENT_NAME_RE = re.compile(r"[0-9]{8}\.json\Z")
_PENDING_NAME_RE = re.compile(
    r"\.pending-[0-9]{8}-[0-9a-f]{64}\.json\Z"
)
_FAILED_APPEND_NAME_RE = re.compile(
    r"\.append-failed-[0-9]{8}-[0-9a-f]{64}\.json\Z"
)
_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"\.[0-9]{6}Z\Z"
)
_REPARSE_ATTRIBUTE: Final[int] = 0x400
_MAX_EVENT_BYTES: Final[int] = MIB


class SecGemmaLeanV33JournalError(RuntimeError):
    """A fixed-code journal rejection that never contains third-party text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class StageConfig:
    stage: str
    last_master_year: int
    last_master_quarter: int
    quarterly_master_count: int
    complete_request_cap: int
    maximum_successful_requests: int
    filing_cutoff: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "master_start": "1994-Q1",
            "last_master_year": self.last_master_year,
            "last_master_quarter": self.last_master_quarter,
            "quarterly_master_count": self.quarterly_master_count,
            "historical_reference_cap": HISTORICAL_REFERENCE_CAP,
            "complete_request_cap": self.complete_request_cap,
            "maximum_successful_requests": self.maximum_successful_requests,
            "filing_cutoff": self.filing_cutoff,
            "lifetime_intent_cap": LIFETIME_INTENT_CAP,
            "successful_response_byte_cap": SUCCESSFUL_RESPONSE_BYTE_CAP,
            "lifetime_received_byte_cap": LIFETIME_RECEIVED_BYTE_CAP,
            "master_decompressed_byte_cap": MASTER_DECOMPRESSED_BYTE_CAP,
        }


STAGE_CONFIGS: Final[dict[str, StageConfig]] = {
    "development": StageConfig("development", 2018, 4, 100, 128, 245, "2018-12-31"),
    "intermediate": StageConfig(
        "intermediate", 2023, 4, 120, 24, 161, "2023-12-31"
    ),
    "final": StageConfig("final", 2026, 3, 131, 16, 164, "2026-07-09"),
}


@dataclass(frozen=True)
class CompleteSubmissionTarget:
    filing_date: str
    accession: str

    @property
    def url(self) -> str:
        return f"{COMPLETE_SUBMISSION_URL_PREFIX}{self.accession}.txt"


@dataclass(frozen=True)
class RequestRole:
    ordinal: int
    role_id: str
    phase: str
    url: str
    body_limit_bytes: int
    decompressed_body_limit_bytes: int | None
    filing_date: str | None = None

    @property
    def reservation_bytes(self) -> int:
        return self.body_limit_bytes + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "role_id": self.role_id,
            "phase": self.phase,
            "url": self.url,
            "body_limit_bytes": self.body_limit_bytes,
            "decompressed_body_limit_bytes": self.decompressed_body_limit_bytes,
            "filing_date": self.filing_date,
            "reservation_bytes": self.reservation_bytes,
        }


@dataclass(frozen=True)
class StageRolePlan:
    stage: str
    historical_count: int
    quarterly_master_count: int
    complete_submission_count: int
    expected_successful_requests: int
    maximum_successful_requests: int
    roles: tuple[RequestRole, ...]

    def commitment_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE_PLAN_SCHEMA_VERSION,
            "stage": self.stage,
            "historical_count": self.historical_count,
            "quarterly_master_count": self.quarterly_master_count,
            "complete_submission_count": self.complete_submission_count,
            "formula": "1 + H + Q + P",
            "expected_successful_requests": self.expected_successful_requests,
            "maximum_successful_requests": self.maximum_successful_requests,
            "roles": [role.as_dict() for role in self.roles],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.commitment_dict())


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError):
        raise SecGemmaLeanV33JournalError("journal_value_invalid") from None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _valid_sha256(value: Any) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _main_role() -> RequestRole:
    return RequestRole(
        1,
        "submissions/main",
        "main_submissions",
        MAIN_SUBMISSIONS_URL,
        MAIN_SUBMISSIONS_BODY_LIMIT,
        None,
    )


def _master_quarters(config: StageConfig) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    year, quarter = 1994, 1
    while (year, quarter) <= (config.last_master_year, config.last_master_quarter):
        result.append((year, quarter))
        year, quarter = (year + 1, 1) if quarter == 4 else (year, quarter + 1)
    if len(result) != config.quarterly_master_count:
        raise SecGemmaLeanV33JournalError("frozen_quarter_formula_invalid")
    return tuple(result)


def _historical_roles(names: Sequence[str], start: int) -> tuple[RequestRole, ...]:
    if type(names) not in {list, tuple}:
        raise SecGemmaLeanV33JournalError("historical_plan_invalid")
    values = list(names)
    if (
        len(values) > HISTORICAL_REFERENCE_CAP
        or any(
            type(name) is not str or _HISTORICAL_NAME_RE.fullmatch(name) is None
            for name in values
        )
        or values != sorted(values)
        or len(set(values)) != len(values)
    ):
        raise SecGemmaLeanV33JournalError("historical_plan_invalid")
    return tuple(
        RequestRole(
            start + index,
            f"submissions/historical/{name}",
            "historical_submissions",
            f"{HISTORICAL_SUBMISSIONS_URL_PREFIX}{name}",
            HISTORICAL_SUBMISSIONS_BODY_LIMIT,
            None,
        )
        for index, name in enumerate(values)
    )


def _master_roles(stage: str, start: int) -> tuple[RequestRole, ...]:
    config = STAGE_CONFIGS[stage]
    return tuple(
        RequestRole(
            start + index,
            f"master/{year}/QTR{quarter}",
            "quarterly_master",
            MASTER_URL_TEMPLATE.format(year=year, quarter=quarter),
            MASTER_COMPRESSED_BODY_LIMIT,
            MASTER_DECOMPRESSED_BODY_LIMIT,
        )
        for index, (year, quarter) in enumerate(_master_quarters(config))
    )


def _complete_roles(
    stage: str, targets: Sequence[CompleteSubmissionTarget], start: int
) -> tuple[RequestRole, ...]:
    if type(targets) not in {list, tuple}:
        raise SecGemmaLeanV33JournalError("complete_target_plan_invalid")
    config = STAGE_CONFIGS[stage]
    checked: list[CompleteSubmissionTarget] = []
    for target in targets:
        if type(target) is not CompleteSubmissionTarget:
            raise SecGemmaLeanV33JournalError("complete_target_plan_invalid")
        try:
            filing_day = date.fromisoformat(target.filing_date)
        except (TypeError, ValueError):
            raise SecGemmaLeanV33JournalError("complete_target_plan_invalid") from None
        if (
            filing_day.isoformat() != target.filing_date
            or not date(1994, 1, 1)
            <= filing_day
            <= date.fromisoformat(config.filing_cutoff)
            or _ACCESSION_RE.fullmatch(target.accession) is None
        ):
            raise SecGemmaLeanV33JournalError("complete_target_plan_invalid")
        checked.append(target)
    if (
        len(checked) > config.complete_request_cap
        or checked != sorted(checked, key=lambda item: (item.filing_date, item.accession))
        or len({item.accession for item in checked}) != len(checked)
    ):
        raise SecGemmaLeanV33JournalError(
            "complete_request_cap_exceeded"
            if len(checked) > config.complete_request_cap
            else "complete_target_plan_invalid"
        )
    return tuple(
        RequestRole(
            start + index,
            f"complete/{target.accession}",
            "complete_submission",
            target.url,
            COMPLETE_SUBMISSION_BODY_LIMIT,
            None,
            target.filing_date,
        )
        for index, target in enumerate(checked)
    )


def build_stage_role_plan(
    stage: str,
    historical_filenames: Sequence[str],
    complete_submissions: Sequence[CompleteSubmissionTarget],
) -> StageRolePlan:
    """Build the final plan for detached comparison, never initial authority."""

    if type(stage) is not str or stage not in STAGE_CONFIGS:
        raise SecGemmaLeanV33JournalError("stage_invalid")
    roles: list[RequestRole] = [_main_role()]
    roles.extend(_historical_roles(historical_filenames, len(roles) + 1))
    roles.extend(_master_roles(stage, len(roles) + 1))
    roles.extend(_complete_roles(stage, complete_submissions, len(roles) + 1))
    config = STAGE_CONFIGS[stage]
    expected = len(roles)
    plan = StageRolePlan(
        stage,
        len(historical_filenames),
        config.quarterly_master_count,
        len(complete_submissions),
        expected,
        config.maximum_successful_requests,
        tuple(roles),
    )
    validate_stage_role_plan(plan)
    return plan


def validate_stage_role_plan(plan: StageRolePlan) -> None:
    if (
        type(plan) is not StageRolePlan
        or type(plan.stage) is not str
        or plan.stage not in STAGE_CONFIGS
        or type(plan.roles) is not tuple
        or any(type(role) is not RequestRole for role in plan.roles)
        or type(plan.historical_count) is not int
        or type(plan.quarterly_master_count) is not int
        or type(plan.complete_submission_count) is not int
        or type(plan.expected_successful_requests) is not int
        or type(plan.maximum_successful_requests) is not int
        or any(
            type(role.ordinal) is not int
            or type(role.role_id) is not str
            or type(role.phase) is not str
            or type(role.url) is not str
            or type(role.body_limit_bytes) is not int
            or (
                role.decompressed_body_limit_bytes is not None
                and type(role.decompressed_body_limit_bytes) is not int
            )
            or (role.filing_date is not None and type(role.filing_date) is not str)
            for role in plan.roles
        )
    ):
        raise SecGemmaLeanV33JournalError("role_plan_invalid")
    config = STAGE_CONFIGS[plan.stage]
    expected = 1 + plan.historical_count + config.quarterly_master_count + plan.complete_submission_count
    if (
        not 0 <= plan.historical_count <= HISTORICAL_REFERENCE_CAP
        or plan.quarterly_master_count != config.quarterly_master_count
        or not 0 <= plan.complete_submission_count <= config.complete_request_cap
        or plan.expected_successful_requests != expected
        or plan.maximum_successful_requests != config.maximum_successful_requests
        or len(plan.roles) != expected
        or expected > config.maximum_successful_requests
        or sum(role.body_limit_bytes for role in plan.roles)
        > SUCCESSFUL_RESPONSE_BYTE_CAP
        or config.quarterly_master_count * MASTER_DECOMPRESSED_BODY_LIMIT
        > MASTER_DECOMPRESSED_BYTE_CAP
        or (
            plan.stage == "final"
            and config.quarterly_master_count * MASTER_DECOMPRESSED_BODY_LIMIT
            != MASTER_DECOMPRESSED_BYTE_CAP
        )
    ):
        raise SecGemmaLeanV33JournalError("role_plan_invalid")
    historical = tuple(
        role.role_id.removeprefix("submissions/historical/")
        for role in plan.roles[1 : 1 + plan.historical_count]
    )
    complete_start = 1 + plan.historical_count + config.quarterly_master_count
    targets = tuple(
        CompleteSubmissionTarget(role.filing_date or "", role.role_id.removeprefix("complete/"))
        for role in plan.roles[complete_start:]
    )
    expected_roles = (
        (_main_role(),)
        + _historical_roles(historical, 2)
        + _master_roles(plan.stage, 2 + plan.historical_count)
        + _complete_roles(plan.stage, targets, complete_start + 1)
    )
    if plan.roles != expected_roles:
        raise SecGemmaLeanV33JournalError("role_plan_invalid")


class PrivateSecContact:
    """Validated private contact with raw/JSON/HTML/percent echo needles."""

    __slots__ = ("__text", "_fingerprint", "_needles")

    def __init__(self, contact: str) -> None:
        if type(contact) is not str:
            raise SecGemmaLeanV33JournalError("private_contact_invalid")
        try:
            raw = contact.encode("utf-8", errors="strict")
            audit = validate_sec_user_agent(contact)
        except (UnicodeEncodeError, SecPointInTimeError):
            raise SecGemmaLeanV33JournalError("private_contact_invalid") from None
        fingerprint = _sha256_bytes(raw)
        audited = audit.sha256.removeprefix("sha256:")
        if not _valid_sha256(audited) or audited != fingerprint:
            raise SecGemmaLeanV33JournalError("private_contact_fingerprint_mismatch")
        forms = {
            raw,
            json.dumps(contact, ensure_ascii=False)[1:-1].encode("utf-8"),
            json.dumps(contact, ensure_ascii=True)[1:-1].encode("ascii"),
            html.escape(contact, quote=True).encode("utf-8"),
            html.escape(contact, quote=False).encode("utf-8"),
            quote_from_bytes(raw, safe="").encode("ascii"),
            quote_plus(contact, safe="").encode("ascii"),
        }
        forms |= {value.lower() for value in forms}
        self.__text = contact
        self._fingerprint = fingerprint
        self._needles = tuple(sorted(forms, key=lambda item: (len(item), item)))

    @property
    def fingerprint_sha256(self) -> str:
        return self._fingerprint

    def request_header_closure(self) -> Callable[[], str]:
        value = self.__text
        return lambda: value

    def serialized_echo_needles(self) -> tuple[bytes, ...]:
        return self._needles

    def contains_echo(self, value: bytes | bytearray | memoryview | str) -> bool:
        if type(value) is str:
            try:
                candidates = (
                    value.encode("utf-8", errors="strict"),
                    json.dumps(value, ensure_ascii=True).encode("ascii"),
                )
            except (UnicodeEncodeError, ValueError):
                return True
        elif type(value) is bytes:
            candidates = (value,)
        elif type(value) in {bytearray, memoryview}:
            candidates = (bytes(value),)
        else:
            raise SecGemmaLeanV33JournalError("privacy_scan_value_invalid")
        return any(needle in candidate for needle in self._needles for candidate in candidates)

    def assert_redacted(self, value: Any) -> None:
        pending: list[tuple[Any, int]] = [(value, 0)]
        seen: set[int] = set()
        while pending:
            current, depth = pending.pop()
            if depth > 96:
                raise SecGemmaLeanV33JournalError("privacy_scan_value_invalid")
            if type(current) in {str, bytes, bytearray, memoryview}:
                if self.contains_echo(current):
                    raise SecGemmaLeanV33JournalError("privacy_echo")
            elif type(current) is dict:
                identity = id(current)
                if identity in seen:
                    continue
                seen.add(identity)
                for key, child in current.items():
                    pending.extend(((key, depth + 1), (child, depth + 1)))
            elif type(current) in {list, tuple}:
                identity = id(current)
                if identity in seen:
                    continue
                seen.add(identity)
                pending.extend((child, depth + 1) for child in current)

    def __repr__(self) -> str:
        return f"PrivateSecContact(<redacted>, fingerprint_sha256={self._fingerprint})"


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def _secure_root(root: Path) -> Path:
    if not isinstance(root, Path) or not root.is_absolute():
        raise SecGemmaLeanV33JournalError("journal_root_invalid")
    cursor = Path(root.anchor)
    for component in (None, *root.parts[1:]):
        if component is not None:
            cursor /= component
        try:
            details = cursor.lstat()
        except OSError:
            raise SecGemmaLeanV33JournalError("journal_root_invalid") from None
        if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode) or _is_reparse(details):
            raise SecGemmaLeanV33JournalError("journal_root_invalid")
    return root


def _regular_file(path: Path) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError:
        raise SecGemmaLeanV33JournalError("journal_invalid") from None
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
    ):
        raise SecGemmaLeanV33JournalError("journal_invalid")
    return details


def _fsync_directory(directory: Path) -> None:
    _secure_root(directory)
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            create = kernel32.CreateFileW
            create.argtypes = (
                wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
            )
            create.restype = wintypes.HANDLE
            flush = kernel32.FlushFileBuffers
            flush.argtypes = (wintypes.HANDLE,)
            flush.restype = wintypes.BOOL
            close = kernel32.CloseHandle
            close.argtypes = (wintypes.HANDLE,)
            close.restype = wintypes.BOOL
            handle = create(str(directory), 0x40000000, 7, None, 3, 0x02000000, None)
            if handle == ctypes.c_void_p(-1).value:
                raise OSError("open")
            try:
                if not flush(handle):
                    raise OSError("flush")
            finally:
                close(handle)
            return
        except (AttributeError, OSError):
            raise SecGemmaLeanV33JournalError("journal_directory_fsync_failed") from None
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError:
        raise SecGemmaLeanV33JournalError("journal_directory_fsync_failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _write_exclusive_fsynced(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        details = _regular_file(path)
        if details.st_size != len(payload) or path.read_bytes() != payload:
            raise OSError("durable journal bytes changed")
    except SecGemmaLeanV33JournalError:
        raise
    except OSError:
        raise SecGemmaLeanV33JournalError(
            "journal_append_durability_failed"
        ) from None


def _leave_append_failure_marker(
    root: Path, *, sequence: int, event_sha256: str
) -> None:
    marker = root / f".append-failed-{sequence:08d}-{event_sha256}.json"
    body = {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "sequence_number": sequence,
        "event_sha256": event_sha256,
        "error_code": "journal_append_durability_failed",
        "authoritative_event_committed": False,
    }
    encoded = _canonical_json_bytes(body)
    try:
        if not marker.exists():
            _write_exclusive_fsynced(marker, encoded)
        _fsync_directory(root)
    except Exception:
        # The visible immutable marker is already enough for immediate replay to
        # fail closed.  Never remove it merely because directory durability is
        # unavailable; a later restart must not bless the candidate event.
        return


@dataclass(frozen=True)
class JournalState:
    stage: str
    authority_sha256: str
    role_plan_sha256: str
    event_count: int
    head_event_sha256: str | None
    invocation_count: int
    active_invocation: int | None
    open_role_id: str | None
    next_role_id: str | None
    next_required_expansion: str | None
    planned_roles: tuple[RequestRole, ...]
    sealed_role_ids: tuple[str, ...]
    phase_expansions: tuple[str, ...]
    historical_count: int | None
    quarterly_master_count: int | None
    complete_submission_count: int | None
    formula_requests: int | None
    sealed_prefix_sha256: str
    last_seal_event_sha256: str | None
    last_parse_receipt_sha256: str | None
    lifetime_intents: int
    http_200_responses: int
    role_seals: int
    successful_response_bytes: int
    lifetime_received_bytes: int
    decompressed_master_bytes: int
    cumulative_active_ms: int
    failure_code: str | None
    terminal_status: str | None
    terminal_code: str | None
    terminal_unclean_active_ms: int
    derived_terminal_code: str | None
    resume_allowed: bool
    can_finalize_pass: bool
    accounting_complete: bool
    source_authoritative: bool
    passed: bool


@dataclass
class _Replay:
    roles: list[RequestRole]
    phase_payloads: list[dict[str, Any]]
    sealed_records: list[dict[str, Any]]
    reservations: dict[str, dict[str, Any]]
    next_role_index: int = 0
    invocation_count: int = 0
    active_invocation: int | None = None
    first_dispatch_pending: bool = False
    active_role_duration_sum_ms: int = 0
    open_role: RequestRole | None = None
    open_intent_hash: str | None = None
    open_response: dict[str, Any] | None = None
    lifetime_intents: int = 0
    response_count: int = 0
    seal_count: int = 0
    successful_response_bytes: int = 0
    decompressed_master_bytes: int = 0
    cumulative_active_ms: int = 0
    failure_code: str | None = None
    deadline_code: str | None = None
    terminal_status: str | None = None
    terminal_code: str | None = None
    terminal_unclean_active_ms: int = 0

    def lifetime_received_bytes(self) -> int:
        return sum(
            reservation["exact_body_bytes"]
            if reservation["sealed"]
            else max(reservation["reservation_bytes"], reservation["observed_bytes"])
            for reservation in self.reservations.values()
        )

    def sealed_prefix_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": SEALED_PREFIX_SCHEMA_VERSION,
                "seals": self.sealed_records,
            }
        )

    def plan_sha256(self, stage: str) -> str:
        return _canonical_sha256(
            {
                "schema_version": PHASE_PLAN_SCHEMA_VERSION,
                "stage": stage,
                "initial_roles": [_main_role().as_dict()],
                "phase_plans": self.phase_payloads,
                "current_roles": [role.as_dict() for role in self.roles],
            }
        )


def _authority(stage: str, contact_fingerprint_sha256: str) -> dict[str, Any]:
    config = STAGE_CONFIGS[stage]
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "stage": stage,
        "contact_fingerprint_sha256": contact_fingerprint_sha256,
        "frozen_stage_config": config.as_dict(),
        "phase_expansion_order": list(PHASE_EXPANSION_ORDER),
        "phase_body_limits": dict(PHASE_BODY_LIMITS),
        "initial_roles": [_main_role().as_dict()],
    }


def _read_events(root: Path) -> list[dict[str, Any]]:
    root = _secure_root(root)
    try:
        entries = list(root.iterdir())
    except OSError:
        raise SecGemmaLeanV33JournalError("journal_invalid") from None
    transaction_markers = [
        path
        for path in entries
        if _PENDING_NAME_RE.fullmatch(path.name) is not None
        or _FAILED_APPEND_NAME_RE.fullmatch(path.name) is not None
    ]
    if transaction_markers:
        for marker in transaction_markers:
            details = _regular_file(marker)
            if not 2 <= details.st_size <= _MAX_EVENT_BYTES:
                raise SecGemmaLeanV33JournalError("journal_append_incomplete")
        raise SecGemmaLeanV33JournalError("journal_append_incomplete")
    if any(_EVENT_NAME_RE.fullmatch(path.name) is None for path in entries):
        raise SecGemmaLeanV33JournalError("journal_invalid")
    events: list[dict[str, Any]] = []
    for sequence, path in enumerate(sorted(entries, key=lambda item: item.name), start=1):
        if path.name != f"{sequence:08d}.json":
            raise SecGemmaLeanV33JournalError("journal_invalid")
        details = _regular_file(path)
        if not 2 <= details.st_size <= _MAX_EVENT_BYTES:
            raise SecGemmaLeanV33JournalError("journal_invalid")
        try:
            raw = path.read_bytes()
            event = json.loads(raw.decode("ascii", errors="strict"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise SecGemmaLeanV33JournalError("journal_invalid") from None
        if type(event) is not dict or raw != _canonical_json_bytes(event):
            raise SecGemmaLeanV33JournalError("journal_invalid")
        events.append(event)
    return events


def _validate_timestamp(value: Any) -> None:
    if type(value) is not str or _UTC_RE.fullmatch(value) is None:
        raise SecGemmaLeanV33JournalError("journal_invalid")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        raise SecGemmaLeanV33JournalError("journal_invalid") from None
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise SecGemmaLeanV33JournalError("journal_invalid")


def _validate_envelope(
    event: Mapping[str, Any], sequence: int, previous: str | None, authority_sha256: str
) -> None:
    keys = {
        "schema_version",
        "sequence_number",
        "previous_event_sha256",
        "authority_sha256",
        "event_type",
        "occurred_at_utc",
        "payload",
        "event_sha256",
    }
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    if (
        set(event) != keys
        or event.get("schema_version") != JOURNAL_SCHEMA_VERSION
        or type(event.get("sequence_number")) is not int
        or event.get("sequence_number") != sequence
        or event.get("previous_event_sha256") != previous
        or event.get("authority_sha256") != authority_sha256
        or type(event.get("event_type")) is not str
        or event.get("event_type") not in EVENT_TYPES
        or not _valid_sha256(event.get("event_sha256"))
        or event.get("event_sha256") != _canonical_sha256(body)
    ):
        raise SecGemmaLeanV33JournalError("journal_invalid")
    _validate_timestamp(event.get("occurred_at_utc"))


def _payload(event: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    value = event.get("payload")
    if type(value) is not dict or set(value) != keys:
        raise SecGemmaLeanV33JournalError("journal_invalid")
    return value


def _roles_from_phase_payload(
    stage: str, phase: str, role_values: Any, start: int
) -> tuple[RequestRole, ...]:
    if type(role_values) is not list or any(type(value) is not dict for value in role_values):
        raise SecGemmaLeanV33JournalError("journal_invalid")
    if phase == "historical_submissions":
        names: list[str] = []
        for value in role_values:
            role_id = value.get("role_id")
            if type(role_id) is not str or not role_id.startswith("submissions/historical/"):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            names.append(role_id.removeprefix("submissions/historical/"))
        expected = _historical_roles(tuple(names), start)
    elif phase == "quarterly_master":
        expected = _master_roles(stage, start)
    elif phase == "complete_submission":
        targets: list[CompleteSubmissionTarget] = []
        for value in role_values:
            role_id = value.get("role_id")
            filing_date = value.get("filing_date")
            if type(role_id) is not str or not role_id.startswith("complete/"):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            targets.append(
                CompleteSubmissionTarget(
                    filing_date if type(filing_date) is str else "",
                    role_id.removeprefix("complete/"),
                )
            )
        expected = _complete_roles(stage, tuple(targets), start)
    else:
        raise SecGemmaLeanV33JournalError("journal_invalid")
    if role_values != [role.as_dict() for role in expected]:
        raise SecGemmaLeanV33JournalError("journal_invalid")
    return expected


def _terminal_payload(
    replay: _Replay,
    stage: str,
    *,
    status: str,
    code: str,
    checkpoint_sha256: str | None,
    unclean_active_ms: int,
) -> dict[str, Any]:
    formula = len(replay.roles) if len(replay.phase_payloads) == 3 else None
    return {
        "status": status,
        "code": code,
        "phase_expansions": [item["phase"] for item in replay.phase_payloads],
        "role_plan_sha256": replay.plan_sha256(stage),
        "formula_requests": formula,
        "lifetime_intents": replay.lifetime_intents,
        "http_200_responses": replay.response_count,
        "role_seals": replay.seal_count,
        "successful_response_bytes": replay.successful_response_bytes,
        "lifetime_received_bytes": replay.lifetime_received_bytes(),
        "decompressed_master_bytes": replay.decompressed_master_bytes,
        "cumulative_active_ms": replay.cumulative_active_ms,
        "unclean_active_ms": unclean_active_ms,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _replay_events(
    events: Sequence[Mapping[str, Any]], stage: str, contact_fingerprint_sha256: str
) -> JournalState:
    if type(stage) is not str or stage not in STAGE_CONFIGS:
        raise SecGemmaLeanV33JournalError("stage_invalid")
    if not _valid_sha256(contact_fingerprint_sha256):
        raise SecGemmaLeanV33JournalError("contact_fingerprint_invalid")
    config = STAGE_CONFIGS[stage]
    authority_sha256 = _canonical_sha256(_authority(stage, contact_fingerprint_sha256))
    replay = _Replay([_main_role()], [], [], {})
    previous: str | None = None
    terminal_seen = False
    for sequence, event in enumerate(events, start=1):
        _validate_envelope(event, sequence, previous, authority_sha256)
        if terminal_seen:
            raise SecGemmaLeanV33JournalError("journal_invalid")
        event_type = event["event_type"]
        if event_type == "invocation_open":
            payload = _payload(event, {"invocation_number", "sealed_roles_before", "phase_expansions_before"})
            if (
                replay.active_invocation is not None
                or replay.failure_code is not None
                or replay.deadline_code is not None
                or type(payload["invocation_number"]) is not int
                or payload["invocation_number"] != replay.invocation_count + 1
                or type(payload["sealed_roles_before"]) is not int
                or payload["sealed_roles_before"] != replay.next_role_index
                or type(payload["phase_expansions_before"]) is not int
                or payload["phase_expansions_before"] != len(replay.phase_payloads)
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.invocation_count += 1
            replay.active_invocation = replay.invocation_count
            replay.first_dispatch_pending = True
            replay.active_role_duration_sum_ms = 0
        elif event_type == "phase_plan":
            payload = _payload(
                event,
                {
                    "phase_plan_schema_version",
                    "expansion_number",
                    "phase",
                    "basis_kind",
                    "basis_receipt_sha256",
                    "sealed_prefix_sha256",
                    "role_count",
                    "roles",
                },
            )
            expansion_index = len(replay.phase_payloads)
            expected_phase = (
                PHASE_EXPANSION_ORDER[expansion_index]
                if expansion_index < len(PHASE_EXPANSION_ORDER)
                else None
            )
            if (
                replay.open_role is not None
                or replay.next_role_index != len(replay.roles)
                or replay.failure_code is not None
                or replay.deadline_code is not None
                or payload["phase_plan_schema_version"] != PHASE_PLAN_SCHEMA_VERSION
                or type(payload["expansion_number"]) is not int
                or payload["expansion_number"] != expansion_index + 1
                or type(payload["phase"]) is not str
                or payload["phase"] != expected_phase
                or type(payload["basis_kind"]) is not str
                or payload["basis_kind"] != PHASE_BASIS_KINDS.get(expected_phase or "")
                or not _valid_sha256(payload["basis_receipt_sha256"])
                or payload["sealed_prefix_sha256"] != replay.sealed_prefix_sha256()
                or type(payload["role_count"]) is not int
                or payload["role_count"] < 0
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            if expected_phase == "historical_submissions":
                if (
                    len(replay.sealed_records) != 1
                    or replay.sealed_records[0]["role_id"] != "submissions/main"
                    or payload["basis_receipt_sha256"]
                    != replay.sealed_records[0]["parse_receipt_sha256"]
                ):
                    raise SecGemmaLeanV33JournalError("journal_invalid")
            roles = _roles_from_phase_payload(stage, expected_phase or "", payload["roles"], len(replay.roles) + 1)
            if payload["role_count"] != len(roles):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            if expected_phase == "quarterly_master" and len(roles) != config.quarterly_master_count:
                raise SecGemmaLeanV33JournalError("journal_invalid")
            if expected_phase == "complete_submission" and len(roles) > config.complete_request_cap:
                raise SecGemmaLeanV33JournalError("journal_invalid")
            if len(replay.roles) + len(roles) > config.maximum_successful_requests:
                raise SecGemmaLeanV33JournalError("journal_invalid")
            if (
                sum(role.body_limit_bytes for role in [*replay.roles, *roles])
                > SUCCESSFUL_RESPONSE_BYTE_CAP
                or config.quarterly_master_count * MASTER_DECOMPRESSED_BODY_LIMIT
                > MASTER_DECOMPRESSED_BYTE_CAP
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.roles.extend(roles)
            replay.phase_payloads.append(dict(payload))
        elif event_type == "role_intent":
            payload = _payload(
                event,
                {
                    "invocation_number", "role_ordinal", "role_id", "phase",
                    "url_sha256", "body_limit_bytes", "reservation_bytes", "dispatch_wait_ms",
                },
            )
            if replay.next_role_index >= len(replay.roles):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            role = replay.roles[replay.next_role_index]
            minimum_wait = FIRST_DISPATCH_WAIT_MS if replay.first_dispatch_pending else GLOBAL_DISPATCH_GAP_MS
            if (
                replay.active_invocation is None
                or replay.open_role is not None
                or replay.failure_code is not None
                or type(payload["invocation_number"]) is not int
                or payload["invocation_number"] != replay.active_invocation
                or type(payload["role_ordinal"]) is not int
                or payload["role_ordinal"] != role.ordinal
                or payload["role_id"] != role.role_id
                or payload["phase"] != role.phase
                or not _valid_sha256(payload["url_sha256"])
                or payload["url_sha256"] != _sha256_bytes(role.url.encode("ascii"))
                or type(payload["body_limit_bytes"]) is not int
                or payload["body_limit_bytes"] != role.body_limit_bytes
                or type(payload["reservation_bytes"]) is not int
                or payload["reservation_bytes"] != role.reservation_bytes
                or not _nonnegative_int(payload["dispatch_wait_ms"])
                or payload["dispatch_wait_ms"] < minimum_wait
                or role.role_id in replay.reservations
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.lifetime_intents += 1
            if replay.lifetime_intents > LIFETIME_INTENT_CAP:
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.reservations[role.role_id] = {
                "reservation_bytes": role.reservation_bytes,
                "observed_bytes": 0,
                "sealed": False,
                "exact_body_bytes": None,
            }
            if replay.lifetime_received_bytes() > LIFETIME_RECEIVED_BYTE_CAP:
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.open_role = role
            replay.open_intent_hash = event["event_sha256"]
            replay.first_dispatch_pending = False
        elif event_type == "response_complete":
            payload = _payload(
                event,
                {
                    "invocation_number", "role_id", "intent_event_sha256", "status_code",
                    "body_bytes", "body_sha256", "transport_receipt_sha256", "request_duration_ms",
                },
            )
            role = replay.open_role
            if (
                role is None
                or replay.open_response is not None
                or type(payload["invocation_number"]) is not int
                or payload["invocation_number"] != replay.active_invocation
                or payload["role_id"] != role.role_id
                or payload["intent_event_sha256"] != replay.open_intent_hash
                or payload["status_code"] != 200
                or type(payload["body_bytes"]) is not int
                or not 1 <= payload["body_bytes"] <= role.body_limit_bytes
                or not _valid_sha256(payload["body_sha256"])
                or not _valid_sha256(payload["transport_receipt_sha256"])
                or not _nonnegative_int(payload["request_duration_ms"])
                or payload["request_duration_ms"] > REQUEST_DEADLINE_MS
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.response_count += 1
            replay.successful_response_bytes += payload["body_bytes"]
            if replay.successful_response_bytes > SUCCESSFUL_RESPONSE_BYTE_CAP:
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.reservations[role.role_id]["observed_bytes"] = payload["body_bytes"]
            replay.open_response = {
                "event_sha256": event["event_sha256"],
                "body_bytes": payload["body_bytes"],
                "body_sha256": payload["body_sha256"],
                "request_duration_ms": payload["request_duration_ms"],
            }
        elif event_type == "role_seal":
            payload = _payload(
                event,
                {
                    "invocation_number", "role_id", "intent_event_sha256",
                    "response_event_sha256", "blob_sha256", "parse_receipt_sha256",
                    "body_bytes", "decompressed_bytes", "role_duration_ms",
                },
            )
            role, response = replay.open_role, replay.open_response
            decompressed = payload["decompressed_bytes"]
            if (
                role is None
                or response is None
                or type(payload["invocation_number"]) is not int
                or payload["invocation_number"] != replay.active_invocation
                or payload["role_id"] != role.role_id
                or payload["intent_event_sha256"] != replay.open_intent_hash
                or payload["response_event_sha256"] != response["event_sha256"]
                or payload["blob_sha256"] != response["body_sha256"]
                or not _valid_sha256(payload["parse_receipt_sha256"])
                or payload["body_bytes"] != response["body_bytes"]
                or not _nonnegative_int(payload["role_duration_ms"])
                or not response["request_duration_ms"] <= payload["role_duration_ms"] <= ROLE_DEADLINE_MS
                or (
                    role.phase == "quarterly_master"
                    and (type(decompressed) is not int or not 1 <= decompressed <= MASTER_DECOMPRESSED_BODY_LIMIT)
                )
                or (role.phase != "quarterly_master" and decompressed is not None)
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            if role.phase == "quarterly_master":
                replay.decompressed_master_bytes += decompressed
                if replay.decompressed_master_bytes > MASTER_DECOMPRESSED_BYTE_CAP:
                    raise SecGemmaLeanV33JournalError("journal_invalid")
            reservation = replay.reservations[role.role_id]
            reservation["sealed"] = True
            reservation["exact_body_bytes"] = response["body_bytes"]
            replay.seal_count += 1
            replay.next_role_index += 1
            replay.active_role_duration_sum_ms += payload["role_duration_ms"]
            replay.sealed_records.append(
                {
                    "ordinal": role.ordinal,
                    "role_id": role.role_id,
                    "role_seal_event_sha256": event["event_sha256"],
                    "parse_receipt_sha256": payload["parse_receipt_sha256"],
                }
            )
            replay.open_role = None
            replay.open_intent_hash = None
            replay.open_response = None
        elif event_type == "role_failure":
            payload = _payload(
                event,
                {
                    "invocation_number", "role_id", "intent_event_sha256",
                    "response_event_sha256", "error_code", "observed_body_bytes",
                    "request_duration_ms", "role_duration_ms",
                },
            )
            role = replay.open_role
            response_hash = None if replay.open_response is None else replay.open_response["event_sha256"]
            if (
                role is None
                or replay.failure_code is not None
                or type(payload["invocation_number"]) is not int
                or payload["invocation_number"] != replay.active_invocation
                or payload["role_id"] != role.role_id
                or payload["intent_event_sha256"] != replay.open_intent_hash
                or payload["response_event_sha256"] != response_hash
                or type(payload["error_code"]) is not str
                or payload["error_code"] not in ROLE_FAILURE_CODES
                or type(payload["observed_body_bytes"]) is not int
                or not 0 <= payload["observed_body_bytes"] <= role.reservation_bytes
                or not _nonnegative_int(payload["request_duration_ms"])
                or not _nonnegative_int(payload["role_duration_ms"])
                or payload["role_duration_ms"] < payload["request_duration_ms"]
                or (
                    replay.open_response is not None
                    and payload["request_duration_ms"]
                    != replay.open_response["request_duration_ms"]
                )
                or (
                    replay.open_response is not None
                    and payload["observed_body_bytes"]
                    != replay.open_response["body_bytes"]
                )
                or (
                    payload["observed_body_bytes"] == role.reservation_bytes
                    and payload["error_code"] != "body_limit_exceeded"
                )
                or (
                    payload["error_code"] == "body_limit_exceeded"
                    and payload["observed_body_bytes"] != role.reservation_bytes
                )
                or (
                    payload["role_duration_ms"] > ROLE_DEADLINE_MS
                    and payload["error_code"] != "role_deadline_exceeded"
                )
                or (
                    payload["error_code"] == "role_deadline_exceeded"
                    and payload["role_duration_ms"] <= ROLE_DEADLINE_MS
                )
                or (
                    payload["request_duration_ms"] > REQUEST_DEADLINE_MS
                    and payload["error_code"]
                    not in {"hard_timeout", "role_deadline_exceeded"}
                )
                or (
                    payload["error_code"] == "hard_timeout"
                    and payload["request_duration_ms"] < REQUEST_DEADLINE_MS
                )
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.reservations[role.role_id]["observed_bytes"] = max(
                replay.reservations[role.role_id]["observed_bytes"], payload["observed_body_bytes"]
            )
            replay.failure_code = payload["error_code"]
            replay.active_role_duration_sum_ms += payload["role_duration_ms"]
            replay.open_role = None
            replay.open_intent_hash = None
            replay.open_response = None
        elif event_type == "invocation_clean_close":
            payload = _payload(
                event,
                {"invocation_number", "invocation_duration_ms", "cumulative_active_ms", "sealed_roles_after", "phase_expansions_after"},
            )
            if (
                replay.active_invocation is None
                or replay.open_role is not None
                or type(payload["invocation_number"]) is not int
                or payload["invocation_number"] != replay.active_invocation
                or not _nonnegative_int(payload["invocation_duration_ms"])
                or payload["invocation_duration_ms"] < replay.active_role_duration_sum_ms
                or payload["cumulative_active_ms"] != replay.cumulative_active_ms + payload["invocation_duration_ms"]
                or type(payload["sealed_roles_after"]) is not int
                or payload["sealed_roles_after"] != replay.next_role_index
                or type(payload["phase_expansions_after"]) is not int
                or payload["phase_expansions_after"] != len(replay.phase_payloads)
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.cumulative_active_ms = payload["cumulative_active_ms"]
            if payload["invocation_duration_ms"] > INVOCATION_DEADLINE_MS and replay.failure_code is None:
                replay.deadline_code = "invocation_deadline_exceeded"
            if replay.cumulative_active_ms > STAGE_ACTIVE_TIME_CAP_MS and replay.failure_code is None:
                replay.deadline_code = "cumulative_source_deadline_exceeded"
            replay.active_invocation = None
            replay.first_dispatch_pending = False
            replay.active_role_duration_sum_ms = 0
        else:
            payload = _payload(
                event,
                {
                    "status", "code", "phase_expansions", "role_plan_sha256",
                    "formula_requests", "lifetime_intents", "http_200_responses",
                    "role_seals", "successful_response_bytes", "lifetime_received_bytes",
                    "decompressed_master_bytes", "cumulative_active_ms", "unclean_active_ms",
                    "checkpoint_sha256",
                },
            )
            expected = _terminal_payload(
                replay, stage, status=payload["status"], code=payload["code"],
                checkpoint_sha256=payload["checkpoint_sha256"],
                unclean_active_ms=payload["unclean_active_ms"],
            )
            if (
                type(payload["status"]) is not str
                or type(payload["code"]) is not str
                or type(payload["phase_expansions"]) is not list
                or not _valid_sha256(payload["role_plan_sha256"])
                or (
                    payload["formula_requests"] is not None
                    and type(payload["formula_requests"]) is not int
                )
                or type(payload["lifetime_intents"]) is not int
                or type(payload["http_200_responses"]) is not int
                or type(payload["role_seals"]) is not int
                or type(payload["successful_response_bytes"]) is not int
                or type(payload["lifetime_received_bytes"]) is not int
                or type(payload["decompressed_master_bytes"]) is not int
                or type(payload["cumulative_active_ms"]) is not int
                or not _nonnegative_int(payload["unclean_active_ms"])
                or (
                    payload["checkpoint_sha256"] is not None
                    and not _valid_sha256(payload["checkpoint_sha256"])
                )
                or payload != expected
            ):
                raise SecGemmaLeanV33JournalError("journal_invalid")
            if payload["status"] == "passed":
                formula = len(replay.roles)
                if (
                    payload["code"] != "passed"
                    or not _valid_sha256(payload["checkpoint_sha256"])
                    or payload["unclean_active_ms"] != 0
                    or replay.active_invocation is not None
                    or replay.failure_code is not None
                    or replay.deadline_code is not None
                    or len(replay.phase_payloads) != 3
                    or replay.next_role_index != formula
                    or replay.lifetime_intents != formula
                    or replay.response_count != formula
                    or replay.seal_count != formula
                    or formula != 1 + replay.phase_payloads[0]["role_count"] + config.quarterly_master_count + replay.phase_payloads[2]["role_count"]
                ):
                    raise SecGemmaLeanV33JournalError("journal_invalid")
            elif payload["status"] == "rejected":
                if payload["code"] not in TERMINAL_REJECTION_CODES or payload["checkpoint_sha256"] is not None:
                    raise SecGemmaLeanV33JournalError("journal_invalid")
                derived = (
                    "open_intent" if replay.open_role is not None
                    else "unclosed_invocation" if replay.active_invocation is not None
                    else replay.failure_code or replay.deadline_code
                )
                if derived is not None and payload["code"] != derived:
                    raise SecGemmaLeanV33JournalError("journal_invalid")
                if replay.active_invocation is None and payload["unclean_active_ms"] != 0:
                    raise SecGemmaLeanV33JournalError("journal_invalid")
            else:
                raise SecGemmaLeanV33JournalError("journal_invalid")
            replay.terminal_status = payload["status"]
            replay.terminal_code = payload["code"]
            replay.terminal_unclean_active_ms = payload["unclean_active_ms"]
            terminal_seen = True
        previous = event["event_sha256"]

    derived: str | None = None
    if replay.terminal_status is None:
        derived = (
            "open_intent" if replay.open_role is not None
            else "unclosed_invocation" if replay.active_invocation is not None
            else replay.failure_code or replay.deadline_code
        )
    expansions = tuple(item["phase"] for item in replay.phase_payloads)
    next_expansion = None
    if replay.next_role_index == len(replay.roles) and len(expansions) < 3:
        next_expansion = PHASE_EXPANSION_ORDER[len(expansions)]
    next_role = None if replay.next_role_index >= len(replay.roles) else replay.roles[replay.next_role_index].role_id
    clean = (
        replay.terminal_status is None
        and replay.active_invocation is None
        and replay.failure_code is None
        and replay.deadline_code is None
    )
    formula = len(replay.roles) if len(expansions) == 3 else None
    return JournalState(
        stage=stage,
        authority_sha256=authority_sha256,
        role_plan_sha256=replay.plan_sha256(stage),
        event_count=len(events),
        head_event_sha256=previous,
        invocation_count=replay.invocation_count,
        active_invocation=replay.active_invocation,
        open_role_id=None if replay.open_role is None else replay.open_role.role_id,
        next_role_id=next_role,
        next_required_expansion=next_expansion,
        planned_roles=tuple(replay.roles),
        sealed_role_ids=tuple(record["role_id"] for record in replay.sealed_records),
        phase_expansions=expansions,
        historical_count=None if len(expansions) < 1 else replay.phase_payloads[0]["role_count"],
        quarterly_master_count=None if len(expansions) < 2 else replay.phase_payloads[1]["role_count"],
        complete_submission_count=None if len(expansions) < 3 else replay.phase_payloads[2]["role_count"],
        formula_requests=formula,
        sealed_prefix_sha256=replay.sealed_prefix_sha256(),
        last_seal_event_sha256=None if not replay.sealed_records else replay.sealed_records[-1]["role_seal_event_sha256"],
        last_parse_receipt_sha256=None if not replay.sealed_records else replay.sealed_records[-1]["parse_receipt_sha256"],
        lifetime_intents=replay.lifetime_intents,
        http_200_responses=replay.response_count,
        role_seals=replay.seal_count,
        successful_response_bytes=replay.successful_response_bytes,
        lifetime_received_bytes=replay.lifetime_received_bytes(),
        decompressed_master_bytes=replay.decompressed_master_bytes,
        cumulative_active_ms=replay.cumulative_active_ms,
        failure_code=replay.failure_code,
        terminal_status=replay.terminal_status,
        terminal_code=replay.terminal_code,
        terminal_unclean_active_ms=replay.terminal_unclean_active_ms,
        derived_terminal_code=derived,
        resume_allowed=clean and (next_role is not None or next_expansion is not None),
        can_finalize_pass=clean and formula is not None and next_role is None,
        accounting_complete=replay.terminal_status == "passed",
        source_authoritative=False,
        passed=replay.terminal_status == "passed",
    )


def validate_detached_journal(
    root: Path, stage: str, contact_fingerprint_sha256: str
) -> JournalState:
    """Deterministically reconstruct phase plans and effect accounting."""

    return _replay_events(_read_events(root), stage, contact_fingerprint_sha256)


class AcquisitionJournal:
    """Durable writer for one dynamic stage journal."""

    __slots__ = (
        "_root",
        "_stage",
        "_contact",
        "_authority_sha256",
        "_owner_invocation",
        "_state",
    )

    def __init__(self, root: Path, *, stage: str, contact: PrivateSecContact) -> None:
        if type(stage) is not str or stage not in STAGE_CONFIGS:
            raise SecGemmaLeanV33JournalError("stage_invalid")
        if type(contact) is not PrivateSecContact:
            raise SecGemmaLeanV33JournalError("private_contact_invalid")
        self._root = _secure_root(root)
        self._stage = stage
        self._contact = contact
        self._authority_sha256 = _canonical_sha256(
            _authority(stage, contact.fingerprint_sha256)
        )
        self._owner_invocation: int | None = None
        self._state = validate_detached_journal(
            self._root, self._stage, self._contact.fingerprint_sha256
        )

    @property
    def state(self) -> JournalState:
        return self._state

    def _refresh(self) -> None:
        self._state = validate_detached_journal(
            self._root, self._stage, self._contact.fingerprint_sha256
        )

    def _append(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if event_type not in EVENT_TYPES or type(payload) is not dict:
            raise SecGemmaLeanV33JournalError("journal_event_invalid")
        self._contact.assert_redacted(payload)
        sequence = self._state.event_count + 1
        body = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "sequence_number": sequence,
            "previous_event_sha256": self._state.head_event_sha256,
            "authority_sha256": self._authority_sha256,
            "event_type": event_type,
            "occurred_at_utc": _utc_now(),
            "payload": dict(payload),
        }
        event = {**body, "event_sha256": _canonical_sha256(body)}
        self._contact.assert_redacted(event)
        encoded = _canonical_json_bytes(event)
        if len(encoded) > _MAX_EVENT_BYTES or self._contact.contains_echo(encoded):
            raise SecGemmaLeanV33JournalError("privacy_echo")
        prior = _read_events(self._root)
        if (
            len(prior) != self._state.event_count
            or (prior and prior[-1]["event_sha256"] != self._state.head_event_sha256)
        ):
            raise SecGemmaLeanV33JournalError("journal_concurrent_change")
        _replay_events(
            [*prior, event], self._stage, self._contact.fingerprint_sha256
        )
        target = self._root / f"{sequence:08d}.json"
        pending = self._root / (
            f".pending-{sequence:08d}-{event['event_sha256']}.json"
        )
        try:
            _write_exclusive_fsynced(pending, encoded)
            _fsync_directory(self._root)
            os.rename(pending, target)
            _regular_file(target)
            if target.read_bytes() != encoded:
                raise OSError("changed")
            _fsync_directory(self._root)
        except Exception:
            _leave_append_failure_marker(
                self._root,
                sequence=sequence,
                event_sha256=event["event_sha256"],
            )
            raise SecGemmaLeanV33JournalError(
                "journal_append_durability_failed"
            ) from None
        self._refresh()
        return event

    def _require_owned_invocation(self) -> int:
        if (
            self._owner_invocation is None
            or self._state.active_invocation != self._owner_invocation
            or self._state.terminal_status is not None
        ):
            raise SecGemmaLeanV33JournalError("invocation_not_owned")
        return self._owner_invocation

    def _require_phase_boundary(self, phase: str) -> None:
        if self._state.terminal_status is not None:
            raise SecGemmaLeanV33JournalError("journal_terminal")
        if self._state.derived_terminal_code is not None:
            if not (
                self._state.active_invocation is not None
                and self._owner_invocation == self._state.active_invocation
                and self._state.open_role_id is None
                and self._state.failure_code is None
            ):
                raise SecGemmaLeanV33JournalError(self._state.derived_terminal_code)
        if self._state.open_role_id is not None or self._state.next_role_id is not None:
            raise SecGemmaLeanV33JournalError("phase_boundary_not_reached")
        if self._state.next_required_expansion != phase:
            raise SecGemmaLeanV33JournalError("phase_expansion_order_invalid")

    def open_invocation(self) -> int:
        if not self._state.resume_allowed:
            raise SecGemmaLeanV33JournalError(
                self._state.derived_terminal_code or "journal_not_resumable"
            )
        invocation = self._state.invocation_count + 1
        self._append(
            "invocation_open",
            {
                "invocation_number": invocation,
                "sealed_roles_before": self._state.role_seals,
                "phase_expansions_before": len(self._state.phase_expansions),
            },
        )
        self._owner_invocation = invocation
        return invocation

    def _append_phase_plan(
        self,
        phase: str,
        roles: tuple[RequestRole, ...],
        basis_receipt_sha256: str,
    ) -> str:
        self._require_phase_boundary(phase)
        if not _valid_sha256(basis_receipt_sha256):
            raise SecGemmaLeanV33JournalError("phase_basis_receipt_invalid")
        payload = {
            "phase_plan_schema_version": PHASE_PLAN_SCHEMA_VERSION,
            "expansion_number": len(self._state.phase_expansions) + 1,
            "phase": phase,
            "basis_kind": PHASE_BASIS_KINDS[phase],
            "basis_receipt_sha256": basis_receipt_sha256,
            "sealed_prefix_sha256": self._state.sealed_prefix_sha256,
            "role_count": len(roles),
            "roles": [role.as_dict() for role in roles],
        }
        return self._append("phase_plan", payload)["event_sha256"]

    def append_historical_phase(self, historical_filenames: Sequence[str]) -> str:
        self._require_phase_boundary("historical_submissions")
        if self._state.last_parse_receipt_sha256 is None:
            raise SecGemmaLeanV33JournalError("main_parse_receipt_missing")
        roles = _historical_roles(
            historical_filenames, len(self._state.planned_roles) + 1
        )
        return self._append_phase_plan(
            "historical_submissions", roles, self._state.last_parse_receipt_sha256
        )

    def append_master_phase(self, *, submissions_snapshot_receipt_sha256: str) -> str:
        self._require_phase_boundary("quarterly_master")
        roles = _master_roles(self._stage, len(self._state.planned_roles) + 1)
        return self._append_phase_plan(
            "quarterly_master", roles, submissions_snapshot_receipt_sha256
        )

    def append_complete_phase(
        self,
        complete_submissions: Sequence[CompleteSubmissionTarget],
        *,
        reconciliation_and_prior_chain_receipt_sha256: str,
    ) -> str:
        self._require_phase_boundary("complete_submission")
        roles = _complete_roles(
            self._stage, complete_submissions, len(self._state.planned_roles) + 1
        )
        return self._append_phase_plan(
            "complete_submission",
            roles,
            reconciliation_and_prior_chain_receipt_sha256,
        )

    def record_role_intent(self, role_id: str, *, dispatch_wait_ms: int) -> str:
        invocation = self._require_owned_invocation()
        if self._state.failure_code is not None or self._state.open_role_id is not None:
            raise SecGemmaLeanV33JournalError("role_intent_forbidden")
        if self._state.next_role_id != role_id:
            raise SecGemmaLeanV33JournalError("role_order_invalid")
        role = self._state.planned_roles[self._state.role_seals]
        invocation_intents = [
            event
            for event in _read_events(self._root)
            if event["event_type"] == "role_intent"
            and event["payload"]["invocation_number"] == invocation
        ]
        minimum = FIRST_DISPATCH_WAIT_MS if not invocation_intents else GLOBAL_DISPATCH_GAP_MS
        if type(dispatch_wait_ms) is not int or dispatch_wait_ms < minimum:
            raise SecGemmaLeanV33JournalError("dispatch_gap_too_short")
        return self._append(
            "role_intent",
            {
                "invocation_number": invocation,
                "role_ordinal": role.ordinal,
                "role_id": role.role_id,
                "phase": role.phase,
                "url_sha256": _sha256_bytes(role.url.encode("ascii")),
                "body_limit_bytes": role.body_limit_bytes,
                "reservation_bytes": role.reservation_bytes,
                "dispatch_wait_ms": dispatch_wait_ms,
            },
        )["event_sha256"]

    def record_response_complete(
        self,
        role_id: str,
        *,
        intent_event_sha256: str,
        body_bytes: int,
        body_sha256: str,
        transport_receipt_sha256: str,
        request_duration_ms: int,
    ) -> str:
        invocation = self._require_owned_invocation()
        if self._state.open_role_id != role_id:
            raise SecGemmaLeanV33JournalError("response_role_invalid")
        role = self._state.planned_roles[self._state.role_seals]
        if type(body_bytes) is int and body_bytes == role.reservation_bytes:
            raise SecGemmaLeanV33JournalError("body_limit_exceeded")
        return self._append(
            "response_complete",
            {
                "invocation_number": invocation,
                "role_id": role_id,
                "intent_event_sha256": intent_event_sha256,
                "status_code": 200,
                "body_bytes": body_bytes,
                "body_sha256": body_sha256,
                "transport_receipt_sha256": transport_receipt_sha256,
                "request_duration_ms": request_duration_ms,
            },
        )["event_sha256"]

    def record_role_seal(
        self,
        role_id: str,
        *,
        intent_event_sha256: str,
        response_event_sha256: str,
        blob_sha256: str,
        parse_receipt_sha256: str,
        body_bytes: int,
        decompressed_bytes: int | None,
        role_duration_ms: int,
    ) -> str:
        invocation = self._require_owned_invocation()
        if self._state.open_role_id != role_id:
            raise SecGemmaLeanV33JournalError("seal_role_invalid")
        return self._append(
            "role_seal",
            {
                "invocation_number": invocation,
                "role_id": role_id,
                "intent_event_sha256": intent_event_sha256,
                "response_event_sha256": response_event_sha256,
                "blob_sha256": blob_sha256,
                "parse_receipt_sha256": parse_receipt_sha256,
                "body_bytes": body_bytes,
                "decompressed_bytes": decompressed_bytes,
                "role_duration_ms": role_duration_ms,
            },
        )["event_sha256"]

    def record_role_failure(
        self,
        role_id: str,
        *,
        intent_event_sha256: str,
        error_code: str,
        observed_body_bytes: int,
        request_duration_ms: int,
        role_duration_ms: int,
        response_event_sha256: str | None = None,
    ) -> str:
        invocation = self._require_owned_invocation()
        if type(error_code) is not str or error_code not in ROLE_FAILURE_CODES:
            raise SecGemmaLeanV33JournalError("unsafe_error_code")
        if self._state.open_role_id != role_id:
            raise SecGemmaLeanV33JournalError("failure_role_invalid")
        return self._append(
            "role_failure",
            {
                "invocation_number": invocation,
                "role_id": role_id,
                "intent_event_sha256": intent_event_sha256,
                "response_event_sha256": response_event_sha256,
                "error_code": error_code,
                "observed_body_bytes": observed_body_bytes,
                "request_duration_ms": request_duration_ms,
                "role_duration_ms": role_duration_ms,
            },
        )["event_sha256"]

    def close_invocation(self, *, invocation_duration_ms: int) -> str:
        invocation = self._require_owned_invocation()
        if self._state.open_role_id is not None:
            raise SecGemmaLeanV33JournalError("open_intent")
        if not _nonnegative_int(invocation_duration_ms):
            raise SecGemmaLeanV33JournalError("invocation_duration_invalid")
        event = self._append(
            "invocation_clean_close",
            {
                "invocation_number": invocation,
                "invocation_duration_ms": invocation_duration_ms,
                "cumulative_active_ms": self._state.cumulative_active_ms + invocation_duration_ms,
                "sealed_roles_after": self._state.role_seals,
                "phase_expansions_after": len(self._state.phase_expansions),
            },
        )
        self._owner_invocation = None
        return event["event_sha256"]

    def seal_terminal_pass(self, *, checkpoint_sha256: str) -> str:
        if not self._state.can_finalize_pass:
            raise SecGemmaLeanV33JournalError(
                self._state.derived_terminal_code or "pass_formula_incomplete"
            )
        payload = {
            "status": "passed",
            "code": "passed",
            "phase_expansions": list(self._state.phase_expansions),
            "role_plan_sha256": self._state.role_plan_sha256,
            "formula_requests": self._state.formula_requests,
            "lifetime_intents": self._state.lifetime_intents,
            "http_200_responses": self._state.http_200_responses,
            "role_seals": self._state.role_seals,
            "successful_response_bytes": self._state.successful_response_bytes,
            "lifetime_received_bytes": self._state.lifetime_received_bytes,
            "decompressed_master_bytes": self._state.decompressed_master_bytes,
            "cumulative_active_ms": self._state.cumulative_active_ms,
            "unclean_active_ms": 0,
            "checkpoint_sha256": checkpoint_sha256,
        }
        return self._append("terminal", payload)["event_sha256"]

    def seal_terminal_rejection(
        self, *, code: str | None = None, unclean_active_ms: int | None = None
    ) -> str:
        if self._state.terminal_status is not None:
            raise SecGemmaLeanV33JournalError("journal_terminal")
        selected = self._state.derived_terminal_code if code is None else code
        if type(selected) is not str or selected not in TERMINAL_REJECTION_CODES:
            raise SecGemmaLeanV33JournalError("unsafe_error_code")
        if self._state.derived_terminal_code is not None and selected != self._state.derived_terminal_code:
            raise SecGemmaLeanV33JournalError("terminal_code_mismatch")
        if self._state.active_invocation is None:
            if unclean_active_ms is not None and (
                type(unclean_active_ms) is not int or unclean_active_ms != 0
            ):
                raise SecGemmaLeanV33JournalError("unclean_duration_invalid")
            unclean = 0
        else:
            if not _nonnegative_int(unclean_active_ms):
                raise SecGemmaLeanV33JournalError("unclean_duration_required")
            unclean = unclean_active_ms
        payload = {
            "status": "rejected",
            "code": selected,
            "phase_expansions": list(self._state.phase_expansions),
            "role_plan_sha256": self._state.role_plan_sha256,
            "formula_requests": self._state.formula_requests,
            "lifetime_intents": self._state.lifetime_intents,
            "http_200_responses": self._state.http_200_responses,
            "role_seals": self._state.role_seals,
            "successful_response_bytes": self._state.successful_response_bytes,
            "lifetime_received_bytes": self._state.lifetime_received_bytes,
            "decompressed_master_bytes": self._state.decompressed_master_bytes,
            "cumulative_active_ms": self._state.cumulative_active_ms,
            "unclean_active_ms": unclean,
            "checkpoint_sha256": None,
        }
        event = self._append("terminal", payload)
        self._owner_invocation = None
        return event["event_sha256"]


__all__ = [
    "AcquisitionJournal",
    "COMPLETE_SUBMISSION_BODY_LIMIT",
    "CompleteSubmissionTarget",
    "EVENT_TYPES",
    "FIRST_DISPATCH_WAIT_MS",
    "GLOBAL_DISPATCH_GAP_MS",
    "HISTORICAL_REFERENCE_CAP",
    "HISTORICAL_SUBMISSIONS_BODY_LIMIT",
    "INVOCATION_DEADLINE_MS",
    "JOURNAL_SCHEMA_VERSION",
    "JournalState",
    "LIFETIME_INTENT_CAP",
    "LIFETIME_RECEIVED_BYTE_CAP",
    "MAIN_SUBMISSIONS_BODY_LIMIT",
    "MASTER_COMPRESSED_BODY_LIMIT",
    "MASTER_DECOMPRESSED_BODY_LIMIT",
    "MASTER_DECOMPRESSED_BYTE_CAP",
    "PHASE_BODY_LIMITS",
    "PHASE_EXPANSION_ORDER",
    "PrivateSecContact",
    "REQUEST_DEADLINE_MS",
    "ROLE_DEADLINE_MS",
    "ROLE_FAILURE_CODES",
    "SOURCE_DIAGNOSTIC_CODES",
    "RequestRole",
    "STAGE_ACTIVE_TIME_CAP_MS",
    "STAGE_CONFIGS",
    "SUCCESSFUL_RESPONSE_BYTE_CAP",
    "SecGemmaLeanV33JournalError",
    "StageConfig",
    "StageRolePlan",
    "TERMINAL_REJECTION_CODES",
    "build_stage_role_plan",
    "validate_detached_journal",
    "validate_stage_role_plan",
]
