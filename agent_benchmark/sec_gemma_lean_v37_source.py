"""Pure offline source sealing for AAPL SEC/Gemma lean evidence v3.7.

This module has no filesystem, network, clock, market-data, or model surface.
It accepts already acquired response bytes and turns them into deterministic,
JSON-safe evidence.  The full complete-submission response, the selected
embedded ``TEXT`` bytes, and the inherited normalized text are deliberately
kept as three different byte identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence
import zlib

from .sec_filing_content import (
    normalize_filing_text,
    parse_complete_submission,
    select_sequence_one_primary_document,
)
from .sec_point_in_time import (
    AAPL_CIK,
    FilingRecord,
    SecPointInTimeError,
    content_sha256,
    parse_submissions_acceptance_datetime as _parse_submissions_acceptance_wall_clock,
)
from .sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
    validate_aapl_session_calendar,
    validate_market_history_session_calendar,
)
from zoneinfo import ZoneInfo


def parse_submissions_acceptance_datetime(value: str) -> datetime:
    """Parse one exact SEC acceptance spelling into New York time."""

    if not isinstance(value, str):
        raise SecPointInTimeError("SEC acceptance datetime must remain text")
    if re.fullmatch(r"[0-9]{14}", value):
        return _parse_submissions_acceptance_wall_clock(value)
    matched = re.fullmatch(
        r"(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2})T"
        r"(?P<time>[0-9]{2}:[0-9]{2}:[0-9]{2})(?:\.0{1,9})?Z",
        value,
    )
    if matched is None:
        return _parse_submissions_acceptance_wall_clock(value)
    try:
        parsed = datetime.strptime(
            f"{matched.group('date')}T{matched.group('time')}",
            "%Y-%m-%dT%H:%M:%S",
        )
    except ValueError as exc:
        raise SecPointInTimeError(
            "acceptanceDateTime is not a valid exact SEC Z timestamp"
        ) from exc
    return parsed.replace(tzinfo=timezone.utc).astimezone(
        ZoneInfo("America/New_York")
    )


SOURCE_SCHEMA_VERSION = "aapl-sec-gemma-lean-v37-source-seal-v1"
DETACHED_REPLAY_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-detached-source-replay-v1"
)
LEGACY_SCIENCE_PROJECTION_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-legacy-science-projection-v1"
)
PRE_MASTER_UPPER_BOUND_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-pre-master-upper-bound-v1"
)
COMPLETED_YEAR_COVERAGE_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-completed-year-source-coverage-v1"
)
COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-compact-role-evidence-v1"
)
COMPACT_SUBMISSIONS_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-compact-submissions-v1"
)
COMPACT_RECONCILIATION_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-compact-reconciliation-v1"
)
COMPACT_COMPLETE_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-compact-complete-v1"
)
COMPACT_ROLE_MANIFEST_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-compact-role-manifest-v1"
)
COMPACT_CHECKPOINT_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-compact-checkpoint-v1"
)
COMPACT_REPLAY_SCHEMA_VERSION = (
    "aapl-sec-gemma-lean-v37-compact-detached-replay-v1"
)

MAIN_SUBMISSIONS_NAME = f"CIK{AAPL_CIK}.json"
MAIN_SUBMISSIONS_URL = (
    f"https://data.sec.gov/submissions/{MAIN_SUBMISSIONS_NAME}"
)
MASTER_ARCHIVE_PREFIX = "https://www.sec.gov/Archives/"
MASTER_START = (1994, 1)
GLOBAL_AVAILABILITY_START = "2000-01-01"
GLOBAL_AVAILABILITY_END = "2026-07-09"
MAX_HISTORICAL_REFERENCES = 16
MAX_MASTER_COMPRESSED_BYTES = 16 * 1024 * 1024
MAX_MASTER_DECOMPRESSED_BYTES = 128 * 1024 * 1024
MASTER_HEADER = "CIK|Company Name|Form Type|Date Filed|Filename"
COMPLETED_YEAR_START = 2000
COMPLETED_YEAR_END = 2025

_ACCESSION_RE = re.compile(r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z")
_HISTORICAL_NAME_RE = re.compile(
    rf"CIK{AAPL_CIK}-submissions-[0-9]{{3}}\.json\Z"
)
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BARE_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_BASENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_SAFE_BLOB_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,255}\Z")
_STAGE_ORDER = ("development", "intermediate", "final")
_EXPECTED_SESSION_SET = frozenset(EXPECTED_MARKET_HISTORY_SESSIONS)
_STAGE_WINDOWS = {
    "development": ("2000-01-01", "2018-12-31"),
    "intermediate": ("2019-01-01", "2023-12-31"),
    "final": ("2024-01-01", "2026-07-09"),
}


class SecGemmaLeanV37SourceError(SecPointInTimeError):
    """Caller-supplied bytes violate the frozen v3.7 source contract."""


@dataclass(frozen=True)
class StageSourceConfig:
    stage: str
    availability_start: str
    availability_end: str
    master_start_year: int
    master_start_quarter: int
    master_end_year: int
    master_end_quarter: int
    fixed_cutoff: str
    quarter_count: int
    i_cap: int
    p_cap: int
    d_min: int
    d_max: int
    maximum_successful_requests: int
    final_completed_year_coverage_gate: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STAGE_SOURCE_CONFIGS: Mapping[str, StageSourceConfig] = MappingProxyType(
    {
        "development": StageSourceConfig(
            stage="development",
            availability_start="2000-01-01",
            availability_end="2018-12-31",
            master_start_year=1994,
            master_start_quarter=1,
            master_end_year=2018,
            master_end_quarter=4,
            fixed_cutoff="2018-12-31",
            quarter_count=100,
            i_cap=128,
            p_cap=128,
            d_min=72,
            d_max=80,
            maximum_successful_requests=245,
        ),
        "intermediate": StageSourceConfig(
            stage="intermediate",
            availability_start="2019-01-01",
            availability_end="2023-12-31",
            master_start_year=1994,
            master_start_quarter=1,
            master_end_year=2023,
            master_end_quarter=4,
            fixed_cutoff="2023-12-31",
            quarter_count=120,
            i_cap=160,
            p_cap=24,
            d_min=19,
            d_max=20,
            maximum_successful_requests=161,
        ),
        "final": StageSourceConfig(
            stage="final",
            availability_start="2024-01-01",
            availability_end="2026-07-09",
            master_start_year=1994,
            master_start_quarter=1,
            master_end_year=2026,
            master_end_quarter=3,
            fixed_cutoff="2026-07-09",
            quarter_count=131,
            i_cap=176,
            p_cap=16,
            d_min=0,
            d_max=12,
            maximum_successful_requests=164,
            final_completed_year_coverage_gate=(
                "availability_years_2000_2025_min_total3_10k1_10q2"
            ),
        ),
    }
)


@dataclass(frozen=True, order=True)
class QuarterRole:
    year: int
    quarter: int
    url: str

    @property
    def key(self) -> tuple[int, int]:
        return self.year, self.quarter

    @property
    def role(self) -> str:
        return f"master-{self.year}-q{self.quarter}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "quarter": self.quarter,
            "role": self.role,
            "url": self.url,
        }


@dataclass(frozen=True)
class HistoricalReference:
    name: str
    url: str
    filing_count: int
    filing_from: str
    filing_to: str
    source_position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubmissionRow:
    accession_number: str
    subject_cik: str
    acceptance_datetime: str
    form: str
    primary_document: str
    items: str
    filing_date: str
    report_date: str
    is_xbrl: bool
    date_of_filing_date_change: str | None
    date_of_filing_date_change_present: bool
    typed_values: Mapping[str, Any] = field(repr=False)
    semantic_identity_sha256: str
    source_name: str
    source_content_sha256: str
    source_position: int
    source_row_order_sha256: str

    @property
    def is_target(self) -> bool:
        return self.form in {"10-K", "10-Q"}

    def semantic_identity(self) -> dict[str, Any]:
        return _canonical_snapshot({
            "accession_number": self.accession_number,
            "subject_cik": self.subject_cik,
            "typed_values": dict(self.typed_values),
            "date_of_filing_date_change_present": (
                self.date_of_filing_date_change_present
            ),
        })

    def layout_provenance(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_content_sha256": self.source_content_sha256,
            "source_position": self.source_position,
            "source_row_order_sha256": self.source_row_order_sha256,
        }


@dataclass(frozen=True)
class SubmissionsSource:
    name: str
    url: str
    payload: bytes = field(repr=False)
    payload_sha256: str
    payload_length: int
    raw_row_count: int
    unique_row_count: int
    row_order_sha256: str
    range_evidence: Mapping[str, Any] | None

    def summary(self) -> dict[str, Any]:
        return _canonical_snapshot({
            "name": self.name,
            "url": self.url,
            "payload_sha256": self.payload_sha256,
            "payload_length": self.payload_length,
            "raw_row_count": self.raw_row_count,
            "unique_row_count": self.unique_row_count,
            "row_order_sha256": self.row_order_sha256,
            "range_evidence": (
                dict(self.range_evidence) if self.range_evidence is not None else None
            ),
        })


@dataclass(frozen=True)
class SubmissionsSnapshot:
    references: tuple[HistoricalReference, ...]
    sources: tuple[SubmissionsSource, ...]
    rows: tuple[SubmissionRow, ...]
    snapshot_sha256: str

    @property
    def historical_reference_count(self) -> int:
        return len(self.references)

    @property
    def main_payload(self) -> bytes:
        return self.sources[0].payload

    @property
    def historical_payloads(self) -> Mapping[str, bytes]:
        return MappingProxyType(
            {source.name: source.payload for source in self.sources[1:]}
        )

    def summary(self) -> dict[str, Any]:
        return _canonical_snapshot({
            "main_name": MAIN_SUBMISSIONS_NAME,
            "historical_references": [item.to_dict() for item in self.references],
            "historical_reference_array_order_sha256": _canonical_sha256(
                [item.to_dict() for item in self.references]
            ),
            "canonical_historical_fetch_order": sorted(
                item.name for item in self.references
            ),
            "sources": [source.summary() for source in self.sources],
            "row_count": len(self.rows),
            "target_row_count": sum(row.is_target for row in self.rows),
            "snapshot_sha256": self.snapshot_sha256,
        })


@dataclass(frozen=True)
class MasterTargetRow:
    accession_number: str
    cik: str
    company_name: str
    form: str
    filing_date: str
    filename: str
    quarter: tuple[int, int]

    @property
    def complete_submission_url(self) -> str:
        return MASTER_ARCHIVE_PREFIX + self.filename

    def to_dict(self) -> dict[str, Any]:
        return {
            "accession_number": self.accession_number,
            "cik": self.cik,
            "company_name": self.company_name,
            "form": self.form,
            "filing_date": self.filing_date,
            "filename": self.filename,
            "quarter": [self.quarter[0], self.quarter[1]],
            "complete_submission_url": self.complete_submission_url,
        }


@dataclass(frozen=True)
class MasterQuarterEvidence:
    role: QuarterRole
    compressed_payload: bytes = field(repr=False)
    compressed_sha256: str
    compressed_length: int
    decompressed_sha256: str
    decompressed_length: int
    raw_row_count: int
    raw_row_order_sha256: str
    target_rows: tuple[MasterTargetRow, ...]

    def summary(self) -> dict[str, Any]:
        return {
            **self.role.to_dict(),
            "compressed_sha256": self.compressed_sha256,
            "compressed_length": self.compressed_length,
            "decompressed_sha256": self.decompressed_sha256,
            "decompressed_length": self.decompressed_length,
            "raw_row_count": self.raw_row_count,
            "raw_row_order_sha256": self.raw_row_order_sha256,
            "target_count": len(self.target_rows),
            "target_rows_sha256": _canonical_sha256(
                [row.to_dict() for row in self.target_rows]
            ),
        }


@dataclass(frozen=True)
class ReconciledTarget:
    submissions: SubmissionRow
    master: MasterTargetRow


@dataclass(frozen=True)
class MasterReconciliation:
    config: StageSourceConfig
    quarter_evidence: tuple[MasterQuarterEvidence, ...]
    targets: tuple[ReconciledTarget, ...]
    reconciliation_sha256: str

    @property
    def accessions(self) -> tuple[str, ...]:
        return tuple(item.submissions.accession_number for item in self.targets)


@dataclass(frozen=True)
class CompleteSourceEvidence:
    accession_number: str
    acquisition_stage: str
    complete_response: bytes = field(repr=False)
    extracted_primary: bytes = field(repr=False)
    normalized_text: bytes = field(repr=False)
    seal_row: Mapping[str, Any]

    @property
    def complete_response_sha256(self) -> str:
        return str(self.seal_row["complete_response"]["sha256"])


@dataclass(frozen=True)
class StageSourceBundle:
    config: StageSourceConfig
    submissions_snapshot: SubmissionsSnapshot
    master_reconciliation: MasterReconciliation
    complete_sources: tuple[CompleteSourceEvidence, ...]
    session_dates: tuple[str, ...]
    current_complete_accessions: tuple[str, ...]
    prior_seal: Mapping[str, Any] | None
    prior_bundle: StageSourceBundle | None = field(repr=False)
    seal: Mapping[str, Any]
    seal_json: bytes = field(repr=False)

    @property
    def stage_source_seal_sha256(self) -> str:
        return str(self.seal["stage_source_seal_sha256"])

    @property
    def sources_by_accession(self) -> Mapping[str, CompleteSourceEvidence]:
        return MappingProxyType(
            {source.accession_number: source for source in self.complete_sources}
        )


@dataclass(frozen=True)
class LegacyScienceDocument:
    accession_number: str
    official_complete_submission_url: str
    raw_primary_document: bytes = field(repr=False)
    normalized_text: bytes = field(repr=False)
    primary_document_sha256: str
    normalized_text_sha256: str
    complete_response_sha256: str
    selected_document_identity: str


@dataclass(frozen=True)
class LegacyScienceProjection:
    stage: str
    documents: tuple[LegacyScienceDocument, ...]
    manifest: Mapping[str, Any]
    manifest_json: bytes = field(repr=False)


@dataclass(frozen=True)
class RoleParseOutput:
    """One compact role result; ``decompressed_bytes`` is a count, never bytes."""

    evidence: Mapping[str, Any]
    historical_filenames: tuple[str, ...] = ()
    decompressed_bytes: int | None = None


@dataclass(frozen=True)
class PhaseOutput:
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class ReconciliationOutput:
    evidence: Mapping[str, Any]
    complete_targets: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class StageOutput:
    """Compact replayable stage authority containing metadata but no raw bytes."""

    seal: Mapping[str, Any]
    submissions_evidence: Mapping[str, Any] = field(repr=False)
    reconciliation_evidence: Mapping[str, Any] = field(repr=False)
    complete_evidence: tuple[Mapping[str, Any], ...] = field(repr=False)
    prior: StageOutput | None = field(default=None, repr=False)

    @property
    def stage_source_seal_sha256(self) -> str:
        return str(self.seal["stage_source_seal_sha256"])


@dataclass(frozen=True)
class CompactReplayInput:
    """Disk-backed prior-stage replay inputs retained without any payload bytes."""

    stage: str
    checkpoint: Mapping[str, Any]
    role_manifests: tuple[Mapping[str, Any], ...]
    prior: CompactReplayInput | None = None


@dataclass(frozen=True)
class CompactReplayResult:
    """Authenticated compact stage authority plus its detached replay receipt."""

    stage_output: StageOutput
    receipt: Mapping[str, Any]


class SourceBlobLoader(Protocol):
    """Pure callback boundary for loading one immutable role blob at a time."""

    def __call__(self, blob_name: str) -> bytes: ...


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise SecGemmaLeanV37SourceError(
            "v3.7 source evidence is not finite canonical JSON"
        ) from None


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_snapshot(value: Any) -> Any:
    """Detach nested metadata through the exact canonical JSON representation."""

    return json.loads(_canonical_json_bytes(value).decode("ascii"))


def _compact_authoritative_calendar_receipt() -> dict[str, Any]:
    experiment_evidence = validate_aapl_session_calendar(EXPECTED_SESSIONS)
    history_evidence = validate_market_history_session_calendar(
        EXPECTED_MARKET_HISTORY_SESSIONS
    )
    expected_keys = {
        "calendar_id",
        "start",
        "end",
        "session_count",
        "sessions_sha256",
        "sessions_canonical_json_sha256",
        "sessions",
    }
    if (
        set(experiment_evidence) != expected_keys
        or set(history_evidence) != expected_keys
    ):
        raise SecGemmaLeanV37SourceError(
            "Authoritative AAPL calendar evidence schema changed"
        )
    experiment_calendar = {
        "calendar_id": experiment_evidence["calendar_id"],
        "start": experiment_evidence["start"],
        "end": experiment_evidence["end"],
        "session_count": experiment_evidence["session_count"],
        "sessions_sha256": experiment_evidence["sessions_sha256"],
        "sessions_canonical_json_sha256": experiment_evidence[
            "sessions_canonical_json_sha256"
        ],
    }
    availability_search_calendar = {
        "calendar_id": history_evidence["calendar_id"],
        "start": history_evidence["start"],
        "end": history_evidence["end"],
        "session_count": history_evidence["session_count"],
        "sessions_sha256": history_evidence["sessions_sha256"],
        "sessions_canonical_json_sha256": history_evidence[
            "sessions_canonical_json_sha256"
        ],
    }
    suffix_start = len(EXPECTED_MARKET_HISTORY_SESSIONS) - len(EXPECTED_SESSIONS)
    if (
        suffix_start != 504
        or suffix_start <= 0
        or EXPECTED_MARKET_HISTORY_SESSIONS[suffix_start:] != EXPECTED_SESSIONS
    ):
        raise SecGemmaLeanV37SourceError(
            "Authoritative experiment calendar is not the exact history suffix"
        )
    exact_boundary_start = (
        date.fromisoformat(history_evidence["start"]) - timedelta(days=1)
    ).isoformat()
    body = {
        "schema_version": (
            "aapl-sec-gemma-lean-v37-authoritative-calendar-receipt-v1"
        ),
        "experiment_calendar": experiment_calendar,
        "availability_search_calendar": availability_search_calendar,
        "suffix_relation": {
            "relation": (
                "experiment_calendar_is_exact_suffix_of_"
                "availability_search_calendar"
            ),
            "availability_prefix_session_count": suffix_start,
            "experiment_suffix_start_index": suffix_start,
            "experiment_suffix_session_count": len(EXPECTED_SESSIONS),
            "experiment_suffix_sessions_sha256": experiment_evidence[
                "sessions_sha256"
            ],
            "experiment_suffix_sessions_canonical_json_sha256": (
                experiment_evidence["sessions_canonical_json_sha256"]
            ),
        },
        "availability_semantics": {
            "boundary_rule": (
                "max_filing_date_exact_acceptance_date_and_"
                "authenticated_filing_date_change"
            ),
            "strict_comparison": "session_strictly_greater_than_boundary",
            "exact_boundary_start": exact_boundary_start,
            "unsupported_prehistory": (
                "availability_session_null_and_stage_assignment_null"
            ),
            "availability_search_calendar_role": "availability_search_only",
            "experiment_calendar_role": "caller_replay_stage_and_science",
            "experiment_stage_start": GLOBAL_AVAILABILITY_START,
        },
    }
    return _canonical_snapshot(
        {
            **body,
            "authoritative_calendar_receipt_sha256": _canonical_sha256(body),
        }
    )


_AUTHORITATIVE_CALENDAR_RECEIPT = _compact_authoritative_calendar_receipt()


def _authoritative_calendar_receipt(values: Sequence[str]) -> dict[str, Any]:
    """Validate the one authoritative calendar and retain its compact identity."""

    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or tuple(values) != EXPECTED_SESSIONS
    ):
        raise SecGemmaLeanV37SourceError(
            "Pinned sessions are not the exact authoritative AAPL calendar"
        )
    return _canonical_snapshot(_AUTHORITATIVE_CALENDAR_RECEIPT)


def _json_object(payload: bytes, *, location: str) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise SecGemmaLeanV37SourceError(f"{location} must be exact response bytes")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecGemmaLeanV37SourceError(
                    "SEC JSON contains a duplicate object key"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except SecGemmaLeanV37SourceError:
        raise SecGemmaLeanV37SourceError(
            f"{location} contains a duplicate JSON object key"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanV37SourceError(f"{location} is not strict UTF-8 JSON") from None
    if type(value) is not dict:
        raise SecGemmaLeanV37SourceError(f"{location} must be a JSON object")
    # This also rejects non-finite values accepted by Python's decoder.
    _canonical_json_bytes(value)
    return value


def _canonical_date(value: Any, *, location: str, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise SecGemmaLeanV37SourceError(f"{location} must remain text")
    if allow_empty and value == "":
        return value
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise SecGemmaLeanV37SourceError(f"{location} is not a canonical ISO date") from None
    if parsed.isoformat() != value:
        raise SecGemmaLeanV37SourceError(f"{location} is not a canonical ISO date")
    return value


def _canonical_cik(value: Any, *, location: str) -> str:
    raw = str(value)
    if not raw.isdigit() or len(raw) > 10:
        raise SecGemmaLeanV37SourceError(f"{location} is not a canonical SEC CIK")
    return str(int(raw)).zfill(10)


def _flag(value: Any, *, location: str) -> bool:
    if type(value) is bool:
        return value
    if type(value) is int and value in {0, 1}:
        return bool(value)
    if type(value) is str and value in {"", "0", "1"}:
        return value == "1"
    raise SecGemmaLeanV37SourceError(f"{location} must be an exact zero/one flag")


def _safe_primary_name(value: str, *, location: str, allow_empty: bool) -> str:
    if allow_empty and value == "":
        return value
    if (
        _SAFE_BASENAME_RE.fullmatch(value) is None
        or PurePosixPath(value).name != value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise SecGemmaLeanV37SourceError(f"{location} is not a safe basename")
    return value


def _stage_config(value: str | StageSourceConfig) -> StageSourceConfig:
    if isinstance(value, StageSourceConfig):
        expected = STAGE_SOURCE_CONFIGS.get(value.stage)
        if expected != value:
            raise SecGemmaLeanV37SourceError("Stage configuration is not frozen v3.7")
        return value
    if type(value) is not str or value not in STAGE_SOURCE_CONFIGS:
        raise SecGemmaLeanV37SourceError("Unknown v3.7 source stage")
    return STAGE_SOURCE_CONFIGS[value]


def quarter_roles(
    stage: str | StageSourceConfig,
) -> tuple[QuarterRole, ...]:
    """Return the frozen canonical quarterly-master role sequence."""

    config = _stage_config(stage)
    result: list[QuarterRole] = []
    year, quarter = config.master_start_year, config.master_start_quarter
    while (year, quarter) <= (config.master_end_year, config.master_end_quarter):
        result.append(
            QuarterRole(
                year=year,
                quarter=quarter,
                url=(
                    "https://www.sec.gov/Archives/edgar/full-index/"
                    f"{year}/QTR{quarter}/master.gz"
                ),
            )
        )
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
    if len(result) != config.quarter_count:
        raise SecGemmaLeanV37SourceError("Frozen quarter count is internally inconsistent")
    return tuple(result)


def _historical_references(main: Mapping[str, Any]) -> tuple[HistoricalReference, ...]:
    if _canonical_cik(main.get("cik", ""), location="main Submissions CIK") != AAPL_CIK:
        raise SecGemmaLeanV37SourceError("Main Submissions response is not Apple")
    filings = main.get("filings")
    if type(filings) is not dict or type(filings.get("recent")) is not dict:
        raise SecGemmaLeanV37SourceError("Main Submissions lacks filings.recent")
    raw_references = filings.get("files")
    if type(raw_references) is not list:
        raise SecGemmaLeanV37SourceError("Main Submissions filings.files is not an array")
    if len(raw_references) > MAX_HISTORICAL_REFERENCES:
        raise SecGemmaLeanV37SourceError("Historical Submissions reference cap exceeded")
    result: list[HistoricalReference] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_references):
        if type(raw) is not dict or set(raw) != {
            "name",
            "filingCount",
            "filingFrom",
            "filingTo",
        }:
            raise SecGemmaLeanV37SourceError(
                f"Historical reference {position} does not have the exact schema"
            )
        name = raw["name"]
        if (
            type(name) is not str
            or name != name.strip()
            or PurePosixPath(name).name != name
            or "\\" in name
            or _HISTORICAL_NAME_RE.fullmatch(name) is None
        ):
            raise SecGemmaLeanV37SourceError("Historical reference name is unsafe")
        if name in seen:
            raise SecGemmaLeanV37SourceError("Historical reference name is duplicated")
        seen.add(name)
        filing_count = raw["filingCount"]
        if type(filing_count) is not int or filing_count < 1:
            raise SecGemmaLeanV37SourceError(
                "Historical reference filingCount must be a positive integer"
            )
        filing_from = _canonical_date(
            raw["filingFrom"], location=f"{name}.filingFrom"
        )
        filing_to = _canonical_date(raw["filingTo"], location=f"{name}.filingTo")
        if filing_from > filing_to:
            raise SecGemmaLeanV37SourceError("Historical reference range is reversed")
        result.append(
            HistoricalReference(
                name=name,
                url=f"https://data.sec.gov/submissions/{name}",
                filing_count=filing_count,
                filing_from=filing_from,
                filing_to=filing_to,
                source_position=position,
            )
        )
    # Preserve the exact main-response array order.  Callers independently use
    # canonical filename order for fetching/parsing the referenced bodies.
    return tuple(result)


def _parse_columns(
    columns: Mapping[str, Any],
    *,
    source_name: str,
    source_sha256: str,
    reference: HistoricalReference | None,
) -> tuple[tuple[SubmissionRow, ...], str, Mapping[str, Any] | None]:
    if type(columns) is not dict:
        raise SecGemmaLeanV37SourceError(f"{source_name} is not a column object")
    mandatory_text = (
        "accessionNumber",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "items",
        "filingDate",
        "reportDate",
    )
    missing = [name for name in mandatory_text if name not in columns]
    if missing:
        raise SecGemmaLeanV37SourceError(
            f"{source_name} is missing mandatory columns: {missing}"
        )
    accessions = columns["accessionNumber"]
    if type(accessions) is not list:
        raise SecGemmaLeanV37SourceError(
            f"{source_name}.accessionNumber is not an array"
        )
    size = len(accessions)
    arrays: dict[str, list[Any]] = {}
    for key, values in columns.items():
        if type(key) is not str or type(values) is not list or len(values) != size:
            raise SecGemmaLeanV37SourceError(
                f"{source_name} columns are not complete row-aligned arrays"
            )
        arrays[key] = values
    change_present = "dateOfFilingDateChange" in arrays
    change_values = arrays.get("dateOfFilingDateChange", [""] * size)
    xbrl_values = arrays.get("isXBRL", [0] * size)
    ordered_keys = tuple(sorted(arrays))
    row_identities: list[str] = []
    provisional: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position in range(size):
        raw_values = {key: arrays[key][position] for key in ordered_keys}
        for name in mandatory_text:
            if type(raw_values[name]) is not str:
                raise SecGemmaLeanV37SourceError(
                    f"{source_name}.{name} values must remain strings"
                )
        accession = raw_values["accessionNumber"]
        if _ACCESSION_RE.fullmatch(accession) is None:
            raise SecGemmaLeanV37SourceError(
                f"{source_name} contains a malformed accession"
            )
        if accession in seen:
            raise SecGemmaLeanV37SourceError(
                "Duplicate accession exists inside one Submissions source"
            )
        seen.add(accession)
        acceptance = raw_values["acceptanceDateTime"]
        if acceptance:
            try:
                parse_submissions_acceptance_datetime(acceptance)
            except SecPointInTimeError:
                raise SecGemmaLeanV37SourceError(
                    f"{source_name} contains malformed acceptanceDateTime"
                ) from None
        form = raw_values["form"]
        if not form or form != form.strip():
            raise SecGemmaLeanV37SourceError(f"{source_name} contains an invalid form")
        upper_form = form.upper()
        if upper_form in {"10-K", "10-Q", "10-K/A", "10-Q/A"} and form != upper_form:
            raise SecGemmaLeanV37SourceError(
                "Periodic Submissions form spelling is not canonical"
            )
        primary = raw_values["primaryDocument"]
        if form in {"10-K", "10-Q"}:
            _safe_primary_name(
                primary,
                location=f"{source_name}.primaryDocument",
                allow_empty=True,
            )
        filing_date = _canonical_date(
            raw_values["filingDate"], location=f"{source_name}.filingDate"
        )
        report_date = _canonical_date(
            raw_values["reportDate"],
            location=f"{source_name}.reportDate",
            allow_empty=True,
        )
        change = change_values[position]
        if type(change) is not str:
            raise SecGemmaLeanV37SourceError(
                "dateOfFilingDateChange values must remain strings"
            )
        change = _canonical_date(
            change,
            location=f"{source_name}.dateOfFilingDateChange",
            allow_empty=True,
        )
        is_xbrl = _flag(xbrl_values[position], location=f"{source_name}.isXBRL")
        semantic = {
            "accession_number": accession,
            "subject_cik": AAPL_CIK,
            "typed_values": raw_values,
            "date_of_filing_date_change_present": change_present,
        }
        semantic_hash = _canonical_sha256(semantic)
        row_identities.append(semantic_hash)
        provisional.append(
            {
                "accession": accession,
                "acceptance": acceptance,
                "form": form,
                "primary": primary,
                "items": raw_values["items"],
                "filing_date": filing_date,
                "report_date": report_date,
                "is_xbrl": is_xbrl,
                "change": change or None,
                "typed_values": raw_values,
                "semantic_hash": semantic_hash,
                "position": position,
            }
        )
    order_hash = _canonical_sha256(row_identities)
    rows = tuple(
        SubmissionRow(
            accession_number=item["accession"],
            subject_cik=AAPL_CIK,
            acceptance_datetime=item["acceptance"],
            form=item["form"],
            primary_document=item["primary"],
            items=item["items"],
            filing_date=item["filing_date"],
            report_date=item["report_date"],
            is_xbrl=item["is_xbrl"],
            date_of_filing_date_change=item["change"],
            date_of_filing_date_change_present=change_present,
            typed_values=item["typed_values"],
            semantic_identity_sha256=item["semantic_hash"],
            source_name=source_name,
            source_content_sha256=source_sha256,
            source_position=item["position"],
            source_row_order_sha256=order_hash,
        )
        for item in provisional
    )
    if reference is None:
        range_evidence = None
    else:
        if size != reference.filing_count or len(seen) != reference.filing_count:
            raise SecGemmaLeanV37SourceError(
                "Historical filingCount does not equal raw and unique row counts"
            )
        filing_dates = [row.filing_date for row in rows]
        if any(
            value < reference.filing_from or value > reference.filing_to
            for value in filing_dates
        ):
            raise SecGemmaLeanV37SourceError(
                "Historical filingDate lies outside its inclusive reference bounds"
            )
        observed_min = min(filing_dates)
        observed_max = max(filing_dates)
        lower_slack = (
            date.fromisoformat(observed_min) - date.fromisoformat(reference.filing_from)
        ).days
        upper_slack = (
            date.fromisoformat(reference.filing_to) - date.fromisoformat(observed_max)
        ).days
        range_evidence = {
            "filing_from": reference.filing_from,
            "filing_to": reference.filing_to,
            "observed_min_filing_date": observed_min,
            "observed_max_filing_date": observed_max,
            "filing_from_attained": observed_min == reference.filing_from,
            "filing_to_attained": observed_max == reference.filing_to,
            "lower_bound_slack_days": lower_slack,
            "upper_bound_slack_days": upper_slack,
            "raw_row_count": size,
            "unique_accession_count": len(seen),
            "row_order_sha256": order_hash,
        }
    return rows, order_hash, range_evidence


def parse_submissions_snapshot(
    main_payload: bytes,
    historical_payloads: Mapping[str, bytes],
) -> SubmissionsSnapshot:
    """Strictly parse one main response and its exact referenced history set."""

    main = _json_object(main_payload, location="main Submissions response")
    references = _historical_references(main)
    if type(historical_payloads) is not dict:
        # Exact dictionaries make caller mutation/subclass tricks fail closed.
        raise SecGemmaLeanV37SourceError(
            "Historical Submissions payloads must be an exact dictionary"
        )
    expected_names = {item.name for item in references}
    if set(historical_payloads) != expected_names or any(
        type(name) is not str or type(payload) is not bytes
        for name, payload in historical_payloads.items()
    ):
        raise SecGemmaLeanV37SourceError(
            "Historical Submissions response set is not exact"
        )
    main_sha = content_sha256(main_payload)
    main_rows, main_order_hash, _ = _parse_columns(
        main["filings"]["recent"],
        source_name=MAIN_SUBMISSIONS_NAME,
        source_sha256=main_sha,
        reference=None,
    )
    sources: list[SubmissionsSource] = [
        SubmissionsSource(
            name=MAIN_SUBMISSIONS_NAME,
            url=MAIN_SUBMISSIONS_URL,
            payload=main_payload,
            payload_sha256=main_sha,
            payload_length=len(main_payload),
            raw_row_count=len(main_rows),
            unique_row_count=len(main_rows),
            row_order_sha256=main_order_hash,
            range_evidence=None,
        )
    ]
    rows = list(main_rows)
    reference_by_name = {item.name: item for item in references}
    for name in sorted(expected_names):
        payload = historical_payloads[name]
        parsed = _json_object(payload, location=f"historical Submissions {name}")
        source_sha = content_sha256(payload)
        parsed_rows, order_hash, range_evidence = _parse_columns(
            parsed,
            source_name=name,
            source_sha256=source_sha,
            reference=reference_by_name[name],
        )
        sources.append(
            SubmissionsSource(
                name=name,
                url=reference_by_name[name].url,
                payload=payload,
                payload_sha256=source_sha,
                payload_length=len(payload),
                raw_row_count=len(parsed_rows),
                unique_row_count=len(parsed_rows),
                row_order_sha256=order_hash,
                range_evidence=range_evidence,
            )
        )
        rows.extend(parsed_rows)
    accessions = [row.accession_number for row in rows]
    if len(accessions) != len(set(accessions)):
        raise SecGemmaLeanV37SourceError(
            "Accession repeats across the main/historical Submissions union"
        )
    rows.sort(key=lambda row: (row.filing_date, row.accession_number))
    summary_body = {
        "references": [item.to_dict() for item in references],
        "sources": [source.summary() for source in sources],
        "semantic_rows_sha256": _canonical_sha256(
            [row.semantic_identity() for row in rows]
        ),
        "layout_rows_sha256": _canonical_sha256(
            [row.layout_provenance() for row in rows]
        ),
        "row_count": len(rows),
    }
    return SubmissionsSnapshot(
        references=references,
        sources=tuple(sources),
        rows=tuple(rows),
        snapshot_sha256=_canonical_sha256(summary_body),
    )


def _quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    if type(year) is not int or type(quarter) is not int or not 1 <= quarter <= 4:
        raise SecGemmaLeanV37SourceError("Master role year/quarter is invalid")
    first_month = 1 + (quarter - 1) * 3
    first = date(year, first_month, 1)
    if quarter == 4:
        following = date(year + 1, 1, 1)
    else:
        following = date(year, first_month + 3, 1)
    return first, following - timedelta(days=1)


def _strict_gzip_decompress(payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise SecGemmaLeanV37SourceError("master.gz must be exact response bytes")
    if len(payload) > MAX_MASTER_COMPRESSED_BYTES:
        raise SecGemmaLeanV37SourceError("master.gz compressed byte cap exceeded")
    if len(payload) < 18 or payload[:2] != b"\x1f\x8b":
        raise SecGemmaLeanV37SourceError("master.gz has invalid gzip magic")
    decoder = zlib.decompressobj(wbits=31)
    chunks: list[bytes] = []
    total = 0
    try:
        input_chunk_size = 64 * 1024
        output_chunk_size = 1024 * 1024
        for offset in range(0, len(payload), input_chunk_size):
            data = payload[offset : offset + input_chunk_size]
            while data:
                remaining = MAX_MASTER_DECOMPRESSED_BYTES - total
                part = decoder.decompress(
                    data,
                    min(output_chunk_size, remaining + 1),
                )
                chunks.append(part)
                total += len(part)
                if total > MAX_MASTER_DECOMPRESSED_BYTES:
                    raise SecGemmaLeanV37SourceError(
                        "master.gz decompressed byte cap exceeded"
                    )
                data = decoder.unconsumed_tail
                if decoder.eof:
                    if (
                        decoder.unused_data
                        or data
                        or offset + input_chunk_size < len(payload)
                    ):
                        raise SecGemmaLeanV37SourceError(
                            "master.gz contains trailing or concatenated member bytes"
                        )
                    data = b""
                elif data and not part:
                    raise SecGemmaLeanV37SourceError(
                        "master.gz decoder made no bounded progress"
                    )
        remaining = MAX_MASTER_DECOMPRESSED_BYTES - total
        tail = decoder.flush(remaining + 1)
        chunks.append(tail)
        total += len(tail)
    except zlib.error:
        raise SecGemmaLeanV37SourceError("master.gz failed CRC/ISIZE/gzip validation") from None
    if total > MAX_MASTER_DECOMPRESSED_BYTES:
        raise SecGemmaLeanV37SourceError("master.gz decompressed byte cap exceeded")
    if not decoder.eof:
        raise SecGemmaLeanV37SourceError("master.gz is truncated before member EOF")
    if decoder.unused_data or decoder.unconsumed_tail:
        raise SecGemmaLeanV37SourceError(
            "master.gz contains trailing or concatenated member bytes"
        )
    return b"".join(chunks)


def parse_strict_master_gzip(
    payload: bytes,
    *,
    year: int,
    quarter: int,
) -> MasterQuarterEvidence:
    """Parse exactly one valid, capped gzip member for one official quarter."""

    first, last = _quarter_bounds(year, quarter)
    role = QuarterRole(
        year=year,
        quarter=quarter,
        url=(
            "https://www.sec.gov/Archives/edgar/full-index/"
            f"{year}/QTR{quarter}/master.gz"
        ),
    )
    decompressed = _strict_gzip_decompress(payload)
    if re.search(rb"\r(?!\n)", decompressed) is not None or re.search(
        rb"[\x0b\x0c\x1c\x1d\x1e\x85]", decompressed
    ) is not None:
        raise SecGemmaLeanV37SourceError(
            "master.gz contains a noncanonical line-control byte"
        )
    text = decompressed.decode("latin-1")
    lines = text.replace("\r\n", "\n").split("\n")
    if lines.count(MASTER_HEADER) != 1:
        raise SecGemmaLeanV37SourceError(
            "master.gz must contain the complete canonical header exactly once"
        )
    start = lines.index(MASTER_HEADER) + 1
    raw_rows: list[str] = []
    targets: list[MasterTargetRow] = []
    seen_raw: set[str] = set()
    seen_target: set[str] = set()
    for line in lines[start:]:
        if not line or set(line) <= {"-"}:
            continue
        values = line.split("|")
        if len(values) != 5:
            raise SecGemmaLeanV37SourceError(
                "master.gz row does not contain exactly five fields"
            )
        if line in seen_raw:
            raise SecGemmaLeanV37SourceError("master.gz contains a repeated raw row")
        seen_raw.add(line)
        raw_rows.append(line)
        cik_raw, company, form, filed, filename = values
        cik = _canonical_cik(cik_raw, location="master CIK")
        is_apple_target = cik == AAPL_CIK and form in {"10-K", "10-Q"}
        if not form or not filename or (is_apple_target and not company) or any(
            character in "\x00\r\n" for character in company + form + filename
        ):
            raise SecGemmaLeanV37SourceError("master.gz row contains an empty field")
        filed = _canonical_date(filed, location="master filed date")
        filed_date = date.fromisoformat(filed)
        if not first <= filed_date <= last:
            raise SecGemmaLeanV37SourceError(
                "master.gz row filed date is outside the requested quarter"
            )
        if is_apple_target:
            expected_prefix = "edgar/data/320193/"
            if not filename.startswith(expected_prefix):
                raise SecGemmaLeanV37SourceError(
                    "Apple master target does not use the exact root archive path"
                )
            tail = filename[len(expected_prefix) :]
            matched = re.fullmatch(
                r"(?P<accession>[0-9]{10}-[0-9]{2}-[0-9]{6})\.txt", tail
            )
            if matched is None:
                raise SecGemmaLeanV37SourceError(
                    "Apple master target filename is not the exact accession path"
                )
            accession = matched.group("accession")
            if accession in seen_target:
                raise SecGemmaLeanV37SourceError(
                    "Apple target accession repeats inside one master"
                )
            seen_target.add(accession)
            targets.append(
                MasterTargetRow(
                    accession_number=accession,
                    cik=cik,
                    company_name=company,
                    form=form,
                    filing_date=filed,
                    filename=filename,
                    quarter=(year, quarter),
                )
            )
    targets.sort(key=lambda item: (item.filing_date, item.accession_number))
    return MasterQuarterEvidence(
        role=role,
        compressed_payload=payload,
        compressed_sha256=content_sha256(payload),
        compressed_length=len(payload),
        decompressed_sha256=content_sha256(decompressed),
        decompressed_length=len(decompressed),
        raw_row_count=len(raw_rows),
        raw_row_order_sha256=_canonical_sha256(raw_rows),
        target_rows=tuple(targets),
    )


def reconcile_master_exact_set(
    snapshot: SubmissionsSnapshot,
    master_quarters: Sequence[MasterQuarterEvidence],
    stage: str | StageSourceConfig,
) -> MasterReconciliation:
    """Require exact Apple non-amended 10-K/10-Q equality with all Q masters."""

    if not isinstance(snapshot, SubmissionsSnapshot):
        raise TypeError("snapshot must be a SubmissionsSnapshot")
    config = _stage_config(stage)
    if isinstance(master_quarters, (str, bytes)) or not isinstance(
        master_quarters, Sequence
    ):
        raise SecGemmaLeanV37SourceError("master quarters must be a sequence")
    expected_roles = quarter_roles(config)
    evidence_by_key: dict[tuple[int, int], MasterQuarterEvidence] = {}
    for evidence in master_quarters:
        if not isinstance(evidence, MasterQuarterEvidence):
            raise SecGemmaLeanV37SourceError("master evidence type is invalid")
        reparsed = parse_strict_master_gzip(
            evidence.compressed_payload,
            year=evidence.role.year,
            quarter=evidence.role.quarter,
        )
        if evidence != reparsed:
            raise SecGemmaLeanV37SourceError(
                "master evidence does not authenticate its compressed bytes"
            )
        if evidence.role.key in evidence_by_key:
            raise SecGemmaLeanV37SourceError("master quarter evidence is duplicated")
        evidence_by_key[evidence.role.key] = evidence
    if set(evidence_by_key) != {role.key for role in expected_roles}:
        raise SecGemmaLeanV37SourceError("master quarter evidence set is not exact")
    ordered_evidence = tuple(evidence_by_key[role.key] for role in expected_roles)
    master_by_accession: dict[str, MasterTargetRow] = {}
    cutoff = config.fixed_cutoff
    for evidence in ordered_evidence:
        for row in evidence.target_rows:
            if row.filing_date > cutoff:
                continue
            if row.accession_number in master_by_accession:
                raise SecGemmaLeanV37SourceError(
                    "Apple target accession repeats across quarterly masters"
                )
            master_by_accession[row.accession_number] = row
    submissions_by_accession: dict[str, SubmissionRow] = {}
    for row in _submissions_candidates(snapshot, config):
        submissions_by_accession[row.accession_number] = row
    if set(submissions_by_accession) != set(master_by_accession):
        missing = sorted(set(submissions_by_accession) - set(master_by_accession))
        unexpected = sorted(set(master_by_accession) - set(submissions_by_accession))
        raise SecGemmaLeanV37SourceError(
            "Master/Submissions Apple target set differs exactly: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    targets: list[ReconciledTarget] = []
    for accession, submissions in submissions_by_accession.items():
        master = master_by_accession[accession]
        if (
            master.cik != submissions.subject_cik
            or master.form != submissions.form
            or master.filing_date != submissions.filing_date
        ):
            raise SecGemmaLeanV37SourceError(
                "Master/Submissions Apple target identity conflicts"
            )
        expected_filename = f"edgar/data/320193/{accession}.txt"
        if master.filename != expected_filename:
            raise SecGemmaLeanV37SourceError(
                "Master Apple target path does not reconcile exactly"
            )
        targets.append(ReconciledTarget(submissions=submissions, master=master))
    targets.sort(
        key=lambda item: (
            item.submissions.filing_date,
            item.submissions.accession_number,
        )
    )
    if len(targets) > config.i_cap:
        raise SecGemmaLeanV37SourceError("Frozen I cap is exceeded")
    body = {
        "stage": config.stage,
        "config": config.to_dict(),
        "quarters": [item.summary() for item in ordered_evidence],
        "targets": [
            {
                "submissions_semantic_identity_sha256": (
                    item.submissions.semantic_identity_sha256
                ),
                "master": item.master.to_dict(),
            }
            for item in targets
        ],
    }
    return MasterReconciliation(
        config=config,
        quarter_evidence=ordered_evidence,
        targets=tuple(targets),
        reconciliation_sha256=_canonical_sha256(body),
    )


def _validate_seal_hash(seal: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(seal, Mapping):
        raise SecGemmaLeanV37SourceError("Prior stage source seal is not an object")
    value = dict(seal)
    supplied = value.pop("stage_source_seal_sha256", None)
    if type(supplied) is not str or _BARE_SHA256_RE.fullmatch(supplied) is None:
        raise SecGemmaLeanV37SourceError("Stage source seal hash is malformed")
    if _canonical_sha256(value) != supplied:
        raise SecGemmaLeanV37SourceError("Stage source seal hash does not validate")
    return dict(seal)


def _authenticated_prior_authority(
    config: StageSourceConfig,
    prior_bundle: StageSourceBundle | None,
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Replay the retained adjacent bundle (and recursively every ancestor)."""

    if config.stage == "development":
        if prior_bundle is not None:
            raise SecGemmaLeanV37SourceError(
                "Development cannot carry prior source authority"
            )
        return None, ()
    if not isinstance(prior_bundle, StageSourceBundle):
        raise SecGemmaLeanV37SourceError(
            "A later v3.7 stage requires its exact adjacent prior raw bundle"
        )
    expected_prior_stage = _STAGE_ORDER[_STAGE_ORDER.index(config.stage) - 1]
    if prior_bundle.config.stage != expected_prior_stage:
        raise SecGemmaLeanV37SourceError("Prior source bundle stage is not adjacent")
    # Detached replay is the authority check.  It recursively authenticates
    # the entire retained chain before any prior accession is subtracted.
    detached_replay_stage_source(prior_bundle)
    validate_stage_source_seal(prior_bundle.seal)
    prior = _canonical_snapshot(prior_bundle.seal)
    targets = prior["target_rows"]
    accessions = tuple(row["accession_number"] for row in targets)
    return prior, accessions


def _submissions_candidates(
    snapshot: SubmissionsSnapshot,
    config: StageSourceConfig,
) -> tuple[SubmissionRow, ...]:
    coverage_start = _quarter_bounds(
        config.master_start_year, config.master_start_quarter
    )[0]
    coverage_end = _quarter_bounds(
        config.master_end_year, config.master_end_quarter
    )[1]
    candidates: list[SubmissionRow] = []
    for row in snapshot.rows:
        if not row.is_target:
            continue
        filed = date.fromisoformat(row.filing_date)
        if filed < coverage_start:
            raise SecGemmaLeanV37SourceError("master_boundary_incomplete")
        if row.filing_date > config.fixed_cutoff:
            continue
        if filed > coverage_end:
            raise SecGemmaLeanV37SourceError(
                "Submissions target lies outside declared master coverage"
            )
        candidates.append(row)
    candidates.sort(key=lambda row: (row.filing_date, row.accession_number))
    return tuple(candidates)


def build_pre_master_submissions_upper_bound(
    snapshot: SubmissionsSnapshot,
    stage: str | StageSourceConfig,
    *,
    prior_bundle: StageSourceBundle | None = None,
) -> Mapping[str, Any]:
    """Reject I/P cap failures before masters without authorizing membership."""

    if not isinstance(snapshot, SubmissionsSnapshot):
        raise TypeError("snapshot must be a SubmissionsSnapshot")
    config = _stage_config(stage)
    prior, _ = _authenticated_prior_authority(config, prior_bundle)
    return _pre_master_upper_bound_from_prior_seal(snapshot, config, prior)


def _pre_master_upper_bound_from_prior_seal(
    snapshot: SubmissionsSnapshot,
    config: StageSourceConfig,
    prior_seal: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if prior_seal is None:
        prior_accessions: tuple[str, ...] = ()
    else:
        validate_stage_source_seal(prior_seal)
        prior_accessions = tuple(
            row["accession_number"] for row in prior_seal["target_rows"]
        )
    candidates = _submissions_candidates(snapshot, config)
    candidate_accessions = tuple(row.accession_number for row in candidates)
    if len(candidates) > config.i_cap:
        raise SecGemmaLeanV37SourceError(
            "Pre-master Submissions I upper bound exceeds the frozen cap"
        )
    prior_set = set(prior_accessions)
    candidate_set = set(candidate_accessions)
    if not prior_set <= candidate_set:
        raise SecGemmaLeanV37SourceError(
            "Pre-master current candidates omit prior sealed targets"
        )
    p_accessions = tuple(
        accession for accession in candidate_accessions if accession not in prior_set
    )
    if len(p_accessions) > config.p_cap:
        raise SecGemmaLeanV37SourceError(
            "Pre-master Submissions P upper bound exceeds the frozen cap"
        )
    body = {
        "schema_version": PRE_MASTER_UPPER_BOUND_SCHEMA_VERSION,
        "stage": config.stage,
        "submissions_snapshot_sha256": snapshot.snapshot_sha256,
        "candidate_ordering": "filing_date_then_accession",
        "candidate_accessions": list(candidate_accessions),
        "candidate_accessions_sha256": _canonical_sha256(candidate_accessions),
        "i_upper_bound": len(candidate_accessions),
        "source_document_accessions": list(p_accessions),
        "source_document_accessions_sha256": _canonical_sha256(p_accessions),
        "p_upper_bound": len(p_accessions),
        "prior_stage_source_seal_sha256": (
            prior_seal["stage_source_seal_sha256"]
            if prior_seal is not None
            else None
        ),
        "authority": "reject_only_submissions_bound_not_master_membership",
    }
    return _canonical_snapshot(
        {**body, "pre_master_upper_bound_sha256": _canonical_sha256(body)}
    )


def derive_complete_submission_plan(
    reconciliation: MasterReconciliation,
    *,
    prior_bundle: StageSourceBundle | None = None,
) -> tuple[ReconciledTarget, ...]:
    """Freeze P only after exact replay of the full retained prior chain."""

    if not isinstance(reconciliation, MasterReconciliation):
        raise TypeError("reconciliation must be a MasterReconciliation")
    config = reconciliation.config
    _, prior_accessions = _authenticated_prior_authority(config, prior_bundle)
    current_accessions = set(reconciliation.accessions)
    prior_set = set(prior_accessions)
    if not prior_set <= current_accessions:
        raise SecGemmaLeanV37SourceError(
            "Current I does not contain every prior sealed accession"
        )
    plan = tuple(
        target
        for target in reconciliation.targets
        if target.submissions.accession_number not in prior_set
    )
    if config.stage == "development" and len(plan) != len(reconciliation.targets):
        raise SecGemmaLeanV37SourceError("Development requires P exactly equal I")
    if len(plan) > config.p_cap:
        raise SecGemmaLeanV37SourceError("Frozen P cap is exceeded")
    return plan


def _canonical_sessions(values: Sequence[str]) -> tuple[str, ...]:
    _authoritative_calendar_receipt(values)
    return tuple(EXPECTED_SESSIONS)


def _conservative_availability_session(boundary: str) -> str | None:
    boundary = _canonical_date(boundary, location="availability boundary")
    exact_boundary_start = _AUTHORITATIVE_CALENDAR_RECEIPT[
        "availability_semantics"
    ]["exact_boundary_start"]
    if boundary < exact_boundary_start:
        return None
    return next(
        (
            session
            for session in EXPECTED_MARKET_HISTORY_SESSIONS
            if session > boundary
        ),
        None,
    )


def _stage_assignment(availability: str | None) -> str | None:
    if availability is None:
        return None
    for stage in _STAGE_ORDER:
        first, last = _STAGE_WINDOWS[stage]
        if first <= availability <= last:
            return stage
    return None


def _selector_record(row: SubmissionRow, normalized_acceptance: str | None) -> FilingRecord:
    # FilingRecord's constructor predates v3.7 and requires an acceptance value.
    # The selector itself reads only accession/form/filing/primary identity.  A
    # local valid placeholder is therefore used solely at that type boundary
    # when both authenticated acceptance sources are missing; it is never
    # returned, hashed, or used to compute availability.
    acceptance = normalized_acceptance or row.filing_date.replace("-", "") + "120000"
    return FilingRecord(
        accession_number=row.accession_number,
        acceptance_datetime=acceptance,
        form=row.form,
        primary_document=row.primary_document,
        items=row.items,
        filing_date=row.filing_date,
        report_date=row.report_date,
        is_xbrl=row.is_xbrl,
        source_name=row.source_name,
        subject_cik=row.subject_cik,
        date_of_filing_date_change=row.date_of_filing_date_change or "",
    )


def _exact_header_sgml_form(
    complete_response: bytes,
    *,
    expected_form: str,
) -> str:
    """Require one exact raw SEC/IMS-header form value for v3.7."""

    text = complete_response.decode("latin-1")
    header_family = "SEC"
    selected_ims_header: str | None = None
    plain_form_pattern = (
        r"^[ \t]*CONFORMED SUBMISSION TYPE:[ \t]+([^\r\n]+)$"
    )
    ims_tokens = (
        "<IMS-DOCUMENT>",
        "</IMS-DOCUMENT>",
        "<IMS-HEADER>",
        "</IMS-HEADER>",
    )
    if any(
        re.search(
            re.escape(token), text, flags=re.IGNORECASE | re.ASCII
        )
        is not None
        for token in ims_tokens
    ):
        documents = list(
            re.finditer(
                r"<DOCUMENT>(.*?)</DOCUMENT>",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        authenticated_spans: list[tuple[int, int]] = []
        if (
            documents
            and len(re.findall(r"<DOCUMENT>", text, flags=re.IGNORECASE))
            == len(documents)
            and len(re.findall(r"</DOCUMENT>", text, flags=re.IGNORECASE))
            == len(documents)
        ):
            for document in documents:
                block = document.group(1)
                text_matches = list(
                    re.finditer(
                        r"<TEXT>(.*?)</TEXT>",
                        block,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                )
                if (
                    len(text_matches) != 1
                    or len(re.findall(r"<TEXT>", block, flags=re.IGNORECASE)) != 1
                    or len(re.findall(r"</TEXT>", block, flags=re.IGNORECASE)) != 1
                ):
                    authenticated_spans = []
                    break
                metadata = block[: text_matches[0].start()]
                raw_types = re.findall(
                    r"^\s*<TYPE>\s*([^\r\n<]+?)\s*$",
                    metadata,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                raw_sequences = re.findall(
                    r"^\s*<SEQUENCE>\s*([^\r\n<]+?)\s*$",
                    metadata,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                if (
                    len(raw_types) != 1
                    or not raw_types[0].strip()
                    or len(raw_sequences) != 1
                    or not raw_sequences[0].strip().isdigit()
                    or int(raw_sequences[0].strip()) < 1
                ):
                    authenticated_spans = []
                    break
                authenticated_spans.append(
                    (
                        document.start(1) + text_matches[0].start(1),
                        document.start(1) + text_matches[0].end(1),
                    )
                )
        structural_characters = list(text)
        if len(authenticated_spans) == len(documents) and documents:
            for start, end in authenticated_spans:
                structural_characters[start:end] = " " * (end - start)
        structural = "".join(structural_characters)
        structural_has_ims = any(
            re.search(
                re.escape(token),
                structural,
                flags=re.IGNORECASE | re.ASCII,
            )
            is not None
            for token in ims_tokens
        )
        if structural_has_ims:
            family_matches = {
                token: list(
                    re.finditer(
                        re.escape(token),
                        structural,
                        flags=re.IGNORECASE | re.ASCII,
                    )
                )
                for token in (
                    "<IMS-DOCUMENT>",
                    "</IMS-DOCUMENT>",
                    "<IMS-HEADER>",
                    "</IMS-HEADER>",
                    "<SEC-DOCUMENT>",
                    "</SEC-DOCUMENT>",
                    "<SEC-HEADER>",
                    "</SEC-HEADER>",
                )
            }
            if (
                any(
                    len(family_matches[token]) != 1
                    for token in ims_tokens
                )
                or any(
                    family_matches[token]
                    for token in (
                        "<SEC-DOCUMENT>",
                        "</SEC-DOCUMENT>",
                        "<SEC-HEADER>",
                        "</SEC-HEADER>",
                    )
                )
                or len(authenticated_spans) != len(documents)
                or not documents
                or not (
                    family_matches["<IMS-DOCUMENT>"][0].start()
                    < family_matches["<IMS-HEADER>"][0].start()
                    < family_matches["</IMS-HEADER>"][0].start()
                    < documents[0].start()
                    <= documents[-1].end()
                    < family_matches["</IMS-DOCUMENT>"][0].start()
                )
            ):
                raise SecGemmaLeanV37SourceError(
                    "Complete submission IMS header is ambiguous"
                )
            header_family = "IMS"
            selected_ims_header = text[
                family_matches["<IMS-HEADER>"][0].end() :
                family_matches["</IMS-HEADER>"][0].start()
            ]
    headers = (
        [selected_ims_header]
        if header_family == "IMS" and selected_ims_header is not None
        else re.findall(
            r"<SEC-HEADER>(.*?)</SEC-HEADER>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if len(headers) != 1:
        raise SecGemmaLeanV37SourceError(
            f"Complete submission {header_family} header is ambiguous"
        )
    raw_forms: list[str] = []
    for pattern in (
        r"<CONFORMED-SUBMISSION-TYPE>([^\r\n<]+)",
        r"<TYPE>([^\r\n<]+)",
        plain_form_pattern,
    ):
        raw_forms.extend(
            re.findall(pattern, headers[0], flags=re.IGNORECASE | re.MULTILINE)
        )
    if len(raw_forms) != 1 or raw_forms[0] != expected_form:
        raise SecGemmaLeanV37SourceError(
            f"{header_family} header form is not the one exact raw 10-K/10-Q form"
        )
    return raw_forms[0]


def _exact_document_form_sequence_topology(
    complete_response: bytes,
    *,
    selected_ordinal: int,
    expected_form: str,
) -> str:
    """Require v3.7's exact all-document TYPE/sequence topology."""

    text = complete_response.decode("latin-1")
    documents = list(
        re.finditer(
            r"<DOCUMENT>(.*?)</DOCUMENT>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not 1 <= selected_ordinal <= len(documents):
        raise SecGemmaLeanV37SourceError("Selected SGML document ordinal is invalid")
    sequence_one_ordinals: list[int] = []
    exact_form_ordinals: list[int] = []
    selected_raw_type: str | None = None
    for ordinal, document in enumerate(documents, start=1):
        block = document.group(1)
        text_start = re.search(r"<TEXT>", block, flags=re.IGNORECASE)
        if text_start is None:
            raise SecGemmaLeanV37SourceError("SGML document lacks TEXT")
        metadata = block[: text_start.start()]
        raw_types = re.findall(
            r"^\s*<TYPE>([^\r\n<]*)$",
            metadata,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        raw_sequences = re.findall(
            r"^\s*<SEQUENCE>\s*([^\r\n<]*)\s*$",
            metadata,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if len(raw_types) != 1 or len(raw_sequences) != 1:
            raise SecGemmaLeanV37SourceError(
                "SGML document TYPE/SEQUENCE fields are not singular"
            )
        sequence_token = raw_sequences[0].strip()
        if not sequence_token.isdigit() or int(sequence_token) < 1:
            raise SecGemmaLeanV37SourceError(
                "SGML document SEQUENCE is not a positive integer"
            )
        if int(sequence_token) == 1:
            sequence_one_ordinals.append(ordinal)
        if raw_types[0] == expected_form:
            exact_form_ordinals.append(ordinal)
        if ordinal == selected_ordinal:
            selected_raw_type = raw_types[0]
    if (
        sequence_one_ordinals != [selected_ordinal]
        or exact_form_ordinals != [selected_ordinal]
        or selected_raw_type != expected_form
    ):
        raise SecGemmaLeanV37SourceError(
            "Complete submission lacks one matching sequence-1 exact-form document"
        )
    return selected_raw_type


def reconcile_complete_submission(
    target: ReconciledTarget,
    complete_response: bytes,
    *,
    session_dates: Sequence[str],
    acquisition_stage: str,
) -> CompleteSourceEvidence:
    """Reconcile one root complete response and seal three byte identities."""

    if not isinstance(target, ReconciledTarget):
        raise TypeError("target must be a ReconciledTarget")
    if type(complete_response) is not bytes:
        raise SecGemmaLeanV37SourceError(
            "Complete submission must be exact response bytes"
        )
    if acquisition_stage not in _STAGE_ORDER:
        raise SecGemmaLeanV37SourceError("Complete source acquisition stage is invalid")
    sessions = _canonical_sessions(session_dates)
    row, master = target.submissions, target.master
    try:
        submission = parse_complete_submission(
            complete_response,
            allow_missing_acceptance=True,
        )
    except (SecPointInTimeError, TypeError, ValueError):
        raise SecGemmaLeanV37SourceError(
            "Complete submission SGML failed strict parsing"
        ) from None
    header = submission.header
    raw_header_form = _exact_header_sgml_form(
        complete_response,
        expected_form=row.form,
    )
    if (
        header.accession_number != row.accession_number
        or header.subject_cik != row.subject_cik
        or header.form != row.form
        or raw_header_form != row.form
        or header.filing_date != row.filing_date
        or master.accession_number != row.accession_number
        or master.cik != row.subject_cik
        or master.form != row.form
        or master.filing_date != row.filing_date
        or master.filename != f"edgar/data/320193/{row.accession_number}.txt"
    ):
        raise SecGemmaLeanV37SourceError(
            "Complete/Submissions/master identity does not reconcile"
        )
    acceptance_sources: dict[str, str | None] = {
        "submissions": None,
        "header": None,
    }
    if row.acceptance_datetime:
        acceptance_sources["submissions"] = (
            parse_submissions_acceptance_datetime(row.acceptance_datetime).strftime(
                "%Y%m%d%H%M%S"
            )
        )
    if header.acceptance_datetime:
        acceptance_sources["header"] = (
            parse_submissions_acceptance_datetime(header.acceptance_datetime).strftime(
                "%Y%m%d%H%M%S"
            )
        )
    supplied_acceptances = {
        value for value in acceptance_sources.values() if value is not None
    }
    if len(supplied_acceptances) > 1:
        raise SecGemmaLeanV37SourceError("Acceptance values do not reconcile lexically")
    normalized_acceptance = (
        next(iter(supplied_acceptances)) if supplied_acceptances else None
    )
    filing_date = date.fromisoformat(row.filing_date)
    if normalized_acceptance is not None:
        acceptance_date = datetime.strptime(
            normalized_acceptance[:8], "%Y%m%d"
        ).date()
        if acceptance_date > filing_date:
            raise SecGemmaLeanV37SourceError(
                "Authenticated acceptance occurs after the filing date"
            )
    submissions_change = row.date_of_filing_date_change
    header_change = header.date_of_filing_date_change
    if submissions_change and header_change and submissions_change != header_change:
        raise SecGemmaLeanV37SourceError(
            "DATE AS OF CHANGE does not reconcile between sources"
        )
    normalized_change = submissions_change or header_change
    try:
        selection = select_sequence_one_primary_document(
            _selector_record(row, normalized_acceptance),
            submission,
        )
    except (SecPointInTimeError, TypeError, ValueError):
        raise SecGemmaLeanV37SourceError(
            "Complete submission has no unique reconciled sequence-1 primary"
        ) from None
    document = selection.document
    exact_raw_type = _exact_document_form_sequence_topology(
        complete_response,
        selected_ordinal=document.ordinal,
        expected_form=row.form,
    )
    if exact_raw_type != row.form:
        raise SecGemmaLeanV37SourceError(
            "Selected SGML TYPE is not the exact raw target form"
        )
    if not (
        0 <= document.text_start_byte <= document.text_end_byte <= len(complete_response)
    ):
        raise SecGemmaLeanV37SourceError("Embedded TEXT byte offsets are invalid")
    extracted = complete_response[
        document.text_start_byte : document.text_end_byte
    ]
    if content_sha256(extracted) != document.text_sha256:
        raise SecGemmaLeanV37SourceError(
            "Embedded TEXT bytes do not match parser provenance"
        )
    normalized = normalize_filing_text(document.text)
    if not normalized.usable:
        raise SecGemmaLeanV37SourceError("Selected embedded TEXT is unusable")
    normalized_bytes = normalized.text.encode("utf-8")
    evidence_dates = [filing_date]
    if normalized_acceptance is not None:
        evidence_dates.append(
            datetime.strptime(normalized_acceptance[:8], "%Y%m%d").date()
        )
    if normalized_change is not None:
        evidence_dates.append(date.fromisoformat(normalized_change))
    boundary = max(evidence_dates).isoformat()
    availability = _conservative_availability_session(boundary)
    assignment = _stage_assignment(availability)
    response_provenance = {
        "url": master.complete_submission_url,
        "sha256": content_sha256(complete_response),
        "length": len(complete_response),
    }
    selection_provenance = {
        "selected_document_ordinal": document.ordinal,
        "sequence": document.sequence,
        "type": document.document_type,
        "raw_type": exact_raw_type,
        "sec_filename": selection.sec_filename,
        "document_identity": selection.document_identity,
        "sgml_document_identity": document.document_identity,
        "submissions_filename": row.primary_document or None,
        "sgml_filename": document.filename,
        "submissions_filename_missing": selection.submissions_filename_missing,
        "sgml_filename_missing": selection.sgml_filename_missing,
    }
    extracted_provenance = {
        "start_byte": document.text_start_byte,
        "end_byte": document.text_end_byte,
        "length": len(extracted),
        "sha256": content_sha256(extracted),
        "encoding": "latin-1 exact one-to-one response slice",
    }
    normalized_provenance = {
        "length": len(normalized_bytes),
        "character_count": normalized.character_count,
        "sha256": content_sha256(normalized_bytes),
        "normalizer": "inherited normalize_filing_text byte-identical rules",
    }
    frozen_prefix = {
        "accession_number": row.accession_number,
        "submissions_semantic_identity": row.semantic_identity(),
        "submissions_semantic_identity_sha256": row.semantic_identity_sha256,
        "master_identity": {
            "accession_number": master.accession_number,
            "cik": master.cik,
            "form": master.form,
            "filing_date": master.filing_date,
            "filename": master.filename,
        },
        "header_identity": {
            "accession_number": header.accession_number,
            "subject_cik": header.subject_cik,
            "form": header.form,
            "raw_form": raw_header_form,
            "filing_date": header.filing_date,
            "acceptance_datetime": header.acceptance_datetime,
            "date_of_filing_date_change": header.date_of_filing_date_change,
        },
        "acceptance": {
            "submissions_source": row.acceptance_datetime or None,
            "header_source": header.acceptance_datetime,
            "normalized_et": normalized_acceptance,
            "submissions_missing": not bool(row.acceptance_datetime),
            "header_missing": header.acceptance_datetime is None,
            "exact": normalized_acceptance is not None,
        },
        "filing_date_change": {
            "submissions_source": submissions_change,
            "submissions_column_present": row.date_of_filing_date_change_present,
            "header_source": header_change,
            "normalized": normalized_change,
            "submissions_missing": submissions_change is None,
            "header_missing": header_change is None,
        },
        "availability_session": availability,
        "availability_not_before_date": boundary,
        "immutable_stage_assignment": assignment,
        "complete_response": response_provenance,
        "primary_selection": selection_provenance,
        "extracted_text": extracted_provenance,
        "normalized_text": normalized_provenance,
    }
    seal_row = _canonical_snapshot({
        "accession_number": row.accession_number,
        "filing_date": row.filing_date,
        "form": row.form,
        "acquisition_stage": acquisition_stage,
        "stage_assignment": assignment,
        "availability_session": availability,
        "exact_acceptance": normalized_acceptance is not None,
        "submissions_layout_provenance": row.layout_provenance(),
        "complete_response": response_provenance,
        "primary_selection": selection_provenance,
        "extracted_text": extracted_provenance,
        "normalized_text": normalized_provenance,
        "frozen_prefix": frozen_prefix,
        "frozen_prefix_sha256": _canonical_sha256(frozen_prefix),
    })
    return CompleteSourceEvidence(
        accession_number=row.accession_number,
        acquisition_stage=acquisition_stage,
        complete_response=complete_response,
        extracted_primary=extracted,
        normalized_text=normalized_bytes,
        seal_row=seal_row,
    )


def validate_cross_stage_frozen_prefix(
    prior_seal: Mapping[str, Any],
    current_seal: Mapping[str, Any],
) -> None:
    """Reject every semantic prior-prefix drift while allowing layout migration."""

    validate_stage_source_seal(prior_seal)
    validate_stage_source_seal(current_seal)
    prior = _validate_seal_hash(prior_seal)
    current = _validate_seal_hash(current_seal)
    prior_stage = prior.get("stage")
    current_stage = current.get("stage")
    if (
        prior_stage not in _STAGE_ORDER
        or current_stage not in _STAGE_ORDER
        or _STAGE_ORDER.index(current_stage) != _STAGE_ORDER.index(prior_stage) + 1
    ):
        raise SecGemmaLeanV37SourceError("Cross-stage source seals are not adjacent")
    expected_chain = [
        *_canonical_snapshot(prior["prior_stage_chain"]),
        {
            "stage": prior_stage,
            "stage_source_seal_sha256": prior["stage_source_seal_sha256"],
        },
    ]
    if (
        current.get("prior_stage_source_seal_sha256")
        != prior["stage_source_seal_sha256"]
        or current.get("prior_stage_chain") != expected_chain
    ):
        raise SecGemmaLeanV37SourceError(
            "Cross-stage adjacent hash/full chain is not exact"
        )
    prior_config = prior.get("config")
    if type(prior_config) is not dict:
        raise SecGemmaLeanV37SourceError("Prior source seal config is invalid")
    prior_cutoff = _canonical_date(
        prior_config.get("fixed_cutoff"), location="prior fixed cutoff"
    )

    def indexed(rows: Any, *, cutoff: str | None) -> dict[str, dict[str, Any]]:
        if type(rows) is not list:
            raise SecGemmaLeanV37SourceError("Stage source target rows are invalid")
        result: dict[str, dict[str, Any]] = {}
        for raw in rows:
            if type(raw) is not dict:
                raise SecGemmaLeanV37SourceError("Stage source target row is invalid")
            accession = raw.get("accession_number")
            filing_date = raw.get("filing_date")
            if (
                type(accession) is not str
                or _ACCESSION_RE.fullmatch(accession) is None
                or type(filing_date) is not str
            ):
                raise SecGemmaLeanV37SourceError(
                    "Stage source target row identity is invalid"
                )
            _canonical_date(filing_date, location="stage source filing date")
            if cutoff is not None and filing_date > cutoff:
                continue
            if accession in result:
                raise SecGemmaLeanV37SourceError(
                    "Stage source target rows contain a duplicate accession"
                )
            prefix = raw.get("frozen_prefix")
            prefix_hash = raw.get("frozen_prefix_sha256")
            if (
                type(prefix) is not dict
                or type(prefix_hash) is not str
                or _BARE_SHA256_RE.fullmatch(prefix_hash) is None
                or _canonical_sha256(prefix) != prefix_hash
            ):
                raise SecGemmaLeanV37SourceError(
                    "Stage source frozen-prefix identity is not hash-bound"
                )
            result[accession] = raw
        return result

    prior_rows = indexed(prior.get("target_rows"), cutoff=None)
    current_prefix = indexed(current.get("target_rows"), cutoff=prior_cutoff)
    if set(prior_rows) != set(current_prefix):
        raise SecGemmaLeanV37SourceError(
            "Cross-stage target prefix has a new, missing, or duplicated accession"
        )
    for accession in sorted(prior_rows):
        left = prior_rows[accession]
        right = current_prefix[accession]
        if (
            left["frozen_prefix_sha256"] != right["frozen_prefix_sha256"]
            or left["frozen_prefix"] != right["frozen_prefix"]
        ):
            raise SecGemmaLeanV37SourceError(
                f"Cross-stage frozen prefix drifted for {accession}"
            )
        # Acquisition stage must remain the original one.  Layout provenance is
        # intentionally not compared: an identical typed row may move between
        # the main and historical Submissions files.
        if left.get("acquisition_stage") != right.get("acquisition_stage"):
            raise SecGemmaLeanV37SourceError(
                f"Cross-stage carried source authority drifted for {accession}"
            )
    prior_order = [row["accession_number"] for row in prior["target_rows"]]
    current_order = [row["accession_number"] for row in current["target_rows"]]
    expected_p = [
        accession for accession in current_order if accession not in set(prior_order)
    ]
    plan = current["source_plan"]
    if (
        plan["carried_accessions"] != prior_order
        or plan["new_complete_submission_accessions"] != expected_p
    ):
        raise SecGemmaLeanV37SourceError(
            "Cross-stage P/carry subtraction is not exactly prior targets"
        )


def _seal_target_rows(
    reconciliation: MasterReconciliation,
    sources: Mapping[str, CompleteSourceEvidence],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in reconciliation.targets:
        accession = target.submissions.accession_number
        source = sources.get(accession)
        if source is None:
            raise SecGemmaLeanV37SourceError(
                "An I member has no exact complete-submission source seal"
            )
        result.append(_canonical_snapshot(source.seal_row))
    return result


def validate_completed_year_source_coverage(
    target_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Require the inherited 2000-2025 availability-year source coverage."""

    if isinstance(target_rows, (str, bytes)) or not isinstance(
        target_rows, Sequence
    ):
        raise SecGemmaLeanV37SourceError(
            "Completed-year coverage rows must be a sequence"
        )
    by_year: dict[int, list[Mapping[str, Any]]] = {
        year: [] for year in range(COMPLETED_YEAR_START, COMPLETED_YEAR_END + 1)
    }
    for row in target_rows:
        if type(row) is not dict:
            raise SecGemmaLeanV37SourceError(
                "Completed-year coverage row is invalid"
            )
        availability = row.get("availability_session")
        assignment = row.get("stage_assignment")
        accession = row.get("accession_number")
        form = row.get("form")
        if (
            type(accession) is not str
            or _ACCESSION_RE.fullmatch(accession) is None
            or form not in {"10-K", "10-Q"}
        ):
            raise SecGemmaLeanV37SourceError(
                "Completed-year coverage row identity is invalid"
            )
        if availability is None:
            if assignment is not None:
                raise SecGemmaLeanV37SourceError(
                    "Completed-year coverage missing availability has an assignment"
                )
            continue
        if (
            type(availability) is not str
            or availability not in _EXPECTED_SESSION_SET
            or assignment != _stage_assignment(availability)
        ):
            raise SecGemmaLeanV37SourceError(
                "Completed-year coverage availability is not authoritative"
            )
        year = int(availability[:4])
        if year in by_year:
            by_year[year].append(row)
    year_receipts: list[dict[str, Any]] = []
    for year in range(COMPLETED_YEAR_START, COMPLETED_YEAR_END + 1):
        rows = sorted(
            by_year[year],
            key=lambda row: (
                row["availability_session"],
                row["accession_number"],
            ),
        )
        forms = [row.get("form") for row in rows]
        total = len(rows)
        ten_k = forms.count("10-K")
        ten_q = forms.count("10-Q")
        if total < 3 or ten_k < 1 or ten_q < 2:
            raise SecGemmaLeanV37SourceError(
                f"Completed availability year {year} lacks 3/1K/2Q source coverage"
            )
        accessions = [row["accession_number"] for row in rows]
        year_receipts.append(
            {
                "year": year,
                "total": total,
                "10-K": ten_k,
                "10-Q": ten_q,
                "accessions": accessions,
                "accessions_sha256": _canonical_sha256(accessions),
            }
        )
    body = {
        "schema_version": COMPLETED_YEAR_COVERAGE_SCHEMA_VERSION,
        "availability_year_start": COMPLETED_YEAR_START,
        "availability_year_end": COMPLETED_YEAR_END,
        "minimum_total": 3,
        "minimum_10-K": 1,
        "minimum_10-Q": 2,
        "ordering": "availability_session_then_accession",
        "years": year_receipts,
    }
    return _canonical_snapshot(
        {**body, "completed_year_coverage_sha256": _canonical_sha256(body)}
    )


def _assemble_stage_seal(
    *,
    config: StageSourceConfig,
    snapshot: SubmissionsSnapshot,
    reconciliation: MasterReconciliation,
    complete_sources: Sequence[CompleteSourceEvidence],
    current_complete_accessions: Sequence[str],
    prior_seal: Mapping[str, Any] | None,
    pre_master_upper_bound: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    source_by_accession: dict[str, CompleteSourceEvidence] = {}
    for source in complete_sources:
        if not isinstance(source, CompleteSourceEvidence):
            raise SecGemmaLeanV37SourceError("Complete source evidence type is invalid")
        if source.accession_number in source_by_accession:
            raise SecGemmaLeanV37SourceError("Complete source evidence is duplicated")
        if content_sha256(source.complete_response) != source.complete_response_sha256:
            raise SecGemmaLeanV37SourceError(
                "Complete response bytes do not authenticate their source seal"
            )
        extracted = source.seal_row.get("extracted_text")
        normalized = source.seal_row.get("normalized_text")
        if (
            type(extracted) is not dict
            or extracted.get("sha256") != content_sha256(source.extracted_primary)
            or extracted.get("length") != len(source.extracted_primary)
            or type(normalized) is not dict
            or normalized.get("sha256") != content_sha256(source.normalized_text)
            or normalized.get("length") != len(source.normalized_text)
        ):
            raise SecGemmaLeanV37SourceError(
                "Extracted/normalized bytes do not authenticate their source seal"
            )
        source_by_accession[source.accession_number] = source
    if set(source_by_accession) != set(reconciliation.accessions):
        raise SecGemmaLeanV37SourceError(
            "Complete source evidence set does not equal I"
        )
    target_rows = _seal_target_rows(reconciliation, source_by_accession)
    return _assemble_stage_seal_metadata(
        config=config,
        snapshot=snapshot,
        reconciliation=reconciliation,
        target_rows=target_rows,
        current_complete_accessions=current_complete_accessions,
        prior_seal=prior_seal,
        pre_master_upper_bound=pre_master_upper_bound,
    )


def _assemble_stage_seal_metadata(
    *,
    config: StageSourceConfig,
    snapshot: SubmissionsSnapshot,
    reconciliation: MasterReconciliation,
    target_rows: Sequence[Mapping[str, Any]],
    current_complete_accessions: Sequence[str],
    prior_seal: Mapping[str, Any] | None,
    pre_master_upper_bound: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    target_rows = [_canonical_snapshot(row) for row in target_rows]
    if len(target_rows) != len(reconciliation.targets):
        raise SecGemmaLeanV37SourceError(
            "Compact/full target-row count does not equal reconciliation I"
        )
    p_accessions = tuple(current_complete_accessions)
    expected_order = tuple(
        target.submissions.accession_number
        for target in reconciliation.targets
        if target.submissions.accession_number in set(p_accessions)
    )
    if p_accessions != expected_order or len(p_accessions) != len(set(p_accessions)):
        raise SecGemmaLeanV37SourceError("P is not in frozen filing-date/accession order")
    if len(p_accessions) > config.p_cap:
        raise SecGemmaLeanV37SourceError("Frozen P cap is exceeded")
    u_rows = sorted(
        (
            row
            for row in target_rows
            if row["stage_assignment"] in _STAGE_ORDER
            and GLOBAL_AVAILABILITY_START
            <= row["availability_session"]
            <= GLOBAL_AVAILABILITY_END
        ),
        key=lambda row: (row["availability_session"], row["accession_number"]),
    )
    d_rows = [row for row in u_rows if row["stage_assignment"] == config.stage]
    if not config.d_min <= len(d_rows) <= config.d_max:
        raise SecGemmaLeanV37SourceError("Frozen admitted D range/cap is violated")
    exact_count = sum(row["exact_acceptance"] is True for row in u_rows)
    acceptance_rate = exact_count / len(u_rows) if u_rows else 0.0
    if acceptance_rate < 0.95:
        raise SecGemmaLeanV37SourceError(
            "Cumulative U exact-acceptance coverage is below 95 percent"
        )
    request_formula = (
        1
        + snapshot.historical_reference_count
        + config.quarter_count
        + len(p_accessions)
    )
    if request_formula > config.maximum_successful_requests:
        raise SecGemmaLeanV37SourceError(
            "Frozen successful-request formula cap is exceeded"
        )
    carried = tuple(
        row["accession_number"]
        for row in target_rows
        if row["accession_number"] not in set(p_accessions)
    )
    if config.stage == "development":
        if prior_seal is not None or carried or p_accessions != tuple(
            row["accession_number"] for row in target_rows
        ):
            raise SecGemmaLeanV37SourceError(
                "Development requires no prior hash, empty carry, and P equal I"
            )
        prior_chain: list[dict[str, str]] = []
    else:
        if prior_seal is None:
            raise SecGemmaLeanV37SourceError(
                "Later stage source seal lacks adjacent prior authority"
            )
        validate_stage_source_seal(prior_seal)
        prior_targets = tuple(
            row["accession_number"] for row in prior_seal["target_rows"]
        )
        if carried != prior_targets:
            raise SecGemmaLeanV37SourceError(
                "Carried accession set is not exactly the prior target set"
            )
        prior_chain = _canonical_snapshot(prior_seal["prior_stage_chain"])
        prior_chain.append(
            {
                "stage": prior_seal["stage"],
                "stage_source_seal_sha256": prior_seal[
                    "stage_source_seal_sha256"
                ],
            }
        )
    expected_pre_master = {
        "candidate_accessions": [row["accession_number"] for row in target_rows],
        "source_document_accessions": list(p_accessions),
        "i_upper_bound": len(target_rows),
        "p_upper_bound": len(p_accessions),
        "submissions_snapshot_sha256": snapshot.snapshot_sha256,
        "prior_stage_source_seal_sha256": (
            prior_seal["stage_source_seal_sha256"]
            if prior_seal is not None
            else None
        ),
    }
    if type(pre_master_upper_bound) is not dict or any(
        pre_master_upper_bound.get(key) != expected
        for key, expected in expected_pre_master.items()
    ):
        raise SecGemmaLeanV37SourceError(
            "Pre-master upper bound does not equal the reconciled source plan"
        )
    final_coverage = (
        validate_completed_year_source_coverage(target_rows)
        if config.stage == "final"
        else None
    )
    body = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "stage": config.stage,
        "config": config.to_dict(),
        "prior_stage_source_seal_sha256": (
            prior_seal.get("stage_source_seal_sha256")
            if prior_seal is not None
            else None
        ),
        "prior_stage_chain": prior_chain,
        "session_calendar": _authoritative_calendar_receipt(EXPECTED_SESSIONS),
        "submissions_snapshot": snapshot.summary(),
        "pre_master_upper_bound": _canonical_snapshot(pre_master_upper_bound),
        "master_quarters": [
            item.summary() for item in reconciliation.quarter_evidence
        ],
        "master_reconciliation_sha256": reconciliation.reconciliation_sha256,
        "source_plan": {
            "new_complete_submission_accessions": list(p_accessions),
            "carried_accessions": list(carried),
            "ordering": "filing_date_then_accession",
            "subtraction": "I_minus_authenticated_adjacent_prior_v37_seals",
        },
        "counts": {
            "H": snapshot.historical_reference_count,
            "Q": config.quarter_count,
            "I": len(target_rows),
            "P": len(p_accessions),
            "U": len(u_rows),
            "D": len(d_rows),
            "exact_acceptance_U": exact_count,
            "successful_request_formula": request_formula,
        },
        "exact_acceptance_rate_U": acceptance_rate,
        "target_rows": target_rows,
        "u_accessions": [row["accession_number"] for row in u_rows],
        "d_accessions": [row["accession_number"] for row in d_rows],
        "completed_year_source_coverage": final_coverage,
        "wrapper_metadata_use": "verification_only",
        "science_input": "selected_embedded_TEXT_only",
    }
    seal_hash = _canonical_sha256(body)
    seal = _canonical_snapshot({**body, "stage_source_seal_sha256": seal_hash})
    seal_json = _canonical_json_bytes(seal)
    validate_stage_source_seal(seal)
    if prior_seal is not None:
        validate_cross_stage_frozen_prefix(prior_seal, seal)
    return seal, seal_json


def _validate_target_row_binding(row: Mapping[str, Any]) -> tuple[str, str]:
    """Validate one public row and bind every duplicate to its frozen prefix."""

    expected_target_keys = {
        "accession_number",
        "filing_date",
        "form",
        "acquisition_stage",
        "stage_assignment",
        "availability_session",
        "exact_acceptance",
        "submissions_layout_provenance",
        "complete_response",
        "primary_selection",
        "extracted_text",
        "normalized_text",
        "frozen_prefix",
        "frozen_prefix_sha256",
    }
    if type(row) is not dict or set(row) != expected_target_keys:
        raise SecGemmaLeanV37SourceError("Stage source target row is invalid")
    accession = row.get("accession_number")
    filing_date = row.get("filing_date")
    prefix = row.get("frozen_prefix")
    prefix_hash = row.get("frozen_prefix_sha256")
    if (
        type(accession) is not str
        or _ACCESSION_RE.fullmatch(accession) is None
        or type(filing_date) is not str
        or type(prefix) is not dict
        or type(prefix_hash) is not str
        or _BARE_SHA256_RE.fullmatch(prefix_hash) is None
        or _canonical_sha256(prefix) != prefix_hash
    ):
        raise SecGemmaLeanV37SourceError("Stage source target row is not bound")
    _canonical_date(filing_date, location="stage target filing date")
    expected_prefix_keys = {
        "accession_number",
        "submissions_semantic_identity",
        "submissions_semantic_identity_sha256",
        "master_identity",
        "header_identity",
        "acceptance",
        "filing_date_change",
        "availability_session",
        "availability_not_before_date",
        "immutable_stage_assignment",
        "complete_response",
        "primary_selection",
        "extracted_text",
        "normalized_text",
    }
    if set(prefix) != expected_prefix_keys:
        raise SecGemmaLeanV37SourceError("Frozen-prefix schema is not exact")
    semantic = prefix.get("submissions_semantic_identity")
    semantic_hash = prefix.get("submissions_semantic_identity_sha256")
    if (
        type(semantic) is not dict
        or set(semantic)
        != {
            "accession_number",
            "subject_cik",
            "typed_values",
            "date_of_filing_date_change_present",
        }
        or type(semantic_hash) is not str
        or _BARE_SHA256_RE.fullmatch(semantic_hash) is None
        or _canonical_sha256(semantic) != semantic_hash
        or semantic.get("accession_number") != accession
        or semantic.get("subject_cik") != AAPL_CIK
        or type(semantic.get("typed_values")) is not dict
        or type(semantic.get("date_of_filing_date_change_present")) is not bool
    ):
        raise SecGemmaLeanV37SourceError(
            "Submissions semantic identity is not exactly bound"
        )
    typed = semantic["typed_values"]
    mandatory_typed_text = {
        "accessionNumber",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "items",
        "filingDate",
        "reportDate",
    }
    if (
        not mandatory_typed_text <= set(typed)
        or any(type(key) is not str for key in typed)
        or any(type(typed[key]) is not str for key in mandatory_typed_text)
        or (
            "dateOfFilingDateChange" in typed
            and type(typed["dateOfFilingDateChange"]) is not str
        )
    ):
        raise SecGemmaLeanV37SourceError(
            "Submissions typed-values schema/types are not exact"
        )
    if "isXBRL" in typed:
        _flag(typed["isXBRL"], location="sealed Submissions isXBRL")
    if semantic["date_of_filing_date_change_present"] != (
        "dateOfFilingDateChange" in typed
    ):
        raise SecGemmaLeanV37SourceError(
            "Submissions change-column presence marker is not exact"
        )
    _canonical_date(
        typed["reportDate"],
        location="sealed Submissions reportDate",
        allow_empty=True,
    )
    master = prefix.get("master_identity")
    if (
        set(master) if type(master) is dict else None
    ) != {"accession_number", "cik", "form", "filing_date", "filename"}:
        raise SecGemmaLeanV37SourceError("Master identity schema is not exact")
    form = row.get("form")
    header = prefix.get("header_identity")
    if type(header) is not dict or set(header) != {
        "accession_number",
        "subject_cik",
        "form",
        "raw_form",
        "filing_date",
        "acceptance_datetime",
        "date_of_filing_date_change",
    }:
        raise SecGemmaLeanV37SourceError("SGML header identity schema is not exact")
    if (
        form not in {"10-K", "10-Q"}
        or prefix.get("accession_number") != accession
        or typed.get("accessionNumber") != accession
        or typed.get("filingDate") != filing_date
        or typed.get("form") != form
        or master.get("accession_number") != accession
        or master.get("cik") != AAPL_CIK
        or master.get("form") != form
        or master.get("filing_date") != filing_date
        or master.get("filename")
        != f"edgar/data/320193/{accession}.txt"
        or header.get("accession_number") != accession
        or header.get("subject_cik") != AAPL_CIK
        or header.get("form") != form
        or header.get("raw_form") != form
        or header.get("filing_date") != filing_date
    ):
        raise SecGemmaLeanV37SourceError(
            "Target, Submissions, and master duplicate identities differ"
        )
    layout = row.get("submissions_layout_provenance")
    if (
        type(layout) is not dict
        or set(layout)
        != {
            "source_name",
            "source_content_sha256",
            "source_position",
            "source_row_order_sha256",
        }
        or type(layout.get("source_name")) is not str
        or type(layout.get("source_position")) is not int
        or layout["source_position"] < 0
        or type(layout.get("source_content_sha256")) is not str
        or _SHA256_RE.fullmatch(layout["source_content_sha256"]) is None
        or type(layout.get("source_row_order_sha256")) is not str
        or _BARE_SHA256_RE.fullmatch(layout["source_row_order_sha256"]) is None
    ):
        raise SecGemmaLeanV37SourceError(
            "Submissions layout provenance is malformed"
        )
    acceptance = prefix.get("acceptance")
    if type(acceptance) is not dict or set(acceptance) != {
        "submissions_source",
        "header_source",
        "normalized_et",
        "submissions_missing",
        "header_missing",
        "exact",
    }:
        raise SecGemmaLeanV37SourceError("Acceptance evidence schema is not exact")
    for name in ("submissions_missing", "header_missing", "exact"):
        if type(acceptance[name]) is not bool:
            raise SecGemmaLeanV37SourceError("Acceptance missingness is not exact")
    normalized_acceptance = acceptance["normalized_et"]
    if normalized_acceptance is not None and (
        type(normalized_acceptance) is not str
        or re.fullmatch(r"[0-9]{14}", normalized_acceptance) is None
    ):
        raise SecGemmaLeanV37SourceError("Normalized acceptance is malformed")
    if (
        acceptance["submissions_missing"]
        != (acceptance["submissions_source"] is None)
        or acceptance["header_missing"]
        != (acceptance["header_source"] is None)
        or acceptance["exact"] != (normalized_acceptance is not None)
        or row.get("exact_acceptance") != acceptance["exact"]
        or acceptance["submissions_source"]
        != (typed.get("acceptanceDateTime") or None)
        or acceptance["header_source"] != header["acceptance_datetime"]
    ):
        raise SecGemmaLeanV37SourceError(
            "Acceptance values and duplicate flags do not reconcile"
        )
    normalized_sources: set[str] = set()
    for raw_acceptance in (
        acceptance["submissions_source"],
        acceptance["header_source"],
    ):
        if raw_acceptance is not None:
            if type(raw_acceptance) is not str:
                raise SecGemmaLeanV37SourceError(
                    "Acceptance source did not remain text"
                )
            try:
                normalized_sources.add(
                    parse_submissions_acceptance_datetime(raw_acceptance).strftime(
                        "%Y%m%d%H%M%S"
                    )
                )
            except SecPointInTimeError:
                raise SecGemmaLeanV37SourceError(
                    "Acceptance source is not lexically exact"
                ) from None
    if (
        len(normalized_sources) > 1
        or normalized_acceptance
        != (next(iter(normalized_sources)) if normalized_sources else None)
    ):
        raise SecGemmaLeanV37SourceError(
            "Normalized acceptance is not derived from exact sources"
        )
    change = prefix.get("filing_date_change")
    if type(change) is not dict or set(change) != {
        "submissions_source",
        "submissions_column_present",
        "header_source",
        "normalized",
        "submissions_missing",
        "header_missing",
    }:
        raise SecGemmaLeanV37SourceError(
            "Filing-date-change evidence schema is not exact"
        )
    for name in (
        "submissions_column_present",
        "submissions_missing",
        "header_missing",
    ):
        if type(change[name]) is not bool:
            raise SecGemmaLeanV37SourceError(
                "Filing-date-change missingness is not exact"
            )
    if (
        change["submissions_missing"] != (change["submissions_source"] is None)
        or change["header_missing"] != (change["header_source"] is None)
        or change["submissions_column_present"]
        != semantic["date_of_filing_date_change_present"]
        or change["submissions_source"]
        != (
            typed.get("dateOfFilingDateChange") or None
            if semantic["date_of_filing_date_change_present"]
            else None
        )
        or change["header_source"] != header["date_of_filing_date_change"]
    ):
        raise SecGemmaLeanV37SourceError(
            "Filing-date-change duplicate flags do not reconcile"
        )
    for name in ("submissions_source", "header_source", "normalized"):
        if change[name] is not None:
            _canonical_date(change[name], location=f"filing-date-change {name}")
    if change["normalized"] != (
        change["submissions_source"] or change["header_source"]
    ):
        raise SecGemmaLeanV37SourceError(
            "Filing-date-change normalized value is not source-derived"
        )
    response = row.get("complete_response")
    extracted = row.get("extracted_text")
    normalized = row.get("normalized_text")
    selection = row.get("primary_selection")
    if type(response) is not dict or set(response) != {"url", "sha256", "length"}:
        raise SecGemmaLeanV37SourceError("Complete-response provenance is not exact")
    if response.get("url") != (
        "https://www.sec.gov/Archives/"
        f"edgar/data/320193/{accession}.txt"
    ):
        raise SecGemmaLeanV37SourceError("Complete-response URL is not exact")
    if type(extracted) is not dict or set(extracted) != {
        "start_byte",
        "end_byte",
        "length",
        "sha256",
        "encoding",
    }:
        raise SecGemmaLeanV37SourceError("Extracted-TEXT provenance is not exact")
    if type(normalized) is not dict or set(normalized) != {
        "length",
        "character_count",
        "sha256",
        "normalizer",
    }:
        raise SecGemmaLeanV37SourceError("Normalized-text provenance is not exact")
    for provenance in (response, extracted, normalized):
        if (
            type(provenance.get("sha256")) is not str
            or _SHA256_RE.fullmatch(provenance["sha256"]) is None
            or type(provenance.get("length")) is not int
            or provenance["length"] < 0
        ):
            raise SecGemmaLeanV37SourceError(
                "Stage source byte provenance is malformed"
            )
    if (
        type(extracted["start_byte"]) is not int
        or type(extracted["end_byte"]) is not int
        or extracted["start_byte"] < 0
        or extracted["end_byte"] - extracted["start_byte"] != extracted["length"]
        or extracted["end_byte"] > response["length"]
        or extracted["encoding"] != "latin-1 exact one-to-one response slice"
        or type(normalized["character_count"]) is not int
        or normalized["character_count"] < 0
        or normalized["normalizer"]
        != "inherited normalize_filing_text byte-identical rules"
    ):
        raise SecGemmaLeanV37SourceError(
            "Extracted/normalized byte geometry is invalid"
        )
    expected_selection_keys = {
        "selected_document_ordinal",
        "sequence",
        "type",
        "raw_type",
        "sec_filename",
        "document_identity",
        "sgml_document_identity",
        "submissions_filename",
        "sgml_filename",
        "submissions_filename_missing",
        "sgml_filename_missing",
    }
    if type(selection) is not dict or set(selection) != expected_selection_keys:
        raise SecGemmaLeanV37SourceError("Primary-selection schema is not exact")
    if (
        selection["sequence"] != 1
        or selection["type"] != form
        or selection["raw_type"] != form
        or type(selection["selected_document_ordinal"]) is not int
        or selection["selected_document_ordinal"] < 1
        or type(selection["submissions_filename_missing"]) is not bool
        or type(selection["sgml_filename_missing"]) is not bool
        or selection["submissions_filename_missing"]
        != (selection["submissions_filename"] is None)
        or selection["sgml_filename_missing"]
        != (selection["sgml_filename"] is None)
    ):
        raise SecGemmaLeanV37SourceError(
            "Primary-selection exact form or missingness is invalid"
        )
    typed_primary = typed["primaryDocument"] or None
    if typed_primary is not None:
        _safe_primary_name(
            typed_primary,
            location="sealed Submissions primaryDocument",
            allow_empty=False,
        )
    sgml_filename = selection["sgml_filename"]
    if sgml_filename is not None:
        if type(sgml_filename) is not str:
            raise SecGemmaLeanV37SourceError(
                "Sealed SGML filename did not remain text"
            )
        _safe_primary_name(
            sgml_filename,
            location="sealed SGML filename",
            allow_empty=False,
        )
    expected_sec_filename = typed_primary or sgml_filename
    expected_sgml_identity = (
        sgml_filename
        if sgml_filename is not None
        else "legacy-sequence-1-no-filename"
    )
    expected_document_identity = (
        expected_sec_filename or "legacy-sequence-1-no-filename"
    )
    if (
        selection["submissions_filename"] != typed_primary
        or (
            typed_primary is not None
            and sgml_filename is not None
            and typed_primary != sgml_filename
        )
        or selection["sec_filename"] != expected_sec_filename
        or selection["sgml_document_identity"] != expected_sgml_identity
        or selection["document_identity"] != expected_document_identity
    ):
        raise SecGemmaLeanV37SourceError(
            "Primary filename/document identities do not reconcile exactly"
        )
    byte_hashes = {
        response["sha256"],
        extracted["sha256"],
        normalized["sha256"],
    }
    if len(byte_hashes) != 3:
        raise SecGemmaLeanV37SourceError(
            "Complete, extracted, and normalized byte identities are not distinct"
        )
    if (
        row["complete_response"] != prefix["complete_response"]
        or row["primary_selection"] != prefix["primary_selection"]
        or row["extracted_text"] != prefix["extracted_text"]
        or row["normalized_text"] != prefix["normalized_text"]
    ):
        raise SecGemmaLeanV37SourceError(
            "Duplicated byte/selection provenance differs from frozen prefix"
        )
    boundary = prefix.get("availability_not_before_date")
    _canonical_date(boundary, location="availability not-before date")
    evidence_dates = [filing_date]
    if normalized_acceptance is not None:
        evidence_dates.append(
            datetime.strptime(normalized_acceptance[:8], "%Y%m%d").date().isoformat()
        )
    if change["normalized"] is not None:
        evidence_dates.append(change["normalized"])
    if boundary != max(evidence_dates):
        raise SecGemmaLeanV37SourceError(
            "Availability boundary is not reconstructed from source dates"
        )
    availability = _conservative_availability_session(boundary)
    assignment = _stage_assignment(availability)
    if (
        prefix.get("availability_session") != availability
        or prefix.get("immutable_stage_assignment") != assignment
        or row.get("availability_session") != availability
        or row.get("stage_assignment") != assignment
    ):
        raise SecGemmaLeanV37SourceError(
            "Availability/session/stage duplicates are not authoritative"
        )
    return accession, filing_date


def validate_stage_source_seal(seal: Mapping[str, Any]) -> None:
    """Validate deterministic structure and internal hashes of a public seal."""

    value = _validate_seal_hash(seal)
    expected_top_level_keys = {
        "schema_version",
        "stage",
        "config",
        "prior_stage_source_seal_sha256",
        "prior_stage_chain",
        "session_calendar",
        "submissions_snapshot",
        "pre_master_upper_bound",
        "master_quarters",
        "master_reconciliation_sha256",
        "source_plan",
        "counts",
        "exact_acceptance_rate_U",
        "target_rows",
        "u_accessions",
        "d_accessions",
        "completed_year_source_coverage",
        "wrapper_metadata_use",
        "science_input",
        "stage_source_seal_sha256",
    }
    if set(value) != expected_top_level_keys:
        raise SecGemmaLeanV37SourceError("Stage source seal schema is not exact")
    if value.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise SecGemmaLeanV37SourceError("Stage source seal schema is invalid")
    stage = value.get("stage")
    config = _stage_config(stage)
    if value.get("config") != config.to_dict():
        raise SecGemmaLeanV37SourceError("Stage source seal config is not frozen")
    if value.get("session_calendar") != _authoritative_calendar_receipt(
        EXPECTED_SESSIONS
    ):
        raise SecGemmaLeanV37SourceError(
            "Stage source seal calendar identity is not authoritative"
        )
    prior_hash = value.get("prior_stage_source_seal_sha256")
    prior_chain = value.get("prior_stage_chain")
    expected_prior_stages = list(_STAGE_ORDER[: _STAGE_ORDER.index(stage)])
    if (
        type(prior_chain) is not list
        or len(prior_chain) != len(expected_prior_stages)
        or any(
            type(entry) is not dict
            or set(entry) != {"stage", "stage_source_seal_sha256"}
            or entry.get("stage") != expected_stage
            or type(entry.get("stage_source_seal_sha256")) is not str
            or _BARE_SHA256_RE.fullmatch(
                entry["stage_source_seal_sha256"]
            )
            is None
            for entry, expected_stage in zip(prior_chain, expected_prior_stages)
        )
        or (
            stage == "development"
            and prior_hash is not None
        )
        or (
            stage != "development"
            and (
                type(prior_hash) is not str
                or _BARE_SHA256_RE.fullmatch(prior_hash) is None
                or prior_hash
                != prior_chain[-1]["stage_source_seal_sha256"]
            )
        )
    ):
        raise SecGemmaLeanV37SourceError(
            "Stage source prior chain/hash authority is invalid"
        )
    targets = value.get("target_rows")
    if type(targets) is not list:
        raise SecGemmaLeanV37SourceError("Stage source target rows are invalid")
    accessions: list[str] = []
    target_order: list[tuple[str, str]] = []
    for row in targets:
        accession, filing_date = _validate_target_row_binding(row)
        accessions.append(accession)
        target_order.append((filing_date, accession))
    if len(accessions) != len(set(accessions)):
        raise SecGemmaLeanV37SourceError("Stage source target rows are duplicated")
    if target_order != sorted(target_order):
        raise SecGemmaLeanV37SourceError(
            "Stage source target rows are not in filing-date/accession order"
        )
    counts = value.get("counts")
    source_plan = value.get("source_plan")
    if type(counts) is not dict or type(source_plan) is not dict:
        raise SecGemmaLeanV37SourceError("Stage source seal counts/plan are invalid")
    if set(source_plan) != {
        "new_complete_submission_accessions",
        "carried_accessions",
        "ordering",
        "subtraction",
    }:
        raise SecGemmaLeanV37SourceError("Stage source plan schema is not exact")
    expected_count_keys = {
        "H",
        "Q",
        "I",
        "P",
        "U",
        "D",
        "exact_acceptance_U",
        "successful_request_formula",
    }
    if set(counts) != expected_count_keys or any(
        type(counts[name]) is not int or counts[name] < 0
        for name in expected_count_keys
    ):
        raise SecGemmaLeanV37SourceError("Stage source count schema is invalid")
    p = source_plan.get("new_complete_submission_accessions")
    carried = source_plan.get("carried_accessions")
    if (
        type(p) is not list
        or type(carried) is not list
        or any(
            type(accession) is not str or _ACCESSION_RE.fullmatch(accession) is None
            for accession in [*p, *carried]
        )
        or counts.get("I") != len(accessions)
        or counts.get("P") != len(p)
        or set(p) | set(carried) != set(accessions)
        or set(p) & set(carried)
    ):
        raise SecGemmaLeanV37SourceError("Stage source plan/counts do not reconcile")
    p_set = set(p)
    carried_set = set(carried)
    if (
        p != [accession for accession in accessions if accession in p_set]
        or carried
        != [accession for accession in accessions if accession in carried_set]
        or source_plan.get("ordering") != "filing_date_then_accession"
        or source_plan.get("subtraction")
        != "I_minus_authenticated_adjacent_prior_v37_seals"
    ):
        raise SecGemmaLeanV37SourceError("Stage source plan order is invalid")
    if stage == "development" and (
        prior_hash is not None
        or carried
        or p != accessions
    ):
        raise SecGemmaLeanV37SourceError(
            "Development source plan must have P equal I and no prior authority"
        )
    if stage != "development" and not carried:
        raise SecGemmaLeanV37SourceError(
            "Later stage source plan must carry the adjacent prior target set"
        )
    current_index = _STAGE_ORDER.index(stage)
    for row in targets:
        acquisition_stage = row.get("acquisition_stage")
        if (
            acquisition_stage not in _STAGE_ORDER[: current_index + 1]
            or (
                row["accession_number"] in p_set
                and acquisition_stage != stage
            )
            or (
                row["accession_number"] in carried_set
                and acquisition_stage == stage
            )
        ):
            raise SecGemmaLeanV37SourceError(
                "Stage source acquisition authority is invalid"
            )
    submissions_summary = value.get("submissions_snapshot")
    masters = value.get("master_quarters")
    pre_master = value.get("pre_master_upper_bound")
    if (
        type(submissions_summary) is not dict
        or type(submissions_summary.get("historical_references")) is not list
        or type(masters) is not list
        or counts.get("H") != len(submissions_summary["historical_references"])
        or counts.get("Q") != len(masters)
        or counts.get("Q") != config.quarter_count
        or counts.get("H") > MAX_HISTORICAL_REFERENCES
        or counts.get("I") > config.i_cap
        or counts.get("P") > config.p_cap
        or counts.get("successful_request_formula")
        != 1 + counts["H"] + counts["Q"] + counts["P"]
        or counts["successful_request_formula"]
        > config.maximum_successful_requests
    ):
        raise SecGemmaLeanV37SourceError(
            "Stage source H/Q/I/P/request counts do not reconcile"
        )
    expected_pre_master_keys = {
        "schema_version",
        "stage",
        "submissions_snapshot_sha256",
        "candidate_ordering",
        "candidate_accessions",
        "candidate_accessions_sha256",
        "i_upper_bound",
        "source_document_accessions",
        "source_document_accessions_sha256",
        "p_upper_bound",
        "prior_stage_source_seal_sha256",
        "authority",
        "pre_master_upper_bound_sha256",
    }
    if type(pre_master) is not dict or set(pre_master) != expected_pre_master_keys:
        raise SecGemmaLeanV37SourceError(
            "Pre-master upper-bound receipt schema is not exact"
        )
    pre_body = dict(pre_master)
    pre_hash = pre_body.pop("pre_master_upper_bound_sha256")
    if (
        type(pre_hash) is not str
        or _BARE_SHA256_RE.fullmatch(pre_hash) is None
        or _canonical_sha256(pre_body) != pre_hash
        or pre_master["schema_version"] != PRE_MASTER_UPPER_BOUND_SCHEMA_VERSION
        or pre_master["stage"] != stage
        or pre_master["submissions_snapshot_sha256"]
        != submissions_summary.get("snapshot_sha256")
        or pre_master["candidate_ordering"] != "filing_date_then_accession"
        or pre_master["candidate_accessions"] != accessions
        or pre_master["candidate_accessions_sha256"]
        != _canonical_sha256(tuple(accessions))
        or pre_master["i_upper_bound"] != len(accessions)
        or pre_master["source_document_accessions"] != p
        or pre_master["source_document_accessions_sha256"]
        != _canonical_sha256(tuple(p))
        or pre_master["p_upper_bound"] != len(p)
        or pre_master["prior_stage_source_seal_sha256"] != prior_hash
        or pre_master["authority"]
        != "reject_only_submissions_bound_not_master_membership"
    ):
        raise SecGemmaLeanV37SourceError(
            "Pre-master upper-bound receipt does not bind the source plan"
        )
    d_accessions = value.get("d_accessions")
    u_accessions = value.get("u_accessions")
    expected_u_rows = sorted(
        (
            row
            for row in targets
            if row["stage_assignment"] in _STAGE_ORDER
            and type(row["availability_session"]) is str
            and GLOBAL_AVAILABILITY_START
            <= row["availability_session"]
            <= GLOBAL_AVAILABILITY_END
        ),
        key=lambda row: (row["availability_session"], row["accession_number"]),
    )
    expected_u_accessions = [row["accession_number"] for row in expected_u_rows]
    expected_d_accessions = [
        row["accession_number"]
        for row in expected_u_rows
        if row["stage_assignment"] == stage
    ]
    if (
        type(d_accessions) is not list
        or type(u_accessions) is not list
        or counts.get("D") != len(d_accessions)
        or counts.get("U") != len(u_accessions)
        or u_accessions != expected_u_accessions
        or d_accessions != expected_d_accessions
        or not config.d_min <= len(d_accessions) <= config.d_max
    ):
        raise SecGemmaLeanV37SourceError("Stage source U/D counts do not reconcile")
    exact_count = sum(
        row.get("exact_acceptance") is True
        for row in targets
        if row["accession_number"] in set(u_accessions)
    )
    expected_rate = exact_count / len(u_accessions) if u_accessions else 0.0
    if (
        counts.get("exact_acceptance_U") != exact_count
        or value.get("exact_acceptance_rate_U") != expected_rate
        or expected_rate < 0.95
        or any(
            row.get("stage_assignment") != stage
            for row in targets
            if row["accession_number"] in set(d_accessions)
        )
    ):
        raise SecGemmaLeanV37SourceError(
            "Stage source acceptance coverage or D assignment is invalid"
        )
    coverage = value.get("completed_year_source_coverage")
    if stage == "final":
        expected_coverage = validate_completed_year_source_coverage(targets)
        if coverage != expected_coverage:
            raise SecGemmaLeanV37SourceError(
                "Final completed-year source gate receipt is not exact"
            )
    elif coverage is not None:
        raise SecGemmaLeanV37SourceError(
            "Completed-year source receipt may appear only in the final stage"
        )


def _parse_master_set(
    config: StageSourceConfig,
    master_gzip_payloads: Mapping[tuple[int, int], bytes],
) -> tuple[MasterQuarterEvidence, ...]:
    if type(master_gzip_payloads) is not dict:
        raise SecGemmaLeanV37SourceError("master.gz payload set must be an exact dictionary")
    expected = {role.key for role in quarter_roles(config)}
    if set(master_gzip_payloads) != expected:
        raise SecGemmaLeanV37SourceError("master.gz payload set is not exact")
    return tuple(
        parse_strict_master_gzip(
            master_gzip_payloads[role.key],
            year=role.year,
            quarter=role.quarter,
        )
        for role in quarter_roles(config)
    )


def _build_stage_source_bundle_core(
    *,
    config: StageSourceConfig,
    snapshot: SubmissionsSnapshot,
    reconciliation: MasterReconciliation,
    session_dates: Sequence[str],
    current_complete_payloads: Mapping[str, bytes],
    carried_payloads: Mapping[str, tuple[bytes, str]],
    prior_bundle: StageSourceBundle | None,
    pre_master_upper_bound: Mapping[str, Any],
) -> StageSourceBundle:
    sessions = _canonical_sessions(session_dates)
    plan = derive_complete_submission_plan(
        reconciliation,
        prior_bundle=prior_bundle,
    )
    prior_seal = (
        _canonical_snapshot(prior_bundle.seal)
        if prior_bundle is not None
        else None
    )
    plan_accessions = tuple(item.submissions.accession_number for item in plan)
    if type(current_complete_payloads) is not dict or set(current_complete_payloads) != set(
        plan_accessions
    ):
        raise SecGemmaLeanV37SourceError(
            "Current complete-submission payload set does not equal P"
        )
    expected_carried = set(reconciliation.accessions) - set(plan_accessions)
    if type(carried_payloads) is not dict or set(carried_payloads) != expected_carried:
        raise SecGemmaLeanV37SourceError(
            "Carried complete-submission payload set is not exact"
        )
    targets_by_accession = {
        item.submissions.accession_number: item for item in reconciliation.targets
    }
    sources: list[CompleteSourceEvidence] = []
    for target in reconciliation.targets:
        accession = target.submissions.accession_number
        if accession in current_complete_payloads:
            payload = current_complete_payloads[accession]
            acquisition_stage = config.stage
        else:
            raw_carried = carried_payloads[accession]
            if (
                type(raw_carried) is not tuple
                or len(raw_carried) != 2
                or type(raw_carried[0]) is not bytes
                or type(raw_carried[1]) is not str
            ):
                raise SecGemmaLeanV37SourceError("Carried source tuple is invalid")
            payload, acquisition_stage = raw_carried
        sources.append(
            reconcile_complete_submission(
                targets_by_accession[accession],
                payload,
                session_dates=sessions,
                acquisition_stage=acquisition_stage,
            )
        )
    seal, seal_json = _assemble_stage_seal(
        config=config,
        snapshot=snapshot,
        reconciliation=reconciliation,
        complete_sources=sources,
        current_complete_accessions=plan_accessions,
        prior_seal=prior_seal,
        pre_master_upper_bound=pre_master_upper_bound,
    )
    return StageSourceBundle(
        config=config,
        submissions_snapshot=snapshot,
        master_reconciliation=reconciliation,
        complete_sources=tuple(sources),
        session_dates=sessions,
        current_complete_accessions=plan_accessions,
        prior_seal=(
            _canonical_snapshot(prior_seal) if prior_seal is not None else None
        ),
        prior_bundle=prior_bundle,
        seal=seal,
        seal_json=seal_json,
    )


def build_stage_source_seal(
    *,
    stage: str | StageSourceConfig,
    main_submissions_payload: bytes,
    historical_submissions_payloads: Mapping[str, bytes],
    master_gzip_payloads: Mapping[tuple[int, int], bytes],
    complete_submission_payloads: Mapping[str, bytes],
    session_dates: Sequence[str],
    prior_bundle: StageSourceBundle | None = None,
) -> StageSourceBundle:
    """Build one cumulative source seal entirely from caller-supplied bytes."""

    config = _stage_config(stage)
    if config.stage == "development":
        if prior_bundle is not None:
            raise SecGemmaLeanV37SourceError("Development cannot carry a prior bundle")
        carried: dict[str, tuple[bytes, str]] = {}
    else:
        if not isinstance(prior_bundle, StageSourceBundle):
            raise SecGemmaLeanV37SourceError(
                "A later source stage requires its exact adjacent prior bundle"
            )
        carried = {
            source.accession_number: (
                source.complete_response,
                source.acquisition_stage,
            )
            for source in prior_bundle.complete_sources
        }
    snapshot = parse_submissions_snapshot(
        main_submissions_payload,
        dict(historical_submissions_payloads),
    )
    # This check deliberately occurs before any master body is parsed.  It is
    # rejection-only; exact master equality remains mandatory below.
    pre_master_upper_bound = build_pre_master_submissions_upper_bound(
        snapshot,
        config,
        prior_bundle=prior_bundle,
    )
    quarters = _parse_master_set(config, dict(master_gzip_payloads))
    reconciliation = reconcile_master_exact_set(snapshot, quarters, config)
    return _build_stage_source_bundle_core(
        config=config,
        snapshot=snapshot,
        reconciliation=reconciliation,
        session_dates=session_dates,
        current_complete_payloads=dict(complete_submission_payloads),
        carried_payloads=carried,
        prior_bundle=prior_bundle,
        pre_master_upper_bound=pre_master_upper_bound,
    )


def detached_replay_stage_source(bundle: StageSourceBundle) -> Mapping[str, Any]:
    """Rebuild a stage from its sealed raw bytes and return an exact receipt."""

    if not isinstance(bundle, StageSourceBundle):
        raise TypeError("bundle must be a StageSourceBundle")
    validate_stage_source_seal(bundle.seal)
    expected_prior_seal = (
        _canonical_snapshot(bundle.prior_bundle.seal)
        if bundle.prior_bundle is not None
        else None
    )
    if bundle.prior_seal != expected_prior_seal:
        raise SecGemmaLeanV37SourceError(
            "Retained prior bundle does not match the deep prior metadata snapshot"
        )
    snapshot = parse_submissions_snapshot(
        bundle.submissions_snapshot.main_payload,
        dict(bundle.submissions_snapshot.historical_payloads),
    )
    pre_master_upper_bound = build_pre_master_submissions_upper_bound(
        snapshot,
        bundle.config,
        prior_bundle=bundle.prior_bundle,
    )
    quarters = _parse_master_set(
        bundle.config,
        {
            evidence.role.key: evidence.compressed_payload
            for evidence in bundle.master_reconciliation.quarter_evidence
        },
    )
    reconciliation = reconcile_master_exact_set(snapshot, quarters, bundle.config)
    current = set(bundle.current_complete_accessions)
    current_payloads = {
        source.accession_number: source.complete_response
        for source in bundle.complete_sources
        if source.accession_number in current
    }
    carried_payloads = {
        source.accession_number: (
            source.complete_response,
            source.acquisition_stage,
        )
        for source in bundle.complete_sources
        if source.accession_number not in current
    }
    replay = _build_stage_source_bundle_core(
        config=bundle.config,
        snapshot=snapshot,
        reconciliation=reconciliation,
        session_dates=bundle.session_dates,
        current_complete_payloads=current_payloads,
        carried_payloads=carried_payloads,
        prior_bundle=bundle.prior_bundle,
        pre_master_upper_bound=pre_master_upper_bound,
    )
    if replay.seal_json != bundle.seal_json or replay.seal != bundle.seal:
        raise SecGemmaLeanV37SourceError(
            "Detached source replay does not reproduce the exact stage seal"
        )
    body = {
        "schema_version": DETACHED_REPLAY_SCHEMA_VERSION,
        "stage": bundle.config.stage,
        "stage_source_seal_sha256": bundle.stage_source_seal_sha256,
        "replayed_stage_source_seal_sha256": replay.stage_source_seal_sha256,
        "exact_seal_json_match": True,
        "fresh_network_provenance_claimed": False,
        "replay_scope": "sealed_exact_source_response_bytes_only",
        "orchestrator_boundary": (
            "acquisition_manifest_must_authenticate_journal_transport_and_blob_pins"
        ),
    }
    return {**body, "detached_replay_sha256": _canonical_sha256(body)}


def build_legacy_science_projection(
    bundle: StageSourceBundle,
) -> LegacyScienceProjection:
    """Project current D into the old raw-primary shape without conflating bytes."""

    if not isinstance(bundle, StageSourceBundle):
        raise TypeError("bundle must be a StageSourceBundle")
    # A projection is scientific input, so a merely self-consistent public
    # seal is insufficient.  Authenticate all retained raw bundles first.
    detached_replay_stage_source(bundle)
    sources = bundle.sources_by_accession
    documents: list[LegacyScienceDocument] = []
    rows_by_accession = {
        row["accession_number"]: row for row in bundle.seal["target_rows"]
    }
    science_rows = sorted(
        (rows_by_accession[accession] for accession in bundle.seal["d_accessions"]),
        key=lambda row: (row["availability_session"], row["accession_number"]),
    )
    for row in science_rows:
        accession = row["accession_number"]
        source = sources[accession]
        documents.append(
            LegacyScienceDocument(
                accession_number=accession,
                official_complete_submission_url=row["complete_response"]["url"],
                raw_primary_document=source.extracted_primary,
                normalized_text=source.normalized_text,
                primary_document_sha256=row["extracted_text"]["sha256"],
                normalized_text_sha256=row["normalized_text"]["sha256"],
                complete_response_sha256=row["complete_response"]["sha256"],
                selected_document_identity=row["primary_selection"][
                    "document_identity"
                ],
            )
        )
    rows = [
        {
            "accession_number": document.accession_number,
            "official_complete_submission_url": (
                document.official_complete_submission_url
            ),
            "raw_primary_document_sha256": document.primary_document_sha256,
            "normalized_text_sha256": document.normalized_text_sha256,
            "complete_response_sha256": document.complete_response_sha256,
            "selected_document_identity": document.selected_document_identity,
            "raw_primary_document_semantics": "selected_embedded_TEXT_bytes",
        }
        for document in documents
    ]
    body = {
        "schema_version": LEGACY_SCIENCE_PROJECTION_SCHEMA_VERSION,
        "stage": bundle.config.stage,
        "stage_source_seal_sha256": bundle.stage_source_seal_sha256,
        "document_count": len(documents),
        "documents": rows,
        "ordering": "availability_session_then_accession",
        "compatibility_boundary": (
            "raw_primary_document is extracted TEXT, never the complete response"
        ),
    }
    manifest = {**body, "projection_sha256": _canonical_sha256(body)}
    return LegacyScienceProjection(
        stage=bundle.config.stage,
        documents=tuple(documents),
        manifest=manifest,
        manifest_json=_canonical_json_bytes(manifest),
    )


# ---------------------------------------------------------------------------
# Compact one-role-at-a-time source pipeline
# ---------------------------------------------------------------------------


def _compact_evidence(body: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _canonical_snapshot(body)
    return _canonical_snapshot(
        {**snapshot, "source_evidence_sha256": _canonical_sha256(snapshot)}
    )


def _validate_compact_evidence(
    evidence: Mapping[str, Any],
    *,
    schema_version: str,
) -> dict[str, Any]:
    if type(evidence) is not dict:
        raise SecGemmaLeanV37SourceError("Compact source evidence is not an object")
    value = _canonical_snapshot(evidence)
    supplied = value.pop("source_evidence_sha256", None)
    if (
        type(supplied) is not str
        or _BARE_SHA256_RE.fullmatch(supplied) is None
        or _canonical_sha256(value) != supplied
        or value.get("schema_version") != schema_version
    ):
        raise SecGemmaLeanV37SourceError(
            "Compact source evidence hash/schema does not validate"
        )
    return _canonical_snapshot(evidence)


def _output_evidence(value: RoleParseOutput | PhaseOutput | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, (RoleParseOutput, PhaseOutput)):
        return _canonical_snapshot(value.evidence)
    if type(value) is dict:
        return _canonical_snapshot(value)
    raise SecGemmaLeanV37SourceError("Compact evidence input type is invalid")


def _submission_row_compact(row: SubmissionRow) -> dict[str, Any]:
    return _canonical_snapshot(
        {
            "accession_number": row.accession_number,
            "subject_cik": row.subject_cik,
            "acceptance_datetime": row.acceptance_datetime,
            "form": row.form,
            "primary_document": row.primary_document,
            "items": row.items,
            "filing_date": row.filing_date,
            "report_date": row.report_date,
            "is_xbrl": row.is_xbrl,
            "date_of_filing_date_change": row.date_of_filing_date_change,
            "date_of_filing_date_change_present": (
                row.date_of_filing_date_change_present
            ),
            "typed_values": row.semantic_identity()["typed_values"],
            "semantic_identity_sha256": row.semantic_identity_sha256,
            "source_name": row.source_name,
            "source_content_sha256": row.source_content_sha256,
            "source_position": row.source_position,
            "source_row_order_sha256": row.source_row_order_sha256,
        }
    )


def _submission_row_from_compact(value: Mapping[str, Any]) -> SubmissionRow:
    expected = {
        "accession_number",
        "subject_cik",
        "acceptance_datetime",
        "form",
        "primary_document",
        "items",
        "filing_date",
        "report_date",
        "is_xbrl",
        "date_of_filing_date_change",
        "date_of_filing_date_change_present",
        "typed_values",
        "semantic_identity_sha256",
        "source_name",
        "source_content_sha256",
        "source_position",
        "source_row_order_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise SecGemmaLeanV37SourceError("Compact Submissions row schema is not exact")
    if type(value["typed_values"]) is not dict:
        raise SecGemmaLeanV37SourceError("Compact Submissions typed values are invalid")
    try:
        row = SubmissionRow(
            accession_number=value["accession_number"],
            subject_cik=value["subject_cik"],
            acceptance_datetime=value["acceptance_datetime"],
            form=value["form"],
            primary_document=value["primary_document"],
            items=value["items"],
            filing_date=value["filing_date"],
            report_date=value["report_date"],
            is_xbrl=value["is_xbrl"],
            date_of_filing_date_change=value["date_of_filing_date_change"],
            date_of_filing_date_change_present=value[
                "date_of_filing_date_change_present"
            ],
            typed_values=_canonical_snapshot(value["typed_values"]),
            semantic_identity_sha256=value["semantic_identity_sha256"],
            source_name=value["source_name"],
            source_content_sha256=value["source_content_sha256"],
            source_position=value["source_position"],
            source_row_order_sha256=value["source_row_order_sha256"],
        )
    except (KeyError, TypeError):
        raise SecGemmaLeanV37SourceError("Compact Submissions row types are invalid") from None
    if (
        type(row.accession_number) is not str
        or _ACCESSION_RE.fullmatch(row.accession_number) is None
        or row.subject_cik != AAPL_CIK
        or type(row.acceptance_datetime) is not str
        or type(row.form) is not str
        or not row.form
        or row.form != row.form.strip()
        or type(row.primary_document) is not str
        or type(row.items) is not str
        or type(row.is_xbrl) is not bool
        or type(row.date_of_filing_date_change_present) is not bool
        or type(row.source_name) is not str
        or type(row.source_position) is not int
        or row.source_position < 0
        or _SHA256_RE.fullmatch(str(row.source_content_sha256)) is None
        or _BARE_SHA256_RE.fullmatch(str(row.source_row_order_sha256)) is None
        or _canonical_sha256(row.semantic_identity())
        != row.semantic_identity_sha256
    ):
        raise SecGemmaLeanV37SourceError("Compact Submissions row is not authenticated")
    upper_form = row.form.upper()
    if upper_form in {"10-K", "10-Q", "10-K/A", "10-Q/A"} and row.form != upper_form:
        raise SecGemmaLeanV37SourceError(
            "Compact periodic Submissions form spelling is not canonical"
        )
    typed = row.typed_values
    mandatory_text = (
        "accessionNumber",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "items",
        "filingDate",
        "reportDate",
    )
    if any(name not in typed or type(typed[name]) is not str for name in mandatory_text):
        raise SecGemmaLeanV37SourceError(
            "Compact Submissions typed values are incomplete"
        )
    change_present = "dateOfFilingDateChange" in typed
    raw_change = typed.get("dateOfFilingDateChange", "")
    if type(raw_change) is not str:
        raise SecGemmaLeanV37SourceError(
            "Compact dateOfFilingDateChange is not a string"
        )
    canonical_change = _canonical_date(
        raw_change,
        location="compact typed filing-date change",
        allow_empty=True,
    )
    try:
        expected_xbrl = _flag(
            typed.get("isXBRL", 0),
            location="compact typed isXBRL",
        )
    except SecGemmaLeanV37SourceError:
        raise
    if (
        typed["accessionNumber"] != row.accession_number
        or typed["acceptanceDateTime"] != row.acceptance_datetime
        or typed["form"] != row.form
        or typed["primaryDocument"] != row.primary_document
        or typed["items"] != row.items
        or typed["filingDate"] != row.filing_date
        or typed["reportDate"] != row.report_date
        or change_present != row.date_of_filing_date_change_present
        or (canonical_change or None) != row.date_of_filing_date_change
        or expected_xbrl is not row.is_xbrl
    ):
        raise SecGemmaLeanV37SourceError(
            "Compact Submissions derived values differ from typed values"
        )
    _canonical_date(row.filing_date, location="compact filing date")
    _canonical_date(row.report_date, location="compact report date", allow_empty=True)
    if row.acceptance_datetime:
        parse_submissions_acceptance_datetime(row.acceptance_datetime)
    if row.date_of_filing_date_change is not None:
        _canonical_date(
            row.date_of_filing_date_change,
            location="compact filing-date change",
        )
    if row.form in {"10-K", "10-Q"}:
        _safe_primary_name(
            row.primary_document,
            location="compact primaryDocument",
            allow_empty=True,
        )
    return row


def _historical_reference_from_compact(value: Mapping[str, Any]) -> HistoricalReference:
    if type(value) is not dict or set(value) != {
        "name",
        "url",
        "filing_count",
        "filing_from",
        "filing_to",
        "source_position",
    }:
        raise SecGemmaLeanV37SourceError("Compact historical reference is invalid")
    reference = HistoricalReference(**value)
    if (
        _HISTORICAL_NAME_RE.fullmatch(reference.name) is None
        or reference.url != f"https://data.sec.gov/submissions/{reference.name}"
        or type(reference.filing_count) is not int
        or reference.filing_count < 1
        or type(reference.source_position) is not int
        or reference.source_position < 0
    ):
        raise SecGemmaLeanV37SourceError("Compact historical reference is malformed")
    _canonical_date(reference.filing_from, location="compact filingFrom")
    _canonical_date(reference.filing_to, location="compact filingTo")
    if reference.filing_from > reference.filing_to:
        raise SecGemmaLeanV37SourceError("Compact historical reference is reversed")
    return reference


def _source_from_summary(summary: Mapping[str, Any]) -> SubmissionsSource:
    if type(summary) is not dict or set(summary) != {
        "name",
        "url",
        "payload_sha256",
        "payload_length",
        "raw_row_count",
        "unique_row_count",
        "row_order_sha256",
        "range_evidence",
    }:
        raise SecGemmaLeanV37SourceError("Compact Submissions source summary is invalid")
    source = SubmissionsSource(
        name=summary["name"],
        url=summary["url"],
        payload=b"",
        payload_sha256=summary["payload_sha256"],
        payload_length=summary["payload_length"],
        raw_row_count=summary["raw_row_count"],
        unique_row_count=summary["unique_row_count"],
        row_order_sha256=summary["row_order_sha256"],
        range_evidence=_canonical_snapshot(summary["range_evidence"]),
    )
    if source.summary() != summary:
        raise SecGemmaLeanV37SourceError("Compact Submissions source summary drifted")
    return source


def _snapshot_from_compact_phase(evidence: Mapping[str, Any]) -> SubmissionsSnapshot:
    value = _validate_compact_evidence(
        evidence,
        schema_version=COMPACT_SUBMISSIONS_SCHEMA_VERSION,
    )
    references = tuple(
        _historical_reference_from_compact(item) for item in value["references"]
    )
    sources = tuple(_source_from_summary(item) for item in value["sources"])
    rows = tuple(_submission_row_from_compact(item) for item in value["rows"])
    snapshot = SubmissionsSnapshot(
        references=references,
        sources=sources,
        rows=rows,
        snapshot_sha256=value["submissions_snapshot"]["snapshot_sha256"],
    )
    if snapshot.summary() != value["submissions_snapshot"]:
        raise SecGemmaLeanV37SourceError("Compact Submissions snapshot does not replay")
    return snapshot


def parse_main(stage: str | StageSourceConfig, payload: bytes) -> RoleParseOutput:
    """Parse one main Submissions body and return compact evidence only."""

    config = _stage_config(stage)
    main = _json_object(payload, location="main Submissions response")
    references = _historical_references(main)
    payload_sha = content_sha256(payload)
    rows, row_order_sha256, _ = _parse_columns(
        main["filings"]["recent"],
        source_name=MAIN_SUBMISSIONS_NAME,
        source_sha256=payload_sha,
        reference=None,
    )
    source = SubmissionsSource(
        name=MAIN_SUBMISSIONS_NAME,
        url=MAIN_SUBMISSIONS_URL,
        payload=b"",
        payload_sha256=payload_sha,
        payload_length=len(payload),
        raw_row_count=len(rows),
        unique_row_count=len(rows),
        row_order_sha256=row_order_sha256,
        range_evidence=None,
    )
    filenames = tuple(sorted(reference.name for reference in references))
    body = {
        "schema_version": COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION,
        "role_type": "submissions_main",
        "stage": config.stage,
        "role_id": "submissions/main",
        "url": MAIN_SUBMISSIONS_URL,
        "source": source.summary(),
        "references": [reference.to_dict() for reference in references],
        "historical_reference_array_order_sha256": _canonical_sha256(
            [reference.to_dict() for reference in references]
        ),
        "historical_filenames": list(filenames),
        "rows": [_submission_row_compact(row) for row in rows],
    }
    return RoleParseOutput(
        evidence=_compact_evidence(body),
        historical_filenames=filenames,
    )


def _validate_main_compact(evidence: Mapping[str, Any], config: StageSourceConfig) -> dict[str, Any]:
    value = _validate_compact_evidence(
        evidence,
        schema_version=COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION,
    )
    if (
        value.get("role_type") != "submissions_main"
        or value.get("stage") != config.stage
        or value.get("role_id") != "submissions/main"
        or value.get("url") != MAIN_SUBMISSIONS_URL
        or type(value.get("references")) is not list
        or len(value["references"]) > MAX_HISTORICAL_REFERENCES
        or type(value.get("historical_filenames")) is not list
        or type(value.get("rows")) is not list
    ):
        raise SecGemmaLeanV37SourceError("Compact main evidence is invalid")
    references = [
        _historical_reference_from_compact(item) for item in value["references"]
    ]
    if (
        [item.source_position for item in references] != list(range(len(references)))
        or value["historical_filenames"]
        != sorted(reference.name for reference in references)
        or value["historical_reference_array_order_sha256"]
        != _canonical_sha256([reference.to_dict() for reference in references])
    ):
        raise SecGemmaLeanV37SourceError("Compact main reference order drifted")
    rows = [_submission_row_from_compact(item) for item in value["rows"]]
    source = _source_from_summary(value["source"])
    if source.name != MAIN_SUBMISSIONS_NAME or source.raw_row_count != len(rows):
        raise SecGemmaLeanV37SourceError("Compact main source counts drifted")
    return value


def parse_historical(
    stage: str | StageSourceConfig,
    filename: str,
    payload: bytes,
    main_evidence: RoleParseOutput | Mapping[str, Any],
) -> RoleParseOutput:
    """Parse one referenced historical body without retaining its bytes."""

    config = _stage_config(stage)
    main = _validate_main_compact(_output_evidence(main_evidence), config)
    references = {
        item["name"]: _historical_reference_from_compact(item)
        for item in main["references"]
    }
    if type(filename) is not str or filename not in references:
        raise SecGemmaLeanV37SourceError(
            "Historical compact role is not referenced by main"
        )
    parsed = _json_object(payload, location=f"historical Submissions {filename}")
    payload_sha = content_sha256(payload)
    rows, row_order_sha256, range_evidence = _parse_columns(
        parsed,
        source_name=filename,
        source_sha256=payload_sha,
        reference=references[filename],
    )
    source = SubmissionsSource(
        name=filename,
        url=references[filename].url,
        payload=b"",
        payload_sha256=payload_sha,
        payload_length=len(payload),
        raw_row_count=len(rows),
        unique_row_count=len(rows),
        row_order_sha256=row_order_sha256,
        range_evidence=range_evidence,
    )
    body = {
        "schema_version": COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION,
        "role_type": "submissions_historical",
        "stage": config.stage,
        "role_id": f"submissions/historical/{filename}",
        "url": references[filename].url,
        "filename": filename,
        "source": source.summary(),
        "rows": [_submission_row_compact(row) for row in rows],
    }
    return RoleParseOutput(evidence=_compact_evidence(body))


def _compact_prior_seal(
    config: StageSourceConfig,
    prior: StageOutput | None,
    *,
    validate: bool = True,
) -> dict[str, Any] | None:
    if config.stage == "development":
        if prior is not None:
            raise SecGemmaLeanV37SourceError("Development compact stage cannot carry prior")
        return None
    if not isinstance(prior, StageOutput):
        raise SecGemmaLeanV37SourceError(
            "Later compact stage requires adjacent compact prior authority"
        )
    expected = _STAGE_ORDER[_STAGE_ORDER.index(config.stage) - 1]
    if prior.seal.get("stage") != expected:
        raise SecGemmaLeanV37SourceError("Compact prior stage is not adjacent")
    if validate:
        _validate_compact_stage_output(prior)
    else:
        validate_stage_source_seal(prior.seal)
    return _canonical_snapshot(prior.seal)


def finalize_submissions(
    stage: str | StageSourceConfig,
    main_evidence: RoleParseOutput | Mapping[str, Any],
    historical_evidence: Sequence[RoleParseOutput | Mapping[str, Any]],
    prior: StageOutput | None = None,
) -> PhaseOutput:
    """Finalize the exact H union and rejection-only pre-master bounds."""

    config = _stage_config(stage)
    main = _validate_main_compact(_output_evidence(main_evidence), config)
    if isinstance(historical_evidence, (str, bytes)) or not isinstance(
        historical_evidence, Sequence
    ):
        raise SecGemmaLeanV37SourceError("Historical compact evidence must be ordered")
    expected_names = list(main["historical_filenames"])
    observed: list[dict[str, Any]] = []
    observed_names: list[str] = []
    for raw in historical_evidence:
        value = _validate_compact_evidence(
            _output_evidence(raw),
            schema_version=COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION,
        )
        if (
            value.get("role_type") != "submissions_historical"
            or value.get("stage") != config.stage
            or type(value.get("filename")) is not str
            or value.get("role_id")
            != f"submissions/historical/{value.get('filename')}"
            or type(value.get("rows")) is not list
        ):
            raise SecGemmaLeanV37SourceError("Historical compact evidence is invalid")
        observed.append(value)
        observed_names.append(value["filename"])
    if observed_names != expected_names or len(observed_names) != len(set(observed_names)):
        raise SecGemmaLeanV37SourceError(
            "Historical compact role set/order is not exact"
        )
    references = [
        _historical_reference_from_compact(item) for item in main["references"]
    ]
    reference_by_name = {item.name: item for item in references}
    main_source = _source_from_summary(main["source"])
    main_rows = [_submission_row_from_compact(item) for item in main["rows"]]
    sources = [main_source]
    rows = list(main_rows)
    for value in observed:
        name = value["filename"]
        source = _source_from_summary(value["source"])
        parsed_rows = [_submission_row_from_compact(item) for item in value["rows"]]
        if (
            name not in reference_by_name
            or source.name != name
            or source.url != reference_by_name[name].url
            or source.raw_row_count != reference_by_name[name].filing_count
            or source.raw_row_count != len(parsed_rows)
        ):
            raise SecGemmaLeanV37SourceError(
                "Historical compact source/reference counts differ"
            )
        sources.append(source)
        rows.extend(parsed_rows)
    accessions = [row.accession_number for row in rows]
    if len(accessions) != len(set(accessions)):
        raise SecGemmaLeanV37SourceError(
            "Accession repeats across compact Submissions union"
        )
    rows.sort(key=lambda row: (row.filing_date, row.accession_number))
    snapshot_body = {
        "references": [item.to_dict() for item in references],
        "sources": [source.summary() for source in sources],
        "semantic_rows_sha256": _canonical_sha256(
            [row.semantic_identity() for row in rows]
        ),
        "layout_rows_sha256": _canonical_sha256(
            [row.layout_provenance() for row in rows]
        ),
        "row_count": len(rows),
    }
    snapshot = SubmissionsSnapshot(
        references=tuple(references),
        sources=tuple(sources),
        rows=tuple(rows),
        snapshot_sha256=_canonical_sha256(snapshot_body),
    )
    prior_seal = _compact_prior_seal(config, prior)
    pre_master = _pre_master_upper_bound_from_prior_seal(
        snapshot,
        config,
        prior_seal,
    )
    body = {
        "schema_version": COMPACT_SUBMISSIONS_SCHEMA_VERSION,
        "stage": config.stage,
        "main_source_evidence_sha256": main["source_evidence_sha256"],
        "historical_source_evidence_sha256": [
            value["source_evidence_sha256"] for value in observed
        ],
        "references": [item.to_dict() for item in references],
        "sources": [source.summary() for source in sources],
        "rows": [_submission_row_compact(row) for row in rows],
        "submissions_snapshot": snapshot.summary(),
        "pre_master_upper_bound": pre_master,
        "prior_stage_source_seal_sha256": (
            prior_seal["stage_source_seal_sha256"]
            if prior_seal is not None
            else None
        ),
    }
    return PhaseOutput(evidence=_compact_evidence(body))


def _master_target_from_compact(value: Mapping[str, Any]) -> MasterTargetRow:
    if type(value) is not dict or set(value) != {
        "accession_number",
        "cik",
        "company_name",
        "form",
        "filing_date",
        "filename",
        "quarter",
        "complete_submission_url",
    }:
        raise SecGemmaLeanV37SourceError("Compact master target schema is not exact")
    quarter = value["quarter"]
    if (
        type(quarter) is not list
        or len(quarter) != 2
        or any(type(item) is not int for item in quarter)
    ):
        raise SecGemmaLeanV37SourceError("Compact master quarter identity is invalid")
    target = MasterTargetRow(
        accession_number=value["accession_number"],
        cik=value["cik"],
        company_name=value["company_name"],
        form=value["form"],
        filing_date=value["filing_date"],
        filename=value["filename"],
        quarter=(quarter[0], quarter[1]),
    )
    if target.to_dict() != value:
        raise SecGemmaLeanV37SourceError("Compact master target identity drifted")
    return target


def _master_evidence_from_compact(value: Mapping[str, Any]) -> MasterQuarterEvidence:
    compact = _validate_compact_evidence(
        value,
        schema_version=COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION,
    )
    if (
        compact.get("role_type") != "master"
        or type(compact.get("year")) is not int
        or type(compact.get("quarter")) is not int
        or type(compact.get("summary")) is not dict
        or type(compact.get("target_rows")) is not list
    ):
        raise SecGemmaLeanV37SourceError("Compact master role evidence is invalid")
    role = QuarterRole(
        year=compact["year"],
        quarter=compact["quarter"],
        url=compact["url"],
    )
    if compact.get("role_id") != f"master/{role.year}/QTR{role.quarter}" or role.url != (
        "https://www.sec.gov/Archives/edgar/full-index/"
        f"{role.year}/QTR{role.quarter}/master.gz"
    ):
        raise SecGemmaLeanV37SourceError("Compact master role identity is not exact")
    summary = compact["summary"]
    targets = tuple(_master_target_from_compact(item) for item in compact["target_rows"])
    evidence = MasterQuarterEvidence(
        role=role,
        compressed_payload=b"",
        compressed_sha256=summary.get("compressed_sha256"),
        compressed_length=summary.get("compressed_length"),
        decompressed_sha256=summary.get("decompressed_sha256"),
        decompressed_length=summary.get("decompressed_length"),
        raw_row_count=summary.get("raw_row_count"),
        raw_row_order_sha256=summary.get("raw_row_order_sha256"),
        target_rows=targets,
    )
    if evidence.summary() != summary:
        raise SecGemmaLeanV37SourceError("Compact master summary does not authenticate")
    return evidence


def parse_master(
    stage: str | StageSourceConfig,
    year: int,
    quarter: int,
    payload: bytes,
) -> RoleParseOutput:
    """Parse one master.gz and return targets plus byte counts, never raw bytes."""

    config = _stage_config(stage)
    allowed = {role.key for role in quarter_roles(config)}
    if (year, quarter) not in allowed:
        raise SecGemmaLeanV37SourceError("Master compact role is outside frozen Q")
    parsed = parse_strict_master_gzip(payload, year=year, quarter=quarter)
    summary = parsed.summary()
    decompressed_bytes = parsed.decompressed_length
    target_rows = [target.to_dict() for target in parsed.target_rows]
    body = {
        "schema_version": COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION,
        "role_type": "master",
        "stage": config.stage,
        "role_id": f"master/{year}/QTR{quarter}",
        "url": parsed.role.url,
        "year": year,
        "quarter": quarter,
        "summary": summary,
        "target_rows": target_rows,
    }
    return RoleParseOutput(
        evidence=_compact_evidence(body),
        decompressed_bytes=decompressed_bytes,
    )


def _compact_target(target: ReconciledTarget) -> dict[str, Any]:
    body = {
        "submissions": _submission_row_compact(target.submissions),
        "master": target.master.to_dict(),
    }
    return _canonical_snapshot({**body, "target_sha256": _canonical_sha256(body)})


def _reconciled_target_from_compact(value: Mapping[str, Any]) -> ReconciledTarget:
    if type(value) is not dict or set(value) != {
        "submissions",
        "master",
        "target_sha256",
    }:
        raise SecGemmaLeanV37SourceError("Compact reconciled target schema is invalid")
    body = {"submissions": value["submissions"], "master": value["master"]}
    if value["target_sha256"] != _canonical_sha256(body):
        raise SecGemmaLeanV37SourceError("Compact reconciled target hash is invalid")
    submissions = _submission_row_from_compact(value["submissions"])
    master = _master_target_from_compact(value["master"])
    if (
        submissions.accession_number != master.accession_number
        or submissions.subject_cik != master.cik
        or submissions.form != master.form
        or submissions.filing_date != master.filing_date
        or master.filename
        != f"edgar/data/320193/{submissions.accession_number}.txt"
    ):
        raise SecGemmaLeanV37SourceError("Compact target identities do not reconcile")
    return ReconciledTarget(submissions=submissions, master=master)


def _validate_compact_reconciliation(
    evidence: Mapping[str, Any],
    config: StageSourceConfig,
) -> dict[str, Any]:
    value = _validate_compact_evidence(
        evidence,
        schema_version=COMPACT_RECONCILIATION_SCHEMA_VERSION,
    )
    if (
        value.get("stage") != config.stage
        or type(value.get("targets")) is not list
        or type(value.get("complete_targets")) is not list
        or type(value.get("master_quarters")) is not list
        or type(value.get("master_target_rows")) is not list
        or type(value.get("master_source_evidence_sha256")) is not list
        or type(value.get("source_plan")) is not dict
    ):
        raise SecGemmaLeanV37SourceError("Compact reconciliation evidence is invalid")
    targets = [_reconciled_target_from_compact(item) for item in value["targets"]]
    order = [
        (target.submissions.filing_date, target.submissions.accession_number)
        for target in targets
    ]
    if order != sorted(order) or len(order) != len(set(order)):
        raise SecGemmaLeanV37SourceError("Compact reconciliation target order is invalid")
    p = value["source_plan"].get("new_complete_submission_accessions")
    carried = value["source_plan"].get("carried_accessions")
    accessions = [target.submissions.accession_number for target in targets]
    if (
        type(p) is not list
        or type(carried) is not list
        or set(p) | set(carried) != set(accessions)
        or set(p) & set(carried)
        or p != [accession for accession in accessions if accession in set(p)]
        or carried
        != [accession for accession in accessions if accession in set(carried)]
        or value["complete_targets"]
        != [item for item in value["targets"] if item["submissions"]["accession_number"] in set(p)]
    ):
        raise SecGemmaLeanV37SourceError("Compact reconciliation P/carry is invalid")
    return value


def reconcile_masters(
    stage: str | StageSourceConfig,
    submissions_evidence: PhaseOutput | Mapping[str, Any],
    ordered_master_evidence: Sequence[RoleParseOutput | Mapping[str, Any]],
    prior: StageOutput | None = None,
) -> ReconciliationOutput:
    """Exact-set reconcile compact Q evidence and freeze P after prior replay."""

    config = _stage_config(stage)
    phase = _validate_compact_evidence(
        _output_evidence(submissions_evidence),
        schema_version=COMPACT_SUBMISSIONS_SCHEMA_VERSION,
    )
    if phase.get("stage") != config.stage:
        raise SecGemmaLeanV37SourceError("Compact Submissions stage differs")
    snapshot = _snapshot_from_compact_phase(phase)
    prior_seal = _compact_prior_seal(config, prior)
    expected_roles = quarter_roles(config)
    if isinstance(ordered_master_evidence, (str, bytes)) or not isinstance(
        ordered_master_evidence, Sequence
    ):
        raise SecGemmaLeanV37SourceError("Compact master evidence must be ordered")
    values: list[dict[str, Any]] = []
    quarters: list[MasterQuarterEvidence] = []
    for raw in ordered_master_evidence:
        value = _validate_compact_evidence(
            _output_evidence(raw),
            schema_version=COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION,
        )
        if value.get("stage") != config.stage:
            raise SecGemmaLeanV37SourceError("Compact master stage differs")
        values.append(value)
        quarters.append(_master_evidence_from_compact(value))
    if [quarter.role.key for quarter in quarters] != [role.key for role in expected_roles]:
        raise SecGemmaLeanV37SourceError("Compact Q role set/order is not exact")
    master_by_accession: dict[str, MasterTargetRow] = {}
    for evidence in quarters:
        for row in evidence.target_rows:
            if row.filing_date > config.fixed_cutoff:
                continue
            if row.accession_number in master_by_accession:
                raise SecGemmaLeanV37SourceError(
                    "Compact Apple target repeats across masters"
                )
            master_by_accession[row.accession_number] = row
    submissions_by_accession = {
        row.accession_number: row for row in _submissions_candidates(snapshot, config)
    }
    if set(submissions_by_accession) != set(master_by_accession):
        raise SecGemmaLeanV37SourceError(
            "Compact master/Submissions exact target set differs"
        )
    targets: list[ReconciledTarget] = []
    for accession, submissions in submissions_by_accession.items():
        master = master_by_accession[accession]
        target = ReconciledTarget(submissions=submissions, master=master)
        _reconciled_target_from_compact(_compact_target(target))
        targets.append(target)
    targets.sort(
        key=lambda target: (
            target.submissions.filing_date,
            target.submissions.accession_number,
        )
    )
    if len(targets) > config.i_cap:
        raise SecGemmaLeanV37SourceError("Compact I cap is exceeded")
    prior_accessions = (
        tuple(row["accession_number"] for row in prior_seal["target_rows"])
        if prior_seal is not None
        else ()
    )
    current_accessions = tuple(target.submissions.accession_number for target in targets)
    if not set(prior_accessions) <= set(current_accessions):
        raise SecGemmaLeanV37SourceError("Compact current I omits prior target")
    if prior_seal is not None:
        prior_rows = {
            row["accession_number"]: row for row in prior_seal["target_rows"]
        }
        for target in targets:
            accession = target.submissions.accession_number
            if accession not in prior_rows:
                continue
            prefix = prior_rows[accession]["frozen_prefix"]
            current_master = {
                "accession_number": target.master.accession_number,
                "cik": target.master.cik,
                "form": target.master.form,
                "filing_date": target.master.filing_date,
                "filename": target.master.filename,
            }
            if (
                prefix["submissions_semantic_identity"]
                != target.submissions.semantic_identity()
                or prefix["submissions_semantic_identity_sha256"]
                != target.submissions.semantic_identity_sha256
                or prefix["master_identity"] != current_master
            ):
                raise SecGemmaLeanV37SourceError(
                    "Compact prior Submissions/master frozen prefix drifted"
                )
    p_accessions = tuple(
        accession for accession in current_accessions if accession not in set(prior_accessions)
    )
    if len(p_accessions) > config.p_cap:
        raise SecGemmaLeanV37SourceError("Compact P cap is exceeded")
    compact_targets = [_compact_target(target) for target in targets]
    complete_targets = [
        item
        for item in compact_targets
        if item["submissions"]["accession_number"] in set(p_accessions)
    ]
    reconciliation_body = {
        "stage": config.stage,
        "config": config.to_dict(),
        "quarters": [item.summary() for item in quarters],
        "targets": [
            {
                "submissions_semantic_identity_sha256": (
                    target.submissions.semantic_identity_sha256
                ),
                "master": target.master.to_dict(),
            }
            for target in targets
        ],
    }
    reconciliation_sha256 = _canonical_sha256(reconciliation_body)
    if phase["pre_master_upper_bound"] != _pre_master_upper_bound_from_prior_seal(
        snapshot,
        config,
        prior_seal,
    ):
        raise SecGemmaLeanV37SourceError("Compact pre-master bound did not replay")
    body = {
        "schema_version": COMPACT_RECONCILIATION_SCHEMA_VERSION,
        "stage": config.stage,
        "submissions_source_evidence_sha256": phase["source_evidence_sha256"],
        "master_source_evidence_sha256": [
            value["source_evidence_sha256"] for value in values
        ],
        "master_quarters": [item.summary() for item in quarters],
        "master_target_rows": [
            [row.to_dict() for row in item.target_rows] for item in quarters
        ],
        "master_reconciliation_sha256": reconciliation_sha256,
        "targets": compact_targets,
        "complete_targets": complete_targets,
        "source_plan": {
            "new_complete_submission_accessions": list(p_accessions),
            "carried_accessions": list(prior_accessions),
            "ordering": "filing_date_then_accession",
            "subtraction": "I_minus_authenticated_adjacent_prior_v37_seals",
        },
        "prior_stage_source_seal_sha256": (
            prior_seal["stage_source_seal_sha256"]
            if prior_seal is not None
            else None
        ),
    }
    evidence = _compact_evidence(body)
    _validate_compact_reconciliation(evidence, config)
    return ReconciliationOutput(
        evidence=evidence,
        complete_targets=tuple(_canonical_snapshot(item) for item in complete_targets),
    )


def parse_complete(
    stage: str | StageSourceConfig,
    target: Mapping[str, Any],
    payload: bytes,
    reconciliation_evidence: ReconciliationOutput | Mapping[str, Any],
) -> RoleParseOutput:
    """Parse one P body using only the authoritative internal session calendar."""

    config = _stage_config(stage)
    evidence_input = (
        reconciliation_evidence.evidence
        if isinstance(reconciliation_evidence, ReconciliationOutput)
        else reconciliation_evidence
    )
    reconciliation = _validate_compact_reconciliation(
        _canonical_snapshot(evidence_input),
        config,
    )
    compact_target = _canonical_snapshot(target)
    if compact_target not in reconciliation["complete_targets"]:
        raise SecGemmaLeanV37SourceError("Complete compact target is not an exact P member")
    reconciled = _reconciled_target_from_compact(compact_target)
    source = reconcile_complete_submission(
        reconciled,
        payload,
        session_dates=EXPECTED_SESSIONS,
        acquisition_stage=config.stage,
    )
    body = {
        "schema_version": COMPACT_COMPLETE_SCHEMA_VERSION,
        "stage": config.stage,
        "role_id": f"complete/{source.accession_number}",
        "url": reconciled.master.complete_submission_url,
        "accession_number": source.accession_number,
        "target_sha256": compact_target["target_sha256"],
        "master_reconciliation_sha256": reconciliation[
            "master_reconciliation_sha256"
        ],
        "seal_row": _canonical_snapshot(source.seal_row),
    }
    # ``source`` and all derived bytes die at this role boundary; only the
    # compact row hashes, byte lengths, offsets, and identities are returned.
    return RoleParseOutput(evidence=_compact_evidence(body))


def _validate_compact_complete(
    evidence: Mapping[str, Any],
    config: StageSourceConfig,
    reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    value = _validate_compact_evidence(
        evidence,
        schema_version=COMPACT_COMPLETE_SCHEMA_VERSION,
    )
    if (
        value.get("stage") != config.stage
        or type(value.get("accession_number")) is not str
        or value.get("role_id") != f"complete/{value.get('accession_number')}"
        or value.get("master_reconciliation_sha256")
        != reconciliation["master_reconciliation_sha256"]
        or type(value.get("seal_row")) is not dict
    ):
        raise SecGemmaLeanV37SourceError("Compact complete evidence is invalid")
    target = next(
        (
            item
            for item in reconciliation["complete_targets"]
            if item["submissions"]["accession_number"]
            == value["accession_number"]
        ),
        None,
    )
    if (
        target is None
        or value.get("target_sha256") != target["target_sha256"]
        or value.get("url") != target["master"]["complete_submission_url"]
        or value["seal_row"].get("accession_number") != value["accession_number"]
    ):
        raise SecGemmaLeanV37SourceError(
            "Compact complete target/URL identity does not reconcile"
        )
    _validate_target_row_binding(value["seal_row"])
    return value


def _compact_reconciliation_objects(
    evidence: Mapping[str, Any],
    config: StageSourceConfig,
) -> MasterReconciliation:
    value = _validate_compact_reconciliation(evidence, config)
    targets = tuple(_reconciled_target_from_compact(item) for item in value["targets"])
    quarters: list[MasterQuarterEvidence] = []
    expected_roles = quarter_roles(config)
    if (
        len(value["master_quarters"]) != len(expected_roles)
        or len(value["master_target_rows"]) != len(expected_roles)
    ):
        raise SecGemmaLeanV37SourceError("Compact reconciliation Q summary count differs")
    for role, summary, raw_targets in zip(
        expected_roles,
        value["master_quarters"],
        value["master_target_rows"],
    ):
        # Target rows are retained by each compact role hash, not duplicated in
        # the public quarter summary.  Reconstruct the exact summary fields for
        # final seal parity; target_rows_sha256 is checked against the role
        # evidence during reconcile_masters.
        if type(raw_targets) is not list:
            raise SecGemmaLeanV37SourceError(
                "Compact reconciliation master targets are invalid"
            )
        quarter_targets = tuple(
            _master_target_from_compact(item) for item in raw_targets
        )
        item = MasterQuarterEvidence(
            role=role,
            compressed_payload=b"",
            compressed_sha256=summary["compressed_sha256"],
            compressed_length=summary["compressed_length"],
            decompressed_sha256=summary["decompressed_sha256"],
            decompressed_length=summary["decompressed_length"],
            raw_row_count=summary["raw_row_count"],
            raw_row_order_sha256=summary["raw_row_order_sha256"],
            target_rows=quarter_targets,
        )
        if item.summary() != summary:
            raise SecGemmaLeanV37SourceError(
                "Compact reconciliation master summary cannot be reconstructed"
            )
        quarters.append(item)
    return MasterReconciliation(
        config=config,
        quarter_evidence=tuple(quarters),
        targets=targets,
        reconciliation_sha256=value["master_reconciliation_sha256"],
    )


def _finalize_stage_core(
    *,
    config: StageSourceConfig,
    submissions_evidence: Mapping[str, Any],
    reconciliation_evidence: Mapping[str, Any],
    complete_evidence: Sequence[Mapping[str, Any]],
    prior: StageOutput | None,
    validate_prior: bool,
) -> StageOutput:
    phase = _validate_compact_evidence(
        submissions_evidence,
        schema_version=COMPACT_SUBMISSIONS_SCHEMA_VERSION,
    )
    if phase.get("stage") != config.stage:
        raise SecGemmaLeanV37SourceError("Compact finalize Submissions stage differs")
    snapshot = _snapshot_from_compact_phase(phase)
    reconciliation_value = _validate_compact_reconciliation(
        reconciliation_evidence,
        config,
    )
    if reconciliation_value["submissions_source_evidence_sha256"] != phase[
        "source_evidence_sha256"
    ]:
        raise SecGemmaLeanV37SourceError(
            "Compact reconciliation does not bind Submissions phase"
        )
    reconciliation = _compact_reconciliation_objects(
        reconciliation_value,
        config,
    )
    prior_seal = _compact_prior_seal(
        config,
        prior,
        validate=validate_prior,
    )
    if phase["prior_stage_source_seal_sha256"] != (
        prior_seal["stage_source_seal_sha256"] if prior_seal is not None else None
    ) or reconciliation_value["prior_stage_source_seal_sha256"] != (
        prior_seal["stage_source_seal_sha256"] if prior_seal is not None else None
    ):
        raise SecGemmaLeanV37SourceError("Compact phase prior authority differs")
    p_accessions = tuple(
        reconciliation_value["source_plan"]["new_complete_submission_accessions"]
    )
    if isinstance(complete_evidence, (str, bytes)) or not isinstance(
        complete_evidence, Sequence
    ):
        raise SecGemmaLeanV37SourceError("Compact complete evidence must be ordered")
    parsed_complete: list[dict[str, Any]] = []
    for raw in complete_evidence:
        parsed_complete.append(
            _validate_compact_complete(
                _canonical_snapshot(raw),
                config,
                reconciliation_value,
            )
        )
    observed_p = tuple(item["accession_number"] for item in parsed_complete)
    if observed_p != p_accessions or len(observed_p) != len(set(observed_p)):
        raise SecGemmaLeanV37SourceError(
            "Compact complete evidence set/order does not equal P"
        )
    complete_by_accession = {
        item["accession_number"]: item["seal_row"] for item in parsed_complete
    }
    prior_rows = (
        {row["accession_number"]: row for row in prior_seal["target_rows"]}
        if prior_seal is not None
        else {}
    )
    target_rows: list[dict[str, Any]] = []
    for target in reconciliation.targets:
        accession = target.submissions.accession_number
        if accession in complete_by_accession:
            row = _canonical_snapshot(complete_by_accession[accession])
        else:
            if accession not in prior_rows:
                raise SecGemmaLeanV37SourceError(
                    "Compact carried target lacks prior sealed row"
                )
            row = _canonical_snapshot(prior_rows[accession])
            row["submissions_layout_provenance"] = (
                target.submissions.layout_provenance()
            )
        target_rows.append(row)
    seal, _ = _assemble_stage_seal_metadata(
        config=config,
        snapshot=snapshot,
        reconciliation=reconciliation,
        target_rows=target_rows,
        current_complete_accessions=p_accessions,
        prior_seal=prior_seal,
        pre_master_upper_bound=phase["pre_master_upper_bound"],
    )
    return StageOutput(
        seal=_canonical_snapshot(seal),
        submissions_evidence=_canonical_snapshot(phase),
        reconciliation_evidence=_canonical_snapshot(reconciliation_value),
        complete_evidence=tuple(
            _canonical_snapshot(item) for item in parsed_complete
        ),
        prior=prior,
    )


def finalize_stage(
    stage: str | StageSourceConfig,
    submissions_evidence: PhaseOutput | Mapping[str, Any],
    reconciliation_evidence: ReconciliationOutput | Mapping[str, Any],
    complete_evidence: Sequence[RoleParseOutput | Mapping[str, Any]],
    prior: StageOutput | None = None,
) -> StageOutput:
    """Finalize one compact stage and reproduce the full source seal exactly."""

    config = _stage_config(stage)
    phase = _output_evidence(submissions_evidence)
    reconciliation = (
        _canonical_snapshot(reconciliation_evidence.evidence)
        if isinstance(reconciliation_evidence, ReconciliationOutput)
        else _canonical_snapshot(reconciliation_evidence)
    )
    completes = [_output_evidence(item) for item in complete_evidence]
    return _finalize_stage_core(
        config=config,
        submissions_evidence=phase,
        reconciliation_evidence=reconciliation,
        complete_evidence=completes,
        prior=prior,
        validate_prior=True,
    )


def _validate_compact_stage_output(output: StageOutput) -> None:
    if not isinstance(output, StageOutput):
        raise SecGemmaLeanV37SourceError("Compact stage authority type is invalid")
    stage = output.seal.get("stage")
    config = _stage_config(stage)
    validate_stage_source_seal(output.seal)
    if output.prior is not None:
        _validate_compact_stage_output(output.prior)
    rebuilt = _finalize_stage_core(
        config=config,
        submissions_evidence=_canonical_snapshot(output.submissions_evidence),
        reconciliation_evidence=_canonical_snapshot(output.reconciliation_evidence),
        complete_evidence=[
            _canonical_snapshot(item) for item in output.complete_evidence
        ],
        prior=output.prior,
        validate_prior=False,
    )
    if rebuilt.seal != output.seal:
        raise SecGemmaLeanV37SourceError(
            "Compact prior authority does not reproduce its exact source seal"
        )


def _opaque_receipt_hash(value: Any, *, location: str) -> str:
    if type(value) is not str or _BARE_SHA256_RE.fullmatch(value) is None:
        raise SecGemmaLeanV37SourceError(f"{location} opaque receipt hash is malformed")
    return value


def _compact_role_body_identity(
    evidence: Mapping[str, Any],
) -> tuple[str, str, str, int]:
    schema = evidence.get("schema_version") if type(evidence) is dict else None
    if schema == COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION:
        value = _validate_compact_evidence(
            evidence,
            schema_version=COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION,
        )
        role_type = value.get("role_type")
        if role_type in {"submissions_main", "submissions_historical"}:
            source = value.get("source")
            if type(source) is not dict:
                raise SecGemmaLeanV37SourceError(
                    "Compact role lacks Submissions body identity"
                )
            body_sha256 = source.get("payload_sha256")
            body_bytes = source.get("payload_length")
        elif role_type == "master":
            summary = value.get("summary")
            if type(summary) is not dict:
                raise SecGemmaLeanV37SourceError(
                    "Compact role lacks master body identity"
                )
            body_sha256 = summary.get("compressed_sha256")
            body_bytes = summary.get("compressed_length")
        else:
            raise SecGemmaLeanV37SourceError("Compact role type has no body identity")
    elif schema == COMPACT_COMPLETE_SCHEMA_VERSION:
        value = _validate_compact_evidence(
            evidence,
            schema_version=COMPACT_COMPLETE_SCHEMA_VERSION,
        )
        response = value.get("seal_row", {}).get("complete_response")
        if type(response) is not dict:
            raise SecGemmaLeanV37SourceError(
                "Compact complete role lacks body identity"
            )
        body_sha256 = response.get("sha256")
        body_bytes = response.get("length")
    else:
        raise SecGemmaLeanV37SourceError(
            "Compact parse output schema has no manifest body contract"
        )
    role_id = value.get("role_id")
    url = value.get("url")
    if (
        type(role_id) is not str
        or type(url) is not str
        or type(body_sha256) is not str
        or _SHA256_RE.fullmatch(body_sha256) is None
        or type(body_bytes) is not int
        or body_bytes < 0
    ):
        raise SecGemmaLeanV37SourceError(
            "Compact parse output body identity is malformed"
        )
    return role_id, url, body_sha256, body_bytes


def build_compact_role_manifest(
    *,
    sequence: int,
    role_id: str,
    url: str,
    blob_name: str,
    payload: bytes,
    parse_output: RoleParseOutput,
    transport_receipt_sha256: str,
    parse_receipt_sha256: str,
) -> Mapping[str, Any]:
    """Build one raw-free role manifest with opaque journal/transport bindings."""

    if type(sequence) is not int or sequence < 0:
        raise SecGemmaLeanV37SourceError("Compact role sequence is invalid")
    if type(role_id) is not str or not role_id or type(url) is not str or not url:
        raise SecGemmaLeanV37SourceError("Compact role identity is invalid")
    if type(blob_name) is not str or _SAFE_BLOB_NAME_RE.fullmatch(blob_name) is None:
        raise SecGemmaLeanV37SourceError("Compact blob name is not a safe logical name")
    if type(payload) is not bytes:
        raise SecGemmaLeanV37SourceError("Compact role manifest requires exact bytes")
    if not isinstance(parse_output, RoleParseOutput):
        raise SecGemmaLeanV37SourceError("Compact role parse output type is invalid")
    evidence = _canonical_snapshot(parse_output.evidence)
    source_hash = evidence.get("source_evidence_sha256")
    _opaque_receipt_hash(source_hash, location="source evidence")
    evidence_role_id, evidence_url, evidence_body_sha256, evidence_body_bytes = (
        _compact_role_body_identity(evidence)
    )
    if (
        role_id != evidence_role_id
        or url != evidence_url
        or content_sha256(payload) != evidence_body_sha256
        or len(payload) != evidence_body_bytes
    ):
        raise SecGemmaLeanV37SourceError(
            "Compact manifest identity/body differs from parse evidence"
        )
    body = {
        "schema_version": COMPACT_ROLE_MANIFEST_SCHEMA_VERSION,
        "sequence": sequence,
        "role_id": role_id,
        "url": url,
        "blob_name": blob_name,
        "body_sha256": content_sha256(payload),
        "body_bytes": len(payload),
        "transport_receipt_sha256": _opaque_receipt_hash(
            transport_receipt_sha256,
            location="transport",
        ),
        "parse_receipt_sha256": _opaque_receipt_hash(
            parse_receipt_sha256,
            location="parse",
        ),
        "source_evidence_sha256": source_hash,
    }
    return _canonical_snapshot(
        {**body, "role_manifest_sha256": _canonical_sha256(body)}
    )


def _validate_compact_role_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "sequence",
        "role_id",
        "url",
        "blob_name",
        "body_sha256",
        "body_bytes",
        "transport_receipt_sha256",
        "parse_receipt_sha256",
        "source_evidence_sha256",
        "role_manifest_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise SecGemmaLeanV37SourceError("Compact role manifest schema is not exact")
    body = _canonical_snapshot(value)
    supplied = body.pop("role_manifest_sha256")
    if (
        value["schema_version"] != COMPACT_ROLE_MANIFEST_SCHEMA_VERSION
        or type(value["sequence"]) is not int
        or value["sequence"] < 0
        or type(value["role_id"]) is not str
        or not value["role_id"]
        or type(value["url"]) is not str
        or not value["url"]
        or type(value["blob_name"]) is not str
        or _SAFE_BLOB_NAME_RE.fullmatch(value["blob_name"]) is None
        or type(value["body_sha256"]) is not str
        or _SHA256_RE.fullmatch(value["body_sha256"]) is None
        or type(value["body_bytes"]) is not int
        or value["body_bytes"] < 0
        or type(supplied) is not str
        or _BARE_SHA256_RE.fullmatch(supplied) is None
        or _canonical_sha256(body) != supplied
    ):
        raise SecGemmaLeanV37SourceError("Compact role manifest is malformed")
    for name in (
        "transport_receipt_sha256",
        "parse_receipt_sha256",
        "source_evidence_sha256",
    ):
        _opaque_receipt_hash(value[name], location=name)
    return _canonical_snapshot(value)


def build_compact_checkpoint(
    *,
    stage: str | StageSourceConfig,
    role_manifests: Sequence[Mapping[str, Any]],
    stage_output: StageOutput,
    main_parse_receipt_sha256: str,
    submissions_snapshot_receipt_sha256: str,
    reconciliation_and_prior_chain_receipt_sha256: str,
) -> Mapping[str, Any]:
    """Bind compact roles, source seal, and opaque phase-plan journal receipts."""

    config = _stage_config(stage)
    _validate_compact_stage_output(stage_output)
    if stage_output.seal["stage"] != config.stage:
        raise SecGemmaLeanV37SourceError("Compact checkpoint stage differs")
    if isinstance(role_manifests, (str, bytes)) or not isinstance(
        role_manifests, Sequence
    ):
        raise SecGemmaLeanV37SourceError("Compact role manifests must be ordered")
    manifests = [_validate_compact_role_manifest(item) for item in role_manifests]
    if [item["sequence"] for item in manifests] != list(range(len(manifests))):
        raise SecGemmaLeanV37SourceError("Compact role manifest sequence is not exact")
    if not manifests or manifests[0]["role_id"] != "submissions/main":
        raise SecGemmaLeanV37SourceError("Compact checkpoint lacks main role first")
    main_receipt = _opaque_receipt_hash(
        main_parse_receipt_sha256,
        location="main parse",
    )
    if main_receipt != manifests[0]["parse_receipt_sha256"]:
        raise SecGemmaLeanV37SourceError(
            "Compact checkpoint main parse receipt differs from main manifest"
        )
    body = {
        "schema_version": COMPACT_CHECKPOINT_SCHEMA_VERSION,
        "stage": config.stage,
        "role_manifest_count": len(manifests),
        "role_manifests_sha256": _canonical_sha256(manifests),
        "expected_stage_source_seal_sha256": stage_output.stage_source_seal_sha256,
        "prior_stage_source_seal_sha256": stage_output.seal[
            "prior_stage_source_seal_sha256"
        ],
        "main_parse_receipt_sha256": main_receipt,
        "submissions_snapshot_receipt_sha256": _opaque_receipt_hash(
            submissions_snapshot_receipt_sha256,
            location="submissions snapshot",
        ),
        "reconciliation_and_prior_chain_receipt_sha256": _opaque_receipt_hash(
            reconciliation_and_prior_chain_receipt_sha256,
            location="reconciliation/prior chain",
        ),
        "submissions_source_evidence_sha256": stage_output.submissions_evidence[
            "source_evidence_sha256"
        ],
        "reconciliation_source_evidence_sha256": stage_output.reconciliation_evidence[
            "source_evidence_sha256"
        ],
        "opaque_receipt_boundary": (
            "acquisition_validates_journal_transport_and_blob_path_schemas"
        ),
    }
    return _canonical_snapshot(
        {**body, "checkpoint_sha256": _canonical_sha256(body)}
    )


def _validate_compact_checkpoint(
    checkpoint: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    config: StageSourceConfig,
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "stage",
        "role_manifest_count",
        "role_manifests_sha256",
        "expected_stage_source_seal_sha256",
        "prior_stage_source_seal_sha256",
        "main_parse_receipt_sha256",
        "submissions_snapshot_receipt_sha256",
        "reconciliation_and_prior_chain_receipt_sha256",
        "submissions_source_evidence_sha256",
        "reconciliation_source_evidence_sha256",
        "opaque_receipt_boundary",
        "checkpoint_sha256",
    }
    if type(checkpoint) is not dict or set(checkpoint) != expected:
        raise SecGemmaLeanV37SourceError("Compact checkpoint schema is not exact")
    value = _canonical_snapshot(checkpoint)
    supplied = value.pop("checkpoint_sha256")
    if (
        checkpoint["schema_version"] != COMPACT_CHECKPOINT_SCHEMA_VERSION
        or checkpoint["stage"] != config.stage
        or type(checkpoint["role_manifest_count"]) is not int
        or checkpoint["role_manifest_count"] != len(manifests)
        or checkpoint["role_manifests_sha256"] != _canonical_sha256(manifests)
        or type(supplied) is not str
        or _BARE_SHA256_RE.fullmatch(supplied) is None
        or _canonical_sha256(value) != supplied
        or checkpoint["opaque_receipt_boundary"]
        != "acquisition_validates_journal_transport_and_blob_path_schemas"
    ):
        raise SecGemmaLeanV37SourceError("Compact checkpoint hash/bindings are invalid")
    for name in (
        "expected_stage_source_seal_sha256",
        "submissions_source_evidence_sha256",
        "reconciliation_source_evidence_sha256",
        "main_parse_receipt_sha256",
        "submissions_snapshot_receipt_sha256",
        "reconciliation_and_prior_chain_receipt_sha256",
    ):
        _opaque_receipt_hash(checkpoint[name], location=name)
    prior_hash = checkpoint["prior_stage_source_seal_sha256"]
    if prior_hash is not None:
        _opaque_receipt_hash(prior_hash, location="prior stage source seal")
    return _canonical_snapshot(checkpoint)


def _load_compact_role(
    *,
    manifests: Sequence[Mapping[str, Any]],
    index: int,
    expected_role_id: str,
    expected_url: str,
    load_blob: SourceBlobLoader,
) -> tuple[bytes, dict[str, Any]]:
    if index >= len(manifests):
        raise SecGemmaLeanV37SourceError(
            f"Compact replay is missing required role {expected_role_id}"
        )
    manifest = manifests[index]
    if (
        manifest["sequence"] != index
        or manifest["role_id"] != expected_role_id
        or manifest["url"] != expected_url
    ):
        raise SecGemmaLeanV37SourceError(
            f"Compact replay role is misordered or mismatched at {index}"
        )
    try:
        payload = load_blob(manifest["blob_name"])
    except Exception:
        raise SecGemmaLeanV37SourceError("Compact blob loader failed safely") from None
    if type(payload) is not bytes:
        raise SecGemmaLeanV37SourceError("Compact blob loader did not return exact bytes")
    if (
        len(payload) != manifest["body_bytes"]
        or content_sha256(payload) != manifest["body_sha256"]
    ):
        raise SecGemmaLeanV37SourceError(
            "Compact loaded blob hash/length does not match its manifest"
        )
    return payload, manifest


def _require_role_source_hash(
    output: RoleParseOutput,
    manifest: Mapping[str, Any],
) -> None:
    if output.evidence.get("source_evidence_sha256") != manifest[
        "source_evidence_sha256"
    ]:
        raise SecGemmaLeanV37SourceError(
            "Compact role parse evidence differs from manifest"
        )


def _detached_replay_compact_core(
    *,
    config: StageSourceConfig,
    checkpoint: Mapping[str, Any],
    role_manifests: Sequence[Mapping[str, Any]],
    load_blob: SourceBlobLoader,
    prior: CompactReplayInput | None,
) -> tuple[StageOutput, dict[str, Any]]:
    if not callable(load_blob):
        raise SecGemmaLeanV37SourceError("Compact blob loader is not callable")
    manifests = [_validate_compact_role_manifest(item) for item in role_manifests]
    if [item["sequence"] for item in manifests] != list(range(len(manifests))):
        raise SecGemmaLeanV37SourceError("Compact manifest sequence is not exact")
    role_ids = [item["role_id"] for item in manifests]
    if len(role_ids) != len(set(role_ids)):
        raise SecGemmaLeanV37SourceError("Compact role manifest is duplicated")
    checkpoint_value = _validate_compact_checkpoint(checkpoint, manifests, config)
    if config.stage == "development":
        if prior is not None:
            raise SecGemmaLeanV37SourceError("Development compact replay cannot carry prior")
        prior_output = None
    else:
        if not isinstance(prior, CompactReplayInput):
            raise SecGemmaLeanV37SourceError(
                "Later compact replay requires disk-backed adjacent prior input"
            )
        expected_prior = _STAGE_ORDER[_STAGE_ORDER.index(config.stage) - 1]
        if prior.stage != expected_prior:
            raise SecGemmaLeanV37SourceError("Compact replay prior stage is not adjacent")
        prior_output, _ = _detached_replay_compact_core(
            config=_stage_config(prior.stage),
            checkpoint=_canonical_snapshot(prior.checkpoint),
            role_manifests=[
                _canonical_snapshot(item) for item in prior.role_manifests
            ],
            load_blob=load_blob,
            prior=prior.prior,
        )
    index = 0
    payload, manifest = _load_compact_role(
        manifests=manifests,
        index=index,
        expected_role_id="submissions/main",
        expected_url=MAIN_SUBMISSIONS_URL,
        load_blob=load_blob,
    )
    main = parse_main(config, payload)
    _require_role_source_hash(main, manifest)
    del payload
    index += 1
    histories: list[RoleParseOutput] = []
    for filename in main.historical_filenames:
        url = f"https://data.sec.gov/submissions/{filename}"
        payload, manifest = _load_compact_role(
            manifests=manifests,
            index=index,
            expected_role_id=f"submissions/historical/{filename}",
            expected_url=url,
            load_blob=load_blob,
        )
        parsed = parse_historical(config, filename, payload, main)
        _require_role_source_hash(parsed, manifest)
        del payload
        histories.append(parsed)
        index += 1
    phase = finalize_submissions(config, main, histories, prior_output)
    if phase.evidence["source_evidence_sha256"] != checkpoint_value[
        "submissions_source_evidence_sha256"
    ]:
        raise SecGemmaLeanV37SourceError(
            "Compact Submissions phase differs from checkpoint"
        )
    masters: list[RoleParseOutput] = []
    for role in quarter_roles(config):
        payload, manifest = _load_compact_role(
            manifests=manifests,
            index=index,
            expected_role_id=f"master/{role.year}/QTR{role.quarter}",
            expected_url=role.url,
            load_blob=load_blob,
        )
        parsed = parse_master(config, role.year, role.quarter, payload)
        _require_role_source_hash(parsed, manifest)
        del payload
        masters.append(parsed)
        index += 1
    reconciliation = reconcile_masters(config, phase, masters, prior_output)
    if reconciliation.evidence["source_evidence_sha256"] != checkpoint_value[
        "reconciliation_source_evidence_sha256"
    ]:
        raise SecGemmaLeanV37SourceError(
            "Compact reconciliation differs from checkpoint"
        )
    completes: list[RoleParseOutput] = []
    for target in reconciliation.complete_targets:
        accession = target["submissions"]["accession_number"]
        url = target["master"]["complete_submission_url"]
        payload, manifest = _load_compact_role(
            manifests=manifests,
            index=index,
            expected_role_id=f"complete/{accession}",
            expected_url=url,
            load_blob=load_blob,
        )
        parsed = parse_complete(config, target, payload, reconciliation)
        _require_role_source_hash(parsed, manifest)
        del payload
        completes.append(parsed)
        index += 1
    if index != len(manifests):
        raise SecGemmaLeanV37SourceError(
            "Compact replay contains orphan role manifests"
        )
    output = finalize_stage(config, phase, reconciliation, completes, prior_output)
    if output.stage_source_seal_sha256 != checkpoint_value[
        "expected_stage_source_seal_sha256"
    ]:
        raise SecGemmaLeanV37SourceError(
            "Compact replay source seal differs from checkpoint"
        )
    expected_prior_hash = (
        prior_output.stage_source_seal_sha256 if prior_output is not None else None
    )
    if checkpoint_value["prior_stage_source_seal_sha256"] != expected_prior_hash:
        raise SecGemmaLeanV37SourceError(
            "Compact replay checkpoint prior hash differs"
        )
    receipt_body = {
        "schema_version": COMPACT_REPLAY_SCHEMA_VERSION,
        "stage": config.stage,
        "checkpoint_sha256": checkpoint_value["checkpoint_sha256"],
        "role_manifest_count": len(manifests),
        "role_manifests_sha256": checkpoint_value["role_manifests_sha256"],
        "stage_source_seal_sha256": output.stage_source_seal_sha256,
        "exact_source_seal_match": True,
        "peak_live_role_payload_count": 1,
        "fresh_network_provenance_claimed": False,
        "main_parse_receipt_sha256": checkpoint_value[
            "main_parse_receipt_sha256"
        ],
        "submissions_snapshot_receipt_sha256": checkpoint_value[
            "submissions_snapshot_receipt_sha256"
        ],
        "reconciliation_and_prior_chain_receipt_sha256": checkpoint_value[
            "reconciliation_and_prior_chain_receipt_sha256"
        ],
        "receipt_validation_boundary": (
            "opaque_hashes_bound_only_acquisition_validates_journal_transport_files"
        ),
    }
    receipt = _canonical_snapshot(
        {**receipt_body, "compact_replay_sha256": _canonical_sha256(receipt_body)}
    )
    return output, receipt


def detached_replay(
    stage: str | StageSourceConfig,
    checkpoint: Mapping[str, Any],
    role_manifests: Sequence[Mapping[str, Any]],
    load_blob: SourceBlobLoader,
    prior: CompactReplayInput | None = None,
) -> Mapping[str, Any]:
    """Sequentially replay disk-backed compact evidence one role blob at a time."""

    return rehydrate_compact_stage(
        stage=stage,
        checkpoint=checkpoint,
        role_manifests=role_manifests,
        load_blob=load_blob,
        prior=prior,
    ).receipt


def rehydrate_compact_stage(
    stage: str | StageSourceConfig,
    checkpoint: Mapping[str, Any],
    role_manifests: Sequence[Mapping[str, Any]],
    load_blob: SourceBlobLoader,
    prior: CompactReplayInput | None = None,
) -> CompactReplayResult:
    """Rehydrate authenticated compact stage carry state after a restart."""

    config = _stage_config(stage)
    stage_output, receipt = _detached_replay_compact_core(
        config=config,
        checkpoint=_canonical_snapshot(checkpoint),
        role_manifests=[_canonical_snapshot(item) for item in role_manifests],
        load_blob=load_blob,
        prior=prior,
    )
    return CompactReplayResult(stage_output=stage_output, receipt=receipt)


# A shorter compatibility spelling for callers that treat replay validation as
# a predicate-producing audit operation.
validate_detached_stage_source_replay = detached_replay_stage_source


__all__ = [
    "COMPACT_CHECKPOINT_SCHEMA_VERSION",
    "COMPACT_COMPLETE_SCHEMA_VERSION",
    "COMPACT_RECONCILIATION_SCHEMA_VERSION",
    "COMPACT_REPLAY_SCHEMA_VERSION",
    "COMPACT_ROLE_EVIDENCE_SCHEMA_VERSION",
    "COMPACT_ROLE_MANIFEST_SCHEMA_VERSION",
    "COMPACT_SUBMISSIONS_SCHEMA_VERSION",
    "DETACHED_REPLAY_SCHEMA_VERSION",
    "COMPLETED_YEAR_COVERAGE_SCHEMA_VERSION",
    "GLOBAL_AVAILABILITY_END",
    "GLOBAL_AVAILABILITY_START",
    "LEGACY_SCIENCE_PROJECTION_SCHEMA_VERSION",
    "MAIN_SUBMISSIONS_NAME",
    "MAIN_SUBMISSIONS_URL",
    "MASTER_HEADER",
    "PRE_MASTER_UPPER_BOUND_SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "STAGE_SOURCE_CONFIGS",
    "CompleteSourceEvidence",
    "CompactReplayInput",
    "CompactReplayResult",
    "HistoricalReference",
    "LegacyScienceDocument",
    "LegacyScienceProjection",
    "MasterQuarterEvidence",
    "MasterReconciliation",
    "MasterTargetRow",
    "QuarterRole",
    "ReconciledTarget",
    "ReconciliationOutput",
    "RoleParseOutput",
    "SecGemmaLeanV37SourceError",
    "StageSourceBundle",
    "StageSourceConfig",
    "StageOutput",
    "PhaseOutput",
    "SourceBlobLoader",
    "SubmissionRow",
    "SubmissionsSnapshot",
    "SubmissionsSource",
    "build_legacy_science_projection",
    "build_compact_checkpoint",
    "build_compact_role_manifest",
    "build_pre_master_submissions_upper_bound",
    "build_stage_source_seal",
    "derive_complete_submission_plan",
    "detached_replay",
    "detached_replay_stage_source",
    "parse_strict_master_gzip",
    "parse_complete",
    "parse_historical",
    "parse_main",
    "parse_master",
    "parse_submissions_snapshot",
    "quarter_roles",
    "finalize_stage",
    "finalize_submissions",
    "reconcile_complete_submission",
    "reconcile_master_exact_set",
    "reconcile_masters",
    "rehydrate_compact_stage",
    "validate_cross_stage_frozen_prefix",
    "validate_completed_year_source_coverage",
    "validate_detached_stage_source_replay",
    "validate_stage_source_seal",
]
