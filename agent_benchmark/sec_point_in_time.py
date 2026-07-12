"""Pure SEC EDGAR point-in-time audit primitives; never performs I/O."""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
from zoneinfo import ZoneInfo

AAPL_CIK = "0000320193"
MAX_AUDIT_REQUESTS = 100
MAX_AUDIT_BYTES = 250 * 1024 * 1024
MAX_AUDIT_SECONDS = 1800.0
_ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}\Z")
_ACCEPTANCE_RE = re.compile(
    rb"<ACCEPTANCE-DATETIME>\s*([0-9]{14})(?=\s|<|$)", re.IGNORECASE
)
_EMAIL_RE = re.compile(
    r"(?<![A-Z0-9._%+-])([A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,}))(?![A-Z0-9._%+-])",
    re.IGNORECASE,
)
_PLACEHOLDERS = ("example", "sample company", "placeholder", "change me", "your email")
_EASTERN = ZoneInfo("America/New_York")
class SecPointInTimeError(ValueError):
    """SEC audit evidence violated its bounded contract."""
class SecAuditLimitError(SecPointInTimeError):
    """A frozen audit resource ceiling was exceeded."""
@dataclass(frozen=True)
class UserAgentAudit:
    sha256: str
    real_contact_validated: bool = True

    @property
    def redacted(self) -> str:
        return f"validated-sec-user-agent ({self.sha256})"
def content_sha256(payload: bytes | bytearray | memoryview) -> str:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    return f"sha256:{hashlib.sha256(bytes(payload)).hexdigest()}"
def validate_sec_user_agent(user_agent: str) -> UserAgentAudit:
    """Return only a hash and validation flag, never the raw contact."""
    if type(user_agent) is not str or not user_agent or user_agent != user_agent.strip():
        raise SecPointInTimeError("SEC User-Agent must identify a contact")
    value = user_agent
    lowered = value.lower()
    matches = list(_EMAIL_RE.finditer(value))
    identity = value[: matches[0].start()].strip(" /;:-") if matches else ""
    domain = matches[0].group(2).lower() if matches else ""
    invalid = (
        len(value) > 512
        or len(matches) != 1
        or not re.search(r"[A-Za-z]{2}", identity)
        or any(token in lowered for token in _PLACEHOLDERS)
        or domain in {"example.com", "example.org", "example.net", "localhost"}
        or domain.endswith((".invalid", ".test"))
        or any(character in value for character in "<>\r\n")
    )
    if invalid:
        raise SecPointInTimeError("SEC User-Agent must contain a real identity and email")
    return UserAgentAudit(content_sha256(value.encode("utf-8")))
@dataclass(frozen=True)
class FilingRecord:
    accession_number: str
    acceptance_datetime: str
    form: str
    primary_document: str
    items: str
    filing_date: str
    report_date: str
    is_xbrl: bool
    source_name: str
    subject_cik: str = AAPL_CIK
    date_of_filing_date_change: str = ""

    def __post_init__(self) -> None:
        if not _ACCESSION_RE.fullmatch(self.accession_number):
            raise SecPointInTimeError("SEC submission has an invalid accession")
        if not isinstance(self.acceptance_datetime, str):
            raise SecPointInTimeError("SEC acceptance datetime must remain text")
        _parse_submissions_acceptance(self.acceptance_datetime)
        _validate_iso_date(self.filing_date, "filingDate", allow_empty=False)
        _validate_iso_date(self.report_date, "reportDate", allow_empty=True)
        _validate_iso_date(
            self.date_of_filing_date_change,
            "dateOfFilingDateChange",
            allow_empty=True,
        )

    @property
    def submitter_cik(self) -> str:
        return self.accession_number[:10]

    @property
    def is_amendment(self) -> bool:
        return self.form.upper().endswith("/A")

    @property
    def change_anomaly_flag(self) -> bool:
        """A preserved SEC filing-date change is objective anomaly evidence."""

        return bool(self.date_of_filing_date_change)


def _validate_iso_date(value: str, name: str, *, allow_empty: bool) -> None:
    if not isinstance(value, str) or (not value and not allow_empty):
        raise SecPointInTimeError(f"{name} must be a canonical ISO date")
    if not value and allow_empty:
        return
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SecPointInTimeError(f"{name} must be a canonical ISO date") from exc
    if parsed.date().isoformat() != value:
        raise SecPointInTimeError(f"{name} must be a canonical ISO date")


def _parse_submissions_acceptance(value: str) -> datetime:
    """Validate SEC's exact SGML digits or timezone-qualified JSON datetime."""

    if re.fullmatch(r"[0-9]{14}", value):
        try:
            return datetime.strptime(value, "%Y%m%d%H%M%S").replace(
                tzinfo=_EASTERN
            )
        except ValueError as exc:
            raise SecPointInTimeError(
                "acceptanceDateTime is not a valid 14-digit timestamp"
            ) from exc
    if "T" not in value:
        raise SecPointInTimeError(
            "acceptanceDateTime must be 14 digits or timezone-qualified ISO datetime"
        )
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SecPointInTimeError(
            "acceptanceDateTime must be 14 digits or timezone-qualified ISO datetime"
        ) from exc
    if parsed.tzinfo is None:
        raise SecPointInTimeError(
            "ISO acceptanceDateTime must contain an explicit timezone"
        )
    return parsed


def parse_submissions_acceptance_datetime(value: str) -> datetime:
    """Return a timezone-aware instant from SEC Submissions metadata."""

    if not isinstance(value, str):
        raise SecPointInTimeError("SEC acceptance datetime must remain text")
    return _parse_submissions_acceptance(value)
def _as_bool(value: Any, name: str) -> bool:
    if value in (True, 1, "1"):
        return True
    if value in (False, 0, "0", ""):
        return False
    raise SecPointInTimeError(f"{name} must be an exact zero/one flag")
def _column_rows(columns: Mapping[str, Any], source: str) -> list[FilingRecord]:
    required = (
        "accessionNumber", "acceptanceDateTime", "form", "primaryDocument",
        "items", "filingDate", "reportDate",
    )
    missing = [name for name in required if name not in columns]
    if missing:
        raise SecPointInTimeError(f"SEC submissions columns missing: {missing}")
    accessions = columns["accessionNumber"]
    if not isinstance(accessions, Sequence) or isinstance(accessions, (str, bytes)):
        raise SecPointInTimeError("SEC submissions columns must be arrays")
    size = len(accessions)
    for values in columns.values():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise SecPointInTimeError("SEC submissions columns must be arrays")
        if len(values) != size:
            raise SecPointInTimeError("SEC submissions columns are not row-aligned")
    xbrl = columns.get("isXBRL", [0] * size)
    filing_date_changes = columns.get("dateOfFilingDateChange", [""] * size)
    if (
        not isinstance(filing_date_changes, Sequence)
        or isinstance(filing_date_changes, (str, bytes))
        or len(filing_date_changes) != size
    ):
        raise SecPointInTimeError(
            "dateOfFilingDateChange must be an optional row-aligned array"
        )
    result: list[FilingRecord] = []
    for position in range(size):
        row = {name: columns[name][position] for name in required}
        if not all(isinstance(row[name], str) for name in required):
            raise SecPointInTimeError("SEC submission text fields must remain strings")
        accession = row["accessionNumber"]
        if not _ACCESSION_RE.fullmatch(accession) or not row["form"]:
            raise SecPointInTimeError("SEC submission has an invalid identity")
        change_date = filing_date_changes[position]
        if not isinstance(change_date, str):
            raise SecPointInTimeError(
                "dateOfFilingDateChange values must remain strings"
            )
        result.append(
            FilingRecord(
                accession_number=accession,
                acceptance_datetime=row["acceptanceDateTime"],
                form=row["form"],
                primary_document=row["primaryDocument"],
                items=row["items"],
                filing_date=row["filingDate"],
                report_date=row["reportDate"],
                is_xbrl=_as_bool(xbrl[position], "isXBRL"),
                source_name=source,
                date_of_filing_date_change=change_date,
            )
        )
    return result
def parse_submissions_rows(
    main_payload: Mapping[str, Any],
    historical_payloads: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[FilingRecord, ...]:
    """Parse Apple main and referenced historical columnar JSON payloads."""
    if not isinstance(main_payload, Mapping):
        raise SecPointInTimeError("main submissions payload must be an object")
    if str(main_payload.get("cik", "")).zfill(10) != AAPL_CIK:
        raise SecPointInTimeError("submissions payload is not Apple CIK 0000320193")
    filings = main_payload.get("filings")
    if not isinstance(filings, Mapping) or not isinstance(filings.get("recent"), Mapping):
        raise SecPointInTimeError("main submissions payload lacks filings.recent")
    rows = _column_rows(filings["recent"], f"CIK{AAPL_CIK}.json")
    provided = dict(historical_payloads or {})
    references = filings.get("files", [])
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise SecPointInTimeError("filings.files must be an array")
    names: list[str] = []
    for item in references:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise SecPointInTimeError("historical submissions reference is invalid")
        name = item["name"]
        if PurePosixPath(name).name != name or not name.endswith(".json"):
            raise SecPointInTimeError("historical submissions filename is unsafe")
        names.append(name)
    if set(names) != set(provided):
        raise SecPointInTimeError("historical submissions payload set is not exact")
    for name in names:
        payload = provided[name]
        if not isinstance(payload, Mapping):
            raise SecPointInTimeError("historical submissions payload must be an object")
        rows.extend(_column_rows(payload, name))
    accessions = [row.accession_number for row in rows]
    if len(accessions) != len(set(accessions)):
        raise SecPointInTimeError("duplicate accession across submissions payloads")
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.filing_date,
                _parse_submissions_acceptance(row.acceptance_datetime).astimezone(
                    timezone.utc
                ),
                row.accession_number,
            ),
        )
    )
@dataclass(frozen=True)
class MasterIndexRecord:
    cik: str
    company_name: str
    form: str
    filing_date: str
    filename: str
    accession_number: str
def parse_master_idx(payload: bytes | str) -> tuple[MasterIndexRecord, ...]:
    text = payload.decode("latin-1") if isinstance(payload, bytes) else payload
    if not isinstance(text, str):
        raise TypeError("master.idx payload must be bytes or text")
    lines = text.splitlines()
    try:
        start = lines.index("CIK|Company Name|Form Type|Date Filed|Filename") + 1
    except ValueError as exc:
        raise SecPointInTimeError("master.idx header is missing") from exc
    result: list[MasterIndexRecord] = []
    reader = csv.reader(io.StringIO("\n".join(lines[start:])), delimiter="|")
    for values in reader:
        if not values or (len(values) == 1 and set(values[0]) <= {"-"}):
            continue
        if len(values) != 5:
            raise SecPointInTimeError("master.idx row does not have five fields")
        cik, company, form, filed, filename = values
        accession = re.search(r"(\d{10}-\d{2}-\d{6})\.txt\Z", filename)
        try:
            normalized_cik = str(int(cik)).zfill(10)
            datetime.strptime(filed, "%Y-%m-%d")
        except (ValueError, TypeError) as exc:
            raise SecPointInTimeError("master.idx row has an invalid CIK or date") from exc
        if accession is None:
            raise SecPointInTimeError("master.idx row has no stable accession path")
        result.append(MasterIndexRecord(
            normalized_cik, company, form, filed, filename, accession.group(1)
        ))
    return tuple(result)
def parse_acceptance_datetime(payload: bytes | str) -> datetime:
    if not isinstance(payload, (bytes, str)):
        raise TypeError("acceptance payload must be bytes or text")
    raw = payload.encode("latin-1") if isinstance(payload, str) else payload
    direct = re.fullmatch(rb"[0-9]{14}", raw.strip())
    matches = [direct.group(0)] if direct else _ACCEPTANCE_RE.findall(raw)
    unique = set(matches)
    if len(unique) != 1:
        raise SecPointInTimeError("SGML must contain one unambiguous ACCEPTANCE-DATETIME")
    try:
        return datetime.strptime(unique.pop().decode("ascii"), "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise SecPointInTimeError("ACCEPTANCE-DATETIME is not a valid timestamp") from exc
@dataclass(frozen=True)
class ArchiveUrls:
    directory: str
    complete_submission: str
    header: str
    index_json: str
    primary_document: str | None
def archive_urls(record: FilingRecord) -> ArchiveUrls:
    if record.subject_cik != AAPL_CIK or not _ACCESSION_RE.fullmatch(record.accession_number):
        raise SecPointInTimeError("filing identity is outside the frozen Apple scope")
    accession = record.accession_number
    base = (
        f"https://www.sec.gov/Archives/edgar/data/{int(AAPL_CIK)}/"
        f"{accession.replace('-', '')}"
    )
    primary: str | None = None
    if record.primary_document:
        name = record.primary_document
        if PurePosixPath(name).name != name or "\\" in name or ".." in name:
            raise SecPointInTimeError("primary document filename is unsafe")
        primary = f"{base}/{quote(name, safe='-._~()')}"
    return ArchiveUrls(
        f"{base}/", f"{base}/{accession}.txt", f"{base}/{accession}.hdr.sgml",
        f"{base}/index.json", primary,
    )
@dataclass
class BudgetCounter:
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    max_requests: int = MAX_AUDIT_REQUESTS
    max_bytes: int = MAX_AUDIT_BYTES
    max_seconds: float = MAX_AUDIT_SECONDS
    requests: int = field(init=False, default=0)
    bytes_received: int = field(init=False, default=0)
    _started_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.max_requests <= MAX_AUDIT_REQUESTS:
            raise SecAuditLimitError("request ceiling may only be tightened")
        if not 1 <= self.max_bytes <= MAX_AUDIT_BYTES:
            raise SecAuditLimitError("byte ceiling may only be tightened")
        if not 0 < self.max_seconds <= MAX_AUDIT_SECONDS:
            raise SecAuditLimitError("time ceiling may only be tightened")
        self._started_at = float(self.clock())
        self.check()

    def check(self) -> None:
        elapsed = float(self.clock()) - self._started_at
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed > self.max_seconds:
            raise SecAuditLimitError("SEC audit wall-clock ceiling exceeded")

    def begin_request(self) -> None:
        self.check()
        if self.requests >= self.max_requests:
            raise SecAuditLimitError("SEC audit request ceiling exceeded")
        self.requests += 1

    def add_bytes(self, count: int) -> None:
        self.check()
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise SecAuditLimitError("response byte count must be a non-negative integer")
        if self.bytes_received + count > self.max_bytes:
            raise SecAuditLimitError("SEC audit byte ceiling exceeded")
        self.bytes_received += count

    def snapshot(self) -> dict[str, int | float]:
        self.check()
        elapsed = float(self.clock()) - self._started_at
        return {
            "requests": self.requests, "bytes_received": self.bytes_received,
            "max_requests": self.max_requests, "max_bytes": self.max_bytes,
            "max_seconds": self.max_seconds, "elapsed_seconds": elapsed,
        }
__all__ = [
    "AAPL_CIK", "MAX_AUDIT_BYTES", "MAX_AUDIT_REQUESTS", "MAX_AUDIT_SECONDS",
    "ArchiveUrls", "BudgetCounter", "FilingRecord", "MasterIndexRecord",
    "SecAuditLimitError", "SecPointInTimeError", "UserAgentAudit", "archive_urls",
    "content_sha256", "parse_acceptance_datetime", "parse_master_idx",
    "parse_submissions_acceptance_datetime", "parse_submissions_rows",
    "validate_sec_user_agent",
]
