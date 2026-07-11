"""Pure SEC complete-submission parsing and point-in-time reconciliation.

The functions in this module operate only on caller-supplied bytes, strings,
metadata objects, and session dates.  They never perform filesystem or network
I/O.  Ambiguous identities, document sections, or archive metadata fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence
import unicodedata
from zoneinfo import ZoneInfo

from .sec_point_in_time import (
    FilingRecord,
    MasterIndexRecord,
    SecPointInTimeError,
    content_sha256,
    parse_acceptance_datetime,
    parse_submissions_acceptance_datetime,
)


CONTRACT_VERSION = "aapl-sec-filing-content-audit-v1"
MINIMUM_USABLE_TEXT_CHARACTERS = 500
EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SGMLHeader:
    accession_number: str
    acceptance_datetime: str | None
    form: str
    filing_date: str
    filer_cik: str
    filer_company: str
    subject_cik: str
    subject_company: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SGMLDocument:
    document_type: str
    sequence: int
    filename: str
    description: str
    text: str
    text_sha256: str

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_text:
            result.pop("text")
        return result


@dataclass(frozen=True)
class CompleteSubmission:
    header: SGMLHeader
    documents: tuple[SGMLDocument, ...]
    submission_sha256: str

    def to_dict(self, *, include_document_text: bool = False) -> dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "documents": [
                document.to_dict(include_text=include_document_text)
                for document in self.documents
            ],
            "submission_sha256": self.submission_sha256,
        }


@dataclass(frozen=True)
class SECIndexItem:
    name: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SECIndex:
    directory_name: str
    parent_directory: str
    items: tuple[SECIndexItem, ...]
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory_name": self.directory_name,
            "parent_directory": self.parent_directory,
            "items": [item.to_dict() for item in self.items],
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class NormalizedText:
    text: str
    sha256: str
    character_count: int
    usable: bool
    minimum_usable_characters: int = MINIMUM_USABLE_TEXT_CHARACTERS

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_text:
            result.pop("text")
        return result


def _coerce_payload(payload: bytes | str, *, field_name: str) -> tuple[str, bytes, str]:
    if isinstance(payload, bytes):
        return payload.decode("latin-1"), payload, "latin-1"
    if isinstance(payload, str):
        return payload, payload.encode("utf-8"), "utf-8"
    raise TypeError(f"{field_name} must be bytes or text")


def _canonical_cik(value: Any, *, field_name: str) -> str:
    raw = str(value or "").strip()
    if not raw.isdigit() or len(raw) > 10:
        raise SecPointInTimeError(f"{field_name} must be a valid SEC CIK")
    return str(int(raw)).zfill(10)


def _canonical_form(value: Any) -> str:
    form = " ".join(str(value or "").upper().split())
    if not form:
        raise SecPointInTimeError("SEC form cannot be empty")
    return form


def _safe_filename(value: Any, *, field_name: str) -> str:
    name = str(value or "").strip()
    if (
        not name
        or PurePosixPath(name).name != name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise SecPointInTimeError(f"{field_name} is not a safe archive filename")
    return name


def _unique_match(text: str, patterns: Sequence[str], *, field_name: str) -> str:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE))
    values = {str(value).strip() for value in matches if str(value).strip()}
    if len(values) != 1:
        raise SecPointInTimeError(f"SGML header has ambiguous or missing {field_name}")
    return values.pop()


_ROLE_MARKERS = {
    "FILER": "filer",
    "FILED BY": "filed_by",
    "FILED-BY": "filed_by",
    "SUBJECT COMPANY": "subject",
    "SUBJECT-COMPANY": "subject",
    "REPORTING-OWNER": "reporting_owner",
    "REPORTING OWNER": "reporting_owner",
}


def _role_marker(line: str) -> str | None:
    value = line.strip().strip("<>").rstrip(":").strip().upper().replace("-", "-")
    return _ROLE_MARKERS.get(value)


def _header_entities(header_text: str) -> dict[str, tuple[str, str]]:
    values: dict[str, dict[str, set[str]]] = {}
    active_role: str | None = None
    for line in header_text.splitlines():
        marker = _role_marker(line)
        if marker is not None:
            active_role = marker
            values.setdefault(marker, {"cik": set(), "company": set()})
            continue
        if active_role is None:
            continue
        cik = re.search(
            r"(?:<CENTRAL-INDEX-KEY>|<CIK>|CENTRAL INDEX KEY:)\s*([0-9]{1,10})",
            line,
            flags=re.IGNORECASE,
        )
        if cik:
            values[active_role]["cik"].add(
                _canonical_cik(cik.group(1), field_name=f"{active_role} CIK")
            )
        company = re.search(
            r"(?:<COMPANY-CONFORMED-NAME>|<CONFORMED-NAME>|COMPANY CONFORMED NAME:)\s*([^<\r\n]+)",
            line,
            flags=re.IGNORECASE,
        )
        if company and company.group(1).strip():
            values[active_role]["company"].add(company.group(1).strip())

    result: dict[str, tuple[str, str]] = {}
    for role, fields in values.items():
        if len(fields["cik"]) > 1 or len(fields["company"]) > 1:
            raise SecPointInTimeError(f"SGML header has ambiguous {role} identity")
        if fields["cik"] or fields["company"]:
            if len(fields["cik"]) != 1 or len(fields["company"]) != 1:
                raise SecPointInTimeError(f"SGML header has incomplete {role} identity")
            result[role] = (next(iter(fields["cik"])), next(iter(fields["company"])))
    return result


def _parse_header(
    header_text: str,
    *,
    allow_missing_acceptance: bool = False,
) -> SGMLHeader:
    acceptance_matches = set(
        re.findall(
            r"<ACCEPTANCE-DATETIME>\s*([0-9]{14})(?=\s|<|$)",
            header_text,
            flags=re.IGNORECASE,
        )
    )
    if not acceptance_matches and allow_missing_acceptance:
        acceptance: datetime | None = None
    else:
        # Ambiguous values always fail, including in tolerant timestamp mode.
        acceptance = parse_acceptance_datetime(header_text)
    accession = _unique_match(
        header_text,
        (
            r"<ACCESSION-NUMBER>\s*([^\s<]+)",
            r"^\s*ACCESSION NUMBER:\s*([^\s]+)\s*$",
        ),
        field_name="accession number",
    )
    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        raise SecPointInTimeError("SGML header accession number is invalid")
    form = _canonical_form(
        _unique_match(
            header_text,
            (
                r"<CONFORMED-SUBMISSION-TYPE>\s*([^\r\n<]+)",
                r"<TYPE>\s*([^\r\n<]+)",
                r"^\s*CONFORMED SUBMISSION TYPE:\s*([^\r\n]+)$",
            ),
            field_name="submission form",
        )
    )
    filing_raw = _unique_match(
        header_text,
        (
            r"<FILED-AS-OF-DATE>\s*([0-9]{8})",
            r"<FILING-DATE>\s*([0-9]{8})",
            r"^\s*FILED AS OF DATE:\s*([0-9]{8})\s*$",
        ),
        field_name="filing date",
    )
    try:
        filing_date = datetime.strptime(filing_raw, "%Y%m%d").date().isoformat()
    except ValueError as exc:
        raise SecPointInTimeError("SGML header filing date is invalid") from exc

    entities = _header_entities(header_text)
    filer = entities.get("filed_by") or entities.get("filer") or entities.get(
        "reporting_owner"
    )
    if filer is None:
        raise SecPointInTimeError("SGML header has no unambiguous filer identity")
    subject = entities.get("subject") or entities.get("filer")
    if subject is None:
        raise SecPointInTimeError("SGML header has no unambiguous subject identity")
    return SGMLHeader(
        accession_number=accession,
        acceptance_datetime=(
            acceptance.strftime("%Y%m%d%H%M%S")
            if acceptance is not None
            else None
        ),
        form=form,
        filing_date=filing_date,
        filer_cik=filer[0],
        filer_company=filer[1],
        subject_cik=subject[0],
        subject_company=subject[1],
    )


def _document_field(block: str, tag: str, *, required: bool) -> str:
    matches = re.findall(
        rf"^\s*<{re.escape(tag)}>\s*([^\r\n<]*)\s*$",
        block,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    values = {value.strip() for value in matches if value.strip()}
    if not values and not required:
        return ""
    if len(values) != 1:
        raise SecPointInTimeError(f"DOCUMENT has ambiguous or missing {tag}")
    return values.pop()


def parse_complete_submission(
    payload: bytes | str,
    *,
    allow_missing_acceptance: bool = False,
) -> CompleteSubmission:
    """Parse one complete-submission SGML payload without external access."""

    text, raw, source_encoding = _coerce_payload(
        payload, field_name="complete-submission payload"
    )
    header_matches = re.findall(
        r"<SEC-HEADER>(.*?)</SEC-HEADER>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if len(header_matches) != 1:
        raise SecPointInTimeError("Complete submission must contain one SEC-HEADER")
    header = _parse_header(
        header_matches[0],
        allow_missing_acceptance=allow_missing_acceptance,
    )
    submission_accessions = {
        value
        for value in re.findall(
            r"<SEC-DOCUMENT>[^\r\n<]*?(\d{10}-\d{2}-\d{6})(?:\.txt)?",
            text,
            flags=re.IGNORECASE,
        )
    }
    if submission_accessions != {header.accession_number}:
        raise SecPointInTimeError(
            "SEC-DOCUMENT accession does not match the SGML header"
        )
    document_blocks = re.findall(
        r"<DOCUMENT>(.*?)</DOCUMENT>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not document_blocks:
        raise SecPointInTimeError("Complete submission contains no DOCUMENT sections")
    if len(re.findall(r"<DOCUMENT>", text, flags=re.IGNORECASE)) != len(
        document_blocks
    ) or len(re.findall(r"</DOCUMENT>", text, flags=re.IGNORECASE)) != len(
        document_blocks
    ):
        raise SecPointInTimeError("Complete submission has unbalanced DOCUMENT sections")

    documents: list[SGMLDocument] = []
    sequences: set[int] = set()
    filenames: set[str] = set()
    for block in document_blocks:
        text_matches = list(
            re.finditer(
                r"<TEXT>(.*?)</TEXT>",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        if len(text_matches) != 1:
            raise SecPointInTimeError("DOCUMENT must contain one unambiguous TEXT section")
        metadata_block = block[: text_matches[0].start()]
        document_type = _canonical_form(
            _document_field(metadata_block, "TYPE", required=True)
        )
        sequence_raw = _document_field(metadata_block, "SEQUENCE", required=True)
        if not sequence_raw.isdigit() or int(sequence_raw) < 1:
            raise SecPointInTimeError("DOCUMENT SEQUENCE must be a positive integer")
        sequence = int(sequence_raw)
        filename = _safe_filename(
            _document_field(metadata_block, "FILENAME", required=True),
            field_name="DOCUMENT filename",
        )
        description = _document_field(metadata_block, "DESCRIPTION", required=False)
        document_text = text_matches[0].group(1)
        if sequence in sequences or filename in filenames:
            raise SecPointInTimeError("DOCUMENT sequences and filenames must be unique")
        sequences.add(sequence)
        filenames.add(filename)
        encoded_text = document_text.encode(source_encoding)
        documents.append(
            SGMLDocument(
                document_type=document_type,
                sequence=sequence,
                filename=filename,
                description=description,
                text=document_text,
                text_sha256=content_sha256(encoded_text),
            )
        )
    documents.sort(key=lambda document: (document.sequence, document.filename))
    return CompleteSubmission(
        header=header,
        documents=tuple(documents),
        submission_sha256=content_sha256(raw),
    )


def parse_sec_index_json(payload: bytes | str | Mapping[str, Any]) -> SECIndex:
    """Parse SEC directory index metadata, preserving only names and sizes."""

    if isinstance(payload, Mapping):
        data = dict(payload)
        encoded = json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    else:
        if isinstance(payload, bytes):
            encoded = payload
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SecPointInTimeError("SEC index payload is not UTF-8 JSON") from exc
        elif isinstance(payload, str):
            text = payload
            encoded = payload.encode("utf-8")
        else:
            raise TypeError("SEC index payload must be bytes, text, or a mapping")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SecPointInTimeError("SEC index payload is not valid JSON") from exc
    directory = data.get("directory") if isinstance(data, Mapping) else None
    if not isinstance(directory, Mapping):
        raise SecPointInTimeError("SEC index lacks a directory object")
    directory_name = str(directory.get("name") or "").strip()
    parent_directory = str(directory.get("parent-dir") or "").strip()
    raw_items = directory.get("item")
    if not isinstance(raw_items, list):
        raise SecPointInTimeError("SEC index directory.item must be an array")
    items: list[SECIndexItem] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise SecPointInTimeError("SEC index item must be an object")
        name = _safe_filename(raw_item.get("name"), field_name="SEC index item name")
        raw_size = raw_item.get("size")
        if isinstance(raw_size, bool):
            raise SecPointInTimeError("SEC index item size must be a non-negative integer")
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise SecPointInTimeError(
                "SEC index item size must be a non-negative integer"
            ) from exc
        if size < 0 or str(raw_size).strip() != str(size):
            raise SecPointInTimeError(
                "SEC index item size must be a canonical non-negative integer"
            )
        if name in seen:
            raise SecPointInTimeError("SEC index contains duplicate item names")
        seen.add(name)
        items.append(SECIndexItem(name=name, size=size))
    items.sort(key=lambda item: item.name)
    return SECIndex(
        directory_name=directory_name,
        parent_directory=parent_directory,
        items=tuple(items),
        payload_sha256=content_sha256(encoded),
    )


def select_primary_document(
    record: FilingRecord,
    submission: CompleteSubmission,
    index: SECIndex,
) -> tuple[SGMLDocument, SECIndexItem]:
    """Select the sole exact SGML/index match to metadata.primary_document."""

    filename = _safe_filename(
        record.primary_document, field_name="FilingRecord primary_document"
    )
    documents = [document for document in submission.documents if document.filename == filename]
    items = [item for item in index.items if item.name == filename]
    if len(documents) != 1 or len(items) != 1:
        raise SecPointInTimeError(
            "Primary document must match exactly one SGML document and one index item"
        )
    if items[0].size <= 0:
        raise SecPointInTimeError("Primary document index size must be positive")
    return documents[0], items[0]


class _VisibleTextParser(HTMLParser):
    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "caption", "div",
        "dl", "dt", "dd", "figcaption", "figure", "footer", "h1", "h2",
        "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav",
        "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot",
        "th", "thead", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._excluded_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style"}:
            self._excluded_depth += 1
        elif self._excluded_depth == 0 and lowered in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style"}:
            if self._excluded_depth > 0:
                self._excluded_depth -= 1
        elif self._excluded_depth == 0 and lowered in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._excluded_depth == 0:
            self.parts.append(data)


def normalize_filing_text(value: str) -> NormalizedText:
    """Return deterministic visible text with scripts/styles excluded."""

    if not isinstance(value, str):
        raise TypeError("filing document text must be a string")
    parser = _VisibleTextParser()
    try:
        parser.feed(value)
        parser.close()
    except Exception as exc:  # pragma: no cover - HTMLParser is deliberately lenient
        raise SecPointInTimeError("Could not normalize filing document text") from exc
    visible = unicodedata.normalize("NFKC", "".join(parser.parts))
    visible = visible.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in visible.split("\n"):
        cleaned = "".join(
            "" if unicodedata.category(character) == "Cf" else character
            for character in line
        )
        cleaned = " ".join(cleaned.split())
        if cleaned:
            lines.append(cleaned)
    normalized = "\n".join(lines)
    character_count = len(normalized)
    return NormalizedText(
        text=normalized,
        sha256=content_sha256(normalized.encode("utf-8")),
        character_count=character_count,
        usable=character_count >= MINIMUM_USABLE_TEXT_CHARACTERS,
    )


def _master_value(master: MasterIndexRecord | Mapping[str, Any], name: str) -> Any:
    if isinstance(master, Mapping):
        return master.get(name)
    return getattr(master, name)


def _normalized_sessions(values: Sequence[date | datetime | str]) -> tuple[date, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("AAPL sessions must be a sequence of dates")
    sessions: list[date] = []
    for value in values:
        if isinstance(value, datetime):
            parsed = value.date()
        elif isinstance(value, date):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = date.fromisoformat(value[:10])
            except ValueError as exc:
                raise SecPointInTimeError("AAPL session contains an invalid date") from exc
        else:
            raise SecPointInTimeError("AAPL session contains an invalid date")
        sessions.append(parsed)
    if len(sessions) != len(set(sessions)):
        raise SecPointInTimeError("AAPL sessions contain duplicate dates")
    return tuple(sorted(sessions))


def conservative_availability_session(
    acceptance: datetime,
    aapl_sessions: Sequence[date | datetime | str],
    *,
    filing_date: str | None = None,
    date_of_filing_date_change: str | None = None,
) -> str | None:
    """Return the first session after the latest defensible catalogue date."""

    if not isinstance(acceptance, datetime):
        raise TypeError("acceptance must be a datetime")
    sessions = _normalized_sessions(aapl_sessions)
    boundaries = [acceptance.date()]
    for label, value in (
        ("filing_date", filing_date),
        ("date_of_filing_date_change", date_of_filing_date_change),
    ):
        if value:
            try:
                boundaries.append(date.fromisoformat(value))
            except (TypeError, ValueError) as exc:
                raise SecPointInTimeError(
                    f"{label} is not a canonical ISO date"
                ) from exc
    not_before = max(boundaries)
    selected = next((session for session in sessions if session > not_before), None)
    return selected.isoformat() if selected is not None else None


def _validate_index_identity(index: SECIndex, record: FilingRecord) -> None:
    accession_compact = record.accession_number.replace("-", "")
    path_parts: list[str] = []
    for raw in (index.parent_directory, index.directory_name):
        path_parts.extend(part for part in PurePosixPath(raw).parts if part not in {"/", ""})
    if accession_compact not in path_parts:
        raise SecPointInTimeError("SEC index directory does not match accession")
    cik_candidates = {part for part in path_parts if part.isdigit() and len(part) <= 10}
    if not any(str(int(value)).zfill(10) == record.subject_cik for value in cik_candidates):
        raise SecPointInTimeError("SEC index directory does not match subject CIK")


def reconcile_filing_content(
    record: FilingRecord,
    master: MasterIndexRecord | Mapping[str, Any],
    submission: CompleteSubmission,
    index: SECIndex,
    aapl_sessions: Sequence[date | datetime | str],
) -> dict[str, Any]:
    """Reconcile all supplied SEC identities and return JSON-safe audit evidence."""

    if not isinstance(record, FilingRecord):
        raise TypeError("record must be a FilingRecord")
    header = submission.header
    master_accession = str(_master_value(master, "accession_number") or "")
    master_cik = _canonical_cik(_master_value(master, "cik"), field_name="master CIK")
    master_form = _canonical_form(_master_value(master, "form"))
    master_date = str(_master_value(master, "filing_date") or "")
    identities = {
        record.accession_number,
        master_accession,
        header.accession_number,
    }
    if len(identities) != 1:
        raise SecPointInTimeError("SEC accession does not reconcile")
    if record.subject_cik != master_cik or record.subject_cik != header.subject_cik:
        raise SecPointInTimeError("SEC subject CIK does not reconcile")
    forms = {_canonical_form(record.form), master_form, header.form}
    if len(forms) != 1:
        raise SecPointInTimeError("SEC form does not reconcile")
    dates = {record.filing_date, master_date, header.filing_date}
    if len(dates) != 1:
        raise SecPointInTimeError("SEC filing date does not reconcile")
    record_acceptance = parse_submissions_acceptance_datetime(
        record.acceptance_datetime
    )
    exact_acceptance_timestamp = header.acceptance_datetime is not None
    if exact_acceptance_timestamp:
        header_acceptance = parse_acceptance_datetime(
            header.acceptance_datetime
        ).replace(tzinfo=EASTERN)
        if record_acceptance.astimezone(
            timezone.utc
        ) != header_acceptance.astimezone(timezone.utc):
            raise SecPointInTimeError("SEC acceptance datetime does not reconcile")
    else:
        # Retrospective Submissions metadata still supplies a conservative date
        # boundary, but it is not counted as exact raw-SGML timestamp evidence.
        header_acceptance = record_acceptance.astimezone(EASTERN)

    master_filename = str(_master_value(master, "filename") or "")
    expected_master_tail = f"{record.accession_number}.txt"
    if PurePosixPath(master_filename).name != expected_master_tail:
        raise SecPointInTimeError("master.idx filename does not match accession")
    master_parts = [
        part for part in PurePosixPath(master_filename).parts if part not in {"/", ""}
    ]
    if not any(
        part.isdigit() and str(int(part)).zfill(10) == record.subject_cik
        for part in master_parts
    ):
        raise SecPointInTimeError("master.idx filename does not match subject CIK")
    _validate_index_identity(index, record)
    complete_name = f"{record.accession_number}.txt"
    complete_items = [item for item in index.items if item.name == complete_name]
    if len(complete_items) != 1 or complete_items[0].size <= 0:
        raise SecPointInTimeError(
            "Complete submission must match one positive-size index item"
        )
    primary, index_item = select_primary_document(record, submission, index)
    if primary.document_type != header.form:
        raise SecPointInTimeError(
            "Primary DOCUMENT type does not match reconciled SEC form"
        )
    normalized = normalize_filing_text(primary.text)
    embedded_document_bytes = primary.text.encode("latin-1").strip(b"\r\n")
    availability = conservative_availability_session(
        header_acceptance,
        aapl_sessions,
        filing_date=record.filing_date,
        date_of_filing_date_change=record.date_of_filing_date_change or None,
    )
    availability_not_before = max(
        value
        for value in (
            header_acceptance.date(),
            date.fromisoformat(record.filing_date),
            (
                date.fromisoformat(record.date_of_filing_date_change)
                if record.date_of_filing_date_change
                else header_acceptance.date()
            ),
        )
    )
    result = {
        "contract_version": CONTRACT_VERSION,
        "status": "usable" if normalized.usable else "unusable_text",
        "identity_reconciled": True,
        "accession_number": record.accession_number,
        "subject_cik": record.subject_cik,
        # The first ten accession digits identify the submitting entity, but
        # older company filings may list only the registrant in the SGML FILER
        # block.  Preserve and compare both identities; do not falsely require
        # a third-party accession submitter to appear as the header filer.
        "accession_submitter_cik": record.submitter_cik,
        "header_filer_cik": header.filer_cik,
        "header_filer_matches_accession_submitter": (
            header.filer_cik == record.submitter_cik
        ),
        "header_filer_matches_subject": header.filer_cik == record.subject_cik,
        "filer_cik": header.filer_cik,
        "subject_company": header.subject_company,
        "filer_company": header.filer_company,
        "form": header.form,
        "filing_date": header.filing_date,
        "acceptance_datetime_et": header.acceptance_datetime,
        "submissions_acceptance_datetime": record.acceptance_datetime,
        "exact_acceptance_timestamp": exact_acceptance_timestamp,
        "acceptance_timezone": "America/New_York",
        "availability_session": availability,
        "availability_session_found": availability is not None,
        "availability_not_before_date": availability_not_before.isoformat(),
        "availability_rule": (
            "first supplied AAPL session strictly after latest acceptance, filing, or filing-date-change date"
        ),
        "primary_document": {
            **primary.to_dict(include_text=False),
            "index_size": index_item.size,
            "embedded_document_bytes_length": len(embedded_document_bytes),
            "embedded_document_bytes_sha256": content_sha256(
                embedded_document_bytes
            ),
            "embedded_document_comparison_rule": (
                "latin-1 roundtrip with only outer CR/LF envelope removed"
            ),
        },
        "normalized_text": normalized.to_dict(include_text=True),
        "submission_sha256": submission.submission_sha256,
        "complete_submission_index_size": complete_items[0].size,
        "index_sha256": index.payload_sha256,
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def audit_filing_content(
    record: FilingRecord,
    master: MasterIndexRecord | Mapping[str, Any],
    complete_submission_payload: bytes | str,
    index_payload: bytes | str | Mapping[str, Any],
    aapl_sessions: Sequence[date | datetime | str],
    *,
    allow_missing_acceptance: bool = False,
) -> dict[str, Any]:
    """Parse and reconcile one caller-supplied filing in a single pure call."""

    submission = parse_complete_submission(
        complete_submission_payload,
        allow_missing_acceptance=allow_missing_acceptance,
    )
    index = parse_sec_index_json(index_payload)
    return reconcile_filing_content(record, master, submission, index, aapl_sessions)


def parse_index_json(payload: bytes | str | Mapping[str, Any]) -> SECIndex:
    """Compatibility spelling for :func:`parse_sec_index_json`."""

    return parse_sec_index_json(payload)


def normalize_document_text(value: str) -> NormalizedText:
    """Compatibility spelling for :func:`normalize_filing_text`."""

    return normalize_filing_text(value)


__all__ = [
    "CONTRACT_VERSION",
    "MINIMUM_USABLE_TEXT_CHARACTERS",
    "CompleteSubmission",
    "NormalizedText",
    "SECIndex",
    "SECIndexItem",
    "SGMLDocument",
    "SGMLHeader",
    "audit_filing_content",
    "conservative_availability_session",
    "normalize_filing_text",
    "normalize_document_text",
    "parse_complete_submission",
    "parse_index_json",
    "parse_sec_index_json",
    "reconcile_filing_content",
    "select_primary_document",
]
