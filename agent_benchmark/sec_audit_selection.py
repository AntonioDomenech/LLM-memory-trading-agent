"""Pure deterministic filing selection for the point-in-time SEC audit.

This module selects metadata records only.  It performs no network access,
document loading, parsing, or content scoring.  Missing requested categories
remain explicit gaps; a filing from another category is never substituted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
import json
import re
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo


CONTRACT_VERSION = "aapl-sec-point-in-time-audit-selection-v1"
EASTERN = ZoneInfo("America/New_York")

CORE_YEARS: tuple[int, ...] = (2000, 2005, 2009, 2014, 2019, 2024)
CORE_CATEGORIES: tuple[str, ...] = ("periodic", "8-k", "def-14a")
EDGE_CATEGORIES: tuple[str, ...] = (
    "earliest-2000",
    "first-xbrl",
    "first-amendment",
    "first-after-1730-et",
    "first-differing-submitter-cik",
    "first-anomaly",
)

HISTORICAL_ROLE = "historical_content_audit_candidate"
STRUCTURAL_ROLE = "structural_only"

_FILING_OUTPUT_FIELDS: tuple[str, ...] = (
    "accession",
    "form",
    "filing_year",
    "filing_date",
    "acceptance_timestamp_et",
    "is_xbrl",
    "submitter_cik",
    "subject_cik",
    "change_anomaly_flag",
)


@dataclass(frozen=True)
class FilingMetadata:
    accession: str
    form: str
    filing_year: int
    filing_date: str
    acceptance_timestamp_et: str
    is_xbrl: bool | None
    submitter_cik: str | None
    subject_cik: str | None
    change_anomaly_flag: bool | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if tuple(payload) != _FILING_OUTPUT_FIELDS:  # pragma: no cover - defensive
            raise RuntimeError("SEC metadata output schema changed")
        return payload


@dataclass(frozen=True)
class _NormalizedFiling:
    metadata: FilingMetadata
    acceptance_et: datetime

    @property
    def accession(self) -> str:
        return self.metadata.accession

    @property
    def form(self) -> str:
        return self.metadata.form

    @property
    def filing_year(self) -> int:
        return self.metadata.filing_year

    @property
    def sort_key(self) -> tuple[datetime, str, str]:
        return (
            self.acceptance_et.astimezone(timezone.utc),
            self.metadata.filing_date,
            self.accession,
        )


def _record_mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record
    if hasattr(record, "__dict__"):
        result = dict(vars(record))
        # Preserve derived properties on the parser's immutable FilingRecord;
        # vars() alone omits submitter_cik and would silently disable that
        # frozen edge-case selector.
        for name in ("submitter_cik", "is_amendment"):
            if hasattr(record, name):
                result[name] = getattr(record, name)
        return result
    raise TypeError("Each SEC filing record must be a mapping or attribute record")


def _first_value(record: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _required_text(value: Any, *, field_name: str) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        raise ValueError(f"{field_name} is required")
    return text_value


def _parse_date(value: Any, *, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text_value = _required_text(value, field_name=field_name)
    if re.fullmatch(r"\d{8}", text_value):
        return datetime.strptime(text_value, "%Y%m%d").date()
    try:
        parsed = date.fromisoformat(text_value[:10])
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc
    return parsed


def _parse_acceptance(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text_value = _required_text(value, field_name="acceptance_timestamp")
        if re.fullmatch(r"\d{14}", text_value):
            parsed = datetime.strptime(text_value, "%Y%m%d%H%M%S")
        else:
            normalized = text_value[:-1] + "+00:00" if text_value.endswith("Z") else text_value
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise ValueError(
                    "acceptance_timestamp must be an ISO datetime or YYYYMMDDHHMMSS"
                ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=EASTERN)
    else:
        parsed = parsed.astimezone(EASTERN)
    return parsed.replace(microsecond=0)


def _optional_boolean(value: Any, *, field_name: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    raise ValueError(f"{field_name} must be boolean, zero/one, or missing")


def _canonical_cik(value: Any) -> str | None:
    if value is None or value == "":
        return None
    raw = str(value).strip().replace("-", "")
    if not raw.isdigit():
        raise ValueError("CIK values must contain only digits")
    return raw.lstrip("0").zfill(10)


def _normalize_record(record: Any) -> _NormalizedFiling:
    source = _record_mapping(record)
    accession = _required_text(
        _first_value(source, ("accession", "accession_number")),
        field_name="accession",
    )
    form = _required_text(
        _first_value(source, ("form", "form_type")), field_name="form"
    ).upper()
    filing_date_value = _first_value(source, ("filing_date", "filed_date"))
    filing_year_value = _first_value(source, ("filing_year", "year"))
    if filing_date_value is None and filing_year_value is None:
        raise ValueError("filing_date or filing_year is required")
    if filing_date_value is None:
        if isinstance(filing_year_value, bool):
            raise ValueError("filing_year must be an integer")
        try:
            filing_year = int(filing_year_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("filing_year must be an integer") from exc
        filing_date = date(filing_year, 1, 1)
    else:
        filing_date = _parse_date(filing_date_value, field_name="filing_date")
        filing_year = filing_date.year
        if filing_year_value is not None:
            if isinstance(filing_year_value, bool):
                raise ValueError("filing_year must be an integer")
            try:
                declared_year = int(filing_year_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("filing_year must be an integer") from exc
            if declared_year != filing_year:
                raise ValueError("filing_year is inconsistent with filing_date")

    acceptance = _parse_acceptance(
        _first_value(
            source,
            ("acceptance_timestamp", "accepted_at", "acceptance_datetime"),
        )
    )
    is_xbrl = _optional_boolean(
        _first_value(source, ("is_xbrl", "xbrl")), field_name="is_xbrl"
    )
    submitter_cik = _canonical_cik(
        _first_value(source, ("submitter_cik", "filer_cik"))
    )
    subject_cik = _canonical_cik(
        _first_value(source, ("subject_cik", "issuer_cik"))
    )
    anomaly_values = [
        source[name]
        for name in ("change_anomaly_flag", "change_flag", "anomaly_flag")
        if name in source
    ]
    anomaly = None
    if anomaly_values:
        parsed_flags = [
            _optional_boolean(value, field_name="change_anomaly_flag")
            for value in anomaly_values
        ]
        known = [value for value in parsed_flags if value is not None]
        anomaly = any(known) if known else None
    metadata = FilingMetadata(
        accession=accession,
        form=form,
        filing_year=filing_year,
        filing_date=filing_date.isoformat(),
        acceptance_timestamp_et=acceptance.isoformat(),
        is_xbrl=is_xbrl,
        submitter_cik=submitter_cik,
        subject_cik=subject_cik,
        change_anomaly_flag=anomaly,
    )
    return _NormalizedFiling(metadata=metadata, acceptance_et=acceptance)


def _normalize_records(records: Iterable[Any]) -> list[_NormalizedFiling]:
    normalized = [_normalize_record(record) for record in records]
    by_accession: dict[str, _NormalizedFiling] = {}
    for filing in normalized:
        existing = by_accession.get(filing.accession)
        if existing is None:
            by_accession[filing.accession] = filing
        elif existing.metadata != filing.metadata:
            raise ValueError(
                f"Conflicting metadata for duplicate accession: {filing.accession}"
            )
    return sorted(by_accession.values(), key=lambda filing: filing.sort_key)


def _evidence_role(year: int | None) -> str:
    return STRUCTURAL_ROLE if year is not None and year >= 2019 else HISTORICAL_ROLE


def _slot(
    *,
    slot_id: str,
    group: str,
    category: str,
    requested_year: int | None,
    filing: _NormalizedFiling | None,
    gap_reason: str | None = None,
    skipped_accessions: tuple[str, ...] = (),
) -> dict[str, Any]:
    selected_year = filing.filing_year if filing is not None else requested_year
    return {
        "slot_id": slot_id,
        "group": group,
        "category": category,
        "requested_year": requested_year,
        "status": "selected" if filing is not None else "gap",
        "evidence_role": _evidence_role(selected_year),
        "filing": filing.metadata.to_dict() if filing is not None else None,
        "gap_reason": None if filing is not None else gap_reason,
        "skipped_already_selected_accessions": list(skipped_accessions),
    }


def _core_matches(filing: _NormalizedFiling, category: str) -> bool:
    if category == "periodic":
        return filing.form in {"10-K", "10-Q"}
    if category == "8-k":
        return filing.form == "8-K"
    if category == "def-14a":
        return filing.form == "DEF 14A"
    raise ValueError(f"Unknown core category: {category}")


def _edge_matches(filing: _NormalizedFiling, category: str) -> bool:
    metadata = filing.metadata
    if category == "earliest-2000":
        return filing.filing_year == 2000
    if category == "first-xbrl":
        return metadata.is_xbrl is True
    if category == "first-amendment":
        return filing.form.endswith("/A")
    if category == "first-after-1730-et":
        local_time = filing.acceptance_et.timetz().replace(tzinfo=None)
        return local_time > time(17, 30)
    if category == "first-differing-submitter-cik":
        return (
            metadata.submitter_cik is not None
            and metadata.subject_cik is not None
            and metadata.submitter_cik != metadata.subject_cik
        )
    if category == "first-anomaly":
        return metadata.change_anomaly_flag is True
    raise ValueError(f"Unknown edge category: {category}")


def select_sec_audit_filings(records: Iterable[Any]) -> dict[str, Any]:
    """Return the frozen 18-core plus six distinct-edge metadata selection.

    Core slots always select the absolute earliest acceptance in their exact
    year/form category.  Edge slots are processed in their frozen order and
    select the earliest qualifying accession not already used by a core or an
    earlier edge slot.  Skipping an already selected filing is allowed only
    within the same edge condition; an unrelated filing is never substituted.
    """

    filings = _normalize_records(records)
    used: set[str] = set()
    core: list[dict[str, Any]] = []
    for year in CORE_YEARS:
        for category in CORE_CATEGORIES:
            candidates = [
                filing
                for filing in filings
                if filing.filing_year == year and _core_matches(filing, category)
            ]
            selected = candidates[0] if candidates else None
            if selected is not None:
                used.add(selected.accession)
            core.append(
                _slot(
                    slot_id=f"core-{year}-{category}",
                    group="core",
                    category=category,
                    requested_year=year,
                    filing=selected,
                    gap_reason="no_filing_in_exact_year_and_form_category",
                )
            )

    edges: list[dict[str, Any]] = []
    for category in EDGE_CATEGORIES:
        candidates = [filing for filing in filings if _edge_matches(filing, category)]
        skipped = tuple(
            filing.accession for filing in candidates if filing.accession in used
        )
        selected = next(
            (filing for filing in candidates if filing.accession not in used), None
        )
        if selected is not None:
            used.add(selected.accession)
            reason = None
        elif candidates:
            reason = "all_qualifying_filings_already_selected"
        else:
            reason = "no_filing_matches_exact_edge_condition"
        edges.append(
            _slot(
                slot_id=f"edge-{category}",
                group="edge",
                category=category,
                requested_year=2000 if category == "earliest-2000" else None,
                filing=selected,
                gap_reason=reason,
                skipped_accessions=skipped,
            )
        )

    selected_accessions = [
        slot["filing"]["accession"]
        for slot in (*core, *edges)
        if slot["status"] == "selected"
    ]
    if len(selected_accessions) != len(set(selected_accessions)):
        raise RuntimeError("SEC audit selections are not accession-distinct")
    result = {
        "contract_version": CONTRACT_VERSION,
        "input_record_count": len(filings),
        "core_slot_count": 18,
        "edge_slot_count": 6,
        "core": core,
        "edge_cases": edges,
        "selected_accessions": selected_accessions,
        "selected_count": len(selected_accessions),
        "gap_count": sum(
            slot["status"] == "gap" for slot in (*core, *edges)
        ),
        "role_policy": {
            "through_2018": HISTORICAL_ROLE,
            "2019_and_later": STRUCTURAL_ROLE,
            "core_2019": STRUCTURAL_ROLE,
            "core_2024": STRUCTURAL_ROLE,
        },
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


__all__ = [
    "CONTRACT_VERSION",
    "CORE_CATEGORIES",
    "CORE_YEARS",
    "EDGE_CATEGORIES",
    "FilingMetadata",
    "HISTORICAL_ROLE",
    "STRUCTURAL_ROLE",
    "select_sec_audit_filings",
]
