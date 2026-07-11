"""Build the frozen SEC audit request plan from already-parsed metadata.

The planner is deliberately pure: it performs no filesystem, network, or model
work.  It validates and derives the URLs an eventual bounded audit may request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import json
from pathlib import PurePosixPath
import re
from typing import Any
from zoneinfo import ZoneInfo

from .sec_audit_selection import select_sec_audit_filings
from .sec_audit_transport import canonical_sec_url
from .sec_point_in_time import (
    AAPL_CIK,
    MAX_AUDIT_REQUESTS,
    FilingRecord,
    SecPointInTimeError,
    archive_urls,
)


CONTRACT_VERSION = "aapl-sec-point-in-time-audit-plan-v1"
AUDIT_START_YEAR = 2000
AUDIT_END_YEAR = 2024
AUDIT_START = date(AUDIT_START_YEAR, 1, 1)
AUDIT_CUTOFF = date(AUDIT_END_YEAR, 12, 31)
MAIN_SUBMISSIONS_NAME = f"CIK{AAPL_CIK}.json"
MAIN_SUBMISSIONS_URL = (
    f"https://data.sec.gov/submissions/{MAIN_SUBMISSIONS_NAME}"
)
_HISTORICAL_NAME_RE = re.compile(
    rf"CIK{AAPL_CIK}-submissions-[0-9]{{3}}\.json\Z"
)
_EASTERN = ZoneInfo("America/New_York")


def _canonical_cik(value: Any) -> str:
    if isinstance(value, bool):
        raise SecPointInTimeError("submissions CIK must contain only digits")
    raw = str(value).strip()
    if not raw.isdigit():
        raise SecPointInTimeError("submissions CIK must contain only digits")
    return raw.lstrip("0").zfill(10)


def _canonical_date(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SecPointInTimeError(f"{field_name} must be a canonical ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecPointInTimeError(
            f"{field_name} must be a canonical ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecPointInTimeError(f"{field_name} must be a canonical ISO date")
    return value


def _acceptance_et(record: FilingRecord) -> datetime:
    value = record.acceptance_datetime
    if re.fullmatch(r"[0-9]{14}", value):
        parsed = datetime.strptime(value, "%Y%m%d%H%M%S")
        return parsed.replace(tzinfo=_EASTERN)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:  # FilingRecord rejects this; retain a hard guard.
        raise SecPointInTimeError(
            "ISO acceptanceDateTime must contain an explicit timezone"
        )
    return parsed.astimezone(_EASTERN).replace(microsecond=0)


def _historical_references(
    main_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(main_payload, Mapping):
        raise SecPointInTimeError("main submissions payload must be an object")
    if _canonical_cik(main_payload.get("cik", "")) != AAPL_CIK:
        raise SecPointInTimeError(
            "submissions payload is not Apple CIK 0000320193"
        )
    filings = main_payload.get("filings")
    if not isinstance(filings, Mapping):
        raise SecPointInTimeError("main submissions payload lacks filings")
    raw_references = filings.get("files", [])
    if not isinstance(raw_references, Sequence) or isinstance(
        raw_references, (str, bytes, bytearray)
    ):
        raise SecPointInTimeError("filings.files must be an array")

    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_references:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise SecPointInTimeError(
                "historical submissions reference is invalid"
            )
        name = item["name"]
        if (
            name != name.strip()
            or PurePosixPath(name).name != name
            or "\\" in name
            or not _HISTORICAL_NAME_RE.fullmatch(name)
        ):
            raise SecPointInTimeError(
                "historical submissions filename is unsafe or outside Apple scope"
            )
        if name in seen:
            raise SecPointInTimeError(
                "duplicate historical submissions reference"
            )
        seen.add(name)
        url = canonical_sec_url(f"https://data.sec.gov/submissions/{name}")
        for supplied_url_field in ("url", "href"):
            if supplied_url_field in item:
                supplied = item[supplied_url_field]
                if not isinstance(supplied, str) or canonical_sec_url(supplied) != url:
                    raise SecPointInTimeError(
                        "historical submissions reference URL is not the official URL"
                    )

        result: dict[str, Any] = {"name": name, "url": url}
        for source_key, output_key in (
            ("filingFrom", "filing_from"),
            ("filingTo", "filing_to"),
        ):
            if source_key in item:
                result[output_key] = _canonical_date(item[source_key], source_key)
        if "filingCount" in item:
            count = item["filingCount"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise SecPointInTimeError(
                    "historical submissions filingCount must be non-negative"
                )
            result["filing_count"] = count
        if (
            "filing_from" in result
            and "filing_to" in result
            and result["filing_from"] > result["filing_to"]
        ):
            raise SecPointInTimeError(
                "historical submissions filing range is reversed"
            )
        references.append(result)
    return tuple(sorted(references, key=lambda item: item["name"]))


def _validate_and_deduplicate(
    filings: Sequence[FilingRecord],
) -> tuple[tuple[FilingRecord, ...], int]:
    if not isinstance(filings, Sequence) or isinstance(
        filings, (str, bytes, bytearray)
    ):
        raise TypeError("filings must be a sequence of FilingRecord values")
    by_accession: dict[str, FilingRecord] = {}
    for record in filings:
        if not isinstance(record, FilingRecord):
            raise TypeError("filings must contain only FilingRecord values")
        if record.subject_cik != AAPL_CIK:
            raise SecPointInTimeError(
                "filing identity must use canonical Apple subject CIK 0000320193"
            )
        if not isinstance(record.form, str) or not record.form.strip():
            raise SecPointInTimeError("filing form must be non-empty text")
        if not isinstance(record.source_name, str) or not record.source_name:
            raise SecPointInTimeError("filing source_name must be non-empty text")
        prior = by_accession.get(record.accession_number)
        if prior is None:
            by_accession[record.accession_number] = record
        elif prior != record:
            raise SecPointInTimeError(
                "conflicting metadata for duplicate accession"
            )
    ordered = tuple(
        sorted(
            by_accession.values(),
            key=lambda record: (
                record.filing_date,
                _acceptance_et(record).astimezone(timezone.utc),
                record.accession_number,
            ),
        )
    )
    return ordered, len(filings)


def _bounded_universe(
    filings: Sequence[FilingRecord],
) -> tuple[tuple[FilingRecord, ...], list[dict[str, Any]]]:
    included: list[FilingRecord] = []
    exclusions: list[dict[str, Any]] = []
    for record in filings:
        reasons: list[str] = []
        filing_date = date.fromisoformat(record.filing_date)
        if filing_date < AUDIT_START:
            reasons.append("filing_date_before_audit_start")
        if filing_date > AUDIT_CUTOFF:
            reasons.append("filing_date_after_cutoff")
        acceptance = _acceptance_et(record)
        if acceptance.date() < AUDIT_START:
            reasons.append("acceptance_before_audit_start")
        if acceptance.date() > AUDIT_CUTOFF:
            reasons.append("acceptance_after_cutoff")
        if record.date_of_filing_date_change:
            change_date = date.fromisoformat(record.date_of_filing_date_change)
            if change_date < AUDIT_START:
                reasons.append("filing_date_change_before_audit_start")
            if change_date > AUDIT_CUTOFF:
                reasons.append("filing_date_change_after_cutoff")
        if reasons:
            exclusions.append(
                {
                    "accession_number": record.accession_number,
                    "filing_date": record.filing_date,
                    "acceptance_datetime_et": acceptance.isoformat(),
                    "reasons": reasons,
                }
            )
        else:
            included.append(record)
    return tuple(included), exclusions


def _periodic_coverage(filings: Sequence[FilingRecord]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_years: list[int] = []
    for year in range(AUDIT_START_YEAR, AUDIT_END_YEAR + 1):
        periodic = [
            record
            for record in filings
            if int(record.filing_date[:4]) == year
            and record.form.upper() in {"10-K", "10-Q"}
        ]
        periodic.sort(
            key=lambda record: (
                _acceptance_et(record).astimezone(timezone.utc),
                record.filing_date,
                record.accession_number,
            )
        )
        form_counts = {
            form: sum(record.form.upper() == form for record in periodic)
            for form in ("10-K", "10-Q")
        }
        covered = bool(periodic)
        if not covered:
            missing_years.append(year)
        rows.append(
            {
                "year": year,
                "covered": covered,
                "periodic_report_count": len(periodic),
                "form_counts": form_counts,
                "accession_numbers": [
                    record.accession_number for record in periodic
                ],
                "source_names": sorted(
                    {record.source_name for record in periodic}
                ),
            }
        )
    return {
        "required_years": list(range(AUDIT_START_YEAR, AUDIT_END_YEAR + 1)),
        "required_forms": ["10-K", "10-Q"],
        "year_count": len(rows),
        "covered_year_count": len(rows) - len(missing_years),
        "missing_years": missing_years,
        "passes": not missing_years,
        "years": rows,
    }


def _quarterly_master_url(filing_date: str) -> tuple[int, int, str]:
    filed = date.fromisoformat(filing_date)
    quarter = (filed.month - 1) // 3 + 1
    url = canonical_sec_url(
        "https://www.sec.gov/Archives/edgar/full-index/"
        f"{filed.year}/QTR{quarter}/master.idx"
    )
    return filed.year, quarter, url


def _archive_request_urls(record: FilingRecord) -> tuple[str, str, str]:
    # SEC catalogues preserve historical accession prefixes that may differ
    # from Apple's subject CIK (for example, an old filing-agent prefix).  The
    # frozen Apple archive path remains under subject CIK 320193.
    urls = archive_urls(record)
    if urls.primary_document is None:
        raise SecPointInTimeError(
            "selected SEC filing has no safe primary-document URL"
        )
    return (
        canonical_sec_url(urls.index_json),
        canonical_sec_url(urls.complete_submission),
        canonical_sec_url(urls.primary_document),
    )


def _request_plan(
    selection: Mapping[str, Any],
    by_accession: Mapping[str, FilingRecord],
    historical_references: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accessions: list[dict[str, Any]] = []
    quarters: dict[tuple[int, int], dict[str, Any]] = {}
    slots = [*selection["core"], *selection["edge_cases"]]
    for slot in slots:
        if slot["status"] != "selected":
            continue
        accession = slot["filing"]["accession"]
        record = by_accession[accession]
        year, quarter, master_url = _quarterly_master_url(record.filing_date)
        (
            index_json_url,
            complete_submission_url,
            primary_document_url,
        ) = _archive_request_urls(record)
        role = slot["evidence_role"]
        accessions.append(
            {
                "accession_number": accession,
                "slot_id": slot["slot_id"],
                "group": slot["group"],
                "category": slot["category"],
                "requested_year": slot["requested_year"],
                "form": record.form.upper(),
                "filing_date": record.filing_date,
                "source_name": record.source_name,
                "evidence_role": role,
                "master_idx_url": master_url,
                "index_json_url": index_json_url,
                "complete_submission_url": complete_submission_url,
                "primary_document_url": primary_document_url,
            }
        )
        quarter_entry = quarters.setdefault(
            (year, quarter),
            {
                "year": year,
                "quarter": quarter,
                "url": master_url,
                "accession_numbers": [],
                "evidence_roles": set(),
            },
        )
        quarter_entry["accession_numbers"].append(accession)
        quarter_entry["evidence_roles"].add(role)

    master_indexes: list[dict[str, Any]] = []
    for key in sorted(quarters):
        item = quarters[key]
        master_indexes.append(
            {
                "year": item["year"],
                "quarter": item["quarter"],
                "url": item["url"],
                "accession_numbers": sorted(item["accession_numbers"]),
                "evidence_roles": sorted(item["evidence_roles"]),
            }
        )

    breakdown = {
        "main_submissions": 1,
        "historical_submissions": len(historical_references),
        "quarterly_master_idx": len(master_indexes),
        "accession_index_json": len(accessions),
        "complete_submissions": len(accessions),
        "primary_documents": len(accessions),
    }
    baseline = sum(breakdown.values())
    estimate = {
        "baseline_requests": baseline,
        "breakdown": breakdown,
        "frozen_max_requests": MAX_AUDIT_REQUESTS,
        "remaining_headroom_for_redirects_and_retries": (
            MAX_AUDIT_REQUESTS - baseline
        ),
        "includes_redirects_or_retries": False,
        "within_frozen_limit": baseline <= MAX_AUDIT_REQUESTS,
    }
    return accessions, master_indexes, estimate


def build_sec_audit_plan(
    filings: Sequence[FilingRecord],
    main_submissions_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a deterministic, JSON-safe plan for the frozen Apple SEC audit."""

    historical_references = _historical_references(main_submissions_payload)
    unique_filings, input_count = _validate_and_deduplicate(filings)
    bounded, exclusions = _bounded_universe(unique_filings)
    coverage = _periodic_coverage(bounded)
    selection = select_sec_audit_filings(bounded)
    by_accession = {record.accession_number: record for record in bounded}
    accession_requests, master_indexes, request_estimate = _request_plan(
        selection, by_accession, historical_references
    )

    allowed_sources = {
        MAIN_SUBMISSIONS_NAME,
        *(item["name"] for item in historical_references),
    }
    observed_sources = sorted({record.source_name for record in unique_filings})
    bounded_sources = sorted({record.source_name for record in bounded})
    unreferenced_sources = sorted(set(observed_sources) - allowed_sources)
    required_sources = {
        MAIN_SUBMISSIONS_NAME,
        *(
            item["name"]
            for item in historical_references
            if item.get("filing_count", 0) > 0
        ),
    }
    missing_required_sources = sorted(required_sources - set(observed_sources))
    source_coverage = {
        "allowed_source_names": sorted(allowed_sources),
        "required_nonempty_source_names": sorted(required_sources),
        "observed_source_names": observed_sources,
        "bounded_observed_source_names": bounded_sources,
        "unreferenced_source_names": unreferenced_sources,
        "missing_required_source_names": missing_required_sources,
        "all_record_sources_referenced": (
            not unreferenced_sources and not missing_required_sources
        ),
    }
    all_slots_selected = (
        selection["selected_count"] == 24 and selection["gap_count"] == 0
    )
    gates = {
        "periodic_report_catalogue_coverage": coverage["passes"],
        "all_24_frozen_slots_selected": all_slots_selected,
        "record_sources_referenced_by_main_payload": source_coverage[
            "all_record_sources_referenced"
        ],
        "baseline_request_estimate_within_limit": request_estimate[
            "within_frozen_limit"
        ],
    }
    result = {
        "contract_version": CONTRACT_VERSION,
        "subject_cik": AAPL_CIK,
        "audit_start": AUDIT_START.isoformat(),
        "audit_cutoff": AUDIT_CUTOFF.isoformat(),
        "input_record_count": input_count,
        "unique_record_count": len(unique_filings),
        "bounded_record_count": len(bounded),
        "excluded_record_count": len(exclusions),
        "exclusions": exclusions,
        "main_submissions": {
            "name": MAIN_SUBMISSIONS_NAME,
            "url": canonical_sec_url(MAIN_SUBMISSIONS_URL),
        },
        "historical_submissions": list(historical_references),
        "catalogue_source_coverage": source_coverage,
        "periodic_report_coverage": coverage,
        "selection": selection,
        "quarterly_master_indexes": master_indexes,
        "accession_requests": accession_requests,
        "request_count_estimate": request_estimate,
        "gates": gates,
        "ready_for_bounded_download": all(gates.values()),
    }
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def plan_sec_audit(
    filings: Sequence[FilingRecord],
    main_submissions_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility spelling for :func:`build_sec_audit_plan`."""

    return build_sec_audit_plan(filings, main_submissions_payload)


__all__ = [
    "AUDIT_CUTOFF",
    "AUDIT_END_YEAR",
    "AUDIT_START",
    "AUDIT_START_YEAR",
    "CONTRACT_VERSION",
    "MAIN_SUBMISSIONS_NAME",
    "MAIN_SUBMISSIONS_URL",
    "build_sec_audit_plan",
    "plan_sec_audit",
]
