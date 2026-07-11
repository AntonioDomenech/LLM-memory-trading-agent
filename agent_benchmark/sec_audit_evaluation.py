"""Pure evaluation gates for the frozen 24-filing SEC audit."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable

from .sec_audit_selection import (
    CORE_CATEGORIES,
    CORE_YEARS,
    EDGE_CATEGORIES,
    HISTORICAL_ROLE,
    STRUCTURAL_ROLE,
)
from .sec_point_in_time import AAPL_CIK


CONTRACT_VERSION = "aapl-sec-point-in-time-audit-evaluation-v1"
REQUIRED_SAMPLE_COUNT = 24
REQUIRED_TIMESTAMP_COUNT = 23
REQUIRED_VISIBLE_TEXT_COUNT = 23
REQUIRED_PERIODIC_YEARS = tuple(range(2000, 2025))
EXPECTED_SLOT_IDS = tuple(
    [
        f"core-{year}-{category}"
        for year in CORE_YEARS
        for category in CORE_CATEGORIES
    ]
    + [f"edge-{category}" for category in EDGE_CATEGORIES]
)
SOURCE_NAMES = ("submissions", "master_idx", "sgml_header", "index_metadata")
_ACCESSION_RE = re.compile(r"\d{10}-\d{2}-\d{6}\Z")


@dataclass(frozen=True)
class NormalizedFilingIdentity:
    accession: str
    form: str
    filing_date: str
    subject_cik: str = AAPL_CIK

    def __post_init__(self) -> None:
        if not _ACCESSION_RE.fullmatch(self.accession):
            raise ValueError("normalized SEC identity has an invalid accession")
        if not isinstance(self.form, str) or not self.form.strip():
            raise ValueError("normalized SEC identity requires a form")
        try:
            date.fromisoformat(self.filing_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("normalized SEC identity has an invalid filing date") from exc
        if self.subject_cik != AAPL_CIK:
            raise ValueError("normalized SEC identity is outside the Apple audit")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizedIndexIdentity:
    accession: str
    subject_cik: str = AAPL_CIK

    def __post_init__(self) -> None:
        if not _ACCESSION_RE.fullmatch(self.accession):
            raise ValueError("normalized SEC index identity has an invalid accession")
        if self.subject_cik != AAPL_CIK:
            raise ValueError("normalized SEC index identity is outside the Apple audit")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FilingAuditResult:
    slot_id: str
    status: str
    evidence_role: str
    selected: NormalizedFilingIdentity | None
    submissions: NormalizedFilingIdentity | None
    master_idx: NormalizedFilingIdentity | None
    sgml_header: NormalizedFilingIdentity | None
    index_metadata: NormalizedIndexIdentity | None
    exact_acceptance_timestamp: bool
    usable_visible_text: bool
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, str) or not self.slot_id:
            raise ValueError("SEC audit slot_id is required")
        if self.status not in {"selected", "gap"}:
            raise ValueError("SEC audit status must be selected or gap")
        if not isinstance(self.exact_acceptance_timestamp, bool):
            raise ValueError("exact_acceptance_timestamp must be boolean")
        if not isinstance(self.usable_visible_text, bool):
            raise ValueError("usable_visible_text must be boolean")

    def source_identities(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in SOURCE_NAMES}


@dataclass(frozen=True)
class PeriodicCoverageResult:
    year: int
    accessions: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.year, bool) or not isinstance(self.year, int):
            raise ValueError("periodic coverage year must be an integer")
        if not isinstance(self.accessions, tuple):
            raise ValueError("periodic coverage accessions must be a tuple")
        if any(not _ACCESSION_RE.fullmatch(value) for value in self.accessions):
            raise ValueError("periodic coverage contains an invalid accession")
        if len(self.accessions) != len(set(self.accessions)):
            raise ValueError("periodic coverage accessions must be unique within year")


def _expected_role(identity: NormalizedFilingIdentity) -> str:
    year = date.fromisoformat(identity.filing_date).year
    return STRUCTURAL_ROLE if year >= 2019 else HISTORICAL_ROLE


def _sorted_slots(results: Iterable[FilingAuditResult]) -> list[FilingAuditResult]:
    slots = list(results)
    if any(not isinstance(item, FilingAuditResult) for item in slots):
        raise TypeError("filing audit results must be FilingAuditResult objects")
    return sorted(
        slots,
        key=lambda item: (
            item.slot_id,
            item.status,
            item.selected.accession if item.selected else "",
            item.evidence_role,
        ),
    )


def _selection_gate(slots: list[FilingAuditResult]) -> dict[str, Any]:
    slot_counts = Counter(item.slot_id for item in slots)
    expected = set(EXPECTED_SLOT_IDS)
    observed = set(slot_counts)
    selected = [item for item in slots if item.status == "selected" and item.selected]
    accessions = [item.selected.accession for item in selected if item.selected]
    invalid_roles = [
        item.slot_id
        for item in selected
        if item.evidence_role not in {HISTORICAL_ROLE, STRUCTURAL_ROLE}
        or item.evidence_role != _expected_role(item.selected)
    ]
    duplicate_slots = sorted(name for name, count in slot_counts.items() if count > 1)
    duplicate_accessions = sorted(
        accession for accession, count in Counter(accessions).items() if count > 1
    )
    missing_slots = sorted(expected - observed)
    extra_slots = sorted(observed - expected)
    gap_slots = sorted(item.slot_id for item in slots if item.status == "gap")
    unresolved_slots = sorted(
        item.slot_id
        for item in slots
        if item.status != "selected" or item.selected is None
    )
    role_counts = Counter(item.evidence_role for item in selected)
    passed = bool(
        len(slots) == REQUIRED_SAMPLE_COUNT
        and not duplicate_slots
        and not missing_slots
        and not extra_slots
        and not gap_slots
        and len(selected) == REQUIRED_SAMPLE_COUNT
        and len(set(accessions)) == REQUIRED_SAMPLE_COUNT
        and not duplicate_accessions
        and not invalid_roles
    )
    return {
        "passed": passed,
        "required_slot_count": REQUIRED_SAMPLE_COUNT,
        "observed_slot_count": len(slots),
        "selected_count": len(selected),
        "unique_selected_accession_count": len(set(accessions)),
        "missing_slot_ids": missing_slots,
        "extra_slot_ids": extra_slots,
        "duplicate_slot_ids": duplicate_slots,
        "duplicate_accessions": duplicate_accessions,
        "gap_slot_ids": gap_slots,
        "unresolved_slot_ids": unresolved_slots,
        "invalid_role_slot_ids": sorted(invalid_roles),
        "role_counts": dict(sorted(role_counts.items())),
    }


def _slot_semantics_gate(slots: list[FilingAuditResult]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    catalogue_only_edges = {
        "edge-first-xbrl",
        "edge-first-after-1730-et",
        "edge-first-differing-submitter-cik",
        "edge-first-anomaly",
    }
    for item in slots:
        identity = item.selected
        if item.status != "selected" or identity is None:
            continue
        year = date.fromisoformat(identity.filing_date).year
        form = " ".join(identity.form.upper().split())
        valid = True
        expected: str | None = None
        if item.slot_id.startswith("core-"):
            parts = item.slot_id.split("-", 2)
            requested_year = int(parts[1])
            category = parts[2]
            if category == "periodic":
                valid = year == requested_year and form in {"10-K", "10-Q"}
                expected = f"year {requested_year} and form 10-K or 10-Q"
            elif category == "8-k":
                valid = year == requested_year and form == "8-K"
                expected = f"year {requested_year} and form 8-K"
            elif category == "def-14a":
                valid = year == requested_year and form == "DEF 14A"
                expected = f"year {requested_year} and form DEF 14A"
        elif item.slot_id == "edge-earliest-2000":
            valid = year == 2000
            expected = "filing year 2000"
        elif item.slot_id == "edge-first-amendment":
            valid = form.endswith("/A")
            expected = "amendment form ending in /A"
        if not valid:
            failures.append(
                {
                    "slot_id": item.slot_id,
                    "observed_year": year,
                    "observed_form": form,
                    "expected": expected,
                }
            )
    return {
        "passed": not failures and len(slots) == REQUIRED_SAMPLE_COUNT,
        "failures": failures,
        "catalogue_predicates_enforced_by_frozen_selection_plan": sorted(
            catalogue_only_edges
        ),
    }


def _agreement_gate(slots: list[FilingAuditResult]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    agreeing = 0
    for item in slots:
        missing_sources: list[str] = []
        mismatched_sources: list[str] = []
        sources = item.source_identities()
        if item.status != "selected" or item.selected is None:
            missing_sources = list(SOURCE_NAMES)
        else:
            for name in SOURCE_NAMES:
                identity = sources[name]
                if identity is None:
                    missing_sources.append(name)
                elif name == "index_metadata":
                    if not isinstance(identity, NormalizedIndexIdentity) or (
                        identity.accession != item.selected.accession
                        or identity.subject_cik != item.selected.subject_cik
                    ):
                        mismatched_sources.append(name)
                elif identity != item.selected:
                    mismatched_sources.append(name)
        passed = not missing_sources and not mismatched_sources
        agreeing += int(passed)
        evidence.append(
            {
                "slot_id": item.slot_id,
                "passed": passed,
                "selected_identity": (
                    item.selected.to_dict() if item.selected is not None else None
                ),
                "missing_sources": missing_sources,
                "mismatched_sources": mismatched_sources,
            }
        )
    evidence.sort(key=lambda item: item["slot_id"])
    return {
        "passed": agreeing == REQUIRED_SAMPLE_COUNT and len(slots) == REQUIRED_SAMPLE_COUNT,
        "required_agreement_count": REQUIRED_SAMPLE_COUNT,
        "agreement_count": agreeing,
        "disagreement_count": len(slots) - agreeing,
        "mutually_represented_field_policy": {
            "submissions_master_sgml": [
                "accession",
                "form",
                "filing_date",
                "subject_cik",
            ],
            "index_metadata": ["accession", "subject_cik"],
        },
        "filings": evidence,
    }


def _boolean_threshold_gate(
    slots: list[FilingAuditResult],
    *,
    attribute: str,
    required: int,
) -> dict[str, Any]:
    passing_slots = sorted(
        item.slot_id
        for item in slots
        if item.status == "selected" and bool(getattr(item, attribute))
    )
    failing_slots = sorted(
        item.slot_id
        for item in slots
        if item.status != "selected" or not bool(getattr(item, attribute))
    )
    count = len(passing_slots)
    return {
        "passed": count >= required and len(slots) == REQUIRED_SAMPLE_COUNT,
        "required_count": required,
        "denominator": REQUIRED_SAMPLE_COUNT,
        "observed_count": count,
        "passing_slot_ids": passing_slots,
        "failing_slot_ids": failing_slots,
    }


def _periodic_gate(
    results: Iterable[PeriodicCoverageResult],
) -> dict[str, Any]:
    values = list(results)
    if any(not isinstance(item, PeriodicCoverageResult) for item in values):
        raise TypeError("periodic results must be PeriodicCoverageResult objects")
    counts = Counter(item.year for item in values)
    expected = set(REQUIRED_PERIODIC_YEARS)
    observed = set(counts)
    by_year = {
        item.year: item
        for item in sorted(values, key=lambda item: (item.year, item.accessions))
    }
    duplicate_years = sorted(year for year, count in counts.items() if count > 1)
    missing_records = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    empty_years = sorted(
        year for year in expected & observed if not by_year[year].accessions
    )
    covered = sorted(
        year for year in expected & observed if by_year[year].accessions
    )
    return {
        "passed": not duplicate_years and not missing_records and not unexpected and not empty_years,
        "required_years": list(REQUIRED_PERIODIC_YEARS),
        "covered_years": covered,
        "missing_year_records": missing_records,
        "years_without_periodic_filing": empty_years,
        "duplicate_year_records": duplicate_years,
        "unexpected_year_records": unexpected,
        "evidence": [
            {
                "year": item.year,
                "periodic_accession_count": len(item.accessions),
                "accessions": list(item.accessions),
            }
            for item in sorted(values, key=lambda item: (item.year, item.accessions))
        ],
    }


def evaluate_sec_audit(
    filing_results: Iterable[FilingAuditResult],
    periodic_results: Iterable[PeriodicCoverageResult],
) -> dict[str, Any]:
    """Evaluate normalized evidence without parsing, I/O, or substitution."""

    slots = _sorted_slots(filing_results)
    selection = _selection_gate(slots)
    slot_semantics = _slot_semantics_gate(slots)
    agreement = _agreement_gate(slots)
    timestamps = _boolean_threshold_gate(
        slots,
        attribute="exact_acceptance_timestamp",
        required=REQUIRED_TIMESTAMP_COUNT,
    )
    visible_text = _boolean_threshold_gate(
        slots,
        attribute="usable_visible_text",
        required=REQUIRED_VISIBLE_TEXT_COUNT,
    )
    periodic = _periodic_gate(periodic_results)
    gaps = selection["gap_slot_ids"]
    unresolved = selection["unresolved_slot_ids"]
    missing_evidence = [
        {
            "slot_id": item.slot_id,
            "status": item.status,
            "reason": item.missing_reason or (
                "selected_status_without_identity"
                if item.status == "selected"
                else "gap_without_reason"
            ),
        }
        for item in slots
        if item.status != "selected" or item.selected is None
    ]
    missing_evidence.extend(
        {
            "slot_id": slot_id,
            "status": "absent",
            "reason": "required_slot_record_absent",
        }
        for slot_id in selection["missing_slot_ids"]
    )
    missing_evidence.sort(key=lambda item: (item["slot_id"], item["status"]))
    explicit_missing = {
        "passed": not unresolved and not selection["missing_slot_ids"],
        "gap_slot_ids": gaps,
        "unresolved_slot_ids": unresolved,
        "absent_slot_ids": selection["missing_slot_ids"],
        "missing_evidence": missing_evidence,
        "substitution_allowed": False,
    }
    gates = {
        "exact_24_unique_selected_filings_and_roles": selection,
        "core_year_form_and_identity_slot_semantics": slot_semantics,
        "all_24_cross_source_identities_agree": agreement,
        "at_least_23_exact_acceptance_timestamps": timestamps,
        "at_least_23_usable_visible_texts": visible_text,
        "periodic_coverage_2000_2024": periodic,
        "explicit_missing_slots_fail_without_substitution": explicit_missing,
    }
    report = {
        "contract_version": CONTRACT_VERSION,
        "evidence_classification": "sec_source_audit_only_not_model_or_trading_evidence",
        "required_sample_count": REQUIRED_SAMPLE_COUNT,
        "gates": gates,
        "overall_pass": bool(all(gate["passed"] for gate in gates.values())),
    }
    json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return report


__all__ = [
    "CONTRACT_VERSION",
    "EXPECTED_SLOT_IDS",
    "FilingAuditResult",
    "NormalizedFilingIdentity",
    "NormalizedIndexIdentity",
    "PeriodicCoverageResult",
    "REQUIRED_PERIODIC_YEARS",
    "REQUIRED_SAMPLE_COUNT",
    "REQUIRED_TIMESTAMP_COUNT",
    "REQUIRED_VISIBLE_TEXT_COUNT",
    "evaluate_sec_audit",
]
