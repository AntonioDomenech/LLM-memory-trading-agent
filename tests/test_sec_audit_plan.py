from __future__ import annotations

import copy

import pytest

from agent_benchmark.sec_audit_plan import (
    AUDIT_CUTOFF,
    MAIN_SUBMISSIONS_NAME,
    build_sec_audit_plan,
    plan_sec_audit,
)
from agent_benchmark.sec_audit_selection import STRUCTURAL_ROLE
from agent_benchmark.sec_point_in_time import (
    AAPL_CIK,
    FilingRecord,
    SecPointInTimeError,
)


def _record(
    accession: str,
    form: str,
    filing_date: str,
    acceptance: str,
    *,
    is_xbrl: bool = False,
    change_date: str = "",
    source_name: str = MAIN_SUBMISSIONS_NAME,
) -> FilingRecord:
    return FilingRecord(
        accession_number=accession,
        acceptance_datetime=acceptance,
        form=form,
        primary_document=f"{accession}.htm",
        items="",
        filing_date=filing_date,
        report_date="",
        is_xbrl=is_xbrl,
        source_name=source_name,
        subject_cik=AAPL_CIK,
        date_of_filing_date_change=change_date,
    )


def _main_payload() -> dict:
    return {
        "cik": "320193",
        "filings": {
            "files": [
                {
                    "name": "CIK0000320193-submissions-002.json",
                    "filingFrom": "1994-01-01",
                    "filingTo": "1999-12-31",
                    "filingCount": 0,
                },
                {
                    "name": "CIK0000320193-submissions-001.json",
                    "filingFrom": "2000-01-01",
                    "filingTo": "2009-12-31",
                    "filingCount": 0,
                },
            ]
        },
    }


def _complete_catalogue() -> list[FilingRecord]:
    records: list[FilingRecord] = []
    serial = 1

    def add(
        year: int,
        month_day: str,
        form: str,
        time: str = "090000",
        *,
        prefix: str = AAPL_CIK,
        is_xbrl: bool = False,
        change_date: str = "",
    ) -> None:
        nonlocal serial
        accession = f"{prefix}-{year % 100:02d}-{serial:06d}"
        compact_date = f"{year}{month_day.replace('-', '')}"
        records.append(
            _record(
                accession,
                form,
                f"{year}-{month_day}",
                f"{compact_date}{time}",
                is_xbrl=is_xbrl,
                change_date=change_date,
            )
        )
        serial += 1

    # The frozen annual coverage gate: amendments do not count.
    for year in range(2000, 2025):
        add(year, "04-15", "10-Q")

    # Exact 8-K and proxy slots for the six frozen core years.
    for year in (2000, 2005, 2009, 2014, 2019, 2024):
        add(year, "05-15", "8-K")
        add(year, "06-15", "DEF 14A")

    # Six distinct edge records, all accepted before their corresponding core
    # records where the selector's absolute chronological ordering matters.
    add(2000, "01-03", "S-8", "080000")
    add(2008, "01-03", "S-3", is_xbrl=True)
    add(2006, "01-03", "10-K/A")
    add(2007, "01-03", "S-4", "173100")
    add(2003, "01-03", "SC 13G", prefix="0000000123")
    add(2010, "01-04", "8-A12B", change_date="2010-01-05")
    return records


def test_builds_complete_deterministic_plan_and_request_budget() -> None:
    records = _complete_catalogue()
    plan = build_sec_audit_plan(records, _main_payload())

    assert plan["audit_cutoff"] == AUDIT_CUTOFF.isoformat()
    assert plan["periodic_report_coverage"]["passes"] is True
    assert plan["periodic_report_coverage"]["covered_year_count"] == 25
    assert len(plan["periodic_report_coverage"]["years"]) == 25
    assert plan["selection"]["selected_count"] == 24
    assert plan["selection"]["gap_count"] == 0
    assert len(plan["accession_requests"]) == 24
    assert len({item["url"] for item in plan["quarterly_master_indexes"]}) == len(
        plan["quarterly_master_indexes"]
    )

    estimate = plan["request_count_estimate"]
    breakdown = estimate["breakdown"]
    assert estimate["baseline_requests"] == sum(breakdown.values())
    assert breakdown == {
        "main_submissions": 1,
        "historical_submissions": 2,
        "quarterly_master_idx": len(plan["quarterly_master_indexes"]),
        "accession_index_json": 24,
        "complete_submissions": 24,
        "primary_documents": 24,
    }
    assert estimate["includes_redirects_or_retries"] is False
    assert estimate["within_frozen_limit"] is True
    assert plan["ready_for_bounded_download"] is True

    core_2024 = next(
        item
        for item in plan["accession_requests"]
        if item["slot_id"] == "core-2024-periodic"
    )
    assert core_2024["evidence_role"] == STRUCTURAL_ROLE
    assert core_2024["index_json_url"].endswith("/index.json")
    assert core_2024["complete_submission_url"].endswith(".txt")
    assert core_2024["primary_document_url"].endswith(".htm")

    assert plan == build_sec_audit_plan(list(reversed(records)), _main_payload())
    assert plan == plan_sec_audit(records, _main_payload())


def test_historical_accession_prefix_does_not_change_apple_archive_root() -> None:
    plan = build_sec_audit_plan(_complete_catalogue(), _main_payload())
    differing = next(
        item
        for item in plan["accession_requests"]
        if item["category"] == "first-differing-submitter-cik"
    )
    assert differing["accession_number"].startswith("0000000123-")
    assert "/Archives/edgar/data/320193/" in differing["index_json_url"]
    assert "/Archives/edgar/data/123/" not in differing["index_json_url"]
    assert "/Archives/edgar/data/320193/" in differing["primary_document_url"]


def test_cutoff_requires_both_filing_and_eastern_acceptance_by_2024() -> None:
    records = _complete_catalogue()
    records.extend(
        [
            _record(
                "0000320193-24-900001",
                "S-3",
                "2024-12-31",
                # Still 2024-12-31 in New York, so this remains admissible.
                "20241231193000",
            ),
            _record(
                "0000320193-25-900002",
                "S-3",
                "2025-01-02",
                "20241231120000",
            ),
            _record(
                "0000320193-24-900003",
                "S-3",
                "2024-12-31",
                "20250101000100",
            ),
            _record(
                "0000320193-24-900004",
                "S-3",
                "2024-12-30",
                "20241230120000",
                change_date="2025-01-02",
            ),
        ]
    )
    plan = build_sec_audit_plan(records, _main_payload())
    exclusions = {
        item["accession_number"]: item["reasons"]
        for item in plan["exclusions"]
    }
    assert "0000320193-24-900001" not in exclusions
    assert exclusions["0000320193-25-900002"] == [
        "filing_date_after_cutoff"
    ]
    assert exclusions["0000320193-24-900003"] == [
        "acceptance_after_cutoff"
    ]
    assert exclusions["0000320193-24-900004"] == [
        "filing_date_change_after_cutoff"
    ]
    assert plan["bounded_record_count"] == len(_complete_catalogue()) + 1


def test_edge_universe_cannot_select_a_pre_2000_filing() -> None:
    records = _complete_catalogue()
    records.append(
        _record(
            "0000320193-99-900001",
            "10-K/A",
            "1999-12-01",
            "19991201120000",
        )
    )
    plan = build_sec_audit_plan(records, _main_payload())
    amendment = next(
        item
        for item in plan["accession_requests"]
        if item["category"] == "first-amendment"
    )
    assert amendment["filing_date"].startswith("2006-")
    exclusion = next(
        item
        for item in plan["exclusions"]
        if item["accession_number"] == "0000320193-99-900001"
    )
    assert exclusion["reasons"] == [
        "filing_date_before_audit_start",
        "acceptance_before_audit_start",
    ]


def test_periodic_coverage_is_exact_and_exposes_every_missing_year() -> None:
    amendment = _record(
        "0000320193-00-000001",
        "10-K/A",
        "2000-01-03",
        "20000103090000",
    )
    plan = build_sec_audit_plan([amendment], {"cik": "320193", "filings": {}})
    coverage = plan["periodic_report_coverage"]
    assert coverage["passes"] is False
    assert coverage["covered_year_count"] == 0
    assert coverage["missing_years"] == list(range(2000, 2025))
    assert coverage["years"][0]["form_counts"] == {"10-K": 0, "10-Q": 0}
    assert plan["selection"]["core"][0]["status"] == "gap"
    assert plan["ready_for_bounded_download"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload.update({"cik": "123"}),
            "not Apple",
        ),
        (
            lambda payload: payload["filings"]["files"][0].update(
                {"name": "../CIK0000320193-submissions-002.json"}
            ),
            "unsafe",
        ),
        (
            lambda payload: payload["filings"]["files"][0].update(
                {"url": "https://example.com/file.json"}
            ),
            "official SEC",
        ),
        (
            lambda payload: payload["filings"]["files"].append(
                copy.deepcopy(payload["filings"]["files"][0])
            ),
            "duplicate",
        ),
        (
            lambda payload: payload["filings"]["files"][0].update(
                {"filingCount": -1}
            ),
            "non-negative",
        ),
    ],
)
def test_historical_submission_references_fail_closed(mutator, message: str) -> None:
    payload = _main_payload()
    mutator(payload)
    with pytest.raises(SecPointInTimeError, match=message):
        build_sec_audit_plan([], payload)


def test_duplicate_accessions_only_collapse_when_metadata_is_identical() -> None:
    record = _record(
        "0000320193-00-000001",
        "10-Q",
        "2000-01-03",
        "20000103090000",
    )
    plan = build_sec_audit_plan(
        [record, record], {"cik": "320193", "filings": {}}
    )
    assert plan["input_record_count"] == 2
    assert plan["unique_record_count"] == 1

    conflict = _record(
        record.accession_number,
        "8-K",
        record.filing_date,
        record.acceptance_datetime,
    )
    with pytest.raises(SecPointInTimeError, match="conflicting metadata"):
        build_sec_audit_plan(
            [record, conflict], {"cik": "320193", "filings": {}}
        )
