from __future__ import annotations

import copy
from datetime import datetime
from types import SimpleNamespace

import pytest

from agent_benchmark.sec_audit_selection import (
    CONTRACT_VERSION,
    CORE_CATEGORIES,
    CORE_YEARS,
    EDGE_CATEGORIES,
    HISTORICAL_ROLE,
    STRUCTURAL_ROLE,
    select_sec_audit_filings,
)
from agent_benchmark.sec_point_in_time import FilingRecord, parse_submissions_rows


def _record(
    accession: str,
    form: str,
    filing_date: str,
    acceptance: str,
    *,
    is_xbrl: bool = False,
    submitter_cik: str = "0000320193",
    subject_cik: str = "0000320193",
    anomaly: bool = False,
    **extra,
) -> dict:
    return {
        "accession": accession,
        "form": form,
        "filing_year": int(filing_date[:4]),
        "filing_date": filing_date,
        "acceptance_timestamp": acceptance,
        "is_xbrl": is_xbrl,
        "submitter_cik": submitter_cik,
        "subject_cik": subject_cik,
        "change_anomaly_flag": anomaly,
        **extra,
    }


def _complete_records() -> list[dict]:
    records: list[dict] = []
    for year in CORE_YEARS:
        records.extend(
            [
                _record(
                    f"{year}-periodic-first",
                    "10-Q",
                    f"{year}-02-01",
                    f"{year}-02-01T09:00:00",
                ),
                _record(
                    f"{year}-periodic-later",
                    "10-K",
                    f"{year}-02-15",
                    f"{year}-02-15T09:00:00",
                ),
                _record(
                    f"{year}-8k-first",
                    "8-K",
                    f"{year}-03-01",
                    f"{year}-03-01T09:00:00",
                ),
                _record(
                    f"{year}-def-first",
                    "DEF 14A",
                    f"{year}-04-01",
                    f"{year}-04-01T09:00:00",
                ),
            ]
        )
    records.extend(
        [
            _record(
                "edge-earliest-2000",
                "S-8",
                "2000-01-03",
                "2000-01-03T08:00:00",
            ),
            _record(
                "edge-first-xbrl",
                "S-3",
                "2008-01-02",
                "2008-01-02T09:00:00",
                is_xbrl=True,
            ),
            _record(
                "edge-first-amendment",
                "10-K/A",
                "2006-01-03",
                "2006-01-03T09:00:00",
            ),
            _record(
                "edge-first-late",
                "S-4",
                "2007-01-03",
                "2007-01-03T17:31:00",
            ),
            _record(
                "edge-first-different-cik",
                "SC 13G",
                "2003-01-03",
                "2003-01-03T09:00:00",
                submitter_cik="0000000123",
                subject_cik="0000320193",
            ),
            _record(
                "edge-first-anomaly",
                "8-A12B",
                "2010-01-04",
                "2010-01-04T09:00:00",
                anomaly=True,
            ),
        ]
    )
    return records


def _selected_by_slot(result: dict) -> dict[str, str]:
    return {
        slot["slot_id"]: slot["filing"]["accession"]
        for slot in (*result["core"], *result["edge_cases"])
        if slot["status"] == "selected"
    }


def test_selects_exact_eighteen_core_and_six_distinct_edges() -> None:
    result = select_sec_audit_filings(_complete_records())
    assert result["contract_version"] == CONTRACT_VERSION
    assert result["core_slot_count"] == 18
    assert result["edge_slot_count"] == 6
    assert len(result["core"]) == 18
    assert len(result["edge_cases"]) == 6
    assert result["gap_count"] == 0
    assert result["selected_count"] == 24
    assert len(result["selected_accessions"]) == len(
        set(result["selected_accessions"])
    )

    selected = _selected_by_slot(result)
    for year in CORE_YEARS:
        assert selected[f"core-{year}-periodic"] == f"{year}-periodic-first"
        assert selected[f"core-{year}-8-k"] == f"{year}-8k-first"
        assert selected[f"core-{year}-def-14a"] == f"{year}-def-first"
    assert [
        selected[f"edge-{category}"] for category in EDGE_CATEGORIES
    ] == [
        "edge-earliest-2000",
        "edge-first-xbrl",
        "edge-first-amendment",
        "edge-first-late",
        "edge-first-different-cik",
        "edge-first-anomaly",
    ]


def test_core_categories_are_exact_and_missing_slots_are_explicit_gaps() -> None:
    records = [
        _record("periodic", "10-Q", "2000-02-01", "2000-02-01T09:00:00"),
        # Amendments and similar-looking forms cannot fill exact core slots.
        _record("periodic-amendment", "10-Q/A", "2000-01-01", "2000-01-01T08:00:00"),
        _record("8k-amendment", "8-K/A", "2000-01-02", "2000-01-02T08:00:00"),
        _record("proxy-other", "DEFA14A", "2000-01-03", "2000-01-03T08:00:00"),
    ]
    result = select_sec_audit_filings(records)
    slots = {slot["slot_id"]: slot for slot in result["core"]}
    assert slots["core-2000-periodic"]["filing"]["accession"] == "periodic"
    assert slots["core-2000-8-k"]["status"] == "gap"
    assert slots["core-2000-def-14a"]["status"] == "gap"
    assert slots["core-2000-8-k"]["gap_reason"] == (
        "no_filing_in_exact_year_and_form_category"
    )
    assert all(
        slot["status"] == "gap"
        for slot in result["core"]
        if slot["requested_year"] != 2000
    )


def test_edge_selection_skips_used_accessions_only_within_same_condition() -> None:
    records = [
        _record(
            "core-xbrl",
            "10-Q",
            "2000-02-01",
            "2000-02-01T09:00:00",
            is_xbrl=True,
        ),
        _record(
            "second-xbrl",
            "S-3",
            "2001-01-02",
            "2001-01-02T09:00:00",
            is_xbrl=True,
        ),
    ]
    result = select_sec_audit_filings(records)
    edges = {slot["category"]: slot for slot in result["edge_cases"]}
    assert edges["first-xbrl"]["filing"]["accession"] == "second-xbrl"
    assert edges["first-xbrl"]["skipped_already_selected_accessions"] == [
        "core-xbrl"
    ]
    assert edges["first-anomaly"]["status"] == "gap"
    assert edges["first-anomaly"]["gap_reason"] == (
        "no_filing_matches_exact_edge_condition"
    )

    only_collision = select_sec_audit_filings(records[:1])
    collision_edge = {
        slot["category"]: slot for slot in only_collision["edge_cases"]
    }["first-xbrl"]
    assert collision_edge["status"] == "gap"
    assert collision_edge["gap_reason"] == "all_qualifying_filings_already_selected"


def test_after_1730_is_strict_eastern_and_acceptance_is_canonicalized() -> None:
    records = [
        _record(
            "exactly-1730",
            "S-3",
            "2002-01-02",
            "2002-01-02T22:30:00Z",  # 17:30 ET; not strictly after.
        ),
        _record(
            "after-1730",
            "S-3",
            "2002-01-03",
            "2002-01-03T22:31:00Z",  # 17:31 ET.
        ),
    ]
    result = select_sec_audit_filings(records)
    edge = {
        slot["category"]: slot for slot in result["edge_cases"]
    }["first-after-1730-et"]
    assert edge["filing"]["accession"] == "after-1730"
    assert edge["filing"]["acceptance_timestamp_et"] == (
        "2002-01-03T17:31:00-05:00"
    )


def test_cik_normalization_prevents_false_difference_and_finds_real_one() -> None:
    records = [
        _record(
            "same-cik",
            "S-3",
            "2001-01-02",
            "2001-01-02T09:00:00",
            submitter_cik="0000320193",
            subject_cik="320193",
        ),
        _record(
            "different-cik",
            "SC 13G",
            "2001-01-03",
            "2001-01-03T09:00:00",
            submitter_cik="123",
            subject_cik="320193",
        ),
    ]
    result = select_sec_audit_filings(records)
    edge = {
        slot["category"]: slot for slot in result["edge_cases"]
    }["first-differing-submitter-cik"]
    assert edge["filing"]["accession"] == "different-cik"
    assert edge["filing"]["submitter_cik"] == "0000000123"
    assert edge["filing"]["subject_cik"] == "0000320193"


def test_2019_and_2024_are_structural_only_and_metadata_output_is_closed() -> None:
    records = _complete_records()
    records[0]["body"] = "must not escape"
    records[0]["document_html"] = "<html>not metadata</html>"
    result = select_sec_audit_filings(records)
    assert result["role_policy"]["core_2019"] == STRUCTURAL_ROLE
    assert result["role_policy"]["core_2024"] == STRUCTURAL_ROLE
    for slot in result["core"]:
        expected = (
            STRUCTURAL_ROLE
            if slot["requested_year"] in (2019, 2024)
            else HISTORICAL_ROLE
        )
        assert slot["evidence_role"] == expected
        if slot["filing"] is not None:
            assert set(slot["filing"]) == {
                "accession",
                "form",
                "filing_year",
                "filing_date",
                "acceptance_timestamp_et",
                "is_xbrl",
                "submitter_cik",
                "subject_cik",
                "date_of_filing_date_change",
                "change_anomaly_flag",
            }
            assert "body" not in slot["filing"]
            assert "document_html" not in slot["filing"]


def test_input_order_and_attribute_records_do_not_change_selection() -> None:
    records = _complete_records()
    first = select_sec_audit_filings(records)
    mixed = [SimpleNamespace(**record) for record in reversed(records)]
    repeated = select_sec_audit_filings(mixed)
    assert repeated == first


def test_duplicate_accessions_collapse_only_when_metadata_is_identical() -> None:
    record = _record(
        "duplicate", "10-Q", "2000-02-01", "2000-02-01T09:00:00"
    )
    result = select_sec_audit_filings([record, copy.deepcopy(record)])
    assert result["input_record_count"] == 1
    assert result["core"][0]["filing"]["accession"] == "duplicate"

    conflict = copy.deepcopy(record)
    conflict["form"] = "8-K"
    with pytest.raises(ValueError, match="Conflicting metadata"):
        select_sec_audit_filings([record, conflict])


def test_invalid_required_metadata_fails_closed() -> None:
    record = _record(
        "bad-year", "10-Q", "2000-02-01", "2000-02-01T09:00:00"
    )
    record["filing_year"] = 2001
    with pytest.raises(ValueError, match="inconsistent"):
        select_sec_audit_filings([record])
    record = _record(
        "bad-cik", "10-Q", "2000-02-01", "2000-02-01T09:00:00"
    )
    record["submitter_cik"] = "not-a-cik"
    with pytest.raises(ValueError, match="CIK"):
        select_sec_audit_filings([record])


def test_parser_record_preserves_derived_submitter_cik_for_selection() -> None:
    record = FilingRecord(
        accession_number="0000912057-00-053623",
        acceptance_datetime="20001214170203",
        form="10-K",
        primary_document="a2032880z10-k.txt",
        items="",
        filing_date="2000-12-14",
        report_date="2000-09-30",
        is_xbrl=False,
        source_name="CIK0000320193-submissions-001.json",
    )
    result = select_sec_audit_filings([record])
    periodic = next(
        slot for slot in result["core"] if slot["slot_id"] == "core-2000-periodic"
    )
    assert periodic["filing"]["submitter_cik"] == "0000912057"
    differing = next(
        slot
        for slot in result["edge_cases"]
        if slot["category"] == "first-differing-submitter-cik"
    )
    assert differing["status"] == "gap"
    assert differing["skipped_already_selected_accessions"] == [
        "0000912057-00-053623"
    ]


def test_filing_date_change_is_preserved_and_cannot_be_negated() -> None:
    record = _record(
        "changed-date",
        "S-3",
        "2010-01-04",
        "2010-01-04T09:00:00",
        anomaly=False,
        dateOfFilingDateChange="2010-01-06",
    )
    result = select_sec_audit_filings([record])
    anomaly = next(
        slot
        for slot in result["edge_cases"]
        if slot["category"] == "first-anomaly"
    )
    assert anomaly["status"] == "selected"
    assert anomaly["filing"]["date_of_filing_date_change"] == "2010-01-06"
    assert anomaly["filing"]["change_anomaly_flag"] is True


def test_filing_records_integrate_every_core_and_edge_selection_role() -> None:
    records: list[FilingRecord] = []
    for position, raw in enumerate(_complete_records(), start=1):
        year = int(raw["filing_year"])
        prefix = (
            str(raw["submitter_cik"])
            if raw["accession"] == "edge-first-different-cik"
            else "0000320193"
        )
        accepted = datetime.fromisoformat(str(raw["acceptance_timestamp"]))
        records.append(
            FilingRecord(
                accession_number=(
                    f"{prefix}-{year % 100:02d}-{position:06d}"
                ),
                acceptance_datetime=accepted.strftime("%Y%m%d%H%M%S"),
                form=str(raw["form"]),
                primary_document=f"filing-{position}.htm",
                items="",
                filing_date=str(raw["filing_date"]),
                report_date="",
                is_xbrl=bool(raw["is_xbrl"]),
                source_name="CIK0000320193-submissions-001.json",
                subject_cik=str(raw["subject_cik"]),
                date_of_filing_date_change=(
                    "2010-01-05"
                    if raw["accession"] == "edge-first-anomaly"
                    else ""
                ),
            )
        )

    columns = {
        "accessionNumber": [record.accession_number for record in records],
        "acceptanceDateTime": [record.acceptance_datetime for record in records],
        "form": [record.form for record in records],
        "primaryDocument": [record.primary_document for record in records],
        "items": [record.items for record in records],
        "filingDate": [record.filing_date for record in records],
        "reportDate": [record.report_date for record in records],
        "isXBRL": [int(record.is_xbrl) for record in records],
        "dateOfFilingDateChange": [
            record.date_of_filing_date_change for record in records
        ],
    }
    parsed = parse_submissions_rows(
        {
            "cik": "320193",
            "filings": {"recent": columns, "files": []},
        }
    )
    assert all(isinstance(record, FilingRecord) for record in parsed)
    result = select_sec_audit_filings(parsed)
    assert result["selected_count"] == 24
    assert result["gap_count"] == 0
    assert len(result["core"]) == 18
    assert all(slot["status"] == "selected" for slot in result["core"])
    assert all(slot["status"] == "selected" for slot in result["edge_cases"])
    selected_edges = {slot["category"]: slot for slot in result["edge_cases"]}
    assert selected_edges["first-xbrl"]["filing"]["is_xbrl"] is True
    assert selected_edges["first-amendment"]["filing"]["form"].endswith("/A")
    assert selected_edges["first-differing-submitter-cik"]["filing"][
        "submitter_cik"
    ] == "0000000123"
    assert selected_edges["first-anomaly"]["filing"][
        "date_of_filing_date_change"
    ] == "2010-01-05"
