from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime

import pytest

from agent_benchmark.sec_point_in_time import (
    AAPL_CIK,
    MAX_AUDIT_BYTES,
    MAX_AUDIT_REQUESTS,
    MAX_AUDIT_SECONDS,
    BudgetCounter,
    FilingRecord,
    SecAuditLimitError,
    SecPointInTimeError,
    archive_urls,
    content_sha256,
    parse_acceptance_datetime,
    parse_master_idx,
    parse_submissions_rows,
    validate_sec_user_agent,
)


def _columns(*rows: dict) -> dict[str, list]:
    names = {
        "accessionNumber",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "items",
        "filingDate",
        "reportDate",
        "isXBRL",
    }
    if any("dateOfFilingDateChange" in row for row in rows):
        names.add("dateOfFilingDateChange")
    return {
        name: [
            row.get(name, "") if name == "dateOfFilingDateChange" else row[name]
            for row in rows
        ]
        for name in names
    }


def _row(
    accession: str,
    *,
    accepted: str,
    form: str,
    primary: str,
    items: str,
    filed: str,
    report: str,
    xbrl: int = 0,
    date_change: str | None = None,
) -> dict:
    result = {
        "accessionNumber": accession,
        "acceptanceDateTime": accepted,
        "form": form,
        "primaryDocument": primary,
        "items": items,
        "filingDate": filed,
        "reportDate": report,
        "isXBRL": xbrl,
    }
    if date_change is not None:
        result["dateOfFilingDateChange"] = date_change
    return result


def _filing(**overrides: object) -> FilingRecord:
    values = {
        "accession_number": "0001628280-16-020309",
        "acceptance_datetime": "20161026164216",
        "form": "10-K",
        "primary_document": "a201610-k9242016.htm",
        "items": "",
        "filing_date": "2016-10-26",
        "report_date": "2016-09-24",
        "is_xbrl": True,
        "source_name": "CIK0000320193.json",
    }
    values.update(overrides)
    return FilingRecord(**values)  # type: ignore[arg-type]


def test_user_agent_validation_returns_only_hash_and_never_contact() -> None:
    private = "Antonio Research antonio-private@antoniodomenech.dev"
    audit = validate_sec_user_agent(private)
    expected = "sha256:" + hashlib.sha256(private.encode()).hexdigest()
    assert audit.sha256 == expected
    assert audit.real_contact_validated is True
    serialized = json.dumps(asdict(audit), sort_keys=True)
    combined = repr(audit) + audit.redacted + serialized
    assert private not in combined
    assert "antonio-private" not in combined
    assert "antoniodomenech.dev" not in combined


@pytest.mark.parametrize(
    "value",
    (
        "",
        "research-bot-without-email",
        "contact@example.com",
        "Sample Company Name AdminContact@sample-company.com",
        "Research Bot <person@real-domain.com>",
        " Antonio Research antonio-private@antoniodomenech.dev ",
    ),
)
def test_user_agent_rejects_blank_or_placeholder_without_echo(value: str) -> None:
    with pytest.raises(SecPointInTimeError) as captured:
        validate_sec_user_agent(value)
    if value:
        assert value not in str(captured.value)
    assert "@" not in str(captured.value)


def test_parse_main_and_historical_submissions_preserves_required_fields() -> None:
    recent = _columns(
        _row(
            "0000320193-24-000123",
            accepted="20241101163001",
            form="10-K",
            primary="aapl-20240928.htm",
            items="1.01,2.02",
            filed="2024-11-01",
            report="2024-09-28",
            xbrl=1,
            date_change="2024-11-04",
        ),
        _row(
            "0000320193-24-000124",
            accepted="20241102120000",
            form="10-K/A",
            primary="aapl-20240928x10ka.htm",
            items="",
            filed="2024-11-02",
            report="2024-09-28",
            xbrl=1,
        ),
    )
    history_name = "CIK0000320193-submissions-001.json"
    historical = _columns(
        _row(
            "0000912057-00-053623",
            accepted="20001214170203",
            form="10-K",
            primary="a2032880z10-k.txt",
            items="",
            filed="2000-12-14",
            report="2000-09-30",
        )
    )
    main = {
        "cik": "320193",
        "filings": {
            "recent": recent,
            "files": [{"name": history_name, "filingFrom": "2000-01-01"}],
        },
    }
    records = parse_submissions_rows(main, {history_name: historical})
    assert [record.accession_number for record in records] == [
        "0000912057-00-053623",
        "0000320193-24-000123",
        "0000320193-24-000124",
    ]
    current = records[1]
    assert current.acceptance_datetime == "20241101163001"
    assert current.form == "10-K"
    assert current.primary_document == "aapl-20240928.htm"
    assert current.items == "1.01,2.02"
    assert current.is_xbrl is True
    assert current.date_of_filing_date_change == "2024-11-04"
    assert current.change_anomaly_flag is True
    assert records[2].is_amendment is True
    assert records[2].accession_number != current.accession_number
    assert records[0].submitter_cik == "0000912057"


def test_submissions_parser_fails_closed_on_misalignment_or_wrong_history() -> None:
    row = _row(
        "0000320193-24-000123",
        accepted="20241101163001",
        form="10-K",
        primary="aapl.htm",
        items="",
        filed="2024-11-01",
        report="2024-09-28",
    )
    columns = _columns(row)
    columns["items"].append("extra")
    main = {"cik": AAPL_CIK, "filings": {"recent": columns, "files": []}}
    with pytest.raises(SecPointInTimeError, match="row-aligned"):
        parse_submissions_rows(main)

    valid = {"cik": AAPL_CIK, "filings": {"recent": _columns(row), "files": []}}
    with pytest.raises(SecPointInTimeError, match="payload set"):
        parse_submissions_rows(valid, {"unreferenced.json": _columns(row)})


@pytest.mark.parametrize(
    "accepted",
    (
        "",
        "20241101163001extra",
        "20241301163001",
        "2024-11-01",
        "2024-11-01T16:30:01",
    ),
)
def test_submissions_parser_rejects_ambiguous_or_invalid_acceptance_formats(
    accepted: str,
) -> None:
    row = _row(
        "0000320193-24-000123",
        accepted=accepted,
        form="10-K",
        primary="aapl.htm",
        items="",
        filed="2024-11-01",
        report="2024-09-28",
    )
    main = {"cik": AAPL_CIK, "filings": {"recent": _columns(row), "files": []}}
    with pytest.raises(SecPointInTimeError, match="acceptanceDateTime"):
        parse_submissions_rows(main)


def test_submissions_parser_accepts_timezone_iso_and_validates_change_date() -> None:
    row = _row(
        "0000320193-24-000123",
        accepted="2024-11-01T20:30:01.000Z",
        form="10-K",
        primary="aapl.htm",
        items="",
        filed="2024-11-01",
        report="2024-09-28",
        date_change="2024-11-04",
    )
    main = {"cik": AAPL_CIK, "filings": {"recent": _columns(row), "files": []}}
    record = parse_submissions_rows(main)[0]
    assert record.acceptance_datetime == "2024-11-01T20:30:01.000Z"
    assert record.date_of_filing_date_change == "2024-11-04"

    row["dateOfFilingDateChange"] = "2024-13-04"
    broken = {"cik": AAPL_CIK, "filings": {"recent": _columns(row), "files": []}}
    with pytest.raises(SecPointInTimeError, match="dateOfFilingDateChange"):
        parse_submissions_rows(broken)


def test_parse_master_idx_preserves_aapl_and_third_party_accessions() -> None:
    payload = """Description: Master Index of EDGAR Dissemination Feed
CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
320193|APPLE COMPUTER INC|10-K|2000-12-14|edgar/data/320193/0000912057-00-053623.txt
789019|MICROSOFT CORP|10-Q|2000-01-25|edgar/data/789019/0001032210-00-000123.txt
"""
    records = parse_master_idx(payload)
    assert len(records) == 2
    apple = records[0]
    assert apple.cik == AAPL_CIK
    assert apple.form == "10-K"
    assert apple.accession_number == "0000912057-00-053623"
    assert apple.filename.endswith("0000912057-00-053623.txt")


def test_parse_acceptance_datetime_accepts_raw_or_sgml_and_rejects_ambiguity() -> None:
    expected = datetime(2016, 10, 26, 16, 42, 16)
    assert parse_acceptance_datetime("20161026164216") == expected
    sgml = b"<SEC-HEADER>\n<ACCEPTANCE-DATETIME>20161026164216\n"
    assert parse_acceptance_datetime(sgml) == expected
    with pytest.raises(SecPointInTimeError, match="unambiguous"):
        parse_acceptance_datetime(
            sgml + b"<ACCEPTANCE-DATETIME>20161026164217\n"
        )
    with pytest.raises(SecPointInTimeError, match="valid timestamp"):
        parse_acceptance_datetime("20161301120000")


def test_archive_urls_are_stable_and_reject_unsafe_primary_document() -> None:
    urls = archive_urls(_filing())
    base = "https://www.sec.gov/Archives/edgar/data/320193/000162828016020309"
    assert urls.directory == base + "/"
    assert urls.complete_submission == base + "/0001628280-16-020309.txt"
    assert urls.header == base + "/0001628280-16-020309.hdr.sgml"
    assert urls.index_json == base + "/index.json"
    assert urls.primary_document == base + "/a201610-k9242016.htm"
    with pytest.raises(SecPointInTimeError, match="unsafe"):
        archive_urls(replace(_filing(), primary_document="../secret.txt"))


def test_content_sha256_is_tagged_and_type_checked() -> None:
    assert content_sha256(b"abc") == (
        "sha256:ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    with pytest.raises(TypeError):
        content_sha256("abc")  # type: ignore[arg-type]


def test_budget_counter_enforces_inclusive_frozen_limits() -> None:
    now = [100.0]
    budget = BudgetCounter(clock=lambda: now[0])
    for _ in range(MAX_AUDIT_REQUESTS):
        budget.begin_request()
    budget.add_bytes(MAX_AUDIT_BYTES)
    now[0] += MAX_AUDIT_SECONDS
    snapshot = budget.snapshot()
    assert snapshot["requests"] == 100
    assert snapshot["bytes_received"] == 250 * 1024 * 1024
    with pytest.raises(SecAuditLimitError, match="request ceiling"):
        budget.begin_request()
    with pytest.raises(SecAuditLimitError, match="byte ceiling"):
        budget.add_bytes(1)
    now[0] += 0.001
    with pytest.raises(SecAuditLimitError, match="wall-clock"):
        budget.snapshot()


def test_budget_limits_can_only_be_tightened() -> None:
    with pytest.raises(SecAuditLimitError, match="request ceiling"):
        BudgetCounter(max_requests=101)
    with pytest.raises(SecAuditLimitError, match="byte ceiling"):
        BudgetCounter(max_bytes=MAX_AUDIT_BYTES + 1)
    with pytest.raises(SecAuditLimitError, match="time ceiling"):
        BudgetCounter(max_seconds=MAX_AUDIT_SECONDS + 1.0)
