from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from datetime import date, timedelta
from functools import lru_cache
import gzip
import hashlib
import json

import pytest

import agent_benchmark.sec_gemma_lean_v33_source as source_module

from agent_benchmark.sec_gemma_lean_v33_source import (
    MASTER_HEADER,
    STAGE_SOURCE_CONFIGS,
    CompactReplayInput,
    CompactReplayResult,
    MasterTargetRow,
    ReconciledTarget,
    RoleParseOutput,
    SecGemmaLeanV33SourceError,
    build_legacy_science_projection,
    build_compact_checkpoint,
    build_compact_role_manifest,
    build_pre_master_submissions_upper_bound,
    build_stage_source_seal,
    derive_complete_submission_plan,
    detached_replay,
    detached_replay_stage_source,
    parse_strict_master_gzip,
    parse_complete,
    parse_historical,
    parse_main,
    parse_master,
    parse_submissions_snapshot,
    quarter_roles,
    reconcile_complete_submission,
    reconcile_master_exact_set,
    reconcile_masters,
    rehydrate_compact_stage,
    finalize_stage,
    finalize_submissions,
    validate_completed_year_source_coverage,
    validate_cross_stage_frozen_prefix,
    validate_stage_source_seal,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _columns(rows: list[dict[str, object]]) -> dict[str, list[object]]:
    names = [
        "accessionNumber",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "items",
        "filingDate",
        "reportDate",
        "isXBRL",
    ]
    if any("dateOfFilingDateChange" in row for row in rows):
        names.append("dateOfFilingDateChange")
    return {
        name: [row.get(name, "") for row in rows]
        for name in names
    }


def _row(
    accession: str,
    filed: str,
    *,
    form: str = "10-Q",
    primary: str = "primary.htm",
    change: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "accessionNumber": accession,
        "acceptanceDateTime": filed.replace("-", "") + "120000",
        "form": form,
        "primaryDocument": primary,
        "items": "",
        "filingDate": filed,
        "reportDate": filed,
        "isXBRL": 1,
    }
    if change is not None:
        result["dateOfFilingDateChange"] = change
    return result


def _main_payload(
    rows: list[dict[str, object]],
    *,
    references: list[dict[str, object]] | None = None,
) -> bytes:
    value = {
        "cik": 320193,
        "filings": {
            "recent": _columns(rows),
            "files": references or [],
        },
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _complete_submission(
    row: dict[str, object],
    *,
    include_filename: bool = True,
) -> bytes:
    accession = str(row["accessionNumber"])
    filed = str(row["filingDate"])
    form = str(row["form"])
    primary = str(row["primaryDocument"])
    submitter = accession[:10]
    acceptance = source_module.parse_submissions_acceptance_datetime(
        str(row["acceptanceDateTime"])
    ).strftime("%Y%m%d%H%M%S")
    filename = f"<FILENAME>{primary}\n" if include_filename and primary else ""
    change = row.get("dateOfFilingDateChange")
    change_line = (
        f"DATE AS OF CHANGE: {str(change).replace('-', '')}\n"
        if change
        else ""
    )
    body = "AAPL filing evidence " + ("stable text " * 60)
    return (
        f"<SEC-DOCUMENT>{accession}.txt\n"
        "<SEC-HEADER>\n"
        f"<ACCESSION-NUMBER>{accession}\n"
        f"<ACCEPTANCE-DATETIME>{acceptance}\n"
        f"<CONFORMED-SUBMISSION-TYPE>{form}\n"
        f"<FILED-AS-OF-DATE>{filed.replace('-', '')}\n"
        f"{change_line}"
        "FILED BY:\n"
        "<COMPANY-CONFORMED-NAME>SUBMITTING AGENT\n"
        f"<CENTRAL-INDEX-KEY>{submitter}\n"
        "SUBJECT COMPANY:\n"
        "<COMPANY-CONFORMED-NAME>APPLE INC\n"
        "<CENTRAL-INDEX-KEY>320193\n"
        "</SEC-HEADER>\n"
        "<DOCUMENT>\n"
        f"<TYPE>{form}\n"
        "<SEQUENCE>1\n"
        f"{filename}"
        "<DESCRIPTION>PRIMARY PERIODIC REPORT\n"
        f"<TEXT><html><body>{body}</body></html></TEXT>\n"
        "</DOCUMENT>\n"
    ).encode("latin-1")


def _master_payload(lines: list[str]) -> bytes:
    text = "Description\n" + MASTER_HEADER + "\n" + "-" * 40 + "\n"
    if lines:
        text += "\n".join(lines) + "\n"
    return gzip.compress(text.encode("latin-1"), mtime=0)


def _master_line(row: dict[str, object]) -> str:
    accession = str(row["accessionNumber"])
    return (
        f"320193|APPLE INC|{row['form']}|{row['filingDate']}|"
        f"edgar/data/320193/{accession}.txt"
    )


def _quarter_for(value: str) -> tuple[int, int]:
    parsed = date.fromisoformat(value)
    return parsed.year, (parsed.month - 1) // 3 + 1


def _target_for_row(row_value: dict[str, object]) -> ReconciledTarget:
    snapshot = parse_submissions_snapshot(_main_payload([row_value]), {})
    row = snapshot.rows[0]
    return ReconciledTarget(
        row,
        MasterTargetRow(
            accession_number=row.accession_number,
            cik="0000320193",
            company_name="APPLE INC",
            form=row.form,
            filing_date=row.filing_date,
            filename=f"edgar/data/320193/{row.accession_number}.txt",
            quarter=_quarter_for(row.filing_date),
        ),
    )


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("stage_source_seal_sha256", None)
    value["stage_source_seal_sha256"] = _canonical_hash(value)
    return value


def test_frozen_stage_configurations_have_exact_quarter_roles() -> None:
    expected = {
        "development": (100, (1994, 1), (2018, 4), 128, 128, 245),
        "intermediate": (120, (1994, 1), (2023, 4), 160, 24, 161),
        "final": (131, (1994, 1), (2026, 3), 176, 16, 164),
    }
    for stage, (count, first, last, i_cap, p_cap, request_cap) in expected.items():
        roles = quarter_roles(stage)
        assert len(roles) == count
        assert roles[0].key == first
        assert roles[1].key == (1994, 2)
        assert roles[-1].key == last
        assert all(
            current.key
            == (
                previous.year + (1 if previous.quarter == 4 else 0),
                1 if previous.quarter == 4 else previous.quarter + 1,
            )
            for previous, current in zip(roles, roles[1:])
        )
        assert STAGE_SOURCE_CONFIGS[stage].i_cap == i_cap
        assert STAGE_SOURCE_CONFIGS[stage].p_cap == p_cap
        assert STAGE_SOURCE_CONFIGS[stage].maximum_successful_requests == request_cap
        assert roles[0].url.endswith(f"/{first[0]}/QTR{first[1]}/master.gz")
        assert roles[1].url.endswith("/1994/QTR2/master.gz")


def test_q1_observed_filing_uses_general_bound_and_pre_1994_rejects() -> None:
    observed = _row(
        "0000320193-94-000002",
        "1994-01-26",
        primary="",
    )
    observed["acceptanceDateTime"] = "1994-01-26T05:00:00.000Z"
    observed["reportDate"] = "1993-12-31"
    snapshot = parse_submissions_snapshot(_main_payload([observed]), {})
    receipt = build_pre_master_submissions_upper_bound(snapshot, "development")
    assert receipt["i_upper_bound"] == 1
    assert receipt["p_upper_bound"] == 1
    assert receipt["candidate_accessions"] == ["0000320193-94-000002"]

    before_boundary = _row(
        "0000320193-93-000001",
        "1993-12-31",
        primary="",
    )
    old_snapshot = parse_submissions_snapshot(_main_payload([before_boundary]), {})
    with pytest.raises(
        SecGemmaLeanV33SourceError,
        match="^master_boundary_incomplete$",
    ):
        build_pre_master_submissions_upper_bound(old_snapshot, "development")


def test_exact_1994_q1_boundary_target_reconciles_by_general_rule() -> None:
    boundary = _row(
        "0000320193-94-000001",
        "1994-01-01",
        primary="",
    )
    boundary["reportDate"] = "1993-09-30"
    snapshot = parse_submissions_snapshot(_main_payload([boundary]), {})
    masters = [
        parse_strict_master_gzip(
            _master_payload([_master_line(boundary)] if role.key == (1994, 1) else []),
            year=role.year,
            quarter=role.quarter,
        )
        for role in quarter_roles("development")
    ]

    reconciled = reconcile_master_exact_set(snapshot, masters, "development")

    assert reconciled.accessions == ("0000320193-94-000001",)
    assert reconciled.targets[0].master.quarter == (1994, 1)
    assert reconciled.targets[0].master.filename == (
        "edgar/data/320193/0000320193-94-000001.txt"
    )


def test_q1_q2_master_authority_is_mandatory_and_q1_must_match() -> None:
    observed = _row(
        "0000320193-94-000002",
        "1994-01-26",
        primary="",
    )
    observed["acceptanceDateTime"] = "1994-01-26T05:00:00.000Z"
    observed["reportDate"] = "1993-12-31"
    snapshot = parse_submissions_snapshot(_main_payload([observed]), {})
    masters = [
        parse_strict_master_gzip(
            _master_payload([_master_line(observed)] if role.key == (1994, 1) else []),
            year=role.year,
            quarter=role.quarter,
        )
        for role in quarter_roles("development")
    ]
    reconciled = reconcile_master_exact_set(snapshot, masters, "development")
    assert reconciled.accessions == ("0000320193-94-000002",)

    without_q2 = [item for item in masters if item.role.key != (1994, 2)]
    with pytest.raises(SecGemmaLeanV33SourceError, match="quarter evidence set"):
        reconcile_master_exact_set(snapshot, without_q2, "development")

    q1_empty = parse_strict_master_gzip(
        _master_payload([]),
        year=1994,
        quarter=1,
    )
    mismatched_q1 = [
        q1_empty if item.role.key == (1994, 1) else item for item in masters
    ]
    with pytest.raises(SecGemmaLeanV33SourceError, match="target set differs"):
        reconcile_master_exact_set(snapshot, mismatched_q1, "development")


def test_submissions_snapshot_accepts_unoccupied_inclusive_bounds_and_layout() -> None:
    history_name = "CIK0000320193-submissions-001.json"
    historical_row = _row(
        "0000912057-00-023442", "2000-05-26", primary=""
    )
    recent_row = _row("1234567890-24-000001", "2024-02-01")
    main = _main_payload(
        [recent_row],
        references=[
            {
                "name": history_name,
                "filingCount": 1,
                "filingFrom": "2000-01-01",
                "filingTo": "2000-12-31",
            }
        ],
    )
    history = json.dumps(_columns([historical_row]), separators=(",", ":")).encode()
    snapshot = parse_submissions_snapshot(main, {history_name: history})
    assert [row.accession_number for row in snapshot.rows] == [
        "0000912057-00-023442",
        "1234567890-24-000001",
    ]
    evidence = snapshot.sources[1].range_evidence
    assert evidence is not None
    assert evidence["filing_from_attained"] is False
    assert evidence["filing_to_attained"] is False
    assert evidence["lower_bound_slack_days"] > 0
    assert evidence["upper_bound_slack_days"] > 0
    assert snapshot.rows[0].source_name == history_name
    assert snapshot.rows[1].source_name == "CIK0000320193.json"

    duplicate = _main_payload(
        [historical_row],
        references=[
            {
                "name": history_name,
                "filingCount": 1,
                "filingFrom": "2000-01-01",
                "filingTo": "2000-12-31",
            }
        ],
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="repeats"):
        parse_submissions_snapshot(duplicate, {history_name: history})

    duplicate_key = main.replace(b'"cik":320193', b'"cik":320193,"cik":320193')
    with pytest.raises(SecGemmaLeanV33SourceError, match="duplicate JSON object key"):
        parse_submissions_snapshot(duplicate_key, {history_name: history})

    float_flag = json.loads(main)
    float_flag["filings"]["recent"]["isXBRL"] = [1.0]
    with pytest.raises(SecGemmaLeanV33SourceError, match="zero/one flag"):
        parse_submissions_snapshot(
            json.dumps(float_flag, separators=(",", ":")).encode(),
            {history_name: history},
        )

    second_name = "CIK0000320193-submissions-002.json"
    second_row = _row("2222222222-01-000002", "2001-06-15")
    second_history = json.dumps(
        _columns([second_row]), separators=(",", ":")
    ).encode()
    first_ref = {
        "name": history_name,
        "filingCount": 1,
        "filingFrom": "2000-01-01",
        "filingTo": "2000-12-31",
    }
    second_ref = {
        "name": second_name,
        "filingCount": 1,
        "filingFrom": "2001-01-01",
        "filingTo": "2001-12-31",
    }
    payload_map = {history_name: history, second_name: second_history}
    ordered = parse_submissions_snapshot(
        _main_payload([recent_row], references=[first_ref, second_ref]), payload_map
    )
    reordered = parse_submissions_snapshot(
        _main_payload([recent_row], references=[second_ref, first_ref]), payload_map
    )
    assert [row.semantic_identity_sha256 for row in ordered.rows] == [
        row.semantic_identity_sha256 for row in reordered.rows
    ]
    assert ordered.snapshot_sha256 != reordered.snapshot_sha256
    assert ordered.summary()["canonical_historical_fetch_order"] == [
        history_name,
        second_name,
    ]


def test_strict_master_gzip_and_exact_apple_projection() -> None:
    row = _row("1234567890-24-000001", "2024-02-01")
    payload = _master_payload([_master_line(row)])
    evidence = parse_strict_master_gzip(payload, year=2024, quarter=1)
    assert not hasattr(evidence, "decompressed_payload")
    assert evidence.raw_row_count == 1
    assert evidence.target_rows[0].accession_number == "1234567890-24-000001"
    assert evidence.target_rows[0].complete_submission_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "1234567890-24-000001.txt"
    )

    with pytest.raises(SecGemmaLeanV33SourceError, match="trailing|concatenated"):
        parse_strict_master_gzip(payload + _master_payload([]), year=2024, quarter=1)
    repeated = _master_payload([_master_line(row), _master_line(row)])
    with pytest.raises(SecGemmaLeanV33SourceError, match="repeated raw row"):
        parse_strict_master_gzip(repeated, year=2024, quarter=1)
    malformed = gzip.compress(
        (MASTER_HEADER + "\n320193|APPLE|10-Q|2024-02-01\n").encode(),
        mtime=0,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="five fields"):
        parse_strict_master_gzip(malformed, year=2024, quarter=1)
    bad_control = gzip.compress(
        (MASTER_HEADER + "\x0b320193|APPLE|10-Q|2024-02-01|x\n").encode("latin-1"),
        mtime=0,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="line-control"):
        parse_strict_master_gzip(bad_control, year=2024, quarter=1)


def test_blank_company_non_targets_are_retained_by_general_target_predicate() -> None:
    lines = [
        "1111111111||8-K|2024-01-02|edgar/data/1111111111/current.txt",
        "2222222222||10-Q|2024-01-03|edgar/data/2222222222/quarterly.txt",
        "320193||8-K|2024-01-04|edgar/data/320193/current.txt",
        (
            "320193|APPLE INC|10-Q|2024-02-01|"
            "edgar/data/320193/0000320193-24-000001.txt"
        ),
    ]
    evidence = parse_strict_master_gzip(
        _master_payload(lines), year=2024, quarter=1
    )
    assert evidence.raw_row_count == len(lines)
    assert evidence.raw_row_order_sha256 == _canonical_hash(lines)
    assert [target.accession_number for target in evidence.target_rows] == [
        "0000320193-24-000001"
    ]

    changed = [*lines]
    changed[0] = (
        "1111111111||8-K|2024-01-02|"
        "edgar/data/1111111111/amended-name.txt"
    )
    changed_evidence = parse_strict_master_gzip(
        _master_payload(changed), year=2024, quarter=1
    )
    assert changed_evidence.raw_row_count == len(lines)
    assert changed_evidence.raw_row_order_sha256 == _canonical_hash(changed)
    assert changed_evidence.raw_row_order_sha256 != evidence.raw_row_order_sha256
    assert changed_evidence.target_rows == evidence.target_rows


def test_blank_company_apple_targets_and_other_empty_fields_reject() -> None:
    for cik in ("320193", "0000320193"):
        for form in ("10-K", "10-Q"):
            line = (
                f"{cik}||{form}|2024-02-01|"
                "edgar/data/320193/0000320193-24-000002.txt"
            )
            with pytest.raises(
                SecGemmaLeanV33SourceError,
                match="master.gz row contains an empty field",
            ):
                parse_strict_master_gzip(
                    _master_payload([line]), year=2024, quarter=1
                )

    empty_cases = (
        (
            "|VENDOR|8-K|2024-02-01|edgar/data/4444444444/current.txt",
            "canonical SEC CIK",
        ),
        (
            "4444444444|VENDOR||2024-02-01|edgar/data/4444444444/current.txt",
            "empty field",
        ),
        (
            "4444444444|VENDOR|8-K||edgar/data/4444444444/current.txt",
            "canonical ISO date",
        ),
        ("4444444444|VENDOR|8-K|2024-02-01|", "empty field"),
    )
    for line, message in empty_cases:
        with pytest.raises(SecGemmaLeanV33SourceError, match=message):
            parse_strict_master_gzip(
                _master_payload([line]), year=2024, quarter=1
            )


def test_parse_master_blank_company_rule_preserves_compact_evidence_without_predecessor_reads() -> None:
    lines = [
        "7777777777||8-K|2004-01-02|edgar/data/7777777777/current.txt",
        (
            "320193|APPLE INC|10-K|2004-02-01|"
            "edgar/data/320193/0000320193-04-000007.txt"
        ),
    ]
    output = parse_master(
        "development",
        2004,
        1,
        _master_payload(lines),
    )
    assert output.evidence["summary"]["raw_row_count"] == len(lines)
    assert output.evidence["summary"]["raw_row_order_sha256"] == _canonical_hash(
        lines
    )
    assert [row["accession_number"] for row in output.evidence["target_rows"]] == [
        "0000320193-04-000007"
    ]
    compact_json = json.dumps(output.evidence, sort_keys=True)
    assert lines[0] not in compact_json
    assert "sec_gemma_lean_v32" not in compact_json
    assert output.historical_filenames == ()
    assert output.decompressed_bytes is not None


def test_full_synthetic_source_seal_replays_blank_company_non_target() -> None:
    main, historical, rows, responses = _synthetic_full_rehearsal_inputs()
    development_rows = [
        row for row in rows if str(row["filingDate"]) <= "2018-12-31"
    ]
    master_payloads = _master_set("development", rows)
    blank_non_target = (
        "4444444444||8-K|1999-10-15|edgar/data/4444444444/current.txt"
    )
    master_payloads[(1999, 4)] = _master_payload([blank_non_target])
    bundle = build_stage_source_seal(
        stage="development",
        main_submissions_payload=main,
        historical_submissions_payloads=historical,
        master_gzip_payloads=master_payloads,
        complete_submission_payloads={
            str(row["accessionNumber"]): responses[str(row["accessionNumber"])]
            for row in development_rows
        },
        session_dates=EXPECTED_SESSIONS,
    )
    quarter = next(
        evidence
        for evidence in bundle.master_reconciliation.quarter_evidence
        if evidence.role.key == (1999, 4)
    )
    assert quarter.raw_row_count == 1
    assert quarter.raw_row_order_sha256 == _canonical_hash([blank_non_target])
    assert quarter.target_rows == ()
    replay = detached_replay_stage_source(bundle)
    assert replay["stage_source_seal_sha256"] == bundle.stage_source_seal_sha256
    assert replay["replayed_stage_source_seal_sha256"] == (
        bundle.stage_source_seal_sha256
    )
    assert replay["exact_seal_json_match"] is True


def test_complete_response_extracted_and_normalized_provenance_are_distinct() -> None:
    raw_row = _row(
        "0000912057-00-023442",
        "2000-05-26",
        primary="legacy-primary.htm",
    )
    snapshot = parse_submissions_snapshot(_main_payload([raw_row]), {})
    row = snapshot.rows[0]
    master = MasterTargetRow(
        accession_number=row.accession_number,
        cik="0000320193",
        company_name="APPLE INC",
        form=row.form,
        filing_date=row.filing_date,
        filename=f"edgar/data/320193/{row.accession_number}.txt",
        quarter=(2000, 2),
    )
    response = _complete_submission(raw_row, include_filename=False)
    source = reconcile_complete_submission(
        ReconciledTarget(row, master),
        response,
        session_dates=EXPECTED_SESSIONS,
        acquisition_stage="development",
    )
    selection = source.seal_row["primary_selection"]
    assert selection["sec_filename"] == "legacy-primary.htm"
    assert selection["document_identity"] == "legacy-primary.htm"
    assert selection["sgml_document_identity"] == (
        "legacy-sequence-1-no-filename"
    )
    assert selection["sgml_filename_missing"] is True
    assert selection["submissions_filename_missing"] is False
    assert source.seal_row["complete_response"]["sha256"] != (
        source.seal_row["extracted_text"]["sha256"]
    )
    assert source.seal_row["extracted_text"]["sha256"] != (
        source.seal_row["normalized_text"]["sha256"]
    )
    start = source.seal_row["extracted_text"]["start_byte"]
    end = source.seal_row["extracted_text"]["end_byte"]
    assert response[start:end] == source.extracted_primary


def _synthetic_full_rehearsal_inputs() -> tuple[
    bytes,
    dict[str, bytes],
    list[dict[str, object]],
    dict[str, bytes],
]:
    development_rows: list[dict[str, object]] = []
    intermediate_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    index = 1

    def add(
        destination: list[dict[str, object]],
        filed: str,
        *,
        form: str,
        change: str | None = None,
    ) -> None:
        nonlocal index
        accession = f"{9_000_000_000 + index:010d}-{filed[2:4]}-{index:06d}"
        destination.append(
            _row(
                accession,
                filed,
                form=form,
                primary=f"p{index}.htm",
                change=change,
            )
        )
        index += 1

    # The observed Q1 filing is an ordinary member.  The pinned session
    # calendar begins in 2000, so the unchanged availability rule assigns it
    # the first supported development session without any special case.
    observed_q1 = _row(
        "0000320193-94-000002",
        "1994-01-26",
        primary="",
    )
    observed_q1["acceptanceDateTime"] = "1994-01-26T05:00:00.000Z"
    observed_q1["reportDate"] = "1993-12-31"
    development_rows.append(observed_q1)

    # Another 55 verification-only members are inside master coverage but
    # delayed beyond the pinned calendar.  The observed row plus the remaining
    # 72 rows produce an admitted development D of 73.
    for offset in range(55):
        year = 1995 + offset // 12
        month = offset % 12 + 1
        add(
            development_rows,
            date(year, month, 1).isoformat(),
            form="10-Q",
            change="2026-07-10",
        )
    for year in range(2000, 2019):
        add(development_rows, f"{year}-02-01", form="10-K")
        add(development_rows, f"{year}-05-01", form="10-Q")
        add(development_rows, f"{year}-08-01", form="10-Q")
    for year in range(2000, 2015):
        add(development_rows, f"{year}-11-01", form="10-Q")
    assert len(development_rows) == 128

    # 19 are admitted in 2019-2023.  Five more are acquired now but their
    # authenticated change dates assign them to 2024.
    for year in range(2019, 2024):
        add(intermediate_rows, f"{year}-02-01", form="10-K")
        add(intermediate_rows, f"{year}-05-01", form="10-Q")
        add(intermediate_rows, f"{year}-08-01", form="10-Q")
    for year in range(2019, 2023):
        add(intermediate_rows, f"{year}-11-01", form="10-Q")
    for offset, form in enumerate(("10-K", "10-Q", "10-Q", "10-Q", "10-Q")):
        add(
            intermediate_rows,
            f"2023-12-{10 + offset:02d}",
            form=form,
            change=f"2024-01-{6 - offset:02d}",
        )
    assert len(intermediate_rows) == 24

    # P is 16.  Together with the five delayed members above, seven new rows
    # enter final D (12 total); nine remain verification-only beyond calendar.
    for month, form in ((2, "10-K"), (5, "10-Q"), (8, "10-Q")):
        add(final_rows, f"2025-{month:02d}-03", form=form)
    for month, form in ((1, "10-K"), (2, "10-Q"), (3, "10-Q"), (4, "10-Q")):
        add(final_rows, f"2026-{month:02d}-03", form=form)
    for offset in range(9):
        add(
            final_rows,
            f"2026-06-{20 + offset:02d}",
            form="10-Q",
            change="2026-07-10",
        )
    assert len(final_rows) == 16

    # Exercise the maximum H=16.  Each referenced body has exactly eight rows.
    references: list[dict[str, object]] = []
    historical: dict[str, bytes] = {}
    ordered_development = sorted(
        development_rows,
        key=lambda row: (str(row["filingDate"]), str(row["accessionNumber"])),
    )
    for number in range(16):
        group = ordered_development[number * 8 : (number + 1) * 8]
        name = f"CIK0000320193-submissions-{number + 1:03d}.json"
        dates = [str(row["filingDate"]) for row in group]
        references.append(
            {
                "name": name,
                "filingCount": 8,
                "filingFrom": min(dates),
                "filingTo": max(dates),
            }
        )
        historical[name] = json.dumps(
            _columns(group), separators=(",", ":")
        ).encode()
    later_rows = intermediate_rows + final_rows
    non_target = _row(
        "7777777777-24-999999",
        "2024-03-01",
        form="8-K",
        primary="xslF345X03/current-report.xml",
    )
    # The full SEC catalogue contains many authenticated non-target forms.
    # Keep one in Submissions only: it must survive compact replay but must
    # never enter I/P/D, the master target set, or complete-response requests.
    main = _main_payload([*later_rows, non_target], references=references)
    all_rows = development_rows + later_rows
    responses = {
        str(row["accessionNumber"]): _complete_submission(row) for row in all_rows
    }
    return main, historical, all_rows, responses


def _master_set(stage: str, rows: list[dict[str, object]]) -> dict[tuple[int, int], bytes]:
    config = STAGE_SOURCE_CONFIGS[stage]
    grouped: dict[tuple[int, int], list[str]] = {
        role.key: [] for role in quarter_roles(stage)
    }
    for row in rows:
        if str(row["filingDate"]) > config.fixed_cutoff:
            continue
        key = _quarter_for(str(row["filingDate"]))
        grouped[key].append(_master_line(row))
    return {key: _master_payload(lines) for key, lines in grouped.items()}


@lru_cache(maxsize=1)
def _built_full_chain() -> tuple[object, ...]:
    main, historical, rows, responses = _synthetic_full_rehearsal_inputs()
    development_rows = [
        row for row in rows if str(row["filingDate"]) <= "2018-12-31"
    ]
    intermediate_rows = [
        row for row in rows if str(row["filingDate"]) <= "2023-12-31"
    ]
    development = build_stage_source_seal(
        stage="development",
        main_submissions_payload=main,
        historical_submissions_payloads=historical,
        master_gzip_payloads=_master_set("development", rows),
        complete_submission_payloads={
            str(row["accessionNumber"]): responses[str(row["accessionNumber"])]
            for row in development_rows
        },
        session_dates=EXPECTED_SESSIONS,
    )
    intermediate = build_stage_source_seal(
        stage="intermediate",
        main_submissions_payload=main,
        historical_submissions_payloads=historical,
        master_gzip_payloads=_master_set("intermediate", rows),
        complete_submission_payloads={
            str(row["accessionNumber"]): responses[str(row["accessionNumber"])]
            for row in intermediate_rows
            if str(row["filingDate"]) > "2018-12-31"
        },
        session_dates=EXPECTED_SESSIONS,
        prior_bundle=development,
    )
    final = build_stage_source_seal(
        stage="final",
        main_submissions_payload=main,
        historical_submissions_payloads=historical,
        master_gzip_payloads=_master_set("final", rows),
        complete_submission_payloads={
            str(row["accessionNumber"]): responses[str(row["accessionNumber"])]
            for row in rows
            if str(row["filingDate"]) > "2023-12-31"
        },
        session_dates=EXPECTED_SESSIONS,
        prior_bundle=intermediate,
    )
    return main, historical, rows, responses, development, intermediate, final


def _receipt_hash(*parts: object) -> str:
    return _canonical_hash(list(parts))


def _build_compact_stage(
    stage: str,
    *,
    prior_output: object | None,
    prior_replay: CompactReplayInput | None,
) -> tuple[object, CompactReplayInput, dict[str, bytes]]:
    main_payload, historical_payloads, rows, responses = (
        _synthetic_full_rehearsal_inputs()
    )
    role_artifacts: list[tuple[str, str, bytes, RoleParseOutput]] = []
    main_output = parse_main(stage, main_payload)
    role_artifacts.append(
        ("submissions/main", source_module.MAIN_SUBMISSIONS_URL, main_payload, main_output)
    )
    historical_outputs: list[RoleParseOutput] = []
    for filename in main_output.historical_filenames:
        payload = historical_payloads[filename]
        output = parse_historical(stage, filename, payload, main_output)
        historical_outputs.append(output)
        role_artifacts.append(
            (
                f"submissions/historical/{filename}",
                f"https://data.sec.gov/submissions/{filename}",
                payload,
                output,
            )
        )
    phase = finalize_submissions(
        stage,
        main_output,
        historical_outputs,
        prior_output,
    )
    master_payloads = _master_set(stage, rows)
    master_outputs: list[RoleParseOutput] = []
    for role in quarter_roles(stage):
        payload = master_payloads[role.key]
        output = parse_master(stage, role.year, role.quarter, payload)
        master_outputs.append(output)
        role_artifacts.append(
            (
                f"master/{role.year}/QTR{role.quarter}",
                role.url,
                payload,
                output,
            )
        )
    reconciliation = reconcile_masters(
        stage,
        phase,
        master_outputs,
        prior_output,
    )
    complete_outputs: list[RoleParseOutput] = []
    for target in reconciliation.complete_targets:
        accession = str(target["submissions"]["accession_number"])
        payload = responses[accession]
        output = parse_complete(stage, target, payload, reconciliation)
        complete_outputs.append(output)
        role_artifacts.append(
            (
                f"complete/{accession}",
                str(target["master"]["complete_submission_url"]),
                payload,
                output,
            )
        )
    stage_output = finalize_stage(
        stage,
        phase,
        reconciliation,
        complete_outputs,
        prior_output,
    )
    blobs: dict[str, bytes] = {}
    manifests: list[dict[str, object]] = []
    for sequence, (role_id, url, payload, output) in enumerate(role_artifacts):
        blob_name = f"{stage}-{sequence:03d}.blob"
        blobs[blob_name] = payload
        manifests.append(
            dict(
                build_compact_role_manifest(
                    sequence=sequence,
                    role_id=role_id,
                    url=url,
                    blob_name=blob_name,
                    payload=payload,
                    parse_output=output,
                    transport_receipt_sha256=_receipt_hash(
                        "transport", stage, role_id
                    ),
                    parse_receipt_sha256=_receipt_hash("parse", stage, role_id),
                )
            )
        )
    checkpoint = build_compact_checkpoint(
        stage=stage,
        role_manifests=manifests,
        stage_output=stage_output,
        main_parse_receipt_sha256=str(manifests[0]["parse_receipt_sha256"]),
        submissions_snapshot_receipt_sha256=_receipt_hash(
            "submissions-phase", stage
        ),
        reconciliation_and_prior_chain_receipt_sha256=_receipt_hash(
            "reconciliation-phase", stage
        ),
    )
    replay_input = CompactReplayInput(
        stage=stage,
        checkpoint=checkpoint,
        role_manifests=tuple(manifests),
        prior=prior_replay,
    )
    return stage_output, replay_input, blobs


@lru_cache(maxsize=1)
def _built_compact_chain() -> tuple[object, ...]:
    development, development_replay, dev_blobs = _build_compact_stage(
        "development",
        prior_output=None,
        prior_replay=None,
    )
    intermediate, intermediate_replay, int_blobs = _build_compact_stage(
        "intermediate",
        prior_output=development,
        prior_replay=development_replay,
    )
    final, final_replay, final_blobs = _build_compact_stage(
        "final",
        prior_output=intermediate,
        prior_replay=intermediate_replay,
    )
    blobs = {**dev_blobs, **int_blobs, **final_blobs}
    return development, intermediate, final, final_replay, blobs


def test_synthetic_full_size_three_stage_rehearsal_replay_and_prefix() -> None:
    _, _, _, _, development, intermediate, final = _built_full_chain()
    assert development.seal["counts"] == {
        "H": 16,
        "Q": 100,
        "I": 128,
        "P": 128,
        "U": 73,
        "D": 73,
        "exact_acceptance_U": 73,
        "successful_request_formula": 245,
    }
    assert [
        (item["year"], item["quarter"])
        for item in development.seal["master_quarters"][:2]
    ] == [(1994, 1), (1994, 2)]
    observed = next(
        row
        for row in development.seal["target_rows"]
        if row["accession_number"] == "0000320193-94-000002"
    )
    assert observed["frozen_prefix"]["master_identity"] == {
        "accession_number": "0000320193-94-000002",
        "cik": "0000320193",
        "form": "10-Q",
        "filing_date": "1994-01-26",
        "filename": "edgar/data/320193/0000320193-94-000002.txt",
    }
    assert observed["primary_selection"]["document_identity"] == (
        "legacy-sequence-1-no-filename"
    )
    assert observed["availability_session"] == EXPECTED_SESSIONS[0]
    assert observed["stage_assignment"] == "development"
    assert detached_replay_stage_source(development)["exact_seal_json_match"] is True

    assert intermediate.seal["counts"]["I"] == 152
    assert intermediate.seal["counts"]["P"] == 24
    assert intermediate.seal["counts"]["D"] == 19
    assert intermediate.seal["counts"]["Q"] == 120
    assert intermediate.seal["counts"]["successful_request_formula"] == 161

    assert final.seal["counts"]["I"] == 168
    assert final.seal["counts"]["P"] == 16
    assert final.seal["counts"]["D"] == 12
    assert final.seal["counts"]["Q"] == 131
    assert final.seal["counts"]["successful_request_formula"] == 164
    assert final.seal["completed_year_source_coverage"]["years"][0]["year"] == 2000
    assert final.seal["completed_year_source_coverage"]["years"][-1]["year"] == 2025
    assert detached_replay_stage_source(final)["exact_seal_json_match"] is True

    projection = build_legacy_science_projection(final)
    assert len(projection.documents) == 12
    projected_order = [document.accession_number for document in projection.documents]
    row_index = {row["accession_number"]: row for row in final.seal["target_rows"]}
    assert projected_order == sorted(
        projected_order,
        key=lambda accession: (
            row_index[accession]["availability_session"], accession
        ),
    )
    projected = projection.documents[0]
    source = final.sources_by_accession[projected.accession_number]
    assert projected.raw_primary_document == source.extracted_primary
    assert projected.raw_primary_document != source.complete_response

    # An identical typed row may migrate between main/history layout evidence.
    layout_only = deepcopy(final.seal)
    layout_only["target_rows"][0]["submissions_layout_provenance"][
        "source_name"
    ] = "CIK0000320193-submissions-999.json"
    layout_only.pop("stage_source_seal_sha256")
    layout_only["stage_source_seal_sha256"] = _canonical_hash(layout_only)
    validate_cross_stage_frozen_prefix(intermediate.seal, layout_only)

    semantic_drift = deepcopy(final.seal)
    semantic_drift["target_rows"][0]["frozen_prefix"][
        "availability_session"
    ] = "2000-02-03"
    semantic_drift["target_rows"][0]["frozen_prefix_sha256"] = _canonical_hash(
        semantic_drift["target_rows"][0]["frozen_prefix"]
    )
    semantic_drift.pop("stage_source_seal_sha256")
    semantic_drift["stage_source_seal_sha256"] = _canonical_hash(semantic_drift)
    with pytest.raises(
        SecGemmaLeanV33SourceError,
        match="drifted|not authoritative",
    ):
        validate_cross_stage_frozen_prefix(intermediate.seal, semantic_drift)


def _cross_stage_drift_row(
    seal: dict[str, object],
    *,
    availability_year: str = "2000",
) -> dict[str, object]:
    return next(
        row
        for row in seal["target_rows"]
        if row["acquisition_stage"] == "development"
        and row["stage_assignment"] == "development"
        and row["form"] == "10-Q"
        and row["accession_number"] != "0000320193-94-000002"
        and row["availability_session"].startswith(availability_year)
    )


def _set_cross_stage_change_date(
    row: dict[str, object],
    value: str,
    *,
    availability_session: str,
    stage_assignment: str,
) -> None:
    prefix = row["frozen_prefix"]
    semantic = prefix["submissions_semantic_identity"]
    semantic["date_of_filing_date_change_present"] = True
    semantic["typed_values"]["dateOfFilingDateChange"] = value
    prefix["header_identity"]["date_of_filing_date_change"] = value
    prefix["filing_date_change"] = {
        "submissions_source": value,
        "submissions_column_present": True,
        "header_source": value,
        "normalized": value,
        "submissions_missing": False,
        "header_missing": False,
    }
    prefix["availability_not_before_date"] = value
    prefix["availability_session"] = availability_session
    prefix["immutable_stage_assignment"] = stage_assignment
    row["availability_session"] = availability_session
    row["stage_assignment"] = stage_assignment


def _refresh_cross_stage_forgery(seal: dict[str, object]) -> None:
    rows = seal["target_rows"]
    u_rows = sorted(
        (
            row
            for row in rows
            if row["stage_assignment"] in {"development", "intermediate", "final"}
            and type(row["availability_session"]) is str
            and "2000-01-01"
            <= row["availability_session"]
            <= "2026-07-09"
        ),
        key=lambda row: (row["availability_session"], row["accession_number"]),
    )
    seal["u_accessions"] = [row["accession_number"] for row in u_rows]
    seal["d_accessions"] = [
        row["accession_number"]
        for row in u_rows
        if row["stage_assignment"] == seal["stage"]
    ]
    seal["counts"]["U"] = len(seal["u_accessions"])
    seal["counts"]["D"] = len(seal["d_accessions"])
    exact_count = sum(row["exact_acceptance"] is True for row in u_rows)
    seal["counts"]["exact_acceptance_U"] = exact_count
    seal["exact_acceptance_rate_U"] = exact_count / len(u_rows)
    seal["completed_year_source_coverage"] = (
        validate_completed_year_source_coverage(rows)
    )


@pytest.mark.parametrize(
    ("drift_kind", "locally_valid"),
    [
        pytest.param("submissions_typed_identity", True, id="submissions-typed"),
        pytest.param("submissions_missingness", True, id="submissions-missingness"),
        pytest.param("master_identity_path", False, id="master-identity-path"),
        pytest.param("header_form", True, id="header-form"),
        pytest.param("acceptance", True, id="acceptance"),
        pytest.param("change_date", True, id="change-date"),
        pytest.param("availability", True, id="availability"),
        pytest.param("stage_assignment", True, id="stage-assignment"),
        pytest.param("complete_response", True, id="complete-response"),
        pytest.param("primary_selection", True, id="primary-selection"),
        pytest.param("extracted_text", True, id="extracted-text"),
        pytest.param("normalized_text", True, id="normalized-text"),
    ],
)
def test_cross_stage_rejects_each_frozen_prefix_drift_kind(
    drift_kind: str,
    locally_valid: bool,
) -> None:
    _, _, _, _, _, intermediate, final = _built_full_chain()
    forged = deepcopy(final.seal)
    row = _cross_stage_drift_row(
        forged,
        availability_year="2014" if drift_kind in {"availability", "stage_assignment"} else "2000",
    )
    prefix = row["frozen_prefix"]
    semantic = prefix["submissions_semantic_identity"]
    typed = semantic["typed_values"]
    drift_sha256 = "sha256:" + _canonical_hash(["cross-stage-drift", drift_kind])

    if drift_kind == "submissions_typed_identity":
        typed["items"] = "forged-item"
    elif drift_kind == "submissions_missingness":
        typed["dateOfFilingDateChange"] = ""
        semantic["date_of_filing_date_change_present"] = True
        prefix["filing_date_change"]["submissions_column_present"] = True
    elif drift_kind == "master_identity_path":
        prefix["master_identity"]["filename"] = (
            "edgar/data/320193/forged-complete-submission.txt"
        )
    elif drift_kind == "header_form":
        row["form"] = "10-K"
        typed["form"] = "10-K"
        prefix["master_identity"]["form"] = "10-K"
        prefix["header_identity"]["form"] = "10-K"
        prefix["header_identity"]["raw_form"] = "10-K"
        prefix["primary_selection"]["type"] = "10-K"
        prefix["primary_selection"]["raw_type"] = "10-K"
        row["primary_selection"] = deepcopy(prefix["primary_selection"])
    elif drift_kind == "acceptance":
        changed = typed["acceptanceDateTime"][:-6] + "130000"
        typed["acceptanceDateTime"] = changed
        prefix["header_identity"]["acceptance_datetime"] = changed
        prefix["acceptance"]["submissions_source"] = changed
        prefix["acceptance"]["header_source"] = changed
        prefix["acceptance"]["normalized_et"] = changed
    elif drift_kind == "change_date":
        _set_cross_stage_change_date(
            row,
            row["filing_date"],
            availability_session=row["availability_session"],
            stage_assignment=row["stage_assignment"],
        )
    elif drift_kind == "availability":
        _set_cross_stage_change_date(
            row,
            "2014-05-02",
            availability_session="2014-05-05",
            stage_assignment="development",
        )
    elif drift_kind == "stage_assignment":
        _set_cross_stage_change_date(
            row,
            "2019-01-01",
            availability_session="2019-01-02",
            stage_assignment="intermediate",
        )
    elif drift_kind == "complete_response":
        prefix["complete_response"]["sha256"] = drift_sha256
        row["complete_response"] = deepcopy(prefix["complete_response"])
    elif drift_kind == "primary_selection":
        prefix["primary_selection"]["selected_document_ordinal"] += 1
        row["primary_selection"] = deepcopy(prefix["primary_selection"])
    elif drift_kind == "extracted_text":
        prefix["extracted_text"]["sha256"] = drift_sha256
        row["extracted_text"] = deepcopy(prefix["extracted_text"])
    elif drift_kind == "normalized_text":
        prefix["normalized_text"]["sha256"] = drift_sha256
        row["normalized_text"] = deepcopy(prefix["normalized_text"])
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(f"unknown drift kind: {drift_kind}")

    prefix["submissions_semantic_identity_sha256"] = _canonical_hash(semantic)
    row["frozen_prefix_sha256"] = _canonical_hash(prefix)
    _refresh_cross_stage_forgery(forged)
    _reseal(forged)

    if locally_valid:
        validate_stage_source_seal(forged)
    with pytest.raises(SecGemmaLeanV33SourceError):
        validate_cross_stage_frozen_prefix(intermediate.seal, forged)


def test_authoritative_calendar_is_exact_and_compact_identity_is_sealed() -> None:
    _, _, _, _, development, _, final = _built_full_chain()
    calendar = final.seal["session_calendar"]
    assert calendar == {
        "calendar_id": "nyse_trading_session_dates_2000_01_01_2026_07_10_v2",
        "start": "2000-01-01",
        "end": "2026-07-10",
        "session_count": len(EXPECTED_SESSIONS),
        "sessions_sha256": calendar["sessions_sha256"],
        "sessions_canonical_json_sha256": calendar[
            "sessions_canonical_json_sha256"
        ],
    }
    target = development.master_reconciliation.targets[0]
    source = development.complete_sources[0]
    with pytest.raises(SecGemmaLeanV33SourceError, match="authoritative"):
        reconcile_complete_submission(
            target,
            source.complete_response,
            session_dates=EXPECTED_SESSIONS[:-1],
            acquisition_stage="development",
        )

    forged = deepcopy(final.seal)
    forged["session_calendar"]["sessions_sha256"] = "sha256:" + "0" * 64
    _reseal(forged)
    with pytest.raises(SecGemmaLeanV33SourceError, match="calendar"):
        validate_stage_source_seal(forged)


def test_pre_master_submissions_bounds_are_rejection_only_and_exact() -> None:
    _, _, _, _, development, intermediate, _ = _built_full_chain()
    dev_receipt = build_pre_master_submissions_upper_bound(
        development.submissions_snapshot,
        "development",
    )
    assert dev_receipt["i_upper_bound"] == 128
    assert dev_receipt["p_upper_bound"] == 128
    assert dev_receipt["authority"] == (
        "reject_only_submissions_bound_not_master_membership"
    )
    intermediate_receipt = build_pre_master_submissions_upper_bound(
        intermediate.submissions_snapshot,
        "intermediate",
        prior_bundle=development,
    )
    assert intermediate_receipt["i_upper_bound"] == 152
    assert intermediate_receipt["p_upper_bound"] == 24

    too_many = [
        _row(
            f"{8_000_000_000 + index:010d}-10-{index:06d}",
            "2010-02-01",
        )
        for index in range(1, 130)
    ]
    snapshot = parse_submissions_snapshot(_main_payload(too_many), {})
    with pytest.raises(SecGemmaLeanV33SourceError, match="I upper bound"):
        build_pre_master_submissions_upper_bound(snapshot, "development")


def test_prior_raw_chain_is_replayed_before_subtraction_and_deep_snapshotted() -> None:
    _, _, _, _, development, intermediate, final = _built_full_chain()
    assert intermediate.prior_bundle is development
    assert final.prior_bundle is intermediate
    assert final.prior_seal == intermediate.seal
    assert final.prior_seal is not intermediate.seal
    assert final.prior_seal["target_rows"] is not intermediate.seal["target_rows"]
    assert [entry["stage"] for entry in final.seal["prior_stage_chain"]] == [
        "development",
        "intermediate",
    ]

    first = development.complete_sources[0]
    forged_source = replace(
        first,
        complete_response=first.complete_response + b"forged",
    )
    forged_development = replace(
        development,
        complete_sources=(forged_source, *development.complete_sources[1:]),
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="authenticate|replay"):
        derive_complete_submission_plan(
            intermediate.master_reconciliation,
            prior_bundle=forged_development,
        )

    forged_ancestor_chain = replace(
        intermediate,
        prior_bundle=forged_development,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="authenticate|replay"):
        derive_complete_submission_plan(
            final.master_reconciliation,
            prior_bundle=forged_ancestor_chain,
        )

    forged_intermediate = replace(intermediate, prior_seal=None)
    with pytest.raises(SecGemmaLeanV33SourceError, match="prior bundle"):
        derive_complete_submission_plan(
            final.master_reconciliation,
            prior_bundle=forged_intermediate,
        )


def test_cross_stage_prior_hash_and_exact_p_carry_are_not_forgeable() -> None:
    _, _, _, _, _, intermediate, final = _built_full_chain()
    bad_link = deepcopy(final.seal)
    bad_link["prior_stage_source_seal_sha256"] = "0" * 64
    bad_link["prior_stage_chain"][-1]["stage_source_seal_sha256"] = "0" * 64
    pre = bad_link["pre_master_upper_bound"]
    pre["prior_stage_source_seal_sha256"] = "0" * 64
    pre_body = dict(pre)
    pre_body.pop("pre_master_upper_bound_sha256")
    pre["pre_master_upper_bound_sha256"] = _canonical_hash(pre_body)
    _reseal(bad_link)
    validate_stage_source_seal(bad_link)
    with pytest.raises(SecGemmaLeanV33SourceError, match="hash/full chain"):
        validate_cross_stage_frozen_prefix(intermediate.seal, bad_link)

    forged = deepcopy(final.seal)
    current_order = [row["accession_number"] for row in forged["target_rows"]]
    plan = forged["source_plan"]
    old_carried = plan["carried_accessions"][-1]
    old_new = plan["new_complete_submission_accessions"][0]
    forged_p = (
        set(plan["new_complete_submission_accessions"]) - {old_new}
    ) | {old_carried}
    forged_carried = (
        set(plan["carried_accessions"]) - {old_carried}
    ) | {old_new}
    plan["new_complete_submission_accessions"] = [
        accession for accession in current_order if accession in forged_p
    ]
    plan["carried_accessions"] = [
        accession for accession in current_order if accession in forged_carried
    ]
    for row in forged["target_rows"]:
        if row["accession_number"] == old_carried:
            row["acquisition_stage"] = "final"
        elif row["accession_number"] == old_new:
            row["acquisition_stage"] = "intermediate"
    pre = forged["pre_master_upper_bound"]
    pre["source_document_accessions"] = list(
        plan["new_complete_submission_accessions"]
    )
    pre["source_document_accessions_sha256"] = _canonical_hash(
        tuple(pre["source_document_accessions"])
    )
    pre_body = dict(pre)
    pre_body.pop("pre_master_upper_bound_sha256")
    pre["pre_master_upper_bound_sha256"] = _canonical_hash(pre_body)
    _reseal(forged)
    validate_stage_source_seal(forged)
    with pytest.raises(
        SecGemmaLeanV33SourceError,
        match="authority drifted|P/carry",
    ):
        validate_cross_stage_frozen_prefix(intermediate.seal, forged)


@pytest.mark.parametrize(
    "field",
    [
        "accession_number",
        "filing_date",
        "form",
        "stage_assignment",
        "availability_session",
        "exact_acceptance",
        "complete_response",
        "primary_selection",
        "extracted_text",
        "normalized_text",
    ],
)
def test_every_duplicated_public_row_field_is_cross_bound_to_prefix(field: str) -> None:
    _, _, _, _, _, _, final = _built_full_chain()
    forged = deepcopy(final.seal)
    row = forged["target_rows"][0]
    if field == "accession_number":
        row[field] = "9999999999-99-999999"
    elif field == "filing_date":
        row[field] = "1995-01-02"
    elif field == "form":
        row[field] = "10-K" if row[field] == "10-Q" else "10-Q"
    elif field == "stage_assignment":
        row[field] = (
            "intermediate" if row[field] == "development" else "development"
        )
    elif field == "availability_session":
        row[field] = next(
            session for session in EXPECTED_SESSIONS if session != row[field]
        )
    elif field == "exact_acceptance":
        row[field] = not row[field]
    elif field == "complete_response":
        row[field]["length"] += 1
    elif field == "primary_selection":
        row[field]["document_identity"] += "-forged"
    elif field == "extracted_text":
        row[field]["sha256"] = "sha256:" + "0" * 64
    elif field == "normalized_text":
        row[field]["sha256"] = "sha256:" + "0" * 64
    _reseal(forged)
    with pytest.raises(SecGemmaLeanV33SourceError):
        validate_stage_source_seal(forged)


def test_u_d_and_science_use_availability_order_while_p_uses_filing_order() -> None:
    _, _, _, _, _, _, final = _built_full_chain()
    row_index = {row["accession_number"]: row for row in final.seal["target_rows"]}
    expected_u = sorted(
        final.seal["u_accessions"],
        key=lambda accession: (
            row_index[accession]["availability_session"], accession
        ),
    )
    assert final.seal["u_accessions"] == expected_u
    assert final.seal["d_accessions"] == [
        accession
        for accession in expected_u
        if row_index[accession]["stage_assignment"] == "final"
    ]
    p = final.seal["source_plan"]["new_complete_submission_accessions"]
    assert p == [
        row["accession_number"]
        for row in final.seal["target_rows"]
        if row["accession_number"] in set(p)
    ]
    projection = build_legacy_science_projection(final)
    assert [document.accession_number for document in projection.documents] == (
        final.seal["d_accessions"]
    )

    forged = deepcopy(final.seal)
    forged["u_accessions"] = list(reversed(forged["u_accessions"]))
    _reseal(forged)
    with pytest.raises(SecGemmaLeanV33SourceError, match="U/D"):
        validate_stage_source_seal(forged)


def test_final_completed_year_gate_is_mandatory_and_reconstructed() -> None:
    _, _, _, _, development, _, final = _built_full_chain()
    rows = deepcopy(final.seal["target_rows"])
    remove_index = next(
        index
        for index, row in enumerate(rows)
        if type(row["availability_session"]) is str
        and row["availability_session"].startswith("2000-")
        and row["form"] == "10-K"
    )
    rows.pop(remove_index)
    with pytest.raises(SecGemmaLeanV33SourceError, match="year 2000"):
        validate_completed_year_source_coverage(rows)

    forged = deepcopy(final.seal)
    forged["completed_year_source_coverage"]["years"][0]["10-K"] = 0
    _reseal(forged)
    with pytest.raises(SecGemmaLeanV33SourceError, match="completed-year source gate"):
        validate_stage_source_seal(forged)

    wrong_stage = deepcopy(development.seal)
    wrong_stage["completed_year_source_coverage"] = {"forged": True}
    _reseal(wrong_stage)
    with pytest.raises(SecGemmaLeanV33SourceError, match="only in the final"):
        validate_stage_source_seal(wrong_stage)


def test_legacy_filename_both_and_one_sided_provenance_states() -> None:
    both_row = _row(
        "0000912057-00-023440",
        "2000-05-26",
        primary="legacy-primary.htm",
    )
    both = reconcile_complete_submission(
        _target_for_row(both_row),
        _complete_submission(both_row),
        session_dates=EXPECTED_SESSIONS,
        acquisition_stage="development",
    )
    assert both.seal_row["primary_selection"]["submissions_filename_missing"] is False
    assert both.seal_row["primary_selection"]["sgml_filename_missing"] is False

    neither_row = _row(
        "0000912057-00-023441",
        "2000-05-26",
        primary="",
    )
    neither = reconcile_complete_submission(
        _target_for_row(neither_row),
        _complete_submission(neither_row, include_filename=False),
        session_dates=EXPECTED_SESSIONS,
        acquisition_stage="development",
    )
    selection = neither.seal_row["primary_selection"]
    assert selection["submissions_filename_missing"] is True
    assert selection["sgml_filename_missing"] is True
    assert selection["document_identity"] == "legacy-sequence-1-no-filename"

    sgml_only_response = _complete_submission(neither_row, include_filename=False).replace(
        b"<SEQUENCE>1\n",
        b"<SEQUENCE>1\n<FILENAME>sgml-only.htm\n",
    )
    sgml_only = reconcile_complete_submission(
        _target_for_row(neither_row),
        sgml_only_response,
        session_dates=EXPECTED_SESSIONS,
        acquisition_stage="development",
    )
    selection = sgml_only.seal_row["primary_selection"]
    assert selection["submissions_filename_missing"] is True
    assert selection["sgml_filename_missing"] is False
    assert selection["document_identity"] == "sgml-only.htm"


def test_raw_header_and_all_document_form_sequence_topology_must_be_exact() -> None:
    row = _row("1234567890-24-000099", "2024-02-01")
    target = _target_for_row(row)
    response = _complete_submission(row)
    second_exact_form = (
        b"<DOCUMENT>\n"
        b"<TYPE>10-Q\n"
        b"<SEQUENCE>2\n"
        b"<FILENAME>second.htm\n"
        b"<DESCRIPTION>SECOND DOCUMENT\n"
        b"<TEXT>quarantined second document</TEXT>\n"
        b"</DOCUMENT>\n"
    )
    bad_responses = (
        response.replace(b"<TYPE>10-Q", b"<TYPE>  10-q  ", 1),
        response.replace(
            b"<CONFORMED-SUBMISSION-TYPE>10-Q",
            b"<CONFORMED-SUBMISSION-TYPE>  10-q  ",
            1,
        ),
        response.replace(b"<TYPE>10-Q\n", b"<TYPE>8-K\n", 1),
        response + second_exact_form,
        response.replace(b"<TYPE>10-Q\n", b"<TYPE>8-K\n", 1)
        + second_exact_form,
    )
    for bad_response in bad_responses:
        with pytest.raises(
            SecGemmaLeanV33SourceError,
            match="form|TYPE|sequence-1|SGML",
        ):
            reconcile_complete_submission(
                target,
                bad_response,
                session_dates=EXPECTED_SESSIONS,
                acquisition_stage="final",
            )

    duplicate_fields = (
        response.replace(
            b"<CONFORMED-SUBMISSION-TYPE>10-Q\n",
            b"<CONFORMED-SUBMISSION-TYPE>10-Q\n"
            b"<CONFORMED-SUBMISSION-TYPE>10-Q\n",
            1,
        ),
        response.replace(
            b"<TYPE>10-Q\n",
            b"<TYPE>10-Q\n<TYPE>10-Q\n",
            1,
        ),
        response.replace(
            b"<SEQUENCE>1\n",
            b"<SEQUENCE>1\n<SEQUENCE>1\n",
            1,
        ),
    )
    for duplicate_response in duplicate_fields:
        # The inherited shared parser keeps its original identical-field
        # behavior; singular raw fields are a v3.3 adapter rule only.
        source_module.parse_complete_submission(duplicate_response)
        with pytest.raises(
            SecGemmaLeanV33SourceError,
            match="form|TYPE/SEQUENCE",
        ):
            reconcile_complete_submission(
                target,
                duplicate_response,
                session_dates=EXPECTED_SESSIONS,
                acquisition_stage="final",
            )


def test_gzip_corrupt_truncated_and_both_size_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _master_payload([])
    corrupt = bytearray(payload)
    corrupt[-8] ^= 1
    with pytest.raises(SecGemmaLeanV33SourceError, match="CRC|gzip"):
        parse_strict_master_gzip(bytes(corrupt), year=2024, quarter=1)
    with pytest.raises(SecGemmaLeanV33SourceError, match="truncated|CRC"):
        parse_strict_master_gzip(payload[:-1], year=2024, quarter=1)

    monkeypatch.setattr(
        source_module,
        "MAX_MASTER_COMPRESSED_BYTES",
        len(payload) - 1,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="compressed byte cap"):
        parse_strict_master_gzip(payload, year=2024, quarter=1)
    monkeypatch.setattr(source_module, "MAX_MASTER_COMPRESSED_BYTES", 1024)
    monkeypatch.setattr(source_module, "MAX_MASTER_DECOMPRESSED_BYTES", 8)
    with pytest.raises(SecGemmaLeanV33SourceError, match="decompressed byte cap"):
        parse_strict_master_gzip(payload, year=2024, quarter=1)


def test_master_exact_set_rejects_missing_unexpected_and_forged_evidence() -> None:
    _, _, _, _, development, _, _ = _built_full_chain()
    evidence = list(development.master_reconciliation.quarter_evidence)
    with pytest.raises(SecGemmaLeanV33SourceError, match="set is not exact"):
        reconcile_master_exact_set(
            development.submissions_snapshot,
            evidence[:-1],
            "development",
        )

    occupied_index = next(
        index for index, item in enumerate(evidence) if item.target_rows
    )
    role = evidence[occupied_index].role
    empty = parse_strict_master_gzip(
        _master_payload([]),
        year=role.year,
        quarter=role.quarter,
    )
    missing = list(evidence)
    missing[occupied_index] = empty
    with pytest.raises(SecGemmaLeanV33SourceError, match="set differs"):
        reconcile_master_exact_set(
            development.submissions_snapshot,
            missing,
            "development",
        )

    empty_index = next(
        index for index, item in enumerate(evidence) if not item.target_rows
    )
    empty_role = evidence[empty_index].role
    first, _ = source_module._quarter_bounds(empty_role.year, empty_role.quarter)
    unexpected_row = _row(
        "7777777777-99-777777",
        first.isoformat(),
    )
    unexpected_evidence = parse_strict_master_gzip(
        _master_payload([_master_line(unexpected_row)]),
        year=empty_role.year,
        quarter=empty_role.quarter,
    )
    unexpected = list(evidence)
    unexpected[empty_index] = unexpected_evidence
    with pytest.raises(SecGemmaLeanV33SourceError, match="set differs"):
        reconcile_master_exact_set(
            development.submissions_snapshot,
            unexpected,
            "development",
        )

    forged = list(evidence)
    forged[occupied_index] = replace(
        evidence[occupied_index],
        target_rows=(),
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="authenticate"):
        reconcile_master_exact_set(
            development.submissions_snapshot,
            forged,
            "development",
        )


def _assert_compact_value_has_no_bytes(
    value: object,
    seen: set[int] | None = None,
) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        pytest.fail("compact authority retained raw payload bytes")
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if is_dataclass(value):
        for item in fields(value):
            _assert_compact_value_has_no_bytes(getattr(value, item.name), seen)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_compact_value_has_no_bytes(key, seen)
            _assert_compact_value_has_no_bytes(item, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _assert_compact_value_has_no_bytes(item, seen)


class _TrackingBlobLoader:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self._blobs = blobs
        self.calls: list[str] = []
        self.active = 0
        self.peak = 0

    def __call__(self, blob_name: str) -> bytes:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls.append(blob_name)
        try:
            return self._blobs[blob_name]
        finally:
            self.active -= 1


def _reseal_manifest(
    manifest: dict[str, object],
    **changes: object,
) -> dict[str, object]:
    changed = deepcopy(manifest)
    changed.update(changes)
    changed.pop("role_manifest_sha256", None)
    changed["role_manifest_sha256"] = _canonical_hash(changed)
    return changed


def _reseal_checkpoint_for_manifests(
    checkpoint: dict[str, object],
    manifests: list[dict[str, object]],
    **changes: object,
) -> dict[str, object]:
    changed = deepcopy(checkpoint)
    changed.update(changes)
    changed["role_manifest_count"] = len(manifests)
    changed["role_manifests_sha256"] = _canonical_hash(manifests)
    changed.pop("checkpoint_sha256", None)
    changed["checkpoint_sha256"] = _canonical_hash(changed)
    return changed


def _development_compact_fixture() -> tuple[object, CompactReplayInput, dict[str, bytes]]:
    development, _, _, final_replay, blobs = _built_compact_chain()
    assert final_replay.prior is not None
    assert final_replay.prior.prior is not None
    return development, final_replay.prior.prior, blobs


def test_compact_three_stage_parity_restart_rehydration_and_no_raw_bytes() -> None:
    development, intermediate, final, final_replay, blobs = _built_compact_chain()
    _, _, _, _, full_development, full_intermediate, full_final = _built_full_chain()
    assert development.seal == full_development.seal
    assert intermediate.seal == full_intermediate.seal
    assert final.seal == full_final.seal

    loader = _TrackingBlobLoader(blobs)
    result = rehydrate_compact_stage(
        "final",
        final_replay.checkpoint,
        final_replay.role_manifests,
        loader,
        final_replay.prior,
    )
    assert isinstance(result, CompactReplayResult)
    assert result.stage_output.seal == full_final.seal
    assert result.receipt["exact_source_seal_match"] is True
    assert result.receipt["peak_live_role_payload_count"] == 1
    assert loader.peak == 1
    assert len(loader.calls) == 245 + 161 + 164
    assert detached_replay(
        "development",
        final_replay.prior.prior.checkpoint,
        final_replay.prior.prior.role_manifests,
        blobs.__getitem__,
    )["exact_source_seal_match"] is True
    _assert_compact_value_has_no_bytes(development)
    _assert_compact_value_has_no_bytes(intermediate)
    _assert_compact_value_has_no_bytes(final)
    _assert_compact_value_has_no_bytes(final_replay)
    _assert_compact_value_has_no_bytes(result)


def test_compact_submissions_preserves_mixed_forms_and_non_target_paths() -> None:
    history_name = "CIK0000320193-submissions-001.json"
    target = _row("1234567890-18-000001", "2018-02-01")
    current_non_target = _row(
        "1234567890-18-000002",
        "2018-02-02",
        form="8-K",
        primary="xslF345X03/current-report.xml",
    )
    historical_non_target = _row(
        "1234567890-17-000001",
        "2017-02-01",
        form="4",
        primary="xslF345X02/ownership.xml",
    )
    main_payload = _main_payload(
        [target, current_non_target],
        references=[
            {
                "name": history_name,
                "filingCount": 1,
                "filingFrom": "2017-01-01",
                "filingTo": "2017-12-31",
            }
        ],
    )
    historical_payload = json.dumps(
        _columns([historical_non_target]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    main = parse_main("development", main_payload)
    historical = parse_historical(
        "development",
        history_name,
        historical_payload,
        main,
    )
    phase = finalize_submissions("development", main, [historical])

    assert phase.evidence["submissions_snapshot"]["row_count"] == 3
    assert phase.evidence["submissions_snapshot"]["target_row_count"] == 1
    rows = phase.evidence["rows"]
    assert {row["form"] for row in rows} == {"10-Q", "8-K", "4"}
    assert {
        row["primary_document"] for row in rows if row["form"] != "10-Q"
    } == {
        "xslF345X03/current-report.xml",
        "xslF345X02/ownership.xml",
    }

    tampered = deepcopy(main.evidence)
    tampered["rows"][0]["form"] = "8-K"
    tampered.pop("source_evidence_sha256")
    tampered["source_evidence_sha256"] = _canonical_hash(tampered)
    with pytest.raises(
        SecGemmaLeanV33SourceError,
        match="derived values differ",
    ):
        finalize_submissions("development", tampered, [historical])


def test_compact_replay_rejects_blob_receipt_evidence_and_callback_tampering() -> None:
    _, replay, blobs = _development_compact_fixture()
    manifests = [dict(item) for item in replay.role_manifests]
    first_blob = str(manifests[0]["blob_name"])
    corrupted_blobs = dict(blobs)
    corrupted_blobs[first_blob] = b"!" + corrupted_blobs[first_blob][1:]
    with pytest.raises(SecGemmaLeanV33SourceError, match="hash/length"):
        detached_replay(
            "development",
            replay.checkpoint,
            manifests,
            corrupted_blobs.__getitem__,
        )

    mismatched_evidence = list(manifests)
    mismatched_evidence[0] = _reseal_manifest(
        mismatched_evidence[0],
        source_evidence_sha256="0" * 64,
    )
    evidence_checkpoint = _reseal_checkpoint_for_manifests(
        dict(replay.checkpoint),
        mismatched_evidence,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="evidence differs"):
        detached_replay(
            "development",
            evidence_checkpoint,
            mismatched_evidence,
            blobs.__getitem__,
        )

    tampered_receipt = list(manifests)
    tampered_receipt[0] = _reseal_manifest(
        tampered_receipt[0],
        parse_receipt_sha256="1" * 64,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="hash/bindings"):
        detached_replay(
            "development",
            replay.checkpoint,
            tampered_receipt,
            blobs.__getitem__,
        )

    def failing_loader(_blob_name: str) -> bytes:
        raise OSError("simulated disk failure")

    with pytest.raises(SecGemmaLeanV33SourceError, match="failed safely"):
        detached_replay(
            "development",
            replay.checkpoint,
            manifests,
            failing_loader,
        )


def test_compact_replay_rejects_missing_orphan_duplicate_and_misordered_roles() -> None:
    _, replay, blobs = _development_compact_fixture()
    manifests = [dict(item) for item in replay.role_manifests]

    missing = manifests[:-1]
    missing_checkpoint = _reseal_checkpoint_for_manifests(
        dict(replay.checkpoint),
        missing,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="missing required role"):
        detached_replay(
            "development",
            missing_checkpoint,
            missing,
            blobs.__getitem__,
        )

    orphan = list(manifests)
    orphan.append(
        _reseal_manifest(
            manifests[-1],
            sequence=len(orphan),
            role_id="orphan/not-in-source-plan",
        )
    )
    orphan_checkpoint = _reseal_checkpoint_for_manifests(
        dict(replay.checkpoint),
        orphan,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="orphan"):
        detached_replay(
            "development",
            orphan_checkpoint,
            orphan,
            blobs.__getitem__,
        )

    duplicated = list(manifests)
    duplicated.append(
        _reseal_manifest(
            manifests[-1],
            sequence=len(duplicated),
        )
    )
    duplicate_checkpoint = _reseal_checkpoint_for_manifests(
        dict(replay.checkpoint),
        duplicated,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="duplicated"):
        detached_replay(
            "development",
            duplicate_checkpoint,
            duplicated,
            blobs.__getitem__,
        )

    misordered = list(manifests)
    misordered[1], misordered[2] = misordered[2], misordered[1]
    misordered[1] = _reseal_manifest(misordered[1], sequence=1)
    misordered[2] = _reseal_manifest(misordered[2], sequence=2)
    misordered_checkpoint = _reseal_checkpoint_for_manifests(
        dict(replay.checkpoint),
        misordered,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="misordered|mismatched"):
        detached_replay(
            "development",
            misordered_checkpoint,
            misordered,
            blobs.__getitem__,
        )


def test_compact_replay_rejects_resealed_checkpoint_and_recursive_prior_blob_tamper() -> None:
    _, replay, blobs = _development_compact_fixture()
    manifests = [dict(item) for item in replay.role_manifests]
    false_seal_checkpoint = _reseal_checkpoint_for_manifests(
        dict(replay.checkpoint),
        manifests,
        expected_stage_source_seal_sha256="2" * 64,
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="source seal differs"):
        detached_replay(
            "development",
            false_seal_checkpoint,
            manifests,
            blobs.__getitem__,
        )

    _, _, _, final_replay, all_blobs = _built_compact_chain()
    assert final_replay.prior is not None
    assert final_replay.prior.prior is not None
    prior_main_blob = str(final_replay.prior.prior.role_manifests[0]["blob_name"])
    tampered_prior_blobs = dict(all_blobs)
    tampered_prior_blobs[prior_main_blob] = (
        b"!" + tampered_prior_blobs[prior_main_blob][1:]
    )
    with pytest.raises(SecGemmaLeanV33SourceError, match="hash/length"):
        detached_replay(
            "final",
            final_replay.checkpoint,
            final_replay.role_manifests,
            tampered_prior_blobs.__getitem__,
            final_replay.prior,
        )
