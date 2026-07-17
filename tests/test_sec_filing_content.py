from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json

import pytest

from agent_benchmark.sec_filing_content import (
    MINIMUM_USABLE_TEXT_CHARACTERS,
    CompleteSubmission,
    SGMLDocument,
    audit_filing_content,
    conservative_availability_session,
    normalize_filing_text,
    parse_complete_submission,
    parse_sec_index_json,
    reconcile_filing_content,
    select_primary_document,
    select_sequence_one_primary_document,
)
from agent_benchmark.sec_point_in_time import (
    AAPL_CIK,
    FilingRecord,
    MasterIndexRecord,
    SecPointInTimeError,
    content_sha256,
)


ACCESSION = "0001628280-16-020309"
PRIMARY = "a201610-k9242016.htm"
ACCEPTED = "20161026164216"
FILED = "2016-10-26"
SUBMITTER_CIK = "0001628280"


def _long_html() -> str:
    paragraph = "Apple filing visible financial and governance disclosure. "
    return (
        "<html><head><style>.hidden{display:none} STYLE SECRET</style>"
        "<script>window.secret='SCRIPT SECRET';</script></head><body>"
        "<h1>Annual Report &amp; Results</h1><p>"
        + paragraph * 20
        + "</p></body></html>"
    )


def _submission_payload(
    *,
    accession: str = ACCESSION,
    accepted: str = ACCEPTED,
    form: str = "10-K",
    filed: str = "20161026",
    filer_cik: str = SUBMITTER_CIK,
    subject_cik: str = AAPL_CIK,
    primary: str = PRIMARY,
    primary_text: str | None = None,
) -> str:
    text = _long_html() if primary_text is None else primary_text
    return f"""<SEC-DOCUMENT>{accession}
<SEC-HEADER>
<ACCEPTANCE-DATETIME>{accepted}
ACCESSION NUMBER: {accession}
CONFORMED SUBMISSION TYPE: {form}
FILED AS OF DATE: {filed}
SUBJECT COMPANY:
  COMPANY DATA:
    COMPANY CONFORMED NAME: APPLE INC
    CENTRAL INDEX KEY: {subject_cik}
FILED BY:
  COMPANY DATA:
    COMPANY CONFORMED NAME: TEST FILING AGENT LLC
    CENTRAL INDEX KEY: {filer_cik}
</SEC-HEADER>
<DOCUMENT>
<TYPE>{form}
<SEQUENCE>1
<FILENAME>{primary}
<DESCRIPTION>PRIMARY FILING DOCUMENT
<TEXT>
{text}
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>exhibit99.htm
<DESCRIPTION>EXHIBIT
<TEXT>
<html><body>Short exhibit.</body></html>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def _record(**overrides: object) -> FilingRecord:
    values = {
        "accession_number": ACCESSION,
        "acceptance_datetime": ACCEPTED,
        "form": "10-K",
        "primary_document": PRIMARY,
        "items": "",
        "filing_date": FILED,
        "report_date": "2016-09-24",
        "is_xbrl": True,
        "source_name": f"CIK{AAPL_CIK}.json",
    }
    values.update(overrides)
    return FilingRecord(**values)  # type: ignore[arg-type]


def _master(**overrides: object) -> MasterIndexRecord:
    values = {
        "cik": AAPL_CIK,
        "company_name": "APPLE INC",
        "form": "10-K",
        "filing_date": FILED,
        "filename": f"edgar/data/320193/{ACCESSION}.txt",
        "accession_number": ACCESSION,
    }
    values.update(overrides)
    return MasterIndexRecord(**values)  # type: ignore[arg-type]


def _index_payload(
    *,
    directory_name: str | None = None,
    primary: str = PRIMARY,
) -> dict:
    compact = ACCESSION.replace("-", "")
    return {
        "directory": {
            "name": directory_name
            or f"/Archives/edgar/data/320193/{compact}",
            "parent-dir": "/Archives/edgar/data/320193",
            "item": [
                {"name": primary, "size": "12345"},
                {"name": "exhibit99.htm", "size": "321"},
                {"name": f"{ACCESSION}.txt", "size": "20000"},
            ],
        }
    }


def test_parse_complete_submission_preserves_header_documents_and_hashes() -> None:
    payload = _submission_payload().encode("latin-1")
    parsed = parse_complete_submission(payload)
    assert parsed.header.accession_number == ACCESSION
    assert parsed.header.acceptance_datetime == ACCEPTED
    assert parsed.header.form == "10-K"
    assert parsed.header.filing_date == FILED
    assert parsed.header.subject_cik == AAPL_CIK
    assert parsed.header.subject_company == "APPLE INC"
    assert parsed.header.filer_cik == SUBMITTER_CIK
    assert parsed.header.filer_company == "TEST FILING AGENT LLC"
    assert parsed.submission_sha256 == content_sha256(payload)
    assert len(parsed.documents) == 2
    primary = parsed.documents[0]
    assert primary.document_type == "10-K"
    assert primary.sequence == 1
    assert primary.filename == PRIMARY
    assert primary.description == "PRIMARY FILING DOCUMENT"
    assert "Annual Report" in primary.text
    assert primary.text_sha256 == content_sha256(primary.text.encode("latin-1"))
    assert parsed.documents[1].sequence == 2
    json.dumps(parsed.to_dict(), sort_keys=True, allow_nan=False)


def test_header_filer_fallback_handles_self_filed_submission() -> None:
    payload = _submission_payload().replace(
        f"SUBJECT COMPANY:\n  COMPANY DATA:\n    COMPANY CONFORMED NAME: APPLE INC\n    CENTRAL INDEX KEY: {AAPL_CIK}\nFILED BY:\n  COMPANY DATA:\n    COMPANY CONFORMED NAME: TEST FILING AGENT LLC\n    CENTRAL INDEX KEY: {SUBMITTER_CIK}",
        f"FILER:\n  COMPANY DATA:\n    COMPANY CONFORMED NAME: APPLE INC\n    CENTRAL INDEX KEY: {AAPL_CIK}",
    ).replace(ACCESSION, "0000320193-16-020309")
    parsed = parse_complete_submission(payload)
    assert parsed.header.filer_cik == AAPL_CIK
    assert parsed.header.subject_cik == AAPL_CIK
    assert parsed.header.filer_company == parsed.header.subject_company == "APPLE INC"


def test_iso_submissions_acceptance_reconciles_to_sgml_eastern_instant() -> None:
    result = audit_filing_content(
        _record(acceptance_datetime="2016-10-26T16:42:16Z"),
        _master(),
        _submission_payload(),
        _index_payload(),
        ["2016-10-27"],
    )
    assert result["acceptance_datetime_et"] == ACCEPTED
    assert result["availability_session"] == "2016-10-27"


def test_availability_waits_for_later_filing_date_change() -> None:
    result = audit_filing_content(
        _record(date_of_filing_date_change="2016-10-28"),
        _master(),
        _submission_payload(),
        _index_payload(),
        ["2016-10-27", "2016-10-31"],
    )
    assert result["availability_not_before_date"] == "2016-10-28"
    assert result["availability_session"] == "2016-10-31"


def test_tolerant_mode_retains_identity_and_text_without_one_sgml_timestamp() -> None:
    missing = _submission_payload().replace(
        f"<ACCEPTANCE-DATETIME>{ACCEPTED}",
        "<ACCEPTANCE-DATETIME>UNAVAILABLE",
    )
    with pytest.raises(SecPointInTimeError, match="unambiguous"):
        audit_filing_content(
            _record(), _master(), missing, _index_payload(), ["2016-10-27"]
        )

    result = audit_filing_content(
        _record(),
        _master(),
        missing,
        _index_payload(),
        ["2016-10-27"],
        allow_missing_acceptance=True,
    )
    assert result["identity_reconciled"] is True
    assert result["exact_acceptance_timestamp"] is False
    assert result["acceptance_datetime_et"] is None
    assert result["availability_session"] == "2016-10-27"
    assert result["normalized_text"]["usable"] is True

    suffixed = _submission_payload().replace(
        f"<ACCEPTANCE-DATETIME>{ACCEPTED}",
        f"<ACCEPTANCE-DATETIME>{ACCEPTED}JUNK",
    )
    suffixed_result = audit_filing_content(
        _record(),
        _master(),
        suffixed,
        _index_payload(),
        ["2016-10-27"],
        allow_missing_acceptance=True,
    )
    assert suffixed_result["exact_acceptance_timestamp"] is False


def test_tagged_header_variant_is_parsed() -> None:
    tagged = _submission_payload().replace(
        f"<SEC-DOCUMENT>{ACCESSION}",
        f"<SEC-DOCUMENT>{ACCESSION}.txt : 20161026",
    ).replace(
        "CONFORMED SUBMISSION TYPE: 10-K",
        "<TYPE>10-K",
    ).replace(
        "FILED AS OF DATE: 20161026",
        "<FILING-DATE>20161026",
    ).replace(
        "COMPANY CONFORMED NAME:", "<CONFORMED-NAME>"
    ).replace(
        "CENTRAL INDEX KEY:", "<CIK>"
    )
    tagged_parsed = parse_complete_submission(tagged)
    assert tagged_parsed.header.accession_number == ACCESSION
    assert tagged_parsed.header.form == "10-K"
    assert tagged_parsed.header.filing_date == FILED
    assert tagged_parsed.header.subject_cik == AAPL_CIK


def test_header_preserves_exact_date_as_of_change() -> None:
    payload = _submission_payload().replace(
        "FILED AS OF DATE: 20161026",
        "FILED AS OF DATE: 20161026\nDATE AS OF CHANGE: 20161028",
    )
    parsed = parse_complete_submission(payload)
    assert parsed.header.date_of_filing_date_change == "2016-10-28"

    conflicting = payload.replace(
        "DATE AS OF CHANGE: 20161028",
        "DATE AS OF CHANGE: 20161028\n<DATE-OF-FILING-DATE-CHANGE>20161029",
    )
    with pytest.raises(SecPointInTimeError, match="ambiguous DATE AS OF CHANGE"):
        parse_complete_submission(conflicting)


def test_pre_edgar7_missing_filenames_use_reserved_sequence_identity() -> None:
    accession = "0000912057-00-023442"
    payload = (
        _submission_payload(
            accession=accession,
            accepted="20000511163000",
            form="10-Q",
            filed="20000511",
            filer_cik="0000912057",
            primary="legacy.htm",
        )
        .replace("<FILENAME>legacy.htm\n", "")
        .replace("<FILENAME>exhibit99.htm\n", "")
    ).encode("latin-1")
    parsed = parse_complete_submission(payload)
    assert all(document.filename is None for document in parsed.documents)
    assert [document.document_identity for document in parsed.documents] == [
        "legacy-sequence-1-no-filename",
        "legacy-sequence-2-no-filename",
    ]
    primary = parsed.documents[0]
    assert payload[primary.text_start_byte : primary.text_end_byte] == primary.text.encode(
        "latin-1"
    )
    selection = select_sequence_one_primary_document(
        _record(
            accession_number=accession,
            acceptance_datetime="20000511163000",
            form="10-Q",
            primary_document="",
            filing_date="2000-05-11",
            report_date="2000-03-31",
        ),
        parsed,
    )
    assert selection.sec_filename is None
    assert selection.document_identity == "legacy-sequence-1-no-filename"
    assert selection.submissions_filename_missing is True
    assert selection.sgml_filename_missing is True


def test_legacy_primary_filename_reconciles_when_only_one_source_supplies_it() -> None:
    accession = "0000912057-00-023442"
    base_payload = _submission_payload(
        accession=accession,
        accepted="20000511163000",
        form="10-Q",
        filed="20000511",
        filer_cik="0000912057",
        primary="legacy.htm",
    )
    base_record = _record(
        accession_number=accession,
        acceptance_datetime="20000511163000",
        form="10-Q",
        primary_document="legacy.htm",
        filing_date="2000-05-11",
        report_date="2000-03-31",
    )

    sgml_missing = parse_complete_submission(
        base_payload.replace("<FILENAME>legacy.htm\n", "")
    )
    submissions_only = select_sequence_one_primary_document(
        base_record,
        sgml_missing,
    )
    assert submissions_only.sec_filename == "legacy.htm"
    assert submissions_only.document_identity == "legacy.htm"
    assert submissions_only.submissions_filename_missing is False
    assert submissions_only.sgml_filename_missing is True
    assert (
        submissions_only.document.document_identity
        == "legacy-sequence-1-no-filename"
    )

    submissions_missing = select_sequence_one_primary_document(
        replace(base_record, primary_document=""),
        parse_complete_submission(base_payload),
    )
    assert submissions_missing.sec_filename == "legacy.htm"
    assert submissions_missing.document_identity == "legacy.htm"
    assert submissions_missing.submissions_filename_missing is True
    assert submissions_missing.sgml_filename_missing is False
    assert submissions_missing.document.document_identity == "legacy.htm"


def test_primary_filename_reconciliation_rejects_conflict_and_bad_names() -> None:
    accession = "0000912057-00-023442"
    payload = _submission_payload(
        accession=accession,
        accepted="20000511163000",
        form="10-Q",
        filed="20000511",
        filer_cik="0000912057",
        primary="legacy.htm",
    )
    submission = parse_complete_submission(payload)
    record = _record(
        accession_number=accession,
        acceptance_datetime="20000511163000",
        form="10-Q",
        primary_document="different.htm",
        filing_date="2000-05-11",
        report_date="2000-03-31",
    )
    with pytest.raises(SecPointInTimeError, match="do not reconcile"):
        select_sequence_one_primary_document(record, submission)

    for unsafe_name in (
        "../legacy.htm",
        "folder/legacy.htm",
        r"folder\legacy.htm",
        "legacy.htm:stream",
        "legacy name.htm",
    ):
        with pytest.raises(SecPointInTimeError, match="safe archive filename"):
            select_sequence_one_primary_document(
                replace(record, primary_document=unsafe_name),
                submission,
            )
    with pytest.raises(SecPointInTimeError, match="reserved legacy identity"):
        select_sequence_one_primary_document(
            replace(
                record,
                primary_document="legacy-sequence-1-no-filename",
            ),
            submission,
        )


def test_sgml_filenames_must_be_safe_nonreserved_and_singular() -> None:
    for unsafe_name in (
        "../escape.htm",
        "folder/escape.htm",
        r"folder\escape.htm",
        "escape.htm:stream",
        "escape name.htm",
    ):
        with pytest.raises(SecPointInTimeError, match="safe archive filename"):
            parse_complete_submission(_submission_payload(primary=unsafe_name))

    with pytest.raises(SecPointInTimeError, match="reserved legacy identity"):
        parse_complete_submission(
            _submission_payload(primary="legacy-sequence-1-no-filename")
        )

    duplicate_across_documents = _submission_payload().replace(
        "<FILENAME>exhibit99.htm",
        f"<FILENAME>{PRIMARY}",
    )
    with pytest.raises(SecPointInTimeError, match="unique"):
        parse_complete_submission(duplicate_across_documents)

    duplicate_in_one_document = _submission_payload().replace(
        f"<FILENAME>{PRIMARY}",
        f"<FILENAME>{PRIMARY}\n<FILENAME>{PRIMARY}",
        1,
    )
    with pytest.raises(SecPointInTimeError, match="ambiguous or missing FILENAME"):
        parse_complete_submission(duplicate_in_one_document)


def test_sequence_one_selection_fails_closed() -> None:
    no_sequence_one = parse_complete_submission(
        _submission_payload().replace("<SEQUENCE>1", "<SEQUENCE>3", 1)
    )
    with pytest.raises(SecPointInTimeError, match="sequence-1 document"):
        select_sequence_one_primary_document(_record(), no_sequence_one)

    duplicate_sequence_one = _submission_payload().replace(
        "<SEQUENCE>2",
        "<SEQUENCE>1",
        1,
    )
    with pytest.raises(SecPointInTimeError, match="sequences and filenames must be unique"):
        parse_complete_submission(duplicate_sequence_one)


@pytest.mark.parametrize(
    "payload",
    (
        _submission_payload().replace("<TEXT>", "<BODY>", 1),
        _submission_payload().replace("</TEXT>", "</BODY>", 1),
        _submission_payload().replace(
            "</TEXT>\n</DOCUMENT>",
            "</TEXT>\n<TEXT>second section</TEXT>\n</DOCUMENT>",
            1,
        ),
        _submission_payload(
            primary_text="<html><body><TEXT>unmatched opening</body></html>"
        ),
        _submission_payload(
            primary_text="<html><body>unmatched closing</TEXT></body></html>"
        ),
    ),
)
def test_document_text_boundaries_must_be_exactly_one_balanced_pair(
    payload: str,
) -> None:
    with pytest.raises(SecPointInTimeError, match="unambiguous TEXT"):
        parse_complete_submission(payload)


def test_latin1_offsets_and_response_extracted_normalized_hashes_are_independent() -> None:
    payload = _submission_payload(
        primary_text=_long_html().replace(
            "Annual Report",
            "Annual Caf\u00e9 Report",
        )
    ).encode("latin-1")
    parsed = parse_complete_submission(payload)

    search_from = 0
    for document in sorted(parsed.documents, key=lambda item: item.ordinal):
        opening = payload.index(b"<TEXT>", search_from)
        expected_start = opening + len(b"<TEXT>")
        expected_end = payload.index(b"</TEXT>", expected_start)
        assert document.text_start_byte == expected_start
        assert document.text_end_byte == expected_end
        extracted = payload[expected_start:expected_end]
        assert extracted == document.text.encode("latin-1")
        assert document.text_sha256 == content_sha256(extracted)
        assert document.text_end_byte - document.text_start_byte == len(extracted)
        search_from = expected_end + len(b"</TEXT>")

    primary = parsed.documents[0]
    normalized = normalize_filing_text(primary.text)
    assert parsed.submission_sha256 == content_sha256(payload)
    assert normalized.sha256 == content_sha256(normalized.text.encode("utf-8"))
    assert len(
        {
            parsed.submission_sha256,
            primary.text_sha256,
            normalized.sha256,
        }
    ) == 3

    exhibit_only_change = payload.replace(
        b"Short exhibit.",
        b"Changed exhibit only.",
    )
    changed = parse_complete_submission(exhibit_only_change)
    changed_normalized = normalize_filing_text(changed.documents[0].text)
    assert changed.submission_sha256 != parsed.submission_sha256
    assert changed.documents[0].text_sha256 == primary.text_sha256
    assert changed_normalized.sha256 == normalized.sha256


def test_missing_filename_after_edgar7_boundary_rejects() -> None:
    payload = _submission_payload().replace("<FILENAME>exhibit99.htm\n", "")
    with pytest.raises(SecPointInTimeError, match="missing FILENAME"):
        parse_complete_submission(payload)


def test_complete_submission_fails_closed_on_ambiguity_or_unbalanced_sections() -> None:
    ambiguous_acceptance = _submission_payload().replace(
        f"<ACCEPTANCE-DATETIME>{ACCEPTED}",
        f"<ACCEPTANCE-DATETIME>{ACCEPTED}\n<ACCEPTANCE-DATETIME>20161026164217",
    )
    with pytest.raises(SecPointInTimeError, match="unambiguous"):
        parse_complete_submission(ambiguous_acceptance)

    duplicate_filename = _submission_payload().replace(
        "<FILENAME>exhibit99.htm", f"<FILENAME>{PRIMARY}"
    )
    with pytest.raises(SecPointInTimeError, match="unique"):
        parse_complete_submission(duplicate_filename)

    with pytest.raises(SecPointInTimeError, match="unbalanced"):
        parse_complete_submission(_submission_payload().replace("</DOCUMENT>", "", 1))

    with pytest.raises(SecPointInTimeError, match="SEC-DOCUMENT accession"):
        parse_complete_submission(
            _submission_payload().replace(
                f"<SEC-DOCUMENT>{ACCESSION}",
                "<SEC-DOCUMENT>0000320193-16-000001",
            )
        )

    # Metadata-looking tags inside document HTML do not alter SGML metadata.
    embedded = _submission_payload(
        primary_text="<html><body>\n<TYPE>FAKE\n<p>Visible body</p></body></html>"
    )
    assert parse_complete_submission(embedded).documents[0].document_type == "10-K"


def test_parse_index_json_preserves_sorted_names_sizes_and_rejects_ambiguity() -> None:
    payload = json.dumps(_index_payload()).encode("utf-8")
    parsed = parse_sec_index_json(payload)
    assert parsed.directory_name.endswith(ACCESSION.replace("-", ""))
    assert parsed.parent_directory.endswith("320193")
    assert [item.name for item in parsed.items] == [
        f"{ACCESSION}.txt",
        PRIMARY,
        "exhibit99.htm",
    ]
    assert {item.name: item.size for item in parsed.items}[PRIMARY] == 12345
    assert parsed.payload_sha256 == content_sha256(payload)
    json.dumps(parsed.to_dict(), sort_keys=True, allow_nan=False)

    duplicate = _index_payload()
    duplicate["directory"]["item"].append({"name": PRIMARY, "size": "12345"})
    with pytest.raises(SecPointInTimeError, match="duplicate"):
        parse_sec_index_json(duplicate)
    invalid_size = _index_payload()
    invalid_size["directory"]["item"][0]["size"] = "001"
    with pytest.raises(SecPointInTimeError, match="canonical"):
        parse_sec_index_json(invalid_size)


def test_primary_selection_requires_exact_metadata_sgml_and_index_match() -> None:
    submission = parse_complete_submission(_submission_payload())
    index = parse_sec_index_json(_index_payload())
    document, item = select_primary_document(_record(), submission, index)
    assert document.filename == item.name == PRIMARY
    with pytest.raises(SecPointInTimeError, match="exactly one"):
        select_primary_document(
            _record(primary_document="missing.htm"), submission, index
        )
    missing_index = parse_sec_index_json(
        {
            "directory": {
                **_index_payload()["directory"],
                "item": [{"name": "exhibit99.htm", "size": "321"}],
            }
        }
    )
    with pytest.raises(SecPointInTimeError, match="exactly one"):
        select_primary_document(_record(), submission, missing_index)


def test_normalization_removes_scripts_styles_and_is_deterministic() -> None:
    html = _long_html() + "<p>Unicode：ＡＰＰＬＥ&nbsp;results</p>"
    first = normalize_filing_text(html)
    repeated = normalize_filing_text(html)
    assert first == repeated
    assert "SCRIPT SECRET" not in first.text
    assert "STYLE SECRET" not in first.text
    assert "Annual Report & Results" in first.text
    assert "Unicode:APPLE results" in first.text
    assert "<" not in first.text
    assert first.character_count == len(first.text)
    assert first.character_count >= MINIMUM_USABLE_TEXT_CHARACTERS
    assert first.usable is True
    assert first.sha256 == (
        "sha256:" + hashlib.sha256(first.text.encode("utf-8")).hexdigest()
    )
    short = normalize_filing_text("<p>short filing</p>")
    assert short.usable is False
    assert short.character_count < 500


def test_normalization_bounds_nfkc_utf8_expansion_before_materializing_output() -> None:
    html = "<p>" + ("\u00bc" * 10) + "</p>"
    normalized = normalize_filing_text(html, max_utf8_bytes=64)
    assert len(normalized.text.encode("utf-8")) == 50

    with pytest.raises(SecPointInTimeError, match="UTF-8 byte limit"):
        normalize_filing_text(html, max_utf8_bytes=49)
    with pytest.raises(ValueError, match="positive integer"):
        normalize_filing_text(html, max_utf8_bytes=True)


def test_reconcile_happy_path_is_json_safe_and_uses_next_session_only() -> None:
    submission = parse_complete_submission(_submission_payload())
    index = parse_sec_index_json(_index_payload())
    result = reconcile_filing_content(
        _record(),
        _master(),
        submission,
        index,
        ["2016-10-25", "2016-10-26", "2016-10-27", "2016-10-28"],
    )
    assert result["identity_reconciled"] is True
    assert result["status"] == "usable"
    assert result["accession_number"] == ACCESSION
    assert result["subject_cik"] == AAPL_CIK
    assert result["filer_cik"] == SUBMITTER_CIK
    assert result["acceptance_datetime_et"] == ACCEPTED
    assert result["availability_session"] == "2016-10-27"
    assert result["availability_session_found"] is True
    assert result["primary_document"]["filename"] == PRIMARY
    assert result["primary_document"]["index_size"] == 12345
    assert result["normalized_text"]["usable"] is True
    assert "SCRIPT SECRET" not in result["normalized_text"]["text"]
    json.dumps(result, sort_keys=True, allow_nan=False)

    one_call = audit_filing_content(
        _record(),
        _master(),
        _submission_payload(),
        _index_payload(),
        ["2016-10-26", "2016-10-27"],
    )
    assert one_call == result


def test_third_party_accession_prefix_need_not_equal_header_filer() -> None:
    # Older Apple filings can use a filing-agent accession prefix while the
    # SGML identity block names Apple as the filer/registrant.
    submission = parse_complete_submission(
        _submission_payload(filer_cik=AAPL_CIK)
    )
    result = reconcile_filing_content(
        _record(),
        _master(),
        submission,
        parse_sec_index_json(_index_payload()),
        ["2016-10-27"],
    )
    assert result["accession_submitter_cik"] == SUBMITTER_CIK
    assert result["header_filer_cik"] == AAPL_CIK
    assert result["header_filer_matches_accession_submitter"] is False
    assert result["header_filer_matches_subject"] is True


@pytest.mark.parametrize(
    ("record", "master", "payload", "index_payload", "message"),
    (
        (
            _record(),
            _master(accession_number="0000320193-16-000001"),
            _submission_payload(),
            _index_payload(),
            "accession",
        ),
        (
            _record(),
            _master(cik="0000789019"),
            _submission_payload(),
            _index_payload(),
            "CIK",
        ),
        (
            _record(),
            _master(form="10-Q"),
            _submission_payload(),
            _index_payload(),
            "form",
        ),
        (
            _record(),
            _master(filing_date="2016-10-27"),
            _submission_payload(),
            _index_payload(),
            "filing date",
        ),
        (
            _record(acceptance_datetime="20161026164217"),
            _master(),
            _submission_payload(),
            _index_payload(),
            "acceptance",
        ),
        (
            _record(),
            _master(),
            _submission_payload(),
            _index_payload(directory_name="/Archives/edgar/data/320193/wrong"),
            "index directory",
        ),
    ),
)
def test_reconciliation_fails_closed_on_identity_disagreement(
    record,
    master,
    payload,
    index_payload,
    message,
) -> None:
    with pytest.raises(SecPointInTimeError, match=message):
        reconcile_filing_content(
            record,
            master,
            parse_complete_submission(payload),
            parse_sec_index_json(index_payload),
            ["2016-10-27"],
        )


def test_short_text_and_missing_future_session_are_explicit_not_ambiguous() -> None:
    result = audit_filing_content(
        _record(),
        _master(),
        _submission_payload(primary_text="<p>short filing</p>"),
        _index_payload(),
        ["2016-10-25", "2016-10-26"],
    )
    assert result["status"] == "unusable_text"
    assert result["normalized_text"]["usable"] is False
    assert result["availability_session"] is None
    assert result["availability_session_found"] is False

    wrong_primary_type = _submission_payload().replace(
        "<TYPE>10-K\n<SEQUENCE>1", "<TYPE>EX-99\n<SEQUENCE>1"
    )
    with pytest.raises(SecPointInTimeError, match="Primary DOCUMENT type"):
        audit_filing_content(
            _record(),
            _master(),
            wrong_primary_type,
            _index_payload(),
            ["2016-10-27"],
        )


def test_availability_refuses_duplicate_or_invalid_sessions() -> None:
    acceptance = datetime(2016, 10, 26, 23, 59, 59)
    assert conservative_availability_session(
        acceptance, ["2016-10-27", "2016-10-28"]
    ) == "2016-10-27"
    with pytest.raises(SecPointInTimeError, match="duplicate"):
        conservative_availability_session(
            acceptance, ["2016-10-27", "2016-10-27"]
        )
    with pytest.raises(SecPointInTimeError, match="invalid"):
        conservative_availability_session(acceptance, ["not-a-date"])
