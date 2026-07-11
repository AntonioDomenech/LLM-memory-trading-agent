from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import pytest

from agent_benchmark.sec_audit_artifact import (
    SecAuditArtifactError,
    seal_artifact,
    verify_artifact,
)
from agent_benchmark.sec_audit_plan import MAIN_SUBMISSIONS_NAME, MAIN_SUBMISSIONS_URL, build_sec_audit_plan
from agent_benchmark.sec_audit_runner import (
    CONTRACT_VERSION,
    SOURCE_FILES,
    run_sec_audit,
    validate_local_filesystem_path,
    verify_production_sec_audit_artifact,
    verify_source_provenance,
)
from agent_benchmark.sec_point_in_time import (
    AAPL_CIK,
    FilingRecord,
    SecPointInTimeError,
    content_sha256,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


PRIVATE_CONTACT = "Private Person private-contact@private-domain.dev"


@dataclass(frozen=True)
class _Audit:
    url: str
    status_code: int
    content_type: str
    size_bytes: int
    content_sha256: str
    cache_hit: bool
    user_agent_sha256: str
    network_requests: int
    retries: int
    redirects: int


class _FixtureTransport:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = dict(payloads)
        self.calls: list[str] = []
        self._user_agent_hash = content_sha256(PRIVATE_CONTACT.encode("utf-8"))

    def fetch(self, url: str) -> tuple[bytes, _Audit]:
        self.calls.append(url)
        if url not in self.payloads:
            raise AssertionError(f"unexpected fixture request: {url}")
        body = self.payloads[url]
        content_type = "application/json" if url.endswith(".json") else "text/plain"
        return body, _Audit(
            url=url,
            status_code=200,
            content_type=content_type,
            size_bytes=len(body),
            content_sha256=content_sha256(body),
            cache_hit=False,
            user_agent_sha256=self._user_agent_hash,
            network_requests=1,
            retries=0,
            redirects=0,
        )

    def safe_state(self) -> dict[str, Any]:
        total = sum(len(self.payloads[url]) for url in self.calls)
        return {
            "user_agent": {
                "sha256": self._user_agent_hash,
                "real_contact_validated": True,
            },
            "budget": {
                "requests": len(self.calls),
                "bytes_received": total,
                "max_requests": 100,
                "max_bytes": 250 * 1024 * 1024,
                "max_seconds": 1800.0,
                "elapsed_seconds": 1.0,
            },
        }


def _source_repo(root: Path) -> tuple[Path, str]:
    repo = root / "source-repo"
    repo.mkdir()
    commands = (
        ("init",),
        ("config", "user.email", "tests@local.invalid"),
        ("config", "user.name", "SEC Audit Tests"),
    )
    for command in commands:
        subprocess.run(
            ["git", "-C", str(repo), *command],
            check=True,
            capture_output=True,
        )
    workspace = Path(__file__).resolve().parents[1]
    for relative in SOURCE_FILES:
        source = workspace / relative
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    (repo / "source.txt").write_text("frozen source\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "freeze source"],
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def test_source_provenance_disables_repository_fsmonitor_hook(tmp_path: Path) -> None:
    repo, commit = _source_repo(tmp_path)
    marker = repo / ".git" / "fsmonitor-invoked"
    hook = repo / ".git" / "hooks" / "fsmonitor-test"
    hook.write_text(
        "#!/bin/sh\n"
        "printf invoked > .git/fsmonitor-invoked\n"
        "printf '0\\n'\n",
        encoding="utf-8",
        newline="\n",
    )
    hook.chmod(0o755)
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.fsmonitor", str(hook)],
        check=True,
        capture_output=True,
    )

    provenance = verify_source_provenance(repo)

    assert provenance["commit"] == commit
    assert not marker.exists()


def test_source_provenance_rejects_promisor_repository_without_fetch(
    tmp_path: Path,
) -> None:
    repo, _ = _source_repo(tmp_path)
    subprocess.run(
        ["git", "-C", str(repo), "config", "remote.origin.promisor", "true"],
        check=True,
        capture_output=True,
    )

    with pytest.raises(SecPointInTimeError, match="partial or promisor"):
        verify_source_provenance(repo)


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC boundary")
def test_source_provenance_rejects_unc_path_before_filesystem_access() -> None:
    with pytest.raises(SecPointInTimeError, match="local device"):
        validate_local_filesystem_path(
            r"\\127.0.0.1\nonexistent-share\source-repo"
        )


def test_source_provenance_rejects_git_config_includes_before_git(
    tmp_path: Path,
) -> None:
    repo, _ = _source_repo(tmp_path)
    with (repo / ".git" / "config").open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            "\n[include]\n"
            "\tpath = //127.0.0.1/nonexistent-share/hostile-config\n"
        )

    with pytest.raises(SecPointInTimeError, match="includes are not allowed"):
        verify_source_provenance(repo)


def test_source_provenance_rejects_bom_hidden_git_include_before_git(
    tmp_path: Path,
) -> None:
    repo, _ = _source_repo(tmp_path)
    config = repo / ".git" / "config"
    original = config.read_bytes()
    config.write_bytes(
        b"\xef\xbb\xbf[include]\n"
        b"\tpath = //127.0.0.1/nonexistent-share/hostile-config\n"
        + original
    )

    with pytest.raises(SecPointInTimeError, match="byte-order mark"):
        verify_source_provenance(repo)


def test_source_provenance_rejects_object_alternates_before_git(
    tmp_path: Path,
) -> None:
    repo, _ = _source_repo(tmp_path)
    info = repo / ".git" / "objects" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "alternates").write_text(
        "//127.0.0.1/nonexistent-share/objects\n",
        encoding="utf-8",
    )

    with pytest.raises(SecPointInTimeError, match="redirect configuration or objects"):
        verify_source_provenance(repo)


def _record(
    serial: int,
    year: int,
    month_day: str,
    form: str,
    *,
    time: str = "090000",
    prefix: str = AAPL_CIK,
    is_xbrl: bool = False,
    change_date: str = "",
) -> FilingRecord:
    accession = f"{prefix}-{year % 100:02d}-{serial:06d}"
    return FilingRecord(
        accession_number=accession,
        acceptance_datetime=f"{year}{month_day.replace('-', '')}{time}",
        form=form,
        primary_document=f"document-{serial}.htm",
        items="",
        filing_date=f"{year}-{month_day}",
        report_date="",
        is_xbrl=is_xbrl,
        source_name=MAIN_SUBMISSIONS_NAME,
        subject_cik=AAPL_CIK,
        date_of_filing_date_change=change_date,
    )


def _complete_catalogue() -> list[FilingRecord]:
    records: list[FilingRecord] = []
    serial = 1

    def add(year: int, month_day: str, form: str, **kwargs: Any) -> None:
        nonlocal serial
        records.append(_record(serial, year, month_day, form, **kwargs))
        serial += 1

    for year in range(2000, 2025):
        add(year, "04-15", "10-Q")
    for year in (2000, 2005, 2009, 2014, 2019, 2024):
        add(year, "05-15", "8-K")
        add(year, "06-15", "DEF 14A")
    add(2000, "01-03", "S-8", time="080000")
    add(2008, "01-03", "S-3", is_xbrl=True)
    add(2006, "01-03", "10-K/A")
    add(2007, "01-03", "S-4", time="173100")
    add(2003, "01-03", "SC 13G", prefix="0000000123")
    add(2010, "01-04", "8-A12B", change_date="2010-01-05")
    return records


def _main_payload(records: list[FilingRecord]) -> dict[str, Any]:
    return {
        "cik": "320193",
        "filings": {
            "recent": {
                "accessionNumber": [item.accession_number for item in records],
                "acceptanceDateTime": [item.acceptance_datetime for item in records],
                "form": [item.form for item in records],
                "primaryDocument": [item.primary_document for item in records],
                "items": [item.items for item in records],
                "filingDate": [item.filing_date for item in records],
                "reportDate": [item.report_date for item in records],
                "isXBRL": [int(item.is_xbrl) for item in records],
                "dateOfFilingDateChange": [
                    item.date_of_filing_date_change for item in records
                ],
            },
            "files": [],
        },
    }


def _master_payload(records: list[FilingRecord]) -> bytes:
    rows = [
        "CIK|Company Name|Form Type|Date Filed|Filename",
        "------------------------------------------------------------",
    ]
    for record in sorted(records, key=lambda value: value.accession_number):
        rows.append(
            "|".join(
                (
                    str(int(AAPL_CIK)),
                    "APPLE INC",
                    record.form,
                    record.filing_date,
                    f"edgar/data/320193/{record.accession_number}.txt",
                )
            )
        )
    return ("\n".join(rows) + "\n").encode("ascii")


def _index_payload(record: FilingRecord) -> bytes:
    compact = record.accession_number.replace("-", "")
    value = {
        "directory": {
            "name": f"/Archives/edgar/data/320193/{compact}",
            "parent-dir": "/Archives/edgar/data/320193",
            "item": [
                {
                    "name": record.primary_document,
                    "size": str(len(_primary_payload(record))),
                },
                {
                    "name": f"{record.accession_number}.txt",
                    "size": str(len(_submission_payload(record))),
                },
            ],
        }
    }
    return json.dumps(value, sort_keys=True).encode("utf-8")


def _submission_payload(record: FilingRecord) -> bytes:
    filed = record.filing_date.replace("-", "")
    primary = _primary_payload(record).decode("ascii")
    return f"""<SEC-DOCUMENT>{record.accession_number}
<SEC-HEADER>
<ACCEPTANCE-DATETIME>{record.acceptance_datetime}
ACCESSION NUMBER: {record.accession_number}
CONFORMED SUBMISSION TYPE: {record.form}
FILED AS OF DATE: {filed}
SUBJECT COMPANY:
  COMPANY DATA:
    COMPANY CONFORMED NAME: APPLE INC
    CENTRAL INDEX KEY: {AAPL_CIK}
FILED BY:
  COMPANY DATA:
    COMPANY CONFORMED NAME: TEST SUBMITTER
    CENTRAL INDEX KEY: {record.submitter_cik}
</SEC-HEADER>
<DOCUMENT>
<TYPE>{record.form}
<SEQUENCE>1
<FILENAME>{record.primary_document}
<DESCRIPTION>PRIMARY FILING DOCUMENT
<TEXT>
{primary}
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
""".encode("ascii")


def _primary_payload(record: FilingRecord) -> bytes:
    paragraph = "Apple filing contains visible financial and governance disclosure. " * 25
    return (
        f"<html><body><h1>Apple filing</h1><p>{paragraph}</p></body></html>"
    ).encode("ascii")


def _fixture(
    records: list[FilingRecord] | None = None,
) -> tuple[_FixtureTransport, list[str], dict[str, Any]]:
    records = list(records or _complete_catalogue())
    main = _main_payload(records)
    plan = build_sec_audit_plan(records, main)
    by_accession = {record.accession_number: record for record in records}
    payloads = {
        MAIN_SUBMISSIONS_URL: json.dumps(main, sort_keys=True).encode("utf-8")
    }
    for master in plan["quarterly_master_indexes"]:
        payloads[master["url"]] = _master_payload(
            [by_accession[value] for value in master["accession_numbers"]]
        )
    for request in plan["accession_requests"]:
        record = by_accession[request["accession_number"]]
        payloads[request["index_json_url"]] = _index_payload(record)
        payloads[request["complete_submission_url"]] = _submission_payload(record)
        payloads[request["primary_document_url"]] = _primary_payload(record)
    return _FixtureTransport(payloads), list(EXPECTED_SESSIONS), plan


def test_mocked_end_to_end_audit_passes_and_seals_exact_evidence(tmp_path: Path) -> None:
    transport, sessions, expected_plan = _fixture()
    source_repo, source_commit = _source_repo(tmp_path)
    artifact_dir = tmp_path / "sealed-sec-audit"
    sealed = run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )

    verified = verify_artifact(
        artifact_dir,
        expected_source_commit=source_commit,
        expected_audit_contract_version=CONTRACT_VERSION,
        expected_checksums_sha256=sealed.checksums_sha256,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    evidence = json.loads((artifact_dir / "filing_evidence.json").read_text("utf-8"))
    ledger = json.loads((artifact_dir / "request_ledger.json").read_text("utf-8"))
    roles = json.loads((artifact_dir / "text_role_manifest.json").read_text("utf-8"))

    assert report["offline_pipeline_pass"] is True
    assert report["overall_pass"] is False
    assert report["transport_trust"] == {
        "cache_hit_count": 0,
        "fresh_official_retrieval_gate_passed": True,
        "mode": "untrusted_injected_test_transport",
        "trusted_production_transport": False,
    }
    assert report["runtime_provenance"]["mode"] == "offline_injected"
    assert report["evaluation"]["overall_pass"] is True
    assert report["behavior_evidence"] == {
        "llm_or_model_calls": 0,
        "outcome_guided_substitutions": 0,
        "paid_api_calls": 0,
        "return_calculations": 0,
        "trading_simulations": 0,
    }
    assert len(evidence) == 24
    assert all(item["status"] == "usable" for item in evidence)
    assert len(roles) == 24
    assert sum(not item["training_eligible_for_pre_2019_development"] for item in roles) == 6
    assert all(
        item["evidence_role"] == "structural_only"
        for item in roles
        if not item["training_eligible_for_pre_2019_development"]
    )
    assert len(list(artifact_dir.glob("text_*.txt"))) == 24
    assert len(ledger) == expected_plan["request_count_estimate"]["baseline_requests"]
    assert len(transport.calls) == len(ledger)
    assert PRIVATE_CONTACT not in b"".join(
        path.read_bytes() for path in artifact_dir.iterdir() if path.is_file()
    ).decode("utf-8", errors="ignore")
    assert "audit_report.json" in verified.checksums

    (artifact_dir / "audit_report.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SecAuditArtifactError, match="checksum mismatch"):
        verify_artifact(
            artifact_dir,
            expected_checksums_sha256=sealed.checksums_sha256,
        )


def test_catalogue_gap_is_sealed_without_opening_document_requests(tmp_path: Path) -> None:
    records = [_record(1, 2000, "04-15", "10-Q")]
    main = _main_payload(records)
    transport = _FixtureTransport(
        {MAIN_SUBMISSIONS_URL: json.dumps(main, sort_keys=True).encode("utf-8")}
    )
    artifact_dir = tmp_path / "failed-audit"
    source_repo, _ = _source_repo(tmp_path)

    sealed = run_sec_audit(
        transport=transport,
        aapl_sessions=EXPECTED_SESSIONS,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )

    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    plan = json.loads((artifact_dir / "audit_plan.json").read_text("utf-8"))
    assert report["overall_pass"] is False
    assert report["document_download_opened"] is False
    assert plan["selection"]["gap_count"] == 23
    assert transport.calls == [MAIN_SUBMISSIONS_URL]
    assert not list(artifact_dir.glob("text_*.txt"))
    verify_artifact(
        artifact_dir,
        expected_checksums_sha256=sealed.checksums_sha256,
    )


def test_fixture_payload_hashes_are_stable() -> None:
    transport, _, _ = _fixture()
    digest = hashlib.sha256()
    for url, payload in sorted(transport.payloads.items()):
        digest.update(url.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
    assert len(digest.hexdigest()) == 64


def test_post_cutoff_identity_is_not_sealed_or_requested(tmp_path: Path) -> None:
    records = _complete_catalogue()
    future = _record(999, 2025, "02-03", "10-Q")
    future_change = _record(
        998,
        2024,
        "12-30",
        "S-3",
        change_date="2025-01-02",
    )
    records.extend((future, future_change))
    transport, sessions, _ = _fixture(records)
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / "bounded-audit"

    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )

    catalogue = json.loads((artifact_dir / "catalogue.json").read_text("utf-8"))
    plan = json.loads((artifact_dir / "audit_plan.json").read_text("utf-8"))
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    assert all(item["accession_number"] != future.accession_number for item in catalogue)
    assert all(
        item["accession_number"] != future_change.accession_number
        for item in catalogue
    )
    assert future.accession_number not in json.dumps(plan, sort_keys=True)
    assert future_change.accession_number not in json.dumps(plan, sort_keys=True)
    assert future.accession_number not in json.dumps(report, sort_keys=True)
    assert future.accession_number not in "\n".join(transport.calls)
    assert report["post_2024_artifact_boundary_gate_passed"] is True
    assert report["offline_pipeline_pass"] is True
    assert report["overall_pass"] is False


def test_historical_reference_ranges_cannot_seal_post_cutoff_dates(
    tmp_path: Path,
) -> None:
    transport, sessions, _ = _fixture()
    main = json.loads(transport.payloads[MAIN_SUBMISSIONS_URL].decode("utf-8"))
    name = "CIK0000320193-submissions-001.json"
    url = f"https://data.sec.gov/submissions/{name}"
    main["filings"]["files"] = [
        {
            "name": name,
            "filingFrom": "2025-01-01",
            "filingTo": "2025-01-31",
            "filingCount": 0,
        }
    ]
    transport.payloads[MAIN_SUBMISSIONS_URL] = json.dumps(
        main, sort_keys=True
    ).encode("utf-8")
    transport.payloads[url] = json.dumps(
        {
            "accessionNumber": [],
            "acceptanceDateTime": [],
            "form": [],
            "primaryDocument": [],
            "items": [],
            "filingDate": [],
            "reportDate": [],
            "isXBRL": [],
            "dateOfFilingDateChange": [],
        },
        sort_keys=True,
    ).encode("utf-8")
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / "redacted-reference-range"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    artifact_plan = (artifact_dir / "audit_plan.json").read_text("utf-8")
    assert "2025-01-01" not in artifact_plan
    assert "2025-01-31" not in artifact_plan
    assert '"name": "CIK0000320193-submissions-001.json"' in artifact_plan


def test_historical_reference_count_must_match_downloaded_rows(
    tmp_path: Path,
) -> None:
    transport, sessions, _ = _fixture()
    main = json.loads(transport.payloads[MAIN_SUBMISSIONS_URL].decode("utf-8"))
    name = "CIK0000320193-submissions-001.json"
    url = f"https://data.sec.gov/submissions/{name}"
    main["filings"]["files"] = [{"name": name, "filingCount": 1}]
    transport.payloads[MAIN_SUBMISSIONS_URL] = json.dumps(
        main, sort_keys=True
    ).encode("utf-8")
    transport.payloads[url] = json.dumps(
        {
            "accessionNumber": [],
            "acceptanceDateTime": [],
            "form": [],
            "primaryDocument": [],
            "items": [],
            "filingDate": [],
            "reportDate": [],
        },
        sort_keys=True,
    ).encode("utf-8")
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / "historical-count-failure"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    assert report["overall_pass"] is False
    assert report["status"] == "failed_before_catalogue_plan"


def test_calendar_must_be_the_exact_frozen_nyse_sequence(tmp_path: Path) -> None:
    transport, _, _ = _fixture()
    source_repo, _ = _source_repo(tmp_path)
    with pytest.raises(SecPointInTimeError, match="frozen NYSE"):
        run_sec_audit(
            transport=transport,
            aapl_sessions=[],
            artifact_dir=tmp_path / "bad-calendar",
            source_repo=source_repo,
        )
    assert transport.calls == []


def test_dirty_or_fictitious_source_provenance_is_rejected_before_fetch(
    tmp_path: Path,
) -> None:
    transport, sessions, _ = _fixture()
    source_repo, _ = _source_repo(tmp_path)
    (source_repo / "source.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(SecPointInTimeError, match="must be clean"):
        run_sec_audit(
            transport=transport,
            aapl_sessions=sessions,
            artifact_dir=tmp_path / "dirty-source",
            source_repo=source_repo,
        )
    assert transport.calls == []


def test_transport_state_is_allowlisted_and_budget_mismatch_cannot_pass(
    tmp_path: Path,
) -> None:
    transport, sessions, _ = _fixture()
    original_safe_state = transport.safe_state

    def unsafe_state() -> dict[str, Any]:
        state = original_safe_state()
        state["raw_user_agent"] = PRIVATE_CONTACT
        state["user_agent"]["raw"] = PRIVATE_CONTACT
        state["budget"]["requests"] += 1
        return state

    transport.safe_state = unsafe_state  # type: ignore[method-assign]
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / "resource-failure"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    assert report["resource_and_user_agent_gate_passed"] is False
    assert report["overall_pass"] is False
    artifact_text = b"".join(
        path.read_bytes() for path in artifact_dir.iterdir() if path.is_file()
    ).decode("utf-8", errors="ignore")
    assert PRIVATE_CONTACT not in artifact_text


def test_false_response_hash_is_rejected(tmp_path: Path) -> None:
    transport, sessions, _ = _fixture()
    original_fetch = transport.fetch

    def false_fetch(url: str) -> tuple[bytes, _Audit]:
        body, audit = original_fetch(url)
        return body, replace(audit, content_sha256="sha256:" + "0" * 64)

    transport.fetch = false_fetch  # type: ignore[method-assign]
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / "false-hash"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    assert report["status"] == "failed_before_catalogue_plan"
    assert report["overall_pass"] is False
    assert report["failure_code"] == "response_audit_payload_mismatch"


def test_impossible_historical_reference_count_stops_after_main_fetch(
    tmp_path: Path,
) -> None:
    main = _main_payload([])
    main["filings"]["files"] = [
        {"name": f"CIK0000320193-submissions-{index:03d}.json"}
        for index in range(1000)
    ]
    transport = _FixtureTransport(
        {MAIN_SUBMISSIONS_URL: json.dumps(main, sort_keys=True).encode("utf-8")}
    )
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / "too-many-references"
    run_sec_audit(
        transport=transport,
        aapl_sessions=EXPECTED_SESSIONS,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    assert report["overall_pass"] is False
    assert report["failure_code"] == "historical_reference_request_limit"
    assert transport.calls == [MAIN_SUBMISSIONS_URL]


def test_primary_document_mismatch_is_retained_as_failed_evidence(
    tmp_path: Path,
) -> None:
    transport, sessions, plan = _fixture()
    target = plan["accession_requests"][0]
    transport.payloads[target["primary_document_url"]] += b" changed"
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / "primary-mismatch"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    evidence = json.loads((artifact_dir / "filing_evidence.json").read_text("utf-8"))
    assert report["overall_pass"] is False
    assert sum(item.get("status") == "failed" for item in evidence) == 1


@pytest.mark.parametrize(
    ("missing_count", "expected_pipeline_pass"), ((1, True), (2, False))
)
def test_runner_realizes_frozen_23_of_24_timestamp_fallback(
    tmp_path: Path,
    missing_count: int,
    expected_pipeline_pass: bool,
) -> None:
    transport, sessions, plan = _fixture()
    for request in plan["accession_requests"][:missing_count]:
        url = request["complete_submission_url"]
        payload = transport.payloads[url]
        transport.payloads[url] = re.sub(
            rb"<ACCEPTANCE-DATETIME>[0-9]{14}",
            b"<ACCEPTANCE-DATETIME>UNAVAILABLE",
            payload,
            count=1,
        )
        index_url = request["index_json_url"]
        index_payload = json.loads(transport.payloads[index_url].decode("utf-8"))
        complete_name = f"{request['accession_number']}.txt"
        complete_item = next(
            item
            for item in index_payload["directory"]["item"]
            if item["name"] == complete_name
        )
        complete_item["size"] = str(len(transport.payloads[url]))
        transport.payloads[index_url] = json.dumps(
            index_payload, sort_keys=True
        ).encode("utf-8")
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / f"timestamp-fallback-{missing_count}"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    timestamp_gate = report["evaluation"]["gates"][
        "at_least_23_exact_acceptance_timestamps"
    ]
    assert timestamp_gate["observed_count"] == 24 - missing_count
    assert report["offline_pipeline_pass"] is expected_pipeline_pass
    assert report["overall_pass"] is False


def test_raw_primary_case_change_cannot_hide_behind_normalized_text(
    tmp_path: Path,
) -> None:
    transport, sessions, plan = _fixture()
    target = plan["accession_requests"][0]["primary_document_url"]
    original = transport.payloads[target]
    changed = original.replace(b"<html>", b"<HTML>", 1)
    assert len(changed) == len(original)
    transport.payloads[target] = changed
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / "raw-primary-mismatch"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    evidence = json.loads((artifact_dir / "filing_evidence.json").read_text("utf-8"))
    failures = [item for item in evidence if item.get("status") == "failed"]
    assert report["overall_pass"] is False
    assert failures[0]["failure_code"] == "primary_document_byte_mismatch"


@pytest.mark.parametrize("attack", ("exception", "url", "counters", "type"))
def test_transport_boundary_seals_failure_without_private_contact(
    tmp_path: Path,
    attack: str,
) -> None:
    transport, sessions, _ = _fixture()
    original_fetch = transport.fetch

    def attacked_fetch(url: str):
        if attack == "exception":
            raise SecPointInTimeError(f"private failure {PRIVATE_CONTACT}")
        body, audit = original_fetch(url)
        if attack == "url":
            audit = replace(
                audit,
                url=f"https://www.sec.gov/Archives/{PRIVATE_CONTACT}",
            )
        elif attack == "counters":
            audit = replace(
                audit,
                network_requests=1,
                retries=1000,
                redirects=1000,
            )
        elif attack == "type":
            return "not bytes", audit
        return body, audit

    transport.fetch = attacked_fetch  # type: ignore[method-assign]
    source_repo, _ = _source_repo(tmp_path)
    artifact_dir = tmp_path / f"private-boundary-{attack}"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
    )
    report = json.loads((artifact_dir / "audit_report.json").read_text("utf-8"))
    artifact_text = b"".join(
        path.read_bytes() for path in artifact_dir.iterdir() if path.is_file()
    ).decode("utf-8", errors="ignore")
    assert report["overall_pass"] is False
    assert PRIVATE_CONTACT not in artifact_text


def test_provenance_rejects_same_paths_with_unrelated_contents(tmp_path: Path) -> None:
    repo = tmp_path / "fake-source"
    repo.mkdir()
    for command in (
        ("init",),
        ("config", "user.email", "tests@local.invalid"),
        ("config", "user.name", "SEC Audit Tests"),
    ):
        subprocess.run(
            ["git", "-C", str(repo), *command],
            check=True,
            capture_output=True,
        )
    for relative in SOURCE_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not the loaded implementation\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "fake source"],
        check=True,
        capture_output=True,
    )
    transport, sessions, _ = _fixture()
    with pytest.raises(SecPointInTimeError, match="does not match the loaded"):
        run_sec_audit(
            transport=transport,
            aapl_sessions=sessions,
            artifact_dir=tmp_path / "fake-source-artifact",
            source_repo=repo,
        )
    assert transport.calls == []


def test_production_semantic_verifier_requires_complete_reconciled_artifact(
    tmp_path: Path,
) -> None:
    transport, sessions, _ = _fixture()
    source_repo, source_commit = _source_repo(tmp_path)
    offline = tmp_path / "offline-complete"
    run_sec_audit(
        transport=transport,
        aapl_sessions=sessions,
        artifact_dir=offline,
        source_repo=source_repo,
    )
    payloads = {
        path.name: path.read_bytes()
        for path in offline.iterdir()
        if path.name not in {"artifact_metadata.json", "checksums.json"}
    }
    report = json.loads(payloads["audit_report.json"].decode("utf-8"))
    report["overall_pass"] = True
    report["transport_trust"].update(
        {
            "trusted_production_transport": True,
            "mode": "bounded_official_sec_transport",
            "cache_hit_count": 0,
            "fresh_official_retrieval_gate_passed": True,
        }
    )
    report["runtime_provenance"]["mode"] = "live_official_sec"
    payloads["audit_report.json"] = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    production = tmp_path / "production-complete"
    sealed = seal_artifact(
        production,
        payloads,
        source_commit=source_commit,
        audit_contract_version=CONTRACT_VERSION,
    )
    verified, verified_report = verify_production_sec_audit_artifact(
        production,
        expected_checksums_sha256=sealed.checksums_sha256,
        expected_source_commit=source_commit,
    )
    assert verified == sealed
    assert verified_report["overall_pass"] is True

    roles = json.loads(payloads["text_role_manifest.json"].decode("utf-8"))
    payloads["text_role_manifest.json"] = (
        json.dumps(roles[:-1], indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    incomplete = tmp_path / "production-incomplete"
    incomplete_seal = seal_artifact(
        incomplete,
        payloads,
        source_commit=source_commit,
        audit_contract_version=CONTRACT_VERSION,
    )
    with pytest.raises(SecPointInTimeError, match="text-role manifest"):
        verify_production_sec_audit_artifact(
            incomplete,
            expected_checksums_sha256=incomplete_seal.checksums_sha256,
            expected_source_commit=source_commit,
        )
