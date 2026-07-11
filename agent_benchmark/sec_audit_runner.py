"""Bounded, injected-transport runner for the frozen Apple SEC audit.

This module joins the already-frozen catalogue, selection, content, evaluation,
and artifact primitives.  It deliberately has no concrete HTTP client or CLI:
callers must inject a transport, which keeps tests offline and prevents an
unconfigured run from contacting the SEC.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from .sec_audit_artifact import ArtifactVerification, seal_artifact
from .sec_audit_evaluation import (
    FilingAuditResult,
    NormalizedFilingIdentity,
    NormalizedIndexIdentity,
    PeriodicCoverageResult,
    evaluate_sec_audit,
)
from .sec_audit_plan import MAIN_SUBMISSIONS_URL, build_sec_audit_plan
from .sec_audit_transport import (
    MAX_REDIRECTS,
    MAX_RETRIES,
    SecAuditTransport,
    canonical_sec_url,
)
from .sec_filing_content import audit_filing_content, normalize_filing_text
from .sec_point_in_time import (
    BudgetCounter,
    FilingRecord,
    MAX_AUDIT_BYTES,
    MAX_AUDIT_REQUESTS,
    MAX_AUDIT_SECONDS,
    MasterIndexRecord,
    SecPointInTimeError,
    content_sha256,
    parse_master_idx,
    parse_submissions_rows,
)
from .sec_session_calendar import validate_aapl_session_calendar


CONTRACT_VERSION = "aapl-sec-point-in-time-audit-runner-v1"
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_FILES = (
    "agent_benchmark/sec_point_in_time.py",
    "agent_benchmark/sec_audit_selection.py",
    "agent_benchmark/sec_audit_transport.py",
    "agent_benchmark/sec_filing_content.py",
    "agent_benchmark/sec_audit_plan.py",
    "agent_benchmark/sec_audit_evaluation.py",
    "agent_benchmark/sec_audit_artifact.py",
    "agent_benchmark/sec_session_calendar.py",
    "agent_benchmark/sec_audit_runner.py",
    "docs/aapl_point_in_time_text_data_audit_v1.md",
)
_AUDIT_FIELDS = frozenset(
    {
        "url",
        "status_code",
        "content_type",
        "size_bytes",
        "content_sha256",
        "cache_hit",
        "user_agent_sha256",
        "network_requests",
        "retries",
        "redirects",
    }
)


class TransportLike(Protocol):
    """The privacy-safe subset supplied by :class:`SecAuditTransport`."""

    def fetch(self, url: str) -> tuple[bytes, Any]: ...

    def safe_state(self) -> Mapping[str, Any]: ...


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SecPointInTimeError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SecPointInTimeError(f"{label} must be a JSON object")
    return value


def _record_dict(record: FilingRecord) -> dict[str, Any]:
    return {
        **asdict(record),
        "submitter_cik": record.submitter_cik,
        "is_amendment": record.is_amendment,
        "change_anomaly_flag": record.change_anomaly_flag,
    }


def _identity_from_record(record: FilingRecord) -> NormalizedFilingIdentity:
    return NormalizedFilingIdentity(
        accession=record.accession_number,
        form=record.form.strip().upper(),
        filing_date=record.filing_date,
        subject_cik=record.subject_cik,
    )


def _identity_from_master(record: MasterIndexRecord) -> NormalizedFilingIdentity:
    return NormalizedFilingIdentity(
        accession=record.accession_number,
        form=record.form.strip().upper(),
        filing_date=record.filing_date,
        subject_cik=record.cik,
    )


def verify_source_provenance(
    source_repo: str | Path,
) -> dict[str, Any]:
    """Bind a run to the clean, existing HEAD of the supplied Git repository."""

    repo = Path(source_repo).resolve()
    if not repo.is_dir():
        raise SecPointInTimeError("source repository directory does not exist")

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), *arguments],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise SecPointInTimeError("could not verify source Git provenance") from exc
        return completed.stdout.strip()

    def git_bytes(*arguments: str) -> bytes:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), *arguments],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SecPointInTimeError("could not verify source Git provenance") from exc
        return completed.stdout

    def normalized_source(payload: bytes) -> bytes:
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise SecPointInTimeError("SEC audit source file is not UTF-8") from exc
        return text.replace("\r\n", "\n").encode("utf-8")

    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    if root != repo:
        raise SecPointInTimeError("source_repo must be the Git worktree root")
    commit = git("rev-parse", "--verify", "HEAD")
    tree = git("rev-parse", "--verify", "HEAD^{tree}")
    if not _COMMIT_RE.fullmatch(commit) or not _COMMIT_RE.fullmatch(tree):
        raise SecPointInTimeError("source Git identity is not full SHA-1")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise SecPointInTimeError("source repository must be clean before audit")
    source_blobs: dict[str, str] = {}
    loaded_root = Path(__file__).resolve().parents[1]
    for name in SOURCE_FILES:
        blob = git("rev-parse", "--verify", f"HEAD:{name}")
        if not _COMMIT_RE.fullmatch(blob):
            raise SecPointInTimeError(
                "source commit does not contain the complete SEC audit implementation"
            )
        loaded_path = loaded_root / name
        if not loaded_path.is_file() or normalized_source(
            loaded_path.read_bytes()
        ) != normalized_source(git_bytes("show", f"HEAD:{name}")):
            raise SecPointInTimeError(
                "source commit does not match the loaded SEC audit implementation"
            )
        source_blobs[name] = blob
    return {
        "commit": commit,
        "tree": tree,
        "clean_worktree": True,
        "head_verified": True,
        "required_source_blobs": source_blobs,
    }


def _safe_audit_dict(
    audit: Any,
    *,
    requested_url: str,
    payload: bytes,
) -> dict[str, Any]:
    if hasattr(audit, "__dataclass_fields__"):
        value = asdict(audit)
    elif isinstance(audit, Mapping):
        value = dict(audit)
    else:
        raise SecPointInTimeError("transport audit must be a dataclass or mapping")
    if set(value) != _AUDIT_FIELDS:
        raise SecPointInTimeError("transport response audit schema is not exact")
    requested = canonical_sec_url(requested_url)
    final_url = canonical_sec_url(value["url"])
    if not isinstance(payload, bytes):
        raise TypeError("transport payload must be bytes")
    if (
        not isinstance(value["status_code"], int)
        or isinstance(value["status_code"], bool)
        or not 200 <= value["status_code"] < 300
        or value["size_bytes"] != len(payload)
        or value["content_sha256"] != content_sha256(payload)
        or not _TAGGED_SHA256_RE.fullmatch(str(value["user_agent_sha256"]))
        or not isinstance(value["cache_hit"], bool)
    ):
        raise SecPointInTimeError("transport response audit does not match payload")
    content_type = value["content_type"]
    if (
        not isinstance(content_type, str)
        or not content_type
        or len(content_type) > 160
        or "@" in content_type
        or any(ord(character) < 32 or ord(character) > 126 for character in content_type)
        or "/" not in content_type.split(";", 1)[0]
        or urlsplit(final_url).query
        or "@" in final_url
        or "%40" in final_url.lower()
        or any(character.isspace() for character in final_url)
    ):
        raise SecPointInTimeError("transport response metadata is unsafe")
    for name in ("network_requests", "retries", "redirects"):
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise SecPointInTimeError("transport response counters are invalid")
    if (value["cache_hit"] and value["network_requests"] != 0) or (
        not value["cache_hit"] and value["network_requests"] < 1
    ):
        raise SecPointInTimeError("transport cache/network accounting is inconsistent")
    if (
        value["retries"] > MAX_RETRIES
        or value["redirects"] > MAX_REDIRECTS
        or (value["redirects"] == 0 and final_url != requested)
        or (
            not value["cache_hit"]
            and value["network_requests"]
            != 1 + value["retries"] + value["redirects"]
        )
    ):
        raise SecPointInTimeError("transport retry/redirect accounting is inconsistent")
    result = {"requested_url": requested, **value, "url": final_url}
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def _safe_transport_state(
    transport: TransportLike,
    request_ledger: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    raw = transport.safe_state()
    if not isinstance(raw, Mapping):
        raise SecPointInTimeError("transport safe_state must be a mapping")
    user_agent = raw.get("user_agent")
    budget = raw.get("budget")
    if not isinstance(user_agent, Mapping) or not isinstance(budget, Mapping):
        raise SecPointInTimeError("transport state lacks user-agent or budget evidence")
    ua_hash = user_agent.get("sha256")
    ua_valid = user_agent.get("real_contact_validated")
    if not isinstance(ua_hash, str) or not _TAGGED_SHA256_RE.fullmatch(ua_hash):
        raise SecPointInTimeError("transport state has an invalid User-Agent hash")
    if ua_valid is not True:
        raise SecPointInTimeError("transport state lacks validated real contact evidence")

    integer_names = ("requests", "bytes_received", "max_requests", "max_bytes")
    values: dict[str, int | float] = {}
    for name in integer_names:
        value = budget.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SecPointInTimeError(f"transport budget {name} is invalid")
        values[name] = value
    for name in ("max_seconds", "elapsed_seconds"):
        value = budget.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise SecPointInTimeError(f"transport budget {name} is invalid")
        values[name] = float(value)
    ledger_requests = sum(int(item["network_requests"]) for item in request_ledger)
    ledger_bytes = sum(int(item["size_bytes"]) for item in request_ledger)
    passed = bool(
        values["requests"] == ledger_requests
        and values["bytes_received"] == ledger_bytes
        and all(item["user_agent_sha256"] == ua_hash for item in request_ledger)
        and values["requests"] <= values["max_requests"] <= MAX_AUDIT_REQUESTS
        and values["bytes_received"] <= values["max_bytes"] <= MAX_AUDIT_BYTES
        and values["elapsed_seconds"] <= values["max_seconds"] <= MAX_AUDIT_SECONDS
    )
    sanitized = {
        "user_agent": {
            "sha256": ua_hash,
            "real_contact_validated": True,
        },
        "budget": values,
        "ledger_reconciled": passed,
    }
    return sanitized, passed


def _sanitized_transport_state(
    transport: TransportLike,
    request_ledger: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bool]:
    try:
        return _safe_transport_state(transport, request_ledger)
    except (SecPointInTimeError, TypeError, ValueError):
        return (
            {
                "user_agent": {
                    "sha256": None,
                    "real_contact_validated": False,
                },
                "budget": None,
                "ledger_reconciled": False,
                "state_valid": False,
            },
            False,
        )


def _safe_error_evidence(error: BaseException) -> dict[str, str]:
    message = str(error)
    lowered = message.lower()
    if "historical submissions references exceed" in lowered:
        code = "historical_reference_request_limit"
    elif "response audit does not match payload" in lowered:
        code = "response_audit_payload_mismatch"
    elif "response metadata is unsafe" in lowered:
        code = "unsafe_response_metadata"
    elif "complete-submission size" in lowered:
        code = "complete_submission_size_mismatch"
    elif "primary-document size" in lowered:
        code = "primary_document_size_mismatch"
    elif "primary-document bytes" in lowered:
        code = "primary_document_byte_mismatch"
    elif "primary document does not match" in lowered:
        code = "primary_document_text_mismatch"
    elif "next aapl session" in lowered:
        code = "availability_session_missing"
    elif "master.idx" in lowered:
        code = "master_index_reconciliation_failure"
    else:
        code = "sec_audit_evidence_rejected"
    if isinstance(error, SecPointInTimeError):
        failure_class = "sec_point_in_time_error"
    elif isinstance(error, TypeError):
        failure_class = "type_error"
    elif isinstance(error, ValueError):
        failure_class = "value_error"
    else:
        failure_class = "audit_error"
    return {
        "failure_class": failure_class,
        "failure_code": code,
        "failure_message_sha256": content_sha256(message.encode("utf-8")),
    }


def _seal_initial_failure(
    *,
    artifact_dir: str | Path,
    source_commit: str,
    provenance: Mapping[str, Any],
    calendar_evidence: Mapping[str, Any],
    transport: TransportLike,
    request_ledger: Sequence[Mapping[str, Any]],
    error: BaseException,
) -> ArtifactVerification:
    transport_state, resource_gate = _sanitized_transport_state(
        transport, request_ledger
    )
    failure = _safe_error_evidence(error)
    report = {
        "contract_version": CONTRACT_VERSION,
        "source_commit": source_commit,
        "source_provenance": dict(provenance),
        "status": "failed_before_catalogue_plan",
        **failure,
        "resource_and_user_agent_gate_passed": resource_gate,
        "overall_pass": False,
        "behavior_evidence": {
            "llm_or_model_calls": 0,
            "return_calculations": 0,
            "trading_simulations": 0,
            "outcome_guided_substitutions": 0,
            "paid_api_calls": 0,
        },
    }
    return seal_artifact(
        artifact_dir,
        {
            "audit_report.json": _json_bytes(report),
            "request_ledger.json": _json_bytes(list(request_ledger)),
            "session_calendar.json": _json_bytes(dict(calendar_evidence)),
            "transport_state.json": _json_bytes(transport_state),
        },
        source_commit=source_commit,
        audit_contract_version=CONTRACT_VERSION,
    )


def _catalogue_from_transport(
    transport: TransportLike,
    request_ledger: list[dict[str, Any]],
) -> tuple[dict[str, Any], tuple[FilingRecord, ...]]:
    main_body, main_audit = transport.fetch(MAIN_SUBMISSIONS_URL)
    request_ledger.append(
        _safe_audit_dict(
            main_audit,
            requested_url=MAIN_SUBMISSIONS_URL,
            payload=main_body,
        )
    )
    main_payload = _json_object(main_body, label="main submissions payload")

    # The empty planner pass performs the frozen validation of historical
    # reference names and URLs before any referenced file is requested.
    bootstrap = build_sec_audit_plan([], main_payload)
    if not bootstrap["request_count_estimate"]["within_frozen_limit"]:
        raise SecPointInTimeError(
            "historical submissions references exceed the frozen request limit"
        )
    historical_payloads: dict[str, dict[str, Any]] = {}
    for reference in bootstrap["historical_submissions"]:
        body, audit = transport.fetch(reference["url"])
        request_ledger.append(
            _safe_audit_dict(
                audit,
                requested_url=reference["url"],
                payload=body,
            )
        )
        historical = _json_object(
            body, label=f"historical submissions payload {reference['name']}"
        )
        accessions = historical.get("accessionNumber")
        filing_dates = historical.get("filingDate")
        if not isinstance(accessions, list) or not isinstance(filing_dates, list):
            raise SecPointInTimeError(
                "historical submissions payload lacks row arrays"
            )
        if (
            "filing_count" in reference
            and len(accessions) != reference["filing_count"]
        ):
            raise SecPointInTimeError(
                "historical submissions row count does not match reference"
            )
        if len(filing_dates) != len(accessions):
            raise SecPointInTimeError(
                "historical submissions filing dates are not row-aligned"
            )
        if "filing_from" in reference and any(
            not isinstance(value, str) or value < reference["filing_from"]
            for value in filing_dates
        ):
            raise SecPointInTimeError(
                "historical submissions row precedes referenced range"
            )
        if "filing_to" in reference and any(
            not isinstance(value, str) or value > reference["filing_to"]
            for value in filing_dates
        ):
            raise SecPointInTimeError(
                "historical submissions row exceeds referenced range"
            )
        historical_payloads[reference["name"]] = historical
    filings = parse_submissions_rows(main_payload, historical_payloads)
    return main_payload, filings


def _exact_master_record(
    records: Sequence[MasterIndexRecord],
    accession: str,
) -> MasterIndexRecord:
    matches = [record for record in records if record.accession_number == accession]
    if len(matches) != 1:
        raise SecPointInTimeError(
            f"master.idx must contain exactly one row for accession {accession}"
        )
    return matches[0]


def _periodic_results(plan: Mapping[str, Any]) -> list[PeriodicCoverageResult]:
    return [
        PeriodicCoverageResult(
            year=int(item["year"]),
            accessions=tuple(item["accession_numbers"]),
        )
        for item in plan["periodic_report_coverage"]["years"]
    ]


def _unopened_results(
    plan: Mapping[str, Any],
    by_accession: Mapping[str, FilingRecord],
) -> list[FilingAuditResult]:
    results: list[FilingAuditResult] = []
    for slot in [*plan["selection"]["core"], *plan["selection"]["edge_cases"]]:
        record = (
            by_accession.get(slot["filing"]["accession"])
            if slot["status"] == "selected"
            else None
        )
        identity = _identity_from_record(record) if record is not None else None
        results.append(
            FilingAuditResult(
                slot_id=slot["slot_id"],
                status=slot["status"],
                evidence_role=slot["evidence_role"],
                selected=identity,
                submissions=identity,
                master_idx=None,
                sgml_header=None,
                index_metadata=None,
                exact_acceptance_timestamp=False,
                usable_visible_text=False,
                missing_reason=slot["gap_reason"] or "document_audit_not_opened",
            )
        )
    return results


def _artifact_safe_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve exclusion reasons without sealing post-cutoff identities."""

    result = copy.deepcopy(dict(plan))
    reason_counts: dict[str, int] = {}
    for exclusion in result.get("exclusions", []):
        for reason in exclusion.get("reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    result["exclusions"] = [
        {"reason": reason, "count": count}
        for reason, count in sorted(reason_counts.items())
    ]
    for reference in result.get("historical_submissions", []):
        reference.pop("filing_from", None)
        reference.pop("filing_to", None)
    result["post_cutoff_identity_policy"] = (
        "counts_and_reasons_only_no_post_2024_accession_or_date_is_sealed"
    )
    return result


def _run_sec_audit(
    *,
    transport: TransportLike,
    aapl_sessions: Sequence[str],
    artifact_dir: str | Path,
    source_repo: str | Path,
    trusted_production_transport: bool,
) -> ArtifactVerification:
    """Run and seal the frozen audit through an explicitly injected transport.

    A catalogue that misses any predeclared slot or annual coverage gate is
    sealed as a failed audit without requesting filing documents.  Individual
    document failures are retained as evidence and cannot be substituted.
    """

    calendar_evidence = validate_aapl_session_calendar(aapl_sessions)
    validated_sessions = calendar_evidence["sessions"]
    provenance = verify_source_provenance(source_repo)
    source_commit = provenance["commit"]
    request_ledger: list[dict[str, Any]] = []
    try:
        main_payload, filings = _catalogue_from_transport(
            transport, request_ledger
        )
        plan = build_sec_audit_plan(filings, main_payload)
    except (SecPointInTimeError, TypeError, ValueError) as exc:
        return _seal_initial_failure(
            artifact_dir=artifact_dir,
            source_commit=source_commit,
            provenance=provenance,
            calendar_evidence=calendar_evidence,
            transport=transport,
            request_ledger=request_ledger,
            error=exc,
        )
    by_accession = {record.accession_number: record for record in filings}
    filing_results = _unopened_results(plan, by_accession)
    filing_evidence: list[dict[str, Any]] = []
    normalized_text_files: dict[str, bytes] = {}
    text_role_entries: list[dict[str, Any]] = []

    if plan["ready_for_bounded_download"]:
        master_by_url: dict[str, tuple[MasterIndexRecord, ...]] = {}
        master_errors: dict[str, dict[str, str]] = {}
        for request in plan["quarterly_master_indexes"]:
            url = request["url"]
            try:
                body, audit = transport.fetch(url)
                request_ledger.append(
                    _safe_audit_dict(audit, requested_url=url, payload=body)
                )
                master_by_url[url] = parse_master_idx(body)
            except (SecPointInTimeError, TypeError, ValueError) as exc:
                master_errors[url] = _safe_error_evidence(exc)

        opened: list[FilingAuditResult] = []
        for request in plan["accession_requests"]:
            accession = request["accession_number"]
            record = by_accession[accession]
            selected_identity = _identity_from_record(record)
            master_identity: NormalizedFilingIdentity | None = None
            try:
                master_url = request["master_idx_url"]
                if master_url in master_errors:
                    raise SecPointInTimeError(
                        master_errors[master_url]["failure_code"]
                    )
                master = _exact_master_record(master_by_url.get(master_url, ()), accession)
                master_identity = _identity_from_master(master)
                index_body, index_audit = transport.fetch(request["index_json_url"])
                request_ledger.append(
                    _safe_audit_dict(
                        index_audit,
                        requested_url=request["index_json_url"],
                        payload=index_body,
                    )
                )
                submission_body, submission_audit = transport.fetch(
                    request["complete_submission_url"]
                )
                request_ledger.append(
                    _safe_audit_dict(
                        submission_audit,
                        requested_url=request["complete_submission_url"],
                        payload=submission_body,
                    )
                )
                primary_body, primary_audit = transport.fetch(
                    request["primary_document_url"]
                )
                primary_audit_entry = _safe_audit_dict(
                    primary_audit,
                    requested_url=request["primary_document_url"],
                    payload=primary_body,
                )
                request_ledger.append(primary_audit_entry)
                evidence = audit_filing_content(
                    record,
                    master,
                    submission_body,
                    index_body,
                    validated_sessions,
                    allow_missing_acceptance=True,
                )
                normalized = evidence["normalized_text"]
                if evidence["complete_submission_index_size"] != len(
                    submission_body
                ):
                    raise SecPointInTimeError(
                        "complete-submission size does not match index.json"
                    )
                if not evidence["availability_session_found"]:
                    raise SecPointInTimeError(
                        "selected filing has no conservative next AAPL session"
                    )
                if evidence["primary_document"]["index_size"] != len(primary_body):
                    raise SecPointInTimeError(
                        "standalone primary-document size does not match index.json"
                    )
                comparable_primary = primary_body.strip(b"\r\n")
                if (
                    evidence["primary_document"][
                        "embedded_document_bytes_sha256"
                    ]
                    != content_sha256(comparable_primary)
                    or evidence["primary_document"][
                        "embedded_document_bytes_length"
                    ]
                    != len(comparable_primary)
                ):
                    raise SecPointInTimeError(
                        "standalone primary-document bytes do not match complete submission"
                    )
                standalone_normalized = normalize_filing_text(
                    primary_body.decode("latin-1")
                )
                if standalone_normalized.sha256 != normalized["sha256"]:
                    raise SecPointInTimeError(
                        "standalone primary document does not match complete submission"
                    )
                text_name = f"text_{accession.replace('-', '')}.txt"
                normalized_text_files[text_name] = normalized["text"].encode("utf-8")
                evidence_without_text = json.loads(json.dumps(evidence))
                evidence_without_text["normalized_text"].pop("text", None)
                evidence_without_text["normalized_text"]["artifact_file"] = text_name
                training_eligible = request["evidence_role"] == "historical_content_audit_candidate"
                evidence_without_text["slot_id"] = request["slot_id"]
                evidence_without_text["evidence_role"] = request["evidence_role"]
                evidence_without_text[
                    "training_eligible_for_pre_2019_development"
                ] = training_eligible
                evidence_without_text["standalone_primary_document"] = {
                    "filename": record.primary_document,
                    "retrieval": primary_audit_entry,
                    "normalized_text": standalone_normalized.to_dict(
                        include_text=False
                    ),
                    "matches_complete_submission_normalized_text": True,
                }
                filing_evidence.append(evidence_without_text)
                text_role_entries.append(
                    {
                        "accession_number": accession,
                        "artifact_file": text_name,
                        "slot_id": request["slot_id"],
                        "evidence_role": request["evidence_role"],
                        "training_eligible_for_pre_2019_development": training_eligible,
                    }
                )
                reconciled_identity = NormalizedFilingIdentity(
                    accession=evidence["accession_number"],
                    form=evidence["form"],
                    filing_date=evidence["filing_date"],
                    subject_cik=evidence["subject_cik"],
                )
                opened.append(
                    FilingAuditResult(
                        slot_id=request["slot_id"],
                        status="selected",
                        evidence_role=request["evidence_role"],
                        selected=selected_identity,
                        submissions=selected_identity,
                        master_idx=master_identity,
                        sgml_header=reconciled_identity,
                        index_metadata=NormalizedIndexIdentity(
                            accession=evidence["accession_number"],
                            subject_cik=evidence["subject_cik"],
                        ),
                        exact_acceptance_timestamp=bool(
                            evidence["exact_acceptance_timestamp"]
                        ),
                        usable_visible_text=bool(normalized["usable"]),
                    )
                )
            except (SecPointInTimeError, TypeError, ValueError) as exc:
                failure = _safe_error_evidence(exc)
                filing_evidence.append(
                    {
                        "accession_number": accession,
                        "slot_id": request["slot_id"],
                        "status": "failed",
                        **failure,
                    }
                )
                opened.append(
                    FilingAuditResult(
                        slot_id=request["slot_id"],
                        status="selected",
                        evidence_role=request["evidence_role"],
                        selected=selected_identity,
                        submissions=selected_identity,
                        master_idx=master_identity,
                        sgml_header=None,
                        index_metadata=None,
                        exact_acceptance_timestamp=False,
                        usable_visible_text=False,
                        missing_reason=failure["failure_code"],
                    )
                )
        filing_results = opened

    evaluation = evaluate_sec_audit(filing_results, _periodic_results(plan))
    transport_state, resource_gate_passed = _sanitized_transport_state(
        transport, request_ledger
    )
    excluded_accessions = {
        item["accession_number"] for item in plan["exclusions"]
    }
    bounded_filings = [
        record
        for record in filings
        if record.accession_number not in excluded_accessions
    ]
    artifact_boundary_passed = bool(
        len(bounded_filings) + len(excluded_accessions) == len(filings)
        and all(
            plan["audit_start"] <= record.filing_date <= plan["audit_cutoff"]
            and (
                not record.date_of_filing_date_change
                or plan["audit_start"]
                <= record.date_of_filing_date_change
                <= plan["audit_cutoff"]
            )
            for record in bounded_filings
        )
    )
    availability_count = sum(
        item.get("availability_session_found") is True
        for item in filing_evidence
        if item.get("status") != "failed"
    )
    offline_pipeline_pass = bool(
        plan["ready_for_bounded_download"]
        and evaluation["overall_pass"]
        and resource_gate_passed
        and artifact_boundary_passed
        and availability_count == 24
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "source_commit": source_commit.lower(),
        "source_provenance": provenance,
        "audit_start": plan["audit_start"],
        "audit_cutoff": plan["audit_cutoff"],
        "catalogue_ready_for_bounded_download": plan["ready_for_bounded_download"],
        "document_download_opened": plan["ready_for_bounded_download"],
        "evaluation": evaluation,
        "resource_evidence": transport_state["budget"],
        "resource_and_user_agent_gate_passed": resource_gate_passed,
        "post_2024_artifact_boundary_gate_passed": artifact_boundary_passed,
        "availability_session_gate": {
            "required_count": 24,
            "observed_count": availability_count,
            "passed": availability_count == 24,
        },
        "transport_trust": {
            "trusted_production_transport": trusted_production_transport,
            "mode": (
                "bounded_official_sec_transport"
                if trusted_production_transport
                else "untrusted_injected_test_transport"
            ),
        },
        "behavior_evidence": {
            "llm_or_model_calls": 0,
            "return_calculations": 0,
            "trading_simulations": 0,
            "outcome_guided_substitutions": 0,
            "paid_api_calls": 0,
        },
        "offline_pipeline_pass": offline_pipeline_pass,
        "overall_pass": bool(
            offline_pipeline_pass and trusted_production_transport
        ),
    }
    files = {
        "audit_plan.json": _json_bytes(_artifact_safe_plan(plan)),
        "audit_report.json": _json_bytes(report),
        "catalogue.json": _json_bytes(
            [_record_dict(record) for record in bounded_filings]
        ),
        "filing_evidence.json": _json_bytes(filing_evidence),
        "request_ledger.json": _json_bytes(request_ledger),
        "session_calendar.json": _json_bytes(calendar_evidence),
        "text_role_manifest.json": _json_bytes(
            sorted(text_role_entries, key=lambda item: item["slot_id"])
        ),
        "transport_state.json": _json_bytes(transport_state),
        **normalized_text_files,
    }
    return seal_artifact(
        artifact_dir,
        files,
        source_commit=source_commit,
        audit_contract_version=CONTRACT_VERSION,
    )


def run_sec_audit(
    *,
    transport: TransportLike,
    aapl_sessions: Sequence[str],
    artifact_dir: str | Path,
    source_repo: str | Path,
) -> ArtifactVerification:
    """Run the offline/injected audit pipeline without a production pass."""

    return _run_sec_audit(
        transport=transport,
        aapl_sessions=aapl_sessions,
        artifact_dir=artifact_dir,
        source_repo=source_repo,
        trusted_production_transport=False,
    )


def run_live_sec_audit(
    *,
    user_agent: str,
    cache_dir: str | Path,
    aapl_sessions: Sequence[str],
    artifact_dir: str | Path,
    source_repo: str | Path,
) -> ArtifactVerification:
    """Run through the internally constructed bounded official SEC transport."""

    import time

    import requests

    budget = BudgetCounter()
    with requests.Session() as session:
        transport = SecAuditTransport(
            session=session,
            cache_dir=Path(cache_dir),
            user_agent=user_agent,
            budget=budget,
            clock=time.monotonic,
            sleep=time.sleep,
        )
        return _run_sec_audit(
            transport=transport,
            aapl_sessions=aapl_sessions,
            artifact_dir=artifact_dir,
            source_repo=source_repo,
            trusted_production_transport=True,
        )


__all__ = [
    "CONTRACT_VERSION",
    "SOURCE_FILES",
    "TransportLike",
    "run_sec_audit",
    "run_live_sec_audit",
    "verify_source_provenance",
]
