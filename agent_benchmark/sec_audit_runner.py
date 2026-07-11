"""Bounded, injected-transport runner for the frozen Apple SEC audit.

This module joins the already-frozen catalogue, selection, content, evaluation,
and artifact primitives.  It deliberately has no concrete HTTP client or CLI:
callers must inject a transport, which keeps tests offline and prevents an
unconfigured run from contacting the SEC.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import ssl
import stat as stat_module
import subprocess
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from .sec_audit_artifact import (
    METADATA_FILENAME,
    ArtifactVerification,
    read_artifact_file,
    seal_artifact,
    verify_artifact,
)
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
from .sec_session_calendar import EXPECTED_SESSIONS, validate_aapl_session_calendar


CONTRACT_VERSION = "aapl-sec-point-in-time-audit-runner-v2"
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_MAX_GIT_CONTROL_BYTES = 1024 * 1024
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
    "agent_benchmark/sec_audit_cli.py",
    "docs/aapl_point_in_time_text_data_audit_v1.md",
    "requirements.txt",
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


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if stat_module.S_ISLNK(details.st_mode):
        return True
    reparse_flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(details, "st_file_attributes", 0) & reparse_flag)


def _windows_path_is_local(path: Path) -> bool:
    if os.name != "nt":
        return True
    raw = str(path).replace("/", "\\")
    if raw.startswith("\\\\"):
        return False
    anchor = path.anchor
    if not anchor:
        return False
    try:
        import ctypes

        drive_type = int(ctypes.windll.kernel32.GetDriveTypeW(str(anchor)))
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    # Removable, fixed, optical, and RAM-disk roots are local. Unknown,
    # missing, and remote roots fail closed.
    return drive_type in {2, 3, 5, 6}


def validate_local_filesystem_path(path: str | Path) -> Path:
    """Return a lexical absolute path only when it cannot address a network FS."""

    lexical = Path(os.path.abspath(Path(path)))
    if not _windows_path_is_local(lexical):
        raise SecPointInTimeError("filesystem path must be on a local device")
    current = lexical
    while True:
        if _is_link_or_reparse_point(current):
            raise SecPointInTimeError(
                "filesystem path cannot contain links or reparse points"
            )
        parent = current.parent
        if parent == current:
            break
        current = parent
    return lexical


def _read_git_control_file(path: Path) -> str:
    validate_local_filesystem_path(path)
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
        try:
            details = os.fstat(descriptor)
            if details.st_size > _MAX_GIT_CONTROL_BYTES:
                raise SecPointInTimeError("source Git control file is too large")
            payload = os.read(descriptor, _MAX_GIT_CONTROL_BYTES + 1)
        finally:
            os.close(descriptor)
    except SecPointInTimeError:
        raise
    except OSError as exc:
        raise SecPointInTimeError("source Git control file is unreadable") from exc
    if len(payload) > _MAX_GIT_CONTROL_BYTES or len(payload) != details.st_size:
        raise SecPointInTimeError("source Git control file changed during read")
    validate_local_filesystem_path(path)
    if payload.startswith(b"\xef\xbb\xbf"):
        raise SecPointInTimeError(
            "source Git control file cannot contain a byte-order mark"
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SecPointInTimeError("source Git control file is not UTF-8") from exc
    if "\ufeff" in text:
        raise SecPointInTimeError(
            "source Git control file cannot contain a byte-order mark"
        )
    return text


def _validate_local_git_metadata(repo: Path) -> None:
    git_dir = validate_local_filesystem_path(repo / ".git")
    if not git_dir.is_dir():
        raise SecPointInTimeError(
            "source repository must use a local standalone Git directory"
        )

    # Refuse any reparse point anywhere Git may read. This prevents a local
    # worktree from silently redirecting objects, refs, or configuration to SMB.
    pending = [git_dir]
    while pending:
        current = pending.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise SecPointInTimeError("source Git metadata is unreadable") from exc
        for entry in entries:
            child = Path(entry.path)
            if _is_link_or_reparse_point(child):
                raise SecPointInTimeError(
                    "source Git metadata cannot contain links or reparse points"
                )
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(child)
            except OSError as exc:
                raise SecPointInTimeError("source Git metadata is unreadable") from exc

    for forbidden in (
        git_dir / "commondir",
        git_dir / "config.worktree",
        git_dir / "objects" / "info" / "alternates",
        git_dir / "objects" / "info" / "http-alternates",
    ):
        if forbidden.exists():
            raise SecPointInTimeError(
                "source Git metadata cannot redirect configuration or objects"
            )

    config = _read_git_control_file(git_dir / "config")
    if re.search(r"(?im)^\s*\[\s*include", config):
        raise SecPointInTimeError("source Git configuration includes are not allowed")
    if re.search(
        r"(?im)^\s*(?:worktree|attributesfile|excludesfile|"
        r"alternaterefscommand|clean|smudge|process|textconv|external)\s*=",
        config,
    ):
        raise SecPointInTimeError(
            "source Git configuration cannot execute or redirect helpers"
        )
    if re.search(r"(?im)^\s*(?:promisor|partialclonefilter)\s*=", config):
        raise SecPointInTimeError(
            "partial or promisor source repositories are not allowed"
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


def _runtime_manifest(*, live: bool) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("requests", "urllib3", "certifi", "tzdata"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "mode": "live_official_sec" if live else "offline_injected",
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "openssl_version": ssl.OPENSSL_VERSION,
        "packages": packages,
        "timezone_data_source": (
            f"tzdata-{packages['tzdata']}"
            if packages["tzdata"] is not None
            else "system-zoneinfo"
        ),
    }


def _read_artifact_json(artifact_dir: Path, name: str) -> Any:
    try:
        return json.loads(
            read_artifact_file(artifact_dir, name).decode(
                "utf-8", errors="strict"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SecPointInTimeError(
            "sealed SEC audit contains unreadable JSON evidence"
        ) from exc


def verify_production_sec_audit_artifact(
    artifact_dir: str | Path,
    *,
    expected_checksums_sha256: str,
    expected_source_commit: str,
) -> tuple[ArtifactVerification, dict[str, Any]]:
    """Verify both byte integrity and complete production-audit semantics."""

    directory = Path(artifact_dir).resolve()
    verification = verify_artifact(
        directory,
        expected_checksums_sha256=expected_checksums_sha256,
        expected_source_commit=expected_source_commit,
        expected_audit_contract_version=CONTRACT_VERSION,
    )
    report = _read_artifact_json(directory, "audit_report.json")
    plan = _read_artifact_json(directory, "audit_plan.json")
    catalogue = _read_artifact_json(directory, "catalogue.json")
    evidence = _read_artifact_json(directory, "filing_evidence.json")
    ledger = _read_artifact_json(directory, "request_ledger.json")
    calendar = _read_artifact_json(directory, "session_calendar.json")
    roles = _read_artifact_json(directory, "text_role_manifest.json")
    transport = _read_artifact_json(directory, "transport_state.json")
    if not all(
        isinstance(value, dict)
        for value in (report, plan, calendar, transport)
    ) or not all(isinstance(value, list) for value in (catalogue, evidence, ledger, roles)):
        raise SecPointInTimeError("sealed SEC audit evidence schemas are invalid")

    text_files = {
        item.get("artifact_file")
        for item in roles
        if isinstance(item, dict) and isinstance(item.get("artifact_file"), str)
    }
    if (
        len(roles) != 24
        or len(text_files) != 24
        or any(not re.fullmatch(r"text_[0-9]{18}\.txt", name) for name in text_files)
    ):
        raise SecPointInTimeError("sealed SEC audit text-role manifest is incomplete")
    expected_files = {
        METADATA_FILENAME,
        "audit_plan.json",
        "audit_report.json",
        "catalogue.json",
        "filing_evidence.json",
        "request_ledger.json",
        "session_calendar.json",
        "text_role_manifest.json",
        "transport_state.json",
        *text_files,
    }
    if set(verification.checksums) != expected_files:
        raise SecPointInTimeError("sealed SEC audit file set is not exact")

    selection = plan.get("selection")
    evaluation = report.get("evaluation")
    trust = report.get("transport_trust")
    behavior = report.get("behavior_evidence")
    availability = report.get("availability_session_gate")
    runtime = report.get("runtime_provenance")
    if not all(
        isinstance(value, dict)
        for value in (
            selection,
            evaluation,
            trust,
            behavior,
            availability,
            runtime,
        )
    ):
        raise SecPointInTimeError("sealed SEC audit report gates are incomplete")
    if (
        report.get("contract_version") != CONTRACT_VERSION
        or report.get("source_commit") != expected_source_commit
        or report.get("overall_pass") is not True
        or report.get("offline_pipeline_pass") is not True
        or report.get("catalogue_ready_for_bounded_download") is not True
        or report.get("document_download_opened") is not True
        or report.get("resource_and_user_agent_gate_passed") is not True
        or report.get("post_2024_artifact_boundary_gate_passed") is not True
        or availability.get("passed") is not True
        or availability.get("observed_count") != 24
        or evaluation.get("overall_pass") is not True
        or plan.get("ready_for_bounded_download") is not True
        or selection.get("selected_count") != 24
        or selection.get("gap_count") != 0
        or trust.get("trusted_production_transport") is not True
        or trust.get("mode") != "bounded_official_sec_transport"
        or trust.get("cache_hit_count") != 0
        or trust.get("fresh_official_retrieval_gate_passed") is not True
        or behavior.get("paid_api_calls") != 0
        or behavior.get("llm_or_model_calls") != 0
        or behavior.get("return_calculations") != 0
        or behavior.get("trading_simulations") != 0
        or behavior.get("outcome_guided_substitutions") != 0
        or runtime != _runtime_manifest(live=True)
    ):
        raise SecPointInTimeError("sealed SEC audit passing gates are invalid")
    gates = evaluation.get("gates")
    plan_gates = plan.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(not isinstance(gate, dict) or gate.get("passed") is not True for gate in gates.values())
        or not isinstance(plan_gates, dict)
        or not plan_gates
        or any(value is not True for value in plan_gates.values())
    ):
        raise SecPointInTimeError("sealed SEC audit component gates did not all pass")

    selected_accessions = selection.get("selected_accessions")
    if (
        not isinstance(selected_accessions, list)
        or len(selected_accessions) != 24
        or len(set(selected_accessions)) != 24
    ):
        raise SecPointInTimeError("sealed SEC audit selection is not exact")
    selected = set(selected_accessions)
    catalogue_accessions = {
        item.get("accession_number") for item in catalogue if isinstance(item, dict)
    }
    evidence_accessions = {
        item.get("accession_number") for item in evidence if isinstance(item, dict)
    }
    role_accessions = {
        item.get("accession_number") for item in roles if isinstance(item, dict)
    }
    if (
        len(evidence) != 24
        or evidence_accessions != selected
        or role_accessions != selected
        or not selected.issubset(catalogue_accessions)
        or any(item.get("status") == "failed" for item in evidence if isinstance(item, dict))
    ):
        raise SecPointInTimeError("sealed SEC audit accession evidence does not reconcile")
    structural = [
        item
        for item in roles
        if isinstance(item, dict) and item.get("evidence_role") == "structural_only"
    ]
    if (
        len(structural) != 6
        or any(item.get("training_eligible_for_pre_2019_development") is not False for item in structural)
    ):
        raise SecPointInTimeError("sealed SEC audit structural-only roles are invalid")

    validated_calendar = validate_aapl_session_calendar(calendar.get("sessions"))
    if calendar != validated_calendar or tuple(calendar["sessions"]) != EXPECTED_SESSIONS:
        raise SecPointInTimeError("sealed SEC audit calendar does not reconcile")
    if not ledger or any(
        not isinstance(item, dict)
        or item.get("cache_hit") is not False
        or canonical_sec_url(item.get("requested_url")) != item.get("requested_url")
        or canonical_sec_url(item.get("url")) != item.get("url")
        for item in ledger
    ):
        raise SecPointInTimeError("sealed SEC audit request ledger is not fresh and official")
    budget = transport.get("budget")
    user_agent = transport.get("user_agent")
    if (
        transport.get("ledger_reconciled") is not True
        or not isinstance(budget, dict)
        or not isinstance(user_agent, dict)
        or user_agent.get("real_contact_validated") is not True
        or budget.get("requests") != sum(item["network_requests"] for item in ledger)
        or budget.get("bytes_received") != sum(item["size_bytes"] for item in ledger)
        or budget.get("requests", MAX_AUDIT_REQUESTS + 1) > MAX_AUDIT_REQUESTS
        or budget.get("bytes_received", MAX_AUDIT_BYTES + 1) > MAX_AUDIT_BYTES
        or budget.get("elapsed_seconds", MAX_AUDIT_SECONDS + 1) > MAX_AUDIT_SECONDS
    ):
        raise SecPointInTimeError("sealed SEC audit transport ledger does not reconcile")

    evidence_by_accession = {
        item["accession_number"]: item for item in evidence if isinstance(item, dict)
    }
    for role in roles:
        name = role["artifact_file"]
        item = evidence_by_accession[role["accession_number"]]
        normalized = item.get("normalized_text")
        if (
            not isinstance(normalized, dict)
            or normalized.get("artifact_file") != name
            or content_sha256(read_artifact_file(directory, name))
            != normalized.get("sha256")
        ):
            raise SecPointInTimeError("sealed SEC audit normalized text does not reconcile")

    final_verification = verify_artifact(
        directory,
        expected_checksums_sha256=expected_checksums_sha256,
        expected_source_commit=expected_source_commit,
        expected_audit_contract_version=CONTRACT_VERSION,
    )
    if final_verification != verification:
        raise SecPointInTimeError("sealed SEC audit changed during semantic verification")
    return final_verification, report


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

    lexical_repo = validate_local_filesystem_path(source_repo)
    repo = lexical_repo.resolve()
    if not repo.is_dir():
        raise SecPointInTimeError("source repository directory does not exist")
    _validate_local_git_metadata(repo)

    safe_git_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    safe_git_environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    safe_git_prefix = [
        "git",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.autocrlf=true",
        "-c",
        "credential.helper=",
        "-c",
        "gc.auto=0",
        "-c",
        "maintenance.auto=false",
        "-c",
        "protocol.allow=never",
        "-c",
        "submodule.recurse=false",
        "-C",
        str(repo),
    ]

    def run_git(
        *arguments: str,
        text: bool,
        allowed_returncodes: tuple[int, ...] = (0,),
    ) -> subprocess.CompletedProcess[Any]:
        try:
            completed = subprocess.run(
                [*safe_git_prefix, *arguments],
                check=False,
                capture_output=True,
                text=text,
                encoding="utf-8" if text else None,
                errors="strict" if text else None,
                env=safe_git_environment,
                stdin=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
            raise SecPointInTimeError("could not verify source Git provenance") from exc
        if completed.returncode not in allowed_returncodes:
            raise SecPointInTimeError("could not verify source Git provenance")
        return completed

    def git(*arguments: str) -> str:
        completed = run_git(*arguments, text=True)
        return completed.stdout.strip()

    def git_bytes(*arguments: str) -> bytes:
        completed = run_git(*arguments, text=False)
        return completed.stdout

    partial_clone = run_git(
        "config",
        "--local",
        "--get-regexp",
        r"^(extensions\.partialclone|remote\..*\.(promisor|partialclonefilter))$",
        text=True,
        allowed_returncodes=(0, 1),
    )
    if partial_clone.returncode == 0 and partial_clone.stdout.strip():
        raise SecPointInTimeError(
            "partial or promisor source repositories are not allowed"
        )

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
    runtime_provenance: Mapping[str, Any],
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
        "runtime_provenance": dict(runtime_provenance),
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
    runtime_provenance: Mapping[str, Any],
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
            runtime_provenance=runtime_provenance,
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
    cache_hit_count = sum(bool(item["cache_hit"]) for item in request_ledger)
    fresh_official_retrieval_gate = bool(
        request_ledger and cache_hit_count == 0
    )
    report = {
        "contract_version": CONTRACT_VERSION,
        "source_commit": source_commit.lower(),
        "source_provenance": provenance,
        "runtime_provenance": dict(runtime_provenance),
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
            "cache_hit_count": cache_hit_count,
            "fresh_official_retrieval_gate_passed": fresh_official_retrieval_gate,
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
            offline_pipeline_pass
            and trusted_production_transport
            and fresh_official_retrieval_gate
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
        runtime_provenance=_runtime_manifest(live=False),
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
            allow_cache_reads=False,
            allow_cache_writes=False,
        )
        return _run_sec_audit(
            transport=transport,
            aapl_sessions=aapl_sessions,
            artifact_dir=artifact_dir,
            source_repo=source_repo,
            trusted_production_transport=True,
            runtime_provenance=_runtime_manifest(live=True),
        )


__all__ = [
    "CONTRACT_VERSION",
    "SOURCE_FILES",
    "TransportLike",
    "run_sec_audit",
    "run_live_sec_audit",
    "verify_production_sec_audit_artifact",
    "verify_source_provenance",
]
