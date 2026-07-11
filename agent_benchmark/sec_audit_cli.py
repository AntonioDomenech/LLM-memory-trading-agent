"""Privacy-safe command line for the bounded official-SEC filing audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import stat as stat_module
import sys
from typing import Any, Sequence

from .sec_audit_artifact import read_artifact_file, verify_artifact
from .sec_audit_runner import (
    CONTRACT_VERSION,
    run_live_sec_audit,
    validate_local_filesystem_path,
    verify_production_sec_audit_artifact,
    verify_source_provenance,
)
from .sec_point_in_time import SecPointInTimeError, validate_sec_user_agent
from .sec_session_calendar import EXPECTED_SESSIONS, validate_aapl_session_calendar


MAX_CONFIG_BYTES = 1024 * 1024
DEFAULT_CONFIG = Path("data/local_config.json")
DEFAULT_CACHE_ROOT = Path("data/cache/sec_point_in_time_audit_v1")
DEFAULT_ARTIFACT_ROOT = Path("e/sec_point_in_time_audit_v1")
SUPPORTED_RUNTIME_POLICY = "python-3.11-to-3.13_requests-2.31plus_urllib3-2.x_v1"
_SAFE_ARTIFACT_NAME_RE = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\Z"
)


class SecAuditCliError(RuntimeError):
    """A safe, fixed-code CLI preflight failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise SecAuditCliError("invalid_arguments")

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if status == 0 and message is None:
            raise SystemExit(0)
        del message
        raise SecAuditCliError("invalid_arguments")


class _PrivateText:
    """A secret value whose normal diagnostics are deliberately redacted."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        self.__value = value

    def reveal_for_transport(self) -> str:
        return self.__value

    def __repr__(self) -> str:
        return "_PrivateText(<redacted>)"

    def __str__(self) -> str:
        return "<redacted>"

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError("private SEC contact state is not serializable")


class _PreparedAudit:
    __slots__ = (
        "__user_agent",
        "repo",
        "cache",
        "artifact",
        "artifact_reference",
        "safe_output",
    )

    def __init__(
        self,
        *,
        user_agent: _PrivateText,
        repo: Path,
        cache: Path,
        artifact: Path,
        artifact_reference: str,
        safe_output: dict[str, Any],
    ) -> None:
        self.__user_agent = user_agent
        self.repo = repo
        self.cache = cache
        self.artifact = artifact
        self.artifact_reference = artifact_reference
        self.safe_output = safe_output

    def reveal_user_agent_for_transport(self) -> str:
        return self.__user_agent.reveal_for_transport()

    def __repr__(self) -> str:
        return (
            "_PreparedAudit(private_contact=<redacted>, "
            f"artifact_reference={self.artifact_reference!r})"
        )

    def __reduce_ex__(self, protocol: int) -> Any:
        del protocol
        raise TypeError("prepared SEC audit state is not serializable")


def _json_output(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _major_minor(value: str) -> tuple[int, int]:
    match = re.match(r"^([0-9]+)\.([0-9]+)", value)
    if match is None:
        raise SecAuditCliError("live_runtime_unsupported")
    return int(match.group(1)), int(match.group(2))


def _validate_live_runtime() -> dict[str, Any]:
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise SecAuditCliError("live_runtime_unsupported")
    try:
        import requests

        requests_version = importlib_metadata.version("requests")
        urllib3_version = importlib_metadata.version("urllib3")
        certifi_version = importlib_metadata.version("certifi")
    except (ImportError, importlib_metadata.PackageNotFoundError) as exc:
        raise SecAuditCliError("live_runtime_dependency_missing") from exc
    if (
        _major_minor(requests_version) < (2, 31)
        or _major_minor(requests_version)[0] != 2
        or _major_minor(urllib3_version)[0] != 2
        or not callable(getattr(requests, "Session", None))
    ):
        raise SecAuditCliError("live_runtime_unsupported")
    junction_api_available = hasattr(Path("."), "is_junction") or hasattr(
        os.path, "isjunction"
    )
    if platform.system() == "Windows" and not junction_api_available:
        raise SecAuditCliError("junction_detection_unavailable")
    return {
        "runtime_policy": SUPPORTED_RUNTIME_POLICY,
        "python_version": platform.python_version(),
        "requests_version": requests_version,
        "urllib3_version": urllib3_version,
        "certifi_version": certifi_version,
        "junction_detection_available": junction_api_available,
    }


def _resolve_repo(path: str | Path) -> Path:
    lexical = _lexical_absolute(Path(path))
    _assert_local_path(lexical, failure_code="source_repo_missing_or_unsafe")
    _assert_no_link_components(lexical, failure_code="source_repo_missing_or_unsafe")
    repo = lexical.resolve()
    if not repo.is_dir():
        raise SecAuditCliError("source_repo_missing")
    return repo


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _assert_local_path(path: Path, *, failure_code: str) -> None:
    try:
        validate_local_filesystem_path(path)
    except SecPointInTimeError as exc:
        raise SecAuditCliError(failure_code) from exc


def _is_link_or_junction(path: Path) -> bool:
    try:
        try:
            details = path.lstat()
        except FileNotFoundError:
            return False
        if stat_module.S_ISLNK(details.st_mode):
            return True
        reparse_flag = getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if getattr(details, "st_file_attributes", 0) & reparse_flag:
            return True
        os_is_junction = getattr(os.path, "isjunction", None)
        if os_is_junction is not None and os_is_junction(path):
            return True
        path_is_junction = getattr(path, "is_junction", None)
        return bool(path_is_junction is not None and path_is_junction())
    except OSError:
        return True


def _assert_no_link_components(path: Path, *, failure_code: str) -> None:
    current = _lexical_absolute(path)
    while True:
        if _is_link_or_junction(current):
            raise SecAuditCliError(failure_code)
        parent = current.parent
        if parent == current:
            break
        current = parent


def _resolve_config(repo: Path, path: str | Path) -> Path:
    candidate = Path(path)
    lexical = _lexical_absolute(repo / candidate if not candidate.is_absolute() else candidate)
    _assert_local_path(lexical, failure_code="sec_config_missing_or_unsafe")
    _assert_no_link_components(
        lexical, failure_code="sec_config_missing_or_unsafe"
    )
    return lexical.resolve()


def _resolve_within(
    repo: Path,
    path: str | Path,
    *,
    root: Path,
    allow_root: bool,
    failure_code: str,
) -> Path:
    allowed_lexical = _lexical_absolute(repo / root)
    _assert_local_path(allowed_lexical, failure_code=failure_code)
    _assert_no_link_components(allowed_lexical, failure_code=failure_code)
    allowed = allowed_lexical.resolve()
    candidate = Path(path)
    candidate_lexical = _lexical_absolute(
        repo / candidate if not candidate.is_absolute() else candidate
    )
    _assert_local_path(candidate_lexical, failure_code=failure_code)
    _assert_no_link_components(candidate_lexical, failure_code=failure_code)
    resolved = candidate_lexical.resolve()
    try:
        relative = resolved.relative_to(allowed)
    except ValueError as exc:
        raise SecAuditCliError(failure_code) from exc
    if not allow_root and not relative.parts:
        raise SecAuditCliError(failure_code)
    return resolved


def _read_config_bytes(config_path: Path) -> tuple[bytes | None, str | None]:
    try:
        _assert_local_path(
            config_path, failure_code="sec_config_missing_or_unsafe"
        )
        _assert_no_link_components(
            config_path, failure_code="sec_config_missing_or_unsafe"
        )
        if not config_path.is_file():
            return None, "sec_config_missing_or_unsafe"
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(config_path, flags)
        try:
            opened = os.fstat(descriptor)
            if opened.st_size > MAX_CONFIG_BYTES:
                return None, "sec_config_missing_or_unsafe"
            raw = os.read(descriptor, MAX_CONFIG_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(raw) > MAX_CONFIG_BYTES or len(raw) != opened.st_size:
            return None, "sec_config_missing_or_unsafe"
        _assert_local_path(
            config_path, failure_code="sec_config_missing_or_unsafe"
        )
        _assert_no_link_components(
            config_path, failure_code="sec_config_missing_or_unsafe"
        )
        current = config_path.stat()
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            current.st_dev,
            current.st_ino,
            current.st_size,
        ):
            return None, "sec_config_changed_during_read"
        return raw, None
    except SecAuditCliError as exc:
        return None, exc.code
    except OSError:
        return None, "sec_config_unreadable"


def _parse_config_bytes(raw: bytes) -> tuple[dict[str, Any] | None, str | None]:
    try:
        text = raw.decode("utf-8", errors="strict")
        payload = json.loads(text)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        return None, "sec_config_unreadable"
    if not isinstance(payload, dict):
        return None, "sec_config_invalid"
    return payload, None


def _validated_private_contact(
    payload: dict[str, Any],
) -> tuple[_PrivateText | None, str | None, str | None]:
    secrets = payload.get("secrets")
    if not isinstance(secrets, dict):
        return None, None, "sec_user_agent_missing"
    user_agent = secrets.get("sec_user_agent")
    if not isinstance(user_agent, str) or not user_agent.strip():
        return None, None, "sec_user_agent_missing"
    try:
        audit = validate_sec_user_agent(user_agent)
    except SecPointInTimeError:
        return None, None, "sec_user_agent_invalid"
    return _PrivateText(user_agent.strip()), audit.sha256, None


def _load_sec_user_agent(config_path: Path) -> tuple[_PrivateText, str]:
    raw, error = _read_config_bytes(config_path)
    if error is not None or raw is None:
        raise SecAuditCliError(error or "sec_config_unreadable")

    payload, error = _parse_config_bytes(raw)
    raw = b""
    if error is not None or payload is None:
        raise SecAuditCliError(error or "sec_config_invalid")

    private_contact, contact_sha256, error = _validated_private_contact(payload)
    payload = None
    if error is not None or private_contact is None or contact_sha256 is None:
        raise SecAuditCliError(error or "sec_user_agent_invalid")
    return private_contact, contact_sha256


def _default_artifact_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fz")
    return f"sec-audit-{stamp}"


def _artifact_reference(repo: Path, artifact: Path, *, generated: bool) -> str:
    if generated:
        return artifact.name
    relative = artifact.relative_to(repo).as_posix().encode("utf-8")
    return "sha256:" + hashlib.sha256(relative).hexdigest()


def _prepare(
    *,
    source_repo: str | Path,
    config: str | Path,
    cache_dir: str | Path,
    artifact_dir: str | Path | None,
) -> _PreparedAudit:
    repo = _resolve_repo(source_repo)
    try:
        provenance = verify_source_provenance(repo)
    except (SecPointInTimeError, TypeError, ValueError) as exc:
        raise SecAuditCliError("source_provenance_invalid") from exc
    calendar = validate_aapl_session_calendar(EXPECTED_SESSIONS)
    runtime = _validate_live_runtime()
    config_path = _resolve_config(repo, config)
    user_agent, user_agent_sha256 = _load_sec_user_agent(config_path)
    cache = _resolve_within(
        repo,
        cache_dir,
        root=DEFAULT_CACHE_ROOT,
        allow_root=True,
        failure_code="cache_path_outside_frozen_root",
    )
    generated_artifact = artifact_dir is None
    requested_artifact = (
        DEFAULT_ARTIFACT_ROOT / _default_artifact_name()
        if generated_artifact
        else Path(artifact_dir)
    )
    artifact_lexical = _lexical_absolute(
        repo / requested_artifact
        if not requested_artifact.is_absolute()
        else requested_artifact
    )
    artifact_root_lexical = _lexical_absolute(repo / DEFAULT_ARTIFACT_ROOT)
    try:
        artifact_relative = artifact_lexical.relative_to(artifact_root_lexical)
    except ValueError as exc:
        raise SecAuditCliError("artifact_path_outside_frozen_root") from exc
    if (
        len(artifact_relative.parts) != 1
        or _SAFE_ARTIFACT_NAME_RE.fullmatch(artifact_relative.name) is None
    ):
        raise SecAuditCliError("artifact_path_invalid")
    artifact = _resolve_within(
        repo,
        requested_artifact,
        root=DEFAULT_ARTIFACT_ROOT,
        allow_root=False,
        failure_code="artifact_path_outside_frozen_root",
    )
    if artifact.exists() or artifact.is_symlink():
        raise SecAuditCliError("artifact_path_already_exists")
    if cache.exists() and (cache.is_symlink() or not cache.is_dir()):
        raise SecAuditCliError("cache_path_unsafe")
    reference = _artifact_reference(repo, artifact, generated=generated_artifact)
    safe = {
        "status": "ready",
        "network_requests_performed": 0,
        "paid_api_calls": 0,
        "llm_or_model_calls": 0,
        "source_commit": provenance["commit"],
        "source_tree": provenance["tree"],
        "source_file_count": len(provenance["required_source_blobs"]),
        "calendar_id": calendar["calendar_id"],
        "calendar_sessions_sha256": calendar["sessions_sha256"],
        "runtime": runtime,
        "sec_user_agent_sha256": user_agent_sha256,
        "sec_user_agent_real_contact_validated": True,
        "artifact_reference": reference,
        "live_execution_requires_explicit_flag": True,
    }
    return _PreparedAudit(
        user_agent=user_agent,
        repo=repo,
        cache=cache,
        artifact=artifact,
        artifact_reference=reference,
        safe_output=safe,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Preflight or execute the bounded Apple SEC filing audit"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Validate readiness without network access (default)",
    )
    mode.add_argument(
        "--execute-live",
        action="store_true",
        help="Contact only official SEC endpoints and seal the audit",
    )
    parser.add_argument("--source-repo", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--artifact-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    execution_started = False
    try:
        args = build_parser().parse_args(argv)
        prepared = _prepare(
            source_repo=args.source_repo,
            config=args.config,
            cache_dir=args.cache_dir,
            artifact_dir=args.artifact_dir,
        )
        if not args.execute_live:
            print(_json_output(prepared.safe_output))
            return 0
        execution_started = True
        _assert_no_link_components(
            prepared.cache, failure_code="cache_path_unsafe"
        )
        _assert_no_link_components(
            prepared.artifact, failure_code="artifact_path_unsafe"
        )
        verification = run_live_sec_audit(
            user_agent=prepared.reveal_user_agent_for_transport(),
            cache_dir=prepared.cache,
            aapl_sessions=EXPECTED_SESSIONS,
            artifact_dir=prepared.artifact,
            source_repo=prepared.repo,
        )
        if Path(verification.artifact_dir).resolve() != prepared.artifact:
            raise SecAuditCliError("sealed_artifact_path_mismatch")
        verified = verify_artifact(
            prepared.artifact,
            expected_checksums_sha256=verification.checksums_sha256,
            expected_source_commit=prepared.safe_output["source_commit"],
            expected_audit_contract_version=CONTRACT_VERSION,
        )
        if verified != verification:
            raise SecAuditCliError("sealed_artifact_verification_mismatch")
        try:
            report = json.loads(
                read_artifact_file(
                    prepared.artifact, "audit_report.json"
                ).decode("utf-8", errors="strict")
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise SecAuditCliError("sealed_report_unreadable") from exc
        if not isinstance(report, dict) or not isinstance(
            report.get("overall_pass"), bool
        ):
            raise SecAuditCliError("sealed_report_invalid")
        if (
            report.get("contract_version") != CONTRACT_VERSION
            or report.get("source_commit") != prepared.safe_output["source_commit"]
        ):
            raise SecAuditCliError("sealed_report_provenance_mismatch")
        if report["overall_pass"]:
            trust = report.get("transport_trust")
            behavior = report.get("behavior_evidence")
            if (
                not isinstance(trust, dict)
                or trust.get("trusted_production_transport") is not True
                or trust.get("fresh_official_retrieval_gate_passed") is not True
                or not isinstance(behavior, dict)
                or behavior.get("paid_api_calls") != 0
                or behavior.get("llm_or_model_calls") != 0
            ):
                raise SecAuditCliError("sealed_passing_report_gates_invalid")
            semantic_verification, semantic_report = (
                verify_production_sec_audit_artifact(
                    prepared.artifact,
                    expected_checksums_sha256=verified.checksums_sha256,
                    expected_source_commit=prepared.safe_output["source_commit"],
                )
            )
            if semantic_verification != verified or semantic_report != report:
                raise SecAuditCliError("sealed_semantic_verification_mismatch")
        output = {
            "status": "passed" if report["overall_pass"] else "failed",
            "overall_pass": report["overall_pass"],
            "artifact_reference": prepared.artifact_reference,
            "checksums_sha256": verified.checksums_sha256,
            "source_commit": verified.source_commit,
            "audit_contract_version": verified.audit_contract_version,
            "paid_api_calls": 0,
            "llm_or_model_calls": 0,
        }
        print(_json_output(output))
        return 0 if report["overall_pass"] else 3
    except SecAuditCliError as exc:
        print(
            _json_output(
                {
                    "status": "blocked",
                    "reason_code": exc.code,
                    "network_requests_performed": (
                        "unknown_but_bounded_by_audit_transport"
                        if execution_started
                        else 0
                    ),
                    "paid_api_calls": 0,
                    "llm_or_model_calls": 0,
                }
            )
        )
        return 2
    except Exception:
        print(
            _json_output(
                {
                    "status": "blocked",
                    "reason_code": (
                        "live_execution_failed_safely"
                        if execution_started
                        else "preflight_failed_safely"
                    ),
                    "network_requests_performed": (
                        "unknown_but_bounded_by_audit_transport"
                        if execution_started
                        else 0
                    ),
                    "paid_api_calls": 0,
                    "llm_or_model_calls": 0,
                }
            )
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["SecAuditCliError", "build_parser", "main"]
