"""Development-only source acquisition for lean SEC/Gemma evidence v3.

This module deliberately stops at source suitability.  It downloads the exact
frozen development SEC and Yahoo inputs, builds the already-preregistered
blinded model requests, seals their bytes in an ignored local checkpoint, and
publishes only a redacted receipt.  It never calls Gemma, computes a trading
result, or opens confirmation/final data.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Any, Final
import uuid

from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    load_allowed_requests,
)


requests = load_allowed_requests()

from agent_benchmark.sec_audit_transport import SecAuditTransport
from agent_benchmark.sec_filing_gemma_contract import (
    build_corpus_universe_manifest,
)
from agent_benchmark.sec_filing_gemma_corpus import (
    MAIN_SUBMISSIONS_URL,
    SecCorpusBudget,
    _acquire_authenticated_stage_access_document_batch,
    _validate_persisted_authenticated_stage_access_batch,
    acquire_official_sec_catalog,
    validate_detached_catalog_replay,
)
from agent_benchmark.sec_gemma_lean_runner import (
    ACTIVE_LOCK_NAME,
    CHECKPOINT_ROOT,
    PREFLIGHT_ARTIFACT_PATH,
    PRIVATE_CONFIG_PATH,
    PREREGISTRATION_COMMIT,
    SCIENTIFIC_PARENT_COMMIT,
    SecGemmaLeanRunnerError,
    _RunLease,
    _canonical_repo_root,
    _contact_fingerprint,
    _default_git,
    _is_reparse,
    _prepare_checkpoint_root,
    _read_private_config_bytes,
    _require_owned_regular_file,
    _safe_directory,
    _validated_sealable_report,
    _verify_repository,
)
from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
    DEVELOPMENT,
    YAHOO_SYMBOLS,
    _PacedSecTransport,
    _acquire_market,
    _build_detached_stage_evidence,
    _build_model_requests,
    _build_model_slice,
    _build_stage_slice,
    _build_universe_event_proofs,
    _bundle,
    _execution_request_accounting,
    _market_commitments,
    _market_security_state,
    _official_primary_url,
    _sec_receipt,
    _validate_sec_detached_replay,
    _validate_one_bundle,
    build_acquisition_plan,
    create_production_market_transport,
    validate_acquisition_plan,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    MAX_MARKET_REQUESTS_PER_STAGE,
    MAX_SEC_REQUESTS_PER_SECOND,
    MAX_SEC_SECONDS,
    build_contract_manifest,
    canonical_json_bytes,
    canonical_sha256,
)
from agent_benchmark.sec_point_in_time import (
    MAX_AUDIT_BYTES,
    MAX_AUDIT_REQUESTS,
    BudgetCounter,
    validate_sec_user_agent,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-development-acquisition-v1"
)
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-source-checkpoint-v1"
)
SEC_PREFIX_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-sec-prefix-v1"
)
SEC_CATALOG_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-sec-catalog-checkpoint-v1"
)
SEC_DOCUMENT_CHUNK_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-sec-document-chunk-v1"
)
TREE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-external-byte-tree-v1"
)
ATTEMPT_EVENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-attempt-event-v1"
)
ACQUISITION_SOURCE_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_lean_acquisition.py"
)
PRIVATE_STAGE_DIRECTORY: Final[str] = "development_source_acquisition"
PRIVATE_SEC_PREFIX_DIRECTORY: Final[str] = "development_sec_prefix"
PRIVATE_SEC_CATALOG_DIRECTORY: Final[str] = "development_sec_catalog"
PRIVATE_SEC_DOCUMENT_CHUNK_PREFIX: Final[str] = "development_sec_documents_"
PRIVATE_ATTEMPT_JOURNAL_DIRECTORY: Final[str] = "development_attempt_journal"
PRIVATE_MANIFEST_NAME: Final[str] = "CHECKPOINT.json"
PRIVATE_TREE_NAME: Final[str] = "BUNDLE_TREE.json"
PUBLIC_ARTIFACT_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_evidence_v3/DEVELOPMENT_ACQUISITION.json"
)
SEC_REQUEST_CAP: Final[int] = MAX_AUDIT_REQUESTS
SEC_BYTE_CAP: Final[int] = MAX_AUDIT_BYTES
SEC_TIMEOUT_SECONDS: Final[float] = 30.0
SEC_DOCUMENT_CHUNK_SIZE: Final[int] = 1
LIFETIME_MARKET_REQUEST_CAP: Final[int] = 12

_SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_UTC_TIMESTAMP_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z\Z"
)

_FROZEN_HELPER_IDENTITIES: Final[dict[str, int]] = {
    "acquire_official_sec_catalog": id(acquire_official_sec_catalog),
    "acquire_authenticated_stage_access_document_batch": id(
        _acquire_authenticated_stage_access_document_batch
    ),
    "validate_persisted_authenticated_stage_access_batch": id(
        _validate_persisted_authenticated_stage_access_batch
    ),
    "validate_detached_catalog_replay": id(validate_detached_catalog_replay),
    "official_primary_url": id(_official_primary_url),
    "build_detached_stage_evidence": id(_build_detached_stage_evidence),
    "build_model_requests": id(_build_model_requests),
    "build_universe_event_proofs": id(_build_universe_event_proofs),
    "market_commitments": id(_market_commitments),
    "build_model_slice": id(_build_model_slice),
    "build_stage_slice": id(_build_stage_slice),
    "bundle": id(_bundle),
    "validate_one_bundle": id(_validate_one_bundle),
    "sec_audit_transport": id(SecAuditTransport),
    "paced_sec_transport": id(_PacedSecTransport),
    "budget_counter": id(BudgetCounter),
    "create_production_market_transport": id(
        create_production_market_transport
    ),
    "acquire_market": id(_acquire_market),
    "run_lease": id(_RunLease),
}


class SecGemmaLeanAcquisitionError(RuntimeError):
    """One fixed public-safe acquisition check failed."""

    def __init__(self, code: str):
        if (
            type(code) is not str
            or not code
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in code
            )
        ):
            code = "invalid_error_code"
        self.code = code
        super().__init__(code)

    def __repr__(self) -> str:
        return f"SecGemmaLeanAcquisitionError(code={self.code!r})"


class _PrivateSecContact:
    """Keep the readable SEC contact away from serialization and repr."""

    __slots__ = ("__value", "_sha256")

    def __init__(self, value: str):
        try:
            audit = validate_sec_user_agent(value)
        except Exception:
            raise SecGemmaLeanAcquisitionError(
                "private_contact_invalid"
            ) from None
        self.__value = value
        self._sha256 = audit.sha256

    @property
    def sha256(self) -> str:
        return self._sha256

    def reveal_for_sec_only(self) -> str:
        return self.__value

    def encoded_for_echo_check(self) -> bytes:
        return self.__value.encode("utf-8")

    def serialized_echo_needles(self) -> tuple[bytes, ...]:
        escaped = json.dumps(
            self.__value,
            ensure_ascii=True,
            allow_nan=False,
        )[1:-1].encode("ascii")
        return tuple(
            dict.fromkeys((self.__value.encode("utf-8"), escaped))
        )

    def __repr__(self) -> str:
        return "_PrivateSecContact(<redacted>)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise SecGemmaLeanAcquisitionError("checkpoint_value_invalid")
    return hashlib.sha256(value).hexdigest()


def _plain_json_bytes(value: Any) -> bytes:
    try:
        return canonical_json_bytes(value)
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_json_invalid"
        ) from None


def _to_plain_json_value(value: Any) -> Any:
    """Recursively detach immutable validator views into strict JSON data."""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SecGemmaLeanAcquisitionError("checkpoint_json_invalid")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise SecGemmaLeanAcquisitionError("checkpoint_json_invalid")
        return {
            key: _to_plain_json_value(item)
            for key, item in value.items()
        }
    if type(value) in {list, tuple}:
        return [_to_plain_json_value(item) for item in value]
    raise SecGemmaLeanAcquisitionError("checkpoint_json_invalid")


def _parse_utc_timestamp(value: Any) -> str:
    if type(value) is not str or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise SecGemmaLeanAcquisitionError("wall_clock_invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise SecGemmaLeanAcquisitionError("wall_clock_invalid") from None
    if parsed.tzinfo != timezone.utc:
        raise SecGemmaLeanAcquisitionError("wall_clock_invalid")
    return value


def _git_blob_bytes(repo_root: Path, revision_path: str) -> bytes:
    result = _default_git(repo_root, ("show", revision_path))
    if result.returncode != 0:
        raise SecGemmaLeanAcquisitionError("git_blob_unavailable")
    return result.stdout


def _preflight_receipt(repo_root: Path) -> dict[str, Any]:
    path = repo_root / Path(PREFLIGHT_ARTIFACT_PATH)
    try:
        raw = path.read_bytes()
        committed = _git_blob_bytes(
            repo_root, f"HEAD:{PREFLIGHT_ARTIFACT_PATH}"
        )
        parsed = json.loads(raw.decode("ascii", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanAcquisitionError(
            "preflight_receipt_invalid"
        ) from None
    if raw != committed:
        raise SecGemmaLeanAcquisitionError("preflight_receipt_uncommitted")
    try:
        validated = _validated_sealable_report(parsed)
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "preflight_receipt_invalid"
        ) from None
    if (
        validated.get("status") != "passed"
        or validated.get("execution_authorized") is not False
        or validated.get("checkpoint", {}).get(
            "acquisition_must_reacquire_and_reverify"
        )
        is not True
    ):
        raise SecGemmaLeanAcquisitionError("preflight_receipt_invalid")
    return dict(validated)


def _verify_frozen_helper_identities() -> None:
    observed = {
        "acquire_official_sec_catalog": id(acquire_official_sec_catalog),
        "acquire_authenticated_stage_access_document_batch": id(
            _acquire_authenticated_stage_access_document_batch
        ),
        "validate_persisted_authenticated_stage_access_batch": id(
            _validate_persisted_authenticated_stage_access_batch
        ),
        "validate_detached_catalog_replay": id(
            validate_detached_catalog_replay
        ),
        "official_primary_url": id(_official_primary_url),
        "build_detached_stage_evidence": id(_build_detached_stage_evidence),
        "build_model_requests": id(_build_model_requests),
        "build_universe_event_proofs": id(_build_universe_event_proofs),
        "market_commitments": id(_market_commitments),
        "build_model_slice": id(_build_model_slice),
        "build_stage_slice": id(_build_stage_slice),
        "bundle": id(_bundle),
        "validate_one_bundle": id(_validate_one_bundle),
        "sec_audit_transport": id(SecAuditTransport),
        "paced_sec_transport": id(_PacedSecTransport),
        "budget_counter": id(BudgetCounter),
        "create_production_market_transport": id(
            create_production_market_transport
        ),
        "acquire_market": id(_acquire_market),
        "run_lease": id(_RunLease),
    }
    if observed != _FROZEN_HELPER_IDENTITIES:
        raise SecGemmaLeanAcquisitionError("frozen_helper_identity_changed")


def _verify_acquisition_readiness(repo_root: Path) -> dict[str, Any]:
    _verify_frozen_helper_identities()
    try:
        repository = dict(_verify_repository(repo_root))
    except SecGemmaLeanRunnerError as exc:
        raise SecGemmaLeanAcquisitionError(exc.code) from None
    source_path = repo_root / Path(ACQUISITION_SOURCE_PATH)
    try:
        source_bytes = source_path.read_bytes()
        committed_source = _git_blob_bytes(
            repo_root, f"HEAD:{ACQUISITION_SOURCE_PATH}"
        )
        executing_path = Path(__file__).resolve(strict=True)
    except OSError:
        raise SecGemmaLeanAcquisitionError(
            "acquisition_source_invalid"
        ) from None
    if (
        executing_path != source_path.resolve(strict=True)
        or source_bytes != committed_source
    ):
        raise SecGemmaLeanAcquisitionError("acquisition_source_invalid")
    preflight = _preflight_receipt(repo_root)
    preflight_head = preflight.get("repository", {}).get("head_commit")
    if (
        type(preflight_head) is not str
        or _SHA1_RE.fullmatch(preflight_head) is None
        or _default_git(
            repo_root,
            (
                "merge-base",
                "--is-ancestor",
                preflight_head,
                repository["head_commit"],
            ),
        ).returncode
        != 0
    ):
        raise SecGemmaLeanAcquisitionError("preflight_ancestry_invalid")
    return {
        "head_commit": repository["head_commit"],
        "cached_upstream_commit": repository["cached_upstream_commit"],
        "branch": repository["branch"],
        "scientific_parent_commit": SCIENTIFIC_PARENT_COMMIT,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "frozen_scientific_source_blobs_sha256": repository[
            "frozen_scientific_source_blobs_sha256"
        ],
        "dependency_closure_sha256": repository["dependency_identity"][
            "verified_v22_dependency_closure_sha256"
        ],
        "acquisition_source_path": ACQUISITION_SOURCE_PATH,
        "acquisition_source_sha256": _sha256_bytes(source_bytes),
        "preflight_sha256": preflight["preflight_sha256"],
        "preflight_artifact_sha256": _sha256_bytes(
            (repo_root / Path(PREFLIGHT_ARTIFACT_PATH)).read_bytes()
        ),
    }


def _load_private_contact(repo_root: Path) -> _PrivateSecContact:
    try:
        expected = _contact_fingerprint(repo_root)
        config_path = _require_owned_regular_file(
            repo_root / Path(PRIVATE_CONFIG_PATH), repo_root
        )
        raw = _read_private_config_bytes(config_path)
    except SecGemmaLeanRunnerError as exc:
        raise SecGemmaLeanAcquisitionError(exc.code) from None
    payload: Any = None
    value: Any = None
    contact: _PrivateSecContact | None = None
    try:
        if raw is None:
            raise ValueError("missing private config")
        payload = json.loads(raw.decode("utf-8", errors="strict"))
        if type(payload) is not dict or type(payload.get("secrets")) is not dict:
            raise ValueError("private config schema")
        value = payload["secrets"].get("sec_user_agent")
        contact = _PrivateSecContact(value)
        if contact.sha256 != expected:
            raise ValueError("private contact changed")
        return contact
    except SecGemmaLeanAcquisitionError:
        raise
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "private_contact_invalid"
        ) from None
    finally:
        raw = b""
        payload = None
        value = None


def _build_sec_session() -> Any:
    session = requests.Session()
    if type(session) is not requests.Session:
        raise SecGemmaLeanAcquisitionError("sec_session_invalid")
    session.trust_env = False
    session.auth = None
    session.cookies.clear()
    session.headers.clear()
    session.proxies.clear()
    adapter = requests.adapters.HTTPAdapter(max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if (
        session.trust_env is not False
        or session.auth is not None
        or len(session.cookies) != 0
        or len(session.headers) != 0
        or session.proxies != {}
        or session.get_adapter("https://") is not adapter
        or session.get_adapter("http://") is not adapter
        or adapter.max_retries.total != 0
    ):
        session.close()
        raise SecGemmaLeanAcquisitionError("sec_session_invalid")
    return session


def _safe_private_stage_root(repo_root: Path) -> Path:
    try:
        return _prepare_checkpoint_root(repo_root)
    except SecGemmaLeanRunnerError as exc:
        raise SecGemmaLeanAcquisitionError(exc.code) from None


def _pid_is_active(pid: int) -> bool:
    if type(pid) is not int or pid <= 0:
        raise SecGemmaLeanAcquisitionError("stale_lock_invalid")
    if pid == os.getpid():
        return True
    if os.name == "nt":
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            error = ctypes.get_last_error()
            if error == 5:  # Access denied still proves a live process exists.
                return True
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise SecGemmaLeanAcquisitionError("stale_lock_check_failed")
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        raise SecGemmaLeanAcquisitionError("stale_lock_check_failed") from None
    return True


def _recover_stale_acquisition_lock(lock_path: Path) -> bool:
    try:
        details = lock_path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise SecGemmaLeanAcquisitionError("stale_lock_invalid") from None
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
        or not 1 <= details.st_size <= 1024
    ):
        raise SecGemmaLeanAcquisitionError("stale_lock_invalid")
    try:
        payload = lock_path.read_bytes()
        value = json.loads(payload.decode("ascii", errors="strict"))
        current = lock_path.stat()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanAcquisitionError("stale_lock_invalid") from None
    expected_payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        if type(value) is dict
        else b""
    )
    if (
        payload != expected_payload
        or set(value) != {"schema_version", "pid", "nonce_sha256"}
        or value["schema_version"] != "sec-gemma-lean-v3-local-lease-v1"
        or type(value["pid"]) is not int
        or value["pid"] <= 0
        or type(value["nonce_sha256"]) is not str
        or _SHA256_RE.fullmatch(value["nonce_sha256"]) is None
        or (details.st_dev, details.st_ino, details.st_size)
        != (current.st_dev, current.st_ino, current.st_size)
    ):
        raise SecGemmaLeanAcquisitionError("stale_lock_invalid")
    if _pid_is_active(value["pid"]):
        return False
    try:
        unchanged = lock_path.lstat()
        if (
            unchanged.st_dev,
            unchanged.st_ino,
            unchanged.st_size,
        ) != (details.st_dev, details.st_ino, details.st_size):
            raise SecGemmaLeanAcquisitionError("stale_lock_changed")
        lock_path.unlink()
    except SecGemmaLeanAcquisitionError:
        raise
    except OSError:
        raise SecGemmaLeanAcquisitionError("stale_lock_cleanup_failed") from None
    return True


def _write_exclusive(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes:
        raise SecGemmaLeanAcquisitionError("checkpoint_value_invalid")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_target_exists"
        ) from None
    except OSError:
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_write_failed"
        ) from None


def _encode_external_tree(
    value: Any,
    *,
    blob_root: Path,
    inventory: dict[str, int],
    depth: int = 0,
) -> dict[str, Any]:
    if depth > 96:
        raise SecGemmaLeanAcquisitionError("checkpoint_tree_too_deep")
    if type(value) is bytes:
        digest = _sha256_bytes(value)
        target = blob_root / f"{digest}.bin"
        prior = inventory.get(digest)
        if prior is None:
            _write_exclusive(target, value)
            inventory[digest] = len(value)
        elif prior != len(value):
            raise SecGemmaLeanAcquisitionError("checkpoint_hash_collision")
        return {
            "node": "bytes",
            "sha256": digest,
            "byte_count": len(value),
        }
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise SecGemmaLeanAcquisitionError("checkpoint_value_invalid")
        return {
            "node": "dict",
            "items": [
                [
                    key,
                    _encode_external_tree(
                        value[key],
                        blob_root=blob_root,
                        inventory=inventory,
                        depth=depth + 1,
                    ),
                ]
                for key in sorted(value)
            ],
        }
    if type(value) in {list, tuple}:
        return {
            "node": "list",
            "items": [
                _encode_external_tree(
                    item,
                    blob_root=blob_root,
                    inventory=inventory,
                    depth=depth + 1,
                )
                for item in value
            ],
        }
    if value is None or type(value) in {bool, int, str}:
        return {"node": "value", "value": value}
    if type(value) is float and math.isfinite(value):
        return {"node": "value", "value": value}
    raise SecGemmaLeanAcquisitionError("checkpoint_value_invalid")


def _regular_single_link_file(path: Path) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError:
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_file_invalid"
        ) from None
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_file_invalid")
    return details


def _owned_file_contains(path: Path, needle: bytes) -> bool:
    if type(needle) is not bytes or not needle:
        raise SecGemmaLeanAcquisitionError("private_contact_invalid")
    before = _regular_single_link_file(path)
    descriptor: int | None = None
    overlap = b""
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise SecGemmaLeanAcquisitionError("checkpoint_file_changed")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            candidate = overlap + chunk
            if needle in candidate:
                return True
            keep = len(needle) - 1
            overlap = candidate[-keep:] if keep else b""
        after = path.stat()
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise SecGemmaLeanAcquisitionError("checkpoint_file_changed")
        return False
    except SecGemmaLeanAcquisitionError:
        raise
    except OSError:
        raise SecGemmaLeanAcquisitionError("checkpoint_file_invalid") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _scan_completed_checkpoint_for_contact(
    stage_root: Path,
    checkpoint: Mapping[str, Any],
    contact: _PrivateSecContact,
) -> None:
    inventory = checkpoint.get("tree", {}).get("blob_inventory")
    if type(inventory) is not list:
        raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
    paths = [
        stage_root / PRIVATE_MANIFEST_NAME,
        stage_root / PRIVATE_TREE_NAME,
        *[
            stage_root / Path(item["path"])
            for item in inventory
            if type(item) is dict and type(item.get("path")) is str
        ],
    ]
    if len(paths) != len(inventory) + 2:
        raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
    for needle in contact.serialized_echo_needles():
        if any(_owned_file_contains(path, needle) for path in paths):
            raise SecGemmaLeanAcquisitionError("private_contact_leaked")


def _assert_value_redacted(value: Any, contact: _PrivateSecContact) -> None:
    """Reject a readable or JSON-escaped SEC identity before any byte is written."""
    needles = contact.serialized_echo_needles()
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    while pending:
        current, depth = pending.pop()
        if depth > 96:
            raise SecGemmaLeanAcquisitionError("checkpoint_tree_too_deep")
        if type(current) is bytes:
            candidates = (current,)
        elif type(current) is str:
            try:
                candidates = (
                    current.encode("utf-8"),
                    json.dumps(
                        current,
                        ensure_ascii=True,
                        allow_nan=False,
                    ).encode("ascii"),
                )
            except (UnicodeEncodeError, ValueError):
                raise SecGemmaLeanAcquisitionError(
                    "checkpoint_value_invalid"
                ) from None
        else:
            candidates = ()
        if any(needle in candidate for needle in needles for candidate in candidates):
            raise SecGemmaLeanAcquisitionError("private_contact_leaked")
        if type(current) is dict:
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            for key, child in current.items():
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))
        elif type(current) in {list, tuple}:
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            pending.extend((child, depth + 1) for child in current)


_ATTEMPT_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "attempt_started",
        "attempt_recovered_after_crash",
        "network_request_intent",
        "network_request_settled",
        "source_unit_sealed",
        "attempt_finished",
        "attempt_interrupted",
        "terminal_outcome",
    }
)


class _AttemptJournal:
    """Append-only, redacted record of every effectful acquisition attempt."""

    __slots__ = (
        "_active_attempt",
        "_authority",
        "_authority_sha256",
        "_contact",
        "_events",
        "_root",
    )

    def __init__(
        self,
        repo_root: Path,
        *,
        readiness: Mapping[str, Any],
        plan: Mapping[str, Any],
        contact: _PrivateSecContact,
    ) -> None:
        self._contact = contact
        self._authority = {
            "repository": dict(readiness),
            "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
            "sec_contact_sha256": contact.sha256,
            "acquisition_schema_version": SCHEMA_VERSION,
            "scientific_parent_commit": SCIENTIFIC_PARENT_COMMIT,
            "preregistration_commit": PREREGISTRATION_COMMIT,
        }
        self._authority_sha256 = canonical_sha256(self._authority)
        checkpoint_root = _safe_private_stage_root(repo_root)
        self._root = checkpoint_root / PRIVATE_ATTEMPT_JOURNAL_DIRECTORY
        try:
            self._root.mkdir(exist_ok=True)
            details = self._root.lstat()
        except OSError:
            raise SecGemmaLeanAcquisitionError("attempt_journal_invalid") from None
        if (
            not stat.S_ISDIR(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or _is_reparse(details)
        ):
            raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
        self._events = self._load_events()
        self._active_attempt: int | None = None
        if self._events:
            first = self._events[0]
            if (
                first["event_type"] != "attempt_started"
                or first["payload"].get("authority") != self._authority
                or any(
                    event["authority_sha256"] != self._authority_sha256
                    for event in self._events
                )
            ):
                raise SecGemmaLeanAcquisitionError(
                    "attempt_journal_authority_changed"
                )

    def _load_events(self) -> list[dict[str, Any]]:
        try:
            entries = list(self._root.iterdir())
        except OSError:
            raise SecGemmaLeanAcquisitionError("attempt_journal_invalid") from None
        event_paths: list[Path] = []
        for path in entries:
            if path.name.startswith(".attempt-event-") and path.name.endswith(
                ".tmp"
            ):
                continue
            if re.fullmatch(r"[0-9]{8}\.json", path.name) is None:
                raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
            event_paths.append(path)
        event_paths.sort(key=lambda path: path.name)
        events: list[dict[str, Any]] = []
        previous: str | None = None
        active_attempt: int | None = None
        greatest_attempt = 0
        request_intents: dict[str, int] = {}
        settled_intents: set[str] = set()
        for expected_sequence, path in enumerate(event_paths, start=1):
            details = _regular_single_link_file(path)
            if details.st_size < 2 or details.st_size > 1024 * 1024:
                raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
            try:
                raw = path.read_bytes()
                event = json.loads(raw.decode("utf-8", errors="strict"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise SecGemmaLeanAcquisitionError(
                    "attempt_journal_invalid"
                ) from None
            body = (
                {
                    key: value
                    for key, value in event.items()
                    if key != "event_sha256"
                }
                if type(event) is dict
                else {}
            )
            if (
                type(event) is not dict
                or raw != _plain_json_bytes(event)
                or set(event)
                != {
                    "schema_version",
                    "sequence_number",
                    "previous_event_sha256",
                    "authority_sha256",
                    "attempt_number",
                    "event_type",
                    "occurred_at_utc",
                    "payload",
                    "event_sha256",
                }
                or event["schema_version"] != ATTEMPT_EVENT_SCHEMA_VERSION
                or event["sequence_number"] != expected_sequence
                or path.name != f"{expected_sequence:08d}.json"
                or event["previous_event_sha256"] != previous
                or type(event["authority_sha256"]) is not str
                or _SHA256_RE.fullmatch(event["authority_sha256"]) is None
                or type(event["attempt_number"]) is not int
                or event["attempt_number"] <= 0
                or event["event_type"] not in _ATTEMPT_EVENT_TYPES
                or type(event["payload"]) is not dict
                or event["event_sha256"] != canonical_sha256(body)
            ):
                raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
            _parse_utc_timestamp(event["occurred_at_utc"])
            event_type = event["event_type"]
            attempt_number = event["attempt_number"]
            payload = event["payload"]
            semantic_valid = False
            if event_type == "attempt_started":
                semantic_valid = (
                    active_attempt is None
                    and attempt_number == greatest_attempt + 1
                    and set(payload) == {"authority"}
                    and type(payload["authority"]) is dict
                )
                if semantic_valid:
                    active_attempt = attempt_number
                    greatest_attempt = attempt_number
            elif event_type == "attempt_recovered_after_crash":
                semantic_valid = (
                    active_attempt == attempt_number
                    and set(payload) == {"disposition", "prior_event_sha256"}
                    and payload["disposition"] == "crash_indeterminate"
                    and type(payload["prior_event_sha256"]) is str
                    and _SHA256_RE.fullmatch(payload["prior_event_sha256"])
                    is not None
                    and payload["prior_event_sha256"] == previous
                )
                if semantic_valid:
                    active_attempt = None
            elif event_type == "network_request_intent":
                semantic_valid = (
                    active_attempt == attempt_number
                    and set(payload)
                    == {"provider", "logical_purpose", "request_key_sha256"}
                    and payload["provider"] in {"sec", "yahoo"}
                    and type(payload["logical_purpose"]) is str
                    and re.fullmatch(
                        r"[a-z0-9_]{1,80}", payload["logical_purpose"]
                    )
                    is not None
                    and type(payload["request_key_sha256"]) is str
                    and _SHA256_RE.fullmatch(payload["request_key_sha256"])
                    is not None
                )
                if semantic_valid:
                    request_intents[event["event_sha256"]] = attempt_number
            elif event_type == "network_request_settled":
                intent = payload.get("intent_event_sha256")
                status_code = payload.get("status_code")
                semantic_valid = (
                    active_attempt == attempt_number
                    and set(payload)
                    == {"intent_event_sha256", "outcome", "status_code"}
                    and type(intent) is str
                    and request_intents.get(intent) == attempt_number
                    and intent not in settled_intents
                    and payload["outcome"]
                    in {"response_started", "transport_error", "interrupted"}
                    and (
                        status_code is None
                        or (
                            type(status_code) is int
                            and 100 <= status_code <= 599
                        )
                    )
                )
                if semantic_valid:
                    settled_intents.add(intent)
            elif event_type == "source_unit_sealed":
                semantic_valid = (
                    active_attempt == attempt_number
                    and set(payload)
                    == {
                        "provider",
                        "unit",
                        "request_count",
                        "response_bytes",
                        "commitment_sha256",
                    }
                    and payload["provider"] in {"sec", "yahoo"}
                    and type(payload["unit"]) is str
                    and re.fullmatch(r"[a-z0-9_]{1,80}", payload["unit"])
                    is not None
                    and type(payload["request_count"]) is int
                    and payload["request_count"] >= 0
                    and type(payload["response_bytes"]) is int
                    and payload["response_bytes"] >= 0
                    and type(payload["commitment_sha256"]) is str
                    and _SHA256_RE.fullmatch(payload["commitment_sha256"])
                    is not None
                )
            elif event_type == "attempt_finished":
                semantic_valid = (
                    active_attempt == attempt_number
                    and set(payload) == {"disposition", "code"}
                    and payload["disposition"]
                    in {
                        "resumable_transient",
                        "terminal_suitability_rejection",
                        "terminal_integrity_rejection",
                        "blocked_indeterminate",
                    }
                    and type(payload["code"]) is str
                    and re.fullmatch(r"[a-z0-9_]{1,100}", payload["code"])
                    is not None
                )
                if semantic_valid:
                    active_attempt = None
            elif event_type == "attempt_interrupted":
                semantic_valid = (
                    active_attempt == attempt_number
                    and payload == {"disposition": "resumable_interrupted"}
                )
                if semantic_valid:
                    active_attempt = None
            elif event_type == "terminal_outcome":
                effect_counts = payload.get("effect_counts")
                if payload.get("status") == "passed":
                    semantic_valid = (
                        active_attempt == attempt_number
                        and set(payload)
                        == {
                            "status",
                            "checkpoint_sha256",
                            "bundle_sha256",
                            "effect_counts",
                        }
                        and type(payload["checkpoint_sha256"]) is str
                        and _SHA256_RE.fullmatch(payload["checkpoint_sha256"])
                        is not None
                        and type(payload["bundle_sha256"]) is str
                        and _SHA256_RE.fullmatch(payload["bundle_sha256"])
                        is not None
                        and type(effect_counts) is dict
                    )
                elif payload.get("status") == "rejected":
                    semantic_valid = (
                        active_attempt == attempt_number
                        and set(payload)
                        == {
                            "status",
                            "disposition",
                            "code",
                            "evidence_commitments",
                        }
                        and payload["disposition"]
                        in {
                            "terminal_suitability_rejection",
                            "terminal_integrity_rejection",
                        }
                        and type(payload["code"]) is str
                        and re.fullmatch(r"[a-z0-9_]{1,100}", payload["code"])
                        is not None
                        and type(payload["evidence_commitments"]) is dict
                    )
                if semantic_valid:
                    active_attempt = None
            if not semantic_valid:
                raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
            events.append(event)
            previous = event["event_sha256"]
        terminal_positions = [
            index
            for index, event in enumerate(events)
            if event["event_type"] == "terminal_outcome"
        ]
        if terminal_positions and terminal_positions != [len(events) - 1]:
            raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
        return events

    def _append(
        self,
        event_type: str,
        *,
        attempt_number: int,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            event_type not in _ATTEMPT_EVENT_TYPES
            or type(attempt_number) is not int
            or attempt_number <= 0
            or type(payload) is not dict
        ):
            raise SecGemmaLeanAcquisitionError("attempt_event_invalid")
        sequence = len(self._events) + 1
        body = {
            "schema_version": ATTEMPT_EVENT_SCHEMA_VERSION,
            "sequence_number": sequence,
            "previous_event_sha256": (
                None if not self._events else self._events[-1]["event_sha256"]
            ),
            "authority_sha256": self._authority_sha256,
            "attempt_number": attempt_number,
            "event_type": event_type,
            "occurred_at_utc": _utc_now(),
            "payload": dict(payload),
        }
        event = {**body, "event_sha256": canonical_sha256(body)}
        _assert_value_redacted(event, self._contact)
        encoded = _plain_json_bytes(event)
        if any(
            needle in encoded for needle in self._contact.serialized_echo_needles()
        ):
            raise SecGemmaLeanAcquisitionError("private_contact_leaked")
        target = self._root / f"{sequence:08d}.json"
        temporary = self._root / f".attempt-event-{uuid.uuid4().hex}.tmp"
        try:
            _write_exclusive(temporary, encoded)
            os.rename(temporary, target)
            if target.read_bytes() != encoded:
                raise OSError("attempt event bytes changed")
        except SecGemmaLeanAcquisitionError:
            temporary.unlink(missing_ok=True)
            raise
        except OSError:
            temporary.unlink(missing_ok=True)
            raise SecGemmaLeanAcquisitionError(
                "attempt_journal_write_failed"
            ) from None
        self._events.append(event)
        return event

    def terminal_event(self) -> dict[str, Any] | None:
        if self._events and self._events[-1]["event_type"] == "terminal_outcome":
            return dict(self._events[-1])
        return None

    def begin_attempt(self) -> int:
        if self.terminal_event() is not None:
            raise SecGemmaLeanAcquisitionError("attempt_journal_terminal")
        attempt_number = max(
            (event["attempt_number"] for event in self._events),
            default=0,
        )
        if attempt_number:
            prior_events = [
                event
                for event in self._events
                if event["attempt_number"] == attempt_number
            ]
            if prior_events[-1]["event_type"] not in {
                "attempt_finished",
                "attempt_interrupted",
                "attempt_recovered_after_crash",
                "terminal_outcome",
            }:
                self._append(
                    "attempt_recovered_after_crash",
                    attempt_number=attempt_number,
                    payload={
                        "disposition": "crash_indeterminate",
                        "prior_event_sha256": prior_events[-1]["event_sha256"],
                    },
                )
        attempt_number += 1
        self._append(
            "attempt_started",
            attempt_number=attempt_number,
            payload={"authority": self._authority},
        )
        self._active_attempt = attempt_number
        return attempt_number

    def request_intent(
        self,
        *,
        provider: str,
        logical_purpose: str,
        request_key_sha256: str,
    ) -> str:
        if (
            self._active_attempt is None
            or provider not in {"sec", "yahoo"}
            or re.fullmatch(r"[a-z0-9_]{1,80}", logical_purpose) is None
            or _SHA256_RE.fullmatch(request_key_sha256) is None
        ):
            raise SecGemmaLeanAcquisitionError("attempt_event_invalid")
        prior_provider_intents = sum(
            event["event_type"] == "network_request_intent"
            and event["payload"]["provider"] == provider
            for event in self._events
        )
        provider_cap = (
            SEC_REQUEST_CAP if provider == "sec" else LIFETIME_MARKET_REQUEST_CAP
        )
        if prior_provider_intents >= provider_cap:
            raise SecGemmaLeanAcquisitionError(
                f"lifetime_{provider}_request_cap_exceeded"
            )
        event = self._append(
            "network_request_intent",
            attempt_number=self._active_attempt,
            payload={
                "provider": provider,
                "logical_purpose": logical_purpose,
                "request_key_sha256": request_key_sha256,
            },
        )
        return event["event_sha256"]

    def request_settled(
        self,
        *,
        intent_event_sha256: str,
        outcome: str,
        status_code: int | None,
    ) -> None:
        if (
            self._active_attempt is None
            or _SHA256_RE.fullmatch(intent_event_sha256) is None
            or outcome
            not in {"response_started", "transport_error", "interrupted"}
            or (
                status_code is not None
                and (type(status_code) is not int or not 100 <= status_code <= 599)
            )
        ):
            raise SecGemmaLeanAcquisitionError("attempt_event_invalid")
        self._append(
            "network_request_settled",
            attempt_number=self._active_attempt,
            payload={
                "intent_event_sha256": intent_event_sha256,
                "outcome": outcome,
                "status_code": status_code,
            },
        )

    def source_unit_sealed(
        self,
        *,
        provider: str,
        unit: str,
        request_count: int,
        response_bytes: int,
        commitment_sha256: str,
    ) -> None:
        if (
            self._active_attempt is None
            or provider not in {"sec", "yahoo"}
            or re.fullmatch(r"[a-z0-9_]{1,80}", unit) is None
            or type(request_count) is not int
            or request_count < 0
            or type(response_bytes) is not int
            or response_bytes < 0
            or _SHA256_RE.fullmatch(commitment_sha256) is None
        ):
            raise SecGemmaLeanAcquisitionError("attempt_event_invalid")
        self._append(
            "source_unit_sealed",
            attempt_number=self._active_attempt,
            payload={
                "provider": provider,
                "unit": unit,
                "request_count": request_count,
                "response_bytes": response_bytes,
                "commitment_sha256": commitment_sha256,
            },
        )

    def has_source_commitment(self, commitment_sha256: str) -> bool:
        if type(commitment_sha256) is not str:
            return False
        return any(
            event["event_type"] == "source_unit_sealed"
            and event["payload"]["commitment_sha256"] == commitment_sha256
            for event in self._events
        )

    def finish_attempt(self, *, disposition: str, code: str) -> None:
        if (
            self._active_attempt is None
            or disposition
            not in {
                "resumable_transient",
                "terminal_suitability_rejection",
                "terminal_integrity_rejection",
                "blocked_indeterminate",
            }
            or re.fullmatch(r"[a-z0-9_]{1,100}", code) is None
        ):
            raise SecGemmaLeanAcquisitionError("attempt_event_invalid")
        self._append(
            "attempt_finished",
            attempt_number=self._active_attempt,
            payload={"disposition": disposition, "code": code},
        )
        self._active_attempt = None

    def interrupt_attempt(self) -> None:
        if self._active_attempt is None:
            raise SecGemmaLeanAcquisitionError("attempt_event_invalid")
        self._append(
            "attempt_interrupted",
            attempt_number=self._active_attempt,
            payload={"disposition": "resumable_interrupted"},
        )
        self._active_attempt = None

    def seal_terminal_success(self, checkpoint: Mapping[str, Any]) -> None:
        if self._active_attempt is None:
            raise SecGemmaLeanAcquisitionError("attempt_event_invalid")
        self._append(
            "terminal_outcome",
            attempt_number=self._active_attempt,
            payload={
                "status": "passed",
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "bundle_sha256": checkpoint["bundle_sha256"],
                "effect_counts": dict(checkpoint["effect_counts"]),
            },
        )
        self._active_attempt = None

    def seal_terminal_failure(
        self,
        *,
        disposition: str,
        code: str,
        evidence_commitments: Mapping[str, Any],
    ) -> None:
        if (
            self._active_attempt is None
            or disposition
            not in {
                "terminal_suitability_rejection",
                "terminal_integrity_rejection",
            }
            or re.fullmatch(r"[a-z0-9_]{1,100}", code) is None
            or type(evidence_commitments) is not dict
        ):
            raise SecGemmaLeanAcquisitionError("attempt_event_invalid")
        self._append(
            "terminal_outcome",
            attempt_number=self._active_attempt,
            payload={
                "status": "rejected",
                "disposition": disposition,
                "code": code,
                "evidence_commitments": dict(evidence_commitments),
            },
        )
        self._active_attempt = None

    def summary(self) -> dict[str, Any]:
        intents = {
            event["event_sha256"]: event
            for event in self._events
            if event["event_type"] == "network_request_intent"
        }
        settlements = {
            event["payload"]["intent_event_sha256"]: event
            for event in self._events
            if event["event_type"] == "network_request_settled"
        }
        if not set(settlements).issubset(intents):
            raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
        providers: dict[str, dict[str, Any]] = {}
        for provider in ("sec", "yahoo"):
            provider_intents = {
                digest
                for digest, event in intents.items()
                if event["payload"]["provider"] == provider
            }
            confirmed = sum(
                settlements[digest]["payload"]["outcome"]
                == "response_started"
                for digest in provider_intents & set(settlements)
            )
            ambiguous_settlements = sum(
                settlements[digest]["payload"]["outcome"]
                in {"transport_error", "interrupted"}
                for digest in provider_intents & set(settlements)
            )
            unsettled = len(provider_intents - set(settlements))
            indeterminate = ambiguous_settlements + unsettled
            sealed_events = [
                event
                for event in self._events
                if event["event_type"] == "source_unit_sealed"
                and event["payload"]["provider"] == provider
            ]
            providers[provider] = {
                "known_network_requests": confirmed,
                "confirmed_response_started_requests": confirmed,
                "ambiguous_settled_request_intents": ambiguous_settlements,
                "unsettled_request_intents": unsettled,
                "indeterminate_request_intents": indeterminate,
                "exact_network_request_count": (
                    confirmed if indeterminate == 0 else None
                ),
                "network_request_count_lower_bound": confirmed,
                "network_request_count_upper_bound": confirmed
                + indeterminate,
                "validated_source_unit_requests": sum(
                    event["payload"]["request_count"]
                    for event in sealed_events
                ),
                "validated_source_unit_bytes": sum(
                    event["payload"]["response_bytes"]
                    for event in sealed_events
                ),
            }
        return {
            "schema_version": ATTEMPT_EVENT_SCHEMA_VERSION,
            "journal_path": (
                f"{CHECKPOINT_ROOT}/{PRIVATE_ATTEMPT_JOURNAL_DIRECTORY}"
            ),
            "ignored_local_state": True,
            "authority_sha256": self._authority_sha256,
            "event_count": len(self._events),
            "head_event_sha256": (
                None if not self._events else self._events[-1]["event_sha256"]
            ),
            "attempt_count": max(
                (event["attempt_number"] for event in self._events),
                default=0,
            ),
            "lifetime_network_accounting": providers,
            "terminal_outcome_recorded": self.terminal_event() is not None,
        }


def _install_sec_request_journal(
    session: Any,
    *,
    journal: _AttemptJournal,
    logical_purpose: str,
) -> None:
    if type(session) is not requests.Session:
        raise SecGemmaLeanAcquisitionError("sec_transport_invalid")
    original_request = session.request

    def journaled_request(
        method: str,
        url: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if method != "GET" or type(url) is not str:
            raise SecGemmaLeanAcquisitionError("sec_transport_invalid")
        intent = journal.request_intent(
            provider="sec",
            logical_purpose=logical_purpose,
            request_key_sha256=_sha256_bytes(url.encode("utf-8")),
        )
        try:
            response = original_request(method, url, *args, **kwargs)
        except BaseException as exc:
            journal.request_settled(
                intent_event_sha256=intent,
                outcome=(
                    "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "transport_error"
                ),
                status_code=None,
            )
            raise
        try:
            status = int(response.status_code)
            if not 100 <= status <= 599:
                raise ValueError("invalid status")
            journal.request_settled(
                intent_event_sha256=intent,
                outcome="response_started",
                status_code=status,
            )
        except BaseException:
            try:
                response.close()
            except Exception:
                pass
            raise
        return response

    session.request = journaled_request


class _JournaledYahooOpener:
    __slots__ = ("_inner", "_journal")

    def __init__(self, inner: Any, journal: _AttemptJournal) -> None:
        self._inner = inner
        self._journal = journal

    def open(self, request: Any, *, timeout: float) -> Any:
        url = getattr(request, "full_url", None)
        if type(url) is not str:
            raise SecGemmaLeanAcquisitionError("market_transport_invalid")
        intent = self._journal.request_intent(
            provider="yahoo",
            logical_purpose="development_market_chart",
            request_key_sha256=_sha256_bytes(url.encode("utf-8")),
        )
        try:
            response = self._inner.open(request, timeout=timeout)
        except BaseException as exc:
            self._journal.request_settled(
                intent_event_sha256=intent,
                outcome=(
                    "interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "transport_error"
                ),
                status_code=None,
            )
            raise
        try:
            raw_status = getattr(response, "status", None)
            status = int(raw_status if raw_status is not None else response.getcode())
            if not 100 <= status <= 599:
                raise ValueError("invalid status")
            self._journal.request_settled(
                intent_event_sha256=intent,
                outcome="response_started",
                status_code=status,
            )
        except BaseException:
            try:
                response.close()
            except Exception:
                pass
            raise
        return response


def _remove_generated_staging(checkpoint_root: Path, staging: Path) -> None:
    """Remove only one verified, generated tree-staging directory."""
    if not staging.exists():
        return
    try:
        root = checkpoint_root.resolve(strict=True)
        parent = staging.parent.resolve(strict=True)
        details = staging.lstat()
    except OSError:
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_staging_cleanup_failed"
        ) from None
    if (
        parent != root
        or not staging.name.startswith(".")
        or not staging.name.endswith(".tmp")
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_staging_cleanup_failed")

    def remove_directory(directory: Path) -> None:
        try:
            entries = list(directory.iterdir())
        except OSError:
            raise SecGemmaLeanAcquisitionError(
                "checkpoint_staging_cleanup_failed"
            ) from None
        for entry in entries:
            try:
                item = entry.lstat()
            except OSError:
                raise SecGemmaLeanAcquisitionError(
                    "checkpoint_staging_cleanup_failed"
                ) from None
            if stat.S_ISREG(item.st_mode) and not stat.S_ISLNK(item.st_mode):
                try:
                    entry.unlink()
                except OSError:
                    raise SecGemmaLeanAcquisitionError(
                        "checkpoint_staging_cleanup_failed"
                    ) from None
            elif (
                stat.S_ISDIR(item.st_mode)
                and not stat.S_ISLNK(item.st_mode)
                and not _is_reparse(item)
            ):
                remove_directory(entry)
            else:
                raise SecGemmaLeanAcquisitionError(
                    "checkpoint_staging_cleanup_failed"
                )
        try:
            directory.rmdir()
        except OSError:
            raise SecGemmaLeanAcquisitionError(
                "checkpoint_staging_cleanup_failed"
            ) from None

    remove_directory(staging)


def _seal_external_tree_unit(
    repo_root: Path,
    *,
    directory_name: str,
    value: dict[str, Any],
    manifest_factory: Any,
    contact: _PrivateSecContact,
    publish_error_code: str,
) -> tuple[Path, dict[str, Any]]:
    checkpoint_root = _safe_private_stage_root(repo_root)
    target = checkpoint_root / directory_name
    if target.exists():
        raise SecGemmaLeanAcquisitionError("checkpoint_target_exists")
    _assert_value_redacted(value, contact)
    staging = checkpoint_root / f".{directory_name}-{uuid.uuid4().hex}.tmp"
    promoted = False
    try:
        staging.mkdir()
        blob_root = staging / "blobs"
        blob_root.mkdir()
        inventory: dict[str, int] = {}
        tree = {
            "schema_version": TREE_SCHEMA_VERSION,
            "root": _encode_external_tree(
                value,
                blob_root=blob_root,
                inventory=inventory,
            ),
        }
        tree_bytes = _plain_json_bytes(tree)
        _write_exclusive(staging / PRIVATE_TREE_NAME, tree_bytes)
        manifest = manifest_factory(tree_bytes, inventory)
        _assert_value_redacted(manifest, contact)
        _write_exclusive(
            staging / PRIVATE_MANIFEST_NAME,
            _plain_json_bytes(manifest),
        )
        _scan_completed_checkpoint_for_contact(staging, manifest, contact)
        os.rename(staging, target)
        promoted = True
    except BaseException as exc:
        if not promoted and staging.exists():
            _remove_generated_staging(checkpoint_root, staging)
        if isinstance(exc, SecGemmaLeanAcquisitionError):
            raise
        if isinstance(exc, OSError):
            raise SecGemmaLeanAcquisitionError(publish_error_code) from None
        raise
    return target, manifest


def _decode_external_tree(
    node: Any,
    *,
    blob_root: Path,
    expected_inventory: Mapping[str, int],
    seen: set[str],
    depth: int = 0,
) -> Any:
    if depth > 96 or type(node) is not dict or type(node.get("node")) is not str:
        raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
    kind = node["node"]
    if kind == "bytes":
        if set(node) != {"node", "sha256", "byte_count"}:
            raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
        digest = node["sha256"]
        byte_count = node["byte_count"]
        if (
            type(digest) is not str
            or _SHA256_RE.fullmatch(digest) is None
            or type(byte_count) is not int
            or byte_count < 0
            or expected_inventory.get(digest) != byte_count
        ):
            raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
        path = blob_root / f"{digest}.bin"
        details = _regular_single_link_file(path)
        if details.st_size != byte_count:
            raise SecGemmaLeanAcquisitionError("checkpoint_blob_invalid")
        try:
            value = path.read_bytes()
        except OSError:
            raise SecGemmaLeanAcquisitionError(
                "checkpoint_blob_invalid"
            ) from None
        if len(value) != byte_count or _sha256_bytes(value) != digest:
            raise SecGemmaLeanAcquisitionError("checkpoint_blob_invalid")
        seen.add(digest)
        return value
    if kind == "dict":
        if set(node) != {"node", "items"} or type(node["items"]) is not list:
            raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
        result: dict[str, Any] = {}
        prior: str | None = None
        for pair in node["items"]:
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or pair[0] in result
                or (prior is not None and pair[0] <= prior)
            ):
                raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
            result[pair[0]] = _decode_external_tree(
                pair[1],
                blob_root=blob_root,
                expected_inventory=expected_inventory,
                seen=seen,
                depth=depth + 1,
            )
            prior = pair[0]
        return result
    if kind == "list":
        if set(node) != {"node", "items"} or type(node["items"]) is not list:
            raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
        return [
            _decode_external_tree(
                item,
                blob_root=blob_root,
                expected_inventory=expected_inventory,
                seen=seen,
                depth=depth + 1,
            )
            for item in node["items"]
        ]
    if kind == "value":
        if set(node) != {"node", "value"}:
            raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
        value = node["value"]
        if value is None or type(value) in {bool, int, str}:
            return value
        if type(value) is float and math.isfinite(value):
            return value
    raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")


def _pilot_selection(model_requests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if type(model_requests) not in {list, tuple} or len(model_requests) < 5:
        raise SecGemmaLeanAcquisitionError("model_request_batch_invalid")
    ranked: list[tuple[int, Mapping[str, Any]]] = []
    for ordinal, request in enumerate(model_requests, start=1):
        if type(request) is not dict or type(request.get("request_bytes")) is not bytes:
            raise SecGemmaLeanAcquisitionError("model_request_batch_invalid")
        ranked.append((ordinal, request))
    chosen = sorted(
        ranked,
        key=lambda item: (
            -len(item[1]["request_bytes"]),
            item[1]["accession_number"],
        ),
    )[:5]
    return [
        {
            "ordinal": ordinal,
            "accession_number": request["accession_number"],
            "request_sha256": request["request_sha256"],
            "request_utf8_bytes": len(request["request_bytes"]),
        }
        for ordinal, request in chosen
    ]


def _build_checkpoint_manifest(
    *,
    readiness: Mapping[str, Any],
    plan: Mapping[str, Any],
    bundle: Mapping[str, Any],
    validation: Mapping[str, Any],
    accounting: Mapping[str, Any],
    tree_bytes: bytes,
    inventory: Mapping[str, int],
    pilot_selection: Sequence[Mapping[str, Any]],
    source_resume: Mapping[str, Any],
    contact_sha256: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise SecGemmaLeanAcquisitionError("monotonic_clock_invalid")
    manifest = bundle["public_manifest"]
    stage_artifact = bundle["private_quarantine"]["sec_replay_evidence"][
        "stage_artifact"
    ]
    unusable = [
        item["accession_number"]
        for item in stage_artifact["documents"]
        if item["normalized_text_usable"] is not True
    ]
    if unusable:
        raise SecGemmaLeanAcquisitionError("filing_text_unusable")
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "passed",
        "stage": DEVELOPMENT,
        "started_at_utc": _parse_utc_timestamp(started_at_utc),
        "finished_at_utc": _parse_utc_timestamp(finished_at_utc),
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "repository": dict(readiness),
        "scientific_parent_commit": SCIENTIFIC_PARENT_COMMIT,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preflight_sha256": readiness["preflight_sha256"],
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "sec_contact_sha256": contact_sha256,
        "bundle_sha256": bundle["bundle_sha256"],
        "public_manifest_sha256": manifest["manifest_sha256"],
        "private_index_sha256": manifest["private_index_sha256"],
        "detached_validation": _to_plain_json_value(validation),
        "request_accounting": dict(accounting),
        "tree": {
            "schema_version": TREE_SCHEMA_VERSION,
            "path": PRIVATE_TREE_NAME,
            "sha256": _sha256_bytes(tree_bytes),
            "byte_count": len(tree_bytes),
            "blob_count": len(inventory),
            "blob_byte_count": sum(inventory.values()),
            "blob_inventory": [
                {
                    "sha256": digest,
                    "byte_count": inventory[digest],
                    "path": f"blobs/{digest}.bin",
                }
                for digest in sorted(inventory)
            ],
        },
        "suitability": {
            "complete_development_universe": True,
            "filing_count": manifest["sec_primary_document_count"],
            "minimum_filing_count": plan["sec"]["minimum_filings"],
            "maximum_filing_count": plan["sec"]["maximum_model_requests"],
            "all_filing_text_usable": True,
            "model_request_count": manifest["model_request_count"],
            "market_response_count": manifest["market_response_count"],
            "market_prefixes_replayed": True,
            "detached_sec_bytes_replayed": True,
            "blinded_requests_replayed": True,
        },
        "pilot_selection": list(pilot_selection),
        "source_resume": dict(source_resume),
        "effect_counts": {
            "official_sec_requests": accounting["sec_request_count"],
            "market_data_requests": accounting["market_request_count"],
            "model_generation_calls": 0,
            "performance_results_opened": 0,
            "confirmation_sources_opened": 0,
            "final_sources_opened": 0,
            "paid_api_calls": 0,
            "trading_actions": 0,
        },
        "model_authorized": False,
        "next_step": "five_largest_development_request_gemma_pilot",
    }
    if (
        manifest["sec_primary_document_count"]
        not in range(
            int(plan["sec"]["minimum_filings"]),
            int(plan["sec"]["maximum_model_requests"]) + 1,
        )
        or manifest["model_request_count"]
        != manifest["sec_primary_document_count"]
        or manifest["market_response_count"] != MAX_MARKET_REQUESTS_PER_STAGE
        or accounting["sec_request_count"] > SEC_REQUEST_CAP
        or accounting["market_request_count"] != MAX_MARKET_REQUESTS_PER_STAGE
        or accounting["network_request_count"]
        != accounting["sec_request_count"] + accounting["market_request_count"]
        or accounting["retry_count"] != 0
        or accounting["redirect_count"] != 0
        or accounting["sec_bytes"] > SEC_BYTE_CAP
        or len(pilot_selection) != 5
        or type(source_resume) is not dict
        or type(source_resume.get("resume_contract")) is not dict
        or source_resume.get("source_resume_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in source_resume.items()
                if key != "source_resume_sha256"
            }
        )
        or source_resume.get("resume_contract", {}).get(
            "aggregate_sec_deadline_applied"
        )
        is not False
        or source_resume.get("resume_contract", {}).get(
            "final_evidence_rebuilt_as_exact_frozen_batch"
        )
        is not True
        or _TAGGED_SHA256_RE.fullmatch(contact_sha256) is None
    ):
        raise SecGemmaLeanAcquisitionError("source_suitability_failed")
    return {**body, "checkpoint_sha256": canonical_sha256(body)}


def _seal_private_checkpoint(
    repo_root: Path,
    *,
    bundle: dict[str, Any],
    checkpoint_body_factory: Any,
    contact: _PrivateSecContact,
) -> tuple[Path, dict[str, Any]]:
    return _seal_external_tree_unit(
        repo_root,
        directory_name=PRIVATE_STAGE_DIRECTORY,
        value=bundle,
        manifest_factory=checkpoint_body_factory,
        contact=contact,
        publish_error_code="checkpoint_publish_failed",
    )


def _sec_prefix_accounting(prefix: Mapping[str, Any]) -> dict[str, int]:
    evidence = prefix["sec_replay_evidence"]
    receipts = [
        *evidence["catalog_request_receipts"],
        *evidence["authenticated_stage_request_receipts"],
    ]
    result = {
        "request_count": len(receipts),
        "byte_count": sum(int(item["size_bytes"]) for item in receipts),
        "network_request_count": sum(
            int(item["network_requests"]) for item in receipts
        ),
        "retry_count": sum(int(item["retries"]) for item in receipts),
        "redirect_count": sum(int(item["redirects"]) for item in receipts),
    }
    if (
        result["request_count"] > SEC_REQUEST_CAP
        or result["byte_count"] > SEC_BYTE_CAP
        or result["network_request_count"] != result["request_count"]
        or result["retry_count"] != 0
        or result["redirect_count"] != 0
    ):
        raise SecGemmaLeanAcquisitionError("sec_prefix_accounting_invalid")
    return result


def _tree_claim(
    tree_bytes: bytes,
    inventory: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": TREE_SCHEMA_VERSION,
        "path": PRIVATE_TREE_NAME,
        "sha256": _sha256_bytes(tree_bytes),
        "byte_count": len(tree_bytes),
        "blob_count": len(inventory),
        "blob_byte_count": sum(inventory.values()),
        "blob_inventory": [
            {
                "sha256": digest,
                "byte_count": inventory[digest],
                "path": f"blobs/{digest}.bin",
            }
            for digest in sorted(inventory)
        ],
    }


def _zero_forbidden_effects(*, official_sec_requests: int) -> dict[str, int]:
    if type(official_sec_requests) is not int or official_sec_requests < 0:
        raise SecGemmaLeanAcquisitionError("request_accounting_invalid")
    return {
        "official_sec_requests": official_sec_requests,
        "market_data_requests": 0,
        "model_generation_calls": 0,
        "performance_results_opened": 0,
        "confirmation_sources_opened": 0,
        "final_sources_opened": 0,
        "paid_api_calls": 0,
        "trading_actions": 0,
    }


def _elapsed_from_manifest(manifest: Mapping[str, Any]) -> float:
    try:
        elapsed = float.fromhex(manifest["elapsed_seconds_hex"])
    except Exception:
        raise SecGemmaLeanAcquisitionError("monotonic_clock_invalid") from None
    if (
        not math.isfinite(elapsed)
        or elapsed < 0.0
        or elapsed.hex() != manifest["elapsed_seconds_hex"]
    ):
        raise SecGemmaLeanAcquisitionError("monotonic_clock_invalid")
    return elapsed


def _validate_sec_catalog_state(
    state: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_contact_sha256: str,
) -> dict[str, Any]:
    expected_keys = {
        "catalog_sources",
        "catalog_request_receipts",
        "catalog_artifact",
        "corpus_universe_manifest",
        "selected_records",
        "document_plan",
    }
    if type(state) is not dict or set(state) != expected_keys:
        raise SecGemmaLeanAcquisitionError("sec_catalog_checkpoint_invalid")
    sources = state["catalog_sources"]
    receipts = state["catalog_request_receipts"]
    artifact = state["catalog_artifact"]
    universe = state["corpus_universe_manifest"]
    if (
        type(sources) is not list
        or not sources
        or type(receipts) is not list
        or len(receipts) != len(sources)
        or type(artifact) is not dict
        or type(universe) is not dict
        or type(state["selected_records"]) is not list
        or type(state["document_plan"]) is not list
        or any(
            type(source) is not dict
            or set(source) != {"name", "url", "body"}
            or type(source["name"]) is not str
            or type(source["url"]) is not str
            or type(source["body"]) is not bytes
            for source in sources
        )
        or any(type(receipt) is not dict for receipt in receipts)
    ):
        raise SecGemmaLeanAcquisitionError("sec_catalog_checkpoint_invalid")
    try:
        validation = dict(
            validate_detached_catalog_replay(
                source_payloads=[
                    {"name": source["name"], "payload": source["body"]}
                    for source in sources
                ],
                request_receipts=receipts,
                catalog_artifact=artifact,
                corpus_universe_manifest=universe,
                expected_source_payload_sha256s={
                    source["name"]: _sha256_bytes(source["body"])
                    for source in sources
                },
                expected_request_receipt_sha256s={
                    source["name"]: receipt["request_receipt_sha256"]
                    for source, receipt in zip(sources, receipts, strict=True)
                },
                expected_request_receipts_sha256=artifact[
                    "request_receipts_sha256"
                ],
                expected_catalog_artifact_sha256=artifact[
                    "catalog_artifact_sha256"
                ],
                expected_corpus_universe_sha256=universe["universe_sha256"],
                expected_calendar_artifact_sha256=build_contract_manifest()[
                    "data"
                ]["market"]["calendar"]["calendar_dates_sha256"],
                session_dates=list(EXPECTED_SESSIONS),
            )
        )
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "sec_catalog_checkpoint_invalid"
        ) from None
    selected = [
        record
        for record in universe["records"]
        if record["artifact_stage"] == plan["sec"]["stage_name"]
    ]
    selected.sort(
        key=lambda item: (
            item["availability_session"],
            item["accession_number"],
        )
    )
    document_plan = [
        {
            "accession_number": item["accession_number"],
            "official_url": _official_primary_url(item),
        }
        for item in selected
    ]
    identity_hashes = {
        receipt.get("user_agent_sha256") for receipt in receipts
    }
    if (
        state["selected_records"] != selected
        or state["document_plan"] != document_plan
        or not int(plan["sec"]["minimum_filings"])
        <= len(selected)
        <= int(plan["sec"]["maximum_model_requests"])
        or identity_hashes != {expected_contact_sha256}
        or artifact.get("user_agent_sha256") != expected_contact_sha256
        or _TAGGED_SHA256_RE.fullmatch(expected_contact_sha256) is None
        or any(
            source["url"] != receipt.get("requested_url")
            or source["url"] != receipt.get("final_url")
            for source, receipt in zip(sources, receipts, strict=True)
        )
    ):
        raise SecGemmaLeanAcquisitionError("sec_catalog_checkpoint_invalid")
    return validation


def _catalog_request_accounting(state: Mapping[str, Any]) -> dict[str, int]:
    receipts = state["catalog_request_receipts"]
    result = {
        "request_count": len(receipts),
        "byte_count": sum(int(receipt["size_bytes"]) for receipt in receipts),
        "network_request_count": sum(
            int(receipt["network_requests"]) for receipt in receipts
        ),
        "retry_count": sum(int(receipt["retries"]) for receipt in receipts),
        "redirect_count": sum(
            int(receipt["redirects"]) for receipt in receipts
        ),
    }
    if (
        result["request_count"] > SEC_REQUEST_CAP
        or result["byte_count"] > SEC_BYTE_CAP
        or result["network_request_count"] != result["request_count"]
        or result["retry_count"] != 0
        or result["redirect_count"] != 0
    ):
        raise SecGemmaLeanAcquisitionError("request_accounting_invalid")
    return result


def _build_sec_catalog_manifest(
    *,
    readiness: Mapping[str, Any],
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    validation: Mapping[str, Any],
    tree_bytes: bytes,
    inventory: Mapping[str, int],
    contact_sha256: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise SecGemmaLeanAcquisitionError("monotonic_clock_invalid")
    accounting = _catalog_request_accounting(state)
    body = {
        "schema_version": SEC_CATALOG_SCHEMA_VERSION,
        "status": "passed",
        "stage": DEVELOPMENT,
        "unit": "official_sec_catalog",
        "started_at_utc": _parse_utc_timestamp(started_at_utc),
        "finished_at_utc": _parse_utc_timestamp(finished_at_utc),
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "repository": dict(readiness),
        "preflight_sha256": readiness["preflight_sha256"],
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "sec_contact_sha256": contact_sha256,
        "catalog_artifact_sha256": state["catalog_artifact"][
            "catalog_artifact_sha256"
        ],
        "corpus_universe_sha256": state["corpus_universe_manifest"][
            "universe_sha256"
        ],
        "document_plan_sha256": canonical_sha256(state["document_plan"]),
        "catalog_source_count": len(state["catalog_sources"]),
        "filing_count": len(state["selected_records"]),
        "detached_validation": _to_plain_json_value(validation),
        "request_accounting": accounting,
        "tree": _tree_claim(tree_bytes, inventory),
        "effect_counts": _zero_forbidden_effects(
            official_sec_requests=accounting["request_count"]
        ),
        "runtime_scope": {
            "transport_deadline_seconds_hex": float(MAX_SEC_SECONDS).hex(),
            "deadline_applies_to_this_catalog_unit_only": True,
            "aggregate_sec_deadline_applied": False,
        },
        "complete_catalog": True,
        "model_authorized": False,
        "next_step": "download_development_sec_document_chunks",
    }
    return {**body, "sec_catalog_checkpoint_sha256": canonical_sha256(body)}


def _load_sec_catalog_checkpoint(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, tree_bytes, inventory, state = _load_external_tree_checkpoint(
        repo_root,
        directory_name=PRIVATE_SEC_CATALOG_DIRECTORY,
        expected_schema_version=SEC_CATALOG_SCHEMA_VERSION,
        self_hash_key="sec_catalog_checkpoint_sha256",
    )
    plan = validate_acquisition_plan(
        build_acquisition_plan(DEVELOPMENT),
        expected_stage=DEVELOPMENT,
    )
    try:
        contact_sha256 = state["catalog_artifact"]["user_agent_sha256"]
        validation = _validate_sec_catalog_state(
            state,
            plan=plan,
            expected_contact_sha256=contact_sha256,
        )
        recomputed = _build_sec_catalog_manifest(
            readiness=manifest["repository"],
            plan=plan,
            state=state,
            validation=validation,
            tree_bytes=tree_bytes,
            inventory=inventory,
            contact_sha256=contact_sha256,
            started_at_utc=manifest["started_at_utc"],
            finished_at_utc=manifest["finished_at_utc"],
            elapsed_seconds=_elapsed_from_manifest(manifest),
        )
    except SecGemmaLeanAcquisitionError:
        raise
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "sec_catalog_checkpoint_invalid"
        ) from None
    if recomputed != manifest:
        raise SecGemmaLeanAcquisitionError("sec_catalog_checkpoint_invalid")
    return manifest, state, validation


def _sec_document_chunk_directory(index: int) -> str:
    if type(index) is not int or index < 0 or index > 9999:
        raise SecGemmaLeanAcquisitionError("sec_document_chunk_invalid")
    return f"{PRIVATE_SEC_DOCUMENT_CHUNK_PREFIX}{index:04d}"


def _document_batch_value(
    document_plan: list[dict[str, str]],
    batch: Any,
) -> dict[str, Any]:
    return {
        "document_plan": [dict(item) for item in document_plan],
        "raw_documents": [
            item.raw_primary_document for item in batch.documents
        ],
        "normalized_documents": [item.normalized_text for item in batch.documents],
        "request_receipts_json": batch.request_receipts_json,
        "byte_manifest_json": batch.byte_manifest_json,
    }


def _validate_document_batch_value(
    value: Mapping[str, Any],
    *,
    expected_plan: list[dict[str, str]],
    expected_contact_sha256: str,
) -> Any:
    if (
        type(value) is not dict
        or set(value)
        != {
            "document_plan",
            "raw_documents",
            "normalized_documents",
            "request_receipts_json",
            "byte_manifest_json",
        }
        or value["document_plan"] != expected_plan
        or type(value["raw_documents"]) is not list
        or type(value["normalized_documents"]) is not list
        or type(value["request_receipts_json"]) is not bytes
        or type(value["byte_manifest_json"]) is not bytes
    ):
        raise SecGemmaLeanAcquisitionError("sec_document_chunk_invalid")
    try:
        return _validate_persisted_authenticated_stage_access_batch(
            authenticated_document_plan=expected_plan,
            raw_documents=tuple(value["raw_documents"]),
            normalized_documents=tuple(value["normalized_documents"]),
            request_receipts_json=value["request_receipts_json"],
            byte_manifest_json=value["byte_manifest_json"],
            expected_max_requests=SEC_REQUEST_CAP,
            expected_max_bytes=SEC_BYTE_CAP,
            expected_max_seconds=float(MAX_SEC_SECONDS),
            expected_user_agent_sha256=expected_contact_sha256,
        )
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "sec_document_chunk_invalid"
        ) from None


def _build_sec_document_chunk_manifest(
    *,
    readiness: Mapping[str, Any],
    plan: Mapping[str, Any],
    value: Mapping[str, Any],
    batch: Any,
    chunk_index: int,
    chunk_count: int,
    corpus_universe_sha256: str,
    tree_bytes: bytes,
    inventory: Mapping[str, int],
    contact_sha256: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if (
        not math.isfinite(elapsed_seconds)
        or elapsed_seconds < 0.0
        or type(chunk_count) is not int
        or not 1 <= chunk_count <= 9999
        or not 0 <= chunk_index < chunk_count
    ):
        raise SecGemmaLeanAcquisitionError("sec_document_chunk_invalid")
    document_plan = value["document_plan"]
    if not document_plan or len(batch.documents) != len(document_plan):
        raise SecGemmaLeanAcquisitionError("sec_document_chunk_invalid")
    byte_manifest = json.loads(batch.byte_manifest_json.decode("utf-8"))
    body = {
        "schema_version": SEC_DOCUMENT_CHUNK_SCHEMA_VERSION,
        "status": "passed",
        "stage": DEVELOPMENT,
        "unit": "official_sec_document_chunk",
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "started_at_utc": _parse_utc_timestamp(started_at_utc),
        "finished_at_utc": _parse_utc_timestamp(finished_at_utc),
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "repository": dict(readiness),
        "preflight_sha256": readiness["preflight_sha256"],
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "sec_contact_sha256": contact_sha256,
        "corpus_universe_sha256": corpus_universe_sha256,
        "document_plan_sha256": canonical_sha256(document_plan),
        "first_accession_number": document_plan[0]["accession_number"],
        "last_accession_number": document_plan[-1]["accession_number"],
        "document_count": len(document_plan),
        "request_receipts_sha256": byte_manifest[
            "request_receipts_sha256"
        ],
        "byte_manifest_sha256": byte_manifest["byte_manifest_sha256"],
        "request_accounting": {
            "request_count": len(document_plan),
            "byte_count": sum(
                len(document.raw_primary_document)
                for document in batch.documents
            ),
            "network_request_count": len(document_plan),
            "retry_count": 0,
            "redirect_count": 0,
        },
        "tree": _tree_claim(tree_bytes, inventory),
        "effect_counts": _zero_forbidden_effects(
            official_sec_requests=len(document_plan)
        ),
        "runtime_scope": {
            "transport_deadline_seconds_hex": float(MAX_SEC_SECONDS).hex(),
            "deadline_applies_to_this_document_chunk_only": True,
            "aggregate_sec_deadline_applied": False,
        },
        "complete_chunk": True,
        "model_authorized": False,
        "next_step": "resume_next_sec_document_chunk",
    }
    return {**body, "sec_document_chunk_sha256": canonical_sha256(body)}


def _load_sec_document_chunk(
    repo_root: Path,
    *,
    expected_plan: list[dict[str, str]],
    chunk_index: int,
    chunk_count: int,
    corpus_universe_sha256: str,
) -> tuple[dict[str, Any], Any]:
    directory = _sec_document_chunk_directory(chunk_index)
    manifest, tree_bytes, inventory, value = _load_external_tree_checkpoint(
        repo_root,
        directory_name=directory,
        expected_schema_version=SEC_DOCUMENT_CHUNK_SCHEMA_VERSION,
        self_hash_key="sec_document_chunk_sha256",
    )
    plan = validate_acquisition_plan(
        build_acquisition_plan(DEVELOPMENT),
        expected_stage=DEVELOPMENT,
    )
    try:
        contact_sha256 = manifest["sec_contact_sha256"]
        batch = _validate_document_batch_value(
            value,
            expected_plan=expected_plan,
            expected_contact_sha256=contact_sha256,
        )
        recomputed = _build_sec_document_chunk_manifest(
            readiness=manifest["repository"],
            plan=plan,
            value=value,
            batch=batch,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            corpus_universe_sha256=corpus_universe_sha256,
            tree_bytes=tree_bytes,
            inventory=inventory,
            contact_sha256=contact_sha256,
            started_at_utc=manifest["started_at_utc"],
            finished_at_utc=manifest["finished_at_utc"],
            elapsed_seconds=_elapsed_from_manifest(manifest),
        )
    except SecGemmaLeanAcquisitionError:
        raise
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "sec_document_chunk_invalid"
        ) from None
    if recomputed != manifest:
        raise SecGemmaLeanAcquisitionError("sec_document_chunk_invalid")
    return manifest, batch


def _aggregate_document_batches(
    batches: Sequence[Any],
    *,
    document_plan: list[dict[str, str]],
    contact_sha256: str,
) -> Any:
    if type(batches) not in {list, tuple} or not batches:
        raise SecGemmaLeanAcquisitionError("sec_document_batch_partial")
    raw_documents: list[bytes] = []
    normalized_documents: list[bytes] = []
    receipts: list[dict[str, Any]] = []
    template: dict[str, Any] | None = None
    sequence_number = 0
    for batch in batches:
        try:
            chunk_receipts = json.loads(batch.request_receipts_json.decode("utf-8"))
            chunk_manifest = json.loads(batch.byte_manifest_json.decode("utf-8"))
        except Exception:
            raise SecGemmaLeanAcquisitionError(
                "sec_document_chunk_invalid"
            ) from None
        if template is None:
            template = {
                key: value
                for key, value in chunk_manifest.items()
                if key
                not in {
                    "byte_manifest_sha256",
                    "document_count",
                    "documents",
                    "request_receipts_sha256",
                    "acquisition_request_count",
                    "acquisition_bytes",
                }
            }
        elif any(
            chunk_manifest.get(key) != value for key, value in template.items()
        ):
            raise SecGemmaLeanAcquisitionError("sec_document_chunk_invalid")
        if len(chunk_receipts) != len(batch.documents):
            raise SecGemmaLeanAcquisitionError("sec_document_chunk_invalid")
        for document, original_receipt in zip(
            batch.documents,
            chunk_receipts,
            strict=True,
        ):
            sequence_number += 1
            receipt_body = {
                key: value
                for key, value in original_receipt.items()
                if key != "request_receipt_sha256"
            }
            receipt_body["sequence_number"] = sequence_number
            receipt = {
                **receipt_body,
                "request_receipt_sha256": canonical_sha256(receipt_body),
            }
            receipts.append(receipt)
            raw_documents.append(document.raw_primary_document)
            normalized_documents.append(document.normalized_text)
    if template is None or sequence_number != len(document_plan):
        raise SecGemmaLeanAcquisitionError("sec_document_batch_partial")
    rows = [
        {
            "sequence_number": index,
            "accession_number": planned["accession_number"],
            "official_url": planned["official_url"],
            "raw_document_sha256": _sha256_bytes(raw),
            "raw_document_bytes": len(raw),
            "normalized_document_sha256": _sha256_bytes(normalized),
            "normalized_document_bytes": len(normalized),
            "request_receipt_sha256": receipt["request_receipt_sha256"],
        }
        for index, (planned, raw, normalized, receipt) in enumerate(
            zip(
                document_plan,
                raw_documents,
                normalized_documents,
                receipts,
                strict=True,
            ),
            start=1,
        )
    ]
    manifest_body = {
        **template,
        "document_count": len(document_plan),
        "documents": rows,
        "request_receipts_sha256": canonical_sha256(receipts),
        "acquisition_request_count": len(document_plan),
        "acquisition_bytes": sum(len(value) for value in raw_documents),
    }
    byte_manifest = {
        **manifest_body,
        "byte_manifest_sha256": canonical_sha256(manifest_body),
    }
    try:
        return _validate_persisted_authenticated_stage_access_batch(
            authenticated_document_plan=document_plan,
            raw_documents=tuple(raw_documents),
            normalized_documents=tuple(normalized_documents),
            request_receipts_json=_plain_json_bytes(receipts),
            byte_manifest_json=_plain_json_bytes(byte_manifest),
            expected_max_requests=SEC_REQUEST_CAP,
            expected_max_bytes=SEC_BYTE_CAP,
            expected_max_seconds=float(MAX_SEC_SECONDS),
            expected_user_agent_sha256=contact_sha256,
        )
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "sec_document_batch_partial"
        ) from None


def _build_sec_prefix_manifest(
    *,
    readiness: Mapping[str, Any],
    plan: Mapping[str, Any],
    prefix: Mapping[str, Any],
    validation: Mapping[str, Any],
    tree_bytes: bytes,
    inventory: Mapping[str, int],
    contact_sha256: str,
    started_at_utc: str,
    finished_at_utc: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise SecGemmaLeanAcquisitionError("monotonic_clock_invalid")
    if type(prefix) is not dict or set(prefix) != {
        "sec_replay_evidence",
        "sec_catalog_sources",
        "sec_primary_documents",
    }:
        raise SecGemmaLeanAcquisitionError("sec_prefix_invalid")
    documents = prefix["sec_primary_documents"]
    evidence = prefix["sec_replay_evidence"]
    stage_documents = evidence["stage_artifact"]["documents"]
    identity_hashes = {
        item["user_agent_sha256"]
        for item in (
            *evidence["catalog_request_receipts"],
            *evidence["authenticated_stage_request_receipts"],
        )
    }
    if (
        type(documents) is not list
        or not int(plan["sec"]["minimum_filings"])
        <= len(documents)
        <= int(plan["sec"]["maximum_model_requests"])
        or len(stage_documents) != len(documents)
        or any(
            item.get("normalized_text_usable") is not True
            for item in stage_documents
        )
        or identity_hashes != {contact_sha256}
        or _TAGGED_SHA256_RE.fullmatch(contact_sha256) is None
    ):
        raise SecGemmaLeanAcquisitionError("sec_prefix_suitability_failed")
    accounting = _sec_prefix_accounting(prefix)
    body = {
        "schema_version": SEC_PREFIX_SCHEMA_VERSION,
        "status": "passed",
        "stage": DEVELOPMENT,
        "unit": "official_sec_development_prefix",
        "started_at_utc": _parse_utc_timestamp(started_at_utc),
        "finished_at_utc": _parse_utc_timestamp(finished_at_utc),
        "elapsed_seconds_hex": elapsed_seconds.hex(),
        "repository": dict(readiness),
        "preflight_sha256": readiness["preflight_sha256"],
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "sec_contact_sha256": contact_sha256,
        "catalog_source_count": len(prefix["sec_catalog_sources"]),
        "filing_count": len(documents),
        "detached_validation": _to_plain_json_value(validation),
        "request_accounting": accounting,
        "tree": {
            "schema_version": TREE_SCHEMA_VERSION,
            "path": PRIVATE_TREE_NAME,
            "sha256": _sha256_bytes(tree_bytes),
            "byte_count": len(tree_bytes),
            "blob_count": len(inventory),
            "blob_byte_count": sum(inventory.values()),
            "blob_inventory": [
                {
                    "sha256": digest,
                    "byte_count": inventory[digest],
                    "path": f"blobs/{digest}.bin",
                }
                for digest in sorted(inventory)
            ],
        },
        "effect_counts": {
            "official_sec_requests": accounting["request_count"],
            "market_data_requests": 0,
            "model_generation_calls": 0,
            "performance_results_opened": 0,
            "confirmation_sources_opened": 0,
            "final_sources_opened": 0,
            "paid_api_calls": 0,
            "trading_actions": 0,
        },
        "resume_contract": {
            "catalog_checkpoint": PRIVATE_SEC_CATALOG_DIRECTORY,
            "document_chunk_prefix": PRIVATE_SEC_DOCUMENT_CHUNK_PREFIX,
            "document_chunk_size": SEC_DOCUMENT_CHUNK_SIZE,
            "document_chunk_count": (
                len(documents) + SEC_DOCUMENT_CHUNK_SIZE - 1
            )
            // SEC_DOCUMENT_CHUNK_SIZE,
            "per_unit_transport_deadline_seconds_hex": float(
                MAX_SEC_SECONDS
            ).hex(),
            "aggregate_sec_deadline_applied": False,
            "final_evidence_rebuilt_as_exact_frozen_batch": True,
        },
        "complete_prefix": True,
        "model_authorized": False,
        "next_step": "complete_development_market_source_unit",
    }
    return {**body, "sec_prefix_sha256": canonical_sha256(body)}


def _seal_sec_prefix(
    repo_root: Path,
    *,
    prefix: dict[str, Any],
    manifest_factory: Any,
    contact: _PrivateSecContact,
) -> tuple[Path, dict[str, Any]]:
    return _seal_external_tree_unit(
        repo_root,
        directory_name=PRIVATE_SEC_PREFIX_DIRECTORY,
        value=prefix,
        manifest_factory=manifest_factory,
        contact=contact,
        publish_error_code="sec_prefix_publish_failed",
    )


def _load_external_tree_checkpoint(
    repo_root: Path,
    *,
    directory_name: str,
    expected_schema_version: str,
    self_hash_key: str,
) -> tuple[dict[str, Any], bytes, dict[str, int], dict[str, Any]]:
    root = _safe_private_stage_root(repo_root) / directory_name
    _safe_directory(root, repo_root)
    try:
        root_details = root.lstat()
    except OSError:
        raise SecGemmaLeanAcquisitionError("checkpoint_missing") from None
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or stat.S_ISLNK(root_details.st_mode)
        or _is_reparse(root_details)
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_missing")
    manifest_path = root / PRIVATE_MANIFEST_NAME
    tree_path = root / PRIVATE_TREE_NAME
    _regular_single_link_file(manifest_path)
    tree_details = _regular_single_link_file(tree_path)
    try:
        manifest_bytes = manifest_path.read_bytes()
        tree_bytes = tree_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
        tree = json.loads(tree_bytes.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_manifest_invalid"
        ) from None
    body = {
        key: value for key, value in manifest.items() if key != self_hash_key
    } if type(manifest) is dict else {}
    tree_claim = manifest.get("tree") if type(manifest) is dict else None
    if (
        type(manifest) is not dict
        or manifest_bytes != _plain_json_bytes(manifest)
        or manifest.get("schema_version") != expected_schema_version
        or manifest.get("status") != "passed"
        or manifest.get("stage") != DEVELOPMENT
        or manifest.get(self_hash_key) != canonical_sha256(body)
        or type(tree) is not dict
        or tree_bytes != _plain_json_bytes(tree)
        or set(tree) != {"schema_version", "root"}
        or tree["schema_version"] != TREE_SCHEMA_VERSION
        or type(tree_claim) is not dict
        or tree_claim.get("path") != PRIVATE_TREE_NAME
        or tree_claim.get("sha256") != _sha256_bytes(tree_bytes)
        or tree_claim.get("byte_count") != len(tree_bytes)
        or tree_details.st_size != len(tree_bytes)
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
    raw_inventory = tree_claim.get("blob_inventory")
    if type(raw_inventory) is not list:
        raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
    inventory: dict[str, int] = {}
    expected_paths: set[str] = set()
    for row in raw_inventory:
        if (
            type(row) is not dict
            or set(row) != {"sha256", "byte_count", "path"}
            or type(row.get("sha256")) is not str
            or _SHA256_RE.fullmatch(row["sha256"]) is None
            or type(row.get("byte_count")) is not int
            or row["byte_count"] < 0
            or row.get("path") != f"blobs/{row['sha256']}.bin"
            or row["sha256"] in inventory
        ):
            raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
        inventory[row["sha256"]] = row["byte_count"]
        expected_paths.add(row["path"])
    blob_root = root / "blobs"
    try:
        blob_root_details = blob_root.lstat()
        if (
            not stat.S_ISDIR(blob_root_details.st_mode)
            or stat.S_ISLNK(blob_root_details.st_mode)
            or _is_reparse(blob_root_details)
        ):
            raise OSError("blob root invalid")
        observed_paths: set[str] = set()
        for item in blob_root.iterdir():
            _regular_single_link_file(item)
            observed_paths.add(item.relative_to(root).as_posix())
    except OSError:
        raise SecGemmaLeanAcquisitionError("checkpoint_blob_invalid") from None
    if (
        observed_paths != expected_paths
        or tree_claim.get("blob_count") != len(inventory)
        or tree_claim.get("blob_byte_count") != sum(inventory.values())
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_blob_invalid")
    seen: set[str] = set()
    value = _decode_external_tree(
        tree["root"],
        blob_root=blob_root,
        expected_inventory=inventory,
        seen=seen,
    )
    if type(value) is not dict or seen != set(inventory):
        raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
    return manifest, tree_bytes, inventory, value


def _load_sec_prefix(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, tree_bytes, inventory, prefix = _load_external_tree_checkpoint(
        repo_root,
        directory_name=PRIVATE_SEC_PREFIX_DIRECTORY,
        expected_schema_version=SEC_PREFIX_SCHEMA_VERSION,
        self_hash_key="sec_prefix_sha256",
    )
    plan = validate_acquisition_plan(
        build_acquisition_plan(DEVELOPMENT),
        expected_stage=DEVELOPMENT,
    )
    try:
        validation = dict(
            _validate_sec_detached_replay(prefix, stage=DEVELOPMENT)
        )
        elapsed = float.fromhex(manifest["elapsed_seconds_hex"])
        if (
            not math.isfinite(elapsed)
            or elapsed < 0.0
            or elapsed.hex() != manifest["elapsed_seconds_hex"]
        ):
            raise ValueError("noncanonical elapsed")
        recomputed = _build_sec_prefix_manifest(
            readiness=manifest["repository"],
            plan=plan,
            prefix=prefix,
            validation=validation,
            tree_bytes=tree_bytes,
            inventory=inventory,
            contact_sha256=prefix["sec_replay_evidence"][
                "stage_artifact"
            ]["user_agent_sha256"],
            started_at_utc=manifest["started_at_utc"],
            finished_at_utc=manifest["finished_at_utc"],
            elapsed_seconds=elapsed,
        )
    except SecGemmaLeanAcquisitionError:
        raise
    except Exception:
        raise SecGemmaLeanAcquisitionError("sec_prefix_invalid") from None
    if recomputed != manifest:
        raise SecGemmaLeanAcquisitionError("sec_prefix_invalid")
    return manifest, prefix, validation


def _source_resume_evidence(
    repo_root: Path,
    prefix_manifest: Mapping[str, Any],
    prefix: Mapping[str, Any],
) -> dict[str, Any]:
    contract = prefix_manifest.get("resume_contract")
    if (
        type(contract) is not dict
        or contract.get("catalog_checkpoint") != PRIVATE_SEC_CATALOG_DIRECTORY
        or contract.get("document_chunk_prefix")
        != PRIVATE_SEC_DOCUMENT_CHUNK_PREFIX
        or contract.get("document_chunk_size") != SEC_DOCUMENT_CHUNK_SIZE
        or type(contract.get("document_chunk_count")) is not int
        or contract["document_chunk_count"] <= 0
        or contract.get("per_unit_transport_deadline_seconds_hex")
        != float(MAX_SEC_SECONDS).hex()
        or contract.get("aggregate_sec_deadline_applied") is not False
        or contract.get("final_evidence_rebuilt_as_exact_frozen_batch") is not True
    ):
        raise SecGemmaLeanAcquisitionError("resume_manifest_set_invalid")
    catalogue, catalog_state, _ = _load_sec_catalog_checkpoint(repo_root)
    rows: list[dict[str, Any]] = [
        {
            "unit": "catalog",
            "index": 0,
            "directory": PRIVATE_SEC_CATALOG_DIRECTORY,
            "manifest_sha256": catalogue["sec_catalog_checkpoint_sha256"],
        }
    ]
    authority_manifests = [catalogue]
    chunk_count = contract["document_chunk_count"]
    if (
        catalogue.get("filing_count") != chunk_count
        or prefix_manifest.get("filing_count") != chunk_count
    ):
        raise SecGemmaLeanAcquisitionError("resume_manifest_set_invalid")
    document_plan = catalog_state["document_plan"]
    selected = catalog_state["selected_records"]
    universe = catalog_state["corpus_universe_manifest"]
    chunk_plans = [
        document_plan[index : index + SEC_DOCUMENT_CHUNK_SIZE]
        for index in range(0, len(document_plan), SEC_DOCUMENT_CHUNK_SIZE)
    ]
    if len(chunk_plans) != chunk_count:
        raise SecGemmaLeanAcquisitionError("resume_manifest_set_invalid")
    batches: list[Any] = []
    for index in range(chunk_count):
        directory = _sec_document_chunk_directory(index)
        manifest, batch = _load_sec_document_chunk(
            repo_root,
            expected_plan=chunk_plans[index],
            chunk_index=index,
            chunk_count=chunk_count,
            corpus_universe_sha256=universe["universe_sha256"],
        )
        if (
            manifest.get("chunk_index") != index
            or manifest.get("chunk_count") != chunk_count
            or manifest.get("document_count") != SEC_DOCUMENT_CHUNK_SIZE
            or manifest.get("corpus_universe_sha256")
            != catalogue.get("corpus_universe_sha256")
        ):
            raise SecGemmaLeanAcquisitionError("resume_manifest_set_invalid")
        authority_manifests.append(manifest)
        batches.append(batch)
        rows.append(
            {
                "unit": "document_chunk",
                "index": index,
                "directory": directory,
                "manifest_sha256": manifest["sec_document_chunk_sha256"],
            }
        )
    checkpoint_root = _safe_private_stage_root(repo_root)
    observed_chunks = {
        item.name
        for item in checkpoint_root.iterdir()
        if item.name.startswith(PRIVATE_SEC_DOCUMENT_CHUNK_PREFIX)
        and not item.name.startswith(".")
    }
    if observed_chunks != {
        _sec_document_chunk_directory(index) for index in range(chunk_count)
    }:
        raise SecGemmaLeanAcquisitionError("resume_manifest_set_invalid")
    for manifest in authority_manifests:
        if (
            manifest.get("repository") != prefix_manifest.get("repository")
            or manifest.get("sec_contact_sha256")
            != prefix_manifest.get("sec_contact_sha256")
            or manifest.get("acquisition_plan_sha256")
            != prefix_manifest.get("acquisition_plan_sha256")
        ):
            raise SecGemmaLeanAcquisitionError("resume_manifest_set_invalid")
    aggregate = _aggregate_document_batches(
        batches,
        document_plan=document_plan,
        contact_sha256=prefix_manifest["sec_contact_sha256"],
    )
    stage_evidence = _build_detached_stage_evidence(
        stage=DEVELOPMENT,
        universe=universe,
        selected=selected,
        document_batch=aggregate,
    )
    reconstructed = {
        "sec_replay_evidence": {
            "catalog_request_receipts": catalog_state[
                "catalog_request_receipts"
            ],
            "catalog_artifact": catalog_state["catalog_artifact"],
            "corpus_universe_manifest": universe,
            **stage_evidence,
        },
        "sec_catalog_sources": catalog_state["catalog_sources"],
        "sec_primary_documents": [
            {
                "accession_number": record["accession_number"],
                "form": record["form"],
                "availability_session": record["availability_session"],
                "acceptance_datetime": record["acceptance_datetime"],
                "official_url": document.url,
                "body": document.raw_primary_document,
            }
            for record, document in zip(
                selected,
                aggregate.documents,
                strict=True,
            )
        ],
    }
    if reconstructed != prefix:
        raise SecGemmaLeanAcquisitionError("resume_manifest_set_invalid")
    body = {
        "resume_contract": dict(contract),
        "sec_prefix_checkpoint_sha256": prefix_manifest[
            "sec_prefix_sha256"
        ],
        "unit_manifest_count": len(rows),
        "unit_manifests": rows,
        "unit_manifest_set_sha256": canonical_sha256(rows),
        "all_units_immutable_and_canonical": True,
    }
    return {**body, "source_resume_sha256": canonical_sha256(body)}


def _load_completed_checkpoint(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _safe_private_stage_root(repo_root) / PRIVATE_STAGE_DIRECTORY
    _safe_directory(root, repo_root)
    try:
        root_details = root.lstat()
    except OSError:
        raise SecGemmaLeanAcquisitionError("checkpoint_missing") from None
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or stat.S_ISLNK(root_details.st_mode)
        or _is_reparse(root_details)
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_missing")
    manifest_path = root / PRIVATE_MANIFEST_NAME
    tree_path = root / PRIVATE_TREE_NAME
    _regular_single_link_file(manifest_path)
    tree_details = _regular_single_link_file(tree_path)
    try:
        manifest_bytes = manifest_path.read_bytes()
        tree_bytes = tree_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
        tree = json.loads(tree_bytes.decode("utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_manifest_invalid"
        ) from None
    if (
        type(manifest) is not dict
        or manifest_bytes != _plain_json_bytes(manifest)
        or type(tree) is not dict
        or tree_bytes != _plain_json_bytes(tree)
        or set(tree) != {"schema_version", "root"}
        or tree["schema_version"] != TREE_SCHEMA_VERSION
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
    body = {
        key: value for key, value in manifest.items() if key != "checkpoint_sha256"
    }
    tree_claim = manifest.get("tree")
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or manifest.get("status") != "passed"
        or manifest.get("stage") != DEVELOPMENT
        or manifest.get("checkpoint_sha256") != canonical_sha256(body)
        or type(tree_claim) is not dict
        or tree_claim.get("path") != PRIVATE_TREE_NAME
        or tree_claim.get("sha256") != _sha256_bytes(tree_bytes)
        or tree_claim.get("byte_count") != len(tree_bytes)
        or tree_details.st_size != len(tree_bytes)
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
    raw_inventory = tree_claim.get("blob_inventory")
    if type(raw_inventory) is not list:
        raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
    expected_inventory: dict[str, int] = {}
    expected_paths: set[str] = set()
    for row in raw_inventory:
        if (
            type(row) is not dict
            or set(row) != {"sha256", "byte_count", "path"}
            or type(row.get("sha256")) is not str
            or _SHA256_RE.fullmatch(row["sha256"]) is None
            or type(row.get("byte_count")) is not int
            or row["byte_count"] < 0
            or row.get("path") != f"blobs/{row['sha256']}.bin"
            or row["sha256"] in expected_inventory
        ):
            raise SecGemmaLeanAcquisitionError("checkpoint_manifest_invalid")
        expected_inventory[row["sha256"]] = row["byte_count"]
        expected_paths.add(row["path"])
    blob_root = root / "blobs"
    try:
        blob_root_details = blob_root.lstat()
        if (
            not stat.S_ISDIR(blob_root_details.st_mode)
            or stat.S_ISLNK(blob_root_details.st_mode)
            or _is_reparse(blob_root_details)
        ):
            raise OSError("blob root is not an owned directory")
        observed_paths: set[str] = set()
        for item in blob_root.iterdir():
            _regular_single_link_file(item)
            observed_paths.add(item.relative_to(root).as_posix())
    except OSError:
        raise SecGemmaLeanAcquisitionError("checkpoint_blob_invalid") from None
    if (
        expected_paths != observed_paths
        or tree_claim.get("blob_count") != len(expected_inventory)
        or tree_claim.get("blob_byte_count") != sum(expected_inventory.values())
    ):
        raise SecGemmaLeanAcquisitionError("checkpoint_blob_invalid")
    seen: set[str] = set()
    bundle = _decode_external_tree(
        tree["root"],
        blob_root=blob_root,
        expected_inventory=expected_inventory,
        seen=seen,
    )
    if seen != set(expected_inventory) or type(bundle) is not dict:
        raise SecGemmaLeanAcquisitionError("checkpoint_tree_invalid")
    plan = validate_acquisition_plan(
        build_acquisition_plan(DEVELOPMENT),
        expected_stage=DEVELOPMENT,
    )
    try:
        validation = dict(_validate_one_bundle(bundle, plan=plan, predecessor=None))
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_scientific_replay_failed"
        ) from None
    stored_accounting = manifest.get("request_accounting")
    market_elapsed_seconds_hex = (
        None
        if type(stored_accounting) is not dict
        else stored_accounting.get("market_elapsed_seconds_hex")
    )
    try:
        elapsed = float.fromhex(manifest["elapsed_seconds_hex"])
        if (
            not math.isfinite(elapsed)
            or elapsed < 0.0
            or elapsed.hex() != manifest["elapsed_seconds_hex"]
        ):
            raise ValueError("noncanonical elapsed time")
        if type(market_elapsed_seconds_hex) is not str:
            raise ValueError("missing market elapsed time")
        accounting = _execution_request_accounting(
            bundle,
            market_elapsed_seconds_hex=market_elapsed_seconds_hex,
        )
        pilot = _pilot_selection(
            bundle["private_quarantine"]["model_requests"]
        )
        prefix_manifest, prefix, _ = _load_sec_prefix(repo_root)
        if (
            prefix_manifest.get("repository") != manifest.get("repository")
            or prefix_manifest.get("sec_contact_sha256")
            != manifest.get("sec_contact_sha256")
            or prefix_manifest.get("acquisition_plan_sha256")
            != manifest.get("acquisition_plan_sha256")
        ):
            raise ValueError("source prefix authority changed")
        source_resume = _source_resume_evidence(
            repo_root,
            prefix_manifest,
            prefix,
        )
        identity_sha256 = bundle["public_manifest"]["sec_identity_sha256"]
        recomputed_manifest = _build_checkpoint_manifest(
            readiness=manifest["repository"],
            plan=plan,
            bundle=bundle,
            validation=validation,
            accounting=accounting,
            tree_bytes=tree_bytes,
            inventory=expected_inventory,
            pilot_selection=pilot,
            source_resume=source_resume,
            contact_sha256=identity_sha256,
            started_at_utc=manifest["started_at_utc"],
            finished_at_utc=manifest["finished_at_utc"],
            elapsed_seconds=elapsed,
        )
    except SecGemmaLeanAcquisitionError:
        raise
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_manifest_invalid"
        ) from None
    if (
        recomputed_manifest != manifest
        or bundle.get("bundle_sha256") != manifest.get("bundle_sha256")
        or bundle.get("public_manifest", {}).get("manifest_sha256")
        != manifest.get("public_manifest_sha256")
        or bundle.get("private_index_sha256")
        != manifest.get("private_index_sha256")
        or _to_plain_json_value(validation)
        != manifest.get("detached_validation")
    ):
        raise SecGemmaLeanAcquisitionError(
            "checkpoint_scientific_replay_failed"
        )
    return manifest, bundle, validation


def _public_receipt(
    *,
    checkpoint: Mapping[str, Any],
    readiness_after: Mapping[str, Any],
    journal_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint.get("repository") != readiness_after:
        raise SecGemmaLeanAcquisitionError("repository_changed_during_acquisition")
    if (
        type(journal_summary) is not dict
        or journal_summary.get("terminal_outcome_recorded") is not True
        or type(journal_summary.get("head_event_sha256")) is not str
        or _SHA256_RE.fullmatch(journal_summary["head_event_sha256"]) is None
        or type(journal_summary.get("event_count")) is not int
        or journal_summary["event_count"] < 2
    ):
        raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
    lifetime = journal_summary.get("lifetime_network_accounting")
    if (
        type(lifetime) is not dict
        or set(lifetime) != {"sec", "yahoo"}
        or lifetime["sec"].get("known_network_requests", -1)
        < checkpoint["request_accounting"]["sec_request_count"]
        or lifetime["yahoo"].get("known_network_requests", -1)
        < checkpoint["request_accounting"]["market_request_count"]
    ):
        raise SecGemmaLeanAcquisitionError("attempt_journal_accounting_invalid")
    suitability = checkpoint["suitability"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "stage": DEVELOPMENT,
        "started_at_utc": checkpoint["started_at_utc"],
        "finished_at_utc": checkpoint["finished_at_utc"],
        "elapsed_seconds_hex": checkpoint["elapsed_seconds_hex"],
        "repository": dict(readiness_after),
        "scientific_parent_commit": SCIENTIFIC_PARENT_COMMIT,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preflight_sha256": checkpoint["preflight_sha256"],
        "acquisition_plan_sha256": checkpoint["acquisition_plan_sha256"],
        "sec_contact_sha256": checkpoint["sec_contact_sha256"],
        "private_checkpoint": {
            "path": f"{CHECKPOINT_ROOT}/{PRIVATE_STAGE_DIRECTORY}",
            "ignored_local_state": True,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "bundle_sha256": checkpoint["bundle_sha256"],
            "public_manifest_sha256": checkpoint["public_manifest_sha256"],
            "private_index_sha256": checkpoint["private_index_sha256"],
            "tree_sha256": checkpoint["tree"]["sha256"],
            "blob_count": checkpoint["tree"]["blob_count"],
            "blob_byte_count": checkpoint["tree"]["blob_byte_count"],
            "reload_and_scientific_replay_passed": True,
        },
        "request_accounting": dict(checkpoint["request_accounting"]),
        "suitability": dict(suitability),
        "pilot_selection": list(checkpoint["pilot_selection"]),
        "source_resume": {
            "resume_contract": dict(
                checkpoint["source_resume"]["resume_contract"]
            ),
            "sec_prefix_checkpoint_sha256": checkpoint["source_resume"][
                "sec_prefix_checkpoint_sha256"
            ],
            "unit_manifest_count": checkpoint["source_resume"][
                "unit_manifest_count"
            ],
            "unit_manifest_set_sha256": checkpoint["source_resume"][
                "unit_manifest_set_sha256"
            ],
            "source_resume_sha256": checkpoint["source_resume"][
                "source_resume_sha256"
            ],
        },
        "effect_counts": dict(checkpoint["effect_counts"]),
        "attempt_history": dict(journal_summary),
        "privacy": {
            "readable_sec_contact_published": False,
            "readable_sec_contact_checkpointed": False,
            "sec_contact_fingerprint_only": True,
            "provider_echo_scan_passed": True,
        },
        "trust": {
            "production_defaults_only": True,
            "test_injections_used": False,
            "old_v22_store_or_vault_authority_used": False,
            "source_bound_lean_execution": True,
        },
        "performance_results_opened": False,
        "model_authorized": False,
        "execution_authorized": False,
        "next_step": "five_largest_development_request_gemma_pilot",
    }
    return {**body, "acquisition_sha256": canonical_sha256(body)}


def _failure_evidence_commitments(checkpoint_root: Path) -> dict[str, Any]:
    directory_names = {
        PRIVATE_SEC_CATALOG_DIRECTORY,
        PRIVATE_SEC_PREFIX_DIRECTORY,
        PRIVATE_STAGE_DIRECTORY,
    }
    try:
        directory_names.update(
            item.name
            for item in checkpoint_root.iterdir()
            if re.fullmatch(
                re.escape(PRIVATE_SEC_DOCUMENT_CHUNK_PREFIX) + r"[0-9]{4}",
                item.name,
            )
            is not None
        )
    except OSError:
        raise SecGemmaLeanAcquisitionError(
            "failure_evidence_unavailable"
        ) from None
    rows: list[dict[str, Any]] = []
    for directory_name in sorted(directory_names):
        path = checkpoint_root / directory_name / PRIVATE_MANIFEST_NAME
        try:
            details = path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            rows.append(
                {
                    "directory": directory_name,
                    "manifest_state": "unreadable",
                    "manifest_byte_count": None,
                    "manifest_bytes_sha256": None,
                }
            )
            continue
        if (
            not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or _is_reparse(details)
            or not 1 <= details.st_size <= 16 * 1024 * 1024
        ):
            rows.append(
                {
                    "directory": directory_name,
                    "manifest_state": "unsafe",
                    "manifest_byte_count": None,
                    "manifest_bytes_sha256": None,
                }
            )
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            rows.append(
                {
                    "directory": directory_name,
                    "manifest_state": "unreadable",
                    "manifest_byte_count": None,
                    "manifest_bytes_sha256": None,
                }
            )
            continue
        rows.append(
            {
                "directory": directory_name,
                "manifest_state": "byte_committed_not_claim_validated",
                "manifest_byte_count": len(payload),
                "manifest_bytes_sha256": _sha256_bytes(payload),
            }
        )
    body = {
        "ignored_checkpoint_root": CHECKPOINT_ROOT,
        "manifest_count": len(rows),
        "manifests": rows,
        "claims_in_failed_manifests_treated_as_authority": False,
    }
    return {**body, "failure_evidence_sha256": canonical_sha256(body)}


def _public_failure_receipt(
    *,
    terminal_event: Mapping[str, Any],
    readiness: Mapping[str, Any],
    plan: Mapping[str, Any],
    contact: _PrivateSecContact,
    journal_summary: Mapping[str, Any],
) -> dict[str, Any]:
    payload = terminal_event.get("payload")
    if (
        type(payload) is not dict
        or payload.get("status") != "rejected"
        or payload.get("disposition")
        not in {
            "terminal_suitability_rejection",
            "terminal_integrity_rejection",
        }
        or type(payload.get("code")) is not str
        or type(payload.get("evidence_commitments")) is not dict
        or journal_summary.get("head_event_sha256")
        != terminal_event.get("event_sha256")
        or journal_summary.get("terminal_outcome_recorded") is not True
    ):
        raise SecGemmaLeanAcquisitionError("attempt_journal_terminal_invalid")
    lifetime = journal_summary.get("lifetime_network_accounting")
    if type(lifetime) is not dict or set(lifetime) != {"sec", "yahoo"}:
        raise SecGemmaLeanAcquisitionError("attempt_journal_invalid")
    sec = lifetime["sec"]
    yahoo = lifetime["yahoo"]
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "rejected",
        "stage": DEVELOPMENT,
        "finished_at_utc": terminal_event["occurred_at_utc"],
        "repository": dict(readiness),
        "scientific_parent_commit": SCIENTIFIC_PARENT_COMMIT,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preflight_sha256": readiness["preflight_sha256"],
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "sec_contact_sha256": contact.sha256,
        "rejection": {
            "disposition": payload["disposition"],
            "code": payload["code"],
            "later_stage_access_stopped": True,
        },
        "failure_evidence": dict(payload["evidence_commitments"]),
        "request_accounting": {
            "sealed_evidence_complete": False,
            "lifetime_network_accounting": dict(lifetime),
            "all_request_effects_exact": (
                sec["indeterminate_request_intents"] == 0
                and yahoo["indeterminate_request_intents"] == 0
            ),
        },
        "effect_counts": {
            "official_sec_requests": sec["exact_network_request_count"],
            "official_sec_requests_lower_bound": sec[
                "network_request_count_lower_bound"
            ],
            "official_sec_requests_upper_bound": sec[
                "network_request_count_upper_bound"
            ],
            "market_data_requests": yahoo["exact_network_request_count"],
            "market_data_requests_lower_bound": yahoo[
                "network_request_count_lower_bound"
            ],
            "market_data_requests_upper_bound": yahoo[
                "network_request_count_upper_bound"
            ],
            "model_generation_calls": 0,
            "performance_results_opened": 0,
            "confirmation_sources_opened": 0,
            "final_sources_opened": 0,
            "paid_api_calls": 0,
            "trading_actions": 0,
        },
        "attempt_history": dict(journal_summary),
        "privacy": {
            "readable_sec_contact_published": False,
            "sec_contact_fingerprint_only": True,
        },
        "performance_results_opened": False,
        "model_authorized": False,
        "execution_authorized": False,
        "next_step": "commit_rejection_and_update_approach_comparison",
    }
    _assert_value_redacted(body, contact)
    return {**body, "acquisition_sha256": canonical_sha256(body)}


def _publish_public_receipt(
    repo_root: Path,
    checkpoint_root: Path,
    receipt: Mapping[str, Any],
) -> None:
    target = repo_root / Path(PUBLIC_ARTIFACT_PATH)
    _safe_directory(target.parent, repo_root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise SecGemmaLeanAcquisitionError(
            "public_artifact_unwritable"
        ) from None
    _safe_directory(target.parent, repo_root)
    if target.exists():
        raise SecGemmaLeanAcquisitionError("public_artifact_already_exists")
    payload = (
        json.dumps(
            dict(receipt),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    temporary = checkpoint_root / f".acquisition-seal-{uuid.uuid4().hex}.tmp"
    _write_exclusive(temporary, payload)
    staged = _regular_single_link_file(temporary)
    created_target = False
    try:
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
        created_target = True
        details = _regular_single_link_file(target)
        if (
            (details.st_dev, details.st_ino, details.st_size)
            != (staged.st_dev, staged.st_ino, staged.st_size)
            or details.st_size != len(payload)
            or target.read_bytes() != payload
        ):
            raise OSError("published bytes changed")
        if os.name != "nt":
            temporary.unlink()
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise SecGemmaLeanAcquisitionError(
            "public_artifact_already_exists"
        ) from None
    except OSError:
        if created_target:
            try:
                current = target.lstat()
                if (current.st_dev, current.st_ino) == (
                    staged.st_dev,
                    staged.st_ino,
                ):
                    target.unlink()
            except OSError:
                pass
        temporary.unlink(missing_ok=True)
        raise SecGemmaLeanAcquisitionError(
            "public_artifact_unwritable"
        ) from None


def _new_sec_unit(
    repo_root: Path,
    contact: _PrivateSecContact,
    journal: _AttemptJournal,
    *,
    logical_purpose: str,
) -> tuple[Any, str, Any, SecCorpusBudget]:
    session = _build_sec_session()
    _install_sec_request_journal(
        session,
        journal=journal,
        logical_purpose=logical_purpose,
    )
    user_agent = contact.reveal_for_sec_only()
    started = time.monotonic()
    transport_budget = BudgetCounter(
        clock=time.monotonic,
        max_requests=SEC_REQUEST_CAP,
        max_bytes=SEC_BYTE_CAP,
        max_seconds=float(MAX_SEC_SECONDS),
    )
    audited = SecAuditTransport(
        session=session,
        cache_dir=repo_root / Path(CHECKPOINT_ROOT) / "disabled_sec_cache",
        user_agent=user_agent,
        budget=transport_budget,
        clock=time.monotonic,
        sleep=time.sleep,
        timeout_seconds=SEC_TIMEOUT_SECONDS,
        max_retries=0,
        max_redirects=0,
        allow_cache_reads=False,
        allow_cache_writes=False,
    )
    paced = _PacedSecTransport(
        audited,
        clock=time.monotonic,
        sleeper=time.sleep,
        deadline=started + float(MAX_SEC_SECONDS),
    )
    corpus_budget = SecCorpusBudget(
        clock=time.monotonic,
        max_requests=SEC_REQUEST_CAP,
        max_bytes=SEC_BYTE_CAP,
        max_seconds=float(MAX_SEC_SECONDS),
    )
    return session, user_agent, paced, corpus_budget


def _acquire_catalog_unit(
    *,
    repo_root: Path,
    plan: Mapping[str, Any],
    contact: _PrivateSecContact,
    journal: _AttemptJournal,
) -> tuple[dict[str, Any], dict[str, Any], str, str, float]:
    started = time.monotonic()
    started_at = _utc_now()
    session, user_agent, transport, budget = _new_sec_unit(
        repo_root,
        contact,
        journal,
        logical_purpose="official_sec_catalog",
    )
    try:
        catalog = acquire_official_sec_catalog(
            transport=transport,
            user_agent=user_agent,
            budget=budget,
            session_dates=EXPECTED_SESSIONS,
        )
        universe = build_corpus_universe_manifest(
            catalog_artifact_sha256=catalog.catalog_artifact_sha256,
            calendar_artifact_sha256=build_contract_manifest()["data"][
                "market"
            ]["calendar"]["calendar_dates_sha256"],
            catalog_total_record_count=catalog.catalog_total_record_count,
            catalog_eligible_record_count=catalog.catalog_eligible_record_count,
            session_dates=EXPECTED_SESSIONS,
            records=[dict(record) for record in catalog.universe_records],
        )
        selected = [
            record
            for record in universe["records"]
            if record["artifact_stage"] == plan["sec"]["stage_name"]
        ]
        selected.sort(
            key=lambda item: (
                item["availability_session"],
                item["accession_number"],
            )
        )
        document_plan = [
            {
                "accession_number": item["accession_number"],
                "official_url": _official_primary_url(item),
            }
            for item in selected
        ]
        state = {
            "catalog_sources": [
                {"name": item.name, "url": item.url, "body": item.payload}
                for item in catalog.sources
            ],
            "catalog_request_receipts": json.loads(
                catalog.request_receipts_json.decode("utf-8", errors="strict")
            ),
            "catalog_artifact": json.loads(
                catalog.artifact_json.decode("utf-8", errors="strict")
            ),
            "corpus_universe_manifest": universe,
            "selected_records": selected,
            "document_plan": document_plan,
        }
        validation = _validate_sec_catalog_state(
            state,
            plan=plan,
            expected_contact_sha256=contact.sha256,
        )
        _assert_value_redacted(state, contact)
    except SecGemmaLeanAcquisitionError:
        raise
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "sec_catalog_acquisition_failed"
        ) from None
    finally:
        user_agent = ""
        try:
            session.close()
        except Exception:
            pass
    finished = time.monotonic()
    return state, validation, started_at, _utc_now(), finished - started


def _acquire_document_chunk_unit(
    *,
    repo_root: Path,
    document_plan: list[dict[str, str]],
    contact: _PrivateSecContact,
    journal: _AttemptJournal,
) -> tuple[dict[str, Any], Any, str, str, float]:
    started = time.monotonic()
    started_at = _utc_now()
    session, user_agent, transport, budget = _new_sec_unit(
        repo_root,
        contact,
        journal,
        logical_purpose="official_sec_primary_document",
    )
    try:
        batch = _acquire_authenticated_stage_access_document_batch(
            authenticated_document_plan=document_plan,
            transport=transport,
            user_agent=user_agent,
            budget=budget,
        )
        value = _document_batch_value(document_plan, batch)
        replayed = _validate_document_batch_value(
            value,
            expected_plan=document_plan,
            expected_contact_sha256=contact.sha256,
        )
        _assert_value_redacted(value, contact)
    except SecGemmaLeanAcquisitionError:
        raise
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "sec_document_chunk_acquisition_failed"
        ) from None
    finally:
        user_agent = ""
        try:
            session.close()
        except Exception:
            pass
    finished = time.monotonic()
    return value, replayed, started_at, _utc_now(), finished - started


def _assert_checkpoint_authority(
    manifest: Mapping[str, Any],
    *,
    readiness: Mapping[str, Any],
    plan: Mapping[str, Any],
    contact: _PrivateSecContact,
    error_code: str,
) -> None:
    if (
        manifest.get("repository") != readiness
        or manifest.get("sec_contact_sha256") != contact.sha256
        or manifest.get("acquisition_plan_sha256")
        != plan["acquisition_plan_sha256"]
    ):
        raise SecGemmaLeanAcquisitionError(error_code)


def _acquire_or_resume_sec_prefix(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    contact: _PrivateSecContact,
    readiness: Mapping[str, Any],
    journal: _AttemptJournal,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_root = _safe_private_stage_root(repo_root)
    prefix_target = checkpoint_root / PRIVATE_SEC_PREFIX_DIRECTORY
    if prefix_target.exists():
        prefix_manifest, sec_prefix, _ = _load_sec_prefix(repo_root)
        _assert_checkpoint_authority(
            prefix_manifest,
            readiness=readiness,
            plan=plan,
            contact=contact,
            error_code="existing_sec_prefix_authority_changed",
        )
        _scan_completed_checkpoint_for_contact(
            prefix_target,
            prefix_manifest,
            contact,
        )
        return prefix_manifest, sec_prefix

    catalog_target = checkpoint_root / PRIVATE_SEC_CATALOG_DIRECTORY
    existing_chunks = {
        item.name
        for item in checkpoint_root.iterdir()
        if item.name.startswith(PRIVATE_SEC_DOCUMENT_CHUNK_PREFIX)
        and not item.name.startswith(".")
    }
    if not catalog_target.exists() and existing_chunks:
        raise SecGemmaLeanAcquisitionError("sec_checkpoint_order_invalid")
    if catalog_target.exists():
        last_sec_network_finished: float | None = None
        catalog_manifest, catalog_state, _ = _load_sec_catalog_checkpoint(
            repo_root
        )
        _assert_checkpoint_authority(
            catalog_manifest,
            readiness=readiness,
            plan=plan,
            contact=contact,
            error_code="existing_sec_catalog_authority_changed",
        )
    else:
        (
            catalog_state,
            catalog_validation,
            catalog_started_at,
            catalog_finished_at,
            catalog_elapsed,
        ) = _acquire_catalog_unit(
            repo_root=repo_root,
            plan=plan,
            contact=contact,
            journal=journal,
        )
        last_sec_network_finished = time.monotonic()

        def build_catalog_manifest(
            tree_bytes: bytes,
            inventory: Mapping[str, int],
        ) -> dict[str, Any]:
            return _build_sec_catalog_manifest(
                readiness=readiness,
                plan=plan,
                state=catalog_state,
                validation=catalog_validation,
                tree_bytes=tree_bytes,
                inventory=inventory,
                contact_sha256=contact.sha256,
                started_at_utc=catalog_started_at,
                finished_at_utc=catalog_finished_at,
                elapsed_seconds=catalog_elapsed,
            )

        _seal_external_tree_unit(
            repo_root,
            directory_name=PRIVATE_SEC_CATALOG_DIRECTORY,
            value=catalog_state,
            manifest_factory=build_catalog_manifest,
            contact=contact,
            publish_error_code="sec_catalog_checkpoint_publish_failed",
        )
        del build_catalog_manifest, catalog_validation
        catalog_manifest, catalog_state, _ = _load_sec_catalog_checkpoint(
            repo_root
        )
        _assert_checkpoint_authority(
            catalog_manifest,
            readiness=readiness,
            plan=plan,
            contact=contact,
            error_code="sec_catalog_authority_changed",
        )
    _scan_completed_checkpoint_for_contact(
        catalog_target,
        catalog_manifest,
        contact,
    )
    catalog_commitment = catalog_manifest["sec_catalog_checkpoint_sha256"]
    if not journal.has_source_commitment(catalog_commitment):
        journal.source_unit_sealed(
            provider="sec",
            unit="official_sec_catalog",
            request_count=catalog_manifest["request_accounting"][
                "request_count"
            ],
            response_bytes=catalog_manifest["request_accounting"]["byte_count"],
            commitment_sha256=catalog_commitment,
        )

    document_plan = catalog_state["document_plan"]
    selected = catalog_state["selected_records"]
    universe = catalog_state["corpus_universe_manifest"]
    chunk_plans = [
        document_plan[index : index + SEC_DOCUMENT_CHUNK_SIZE]
        for index in range(0, len(document_plan), SEC_DOCUMENT_CHUNK_SIZE)
    ]
    expected_chunk_names = {
        _sec_document_chunk_directory(index)
        for index in range(len(chunk_plans))
    }
    if not existing_chunks.issubset(expected_chunk_names):
        raise SecGemmaLeanAcquisitionError("sec_document_chunk_unexpected")
    seen_missing = False
    for index in range(len(chunk_plans)):
        exists = (
            checkpoint_root / _sec_document_chunk_directory(index)
        ).exists()
        if not exists:
            seen_missing = True
        elif seen_missing:
            raise SecGemmaLeanAcquisitionError("sec_document_chunk_gap")

    chunk_manifests: list[dict[str, Any]] = []
    batches: list[Any] = []
    for chunk_index, chunk_plan in enumerate(chunk_plans):
        directory = _sec_document_chunk_directory(chunk_index)
        target = checkpoint_root / directory
        if target.exists():
            chunk_manifest, batch = _load_sec_document_chunk(
                repo_root,
                expected_plan=chunk_plan,
                chunk_index=chunk_index,
                chunk_count=len(chunk_plans),
                corpus_universe_sha256=universe["universe_sha256"],
            )
            _assert_checkpoint_authority(
                chunk_manifest,
                readiness=readiness,
                plan=plan,
                contact=contact,
                error_code="existing_sec_document_chunk_authority_changed",
            )
        else:
            if last_sec_network_finished is not None:
                now = time.monotonic()
                earliest = last_sec_network_finished + (
                    1.0 / MAX_SEC_REQUESTS_PER_SECOND
                )
                if not math.isfinite(now) or now < 0.0:
                    raise SecGemmaLeanAcquisitionError(
                        "monotonic_clock_invalid"
                    )
                if now < earliest:
                    time.sleep(earliest - now)
            (
                chunk_value,
                batch,
                chunk_started_at,
                chunk_finished_at,
                chunk_elapsed,
            ) = _acquire_document_chunk_unit(
                repo_root=repo_root,
                document_plan=chunk_plan,
                contact=contact,
                journal=journal,
            )
            last_sec_network_finished = time.monotonic()

            def build_chunk_manifest(
                tree_bytes: bytes,
                inventory: Mapping[str, int],
                *,
                value: dict[str, Any] = chunk_value,
                replayed_batch: Any = batch,
                index: int = chunk_index,
                started_at: str = chunk_started_at,
                finished_at: str = chunk_finished_at,
                elapsed: float = chunk_elapsed,
            ) -> dict[str, Any]:
                return _build_sec_document_chunk_manifest(
                    readiness=readiness,
                    plan=plan,
                    value=value,
                    batch=replayed_batch,
                    chunk_index=index,
                    chunk_count=len(chunk_plans),
                    corpus_universe_sha256=universe["universe_sha256"],
                    tree_bytes=tree_bytes,
                    inventory=inventory,
                    contact_sha256=contact.sha256,
                    started_at_utc=started_at,
                    finished_at_utc=finished_at,
                    elapsed_seconds=elapsed,
                )

            _seal_external_tree_unit(
                repo_root,
                directory_name=directory,
                value=chunk_value,
                manifest_factory=build_chunk_manifest,
                contact=contact,
                publish_error_code="sec_document_chunk_publish_failed",
            )
            del build_chunk_manifest, chunk_value
            chunk_manifest, batch = _load_sec_document_chunk(
                repo_root,
                expected_plan=chunk_plan,
                chunk_index=chunk_index,
                chunk_count=len(chunk_plans),
                corpus_universe_sha256=universe["universe_sha256"],
            )
            _assert_checkpoint_authority(
                chunk_manifest,
                readiness=readiness,
                plan=plan,
                contact=contact,
                error_code="sec_document_chunk_authority_changed",
            )
        _scan_completed_checkpoint_for_contact(target, chunk_manifest, contact)
        chunk_commitment = chunk_manifest["sec_document_chunk_sha256"]
        if not journal.has_source_commitment(chunk_commitment):
            journal.source_unit_sealed(
                provider="sec",
                unit="official_sec_primary_document",
                request_count=chunk_manifest["request_accounting"][
                    "request_count"
                ],
                response_bytes=chunk_manifest["request_accounting"]["byte_count"],
                commitment_sha256=chunk_commitment,
            )
        chunk_manifests.append(chunk_manifest)
        batches.append(batch)

    aggregate = _aggregate_document_batches(
        batches,
        document_plan=document_plan,
        contact_sha256=contact.sha256,
    )
    stage_evidence = _build_detached_stage_evidence(
        stage=DEVELOPMENT,
        universe=universe,
        selected=selected,
        document_batch=aggregate,
    )
    documents = [
        {
            "accession_number": record["accession_number"],
            "form": record["form"],
            "availability_session": record["availability_session"],
            "acceptance_datetime": record["acceptance_datetime"],
            "official_url": document.url,
            "body": document.raw_primary_document,
        }
        for record, document in zip(selected, aggregate.documents, strict=True)
    ]
    sec_prefix = {
        "sec_replay_evidence": {
            "catalog_request_receipts": catalog_state[
                "catalog_request_receipts"
            ],
            "catalog_artifact": catalog_state["catalog_artifact"],
            "corpus_universe_manifest": universe,
            **stage_evidence,
        },
        "sec_catalog_sources": catalog_state["catalog_sources"],
        "sec_primary_documents": documents,
    }
    _assert_value_redacted(sec_prefix, contact)
    try:
        sec_validation = dict(
            _validate_sec_detached_replay(sec_prefix, stage=DEVELOPMENT)
        )
    except Exception:
        raise SecGemmaLeanAcquisitionError("sec_acquisition_failed") from None
    sec_elapsed = _elapsed_from_manifest(catalog_manifest) + sum(
        _elapsed_from_manifest(manifest) for manifest in chunk_manifests
    )
    sec_finished_at = _utc_now()

    def build_sec_prefix_manifest(
        tree_bytes: bytes,
        inventory: Mapping[str, int],
    ) -> dict[str, Any]:
        return _build_sec_prefix_manifest(
            readiness=readiness,
            plan=plan,
            prefix=sec_prefix,
            validation=sec_validation,
            tree_bytes=tree_bytes,
            inventory=inventory,
            contact_sha256=contact.sha256,
            started_at_utc=catalog_manifest["started_at_utc"],
            finished_at_utc=sec_finished_at,
            elapsed_seconds=sec_elapsed,
        )

    _seal_sec_prefix(
        repo_root,
        prefix=sec_prefix,
        manifest_factory=build_sec_prefix_manifest,
        contact=contact,
    )
    del build_sec_prefix_manifest, sec_prefix, sec_validation
    prefix_manifest, sec_prefix, _ = _load_sec_prefix(repo_root)
    _assert_checkpoint_authority(
        prefix_manifest,
        readiness=readiness,
        plan=plan,
        contact=contact,
        error_code="sec_prefix_authority_changed",
    )
    _scan_completed_checkpoint_for_contact(
        prefix_target,
        prefix_manifest,
        contact,
    )
    return prefix_manifest, sec_prefix


def _acquire_bundle(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    contact: _PrivateSecContact,
    readiness: Mapping[str, Any],
    journal: _AttemptJournal,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    prefix_manifest, sec_prefix = _acquire_or_resume_sec_prefix(
        repo_root=repo_root,
        plan=plan,
        contact=contact,
        readiness=readiness,
        journal=journal,
    )
    source_resume = _source_resume_evidence(
        repo_root,
        prefix_manifest,
        sec_prefix,
    )
    post_sec_started = time.monotonic()
    evidence = sec_prefix["sec_replay_evidence"]
    universe = evidence["corpus_universe_manifest"]
    catalog_sources = sec_prefix["sec_catalog_sources"]
    documents = sec_prefix["sec_primary_documents"]

    try:
        market_parent_deadline = time.monotonic() + float(MAX_SEC_SECONDS)
        market_transport = create_production_market_transport(
            plan,
            deadline_monotonic=market_parent_deadline,
        )
        market_transport._opener = _JournaledYahooOpener(
            market_transport._opener,
            journal,
        )
        market_security = _market_security_state(market_transport)
        (
            market_responses,
            market_receipts,
            market_elapsed_seconds_hex,
        ) = _acquire_market(
            transport=market_transport,
            plan=plan,
            clock=time.monotonic,
            combined_deadline=market_parent_deadline,
            security_state=market_security,
        )
    except Exception:
        raise SecGemmaLeanAcquisitionError("market_acquisition_failed") from None

    try:
        model_requests = _build_model_requests(
            documents,
            carry_in_documents=[],
            stage=DEVELOPMENT,
        )
        sec_receipts: list[dict[str, Any]] = []
        for source, receipt in zip(
            catalog_sources,
            evidence["catalog_request_receipts"],
            strict=True,
        ):
            sec_receipts.append(
                _sec_receipt(
                    sequence_number=0,
                    purpose=(
                        "apple_main_submissions"
                        if source["url"] == MAIN_SUBMISSIONS_URL
                        else "apple_historical_submissions"
                    ),
                    raw_receipt=receipt,
                    payload=source["body"],
                )
            )
        for document, receipt in zip(
            documents,
            evidence["authenticated_stage_request_receipts"],
            strict=True,
        ):
            sec_receipts.append(
                _sec_receipt(
                    sequence_number=0,
                    purpose="development_official_primary_document",
                    raw_receipt=receipt,
                    payload=document["body"],
                )
            )
        all_receipts = sec_receipts + market_receipts
        all_receipts = [
            {
                **{
                    key: value
                    for key, value in receipt.items()
                    if key not in {"sequence_number", "receipt_sha256"}
                },
                "sequence_number": index,
            }
            for index, receipt in enumerate(all_receipts, start=1)
        ]
        all_receipts = [
            {**receipt, "receipt_sha256": canonical_sha256(receipt)}
            for receipt in all_receipts
        ]
        market_commitments, current_rows = _market_commitments(
            {item["symbol"]: item["body"] for item in market_responses},
            plan,
        )
        universe_event_proofs = _build_universe_event_proofs(
            stage=DEVELOPMENT,
            universe=universe,
            current_documents=documents,
            predecessor_bundles=[],
        )
        model_slice = _build_model_slice(
            plan=plan,
            model_requests=model_requests,
            universe_event_proofs=universe_event_proofs,
        )
        stage_slice = _build_stage_slice(
            plan=plan,
            rows_by_symbol=current_rows,
            model_requests=model_requests,
            universe_event_proofs=universe_event_proofs,
        )
        private = {
            "sec_replay_evidence": evidence,
            "sec_catalog_sources": catalog_sources,
            "sec_primary_documents": documents,
            "carry_in_documents": [],
            "market_responses": market_responses,
            "request_receipts": all_receipts,
            "model_requests": model_requests,
            "model_slice": model_slice,
            "stage_slice": stage_slice,
        }
        _assert_value_redacted(private, contact)
        bundle = _bundle(
            plan=plan,
            identity_sha256=contact.sha256,
            predecessor_bundle_sha256=None,
            private=private,
            market_commitments=market_commitments,
            prefix_continuity_verified=True,
        )
        _assert_value_redacted(bundle, contact)
        validation = dict(
            _validate_one_bundle(bundle, plan=plan, predecessor=None)
        )
        accounting = _execution_request_accounting(
            bundle,
            market_elapsed_seconds_hex=market_elapsed_seconds_hex,
        )
        market_commitment = canonical_sha256(market_receipts)
        if not journal.has_source_commitment(market_commitment):
            journal.source_unit_sealed(
                provider="yahoo",
                unit="development_market_chart_batch",
                request_count=len(market_receipts),
                response_bytes=sum(
                    int(receipt["byte_count"]) for receipt in market_receipts
                ),
                commitment_sha256=market_commitment,
            )
    except SecGemmaLeanAcquisitionError:
        raise
    except Exception:
        raise SecGemmaLeanAcquisitionError(
            "source_suitability_failed"
        ) from None
    finished_at = _utc_now()
    source_elapsed = _elapsed_from_manifest(prefix_manifest) + (
        time.monotonic() - post_sec_started
    )
    return bundle, market_elapsed_seconds_hex, {
        "validation": validation,
        "accounting": accounting,
        "source_started_at_utc": prefix_manifest["started_at_utc"],
        "source_finished_at_utc": finished_at,
        "source_elapsed_seconds": source_elapsed,
        "source_resume": source_resume,
    }


def _verify_local_function_identities() -> None:
    observed = {
        "acquire_bundle": id(_acquire_bundle),
        "build_sec_session": id(_build_sec_session),
        "seal_private_checkpoint": id(_seal_private_checkpoint),
        "load_completed_checkpoint": id(_load_completed_checkpoint),
        "seal_sec_prefix": id(_seal_sec_prefix),
        "load_sec_prefix": id(_load_sec_prefix),
        "load_external_tree_checkpoint": id(
            _load_external_tree_checkpoint
        ),
        "seal_external_tree_unit": id(_seal_external_tree_unit),
        "assert_value_redacted": id(_assert_value_redacted),
        "load_sec_catalog_checkpoint": id(_load_sec_catalog_checkpoint),
        "load_sec_document_chunk": id(_load_sec_document_chunk),
        "source_resume_evidence": id(_source_resume_evidence),
        "to_plain_json_value": id(_to_plain_json_value),
        "aggregate_document_batches": id(_aggregate_document_batches),
        "new_sec_unit": id(_new_sec_unit),
        "acquire_catalog_unit": id(_acquire_catalog_unit),
        "acquire_document_chunk_unit": id(_acquire_document_chunk_unit),
        "assert_checkpoint_authority": id(_assert_checkpoint_authority),
        "acquire_or_resume_sec_prefix": id(_acquire_or_resume_sec_prefix),
        "attempt_journal": id(_AttemptJournal),
        "install_sec_request_journal": id(_install_sec_request_journal),
        "journaled_yahoo_opener": id(_JournaledYahooOpener),
        "failure_disposition": id(_failure_disposition),
        "failure_evidence_commitments": id(_failure_evidence_commitments),
        "build_public_failure_receipt": id(_public_failure_receipt),
        "publish_public_receipt": id(_publish_public_receipt),
        "build_public_receipt": id(_public_receipt),
        "verify_acquisition_readiness": id(_verify_acquisition_readiness),
        "load_private_contact": id(_load_private_contact),
        "scan_completed_checkpoint_for_contact": id(
            _scan_completed_checkpoint_for_contact
        ),
        "recover_stale_acquisition_lock": id(
            _recover_stale_acquisition_lock
        ),
        "run_development_acquisition": id(run_development_acquisition),
    }
    if observed != _FROZEN_LOCAL_FUNCTION_IDENTITIES:
        raise SecGemmaLeanAcquisitionError(
            "production_function_identity_changed"
        )


def _failure_disposition(code: str) -> str:
    if code in {
        "repository_changed_during_acquisition",
        "production_function_identity_changed",
        "frozen_helper_identity_changed",
        "attempt_journal_invalid",
        "attempt_journal_write_failed",
        "attempt_journal_authority_changed",
    }:
        return "blocked_indeterminate"
    if code in {
        "sec_catalog_acquisition_failed",
        "sec_document_chunk_acquisition_failed",
        "market_acquisition_failed",
        "sec_catalog_checkpoint_publish_failed",
        "sec_document_chunk_publish_failed",
        "sec_prefix_publish_failed",
        "checkpoint_publish_failed",
        "public_artifact_unwritable",
    }:
        return "resumable_transient"
    if code in {
        "filing_text_unusable",
        "sec_prefix_suitability_failed",
    }:
        return "terminal_suitability_rejection"
    if code in {
        "private_contact_leaked",
        "private_contact_echoed_by_provider",
        "sec_catalog_checkpoint_invalid",
        "sec_document_chunk_invalid",
        "resume_manifest_set_invalid",
        "sec_prefix_invalid",
        "checkpoint_manifest_invalid",
        "checkpoint_scientific_replay_failed",
        "sec_prefix_accounting_invalid",
        "existing_checkpoint_authority_changed",
        "existing_sec_prefix_authority_changed",
        "existing_sec_catalog_authority_changed",
        "existing_sec_document_chunk_authority_changed",
        "sec_checkpoint_order_invalid",
        "sec_document_chunk_gap",
        "sec_document_chunk_unexpected",
    }:
        return "terminal_integrity_rejection"
    return "blocked_indeterminate"


def run_development_acquisition(
    repo_root: Path,
) -> dict[str, Any]:
    """Run and seal the fixed source-only development acquisition."""
    _verify_local_function_identities()
    root = _canonical_repo_root(repo_root)
    if (root / Path(PUBLIC_ARTIFACT_PATH)).exists():
        raise SecGemmaLeanAcquisitionError("public_artifact_already_exists")
    checkpoint_root = _safe_private_stage_root(root)
    lock_path = checkpoint_root / ACTIVE_LOCK_NAME
    _recover_stale_acquisition_lock(lock_path)
    plan = validate_acquisition_plan(
        build_acquisition_plan(DEVELOPMENT),
        expected_stage=DEVELOPMENT,
    )
    with _RunLease(lock_path):
        readiness_before = _verify_acquisition_readiness(root)
        contact = _load_private_contact(root)
        journal = _AttemptJournal(
            root,
            readiness=readiness_before,
            plan=plan,
            contact=contact,
        )
        prior_terminal = journal.terminal_event()
        attempt_active = prior_terminal is None
        if attempt_active:
            journal.begin_attempt()
        private_target = checkpoint_root / PRIVATE_STAGE_DIRECTORY
        try:
            if (
                prior_terminal is not None
                and prior_terminal["payload"].get("status") == "rejected"
            ):
                _verify_local_function_identities()
                receipt = _public_failure_receipt(
                    terminal_event=prior_terminal,
                    readiness=readiness_before,
                    plan=plan,
                    contact=contact,
                    journal_summary=journal.summary(),
                )
                _publish_public_receipt(root, checkpoint_root, receipt)
                return receipt
            if private_target.exists():
                checkpoint, reloaded_bundle, reloaded_validation = (
                    _load_completed_checkpoint(root)
                )
                del reloaded_bundle, reloaded_validation
                if (
                    checkpoint.get("repository") != readiness_before
                    or checkpoint.get("sec_contact_sha256") != contact.sha256
                    or checkpoint.get("acquisition_plan_sha256")
                    != plan["acquisition_plan_sha256"]
                ):
                    raise SecGemmaLeanAcquisitionError(
                        "existing_checkpoint_authority_changed"
                    )
            else:
                if prior_terminal is not None:
                    raise SecGemmaLeanAcquisitionError(
                        "terminal_checkpoint_missing"
                    )
                bundle, _, detail = _acquire_bundle(
                    repo_root=root,
                    plan=plan,
                    contact=contact,
                    readiness=readiness_before,
                    journal=journal,
                )
                pilot = _pilot_selection(
                    bundle["private_quarantine"]["model_requests"]
                )

                def build_manifest(
                    tree_bytes: bytes,
                    inventory: Mapping[str, int],
                ) -> dict[str, Any]:
                    return _build_checkpoint_manifest(
                        readiness=readiness_before,
                        plan=plan,
                        bundle=bundle,
                        validation=detail["validation"],
                        accounting=detail["accounting"],
                        tree_bytes=tree_bytes,
                        inventory=inventory,
                        pilot_selection=pilot,
                        source_resume=detail["source_resume"],
                        contact_sha256=contact.sha256,
                        started_at_utc=detail["source_started_at_utc"],
                        finished_at_utc=detail["source_finished_at_utc"],
                        elapsed_seconds=detail["source_elapsed_seconds"],
                    )

                _seal_private_checkpoint(
                    root,
                    bundle=bundle,
                    checkpoint_body_factory=build_manifest,
                    contact=contact,
                )
                del build_manifest, bundle, detail, pilot
                checkpoint, reloaded_bundle, reloaded_validation = (
                    _load_completed_checkpoint(root)
                )
                del reloaded_bundle, reloaded_validation
            readiness_after = _verify_acquisition_readiness(root)
            if readiness_after != readiness_before:
                raise SecGemmaLeanAcquisitionError(
                    "repository_changed_during_acquisition"
                )
            _verify_local_function_identities()
            _scan_completed_checkpoint_for_contact(
                private_target,
                checkpoint,
                contact,
            )
            if prior_terminal is None:
                journal.seal_terminal_success(checkpoint)
            else:
                terminal_payload = prior_terminal["payload"]
                if (
                    terminal_payload.get("status") != "passed"
                    or terminal_payload.get("checkpoint_sha256")
                    != checkpoint["checkpoint_sha256"]
                    or terminal_payload.get("bundle_sha256")
                    != checkpoint["bundle_sha256"]
                    or terminal_payload.get("effect_counts")
                    != checkpoint["effect_counts"]
                ):
                    raise SecGemmaLeanAcquisitionError(
                        "attempt_journal_terminal_invalid"
                    )
            receipt = _public_receipt(
                checkpoint=checkpoint,
                readiness_after=readiness_after,
                journal_summary=journal.summary(),
            )
            checkpoint_manifest_bytes = (
                private_target / PRIVATE_MANIFEST_NAME
            ).read_bytes()
            public_bytes = _plain_json_bytes(receipt)
            if any(
                needle in checkpoint_manifest_bytes or needle in public_bytes
                for needle in contact.serialized_echo_needles()
            ):
                raise SecGemmaLeanAcquisitionError("private_contact_leaked")
            _publish_public_receipt(root, checkpoint_root, receipt)
        except KeyboardInterrupt:
            if attempt_active and journal.terminal_event() is None:
                journal.interrupt_attempt()
            raise
        except SecGemmaLeanAcquisitionError as exc:
            if attempt_active and journal.terminal_event() is None:
                disposition = _failure_disposition(exc.code)
                if disposition in {
                    "terminal_suitability_rejection",
                    "terminal_integrity_rejection",
                }:
                    evidence = _failure_evidence_commitments(checkpoint_root)
                    journal.seal_terminal_failure(
                        disposition=disposition,
                        code=exc.code,
                        evidence_commitments=evidence,
                    )
                    terminal = journal.terminal_event()
                    if terminal is None:
                        raise SecGemmaLeanAcquisitionError(
                            "attempt_journal_terminal_invalid"
                        )
                    receipt = _public_failure_receipt(
                        terminal_event=terminal,
                        readiness=readiness_before,
                        plan=plan,
                        contact=contact,
                        journal_summary=journal.summary(),
                    )
                    _publish_public_receipt(root, checkpoint_root, receipt)
                    return receipt
                journal.finish_attempt(disposition=disposition, code=exc.code)
            raise
        except Exception:
            if attempt_active and journal.terminal_event() is None:
                journal.finish_attempt(
                    disposition="blocked_indeterminate",
                    code="unexpected_acquisition_failure",
                )
            raise
    return receipt


_FROZEN_LOCAL_FUNCTION_IDENTITIES: Final[dict[str, int]] = {
    "acquire_bundle": id(_acquire_bundle),
    "build_sec_session": id(_build_sec_session),
    "seal_private_checkpoint": id(_seal_private_checkpoint),
    "load_completed_checkpoint": id(_load_completed_checkpoint),
    "seal_sec_prefix": id(_seal_sec_prefix),
    "load_sec_prefix": id(_load_sec_prefix),
    "load_external_tree_checkpoint": id(_load_external_tree_checkpoint),
    "seal_external_tree_unit": id(_seal_external_tree_unit),
    "assert_value_redacted": id(_assert_value_redacted),
    "load_sec_catalog_checkpoint": id(_load_sec_catalog_checkpoint),
    "load_sec_document_chunk": id(_load_sec_document_chunk),
    "source_resume_evidence": id(_source_resume_evidence),
    "to_plain_json_value": id(_to_plain_json_value),
    "aggregate_document_batches": id(_aggregate_document_batches),
    "new_sec_unit": id(_new_sec_unit),
    "acquire_catalog_unit": id(_acquire_catalog_unit),
    "acquire_document_chunk_unit": id(_acquire_document_chunk_unit),
    "assert_checkpoint_authority": id(_assert_checkpoint_authority),
    "acquire_or_resume_sec_prefix": id(_acquire_or_resume_sec_prefix),
    "attempt_journal": id(_AttemptJournal),
    "install_sec_request_journal": id(_install_sec_request_journal),
    "journaled_yahoo_opener": id(_JournaledYahooOpener),
    "failure_disposition": id(_failure_disposition),
    "failure_evidence_commitments": id(_failure_evidence_commitments),
    "build_public_failure_receipt": id(_public_failure_receipt),
    "publish_public_receipt": id(_publish_public_receipt),
    "build_public_receipt": id(_public_receipt),
    "verify_acquisition_readiness": id(_verify_acquisition_readiness),
    "load_private_contact": id(_load_private_contact),
    "scan_completed_checkpoint_for_contact": id(
        _scan_completed_checkpoint_for_contact
    ),
    "recover_stale_acquisition_lock": id(_recover_stale_acquisition_lock),
    "run_development_acquisition": id(run_development_acquisition),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lean SEC/Gemma v3 development source acquisition"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
    )
    arguments = parser.parse_args(argv)
    try:
        report = run_development_acquisition(arguments.repo_root)
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "code": "user_interrupt"}))
        return 130
    except SecGemmaLeanAcquisitionError as exc:
        print(json.dumps({"status": "error", "code": exc.code}))
        return 1
    except Exception:
        print(
            json.dumps(
                {"status": "error", "code": "unexpected_acquisition_failure"}
            )
        )
        return 1
    output = {
        "status": report["status"],
        "artifact": PUBLIC_ARTIFACT_PATH,
        "acquisition_sha256": report["acquisition_sha256"],
        "model_generation_calls": report["effect_counts"][
            "model_generation_calls"
        ],
        "performance_results_opened": report["effect_counts"][
            "performance_results_opened"
        ],
        "next_step": report["next_step"],
    }
    if report["status"] == "passed":
        output.update(
            {
                "filing_count": report["suitability"]["filing_count"],
                "market_response_count": report["suitability"][
                    "market_response_count"
                ],
            }
        )
        exit_code = 0
    else:
        output["rejection"] = report["rejection"]
        exit_code = 2
    print(json.dumps(output, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PUBLIC_ARTIFACT_PATH",
    "SecGemmaLeanAcquisitionError",
    "run_development_acquisition",
]
