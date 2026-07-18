"""Disk-backed zero-retry SEC acquisition for lean Gemma evidence v3.6.

The orchestrator deliberately owns no HTTP implementation and no SEC parser.
Those two boundaries are injected.  Its job is narrower: pace every dispatch,
make each successful byte effect durable, bind the transport/source receipts to
the append-only journal, and prove the finished inventory by detached replay.

Raw responses, the append-only journal, and its compact checkpoint live only in
the ignored ``data/...`` root.  The tracked ``e/...`` root receives one final
redacted source-authority receipt after detached replay; neither side may contain
the SEC contact text.  A crash never authorizes adoption or a resend.  Any
incomplete intent or orphan artifact therefore fails closed for human inspection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Final, Protocol

import agent_benchmark.sec_gemma_lean_v36_journal as journal_module
from agent_benchmark.sec_gemma_lean_v36_journal import (
    AcquisitionJournal,
    CompleteSubmissionTarget,
    FIRST_DISPATCH_WAIT_MS,
    GLOBAL_DISPATCH_GAP_MS,
    INVOCATION_DEADLINE_MS,
    PrivateSecContact,
    REQUEST_DEADLINE_MS,
    ROLE_DEADLINE_MS,
    SOURCE_DIAGNOSTIC_CODES,
    STAGE_ACTIVE_TIME_CAP_MS,
    RequestRole,
    SecGemmaLeanV36JournalError,
    validate_detached_journal,
)
from agent_benchmark.sec_gemma_lean_v36_source import (
    SecGemmaLeanV36SourceError,
)
from agent_benchmark.sec_gemma_lean_v36_transport import (
    SecGemmaLeanV36TransportError,
    StrictSecTransport,
    TransportResult,
    validate_transport_receipt,
)


ACQUISITION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-6-acquisition-v1"
)
PARSE_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-6-parse-receipt-v1"
)
PHASE_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-6-phase-receipt-v1"
)
CHUNK_BYTES: Final[int] = 64 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:(?P<digest>[0-9a-f]{64})\Z")
_ROLE_ARTIFACT_RE = re.compile(
    r"(?P<sequence>[0-9]{6})-(?P<sha>[0-9a-f]{64})\.(?P<suffix>json|blob)\Z"
)
_PHASE_ARTIFACT_RE = re.compile(
    r"(?P<phase>submissions|reconciliation)-(?P<sha>[0-9a-f]{64})\.json\Z"
)
_CHECKPOINT_RE = re.compile(r"checkpoint-(?P<sha>[0-9a-f]{64})\.json\Z")
_PUBLIC_AUTHORITY_RE = re.compile(
    r"source-authority-(?P<sha>[0-9a-f]{64})\.json\Z"
)
_REPARSE_ATTRIBUTE: Final[int] = 0x400
_STAGES: Final[tuple[str, ...]] = ("development", "intermediate", "final")
ROOT_NAMESPACE: Final[str] = "aapl_sec_gemma_lean_evidence_v3_6"
GLOBAL_LEDGER_NAME: Final[str] = "global_dispatch.json"
PRODUCTION_PREFLIGHT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-evidence-v3-6-preflight-v1"
)
PRODUCTION_PREFLIGHT_PATH: Final[str] = (
    "e/aapl_sec_gemma_lean_evidence_v3_6/PREFLIGHT.json"
)
PRODUCTION_BRANCH: Final[str] = "codex/aapl-sec-gemma-lean-evidence-v3-6"
PRODUCTION_UPSTREAM: Final[str] = f"origin/{PRODUCTION_BRANCH}"
PRIVATE_CONFIG_PATH: Final[str] = "data/local_config.json"
PRODUCTION_MODULE_CLOSURE: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "agent_benchmark/__init__.py",
            "agent_benchmark/sec_filing_content.py",
            "agent_benchmark/sec_filing_gemma_extractor_prompt.py",
            "agent_benchmark/sec_filing_gemma_extractor_schema.py",
            "agent_benchmark/sec_gemma_lean_v36_acquisition.py",
            "agent_benchmark/sec_gemma_lean_v36_delta.py",
            "agent_benchmark/sec_gemma_lean_v36_journal.py",
            "agent_benchmark/sec_gemma_lean_v36_preflight.py",
            "agent_benchmark/sec_gemma_lean_v36_source.py",
            "agent_benchmark/sec_gemma_lean_v36_transport.py",
            "agent_benchmark/sec_gemma_online_risk_overlay_contract.py",
            "agent_benchmark/sec_gemma_online_risk_overlay_runtime.py",
            "agent_benchmark/sec_point_in_time.py",
            "agent_benchmark/sec_session_calendar.py",
        }
    )
)


def _repository_root() -> Path:
    try:
        return Path(__file__).resolve(strict=True).parents[1]
    except (OSError, IndexError):
        raise SecGemmaLeanV36AcquisitionError("root_invalid") from None


class SecGemmaLeanV36AcquisitionError(RuntimeError):
    """A fixed-code acquisition rejection with no third-party text."""

    def __init__(self, code: str) -> None:
        if type(code) is not str or re.fullmatch(r"[a-z0-9_]{3,80}", code) is None:
            code = "acquisition_error"
        super().__init__(code)
        self.code = code


def _source_diagnostic_code(error: BaseException) -> str:
    """Return the sole safe source diagnostic without exposing other text."""

    if type(error) is SecGemmaLeanV36SourceError:
        message = str(error)
        if message in SOURCE_DIAGNOSTIC_CODES:
            return message
    return "source_rejected"


def _bare_manifest_body_sha256(
    value: object,
    *,
    error_code: str = "artifact_inventory_invalid",
) -> str:
    """Authenticate the compact manifest tag and return its bare digest."""

    match = (
        _TAGGED_SHA256_RE.fullmatch(value)
        if type(value) is str
        else None
    )
    if match is None:
        raise SecGemmaLeanV36AcquisitionError(error_code)
    return match.group("digest")


class SourceAdapter(Protocol):
    """The compact source surface consumed by this orchestrator."""

    CompactReplayInput: type[Any]

    def parse_main(self, stage: str, payload: bytes) -> Any: ...

    def parse_historical(
        self, stage: str, filename: str, payload: bytes, main_evidence: Mapping[str, Any]
    ) -> Any: ...

    def finalize_submissions(
        self,
        stage: str,
        main_evidence: Mapping[str, Any],
        historical_evidence: Sequence[Mapping[str, Any]],
        prior: Any | None,
    ) -> Any: ...

    def parse_master(
        self, stage: str, year: int, quarter: int, payload: bytes
    ) -> Any: ...

    def reconcile_masters(
        self,
        stage: str,
        submissions_evidence: Mapping[str, Any],
        master_evidence: Sequence[Mapping[str, Any]],
        prior: Any | None,
    ) -> Any: ...

    def parse_complete(
        self,
        stage: str,
        target: Mapping[str, Any],
        payload: bytes,
        reconciliation_evidence: Mapping[str, Any],
    ) -> Any: ...

    def finalize_stage(
        self,
        stage: str,
        submissions_evidence: Mapping[str, Any],
        reconciliation_evidence: Mapping[str, Any],
        complete_evidence: Sequence[Mapping[str, Any]],
        prior: Any | None,
    ) -> Any: ...

    def build_compact_role_manifest(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def build_compact_checkpoint(self, **kwargs: Any) -> Mapping[str, Any]: ...

    def detached_replay(
        self,
        stage: str,
        checkpoint: Mapping[str, Any],
        role_manifests: Sequence[Mapping[str, Any]],
        load_blob: Callable[[str], bytes],
        prior: Any | None = None,
    ) -> Mapping[str, Any]: ...

    def rehydrate_compact_stage(
        self,
        stage: str,
        checkpoint: Mapping[str, Any],
        role_manifests: Sequence[Mapping[str, Any]],
        load_blob: Callable[[str], bytes],
        prior: Any | None = None,
    ) -> Any: ...


class Transport(Protocol):
    def safe_state(self) -> Mapping[str, Any]: ...

    def fetch(
        self,
        url: str,
        *,
        role_id: str,
        intent_event_sha256: str,
        body_limit: int,
    ) -> TransportResult: ...


TransportFactory = Callable[[Path], Transport]


@dataclass(frozen=True)
class DispatchReceipt:
    purpose: str
    waited_milliseconds: int
    dispatched_monotonic: float
    intent_event_sha256: str
    callback_result: Any = field(repr=False)
    callback_error: Exception | None = field(repr=False)


class SharedSecDispatchLedger:
    """One process-wide lock and clock for all SEC purposes and stages."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        ledger_path: Path | None = None,
        lock_timeout_seconds: float = 35.0,
    ) -> None:
        if not callable(monotonic) or not callable(sleep):
            raise SecGemmaLeanV36AcquisitionError("dispatch_clock_invalid")
        if (
            type(lock_timeout_seconds) not in {int, float}
            or not math.isfinite(float(lock_timeout_seconds))
            or not 0.0 <= float(lock_timeout_seconds) <= 35.0
        ):
            raise SecGemmaLeanV36AcquisitionError("dispatch_clock_invalid")
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._ledger_path = (
            _repository_root() / "data" / ROOT_NAMESPACE / GLOBAL_LEDGER_NAME
            if ledger_path is None
            else ledger_path
        )
        if (
            not isinstance(self._ledger_path, Path)
            or not self._ledger_path.is_absolute()
        ):
            raise SecGemmaLeanV36AcquisitionError("dispatch_ledger_invalid")
        self._last_dispatch: float | None = None

    @property
    def ledger_path(self) -> Path:
        """The exact durable state path used by this ledger."""

        return self._ledger_path

    def dispatch(
        self,
        *,
        purpose: str,
        first_in_invocation: bool,
        record_intent: Callable[[int], str],
        dispatch_callback: Callable[[str], Any],
    ) -> DispatchReceipt:
        if (
            type(purpose) is not str
            or re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", purpose) is None
            or type(first_in_invocation) is not bool
            or not callable(record_intent)
            or not callable(dispatch_callback)
        ):
            raise SecGemmaLeanV36AcquisitionError("dispatch_request_invalid")
        entered = self._read_clock()
        if not _PROCESS_DISPATCH_LOCK.acquire(
            timeout=self._lock_timeout_seconds
        ):
            raise SecGemmaLeanV36AcquisitionError("dispatch_lock_conflict")
        try:
            with _locked_file(
                self._ledger_path.with_suffix(".lock"),
                busy_code="dispatch_lock_conflict",
                wait_timeout_seconds=self._lock_timeout_seconds,
            ):
                return self._dispatch_locked(
                    entered=entered,
                    purpose=purpose,
                    first_in_invocation=first_in_invocation,
                    record_intent=record_intent,
                    dispatch_callback=dispatch_callback,
                )
        finally:
            try:
                _PROCESS_DISPATCH_LOCK.release()
            except RuntimeError:
                raise SecGemmaLeanV36AcquisitionError(
                    "dispatch_lock_release_failed"
                ) from None

    def _dispatch_locked(
        self,
        *,
        entered: float,
        purpose: str,
        first_in_invocation: bool,
        record_intent: Callable[[int], str],
        dispatch_callback: Callable[[str], Any],
    ) -> DispatchReceipt:
        now = self._read_clock()
        persisted, in_flight = self._read_persisted_dispatch()
        if in_flight:
            raise SecGemmaLeanV36AcquisitionError("dispatch_indeterminate")
        prior_dispatch = self._last_dispatch
        if persisted is not None:
            prior_dispatch = (
                persisted
                if prior_dispatch is None
                else max(prior_dispatch, persisted)
            )
        if prior_dispatch is not None and prior_dispatch > now:
            # A reboot resets monotonic clocks.  Every new acquisition
            # invocation still waits the full first-dispatch second.
            prior_dispatch = None
        local_delay = (
            FIRST_DISPATCH_WAIT_MS / 1000.0 if first_in_invocation else 0.0
        )
        target = entered + local_delay
        if prior_dispatch is not None:
            target = max(
                target,
                prior_dispatch + GLOBAL_DISPATCH_GAP_MS / 1000.0,
            )
        if target > now:
            try:
                self._sleep(target - now)
            except Exception:
                raise SecGemmaLeanV36AcquisitionError(
                    "dispatch_clock_invalid"
                ) from None
        ready = self._read_clock()
        if ready + 1e-9 < target:
            raise SecGemmaLeanV36AcquisitionError("dispatch_gap_too_short")
        wait_origin = entered if first_in_invocation else prior_dispatch
        if wait_origin is None:
            raise SecGemmaLeanV36AcquisitionError("dispatch_gap_too_short")
        waited_ms = int(math.ceil(max(0.0, ready - wait_origin) * 1000.0))
        minimum = (
            FIRST_DISPATCH_WAIT_MS
            if first_in_invocation
            else GLOBAL_DISPATCH_GAP_MS
        )
        if waited_ms < minimum:
            # A non-first call can only have no predecessor if a foreign
            # caller tries to bypass the normal invocation contract.
            raise SecGemmaLeanV36AcquisitionError("dispatch_gap_too_short")
        result = record_intent(waited_ms)
        if type(result) is not str or _SHA256_RE.fullmatch(result) is None:
            raise SecGemmaLeanV36AcquisitionError("journal_binding_invalid")
        self._write_persisted_dispatch(
            prior_dispatch,
            in_flight=True,
        )
        dispatched = self._read_clock()
        if dispatched < ready:
            raise SecGemmaLeanV36AcquisitionError("dispatch_clock_invalid")
        callback_result: Any = None
        callback_error: Exception | None = None
        try:
            callback_result = dispatch_callback(result)
        except Exception as error:
            callback_error = error
        finally:
            self._last_dispatch = dispatched
            self._write_persisted_dispatch(
                dispatched,
                in_flight=False,
            )
        return DispatchReceipt(
            purpose=purpose,
            waited_milliseconds=waited_ms,
            dispatched_monotonic=dispatched,
            intent_event_sha256=result,
            callback_result=callback_result,
            callback_error=callback_error,
        )

    def _read_clock(self) -> float:
        try:
            value = float(self._monotonic())
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("dispatch_clock_invalid") from None
        if not math.isfinite(value) or value < 0.0:
            raise SecGemmaLeanV36AcquisitionError("dispatch_clock_invalid")
        return value

    def _read_persisted_dispatch(self) -> tuple[float | None, bool]:
        try:
            parent = _secure_directory(self._ledger_path.parent)
            marker_prefixes = (
                f".{self._ledger_path.name}.pending-",
                f".{self._ledger_path.name}.failed-",
            )
            if any(
                entry.name.startswith(marker_prefixes)
                for entry in parent.iterdir()
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "dispatch_indeterminate"
                )
            try:
                before = self._ledger_path.lstat()
            except FileNotFoundError:
                return None, False
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or _is_reparse(before)
                or before.st_nlink != 1
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "dispatch_ledger_invalid"
                )
            with self._ledger_path.open("rb") as handle:
                opened = os.fstat(handle.fileno())
                if (
                    opened.st_dev != before.st_dev
                    or opened.st_ino != before.st_ino
                    or not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                ):
                    raise SecGemmaLeanV36AcquisitionError(
                        "dispatch_ledger_invalid"
                    )
                raw = handle.read()
            after = self._ledger_path.lstat()
            if (
                after.st_dev != opened.st_dev
                or after.st_ino != opened.st_ino
                or after.st_size != len(raw)
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "dispatch_ledger_invalid"
                )
            value = json.loads(raw.decode("ascii", errors="strict"))
            if type(value) is not dict or _canonical_bytes(value) != raw:
                raise SecGemmaLeanV36AcquisitionError(
                    "dispatch_ledger_invalid"
                )
        except SecGemmaLeanV36AcquisitionError as error:
            if error.code == "dispatch_indeterminate":
                raise
            raise SecGemmaLeanV36AcquisitionError(
                "dispatch_ledger_invalid"
            ) from None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise SecGemmaLeanV36AcquisitionError("dispatch_ledger_invalid") from None
        if (
            type(value) is not dict
            or set(value)
            != {"schema_version", "last_dispatch_monotonic", "in_flight"}
            or value["schema_version"] != ACQUISITION_SCHEMA_VERSION
            or (
                value["last_dispatch_monotonic"] is not None
                and type(value["last_dispatch_monotonic"]) not in {int, float}
            )
            or type(value["in_flight"]) is not bool
        ):
            raise SecGemmaLeanV36AcquisitionError("dispatch_ledger_invalid")
        if value["last_dispatch_monotonic"] is None:
            return None, value["in_flight"]
        observed = float(value["last_dispatch_monotonic"])
        if not math.isfinite(observed) or observed < 0.0:
            raise SecGemmaLeanV36AcquisitionError("dispatch_ledger_invalid")
        return observed, value["in_flight"]

    def _write_persisted_dispatch(
        self, value: float | None, *, in_flight: bool
    ) -> None:
        if (
            value is not None
            and (not math.isfinite(value) or value < 0.0)
        ) or type(in_flight) is not bool:
            raise SecGemmaLeanV36AcquisitionError("dispatch_ledger_invalid")
        payload = _canonical_bytes(
            {
                "schema_version": ACQUISITION_SCHEMA_VERSION,
                "last_dispatch_monotonic": value,
                "in_flight": in_flight,
            }
        )
        pending = self._ledger_path.parent / (
            f".{self._ledger_path.name}.pending-"
            f"{os.getpid()}-{threading.get_ident()}"
        )
        renamed = False
        try:
            parent = _secure_directory(self._ledger_path.parent)
            if any(
                entry.name.startswith(
                    (
                        f".{self._ledger_path.name}.pending-",
                        f".{self._ledger_path.name}.failed-",
                    )
                )
                for entry in parent.iterdir()
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "dispatch_indeterminate"
                )
            try:
                existing = self._ledger_path.lstat()
            except FileNotFoundError:
                existing = None
            if existing is not None and (
                not stat.S_ISREG(existing.st_mode)
                or stat.S_ISLNK(existing.st_mode)
                or _is_reparse(existing)
                or existing.st_nlink != 1
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "dispatch_ledger_invalid"
                )
            with pending.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            pending_details = _secure_regular_file(pending)
            if (
                pending_details.st_size != len(payload)
                or pending.read_bytes() != payload
            ):
                raise OSError("dispatch pending changed")
            os.replace(pending, self._ledger_path)
            renamed = True
            state_details = _secure_regular_file(self._ledger_path)
            if (
                state_details.st_size != len(payload)
                or self._ledger_path.read_bytes() != payload
            ):
                raise OSError("dispatch state changed")
            journal_module._fsync_directory(parent)
        except SecGemmaLeanV36AcquisitionError as error:
            if renamed:
                failure = self._ledger_path.parent / (
                    f".{self._ledger_path.name}.failed-"
                    f"{os.getpid()}-{threading.get_ident()}"
                )
                try:
                    with failure.open("xb") as handle:
                        handle.write(b"dispatch persistence indeterminate\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                except Exception:
                    pass
            if error.code == "dispatch_indeterminate":
                raise
            raise SecGemmaLeanV36AcquisitionError(
                "dispatch_ledger_invalid"
            ) from None
        except Exception:
            if renamed:
                failure = self._ledger_path.parent / (
                    f".{self._ledger_path.name}.failed-"
                    f"{os.getpid()}-{threading.get_ident()}"
                )
                try:
                    with failure.open("xb") as handle:
                        handle.write(b"dispatch persistence indeterminate\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    journal_module._fsync_directory(failure.parent)
                except Exception:
                    pass
            raise SecGemmaLeanV36AcquisitionError(
                "dispatch_ledger_invalid"
            ) from None


GLOBAL_SEC_DISPATCH_LEDGER = SharedSecDispatchLedger()


_PROCESS_DISPATCH_LOCK = threading.Lock()


@dataclass(frozen=True)
class _Layout:
    private_root: Path
    public_root: Path
    private_stage: Path
    public_stage: Path
    journal: Path
    temporary: Path
    blobs: Path
    transport_receipts: Path
    parse_receipts: Path
    manifests: Path
    phase_receipts: Path
    run_lock: Path


@dataclass(frozen=True)
class _RoleRecord:
    manifest: Mapping[str, Any]
    parse_receipt: Mapping[str, Any]
    transport_receipt: Mapping[str, Any]
    blob_path: Path


@dataclass(frozen=True)
class AcquisitionResult:
    stage: str
    checkpoint_path: Path
    checkpoint_file_sha256: str
    checkpoint: Mapping[str, Any]
    replay_receipt: Mapping[str, Any]
    journal_state: Any
    stage_output: Any
    compact_replay_input: Any
    public_authority_path: Path
    accounting_complete: bool
    source_authoritative: bool
    blob_paths: Mapping[str, Path] = field(repr=False)


@dataclass(frozen=True)
class _ProductionDevelopmentGate:
    repository_root: Path
    private_root: Path
    public_root: Path
    authorized_head: str
    preflight_sha256: str
    private_contact: str = field(repr=False)


class _TerminalRoleFailure(Exception):
    def __init__(self, code: str, role_duration_ms: int) -> None:
        super().__init__(code)
        self.code = code
        self.role_duration_ms = role_duration_ms


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


@contextmanager
def _locked_file(
    path: Path,
    *,
    busy_code: str,
    wait_timeout_seconds: float = 0.0,
):
    """Hold one blocking one-byte advisory lock on Windows or POSIX."""

    handle = None
    created = False
    try:
        _secure_directory(path.parent)
        try:
            before = path.lstat()
        except FileNotFoundError:
            before = None
        if before is None:
            try:
                handle = path.open("x+b")
                created = True
            except FileExistsError:
                before = path.lstat()
                handle = path.open("r+b")
        else:
            handle = path.open("r+b")
        opened = os.fstat(handle.fileno())
        after = path.lstat()
        expected = opened if before is None else before
        if (
            not stat.S_ISREG(expected.st_mode)
            or stat.S_ISLNK(expected.st_mode)
            or _is_reparse(expected)
            or expected.st_nlink != 1
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or expected.st_dev != opened.st_dev
            or expected.st_ino != opened.st_ino
        ):
            raise OSError("unsafe lock path")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        elif handle.tell() != 1:
            raise OSError("invalid lock size")
        handle.seek(0)
        if created:
            journal_module._fsync_directory(path.parent)
    except OSError:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        raise SecGemmaLeanV36AcquisitionError("run_lock_invalid") from None
    except (SecGemmaLeanV36AcquisitionError, SecGemmaLeanV36JournalError):
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        raise SecGemmaLeanV36AcquisitionError("run_lock_invalid") from None
    acquired = False
    deadline = time.monotonic() + max(0.0, wait_timeout_seconds)
    while not acquired:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            if time.monotonic() >= deadline:
                handle.close()
                raise SecGemmaLeanV36AcquisitionError(busy_code) from None
            time.sleep(0.01)
    try:
        yield
    finally:
        release_failed = False
        try:
            handle.seek(0)
            if acquired and os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif acquired:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            release_failed = True
        try:
            handle.close()
        except OSError:
            release_failed = True
        if release_failed:
            raise SecGemmaLeanV36AcquisitionError(
                "run_lock_release_failed"
            ) from None


def _secure_directory(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise SecGemmaLeanV36AcquisitionError("root_invalid")
    try:
        resolved = path.resolve(strict=True)
        details = path.lstat()
    except OSError:
        raise SecGemmaLeanV36AcquisitionError("root_invalid") from None
    if (
        resolved != path
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
    ):
        raise SecGemmaLeanV36AcquisitionError("root_invalid")
    return path


def _secure_regular_file(path: Path) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError:
        raise SecGemmaLeanV36AcquisitionError("artifact_invalid") from None
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
    ):
        raise SecGemmaLeanV36AcquisitionError("artifact_invalid")
    return details


def _canonical_bytes(value: Any) -> bytes:
    pending = [value]
    while pending:
        current = pending.pop()
        if type(current) is dict:
            if any(type(key) is not str for key in current):
                raise SecGemmaLeanV36AcquisitionError("non_json_source_output")
            pending.extend(current.values())
        elif type(current) in {list, tuple}:
            pending.extend(current)
        elif current is None or type(current) in {str, bool, int}:
            continue
        elif type(current) is float and math.isfinite(current):
            continue
        else:
            raise SecGemmaLeanV36AcquisitionError("non_json_source_output")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        json.loads(encoded.decode("ascii"))
        return encoded
    except (TypeError, ValueError, UnicodeError):
        raise SecGemmaLeanV36AcquisitionError("non_json_source_output") from None


def _canonical_mapping(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecGemmaLeanV36AcquisitionError("non_json_source_output")
    return json.loads(_canonical_bytes(value).decode("ascii"))


def _build_public_authority_value(
    *,
    stage: str,
    checkpoint_file_sha256: str,
    checkpoint: Mapping[str, Any],
    replay_receipt: Mapping[str, Any],
    journal_preterminal_head_sha256: str,
    journal_state: Any,
) -> dict[str, Any]:
    return _canonical_mapping(
        {
            "schema_version": ACQUISITION_SCHEMA_VERSION,
            "stage": stage,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "stage_source_seal_sha256": checkpoint[
                "expected_stage_source_seal_sha256"
            ],
            "role_manifest_count": checkpoint["role_manifest_count"],
            "role_manifests_sha256": checkpoint["role_manifests_sha256"],
            "journal_preterminal_head_sha256": (
                journal_preterminal_head_sha256
            ),
            "role_plan_sha256": journal_state.role_plan_sha256,
            "formula_requests": journal_state.formula_requests,
            "source_replay_receipt": dict(replay_receipt),
            "source_authoritative": True,
            "peak_live_role_payload_count": 1,
            "contains_private_paths": False,
            "contains_contact_text": False,
        }
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, maximum_bytes: int | None = None) -> tuple[str, int]:
    _secure_regular_file(path)
    digest = hashlib.sha256()
    counted = 0
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(CHUNK_BYTES)
                if not chunk:
                    break
                counted += len(chunk)
                if maximum_bytes is not None and counted > maximum_bytes:
                    raise SecGemmaLeanV36AcquisitionError("artifact_invalid")
                digest.update(chunk)
    except SecGemmaLeanV36AcquisitionError:
        raise
    except OSError:
        raise SecGemmaLeanV36AcquisitionError("artifact_invalid") from None
    return digest.hexdigest(), counted


def _read_canonical_mapping(path: Path) -> dict[str, Any]:
    details = _secure_regular_file(path)
    if not 2 <= details.st_size <= 512 * 1024 * 1024:
        raise SecGemmaLeanV36AcquisitionError("artifact_invalid")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanV36AcquisitionError("artifact_invalid") from None
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise SecGemmaLeanV36AcquisitionError("artifact_invalid")
    return value


def _write_immutable(path: Path, payload: bytes) -> None:
    if type(payload) is not bytes or not payload:
        raise SecGemmaLeanV36AcquisitionError("artifact_persistence_failed")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        details = _secure_regular_file(path)
        if details.st_size != len(payload) or path.read_bytes() != payload:
            raise OSError("changed")
        journal_module._fsync_directory(path.parent)
    except SecGemmaLeanV36AcquisitionError:
        raise
    except (OSError, SecGemmaLeanV36JournalError):
        raise SecGemmaLeanV36AcquisitionError(
            "artifact_persistence_failed"
        ) from None


def _persist_hashed_json(directory: Path, stem: str, value: Mapping[str, Any]) -> tuple[Path, str]:
    payload = _canonical_bytes(dict(value))
    digest = _sha256_bytes(payload)
    path = directory / f"{stem}-{digest}.json"
    _write_immutable(path, payload)
    observed, length = _sha256_file(path)
    if observed != digest or length != len(payload):
        raise SecGemmaLeanV36AcquisitionError("artifact_persistence_failed")
    return path, digest


def _mkdir(path: Path) -> None:
    try:
        created = not path.exists()
        path.mkdir()
        if created:
            journal_module._fsync_directory(path.parent)
    except FileExistsError:
        pass
    except (OSError, SecGemmaLeanV36JournalError):
        raise SecGemmaLeanV36AcquisitionError("root_invalid") from None
    _secure_directory(path)


def _prepare_layout(private_root: Path, public_root: Path, stage: str) -> _Layout:
    private_root = _secure_directory(private_root)
    public_root = _secure_directory(public_root)
    try:
        if private_root == public_root or private_root in public_root.parents or public_root in private_root.parents:
            raise SecGemmaLeanV36AcquisitionError("root_overlap")
        repo_root = _repository_root()
        private_relative = private_root.relative_to(repo_root)
        public_relative = public_root.relative_to(repo_root)
    except ValueError:
        raise SecGemmaLeanV36AcquisitionError("root_namespace_invalid") from None
    if private_relative.parts != ("data", ROOT_NAMESPACE) or public_relative.parts != (
        "e",
        ROOT_NAMESPACE,
    ):
        raise SecGemmaLeanV36AcquisitionError("root_namespace_invalid")
    try:
        ignore_lines = {
            line.strip()
            for line in (repo_root / ".gitignore").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError:
        raise SecGemmaLeanV36AcquisitionError("private_root_not_ignored") from None
    if "data/" not in ignore_lines and "/data/" not in ignore_lines:
        raise SecGemmaLeanV36AcquisitionError("private_root_not_ignored")
    private_stage = private_root / stage
    public_stage = public_root / stage
    _mkdir(private_stage)
    _mkdir(public_stage)
    names = (
        "temporary",
        "blobs",
        "transport_receipts",
        "parse_receipts",
        "manifests",
        "phase_receipts",
    )
    private_directories = {name: private_stage / name for name in names}
    for path in private_directories.values():
        _mkdir(path)
    journal = private_stage / "journal"
    _mkdir(journal)
    run_lock = private_stage / ".run.lock"
    return _Layout(
        private_root=private_root,
        public_root=public_root,
        private_stage=private_stage,
        public_stage=public_stage,
        journal=journal,
        temporary=private_directories["temporary"],
        blobs=private_directories["blobs"],
        transport_receipts=private_directories["transport_receipts"],
        parse_receipts=private_directories["parse_receipts"],
        manifests=private_directories["manifests"],
        phase_receipts=private_directories["phase_receipts"],
        run_lock=run_lock,
    )


def strict_transport_factory(private_contact: str) -> TransportFactory:
    """Return a closure that creates the reviewed production transport."""

    if type(private_contact) is not str:
        raise SecGemmaLeanV36AcquisitionError("private_contact_invalid")

    def factory(temporary_directory: Path) -> StrictSecTransport:
        return StrictSecTransport(
            private_contact=private_contact,
            temporary_directory=temporary_directory,
        )

    return factory


class DiskBackedSecAcquisition:
    """One exact H -> Q -> P stage acquisition with durable resume checks."""

    def __init__(
        self,
        *,
        stage: str,
        private_root: Path,
        public_root: Path,
        private_contact: str,
        transport_factory: TransportFactory,
        source_adapter: SourceAdapter,
        purpose: str = "lean-v36-stage-source",
        prior: AcquisitionResult | None = None,
        dispatch_ledger: SharedSecDispatchLedger = GLOBAL_SEC_DISPATCH_LEDGER,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if stage not in _STAGES:
            raise SecGemmaLeanV36AcquisitionError("stage_invalid")
        if (
            type(purpose) is not str
            or re.fullmatch(r"[A-Za-z0-9._/-]{1,120}", purpose) is None
            or not callable(transport_factory)
            or not callable(monotonic)
            or not isinstance(dispatch_ledger, SharedSecDispatchLedger)
        ):
            raise SecGemmaLeanV36AcquisitionError("configuration_invalid")
        if (
            not isinstance(private_root, Path)
            or dispatch_ledger.ledger_path != private_root / GLOBAL_LEDGER_NAME
        ):
            raise SecGemmaLeanV36AcquisitionError(
                "dispatch_ledger_namespace_invalid"
            )
        try:
            contact = PrivateSecContact(private_contact)
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("private_contact_invalid") from None
        self._stage = stage
        self._purpose = purpose
        self._contact = contact
        self._layout = _prepare_layout(private_root, public_root, stage)
        self._source = source_adapter
        self._dispatch_ledger = dispatch_ledger
        self._monotonic = monotonic
        self._run_lock = threading.Lock()
        self._prior = self._validate_prior(prior)
        self._validate_source_surface()
        try:
            transport = transport_factory(self._layout.temporary)
            safe_state = _canonical_mapping(dict(transport.safe_state()))
        except SecGemmaLeanV36AcquisitionError:
            raise
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("transport_invalid") from None
        if (
            safe_state.get("temporary_directory") != str(self._layout.temporary)
            or safe_state.get("contact_fingerprint_sha256")
            != contact.fingerprint_sha256
        ):
            raise SecGemmaLeanV36AcquisitionError("transport_invalid")
        self._transport = transport
        self._contact.assert_redacted(safe_state)
        try:
            self._journal = AcquisitionJournal(
                self._layout.journal,
                stage=stage,
                contact=contact,
            )
        except SecGemmaLeanV36JournalError as error:
            raise SecGemmaLeanV36AcquisitionError(error.code) from None
        self._audit_layout_names()

    def __repr__(self) -> str:
        return (
            "DiskBackedSecAcquisition("
            f"stage={self._stage!r}, purpose={self._purpose!r}, "
            f"contact_fingerprint_sha256={self._contact.fingerprint_sha256!r})"
        )

    @property
    def journal_state(self) -> Any:
        return self._journal.state

    def _validate_prior(
        self, prior: AcquisitionResult | None
    ) -> AcquisitionResult | None:
        index = _STAGES.index(self._stage)
        if index == 0:
            if prior is not None:
                raise SecGemmaLeanV36AcquisitionError("prior_stage_invalid")
            return None
        if (
            type(prior) is not AcquisitionResult
            or prior.stage != _STAGES[index - 1]
            or not prior.accounting_complete
            or not prior.source_authoritative
            or prior.stage_output is None
            or prior.compact_replay_input is None
        ):
            raise SecGemmaLeanV36AcquisitionError("prior_stage_invalid")
        return prior

    def _validate_source_surface(self) -> None:
        methods = (
            "parse_main",
            "parse_historical",
            "finalize_submissions",
            "parse_master",
            "reconcile_masters",
            "parse_complete",
            "finalize_stage",
            "build_compact_role_manifest",
            "build_compact_checkpoint",
            "detached_replay",
            "rehydrate_compact_stage",
        )
        if self._source is None or any(
            not callable(getattr(self._source, name, None)) for name in methods
        ) or not isinstance(getattr(self._source, "CompactReplayInput", None), type):
            raise SecGemmaLeanV36AcquisitionError("source_adapter_invalid")

    def _now(self) -> float:
        try:
            value = float(self._monotonic())
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("clock_invalid") from None
        if not math.isfinite(value) or value < 0.0:
            raise SecGemmaLeanV36AcquisitionError("clock_invalid")
        return value

    def _audit_layout_names(self) -> None:
        private_expected = {
            "temporary",
            "blobs",
            "transport_receipts",
            "parse_receipts",
            "manifests",
            "phase_receipts",
            "journal",
        }
        try:
            private_entries = list(self._layout.private_stage.iterdir())
            public_entries = list(self._layout.public_stage.iterdir())
        except OSError:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid") from None
        private_directories = {
            item.name for item in private_entries if item.name in private_expected
        }
        if private_directories != private_expected:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        for item in private_entries:
            if item.name in private_expected:
                _secure_directory(item)
            elif item.name == ".run.lock":
                _secure_regular_file(item)
            elif _CHECKPOINT_RE.fullmatch(item.name) is not None:
                _secure_regular_file(item)
            else:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
        for item in public_entries:
            if _PUBLIC_AUTHORITY_RE.fullmatch(item.name) is not None:
                _secure_regular_file(item)
            else:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )

    def _files(self, directory: Path) -> list[Path]:
        _secure_directory(directory)
        try:
            values = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid") from None
        for value in values:
            _secure_regular_file(value)
        return values

    def _validate_source_evidence(self, raw: Any) -> dict[str, Any]:
        evidence = _canonical_mapping(raw)
        supplied = evidence.get("source_evidence_sha256")
        unsigned = dict(evidence)
        unsigned.pop("source_evidence_sha256", None)
        if (
            type(supplied) is not str
            or _SHA256_RE.fullmatch(supplied) is None
            or _sha256_bytes(_canonical_bytes(unsigned)) != supplied
        ):
            raise SecGemmaLeanV36AcquisitionError("source_evidence_invalid")
        self._contact.assert_redacted(evidence)
        return evidence

    def _parse_output_parts(
        self, output: Any
    ) -> tuple[dict[str, Any], tuple[str, ...], int | None]:
        try:
            evidence = self._validate_source_evidence(output.evidence)
            historical = output.historical_filenames
            decompressed = output.decompressed_bytes
        except SecGemmaLeanV36AcquisitionError:
            raise
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("source_output_invalid") from None
        if (
            type(historical) is not tuple
            or any(type(name) is not str for name in historical)
            or tuple(sorted(historical)) != historical
            or len(historical) != len(set(historical))
            or (
                decompressed is not None
                and (type(decompressed) is not int or decompressed < 0)
            )
        ):
            raise SecGemmaLeanV36AcquisitionError("source_output_invalid")
        try:
            visible = vars(output)
        except TypeError:
            raise SecGemmaLeanV36AcquisitionError("source_output_invalid") from None
        if set(visible) != {
            "evidence",
            "historical_filenames",
            "decompressed_bytes",
        }:
            raise SecGemmaLeanV36AcquisitionError("source_output_invalid")
        _canonical_bytes(
            {
                "evidence": evidence,
                "historical_filenames": list(historical),
                "decompressed_bytes": decompressed,
            }
        )
        return evidence, historical, decompressed

    def _phase_output_evidence(self, output: Any) -> dict[str, Any]:
        try:
            evidence = self._validate_source_evidence(output.evidence)
            visible = vars(output)
        except SecGemmaLeanV36AcquisitionError:
            raise
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("source_output_invalid") from None
        if set(visible) != {"evidence"}:
            raise SecGemmaLeanV36AcquisitionError("source_output_invalid")
        _canonical_bytes({"evidence": evidence})
        return evidence

    def _reconciliation_output_parts(
        self, output: Any
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
        try:
            evidence = self._validate_source_evidence(output.evidence)
            targets = output.complete_targets
            visible = vars(output)
        except SecGemmaLeanV36AcquisitionError:
            raise
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("source_output_invalid") from None
        if type(targets) is not tuple or set(visible) != {
            "evidence",
            "complete_targets",
        }:
            raise SecGemmaLeanV36AcquisitionError("source_output_invalid")
        canonical_targets = tuple(_canonical_mapping(item) for item in targets)
        _canonical_bytes(
            {"evidence": evidence, "complete_targets": list(canonical_targets)}
        )
        return evidence, canonical_targets

    def _validate_transport_result(
        self,
        result: TransportResult,
        *,
        role: RequestRole,
        intent_event_sha256: str,
    ) -> dict[str, Any]:
        if type(result) is not TransportResult:
            raise SecGemmaLeanV36AcquisitionError("transport_result_invalid")
        try:
            temporary = result.temporary_blob_path
            details = _secure_regular_file(temporary)
            receipt = _canonical_mapping(dict(result.transport_receipt))
            metadata = {
                "request_url": result.request_url,
                "observed_url": result.observed_url,
                "role_class": result.role_class,
                "role_id": result.role_id,
                "intent_event_sha256": result.intent_event_sha256,
                "body_limit_bytes": result.body_limit_bytes,
                "temporary_blob_path": str(temporary),
                "status_code": result.status_code,
                "framing": result.framing,
                "declared_content_length": result.declared_content_length,
                "content_encoding": result.content_encoding,
                "body_bytes": result.body_bytes,
                "body_sha256": result.body_sha256,
                "raw_headers_sha256": result.raw_headers_sha256,
                "raw_headers_bytes": result.raw_headers_bytes,
                "contact_fingerprint_sha256": result.contact_fingerprint_sha256,
                "execution_identity": _canonical_mapping(
                    dict(result.execution_identity)
                ),
            }
            validate_transport_receipt(receipt, expected_metadata=metadata)
        except SecGemmaLeanV36AcquisitionError:
            raise
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("transport_result_invalid") from None
        if (
            temporary.parent != self._layout.temporary
            or temporary.resolve(strict=True) != temporary
            or result.schema_version
            != "aapl-sec-gemma-lean-evidence-v3-6-transport-v1"
            or result.request_url != role.url
            or result.observed_url != role.url
            or result.role_id != role.role_id
            or result.intent_event_sha256 != intent_event_sha256
            or result.body_limit_bytes != role.body_limit_bytes
            or result.status_code != 200
            or details.st_size != result.body_bytes
            or type(result.elapsed_milliseconds) is not int
            or not 0 <= result.elapsed_milliseconds <= REQUEST_DEADLINE_MS
            or result.contact_fingerprint_sha256
            != self._contact.fingerprint_sha256
            or result.transport_receipt_sha256
            != receipt.get("transport_receipt_sha256")
        ):
            raise SecGemmaLeanV36AcquisitionError("transport_result_invalid")
        self._contact.assert_redacted(receipt)
        return receipt

    def _promote_blob(
        self, result: TransportResult, *, sequence: int
    ) -> tuple[str, Path]:
        blob_name = f"{sequence:06d}-{self._stage}-{result.body_sha256}.blob"
        final = self._layout.blobs / blob_name
        try:
            try:
                final.lstat()
            except FileNotFoundError:
                pass
            else:
                raise OSError("blob target already exists")
            temporary_details = _secure_regular_file(result.temporary_blob_path)
            blob_directory = _secure_directory(self._layout.blobs)
            temporary_directory = _secure_directory(self._layout.temporary)
            if (
                temporary_details.st_dev != blob_directory.lstat().st_dev
                or temporary_details.st_dev != temporary_directory.lstat().st_dev
            ):
                raise OSError("cross-device blob promotion")
            # The temporary filename is preregistered in the transport result.
            # A same-filesystem rename is the one atomic publication point.
            os.rename(result.temporary_blob_path, final)
            details = _secure_regular_file(final)
            if details.st_size != result.body_bytes:
                raise OSError("promoted blob size changed")
            with final.open("r+b") as handle:
                os.fsync(handle.fileno())
            journal_module._fsync_directory(self._layout.blobs)
            journal_module._fsync_directory(self._layout.temporary)
        except Exception:
            raise SecGemmaLeanV36AcquisitionError(
                "blob_persistence_failed"
            ) from None
        return blob_name, final

    def _load_exact_blob(
        self, path: Path, *, expected_sha256: str, expected_bytes: int
    ) -> bytes:
        details = _secure_regular_file(path)
        if details.st_size != expected_bytes:
            raise SecGemmaLeanV36AcquisitionError("artifact_invalid")
        try:
            payload = path.read_bytes()
        except OSError:
            raise SecGemmaLeanV36AcquisitionError("artifact_invalid") from None
        if len(payload) != expected_bytes or _sha256_bytes(payload) != expected_sha256:
            raise SecGemmaLeanV36AcquisitionError("artifact_invalid")
        self._contact.assert_redacted(payload)
        return payload

    def _persist_transport_receipt(
        self,
        *,
        sequence: int,
        receipt: Mapping[str, Any],
        receipt_sha256: str,
    ) -> Path:
        if _SHA256_RE.fullmatch(receipt_sha256) is None:
            raise SecGemmaLeanV36AcquisitionError("transport_result_invalid")
        payload = _canonical_bytes(dict(receipt))
        path = self._layout.transport_receipts / (
            f"{sequence:06d}-{receipt_sha256}.json"
        )
        _write_immutable(path, payload)
        return path

    def _persist_parse_receipt(
        self,
        *,
        sequence: int,
        role: RequestRole,
        intent_event_sha256: str,
        response_event_sha256: str,
        blob_name: str,
        body_sha256: str,
        body_bytes: int,
        transport_receipt_sha256: str,
        evidence: Mapping[str, Any],
        historical_filenames: tuple[str, ...],
        decompressed_bytes: int | None,
    ) -> tuple[Path, str, dict[str, Any]]:
        receipt = {
            "schema_version": PARSE_RECEIPT_SCHEMA_VERSION,
            "stage": self._stage,
            "sequence": sequence,
            "role_id": role.role_id,
            "url": role.url,
            "intent_event_sha256": intent_event_sha256,
            "response_event_sha256": response_event_sha256,
            "blob_name": blob_name,
            "body_sha256": body_sha256,
            "body_bytes": body_bytes,
            "transport_receipt_sha256": transport_receipt_sha256,
            "source_evidence_sha256": evidence["source_evidence_sha256"],
            "source_evidence": dict(evidence),
            "historical_filenames": list(historical_filenames),
            "decompressed_bytes": decompressed_bytes,
        }
        self._contact.assert_redacted(receipt)
        path, digest = _persist_hashed_json(
            self._layout.parse_receipts,
            f"{sequence:06d}",
            receipt,
        )
        return path, digest, receipt

    def _persist_manifest(
        self, *, sequence: int, manifest: Mapping[str, Any]
    ) -> tuple[Path, dict[str, Any]]:
        value = _canonical_mapping(dict(manifest))
        _bare_manifest_body_sha256(
            value.get("body_sha256"),
            error_code="source_manifest_invalid",
        )
        supplied = value.get("role_manifest_sha256")
        unsigned = dict(value)
        unsigned.pop("role_manifest_sha256", None)
        if (
            type(supplied) is not str
            or _SHA256_RE.fullmatch(supplied) is None
            or _sha256_bytes(_canonical_bytes(unsigned)) != supplied
            or value.get("sequence") != sequence
        ):
            raise SecGemmaLeanV36AcquisitionError("source_manifest_invalid")
        self._contact.assert_redacted(value)
        path = self._layout.manifests / f"{sequence:06d}-{supplied}.json"
        _write_immutable(path, _canonical_bytes(value))
        return path, value

    def _parse_role(
        self,
        role: RequestRole,
        payload: bytes,
        compact: Mapping[str, Any],
    ) -> Any:
        if role.phase == "main_submissions":
            return self._source.parse_main(self._stage, payload)
        if role.phase == "historical_submissions":
            prefix = "submissions/historical/"
            if not role.role_id.startswith(prefix):
                raise SecGemmaLeanV36AcquisitionError("role_identity_invalid")
            return self._source.parse_historical(
                self._stage,
                role.role_id[len(prefix) :],
                payload,
                compact["main"],
            )
        if role.phase == "quarterly_master":
            match = re.fullmatch(r"master/([0-9]{4})/QTR([1-4])", role.role_id)
            if match is None:
                raise SecGemmaLeanV36AcquisitionError("role_identity_invalid")
            return self._source.parse_master(
                self._stage,
                int(match.group(1)),
                int(match.group(2)),
                payload,
            )
        if role.phase == "complete_submission":
            prefix = "complete/"
            accession = role.role_id[len(prefix) :] if role.role_id.startswith(prefix) else ""
            targets = compact.get("complete_targets")
            if type(targets) is not tuple:
                raise SecGemmaLeanV36AcquisitionError("phase_state_invalid")
            target = next(
                (
                    item
                    for item in targets
                    if item.get("submissions", {}).get("accession_number")
                    == accession
                ),
                None,
            )
            if target is None:
                raise SecGemmaLeanV36AcquisitionError("role_identity_invalid")
            return self._source.parse_complete(
                self._stage,
                target,
                payload,
                compact["reconciliation"],
            )
        raise SecGemmaLeanV36AcquisitionError("role_identity_invalid")

    def _role_duration_ms(self, started: float) -> int:
        elapsed = self._now() - started
        if elapsed < 0.0:
            raise SecGemmaLeanV36AcquisitionError("clock_invalid")
        return int(math.ceil(elapsed * 1000.0))

    def _record_role_failure(
        self,
        *,
        role: RequestRole,
        intent_event_sha256: str,
        code: str,
        observed_body_bytes: int,
        request_duration_ms: int,
        role_started: float,
        response_event_sha256: str | None = None,
    ) -> _TerminalRoleFailure:
        role_duration = max(
            self._role_duration_ms(role_started), request_duration_ms
        )
        selected = code
        observed = observed_body_bytes
        request_ms = request_duration_ms
        if role_duration > ROLE_DEADLINE_MS:
            selected = "role_deadline_exceeded"
        elif selected == "hard_timeout":
            request_ms = max(request_ms, REQUEST_DEADLINE_MS)
        elif request_ms > REQUEST_DEADLINE_MS:
            selected = "hard_timeout"
            request_ms = max(request_ms, REQUEST_DEADLINE_MS)
        if selected == "body_limit_exceeded":
            observed = role.reservation_bytes
        else:
            observed = min(max(0, observed), role.body_limit_bytes)
        role_duration = max(role_duration, request_ms)
        try:
            self._journal.record_role_failure(
                role.role_id,
                intent_event_sha256=intent_event_sha256,
                response_event_sha256=response_event_sha256,
                error_code=selected,
                observed_body_bytes=observed,
                request_duration_ms=request_ms,
                role_duration_ms=role_duration,
            )
        except SecGemmaLeanV36JournalError as error:
            raise SecGemmaLeanV36AcquisitionError(error.code) from None
        return _TerminalRoleFailure(selected, role_duration)

    def _process_role(
        self,
        role: RequestRole,
        *,
        first_in_invocation: bool,
        compact: Mapping[str, Any],
    ) -> tuple[_RoleRecord, int]:
        sequence = role.ordinal - 1
        role_started = self._now()
        request_started: list[float] = []

        def fetch(intent_hash: str) -> TransportResult:
            request_started.append(self._now())
            return self._transport.fetch(
                role.url,
                role_id=role.role_id,
                intent_event_sha256=intent_hash,
                body_limit=role.body_limit_bytes,
            )

        try:
            dispatch = self._dispatch_ledger.dispatch(
                purpose=f"{self._purpose}/{self._stage}",
                first_in_invocation=first_in_invocation,
                record_intent=lambda wait_ms: self._journal.record_role_intent(
                    role.role_id,
                    dispatch_wait_ms=wait_ms,
                ),
                dispatch_callback=fetch,
            )
        except SecGemmaLeanV36JournalError as error:
            raise SecGemmaLeanV36AcquisitionError(error.code) from None
        intent_hash = dispatch.intent_event_sha256
        request_ms = (
            self._role_duration_ms(request_started[0])
            if request_started
            else 0
        )
        if isinstance(dispatch.callback_error, SecGemmaLeanV36TransportError):
            error = dispatch.callback_error
            raise self._record_role_failure(
                role=role,
                intent_event_sha256=intent_hash,
                code=error.code,
                observed_body_bytes=error.observed_body_bytes,
                request_duration_ms=request_ms,
                role_started=role_started,
            )
        if dispatch.callback_error is not None:
            raise self._record_role_failure(
                role=role,
                intent_event_sha256=intent_hash,
                code="transport_error",
                observed_body_bytes=0,
                request_duration_ms=request_ms,
                role_started=role_started,
            ) from None
        result = dispatch.callback_result
        reported_request_ms = (
            result.elapsed_milliseconds
            if type(result) is TransportResult
            and type(result.elapsed_milliseconds) is int
            and result.elapsed_milliseconds >= 0
            else 0
        )
        measured_request_ms = max(request_ms, reported_request_ms)
        observed = (
            result.body_bytes
            if type(result) is TransportResult
            and type(result.body_bytes) is int
            else 0
        )
        if measured_request_ms > REQUEST_DEADLINE_MS:
            raise self._record_role_failure(
                role=role,
                intent_event_sha256=intent_hash,
                code="hard_timeout",
                observed_body_bytes=observed,
                request_duration_ms=measured_request_ms,
                role_started=role_started,
            )

        try:
            transport_receipt = self._validate_transport_result(
                result,
                role=role,
                intent_event_sha256=intent_hash,
            )
            blob_name, blob_path = self._promote_blob(result, sequence=sequence)
            self._persist_transport_receipt(
                sequence=sequence,
                receipt=transport_receipt,
                receipt_sha256=result.transport_receipt_sha256,
            )
            payload = self._load_exact_blob(
                blob_path,
                expected_sha256=result.body_sha256,
                expected_bytes=result.body_bytes,
            )
        except SecGemmaLeanV36JournalError as error:
            code = "privacy_echo" if error.code == "privacy_echo" else "blob_persistence_failed"
            raise self._record_role_failure(
                role=role,
                intent_event_sha256=intent_hash,
                code=code,
                observed_body_bytes=observed,
                request_duration_ms=measured_request_ms,
                role_started=role_started,
            ) from None
        except Exception:
            raise self._record_role_failure(
                role=role,
                intent_event_sha256=intent_hash,
                code="blob_persistence_failed",
                observed_body_bytes=observed,
                request_duration_ms=measured_request_ms,
                role_started=role_started,
            ) from None

        try:
            response_hash = self._journal.record_response_complete(
                role.role_id,
                intent_event_sha256=intent_hash,
                body_bytes=result.body_bytes,
                body_sha256=result.body_sha256,
                transport_receipt_sha256=result.transport_receipt_sha256,
                request_duration_ms=measured_request_ms,
            )
        except SecGemmaLeanV36JournalError as error:
            del payload
            raise SecGemmaLeanV36AcquisitionError(error.code) from None

        try:
            parse_output = self._parse_role(role, payload, compact)
            evidence, historical, decompressed = self._parse_output_parts(
                parse_output
            )
            _parse_path, parse_hash, parse_receipt = self._persist_parse_receipt(
                sequence=sequence,
                role=role,
                intent_event_sha256=intent_hash,
                response_event_sha256=response_hash,
                blob_name=blob_name,
                body_sha256=result.body_sha256,
                body_bytes=result.body_bytes,
                transport_receipt_sha256=result.transport_receipt_sha256,
                evidence=evidence,
                historical_filenames=historical,
                decompressed_bytes=decompressed,
            )
            manifest = self._source.build_compact_role_manifest(
                sequence=sequence,
                role_id=role.role_id,
                url=role.url,
                blob_name=blob_name,
                payload=payload,
                parse_output=parse_output,
                transport_receipt_sha256=result.transport_receipt_sha256,
                parse_receipt_sha256=parse_hash,
            )
            _manifest_path, manifest_value = self._persist_manifest(
                sequence=sequence,
                manifest=manifest,
            )
        except Exception:
            del payload
            raise self._record_role_failure(
                role=role,
                intent_event_sha256=intent_hash,
                response_event_sha256=response_hash,
                code="parse_rejected",
                observed_body_bytes=result.body_bytes,
                request_duration_ms=measured_request_ms,
                role_started=role_started,
            ) from None
        finally:
            if "payload" in locals():
                del payload

        role_duration = max(
            self._role_duration_ms(role_started), measured_request_ms
        )
        if role_duration > ROLE_DEADLINE_MS:
            raise self._record_role_failure(
                role=role,
                intent_event_sha256=intent_hash,
                response_event_sha256=response_hash,
                code="role_deadline_exceeded",
                observed_body_bytes=result.body_bytes,
                request_duration_ms=measured_request_ms,
                role_started=role_started,
            )
        try:
            self._journal.record_role_seal(
                role.role_id,
                intent_event_sha256=intent_hash,
                response_event_sha256=response_hash,
                blob_sha256=result.body_sha256,
                parse_receipt_sha256=parse_hash,
                body_bytes=result.body_bytes,
                decompressed_bytes=decompressed,
                role_duration_ms=role_duration,
            )
        except SecGemmaLeanV36JournalError as error:
            raise SecGemmaLeanV36AcquisitionError(error.code) from None
        return (
            _RoleRecord(
                manifest=manifest_value,
                parse_receipt=parse_receipt,
                transport_receipt=transport_receipt,
                blob_path=blob_path,
            ),
            role_duration,
        )

    def _persist_phase_receipt(
        self,
        *,
        phase: str,
        evidence: Mapping[str, Any],
        input_parse_receipt_sha256: Sequence[str],
        upstream_phase_receipt_sha256: str | None,
        complete_targets: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[dict[str, Any], str]:
        if phase not in {"submissions", "reconciliation"}:
            raise SecGemmaLeanV36AcquisitionError("phase_state_invalid")
        hashes = list(input_parse_receipt_sha256)
        if any(
            type(value) is not str or _SHA256_RE.fullmatch(value) is None
            for value in hashes
        ):
            raise SecGemmaLeanV36AcquisitionError("phase_state_invalid")
        receipt = {
            "schema_version": PHASE_RECEIPT_SCHEMA_VERSION,
            "stage": self._stage,
            "phase": phase,
            "input_parse_receipt_sha256": hashes,
            "upstream_phase_receipt_sha256": upstream_phase_receipt_sha256,
            "prior_stage_source_seal_sha256": (
                None
                if self._prior is None
                else self._prior.checkpoint[
                    "expected_stage_source_seal_sha256"
                ]
            ),
            "source_evidence_sha256": evidence["source_evidence_sha256"],
            "source_evidence": dict(evidence),
            "complete_targets": [dict(item) for item in complete_targets],
        }
        self._contact.assert_redacted(receipt)
        _path, digest = _persist_hashed_json(
            self._layout.phase_receipts,
            phase,
            receipt,
        )
        return receipt, digest

    def _journal_events(self) -> list[dict[str, Any]]:
        try:
            return journal_module._read_events(self._layout.journal)
        except SecGemmaLeanV36JournalError as error:
            raise SecGemmaLeanV36AcquisitionError(error.code) from None

    def _validate_manifest(self, value: Mapping[str, Any], sequence: int) -> dict[str, Any]:
        manifest = _canonical_mapping(dict(value))
        expected = {
            "schema_version",
            "sequence",
            "role_id",
            "url",
            "blob_name",
            "body_sha256",
            "body_bytes",
            "transport_receipt_sha256",
            "parse_receipt_sha256",
            "source_evidence_sha256",
            "role_manifest_sha256",
        }
        supplied = manifest.get("role_manifest_sha256")
        unsigned = dict(manifest)
        unsigned.pop("role_manifest_sha256", None)
        if (
            set(manifest) != expected
            or manifest.get("sequence") != sequence
            or type(supplied) is not str
            or _SHA256_RE.fullmatch(supplied) is None
            or _sha256_bytes(_canonical_bytes(unsigned)) != supplied
        ):
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        _bare_manifest_body_sha256(manifest.get("body_sha256"))
        for name in (
            "transport_receipt_sha256",
            "parse_receipt_sha256",
            "source_evidence_sha256",
        ):
            if (
                type(manifest.get(name)) is not str
                or _SHA256_RE.fullmatch(manifest[name]) is None
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
        return manifest

    def _validate_parse_receipt(
        self,
        value: Mapping[str, Any],
        *,
        sequence: int,
        role: RequestRole,
    ) -> dict[str, Any]:
        receipt = _canonical_mapping(dict(value))
        expected = {
            "schema_version",
            "stage",
            "sequence",
            "role_id",
            "url",
            "intent_event_sha256",
            "response_event_sha256",
            "blob_name",
            "body_sha256",
            "body_bytes",
            "transport_receipt_sha256",
            "source_evidence_sha256",
            "source_evidence",
            "historical_filenames",
            "decompressed_bytes",
        }
        if (
            set(receipt) != expected
            or receipt.get("schema_version") != PARSE_RECEIPT_SCHEMA_VERSION
            or receipt.get("stage") != self._stage
            or receipt.get("sequence") != sequence
            or receipt.get("role_id") != role.role_id
            or receipt.get("url") != role.url
            or type(receipt.get("historical_filenames")) is not list
            or receipt["historical_filenames"]
            != sorted(receipt["historical_filenames"])
            or len(receipt["historical_filenames"])
            != len(set(receipt["historical_filenames"]))
            or (
                receipt["decompressed_bytes"] is not None
                and (
                    type(receipt["decompressed_bytes"]) is not int
                    or receipt["decompressed_bytes"] < 0
                )
            )
        ):
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        evidence = self._validate_source_evidence(receipt["source_evidence"])
        if receipt["source_evidence_sha256"] != evidence["source_evidence_sha256"]:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        return receipt

    def _validate_stored_transport_receipt(
        self,
        receipt: Mapping[str, Any],
        *,
        role: RequestRole,
        intent_event_sha256: str,
        body_sha256: str,
        body_bytes: int,
    ) -> dict[str, Any]:
        value = _canonical_mapping(dict(receipt))
        try:
            http = value["http"]
            header = value["raw_response_headers"]
            metadata = {
                "request_url": role.url,
                "observed_url": http["observed_url"],
                "role_class": value["role"]["role_class"],
                "role_id": role.role_id,
                "intent_event_sha256": intent_event_sha256,
                "body_limit_bytes": role.body_limit_bytes,
                "temporary_blob_path": str(self._layout.temporary / "detached.tmp"),
                "status_code": http["status_code"],
                "framing": http["framing_mode"],
                "declared_content_length": http["declared_content_length"],
                "content_encoding": (
                    None
                    if http["content_encoding"] == "absent"
                    else http["content_encoding"]
                ),
                "body_bytes": body_bytes,
                "body_sha256": body_sha256,
                "raw_headers_sha256": header["sha256"],
                "raw_headers_bytes": header["byte_length"],
                "contact_fingerprint_sha256": self._contact.fingerprint_sha256,
                "execution_identity": value["execution_identity"],
            }
            validate_transport_receipt(value, expected_metadata=metadata)
        except Exception:
            raise SecGemmaLeanV36AcquisitionError(
                "artifact_inventory_invalid"
            ) from None
        self._contact.assert_redacted(value)
        return value

    def _load_role_records(self) -> list[_RoleRecord]:
        try:
            state = validate_detached_journal(
                self._layout.journal,
                self._stage,
                self._contact.fingerprint_sha256,
            )
        except SecGemmaLeanV36JournalError as error:
            raise SecGemmaLeanV36AcquisitionError(error.code) from None
        count = state.role_seals
        blob_files = self._files(self._layout.blobs)
        transport_files = self._files(self._layout.transport_receipts)
        parse_files = self._files(self._layout.parse_receipts)
        manifest_files = self._files(self._layout.manifests)
        if not all(
            len(values) == count
            for values in (blob_files, transport_files, parse_files, manifest_files)
        ):
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        if self._files(self._layout.temporary):
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        events = self._journal_events()
        intents = {
            event["payload"]["role_id"]: event
            for event in events
            if event["event_type"] == "role_intent"
        }
        responses = {
            event["payload"]["role_id"]: event
            for event in events
            if event["event_type"] == "response_complete"
        }
        seals = {
            event["payload"]["role_id"]: event
            for event in events
            if event["event_type"] == "role_seal"
        }
        records: list[_RoleRecord] = []
        for sequence in range(count):
            role = state.planned_roles[sequence]
            prefixes = f"{sequence:06d}-"
            candidates = [
                [path for path in values if path.name.startswith(prefixes)]
                for values in (
                    blob_files,
                    transport_files,
                    parse_files,
                    manifest_files,
                )
            ]
            if any(len(items) != 1 for items in candidates):
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            blob_path = candidates[0][0]
            transport_path = candidates[1][0]
            parse_path = candidates[2][0]
            manifest_path = candidates[3][0]
            manifest = self._validate_manifest(
                _read_canonical_mapping(manifest_path), sequence
            )
            body_sha256 = _bare_manifest_body_sha256(
                manifest["body_sha256"]
            )
            parse_receipt = self._validate_parse_receipt(
                _read_canonical_mapping(parse_path),
                sequence=sequence,
                role=role,
            )
            parse_file_sha, _ = _sha256_file(parse_path)
            intent = intents.get(role.role_id)
            response = responses.get(role.role_id)
            seal = seals.get(role.role_id)
            if intent is None or response is None or seal is None:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            transport_receipt = self._validate_stored_transport_receipt(
                _read_canonical_mapping(transport_path),
                role=role,
                intent_event_sha256=intent["event_sha256"],
                body_sha256=body_sha256,
                body_bytes=manifest["body_bytes"],
            )
            blob_sha, blob_bytes = _sha256_file(
                blob_path,
                maximum_bytes=role.body_limit_bytes,
            )
            if (
                blob_path.name != manifest["blob_name"]
                or blob_sha != body_sha256
                or blob_bytes != manifest["body_bytes"]
                or parse_file_sha != manifest["parse_receipt_sha256"]
                or parse_path.name
                != f"{sequence:06d}-{manifest['parse_receipt_sha256']}.json"
                or manifest_path.name
                != f"{sequence:06d}-{manifest['role_manifest_sha256']}.json"
                or transport_path.name
                != f"{sequence:06d}-{manifest['transport_receipt_sha256']}.json"
                or transport_receipt["transport_receipt_sha256"]
                != manifest["transport_receipt_sha256"]
                or parse_receipt["source_evidence_sha256"]
                != manifest["source_evidence_sha256"]
                or parse_receipt["blob_name"] != manifest["blob_name"]
                or parse_receipt["body_sha256"] != body_sha256
                or parse_receipt["body_bytes"] != manifest["body_bytes"]
                or parse_receipt["transport_receipt_sha256"]
                != manifest["transport_receipt_sha256"]
                or parse_receipt["intent_event_sha256"]
                != intent["event_sha256"]
                or parse_receipt["response_event_sha256"]
                != response["event_sha256"]
                or response["payload"]["body_sha256"] != body_sha256
                or response["payload"]["body_bytes"] != manifest["body_bytes"]
                or response["payload"]["transport_receipt_sha256"]
                != manifest["transport_receipt_sha256"]
                or seal["payload"]["parse_receipt_sha256"]
                != manifest["parse_receipt_sha256"]
                or seal["payload"]["blob_sha256"] != body_sha256
                or seal["payload"]["body_bytes"] != manifest["body_bytes"]
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            records.append(
                _RoleRecord(
                    manifest=manifest,
                    parse_receipt=parse_receipt,
                    transport_receipt=transport_receipt,
                    blob_path=blob_path,
                )
            )
        return records

    def _validate_phase_receipt(
        self,
        value: Mapping[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        receipt = _canonical_mapping(dict(value))
        expected = {
            "schema_version",
            "stage",
            "phase",
            "input_parse_receipt_sha256",
            "upstream_phase_receipt_sha256",
            "prior_stage_source_seal_sha256",
            "source_evidence_sha256",
            "source_evidence",
            "complete_targets",
        }
        if (
            set(receipt) != expected
            or receipt.get("schema_version") != PHASE_RECEIPT_SCHEMA_VERSION
            or receipt.get("stage") != self._stage
            or receipt.get("phase") != phase
            or type(receipt.get("input_parse_receipt_sha256")) is not list
            or any(
                type(item) is not str or _SHA256_RE.fullmatch(item) is None
                for item in receipt["input_parse_receipt_sha256"]
            )
            or type(receipt.get("complete_targets")) is not list
        ):
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        evidence = self._validate_source_evidence(receipt["source_evidence"])
        if receipt["source_evidence_sha256"] != evidence["source_evidence_sha256"]:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        return receipt

    def _load_phase_receipts(
        self,
        records: Sequence[_RoleRecord],
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None, str | None]:
        state = self._journal.state
        files = self._files(self._layout.phase_receipts)
        expected_count = int(len(state.phase_expansions) >= 2) + int(
            len(state.phase_expansions) >= 3
        )
        if len(files) != expected_count:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        by_phase: dict[str, tuple[dict[str, Any], str]] = {}
        for path in files:
            match = _PHASE_ARTIFACT_RE.fullmatch(path.name)
            if match is None or match.group("phase") in by_phase:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            observed, _ = _sha256_file(path)
            if observed != match.group("sha"):
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            phase = match.group("phase")
            by_phase[phase] = (
                self._validate_phase_receipt(
                    _read_canonical_mapping(path), phase=phase
                ),
                observed,
            )
        submissions = by_phase.get("submissions")
        reconciliation = by_phase.get("reconciliation")
        parse_hashes = [
            str(record.manifest["parse_receipt_sha256"]) for record in records
        ]
        role_phases = [
            self._journal.state.planned_roles[index].phase
            for index in range(len(records))
        ]
        if submissions is not None:
            expected_inputs = [
                parse_hashes[index]
                for index, phase in enumerate(role_phases)
                if phase in {"main_submissions", "historical_submissions"}
            ]
            if (
                submissions[0]["input_parse_receipt_sha256"] != expected_inputs
                or submissions[0]["upstream_phase_receipt_sha256"] is not None
                or submissions[0]["complete_targets"] != []
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
        if reconciliation is not None:
            expected_inputs = [
                parse_hashes[index]
                for index, phase in enumerate(role_phases)
                if phase == "quarterly_master"
            ]
            if (
                submissions is None
                or reconciliation[0]["input_parse_receipt_sha256"]
                != expected_inputs
                or reconciliation[0]["upstream_phase_receipt_sha256"]
                != submissions[1]
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
        phase_events = [
            event
            for event in self._journal_events()
            if event["event_type"] == "phase_plan"
        ]
        if len(phase_events) != len(state.phase_expansions):
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        expected_bases: list[str] = []
        if len(state.phase_expansions) >= 1:
            if not records:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            expected_bases.append(str(records[0].manifest["parse_receipt_sha256"]))
        if len(state.phase_expansions) >= 2:
            if submissions is None:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            expected_bases.append(submissions[1])
        if len(state.phase_expansions) >= 3:
            if reconciliation is None:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            expected_bases.append(reconciliation[1])
        if [
            event["payload"]["basis_receipt_sha256"] for event in phase_events
        ] != expected_bases:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        expected_prior_hash = (
            None
            if self._prior is None
            else self._prior.checkpoint["expected_stage_source_seal_sha256"]
        )
        for item in (submissions, reconciliation):
            if item is not None and item[0]["prior_stage_source_seal_sha256"] != expected_prior_hash:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
        return (
            None if submissions is None else submissions[0],
            None if submissions is None else submissions[1],
            None if reconciliation is None else reconciliation[0],
            None if reconciliation is None else reconciliation[1],
        )

    def _restore_compact_state(
        self,
        records: Sequence[_RoleRecord],
        submissions_receipt: Mapping[str, Any] | None,
        reconciliation_receipt: Mapping[str, Any] | None,
        *,
        prior_stage_output: Any | None,
    ) -> dict[str, Any]:
        compact: dict[str, Any] = {
            "main": None,
            "historical": [],
            "masters": [],
            "completes": [],
            "submissions": (
                None
                if submissions_receipt is None
                else submissions_receipt["source_evidence"]
            ),
            "reconciliation": (
                None
                if reconciliation_receipt is None
                else reconciliation_receipt["source_evidence"]
            ),
            "complete_targets": (
                ()
                if reconciliation_receipt is None
                else tuple(reconciliation_receipt["complete_targets"])
            ),
            "prior_stage_output": prior_stage_output,
        }
        for index, record in enumerate(records):
            phase = self._journal.state.planned_roles[index].phase
            evidence = record.parse_receipt["source_evidence"]
            if phase == "main_submissions":
                compact["main"] = evidence
            elif phase == "historical_submissions":
                compact["historical"].append(evidence)
            elif phase == "quarterly_master":
                compact["masters"].append(evidence)
            elif phase == "complete_submission":
                compact["completes"].append(evidence)
        compact["historical"] = tuple(compact["historical"])
        compact["masters"] = tuple(compact["masters"])
        compact["completes"] = tuple(compact["completes"])
        return compact

    def _checkpoint_and_public_inventory(
        self,
    ) -> tuple[Path | None, str | None, dict[str, Any] | None, Path | None, dict[str, Any] | None]:
        state = self._journal.state
        try:
            private_top = list(self._layout.private_stage.iterdir())
        except OSError:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid") from None
        checkpoint_files = [
            path
            for path in private_top
            if _CHECKPOINT_RE.fullmatch(path.name) is not None
        ]
        public_files = self._files(self._layout.public_stage)
        if state.passed:
            if len(checkpoint_files) != 1 or len(public_files) != 1:
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
        elif checkpoint_files or public_files:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        if not state.passed:
            return None, None, None, None, None
        checkpoint_path = checkpoint_files[0]
        checkpoint_match = _CHECKPOINT_RE.fullmatch(checkpoint_path.name)
        public_path = public_files[0]
        public_match = _PUBLIC_AUTHORITY_RE.fullmatch(public_path.name)
        if checkpoint_match is None or public_match is None:
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        checkpoint_sha, _ = _sha256_file(checkpoint_path)
        public_sha, _ = _sha256_file(public_path)
        if checkpoint_sha != checkpoint_match.group("sha") or public_sha != public_match.group("sha"):
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        checkpoint = _read_canonical_mapping(checkpoint_path)
        public = _read_canonical_mapping(public_path)
        public_expected = {
            "schema_version",
            "stage",
            "checkpoint_file_sha256",
            "checkpoint_sha256",
            "stage_source_seal_sha256",
            "role_manifest_count",
            "role_manifests_sha256",
            "journal_preterminal_head_sha256",
            "role_plan_sha256",
            "formula_requests",
            "source_replay_receipt",
            "source_authoritative",
            "peak_live_role_payload_count",
            "contains_private_paths",
            "contains_contact_text",
        }
        events = self._journal_events()
        terminal = events[-1] if events else None
        if (
            terminal is None
            or terminal["event_type"] != "terminal"
            or terminal["payload"]["status"] != "passed"
            or terminal["payload"]["checkpoint_sha256"] != checkpoint_sha
            or set(public) != public_expected
            or public.get("schema_version") != ACQUISITION_SCHEMA_VERSION
            or public.get("stage") != self._stage
            or public.get("checkpoint_file_sha256") != checkpoint_sha
            or public.get("checkpoint_sha256") != checkpoint.get("checkpoint_sha256")
            or public.get("journal_preterminal_head_sha256")
            != terminal["previous_event_sha256"]
            or public.get("source_authoritative") is not True
            or public.get("peak_live_role_payload_count") != 1
            or public.get("contains_private_paths") is not False
            or public.get("contains_contact_text") is not False
            or public.get("role_plan_sha256") != state.role_plan_sha256
            or public.get("formula_requests") != state.formula_requests
            or public.get("role_manifest_count")
            != checkpoint.get("role_manifest_count")
            or public.get("role_manifests_sha256")
            != checkpoint.get("role_manifests_sha256")
            or public.get("stage_source_seal_sha256")
            != checkpoint.get("expected_stage_source_seal_sha256")
        ):
            raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
        self._contact.assert_redacted(checkpoint)
        self._contact.assert_redacted(public)
        return checkpoint_path, checkpoint_sha, checkpoint, public_path, public

    def _blob_loader(
        self, current_records: Sequence[_RoleRecord]
    ) -> tuple[Callable[[str], bytes], dict[str, Path]]:
        paths: dict[str, Path] = {}
        if self._prior is not None:
            paths.update(dict(self._prior.blob_paths))
        for record in current_records:
            name = str(record.manifest["blob_name"])
            if name in paths or record.blob_path.name != name:
                raise SecGemmaLeanV36AcquisitionError("artifact_inventory_invalid")
            paths[name] = record.blob_path

        def load_blob(blob_name: str) -> bytes:
            if type(blob_name) is not str or blob_name not in paths:
                raise SecGemmaLeanV36AcquisitionError("artifact_invalid")
            path = paths[blob_name]
            digest, length = _sha256_file(path)
            expected_manifest = next(
                (
                    record.manifest
                    for record in current_records
                    if record.manifest["blob_name"] == blob_name
                ),
                None,
            )
            if expected_manifest is not None:
                expected_digest = _bare_manifest_body_sha256(
                    expected_manifest.get("body_sha256"),
                    error_code="artifact_invalid",
                )
                if (
                    digest != expected_digest
                    or length != expected_manifest["body_bytes"]
                ):
                    raise SecGemmaLeanV36AcquisitionError("artifact_invalid")
            try:
                payload = path.read_bytes()
            except OSError:
                raise SecGemmaLeanV36AcquisitionError("artifact_invalid") from None
            self._contact.assert_redacted(payload)
            return payload

        return load_blob, paths

    def _authenticate_prior(self) -> Any | None:
        if self._prior is None:
            return None
        prior = self._prior
        expected_blobs: dict[str, tuple[str, int, str]] = {}
        chain: list[dict[str, Any]] = []
        expected_stages = tuple(
            reversed(_STAGES[: _STAGES.index(prior.stage) + 1])
        )
        try:
            replay_input = prior.compact_replay_input
            seen_inputs: set[int] = set()
            for expected_stage in expected_stages:
                if (
                    replay_input is None
                    or not isinstance(
                        replay_input, self._source.CompactReplayInput
                    )
                    or id(replay_input) in seen_inputs
                    or replay_input.stage != expected_stage
                    or type(replay_input.role_manifests) is not tuple
                ):
                    raise SecGemmaLeanV36AcquisitionError(
                        "prior_stage_invalid"
                    )
                seen_inputs.add(id(replay_input))
                private_stage = _secure_directory(
                    self._layout.private_root / expected_stage
                )
                public_stage = _secure_directory(
                    self._layout.public_root / expected_stage
                )
                journal_root = _secure_directory(private_stage / "journal")
                state = validate_detached_journal(
                    journal_root,
                    expected_stage,
                    self._contact.fingerprint_sha256,
                )
                events = journal_module._read_events(journal_root)
                if (
                    not state.accounting_complete
                    or not state.passed
                    or not events
                    or events[-1]["event_type"] != "terminal"
                    or events[-1]["payload"]["status"] != "passed"
                ):
                    raise SecGemmaLeanV36AcquisitionError(
                        "prior_stage_invalid"
                    )

                checkpoint_files = [
                    path
                    for path in private_stage.iterdir()
                    if _CHECKPOINT_RE.fullmatch(path.name) is not None
                ]
                if len(checkpoint_files) != 1:
                    raise SecGemmaLeanV36AcquisitionError(
                        "prior_stage_invalid"
                    )
                checkpoint_path = checkpoint_files[0]
                checkpoint_match = _CHECKPOINT_RE.fullmatch(
                    checkpoint_path.name
                )
                checkpoint_file_sha, _ = _sha256_file(checkpoint_path)
                checkpoint = _read_canonical_mapping(checkpoint_path)
                self._contact.assert_redacted(checkpoint)
                if (
                    checkpoint_match is None
                    or checkpoint_match.group("sha") != checkpoint_file_sha
                    or _canonical_mapping(dict(replay_input.checkpoint))
                    != checkpoint
                ):
                    raise SecGemmaLeanV36AcquisitionError(
                        "prior_stage_invalid"
                    )

                audit = object.__new__(DiskBackedSecAcquisition)
                audit._stage = expected_stage
                audit._contact = self._contact
                audit._source = self._source
                audit._layout = _Layout(
                    private_root=self._layout.private_root,
                    public_root=self._layout.public_root,
                    private_stage=private_stage,
                    public_stage=public_stage,
                    journal=journal_root,
                    temporary=private_stage / "temporary",
                    blobs=private_stage / "blobs",
                    transport_receipts=private_stage / "transport_receipts",
                    parse_receipts=private_stage / "parse_receipts",
                    manifests=private_stage / "manifests",
                    phase_receipts=private_stage / "phase_receipts",
                    run_lock=private_stage / ".run.lock",
                )
                audit._journal = AcquisitionJournal(
                    journal_root,
                    stage=expected_stage,
                    contact=self._contact,
                )
                audit._prior = (
                    None
                    if replay_input.prior is None
                    else SimpleNamespace(
                        checkpoint=_canonical_mapping(
                            dict(replay_input.prior.checkpoint)
                        )
                    )
                )
                audit._audit_layout_names()
                audited_records = audit._load_role_records()
                audit._load_phase_receipts(audited_records)
                (
                    audited_checkpoint_path,
                    audited_checkpoint_sha,
                    audited_checkpoint,
                    audited_public_path,
                    audited_public,
                ) = audit._checkpoint_and_public_inventory()
                if (
                    audited_checkpoint_path != checkpoint_path
                    or audited_checkpoint_sha != checkpoint_file_sha
                    or audited_checkpoint != checkpoint
                    or audited_public_path is None
                    or audited_public is None
                ):
                    raise SecGemmaLeanV36AcquisitionError(
                        "prior_stage_invalid"
                    )

                disk_manifests = [
                    dict(record.manifest) for record in audited_records
                ]
                for manifest in disk_manifests:
                    self._contact.assert_redacted(manifest)
                    blob_name = manifest.get("blob_name")
                    body_sha256 = manifest.get("body_sha256")
                    body_bytes = manifest.get("body_bytes")
                    if (
                        type(blob_name) is not str
                        or blob_name in expected_blobs
                        or type(body_bytes) is not int
                        or body_bytes < 0
                    ):
                        raise SecGemmaLeanV36AcquisitionError(
                            "prior_stage_invalid"
                        )
                    expected_blobs[blob_name] = (
                        _bare_manifest_body_sha256(
                            body_sha256,
                            error_code="prior_stage_invalid",
                        ),
                        body_bytes,
                        expected_stage,
                    )
                if (
                    tuple(
                        _canonical_mapping(dict(value))
                        for value in replay_input.role_manifests
                    )
                    != tuple(disk_manifests)
                    or len(disk_manifests) != state.role_seals
                    or checkpoint.get("role_manifest_count")
                    != len(disk_manifests)
                ):
                    raise SecGemmaLeanV36AcquisitionError(
                        "prior_stage_invalid"
                    )

                public_path = audited_public_path
                public_value = audited_public
                self._contact.assert_redacted(public_value)
                chain.append(
                    {
                        "input": replay_input,
                        "stage": expected_stage,
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_file_sha256": checkpoint_file_sha,
                        "checkpoint": checkpoint,
                        "manifests": tuple(disk_manifests),
                        "state": state,
                        "events": events,
                        "public_path": public_path,
                        "public": public_value,
                    }
                )
                replay_input = replay_input.prior
            if replay_input is not None:
                raise SecGemmaLeanV36AcquisitionError(
                    "prior_stage_invalid"
                )
            supplied_paths = dict(prior.blob_paths)
        except SecGemmaLeanV36AcquisitionError:
            raise SecGemmaLeanV36AcquisitionError(
                "prior_stage_invalid"
            ) from None
        except Exception:
            raise SecGemmaLeanV36AcquisitionError(
                "prior_stage_invalid"
            ) from None

        immediate = chain[0]
        if (
            immediate["checkpoint_path"] != prior.checkpoint_path
            or immediate["checkpoint_file_sha256"]
            != prior.checkpoint_file_sha256
            or immediate["checkpoint"] != prior.checkpoint
            or immediate["state"] != prior.journal_state
            or immediate["public_path"] != prior.public_authority_path
        ):
            raise SecGemmaLeanV36AcquisitionError("prior_stage_invalid")
        if set(supplied_paths) != set(expected_blobs):
            raise SecGemmaLeanV36AcquisitionError("prior_stage_invalid")

        def load_prior_blob(name: str) -> bytes:
            try:
                expected_sha, expected_bytes, expected_stage = expected_blobs[
                    name
                ]
                path = supplied_paths[name]
            except (KeyError, TypeError):
                raise SecGemmaLeanV36AcquisitionError(
                    "prior_stage_invalid"
                ) from None
            if (
                not isinstance(path, Path)
                or path.name != name
                or path.parent
                != self._layout.private_root / expected_stage / "blobs"
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "prior_stage_invalid"
                )
            try:
                return self._load_exact_blob(
                    path,
                    expected_sha256=expected_sha,
                    expected_bytes=expected_bytes,
                )
            except (SecGemmaLeanV36AcquisitionError, SecGemmaLeanV36JournalError):
                raise SecGemmaLeanV36AcquisitionError(
                    "prior_stage_invalid"
                ) from None

        # Authenticate every supplied path before handing a loader to the
        # source replay.  Payloads are discarded one at a time.
        for blob_name in expected_blobs:
            payload = load_prior_blob(blob_name)
            del payload
        immediate_stage_output = None
        immediate_receipt = None
        try:
            for item in reversed(chain):
                result = self._source.rehydrate_compact_stage(
                    item["stage"],
                    item["checkpoint"],
                    item["manifests"],
                    load_prior_blob,
                    item["input"].prior,
                )
                receipt = _canonical_mapping(dict(result.receipt))
                stage_output = result.stage_output
                expected_public = _build_public_authority_value(
                    stage=item["stage"],
                    checkpoint_file_sha256=item[
                        "checkpoint_file_sha256"
                    ],
                    checkpoint=item["checkpoint"],
                    replay_receipt=receipt,
                    journal_preterminal_head_sha256=item["events"][-1][
                        "previous_event_sha256"
                    ],
                    journal_state=item["state"],
                )
                if (
                    stage_output.stage_source_seal_sha256
                    != item["checkpoint"][
                        "expected_stage_source_seal_sha256"
                    ]
                    or item["public"] != expected_public
                ):
                    raise SecGemmaLeanV36AcquisitionError(
                        "prior_stage_invalid"
                    )
                self._contact.assert_redacted(receipt)
                self._contact.assert_redacted(expected_public)
                if item is immediate:
                    immediate_stage_output = stage_output
                    immediate_receipt = receipt
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("prior_stage_invalid") from None
        if (
            immediate_stage_output is None
            or immediate_receipt != prior.replay_receipt
        ):
            raise SecGemmaLeanV36AcquisitionError("prior_stage_invalid")
        return immediate_stage_output

    def _rehydrate_current(
        self,
        *,
        checkpoint: Mapping[str, Any],
        records: Sequence[_RoleRecord],
        load_blob: Callable[[str], bytes],
    ) -> tuple[Any, dict[str, Any], Any]:
        manifests = tuple(record.manifest for record in records)
        prior_input = (
            None if self._prior is None else self._prior.compact_replay_input
        )
        try:
            result = self._source.rehydrate_compact_stage(
                self._stage,
                checkpoint,
                manifests,
                load_blob,
                prior_input,
            )
            stage_output = result.stage_output
            receipt = _canonical_mapping(dict(result.receipt))
            replay_input = self._source.CompactReplayInput(
                stage=self._stage,
                checkpoint=dict(checkpoint),
                role_manifests=manifests,
                prior=prior_input,
            )
        except Exception:
            raise SecGemmaLeanV36AcquisitionError("source_replay_rejected") from None
        if (
            receipt.get("stage") != self._stage
            or receipt.get("exact_source_seal_match") is not True
            or receipt.get("peak_live_role_payload_count") != 1
            or receipt.get("checkpoint_sha256") != checkpoint.get("checkpoint_sha256")
            or receipt.get("main_parse_receipt_sha256")
            != checkpoint.get("main_parse_receipt_sha256")
            or receipt.get("submissions_snapshot_receipt_sha256")
            != checkpoint.get("submissions_snapshot_receipt_sha256")
            or receipt.get("reconciliation_and_prior_chain_receipt_sha256")
            != checkpoint.get("reconciliation_and_prior_chain_receipt_sha256")
            or stage_output.stage_source_seal_sha256
            != checkpoint.get("expected_stage_source_seal_sha256")
        ):
            raise SecGemmaLeanV36AcquisitionError("source_replay_rejected")
        self._contact.assert_redacted(receipt)
        return stage_output, receipt, replay_input

    def _persist_checkpoint(
        self, checkpoint: Mapping[str, Any]
    ) -> tuple[Path, str, dict[str, Any]]:
        value = _canonical_mapping(dict(checkpoint))
        supplied = value.get("checkpoint_sha256")
        unsigned = dict(value)
        unsigned.pop("checkpoint_sha256", None)
        if (
            type(supplied) is not str
            or _SHA256_RE.fullmatch(supplied) is None
            or _sha256_bytes(_canonical_bytes(unsigned)) != supplied
        ):
            raise SecGemmaLeanV36AcquisitionError("checkpoint_invalid")
        self._contact.assert_redacted(value)
        path, file_sha = _persist_hashed_json(
            self._layout.private_stage,
            "checkpoint",
            value,
        )
        return path, file_sha, value

    def _public_authority_value(
        self,
        *,
        checkpoint_file_sha256: str,
        checkpoint: Mapping[str, Any],
        replay_receipt: Mapping[str, Any],
        journal_preterminal_head_sha256: str,
    ) -> dict[str, Any]:
        state = self._journal.state
        value = _build_public_authority_value(
            stage=self._stage,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint=checkpoint,
            replay_receipt=replay_receipt,
            journal_preterminal_head_sha256=journal_preterminal_head_sha256,
            journal_state=state,
        )
        self._contact.assert_redacted(value)
        return value

    def _persist_public_authority(
        self,
        *,
        checkpoint_file_sha256: str,
        checkpoint: Mapping[str, Any],
        replay_receipt: Mapping[str, Any],
    ) -> Path:
        value = self._public_authority_value(
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint=checkpoint,
            replay_receipt=replay_receipt,
            journal_preterminal_head_sha256=self._journal.state.head_event_sha256,
        )
        path, _ = _persist_hashed_json(
            self._layout.public_stage,
            "source-authority",
            value,
        )
        return path

    def _terminal_result(
        self,
        *,
        records: Sequence[_RoleRecord],
        checkpoint_path: Path,
        checkpoint_file_sha256: str,
        checkpoint: Mapping[str, Any],
        public_path: Path,
    ) -> AcquisitionResult:
        load_blob, blob_paths = self._blob_loader(records)
        stage_output, replay_receipt, replay_input = self._rehydrate_current(
            checkpoint=checkpoint,
            records=records,
            load_blob=load_blob,
        )
        state = validate_detached_journal(
            self._layout.journal,
            self._stage,
            self._contact.fingerprint_sha256,
        )
        if not state.accounting_complete or not state.passed:
            raise SecGemmaLeanV36AcquisitionError("accounting_incomplete")
        events = self._journal_events()
        if not events or events[-1]["event_type"] != "terminal":
            raise SecGemmaLeanV36AcquisitionError(
                "artifact_inventory_invalid"
            )
        expected_public = self._public_authority_value(
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint=checkpoint,
            replay_receipt=replay_receipt,
            journal_preterminal_head_sha256=events[-1][
                "previous_event_sha256"
            ],
        )
        if _read_canonical_mapping(public_path) != expected_public:
            raise SecGemmaLeanV36AcquisitionError(
                "artifact_inventory_invalid"
            )
        return AcquisitionResult(
            stage=self._stage,
            checkpoint_path=checkpoint_path,
            checkpoint_file_sha256=checkpoint_file_sha256,
            checkpoint=dict(checkpoint),
            replay_receipt=replay_receipt,
            journal_state=state,
            stage_output=stage_output,
            compact_replay_input=replay_input,
            public_authority_path=public_path,
            accounting_complete=True,
            source_authoritative=True,
            blob_paths=blob_paths,
        )

    def _reject_clean_boundary(self, code: str = "source_rejected") -> None:
        try:
            self._journal.seal_terminal_rejection(code=code)
        except SecGemmaLeanV36JournalError as error:
            raise SecGemmaLeanV36AcquisitionError(error.code) from None
        raise SecGemmaLeanV36AcquisitionError(code)

    def run(self) -> AcquisitionResult:
        """Acquire or revalidate one stage; never retry or adopt crash debris."""

        if not self._run_lock.acquire(blocking=False):
            raise SecGemmaLeanV36AcquisitionError("active_run_conflict")
        try:
            with _locked_file(
                self._layout.run_lock,
                busy_code="active_run_conflict",
            ):
                return self._run_locked()
        finally:
            try:
                self._run_lock.release()
            except RuntimeError:
                raise SecGemmaLeanV36AcquisitionError(
                    "run_lock_release_failed"
                ) from None

    def _run_locked(self) -> AcquisitionResult:
        self._audit_layout_names()
        records = self._load_role_records()
        (
            submissions_receipt,
            submissions_receipt_sha,
            reconciliation_receipt,
            reconciliation_receipt_sha,
        ) = self._load_phase_receipts(records)
        (
            checkpoint_path,
            checkpoint_file_sha,
            checkpoint,
            public_path,
            _public,
        ) = self._checkpoint_and_public_inventory()
        state = self._journal.state
        if state.terminal_status == "rejected":
            raise SecGemmaLeanV36AcquisitionError(
                state.terminal_code or "source_rejected"
            )
        if state.passed:
            if (
                checkpoint_path is None
                or checkpoint_file_sha is None
                or checkpoint is None
                or public_path is None
            ):
                raise SecGemmaLeanV36AcquisitionError(
                    "artifact_inventory_invalid"
                )
            return self._terminal_result(
                records=records,
                checkpoint_path=checkpoint_path,
                checkpoint_file_sha256=checkpoint_file_sha,
                checkpoint=checkpoint,
                public_path=public_path,
            )
        if (
            state.active_invocation is not None
            or state.open_role_id is not None
            or state.failure_code is not None
            or state.derived_terminal_code is not None
        ):
            raise SecGemmaLeanV36AcquisitionError(
                state.derived_terminal_code or "unclosed_invocation"
            )

        prior_stage_output = self._authenticate_prior()
        compact = self._restore_compact_state(
            records,
            submissions_receipt,
            reconciliation_receipt,
            prior_stage_output=prior_stage_output,
        )
        while True:
            state = self._journal.state
            if state.next_role_id is not None:
                if (
                    STAGE_ACTIVE_TIME_CAP_MS - state.cumulative_active_ms
                    < ROLE_DEADLINE_MS
                ):
                    self._reject_clean_boundary(
                        "cumulative_source_deadline_exceeded"
                    )
                try:
                    self._journal.open_invocation()
                except SecGemmaLeanV36JournalError as error:
                    raise SecGemmaLeanV36AcquisitionError(error.code) from None
                invocation_started = self._now()
                invocation_role_ms = 0
                first = True
                stage_deadline_stop = False
                try:
                    while self._journal.state.next_role_id is not None:
                        invocation_elapsed = self._role_duration_ms(
                            invocation_started
                        )
                        stage_elapsed = (
                            self._journal.state.cumulative_active_ms
                            + invocation_elapsed
                        )
                        # Never start a request unless the entire ten-minute
                        # role budget still fits inside both live deadlines.
                        if (
                            STAGE_ACTIVE_TIME_CAP_MS - stage_elapsed
                            < ROLE_DEADLINE_MS
                        ):
                            stage_deadline_stop = True
                            break
                        if (
                            INVOCATION_DEADLINE_MS - invocation_elapsed
                            < ROLE_DEADLINE_MS
                        ):
                            break
                        role = self._journal.state.planned_roles[
                            self._journal.state.role_seals
                        ]
                        record, role_ms = self._process_role(
                            role,
                            first_in_invocation=first,
                            compact=compact,
                        )
                        records.append(record)
                        evidence = record.parse_receipt["source_evidence"]
                        if role.phase == "main_submissions":
                            compact["main"] = evidence
                        elif role.phase == "historical_submissions":
                            compact["historical"] = (
                                *compact["historical"],
                                evidence,
                            )
                        elif role.phase == "quarterly_master":
                            compact["masters"] = (*compact["masters"], evidence)
                        elif role.phase == "complete_submission":
                            compact["completes"] = (
                                *compact["completes"],
                                evidence,
                            )
                        invocation_role_ms += role_ms
                        first = False
                except _TerminalRoleFailure as failure:
                    invocation_role_ms += failure.role_duration_ms
                    invocation_ms = max(
                        invocation_role_ms,
                        self._role_duration_ms(invocation_started),
                    )
                    try:
                        self._journal.close_invocation(
                            invocation_duration_ms=invocation_ms
                        )
                        self._journal.seal_terminal_rejection()
                    except SecGemmaLeanV36JournalError as error:
                        raise SecGemmaLeanV36AcquisitionError(error.code) from None
                    raise SecGemmaLeanV36AcquisitionError(failure.code) from None
                invocation_ms = max(
                    invocation_role_ms,
                    self._role_duration_ms(invocation_started),
                )
                try:
                    self._journal.close_invocation(
                        invocation_duration_ms=invocation_ms
                    )
                except SecGemmaLeanV36JournalError as error:
                    raise SecGemmaLeanV36AcquisitionError(error.code) from None
                closed_state = self._journal.state
                if closed_state.derived_terminal_code is not None:
                    try:
                        self._journal.seal_terminal_rejection()
                    except SecGemmaLeanV36JournalError as error:
                        raise SecGemmaLeanV36AcquisitionError(error.code) from None
                    raise SecGemmaLeanV36AcquisitionError(
                        closed_state.derived_terminal_code
                    )
                if stage_deadline_stop:
                    self._reject_clean_boundary(
                        "cumulative_source_deadline_exceeded"
                    )
                continue

            expansion = state.next_required_expansion
            if expansion == "historical_submissions":
                if not records or compact["main"] is None:
                    raise SecGemmaLeanV36AcquisitionError("phase_state_invalid")
                names = tuple(records[0].parse_receipt["historical_filenames"])
                try:
                    self._journal.append_historical_phase(names)
                except SecGemmaLeanV36JournalError as error:
                    raise SecGemmaLeanV36AcquisitionError(error.code) from None
                continue

            if expansion == "quarterly_master":
                try:
                    output = self._source.finalize_submissions(
                        self._stage,
                        compact["main"],
                        compact["historical"],
                        prior_stage_output,
                    )
                    evidence = self._phase_output_evidence(output)
                except Exception as error:
                    self._reject_clean_boundary(
                        _source_diagnostic_code(error)
                    )
                input_hashes = [
                    str(record.manifest["parse_receipt_sha256"])
                    for index, record in enumerate(records)
                    if self._journal.state.planned_roles[index].phase
                    in {"main_submissions", "historical_submissions"}
                ]
                submissions_receipt, submissions_receipt_sha = (
                    self._persist_phase_receipt(
                        phase="submissions",
                        evidence=evidence,
                        input_parse_receipt_sha256=input_hashes,
                        upstream_phase_receipt_sha256=None,
                    )
                )
                compact["submissions"] = evidence
                try:
                    self._journal.append_master_phase(
                        submissions_snapshot_receipt_sha256=(
                            submissions_receipt_sha
                        )
                    )
                except SecGemmaLeanV36JournalError as error:
                    raise SecGemmaLeanV36AcquisitionError(error.code) from None
                continue

            if expansion == "complete_submission":
                if submissions_receipt_sha is None:
                    raise SecGemmaLeanV36AcquisitionError("phase_state_invalid")
                try:
                    output = self._source.reconcile_masters(
                        self._stage,
                        compact["submissions"],
                        compact["masters"],
                        prior_stage_output,
                    )
                    evidence, targets = self._reconciliation_output_parts(output)
                except Exception as error:
                    self._reject_clean_boundary(
                        _source_diagnostic_code(error)
                    )
                input_hashes = [
                    str(record.manifest["parse_receipt_sha256"])
                    for index, record in enumerate(records)
                    if self._journal.state.planned_roles[index].phase
                    == "quarterly_master"
                ]
                reconciliation_receipt, reconciliation_receipt_sha = (
                    self._persist_phase_receipt(
                        phase="reconciliation",
                        evidence=evidence,
                        input_parse_receipt_sha256=input_hashes,
                        upstream_phase_receipt_sha256=(
                            submissions_receipt_sha
                        ),
                        complete_targets=targets,
                    )
                )
                compact["reconciliation"] = evidence
                compact["complete_targets"] = targets
                complete_targets: list[CompleteSubmissionTarget] = []
                try:
                    for target in targets:
                        submissions = target["submissions"]
                        complete_targets.append(
                            CompleteSubmissionTarget(
                                filing_date=submissions["filing_date"],
                                accession=submissions["accession_number"],
                            )
                        )
                except Exception:
                    self._reject_clean_boundary()
                try:
                    self._journal.append_complete_phase(
                        tuple(complete_targets),
                        reconciliation_and_prior_chain_receipt_sha256=(
                            reconciliation_receipt_sha
                        ),
                    )
                except SecGemmaLeanV36JournalError as error:
                    raise SecGemmaLeanV36AcquisitionError(error.code) from None
                continue

            if state.can_finalize_pass:
                if (
                    submissions_receipt_sha is None
                    or reconciliation_receipt_sha is None
                ):
                    raise SecGemmaLeanV36AcquisitionError("phase_state_invalid")
                try:
                    live_stage_output = self._source.finalize_stage(
                        self._stage,
                        compact["submissions"],
                        compact["reconciliation"],
                        compact["completes"],
                        prior_stage_output,
                    )
                    checkpoint_value = self._source.build_compact_checkpoint(
                        stage=self._stage,
                        role_manifests=tuple(
                            record.manifest for record in records
                        ),
                        stage_output=live_stage_output,
                        submissions_snapshot_receipt_sha256=(
                            submissions_receipt_sha
                        ),
                        reconciliation_and_prior_chain_receipt_sha256=(
                            reconciliation_receipt_sha
                        ),
                    )
                    (
                        checkpoint_path,
                        checkpoint_file_sha,
                        checkpoint,
                    ) = self._persist_checkpoint(checkpoint_value)
                    load_blob, _blob_paths = self._blob_loader(records)
                    (
                        replayed_stage_output,
                        replay_receipt,
                        _replay_input,
                    ) = self._rehydrate_current(
                        checkpoint=checkpoint,
                        records=records,
                        load_blob=load_blob,
                    )
                    if (
                        replayed_stage_output.stage_source_seal_sha256
                        != live_stage_output.stage_source_seal_sha256
                    ):
                        raise SecGemmaLeanV36AcquisitionError(
                            "source_replay_rejected"
                        )
                    public_path = self._persist_public_authority(
                        checkpoint_file_sha256=checkpoint_file_sha,
                        checkpoint=checkpoint,
                        replay_receipt=replay_receipt,
                    )
                except SecGemmaLeanV36AcquisitionError:
                    raise
                except Exception as error:
                    self._reject_clean_boundary(
                        _source_diagnostic_code(error)
                    )
                try:
                    self._journal.seal_terminal_pass(
                        checkpoint_sha256=checkpoint_file_sha
                    )
                except SecGemmaLeanV36JournalError as error:
                    raise SecGemmaLeanV36AcquisitionError(error.code) from None
                self._checkpoint_and_public_inventory()
                return self._terminal_result(
                    records=records,
                    checkpoint_path=checkpoint_path,
                    checkpoint_file_sha256=checkpoint_file_sha,
                    checkpoint=checkpoint,
                    public_path=public_path,
                )
            raise SecGemmaLeanV36AcquisitionError("phase_state_invalid")


def _production_git_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _production_git_binary() -> str:
    candidate_text = shutil.which("git")
    if candidate_text is None:
        raise SecGemmaLeanV36AcquisitionError(
            "production_git_check_unavailable"
        )
    try:
        candidate = Path(candidate_text)
        if not candidate.is_absolute():
            raise OSError("git path is not absolute")
        absolute = candidate.absolute()
        resolved = absolute.resolve(strict=True)
        details = absolute.lstat()
        if (
            absolute != resolved
            or not resolved.is_absolute()
            or not stat.S_ISREG(details.st_mode)
            or stat.S_ISLNK(details.st_mode)
            or _is_reparse(details)
            or not os.access(resolved, os.X_OK)
        ):
            raise OSError("git executable is unsafe")
    except OSError:
        raise SecGemmaLeanV36AcquisitionError(
            "production_git_check_unavailable"
        ) from None
    return str(resolved)


def _production_git(repo_root: Path, *arguments: str) -> bytes:
    git_binary = _production_git_binary()
    try:
        completed = subprocess.run(
            [git_binary, *arguments],
            cwd=repo_root,
            env=_production_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        raise SecGemmaLeanV36AcquisitionError(
            "production_git_check_unavailable"
        ) from None
    if completed.returncode != 0:
        raise SecGemmaLeanV36AcquisitionError(
            "production_git_gate_invalid"
        )
    return bytes(completed.stdout)


def _production_git_text(repo_root: Path, *arguments: str) -> str:
    try:
        return _production_git(repo_root, *arguments).decode(
            "utf-8", errors="strict"
        ).strip()
    except UnicodeDecodeError:
        raise SecGemmaLeanV36AcquisitionError(
            "production_git_gate_invalid"
        ) from None


def _read_production_json(
    path: Path, *, maximum_bytes: int, code: str
) -> tuple[dict[str, Any], bytes]:
    try:
        details = _secure_regular_file(path)
        if not 2 <= details.st_size <= maximum_bytes:
            raise SecGemmaLeanV36AcquisitionError(code)
        raw = path.read_bytes()
        if len(raw) != details.st_size:
            raise SecGemmaLeanV36AcquisitionError(code)

        def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in values:
                if key in result:
                    raise SecGemmaLeanV36AcquisitionError(code)
                result[key] = value
            return result

        def reject_constant(_value: str) -> Any:
            raise SecGemmaLeanV36AcquisitionError(code)

        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except SecGemmaLeanV36AcquisitionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaLeanV36AcquisitionError(code) from None
    if type(value) is not dict:
        raise SecGemmaLeanV36AcquisitionError(code)
    return value, raw


def _production_module_name(path: str) -> str:
    if path == "agent_benchmark/__init__.py":
        return "agent_benchmark"
    if not path.startswith("agent_benchmark/") or not path.endswith(".py"):
        raise SecGemmaLeanV36AcquisitionError(
            "production_execution_closure_invalid"
        )
    return path[:-3].replace("/", ".")


def _load_production_module_closure() -> dict[str, Any]:
    """Import the complete fixed local module set used by production."""

    loaded: dict[str, Any] = {}
    try:
        for path in PRODUCTION_MODULE_CLOSURE:
            module_name = _production_module_name(path)
            loaded[module_name] = importlib.import_module(module_name)
    except Exception:
        raise SecGemmaLeanV36AcquisitionError(
            "production_execution_closure_invalid"
        ) from None
    return loaded


def _verify_production_module_closure(
    repo_root: Path,
    head: str,
    *,
    loaded_modules: Mapping[str, Any] | None = None,
    blob_reader: Callable[[Path, str, str], bytes] | None = None,
) -> dict[str, str]:
    """Bind every loaded local production module to its exact HEAD bytes."""

    modules = sys.modules if loaded_modules is None else loaded_modules
    read_blob = (
        (lambda root, commit, path: _production_git(
            root, "cat-file", "blob", f"{commit}:{path}"
        ))
        if blob_reader is None
        else blob_reader
    )
    try:
        root = _secure_directory(repo_root.absolute())
        if root != _repository_root():
            raise SecGemmaLeanV36AcquisitionError(
                "production_execution_closure_invalid"
            )
        if re.fullmatch(r"[0-9a-f]{40}", head) is None:
            raise SecGemmaLeanV36AcquisitionError(
                "production_execution_closure_invalid"
            )
        if PRODUCTION_MODULE_CLOSURE != tuple(
            sorted(set(PRODUCTION_MODULE_CLOSURE))
        ):
            raise SecGemmaLeanV36AcquisitionError(
                "production_execution_closure_invalid"
            )
        hashes: dict[str, str] = {}
        for relative in PRODUCTION_MODULE_CLOSURE:
            module_name = _production_module_name(relative)
            module = modules.get(module_name)
            if module is None:
                raise SecGemmaLeanV36AcquisitionError(
                    "production_execution_closure_invalid"
                )
            expected = (root / Path(relative)).absolute()
            details = _secure_regular_file(expected)
            if expected.resolve(strict=True) != expected:
                raise SecGemmaLeanV36AcquisitionError(
                    "production_execution_closure_invalid"
                )
            actual = Path(getattr(module, "__file__", "")).resolve(
                strict=True
            )
            if actual != expected:
                raise SecGemmaLeanV36AcquisitionError(
                    "production_execution_closure_invalid"
                )
            raw = expected.read_bytes()
            if len(raw) != details.st_size or len(raw) > 16 * 1024 * 1024:
                raise SecGemmaLeanV36AcquisitionError(
                    "production_execution_closure_invalid"
                )
            committed = read_blob(root, head, relative)
            if type(committed) is not bytes or committed != raw:
                raise SecGemmaLeanV36AcquisitionError(
                    "production_execution_closure_invalid"
                )
            hashes[relative] = _sha256_bytes(raw)
        return hashes
    except SecGemmaLeanV36AcquisitionError as error:
        if error.code == "production_execution_closure_invalid":
            raise
        raise SecGemmaLeanV36AcquisitionError(
            "production_execution_closure_invalid"
        ) from None
    except (OSError, RuntimeError, TypeError, ValueError):
        raise SecGemmaLeanV36AcquisitionError(
            "production_execution_closure_invalid"
        ) from None


def _verify_production_development_gate(
    repo_root: Path,
) -> _ProductionDevelopmentGate:
    """Authenticate the committed/pushed zero-effect preflight and contact."""

    try:
        root = _secure_directory(repo_root.absolute())
        if root != _repository_root() or not (root / ".git").exists():
            raise SecGemmaLeanV36AcquisitionError(
                "production_repository_invalid"
            )
        private_root = _secure_directory(root / "data" / ROOT_NAMESPACE)
        public_root = _secure_directory(root / "e" / ROOT_NAMESPACE)
    except SecGemmaLeanV36AcquisitionError:
        raise
    except OSError:
        raise SecGemmaLeanV36AcquisitionError(
            "production_repository_invalid"
        ) from None
    try:
        from agent_benchmark import sec_gemma_lean_v36_preflight as preflight

        validation = preflight.validate_committed_preflight_for_acquisition(
            root
        )
    except Exception:
        raise SecGemmaLeanV36AcquisitionError(
            "production_preflight_invalid"
        ) from None
    expected_validation_keys = {
        "schema_version",
        "validated",
        "read_only_validation",
        "evidence_commit",
        "evidence_tree",
        "implementation_commit",
        "implementation_tree",
        "preflight_sha256",
        "implementation_delta_manifest_sha256",
        "contact_sha256",
        "durable_probe_state",
        "artifact_only_evidence_commit",
        "raw_artifact_equals_committed_blob",
        "current_branch_upstream_clean",
    }
    if (
        type(validation) is not dict
        or set(validation) != expected_validation_keys
        or validation.get("schema_version")
        != "aapl-sec-gemma-lean-evidence-v3-6-acquisition-gate-v1"
        or validation.get("validated") is not True
        or validation.get("read_only_validation") is not True
        or validation.get("artifact_only_evidence_commit") is not True
        or validation.get("raw_artifact_equals_committed_blob") is not True
        or validation.get("current_branch_upstream_clean") is not True
        or re.fullmatch(
            r"[0-9a-f]{40}", validation.get("evidence_commit", "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}", validation.get("evidence_tree", "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}", validation.get("implementation_commit", "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{40}", validation.get("implementation_tree", "")
        )
        is None
        or _SHA256_RE.fullmatch(validation.get("preflight_sha256", ""))
        is None
        or _SHA256_RE.fullmatch(
            validation.get("implementation_delta_manifest_sha256", "")
        )
        is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", validation.get("contact_sha256", "")
        )
        is None
        or type(validation.get("durable_probe_state")) is not dict
    ):
        raise SecGemmaLeanV36AcquisitionError(
            "production_preflight_invalid"
        )

    try:
        shown_root = Path(
            _production_git_text(root, "rev-parse", "--show-toplevel")
        ).resolve(strict=True)
    except (OSError, SecGemmaLeanV36AcquisitionError):
        raise SecGemmaLeanV36AcquisitionError(
            "production_repository_invalid"
        ) from None
    if shown_root != root:
        raise SecGemmaLeanV36AcquisitionError(
            "production_repository_invalid"
        )
    try:
        tracked_config = _production_git(
            root, "ls-files", "--", PRIVATE_CONFIG_PATH
        )
        _production_git(
            root,
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            PRIVATE_CONFIG_PATH,
        )
    except SecGemmaLeanV36AcquisitionError:
        raise SecGemmaLeanV36AcquisitionError(
            "private_config_not_private"
        ) from None
    if tracked_config:
        raise SecGemmaLeanV36AcquisitionError(
            "private_config_not_private"
        )

    config_path = root / Path(PRIVATE_CONFIG_PATH)
    config, config_raw = _read_production_json(
        config_path,
        maximum_bytes=1024 * 1024,
        code="private_config_invalid",
    )
    try:
        secrets = config.get("secrets")
        contact_text = (
            secrets.get("sec_user_agent") if type(secrets) is dict else None
        )
        contact = PrivateSecContact(contact_text)
    except Exception:
        raise SecGemmaLeanV36AcquisitionError(
            "private_contact_invalid"
        ) from None
    finally:
        config_raw = b""
        config = {}
    if (
        validation["contact_sha256"]
        != f"sha256:{contact.fingerprint_sha256}"
    ):
        raise SecGemmaLeanV36AcquisitionError(
            "private_contact_changed"
        )
    return _ProductionDevelopmentGate(
        repository_root=root,
        private_root=private_root,
        public_root=public_root,
        authorized_head=validation["evidence_commit"],
        preflight_sha256=validation["preflight_sha256"],
        private_contact=contact_text,
    )


def _run_development_with_gate(
    gate: _ProductionDevelopmentGate,
    *,
    transport_factory: TransportFactory,
    source_adapter: SourceAdapter,
    dispatch_ledger: SharedSecDispatchLedger,
    monotonic: Callable[[], float] = time.monotonic,
    acquisition_factory: type[DiskBackedSecAcquisition] = DiskBackedSecAcquisition,
) -> AcquisitionResult:
    if (
        type(gate) is not _ProductionDevelopmentGate
        or gate.private_root
        != gate.repository_root / "data" / ROOT_NAMESPACE
        or gate.public_root != gate.repository_root / "e" / ROOT_NAMESPACE
        or dispatch_ledger.ledger_path
        != gate.private_root / GLOBAL_LEDGER_NAME
    ):
        raise SecGemmaLeanV36AcquisitionError(
            "production_gate_invalid"
        )
    acquisition = acquisition_factory(
        stage="development",
        private_root=gate.private_root,
        public_root=gate.public_root,
        private_contact=gate.private_contact,
        transport_factory=transport_factory,
        source_adapter=source_adapter,
        prior=None,
        dispatch_ledger=dispatch_ledger,
        monotonic=monotonic,
    )
    return acquisition.run()


def run_production_acquisition(
    stage: str, repo_root: Path | None = None
) -> AcquisitionResult:
    """Run the fixed production development source acquisition.

    Intermediate/final are deliberately unavailable here until their separate
    scientific authorization verifier is integrated.  They cannot be reached
    by supplying a caller-built prior object.
    """

    if stage not in _STAGES:
        raise SecGemmaLeanV36AcquisitionError("stage_invalid")
    if stage != "development":
        raise SecGemmaLeanV36AcquisitionError(
            "later_stage_scientific_authorization_unavailable"
        )
    root = _repository_root() if repo_root is None else repo_root
    gate = _verify_production_development_gate(root)
    loaded = _load_production_module_closure()
    source_adapter = loaded["agent_benchmark.sec_gemma_lean_v36_source"]
    _verify_production_module_closure(
        gate.repository_root,
        gate.authorized_head,
        loaded_modules=loaded,
    )
    # Recheck every mutable repository identity fact after imports and raw
    # source binding, immediately before constructing the transport/runner.
    if (
        _production_git_text(gate.repository_root, "branch", "--show-current")
        != PRODUCTION_BRANCH
        or _production_git_text(
            gate.repository_root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        != PRODUCTION_UPSTREAM
        or _production_git_text(gate.repository_root, "rev-parse", "HEAD")
        != gate.authorized_head
        or _production_git_text(
            gate.repository_root, "rev-parse", "@{upstream}"
        )
        != gate.authorized_head
        or _production_git(
            gate.repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    ):
        raise SecGemmaLeanV36AcquisitionError(
            "production_repository_changed"
        )

    ledger = SharedSecDispatchLedger(
        ledger_path=gate.private_root / GLOBAL_LEDGER_NAME
    )
    return _run_development_with_gate(
        gate,
        transport_factory=strict_transport_factory(gate.private_contact),
        source_adapter=source_adapter,
        dispatch_ledger=ledger,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="AAPL SEC/Gemma lean v3.6 source acquisition"
    )
    parser.add_argument(
        "stage", choices=("development", "intermediate", "final")
    )
    parser.add_argument("--repo-root", default=".")
    arguments = parser.parse_args(argv)
    try:
        result = run_production_acquisition(
            arguments.stage, Path(arguments.repo_root).absolute()
        )
    except SecGemmaLeanV36AcquisitionError as error:
        print(json.dumps({"status": "failed", "code": error.code}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "stage": result.stage,
                "checkpoint_file_sha256": result.checkpoint_file_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ACQUISITION_SCHEMA_VERSION",
    "AcquisitionResult",
    "DiskBackedSecAcquisition",
    "DispatchReceipt",
    "GLOBAL_LEDGER_NAME",
    "GLOBAL_SEC_DISPATCH_LEDGER",
    "PARSE_RECEIPT_SCHEMA_VERSION",
    "PHASE_RECEIPT_SCHEMA_VERSION",
    "ROOT_NAMESPACE",
    "PRODUCTION_PREFLIGHT_PATH",
    "SecGemmaLeanV36AcquisitionError",
    "SharedSecDispatchLedger",
    "SourceAdapter",
    "Transport",
    "TransportFactory",
    "main",
    "run_production_acquisition",
    "strict_transport_factory",
]


if __name__ == "__main__":
    raise SystemExit(main())
