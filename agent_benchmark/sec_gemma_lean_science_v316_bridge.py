"""Read-only v3.8-to-v3.16 SEC science bridge.

The module deliberately has no network, model, market-data, clock, or write
surface.  A filesystem authority is accepted only after the existing v3.8
hash-chained journal and compact replay have been authenticated.  The second
pass then keeps at most one complete-submission response alive while it
recreates the exact legacy selected-document inputs.

Private identities are retained only in the returned in-process projection.
The projection manifest is deliberately aggregate-only and is safe for a
later redacted public artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Final, Protocol
from urllib.parse import quote

from .sec_filing_content import normalize_filing_text
from .sec_gemma_lean_v38_journal import (
    PrivateSecContact,
    SecGemmaLeanV38JournalError,
    validate_detached_journal,
)
from . import sec_gemma_lean_v38_journal as _v38_journal
from . import sec_gemma_lean_v38_source as _v38_source
from .sec_gemma_lean_v38_source import LegacyScienceDocument
from .sec_gemma_lean_v38_transport import validate_transport_receipt


BRIDGE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-16-streaming-bridge-v1"
)
_STAGE: Final[str] = "development"
_PRIVATE_NAMESPACE: Final[str] = "aapl_sec_gemma_lean_evidence_v3_8"
_ACCESSION_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z"
)
_BARE_SHA_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA_RE: Final[re.Pattern[str]] = re.compile(
    r"sha256:(?P<digest>[0-9a-f]{64})\Z"
)
_SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z"
)
_CHECKPOINT_RE: Final[re.Pattern[str]] = re.compile(
    r"checkpoint-(?P<sha>[0-9a-f]{64})\.json\Z"
)
_ROLE_FILE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<sequence>[0-9]{6})-(?P<sha>[0-9a-f]{64})\.(?P<suffix>json|blob)\Z"
)
_PHASE_FILE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<phase>submissions|reconciliation)-(?P<sha>[0-9a-f]{64})\.json\Z"
)
_PUBLIC_AUTHORITY_RE: Final[re.Pattern[str]] = re.compile(
    r"source-authority-(?P<sha>[0-9a-f]{64})\.json\Z"
)
_REPARSE_ATTRIBUTE: Final[int] = 0x400
_AUTHORITY_TOKEN: Final[object] = object()


class BridgeViolation(RuntimeError):
    """One fixed, redacted bridge rejection code."""

    def __init__(self, code: str) -> None:
        if type(code) is not str or re.fullmatch(r"[a-z0-9_]{3,80}", code) is None:
            code = "bridge_rejected"
        super().__init__(code)
        self.code = code


class _SourceApi(Protocol):
    def rehydrate_compact_stage(
        self,
        stage: str,
        checkpoint: Mapping[str, Any],
        role_manifests: Sequence[Mapping[str, Any]],
        load_blob: Callable[[str], bytes],
        prior: Any | None = None,
    ) -> Any: ...

    def parse_complete(
        self,
        stage: str,
        target: Mapping[str, Any],
        payload: bytes,
        reconciliation_evidence: Mapping[str, Any],
    ) -> Any: ...


@dataclass(frozen=True)
class _AuthorityExpectations:
    checkpoint_file_sha256: str
    logical_checkpoint_sha256: str
    stage_source_seal_sha256: str
    compact_replay_sha256: str
    role_manifests_sha256: str
    role_plan_sha256: str
    document_count: int
    inventory_file_count: int | None = None
    inventory_byte_count: int | None = None
    inventory_sha256: str | None = None


@dataclass(frozen=True)
class _FileSnapshot:
    category: str
    ordinal: int
    path: Path = field(repr=False, compare=False)
    sha256: str
    byte_count: int
    device: int = field(repr=False)
    inode: int = field(repr=False)
    links: int = field(repr=False)
    modified_ns: int = field(repr=False)
    changed_ns: int = field(repr=False)

    def identity(self) -> tuple[Any, ...]:
        return (
            self.category,
            self.ordinal,
            self.path,
            self.sha256,
            self.byte_count,
            self.device,
            self.inode,
            self.links,
            self.modified_ns,
            self.changed_ns,
        )


@dataclass(frozen=True, init=False)
class AuthenticatedV38Authority:
    """Opaque, authenticated v3.8 development authority.

    Direct construction is disabled.  Production values come from
    :func:`authenticate_v38_private_root`; tests may exercise the same compact
    replay boundary through the private dependency-injected factory below.
    """

    checkpoint: Mapping[str, Any] = field(repr=False)
    role_manifests: tuple[Mapping[str, Any], ...] = field(repr=False)
    stage_output: Any = field(repr=False)
    compact_replay_receipt: Mapping[str, Any] = field(repr=False)
    public_authority: Mapping[str, Any] = field(repr=False)
    _expectations: _AuthorityExpectations = field(repr=False)
    _blob_loader: Callable[[str], bytes] = field(repr=False, compare=False)
    _source_api: _SourceApi = field(repr=False, compare=False)
    _stage_output_sha256: str = field(repr=False)
    _inventory: tuple[_FileSnapshot, ...] | None = field(
        repr=False, compare=False
    )
    _inventory_builder: Callable[[], tuple[_FileSnapshot, ...]] | None = field(
        repr=False, compare=False
    )
    _token: object = field(repr=False, compare=False)

    @classmethod
    def _create(
        cls,
        *,
        checkpoint: Mapping[str, Any],
        role_manifests: Sequence[Mapping[str, Any]],
        stage_output: Any,
        compact_replay_receipt: Mapping[str, Any],
        public_authority: Mapping[str, Any],
        expectations: _AuthorityExpectations,
        blob_loader: Callable[[str], bytes],
        source_api: _SourceApi,
        inventory: tuple[_FileSnapshot, ...] | None = None,
        inventory_builder: Callable[[], tuple[_FileSnapshot, ...]] | None = None,
    ) -> "AuthenticatedV38Authority":
        value = object.__new__(cls)
        object.__setattr__(value, "checkpoint", _json_copy(checkpoint))
        object.__setattr__(
            value,
            "role_manifests",
            tuple(_json_copy(item) for item in role_manifests),
        )
        object.__setattr__(value, "stage_output", stage_output)
        object.__setattr__(
            value, "compact_replay_receipt", _json_copy(compact_replay_receipt)
        )
        object.__setattr__(value, "public_authority", _json_copy(public_authority))
        object.__setattr__(value, "_expectations", expectations)
        object.__setattr__(value, "_blob_loader", blob_loader)
        object.__setattr__(value, "_source_api", source_api)
        object.__setattr__(
            value, "_stage_output_sha256", _stage_output_sha256(stage_output)
        )
        object.__setattr__(value, "_inventory", inventory)
        object.__setattr__(value, "_inventory_builder", inventory_builder)
        object.__setattr__(value, "_token", _AUTHORITY_TOKEN)
        return value


@dataclass(frozen=True)
class ScienceBridgeProjection:
    """The private exact bridge result plus an aggregate-only manifest."""

    stage: str
    legacy_documents: tuple[LegacyScienceDocument, ...] = field(repr=False)
    compatibility_records: tuple[Mapping[str, Any], ...] = field(repr=False)
    compatibility_manifest: Mapping[str, Any] = field(repr=False)
    records: tuple[Mapping[str, Any], ...] = field(repr=False)
    primary_documents: tuple[Mapping[str, Any], ...] = field(repr=False)
    universe: Mapping[str, Any] = field(repr=False)
    content_manifest: Mapping[str, Any] = field(repr=False)
    universe_event_proofs: tuple[Mapping[str, Any], ...] = field(repr=False)
    source_order: tuple[Mapping[str, Any], ...] = field(repr=False)
    event_order: tuple[Mapping[str, Any], ...] = field(repr=False)
    prior_links: tuple[Mapping[str, Any], ...] = field(repr=False)
    legacy_projection_manifest: Mapping[str, Any] = field(repr=False)
    manifest: Mapping[str, Any]
    manifest_json: bytes = field(repr=False)


def _contract_module() -> Any:
    # The contract is a separate allowlisted v3.16 file and may be written by a
    # parallel implementation task.  A local import keeps this module
    # importable while preserving a strict runtime dependency.
    try:
        from . import sec_gemma_lean_science_v316_contract as contract
    except Exception:
        raise BridgeViolation("contract_unavailable") from None
    return contract


def _canonical_bytes(value: Any, *, ensure_ascii: bool = True) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ).encode("ascii" if ensure_ascii else "utf-8")
    except Exception:
        raise BridgeViolation("canonical_value_invalid") from None


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _v316_sha(value: Any) -> str:
    try:
        return _contract_module().canonical_sha256(value)
    except BridgeViolation:
        raise
    except Exception:
        raise BridgeViolation("v316_canonical_value_invalid") from None


def _v316_self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise BridgeViolation("v316_self_hash_invalid")
    body = dict(value)
    return {**body, field: _v316_sha(body)}


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(_canonical_bytes(value).decode("ascii"))
    except BridgeViolation:
        raise
    except Exception:
        raise BridgeViolation("canonical_value_invalid") from None


def _file_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bare_sha(value: Any, *, code: str = "hash_invalid") -> str:
    if type(value) is str and _BARE_SHA_RE.fullmatch(value) is not None:
        return value
    match = _TAGGED_SHA_RE.fullmatch(value) if type(value) is str else None
    if match is None:
        raise BridgeViolation(code)
    return match.group("digest")


def _source_expectations() -> _AuthorityExpectations:
    contract = _contract_module()
    try:
        return _AuthorityExpectations(
            checkpoint_file_sha256=contract.V38_CHECKPOINT_FILE_SHA256,
            logical_checkpoint_sha256=contract.V38_LOGICAL_CHECKPOINT_SHA256,
            stage_source_seal_sha256=contract.V38_STAGE_SOURCE_SEAL_SHA256,
            compact_replay_sha256=contract.V38_COMPACT_REPLAY_SHA256,
            role_manifests_sha256=contract.V38_ROLE_MANIFEST_INVENTORY_SHA256,
            role_plan_sha256=contract.V38_ROLE_PLAN_SHA256,
            document_count=contract.DEVELOPMENT_DOCUMENT_COUNT,
            inventory_file_count=contract.V38_INVENTORY_FILE_COUNT,
            inventory_byte_count=contract.V38_INVENTORY_BYTE_COUNT,
            inventory_sha256=contract.V38_INVENTORY_SHA256,
        )
    except Exception:
        raise BridgeViolation("contract_source_pins_invalid") from None


def _is_reparse(details: os.stat_result) -> bool:
    return bool(getattr(details, "st_file_attributes", 0) & _REPARSE_ATTRIBUTE)


def _secure_directory(path: Path) -> None:
    try:
        details = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        raise BridgeViolation("private_authority_layout_invalid") from None
    if (
        not path.is_absolute()
        or resolved != path
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
    ):
        raise BridgeViolation("private_authority_layout_invalid")


def _secure_file(path: Path) -> os.stat_result:
    try:
        details = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        raise BridgeViolation("private_authority_artifact_invalid") from None
    if (
        not path.is_absolute()
        or resolved != path
        or not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or details.st_nlink != 1
    ):
        raise BridgeViolation("private_authority_artifact_invalid")
    return details


def _stable_path_read(
    path: Path,
    *,
    contact: PrivateSecContact | None = None,
    capture_bytes: bool = False,
    minimum_size: int | None = None,
    maximum_size: int | None = None,
) -> tuple[str, int, os.stat_result, bytes | None]:
    """Hash and optionally retain bytes from one stable no-follow descriptor."""

    before = _secure_file(path)
    if (
        (minimum_size is not None and before.st_size < minimum_size)
        or (maximum_size is not None and before.st_size > maximum_size)
    ):
        raise BridgeViolation("private_authority_artifact_invalid")
    digest = hashlib.sha256()
    counted = 0
    carry = b""
    captured: list[bytes] | None = [] if capture_bytes else None
    max_needle = 0
    if contact is not None:
        max_needle = max(map(len, contact.serialized_echo_needles()), default=0)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOINHERIT", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_reparse(opened)
            or opened.st_nlink != 1
        ):
            raise BridgeViolation("private_authority_artifact_invalid")
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            counted += len(chunk)
            digest.update(chunk)
            if captured is not None:
                captured.append(chunk)
            if contact is not None:
                candidate = carry + chunk
                if contact.contains_echo(candidate):
                    raise BridgeViolation("private_contact_echo")
                carry = candidate[-(max_needle - 1) :] if max_needle > 1 else b""
        opened_after = os.fstat(descriptor)
    except BridgeViolation:
        raise
    except OSError:
        raise BridgeViolation("private_authority_artifact_invalid") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    after = _secure_file(path)
    # On Windows, CPython may expose creation time through ``Path.lstat()``
    # while ``os.fstat()`` on the already-open handle exposes last-write time
    # through ``st_ctime_ns``.  Compare ctime only across the two path-visible
    # snapshots, whose API semantics match, while retaining the descriptor
    # identity/size/mtime/link checks that bind the bytes read above.
    visible_stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    descriptor_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    ) == (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_nlink,
    )
    descriptor_stable = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_nlink,
    ) == (
        opened_after.st_dev,
        opened_after.st_ino,
        opened_after.st_size,
        opened_after.st_mtime_ns,
        opened_after.st_nlink,
    )
    if (
        not visible_stable
        or not descriptor_identity
        or not descriptor_stable
        or counted != before.st_size
    ):
        raise BridgeViolation("private_authority_changed_during_read")
    raw = b"".join(captured) if captured is not None else None
    return digest.hexdigest(), counted, after, raw


def _hash_path(
    path: Path, *, contact: PrivateSecContact | None = None
) -> tuple[str, int, os.stat_result]:
    digest, counted, after, _ = _stable_path_read(path, contact=contact)
    return digest, counted, after


def _read_stable_bytes(
    path: Path,
    *,
    contact: PrivateSecContact | None = None,
    minimum_size: int | None = None,
    maximum_size: int | None = None,
) -> tuple[bytes, str, int, os.stat_result]:
    digest, counted, after, raw = _stable_path_read(
        path,
        contact=contact,
        capture_bytes=True,
        minimum_size=minimum_size,
        maximum_size=maximum_size,
    )
    if type(raw) is not bytes:
        raise BridgeViolation("private_authority_artifact_invalid")
    return raw, digest, counted, after


def _read_canonical_mapping(
    path: Path, *, contact: PrivateSecContact | None = None
) -> dict[str, Any]:
    value, _, _, _ = _read_canonical_mapping_with_identity(path, contact=contact)
    return value


def _read_canonical_mapping_with_identity(
    path: Path, *, contact: PrivateSecContact | None = None
) -> tuple[dict[str, Any], str, int, os.stat_result]:
    raw, digest, size, details = _read_stable_bytes(
        path,
        contact=contact,
        minimum_size=2,
        maximum_size=512 * 1024 * 1024,
    )
    try:
        value = json.loads(raw.decode("ascii", errors="strict"))
    except Exception:
        raise BridgeViolation("private_authority_artifact_invalid") from None
    if type(value) is not dict or _canonical_bytes(value) != raw:
        raise BridgeViolation("private_authority_artifact_invalid")
    return value, digest, size, details


def _read_authenticated_blob(
    path: Path,
    *,
    expected_sha256: Any,
    expected_size: Any,
    contact: PrivateSecContact | None = None,
) -> bytes:
    """Authenticate and return one blob from the same stable descriptor read."""

    try:
        if (
            type(expected_size) is not int
            or expected_size <= 0
            or expected_size > 512 * 1024 * 1024
        ):
            raise BridgeViolation("blob_identity_invalid")
        expected_digest = _bare_sha(
            expected_sha256, code="blob_identity_invalid"
        )
        payload, digest, size, _ = _read_stable_bytes(
            path,
            contact=contact,
            minimum_size=expected_size,
            maximum_size=expected_size,
        )
    except BridgeViolation:
        raise BridgeViolation("blob_identity_invalid") from None
    if digest != expected_digest or size != expected_size:
        raise BridgeViolation("blob_identity_invalid")
    return payload


def _git(repo_root: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=NUL", *arguments],
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30,
        )
        return result.stdout.decode("ascii", errors="strict").strip()
    except Exception:
        raise BridgeViolation("repository_authority_invalid") from None


def _git_blob(repo_root: Path, commit: str, path: str) -> str:
    raw = _git(repo_root, "ls-tree", commit, "--", path)
    parts = raw.split()
    if len(parts) < 4 or parts[1] != "blob" or not re.fullmatch(r"[0-9a-f]{40}", parts[2]):
        raise BridgeViolation("repository_authority_invalid")
    return parts[2]


def _authenticate_public_source(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _contract_module()
    try:
        if _git(repo_root, "rev-parse", f"{contract.V38_SOURCE_COMMIT}^{{tree}}") != contract.V38_SOURCE_TREE:
            raise BridgeViolation("repository_authority_invalid")
        checks = (
            (contract.V38_TERMINAL_PATH, contract.V38_TERMINAL_BLOB_SHA1),
            (
                contract.V38_SOURCE_AUTHORITY_PATH,
                contract.V38_SOURCE_AUTHORITY_BLOB_SHA1,
            ),
            (
                "agent_benchmark/sec_gemma_lean_v38_source.py",
                contract.V38_SOURCE_MODULE_GIT_BLOB_SHA1,
            ),
            (
                "agent_benchmark/sec_gemma_lean_v38_acquisition.py",
                contract.V38_ACQUISITION_MODULE_GIT_BLOB_SHA1,
            ),
        )
        for path, expected in checks:
            if _git_blob(repo_root, contract.V38_SOURCE_COMMIT, path) != expected:
                raise BridgeViolation("repository_authority_invalid")
        literal_checks = (
            (contract.V38_TERMINAL_PATH, contract.V38_TERMINAL_LITERAL_SHA256),
            (
                contract.V38_SOURCE_AUTHORITY_PATH,
                contract.V38_SOURCE_AUTHORITY_LITERAL_SHA256,
            ),
            (
                "agent_benchmark/sec_gemma_lean_v38_source.py",
                contract.V38_SOURCE_MODULE_LITERAL_SHA256,
            ),
            (
                "agent_benchmark/sec_gemma_lean_v38_acquisition.py",
                contract.V38_ACQUISITION_MODULE_LITERAL_SHA256,
            ),
        )
        authenticated_payloads: dict[str, bytes] = {}
        parsed_paths = {
            contract.V38_TERMINAL_PATH,
            contract.V38_SOURCE_AUTHORITY_PATH,
        }
        for relative, expected in literal_checks:
            path = (repo_root / Path(relative)).resolve(strict=True)
            if repo_root not in path.parents:
                raise BridgeViolation("repository_authority_invalid")
            if relative in parsed_paths:
                raw, digest, _, _ = _read_stable_bytes(
                    path, minimum_size=2, maximum_size=512 * 1024 * 1024
                )
                authenticated_payloads[relative] = raw
            else:
                digest, _, _ = _hash_path(path)
            if digest != expected:
                raise BridgeViolation("repository_authority_invalid")
        terminal = json.loads(
            authenticated_payloads[contract.V38_TERMINAL_PATH].decode(
                "utf-8", errors="strict"
            )
        )
        public = json.loads(
            authenticated_payloads[contract.V38_SOURCE_AUTHORITY_PATH].decode(
                "ascii", errors="strict"
            )
        )
        if type(terminal) is not dict or type(public) is not dict:
            raise BridgeViolation("public_source_authority_invalid")
        supplied = terminal.get("acquisition_sha256")
        unsigned = dict(terminal)
        unsigned.pop("acquisition_sha256", None)
        if (
            supplied != contract.V38_TERMINAL_INTERNAL_SHA256
            or _canonical_sha(unsigned) != supplied
        ):
            raise BridgeViolation("public_source_authority_invalid")
        return terminal, public
    except BridgeViolation:
        raise
    except Exception:
        raise BridgeViolation("public_source_authority_invalid") from None


def _list_exact_files(directory: Path) -> list[Path]:
    _secure_directory(directory)
    try:
        values = sorted(directory.iterdir(), key=lambda item: item.name)
    except OSError:
        raise BridgeViolation("private_authority_layout_invalid") from None
    for value in values:
        _secure_file(value)
    return values


def _inventory_digest(rows: Sequence[_FileSnapshot]) -> str:
    # This is the exact public v3.8 terminal basis: category, zero-based
    # ordinal, raw file SHA-256, and byte count, in category/ordinal order.
    # Preserve the historical serialized key spellings exactly.
    material = [
        {
            "category": row.category,
            "ordinal": row.ordinal,
            "sha256": row.sha256,
            "bytes": row.byte_count,
        }
        for row in rows
    ]
    return _canonical_sha(material)


def _snapshot_inventory(
    *,
    private_stage: Path,
    public_stage: Path,
    contact: PrivateSecContact | None,
) -> tuple[_FileSnapshot, ...]:
    expected_directories = {
        "temporary",
        "blobs",
        "transport_receipts",
        "parse_receipts",
        "manifests",
        "phase_receipts",
        "journal",
    }
    _secure_directory(private_stage)
    _secure_directory(public_stage)
    try:
        top = list(private_stage.iterdir())
    except OSError:
        raise BridgeViolation("private_authority_layout_invalid") from None
    observed_directories = {item.name for item in top if item.is_dir()}
    if observed_directories != expected_directories:
        raise BridgeViolation("private_authority_layout_invalid")
    checkpoints = [item for item in top if _CHECKPOINT_RE.fullmatch(item.name)]
    locks = [item for item in top if item.name == ".run.lock"]
    if len(checkpoints) != 1 or len(locks) != 1:
        raise BridgeViolation("private_authority_layout_invalid")
    allowed = expected_directories | {checkpoints[0].name, ".run.lock"}
    if {item.name for item in top} != allowed:
        raise BridgeViolation("private_authority_layout_invalid")
    categories: dict[str, list[Path]] = {
        "blob": _list_exact_files(private_stage / "blobs"),
        "checkpoint": checkpoints,
        "journal": _list_exact_files(private_stage / "journal"),
        "manifest": _list_exact_files(private_stage / "manifests"),
        "parse_receipt": _list_exact_files(private_stage / "parse_receipts"),
        "phase_receipt": _list_exact_files(private_stage / "phase_receipts"),
        "public_authority": _list_exact_files(public_stage),
        "run_lock": locks,
        "transport_receipt": _list_exact_files(
            private_stage / "transport_receipts"
        ),
    }
    expected_category_counts = {
        "blob": 199,
        "checkpoint": 1,
        "journal": 609,
        "manifest": 199,
        "parse_receipt": 199,
        "phase_receipt": 2,
        "public_authority": 1,
        "run_lock": 1,
        "transport_receipt": 199,
    }
    if {
        name: len(paths) for name, paths in categories.items()
    } != expected_category_counts:
        raise BridgeViolation("private_authority_layout_invalid")
    if _list_exact_files(private_stage / "temporary"):
        raise BridgeViolation("private_authority_layout_invalid")
    rows: list[_FileSnapshot] = []
    for category in sorted(categories):
        for ordinal, path in enumerate(categories[category]):
            digest, size, details = _hash_path(path, contact=contact)
            rows.append(
                _FileSnapshot(
                    category=category,
                    ordinal=ordinal,
                    path=path,
                    sha256=digest,
                    byte_count=size,
                    device=details.st_dev,
                    inode=details.st_ino,
                    links=details.st_nlink,
                    modified_ns=details.st_mtime_ns,
                    changed_ns=details.st_ctime_ns,
                )
            )
    run_lock = next(item for item in rows if item.category == "run_lock")
    if run_lock.byte_count != 1 or run_lock.sha256 != _file_sha(b"\0"):
        raise BridgeViolation("private_authority_layout_invalid")
    return tuple(rows)


def _validate_manifest(value: Mapping[str, Any], sequence: int) -> dict[str, Any]:
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
    item = _json_copy(value)
    supplied = item.get("role_manifest_sha256")
    unsigned = dict(item)
    unsigned.pop("role_manifest_sha256", None)
    if (
        set(item) != expected
        or item.get("sequence") != sequence
        or type(supplied) is not str
        or _BARE_SHA_RE.fullmatch(supplied) is None
        or _canonical_sha(unsigned) != supplied
        or type(item.get("role_id")) is not str
        or type(item.get("url")) is not str
        or type(item.get("blob_name")) is not str
        or _SAFE_NAME_RE.fullmatch(item["blob_name"]) is None
        or type(item.get("body_bytes")) is not int
        or item["body_bytes"] < 1
    ):
        raise BridgeViolation("role_manifest_invalid")
    _bare_sha(item["body_sha256"], code="role_manifest_invalid")
    for name in (
        "transport_receipt_sha256",
        "parse_receipt_sha256",
        "source_evidence_sha256",
    ):
        if type(item.get(name)) is not str or _BARE_SHA_RE.fullmatch(item[name]) is None:
            raise BridgeViolation("role_manifest_invalid")
    return item


def _validate_source_evidence(value: Any) -> dict[str, Any]:
    item = _json_copy(value)
    supplied = item.get("source_evidence_sha256")
    unsigned = dict(item)
    unsigned.pop("source_evidence_sha256", None)
    if (
        type(supplied) is not str
        or _BARE_SHA_RE.fullmatch(supplied) is None
        or _canonical_sha(unsigned) != supplied
    ):
        raise BridgeViolation("source_evidence_invalid")
    return item


def _authenticate_role_files(
    *,
    private_stage: Path,
    state: Any,
    contact: PrivateSecContact,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Path]]:
    directories = {
        "blob": private_stage / "blobs",
        "transport": private_stage / "transport_receipts",
        "parse": private_stage / "parse_receipts",
        "manifest": private_stage / "manifests",
    }
    files = {name: _list_exact_files(path) for name, path in directories.items()}
    count = state.role_seals
    if count != 199 or any(len(values) != count for values in files.values()):
        raise BridgeViolation("role_inventory_invalid")
    try:
        events = _v38_journal._read_events(private_stage / "journal")
    except Exception:
        raise BridgeViolation("journal_replay_invalid") from None
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {
        "role_intent": {},
        "response_complete": {},
        "role_seal": {},
    }
    for event in events:
        event_type = event.get("event_type")
        if event_type in grouped:
            role_id = event.get("payload", {}).get("role_id")
            if type(role_id) is not str or role_id in grouped[event_type]:
                raise BridgeViolation("journal_replay_invalid")
            grouped[event_type][role_id] = event
    manifests: list[dict[str, Any]] = []
    blob_paths: dict[str, Path] = {}
    for sequence, role in enumerate(state.planned_roles):
        prefix = f"{sequence:06d}-"
        chosen: dict[str, Path] = {}
        for category, values in files.items():
            matches = [path for path in values if path.name.startswith(prefix)]
            if len(matches) != 1:
                raise BridgeViolation("role_inventory_invalid")
            chosen[category] = matches[0]
        manifest_value, _, _, _ = _read_canonical_mapping_with_identity(
            chosen["manifest"], contact=contact
        )
        manifest = _validate_manifest(manifest_value, sequence)
        if manifest["role_id"] != role.role_id or manifest["url"] != role.url:
            raise BridgeViolation("role_inventory_invalid")
        parse_receipt, parse_hash, _, _ = _read_canonical_mapping_with_identity(
            chosen["parse"], contact=contact
        )
        parse_expected = {
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
        evidence = _validate_source_evidence(parse_receipt.get("source_evidence"))
        if (
            set(parse_receipt) != parse_expected
            or parse_receipt.get("stage") != _STAGE
            or parse_receipt.get("sequence") != sequence
            or parse_receipt.get("role_id") != role.role_id
            or parse_receipt.get("url") != role.url
            or parse_receipt.get("source_evidence_sha256")
            != evidence["source_evidence_sha256"]
            or parse_receipt.get("blob_name") != manifest["blob_name"]
            or parse_receipt.get("body_sha256")
            != _bare_sha(manifest["body_sha256"])
            or parse_receipt.get("body_bytes") != manifest["body_bytes"]
            or parse_receipt.get("transport_receipt_sha256")
            != manifest["transport_receipt_sha256"]
        ):
            raise BridgeViolation("parse_receipt_invalid")
        blob_hash, blob_bytes, _ = _hash_path(chosen["blob"], contact=contact)
        transport = _read_canonical_mapping(chosen["transport"], contact=contact)
        intent = grouped["role_intent"].get(role.role_id)
        response = grouped["response_complete"].get(role.role_id)
        seal = grouped["role_seal"].get(role.role_id)
        if intent is None or response is None or seal is None:
            raise BridgeViolation("journal_role_binding_invalid")
        try:
            http = transport["http"]
            header = transport["raw_response_headers"]
            metadata = {
                "request_url": role.url,
                "observed_url": http["observed_url"],
                "role_class": transport["role"]["role_class"],
                "role_id": role.role_id,
                "intent_event_sha256": intent["event_sha256"],
                "body_limit_bytes": role.body_limit_bytes,
                "temporary_blob_path": str(private_stage / "temporary" / "detached.tmp"),
                "status_code": http["status_code"],
                "framing": http["framing_mode"],
                "declared_content_length": http["declared_content_length"],
                "content_encoding": (
                    None
                    if http["content_encoding"] == "absent"
                    else http["content_encoding"]
                ),
                "body_bytes": manifest["body_bytes"],
                "body_sha256": _bare_sha(manifest["body_sha256"]),
                "raw_headers_sha256": header["sha256"],
                "raw_headers_bytes": header["byte_length"],
                "contact_fingerprint_sha256": contact.fingerprint_sha256,
                "execution_identity": transport["execution_identity"],
            }
            validate_transport_receipt(transport, expected_metadata=metadata)
        except Exception:
            raise BridgeViolation("transport_receipt_invalid") from None
        if (
            chosen["blob"].name != manifest["blob_name"]
            or blob_hash != _bare_sha(manifest["body_sha256"])
            or blob_bytes != manifest["body_bytes"]
            or parse_hash != manifest["parse_receipt_sha256"]
            or chosen["parse"].name
            != f"{sequence:06d}-{manifest['parse_receipt_sha256']}.json"
            or chosen["manifest"].name
            != f"{sequence:06d}-{manifest['role_manifest_sha256']}.json"
            or chosen["transport"].name
            != f"{sequence:06d}-{manifest['transport_receipt_sha256']}.json"
            or transport.get("transport_receipt_sha256")
            != manifest["transport_receipt_sha256"]
            or parse_receipt["intent_event_sha256"] != intent["event_sha256"]
            or parse_receipt["response_event_sha256"] != response["event_sha256"]
            or response["payload"].get("body_sha256") != blob_hash
            or response["payload"].get("body_bytes") != blob_bytes
            or response["payload"].get("transport_receipt_sha256")
            != manifest["transport_receipt_sha256"]
            or seal["payload"].get("parse_receipt_sha256") != parse_hash
            or seal["payload"].get("blob_sha256") != blob_hash
            or seal["payload"].get("body_bytes") != blob_bytes
        ):
            raise BridgeViolation("journal_role_binding_invalid")
        if manifest["blob_name"] in blob_paths:
            raise BridgeViolation("role_inventory_invalid")
        manifests.append(manifest)
        blob_paths[manifest["blob_name"]] = chosen["blob"]
    return tuple(manifests), blob_paths


def _authenticate_phase_receipts(
    *,
    private_stage: Path,
    state: Any,
    manifests: Sequence[Mapping[str, Any]],
    contact: PrivateSecContact,
) -> None:
    files = _list_exact_files(private_stage / "phase_receipts")
    if len(files) != 2:
        raise BridgeViolation("phase_receipt_invalid")
    by_phase: dict[str, tuple[dict[str, Any], str]] = {}
    for path in files:
        match = _PHASE_FILE_RE.fullmatch(path.name)
        if match is None or match.group("phase") in by_phase:
            raise BridgeViolation("phase_receipt_invalid")
        value, digest, _, _ = _read_canonical_mapping_with_identity(
            path, contact=contact
        )
        if digest != match.group("sha"):
            raise BridgeViolation("phase_receipt_invalid")
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
        evidence = _validate_source_evidence(value.get("source_evidence"))
        if (
            set(value) != expected
            or value.get("stage") != _STAGE
            or value.get("phase") != match.group("phase")
            or value.get("source_evidence_sha256") != evidence["source_evidence_sha256"]
            or value.get("prior_stage_source_seal_sha256") is not None
            or type(value.get("input_parse_receipt_sha256")) is not list
            or type(value.get("complete_targets")) is not list
        ):
            raise BridgeViolation("phase_receipt_invalid")
        by_phase[match.group("phase")] = (value, digest)
    if set(by_phase) != {"submissions", "reconciliation"}:
        raise BridgeViolation("phase_receipt_invalid")
    phases = [role.phase for role in state.planned_roles]
    submission_inputs = [
        manifests[i]["parse_receipt_sha256"]
        for i, phase in enumerate(phases)
        if phase in {"main_submissions", "historical_submissions"}
    ]
    master_inputs = [
        manifests[i]["parse_receipt_sha256"]
        for i, phase in enumerate(phases)
        if phase == "quarterly_master"
    ]
    submissions, submissions_hash = by_phase["submissions"]
    reconciliation, reconciliation_hash = by_phase["reconciliation"]
    if (
        submissions["input_parse_receipt_sha256"] != submission_inputs
        or submissions["upstream_phase_receipt_sha256"] is not None
        or submissions["complete_targets"] != []
        or reconciliation["input_parse_receipt_sha256"] != master_inputs
        or reconciliation["upstream_phase_receipt_sha256"] != submissions_hash
    ):
        raise BridgeViolation("phase_receipt_invalid")
    try:
        phase_events = [
            event
            for event in _v38_journal._read_events(private_stage / "journal")
            if event["event_type"] == "phase_plan"
        ]
        observed_bases = [
            event["payload"]["basis_receipt_sha256"] for event in phase_events
        ]
    except Exception:
        raise BridgeViolation("phase_receipt_invalid") from None
    if observed_bases != [
        manifests[0]["parse_receipt_sha256"],
        submissions_hash,
        reconciliation_hash,
    ]:
        raise BridgeViolation("phase_receipt_invalid")


def _authenticate_compact_replay(
    *,
    checkpoint_file_sha256: str,
    checkpoint: Mapping[str, Any],
    role_manifests: Sequence[Mapping[str, Any]],
    blob_loader: Callable[[str], bytes],
    source_api: _SourceApi,
    expectations: _AuthorityExpectations,
    public_authority: Mapping[str, Any] | None = None,
    inventory: tuple[_FileSnapshot, ...] | None = None,
    inventory_builder: Callable[[], tuple[_FileSnapshot, ...]] | None = None,
) -> AuthenticatedV38Authority:
    if not callable(blob_loader):
        raise BridgeViolation("blob_loader_invalid")
    manifests = tuple(_validate_manifest(item, index) for index, item in enumerate(role_manifests))
    if (
        checkpoint_file_sha256 != expectations.checkpoint_file_sha256
        or checkpoint.get("stage") != _STAGE
        or checkpoint.get("checkpoint_sha256") != expectations.logical_checkpoint_sha256
        or checkpoint.get("expected_stage_source_seal_sha256")
        != expectations.stage_source_seal_sha256
        or checkpoint.get("role_manifest_count") != len(manifests)
        or checkpoint.get("role_manifests_sha256")
        != expectations.role_manifests_sha256
        or _canonical_sha(list(manifests)) != expectations.role_manifests_sha256
    ):
        raise BridgeViolation("compact_checkpoint_invalid")
    try:
        replay = source_api.rehydrate_compact_stage(
            _STAGE,
            _json_copy(checkpoint),
            tuple(_json_copy(item) for item in manifests),
            blob_loader,
            None,
        )
        stage_output = replay.stage_output
        receipt = _json_copy(replay.receipt)
    except Exception:
        raise BridgeViolation("compact_replay_rejected") from None
    if (
        receipt.get("stage") != _STAGE
        or receipt.get("checkpoint_sha256") != expectations.logical_checkpoint_sha256
        or receipt.get("compact_replay_sha256") != expectations.compact_replay_sha256
        or receipt.get("role_manifests_sha256") != expectations.role_manifests_sha256
        or receipt.get("stage_source_seal_sha256")
        != expectations.stage_source_seal_sha256
        or receipt.get("exact_source_seal_match") is not True
        or receipt.get("peak_live_role_payload_count") != 1
        or receipt.get("fresh_network_provenance_claimed") is not False
        or getattr(stage_output, "stage_source_seal_sha256", None)
        != expectations.stage_source_seal_sha256
    ):
        raise BridgeViolation("compact_replay_rejected")
    public = _json_copy(public_authority or {})
    if public and (
        public.get("source_authoritative") is not True
        or public.get("checkpoint_file_sha256") != expectations.checkpoint_file_sha256
        or public.get("checkpoint_sha256") != expectations.logical_checkpoint_sha256
        or public.get("stage_source_seal_sha256")
        != expectations.stage_source_seal_sha256
        or public.get("role_manifests_sha256") != expectations.role_manifests_sha256
        or public.get("source_replay_receipt") != receipt
    ):
        raise BridgeViolation("public_source_authority_invalid")
    return AuthenticatedV38Authority._create(
        checkpoint=checkpoint,
        role_manifests=manifests,
        stage_output=stage_output,
        compact_replay_receipt=receipt,
        public_authority=public,
        expectations=expectations,
        blob_loader=blob_loader,
        source_api=source_api,
        inventory=inventory,
        inventory_builder=inventory_builder,
    )


def authenticate_v38_private_root(
    repo_root: Path,
    private_root: Path,
    *,
    readable_contact: str | None = None,
) -> AuthenticatedV38Authority:
    """Authenticate the exact preserved v3.8 store without mutating it.

    No lock is acquired and no directory or file is created.  The readable SEC
    contact is required solely to reproduce the already-sealed private
    fingerprint and to scan every serialized byte for an echo.
    """

    try:
        if not isinstance(repo_root, Path) or not isinstance(private_root, Path):
            raise BridgeViolation("private_authority_root_invalid")
        repo = repo_root.resolve(strict=True)
        private = private_root.resolve(strict=True)
        _secure_directory(repo)
        _secure_directory(private)
        if private != repo / "data" / _PRIVATE_NAMESPACE:
            raise BridgeViolation("private_authority_root_invalid")
        if readable_contact is None:
            raise BridgeViolation("private_contact_required")
        try:
            contact = PrivateSecContact(readable_contact)
        except Exception:
            raise BridgeViolation("private_contact_invalid") from None
        expectations = _source_expectations()
        terminal, public = _authenticate_public_source(repo)
        private_stage = private / _STAGE
        public_stage = repo / "e" / _PRIVATE_NAMESPACE / _STAGE
        _secure_directory(private_stage)
        _secure_directory(public_stage)

        before = _snapshot_inventory(
            private_stage=private_stage,
            public_stage=public_stage,
            contact=contact,
        )

        def inventory_builder() -> tuple[_FileSnapshot, ...]:
            # The returned authority must not retain the readable contact or
            # any reversible echo needle.  The initial pass already scanned
            # every byte; subsequent passes require exact hash/stat equality.
            return _snapshot_inventory(
                private_stage=private_stage,
                public_stage=public_stage,
                contact=None,
            )
        if (
            expectations.inventory_file_count is None
            or expectations.inventory_byte_count is None
            or expectations.inventory_sha256 is None
            or len(before) != expectations.inventory_file_count
            or sum(item.byte_count for item in before)
            != expectations.inventory_byte_count
            or _inventory_digest(before) != expectations.inventory_sha256
        ):
            raise BridgeViolation("private_inventory_invalid")
        try:
            state = validate_detached_journal(
                private_stage / "journal",
                _STAGE,
                contact.fingerprint_sha256,
            )
        except SecGemmaLeanV38JournalError:
            raise BridgeViolation("journal_replay_invalid") from None
        role_counts = getattr(_contract_module(), "V38_ROLE_COUNTS", None)
        expected_counts = {"I": 97, "U": 75, "D": 75}
        if isinstance(role_counts, (Mapping, tuple, list)):
            try:
                expected_counts = dict(role_counts)
            except Exception:
                raise BridgeViolation("contract_source_pins_invalid") from None
        if (
            state.passed is not True
            or state.accounting_complete is not True
            or state.terminal_status != "passed"
            or state.role_plan_sha256 != expectations.role_plan_sha256
            or state.formula_requests != 199
            or state.role_seals != 199
            or state.lifetime_intents != 199
            or state.http_200_responses != 199
            or state.historical_count != 1
            or state.quarterly_master_count != 100
            or state.complete_submission_count != expected_counts["I"]
            or state.open_role_id is not None
            or state.active_invocation is not None
        ):
            raise BridgeViolation("journal_replay_invalid")
        manifests, blob_paths = _authenticate_role_files(
            private_stage=private_stage,
            state=state,
            contact=contact,
        )
        _authenticate_phase_receipts(
            private_stage=private_stage,
            state=state,
            manifests=manifests,
            contact=contact,
        )
        checkpoints = [
            path
            for path in private_stage.iterdir()
            if _CHECKPOINT_RE.fullmatch(path.name)
        ]
        if len(checkpoints) != 1:
            raise BridgeViolation("compact_checkpoint_invalid")
        checkpoint_path = checkpoints[0]
        checkpoint, checkpoint_hash, _, _ = _read_canonical_mapping_with_identity(
            checkpoint_path, contact=contact
        )
        match = _CHECKPOINT_RE.fullmatch(checkpoint_path.name)
        if match is None or match.group("sha") != checkpoint_hash:
            raise BridgeViolation("compact_checkpoint_invalid")

        def load_blob(name: str) -> bytes:
            if type(name) is not str or name not in blob_paths:
                raise BridgeViolation("blob_identity_invalid")
            path = blob_paths[name]
            expected = manifests[int(name[:6])]
            return _read_authenticated_blob(
                path,
                expected_sha256=expected["body_sha256"],
                expected_size=expected["body_bytes"],
                contact=contact,
            )

        authority = _authenticate_compact_replay(
            checkpoint_file_sha256=checkpoint_hash,
            checkpoint=checkpoint,
            role_manifests=manifests,
            blob_loader=load_blob,
            source_api=_v38_source,
            expectations=expectations,
            public_authority=public,
            inventory=before,
            inventory_builder=inventory_builder,
        )
        seal = authority.stage_output.seal
        if (
            seal.get("counts", {}).get("I") != expected_counts["I"]
            or seal.get("counts", {}).get("U") != expected_counts["U"]
            or seal.get("counts", {}).get("D") != expected_counts["D"]
            or seal.get("role_plan_sha256") != expectations.role_plan_sha256
        ):
            # Some v3.8 seals expose counts at the root rather than under one
            # summary member.  Require the exact public terminal counts below
            # as a second, independent binding in either representation.
            terminal_counts = terminal.get("stage_seal_evidence", {}).get("counts", {})
            if any(terminal_counts.get(name) != value for name, value in expected_counts.items()):
                raise BridgeViolation("source_role_counts_invalid")
        after = inventory_builder()
        if tuple(item.identity() for item in after) != tuple(
            item.identity() for item in before
        ):
            raise BridgeViolation("private_authority_changed_during_read")
        return authority
    except BridgeViolation:
        raise
    except Exception:
        raise BridgeViolation("private_authority_rejected") from None


def _typed_json_identity(value: Any) -> Any:
    if value is None:
        return {"type": "null", "value": None}
    if type(value) is bool:
        return {"type": "boolean", "value": value}
    if type(value) is int:
        return {"type": "integer", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise BridgeViolation("typed_row_invalid")
        return {"type": "float", "value_hex": value.hex()}
    if type(value) is str:
        return {"type": "string", "value": value}
    if type(value) is list:
        return {"type": "array", "value": [_typed_json_identity(item) for item in value]}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise BridgeViolation("typed_row_invalid")
        return {
            "type": "object",
            "value": {key: _typed_json_identity(value[key]) for key in sorted(value)},
        }
    raise BridgeViolation("typed_row_invalid")


def _stage_output_sha256(output: Any) -> str:
    try:
        return _canonical_sha(
            {
                "seal": output.seal,
                "submissions_evidence": getattr(
                    output, "submissions_evidence", None
                ),
                "reconciliation_evidence": output.reconciliation_evidence,
                "complete_evidence": list(output.complete_evidence),
                "prior_present": getattr(output, "prior", None) is not None,
            }
        )
    except Exception:
        raise BridgeViolation("authority_object_invalid") from None


def _official_primary_url(accession: str, filename: str) -> str:
    if (
        _ACCESSION_RE.fullmatch(accession) is None
        or _SAFE_NAME_RE.fullmatch(filename) is None
    ):
        raise BridgeViolation("legacy_record_invalid")
    return (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        f"{accession.replace('-', '')}/"
        f"{quote(filename, safe='-._~', encoding='utf-8', errors='strict')}"
    )


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        contract = _contract_module()
        prefix = row["frozen_prefix"]
        semantic = prefix["submissions_semantic_identity"]
        typed = semantic["typed_values"]
        layout = row["submissions_layout_provenance"]
        acceptance = prefix["acceptance"]["normalized_et"]
        change = prefix["filing_date_change"]["normalized"]
        selection = row["primary_selection"]
        source_name = layout["source_name"]
        source_url = (
            _v38_source.MAIN_SUBMISSIONS_URL
            if source_name == _v38_source.MAIN_SUBMISSIONS_NAME
            else f"https://data.sec.gov/submissions/{source_name}"
        )
        typed_hash = _v316_sha(_typed_json_identity(typed))
        raw_primary = typed["primaryDocument"]
        submissions_filename = raw_primary or None
        sgml_filename = selection["sgml_filename"]
        sec_filename = submissions_filename or sgml_filename
        selected_identity = sec_filename or contract.LEGACY_MISSING_DOCUMENT_IDENTITY
        if sec_filename is None:
            primary_url = None
        else:
            primary_url = _official_primary_url(row["accession_number"], sec_filename)
        try:
            source_content_sha256 = contract.strip_exact_sha256_tag(
                layout["source_content_sha256"]
            )
        except Exception:
            raise BridgeViolation("source_record_invalid") from None
        source_body = {
            "schema_version": contract.PRIVATE_SOURCE_IDENTITY_SCHEMA_VERSION,
            "source_url": source_url,
            "source_content_sha256": source_content_sha256,
            "raw_row_identity_sha256": typed_hash,
            "accession_number": row["accession_number"],
            "subject_cik": "0000320193",
            "form": row["form"],
            "acceptance_datetime_source": typed["acceptanceDateTime"],
            "acceptance_datetime": acceptance,
            "filing_date": row["filing_date"],
            "filing_date_change": change,
            "submissions_primary_document_raw": raw_primary,
            "submissions_filename": submissions_filename,
            "sgml_filename": sgml_filename,
            "sec_filename": sec_filename,
            "selected_document_identity": selected_identity,
            "complete_response_sha256": row["complete_response"]["sha256"],
        }
        if set(source_body) != set(contract.PRIVATE_SOURCE_IDENTITY_FIELDS):
            raise BridgeViolation("source_record_invalid")
        compatibility_record = {
            "schema_version": contract.NULLABLE_SOURCE_RECORD_SCHEMA_VERSION,
            "accession_number": row["accession_number"],
            "subject_cik": "0000320193",
            "form": row["form"],
            "acceptance_datetime": acceptance,
            "availability_session": row["availability_session"],
            "filing_date": row["filing_date"],
            "filing_date_change": change,
            "official_complete_submission_url": row["complete_response"]["url"],
            "selected_document_identity": selected_identity,
            "primary_document_filename": sec_filename,
            "official_primary_document_url": primary_url,
            "selected_text_start_byte": row["extracted_text"]["start_byte"],
            "selected_text_end_byte": row["extracted_text"]["end_byte"],
            "selected_text_length": row["extracted_text"]["length"],
            "primary_document_body_kind": contract.PRIMARY_DOCUMENT_BODY_KIND,
            "source_record_sha256": _v316_sha(source_body),
            "raw_primary_document_sha256": row["extracted_text"]["sha256"],
            "normalized_text_sha256": row["normalized_text"]["sha256"],
            "complete_response_sha256": row["complete_response"]["sha256"],
            "normalized_text_length": row["normalized_text"]["length"],
            "complete_response_length": row["complete_response"]["length"],
        }
        if (
            semantic["accession_number"] != row["accession_number"]
            or semantic["subject_cik"] != "0000320193"
            or typed["accessionNumber"] != row["accession_number"]
            or typed["form"] != row["form"]
            or typed["filingDate"] != row["filing_date"]
            or type(raw_primary) is not str
            or selection["submissions_filename"] != submissions_filename
            or selection["sgml_filename"] != sgml_filename
            or selection["sec_filename"] != sec_filename
            or selection["document_identity"] != selected_identity
            or selection["submissions_filename_missing"]
            != (submissions_filename is None)
            or selection["sgml_filename_missing"] != (sgml_filename is None)
            or (
                submissions_filename is not None
                and sgml_filename is not None
                and submissions_filename != sgml_filename
            )
            or type(acceptance) is not str
            or re.fullmatch(r"[0-9]{14}", acceptance) is None
            or _ACCESSION_RE.fullmatch(compatibility_record["accession_number"])
            is None
            or compatibility_record["form"] not in {"10-K", "10-Q"}
            or (
                sec_filename is not None
                and _SAFE_NAME_RE.fullmatch(sec_filename) is None
            )
        ):
            raise BridgeViolation("source_record_invalid")
        try:
            validated = contract.validate_compatibility_record(compatibility_record)
        except Exception:
            raise BridgeViolation("compatibility_record_invalid") from None
        return {
            "typed_row_sha256": typed_hash,
            "source_identity": source_body,
            "source_record_sha256": compatibility_record["source_record_sha256"],
            "compatibility_record": validated,
            "compatibility_record_sha256": _v316_sha(validated),
        }
    except BridgeViolation:
        raise
    except Exception:
        raise BridgeViolation("compatibility_record_invalid") from None


def _load_one_document(
    *,
    row: Mapping[str, Any],
    target: Mapping[str, Any],
    expected_complete_evidence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    loader: Callable[[str], bytes],
    source_api: _SourceApi,
    reconciliation_evidence: Mapping[str, Any],
) -> tuple[LegacyScienceDocument, dict[str, Any]]:
    try:
        payload = loader(manifest["blob_name"])
    except BridgeViolation:
        raise
    except Exception:
        raise BridgeViolation("blob_loader_rejected") from None
    if type(payload) is not bytes:
        raise BridgeViolation("blob_loader_rejected")
    if (
        len(payload) != manifest["body_bytes"]
        or _file_sha(payload) != _bare_sha(manifest["body_sha256"])
        or len(payload) != row["complete_response"]["length"]
        or _file_sha(payload) != _bare_sha(row["complete_response"]["sha256"])
    ):
        raise BridgeViolation("blob_identity_invalid")
    try:
        reparsed = source_api.parse_complete(
            _STAGE, target, payload, reconciliation_evidence
        )
        evidence = _json_copy(reparsed.evidence)
    except Exception:
        raise BridgeViolation("complete_submission_reparse_rejected") from None
    if (
        evidence != expected_complete_evidence
        or evidence.get("source_evidence_sha256")
        != manifest["source_evidence_sha256"]
        or evidence.get("seal_row") != row
    ):
        raise BridgeViolation("complete_submission_parity_invalid")
    try:
        extracted_meta = row["extracted_text"]
        normalized_meta = row["normalized_text"]
        start = extracted_meta["start_byte"]
        end = extracted_meta["end_byte"]
        raw = bytes(payload[start:end])
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= len(payload)
            or len(raw) != extracted_meta["length"]
            or _file_sha(raw) != _bare_sha(extracted_meta["sha256"])
        ):
            raise BridgeViolation("selected_text_identity_invalid")
        normalized_result = normalize_filing_text(raw.decode("latin-1"))
        normalized = normalized_result.text.encode("utf-8")
        if (
            not normalized_result.usable
            or len(normalized) != normalized_meta["length"]
            or normalized_result.character_count != normalized_meta["character_count"]
            or _file_sha(normalized) != _bare_sha(normalized_meta["sha256"])
        ):
            raise BridgeViolation("normalized_text_identity_invalid")
        document = LegacyScienceDocument(
            accession_number=row["accession_number"],
            official_complete_submission_url=row["complete_response"]["url"],
            raw_primary_document=raw,
            normalized_text=normalized,
            # Preserve the existing v3.8 LegacyScienceDocument spelling
            # exactly.  Those three identities are tagged ``sha256:...``;
            # bare hashes are used only for file comparisons above.
            primary_document_sha256=extracted_meta["sha256"],
            normalized_text_sha256=normalized_meta["sha256"],
            complete_response_sha256=row["complete_response"]["sha256"],
            selected_document_identity=row["primary_selection"]["document_identity"],
        )
        identity = _row_identity(row)
        return document, identity
    finally:
        # The full complete-submission response must die before the caller can
        # invoke the loader for the next row.  Only the selected TEXT and its
        # normalized bytes escape this function.
        del payload


def _verify_authority_object(authority: AuthenticatedV38Authority) -> None:
    if type(authority) is not AuthenticatedV38Authority:
        raise BridgeViolation("authority_object_invalid")
    checkpoint_body = dict(authority.checkpoint)
    checkpoint_supplied = checkpoint_body.pop("checkpoint_sha256", None)
    receipt_body = dict(authority.compact_replay_receipt)
    receipt_supplied = receipt_body.pop("compact_replay_sha256", None)
    if (
        authority._token is not _AUTHORITY_TOKEN
        or checkpoint_supplied != authority._expectations.logical_checkpoint_sha256
        or _canonical_sha(checkpoint_body) != checkpoint_supplied
        or receipt_supplied != authority._expectations.compact_replay_sha256
        or _canonical_sha(receipt_body) != receipt_supplied
        or _canonical_sha(list(authority.role_manifests))
        != authority._expectations.role_manifests_sha256
        or getattr(authority.stage_output, "stage_source_seal_sha256", None)
        != authority._expectations.stage_source_seal_sha256
        or _stage_output_sha256(authority.stage_output)
        != authority._stage_output_sha256
    ):
        raise BridgeViolation("authority_object_invalid")
    if authority._inventory is not None:
        if authority._inventory_builder is None:
            raise BridgeViolation("authority_object_invalid")
        current = authority._inventory_builder()
        if tuple(item.identity() for item in current) != tuple(
            item.identity() for item in authority._inventory
        ):
            raise BridgeViolation("private_authority_changed_during_read")


def build_streaming_science_projection(
    authority: AuthenticatedV38Authority,
    *,
    blob_loader: Callable[[str], bytes] | None = None,
) -> ScienceBridgeProjection:
    """Stream exact v3.8 D blobs into the frozen legacy development shape."""

    try:
        _verify_authority_object(authority)
        loader = authority._blob_loader if blob_loader is None else blob_loader
        if not callable(loader):
            raise BridgeViolation("blob_loader_invalid")
        output = authority.stage_output
        seal = output.seal
        if seal.get("stage") != _STAGE:
            raise BridgeViolation("stage_scope_invalid")
        target_rows = seal.get("target_rows")
        d_accessions = seal.get("d_accessions")
        if type(target_rows) is not list or type(d_accessions) is not list:
            raise BridgeViolation("development_membership_invalid")
        if (
            len(d_accessions) != authority._expectations.document_count
            or len(d_accessions) != len(set(d_accessions))
            or any(type(item) is not str for item in d_accessions)
        ):
            raise BridgeViolation("development_membership_invalid")
        by_accession: dict[str, Mapping[str, Any]] = {}
        for row in target_rows:
            if type(row) is not dict or type(row.get("accession_number")) is not str:
                raise BridgeViolation("development_membership_invalid")
            accession = row["accession_number"]
            if accession in by_accession:
                raise BridgeViolation("development_membership_invalid")
            by_accession[accession] = row
        if not set(d_accessions) <= set(by_accession):
            raise BridgeViolation("development_membership_invalid")
        rows = sorted(
            (by_accession[accession] for accession in d_accessions),
            key=lambda row: (row["availability_session"], row["accession_number"]),
        )
        if [row["accession_number"] for row in rows] != sorted(
            d_accessions,
            key=lambda accession: (
                by_accession[accession]["availability_session"], accession
            ),
        ):
            raise BridgeViolation("source_order_invalid")
        complete_manifests: dict[str, Mapping[str, Any]] = {}
        for manifest in authority.role_manifests:
            role_id = manifest["role_id"]
            if role_id.startswith("complete/"):
                accession = role_id.removeprefix("complete/")
                if accession in complete_manifests:
                    raise BridgeViolation("complete_manifest_membership_invalid")
                complete_manifests[accession] = manifest
        expected_evidence: dict[str, Mapping[str, Any]] = {}
        for evidence in output.complete_evidence:
            accession = evidence.get("accession_number")
            if type(accession) is not str or accession in expected_evidence:
                raise BridgeViolation("complete_manifest_membership_invalid")
            expected_evidence[accession] = evidence
        reconciliation = output.reconciliation_evidence
        targets = reconciliation.get("complete_targets")
        if type(targets) is not list:
            raise BridgeViolation("complete_manifest_membership_invalid")
        targets_by_accession: dict[str, Mapping[str, Any]] = {}
        for target in targets:
            try:
                accession = target["submissions"]["accession_number"]
            except Exception:
                raise BridgeViolation("complete_manifest_membership_invalid") from None
            if type(accession) is not str or accession in targets_by_accession:
                raise BridgeViolation("complete_manifest_membership_invalid")
            targets_by_accession[accession] = target
        d_set = set(d_accessions)
        if (
            not d_set <= set(complete_manifests)
            or not d_set <= set(expected_evidence)
            or not d_set <= set(targets_by_accession)
        ):
            raise BridgeViolation("complete_manifest_membership_invalid")

        contract = _contract_module()
        if (
            contract.canonical_sha256(contract.build_source_authority_pins())
            != contract.SOURCE_AUTHORITY_PINS_SHA256
        ):
            raise BridgeViolation("source_authority_pins_invalid")

        documents: list[LegacyScienceDocument] = []
        compatibility_records: list[dict[str, Any]] = []
        primary_documents: list[dict[str, Any]] = []
        source_order: list[dict[str, Any]] = []
        document_by_accession: dict[str, LegacyScienceDocument] = {}
        compatibility_by_accession: dict[str, dict[str, Any]] = {}
        compatibility_hash_by_accession: dict[str, str] = {}
        for source_sequence, row in enumerate(rows):
            accession = row["accession_number"]
            document, identities = _load_one_document(
                row=row,
                target=targets_by_accession[accession],
                expected_complete_evidence=expected_evidence[accession],
                manifest=complete_manifests[accession],
                loader=loader,
                source_api=authority._source_api,
                reconciliation_evidence=reconciliation,
            )
            compatibility = dict(identities["compatibility_record"])
            compatibility_hash = identities["compatibility_record_sha256"]
            documents.append(document)
            compatibility_records.append(compatibility)
            document_by_accession[accession] = document
            compatibility_by_accession[accession] = compatibility
            compatibility_hash_by_accession[accession] = compatibility_hash
            primary_documents.append(
                {
                    "accession_number": accession,
                    "form": compatibility["form"],
                    "availability_session": compatibility["availability_session"],
                    "acceptance_datetime": compatibility["acceptance_datetime"],
                    "primary_document_filename": compatibility[
                        "primary_document_filename"
                    ],
                    "selected_document_identity": compatibility[
                        "selected_document_identity"
                    ],
                    "official_complete_submission_url": compatibility[
                        "official_complete_submission_url"
                    ],
                    "official_primary_document_url": compatibility[
                        "official_primary_document_url"
                    ],
                    "body": document.raw_primary_document,
                }
            )
            body = {
                "sequence": source_sequence,
                "accession_number": accession,
                "form": compatibility["form"],
                "filing_date": compatibility["filing_date"],
                "acceptance_datetime": compatibility["acceptance_datetime"],
                "availability_session": compatibility["availability_session"],
                "primary_document_filename": compatibility[
                    "primary_document_filename"
                ],
                "selected_document_identity": compatibility[
                    "selected_document_identity"
                ],
                "source_record_sha256": compatibility["source_record_sha256"],
                "compatibility_record_sha256": compatibility_hash,
                "complete_response_sha256": contract.strip_exact_sha256_tag(
                    compatibility["complete_response_sha256"]
                ),
                "selected_text_sha256": contract.strip_exact_sha256_tag(
                    compatibility["raw_primary_document_sha256"]
                ),
                "normalized_text_sha256": contract.strip_exact_sha256_tag(
                    compatibility["normalized_text_sha256"]
                ),
            }
            source_order.append(_v316_self_hash(body, "source_order_row_sha256"))

        missing_bitmap = [
            item["primary_document_filename"] is None
            and item["official_primary_document_url"] is None
            for item in compatibility_records
        ]
        missing_count = sum(missing_bitmap)
        present_count = len(missing_bitmap) - missing_count
        if (
            len(compatibility_records) != contract.DEVELOPMENT_DOCUMENT_COUNT
            or present_count != contract.DEVELOPMENT_FILENAME_PRESENT_COUNT
            or missing_count != contract.DEVELOPMENT_FILENAME_MISSING_COUNT
        ):
            raise BridgeViolation("nullable_filename_count_invalid")

        prior_by_form: dict[str, str] = {}
        prior_links: list[dict[str, Any]] = []
        for item in source_order:
            prior = prior_by_form.get(item["form"])
            prior_document = None if prior is None else document_by_accession[prior]
            body = {
                "accession_number": item["accession_number"],
                "form": item["form"],
                "prior_accession_number": prior,
                "current_normalized_text_sha256": item["normalized_text_sha256"],
                "prior_normalized_text_sha256": (
                    None
                    if prior_document is None
                    else contract.strip_exact_sha256_tag(
                        prior_document.normalized_text_sha256
                    )
                ),
                "policy": contract.PRIOR_SELECTION_POLICY,
            }
            prior_links.append(_v316_self_hash(body, "prior_link_sha256"))
            prior_by_form[item["form"]] = item["accession_number"]
        first_by_form: dict[str, Mapping[str, Any]] = {}
        for link in prior_links:
            first_by_form.setdefault(link["form"], link)
        if set(first_by_form) != {"10-K", "10-Q"} or any(
            item["prior_accession_number"] is not None
            for item in first_by_form.values()
        ):
            raise BridgeViolation("prior_link_invalid")

        event_rows = sorted(
            source_order,
            key=lambda item: (
                item["availability_session"],
                item["acceptance_datetime"],
                item["accession_number"],
            ),
        )
        event_order: list[dict[str, Any]] = []
        for event_sequence, item in enumerate(event_rows):
            body = {
                "sequence": event_sequence,
                "source_sequence": item["sequence"],
                "accession_number": item["accession_number"],
                "form": item["form"],
                "availability_session": item["availability_session"],
                "acceptance_datetime": item["acceptance_datetime"],
                "compatibility_record_sha256": item[
                    "compatibility_record_sha256"
                ],
            }
            event_order.append(_v316_self_hash(body, "event_order_row_sha256"))

        legacy_rows = [
            {
                "accession_number": document.accession_number,
                "official_complete_submission_url": document.official_complete_submission_url,
                "raw_primary_document_sha256": document.primary_document_sha256,
                "normalized_text_sha256": document.normalized_text_sha256,
                "complete_response_sha256": document.complete_response_sha256,
                "selected_document_identity": document.selected_document_identity,
                "raw_primary_document_semantics": "selected_embedded_TEXT_bytes",
            }
            for document in documents
        ]
        legacy_body = {
            "schema_version": _v38_source.LEGACY_SCIENCE_PROJECTION_SCHEMA_VERSION,
            "stage": _STAGE,
            "stage_source_seal_sha256": authority._expectations.stage_source_seal_sha256,
            "document_count": len(documents),
            "documents": legacy_rows,
            "ordering": "availability_session_then_accession",
            "compatibility_boundary": (
                "raw_primary_document is extracted TEXT, never the complete response"
            ),
        }
        legacy_manifest = {
            **legacy_body,
            "projection_sha256": _canonical_sha(legacy_body),
        }
        expected_legacy_rows = [
            {
                "accession_number": row["accession_number"],
                "official_complete_submission_url": row["complete_response"]["url"],
                "raw_primary_document_sha256": row["extracted_text"]["sha256"],
                "normalized_text_sha256": row["normalized_text"]["sha256"],
                "complete_response_sha256": row["complete_response"]["sha256"],
                "selected_document_identity": row["primary_selection"]["document_identity"],
                "raw_primary_document_semantics": "selected_embedded_TEXT_bytes",
            }
            for row in rows
        ]
        if legacy_rows != expected_legacy_rows:
            raise BridgeViolation("legacy_projection_parity_invalid")

        compatibility_body = {
            "schema_version": contract.COMPATIBILITY_MANIFEST_SCHEMA_VERSION,
            "source_authority_pins_sha256": contract.SOURCE_AUTHORITY_PINS_SHA256,
            "science_contract_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
            "legacy_source_projection_sha256": legacy_manifest["projection_sha256"],
            "document_count": len(compatibility_records),
            "filename_present_count": present_count,
            "filename_missing_count": missing_count,
            "ordered_compatibility_record_sha256s": [
                compatibility_hash_by_accession[item["accession_number"]]
                for item in source_order
            ],
            "nullable_filename_bitmap_sha256": _v316_sha(missing_bitmap),
            "source_order_sha256": _v316_sha(source_order),
            "event_order_sha256": _v316_sha(event_order),
            "prior_links_sha256": _v316_sha(prior_links),
        }
        compatibility_manifest = _v316_self_hash(
            compatibility_body, "compatibility_manifest_sha256"
        )
        try:
            contract.validate_compatibility_manifest(compatibility_manifest)
        except Exception:
            raise BridgeViolation("compatibility_manifest_invalid") from None

        universe_records: list[dict[str, Any]] = []
        for compatibility in compatibility_records:
            accession = compatibility["accession_number"]
            universe_records.append(
                {
                    "accession_number": accession,
                    "subject_cik": compatibility["subject_cik"],
                    "form": compatibility["form"],
                    "acceptance_datetime": compatibility["acceptance_datetime"],
                    "filing_date": compatibility["filing_date"],
                    "filing_date_change": compatibility["filing_date_change"],
                    "availability_session": compatibility["availability_session"],
                    "artifact_stage": _STAGE,
                    "primary_document_filename": compatibility[
                        "primary_document_filename"
                    ],
                    "selected_document_identity": compatibility[
                        "selected_document_identity"
                    ],
                    "official_complete_submission_url": compatibility[
                        "official_complete_submission_url"
                    ],
                    "official_primary_document_url": compatibility[
                        "official_primary_document_url"
                    ],
                    "source_record_sha256": compatibility["source_record_sha256"],
                    "selected_text_sha256": contract.strip_exact_sha256_tag(
                        compatibility["raw_primary_document_sha256"]
                    ),
                    "normalized_text_sha256": contract.strip_exact_sha256_tag(
                        compatibility["normalized_text_sha256"]
                    ),
                    "complete_response_sha256": contract.strip_exact_sha256_tag(
                        compatibility["complete_response_sha256"]
                    ),
                    "compatibility_record_sha256": compatibility_hash_by_accession[
                        accession
                    ],
                }
            )
        universe_semantic = {
            "schema_version": contract.NULLABLE_UNIVERSE_SEMANTIC_SCHEMA_VERSION,
            "calendar_sessions_sha256": contract.CALENDAR_SESSIONS_SHA256,
            "records": [
                {
                    "accession_number": item["accession_number"],
                    "form": item["form"],
                    "availability_session": item["availability_session"],
                    "artifact_stage": item["artifact_stage"],
                }
                for item in universe_records
            ],
        }
        universe_body = {
            "schema_version": contract.NULLABLE_UNIVERSE_SCHEMA_VERSION,
            "source_authority_pins_sha256": contract.SOURCE_AUTHORITY_PINS_SHA256,
            "science_contract_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
            "legacy_source_projection_sha256": legacy_manifest["projection_sha256"],
            "compatibility_manifest_sha256": compatibility_manifest[
                "compatibility_manifest_sha256"
            ],
            "calendar_sessions_sha256": contract.CALENDAR_SESSIONS_SHA256,
            "document_count": len(universe_records),
            "exact_acceptance_timestamp_count": sum(
                type(item["acceptance_datetime"]) is str
                and re.fullmatch(r"[0-9]{14}", item["acceptance_datetime"])
                is not None
                for item in universe_records
            ),
            "stage_counts": {_STAGE: len(universe_records)},
            "records": universe_records,
            "universe_semantic_sha256": _v316_sha(universe_semantic),
        }
        universe = _v316_self_hash(universe_body, "universe_sha256")
        try:
            contract.validate_nullable_universe_manifest(universe)
        except Exception:
            raise BridgeViolation("nullable_universe_invalid") from None

        content_records: list[dict[str, Any]] = []
        for compatibility in compatibility_records:
            accession = compatibility["accession_number"]
            document = document_by_accession[accession]
            content_records.append(
                {
                    "accession_number": accession,
                    "form": compatibility["form"],
                    "availability_session": compatibility["availability_session"],
                    "primary_document_filename": compatibility[
                        "primary_document_filename"
                    ],
                    "selected_document_identity": compatibility[
                        "selected_document_identity"
                    ],
                    "official_complete_submission_url": compatibility[
                        "official_complete_submission_url"
                    ],
                    "official_primary_document_url": compatibility[
                        "official_primary_document_url"
                    ],
                    "compatibility_record_sha256": compatibility_hash_by_accession[
                        accession
                    ],
                    "selected_text_sha256": contract.strip_exact_sha256_tag(
                        compatibility["raw_primary_document_sha256"]
                    ),
                    "normalized_text_sha256": contract.strip_exact_sha256_tag(
                        compatibility["normalized_text_sha256"]
                    ),
                    "complete_response_sha256": contract.strip_exact_sha256_tag(
                        compatibility["complete_response_sha256"]
                    ),
                    "selected_text_bytes": len(document.raw_primary_document),
                    "normalized_text_bytes": len(document.normalized_text),
                }
            )
        content_body = {
            "schema_version": contract.NULLABLE_CONTENT_SCHEMA_VERSION,
            "artifact_stage": _STAGE,
            "universe_sha256": universe["universe_sha256"],
            "compatibility_manifest_sha256": compatibility_manifest[
                "compatibility_manifest_sha256"
            ],
            "document_count": len(content_records),
            "documents": content_records,
        }
        content_manifest = _v316_self_hash(
            content_body, "content_manifest_sha256"
        )
        try:
            contract.validate_nullable_content_manifest(content_manifest)
        except Exception:
            raise BridgeViolation("nullable_content_invalid") from None

        universe_by_accession = {
            item["accession_number"]: item for item in universe_records
        }
        content_by_accession = {
            item["accession_number"]: item for item in content_records
        }
        prior_by_accession = {
            item["accession_number"]: item["prior_accession_number"]
            for item in prior_links
        }
        universe_event_proofs: list[dict[str, Any]] = []
        for event in event_order:
            accession = event["accession_number"]
            current = universe_by_accession[accession]
            current_content = content_by_accession[accession]
            prior_accession = prior_by_accession[accession]
            prior_record = (
                None
                if prior_accession is None
                else universe_by_accession[prior_accession]
            )
            prior_content = (
                None
                if prior_accession is None
                else content_by_accession[prior_accession]
            )
            proof_body = {
                "schema_version": (
                    contract.NULLABLE_UNIVERSE_EVENT_PROOF_SCHEMA_VERSION
                ),
                "compatibility_manifest_sha256": compatibility_manifest[
                    "compatibility_manifest_sha256"
                ],
                "universe_sha256": universe["universe_sha256"],
                "calendar_sessions_sha256": contract.CALENDAR_SESSIONS_SHA256,
                "current_record": current,
                "current_record_sha256": _v316_sha(current),
                "current_content_record": current_content,
                "current_content_record_sha256": _v316_sha(current_content),
                "current_filing_sha256": current_content[
                    "normalized_text_sha256"
                ],
                "prior_same_form_record": prior_record,
                "prior_same_form_record_sha256": (
                    None if prior_record is None else _v316_sha(prior_record)
                ),
                "prior_same_form_content_record": prior_content,
                "prior_same_form_content_record_sha256": (
                    None if prior_content is None else _v316_sha(prior_content)
                ),
                "prior_same_form_filing_sha256": (
                    None
                    if prior_content is None
                    else prior_content["normalized_text_sha256"]
                ),
                "content_manifest_sha256": content_manifest[
                    "content_manifest_sha256"
                ],
                "prior_selection": contract.PRIOR_SELECTION_POLICY,
            }
            proof = _v316_self_hash(
                proof_body, "universe_event_proof_sha256"
            )
            try:
                contract.validate_nullable_universe_event_proof(
                    proof,
                    universe=universe,
                    content_manifest=content_manifest,
                    compatibility_records=compatibility_records,
                )
            except Exception:
                raise BridgeViolation("nullable_universe_proof_invalid") from None
            universe_event_proofs.append(proof)

        try:
            scientific = contract.build_frozen_science_projection()
            verified = contract.verify_frozen_science_projection(scientific)
            if verified != scientific:
                raise BridgeViolation("science_projection_invalid")
        except BridgeViolation:
            raise
        except Exception:
            raise BridgeViolation("science_projection_invalid") from None
        manifest_body = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "stage": _STAGE,
            "document_count": len(documents),
            "filename_present_count": present_count,
            "filename_missing_count": missing_count,
            "source_authority_pins_sha256": contract.SOURCE_AUTHORITY_PINS_SHA256,
            "source_authority_base_commit": contract.V38_SOURCE_COMMIT,
            "source_authority_base_tree": contract.V38_SOURCE_TREE,
            "source_inventory_sha256": contract.V38_INVENTORY_SHA256,
            "stage_source_seal_sha256": authority._expectations.stage_source_seal_sha256,
            "checkpoint_sha256": authority._expectations.logical_checkpoint_sha256,
            "compact_replay_sha256": authority._expectations.compact_replay_sha256,
            "role_manifests_sha256": authority._expectations.role_manifests_sha256,
            "role_plan_sha256": authority._expectations.role_plan_sha256,
            "science_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
            "legacy_projection_sha256": legacy_manifest["projection_sha256"],
            "compatibility_manifest_sha256": compatibility_manifest[
                "compatibility_manifest_sha256"
            ],
            "universe_sha256": universe["universe_sha256"],
            "content_manifest_sha256": content_manifest[
                "content_manifest_sha256"
            ],
            "calendar_sessions_sha256": contract.CALENDAR_SESSIONS_SHA256,
            "universe_event_proofs_sha256": _v316_sha(universe_event_proofs),
            "documents_sha256": _canonical_sha(legacy_rows),
            "records_sha256": _v316_sha(universe_records),
            "source_order_sha256": compatibility_manifest["source_order_sha256"],
            "event_order_sha256": compatibility_manifest["event_order_sha256"],
            "prior_links_sha256": compatibility_manifest["prior_links_sha256"],
            "primary_documents_sha256": _v316_sha(
                [
                    {
                        key: (_file_sha(value) if key == "body" else value)
                        for key, value in item.items()
                    }
                    for item in primary_documents
                ]
            ),
            "set_parity": True,
            "source_sequence_parity": True,
            "legacy_projection_parity": True,
            "typed_identity_parity": True,
            "nullable_filename_parity": True,
            "no_fabricated_primary_url": True,
            "prior_links_are_internal_only": True,
            "first_10k_and_10q_have_no_prior": True,
            "peak_live_complete_submission_blob_count": 1,
            "sec_request_count": 0,
            "confirmation_or_final_opened": False,
            "contains_private_rows": False,
            "contains_accessions_urls_filenames_or_bodies": False,
        }
        manifest = _v316_self_hash(manifest_body, "bridge_sha256")
        manifest_json = contract.canonical_json_bytes(manifest)
        if authority._inventory is not None:
            _verify_authority_object(authority)
        return ScienceBridgeProjection(
            stage=_STAGE,
            legacy_documents=tuple(documents),
            compatibility_records=tuple(
                MappingProxyType(dict(item)) for item in compatibility_records
            ),
            compatibility_manifest=MappingProxyType(compatibility_manifest),
            records=tuple(
                MappingProxyType(dict(item)) for item in universe_records
            ),
            primary_documents=tuple(
                MappingProxyType(dict(item)) for item in primary_documents
            ),
            universe=MappingProxyType(universe),
            content_manifest=MappingProxyType(content_manifest),
            universe_event_proofs=tuple(
                MappingProxyType(dict(item)) for item in universe_event_proofs
            ),
            source_order=tuple(MappingProxyType(dict(item)) for item in source_order),
            event_order=tuple(MappingProxyType(dict(item)) for item in event_order),
            prior_links=tuple(MappingProxyType(dict(item)) for item in prior_links),
            legacy_projection_manifest=MappingProxyType(legacy_manifest),
            manifest=MappingProxyType(manifest),
            manifest_json=manifest_json,
        )
    except BridgeViolation:
        raise
    except Exception:
        raise BridgeViolation("streaming_projection_rejected") from None


__all__ = [
    "AuthenticatedV38Authority",
    "BRIDGE_SCHEMA_VERSION",
    "BridgeViolation",
    "ScienceBridgeProjection",
    "authenticate_v38_private_root",
    "build_streaming_science_projection",
]
