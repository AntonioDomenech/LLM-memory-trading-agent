"""Source-role identity audits for the frozen SEC/Gemma experiment.

The candidate manifest binds one SHA-256 value for each conceptual source role,
but a digest alone does not prove which repository path supplied the bytes.  This
module fixes the unambiguous role-to-path assignments.  The detached audit
hashes caller-supplied evidence without granting authority.  The runtime audit
instead reads the canonical files underneath the checkout that actually loaded
the participating modules, rejects path aliases and reparse points, and binds
the resulting role/path/hash tree to the candidate.  That path/file audit does
not prove that the current source bytes created every already-loaded code
object, so runtime completeness remains false pending an owned startup/import
attestation.

The trading-ledger role does not yet have one frozen source owner.  It remains
explicitly unresolved instead of being aliased to a convenient existing file.
Consequently the current receipt is always non-authorizing and incomplete; the
strict completeness validator raises until that source owner is implemented
and frozen in this mapping.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
import hashlib
import hmac
import math
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    REQUIRED_SOURCE_HASHES,
    SecFilingGemmaContractError,
    canonical_sha256,
    validate_candidate_manifest,
)


SOURCE_IDENTITY_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-source-identity-audit-v8"
)
SOURCE_TREE_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-source-tree-v1"
MAX_RUNTIME_SOURCE_BYTES_PER_FILE: Final[int] = 8 * 1024 * 1024
MAX_RUNTIME_SOURCE_BYTES_TOTAL: Final[int] = 64 * 1024 * 1024

# A ``None`` value is an intentional fail-closed declaration that the
# conceptual role has no single frozen repository source owner yet.  The
# prompt and extractor schema now have distinct frozen source owners.  The
# trading ledger is still split across prediction/scoring/proof components.
# Fixed-provider development market acquisition now has one frozen source
# owner; reveal-store ownership of its effects remains a separate requirement.
# The bounded owned SEC stage runner has a distinct source owner, while its
# market/model/artifact coverage remains intentionally incomplete and
# non-authorizing.
_CANONICAL_SOURCE_ROLE_PATH_ITEMS: Final[tuple[tuple[str, str | None], ...]] = (
    (
        "artifact_sealer",
        "agent_benchmark/sec_filing_gemma_artifact_sealer.py",
    ),
    ("calendar", "agent_benchmark/sec_session_calendar.py"),
    ("cftc_cot_policy", "agent_benchmark/cftc_cot_policy.py"),
    ("content_normalizer", "agent_benchmark/sec_filing_content.py"),
    ("contract", "agent_benchmark/sec_filing_gemma_contract.py"),
    ("deterministic_aapl", "agent_benchmark/deterministic_aapl.py"),
    ("direct_edge_features", "agent_benchmark/direct_edge_features.py"),
    ("downside_features", "agent_benchmark/downside_features.py"),
    ("extractor", "agent_benchmark/sec_filing_gemma_ollama.py"),
    (
        "extractor_prompt",
        "agent_benchmark/sec_filing_gemma_extractor_prompt.py",
    ),
    (
        "extractor_schema",
        "agent_benchmark/sec_filing_gemma_extractor_schema.py",
    ),
    ("learner", "agent_benchmark/sec_filing_gemma_learner.py"),
    ("learner_fit", "agent_benchmark/sec_filing_gemma_learner_fit.py"),
    (
        "learner_prediction",
        "agent_benchmark/sec_filing_gemma_learner_prediction.py",
    ),
    ("ledger", None),
    (
        "market_acquirer",
        "agent_benchmark/sec_filing_gemma_market_acquirer.py",
    ),
    ("market_evidence", "agent_benchmark/sec_filing_gemma_market_evidence.py"),
    ("market_features", "agent_benchmark/sec_filing_gemma_features.py"),
    (
        "market_source_bytes",
        "agent_benchmark/sec_filing_gemma_market_source_bytes.py",
    ),
    ("no_leverage", "agent_benchmark/sec_filing_gemma_no_leverage.py"),
    ("package_init", "agent_benchmark/__init__.py"),
    ("preprocessor", "agent_benchmark/sec_filing_gemma_preprocessor.py"),
    (
        "prediction_evidence",
        "agent_benchmark/sec_filing_gemma_prediction_evidence.py",
    ),
    (
        "reveal_registry",
        "agent_benchmark/sec_filing_gemma_reveal_registry.py",
    ),
    ("reveal_store", "agent_benchmark/sec_filing_gemma_reveal_store.py"),
    ("runner", "agent_benchmark/sec_filing_gemma_stage_runner.py"),
    ("scorer", "agent_benchmark/sec_filing_gemma_scoring.py"),
    ("sec_acquirer", "agent_benchmark/sec_audit_transport.py"),
    ("sec_audit_artifact", "agent_benchmark/sec_audit_artifact.py"),
    ("sec_audit_evaluation", "agent_benchmark/sec_audit_evaluation.py"),
    ("sec_audit_plan", "agent_benchmark/sec_audit_plan.py"),
    ("sec_audit_selection", "agent_benchmark/sec_audit_selection.py"),
    ("sec_audit_verifier", "agent_benchmark/sec_audit_runner.py"),
    (
        "sec_corpus_selector",
        "agent_benchmark/sec_filing_gemma_corpus.py",
    ),
    ("sec_point_in_time", "agent_benchmark/sec_point_in_time.py"),
    ("source_identity", "agent_benchmark/sec_filing_gemma_source_identity.py"),
    ("stage_access", "agent_benchmark/sec_filing_gemma_stage_access.py"),
    (
        "stage_authorization",
        "agent_benchmark/sec_filing_gemma_stage_authorization.py",
    ),
    (
        "stage_verifier",
        "agent_benchmark/sec_filing_gemma_stage_verifier.py",
    ),
    (
        "training_membership",
        "agent_benchmark/sec_filing_gemma_training_membership.py",
    ),
    ("unleveraged_aapl", "agent_benchmark/unleveraged_aapl.py"),
)

CANONICAL_SOURCE_ROLE_PATHS: Final[Mapping[str, str | None]] = MappingProxyType(
    dict(_CANONICAL_SOURCE_ROLE_PATH_ITEMS)
)
UNRESOLVED_SOURCE_ROLES: Final[tuple[str, ...]] = tuple(
    role for role, path in _CANONICAL_SOURCE_ROLE_PATH_ITEMS if path is None
)

_REPOSITORY_PATH_RE = re.compile(
    r"agent_benchmark/[a-z0-9_]+(?:/[a-z0-9_]+)*\.py\Z"
)
_NATIVE_PATH_TYPE: Final[type[Path]] = type(Path())


class SecFilingGemmaSourceIdentityError(SecFilingGemmaContractError):
    """Raised when source roles, paths, bytes, or candidate pins disagree."""


class SecFilingGemmaSourceIdentityIncompleteError(
    SecFilingGemmaSourceIdentityError
):
    """Raised when a complete identity is requested while roles are unresolved."""


def _plain_json(value: Any, location: str) -> Any:
    """Detach built-in JSON values and reject live/custom mapping views."""

    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or key in result:
                raise SecFilingGemmaSourceIdentityError(
                    f"{location} must have unique built-in string keys"
                )
            result[key] = _plain_json(child, f"{location}.{key}")
        return result
    if type(value) is list:
        return [
            _plain_json(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise SecFilingGemmaSourceIdentityError(
        f"{location} must be detached built-in finite JSON"
    )


def _exact_role_mapping(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise SecFilingGemmaSourceIdentityError(
            f"{location} must be one detached built-in role mapping"
        )
    expected = set(REQUIRED_SOURCE_HASHES)
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaSourceIdentityError(
            f"Invalid {location} roles; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _validate_repository_path(value: Any, location: str) -> str:
    if type(value) is not str or _REPOSITORY_PATH_RE.fullmatch(value) is None:
        raise SecFilingGemmaSourceIdentityError(
            f"{location} must be one canonical repository-relative Python path"
        )
    if (
        value.startswith("/")
        or "\\" in value
        or "//" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise SecFilingGemmaSourceIdentityError(
            f"{location} contains traversal or a path alias"
        )
    return value


def _frozen_role_paths() -> dict[str, str | None]:
    mapping = dict(_CANONICAL_SOURCE_ROLE_PATH_ITEMS)
    expected = set(REQUIRED_SOURCE_HASHES)
    if set(mapping) != expected or len(mapping) != len(_CANONICAL_SOURCE_ROLE_PATH_ITEMS):
        raise SecFilingGemmaSourceIdentityError(
            "Frozen source-role mapping does not cover the required roles exactly"
        )
    resolved_paths: list[str] = []
    for role in REQUIRED_SOURCE_HASHES:
        path = mapping[role]
        if path is not None:
            resolved_paths.append(
                _validate_repository_path(path, f"frozen path for {role}")
            )
    folded = [path.casefold() for path in resolved_paths]
    if len(folded) != len(set(folded)):
        raise SecFilingGemmaSourceIdentityError(
            "Frozen source roles contain a repository-path alias"
        )
    return {role: mapping[role] for role in REQUIRED_SOURCE_HASHES}


def canonical_source_tree_sha256(source_hashes: Mapping[str, str]) -> str:
    """Return the candidate binding for the exact role/path/hash tree.

    Unresolved conceptual roles remain present with ``None`` paths and their
    candidate pins.  They therefore cannot disappear from the tree merely
    because the current implementation cannot yet verify their bytes.
    """

    supplied = _exact_role_mapping(source_hashes, "source_hashes")
    normalized_hashes: dict[str, str] = {}
    for role in REQUIRED_SOURCE_HASHES:
        value = supplied[role]
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise SecFilingGemmaSourceIdentityError(
                f"source_hashes.{role} must be one lowercase SHA-256 digest"
            )
        normalized_hashes[role] = value
    tree = {
        "schema_version": SOURCE_TREE_SCHEMA_VERSION,
        "source_role_paths": _frozen_role_paths(),
        "source_hashes": normalized_hashes,
    }
    return canonical_sha256(tree)


def _repository_module_name(repository_path: str) -> str:
    path = _validate_repository_path(repository_path, "source repository path")
    if path == "agent_benchmark/__init__.py":
        return "agent_benchmark"
    return path[:-3].replace("/", ".")


def _module_repository_path(module_name: str, *, location: str) -> str | None:
    if module_name == "agent_benchmark":
        return "agent_benchmark/__init__.py"
    if not module_name.startswith("agent_benchmark."):
        return None
    path = module_name.replace(".", "/") + ".py"
    return _validate_repository_path(path, location)


def _resolved_import_from_module(
    node: ast.ImportFrom,
    *,
    source_module_name: str,
    source_path: str,
) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = source_module_name.split(".")[:-1]
    if node.level > len(package_parts):
        raise SecFilingGemmaSourceIdentityError(
            f"Static import in {source_path} traverses outside agent_benchmark"
        )
    anchor_parts = package_parts[: len(package_parts) - (node.level - 1)]
    if node.module:
        anchor_parts.extend(node.module.split("."))
    return ".".join(anchor_parts)


def _static_local_import_paths(payload: bytes, *, source_path: str) -> tuple[str, ...]:
    """Return every statically imported local Python module path."""

    try:
        tree = ast.parse(payload, filename=source_path, mode="exec")
    except (SyntaxError, TypeError, ValueError) as exc:
        raise SecFilingGemmaSourceIdentityError(
            f"Pinned source {source_path} is not parseable Python"
        ) from exc
    source_module = _repository_module_name(source_path)
    dependencies: set[str] = set()
    if source_path != "agent_benchmark/__init__.py":
        dependencies.add("agent_benchmark/__init__.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                dependency = _module_repository_path(
                    alias.name,
                    location=f"static import in {source_path}",
                )
                if dependency is not None:
                    dependencies.add(dependency)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        imported_module = _resolved_import_from_module(
            node,
            source_module_name=source_module,
            source_path=source_path,
        )
        if imported_module == "agent_benchmark":
            for alias in node.names:
                if alias.name == "*":
                    raise SecFilingGemmaSourceIdentityError(
                        f"Static package wildcard import in {source_path} is not auditable"
                    )
                dependency = _module_repository_path(
                    f"agent_benchmark.{alias.name}",
                    location=f"static import in {source_path}",
                )
                if dependency is not None:
                    dependencies.add(dependency)
            continue
        dependency = _module_repository_path(
            imported_module,
            location=f"static import in {source_path}",
        )
        if dependency is not None:
            dependencies.add(dependency)
    dependencies.discard(source_path)
    return tuple(sorted(dependencies))


def _validate_declared_local_import_closure(
    *,
    paths: Mapping[str, str | None],
    payloads: Mapping[str, bytes | None],
) -> dict[str, list[str]]:
    """Require the declared path set to close over every static local import."""

    declared_paths = {path for path in paths.values() if path is not None}
    imports_by_source: dict[str, list[str]] = {}
    for role in REQUIRED_SOURCE_HASHES:
        source_path = paths[role]
        if source_path is None:
            continue
        payload = payloads[role]
        if type(payload) is not bytes:  # pragma: no cover - normalized by caller
            raise SecFilingGemmaSourceIdentityError(
                f"Resolved source role {role} has no bytes for import closure"
            )
        imported_paths = _static_local_import_paths(
            payload,
            source_path=source_path,
        )
        missing = sorted(set(imported_paths) - declared_paths)
        if missing:
            raise SecFilingGemmaSourceIdentityError(
                f"Pinned source {source_path} imports undeclared local sources: {missing}"
            )
        imports_by_source[source_path] = list(imported_paths)
    return {
        source_path: imports_by_source[source_path]
        for source_path in sorted(imports_by_source)
    }


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _lstat_without_alias(path: Path, location: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SecFilingGemmaSourceIdentityError(
            f"{location} is not an accessible filesystem object"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
        raise SecFilingGemmaSourceIdentityError(
            f"{location} must not be a symlink or reparse-point alias"
        )
    return metadata


def _canonical_runtime_root(value: Any) -> Path:
    if type(value) not in {str, _NATIVE_PATH_TYPE}:
        raise SecFilingGemmaSourceIdentityError(
            "repository_root must be one absolute built-in path"
        )
    root = Path(value)
    if not root.is_absolute():
        raise SecFilingGemmaSourceIdentityError(
            "repository_root must be absolute"
        )
    metadata = _lstat_without_alias(root, "repository_root")
    if not stat.S_ISDIR(metadata.st_mode):
        raise SecFilingGemmaSourceIdentityError(
            "repository_root must be a directory"
        )
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:  # pragma: no cover - lstat already gives the usual path
        raise SecFilingGemmaSourceIdentityError(
            "repository_root cannot be resolved"
        ) from exc
    if resolved != root:
        raise SecFilingGemmaSourceIdentityError(
            "repository_root must already be its canonical absolute path"
        )
    return root


def _runtime_repository_root() -> Path:
    """Derive the checkout from this loaded module, never from the caller."""

    module_path = Path(__file__)
    try:
        resolved_module_path = module_path.resolve(strict=True)
    except OSError as exc:
        raise SecFilingGemmaSourceIdentityError(
            "Loaded source-identity module path cannot be resolved"
        ) from exc
    if not module_path.is_absolute() or module_path != resolved_module_path:
        raise SecFilingGemmaSourceIdentityError(
            "Loaded source-identity module path uses a filesystem alias"
        )
    root = _canonical_runtime_root(resolved_module_path.parents[1])
    expected = root / "agent_benchmark" / "sec_filing_gemma_source_identity.py"
    if resolved_module_path != expected:
        raise SecFilingGemmaSourceIdentityError(
            "Loaded source-identity module is outside its canonical checkout path"
        )
    return root


def _verify_runtime_path_components(root: Path, relative_path: str) -> Path:
    current = root
    parts = relative_path.split("/")
    for index, part in enumerate(parts):
        current = current / part
        metadata = _lstat_without_alias(
            current,
            f"runtime source component {relative_path!r}",
        )
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise SecFilingGemmaSourceIdentityError(
                f"runtime source parent for {relative_path!r} is not a directory"
            )
        if index == len(parts) - 1 and not stat.S_ISREG(metadata.st_mode):
            raise SecFilingGemmaSourceIdentityError(
                f"runtime source {relative_path!r} is not a regular file"
            )
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:  # pragma: no cover - every component was lstat'd
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source {relative_path!r} cannot be resolved"
        ) from exc
    expected = root.joinpath(*parts)
    if resolved != expected:
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source {relative_path!r} resolves through a path alias"
        )
    return expected


def _read_bounded_regular_source(path: Path, relative_path: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source {relative_path!r} cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SecFilingGemmaSourceIdentityError(
                f"runtime source {relative_path!r} changed file type"
            )
        if before.st_size <= 0 or before.st_size > MAX_RUNTIME_SOURCE_BYTES_PER_FILE:
            raise SecFilingGemmaSourceIdentityError(
                f"runtime source {relative_path!r} exceeds the byte budget"
            )
        payload = bytearray()
        while len(payload) <= MAX_RUNTIME_SOURCE_BYTES_PER_FILE:
            chunk = os.read(
                descriptor,
                min(64 * 1024, MAX_RUNTIME_SOURCE_BYTES_PER_FILE + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) != before.st_size or len(payload) > MAX_RUNTIME_SOURCE_BYTES_PER_FILE:
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source {relative_path!r} changed size or exceeded its cap"
        )
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source {relative_path!r} changed while it was read"
        )
    current = _lstat_without_alias(path, f"runtime source {relative_path!r}")
    identity_current = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if identity_current != identity_after:
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source {relative_path!r} was replaced during audit"
        )
    return bytes(payload)


def _verify_loaded_module_path(relative_path: str, source_path: Path) -> str:
    module_name = _repository_module_name(relative_path)
    module = sys.modules.get(module_name)
    if module is None:
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source module {module_name} was not already loaded; "
            "source audit must never import candidate code"
        )
    module_file = getattr(module, "__file__", None)
    if type(module_file) is not str:
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source module {module_name} has no canonical source file"
        )
    raw_loaded_path = Path(module_file)
    if not raw_loaded_path.is_absolute():
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source module {module_name} uses a relative path alias"
        )
    try:
        loaded_path = raw_loaded_path.resolve(strict=True)
        expected_path = source_path.resolve(strict=True)
    except OSError as exc:
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source module {module_name} path cannot be resolved"
        ) from exc
    if (
        module_file != os.fspath(loaded_path)
        or raw_loaded_path != loaded_path
        or loaded_path != expected_path
        or raw_loaded_path.suffix != ".py"
    ):
        raise SecFilingGemmaSourceIdentityError(
            f"runtime source module {module_name} was loaded from another path"
        )
    return module_name


def _normalize_supplied_paths(value: Any) -> dict[str, str | None]:
    supplied = _exact_role_mapping(value, "source_paths_by_role")
    frozen = _frozen_role_paths()
    normalized: dict[str, str | None] = {}
    for role in REQUIRED_SOURCE_HASHES:
        expected_path = frozen[role]
        observed_path = supplied[role]
        if expected_path is None:
            if observed_path is not None:
                raise SecFilingGemmaSourceIdentityError(
                    f"Unresolved source role {role} cannot claim an invented path"
                )
            normalized[role] = None
            continue
        observed = _validate_repository_path(
            observed_path, f"source_paths_by_role.{role}"
        )
        if observed != expected_path:
            raise SecFilingGemmaSourceIdentityError(
                f"Source role {role} uses a path alias or another role's path"
            )
        normalized[role] = observed
    return normalized


def _normalize_source_bytes(
    value: Any, *, paths: Mapping[str, str | None]
) -> dict[str, bytes | None]:
    supplied = _exact_role_mapping(value, "source_bytes_by_role")
    normalized: dict[str, bytes | None] = {}
    for role in REQUIRED_SOURCE_HASHES:
        payload = supplied[role]
        if paths[role] is None:
            if payload is not None:
                raise SecFilingGemmaSourceIdentityError(
                    f"Unresolved source role {role} cannot claim detached bytes"
                )
            normalized[role] = None
            continue
        if type(payload) is not bytes or not payload:
            raise SecFilingGemmaSourceIdentityError(
                f"source_bytes_by_role.{role} must be nonempty detached bytes"
            )
        normalized[role] = payload
    return normalized


def audit_candidate_source_identity(
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    source_paths_by_role: Mapping[str, str | None],
    source_bytes_by_role: Mapping[str, bytes | None],
) -> dict[str, Any]:
    """Hash every resolved source and return a canonical non-authorizing audit."""

    candidate = _plain_json(candidate_manifest, "candidate_manifest")
    try:
        candidate_hash = validate_candidate_manifest(
            candidate, expected_candidate_sha256=expected_candidate_sha256
        )
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaSourceIdentityError(
            "Candidate source-hash binding is not canonical"
        ) from exc

    paths = _normalize_supplied_paths(source_paths_by_role)
    payloads = _normalize_source_bytes(source_bytes_by_role, paths=paths)
    local_imports = _validate_declared_local_import_closure(
        paths=paths,
        payloads=payloads,
    )
    candidate_hashes = {
        role: candidate["bindings"]["source_hashes"][role]
        for role in REQUIRED_SOURCE_HASHES
    }
    computed_tree_hash = canonical_source_tree_sha256(candidate_hashes)
    candidate_tree_hash = candidate["bindings"]["source_tree_sha256"]
    if not hmac.compare_digest(computed_tree_hash, candidate_tree_hash):
        raise SecFilingGemmaSourceIdentityError(
            "Candidate source_tree_sha256 does not bind the canonical role/path/hash tree"
        )
    resolved_hashes: dict[str, str] = {}
    resolved_sources: list[dict[str, Any]] = []
    unresolved_sources: list[dict[str, str]] = []
    total_bytes = 0
    for role in REQUIRED_SOURCE_HASHES:
        path = paths[role]
        payload = payloads[role]
        candidate_pin = candidate_hashes[role]
        if path is None:
            unresolved_sources.append(
                {
                    "role": role,
                    "candidate_source_sha256": candidate_pin,
                    "reason": "no_single_frozen_repository_source_owner",
                }
            )
            continue
        if payload is None:  # pragma: no cover - normalized above
            raise SecFilingGemmaSourceIdentityError(
                f"Resolved source role {role} has no detached bytes"
            )
        digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(digest, candidate_pin):
            raise SecFilingGemmaSourceIdentityError(
                f"Exact source bytes for {role} differ from the candidate pin"
            )
        resolved_hashes[role] = digest
        total_bytes += len(payload)
        resolved_sources.append(
            {
                "role": role,
                "repository_path": path,
                "byte_count": len(payload),
                "source_sha256": digest,
                "candidate_source_sha256": candidate_pin,
            }
        )

    unresolved_roles = [item["role"] for item in unresolved_sources]
    role_ownership_complete = not unresolved_roles
    body = {
        "schema_version": SOURCE_IDENTITY_RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "detached_non_authorizing_source_identity_audit",
        "candidate_sha256": candidate_hash,
        "candidate_source_tree_sha256": candidate_tree_hash,
        "computed_source_tree_sha256": computed_tree_hash,
        "required_source_roles": list(REQUIRED_SOURCE_HASHES),
        "source_role_paths": paths,
        "source_role_paths_sha256": canonical_sha256(paths),
        "declared_static_local_imports_by_source": local_imports,
        "declared_static_local_import_closure_sha256": canonical_sha256(
            local_imports
        ),
        "declared_static_local_import_closure_complete": True,
        "candidate_source_hashes": candidate_hashes,
        "candidate_source_hashes_sha256": canonical_sha256(candidate_hashes),
        "resolved_source_hashes": resolved_hashes,
        "resolved_source_hashes_sha256": canonical_sha256(resolved_hashes),
        "resolved_sources": resolved_sources,
        "unresolved_sources": unresolved_sources,
        "unresolved_roles": unresolved_roles,
        "resolved_role_count": len(resolved_sources),
        "unresolved_role_count": len(unresolved_sources),
        "source_byte_count_total": total_bytes,
        "role_ownership_complete": role_ownership_complete,
        "runtime_source_files_verified": False,
        "runtime_module_paths_verified": False,
        "complete": False,
        "authorizes": False,
        "authorization_scope": "none",
    }
    return {**body, "source_identity_receipt_sha256": canonical_sha256(body)}


def audit_runtime_candidate_source_identity(
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    """Audit candidate pins against canonical files loaded by this process.

    This is intentionally still non-authorizing while conceptual source roles
    remain unresolved.  Unlike :func:`audit_candidate_source_identity`, the
    caller cannot provide detached bytes or alternate paths.
    """

    root = _runtime_repository_root()
    candidate = _plain_json(candidate_manifest, "candidate_manifest")
    try:
        candidate_hash = validate_candidate_manifest(
            candidate,
            expected_candidate_sha256=expected_candidate_sha256,
        )
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaSourceIdentityError(
            "Candidate runtime source binding is not canonical"
        ) from exc

    paths = _frozen_role_paths()
    payloads: dict[str, bytes | None] = {}
    loaded_modules: list[dict[str, str]] = []
    total_bytes = 0
    for role in REQUIRED_SOURCE_HASHES:
        relative_path = paths[role]
        if relative_path is None:
            payloads[role] = None
            continue
        source_path = _verify_runtime_path_components(root, relative_path)
        module_name = _verify_loaded_module_path(relative_path, source_path)
        payload = _read_bounded_regular_source(source_path, relative_path)
        total_bytes += len(payload)
        if total_bytes > MAX_RUNTIME_SOURCE_BYTES_TOTAL:
            raise SecFilingGemmaSourceIdentityError(
                "Runtime source tree exceeds the aggregate byte budget"
            )
        payloads[role] = payload
        loaded_modules.append(
            {
                "role": role,
                "module_name": module_name,
                "repository_path": relative_path,
            }
        )

    detached = audit_candidate_source_identity(
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate_hash,
        source_paths_by_role=paths,
        source_bytes_by_role=payloads,
    )
    body = {
        key: value
        for key, value in detached.items()
        if key != "source_identity_receipt_sha256"
    }
    body.update(
        {
            "receipt_kind": "runtime_non_authorizing_source_identity_audit",
            "runtime_source_files_verified": True,
            "runtime_module_paths_verified": True,
            "runtime_module_files_attested": True,
            # Python exposes the loaded path, but not a proof that the current
            # source bytes at that path are the bytes that created every live
            # code object.  Keep completeness false until a fresh owned worker
            # binds startup/import to the already hashed source tree.
            "runtime_executing_code_bytes_attested": False,
            "caller_supplied_paths_or_bytes_accepted": False,
            "runtime_modules": loaded_modules,
            "runtime_modules_sha256": canonical_sha256(loaded_modules),
            "complete": False,
        }
    )
    return {**body, "source_identity_receipt_sha256": canonical_sha256(body)}


def validate_complete_candidate_source_identity(
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    source_paths_by_role: Mapping[str, str | None],
    source_bytes_by_role: Mapping[str, bytes | None],
) -> dict[str, Any]:
    """Require complete source ownership; never grant stage authorization."""

    receipt = audit_candidate_source_identity(
        candidate_manifest=candidate_manifest,
        expected_candidate_sha256=expected_candidate_sha256,
        source_paths_by_role=source_paths_by_role,
        source_bytes_by_role=source_bytes_by_role,
    )
    if not receipt["complete"]:
        raise SecFilingGemmaSourceIdentityIncompleteError(
            "Detached source evidence is non-authorizing and source identity "
            "remains incomplete for roles: "
            + ", ".join(receipt["unresolved_roles"])
        )
    return receipt


def validate_complete_runtime_candidate_source_identity(
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    """Require all source owners and actual runtime files to be verified."""

    receipt = audit_runtime_candidate_source_identity(
        candidate_manifest=candidate_manifest,
        expected_candidate_sha256=expected_candidate_sha256,
    )
    if not receipt["complete"]:
        unresolved = receipt["unresolved_roles"]
        detail = (
            "roles: " + ", ".join(unresolved)
            if unresolved
            else "executing code bytes are not startup-attested"
        )
        raise SecFilingGemmaSourceIdentityIncompleteError(
            "Runtime source identity remains incomplete for " + detail
        )
    return receipt


__all__ = [
    "CANONICAL_SOURCE_ROLE_PATHS",
    "SOURCE_IDENTITY_RECEIPT_SCHEMA_VERSION",
    "SOURCE_TREE_SCHEMA_VERSION",
    "UNRESOLVED_SOURCE_ROLES",
    "SecFilingGemmaSourceIdentityError",
    "SecFilingGemmaSourceIdentityIncompleteError",
    "audit_candidate_source_identity",
    "audit_runtime_candidate_source_identity",
    "canonical_source_tree_sha256",
    "validate_complete_candidate_source_identity",
    "validate_complete_runtime_candidate_source_identity",
]
