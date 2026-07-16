"""Live Git and source-tree verification for the SEC/Gemma overlay v2.1.

The verifier performs read-only local checks.  It does not fetch, mutate Git,
open a network connection, or trust caller-supplied commit and source hashes.
The returned object is an in-process attestation; effectful code must verify it
again immediately before consuming an attempt.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import hmac
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Final, Mapping

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    BRANCH_NAME,
    CONTRACT_SHA256,
    NEW_SOURCE_FILES,
    SOURCE_PIN_FILES,
    SOURCE_PINS,
    canonical_sha256,
)


SOURCE_VERIFICATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-source-verification-v3"
)
PREREGISTRATION_COMMIT: Final[str] = (
    "939856e6773fc6dd5bbf468193a29f34c8724c6e"
)
EXPECTED_ORIGIN_URL: Final[str] = (
    "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git"
)
CONTRACT_SOURCE_PATH: Final[str] = (
    "agent_benchmark/sec_gemma_online_risk_overlay_contract.py"
)

REQUIRED_NEW_SOURCE_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    dict(NEW_SOURCE_FILES)
)

_DEPENDENCY_ROOT_NEW_ROLES: Final[tuple[str, ...]] = (
    "production",
    "acquisition",
    "runtime",
    "publisher",
    "registry",
    "vault",
    "runner",
)
_DEPENDENCY_ROOT_REUSED_ROLES: Final[tuple[str, ...]] = (
    "sec_audit_transport",
    "sec_point_in_time",
    "sec_filing_gemma_corpus",
    "sec_filing_gemma_ollama",
)
_ALLOWED_EXTERNAL_IMPORT_DISTRIBUTIONS: Final[Mapping[str, str]] = (
    MappingProxyType(
        {
            "requests": "requests",
            "urllib3": "urllib3",
            "certifi": "certifi",
            "charset_normalizer": "charset-normalizer",
            "idna": "idna",
        }
    )
)
_NUMERICAL_TIME_IMPORT_DISTRIBUTIONS: Final[Mapping[str, str]] = (
    MappingProxyType(
        {
            "numpy": "numpy",
            "tzdata": "tzdata",
        }
    )
)
NUMERICAL_TIME_IMPORT_ROOTS: Final[frozenset[str]] = frozenset(
    _NUMERICAL_TIME_IMPORT_DISTRIBUTIONS
)
_NUMERICAL_IMPORT_SOURCE_ALLOWLIST: Final[
    Mapping[str, frozenset[str]]
] = MappingProxyType(
    {
        SOURCE_PIN_FILES["sec_filing_gemma_learner"]: frozenset(
            {"numpy"}
        ),
    }
)
_FORBIDDEN_NETWORK_IMPORT_ROOTS: Final[frozenset[str]] = frozenset(
    {"yfinance"}
)
_FORBIDDEN_LOCAL_DEPENDENCY_PATHS: Final[frozenset[str]] = frozenset(
    {
        "agent_benchmark/benchmark_engine.py",
        "agent_benchmark/deterministic_aapl.py",
        "agent_benchmark/market_data.py",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_ROLE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_VERIFIED_SENTINEL = object()
_REQUESTS_RUNTIME_IMPORTS: Final[frozenset[str]] = frozenset(
    _ALLOWED_EXTERNAL_IMPORT_DISTRIBUTIONS
)
_REQUESTS_OPTIONAL_IMPORTS: Final[frozenset[str]] = frozenset(
    {
        "brotli",
        "brotlicffi",
        "chardet",
        "h2",
        "simplejson",
        "socks",
        "zstandard",
    }
)


class SecGemmaOnlineRiskOverlaySourceVerificationError(RuntimeError):
    """The live repository differs from the committed experiment binding."""


class _FrozenRequestsImportGuard(importlib.abc.MetaPathFinder):
    """Reject external modules outside the frozen Requests dependency set."""

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        del path, target
        root = fullname.split(".", 1)[0]
        if (
            root in _REQUESTS_RUNTIME_IMPORTS
            or root == "agent_benchmark"
            or root == "__future__"
            or root in sys.stdlib_module_names
        ):
            return None
        raise ModuleNotFoundError(
            "External import is outside the frozen Requests runtime closure",
            name=fullname,
        )


def _is_reparse(details: os.stat_result) -> bool:
    attributes = getattr(details, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & flag)


def _canonical_repo_root(value: Path) -> Path:
    if not isinstance(value, Path) or not value.is_absolute():
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Repository root must be an absolute pathlib.Path"
        )
    try:
        details = value.lstat()
        resolved = value.resolve(strict=True)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Repository root is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or _is_reparse(details)
        or resolved != value
    ):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Repository root must be one canonical real directory"
        )
    git_dir = value / ".git"
    try:
        git_details = git_dir.lstat()
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Repository must contain one real .git directory"
        ) from exc
    if (
        not stat.S_ISDIR(git_details.st_mode)
        or stat.S_ISLNK(git_details.st_mode)
        or _is_reparse(git_details)
    ):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Repository .git metadata cannot be linked or substituted"
        )
    return value


def _git(
    repo_root: Path,
    *arguments: str,
) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Local Git verification could not run"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Local Git verification failed: {detail or arguments[0]}"
        )
    return bytes(result.stdout)


def _git_text(repo_root: Path, *arguments: str) -> str:
    try:
        return _git(repo_root, *arguments).decode(
            "utf-8", errors="strict"
        ).strip()
    except UnicodeDecodeError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Git identity output is not strict UTF-8"
        ) from exc


def _canonical_source_path(value: str, location: str) -> str:
    if type(value) is not str or "\\" in value:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"{location} must be one canonical repository-relative path"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.suffix != ".py"
        or str(pure) != value
    ):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"{location} must be one canonical Python source path"
        )
    return value


def _inventory() -> dict[str, str]:
    observed = dict(REQUIRED_NEW_SOURCE_PATHS)
    if set(observed) != set(REQUIRED_NEW_SOURCE_PATHS):
        raise RuntimeError("Required source inventory contains duplicate roles")
    for role, path in observed.items():
        if _ROLE_RE.fullmatch(role) is None:
            raise RuntimeError("Required source inventory role is invalid")
        _canonical_source_path(path, f"required source {role}")
    if len(set(observed.values())) != len(observed):
        raise RuntimeError("Required source inventory aliases a source path")
    return dict(sorted(observed.items()))


def _source_imports(
    *,
    relative_path: str,
    payload: bytes,
) -> tuple[set[str], set[str]]:
    try:
        text = payload.decode("utf-8", errors="strict")
        tree = ast.parse(text, filename=relative_path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Participating source cannot be parsed: {relative_path}"
        ) from exc
    current_parent = list(PurePosixPath(relative_path).parent.parts)
    local_paths: set[str] = set()
    external_roots: set[str] = set()

    def add_module(module: str) -> None:
        root = module.split(".", 1)[0]
        if root == "agent_benchmark":
            local_paths.add(module.replace(".", "/") + ".py")
        elif root not in sys.stdlib_module_names and root != "__future__":
            external_roots.add(root)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module is not None:
                add_module(node.module)
            continue
        keep = len(current_parent) - (node.level - 1)
        if keep < 1:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                f"Relative import escapes agent_benchmark: {relative_path}"
            )
        base = current_parent[:keep]
        if node.module is not None:
            base.extend(node.module.split("."))
            local_paths.add("/".join(base) + ".py")
        else:
            for alias in node.names:
                local_paths.add("/".join([*base, alias.name]) + ".py")
    return local_paths, external_roots


def _dependency_closure(
    repo_root: Path,
    *,
    participant_paths: set[str],
    new_inventory: Mapping[str, str],
) -> tuple[list[str], set[str]]:
    roots = [
        new_inventory[role] for role in _DEPENDENCY_ROOT_NEW_ROLES
    ]
    roots.extend(
        SOURCE_PIN_FILES[role]
        for role in _DEPENDENCY_ROOT_REUSED_ROLES
    )
    pending = list(roots)
    visited: set[str] = set()
    external_imports: set[str] = set()
    while pending:
        relative_path = _canonical_source_path(
            pending.pop(), "dependency source"
        )
        if relative_path in visited:
            continue
        payload = _verify_real_source(repo_root, relative_path)
        local_imports, external_roots = _source_imports(
            relative_path=relative_path,
            payload=payload,
        )
        forbidden_network = (
            external_roots & _FORBIDDEN_NETWORK_IMPORT_ROOTS
        )
        if forbidden_network:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Dependency closure imports a forbidden network-capable "
                f"distribution: {sorted(forbidden_network)[0]}"
            )
        allowed_numerical = _NUMERICAL_IMPORT_SOURCE_ALLOWLIST.get(
            relative_path, frozenset()
        )
        unscoped_numerical = (
            external_roots & NUMERICAL_TIME_IMPORT_ROOTS
        ) - allowed_numerical
        if unscoped_numerical:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Numerical/time distribution import is outside its exact "
                f"source-pinned adapter: {sorted(unscoped_numerical)[0]}"
            )
        external_imports.update(external_roots)
        visited.add(relative_path)
        for imported_path in sorted(local_imports):
            if imported_path in _FORBIDDEN_LOCAL_DEPENDENCY_PATHS:
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Dependency closure imports a forbidden market/network "
                    f"path: {imported_path}"
                )
            candidate = repo_root / PurePosixPath(imported_path)
            if not candidate.exists():
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Dependency closure imports a missing local source: "
                    f"{imported_path}"
                )
            if imported_path not in visited:
                pending.append(imported_path)
    dependencies = sorted(visited - participant_paths)
    return dependencies, external_imports


def _external_distribution_identity(
    import_name: str,
) -> tuple[
    str,
    str,
    str,
    tuple[tuple[str, str], ...],
    str,
    str | None,
]:
    """Bind every importable module file shipped by one allowed distribution."""

    distribution_name = _ALLOWED_EXTERNAL_IMPORT_DISTRIBUTIONS.get(
        import_name
    )
    if distribution_name is None:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Production imports an unlisted external distribution: "
            f"{import_name}"
        )
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Allowed production distribution is unavailable: {import_name}"
        ) from exc
    distribution_files = distribution.files
    if distribution_files is None:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Allowed production distribution has no exact file inventory: "
            f"{import_name}"
        )
    module_suffixes = {
        *importlib.machinery.SOURCE_SUFFIXES,
        *importlib.machinery.EXTENSION_SUFFIXES,
    }
    module_files: list[tuple[str, str]] = []
    observed_paths: set[str] = set()
    for raw_path in distribution_files:
        relative_text = str(raw_path).replace("\\", "/")
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or str(relative) != relative_text
            or not any(relative_text.endswith(suffix) for suffix in module_suffixes)
        ):
            continue
        if relative_text in observed_paths:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Allowed production distribution repeats a module file: "
                f"{import_name}"
            )
        observed_paths.add(relative_text)
        module_path = Path(distribution.locate_file(raw_path))
        try:
            resolved = module_path.resolve(strict=True)
            details = resolved.stat()
            module_bytes = resolved.read_bytes()
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                f"Allowed production module cannot be hashed: "
                f"{import_name}:{relative_text}"
            ) from exc
        if not stat.S_ISREG(details.st_mode):
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                f"Allowed production module is not a regular file: "
                f"{import_name}:{relative_text}"
            )
        module_files.append(
            (relative_text, hashlib.sha256(module_bytes).hexdigest())
        )
    module_files.sort(key=lambda item: item[0])
    if not module_files:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Allowed production distribution has no module files: "
            f"{import_name}"
        )
    module_files_sha256 = canonical_sha256(
        [
            {"path": path, "sha256": digest}
            for path, digest in module_files
        ]
    )
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is not None:
        try:
            canonical_direct_url = json.dumps(
                json.loads(direct_url),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (json.JSONDecodeError, UnicodeEncodeError, ValueError) as exc:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                f"Distribution direct_url metadata is invalid: {import_name}"
            ) from exc
        direct_url_sha256: str | None = hashlib.sha256(
            canonical_direct_url
        ).hexdigest()
    else:
        direct_url_sha256 = None
    return (
        import_name,
        distribution_name,
        distribution.version,
        tuple(module_files),
        module_files_sha256,
        direct_url_sha256,
    )


def _external_distribution_identities(
    external_imports: set[str],
) -> tuple[
    tuple[
        str,
        str,
        str,
        tuple[tuple[str, str], ...],
        str,
        str | None,
    ],
    ...,
]:
    bound = set(external_imports)
    if "requests" in bound:
        bound.update(_ALLOWED_EXTERNAL_IMPORT_DISTRIBUTIONS)
    return tuple(
        _external_distribution_identity(import_name)
        for import_name in sorted(bound)
    )


def _numerical_time_runtime_import_roots(
    repo_root: Path,
) -> tuple[str, ...]:
    """Probe the exact isolated learner/time import for external packages."""

    distribution_roots = sorted(
        {
            str(
                Path(
                    importlib.metadata.distribution(
                        distribution_name
                    ).locate_file("")
                ).resolve(strict=True)
            )
            for distribution_name in (
                _NUMERICAL_TIME_IMPORT_DISTRIBUTIONS.values()
            )
        }
    )
    script = r"""
import importlib.metadata
import json
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve(strict=True)
distribution_roots = [
    Path(item).resolve(strict=True)
    for item in json.loads(sys.argv[2])
]
for item in reversed(distribution_roots):
    sys.path.insert(0, str(item))
sys.path.insert(0, str(repo_root))
before = set(sys.modules)
import agent_benchmark.sec_filing_gemma_learner
from zoneinfo import ZoneInfo
ZoneInfo("America/New_York")
loaded = sorted(set(sys.modules) - before)
packages = importlib.metadata.packages_distributions()
owned = {}
unowned = set()
for module_name in loaded:
    module = sys.modules[module_name]
    module_file = getattr(module, "__file__", None)
    if not module_file:
        continue
    resolved = Path(module_file).resolve(strict=True)
    if resolved == repo_root or repo_root in resolved.parents:
        continue
    if not any(
        resolved == root or root in resolved.parents
        for root in distribution_roots
    ):
        continue
    import_root = module_name.split(".", 1)[0]
    distributions = sorted(packages.get(import_root, ()))
    if not distributions:
        unowned.add(import_root)
        continue
    owned[import_root] = distributions
print(
    json.dumps(
        {"owned": owned, "unowned": sorted(unowned)},
        sort_keys=True,
        separators=(",", ":"),
    )
)
"""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("PYTHON")
    }
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                script,
                str(repo_root),
                json.dumps(distribution_roots, separators=(",", ":")),
            ],
            cwd=repo_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Isolated numerical/time runtime import probe could not run"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Isolated numerical/time runtime import probe failed: "
            f"{detail or result.returncode}"
        )
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Numerical/time runtime import probe returned invalid JSON"
        ) from exc
    expected = {
        "owned": {
            import_name: [distribution_name]
            for import_name, distribution_name in sorted(
                _NUMERICAL_TIME_IMPORT_DISTRIBUTIONS.items()
            )
        },
        "unowned": [],
    }
    if observed != expected:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Numerical/time runtime loaded an unbound external distribution"
        )
    return tuple(sorted(observed["owned"]))


def _numerical_time_distribution_identity(
    import_name: str,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[tuple[str, str], ...],
    str,
    str | None,
]:
    """Bind active package origin plus all package and native runtime files."""

    distribution_name = _NUMERICAL_TIME_IMPORT_DISTRIBUTIONS.get(
        import_name
    )
    if distribution_name is None:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Numerical/time runtime imports an unlisted distribution: "
            f"{import_name}"
        )
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Numerical/time distribution is unavailable: {import_name}"
        ) from exc
    distribution_files = distribution.files
    if distribution_files is None:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Numerical/time distribution has no exact file inventory: "
            f"{import_name}"
        )
    package_roots = {import_name, f"{import_name}.libs"}
    runtime_files: list[tuple[str, str]] = []
    resolved_files: dict[Path, tuple[str, str]] = {}
    observed_paths: set[str] = set()
    for raw_path in distribution_files:
        relative_text = str(raw_path).replace("\\", "/")
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or str(relative) != relative_text
            or relative.parts[0] not in package_roots
        ):
            continue
        if relative_text in observed_paths:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Numerical/time distribution repeats a runtime file: "
                f"{import_name}"
            )
        observed_paths.add(relative_text)
        runtime_path = Path(distribution.locate_file(raw_path))
        try:
            resolved = runtime_path.resolve(strict=True)
            details = resolved.stat()
            runtime_bytes = resolved.read_bytes()
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Numerical/time runtime file cannot be hashed: "
                f"{import_name}:{relative_text}"
            ) from exc
        if not stat.S_ISREG(details.st_mode):
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Numerical/time runtime file is not regular: "
                f"{import_name}:{relative_text}"
            )
        digest = hashlib.sha256(runtime_bytes).hexdigest()
        runtime_files.append((relative_text, digest))
        resolved_files[resolved] = (relative_text, digest)
    runtime_files.sort(key=lambda item: item[0])
    if not runtime_files:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Numerical/time distribution has no runtime files: "
            f"{import_name}"
        )
    runtime_files_sha256 = canonical_sha256(
        [
            {"path": path, "sha256": digest}
            for path, digest in runtime_files
        ]
    )
    spec = importlib.util.find_spec(import_name)
    if spec is None or spec.origin in {None, "built-in", "frozen"}:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Numerical/time active module origin is unavailable: {import_name}"
        )
    try:
        active_origin = Path(spec.origin).resolve(strict=True)
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Numerical/time active module origin is invalid: {import_name}"
        ) from exc
    origin_identity = resolved_files.get(active_origin)
    if origin_identity is None:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Numerical/time active module origin is outside its bound "
            f"distribution: {import_name}"
        )
    module_origin, module_origin_sha256 = origin_identity
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is not None:
        try:
            canonical_direct_url = json.dumps(
                json.loads(direct_url),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        except (json.JSONDecodeError, UnicodeEncodeError, ValueError) as exc:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Numerical/time distribution direct_url metadata is invalid: "
                f"{import_name}"
            ) from exc
        direct_url_sha256: str | None = hashlib.sha256(
            canonical_direct_url
        ).hexdigest()
    else:
        direct_url_sha256 = None
    return (
        import_name,
        distribution_name,
        distribution.version,
        module_origin,
        module_origin_sha256,
        tuple(runtime_files),
        runtime_files_sha256,
        direct_url_sha256,
    )


def _numerical_time_distribution_identities(
    repo_root: Path,
    external_imports: set[str],
) -> tuple[
    tuple[
        str,
        str,
        str,
        str,
        str,
        tuple[tuple[str, str], ...],
        str,
        str | None,
    ],
    ...,
]:
    if "numpy" not in external_imports:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Exact pinned NumPy learner is absent from the dependency closure"
        )
    import_roots = _numerical_time_runtime_import_roots(repo_root)
    return tuple(
        _numerical_time_distribution_identity(import_name)
        for import_name in import_roots
    )


def _allowed_requests_runtime_files(
) -> tuple[
    dict[Path, tuple[str, str, str]],
    dict[str, Path],
]:
    """Resolve and hash the exact files eligible for a Requests runtime."""

    allowed_files: dict[Path, tuple[str, str, str]] = {}
    package_entries: dict[str, Path] = {}
    for import_name in sorted(_REQUESTS_RUNTIME_IMPORTS):
        (
            observed_import,
            distribution_name,
            _version,
            module_files,
            _module_files_sha256,
            _direct_url_sha256,
        ) = _external_distribution_identity(import_name)
        if observed_import != import_name:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Allowed Requests distribution identity changed"
            )
        distribution = importlib.metadata.distribution(distribution_name)
        expected_entry_relative = f"{import_name}/__init__.py"
        expected_entry: Path | None = None
        for relative_path, digest in module_files:
            try:
                resolved = Path(
                    distribution.locate_file(PurePosixPath(relative_path))
                ).resolve(strict=True)
            except OSError as exc:
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Allowed Requests runtime file is unavailable"
                ) from exc
            prior = allowed_files.get(resolved)
            material = (import_name, relative_path, digest)
            if prior is not None and prior != material:
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Allowed Requests distributions alias a runtime file"
                )
            allowed_files[resolved] = material
            if relative_path == expected_entry_relative:
                expected_entry = resolved
        if expected_entry is None:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Allowed Requests distribution lacks its package entry"
            )
        try:
            spec = importlib.util.find_spec(import_name)
            observed_origin = (
                Path(spec.origin).resolve(strict=True)
                if spec is not None and spec.origin is not None
                else None
            )
        except (ImportError, AttributeError, OSError, ValueError) as exc:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Allowed Requests package origin cannot be resolved"
            ) from exc
        if observed_origin != expected_entry:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Allowed Requests package is shadowed or redirected"
            )
        package_entries[import_name] = expected_entry
    return allowed_files, package_entries


def _audit_allowed_requests_modules(
    *,
    requests_module: Any,
    allowed_files: Mapping[Path, tuple[str, str, str]],
    package_entries: Mapping[str, Path],
    modules_before_load: set[str],
) -> None:
    """Prove the selected resolver and every newly loaded external module."""

    try:
        charset_normalizer = importlib.import_module("charset_normalizer")
        requests_compat = importlib.import_module("requests.compat")
        requests_models = importlib.import_module("requests.models")
        requests_packages = importlib.import_module("requests.packages")
        urllib3_response = importlib.import_module("urllib3.response")
        urllib3_request = importlib.import_module("urllib3.util.request")
    except ImportError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Allowed Requests runtime is incomplete"
        ) from exc
    if (
        requests_compat.chardet is not charset_normalizer
        or requests_models.chardet is not charset_normalizer
        or requests_packages.chardet is not charset_normalizer
        or any(
            name == "chardet" or name.startswith("chardet.")
            for name in sys.modules
        )
    ):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Requests did not select the frozen charset-normalizer resolver"
        )
    session = None
    try:
        session = requests_module.Session()
        if type(session) is not requests_module.Session:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Requests runtime constructed a foreign session"
            )
        session.trust_env = False
        prepared = session.prepare_request(
            requests_module.Request(
                "GET",
                "https://example.invalid/frozen-requests-runtime-smoke",
            )
        )
        adapter = session.get_adapter(prepared.url)
        if (
            prepared.method != "GET"
            or prepared.url
            != "https://example.invalid/frozen-requests-runtime-smoke"
            or type(adapter).__module__ != "requests.adapters"
            or type(adapter).__name__ != "HTTPAdapter"
        ):
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Requests no-network runtime smoke failed"
            )
    finally:
        if session is not None:
            session.close()
    if (
        requests_compat.has_simplejson is not False
        or requests_compat.json is not json
        or requests_models.complexjson is not json
        or urllib3_response.brotli is not None
        or urllib3_response.HAS_ZSTD is not False
        or "br" in urllib3_request.ACCEPT_ENCODING.split(",")
        or "zstd" in urllib3_request.ACCEPT_ENCODING.split(",")
        or any(
            name in sys.modules
            or any(
                loaded.startswith(f"{name}.") for loaded in sys.modules
            )
            for name in _REQUESTS_OPTIONAL_IMPORTS
        )
    ):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Requests loaded an unlisted optional runtime"
        )

    observed_top_level: set[str] = set()
    checked_hashes: dict[Path, str] = {}
    names_to_check = set(sys.modules) - modules_before_load
    names_to_check.update(
        name
        for name in sys.modules
        if (
            name.split(".", 1)[0] in _REQUESTS_RUNTIME_IMPORTS
            or name.startswith("requests.packages.")
        )
    )
    for module_name in sorted(names_to_check):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        raw_path = getattr(module, "__file__", None)
        root = module_name.split(".", 1)[0]
        if raw_path is None:
            if (
                root not in sys.stdlib_module_names
                and root not in {"__future__", "agent_benchmark"}
            ):
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Requests loaded an unbound module without a source file"
                )
            continue
        try:
            resolved = Path(raw_path).resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Requests loaded a module with an invalid source path"
            ) from exc
        expected = allowed_files.get(resolved)
        if expected is None:
            if (
                root not in sys.stdlib_module_names
                and root not in {"__future__", "agent_benchmark"}
            ):
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Requests loaded an external module outside the frozen inventory"
                )
            continue
        import_name, _relative_path, expected_digest = expected
        digest = checked_hashes.get(resolved)
        if digest is None:
            try:
                digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            except OSError as exc:
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Requests runtime module cannot be rehashed"
                ) from exc
            checked_hashes[resolved] = digest
        if not hmac.compare_digest(digest, expected_digest):
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Requests runtime module differs from its bound inventory"
            )
        if module_name in _REQUESTS_RUNTIME_IMPORTS:
            if resolved != package_entries[module_name]:
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Requests top-level package resolved to the wrong file"
                )
            observed_top_level.add(import_name)
    if observed_top_level != _REQUESTS_RUNTIME_IMPORTS:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Requests runtime did not load the exact frozen distribution set"
        )
    if requests_module is not sys.modules.get("requests"):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Requests runtime module identity changed"
        )


def load_allowed_requests() -> Any:
    """Load Requests through its frozen charset-normalizer-only dependency path."""

    if any(
        name == "chardet" or name.startswith("chardet.")
        for name in sys.modules
    ):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Chardet was loaded before the frozen Requests runtime"
        )
    allowed_files, package_entries = _allowed_requests_runtime_files()
    before = set(sys.modules)
    guard = _FrozenRequestsImportGuard()
    sys.meta_path.insert(0, guard)
    guard_removed = False
    try:
        requests_module = importlib.import_module("requests")
        _audit_allowed_requests_modules(
            requests_module=requests_module,
            allowed_files=allowed_files,
            package_entries=package_entries,
            modules_before_load=before,
        )
    finally:
        try:
            sys.meta_path.remove(guard)
            guard_removed = True
        except ValueError:
            guard_removed = False
    if not guard_removed:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Frozen Requests import guard was displaced"
        )
    return requests_module


def _verify_real_source(repo_root: Path, relative_path: str) -> bytes:
    path = repo_root
    for part in PurePosixPath(relative_path).parts:
        path = path / part
        try:
            details = path.lstat()
        except OSError as exc:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                f"Required source is missing: {relative_path}"
            ) from exc
        if stat.S_ISLNK(details.st_mode) or _is_reparse(details):
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                f"Required source is linked or reparsed: {relative_path}"
            )
    if not stat.S_ISREG(details.st_mode):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Required source is not a regular file: {relative_path}"
        )
    if details.st_nlink != 1:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Required source is hard-linked: {relative_path}"
        )
    try:
        if path.resolve(strict=True) != path:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                f"Required source path is aliased: {relative_path}"
            )
        return path.read_bytes()
    except OSError as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Required source cannot be read: {relative_path}"
        ) from exc


def _verify_tracked_blob(
    repo_root: Path,
    relative_path: str,
    worktree_bytes: bytes,
) -> str:
    stage = _git_text(repo_root, "ls-files", "--stage", "--", relative_path)
    lines = stage.splitlines()
    if len(lines) != 1:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Required source is not one tracked Git blob: {relative_path}"
        )
    prefix, separator, observed_path = lines[0].partition("\t")
    fields = prefix.split()
    if (
        separator != "\t"
        or observed_path != relative_path
        or len(fields) != 3
        or fields[0] not in {"100644", "100755"}
        or fields[2] != "0"
    ):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Required source has a non-regular Git mode: {relative_path}"
        )
    committed = _git(repo_root, "show", f"HEAD:{relative_path}")
    if committed != worktree_bytes:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            f"Required source differs from its committed blob: {relative_path}"
        )
    return hashlib.sha256(worktree_bytes).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSourceTree:
    """Opaque evidence produced only by a successful live verification."""

    repo_root: Path
    branch: str
    origin_url: str
    preregistration_commit: str
    head_commit: str
    upstream_commit: str
    contract_source_sha256: str
    reused_sources: tuple[tuple[str, str, str], ...]
    new_sources: tuple[tuple[str, str, str], ...]
    dependency_sources: tuple[tuple[str, str], ...]
    external_distributions: tuple[
        tuple[
            str,
            str,
            str,
            tuple[tuple[str, str], ...],
            str,
            str | None,
        ],
        ...,
    ]
    numerical_time_distributions: tuple[
        tuple[
            str,
            str,
            str,
            str,
            str,
            tuple[tuple[str, str], ...],
            str,
            str | None,
        ],
        ...,
    ]
    dependency_closure_sha256: str
    verification_sha256: str
    _sentinel: object

    def __init__(
        self,
        *,
        repo_root: Path,
        branch: str,
        origin_url: str,
        preregistration_commit: str,
        head_commit: str,
        upstream_commit: str,
        contract_source_sha256: str,
        reused_sources: tuple[tuple[str, str, str], ...],
        new_sources: tuple[tuple[str, str, str], ...],
        dependency_sources: tuple[tuple[str, str], ...],
        external_distributions: tuple[
            tuple[
                str,
                str,
                str,
                tuple[tuple[str, str], ...],
                str,
                str | None,
            ],
            ...,
        ],
        numerical_time_distributions: tuple[
            tuple[
                str,
                str,
                str,
                str,
                str,
                tuple[tuple[str, str], ...],
                str,
                str | None,
            ],
            ...,
        ],
        dependency_closure_sha256: str,
        verification_sha256: str,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _VERIFIED_SENTINEL:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Verified source evidence can only be issued by the live verifier"
            )
        for name, value in (
            ("head commit", head_commit),
            ("upstream commit", upstream_commit),
            ("preregistration commit", preregistration_commit),
        ):
            if _COMMIT_RE.fullmatch(value) is None:
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    f"{name} is invalid"
                )
        for name, value in (
            ("contract source hash", contract_source_sha256),
            ("dependency closure hash", dependency_closure_sha256),
            ("verification hash", verification_sha256),
        ):
            if _SHA256_RE.fullmatch(value) is None:
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    f"{name} is invalid"
                )
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "origin_url", origin_url)
        object.__setattr__(
            self, "preregistration_commit", preregistration_commit
        )
        object.__setattr__(self, "head_commit", head_commit)
        object.__setattr__(self, "upstream_commit", upstream_commit)
        object.__setattr__(
            self, "contract_source_sha256", contract_source_sha256
        )
        object.__setattr__(self, "reused_sources", reused_sources)
        object.__setattr__(self, "new_sources", new_sources)
        object.__setattr__(self, "dependency_sources", dependency_sources)
        object.__setattr__(
            self, "external_distributions", external_distributions
        )
        object.__setattr__(
            self,
            "numerical_time_distributions",
            numerical_time_distributions,
        )
        object.__setattr__(
            self,
            "dependency_closure_sha256",
            dependency_closure_sha256,
        )
        object.__setattr__(self, "verification_sha256", verification_sha256)
        object.__setattr__(self, "_sentinel", _sentinel)


def is_verified_source_tree(value: Any) -> bool:
    return (
        type(value) is VerifiedSourceTree
        and getattr(value, "_sentinel", None) is _VERIFIED_SENTINEL
    )


def source_verification_material(
    value: VerifiedSourceTree,
) -> dict[str, Any]:
    if not is_verified_source_tree(value):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Source evidence was not issued by the live verifier"
        )
    return {
        "schema_version": SOURCE_VERIFICATION_SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "branch": value.branch,
        "origin_url": value.origin_url,
        "preregistration_commit": value.preregistration_commit,
        "head_commit": value.head_commit,
        "upstream_commit": value.upstream_commit,
        "contract_source": {
            "path": CONTRACT_SOURCE_PATH,
            "sha256": value.contract_source_sha256,
        },
        "reused_sources": [
            {"role": role, "path": path, "sha256": digest}
            for role, path, digest in value.reused_sources
        ],
        "new_sources": [
            {"role": role, "path": path, "sha256": digest}
            for role, path, digest in value.new_sources
        ],
        "dependency_sources": [
            {"path": path, "sha256": digest}
            for path, digest in value.dependency_sources
        ],
        "external_distributions": [
            {
                "import_name": import_name,
                "distribution": distribution,
                "version": version,
                "module_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in module_files
                ],
                "module_files_sha256": module_files_sha256,
                "direct_url_sha256": direct_url_sha256,
            }
            for (
                import_name,
                distribution,
                version,
                module_files,
                module_files_sha256,
                direct_url_sha256,
            ) in value.external_distributions
        ],
        "numerical_time_distributions": [
            {
                "import_name": import_name,
                "distribution": distribution,
                "version": version,
                "module_origin": {
                    "path": module_origin,
                    "sha256": module_origin_sha256,
                },
                "runtime_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in runtime_files
                ],
                "runtime_files_sha256": runtime_files_sha256,
                "direct_url_sha256": direct_url_sha256,
            }
            for (
                import_name,
                distribution,
                version,
                module_origin,
                module_origin_sha256,
                runtime_files,
                runtime_files_sha256,
                direct_url_sha256,
            ) in value.numerical_time_distributions
        ],
        "dependency_closure_sha256": value.dependency_closure_sha256,
    }


def verified_numerical_time_runtime_files(
    value: VerifiedSourceTree,
) -> tuple[tuple[str, str, Path], ...]:
    """Return rehashed absolute files permitted for numerical/time imports."""

    if not is_verified_source_tree(value):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Numerical/time runtime files require live source verification"
        )
    result: list[tuple[str, str, Path]] = []
    for (
        import_name,
        distribution_name,
        version,
        _module_origin,
        _module_origin_sha256,
        runtime_files,
        _runtime_files_sha256,
        _direct_url_sha256,
    ) in value.numerical_time_distributions:
        try:
            distribution = importlib.metadata.distribution(
                distribution_name
            )
        except importlib.metadata.PackageNotFoundError as exc:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Verified numerical/time distribution disappeared: "
                f"{import_name}"
            ) from exc
        if distribution.version != version:
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                "Verified numerical/time distribution version changed: "
                f"{import_name}"
            )
        for relative_path, expected_sha256 in runtime_files:
            try:
                resolved = Path(
                    distribution.locate_file(
                        PurePosixPath(relative_path)
                    )
                ).resolve(strict=True)
                payload = resolved.read_bytes()
            except OSError as exc:
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Verified numerical/time runtime file disappeared: "
                    f"{import_name}:{relative_path}"
                ) from exc
            if not hmac.compare_digest(
                hashlib.sha256(payload).hexdigest(),
                expected_sha256,
            ):
                raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                    "Verified numerical/time runtime file changed: "
                    f"{import_name}:{relative_path}"
                )
            result.append((import_name, relative_path, resolved))
    return tuple(result)


def verify_live_source_tree(repo_root: Path) -> VerifiedSourceTree:
    """Verify the exact pushed branch and every participating source byte."""

    root = _canonical_repo_root(repo_root)
    toplevel = Path(_git_text(root, "rev-parse", "--show-toplevel"))
    if toplevel != root:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Git toplevel differs from the requested repository root"
        )
    branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    if branch != BRANCH_NAME:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Live Git branch differs from the preregistered branch"
        )
    origin_url = _git_text(root, "remote", "get-url", "origin")
    if origin_url != EXPECTED_ORIGIN_URL:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Git origin differs from the preregistered repository"
        )
    head = _git_text(root, "rev-parse", "--verify", "HEAD")
    upstream_ref = f"refs/remotes/origin/{BRANCH_NAME}"
    upstream = _git_text(root, "rev-parse", "--verify", upstream_ref)
    if (
        _COMMIT_RE.fullmatch(head) is None
        or _COMMIT_RE.fullmatch(upstream) is None
        or not hmac.compare_digest(head, upstream)
    ):
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "HEAD does not equal the pinned origin branch"
        )
    try:
        status = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                PREREGISTRATION_COMMIT,
                head,
            ],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Preregistration ancestry verification could not run"
        ) from exc
    if status.returncode != 0:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Preregistration commit is not an ancestor of the implementation"
        )

    new_inventory = _inventory()
    tracked_overlay_sources = {
        line
        for line in _git_text(
            root,
            "ls-files",
            "--",
            "agent_benchmark/sec_gemma_online_risk_overlay_*.py",
        ).splitlines()
        if line
    }
    expected_overlay_sources = {
        CONTRACT_SOURCE_PATH,
        *new_inventory.values(),
    }
    if tracked_overlay_sources != expected_overlay_sources:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Tracked overlay source inventory is missing, broadened, or stale"
        )
    overlay_status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "agent_benchmark/sec_gemma_online_risk_overlay_*.py",
    )
    if overlay_status:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Overlay source inventory is dirty, untracked, or broadened"
        )
    if set(SOURCE_PINS) != set(SOURCE_PIN_FILES):
        raise RuntimeError("Frozen reused source inventory is inconsistent")
    participant_paths = {
        CONTRACT_SOURCE_PATH,
        *[SOURCE_PIN_FILES[role] for role in sorted(SOURCE_PIN_FILES)],
        *new_inventory.values(),
    }
    dependency_paths, external_imports = _dependency_closure(
        root,
        participant_paths=participant_paths,
        new_inventory=new_inventory,
    )
    forbidden_dependencies = (
        set(dependency_paths) & _FORBIDDEN_LOCAL_DEPENDENCY_PATHS
    )
    if forbidden_dependencies:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Dependency closure contains a forbidden market/network path: "
            f"{sorted(forbidden_dependencies)[0]}"
        )
    allowed_static_external = {
        *_ALLOWED_EXTERNAL_IMPORT_DISTRIBUTIONS,
        "numpy",
    }
    unlisted_external = external_imports - allowed_static_external
    if unlisted_external:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Production imports an unlisted external distribution: "
            f"{sorted(unlisted_external)[0]}"
        )
    all_paths = [*sorted(participant_paths), *dependency_paths]
    if len(set(all_paths)) != len(all_paths):
        raise RuntimeError("Contract, reused, and new source paths overlap")
    for index, path in enumerate(all_paths):
        all_paths[index] = _canonical_source_path(
            path, f"participating source {index}"
        )
    dirty = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *all_paths,
    )
    if dirty:
        raise SecGemmaOnlineRiskOverlaySourceVerificationError(
            "Participating source tree is dirty or untracked"
        )

    observed_hashes: dict[str, str] = {}
    for relative_path in all_paths:
        worktree = _verify_real_source(root, relative_path)
        observed_hashes[relative_path] = _verify_tracked_blob(
            root, relative_path, worktree
        )
    for role in sorted(SOURCE_PINS):
        expected = SOURCE_PINS[role]
        observed = observed_hashes[SOURCE_PIN_FILES[role]]
        if not hmac.compare_digest(observed, expected):
            raise SecGemmaOnlineRiskOverlaySourceVerificationError(
                f"Frozen reused source pin changed for role {role}"
            )

    reused = tuple(
        (
            role,
            SOURCE_PIN_FILES[role],
            observed_hashes[SOURCE_PIN_FILES[role]],
        )
        for role in sorted(SOURCE_PINS)
    )
    new = tuple(
        (role, path, observed_hashes[path])
        for role, path in sorted(new_inventory.items())
    )
    dependencies = tuple(
        (path, observed_hashes[path]) for path in dependency_paths
    )
    external_distributions = _external_distribution_identities(
        external_imports
        & set(_ALLOWED_EXTERNAL_IMPORT_DISTRIBUTIONS)
    )
    numerical_time_distributions = (
        _numerical_time_distribution_identities(
            root,
            external_imports,
        )
    )
    dependency_material = {
        "dependency_sources": [
            {"path": path, "sha256": digest}
            for path, digest in dependencies
        ],
        "external_distributions": [
            {
                "import_name": import_name,
                "distribution": distribution,
                "version": version,
                "module_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in module_files
                ],
                "module_files_sha256": module_files_sha256,
                "direct_url_sha256": direct_url_sha256,
            }
            for (
                import_name,
                distribution,
                version,
                module_files,
                module_files_sha256,
                direct_url_sha256,
            ) in external_distributions
        ],
        "numerical_time_distributions": [
            {
                "import_name": import_name,
                "distribution": distribution,
                "version": version,
                "module_origin": {
                    "path": module_origin,
                    "sha256": module_origin_sha256,
                },
                "runtime_files": [
                    {"path": path, "sha256": digest}
                    for path, digest in runtime_files
                ],
                "runtime_files_sha256": runtime_files_sha256,
                "direct_url_sha256": direct_url_sha256,
            }
            for (
                import_name,
                distribution,
                version,
                module_origin,
                module_origin_sha256,
                runtime_files,
                runtime_files_sha256,
                direct_url_sha256,
            ) in numerical_time_distributions
        ],
    }
    dependency_closure_sha256 = canonical_sha256(dependency_material)
    material = {
        "schema_version": SOURCE_VERIFICATION_SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "branch": branch,
        "origin_url": origin_url,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "head_commit": head,
        "upstream_commit": upstream,
        "contract_source": {
            "path": CONTRACT_SOURCE_PATH,
            "sha256": observed_hashes[CONTRACT_SOURCE_PATH],
        },
        "reused_sources": [
            {"role": role, "path": path, "sha256": digest}
            for role, path, digest in reused
        ],
        "new_sources": [
            {"role": role, "path": path, "sha256": digest}
            for role, path, digest in new
        ],
        **dependency_material,
        "dependency_closure_sha256": dependency_closure_sha256,
    }
    verification_hash = canonical_sha256(material)
    return VerifiedSourceTree(
        repo_root=root,
        branch=branch,
        origin_url=origin_url,
        preregistration_commit=PREREGISTRATION_COMMIT,
        head_commit=head,
        upstream_commit=upstream,
        contract_source_sha256=observed_hashes[CONTRACT_SOURCE_PATH],
        reused_sources=reused,
        new_sources=new,
        dependency_sources=dependencies,
        external_distributions=external_distributions,
        numerical_time_distributions=numerical_time_distributions,
        dependency_closure_sha256=dependency_closure_sha256,
        verification_sha256=verification_hash,
        _sentinel=_VERIFIED_SENTINEL,
    )


__all__ = [
    "CONTRACT_SOURCE_PATH",
    "EXPECTED_ORIGIN_URL",
    "NUMERICAL_TIME_IMPORT_ROOTS",
    "PREREGISTRATION_COMMIT",
    "REQUIRED_NEW_SOURCE_PATHS",
    "SOURCE_VERIFICATION_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlaySourceVerificationError",
    "VerifiedSourceTree",
    "is_verified_source_tree",
    "load_allowed_requests",
    "source_verification_material",
    "verified_numerical_time_runtime_files",
    "verify_live_source_tree",
]
