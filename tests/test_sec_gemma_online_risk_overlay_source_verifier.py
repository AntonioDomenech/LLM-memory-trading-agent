from __future__ import annotations

import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import sys

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    BRANCH_NAME,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    _DEPENDENCY_ROOT_REUSED_ROLES,
    _dependency_closure,
    _external_distribution_identities,
    _external_distribution_identity,
    _numerical_time_distribution_identity,
    EXPECTED_ORIGIN_URL,
    NUMERICAL_TIME_IMPORT_ROOTS,
    PREREGISTRATION_COMMIT,
    REQUIRED_NEW_SOURCE_PATHS,
    SOURCE_PIN_FILES,
    SecGemmaOnlineRiskOverlaySourceVerificationError,
    load_allowed_requests,
    source_verification_material,
    verified_numerical_time_runtime_files,
    verify_live_source_tree,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    canonical_sha256,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=30,
        text=True,
    )
    return result.stdout.strip()


def _build_verified_repo(destination: Path) -> Path:
    destination.mkdir()
    shutil.copytree(
        WORKSPACE_ROOT / ".git",
        destination / ".git",
        ignore=shutil.ignore_patterns("turn-diffs"),
    )
    _git(destination, "config", "core.longpaths", "true")
    _git(destination, "config", "core.autocrlf", "false")
    _git(destination, "reset", "--quiet", "--hard", PREREGISTRATION_COMMIT)
    _git(destination, "checkout", "--quiet", "--detach", PREREGISTRATION_COMMIT)
    _git(destination, "switch", "--quiet", "-C", BRANCH_NAME)
    for relative in REQUIRED_NEW_SOURCE_PATHS.values():
        source = WORKSPACE_ROOT / Path(relative)
        target = destination / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, target)
        else:
            target.write_text('"""Test replay placeholder."""\n', encoding="utf-8")
    _git(destination, "add", "--", *REQUIRED_NEW_SOURCE_PATHS.values())
    _git(destination, "config", "user.name", "Verifier Test")
    _git(destination, "config", "user.email", "verifier@example.invalid")
    _git(destination, "commit", "--quiet", "-m", "test implementation")
    _git(destination, "remote", "set-url", "origin", EXPECTED_ORIGIN_URL)
    head = _git(destination, "rev-parse", "HEAD")
    _git(
        destination,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        head,
    )
    return destination.resolve()


@pytest.fixture
def verified_repo(tmp_path: Path) -> Path:
    return _build_verified_repo(tmp_path / "repo")


def test_live_verifier_binds_exact_branch_git_and_source_inventory(
    verified_repo: Path,
) -> None:
    from agent_benchmark.sec_gemma_online_risk_overlay_production import (
        issue_verified_production_authority,
    )

    verified = verify_live_source_tree(verified_repo)
    material = source_verification_material(verified)
    authority = issue_verified_production_authority(verified)

    assert verified.repo_root == verified_repo
    assert verified.head_commit == verified.upstream_commit
    assert verified.preregistration_commit == PREREGISTRATION_COMMIT
    assert material["origin_url"] == EXPECTED_ORIGIN_URL
    assert {
        item["role"]: item["path"] for item in material["new_sources"]
    } == dict(REQUIRED_NEW_SOURCE_PATHS)
    assert len(material["dependency_closure_sha256"]) == 64
    assert material["dependency_sources"] == sorted(
        material["dependency_sources"], key=lambda item: item["path"]
    )
    assert {
        "agent_benchmark/benchmark_engine.py",
        "agent_benchmark/deterministic_aapl.py",
        "agent_benchmark/market_data.py",
    }.isdisjoint(
        item["path"] for item in material["dependency_sources"]
    )
    assert isinstance(material["external_distributions"], list)
    assert {
        item["import_name"]
        for item in material["numerical_time_distributions"]
    } == set(NUMERICAL_TIME_IMPORT_ROOTS)
    assert {
        item["import_name"]
        for item in material["external_distributions"]
    } == {
        "requests",
        "urllib3",
        "certifi",
        "charset_normalizer",
        "idna",
    }
    assert authority._payload[
        "numerical_time_distributions_sha256"
    ] == canonical_sha256(material["numerical_time_distributions"])


def test_dirty_or_missing_required_source_fails(
    verified_repo: Path,
) -> None:
    path = verified_repo / next(iter(REQUIRED_NEW_SOURCE_PATHS.values()))
    path.write_bytes(path.read_bytes() + b"\n# dirty\n")
    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="dirty|untracked",
    ):
        verify_live_source_tree(verified_repo)

    _git(verified_repo, "restore", "--", path.relative_to(verified_repo).as_posix())
    path.unlink()
    with pytest.raises(SecGemmaOnlineRiskOverlaySourceVerificationError):
        verify_live_source_tree(verified_repo)


def test_wrong_branch_or_origin_fails(verified_repo: Path) -> None:
    _git(verified_repo, "branch", "-m", "codex/wrong-branch")
    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="branch",
    ):
        verify_live_source_tree(verified_repo)

    _git(verified_repo, "branch", "-m", BRANCH_NAME)
    _git(verified_repo, "remote", "set-url", "origin", "https://invalid/repo.git")
    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="origin",
    ):
        verify_live_source_tree(verified_repo)


def test_wrong_upstream_fails(verified_repo: Path) -> None:
    _git(
        verified_repo,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        PREREGISTRATION_COMMIT,
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="HEAD",
    ):
        verify_live_source_tree(verified_repo)


def test_extra_tracked_overlay_source_fails_exact_inventory(
    verified_repo: Path,
) -> None:
    extra = (
        verified_repo
        / "agent_benchmark"
        / "sec_gemma_online_risk_overlay_unbound.py"
    )
    extra.write_text('"""Unbound production source."""\n', encoding="utf-8")
    _git(verified_repo, "add", "--", extra.relative_to(verified_repo).as_posix())
    _git(verified_repo, "commit", "--quiet", "-m", "add unbound source")
    head = _git(verified_repo, "rev-parse", "HEAD")
    _git(
        verified_repo,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        head,
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="inventory",
    ):
        verify_live_source_tree(verified_repo)


def test_extra_untracked_overlay_source_fails_exact_inventory(
    verified_repo: Path,
) -> None:
    extra = (
        verified_repo
        / "agent_benchmark"
        / "sec_gemma_online_risk_overlay_unbound.py"
    )
    extra.write_text('"""Unbound production source."""\n', encoding="utf-8")
    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="inventory",
    ):
        verify_live_source_tree(verified_repo)


def test_local_dependency_is_mechanically_added_to_closure(
    verified_repo: Path,
) -> None:
    production = (
        verified_repo
        / REQUIRED_NEW_SOURCE_PATHS["production"]
    )
    production.write_text(
        "import agent_benchmark.unbound_dependency\n",
        encoding="utf-8",
    )
    dependency = verified_repo / "agent_benchmark" / "unbound_dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    _git(
        verified_repo,
        "add",
        "--",
        production.relative_to(verified_repo).as_posix(),
        dependency.relative_to(verified_repo).as_posix(),
    )
    _git(verified_repo, "commit", "--quiet", "-m", "add dependency")
    head = _git(verified_repo, "rev-parse", "HEAD")
    _git(
        verified_repo,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        head,
    )

    verified = verify_live_source_tree(verified_repo)
    material = source_verification_material(verified)

    assert "agent_benchmark/unbound_dependency.py" in {
        item["path"] for item in material["dependency_sources"]
    }


def test_numpy_import_outside_exact_pinned_learner_fails_closed(
    verified_repo: Path,
) -> None:
    production = verified_repo / REQUIRED_NEW_SOURCE_PATHS["production"]
    production.write_text("import numpy\n", encoding="utf-8")
    _git(
        verified_repo,
        "add",
        "--",
        production.relative_to(verified_repo).as_posix(),
    )
    _git(verified_repo, "commit", "--quiet", "-m", "import unlisted package")
    head = _git(verified_repo, "rev-parse", "HEAD")
    _git(
        verified_repo,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        head,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="outside its exact source-pinned adapter",
    ):
        verify_live_source_tree(verified_repo)


def test_yfinance_import_fails_as_forbidden_network_path(
    verified_repo: Path,
) -> None:
    production = verified_repo / REQUIRED_NEW_SOURCE_PATHS["production"]
    production.write_text("import yfinance\n", encoding="utf-8")
    _git(
        verified_repo,
        "add",
        "--",
        production.relative_to(verified_repo).as_posix(),
    )
    _git(verified_repo, "commit", "--quiet", "-m", "import yfinance")
    head = _git(verified_repo, "rev-parse", "HEAD")
    _git(
        verified_repo,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        head,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="forbidden network-capable distribution",
    ):
        verify_live_source_tree(verified_repo)


@pytest.mark.parametrize(
    "module_name",
    (
        "benchmark_engine",
        "deterministic_aapl",
        "market_data",
    ),
)
def test_forbidden_market_dependency_paths_fail_closed(
    verified_repo: Path,
    module_name: str,
) -> None:
    production = verified_repo / REQUIRED_NEW_SOURCE_PATHS["production"]
    production.write_text(
        f"import agent_benchmark.{module_name}\n",
        encoding="utf-8",
    )
    _git(
        verified_repo,
        "add",
        "--",
        production.relative_to(verified_repo).as_posix(),
    )
    _git(
        verified_repo,
        "commit",
        "--quiet",
        "-m",
        f"import forbidden {module_name}",
    )
    head = _git(verified_repo, "rev-parse", "HEAD")
    _git(
        verified_repo,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        head,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="forbidden market/network path",
    ):
        verify_live_source_tree(verified_repo)


def test_transitive_numpy_import_outside_pinned_learner_fails_without_git(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "closure").resolve()
    (root / "agent_benchmark").mkdir(parents=True)
    inventory = {
        role: f"agent_benchmark/{role}.py"
        for role in REQUIRED_NEW_SOURCE_PATHS
    }
    for relative in {
        *inventory.values(),
        *(
            SOURCE_PIN_FILES[role]
            for role in _DEPENDENCY_ROOT_REUSED_ROLES
        ),
    }:
        path = root / PurePosixPath(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Closure fixture."""\n', encoding="utf-8")
    production = root / PurePosixPath(inventory["production"])
    production.write_text(
        "import agent_benchmark.transitive_dependency\n",
        encoding="utf-8",
    )
    transitive = root / "agent_benchmark" / "transitive_dependency.py"
    transitive.write_text("import numpy\n", encoding="utf-8")
    participant_paths = {
        *inventory.values(),
        *(
            SOURCE_PIN_FILES[role]
            for role in _DEPENDENCY_ROOT_REUSED_ROLES
        ),
    }

    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="outside its exact source-pinned adapter",
    ):
        _dependency_closure(
            root,
            participant_paths=participant_paths,
            new_inventory=inventory,
        )


@pytest.mark.parametrize(
    ("import_name", "distribution_name", "required_module"),
    [
        ("requests", "requests", "requests/adapters.py"),
        ("urllib3", "urllib3", "urllib3/connectionpool.py"),
        ("certifi", "certifi", "certifi/core.py"),
        (
            "charset_normalizer",
            "charset-normalizer",
            "charset_normalizer/api.py",
        ),
        ("idna", "idna", "idna/core.py"),
    ],
)
def test_external_identity_binds_every_distribution_module_file(
    import_name: str,
    distribution_name: str,
    required_module: str,
) -> None:
    identity = _external_distribution_identity(import_name)
    (
        observed_import,
        observed_distribution,
        version,
        module_files,
        module_files_sha256,
        _direct_url_sha256,
    ) = identity
    distribution = importlib.metadata.distribution(distribution_name)
    suffixes = {
        *importlib.machinery.SOURCE_SUFFIXES,
        *importlib.machinery.EXTENSION_SUFFIXES,
    }
    expected_paths = sorted(
        normalized
        for raw in distribution.files or ()
        if (
            (normalized := str(raw).replace("\\", "/"))
            and not PurePosixPath(normalized).is_absolute()
            and all(
                part not in {"", ".", ".."}
                for part in PurePosixPath(normalized).parts
            )
            and any(normalized.endswith(suffix) for suffix in suffixes)
        )
    )

    assert observed_import == import_name
    assert observed_distribution == distribution_name
    assert version == distribution.version
    assert [path for path, _digest in module_files] == expected_paths
    assert required_module in expected_paths
    assert module_files_sha256 == canonical_sha256(
        [
            {"path": path, "sha256": digest}
            for path, digest in module_files
        ]
    )


@pytest.mark.parametrize(
    ("import_name", "required_runtime_path"),
    [
        ("numpy", "numpy/__init__.py"),
        ("tzdata", "tzdata/zoneinfo/America/New_York"),
    ],
)
def test_numerical_time_identity_binds_origin_native_and_data_files(
    import_name: str,
    required_runtime_path: str,
) -> None:
    (
        observed_import,
        distribution_name,
        version,
        module_origin,
        module_origin_sha256,
        runtime_files,
        runtime_files_sha256,
        _direct_url_sha256,
    ) = _numerical_time_distribution_identity(import_name)
    distribution = importlib.metadata.distribution(distribution_name)
    runtime_map = dict(runtime_files)

    assert observed_import == import_name
    assert version == distribution.version
    assert module_origin in runtime_map
    assert runtime_map[module_origin] == module_origin_sha256
    assert required_runtime_path in runtime_map
    assert runtime_files_sha256 == canonical_sha256(
        [
            {"path": path, "sha256": digest}
            for path, digest in runtime_files
        ]
    )
    if import_name == "numpy":
        assert any(
            path.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
            or path.lower().endswith(".dll")
            for path in runtime_map
        )


def test_verified_numerical_time_runtime_files_are_exactly_rehashed(
    verified_repo: Path,
) -> None:
    verified = verify_live_source_tree(verified_repo)
    files = verified_numerical_time_runtime_files(verified)

    assert {import_name for import_name, _path, _absolute in files} == set(
        NUMERICAL_TIME_IMPORT_ROOTS
    )
    assert any(
        import_name == "tzdata"
        and relative_path == "tzdata/zoneinfo/America/New_York"
        and absolute_path.is_file()
        for import_name, relative_path, absolute_path in files
    )


def test_requests_closure_binds_exact_allowed_distribution_set() -> None:
    identities = _external_distribution_identities({"requests", "urllib3"})

    assert [item[0] for item in identities] == [
        "certifi",
        "charset_normalizer",
        "idna",
        "requests",
        "urllib3",
    ]


def _isolated_python(
    script: str,
    *,
    extra_pythonpath: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    if extra_pythonpath is not None:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(extra_pythonpath)
            if not existing
            else f"{extra_pythonpath}{os.pathsep}{existing}"
        )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=WORKSPACE_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
        text=True,
    )


def test_allowed_requests_loader_fresh_process_uses_exact_five_distributions(
) -> None:
    result = _isolated_python(
        """
import importlib.metadata
import json
import sys
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import load_allowed_requests
before = set(sys.modules)
requests = load_allowed_requests()
distribution_map = importlib.metadata.packages_distributions()
owners = set()
for module_name in set(sys.modules) - before:
    root = module_name.split(".", 1)[0]
    if root in sys.stdlib_module_names or root in {"__future__", "agent_benchmark"}:
        continue
    owners.update(distribution_map.get(root, ()))
print(json.dumps({
    "owners": sorted(owners),
    "resolver": requests.compat.chardet.__name__,
    "top_level_chardet": sorted(
        name for name in sys.modules
        if name == "chardet" or name.startswith("chardet.")
    ),
}, sort_keys=True))
"""
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "owners": [
            "certifi",
            "charset-normalizer",
            "idna",
            "requests",
            "urllib3",
        ],
        "resolver": "charset_normalizer",
        "top_level_chardet": [],
    }


def test_allowed_requests_loader_blocks_optional_shadow_modules(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "optional-shadow-executed"
    shadow_source = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
    )
    for name in (
        "brotli",
        "brotlicffi",
        "chardet",
        "h2",
        "simplejson",
        "socks",
        "zstandard",
    ):
        (tmp_path / f"{name}.py").write_text(shadow_source, encoding="utf-8")

    result = _isolated_python(
        """
import sys
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import load_allowed_requests
requests = load_allowed_requests()
assert requests.compat.chardet.__name__ == "charset_normalizer"
assert not any(
    name == "chardet" or name.startswith("chardet.")
    for name in sys.modules
)
""",
        extra_pythonpath=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("shadowed_name", ["requests", "charset_normalizer"])
def test_allowed_requests_loader_rejects_allowed_package_shadow_before_execution(
    tmp_path: Path,
    shadowed_name: str,
) -> None:
    marker = tmp_path / f"{shadowed_name}-executed"
    package = tmp_path / shadowed_name
    package.mkdir()
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    result = _isolated_python(
        """
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import load_allowed_requests
load_allowed_requests()
""",
        extra_pythonpath=tmp_path,
    )

    assert result.returncode != 0
    assert "shadowed or redirected" in result.stderr
    assert not marker.exists()


def test_allowed_requests_loader_isolates_preloaded_chardet() -> None:
    result = _isolated_python(
        """
import sys
import chardet
ambient_chardet = chardet
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import load_allowed_requests
requests = load_allowed_requests()
assert ambient_chardet.__name__ == "chardet"
assert requests.compat.chardet.__name__ == "charset_normalizer"
assert sys.modules["requests"] is requests
assert not any(
    name == "chardet" or name.startswith("chardet.")
    for name in sys.modules
)
"""
    )

    assert result.returncode == 0, result.stderr


def test_allowed_requests_loader_replaces_preloaded_wrong_requests() -> None:
    result = _isolated_python(
        """
import sys
import requests
assert requests.compat.chardet.__name__ == "chardet"
ambient_requests = requests
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import load_allowed_requests
verified_requests = load_allowed_requests()
assert verified_requests is not ambient_requests
assert sys.modules["requests"] is verified_requests
assert verified_requests.compat.chardet.__name__ == "charset_normalizer"
assert not any(
    name == "chardet" or name.startswith("chardet.")
    for name in sys.modules
)
"""
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module_name",
    [
        "agent_benchmark.sec_gemma_online_risk_overlay_acquisition",
        "agent_benchmark.sec_gemma_online_risk_overlay_production",
    ],
)
def test_direct_production_module_imports_use_allowed_requests(
    module_name: str,
) -> None:
    result = _isolated_python(
        f"""
import importlib
import sys
module = importlib.import_module({module_name!r})
assert module.requests.compat.chardet.__name__ == "charset_normalizer"
assert not any(
    name == "chardet" or name.startswith("chardet.")
    for name in sys.modules
)
"""
    )

    assert result.returncode == 0, result.stderr


def test_allowed_requests_loader_is_idempotent() -> None:
    first = load_allowed_requests()
    import chardet

    second = load_allowed_requests()

    assert first is second
    assert first.compat.chardet.__name__ == "charset_normalizer"
    assert chardet.__name__ == "chardet"


@pytest.mark.parametrize(
    "module_name",
    [
        "agent_benchmark.sec_gemma_online_risk_overlay_acquisition",
        "agent_benchmark.sec_gemma_online_risk_overlay_production",
    ],
)
def test_production_module_import_is_independent_of_ambient_requests_order(
    module_name: str,
) -> None:
    result = _isolated_python(
        f"""
import importlib
import requests as ambient_requests
assert ambient_requests.compat.chardet.__name__ == "chardet"
module = importlib.import_module({module_name!r})
assert module.requests is not ambient_requests
assert module.requests.compat.chardet.__name__ == "charset_normalizer"
"""
    )

    assert result.returncode == 0, result.stderr


def test_preregistration_must_be_an_ancestor(
    verified_repo: Path,
) -> None:
    paths = list(REQUIRED_NEW_SOURCE_PATHS.values())
    _git(verified_repo, "checkout", "--orphan", "unrelated")
    _git(verified_repo, "add", "-A")
    _git(verified_repo, "commit", "--quiet", "-m", "unrelated implementation")
    _git(verified_repo, "branch", "-M", BRANCH_NAME)
    head = _git(verified_repo, "rev-parse", "HEAD")
    _git(
        verified_repo,
        "update-ref",
        f"refs/remotes/origin/{BRANCH_NAME}",
        head,
    )
    assert paths
    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="ancestor",
    ):
        verify_live_source_tree(verified_repo)


def test_hardlinked_source_and_nonroot_path_fail(
    verified_repo: Path,
) -> None:
    relative = next(iter(REQUIRED_NEW_SOURCE_PATHS.values()))
    source = verified_repo / relative
    target = verified_repo / "hardlink-target.py"
    shutil.copy2(source, target)
    source.unlink()
    os.link(target, source)

    with pytest.raises(
        SecGemmaOnlineRiskOverlaySourceVerificationError,
        match="hard-linked",
    ):
        verify_live_source_tree(verified_repo)
    with pytest.raises(SecGemmaOnlineRiskOverlaySourceVerificationError):
        verify_live_source_tree(verified_repo / "agent_benchmark")
