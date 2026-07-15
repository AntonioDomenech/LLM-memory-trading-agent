from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace
from typing import Any, Mapping

import pytest

from agent_benchmark import (
    contextual_expert_aggregation_audit_bootstrap as bootstrap,
)


def _git(repo: Path, *args: str) -> str:
    value = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return value.stdout.strip()


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="\n")


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.name", "Audit Bootstrap Test")
    _git(repo, "config", "user.email", "audit-bootstrap@example.invalid")
    _write(repo / ".gitattributes", "*.py text eol=lf\n")
    _write(repo / ".gitignore", "__pycache__/\n*.py[cod]\n*.pyd\n*.so\n")
    _write(repo / "requirements.txt", "pytest\n")
    _write(repo / "agent_benchmark/__init__.py", "\n")
    _write(
        repo
        / "agent_benchmark/contextual_expert_aggregation_audit_bootstrap.py",
        "# synthetic tracked audit bootstrap\n",
    )
    _write(
        repo / "agent_benchmark/contextual_expert_aggregation_audit_runner.py",
        "def run_stage(stage): return {'stage_pass': True}\n",
    )
    _write(
        repo / "agent_benchmark/contextual_expert_aggregation_audit_verifier.py",
        "def verify_stage(stage): return {'verified': True}\n",
    )
    _write(repo / "agent_benchmark/model.py", "VALUE = 1\n")
    _write(repo / "agent_benchmark/native.pyd", b"synthetic-native")
    _write(repo / "root_tool.py", "ROOT = True\n")
    _write(repo / "root_package/__init__.py", "\n")
    _write(repo / "root_package/clean.py", "CLEAN = True\n")
    _write(repo / "not_a_package/ignored.txt", "not executable\n")
    _write(repo / "e/aapl_spy_qqq_through_2023.csv", "never,opened\n")
    _git(repo, "add", "--all")
    _git(repo, "add", "--force", "agent_benchmark/native.pyd")
    _git(repo, "commit", "--quiet", "-m", "synthetic audit bootstrap")
    return repo


def _guard(
    repo: Path,
    *,
    argv: tuple[str, ...] = ("stage", "audit"),
    cwd: Path | None = None,
    isolated: int = 1,
    dont_write_bytecode_flag: int = 1,
    dont_write_bytecode_value: bool = True,
    optimize: int = 0,
    orig_argv: tuple[str, ...] | None = None,
    preimport_sys_path: tuple[str, ...] | None = None,
    dispatch: bootstrap.DispatchCallable | None = None,
) -> tuple[Mapping[str, Any], int]:
    selected_dispatch = dispatch or (
        lambda operation, stage: (
            {"operation": operation, "stage": stage, "stage_pass": True},
            0,
        )
    )
    return bootstrap._guarded_main(
        argv,
        repo_root=repo.resolve(),
        cwd=(cwd or repo).resolve(),
        isolated=isolated,
        dont_write_bytecode_flag=dont_write_bytecode_flag,
        dont_write_bytecode_value=dont_write_bytecode_value,
        optimize=optimize,
        orig_argv=(
            orig_argv
            if orig_argv is not None
            else (
                "python",
                "-I",
                "-B",
                "agent_benchmark/"
                "contextual_expert_aggregation_audit_bootstrap.py",
                *argv,
            )
        ),
        warn_options=(),
        xoptions={},
        preimport_sys_path=(
            preimport_sys_path
            if preimport_sys_path is not None
            else (str(repo.parent / "outside-stdlib"),)
        ),
        preloaded_agent_modules=(),
        dispatch=selected_dispatch,
    )


def _assert_surface_violation(
    exc: pytest.ExceptionInfo[bootstrap.BootstrapSecurityError],
    key: str,
    path: str,
) -> None:
    assert exc.value.code == "unsafe_execution_surface"
    assert path in exc.value.details["violations"][key]


def test_bootstrap_module_loads_only_standard_library_modules() -> None:
    source_path = Path(bootstrap.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


def test_valid_guard_attests_candidates_and_never_reads_market_csv(
    clean_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_bytes = Path.read_bytes
    observed: dict[str, object] = {}

    def guarded_read_bytes(path: Path) -> bytes:
        if path.suffix.lower() == ".csv":
            raise AssertionError("bootstrap opened a market CSV")
        return original_read_bytes(path)

    def dispatch(operation: str, stage: str) -> tuple[Mapping[str, Any], int]:
        observed["operation"] = operation
        observed["stage"] = stage
        observed["root_first"] = Path(sys.path[0]).resolve()
        observed["git_index"] = os.environ.get("GIT_INDEX_FILE")
        observed["python_path"] = os.environ.get("PYTHONPATH")
        observed["active"] = isinstance(
            getattr(sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE),
            MappingProxyType,
        )
        return {"stage_pass": True}, 0

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    payload, exit_code = _guard(clean_repo, dispatch=dispatch)

    assert exit_code == 0
    assert observed == {
        "operation": "stage",
        "stage": "audit",
        "root_first": clean_repo.resolve(),
        "git_index": None,
        "python_path": None,
        "active": True,
    }
    assert not hasattr(sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE)
    assert isinstance(payload, MappingProxyType)
    attestation = payload["bootstrap_attestation"]
    assert attestation["invocation"] == {"operation": "stage", "stage": "audit"}
    assert attestation["checks"][
        "all_import_candidates_head_index_worktree_equal"
    ]
    candidates = attestation["execution_surface"]["candidate_paths"]
    assert "root_tool.py" in candidates
    assert "root_package/clean.py" in candidates
    assert "agent_benchmark/native.pyd" in candidates
    assert "e/aapl_spy_qqq_through_2023.csv" not in candidates


@pytest.mark.parametrize(
    "argv",
    (
        (),
        ("stage",),
        ("stage", "development"),
        ("verify", "confirmation"),
        ("run", "audit"),
        ("stage", "audit", "extra"),
    ),
)
def test_only_exact_audit_commands_are_allowed(
    clean_repo: Path,
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo, argv=argv)
    assert exc.value.code == "unsafe_argv"


@pytest.mark.parametrize(
    ("isolated", "bytecode_flag", "bytecode_value", "code"),
    (
        (0, 1, True, "isolated_mode_required"),
        (1, 0, True, "bytecode_writes_disabled_required"),
        (1, 1, False, "bytecode_writes_disabled_required"),
    ),
)
def test_python_i_and_b_flags_are_required(
    clean_repo: Path,
    isolated: int,
    bytecode_flag: int,
    bytecode_value: bool,
    code: str,
) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(
            clean_repo,
            isolated=isolated,
            dont_write_bytecode_flag=bytecode_flag,
            dont_write_bytecode_value=bytecode_value,
        )
    assert exc.value.code == code


@pytest.mark.parametrize(
    ("relative", "content"),
    (
        ("agent_benchmark/foreign.py", "FOREIGN = True\n"),
        ("agent_benchmark/foreign.pyd", b"foreign-native"),
        ("json.py", "SHADOW = True\n"),
        ("namespace/foreign.py", "FOREIGN = True\n"),
    ),
)
def test_untracked_python_and_native_candidates_fail_closed(
    clean_repo: Path,
    relative: str,
    content: str | bytes,
) -> None:
    _write(clean_repo / relative, content)
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)
    _assert_surface_violation(exc, "untracked_or_index_missing", relative)


@pytest.mark.parametrize(
    ("relative", "content"),
    (
        ("agent_benchmark/model.py", "VALUE = 2\n"),
        ("agent_benchmark/native.pyd", b"changed-native"),
    ),
)
def test_modified_tracked_candidates_fail_closed(
    clean_repo: Path,
    relative: str,
    content: str | bytes,
) -> None:
    _write(clean_repo / relative, content)
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)
    _assert_surface_violation(exc, "index_worktree_divergence", relative)


def test_staged_candidate_divergent_from_head_fails_closed(clean_repo: Path) -> None:
    relative = "agent_benchmark/model.py"
    _write(clean_repo / relative, "VALUE = 2\n")
    _git(clean_repo, "add", relative)
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)
    _assert_surface_violation(exc, "head_index_divergence", relative)


def test_deleted_tracked_candidate_fails_closed(clean_repo: Path) -> None:
    relative = "agent_benchmark/model.py"
    (clean_repo / relative).unlink()
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)
    _assert_surface_violation(exc, "worktree_missing", relative)


def test_dispatch_environment_and_sys_path_are_restored_after_error(
    clean_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_INDEX_FILE", "foreign-index")
    monkeypatch.setenv("PYTHONPATH", "foreign-python-path")
    original_path = list(sys.path)

    def failing_dispatch(
        operation: str,
        stage: str,
    ) -> tuple[Mapping[str, Any], int]:
        assert "GIT_INDEX_FILE" not in os.environ
        assert "PYTHONPATH" not in os.environ
        assert Path(sys.path[0]).resolve() == clean_repo.resolve()
        raise RuntimeError("synthetic dispatch failure")

    with pytest.raises(RuntimeError, match="synthetic dispatch failure"):
        _guard(clean_repo, dispatch=failing_dispatch)
    assert os.environ["GIT_INDEX_FILE"] == "foreign-index"
    assert os.environ["PYTHONPATH"] == "foreign-python-path"
    assert sys.path == original_path
    assert not hasattr(sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE)


def test_preimport_repository_path_fails_before_dispatch(clean_repo: Path) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo, preimport_sys_path=(str(clean_repo),))
    assert exc.value.code == "unsafe_preimport_sys_path"


def test_runner_and_verifier_require_an_active_attestation(
    clean_repo: Path,
) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        bootstrap.require_active_attestation(
            operation="stage",
            stage="audit",
            repo_root=clean_repo,
        )
    assert exc.value.code == "missing_active_attestation"


def test_import_dispatch_has_only_the_two_frozen_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_import(module_name: str) -> SimpleNamespace:
        calls.append(("import", module_name))
        if module_name.endswith("_runner"):
            return SimpleNamespace(
                run_stage=lambda stage: {"stage_pass": True, "stage": stage}
            )
        return SimpleNamespace(
            verify_stage=lambda stage: {"verified": True, "stage": stage}
        )

    monkeypatch.setattr(importlib, "import_module", fake_import)
    stage_result, stage_exit = bootstrap._import_dispatch("stage", "audit")
    verify_result, verify_exit = bootstrap._import_dispatch("verify", "audit")

    assert stage_result == {"stage_pass": True, "stage": "audit"}
    assert verify_result == {"verified": True, "stage": "audit"}
    assert stage_exit == verify_exit == 0
    assert calls == [
        (
            "import",
            "agent_benchmark.contextual_expert_aggregation_audit_runner",
        ),
        (
            "import",
            "agent_benchmark.contextual_expert_aggregation_audit_verifier",
        ),
    ]


def test_attestation_finishes_before_fixed_runner_import(
    clean_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_attest = bootstrap._attest_preimport_environment

    def recording_attest(**kwargs: object) -> Mapping[str, Any]:
        events.append("attest")
        return original_attest(**kwargs)  # type: ignore[arg-type]

    def fake_import(module_name: str) -> SimpleNamespace:
        events.append(f"import:{module_name}")
        return SimpleNamespace(
            run_stage=lambda stage: {"stage": stage, "stage_pass": True}
        )

    monkeypatch.setattr(bootstrap, "_attest_preimport_environment", recording_attest)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    _guard(clean_repo, dispatch=bootstrap._import_dispatch)

    assert events == [
        "attest",
        "import:agent_benchmark.contextual_expert_aggregation_audit_runner",
    ]
