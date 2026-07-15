from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping

import pytest

from agent_benchmark import contextual_expert_aggregation_bootstrap as bootstrap


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


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
    _git(repo, "config", "user.name", "Bootstrap Test")
    _git(repo, "config", "user.email", "bootstrap@example.invalid")
    _write(repo / ".gitignore", "__pycache__/\n*.py[cod]\n*.pyd\n*.so\n")
    _write(repo / "agent_benchmark/__init__.py", "\n")
    _write(
        repo / "agent_benchmark/contextual_expert_aggregation_bootstrap.py",
        "# synthetic tracked bootstrap\n",
    )
    _write(repo / "agent_benchmark/model.py", "VALUE = 1\n")
    _write(repo / "root_tool.py", "ROOT = True\n")
    _write(repo / "root_package/__init__.py", "\n")
    _write(repo / "root_package/clean.py", "CLEAN = True\n")
    _write(repo / "not_a_package/ignored.txt", "not importable\n")
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", "synthetic baseline")
    return repo


def _guard(
    repo: Path,
    *,
    argv: tuple[str, ...] = ("stage", "development"),
    cwd: Path | None = None,
    isolated: int = 1,
    dont_write_bytecode_flag: int = 1,
    dont_write_bytecode_value: bool = True,
    optimize: int = 0,
    orig_argv: tuple[str, ...] | None = None,
    preimport_sys_path: tuple[str, ...] = ("C:/stdlib",),
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
                "agent_benchmark/contextual_expert_aggregation_bootstrap.py",
                *argv,
            )
        ),
        warn_options=(),
        xoptions={},
        preimport_sys_path=preimport_sys_path,
        preloaded_agent_modules=(),
        dispatch=selected_dispatch,
    )


def _assert_violation(exc: pytest.ExceptionInfo[bootstrap.BootstrapSecurityError], key: str, path: str) -> None:
    assert exc.value.code == "unsafe_execution_surface"
    violations = exc.value.details["violations"]
    assert path in violations[key]


def test_valid_guard_returns_deeply_immutable_attestation_and_fake_dispatch(
    clean_repo: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_dispatch(operation: str, stage: str) -> tuple[Mapping[str, Any], int]:
        assert isinstance(
            getattr(sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE),
            MappingProxyType,
        )
        calls.append((operation, stage))
        return {"verified": True, "stage_pass": True}, 0

    payload, exit_code = _guard(
        clean_repo,
        argv=("verify", "confirmation"),
        dispatch=fake_dispatch,
    )

    assert exit_code == 0
    assert not hasattr(sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE)
    assert calls == [("verify", "confirmation")]
    assert isinstance(payload, MappingProxyType)
    attestation = payload["bootstrap_attestation"]
    assert isinstance(attestation, MappingProxyType)
    assert isinstance(attestation["execution_surface"], MappingProxyType)
    assert attestation["invocation"] == {
        "operation": "verify",
        "stage": "confirmation",
    }
    assert attestation["checks"]["all_import_candidates_head_index_worktree_equal"]
    candidates = attestation["execution_surface"]["candidate_paths"]
    assert "root_tool.py" in candidates
    assert "root_package/clean.py" in candidates
    assert "not_a_package/ignored.txt" not in candidates
    assert len(attestation["attestation_sha256"]) == 64
    with pytest.raises(TypeError):
        attestation["checks"]["fixed_argv"] = False


def test_dispatch_target_requires_process_active_bootstrap(
    clean_repo: Path,
) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        bootstrap.require_active_attestation(
            operation="stage",
            stage="development",
            repo_root=clean_repo,
        )
    assert exc.value.code == "missing_active_attestation"


def test_dispatch_sanitizes_git_environment_and_restores_on_error(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_INDEX_FILE", "foreign-index")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "foreign-config")
    observed: dict[str, str | None] = {}

    def failing_dispatch(
        operation: str, stage: str
    ) -> tuple[Mapping[str, Any], int]:
        observed["index"] = os.environ.get("GIT_INDEX_FILE")
        observed["config"] = os.environ.get("GIT_CONFIG_GLOBAL")
        observed["optional_locks"] = os.environ.get("GIT_OPTIONAL_LOCKS")
        raise RuntimeError("synthetic dispatch failure")

    with pytest.raises(RuntimeError, match="synthetic dispatch failure"):
        _guard(clean_repo, dispatch=failing_dispatch)
    assert observed == {
        "index": None,
        "config": None,
        "optional_locks": "0",
    }
    assert os.environ["GIT_INDEX_FILE"] == "foreign-index"
    assert os.environ["GIT_CONFIG_GLOBAL"] == "foreign-config"
    assert not hasattr(sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE)


@pytest.mark.parametrize("root_name", ("data", "e", "ui"))
def test_explicit_nonruntime_root_is_blocked_during_dispatch(
    clean_repo: Path, root_name: str
) -> None:
    _write(clean_repo / root_name / "foreign.py", "FOREIGN = True\n")

    def importing_dispatch(
        operation: str, stage: str
    ) -> tuple[Mapping[str, Any], int]:
        importlib.import_module(f"{root_name}.foreign")
        return {}, 0

    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo, dispatch=importing_dispatch)
    assert exc.value.code == "blocked_root_import"


@pytest.mark.parametrize(
    ("relative_path", "content", "violation_key"),
    (
        ("agent_benchmark/__pycache__/foreign.pyc", b"foreign", "ignored"),
        ("agent_benchmark/foreign.pyd", b"native", "ignored"),
        ("json.py", "SHADOW = True\n", "untracked_or_index_missing"),
        (
            "namespace_dir/foreign.py",
            "FOREIGN = True\n",
            "untracked_or_index_missing",
        ),
        (
            "namespace_cache/__pycache__/foreign.pyc",
            b"foreign",
            "ignored",
        ),
        (
            "audit-hyphen/foreign.py",
            "FOREIGN = True\n",
            "untracked_or_index_missing",
        ),
        ("root_package/foreign.py", "FOREIGN = True\n", "untracked_or_index_missing"),
    ),
)
def test_ignored_native_and_root_shadow_candidates_fail_closed(
    clean_repo: Path,
    relative_path: str,
    content: str | bytes,
    violation_key: str,
) -> None:
    _write(clean_repo / relative_path, content)

    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)

    _assert_violation(exc, violation_key, relative_path)


def test_modified_tracked_candidate_fails_closed(clean_repo: Path) -> None:
    _write(clean_repo / "agent_benchmark/model.py", "VALUE = 2\n")

    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)

    _assert_violation(
        exc,
        "index_worktree_divergence",
        "agent_benchmark/model.py",
    )


def test_staged_candidate_divergent_from_head_fails_closed(clean_repo: Path) -> None:
    _write(clean_repo / "agent_benchmark/model.py", "VALUE = 3\n")
    _git(clean_repo, "add", "agent_benchmark/model.py")

    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)

    _assert_violation(
        exc,
        "head_index_divergence",
        "agent_benchmark/model.py",
    )


def test_deleted_tracked_candidate_fails_closed(clean_repo: Path) -> None:
    (clean_repo / "root_package/clean.py").unlink()

    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)

    _assert_violation(exc, "worktree_missing", "root_package/clean.py")


def test_deleted_package_init_does_not_hide_versioned_package_files(
    clean_repo: Path,
) -> None:
    (clean_repo / "root_package/__init__.py").unlink()
    _write(clean_repo / "root_package/foreign.py", "FOREIGN = True\n")

    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)

    violations = exc.value.details["violations"]
    assert "root_package/__init__.py" in violations["worktree_missing"]
    assert "root_package/foreign.py" in violations["untracked_or_index_missing"]


@pytest.mark.parametrize(
    ("isolated", "flag", "value", "expected_code"),
    (
        (0, 1, True, "isolated_mode_required"),
        (1, 0, True, "bytecode_writes_disabled_required"),
        (1, 1, False, "bytecode_writes_disabled_required"),
    ),
)
def test_required_python_flags_fail_closed(
    clean_repo: Path,
    isolated: int,
    flag: int,
    value: bool,
    expected_code: str,
) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(
            clean_repo,
            isolated=isolated,
            dont_write_bytecode_flag=flag,
            dont_write_bytecode_value=value,
        )
    assert exc.value.code == expected_code


@pytest.mark.parametrize(
    ("optimize", "orig_argv"),
    (
        (1, None),
        (
            0,
            (
                "python",
                "-I",
                "-B",
                "-O",
                "agent_benchmark/contextual_expert_aggregation_bootstrap.py",
                "stage",
                "development",
            ),
        ),
    ),
)
def test_optimization_or_extra_interpreter_flags_fail_closed(
    clean_repo: Path,
    optimize: int,
    orig_argv: tuple[str, ...] | None,
) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo, optimize=optimize, orig_argv=orig_argv)
    assert exc.value.code == "noncanonical_interpreter_flags"


def test_wrong_working_directory_fails_before_git_or_dispatch(
    clean_repo: Path,
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    called = False

    def fake_dispatch(operation: str, stage: str) -> tuple[Mapping[str, Any], int]:
        nonlocal called
        called = True
        return {}, 0

    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo, cwd=elsewhere, dispatch=fake_dispatch)
    assert exc.value.code == "wrong_working_directory"
    assert not called


@pytest.mark.parametrize(
    "argv",
    (
        (),
        ("stage",),
        ("stage", "development", "extra"),
        ("run", "development"),
        ("stage", "2024"),
        ("--help", "development"),
    ),
)
def test_unsafe_argv_fails_before_repository_inspection(
    tmp_path: Path,
    argv: tuple[str, ...],
) -> None:
    nonexistent = tmp_path / "not-a-repository"
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(nonexistent, argv=argv)
    assert exc.value.code == "unsafe_argv"


def test_preimport_repo_path_fails_closed(clean_repo: Path) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo, preimport_sys_path=(str(clean_repo), "C:/stdlib"))
    assert exc.value.code == "unsafe_preimport_sys_path"


@pytest.mark.parametrize("control_name", ("exclude", "attributes"))
def test_effective_local_git_control_file_fails_closed(
    clean_repo: Path,
    control_name: str,
) -> None:
    info_file = clean_repo / ".git/info" / control_name
    _write(info_file, "*.py\n")

    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)

    assert exc.value.code == "unsafe_local_git_controls"


def test_exact_codex_info_exclusion_is_attested_but_extra_rules_fail(
    clean_repo: Path,
) -> None:
    info_file = clean_repo / ".git/info/exclude"
    _write(info_file, ".codex/\n")
    payload, code = _guard(clean_repo)
    assert code == 0
    assert tuple(
        payload["bootstrap_attestation"]["local_git_controls"][
            "approved_info_exclude_lines"
        ]
    ) == (".codex/",)

    _write(info_file, ".codex/\n*.py\n")
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)
    assert exc.value.code == "unsafe_local_git_controls"


@pytest.mark.parametrize(
    "content",
    (
        "   .codex/   \n",
        ".codex/ \n",
        " #not-a-column-zero-comment\n.codex/\n",
        "   \n.codex/\n",
    ),
)
def test_codex_info_exclusion_rejects_nonliteral_whitespace_variants(
    clean_repo: Path, content: str
) -> None:
    _write(clean_repo / ".git/info/exclude", content)
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)
    assert exc.value.code == "unsafe_local_git_controls"


def test_security_error_mapping_is_immutable_and_json_safe(clean_repo: Path) -> None:
    _write(clean_repo / "json.py", "SHADOW = True\n")
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo)
    mapping = exc.value.as_mapping()
    assert isinstance(mapping, MappingProxyType)
    json.dumps(bootstrap._thaw(mapping), sort_keys=True, allow_nan=False)
    with pytest.raises(TypeError):
        mapping["code"] = "changed"
