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
    contextual_expert_aggregation_2024_audit_bootstrap as bootstrap,
)
from agent_benchmark import (
    contextual_expert_aggregation_2024_audit_control as control,
)


SYNTHETIC_DEPENDENCIES = (
    Path(".gitattributes"),
    Path(".gitignore"),
    Path("requirements.txt"),
    Path("agent_benchmark/__init__.py"),
    Path(
        "agent_benchmark/"
        "contextual_expert_aggregation_2024_audit_bootstrap.py"
    ),
    Path("agent_benchmark/contextual_expert_aggregation_2024_audit_runner.py"),
    Path("agent_benchmark/contextual_expert_aggregation_2024_audit_verifier.py"),
    Path("agent_benchmark/model.py"),
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
def clean_repo(tmp_path: Path) -> SimpleNamespace:
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(origin)], check=True
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet", "-b", "audit-test")
    _git(repo, "config", "user.name", "2024 Bootstrap Test")
    _git(repo, "config", "user.email", "bootstrap@example.invalid")
    _write(repo / ".gitattributes", "* -text\n")
    _write(repo / ".gitignore", "__pycache__/\n*.py[cod]\n")
    _write(repo / "requirements.txt", "pytest\n")
    _write(repo / "agent_benchmark/__init__.py", "\n")
    _write(
        repo
        / "agent_benchmark/"
        "contextual_expert_aggregation_2024_audit_bootstrap.py",
        "# synthetic tracked bootstrap\n",
    )
    _write(
        repo
        / "agent_benchmark/"
        "contextual_expert_aggregation_2024_audit_runner.py",
        "def run_stage(stage): return {'stage_pass': True}\n",
    )
    _write(
        repo
        / "agent_benchmark/"
        "contextual_expert_aggregation_2024_audit_verifier.py",
        "def verify_stage(stage): return {'verified': True}\n",
    )
    _write(repo / "agent_benchmark/model.py", "VALUE = 1\n")
    _write(
        repo
        / "e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/"
        "authorized_inputs/aapl_spy_qqq_through_2024.csv",
        "bounded,input\n",
    )
    _write(
        repo
        / "e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/"
        "input_snapshot_receipt.json",
        "{}\n",
    )
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", "synthetic bootstrap root")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "--quiet", "-u", "origin", "audit-test")
    return SimpleNamespace(
        repo=repo.resolve(),
        origin=_git(repo, "remote", "get-url", "origin"),
        commit=_git(repo, "rev-parse", "HEAD"),
        input_blob=_git(
            repo,
            "rev-parse",
            "HEAD:e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/"
            "authorized_inputs/aapl_spy_qqq_through_2024.csv",
        ),
        receipt_blob=_git(
            repo,
            "rev-parse",
            "HEAD:e/aapl_causal_contextual_expert_aggregation_2024_audit_v2/"
            "input_snapshot_receipt.json",
        ),
    )


def _configure(
    monkeypatch: pytest.MonkeyPatch, fixture: SimpleNamespace
) -> None:
    monkeypatch.setattr(
        bootstrap, "FROZEN_DEPENDENCY_PATHS", SYNTHETIC_DEPENDENCIES
    )
    monkeypatch.setattr(bootstrap, "_EXPECTED_BRANCH", "audit-test")
    monkeypatch.setattr(
        bootstrap, "_PREREGISTRATION_COMMIT", fixture.commit
    )
    monkeypatch.setattr(
        bootstrap, "_EXPECTED_ORIGIN_URLS", frozenset({fixture.origin})
    )
    monkeypatch.setattr(bootstrap, "_INPUT_GIT_BLOB", fixture.input_blob)
    monkeypatch.setattr(bootstrap, "_RECEIPT_GIT_BLOB", fixture.receipt_blob)


class _Clock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _guard(
    fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    argv: tuple[str, ...] = ("stage", "audit_2024"),
    dispatch: bootstrap.DispatchCallable | None = None,
    clock: bootstrap.ClockCallable | None = None,
    preimport_sys_path: tuple[str, ...] | None = None,
) -> tuple[Mapping[str, Any], int]:
    _configure(monkeypatch, fixture)
    selected_dispatch = dispatch or (
        lambda operation, stage: (
            {"operation": operation, "stage": stage, "stage_pass": True},
            0,
        )
    )
    selected_clock = clock or (
        _Clock(100.0, 101.0) if argv[0:1] == ("verify",) else _Clock(100.0)
    )
    return bootstrap._guarded_main(
        argv,
        repo_root=fixture.repo,
        cwd=fixture.repo,
        isolated=1,
        dont_write_bytecode_flag=1,
        dont_write_bytecode_value=True,
        optimize=0,
        orig_argv=(
            "python",
            "-I",
            "-B",
            bootstrap._BOOTSTRAP_RELATIVE_PATH.as_posix(),
            *argv,
        ),
        warn_options=(),
        xoptions={},
        preimport_sys_path=(
            preimport_sys_path
            if preimport_sys_path is not None
            else (str(fixture.repo.parent / "external-stdlib"),)
        ),
        preloaded_agent_modules=(),
        dispatch=selected_dispatch,
        clock=selected_clock,
    )


def test_bootstrap_module_has_only_standard_library_imports() -> None:
    source_path = Path(bootstrap.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


def test_bootstrap_literal_dependency_inventory_matches_control() -> None:
    assert bootstrap.FROZEN_DEPENDENCY_PATHS == control.FROZEN_DEPENDENCY_PATHS
    assert bootstrap._INPUT_RELATIVE_PATH not in bootstrap.FROZEN_DEPENDENCY_PATHS
    assert bootstrap._RECEIPT_RELATIVE_PATH not in bootstrap.FROZEN_DEPENDENCY_PATHS


@pytest.mark.parametrize(
    "argv",
    (
        (),
        ("stage",),
        ("stage", "audit"),
        ("verify", "audit"),
        ("run", "audit_2024"),
        ("stage", "audit_2024", "extra"),
    ),
)
def test_only_exact_2024_commands_are_allowed(argv: tuple[str, ...]) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        bootstrap._parse_fixed_argv(argv)
    assert exc.value.code == "unsafe_argv"


def test_guard_hardens_and_attests_without_opening_input_or_receipt(
    clean_repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = (
        clean_repo.repo / bootstrap._INPUT_RELATIVE_PATH
    ).resolve()
    receipt_path = (
        clean_repo.repo / bootstrap._RECEIPT_RELATIVE_PATH
    ).resolve()
    input_path.write_text("locally,changed\n", encoding="utf-8")
    receipt_path.write_text('{"local":"changed"}\n', encoding="utf-8")
    original_read_bytes = Path.read_bytes
    original_scandir = os.scandir
    git_calls: list[tuple[str, ...]] = []
    original_run_git = bootstrap._run_git
    observed: dict[str, Any] = {}

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() in {input_path, receipt_path}:
            raise AssertionError("bootstrap opened bounded worktree bytes")
        return original_read_bytes(path)

    def broad_scan_forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("bootstrap attempted broad directory enumeration")

    def recording_git(
        executable: Path,
        repo_root: Path,
        args: tuple[str, ...],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        git_calls.append(tuple(args))
        return original_run_git(
            executable, repo_root, args, check=check
        )

    def dispatch(
        operation: str, stage: str
    ) -> tuple[Mapping[str, Any], int]:
        attestation = getattr(
            sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE
        )
        observed["operation"] = operation
        observed["stage"] = stage
        observed["attestation"] = attestation
        observed["root_first"] = Path(sys.path[0]).resolve()
        observed["pythonpath"] = os.environ.get("PYTHONPATH")
        return {"stage_pass": True}, 0

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(os, "scandir", broad_scan_forbidden)
    monkeypatch.setattr(bootstrap, "_run_git", recording_git)
    try:
        payload, exit_code = _guard(
            clean_repo, monkeypatch, dispatch=dispatch
        )
    finally:
        monkeypatch.setattr(os, "scandir", original_scandir)

    assert exit_code == 0
    assert observed["operation"] == "stage"
    assert observed["stage"] == "audit_2024"
    assert observed["root_first"] == clean_repo.repo
    assert observed["pythonpath"] is None
    attestation = observed["attestation"]
    assert isinstance(attestation, MappingProxyType)
    assert attestation["checks"]["bounded_input_worktree_unopened"] is True
    assert attestation["checks"]["receipt_worktree_unopened"] is True
    assert attestation["execution_surface"][
        "broad_repository_enumeration_performed"
    ] is False
    assert payload["bootstrap_exit"] is None
    assert not hasattr(sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE)
    for call in git_calls:
        if call[:1] == ("ls-tree",) or call[:1] == ("ls-files",):
            assert "--" in call
            pathspec = call[-1]
            assert pathspec.startswith(":(top,literal)")
        assert call[:3] != ("ls-tree", "-r", "-z")


def test_modified_literal_dependency_fails_closed(
    clean_repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    (clean_repo.repo / "agent_benchmark/model.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo, monkeypatch)
    assert exc.value.code == "dependency_identity_changed"


def test_staged_input_change_is_detected_without_worktree_read(
    clean_repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = clean_repo.repo / bootstrap._INPUT_RELATIVE_PATH
    input_path.write_text("staged,change\n", encoding="utf-8")
    _git(clean_repo.repo, "add", input_path.relative_to(clean_repo.repo).as_posix())
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.resolve() == input_path.resolve():
            raise AssertionError("bootstrap opened staged bounded bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(clean_repo, monkeypatch)
    assert exc.value.code == "unopened_control_identity_changed"


def test_control_tree_and_unlisted_agent_imports_are_blocked(
    clean_repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(monkeypatch, clean_repo)
    blocker = bootstrap._BlockedRepositoryImportFinder(clean_repo.repo)
    with pytest.raises(bootstrap.BootstrapSecurityError) as control_exc:
        blocker.find_spec("e.hidden_result")
    assert control_exc.value.code == "blocked_control_tree_import"

    _write(clean_repo.repo / "agent_benchmark/foreign.py", "VALUE = 1\n")
    with pytest.raises(bootstrap.BootstrapSecurityError) as agent_exc:
        blocker.find_spec("agent_benchmark.foreign")
    assert agent_exc.value.code == "unattested_agent_import"


def test_import_dispatch_has_only_fixed_runner_and_verifier_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_import(module_name: str) -> SimpleNamespace:
        calls.append(module_name)
        if module_name.endswith("_runner"):
            return SimpleNamespace(
                run_stage=lambda stage: {"stage": stage, "stage_pass": True}
            )
        return SimpleNamespace(
            verify_stage=lambda stage: {"stage": stage, "verified": True}
        )

    monkeypatch.setattr(importlib, "import_module", fake_import)
    stage_result, stage_exit = bootstrap._import_dispatch(
        "stage", "audit_2024"
    )
    verify_result, verify_exit = bootstrap._import_dispatch(
        "verify", "audit_2024"
    )
    assert stage_result == {"stage": "audit_2024", "stage_pass": True}
    assert verify_result == {"stage": "audit_2024", "verified": True}
    assert stage_exit == verify_exit == 0
    assert calls == [
        "agent_benchmark.contextual_expert_aggregation_2024_audit_runner",
        "agent_benchmark.contextual_expert_aggregation_2024_audit_verifier",
    ]


def test_verify_uses_entry_clock_for_strict_bootstrap_exit(
    clean_repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, exit_code = _guard(
        clean_repo,
        monkeypatch,
        argv=("verify", "audit_2024"),
        clock=_Clock(100.0, 1_899.999),
    )
    assert exit_code == 0
    assert payload["bootstrap_attestation"]["monotonic_start"] == 100.0
    assert payload["bootstrap_exit"] == {
        "phase": "bootstrap_exit",
        "elapsed_seconds": pytest.approx(1_799.999),
        "strict_limit_seconds": 1_800.0,
        "passed": True,
    }


def test_dispatch_can_recover_the_same_frozen_entry_attestation(
    clean_repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_flags = sys.flags
    original_dont_write_bytecode = sys.dont_write_bytecode

    class FlagsProxy:
        isolated = 1
        dont_write_bytecode = 1

        def __getattr__(self, name: str) -> object:
            return getattr(original_flags, name)

    def dispatch(
        operation: str, stage: str
    ) -> tuple[Mapping[str, Any], int]:
        sys.flags = FlagsProxy()  # type: ignore[assignment]
        sys.dont_write_bytecode = True
        try:
            attestation = bootstrap.require_active_attestation(
                operation, stage, clean_repo.repo
            )
            assert attestation["monotonic_start"] == 100.0
            assert bootstrap.bootstrap_elapsed(
                attestation, clock=lambda: 101.0
            ) == 1.0
        finally:
            sys.flags = original_flags
            sys.dont_write_bytecode = original_dont_write_bytecode
        return {"stage_pass": True}, 0

    _guard(clean_repo, monkeypatch, dispatch=dispatch, clock=_Clock(100.0))


def test_bootstrap_exit_at_exact_limit_fails_and_restores_process_state(
    clean_repo: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "foreign-python-path")
    original_path = list(sys.path)
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        _guard(
            clean_repo,
            monkeypatch,
            argv=("verify", "audit_2024"),
            clock=_Clock(100.0, 1_900.0),
        )
    assert exc.value.code == "bootstrap_deadline_exceeded"
    assert os.environ["PYTHONPATH"] == "foreign-python-path"
    assert sys.path == original_path
    assert not hasattr(sys, bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE)


def test_elapsed_helpers_reject_backward_or_unknown_samples() -> None:
    attestation = {"monotonic_start": 100.0}
    assert bootstrap.require_bootstrap_deadline(
        attestation, "preverify", clock=lambda: 101.0
    ) == 1.0
    with pytest.raises(bootstrap.BootstrapSecurityError) as backward:
        bootstrap.bootstrap_elapsed(attestation, clock=lambda: 99.0)
    assert backward.value.code == "invalid_bootstrap_clock"
    with pytest.raises(bootstrap.BootstrapSecurityError) as unknown:
        bootstrap.require_bootstrap_deadline(
            attestation, "unknown", clock=lambda: 101.0
        )
    assert unknown.value.code == "invalid_deadline_phase"


def test_runner_and_verifier_require_active_attestation(
    clean_repo: SimpleNamespace,
) -> None:
    with pytest.raises(bootstrap.BootstrapSecurityError) as exc:
        bootstrap.require_active_attestation(
            "stage", "audit_2024", clean_repo.repo
        )
    assert exc.value.code == "missing_active_attestation"
