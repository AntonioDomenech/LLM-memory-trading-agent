from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_benchmark import contextual_expert_aggregation_audit_runner as runner
from agent_benchmark import contextual_expert_aggregation_experiment as experiment


Error = runner.ContextualExpertAggregationAuditError


def _git_identity(*, commit: str = "a" * 40) -> dict[str, object]:
    return {
        "branch": runner.EXPECTED_BRANCH,
        "commit": commit,
        "upstream": f"origin/{runner.EXPECTED_BRANCH}",
        "upstream_commit": commit,
        "origin_url": sorted(experiment.EXPECTED_ORIGIN_URLS)[0],
        "origin_repository": experiment.EXPECTED_ORIGIN_REPOSITORY,
        "head_equals_upstream": True,
        "dirty": False,
        "cleanliness_scope": "all_paths_except_unopened_input",
        "prelock_input_worktree_bytes_opened": False,
        "input_path": runner.AUDIT_INPUT_PATH.as_posix(),
        "input_head_object_id": runner.AUDIT_INPUT_SPEC.git_blob,
        "input_index_object_id": runner.AUDIT_INPUT_SPEC.git_blob,
        "input_head_index_equal": True,
        "tracked_dependency_identity": {},
        "runtime_versions": {
            "python": runner.platform.python_version(),
            "numpy": runner.np.__version__,
            "pandas": runner.pd.__version__,
        },
    }


def _parent_verification() -> dict[str, object]:
    return {
        "semantic_evidence_sha256": runner.PARENT_SEMANTIC_EVIDENCE_SHA256,
        "manifest_self_sha256": runner.PARENT_MANIFEST_SELF_SHA256,
    }


def _install_output_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    parent_relative = Path("e/audit-test")
    final_relative = parent_relative / "run-test"
    lock_relative = parent_relative / runner.ATTEMPT_LOCK_FILENAME
    monkeypatch.setattr(runner, "OUTPUT_PARENT", parent_relative)
    monkeypatch.setattr(runner, "OUTPUT_DIRECTORY", final_relative)
    monkeypatch.setattr(runner, "ATTEMPT_LOCK_PATH", lock_relative)
    (tmp_path / "e").mkdir()
    monkeypatch.setattr(experiment, "_fsync_directory", lambda _path: True)

    def exclusive_write(path: Path, payload: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(payload)

    monkeypatch.setattr(experiment, "_exclusive_write", exclusive_write)
    return tmp_path / lock_relative


def test_frozen_inventory_contains_all_inherited_and_audit_dependencies() -> None:
    frozen = set(runner.FROZEN_DEPENDENCY_PATHS)
    assert set(experiment.FROZEN_DEPENDENCY_PATHS) <= frozen
    assert Path(
        "agent_benchmark/contextual_expert_aggregation_bootstrap.py"
    ) in frozen
    assert Path(
        "agent_benchmark/contextual_expert_aggregation_audit_bootstrap.py"
    ) in frozen
    assert Path(
        "tests/test_contextual_expert_aggregation_audit_verifier.py"
    ) in frozen
    assert Path("tests/conftest.py") in frozen


def test_prelock_git_identity_never_opens_the_audit_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    commit = "b" * 40
    opened: list[str] = []
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        normalized = path.as_posix()
        if normalized.endswith(runner.AUDIT_INPUT_PATH.as_posix()):
            raise AssertionError("pre-lock path opened the audit input")
        opened.append(normalized)
        return original_read_bytes(path)

    def git_text(_root: Path, *args: str) -> str:
        if args == ("rev-parse", "--show-toplevel"):
            return str(root)
        if args == ("symbolic-ref", "--quiet", "--short", "HEAD"):
            return runner.EXPECTED_BRANCH
        if args == ("rev-parse", "HEAD"):
            return commit
        if args[-1:] == ("@{upstream}",) and "--abbrev-ref" in args:
            return f"origin/{runner.EXPECTED_BRANCH}"
        if args == ("rev-parse", "@{upstream}"):
            return commit
        if args == ("remote", "get-url", "origin"):
            return sorted(experiment.EXPECTED_ORIGIN_URLS)[0]
        if args == (
            "rev-parse",
            f"HEAD:{runner.AUDIT_INPUT_PATH.as_posix()}",
        ):
            return runner.AUDIT_INPUT_SPEC.git_blob
        raise AssertionError(f"unexpected Git query: {args!r}")

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(runner, "_git_text", git_text)
    monkeypatch.setattr(
        runner,
        "_tracked_index_object_id",
        lambda *_args: runner.AUDIT_INPUT_SPEC.git_blob,
    )
    monkeypatch.setattr(runner, "_status_excluding_authorized_paths", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(runner, "_dependency_identity", lambda _root: {})

    result = runner.require_prelock_git_identity(root)

    assert result == _git_identity(commit=commit)
    assert result["prelock_input_worktree_bytes_opened"] is False
    assert not opened


def test_attempt_lock_is_persistent_canonical_and_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = _install_output_paths(monkeypatch, tmp_path)
    created = runner.create_attempt_lock(
        repo_root=tmp_path,
        prelock_git_identity=_git_identity(),
        parent_verification=_parent_verification(),
    )

    assert lock_path.read_bytes() == created.bytes_value
    assert runner.read_attempt_lock(tmp_path) == created
    with pytest.raises(Error, match="already been consumed"):
        runner.create_attempt_lock(
            repo_root=tmp_path,
            prelock_git_identity=_git_identity(),
            parent_verification=_parent_verification(),
        )
    assert lock_path.read_bytes() == created.bytes_value


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("created_at_utc", "not-a-time"),
        ("git_commit", True),
        ("parent_verification_sha256", "sha256:" + "z" * 64),
        ("attempt_number", True),
    ),
)
def test_attempt_lock_rejects_noncanonical_typed_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    changed: object,
) -> None:
    lock_path = _install_output_paths(monkeypatch, tmp_path)
    runner.create_attempt_lock(
        repo_root=tmp_path,
        prelock_git_identity=_git_identity(),
        parent_verification=_parent_verification(),
    )
    value = json.loads(lock_path.read_text(encoding="utf-8"))
    value[field] = changed
    lock_path.write_bytes(runner._json_bytes(value))

    with pytest.raises(Error, match="frozen contract"):
        runner.read_attempt_lock(tmp_path)


def test_postlock_output_inventory_fails_before_any_input_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / runner.OUTPUT_PARENT
    parent.mkdir(parents=True)
    (parent / runner.ATTEMPT_LOCK_FILENAME).write_text("lock", encoding="utf-8")
    (parent / "unexpected").mkdir()
    monkeypatch.setattr(
        experiment,
        "tracked_file_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("input identity must not be read after output tamper")
        ),
    )

    with pytest.raises(Error, match="unexpected output"):
        runner.require_postlock_input_identity(tmp_path, _git_identity())


def test_locked_loader_rejects_changed_lock_before_postlock_input_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runner.AuditLockEvidence(
        path="lock.json",
        sha256="sha256:" + "1" * 64,
        content={"value": 1},
        bytes_value=b"one\n",
        parent_directory_fsync_supported=True,
    )
    changed = runner.AuditLockEvidence(
        path="lock.json",
        sha256="sha256:" + "2" * 64,
        content={"value": 2},
        bytes_value=b"two\n",
        parent_directory_fsync_supported=True,
    )
    monkeypatch.setattr(runner, "read_attempt_lock", lambda _root: changed)
    monkeypatch.setattr(
        runner,
        "require_postlock_input_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("post-lock input access must not occur")
        ),
    )

    with pytest.raises(Error, match="changed before the first input read"):
        runner.load_locked_audit_snapshot(
            tmp_path,
            development=SimpleNamespace(),
            prelock_git_identity=_git_identity(),
            lock=original,
            deadline=experiment.StageDeadline(),
        )


def test_deadline_overrun_before_lock_never_consumes_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_benchmark import contextual_expert_aggregation_audit_bootstrap as bootstrap

    phases: list[str] = []

    class Deadline:
        def __init__(self, *, limit_seconds: float) -> None:
            assert limit_seconds == runner.RUN_TIME_LIMIT_SECONDS

        def check(self, phase: str) -> float:
            phases.append(phase)
            if phase == "immediately before irreversible audit attempt lock":
                raise experiment.ContextualExpertAggregationExperimentError(
                    "simulated deadline overrun"
                )
            return 0.0

    identity = _git_identity()
    monkeypatch.setattr(bootstrap, "require_active_attestation", lambda **_kwargs: {})
    monkeypatch.setattr(experiment, "StageDeadline", Deadline)
    monkeypatch.setattr(runner, "require_unused_destination", lambda _root: tmp_path)
    monkeypatch.setattr(runner, "require_prelock_git_identity", lambda _root: identity)
    monkeypatch.setattr(
        runner,
        "prepare_rejected_development",
        lambda *_args, **_kwargs: SimpleNamespace(
            parent_verification=_parent_verification()
        ),
    )
    monkeypatch.setattr(
        runner,
        "create_attempt_lock",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("deadline failure must occur before lock creation")
        ),
    )

    with pytest.raises(
        experiment.ContextualExpertAggregationExperimentError,
        match="deadline overrun",
    ):
        runner.run_audit(tmp_path)

    assert phases[-1] == "immediately before irreversible audit attempt lock"
    assert not (tmp_path / runner.ATTEMPT_LOCK_PATH).exists()


def test_locked_loader_checks_deadline_before_lock_or_input_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Expired:
        def check(self, phase: str) -> float:
            assert phase == "before durable audit lock readback"
            raise experiment.ContextualExpertAggregationExperimentError("expired")

    monkeypatch.setattr(
        runner,
        "read_attempt_lock",
        lambda _root: (_ for _ in ()).throw(
            AssertionError("expired loader must not read even the lock")
        ),
    )

    with pytest.raises(
        experiment.ContextualExpertAggregationExperimentError, match="expired"
    ):
        runner.load_locked_audit_snapshot(
            tmp_path,
            development=SimpleNamespace(),
            prelock_git_identity=_git_identity(),
            lock=SimpleNamespace(),
            deadline=Expired(),
        )


def test_attempt_lock_path_is_hardened_before_any_byte_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment,
        "_harden_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            experiment.ContextualExpertAggregationExperimentError("redirected")
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("redirected lock bytes must not be opened")
        ),
    )

    with pytest.raises(Error, match="redirected or missing"):
        runner.read_attempt_lock(tmp_path)
