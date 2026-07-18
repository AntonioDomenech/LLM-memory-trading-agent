from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import agent_benchmark.sec_gemma_lean_runner as lean
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    RUNTIME_FINGERPRINT_SHA256,
    canonical_sha256,
)


HEAD = "1" * 40
CONTACT = "Lean Research contact@valid-domain.dev"
CONTACT_SHA256 = "sha256:" + hashlib.sha256(CONTACT.encode()).hexdigest()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    return root.resolve()


def _blob(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def _fake_git(
    *,
    branch: str = lean.BRANCH_NAME,
    status: bytes = b"",
    upstream: str = HEAD,
    parent_tree: str = lean.SCIENTIFIC_PARENT_TREE,
    drift_path: str | None = None,
    config_tracked: bool = False,
    config_ignored: bool = True,
    checkpoint_ignored: bool = True,
):
    def run(repo_root: Path, arguments: tuple[str, ...]) -> lean._GitResult:
        del repo_root
        if arguments == ("branch", "--show-current"):
            return lean._GitResult(0, branch.encode())
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return lean._GitResult(0, status)
        if arguments == ("remote", "get-url", "origin"):
            return lean._GitResult(0, lean.EXPECTED_ORIGIN_URL.encode())
        if arguments == ("rev-parse", "HEAD"):
            return lean._GitResult(0, HEAD.encode())
        if arguments == (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            value = f"origin/{lean.BRANCH_NAME}".encode()
            return lean._GitResult(0, value)
        if arguments == ("rev-parse", "@{upstream}"):
            return lean._GitResult(0, upstream.encode())
        if arguments[:2] == ("merge-base", "--is-ancestor"):
            return lean._GitResult(0, b"")
        if arguments == (
            "rev-parse",
            f"{lean.SCIENTIFIC_PARENT_COMMIT}^{{tree}}",
        ):
            return lean._GitResult(0, parent_tree.encode())
        if arguments[:1] == ("rev-parse",) and len(arguments) == 2:
            spec = arguments[1]
            if spec in {
                f"HEAD:{lean.PREREGISTRATION_PATH}",
                f"{lean.PREREGISTRATION_COMMIT}:{lean.PREREGISTRATION_PATH}",
            }:
                return lean._GitResult(
                    0,
                    _blob(lean.PREREGISTRATION_PATH).encode(),
                )
            if ":" in spec:
                _, path = spec.split(":", 1)
                value = _blob(path)
                if drift_path == path and spec.startswith(
                    lean.SCIENTIFIC_PARENT_COMMIT
                ):
                    value = "f" * 40
                return lean._GitResult(0, value.encode())
        if arguments[:3] == (
            "ls-files",
            "--error-unmatch",
            "--",
        ):
            return lean._GitResult(0 if config_tracked else 1, b"")
        if arguments[:4] == (
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
        ):
            path = arguments[4]
            if path == lean.PRIVATE_CONFIG_PATH:
                return lean._GitResult(0 if config_ignored else 1, b"")
            return lean._GitResult(0 if checkpoint_ignored else 1, b"")
        raise AssertionError(f"unexpected git call: {arguments!r}")

    return run


def _runtime_receipt() -> dict[str, Any]:
    return {
        "model_name": "gemma4:12b",
        "runtime_fingerprint_sha256": RUNTIME_FINGERPRINT_SHA256,
        "runtime_receipt_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "ordered_layer_content_sha256s": ["5" * 64],
        "version_response_sha256": "6" * 64,
        "show_response_semantic_sha256": "7" * 64,
        "show_response_raw_sha256": "8" * 64,
    }


def _source_identity() -> dict[str, Any]:
    return {
        "executing_repository_root_bound": True,
        "loaded_source_modules_bound": [],
    }


def _dependency_identity() -> dict[str, Any]:
    return {
        "verified_v22_dependency_closure_sha256": (
            lean.V22_DEPENDENCY_CLOSURE_SHA256
        ),
        "dependency_identity_recomputed": True,
    }


def test_repository_verifier_binds_clean_pushed_branch_and_parent(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    result = lean._verify_repository(
        root,
        git_runner=_fake_git(),
        source_tree_verifier=lambda _: _source_identity(),
        dependency_verifier=lambda _: _dependency_identity(),
    )

    assert result["branch"] == lean.BRANCH_NAME
    assert result["head_commit"] == HEAD
    assert result["cached_upstream_commit"] == HEAD
    assert result["remote_state_basis"] == "local_cached_origin_tracking_ref"
    assert result["scientific_parent_tree"] == lean.SCIENTIFIC_PARENT_TREE
    assert result["frozen_scientific_source_count"] == len(
        lean.FROZEN_SCIENTIFIC_PATHS
    )
    assert result["frozen_scientific_source_blobs_sha256"] == canonical_sha256(
        result["frozen_scientific_source_blobs"]
    )
    assert result["dependency_identity"] == _dependency_identity()


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"branch": "codex/wrong"}, "wrong_branch"),
        ({"status": b"?? unexpected.py\n"}, "working_tree_not_clean"),
        ({"upstream": "9" * 40}, "head_not_in_cached_origin"),
        ({"parent_tree": "8" * 40}, "scientific_parent_changed"),
        (
            {"drift_path": lean.FROZEN_SCIENTIFIC_PATHS[0]},
            "scientific_source_changed",
        ),
    ],
)
def test_repository_verifier_fails_closed(
    tmp_path: Path,
    kwargs: dict[str, Any],
    code: str,
) -> None:
    root = _repo(tmp_path)
    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._verify_repository(
            root,
            git_runner=_fake_git(**kwargs),
            source_tree_verifier=lambda _: _source_identity(),
            dependency_verifier=lambda _: _dependency_identity(),
        )
    assert caught.value.code == code


def test_dependency_identity_is_recomputed_from_frozen_closure() -> None:
    root = Path(lean.__file__).resolve().parents[1]

    result = lean._verify_dependency_identity(root)

    assert result == {
        "verified_v22_dependency_closure_sha256": (
            lean.V22_DEPENDENCY_CLOSURE_SHA256
        ),
        "dependency_source_count": 1,
        "external_distribution_count": 5,
        "numerical_time_distribution_count": 2,
        "dependency_identity_recomputed": True,
    }


def test_runtime_modules_share_exact_verified_requests_identity() -> None:
    repository_root = Path(lean.__file__).resolve(strict=True).parents[1]
    child = r"""
import json
import os
import sys

import agent_benchmark.sec_gemma_lean_runner as lean


def identity_report(verified_requests=None, ollama=None, production=None):
    process_requests = sys.modules.get("requests")
    return {
        "python_dont_write_bytecode": os.environ.get(
            "PYTHONDONTWRITEBYTECODE"
        ),
        "process_requests_id": (
            None if process_requests is None else id(process_requests)
        ),
        "verified_requests_id": (
            None if verified_requests is None else id(verified_requests)
        ),
        "ollama_requests_id": (
            None
            if ollama is None
            else id(getattr(ollama, "requests", None))
        ),
        "production_requests_id": (
            None
            if production is None
            else id(getattr(production, "requests", None))
        ),
    }


try:
    verified_requests, ollama, production = (
        lean._load_verified_runtime_modules()
    )
except BaseException as error:
    report = identity_report()
    report["error_type"] = type(error).__name__
    report["error"] = str(error)
    print(json.dumps(report, sort_keys=True), file=sys.stderr)
    raise

checks = {
    "sys_modules_requests": (
        sys.modules.get("requests") is verified_requests
    ),
    "ollama_requests": ollama.requests is verified_requests,
    "production_requests": production.requests is verified_requests,
}
if not all(checks.values()):
    report = identity_report(verified_requests, ollama, production)
    report["identity_checks"] = checks
    print(json.dumps(report, sort_keys=True), file=sys.stderr)
    raise SystemExit(17)
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", child],
        cwd=repository_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, (
        "fresh runtime identity subprocess failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def test_execution_binding_rejects_a_different_checkout(tmp_path: Path) -> None:
    root = _repo(tmp_path)

    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._verify_execution_and_source_bytes(root, git_runner=_fake_git())

    assert caught.value.code == "execution_root_mismatch"


def test_safe_source_reader_rejects_hard_links(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    source = root / "source.py"
    alias = root / "alias.py"
    source.write_bytes(b"print('bound')\n")
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("hard links are unavailable on this test filesystem")

    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._read_safe_source(root, "source.py")

    assert caught.value.code == "executed_source_invalid"


def test_private_contact_returns_only_fingerprint(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    config = root / lean.PRIVATE_CONFIG_PATH
    config.parent.mkdir()
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": CONTACT}}),
        encoding="utf-8",
    )

    fingerprint = lean._contact_fingerprint(root, git_runner=_fake_git())

    assert fingerprint == CONTACT_SHA256
    assert CONTACT not in fingerprint


@pytest.mark.parametrize(
    ("tracked", "ignored", "code"),
    [
        (True, True, "private_config_tracked"),
        (False, False, "private_config_not_ignored"),
    ],
)
def test_private_config_must_be_untracked_and_ignored(
    tmp_path: Path,
    tracked: bool,
    ignored: bool,
    code: str,
) -> None:
    root = _repo(tmp_path)
    config = root / lean.PRIVATE_CONFIG_PATH
    config.parent.mkdir()
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": CONTACT}}),
        encoding="utf-8",
    )
    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._contact_fingerprint(
            root,
            git_runner=_fake_git(
                config_tracked=tracked,
                config_ignored=ignored,
            ),
        )
    assert caught.value.code == code


def test_invalid_contact_never_appears_in_public_error(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    config = root / lean.PRIVATE_CONFIG_PATH
    config.parent.mkdir()
    secret = "Private Person private@example.com"
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": secret}}),
        encoding="utf-8",
    )
    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._contact_fingerprint(root, git_runner=_fake_git())
    assert caught.value.code == "private_contact_invalid"
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_invalid_loader_result_is_deleted_from_contact_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_benchmark.sec_point_in_time as point_in_time

    root = _repo(tmp_path)
    config = root / lean.PRIVATE_CONFIG_PATH
    config.parent.mkdir()
    secret = "Private Person private@real-domain.dev"
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": secret}}),
        encoding="utf-8",
    )

    class InvalidAudit:
        real_contact_validated = True
        sha256 = secret

    monkeypatch.setattr(
        point_in_time,
        "validate_sec_user_agent",
        lambda _: InvalidAudit(),
    )

    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._contact_fingerprint(root, git_runner=_fake_git())

    traceback = caught.value.__traceback__
    contact_frame = None
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == "_contact_fingerprint":
            contact_frame = traceback.tb_frame
            break
        traceback = traceback.tb_next
    assert contact_frame is not None
    assert secret not in repr(contact_frame.f_locals)


def test_checkpoint_probe_is_ignored_and_leaves_no_file(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    checkpoint = lean._prepare_checkpoint_root(
        root,
        git_runner=_fake_git(),
    )
    lean._write_probe(checkpoint)

    assert checkpoint.is_dir()
    assert list(checkpoint.iterdir()) == []


def test_existing_run_lock_blocks_second_owner_and_cleans_up(
    tmp_path: Path,
) -> None:
    lock = tmp_path / lean.ACTIVE_LOCK_NAME
    with lean._RunLease(lock):
        assert lock.exists()
        with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
            with lean._RunLease(lock):
                pass
        assert caught.value.code == "active_run_present"
    assert not lock.exists()


def test_run_lock_close_failure_still_removes_owned_marker(tmp_path: Path) -> None:
    class FailingHandle:
        def close(self) -> None:
            raise OSError("simulated close failure")

    lock = tmp_path / lean.ACTIVE_LOCK_NAME
    payload = b"owned marker"
    lock.write_bytes(payload)
    lease = lean._RunLease(lock)
    details = lock.stat()
    lease._identity = (details.st_dev, details.st_ino)
    lease._payload_sha256 = hashlib.sha256(payload).hexdigest()
    lease._handle = FailingHandle()

    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lease.__exit__(None, None, None)

    assert caught.value.code == "run_lock_cleanup_failed"
    assert not lock.exists()


def test_run_lock_never_deletes_a_replaced_marker(tmp_path: Path) -> None:
    lock = tmp_path / lean.ACTIVE_LOCK_NAME
    original = b"original marker"
    replacement = b"replacement marker"
    lock.write_bytes(original)
    lease = lean._RunLease(lock)
    details = lock.stat()
    lease._identity = (details.st_dev, details.st_ino)
    lease._payload_sha256 = hashlib.sha256(original).hexdigest()
    lock.unlink()
    lock.write_bytes(replacement)

    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lease.__exit__(None, None, None)

    assert caught.value.code == "run_lock_cleanup_failed"
    assert lock.read_bytes() == replacement


def test_runtime_verification_uses_probe_bytes_and_reports_no_generation() -> None:
    calls: list[tuple[bytes, bytes]] = []

    def probe() -> dict[str, str]:
        return {
            "version_response_hex": b'{"version":"0.32.0"}'.hex(),
            "show_response_hex": b'{"model":"gemma4:12b"}'.hex(),
        }

    def verify(
        *,
        version_response_bytes: bytes,
        show_response_bytes: bytes,
    ) -> dict[str, Any]:
        calls.append((version_response_bytes, show_response_bytes))
        return _runtime_receipt()

    result = lean._verify_runtime(
        probe_loader=probe,
        bundle_verifier=verify,
    )

    assert len(calls) == 1
    assert result["model_name"] == "gemma4:12b"
    assert result["runtime_fingerprint_sha256"] == RUNTIME_FINGERPRINT_SHA256
    assert result["identity_probe"] == {
        "restricted_loopback_sequence_enforced": False,
        "observed_request_count": None,
        "test_injections_used": True,
    }
    assert result["observed_effect_counts"] == {
        "external_network_requests": None,
        "loopback_identity_requests": None,
        "model_generation_calls": None,
    }
    assert "chat" not in json.dumps(result).lower()


def test_restricted_runtime_transport_allows_only_exact_recorded_sequence() -> None:
    class Delegate:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []
            self.closed = False

        def request(self, method: str, endpoint: str, **kwargs: Any) -> object:
            del kwargs
            self.calls.append((method, endpoint))
            return object()

        def close(self) -> None:
            self.closed = True

    version = "http://127.0.0.1:11434/api/version"
    show = "http://127.0.0.1:11434/api/show"
    delegate = Delegate()
    transport = lean._RestrictedRuntimeProbeTransport(
        delegate,
        (("GET", version), ("POST", show)),
    )

    transport.request("GET", version)
    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        transport.request("POST", "http://127.0.0.1:11434/api/chat")
    assert caught.value.code == "runtime_probe_request_forbidden"
    transport.request("POST", show)
    with pytest.raises(lean.SecGemmaLeanRunnerError):
        transport.request("GET", version)
    transport.close()

    assert transport.calls == (("GET", version), ("POST", show))
    assert delegate.calls == list(transport.calls)
    assert delegate.closed is True


def test_runtime_semantic_or_fingerprint_change_fails_closed() -> None:
    receipt = _runtime_receipt()
    receipt["runtime_fingerprint_sha256"] = "0" * 64

    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._verify_runtime(
            probe_loader=lambda: {
                "version_response_hex": "7b7d",
                "show_response_hex": "7b7d",
            },
            bundle_verifier=lambda **_: receipt,
        )
    assert caught.value.code == "runtime_identity_invalid"


def test_complete_preflight_receipt_is_self_hashed_and_secret_free(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    checkpoint = root / "data" / "checkpoint"
    checkpoint.mkdir(parents=True)
    readings = iter([10.0, 12.5])

    report = lean._run_untrusted_preflight(
        root,
        repository_verifier=lambda _: {"head_commit": HEAD},
        contact_verifier=lambda _: CONTACT_SHA256,
        checkpoint_preparer=lambda _: checkpoint,
        runtime_verifier=lambda: {
            "model_name": "gemma4:12b",
            "runtime_fingerprint_sha256": RUNTIME_FINGERPRINT_SHA256,
        },
        clock=lambda: next(readings),
        now=lambda: "2026-07-17T07:00:00Z",
    )

    body = {key: value for key, value in report.items() if key != "preflight_sha256"}
    assert report["preflight_sha256"] == canonical_sha256(body)
    assert report["elapsed_seconds_hex"] == (2.5).hex()
    assert set(report["effect_counts"].values()) == {None}
    assert report["status"] == "test_only_untrusted"
    assert report["trust"] == {
        "production_defaults_only": False,
        "test_injections_used": True,
        "eligible_for_public_seal": False,
    }
    assert report["execution_authorized"] is False
    assert CONTACT not in json.dumps(report)
    assert not (checkpoint / lean.ACTIVE_LOCK_NAME).exists()


@pytest.mark.parametrize(
    ("readings", "created_at", "code"),
    [
        ([2.0, 1.0], "2026-07-17T07:00:00Z", "monotonic_clock_invalid"),
        ([1.0, float("inf")], "2026-07-17T07:00:00Z", "monotonic_clock_invalid"),
        ([1.0, 2.0], "bananaZ", "wall_clock_invalid"),
        ([1.0, 2.0], "2026-02-30T07:00:00Z", "wall_clock_invalid"),
    ],
)
def test_preflight_rejects_invalid_elapsed_or_wall_time(
    tmp_path: Path,
    readings: list[float],
    created_at: str,
    code: str,
) -> None:
    root = _repo(tmp_path)
    checkpoint = root / "data" / "checkpoint"
    checkpoint.mkdir(parents=True)
    clock_values = iter(readings)

    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._run_untrusted_preflight(
            root,
            repository_verifier=lambda _: {"head_commit": HEAD},
            contact_verifier=lambda _: CONTACT_SHA256,
            checkpoint_preparer=lambda _: checkpoint,
            runtime_verifier=lambda: {},
            clock=clock_values.__next__,
            now=lambda: created_at,
        )

    assert caught.value.code == code
    assert not (checkpoint / lean.ACTIVE_LOCK_NAME).exists()


def test_untrusted_preflight_cannot_be_sealed_through_public_runner(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    checkpoint = root / "data" / "checkpoint"
    checkpoint.mkdir(parents=True)

    report = lean._run_untrusted_preflight(
        root,
        repository_verifier=lambda _: {"head_commit": HEAD},
        contact_verifier=lambda _: CONTACT_SHA256,
        checkpoint_preparer=lambda _: checkpoint,
        runtime_verifier=lambda: {},
        clock=iter([1.0, 2.0]).__next__,
        now=lambda: "2026-07-17T07:00:00Z",
    )
    assert report["status"] == "test_only_untrusted"

    with pytest.raises(TypeError):
        lean._run_preflight_core(
            root,
            seal=True,
            repository_verifier=lambda _: {"head_commit": HEAD},
            contact_verifier=lambda _: CONTACT_SHA256,
            checkpoint_preparer=lambda _: checkpoint,
            runtime_verifier=lambda: {},
            clock=lambda: 1.0,
        )
    assert not (root / lean.PREFLIGHT_ARTIFACT_PATH).exists()


def test_canonical_serializer_rejects_untrusted_test_report(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    checkpoint = root / "data" / "checkpoint"
    checkpoint.mkdir(parents=True)
    report = lean._run_untrusted_preflight(
        root,
        repository_verifier=lambda _: {"head_commit": HEAD},
        contact_verifier=lambda _: CONTACT_SHA256,
        checkpoint_preparer=lambda _: checkpoint,
        runtime_verifier=lambda: {},
        clock=iter([1.0, 2.0]).__next__,
        now=lambda: "2026-07-17T07:00:00Z",
    )
    with pytest.raises(lean.SecGemmaLeanRunnerError) as caught:
        lean._atomic_publish_preflight_report(root, report, checkpoint)
    assert caught.value.code == "untrusted_report_not_sealable"
    assert not (root / lean.PREFLIGHT_ARTIFACT_PATH).exists()


def test_cli_failure_is_fixed_and_cannot_print_private_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "Never Print secret@private-domain.dev"

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise lean.SecGemmaLeanRunnerError("private_contact_invalid")

    monkeypatch.setattr(lean, "run_preflight", fail)
    assert lean.main(["preflight", "--repo-root", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert secret not in captured.err
    assert json.loads(captured.err)["reason_code"] == "private_contact_invalid"


def test_frozen_inventory_covers_complete_v22_source_and_dependency_set() -> None:
    frozen = set(lean.FROZEN_SCIENTIFIC_PATHS)
    assert set(lean.SOURCE_PIN_FILES.values()).issubset(frozen)
    assert set(lean.NEW_SOURCE_FILES.values()).issubset(frozen)
    assert lean.PACKAGE_INIT_SOURCE_PATH in frozen
    assert lean.MARKET_EVIDENCE_SOURCE_PATH in frozen
    joined = "\n".join(frozen)
    for required in (
        "_attempt.py",
        "_features.py",
        "_learner.py",
        "_ledger.py",
        "_metrics.py",
        "_publisher.py",
        "_production.py",
        "_registry.py",
        "_replay.py",
        "_runner.py",
        "_source_verifier.py",
        "_store.py",
        "_vault.py",
        "sec_filing_gemma_market_evidence.py",
    ):
        assert required in joined
