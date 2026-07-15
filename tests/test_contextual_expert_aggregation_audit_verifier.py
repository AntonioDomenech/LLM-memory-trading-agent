from __future__ import annotations

import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_audit_runner as runner
from agent_benchmark import contextual_expert_aggregation_audit_verifier as verifier
from agent_benchmark import contextual_expert_aggregation_experiment as experiment


Error = verifier.ContextualExpertAggregationAuditVerificationError


def _runtime() -> dict[str, Any]:
    return {
        "runtime_evidence_schema_version": 1,
        "stage": verifier.AUDIT_STAGE,
        "seconds_before_seal": 12.5,
        "strict_limit_seconds": 3600.0,
        "within_limit_before_seal": True,
        "monotonic_clock": True,
        "strict_limit_rechecked_immediately_before_promotion": True,
        "network_access": False,
        "news_access": False,
        "llm_calls": 0,
        "api_calls": 0,
        "external_model_calls": 0,
        "external_cost_usd": 0.0,
    }


def test_bundle_location_rejects_every_noncontract_directory(tmp_path: Path) -> None:
    unauthorized = tmp_path / "copied-or-renamed-result"
    unauthorized.mkdir()
    with pytest.raises(Error, match="unauthorized bundle location"):
        verifier._bundle_location(unauthorized, repo_root=tmp_path)


def test_standalone_verifier_hardens_final_path_before_manifest_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        verifier,
        "_bundle_location",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            Error("redirected final bundle")
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manifest bytes must not be opened before hardening")
        ),
    )

    with pytest.raises(Error, match="redirected final bundle"):
        verifier.verify_audit(tmp_path)


def test_runtime_payload_accepts_only_the_exact_canonical_evidence() -> None:
    value = _runtime()
    assert verifier._runtime_payload(value) == value


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("runtime_evidence_schema_version", True),
        ("seconds_before_seal", "12.5"),
        ("seconds_before_seal", 12),
        ("strict_limit_seconds", 3600),
        ("within_limit_before_seal", 1),
        ("llm_calls", False),
        ("api_calls", 0.0),
        ("external_model_calls", False),
        ("external_cost_usd", 0),
    ],
)
def test_runtime_payload_rejects_type_confusion(
    field: str, replacement: Any
) -> None:
    value = _runtime()
    value[field] = replacement
    with pytest.raises(Error, match="runtime"):
        verifier._runtime_payload(value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seconds_before_seal", -0.01),
        ("seconds_before_seal", 3600.0),
        ("seconds_before_seal", float("nan")),
        ("strict_limit_seconds", 3599.0),
        ("external_cost_usd", 0.01),
        ("network_access", True),
        ("news_access", True),
        ("llm_calls", 1),
        ("api_calls", 1),
        ("external_model_calls", 1),
    ],
)
def test_runtime_payload_rejects_values_outside_the_contract(
    field: str, replacement: Any
) -> None:
    value = _runtime()
    value[field] = replacement
    with pytest.raises(Error, match="runtime"):
        verifier._runtime_payload(value)


def test_runtime_payload_rejects_extra_inventory() -> None:
    value = _runtime()
    value["unregistered_clock_claim"] = True
    with pytest.raises(Error, match="wrong exact inventory"):
        verifier._runtime_payload(value)


def _install_git_reconstruction(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> dict[str, Any]:
    input_relative = runner.AUDIT_INPUT_PATH.as_posix()
    branch = runner.EXPECTED_BRANCH
    commit = "a" * 40
    input_blob = runner.AUDIT_INPUT_SPEC.git_blob
    origin_url = sorted(experiment.EXPECTED_ORIGIN_URLS)[0]
    responses = {
        ("symbolic-ref", "--quiet", "--short", "HEAD"): branch,
        ("rev-parse", "HEAD"): commit,
        (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ): f"origin/{branch}",
        ("rev-parse", "@{upstream}"): commit,
        ("remote", "get-url", "origin"): origin_url,
        ("rev-parse", f"HEAD:{input_relative}"): input_blob,
    }

    def git_text(observed_root: Path, *args: str) -> str:
        assert observed_root == root.resolve()
        return responses[tuple(args)]

    dependencies = {
        "agent_benchmark/contextual_expert_aggregation_audit_verifier.py": {
            "head_object_id": "c" * 40,
            "index_object_id": "c" * 40,
        }
    }
    monkeypatch.setattr(runner, "_git_text", git_text)
    monkeypatch.setattr(
        runner,
        "_tracked_index_object_id",
        lambda observed_root, path: (
            input_blob
            if observed_root == root.resolve() and path == input_relative
            else pytest.fail("unexpected tracked-index lookup")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_dependency_identity",
        lambda observed_root: (
            dependencies
            if observed_root == root.resolve()
            else pytest.fail("unexpected dependency lookup")
        ),
    )
    return {
        "branch": branch,
        "commit": commit,
        "upstream": f"origin/{branch}",
        "upstream_commit": commit,
        "origin_url": origin_url,
        "origin_repository": experiment.EXPECTED_ORIGIN_REPOSITORY,
        "head_equals_upstream": True,
        "dirty": False,
        "cleanliness_scope": "all_paths_except_unopened_input",
        "prelock_input_worktree_bytes_opened": False,
        "input_path": input_relative,
        "input_head_object_id": input_blob,
        "input_index_object_id": input_blob,
        "input_head_index_equal": True,
        "tracked_dependency_identity": dependencies,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def test_prelock_git_identity_is_reconstructed_field_for_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _install_git_reconstruction(monkeypatch, tmp_path)
    assert verifier._validated_prelock_git_identity(tmp_path, expected) == expected


def test_prelock_git_identity_rejects_extra_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = _install_git_reconstruction(monkeypatch, tmp_path)
    expected["postlock_worktree_claim"] = "not authorized"
    with pytest.raises(Error, match="wrong exact inventory"):
        verifier._validated_prelock_git_identity(tmp_path, expected)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("branch", "codex/different-branch"),
        ("input_index_object_id", "d" * 40),
        ("prelock_input_worktree_bytes_opened", True),
    ],
)
def test_prelock_git_identity_rejects_tampered_reconstructed_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: Any,
) -> None:
    expected = _install_git_reconstruction(monkeypatch, tmp_path)
    expected[field] = replacement
    with pytest.raises(Error, match="differs from reconstruction"):
        verifier._validated_prelock_git_identity(tmp_path, expected)


class _Deadline:
    def __init__(self) -> None:
        self.phases: list[str] = []

    def check(self, phase: str) -> float:
        self.phases.append(phase)
        return float(len(self.phases))


def _install_semantic_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, _Deadline, dict[str, bytes], runner.AuditLockEvidence]:
    root = tmp_path.resolve()
    # Keep the synthetic path below the Windows MAX_PATH boundary while still
    # exercising the verifier's exact authorized-location calculation.
    monkeypatch.setattr(verifier, "OUTPUT_DIRECTORY", Path("sealed-audit"))
    monkeypatch.setattr(verifier, "OUTPUT_PARENT", Path("sealed-parent"))
    bundle_path = root / verifier.OUTPUT_DIRECTORY
    bundle_path.mkdir(parents=True)

    prelock = {"commit": "a" * 40, "identity": "synthetic-prelock"}
    parent_verification = {
        "verified": True,
        "status": "REJECTED",
        "semantic_evidence_sha256": "sha256:" + "1" * 64,
    }
    runtime = _runtime()
    lock_bytes = experiment.pretty_json_bytes(
        {"schema_version": 1, "synthetic_lock": True}
    )
    lock = runner.AuditLockEvidence(
        path=runner.ATTEMPT_LOCK_PATH.as_posix(),
        sha256=experiment.sha256_bytes(lock_bytes),
        content={
            "git_commit": prelock["commit"],
            "prelock_git_identity_sha256": runner._sha256_json(prelock),
            "parent_verification_sha256": runner._sha256_json(
                parent_verification
            ),
        },
        bytes_value=lock_bytes,
        parent_directory_fsync_supported=True,
    )
    gate_report = {"passed": True, "checks": {"synthetic": True}}
    integrity = {"synthetic_integrity": True}
    metrics = {"synthetic_metric": 1.25}
    checkpoint = {"checkpoint_sha256": "sha256:" + "2" * 64}
    report = {
        "status": "AUDIT_PASSED",
        "historical_policy_candidate_for_2024_audit": True,
        "learning_candidate_for_2024_audit": True,
    }
    payloads = {
        name: f"deterministic:{name}\n".encode("ascii")
        for name in sorted(verifier.AUDIT_PAYLOAD_NAMES)
    }
    payloads["rejected_parent_verification.json"] = experiment.pretty_json_bytes(
        parent_verification
    )
    payloads[verifier.ATTEMPT_LOCK_FILENAME] = lock_bytes
    payloads["audit_runtime_cost_evidence.json"] = experiment.pretty_json_bytes(
        runtime
    )
    for name, value in payloads.items():
        (bundle_path / name).write_bytes(value)

    manifest = experiment.self_hashed_manifest(
        {
            "contract_version": verifier.CONTRACT_VERSION,
            "stage": verifier.AUDIT_STAGE,
            "stage_pass": True,
            "run_id": verifier.AUDIT_RUN_ID,
            "evidence_classification": runner.EVIDENCE_CLASSIFICATION,
            "git_identity": prelock,
            "parent_manifest_sha256": runner.PARENT_MANIFEST_SELF_SHA256,
            "parent_semantic_evidence_sha256": (
                runner.PARENT_SEMANTIC_EVIDENCE_SHA256
            ),
            "input_raw_sha256": runner.AUDIT_INPUT_SPEC.raw_sha256,
            "input_canonical_sha256": runner.AUDIT_INPUT_SPEC.canonical_sha256,
            "attempt_lock_sha256": lock.sha256,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "report_status": report["status"],
            "historical_policy_candidate_for_2024_audit": True,
            "learning_candidate_for_2024_audit": True,
            "parent_rejection_remains_final": True,
            "post_2023_market_values_accessed": False,
            "runtime_cost_evidence": runtime,
            "payload_sha256": {
                name: experiment.sha256_bytes(payloads[name])
                for name in sorted(payloads)
            },
        }
    )
    bundle = experiment.VerifiedBundle(
        directory=bundle_path,
        manifest_path=bundle_path / "stage_manifest.json",
        manifest=manifest,
        payload_sha256=manifest["payload_sha256"],
        checksums={},
    )
    development = SimpleNamespace(parent_verification=parent_verification)
    snapshot = object()
    input_provenance = {"audit_provenance_sha256": "sha256:" + "3" * 64}
    computation = object()

    monkeypatch.setattr(
        experiment,
        "_verify_bundle_identity",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        verifier,
        "_validated_prelock_git_identity",
        lambda observed_root, value: (
            prelock
            if observed_root == root and value == prelock
            else pytest.fail("unexpected pre-lock identity verification")
        ),
    )
    monkeypatch.setattr(
        runner,
        "prepare_rejected_development",
        lambda observed_root, deadline: (
            development
            if observed_root == root
            else pytest.fail("unexpected development root")
        ),
    )
    monkeypatch.setattr(runner, "read_attempt_lock", lambda observed_root: lock)
    monkeypatch.setattr(
        runner,
        "load_locked_audit_snapshot",
        lambda *args, **kwargs: (snapshot, input_provenance),
    )
    monkeypatch.setattr(
        runner,
        "compute_audit_continuation",
        lambda *args, **kwargs: computation,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_audit",
        lambda observed: (integrity, gate_report, metrics)
        if observed is computation
        else pytest.fail("unexpected computation"),
    )
    monkeypatch.setattr(
        runner,
        "build_audit_payloads",
        lambda **kwargs: (payloads, checkpoint, report),
    )
    deadline = _Deadline()
    return bundle_path, manifest["manifest_sha256"], deadline, payloads, lock


def test_verifier_rejects_sealed_lock_byte_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest_sha, deadline, _, _ = _install_semantic_regeneration(
        tmp_path, monkeypatch
    )
    (bundle / verifier.ATTEMPT_LOCK_FILENAME).write_bytes(b"different lock bytes")
    with pytest.raises(Error, match="persistent lock"):
        verifier.verify_private_audit_bundle(
            bundle,
            repo_root=tmp_path,
            expected_manifest_sha256=manifest_sha,
            deadline=deadline,
        )


def test_verifier_rejects_payload_byte_mismatch_after_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest_sha, deadline, _, _ = _install_semantic_regeneration(
        tmp_path, monkeypatch
    )
    (bundle / "audit_metrics.json").write_bytes(b"tampered after seal\n")
    with pytest.raises(Error, match="payload differs from regeneration: audit_metrics.json"):
        verifier.verify_private_audit_bundle(
            bundle,
            repo_root=tmp_path,
            expected_manifest_sha256=manifest_sha,
            deadline=deadline,
        )


def test_successful_mocked_regeneration_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, manifest_sha, deadline, payloads, lock = _install_semantic_regeneration(
        tmp_path, monkeypatch
    )
    first = verifier.verify_private_audit_bundle(
        bundle,
        repo_root=tmp_path,
        expected_manifest_sha256=manifest_sha,
        deadline=deadline,
    )
    second = verifier.verify_private_audit_bundle(
        bundle,
        repo_root=tmp_path,
        expected_manifest_sha256=manifest_sha,
        deadline=deadline,
    )
    assert first == second
    assert first["verified"] is True
    assert first["stage_pass"] is True
    assert first["manifest_sha256"] == manifest_sha
    assert first["attempt_lock_sha256"] == lock.sha256
    assert first["exact_payload_count"] == len(payloads)
    assert first["semantic_evidence_sha256"].startswith("sha256:")
    assert deadline.phases == [
        "audit verifier entry",
        "after independently preparing rejected parent",
        "after independent audit semantic regeneration",
        "audit verifier entry",
        "after independently preparing rejected parent",
        "after independent audit semantic regeneration",
    ]
