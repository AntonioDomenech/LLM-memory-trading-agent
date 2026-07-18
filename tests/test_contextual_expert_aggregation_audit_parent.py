from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_benchmark import contextual_expert_aggregation_audit_parent as parent


ROOT = Path(__file__).resolve().parents[1]
Error = parent.RejectedParentVerificationError
_HISTORICAL_GITATTRIBUTES_SHA256 = (
    "sha256:cb8b13701b4982d330494139defb21260d0d32ad96fe47c8e5346bb7a3503440"
)
_HISTORICAL_GITATTRIBUTES_BLOB = "1131dc857b36d18c2c9824f6a3c49eb083ec936d"
_CURRENT_GITATTRIBUTES_SHA256 = (
    "sha256:edd61a2af1285ea6a4c3acde2d90f2fea0843a0ca43143af0d06a0fbe26f9498"
)
_CURRENT_GITATTRIBUTES_BLOB = "ff1b62569a04f331566c37885fa82cc6b03bf57c"


def _expected_semantics() -> parent._RegeneratedSemantics:
    return parent._RegeneratedSemantics(
        stage_pass=False,
        status=parent.PARENT_STATUS,
        checkpoint_self_sha256=parent.PARENT_CHECKPOINT_SELF_SHA256,
        gate_report_file_sha256=parent.PARENT_GATE_REPORT_PAYLOAD_SHA256,
        report_file_sha256=parent.PARENT_REPORT_PAYLOAD_SHA256,
        semantic_evidence_sha256=parent.PARENT_SEMANTIC_EVIDENCE_SHA256,
    )


def test_exact_rejected_parent_at_frozen_identity_returns_typed_lock_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    adapted_live_paths: list[str] = []
    original_read_bytes = Path.read_bytes
    original_tracked_file_identity = parent._experiment.tracked_file_identity

    def guarded_read_bytes(path: Path) -> bytes:
        normalized = path.as_posix()
        if "aapl_spy_qqq_through_2023.csv" in normalized:
            raise AssertionError("parent verifier opened the audit input")
        opened.append(normalized)
        return original_read_bytes(path)

    def tracked_file_identity_with_known_gitattributes_drift(
        repo_root: Path,
        path: Path,
        *,
        expected_sha256: str | None = None,
        expected_git_blob: str | None = None,
        require_literal_local_bytes: bool = False,
        allow_crlf_equivalent: bool = False,
    ) -> parent._experiment.TrackedFileIdentity:
        relative = path.relative_to(repo_root).as_posix()
        if relative != ".gitattributes":
            return original_tracked_file_identity(
                repo_root,
                path,
                expected_sha256=expected_sha256,
                expected_git_blob=expected_git_blob,
                require_literal_local_bytes=require_literal_local_bytes,
                allow_crlf_equivalent=allow_crlf_equivalent,
            )

        # The production verifier still rejects this drift. This historical
        # success-path test adapts only the one exact repository-wide EOL rule
        # added after the rejected parent, while pinning both byte identities.
        assert expected_sha256 == _HISTORICAL_GITATTRIBUTES_SHA256
        assert expected_git_blob == _HISTORICAL_GITATTRIBUTES_BLOB
        identity = original_tracked_file_identity(
            repo_root,
            path,
            expected_sha256=_CURRENT_GITATTRIBUTES_SHA256,
            expected_git_blob=_CURRENT_GITATTRIBUTES_BLOB,
            require_literal_local_bytes=require_literal_local_bytes,
            allow_crlf_equivalent=allow_crlf_equivalent,
        )
        assert identity == parent._experiment.TrackedFileIdentity(
            path=".gitattributes",
            sha256=_CURRENT_GITATTRIBUTES_SHA256,
            git_blob=_CURRENT_GITATTRIBUTES_BLOB,
        )
        adapted_live_paths.append(relative)
        return identity

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(
        parent._experiment,
        "tracked_file_identity",
        tracked_file_identity_with_known_gitattributes_drift,
    )
    monkeypatch.setattr(
        parent,
        "_regenerate_rejected_semantics",
        lambda *args, **kwargs: _expected_semantics(),
    )
    evidence = parent.verify_rejected_parent(ROOT)

    assert isinstance(evidence, parent.RejectedParentEvidence)
    assert evidence.verified is True
    assert evidence.stage_pass is False
    assert evidence.status == "REJECTED"
    assert evidence.parent_rejection_remains_final is True
    assert evidence.failed_checks == parent.PARENT_FAILED_CHECKS
    assert len(evidence.exact_payload_names) == 47
    assert len(evidence.checksummed_names) == 48
    assert evidence.semantic_evidence_sha256 == (
        parent.PARENT_SEMANTIC_EVIDENCE_SHA256
    )
    assert len(evidence.historical_dependency_paths) > 20
    assert all(
        value.startswith("agent_benchmark/")
        or value in {".gitattributes", ".gitignore", "requirements.txt"}
        for value in evidence.live_semantic_dependency_paths
    )
    assert opened
    assert not any("aapl_spy_qqq_through_2023.csv" in value for value in opened)
    assert adapted_live_paths
    assert set(adapted_live_paths) == {".gitattributes"}

    lock = evidence.lock_payload()
    assert lock["verified"] is True
    assert lock["stage_pass"] is False
    assert lock["parent_rejection_remains_final"] is True
    assert lock["failed_checks"] == list(parent.PARENT_FAILED_CHECKS)
    assert "bundle" not in lock
    assert "manifest_bytes" not in lock


def test_static_claims_reject_a_parent_relabelled_as_passing() -> None:
    bundle, _, _ = parent._load_exact_rejected_bundle(ROOT)
    changed_manifest = dict(bundle.manifest)
    changed_manifest["stage_pass"] = True
    changed = replace(bundle, manifest=changed_manifest)

    with pytest.raises(Error, match="exact frozen rejected"):
        parent._require_exact_static_rejection(changed)


def test_static_claims_reject_a_different_sole_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _, _ = parent._load_exact_rejected_bundle(ROOT)
    original = parent._json_payload
    actual_gate = original(bundle, parent._GATE_REPORT_FILENAME)
    actual_report = original(bundle, parent._REPORT_FILENAME)

    changed_gate = dict(actual_gate)
    changed_checks = dict(actual_gate["checks"])
    changed_checks[parent.PARENT_FAILED_CHECKS[0]] = True
    changed_checks["base_5bps.active_edge_gt_0_001"] = False
    changed_gate["checks"] = changed_checks
    changed_gate["failed_checks"] = ["base_5bps.active_edge_gt_0_001"]
    changed_report = dict(actual_report)
    changed_report["gate_report"] = changed_gate

    def changed_json_payload(_bundle: object, filename: str) -> dict[str, object]:
        if filename == parent._GATE_REPORT_FILENAME:
            return changed_gate
        if filename == parent._REPORT_FILENAME:
            return changed_report
        return original(bundle, filename)

    monkeypatch.setattr(parent, "_json_payload", changed_json_payload)
    with pytest.raises(Error, match="35-of-36"):
        parent._require_exact_static_rejection(bundle)


def test_static_claims_reject_checkpoint_self_hash_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _, _ = parent._load_exact_rejected_bundle(ROOT)
    original = parent._json_payload

    def changed_json_payload(_bundle: object, filename: str) -> dict[str, object]:
        value = original(bundle, filename)
        if filename == parent._CHECKPOINT_FILENAME:
            value = dict(value)
            value["checkpoint_sha256"] = "sha256:" + "0" * 64
        return value

    monkeypatch.setattr(parent, "_json_payload", changed_json_payload)
    with pytest.raises(Error, match="checkpoint self-hash"):
        parent._require_exact_static_rejection(bundle)


def test_regenerated_semantics_must_equal_the_preregistered_hash() -> None:
    changed = replace(
        _expected_semantics(), semantic_evidence_sha256="sha256:" + "0" * 64
    )
    with pytest.raises(Error, match="exact frozen v2 rejection"):
        parent._require_exact_regenerated_semantics(changed)


def test_manifest_byte_tamper_fails_before_generic_bundle_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / parent.PARENT_DIRECTORY
    directory.mkdir(parents=True)
    source = ROOT / parent.PARENT_DIRECTORY
    (directory / "stage_manifest.json").write_bytes(
        (source / "stage_manifest.json").read_bytes() + b" "
    )
    (directory / "checksums.json").write_bytes(
        (source / "checksums.json").read_bytes()
    )
    monkeypatch.setattr(
        parent._experiment,
        "verify_exact_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generic verifier must not run after seal tamper")
        ),
    )

    with pytest.raises(Error, match="manifest or checksum bytes changed"):
        parent._load_exact_rejected_bundle(tmp_path)


def test_historical_dependency_content_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _, _ = parent._load_exact_rejected_bundle(ROOT)
    original = parent._git_bytes

    def changed_git_bytes(repo_root: Path, *args: str) -> bytes:
        if args[:1] == ("show",) and args[1].endswith(":.gitattributes"):
            return b"tampered historical dependency\n"
        return original(repo_root, *args)

    monkeypatch.setattr(parent, "_git_bytes", changed_git_bytes)
    with pytest.raises(Error, match="Historical v2 dependency differs"):
        parent._require_parent_git_and_dependency_continuity(ROOT, bundle)


def test_a_passing_semantic_regeneration_is_never_authorized() -> None:
    changed = replace(_expected_semantics(), stage_pass=True, status="PASSED")
    with pytest.raises(Error, match="exact frozen v2 rejection"):
        parent._require_exact_regenerated_semantics(changed)
