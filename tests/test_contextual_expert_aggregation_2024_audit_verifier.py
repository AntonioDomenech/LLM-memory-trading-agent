from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_benchmark import contextual_expert_aggregation_2024_audit_artifacts as artifacts
from agent_benchmark import contextual_expert_aggregation_2024_audit_verifier as verifier


Error = verifier.ContextualExpertAggregation2024AuditVerificationError
_DIGEST = "sha256:" + "1" * 64


def _runtime(*, provisional: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "runtime_evidence_schema_version": 1,
        "contract_version": artifacts.CONTRACT_VERSION,
        "run_id": artifacts.AUDIT_RUN_ID,
        "stage": artifacts.AUDIT_STAGE,
        "clock_name": "time.monotonic",
        "command_start_monotonic": 100.0,
        "stage_deadline_seconds": 1800.0,
        "finalization_reserve_seconds": 5.0,
        "samples": {
            "preseal": 10.0,
            "post_private_verify": None if provisional else 20.0,
            "prepromotion": None if provisional else 30.0,
        },
        "stage_internal_deadline_pass": not provisional,
        "final_commit_sample_location": "stage_stdout_and_preservation_evidence",
        "network_access": False,
        "news_access": False,
        "llm_calls": 0,
        "api_calls": 0,
        "external_model_calls": 0,
        "external_cost_usd": 0.0,
    }
    value["runtime_evidence_sha256"] = verifier._sha256_json(value)
    return value


def _payloads() -> dict[str, bytes]:
    payloads = {
        name: artifacts.canonical_json_line_bytes(
            {"schema_version": 1, "synthetic_fixture": name}
        )
        for name in artifacts.PAYLOAD_FILE_ORDER
    }
    payloads[".gitattributes"] = artifacts.GIT_ATTRIBUTES_BYTES
    return payloads


def _manifest_fields() -> dict[str, Any]:
    return {
        "manifest_schema_version": artifacts.MANIFEST_SCHEMA_VERSION,
        "contract_version": artifacts.CONTRACT_VERSION,
        "verifier_id": artifacts.VERIFIER_ID,
        "run_id": artifacts.AUDIT_RUN_ID,
        "stage": artifacts.AUDIT_STAGE,
        "status": "POST_HOC_FROZEN_POLICY_2024_REPLICATION_PASS",
        "stage_pass": True,
        "evidence_classification": "synthetic_verifier_fixture",
        "preregistration_commit": artifacts.PREREGISTRATION_COMMIT,
        "git_identity": {"synthetic_fixture": True},
        "dependency_identity_sha256": _DIGEST,
        "artifact_schema_registry_sha256": (
            artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
        ),
        "parent_manifest_file_sha256": artifacts.PARENT_MANIFEST_FILE_SHA256,
        "parent_manifest_self_sha256": artifacts.PARENT_MANIFEST_SELF_SHA256,
        "parent_checkpoint_file_sha256": artifacts.PARENT_CHECKPOINT_FILE_SHA256,
        "parent_checkpoint_self_sha256": artifacts.PARENT_CHECKPOINT_SELF_SHA256,
        "receipt_file_sha256": _DIGEST,
        "input_raw_sha256": _DIGEST,
        "input_canonical_sha256": _DIGEST,
        "attempt_lock_file_sha256": _DIGEST,
        "runtime_cost_evidence_file_sha256": _DIGEST,
        "gate_report_file_sha256": _DIGEST,
        "learning_classification": "unexercised",
        "later_market_data_accessed": False,
        "prior_2024_artifact_accessed": False,
        "external_cost_usd": 0.0,
    }


def _configure_synthetic_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    final_relative = Path("synthetic-final-bundle")
    # These failure-path tests exercise both modes against the same compact
    # synthetic location; production constants remain distinct.
    private_relative = final_relative
    marker_relative = Path("synthetic-success-marker.json")
    for module in (artifacts, verifier):
        monkeypatch.setattr(
            module, "OUTPUT_DIRECTORY", final_relative, raising=False
        )
        monkeypatch.setattr(
            module, "PRIVATE_DIRECTORY", private_relative, raising=False
        )
        monkeypatch.setattr(
            module, "SUCCESS_MARKER_PATH", marker_relative, raising=False
        )
    return tmp_path / final_relative, tmp_path / marker_relative


def _write_synthetic_bundle(directory: Path) -> artifacts.BundleMetadata:
    payloads = _payloads()
    metadata = artifacts.build_bundle_metadata(
        payloads, manifest_fields=_manifest_fields()
    )
    artifacts.write_private_bundle(
        directory, payloads=payloads, metadata=metadata
    )
    return metadata


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in sorted(directory.iterdir(), key=lambda path: path.name)
        if item.is_file()
    }


def _verify_private(
    directory: Path, *, repo_root: Path, manifest_sha256: str
) -> dict[str, Any]:
    return verifier.verify_private_audit_bundle(
        directory,
        repo_root=repo_root,
        expected_manifest_sha256=manifest_sha256,
        allow_provisional_runtime=True,
    )


def test_private_verifier_rejects_unauthorized_location_before_bundle_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_synthetic_paths(tmp_path, monkeypatch)
    unauthorized = tmp_path / "synthetic-copy"
    unauthorized.mkdir()

    with pytest.raises(Error, match="location|authorized"):
        _verify_private(
            unauthorized,
            repo_root=tmp_path,
            manifest_sha256=_DIGEST,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_private_verifier_rejects_nonexact_bundle_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    directory, _ = _configure_synthetic_paths(tmp_path, monkeypatch)
    metadata = _write_synthetic_bundle(directory)
    if mutation == "missing":
        (directory / artifacts.BUNDLE_FILE_ORDER[-1]).unlink()
    else:
        (directory / "synthetic-extra-entry").write_bytes(b"extra")

    before = _snapshot(directory)
    with pytest.raises(Error, match="inventory|43"):
        _verify_private(
            directory,
            repo_root=tmp_path,
            manifest_sha256=metadata.manifest["manifest_sha256"],
        )
    assert _snapshot(directory) == before


def test_private_verifier_rejects_payload_hash_tampering_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _ = _configure_synthetic_paths(tmp_path, monkeypatch)
    metadata = _write_synthetic_bundle(directory)
    target = directory / artifacts.PAYLOAD_FILE_ORDER[-1]
    target.write_bytes(target.read_bytes() + b"tampered")
    before = _snapshot(directory)

    with pytest.raises(Error, match="checksum|hash|payload"):
        _verify_private(
            directory,
            repo_root=tmp_path,
            manifest_sha256=metadata.manifest["manifest_sha256"],
        )
    assert _snapshot(directory) == before


def test_private_verifier_rejects_self_consistent_manifest_schema_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _ = _configure_synthetic_paths(tmp_path, monkeypatch)
    metadata = _write_synthetic_bundle(directory)
    manifest = dict(metadata.manifest)
    manifest["synthetic_unregistered_claim"] = True
    unsigned = {
        name: value
        for name, value in manifest.items()
        if name != "manifest_sha256"
    }
    manifest["manifest_sha256"] = artifacts._sha256_json(unsigned)
    manifest_payload = artifacts.canonical_json_line_bytes(manifest)
    (directory / "stage_manifest.json").write_bytes(manifest_payload)
    checksums = dict(metadata.checksums)
    checksums["stage_manifest.json"] = artifacts._sha256_bytes(manifest_payload)
    (directory / "checksums.json").write_bytes(
        artifacts.checksums_bytes(checksums)
    )
    before = _snapshot(directory)

    with pytest.raises(Error, match="schema|manifest|inventory"):
        _verify_private(
            directory,
            repo_root=tmp_path,
            manifest_sha256=manifest["manifest_sha256"],
        )
    assert _snapshot(directory) == before


def _write_success_marker(
    path: Path, metadata: artifacts.BundleMetadata
) -> dict[str, Any]:
    marker = artifacts.build_success_marker(
        {
            "marker_schema_version": artifacts.MARKER_SCHEMA_VERSION,
            "contract_version": artifacts.CONTRACT_VERSION,
            "run_id": artifacts.AUDIT_RUN_ID,
            "stage": artifacts.AUDIT_STAGE,
            "preregistration_commit": artifacts.PREREGISTRATION_COMMIT,
            "git_commit": "a" * 40,
            "attempt_lock_file_sha256": _DIGEST,
            "final_relative_path": artifacts.OUTPUT_DIRECTORY.as_posix(),
            "final_inventory_sha256": artifacts.FINAL_INVENTORY_SHA256,
            "stage_manifest_file_sha256": artifacts._sha256_bytes(
                metadata.manifest_bytes
            ),
            "stage_manifest_self_sha256": metadata.manifest["manifest_sha256"],
            "checksums_file_sha256": artifacts._sha256_bytes(
                metadata.checksums_bytes
            ),
            "marker_preparation_elapsed_seconds": 1.0,
            "stage_deadline_seconds": artifacts.STAGE_DEADLINE_SECONDS,
            "marker_preparation_deadline_pass": True,
            "external_cost_usd": 0.0,
        }
    )
    path.write_bytes(artifacts.success_marker_bytes(marker))
    return marker


def test_promoted_verifier_rejects_success_marker_self_hash_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, marker_path = _configure_synthetic_paths(tmp_path, monkeypatch)
    metadata = _write_synthetic_bundle(directory)
    marker = _write_success_marker(marker_path, metadata)
    marker["marker_sha256"] = "sha256:" + "0" * 64
    marker_path.write_bytes(artifacts.canonical_json_line_bytes(marker))
    before_bundle = _snapshot(directory)
    before_marker = marker_path.read_bytes()

    with pytest.raises(Error, match="marker|self-hash|hash"):
        verifier.verify_audit(repo_root=tmp_path, clock=lambda: 0.0)
    assert _snapshot(directory) == before_bundle
    assert marker_path.read_bytes() == before_marker


def test_verify_stage_rejects_every_noncontract_stage() -> None:
    with pytest.raises(Error, match="stage"):
        verifier.verify_stage("synthetic-other-stage")


def test_runtime_evidence_accepts_only_final_or_exact_provisional_state() -> None:
    final = _runtime()
    provisional = _runtime(provisional=True)

    assert verifier._runtime_payload(final) == final
    assert verifier._runtime_payload(
        provisional, allow_provisional_runtime=True
    ) == provisional
    with pytest.raises(Error, match="runtime|float"):
        verifier._runtime_payload(provisional)

    partial = _runtime(provisional=True)
    partial["samples"]["post_private_verify"] = 20.0
    partial["runtime_evidence_sha256"] = verifier._sha256_json(
        {
            key: value
            for key, value in partial.items()
            if key != "runtime_evidence_sha256"
        }
    )
    with pytest.raises(Error, match="provisional"):
        verifier._runtime_payload(partial, allow_provisional_runtime=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["samples"].update(prepromotion=20.0), "increasing"),
        (lambda value: value["samples"].update(prepromotion=1795.0), "reserve"),
        (lambda value: value.update(network_access=True), "external"),
        (lambda value: value.update(llm_calls=1), "model|API"),
        (lambda value: value.update(runtime_evidence_sha256=_DIGEST), "self-hash"),
    ],
)
def test_runtime_evidence_rejects_deadline_external_use_and_hash_tampering(
    mutate: Any, message: str
) -> None:
    value = _runtime()
    mutate(value)
    if "runtime_evidence_sha256" not in value or value[
        "runtime_evidence_sha256"
    ] != _DIGEST:
        value["runtime_evidence_sha256"] = verifier._sha256_json(
            {
                key: item
                for key, item in value.items()
                if key != "runtime_evidence_sha256"
            }
        )
    with pytest.raises(Error, match=message):
        verifier._runtime_payload(value)


def test_verification_clock_uses_strict_1800_second_bound() -> None:
    values = iter((100.0, 101.0, 1899.999, 1900.0))
    clock = verifier.VerificationClock.start(now=lambda: next(values))
    assert clock.check("preverify") == 1.0
    assert clock.check("postreconstruction") == pytest.approx(1799.999)
    with pytest.raises(Error, match="deadline"):
        clock.check("too_late")


def test_deep_schema_validator_rejects_unknown_nested_fields() -> None:
    schema = {
        "type": "object",
        "fields": {
            "identity": "sha256",
            "nested": {
                "type": "object",
                "fields": {"passed": "bool"},
            },
        },
    }
    verifier._validate_schema(
        {"identity": _DIGEST, "nested": {"passed": True}},
        schema,
        field_name="synthetic",
    )
    with pytest.raises(Error, match="inventory"):
        verifier._validate_schema(
            {
                "identity": _DIGEST,
                "nested": {"passed": True, "unregistered": True},
            },
            schema,
            field_name="synthetic",
        )


def test_regenerated_payload_comparison_is_exact_and_byte_based() -> None:
    files = _payloads()
    bundle = SimpleNamespace(files=files)
    verifier._compare_regenerated_payloads(bundle, dict(files))

    wrong = dict(files)
    wrong[artifacts.PAYLOAD_FILE_ORDER[-1]] += b"tampered"
    with pytest.raises(Error, match="differs from regeneration"):
        verifier._compare_regenerated_payloads(bundle, wrong)
    missing = dict(files)
    missing.pop(artifacts.PAYLOAD_FILE_ORDER[-1])
    with pytest.raises(Error, match="41-file inventory"):
        verifier._compare_regenerated_payloads(bundle, missing)
