from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_benchmark import sec_audit_artifact as artifact
from agent_benchmark.sec_audit_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    CHECKSUMS_FILENAME,
    METADATA_FILENAME,
    SecAuditArtifactError,
    seal_artifact,
    verify_artifact,
)


SOURCE_COMMIT = "a" * 40
CONTRACT_VERSION = "aapl-sec-point-in-time-audit-v1"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sealed(tmp_path: Path, name: str = "sealed") -> Path:
    destination = tmp_path / name
    seal_artifact(
        destination,
        {"sample.json": b'{"ok":true}\n', "raw.txt": b"raw\x00bytes"},
        source_commit=SOURCE_COMMIT,
        audit_contract_version=CONTRACT_VERSION,
    )
    return destination


def _checksums_hash(destination: Path) -> str:
    payload = (destination / CHECKSUMS_FILENAME).read_bytes()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_seal_is_deterministic_and_records_provenance(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = seal_artifact(
        first,
        {"z.bin": b"last", "a.bin": b"first"},
        source_commit=SOURCE_COMMIT.upper(),
        audit_contract_version=CONTRACT_VERSION,
    )
    second_result = seal_artifact(
        second,
        {"a.bin": b"first", "z.bin": b"last"},
        source_commit=SOURCE_COMMIT,
        audit_contract_version=CONTRACT_VERSION,
    )

    assert (first / CHECKSUMS_FILENAME).read_bytes() == (
        second / CHECKSUMS_FILENAME
    ).read_bytes()
    assert (first / METADATA_FILENAME).read_bytes() == (
        second / METADATA_FILENAME
    ).read_bytes()
    metadata = json.loads((first / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata == {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "audit_contract_version": CONTRACT_VERSION,
        "source_commit": SOURCE_COMMIT,
    }
    checksums = json.loads((first / CHECKSUMS_FILENAME).read_text(encoding="utf-8"))
    assert set(checksums) == {"a.bin", "z.bin", METADATA_FILENAME}
    assert checksums["a.bin"] == hashlib.sha256(b"first").hexdigest()
    assert first_result.checksums == checksums
    assert first_result.checksums_sha256 == second_result.checksums_sha256
    assert verify_artifact(
        first, expected_checksums_sha256=first_result.checksums_sha256
    ) == first_result


@pytest.mark.parametrize("mutation", ("extra", "missing", "tampered"))
def test_verifier_rejects_extra_missing_and_tampered_files(
    tmp_path: Path, mutation: str
) -> None:
    destination = _sealed(tmp_path)
    expected_hash = _checksums_hash(destination)
    if mutation == "extra":
        (destination / "extra.bin").write_bytes(b"extra")
        match = "file set is not exact"
    elif mutation == "missing":
        (destination / "raw.txt").unlink()
        match = "file set is not exact"
    else:
        (destination / "raw.txt").write_bytes(b"changed")
        match = "checksum mismatch"

    with pytest.raises(SecAuditArtifactError, match=match):
        verify_artifact(
            destination, expected_checksums_sha256=expected_hash
        )


def test_verifier_rejects_tampered_metadata_and_manifest(tmp_path: Path) -> None:
    metadata_artifact = _sealed(tmp_path, "metadata-tamper")
    metadata_hash = _checksums_hash(metadata_artifact)
    metadata = json.loads(
        (metadata_artifact / METADATA_FILENAME).read_text(encoding="utf-8")
    )
    metadata["source_commit"] = "b" * 40
    (metadata_artifact / METADATA_FILENAME).write_bytes(_json_bytes(metadata))
    with pytest.raises(SecAuditArtifactError, match="checksum mismatch"):
        verify_artifact(
            metadata_artifact, expected_checksums_sha256=metadata_hash
        )

    manifest_artifact = _sealed(tmp_path, "manifest-tamper")
    manifest_hash = _checksums_hash(manifest_artifact)
    manifest = json.loads(
        (manifest_artifact / CHECKSUMS_FILENAME).read_text(encoding="utf-8")
    )
    manifest["raw.txt"] = "0" * 64
    (manifest_artifact / CHECKSUMS_FILENAME).write_bytes(_json_bytes(manifest))
    with pytest.raises(SecAuditArtifactError, match="checksum mismatch"):
        verify_artifact(
            manifest_artifact, expected_checksums_sha256=manifest_hash
        )


def test_external_manifest_hash_rejects_coordinated_payload_rehash(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "coordinated-tamper"
    sealed = seal_artifact(
        destination,
        {"report.json": b'{"overall_pass":false}\n'},
        source_commit=SOURCE_COMMIT,
        audit_contract_version=CONTRACT_VERSION,
    )
    changed = b'{"overall_pass":true}\n'
    (destination / "report.json").write_bytes(changed)
    checksums = json.loads(
        (destination / CHECKSUMS_FILENAME).read_text(encoding="utf-8")
    )
    checksums["report.json"] = hashlib.sha256(changed).hexdigest()
    (destination / CHECKSUMS_FILENAME).write_bytes(_json_bytes(checksums))

    with pytest.raises(SecAuditArtifactError, match="manifest does not match"):
        verify_artifact(
            destination,
            expected_checksums_sha256=sealed.checksums_sha256,
        )


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../escape.bin",
        "nested/file.bin",
        r"nested\file.bin",
        "/absolute.bin",
        r"C:\absolute.bin",
        "C:drive-relative.bin",
        ".",
        "..",
        "checksums.json",
        "ARTIFACT_METADATA.JSON",
    ),
)
def test_seal_rejects_path_traversal_and_reserved_names_before_writing(
    tmp_path: Path, unsafe_name: str
) -> None:
    destination = tmp_path / "sealed"
    with pytest.raises(SecAuditArtifactError):
        seal_artifact(
            destination,
            {unsafe_name: b"escape"},
            source_commit=SOURCE_COMMIT,
            audit_contract_version=CONTRACT_VERSION,
        )
    assert not destination.exists()
    assert not (tmp_path.parent / "escape.bin").exists()


def test_verifier_rejects_path_traversal_in_manifest_and_nested_entries(
    tmp_path: Path,
) -> None:
    destination = _sealed(tmp_path, "manifest-path")
    destination_hash = _checksums_hash(destination)
    checksums = json.loads(
        (destination / CHECKSUMS_FILENAME).read_text(encoding="utf-8")
    )
    checksums["../outside.bin"] = checksums.pop("raw.txt")
    (destination / CHECKSUMS_FILENAME).write_bytes(_json_bytes(checksums))
    with pytest.raises(SecAuditArtifactError, match="safe relative component"):
        verify_artifact(
            destination, expected_checksums_sha256=destination_hash
        )

    nested_artifact = _sealed(tmp_path, "nested-entry")
    nested_hash = _checksums_hash(nested_artifact)
    (nested_artifact / "nested").mkdir()
    (nested_artifact / "nested" / "file.bin").write_bytes(b"unexpected")
    with pytest.raises(SecAuditArtifactError, match="flat regular files"):
        verify_artifact(nested_artifact, expected_checksums_sha256=nested_hash)


def test_verifier_can_bind_external_identity(tmp_path: Path) -> None:
    destination = _sealed(tmp_path)
    expected_hash = _checksums_hash(destination)
    verified = verify_artifact(
        destination,
        expected_checksums_sha256=expected_hash,
        expected_source_commit=SOURCE_COMMIT.upper(),
        expected_audit_contract_version=CONTRACT_VERSION,
    )
    verify_artifact(
        destination, expected_checksums_sha256=verified.checksums_sha256
    )

    with pytest.raises(SecAuditArtifactError, match="source commit"):
        verify_artifact(
            destination,
            expected_checksums_sha256=expected_hash,
            expected_source_commit="b" * 40,
        )
    with pytest.raises(SecAuditArtifactError, match="contract version"):
        verify_artifact(
            destination,
            expected_checksums_sha256=expected_hash,
            expected_audit_contract_version="different-contract-v1",
        )
    with pytest.raises(SecAuditArtifactError, match="manifest does not match"):
        verify_artifact(
            destination, expected_checksums_sha256="sha256:" + "0" * 64
        )


def test_seal_rejects_existing_destination_without_modifying_it(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_bytes(b"keep")

    with pytest.raises(SecAuditArtifactError, match="already exists"):
        seal_artifact(
            destination,
            {"new.txt": b"new"},
            source_commit=SOURCE_COMMIT,
            audit_contract_version=CONTRACT_VERSION,
        )
    assert sentinel.read_bytes() == b"keep"
    assert list(destination.iterdir()) == [sentinel]


def test_failed_seal_never_promotes_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "atomic"
    real_write = artifact._write_bytes

    def fail_during_payload_write(path: Path, payload: bytes) -> None:
        if path.name == "b.bin":
            raise OSError("injected write failure")
        real_write(path, payload)

    monkeypatch.setattr(artifact, "_write_bytes", fail_during_payload_write)
    with pytest.raises(OSError, match="injected write failure"):
        seal_artifact(
            destination,
            {"a.bin": b"a", "b.bin": b"b"},
            source_commit=SOURCE_COMMIT,
            audit_contract_version=CONTRACT_VERSION,
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".atomic.*.sealing")) == []


def test_inputs_are_closed_and_bytes_only(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be bytes"):
        seal_artifact(
            tmp_path / "bytearray",
            {"payload.bin": bytearray(b"not exact bytes")},  # type: ignore[dict-item]
            source_commit=SOURCE_COMMIT,
            audit_contract_version=CONTRACT_VERSION,
        )
    with pytest.raises(SecAuditArtifactError, match="source_commit"):
        seal_artifact(
            tmp_path / "short-commit",
            {},
            source_commit="abc123",
            audit_contract_version=CONTRACT_VERSION,
        )
    with pytest.raises(SecAuditArtifactError, match="version token"):
        seal_artifact(
            tmp_path / "bad-contract",
            {},
            source_commit=SOURCE_COMMIT,
            audit_contract_version="has spaces",
        )
