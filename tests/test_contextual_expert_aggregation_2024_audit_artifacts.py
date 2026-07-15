from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_2024_audit_artifacts as artifacts


_SCHEMA_PRIMITIVES = {
    "string",
    "bool",
    "int",
    "float",
    "number",
    "sha256",
    "git_commit",
    "iso_date",
    "null",
}


def _assert_supported_frozen_schema(schema: Any, *, path: str = "schema") -> None:
    if type(schema) is str:
        assert "PENDING" not in schema.upper(), f"{path} is still pending"
        if schema in _SCHEMA_PRIMITIVES:
            return
        if schema.startswith("list[") and schema.endswith("]"):
            _assert_supported_frozen_schema(schema[5:-1], path=f"{path}.items")
            return
        if schema.startswith("map[string,") and schema.endswith("]"):
            _assert_supported_frozen_schema(
                schema[len("map[string,") : -1], path=f"{path}.values"
            )
            return
        raise AssertionError(f"{path} uses unknown primitive {schema!r}")

    assert isinstance(schema, Mapping), f"{path} is not a schema descriptor"
    schema_type = schema.get("type")
    if schema_type == "object":
        assert set(schema) in (
            {"type", "fields"},
            {"type", "fields", "self_hash_field"},
        ), f"{path} object descriptor keys changed"
        fields = schema["fields"]
        assert isinstance(fields, Mapping) and fields, f"{path} fields are empty"
        assert all(type(name) is str and name for name in fields)
        if "self_hash_field" in schema:
            assert schema["self_hash_field"] in fields
        for name, child in fields.items():
            _assert_supported_frozen_schema(child, path=f"{path}.{name}")
        return
    if schema_type == "fixed_map":
        assert set(schema) == {"type", "keys", "values"}
        keys = schema["keys"]
        assert (
            type(keys) is list
            and keys
            and all(type(key) is str and key for key in keys)
            and len(keys) == len(set(keys))
        ), f"{path} fixed-map keys are malformed"
        _assert_supported_frozen_schema(schema["values"], path=f"{path}.values")
        return
    if schema_type == "list":
        assert set(schema) in ({"type", "items"}, {"type", "items", "length"})
        if "length" in schema:
            assert type(schema["length"]) is int and schema["length"] >= 0
        _assert_supported_frozen_schema(schema["items"], path=f"{path}.items")
        return
    if schema_type == "map":
        assert set(schema) == {"type", "values"}
        _assert_supported_frozen_schema(schema["values"], path=f"{path}.values")
        return
    if schema_type == "enum":
        assert set(schema) == {"type", "values"}
        assert type(schema["values"]) is list and schema["values"]
        return
    if schema_type == "literal":
        assert set(schema) == {"type", "value"}
        return
    if schema_type == "nullable":
        assert set(schema) == {"type", "schema"}
        _assert_supported_frozen_schema(schema["schema"], path=f"{path}.schema")
        return
    raise AssertionError(f"{path} uses unknown descriptor type {schema_type!r}")


def _digest(token: str = "a") -> str:
    return "sha256:" + token * 64


def _payloads() -> dict[str, bytes]:
    return {
        name: (
            artifacts.GIT_ATTRIBUTES_BYTES
            if name == ".gitattributes"
            else f"synthetic:{position}\n".encode("ascii")
        )
        for position, name in enumerate(artifacts.PAYLOAD_FILE_ORDER)
    }


def _manifest_fields() -> dict[str, object]:
    return {
        "manifest_schema_version": artifacts.MANIFEST_SCHEMA_VERSION,
        "contract_version": artifacts.CONTRACT_VERSION,
        "verifier_id": artifacts.VERIFIER_ID,
        "run_id": artifacts.AUDIT_RUN_ID,
        "stage": artifacts.AUDIT_STAGE,
        "status": "REJECTED_2024",
        "stage_pass": False,
        "evidence_classification": "post_hoc_frozen_policy_2024_replication",
        "preregistration_commit": artifacts.PREREGISTRATION_COMMIT,
        "git_identity": {},
        "dependency_identity_sha256": _digest("1"),
        "artifact_schema_registry_sha256": artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256,
        "parent_manifest_file_sha256": artifacts.PARENT_MANIFEST_FILE_SHA256,
        "parent_manifest_self_sha256": artifacts.PARENT_MANIFEST_SELF_SHA256,
        "parent_checkpoint_file_sha256": artifacts.PARENT_CHECKPOINT_FILE_SHA256,
        "parent_checkpoint_self_sha256": artifacts.PARENT_CHECKPOINT_SELF_SHA256,
        "receipt_file_sha256": _digest("2"),
        "input_raw_sha256": _digest("3"),
        "input_canonical_sha256": _digest("4"),
        "attempt_lock_file_sha256": _digest("5"),
        "runtime_cost_evidence_file_sha256": _digest("6"),
        "gate_report_file_sha256": _digest("7"),
        "learning_classification": "unexercised",
        "later_market_data_accessed": False,
        "prior_2024_artifact_accessed": False,
        "external_cost_usd": 0.0,
    }


def _marker_fields() -> dict[str, object]:
    return {
        "marker_schema_version": artifacts.MARKER_SCHEMA_VERSION,
        "contract_version": artifacts.CONTRACT_VERSION,
        "run_id": artifacts.AUDIT_RUN_ID,
        "stage": artifacts.AUDIT_STAGE,
        "preregistration_commit": artifacts.PREREGISTRATION_COMMIT,
        "git_commit": "b" * 40,
        "attempt_lock_file_sha256": _digest("1"),
        "final_relative_path": artifacts.OUTPUT_DIRECTORY.as_posix(),
        "final_inventory_sha256": artifacts.FINAL_INVENTORY_SHA256,
        "stage_manifest_file_sha256": _digest("2"),
        "stage_manifest_self_sha256": _digest("3"),
        "checksums_file_sha256": _digest("4"),
        "marker_preparation_elapsed_seconds": 12.5,
        "stage_deadline_seconds": artifacts.STAGE_DEADLINE_SECONDS,
        "marker_preparation_deadline_pass": True,
        "external_cost_usd": 0.0,
    }


def _omitted_field_hash(value: dict[str, object], field: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_frozen_inventory_and_schema_hashes_are_exact() -> None:
    assert len(artifacts.BUNDLE_FILE_ORDER) == 43
    assert len(artifacts.PAYLOAD_FILE_ORDER) == 41
    assert len(artifacts.CHECKSUM_FILE_ORDER) == 42
    assert set(artifacts.PAYLOAD_FILE_ORDER) == artifacts.PAYLOAD_FILENAMES
    assert set(artifacts.CHECKSUM_FILE_ORDER) == artifacts.CHECKSUM_FILENAMES
    assert artifacts.CHECKSUM_FILE_ORDER[-1] == "stage_manifest.json"
    assert "checksums.json" not in artifacts.CHECKSUM_FILENAMES
    assert artifacts.SUCCESS_MARKER_FILENAME not in artifacts.BUNDLE_FILENAMES
    assert artifacts.PENDING_SUCCESS_MARKER_FILENAME not in artifacts.BUNDLE_FILENAMES
    assert artifacts.GIT_ATTRIBUTES_BYTES == b"* -text\n"
    assert artifacts.artifact_schema_registry_sha256() == (
        artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
    )

    expected = {
        "fixed_features": "8282e3a6cd1ef04f26bc2131d9ec9827a3f0aaa706f5f68fd00cae7ff8c42a20",
        "forecast": "d08661dc9ea8526c9697750143fa061cba2c197641f6c8480fa25c7a512ea9b1",
        "fixed_comparators": "0c6815db9e6e2e7e550e909611129b75ed2329783d2609ed35b561cc241615a2",
        "matured_lessons": "10cf64eea698fa0761acbaa56793bb92ddcf3b50098b21832cae06b43e8afb4c",
        "diagnostics": "558403da0eccf05db0313c8668840f9726ea2ad1652ff0695a24eafa052ba4e0",
        "ledger": "9f1f1b34d3f0db6cedf9b7cbd59ba9c75b8e2be2055e761568397e60fd386971",
    }
    assert dict(artifacts.TABLE_COLUMN_SHA256) == expected


def test_json_schema_registry_is_complete_deep_and_nonprovisional() -> None:
    descriptors = artifacts.ARTIFACT_SCHEMA_REGISTRY["json_artifacts"]
    expected = {
        filename
        for filename in artifacts.PAYLOAD_FILE_ORDER
        if filename.endswith(".json")
        and not filename.endswith(".table.json")
        and filename
        not in {
            artifacts.ATTEMPT_LOCK_FILENAME,
            "input_snapshot_receipt.json",
            "parent_stage_manifest.json",
            "parent_checksums.json",
        }
    }
    assert set(descriptors) == expected
    assert len(descriptors) == 16
    for filename, descriptor in descriptors.items():
        assert set(descriptor) == {
            "kind",
            "schema_version",
            "evidence_classification",
            "literal_nested_schema",
            "unknown_or_missing_keys_fatal",
        }
        assert descriptor["kind"] == "canonical_json"
        assert descriptor["schema_version"] == 1
        assert descriptor["unknown_or_missing_keys_fatal"] is True
        _assert_supported_frozen_schema(
            descriptor["literal_nested_schema"], path=filename
        )
    encoded_descriptors = json.dumps(descriptors, sort_keys=True)
    assert "map[string," not in encoded_descriptors
    assert '"type": "map"' not in encoded_descriptors
    assert not hasattr(artifacts, "JSON_SCHEMA_REGISTRY_DEEP_SHAPES_PENDING")


def test_schema_registry_walker_rejects_pending_and_unknown_descriptors() -> None:
    for invalid in (
        "PENDING_RUNNER_PAYLOAD_BUILDER_FREEZE",
        "unknown_primitive",
        {"type": "unknown_descriptor", "values": []},
        {"type": "object", "fields": {"nested": "unknown_primitive"}},
    ):
        with pytest.raises(AssertionError):
            _assert_supported_frozen_schema(invalid)


def test_registered_table_requires_row_count_schema_and_one_lf() -> None:
    dates = pd.bdate_range("2024-01-02", periods=252, name="date")
    frame = pd.DataFrame(
        {
            column: [float(position % 2) for position in range(len(dates))]
            for column in artifacts.FIXED_COMPARATOR_SCHEMA.columns
        },
        index=dates,
    )
    filename = "audit_forecast__fixed_comparators.table.json"

    payload = artifacts.canonical_registered_table_bytes(filename, frame)

    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")
    parsed = artifacts.parse_registered_table_bytes(filename, payload)
    assert parsed.index.equals(frame.index.rename("decision_date"))
    assert parsed.to_numpy().tolist() == frame.to_numpy().tolist()

    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.canonical_registered_table_bytes(filename, frame.iloc[:-1])
    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.parse_registered_table_bytes(filename, payload[:-1])
    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.parse_registered_table_bytes(filename, payload + b"\n")
    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.canonical_registered_table_bytes(
            filename, frame.reset_index(drop=True)
        )


def test_canonical_json_parser_rejects_duplicates_and_noncanonical_bytes() -> None:
    value = {"b": [1, 2], "a": {"finite": 1.5}}
    payload = artifacts.canonical_json_line_bytes(value)
    assert artifacts.parse_canonical_json_line(payload) == value

    for invalid in (
        b'{"a":1,"a":2}\n',
        b'{"b":2, "a":1}\n',
        b'{"a":NaN}\n',
        b'{"a":1}',
        b'{"a":1}\n\n',
    ):
        with pytest.raises(
            artifacts.ContextualExpertAggregation2024AuditArtifactError
        ):
            artifacts.parse_canonical_json_line(invalid)


def test_manifest_and_checksums_use_exact_domains_and_omitted_field_self_hash() -> None:
    payloads = _payloads()
    metadata = artifacts.build_bundle_metadata(
        payloads, manifest_fields=_manifest_fields()
    )

    assert metadata.manifest["manifest_sha256"] == _omitted_field_hash(
        dict(metadata.manifest), "manifest_sha256"
    )
    assert artifacts.parse_stage_manifest_bytes(metadata.manifest_bytes) == dict(
        metadata.manifest
    )
    assert set(metadata.manifest["payload_sha256"]) == artifacts.PAYLOAD_FILENAMES
    assert set(metadata.checksums) == artifacts.CHECKSUM_FILENAMES
    assert metadata.checksums["stage_manifest.json"] == (
        "sha256:" + hashlib.sha256(metadata.manifest_bytes).hexdigest()
    )
    assert artifacts.parse_checksums_bytes(metadata.checksums_bytes) == dict(
        metadata.checksums
    )

    tampered = dict(metadata.manifest)
    tampered["evidence_classification"] = "tampered"
    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.parse_stage_manifest(tampered)
    wrong_fields = _manifest_fields()
    wrong_fields["extra"] = True
    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.build_bundle_metadata(payloads, manifest_fields=wrong_fields)


def test_success_marker_is_outside_bundle_and_strictly_deadline_bound() -> None:
    marker = artifacts.build_success_marker(_marker_fields())
    payload = artifacts.success_marker_bytes(marker)

    assert marker["marker_sha256"] == _omitted_field_hash(marker, "marker_sha256")
    assert artifacts.parse_success_marker_bytes(payload) == marker
    assert marker["final_inventory_sha256"] == artifacts.final_inventory_sha256()

    deadline = _marker_fields()
    deadline["marker_preparation_elapsed_seconds"] = artifacts.STAGE_DEADLINE_SECONDS
    deadline["marker_preparation_deadline_pass"] = False
    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.build_success_marker(deadline)
    inconsistent = _marker_fields()
    inconsistent["marker_preparation_elapsed_seconds"] = (
        artifacts.STAGE_DEADLINE_SECONDS
    )
    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.build_success_marker(inconsistent)


def test_private_bundle_writer_reconstructs_and_writes_exact_43_files(tmp_path) -> None:
    payloads = _payloads()
    metadata = artifacts.build_bundle_metadata(
        payloads, manifest_fields=_manifest_fields()
    )
    private = tmp_path / "private"

    artifacts.write_private_bundle(private, payloads=payloads, metadata=metadata)

    assert {entry.name for entry in private.iterdir()} == artifacts.BUNDLE_FILENAMES
    assert (private / "stage_manifest.json").read_bytes() == metadata.manifest_bytes
    assert (private / "checksums.json").read_bytes() == metadata.checksums_bytes
    with pytest.raises(artifacts.ContextualExpertAggregation2024AuditArtifactError):
        artifacts.write_private_bundle(private, payloads=payloads, metadata=metadata)
