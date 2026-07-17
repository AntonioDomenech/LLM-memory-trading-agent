from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_benchmark import sec_gemma_lean_v31_preflight as preflight


IMPLEMENTATION_COMMIT = "1" * 40
IMPLEMENTATION_TREE = "2" * 40
EVIDENCE_COMMIT = "3" * 40
EVIDENCE_TREE = "4" * 40
CONTACT_SHA256 = "sha256:" + "5" * 64


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    return root


def _tags() -> bytes:
    return json.dumps(
        {
            "models": [
                {
                    "name": preflight.MODEL_NAME,
                    "model": preflight.MODEL_NAME,
                    "digest": preflight.MODEL_MANIFEST_SHA256,
                }
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _runtime(*, injected: bool) -> dict[str, Any]:
    return {
        "model_name": preflight.MODEL_NAME,
        "runtime_fingerprint_sha256": preflight.RUNTIME_FINGERPRINT_SHA256,
        "runtime_receipt_sha256": "6" * 64,
        "model_manifest_sha256": preflight.MODEL_MANIFEST_SHA256,
        "model_config_sha256": preflight.MODEL_CONFIG_DIGEST,
        "ordered_layer_content_sha256s": list(preflight.MODEL_LAYER_DIGESTS),
        "ollama_version": preflight.OLLAMA_VERSION,
        "ollama_version_response_sha256": preflight.OLLAMA_VERSION_RESPONSE_SHA256,
        "ollama_show_semantic_sha256": preflight.OLLAMA_SHOW_SEMANTIC_SHA256,
        "ollama_show_raw_sha256_diagnostic": "7" * 64,
        "model_info_sha256": preflight.MODEL_INFO_SHA256,
        "tags_identity": {
            "model_name": preflight.MODEL_NAME,
            "unique_model_match": True,
            "manifest_digest": preflight.MODEL_MANIFEST_SHA256,
            "tags_response_sha256": hashlib.sha256(_tags()).hexdigest(),
        },
        "version_identity_basis": {
            "live_api_version_call_made": False,
            "canonical_response_constructed_locally": True,
            "canonical_response_ascii": preflight.OLLAMA_VERSION_CANONICAL_BYTES.decode("ascii"),
            "basis": "inherited_preregistered_ollama_version_pin",
        },
        "identity_probe": {
            "loopback_host": preflight.LOOPBACK_HOST,
            "loopback_port": preflight.LOOPBACK_PORT,
            "restricted_sequence_enforced": True,
            "request_sequence": [list(item) for item in preflight.ALLOWED_IDENTITY_SEQUENCE],
            "observed_request_count": 2,
            "tags_requests": 1,
            "show_requests": 1,
            "version_requests": 0,
            "chat_or_generation_requests": 0,
            "test_injections_used": injected,
        },
    }


class _Delegate:
    def __init__(self, responses: list[bytes | BaseException]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, bytes]] = []

    def request(self, method: str, path: str, *, body: bytes) -> bytes:
        self.calls.append((method, path, body))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _bundle_verifier_recorder(seen: dict[str, bytes]):
    def verify(*, version_response_bytes: bytes, show_response_bytes: bytes) -> dict[str, Any]:
        seen["version"] = version_response_bytes
        seen["show"] = show_response_bytes
        result = _runtime(injected=True)
        return {
            "model_name": result["model_name"],
            "manifest_sha256": result["model_manifest_sha256"],
            "config_sha256": result["model_config_sha256"],
            "ordered_layer_content_sha256s": result["ordered_layer_content_sha256s"],
            "version_response_sha256": result["ollama_version_response_sha256"],
            "show_response_semantic_sha256": result["ollama_show_semantic_sha256"],
            "show_response_raw_sha256": result["ollama_show_raw_sha256_diagnostic"],
            "model_info_sha256": result["model_info_sha256"],
            "runtime_fingerprint_sha256": result["runtime_fingerprint_sha256"],
            "runtime_receipt_sha256": result["runtime_receipt_sha256"],
        }

    return verify


def test_offline_identity_probe_is_exactly_tags_then_show_and_local_version() -> None:
    show = b'{"verified":"show"}'
    delegate = _Delegate([_tags(), show])
    seen: dict[str, bytes] = {}
    result = preflight._verify_runtime_identity(
        delegate=delegate,
        bundle_verifier=_bundle_verifier_recorder(seen),
    )
    assert delegate.calls == [
        ("GET", "/api/tags", b""),
        ("POST", "/api/show", preflight.SHOW_REQUEST_BYTES),
    ]
    assert result["identity_probe"]["observed_request_count"] == 2
    assert result["identity_probe"]["version_requests"] == 0
    assert seen == {
        "version": preflight.OLLAMA_VERSION_CANONICAL_BYTES,
        "show": show,
    }
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="forbidden"):
        transport = preflight._RestrictedIdentityTransport(_Delegate([b"{}", b"{}", b"{}"]))
        transport.request("GET", "/api/tags", body=b"")
        transport.request("POST", "/api/show", body=preflight.SHOW_REQUEST_BYTES)
        transport.request("POST", "/api/chat", body=b"{}")


def test_production_trust_cannot_be_combined_with_injected_transport() -> None:
    delegate = _Delegate([_tags(), b"{}"])
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="transport"):
        preflight._verify_runtime_identity(trusted=True, delegate=delegate)
    assert delegate.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        b'{"models":[],"models":[]}',
        json.dumps(
            {
                "models": [
                    {"name": preflight.MODEL_NAME, "digest": preflight.MODEL_MANIFEST_SHA256},
                    {"model": preflight.MODEL_NAME, "digest": preflight.MODEL_MANIFEST_SHA256},
                ]
            }
        ).encode(),
        json.dumps(
            {"models": [{"name": preflight.MODEL_NAME, "digest": "0" * 64}]}
        ).encode(),
    ],
)
def test_tags_identity_is_unique_and_exactly_digest_pinned(payload: bytes) -> None:
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="tags"):
        preflight._verify_tags(payload)


def test_durable_intent_makes_crash_after_first_call_nonretryable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls: list[str] = []

    def crash_after_first() -> dict[str, Any]:
        calls.append("GET /api/tags")
        raise preflight.SecGemmaLeanV31PreflightError("runtime_identity_invalid")

    repository = {"head_commit": IMPLEMENTATION_COMMIT, "head_tree": IMPLEMENTATION_TREE}
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="runtime_identity"):
        preflight._durable_identity_probe(
            checkpoint,
            repository,
            contact_fingerprint=CONTACT_SHA256,
            runtime_verifier=crash_after_first,
        )
    assert (checkpoint / preflight.TEST_PROBE_INTENT_NAME).is_file()
    assert not (checkpoint / preflight.TEST_PROBE_COMPLETION_NAME).exists()
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="already_attempted"):
        preflight._durable_identity_probe(
            checkpoint,
            repository,
            contact_fingerprint=CONTACT_SHA256,
            runtime_verifier=crash_after_first,
        )
    assert calls == ["GET /api/tags"]


def test_durable_success_keeps_hash_only_markers_and_forbids_reprobe(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    repository = {"head_commit": IMPLEMENTATION_COMMIT, "head_tree": IMPLEMENTATION_TREE}
    runtime, evidence = preflight._durable_identity_probe(
        checkpoint,
        repository,
        contact_fingerprint=CONTACT_SHA256,
        runtime_verifier=lambda: _runtime(injected=True),
    )
    assert runtime["identity_probe"]["observed_request_count"] == 2
    assert evidence["request_sequence"] == [["GET", "/api/tags"], ["POST", "/api/show"]]
    completion = (checkpoint / preflight.TEST_PROBE_COMPLETION_NAME).read_text("ascii")
    assert "show_response_body" not in completion
    assert "tags_response_body" not in completion
    assert '"response_bodies_retained":false' in completion
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="already_attempted"):
        preflight._durable_identity_probe(
            checkpoint,
            repository,
            contact_fingerprint=CONTACT_SHA256,
            runtime_verifier=lambda: _runtime(injected=True),
        )


def _source_identity() -> dict[str, Any]:
    hashes = {path: "8" * 64 for path in preflight.EXECUTED_SOURCE_CLOSURE}
    loaded = [
        "agent_benchmark",
        "agent_benchmark.sec_gemma_lean_v31_delta",
        "agent_benchmark.sec_point_in_time",
        "agent_benchmark.sec_gemma_online_risk_overlay_runtime",
        "agent_benchmark.sec_gemma_online_risk_overlay_contract",
        "agent_benchmark.sec_filing_gemma_extractor_prompt",
        "agent_benchmark.sec_filing_gemma_extractor_schema",
    ]
    return {
        "raw_worktree_bytes_equal_head_blobs": True,
        "raw_bytes_comparison_catches_crlf_conversion": True,
        "runner_source_path": preflight.RUNNER_SOURCE_PATH,
        "source_count": len(hashes),
        "source_sha256s": hashes,
        "source_inventory_sha256": preflight._canonical_sha256(hashes),
        "loaded_source_modules_bound": loaded,
    }


def _report_repository() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = {
        "implementation": {
            "commit": IMPLEMENTATION_COMMIT,
            "tree": IMPLEMENTATION_TREE,
        },
        "manifest_sha256": "9" * 64,
    }
    repository = {
        "branch": preflight.BRANCH_NAME,
        "head_commit": IMPLEMENTATION_COMMIT,
        "head_tree": IMPLEMENTATION_TREE,
        "upstream_ref": preflight.EXPECTED_UPSTREAM,
        "cached_upstream_commit": IMPLEMENTATION_COMMIT,
        "remote_state_basis": "local_cached_origin_tracking_ref",
        "origin_url": preflight.EXPECTED_ORIGIN_URL,
        "working_tree_clean_at_verification": True,
        "preregistration_commit": preflight.delta.PREREG_COMMIT,
        "preregistration_tree": preflight.delta.PREREG_TREE,
        "preregistration_document_path": preflight.delta.PREREG_DOC_PATH,
        "preregistration_document_blob": preflight.delta.PREREG_DOC_BLOB,
        "preregistration_document_sha256": preflight.delta.PREREG_DOC_SHA256,
        "implementation_delta_manifest": manifest,
        "implementation_delta_manifest_sha256": manifest["manifest_sha256"],
        "implementation_delta_recomputed_from_committed_blobs": True,
        "executed_source_identity": _source_identity(),
    }
    return repository, deepcopy(manifest)


def _production_markers(
    checkpoint: Path,
    repository: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    nonce = "a" * 64
    intent_body = {
        "schema_version": "aapl-sec-gemma-lean-evidence-v3-1-identity-intent-v1",
        "test_injections_used": False,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "implementation_tree": IMPLEMENTATION_TREE,
        "implementation_delta_manifest_sha256": repository[
            "implementation_delta_manifest_sha256"
        ],
        "source_inventory_sha256": repository["executed_source_identity"][
            "source_inventory_sha256"
        ],
        "private_contact_sha256": CONTACT_SHA256,
        "attempt_nonce_sha256": nonce,
        "loopback_host": preflight.LOOPBACK_HOST,
        "loopback_port": preflight.LOOPBACK_PORT,
        "request_sequence": [list(item) for item in preflight.ALLOWED_IDENTITY_SEQUENCE],
        "request_count": 2,
        "version_requests": 0,
        "chat_or_generation_requests": 0,
        "external_network_requests": 0,
    }
    intent = preflight._write_durable_marker(
        checkpoint, preflight.PROBE_INTENT_NAME, intent_body
    )
    completion_body = {
        "schema_version": "aapl-sec-gemma-lean-evidence-v3-1-identity-complete-v1",
        "test_injections_used": False,
        "attempt_nonce_sha256": nonce,
        "intent_marker_sha256": intent["marker_sha256"],
        "intent_file_sha256": intent["file_sha256"],
        "request_sequence": [list(item) for item in preflight.ALLOWED_IDENTITY_SEQUENCE],
        "request_count": 2,
        "tags_response_sha256": runtime["tags_identity"]["tags_response_sha256"],
        "show_response_raw_sha256": runtime["ollama_show_raw_sha256_diagnostic"],
        "show_response_semantic_sha256": runtime["ollama_show_semantic_sha256"],
        "runtime_receipt_sha256": runtime["runtime_receipt_sha256"],
        "runtime_fingerprint_sha256": runtime["runtime_fingerprint_sha256"],
        "model_manifest_sha256": runtime["model_manifest_sha256"],
        "response_bodies_retained": False,
        "identity_verified": True,
    }
    completion = preflight._write_durable_marker(
        checkpoint, preflight.PROBE_COMPLETION_NAME, completion_body
    )
    return {
        "intent": intent,
        "completion": completion,
        "request_sequence": [list(item) for item in preflight.ALLOWED_IDENTITY_SEQUENCE],
        "markers_retained_after_success": True,
        "retry_after_any_intent_is_forbidden": True,
        "response_bodies_retained": False,
    }


def _valid_report(
    root: Path, *, publish_complete: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = root / Path(preflight.CHECKPOINT_ROOT)
    checkpoint.mkdir(parents=True)
    repository, manifest = _report_repository()
    runtime = _runtime(injected=False)
    evidence = _production_markers(checkpoint, repository, runtime)
    report = preflight._build_report(
        repository_before=repository,
        repository_after=repository,
        contact_fingerprint=CONTACT_SHA256,
        runtime=runtime,
        created_at="2026-07-17T12:00:00Z",
        elapsed_seconds=1.0,
        trusted=True,
        artifact_will_dirty=True,
        probe_evidence=evidence,
    )
    assert preflight._validate_sealable_report(report) == report
    if publish_complete:
        preflight._write_publish_complete(checkpoint, report)
    return report, manifest


def _write_artifact(root: Path, report: dict[str, Any], *, canonical: bool = True) -> bytes:
    path = root / Path(preflight.PREFLIGHT_ARTIFACT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        preflight._serialize_preflight_report(report)
        if canonical
        else preflight._canonical_json(report) + b"\n"
    )
    path.write_bytes(payload)
    return payload


def _validator_kwargs(
    root: Path,
    report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = root / Path(preflight.PREFLIGHT_ARTIFACT_PATH)

    def git_runner(_: Path, arguments: tuple[str, ...]) -> preflight._GitResult:
        mapping: dict[tuple[str, ...], bytes] = {
            ("rev-parse", "--show-toplevel"): (str(root) + "\n").encode(),
            ("branch", "--show-current"): (preflight.BRANCH_NAME + "\n").encode(),
            ("status", "--porcelain=v1", "--untracked-files=all"): b"",
            ("remote", "get-url", "origin"): (preflight.EXPECTED_ORIGIN_URL + "\n").encode(),
            ("rev-parse", "HEAD"): (EVIDENCE_COMMIT + "\n").encode(),
            ("rev-parse", "HEAD^{tree}"): (EVIDENCE_TREE + "\n").encode(),
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): (preflight.EXPECTED_UPSTREAM + "\n").encode(),
            ("rev-parse", "@{upstream}"): (EVIDENCE_COMMIT + "\n").encode(),
            ("rev-list", "--parents", "-n", "1", EVIDENCE_COMMIT): (
                f"{EVIDENCE_COMMIT} {IMPLEMENTATION_COMMIT}\n"
            ).encode(),
            (
                "diff-tree",
                "--name-status",
                "-r",
                "-z",
                "--no-renames",
                IMPLEMENTATION_COMMIT,
                EVIDENCE_COMMIT,
            ): b"A\0" + preflight.PREFLIGHT_ARTIFACT_PATH.encode() + b"\0",
        }
        if arguments[0:2] == ("merge-base", "--is-ancestor"):
            return preflight._GitResult(0, b"")
        if arguments == (
            "cat-file",
            "blob",
            f"{EVIDENCE_COMMIT}:{preflight.PREFLIGHT_ARTIFACT_PATH}",
        ):
            return preflight._GitResult(0, artifact.read_bytes())
        if arguments not in mapping:
            raise AssertionError(f"unexpected Git call: {arguments!r}")
        return preflight._GitResult(0, mapping[arguments])

    def delta_builder(*_: Any, **__: Any) -> dict[str, Any]:
        return deepcopy(manifest)

    def delta_validator(*_: Any, **__: Any) -> dict[str, Any]:
        return deepcopy(manifest)

    return {
        "git_runner": git_runner,
        "delta_builder": delta_builder,
        "delta_validator": delta_validator,
        "contact_verifier": lambda _: CONTACT_SHA256,
        "source_verifier": lambda _root, _head: deepcopy(
            report["repository"]["executed_source_identity"] if source is None else source
        ),
    }


def _rehash(report: dict[str, Any]) -> None:
    body = {key: value for key, value in report.items() if key != "preflight_sha256"}
    report["preflight_sha256"] = preflight._canonical_sha256(body)


def test_strict_acquisition_validator_accepts_only_full_committed_evidence(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    result = preflight._validate_committed_preflight_for_acquisition(
        root, **_validator_kwargs(root, report, manifest)
    )
    assert result["validated"] is True
    assert result["artifact_only_evidence_commit"] is True
    assert result["durable_probe_state"]["exact_request_sequence_verified"] is True


def test_minimal_self_rehashed_forged_receipt_is_rejected(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    report, manifest = _valid_report(root)
    forged = {"status": "passed"}
    forged["preflight_sha256"] = preflight._canonical_sha256({"status": "passed"})
    _write_artifact(root, forged)
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError):
        preflight._validate_committed_preflight_for_acquisition(
            root, **_validator_kwargs(root, report, manifest)
        )


def test_extra_durable_field_and_noncanonical_artifact_bytes_reject(tmp_path: Path) -> None:
    root = _repo(tmp_path / "extra")
    report, _ = _valid_report(root)
    report["checkpoint"]["durable_identity_probe_evidence"]["rogue"] = True
    _rehash(report)
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="untrusted"):
        preflight._validate_sealable_report(report)

    root = _repo(tmp_path / "encoding")
    report, manifest = _valid_report(root)
    _write_artifact(root, report, canonical=False)
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="committed_preflight"):
        preflight._validate_committed_preflight_for_acquisition(
            root, **_validator_kwargs(root, report, manifest)
        )


def test_altered_marker_delta_source_and_contact_each_reject(tmp_path: Path) -> None:
    root = _repo(tmp_path / "marker")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    marker = root / preflight.CHECKPOINT_ROOT / preflight.PROBE_INTENT_NAME
    marker.write_bytes(marker.read_bytes().replace(b'"request_count":2', b'"request_count":3'))
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="identity_probe"):
        preflight._validate_committed_preflight_for_acquisition(
            root, **_validator_kwargs(root, report, manifest)
        )

    root = _repo(tmp_path / "delta")
    report, manifest = _valid_report(root)
    report["repository"]["implementation_delta_manifest"]["rogue"] = True
    _rehash(report)
    _write_artifact(root, report)
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="implementation_delta"):
        preflight._validate_committed_preflight_for_acquisition(
            root, **_validator_kwargs(root, report, manifest)
        )

    root = _repo(tmp_path / "source")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    altered_source = deepcopy(report["repository"]["executed_source_identity"])
    first = next(iter(altered_source["source_sha256s"]))
    altered_source["source_sha256s"][first] = "0" * 64
    altered_source["source_inventory_sha256"] = preflight._canonical_sha256(
        altered_source["source_sha256s"]
    )
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="executed_source"):
        preflight._validate_committed_preflight_for_acquisition(
            root,
            **_validator_kwargs(root, report, manifest, source=altered_source),
        )

    root = _repo(tmp_path / "contact")
    report, manifest = _valid_report(root)
    report["private_sec_contact"]["sha256"] = "sha256:" + "f" * 64
    _rehash(report)
    _write_artifact(root, report)
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError):
        preflight._validate_committed_preflight_for_acquisition(
            root, **_validator_kwargs(root, report, manifest)
        )


def test_prereg_ancestor_and_final_git_contact_marker_rebounds_reject(tmp_path: Path) -> None:
    root = _repo(tmp_path / "ancestor")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    kwargs = _validator_kwargs(root, report, manifest)
    base_git = kwargs["git_runner"]

    def no_prereg(repo: Path, arguments: tuple[str, ...]) -> preflight._GitResult:
        if arguments == (
            "merge-base",
            "--is-ancestor",
            preflight.delta.PREREG_COMMIT,
            IMPLEMENTATION_COMMIT,
        ):
            return preflight._GitResult(1, b"")
        return base_git(repo, arguments)

    kwargs["git_runner"] = no_prereg
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="preregistration"):
        preflight._validate_committed_preflight_for_acquisition(root, **kwargs)

    root = _repo(tmp_path / "git_toctou")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    kwargs = _validator_kwargs(root, report, manifest)
    base_git = kwargs["git_runner"]
    tree_reads = 0

    def changing_tree(repo: Path, arguments: tuple[str, ...]) -> preflight._GitResult:
        nonlocal tree_reads
        if arguments == ("rev-parse", "HEAD^{tree}"):
            tree_reads += 1
            if tree_reads > 1:
                return preflight._GitResult(0, ("b" * 40 + "\n").encode())
        return base_git(repo, arguments)

    kwargs["git_runner"] = changing_tree
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="changed"):
        preflight._validate_committed_preflight_for_acquisition(root, **kwargs)

    root = _repo(tmp_path / "contact_toctou")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    kwargs = _validator_kwargs(root, report, manifest)
    contacts = iter((CONTACT_SHA256, "sha256:" + "e" * 64))
    kwargs["contact_verifier"] = lambda _: next(contacts)
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="changed"):
        preflight._validate_committed_preflight_for_acquisition(root, **kwargs)

    root = _repo(tmp_path / "marker_toctou")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    kwargs = _validator_kwargs(root, report, manifest)
    marker_calls = 0

    def changing_marker(repo: Path, value: dict[str, Any]) -> dict[str, Any]:
        nonlocal marker_calls
        result = preflight._validate_durable_probe_state(repo, value)
        marker_calls += 1
        if marker_calls == 1:
            marker_path = repo / preflight.CHECKPOINT_ROOT / preflight.PROBE_COMPLETION_NAME
            marker_path.write_bytes(
                marker_path.read_bytes().replace(b'"request_count":2', b'"request_count":3')
            )
        return result

    kwargs["marker_verifier"] = changing_marker
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="identity_probe"):
        preflight._validate_committed_preflight_for_acquisition(root, **kwargs)


def test_publish_failure_leaves_orphan_unusable_and_publish_is_after_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    report, _ = _valid_report(root, publish_complete=False)
    checkpoint = root / preflight.CHECKPOINT_ROOT

    def fail_parent_fsync(_: Path) -> None:
        raise OSError("simulated post-rename fsync failure")

    monkeypatch.setattr(preflight, "_fsync_directory", fail_parent_fsync)
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="artifact_unwritable"):
        preflight._atomic_publish(root, report, checkpoint)
    assert (root / preflight.PREFLIGHT_ARTIFACT_PATH).is_file()
    assert not (checkpoint / preflight.PUBLISH_COMPLETE_NAME).exists()
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="publish_state"):
        preflight._validate_publish_complete(root, report)

    import ast
    import inspect

    tree = ast.parse(inspect.getsource(preflight.run_preflight))
    with_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.With)]
    assert len(with_nodes) == 1
    calls_inside_lease = {
        node.func.id
        for node in ast.walk(with_nodes[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_atomic_publish" not in calls_inside_lease


def test_test_markers_publish_marker_and_active_lock_cannot_masquerade(
    tmp_path: Path,
) -> None:
    test_checkpoint = tmp_path / "test-markers"
    test_checkpoint.mkdir()
    repository = {"head_commit": IMPLEMENTATION_COMMIT, "head_tree": IMPLEMENTATION_TREE}
    _, test_evidence = preflight._durable_identity_probe(
        test_checkpoint,
        repository,
        contact_fingerprint=CONTACT_SHA256,
        runtime_verifier=lambda: _runtime(injected=True),
    )
    assert test_evidence["intent"]["path"].endswith(preflight.TEST_PROBE_INTENT_NAME)
    assert test_evidence["completion"]["path"].endswith(
        preflight.TEST_PROBE_COMPLETION_NAME
    )
    intent = json.loads(
        (test_checkpoint / preflight.TEST_PROBE_INTENT_NAME).read_text("ascii")
    )
    assert intent["test_injections_used"] is True
    assert "test-identity-intent" in intent["schema_version"]

    root = _repo(tmp_path / "missing_publish")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    (root / preflight.CHECKPOINT_ROOT / preflight.PUBLISH_COMPLETE_NAME).unlink()
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="publish_state"):
        preflight._validate_committed_preflight_for_acquisition(
            root, **_validator_kwargs(root, report, manifest)
        )

    root = _repo(tmp_path / "changed_publish")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    publish = root / preflight.CHECKPOINT_ROOT / preflight.PUBLISH_COMPLETE_NAME
    publish.write_bytes(
        publish.read_bytes().replace(b'"artifact_path":"e/', b'"artifact_path":"x/')
    )
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="publish_state"):
        preflight._validate_committed_preflight_for_acquisition(
            root, **_validator_kwargs(root, report, manifest)
        )

    root = _repo(tmp_path / "active")
    report, manifest = _valid_report(root)
    _write_artifact(root, report)
    (root / preflight.CHECKPOINT_ROOT / preflight.ACTIVE_LOCK_NAME).write_bytes(b"active")
    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="active_run"):
        preflight._validate_committed_preflight_for_acquisition(
            root, **_validator_kwargs(root, report, manifest)
        )


def test_git_environment_drops_redirect_controls_and_source_crlf_is_raw(tmp_path: Path) -> None:
    environment = preflight._git_environment()
    assert "GIT_DIR" not in environment
    assert "GIT_WORK_TREE" not in environment
    assert "GIT_INDEX_FILE" not in environment
    assert "PATH" not in environment
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"

    root = _repo(tmp_path)
    path = "agent_benchmark/example.py"
    source = root / Path(path)
    source.parent.mkdir()
    source.write_bytes(b"VALUE = 1\r\n")

    def committed(_: Path, arguments: tuple[str, ...]) -> preflight._GitResult:
        assert arguments == ("cat-file", "blob", f"{IMPLEMENTATION_COMMIT}:{path}")
        return preflight._GitResult(0, b"VALUE = 1\n")

    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="executed_source"):
        preflight._verify_execution_closure(
            root,
            IMPLEMENTATION_COMMIT,
            git_runner=committed,
            paths=(path,),
            bind_runner=False,
        )


def test_private_contact_is_untracked_ignored_and_only_fingerprint_escapes(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    config = root / preflight.PRIVATE_CONFIG_PATH
    config.parent.mkdir()
    secret = "Alder Research security-team@alder-research.testmail.example"
    config.write_text(
        json.dumps(
            {
                "settings": {"unrelated": True},
                "secrets": {"sec_user_agent": secret, "unrelated": "kept-private"},
            }
        ),
        encoding="utf-8",
    )

    def safe_git(_: Path, arguments: tuple[str, ...]) -> preflight._GitResult:
        if arguments[0] == "ls-files":
            return preflight._GitResult(1, b"")
        if arguments[0] == "check-ignore":
            return preflight._GitResult(0, b"")
        raise AssertionError(arguments)

    expected = "sha256:" + hashlib.sha256(secret.encode()).hexdigest()
    validator = lambda _: SimpleNamespace(
        sha256=expected,
        real_contact_validated=True,
    )
    observed = preflight._verify_private_contact(
        root,
        git_runner=safe_git,
        validator=validator,
    )
    assert observed == expected
    assert secret not in repr(observed)

    def tracked(_: Path, arguments: tuple[str, ...]) -> preflight._GitResult:
        return preflight._GitResult(0, b"")

    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="tracked"):
        preflight._verify_private_contact(root, git_runner=tracked, validator=validator)

    def not_ignored(_: Path, arguments: tuple[str, ...]) -> preflight._GitResult:
        return preflight._GitResult(1, b"")

    with pytest.raises(preflight.SecGemmaLeanV31PreflightError, match="not_ignored"):
        preflight._verify_private_contact(root, git_runner=not_ignored, validator=validator)
