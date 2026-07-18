from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from agent_benchmark import sec_gemma_lean_science_v39_contract as contract
from agent_benchmark import sec_gemma_lean_science_v39_preflight as preflight
from agent_benchmark import sec_gemma_lean_science_v39_runner as runner


SHA1_A = "1" * 40
SHA1_B = "2" * 40
SHA64 = {
    name: f"{index:x}" * 64
    for index, name in enumerate(
        (
            "legacy",
            "documents",
            "records",
            "source_order",
            "events",
            "prior_links",
            "primary",
            "preprocessed",
            "requests",
            "pilot",
            "remaining",
        ),
        start=1,
    )
}


def _assert_code(code: str, function, *args, **kwargs) -> None:
    with pytest.raises(preflight.V39PreflightError) as captured:
        function(*args, **kwargs)
    assert captured.value.code == code
    assert str(captured.value) == code


def _assert_failed_preflight_artifact(root: Path, code: str) -> dict[str, object]:
    payload = (root / contract.PREFLIGHT_ARTIFACT_PATH).read_bytes()
    value = json.loads(payload)
    assert payload == contract.canonical_json_bytes(value) + b"\n"
    assert preflight.validate_public_failed_preflight_artifact(value) == value
    assert value["status"] == "failed"
    assert value["failure_code"] == code
    assert value["preflight_consumed"] is True
    assert value["failure_preserved"] is True
    assert value["rerun_authorized"] is False
    assert value["development_authorized"] is False
    return value


def _source_inventory() -> list[dict[str, object]]:
    rows = []
    for index, path in enumerate(contract.IMPLEMENTATION_ALLOWED_PATHS, start=1):
        rows.append(
            {
                "path": path,
                "git_blob_sha1": f"{index:x}" * 40,
                "literal_sha256": f"{index:x}" * 64,
                "byte_count": index,
            }
        )
    return rows


def _repository_snapshot() -> dict[str, object]:
    rows = _source_inventory()
    predecessor = "e" * 64
    return {
        "ancestry": {
            "branch": contract.BRANCH_NAME,
            "commit": SHA1_A,
            "tree": SHA1_B,
            "parent": contract.PREREGISTRATION_COMMIT,
            "local_head": SHA1_A,
            "remote_head": SHA1_A,
            "changed_paths": {
                path: "A" for path in contract.IMPLEMENTATION_ALLOWED_PATHS
            },
            "clean_worktree": True,
            "preregistration_authenticated": True,
            "predecessor_blobs_unchanged": True,
        },
        "source_inventory": rows,
        "source_inventory_sha256": contract.canonical_sha256(rows),
        "production_source_inventory_sha256": contract.canonical_sha256(rows[:6]),
        "test_source_inventory_sha256": contract.canonical_sha256(rows[6:]),
        "predecessor_inventory_sha256": predecessor,
        "unchanged_predecessor_inventory_sha256": predecessor,
        "preregistration_authenticated": True,
    }


def _bridge_manifest() -> dict[str, object]:
    body = {
        "schema_version": "aapl-sec-gemma-lean-science-v3-9-streaming-bridge-v1",
        "stage": "development",
        "document_count": 75,
        "source_authority_pins_sha256": contract.canonical_sha256(
            contract.build_source_authority_pins()
        ),
        "source_authority_base_commit": contract.BASE_COMMIT,
        "source_authority_base_tree": contract.BASE_TREE,
        "source_inventory_sha256": contract.V38_INVENTORY_SHA256,
        "stage_source_seal_sha256": contract.V38_STAGE_SOURCE_SEAL_SHA256,
        "checkpoint_sha256": contract.V38_LOGICAL_CHECKPOINT_SHA256,
        "compact_replay_sha256": contract.V38_COMPACT_REPLAY_SHA256,
        "role_manifests_sha256": contract.V38_ROLE_MANIFEST_INVENTORY_SHA256,
        "role_plan_sha256": contract.V38_ROLE_PLAN_SHA256,
        "science_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
        "legacy_projection_sha256": SHA64["legacy"],
        "documents_sha256": SHA64["documents"],
        "records_sha256": SHA64["records"],
        "source_order_sha256": SHA64["source_order"],
        "event_order_sha256": SHA64["events"],
        "prior_links_sha256": SHA64["prior_links"],
        "primary_documents_sha256": SHA64["primary"],
        "set_parity": True,
        "source_sequence_parity": True,
        "legacy_projection_parity": True,
        "typed_identity_parity": True,
        "prior_links_are_internal_only": True,
        "first_10k_and_10q_have_no_prior": True,
        "peak_live_complete_submission_blob_count": 1,
        "sec_request_count": 0,
        "confirmation_or_final_opened": False,
        "contains_private_rows": False,
        "contains_accessions_urls_filenames_or_bodies": False,
    }
    return {**body, "bridge_sha256": contract.canonical_sha256(body)}


def _commitments() -> dict[str, object]:
    return {
        "schema_version": preflight.REQUEST_COMMITMENTS_SCHEMA_VERSION,
        "stage": "development",
        "document_count": 75,
        "record_count": 75,
        "event_count": 75,
        "request_count": 75,
        "pilot_count": 5,
        "remaining_count": 70,
        "documents_sha256": SHA64["documents"],
        "records_sha256": SHA64["records"],
        "events_sha256": SHA64["events"],
        "preprocessed_events_sha256": SHA64["preprocessed"],
        "canonical_requests_sha256": SHA64["requests"],
        "pilot_order_sha256": SHA64["pilot"],
        "remaining_order_sha256": SHA64["remaining"],
        "source_order_sha256": SHA64["source_order"],
        "prior_links_sha256": SHA64["prior_links"],
        "set_parity": True,
        "sequence_parity": True,
        "prior_link_parity": True,
        "confirmation_or_final_opened": False,
        "contains_private_rows": False,
    }


def _offline_tests() -> dict[str, object]:
    return {
        "schema_version": preflight.OFFLINE_TEST_REPORT_SCHEMA_VERSION,
        "suite": "complete_repository_pytest",
        "passed": True,
        "exit_code": 0,
        "command_profile_sha256": preflight.OFFLINE_TEST_COMMAND_PROFILE_SHA256,
    }


def test_complete_offline_suite_binds_the_six_hour_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preflight.subprocess, "run", run)
    report = preflight._run_complete_offline_suite(tmp_path)

    assert preflight.OFFLINE_TEST_TIMEOUT_SECONDS == 21_600
    assert preflight.OFFLINE_TEST_COMMAND_PROFILE["timeout_seconds"] == 21_600
    assert observed["command"] == [preflight.sys.executable, "-m", "pytest", "-q"]
    assert observed["cwd"] == tmp_path
    assert observed["timeout"] == 21_600
    assert observed["stdin"] is preflight.subprocess.DEVNULL
    assert observed["stdout"] is preflight.subprocess.DEVNULL
    assert observed["stderr"] is preflight.subprocess.DEVNULL
    assert report == _offline_tests()


def _dependencies(
    root: Path,
    *,
    effects: dict[str, int] | None = None,
    commitments: dict[str, object] | None = None,
    tokens: tuple[bytes | str, ...] | None = None,
    calls: dict[str, int] | None = None,
) -> preflight.PreflightDependencies:
    counter = calls if calls is not None else {}
    projection = SimpleNamespace(manifest=_bridge_manifest())

    def authenticate(_root: Path) -> object:
        counter["authenticate"] = counter.get("authenticate", 0) + 1
        return object()

    def build_projection(_authority: object) -> object:
        counter["projection"] = counter.get("projection", 0) + 1
        return projection

    zero = (
        effects
        if effects is not None
        else contract.build_effect_budgets()["zero_effect_preflight"]
    )
    return preflight.PreflightDependencies(
        inspect_repository=lambda _root: _repository_snapshot(),
        authenticate_source=authenticate,
        build_projection=build_projection,
        build_request_commitments=lambda _projection: (
            commitments if commitments is not None else _commitments()
        ),
        effect_snapshot=lambda: copy.deepcopy(zero),
        run_offline_tests=lambda _root: _offline_tests(),
        privacy_tokens=lambda _root: (
            tokens
            if tokens is not None
            else (b"SYNTHETIC-PRIVATE-CONTACT", str(root))
        ),
    )


def _make_repo(root: Path) -> None:
    root.mkdir()
    (root / ".git").mkdir()


def test_validators_reject_scope_count_parity_and_effect_changes() -> None:
    assert preflight.validate_repository_snapshot(_repository_snapshot()) == (
        _repository_snapshot()
    )
    assert preflight.validate_bridge_manifest(_bridge_manifest()) == _bridge_manifest()
    assert preflight.validate_request_commitments(_commitments()) == _commitments()
    assert preflight.validate_offline_test_report(_offline_tests()) == _offline_tests()

    changed = _bridge_manifest()
    changed["confirmation_or_final_opened"] = True
    body = dict(changed)
    body.pop("bridge_sha256")
    changed["bridge_sha256"] = contract.canonical_sha256(body)
    _assert_code(
        "preflight_bridge_manifest_invalid",
        preflight.validate_bridge_manifest,
        changed,
    )
    changed_commitments = _commitments()
    changed_commitments["request_count"] = 74
    _assert_code(
        "preflight_request_commitments_invalid",
        preflight.validate_request_commitments,
        changed_commitments,
    )
    changed_repo = _repository_snapshot()
    changed_repo["source_inventory"][0]["literal_sha256"] = "f" * 64
    _assert_code(
        "preflight_repository_identity_invalid",
        preflight.validate_repository_snapshot,
        changed_repo,
    )


def test_run_preflight_seals_private_content_addressed_and_redacted_public(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    calls: dict[str, int] = {}
    artifact = preflight.run_preflight(
        root,
        dependencies=_dependencies(root, calls=calls),
    )
    assert calls == {"authenticate": 1, "projection": 1}
    assert artifact["status"] == "passed"
    assert artifact["development_authorized"] is False
    assert artifact["eligible_for_public_seal"] is True
    assert artifact["counts"]["canonical_requests"] == 75
    assert artifact["counts"]["pilots"] == 5
    assert artifact["counts"]["remaining"] == 70
    assert artifact["effect_counts"] == contract.build_effect_budgets()[
        "zero_effect_preflight"
    ]
    assert preflight.validate_public_preflight_artifact(artifact) == artifact

    public_path = root / contract.PREFLIGHT_ARTIFACT_PATH
    public_payload = public_path.read_bytes()
    assert public_payload == contract.canonical_json_bytes(artifact) + b"\n"
    assert b"SYNTHETIC-PRIVATE-CONTACT" not in public_payload
    assert str(root).encode() not in public_payload
    assert b"0000320193-24-000001" not in public_payload
    assert b"https://" not in public_payload
    assert b"canonical_request_body" not in public_payload
    assert not any("path" in key for key in artifact["aggregates"])

    private_root = root / contract.PRIVATE_PREFLIGHT_NAMESPACE
    manifests = list((private_root / "manifests").iterdir())
    assert len(manifests) == 1
    private_payload = manifests[0].read_bytes()
    assert manifests[0].name == f"{artifact['private_manifest_literal_sha256']}.json"
    private_value = json.loads(private_payload)
    assert preflight.validate_private_preflight_manifest(private_value) == private_value
    assert private_value["private_manifest_sha256"] == artifact[
        "private_manifest_sha256"
    ]
    assert b"SYNTHETIC-PRIVATE-CONTACT" not in private_payload
    assert str(root).encode() not in private_payload
    assert set(item.name for item in private_root.iterdir()) == {
        "preflight-intent.json",
        "manifests",
        "preflight-complete.json",
    }


def test_preflight_is_once_only_and_rerun_does_not_invoke_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    calls: dict[str, int] = {}
    dependencies = _dependencies(root, calls=calls)
    preflight.run_preflight(root, dependencies=dependencies)
    assert calls == {"authenticate": 1, "projection": 1}
    _assert_code(
        "preflight_already_consumed",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    assert calls == {"authenticate": 1, "projection": 1}


def test_nonzero_effect_and_later_stage_rejections_leave_consumed_intent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "nonzero"
    _make_repo(root)
    nonzero = contract.build_effect_budgets()["zero_effect_preflight"]
    nonzero["sec_requests"] = 1
    _assert_code(
        "preflight_nonzero_effect",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, effects=nonzero),
    )
    assert (root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "preflight-intent.json").is_file()
    failed = _assert_failed_preflight_artifact(root, "preflight_nonzero_effect")
    _assert_code(
        "preflight_already_consumed",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root),
    )
    assert json.loads((root / contract.PREFLIGHT_ARTIFACT_PATH).read_bytes()) == failed

    root = tmp_path / "later-stage"
    _make_repo(root)
    changed = _commitments()
    changed["confirmation_or_final_opened"] = True
    _assert_code(
        "preflight_request_commitments_invalid",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, commitments=changed),
    )
    _assert_failed_preflight_artifact(
        root, "preflight_request_commitments_invalid"
    )


def test_privacy_uncertainty_fails_before_publication(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    # The implementation commit is deliberately declared forbidden; it would
    # occur in both manifests, so the scan must fail closed.
    _assert_code(
        "preflight_private_privacy_scan_failed",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, tokens=(SHA1_A, str(root))),
    )
    failed = _assert_failed_preflight_artifact(
        root, "preflight_private_privacy_scan_failed"
    )
    payload = contract.canonical_json_bytes(failed)
    assert SHA1_A.encode("utf-8") not in payload
    assert str(root).encode("utf-8") not in payload


def test_dependency_failure_is_preserved_redacted_and_never_rerunnable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    base = _dependencies(root)
    calls = {"authenticate": 0}

    def fail_with_private_detail(_root: Path) -> object:
        calls["authenticate"] += 1
        raise RuntimeError("Synthetic Private Contact must never be published")

    dependencies = preflight.PreflightDependencies(
        **{**base.__dict__, "authenticate_source": fail_with_private_detail}
    )
    _assert_code(
        "preflight_dependency_failed",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    failed = _assert_failed_preflight_artifact(
        root, "preflight_dependency_failed"
    )
    assert "Synthetic Private Contact" not in json.dumps(failed)
    _assert_code(
        "preflight_already_consumed",
        preflight.run_preflight,
        root,
        dependencies=dependencies,
    )
    assert calls == {"authenticate": 1}
    assert list((root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "manifests").iterdir()) == []


def test_interrupted_failed_preflight_pending_is_promoted_before_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    target = root / contract.PREFLIGHT_ARTIFACT_PATH
    pending = target.with_name(f"{target.name}.v39-pending")
    real_replace = preflight.os.replace

    def interrupt_promotion(source: object, destination: object) -> None:
        if Path(source) == pending and Path(destination) == target:
            raise OSError("synthetic interruption after durable pending write")
        real_replace(source, destination)

    monkeypatch.setattr(preflight.os, "replace", interrupt_promotion)
    nonzero = contract.build_effect_budgets()["zero_effect_preflight"]
    nonzero["sec_requests"] = 1
    _assert_code(
        "preflight_failure_preservation_failed",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, effects=nonzero),
    )
    assert pending.is_file()
    assert not target.exists()

    monkeypatch.setattr(preflight.os, "replace", real_replace)
    calls: dict[str, int] = {}
    _assert_code(
        "preflight_already_consumed",
        preflight.run_preflight,
        root,
        dependencies=_dependencies(root, calls=calls),
    )
    assert calls == {}
    assert not pending.exists()
    _assert_failed_preflight_artifact(root, "preflight_nonzero_effect")


def test_public_and_private_self_hashes_reject_rehashed_semantic_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    artifact = preflight.run_preflight(root, dependencies=_dependencies(root))
    changed = copy.deepcopy(artifact)
    changed["counts"]["canonical_requests"] = 74
    unsigned = dict(changed)
    unsigned.pop("public_artifact_sha256")
    changed["public_artifact_sha256"] = contract.canonical_sha256(unsigned)
    _assert_code(
        "preflight_public_artifact_invalid",
        preflight.validate_public_preflight_artifact,
        changed,
    )

    private_path = next(
        (root / contract.PRIVATE_PREFLIGHT_NAMESPACE / "manifests").iterdir()
    )
    private_value = json.loads(private_path.read_bytes())
    private_value["request_commitments"]["request_count"] = 74
    unsigned = dict(private_value)
    unsigned.pop("private_manifest_sha256")
    private_value["private_manifest_sha256"] = contract.canonical_sha256(unsigned)
    _assert_code(
        "preflight_request_commitments_invalid",
        preflight.validate_private_preflight_manifest,
        private_value,
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


@pytest.mark.parametrize("descendant", ("pause", "continuation"))
def test_reconstruct_frozen_authority_rejects_merge_descendants(
    tmp_path: Path, descendant: str
) -> None:
    remote = tmp_path / f"{descendant}-remote.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True
    )
    root = tmp_path / f"{descendant}-work"
    subprocess.run(
        ["git", "clone", str(remote), str(root)], check=True, capture_output=True
    )
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "checkout", "-b", contract.BRANCH_NAME)
    preflight_path = root / contract.PREFLIGHT_ARTIFACT_PATH
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_bytes(b'{"synthetic_preflight":true}')
    _git(root, "add", "--", contract.PREFLIGHT_ARTIFACT_PATH)
    _git(root, "commit", "-m", "synthetic preflight")
    preflight_commit = _git(root, "rev-parse", "HEAD")

    first_parent = preflight_commit
    if descendant == "continuation":
        pause_path = root / contract.PAUSE_ARTIFACT_PATH
        pause_path.write_bytes(b'{"synthetic_pause":true}')
        _git(root, "add", "--", contract.PAUSE_ARTIFACT_PATH)
        _git(root, "commit", "-m", "synthetic single-parent pause")
        first_parent = _git(root, "rev-parse", "HEAD")
        target_path = root / contract.CONTINUATION_PREREGISTRATION_PATH
        target_relative = contract.CONTINUATION_PREREGISTRATION_PATH
        target_payload = b"# Synthetic merge continuation\n"
    else:
        target_path = root / contract.PAUSE_ARTIFACT_PATH
        target_relative = contract.PAUSE_ARTIFACT_PATH
        target_payload = b'{"synthetic_merge_pause":true}'

    first_parent_tree = _git(root, "rev-parse", f"{first_parent}^{{tree}}")
    side_parent = _git(
        root,
        "commit-tree",
        first_parent_tree,
        "-p",
        first_parent,
        "-m",
        "synthetic empty side parent",
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(target_payload)
    _git(root, "add", "--", target_relative)
    merge_tree = _git(root, "write-tree")
    merge_commit = _git(
        root,
        "commit-tree",
        merge_tree,
        "-p",
        first_parent,
        "-p",
        side_parent,
        "-m",
        f"synthetic merge {descendant}",
    )
    _git(
        root,
        "update-ref",
        f"refs/heads/{contract.BRANCH_NAME}",
        merge_commit,
        first_parent,
    )
    _git(root, "push", "-u", "origin", contract.BRANCH_NAME)
    assert len(_git(root, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3
    assert _git(root, "diff", "--name-status", first_parent, merge_commit) == (
        f"A\t{target_relative}"
    )
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""
    _assert_code(
        "execution_preflight_descendant_invalid",
        preflight.reconstruct_frozen_execution_authority,
        root,
    )


def test_pushed_execution_authority_replays_public_private_and_source_inventory(
    tmp_path: Path,
) -> None:
    source_repo = Path(__file__).resolve().parents[1]
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", str(remote), str(work)], check=True, capture_output=True)
    _git(work, "config", "core.longpaths", "true")
    _git(work, "config", "user.name", "Synthetic Test")
    _git(work, "config", "user.email", "synthetic@example.invalid")
    _git(work, "fetch", str(source_repo), contract.PREREGISTRATION_COMMIT)
    _git(work, "checkout", "-b", contract.BRANCH_NAME, contract.PREREGISTRATION_COMMIT)
    for index, relative in enumerate(contract.IMPLEMENTATION_ALLOWED_PATHS):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"# synthetic implementation {index}\n".encode("utf-8"))
    _git(work, "add", "--", *contract.IMPLEMENTATION_ALLOWED_PATHS)
    _git(work, "commit", "-m", "synthetic v3.9 implementation")
    _git(work, "push", "-u", "origin", contract.BRANCH_NAME)

    config = work / "data" / "local_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    fake_contact = "Synthetic Contact synthetic@example.invalid"
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": fake_contact}}),
        encoding="utf-8",
    )
    dependencies = _dependencies(work)
    dependencies = preflight.PreflightDependencies(
        inspect_repository=preflight.inspect_pushed_implementation,
        authenticate_source=dependencies.authenticate_source,
        build_projection=dependencies.build_projection,
        build_request_commitments=dependencies.build_request_commitments,
        effect_snapshot=dependencies.effect_snapshot,
        run_offline_tests=dependencies.run_offline_tests,
        privacy_tokens=lambda root: (
            fake_contact,
            str(root),
            str(root / preflight.V38_PRIVATE_ROOT),
            str(root / contract.PRIVATE_NAMESPACE),
        ),
    )
    artifact = preflight.run_preflight(work, dependencies=dependencies)
    _git(work, "add", "--", contract.PREFLIGHT_ARTIFACT_PATH)
    _git(work, "commit", "-m", "synthetic v3.9 preflight evidence")
    _git(work, "push", "origin", contract.BRANCH_NAME)

    receipt = preflight.authenticate_execution_preflight(work)
    authority = preflight.load_execution_authority(work)
    assert receipt["development_authorized"] is True
    assert receipt["authority_scope"] == "direct_pushed_preflight"
    assert receipt["pushed_preflight_gate_passed"] is True
    assert receipt["implementation_commit"] == artifact["implementation_commit"]
    assert receipt["public_artifact_sha256"] == artifact["public_artifact_sha256"]
    assert receipt["canonical_requests_sha256"] == SHA64["requests"]
    assert set(authority) == {
        "plan",
        "attempt",
        "implementation",
        "preflight",
        "source",
        "science",
        "effect_budget",
        "request_order",
        "pilot_order",
    }
    assert authority["attempt"] == contract.DEVELOPMENT_ATTEMPT_ID
    assert authority["request_order"]["canonical_requests_sha256"] == SHA64["requests"]
    assert fake_contact not in json.dumps(authority, sort_keys=True)
    assert str(work) not in json.dumps(authority, sort_keys=True)

    pause_path = work / contract.PAUSE_ARTIFACT_PATH
    pause_path.parent.mkdir(parents=True, exist_ok=True)
    pause_path.write_bytes(b'{"synthetic_pause":true}\n')
    _git(work, "add", "--", contract.PAUSE_ARTIFACT_PATH)
    _git(work, "commit", "-m", "synthetic clean pause")
    _git(work, "push", "origin", contract.BRANCH_NAME)
    assert preflight.reconstruct_frozen_execution_authority(work) == authority
    assert preflight.load_execution_authority(work) == authority
    _assert_code(
        "execution_preflight_ancestry_invalid",
        preflight.authenticate_execution_preflight,
        work,
    )

    continuation_path = work / contract.CONTINUATION_PREREGISTRATION_PATH
    continuation_path.parent.mkdir(parents=True, exist_ok=True)
    continuation_path.write_bytes(b"# Synthetic continuation\n")
    _git(work, "add", "--", contract.CONTINUATION_PREREGISTRATION_PATH)
    _git(work, "commit", "-m", "synthetic continuation preregistration")
    _git(work, "push", "origin", contract.BRANCH_NAME)
    assert preflight.reconstruct_frozen_execution_authority(work) == authority
    assert preflight.load_execution_authority(work) == authority


def test_load_private_contact_uses_only_synthetic_ignored_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _make_repo(root)
    config = root / "data" / "local_config.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps({"secrets": {"sec_user_agent": "Synthetic Contact"}}),
        encoding="utf-8",
    )
    assert preflight.load_private_contact(root) == "Synthetic Contact"


def test_privacy_scanners_reject_json_escaped_windows_paths_and_parsed_values() -> None:
    windows_path = r"C:\Users\Synthetic\private evidence"
    escaped = json.dumps(
        {"nested": {"value": windows_path}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert windows_path.encode("utf-8") not in escaped
    _assert_code(
        "preflight_private_privacy_scan_failed",
        preflight._scan_private_bytes,
        escaped,
        (windows_path.encode("utf-8"),),
    )

    contact = "Synthetíc Contact"
    unicode_escaped = json.dumps(
        {"contact": contact}, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    assert contact.encode("utf-8") not in unicode_escaped
    _assert_code(
        "preflight_public_privacy_scan_failed",
        preflight._scan_public_bytes,
        unicode_escaped,
        (contact.encode("utf-8"),),
    )


@pytest.mark.parametrize(
    "private_token",
    ("Synthetíc Private Contact", "Synthetíc Private Contact".encode("utf-8")),
)
def test_privacy_token_collection_rejects_non_ascii_without_echo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    private_token: str | bytes,
) -> None:
    with pytest.raises(preflight.V39PreflightError) as captured:
        preflight._privacy_token_bytes(
            tmp_path,
            lambda _root: ("safe-ascii-token", private_token),
        )
    assert captured.value.code == "preflight_privacy_tokens_invalid"
    assert "Synthetíc Private Contact" not in str(captured.value)
    assert "Synthetíc Private Contact" not in repr(captured.value)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_result_effect_validation_recursively_thaws_store_mapping_proxies() -> None:
    snapshot = SimpleNamespace(
        request_intent_count=0,
        response_count=0,
        checkpoint_count=0,
        yahoo_body_bytes=0,
        yahoo_intent_count=0,
        identity_intent_count=0,
        gemma_intent_count=0,
        yahoo_response_count=0,
        identity_response_count=0,
        gemma_response_count=0,
    )
    ordinary = runner.build_effect_report(snapshot)
    frozen = MappingProxyType(
        {
            key: (
                MappingProxyType(dict(value))
                if isinstance(value, dict)
                else value
            )
            for key, value in ordinary.items()
        }
    )
    assert preflight._validate_effect_report(
        frozen,
        snapshot=snapshot,
        completed_terminal=False,
        parent_kind="preflight",
    ) == ordinary


def _preflight_receipt_for_result(
    *, implementation: str, implementation_tree: str, preflight_commit: str, preflight_tree: str
) -> dict[str, object]:
    return {
        "schema_version": preflight.EXECUTION_AUTHORITY_SCHEMA_VERSION,
        "authority_scope": "descendant_reconstruction",
        "stage": contract.DEVELOPMENT_COMMAND,
        "branch": contract.BRANCH_NAME,
        "preflight_commit": preflight_commit,
        "preflight_tree": preflight_tree,
        "implementation_commit": implementation,
        "implementation_tree": implementation_tree,
        "public_artifact_git_blob_sha1": "a" * 40,
        "public_artifact_sha256": "a" * 64,
        "public_artifact_literal_sha256": "b" * 64,
        "private_manifest_sha256": "c" * 64,
        "private_manifest_literal_sha256": "d" * 64,
        "bridge_sha256": _bridge_manifest()["bridge_sha256"],
        "canonical_requests_sha256": SHA64["requests"],
        "pilot_order_sha256": SHA64["pilot"],
        "remaining_order_sha256": SHA64["remaining"],
        "production_source_inventory_sha256": "e" * 64,
        "test_source_inventory_sha256": "f" * 64,
        "effect_counts": contract.build_effect_budgets()["zero_effect_preflight"],
        "pushed_preflight_gate_passed": True,
        "development_authorized": False,
        "execution_authority_sha256": "0" * 64,
    }


class _FakeReadOnlyStore:
    def __init__(self, snapshot: object, receipt: object | None) -> None:
        self.snapshot = snapshot
        self._receipt = receipt
        self.closed = False

    def committed_terminal_evidence(self) -> object:
        if self._receipt is None:
            raise RuntimeError("synthetic_terminal_not_committed")
        return self._receipt

    def close(self) -> None:
        self.closed = True


def _synthetic_continuation_prerequisites(
    authority: dict[str, object],
) -> tuple[object, dict[str, object], dict[str, object], dict[str, object]]:
    execution_calls = tuple(
        SimpleNamespace(
            execution_ordinal=index,
            request={
                "request_sha256": contract.sha256_bytes(
                    f"synthetic-request-{index:03d}".encode("ascii")
                )
            },
            request_byte_count=1_000 + index,
        )
        for index in range(1, contract.DEVELOPMENT_DOCUMENT_COUNT + 1)
    )
    plan_body = {
        "remaining_order_sha256": authority["request_order"][
            "remaining_order_sha256"
        ]
    }
    plan = SimpleNamespace(
        execution_calls=execution_calls,
        manifest={
            **plan_body,
            "model_plan_sha256": contract.canonical_sha256(plan_body),
        },
    )
    generation_events = [
        contract.sha256_bytes(f"pilot-event-{index}".encode("ascii"))
        for index in range(1, contract.DEVELOPMENT_PILOT_COUNT + 1)
    ]
    guard_body = {
        "schema_version": runner.RUNTIME_GUARD_SCHEMA_VERSION,
        "segment_id": "pilot",
        "store_segment_id": "initial",
        "generation_count": contract.DEVELOPMENT_PILOT_COUNT,
        "pre_runtime_receipt_sha256": "a" * 64,
        "post_runtime_receipt_sha256": "b" * 64,
        "stable_runtime_identity_sha256": "c" * 64,
        "ordered_generation_response_event_sha256s": generation_events,
        "ordered_generation_response_events_sha256": contract.canonical_sha256(
            generation_events
        ),
        "raw_show_hash_is_diagnostic_only": True,
        "modified_at_is_excluded_only": True,
        "identity_http_request_count": 4,
        "retry_count": 0,
    }
    pilot_guard = {
        **guard_body,
        "segment_guard_sha256": contract.canonical_sha256(guard_body),
    }
    durations = [600_000_000_000 + index for index in range(5)]
    pilot_rows = [
        {
            "execution_ordinal": call.execution_ordinal,
            "request_sha256": call.request["request_sha256"],
            "request_byte_count": call.request_byte_count,
            "duration_ns": duration,
        }
        for call, duration in zip(
            execution_calls[: contract.DEVELOPMENT_PILOT_COUNT],
            durations,
            strict=True,
        )
    ]
    latency_body = {
        "schema_version": runner.LATENCY_RECEIPT_SCHEMA_VERSION,
        "selection": "five_largest_request_bytes_desc_accession_asc",
        "remaining_order": "availability_acceptance_accession_ascending",
        "pilot_rows": pilot_rows,
        "pilot_order_sha256": contract.canonical_sha256(pilot_rows),
        "pilot_count": contract.DEVELOPMENT_PILOT_COUNT,
        "remaining_count": contract.DEVELOPMENT_REMAINING_COUNT,
        "formula": "sum(pilot_duration_ns)+70*max(pilot_duration_ns)",
        "projected_ns": contract.projected_pilot_ns(durations),
        "threshold_ns": contract.PILOT_PROJECTED_THRESHOLD_NS,
        "pause_required": True,
        "timing_interval": (
            "monotonic_ns_after_durable_intent_before_transport_through_"
            "bounded_body_framing_and_close_before_persistence_or_semantic_parse"
        ),
    }
    latency = {
        **latency_body,
        "latency_receipt_sha256": contract.canonical_sha256(latency_body),
    }
    pause_request_count = (
        contract.YAHOO_REQUEST_COUNT + 4 + contract.DEVELOPMENT_PILOT_COUNT
    )
    pause_snapshot = SimpleNamespace(
        yahoo_intent_count=contract.YAHOO_REQUEST_COUNT,
        identity_intent_count=4,
        gemma_intent_count=contract.DEVELOPMENT_PILOT_COUNT,
        yahoo_response_count=contract.YAHOO_REQUEST_COUNT,
        identity_response_count=4,
        gemma_response_count=contract.DEVELOPMENT_PILOT_COUNT,
        request_intent_count=pause_request_count,
        response_count=pause_request_count,
        checkpoint_count=pause_request_count + 1,
        yahoo_body_bytes=123,
    )
    pause = runner.build_public_pause_artifact(
        authority=authority,
        plan=plan,
        pilot_guard=pilot_guard,
        latency_receipt=latency,
        effect_report=runner.build_effect_report(pause_snapshot),
    )
    return plan, pilot_guard, latency, pause


def _make_result_gate_repo(
    tmp_path: Path,
    *,
    commit_result: bool = True,
    completed: bool = False,
    continued: bool = False,
    mutated_pause: bool = False,
    arbitrary_continuation: bool = False,
) -> dict[str, object]:
    assert not continued or completed
    assert not (mutated_pause or arbitrary_continuation) or continued
    remote = tmp_path / "result-remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    root = tmp_path / "result-work"
    subprocess.run(["git", "clone", str(remote), str(root)], check=True, capture_output=True)
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "checkout", "-b", contract.BRANCH_NAME)
    comparison = (
        "| Approach | Design | Development | Real money | Decision |\n"
        "|---|---|---|---|---|\n"
        "| SEC/Gemma lean evidence v3.8 | Frozen | Passed | No | Frozen |\n"
    ).encode("utf-8")
    comparison_path = root / contract.COMPARISON_PATH
    comparison_path.parent.mkdir(parents=True)
    comparison_path.write_bytes(comparison)
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    _git(root, "add", "--", ".gitignore", contract.COMPARISON_PATH)
    _git(root, "commit", "-m", "synthetic implementation parent")
    implementation = _git(root, "rev-parse", "HEAD")
    implementation_tree = _git(root, "rev-parse", "HEAD^{tree}")

    preflight_path = root / contract.PREFLIGHT_ARTIFACT_PATH
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.write_bytes(b'{"synthetic_preflight":true}\n')
    _git(root, "add", "--", contract.PREFLIGHT_ARTIFACT_PATH)
    _git(root, "commit", "-m", "synthetic preflight")
    preflight_commit = _git(root, "rev-parse", "HEAD")
    preflight_tree = _git(root, "rev-parse", "HEAD^{tree}")
    receipt_authority = _preflight_receipt_for_result(
        implementation=implementation,
        implementation_tree=implementation_tree,
        preflight_commit=preflight_commit,
        preflight_tree=preflight_tree,
    )
    authority = preflight._store_execution_authority(receipt_authority)
    plan, pilot_guard, pilot_latency, canonical_pause = (
        _synthetic_continuation_prerequisites(authority)
    )
    invocation_parent = preflight_commit
    invocation_parent_kind = "preflight"
    continuation_sha256: str | None = None
    committed_pause: dict[str, object] | None = None
    continuation: bytes | None = None
    if continued:
        committed_pause = copy.deepcopy(canonical_pause)
        if mutated_pause:
            committed_pause["implementation_commit"] = "f" * 40
            pause_body = dict(committed_pause)
            pause_body.pop("pause_artifact_sha256")
            committed_pause["pause_artifact_sha256"] = contract.canonical_sha256(
                pause_body
            )
        pause_path = root / contract.PAUSE_ARTIFACT_PATH
        pause_path.write_bytes(contract.canonical_json_bytes(committed_pause))
        _git(root, "add", "--", contract.PAUSE_ARTIFACT_PATH)
        _git(root, "commit", "-m", "synthetic pause")
        continuation = (
            b"# Arbitrary but historically hash-bound continuation\n"
            if arbitrary_continuation
            else runner.build_continuation_preregistration(committed_pause)
        )
        continuation_path = root / contract.CONTINUATION_PREREGISTRATION_PATH
        continuation_path.parent.mkdir(parents=True, exist_ok=True)
        continuation_path.write_bytes(continuation)
        _git(root, "add", "--", contract.CONTINUATION_PREREGISTRATION_PATH)
        _git(root, "commit", "-m", "synthetic continuation")
        invocation_parent = _git(root, "rev-parse", "HEAD")
        invocation_parent_kind = "continuation"
        continuation_sha256 = contract.sha256_bytes(continuation)

    terminal_status = "completed" if completed else "rejected"
    terminal_code = "development_pass" if completed else "synthetic_failure"
    yahoo_count = contract.YAHOO_REQUEST_COUNT if completed else 0
    identity_count = (8 if continued else 4) if completed else 0
    generation_count = contract.DEVELOPMENT_DOCUMENT_COUNT if completed else 0
    total_count = yahoo_count + identity_count + generation_count
    summaries = (
        (
            SimpleNamespace(generation_count=contract.DEVELOPMENT_PILOT_COUNT),
            SimpleNamespace(generation_count=contract.DEVELOPMENT_REMAINING_COUNT),
        )
        if continued
        else (SimpleNamespace(generation_count=contract.DEVELOPMENT_DOCUMENT_COUNT),)
        if completed
        else ()
    )
    snapshot = SimpleNamespace(
        status=terminal_status,
        terminal_code=terminal_code,
        journal_head_sha256="1" * 64,
        request_intent_count=total_count,
        yahoo_intent_count=yahoo_count,
        identity_intent_count=identity_count,
        gemma_intent_count=generation_count,
        response_count=total_count,
        checkpoint_count=total_count,
        yahoo_response_count=yahoo_count,
        yahoo_body_bytes=123 if completed else 0,
        identity_response_count=identity_count,
        gemma_response_count=generation_count,
        segment_summaries=summaries,
        paused=continued,
        continuation_authorized=continued,
        continuation_sha256=continuation_sha256,
        continuation_commit=invocation_parent if continued else None,
        continuation_permission_sha256="5" * 64 if continued else None,
        market_values_opened=completed,
        model_responses_opened=completed,
    )
    effect_report = runner.build_effect_report(snapshot)
    if completed:
        semantic: dict[str, object] = {}
        deterministic: dict[str, object] = {}
        no_leverage: dict[str, object] = {}
        science_summary = {
            "gate_report": {"passed": True, "failed_checks": []},
            "no_leverage_proofs": no_leverage,
            "no_leverage_proofs_sha256": contract.canonical_sha256(no_leverage),
        }
        material_body = {
            "schema_version": runner.PRIVATE_TERMINAL_SCHEMA_VERSION,
            "stage": contract.DEVELOPMENT_COMMAND,
            "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
            "invocation_parent": invocation_parent,
            "invocation_parent_kind": invocation_parent_kind,
            "route": "paused_resumed" if continued else "normal",
            "authority_sha256": contract.canonical_sha256(authority),
            "bridge_manifest": _bridge_manifest(),
            "model_plan_manifest": {},
            "runtime_segment_guards": [{}, {}] if continued else [{}],
            "runtime_aggregate": {},
            "latency_receipt": {},
            "stage_slice_sha256": "4" * 64,
            "semantic_payload": semantic,
            "semantic_payload_sha256": contract.canonical_sha256(semantic),
            "deterministic_payload": deterministic,
            "deterministic_payload_sha256": contract.canonical_sha256(
                deterministic
            ),
            "science_summary": science_summary,
            "science_summary_sha256": contract.canonical_sha256(science_summary),
            "effect_report": effect_report,
            "market_values_opened": True,
            "model_responses_opened": True,
            "confirmation_and_final_opened": False,
            "raw_sec_yahoo_or_gemma_response_copied": False,
        }
        material = {
            **material_body,
            "private_terminal_material_sha256": contract.canonical_sha256(
                material_body
            ),
        }
    else:
        material = runner.build_failure_terminal_material(
            authority=authority,
            terminal_code=terminal_code,
            terminal_status=terminal_status,
            effect_report=effect_report,
            snapshot=SimpleNamespace(
                journal_head_sha256="2" * 64, segment_summaries=()
            ),
            invocation_parent=invocation_parent,
            invocation_parent_kind=invocation_parent_kind,
            market_values_opened=False,
            model_responses_opened=False,
            source_authenticated=False,
        )
    terminal_envelope = {
        "schema_version": preflight.TERMINAL_EVIDENCE_SCHEMA_VERSION,
        "authority_sha256": contract.canonical_sha256(authority),
        "journal_head_before_terminal_sha256": "2" * 64,
        "status": terminal_status,
        "terminal_code": terminal_code,
        "evidence": material,
    }
    terminal_envelope_bytes = contract.canonical_json_bytes(terminal_envelope)
    terminal_receipt = SimpleNamespace(
        event_sha256="1" * 64,
        evidence_sha256=contract.sha256_bytes(terminal_envelope_bytes),
        evidence_bytes=len(terminal_envelope_bytes),
        journal_head_before_terminal_sha256="2" * 64,
        status=terminal_status,
        terminal_code=terminal_code,
        evidence=material,
    )
    public = runner.build_public_terminal_artifact(
        authority=authority, terminal_receipt=terminal_receipt
    )
    if commit_result:
        result_path = root / contract.RESULT_ARTIFACT_PATH
        result_path.write_bytes(contract.canonical_json_bytes(public))
        comparison_path.write_bytes(runner.build_comparison_update(comparison, public))
        _git(root, "add", "--", contract.RESULT_ARTIFACT_PATH, contract.COMPARISON_PATH)
        _git(root, "commit", "-m", "synthetic terminal result")
    _git(root, "push", "-u", "origin", contract.BRANCH_NAME)

    v39 = root / contract.PRIVATE_NAMESPACE
    v38 = root / preflight.V38_PRIVATE_ROOT
    v39.mkdir(parents=True)
    v38.mkdir(parents=True)
    (v39 / "synthetic.json").write_bytes(b'{"sealed":true}')
    (v38 / "synthetic.json").write_bytes(b'{"frozen":true}')
    projection = SimpleNamespace(manifest=_bridge_manifest())
    store = _FakeReadOnlyStore(snapshot, terminal_receipt)
    calls: list[str] = []

    def replay_failure(**_kwargs: object) -> dict[str, object]:
        calls.append("failure_replay")
        return copy.deepcopy(material)

    def replay_completed(**_kwargs: object) -> dict[str, object]:
        calls.append("completed_replay")
        return copy.deepcopy(material)

    def rebuild_pilot(**_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        calls.append("pilot_replay")
        return copy.deepcopy(pilot_guard), copy.deepcopy(pilot_latency)

    dependencies = preflight.PushedResultDependencies(
        authenticate_preflight_revision=lambda _root, _commit: receipt_authority,
        load_private_contact=lambda _root: "Synthetic Private Contact",
        authenticate_source=lambda _root, _contact: calls.append("source") or object(),
        build_projection=lambda _source: projection,
        rebuild_attempt_context=lambda _execution, _projection: {
            "authority": authority,
            "plan": plan,
        },
        open_store=lambda _path, _authority: store,
        build_effect_report=runner.build_effect_report,
        build_public_effect_report=runner.build_public_effect_report,
        rebuild_pilot_evidence=rebuild_pilot,
        build_public_pause_artifact=runner.build_public_pause_artifact,
        build_continuation_preregistration=runner.build_continuation_preregistration,
        replay_completed_terminal=replay_completed,
        replay_failure_terminal=replay_failure,
        build_public_terminal_artifact=runner.build_public_terminal_artifact,
        build_comparison_update=runner.build_comparison_update,
        privacy_tokens=lambda path, contact: (
            contact,
            str(path),
            str(path / preflight.V38_PRIVATE_ROOT),
            str(path / contract.PRIVATE_NAMESPACE),
        ),
    )
    return {
        "root": root,
        "dependencies": dependencies,
        "authority": authority,
        "receipt_authority": receipt_authority,
        "snapshot": snapshot,
        "terminal_receipt": terminal_receipt,
        "public": public,
        "material": material,
        "store": store,
        "calls": calls,
        "preflight_commit": preflight_commit,
        "invocation_parent": invocation_parent,
        "comparison_before": comparison,
        "canonical_pause": canonical_pause,
        "committed_pause": committed_pause,
        "continuation": continuation,
    }


def test_pushed_result_gate_replays_failure_and_returns_only_redacted_receipt(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path)
    root = fixture["root"]
    before_head = _git(root, "rev-parse", "HEAD")
    receipt = preflight.authenticate_pushed_result(
        root, dependencies=fixture["dependencies"]
    )
    assert receipt["status"] == "passed"
    assert receipt["development_outcome"] == "rejected"
    assert receipt["route"] == "normal"
    assert receipt["confirmation_preregistration_eligible"] is False
    assert receipt["confirmation_execution_authorized"] is False
    assert receipt["final_execution_authorized"] is False
    assert receipt["real_money_authorized"] is False
    assert receipt["gates"]["failure_material_replayed"] is True
    assert receipt["gates"]["deterministic_science_replayed"] is False
    assert receipt["pushed_result_gate_sha256"] == contract.canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "pushed_result_gate_sha256"
        }
    )
    serialized = contract.canonical_json_bytes(receipt)
    assert b"Synthetic Private Contact" not in serialized
    assert str(root).encode() not in serialized
    assert fixture["calls"] == ["source", "failure_replay"]
    assert fixture["store"].closed is True
    assert _git(root, "rev-parse", "HEAD") == before_head
    assert _git(root, "status", "--porcelain=v1", "--untracked-files=all") == ""


def test_load_execution_authority_reconstructs_clean_pushed_result_without_gate_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_result_gate_repo(tmp_path)
    calls: list[tuple[str, bool]] = []

    def authenticate(
        _root: Path,
        *,
        preflight_commit: str,
        direct_head: bool,
        allow_publication_dirty: bool = False,
    ) -> dict[str, object]:
        calls.append((preflight_commit, direct_head))
        assert allow_publication_dirty is False
        return fixture["receipt_authority"]

    monkeypatch.setattr(preflight, "_authenticate_preflight_revision", authenticate)
    authority = preflight.load_execution_authority(fixture["root"])
    assert authority == fixture["authority"]
    assert calls == [(fixture["preflight_commit"], False)]
    assert "pushed_result_gate_passed" not in authority


def test_pushed_result_cli_is_separate_and_never_invokes_one_shot_preflight(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        preflight,
        "run_preflight",
        lambda *_args, **_kwargs: pytest.fail("one-shot preflight was invoked"),
    )
    monkeypatch.setattr(
        preflight,
        "authenticate_pushed_result",
        lambda *_args, **_kwargs: {
            "status": "passed",
            "pushed_result_gate_sha256": "a" * 64,
        },
    )
    assert preflight.main(["--pushed-result", "."]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "passed",
        "pushed_result_gate_sha256": "a" * 64,
    }


def test_pushed_result_gate_rejects_independent_replay_mismatch(tmp_path: Path) -> None:
    fixture = _make_result_gate_repo(tmp_path)
    dependencies = fixture["dependencies"]
    mismatched = preflight.PushedResultDependencies(
        **{
            **dependencies.__dict__,
            "replay_failure_terminal": lambda **_kwargs: {
                "private_terminal_material_sha256": "0" * 64
            },
        }
    )
    _assert_code(
        "pushed_result_independent_replay_failed",
        preflight.authenticate_pushed_result,
        fixture["root"],
        dependencies=mismatched,
    )


def test_pushed_result_gate_requires_completed_science_and_exact_effect_replay(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, completed=True)
    receipt = preflight.authenticate_pushed_result(
        fixture["root"], dependencies=fixture["dependencies"]
    )
    assert receipt["development_outcome"] == "passed"
    assert receipt["confirmation_preregistration_eligible"] is True
    assert receipt["gates"]["deterministic_science_replayed"] is True
    assert receipt["gates"]["failure_material_replayed"] is False
    assert receipt["science_summary_sha256"] == contract.canonical_sha256(
        fixture["material"]["science_summary"]
    )
    assert fixture["calls"] == ["source", "completed_replay"]


def test_pushed_result_gate_reconstructs_f_s_c_and_binds_exact_continuation_commit(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, completed=True, continued=True)
    receipt = preflight.authenticate_pushed_result(
        fixture["root"], dependencies=fixture["dependencies"]
    )
    assert receipt["route"] == "paused_resumed"
    assert receipt["authorized_parent_kind"] == "continuation"
    assert receipt["authorized_parent"] == fixture["invocation_parent"]
    assert receipt["development_outcome"] == "passed"
    assert fixture["committed_pause"] == fixture["canonical_pause"]
    assert fixture["continuation"] == runner.build_continuation_preregistration(
        fixture["canonical_pause"]
    )
    assert fixture["calls"] == ["source", "pilot_replay", "completed_replay"]


def test_pushed_result_gate_rejects_same_continuation_bytes_under_other_commit(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, completed=True, continued=True)
    assert fixture["snapshot"].continuation_sha256 == contract.sha256_bytes(
        fixture["continuation"]
    )
    assert fixture["snapshot"].continuation_commit == fixture["invocation_parent"]
    fixture["snapshot"].continuation_commit = "7" * 40
    _assert_code(
        "pushed_result_continuation_binding_invalid",
        preflight.authenticate_pushed_result,
        fixture["root"],
        dependencies=fixture["dependencies"],
    )


def test_pushed_result_gate_rejects_self_hashed_pause_mutation(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(
        tmp_path, completed=True, continued=True, mutated_pause=True
    )
    committed_pause = fixture["committed_pause"]
    assert contract.validate_self_sha256(
        committed_pause, field="pause_artifact_sha256"
    ) == committed_pause
    assert committed_pause["implementation_commit"] == "f" * 40
    assert committed_pause != fixture["canonical_pause"]
    _assert_code(
        "pushed_result_pause_artifact_invalid",
        preflight.authenticate_pushed_result,
        fixture["root"],
        dependencies=fixture["dependencies"],
    )


def test_pushed_result_gate_rejects_arbitrary_hash_bound_continuation_document(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(
        tmp_path, completed=True, continued=True, arbitrary_continuation=True
    )
    arbitrary = fixture["continuation"]
    assert fixture["snapshot"].continuation_sha256 == contract.sha256_bytes(
        arbitrary
    )
    assert arbitrary != runner.build_continuation_preregistration(
        fixture["canonical_pause"]
    )
    _assert_code(
        "pushed_result_continuation_invalid",
        preflight.authenticate_pushed_result,
        fixture["root"],
        dependencies=fixture["dependencies"],
    )


def test_publication_recovery_accepts_only_exact_idempotent_partial_bytes(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, commit_result=False)
    root = fixture["root"]
    # Reproduce the exact two public files as an interrupted, uncommitted
    # publication while both local and remote authority remain exactly F.
    public = fixture["public"]
    result_path = root / contract.RESULT_ARTIFACT_PATH
    result_bytes = contract.canonical_json_bytes(public)
    result_pending = root / f"{contract.RESULT_ARTIFACT_PATH}.v39-pending"
    result_pending.write_bytes(result_bytes)
    comparison_path = root / contract.COMPARISON_PATH
    derived = runner.build_comparison_update(fixture["comparison_before"], public)
    pending = root / f"{contract.COMPARISON_PATH}.v39-pending"
    pending.write_bytes(derived)

    recovery = preflight.PublicationRecoveryDependencies(
        authenticate_preflight_revision=lambda _root, _commit: fixture[
            "receipt_authority"
        ],
        open_store=lambda _path, _authority: _FakeReadOnlyStore(
            fixture["snapshot"], fixture["terminal_receipt"]
        ),
        build_public_terminal_artifact=runner.build_public_terminal_artifact,
        build_comparison_update=runner.build_comparison_update,
    )
    assert preflight.load_publication_recovery_authority(
        root, dependencies=recovery
    ) == fixture["authority"]

    pending.write_bytes(b"tampered")
    _assert_code(
        "publication_recovery_pending_invalid",
        preflight.load_publication_recovery_authority,
        root,
        dependencies=recovery,
    )


def test_publication_recovery_accepts_only_exact_pause_pending_bytes(
    tmp_path: Path,
) -> None:
    fixture = _make_result_gate_repo(tmp_path, commit_result=False)
    root = fixture["root"]
    pause_body = {
        "status": "paused_for_justification",
        "stage": contract.DEVELOPMENT_COMMAND,
        "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
        "branch": contract.BRANCH_NAME,
        "preflight_commit": fixture["preflight_commit"],
        "privacy_passed": True,
    }
    pause = {
        **pause_body,
        "pause_artifact_sha256": contract.canonical_sha256(pause_body),
    }
    pause_bytes = contract.canonical_json_bytes(pause)
    pause_path = root / contract.PAUSE_ARTIFACT_PATH
    pause_pending = root / f"{contract.PAUSE_ARTIFACT_PATH}.v39-pending"
    pause_pending.write_bytes(pause_bytes)
    pause_snapshot = SimpleNamespace(
        status="paused", paused=True, continuation_authorized=False
    )
    recovery = preflight.PublicationRecoveryDependencies(
        authenticate_preflight_revision=lambda _root, _commit: fixture[
            "receipt_authority"
        ],
        open_store=lambda _path, _authority: _FakeReadOnlyStore(
            pause_snapshot, None
        ),
        build_public_terminal_artifact=runner.build_public_terminal_artifact,
        build_comparison_update=runner.build_comparison_update,
        rebuild_public_pause_artifact=lambda **_kwargs: pause,
    )
    assert preflight.load_publication_recovery_authority(
        root, dependencies=recovery
    ) == fixture["authority"]

    pause_pending.write_bytes(b"truncated")
    _assert_code(
        "publication_recovery_pause_invalid",
        preflight.load_publication_recovery_authority,
        root,
        dependencies=recovery,
    )
