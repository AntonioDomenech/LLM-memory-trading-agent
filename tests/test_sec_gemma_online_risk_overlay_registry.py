from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
    validate_implementation_manifest,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    BRANCH_NAME,
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    FINAL_ATTEMPT_ID,
    FINAL_REGISTRY_AUTHORIZATION_FIELDS,
    FINAL_REGISTRY_SUCCESSOR_FIELDS,
    SOURCE_PIN_FILES,
    SOURCE_PINS,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
    FINAL_REGISTRY_SUCCESSOR,
    PUBLICATION_GENESIS_SHA256,
    REGISTERED_UNRUN,
    SCORED_PASS,
    TERMINAL_PASS,
    ExternalGitTagPublisher,
)
from agent_benchmark.sec_gemma_online_risk_overlay_registry import (
    FINAL_REGISTRY_AUTHORIZATION_SCHEMA_VERSION,
    FINAL_REGISTRY_SUCCESSOR_SCHEMA_VERSION,
    FINAL_REGISTRY_VERIFIER_ID,
    FinalRegistryAuthorizer,
    SecGemmaOnlineRiskOverlayRegistryError,
    VerifiedFinalRegistryAuthorization,
    build_final_registry_successor,
    is_verified_final_registry_authorization,
    validate_predecessor_registry_pin,
)
from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    CONTRACT_SOURCE_PATH,
    EXPECTED_ORIGIN_URL,
    PREREGISTRATION_COMMIT,
    REQUIRED_NEW_SOURCE_PATHS,
    SOURCE_VERIFICATION_SCHEMA_VERSION,
)
from tests.sec_gemma_online_risk_overlay_helpers import (
    build_synthetic_numerical_time_distributions,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _manifest(commit: str) -> dict[str, Any]:
    contract_source = {
        "path": CONTRACT_SOURCE_PATH,
        "sha256": _digest("contract-source"),
    }
    reused = [
        {
            "role": role,
            "path": SOURCE_PIN_FILES[role],
            "sha256": SOURCE_PINS[role],
        }
        for role in sorted(SOURCE_PINS)
    ]
    new = [
        {
            "role": role,
            "path": REQUIRED_NEW_SOURCE_PATHS[role],
            "sha256": _digest(f"new:{role}"),
        }
        for role in sorted(REQUIRED_NEW_SOURCE_PATHS)
    ]
    numerical_time_distributions = (
        build_synthetic_numerical_time_distributions(
            digest=_digest,
            canonical_sha256=canonical_sha256,
        )
    )
    dependency_material = {
        "dependency_sources": [],
        "external_distributions": [],
        "numerical_time_distributions": numerical_time_distributions,
    }
    dependency_closure = canonical_sha256(dependency_material)
    source_tree = {
        "contract_source": contract_source,
        "reused_sources": reused,
        "new_sources": new,
        **dependency_material,
        "dependency_closure_sha256": dependency_closure,
    }
    verification_material = {
        "schema_version": SOURCE_VERIFICATION_SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "branch": BRANCH_NAME,
        "origin_url": EXPECTED_ORIGIN_URL,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "head_commit": commit,
        "upstream_commit": commit,
        "contract_source": contract_source,
        "reused_sources": reused,
        "new_sources": new,
        **dependency_material,
        "dependency_closure_sha256": dependency_closure,
    }
    body = {
        "schema_version": IMPLEMENTATION_MANIFEST_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "contract_source": contract_source,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "implementation_commit": commit,
        "branch": BRANCH_NAME,
        "upstream_ref": f"origin/{BRANCH_NAME}",
        "upstream_commit": commit,
        "origin_url": EXPECTED_ORIGIN_URL,
        "clean_tracked_tree": True,
        "head_matches_upstream": True,
        "preregistration_is_ancestor": True,
        "reused_sources": reused,
        "new_sources": new,
        **dependency_material,
        "dependency_closure_sha256": dependency_closure,
        "source_tree_sha256": canonical_sha256(source_tree),
        "source_verification_sha256": canonical_sha256(
            verification_material
        ),
        "effects_permitted": False,
    }
    return validate_implementation_manifest(
        {
            **body,
            "implementation_manifest_sha256": canonical_sha256(body),
        }
    )


def _local_remote(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    bare.mkdir()
    work.mkdir()
    _git(bare, "init", "--bare")
    _git(bare, "config", "core.longpaths", "true")
    _git(work, "init")
    _git(work, "config", "user.name", "Registry Test")
    _git(work, "config", "user.email", "registry@example.invalid")
    (work / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    source_pin = (
        WORKSPACE_ROOT
        / "docs/protocol_evidence/"
        "sec_gemma_reveal_registry_initial_pin.json"
    )
    target_pin = (
        work
        / "docs/protocol_evidence/"
        "sec_gemma_reveal_registry_initial_pin.json"
    )
    target_pin.parent.mkdir(parents=True)
    target_pin.write_bytes(source_pin.read_bytes())
    _git(work, "add", ".")
    _git(work, "commit", "-m", "frozen implementation")
    _git(work, "branch", "-M", BRANCH_NAME)
    _git(work, "remote", "add", "origin", EXPECTED_ORIGIN_URL)
    _git(
        work,
        "config",
        f"url.{bare.as_uri()}.insteadOf",
        EXPECTED_ORIGIN_URL,
    )
    _git(work, "push", "--set-upstream", "origin", BRANCH_NAME)
    commit = _git(work, "rev-parse", "HEAD")
    return work, _manifest(commit)


class FakeTerminalAnchorStore:
    def __init__(self, binding: dict[str, Any]) -> None:
        self.binding = copy.deepcopy(binding)
        self.calls: list[str] = []

    def terminal_anchor_binding(
        self,
        attempt_id: str,
    ) -> dict[str, Any]:
        self.calls.append(attempt_id)
        return copy.deepcopy(self.binding)


def _confirmation_binding(
    *,
    work: Path,
    manifest: dict[str, Any],
) -> tuple[FakeTerminalAnchorStore, ExternalGitTagPublisher]:
    publisher = ExternalGitTagPublisher(
        repo_root=work,
        implementation_manifest=manifest,
        test_only_allow_url_rewrite=True,
    )
    receipt = publisher.publish(
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        terminal_status=TERMINAL_PASS,
        report_kind=SCORED_PASS,
        artifact_sha256=_digest("confirmation joint report"),
        predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
        deadline_monotonic=10**12,
    )
    binding = {
        "terminal_evidence": {
            "attempt_id": CONFIRMATION_ATTEMPT_ID,
            "terminal_status": TERMINAL_PASS,
            "external_publication_sha256": (
                receipt.publication_sha256
            ),
        },
        "external_publication": receipt.publication,
        "artifact_receipt": {
            "payload_sha256": _digest("confirmation artifact"),
        },
    }
    return FakeTerminalAnchorStore(binding), publisher


def test_frozen_pin_builds_exact_successor_fields() -> None:
    manifest = _manifest("1" * 40)
    access = build_contract_manifest()["stage_access"]["final"]
    raw_pin = json.loads(
        (
            WORKSPACE_ROOT / access["predecessor_registry_pin_file"]
        ).read_text(encoding="utf-8")
    )
    pin = validate_predecessor_registry_pin(raw_pin)
    successor = build_final_registry_successor(
        implementation_manifest=manifest,
        predecessor_pin=pin,
    )

    assert tuple(successor) == FINAL_REGISTRY_SUCCESSOR_FIELDS
    assert successor["schema_version"] == (
        FINAL_REGISTRY_SUCCESSOR_SCHEMA_VERSION
    )
    assert successor["predecessor_reveal_count"] == 10
    assert successor["successor_ordinal"] == 11
    assert successor["status"] == REGISTERED_UNRUN
    assert successor["final_attempt_id"] == FINAL_ATTEMPT_ID
    assert successor["successor_registry_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in successor.items()
            if key != "successor_registry_sha256"
        }
    )


def test_authorizer_publishes_before_issuing_opaque_authority(
    tmp_path: Path,
) -> None:
    work, manifest = _local_remote(tmp_path)
    store, publisher = _confirmation_binding(
        work=work,
        manifest=manifest,
    )
    authority = FinalRegistryAuthorizer(
        repo_root=work,
        publisher=publisher,
        store=store,
    ).authorize_final(
        implementation_manifest=manifest,
        deadline_monotonic=10**12,
    )

    assert is_verified_final_registry_authorization(authority)
    authorization = authority.authorization
    assert tuple(authorization) == FINAL_REGISTRY_AUTHORIZATION_FIELDS
    assert authorization["schema_version"] == (
        FINAL_REGISTRY_AUTHORIZATION_SCHEMA_VERSION
    )
    assert authorization["verifier_id"] == FINAL_REGISTRY_VERIFIER_ID
    assert authorization["final_attempt_id"] == FINAL_ATTEMPT_ID
    assert authorization["successor_registry_sha256"] == (
        authority.successor["successor_registry_sha256"]
    )
    assert authorization["external_publication_sha256"] == (
        authority.external_publication.publication_sha256
    )
    final_publication = authority.external_publication.publication
    assert final_publication["report_kind"] == FINAL_REGISTRY_SUCCESSOR
    assert final_publication["terminal_status"] == REGISTERED_UNRUN
    assert final_publication["artifact_sha256"] == (
        authority.successor_registry_sha256
    )
    assert store.calls == [CONFIRMATION_ATTEMPT_ID]


def test_authorizer_recovers_exact_successor_after_publish_crash_window(
    tmp_path: Path,
) -> None:
    work, manifest = _local_remote(tmp_path)
    store, publisher = _confirmation_binding(
        work=work,
        manifest=manifest,
    )
    access = build_contract_manifest()["stage_access"]["final"]
    predecessor_pin = validate_predecessor_registry_pin(
        json.loads(
            (
                work / access["predecessor_registry_pin_file"]
            ).read_text(encoding="utf-8")
        )
    )
    successor = build_final_registry_successor(
        implementation_manifest=manifest,
        predecessor_pin=predecessor_pin,
    )
    predecessor_publication_sha256 = store.binding[
        "external_publication"
    ]["publication_sha256"]
    published_before_crash = publisher.publish(
        attempt_id=FINAL_ATTEMPT_ID,
        terminal_status=REGISTERED_UNRUN,
        report_kind=FINAL_REGISTRY_SUCCESSOR,
        artifact_sha256=successor["successor_registry_sha256"],
        predecessor_publication_sha256=(
            predecessor_publication_sha256
        ),
        deadline_monotonic=10**12,
    )

    recovered = FinalRegistryAuthorizer(
        repo_root=work,
        publisher=publisher,
        store=store,
    ).authorize_final(
        implementation_manifest=manifest,
        deadline_monotonic=10**12,
    )

    assert is_verified_final_registry_authorization(recovered)
    assert recovered.external_publication.publication == (
        published_before_crash.publication
    )
    assert recovered.successor == successor


def test_tampered_confirmation_anchor_cannot_authorize_final(
    tmp_path: Path,
) -> None:
    work, manifest = _local_remote(tmp_path)
    store, publisher = _confirmation_binding(
        work=work,
        manifest=manifest,
    )
    store.binding["terminal_evidence"][
        "external_publication_sha256"
    ] = _digest("fabricated")
    authorizer = FinalRegistryAuthorizer(
        repo_root=work,
        publisher=publisher,
        store=store,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRegistryError,
        match="exact published pass",
    ):
        authorizer.authorize_final(
            implementation_manifest=manifest,
            deadline_monotonic=10**12,
        )


def test_tampered_pin_bytes_fail_before_successor_publication(
    tmp_path: Path,
) -> None:
    work, manifest = _local_remote(tmp_path)
    store, publisher = _confirmation_binding(
        work=work,
        manifest=manifest,
    )
    access = build_contract_manifest()["stage_access"]["final"]
    pin_path = work / access["predecessor_registry_pin_file"]
    pin_path.write_bytes(pin_path.read_bytes() + b"\n")
    authorizer = FinalRegistryAuthorizer(
        repo_root=work,
        publisher=publisher,
        store=store,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayRegistryError,
        match="file hash changed",
    ):
        authorizer.authorize_final(
            implementation_manifest=manifest,
            deadline_monotonic=10**12,
        )


def test_direct_final_authority_construction_is_forbidden() -> None:
    with pytest.raises(SecGemmaOnlineRiskOverlayRegistryError):
        VerifiedFinalRegistryAuthorization(
            authorization={},
            successor={},
            external_publication=None,  # type: ignore[arg-type]
            _sentinel=object(),
        )
