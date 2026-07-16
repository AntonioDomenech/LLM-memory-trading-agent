from __future__ import annotations

import copy
import hashlib
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
    EXTERNAL_PUBLICATION_FIELDS,
    EXTERNAL_TAG_MESSAGE_FIELDS,
    EXTERNAL_TAG_REF_TEMPLATE,
    SOURCE_PIN_FILES,
    SOURCE_PINS,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
    EXTERNAL_PUBLICATION_SCHEMA_VERSION,
    EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION,
    PUBLICATION_GENESIS_SHA256,
    SCORED_PASS,
    TERMINAL_PASS,
    ExternalGitTagPublisher,
    SecGemmaOnlineRiskOverlayPublisherError,
    VerifiedExternalPublication,
    build_external_tag_message,
    is_verified_external_publication,
    validate_external_publication,
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


def _local_remote(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    bare.mkdir()
    work.mkdir()
    _git(bare, "init", "--bare")
    _git(bare, "config", "core.longpaths", "true")
    _git(work, "init")
    _git(work, "config", "user.name", "Publisher Test")
    _git(work, "config", "user.email", "publisher@example.invalid")
    (work / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _git(work, "add", "tracked.txt")
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
    return work, bare, _manifest(commit)


def test_tag_message_uses_exact_v2_1_contract_fields() -> None:
    manifest = _manifest("1" * 40)
    artifact = _digest("joint")
    message = build_external_tag_message(
        implementation_manifest=manifest,
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        terminal_status=TERMINAL_PASS,
        report_kind=SCORED_PASS,
        artifact_sha256=artifact,
        predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
    )

    assert tuple(message) == EXTERNAL_TAG_MESSAGE_FIELDS
    assert (
        message["schema_version"]
        == EXTERNAL_TAG_MESSAGE_SCHEMA_VERSION
    )
    assert "v2-1" in message["schema_version"]
    assert message["external_cost_usd"] == 0.0


def test_local_bare_remote_publication_returns_opaque_receipt(
    tmp_path: Path,
) -> None:
    work, _, manifest = _local_remote(tmp_path)
    artifact = _digest("joint")
    publisher = ExternalGitTagPublisher(
        repo_root=work,
        implementation_manifest=manifest,
        test_only_allow_url_rewrite=True,
    )

    receipt = publisher.publish(
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        terminal_status=TERMINAL_PASS,
        report_kind=SCORED_PASS,
        artifact_sha256=artifact,
        predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
        deadline_monotonic=10**12,
    )

    assert is_verified_external_publication(receipt)
    publication = receipt.publication
    assert tuple(publication) == EXTERNAL_PUBLICATION_FIELDS
    assert publication["schema_version"] == (
        EXTERNAL_PUBLICATION_SCHEMA_VERSION
    )
    assert publication["artifact_sha256"] == artifact
    assert publication["remote_url"] == EXPECTED_ORIGIN_URL
    assert publication["remote_peeled_commit"] == (
        manifest["implementation_commit"]
    )
    assert validate_external_publication(
        publication,
        implementation_manifest=manifest,
    ) == publication
    expected_ref = EXTERNAL_TAG_REF_TEMPLATE.format(
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        report_kind=SCORED_PASS,
        artifact_sha256=artifact,
    )
    remote = _git(
        work,
        "ls-remote",
        "--tags",
        "origin",
        expected_ref,
        f"{expected_ref}^{{}}",
    )
    assert publication["remote_tag_object_sha1"] in remote
    assert manifest["implementation_commit"] in remote


def test_tag_reuse_is_rejected_without_force(tmp_path: Path) -> None:
    work, _, manifest = _local_remote(tmp_path)
    publisher = ExternalGitTagPublisher(
        repo_root=work,
        implementation_manifest=manifest,
        test_only_allow_url_rewrite=True,
    )
    arguments = {
        "attempt_id": CONFIRMATION_ATTEMPT_ID,
        "terminal_status": TERMINAL_PASS,
        "report_kind": SCORED_PASS,
        "artifact_sha256": _digest("joint"),
        "predecessor_publication_sha256": (
            PUBLICATION_GENESIS_SHA256
        ),
        "deadline_monotonic": 10**12,
    }
    publisher.publish(**arguments)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="cannot be reused",
    ):
        publisher.publish(**arguments)


def test_tampered_publication_and_direct_opaque_construction_fail(
    tmp_path: Path,
) -> None:
    work, _, manifest = _local_remote(tmp_path)
    receipt = ExternalGitTagPublisher(
        repo_root=work,
        implementation_manifest=manifest,
        test_only_allow_url_rewrite=True,
    ).publish(
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        terminal_status=TERMINAL_PASS,
        report_kind=SCORED_PASS,
        artifact_sha256=_digest("joint"),
        predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
        deadline_monotonic=10**12,
    )
    tampered = copy.deepcopy(receipt.publication)
    tampered["artifact_sha256"] = _digest("tampered")

    with pytest.raises(SecGemmaOnlineRiskOverlayPublisherError):
        validate_external_publication(
            tampered,
            implementation_manifest=manifest,
        )
    with pytest.raises(SecGemmaOnlineRiskOverlayPublisherError):
        VerifiedExternalPublication(
            publication=receipt.publication,
            _sentinel=object(),
        )
    assert not is_verified_external_publication(receipt.publication)


def test_dirty_repository_and_expired_deadline_fail_before_push(
    tmp_path: Path,
) -> None:
    work, _, manifest = _local_remote(tmp_path)
    publisher = ExternalGitTagPublisher(
        repo_root=work,
        implementation_manifest=manifest,
        test_only_allow_url_rewrite=True,
    )
    (work / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    arguments = {
        "attempt_id": CONFIRMATION_ATTEMPT_ID,
        "terminal_status": TERMINAL_PASS,
        "report_kind": SCORED_PASS,
        "artifact_sha256": _digest("joint"),
        "predecessor_publication_sha256": (
            PUBLICATION_GENESIS_SHA256
        ),
    }

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="not clean",
    ):
        publisher.publish(
            **arguments,
            deadline_monotonic=10**12,
        )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="expired",
    ):
        publisher.publish(
            **arguments,
            deadline_monotonic=-1.0,
        )


def test_production_publisher_rejects_git_url_rewrites(
    tmp_path: Path,
) -> None:
    work, _, manifest = _local_remote(tmp_path)
    publisher = ExternalGitTagPublisher(
        repo_root=work,
        implementation_manifest=manifest,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayPublisherError,
        match="redirected or rewritten",
    ):
        publisher.publish(
            attempt_id=CONFIRMATION_ATTEMPT_ID,
            terminal_status=TERMINAL_PASS,
            report_kind=SCORED_PASS,
            artifact_sha256=_digest("joint"),
            predecessor_publication_sha256=(
                PUBLICATION_GENESIS_SHA256
            ),
            deadline_monotonic=10**12,
        )


def test_publication_never_executes_repository_pre_push_hook(
    tmp_path: Path,
) -> None:
    work, _, manifest = _local_remote(tmp_path)
    sentinel = tmp_path / "pre-push-hook-ran"
    hook = work / ".git" / "hooks" / "pre-push"
    hook.write_text(
        "#!/bin/sh\n"
        f"printf ran > '{sentinel.as_posix()}'\n"
        "exit 91\n",
        encoding="utf-8",
        newline="\n",
    )
    hook.chmod(0o755)
    publisher = ExternalGitTagPublisher(
        repo_root=work,
        implementation_manifest=manifest,
        test_only_allow_url_rewrite=True,
    )

    receipt = publisher.publish(
        attempt_id=CONFIRMATION_ATTEMPT_ID,
        terminal_status=TERMINAL_PASS,
        report_kind=SCORED_PASS,
        artifact_sha256=_digest("hook-proof"),
        predecessor_publication_sha256=PUBLICATION_GENESIS_SHA256,
        deadline_monotonic=10**12,
    )

    assert is_verified_external_publication(receipt)
    assert not sentinel.exists()
