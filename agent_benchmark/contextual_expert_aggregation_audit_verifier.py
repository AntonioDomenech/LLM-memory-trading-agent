"""Independent semantic verifier for the one-shot v2 continuation audit."""

from __future__ import annotations

import json
import math
import platform
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import contextual_expert_aggregation_experiment as _experiment
from .contextual_expert_aggregation_audit_artifacts import (
    ATTEMPT_LOCK_FILENAME,
    AUDIT_PAYLOAD_NAMES,
    AUDIT_RUN_ID,
    AUDIT_STAGE,
    CONTRACT_VERSION,
    OUTPUT_DIRECTORY,
    OUTPUT_PARENT,
)
from . import contextual_expert_aggregation_audit_runner as _runner


VERIFIER_ID = "contextual-expert-aggregation-post-rejection-audit-verifier-v2"
VERIFIER_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_audit_verifier.py"
)


class ContextualExpertAggregationAuditVerificationError(RuntimeError):
    """Raised when sealed audit bytes do not independently regenerate."""


def _json_payload(
    bundle: _experiment.VerifiedBundle, filename: str
) -> dict[str, Any]:
    try:
        payload = (bundle.directory / filename).read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContextualExpertAggregationAuditVerificationError(
            f"audit payload is unreadable: {filename}"
        ) from exc
    if (
        not isinstance(value, dict)
        or _experiment.pretty_json_bytes(value) != payload
    ):
        raise ContextualExpertAggregationAuditVerificationError(
            f"audit payload is not canonical object JSON: {filename}"
        )
    return value


def _bundle_location(
    directory: Path, *, repo_root: Path
) -> tuple[Path, tuple[str, ...]]:
    try:
        root = _experiment._harden_path(
            _experiment._lexical_absolute(repo_root),
            label="Audit verifier repository root",
            require_exists=True,
        )
        bundle = _experiment._lexical_absolute(directory)
        final = _experiment._lexical_absolute(root / OUTPUT_DIRECTORY)
        private = _experiment._lexical_absolute(
            root / OUTPUT_PARENT / ".audit-v2.sealing" / AUDIT_RUN_ID
        )
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise ContextualExpertAggregationAuditVerificationError(
            "audit verifier path hardening failed"
        ) from exc
    if bundle not in {final, private}:
        raise ContextualExpertAggregationAuditVerificationError(
            "audit verifier received an unauthorized bundle location"
        )
    try:
        hardened = _experiment._harden_path(
            bundle, label="Audit verifier bundle", require_exists=True
        )
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise ContextualExpertAggregationAuditVerificationError(
            "audit verifier bundle is redirected or missing"
        ) from exc
    if bundle == final:
        return hardened, (AUDIT_RUN_ID,)
    if bundle == private:
        return hardened, (".audit-v2.sealing",)
    raise AssertionError("unreachable frozen audit bundle location")


def _runtime_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "runtime_evidence_schema_version",
        "stage",
        "seconds_before_seal",
        "strict_limit_seconds",
        "within_limit_before_seal",
        "monotonic_clock",
        "strict_limit_rechecked_immediately_before_promotion",
        "network_access",
        "news_access",
        "llm_calls",
        "api_calls",
        "external_model_calls",
        "external_cost_usd",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContextualExpertAggregationAuditVerificationError(
            "audit runtime evidence has the wrong exact inventory"
        )
    if (
        type(value["runtime_evidence_schema_version"]) is not int
        or type(value["seconds_before_seal"]) is not float
        or type(value["strict_limit_seconds"]) is not float
        or type(value["llm_calls"]) is not int
        or type(value["api_calls"]) is not int
        or type(value["external_model_calls"]) is not int
        or type(value["external_cost_usd"]) is not float
    ):
        raise ContextualExpertAggregationAuditVerificationError(
            "audit runtime evidence contains a noncanonical numeric type"
        )
    elapsed = value["seconds_before_seal"]
    limit = value["strict_limit_seconds"]
    cost = value["external_cost_usd"]
    if (
        not math.isfinite(elapsed)
        or not 0.0 <= elapsed < _runner.RUN_TIME_LIMIT_SECONDS
        or limit != _runner.RUN_TIME_LIMIT_SECONDS
        or cost != 0.0
        or value["runtime_evidence_schema_version"] != 1
        or value["stage"] != AUDIT_STAGE
        or value["within_limit_before_seal"] is not True
        or value["monotonic_clock"] is not True
        or value["strict_limit_rechecked_immediately_before_promotion"] is not True
        or value["network_access"] is not False
        or value["news_access"] is not False
        or value["llm_calls"] != 0
        or value["api_calls"] != 0
        or value["external_model_calls"] != 0
    ):
        raise ContextualExpertAggregationAuditVerificationError(
            "audit runtime/cost evidence violates the frozen contract"
        )
    return dict(value)


def _validated_prelock_git_identity(
    repo_root: Path, value: Any
) -> dict[str, Any]:
    """Independently reconstruct every pre-lock manifest field.

    The post-lock verifier must not accept observed claims merely because the
    lock and manifest repeat them.  Reconstructing the exact object from live
    Git metadata and the frozen dependency inventory rejects added fields and
    altered historical assertions alike.
    """

    root = repo_root.resolve()
    expected_keys = {
        "branch",
        "commit",
        "upstream",
        "upstream_commit",
        "origin_url",
        "origin_repository",
        "head_equals_upstream",
        "dirty",
        "cleanliness_scope",
        "prelock_input_worktree_bytes_opened",
        "input_path",
        "input_head_object_id",
        "input_index_object_id",
        "input_head_index_equal",
        "tracked_dependency_identity",
        "runtime_versions",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ContextualExpertAggregationAuditVerificationError(
            "audit manifest pre-lock Git identity has the wrong exact inventory"
        )
    input_relative = _runner.AUDIT_INPUT_PATH.as_posix()
    branch = _runner._git_text(
        root, "symbolic-ref", "--quiet", "--short", "HEAD"
    )
    commit = _runner._git_text(root, "rev-parse", "HEAD")
    upstream = _runner._git_text(
        root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    upstream_commit = _runner._git_text(root, "rev-parse", "@{upstream}")
    origin_url = _runner._git_text(root, "remote", "get-url", "origin")
    head_blob = _runner._git_text(root, "rev-parse", f"HEAD:{input_relative}")
    index_blob = _runner._tracked_index_object_id(root, input_relative)
    if (
        branch != _runner.EXPECTED_BRANCH
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
        or upstream != f"origin/{branch}"
        or upstream_commit != commit
        or origin_url not in _experiment.EXPECTED_ORIGIN_URLS
        or head_blob != _runner.AUDIT_INPUT_SPEC.git_blob
        or index_blob != _runner.AUDIT_INPUT_SPEC.git_blob
    ):
        raise ContextualExpertAggregationAuditVerificationError(
            "current Git identity is not the frozen pushed audit identity"
        )
    expected = {
        "branch": branch,
        "commit": commit,
        "upstream": upstream,
        "upstream_commit": upstream_commit,
        "origin_url": origin_url,
        "origin_repository": _experiment.EXPECTED_ORIGIN_REPOSITORY,
        "head_equals_upstream": True,
        "dirty": False,
        "cleanliness_scope": "all_paths_except_unopened_input",
        "prelock_input_worktree_bytes_opened": False,
        "input_path": input_relative,
        "input_head_object_id": head_blob,
        "input_index_object_id": index_blob,
        "input_head_index_equal": True,
        "tracked_dependency_identity": _runner._dependency_identity(root),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    if dict(value) != expected:
        raise ContextualExpertAggregationAuditVerificationError(
            "audit manifest pre-lock Git identity differs from reconstruction"
        )
    return expected


def _expected_manifest(
    *,
    observed: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    prelock_git_identity: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    report: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    lock_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "stage": AUDIT_STAGE,
        "stage_pass": bool(gate_report["passed"]),
        "run_id": AUDIT_RUN_ID,
        "evidence_classification": _runner.EVIDENCE_CLASSIFICATION,
        "git_identity": dict(prelock_git_identity),
        "parent_manifest_sha256": _runner.PARENT_MANIFEST_SELF_SHA256,
        "parent_semantic_evidence_sha256": (
            _runner.PARENT_SEMANTIC_EVIDENCE_SHA256
        ),
        "input_raw_sha256": _runner.AUDIT_INPUT_SPEC.raw_sha256,
        "input_canonical_sha256": _runner.AUDIT_INPUT_SPEC.canonical_sha256,
        "attempt_lock_sha256": lock_sha256,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "report_status": report["status"],
        "historical_policy_candidate_for_2024_audit": report[
            "historical_policy_candidate_for_2024_audit"
        ],
        "learning_candidate_for_2024_audit": report[
            "learning_candidate_for_2024_audit"
        ],
        "parent_rejection_remains_final": True,
        "post_2023_market_values_accessed": False,
        "runtime_cost_evidence": dict(runtime),
    }
    expected = _experiment.self_hashed_manifest(
        {
            "contract_version": CONTRACT_VERSION,
            **expected_fields,
            "payload_sha256": {
                name: _experiment.sha256_bytes(payloads[name])
                for name in sorted(payloads)
            },
        }
    )
    if dict(observed) != expected:
        raise ContextualExpertAggregationAuditVerificationError(
            "audit manifest differs from independent deterministic regeneration"
        )
    return expected


def verify_private_audit_bundle(
    directory: Path,
    *,
    repo_root: Path,
    expected_manifest_sha256: str,
    deadline: _experiment.StageDeadline,
) -> dict[str, Any]:
    """Regenerate every semantic payload in a private or promoted bundle."""

    root = _experiment._lexical_absolute(Path(repo_root))
    bundle_path, allowed_output_directories = _bundle_location(
        Path(directory), repo_root=root
    )
    deadline.check("audit verifier entry")
    try:
        bundle = _experiment._verify_bundle_identity(
            bundle_path,
            expected_contract_version=CONTRACT_VERSION,
            expected_stage=AUDIT_STAGE,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_payload_names=AUDIT_PAYLOAD_NAMES,
        )
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise ContextualExpertAggregationAuditVerificationError(
            "exact audit byte-bundle verification failed"
        ) from exc
    manifest = dict(bundle.manifest)
    prelock = _validated_prelock_git_identity(
        root, manifest.get("git_identity")
    )

    development = _runner.prepare_rejected_development(root, deadline=deadline)
    deadline.check("after independently preparing rejected parent")
    sealed_parent = _json_payload(bundle, "rejected_parent_verification.json")
    if sealed_parent != dict(development.parent_verification):
        raise ContextualExpertAggregationAuditVerificationError(
            "sealed rejected-parent proof differs from regeneration"
        )
    lock = _runner.read_attempt_lock(root)
    try:
        sealed_lock = (bundle.directory / ATTEMPT_LOCK_FILENAME).read_bytes()
    except OSError as exc:
        raise ContextualExpertAggregationAuditVerificationError(
            "sealed attempt lock is unreadable"
        ) from exc
    if sealed_lock != lock.bytes_value:
        raise ContextualExpertAggregationAuditVerificationError(
            "sealed attempt lock differs from the persistent lock"
        )
    if (
        lock.content.get("git_commit") != prelock.get("commit")
        or lock.content.get("prelock_git_identity_sha256")
        != _runner._sha256_json(prelock)
        or lock.content.get("parent_verification_sha256")
        != _runner._sha256_json(development.parent_verification)
    ):
        raise ContextualExpertAggregationAuditVerificationError(
            "attempt lock does not bind the sealed Git and parent evidence"
        )
    snapshot, input_provenance = _runner.load_locked_audit_snapshot(
        root,
        development=development,
        prelock_git_identity=prelock,
        lock=lock,
        deadline=deadline,
        allowed_output_directories=allowed_output_directories,
    )
    computation = _runner.compute_audit_continuation(
        root,
        development=development,
        snapshot=snapshot,
        input_provenance=input_provenance,
        deadline=deadline,
    )
    integrity, gate_report, metrics = _runner.evaluate_audit(computation)
    runtime = _runtime_payload(
        _json_payload(bundle, "audit_runtime_cost_evidence.json")
    )
    expected_payloads, checkpoint, report = _runner.build_audit_payloads(
        computation=computation,
        development=development,
        parent_verification=development.parent_verification,
        input_provenance=input_provenance,
        lock=lock,
        integrity=integrity,
        gate_report=gate_report,
        metrics=metrics,
        runtime=runtime,
    )
    if set(expected_payloads) != set(AUDIT_PAYLOAD_NAMES):
        raise ContextualExpertAggregationAuditVerificationError(
            "independent audit regeneration produced the wrong inventory"
        )
    for filename, expected in expected_payloads.items():
        try:
            observed = (bundle.directory / filename).read_bytes()
        except OSError as exc:
            raise ContextualExpertAggregationAuditVerificationError(
                f"sealed audit payload became unreadable: {filename}"
            ) from exc
        if observed != expected:
            raise ContextualExpertAggregationAuditVerificationError(
                f"sealed audit payload differs from regeneration: {filename}"
            )
    expected_manifest = _expected_manifest(
        observed=manifest,
        payloads=expected_payloads,
        prelock_git_identity=prelock,
        gate_report=gate_report,
        report=report,
        checkpoint=checkpoint,
        lock_sha256=lock.sha256,
        runtime=runtime,
    )
    deadline.check("after independent audit semantic regeneration")
    evidence = {
        "verifier_id": VERIFIER_ID,
        "verified": True,
        "contract_version": CONTRACT_VERSION,
        "stage": AUDIT_STAGE,
        "run_id": AUDIT_RUN_ID,
        "stage_pass": bool(gate_report["passed"]),
        "status": report["status"],
        "manifest_sha256": expected_manifest["manifest_sha256"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "attempt_lock_sha256": lock.sha256,
        "parent_semantic_evidence_sha256": (
            _runner.PARENT_SEMANTIC_EVIDENCE_SHA256
        ),
        "parent_rejection_remains_final": True,
        "post_2023_market_values_accessed": False,
        "exact_payload_count": len(expected_payloads),
        "semantic_evidence_sha256": "",
    }
    evidence["semantic_evidence_sha256"] = _runner._sha256_json(
        {name: value for name, value in evidence.items() if name != "semantic_evidence_sha256"}
    )
    return evidence


def verify_audit(
    repo_root: Path | None = None,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else _experiment._lexical_absolute(Path(repo_root))
    )
    bundle, _ = _bundle_location(root / OUTPUT_DIRECTORY, repo_root=root)
    if expected_manifest_sha256 is None:
        try:
            manifest = json.loads(
                (bundle / "stage_manifest.json").read_text(encoding="utf-8")
            )
            expected_manifest_sha256 = manifest["manifest_sha256"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise ContextualExpertAggregationAuditVerificationError(
                "promoted audit manifest is unreadable"
            ) from exc
    deadline = _experiment.StageDeadline(
        limit_seconds=_runner.RUN_TIME_LIMIT_SECONDS
    )
    return verify_private_audit_bundle(
        bundle,
        repo_root=root,
        expected_manifest_sha256=expected_manifest_sha256,
        deadline=deadline,
    )


def verify_stage(stage: str) -> dict[str, Any]:
    if stage != AUDIT_STAGE:
        raise ContextualExpertAggregationAuditVerificationError(
            "the audit verifier supports only the frozen audit stage"
        )
    from .contextual_expert_aggregation_audit_bootstrap import (
        require_active_attestation,
    )

    root = Path(__file__).resolve().parents[1]
    require_active_attestation(operation="verify", stage=stage, repo_root=root)
    return verify_audit(root)


__all__ = [
    "ContextualExpertAggregationAuditVerificationError",
    "VERIFIER_ID",
    "VERIFIER_IMPLEMENTATION_PATH",
    "verify_audit",
    "verify_private_audit_bundle",
    "verify_stage",
]
