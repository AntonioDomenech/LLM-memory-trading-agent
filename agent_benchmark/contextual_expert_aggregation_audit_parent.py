"""Exact rejected-parent verifier for the contextual aggregation v2 audit.

The ordinary v2 confirmation path intentionally authorizes only a passing
development bundle.  The post-rejection audit instead needs to accept one and
only one immutable rejected bundle.  This module therefore binds the exact
sealed bytes, the historical implementation and preservation commits, the
35-of-36 rejection, and a fresh independent through-2018 regeneration.

The v2 semantic verifier's live Git check requires execution on the old v2
branch.  That condition cannot be true on the preregistered audit branch.  The
wrapper below reuses the v2 computation, gate, payload, report, manifest, and
semantic-hash logic, while replacing only that live old-branch equality with
two stronger audit-appropriate checks:

* every dependency recorded by the parent is reproduced from the exact
  historical implementation commit; and
* every executable v2 dependency used for regeneration still has the same
  HEAD, index, and worktree content.

No through-2023 path is named or read here.  The only market data this module
can load is the already sealed through-2018 development input used by the v2
regenerator.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import contextual_expert_aggregation_artifacts as _artifacts
from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_stage as _stage_runner
from . import contextual_expert_aggregation_verifier as _v2_verifier


VERIFIER_ID = "contextual-expert-aggregation-rejected-parent-verifier-v2"
VERIFIER_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_audit_parent.py"
)

PARENT_DIRECTORY = _artifacts.OUTPUT_DIRECTORY_BY_STAGE[
    _artifacts.DEVELOPMENT_STAGE
]
PARENT_CONTRACT_VERSION = "aapl-causal-contextual-expert-aggregation-v2"
PARENT_STAGE = "development"
PARENT_RUN_ID = "contextual-expert-aggregation-development-v2"
PARENT_STATUS = "REJECTED"
PARENT_IMPLEMENTATION_COMMIT = "25609a514a9f36bbd09597a1b98ddd2fc753f1cd"
PARENT_REJECTION_PRESERVATION_COMMIT = (
    "7bd9e6e992e47cd34cb4e4312661a767cd15fce5"
)
PARENT_MANIFEST_SELF_SHA256 = (
    "sha256:6cb657d1d9c323bd96a84b598b9b993e2444f756a65521aa1bf28207f7a63539"
)
PARENT_MANIFEST_FILE_SHA256 = (
    "sha256:fbc55137d612699f794af4eedaf1e4600b4e0e916f954b962ced34f030752499"
)
PARENT_CHECKSUMS_FILE_SHA256 = (
    "sha256:e621c845ef17d2f9e4e87efaae083e036200043fbf5ba1619800bde8df04c6a6"
)
PARENT_CHECKPOINT_SELF_SHA256 = (
    "sha256:e195356c0540a4a05ea635927fa466c1c85948beb7de0191e80f5bc10d60fd03"
)
PARENT_CHECKPOINT_PAYLOAD_SHA256 = (
    "sha256:abe07112ce2bb72292d65b357dc006ab3be268f8b0d02077563c59fa71a22538"
)
PARENT_REPORT_PAYLOAD_SHA256 = (
    "sha256:d08e3b650738dad248ce633c4e8077def9c89a614203dfcfc4110b787287e815"
)
PARENT_GATE_REPORT_PAYLOAD_SHA256 = (
    "sha256:b28b06d28f749efe10b2f853f55386bab957dc4a68480cbaa7ce815ffa6f4e39"
)
PARENT_SEMANTIC_EVIDENCE_SHA256 = (
    "sha256:964ad44206d3a1cb481a330317217a2d910101b34e35240aa1186c8a6069e944"
)
PARENT_CAUSAL_PROOF_SELF_SHA256 = (
    "sha256:dc7715dc5ad21f6f86cba0aee6488c9a24859dc19aa761115d48e2a2f199b350"
)
PARENT_PASSED_CHECK_COUNT = 35
PARENT_TOTAL_CHECK_COUNT = 36
PARENT_FAILED_CHECKS = (
    "stress_10bps.full_account_beats_best_fixed_by_gt_0_0001",
)

_CHECKPOINT_FILENAME = _experiment.DEVELOPMENT_CHECKPOINT_FILENAME
_GATE_REPORT_FILENAME = _experiment.DEVELOPMENT_GATE_REPORT_FILENAME
_REPORT_FILENAME = _experiment.DEVELOPMENT_REPORT_FILENAME
_CAUSAL_PROOF_FILENAME = "development_parent_causal_prefix_proof.json"


class RejectedParentVerificationError(RuntimeError):
    """The exact rejected development parent could not be reproduced."""


@dataclass(frozen=True)
class _RegeneratedSemantics:
    stage_pass: bool
    status: str
    checkpoint_self_sha256: str
    gate_report_file_sha256: str
    report_file_sha256: str
    semantic_evidence_sha256: str


@dataclass(frozen=True)
class RejectedParentEvidence:
    """Typed, byte-bound evidence suitable for the audit attempt lock."""

    verifier_id: str
    verifier_dependency_path: str
    verified: bool
    parent_rejection_remains_final: bool
    contract_version: str
    stage: str
    run_id: str
    status: str
    stage_pass: bool
    implementation_commit: str
    rejection_preservation_commit: str
    manifest_path: str
    manifest_self_sha256: str
    manifest_file_sha256: str
    checksums_file_sha256: str
    checkpoint_self_sha256: str
    checkpoint_payload_sha256: str
    report_payload_sha256: str
    gate_report_payload_sha256: str
    semantic_evidence_sha256: str
    parent_causal_proof_self_sha256: str
    passed_check_count: int
    total_check_count: int
    failed_checks: tuple[str, ...]
    exact_payload_names: tuple[str, ...]
    checksummed_names: tuple[str, ...]
    historical_dependency_paths: tuple[str, ...]
    live_semantic_dependency_paths: tuple[str, ...]
    old_branch_equality_replaced_by_audit_continuity: bool
    bundle: _experiment.VerifiedBundle
    manifest_bytes: bytes
    checksums_bytes: bytes

    def lock_payload(self) -> dict[str, Any]:
        """Return the stable JSON-compatible subset for an attempt lock."""

        return {
            "verifier_id": self.verifier_id,
            "verifier_dependency_path": self.verifier_dependency_path,
            "verified": self.verified,
            "parent_rejection_remains_final": (
                self.parent_rejection_remains_final
            ),
            "contract_version": self.contract_version,
            "stage": self.stage,
            "run_id": self.run_id,
            "status": self.status,
            "stage_pass": self.stage_pass,
            "implementation_commit": self.implementation_commit,
            "rejection_preservation_commit": (
                self.rejection_preservation_commit
            ),
            "manifest_path": self.manifest_path,
            "manifest_self_sha256": self.manifest_self_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "checksums_file_sha256": self.checksums_file_sha256,
            "checkpoint_self_sha256": self.checkpoint_self_sha256,
            "checkpoint_payload_sha256": self.checkpoint_payload_sha256,
            "report_payload_sha256": self.report_payload_sha256,
            "gate_report_payload_sha256": self.gate_report_payload_sha256,
            "semantic_evidence_sha256": self.semantic_evidence_sha256,
            "parent_causal_proof_self_sha256": (
                self.parent_causal_proof_self_sha256
            ),
            "passed_check_count": self.passed_check_count,
            "total_check_count": self.total_check_count,
            "failed_checks": list(self.failed_checks),
            "exact_payload_names": list(self.exact_payload_names),
            "checksummed_names": list(self.checksummed_names),
            "historical_dependency_paths": list(
                self.historical_dependency_paths
            ),
            "live_semantic_dependency_paths": list(
                self.live_semantic_dependency_paths
            ),
            "old_branch_equality_replaced_by_audit_continuity": (
                self.old_branch_equality_replaced_by_audit_continuity
            ),
        }


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=30.0,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RejectedParentVerificationError(
            "Rejected-parent Git continuity verification failed"
        ) from exc


def _git_text(repo_root: Path, *args: str) -> str:
    try:
        return _git_bytes(repo_root, *args).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RejectedParentVerificationError(
            "Rejected-parent Git metadata is not UTF-8"
        ) from exc


def _require_repository_root(repo_root: Path) -> Path:
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise RejectedParentVerificationError(
            "Rejected-parent verifier requires an existing repository root"
        )
    actual = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if actual != root:
        raise RejectedParentVerificationError(
            "repo_root must be the actual Git repository root"
        )
    return root


def _require_commit(repo_root: Path, commit: str) -> None:
    observed = _git_text(repo_root, "rev-parse", f"{commit}^{{commit}}")
    if observed != commit:
        raise RejectedParentVerificationError(
            "A frozen rejected-parent commit identity changed"
        )


def _require_ancestor(repo_root: Path, older: str, newer: str) -> None:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=repo_root,
            capture_output=True,
        )
    except OSError as exc:
        raise RejectedParentVerificationError(
            "Rejected-parent ancestry verification failed"
        ) from exc
    if result.returncode != 0:
        raise RejectedParentVerificationError(
            "Frozen implementation or rejection commit is not an ancestor"
        )


def _require_parent_git_and_dependency_continuity(
    repo_root: Path,
    bundle: _experiment.VerifiedBundle,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    """Bind historical dependencies and the live executable v2 subset.

    Documentation can legitimately change on the audit branch because the
    audit itself is preregistered there.  Every parent dependency, including
    documentation, is still reproduced byte-for-byte from the implementation
    commit.  In addition, every Python/runtime dependency used for the fresh
    semantic regeneration must remain byte-identical at HEAD, in the index,
    and in the worktree.
    """

    identity_value = bundle.manifest.get("git_identity")
    if not isinstance(identity_value, Mapping):
        raise RejectedParentVerificationError(
            "Rejected parent omits its frozen Git identity"
        )
    identity = dict(identity_value)
    if (
        identity.get("branch") != _experiment.EXPECTED_BRANCH
        or identity.get("commit") != PARENT_IMPLEMENTATION_COMMIT
        or identity.get("upstream")
        != f"origin/{_experiment.EXPECTED_BRANCH}"
        or identity.get("upstream_remote") != "origin"
        or identity.get("upstream_branch") != _experiment.EXPECTED_BRANCH
        or identity.get("upstream_commit") != PARENT_IMPLEMENTATION_COMMIT
        or identity.get("origin_url") not in _experiment.EXPECTED_ORIGIN_URLS
        or identity.get("origin_repository")
        != _experiment.EXPECTED_ORIGIN_REPOSITORY
        or identity.get("head_equals_upstream") is not True
        or identity.get("dirty") is not False
        or identity.get("cleanliness_scope")
        != "git_visible_index_and_worktree_after_attempt_lock"
    ):
        raise RejectedParentVerificationError(
            "Rejected parent has the wrong historical Git identity"
        )

    dependencies_value = identity.get("tracked_dependency_identity")
    if not isinstance(dependencies_value, Mapping):
        raise RejectedParentVerificationError(
            "Rejected parent omits its frozen dependencies"
        )
    dependencies = dict(dependencies_value)
    expected_paths = {
        path.as_posix() for path in _experiment._frozen_dependency_paths()
    }
    if set(dependencies) != expected_paths:
        raise RejectedParentVerificationError(
            "Rejected parent has the wrong frozen dependency inventory"
        )

    _require_commit(repo_root, PARENT_IMPLEMENTATION_COMMIT)
    _require_commit(repo_root, PARENT_REJECTION_PRESERVATION_COMMIT)
    head = _git_text(repo_root, "rev-parse", "HEAD")
    _require_ancestor(
        repo_root,
        PARENT_IMPLEMENTATION_COMMIT,
        PARENT_REJECTION_PRESERVATION_COMMIT,
    )
    _require_ancestor(repo_root, PARENT_REJECTION_PRESERVATION_COMMIT, head)

    historical_paths: list[str] = []
    live_paths: list[str] = []
    for relative in sorted(expected_paths):
        recorded = dependencies.get(relative)
        if not isinstance(recorded, Mapping) or set(recorded) != {
            "sha256",
            "git_blob",
        }:
            raise RejectedParentVerificationError(
                "Rejected parent has malformed dependency evidence"
            )
        historical_blob = _git_text(
            repo_root, "rev-parse", f"{PARENT_IMPLEMENTATION_COMMIT}:{relative}"
        )
        historical_bytes = _git_bytes(
            repo_root, "show", f"{PARENT_IMPLEMENTATION_COMMIT}:{relative}"
        )
        if (
            historical_blob != recorded.get("git_blob")
            or _experiment.sha256_bytes(historical_bytes)
            != recorded.get("sha256")
        ):
            raise RejectedParentVerificationError(
                "Historical v2 dependency differs from the parent manifest"
            )
        historical_paths.append(relative)

        is_live_semantic_dependency = (
            relative.startswith("agent_benchmark/")
            or relative in {".gitattributes", ".gitignore", "requirements.txt"}
        )
        if is_live_semantic_dependency:
            try:
                _experiment.tracked_file_identity(
                    repo_root,
                    repo_root / relative,
                    expected_sha256=str(recorded["sha256"]),
                    expected_git_blob=str(recorded["git_blob"]),
                    allow_crlf_equivalent=True,
                )
            except _experiment.ContextualExpertAggregationExperimentError as exc:
                raise RejectedParentVerificationError(
                    "Live v2 semantic dependency differs from the rejected parent"
                ) from exc
            live_paths.append(relative)

    if identity.get("runtime_versions") != _experiment._runtime_versions():
        raise RejectedParentVerificationError(
            "Runtime versions differ from the rejected parent"
        )

    try:
        unchanged = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                PARENT_REJECTION_PRESERVATION_COMMIT,
                "HEAD",
                "--",
                PARENT_DIRECTORY.as_posix(),
            ],
            cwd=repo_root,
            capture_output=True,
        )
    except OSError as exc:
        raise RejectedParentVerificationError(
            "Rejected-parent preservation comparison failed"
        ) from exc
    if unchanged.returncode != 0:
        raise RejectedParentVerificationError(
            "Rejected bundle changed after its preservation commit"
        )
    return identity, tuple(historical_paths), tuple(live_paths)


def _json_payload(
    bundle: _experiment.VerifiedBundle, filename: str
) -> dict[str, Any]:
    try:
        payload = _v2_verifier._payload(bundle, filename)
        if filename == _CHECKPOINT_FILENAME:
            value = json.loads(
                payload.decode("utf-8"),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant {token}")
                ),
            )
            if _experiment.canonical_json_bytes(value) != payload:
                raise ValueError("checkpoint is not canonical compact JSON")
        else:
            value = _stage_runner._strict_json(payload, field=filename)
        if not isinstance(value, dict):
            raise ValueError("payload is not a JSON object")
        return value
    except Exception as exc:
        raise RejectedParentVerificationError(
            f"Rejected-parent payload is invalid: {filename}"
        ) from exc


def _require_exact_static_rejection(
    bundle: _experiment.VerifiedBundle,
) -> None:
    manifest = bundle.manifest
    if (
        manifest.get("contract_version") != PARENT_CONTRACT_VERSION
        or manifest.get("stage") != PARENT_STAGE
        or manifest.get("run_id") != PARENT_RUN_ID
        or manifest.get("stage_pass") is not False
        or manifest.get("manifest_sha256") != PARENT_MANIFEST_SELF_SHA256
        or manifest.get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SELF_SHA256
        or manifest.get("payload_sha256") != dict(bundle.payload_sha256)
    ):
        raise RejectedParentVerificationError(
            "Parent is not the exact frozen rejected v2 development stage"
        )
    if set(bundle.payload_sha256) != set(_artifacts.DEVELOPMENT_PAYLOAD_NAMES):
        raise RejectedParentVerificationError(
            "Rejected parent has the wrong exact 47-payload inventory"
        )
    if set(bundle.checksums) != {
        *bundle.payload_sha256,
        "stage_manifest.json",
    } or len(bundle.checksums) != 48:
        raise RejectedParentVerificationError(
            "Rejected parent must contain exactly 48 checksummed files"
        )
    exact_hashes = {
        _CHECKPOINT_FILENAME: PARENT_CHECKPOINT_PAYLOAD_SHA256,
        _GATE_REPORT_FILENAME: PARENT_GATE_REPORT_PAYLOAD_SHA256,
        _REPORT_FILENAME: PARENT_REPORT_PAYLOAD_SHA256,
    }
    for filename, expected in exact_hashes.items():
        if bundle.payload_sha256.get(filename) != expected:
            raise RejectedParentVerificationError(
                f"Rejected-parent exact hash changed: {filename}"
            )

    gate = _json_payload(bundle, _GATE_REPORT_FILENAME)
    report = _json_payload(bundle, _REPORT_FILENAME)
    checkpoint = _json_payload(bundle, _CHECKPOINT_FILENAME)
    proof = _json_payload(bundle, _CAUSAL_PROOF_FILENAME)
    checks = gate.get("checks")
    if not isinstance(checks, dict) or any(
        type(value) is not bool for value in checks.values()
    ):
        raise RejectedParentVerificationError(
            "Rejected-parent gate checks are malformed"
        )
    false_checks = tuple(sorted(name for name, value in checks.items() if not value))
    if (
        gate.get("passed") is not False
        or gate.get("fatal_integrity_rejection") is not False
        or gate.get("passed_count") != PARENT_PASSED_CHECK_COUNT
        or gate.get("total_count") != PARENT_TOTAL_CHECK_COUNT
        or len(checks) != PARENT_TOTAL_CHECK_COUNT
        or tuple(gate.get("failed_checks", ())) != PARENT_FAILED_CHECKS
        or false_checks != PARENT_FAILED_CHECKS
        or report.get("contract_version") != PARENT_CONTRACT_VERSION
        or report.get("stage") != PARENT_STAGE
        or report.get("run_id") != PARENT_RUN_ID
        or report.get("stage_pass") is not False
        or report.get("status") != PARENT_STATUS
        or report.get("gate_report") != gate
    ):
        raise RejectedParentVerificationError(
            "Rejected report and exact 35-of-36 gate result disagree"
        )
    try:
        parsed_checkpoint = _artifacts.parse_composite_stage_checkpoint_bytes(
            _v2_verifier._payload(bundle, _CHECKPOINT_FILENAME)
        )
    except Exception as exc:
        raise RejectedParentVerificationError(
            "Rejected-parent checkpoint is invalid"
        ) from exc
    if (
        checkpoint.get("checkpoint_sha256")
        != PARENT_CHECKPOINT_SELF_SHA256
        or parsed_checkpoint.checkpoint_sha256
        != PARENT_CHECKPOINT_SELF_SHA256
    ):
        raise RejectedParentVerificationError(
            "Rejected-parent checkpoint self-hash changed"
        )
    unsigned_proof = dict(proof)
    recorded_proof_hash = unsigned_proof.pop("proof_sha256", None)
    if (
        recorded_proof_hash != PARENT_CAUSAL_PROOF_SELF_SHA256
        or _stage_runner._sha256_json(unsigned_proof)
        != PARENT_CAUSAL_PROOF_SELF_SHA256
    ):
        raise RejectedParentVerificationError(
            "Rejected-parent causal-prefix proof changed"
        )


def _regenerate_rejected_semantics(
    bundle: _experiment.VerifiedBundle,
    *,
    repo_root: Path,
    sealed_git_identity: Mapping[str, Any],
) -> _RegeneratedSemantics:
    """Reproduce v2 semantics, replacing only its obsolete live branch check."""

    try:
        computation = _v2_verifier._development_computation(repo_root)
        integrity, gate_report, metrics = _stage_runner._gate_and_metrics(
            computation
        )
        expected_payloads, checkpoint = _stage_runner._base_payloads(
            computation,
            integrity=integrity,
            gate_report=gate_report,
            metrics=metrics,
        )
        expected_names = _artifacts.DEVELOPMENT_PAYLOAD_NAMES
        if set(expected_payloads) != set(expected_names) - {
            "development_runtime_cost_evidence.json",
            "report.json",
        }:
            raise RejectedParentVerificationError(
                "V2 regeneration produced the wrong base payload inventory"
            )
        for filename, expected in expected_payloads.items():
            _v2_verifier._require_exact_payload(bundle, filename, expected)

        runtime = _v2_verifier._runtime_evidence(bundle, stage=PARENT_STAGE)
        expected_payloads["development_runtime_cost_evidence.json"] = (
            _stage_runner._json_bytes(runtime)
        )
        report = _stage_runner._report(
            computation=computation,
            gate_report=gate_report,
            metrics=metrics,
            runtime=runtime,
            checkpoint=checkpoint,
        )
        report_bytes = _stage_runner._json_bytes(report)
        _v2_verifier._require_exact_payload(bundle, _REPORT_FILENAME, report_bytes)
        expected_payloads[_REPORT_FILENAME] = report_bytes
        if set(expected_payloads) != set(expected_names):
            raise RejectedParentVerificationError(
                "V2 regeneration produced the wrong exact payload inventory"
            )

        manifest_fields = _stage_runner._manifest_fields(
            computation=computation,
            git_identity=dict(sealed_git_identity),
            gate_report=gate_report,
            runtime=runtime,
            checkpoint=checkpoint,
        )
        expected_manifest = _experiment.self_hashed_manifest(
            {
                "contract_version": PARENT_CONTRACT_VERSION,
                **manifest_fields,
                "payload_sha256": {
                    name: _experiment.sha256_bytes(expected_payloads[name])
                    for name in sorted(expected_payloads)
                },
            }
        )
        expected_manifest_bytes = _experiment.pretty_json_bytes(expected_manifest)
        if (
            expected_manifest != dict(bundle.manifest)
            or expected_manifest_bytes != bundle.manifest_path.read_bytes()
        ):
            raise RejectedParentVerificationError(
                "Independently regenerated rejected-parent manifest differs"
            )
        expected_checksums = {
            **{
                name: _experiment.sha256_bytes(expected_payloads[name])
                for name in sorted(expected_payloads)
            },
            "stage_manifest.json": _experiment.sha256_bytes(
                expected_manifest_bytes
            ),
        }
        observed_checksums_bytes = (
            bundle.directory / "checksums.json"
        ).read_bytes()
        if (
            expected_checksums != dict(bundle.checksums)
            or observed_checksums_bytes
            != _experiment.pretty_json_bytes(expected_checksums)
        ):
            raise RejectedParentVerificationError(
                "Independently regenerated rejected-parent checksums differ"
            )
        semantic_hash = _v2_verifier._semantic_evidence_hash(
            bundle=bundle,
            stage=PARENT_STAGE,
            checkpoint=checkpoint,
            gate_report=gate_report,
            report=report,
        )
    except RejectedParentVerificationError:
        raise
    except Exception as exc:
        raise RejectedParentVerificationError(
            "Independent rejected-parent semantic regeneration failed"
        ) from exc
    return _RegeneratedSemantics(
        stage_pass=bool(gate_report["passed"]),
        status=str(report["status"]),
        checkpoint_self_sha256=str(checkpoint["checkpoint_sha256"]),
        gate_report_file_sha256=bundle.payload_sha256[_GATE_REPORT_FILENAME],
        report_file_sha256=bundle.payload_sha256[_REPORT_FILENAME],
        semantic_evidence_sha256=semantic_hash,
    )


def _require_exact_regenerated_semantics(
    semantics: _RegeneratedSemantics,
) -> None:
    if (
        semantics.stage_pass is not False
        or semantics.status != PARENT_STATUS
        or semantics.checkpoint_self_sha256
        != PARENT_CHECKPOINT_SELF_SHA256
        or semantics.gate_report_file_sha256
        != PARENT_GATE_REPORT_PAYLOAD_SHA256
        or semantics.report_file_sha256 != PARENT_REPORT_PAYLOAD_SHA256
        or semantics.semantic_evidence_sha256
        != PARENT_SEMANTIC_EVIDENCE_SHA256
    ):
        raise RejectedParentVerificationError(
            "Regenerated semantics are not the exact frozen v2 rejection"
        )


def _load_exact_rejected_bundle(
    repo_root: Path,
) -> tuple[_experiment.VerifiedBundle, bytes, bytes]:
    directory = repo_root / PARENT_DIRECTORY
    manifest_path = directory / "stage_manifest.json"
    checksums_path = directory / "checksums.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        checksums_bytes = checksums_path.read_bytes()
    except OSError as exc:
        raise RejectedParentVerificationError(
            "Exact rejected-parent seal metadata is unreadable"
        ) from exc
    if (
        _experiment.sha256_bytes(manifest_bytes)
        != PARENT_MANIFEST_FILE_SHA256
        or _experiment.sha256_bytes(checksums_bytes)
        != PARENT_CHECKSUMS_FILE_SHA256
    ):
        raise RejectedParentVerificationError(
            "Exact rejected-parent manifest or checksum bytes changed"
        )
    try:
        bundle = _experiment.verify_exact_bundle(
            directory,
            expected_contract_version=PARENT_CONTRACT_VERSION,
            expected_stage=PARENT_STAGE,
            expected_manifest_sha256=PARENT_MANIFEST_SELF_SHA256,
            expected_payload_names=_artifacts.DEVELOPMENT_PAYLOAD_NAMES,
            require_stage_pass=False,
            repo_root=repo_root,
        )
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise RejectedParentVerificationError(
            "Exact rejected-parent bundle verification failed"
        ) from exc
    return bundle, manifest_bytes, checksums_bytes


def verify_rejected_parent(repo_root: Path) -> RejectedParentEvidence:
    """Verify and regenerate the one authorized rejected v2 parent.

    This function intentionally has no caller-supplied bundle path, identity,
    hashes, or semantic callback.  The audit runner can therefore bind only the
    preregistered parent.  A passing parent or any differently rejected bundle
    is rejected.
    """

    root = _require_repository_root(repo_root)
    bundle, manifest_bytes, checksums_bytes = _load_exact_rejected_bundle(root)
    _require_exact_static_rejection(bundle)
    (
        sealed_git_identity,
        historical_dependencies,
        live_semantic_dependencies,
    ) = _require_parent_git_and_dependency_continuity(root, bundle)
    semantics = _regenerate_rejected_semantics(
        bundle,
        repo_root=root,
        sealed_git_identity=sealed_git_identity,
    )
    _require_exact_regenerated_semantics(semantics)

    manifest_relative = (PARENT_DIRECTORY / "stage_manifest.json").as_posix()
    return RejectedParentEvidence(
        verifier_id=VERIFIER_ID,
        verifier_dependency_path=VERIFIER_IMPLEMENTATION_PATH.as_posix(),
        verified=True,
        parent_rejection_remains_final=True,
        contract_version=PARENT_CONTRACT_VERSION,
        stage=PARENT_STAGE,
        run_id=PARENT_RUN_ID,
        status=PARENT_STATUS,
        stage_pass=False,
        implementation_commit=PARENT_IMPLEMENTATION_COMMIT,
        rejection_preservation_commit=PARENT_REJECTION_PRESERVATION_COMMIT,
        manifest_path=manifest_relative,
        manifest_self_sha256=PARENT_MANIFEST_SELF_SHA256,
        manifest_file_sha256=PARENT_MANIFEST_FILE_SHA256,
        checksums_file_sha256=PARENT_CHECKSUMS_FILE_SHA256,
        checkpoint_self_sha256=PARENT_CHECKPOINT_SELF_SHA256,
        checkpoint_payload_sha256=PARENT_CHECKPOINT_PAYLOAD_SHA256,
        report_payload_sha256=PARENT_REPORT_PAYLOAD_SHA256,
        gate_report_payload_sha256=PARENT_GATE_REPORT_PAYLOAD_SHA256,
        semantic_evidence_sha256=PARENT_SEMANTIC_EVIDENCE_SHA256,
        parent_causal_proof_self_sha256=PARENT_CAUSAL_PROOF_SELF_SHA256,
        passed_check_count=PARENT_PASSED_CHECK_COUNT,
        total_check_count=PARENT_TOTAL_CHECK_COUNT,
        failed_checks=PARENT_FAILED_CHECKS,
        exact_payload_names=tuple(sorted(bundle.payload_sha256)),
        checksummed_names=tuple(sorted(bundle.checksums)),
        historical_dependency_paths=historical_dependencies,
        live_semantic_dependency_paths=live_semantic_dependencies,
        old_branch_equality_replaced_by_audit_continuity=True,
        bundle=bundle,
        manifest_bytes=manifest_bytes,
        checksums_bytes=checksums_bytes,
    )


__all__ = [
    "PARENT_CHECKPOINT_PAYLOAD_SHA256",
    "PARENT_CHECKPOINT_SELF_SHA256",
    "PARENT_FAILED_CHECKS",
    "PARENT_GATE_REPORT_PAYLOAD_SHA256",
    "PARENT_IMPLEMENTATION_COMMIT",
    "PARENT_MANIFEST_FILE_SHA256",
    "PARENT_MANIFEST_SELF_SHA256",
    "PARENT_REJECTION_PRESERVATION_COMMIT",
    "PARENT_REPORT_PAYLOAD_SHA256",
    "PARENT_SEMANTIC_EVIDENCE_SHA256",
    "RejectedParentEvidence",
    "RejectedParentVerificationError",
    "VERIFIER_ID",
    "VERIFIER_IMPLEMENTATION_PATH",
    "verify_rejected_parent",
]
