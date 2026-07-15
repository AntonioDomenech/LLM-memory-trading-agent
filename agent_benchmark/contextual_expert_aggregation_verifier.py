"""Independent exact verifier for sealed contextual aggregation stages.

The authorized command surface is deliberately fixed::

    python -I -B agent_benchmark/contextual_expert_aggregation_bootstrap.py verify development
    python -I -B agent_benchmark/contextual_expert_aggregation_bootstrap.py verify confirmation

The verifier never trusts sealed summaries.  It reloads the authorized local
inputs, replays the model, rebuilds the continuous ledgers and derived
evidence, reapplies the gates, and compares canonical bytes before accepting a
bundle.  It performs no network, news, API, or LLM work.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from . import contextual_expert_aggregation_artifacts as _artifacts
from . import contextual_expert_aggregation_bootstrap as _bootstrap
from . import contextual_expert_aggregation_experiment as _experiment
from . import contextual_expert_aggregation_replay as _replay
from . import contextual_expert_aggregation_stage as _stage_runner


VERIFIER_ID = "contextual-expert-aggregation-exact-verifier-v1"
VERIFIER_IMPLEMENTATION_PATH = Path(
    "agent_benchmark/contextual_expert_aggregation_verifier.py"
)
FROZEN_COMMAND_BY_STAGE: Mapping[str, tuple[str, ...]] = {
    stage: (
        "python",
        "-I",
        "-B",
        "agent_benchmark/contextual_expert_aggregation_bootstrap.py",
        "verify",
        stage,
    )
    for stage in _artifacts.STAGE_ORDER
}

_CONTRACT_VERSION = _artifacts.CONTRACT_VERSION
_DEVELOPMENT = _artifacts.DEVELOPMENT_STAGE
_CONFIRMATION = _artifacts.CONFIRMATION_STAGE
_STAGES = _artifacts.STAGE_ORDER
_ARM_ORDER = _artifacts.ARM_ORDER
_ONLINE_FULL = _replay.ONLINE_FULL_ARM

_RUNTIME_PHASES: Mapping[str, tuple[str, ...]] = {
    _DEVELOPMENT: (
        "development git authorization",
        "development authorized input loaded",
        "development replay completed",
        "development fixed parent prefix proved",
        "development ledgers and differences completed",
        "development gates completed",
        "development payload construction completed",
    ),
    _CONFIRMATION: (
        "confirmation attempt durably authorized",
        "development checkpoint and account prefix replayed",
        "confirmation replay forks completed",
        "confirmation ledgers and differences completed",
        "confirmation gates completed",
        "confirmation payload construction completed",
    ),
}

_RUNTIME_KEYS = frozenset(
    {
        "runtime_evidence_schema_version",
        "stage",
        "clock",
        "limit_seconds",
        "strict_boundary",
        "phase_elapsed_seconds",
        "seconds_before_payload_finalization",
        "within_limit_before_payload_finalization",
        "authoritative_last_check",
        "network_access",
        "news_access",
        "llm_calls",
        "api_calls",
        "external_cost_usd",
    }
)

_AUTHORIZATION_KEYS = frozenset(
    {
        "authorization_schema_version",
        "development_manifest_path",
        "development_manifest_sha256",
        "development_checkpoint_sha256",
        "development_gate_report_sha256",
        "attempt_identity_sha256",
        "attempt_lock_filename",
        "attempt_lock_sha256",
        "attempt_lock_parent_fsync_supported",
        "prelock_git_identity",
        "postlock_git_identity",
        "development_verification",
        "attempt_lock_payload",
        "confirmation_input_opened_before_lock",
        "authorization_sha256",
    }
)

_GIT_METADATA_KEYS = frozenset(
    {
        "branch",
        "commit",
        "upstream",
        "upstream_remote",
        "upstream_branch",
        "upstream_commit",
        "origin_url",
        "origin_repository",
        "head_equals_upstream",
    }
)
_PRELOCK_GIT_IDENTITY_KEYS = _GIT_METADATA_KEYS | {
    "dirty",
    "cleanliness_scope",
    "confirmation_input_bytes_opened",
    "tracked_dependency_identity",
    "runtime_versions",
}
_POSTLOCK_GIT_IDENTITY_KEYS = _GIT_METADATA_KEYS | {
    "dirty",
    "cleanliness_scope",
    "tracked_dependency_identity",
    "runtime_versions",
}


class _VerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class _SemanticVerification:
    stage: str
    stage_pass: bool
    status: str
    run_id: str
    checkpoint_sha256: str
    gate_report_file_sha256: str
    report_file_sha256: str
    semantic_evidence_sha256: str


@dataclass(frozen=True)
class _ConfirmationParent:
    authorization: Mapping[str, Any]
    bundle: _experiment.VerifiedBundle
    evidence: _experiment.DevelopmentVerificationEvidence
    manifest_bytes: bytes
    checksums_bytes: bytes


def _selected_stage(value: Any) -> str:
    if type(value) is not str or value not in _STAGES:
        raise _VerificationError(
            "stage must be exactly development or confirmation"
        )
    return value


def _payload(bundle: _experiment.VerifiedBundle, filename: str) -> bytes:
    expected = bundle.payload_sha256.get(filename)
    if expected is None:
        raise _VerificationError(f"sealed bundle omits {filename}")
    try:
        value = (bundle.directory / filename).read_bytes()
    except OSError as exc:
        raise _VerificationError(f"sealed payload is unreadable: {filename}") from exc
    if _experiment.sha256_bytes(value) != expected:
        raise _VerificationError(
            f"sealed payload changed after bundle verification: {filename}"
        )
    return value


def _json_payload(
    bundle: _experiment.VerifiedBundle, filename: str
) -> dict[str, Any]:
    value = _stage_runner._strict_json(_payload(bundle, filename), field=filename)
    if not isinstance(value, dict):
        raise _VerificationError(f"sealed JSON payload is not an object: {filename}")
    return value


def _repo_root_for_bundle(
    bundle: _experiment.VerifiedBundle, *, stage: str
) -> Path:
    selected = _selected_stage(stage)
    relative = _artifacts.OUTPUT_DIRECTORY_BY_STAGE[selected]
    directory = bundle.directory.resolve()
    root = directory
    for _ in relative.parts:
        root = root.parent
    if (root / relative).resolve() != directory:
        raise _VerificationError(
            "sealed bundle is not at the frozen repository-relative path"
        )
    return root


def _require_exact_payload(
    bundle: _experiment.VerifiedBundle,
    filename: str,
    expected: bytes,
) -> None:
    observed = _payload(bundle, filename)
    if observed != expected:
        raise _VerificationError(
            f"independently regenerated payload differs: {filename}"
        )


def _runtime_evidence(
    bundle: _experiment.VerifiedBundle, *, stage: str
) -> dict[str, Any]:
    selected = _selected_stage(stage)
    filename = f"{selected}_runtime_cost_evidence.json"
    runtime = _json_payload(bundle, filename)
    if set(runtime) != set(_RUNTIME_KEYS):
        raise _VerificationError("runtime evidence has the wrong exact inventory")
    phases = runtime.get("phase_elapsed_seconds")
    if not isinstance(phases, list) or len(phases) != len(
        _RUNTIME_PHASES[selected]
    ):
        raise _VerificationError("runtime evidence has the wrong phase inventory")
    elapsed: list[float] = []
    for expected_phase, item in zip(_RUNTIME_PHASES[selected], phases):
        if not isinstance(item, dict) or set(item) != {
            "phase",
            "elapsed_seconds",
        }:
            raise _VerificationError("runtime phase evidence is malformed")
        value = item["elapsed_seconds"]
        if (
            item["phase"] != expected_phase
            or type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise _VerificationError("runtime phase evidence changed")
        elapsed.append(value)
    limit = runtime.get("limit_seconds")
    if (
        type(limit) is not float
        or limit != _experiment.RUN_TIME_LIMIT_SECONDS
        or any(right < left for left, right in zip(elapsed, elapsed[1:]))
        or any(value >= limit for value in elapsed)
        or type(runtime.get("runtime_evidence_schema_version")) is not int
        or runtime.get("runtime_evidence_schema_version")
        != _stage_runner.STAGE_EVIDENCE_SCHEMA_VERSION
        or runtime.get("stage") != selected
        or runtime.get("clock") != "monotonic"
        or runtime.get("strict_boundary") != "elapsed_seconds < limit_seconds"
        or type(runtime.get("seconds_before_payload_finalization")) is not float
        or runtime.get("seconds_before_payload_finalization") != elapsed[-1]
        or runtime.get("within_limit_before_payload_finalization") is not True
        or runtime.get("authoritative_last_check")
        != "seal_exact_bundle immediately before atomic promotion"
        or runtime.get("network_access") is not False
        or runtime.get("news_access") is not False
        or type(runtime.get("llm_calls")) is not int
        or runtime.get("llm_calls") != 0
        or type(runtime.get("api_calls")) is not int
        or runtime.get("api_calls") != 0
        or type(runtime.get("external_cost_usd")) is not float
        or runtime.get("external_cost_usd") != 0.0
    ):
        raise _VerificationError("runtime or zero-cost evidence changed")
    return runtime


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise _VerificationError("Git verification command failed") from exc


def _validate_manifest_git_identity(
    repo_root: Path, value: Any, *, stage: str
) -> dict[str, Any]:
    selected = _selected_stage(stage)
    if not isinstance(value, Mapping):
        raise _VerificationError("manifest Git identity is not an object")
    identity = dict(value)
    if set(identity) != set(_POSTLOCK_GIT_IDENTITY_KEYS):
        raise _VerificationError("manifest Git identity has the wrong inventory")
    commit = identity.get("commit")
    if (
        identity.get("branch") != _experiment.EXPECTED_BRANCH
        or identity.get("upstream")
        != f"origin/{_experiment.EXPECTED_BRANCH}"
        or identity.get("upstream_remote") != "origin"
        or identity.get("upstream_branch") != _experiment.EXPECTED_BRANCH
        or identity.get("upstream_commit") != commit
        or identity.get("origin_url") not in _experiment.EXPECTED_ORIGIN_URLS
        or identity.get("origin_repository")
        != _experiment.EXPECTED_ORIGIN_REPOSITORY
        or identity.get("head_equals_upstream") is not True
        or identity.get("dirty") is not False
        or identity.get("cleanliness_scope")
        != "git_visible_index_and_worktree_after_attempt_lock"
        or type(commit) is not str
        or len(commit) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise _VerificationError("manifest Git identity claims changed")
    current = _experiment._git_metadata_identity(repo_root)
    for field in (
        "branch",
        "upstream",
        "upstream_remote",
        "upstream_branch",
        "origin_url",
        "origin_repository",
        "head_equals_upstream",
    ):
        if identity[field] != current[field]:
            raise _VerificationError(
                "current repository identity differs from the sealed run"
            )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current["commit"]],
        cwd=repo_root,
        capture_output=True,
    ).returncode != 0:
        raise _VerificationError("sealed run commit is not an ancestor of HEAD")
    revision = f"{commit}..{current['commit']}"
    all_delta = _nul_path_set(
        _git_bytes(
            repo_root,
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACDMRTUXB",
            revision,
            "--",
        ),
        field="tracked descendant tree delta",
    )
    added_delta = _nul_path_set(
        _git_bytes(
            repo_root,
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=A",
            revision,
            "--",
        ),
        field="added descendant tree delta",
    )
    if current["commit"] == commit:
        if all_delta or added_delta:
            raise _VerificationError("same-commit tracked tree delta is nonempty")
    else:
        output_directory = _artifacts.OUTPUT_DIRECTORY_BY_STAGE[selected]
        expected_delta = {
            (output_directory / filename).as_posix().encode("utf-8")
            for filename in (
                *sorted(_artifacts.payload_names_for_stage(selected)),
                "stage_manifest.json",
                "checksums.json",
            )
        }
        if all_delta != expected_delta or added_delta != expected_delta:
            raise _VerificationError(
                "descendant commits changed files beyond the exact stage output"
            )
    dependencies = identity.get("tracked_dependency_identity")
    current_dependencies = _experiment._tracked_dependency_identity(repo_root)
    if dependencies != current_dependencies:
        raise _VerificationError(
            "frozen dependency identity differs from the sealed run"
        )
    if identity.get("runtime_versions") != _experiment._runtime_versions():
        raise _VerificationError("runtime versions differ from the sealed run")
    return identity


def _post_run_worktree_policy(repo_root: Path, *, stage: str) -> str:
    selected = _selected_stage(stage)
    expected_names = _artifacts.payload_names_for_stage(selected)
    relative_directory = _artifacts.OUTPUT_DIRECTORY_BY_STAGE[selected]
    expected_paths = {
        (relative_directory / filename).as_posix().encode("utf-8")
        for filename in (
            *sorted(expected_names),
            "stage_manifest.json",
            "checksums.json",
        )
    }
    raw = _git_bytes(
        repo_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if not raw:
        for encoded in sorted(expected_paths):
            relative = encoded.decode("utf-8")
            _experiment.tracked_file_identity(
                repo_root,
                repo_root / Path(relative),
                require_literal_local_bytes=True,
            )
        return "clean_after_output_commit"
    if not raw.endswith(b"\0"):
        raise _VerificationError("NUL-delimited Git status is truncated")
    records = raw[:-1].split(b"\0")
    observed: set[bytes] = set()
    for record in records:
        if len(record) < 4 or not record.startswith(b"?? "):
            raise _VerificationError(
                "post-run worktree contains tracked, staged, or foreign dirt"
            )
        path = record[3:]
        if not path or path in observed:
            raise _VerificationError("post-run Git status path is invalid")
        observed.add(path)
    if observed != expected_paths:
        raise _VerificationError(
            "untracked worktree is not exactly the generated stage output"
        )
    return "exact_generated_output_untracked_before_commit"


def _nul_path_set(raw: bytes, *, field: str) -> set[bytes]:
    if not raw:
        return set()
    if not raw.endswith(b"\0"):
        raise _VerificationError(f"{field} is not NUL terminated")
    paths = raw[:-1].split(b"\0")
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise _VerificationError(f"{field} contains invalid or duplicate paths")
    return set(paths)


def _development_computation(repo_root: Path) -> _stage_runner.StageComputation:
    snapshot = _experiment.load_authorized_prices(
        repo_root, stage=_DEVELOPMENT
    )
    replays, replay_extra = _stage_runner._development_replays(snapshot.frame)
    fixed = _stage_runner._fixed_features_for_replays(replays)
    parent_proof, source_provenance = _stage_runner._fixed_parent_prefix_proof(
        repo_root,
        stage=_DEVELOPMENT,
        fixed_features=fixed,
    )
    ledgers, accounts = _stage_runner._run_development_ledgers(
        snapshot=snapshot, replays=replays
    )
    episodes, xors = _stage_runner._extract_episodes_and_xors(ledgers)
    return _stage_runner.StageComputation(
        stage=_DEVELOPMENT,
        snapshot=snapshot,
        replays=replays,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        fixed_parent_proof=parent_proof,
        source_bundle_provenance=source_provenance,
        replay_diagnostics=_stage_runner._replay_diagnostics_payload(
            stage=_DEVELOPMENT,
            replays=replays,
            extra=replay_extra,
        ),
        prefix_continuity_proof=None,
        confirmation_authorization=None,
        development_parent_manifest_bytes=None,
        development_parent_checksums_bytes=None,
    )


def _development_verification_mapping(
    evidence: _experiment.DevelopmentVerificationEvidence,
    *,
    prelock_identity: Mapping[str, Any],
) -> dict[str, Any]:
    dependencies = prelock_identity.get("tracked_dependency_identity")
    if not isinstance(dependencies, Mapping):
        raise _VerificationError(
            "confirmation prelock identity omits frozen dependencies"
        )
    dependency = dependencies.get(evidence.verifier_dependency_path)
    if not isinstance(dependency, Mapping):
        raise _VerificationError(
            "confirmation prelock identity omits the verifier dependency"
        )
    return {
        "verifier_id": evidence.verifier_id,
        "verifier_dependency_path": evidence.verifier_dependency_path,
        "verifier_dependency_identity": dict(dependency),
        "exact_payload_names": list(evidence.exact_payload_names),
        "report_sha256": evidence.report_sha256,
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "gate_report_sha256": evidence.gate_report_sha256,
        "semantic_evidence_sha256": evidence.semantic_evidence_sha256,
        "passed": True,
    }


def _direct_confirmation_parent(
    repo_root: Path,
    confirmation_bundle: _experiment.VerifiedBundle,
) -> _ConfirmationParent:
    authorization = _json_payload(
        confirmation_bundle, "confirmation_attempt_authorization.json"
    )
    if set(authorization) != set(_AUTHORIZATION_KEYS):
        raise _VerificationError(
            "confirmation authorization has the wrong exact inventory"
        )
    unsigned_authorization = dict(authorization)
    supplied_authorization_hash = unsigned_authorization.pop(
        "authorization_sha256"
    )
    if (
        supplied_authorization_hash
        != _stage_runner._sha256_json(unsigned_authorization)
        or authorization.get("authorization_schema_version")
        != _stage_runner.STAGE_EVIDENCE_SCHEMA_VERSION
        or authorization.get("confirmation_input_opened_before_lock") is not False
        or authorization.get("attempt_lock_parent_fsync_supported") is not True
    ):
        raise _VerificationError("confirmation authorization self-proof changed")

    expected_manifest_relative = (
        _stage_runner.DEVELOPMENT_MANIFEST_PATH.as_posix()
    )
    if authorization.get("development_manifest_path") != expected_manifest_relative:
        raise _VerificationError(
            "confirmation authorization names the wrong development parent"
        )
    manifest_path = repo_root / _stage_runner.DEVELOPMENT_MANIFEST_PATH
    parent = _experiment.verify_exact_bundle(
        manifest_path.parent,
        expected_contract_version=_CONTRACT_VERSION,
        expected_stage=_DEVELOPMENT,
        expected_manifest_sha256=authorization.get(
            "development_manifest_sha256"
        ),
        expected_payload_names=_artifacts.DEVELOPMENT_PAYLOAD_NAMES,
        require_stage_pass=True,
        repo_root=repo_root,
    )
    evidence = verify_development_authorization_bundle(
        parent, _experiment.DEVELOPMENT_PRICE_SPEC
    )
    if (
        authorization.get("development_checkpoint_sha256")
        != parent.payload_sha256[
            _experiment.DEVELOPMENT_CHECKPOINT_FILENAME
        ]
        or authorization.get("development_gate_report_sha256")
        != parent.payload_sha256[
            _experiment.DEVELOPMENT_GATE_REPORT_FILENAME
        ]
    ):
        raise _VerificationError(
            "confirmation authorization parent file hashes changed"
        )

    prelock = authorization.get("prelock_git_identity")
    postlock = authorization.get("postlock_git_identity")
    if not isinstance(prelock, Mapping) or not isinstance(postlock, Mapping):
        raise _VerificationError("confirmation Git identities are malformed")
    if (
        set(prelock) != set(_PRELOCK_GIT_IDENTITY_KEYS)
        or set(postlock) != set(_POSTLOCK_GIT_IDENTITY_KEYS)
    ):
        raise _VerificationError(
            "confirmation Git identities have the wrong exact inventory"
        )
    try:
        _experiment._require_prelock_postlock_identity(prelock, postlock)
    except _experiment.ContextualExpertAggregationExperimentError as exc:
        raise _VerificationError(
            "confirmation Git identity changed across locking"
        ) from exc
    if (
        prelock.get("dirty") is not None
        or prelock.get("cleanliness_scope")
        != "git_metadata_and_frozen_dependency_paths_only"
        or prelock.get("confirmation_input_bytes_opened") is not False
        or postlock.get("dirty") is not False
        or postlock.get("cleanliness_scope")
        != "git_visible_index_and_worktree_after_attempt_lock"
    ):
        raise _VerificationError("confirmation prelock/postlock claims changed")
    verification = _development_verification_mapping(
        evidence, prelock_identity=prelock
    )
    if authorization.get("development_verification") != verification:
        raise _VerificationError(
            "confirmation authorization development verification changed"
        )

    development_manifest_sha256 = str(
        authorization["development_manifest_sha256"]
    )
    attempt_identity = _experiment._attempt_identity_sha256(
        development_manifest_sha256
    )
    expected_filename = (
        "confirmation-attempt-"
        f"{attempt_identity.removeprefix('sha256:')}.json"
    )
    if (
        authorization.get("attempt_identity_sha256") != attempt_identity
        or authorization.get("attempt_lock_filename") != expected_filename
    ):
        raise _VerificationError("confirmation attempt identity changed")
    common = _experiment._git_common_directory(repo_root)
    lock_path = (
        common
        / _experiment.CONFIRMATION_REGISTRY_DIRECTORY
        / expected_filename
    )
    try:
        lock_path = _experiment._harden_path(
            lock_path,
            label="Confirmation verification attempt lock",
            require_exists=True,
        )
        lock_bytes = lock_path.read_bytes()
        lock_value = json.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _VerificationError(
            "durable confirmation attempt lock is unreadable"
        ) from exc
    expected_lock = _experiment._attempt_lock_value(
        repo_root=repo_root,
        development_manifest_path=manifest_path,
        development_manifest_sha256=development_manifest_sha256,
        development_checkpoint_sha256=evidence.checkpoint_sha256,
        development_gate_report_sha256=evidence.gate_report_sha256,
        prelock_git_identity=prelock,
        development_verification=verification,
    )
    if (
        lock_value != expected_lock
        or authorization.get("attempt_lock_payload") != expected_lock
        or authorization.get("attempt_lock_sha256")
        != _experiment.sha256_bytes(lock_bytes)
        or lock_bytes != _experiment.pretty_json_bytes(expected_lock)
    ):
        raise _VerificationError(
            "durable confirmation attempt lock differs from authorization"
        )

    parent_manifest_bytes, parent_checksums_bytes = (
        _stage_runner._verified_parent_metadata_bytes(parent)
    )
    if (
        parent_manifest_bytes
        != _payload(
            confirmation_bundle, "development_parent_manifest.json"
        )
        or parent_checksums_bytes
        != _payload(
            confirmation_bundle, "development_parent_checksums.json"
        )
    ):
        raise _VerificationError(
            "confirmation embedded development parent metadata changed"
        )
    return _ConfirmationParent(
        authorization=dict(authorization),
        bundle=parent,
        evidence=evidence,
        manifest_bytes=parent_manifest_bytes,
        checksums_bytes=parent_checksums_bytes,
    )


def _load_confirmation_prices_direct(
    repo_root: Path,
    *,
    development_snapshot: _experiment.LoadedPriceSnapshot,
) -> _experiment.LoadedPriceSnapshot:
    """Load confirmation only after the caller directly verified the lock."""

    spec = _experiment.CONFIRMATION_PRICE_SPEC
    tracked = _experiment.tracked_file_identity(
        repo_root,
        repo_root / spec.relative_path,
        expected_sha256=spec.raw_sha256,
        expected_git_blob=spec.git_blob,
        require_literal_local_bytes=True,
    )
    _experiment.verify_authorized_price_lineage(repo_root, spec)
    raw = _experiment.load_bounded_price_snapshot(
        repo_root / spec.relative_path, spec=spec
    )
    provenance = {
        **dict(raw.provenance),
        "tracked_input": {
            "path": tracked.path,
            "sha256": tracked.sha256,
            "git_blob": tracked.git_blob,
        },
        "authorized_loader_lineage": {
            "source_manifest_sha256": spec.source_bundle.manifest_self_sha256,
            "source_provenance_sha256": spec.source_provenance_sha256,
            "source_parent_manifest_sha256": (
                spec.source_parent_bundle.manifest_self_sha256
            ),
        },
    }
    snapshot = _experiment._attest_loaded_snapshot(
        _experiment.LoadedPriceSnapshot(
            spec=raw.spec,
            frame=raw.frame,
            raw_sha256=raw.raw_sha256,
            canonical_csv_bytes=raw.canonical_csv_bytes,
            provenance=provenance,
        ),
        authorized_lineage=True,
    )
    _experiment.require_loaded_snapshot_integrity(
        snapshot, expected_spec=spec
    )
    _experiment.require_snapshot_prefix(
        development_snapshot,
        snapshot,
        development_path=repo_root / _experiment.DEVELOPMENT_PRICE_SPEC.relative_path,
        confirmation_path=repo_root / spec.relative_path,
    )
    return snapshot


def _confirmation_computation(
    repo_root: Path,
    bundle: _experiment.VerifiedBundle,
) -> _stage_runner.StageComputation:
    # This direct parent/lock proof intentionally precedes the only call that
    # can open the through-2023 authorized price snapshot.
    parent = _direct_confirmation_parent(repo_root, bundle)
    development_snapshot = _experiment.load_authorized_prices(
        repo_root, stage=_DEVELOPMENT
    )
    development_online = _replay.replay_from_empty(development_snapshot.frame)
    development_replays = {arm: development_online for arm in _ARM_ORDER}
    checkpoint_payload = _payload(
        parent.bundle, "development_checkpoint_through_2018.json"
    )
    development_checkpoint = (
        _artifacts.parse_composite_stage_checkpoint_bytes(checkpoint_payload)
    )
    checkpoint_payload_sha256 = (
        _stage_runner._require_authorized_checkpoint_payload(
            checkpoint_payload,
            development_checkpoint,
            authorized_payload_sha256=parent.evidence.checkpoint_sha256,
        )
    )
    replay_proofs: dict[str, Any] = {}
    for arm in _ARM_ORDER:
        if (
            development_checkpoint.replay_checkpoints[arm].to_dict()
            != development_online.checkpoint.to_dict()
        ):
            raise _VerificationError(
                f"development checkpoint does not replay for confirmation arm {arm}"
            )
        proof = _replay.prove_replay_prefix(
            development_online, development_online
        )
        proof.require()
        replay_proofs[arm] = asdict(proof)
    parent_replay_artifacts = (
        _stage_runner._require_development_replay_artifacts(
            parent.bundle, regenerated=development_online
        )
    )

    snapshot = _load_confirmation_prices_direct(
        repo_root, development_snapshot=development_snapshot
    )
    suffix = snapshot.frame.loc[
        snapshot.frame.index
        > pd.Timestamp(development_online.checkpoint.checkpoint_date)
    ].copy()
    if suffix.empty:
        raise _VerificationError("confirmation suffix is empty")
    forks = _replay.fork_confirmation_arms(
        suffix,
        development_online.checkpoint,
        historical_prefix=development_snapshot.frame,
    )
    replays = dict(forks.arms)
    full_online = _replay.replay_from_empty(snapshot.frame)
    resume = _replay.verify_resume_equivalence(full_online, replays[_ONLINE_FULL])
    resume.require()
    fixed = _stage_runner._fixed_features_for_replays(replays)
    parent_proof, source_provenance = (
        _stage_runner._fixed_parent_prefix_proof(
            repo_root,
            stage=_CONFIRMATION,
            fixed_features=fixed,
        )
    )
    ledgers, accounts = _stage_runner._run_confirmation_ledgers(
        development_bundle=parent.bundle,
        development_snapshot=development_snapshot,
        confirmation_snapshot=snapshot,
        development_replays=development_replays,
        confirmation_replays=replays,
        development_checkpoint=development_checkpoint,
    )
    episodes, xors = _stage_runner._extract_episodes_and_xors(ledgers)
    prefix_proof = {
        "proof_schema_version": _stage_runner.STAGE_EVIDENCE_SCHEMA_VERSION,
        "development_manifest_sha256": parent.bundle.manifest[
            "manifest_sha256"
        ],
        "development_checkpoint_payload_sha256": checkpoint_payload_sha256,
        "development_checkpoint_self_sha256": (
            development_checkpoint.checkpoint_sha256
        ),
        "fork_parent_checkpoint_sha256": forks.checkpoint_sha256,
        "arm_replay_prefix_proofs": replay_proofs,
        "online_resume_equivalence": asdict(resume),
        "development_parent_replay_artifacts": parent_replay_artifacts,
        "development_account_prefix_verified_for_every_cost_policy": True,
        "confirmation_suffix_first_session": (
            suffix.index[0].date().isoformat()
        ),
        "confirmation_suffix_last_session": (
            suffix.index[-1].date().isoformat()
        ),
    }
    prefix_proof["proof_sha256"] = _stage_runner._sha256_json(prefix_proof)
    return _stage_runner.StageComputation(
        stage=_CONFIRMATION,
        snapshot=snapshot,
        replays=replays,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        fixed_parent_proof=parent_proof,
        source_bundle_provenance=source_provenance,
        replay_diagnostics=_stage_runner._replay_diagnostics_payload(
            stage=_CONFIRMATION,
            replays=replays,
            extra={
                "common_fork_checkpoint_sha256": forks.checkpoint_sha256,
                "online_resume_equivalence": asdict(resume),
                "frozen_post_cutoff_admitted_events": int(
                    replays[_replay.FROZEN_2018_ARM].diagnostics.admitted_events
                ),
            },
        ),
        prefix_continuity_proof=prefix_proof,
        confirmation_authorization=parent.authorization,
        development_parent_manifest_bytes=parent.manifest_bytes,
        development_parent_checksums_bytes=parent.checksums_bytes,
    )


def _semantic_evidence_hash(
    *,
    bundle: _experiment.VerifiedBundle,
    stage: str,
    checkpoint: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    report: Mapping[str, Any],
) -> str:
    evidence = {
        "verifier_id": VERIFIER_ID,
        "contract_version": _CONTRACT_VERSION,
        "stage": _selected_stage(stage),
        "manifest_sha256": bundle.manifest["manifest_sha256"],
        "exact_payload_sha256": {
            name: bundle.payload_sha256[name]
            for name in sorted(bundle.payload_sha256)
        },
        "composite_checkpoint_internal_sha256": checkpoint[
            "checkpoint_sha256"
        ],
        "gate_report_file_sha256": bundle.payload_sha256[
            f"{stage}_gate_report.json"
        ],
        "gate_pass": bool(gate_report["passed"]),
        "report_file_sha256": bundle.payload_sha256["report.json"],
        "report_status": report["status"],
        "independent_regeneration_exact": True,
    }
    return _experiment.sha256_bytes(
        _experiment.canonical_json_bytes(evidence)
    )


def _verify_semantics(
    bundle: _experiment.VerifiedBundle,
    *,
    stage: str,
    repo_root: Path,
) -> _SemanticVerification:
    selected = _selected_stage(stage)
    computation = (
        _development_computation(repo_root)
        if selected == _DEVELOPMENT
        else _confirmation_computation(repo_root, bundle)
    )
    integrity, gate_report, metrics = _stage_runner._gate_and_metrics(computation)
    expected_payloads, checkpoint = _stage_runner._base_payloads(
        computation,
        integrity=integrity,
        gate_report=gate_report,
        metrics=metrics,
    )
    expected_names = _artifacts.payload_names_for_stage(selected)
    if set(expected_payloads) != set(expected_names) - {
        f"{selected}_runtime_cost_evidence.json",
        "report.json",
    }:
        raise _VerificationError(
            "independent regeneration produced the wrong base inventory"
        )
    for filename, expected in expected_payloads.items():
        _require_exact_payload(bundle, filename, expected)

    runtime = _runtime_evidence(bundle, stage=selected)
    expected_payloads[f"{selected}_runtime_cost_evidence.json"] = (
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
    _require_exact_payload(bundle, "report.json", report_bytes)
    expected_payloads["report.json"] = report_bytes
    if set(expected_payloads) != set(expected_names):
        raise _VerificationError(
            "independent regeneration produced the wrong exact inventory"
        )

    git_identity = _validate_manifest_git_identity(
        repo_root, bundle.manifest.get("git_identity"), stage=selected
    )
    if (
        selected == _CONFIRMATION
        and computation.confirmation_authorization is not None
        and git_identity
        != dict(
            computation.confirmation_authorization["postlock_git_identity"]
        )
    ):
        raise _VerificationError(
            "confirmation manifest Git identity differs from durable authorization"
        )
    manifest_fields = _stage_runner._manifest_fields(
        computation=computation,
        git_identity=git_identity,
        gate_report=gate_report,
        runtime=runtime,
        checkpoint=checkpoint,
    )
    expected_manifest = _experiment.self_hashed_manifest(
        {
            "contract_version": _CONTRACT_VERSION,
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
        raise _VerificationError(
            "independently regenerated manifest claims differ"
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
    try:
        observed_checksums_bytes = (
            bundle.directory / "checksums.json"
        ).read_bytes()
    except OSError as exc:
        raise _VerificationError("sealed checksum metadata is unreadable") from exc
    if (
        dict(bundle.checksums) != expected_checksums
        or observed_checksums_bytes
        != _experiment.pretty_json_bytes(expected_checksums)
    ):
        raise _VerificationError(
            "independently regenerated checksum metadata differs"
        )
    semantic_hash = _semantic_evidence_hash(
        bundle=bundle,
        stage=selected,
        checkpoint=checkpoint,
        gate_report=gate_report,
        report=report,
    )
    return _SemanticVerification(
        stage=selected,
        stage_pass=bool(gate_report["passed"]),
        status=str(report["status"]),
        run_id=str(bundle.manifest["run_id"]),
        checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        gate_report_file_sha256=bundle.payload_sha256[
            f"{selected}_gate_report.json"
        ],
        report_file_sha256=bundle.payload_sha256["report.json"],
        semantic_evidence_sha256=semantic_hash,
    )


def verify_development_authorization_bundle(
    bundle: _experiment.VerifiedBundle,
    spec: _experiment.AuthorizedPriceSpec,
) -> _experiment.DevelopmentVerificationEvidence:
    """Regenerate and authorize one exact passing 47-payload development bundle."""

    if not isinstance(bundle, _experiment.VerifiedBundle):
        raise _VerificationError("development verifier requires a VerifiedBundle")
    if spec != _experiment.DEVELOPMENT_PRICE_SPEC:
        raise _VerificationError(
            "development verifier requires the frozen development price spec"
        )
    expected_names = _artifacts.DEVELOPMENT_PAYLOAD_NAMES
    if set(bundle.payload_sha256) != set(expected_names):
        raise _VerificationError(
            "development authorization requires the exact 47-payload inventory"
        )
    repo_root = _repo_root_for_bundle(bundle, stage=_DEVELOPMENT)
    fresh = _experiment.verify_exact_bundle(
        bundle.directory,
        expected_contract_version=_CONTRACT_VERSION,
        expected_stage=_DEVELOPMENT,
        expected_manifest_sha256=bundle.manifest.get("manifest_sha256"),
        expected_payload_names=expected_names,
        require_stage_pass=True,
        repo_root=repo_root,
    )
    if (
        dict(fresh.manifest) != dict(bundle.manifest)
        or dict(fresh.payload_sha256) != dict(bundle.payload_sha256)
        or dict(fresh.checksums) != dict(bundle.checksums)
    ):
        raise _VerificationError(
            "development bundle changed before semantic authorization"
        )
    result = _verify_semantics(
        fresh, stage=_DEVELOPMENT, repo_root=repo_root
    )
    if result.stage_pass is not True or result.status != "PASSED":
        raise _VerificationError(
            "development authorization requires an independently reproduced pass"
        )
    return _experiment.DevelopmentVerificationEvidence(
        verifier_id=VERIFIER_ID,
        verifier_dependency_path=VERIFIER_IMPLEMENTATION_PATH.as_posix(),
        passed=True,
        exact_payload_names=tuple(sorted(expected_names)),
        report_sha256=result.report_file_sha256,
        checkpoint_sha256=fresh.payload_sha256[
            _experiment.DEVELOPMENT_CHECKPOINT_FILENAME
        ],
        gate_report_sha256=result.gate_report_file_sha256,
        semantic_evidence_sha256=result.semantic_evidence_sha256,
    )


DEVELOPMENT_VERIFIER_REGISTRATION = (
    _experiment.DevelopmentVerifierRegistration(
        verifier_id=VERIFIER_ID,
        dependency_path=VERIFIER_IMPLEMENTATION_PATH,
        expected_payload_names=frozenset(
            _artifacts.DEVELOPMENT_PAYLOAD_NAMES
        ),
        verify=verify_development_authorization_bundle,
    )
)


def verify_stage(
    stage: str, repo_root: Path | None = None
) -> dict[str, Any]:
    """Verify one fixed sealed stage, including its post-run Git state."""

    selected = _selected_stage(stage)
    root = Path.cwd().resolve() if repo_root is None else Path(repo_root).resolve()
    _bootstrap.require_active_attestation(
        operation="verify", stage=selected, repo_root=root
    )
    bundle = _experiment.verify_exact_bundle(
        root / _artifacts.OUTPUT_DIRECTORY_BY_STAGE[selected],
        expected_contract_version=_CONTRACT_VERSION,
        expected_stage=selected,
        expected_payload_names=_artifacts.payload_names_for_stage(selected),
    )
    worktree_mode = _post_run_worktree_policy(root, stage=selected)
    semantic = _verify_semantics(bundle, stage=selected, repo_root=root)
    return {
        "verifier_id": VERIFIER_ID,
        "contract_version": _CONTRACT_VERSION,
        "stage": selected,
        "verified": True,
        "stage_pass": semantic.stage_pass,
        "status": semantic.status,
        "run_id": semantic.run_id,
        "artifact_directory": str(bundle.directory),
        "manifest_sha256": bundle.manifest["manifest_sha256"],
        "checkpoint_sha256": semantic.checkpoint_sha256,
        "semantic_evidence_sha256": semantic.semantic_evidence_sha256,
        "post_run_worktree_mode": worktree_mode,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=_STAGES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = verify_stage(args.stage)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEVELOPMENT_VERIFIER_REGISTRATION",
    "FROZEN_COMMAND_BY_STAGE",
    "VERIFIER_ID",
    "VERIFIER_IMPLEMENTATION_PATH",
    "verify_development_authorization_bundle",
    "verify_stage",
]
