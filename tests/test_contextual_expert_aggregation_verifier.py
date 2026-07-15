from __future__ import annotations

import json
import os
import subprocess
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_artifacts as artifacts
from agent_benchmark import contextual_expert_aggregation_experiment as experiment
from agent_benchmark import contextual_expert_aggregation_replay as replay
from agent_benchmark import contextual_expert_aggregation_stage as stage_runner
from agent_benchmark import contextual_expert_aggregation_verifier as verifier


Error = verifier._VerificationError


def _small_development_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2004-06-01", "2004-12-31")
    dates = dates.append(
        pd.DatetimeIndex(
            ["2005-01-03", "2005-01-04"]
            + [f"{year}-12-31" for year in range(2006, 2019)]
        )
    )
    values = np.linspace(90.0, 110.0, len(dates))
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values * 1.001,
            "aapl_adj_close": values * 1.001,
            "spy_adj_close": np.linspace(100.0, 120.0, len(dates)),
            "qqq_adj_close": np.linspace(80.0, 105.0, len(dates)),
        },
        index=dates,
    )


def _small_confirmation_suffix() -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        [
            "2019-01-02",
            "2019-12-31",
            "2020-12-31",
            "2021-12-31",
            "2022-12-30",
            "2023-12-29",
        ]
    )
    values = np.linspace(111.0, 116.0, len(dates))
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values * 1.001,
            "aapl_adj_close": values * 1.001,
            "spy_adj_close": np.linspace(121.0, 126.0, len(dates)),
            "qqq_adj_close": np.linspace(106.0, 111.0, len(dates)),
        },
        index=dates,
    )


def _runtime(stage: str) -> dict[str, object]:
    phases = [
        {"phase": name, "elapsed_seconds": float(position + 1) / 10.0}
        for position, name in enumerate(verifier._RUNTIME_PHASES[stage])
    ]
    return {
        "runtime_evidence_schema_version": stage_runner.STAGE_EVIDENCE_SCHEMA_VERSION,
        "stage": stage,
        "clock": "monotonic",
        "limit_seconds": experiment.RUN_TIME_LIMIT_SECONDS,
        "strict_boundary": "elapsed_seconds < limit_seconds",
        "phase_elapsed_seconds": phases,
        "seconds_before_payload_finalization": phases[-1]["elapsed_seconds"],
        "within_limit_before_payload_finalization": True,
        "authoritative_last_check": (
            "seal_exact_bundle immediately before atomic promotion"
        ),
        "network_access": False,
        "news_access": False,
        "llm_calls": 0,
        "api_calls": 0,
        "external_cost_usd": 0.0,
    }


def _git_identity() -> dict[str, object]:
    commit = "1" * 40
    return {
        "branch": experiment.EXPECTED_BRANCH,
        "commit": commit,
        "upstream": f"origin/{experiment.EXPECTED_BRANCH}",
        "upstream_remote": "origin",
        "upstream_branch": experiment.EXPECTED_BRANCH,
        "upstream_commit": commit,
        "origin_url": next(iter(experiment.EXPECTED_ORIGIN_URLS)),
        "origin_repository": experiment.EXPECTED_ORIGIN_REPOSITORY,
        "head_equals_upstream": True,
        "dirty": False,
        "cleanliness_scope": "git_visible_index_and_worktree_after_attempt_lock",
        "tracked_dependency_identity": {},
        "runtime_versions": experiment._runtime_versions(),
    }


def _write_bundle(
    directory: Path,
    *,
    stage: str,
    manifest_fields: dict[str, object],
    payloads: dict[str, bytes],
) -> experiment.VerifiedBundle:
    directory.mkdir(parents=True)
    payload_hashes = {
        name: experiment.sha256_bytes(payload)
        for name, payload in sorted(payloads.items())
    }
    manifest = experiment.self_hashed_manifest(
        {
            "contract_version": artifacts.CONTRACT_VERSION,
            **manifest_fields,
            "payload_sha256": payload_hashes,
        }
    )
    manifest_bytes = experiment.pretty_json_bytes(manifest)
    checksums = {
        **payload_hashes,
        "stage_manifest.json": experiment.sha256_bytes(manifest_bytes),
    }
    for filename, payload in payloads.items():
        (directory / filename).write_bytes(payload)
    (directory / "stage_manifest.json").write_bytes(manifest_bytes)
    (directory / "checksums.json").write_bytes(
        experiment.pretty_json_bytes(checksums)
    )
    return experiment.verify_exact_bundle(
        directory,
        expected_contract_version=artifacts.CONTRACT_VERSION,
        expected_stage=stage,
        expected_payload_names=artifacts.payload_names_for_stage(stage),
    )


def _synthetic_development_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    experiment.VerifiedBundle,
    object,
    dict[str, bytes],
]:
    frame = _small_development_frame()
    start = frame.index[0].date().isoformat()
    monkeypatch.setattr(
        artifacts,
        "SOURCE_START_SESSION_BY_STAGE",
        {"development": start, "confirmation": start},
    )
    monkeypatch.setattr(
        artifacts,
        "SOURCE_SESSION_COUNT_BY_STAGE",
        {"development": len(frame), "confirmation": len(frame) + 3},
    )
    digest = "sha256:" + "a" * 64
    snapshot = SimpleNamespace(
        frame=frame,
        provenance={"source": "synthetic"},
        canonical_csv_bytes=b"synthetic authorized development prices\n",
        raw_sha256=digest,
        spec=SimpleNamespace(canonical_sha256=digest),
    )
    replays, replay_extra = stage_runner._development_replays(frame)
    ledgers, accounts = stage_runner._run_development_ledgers(
        snapshot=snapshot, replays=replays
    )
    episodes, xors = stage_runner._extract_episodes_and_xors(ledgers)
    parent_proof = {"proof_sha256": digest}
    source_provenance = {"provenance_sha256": digest}
    computation = stage_runner.StageComputation(
        stage="development",
        snapshot=snapshot,
        replays=replays,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        fixed_parent_proof=parent_proof,
        source_bundle_provenance=source_provenance,
        replay_diagnostics=stage_runner._replay_diagnostics_payload(
            stage="development", replays=replays, extra=replay_extra
        ),
        prefix_continuity_proof=None,
        confirmation_authorization=None,
        development_parent_manifest_bytes=None,
        development_parent_checksums_bytes=None,
    )
    integrity, gate_report, metrics = stage_runner._gate_and_metrics(computation)
    payloads, checkpoint = stage_runner._base_payloads(
        computation,
        integrity=integrity,
        gate_report=gate_report,
        metrics=metrics,
    )
    runtime = _runtime("development")
    payloads["development_runtime_cost_evidence.json"] = (
        stage_runner._json_bytes(runtime)
    )
    report = stage_runner._report(
        computation=computation,
        gate_report=gate_report,
        metrics=metrics,
        runtime=runtime,
        checkpoint=checkpoint,
    )
    payloads["report.json"] = stage_runner._json_bytes(report)
    git_identity = _git_identity()
    manifest_fields = stage_runner._manifest_fields(
        computation=computation,
        git_identity=git_identity,
        gate_report=gate_report,
        runtime=runtime,
        checkpoint=checkpoint,
    )
    directory = (
        tmp_path / artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
    )
    bundle = _write_bundle(
        directory,
        stage="development",
        manifest_fields=manifest_fields,
        payloads=payloads,
    )
    monkeypatch.setattr(
        experiment,
        "load_authorized_prices",
        lambda repo_root, stage: snapshot,
    )
    monkeypatch.setattr(
        stage_runner,
        "_fixed_parent_prefix_proof",
        lambda *args, **kwargs: (parent_proof, source_provenance),
    )
    monkeypatch.setattr(
        verifier,
        "_validate_manifest_git_identity",
        lambda repo_root, value, *, stage: dict(value),
    )
    return bundle, computation, payloads


def test_frozen_public_exports_and_immutable_registration() -> None:
    assert verifier.VERIFIER_ID == (
        "contextual-expert-aggregation-exact-verifier-v1"
    )
    assert verifier.VERIFIER_IMPLEMENTATION_PATH == Path(
        "agent_benchmark/contextual_expert_aggregation_verifier.py"
    )
    registration = verifier.DEVELOPMENT_VERIFIER_REGISTRATION
    assert registration.verifier_id == verifier.VERIFIER_ID
    assert registration.dependency_path == verifier.VERIFIER_IMPLEMENTATION_PATH
    assert registration.expected_payload_names == frozenset(
        artifacts.DEVELOPMENT_PAYLOAD_NAMES
    )
    assert registration.verify is verifier.verify_development_authorization_bundle
    with pytest.raises(FrozenInstanceError):
        registration.verifier_id = "changed"  # type: ignore[misc]
    assert verifier.__all__ == [
        "DEVELOPMENT_VERIFIER_REGISTRATION",
        "FROZEN_COMMAND_BY_STAGE",
        "VERIFIER_ID",
        "VERIFIER_IMPLEMENTATION_PATH",
        "verify_development_authorization_bundle",
        "verify_stage",
    ]
    assert verifier.FROZEN_COMMAND_BY_STAGE == {
        selected: (
            "python",
            "-I",
            "-B",
            "agent_benchmark/contextual_expert_aggregation_bootstrap.py",
            "verify",
            selected,
        )
        for selected in artifacts.STAGE_ORDER
    }


def test_cli_has_only_one_frozen_stage_argument() -> None:
    parser = verifier._build_parser()
    assert parser.parse_args(["development"]).stage == "development"
    assert parser.parse_args(["confirmation"]).stage == "confirmation"
    for bad in (
        [],
        ["validation"],
        ["development", "--repo-root", "elsewhere"],
        ["confirmation", "--manifest", "other.json"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_nul_safe_post_run_policy_allows_only_clean_commit_or_exact_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = (
        *sorted(artifacts.DEVELOPMENT_PAYLOAD_NAMES),
        "stage_manifest.json",
        "checksums.json",
    )
    prefix = artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
    exact = b"".join(
        (prefix / name).as_posix().encode() + b"\0" for name in names
    )
    monkeypatch.setattr(
        experiment,
        "_content_aware_git_deltas",
        lambda *args: (b"", b"", exact),
    )
    assert verifier._post_run_worktree_policy(
        tmp_path, stage="development"
    ) == "exact_generated_output_untracked_before_commit"

    monkeypatch.setattr(
        experiment,
        "_content_aware_git_deltas",
        lambda *args: (b"", b"", exact + b"foreign file\nwith newline\0"),
    )
    with pytest.raises(Error, match="not exactly"):
        verifier._post_run_worktree_policy(tmp_path, stage="development")

    monkeypatch.setattr(
        experiment,
        "_content_aware_git_deltas",
        lambda *args: (b"tracked.py\0", b"", b""),
    )
    with pytest.raises(Error, match="tracked, staged"):
        verifier._post_run_worktree_policy(tmp_path, stage="development")

    seen: list[Path] = []
    monkeypatch.setattr(
        experiment,
        "_content_aware_git_deltas",
        lambda *args: (b"", b"", b""),
    )
    monkeypatch.setattr(
        experiment,
        "tracked_file_identity",
        lambda root, path, **kwargs: seen.append(path),
    )
    assert verifier._post_run_worktree_policy(
        tmp_path, stage="development"
    ) == "clean_after_output_commit"
    assert len(seen) == len(names)


def test_post_run_policy_uses_exact_bootstrap_crlf_environment(
    tmp_path: Path,
) -> None:
    from agent_benchmark import contextual_expert_aggregation_bootstrap as bootstrap

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "--quiet")
    git("config", "user.name", "Verifier Status Test")
    git("config", "user.email", "verifier@example.invalid")
    dependency = repo / "dependency.py"
    dependency.write_bytes(b"line1\nline2\n")
    git("add", "dependency.py")
    git("commit", "--quiet", "-m", "baseline")
    dependency.write_bytes(b"line1\r\nline2\r\n")
    output = artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
    for name in (
        *artifacts.DEVELOPMENT_PAYLOAD_NAMES,
        "stage_manifest.json",
        "checksums.json",
    ):
        destination = repo / output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"synthetic\n")

    original_environment = dict(os.environ)
    safe_environment = bootstrap._git_environment()
    try:
        os.environ.clear()
        os.environ.update(safe_environment)
        policy = verifier._post_run_worktree_policy(
            repo,
            stage="development",
        )
    finally:
        os.environ.clear()
        os.environ.update(original_environment)

    assert policy == "exact_generated_output_untracked_before_commit"


def test_manifest_git_identity_allows_only_the_exact_stage_output_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _git_identity()
    current = {
        name: value
        for name, value in identity.items()
        if name
        not in {
            "dirty",
            "cleanliness_scope",
            "tracked_dependency_identity",
            "runtime_versions",
        }
    }
    current["commit"] = "2" * 40
    current["upstream_commit"] = current["commit"]
    monkeypatch.setattr(experiment, "_git_metadata_identity", lambda root: current)
    monkeypatch.setattr(experiment, "_tracked_dependency_identity", lambda root: {})
    monkeypatch.setattr(
        experiment,
        "_runtime_versions",
        lambda: identity["runtime_versions"],
    )
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    output = artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
    expected = {
        (output / name).as_posix().encode("utf-8")
        for name in (
            *sorted(artifacts.DEVELOPMENT_PAYLOAD_NAMES),
            "stage_manifest.json",
            "checksums.json",
        )
    }

    def encoded(paths: set[bytes]) -> bytes:
        return b"".join(path + b"\0" for path in sorted(paths))

    monkeypatch.setattr(verifier, "_git_bytes", lambda *args: encoded(expected))
    assert verifier._validate_manifest_git_identity(
        tmp_path, identity, stage="development"
    ) == identity

    foreign = expected | {b"agent_benchmark/shadow.py"}
    monkeypatch.setattr(verifier, "_git_bytes", lambda *args: encoded(foreign))
    with pytest.raises(Error, match="beyond the exact stage output"):
        verifier._validate_manifest_git_identity(
            tmp_path, identity, stage="development"
        )

    def modified_delta(root: Path, *args: str) -> bytes:
        return encoded(expected if "--diff-filter=A" not in args else set())

    monkeypatch.setattr(verifier, "_git_bytes", modified_delta)
    with pytest.raises(Error, match="beyond the exact stage output"):
        verifier._validate_manifest_git_identity(
            tmp_path, identity, stage="development"
        )


def test_manifest_descendant_delta_cannot_hide_foreign_gitlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_benchmark import contextual_expert_aggregation_bootstrap as bootstrap

    repo = tmp_path / "repo"
    repo.mkdir()

    def git_bytes(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout

    git_bytes("init", "--quiet")
    git_bytes("config", "user.name", "Gitlink Delta Test")
    git_bytes("config", "user.email", "gitlink@example.invalid")
    (repo / "baseline.txt").write_bytes(b"baseline\n")
    git_bytes("add", "baseline.txt")
    git_bytes("commit", "--quiet", "-m", "baseline")
    baseline = git_bytes("rev-parse", "HEAD").decode().strip()

    output = artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
    for name in (
        *artifacts.DEVELOPMENT_PAYLOAD_NAMES,
        "stage_manifest.json",
        "checksums.json",
    ):
        destination = repo / output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            b"* -text\n" if name == ".gitattributes" else b"synthetic\n"
        )
    git_bytes("add", output.as_posix())
    git_bytes(
        "update-index",
        "--add",
        "--cacheinfo",
        "160000",
        baseline,
        "vendor/foreign",
    )
    git_bytes("commit", "--quiet", "-m", "outputs plus foreign gitlink")
    current_commit = git_bytes("rev-parse", "HEAD").decode().strip()
    git_bytes("config", "diff.ignoreSubmodules", "all")
    hidden = git_bytes(
        "diff",
        "--name-only",
        "-z",
        "--diff-filter=ACDMRTUXB",
        f"{baseline}..{current_commit}",
        "--",
    )
    assert b"vendor/foreign\0" not in hidden

    identity = _git_identity()
    identity["commit"] = baseline
    identity["upstream_commit"] = baseline
    current = {
        key: value
        for key, value in identity.items()
        if key
        not in {
            "dirty",
            "cleanliness_scope",
            "tracked_dependency_identity",
            "runtime_versions",
        }
    }
    current["commit"] = current_commit
    current["upstream_commit"] = current_commit
    monkeypatch.setattr(experiment, "_git_metadata_identity", lambda root: current)
    monkeypatch.setattr(experiment, "_tracked_dependency_identity", lambda root: {})
    monkeypatch.setattr(
        experiment,
        "_runtime_versions",
        lambda: identity["runtime_versions"],
    )

    original_environment = dict(os.environ)
    safe_environment = bootstrap._git_environment()
    try:
        os.environ.clear()
        os.environ.update(safe_environment)
        with pytest.raises(Error, match="beyond the exact stage output"):
            verifier._validate_manifest_git_identity(
                repo,
                identity,
                stage="development",
            )
    finally:
        os.environ.clear()
        os.environ.update(original_environment)


def test_runtime_validation_rejects_boundary_and_noncanonical_phase(
    tmp_path: Path,
) -> None:
    value = _runtime("development")
    payload = stage_runner._json_bytes(value)
    path = tmp_path / "development_runtime_cost_evidence.json"
    path.write_bytes(payload)
    bundle = experiment.VerifiedBundle(
        directory=tmp_path,
        manifest_path=tmp_path / "stage_manifest.json",
        manifest={},
        payload_sha256={path.name: experiment.sha256_bytes(payload)},
        checksums={},
    )
    assert verifier._runtime_evidence(bundle, stage="development") == value

    bad = dict(value)
    bad["phase_elapsed_seconds"] = list(value["phase_elapsed_seconds"])  # type: ignore[arg-type]
    bad["phase_elapsed_seconds"][-1] = {  # type: ignore[index]
        "phase": verifier._RUNTIME_PHASES["development"][-1],
        "elapsed_seconds": experiment.RUN_TIME_LIMIT_SECONDS,
    }
    bad["seconds_before_payload_finalization"] = experiment.RUN_TIME_LIMIT_SECONDS
    bad_payload = stage_runner._json_bytes(bad)
    path.write_bytes(bad_payload)
    bundle = experiment.VerifiedBundle(
        directory=tmp_path,
        manifest_path=tmp_path / "stage_manifest.json",
        manifest={},
        payload_sha256={path.name: experiment.sha256_bytes(bad_payload)},
        checksums={},
    )
    with pytest.raises(Error, match="runtime"):
        verifier._runtime_evidence(bundle, stage="development")


def test_synthetic_development_is_regenerated_byte_for_byte_and_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, _, payloads = _synthetic_development_bundle(tmp_path, monkeypatch)
    semantic = verifier._verify_semantics(
        bundle, stage="development", repo_root=tmp_path
    )
    assert semantic.stage == "development"
    assert semantic.status in {"PASSED", "REJECTED"}
    assert semantic.semantic_evidence_sha256.startswith("sha256:")

    report = json.loads(payloads["report.json"])
    report["status"] = "PASSED" if report["status"] == "REJECTED" else "REJECTED"
    tampered_payloads = dict(payloads)
    tampered_payloads["report.json"] = stage_runner._json_bytes(report)
    tampered_directory = (
        tmp_path
        / "tampered"
        / artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
    )
    tampered = _write_bundle(
        tampered_directory,
        stage="development",
        manifest_fields={
            name: value
            for name, value in bundle.manifest.items()
            if name not in {"contract_version", "payload_sha256", "manifest_sha256"}
        },
        payloads=tampered_payloads,
    )
    with pytest.raises(Error, match="report.json"):
        verifier._verify_semantics(
            tampered, stage="development", repo_root=tmp_path
        )

    checksums_path = bundle.directory / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums_path.write_text(
        json.dumps(checksums, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(Error, match="checksum metadata"):
        verifier._verify_semantics(
            bundle, stage="development", repo_root=tmp_path
        )


def test_synthetic_confirmation_regenerates_suffix_and_full_continuous_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_bundle, development_computation, _ = (
        _synthetic_development_bundle(tmp_path, monkeypatch)
    )
    suffix = _small_confirmation_suffix()
    development_frame = development_computation.snapshot.frame
    confirmation_frame = pd.concat([development_frame, suffix])
    start = development_frame.index[0].date().isoformat()
    monkeypatch.setattr(
        artifacts,
        "SOURCE_SESSION_COUNT_BY_STAGE",
        {
            "development": len(development_frame),
            "confirmation": len(confirmation_frame),
        },
    )
    monkeypatch.setattr(
        artifacts,
        "SOURCE_START_SESSION_BY_STAGE",
        {"development": start, "confirmation": start},
    )
    development_online = development_computation.replays["online_full"]
    forks = replay.fork_confirmation_arms(
        suffix,
        development_online.checkpoint,
        historical_prefix=development_frame,
    )
    confirmation_replays = dict(forks.arms)
    digest = "sha256:" + "4" * 64
    snapshot = SimpleNamespace(
        frame=confirmation_frame,
        provenance={"source": "synthetic-confirmation"},
        canonical_csv_bytes=b"synthetic authorized confirmation prices\n",
        raw_sha256=digest,
        spec=SimpleNamespace(canonical_sha256=digest),
    )
    checkpoint_payload = (
        development_bundle.directory
        / "development_checkpoint_through_2018.json"
    ).read_bytes()
    development_checkpoint = artifacts.parse_composite_stage_checkpoint_bytes(
        checkpoint_payload
    )
    ledgers, accounts = stage_runner._run_confirmation_ledgers(
        development_bundle=development_bundle,
        development_snapshot=development_computation.snapshot,
        confirmation_snapshot=snapshot,
        development_replays=development_computation.replays,
        confirmation_replays=confirmation_replays,
        development_checkpoint=development_checkpoint,
    )
    episodes, xors = stage_runner._extract_episodes_and_xors(ledgers)
    full_online = replay.replay_from_empty(confirmation_frame)
    resume = replay.verify_resume_equivalence(
        full_online, confirmation_replays["online_full"]
    )
    resume.require()
    replay_proofs = {
        arm: asdict(
            replay.prove_replay_prefix(development_online, development_online)
        )
        for arm in artifacts.ARM_ORDER
    }
    parent_replay_artifacts = stage_runner._require_development_replay_artifacts(
        development_bundle, regenerated=development_online
    )
    checkpoint_payload_sha256 = experiment.sha256_bytes(checkpoint_payload)
    prefix_proof: dict[str, object] = {
        "proof_schema_version": stage_runner.STAGE_EVIDENCE_SCHEMA_VERSION,
        "development_manifest_sha256": development_bundle.manifest[
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
        "confirmation_suffix_first_session": "2019-01-02",
        "confirmation_suffix_last_session": "2023-12-29",
    }
    prefix_proof["proof_sha256"] = stage_runner._sha256_json(prefix_proof)
    git_identity = _git_identity()
    authorization = {
        "development_manifest_sha256": development_bundle.manifest[
            "manifest_sha256"
        ],
        "postlock_git_identity": git_identity,
    }
    parent_manifest_bytes = development_bundle.manifest_path.read_bytes()
    parent_checksums_bytes = (
        development_bundle.directory / "checksums.json"
    ).read_bytes()
    fixed_parent_proof = {"proof_sha256": digest}
    source_provenance = {"provenance_sha256": digest}
    computation = stage_runner.StageComputation(
        stage="confirmation",
        snapshot=snapshot,
        replays=confirmation_replays,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        fixed_parent_proof=fixed_parent_proof,
        source_bundle_provenance=source_provenance,
        replay_diagnostics=stage_runner._replay_diagnostics_payload(
            stage="confirmation",
            replays=confirmation_replays,
            extra={
                "common_fork_checkpoint_sha256": forks.checkpoint_sha256,
                "online_resume_equivalence": asdict(resume),
                "frozen_post_cutoff_admitted_events": int(
                    confirmation_replays[
                        "frozen_2018"
                    ].diagnostics.admitted_events
                ),
            },
        ),
        prefix_continuity_proof=prefix_proof,
        confirmation_authorization=authorization,
        development_parent_manifest_bytes=parent_manifest_bytes,
        development_parent_checksums_bytes=parent_checksums_bytes,
    )
    integrity, gate_report, metrics = stage_runner._gate_and_metrics(computation)
    payloads, checkpoint = stage_runner._base_payloads(
        computation,
        integrity=integrity,
        gate_report=gate_report,
        metrics=metrics,
    )
    runtime = _runtime("confirmation")
    payloads["confirmation_runtime_cost_evidence.json"] = (
        stage_runner._json_bytes(runtime)
    )
    report = stage_runner._report(
        computation=computation,
        gate_report=gate_report,
        metrics=metrics,
        runtime=runtime,
        checkpoint=checkpoint,
    )
    payloads["report.json"] = stage_runner._json_bytes(report)
    manifest_fields = stage_runner._manifest_fields(
        computation=computation,
        git_identity=git_identity,
        gate_report=gate_report,
        runtime=runtime,
        checkpoint=checkpoint,
    )
    confirmation_directory = (
        tmp_path / artifacts.OUTPUT_DIRECTORY_BY_STAGE["confirmation"]
    )
    confirmation_bundle = _write_bundle(
        confirmation_directory,
        stage="confirmation",
        manifest_fields=manifest_fields,
        payloads=payloads,
    )
    evidence = experiment.DevelopmentVerificationEvidence(
        verifier_id=verifier.VERIFIER_ID,
        verifier_dependency_path=verifier.VERIFIER_IMPLEMENTATION_PATH.as_posix(),
        passed=True,
        exact_payload_names=tuple(sorted(artifacts.DEVELOPMENT_PAYLOAD_NAMES)),
        report_sha256=development_bundle.payload_sha256["report.json"],
        checkpoint_sha256=checkpoint_payload_sha256,
        gate_report_sha256=development_bundle.payload_sha256[
            "development_gate_report.json"
        ],
        semantic_evidence_sha256="sha256:" + "5" * 64,
    )
    parent = verifier._ConfirmationParent(
        authorization=authorization,
        bundle=development_bundle,
        evidence=evidence,
        manifest_bytes=parent_manifest_bytes,
        checksums_bytes=parent_checksums_bytes,
    )
    monkeypatch.setattr(
        verifier, "_direct_confirmation_parent", lambda *args, **kwargs: parent
    )
    monkeypatch.setattr(
        verifier,
        "_load_confirmation_prices_direct",
        lambda *args, **kwargs: snapshot,
    )
    monkeypatch.setattr(
        stage_runner,
        "_fixed_parent_prefix_proof",
        lambda *args, **kwargs: (fixed_parent_proof, source_provenance),
    )
    semantic = verifier._verify_semantics(
        confirmation_bundle,
        stage="confirmation",
        repo_root=tmp_path,
    )
    assert semantic.stage == "confirmation"
    sealed_suffix = artifacts.parse_canonical_table_bytes(
        payloads["confirmation_forecast__online_full.table.json"],
        schema=stage_runner.FORECAST_SCHEMA,
    )
    assert sealed_suffix.index.equals(suffix.index.rename("decision_date"))
    full_ledger = stage_runner.parse_ledger_table_bytes(
        payloads[
            "confirmation_ledger__base_5bps__online_full.table.json"
        ]
    )
    assert full_ledger.iloc[0]["fill_date"] == "2005-01-03"
    assert full_ledger.iloc[-1]["fill_date"] == "2023-12-29"


def test_development_authorization_binds_file_hash_domains_and_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_file_hash = "sha256:" + "2" * 64
    gate_file_hash = "sha256:" + "3" * 64
    report_file_hash = "sha256:" + "4" * 64
    payload_hashes = {
        name: "sha256:" + "1" * 64
        for name in artifacts.DEVELOPMENT_PAYLOAD_NAMES
    }
    payload_hashes[experiment.DEVELOPMENT_CHECKPOINT_FILENAME] = (
        checkpoint_file_hash
    )
    payload_hashes[experiment.DEVELOPMENT_GATE_REPORT_FILENAME] = gate_file_hash
    payload_hashes[experiment.DEVELOPMENT_REPORT_FILENAME] = report_file_hash
    bundle = experiment.VerifiedBundle(
        directory=(
            tmp_path / artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
        ),
        manifest_path=tmp_path / "stage_manifest.json",
        manifest={},
        payload_sha256=payload_hashes,
        checksums={},
    )
    semantic = verifier._SemanticVerification(
        stage="development",
        stage_pass=True,
        status="PASSED",
        run_id=artifacts.RUN_ID_BY_STAGE["development"],
        # Deliberately distinct internal composite hash.
        checkpoint_sha256="sha256:" + "9" * 64,
        gate_report_file_sha256=gate_file_hash,
        report_file_sha256=report_file_hash,
        semantic_evidence_sha256="sha256:" + "5" * 64,
    )
    monkeypatch.setattr(verifier, "_verify_semantics", lambda *args, **kwargs: semantic)
    monkeypatch.setattr(
        experiment,
        "verify_exact_bundle",
        lambda *args, **kwargs: bundle,
    )
    evidence = verifier.verify_development_authorization_bundle(
        bundle, experiment.DEVELOPMENT_PRICE_SPEC
    )
    assert evidence.checkpoint_sha256 == checkpoint_file_hash
    assert evidence.checkpoint_sha256 != semantic.checkpoint_sha256
    assert evidence.gate_report_sha256 == gate_file_hash
    assert evidence.report_sha256 == report_file_hash
    assert len(evidence.exact_payload_names) == 47

    bundle.payload_sha256.pop(next(iter(bundle.payload_sha256)))  # type: ignore[attr-defined]
    with pytest.raises(Error, match="47-payload"):
        verifier.verify_development_authorization_bundle(
            bundle, experiment.DEVELOPMENT_PRICE_SPEC
        )


def test_direct_confirmation_parent_validates_actual_lock_without_broad_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    manifest_path = root / stage_runner.DEVELOPMENT_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True)
    parent_manifest = {"manifest_sha256": "sha256:" + "6" * 64}
    parent_manifest_bytes = experiment.pretty_json_bytes(parent_manifest)
    manifest_path.write_bytes(parent_manifest_bytes)
    parent_checksums = {
        "stage_manifest.json": experiment.sha256_bytes(parent_manifest_bytes)
    }
    parent_checksums_bytes = experiment.pretty_json_bytes(parent_checksums)
    (manifest_path.parent / "checksums.json").write_bytes(parent_checksums_bytes)
    checkpoint_file_hash = "sha256:" + "7" * 64
    gate_file_hash = "sha256:" + "8" * 64
    report_file_hash = "sha256:" + "9" * 64
    parent = experiment.VerifiedBundle(
        directory=manifest_path.parent,
        manifest_path=manifest_path,
        manifest=parent_manifest,
        payload_sha256={
            experiment.DEVELOPMENT_CHECKPOINT_FILENAME: checkpoint_file_hash,
            experiment.DEVELOPMENT_GATE_REPORT_FILENAME: gate_file_hash,
            experiment.DEVELOPMENT_REPORT_FILENAME: report_file_hash,
        },
        checksums=parent_checksums,
    )
    evidence = experiment.DevelopmentVerificationEvidence(
        verifier_id=verifier.VERIFIER_ID,
        verifier_dependency_path=verifier.VERIFIER_IMPLEMENTATION_PATH.as_posix(),
        passed=True,
        exact_payload_names=tuple(sorted(artifacts.DEVELOPMENT_PAYLOAD_NAMES)),
        report_sha256=report_file_hash,
        checkpoint_sha256=checkpoint_file_hash,
        gate_report_sha256=gate_file_hash,
        semantic_evidence_sha256="sha256:" + "a" * 64,
    )
    dependency = {"sha256": "sha256:" + "b" * 64, "git_blob": "c" * 40}
    shared_identity = {
        "branch": experiment.EXPECTED_BRANCH,
        "commit": "d" * 40,
        "upstream": f"origin/{experiment.EXPECTED_BRANCH}",
        "upstream_remote": "origin",
        "upstream_branch": experiment.EXPECTED_BRANCH,
        "upstream_commit": "d" * 40,
        "origin_url": next(iter(experiment.EXPECTED_ORIGIN_URLS)),
        "origin_repository": experiment.EXPECTED_ORIGIN_REPOSITORY,
        "head_equals_upstream": True,
        "tracked_dependency_identity": {
            verifier.VERIFIER_IMPLEMENTATION_PATH.as_posix(): dependency
        },
        "runtime_versions": experiment._runtime_versions(),
    }
    prelock = {
        **shared_identity,
        "dirty": None,
        "cleanliness_scope": "git_metadata_and_frozen_dependency_paths_only",
        "confirmation_input_bytes_opened": False,
    }
    postlock = {
        **shared_identity,
        "dirty": False,
        "cleanliness_scope": "git_visible_index_and_worktree_after_attempt_lock",
    }
    verification = verifier._development_verification_mapping(
        evidence, prelock_identity=prelock
    )
    attempt_identity = experiment._attempt_identity_sha256(
        parent_manifest["manifest_sha256"]
    )
    lock_filename = (
        "confirmation-attempt-"
        f"{attempt_identity.removeprefix('sha256:')}.json"
    )
    lock_value = experiment._attempt_lock_value(
        repo_root=root,
        development_manifest_path=manifest_path,
        development_manifest_sha256=parent_manifest["manifest_sha256"],
        development_checkpoint_sha256=checkpoint_file_hash,
        development_gate_report_sha256=gate_file_hash,
        prelock_git_identity=prelock,
        development_verification=verification,
    )
    common = root / "git-common"
    lock_path = common / experiment.CONFIRMATION_REGISTRY_DIRECTORY / lock_filename
    lock_path.parent.mkdir(parents=True)
    lock_bytes = experiment.pretty_json_bytes(lock_value)
    lock_path.write_bytes(lock_bytes)
    authorization: dict[str, object] = {
        "authorization_schema_version": stage_runner.STAGE_EVIDENCE_SCHEMA_VERSION,
        "development_manifest_path": stage_runner.DEVELOPMENT_MANIFEST_PATH.as_posix(),
        "development_manifest_sha256": parent_manifest["manifest_sha256"],
        "development_checkpoint_sha256": checkpoint_file_hash,
        "development_gate_report_sha256": gate_file_hash,
        "attempt_identity_sha256": attempt_identity,
        "attempt_lock_filename": lock_filename,
        "attempt_lock_sha256": experiment.sha256_bytes(lock_bytes),
        "attempt_lock_parent_fsync_supported": True,
        "prelock_git_identity": prelock,
        "postlock_git_identity": postlock,
        "development_verification": verification,
        "attempt_lock_payload": lock_value,
        "confirmation_input_opened_before_lock": False,
    }
    authorization["authorization_sha256"] = stage_runner._sha256_json(
        authorization
    )
    confirmation_directory = root / "confirmation-output"
    confirmation_directory.mkdir()
    auth_bytes = stage_runner._json_bytes(authorization)
    embedded_manifest_name = "development_parent_manifest.json"
    embedded_checksums_name = "development_parent_checksums.json"
    (confirmation_directory / "confirmation_attempt_authorization.json").write_bytes(
        auth_bytes
    )
    (confirmation_directory / embedded_manifest_name).write_bytes(
        parent_manifest_bytes
    )
    (confirmation_directory / embedded_checksums_name).write_bytes(
        parent_checksums_bytes
    )
    confirmation = experiment.VerifiedBundle(
        directory=confirmation_directory,
        manifest_path=confirmation_directory / "stage_manifest.json",
        manifest={},
        payload_sha256={
            "confirmation_attempt_authorization.json": experiment.sha256_bytes(
                auth_bytes
            ),
            embedded_manifest_name: experiment.sha256_bytes(parent_manifest_bytes),
            embedded_checksums_name: experiment.sha256_bytes(parent_checksums_bytes),
        },
        checksums={},
    )
    monkeypatch.setattr(experiment, "verify_exact_bundle", lambda *args, **kwargs: parent)
    monkeypatch.setattr(
        verifier,
        "verify_development_authorization_bundle",
        lambda *args, **kwargs: evidence,
    )
    monkeypatch.setattr(experiment, "_git_common_directory", lambda root: common)
    monkeypatch.setattr(
        experiment,
        "verify_confirmation_authorization",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("broad confirmation authorization must not be called")
        ),
    )
    result = verifier._direct_confirmation_parent(root, confirmation)
    assert result.evidence == evidence
    assert result.authorization == authorization

    for target, field in (
        ("prelock_git_identity", "UNAUTHORIZED_EXTRA"),
        ("postlock_git_identity", "runtime_versions"),
    ):
        malformed = json.loads(json.dumps(authorization))
        identity_value = malformed[target]
        assert isinstance(identity_value, dict)
        if field == "UNAUTHORIZED_EXTRA":
            identity_value[field] = "rejected"
        else:
            identity_value.pop(field)
        malformed.pop("authorization_sha256")
        malformed["authorization_sha256"] = stage_runner._sha256_json(
            malformed
        )
        malformed_bytes = stage_runner._json_bytes(malformed)
        auth_path = (
            confirmation_directory
            / "confirmation_attempt_authorization.json"
        )
        auth_path.write_bytes(malformed_bytes)
        confirmation.payload_sha256[
            "confirmation_attempt_authorization.json"
        ] = experiment.sha256_bytes(malformed_bytes)
        with pytest.raises(Error, match="wrong exact inventory"):
            verifier._direct_confirmation_parent(root, confirmation)

    auth_path.write_bytes(auth_bytes)
    confirmation.payload_sha256[
        "confirmation_attempt_authorization.json"
    ] = experiment.sha256_bytes(auth_bytes)

    lock_path.write_bytes(lock_bytes + b" ")
    with pytest.raises(Error, match="lock"):
        verifier._direct_confirmation_parent(root, confirmation)


def test_confirmation_proves_parent_before_any_confirmation_price_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def stop_at_parent(*args: object, **kwargs: object) -> object:
        calls.append("parent_and_lock")
        raise Error("stop after direct parent proof")

    monkeypatch.setattr(verifier, "_direct_confirmation_parent", stop_at_parent)
    monkeypatch.setattr(
        verifier,
        "_load_confirmation_prices_direct",
        lambda *args, **kwargs: calls.append("confirmation_prices"),
    )
    with pytest.raises(Error, match="direct parent"):
        verifier._confirmation_computation(
            tmp_path,
            SimpleNamespace(),  # type: ignore[arg-type]
        )
    assert calls == ["parent_and_lock"]


def test_verify_stage_reports_rejected_as_successfully_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = SimpleNamespace(
        directory=tmp_path,
        manifest={"manifest_sha256": "sha256:" + "e" * 64},
    )
    semantic = verifier._SemanticVerification(
        stage="development",
        stage_pass=False,
        status="REJECTED",
        run_id=artifacts.RUN_ID_BY_STAGE["development"],
        checkpoint_sha256="sha256:" + "f" * 64,
        gate_report_file_sha256="sha256:" + "1" * 64,
        report_file_sha256="sha256:" + "2" * 64,
        semantic_evidence_sha256="sha256:" + "3" * 64,
    )
    monkeypatch.setattr(
        verifier._bootstrap,
        "require_active_attestation",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        experiment, "verify_exact_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(
        verifier,
        "_post_run_worktree_policy",
        lambda *args, **kwargs: "exact_generated_output_untracked_before_commit",
    )
    monkeypatch.setattr(verifier, "_verify_semantics", lambda *args, **kwargs: semantic)
    result = verifier.verify_stage("development", repo_root=tmp_path)
    assert result["verified"] is True
    assert result["stage_pass"] is False
    assert result["status"] == "REJECTED"


def test_verify_stage_requires_isolated_bootstrap_before_bundle_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        experiment,
        "verify_exact_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bundle must not be opened")
        ),
    )
    with pytest.raises(
        verifier._bootstrap.BootstrapSecurityError,
        match="isolated bootstrap",
    ):
        verifier.verify_stage("development", repo_root=tmp_path)
