from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_experiment as experiment
from agent_benchmark.contextual_expert_aggregation import (
    ContextualExpertAggregator,
    MarketSession,
)


Error = experiment.ContextualExpertAggregationExperimentError


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _raw_csv(rows: list[tuple[str, float, float, float, float, float]]) -> bytes:
    lines = [",".join(experiment.PHYSICAL_PRICE_COLUMNS)]
    lines.extend(
        f"{day},{aapl_open:g},{aapl_close:g},{aapl_adj_close:g},"
        f"{spy_adj_close:g},{qqq_adj_close:g}"
        for (
            day,
            aapl_open,
            aapl_close,
            aapl_adj_close,
            spy_adj_close,
            qqq_adj_close,
        ) in rows
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _canonical_bytes(
    rows: list[tuple[str, float, float, float, float, float]],
) -> bytes:
    frame = pd.DataFrame(
        {
            "aapl_open": [row[1] for row in rows],
            "aapl_close": [row[2] for row in rows],
            "aapl_adj_close": [row[3] for row in rows],
            "spy_adj_close": [row[4] for row in rows],
            "qqq_adj_close": [row[5] for row in rows],
        },
        index=pd.DatetimeIndex([row[0] for row in rows], name="date"),
        dtype=float,
    )
    frame["aapl_adj_open"] = (
        frame["aapl_open"]
        * frame["aapl_adj_close"]
        / frame["aapl_close"]
    )
    frame = frame.loc[:, list(experiment.CANONICAL_PRICE_COLUMNS)]
    return experiment.canonical_price_csv_bytes(frame)


def _spec(
    relative_path: str,
    rows: list[tuple[str, float, float, float, float, float]],
    *,
    stage: str = "development",
) -> experiment.AuthorizedPriceSpec:
    date_payload = "".join(f"{row[0]}\n" for row in rows).encode("ascii")
    return experiment.AuthorizedPriceSpec(
        stage=stage,
        relative_path=Path(relative_path),
        raw_sha256=experiment.sha256_bytes(_raw_csv(rows)),
        git_blob=None,
        first_session=rows[0][0],
        last_session=rows[-1][0],
        rows=len(rows),
        date_sequence_sha256=hashlib.sha256(date_payload).hexdigest(),
        canonical_sha256=experiment.sha256_bytes(_canonical_bytes(rows)),
    )


def _write_snapshot(
    root: Path,
    relative_path: str,
    rows: list[tuple[str, float, float, float, float, float]],
) -> Path:
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_raw_csv(rows))
    return destination


def _fake_git_identity() -> dict[str, Any]:
    commit = "f" * 40
    dependency_path = experiment.RUNNER_TEST_PATH.as_posix()
    return {
        "branch": experiment.EXPECTED_BRANCH,
        "commit": commit,
        "upstream": f"origin/{experiment.EXPECTED_BRANCH}",
        "upstream_remote": "origin",
        "upstream_branch": experiment.EXPECTED_BRANCH,
        "upstream_commit": commit,
        "origin_url": "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git",
        "origin_repository": experiment.EXPECTED_ORIGIN_REPOSITORY,
        "head_equals_upstream": True,
        "dirty": False,
        "cleanliness_scope": "complete_index_and_worktree_after_attempt_lock",
        "tracked_dependency_identity": {
            dependency_path: {
                "sha256": "sha256:" + "1" * 64,
                "git_blob": "a" * 40,
            }
        },
        "runtime_versions": {
            "python": "synthetic",
            "numpy": "synthetic",
            "pandas": "synthetic",
        },
    }


def _fake_prelock_git_identity() -> dict[str, Any]:
    value = _fake_git_identity()
    return {
        **value,
        "dirty": None,
        "cleanliness_scope": "git_metadata_and_frozen_dependency_paths_only",
        "confirmation_input_bytes_opened": False,
    }


def _synthetic_model_checkpoint() -> dict[str, Any]:
    model = ContextualExpertAggregator()
    for row in DEV_ROWS:
        model.process_session(
            MarketSession(
                session_date=row[0],
                aapl_open=row[1],
                aapl_close=row[2],
                aapl_adj_close=row[3],
                spy_adj_close=row[4],
                qqq_adj_close=row[5],
            )
        )
    return model.to_dict()


def _synthetic_account_state() -> dict[str, Any]:
    state = {
        "cash": 0.0,
        "shares": 100.0,
        "last_fill_date": "2018-12-31",
        "previous_requested_target": 1.0,
        "last_equity": 1000.0,
        "running_peak": 1000.0,
        "pending_decision_date": "2018-12-31",
        "pending_target_exposure": 1.0,
    }
    return {
        cost: {policy: dict(state) for policy in experiment.ACCOUNT_POLICY_NAMES}
        for cost in experiment.ACCOUNT_COST_NAMES
    }


def _cooldown_rows_from_model(model_checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "decision_date": row["session_date"],
            "contextual_virtual_signal": row["contextual_signal"],
            "weak_trend_virtual_signal": row["weak_trend_signal"],
            "union_candidate_signal": bool(
                row["contextual_signal"] or row["weak_trend_signal"]
            ),
            "canonical_union_opportunity": row[
                "canonical_union_opportunity"
            ],
        }
        for row in model_checkpoint["state"]["market_history"]
    ]


def _synthetic_checkpoint() -> dict[str, Any]:
    model_checkpoint = _synthetic_model_checkpoint()
    return {
        "contract_version": experiment.CONTRACT_VERSION,
        "checkpoint_schema_version": experiment.CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_cutoff": "2018-12-31",
        "last_observed_session": "2018-12-31",
        "model_checkpoint": model_checkpoint,
        "union_cooldown": {
            "prior_canonical_union_opportunity": False,
            "last_rows": _cooldown_rows_from_model(model_checkpoint),
        },
        "administrative_accounts": _synthetic_account_state(),
    }


def _boundary_cooldown_checkpoint(
    *, first_retained_accepted: bool
) -> dict[str, Any]:
    model = ContextualExpertAggregator()
    dates = [
        value.date().isoformat()
        for value in pd.bdate_range(end="2018-12-31", periods=23)
    ]
    for index, session_date in enumerate(dates):
        contextual = index in {0, 1}
        opportunity = index == 0 or (
            index == 1 and first_retained_accepted
        )
        model.process_session(
            MarketSession(
                session_date=session_date,
                aapl_open=100.0 + index,
                aapl_close=100.0 + index,
                aapl_adj_close=100.0 + index,
                spy_adj_close=200.0 + index,
                qqq_adj_close=300.0 + index,
                contextual_signal=contextual,
                weak_trend_signal=False,
                canonical_union_opportunity=opportunity,
            )
        )
    model_checkpoint = model.to_dict()
    checkpoint = _synthetic_checkpoint()
    checkpoint["model_checkpoint"] = model_checkpoint
    checkpoint["union_cooldown"] = {
        "prior_canonical_union_opportunity": True,
        "last_rows": _cooldown_rows_from_model(model_checkpoint),
    }
    return checkpoint


def _install_synthetic_exact_verifier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload_names: set[str],
    accept: bool = True,
) -> None:
    dependency_path = experiment.RUNNER_TEST_PATH

    def verify(
        bundle: experiment.VerifiedBundle,
        spec: experiment.AuthorizedPriceSpec,
    ) -> experiment.DevelopmentVerificationEvidence:
        assert spec is experiment.DEVELOPMENT_PRICE_SPEC
        if not accept:
            raise Error("synthetic exact verifier rejected fake development")
        semantic = experiment.sha256_bytes(
            experiment.canonical_json_bytes(
                {
                    "manifest": bundle.manifest["manifest_sha256"],
                    "payloads": bundle.payload_sha256,
                }
            )
        )
        return experiment.DevelopmentVerificationEvidence(
            verifier_id="synthetic-exact-development-verifier-v1",
            verifier_dependency_path=dependency_path.as_posix(),
            passed=True,
            exact_payload_names=tuple(sorted(payload_names)),
            report_sha256=bundle.payload_sha256[
                experiment.DEVELOPMENT_REPORT_FILENAME
            ],
            checkpoint_sha256=bundle.payload_sha256[
                experiment.DEVELOPMENT_CHECKPOINT_FILENAME
            ],
            gate_report_sha256=bundle.payload_sha256[
                experiment.DEVELOPMENT_GATE_REPORT_FILENAME
            ],
            semantic_evidence_sha256=semantic,
        )

    monkeypatch.setattr(
        experiment,
        "FROZEN_DEPENDENCY_PATHS",
        (dependency_path,),
    )
    monkeypatch.setattr(
        experiment,
        "_REGISTERED_DEVELOPMENT_VERIFIER",
        experiment.DevelopmentVerifierRegistration(
            verifier_id="synthetic-exact-development-verifier-v1",
            dependency_path=dependency_path,
            expected_payload_names=frozenset(payload_names),
            verify=verify,
        ),
    )


def _authorized_synthetic_snapshot(
    path: Path, *, spec: experiment.AuthorizedPriceSpec
) -> experiment.LoadedPriceSnapshot:
    bounded = experiment.load_bounded_price_snapshot(path, spec=spec)
    provenance = {
        **bounded.provenance,
        "tracked_input": {
            "path": spec.relative_path.as_posix(),
            "sha256": spec.raw_sha256,
            "git_blob": spec.git_blob,
        },
        "authorized_loader_lineage": {
            "source_manifest_sha256": (
                None
                if spec.source_bundle is None
                else spec.source_bundle.manifest_self_sha256
            ),
            "source_provenance_sha256": spec.source_provenance_sha256,
            "source_parent_manifest_sha256": (
                None
                if spec.source_parent_bundle is None
                else spec.source_parent_bundle.manifest_self_sha256
            ),
        },
    }
    return experiment._attest_loaded_snapshot(
        experiment.LoadedPriceSnapshot(
            spec=bounded.spec,
            frame=bounded.frame,
            raw_sha256=bounded.raw_sha256,
            canonical_csv_bytes=bounded.canonical_csv_bytes,
            provenance=provenance,
        ),
        authorized_lineage=True,
    )


def _seal_synthetic_development_authorization(
    root: Path,
    *,
    price_spec: experiment.AuthorizedPriceSpec,
    git_identity: dict[str, Any],
    gate_pass: bool = True,
    checkpoint: dict[str, Any] | None = None,
) -> experiment.SealedBundle:
    gate = {"passed": gate_pass, "checks": {"synthetic": gate_pass}}
    report = {
        "contract_version": experiment.CONTRACT_VERSION,
        "stage": "development",
        "run_id": "development-run",
        "gate_report": gate,
    }
    checkpoint_value = _synthetic_checkpoint() if checkpoint is None else checkpoint
    source = {
        "source_type": "physically_bounded_local_csv",
        "bounded_first_date": price_spec.first_session,
        "bounded_last_date": price_spec.last_session,
        "bounded_rows": price_spec.rows,
        "bounded_result_sha256": price_spec.canonical_sha256,
        "network_access": False,
        "physical_snapshot_has_later_rows": False,
        "rows_after_bound_returned": False,
        "raw_sha256": price_spec.raw_sha256,
        "session_coverage": {
            "first_session": price_spec.first_session,
            "last_session": price_spec.last_session,
            "observations": price_spec.rows,
            "date_sequence_sha256": price_spec.date_sequence_sha256,
        },
        "adjusted_open_formula": "aapl_open * aapl_adj_close / aapl_close",
        "tracked_input": {
            "path": price_spec.relative_path.as_posix(),
            "sha256": price_spec.raw_sha256,
            "git_blob": price_spec.git_blob,
        },
    }
    payloads = {
        ".gitattributes": b"* -text\n",
        "report.json": experiment.pretty_json_bytes(report),
        "input_provenance.json": experiment.pretty_json_bytes(source),
        "development_prices_through_2018.csv": _canonical_bytes(DEV_ROWS),
        "development_checkpoint_through_2018.json": experiment.pretty_json_bytes(
            checkpoint_value
        ),
        "development_gate_report.json": experiment.pretty_json_bytes(gate),
    }
    return experiment.seal_exact_bundle(
        root / "development-run",
        manifest_fields={
            "stage": "development",
            "stage_pass": True,
            "run_id": "development-run",
            "bounded_result_sha256": price_spec.canonical_sha256,
            "source_provenance": source,
            "git_identity": git_identity,
        },
        payloads=payloads,
        expected_payload_names=set(payloads),
        deadline=experiment.StageDeadline(FakeClock()),
    )


def _patch_synthetic_authorization_environment(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    git_identity: dict[str, Any],
    clean_git: Any | None = None,
    fsync_directory: Any | None = None,
) -> Path:
    real_verify = experiment.verify_exact_bundle

    def local_verify(directory: Path, **kwargs: Any) -> experiment.VerifiedBundle:
        kwargs.pop("repo_root", None)
        return real_verify(directory, **kwargs)

    monkeypatch.setattr(experiment, "verify_exact_bundle", local_verify)
    monkeypatch.setattr(
        experiment,
        "_path_scoped_prelock_git_identity",
        lambda repo_root: _fake_prelock_git_identity(),
    )
    monkeypatch.setattr(
        experiment,
        "clean_git_identity",
        clean_git
        if clean_git is not None
        else lambda *args, **kwargs: git_identity,
    )
    git_common = root / ".git"
    git_common.mkdir(exist_ok=True)
    monkeypatch.setattr(
        experiment, "_git_common_directory", lambda repo_root: git_common
    )
    monkeypatch.setattr(
        experiment,
        "_fsync_directory",
        fsync_directory
        if fsync_directory is not None
        else lambda directory: True,
    )
    return git_common


def _rewrite_bundle_manifest_field(
    directory: Path, *, field: str, value: Any
) -> None:
    manifest_path = directory / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_sha256")
    manifest[field] = value
    signed = experiment.self_hashed_manifest(manifest)
    manifest_bytes = experiment.pretty_json_bytes(signed)
    manifest_path.write_bytes(manifest_bytes)

    checksums_path = directory / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    checksums["stage_manifest.json"] = experiment.sha256_bytes(manifest_bytes)
    checksums_path.write_bytes(experiment.pretty_json_bytes(checksums))


DEV_ROWS = [
    ("2018-12-28", 10.0, 5.0, 2.0, 20.0, 30.0),
    ("2018-12-31", 12.0, 6.0, 3.0, 21.0, 31.0),
]
CONFIRM_ROWS = [
    *DEV_ROWS,
    ("2019-01-02", 14.0, 7.0, 3.5, 22.0, 32.0),
]


def test_frozen_price_contract_never_authorizes_post_2023() -> None:
    assert set(experiment.AUTHORIZED_PRICE_SPECS) == {
        "development",
        "confirmation",
    }
    assert experiment.DEVELOPMENT_PRICE_SPEC.last_session == "2018-12-31"
    assert experiment.CONFIRMATION_PRICE_SPEC.last_session == "2023-12-29"
    assert experiment.CONFIRMATION_PRICE_SPEC.relative_path.name.endswith(
        "through_2023.csv"
    )
    assert experiment.PHYSICAL_PRICE_COLUMNS == (
        "date",
        "aapl_open",
        "aapl_close",
        "aapl_adj_close",
        "spy_adj_close",
        "qqq_adj_close",
    )


def test_actual_frozen_constants_match_tracked_local_metadata_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    # The through-2023 file is intentionally not opened here: confirmation
    # bytes are unavailable to every pre-lock metadata test.
    for spec in (experiment.DEVELOPMENT_PRICE_SPEC,):
        path = repo_root / spec.relative_path
        payload = path.read_bytes()
        lines = payload.splitlines()
        assert experiment.sha256_bytes(payload) == spec.raw_sha256
        assert len(lines) == spec.rows + 1
        assert (
            lines[0]
            .decode("utf-8")
            .lstrip("\ufeff")
            .replace('"', "")
            .split(",")
            == list(experiment.PHYSICAL_PRICE_COLUMNS)
        )
        assert lines[1].split(b",", 1)[0].strip(b'"').decode() == spec.first_session
        assert lines[-1].split(b",", 1)[0].strip(b'"').decode() == spec.last_session
        tracked = experiment.tracked_file_identity(
            repo_root,
            path,
            expected_sha256=spec.raw_sha256,
            expected_git_blob=spec.git_blob,
            require_literal_local_bytes=True,
        )
        assert tracked.path == spec.relative_path.as_posix()
        assert spec.source_bundle is not None
        source_manifest = repo_root / spec.source_bundle.manifest_path
        assert (
            experiment.sha256_bytes(source_manifest.read_bytes())
            == spec.source_bundle.manifest_file_sha256
        )
        assert (
            experiment.sha256_bytes(
                (source_manifest.parent / "checksums.json").read_bytes()
            )
            == spec.source_bundle.checksums_file_sha256
        )
        assert spec.source_provenance_sha256 is not None
        assert (
            experiment.sha256_bytes(
                (
                    source_manifest.parent / spec.source_provenance_filename
                ).read_bytes()
            )
            == spec.source_provenance_sha256
        )
    confirmation = experiment.CONFIRMATION_PRICE_SPEC
    assert confirmation.rows == 6244
    assert confirmation.first_session == "1999-03-10"
    assert confirmation.last_session == "2023-12-29"
    assert confirmation.raw_sha256.startswith("sha256:")
    assert confirmation.canonical_sha256.startswith("sha256:")


def test_synthetic_loader_validates_raw_and_derived_canonical_hash(
    tmp_path: Path,
) -> None:
    path = _write_snapshot(tmp_path, "prices.csv", DEV_ROWS)
    spec = _spec("prices.csv", DEV_ROWS)

    loaded = experiment.load_bounded_price_snapshot(path, spec=spec)

    assert list(loaded.frame.columns) == list(experiment.CANONICAL_PRICE_COLUMNS)
    assert loaded.frame.index.tolist() == [
        pd.Timestamp("2018-12-28"),
        pd.Timestamp("2018-12-31"),
    ]
    assert loaded.frame["aapl_adj_open"].tolist() == [4.0, 6.0]
    assert loaded.raw_sha256 == spec.raw_sha256
    assert loaded.provenance["physical_snapshot_has_later_rows"] is False
    assert loaded.provenance["rows_after_bound_returned"] is False
    assert loaded.provenance["network_access"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.replace(b"qqq_adj_close", b"unexpected_col"),
        lambda value: value.replace(b"2018-12-31", b"2018-12-28"),
        lambda value: value.replace(b",12,6,3,", b",0,6,3,"),
    ],
)
def test_loader_fails_closed_on_schema_date_or_price_changes(
    tmp_path: Path, mutator: Any
) -> None:
    path = _write_snapshot(tmp_path, "prices.csv", DEV_ROWS)
    changed = mutator(path.read_bytes())
    path.write_bytes(changed)
    spec = replace(_spec("prices.csv", DEV_ROWS), raw_sha256=experiment.sha256_bytes(changed))

    with pytest.raises(Error):
        experiment.load_bounded_price_snapshot(path, spec=spec)


def test_loader_rejects_even_one_unexpected_physical_later_row(
    tmp_path: Path,
) -> None:
    path = _write_snapshot(tmp_path, "prices.csv", CONFIRM_ROWS)
    spec = replace(
        _spec("prices.csv", CONFIRM_ROWS),
        rows=2,
        last_session="2018-12-31",
    )

    with pytest.raises(Error, match="row count"):
        experiment.load_bounded_price_snapshot(path, spec=spec)


def test_raw_and_canonical_confirmation_prefix_is_exact(tmp_path: Path) -> None:
    development_path = _write_snapshot(tmp_path, "dev.csv", DEV_ROWS)
    confirmation_path = _write_snapshot(tmp_path, "confirmation.csv", CONFIRM_ROWS)
    development = experiment.load_bounded_price_snapshot(
        development_path, spec=_spec("dev.csv", DEV_ROWS)
    )
    confirmation = experiment.load_bounded_price_snapshot(
        confirmation_path,
        spec=_spec("confirmation.csv", CONFIRM_ROWS, stage="confirmation"),
    )

    experiment.require_snapshot_prefix(
        development,
        confirmation,
        development_path=development_path,
        confirmation_path=confirmation_path,
    )

    assert confirmation.frame.index[len(development.frame)] == pd.Timestamp(
        "2019-01-02"
    )


def test_numerically_valid_but_revised_confirmation_prefix_is_rejected(
    tmp_path: Path,
) -> None:
    revised = [
        ("2018-12-28", 10.5, 5.0, 2.0, 20.0, 30.0),
        DEV_ROWS[1],
        CONFIRM_ROWS[-1],
    ]
    development_path = _write_snapshot(tmp_path, "dev.csv", DEV_ROWS)
    confirmation_path = _write_snapshot(tmp_path, "confirmation.csv", revised)
    development = experiment.load_bounded_price_snapshot(
        development_path, spec=_spec("dev.csv", DEV_ROWS)
    )
    confirmation = experiment.load_bounded_price_snapshot(
        confirmation_path,
        spec=_spec("confirmation.csv", revised, stage="confirmation"),
    )

    with pytest.raises(Error, match="prefix"):
        experiment.require_snapshot_prefix(
            development,
            confirmation,
            development_path=development_path,
            confirmation_path=confirmation_path,
        )


def test_snapshot_integrity_rejects_constructible_fabrication(
    tmp_path: Path,
) -> None:
    spec = _spec("dev.csv", DEV_ROWS)
    path = _write_snapshot(tmp_path, spec.relative_path.as_posix(), DEV_ROWS)
    authorized = _authorized_synthetic_snapshot(path, spec=spec)
    fabricated = experiment.LoadedPriceSnapshot(
        spec=authorized.spec,
        frame=authorized.frame.copy(),
        raw_sha256=authorized.raw_sha256,
        canonical_csv_bytes=authorized.canonical_csv_bytes,
        provenance=dict(authorized.provenance),
    )

    with pytest.raises(Error, match="authorized-loader lineage attestation"):
        experiment.require_loaded_snapshot_integrity(
            fabricated, expected_spec=spec
        )


@pytest.mark.parametrize(
    ("flag", "bad_value"),
    [
        ("network_access", True),
        ("physical_snapshot_has_later_rows", True),
        ("rows_after_bound_returned", True),
    ],
)
def test_snapshot_integrity_rejects_bad_provenance_flags_even_when_attested(
    tmp_path: Path,
    flag: str,
    bad_value: bool,
) -> None:
    spec = _spec("dev.csv", DEV_ROWS)
    path = _write_snapshot(tmp_path, spec.relative_path.as_posix(), DEV_ROWS)
    authorized = _authorized_synthetic_snapshot(path, spec=spec)
    object.__setattr__(
        authorized,
        "provenance",
        {**authorized.provenance, flag: bad_value},
    )

    with pytest.raises(Error, match="provenance flags"):
        experiment.require_loaded_snapshot_integrity(
            authorized, expected_spec=spec
        )


def test_confirmation_rejects_wrong_development_before_reading_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_path = _write_snapshot(tmp_path, "dev.csv", DEV_ROWS)
    wrong_development = experiment.load_bounded_price_snapshot(
        development_path, spec=_spec("dev.csv", DEV_ROWS)
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("confirmation identity or bytes were accessed")

    monkeypatch.setattr(experiment, "tracked_file_identity", forbidden)
    monkeypatch.setattr(experiment, "verify_authorized_price_lineage", forbidden)

    with pytest.raises(Error, match="authorized-loader lineage"):
        experiment.load_authorized_prices(
            tmp_path,
            stage="confirmation",
            development_snapshot=wrong_development,
        )


def test_deadline_is_strict_at_3600_and_rejects_backward_clock() -> None:
    clock = FakeClock()
    deadline = experiment.StageDeadline(clock)
    clock.value = experiment.RUN_TIME_LIMIT_SECONDS - 0.001
    assert deadline.check("before") < experiment.RUN_TIME_LIMIT_SECONDS
    clock.value = experiment.RUN_TIME_LIMIT_SECONDS
    with pytest.raises(Error, match="deadline"):
        deadline.check("boundary")

    backwards = FakeClock(5.0)
    deadline = experiment.StageDeadline(backwards)
    backwards.value = 4.0
    with pytest.raises(Error, match="backwards"):
        deadline.elapsed()

    with pytest.raises(Error, match="at most 3600"):
        experiment.StageDeadline(
            FakeClock(),
            limit_seconds=experiment.RUN_TIME_LIMIT_SECONDS + 0.001,
        )
    shorter = experiment.StageDeadline(FakeClock(), limit_seconds=60.0)
    assert shorter.limit_seconds == 60.0


def test_directory_fsync_uses_real_host_durability_path(tmp_path: Path) -> None:
    directory = tmp_path / "durable-directory"
    directory.mkdir()
    experiment._exclusive_write(directory / "evidence.json", b"{}\n")

    assert experiment._fsync_directory(directory) is True


def test_atomic_seal_self_hashes_and_has_exact_inventory(tmp_path: Path) -> None:
    output = tmp_path / "run-1"
    deadline = experiment.StageDeadline(FakeClock())

    sealed = experiment.seal_exact_bundle(
        output,
        manifest_fields={
            "stage": "development",
            "stage_pass": False,
            "run_id": "run-1",
        },
        payloads={"report.json": b"{}\n", ".gitattributes": b"* -text\n"},
        expected_payload_names={"report.json", ".gitattributes"},
        deadline=deadline,
    )

    verified = experiment.verify_exact_bundle(
        output,
        expected_contract_version=experiment.CONTRACT_VERSION,
        expected_stage="development",
        expected_payload_names={"report.json", ".gitattributes"},
    )
    assert sealed.manifest == verified.manifest
    assert set(path.name for path in output.iterdir()) == {
        "report.json",
        ".gitattributes",
        "stage_manifest.json",
        "checksums.json",
    }
    checksums = json.loads((output / "checksums.json").read_text("utf-8"))
    assert "checksums.json" not in checksums
    assert checksums["stage_manifest.json"] == experiment.sha256_bytes(
        (output / "stage_manifest.json").read_bytes()
    )
    assert verified.manifest["contract_version"] == experiment.CONTRACT_VERSION


@pytest.mark.parametrize(
    ("manifest_fields", "expected_names"),
    [
        ({"stage_pass": False, "run_id": "run"}, {"report.json"}),
        (
            {"stage": "audit", "stage_pass": False, "run_id": "run"},
            {"report.json"},
        ),
        (
            {"stage": "development", "stage_pass": 1, "run_id": "run"},
            {"report.json"},
        ),
        (
            {"stage": "development", "stage_pass": False, "run_id": "../run"},
            {"report.json"},
        ),
        (
            {
                "contract_version": experiment.CONTRACT_VERSION,
                "stage": "development",
                "stage_pass": False,
                "run_id": "run",
            },
            {"report.json"},
        ),
        (
            {"stage": "development", "stage_pass": False, "run_id": "run"},
            {"report.json", "missing.json"},
        ),
    ],
)
def test_seal_rejects_missing_or_wrong_metadata_and_inventory(
    tmp_path: Path,
    manifest_fields: dict[str, Any],
    expected_names: set[str],
) -> None:
    with pytest.raises(Error):
        experiment.seal_exact_bundle(
            tmp_path / "invalid-run",
            manifest_fields=manifest_fields,
            payloads={"report.json": b"{}\n"},
            expected_payload_names=expected_names,
            deadline=experiment.StageDeadline(FakeClock()),
        )
    assert not (tmp_path / "invalid-run").exists()


def test_exact_bundle_verifier_rejects_extra_or_tampered_files(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    experiment.seal_exact_bundle(
        output,
        manifest_fields={
            "stage": "development",
            "stage_pass": False,
            "run_id": "run",
        },
        payloads={"report.json": b"{}\n"},
        expected_payload_names={"report.json"},
        deadline=experiment.StageDeadline(FakeClock()),
    )
    (output / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(Error, match="missing or extra"):
        experiment.verify_exact_bundle(
            output,
            expected_contract_version=experiment.CONTRACT_VERSION,
            expected_stage="development",
        )
    (output / "extra.txt").unlink()
    (output / "report.json").write_text("changed", encoding="utf-8")
    with pytest.raises(Error, match="checksum"):
        experiment.verify_exact_bundle(
            output,
            expected_contract_version=experiment.CONTRACT_VERSION,
            expected_stage="development",
        )


def test_exact_bundle_verifier_rejects_manifest_metadata_tamper(
    tmp_path: Path,
) -> None:
    output = tmp_path / "metadata-run"
    experiment.seal_exact_bundle(
        output,
        manifest_fields={
            "stage": "development",
            "stage_pass": False,
            "run_id": "metadata-run",
        },
        payloads={"report.json": b"{}\n"},
        expected_payload_names={"report.json"},
        deadline=experiment.StageDeadline(FakeClock()),
    )
    manifest_path = output / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stage"] = "confirmation"
    manifest_path.write_bytes(experiment.pretty_json_bytes(manifest))

    with pytest.raises(Error, match="self-hash"):
        experiment.verify_exact_bundle(
            output,
            expected_contract_version=experiment.CONTRACT_VERSION,
            expected_stage="development",
        )


def test_exact_bundle_verifier_intrinsically_binds_run_id_to_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "original-run"
    experiment.seal_exact_bundle(
        output,
        manifest_fields={
            "stage": "development",
            "stage_pass": False,
            "run_id": "original-run",
        },
        payloads={"report.json": b"{}\n"},
        expected_payload_names={"report.json"},
        deadline=experiment.StageDeadline(FakeClock()),
    )
    with pytest.raises(TypeError):
        experiment.verify_exact_bundle(output)  # type: ignore[call-arg]
    renamed = output.rename(tmp_path / "renamed-run")

    with pytest.raises(Error, match="directory name"):
        experiment.verify_exact_bundle(
            renamed,
            expected_contract_version=experiment.CONTRACT_VERSION,
            expected_stage="development",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract_version", 1),
        ("contract_version", "foreign-contract-v1"),
        ("stage", 1),
        ("stage", "audit"),
        ("stage", "../confirmation"),
        ("stage_pass", 1),
        ("run_id", ["manifest-run"]),
        ("run_id", "different-run"),
    ],
)
def test_exact_bundle_verifier_rejects_invalid_or_foreign_manifest_identity(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    output = tmp_path / "manifest-run"
    experiment.seal_exact_bundle(
        output,
        manifest_fields={
            "stage": "development",
            "stage_pass": False,
            "run_id": "manifest-run",
        },
        payloads={"report.json": b"{}\n"},
        expected_payload_names={"report.json"},
        deadline=experiment.StageDeadline(FakeClock()),
    )
    _rewrite_bundle_manifest_field(output, field=field, value=value)

    with pytest.raises(Error):
        experiment.verify_exact_bundle(
            output,
            expected_contract_version=experiment.CONTRACT_VERSION,
            expected_stage="development",
        )


def test_exact_bundle_verifier_rejects_symlink_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "symlink-run"
    experiment.seal_exact_bundle(
        output,
        manifest_fields={
            "stage": "development",
            "stage_pass": False,
            "run_id": "symlink-run",
        },
        payloads={"report.json": b"{}\n"},
        expected_payload_names={"report.json"},
        deadline=experiment.StageDeadline(FakeClock()),
    )
    link = output / "linked-report.json"
    link.write_bytes(b"symlink-target-placeholder")
    original_is_reparse = experiment._path_is_reparse_point

    def reports_test_link_as_symlink(path: Path) -> bool:
        return path == link or original_is_reparse(path)

    monkeypatch.setattr(
        experiment, "_path_is_reparse_point", reports_test_link_as_symlink
    )

    with pytest.raises(Error, match="regular flat files"):
        experiment.verify_exact_bundle(
            output,
            expected_contract_version=experiment.CONTRACT_VERSION,
            expected_stage="development",
        )


@pytest.mark.parametrize("redirected_path", ["final", "temporary"])
def test_seal_rejects_final_or_temporary_reparse_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    redirected_path: str,
) -> None:
    final = tmp_path / "redirect-run"
    temporary_root = tmp_path / ".redirect-run.sealing"
    target = final if redirected_path == "final" else temporary_root
    original_is_reparse = experiment._path_is_reparse_point

    def reports_redirection(path: Path) -> bool:
        return experiment._lexical_absolute(path) == target or original_is_reparse(
            path
        )

    monkeypatch.setattr(experiment, "_path_is_reparse_point", reports_redirection)

    with pytest.raises(Error, match="symlink|junction|reparse"):
        experiment.seal_exact_bundle(
            final,
            manifest_fields={
                "stage": "development",
                "stage_pass": False,
                "run_id": "redirect-run",
            },
            payloads={"report.json": b"{}\n"},
            expected_payload_names={"report.json"},
            deadline=experiment.StageDeadline(FakeClock()),
        )
    assert not final.exists()
    assert not temporary_root.exists()


@pytest.mark.parametrize("redirected_path", ["final", "temporary_root", "temporary"])
def test_seal_rechecks_redirection_after_untrusted_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    redirected_path: str,
) -> None:
    final = tmp_path / "callback-redirect-run"
    temporary_root = tmp_path / ".callback-redirect-run.sealing"
    temporary = temporary_root / final.name
    targets = {
        "final": final,
        "temporary_root": temporary_root,
        "temporary": temporary,
    }
    target = targets[redirected_path]
    original_is_reparse = experiment._path_is_reparse_point
    redirected = False

    def reports_redirection(path: Path) -> bool:
        return (
            redirected and experiment._lexical_absolute(path) == target
        ) or original_is_reparse(path)

    def redirect_after_seal() -> None:
        nonlocal redirected
        redirected = True

    monkeypatch.setattr(experiment, "_path_is_reparse_point", reports_redirection)

    with pytest.raises(Error, match="symlink|junction|reparse"):
        experiment.seal_exact_bundle(
            final,
            manifest_fields={
                "stage": "development",
                "stage_pass": False,
                "run_id": "callback-redirect-run",
            },
            payloads={"report.json": b"{}\n"},
            expected_payload_names={"report.json"},
            deadline=experiment.StageDeadline(FakeClock()),
            before_promote=redirect_after_seal,
        )
    assert not final.exists()
    if redirected_path == "final":
        assert not temporary_root.exists()
    else:
        # Refuse cleanup when the private tree itself may have been redirected.
        assert temporary_root.exists()


def test_deadline_failure_before_promotion_leaves_no_partial_bundle(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    deadline = experiment.StageDeadline(clock)
    output = tmp_path / "late-run"

    def expire() -> None:
        clock.value = experiment.RUN_TIME_LIMIT_SECONDS

    with pytest.raises(Error, match="deadline"):
        experiment.seal_exact_bundle(
            output,
            manifest_fields={
                "stage": "development",
                "stage_pass": False,
                "run_id": "late-run",
            },
            payloads={"report.json": b"{}\n"},
            expected_payload_names={"report.json"},
            deadline=deadline,
            before_promote=expire,
        )

    assert not output.exists()
    assert not (tmp_path / ".late-run.sealing").exists()


@pytest.mark.parametrize("mutation", ["payload", "extra"])
def test_callback_mutation_is_caught_by_final_reverification(
    tmp_path: Path, mutation: str
) -> None:
    output = tmp_path / "callback-run"
    private_root = tmp_path / ".callback-run.sealing"
    private = private_root / "callback-run"

    def mutate_private_directory() -> None:
        if mutation == "payload":
            (private / "report.json").write_bytes(b"changed\n")
        else:
            (private / "extra.json").write_bytes(b"{}\n")

    with pytest.raises(Error):
        experiment.seal_exact_bundle(
            output,
            manifest_fields={
                "stage": "development",
                "stage_pass": False,
                "run_id": "callback-run",
            },
            payloads={"report.json": b"{}\n"},
            expected_payload_names={"report.json"},
            deadline=experiment.StageDeadline(FakeClock()),
            before_promote=mutate_private_directory,
        )
    assert not output.exists()
    assert not private_root.exists()


def test_existing_final_or_stale_sealing_directory_is_never_reused(
    tmp_path: Path,
) -> None:
    stale = tmp_path / ".run.sealing"
    stale.mkdir()
    kwargs = {
        "manifest_fields": {
            "stage": "development",
            "stage_pass": False,
            "run_id": "run",
        },
        "payloads": {"report.json": b"{}\n"},
        "expected_payload_names": {"report.json"},
        "deadline": experiment.StageDeadline(FakeClock()),
    }
    with pytest.raises(Error, match="Stale"):
        experiment.seal_exact_bundle(tmp_path / "run", **kwargs)
    assert stale.is_dir()

    stale.rmdir()
    (tmp_path / "run").mkdir()
    with pytest.raises(Error, match="already exists"):
        experiment.seal_exact_bundle(tmp_path / "run", **kwargs)


def test_parent_link_binds_self_hash_and_exact_embedded_manifest(
    tmp_path: Path,
) -> None:
    parent = experiment.seal_exact_bundle(
        tmp_path / "parent",
        manifest_fields={
            "stage": "development",
            "stage_pass": True,
            "run_id": "parent",
        },
        payloads={"report.json": b"{}\n"},
        expected_payload_names={"report.json"},
        deadline=experiment.StageDeadline(FakeClock()),
    )
    parent_verified = experiment.verify_exact_bundle(
        parent.directory,
        expected_contract_version=experiment.CONTRACT_VERSION,
        expected_stage="development",
    )
    child = experiment.seal_exact_bundle(
        tmp_path / "child",
        manifest_fields={
            "stage": "confirmation",
            "stage_pass": False,
            "run_id": "child",
            "parent_manifest_sha256": parent.manifest["manifest_sha256"],
        },
        payloads={
            "authorized_development_manifest.json": parent.manifest_path.read_bytes()
        },
        expected_payload_names={"authorized_development_manifest.json"},
        deadline=experiment.StageDeadline(FakeClock()),
    )
    child_verified = experiment.verify_exact_bundle(
        child.directory,
        expected_contract_version=experiment.CONTRACT_VERSION,
        expected_stage="confirmation",
    )

    experiment.require_parent_link(
        child_verified,
        parent_verified,
        embedded_parent_filename="authorized_development_manifest.json",
    )

    wrong = replace(
        child_verified,
        manifest={**child_verified.manifest, "parent_manifest_sha256": "sha256:" + "0" * 64},
    )
    with pytest.raises(Error, match="parent"):
        experiment.require_parent_link(wrong, parent_verified)


def test_cooldown_checkpoint_proves_boundary_suppression_and_rejects_fake_acceptance(
) -> None:
    suppressed = _boundary_cooldown_checkpoint(first_retained_accepted=False)
    experiment._validate_development_checkpoint(suppressed)

    fake_accepted = _boundary_cooldown_checkpoint(first_retained_accepted=True)
    with pytest.raises(Error, match="one-session rule"):
        experiment._validate_development_checkpoint(fake_accepted)

    fabricated_prior = json.loads(json.dumps(suppressed))
    fabricated_prior["union_cooldown"][
        "prior_canonical_union_opportunity"
    ] = False
    with pytest.raises(Error, match="one-session rule"):
        experiment._validate_development_checkpoint(fabricated_prior)


def test_confirmation_authorization_creates_durable_one_shot_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    price_spec = _spec("dev.csv", DEV_ROWS)
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", price_spec)
    git_identity = _fake_git_identity()
    development = _seal_synthetic_development_authorization(
        tmp_path,
        price_spec=price_spec,
        git_identity=git_identity,
    )
    _install_synthetic_exact_verifier(
        monkeypatch,
        payload_names=set(development.manifest["payload_sha256"]),
    )
    real_verify = experiment.verify_exact_bundle

    def local_verify(directory: Path, **kwargs: Any) -> experiment.VerifiedBundle:
        kwargs.pop("repo_root", None)
        return real_verify(directory, **kwargs)

    monkeypatch.setattr(experiment, "verify_exact_bundle", local_verify)
    prelock_identity = _fake_prelock_git_identity()
    monkeypatch.setattr(
        experiment,
        "_path_scoped_prelock_git_identity",
        lambda root: prelock_identity,
    )
    monkeypatch.setattr(
        experiment,
        "clean_git_identity",
        lambda *args, **kwargs: git_identity,
    )
    git_common = tmp_path / ".git"
    git_common.mkdir()
    monkeypatch.setattr(experiment, "_git_common_directory", lambda root: git_common)
    fsync_calls: list[Path] = []
    real_fsync_directory = experiment._fsync_directory

    def fsync_directory(path: Path) -> bool:
        fsync_calls.append(Path(path))
        return real_fsync_directory(path)

    monkeypatch.setattr(experiment, "_fsync_directory", fsync_directory)

    authorization = experiment.authorize_confirmation_attempt(
        repo_root=tmp_path,
        development_manifest_path=development.manifest_path,
    )

    lock = authorization.attempt_lock_path
    assert lock.is_file()
    assert lock.parent == git_common / experiment.CONFIRMATION_REGISTRY_DIRECTORY
    assert authorization.attempt_identity_sha256.removeprefix("sha256:") in lock.name
    assert authorization.attempt_lock_parent_fsync_supported is True
    assert lock.parent in fsync_calls
    assert authorization.attempt_lock_sha256 == experiment.sha256_bytes(
        lock.read_bytes()
    )
    verified = experiment.verify_confirmation_authorization(
        tmp_path, authorization
    )
    assert verified.manifest["manifest_sha256"] == development.manifest[
        "manifest_sha256"
    ]
    with pytest.raises(Error, match="already been consumed"):
        experiment.authorize_confirmation_attempt(
            repo_root=tmp_path,
            development_manifest_path=development.manifest_path,
        )

    lock.write_bytes(
        lock.read_bytes().replace(b"attempt_consumed", b"attempt_tampered")
    )
    with pytest.raises(Error, match="lock"):
        experiment.verify_confirmation_authorization(tmp_path, authorization)

    with pytest.raises(TypeError):
        experiment.authorize_confirmation_attempt(
            repo_root=tmp_path,
            development_manifest_path=development.manifest_path,
            attempt_lock_path=tmp_path / "alternate-lock.json",  # type: ignore[call-arg]
            confirmation_run_id="alternate-run",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("failure", ["gate", "checkpoint"])
def test_confirmation_authorization_rejects_bad_gate_or_checkpoint_before_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    price_spec = _spec("dev.csv", DEV_ROWS)
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", price_spec)
    git_identity = _fake_git_identity()
    development = _seal_synthetic_development_authorization(
        tmp_path,
        price_spec=price_spec,
        git_identity=git_identity,
        gate_pass=failure != "gate",
    )
    _install_synthetic_exact_verifier(
        monkeypatch,
        payload_names=set(development.manifest["payload_sha256"]),
    )
    if failure == "checkpoint":
        checkpoint = development.directory / "development_checkpoint_through_2018.json"
        checkpoint.write_bytes(b"{}\n")
    real_verify = experiment.verify_exact_bundle

    def local_verify(directory: Path, **kwargs: Any) -> experiment.VerifiedBundle:
        kwargs.pop("repo_root", None)
        return real_verify(directory, **kwargs)

    monkeypatch.setattr(experiment, "verify_exact_bundle", local_verify)
    monkeypatch.setattr(
        experiment,
        "_path_scoped_prelock_git_identity",
        lambda root: _fake_prelock_git_identity(),
    )
    monkeypatch.setattr(
        experiment,
        "clean_git_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("post-lock Git proof must not run")
        ),
    )
    git_common = tmp_path / ".git"
    git_common.mkdir()
    monkeypatch.setattr(experiment, "_git_common_directory", lambda root: git_common)
    monkeypatch.setattr(experiment, "_fsync_directory", lambda path: True)

    with pytest.raises(Error):
        experiment.authorize_confirmation_attempt(
            repo_root=tmp_path,
            development_manifest_path=development.manifest_path,
        )
    assert not list(git_common.rglob("confirmation-attempt-*.json"))


def test_confirmation_authorization_never_opens_confirmation_input_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    price_spec = _spec("dev.csv", DEV_ROWS)
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", price_spec)
    git_identity = _fake_git_identity()
    development = _seal_synthetic_development_authorization(
        tmp_path,
        price_spec=price_spec,
        git_identity=git_identity,
    )
    _install_synthetic_exact_verifier(
        monkeypatch,
        payload_names=set(development.manifest["payload_sha256"]),
    )
    confirmation_path = tmp_path / "synthetic-confirmation-through-2023.csv"
    confirmation_path.write_bytes(b"must-not-be-opened-before-lock\n")
    original_read_bytes = Path.read_bytes
    confirmation_reads: list[Path] = []

    def guarded_read_bytes(path: Path) -> bytes:
        if experiment._lexical_absolute(path) == confirmation_path:
            confirmation_reads.append(path)
            raise AssertionError("confirmation bytes opened before lock")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    git_common: Path

    def clean_after_lock(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert list(git_common.rglob("confirmation-attempt-*.json"))
        return git_identity

    git_common = _patch_synthetic_authorization_environment(
        monkeypatch,
        tmp_path,
        git_identity=git_identity,
        clean_git=clean_after_lock,
    )

    authorization = experiment.authorize_confirmation_attempt(
        repo_root=tmp_path,
        development_manifest_path=development.manifest_path,
    )

    lock_value = json.loads(
        original_read_bytes(authorization.attempt_lock_path).decode("utf-8")
    )
    assert confirmation_reads == []
    assert lock_value["confirmation_input_bytes_opened_before_lock"] is False
    assert lock_value["postlock_complete_cleanliness_required"] is True


@pytest.mark.parametrize("verifier_mode", ["missing", "rejecting"])
def test_confirmation_authorization_requires_registered_accepting_exact_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verifier_mode: str,
) -> None:
    price_spec = _spec("dev.csv", DEV_ROWS)
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", price_spec)
    git_identity = _fake_git_identity()
    development = _seal_synthetic_development_authorization(
        tmp_path,
        price_spec=price_spec,
        git_identity=git_identity,
    )
    if verifier_mode == "missing":
        monkeypatch.setattr(experiment, "_REGISTERED_DEVELOPMENT_VERIFIER", None)
    else:
        _install_synthetic_exact_verifier(
            monkeypatch,
            payload_names=set(development.manifest["payload_sha256"]),
            accept=False,
        )
    git_common = _patch_synthetic_authorization_environment(
        monkeypatch,
        tmp_path,
        git_identity=git_identity,
        clean_git=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("post-lock Git proof must not run")
        ),
    )

    with pytest.raises(Error, match="not registered|rejected fake development"):
        experiment.authorize_confirmation_attempt(
            repo_root=tmp_path,
            development_manifest_path=development.manifest_path,
        )
    assert not list(git_common.rglob("confirmation-attempt-*.json"))


@pytest.mark.parametrize(
    "checkpoint_failure",
    [
        "model_restore",
        "cooldown",
        "fake_unresolved_acceptance",
        "negative_account",
        "missing_account_key",
    ],
)
def test_confirmation_authorization_rejects_semantically_invalid_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_failure: str,
) -> None:
    checkpoint = json.loads(json.dumps(_synthetic_checkpoint()))
    if checkpoint_failure == "model_restore":
        checkpoint["model_checkpoint"] = {}
    elif checkpoint_failure == "cooldown":
        checkpoint["union_cooldown"]["last_rows"][0][
            "contextual_virtual_signal"
        ] = True
    elif checkpoint_failure == "fake_unresolved_acceptance":
        last_row = checkpoint["union_cooldown"]["last_rows"][-1]
        last_row["contextual_virtual_signal"] = True
        last_row["union_candidate_signal"] = True
        last_row["canonical_union_opportunity"] = True
    elif checkpoint_failure == "negative_account":
        checkpoint["administrative_accounts"]["base_5bps"]["online"][
            "cash"
        ] = -1.0
    else:
        del checkpoint["administrative_accounts"]["base_5bps"]["online"][
            "pending_target_exposure"
        ]

    price_spec = _spec("dev.csv", DEV_ROWS)
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", price_spec)
    git_identity = _fake_git_identity()
    development = _seal_synthetic_development_authorization(
        tmp_path,
        price_spec=price_spec,
        git_identity=git_identity,
        checkpoint=checkpoint,
    )
    _install_synthetic_exact_verifier(
        monkeypatch,
        payload_names=set(development.manifest["payload_sha256"]),
    )
    git_common = _patch_synthetic_authorization_environment(
        monkeypatch,
        tmp_path,
        git_identity=git_identity,
        clean_git=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("post-lock Git proof must not run")
        ),
    )

    with pytest.raises(Error):
        experiment.authorize_confirmation_attempt(
            repo_root=tmp_path,
            development_manifest_path=development.manifest_path,
        )
    assert not list(git_common.rglob("confirmation-attempt-*.json"))


def test_confirmation_authorization_fails_closed_when_lock_parent_fsync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    price_spec = _spec("dev.csv", DEV_ROWS)
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", price_spec)
    git_identity = _fake_git_identity()
    development = _seal_synthetic_development_authorization(
        tmp_path,
        price_spec=price_spec,
        git_identity=git_identity,
    )
    _install_synthetic_exact_verifier(
        monkeypatch,
        payload_names=set(development.manifest["payload_sha256"]),
    )

    def fsync_until_lock_exists(directory: Path) -> bool:
        return not list(Path(directory).glob("confirmation-attempt-*.json"))

    git_common = _patch_synthetic_authorization_environment(
        monkeypatch,
        tmp_path,
        git_identity=git_identity,
        clean_git=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("post-lock Git proof must not run")
        ),
        fsync_directory=fsync_until_lock_exists,
    )

    with pytest.raises(Error, match="could not be fsynced"):
        experiment.authorize_confirmation_attempt(
            repo_root=tmp_path,
            development_manifest_path=development.manifest_path,
        )
    assert len(list(git_common.rglob("confirmation-attempt-*.json"))) == 1


def test_confirmation_authorization_consumes_lock_before_postlock_dirty_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    price_spec = _spec("dev.csv", DEV_ROWS)
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", price_spec)
    git_identity = _fake_git_identity()
    development = _seal_synthetic_development_authorization(
        tmp_path,
        price_spec=price_spec,
        git_identity=git_identity,
    )
    _install_synthetic_exact_verifier(
        monkeypatch,
        payload_names=set(development.manifest["payload_sha256"]),
    )
    git_common: Path

    def reject_dirty_after_lock(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert list(git_common.rglob("confirmation-attempt-*.json"))
        raise Error("synthetic post-lock dirt")

    git_common = _patch_synthetic_authorization_environment(
        monkeypatch,
        tmp_path,
        git_identity=git_identity,
        clean_git=reject_dirty_after_lock,
    )

    with pytest.raises(Error, match="post-lock dirt"):
        experiment.authorize_confirmation_attempt(
            repo_root=tmp_path,
            development_manifest_path=development.manifest_path,
        )
    assert len(list(git_common.rglob("confirmation-attempt-*.json"))) == 1


def test_confirmation_registry_rejects_reparse_lock_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_common = tmp_path / ".git"
    git_common.mkdir()
    monkeypatch.setattr(
        experiment, "_git_common_directory", lambda repo_root: git_common
    )
    redirected = git_common / experiment.CONFIRMATION_REGISTRY_DIRECTORY.parts[0]
    original_is_reparse = experiment._path_is_reparse_point

    def reports_redirected_registry(path: Path) -> bool:
        return experiment._lexical_absolute(path) == redirected or original_is_reparse(
            path
        )

    monkeypatch.setattr(
        experiment, "_path_is_reparse_point", reports_redirected_registry
    )

    with pytest.raises(Error, match="symlink|junction|reparse"):
        experiment._ensure_confirmation_registry_parent(tmp_path)
    assert not redirected.exists()


def test_confirmation_authorization_and_lock_precede_confirmation_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_spec = _spec("authorized/dev.csv", DEV_ROWS)
    confirmation_spec = replace(
        _spec(
            "authorized/confirmation.csv",
            CONFIRM_ROWS,
            stage="confirmation",
        ),
        git_blob="a" * 40,
    )
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", development_spec)
    monkeypatch.setattr(experiment, "CONFIRMATION_PRICE_SPEC", confirmation_spec)
    monkeypatch.setattr(
        experiment,
        "AUTHORIZED_PRICE_SPECS",
        {"development": development_spec, "confirmation": confirmation_spec},
    )
    development_path = _write_snapshot(
        tmp_path, development_spec.relative_path.as_posix(), DEV_ROWS
    )
    _write_snapshot(
        tmp_path, confirmation_spec.relative_path.as_posix(), CONFIRM_ROWS
    )
    development_snapshot = _authorized_synthetic_snapshot(
        development_path, spec=development_spec
    )
    lock = tmp_path / "attempt.json"
    lock.write_bytes(b"durable-lock")
    authorization = experiment.ConfirmationAuthorization(
        development_manifest_path=tmp_path / "stage_manifest.json",
        development_manifest_sha256="sha256:" + "1" * 64,
        development_checkpoint_sha256="sha256:" + "2" * 64,
        development_gate_report_sha256="sha256:" + "3" * 64,
        git_identity=_fake_git_identity(),
        prelock_git_identity=_fake_prelock_git_identity(),
        development_verification={"passed": True},
        attempt_identity_sha256="sha256:" + "4" * 64,
        attempt_lock_path=lock,
        attempt_lock_sha256=experiment.sha256_bytes(lock.read_bytes()),
        attempt_lock_parent_fsync_supported=True,
    )
    events: list[str] = []

    def verify_authorization(root: Path, value: Any) -> Any:
        assert value is authorization
        assert lock.is_file()
        events.append("authorization")
        return None

    def tracked(root: Path, path: Path, **kwargs: Any) -> experiment.TrackedFileIdentity:
        events.append("tracked")
        return experiment.TrackedFileIdentity(
            confirmation_spec.relative_path.as_posix(),
            confirmation_spec.raw_sha256,
            "a" * 40,
        )

    original_load = experiment.load_bounded_price_snapshot

    def load(path: Path, *, spec: experiment.AuthorizedPriceSpec) -> Any:
        events.append("read")
        return original_load(path, spec=spec)

    monkeypatch.setattr(
        experiment, "verify_confirmation_authorization", verify_authorization
    )
    monkeypatch.setattr(experiment, "tracked_file_identity", tracked)
    monkeypatch.setattr(
        experiment,
        "verify_authorized_price_lineage",
        lambda root, spec: events.append("lineage"),
    )
    monkeypatch.setattr(experiment, "load_bounded_price_snapshot", load)

    result = experiment.load_authorized_prices(
        tmp_path,
        stage="confirmation",
        development_snapshot=development_snapshot,
        confirmation_authorization=authorization,
    )

    assert result.spec == confirmation_spec
    assert events == ["authorization", "tracked", "lineage", "read"]


def test_confirmation_rejects_frame_byte_drift_before_authorization_or_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_spec = _spec("dev.csv", DEV_ROWS)
    monkeypatch.setattr(experiment, "DEVELOPMENT_PRICE_SPEC", development_spec)
    development_path = _write_snapshot(tmp_path, "dev.csv", DEV_ROWS)
    development = _authorized_synthetic_snapshot(
        development_path, spec=development_spec
    )
    development.frame.iloc[0, 0] = 999.0

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("authorization or confirmation bytes were accessed")

    monkeypatch.setattr(experiment, "verify_confirmation_authorization", forbidden)
    monkeypatch.setattr(experiment, "tracked_file_identity", forbidden)
    with pytest.raises(Error, match="canonical bytes"):
        experiment.load_authorized_prices(
            tmp_path,
            stage="confirmation",
            development_snapshot=development,
            confirmation_authorization=object(),
        )


def test_tracked_identity_hashes_head_blob_and_rejects_literal_input_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dependency.py"
    path.write_bytes(b"line1\r\nline2\r\n")
    committed = b"line1\nline2\n"
    blob = "a" * 40

    def fake_bytes(root: Path, *args: str) -> bytes:
        if args[0] == "ls-files":
            return b"dependency.py\n"
        if args[0] == "show":
            return committed
        raise AssertionError(args)

    def fake_text(root: Path, *args: str) -> str:
        if args[:2] == ("rev-parse", "HEAD:dependency.py"):
            return blob
        if args[:2] == ("ls-files", "--stage"):
            return f"100644 {blob} 0\tdependency.py"
        if args[0] == "hash-object":
            return blob
        raise AssertionError(args)

    monkeypatch.setattr(experiment, "_git_bytes", fake_bytes)
    monkeypatch.setattr(experiment, "_git_text", fake_text)

    identity = experiment.tracked_file_identity(tmp_path, path)
    assert identity.sha256 == experiment.sha256_bytes(committed)
    assert identity.git_blob == blob
    with pytest.raises(Error, match="byte-for-byte"):
        experiment.tracked_file_identity(
            tmp_path, path, require_literal_local_bytes=True
        )


def test_git_identity_rejects_reparse_repository_root_before_git_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = experiment._lexical_absolute(tmp_path)
    original_is_reparse = experiment._path_is_reparse_point

    def reports_redirected_root(path: Path) -> bool:
        return experiment._lexical_absolute(path) == repository or original_is_reparse(
            path
        )

    def forbidden_git(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("Git was invoked through a redirected root")

    monkeypatch.setattr(experiment, "_path_is_reparse_point", reports_redirected_root)
    monkeypatch.setattr(experiment, "_git_text", forbidden_git)

    with pytest.raises(Error, match="symlink|junction|reparse"):
        experiment.clean_git_identity(tmp_path)


def test_prelock_git_identity_is_path_scoped_and_declares_no_confirmation_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = {
        key: value
        for key, value in _fake_prelock_git_identity().items()
        if key
        not in {
            "dirty",
            "cleanliness_scope",
            "confirmation_input_bytes_opened",
            "tracked_dependency_identity",
            "runtime_versions",
        }
    }
    dependencies = _fake_git_identity()["tracked_dependency_identity"]
    versions = _fake_git_identity()["runtime_versions"]
    monkeypatch.setattr(
        experiment, "_git_metadata_identity", lambda repo_root: metadata
    )
    monkeypatch.setattr(
        experiment,
        "_tracked_dependency_identity",
        lambda repo_root: dependencies,
    )
    monkeypatch.setattr(experiment, "_runtime_versions", lambda: versions)

    identity = experiment._path_scoped_prelock_git_identity(tmp_path)

    assert identity["dirty"] is None
    assert (
        identity["cleanliness_scope"]
        == "git_metadata_and_frozen_dependency_paths_only"
    )
    assert identity["confirmation_input_bytes_opened"] is False
    assert identity["tracked_dependency_identity"] == dependencies


def test_frozen_dependencies_are_nonempty_and_have_no_public_empty_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(experiment, "FROZEN_DEPENDENCY_PATHS", ())
    with pytest.raises(Error, match="nonempty"):
        experiment._frozen_dependency_paths()
    with pytest.raises(TypeError):
        experiment.clean_git_identity(  # type: ignore[call-arg]
            tmp_path, dependency_paths=()
        )


def test_git_metadata_requires_exact_origin_current_branch_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "b" * 40

    def fake_text(root: Path, *args: str) -> str:
        values = {
            ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "test-branch",
            ("rev-parse", "HEAD"): commit,
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "local/test-branch",
            ("rev-parse", "@{upstream}"): commit,
            ("remote", "get-url", "origin"): "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git",
        }
        return values[tuple(args)]

    monkeypatch.setattr(experiment, "_git_text", fake_text)

    with pytest.raises(Error, match="origin/current-branch upstream"):
        experiment._git_metadata_identity(
            tmp_path, expected_branch="test-branch"
        )


def test_git_metadata_rejects_local_attacker_mirror_as_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "b" * 40

    def fake_text(root: Path, *args: str) -> str:
        values = {
            ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "test-branch",
            ("rev-parse", "HEAD"): commit,
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/test-branch",
            ("rev-parse", "@{upstream}"): commit,
            ("remote", "get-url", "origin"): str(
                tmp_path / "attacker-mirror.git"
            ),
        }
        return values[tuple(args)]

    monkeypatch.setattr(experiment, "_git_text", fake_text)

    with pytest.raises(Error, match="frozen origin repository"):
        experiment._git_metadata_identity(
            tmp_path, expected_branch="test-branch"
        )


def test_clean_git_identity_requires_upstream_and_hashes_code_and_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "b" * 40
    requested: list[str] = []

    def fake_text(root: Path, *args: str) -> str:
        key = tuple(args)
        values = {
            ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "test-branch",
            ("rev-parse", "HEAD"): commit,
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/test-branch",
            ("rev-parse", "@{upstream}"): commit,
            ("remote", "get-url", "origin"): "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git",
        }
        return values[key]

    def fake_identity(root: Path, path: Path, **_: Any) -> experiment.TrackedFileIdentity:
        requested.append(path.name)
        return experiment.TrackedFileIdentity(
            path.name, "sha256:" + "1" * 64, "c" * 40
        )

    monkeypatch.setattr(experiment, "_git_text", fake_text)
    monkeypatch.setattr(experiment, "tracked_file_identity", fake_identity)
    dependencies = (Path("model.py"), Path("test_model.py"))
    monkeypatch.setattr(experiment, "FROZEN_DEPENDENCY_PATHS", dependencies)

    identity = experiment.clean_git_identity(
        tmp_path,
        expected_branch="test-branch",
    )

    assert identity["head_equals_upstream"] is True
    assert requested == ["model.py", "test_model.py"]
    assert set(identity["tracked_dependency_identity"]) == {
        "model.py",
        "test_model.py",
    }

    original = experiment._git_text

    def unpushed(root: Path, *args: str) -> str:
        if tuple(args) == ("rev-parse", "@{upstream}"):
            return "d" * 40
        return original(root, *args)

    monkeypatch.setattr(experiment, "_git_text", unpushed)
    with pytest.raises(Error, match="upstream"):
        experiment.clean_git_identity(
            tmp_path,
            expected_branch="test-branch",
        )


@pytest.mark.parametrize("failure", ["dirty", "wrong_branch", "missing_upstream"])
def test_clean_git_identity_rejects_dirty_wrong_branch_or_missing_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    commit = "e" * 40

    def fake_text(root: Path, *args: str) -> str:
        key = tuple(args)
        if failure == "missing_upstream" and "@{upstream}" in key:
            raise subprocess.CalledProcessError(1, ["git", *args])
        values = {
            ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
            ("status", "--porcelain=v1", "--untracked-files=all"): (
                " M dirty.py" if failure == "dirty" else ""
            ),
            ("symbolic-ref", "--quiet", "--short", "HEAD"): (
                "wrong-branch"
                if failure == "wrong_branch"
                else experiment.EXPECTED_BRANCH
            ),
            ("rev-parse", "HEAD"): commit,
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): f"origin/{experiment.EXPECTED_BRANCH}",
            ("rev-parse", "@{upstream}"): commit,
            ("remote", "get-url", "origin"): "https://github.com/AntonioDomenech/LLM-memory-trading-agent.git",
        }
        return values[key]

    monkeypatch.setattr(experiment, "_git_text", fake_text)
    monkeypatch.setattr(
        experiment, "FROZEN_DEPENDENCY_PATHS", (Path("model.py"),)
    )
    monkeypatch.setattr(
        experiment,
        "tracked_file_identity",
        lambda root, path: experiment.TrackedFileIdentity(
            "model.py", "sha256:" + "1" * 64, "a" * 40
        ),
    )
    with pytest.raises(Error):
        experiment.clean_git_identity(tmp_path)


def test_dependency_continuity_binds_runtime_code_and_test_hashes() -> None:
    current = {
        **_fake_git_identity(),
        "tracked_dependency_identity": {
            "model.py": {"sha256": "sha256:" + "1" * 64, "git_blob": "a" * 40},
            "test_model.py": {"sha256": "sha256:" + "2" * 64, "git_blob": "b" * 40},
        },
        "runtime_versions": {"python": "test", "pandas": "test", "numpy": "test"},
    }
    parent = {"git_identity": current}
    experiment.require_dependency_continuity(current, parent)

    after_artifact_commit = {
        **current,
        "commit": "d" * 40,
        "upstream_commit": "d" * 40,
    }
    experiment.require_dependency_continuity(after_artifact_commit, parent)

    changed = {
        **current,
        "tracked_dependency_identity": {
            **current["tracked_dependency_identity"],
            "test_model.py": {"sha256": "sha256:" + "3" * 64, "git_blob": "c" * 40},
        },
    }
    with pytest.raises(Error, match="changed"):
        experiment.require_dependency_continuity(changed, parent)

    attacker_parent = {
        "git_identity": {
            **current,
            "origin_url": "C:/local/attacker-mirror.git",
        }
    }
    with pytest.raises(Error, match="Git origin"):
        experiment.require_dependency_continuity(current, attacker_parent)


def test_self_hashed_manifest_rejects_caller_supplied_hash() -> None:
    value = experiment.self_hashed_manifest({"stage": "development"})
    unsigned = dict(value)
    recorded = unsigned.pop("manifest_sha256")
    assert recorded == experiment.sha256_bytes(
        experiment.canonical_json_bytes(unsigned)
    )
    with pytest.raises(Error, match="predeclare"):
        experiment.self_hashed_manifest(value)
