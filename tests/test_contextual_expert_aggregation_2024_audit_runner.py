from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pandas as pd

from agent_benchmark import contextual_expert_aggregation_2024_audit_artifacts as artifacts
from agent_benchmark import contextual_expert_aggregation_2024_audit_bootstrap as bootstrap
from agent_benchmark import contextual_expert_aggregation_2024_audit_runner as runner
from agent_benchmark import contextual_expert_aggregation_2024_audit_verifier as verifier


def _sequence_clock(*values: float):
    iterator = iter(values)
    return lambda: next(iterator)


def _synthetic_payloads(token: str = "same") -> dict[str, bytes]:
    return {
        filename: (
            artifacts.GIT_ATTRIBUTES_BYTES
            if filename == ".gitattributes"
            else f"{token}:{filename}\n".encode("ascii")
        )
        for filename in artifacts.PAYLOAD_FILE_ORDER
    }


def _causal_proof_fixture() -> tuple[SimpleNamespace, SimpleNamespace, pd.DataFrame]:
    index = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04"], name="date"
    )
    event = {
        "signal_date": "2024-01-02",
        "maturity_date": "2024-01-04",
        "signal_market_state": "fast_on_slow_on",
        "entry_adjusted_open": 101.0,
        "exit_adjusted_open": 102.0,
        "net_cash_log_edge_10bps": -0.01,
        "reward_range": 0.01,
        "admitted": True,
        "advice__always_long": False,
        "advice__union_cash": True,
        "advice__contextual_only": True,
        "advice__weak_trend_only": False,
        "reward__always_long": 0.0,
        "reward__union_cash": -0.01,
        "reward__contextual_only": -0.01,
        "reward__weak_trend_only": 0.0,
    }
    matured = pd.DataFrame([event])
    forecast = pd.DataFrame(index=index)
    forecast["matured_on_close_count"] = [0, 0, 1]
    forecast["matured_admitted"] = [False, False, True]
    forecast["matured_lesson_count"] = [0, 0, 1]
    forecast["admitted_lesson_count"] = [0, 0, 1]
    forecast["pending_lesson_count"] = [1, 1, 0]
    for event_name, forecast_name in {
        "signal_date": "matured_signal_date",
        "signal_market_state": "matured_signal_market_state",
        "entry_adjusted_open": "matured_entry_adjusted_open",
        "exit_adjusted_open": "matured_exit_adjusted_open",
        "net_cash_log_edge_10bps": "matured_net_cash_log_edge_10bps",
        "reward_range": "matured_reward_range",
        "admitted": "matured_admitted",
        **{
            f"advice__{name}": f"matured_advice__{name}"
            for name in (
                "always_long",
                "union_cash",
                "contextual_only",
                "weak_trend_only",
            )
        },
        **{
            f"reward__{name}": f"matured_reward__{name}"
            for name in (
                "always_long",
                "union_cash",
                "contextual_only",
                "weak_trend_only",
            )
        },
    }.items():
        default = False if isinstance(event[event_name], bool) else ""
        if isinstance(event[event_name], float):
            default = 0.0
        forecast[forecast_name] = [default, default, event[event_name]]
    opportunity = pd.DataFrame(
        {"unfiltered_union_signal": [True, False, False]}, index=index
    )
    parent = SimpleNamespace(
        pending_lessons=(),
        model_payload={
            "state": {
                "processed_session_count": 0,
                "matured_lesson_count": 0,
                "admitted_lesson_count": 0,
            }
        },
    )
    scenario = SimpleNamespace(
        opportunity_frame=opportunity,
        forecast=forecast,
        matured_lessons=matured,
        checkpoint=SimpleNamespace(
            pending_lessons=(),
            model_payload={
                "state": {
                    "processed_session_count": 3,
                    "matured_lesson_count": 1,
                    "admitted_lesson_count": 1,
                }
            },
        ),
        diagnostics=SimpleNamespace(
            processed_rows=3,
            accepted_opportunities=1,
            matured_events=1,
            admitted_events=1,
            ending_pending_lessons=0,
        ),
    )
    prefix = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    return scenario, parent, prefix


def test_lesson_conservation_rejects_missing_or_duplicate_lesson() -> None:
    scenario, parent, _ = _causal_proof_fixture()
    assert runner._lesson_conservation(scenario, parent_checkpoint=parent) is True

    missing = copy.deepcopy(scenario)
    missing.matured_lessons = missing.matured_lessons.iloc[0:0]
    assert runner._lesson_conservation(missing, parent_checkpoint=parent) is False

    duplicate = copy.deepcopy(scenario)
    duplicate.checkpoint.pending_lessons = ({"signal_date": "2024-01-02"},)
    duplicate.diagnostics.ending_pending_lessons = 1
    assert runner._lesson_conservation(duplicate, parent_checkpoint=parent) is False


def test_causal_update_order_rejects_wrong_schedule_and_cumulative_count() -> None:
    scenario, parent, prefix = _causal_proof_fixture()
    assert (
        runner._causal_update_order(
            scenario,
            parent_checkpoint=parent,
            historical_prefix=prefix,
        )
        is True
    )

    wrong_schedule = copy.deepcopy(scenario)
    wrong_schedule.matured_lessons.loc[0, "maturity_date"] = "2024-01-03"
    assert (
        runner._causal_update_order(
            wrong_schedule,
            parent_checkpoint=parent,
            historical_prefix=prefix,
        )
        is False
    )

    wrong_count = copy.deepcopy(scenario)
    wrong_count.forecast.loc[pd.Timestamp("2024-01-03"), "pending_lesson_count"] = 0
    assert (
        runner._causal_update_order(
            wrong_count,
            parent_checkpoint=parent,
            historical_prefix=prefix,
        )
        is False
    )


def test_runtime_clock_and_runtime_evidence_enforce_order_and_deadlines() -> None:
    clock = runner.RuntimeClock(
        start=100.0,
        clock=_sequence_clock(100.25, 100.5, 100.75),
    )

    assert clock.sample("preseal", limit=1_800.0) == pytest.approx(0.25)
    provisional = runner._runtime_payload(clock=clock, provisional=True)
    assert provisional["samples"] == {
        "preseal": pytest.approx(0.25),
        "post_private_verify": None,
        "prepromotion": None,
    }
    assert provisional["stage_internal_deadline_pass"] is False

    assert clock.sample("post_private_verify", limit=1_800.0) == pytest.approx(0.5)
    assert clock.sample("prepromotion", limit=1_795.0) == pytest.approx(0.75)
    final = runner._runtime_payload(clock=clock, provisional=False)
    assert final["samples"] == {
        "preseal": pytest.approx(0.25),
        "post_private_verify": pytest.approx(0.5),
        "prepromotion": pytest.approx(0.75),
    }
    assert final["stage_internal_deadline_pass"] is True

    with pytest.raises(runner.ContextualExpertAggregation2024AuditError):
        clock.sample("preseal", limit=1_800.0)

    at_deadline = runner.RuntimeClock(start=10.0, clock=lambda: 11.0)
    with pytest.raises(
        runner.ContextualExpertAggregation2024AuditError,
        match="deadline",
    ):
        at_deadline.require(limit=1.0)

    before_start = runner.RuntimeClock(start=10.0, clock=lambda: 9.9)
    with pytest.raises(
        runner.ContextualExpertAggregation2024AuditError,
        match="backwards",
    ):
        before_start.elapsed()


def test_final_runtime_payload_rejects_out_of_order_named_samples() -> None:
    clock = runner.RuntimeClock(start=0.0, clock=lambda: 0.0)
    clock.samples = {
        "preseal": 3.0,
        "post_private_verify": 2.0,
        "prepromotion": 4.0,
    }

    with pytest.raises(
        runner.ContextualExpertAggregation2024AuditError,
        match="strictly increasing",
    ):
        runner._runtime_payload(clock=clock, provisional=False)


def test_runtime_clock_rejects_a_sample_earlier_than_the_previous_sample() -> None:
    clock = runner.RuntimeClock(
        start=10.0,
        clock=_sequence_clock(12.0, 11.0),
    )
    clock.sample("preseal", limit=1_800.0)

    with pytest.raises(
        runner.ContextualExpertAggregation2024AuditError,
        match="strictly increasing",
    ):
        clock.sample("post_private_verify", limit=1_800.0)


def test_payload_builder_emits_the_exact_frozen_41_file_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def object_payload(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args, kwargs
        return {"synthetic": True}

    for name in (
        "_build_checkpoint",
        "_source_provenance_payload",
        "_known_result_isolation_payload",
        "_integrity_payload",
        "_report_payload",
        "_prefix_payload",
        "_pending_payload",
        "_replay_payload",
        "_metrics_payload",
        "_gate_payload",
        "_episodes_payload",
        "_xor_payload",
    ):
        monkeypatch.setattr(runner, name, object_payload)

    serialized_tables: list[str] = []

    def table_payload(filename: str, frame: object) -> bytes:
        del frame
        serialized_tables.append(filename)
        return f"synthetic-table:{filename}\n".encode("ascii")

    monkeypatch.setattr(
        runner._artifacts,
        "canonical_registered_table_bytes",
        table_payload,
    )

    parent = SimpleNamespace(
        manifest_bytes=b"synthetic-parent-manifest\n",
        checksums_bytes=b"synthetic-parent-checksums\n",
        lock_payload=lambda: {"synthetic_parent": True},
    )
    bounded = SimpleNamespace(
        receipt_bytes=b"synthetic-receipt\n",
        input_bytes=b"synthetic-bounded-input\n",
    )
    lock = SimpleNamespace(bytes_value=b"synthetic-lock\n")
    computation = SimpleNamespace(
        fixed_features=object(),
        fixed_comparator_forecast=object(),
        state_weight_diagnostics=object(),
        forecasts={scenario: object() for scenario in artifacts.SCENARIO_ORDER},
        matured_lessons={
            scenario: object() for scenario in artifacts.SCENARIO_ORDER
        },
        ledgers={
            cost: {policy: object() for policy in artifacts.POLICY_ORDER}
            for cost in artifacts.COST_ORDER
        },
    )

    payloads = runner._build_reproducible_payloads(
        parent=parent,
        bounded=bounded,
        lock=lock,
        git_identity={},
        computation=computation,
        checks={},
        evaluation={},
        runtime_payload={"synthetic": True},
    )

    assert tuple(payloads) == artifacts.PAYLOAD_FILE_ORDER
    assert len(payloads) == 41
    assert set(payloads) == artifacts.PAYLOAD_FILENAMES
    assert payloads[".gitattributes"] == artifacts.GIT_ATTRIBUTES_BYTES
    assert set(serialized_tables) == {
        filename
        for filename in artifacts.PAYLOAD_FILE_ORDER
        if filename.endswith(".table.json")
    }


def test_private_runtime_rewrite_allows_only_runtime_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisional = _synthetic_payloads("provisional")
    final = dict(provisional)
    final["audit_runtime_cost_evidence.json"] = b"final-runtime\n"
    final["report.json"] = b"final-report\n"
    metadata = SimpleNamespace(
        manifest_bytes=b"final-manifest\n",
        checksums_bytes=b"final-checksums\n",
    )
    replacements: list[tuple[str, bytes]] = []
    asserted: list[tuple[object, object]] = []
    monkeypatch.setattr(
        runner,
        "_atomic_replace_file",
        lambda path, payload: replacements.append((path.name, payload)),
    )
    monkeypatch.setattr(
        runner,
        "_assert_exact_bundle_bytes",
        lambda directory, *, payloads, metadata: asserted.append(
            (payloads, metadata)
        ),
    )

    runner._rewrite_private_runtime(
        tmp_path,
        provisional_payloads=provisional,
        final_payloads=final,
        metadata=metadata,
    )

    assert replacements == [
        ("audit_runtime_cost_evidence.json", b"final-runtime\n"),
        ("report.json", b"final-report\n"),
        ("stage_manifest.json", b"final-manifest\n"),
        ("checksums.json", b"final-checksums\n"),
    ]
    assert asserted == [(final, metadata)]

    replacements.clear()
    unauthorized = dict(final)
    unauthorized["audit_metrics.json"] = b"changed-economic-evidence\n"
    with pytest.raises(
        runner.ContextualExpertAggregation2024AuditError,
        match="unauthorized economic payload",
    ):
        runner._rewrite_private_runtime(
            tmp_path,
            provisional_payloads=provisional,
            final_payloads=unauthorized,
            metadata=metadata,
        )
    assert replacements == []


def test_run_audit_verifies_provisional_and_final_runtime_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_root = tmp_path / artifacts.CONTROL_ROOT
    control_root.mkdir(parents=True)
    digest_a = "sha256:" + "a" * 64
    digest_b = "sha256:" + "b" * 64
    digest_c = "sha256:" + "c" * 64
    git_identity = {"commit": "c" * 40}
    parent = SimpleNamespace(
        checkpoint=object(),
        lock_payload=lambda: {"synthetic_parent": True},
    )
    lock = SimpleNamespace(raw_sha256=digest_c)
    bounded = object()
    computation = object()
    evaluation = {
        "gate_report": {
            "status": "REJECTED_2024",
            "stage_pass": False,
            "learning_classification": "unexercised",
            "learning_candidate_for_2025_shadow": False,
        }
    }
    provisional_metadata = SimpleNamespace(
        manifest={"manifest_sha256": digest_a},
        manifest_bytes=b"provisional-manifest\n",
        checksums_bytes=b"provisional-checksums\n",
    )
    final_metadata = SimpleNamespace(
        manifest={"manifest_sha256": digest_b},
        manifest_bytes=b"final-manifest\n",
        checksums_bytes=b"final-checksums\n",
    )
    metadata = iter((provisional_metadata, final_metadata))

    monkeypatch.setattr(runner._control, "harden_runtime_paths", lambda root: None)
    monkeypatch.setattr(
        runner._control,
        "attest_prelock_git",
        lambda root: dict(git_identity),
    )
    monkeypatch.setattr(
        runner._control,
        "require_pristine_control_root",
        lambda root: {"synthetic": "pristine"},
    )
    monkeypatch.setattr(
        runner._control,
        "create_attempt_lock",
        lambda **kwargs: lock,
    )
    monkeypatch.setattr(
        runner._control,
        "reattest_after_lock",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner._control,
        "read_postlock_tracked_file",
        lambda *args, relative_path, **kwargs: (
            b"synthetic-receipt\n"
            if relative_path == runner._input.RECEIPT_PATH
            else b"synthetic-input\n"
        ),
    )
    monkeypatch.setattr(
        runner._parent,
        "verify_parent_bundle",
        lambda root: parent,
    )
    monkeypatch.setattr(
        runner._input,
        "parse_sanitized_receipt_bytes",
        lambda payload: {"synthetic": True},
    )
    monkeypatch.setattr(
        runner._input,
        "load_postlock_bounded_snapshot",
        lambda **kwargs: bounded,
    )
    monkeypatch.setattr(
        runner._computation,
        "compute_2024_continuation",
        lambda **kwargs: computation,
    )
    monkeypatch.setattr(runner, "_integrity_checks", lambda **kwargs: {})
    monkeypatch.setattr(runner, "_evaluate", lambda *args, **kwargs: evaluation)
    monkeypatch.setattr(
        runner,
        "_build_reproducible_payloads",
        lambda **kwargs: _synthetic_payloads(
            "final"
            if kwargs["runtime_payload"]["stage_internal_deadline_pass"]
            else "provisional"
        ),
    )
    monkeypatch.setattr(runner, "_manifest_fields", lambda **kwargs: {})
    monkeypatch.setattr(
        runner._artifacts,
        "build_bundle_metadata",
        lambda *args, **kwargs: next(metadata),
    )
    monkeypatch.setattr(
        runner._artifacts,
        "write_private_bundle",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner, "_assert_control_inventory", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_assert_exact_bundle_bytes", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_rewrite_private_runtime", lambda *a, **k: None)
    verifier_calls: list[tuple[Path, str, bool]] = []

    def verify_private(
        directory: Path,
        *,
        repo_root: Path,
        expected_manifest_sha256: str,
        allow_provisional_runtime: bool,
    ) -> None:
        assert repo_root == tmp_path
        verifier_calls.append(
            (
                directory,
                expected_manifest_sha256,
                allow_provisional_runtime,
            )
        )

    monkeypatch.setattr(verifier, "verify_private_audit_bundle", verify_private)
    monkeypatch.setattr(
        runner._artifacts,
        "build_success_marker",
        lambda fields: {**fields, "marker_sha256": digest_c},
    )
    monkeypatch.setattr(
        runner._artifacts,
        "success_marker_bytes",
        lambda marker: b"synthetic-success-marker\n",
    )
    monkeypatch.setattr(runner, "_exclusive_marker", lambda *a, **k: None)
    monkeypatch.setattr(
        runner._experiment,
        "_fsync_directory",
        lambda path: None,
    )
    replacements: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        runner.os,
        "replace",
        lambda source, target: replacements.append((Path(source), Path(target))),
    )
    monkeypatch.setattr(
        runner,
        "_failure_transition",
        lambda *args, **kwargs: pytest.fail("success path entered failure transition"),
    )
    clock = runner.RuntimeClock(
        start=0.0,
        clock=_sequence_clock(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    )

    result = runner._run_audit(
        tmp_path,
        bootstrap_attestation={"synthetic": True},
        clock=clock,
    )

    private = tmp_path / artifacts.PRIVATE_DIRECTORY
    assert verifier_calls == [
        (private, digest_a, True),
        (private, digest_b, False),
    ]
    assert replacements == [
        (private, tmp_path / artifacts.OUTPUT_DIRECTORY),
        (
            tmp_path / artifacts.PENDING_SUCCESS_MARKER_PATH,
            tmp_path / artifacts.SUCCESS_MARKER_PATH,
        ),
    ]
    assert result["manifest_sha256"] == digest_b
    assert result["success_marker_sha256"] == digest_c
    assert result["stage_runtime_samples"]["live_prepromotion_guard"] == 0.6


def test_pending_marker_is_exclusive_fsynced_read_back_and_parsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending = tmp_path / "pending-marker"
    payload = b'{"synthetic":true}\n'
    fsynced: list[Path] = []
    parsed: list[bytes] = []

    def exclusive_write(path: Path, content: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()

    monkeypatch.setattr(runner._experiment, "_exclusive_write", exclusive_write)
    monkeypatch.setattr(
        runner._experiment,
        "_fsync_directory",
        lambda path: fsynced.append(path),
    )
    monkeypatch.setattr(
        runner._artifacts,
        "parse_success_marker_bytes",
        lambda content: parsed.append(content),
    )

    runner._exclusive_marker(pending, payload)

    assert pending.read_bytes() == payload
    assert fsynced == [tmp_path]
    assert parsed == [payload]
    with pytest.raises(FileExistsError):
        runner._exclusive_marker(pending, payload)


def test_failure_transition_does_nothing_before_attempt_lock(
    tmp_path: Path,
) -> None:
    pending = tmp_path / artifacts.PENDING_SUCCESS_MARKER_PATH
    private = tmp_path / artifacts.PRIVATE_DIRECTORY
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"preexisting-marker")
    private.mkdir(parents=True)
    (private / "evidence").write_bytes(b"preexisting-evidence")

    runner._failure_transition(
        tmp_path,
        attempt_locked=False,
        marker_committed=False,
    )

    assert pending.read_bytes() == b"preexisting-marker"
    assert (private / "evidence").read_bytes() == b"preexisting-evidence"
    assert not (tmp_path / artifacts.FAILED_DIRECTORY).exists()


@pytest.mark.parametrize(
    ("source_relative", "marker_relative"),
    [
        (artifacts.PRIVATE_DIRECTORY, artifacts.PENDING_SUCCESS_MARKER_PATH),
        (artifacts.OUTPUT_DIRECTORY, artifacts.SUCCESS_MARKER_PATH),
    ],
)
def test_locked_failure_removes_uncommitted_marker_and_preserves_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_relative: Path,
    marker_relative: Path,
) -> None:
    source = tmp_path / source_relative
    marker = tmp_path / marker_relative
    failed = tmp_path / artifacts.FAILED_DIRECTORY
    source.mkdir(parents=True)
    (source / "evidence").write_bytes(b"sealed-evidence")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(b"uncommitted-marker")
    fsynced: list[Path] = []
    monkeypatch.setattr(
        runner._experiment,
        "_fsync_directory",
        lambda path: fsynced.append(path),
    )
    monkeypatch.setattr(runner._control, "harden_runtime_paths", lambda root: {})
    terminal_inventory: list[tuple[Path, list[str]]] = []
    monkeypatch.setattr(
        runner,
        "_assert_control_inventory",
        lambda root, *, run_entries, marker=None: terminal_inventory.append(
            (Path(root), list(run_entries))
        ),
    )

    runner._failure_transition(
        tmp_path,
        attempt_locked=True,
        marker_committed=False,
    )

    assert not marker.exists()
    assert not source.exists()
    assert (failed / "evidence").read_bytes() == b"sealed-evidence"
    assert marker.parent in fsynced
    assert failed.parent in fsynced
    assert terminal_inventory == [(tmp_path, [failed.name])]


def test_committed_success_is_never_cleaned_up(
    tmp_path: Path,
) -> None:
    success = tmp_path / artifacts.SUCCESS_MARKER_PATH
    final = tmp_path / artifacts.OUTPUT_DIRECTORY
    success.parent.mkdir(parents=True)
    success.write_bytes(b"committed-marker")
    final.mkdir(parents=True)
    (final / "evidence").write_bytes(b"committed-evidence")

    runner._failure_transition(
        tmp_path,
        attempt_locked=True,
        marker_committed=True,
    )

    assert success.read_bytes() == b"committed-marker"
    assert (final / "evidence").read_bytes() == b"committed-evidence"
    assert not (tmp_path / artifacts.FAILED_DIRECTORY).exists()


def test_run_stage_rejects_wrong_stage_before_bootstrap() -> None:
    with pytest.raises(
        runner.ContextualExpertAggregation2024AuditError,
        match="only audit_2024",
    ):
        runner.run_stage("not-audit-2024")


def test_run_stage_rejects_missing_active_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(
        sys,
        bootstrap._ACTIVE_ATTESTATION_ATTRIBUTE,
        raising=False,
    )

    with pytest.raises(
        bootstrap.BootstrapSecurityError,
        match="isolated 2024 bootstrap",
    ):
        runner.run_stage(artifacts.AUDIT_STAGE)
