from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.contextual_expert_aggregation_artifacts import (
    ARM_ORDER,
    COST_BPS,
    COST_ORDER,
    FIRST_CONFIRMATION_SESSION,
    POLICY_ORDER,
    SOURCE_SESSION_COUNT_BY_STAGE,
)
from agent_benchmark.contextual_expert_aggregation_audit_artifacts import (
    ATTEMPT_LOCK_FILENAME,
    AUDIT_PAYLOAD_NAMES,
    AUDIT_RUN_ID,
    AUDIT_STAGE,
    CONTRACT_VERSION,
    LAST_OBSERVED_SESSION,
    SOURCE_SESSION_COUNT,
    SOURCE_START_SESSION,
    ContextualExpertAggregationAuditArtifactError,
    audit_checkpoint_bytes,
    build_audit_checkpoint,
    parse_audit_checkpoint,
    parse_audit_checkpoint_bytes,
    seal_audit_bundle,
)
from agent_benchmark.contextual_expert_aggregation_experiment import StageDeadline
from agent_benchmark.contextual_expert_aggregation_ledger import (
    run_continuous_ledger,
)
from agent_benchmark.contextual_expert_aggregation_replay import (
    fork_confirmation_arms,
    replay_from_empty,
)


def _market_frame(*, first: str, end: str, periods: int) -> pd.DataFrame:
    tail = pd.date_range(end=end, periods=periods - 1)
    index = pd.DatetimeIndex(
        [pd.Timestamp(first), *tail], name="date"
    )
    values = 100.0 + np.linspace(0.0, 20.0, periods)
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values,
            "aapl_adj_close": values,
            "spy_adj_close": values + 20.0,
            "qqq_adj_close": values + 30.0,
        },
        index=index,
    )


@pytest.fixture(scope="module")
def terminal_replays():
    development_count = SOURCE_SESSION_COUNT_BY_STAGE["development"]
    development = _market_frame(
        first=SOURCE_START_SESSION,
        end="2018-12-31",
        periods=development_count,
    )
    prefix = replay_from_empty(development)
    suffix = _market_frame(
        first=FIRST_CONFIRMATION_SESSION,
        end=LAST_OBSERVED_SESSION,
        periods=SOURCE_SESSION_COUNT - development_count,
    )
    forks = fork_confirmation_arms(
        suffix, prefix.checkpoint, historical_prefix=development
    )
    return forks.arms


def _accounts() -> dict[str, dict[str, object]]:
    index = pd.DatetimeIndex(
        ["2005-01-03", LAST_OBSERVED_SESSION], name="date"
    )
    opens = pd.Series([100.0, 120.0], index=index, dtype=float)
    result: dict[str, dict[str, object]] = {}
    for cost in COST_ORDER:
        result[cost] = {}
        for policy in POLICY_ORDER:
            run = run_continuous_ledger(
                opens,
                pd.Series([1, 1], index=index, dtype=int),
                policy_name=policy,
                cost_bps=COST_BPS[cost],
            )
            result[cost][policy] = run.state
    return result


def _checkpoint(terminal_replays) -> dict[str, object]:
    return build_audit_checkpoint(
        replay_checkpoints={arm: terminal_replays[arm].checkpoint for arm in ARM_ORDER},
        administrative_accounts=_accounts(),
    )


def test_payload_inventory_and_identity_are_frozen() -> None:
    assert CONTRACT_VERSION.endswith("2019-2023-audit-v2")
    assert AUDIT_STAGE == "audit"
    assert len(AUDIT_PAYLOAD_NAMES) == 51
    assert ATTEMPT_LOCK_FILENAME in AUDIT_PAYLOAD_NAMES
    assert "audit_prices_through_2023.csv" in AUDIT_PAYLOAD_NAMES
    assert "stage_manifest.json" not in AUDIT_PAYLOAD_NAMES
    assert "checksums.json" not in AUDIT_PAYLOAD_NAMES


def test_checkpoint_round_trips_and_is_canonical(terminal_replays) -> None:
    checkpoint = _checkpoint(terminal_replays)
    assert checkpoint["checkpoint_cutoff"] == "2023-12-31"
    assert checkpoint["last_observed_session"] == LAST_OBSERVED_SESSION
    assert parse_audit_checkpoint(checkpoint) == checkpoint
    payload = audit_checkpoint_bytes(checkpoint)
    assert payload.endswith(b"\n")
    assert parse_audit_checkpoint_bytes(payload) == checkpoint


def test_checkpoint_rejects_changed_boundary_even_when_rehashed(
    terminal_replays,
) -> None:
    checkpoint = copy.deepcopy(_checkpoint(terminal_replays))
    checkpoint["last_observed_session"] = "2023-12-28"
    with pytest.raises(
        ContextualExpertAggregationAuditArtifactError,
        match="metadata changed",
    ):
        parse_audit_checkpoint(checkpoint)


def test_checkpoint_rejects_negative_account_cash(terminal_replays) -> None:
    checkpoint = copy.deepcopy(_checkpoint(terminal_replays))
    account = checkpoint["administrative_accounts"]["base_5bps"]["online_full"]
    account["account_state"]["cash"] = -1.0
    with pytest.raises(Exception):
        build_audit_checkpoint(
            replay_checkpoints=checkpoint["replay_checkpoints"],
            administrative_accounts=checkpoint["administrative_accounts"],
        )


def test_seal_requires_exact_inventory_and_promotes_after_callback(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "audit"
    parent.mkdir()
    (parent / ATTEMPT_LOCK_FILENAME).write_bytes(b"lock")
    final = parent / AUDIT_RUN_ID
    payloads = {name: f"payload:{name}\n".encode() for name in AUDIT_PAYLOAD_NAMES}
    observed: dict[str, object] = {}

    def inspect(temporary: Path, manifest: object) -> None:
        observed["temporary"] = temporary
        observed["manifest"] = manifest
        assert (temporary / "report.json").is_file()
        assert json.loads((temporary / "stage_manifest.json").read_text())["stage"] == AUDIT_STAGE

    sealed = seal_audit_bundle(
        final,
        manifest_fields={
            "stage": AUDIT_STAGE,
            "stage_pass": False,
            "run_id": AUDIT_RUN_ID,
        },
        payloads=payloads,
        deadline=StageDeadline(),
        before_promote=inspect,
    )
    assert sealed.directory == final
    assert final.is_dir()
    assert observed["manifest"] == sealed.manifest
    assert sealed.manifest["contract_version"] == CONTRACT_VERSION
    assert sealed.manifest["stage_pass"] is False


def test_seal_rejects_missing_payload_before_writing(tmp_path: Path) -> None:
    parent = tmp_path / "audit"
    parent.mkdir()
    (parent / ATTEMPT_LOCK_FILENAME).write_bytes(b"lock")
    final = parent / AUDIT_RUN_ID
    payloads = {name: b"x" for name in AUDIT_PAYLOAD_NAMES}
    payloads.pop("report.json")
    with pytest.raises(
        ContextualExpertAggregationAuditArtifactError,
        match="inventory differs",
    ):
        seal_audit_bundle(
            final,
            manifest_fields={
                "stage": AUDIT_STAGE,
                "stage_pass": False,
                "run_id": AUDIT_RUN_ID,
            },
            payloads=payloads,
            deadline=StageDeadline(),
        )
    assert not final.exists()


def test_seal_fsyncs_private_bundle_then_output_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "audit"
    parent.mkdir()
    (parent / ATTEMPT_LOCK_FILENAME).write_bytes(b"lock")
    final = parent / AUDIT_RUN_ID
    payloads = {name: f"payload:{name}\n".encode() for name in AUDIT_PAYLOAD_NAMES}
    calls: list[Path] = []

    def record(path: Path) -> bool:
        calls.append(path)
        return True

    monkeypatch.setattr(
        "agent_benchmark.contextual_expert_aggregation_audit_artifacts._experiment._fsync_directory",
        record,
    )
    seal_audit_bundle(
        final,
        manifest_fields={
            "stage": AUDIT_STAGE,
            "stage_pass": False,
            "run_id": AUDIT_RUN_ID,
        },
        payloads=payloads,
        deadline=StageDeadline(),
    )

    assert calls[-2:] == [parent / ".audit-v2.sealing" / AUDIT_RUN_ID, parent]
    assert final.is_dir()
    assert not (parent / ".audit-v2.sealing").exists()


def test_seal_parent_fsync_failure_after_rename_removes_promoted_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "audit"
    parent.mkdir()
    lock = parent / ATTEMPT_LOCK_FILENAME
    lock.write_bytes(b"lock")
    final = parent / AUDIT_RUN_ID
    payloads = {name: f"payload:{name}\n".encode() for name in AUDIT_PAYLOAD_NAMES}

    def fail_parent(path: Path) -> bool:
        if path == parent:
            raise OSError("simulated parent fsync failure")
        return True

    monkeypatch.setattr(
        "agent_benchmark.contextual_expert_aggregation_audit_artifacts._experiment._fsync_directory",
        fail_parent,
    )
    with pytest.raises(OSError, match="simulated parent fsync failure"):
        seal_audit_bundle(
            final,
            manifest_fields={
                "stage": AUDIT_STAGE,
                "stage_pass": False,
                "run_id": AUDIT_RUN_ID,
            },
            payloads=payloads,
            deadline=StageDeadline(),
        )

    assert lock.read_bytes() == b"lock"
    assert not final.exists()
    assert not (parent / ".audit-v2.sealing").exists()
    assert not (parent / ".audit-v2.failed").exists()
