from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from agent_benchmark import (
    contextual_expert_aggregation_2024_audit_parent as parent,
)


ROOT = Path(__file__).resolve().parents[1]
Error = parent.ParentBundleVerificationError
_FORBIDDEN_READ_TOKENS = (
    "aapl_causal_contextual_expert_aggregation_2024_audit_v2",
    "aapl_spy_qqq_through_2024.csv",
    "input_snapshot_receipt.json",
    "preparation_quarantine",
    "audit_2024_stage_success",
)


@pytest.fixture(scope="module")
def verified_parent() -> tuple[parent.ParentBundleEvidence, tuple[str, ...]]:
    opened: list[str] = []
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    monkeypatch = pytest.MonkeyPatch()

    def guard(path: Path) -> None:
        normalized = path.as_posix().lower()
        if any(token in normalized for token in _FORBIDDEN_READ_TOKENS):
            raise AssertionError(f"parent verifier opened forbidden path: {path}")
        opened.append(normalized)

    def guarded_read_bytes(path: Path) -> bytes:
        guard(path)
        return original_read_bytes(path)

    def guarded_read_text(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        guard(path)
        return original_read_text(path, encoding=encoding, errors=errors)

    def policy_rerun_forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("parent verification must not rerun policy replay")

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(
        parent._replay, "replay_from_empty", policy_rerun_forbidden
    )
    try:
        evidence = parent.verify_parent_bundle(ROOT)
    finally:
        monkeypatch.undo()
    return evidence, tuple(opened)


def test_exact_parent_returns_runner_ready_evidence_without_2024_reads(
    verified_parent: tuple[parent.ParentBundleEvidence, tuple[str, ...]],
) -> None:
    evidence, opened = verified_parent

    assert isinstance(evidence, parent.ParentBundleEvidence)
    assert evidence.verified is True
    assert evidence.stage_pass is True
    assert evidence.status == parent.PARENT_STATUS
    assert evidence.historical_policy_candidate is True
    assert evidence.learning_candidate is False
    assert evidence.exact_payload_names == tuple(
        sorted(parent._audit_artifacts.AUDIT_PAYLOAD_NAMES)
    )
    assert evidence.checkpoint.digest_sha256 == parent.SELECTED_REPLAY_DIGEST
    assert evidence.model_payload["runtime"] == {
        "learning_mode": "causal_online",
        "frozen_cutoff": None,
        "ablation_mode": "full",
    }
    assert opened
    assert not any(
        token in path for token in _FORBIDDEN_READ_TOKENS for path in opened
    )
    assert not hasattr(parent, "_audit_runner")

    for cost in parent._audit_artifacts.COST_ORDER:
        assert tuple(evidence.accounts[cost]) == parent.REQUIRED_POLICIES
        assert tuple(evidence.ledgers[cost]) == parent.REQUIRED_POLICIES
        for policy in parent.REQUIRED_POLICIES:
            state = evidence.accounts[cost][policy]
            ledger = evidence.ledgers[cost][policy]
            assert isinstance(state, parent._ledger.AccountState)
            assert isinstance(ledger, pd.DataFrame)
            assert len(ledger) == parent.PARENT_LEDGER_ROW_COUNT
            assert ledger.iloc[-1]["row_sha256"] == state.ledger_tip_sha256
            assert (
                evidence.account_checkpoint_sha256[cost][policy]
                == parent._ACCOUNT_STATE_SHA256[cost][policy]
            )

    lead = evidence.ledger_for("base_5bps", "frozen_2023_lead")
    shadow = evidence.ledger_for("base_5bps", "online_2024_shadow")
    online = evidence.ledger_for("base_5bps", "online_full")
    assert lead.equals(online)
    assert shadow.equals(online)
    assert lead is not evidence.ledgers["base_5bps"]["online_full"]


def test_parent_verification_payload_has_the_frozen_nested_shape(
    verified_parent: tuple[parent.ParentBundleEvidence, tuple[str, ...]],
) -> None:
    evidence, _ = verified_parent
    payload = evidence.lock_payload()
    assert set(payload) == {
        "verifier_id",
        "verifier_dependency_path",
        "verified",
        "parent_directory",
        "preservation_commit",
        "original_run_commit",
        "contract_version",
        "stage",
        "run_id",
        "status",
        "stage_pass",
        "historical_policy_candidate",
        "learning_candidate",
        "parent_rejection_remains_final",
        "post_2023_market_values_accessed",
        "passed_gate_count",
        "total_gate_count",
        "manifest_file_sha256",
        "manifest_self_sha256",
        "checksums_file_sha256",
        "report_file_sha256",
        "report_self_sha256",
        "gate_report_file_sha256",
        "metrics_file_sha256",
        "integrity_file_sha256",
        "integrity_self_sha256",
        "checkpoint_file_sha256",
        "checkpoint_self_sha256",
        "exact_payload_names",
        "payload_sha256",
        "selected_checkpoint",
        "account_identity",
    }
    assert set(payload["selected_checkpoint"]) == {
        "policy",
        "digest_sha256",
        "checkpoint_date",
        "source_session_count",
        "runtime",
        "processed_session_count",
        "admitted_lesson_count",
        "matured_lesson_count",
        "pending_lesson_count",
        "model_state_sha256",
        "pending_state_sha256",
    }
    assert set(payload["selected_checkpoint"]["runtime"]) == {
        "learning_mode",
        "frozen_cutoff",
        "ablation_mode",
    }
    assert set(payload["account_identity"]) == set(
        parent._audit_artifacts.COST_ORDER
    )
    for cost in parent._audit_artifacts.COST_ORDER:
        assert tuple(payload["account_identity"][cost]) == parent.REQUIRED_POLICIES
        for policy in parent.REQUIRED_POLICIES:
            assert set(payload["account_identity"][cost][policy]) == {
                "account_state_sha256",
                "ledger_tip_sha256",
                "ledger_row_count",
            }
    assert "bundle" not in payload
    assert "checkpoint_bytes" not in payload
    assert "ledgers" not in payload
    assert parent._experiment.pretty_json_bytes(payload)


def test_selected_online_checkpoint_runtime_tamper_fails_closed(
    verified_parent: tuple[parent.ParentBundleEvidence, tuple[str, ...]],
) -> None:
    evidence, _ = verified_parent
    model_payload = copy.deepcopy(dict(evidence.checkpoint.model_payload))
    model_payload["runtime"]["learning_mode"] = "frozen_cutoff"
    changed = replace(evidence.checkpoint, model_payload=model_payload)

    with pytest.raises(Error, match="selected online_full"):
        parent._require_selected_checkpoint(changed)


@pytest.mark.parametrize("tamper", ["checkpoint_hash", "ledger_tip"])
def test_pinned_account_identity_tamper_fails_closed(
    verified_parent: tuple[parent.ParentBundleEvidence, tuple[str, ...]],
    tamper: str,
) -> None:
    evidence, _ = verified_parent
    cost = "base_5bps"
    policy = "online_full"
    state = evidence.accounts[cost][policy]
    checkpoint_value = state.to_checkpoint()
    if tamper == "checkpoint_hash":
        checkpoint_value["account_state_sha256"] = "sha256:" + "0" * 64
    else:
        state = replace(state, ledger_tip_sha256="sha256:" + "0" * 64)

    with pytest.raises(Error, match="pinned parent account"):
        parent._require_account_identity(
            cost=cost,
            policy=policy,
            checkpoint_value=checkpoint_value,
            state=state,
        )


def test_parent_gate_relabel_tamper_fails_closed(
    verified_parent: tuple[parent.ParentBundleEvidence, tuple[str, ...]],
) -> None:
    evidence, _ = verified_parent
    manifest = dict(evidence.bundle.manifest)
    report, _ = parent._canonical_object(evidence.bundle, parent.REPORT_FILENAME)
    gate, _ = parent._canonical_object(
        evidence.bundle, parent.GATE_REPORT_FILENAME
    )
    integrity, _ = parent._canonical_object(
        evidence.bundle, parent.INTEGRITY_FILENAME
    )
    changed_gate = copy.deepcopy(gate)
    changed_gate["passed_count"] = 63

    with pytest.raises(Error, match="64-of-64"):
        parent._require_parent_claims(
            manifest, report, changed_gate, integrity
        )


def test_continuous_ledger_tip_tamper_fails_closed(
    verified_parent: tuple[parent.ParentBundleEvidence, tuple[str, ...]],
) -> None:
    evidence, _ = verified_parent
    cost = "stress_10bps"
    policy = "contextual_only"
    changed = evidence.ledger_for(cost, policy)
    changed.loc[changed.index[-1], "row_sha256"] = "sha256:" + "0" * 64

    with pytest.raises(Error, match="ledger boundary"):
        parent._verify_ledger_frame(
            changed,
            cost=cost,
            policy=policy,
            expected_end_state=evidence.accounts[cost][policy],
        )


def test_manifest_raw_byte_tamper_stops_before_generic_bundle_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = ROOT / parent.PARENT_DIRECTORY
    directory = tmp_path / parent.PARENT_DIRECTORY
    directory.mkdir(parents=True)
    (directory / "stage_manifest.json").write_bytes(
        (source / "stage_manifest.json").read_bytes() + b" "
    )
    (directory / "checksums.json").write_bytes(
        (source / "checksums.json").read_bytes()
    )
    monkeypatch.setattr(
        parent._experiment,
        "_verify_bundle_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generic verifier must not run after raw seal tamper")
        ),
    )

    with pytest.raises(Error, match="raw bytes changed"):
        parent._load_exact_parent_bundle(tmp_path)
