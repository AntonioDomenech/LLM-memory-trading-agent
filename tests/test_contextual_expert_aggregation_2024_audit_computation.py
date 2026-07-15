from __future__ import annotations

import hashlib
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_2024_audit_artifacts as artifacts
from agent_benchmark import contextual_expert_aggregation_2024_audit_computation as computation
from agent_benchmark import contextual_expert_aggregation_2024_audit_input as audit_input
from agent_benchmark import contextual_expert_aggregation_2024_audit_parent as audit_parent
from agent_benchmark import contextual_expert_aggregation_2024_audit_replay as audit_replay
from agent_benchmark import contextual_expert_aggregation_2024_audit_runner as audit_runner
from agent_benchmark import contextual_expert_aggregation_2024_audit_verifier as audit_verifier
from agent_benchmark import contextual_expert_aggregation_audit_artifacts as parent_artifacts
from agent_benchmark import contextual_expert_aggregation_experiment as experiment
from agent_benchmark import contextual_expert_aggregation_ledger as ledger
from agent_benchmark import contextual_expert_aggregation_replay as replay


_PARENT_ROWS = computation.EXPECTED_FULL_LEDGER_ROWS - computation.EXPECTED_SUFFIX_ROWS
_COST_BPS = {"base_5bps": 5.0, "stress_10bps": 10.0}


def _sha256(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('ascii')).hexdigest()}"


def _select_sessions(index: pd.DatetimeIndex, count: int) -> pd.DatetimeIndex:
    positions = np.linspace(0, len(index) - 1, num=count, dtype=int)
    result = pd.DatetimeIndex(index[positions], name="date")
    assert len(result) == count
    assert result.is_monotonic_increasing and not result.has_duplicates
    return result


def _synthetic_indexes() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    # The parent replay has 6,244 source rows, but its account starts at the
    # fixed 2005-01-03 inception and therefore has exactly 4,781 ledger rows.
    pre_inception = pd.bdate_range(end="2004-12-31", periods=6_244 - _PARENT_ROWS)
    post_inception = _select_sessions(
        pd.bdate_range("2005-01-03", "2023-12-29"), _PARENT_ROWS
    )
    suffix = _select_sessions(
        pd.bdate_range("2024-01-02", "2024-12-31"),
        computation.EXPECTED_SUFFIX_ROWS,
    )
    prefix = pd.DatetimeIndex(pre_inception.append(post_inception), name="date")
    assert len(prefix) == 6_244
    assert prefix[-1] == pd.Timestamp(audit_replay.FROZEN_CUTOFF)
    assert post_inception[0] == pd.Timestamp(ledger.ACCOUNT_INCEPTION_FILL_DATE)
    assert suffix[-1] == pd.Timestamp("2024-12-31")
    return prefix, suffix


def _synthetic_market(index: pd.DatetimeIndex) -> pd.DataFrame:
    step = np.arange(len(index), dtype=float)
    close = 100.0 * np.exp(
        0.00012 * step
        + 0.030 * np.sin(step / 17.0)
        + 0.012 * np.sin(step / 5.0)
    )
    open_price = close * np.exp(
        -0.027 * np.sin(step / 7.0) - 0.009 * np.cos(step / 19.0)
    )
    return pd.DataFrame(
        {
            "aapl_open": open_price,
            "aapl_close": close,
            "aapl_adj_close": close,
            "spy_adj_close": 180.0
            * np.exp(0.00008 * step + 0.018 * np.sin(step / 23.0)),
            "qqq_adj_close": 140.0
            * np.exp(0.00010 * step + 0.022 * np.sin(step / 29.0)),
        },
        index=pd.DatetimeIndex(index, name="date"),
    )


def _parent_policy_targets(
    parent_replay: replay.ReplayResult,
) -> dict[str, pd.Series]:
    index = parent_replay.forecast.index
    return {
        "online_full": parent_replay.forecast[
            "learner_target_exposure"
        ].astype(int),
        "aapl_buy_hold": pd.Series(1, index=index, dtype=int),
        "always_long": pd.Series(1, index=index, dtype=int),
        "exact_union_cash": parent_replay.opportunity_frame[
            "fixed_union_cash_target_exposure"
        ].astype(int),
        "contextual_only": parent_replay.opportunity_frame[
            "fixed_contextual_only_target_exposure"
        ].astype(int),
    }


def _synthetic_parent(
    prefix: pd.DataFrame,
) -> audit_parent.ParentBundleEvidence:
    parent_replay = replay.replay_from_empty(prefix)
    targets = _parent_policy_targets(parent_replay)
    accounts: dict[str, dict[str, ledger.AccountState]] = {}
    ledgers: dict[str, dict[str, pd.DataFrame]] = {}
    account_hashes: dict[str, dict[str, str]] = {}
    for cost in artifacts.COST_ORDER:
        accounts[cost] = {}
        ledgers[cost] = {}
        account_hashes[cost] = {}
        for policy in audit_parent.REQUIRED_POLICIES:
            run = ledger.run_continuous_ledger(
                prefix["aapl_adj_open"],
                targets[policy],
                policy_name=policy,
                cost_bps=_COST_BPS[cost],
            )
            assert len(run.ledger) == _PARENT_ROWS
            assert run.state.ledger_row_count == _PARENT_ROWS
            accounts[cost][policy] = run.state
            ledgers[cost][policy] = run.ledger
            account_hashes[cost][policy] = run.state.to_checkpoint()[
                "account_state_sha256"
            ]

    dummy_bundle = experiment.VerifiedBundle(
        directory=Path("synthetic-parent"),
        manifest_path=Path("synthetic-parent/stage_manifest.json"),
        manifest={},
        payload_sha256={},
        checksums={},
    )
    return audit_parent.ParentBundleEvidence(
        verifier_id="synthetic-parent-verifier",
        verifier_dependency_path="synthetic",
        verified=True,
        parent_directory="synthetic-parent",
        preservation_commit="0" * 40,
        original_run_commit="1" * 40,
        contract_version="synthetic-parent-v1",
        stage="audit",
        run_id="synthetic-parent",
        status="SYNTHETIC_VERIFIED",
        stage_pass=True,
        historical_policy_candidate=True,
        learning_candidate=False,
        passed_gate_count=1,
        total_gate_count=1,
        manifest_file_sha256=_sha256("manifest-file"),
        manifest_self_sha256=_sha256("manifest-self"),
        checksums_file_sha256=_sha256("checksums"),
        report_file_sha256=_sha256("report-file"),
        report_self_sha256=_sha256("report-self"),
        gate_report_file_sha256=_sha256("gate-report"),
        metrics_file_sha256=_sha256("metrics"),
        integrity_file_sha256=_sha256("integrity-file"),
        integrity_self_sha256=_sha256("integrity-self"),
        checkpoint_file_sha256=_sha256("checkpoint-file"),
        checkpoint_self_sha256=_sha256("checkpoint-self"),
        exact_payload_names=(),
        payload_sha256={},
        bundle=dummy_bundle,
        manifest_bytes=b"{}\n",
        checksums_bytes=b"{}\n",
        checkpoint_bytes=b"{}\n",
        parsed_checkpoint=parent_replay.checkpoint.to_dict(),
        checkpoint=parent_replay.checkpoint,
        model_payload=parent_replay.checkpoint.model_payload,
        accounts=accounts,
        account_checkpoint_sha256=account_hashes,
        ledgers=ledgers,
    )


@pytest.fixture(scope="module")
def synthetic_computations() -> tuple[
    computation.AuditComputation,
    audit_replay.TwoScenarioFork,
]:
    prefix_index, suffix_index = _synthetic_indexes()
    combined = replay.canonical_market_frame(
        _synthetic_market(prefix_index.append(suffix_index))
    )
    prefix = combined.loc[prefix_index].copy()
    suffix = combined.loc[suffix_index].copy()
    parent = _synthetic_parent(prefix)
    bounded = audit_input.BoundedAuditSnapshot(
        snapshot=cast(Any, None),
        input_bytes=b"synthetic-bounded-input",
        receipt={"synthetic": True},
        receipt_bytes=b'{"synthetic":true}\n',
        historical_prefix=prefix,
        suffix=suffix,
        prefix_continuity={"synthetic_causal_fixture": True},
    )
    result = computation.compute_2024_continuation(
        parent=parent, bounded=bounded
    )
    repeated_fork = audit_replay.fork_2024_scenarios(
        bounded.suffix,
        parent.checkpoint,
        historical_prefix=bounded.historical_prefix,
    )
    return result, repeated_fork


def test_full_synthetic_causal_continuation_has_exact_boundaries_and_policies(
    synthetic_computations: tuple[
        computation.AuditComputation,
        audit_replay.TwoScenarioFork,
    ],
) -> None:
    result, _ = synthetic_computations

    assert tuple(result.forecasts) == audit_replay.SCENARIO_ORDER
    assert tuple(result.matured_lessons) == audit_replay.SCENARIO_ORDER
    assert len(result.fixed_features) == computation.EXPECTED_SUFFIX_ROWS
    assert len(result.state_weight_diagnostics) == 2 * computation.EXPECTED_SUFFIX_ROWS
    assert result.fixed_features.index.equals(result.bounded.suffix.index)
    assert tuple(result.fixed_comparator_forecast.columns) == (
        "fixed_always_long_target_exposure",
        "fixed_union_cash_target_exposure",
        "fixed_contextual_only_target_exposure",
    )

    lead = result.forks.scenarios[audit_replay.FROZEN_2023_LEAD]
    shadow = result.forks.scenarios[audit_replay.ONLINE_2024_SHADOW]
    assert lead.diagnostics.admitted_events == 0
    assert lead.checkpoint.model_payload["runtime"] == {
        "learning_mode": replay.FROZEN_CUTOFF_MODE,
        "frozen_cutoff": audit_replay.FROZEN_CUTOFF,
        "ablation_mode": replay.FULL_MODE,
    }
    assert shadow.checkpoint.model_payload["runtime"] == {
        "learning_mode": replay.CAUSAL_ONLINE_MODE,
        "frozen_cutoff": None,
        "ablation_mode": replay.FULL_MODE,
    }
    assert lead.opportunity_frame.equals(shadow.opportunity_frame)
    for frame in result.matured_lessons.values():
        assert all(
            pd.Timestamp(signal) < pd.Timestamp(maturity)
            for signal, maturity in zip(frame["signal_date"], frame["maturity_date"])
        )

    for cost in artifacts.COST_ORDER:
        assert tuple(result.ledgers[cost]) == artifacts.POLICY_ORDER
        assert tuple(result.accounts[cost]) == artifacts.POLICY_ORDER
        for policy in artifacts.POLICY_ORDER:
            full = result.ledgers[cost][policy]
            suffix = result.suffix_ledgers[cost][policy]
            account = result.accounts[cost][policy]
            parent_state = result.parent.account_for(cost, policy)
            assert len(full) == computation.EXPECTED_FULL_LEDGER_ROWS
            assert len(suffix) == computation.EXPECTED_SUFFIX_ROWS
            assert account.ledger_row_count == computation.EXPECTED_FULL_LEDGER_ROWS
            assert account.inception_count == parent_state.inception_count == 1
            assert suffix["inception_fill"].eq(False).all()
            assert suffix.iloc[0]["previous_row_sha256"] == parent_state.ledger_tip_sha256
            assert full.iloc[-1]["row_sha256"] == account.ledger_tip_sha256
            assert set(full["requested_target_exposure"]) <= {0, 1}
            assert set(full["post_fill_exposure"]) <= {0, 1}
            assert full["cash_after_fill"].ge(0.0).all()
            assert full["shares_after_fill"].ge(0.0).all()
            assert full["margin_interest"].eq(0.0).all()
            assert full["cash_interest"].eq(0.0).all()
            assert full["fees"].eq(0.0).all()
            assert result.episodes[cost][policy].reconciliation["passed"] is True

        assert result.accounts[cost][audit_replay.FROZEN_2023_LEAD].policy_name == "online_full"
        assert result.accounts[cost][audit_replay.ONLINE_2024_SHADOW].policy_name == "online_full"
        assert result.accounts[cost]["aapl_buy_hold"].policy_name == "aapl_buy_hold"
        assert result.accounts[cost]["always_long"].policy_name == "always_long"
        assert result.xors[cost].reconciliation["passed"] is True

    assert all(result.integrity_proofs.values())


def test_synthetic_full_computation_is_deterministic_across_both_costs(
    synthetic_computations: tuple[
        computation.AuditComputation,
        audit_replay.TwoScenarioFork,
    ],
) -> None:
    first, repeated_fork = synthetic_computations

    assert first.forks.parent_checkpoint_sha256 == repeated_fork.parent_checkpoint_sha256
    for scenario in audit_replay.SCENARIO_ORDER:
        repeated = repeated_fork.scenarios[scenario]
        assert first.forecasts[scenario].equals(repeated.forecast)
        assert first.matured_lessons[scenario].equals(repeated.matured_lessons)
        assert (
            first.forks.scenarios[scenario].checkpoint.digest_sha256
            == repeated.checkpoint.digest_sha256
        )
        assert first.action_stream_sha256[scenario] == computation._action_hash(
            repeated.forecast, column="learner_target_exposure"
        )
    for cost in artifacts.COST_ORDER:
        for policy in artifacts.POLICY_ORDER:
            reconstructed = pd.concat(
                [
                    first.parent.ledger_for(cost, policy),
                    first.suffix_ledgers[cost][policy],
                ],
                ignore_index=True,
            )
            assert first.ledgers[cost][policy].equals(reconstructed)
            assert (
                first.ledgers[cost][policy].iloc[-1]["row_sha256"]
                == first.accounts[cost][policy].ledger_tip_sha256
            )


def test_synthetic_integrity_inventory_proves_lesson_conservation_and_checkpoints(
    synthetic_computations: tuple[
        computation.AuditComputation,
        audit_replay.TwoScenarioFork,
    ],
) -> None:
    result, _ = synthetic_computations
    commit = "c" * 40
    bounded_evidence = SimpleNamespace(
        snapshot=SimpleNamespace(
            raw_sha256=audit_input.INPUT_RAW_SHA256,
            spec=SimpleNamespace(
                canonical_sha256=audit_input.INPUT_CANONICAL_SHA256
            ),
        ),
        historical_prefix=result.bounded.historical_prefix,
        suffix=result.bounded.suffix,
        prefix_continuity={
            "prefix_market_chain_exact": True,
            "parent_checkpoint_sha256": result.parent.checkpoint.digest_sha256,
        },
        receipt_bytes=b"synthetic-receipt",
        receipt={"synthetic": True},
    )
    git_identity = {
        "branch": artifacts.EXPECTED_BRANCH,
        "commit": commit,
        "upstream_commit": commit,
        "preregistration_is_ancestor": True,
    }
    lock = SimpleNamespace(
        content={
            "git_commit": commit,
            "artifact_schema_registry_sha256": (
                artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
            ),
        },
        raw_sha256=_sha256("synthetic-lock"),
    )

    checks = audit_runner._integrity_checks(
        computation=result,
        parent=result.parent,
        bounded=bounded_evidence,
        lock=lock,
        git_identity=git_identity,
    )

    assert set(result.integrity_proofs) == audit_runner._EXPECTED_COMPUTATION_PROOFS
    assert checks["one_lesson_per_eligible_opportunity"] is True
    assert checks["pending_and_cooldown_state_exact"] is True
    # The deliberately synthetic receipt is correctly classified as a failed
    # integrity item rather than aborting evidence construction.
    assert checks["sanitized_receipt_identity"] is False
    evaluated = audit_runner._evaluate(result, checks=checks)
    assert evaluated["gate_report"]["status"] == "REJECTED_2024_INTEGRITY"
    assert evaluated["gate_report"]["stage_pass"] is False


def test_full_synthetic_payloads_match_every_frozen_deep_json_schema(
    synthetic_computations: tuple[
        computation.AuditComputation,
        audit_replay.TwoScenarioFork,
    ],
) -> None:
    result, _ = synthetic_computations
    parent_names = tuple(sorted(parent_artifacts.AUDIT_PAYLOAD_NAMES))
    parent_payload_hashes = {
        name: _sha256(f"parent-payload:{name}") for name in parent_names
    }
    parent = replace(
        result.parent,
        exact_payload_names=parent_names,
        payload_sha256=parent_payload_hashes,
    )
    proof = {
        "prefix_proof_schema_version": 1,
        "through_2023_rows": 6_244,
        "through_2023_last_session": audit_replay.FROZEN_CUTOFF,
        "through_2023_canonical_sha256": audit_input.PREFIX_CANONICAL_SHA256,
        "parent_checkpoint_sha256": parent.checkpoint.digest_sha256,
        "parent_checkpoint_source_sessions": parent.checkpoint.source_session_count,
        "prefix_market_chain_exact": True,
        "suffix_first_session": result.bounded.suffix.index[0].date().isoformat(),
        "suffix_last_session": result.bounded.suffix.index[-1].date().isoformat(),
        "suffix_rows": len(result.bounded.suffix),
        "suffix_released_only_after_prefix_proof": True,
    }
    bounded = SimpleNamespace(
        snapshot=SimpleNamespace(
            raw_sha256=audit_input.INPUT_RAW_SHA256,
            spec=audit_input.INPUT_SPEC,
        ),
        input_bytes=b"synthetic-bounded-input",
        receipt=copy.deepcopy(audit_input.RECEIPT_EXPECTED),
        receipt_bytes=b"synthetic-sanitized-receipt",
        historical_prefix=result.bounded.historical_prefix,
        suffix=result.bounded.suffix,
        prefix_continuity=proof,
    )
    synthetic = replace(result, parent=parent, bounded=cast(Any, bounded))
    commit = "c" * 40
    dependency_path = "agent_benchmark/synthetic_dependency.py"
    git_identity = {
        "branch": artifacts.EXPECTED_BRANCH,
        "commit": commit,
        "upstream": f"origin/{artifacts.EXPECTED_BRANCH}",
        "upstream_commit": commit,
        "origin_url": "https://github.com/synthetic/repository.git",
        "origin_repository": "synthetic/repository",
        "preregistration_commit": artifacts.PREREGISTRATION_COMMIT,
        "preregistration_is_ancestor": True,
        "head_equals_upstream": True,
        "dependency_identity": {
            "schema_version": 1,
            "files": {
                dependency_path: {
                    "head_blob": "a" * 40,
                    "index_blob": "a" * 40,
                    "worktree_raw_sha256": _sha256("dependency"),
                }
            },
        },
        "dependency_identity_sha256": _sha256("dependencies"),
        "input_git_identity": {
            "path": audit_input.INPUT_PATH.as_posix(),
            "head_blob": audit_input.INPUT_GIT_BLOB,
            "index_blob": audit_input.INPUT_GIT_BLOB,
        },
        "receipt_git_identity": {
            "path": audit_input.RECEIPT_PATH.as_posix(),
            "head_blob": audit_input.RECEIPT_GIT_BLOB,
            "index_blob": audit_input.RECEIPT_GIT_BLOB,
        },
        "prelock_input_worktree_bytes_opened": False,
        "prelock_receipt_worktree_bytes_opened": False,
        "runtime_versions": {
            "python": "synthetic",
            "numpy": "synthetic",
            "pandas": "synthetic",
        },
    }
    lock_content = {
        "git_commit": commit,
        "artifact_schema_registry_sha256": (
            artifacts.ARTIFACT_SCHEMA_REGISTRY_SHA256
        ),
    }
    lock_bytes = artifacts.canonical_json_line_bytes(lock_content)
    lock = audit_runner._control.AttemptLockEvidence(
        path=Path("synthetic-lock.json"),
        content=lock_content,
        bytes_value=lock_bytes,
        raw_sha256=experiment.sha256_bytes(lock_bytes),
    )
    checks = audit_runner._integrity_checks(
        computation=synthetic,
        parent=parent,
        bounded=bounded,
        lock=lock,
        git_identity=git_identity,
    )
    evaluated = audit_runner._evaluate(synthetic, checks=checks)
    runtime_clock = audit_runner.RuntimeClock(start=0.0, clock=lambda: 0.0)
    runtime_clock.samples = {
        "preseal": 1.0,
        "post_private_verify": 2.0,
        "prepromotion": 3.0,
    }
    runtime = audit_runner._runtime_payload(
        clock=runtime_clock, provisional=False
    )
    payloads = audit_runner._build_reproducible_payloads(
        parent=parent,
        bounded=bounded,
        lock=lock,
        git_identity=git_identity,
        computation=synthetic,
        checks=checks,
        evaluation=evaluated,
        runtime_payload=runtime,
    )

    registry = artifacts.ARTIFACT_SCHEMA_REGISTRY["json_artifacts"]
    for filename, descriptor in registry.items():
        value = artifacts.parse_canonical_json_line(
            payloads[filename], field=filename
        )
        audit_verifier._validate_schema(
            value,
            descriptor["literal_nested_schema"],
            field_name=filename,
        )
