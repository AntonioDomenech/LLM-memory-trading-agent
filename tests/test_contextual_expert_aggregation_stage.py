from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agent_benchmark import contextual_expert_aggregation_artifacts as artifacts
from agent_benchmark import contextual_expert_aggregation_ledger as ledger
from agent_benchmark import contextual_expert_aggregation_replay as replay
from agent_benchmark import contextual_expert_aggregation_stage as stage


Error = stage.ContextualExpertAggregationStageError


def _opens() -> pd.Series:
    index = pd.DatetimeIndex(
        ["2005-01-03", "2005-01-04", "2005-01-05", "2005-01-06"]
    )
    return pd.Series([100.0, 101.0, 99.0, 102.0], index=index, dtype=float)


def _synthetic_replay_frame() -> pd.DataFrame:
    all_dates = pd.bdate_range("1999-03-10", "2018-12-31")
    anchors = pd.DatetimeIndex(["1999-03-10", "2005-01-03", "2018-12-31"])
    removable = all_dates.difference(anchors)
    remove_count = len(all_dates) - artifacts.SOURCE_SESSION_COUNT_BY_STAGE[
        "development"
    ]
    remove_positions = np.linspace(
        0, len(removable) - 1, remove_count, dtype=int
    )
    dates = all_dates.difference(removable[remove_positions])
    assert len(dates) == artifacts.SOURCE_SESSION_COUNT_BY_STAGE["development"]
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


def _small_development_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2004-06-01", "2004-12-31")
    dates = dates.append(
        pd.DatetimeIndex(["2005-01-03", "2005-01-04", "2018-12-31"])
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
    dates = pd.DatetimeIndex(["2019-01-02", "2019-01-03", "2023-12-29"])
    values = np.array([111.0, 112.0, 115.0])
    return pd.DataFrame(
        {
            "aapl_open": values,
            "aapl_close": values * 1.001,
            "aapl_adj_close": values * 1.001,
            "spy_adj_close": np.array([121.0, 122.0, 125.0]),
            "qqq_adj_close": np.array([106.0, 107.0, 110.0]),
        },
        index=dates,
    )


def _all_policy_ledgers(
    *, online_targets: list[int], other_targets: list[int] | None = None
) -> dict[str, dict[str, pd.DataFrame]]:
    opens = _opens()
    fallback = other_targets or [1, 1, 1, 1]
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for cost_name in artifacts.COST_ORDER:
        result[cost_name] = {}
        for policy in artifacts.POLICY_ORDER:
            targets = online_targets if policy == replay.ONLINE_FULL_ARM else fallback
            run = ledger.run_continuous_ledger(
                opens,
                pd.Series(targets, index=opens.index),
                policy_name=policy,
                cost_bps=artifacts.COST_BPS[cost_name],
            )
            result[cost_name][policy] = run.ledger
    return result


def _terminal_evidence(
    *, online_targets: list[int], other_targets: list[int] | None = None
):
    ledgers = _all_policy_ledgers(
        online_targets=online_targets, other_targets=other_targets
    )
    return stage._extract_episodes_and_xors(ledgers)


def test_public_cli_has_only_one_frozen_stage_argument() -> None:
    parser = stage._build_parser()
    assert parser.parse_args(["development"]).stage == "development"
    assert parser.parse_args(["confirmation"]).stage == "confirmation"
    for bad in (
        [],
        ["validation"],
        ["development", "--repo-root", "elsewhere"],
        ["confirmation", "--run-id", "retry"],
        ["development", "--output-dir", "elsewhere"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_frozen_paths_run_ids_and_policy_inventory_are_artifact_owned() -> None:
    assert stage.RUN_ID_BY_STAGE == artifacts.RUN_ID_BY_STAGE
    assert stage.RUNTIME_PHASES_BY_STAGE is artifacts.RUNTIME_PHASES_BY_STAGE
    assert stage.OUTPUT_DIRECTORY_BY_STAGE == artifacts.OUTPUT_DIRECTORY_BY_STAGE
    assert stage.POLICY_ORDER == artifacts.POLICY_ORDER
    assert stage.COST_ORDER == artifacts.COST_ORDER
    assert stage.DEVELOPMENT_MANIFEST_PATH == (
        artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"] / "stage_manifest.json"
    )
    assert stage.FROZEN_COMMAND_BY_STAGE == {
        selected: (
            "python",
            "-I",
            "-B",
            "agent_benchmark/contextual_expert_aggregation_bootstrap.py",
            "stage",
            selected,
        )
        for selected in artifacts.STAGE_ORDER
    }


def test_confirmation_checkpoint_authorization_uses_payload_hash_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    replays, _ = stage._development_replays(frame)
    _, accounts = stage._run_development_ledgers(
        snapshot=SimpleNamespace(frame=frame), replays=replays
    )
    value = artifacts.build_composite_stage_checkpoint(
        stage="development",
        replay_checkpoints={
            arm: replays[arm].checkpoint for arm in artifacts.ARM_ORDER
        },
        administrative_accounts=accounts,
    )
    payload = artifacts.composite_stage_checkpoint_bytes(value)
    parsed = artifacts.parse_composite_stage_checkpoint_bytes(payload)
    payload_hash = stage._experiment.sha256_bytes(payload)

    assert payload_hash != parsed.checkpoint_sha256
    assert stage._require_authorized_checkpoint_payload(
        payload,
        parsed,
        authorized_payload_sha256=payload_hash,
    ) == payload_hash
    with pytest.raises(Error, match="payload changed"):
        stage._require_authorized_checkpoint_payload(
            payload,
            parsed,
            authorized_payload_sha256=parsed.checkpoint_sha256,
        )


def test_ledger_typed_table_roundtrip_preserves_canonical_semantics() -> None:
    opens = _opens()
    run = ledger.run_continuous_ledger(
        opens,
        pd.Series([1, 0, 1, 1], index=opens.index),
        policy_name="online_full",
        cost_bps=5.0,
    )
    payload = stage.ledger_table_bytes(run.ledger)
    restored = stage.parse_ledger_table_bytes(payload)
    end = ledger.verify_ledger(
        restored,
        start_state=ledger.AccountState.initial(
            policy_name="online_full", cost_bps=5.0
        ),
    )
    assert end == run.state
    assert restored.to_dict(orient="records") == run.ledger.to_dict(orient="records")


def test_forecast_schema_preserves_nullable_weights_without_nan_json() -> None:
    row: dict[str, object] = {}
    for name, scalar_type in zip(
        stage.FORECAST_SCHEMA.columns, stage.FORECAST_SCHEMA.column_types
    ):
        if scalar_type == "bool":
            row[name] = False
        elif scalar_type == "int":
            row[name] = 0
        elif scalar_type == "string":
            row[name] = ""
        elif scalar_type == "nullable_float":
            row[name] = np.nan
        else:
            row[name] = 1.0
    row["market_state"] = "unknown"
    row["action"] = "LONG"
    frame = pd.DataFrame([row], columns=stage.FORECAST_SCHEMA.columns)
    frame.index = pd.DatetimeIndex(["2005-01-03"], name="decision_date")
    payload = stage.table_bytes(frame, schema=stage.FORECAST_SCHEMA)
    assert b"NaN" not in payload
    restored = artifacts.parse_canonical_table_bytes(
        payload, schema=stage.FORECAST_SCHEMA
    )
    assert restored.iloc[0]["rho__half_life_8"] is None


def test_final_close_unexecuted_cash_decision_is_not_fatal() -> None:
    episodes, xors = _terminal_evidence(
        online_targets=[1, 1, 1, 0],
        other_targets=[1, 1, 1, 1],
    )
    unresolved = episodes["base_5bps"]["online_full"].unresolved
    assert unresolved.iloc[0]["status"] == "pending_cash_entry_unexecuted"
    assert (
        stage.classify_terminal_residuals(
            episodes, xors, stage="development"
        )
        == []
    )


def test_executed_terminal_cash_and_partial_xor_are_deterministic_fatal_reasons() -> None:
    episodes, xors = _terminal_evidence(
        online_targets=[1, 0, 0, 0],
        other_targets=[1, 1, 1, 1],
    )
    reasons = stage.classify_terminal_residuals(
        episodes, xors, stage="confirmation"
    )
    assert reasons == sorted(
        reasons,
        key=lambda value: tuple(
            str(value.get(name, ""))
            for name in (
                "reason_code",
                "cost",
                "policy",
                "comparison",
                "entry_fill_date",
                "last_fill_date",
                "mark_date",
            )
        ),
    )
    assert {item["reason_code"] for item in reasons} == {
        "executed_terminal_cash_residual",
        "partial_xor_boundary",
    }
    integrity = stage._integrity_evidence(
        stage="confirmation", fatal_reasons=reasons
    )
    report = stage.build_fatal_gate_report(
        stage="confirmation", integrity=integrity, fatal_reasons=reasons
    )
    assert report["passed"] is False
    assert report["fatal_integrity_rejection"] is True
    assert report["normal_economic_gates_evaluated"] is False
    assert report["checks"][
        "fatal.no_executed_terminal_cash_residual"
    ] is False
    assert report["checks"]["fatal.no_partial_xor_boundary"] is False
    assert report["failed_checks"] == [
        name for name, value in report["checks"].items() if not value
    ]


def test_fatal_report_rejects_nonfatal_or_wrong_integrity_inventory() -> None:
    integrity = {
        name: True for name in stage._integrity_inventory("development")
    }
    with pytest.raises(Error, match="at least one"):
        stage.build_fatal_gate_report(
            stage="development", integrity=integrity, fatal_reasons=[]
        )
    integrity.pop(next(iter(integrity)))
    with pytest.raises(Error, match="inventory"):
        stage.build_fatal_gate_report(
            stage="development",
            integrity=integrity,
            fatal_reasons=[
                {
                    "reason_code": "partial_xor_boundary",
                    "cost": "base_5bps",
                }
            ],
        )


def test_normal_reconciliation_path_uses_exact_ledger_api() -> None:
    ledgers = _all_policy_ledgers(
        online_targets=[1, 1, 1, 1],
        other_targets=[1, 1, 1, 1],
    )
    episodes, xors = stage._extract_episodes_and_xors(ledgers)
    stage._reconcile_evidence(
        SimpleNamespace(
            stage="development",
            ledgers=ledgers,
            episodes=episodes,
            xors=xors,
        )
    )


def test_development_arm_targets_all_use_online_full_seed() -> None:
    index = pd.DatetimeIndex(["2005-01-03", "2005-01-04"])
    forecast = pd.DataFrame(
        {"learner_target_exposure": [1.0, 0.0]}, index=index
    )
    fixed = pd.DataFrame(
        {
            "fixed_always_long_target_exposure": [1.0, 1.0],
            "fixed_union_cash_target_exposure": [1.0, 0.0],
            "fixed_contextual_only_target_exposure": [1.0, 1.0],
            "fixed_weak_trend_only_target_exposure": [1.0, 1.0],
        },
        index=index,
    )
    replays = {
        arm: SimpleNamespace(
            forecast=(
                forecast
                if arm == "online_full"
                else pd.DataFrame(
                    {"learner_target_exposure": [0.0, 0.0]}, index=index
                )
            )
        )
        for arm in artifacts.ARM_ORDER
    }
    for arm in artifacts.ARM_ORDER:
        result = stage._policy_target(
            stage="development",
            policy=arm,
            replays=replays,
            fixed_features=fixed,
        )
        assert result.tolist() == [1.0, 0.0]
    confirmation = stage._policy_target(
        stage="confirmation",
        policy="frozen_2018",
        replays=replays,
        fixed_features=fixed,
    )
    assert confirmation.tolist() == [0.0, 0.0]


def test_development_replay_and_accounts_form_exact_four_arm_seed_checkpoint() -> None:
    frame = _synthetic_replay_frame()
    replays, proof = stage._development_replays(frame)
    snapshot = SimpleNamespace(frame=frame)
    ledgers, accounts = stage._run_development_ledgers(
        snapshot=snapshot, replays=replays
    )
    checkpoint = artifacts.build_composite_stage_checkpoint(
        stage="development",
        replay_checkpoints={
            arm: replays[arm].checkpoint for arm in artifacts.ARM_ORDER
        },
        administrative_accounts=accounts,
    )
    parsed = artifacts.parse_composite_stage_checkpoint(checkpoint)
    assert proof["seed_target_source_policy"] == "online_full"
    assert all(
        item["passed"]
        for item in proof["development_arm_replay_identity"].values()
    )
    assert parsed.arm_seed_equivalence is not None
    assert all(
        parsed.arm_seed_equivalence["by_cost"][cost]["all_equal"]
        for cost in artifacts.COST_ORDER
    )
    for cost in artifacts.COST_ORDER:
        for arm in artifacts.ARM_ORDER:
            assert (
                ledgers[cost][arm]["close_decision_target_exposure"].tolist()
                == ledgers[cost]["online_full"][
                    "close_decision_target_exposure"
                ].tolist()
            )

    episodes, xors = stage._extract_episodes_and_xors(ledgers)
    digest = "sha256:" + "b" * 64
    computation = stage.StageComputation(
        stage="development",
        snapshot=SimpleNamespace(
            provenance={"source": "synthetic"},
            canonical_csv_bytes=b"synthetic authorized prices\n",
        ),
        replays=replays,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        parent_causal_prefix_proof={"proof_sha256": digest},
        source_bundle_provenance={"provenance_sha256": digest},
        replay_diagnostics={"diagnostics_sha256": digest},
        prefix_continuity_proof=None,
        confirmation_authorization=None,
        development_parent_manifest_bytes=None,
        development_parent_checksums_bytes=None,
    )
    integrity = {
        name: True for name in stage._integrity_inventory("development")
    }
    payloads, sealed_checkpoint = stage._base_payloads(
        computation,
        integrity=integrity,
        gate_report={"passed": False},
        metrics={"stage": "development"},
    )
    assert sealed_checkpoint == checkpoint
    assert set(payloads) == set(artifacts.DEVELOPMENT_PAYLOAD_NAMES) - {
        "development_runtime_cost_evidence.json",
        "report.json",
    }
    for filename, payload in payloads.items():
        if filename.endswith(".table.json"):
            stage._validate_table_payload(filename, payload)


def test_confirmation_payloads_are_suffix_models_with_continuous_full_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_frame = _small_development_frame()
    suffix_frame = _small_confirmation_suffix()
    confirmation_frame = pd.concat([development_frame, suffix_frame])
    start = development_frame.index[0].date().isoformat()
    monkeypatch.setattr(
        artifacts,
        "SOURCE_START_SESSION_BY_STAGE",
        {"development": start, "confirmation": start},
    )
    monkeypatch.setattr(
        artifacts,
        "SOURCE_SESSION_COUNT_BY_STAGE",
        {
            "development": len(development_frame),
            "confirmation": len(confirmation_frame),
        },
    )

    development_replays, _ = stage._development_replays(development_frame)
    development_ledgers, development_accounts = stage._run_development_ledgers(
        snapshot=SimpleNamespace(frame=development_frame),
        replays=development_replays,
    )
    development_checkpoint_value = artifacts.build_composite_stage_checkpoint(
        stage="development",
        replay_checkpoints={
            arm: development_replays[arm].checkpoint
            for arm in artifacts.ARM_ORDER
        },
        administrative_accounts=development_accounts,
    )
    development_checkpoint = artifacts.parse_composite_stage_checkpoint(
        development_checkpoint_value
    )
    forks = replay.fork_confirmation_arms(
        suffix_frame,
        development_replays["online_full"].checkpoint,
        historical_prefix=development_frame,
    )
    confirmation_replays = dict(forks.arms)

    hashes: dict[str, str] = {}
    for cost in artifacts.COST_ORDER:
        for policy in artifacts.POLICY_ORDER:
            filename = f"development_ledger__{cost}__{policy}.table.json"
            payload = stage.ledger_table_bytes(development_ledgers[cost][policy])
            (tmp_path / filename).write_bytes(payload)
            hashes[filename] = stage._experiment.sha256_bytes(payload)
    bundle = stage._experiment.VerifiedBundle(
        directory=tmp_path,
        manifest_path=tmp_path / "stage_manifest.json",
        manifest={},
        payload_sha256=hashes,
        checksums={},
    )
    ledgers, accounts = stage._run_confirmation_ledgers(
        development_bundle=bundle,
        development_snapshot=SimpleNamespace(frame=development_frame),
        confirmation_snapshot=SimpleNamespace(frame=confirmation_frame),
        development_replays=development_replays,
        confirmation_replays=confirmation_replays,
        development_checkpoint=development_checkpoint,
    )
    episodes, xors = stage._extract_episodes_and_xors(ledgers)
    digest = "sha256:" + "c" * 64
    computation = stage.StageComputation(
        stage="confirmation",
        snapshot=SimpleNamespace(
            provenance={"source": "synthetic"},
            canonical_csv_bytes=b"synthetic authorized prices\n",
        ),
        replays=confirmation_replays,
        ledgers=ledgers,
        accounts=accounts,
        episodes=episodes,
        xors=xors,
        parent_causal_prefix_proof={"proof_sha256": digest},
        source_bundle_provenance={"provenance_sha256": digest},
        replay_diagnostics=stage._replay_diagnostics_payload(
            stage="confirmation", replays=confirmation_replays, extra={}
        ),
        prefix_continuity_proof={"proof_sha256": digest},
        confirmation_authorization={"development_manifest_sha256": digest},
        development_parent_manifest_bytes=b"{}\n",
        development_parent_checksums_bytes=b"{}\n",
    )
    integrity = {
        name: True for name in stage._integrity_inventory("confirmation")
    }
    payloads, checkpoint = stage._base_payloads(
        computation,
        integrity=integrity,
        gate_report={"passed": False},
        metrics={"stage": "confirmation"},
    )
    assert set(payloads) == set(artifacts.CONFIRMATION_PAYLOAD_NAMES) - {
        "confirmation_runtime_cost_evidence.json",
        "report.json",
    }
    parsed_checkpoint = artifacts.parse_composite_stage_checkpoint(checkpoint)
    assert parsed_checkpoint.arm_seed_equivalence is None
    suffix_forecast = artifacts.parse_canonical_table_bytes(
        payloads["confirmation_forecast__online_full.table.json"],
        schema=stage.FORECAST_SCHEMA,
    )
    assert suffix_forecast.index.equals(
        suffix_frame.index.rename("decision_date")
    )
    full_ledger = stage.parse_ledger_table_bytes(
        payloads["confirmation_ledger__base_5bps__online_full.table.json"]
    )
    assert full_ledger.iloc[0]["fill_date"] == "2005-01-03"
    assert full_ledger.iloc[-1]["fill_date"] == "2023-12-29"
    for filename, payload in payloads.items():
        if filename.endswith(".table.json"):
            stage._validate_table_payload(filename, payload)


def test_parent_causal_projection_requires_exact_dates_booleans_and_finiteness() -> None:
    dates = ["2019-01-02", "2019-01-03"]
    raw: dict[str, object] = {"decision_date": dates}
    types = dict(
        zip(
            stage.PARENT_CAUSAL_PROJECTION_COLUMNS,
            stage.PARENT_CAUSAL_PROJECTION_SCHEMA.column_types,
        )
    )
    for name in replay.FIXED_CAUSAL_SIGNAL_COLUMNS:
        raw[name] = [False, True] if types[name] == "bool" else [0.1, 0.2]
    raw["contextual_prior_intraday_percentile"] = [np.nan, 0.2]
    frame = pd.DataFrame(raw)
    expected = pd.DatetimeIndex(dates, name="decision_date")
    projected = stage._strict_parent_causal_projection(
        frame, expected_index=expected
    )
    assert tuple(projected.columns) == stage.PARENT_CAUSAL_PROJECTION_COLUMNS
    assert np.isnan(projected.iloc[0]["contextual_prior_intraday_percentile"])
    assert b"NaN" not in stage.table_bytes(
        projected, schema=stage.PARENT_CAUSAL_PROJECTION_SCHEMA
    )

    bad = frame.copy()
    bad["contextual_ready"] = [1, 0]
    with pytest.raises(Error, match="boolean"):
        stage._strict_parent_causal_projection(bad, expected_index=expected)
    bad = frame.copy()
    bad["contextual_prior_intraday_percentile"] = [np.inf, 0.2]
    with pytest.raises(Error, match="infinite"):
        stage._strict_parent_causal_projection(bad, expected_index=expected)
    with pytest.raises(Error, match="date sequence"):
        stage._strict_parent_causal_projection(
            frame, expected_index=pd.DatetimeIndex(dates[::-1], name="decision_date")
        )


def test_comparator_projection_independently_enforces_union_gating() -> None:
    index = pd.DatetimeIndex(
        ["2006-02-14", "2006-02-15", "2006-02-16"], name="decision_date"
    )
    fixed = pd.DataFrame(
        {
            "unfiltered_union_signal": [True, False, True],
            "contextual_virtual_signal": [False, True, False],
            "weak_trend_virtual_signal": [True, False, True],
            "fixed_always_long_target_exposure": [1.0, 1.0, 1.0],
            "fixed_union_cash_target_exposure": [0.0, 1.0, 0.0],
            "fixed_contextual_only_target_exposure": [1.0, 1.0, 1.0],
            "fixed_weak_trend_only_target_exposure": [0.0, 1.0, 0.0],
        },
        index=index,
    )
    observed, expected, invariants = stage._derived_comparator_projection(fixed)
    assert stage.table_bytes(
        observed, schema=stage.COMPARATOR_PROJECTION_SCHEMA
    ) == stage.table_bytes(expected, schema=stage.COMPARATOR_PROJECTION_SCHEMA)
    assert all(invariants.values())

    resurrected = fixed.copy()
    resurrected.loc[
        pd.Timestamp("2006-02-15"), "fixed_contextual_only_target_exposure"
    ] = 0.0
    with pytest.raises(Error, match="union-gated contract"):
        stage._derived_comparator_projection(resurrected)

    nonbinary = fixed.copy()
    nonbinary.iloc[0, nonbinary.columns.get_loc(
        "fixed_union_cash_target_exposure"
    )] = 0.5
    with pytest.raises(Error, match="not binary"):
        stage._derived_comparator_projection(nonbinary)


def test_parent_causal_proof_excludes_legacy_comparator_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = pd.DatetimeIndex(
        ["2006-02-14", "2006-02-15"], name="decision_date"
    )
    types = dict(
        zip(
            stage.PARENT_CAUSAL_PROJECTION_COLUMNS,
            stage.PARENT_CAUSAL_PROJECTION_SCHEMA.column_types,
        )
    )
    causal: dict[str, object] = {}
    for name in stage.PARENT_CAUSAL_PROJECTION_COLUMNS:
        causal[name] = [False, False] if types[name] == "bool" else [0.1, 0.2]
    causal.update(
        {
            "contextual_raw_signal": [False, True],
            "contextual_virtual_signal": [False, True],
            "weak_trend_raw_signal": [True, False],
            "weak_trend_virtual_signal": [True, False],
            "unfiltered_union_candidate_signal": [True, True],
            "unfiltered_union_signal": [True, False],
            "unfiltered_union_signal_blocked": [False, True],
        }
    )
    fixed = pd.DataFrame(causal, index=index)
    fixed["fixed_always_long_target_exposure"] = [1.0, 1.0]
    fixed["fixed_union_cash_target_exposure"] = [0.0, 1.0]
    fixed["fixed_contextual_only_target_exposure"] = [1.0, 1.0]
    fixed["fixed_weak_trend_only_target_exposure"] = [0.0, 1.0]

    parent = fixed.loc[:, list(stage.PARENT_CAUSAL_PROJECTION_COLUMNS)].copy()
    parent.insert(0, "decision_date", [value.date().isoformat() for value in index])
    # These legacy targets deliberately disagree on the union-suppressed
    # second row. V2 verifies their source file hash but never treats them as
    # current comparator truth.
    parent["always_long_target_exposure"] = [1.0, 1.0]
    parent["unfiltered_union_target_exposure"] = [0.0, 1.0]
    parent["unfiltered_contextual_target_exposure"] = [1.0, 0.0]
    parent["unfiltered_weak_trend_target_exposure"] = [0.0, 1.0]
    filename = stage.PARENT_FORECAST_BY_STAGE["development"]
    parent_bytes = parent.to_csv(index=False, lineterminator="\n").encode("utf-8")
    (tmp_path / filename).write_bytes(parent_bytes)
    digest = stage._experiment.sha256_bytes(parent_bytes)
    source = SimpleNamespace(
        directory=tmp_path,
        payload_sha256={filename: digest},
        manifest={"manifest_sha256": "sha256:" + "a" * 64},
    )
    monkeypatch.setattr(
        stage._experiment,
        "verify_frozen_bundle_identity",
        lambda repo_root, identity: source,
    )
    monkeypatch.setattr(
        stage,
        "_source_bundle_provenance",
        lambda repo_root, *, stage: {"provenance_sha256": "sha256:" + "b" * 64},
    )

    proof, _ = stage._parent_causal_prefix_proof(
        tmp_path, stage="development", fixed_features=fixed
    )
    assert proof["proof_schema_version"] == 2
    assert proof["causal_exact"]
    assert proof["comparator_exact"]
    assert proof["legacy_parent_comparator_targets_used"] is False
    assert all(proof["comparator_invariants"].values())
    assert (
        proof["parent_causal_projection_sha256"]
        == proof["generated_causal_projection_sha256"]
    )
    assert (
        proof["expected_comparator_projection_sha256"]
        == proof["generated_comparator_projection_sha256"]
    )

    tampered = fixed.copy()
    tampered.iloc[0, tampered.columns.get_loc("aapl_intraday_return")] += 0.01
    with pytest.raises(Error, match="causal signals differ"):
        stage._parent_causal_prefix_proof(
            tmp_path, stage="development", fixed_features=tampered
        )


def test_private_hook_rejects_wrong_inventory_before_reading_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(Error, match="inventory"):
        stage.validate_private_stage_bundle(
            tmp_path / "missing",
            stage="development",
            expected_payloads={"report.json": b"{}\n"},
        )


def test_development_parent_read_is_bound_to_authorized_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_verify(directory: Path, **kwargs: object) -> object:
        captured["directory"] = directory
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(stage._experiment, "verify_exact_bundle", fake_verify)
    manifest_sha256 = "sha256:" + "a" * 64
    result = stage._read_development_bundle(
        tmp_path, expected_manifest_sha256=manifest_sha256
    )
    assert result is sentinel
    assert captured["directory"] == (
        tmp_path / artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
    )
    assert captured["expected_manifest_sha256"] == manifest_sha256
    assert captured["require_stage_pass"] is True


def test_run_stage_uses_only_frozen_output_and_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_names = set(artifacts.DEVELOPMENT_PAYLOAD_NAMES)
    computation = SimpleNamespace(stage="development")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        stage._bootstrap,
        "require_active_attestation",
        lambda **kwargs: {},
    )

    monkeypatch.setattr(
        stage._experiment,
        "clean_git_identity",
        lambda root: {"branch": "frozen"},
    )
    monkeypatch.setattr(stage, "_ensure_output_parent", lambda root: root / artifacts.OUTPUT_PARENT)
    monkeypatch.setattr(stage, "_compute_development", lambda root, trace: computation)
    monkeypatch.setattr(
        stage,
        "_gate_and_metrics",
        lambda value: ({"proof": True}, {"passed": False, "failed_checks": ["x"]}, {}),
    )

    base_payloads = {
        name: b"{}\n"
        for name in expected_names
        if name not in {"development_runtime_cost_evidence.json", "report.json"}
    }
    monkeypatch.setattr(
        stage,
        "_base_payloads",
        lambda *args, **kwargs: (dict(base_payloads), {"checkpoint_sha256": "sha256:" + "1" * 64}),
    )
    monkeypatch.setattr(
        stage,
        "_report",
        lambda **kwargs: {"status": "REJECTED"},
    )
    monkeypatch.setattr(stage, "_manifest_fields", lambda **kwargs: {"frozen": True})

    def fake_seal(final, **kwargs):
        captured["final"] = final
        captured["payloads"] = kwargs["payloads"]
        captured["expected"] = kwargs["expected_payload_names"]
        return SimpleNamespace(
            directory=final,
            manifest={"manifest_sha256": "sha256:" + "2" * 64},
        )

    monkeypatch.setattr(stage._experiment, "seal_exact_bundle", fake_seal)
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    result = stage.run_stage(
        "development", repo_root=tmp_path, clock=lambda: next(ticks)
    )
    assert captured["final"] == tmp_path / artifacts.OUTPUT_DIRECTORY_BY_STAGE["development"]
    assert set(captured["payloads"]) == expected_names
    assert set(captured["expected"]) == expected_names
    assert result["run_id"] == artifacts.RUN_ID_BY_STAGE["development"]
    assert result["stage_pass"] is False


def test_run_stage_requires_isolated_bootstrap_before_git_or_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stage._experiment,
        "clean_git_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Git must not be reached")
        ),
    )
    with pytest.raises(
        stage._bootstrap.BootstrapSecurityError,
        match="isolated bootstrap",
    ):
        stage.run_stage("development", repo_root=tmp_path)


def test_main_returns_two_for_authentic_rejection_and_accepts_no_path_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        stage,
        "run_stage",
        lambda selected: {
            "stage": selected,
            "stage_pass": False,
            "status": "REJECTED",
        },
    )
    assert stage.main(["development"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "REJECTED"
