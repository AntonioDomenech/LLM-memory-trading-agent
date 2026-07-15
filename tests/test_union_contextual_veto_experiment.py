from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.chronological_exhaustion_expert import (
    canonicalize_one_session_signals,
)
from agent_benchmark.deterministic_aapl import EvaluationPeriod
import agent_benchmark.union_contextual_veto as veto_model
import agent_benchmark.union_contextual_veto_experiment as experiment


def _market_frame(
    periods: int = 180, *, start: str = "2000-01-03"
) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods)
    aapl = np.linspace(140.0, 90.0, periods)
    spy = np.linspace(300.0, 220.0, periods)
    qqq = np.linspace(240.0, 160.0, periods)
    return pd.DataFrame(
        {
            "aapl_open": aapl.copy(),
            "aapl_close": aapl,
            "aapl_adj_close": aapl,
            "spy_adj_close": spy,
            "qqq_adj_close": qqq,
        },
        index=index,
    )


def _forecast_with_stream(
    frame: pd.DataFrame,
    *,
    raw_positions: list[int],
    model_veto_positions: list[int] | None = None,
) -> pd.DataFrame:
    """Return a contract-valid core forecast with an explicit synthetic stream."""

    forecast = veto_model.build_union_contextual_veto_forecast(
        frame, learning_mode=veto_model.CAUSAL_ONLINE_MODE
    )
    attrs = copy.deepcopy(forecast.attrs)
    raw = pd.Series(False, index=forecast.index, dtype=bool)
    # Forty isolated, pre-evaluation opportunities create a causal warmup state
    # for tests that need a true veto prediction. They are masked by each test's
    # administrative start and therefore never enter its execution ledger.
    warmup_positions = list(range(2, 82, 2))
    raw.iloc[sorted(set(warmup_positions + raw_positions))] = True
    model_veto = pd.Series(False, index=forecast.index, dtype=bool)
    model_veto.iloc[model_veto_positions or []] = True
    assert not bool((model_veto & ~raw).any())

    forecast["contextual_virtual_signal"] = raw
    forecast["weak_trend_virtual_signal"] = False
    forecast["union_candidate_signal"] = raw
    continuous_union = canonicalize_one_session_signals(raw)
    forecast["canonical_union_cash_signal"] = continuous_union
    available = pd.Series(False, index=forecast.index, dtype=bool)
    if len(available) > 2:
        available.iloc[:-2] = True
    forecast["stage_outcome_available"] = available

    for name in experiment.FEATURE_COLUMNS:
        forecast.loc[raw, name] = 0.0
    forecast.loc[raw, "feature_intercept"] = 1.0

    # _stage_targets audits the one-to-one shadow opportunity accounting. The
    # synthetic stream schedules each resolved continuous opportunity at t+2
    # and marks a tail opportunity pending, exactly as the core contract does.
    forecast["shadow_matures_on_close"] = pd.NaT
    forecast["shadow_pending"] = continuous_union & ~available
    forecast["shadow_matured_now"] = continuous_union.shift(
        2, fill_value=False
    ).astype(bool)
    forecast["shadow_signal_close"] = pd.NaT
    forecast["shadow_label_10bps"] = np.nan
    for position in np.flatnonzero(continuous_union.to_numpy(dtype=bool)):
        if bool(available.iloc[position]):
            forecast.iloc[
                position,
                forecast.columns.get_loc("shadow_matures_on_close"),
            ] = forecast.index[position + 2]
            forecast.iloc[
                position + 2,
                forecast.columns.get_loc("shadow_signal_close"),
            ] = forecast.index[position]
            label = math.log(
                frame["aapl_open"].iloc[position + 1]
                / frame["aapl_open"].iloc[position + 2]
            ) + math.log(0.999 / 1.001)
            forecast.iloc[
                position + 2,
                forecast.columns.get_loc("shadow_label_10bps"),
            ] = label
    lesson_added = forecast["shadow_matured_now"].astype(bool)
    forecast["shadow_lesson_added_now"] = lesson_added
    n_raw = lesson_added.astype(int).cumsum()
    forecast["n_raw"] = n_raw
    forecast["n_eff"] = n_raw.astype(float)
    forecast["model_ready"] = (n_raw >= 40) & (n_raw >= 30)
    forecast.loc[raw, "model_mu"] = 0.0
    forecast.loc[raw, "model_se"] = 0.01
    forecast.loc[raw, "model_upper"] = 0.002
    forecast.loc[model_veto, "model_upper"] = -0.002
    expected_model_veto = (
        raw
        & forecast["model_ready"].astype(bool)
        & (forecast["model_upper"].astype(float) < -0.001)
    )
    assert expected_model_veto.equals(model_veto)
    forecast["model_veto_prediction"] = model_veto
    forecast["veto"] = continuous_union & model_veto
    learner_cash = continuous_union & ~model_veto
    forecast["learner_cash_signal"] = learner_cash
    forecast["target_exposure"] = np.where(learner_cash, 0.0, 1.0)
    forecast.attrs = attrs
    return forecast


def _passing_integrity() -> dict[str, bool]:
    return {
        "passed": True,
        "learner_cash_subset_of_union": True,
        "pure_veto_identity": True,
        "frozen_veto_rule_exact": True,
        "union_cooldown_applied_before_veto": True,
        "shadow_count_equality": True,
        "shadow_maturity_and_label_formula_exact": True,
        "causal_lesson_admission_exact": True,
        "same_action_stream_all_costs": True,
        "all_policy_ledgers_unleveraged": True,
        "episode_and_veto_edge_identity": True,
    }


def _always_long_result() -> dict[str, object]:
    return {
        "total_active_log_edge": 0.0,
        "comparison": {"relative_wealth_vs_aapl_buy_hold": 0.0},
    }


def _veto_summary(*, count: int = 10) -> dict[str, float | int]:
    return {
        "veto_count": count,
        "beneficial_veto_rate": 0.75,
        "mean_veto_benefit": 0.01,
        "median_veto_benefit": 0.01,
        "maximum_positive_veto_share": 0.20,
        "total_veto_benefit": 0.10,
    }


def _development_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    annual = {
        str(year): {"active_log_edge": 0.01, "aapl_buy_hold_return": 0.0}
        for year in range(2005, 2019)
    }
    folds = {
        f"{year}_{year + 1}": {
            "active_log_edge": 0.02,
            "aapl_buy_hold_return": 0.0,
        }
        for year in range(2005, 2019, 2)
    }
    incremental_folds = {
        f"{year}_{year + 1}": 0.01 for year in range(2005, 2019, 2)
    }
    for cost_name, _ in experiment.COST_SCENARIOS:
        metrics[cost_name] = {
            "learner": {
                "total_active_log_edge": 0.14,
                "cash_episode_count": 20,
                "cash_episode_win_rate": 0.75,
                "mean_cash_episode_edge": 0.01,
                "median_cash_episode_edge": 0.01,
                "maximum_positive_episode_share": 0.20,
                "comparison": {"relative_wealth_vs_aapl_buy_hold": 0.10},
                "periods": {**copy.deepcopy(annual), **copy.deepcopy(folds)},
            },
            "union": {
                "cash_episode_count": experiment.UNION_REFERENCE[cost_name][
                    "episodes"
                ],
                "total_active_log_edge": experiment.UNION_REFERENCE[cost_name][
                    "total_active_log_edge"
                ],
            },
            "always_long": _always_long_result(),
            "learner_vs_union": {
                "total_active_log_edge": 0.02,
                "periods": dict(incremental_folds),
                "veto_benefit": _veto_summary(),
            },
        }
    return metrics


def _validation_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    annual_edges = [0.03, 0.02, 0.01, -0.002, -0.002]
    incremental_edges = [0.001, 0.001, 0.0, 0.0, 0.0]
    for cost_name, _ in experiment.COST_SCENARIOS:
        periods = {
            str(year): {
                "active_log_edge": annual_edges[offset],
                "aapl_buy_hold_return": -0.10 if year == 2020 else 0.10,
            }
            for offset, year in enumerate(range(2019, 2024))
        }
        metrics[cost_name] = {
            "learner": {
                "total_active_log_edge": sum(annual_edges),
                "cash_episode_count": 5,
                "mean_cash_episode_edge": 0.01,
                "median_cash_episode_edge": 0.01,
                "maximum_positive_episode_share": 0.40,
                "periods": periods,
            },
            "always_long": _always_long_result(),
            "learner_vs_union": {
                "total_active_log_edge": 0.002,
                "periods": {
                    str(year): incremental_edges[offset]
                    for offset, year in enumerate(range(2019, 2024))
                },
                "veto_benefit": _veto_summary(count=3),
            },
        }
    return metrics


def _final_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    for cost_name, _ in experiment.COST_SCENARIOS:
        metrics[cost_name] = {
            "learner": {
                "periods": {
                    name: {"active_log_edge": 0.0011}
                    for name in ("2024", "2025", "2026_ytd")
                }
            },
            "always_long": _always_long_result(),
            "learner_vs_union": {
                "total_active_log_edge": 0.0002,
                "periods": {"2024": 0.0, "2025": 0.0, "2026_ytd": -0.01},
            },
        }
    return metrics


def test_stage_masks_before_cooldown_and_uses_boundary_model_prediction():
    frame = _market_frame()
    start_position = 110
    forecast = _forecast_with_stream(
        frame,
        raw_positions=[start_position - 1, start_position],
        model_veto_positions=[start_position],
    )
    start = frame.index[start_position]

    # The continuous stream consumes the pre-boundary opportunity and therefore
    # has no veto at the boundary, even though the model prediction is true.
    assert bool(forecast.iloc[start_position - 1]["canonical_union_cash_signal"])
    assert not bool(forecast.iloc[start_position]["canonical_union_cash_signal"])
    assert bool(forecast.iloc[start_position]["model_veto_prediction"])
    assert not bool(forecast.iloc[start_position]["veto"])

    targets, integrity = experiment._stage_targets(
        frame, forecast, administrative_start=start
    )

    # Administrative masking happens before cooldown. The boundary candidate
    # becomes the stage union episode and is vetoed from model_veto_prediction,
    # not from the different continuous-veto stream.
    assert targets["union"].iloc[start_position] == 0.0
    assert targets["learner"].iloc[start_position] == 1.0
    assert integrity["stage_union_cash_count"] == 1
    assert integrity["stage_veto_count"] == 1


def test_unresolved_tail_is_masked_and_a_false_mask_is_rejected():
    frame = _market_frame()
    unresolved = len(frame) - 2
    forecast = _forecast_with_stream(frame, raw_positions=[unresolved])

    targets, _ = experiment._stage_targets(
        frame, forecast, administrative_start=frame.index[20]
    )
    assert bool(forecast.iloc[unresolved]["canonical_union_cash_signal"])
    assert not bool(forecast.iloc[unresolved]["stage_outcome_available"])
    assert targets["union"].iloc[unresolved] == 1.0
    assert targets["learner"].iloc[unresolved] == 1.0

    tampered = forecast.copy()
    tampered.attrs = copy.deepcopy(forecast.attrs)
    tampered.iloc[
        unresolved, tampered.columns.get_loc("stage_outcome_available")
    ] = True
    with pytest.raises(experiment.UnionContextualVetoExperimentError, match="tail"):
        experiment._stage_targets(
            frame, tampered, administrative_start=frame.index[20]
        )


def test_pure_veto_subset_does_not_recanonicalize_and_resurrect_t_plus_one():
    frame = _market_frame()
    first = 110
    forecast = _forecast_with_stream(
        frame,
        raw_positions=[first, first + 1],
        model_veto_positions=[first],
    )

    targets, integrity = experiment._stage_targets(
        frame, forecast, administrative_start=frame.index[20]
    )

    assert targets["union"].iloc[first] == 0.0
    assert targets["learner"].iloc[first] == 1.0
    assert targets["union"].iloc[first + 1] == 1.0
    assert targets["learner"].iloc[first + 1] == 1.0
    learner_cash = targets["learner"].eq(0.0)
    union_cash = targets["union"].eq(0.0)
    assert not bool((learner_cash & ~union_cash).any())
    assert integrity["learner_cash_subset_of_union"] is True
    assert integrity["union_cooldown_applied_before_veto"] is True


def test_veto_benefit_exactly_reconciles_incremental_policy_edge():
    frame = _market_frame()
    decision = 110
    forecast = _forecast_with_stream(
        frame, raw_positions=[decision], model_veto_positions=[decision]
    )
    period = EvaluationPeriod(
        "synthetic",
        frame.index[100].date().isoformat(),
        frame.index[140].date().isoformat(),
    )

    metrics, _, _, benefits, integrity = experiment._evaluate_policy_set(
        frame,
        forecast,
        periods=(period,),
        administrative_start=frame.index[100],
    )

    for cost_name, cost_bps in experiment.COST_SCENARIOS:
        rows = benefits[cost_name]
        assert len(rows) == 1
        cost = cost_bps / 10_000.0
        expected = -(
            math.log(
                frame["aapl_open"].iloc[decision + 1]
                / frame["aapl_open"].iloc[decision + 2]
            )
            + math.log((1.0 - cost) / (1.0 + cost))
        )
        assert rows.iloc[0]["veto_benefit"] == pytest.approx(expected)
        incremental = metrics[cost_name]["learner_vs_union"]
        assert incremental["total_active_log_edge"] == pytest.approx(expected)
        assert incremental["periods"]["synthetic"] == pytest.approx(expected)
        assert incremental["veto_benefit"]["total_veto_benefit"] == pytest.approx(
            expected
        )
        assert abs(incremental["veto_benefit_identity_error"]) <= 1e-10
    assert integrity["episode_and_veto_edge_identity"] is True
    assert integrity["all_policy_ledgers_unleveraged"] is True


def test_development_passes_only_with_the_frozen_union_references():
    report = experiment.apply_development_gates(
        _development_metrics(), _passing_integrity()
    )
    assert report["passed"] is True
    for cost_name, _ in experiment.COST_SCENARIOS:
        assert report["gates"][f"{cost_name}_union_reference_episode_count"]
        assert report["gates"][f"{cost_name}_union_reference_active_log_edge"]


@pytest.mark.parametrize("cost_name", ["base_5bps", "stress_10bps"])
def test_development_rejects_any_material_union_reference_change(cost_name: str):
    wrong_count = _development_metrics()
    wrong_count[cost_name]["union"]["cash_episode_count"] -= 1
    count_report = experiment.apply_development_gates(
        wrong_count, _passing_integrity()
    )
    assert not count_report["gates"][
        f"{cost_name}_union_reference_episode_count"
    ]
    assert count_report["passed"] is False

    wrong_edge = _development_metrics()
    wrong_edge[cost_name]["union"]["total_active_log_edge"] += 1e-9
    edge_report = experiment.apply_development_gates(
        wrong_edge, _passing_integrity()
    )
    assert not edge_report["gates"][
        f"{cost_name}_union_reference_active_log_edge"
    ]
    assert edge_report["passed"] is False


def test_validation_gates_are_causal_online_and_strict_at_union_threshold():
    metrics = _validation_metrics()
    report = experiment.apply_validation_gates(metrics, _passing_integrity())

    assert report["passed"] is True
    assert report["primary_evidence"] == "causal_online"
    assert report["frozen_diagnostic_can_rescue"] is False

    for cost_name, _ in experiment.COST_SCENARIOS:
        boundary = copy.deepcopy(metrics)
        boundary[cost_name]["learner_vs_union"]["total_active_log_edge"] = (
            experiment.STRICT_UNION_IMPROVEMENT
        )
        failed = experiment.apply_validation_gates(
            boundary, _passing_integrity()
        )
        assert not failed["gates"][
            f"{cost_name}_learner_beats_union_by_more_than_0001"
        ]
        assert failed["passed"] is False


def test_final_gates_use_strict_material_edges_and_inclusive_nonnegative_count():
    metrics = _final_metrics()
    report = experiment.apply_final_gates(metrics, _passing_integrity())

    assert report["passed"] is True
    assert report["primary_evidence"] == "causal_online"
    assert report["frozen_or_lifetime_diagnostic_can_rescue"] is False
    for cost_name, _ in experiment.COST_SCENARIOS:
        assert report["gates"][
            f"{cost_name}_minimum_two_nonnegative_incremental_periods"
        ]

        material_boundary = copy.deepcopy(metrics)
        material_boundary[cost_name]["learner"]["periods"]["2025"][
            "active_log_edge"
        ] = experiment.FINAL_MATERIAL_ACTIVE_LOG_EDGE
        material_report = experiment.apply_final_gates(
            material_boundary, _passing_integrity()
        )
        assert not material_report["gates"][
            f"{cost_name}_2025_active_log_edge_above_001"
        ]

        union_boundary = copy.deepcopy(metrics)
        union_boundary[cost_name]["learner_vs_union"]["total_active_log_edge"] = (
            experiment.STRICT_UNION_IMPROVEMENT
        )
        union_report = experiment.apply_final_gates(
            union_boundary, _passing_integrity()
        )
        assert not union_report["gates"][
            f"{cost_name}_continuous_learner_beats_union_by_more_than_0001"
        ]

        one_nonnegative = copy.deepcopy(metrics)
        one_nonnegative[cost_name]["learner_vs_union"]["periods"] = {
            "2024": -0.01,
            "2025": 0.0,
            "2026_ytd": -0.01,
        }
        count_report = experiment.apply_final_gates(
            one_nonnegative, _passing_integrity()
        )
        assert not count_report["gates"][
            f"{cost_name}_minimum_two_nonnegative_incremental_periods"
        ]


@pytest.mark.parametrize(
    ("function_name", "manifest_keyword"),
    [
        ("run_validation", "development_manifest"),
        ("run_final", "validation_manifest"),
    ],
)
def test_invalid_parent_is_rejected_before_any_later_price_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    function_name: str,
    manifest_keyword: str,
):
    loader_called = False

    monkeypatch.setattr(experiment, "_clean_git_identity", lambda _root: {})

    def reject_parent(**_kwargs):
        raise experiment.UnionContextualVetoExperimentError("parent rejected")

    def forbidden_loader(*_args, **_kwargs):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("later market loader must not run")

    monkeypatch.setattr(experiment, "_validated_prior_manifest", reject_parent)
    monkeypatch.setattr(experiment, "load_bounded_prices", forbidden_loader)
    kwargs = {
        "repo_root": tmp_path,
        "price_artifact": tmp_path / "must-not-open.csv",
        manifest_keyword: tmp_path / "invalid-parent.json",
        "output_dir": tmp_path / "out",
        "run_id": "synthetic-parent-rejection",
    }

    with pytest.raises(
        experiment.UnionContextualVetoExperimentError, match="parent rejected"
    ):
        getattr(experiment, function_name)(**kwargs)
    assert loader_called is False


def test_checkpoint_continuity_is_exact_and_any_tamper_is_rejected(tmp_path: Path):
    frame = _market_frame()
    forecast = veto_model.build_union_contextual_veto_forecast(
        frame, learning_mode=veto_model.CAUSAL_ONLINE_MODE
    )
    cutoff = frame.index[-1]
    checkpoint = experiment._checkpoint_from_forecast(
        forecast,
        cutoff=cutoff,
        learning_mode=veto_model.CAUSAL_ONLINE_MODE,
    )
    parent_manifest = tmp_path / "stage_manifest.json"
    checkpoint_name = "checkpoint.json"
    checkpoint_path = tmp_path / checkpoint_name
    checkpoint_path.write_bytes(experiment._pretty_json_bytes(checkpoint))

    experiment._require_checkpoint_continuity(
        forecast,
        cutoff=cutoff,
        parent_manifest_path=parent_manifest,
        checkpoint_filename=checkpoint_name,
    )

    tampered = copy.deepcopy(checkpoint)
    tampered["sufficient_state"]["state_vector"]["state_b_0"] += 1.0
    checkpoint_path.write_bytes(experiment._pretty_json_bytes(tampered))
    with pytest.raises(
        experiment.UnionContextualVetoExperimentError,
        match="does not match",
    ):
        experiment._require_checkpoint_continuity(
            forecast,
            cutoff=cutoff,
            parent_manifest_path=parent_manifest,
            checkpoint_filename=checkpoint_name,
        )

    with pytest.raises(
        experiment.UnionContextualVetoExperimentError,
        match="physically bounded",
    ):
        experiment._checkpoint_from_forecast(
            forecast,
            cutoff=frame.index[-2],
            learning_mode=veto_model.CAUSAL_ONLINE_MODE,
        )


def test_manifest_pass_claim_cannot_launder_gate_or_report_failure():
    gate = {"passed": True, "gates": {"synthetic": True}, "failures": []}
    manifest = {"stage_pass": True, "run_id": "synthetic-run"}
    report = {
        "contract_version": experiment.CONTRACT_VERSION,
        "stage": "development",
        "run_id": "synthetic-run",
        "gate_report": gate,
    }
    experiment._require_pass_evidence_consistency(
        manifest, gate, report, expected_stage="development"
    )

    failed_gate = {"passed": False, "gates": {"synthetic": False}}
    with pytest.raises(
        experiment.UnionContextualVetoExperimentError, match="do not agree"
    ):
        experiment._require_pass_evidence_consistency(
            manifest,
            failed_gate,
            {**report, "gate_report": failed_gate},
            expected_stage="development",
        )

    mismatched_report = copy.deepcopy(report)
    mismatched_report["gate_report"] = {
        "passed": True,
        "gates": {"different": True},
        "failures": [],
    }
    with pytest.raises(
        experiment.UnionContextualVetoExperimentError, match="do not agree"
    ):
        experiment._require_pass_evidence_consistency(
            manifest, gate, mismatched_report, expected_stage="development"
        )

    with pytest.raises(
        experiment.UnionContextualVetoExperimentError, match="do not agree"
    ):
        experiment._require_pass_evidence_consistency(
            {**manifest, "stage_pass": False},
            gate,
            report,
            expected_stage="development",
        )


def test_deadline_failure_before_promotion_leaves_no_partial_bundle(tmp_path: Path):
    checked: list[str] = []

    class RejectPromotion:
        def check(self, location: str) -> None:
            checked.append(location)
            raise experiment.UnionContextualVetoExperimentError(
                "synthetic deadline"
            )

    with pytest.raises(
        experiment.UnionContextualVetoExperimentError,
        match="synthetic deadline",
    ):
        experiment._stage_bundle(
            output_dir=tmp_path,
            run_id="deadline-run",
            stage="development",
            stage_pass=True,
            report={"synthetic": True},
            payloads={"payload.txt": b"synthetic\n"},
            source_provenance={
                "source_path": "synthetic.csv",
                "bounded_result_sha256": "sha256:" + "0" * 64,
            },
            git_identity={},
            parent_manifest=None,
            deadline=RejectPromotion(),
        )

    assert checked == ["before artifact promotion"]
    assert not (tmp_path / "deadline-run").exists()
    assert not list(tmp_path.glob(".deadline-run.*.sealing"))
    assert not list(tmp_path.iterdir())


def test_git_identity_hashes_committed_blob_not_crlf_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dependency = Path("dependency.py")
    contract = Path("contract.md")
    (tmp_path / dependency).write_bytes(b"line\r\n")
    (tmp_path / contract).write_bytes(b"contract\r\n")
    committed = {
        dependency.as_posix(): b"line\n",
        contract.as_posix(): b"contract\n",
    }

    def fake_git_text(_root: Path, *args: str) -> str:
        commands = {
            ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "codex/test",
            ("rev-parse", "HEAD"): "a" * 40,
        }
        return commands[args]

    def fake_git_bytes(_root: Path, *args: str) -> bytes:
        if args[0] == "ls-files":
            return (args[-1] + "\n").encode()
        if args[0] == "show":
            return committed[args[1].removeprefix("HEAD:")]
        raise AssertionError(args)

    monkeypatch.setattr(experiment, "IMPLEMENTATION_PATHS", (dependency,))
    monkeypatch.setattr(experiment, "CONTRACT_PATH", contract)
    monkeypatch.setattr(experiment, "_git_text", fake_git_text)
    monkeypatch.setattr(experiment, "_git_bytes", fake_git_bytes)

    identity = experiment._clean_git_identity(tmp_path)

    assert identity["tracked_dependency_sha256"][dependency.as_posix()] == (
        experiment._sha256(b"line\n")
    )
    assert identity["tracked_dependency_sha256"][contract.as_posix()] == (
        experiment._sha256(b"contract\n")
    )
    assert identity["tracked_dependency_sha256"][dependency.as_posix()] != (
        experiment._sha256(b"line\r\n")
    )
