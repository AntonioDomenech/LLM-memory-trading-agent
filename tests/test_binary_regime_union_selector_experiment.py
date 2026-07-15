from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_benchmark.chronological_exhaustion_expert import (
    canonicalize_one_session_signals,
)
from agent_benchmark.deterministic_aapl import EvaluationPeriod
import agent_benchmark.binary_regime_union_selector as selector
import agent_benchmark.binary_regime_union_selector_experiment as experiment


def _market_frame(
    periods: int = 800,
    *,
    start: str = "2004-01-02",
    risk_on: bool = False,
) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=periods)
    aapl = np.linspace(140.0, 90.0, periods)
    if risk_on:
        spy = np.linspace(200.0, 360.0, periods)
        qqq = np.linspace(150.0, 330.0, periods)
    else:
        spy = np.linspace(360.0, 200.0, periods)
        qqq = np.linspace(330.0, 150.0, periods)
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


def _synthetic_forecast(
    frame: pd.DataFrame,
    *,
    raw_positions: list[int],
) -> pd.DataFrame:
    """Build a core-contract forecast around an explicit synthetic raw union."""

    forecast = selector.build_binary_regime_union_selector_forecast(
        frame, learning_mode=selector.CAUSAL_ONLINE_MODE
    )
    attrs = copy.deepcopy(forecast.attrs)
    raw = pd.Series(False, index=forecast.index, dtype=bool)
    raw.iloc[raw_positions] = True
    shadow = canonicalize_one_session_signals(raw)
    available = pd.Series(False, index=forecast.index, dtype=bool)
    if len(available) > 2:
        available.iloc[:-2] = True

    forecast["contextual_virtual_signal"] = raw
    forecast["weak_trend_virtual_signal"] = False
    forecast["union_candidate_signal"] = raw
    forecast["shadow_canonical_union_signal"] = shadow
    forecast["canonical_union_cash_signal"] = shadow
    forecast["shadow_matures_on_close"] = pd.NaT
    forecast["shadow_matured_now"] = shadow.shift(2, fill_value=False).astype(bool)
    forecast["shadow_signal_close"] = pd.NaT
    forecast["shadow_opportunity_risk_on"] = False
    forecast["shadow_matured_signal_risk_on"] = False
    forecast["shadow_label_10bps"] = np.nan
    forecast["shadow_lesson_added_now"] = forecast["shadow_matured_now"]
    forecast["shadow_pending"] = shadow & ~available

    risk_on_values = forecast["risk_on"].astype(bool)
    opens = (
        frame["aapl_open"] * frame["aapl_adj_close"] / frame["aapl_close"]
    ).to_numpy(dtype=float)
    shadow_positions = np.flatnonzero(shadow.to_numpy(dtype=bool))
    for position in shadow_positions:
        forecast.iloc[
            position,
            forecast.columns.get_loc("shadow_opportunity_risk_on"),
        ] = bool(risk_on_values.iloc[position])
        if position + 2 >= len(frame):
            continue
        forecast.iloc[
            position,
            forecast.columns.get_loc("shadow_matures_on_close"),
        ] = forecast.index[position + 2]
        forecast.iloc[
            position + 2,
            forecast.columns.get_loc("shadow_signal_close"),
        ] = forecast.index[position]
        forecast.iloc[
            position + 2,
            forecast.columns.get_loc("shadow_matured_signal_risk_on"),
        ] = bool(risk_on_values.iloc[position])
        forecast.iloc[
            position + 2,
            forecast.columns.get_loc("shadow_label_10bps"),
        ] = math.log(opens[position + 1] / opens[position + 2]) + math.log(
            0.999 / 1.001
        )

    states = selector.initialize_regime_states()
    for current in range(len(forecast)):
        signal_position = current - 2
        if signal_position >= 0 and bool(shadow.iloc[signal_position]):
            signal_risk_on = bool(risk_on_values.iloc[signal_position])
            regime = (
                selector.RISK_ON_REGIME
                if signal_risk_on
                else selector.NOT_RISK_ON_REGIME
            )
            default_cash = not signal_risk_on
            states[regime] = selector.update_ew_regime_state(
                states[regime],
                float(forecast.iloc[current]["shadow_label_10bps"]),
                structural_default_cash=default_cash,
            )
        for regime in selector.REGIME_NAMES:
            state = states[regime]
            forecast.iloc[
                current, forecast.columns.get_loc(f"{regime}_n_raw")
            ] = state.n_raw
            forecast.iloc[
                current, forecast.columns.get_loc(f"{regime}_n_eff")
            ] = state.n_eff
            forecast.iloc[
                current,
                forecast.columns.get_loc(f"{regime}_weighted_label_sum"),
            ] = state.weighted_label_sum
            forecast.iloc[
                current,
                forecast.columns.get_loc(
                    f"{regime}_weighted_squared_label_sum"
                ),
            ] = state.weighted_squared_label_sum
            forecast.iloc[
                current, forecast.columns.get_loc(f"{regime}_mean")
            ] = state.mean
            forecast.iloc[
                current, forecast.columns.get_loc(f"{regime}_ready")
            ] = state.ready
            forecast.iloc[
                current, forecast.columns.get_loc(f"{regime}_cash_selected")
            ] = state.cash_selected

    forecast["selector_regime_n_eff"] = np.nan
    forecast["selector_regime_mean"] = np.nan
    forecast["selector_regime_ready"] = False
    forecast["selector_cash_prediction"] = False
    forecast["selector_skip_prediction"] = False
    for position in np.flatnonzero(raw.to_numpy(dtype=bool)):
        regime = (
            selector.RISK_ON_REGIME
            if bool(risk_on_values.iloc[position])
            else selector.NOT_RISK_ON_REGIME
        )
        state = selector.EwRegimeState(
            n_raw=int(forecast.iloc[position][f"{regime}_n_raw"]),
            n_eff=float(forecast.iloc[position][f"{regime}_n_eff"]),
            weighted_label_sum=float(
                forecast.iloc[position][f"{regime}_weighted_label_sum"]
            ),
            weighted_squared_label_sum=float(
                forecast.iloc[position][f"{regime}_weighted_squared_label_sum"]
            ),
            cash_selected=bool(
                forecast.iloc[position][f"{regime}_cash_selected"]
            ),
        )
        forecast.iloc[
            position, forecast.columns.get_loc("selector_regime_n_eff")
        ] = state.n_eff
        forecast.iloc[
            position, forecast.columns.get_loc("selector_regime_mean")
        ] = state.mean
        forecast.iloc[
            position, forecast.columns.get_loc("selector_regime_ready")
        ] = state.ready
        forecast.iloc[
            position, forecast.columns.get_loc("selector_cash_prediction")
        ] = state.cash_selected
        forecast.iloc[
            position, forecast.columns.get_loc("selector_skip_prediction")
        ] = not state.cash_selected

    cash_prediction = forecast["selector_cash_prediction"].astype(bool)
    skip_prediction = forecast["selector_skip_prediction"].astype(bool)
    forecast["selector_skip_signal"] = shadow & skip_prediction
    learner = shadow & cash_prediction
    forecast["learner_cash_signal"] = learner
    forecast["target_exposure"] = np.where(learner, 0.0, 1.0)

    pending: list[selector.PendingRegimeLesson] = []
    for position in shadow_positions:
        if position + 2 < len(frame):
            continue
        remaining = int(position + 2 - (len(frame) - 1))
        pending.append(
            selector.PendingRegimeLesson(
                signal_close=forecast.index[position],
                sessions_until_maturity=remaining,
                risk_on=bool(risk_on_values.iloc[position]),
                entry_adjusted_open=(
                    float(opens[position + 1])
                    if position + 1 < len(frame)
                    else None
                ),
                learning_eligible=True,
            )
        )
    attrs["final_states"] = selector.serialize_regime_states(states)
    attrs["pending_lessons"] = selector.serialize_pending_regime_lessons(pending)
    forecast.attrs = attrs
    return forecast


def _position_on_or_after(frame: pd.DataFrame, date: str) -> int:
    return int(frame.index.searchsorted(pd.Timestamp(date)))


def _passing_integrity() -> dict[str, object]:
    return {
        "passed": True,
        "continuous_account_start": "2005-01-01",
        "account_cooldown_reset_count_after_inception": 0,
        "account_union_derived_from_raw_after_pre_inception_removal": True,
        "unresolved_tail_masked_after_cooldown": True,
        "selector_cash_subset_of_union": True,
        "pure_union_filter_identity": True,
        "frozen_binary_regime_rule_exact": True,
        "selector_state_replay_exact": True,
        "shadow_count_equality": True,
        "shadow_maturity_and_label_formula_exact": True,
        "causal_lesson_admission_exact": True,
        "same_action_stream_all_costs": True,
        "all_policy_ledgers_unleveraged": True,
        "continuous_account_episode_identity": True,
        "reporting_episode_and_veto_edge_identity": True,
        "account_selector_cash_count": 101,
        "account_veto_count": 20,
        "regime_switch_counts": {
            "risk_on": 0,
            "not_risk_on": 0,
        },
        "final_replayed_states": {
            regime: {
                "n_raw": reference["n_raw"],
                "n_eff": reference["n_eff"],
                "mean": reference["mean"],
                "cash_selected": reference["latch"] == "CASH",
            }
            for regime, reference in experiment.CALIBRATION_REFERENCE.items()
        },
    }


def _always_long_result() -> dict[str, object]:
    return {
        "total_active_log_edge": 0.0,
        "continuous_account_active_log_edge": 0.0,
        "comparison": {"relative_wealth_vs_aapl_buy_hold": 0.0},
    }


def _veto_summary(*, count: int = 10) -> dict[str, float | int]:
    return {
        "veto_count": count,
        "beneficial_veto_rate": 0.50,
        "mean_veto_benefit": 0.01,
        "median_veto_benefit": 0.01,
        "maximum_positive_veto_share": 0.50,
        "total_veto_benefit": 0.10,
    }


def _development_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    folds = {f"{year}_{year + 1}": 0.01 for year in range(2005, 2019, 2)}
    for cost_name, _ in experiment.COST_SCENARIOS:
        selector_edge = experiment.SELECTOR_REFERENCE[cost_name]
        union_edge = experiment.UNION_REFERENCE[cost_name][
            "total_active_log_edge"
        ]
        metrics[cost_name] = {
            "selector": {"total_active_log_edge": selector_edge},
            "union": {
                "cash_episode_count": experiment.UNION_REFERENCE[cost_name][
                    "episodes"
                ],
                "total_active_log_edge": union_edge,
            },
            "always_long": _always_long_result(),
            "selector_vs_union": {
                "total_active_log_edge": selector_edge - union_edge,
                "periods": dict(folds),
                "veto_benefit": _veto_summary(),
            },
        }
    return metrics


def _validation_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    annual_edges = [0.01, 0.01, 0.01, -0.001, -0.001]
    incremental = [0.001, 0.001, 0.0, 0.0, 0.0]
    for cost_name, _ in experiment.COST_SCENARIOS:
        periods = {
            str(year): {
                "active_log_edge": annual_edges[offset],
                "aapl_buy_hold_return": -0.10 if year == 2020 else 0.10,
            }
            for offset, year in enumerate(range(2019, 2024))
        }
        metrics[cost_name] = {
            "selector": {
                "total_active_log_edge": sum(annual_edges),
                "cash_episode_count": 4,
                "mean_cash_episode_edge": 0.01,
                "median_cash_episode_edge": 0.01,
                "maximum_positive_episode_share": 0.40,
                "periods": periods,
            },
            "always_long": _always_long_result(),
            "selector_vs_union": {
                "total_active_log_edge": sum(incremental),
                "periods": {
                    str(year): incremental[offset]
                    for offset, year in enumerate(range(2019, 2024))
                },
                "veto_benefit": _veto_summary(count=3),
            },
        }
    return metrics


def _final_metrics() -> dict[str, object]:
    metrics: dict[str, object] = {}
    incremental = {"2024": 0.0001, "2025": 0.0001, "2026_ytd": -0.00005}
    for cost_name, _ in experiment.COST_SCENARIOS:
        metrics[cost_name] = {
            "selector": {
                "periods": {
                    name: {"active_log_edge": 0.0011}
                    for name in ("2024", "2025", "2026_ytd")
                }
            },
            "always_long": _always_long_result(),
            "selector_vs_union": {
                "total_active_log_edge": sum(incremental.values()),
                "periods": dict(incremental),
            },
        }
    return metrics


def test_account_inception_removes_pre_2005_candidate_before_one_cooldown():
    frame = _market_frame()
    start = _position_on_or_after(frame, "2005-01-01")
    forecast = _synthetic_forecast(
        frame, raw_positions=[start - 1, start]
    )

    # The continuous 2000+ shadow stream consumes the pre-inception signal.
    assert bool(forecast.iloc[start - 1]["shadow_canonical_union_signal"])
    assert not bool(forecast.iloc[start]["shadow_canonical_union_signal"])

    targets, integrity = experiment._stage_targets(
        frame, forecast, administrative_start=pd.Timestamp("2005-01-01")
    )

    # The account independently removes pre-inception candidates before its
    # one and only cooldown, so the first 2005 candidate is executable.
    assert targets["union"].iloc[start] == 0.0
    assert targets["selector"].iloc[start] == 0.0
    assert integrity["continuous_account_start"] == "2005-01-01"
    assert integrity["account_cooldown_reset_count_after_inception"] == 0


def test_stage_target_builder_rejects_any_second_account_start():
    frame = _market_frame()
    forecast = _synthetic_forecast(frame, raw_positions=[100])
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="begin exactly once|2005",
    ):
        experiment._stage_targets(
            frame,
            forecast,
            administrative_start=pd.Timestamp("2019-01-01"),
        )


def test_reporting_boundary_does_not_reset_union_cooldown():
    frame = _market_frame()
    boundary = _position_on_or_after(frame, "2006-01-03")
    forecast = _synthetic_forecast(
        frame, raw_positions=[boundary - 1, boundary]
    )
    targets, _ = experiment._stage_targets(
        frame, forecast, administrative_start=pd.Timestamp("2005-01-01")
    )

    assert targets["union"].iloc[boundary - 1] == 0.0
    assert targets["union"].iloc[boundary] == 1.0
    assert targets["selector"].iloc[boundary] == 1.0


def test_unresolved_tail_is_masked_after_cooldown_and_never_resurrected():
    frame = _market_frame()
    unresolved = len(frame) - 2
    forecast = _synthetic_forecast(
        frame, raw_positions=[unresolved, unresolved + 1]
    )
    targets, integrity = experiment._stage_targets(
        frame, forecast, administrative_start=pd.Timestamp("2005-01-01")
    )

    assert bool(forecast.iloc[unresolved]["shadow_canonical_union_signal"])
    assert bool(forecast.iloc[unresolved]["shadow_pending"])
    assert not bool(
        forecast.iloc[unresolved + 1]["shadow_canonical_union_signal"]
    )
    assert targets["union"].iloc[unresolved] == 1.0
    assert targets["union"].iloc[unresolved + 1] == 1.0
    assert targets["selector"].iloc[unresolved] == 1.0
    assert integrity["unresolved_tail_masked_after_cooldown"] is True


def test_risk_on_selector_is_pure_union_veto_without_recanonicalization():
    frame = _market_frame(risk_on=True)
    decision = _position_on_or_after(frame, "2005-06-01")
    forecast = _synthetic_forecast(
        frame, raw_positions=[decision, decision + 1]
    )
    targets, integrity = experiment._stage_targets(
        frame, forecast, administrative_start=pd.Timestamp("2005-01-01")
    )

    assert bool(forecast.iloc[decision]["risk_on"])
    assert bool(forecast.iloc[decision]["selector_skip_prediction"])
    assert targets["union"].iloc[decision] == 0.0
    assert targets["selector"].iloc[decision] == 1.0
    assert targets["union"].iloc[decision + 1] == 1.0
    assert targets["selector"].iloc[decision + 1] == 1.0
    selector_cash = targets["selector"].eq(0.0)
    union_cash = targets["union"].eq(0.0)
    assert not bool((selector_cash & ~union_cash).any())
    assert integrity["selector_cash_subset_of_union"] is True
    assert integrity["pure_union_filter_identity"] is True


def test_runner_independently_rejects_regime_or_state_tampering():
    frame = _market_frame(risk_on=True)
    decision = _position_on_or_after(frame, "2005-06-01")
    forecast = _synthetic_forecast(frame, raw_positions=[decision])

    wrong_regime = forecast.copy()
    wrong_regime.attrs = copy.deepcopy(forecast.attrs)
    wrong_regime.iloc[decision, wrong_regime.columns.get_loc("risk_on")] = False
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="Risk-on regime",
    ):
        experiment._stage_targets(
            frame,
            wrong_regime,
            administrative_start=pd.Timestamp("2005-01-01"),
        )

    wrong_state = forecast.copy()
    wrong_state.attrs = copy.deepcopy(forecast.attrs)
    wrong_state.iloc[
        decision, wrong_state.columns.get_loc("risk_on_n_eff")
    ] += 1.0
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="state replay mismatch",
    ):
        experiment._stage_targets(
            frame,
            wrong_state,
            administrative_start=pd.Timestamp("2005-01-01"),
        )


def test_veto_benefit_exactly_reconciles_selector_minus_union_at_both_costs():
    frame = _market_frame(risk_on=True)
    decision = _position_on_or_after(frame, "2005-06-01")
    # Make the avoided union cash episode economically non-zero.
    frame.iloc[decision + 2, frame.columns.get_loc("aapl_open")] *= 1.10
    forecast = _synthetic_forecast(frame, raw_positions=[decision])
    period = EvaluationPeriod(
        "synthetic",
        "2005-01-01",
        frame.index[-1].date().isoformat(),
    )

    metrics, ledgers, episodes, benefits, integrity = (
        experiment._evaluate_policy_set(
            frame,
            forecast,
            periods=(period,),
            administrative_start=pd.Timestamp("2005-01-01"),
        )
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
        incremental = metrics[cost_name]["selector_vs_union"]
        assert incremental["total_active_log_edge"] == pytest.approx(expected)
        assert incremental["periods"]["synthetic"] == pytest.approx(expected)
        assert incremental["veto_benefit"]["total_veto_benefit"] == pytest.approx(
            expected
        )
        assert abs(incremental["veto_benefit_identity_error"]) <= 1e-10
        assert len(episodes[cost_name]["union"]) == 1
        assert len(episodes[cost_name]["selector"]) == 0
        assert set(ledgers[cost_name]["ledger_role"]) == {
            "strategy",
            "benchmark",
        }
    assert integrity["reporting_episode_and_veto_edge_identity"] is True
    assert integrity["continuous_account_episode_identity"] is True
    assert integrity["all_policy_ledgers_unleveraged"] is True


def test_cross_year_episode_is_attributed_once_by_entry_open():
    frame = _market_frame(risk_on=False)
    decision = int(frame.index.get_loc(pd.Timestamp("2005-12-30")))
    forecast = _synthetic_forecast(frame, raw_positions=[decision])
    periods = (
        EvaluationPeriod("2005", "2005-01-01", "2005-12-31"),
        EvaluationPeriod(
            "2006_plus",
            "2006-01-01",
            frame.index[-1].date().isoformat(),
        ),
    )

    metrics, _, episodes, _, _ = experiment._evaluate_policy_set(
        frame,
        forecast,
        periods=periods,
        administrative_start=pd.Timestamp("2005-01-01"),
    )

    for cost_name, _ in experiment.COST_SCENARIOS:
        rows = episodes[cost_name]["selector"]
        assert len(rows) == 1
        assert pd.Timestamp(rows.iloc[0]["decision_date"]).year == 2005
        assert pd.Timestamp(rows.iloc[0]["entry_date"]).year == 2006
        result = metrics[cost_name]["selector"]
        assert result["periods"]["2005"]["active_log_edge"] == pytest.approx(0.0)
        assert result["periods"]["2006_plus"]["active_log_edge"] == pytest.approx(
            float(rows.iloc[0]["net_active_log_edge"])
        )


def test_continuous_ledgers_are_long_cash_only_and_always_long_matches_benchmark():
    frame = _market_frame(risk_on=False)
    first = _position_on_or_after(frame, "2005-06-01")
    second = first + 10
    forecast = _synthetic_forecast(frame, raw_positions=[first, second])
    period = EvaluationPeriod(
        "synthetic",
        "2005-01-01",
        frame.index[-1].date().isoformat(),
    )
    metrics, ledgers, _, _, integrity = experiment._evaluate_policy_set(
        frame,
        forecast,
        periods=(period,),
        administrative_start=pd.Timestamp("2005-01-01"),
    )

    for cost_name, _ in experiment.COST_SCENARIOS:
        ledger = ledgers[cost_name]
        strategies = ledger.loc[ledger["ledger_role"].eq("strategy")]
        assert (strategies["cash"] >= -1e-9).all()
        assert (strategies["shares"] >= -1e-9).all()
        assert (strategies["new_exposure_after_fill"] <= 1.0 + 1e-12).all()
        assert (strategies["new_exposure_after_fill"] >= -1e-12).all()
        always_long = metrics[cost_name]["always_long"]
        assert always_long["total_active_log_edge"] == pytest.approx(0.0)
        assert always_long["comparison"]["relative_wealth_vs_aapl_buy_hold"] == (
            pytest.approx(0.0)
        )
    assert integrity["all_policy_ledgers_unleveraged"] is True
    assert integrity["same_action_stream_all_costs"] is True


def test_development_gates_require_exact_calibration_and_label_it_training_only():
    report = experiment.apply_development_gates(
        _development_metrics(), _passing_integrity()
    )
    assert report["passed"] is True
    assert report["evidence_role"] == "calibration_training_sanity_only"
    assert report["development_performance_is_holdout_evidence"] is False
    assert report["gates"]["minimum_eight_vetoes"]
    assert report["gates"][
        "stress_10bps_beneficial_veto_rate_at_least_50pct"
    ]
    assert report["gates"]["stress_10bps_veto_benefit_not_concentrated"]


@pytest.mark.parametrize("regime", ["risk_on", "not_risk_on"])
def test_development_rejects_any_calibration_state_change(regime: str):
    integrity = _passing_integrity()
    integrity["final_replayed_states"][regime]["mean"] += 1e-5
    report = experiment.apply_development_gates(_development_metrics(), integrity)
    assert report["passed"] is False
    assert not report["gates"][f"{regime}_discounted_mean_reference"]


@pytest.mark.parametrize(
    ("field", "bad_value", "gate"),
    [
        (
            "account_veto_count",
            19,
            "training_account_exactly_twenty_vetoes",
        ),
        (
            "account_selector_cash_count",
            100,
            "training_account_exactly_101_selector_cash_actions",
        ),
    ],
)
def test_development_rejects_any_exact_training_action_count_change(
    field: str, bad_value: int, gate: str
):
    integrity = _passing_integrity()
    integrity[field] = bad_value
    report = experiment.apply_development_gates(
        _development_metrics(), integrity
    )
    assert report["passed"] is False
    assert report["gates"][gate] is False


@pytest.mark.parametrize("regime", ["risk_on", "not_risk_on"])
def test_development_rejects_any_training_latch_switch(regime: str):
    integrity = _passing_integrity()
    integrity["regime_switch_counts"][regime] = 1
    report = experiment.apply_development_gates(
        _development_metrics(), integrity
    )
    assert report["passed"] is False
    assert report["gates"][
        f"{regime}_latch_never_switched_in_training"
    ] is False


@pytest.mark.parametrize("cost_name", ["base_5bps", "stress_10bps"])
def test_development_rejects_union_or_selector_reference_change(cost_name: str):
    wrong_union = _development_metrics()
    wrong_union[cost_name]["union"]["total_active_log_edge"] += 1e-9
    union_report = experiment.apply_development_gates(
        wrong_union, _passing_integrity()
    )
    assert union_report["passed"] is False
    assert not union_report["gates"][
        f"{cost_name}_union_reference_active_log_edge"
    ]

    wrong_selector = _development_metrics()
    wrong_selector[cost_name]["selector"]["total_active_log_edge"] += 1e-9
    selector_report = experiment.apply_development_gates(
        wrong_selector, _passing_integrity()
    )
    assert selector_report["passed"] is False
    assert not selector_report["gates"][
        f"{cost_name}_selector_training_edge_reference"
    ]


def test_all_integrity_requirements_are_gate_bearing():
    metrics = _development_metrics()
    base = _passing_integrity()
    boolean_requirements = [
        "account_union_derived_from_raw_after_pre_inception_removal",
        "unresolved_tail_masked_after_cooldown",
        "selector_cash_subset_of_union",
        "pure_union_filter_identity",
        "frozen_binary_regime_rule_exact",
        "selector_state_replay_exact",
        "shadow_count_equality",
        "shadow_maturity_and_label_formula_exact",
        "causal_lesson_admission_exact",
        "same_action_stream_all_costs",
        "all_policy_ledgers_unleveraged",
        "continuous_account_episode_identity",
        "reporting_episode_and_veto_edge_identity",
    ]
    for name in boolean_requirements:
        tampered = copy.deepcopy(base)
        tampered[name] = False
        report = experiment.apply_development_gates(metrics, tampered)
        assert report["passed"] is False, name

    reset = copy.deepcopy(base)
    reset["account_cooldown_reset_count_after_inception"] = 1
    assert not experiment.apply_development_gates(metrics, reset)["passed"]


def test_validation_gates_are_online_only_and_use_strict_thresholds():
    metrics = _validation_metrics()
    report = experiment.apply_validation_gates(metrics, _passing_integrity())
    assert report["passed"] is True
    assert report["primary_evidence"] == "causal_online_selector_level_holdout"
    assert report["frozen_diagnostic_can_rescue"] is False

    for cost_name, _ in experiment.COST_SCENARIOS:
        material_boundary = copy.deepcopy(metrics)
        material_boundary[cost_name]["selector"]["total_active_log_edge"] = (
            experiment.FINAL_MATERIAL_ACTIVE_LOG_EDGE
        )
        failed = experiment.apply_validation_gates(
            material_boundary, _passing_integrity()
        )
        assert not failed["gates"][
            f"{cost_name}_aggregate_active_log_edge_above_001"
        ]

        union_boundary = copy.deepcopy(metrics)
        union_boundary[cost_name]["selector_vs_union"][
            "total_active_log_edge"
        ] = experiment.STRICT_UNION_IMPROVEMENT
        failed = experiment.apply_validation_gates(
            union_boundary, _passing_integrity()
        )
        assert not failed["gates"][
            f"{cost_name}_selector_beats_union_by_more_than_0001"
        ]


def test_validation_requires_best_year_robustness_and_negative_year_edge():
    metrics = _validation_metrics()
    concentrated = copy.deepcopy(metrics)
    for cost_name, _ in experiment.COST_SCENARIOS:
        annual = concentrated[cost_name]["selector"]["periods"]
        annual["2019"]["active_log_edge"] = 0.05
        for year in ("2020", "2021", "2022", "2023"):
            annual[year]["active_log_edge"] = -0.005
        concentrated[cost_name]["selector"]["total_active_log_edge"] = 0.03
    report = experiment.apply_validation_gates(
        concentrated, _passing_integrity()
    )
    assert report["passed"] is False
    assert not report["gates"][
        "stress_10bps_positive_after_removing_best_validation_year"
    ]
    assert not report["gates"][
        "stress_10bps_positive_negative_aapl_year_aggregate"
    ]


def test_final_gates_use_strict_period_edges_and_allow_equality_as_nonnegative():
    metrics = _final_metrics()
    report = experiment.apply_final_gates(metrics, _passing_integrity())
    assert report["passed"] is True
    assert report["primary_evidence"] == "causal_online_repeated_historical_audit"
    assert report["frozen_or_lifetime_diagnostic_can_rescue"] is False

    for cost_name, _ in experiment.COST_SCENARIOS:
        boundary = copy.deepcopy(metrics)
        boundary[cost_name]["selector"]["periods"]["2025"][
            "active_log_edge"
        ] = experiment.FINAL_MATERIAL_ACTIVE_LOG_EDGE
        failed = experiment.apply_final_gates(boundary, _passing_integrity())
        assert not failed["gates"][
            f"{cost_name}_2025_active_log_edge_above_001"
        ]

        one_nonnegative = copy.deepcopy(metrics)
        one_nonnegative[cost_name]["selector_vs_union"]["periods"] = {
            "2024": -0.01,
            "2025": 0.0,
            "2026_ytd": -0.01,
        }
        failed = experiment.apply_final_gates(
            one_nonnegative, _passing_integrity()
        )
        assert not failed["gates"][
            f"{cost_name}_minimum_two_nonnegative_incremental_periods"
        ]

        two_nonnegative = copy.deepcopy(metrics)
        two_nonnegative[cost_name]["selector_vs_union"]["periods"] = {
            "2024": 0.0,
            "2025": 0.0,
            "2026_ytd": -0.01,
        }
        passed = experiment.apply_final_gates(
            two_nonnegative, _passing_integrity()
        )
        assert passed["gates"][
            f"{cost_name}_minimum_two_nonnegative_incremental_periods"
        ]


def test_checkpoint_carries_both_states_pending_lessons_and_cooldown_context():
    frame = _market_frame()
    pending_position = len(frame) - 2
    forecast = _synthetic_forecast(frame, raw_positions=[pending_position])
    checkpoint = experiment._checkpoint_from_forecast(
        frame,
        forecast,
        cutoff=frame.index[-1],
        learning_mode=selector.CAUSAL_ONLINE_MODE,
    )

    assert checkpoint["account_start"] == "2005-01-01"
    assert checkpoint["account_reset_count_after_inception"] == 0
    assert checkpoint["regime_feature_order"] == list(
        selector.REGIME_FEATURE_COLUMNS
    )
    assert set(checkpoint["sufficient_states"]) == set(selector.REGIME_NAMES)
    assert len(checkpoint["account_trailing_cooldown_context"]) == 2
    assert len(checkpoint["pending_shadow_opportunities"]) == 1
    pending = checkpoint["pending_shadow_opportunities"][0]
    assert pending["signal_close"] == frame.index[pending_position].date().isoformat()
    assert pending["sessions_until_maturity"] == 1


def test_checkpoint_rejects_every_pending_physical_tail_field_tamper():
    frame = _market_frame()
    pending_position = len(frame) - 2
    forecast = _synthetic_forecast(frame, raw_positions=[pending_position])

    for field in (
        "sessions_until_maturity",
        "entry_adjusted_open",
        "learning_eligible",
    ):
        bad = forecast.copy()
        bad.attrs = copy.deepcopy(forecast.attrs)
        lesson = bad.attrs["pending_lessons"]["lessons"][0]
        if field == "sessions_until_maturity":
            lesson["sessions_until_maturity"] = 2
            lesson["entry_adjusted_open"] = None
        elif field == "entry_adjusted_open":
            lesson["entry_adjusted_open"] = float(
                lesson["entry_adjusted_open"]
            ) * 1.01
        else:
            lesson["learning_eligible"] = False

        with pytest.raises(
            experiment.BinaryRegimeUnionSelectorExperimentError,
            match="physical tail",
        ):
            experiment._checkpoint_from_forecast(
                frame,
                bad,
                cutoff=frame.index[-1],
                learning_mode=selector.CAUSAL_ONLINE_MODE,
            )


def test_checkpoint_rejects_serialized_state_or_physical_cutoff_tamper():
    frame = _market_frame()
    forecast = _synthetic_forecast(frame, raw_positions=[100])
    bad = forecast.copy()
    bad.attrs = copy.deepcopy(forecast.attrs)
    bad.attrs["final_states"] = copy.deepcopy(bad.attrs["final_states"])
    bad.attrs["final_states"]["states"]["risk_on"]["n_eff"] += 1.0
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="disagrees|malformed|state",
    ):
        experiment._checkpoint_from_forecast(
            frame,
            bad,
            cutoff=frame.index[-1],
            learning_mode=selector.CAUSAL_ONLINE_MODE,
        )

    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="physically bounded",
    ):
        experiment._checkpoint_from_forecast(
            frame,
            forecast,
            cutoff=frame.index[-2],
            learning_mode=selector.CAUSAL_ONLINE_MODE,
        )


def test_checkpoint_continuity_requires_byte_equivalent_regenerated_state(
    tmp_path: Path,
):
    frame = _market_frame()
    forecast = _synthetic_forecast(frame, raw_positions=[100])
    checkpoint = experiment._checkpoint_from_forecast(
        frame,
        forecast,
        cutoff=frame.index[-1],
        learning_mode=selector.CAUSAL_ONLINE_MODE,
    )
    parent_manifest = tmp_path / "stage_manifest.json"
    checkpoint_name = "checkpoint.json"
    checkpoint_path = tmp_path / checkpoint_name
    checkpoint_path.write_bytes(experiment._pretty_json_bytes(checkpoint))

    experiment._require_checkpoint_continuity(
        frame,
        forecast,
        cutoff=frame.index[-1],
        parent_manifest_path=parent_manifest,
        checkpoint_filename=checkpoint_name,
    )

    tampered = copy.deepcopy(checkpoint)
    tampered["account_trailing_cooldown_context"][-1][
        "union_candidate_signal"
    ] = not tampered["account_trailing_cooldown_context"][-1][
        "union_candidate_signal"
    ]
    checkpoint_path.write_bytes(experiment._pretty_json_bytes(tampered))
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="does not match",
    ):
        experiment._require_checkpoint_continuity(
            frame,
            forecast,
            cutoff=frame.index[-1],
            parent_manifest_path=parent_manifest,
            checkpoint_filename=checkpoint_name,
        )


def test_manifest_pass_claim_cannot_launder_report_or_gate_failure():
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

    for bad_manifest, bad_gate, bad_report in (
        ({**manifest, "stage_pass": False}, gate, report),
        (manifest, {"passed": False, "gates": {}}, report),
        (manifest, gate, {**report, "run_id": "other"}),
    ):
        with pytest.raises(
            experiment.BinaryRegimeUnionSelectorExperimentError,
            match="do not agree",
        ):
            experiment._require_pass_evidence_consistency(
                bad_manifest,
                bad_gate,
                bad_report,
                expected_stage="development",
            )


def test_deadline_failure_before_promotion_leaves_no_partial_bundle(tmp_path: Path):
    checked: list[str] = []

    class RejectPromotion:
        def check(self, location: str) -> None:
            checked.append(location)
            raise experiment.BinaryRegimeUnionSelectorExperimentError(
                "synthetic deadline"
            )

    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
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
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/codex/test",
            ("rev-parse", "@{upstream}"): "a" * 40,
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
    assert identity["upstream"] == "origin/codex/test"
    assert identity["upstream_commit"] == identity["commit"]
    assert identity["head_equals_upstream"] is True


def test_git_identity_rejects_unpushed_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def fake_git_text(_root: Path, *args: str) -> str:
        commands = {
            ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()),
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "codex/test",
            ("rev-parse", "HEAD"): "a" * 40,
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/codex/test",
            ("rev-parse", "@{upstream}"): "b" * 40,
        }
        return commands[args]

    monkeypatch.setattr(experiment, "_git_text", fake_git_text)

    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="HEAD to equal its pushed upstream commit",
    ):
        experiment._clean_git_identity(tmp_path)


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
        raise experiment.BinaryRegimeUnionSelectorExperimentError(
            "parent rejected"
        )

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
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="parent rejected",
    ):
        getattr(experiment, function_name)(**kwargs)
    assert loader_called is False


@pytest.mark.parametrize(
    (
        "function_name",
        "manifest_keyword",
        "parent_end",
        "later_end",
    ),
    [
        (
            "run_validation",
            "development_manifest",
            experiment.DEVELOPMENT_END,
            experiment.VALIDATION_END,
        ),
        (
            "run_final",
            "validation_manifest",
            experiment.VALIDATION_END,
            experiment.FINAL_END,
        ),
    ],
)
def test_parent_bounded_load_and_checkpoint_finish_before_later_loader_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    function_name: str,
    manifest_keyword: str,
    parent_end: pd.Timestamp,
    later_end: pd.Timestamp,
):
    """No loader call authorized to return later rows may precede checkpoint proof."""

    events: list[tuple[str, pd.Timestamp | None]] = []
    parent_only_frame = _market_frame(80, start="2017-01-03")

    class StopAfterLaterLoader(RuntimeError):
        pass

    def fake_loader(
        _path: Path,
        *,
        end: pd.Timestamp,
        required_last_session: pd.Timestamp,
    ):
        del required_last_session
        bounded_end = pd.Timestamp(end)
        events.append(("load", bounded_end))
        assert bounded_end in {parent_end, later_end}
        return parent_only_frame, {
            "source_type": "synthetic",
            "source_path": "synthetic.csv",
            "bounded_result_sha256": "sha256:" + "0" * 64,
        }

    def fake_checkpoint(*_args, cutoff: pd.Timestamp, **_kwargs):
        events.append(("checkpoint", pd.Timestamp(cutoff)))

    def stop_after_exact_bound(_provenance, *, stage: str):
        del stage
        events.append(("exact_bound", later_end))
        raise StopAfterLaterLoader("later loader reached")

    monkeypatch.setattr(experiment, "_clean_git_identity", lambda _root: {})
    monkeypatch.setattr(
        experiment,
        "_validated_prior_manifest",
        lambda **_kwargs: {"synthetic": True},
    )
    monkeypatch.setattr(
        experiment, "_require_dependency_continuity", lambda *_args: None
    )
    monkeypatch.setattr(
        experiment, "_tracked_input_identity", lambda *_args: {}
    )
    monkeypatch.setattr(experiment, "load_bounded_prices", fake_loader)
    monkeypatch.setattr(
        experiment,
        "build_binary_regime_union_selector_forecast",
        lambda frame, **_kwargs: _synthetic_forecast(frame, raw_positions=[]),
    )
    monkeypatch.setattr(
        experiment, "_require_checkpoint_continuity", fake_checkpoint
    )
    monkeypatch.setattr(
        experiment, "_require_source_continuity", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        experiment,
        "_require_exact_physical_stage_bound",
        stop_after_exact_bound,
    )
    kwargs = {
        "repo_root": tmp_path,
        "price_artifact": tmp_path / "synthetic.csv",
        manifest_keyword: tmp_path / "parent.json",
        "output_dir": tmp_path / "out",
        "run_id": "synthetic-loader-order",
    }

    with pytest.raises(StopAfterLaterLoader, match="later loader reached"):
        getattr(experiment, function_name)(**kwargs)

    assert events == [
        ("load", parent_end),
        ("checkpoint", parent_end),
        ("load", later_end),
        ("exact_bound", later_end),
    ]


def test_full_stage_bound_rejects_any_physical_or_returned_later_rows():
    exact = {
        "physical_snapshot_has_later_rows": False,
        "rows_after_bound_returned": False,
    }
    experiment._require_exact_physical_stage_bound(exact, stage="Synthetic")

    for field in (
        "physical_snapshot_has_later_rows",
        "rows_after_bound_returned",
    ):
        invalid = dict(exact)
        invalid[field] = True
        with pytest.raises(
            experiment.BinaryRegimeUnionSelectorExperimentError,
            match="physically exact|after its authorized bound",
        ):
            experiment._require_exact_physical_stage_bound(
                invalid, stage="Synthetic"
            )


@pytest.mark.parametrize(
    ("function_name", "manifest_keyword", "parent_end", "later_end"),
    [
        (
            "run_validation",
            "development_manifest",
            experiment.DEVELOPMENT_END,
            experiment.VALIDATION_END,
        ),
        (
            "run_final",
            "validation_manifest",
            experiment.VALIDATION_END,
            experiment.FINAL_END,
        ),
    ],
)
def test_checkpoint_failure_prevents_any_full_bound_loader_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    function_name: str,
    manifest_keyword: str,
    parent_end: pd.Timestamp,
    later_end: pd.Timestamp,
):
    load_ends: list[pd.Timestamp] = []
    parent_only_frame = _market_frame(80, start="2017-01-03")

    def fake_loader(
        _path: Path,
        *,
        end: pd.Timestamp,
        required_last_session: pd.Timestamp,
    ):
        del required_last_session
        bounded_end = pd.Timestamp(end)
        load_ends.append(bounded_end)
        if bounded_end == later_end:
            raise AssertionError("full-bound loader must not run")
        return parent_only_frame, {
            "source_type": "synthetic",
            "source_path": "synthetic.csv",
            "bounded_result_sha256": "sha256:" + "0" * 64,
        }

    monkeypatch.setattr(experiment, "_clean_git_identity", lambda _root: {})
    monkeypatch.setattr(
        experiment,
        "_validated_prior_manifest",
        lambda **_kwargs: {"synthetic": True},
    )
    monkeypatch.setattr(
        experiment, "_require_dependency_continuity", lambda *_args: None
    )
    monkeypatch.setattr(
        experiment, "_tracked_input_identity", lambda *_args: {}
    )
    monkeypatch.setattr(experiment, "load_bounded_prices", fake_loader)
    monkeypatch.setattr(
        experiment,
        "build_binary_regime_union_selector_forecast",
        lambda frame, **_kwargs: _synthetic_forecast(frame, raw_positions=[]),
    )
    monkeypatch.setattr(
        experiment,
        "_require_checkpoint_continuity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            experiment.BinaryRegimeUnionSelectorExperimentError(
                "synthetic checkpoint mismatch"
            )
        ),
    )
    monkeypatch.setattr(
        experiment, "_require_source_continuity", lambda *_args, **_kwargs: None
    )
    kwargs = {
        "repo_root": tmp_path,
        "price_artifact": tmp_path / "synthetic.csv",
        manifest_keyword: tmp_path / "parent.json",
        "output_dir": tmp_path / "out",
        "run_id": "synthetic-checkpoint-rejection",
    }
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="checkpoint mismatch",
    ):
        getattr(experiment, function_name)(**kwargs)
    assert load_ends == [parent_end]


def test_parent_payload_inventory_rejects_missing_or_unsafe_entries():
    required = {
        name: "sha256:" + "0" * 64
        for name in experiment._required_parent_payloads("development")
    }
    assert experiment._validated_payload_inventory(
        required, expected_stage="development"
    ) == required

    missing = dict(required)
    missing.pop(next(iter(missing)))
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="incomplete|unsafe",
    ):
        experiment._validated_payload_inventory(
            missing, expected_stage="development"
        )

    unsafe = {**required, "../escape.json": "sha256:" + "0" * 64}
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="incomplete|unsafe",
    ):
        experiment._validated_payload_inventory(
            unsafe, expected_stage="development"
        )


def test_dependency_continuity_binds_runtime_and_all_implementation_hashes():
    identity = {
        "tracked_dependency_sha256": {"core.py": "sha256:" + "1" * 64},
        "runtime_versions": {"python": "synthetic"},
    }
    parent = {"git_identity": copy.deepcopy(identity)}
    experiment._require_dependency_continuity(identity, parent)

    changed = copy.deepcopy(identity)
    changed["runtime_versions"]["python"] = "changed"
    with pytest.raises(
        experiment.BinaryRegimeUnionSelectorExperimentError,
        match="implementation|runtime",
    ):
        experiment._require_dependency_continuity(changed, parent)


def test_successful_bundle_records_zero_external_cost_and_no_powerful_execution(
    tmp_path: Path,
):
    class AcceptDeadline:
        def check(self, _location: str) -> None:
            return None

    result = experiment._stage_bundle(
        output_dir=tmp_path,
        run_id="sealed-synthetic",
        stage="development",
        stage_pass=False,
        report={"synthetic": True},
        payloads={"payload.txt": b"synthetic\n"},
        source_provenance={
            "source_path": "synthetic.csv",
            "bounded_result_sha256": "sha256:" + "0" * 64,
        },
        git_identity={},
        parent_manifest=None,
        deadline=AcceptDeadline(),
    )
    run_dir = Path(result["artifact_dir"])
    manifest = experiment._json_object(
        run_dir / "stage_manifest.json", description="synthetic manifest"
    )
    execution = manifest["execution"]
    assert execution["actions"] == ["LONG_100_PERCENT", "CASH_100_PERCENT"]
    assert execution["continuous_account_start"] == "2005-01-01"
    assert execution["account_resets_after_inception"] == 0
    assert execution["maximum_target_exposure"] == pytest.approx(1.0)
    assert execution["shorting"] is False
    assert execution["leverage"] is False
    assert execution["borrowing"] is False
    assert execution["network_access"] is False
    assert execution["llm_calls"] == 0
    assert execution["api_calls"] == 0
    assert execution["estimated_external_cost_usd"] == pytest.approx(0.0)
    assert (run_dir / "checksums.json").exists()
