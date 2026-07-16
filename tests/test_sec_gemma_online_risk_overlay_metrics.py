from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    DEVELOPMENT_BLOCKS,
    POSITIVE_EDGE_TOLERANCE,
    build_contract_manifest,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_ledger import (
    build_baseline_signal_row,
    run_binary_ledger,
)
import agent_benchmark.sec_gemma_online_risk_overlay_metrics as metrics
from agent_benchmark.sec_gemma_online_risk_overlay_replay import (
    replay_sec_gemma_online_risk_overlay_chronology,
)
from agent_benchmark.sec_gemma_online_risk_overlay_policy import (
    replay_nonoverlapping_overlay_policy,
)
from tests import test_sec_gemma_online_risk_overlay_learner as learner_helpers


def _sessions(*, include_2025: bool = True) -> list[str]:
    result = [
        "2000-01-03",
        "2004-12-31",
        "2005-01-03",
        "2018-12-31",
        "2023-12-29",
        "2024-01-02",
        "2024-12-31",
    ]
    if include_2025:
        result.extend(["2025-01-02", "2025-12-31"])
    result.extend(["2026-01-02", "2026-07-09"])
    return result


def _market(sessions: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "session": session,
            "adjusted_open_hex": (
                100.0 * math.exp(0.08 * position)
            ).hex(),
            "adjusted_close_hex": (
                101.0 * math.exp(0.08 * position)
            ).hex(),
        }
        for position, session in enumerate(sessions)
    ]


def _replay_pair(
    *,
    include_2025: bool = True,
    active_baseline_positions: set[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sessions = _sessions(include_2025=include_2025)
    market = _market(sessions)
    active = active_baseline_positions or set()
    signals = [
        build_baseline_signal_row(
            session=session,
            unfiltered_union_signal=position in active,
        )
        for position, session in enumerate(sessions)
    ]

    def run(
        arm: str, frozen_before_boundary: str | None
    ) -> dict[str, Any]:
        return replay_sec_gemma_online_risk_overlay_chronology(
            market_rows=market,
            expected_market_rows_sha256=canonical_sha256(market),
            baseline_signals=signals,
            expected_baseline_signals_sha256=canonical_sha256(signals),
            feature_rows=[],
            expected_feature_row_sha256s=[],
            arm=arm,
            replay_id=f"metrics-{arm}",
            frozen_before_boundary=frozen_before_boundary,
        )

    return (
        run("semantic", "2024-01-02"),
        run("no_filing_meaning", None),
        run("no_gemma_channel", None),
    )


def _family(
    branch: dict[str, Any], account: str
) -> dict[str, Any]:
    return {
        cost: branch["ledgers"][cost][account]
        for cost in ("cost_5bps", "cost_10bps")
    }


def _final_input(
    *,
    include_2025: bool = True,
    active_baseline_positions: set[int] | None = None,
) -> dict[str, Any]:
    semantic, ablation, no_gemma = _replay_pair(
        include_2025=include_2025,
        active_baseline_positions=active_baseline_positions,
    )
    frozen = semantic["frozen_control"]
    assert frozen is not None
    return metrics.build_stage_metrics_input(
        stage="final",
        ledgers={
            "semantic": _family(semantic["primary"], "combined"),
            "baseline": _family(semantic["primary"], "baseline"),
            "aapl_buy_and_hold": _family(
                semantic["primary"], "aapl_buy_and_hold"
            ),
            "no_filing_meaning": _family(
                ablation["primary"], "combined"
            ),
            "no_gemma_channel": _family(
                no_gemma["primary"], "combined"
            ),
        },
        frozen_control_ledgers={
            "through_2023": _family(frozen, "combined")
        },
        feature_rows=[],
        semantic_predictions=[],
        no_filing_meaning_predictions=[],
        no_gemma_channel_predictions=[],
        frozen_predictions={"through_2023": []},
        learner_lessons=[],
        semantic_policy_actions=[],
        no_filing_meaning_policy_actions=[],
        no_gemma_channel_policy_actions=[],
        frozen_policy_actions={"through_2023": []},
        semantic_overlay_episodes=[],
    )


@pytest.fixture(scope="module")
def empty_final_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    input_artifact = _final_input()
    stage_metrics = metrics.build_stage_metrics(
        input_artifact,
        expected_stage_metrics_input_sha256=input_artifact[
            "stage_metrics_input_sha256"
        ],
    )
    return input_artifact, stage_metrics


def _target_rows(
    sessions: list[str], targets: list[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for session, target in zip(sessions, targets, strict=True):
        body = {"session": session, "target_exposure": target}
        result.append({**body, "target_row_sha256": canonical_sha256(body)})
    return result


def _empty_stage_input(stage: str) -> dict[str, Any]:
    if stage == "development":
        session_set = {"2000-01-03", "2004-12-31"}
        for _, first, last in DEVELOPMENT_BLOCKS:
            session_set.update((first, last))
        for year in range(2005, 2019):
            session_set.add(f"{year}-06-30")
        sessions = sorted(session_set)
        boundaries = {
            block_id: first
            for block_id, first, _ in DEVELOPMENT_BLOCKS
        }
    elif stage == "confirmation":
        sessions = [
            "2000-01-03",
            "2004-12-31",
            "2005-01-03",
            "2018-12-31",
            "2019-01-02",
            "2019-12-31",
            "2020-12-31",
            "2021-12-31",
            "2022-12-30",
            "2023-12-29",
        ]
        boundaries = {"through_2018": "2019-01-02"}
    else:
        raise AssertionError(stage)
    market = _market(sessions)
    signals = [
        build_baseline_signal_row(
            session=session,
            unfiltered_union_signal=False,
        )
        for session in sessions
    ]

    def run(
        arm: str,
        *,
        replay_id: str,
        boundary: str | None = None,
    ) -> dict[str, Any]:
        return replay_sec_gemma_online_risk_overlay_chronology(
            market_rows=market,
            expected_market_rows_sha256=canonical_sha256(market),
            baseline_signals=signals,
            expected_baseline_signals_sha256=canonical_sha256(signals),
            feature_rows=[],
            expected_feature_row_sha256s=[],
            arm=arm,
            replay_id=replay_id,
            frozen_before_boundary=boundary,
        )

    semantic = run("semantic", replay_id=f"{stage}-semantic")
    ablation = run(
        "no_filing_meaning", replay_id=f"{stage}-ablation"
    )
    no_gemma = run(
        "no_gemma_channel", replay_id=f"{stage}-no-gemma"
    )
    controls = {
        control_id: run(
            "semantic",
            replay_id=f"{stage}-{control_id}",
            boundary=boundary,
        )["frozen_control"]
        for control_id, boundary in boundaries.items()
    }
    assert all(control is not None for control in controls.values())
    return metrics.build_stage_metrics_input(
        stage=stage,
        ledgers={
            "semantic": _family(semantic["primary"], "combined"),
            "baseline": _family(semantic["primary"], "baseline"),
            "aapl_buy_and_hold": _family(
                semantic["primary"], "aapl_buy_and_hold"
            ),
            "no_filing_meaning": _family(
                ablation["primary"], "combined"
            ),
            "no_gemma_channel": _family(
                no_gemma["primary"], "combined"
            ),
        },
        frozen_control_ledgers={
            control_id: _family(control, "combined")
            for control_id, control in controls.items()
            if control is not None
        },
        feature_rows=[],
        semantic_predictions=[],
        no_filing_meaning_predictions=[],
        no_gemma_channel_predictions=[],
        frozen_predictions={
            control_id: []
            for control_id in controls
        },
        learner_lessons=[],
        semantic_policy_actions=[],
        no_filing_meaning_policy_actions=[],
        no_gemma_channel_policy_actions=[],
        frozen_policy_actions={
            control_id: []
            for control_id in controls
        },
        semantic_overlay_episodes=[],
    )


def test_destination_open_costs_stay_in_the_destination_year_and_close_is_diagnostic() -> None:
    # Position 6 is the 2024-12-31 close, so its baseline signal sells at
    # the 2025-01-02 destination open. Position 9 sells at 2026-07-09.
    input_artifact = _final_input(
        active_baseline_positions={6, 9}
    )
    stage_metrics = metrics.build_stage_metrics(
        input_artifact,
        expected_stage_metrics_input_sha256=input_artifact[
            "stage_metrics_input_sha256"
        ],
    )
    windows = stage_metrics["window_metrics"]
    for valuation in metrics.VALUATION_METHODS:
        edge_2024 = float.fromhex(
            windows[valuation]["cost_10bps"]["semantic_vs_aapl"][
                "2024"
            ]["log_edge_hex"]
        )
        assert abs(edge_2024) <= 1e-15

    open_continuous = float.fromhex(
        windows["adjusted_open"]["cost_10bps"]["semantic_vs_aapl"][
            "final_continuous"
        ]["log_edge_hex"]
    )
    close_continuous = float.fromhex(
        windows["terminal_adjusted_close"]["cost_10bps"][
            "semantic_vs_aapl"
        ]["final_continuous"]["log_edge_hex"]
    )
    final_market_row = input_artifact["ledgers"]["semantic"][
        "cost_10bps"
    ]["ledger_rows"][-1]
    expected_close_delta = -math.log(
        float.fromhex(final_market_row["adjusted_close_hex"])
        / float.fromhex(final_market_row["adjusted_open_hex"])
    )
    assert close_continuous - open_continuous == pytest.approx(
        expected_close_delta, abs=1e-15
    )


def test_xor_interval_includes_entry_and_return_to_equality_cost_rows() -> None:
    sessions = [
        "2000-01-03",
        "2000-01-04",
        "2000-01-05",
        "2000-01-06",
        "2000-01-07",
    ]
    market = [
        {
            "session": session,
            "adjusted_open_hex": (100.0 + 10.0 * position).hex(),
            "adjusted_close_hex": (101.0 + 10.0 * position).hex(),
        }
        for position, session in enumerate(sessions)
    ]
    left = run_binary_ledger(
        market_rows=market,
        target_rows=_target_rows(sessions, [1, 0, 0, 1, 1]),
        policy_id="left",
        cost_bps=10,
    )
    right = run_binary_ledger(
        market_rows=market,
        target_rows=_target_rows(sessions, [1, 1, 1, 1, 1]),
        policy_id="right",
        cost_bps=10,
    )
    attribution = metrics._xor_attribution(
        left,
        right,
        comparison_id="test",
        first=sessions[0],
        last=sessions[-1],
    )
    assert attribution["complete_count"] == 1
    interval = attribution["intervals"][0]
    assert interval["start_session"] == sessions[1]
    assert interval["return_to_equality_session"] == sessions[3]
    assert interval["assigned_row_count"] == 3
    expected = math.fsum(
        math.log(float.fromhex(left["ledger_rows"][position][
            "period_factor_hex"
        ]))
        - math.log(float.fromhex(right["ledger_rows"][position][
            "period_factor_hex"
        ]))
        for position in (1, 2, 3)
    )
    assert float.fromhex(interval["contribution_hex"]) == expected

    carried = metrics._xor_attribution(
        left,
        right,
        comparison_id="carried",
        first=sessions[2],
        last=sessions[-1],
    )
    assert carried["complete_count"] == 0
    assert carried["intervals"][0]["started_inside_window"] is False


def test_rehashed_pending_episode_cannot_be_forged_into_a_complete_episode() -> None:
    sessions = ["2026-07-08", "2026-07-09"]
    prediction_body = {
        "accession_number": "0000320193-26-000001",
        "decision_session": "2026-07-09",
        "acceptance_datetime": "20260709163000",
        "prediction_available": True,
        "learner_ready": True,
        "raw_gate_pass": True,
    }
    prediction = {
        **prediction_body,
        "prediction_row_sha256": canonical_sha256(prediction_body),
    }
    policy = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=[prediction],
        policy_id="pending-test",
    )
    expected = metrics._expected_episode_observations(
        scheduled_overlays=policy["scheduled_overlays"],
        market_sessions=sessions,
    )
    assert expected[0]["realization_status"] == "pending_entry"

    forged = copy.deepcopy(expected[0])
    forged["entry_session"] = sessions[0]
    forged["exit_session"] = sessions[1]
    forged["entry_position"] = 0
    forged["exit_position"] = 1
    forged["realization_status"] = "complete"
    forged["overlay_episode_observation_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in forged.items()
            if key != "overlay_episode_observation_sha256"
        }
    )
    validated_forgery = metrics._validate_episodes(
        [forged], location="forged episodes"
    )
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="deterministic schedule realization",
    ):
        metrics._require_episode_observations_match_policy(
            episodes=validated_forgery,
            scheduled_overlays=policy["scheduled_overlays"],
            market_sessions=sessions,
        )


@pytest.mark.parametrize("stage", ["development", "confirmation"])
def test_all_stage_gate_reports_match_the_literal_contract_key_set(
    stage: str,
) -> None:
    input_artifact = _empty_stage_input(stage)
    stage_metrics = metrics.build_stage_metrics(
        input_artifact,
        expected_stage_metrics_input_sha256=input_artifact[
            "stage_metrics_input_sha256"
        ],
    )
    report = metrics.build_stage_gate_report(
        stage_metrics,
        expected_stage_metrics_sha256=stage_metrics[
            "stage_metrics_sha256"
        ],
    )
    assert set(report["checks"]) == set(
        build_contract_manifest()["gates"][stage]
    )
    assert stage_metrics["calendar_diagnostics"][
        "by_valuation_and_cost"
    ]
    assert "semantic_vs_no_gemma_channel" in stage_metrics[
        "window_metrics"
    ]["adjusted_open"]["cost_10bps"]
    assert stage_metrics[
        "semantic_vs_no_gemma_channel_action_differences"
    ]["action_difference_count"] == 0
    assert stage_metrics[
        "semantic_vs_no_gemma_channel_xor_attribution_10bps"
    ]["complete_count"] == 0
    assert (
        stage_metrics["development_block_diagnostics"] is not None
    ) is (stage == "development")


@pytest.mark.parametrize("stage", ["development", "confirmation"])
def test_pre_final_qualification_uses_adjusted_open_while_still_reporting_close(
    stage: str,
) -> None:
    input_artifact = _empty_stage_input(stage)
    stage_metrics = metrics.build_stage_metrics(
        input_artifact,
        expected_stage_metrics_input_sha256=input_artifact[
            "stage_metrics_input_sha256"
        ],
    )
    if stage == "development":
        stage_metrics["window_metrics"]["adjusted_open"]["cost_10bps"][
            "semantic_vs_aapl"
        ]["development_total"]["log_edge_hex"] = 0.02.hex()
        stage_metrics["window_metrics"]["terminal_adjusted_close"][
            "cost_10bps"
        ]["semantic_vs_aapl"]["development_total"][
            "log_edge_hex"
        ] = (-1.0).hex()
        checks = metrics._development_checks(stage_metrics)
        assert checks[
            "combined_total_active_log_edge_10bps_at_least"
        ] is True
    else:
        for cost in ("cost_5bps", "cost_10bps"):
            stage_metrics["window_metrics"]["adjusted_open"][cost][
                "semantic_vs_aapl"
            ]["confirmation_total"]["log_edge_hex"] = 0.001.hex()
            stage_metrics["window_metrics"][
                "terminal_adjusted_close"
            ][cost]["semantic_vs_aapl"]["confirmation_total"][
                "log_edge_hex"
            ] = (-1.0).hex()
        checks = metrics._confirmation_checks(stage_metrics)
        assert checks[
            "combined_active_log_edge_positive_at_5_and_10bps"
        ] is True


def test_missing_final_period_fails_instead_of_becoming_an_empty_metric() -> None:
    input_artifact = _final_input(include_2025=False)
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="no physical ledger support",
    ):
        metrics.build_stage_metrics(
            input_artifact,
            expected_stage_metrics_input_sha256=input_artifact[
                "stage_metrics_input_sha256"
            ],
        )


def test_input_and_output_validators_reject_extra_missing_and_tampered_evidence(
    empty_final_artifacts: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    input_artifact, stage_metrics = empty_final_artifacts
    assert (
        metrics.validate_stage_metrics_input(
            input_artifact,
            expected_stage_metrics_input_sha256=input_artifact[
                "stage_metrics_input_sha256"
            ],
        )
        == input_artifact["stage_metrics_input_sha256"]
    )
    assert (
        metrics.validate_stage_metrics(
            stage_metrics,
            expected_stage_metrics_sha256=stage_metrics[
                "stage_metrics_sha256"
            ],
            metrics_input=input_artifact,
            expected_stage_metrics_input_sha256=input_artifact[
                "stage_metrics_input_sha256"
            ],
        )
        == stage_metrics["stage_metrics_sha256"]
    )

    extra = copy.deepcopy(input_artifact)
    extra["unexpected"] = True
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="keys changed",
    ):
        metrics.validate_stage_metrics_input(
            extra,
            expected_stage_metrics_input_sha256=input_artifact[
                "stage_metrics_input_sha256"
            ],
        )

    missing = copy.deepcopy(input_artifact)
    del missing["feature_rows"]
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="keys changed",
    ):
        metrics.validate_stage_metrics_input(
            missing,
            expected_stage_metrics_input_sha256=input_artifact[
                "stage_metrics_input_sha256"
            ],
        )

    omitted_diagnostic = copy.deepcopy(input_artifact)
    del omitted_diagnostic["ledgers"]["no_gemma_channel"]
    omitted_diagnostic["stage_metrics_input_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in omitted_diagnostic.items()
            if key != "stage_metrics_input_sha256"
        }
    )
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="five declared accounts",
    ):
        metrics.validate_stage_metrics_input(
            omitted_diagnostic,
            expected_stage_metrics_input_sha256=omitted_diagnostic[
                "stage_metrics_input_sha256"
            ],
        )

    changed_ledger = copy.deepcopy(input_artifact)
    changed_ledger["ledgers"]["semantic"]["cost_10bps"]["ledger_rows"][
        -1
    ]["period_factor_hex"] = 2.0.hex()
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="self-hash changed|batch hash changed|arithmetic changed",
    ):
        metrics.validate_stage_metrics_input(
            changed_ledger,
            expected_stage_metrics_input_sha256=input_artifact[
                "stage_metrics_input_sha256"
            ],
        )

    changed_metrics = copy.deepcopy(stage_metrics)
    changed_metrics["window_metrics"]["adjusted_open"]["cost_5bps"][
        "semantic_vs_aapl"
    ]["2025"]["log_edge_hex"] = 1.0.hex()
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="hash changed|self-hash changed|reconstruction",
    ):
        metrics.validate_stage_metrics(
            changed_metrics,
            expected_stage_metrics_sha256=stage_metrics[
                "stage_metrics_sha256"
            ],
            metrics_input=input_artifact,
            expected_stage_metrics_input_sha256=input_artifact[
                "stage_metrics_input_sha256"
            ],
        )


def test_nonempty_replay_predictions_and_actions_are_bound_to_features() -> None:
    sessions = _sessions()
    market = _market(sessions)
    signals = [
        build_baseline_signal_row(
            session=session,
            unfiltered_union_signal=False,
        )
        for session in sessions
    ]
    feature = learner_helpers._feature(
        901, decision_session="2024-12-31"
    )

    def run(
        arm: str, frozen_before_boundary: str | None
    ) -> dict[str, Any]:
        return replay_sec_gemma_online_risk_overlay_chronology(
            market_rows=market,
            expected_market_rows_sha256=canonical_sha256(market),
            baseline_signals=signals,
            expected_baseline_signals_sha256=canonical_sha256(signals),
            feature_rows=[feature],
            expected_feature_row_sha256s=[
                feature["feature_row_sha256"]
            ],
            arm=arm,
            replay_id=f"bound-{arm}",
            frozen_before_boundary=frozen_before_boundary,
        )

    semantic = run("semantic", "2024-01-02")
    ablation = run("no_filing_meaning", None)
    no_gemma = run("no_gemma_channel", None)
    frozen = semantic["frozen_control"]
    assert frozen is not None
    input_artifact = metrics.build_stage_metrics_input(
        stage="final",
        ledgers={
            "semantic": _family(semantic["primary"], "combined"),
            "baseline": _family(semantic["primary"], "baseline"),
            "aapl_buy_and_hold": _family(
                semantic["primary"], "aapl_buy_and_hold"
            ),
            "no_filing_meaning": _family(
                ablation["primary"], "combined"
            ),
            "no_gemma_channel": _family(
                no_gemma["primary"], "combined"
            ),
        },
        frozen_control_ledgers={
            "through_2023": _family(frozen, "combined")
        },
        feature_rows=[feature],
        semantic_predictions=semantic["primary"]["predictions"],
        no_filing_meaning_predictions=ablation["primary"][
            "predictions"
        ],
        no_gemma_channel_predictions=no_gemma["primary"][
            "predictions"
        ],
        frozen_predictions={
            "through_2023": frozen["predictions"]
        },
        learner_lessons=semantic["learner_lessons"],
        semantic_policy_actions=semantic["primary"]["policy_replay"][
            "policy_actions"
        ],
        no_filing_meaning_policy_actions=ablation["primary"][
            "policy_replay"
        ]["policy_actions"],
        no_gemma_channel_policy_actions=no_gemma["primary"][
            "policy_replay"
        ]["policy_actions"],
        frozen_policy_actions={
            "through_2023": frozen["policy_replay"]["policy_actions"]
        },
        semantic_overlay_episodes=semantic["primary"]["target_stream"][
            "overlay_episodes"
        ],
    )
    stage_metrics = metrics.build_stage_metrics(
        input_artifact,
        expected_stage_metrics_input_sha256=input_artifact[
            "stage_metrics_input_sha256"
        ],
    )
    assert stage_metrics["coverage_metrics"]["eligible_call_count"] == 1

    changed = copy.deepcopy(input_artifact)
    changed["semantic_policy_actions"][0][
        "prediction_row_sha256"
    ] = "f" * 64
    changed["semantic_policy_actions"][0][
        "policy_action_sha256"
    ] = canonical_sha256(
        {
            key: value
            for key, value in changed["semantic_policy_actions"][0].items()
            if key != "policy_action_sha256"
        }
    )
    changed["semantic_policy_actions_sha256"] = canonical_sha256(
        changed["semantic_policy_actions"]
    )
    changed["stage_metrics_input_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in changed.items()
            if key != "stage_metrics_input_sha256"
        }
    )
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="differs from its prediction",
    ):
        metrics.validate_stage_metrics_input(
            changed,
            expected_stage_metrics_input_sha256=changed[
                "stage_metrics_input_sha256"
            ],
        )


def _passing_final_gate_metrics(
    source: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(source)
    for valuation in metrics.VALUATION_METHODS:
        active_5 = value["window_metrics"][valuation]["cost_5bps"][
            "semantic_vs_aapl"
        ]
        for period in ("2024", "2025", "2026_ytd"):
            active_5[period]["log_edge_hex"] = 0.005.hex()
        active_5["final_continuous"]["log_edge_hex"] = 0.02.hex()

        active_10 = value["window_metrics"][valuation]["cost_10bps"][
            "semantic_vs_aapl"
        ]
        for period in ("2024", "2025", "2026_ytd"):
            active_10[period]["log_edge_hex"] = (
                POSITIVE_EDGE_TOLERANCE * 2.0
            ).hex()

        baseline_10 = value["window_metrics"][valuation]["cost_10bps"][
            "semantic_vs_baseline"
        ]
        baseline_10["final_continuous"]["log_edge_hex"] = 0.001.hex()
        baseline_10["2024"]["log_edge_hex"] = 0.001.hex()
        baseline_10["2025"]["log_edge_hex"] = 0.001.hex()
        baseline_10["2026_ytd"]["log_edge_hex"] = (-0.001).hex()

        no_meaning_10 = value["window_metrics"][valuation][
            "cost_10bps"
        ]["semantic_vs_no_filing_meaning"]
        no_meaning_10["final_continuous"]["log_edge_hex"] = 0.001.hex()
        value["frozen_window_metrics"][valuation]["cost_10bps"][
            "final_continuous"
        ]["log_edge_hex"] = 0.001.hex()

    value["coverage_metrics"][
        "schema_valid_extraction_rate_hex"
    ] = 0.90.hex()
    value["coverage_metrics"][
        "nonzero_filing_meaning_row_count"
    ] = 3
    value[
        "semantic_vs_no_filing_meaning_action_differences"
    ]["action_difference_count"] = 2
    value["online_vs_frozen_action_differences"][
        "action_difference_count"
    ] = 3
    episodes = value["semantic_overlay_episode_attribution_10bps"]
    episodes["complete_count"] = 6
    episodes["win_rate_hex"] = 0.55.hex()
    episodes["largest_positive_share_hex"] = 0.50.hex()
    return value


def test_gate_thresholds_use_unrounded_values_and_exact_inclusive_or_strict_rules(
    empty_final_artifacts: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, stage_metrics = empty_final_artifacts
    passing = _passing_final_gate_metrics(stage_metrics)
    checks = metrics._final_checks(passing)
    assert all(checks.values())

    below_unrounded = copy.deepcopy(passing)
    below_unrounded["window_metrics"]["adjusted_open"]["cost_5bps"][
        "semantic_vs_aapl"
    ]["2025"]["log_edge_hex"] = (0.005 - 1e-15).hex()
    assert (
        metrics._final_checks(below_unrounded)[
            "active_edge_5bps_each_2024_2025_2026_ytd_at_least"
        ]
        is False
    )

    strict_tie = copy.deepcopy(passing)
    strict_tie["window_metrics"]["adjusted_open"]["cost_10bps"][
        "semantic_vs_aapl"
    ]["2025"]["log_edge_hex"] = POSITIVE_EDGE_TOLERANCE.hex()
    assert (
        metrics._final_checks(strict_tie)[
            "active_edge_positive_each_period_at_10bps"
        ]
        is False
    )

    concentrated = copy.deepcopy(passing)
    concentrated["semantic_overlay_episode_attribution_10bps"][
        "largest_positive_share_hex"
    ] = (0.50 + 1e-15).hex()
    assert (
        metrics._final_checks(concentrated)[
            "largest_positive_episode_share_10bps_at_most"
        ]
        is False
    )


def test_calendar_diagnostics_explicitly_mark_negative_aapl_periods(
    empty_final_artifacts: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, stage_metrics = empty_final_artifacts
    windows = copy.deepcopy(stage_metrics["window_metrics"])
    for valuation in metrics.VALUATION_METHODS:
        row = windows[valuation]["cost_10bps"]["semantic_vs_aapl"][
            "2025"
        ]
        row["right_log_return_hex"] = (-0.01).hex()
        row["log_edge_hex"] = 0.001.hex()
    diagnostics, blocks = metrics._calendar_and_block_diagnostics(
        stage="final",
        window_metrics=windows,
    )
    assert blocks is None
    for valuation in metrics.VALUATION_METHODS:
        slice_ = diagnostics["by_valuation_and_cost"][valuation][
            "cost_10bps"
        ]
        row = next(
            item for item in slice_["rows"] if item["window_id"] == "2025"
        )
        assert row["negative_aapl"] is True
        assert row["active_edge_strictly_positive"] is True
        assert slice_["negative_aapl_window_count"] == 1
        assert slice_[
            "positive_active_negative_aapl_window_count"
        ] == 1


def test_brier_uses_identical_mature_available_support_and_binary_edge_target() -> None:
    semantic = [
        {
            "accession_number": "a",
            "decision_session": "2024-02-01",
            "fitted_prediction_available": True,
            "prediction_row_sha256": "1" * 64,
            "gate_audit": {"probability_hex": 0.8.hex()},
        },
        {
            "accession_number": "pending",
            "decision_session": "2026-07-01",
            "fitted_prediction_available": True,
            "prediction_row_sha256": "2" * 64,
            "gate_audit": {"probability_hex": 0.2.hex()},
        },
    ]
    ablation = [
        {
            **semantic[0],
            "prediction_row_sha256": "3" * 64,
            "gate_audit": {"probability_hex": 0.5.hex()},
        },
        {
            **semantic[1],
            "prediction_row_sha256": "4" * 64,
            "gate_audit": {"probability_hex": 0.7.hex()},
        },
    ]
    lessons = [
        {
            "accession_number": "a",
            "maturity_session": "2024-03-01",
            "trainable": True,
            "binary_cash_win_target": 1,
            "lesson_row_sha256": "5" * 64,
        },
        {
            "accession_number": "pending",
            "maturity_session": "2026-07-10",
            "trainable": True,
            "binary_cash_win_target": 0,
            "lesson_row_sha256": "6" * 64,
        },
    ]
    report = metrics._brier_metrics(
        stage="final",
        semantic_predictions=semantic,
        no_meaning_predictions=ablation,
        no_gemma_predictions=ablation,
        lessons=lessons,
    )
    assert report["support_count"] == 1
    assert float.fromhex(report["semantic_brier_score_hex"]) == pytest.approx(
        0.04
    )
    assert float.fromhex(report["ablation_brier_score_hex"]) == pytest.approx(
        0.25
    )
    assert float.fromhex(report["no_gemma_brier_score_hex"]) == pytest.approx(
        0.25
    )
    assert float.fromhex(report["relative_improvement_hex"]) == pytest.approx(
        0.84
    )


def test_gate_report_validator_rejects_added_or_removed_checks(
    empty_final_artifacts: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _, stage_metrics = empty_final_artifacts
    report = metrics.build_stage_gate_report(
        stage_metrics,
        expected_stage_metrics_sha256=stage_metrics[
            "stage_metrics_sha256"
        ],
    )
    assert set(report["checks"]) == set(
        build_contract_manifest()["gates"]["final"]
    )
    assert (
        metrics.validate_stage_gate_report(
            report,
            expected_gate_report_sha256=report["gate_report_sha256"],
            metrics=stage_metrics,
            expected_stage_metrics_sha256=stage_metrics[
                "stage_metrics_sha256"
            ],
        )
        == report["gate_report_sha256"]
    )

    added = copy.deepcopy(report)
    added["checks"]["invented_check"] = True
    added["checks_sha256"] = canonical_sha256(added["checks"])
    added["gate_report_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in added.items()
            if key != "gate_report_sha256"
        }
    )
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="deterministic reconstruction",
    ):
        metrics.validate_stage_gate_report(
            added,
            expected_gate_report_sha256=added["gate_report_sha256"],
            metrics=stage_metrics,
            expected_stage_metrics_sha256=stage_metrics[
                "stage_metrics_sha256"
            ],
        )

    removed = copy.deepcopy(report)
    removed_name = next(iter(removed["checks"]))
    del removed["checks"][removed_name]
    removed["checks_sha256"] = canonical_sha256(removed["checks"])
    removed["gate_report_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in removed.items()
            if key != "gate_report_sha256"
        }
    )
    with pytest.raises(
        metrics.SecGemmaOnlineRiskOverlayMetricsError,
        match="deterministic reconstruction",
    ):
        metrics.validate_stage_gate_report(
            removed,
            expected_gate_report_sha256=removed["gate_report_sha256"],
            metrics=stage_metrics,
            expected_stage_metrics_sha256=stage_metrics[
                "stage_metrics_sha256"
            ],
        )
