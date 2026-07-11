from __future__ import annotations

import copy
from datetime import date, timedelta
import hashlib
import math

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    ACTIVE_EDGE_TOLERANCE,
    CANDIDATE_IDS,
    canonical_sha256,
)
from agent_benchmark.sec_filing_gemma_market_evidence import MARKET_ROW_SCHEMA_VERSION
from agent_benchmark.sec_filing_gemma_no_leverage import (
    SecFilingGemmaNoLeverageError,
    validate_sec_gemma_no_leverage_proof,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    AVAILABLE_PREDICTION_STATUS,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS
from agent_benchmark.sec_filing_gemma_scoring import (
    GATE_RECEIPT_SCHEMA_VERSION,
    SCORE_RECEIPT_SCHEMA_VERSION,
    SecFilingGemmaScoringError,
    build_development_ranking_receipt,
    build_score_receipt,
    build_stage_gate_receipt,
    decode_score_float_hex,
    validate_development_ranking_receipt,
    validate_score_receipt,
    validate_stage_gate_receipt,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sessions(start: str, count: int) -> list[str]:
    current = date.fromisoformat(start)
    result: list[str] = []
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _market(stage: str, sessions: list[str], opens: list[float], closes=None) -> dict:
    closes = opens if closes is None else closes
    genesis = _h(f"{stage}-market-genesis")
    parent = genesis
    rows = []
    for index, (session, adjusted_open, adjusted_close) in enumerate(
        zip(sessions, opens, closes, strict=True)
    ):
        body = {
            "schema_version": MARKET_ROW_SCHEMA_VERSION,
            "row_index": index,
            "session": session,
            "observations": {
                "AAPL": {
                    "available": True,
                    "adjusted_open_hex": float(adjusted_open).hex(),
                    "adjusted_close_hex": float(adjusted_close).hex(),
                }
            },
            "previous_row_sha256": parent,
        }
        row = {**body, "row_sha256": canonical_sha256(body)}
        rows.append(row)
        parent = row["row_sha256"]
    body = {
        "schema_version": "test-market-stage-v1",
        "artifact_stage": stage,
        "row_chain_genesis_sha256": genesis,
        "row_count": len(rows),
        "row_chain_tip_sha256": parent,
        "rows": rows,
    }
    return {**body, "market_stage_manifest_sha256": canonical_sha256(body)}


def _state(position: str, origin=None, fill=None, exit_session=None) -> dict:
    return {
        "position_at_decision_close": position,
        "episode_origin_decision_session": origin,
        "episode_fill_session": fill,
        "episode_exit_session": exit_session,
    }


def _prediction_row(
    *,
    sessions: list[str],
    decision_index: int,
    start_semantic: bool,
    start_ablation: bool = False,
    exit_session: str | None = None,
    fill_session: str | None = None,
    probability: float = 0.8,
    ablation_probability: float = 0.5,
    training_set_count: int = 2,
    training_positive_count: int = 1,
) -> dict:
    decision = sessions[decision_index]
    fill = fill_session or sessions[decision_index + 1]
    exit_value = exit_session or sessions[decision_index + 21]
    actions = {
        "p50_e0": {
            "semantic": "START_CASH_EPISODE" if start_semantic else "STAY_LONG",
            "ablation": "START_CASH_EPISODE" if start_ablation else "STAY_LONG",
        }
    }
    inputs = {
        "p50_e0": {
            "semantic": _state("LONG"),
            "ablation": _state("LONG"),
        }
    }
    outputs = {
        "p50_e0": {
            "semantic": (
                _state("CASH", decision, fill, exit_value)
                if start_semantic
                else _state("LONG")
            ),
            "ablation": (
                _state("CASH", decision, fill, exit_value)
                if start_ablation
                else _state("LONG")
            ),
        }
    }
    body = {
        "schema_version": "test-prediction-row-v1",
        "sequence_number": 1,
        "decision_session": decision,
        "fill_session": fill,
        "cash_exit_session": exit_value,
        "label_maturity_session": exit_value,
        "horizon_sessions": 20,
        "fold_id": "fold_1",
        "fold_context": {
            "fold_train_cutoff_session": "2004-12-31",
            "training_set_count": training_set_count,
            "training_positive_count": training_positive_count,
        },
        "prediction_status": AVAILABLE_PREDICTION_STATUS,
        "semantic_cash_probability_hex": float(probability).hex(),
        "ablation_cash_probability_hex": float(ablation_probability).hex(),
        "effective_episode_actions": actions,
        "candidate_policy_input_states": inputs,
        "candidate_policy_output_states": outputs,
    }
    return {**body, "prediction_row_sha256": canonical_sha256(body)}


def _prefix(rows: list[dict]) -> dict:
    body = {
        "schema_version": "test-prediction-prefix-v1",
        "candidate_sha256": _h("candidate"),
        "row_count": len(rows),
        "rows_sha256": canonical_sha256(rows),
        "tip_sha256": rows[-1]["prediction_row_sha256"],
        "rows": rows,
    }
    return {**body, "prediction_prefix_sha256": canonical_sha256(body)}


def _label_ledger(rows: list[dict], market: dict, include: list[bool] | None = None) -> dict:
    include = [True] * len(rows) if include is None else include
    prices = {
        row["session"]: float.fromhex(
            row["observations"]["AAPL"]["adjusted_open_hex"]
        )
        for row in market["rows"]
    }
    entries = []
    for row, keep in zip(rows, include, strict=True):
        if not keep:
            continue
        rate = 10 / 10_000
        edge = (
            math.log1p(-rate)
            - math.log1p(rate)
            - math.log(prices[row["cash_exit_session"]] / prices[row["fill_session"]])
        )
        if edge == 0.0:
            edge = 0.0
        body = {
            "schema_version": "test-label-release-v1",
            "prediction_row_sha256": row["prediction_row_sha256"],
            "decision_session": row["decision_session"],
            "label_maturity_session": row["label_maturity_session"],
            "cost_bps": 10,
            "cash_active_log_edge_10bps_hex": edge.hex(),
            "cash_beats_long_10bps": edge > ACTIVE_EDGE_TOLERANCE,
        }
        entries.append({**body, "release_sha256": canonical_sha256(body)})
    body = {
        "schema_version": "test-label-ledger-v1",
        "release_count": len(entries),
        "entries": entries,
    }
    return {**body, "label_release_ledger_sha256": canonical_sha256(body)}


def _evidence(
    *,
    start: str = "2005-01-03",
    count: int = 30,
    opens: list[float] | None = None,
    closes: list[float] | None = None,
    start_semantic: bool = True,
    start_ablation: bool = False,
    decision_index: int = 0,
    exit_session: str | None = None,
    fill_session: str | None = None,
    include_label: bool = True,
    probability: float = 0.8,
    ablation_probability: float = 0.5,
    training_set_count: int = 2,
    training_positive_count: int = 1,
) -> dict:
    sessions = _sessions(start, count)
    opens = [100.0 + index for index in range(count)] if opens is None else opens
    market = _market("development", sessions, opens, closes)
    row = _prediction_row(
        sessions=sessions,
        decision_index=decision_index,
        start_semantic=start_semantic,
        start_ablation=start_ablation,
        exit_session=exit_session,
        fill_session=fill_session,
        probability=probability,
        ablation_probability=ablation_probability,
        training_set_count=training_set_count,
        training_positive_count=training_positive_count,
    )
    prefix = _prefix([row])
    labels = _label_ledger([row], market, [include_label])
    cutoff = sessions[-1]
    kwargs = {
        "prediction_prefix": prefix,
        "expected_prediction_prefix_sha256": prefix["prediction_prefix_sha256"],
        "label_release_evidence": labels,
        "expected_label_release_ledger_sha256": labels[
            "label_release_ledger_sha256"
        ],
        "market_stage": market,
        "expected_market_stage_manifest_sha256": market[
            "market_stage_manifest_sha256"
        ],
        "selected_candidate_id": "p50_e0",
        "selected_variant": "semantic",
        "cost_bps": 10,
        "stage": "development",
        "score_cutoff_session": cutoff,
        "terminal_convention": "adjusted_open",
    }
    return {"sessions": sessions, "row": row, "market": market, "kwargs": kwargs}


def _intermediate_boundary_evidence(
    *,
    stage_session_count: int,
    start_semantic: bool = True,
    falling_stage_prices: bool = False,
    decision_offset_from_stage_start: int = -5,
    first_stage_open: float | None = None,
) -> dict:
    full_sessions = [
        session
        for session in EXPECTED_SESSIONS
        if "2018-11-01" <= session <= "2019-03-29"
    ]
    stage_start_index = full_sessions.index("2019-01-02")
    decision_index = stage_start_index + decision_offset_from_stage_start
    cutoff_index = stage_start_index + stage_session_count - 1
    market_sessions = full_sessions[: cutoff_index + 1]
    opens = [100.0] * len(market_sessions)
    if falling_stage_prices:
        for offset, index in enumerate(
            range(stage_start_index, len(market_sessions))
        ):
            opens[index] = 100.0 - 5.0 * offset
    if first_stage_open is not None:
        for index in range(stage_start_index, len(market_sessions)):
            opens[index] = first_stage_open
    market = _market("intermediate", market_sessions, opens)
    row = _prediction_row(
        sessions=full_sessions,
        decision_index=decision_index,
        start_semantic=start_semantic,
    )
    prefix = _prefix([row])
    labels = _label_ledger([row], market, [False])
    kwargs = {
        "prediction_prefix": prefix,
        "expected_prediction_prefix_sha256": prefix["prediction_prefix_sha256"],
        "label_release_evidence": labels,
        "expected_label_release_ledger_sha256": labels[
            "label_release_ledger_sha256"
        ],
        "market_stage": market,
        "expected_market_stage_manifest_sha256": market[
            "market_stage_manifest_sha256"
        ],
        "selected_candidate_id": "p50_e0",
        "selected_variant": "semantic",
        "cost_bps": 10,
        "stage": "intermediate",
        "score_cutoff_session": market_sessions[-1],
        "terminal_convention": "adjusted_open",
    }
    return {
        "sessions": market_sessions,
        "stage_start_index": stage_start_index,
        "row": row,
        "market": market,
        "kwargs": kwargs,
    }


def _build(evidence: dict, **overrides) -> dict:
    kwargs = dict(evidence["kwargs"])
    kwargs.update(overrides)
    return build_score_receipt(**kwargs)


def _validate(evidence: dict, receipt: dict, **overrides) -> str:
    kwargs = dict(evidence["kwargs"])
    kwargs.update(overrides)
    return validate_score_receipt(
        receipt,
        expected_score_receipt_sha256=receipt["score_receipt_sha256"],
        **kwargs,
    )


def test_t_plus_1_t_plus_21_two_leg_cost_and_roundtrip_replay() -> None:
    evidence = _evidence()
    receipt = _build(evidence)
    ledger = receipt["ledger"]
    metrics = receipt["metrics"]

    assert evidence["row"]["fill_session"] == evidence["sessions"][1]
    assert evidence["row"]["cash_exit_session"] == evidence["sessions"][21]
    assert [row["target_exposure"] for row in ledger[1:21]] == [0] * 20
    assert ledger[21]["target_exposure"] == 1
    assert sum(row["strategy_position_changed"] for row in ledger) == 3
    assert sum(row["benchmark_position_changed"] for row in ledger) == 1
    episode_edge = decode_score_float_hex(
        metrics["episode_records"][0]["active_log_edge_hex"]
    )
    expected = (
        math.log1p(-0.001)
        - math.log1p(0.001)
        - math.log((100.0 + 21) / (100.0 + 1))
    )
    assert math.isclose(episode_edge, expected, abs_tol=1e-15)
    assert metrics["cash_days"] == 20
    assert metrics["completed_cash_episodes"] == 1
    assert _validate(evidence, receipt) == receipt["score_receipt_sha256"]


def test_five_and_ten_bps_charge_both_episode_legs() -> None:
    evidence = _evidence()
    receipt_5 = _build(evidence, cost_bps=5)
    receipt_10 = _build(evidence, cost_bps=10)
    edge_5 = decode_score_float_hex(
        receipt_5["metrics"]["episode_records"][0]["active_log_edge_hex"]
    )
    edge_10 = decode_score_float_hex(
        receipt_10["metrics"]["episode_records"][0]["active_log_edge_hex"]
    )
    expected_difference = (
        math.log1p(-0.0005)
        - math.log1p(0.0005)
        - math.log1p(-0.001)
        + math.log1p(0.001)
    )
    assert edge_5 > edge_10
    assert math.isclose(edge_5 - edge_10, expected_difference, abs_tol=1e-18)


def test_brier_uses_exact_mature_rows_causal_beta_climatology_and_ablation_tie() -> None:
    evidence = _evidence(
        opens=[100.0 - index for index in range(30)],
        probability=0.8,
        ablation_probability=0.8,
    )
    brier = _build(evidence)["metrics"]["brier"]
    assert brier["row_count"] == 1
    assert decode_score_float_hex(brier["brier_score_hex"]) == pytest.approx(0.04)
    assert decode_score_float_hex(
        brier["causal_climatology_brier_score_hex"]
    ) == 0.25
    assert decode_score_float_hex(brier["ablation_brier_score_hex"]) == pytest.approx(
        0.04
    )
    assert decode_score_float_hex(
        brier["relative_improvement_vs_ablation_hex"]
    ) == 0.0


def test_brier_climatology_uses_the_complete_frozen_training_counts() -> None:
    evidence = _evidence(
        opens=[100.0 - index for index in range(30)],
        probability=0.8,
        training_set_count=100,
        training_positive_count=80,
    )
    brier = _build(evidence)["metrics"]["brier"]
    record = brier["records"][0]
    expected_climatology = 81.0 / 102.0

    assert record["training_set_count"] == 100
    assert record["training_positive_count"] == 80
    assert decode_score_float_hex(
        record["causal_climatology_probability_hex"]
    ) == expected_climatology
    assert decode_score_float_hex(
        brier["causal_climatology_brier_score_hex"]
    ) == pytest.approx((expected_climatology - 1.0) ** 2)


@pytest.mark.parametrize("field", ["fill_session", "cash_exit_session"])
def test_wrong_t_plus_1_or_t_plus_21_is_rejected_even_when_rehashed(field: str) -> None:
    evidence = _evidence()
    prefix = copy.deepcopy(evidence["kwargs"]["prediction_prefix"])
    row = prefix["rows"][0]
    row[field] = evidence["sessions"][2 if field == "fill_session" else 20]
    if field == "cash_exit_session":
        row["label_maturity_session"] = row[field]
    row_body = {key: row[key] for key in row if key != "prediction_row_sha256"}
    row["prediction_row_sha256"] = canonical_sha256(row_body)
    prefix["rows_sha256"] = canonical_sha256(prefix["rows"])
    prefix["tip_sha256"] = row["prediction_row_sha256"]
    prefix_body = {
        key: prefix[key] for key in prefix if key != "prediction_prefix_sha256"
    }
    prefix["prediction_prefix_sha256"] = canonical_sha256(prefix_body)
    with pytest.raises(
        SecFilingGemmaScoringError, match=r"not exactly t\+1|not exactly t\+21"
    ):
        _build(
            evidence,
            prediction_prefix=prefix,
            expected_prediction_prefix_sha256=prefix["prediction_prefix_sha256"],
        )


def test_decision_on_cutoff_has_no_scored_fill_or_episode() -> None:
    all_sessions = _sessions("2005-01-03", 35)
    evidence = _evidence(
        count=10,
        decision_index=9,
        fill_session=all_sessions[10],
        exit_session=all_sessions[30],
        include_label=False,
    )
    receipt = _build(evidence)
    assert all(row["target_exposure"] == 1 for row in receipt["ledger"])
    assert receipt["metrics"]["cash_episodes"] == 0
    assert receipt["metrics"]["open_cash_episodes_at_cutoff"] == 0


def test_open_terminal_episode_is_valued_but_excluded_from_mature_statistics() -> None:
    all_sessions = _sessions("2005-01-03", 35)
    opens = [100.0 + index for index in range(10)]
    closes = list(opens)
    closes[-1] = opens[-1] * 0.8
    evidence = _evidence(
        count=10,
        opens=opens,
        closes=closes,
        exit_session=all_sessions[21],
        include_label=False,
    )
    open_receipt = _build(evidence)
    close_receipt = _build(
        evidence, terminal_convention="terminal_adjusted_close"
    )
    open_metrics = open_receipt["metrics"]
    close_metrics = close_receipt["metrics"]

    assert open_metrics["cash_episodes"] == 1
    assert open_metrics["open_cash_episodes_at_cutoff"] == 1
    assert open_metrics["completed_cash_episodes"] == 0
    assert open_metrics["episode_win_rate_hex"] is None
    assert open_metrics["brier"]["row_count"] == 0
    assert decode_score_float_hex(
        close_metrics["total_active_log_edge_hex"]
    ) > decode_score_float_hex(open_metrics["total_active_log_edge_hex"])


def test_open_episode_contribution_is_included_in_concentration() -> None:
    evidence = _evidence(
        count=10,
        opens=[100.0, 100.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        exit_session=_sessions("2005-01-03", 30)[21],
        include_label=False,
    )
    metrics = _build(evidence)["metrics"]
    record = metrics["episode_records"][0]

    assert record["status"] == "open_at_cutoff"
    assert record["full_horizon_active_log_edge_hex"] is None
    assert metrics["completed_cash_episodes"] == 0
    assert metrics["episode_win_rate_hex"] is None
    assert metrics["mean_episode_active_log_edge_hex"] is None
    assert decode_score_float_hex(
        record["stage_active_log_edge_hex"]
    ) == decode_score_float_hex(metrics["total_active_log_edge_hex"])
    assert decode_score_float_hex(
        metrics["maximum_single_episode_positive_edge_share_hex"]
    ) == 1.0


def test_carry_in_episode_does_not_rebuy_benchmark_at_stage_boundary() -> None:
    evidence = _intermediate_boundary_evidence(stage_session_count=5)
    receipt = _build(evidence)
    first = receipt["ledger"][0]
    metrics = receipt["metrics"]

    assert receipt["configuration"]["initial_strategy_exposure"] == 0
    assert receipt["configuration"]["initial_benchmark_exposure"] == 1
    assert first["target_exposure"] == 0
    assert first["strategy_position_changed"] is False
    assert first["benchmark_position_changed"] is False
    assert decode_score_float_hex(metrics["total_active_log_edge_hex"]) == 0.0
    assert metrics["cash_episodes"] == 0
    assert metrics["contributing_cash_episodes"] == 1
    assert metrics["open_cash_episodes_at_cutoff"] == 0
    assert metrics["open_contributing_cash_episodes_at_cutoff"] == 1


def test_carry_in_episode_includes_the_prior_open_to_boundary_open_return() -> None:
    evidence = _intermediate_boundary_evidence(
        stage_session_count=5,
        first_stage_open=80.0,
    )
    receipt = _build(evidence)
    first = receipt["ledger"][0]
    metrics = receipt["metrics"]

    assert decode_score_float_hex(
        receipt["configuration"]["prior_adjusted_open_hex"]
    ) == 100.0
    assert first["holding_exposure_for_return"] == 0
    assert first["strategy_position_changed"] is False
    assert first["benchmark_position_changed"] is False
    expected = -math.log(80.0 / 100.0)
    assert decode_score_float_hex(metrics["total_active_log_edge_hex"]) == pytest.approx(
        expected, abs=1e-15
    )
    assert decode_score_float_hex(
        metrics["episode_records"][0]["stage_active_log_edge_hex"]
    ) == pytest.approx(expected, abs=1e-15)
    assert decode_score_float_hex(
        metrics["maximum_single_episode_positive_edge_share_hex"]
    ) == 1.0


def test_no_carry_boundary_inherits_long_without_new_entry_cost() -> None:
    evidence = _intermediate_boundary_evidence(
        stage_session_count=5,
        start_semantic=False,
    )
    receipt = _build(evidence)
    first = receipt["ledger"][0]

    assert receipt["configuration"]["initial_strategy_exposure"] == 1
    assert receipt["configuration"]["initial_benchmark_exposure"] == 1
    assert first["strategy_position_changed"] is False
    assert first["benchmark_position_changed"] is False
    assert decode_score_float_hex(
        receipt["metrics"]["total_active_log_edge_hex"]
    ) == 0.0


def test_episode_fill_on_first_real_stage_session_charges_one_sell() -> None:
    evidence = _intermediate_boundary_evidence(
        stage_session_count=5,
        decision_offset_from_stage_start=-1,
    )
    receipt = _build(evidence)
    first = receipt["ledger"][0]

    assert first["session"] == "2019-01-02"
    assert evidence["row"]["fill_session"] == first["session"]
    assert receipt["configuration"]["initial_strategy_exposure"] == 1
    assert first["strategy_position_changed"] is True
    assert first["target_exposure"] == 0
    assert first["benchmark_position_changed"] is False
    assert decode_score_float_hex(
        receipt["metrics"]["total_active_log_edge_hex"]
    ) == pytest.approx(math.log1p(-0.001), abs=1e-15)


def test_episode_exit_on_first_real_stage_session_charges_one_buy() -> None:
    evidence = _intermediate_boundary_evidence(
        stage_session_count=5,
        decision_offset_from_stage_start=-21,
    )
    receipt = _build(evidence)
    first = receipt["ledger"][0]

    assert first["session"] == "2019-01-02"
    assert evidence["row"]["cash_exit_session"] == first["session"]
    assert receipt["configuration"]["initial_strategy_exposure"] == 0
    assert first["strategy_position_changed"] is True
    assert first["target_exposure"] == 1
    assert first["benchmark_position_changed"] is False
    assert decode_score_float_hex(
        receipt["metrics"]["total_active_log_edge_hex"]
    ) == pytest.approx(-math.log1p(0.001), abs=1e-15)


def test_cross_boundary_episode_charges_only_its_in_stage_exit_fill() -> None:
    evidence = _intermediate_boundary_evidence(stage_session_count=20)
    receipt = _build(evidence)
    metrics = receipt["metrics"]
    record = metrics["episode_records"][0]

    assert sum(row["strategy_position_changed"] for row in receipt["ledger"]) == 1
    assert sum(row["benchmark_position_changed"] for row in receipt["ledger"]) == 0
    assert decode_score_float_hex(record["stage_active_log_edge_hex"]) == pytest.approx(
        -math.log1p(0.001), abs=1e-15
    )
    assert decode_score_float_hex(
        record["full_horizon_active_log_edge_hex"]
    ) == pytest.approx(math.log1p(-0.001) - math.log1p(0.001), abs=1e-15)
    assert metrics["cash_episodes"] == 0
    assert metrics["contributing_cash_episodes"] == 1
    assert metrics["completed_cash_episodes"] == 0


def test_carry_in_positive_edge_participates_in_concentration() -> None:
    evidence = _intermediate_boundary_evidence(
        stage_session_count=5,
        falling_stage_prices=True,
    )
    metrics = _build(evidence)["metrics"]

    assert metrics["cash_episodes"] == 0
    assert metrics["contributing_cash_episodes"] == 1
    assert decode_score_float_hex(metrics["total_active_log_edge_hex"]) > 0.0
    assert decode_score_float_hex(
        metrics["maximum_single_episode_positive_edge_share_hex"]
    ) == 1.0
    assert abs(
        decode_score_float_hex(metrics["unattributed_active_log_edge_hex"])
    ) <= ACTIVE_EDGE_TOLERANCE


def test_cross_year_episode_keeps_state_and_attributes_daily_edge_by_session_year() -> None:
    evidence = _evidence(start="2005-12-19", count=30)
    receipt = _build(evidence)
    metrics = receipt["metrics"]
    edges = {
        key: decode_score_float_hex(value)
        for key, value in metrics["period_active_log_edges_hex"].items()
    }
    total = decode_score_float_hex(metrics["total_active_log_edge_hex"])

    assert metrics["period_cash_episode_counts"]["2005"] == 1
    assert metrics["period_cash_episode_counts"]["2006"] == 0
    assert any(
        row["session"].startswith("2006-") and row["target_exposure"] == 0
        for row in receipt["ledger"]
    )
    assert math.isclose(edges["2005"] + edges["2006"], total, abs_tol=1e-15)


def test_negative_buy_and_hold_year_behavior_is_measured_directly() -> None:
    opens = [100.0 - index for index in range(30)]
    evidence = _evidence(opens=opens)
    receipt = _build(evidence)
    metrics = receipt["metrics"]

    assert "2005" in metrics["negative_buy_hold_periods"]
    assert decode_score_float_hex(
        metrics["period_buy_hold_log_returns_hex"]["2005"]
    ) < 0.0
    assert decode_score_float_hex(
        metrics["negative_buy_hold_period_active_log_edges_hex"]["2005"]
    ) > ACTIVE_EDGE_TOLERANCE
    assert decode_score_float_hex(
        metrics["negative_buy_hold_period_win_rate_hex"]
    ) == 1.0


def test_252_and_756_session_month_end_windows_use_strict_win_tolerance() -> None:
    count = 800
    evidence = _evidence(
        count=count,
        opens=[100.0 + index / 10.0 for index in range(count)],
    )
    metrics = _build(evidence)["metrics"]
    assert metrics["rolling_252_session_month_end_observations"] > 0
    assert metrics["rolling_756_session_month_end_observations"] > 0
    assert decode_score_float_hex(
        metrics["rolling_252_session_month_end_win_rate_hex"]
    ) == 0.0
    assert decode_score_float_hex(
        metrics["rolling_756_session_month_end_win_rate_hex"]
    ) == 0.0


def _rehash_tampered_score(receipt: dict) -> dict:
    value = copy.deepcopy(receipt)
    parent = value["ledger_genesis_sha256"]
    for index, row in enumerate(value["ledger"]):
        row["row_index"] = index
        row["previous_ledger_row_sha256"] = parent
        body = {key: row[key] for key in row if key != "ledger_row_sha256"}
        row["ledger_row_sha256"] = canonical_sha256(body)
        parent = row["ledger_row_sha256"]
    value["ledger_tip_sha256"] = parent
    value["ledger_row_count"] = len(value["ledger"])
    value["ledger_rows_sha256"] = canonical_sha256(value["ledger"])
    body = {key: value[key] for key in value if key != "score_receipt_sha256"}
    value["score_receipt_sha256"] = canonical_sha256(body)
    return value


@pytest.mark.parametrize("exposure", [0.5, math.nextafter(1.0, math.inf)])
def test_fractional_or_one_ulp_leveraged_exposure_fails_replay(exposure: float) -> None:
    evidence = _evidence()
    forged = copy.deepcopy(_build(evidence))
    forged["ledger"][0]["target_exposure"] = exposure
    forged = _rehash_tampered_score(forged)
    with pytest.raises(SecFilingGemmaScoringError, match="deterministic ledger replay"):
        _validate(evidence, forged)


@pytest.mark.parametrize("mutation", ["benchmark_price", "benchmark_session"])
def test_different_benchmark_prices_or_sessions_fail_same_ledger_replay(
    mutation: str,
) -> None:
    evidence = _evidence()
    forged = copy.deepcopy(_build(evidence))
    if mutation == "benchmark_price":
        forged["ledger"][2]["benchmark_wealth_hex"] = (999.0).hex()
    else:
        forged["ledger"][2]["session"] = "2005-12-30"
    forged = _rehash_tampered_score(forged)
    with pytest.raises(SecFilingGemmaScoringError, match="deterministic ledger replay"):
        _validate(evidence, forged)


def test_rehashed_metric_or_terminal_tamper_is_rejected() -> None:
    evidence = _evidence()
    forged = copy.deepcopy(_build(evidence))
    forged["metrics"]["cash_days"] += 1
    body = {key: forged[key] for key in forged if key != "score_receipt_sha256"}
    forged["score_receipt_sha256"] = canonical_sha256(body)
    with pytest.raises(SecFilingGemmaScoringError, match="deterministic ledger replay"):
        _validate(evidence, forged)


def test_independent_zero_tolerance_no_leverage_proof_replays_valid_score() -> None:
    receipt = _build(_evidence())

    proof = validate_sec_gemma_no_leverage_proof(
        receipt,
        expected_score_receipt_sha256=receipt["score_receipt_sha256"],
    )

    assert proof["exact_binary_exposure"] is True
    assert proof["exact_cash_share_identity"] is True
    assert proof["exact_transaction_cost_replay"] is True
    assert decode_score_float_hex(proof["proof_tolerance_hex"]) == 0.0
    assert proof["shorting"] is False
    assert proof["borrowing"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "fractional_target",
        "leveraged_shares",
        "negative_cash",
        "margin_debt",
        "wealth",
        "genesis",
    ],
)
def test_independent_no_leverage_proof_rejects_rehashed_execution_tamper(
    mutation: str,
) -> None:
    forged = copy.deepcopy(_build(_evidence()))
    if mutation == "fractional_target":
        forged["ledger"][0]["target_exposure"] = 0.5
    elif mutation == "leveraged_shares":
        shares = decode_score_float_hex(forged["ledger"][0]["strategy_shares_hex"])
        forged["ledger"][0]["strategy_shares_hex"] = (shares * 2.0).hex()
    elif mutation == "negative_cash":
        forged["ledger"][1]["strategy_cash_hex"] = (-1.0).hex()
    elif mutation == "margin_debt":
        forged["ledger"][0]["margin_debt_hex"] = math.nextafter(0.0, 1.0).hex()
    elif mutation == "wealth":
        wealth = decode_score_float_hex(forged["ledger"][0]["strategy_wealth_hex"])
        forged["ledger"][0]["strategy_wealth_hex"] = math.nextafter(
            wealth, math.inf
        ).hex()
    else:
        forged["ledger_genesis_sha256"] = _h("forged ledger genesis")
    forged = _rehash_tampered_score(forged)

    with pytest.raises(SecFilingGemmaNoLeverageError):
        validate_sec_gemma_no_leverage_proof(
            forged,
            expected_score_receipt_sha256=forged["score_receipt_sha256"],
        )


def test_independent_no_leverage_proof_requires_external_score_pin() -> None:
    receipt = _build(_evidence())

    with pytest.raises(SecFilingGemmaNoLeverageError, match="externally pinned"):
        validate_sec_gemma_no_leverage_proof(
            receipt,
            expected_score_receipt_sha256=_h("wrong score pin"),
        )


def test_rehashed_label_edge_or_binary_boundary_tamper_is_recomputed_from_market() -> None:
    evidence = _evidence()
    labels = copy.deepcopy(evidence["kwargs"]["label_release_evidence"])
    entry = labels["entries"][0]
    edge = float.fromhex(entry["cash_active_log_edge_10bps_hex"])
    entry["cash_active_log_edge_10bps_hex"] = math.nextafter(
        edge, math.inf
    ).hex()
    body = {key: entry[key] for key in entry if key != "release_sha256"}
    entry["release_sha256"] = canonical_sha256(body)
    ledger_body = {
        key: labels[key]
        for key in labels
        if key != "label_release_ledger_sha256"
    }
    labels["label_release_ledger_sha256"] = canonical_sha256(ledger_body)
    with pytest.raises(SecFilingGemmaScoringError, match="reconcile to exact market"):
        _build(
            evidence,
            label_release_evidence=labels,
            expected_label_release_ledger_sha256=labels[
                "label_release_ledger_sha256"
            ],
        )


def _fh(value: float) -> str:
    number = float(value)
    if number == 0.0:
        number = 0.0
    return number.hex()


def _synthetic_metrics(
    stage: str,
    *,
    total: float,
    period_edge: float,
    ablation_relative: float = 0.01,
    concentration: float = 0.5,
    drawdown_disadvantage: float = 1.0,
    mean_episode: float = 0.001,
) -> dict:
    periods = {
        "development": [str(year) for year in range(2005, 2019)],
        "intermediate": [str(year) for year in range(2019, 2024)],
        "final": ["2024", "2025", "2026_ytd"],
    }[stage]
    folds = {
        f"fold_{index}": _fh(0.001) for index in range(1, 6)
    } if stage == "development" else {}
    return {
        "total_active_log_edge_hex": _fh(total),
        "period_active_log_edges_hex": {
            period: _fh(period_edge) for period in periods
        },
        "period_buy_hold_log_returns_hex": {
            period: _fh(-0.01 if period == periods[0] else 0.01)
            for period in periods
        },
        "annual_win_rate_hex": _fh(0.60),
        "winning_year_count": 3 if stage == "intermediate" else len(periods),
        "median_annual_active_log_edge_hex": _fh(0.0005),
        "active_log_edge_without_best_year_hex": _fh(0.005),
        "largest_positive_year_share_hex": _fh(
            0.45 if stage == "development" else 0.60
        ),
        "fold_active_log_edges_hex": folds,
        "positive_fold_count": 5 if stage == "development" else 0,
        "rolling_252_session_month_end_win_rate_hex": _fh(0.60),
        "rolling_756_session_month_end_win_rate_hex": _fh(0.70),
        "cash_days": 30,
        "cash_episodes": 12 if stage != "final" else 6,
        "cash_day_rate_hex": _fh(0.20),
        "period_cash_episode_counts": {
            period: (2 if stage == "final" else 1) for period in periods
        },
        "episode_win_rate_hex": _fh(0.55),
        "mean_episode_active_log_edge_hex": _fh(mean_episode),
        "maximum_single_episode_positive_edge_share_hex": _fh(concentration),
        "aggregate_active_log_edge_in_negative_buy_hold_periods_hex": _fh(0.01),
        "negative_buy_hold_period_win_rate_hex": _fh(0.60),
        "drawdown_disadvantage_percentage_points_hex": _fh(
            drawdown_disadvantage
        ),
        "brier": {
            "brier_score_hex": _fh(0.10),
            "relative_improvement_vs_climatology_hex": _fh(0.02),
            "relative_improvement_vs_ablation_hex": _fh(ablation_relative),
        },
    }


def _synthetic_score(
    *,
    stage: str,
    candidate: str,
    variant: str,
    cost: int,
    terminal: str,
    metrics: dict,
) -> dict:
    body = {
        "schema_version": SCORE_RECEIPT_SCHEMA_VERSION,
        "contract_sha256": _h("contract"),
        "inputs": {
            "prediction_prefix_sha256": _h("prediction"),
            "label_release_ledger_sha256": _h("labels"),
            "market_stage_manifest_sha256": _h("market"),
        },
        "configuration": {
            "stage": stage,
            "score_start_session": {
                "development": "2005-01-03",
                "intermediate": "2019-01-02",
                "final": "2024-01-02",
            }[stage],
            "score_cutoff_session": {
                "development": "2018-12-31",
                "intermediate": "2023-12-29",
                "final": "2026-07-09",
            }[stage],
            "selected_candidate_id": candidate,
            "selected_variant": variant,
            "cost_bps": cost,
            "terminal_convention": terminal,
        },
        "ledger_genesis_sha256": _h("ledger-genesis"),
        "ledger_tip_sha256": _h("ledger-tip"),
        "ledger_row_count": 1,
        "ledger_rows_sha256": _h("ledger-rows"),
        "ledger": [{}],
        "terminal": {},
        "metrics": metrics,
    }
    return {**body, "score_receipt_sha256": canonical_sha256(body)}


def _scenario_set(stage: str, *, candidate: str = "p50_e0") -> list[dict]:
    terminals = (
        ["adjusted_open", "terminal_adjusted_close"]
        if stage == "final"
        else ["adjusted_open"]
    )
    receipts = []
    for variant in ("semantic", "ablation"):
        for cost in (5, 10):
            for terminal in terminals:
                if stage == "development":
                    total = 0.02 if cost == 5 else (
                        math.nextafter(0.03, math.inf)
                        if variant == "semantic"
                        else 0.02
                    )
                    period_edge = 0.002
                elif stage == "intermediate":
                    if cost == 5:
                        total = 0.0125 if variant == "semantic" else 0.01
                    else:
                        total = 0.001 if variant == "semantic" else -0.001
                    period_edge = 0.002
                else:
                    if cost == 5:
                        total = 0.02 if variant == "semantic" else 0.0
                        period_edge = 0.005 if variant == "semantic" else 0.0
                    else:
                        total = 0.003 if variant == "semantic" else 0.0
                        period_edge = 0.001 if variant == "semantic" else 0.0
                metrics = _synthetic_metrics(
                    stage,
                    total=total,
                    period_edge=period_edge,
                )
                receipts.append(
                    _synthetic_score(
                        stage=stage,
                        candidate=candidate,
                        variant=variant,
                        cost=cost,
                        terminal=terminal,
                        metrics=metrics,
                    )
                )
    return receipts


def _gate(receipts: list[dict], stage: str) -> dict:
    return build_stage_gate_receipt(
        receipts,
        expected_score_receipt_sha256s=[
            receipt["score_receipt_sha256"] for receipt in receipts
        ],
        stage=stage,
    )


def _rehash_score(receipt: dict) -> dict:
    value = copy.deepcopy(receipt)
    body = {key: value[key] for key in value if key != "score_receipt_sha256"}
    value["score_receipt_sha256"] = canonical_sha256(body)
    return value


def test_development_inclusive_boundaries_and_ablation_fold_ties_pass() -> None:
    receipts = _scenario_set("development")
    gate = _gate(receipts, "development")
    assert gate["schema_version"] == GATE_RECEIPT_SCHEMA_VERSION
    assert gate["passed"] is True
    assert gate["checks"]["each_fold_edge_vs_ablation_5bps"] is True
    assert gate["checks"]["each_fold_edge_vs_ablation_10bps"] is True
    assert validate_stage_gate_receipt(
        gate,
        expected_gate_receipt_sha256=gate["gate_receipt_sha256"],
        score_receipts=receipts,
        expected_score_receipt_sha256s=[
            receipt["score_receipt_sha256"] for receipt in receipts
        ],
        stage="development",
    ) == gate["gate_receipt_sha256"]


def test_one_ulp_ablation_fold_advantage_breaks_inclusive_no_worse_gate() -> None:
    receipts = _scenario_set("development")
    target = next(
        receipt
        for receipt in receipts
        if receipt["configuration"]["selected_variant"] == "ablation"
        and receipt["configuration"]["cost_bps"] == 10
    )
    target["metrics"]["fold_active_log_edges_hex"]["fold_1"] = _fh(
        math.nextafter(0.001, math.inf)
    )
    changed = _rehash_score(target)
    receipts[receipts.index(target)] = changed
    gate = _gate(receipts, "development")
    assert gate["checks"]["each_fold_edge_vs_ablation_10bps"] is False
    assert gate["passed"] is False


def test_intermediate_exclusive_zero_and_positive_dust_boundaries() -> None:
    receipts = _scenario_set("intermediate")
    semantic_10 = next(
        receipt
        for receipt in receipts
        if receipt["configuration"]["selected_variant"] == "semantic"
        and receipt["configuration"]["cost_bps"] == 10
    )
    semantic_10["metrics"]["total_active_log_edge_hex"] = _fh(0.0)
    semantic_10["metrics"]["mean_episode_active_log_edge_hex"] = _fh(0.0)
    changed = _rehash_score(semantic_10)
    receipts[receipts.index(semantic_10)] = changed
    zero_gate = _gate(receipts, "intermediate")
    assert zero_gate["checks"]["10bps_total_active_log_edge_exclusive"] is False
    assert zero_gate["checks"]["10bps_mean_episode_edge_exclusive"] is False

    dust = math.nextafter(0.0, math.inf)
    changed["metrics"]["total_active_log_edge_hex"] = _fh(dust)
    changed["metrics"]["mean_episode_active_log_edge_hex"] = _fh(dust)
    changed = _rehash_score(changed)
    for index, receipt in enumerate(receipts):
        if (
            receipt["configuration"]["selected_variant"] == "semantic"
            and receipt["configuration"]["cost_bps"] == 10
        ):
            receipts[index] = changed
    dust_gate = _gate(receipts, "intermediate")
    assert dust_gate["checks"]["10bps_total_active_log_edge_exclusive"] is True
    assert dust_gate["checks"]["10bps_mean_episode_edge_exclusive"] is True


@pytest.mark.parametrize(
    ("field", "value", "check_fragment"),
    [
        (
            "maximum_single_episode_positive_edge_share_hex",
            math.nextafter(0.5, math.inf),
            "episode_positive_edge_concentration",
        ),
        (
            "drawdown_disadvantage_percentage_points_hex",
            math.nextafter(1.0, math.inf),
            "drawdown_disadvantage",
        ),
    ],
)
def test_final_concentration_and_drawdown_exact_boundaries(
    field: str, value: float, check_fragment: str
) -> None:
    receipts = _scenario_set("final")
    passing = _gate(receipts, "final")
    assert passing["passed"] is True
    for index, receipt in enumerate(receipts):
        if receipt["configuration"]["selected_variant"] == "semantic":
            receipt["metrics"][field] = _fh(value)
            receipts[index] = _rehash_score(receipt)
    failed = _gate(receipts, "final")
    assert any(
        not result and check_fragment in name
        for name, result in failed["checks"].items()
    )
    assert failed["passed"] is False


def test_final_ablation_ties_fail_exclusive_advantage_and_period_count() -> None:
    receipts = _scenario_set("final")
    semantic_by_key = {
        (
            receipt["configuration"]["cost_bps"],
            receipt["configuration"]["terminal_convention"],
        ): receipt
        for receipt in receipts
        if receipt["configuration"]["selected_variant"] == "semantic"
    }
    for index, receipt in enumerate(receipts):
        config = receipt["configuration"]
        if config["selected_variant"] == "ablation" and config["cost_bps"] == 10:
            semantic = semantic_by_key[(10, config["terminal_convention"])]
            receipt["metrics"]["total_active_log_edge_hex"] = semantic["metrics"][
                "total_active_log_edge_hex"
            ]
            receipt["metrics"]["period_active_log_edges_hex"] = copy.deepcopy(
                semantic["metrics"]["period_active_log_edges_hex"]
            )
            receipts[index] = _rehash_score(receipt)
    gate = _gate(receipts, "final")
    assert all(
        not result
        for name, result in gate["checks"].items()
        if "vs_ablation" in name or "periods_beating_ablation" in name
    )


def test_development_ranking_uses_frozen_order_after_exact_ties() -> None:
    gates = []
    for candidate in CANDIDATE_IDS:
        gate = _gate(_scenario_set("development", candidate=candidate), "development")
        gates.append(gate)
    ranking = build_development_ranking_receipt(
        gates,
        expected_gate_receipt_sha256s=[
            gate["gate_receipt_sha256"] for gate in gates
        ],
    )
    assert ranking["selected_candidate_id"] == CANDIDATE_IDS[0]
    assert validate_development_ranking_receipt(
        ranking,
        expected_ranking_receipt_sha256=ranking["ranking_receipt_sha256"],
        gate_receipts=gates,
        expected_gate_receipt_sha256s=[
            gate["gate_receipt_sha256"] for gate in gates
        ],
    ) == ranking["ranking_receipt_sha256"]
