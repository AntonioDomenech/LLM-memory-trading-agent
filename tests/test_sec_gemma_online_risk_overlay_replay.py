from __future__ import annotations

import copy
from typing import Any

import pytest

from agent_benchmark import sec_gemma_online_risk_overlay_replay as replay_module
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_replay import (
    SecGemmaOnlineRiskOverlayReplayError,
    replay_sec_gemma_online_risk_overlay_chronology,
    validate_sec_gemma_online_risk_overlay_chronology,
)
from tests import test_sec_gemma_online_risk_overlay_learner as learner_helpers
from tests import test_sec_gemma_online_risk_overlay_ledger as ledger_helpers


def _market(count: int) -> list[dict[str, Any]]:
    return ledger_helpers._market(count)


def _signals(
    market: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return ledger_helpers._signals(market)


def _feature(
    ordinal: int,
    market: list[dict[str, Any]],
    position: int,
    *,
    available: bool = True,
    acceptance_datetime: str | None | object = ...,
) -> dict[str, Any]:
    row = learner_helpers._feature(
        ordinal,
        available=available,
        decision_session=market[position]["session"],
    )
    if acceptance_datetime is not ...:
        row["acceptance_datetime"] = acceptance_datetime
        learner_helpers._rehash_feature(row)
    return row


def _run(
    market: list[dict[str, Any]],
    features: list[dict[str, Any]],
    *,
    replay_id: str = "synthetic",
    frozen_before_boundary: str | None = None,
) -> dict[str, Any]:
    signals = _signals(market)
    return replay_sec_gemma_online_risk_overlay_chronology(
        market_rows=market,
        expected_market_rows_sha256=canonical_sha256(market),
        baseline_signals=signals,
        expected_baseline_signals_sha256=canonical_sha256(signals),
        feature_rows=features,
        expected_feature_row_sha256s=[
            row["feature_row_sha256"] for row in features
        ],
        arm="semantic",
        replay_id=replay_id,
        frozen_before_boundary=frozen_before_boundary,
    )


def _prediction_by_accession(
    replay: dict[str, Any],
    accession: str,
    *,
    frozen: bool = False,
) -> dict[str, Any]:
    branch = (
        replay["frozen_control"] if frozen else replay["primary"]
    )
    assert branch is not None
    return next(
        row
        for row in branch["predictions"]
        if row["accession_number"] == accession
    )


def test_t_plus_21_label_is_admitted_at_same_session_close() -> None:
    market = _market(45)
    origin = _feature(1, market, 2)
    maturity_position = 2 + 21
    current = _feature(2, market, maturity_position)

    replay = _run(market, [origin, current])

    assert replay["label_intents"][0]["maturity_position"] == 23
    assert replay["learner_lessons"][0]["maturity_session"] == market[
        maturity_position
    ]["session"]
    prediction = _prediction_by_accession(
        replay, current["accession_number"]
    )
    assert prediction["fit_audit"]["supplied_matured_lesson_count"] == 1
    assert prediction["fit_audit"]["admitted_lessons"][0][
        "accession_number"
    ] == origin["accession_number"]


def test_future_outcome_is_not_materialized_before_earlier_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market(45)
    origin = _feature(1, market, 2)
    maturity_position = 23
    current = _feature(2, market, maturity_position)
    events: list[tuple[str, str, str, int]] = []
    original_lesson = replay_module.build_mature_counterfactual_lesson
    original_prediction = (
        replay_module.build_online_overlay_prediction_from_fit
    )

    def observed_lesson(**kwargs: Any) -> dict[str, Any]:
        prefix = kwargs["market_rows"]
        decision_position = next(
            index
            for index, row in enumerate(market)
            if row["session"] == kwargs["decision_session"]
        )
        events.append(
            (
                "lesson",
                kwargs["accession_number"],
                kwargs["as_of_session"],
                len(prefix),
            )
        )
        assert prefix[-1]["session"] == kwargs["as_of_session"]
        assert len(prefix) == decision_position + 22
        return original_lesson(**kwargs)

    def observed_prediction(**kwargs: Any) -> dict[str, Any]:
        feature = kwargs["feature_row"]
        events.append(
            (
                "prediction",
                feature["accession_number"],
                feature["decision_session"],
                0,
            )
        )
        return original_prediction(**kwargs)

    monkeypatch.setattr(
        replay_module,
        "build_mature_counterfactual_lesson",
        observed_lesson,
    )
    monkeypatch.setattr(
        replay_module,
        "build_online_overlay_prediction_from_fit",
        observed_prediction,
    )

    _run(market, [origin, current], replay_id="outcome-order")

    origin_prediction = events.index(
        (
            "prediction",
            origin["accession_number"],
            origin["decision_session"],
            0,
        )
    )
    origin_lesson = events.index(
        (
            "lesson",
            origin["accession_number"],
            market[maturity_position]["session"],
            maturity_position + 1,
        )
    )
    current_prediction = events.index(
        (
            "prediction",
            current["accession_number"],
            current["decision_session"],
            0,
        )
    )
    assert origin_prediction < origin_lesson < current_prediction


def test_frozen_boundary_excludes_equal_maturity_but_includes_prior_no_filing() -> None:
    market = _market(55)
    prior = _feature(1, market, 2)
    equal = _feature(2, market, 4)
    boundary_position = 25
    boundary = market[boundary_position]["session"]
    current = _feature(3, market, boundary_position)

    replay = _run(
        market,
        [prior, equal, current],
        frozen_before_boundary=boundary,
    )

    online = _prediction_by_accession(
        replay, current["accession_number"]
    )
    frozen = _prediction_by_accession(
        replay, current["accession_number"], frozen=True
    )
    assert prior["decision_session"] != market[23]["session"]
    assert online["fit_audit"]["supplied_matured_lesson_count"] == 2
    assert frozen["fit_audit"]["supplied_matured_lesson_count"] == 1
    assert frozen["fit_audit"]["admitted_lessons"][0][
        "accession_number"
    ] == prior["accession_number"]
    assert frozen["fit_as_of_session"] == market[
        boundary_position - 1
    ]["session"]
    assert frozen["frozen_fit_reuse"] is True


def test_same_session_filings_share_fit_and_ambiguity_fails_preconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market(35)
    session = market[10]["session"]
    first = _feature(
        1,
        market,
        10,
        acceptance_datetime=session.replace("-", "") + "160000",
    )
    second = _feature(
        2,
        market,
        10,
        acceptance_datetime=session.replace("-", "") + "170000",
    )
    replay = _run(market, [second, first])

    predictions = replay["primary"]["predictions"]
    assert [row["accession_number"] for row in predictions] == [
        first["accession_number"],
        second["accession_number"],
    ]
    assert predictions[0]["fit_audit_sha256"] == predictions[1][
        "fit_audit_sha256"
    ]
    assert len(replay["primary"]["fit_checkpoints"]) == 1

    ambiguous = copy.deepcopy(second)
    ambiguous["acceptance_datetime"] = None
    learner_helpers._rehash_feature(ambiguous)

    def forbidden_output(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("artifact construction occurred before ambiguity")

    monkeypatch.setattr(
        "agent_benchmark.sec_gemma_online_risk_overlay_replay."
        "build_combined_target_rows",
        forbidden_output,
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayReplayError,
        match="ambiguous",
    ):
        _run(market, [first, ambiguous], replay_id="ambiguous")


def test_unavailable_feature_matures_as_audit_only_lesson() -> None:
    market = _market(35)
    unavailable = _feature(1, market, 2, available=False)

    replay = _run(market, [unavailable])

    counterfactual = replay["counterfactual_lessons"][0]
    lesson = replay["learner_lessons"][0]
    terminal = replay["primary"]["terminal_fit_audit"]
    assert counterfactual["audit_only"] is True
    assert counterfactual["train_eligible"] is False
    assert lesson["trainable"] is False
    assert terminal["supplied_matured_lesson_count"] == 1
    assert terminal["audit_only_lesson_count"] == 1
    assert terminal["training_row_count"] == 0


def test_terminal_fit_admits_maturity_without_a_terminal_filing() -> None:
    market = _market(30)
    feature = _feature(1, market, 2)

    replay = _run(market, [feature])

    assert replay["learner_lessons"][0]["maturity_session"] == market[23][
        "session"
    ]
    terminal = replay["primary"]["terminal_fit_audit"]
    assert terminal["as_of_session"] == market[-1]["session"]
    assert terminal["supplied_matured_lesson_count"] == 1
    assert replay["primary"]["fit_checkpoints"][0]["as_of_session"] == market[
        2
    ]["session"]


def test_terminal_pending_label_and_next_open_overlay_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market(12)
    feature = _feature(1, market, 11)

    def passing_prediction(
        *,
        feature_row: dict[str, Any],
        expected_feature_row_sha256: str,
        fit_audit: dict[str, Any],
        expected_fit_audit_sha256: str,
    ) -> dict[str, Any]:
        assert expected_feature_row_sha256 == feature_row[
            "feature_row_sha256"
        ]
        assert expected_fit_audit_sha256 == fit_audit["fit_audit_sha256"]
        body = {
            "accession_number": feature_row["accession_number"],
            "decision_session": feature_row["decision_session"],
            "acceptance_datetime": feature_row["acceptance_datetime"],
            "prediction_available": True,
            "learner_ready": True,
            "raw_gate_pass": True,
            "fit_audit_sha256": fit_audit["fit_audit_sha256"],
        }
        return {
            **body,
            "prediction_row_sha256": canonical_sha256(body),
        }

    monkeypatch.setattr(
        "agent_benchmark.sec_gemma_online_risk_overlay_replay."
        "build_online_overlay_prediction_from_fit",
        passing_prediction,
    )
    validation_calls: list[str] = []

    def validate_passing_prediction(
        prediction_row: dict[str, Any],
        *,
        expected_prediction_row_sha256: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert prediction_row["prediction_row_sha256"] == (
            expected_prediction_row_sha256
        )
        validation_calls.append(expected_prediction_row_sha256)
        return copy.deepcopy(prediction_row)

    monkeypatch.setattr(
        "agent_benchmark.sec_gemma_online_risk_overlay_replay."
        "validate_online_overlay_prediction_from_fit",
        validate_passing_prediction,
    )
    replay = _run(market, [feature], replay_id="pending")

    assert len(validation_calls) == 1
    assert replay["pending_label_intents"] == replay["label_intents"]
    pending = replay["primary"]["target_stream"][
        "terminal_pending_state"
    ]
    assert pending["baseline_next_open_action"][
        "realization_status"
    ] == "pending_next_open"
    assert pending["sec_overlay_pending_boundary"][
        "realization_status"
    ] == "pending_entry"
    assert replay["primary"]["policy_replay"]["policy_actions"][0][
        "schedule_overlay"
    ] is True


def test_frozen_control_clones_active_prefork_policy_and_ledger_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _market(60)
    before = _feature(1, market, 10)
    after = _feature(2, market, 35)
    boundary_position = 15
    boundary = market[boundary_position]["session"]

    def branch_sensitive_prediction(
        *,
        feature_row: dict[str, Any],
        expected_feature_row_sha256: str,
        fit_audit: dict[str, Any],
        expected_fit_audit_sha256: str,
    ) -> dict[str, Any]:
        assert expected_feature_row_sha256 == feature_row[
            "feature_row_sha256"
        ]
        assert expected_fit_audit_sha256 == fit_audit["fit_audit_sha256"]
        gate = (
            feature_row["accession_number"] == before["accession_number"]
            or fit_audit["as_of_session"] == feature_row["decision_session"]
        )
        body = {
            "accession_number": feature_row["accession_number"],
            "decision_session": feature_row["decision_session"],
            "acceptance_datetime": feature_row["acceptance_datetime"],
            "prediction_available": True,
            "learner_ready": True,
            "raw_gate_pass": gate,
            "fit_audit_sha256": fit_audit["fit_audit_sha256"],
        }
        return {
            **body,
            "prediction_row_sha256": canonical_sha256(body),
        }

    monkeypatch.setattr(
        "agent_benchmark.sec_gemma_online_risk_overlay_replay."
        "build_online_overlay_prediction_from_fit",
        branch_sensitive_prediction,
    )
    monkeypatch.setattr(
        "agent_benchmark.sec_gemma_online_risk_overlay_replay."
        "validate_online_overlay_prediction_from_fit",
        lambda prediction_row, **kwargs: copy.deepcopy(prediction_row),
    )
    replay = _run(
        market,
        [before, after],
        replay_id="fork",
        frozen_before_boundary=boundary,
    )

    primary = replay["primary"]
    frozen = replay["frozen_control"]
    assert frozen is not None
    assert primary["policy_replay"]["policy_actions"][0] == frozen[
        "policy_replay"
    ]["policy_actions"][0]
    assert primary["policy_replay"]["scheduled_overlays"][0] == frozen[
        "policy_replay"
    ]["scheduled_overlays"][0]
    assert primary["target_stream"]["target_rows"][: boundary_position + 1] == (
        frozen["target_stream"]["target_rows"][: boundary_position + 1]
    )
    for cost in ("cost_5bps", "cost_10bps"):
        assert primary["ledgers"][cost]["combined"]["ledger_rows"][
            : boundary_position + 1
        ] == frozen["ledgers"][cost]["combined"]["ledger_rows"][
            : boundary_position + 1
        ]
    assert primary["policy_replay"]["policy_actions"][1][
        "schedule_overlay"
    ] is True
    assert frozen["policy_replay"]["policy_actions"][1][
        "schedule_overlay"
    ] is False


def test_prefix_extension_preserves_all_prior_causal_rows() -> None:
    full_market = _market(65)
    short_market = full_market[:40]
    first = _feature(1, full_market, 2)
    second = _feature(2, full_market, 10)
    later = _feature(3, full_market, 45)

    short = _run(
        short_market,
        [first, second],
        replay_id="prefix",
    )
    extended = _run(
        full_market,
        [first, second, later],
        replay_id="prefix",
    )

    assert short["label_intents"] == extended["label_intents"][:2]
    assert short["learner_lessons"] == extended["learner_lessons"][:2]
    assert short["primary"]["predictions"] == extended["primary"][
        "predictions"
    ][:2]
    assert short["primary"]["policy_replay"]["policy_actions"] == extended[
        "primary"
    ]["policy_replay"]["policy_actions"][:2]
    assert short["primary"]["target_stream"]["target_rows"] == extended[
        "primary"
    ]["target_stream"]["target_rows"][: len(short_market)]
    for cost in ("cost_5bps", "cost_10bps"):
        for name in ("combined", "baseline", "aapl_buy_and_hold"):
            assert short["primary"]["ledgers"][cost][name][
                "ledger_rows"
            ] == extended["primary"]["ledgers"][cost][name][
                "ledger_rows"
            ][: len(short_market)]


def test_external_pins_and_complete_replay_validation_are_exact() -> None:
    market = _market(28)
    feature = _feature(1, market, 2)
    signals = _signals(market)
    replay = _run(market, [feature], replay_id="validation")

    assert (
        validate_sec_gemma_online_risk_overlay_chronology(
            replay,
            expected_chronological_replay_sha256=replay[
                "chronological_replay_sha256"
            ],
            market_rows=market,
            expected_market_rows_sha256=canonical_sha256(market),
            baseline_signals=signals,
            expected_baseline_signals_sha256=canonical_sha256(signals),
            feature_rows=[feature],
            expected_feature_row_sha256s=[
                feature["feature_row_sha256"]
            ],
            arm="semantic",
            replay_id="validation",
        )
        == replay["chronological_replay_sha256"]
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayReplayError,
        match="expected feature hashes do not match",
    ):
        replay_sec_gemma_online_risk_overlay_chronology(
            market_rows=market,
            expected_market_rows_sha256=canonical_sha256(market),
            baseline_signals=signals,
            expected_baseline_signals_sha256=canonical_sha256(signals),
            feature_rows=[feature],
            expected_feature_row_sha256s=["f" * 64],
            arm="semantic",
            replay_id="bad-pin",
        )


def test_portfolio_rejects_a_market_prefix_before_the_2000_genesis() -> None:
    market = _market(8)
    market[0]["session"] = "1999-12-31"
    signals = _signals(market)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayReplayError,
        match="2000-01-03",
    ):
        replay_sec_gemma_online_risk_overlay_chronology(
            market_rows=market,
            expected_market_rows_sha256=canonical_sha256(market),
            baseline_signals=signals,
            expected_baseline_signals_sha256=canonical_sha256(signals),
            feature_rows=[],
            expected_feature_row_sha256s=[],
            arm="semantic",
            replay_id="bad-genesis",
        )
