from __future__ import annotations

import copy
from typing import Any

import pandas as pd
import pytest

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_policy import (
    SecGemmaOnlineRiskOverlayPolicyError,
    replay_nonoverlapping_overlay_policy,
    validate_overlay_policy_replay,
)


def _sessions(count: int = 80) -> list[str]:
    return [
        stamp.strftime("%Y-%m-%d")
        for stamp in pd.bdate_range("2000-01-03", periods=count)
    ]


def _prediction(
    sessions: list[str],
    *,
    position: int,
    accession: str,
    acceptance: str | None,
    available: bool = True,
    ready: bool = True,
    passes: bool = True,
) -> dict[str, Any]:
    body = {
        "accession_number": accession,
        "decision_session": sessions[position],
        "acceptance_datetime": acceptance,
        "prediction_available": available,
        "learner_ready": ready,
        "raw_gate_pass": passes,
        "probability_hex": 0.60.hex() if available and ready else None,
        "expected_edge_hex": 0.01.hex() if available and ready else None,
    }
    return {**body, "prediction_row_sha256": canonical_sha256(body)}


def test_same_session_events_use_exact_acceptance_then_accession_order() -> None:
    sessions = _sessions()
    later = _prediction(
        sessions,
        position=5,
        accession="b",
        acceptance="20000110170000",
    )
    earlier = _prediction(
        sessions,
        position=5,
        accession="a",
        acceptance="20000110160000",
    )
    replay = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=[later, earlier],
        policy_id="semantic",
    )
    actions = replay["policy_actions"]

    assert [row["accession_number"] for row in actions] == ["a", "b"]
    assert actions[0]["schedule_overlay"] is True
    assert actions[1]["schedule_overlay"] is False
    assert actions[1]["nonoverlap_blocked"] is True


def test_missing_acceptance_is_allowed_for_one_event_but_ambiguous_for_two() -> None:
    sessions = _sessions()
    one = _prediction(
        sessions,
        position=5,
        accession="a",
        acceptance=None,
    )
    replay = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=[one],
        policy_id="semantic",
    )
    assert replay["policy_actions"][0]["schedule_overlay"] is True

    two = _prediction(
        sessions,
        position=5,
        accession="b",
        acceptance="20000110170000",
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayPolicyError,
        match="ambiguous",
    ):
        replay_nonoverlapping_overlay_policy(
            market_sessions=sessions,
            prediction_rows=[one, two],
            policy_id="semantic",
        )


def test_active_overlay_blocks_extension_but_allows_new_decision_after_exit() -> None:
    sessions = _sessions()
    predictions = [
        _prediction(
            sessions,
            position=5,
            accession="a",
            acceptance="20000110160000",
        ),
        _prediction(
            sessions,
            position=25,
            accession="b",
            acceptance="20000207160000",
        ),
        _prediction(
            sessions,
            position=26,
            accession="c",
            acceptance="20000208160000",
        ),
    ]
    replay = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=predictions,
        policy_id="semantic",
    )
    actions = replay["policy_actions"]

    assert actions[0]["schedule_overlay"] is True
    assert actions[1]["schedule_overlay"] is False
    assert actions[1]["effective_action_reason"] == (
        "active_or_pending_overlay_nonoverlap"
    )
    assert actions[2]["schedule_overlay"] is True


@pytest.mark.parametrize(
    "available, ready, passes, reason",
    [
        (False, False, False, "prediction_unavailable"),
        (True, False, False, "learner_unready"),
        (True, True, False, "gate_failed"),
    ],
)
def test_unavailable_unready_and_gate_failure_never_schedule(
    available: bool,
    ready: bool,
    passes: bool,
    reason: str,
) -> None:
    sessions = _sessions()
    prediction = _prediction(
        sessions,
        position=5,
        accession="a",
        acceptance="20000110160000",
        available=available,
        ready=ready,
        passes=passes,
    )
    replay = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=[prediction],
        policy_id="semantic",
    )

    assert replay["policy_actions"][0]["schedule_overlay"] is False
    assert replay["policy_actions"][0]["effective_action_reason"] == reason


def test_last_close_passing_decision_is_sealed_as_pending_entry() -> None:
    sessions = _sessions()
    prediction = _prediction(
        sessions,
        position=len(sessions) - 1,
        accession="a",
        acceptance="20000421160000",
    )
    replay = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=[prediction],
        policy_id="semantic",
    )
    action = replay["policy_actions"][0]
    observation = replay["policy_prefix_observations"][0]

    assert action["schedule_overlay"] is True
    assert action["overlay_schedule_sha256"] == replay[
        "scheduled_overlays"
    ][0]["overlay_schedule_sha256"]
    assert observation["realization_status"] == "pending_entry"
    assert observation["entry_session"] is None
    assert observation["exit_session"] is None


def test_action_and_schedule_identity_survive_market_prefix_extension() -> None:
    sessions = _sessions(40)
    prediction = _prediction(
        sessions,
        position=5,
        accession="a",
        acceptance="20000110160000",
    )
    later = _prediction(
        sessions,
        position=30,
        accession="b",
        acceptance="20000214160000",
    )
    short = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions[:10],
        prediction_rows=[prediction],
        policy_id="semantic",
    )
    extended = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=[prediction, later],
        policy_id="semantic",
    )

    assert short["policy_actions"] == extended["policy_actions"][:1]
    assert short["scheduled_overlays"] == extended["scheduled_overlays"][:1]
    assert (
        short["policy_actions"][0]["policy_action_sha256"]
        == extended["policy_actions"][0]["policy_action_sha256"]
    )
    assert (
        short["scheduled_overlays"][0]["overlay_schedule_sha256"]
        == extended["scheduled_overlays"][0]["overlay_schedule_sha256"]
    )
    assert (
        short["policy_prefix_observations"][0]["realization_status"]
        == "active_pending_exit"
    )
    assert (
        extended["policy_prefix_observations"][0][
            "realization_status"
        ]
        == "complete"
    )
    assert (
        short["policy_prefix_observations"][0][
            "policy_prefix_observation_sha256"
        ]
        != extended["policy_prefix_observations"][0][
            "policy_prefix_observation_sha256"
        ]
    )


def test_policy_replay_rejects_prediction_tampering_and_replays_exactly() -> None:
    sessions = _sessions()
    prediction = _prediction(
        sessions,
        position=5,
        accession="a",
        acceptance="20000110160000",
    )
    replay = replay_nonoverlapping_overlay_policy(
        market_sessions=sessions,
        prediction_rows=[prediction],
        policy_id="semantic",
    )
    assert validate_overlay_policy_replay(
        replay,
        expected_policy_replay_sha256=replay["policy_replay_sha256"],
        market_sessions=sessions,
        prediction_rows=[prediction],
        policy_id="semantic",
    ) == replay["policy_replay_sha256"]

    changed = copy.deepcopy(prediction)
    changed["raw_gate_pass"] = False
    with pytest.raises(
        SecGemmaOnlineRiskOverlayPolicyError,
        match="self-hash",
    ):
        replay_nonoverlapping_overlay_policy(
            market_sessions=sessions,
            prediction_rows=[changed],
            policy_id="semantic",
        )
