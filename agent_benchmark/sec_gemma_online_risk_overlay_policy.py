"""Pure non-overlapping overlay policy for sealed prediction rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
from datetime import date
import hmac
from typing import Any, Final

from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    canonical_sha256,
)


POLICY_REPLAY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-policy-replay-v2"
)
POLICY_ACTION_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-policy-action-row-v2"
)
POLICY_OVERLAY_SCHEDULE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-schedule-row-v1"
)
POLICY_PREFIX_OBSERVATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-1-prefix-observation-row-v1"
)


class SecGemmaOnlineRiskOverlayPolicyError(ValueError):
    """Raised when predictions cannot produce one exact action stream."""


def _iso_date(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise SecGemmaOnlineRiskOverlayPolicyError(
            f"{location} must be an ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecGemmaOnlineRiskOverlayPolicyError(
            f"{location} must be an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecGemmaOnlineRiskOverlayPolicyError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _sha256(value: Any, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SecGemmaOnlineRiskOverlayPolicyError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _canonical_sessions(sessions: Sequence[str]) -> list[str]:
    if isinstance(sessions, (str, bytes)) or not isinstance(
        sessions, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayPolicyError(
            "market sessions must be a sequence"
        )
    result: list[str] = []
    previous: str | None = None
    for ordinal, value in enumerate(sessions, start=1):
        session = _iso_date(value, f"market session {ordinal}")
        if previous is not None and session <= previous:
            raise SecGemmaOnlineRiskOverlayPolicyError(
                "market sessions must be strictly increasing"
            )
        result.append(session)
        previous = session
    if not result:
        raise SecGemmaOnlineRiskOverlayPolicyError(
            "market sessions must not be empty"
        )
    return result


def _validated_prediction_rows(
    predictions: Sequence[Mapping[str, Any]],
    *,
    permitted_sessions: set[str],
) -> list[dict[str, Any]]:
    if isinstance(predictions, (str, bytes)) or not isinstance(
        predictions, Sequence
    ):
        raise SecGemmaOnlineRiskOverlayPolicyError(
            "prediction rows must be a sequence"
        )
    result: list[dict[str, Any]] = []
    seen_accessions: set[str] = set()
    for ordinal, raw in enumerate(predictions, start=1):
        if not isinstance(raw, Mapping):
            raise SecGemmaOnlineRiskOverlayPolicyError(
                f"prediction row {ordinal} must be a mapping"
            )
        row = copy.deepcopy(dict(raw))
        required = {
            "accession_number",
            "decision_session",
            "acceptance_datetime",
            "prediction_available",
            "learner_ready",
            "raw_gate_pass",
            "prediction_row_sha256",
        }
        if not required.issubset(row):
            raise SecGemmaOnlineRiskOverlayPolicyError(
                "prediction row is missing a required field"
            )
        accession = row["accession_number"]
        if (
            not isinstance(accession, str)
            or not accession
            or accession in seen_accessions
        ):
            raise SecGemmaOnlineRiskOverlayPolicyError(
                "prediction accession is invalid or duplicated"
            )
        seen_accessions.add(accession)
        session = _iso_date(
            row["decision_session"],
            f"prediction row {ordinal}.decision_session",
        )
        if session not in permitted_sessions:
            raise SecGemmaOnlineRiskOverlayPolicyError(
                "prediction decision lies outside the market sessions"
            )
        acceptance = row["acceptance_datetime"]
        if acceptance is not None and (
            not isinstance(acceptance, str)
            or len(acceptance) != 14
            or not acceptance.isascii()
            or not acceptance.isdigit()
        ):
            raise SecGemmaOnlineRiskOverlayPolicyError(
                "prediction acceptance timestamp is invalid"
            )
        for name in (
            "prediction_available",
            "learner_ready",
            "raw_gate_pass",
        ):
            if type(row[name]) is not bool:
                raise SecGemmaOnlineRiskOverlayPolicyError(
                    f"prediction {name} must be Boolean"
                )
        if row["raw_gate_pass"] and not (
            row["prediction_available"] and row["learner_ready"]
        ):
            raise SecGemmaOnlineRiskOverlayPolicyError(
                "unavailable or unready prediction cannot pass the raw gate"
            )
        observed = _sha256(
            row["prediction_row_sha256"],
            f"prediction row {ordinal}.prediction_row_sha256",
        )
        body = {
            key: row[key]
            for key in row
            if key != "prediction_row_sha256"
        }
        if not hmac.compare_digest(observed, canonical_sha256(body)):
            raise SecGemmaOnlineRiskOverlayPolicyError(
                "prediction row self-hash changed"
            )
        result.append(row)
    return result


def _ordered_predictions(
    predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_session[row["decision_session"]].append(row)
    ordered: list[dict[str, Any]] = []
    for session in sorted(by_session):
        rows = by_session[session]
        if len(rows) > 1 and any(
            row["acceptance_datetime"] is None for row in rows
        ):
            raise SecGemmaOnlineRiskOverlayPolicyError(
                "same-session filing order is ambiguous without exact "
                "acceptance timestamps"
            )
        rows.sort(
            key=lambda row: (
                "" if row["acceptance_datetime"] is None else row[
                    "acceptance_datetime"
                ],
                row["accession_number"],
            )
        )
        ordered.extend(rows)
    return ordered


def _build_overlay_schedule(
    *,
    policy_id: str,
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one schedule without any market-prefix realization fields."""

    body = {
        "schema_version": POLICY_OVERLAY_SCHEDULE_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "policy_id": policy_id,
        "accession_number": prediction["accession_number"],
        "decision_session": prediction["decision_session"],
        "prediction_row_sha256": prediction["prediction_row_sha256"],
        "entry_session_offset": 1,
        "exit_session_offset": LABEL_MATURITY_OFFSET,
        "horizon_sessions": HORIZON_SESSIONS,
    }
    return {**body, "overlay_schedule_sha256": canonical_sha256(body)}


def _build_prefix_observation(
    *,
    sessions: Sequence[str],
    decision_position: int,
    policy_id: str,
    action: Mapping[str, Any],
) -> dict[str, Any]:
    """Observe schedule realization in one prefix without changing the action."""

    schedule_hash = action["overlay_schedule_sha256"]
    scheduled = action["schedule_overlay"]
    if not scheduled:
        entry_position = None
        exit_position = None
        entry_session = None
        exit_session = None
        status = "not_scheduled"
    else:
        entry_position = decision_position + 1
        exit_position = decision_position + LABEL_MATURITY_OFFSET
        entry_session = (
            sessions[entry_position]
            if entry_position < len(sessions)
            else None
        )
        exit_session = (
            sessions[exit_position]
            if exit_position < len(sessions)
            else None
        )
        status = (
            "pending_entry"
            if entry_session is None
            else "complete"
            if exit_session is not None
            else "active_pending_exit"
        )
    body = {
        "schema_version": POLICY_PREFIX_OBSERVATION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "policy_id": policy_id,
        "policy_action_sha256": action["policy_action_sha256"],
        "overlay_schedule_sha256": schedule_hash,
        "accession_number": action["accession_number"],
        "decision_session": action["decision_session"],
        "market_prefix_last_session": sessions[-1],
        "market_sessions_sha256": canonical_sha256(sessions),
        "entry_position": entry_position,
        "exit_position": exit_position,
        "entry_session": entry_session,
        "exit_session": exit_session,
        "realization_status": status,
    }
    return {
        **body,
        "policy_prefix_observation_sha256": canonical_sha256(body),
    }


def replay_nonoverlapping_overlay_policy(
    *,
    market_sessions: Sequence[str],
    prediction_rows: Sequence[Mapping[str, Any]],
    policy_id: str,
) -> dict[str, Any]:
    """Apply the fixed gate result and exact non-overlap rule."""

    sessions = _canonical_sessions(market_sessions)
    if not isinstance(policy_id, str) or not policy_id:
        raise SecGemmaOnlineRiskOverlayPolicyError("policy_id is invalid")
    positions = {
        session: position for position, session in enumerate(sessions)
    }
    predictions = _validated_prediction_rows(
        prediction_rows,
        permitted_sessions=set(sessions),
    )
    ordered = _ordered_predictions(predictions)
    blocked_until_decision_position = -1
    actions: list[dict[str, Any]] = []
    schedules: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for action_ordinal, prediction in enumerate(ordered, start=1):
        decision_position = positions[prediction["decision_session"]]
        nonoverlap_blocked = (
            decision_position < blocked_until_decision_position
        )
        schedule = prediction["raw_gate_pass"] and not nonoverlap_blocked
        if schedule:
            blocked_until_decision_position = (
                decision_position + LABEL_MATURITY_OFFSET
            )
        reason = (
            "scheduled"
            if schedule
            else "prediction_unavailable"
            if not prediction["prediction_available"]
            else "learner_unready"
            if not prediction["learner_ready"]
            else "gate_failed"
            if not prediction["raw_gate_pass"]
            else "active_or_pending_overlay_nonoverlap"
        )
        overlay_schedule = (
            _build_overlay_schedule(
                policy_id=policy_id,
                prediction=prediction,
            )
            if schedule
            else None
        )
        body = {
            "schema_version": POLICY_ACTION_ROW_SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "policy_id": policy_id,
            "action_ordinal": action_ordinal,
            "prediction_row_sha256": prediction["prediction_row_sha256"],
            "accession_number": prediction["accession_number"],
            "decision_session": prediction["decision_session"],
            "acceptance_datetime": prediction["acceptance_datetime"],
            "prediction_available": prediction["prediction_available"],
            "learner_ready": prediction["learner_ready"],
            "raw_gate_pass": prediction["raw_gate_pass"],
            "nonoverlap_blocked": nonoverlap_blocked,
            "schedule_overlay": schedule,
            "effective_action_reason": reason,
            "overlay_schedule_sha256": (
                None
                if overlay_schedule is None
                else overlay_schedule["overlay_schedule_sha256"]
            ),
        }
        action = {**body, "policy_action_sha256": canonical_sha256(body)}
        actions.append(action)
        if overlay_schedule is not None:
            schedules.append(overlay_schedule)
        observations.append(
            _build_prefix_observation(
                sessions=sessions,
                decision_position=decision_position,
                policy_id=policy_id,
                action=action,
            )
        )
    body = {
        "schema_version": POLICY_REPLAY_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "policy_id": policy_id,
        "market_sessions_sha256": canonical_sha256(sessions),
        "source_prediction_rows_sha256": canonical_sha256(predictions),
        "ordered_prediction_row_sha256s": [
            row["prediction_row_sha256"] for row in ordered
        ],
        "ordered_prediction_rows_sha256": canonical_sha256(
            [row["prediction_row_sha256"] for row in ordered]
        ),
        "policy_actions": actions,
        "policy_actions_sha256": canonical_sha256(actions),
        "scheduled_overlays": schedules,
        "scheduled_overlays_sha256": canonical_sha256(schedules),
        "policy_prefix_observations": observations,
        "policy_prefix_observations_sha256": canonical_sha256(
            observations
        ),
    }
    return {**body, "policy_replay_sha256": canonical_sha256(body)}


def validate_overlay_policy_replay(
    replay: Mapping[str, Any],
    *,
    expected_policy_replay_sha256: str,
    market_sessions: Sequence[str],
    prediction_rows: Sequence[Mapping[str, Any]],
    policy_id: str,
) -> str:
    """Rebuild one policy replay and require byte-level identity."""

    if not isinstance(replay, Mapping):
        raise SecGemmaOnlineRiskOverlayPolicyError(
            "policy replay must be a mapping"
        )
    rebuilt = replay_nonoverlapping_overlay_policy(
        market_sessions=market_sessions,
        prediction_rows=prediction_rows,
        policy_id=policy_id,
    )
    if dict(replay) != rebuilt:
        raise SecGemmaOnlineRiskOverlayPolicyError(
            "policy replay differs from deterministic reconstruction"
        )
    observed = _sha256(
        replay.get("policy_replay_sha256"), "policy replay hash"
    )
    expected = _sha256(
        expected_policy_replay_sha256, "expected policy replay hash"
    )
    if not hmac.compare_digest(observed, expected):
        raise SecGemmaOnlineRiskOverlayPolicyError(
            "policy replay is not externally pinned"
        )
    return observed


__all__ = [
    "POLICY_ACTION_ROW_SCHEMA_VERSION",
    "POLICY_OVERLAY_SCHEDULE_SCHEMA_VERSION",
    "POLICY_PREFIX_OBSERVATION_SCHEMA_VERSION",
    "POLICY_REPLAY_SCHEMA_VERSION",
    "SecGemmaOnlineRiskOverlayPolicyError",
    "replay_nonoverlapping_overlay_policy",
    "validate_overlay_policy_replay",
]
