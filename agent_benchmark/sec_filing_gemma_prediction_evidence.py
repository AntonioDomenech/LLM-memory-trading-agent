"""Pure append-only evidence for SEC/Gemma predictions and causal labels.

This module performs no filesystem, network, market-data, SEC, or model I/O.
Each prediction is appended using only a previously pinned prefix plus the
current event's inputs.  The resulting prefix is externally sealed before any
outcome is admitted.  Labels are released only for sealed rows whose maturity
session strictly precedes the current decision session.

The seal protocol intentionally makes no claim that a backtest artifact was
created at a historical wall-clock time.  It proves ordering inside the current
experiment: prediction prefix first, external immutable checksum second, label
release last.  Durable storage and timestamp attestation belong to the
effectful sealer/store.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import copy
import hmac
import math
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    ACTIVE_EDGE_TOLERANCE,
    BRIER_TARGET_COST_BPS,
    CANDIDATE_IDS,
    DEVELOPMENT_FOLD_SPECS,
    HORIZON_SESSIONS,
    LABEL_MATURITY_OFFSET,
    STAGE_ORDER,
    STAGE_WINDOWS,
    SecFilingGemmaContractError,
    build_contract_manifest,
    canonical_session_calendar,
    canonical_sha256,
    session_calendar_sha256,
)


PREDICTION_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-pre-label-prediction-row-v1"
)
PREDICTION_PREFIX_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-pre-label-prediction-prefix-v1"
)
PRELABEL_SEAL_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-pre-label-seal-entry-v1"
)
PRELABEL_SEAL_LEDGER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-pre-label-seal-ledger-v1"
)
LABEL_RELEASE_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-label-release-entry-v1"
)
LABEL_RELEASE_LEDGER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-label-release-ledger-v1"
)

AVAILABLE_PREDICTION_STATUS: Final[str] = "available_pre_label"
UNAVAILABLE_PREDICTION_STATUS: Final[str] = "unavailable_pre_label"
PREDICTION_STATUSES: Final[tuple[str, str]] = (
    AVAILABLE_PREDICTION_STATUS,
    UNAVAILABLE_PREDICTION_STATUS,
)
UNAVAILABLE_REASONS: Final[tuple[str, ...]] = (
    "missing_required_market_features",
    "missing_required_extraction_features",
    "missing_required_market_and_extraction_features",
)
MODEL_VARIANTS: Final[tuple[str, str]] = ("semantic", "ablation")
INTERMEDIATE_FOLD_ID: Final[str] = "intermediate_frozen_through_2018"
FINAL_FOLD_ID: Final[str] = "final_frozen_through_2023"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ACCESSION_RE = re.compile(r"0000320193-[0-9]{2}-[0-9]{6}\Z")

_FOLD_CONTEXT_KEYS = {
    "fold_train_cutoff_session",
    "training_set_count",
    "training_set_membership_sha256",
    "semantic_training_feature_matrix_sha256",
    "ablation_training_feature_matrix_sha256",
    "training_binary_target_sha256",
    "training_edge_target_sha256",
    "training_set_max_label_maturity_session",
    "semantic_fold_state_sha256",
    "ablation_fold_state_sha256",
}
_EVENT_BINDING_KEYS = {
    "accession_number",
    "form",
    "stage",
    "decision_session",
    "extraction_identity_sha256",
    "market_prefix_chain_identity_sha256",
    "market_feature_row_sha256",
    "fold_id",
}
_PREDICTION_SPEC_KEYS = _EVENT_BINDING_KEYS | {
    "prediction_status",
    "unavailable_reason",
    "semantic_cash_probability",
    "semantic_expected_edge",
    "ablation_cash_probability",
    "ablation_expected_edge",
}
_POLICY_STATE_KEYS = {
    "position_at_decision_close",
    "episode_origin_decision_session",
    "episode_fill_session",
    "episode_exit_session",
}
_PREDICTION_ROW_KEYS = {
    "schema_version",
    "sequence_number",
    "prior_prediction_prefix_sha256",
    "parent_prediction_sha256",
    "contract_sha256",
    "candidate_sha256",
    "corpus_universe_sha256",
    "calendar_sessions_sha256",
    "event_sequence_sha256",
    "fold_context",
    "fold_context_sha256",
    "accession_number",
    "form",
    "stage",
    "decision_session",
    "extraction_identity_sha256",
    "market_prefix_chain_identity_sha256",
    "market_feature_row_sha256",
    "market_feature_cutoff_session",
    "fill_session",
    "cash_exit_session",
    "label_maturity_session",
    "horizon_sessions",
    "fold_id",
    "semantic_cash_probability_hex",
    "semantic_expected_edge_hex",
    "ablation_cash_probability_hex",
    "ablation_expected_edge_hex",
    "raw_gate_signals",
    "raw_gate_signals_sha256",
    "effective_episode_actions",
    "effective_episode_actions_sha256",
    "candidate_policy_input_states",
    "candidate_policy_input_states_sha256",
    "candidate_policy_output_states",
    "candidate_policy_output_states_sha256",
    "unavailable_fail_safe_action_semantics",
    "prediction_status",
    "unavailable_reason",
    "prediction_row_sha256",
}
_PREFIX_KEYS = {
    "schema_version",
    "contract_sha256",
    "candidate_sha256",
    "corpus_universe_sha256",
    "calendar_sessions_sha256",
    "initial_event_sequence_sha256",
    "event_sequence_sha256",
    "row_count",
    "genesis_sha256",
    "parent_prefix_sha256",
    "parent_tip_sha256",
    "appended_row_sha256",
    "tip_sha256",
    "rows_sha256",
    "rows",
    "prediction_prefix_sha256",
}
_SEAL_ENTRY_KEYS = {
    "schema_version",
    "sequence_number",
    "parent_seal_sha256",
    "prediction_sequence_number",
    "prediction_row_sha256",
    "prediction_prefix_sha256",
    "accession_number",
    "decision_session",
    "label_maturity_session",
    "seal_protocol_phase",
    "prediction_artifact_checksum_sha256",
    "seal_sha256",
}
_SEAL_LEDGER_KEYS = {
    "schema_version",
    "contract_sha256",
    "candidate_sha256",
    "corpus_universe_sha256",
    "calendar_sessions_sha256",
    "event_sequence_sha256",
    "prediction_prefix_sha256",
    "sealed_prediction_count",
    "sealed_prediction_tip_sha256",
    "genesis_sha256",
    "tip_sha256",
    "entries",
    "prelabel_seal_ledger_sha256",
}
_LABEL_SPEC_KEYS = {
    "cash_active_log_edge_10bps",
    "strategy_ledger_slice_sha256",
    "benchmark_ledger_slice_sha256",
    "outcome_ledger_row_sha256",
}
_LABEL_ENTRY_KEYS = {
    "schema_version",
    "sequence_number",
    "parent_release_sha256",
    "prediction_sequence_number",
    "prediction_row_sha256",
    "prediction_prefix_sha256",
    "prediction_artifact_checksum_sha256",
    "prelabel_seal_sha256",
    "accession_number",
    "decision_session",
    "prediction_status",
    "unavailable_reason",
    "label_maturity_session",
    "release_session",
    "first_eligible_prediction_session",
    "as_of_decision_session",
    "horizon_sessions",
    "cost_bps",
    "cash_active_log_edge_10bps_hex",
    "cash_beats_long_10bps",
    "strategy_ledger_slice_sha256",
    "benchmark_ledger_slice_sha256",
    "outcome_ledger_row_sha256",
    "label_binding_sha256",
    "release_sha256",
}
_LABEL_LEDGER_KEYS = {
    "schema_version",
    "contract_sha256",
    "candidate_sha256",
    "corpus_universe_sha256",
    "calendar_sessions_sha256",
    "event_sequence_sha256",
    "prediction_prefix_sha256",
    "prelabel_seal_ledger_sha256",
    "as_of_decision_session",
    "released_prediction_sequence_sha256",
    "release_count",
    "genesis_sha256",
    "tip_sha256",
    "entries",
    "label_release_ledger_sha256",
}


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise SecFilingGemmaContractError(f"{location} must be a string-keyed mapping")
    return value


def _expect_keys(value: Mapping[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SecFilingGemmaContractError(
            f"Invalid {location} keys; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SecFilingGemmaContractError(
            f"{location} must be an integer >= {minimum}"
        )
    return value


def _strict_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise SecFilingGemmaContractError(f"{location} must be a boolean")
    return value


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaContractError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _tagged_sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _TAGGED_SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaContractError(
            f"{location} must be a lowercase sha256:<digest> checksum"
        )
    return value


def _iso_date(value: Any, location: str) -> date:
    if not isinstance(value, str):
        raise SecFilingGemmaContractError(f"{location} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecFilingGemmaContractError(f"{location} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise SecFilingGemmaContractError(
            f"{location} must use canonical YYYY-MM-DD form"
        )
    return parsed


def _encode_float_hex(value: Any, location: str, *, probability: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SecFilingGemmaContractError(f"{location} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SecFilingGemmaContractError(f"{location} must be finite")
    if probability and not 0.0 <= number <= 1.0:
        raise SecFilingGemmaContractError(f"{location} must be between zero and one")
    if number == 0.0:
        number = 0.0
    return number.hex()


def _decode_float_hex(value: Any, location: str, *, probability: bool = False) -> float:
    if not isinstance(value, str):
        raise SecFilingGemmaContractError(
            f"{location} must be a canonical hexadecimal float"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecFilingGemmaContractError(
            f"{location} must be a canonical hexadecimal float"
        ) from exc
    if value != _encode_float_hex(number, location, probability=probability):
        raise SecFilingGemmaContractError(
            f"{location} must be a canonical hexadecimal float"
        )
    return number


def _canonical_sessions(
    session_dates: Sequence[str], expected_calendar_sessions_sha256: str
) -> tuple[tuple[str, ...], str]:
    sessions = canonical_session_calendar(session_dates)
    observed = session_calendar_sha256(sessions)
    expected = _sha256(
        expected_calendar_sessions_sha256, "expected_calendar_sessions_sha256"
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaContractError(
            "Prediction evidence calendar is not externally pinned"
        )
    return sessions, observed


def _session_offset(value: str, offset: int, sessions: Sequence[str]) -> str:
    try:
        index = sessions.index(value)
    except ValueError as exc:
        raise SecFilingGemmaContractError(
            "Prediction timeline date is not an exact frozen NYSE session"
        ) from exc
    target = index + offset
    if target >= len(sessions):
        raise SecFilingGemmaContractError(
            "Frozen calendar does not cover the prediction timeline"
        )
    return sessions[target]


def _stage_for_session(value: str) -> str:
    parsed = _iso_date(value, "decision_session")
    for stage in STAGE_ORDER:
        first = _iso_date(STAGE_WINDOWS[stage][0], f"{stage} first")
        last = _iso_date(STAGE_WINDOWS[stage][1], f"{stage} last")
        if first <= parsed <= last:
            return stage
    raise SecFilingGemmaContractError(
        "Prediction decision is outside the frozen stage windows"
    )


def _expected_fold(stage: str, decision_session: str) -> tuple[str, str, str]:
    decision = _iso_date(decision_session, "decision_session")
    if stage == "development":
        for fold_id, cutoff, test_first, test_last in DEVELOPMENT_FOLD_SPECS:
            if _iso_date(test_first, "fold first") <= decision <= _iso_date(
                test_last, "fold last"
            ):
                return fold_id, cutoff, test_first
        raise SecFilingGemmaContractError(
            "Development prediction is outside every frozen out-of-fold window"
        )
    if stage == "intermediate":
        return INTERMEDIATE_FOLD_ID, "2018-12-31", "2019-01-01"
    if stage == "final":
        return FINAL_FOLD_ID, "2023-12-31", "2024-01-01"
    raise SecFilingGemmaContractError("Prediction stage is invalid")


def _candidate_thresholds() -> dict[str, tuple[float, float]]:
    grid = build_contract_manifest()["predictor"]["candidate_grid"]
    result = {
        item["candidate_id"]: (
            float(item["probability_gate"]),
            float(item["expected_edge_gate"]),
        )
        for item in grid
    }
    if tuple(result) != CANDIDATE_IDS:
        raise SecFilingGemmaContractError("Frozen candidate grid order changed")
    return result


def _normalize_fold_context(
    value: Mapping[str, Any], *, fold_id: str, stage: str, decision_session: str
) -> dict[str, Any]:
    context = _expect_mapping(value, "current_fold_context")
    _expect_keys(context, _FOLD_CONTEXT_KEYS, "current_fold_context")
    expected_fold_id, expected_cutoff, test_first = _expected_fold(
        stage, decision_session
    )
    if fold_id != expected_fold_id:
        raise SecFilingGemmaContractError("Current event uses the wrong frozen fold")
    cutoff = _iso_date(
        context["fold_train_cutoff_session"], "fold_train_cutoff_session"
    )
    if cutoff.isoformat() != expected_cutoff:
        raise SecFilingGemmaContractError("Current fold train cutoff changed")
    maximum = _iso_date(
        context["training_set_max_label_maturity_session"],
        "training_set_max_label_maturity_session",
    )
    if not (maximum <= cutoff and maximum < _iso_date(test_first, "test_first")):
        raise SecFilingGemmaContractError(
            "Fold training includes a label not mature before its test window"
        )
    normalized: dict[str, Any] = {
        "fold_train_cutoff_session": cutoff.isoformat(),
        "training_set_count": _strict_int(
            context["training_set_count"], "training_set_count", minimum=1
        ),
        "training_set_max_label_maturity_session": maximum.isoformat(),
    }
    for key in sorted(
        _FOLD_CONTEXT_KEYS
        - {
            "fold_train_cutoff_session",
            "training_set_count",
            "training_set_max_label_maturity_session",
        }
    ):
        normalized[key] = _sha256(context[key], key)
    if (
        normalized["semantic_fold_state_sha256"]
        == normalized["ablation_fold_state_sha256"]
        or normalized["semantic_training_feature_matrix_sha256"]
        == normalized["ablation_training_feature_matrix_sha256"]
    ):
        raise SecFilingGemmaContractError(
            "Semantic and ablation fold artifacts must remain distinct"
        )
    return normalized


def _normalize_event_binding(
    value: Mapping[str, Any], *, sessions: Sequence[str]
) -> dict[str, str]:
    event = _expect_mapping(value, "current_event_binding")
    _expect_keys(event, _EVENT_BINDING_KEYS, "current_event_binding")
    accession = event["accession_number"]
    if not isinstance(accession, str) or _ACCESSION_RE.fullmatch(accession) is None:
        raise SecFilingGemmaContractError("Prediction event is not an Apple accession")
    if event["form"] not in {"10-K", "10-Q"}:
        raise SecFilingGemmaContractError("Prediction event form is ineligible")
    decision = _iso_date(event["decision_session"], "decision_session").isoformat()
    _session_offset(decision, 0, sessions)
    stage = event["stage"]
    if stage != _stage_for_session(decision):
        raise SecFilingGemmaContractError(
            "Prediction stage does not match its decision session"
        )
    fold_id, _, _ = _expected_fold(stage, decision)
    if event["fold_id"] != fold_id:
        raise SecFilingGemmaContractError("Prediction event uses the wrong frozen fold")
    return {
        "accession_number": accession,
        "form": event["form"],
        "stage": stage,
        "decision_session": decision,
        "extraction_identity_sha256": _sha256(
            event["extraction_identity_sha256"], "extraction_identity_sha256"
        ),
        "market_prefix_chain_identity_sha256": _sha256(
            event["market_prefix_chain_identity_sha256"],
            "market_prefix_chain_identity_sha256",
        ),
        "market_feature_row_sha256": _sha256(
            event["market_feature_row_sha256"], "market_feature_row_sha256"
        ),
        "fold_id": fold_id,
    }


def _normalize_authoritative_event_sequence(
    value: Sequence[Mapping[str, Any]], *, sessions: Sequence[str]
) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SecFilingGemmaContractError(
            "expected_event_bindings must be an authoritative sequence"
        )
    if not value:
        raise SecFilingGemmaContractError(
            "Authoritative expected event sequence cannot be empty"
        )
    normalized: list[dict[str, str]] = []
    previous_key: tuple[str, str] | None = None
    accessions: set[str] = set()
    for raw in value:
        event = _normalize_event_binding(raw, sessions=sessions)
        key = (event["decision_session"], event["accession_number"])
        if previous_key is not None and key <= previous_key:
            raise SecFilingGemmaContractError(
                "Authoritative event sequence is duplicated or reordered"
            )
        if event["accession_number"] in accessions:
            raise SecFilingGemmaContractError(
                "Authoritative event sequence duplicates an accession"
            )
        previous_key = key
        accessions.add(event["accession_number"])
        normalized.append(event)
    return normalized


def _prediction_genesis(
    *,
    contract_hash: str,
    candidate_hash: str,
    universe_hash: str,
    calendar_hash: str,
    initial_event_sequence_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "domain": "aapl-sec-gemma-prediction-genesis-v1",
            "contract_sha256": contract_hash,
            "candidate_sha256": candidate_hash,
            "corpus_universe_sha256": universe_hash,
            "calendar_sessions_sha256": calendar_hash,
            "initial_event_sequence_sha256": _sha256(
                initial_event_sequence_sha256,
                "initial_event_sequence_sha256",
            ),
        }
    )


def _inactive_state() -> dict[str, str | None]:
    return {
        "position_at_decision_close": "LONG",
        "episode_origin_decision_session": None,
        "episode_fill_session": None,
        "episode_exit_session": None,
    }


def _initial_policy_states() -> dict[str, dict[str, dict[str, str | None]]]:
    return {
        candidate_id: {variant: _inactive_state() for variant in MODEL_VARIANTS}
        for candidate_id in CANDIDATE_IDS
    }


def _active_state(
    decision_session: str, sessions: Sequence[str]
) -> dict[str, str | None]:
    return {
        "position_at_decision_close": "CASH",
        "episode_origin_decision_session": decision_session,
        "episode_fill_session": _session_offset(decision_session, 1, sessions),
        "episode_exit_session": _session_offset(
            decision_session, LABEL_MATURITY_OFFSET, sessions
        ),
    }


def _normalize_policy_state(
    value: Mapping[str, Any], *, sessions: Sequence[str], location: str
) -> dict[str, str | None]:
    state = _expect_mapping(value, location)
    _expect_keys(state, _POLICY_STATE_KEYS, location)
    position = state["position_at_decision_close"]
    origin = state["episode_origin_decision_session"]
    fill = state["episode_fill_session"]
    exit_session = state["episode_exit_session"]
    if position == "LONG":
        if any(item is not None for item in (origin, fill, exit_session)):
            raise SecFilingGemmaContractError(
                f"{location} LONG state cannot retain a cash episode"
            )
        return _inactive_state()
    if position != "CASH" or any(
        not isinstance(item, str) for item in (origin, fill, exit_session)
    ):
        raise SecFilingGemmaContractError(f"{location} is not a canonical policy state")
    origin_date = _iso_date(origin, f"{location}.origin").isoformat()
    fill_date = _iso_date(fill, f"{location}.fill").isoformat()
    exit_date = _iso_date(exit_session, f"{location}.exit").isoformat()
    _session_offset(origin_date, 0, sessions)
    if fill_date != _session_offset(origin_date, 1, sessions):
        raise SecFilingGemmaContractError(f"{location} cash fill is not t+1")
    if exit_date != _session_offset(origin_date, LABEL_MATURITY_OFFSET, sessions):
        raise SecFilingGemmaContractError(f"{location} cash exit is not t+21")
    return {
        "position_at_decision_close": "CASH",
        "episode_origin_decision_session": origin_date,
        "episode_fill_session": fill_date,
        "episode_exit_session": exit_date,
    }


def _normalize_policy_states(
    value: Mapping[str, Any], *, sessions: Sequence[str], location: str
) -> dict[str, dict[str, dict[str, str | None]]]:
    states = _expect_mapping(value, location)
    _expect_keys(states, set(CANDIDATE_IDS), location)
    normalized: dict[str, dict[str, dict[str, str | None]]] = {}
    for candidate_id in CANDIDATE_IDS:
        variants = _expect_mapping(states[candidate_id], f"{location}.{candidate_id}")
        _expect_keys(variants, set(MODEL_VARIANTS), f"{location}.{candidate_id}")
        normalized[candidate_id] = {
            variant: _normalize_policy_state(
                variants[variant],
                sessions=sessions,
                location=f"{location}.{candidate_id}.{variant}",
            )
            for variant in MODEL_VARIANTS
        }
    return normalized


def _roll_policy_states(
    previous: Mapping[str, Any], *, decision_session: str, sessions: Sequence[str]
) -> dict[str, dict[str, dict[str, str | None]]]:
    normalized = _normalize_policy_states(
        previous, sessions=sessions, location="previous_policy_states"
    )
    rolled: dict[str, dict[str, dict[str, str | None]]] = {}
    for candidate_id in CANDIDATE_IDS:
        rolled[candidate_id] = {}
        for variant in MODEL_VARIANTS:
            state = normalized[candidate_id][variant]
            if (
                state["position_at_decision_close"] == "CASH"
                and decision_session < state["episode_exit_session"]
            ):
                rolled[candidate_id][variant] = copy.deepcopy(state)
            else:
                rolled[candidate_id][variant] = _inactive_state()
    return rolled


def _raw_gate_signals(
    *,
    semantic_probability: float,
    semantic_edge: float,
    ablation_probability: float,
    ablation_edge: float,
) -> dict[str, dict[str, str]]:
    values = {
        "semantic": (semantic_probability, semantic_edge),
        "ablation": (ablation_probability, ablation_edge),
    }
    signals: dict[str, dict[str, str]] = {}
    for candidate_id, (probability_gate, edge_gate) in _candidate_thresholds().items():
        signals[candidate_id] = {}
        for variant in MODEL_VARIANTS:
            probability, edge = values[variant]
            signals[candidate_id][variant] = (
                "CASH"
                if probability >= probability_gate and edge >= edge_gate
                else "LONG"
            )
    return signals


def _normalize_signal_or_action_map(
    value: Mapping[str, Any], *, location: str, allowed: set[str]
) -> dict[str, dict[str, str]]:
    mapping = _expect_mapping(value, location)
    _expect_keys(mapping, set(CANDIDATE_IDS), location)
    normalized: dict[str, dict[str, str]] = {}
    for candidate_id in CANDIDATE_IDS:
        variants = _expect_mapping(mapping[candidate_id], f"{location}.{candidate_id}")
        _expect_keys(variants, set(MODEL_VARIANTS), f"{location}.{candidate_id}")
        normalized[candidate_id] = {}
        for variant in MODEL_VARIANTS:
            item = variants[variant]
            if item not in allowed:
                raise SecFilingGemmaContractError(
                    f"{location}.{candidate_id}.{variant} is invalid"
                )
            normalized[candidate_id][variant] = item
    return normalized


def _available_policy_transition(
    input_states: Mapping[str, Any],
    raw_signals: Mapping[str, Any],
    *,
    decision_session: str,
    sessions: Sequence[str],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, dict[str, str | None]]],
]:
    inputs = _normalize_policy_states(
        input_states, sessions=sessions, location="candidate_policy_input_states"
    )
    signals = _normalize_signal_or_action_map(
        raw_signals, location="raw_gate_signals", allowed={"LONG", "CASH"}
    )
    actions: dict[str, dict[str, str]] = {}
    outputs: dict[str, dict[str, dict[str, str | None]]] = {}
    for candidate_id in CANDIDATE_IDS:
        actions[candidate_id] = {}
        outputs[candidate_id] = {}
        for variant in MODEL_VARIANTS:
            state = inputs[candidate_id][variant]
            signal = signals[candidate_id][variant]
            if state["position_at_decision_close"] == "CASH":
                # A new signal is diagnostic only.  The original exit is copied
                # byte-for-byte, so no filing can extend an active episode.
                actions[candidate_id][variant] = "HOLD_EXISTING_CASH_EPISODE"
                outputs[candidate_id][variant] = copy.deepcopy(state)
            elif signal == "CASH":
                actions[candidate_id][variant] = "START_CASH_EPISODE"
                outputs[candidate_id][variant] = _active_state(
                    decision_session, sessions
                )
            else:
                actions[candidate_id][variant] = "STAY_LONG"
                outputs[candidate_id][variant] = _inactive_state()
    return actions, outputs


def _unavailable_policy_transition(
    input_states: Mapping[str, Any], *, sessions: Sequence[str]
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, dict[str, str | None]]],
]:
    # Unavailable rows make no probability or gate claim and cannot start an
    # episode.  Their effective policy action is nevertheless explicit for
    # every path: an inactive path stays LONG, while an already-open episode
    # remains CASH until its original immutable exit.
    inputs = _normalize_policy_states(
        input_states, sessions=sessions, location="candidate_policy_input_states"
    )
    actions: dict[str, dict[str, str]] = {}
    outputs: dict[str, dict[str, dict[str, str | None]]] = {}
    for candidate_id in CANDIDATE_IDS:
        actions[candidate_id] = {}
        outputs[candidate_id] = {}
        for variant in MODEL_VARIANTS:
            state = inputs[candidate_id][variant]
            if state["position_at_decision_close"] == "CASH":
                actions[candidate_id][variant] = "HOLD_EXISTING_CASH_EPISODE"
            else:
                actions[candidate_id][variant] = "STAY_LONG"
            outputs[candidate_id][variant] = copy.deepcopy(state)
    return actions, outputs


def _normalize_spec(
    value: Mapping[str, Any], *, event: Mapping[str, str]
) -> dict[str, Any]:
    spec = _expect_mapping(value, "prediction_spec")
    _expect_keys(spec, _PREDICTION_SPEC_KEYS, "prediction_spec")
    if {key: spec[key] for key in _EVENT_BINDING_KEYS} != dict(event):
        raise SecFilingGemmaContractError(
            "Prediction spec does not match its current external event binding"
        )
    status = spec["prediction_status"]
    if status not in PREDICTION_STATUSES:
        raise SecFilingGemmaContractError("Prediction status is invalid")
    values = (
        spec["semantic_cash_probability"],
        spec["semantic_expected_edge"],
        spec["ablation_cash_probability"],
        spec["ablation_expected_edge"],
    )
    if status == UNAVAILABLE_PREDICTION_STATUS:
        if spec["unavailable_reason"] not in UNAVAILABLE_REASONS:
            raise SecFilingGemmaContractError(
                "Unavailable prediction requires an exact frozen reason"
            )
        if any(item is not None for item in values):
            raise SecFilingGemmaContractError(
                "Unavailable prediction cannot claim probabilities or edges"
            )
        return {
            "prediction_status": status,
            "unavailable_reason": spec["unavailable_reason"],
            "semantic_cash_probability_hex": None,
            "semantic_expected_edge_hex": None,
            "ablation_cash_probability_hex": None,
            "ablation_expected_edge_hex": None,
        }
    if spec["unavailable_reason"] is not None:
        raise SecFilingGemmaContractError(
            "Available prediction cannot carry an unavailable reason"
        )
    return {
        "prediction_status": status,
        "unavailable_reason": None,
        "semantic_cash_probability_hex": _encode_float_hex(
            values[0], "semantic_cash_probability", probability=True
        ),
        "semantic_expected_edge_hex": _encode_float_hex(
            values[1], "semantic_expected_edge"
        ),
        "ablation_cash_probability_hex": _encode_float_hex(
            values[2], "ablation_cash_probability", probability=True
        ),
        "ablation_expected_edge_hex": _encode_float_hex(
            values[3], "ablation_expected_edge"
        ),
    }


def _prefix_body(
    *,
    contract_hash: str,
    candidate_hash: str,
    universe_hash: str,
    calendar_hash: str,
    initial_event_sequence_sha256: str,
    event_sequence_sha256: str,
    genesis_hash: str,
    parent_prefix_hash: str | None,
    parent_tip_hash: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    copied_rows = copy.deepcopy(list(rows))
    return {
        "schema_version": PREDICTION_PREFIX_SCHEMA_VERSION,
        "contract_sha256": contract_hash,
        "candidate_sha256": candidate_hash,
        "corpus_universe_sha256": universe_hash,
        "calendar_sessions_sha256": calendar_hash,
        "initial_event_sequence_sha256": _sha256(
            initial_event_sequence_sha256,
            "initial_event_sequence_sha256",
        ),
        "event_sequence_sha256": _sha256(
            event_sequence_sha256, "event_sequence_sha256"
        ),
        "row_count": len(copied_rows),
        "genesis_sha256": genesis_hash,
        "parent_prefix_sha256": parent_prefix_hash,
        "parent_tip_sha256": parent_tip_hash,
        "appended_row_sha256": copied_rows[-1]["prediction_row_sha256"],
        "tip_sha256": copied_rows[-1]["prediction_row_sha256"],
        "rows_sha256": canonical_sha256(copied_rows),
        "rows": copied_rows,
    }


def append_prediction_row(
    prior_prefix: Mapping[str, Any] | None,
    prediction_spec: Mapping[str, Any],
    *,
    current_event_binding: Mapping[str, Any],
    current_fold_context: Mapping[str, Any],
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
    candidate_sha256: str,
    corpus_universe_sha256: str,
    expected_event_bindings: Sequence[Mapping[str, Any]],
    expected_prior_prefix_sha256: str | None,
    expected_prior_tip_sha256: str | None,
) -> dict[str, Any]:
    """Append one event using only a pinned prior prefix and current inputs."""

    sessions, calendar_hash = _canonical_sessions(
        session_dates, expected_calendar_sessions_sha256
    )
    contract_hash = canonical_sha256(build_contract_manifest())
    candidate_hash = _sha256(candidate_sha256, "candidate_sha256")
    universe_hash = _sha256(corpus_universe_sha256, "corpus_universe_sha256")
    events = _normalize_authoritative_event_sequence(
        expected_event_bindings, sessions=sessions
    )
    event = _normalize_event_binding(current_event_binding, sessions=sessions)
    if events[-1] != event:
        raise SecFilingGemmaContractError(
            "Current append is not the last authoritative expected event"
        )
    initial_event_sequence_hash = canonical_sha256(events[:1])
    event_sequence_hash = canonical_sha256(events)
    genesis = _prediction_genesis(
        contract_hash=contract_hash,
        candidate_hash=candidate_hash,
        universe_hash=universe_hash,
        calendar_hash=calendar_hash,
        initial_event_sequence_sha256=initial_event_sequence_hash,
    )
    fold_context = _normalize_fold_context(
        current_fold_context,
        fold_id=event["fold_id"],
        stage=event["stage"],
        decision_session=event["decision_session"],
    )
    normalized_spec = _normalize_spec(prediction_spec, event=event)

    if prior_prefix is None:
        if len(events) != 1:
            raise SecFilingGemmaContractError(
                "First append requires exactly its one authoritative event"
            )
        if (
            expected_prior_prefix_sha256 is not None
            or expected_prior_tip_sha256 is not None
        ):
            raise SecFilingGemmaContractError(
                "First prediction cannot claim a prior prefix or tip"
            )
        prior_rows: list[dict[str, Any]] = []
        parent_prefix_hash = None
        parent_tip_hash = genesis
        input_states = _initial_policy_states()
    else:
        if len(events) < 2:
            raise SecFilingGemmaContractError(
                "Noninitial append requires its complete authoritative prefix"
            )
        if (
            expected_prior_prefix_sha256 is None
            or expected_prior_tip_sha256 is None
        ):
            raise SecFilingGemmaContractError(
                "Append requires externally pinned prior prefix and tip"
            )
        prior_summary = validate_prediction_prefix(
            prior_prefix,
            session_dates=sessions,
            expected_calendar_sessions_sha256=calendar_hash,
            expected_candidate_sha256=candidate_hash,
            expected_corpus_universe_sha256=universe_hash,
            expected_event_bindings=events[:-1],
            expected_prediction_prefix_sha256=expected_prior_prefix_sha256,
            expected_tip_sha256=expected_prior_tip_sha256,
        )
        parent_prefix_hash = prior_summary["prediction_prefix_sha256"]
        parent_tip_hash = prior_summary["tip_sha256"]
        prior_rows = copy.deepcopy(list(prior_prefix["rows"]))
        previous = prior_rows[-1]
        if (
            event["decision_session"],
            event["accession_number"],
        ) <= (
            previous["decision_session"],
            previous["accession_number"],
        ):
            raise SecFilingGemmaContractError(
                "Prediction append is duplicated, reordered, or from the past"
            )
        input_states = _roll_policy_states(
            previous["candidate_policy_output_states"],
            decision_session=event["decision_session"],
            sessions=sessions,
        )

    decision = event["decision_session"]
    fill = _session_offset(decision, 1, sessions)
    maturity = _session_offset(decision, LABEL_MATURITY_OFFSET, sessions)
    status = normalized_spec["prediction_status"]
    if status == AVAILABLE_PREDICTION_STATUS:
        semantic_probability = _decode_float_hex(
            normalized_spec["semantic_cash_probability_hex"],
            "semantic_cash_probability_hex",
            probability=True,
        )
        semantic_edge = _decode_float_hex(
            normalized_spec["semantic_expected_edge_hex"],
            "semantic_expected_edge_hex",
        )
        ablation_probability = _decode_float_hex(
            normalized_spec["ablation_cash_probability_hex"],
            "ablation_cash_probability_hex",
            probability=True,
        )
        ablation_edge = _decode_float_hex(
            normalized_spec["ablation_expected_edge_hex"],
            "ablation_expected_edge_hex",
        )
        raw_signals = _raw_gate_signals(
            semantic_probability=semantic_probability,
            semantic_edge=semantic_edge,
            ablation_probability=ablation_probability,
            ablation_edge=ablation_edge,
        )
        effective_actions, output_states = _available_policy_transition(
            input_states,
            raw_signals,
            decision_session=decision,
            sessions=sessions,
        )
        raw_signals_hash: str | None = canonical_sha256(raw_signals)
        actions_hash: str | None = canonical_sha256(effective_actions)
        fail_safe = None
    else:
        raw_signals = None
        effective_actions, output_states = _unavailable_policy_transition(
            input_states, sessions=sessions
        )
        raw_signals_hash = None
        actions_hash = canonical_sha256(effective_actions)
        fail_safe = "NO_NEW_EPISODE"

    row_body: dict[str, Any] = {
        "schema_version": PREDICTION_ROW_SCHEMA_VERSION,
        "sequence_number": len(prior_rows) + 1,
        "prior_prediction_prefix_sha256": parent_prefix_hash,
        "parent_prediction_sha256": parent_tip_hash,
        "contract_sha256": contract_hash,
        "candidate_sha256": candidate_hash,
        "corpus_universe_sha256": universe_hash,
        "calendar_sessions_sha256": calendar_hash,
        "event_sequence_sha256": event_sequence_hash,
        "fold_context": fold_context,
        "fold_context_sha256": canonical_sha256(fold_context),
        **event,
        "market_feature_cutoff_session": decision,
        "fill_session": fill,
        "cash_exit_session": maturity,
        "label_maturity_session": maturity,
        "horizon_sessions": HORIZON_SESSIONS,
        **normalized_spec,
        "raw_gate_signals": raw_signals,
        "raw_gate_signals_sha256": raw_signals_hash,
        "effective_episode_actions": effective_actions,
        "effective_episode_actions_sha256": actions_hash,
        "candidate_policy_input_states": input_states,
        "candidate_policy_input_states_sha256": canonical_sha256(input_states),
        "candidate_policy_output_states": output_states,
        "candidate_policy_output_states_sha256": canonical_sha256(output_states),
        "unavailable_fail_safe_action_semantics": fail_safe,
    }
    row = {**row_body, "prediction_row_sha256": canonical_sha256(row_body)}
    rows = prior_rows + [row]
    prefix_body = _prefix_body(
        contract_hash=contract_hash,
        candidate_hash=candidate_hash,
        universe_hash=universe_hash,
        calendar_hash=calendar_hash,
        initial_event_sequence_sha256=initial_event_sequence_hash,
        event_sequence_sha256=event_sequence_hash,
        genesis_hash=genesis,
        parent_prefix_hash=parent_prefix_hash,
        parent_tip_hash=parent_tip_hash,
        rows=rows,
    )
    prefix = {
        **prefix_body,
        "prediction_prefix_sha256": canonical_sha256(prefix_body),
    }
    validate_prediction_prefix(
        prefix,
        session_dates=sessions,
        expected_calendar_sessions_sha256=calendar_hash,
        expected_candidate_sha256=candidate_hash,
        expected_corpus_universe_sha256=universe_hash,
        expected_event_bindings=events,
        expected_prediction_prefix_sha256=prefix["prediction_prefix_sha256"],
        expected_tip_sha256=row["prediction_row_sha256"],
    )
    return prefix


def validate_prediction_prefix(
    value: Mapping[str, Any],
    *,
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
    expected_candidate_sha256: str,
    expected_corpus_universe_sha256: str,
    expected_event_bindings: Sequence[Mapping[str, Any]],
    expected_prediction_prefix_sha256: str | None = None,
    expected_tip_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay every row and every cumulative prefix ancestry link."""

    prefix = _expect_mapping(value, "prediction prefix")
    _expect_keys(prefix, _PREFIX_KEYS, "prediction prefix")
    sessions, calendar_hash = _canonical_sessions(
        session_dates, expected_calendar_sessions_sha256
    )
    contract_hash = canonical_sha256(build_contract_manifest())
    candidate_hash = _sha256(expected_candidate_sha256, "expected_candidate_sha256")
    universe_hash = _sha256(
        expected_corpus_universe_sha256, "expected_corpus_universe_sha256"
    )
    rows_value = prefix["rows"]
    if not isinstance(rows_value, list) or not rows_value:
        raise SecFilingGemmaContractError("Prediction prefix rows must be nonempty")
    external_events = _normalize_authoritative_event_sequence(
        expected_event_bindings, sessions=sessions
    )
    if len(external_events) != len(rows_value):
        raise SecFilingGemmaContractError(
            "Prediction prefix omits or adds an authoritative expected event"
        )
    initial_event_sequence_hash = canonical_sha256(external_events[:1])
    event_sequence_hash = canonical_sha256(external_events)
    expected_header = {
        "schema_version": PREDICTION_PREFIX_SCHEMA_VERSION,
        "contract_sha256": contract_hash,
        "candidate_sha256": candidate_hash,
        "corpus_universe_sha256": universe_hash,
        "calendar_sessions_sha256": calendar_hash,
        "initial_event_sequence_sha256": initial_event_sequence_hash,
        "event_sequence_sha256": event_sequence_hash,
    }
    for key, expected in expected_header.items():
        if prefix[key] != expected:
            raise SecFilingGemmaContractError(
                f"Prediction prefix {key} is not the current bound value"
            )
    if _strict_int(prefix["row_count"], "row_count", minimum=1) != len(rows_value):
        raise SecFilingGemmaContractError("Prediction prefix row count is wrong")
    if prefix["rows_sha256"] != canonical_sha256(rows_value):
        raise SecFilingGemmaContractError("Prediction prefix rows hash is wrong")

    genesis = _prediction_genesis(
        contract_hash=contract_hash,
        candidate_hash=candidate_hash,
        universe_hash=universe_hash,
        calendar_hash=calendar_hash,
        initial_event_sequence_sha256=initial_event_sequence_hash,
    )
    if prefix["genesis_sha256"] != genesis:
        raise SecFilingGemmaContractError("Prediction genesis changed")
    parent_prefix_hash: str | None = None
    parent_tip_hash = genesis
    previous_key: tuple[str, str] | None = None
    previous_output_states = _initial_policy_states()
    ancestry: list[str] = []
    accessions: set[str] = set()
    fold_contexts_seen: dict[str, dict[str, Any]] = {}

    for index, (raw_row, event) in enumerate(
        zip(rows_value, external_events, strict=True), start=1
    ):
        row = _expect_mapping(raw_row, f"prediction rows[{index - 1}]")
        _expect_keys(row, _PREDICTION_ROW_KEYS, f"prediction rows[{index - 1}]")
        if row["schema_version"] != PREDICTION_ROW_SCHEMA_VERSION:
            raise SecFilingGemmaContractError("Prediction row schema changed")
        if _strict_int(row["sequence_number"], "sequence_number", minimum=1) != index:
            raise SecFilingGemmaContractError(
                "Prediction sequence is omitted, duplicated, or reordered"
            )
        if (
            row["prior_prediction_prefix_sha256"] != parent_prefix_hash
            or row["parent_prediction_sha256"] != parent_tip_hash
        ):
            raise SecFilingGemmaContractError(
                "Prediction row is attached to the wrong prior prefix or tip"
            )
        for key, expected in (
            ("contract_sha256", contract_hash),
            ("candidate_sha256", candidate_hash),
            ("corpus_universe_sha256", universe_hash),
            ("calendar_sessions_sha256", calendar_hash),
        ):
            if row[key] != expected:
                raise SecFilingGemmaContractError(f"Prediction row {key} changed")
        current_event_sequence_hash = canonical_sha256(external_events[:index])
        if row["event_sequence_sha256"] != current_event_sequence_hash:
            raise SecFilingGemmaContractError(
                "Prediction row does not bind its authoritative event prefix"
            )
        for key in _EVENT_BINDING_KEYS:
            if row[key] != event[key]:
                raise SecFilingGemmaContractError(
                    "Prediction row is bound to another external event"
                )
        key = (event["decision_session"], event["accession_number"])
        if previous_key is not None and key <= previous_key:
            raise SecFilingGemmaContractError(
                "Prediction prefix is duplicated, omitted, or reordered"
            )
        previous_key = key
        if event["accession_number"] in accessions:
            raise SecFilingGemmaContractError("Prediction accession is duplicated")
        accessions.add(event["accession_number"])

        fold_context = _normalize_fold_context(
            row["fold_context"],
            fold_id=event["fold_id"],
            stage=event["stage"],
            decision_session=event["decision_session"],
        )
        if row["fold_context_sha256"] != canonical_sha256(fold_context):
            raise SecFilingGemmaContractError("Current fold context hash is wrong")
        prior_fold_context = fold_contexts_seen.setdefault(
            event["fold_id"], fold_context
        )
        if prior_fold_context != fold_context:
            raise SecFilingGemmaContractError(
                "Training set or model state changed inside a fixed fold"
            )
        decision = event["decision_session"]
        fill = _session_offset(decision, 1, sessions)
        maturity = _session_offset(decision, LABEL_MATURITY_OFFSET, sessions)
        if row["market_feature_cutoff_session"] != decision:
            raise SecFilingGemmaContractError(
                "Market features must stop at completed decision-session close t"
            )
        if row["fill_session"] != fill:
            raise SecFilingGemmaContractError("Prediction fill is not exactly t+1")
        if (
            row["cash_exit_session"] != maturity
            or row["label_maturity_session"] != maturity
        ):
            raise SecFilingGemmaContractError(
                "Cash exit and label maturity must both be exactly t+21"
            )
        if _strict_int(row["horizon_sessions"], "horizon_sessions") != HORIZON_SESSIONS:
            raise SecFilingGemmaContractError("Prediction horizon changed")

        expected_inputs = (
            _initial_policy_states()
            if index == 1
            else _roll_policy_states(
                previous_output_states,
                decision_session=decision,
                sessions=sessions,
            )
        )
        observed_inputs = _normalize_policy_states(
            row["candidate_policy_input_states"],
            sessions=sessions,
            location="candidate_policy_input_states",
        )
        if observed_inputs != expected_inputs:
            raise SecFilingGemmaContractError(
                "Per-candidate policy input states do not extend the prior prefix"
            )
        if row["candidate_policy_input_states_sha256"] != canonical_sha256(
            expected_inputs
        ):
            raise SecFilingGemmaContractError("Policy input-state hash is wrong")

        status = row["prediction_status"]
        if status == AVAILABLE_PREDICTION_STATUS:
            if row["unavailable_reason"] is not None:
                raise SecFilingGemmaContractError(
                    "Available prediction has an unavailable reason"
                )
            semantic_probability = _decode_float_hex(
                row["semantic_cash_probability_hex"],
                "semantic_cash_probability_hex",
                probability=True,
            )
            semantic_edge = _decode_float_hex(
                row["semantic_expected_edge_hex"], "semantic_expected_edge_hex"
            )
            ablation_probability = _decode_float_hex(
                row["ablation_cash_probability_hex"],
                "ablation_cash_probability_hex",
                probability=True,
            )
            ablation_edge = _decode_float_hex(
                row["ablation_expected_edge_hex"], "ablation_expected_edge_hex"
            )
            expected_signals = _raw_gate_signals(
                semantic_probability=semantic_probability,
                semantic_edge=semantic_edge,
                ablation_probability=ablation_probability,
                ablation_edge=ablation_edge,
            )
            observed_signals = _normalize_signal_or_action_map(
                row["raw_gate_signals"],
                location="raw_gate_signals",
                allowed={"LONG", "CASH"},
            )
            if observed_signals != expected_signals:
                raise SecFilingGemmaContractError("Raw gate signals do not reconcile")
            if row["raw_gate_signals_sha256"] != canonical_sha256(expected_signals):
                raise SecFilingGemmaContractError("Raw gate-signal hash is wrong")
            expected_actions, expected_outputs = _available_policy_transition(
                expected_inputs,
                expected_signals,
                decision_session=decision,
                sessions=sessions,
            )
            observed_actions = _normalize_signal_or_action_map(
                row["effective_episode_actions"],
                location="effective_episode_actions",
                allowed={
                    "STAY_LONG",
                    "START_CASH_EPISODE",
                    "HOLD_EXISTING_CASH_EPISODE",
                },
            )
            if observed_actions != expected_actions:
                raise SecFilingGemmaContractError(
                    "Effective episode actions do not reconcile"
                )
            if row["effective_episode_actions_sha256"] != canonical_sha256(
                expected_actions
            ):
                raise SecFilingGemmaContractError("Effective action hash is wrong")
            if row["unavailable_fail_safe_action_semantics"] is not None:
                raise SecFilingGemmaContractError(
                    "Available prediction cannot claim an unavailable fail-safe"
                )
        elif status == UNAVAILABLE_PREDICTION_STATUS:
            if row["unavailable_reason"] not in UNAVAILABLE_REASONS:
                raise SecFilingGemmaContractError(
                    "Unavailable prediction reason is invalid"
                )
            for field in (
                "semantic_cash_probability_hex",
                "semantic_expected_edge_hex",
                "ablation_cash_probability_hex",
                "ablation_expected_edge_hex",
                "raw_gate_signals",
                "raw_gate_signals_sha256",
            ):
                if row[field] is not None:
                    raise SecFilingGemmaContractError(
                        "Unavailable prediction claims a probability or gate signal"
                    )
            if row["unavailable_fail_safe_action_semantics"] != "NO_NEW_EPISODE":
                raise SecFilingGemmaContractError(
                    "Unavailable prediction fail-safe must forbid a new episode"
                )
            expected_actions, expected_outputs = _unavailable_policy_transition(
                expected_inputs, sessions=sessions
            )
            observed_actions = _normalize_signal_or_action_map(
                row["effective_episode_actions"],
                location="effective_episode_actions",
                allowed={"STAY_LONG", "HOLD_EXISTING_CASH_EPISODE"},
            )
            if observed_actions != expected_actions:
                raise SecFilingGemmaContractError(
                    "Unavailable effective actions do not match current policy states"
                )
            if row["effective_episode_actions_sha256"] != canonical_sha256(
                expected_actions
            ):
                raise SecFilingGemmaContractError(
                    "Unavailable effective action hash is wrong"
                )
        else:
            raise SecFilingGemmaContractError("Prediction status is invalid")

        observed_outputs = _normalize_policy_states(
            row["candidate_policy_output_states"],
            sessions=sessions,
            location="candidate_policy_output_states",
        )
        if observed_outputs != expected_outputs:
            raise SecFilingGemmaContractError(
                "Policy output states changed or extended an active episode"
            )
        if row["candidate_policy_output_states_sha256"] != canonical_sha256(
            expected_outputs
        ):
            raise SecFilingGemmaContractError("Policy output-state hash is wrong")

        row_body = {
            key: row[key]
            for key in _PREDICTION_ROW_KEYS
            if key != "prediction_row_sha256"
        }
        row_hash = canonical_sha256(row_body)
        if not hmac.compare_digest(
            row_hash, _sha256(row["prediction_row_sha256"], "prediction_row_sha256")
        ):
            raise SecFilingGemmaContractError("Prediction row hash is not canonical")

        current_body = _prefix_body(
            contract_hash=contract_hash,
            candidate_hash=candidate_hash,
            universe_hash=universe_hash,
            calendar_hash=calendar_hash,
            initial_event_sequence_sha256=initial_event_sequence_hash,
            event_sequence_sha256=current_event_sequence_hash,
            genesis_hash=genesis,
            parent_prefix_hash=parent_prefix_hash,
            parent_tip_hash=parent_tip_hash,
            rows=rows_value[:index],
        )
        current_prefix_hash = canonical_sha256(current_body)
        ancestry.append(current_prefix_hash)
        parent_prefix_hash = current_prefix_hash
        parent_tip_hash = row_hash
        previous_output_states = expected_outputs

    final_body = _prefix_body(
        contract_hash=contract_hash,
        candidate_hash=candidate_hash,
        universe_hash=universe_hash,
        calendar_hash=calendar_hash,
        initial_event_sequence_sha256=initial_event_sequence_hash,
        event_sequence_sha256=event_sequence_hash,
        genesis_hash=genesis,
        parent_prefix_hash=(ancestry[-2] if len(ancestry) > 1 else None),
        parent_tip_hash=(rows_value[-2]["prediction_row_sha256"] if len(rows_value) > 1 else genesis),
        rows=rows_value,
    )
    expected_prefix = {
        **final_body,
        "prediction_prefix_sha256": canonical_sha256(final_body),
    }
    if prefix != expected_prefix:
        raise SecFilingGemmaContractError(
            "Cumulative prediction prefix does not replay its exact ancestry"
        )
    prefix_hash = expected_prefix["prediction_prefix_sha256"]
    tip_hash = rows_value[-1]["prediction_row_sha256"]
    if expected_prediction_prefix_sha256 is not None and not hmac.compare_digest(
        prefix_hash,
        _sha256(
            expected_prediction_prefix_sha256,
            "expected_prediction_prefix_sha256",
        ),
    ):
        raise SecFilingGemmaContractError("Prediction prefix is not externally pinned")
    if expected_tip_sha256 is not None and not hmac.compare_digest(
        tip_hash, _sha256(expected_tip_sha256, "expected_tip_sha256")
    ):
        raise SecFilingGemmaContractError("Prediction tip is not externally pinned")
    return {
        "row_count": len(rows_value),
        "tip_sha256": tip_hash,
        "prediction_prefix_sha256": prefix_hash,
        "event_sequence_sha256": event_sequence_hash,
        "prefix_ancestry_sha256s": ancestry,
        "candidate_policy_output_states": copy.deepcopy(previous_output_states),
    }


def build_prediction_ledger(
    prediction_specs: Sequence[Mapping[str, Any]],
    *,
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
    candidate_sha256: str,
    corpus_universe_sha256: str,
    expected_event_bindings: Sequence[Mapping[str, Any]],
    fold_contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Convenience replay over causal event bindings and fixed fold state."""

    if (
        isinstance(prediction_specs, (str, bytes))
        or not isinstance(prediction_specs, Sequence)
        or isinstance(expected_event_bindings, (str, bytes))
        or not isinstance(expected_event_bindings, Sequence)
        or len(prediction_specs) != len(expected_event_bindings)
        or not prediction_specs
    ):
        raise SecFilingGemmaContractError(
            "Prediction specs and event bindings must be equal nonempty sequences"
        )
    fold_map = _expect_mapping(fold_contexts, "fold_contexts")
    prefix: dict[str, Any] | None = None
    for index, (spec, event_value) in enumerate(
        zip(prediction_specs, expected_event_bindings, strict=True), start=1
    ):
        event = _expect_mapping(event_value, "event binding")
        fold_id = event.get("fold_id")
        if fold_id not in fold_map:
            raise SecFilingGemmaContractError(
                "Current event lacks its fixed fold context"
            )
        prior_hash = None if prefix is None else prefix["prediction_prefix_sha256"]
        prior_tip = None if prefix is None else prefix["tip_sha256"]
        prefix = append_prediction_row(
            prefix,
            spec,
            current_event_binding=event,
            current_fold_context=fold_map[fold_id],
            session_dates=session_dates,
            expected_calendar_sessions_sha256=expected_calendar_sessions_sha256,
            candidate_sha256=candidate_sha256,
            corpus_universe_sha256=corpus_universe_sha256,
            expected_event_bindings=expected_event_bindings[:index],
            expected_prior_prefix_sha256=prior_hash,
            expected_prior_tip_sha256=prior_tip,
        )
    assert prefix is not None
    validate_prediction_prefix(
        prefix,
        session_dates=session_dates,
        expected_calendar_sessions_sha256=expected_calendar_sessions_sha256,
        expected_candidate_sha256=candidate_sha256,
        expected_corpus_universe_sha256=corpus_universe_sha256,
        expected_event_bindings=expected_event_bindings,
    )
    return prefix


def validate_prediction_ledger(
    value: Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Compatibility name for validating the current cumulative prefix."""

    return validate_prediction_prefix(value, **kwargs)


def prediction_prefix_sha256(
    prediction_prefix: Mapping[str, Any], *, through_sequence_number: int
) -> str:
    count = _strict_int(
        through_sequence_number, "through_sequence_number", minimum=1
    )
    rows = prediction_prefix.get("rows")
    if not isinstance(rows, list) or count > len(rows):
        raise SecFilingGemmaContractError("Requested prediction prefix is unavailable")
    # Every row carries the prior prefix, allowing the exact ancestry hash to be
    # recovered without any future context map.
    if count == len(rows):
        return _sha256(
            prediction_prefix["prediction_prefix_sha256"],
            "prediction_prefix_sha256",
        )
    contract_hash = _sha256(prediction_prefix["contract_sha256"], "contract_sha256")
    candidate_hash = _sha256(prediction_prefix["candidate_sha256"], "candidate_sha256")
    universe_hash = _sha256(
        prediction_prefix["corpus_universe_sha256"], "corpus_universe_sha256"
    )
    calendar_hash = _sha256(
        prediction_prefix["calendar_sessions_sha256"], "calendar_sessions_sha256"
    )
    genesis = _sha256(prediction_prefix["genesis_sha256"], "genesis_sha256")
    row = rows[count - 1]
    body = _prefix_body(
        contract_hash=contract_hash,
        candidate_hash=candidate_hash,
        universe_hash=universe_hash,
        calendar_hash=calendar_hash,
        initial_event_sequence_sha256=prediction_prefix[
            "initial_event_sequence_sha256"
        ],
        event_sequence_sha256=row["event_sequence_sha256"],
        genesis_hash=genesis,
        parent_prefix_hash=row["prior_prediction_prefix_sha256"],
        parent_tip_hash=row["parent_prediction_sha256"],
        rows=rows[:count],
    )
    return canonical_sha256(body)


def _normalize_external_checksums(
    value: Mapping[str, str], *, expected_prefixes: Sequence[str]
) -> dict[str, str]:
    checksums = _expect_mapping(
        value, "external_artifact_checksums_by_prefix_sha256"
    )
    if set(checksums) != set(expected_prefixes):
        raise SecFilingGemmaContractError(
            "External checksum map omits or adds a prediction prefix"
        )
    return {
        prefix: _tagged_sha256(checksums[prefix], f"external checksum {prefix}")
        for prefix in expected_prefixes
    }


def _seal_genesis(prediction_prefix: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "domain": "aapl-sec-gemma-pre-label-seal-genesis-v1",
            "contract_sha256": prediction_prefix["contract_sha256"],
            "candidate_sha256": prediction_prefix["candidate_sha256"],
            "corpus_universe_sha256": prediction_prefix[
                "corpus_universe_sha256"
            ],
            "calendar_sessions_sha256": prediction_prefix[
                "calendar_sessions_sha256"
            ],
            "event_sequence_sha256": prediction_prefix[
                "event_sequence_sha256"
            ],
        }
    )


def build_prelabel_seal_ledger(
    prediction_prefix: Mapping[str, Any],
    *,
    external_artifact_checksums_by_prefix_sha256: Mapping[str, str],
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
    expected_candidate_sha256: str,
    expected_corpus_universe_sha256: str,
    expected_event_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = validate_prediction_prefix(
        prediction_prefix,
        session_dates=session_dates,
        expected_calendar_sessions_sha256=expected_calendar_sessions_sha256,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_corpus_universe_sha256=expected_corpus_universe_sha256,
        expected_event_bindings=expected_event_bindings,
    )
    prefixes = summary["prefix_ancestry_sha256s"]
    checksums = _normalize_external_checksums(
        external_artifact_checksums_by_prefix_sha256,
        expected_prefixes=prefixes,
    )
    genesis = _seal_genesis(prediction_prefix)
    parent = genesis
    entries: list[dict[str, Any]] = []
    for index, (row, prefix_hash) in enumerate(
        zip(prediction_prefix["rows"], prefixes, strict=True), start=1
    ):
        body = {
            "schema_version": PRELABEL_SEAL_ENTRY_SCHEMA_VERSION,
            "sequence_number": index,
            "parent_seal_sha256": parent,
            "prediction_sequence_number": row["sequence_number"],
            "prediction_row_sha256": row["prediction_row_sha256"],
            "prediction_prefix_sha256": prefix_hash,
            "accession_number": row["accession_number"],
            "decision_session": row["decision_session"],
            "label_maturity_session": row["label_maturity_session"],
            "seal_protocol_phase": "pre_label_before_outcome_access",
            "prediction_artifact_checksum_sha256": checksums[prefix_hash],
        }
        entry = {**body, "seal_sha256": canonical_sha256(body)}
        entries.append(entry)
        parent = entry["seal_sha256"]
    body = {
        "schema_version": PRELABEL_SEAL_LEDGER_SCHEMA_VERSION,
        "contract_sha256": prediction_prefix["contract_sha256"],
        "candidate_sha256": prediction_prefix["candidate_sha256"],
        "corpus_universe_sha256": prediction_prefix["corpus_universe_sha256"],
        "calendar_sessions_sha256": prediction_prefix["calendar_sessions_sha256"],
        "event_sequence_sha256": summary["event_sequence_sha256"],
        "prediction_prefix_sha256": summary["prediction_prefix_sha256"],
        "sealed_prediction_count": len(entries),
        "sealed_prediction_tip_sha256": summary["tip_sha256"],
        "genesis_sha256": genesis,
        "tip_sha256": parent,
        "entries": entries,
    }
    return {**body, "prelabel_seal_ledger_sha256": canonical_sha256(body)}


def validate_prelabel_seal_ledger(
    value: Mapping[str, Any],
    *,
    prediction_prefix: Mapping[str, Any],
    external_artifact_checksums_by_prefix_sha256: Mapping[str, str],
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
    expected_candidate_sha256: str,
    expected_corpus_universe_sha256: str,
    expected_event_bindings: Sequence[Mapping[str, Any]],
    expected_seal_ledger_sha256: str | None = None,
    expected_tip_sha256: str | None = None,
) -> dict[str, Any]:
    prediction_summary = validate_prediction_prefix(
        prediction_prefix,
        session_dates=session_dates,
        expected_calendar_sessions_sha256=expected_calendar_sessions_sha256,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_corpus_universe_sha256=expected_corpus_universe_sha256,
        expected_event_bindings=expected_event_bindings,
    )
    ledger = _expect_mapping(value, "pre-label seal ledger")
    _expect_keys(ledger, _SEAL_LEDGER_KEYS, "pre-label seal ledger")
    expected_header = {
        "schema_version": PRELABEL_SEAL_LEDGER_SCHEMA_VERSION,
        "contract_sha256": prediction_prefix["contract_sha256"],
        "candidate_sha256": prediction_prefix["candidate_sha256"],
        "corpus_universe_sha256": prediction_prefix["corpus_universe_sha256"],
        "calendar_sessions_sha256": prediction_prefix["calendar_sessions_sha256"],
        "event_sequence_sha256": prediction_summary["event_sequence_sha256"],
        "prediction_prefix_sha256": prediction_summary["prediction_prefix_sha256"],
    }
    for key, expected in expected_header.items():
        if ledger[key] != expected:
            raise SecFilingGemmaContractError(f"Seal ledger {key} binding changed")
    rows = prediction_prefix["rows"]
    entries = ledger["entries"]
    if not isinstance(entries, list) or len(entries) != len(rows):
        raise SecFilingGemmaContractError(
            "Seal ledger omits, duplicates, or adds a prediction"
        )
    if _strict_int(
        ledger["sealed_prediction_count"], "sealed_prediction_count", minimum=1
    ) != len(rows):
        raise SecFilingGemmaContractError("Seal count does not reconcile")
    if ledger["sealed_prediction_tip_sha256"] != prediction_summary["tip_sha256"]:
        raise SecFilingGemmaContractError("Sealed prediction tip is wrong")
    prefixes = prediction_summary["prefix_ancestry_sha256s"]
    checksums = _normalize_external_checksums(
        external_artifact_checksums_by_prefix_sha256,
        expected_prefixes=prefixes,
    )
    genesis = _seal_genesis(prediction_prefix)
    if ledger["genesis_sha256"] != genesis:
        raise SecFilingGemmaContractError("Seal genesis changed")
    parent = genesis
    entries_by_row: dict[str, dict[str, Any]] = {}
    for index, (raw_entry, row, prefix_hash) in enumerate(
        zip(entries, rows, prefixes, strict=True), start=1
    ):
        entry = _expect_mapping(raw_entry, f"seal entries[{index - 1}]")
        _expect_keys(entry, _SEAL_ENTRY_KEYS, f"seal entries[{index - 1}]")
        expected_values = {
            "schema_version": PRELABEL_SEAL_ENTRY_SCHEMA_VERSION,
            "sequence_number": index,
            "parent_seal_sha256": parent,
            "prediction_sequence_number": row["sequence_number"],
            "prediction_row_sha256": row["prediction_row_sha256"],
            "prediction_prefix_sha256": prefix_hash,
            "accession_number": row["accession_number"],
            "decision_session": row["decision_session"],
            "label_maturity_session": row["label_maturity_session"],
            "seal_protocol_phase": "pre_label_before_outcome_access",
            "prediction_artifact_checksum_sha256": checksums[prefix_hash],
        }
        for key, expected in expected_values.items():
            if entry[key] != expected:
                raise SecFilingGemmaContractError(
                    "Seal is reordered, cross-prefix, or checksum-mismatched"
                )
        if entry["prediction_row_sha256"] in entries_by_row:
            raise SecFilingGemmaContractError("Prediction is sealed twice")
        body = {key: entry[key] for key in _SEAL_ENTRY_KEYS if key != "seal_sha256"}
        entry_hash = canonical_sha256(body)
        if not hmac.compare_digest(
            entry_hash, _sha256(entry["seal_sha256"], "seal_sha256")
        ):
            raise SecFilingGemmaContractError("Seal hash is not canonical")
        entries_by_row[entry["prediction_row_sha256"]] = copy.deepcopy(dict(entry))
        parent = entry_hash
    if ledger["tip_sha256"] != parent:
        raise SecFilingGemmaContractError("Seal tip is wrong")
    body = {
        key: ledger[key]
        for key in _SEAL_LEDGER_KEYS
        if key != "prelabel_seal_ledger_sha256"
    }
    ledger_hash = canonical_sha256(body)
    if not hmac.compare_digest(
        ledger_hash,
        _sha256(
            ledger["prelabel_seal_ledger_sha256"],
            "prelabel_seal_ledger_sha256",
        ),
    ):
        raise SecFilingGemmaContractError("Seal ledger hash is not canonical")
    if expected_seal_ledger_sha256 is not None and not hmac.compare_digest(
        ledger_hash,
        _sha256(expected_seal_ledger_sha256, "expected_seal_ledger_sha256"),
    ):
        raise SecFilingGemmaContractError("Seal ledger is not externally pinned")
    if expected_tip_sha256 is not None and not hmac.compare_digest(
        parent, _sha256(expected_tip_sha256, "expected_tip_sha256")
    ):
        raise SecFilingGemmaContractError("Seal tip is not externally pinned")
    return {
        "sealed_prediction_count": len(rows),
        "tip_sha256": parent,
        "prelabel_seal_ledger_sha256": ledger_hash,
        "entries_by_prediction_sha256": entries_by_row,
    }


def _normalize_label_specs(
    value: Mapping[str, Mapping[str, Any]], *, expected_rows: Sequence[str]
) -> dict[str, dict[str, Any]]:
    specs = _expect_mapping(value, "labels_by_prediction_sha256")
    if set(specs) != set(expected_rows):
        raise SecFilingGemmaContractError(
            "Labels contain an unsealed/premature row or omit an eligible row"
        )
    normalized: dict[str, dict[str, Any]] = {}
    for row_hash in expected_rows:
        raw = _expect_mapping(specs[row_hash], f"label {row_hash}")
        _expect_keys(raw, _LABEL_SPEC_KEYS, f"label {row_hash}")
        normalized[row_hash] = {
            "cash_active_log_edge_10bps_hex": _encode_float_hex(
                raw["cash_active_log_edge_10bps"],
                "cash_active_log_edge_10bps",
            ),
            "strategy_ledger_slice_sha256": _sha256(
                raw["strategy_ledger_slice_sha256"],
                "strategy_ledger_slice_sha256",
            ),
            "benchmark_ledger_slice_sha256": _sha256(
                raw["benchmark_ledger_slice_sha256"],
                "benchmark_ledger_slice_sha256",
            ),
            "outcome_ledger_row_sha256": _sha256(
                raw["outcome_ledger_row_sha256"], "outcome_ledger_row_sha256"
            ),
        }
    return normalized


def _release_genesis(
    prediction_prefix_hash: str,
    seal_ledger_hash: str,
    event_sequence_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "domain": "aapl-sec-gemma-label-release-genesis-v1",
            "prediction_prefix_sha256": prediction_prefix_hash,
            "prelabel_seal_ledger_sha256": seal_ledger_hash,
            "event_sequence_sha256": _sha256(
                event_sequence_sha256, "event_sequence_sha256"
            ),
        }
    )


def build_label_release_ledger(
    prediction_prefix: Mapping[str, Any],
    prelabel_seal_ledger: Mapping[str, Any],
    *,
    as_of_decision_session: str,
    labels_by_prediction_sha256: Mapping[str, Mapping[str, Any]],
    external_artifact_checksums_by_prefix_sha256: Mapping[str, str],
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
    expected_candidate_sha256: str,
    expected_corpus_universe_sha256: str,
    expected_event_bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seal_summary = validate_prelabel_seal_ledger(
        prelabel_seal_ledger,
        prediction_prefix=prediction_prefix,
        external_artifact_checksums_by_prefix_sha256=(
            external_artifact_checksums_by_prefix_sha256
        ),
        session_dates=session_dates,
        expected_calendar_sessions_sha256=expected_calendar_sessions_sha256,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_corpus_universe_sha256=expected_corpus_universe_sha256,
        expected_event_bindings=expected_event_bindings,
    )
    sessions, _ = _canonical_sessions(
        session_dates, expected_calendar_sessions_sha256
    )
    as_of = _iso_date(
        as_of_decision_session, "as_of_decision_session"
    ).isoformat()
    _session_offset(as_of, 0, sessions)
    seals = seal_summary["entries_by_prediction_sha256"]
    eligible_rows = [
        row
        for row in prediction_prefix["rows"]
        if row["prediction_row_sha256"] in seals
        and row["label_maturity_session"] < as_of
    ]
    labels = _normalize_label_specs(
        labels_by_prediction_sha256,
        expected_rows=[row["prediction_row_sha256"] for row in eligible_rows],
    )
    genesis = _release_genesis(
        prediction_prefix["prediction_prefix_sha256"],
        seal_summary["prelabel_seal_ledger_sha256"],
        prediction_prefix["event_sequence_sha256"],
    )
    parent = genesis
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(eligible_rows, start=1):
        row_hash = row["prediction_row_sha256"]
        seal = seals[row_hash]
        label = labels[row_hash]
        edge = _decode_float_hex(
            label["cash_active_log_edge_10bps_hex"],
            "cash_active_log_edge_10bps_hex",
        )
        first_eligible = _session_offset(
            row["label_maturity_session"], 1, sessions
        )
        if first_eligible > as_of:
            raise SecFilingGemmaContractError(
                "Label is not eligible for the as-of decision session"
            )
        binding = {
            "prediction_row_sha256": row_hash,
            "prelabel_seal_sha256": seal["seal_sha256"],
            "prediction_artifact_checksum_sha256": seal[
                "prediction_artifact_checksum_sha256"
            ],
            "horizon_sessions": HORIZON_SESSIONS,
            "cost_bps": BRIER_TARGET_COST_BPS,
            "cash_active_log_edge_10bps_hex": label[
                "cash_active_log_edge_10bps_hex"
            ],
            "cash_beats_long_10bps": edge > ACTIVE_EDGE_TOLERANCE,
            "strategy_ledger_slice_sha256": label[
                "strategy_ledger_slice_sha256"
            ],
            "benchmark_ledger_slice_sha256": label[
                "benchmark_ledger_slice_sha256"
            ],
            "outcome_ledger_row_sha256": label["outcome_ledger_row_sha256"],
        }
        body = {
            "schema_version": LABEL_RELEASE_ENTRY_SCHEMA_VERSION,
            "sequence_number": index,
            "parent_release_sha256": parent,
            "prediction_sequence_number": row["sequence_number"],
            "prediction_row_sha256": row_hash,
            "prediction_prefix_sha256": seal["prediction_prefix_sha256"],
            "prediction_artifact_checksum_sha256": seal[
                "prediction_artifact_checksum_sha256"
            ],
            "prelabel_seal_sha256": seal["seal_sha256"],
            "accession_number": row["accession_number"],
            "decision_session": row["decision_session"],
            "prediction_status": row["prediction_status"],
            "unavailable_reason": row["unavailable_reason"],
            "label_maturity_session": row["label_maturity_session"],
            "release_session": first_eligible,
            "first_eligible_prediction_session": first_eligible,
            "as_of_decision_session": as_of,
            **binding,
            "label_binding_sha256": canonical_sha256(binding),
        }
        entry = {**body, "release_sha256": canonical_sha256(body)}
        entries.append(entry)
        parent = entry["release_sha256"]
    body = {
        "schema_version": LABEL_RELEASE_LEDGER_SCHEMA_VERSION,
        "contract_sha256": prediction_prefix["contract_sha256"],
        "candidate_sha256": prediction_prefix["candidate_sha256"],
        "corpus_universe_sha256": prediction_prefix["corpus_universe_sha256"],
        "calendar_sessions_sha256": prediction_prefix["calendar_sessions_sha256"],
        "event_sequence_sha256": prediction_prefix["event_sequence_sha256"],
        "prediction_prefix_sha256": prediction_prefix[
            "prediction_prefix_sha256"
        ],
        "prelabel_seal_ledger_sha256": seal_summary[
            "prelabel_seal_ledger_sha256"
        ],
        "as_of_decision_session": as_of,
        "released_prediction_sequence_sha256": canonical_sha256(
            [entry["prediction_row_sha256"] for entry in entries]
        ),
        "release_count": len(entries),
        "genesis_sha256": genesis,
        "tip_sha256": parent,
        "entries": entries,
    }
    return {**body, "label_release_ledger_sha256": canonical_sha256(body)}


def validate_label_release_ledger(
    value: Mapping[str, Any],
    *,
    prediction_prefix: Mapping[str, Any],
    prelabel_seal_ledger: Mapping[str, Any],
    labels_by_prediction_sha256: Mapping[str, Mapping[str, Any]],
    external_artifact_checksums_by_prefix_sha256: Mapping[str, str],
    session_dates: Sequence[str],
    expected_calendar_sessions_sha256: str,
    expected_candidate_sha256: str,
    expected_corpus_universe_sha256: str,
    expected_event_bindings: Sequence[Mapping[str, Any]],
    expected_label_release_ledger_sha256: str | None = None,
    expected_tip_sha256: str | None = None,
) -> dict[str, Any]:
    seal_summary = validate_prelabel_seal_ledger(
        prelabel_seal_ledger,
        prediction_prefix=prediction_prefix,
        external_artifact_checksums_by_prefix_sha256=(
            external_artifact_checksums_by_prefix_sha256
        ),
        session_dates=session_dates,
        expected_calendar_sessions_sha256=expected_calendar_sessions_sha256,
        expected_candidate_sha256=expected_candidate_sha256,
        expected_corpus_universe_sha256=expected_corpus_universe_sha256,
        expected_event_bindings=expected_event_bindings,
    )
    ledger = _expect_mapping(value, "label release ledger")
    _expect_keys(ledger, _LABEL_LEDGER_KEYS, "label release ledger")
    sessions, _ = _canonical_sessions(
        session_dates, expected_calendar_sessions_sha256
    )
    as_of = _iso_date(
        ledger["as_of_decision_session"], "as_of_decision_session"
    ).isoformat()
    _session_offset(as_of, 0, sessions)
    expected_header = {
        "schema_version": LABEL_RELEASE_LEDGER_SCHEMA_VERSION,
        "contract_sha256": prediction_prefix["contract_sha256"],
        "candidate_sha256": prediction_prefix["candidate_sha256"],
        "corpus_universe_sha256": prediction_prefix["corpus_universe_sha256"],
        "calendar_sessions_sha256": prediction_prefix["calendar_sessions_sha256"],
        "event_sequence_sha256": prediction_prefix["event_sequence_sha256"],
        "prediction_prefix_sha256": prediction_prefix[
            "prediction_prefix_sha256"
        ],
        "prelabel_seal_ledger_sha256": seal_summary[
            "prelabel_seal_ledger_sha256"
        ],
    }
    for key, expected in expected_header.items():
        if ledger[key] != expected:
            raise SecFilingGemmaContractError(f"Label ledger {key} binding changed")
    seals = seal_summary["entries_by_prediction_sha256"]
    eligible_rows = [
        row
        for row in prediction_prefix["rows"]
        if row["prediction_row_sha256"] in seals
        and row["label_maturity_session"] < as_of
    ]
    labels = _normalize_label_specs(
        labels_by_prediction_sha256,
        expected_rows=[row["prediction_row_sha256"] for row in eligible_rows],
    )
    entries = ledger["entries"]
    if not isinstance(entries, list) or len(entries) != len(eligible_rows):
        raise SecFilingGemmaContractError(
            "Label ledger contains an unsealed, duplicate, premature, or omitted row"
        )
    if _strict_int(ledger["release_count"], "release_count") != len(entries):
        raise SecFilingGemmaContractError("Label release count is wrong")
    expected_sequence = [row["prediction_row_sha256"] for row in eligible_rows]
    if ledger["released_prediction_sequence_sha256"] != canonical_sha256(
        expected_sequence
    ):
        raise SecFilingGemmaContractError("Released prediction sequence changed")
    genesis = _release_genesis(
        prediction_prefix["prediction_prefix_sha256"],
        seal_summary["prelabel_seal_ledger_sha256"],
        prediction_prefix["event_sequence_sha256"],
    )
    if ledger["genesis_sha256"] != genesis:
        raise SecFilingGemmaContractError("Label release genesis changed")
    parent = genesis
    seen: set[str] = set()
    for index, (raw_entry, row) in enumerate(
        zip(entries, eligible_rows, strict=True), start=1
    ):
        entry = _expect_mapping(raw_entry, f"label entries[{index - 1}]")
        _expect_keys(entry, _LABEL_ENTRY_KEYS, f"label entries[{index - 1}]")
        row_hash = row["prediction_row_sha256"]
        if row_hash in seen:
            raise SecFilingGemmaContractError("Prediction label is duplicated")
        seen.add(row_hash)
        seal = seals.get(row_hash)
        if seal is None:
            raise SecFilingGemmaContractError("Label belongs to an unsealed prediction")
        label = labels[row_hash]
        edge = _decode_float_hex(
            label["cash_active_log_edge_10bps_hex"],
            "cash_active_log_edge_10bps_hex",
        )
        first_eligible = _session_offset(
            row["label_maturity_session"], 1, sessions
        )
        if not (row["label_maturity_session"] < as_of and first_eligible <= as_of):
            raise SecFilingGemmaContractError(
                "Label maturity must strictly precede the as-of decision"
            )
        binding = {
            "prediction_row_sha256": row_hash,
            "prelabel_seal_sha256": seal["seal_sha256"],
            "prediction_artifact_checksum_sha256": seal[
                "prediction_artifact_checksum_sha256"
            ],
            "horizon_sessions": HORIZON_SESSIONS,
            "cost_bps": BRIER_TARGET_COST_BPS,
            "cash_active_log_edge_10bps_hex": label[
                "cash_active_log_edge_10bps_hex"
            ],
            "cash_beats_long_10bps": edge > ACTIVE_EDGE_TOLERANCE,
            "strategy_ledger_slice_sha256": label[
                "strategy_ledger_slice_sha256"
            ],
            "benchmark_ledger_slice_sha256": label[
                "benchmark_ledger_slice_sha256"
            ],
            "outcome_ledger_row_sha256": label["outcome_ledger_row_sha256"],
        }
        expected_values = {
            "schema_version": LABEL_RELEASE_ENTRY_SCHEMA_VERSION,
            "sequence_number": index,
            "parent_release_sha256": parent,
            "prediction_sequence_number": row["sequence_number"],
            "prediction_row_sha256": row_hash,
            "prediction_prefix_sha256": seal["prediction_prefix_sha256"],
            "prediction_artifact_checksum_sha256": seal[
                "prediction_artifact_checksum_sha256"
            ],
            "prelabel_seal_sha256": seal["seal_sha256"],
            "accession_number": row["accession_number"],
            "decision_session": row["decision_session"],
            "prediction_status": row["prediction_status"],
            "unavailable_reason": row["unavailable_reason"],
            "label_maturity_session": row["label_maturity_session"],
            "release_session": first_eligible,
            "first_eligible_prediction_session": first_eligible,
            "as_of_decision_session": as_of,
            **binding,
            "label_binding_sha256": canonical_sha256(binding),
        }
        for key, expected in expected_values.items():
            if entry[key] != expected:
                raise SecFilingGemmaContractError(
                    "Label is cross-row, checksum-mismatched, or premature"
                )
        _strict_bool(entry["cash_beats_long_10bps"], "cash_beats_long_10bps")
        body = {key: entry[key] for key in _LABEL_ENTRY_KEYS if key != "release_sha256"}
        entry_hash = canonical_sha256(body)
        if not hmac.compare_digest(
            entry_hash, _sha256(entry["release_sha256"], "release_sha256")
        ):
            raise SecFilingGemmaContractError("Label release hash is not canonical")
        parent = entry_hash
    if ledger["tip_sha256"] != parent:
        raise SecFilingGemmaContractError("Label release tip is wrong")
    body = {
        key: ledger[key]
        for key in _LABEL_LEDGER_KEYS
        if key != "label_release_ledger_sha256"
    }
    ledger_hash = canonical_sha256(body)
    if not hmac.compare_digest(
        ledger_hash,
        _sha256(
            ledger["label_release_ledger_sha256"],
            "label_release_ledger_sha256",
        ),
    ):
        raise SecFilingGemmaContractError("Label release ledger is not canonical")
    if expected_label_release_ledger_sha256 is not None and not hmac.compare_digest(
        ledger_hash,
        _sha256(
            expected_label_release_ledger_sha256,
            "expected_label_release_ledger_sha256",
        ),
    ):
        raise SecFilingGemmaContractError("Label release ledger is not externally pinned")
    if expected_tip_sha256 is not None and not hmac.compare_digest(
        parent, _sha256(expected_tip_sha256, "expected_tip_sha256")
    ):
        raise SecFilingGemmaContractError("Label release tip is not externally pinned")
    return {
        "release_count": len(entries),
        "tip_sha256": parent,
        "label_release_ledger_sha256": ledger_hash,
    }


__all__ = [
    "AVAILABLE_PREDICTION_STATUS",
    "FINAL_FOLD_ID",
    "INTERMEDIATE_FOLD_ID",
    "LABEL_RELEASE_ENTRY_SCHEMA_VERSION",
    "LABEL_RELEASE_LEDGER_SCHEMA_VERSION",
    "MODEL_VARIANTS",
    "PREDICTION_PREFIX_SCHEMA_VERSION",
    "PREDICTION_ROW_SCHEMA_VERSION",
    "PREDICTION_STATUSES",
    "PRELABEL_SEAL_ENTRY_SCHEMA_VERSION",
    "PRELABEL_SEAL_LEDGER_SCHEMA_VERSION",
    "UNAVAILABLE_PREDICTION_STATUS",
    "UNAVAILABLE_REASONS",
    "append_prediction_row",
    "build_label_release_ledger",
    "build_prediction_ledger",
    "build_prelabel_seal_ledger",
    "prediction_prefix_sha256",
    "validate_label_release_ledger",
    "validate_prediction_ledger",
    "validate_prediction_prefix",
    "validate_prelabel_seal_ledger",
]
