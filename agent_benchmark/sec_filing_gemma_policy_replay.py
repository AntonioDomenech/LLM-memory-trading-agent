"""Owned development-only threshold and policy replay for SEC/Gemma OOF rows.

This module converts an externally pinned raw numeric OOF prediction batch into
the repository's existing append-only policy prefix.  It has no learner-state,
feature, label, outcome, price, filesystem, model-transport, network, scoring,
ranking, refit, sealing, or promotion authority.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hmac
import math
import re
from typing import Any, Final

from .sec_filing_gemma_contract import (
    CANDIDATE_IDS,
    DEVELOPMENT_POLICY_SESSION_DATES,
    SecFilingGemmaContractError,
    build_contract_manifest,
    canonical_sha256,
    development_policy_session_calendar_sha256,
    market_session_calendar_sha256,
)
from .sec_filing_gemma_learner_prediction import (
    OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION,
    SecFilingGemmaLearnerPredictionError,
    validate_owned_development_oof_prediction_batch_structure,
)
from .sec_filing_gemma_prediction_evidence import (
    AVAILABLE_PREDICTION_STATUS,
    MODEL_VARIANTS,
    UNAVAILABLE_PREDICTION_STATUS,
    build_prediction_ledger,
)
from .sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
)


OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-policy-replay-projection-v1"
)
OWNED_DEVELOPMENT_POLICY_REPLAY_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-policy-replay-batch-v2"
)
POLICY_REPLAY_BINDING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-policy-replay-binding-v1"
)
POLICY_REPLAY_INPUT_SPEC_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-policy-replay-input-spec-v1"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DEVELOPMENT_CUTOFF_SESSION: Final[str] = "2018-12-31"
_THRESHOLD_COMPARISON_RULE: Final[str] = (
    "probability_gte_and_expected_edge_gte"
)
_EPISODE_RULE: Final[str] = (
    "accepted_after_close_fill_t_plus_1_exit_t_plus_21_fixed_20_session_"
    "cash_episode_never_extend_scheduled_or_active_episode"
)
_UNAVAILABLE_PREDICTION_RULE: Final[str] = (
    "unavailable_prediction_starts_no_new_cash_episode_"
    "existing_episode_keeps_original_exit"
)
_POLICY_REPLAY_ORDER_RULE: Final[str] = (
    "source_raw_prediction_ordinal_ascending_exactly_once"
)
_SOURCE_MARKET_CALENDAR_SHA256: Final[str] = market_session_calendar_sha256(
    EXPECTED_MARKET_HISTORY_SESSIONS
)
_POLICY_SESSION_DATES: Final[tuple[str, ...]] = DEVELOPMENT_POLICY_SESSION_DATES
_POLICY_CALENDAR_SHA256: Final[str] = development_policy_session_calendar_sha256(
    _POLICY_SESSION_DATES
)

_EVENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "accession_number",
        "form",
        "stage",
        "decision_session",
        "extraction_identity_sha256",
        "market_prefix_chain_identity_sha256",
        "market_feature_row_sha256",
        "fold_id",
    }
)
_COMPONENT_FIELDS: Final[tuple[str, str, str, str]] = (
    "semantic_cash_probability_hex",
    "semantic_expected_edge_hex",
    "ablation_cash_probability_hex",
    "ablation_expected_edge_hex",
)
_INPUT_SPEC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "input_ordinal",
        "source_raw_prediction_row_sha256",
        "source_feature_row_sha256",
        "event_binding_sha256",
        "decision_session",
        "accession_number",
        "fold_id",
        "prediction_fold_context_sha256",
        "semantic_learner_state_sha256",
        "ablation_learner_state_sha256",
        "prediction_status",
        "unavailable_reason",
        "numerical_components_sha256",
        "policy_replay_input_spec_sha256",
    }
)
_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "binding_ordinal",
        "source_raw_prediction_row_sha256",
        "source_event_binding_sha256",
        "source_prediction_fold_context_sha256",
        "policy_prediction_row_sha256",
        "prediction_status",
        "unavailable_reason",
        "source_numerical_components_sha256",
        "policy_numerical_components_sha256",
        "numerical_components_equal_raw",
        "raw_gate_signals_sha256",
        "effective_episode_actions_sha256",
        "candidate_policy_input_states_sha256",
        "candidate_policy_output_states_sha256",
        "policy_replay_binding_sha256",
    }
)
_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact_stage",
        "development_root_scope_sha256",
        "development_policy_replay_plan_sha256",
        "source_prediction_projection_sha256",
        "source_prediction_batch_schema_version",
        "source_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "source_market_calendar_sessions_sha256",
        "policy_calendar_sessions_sha256",
        "development_cutoff_session",
        "source_prediction_event_count",
        "available_prediction_count",
        "unavailable_prediction_count",
        "candidate_count",
        "candidate_ids",
        "model_variant_count",
        "model_variant_ids",
        "candidate_grid",
        "candidate_grid_sha256",
        "threshold_comparison_rule",
        "cash_episode_rule",
        "unavailable_prediction_rule",
        "policy_replay_order_rule",
        "policy_replay_input_count",
        "policy_replay_input_specs_sha256",
        "policy_prefix",
        "policy_prefix_sha256",
        "binding_count",
        "raw_to_policy_bindings",
        "raw_to_policy_bindings_sha256",
        "prediction_components_equal_source",
        "policy_prefix_included",
        "raw_to_policy_bindings_included",
        "threshold_evaluation_authorized",
        "policy_transition_authorized",
        "source_feature_rows_included",
        "learner_states_included",
        "labels_included",
        "outcomes_included",
        "market_prices_included",
        "post_2018_prediction_rows_included",
        "post_2018_market_or_outcome_data_included",
        "calendar_schedule_after_cutoff_used_only_for_open_episode_dates",
        "numeric_prediction_authorized",
        "learner_fit_authorized",
        "model_transport_authorized",
        "network_access_authorized",
        "scoring_authorized",
        "candidate_selection_authorized",
        "ranking_authorized",
        "sealing_authorized",
        "refit_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
        "policy_replay_batch_sha256",
    }
)


class SecFilingGemmaPolicyReplayError(SecFilingGemmaContractError):
    """A raw OOF batch cannot produce one exact development policy replay."""


def _expect_dict(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecFilingGemmaPolicyReplayError(
            f"{location} must be an exact built-in object"
        )
    return value


def _expect_list(value: Any, location: str) -> list[Any]:
    if type(value) is not list:
        raise SecFilingGemmaPolicyReplayError(
            f"{location} must be an exact built-in list"
        )
    return value


def _expect_keys(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    if set(value) != set(expected):
        raise SecFilingGemmaPolicyReplayError(
            f"{location} keys changed; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _detached(value: Any, location: str, *, depth: int = 0) -> Any:
    if depth > 64:
        raise SecFilingGemmaPolicyReplayError(f"{location} is too deeply nested")
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SecFilingGemmaPolicyReplayError(f"{location} is non-finite")
        return value
    if type(value) is list:
        return [
            _detached(item, f"{location}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise SecFilingGemmaPolicyReplayError(
                f"{location} keys must be exact strings"
            )
        return {
            key: _detached(item, f"{location}.{key}", depth=depth + 1)
            for key, item in value.items()
        }
    raise SecFilingGemmaPolicyReplayError(
        f"{location} contains a non-JSON or subclass value"
    )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaPolicyReplayError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecFilingGemmaPolicyReplayError(
            f"{location} must be an exact integer >= {minimum}"
        )
    return value


def _self_hash(value: Mapping[str, Any], field: str, location: str) -> str:
    observed = _sha256(value.get(field), f"{location}.{field}")
    body = {key: value[key] for key in value if key != field}
    if not hmac.compare_digest(observed, canonical_sha256(body)):
        raise SecFilingGemmaPolicyReplayError(f"{location} self-hash changed")
    return observed


def _canonical_float_hex(
    value: Any, location: str, *, probability: bool = False
) -> float:
    if type(value) is not str:
        raise SecFilingGemmaPolicyReplayError(
            f"{location} must be canonical float.hex text"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecFilingGemmaPolicyReplayError(
            f"{location} must be canonical float.hex text"
        ) from exc
    if not math.isfinite(number) or number.hex() != value:
        raise SecFilingGemmaPolicyReplayError(
            f"{location} must be canonical finite float.hex text"
        )
    if probability and not 0.0 <= number <= 1.0:
        raise SecFilingGemmaPolicyReplayError(
            f"{location} probability lies outside [0,1]"
        )
    return number


def _component_body(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prediction_status": row["prediction_status"],
        "unavailable_reason": row["unavailable_reason"],
        **{field: row[field] for field in _COMPONENT_FIELDS},
    }


def _validated_source_batch(
    source_prediction_batch: Mapping[str, Any],
    *,
    expected_source_prediction_batch_sha256: str,
) -> dict[str, Any]:
    value = _expect_dict(
        _detached(source_prediction_batch, "source raw OOF prediction batch"),
        "source raw OOF prediction batch",
    )
    try:
        observed = validate_owned_development_oof_prediction_batch_structure(
            value,
            expected_prediction_batch_sha256=(
                expected_source_prediction_batch_sha256
            ),
        )
    except (SecFilingGemmaLearnerPredictionError, TypeError, ValueError) as exc:
        raise SecFilingGemmaPolicyReplayError(
            "Source raw OOF prediction batch failed exact structural replay"
        ) from exc
    expected_contract = canonical_sha256(build_contract_manifest())
    if (
        value["schema_version"]
        != OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION
        or observed != value["prediction_batch_sha256"]
        or value["artifact_stage"] != "development"
        or value["contract_sha256"] != expected_contract
        or value["calendar_sessions_sha256"]
        != _SOURCE_MARKET_CALENDAR_SHA256
        or value["development_cutoff_session"]
        != _DEVELOPMENT_CUTOFF_SESSION
        or value["prediction_fold_ids"]
        != ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
        or value["model_variant_ids"] != list(MODEL_VARIANTS)
    ):
        raise SecFilingGemmaPolicyReplayError(
            "Source raw OOF prediction batch crossed the frozen development boundary"
        )
    return value


def derive_development_policy_replay_input_specs(
    source_prediction_batch: Mapping[str, Any],
    *,
    expected_source_prediction_batch_sha256: str,
) -> list[dict[str, Any]]:
    """Derive compact, label-free authorization specs from exact raw rows."""

    source = _validated_source_batch(
        source_prediction_batch,
        expected_source_prediction_batch_sha256=(
            expected_source_prediction_batch_sha256
        ),
    )
    specs: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(source["raw_prediction_rows"], start=1):
        row = _expect_dict(raw, f"source raw prediction row {ordinal}")
        event = _expect_dict(row["event_binding"], f"source event {ordinal}")
        components_hash = canonical_sha256(_component_body(row))
        body = {
            "schema_version": POLICY_REPLAY_INPUT_SPEC_SCHEMA_VERSION,
            "input_ordinal": ordinal,
            "source_raw_prediction_row_sha256": row[
                "raw_prediction_row_sha256"
            ],
            "source_feature_row_sha256": row["source_feature_row_sha256"],
            "event_binding_sha256": row["event_binding_sha256"],
            "decision_session": event["decision_session"],
            "accession_number": event["accession_number"],
            "fold_id": row["fold_id"],
            "prediction_fold_context_sha256": row[
                "prediction_fold_context_sha256"
            ],
            "semantic_learner_state_sha256": row[
                "semantic_learner_state_sha256"
            ],
            "ablation_learner_state_sha256": row[
                "ablation_learner_state_sha256"
            ],
            "prediction_status": row["prediction_status"],
            "unavailable_reason": row["unavailable_reason"],
            "numerical_components_sha256": components_hash,
        }
        spec = {
            **body,
            "policy_replay_input_spec_sha256": canonical_sha256(body),
        }
        _expect_keys(
            spec,
            _INPUT_SPEC_KEYS,
            f"development policy replay input spec {ordinal}",
        )
        specs.append(spec)
    return specs


def _prediction_spec_from_raw(row: Mapping[str, Any]) -> dict[str, Any]:
    event = _expect_dict(row["event_binding"], "raw prediction event binding")
    _expect_keys(event, _EVENT_KEYS, "raw prediction event binding")
    status = row["prediction_status"]
    spec: dict[str, Any] = {
        **copy.deepcopy(event),
        "prediction_status": status,
        "unavailable_reason": row["unavailable_reason"],
    }
    if status == AVAILABLE_PREDICTION_STATUS:
        spec.update(
            {
                "semantic_cash_probability": _canonical_float_hex(
                    row["semantic_cash_probability_hex"],
                    "semantic_cash_probability_hex",
                    probability=True,
                ),
                "semantic_expected_edge": _canonical_float_hex(
                    row["semantic_expected_edge_hex"],
                    "semantic_expected_edge_hex",
                ),
                "ablation_cash_probability": _canonical_float_hex(
                    row["ablation_cash_probability_hex"],
                    "ablation_cash_probability_hex",
                    probability=True,
                ),
                "ablation_expected_edge": _canonical_float_hex(
                    row["ablation_expected_edge_hex"],
                    "ablation_expected_edge_hex",
                ),
            }
        )
    elif status == UNAVAILABLE_PREDICTION_STATUS:
        spec.update(
            {
                "semantic_cash_probability": None,
                "semantic_expected_edge": None,
                "ablation_cash_probability": None,
                "ablation_expected_edge": None,
            }
        )
    else:
        raise SecFilingGemmaPolicyReplayError("Raw prediction status changed")
    return spec


def _candidate_grid() -> list[dict[str, Any]]:
    grid = copy.deepcopy(build_contract_manifest()["predictor"]["candidate_grid"])
    if (
        type(grid) is not list
        or [item.get("candidate_id") for item in grid] != list(CANDIDATE_IDS)
        or any(type(item) is not dict for item in grid)
    ):
        raise SecFilingGemmaPolicyReplayError("Frozen candidate grid changed")
    return grid


def _build_policy_prefix(source: Mapping[str, Any]) -> dict[str, Any]:
    rows = _expect_list(source["raw_prediction_rows"], "source raw rows")
    specs: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    fold_contexts: dict[str, dict[str, Any]] = {}
    for ordinal, raw in enumerate(rows, start=1):
        row = _expect_dict(raw, f"source raw prediction row {ordinal}")
        spec = _prediction_spec_from_raw(row)
        event = copy.deepcopy(row["event_binding"])
        fold_id = row["fold_id"]
        context = copy.deepcopy(row["prediction_fold_context"])
        prior = fold_contexts.get(fold_id)
        if prior is not None and prior != context:
            raise SecFilingGemmaPolicyReplayError(
                "A frozen fold context changed inside its OOF window"
            )
        fold_contexts[fold_id] = context
        specs.append(spec)
        events.append(event)
    if set(fold_contexts) != {"fold_1", "fold_2", "fold_3", "fold_4", "fold_5"}:
        raise SecFilingGemmaPolicyReplayError(
            "Policy replay does not cover all five frozen development folds"
        )
    try:
        return build_prediction_ledger(
            specs,
            session_dates=_POLICY_SESSION_DATES,
            expected_calendar_sessions_sha256=_POLICY_CALENDAR_SHA256,
            candidate_sha256=source["candidate_sha256"],
            corpus_universe_sha256=source["corpus_universe_sha256"],
            expected_event_bindings=events,
            fold_contexts=fold_contexts,
        )
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaPolicyReplayError(
            "Raw OOF components failed exact threshold and policy replay"
        ) from exc


def _raw_to_policy_bindings(
    source: Mapping[str, Any], policy_prefix: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw_rows = _expect_list(source["raw_prediction_rows"], "source raw rows")
    policy_rows = _expect_list(policy_prefix.get("rows"), "policy prefix rows")
    if len(raw_rows) != len(policy_rows):
        raise SecFilingGemmaPolicyReplayError(
            "Policy replay did not map every raw row exactly once"
        )
    bindings: list[dict[str, Any]] = []
    for ordinal, (raw_value, policy_value) in enumerate(
        zip(raw_rows, policy_rows, strict=True), start=1
    ):
        raw = _expect_dict(raw_value, f"source raw row {ordinal}")
        policy = _expect_dict(policy_value, f"policy row {ordinal}")
        event = _expect_dict(raw["event_binding"], f"source event {ordinal}")
        policy_event = {key: policy.get(key) for key in _EVENT_KEYS}
        source_components = _component_body(raw)
        policy_components = _component_body(policy)
        source_component_hash = canonical_sha256(source_components)
        policy_component_hash = canonical_sha256(policy_components)
        if (
            raw["prediction_ordinal"] != ordinal
            or policy.get("sequence_number") != ordinal
            or policy_event != event
            or policy.get("fold_context") != raw["prediction_fold_context"]
            or policy.get("fold_context_sha256")
            != raw["prediction_fold_context_sha256"]
            or policy_components != source_components
            or source_component_hash != policy_component_hash
        ):
            raise SecFilingGemmaPolicyReplayError(
                "Policy row changed its raw event, fold context, or numeric components"
            )
        for field in (
            "prediction_row_sha256",
            "effective_episode_actions_sha256",
            "candidate_policy_input_states_sha256",
            "candidate_policy_output_states_sha256",
        ):
            _sha256(policy.get(field), f"policy row {ordinal}.{field}")
        gate_hash = policy.get("raw_gate_signals_sha256")
        if gate_hash is not None:
            _sha256(gate_hash, f"policy row {ordinal}.raw_gate_signals_sha256")
        body = {
            "schema_version": POLICY_REPLAY_BINDING_SCHEMA_VERSION,
            "binding_ordinal": ordinal,
            "source_raw_prediction_row_sha256": raw[
                "raw_prediction_row_sha256"
            ],
            "source_event_binding_sha256": raw["event_binding_sha256"],
            "source_prediction_fold_context_sha256": raw[
                "prediction_fold_context_sha256"
            ],
            "policy_prediction_row_sha256": policy["prediction_row_sha256"],
            "prediction_status": raw["prediction_status"],
            "unavailable_reason": raw["unavailable_reason"],
            "source_numerical_components_sha256": source_component_hash,
            "policy_numerical_components_sha256": policy_component_hash,
            "numerical_components_equal_raw": True,
            "raw_gate_signals_sha256": gate_hash,
            "effective_episode_actions_sha256": policy[
                "effective_episode_actions_sha256"
            ],
            "candidate_policy_input_states_sha256": policy[
                "candidate_policy_input_states_sha256"
            ],
            "candidate_policy_output_states_sha256": policy[
                "candidate_policy_output_states_sha256"
            ],
        }
        bindings.append(
            {**body, "policy_replay_binding_sha256": canonical_sha256(body)}
        )
    return bindings


def build_owned_development_policy_replay_batch(
    *,
    source_prediction_batch: Mapping[str, Any],
    expected_source_prediction_batch_sha256: str,
    expected_development_policy_replay_plan_sha256: str,
    expected_source_prediction_projection_sha256: str,
) -> dict[str, Any]:
    """Apply only frozen thresholds and chronological LONG/CASH transitions."""

    source = _validated_source_batch(
        source_prediction_batch,
        expected_source_prediction_batch_sha256=(
            expected_source_prediction_batch_sha256
        ),
    )
    plan_hash = _sha256(
        expected_development_policy_replay_plan_sha256,
        "expected development policy replay plan hash",
    )
    projection_hash = _sha256(
        expected_source_prediction_projection_sha256,
        "expected source prediction projection hash",
    )
    input_specs = derive_development_policy_replay_input_specs(
        source,
        expected_source_prediction_batch_sha256=source[
            "prediction_batch_sha256"
        ],
    )
    policy_prefix = _build_policy_prefix(source)
    bindings = _raw_to_policy_bindings(source, policy_prefix)
    grid = _candidate_grid()
    false_flags = {
        "source_feature_rows_included": False,
        "learner_states_included": False,
        "labels_included": False,
        "outcomes_included": False,
        "market_prices_included": False,
        "post_2018_prediction_rows_included": False,
        "post_2018_market_or_outcome_data_included": False,
        "numeric_prediction_authorized": False,
        "learner_fit_authorized": False,
        "model_transport_authorized": False,
        "network_access_authorized": False,
        "scoring_authorized": False,
        "candidate_selection_authorized": False,
        "ranking_authorized": False,
        "sealing_authorized": False,
        "refit_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    body = {
        "schema_version": OWNED_DEVELOPMENT_POLICY_REPLAY_BATCH_SCHEMA_VERSION,
        "artifact_stage": "development",
        "development_root_scope_sha256": source[
            "development_root_scope_sha256"
        ],
        "development_policy_replay_plan_sha256": plan_hash,
        "source_prediction_projection_sha256": projection_hash,
        "source_prediction_batch_schema_version": source["schema_version"],
        "source_prediction_batch_sha256": source["prediction_batch_sha256"],
        "source_raw_prediction_rows_sha256": source[
            "raw_prediction_rows_sha256"
        ],
        "source_raw_prediction_tip_sha256": source[
            "raw_prediction_tip_sha256"
        ],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "corpus_universe_sha256": source["corpus_universe_sha256"],
        "source_market_calendar_sessions_sha256": source[
            "calendar_sessions_sha256"
        ],
        "policy_calendar_sessions_sha256": _POLICY_CALENDAR_SHA256,
        "development_cutoff_session": source["development_cutoff_session"],
        "source_prediction_event_count": source["prediction_event_count"],
        "available_prediction_count": source["available_prediction_count"],
        "unavailable_prediction_count": source["unavailable_prediction_count"],
        "candidate_count": len(CANDIDATE_IDS),
        "candidate_ids": list(CANDIDATE_IDS),
        "model_variant_count": len(MODEL_VARIANTS),
        "model_variant_ids": list(MODEL_VARIANTS),
        "candidate_grid": grid,
        "candidate_grid_sha256": canonical_sha256(grid),
        "threshold_comparison_rule": _THRESHOLD_COMPARISON_RULE,
        "cash_episode_rule": _EPISODE_RULE,
        "unavailable_prediction_rule": _UNAVAILABLE_PREDICTION_RULE,
        "policy_replay_order_rule": _POLICY_REPLAY_ORDER_RULE,
        "policy_replay_input_count": len(input_specs),
        "policy_replay_input_specs_sha256": canonical_sha256(input_specs),
        "policy_prefix": copy.deepcopy(policy_prefix),
        "policy_prefix_sha256": policy_prefix["prediction_prefix_sha256"],
        "binding_count": len(bindings),
        "raw_to_policy_bindings": bindings,
        "raw_to_policy_bindings_sha256": canonical_sha256(bindings),
        "prediction_components_equal_source": True,
        "policy_prefix_included": True,
        "raw_to_policy_bindings_included": True,
        "threshold_evaluation_authorized": True,
        "policy_transition_authorized": True,
        **false_flags,
        "calendar_schedule_after_cutoff_used_only_for_open_episode_dates": True,
    }
    return {**body, "policy_replay_batch_sha256": canonical_sha256(body)}


def _validated_policy_batch_structure(raw: Any) -> dict[str, Any]:
    value = _expect_dict(
        _detached(raw, "owned development policy replay batch"),
        "owned development policy replay batch",
    )
    _expect_keys(value, _BATCH_KEYS, "owned development policy replay batch")
    _self_hash(
        value,
        "policy_replay_batch_sha256",
        "owned development policy replay batch",
    )
    bindings = _expect_list(
        value["raw_to_policy_bindings"], "policy replay bindings"
    )
    for ordinal, raw_binding in enumerate(bindings, start=1):
        binding = _expect_dict(raw_binding, f"policy replay binding {ordinal}")
        _expect_keys(binding, _BINDING_KEYS, f"policy replay binding {ordinal}")
        if (
            binding["schema_version"] != POLICY_REPLAY_BINDING_SCHEMA_VERSION
            or _strict_int(
                binding["binding_ordinal"],
                f"policy replay binding {ordinal}.ordinal",
                minimum=1,
            )
            != ordinal
            or binding["numerical_components_equal_raw"] is not True
            or binding["source_numerical_components_sha256"]
            != binding["policy_numerical_components_sha256"]
        ):
            raise SecFilingGemmaPolicyReplayError(
                "Policy replay binding identity or numerical equality changed"
            )
        _self_hash(
            binding,
            "policy_replay_binding_sha256",
            f"policy replay binding {ordinal}",
        )
    prefix = _expect_dict(value["policy_prefix"], "policy prefix")
    if (
        value["schema_version"]
        != OWNED_DEVELOPMENT_POLICY_REPLAY_BATCH_SCHEMA_VERSION
        or value["artifact_stage"] != "development"
        or value["source_prediction_batch_schema_version"]
        != OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION
        or value["contract_sha256"]
        != canonical_sha256(build_contract_manifest())
        or value["source_market_calendar_sessions_sha256"]
        != _SOURCE_MARKET_CALENDAR_SHA256
        or value["policy_calendar_sessions_sha256"]
        != _POLICY_CALENDAR_SHA256
        or value["development_cutoff_session"]
        != _DEVELOPMENT_CUTOFF_SESSION
        or value["candidate_count"] != len(CANDIDATE_IDS)
        or value["candidate_ids"] != list(CANDIDATE_IDS)
        or value["model_variant_count"] != len(MODEL_VARIANTS)
        or value["model_variant_ids"] != list(MODEL_VARIANTS)
        or value["candidate_grid"] != _candidate_grid()
        or value["candidate_grid_sha256"]
        != canonical_sha256(value["candidate_grid"])
        or value["threshold_comparison_rule"] != _THRESHOLD_COMPARISON_RULE
        or value["cash_episode_rule"] != _EPISODE_RULE
        or value["unavailable_prediction_rule"]
        != _UNAVAILABLE_PREDICTION_RULE
        or value["policy_replay_order_rule"] != _POLICY_REPLAY_ORDER_RULE
        or value["policy_prefix_sha256"]
        != prefix.get("prediction_prefix_sha256")
        or value["binding_count"] != len(bindings)
        or value["binding_count"] != value["source_prediction_event_count"]
        or value["policy_replay_input_count"] != len(bindings)
        or value["raw_to_policy_bindings_sha256"]
        != canonical_sha256(bindings)
    ):
        raise SecFilingGemmaPolicyReplayError(
            "Owned development policy replay identity or count changed"
        )
    for field in (
        "development_root_scope_sha256",
        "development_policy_replay_plan_sha256",
        "source_prediction_projection_sha256",
        "source_prediction_batch_sha256",
        "source_raw_prediction_rows_sha256",
        "source_raw_prediction_tip_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "source_market_calendar_sessions_sha256",
        "policy_calendar_sessions_sha256",
        "candidate_grid_sha256",
        "policy_replay_input_specs_sha256",
        "policy_prefix_sha256",
        "raw_to_policy_bindings_sha256",
    ):
        _sha256(value[field], f"owned policy replay.{field}")
    true_flags = (
        "prediction_components_equal_source",
        "policy_prefix_included",
        "raw_to_policy_bindings_included",
        "threshold_evaluation_authorized",
        "policy_transition_authorized",
        "calendar_schedule_after_cutoff_used_only_for_open_episode_dates",
    )
    false_flags = (
        "source_feature_rows_included",
        "learner_states_included",
        "labels_included",
        "outcomes_included",
        "market_prices_included",
        "post_2018_prediction_rows_included",
        "post_2018_market_or_outcome_data_included",
        "numeric_prediction_authorized",
        "learner_fit_authorized",
        "model_transport_authorized",
        "network_access_authorized",
        "scoring_authorized",
        "candidate_selection_authorized",
        "ranking_authorized",
        "sealing_authorized",
        "refit_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    )
    if any(type(value[field]) is not bool or value[field] is not True for field in true_flags):
        raise SecFilingGemmaPolicyReplayError(
            "Policy replay lost one of its two bounded authorities or output proofs"
        )
    if any(type(value[field]) is not bool or value[field] is not False for field in false_flags):
        raise SecFilingGemmaPolicyReplayError(
            "Policy replay crossed its threshold-and-transition-only boundary"
        )
    return value


def validate_owned_development_policy_replay_batch(
    batch: Mapping[str, Any],
    *,
    source_prediction_batch: Mapping[str, Any],
    expected_source_prediction_batch_sha256: str,
    expected_development_policy_replay_plan_sha256: str,
    expected_source_prediction_projection_sha256: str,
    expected_policy_replay_batch_sha256: str,
) -> str:
    """Validate by exact deterministic rebuild without fitting or predicting."""

    value = _validated_policy_batch_structure(batch)
    observed = value["policy_replay_batch_sha256"]
    expected_hash = _sha256(
        expected_policy_replay_batch_sha256,
        "expected owned development policy replay batch hash",
    )
    if not hmac.compare_digest(observed, expected_hash):
        raise SecFilingGemmaPolicyReplayError(
            "Development policy replay batch is not externally pinned"
        )
    source = _validated_source_batch(
        source_prediction_batch,
        expected_source_prediction_batch_sha256=(
            expected_source_prediction_batch_sha256
        ),
    )
    expected_plan_hash = _sha256(
        expected_development_policy_replay_plan_sha256,
        "expected development policy replay plan hash",
    )
    expected_projection_hash = _sha256(
        expected_source_prediction_projection_sha256,
        "expected source prediction projection hash",
    )
    if (
        value["development_root_scope_sha256"]
        != source["development_root_scope_sha256"]
        or value["development_policy_replay_plan_sha256"]
        != expected_plan_hash
        or value["source_prediction_projection_sha256"]
        != expected_projection_hash
    ):
        raise SecFilingGemmaPolicyReplayError(
            "Development policy replay crossed its plan, projection, or root ancestry"
        )
    rebuilt = build_owned_development_policy_replay_batch(
        source_prediction_batch=source_prediction_batch,
        expected_source_prediction_batch_sha256=(
            expected_source_prediction_batch_sha256
        ),
        expected_development_policy_replay_plan_sha256=(
            expected_development_policy_replay_plan_sha256
        ),
        expected_source_prediction_projection_sha256=(
            expected_source_prediction_projection_sha256
        ),
    )
    if value != rebuilt:
        raise SecFilingGemmaPolicyReplayError(
            "Development policy replay differs from exact deterministic rebuild"
        )
    return observed


__all__ = [
    "OWNED_DEVELOPMENT_POLICY_REPLAY_BATCH_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION",
    "POLICY_REPLAY_BINDING_SCHEMA_VERSION",
    "POLICY_REPLAY_INPUT_SPEC_SCHEMA_VERSION",
    "SecFilingGemmaPolicyReplayError",
    "build_owned_development_policy_replay_batch",
    "derive_development_policy_replay_input_specs",
    "validate_owned_development_policy_replay_batch",
]
