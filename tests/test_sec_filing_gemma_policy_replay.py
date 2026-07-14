from __future__ import annotations

import copy
import inspect
import json
import math
import socket
from typing import Any

import pytest

from agent_benchmark.sec_filing_gemma_contract import (
    CANDIDATE_IDS,
    DEVELOPMENT_FOLD_SPECS,
    build_contract_manifest,
    canonical_sha256,
    development_policy_session_calendar_sha256,
    market_session_calendar_sha256,
    session_calendar_sha256,
)
from agent_benchmark.sec_filing_gemma_learner import (
    MODEL_TYPE,
    STATE_SCHEMA_VERSION,
    SecFilingGemmaTwoHeadLearner,
)
from agent_benchmark.sec_filing_gemma_learner_prediction import (
    DEVELOPMENT_OOF_RAW_PREDICTION_ROW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION,
    SecFilingGemmaLearnerPredictionError,
    validate_owned_development_oof_prediction_batch_structure,
)
from agent_benchmark.sec_filing_gemma_policy_replay import (
    OWNED_DEVELOPMENT_POLICY_REPLAY_BATCH_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION,
    SecFilingGemmaPolicyReplayError,
    build_owned_development_policy_replay_batch,
    derive_development_policy_replay_input_specs,
    validate_owned_development_policy_replay_batch,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import MODEL_VARIANTS
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
)


def _digest(label: str) -> str:
    return canonical_sha256({"policy-replay-test": label})


_POLICY_REPLAY_PLAN_SHA256 = _digest("policy-replay-plan")
_SOURCE_PREDICTION_PROJECTION_SHA256 = _digest("source-prediction-projection")


def _fold_context(fold_id: str, cutoff: str) -> dict[str, Any]:
    ordinal = int(fold_id[-1])
    semantic_state = _digest(f"semantic-state-{fold_id}")
    ablation_state = _digest(f"ablation-state-{fold_id}")
    return {
        "fold_train_cutoff_session": cutoff,
        "training_set_count": 40 + ordinal,
        "training_positive_count": 10 + ordinal,
        "training_set_membership_sha256": _digest(f"membership-{fold_id}"),
        "semantic_training_feature_matrix_sha256": _digest(
            f"semantic-matrix-{fold_id}"
        ),
        "ablation_training_feature_matrix_sha256": _digest(
            f"ablation-matrix-{fold_id}"
        ),
        "training_binary_target_sha256": _digest(f"binary-{fold_id}"),
        "training_edge_target_sha256": _digest(f"edge-{fold_id}"),
        "training_set_max_label_maturity_session": cutoff,
        "semantic_fold_state_sha256": semantic_state,
        "ablation_fold_state_sha256": ablation_state,
    }


def _event(ordinal: int, session: str, fold_id: str) -> dict[str, Any]:
    year = session[2:4]
    return {
        "accession_number": f"0000320193-{year}-{ordinal:06d}",
        "form": "10-K" if ordinal % 2 else "10-Q",
        "stage": "development",
        "decision_session": session,
        "extraction_identity_sha256": _digest(f"extraction-{ordinal}"),
        "market_prefix_chain_identity_sha256": _digest(
            f"market-prefix-{ordinal}"
        ),
        "market_feature_row_sha256": _digest(f"market-feature-{ordinal}"),
        "fold_id": fold_id,
    }


def _raw_chain_genesis(batch: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "chain_domain": (
                "aapl-sec-gemma-development-oof-raw-prediction-chain-genesis-v1"
            ),
            "development_root_scope_sha256": batch[
                "development_root_scope_sha256"
            ],
            "contract_sha256": batch["contract_sha256"],
            "candidate_sha256": batch["candidate_sha256"],
            "corpus_universe_sha256": batch["corpus_universe_sha256"],
            "calendar_sessions_sha256": batch["calendar_sessions_sha256"],
            "development_cutoff_session": batch["development_cutoff_session"],
        }
    )


def _rechain_and_rehash(batch: dict[str, Any]) -> dict[str, Any]:
    parent = _raw_chain_genesis(batch)
    available = 0
    for ordinal, row in enumerate(batch["raw_prediction_rows"], start=1):
        row["prediction_ordinal"] = ordinal
        row["parent_raw_prediction_row_sha256"] = parent
        row["event_binding_sha256"] = canonical_sha256(row["event_binding"])
        row["fold_id"] = row["event_binding"]["fold_id"]
        row["prediction_fold_context_sha256"] = canonical_sha256(
            row["prediction_fold_context"]
        )
        row["semantic_learner_state_sha256"] = row[
            "prediction_fold_context"
        ]["semantic_fold_state_sha256"]
        row["ablation_learner_state_sha256"] = row[
            "prediction_fold_context"
        ]["ablation_fold_state_sha256"]
        body = {
            key: value
            for key, value in row.items()
            if key != "raw_prediction_row_sha256"
        }
        row["raw_prediction_row_sha256"] = canonical_sha256(body)
        parent = row["raw_prediction_row_sha256"]
        available += int(row["prediction_status"] == "available_pre_label")
    batch["prediction_event_count"] = len(batch["raw_prediction_rows"])
    batch["available_prediction_count"] = available
    batch["unavailable_prediction_count"] = len(batch["raw_prediction_rows"]) - available
    batch["prediction_call_count"] = 2 * available
    batch["maximum_prediction_calls"] = 2 * available
    batch["raw_prediction_rows_sha256"] = canonical_sha256(
        batch["raw_prediction_rows"]
    )
    batch["raw_prediction_tip_sha256"] = parent
    body = {
        key: value for key, value in batch.items() if key != "prediction_batch_sha256"
    }
    batch["prediction_batch_sha256"] = canonical_sha256(body)
    return batch


def _raw_batch(
    *,
    first_probability: float = 0.50,
    first_edge: float = 0.0025,
) -> dict[str, Any]:
    fold_by_id = {fold_id: cutoff for fold_id, cutoff, _first, _last in DEVELOPMENT_FOLD_SPECS}
    row_specs = [
        ("2005-01-03", "fold_1", "available_pre_label"),
        ("2005-01-04", "fold_1", "unavailable_pre_label"),
        ("2005-01-05", "fold_1", "available_pre_label"),
        ("2008-01-02", "fold_2", "available_pre_label"),
        ("2011-01-03", "fold_3", "available_pre_label"),
        ("2014-01-02", "fold_4", "available_pre_label"),
        ("2017-01-03", "fold_5", "available_pre_label"),
    ]
    rows: list[dict[str, Any]] = []
    for ordinal, (session, fold_id, status) in enumerate(row_specs, start=1):
        event = _event(ordinal, session, fold_id)
        context = _fold_context(fold_id, fold_by_id[fold_id])
        if status == "available_pre_label":
            if ordinal == 1:
                semantic_probability = first_probability
                semantic_edge = first_edge
            elif ordinal == 3:
                semantic_probability = 0.90
                semantic_edge = 0.10
            elif ordinal == len(row_specs):
                semantic_probability = 0.60
                semantic_edge = 0.01
            else:
                semantic_probability = 0.10
                semantic_edge = -0.01
            components = {
                "semantic_cash_probability_hex": semantic_probability.hex(),
                "semantic_expected_edge_hex": semantic_edge.hex(),
                "ablation_cash_probability_hex": (0.55 if ordinal == 1 else 0.10).hex(),
                "ablation_expected_edge_hex": (0.0025 if ordinal == 1 else -0.01).hex(),
            }
            unavailable_reason = None
        else:
            components = {
                "semantic_cash_probability_hex": None,
                "semantic_expected_edge_hex": None,
                "ablation_cash_probability_hex": None,
                "ablation_expected_edge_hex": None,
            }
            unavailable_reason = "missing_required_market_features"
        rows.append(
            {
                "schema_version": DEVELOPMENT_OOF_RAW_PREDICTION_ROW_SCHEMA_VERSION,
                "prediction_ordinal": ordinal,
                "parent_raw_prediction_row_sha256": _digest("placeholder"),
                "source_prediction_feature_input_sha256": _digest(
                    f"feature-input-{ordinal}"
                ),
                "source_feature_row_sha256": _digest(f"feature-row-{ordinal}"),
                "event_binding": event,
                "event_binding_sha256": canonical_sha256(event),
                "fold_id": fold_id,
                "prediction_fold_context": context,
                "prediction_fold_context_sha256": canonical_sha256(context),
                "semantic_learner_state_sha256": context[
                    "semantic_fold_state_sha256"
                ],
                "ablation_learner_state_sha256": context[
                    "ablation_fold_state_sha256"
                ],
                "prediction_status": status,
                "unavailable_reason": unavailable_reason,
                **components,
                "raw_prediction_row_sha256": _digest(f"placeholder-row-{ordinal}"),
            }
        )
    batch = {
        "schema_version": OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION,
        "artifact_stage": "development",
        "development_root_scope_sha256": _digest("root-scope"),
        "development_oof_prediction_plan_sha256": _digest("prediction-plan"),
        "source_feature_batch_sha256": _digest("source-feature-batch"),
        "prediction_feature_batch_sha256": _digest("prediction-feature-batch"),
        "source_learner_fit_batch_sha256": _digest("learner-fit-batch"),
        "prediction_fold_model_bundle_sha256": _digest("fold-model-bundle"),
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "candidate_sha256": _digest("candidate"),
        "corpus_universe_sha256": _digest("corpus"),
        "calendar_sessions_sha256": market_session_calendar_sha256(
            EXPECTED_MARKET_HISTORY_SESSIONS
        ),
        "development_cutoff_session": "2018-12-31",
        "prediction_event_count": len(rows),
        "available_prediction_count": 0,
        "unavailable_prediction_count": 0,
        "prediction_call_count": 0,
        "prediction_fold_count": 5,
        "prediction_fold_ids": [f"fold_{index}" for index in range(1, 6)],
        "model_variant_count": 2,
        "model_variant_ids": list(MODEL_VARIANTS),
        "learner_model_type": MODEL_TYPE,
        "learner_state_schema_version": STATE_SCHEMA_VERSION,
        "feature_schema_sha256": _digest("feature-schema"),
        "maximum_prediction_calls": 0,
        "maximum_prediction_seconds": 60,
        "prediction_order_rule": (
            "source_feature_event_order_filtered_to_five_frozen_development_oof_windows"
        ),
        "raw_prediction_rows": rows,
        "raw_prediction_rows_sha256": _digest("placeholder-rows"),
        "raw_prediction_tip_sha256": _digest("placeholder-tip"),
        "source_feature_rows_included": False,
        "learner_states_included": False,
        "training_membership_rows_included": False,
        "training_feature_matrices_included": False,
        "training_target_vectors_included": False,
        "labels_included": False,
        "outcomes_included": False,
        "post_2018_data_included": False,
        "prediction_components_included": True,
        "candidate_selection_authorized": False,
        "threshold_action_authorized": False,
        "policy_transition_authorized": False,
        "scoring_authorized": False,
        "sealing_authorized": False,
        "label_release_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
        "prediction_batch_sha256": _digest("placeholder-batch"),
    }
    return _rechain_and_rehash(batch)


def _build(batch: dict[str, Any]) -> dict[str, Any]:
    return build_owned_development_policy_replay_batch(
        source_prediction_batch=batch,
        expected_source_prediction_batch_sha256=batch["prediction_batch_sha256"],
        expected_development_policy_replay_plan_sha256=(
            _POLICY_REPLAY_PLAN_SHA256
        ),
        expected_source_prediction_projection_sha256=(
            _SOURCE_PREDICTION_PROJECTION_SHA256
        ),
    )


def test_policy_replay_is_exact_owned_numeric_only_and_deterministic() -> None:
    source = _raw_batch()
    result = _build(source)
    assert result["schema_version"] == OWNED_DEVELOPMENT_POLICY_REPLAY_BATCH_SCHEMA_VERSION
    assert OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION.endswith("-v1")
    assert result["candidate_ids"] == list(CANDIDATE_IDS)
    assert result["model_variant_ids"] == list(MODEL_VARIANTS)
    assert result["development_root_scope_sha256"] == source[
        "development_root_scope_sha256"
    ]
    assert result["development_policy_replay_plan_sha256"] == (
        _POLICY_REPLAY_PLAN_SHA256
    )
    assert result["source_prediction_projection_sha256"] == (
        _SOURCE_PREDICTION_PROJECTION_SHA256
    )
    assert result["threshold_evaluation_authorized"] is True
    assert result["policy_transition_authorized"] is True
    assert result["unavailable_prediction_rule"] == (
        "unavailable_prediction_starts_no_new_cash_episode_"
        "existing_episode_keeps_original_exit"
    )
    assert result["policy_replay_order_rule"] == (
        "source_raw_prediction_ordinal_ascending_exactly_once"
    )
    assert result["labels_included"] is False
    assert result["outcomes_included"] is False
    assert result["market_prices_included"] is False
    assert result["holdout_access_authorized"] is False
    assert result["source_prediction_event_count"] == len(
        result["policy_prefix"]["rows"]
    ) == len(result["raw_to_policy_bindings"])
    for raw, policy, binding in zip(
        source["raw_prediction_rows"],
        result["policy_prefix"]["rows"],
        result["raw_to_policy_bindings"],
        strict=True,
    ):
        for field in (
            "semantic_cash_probability_hex",
            "semantic_expected_edge_hex",
            "ablation_cash_probability_hex",
            "ablation_expected_edge_hex",
        ):
            assert policy[field] == raw[field]
        assert binding["source_raw_prediction_row_sha256"] == raw[
            "raw_prediction_row_sha256"
        ]
        assert binding["policy_prediction_row_sha256"] == policy[
            "prediction_row_sha256"
        ]
        assert binding["source_numerical_components_sha256"] == binding[
            "policy_numerical_components_sha256"
        ]
    assert _build(source) == result
    assert validate_owned_development_policy_replay_batch(
        result,
        source_prediction_batch=source,
        expected_source_prediction_batch_sha256=source["prediction_batch_sha256"],
        expected_development_policy_replay_plan_sha256=(
            _POLICY_REPLAY_PLAN_SHA256
        ),
        expected_source_prediction_projection_sha256=(
            _SOURCE_PREDICTION_PROJECTION_SHA256
        ),
        expected_policy_replay_batch_sha256=result["policy_replay_batch_sha256"],
    ) == result["policy_replay_batch_sha256"]


@pytest.mark.parametrize(
    ("probability", "expected_signal"),
    [
        (math.nextafter(0.50, -math.inf), "LONG"),
        (0.50, "CASH"),
        (math.nextafter(0.50, math.inf), "CASH"),
    ],
)
def test_probability_threshold_uses_exact_gte_boundary(
    probability: float, expected_signal: str
) -> None:
    result = _build(_raw_batch(first_probability=probability, first_edge=0.01))
    first = result["policy_prefix"]["rows"][0]
    assert first["raw_gate_signals"]["p50_e0"]["semantic"] == expected_signal


@pytest.mark.parametrize(
    ("edge", "expected_signal"),
    [
        (math.nextafter(0.0025, -math.inf), "LONG"),
        (0.0025, "CASH"),
        (math.nextafter(0.0025, math.inf), "CASH"),
    ],
)
def test_expected_edge_threshold_uses_exact_gte_boundary(
    edge: float, expected_signal: str
) -> None:
    result = _build(_raw_batch(first_probability=0.90, first_edge=edge))
    first = result["policy_prefix"]["rows"][0]
    assert first["raw_gate_signals"]["p50_e25"]["semantic"] == expected_signal


def test_unavailable_row_cannot_start_or_extend_and_episode_is_t_plus_1_t_plus_21() -> None:
    result = _build(_raw_batch(first_probability=0.90, first_edge=0.10))
    rows = result["policy_prefix"]["rows"]
    first_state = rows[0]["candidate_policy_output_states"]["p50_e25"]["semantic"]
    unavailable_state = rows[1]["candidate_policy_output_states"]["p50_e25"]["semantic"]
    later_state = rows[2]["candidate_policy_output_states"]["p50_e25"]["semantic"]
    decision_index = EXPECTED_SESSIONS.index("2005-01-03")
    assert first_state["episode_fill_session"] == EXPECTED_SESSIONS[decision_index + 1]
    assert first_state["episode_exit_session"] == EXPECTED_SESSIONS[decision_index + 21]
    assert rows[1]["raw_gate_signals"] is None
    assert rows[1]["unavailable_fail_safe_action_semantics"] == "NO_NEW_EPISODE"
    assert rows[1]["candidate_policy_input_states"]["p50_e25"]["semantic"][
        "episode_phase"
    ] == "ACTIVE"
    assert rows[1]["candidate_policy_input_states"]["p50_e25"]["semantic"][
        "position_at_decision_close"
    ] == "CASH"
    assert rows[1]["effective_episode_actions"]["p50_e25"]["semantic"] == (
        "HOLD_EXISTING_CASH_EPISODE"
    )
    assert rows[2]["effective_episode_actions"]["p50_e25"]["semantic"] == (
        "HOLD_EXISTING_CASH_EPISODE"
    )
    assert unavailable_state["episode_origin_decision_session"] == first_state[
        "episode_origin_decision_session"
    ]
    assert later_state["episode_exit_session"] == first_state["episode_exit_session"]


def test_late_2018_signal_uses_only_calendar_schedule_for_2019_exit_dates() -> None:
    source = _raw_batch()
    late = copy.deepcopy(source["raw_prediction_rows"][-1])
    late_event = _event(8, "2018-12-31", "fold_5")
    late.update(
        {
            "prediction_ordinal": 8,
            "source_prediction_feature_input_sha256": _digest(
                "feature-input-late-2018"
            ),
            "source_feature_row_sha256": _digest("feature-row-late-2018"),
            "event_binding": late_event,
            "event_binding_sha256": canonical_sha256(late_event),
            "semantic_cash_probability_hex": (0.90).hex(),
            "semantic_expected_edge_hex": (0.10).hex(),
            "ablation_cash_probability_hex": (0.10).hex(),
            "ablation_expected_edge_hex": (-0.01).hex(),
        }
    )
    source["raw_prediction_rows"].append(late)
    _rechain_and_rehash(source)
    result = _build(source)
    policy = result["policy_prefix"]["rows"][-1]
    state = policy["candidate_policy_output_states"]["p50_e25"]["semantic"]
    decision_index = EXPECTED_SESSIONS.index("2018-12-31")
    assert policy["decision_session"] == "2018-12-31"
    assert state["episode_fill_session"] == EXPECTED_SESSIONS[decision_index + 1]
    assert state["episode_exit_session"] == EXPECTED_SESSIONS[decision_index + 21]
    assert state["episode_fill_session"] > "2018-12-31"
    assert state["episode_exit_session"] > "2018-12-31"
    assert result[
        "calendar_schedule_after_cutoff_used_only_for_open_episode_dates"
    ] is True
    assert result["post_2018_prediction_rows_included"] is False
    assert result["post_2018_market_or_outcome_data_included"] is False
    assert result["market_prices_included"] is False
    authorized_schedule = EXPECTED_SESSIONS[: decision_index + 22]
    assert authorized_schedule[-1] == "2019-01-31"
    assert result[
        "policy_calendar_sessions_sha256"
    ] == development_policy_session_calendar_sha256(authorized_schedule)
    assert result[
        "policy_calendar_sessions_sha256"
    ] != session_calendar_sha256(EXPECTED_SESSIONS)

    stack: list[Any] = [result]
    observed_keys: set[str] = set()
    while stack:
        value = stack.pop()
        if type(value) is dict:
            observed_keys.update(value)
            stack.extend(value.values())
        elif type(value) is list:
            stack.extend(value)
    assert {
        "adjusted_open_hex",
        "adjusted_close_hex",
        "feature_values_hex",
        "cash_active_log_edge_10bps_hex",
        "cash_beats_long_10bps",
    }.isdisjoint(observed_keys)


def test_later_raw_change_cannot_rewrite_earlier_policy_rows_or_bindings() -> None:
    source = _raw_batch()
    baseline = _build(source)
    changed = copy.deepcopy(source)
    last = changed["raw_prediction_rows"][-1]
    last["semantic_cash_probability_hex"] = (0.20).hex()
    last["semantic_expected_edge_hex"] = (-0.02).hex()
    _rechain_and_rehash(changed)
    replayed = _build(changed)
    assert replayed["policy_prefix"]["rows"][:-1] == baseline["policy_prefix"][
        "rows"
    ][:-1]
    assert replayed["raw_to_policy_bindings"][:-1] == baseline[
        "raw_to_policy_bindings"
    ][:-1]
    assert replayed["policy_prefix"]["rows"][-1] != baseline["policy_prefix"][
        "rows"
    ][-1]


def test_structure_pin_and_compact_input_specs_reject_crossed_batch() -> None:
    source = _raw_batch()
    changed = copy.deepcopy(source)
    changed["raw_prediction_rows"][-1]["semantic_expected_edge_hex"] = (0.02).hex()
    _rechain_and_rehash(changed)
    with pytest.raises(
        SecFilingGemmaLearnerPredictionError, match="externally pinned"
    ):
        validate_owned_development_oof_prediction_batch_structure(
            changed,
            expected_prediction_batch_sha256=source["prediction_batch_sha256"],
        )
    specs = derive_development_policy_replay_input_specs(
        source,
        expected_source_prediction_batch_sha256=source["prediction_batch_sha256"],
    )
    assert len(specs) == source["prediction_event_count"]
    assert [spec["input_ordinal"] for spec in specs] == list(
        range(1, len(specs) + 1)
    )
    assert all(
        "label" not in key and "outcome" not in key
        for spec in specs
        for key in spec
    )


@pytest.mark.parametrize(
    "mutation", ["omit", "duplicate", "reorder", "future", "context", "status"]
)
def test_omission_duplicate_reorder_future_context_and_status_fail_closed(
    mutation: str,
) -> None:
    source = _raw_batch()
    changed = copy.deepcopy(source)
    rows = changed["raw_prediction_rows"]
    if mutation == "omit":
        del rows[1]
    elif mutation == "duplicate":
        rows.insert(1, copy.deepcopy(rows[0]))
    elif mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "future":
        rows[-1]["event_binding"]["decision_session"] = "2019-01-02"
    elif mutation == "context":
        rows[2]["prediction_fold_context"]["training_set_count"] += 1
    else:
        rows[1]["prediction_status"] = "available_pre_label"
    _rechain_and_rehash(changed)
    with pytest.raises(SecFilingGemmaPolicyReplayError):
        if mutation == "omit":
            build_owned_development_policy_replay_batch(
                source_prediction_batch=changed,
                expected_source_prediction_batch_sha256=source[
                    "prediction_batch_sha256"
                ],
                expected_development_policy_replay_plan_sha256=(
                    _POLICY_REPLAY_PLAN_SHA256
                ),
                expected_source_prediction_projection_sha256=(
                    _SOURCE_PREDICTION_PROJECTION_SHA256
                ),
            )
        else:
            _build(changed)


def test_noncanonical_float_hex_is_rejected_even_when_every_hash_is_rebuilt() -> None:
    changed = _raw_batch()
    changed["raw_prediction_rows"][0]["semantic_cash_probability_hex"] = "0x1p-1"
    _rechain_and_rehash(changed)
    with pytest.raises(SecFilingGemmaPolicyReplayError, match="structural replay"):
        _build(changed)


def test_rehashed_output_tamper_is_rejected_by_exact_rebuild() -> None:
    source = _raw_batch()
    result = _build(source)
    changed = copy.deepcopy(result)
    binding = changed["raw_to_policy_bindings"][0]
    binding["source_raw_prediction_row_sha256"] = _digest("crossed-row")
    binding_body = {
        key: value
        for key, value in binding.items()
        if key != "policy_replay_binding_sha256"
    }
    binding["policy_replay_binding_sha256"] = canonical_sha256(binding_body)
    changed["raw_to_policy_bindings_sha256"] = canonical_sha256(
        changed["raw_to_policy_bindings"]
    )
    changed_body = {
        key: value
        for key, value in changed.items()
        if key != "policy_replay_batch_sha256"
    }
    changed["policy_replay_batch_sha256"] = canonical_sha256(changed_body)
    with pytest.raises(SecFilingGemmaPolicyReplayError, match="exact deterministic rebuild"):
        validate_owned_development_policy_replay_batch(
            changed,
            source_prediction_batch=source,
            expected_source_prediction_batch_sha256=source[
                "prediction_batch_sha256"
            ],
            expected_development_policy_replay_plan_sha256=(
                _POLICY_REPLAY_PLAN_SHA256
            ),
            expected_source_prediction_projection_sha256=(
                _SOURCE_PREDICTION_PROJECTION_SHA256
            ),
            expected_policy_replay_batch_sha256=changed[
                "policy_replay_batch_sha256"
            ],
        )


@pytest.mark.parametrize(
    "field",
    [
        "development_root_scope_sha256",
        "development_policy_replay_plan_sha256",
        "source_prediction_projection_sha256",
    ],
)
def test_rehashed_plan_projection_or_root_crossing_is_rejected(field: str) -> None:
    source = _raw_batch()
    changed = _build(source)
    changed[field] = _digest(f"crossed-{field}")
    changed_body = {
        key: value
        for key, value in changed.items()
        if key != "policy_replay_batch_sha256"
    }
    changed["policy_replay_batch_sha256"] = canonical_sha256(changed_body)
    with pytest.raises(SecFilingGemmaPolicyReplayError, match="ancestry"):
        validate_owned_development_policy_replay_batch(
            changed,
            source_prediction_batch=source,
            expected_source_prediction_batch_sha256=source[
                "prediction_batch_sha256"
            ],
            expected_development_policy_replay_plan_sha256=(
                _POLICY_REPLAY_PLAN_SHA256
            ),
            expected_source_prediction_projection_sha256=(
                _SOURCE_PREDICTION_PROJECTION_SHA256
            ),
            expected_policy_replay_batch_sha256=changed[
                "policy_replay_batch_sha256"
            ],
        )


@pytest.mark.parametrize(
    "invalid_field",
    ["plan", "projection"],
)
def test_builder_requires_bare_plan_and_projection_hashes(invalid_field: str) -> None:
    source = _raw_batch()
    plan_hash = _POLICY_REPLAY_PLAN_SHA256
    projection_hash = _SOURCE_PREDICTION_PROJECTION_SHA256
    if invalid_field == "plan":
        plan_hash = "sha256:not-bare"
    else:
        projection_hash = "not-a-sha256"
    with pytest.raises(SecFilingGemmaPolicyReplayError, match="lowercase SHA-256"):
        build_owned_development_policy_replay_batch(
            source_prediction_batch=source,
            expected_source_prediction_batch_sha256=source[
                "prediction_batch_sha256"
            ],
            expected_development_policy_replay_plan_sha256=plan_hash,
            expected_source_prediction_projection_sha256=projection_hash,
        )


def test_policy_replay_never_fits_predicts_opens_files_or_uses_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _raw_batch()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("forbidden effect")

    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "fit", forbidden)
    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "from_state", forbidden)
    monkeypatch.setattr(SecFilingGemmaTwoHeadLearner, "predict_components", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    result = _build(source)
    assert result["numeric_prediction_authorized"] is False
    assert result["model_transport_authorized"] is False
    assert result["network_access_authorized"] is False
    assert result["scoring_authorized"] is False
    source_text = inspect.getsource(
        __import__(
            "agent_benchmark.sec_filing_gemma_policy_replay", fromlist=["*"]
        )
    )
    for forbidden_import in (
        "sec_filing_gemma_features",
        "sec_filing_gemma_scoring",
        "urllib",
        "requests",
    ):
        assert forbidden_import not in source_text
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    assert "2019-01-02" not in encoded
