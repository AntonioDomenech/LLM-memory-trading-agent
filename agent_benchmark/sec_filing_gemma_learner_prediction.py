"""Pure numeric development OOF prediction for the SEC/Gemma experiment.

This module receives only compact causal feature rows and the ten already-fitted
development OOF learner states.  It never fits, selects a candidate, evaluates
thresholds, changes a policy state, scores, seals, reads outcomes, or performs
filesystem/model/network I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import copy
import hmac
import math
import re
import time
from typing import Any, Final

import numpy as np

from .sec_filing_gemma_contract import (
    DEVELOPMENT_FOLD_SPECS,
    SecFilingGemmaContractError,
    build_contract_manifest,
    canonical_sha256,
)
from .sec_filing_gemma_features import (
    ABLATION_FEATURE_COLUMNS,
    FILING_CALENDAR_FEATURE_COLUMNS,
    FEATURE_ROW_SCHEMA_VERSION,
    FEATURE_UNAVAILABLE_REASONS,
    MARKET_FEATURE_ROW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_FEATURE_BATCH_SCHEMA_VERSION,
    SEMANTIC_AGGREGATE_FEATURE_COLUMNS,
    SEMANTIC_FEATURE_COLUMNS,
)
from .direct_edge_features import MARKET_SENTIMENT_FEATURE_COLUMNS
from .downside_features import PRICE_FEATURE_COLUMNS
from .sec_filing_gemma_learner import (
    MODEL_TYPE as LEARNER_MODEL_TYPE,
    STATE_SCHEMA_VERSION as LEARNER_STATE_SCHEMA_VERSION,
    SecFilingGemmaLearnerError,
    SecFilingGemmaTwoHeadLearner,
)
from .sec_filing_gemma_learner_fit import (
    DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION,
    DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION,
)
from .sec_filing_gemma_stage_authorization import (
    SecFilingGemmaStageAuthorizationError,
    validate_development_oof_prediction_plan,
)


OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-oof-prediction-projection-v1"
)
DEVELOPMENT_OOF_PREDICTION_FEATURE_INPUT_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-prediction-feature-input-row-v1"
)
DEVELOPMENT_OOF_PREDICTION_FEATURE_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-prediction-feature-batch-v1"
)
DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-prediction-fold-model-v1"
)
DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-prediction-fold-model-bundle-v1"
)
DEVELOPMENT_OOF_RAW_PREDICTION_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-raw-prediction-row-v1"
)
OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-oof-prediction-batch-v1"
)

_AUTHORIZED_FOLD_IDS: Final[tuple[str, ...]] = tuple(
    item[0] for item in DEVELOPMENT_FOLD_SPECS
)
_VARIANT_IDS: Final[tuple[str, str]] = ("semantic", "ablation")
_PREDICTION_ORDER_RULE: Final[str] = (
    "source_feature_event_order_filtered_to_five_frozen_development_oof_windows"
)
_MAXIMUM_PREDICTION_SECONDS: Final[int] = 60
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_EVENT_BINDING_KEYS: Final[frozenset[str]] = frozenset(
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
_FEATURE_INPUT_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "prediction_ordinal",
        "source_event_ordinal",
        "source_feature_row_sha256",
        "event_binding",
        "event_binding_sha256",
        "source_unavailable_reasons",
        "source_unavailable_reasons_sha256",
        "prediction_available",
        "unavailable_reason",
        "semantic_feature_schema_sha256",
        "semantic_feature_values_hex",
        "semantic_feature_values_sha256",
        "ablation_feature_schema_sha256",
        "ablation_feature_values_hex",
        "ablation_feature_values_sha256",
        "prediction_feature_input_sha256",
    }
)
_FEATURE_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact_stage",
        "source_feature_batch_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "source_event_count",
        "prediction_event_count",
        "prediction_fold_count",
        "prediction_fold_ids",
        "feature_names",
        "feature_schema_sha256",
        "prediction_order_rule",
        "prediction_feature_rows",
        "prediction_feature_rows_sha256",
        "labels_included",
        "outcomes_included",
        "training_membership_included",
        "learner_states_included",
        "prediction_components_included",
        "candidate_selection_authorized",
        "threshold_action_authorized",
        "policy_transition_authorized",
        "scoring_authorized",
        "sealing_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
        "prediction_feature_batch_sha256",
    }
)
_PREDICTION_FOLD_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fold_train_cutoff_session",
        "training_set_count",
        "training_positive_count",
        "training_set_membership_sha256",
        "semantic_training_feature_matrix_sha256",
        "ablation_training_feature_matrix_sha256",
        "training_binary_target_sha256",
        "training_edge_target_sha256",
        "training_set_max_label_maturity_session",
        "semantic_fold_state_sha256",
        "ablation_fold_state_sha256",
    }
)
_FOLD_MODEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "fold_ordinal",
        "fold_id",
        "prediction_window_first_date",
        "prediction_window_last_date",
        "source_learner_fit_view_sha256",
        "prediction_fold_context",
        "prediction_fold_context_sha256",
        "semantic_fit_record_sha256",
        "semantic_learner_state",
        "semantic_learner_state_sha256",
        "ablation_fit_record_sha256",
        "ablation_learner_state",
        "ablation_learner_state_sha256",
        "feature_names",
        "feature_schema_sha256",
        "prediction_fold_model_sha256",
    }
)
_FOLD_MODEL_BUNDLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact_stage",
        "source_learner_fit_batch_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "fold_model_count",
        "fold_ids",
        "model_variant_count",
        "model_variant_ids",
        "learner_model_type",
        "learner_state_schema_version",
        "feature_names",
        "feature_schema_sha256",
        "fold_models",
        "fold_models_sha256",
        "deferred_training_view_state_included",
        "training_membership_rows_included",
        "training_feature_matrices_included",
        "training_target_vectors_included",
        "prediction_components_included",
        "candidate_selection_authorized",
        "threshold_action_authorized",
        "policy_transition_authorized",
        "scoring_authorized",
        "sealing_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
        "prediction_fold_model_bundle_sha256",
    }
)
_INPUT_SPEC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prediction_ordinal",
        "source_event_ordinal",
        "fold_ordinal",
        "fold_id",
        "decision_session",
        "accession_number",
        "source_feature_row_sha256",
        "event_binding_sha256",
        "prediction_available",
        "unavailable_reason",
        "semantic_feature_schema_sha256",
        "semantic_feature_values_sha256",
        "ablation_feature_schema_sha256",
        "ablation_feature_values_sha256",
        "prediction_feature_input_sha256",
        "prediction_input_spec_sha256",
    }
)
_FOLD_MODEL_SPEC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fold_ordinal",
        "fold_id",
        "prediction_window_first_date",
        "prediction_window_last_date",
        "source_learner_fit_view_sha256",
        "prediction_fold_context_sha256",
        "semantic_fit_record_sha256",
        "semantic_learner_state_sha256",
        "ablation_fit_record_sha256",
        "ablation_learner_state_sha256",
        "feature_schema_sha256",
        "prediction_fold_model_sha256",
        "prediction_fold_model_spec_sha256",
    }
)
_RAW_PREDICTION_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "prediction_ordinal",
        "parent_raw_prediction_row_sha256",
        "source_prediction_feature_input_sha256",
        "source_feature_row_sha256",
        "event_binding",
        "event_binding_sha256",
        "fold_id",
        "prediction_fold_context",
        "prediction_fold_context_sha256",
        "semantic_learner_state_sha256",
        "ablation_learner_state_sha256",
        "prediction_status",
        "unavailable_reason",
        "semantic_cash_probability_hex",
        "semantic_expected_edge_hex",
        "ablation_cash_probability_hex",
        "ablation_expected_edge_hex",
        "raw_prediction_row_sha256",
    }
)
_PREDICTION_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "artifact_stage",
        "development_root_scope_sha256",
        "development_oof_prediction_plan_sha256",
        "source_feature_batch_sha256",
        "prediction_feature_batch_sha256",
        "source_learner_fit_batch_sha256",
        "prediction_fold_model_bundle_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "prediction_event_count",
        "available_prediction_count",
        "unavailable_prediction_count",
        "prediction_call_count",
        "prediction_fold_count",
        "prediction_fold_ids",
        "model_variant_count",
        "model_variant_ids",
        "learner_model_type",
        "learner_state_schema_version",
        "feature_schema_sha256",
        "maximum_prediction_calls",
        "maximum_prediction_seconds",
        "prediction_order_rule",
        "raw_prediction_rows",
        "raw_prediction_rows_sha256",
        "raw_prediction_tip_sha256",
        "source_feature_rows_included",
        "learner_states_included",
        "training_membership_rows_included",
        "training_feature_matrices_included",
        "training_target_vectors_included",
        "labels_included",
        "outcomes_included",
        "post_2018_data_included",
        "prediction_components_included",
        "candidate_selection_authorized",
        "threshold_action_authorized",
        "policy_transition_authorized",
        "scoring_authorized",
        "sealing_authorized",
        "label_release_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
        "prediction_batch_sha256",
    }
)

_AVAILABLE_PREDICTION_STATUS: Final[str] = "available_pre_label"
_UNAVAILABLE_PREDICTION_STATUS: Final[str] = "unavailable_pre_label"
_RAW_PREDICTION_CHAIN_DOMAIN: Final[str] = (
    "aapl-sec-gemma-development-oof-raw-prediction-chain-genesis-v1"
)
_PREDICTION_COMPONENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "cash_win_probability_10bps",
        "expected_active_log_edge_10bps",
    }
)

_SOURCE_FEATURE_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "development_root_scope_sha256",
        "feature_assembly_plan_sha256", "candidate_sha256",
        "corpus_universe_sha256", "development_sec_reader_receipt_sha256",
        "development_market_reader_receipt_sha256",
        "development_model_reader_receipt_sha256", "event_count",
        "event_plan_sha256", "feature_row_schema_version",
        "feature_row_sha256s", "feature_rows_sha256", "feature_rows",
        "labels_included", "outcomes_included",
        "post_decision_market_rows_included", "training_membership_included",
        "learner_fit_authorized", "stage_promotion_authorized",
        "production_authorized", "feature_batch_sha256",
    }
)
_SOURCE_FEATURE_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "accession_number", "form", "artifact_stage",
        "decision_session", "extraction_status",
        "extraction_evidence_authenticated", "document_quality",
        "semantic_available", "market_available", "prediction_available",
        "fit_eligible", "unavailable_reasons", "missing_market_observations",
        "missing_market_observations_sha256", "bindings", "bindings_sha256",
        "market_feature_row_sha256", "price_regime_features_hex",
        "market_sentiment_features_hex", "filing_calendar_features_hex",
        "semantic_aggregate_features_hex", "semantic_feature_names",
        "semantic_feature_schema_sha256", "semantic_feature_values_hex",
        "ablation_feature_names", "ablation_feature_schema_sha256",
        "ablation_feature_values_hex", "feature_row_sha256",
    }
)
_SOURCE_FEATURE_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "contract_sha256", "corpus_universe_sha256",
        "universe_event_proof_sha256", "current_universe_record_sha256",
        "prior_same_form_universe_record_sha256", "current_source_record_sha256",
        "prior_same_form_source_record_sha256", "current_filing_sha256",
        "prior_same_form_filing_sha256", "extraction_event_proof_sha256",
        "extraction_identity_sha256", "extraction_evidence_sha256",
        "extraction_output_sha256", "extraction_output_canonical_sha256",
        "market_prefix_sha256", "market_prefix_proof_sha256",
        "market_stage_manifest_sha256", "source_manifest_sha256",
        "market_prefix_chain_identity_sha256",
    }
)
_SOURCE_FIT_VIEW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "source_training_view_ordinal", "training_view_id",
        "source_training_view_sha256", "training_row_count",
        "training_positive_count", "learner_input_context",
        "learner_input_context_sha256", "semantic_fit_record_sha256",
        "ablation_fit_record_sha256", "semantic_learner_state_sha256",
        "ablation_learner_state_sha256", "prediction_fold_context",
        "prediction_fold_context_sha256", "beta_1_1_climatology_hex",
        "learner_fit_view_sha256",
    }
)
_SOURCE_FIT_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "fit_ordinal", "source_training_view_ordinal",
        "training_view_id", "head_variant", "source_training_view_sha256",
        "training_input_identity", "training_input_identity_sha256",
        "fit_metadata", "fit_metadata_sha256", "learner_state",
        "learner_state_sha256", "learner_parameters_sha256",
        "learner_fit_record_sha256",
    }
)
_SOURCE_FIT_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version", "artifact_stage", "development_root_scope_sha256",
        "development_oof_learner_fit_plan_sha256",
        "source_training_membership_assembly_plan_sha256",
        "source_training_membership_projection_sha256",
        "source_training_membership_batch_sha256", "contract_sha256",
        "candidate_sha256", "corpus_universe_sha256",
        "calendar_sessions_sha256", "development_cutoff_session",
        "source_training_view_count", "source_training_view_ids",
        "authorized_training_view_count", "authorized_training_view_ids",
        "deferred_training_view_count", "deferred_training_views",
        "deferred_training_views_sha256", "model_variant_count",
        "model_variant_ids", "learner_fit_count", "learner_state_count",
        "learner_model_type", "learner_state_schema_version",
        "learner_config_sha256", "fit_order_rule", "maximum_fit_seconds",
        "learner_fit_views", "learner_fit_views_sha256",
        "learner_fit_records", "learner_fit_records_sha256",
        "learner_states_sha256", "source_training_membership_batch_included",
        "training_membership_rows_included", "training_feature_matrices_included",
        "training_target_vectors_included", "deferred_training_view_state_included",
        "learner_states_included", "compact_fit_audit_included",
        "prediction_included", "prediction_authorized",
        "candidate_selection_authorized", "threshold_action_authorized",
        "holdout_access_authorized", "ledger_mutation_authorized",
        "stage_promotion_authorized", "production_authorized",
        "learner_fit_batch_sha256",
    }
)


def _expect_dict(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} must be an exact built-in object"
        )
    return value


def _expect_list(value: Any, location: str) -> list[Any]:
    if type(value) is not list:
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} must be an exact built-in list"
        )
    return value


def _expect_keys(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    if set(value) != set(expected):
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} keys changed; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _detached(value: Any, location: str, *, depth: int = 0) -> Any:
    if depth > 64:
        raise SecFilingGemmaLearnerPredictionError(f"{location} is too deeply nested")
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is list:
        return [
            _detached(item, f"{location}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise SecFilingGemmaLearnerPredictionError(
                f"{location} keys must be exact strings"
            )
        return {
            key: _detached(item, f"{location}.{key}", depth=depth + 1)
            for key, item in value.items()
        }
    raise SecFilingGemmaLearnerPredictionError(
        f"{location} contains a non-JSON or subclass value"
    )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} must be an exact integer >= {minimum}"
        )
    return value


def _iso_date(value: Any, location: str) -> str:
    if type(value) is not str:
        raise SecFilingGemmaLearnerPredictionError(f"{location} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} must be an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise SecFilingGemmaLearnerPredictionError(f"{location} is not canonical")
    return value


def _float_hex(value: Any, location: str, *, probability: bool = False) -> float:
    if type(value) is not str:
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} must be canonical float.hex text"
        )
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} must be canonical float.hex text"
        ) from exc
    if (
        not math.isfinite(number)
        or number.hex() != value
        or (probability and not 0.0 <= number <= 1.0)
    ):
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} is not canonical finite float.hex text"
        )
    return number


def _self_hash(value: Mapping[str, Any], field: str, location: str) -> str:
    observed = _sha256(value.get(field), f"{location}.{field}")
    body = {key: value[key] for key in value if key != field}
    expected = canonical_sha256(body)
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaLearnerPredictionError(f"{location} checksum changed")
    return observed


def _fold_for_session(session: str) -> tuple[int, str, str, str]:
    parsed = _iso_date(session, "prediction decision session")
    for ordinal, (fold_id, _cutoff, first, last) in enumerate(
        DEVELOPMENT_FOLD_SPECS, start=1
    ):
        if first <= parsed <= last:
            return ordinal, fold_id, first, last
    raise SecFilingGemmaLearnerPredictionError(
        "Prediction event is outside the five frozen development OOF windows"
    )


def _maybe_fold_for_session(session: str) -> tuple[int, str, str, str] | None:
    for ordinal, (fold_id, _cutoff, first, last) in enumerate(
        DEVELOPMENT_FOLD_SPECS, start=1
    ):
        if first <= session <= last:
            return ordinal, fold_id, first, last
    return None


def _mapped_unavailable_reason(reasons: list[str]) -> str | None:
    market = "missing_required_market_history" in reasons
    extraction = "missing_or_unauthenticated_extraction_evidence" in reasons
    if market and extraction:
        return "missing_required_market_and_extraction_features"
    if market:
        return "missing_required_market_features"
    if extraction:
        return "missing_required_extraction_features"
    return None


def _feature_hex_mapping(
    raw: Any,
    *,
    names: tuple[str, ...],
    location: str,
) -> dict[str, str | None]:
    value = _expect_dict(raw, location)
    if list(value) != list(names):
        raise SecFilingGemmaLearnerPredictionError(
            f"{location} names or order changed"
        )
    result: dict[str, str | None] = {}
    for name in names:
        item = value[name]
        if item is None:
            result[name] = None
        else:
            _float_hex(item, f"{location}.{name}")
            result[name] = item
    return result


class SecFilingGemmaLearnerPredictionError(SecFilingGemmaContractError):
    """Raised when numeric OOF prediction crosses its frozen boundary."""


def derive_development_oof_prediction_feature_batch(
    source_feature_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Project all OOF-window filing features without consulting outcomes."""

    source = _expect_dict(
        _detached(source_feature_batch, "source feature batch"),
        "source feature batch",
    )
    _expect_keys(source, _SOURCE_FEATURE_BATCH_KEYS, "source feature batch")
    _self_hash(source, "feature_batch_sha256", "source feature batch")
    rows = _expect_list(source["feature_rows"], "source feature rows")
    row_hashes = _expect_list(
        source["feature_row_sha256s"], "source feature-row hashes"
    )
    event_count = _strict_int(source["event_count"], "source event count", minimum=1)
    expected_source_flags = {
        "labels_included": False,
        "outcomes_included": False,
        "post_decision_market_rows_included": False,
        "training_membership_included": False,
        "learner_fit_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    if (
        source["schema_version"]
        != OWNED_DEVELOPMENT_FEATURE_BATCH_SCHEMA_VERSION
        or source["feature_row_schema_version"] != FEATURE_ROW_SCHEMA_VERSION
        or event_count != len(rows)
        or len(row_hashes) != len(rows)
        or source["feature_rows_sha256"] != canonical_sha256(rows)
        or any(
            type(source[field]) is not bool or source[field] is not expected
            for field, expected in expected_source_flags.items()
        )
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Source feature batch identity or capability boundary changed"
        )
    for field in (
        "feature_batch_sha256",
        "feature_assembly_plan_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "development_sec_reader_receipt_sha256",
        "development_market_reader_receipt_sha256",
        "development_model_reader_receipt_sha256",
        "event_plan_sha256",
        "feature_rows_sha256",
    ):
        _sha256(source[field], f"source feature batch.{field}")

    feature_names = list(SEMANTIC_FEATURE_COLUMNS)
    if feature_names != list(ABLATION_FEATURE_COLUMNS):
        raise SecFilingGemmaLearnerPredictionError(
            "Semantic and ablation prediction schemas diverged"
        )
    feature_schema_hash = canonical_sha256(feature_names)
    compact_rows: list[dict[str, Any]] = []
    prior_key: tuple[str, str] | None = None
    accessions: set[str] = set()
    prediction_sessions: set[str] = set()

    for source_ordinal, raw_row in enumerate(rows, start=1):
        row = _expect_dict(raw_row, f"source feature row {source_ordinal}")
        _expect_keys(row, _SOURCE_FEATURE_ROW_KEYS, f"source feature row {source_ordinal}")
        row_hash = _self_hash(
            row, "feature_row_sha256", f"source feature row {source_ordinal}"
        )
        if row_hashes[source_ordinal - 1] != row_hash:
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature-row hash order changed"
            )
        accession = row["accession_number"]
        form = row["form"]
        session = _iso_date(row["decision_session"], "source decision session")
        if (
            row["schema_version"] != FEATURE_ROW_SCHEMA_VERSION
            or row["artifact_stage"] != "development"
            or type(accession) is not str
            or not accession
            or form not in {"10-K", "10-Q"}
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature event identity changed"
            )
        current_key = (session, accession)
        if prior_key is not None and current_key <= prior_key:
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature events are duplicated or reordered"
            )
        if accession in accessions:
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature events duplicate an accession"
            )
        prior_key = current_key
        accessions.add(accession)

        bindings = _expect_dict(
            row["bindings"], f"source feature row {source_ordinal} bindings"
        )
        _expect_keys(
            bindings,
            _SOURCE_FEATURE_BINDING_KEYS,
            f"source feature row {source_ordinal} bindings",
        )
        if row["bindings_sha256"] != canonical_sha256(bindings):
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature bindings checksum changed"
            )
        for field in (
            "contract_sha256",
            "corpus_universe_sha256",
            "extraction_identity_sha256",
            "market_prefix_chain_identity_sha256",
        ):
            _sha256(bindings[field], f"source feature bindings.{field}")
        _sha256(row["market_feature_row_sha256"], "source market feature-row hash")
        if (
            bindings["contract_sha256"] != canonical_sha256(build_contract_manifest())
            or bindings["corpus_universe_sha256"] != source["corpus_universe_sha256"]
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature row crossed the contract or corpus universe"
            )

        reasons = _expect_list(
            row["unavailable_reasons"],
            f"source feature row {source_ordinal} unavailable reasons",
        )
        if (
            any(type(item) is not str or item not in FEATURE_UNAVAILABLE_REASONS for item in reasons)
            or len(reasons) != len(set(reasons))
            or type(row["prediction_available"]) is not bool
            or type(row["fit_eligible"]) is not bool
            or row["prediction_available"] is not (not reasons)
            or row["fit_eligible"] is not row["prediction_available"]
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature availability boundary changed"
            )
        mapped_reason = _mapped_unavailable_reason(reasons)
        if (mapped_reason is None) is not row["prediction_available"]:
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature unavailable reasons are incomplete"
            )

        missing_market = _expect_list(
            row["missing_market_observations"],
            f"source feature row {source_ordinal} missing market observations",
        )
        if (
            any(type(item) is not str or not item for item in missing_market)
            or len(missing_market) != len(set(missing_market))
            or row["missing_market_observations_sha256"]
            != canonical_sha256(missing_market)
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source missing-market evidence changed"
            )
        price_features = _feature_hex_mapping(
            row["price_regime_features_hex"],
            names=PRICE_FEATURE_COLUMNS,
            location=f"source feature row {source_ordinal} price features",
        )
        sentiment_features = _feature_hex_mapping(
            row["market_sentiment_features_hex"],
            names=MARKET_SENTIMENT_FEATURE_COLUMNS,
            location=f"source feature row {source_ordinal} sentiment features",
        )
        calendar_features = _feature_hex_mapping(
            row["filing_calendar_features_hex"],
            names=FILING_CALENDAR_FEATURE_COLUMNS,
            location=f"source feature row {source_ordinal} calendar features",
        )
        semantic_features = _feature_hex_mapping(
            row["semantic_aggregate_features_hex"],
            names=SEMANTIC_AGGREGATE_FEATURE_COLUMNS,
            location=f"source feature row {source_ordinal} semantic features",
        )
        expected_market_available = not missing_market and all(
            item is not None
            for item in (*price_features.values(), *sentiment_features.values())
        )
        market_feature_body = {
            "schema_version": MARKET_FEATURE_ROW_SCHEMA_VERSION,
            "accession_number": accession,
            "decision_session": session,
            "market_prefix_chain_identity_sha256": bindings[
                "market_prefix_chain_identity_sha256"
            ],
            "market_available": row["market_available"],
            "missing_market_observations": missing_market,
            "missing_market_observations_sha256": row[
                "missing_market_observations_sha256"
            ],
            "price_regime_features_hex": price_features,
            "market_sentiment_features_hex": sentiment_features,
        }
        if (
            type(row["market_available"]) is not bool
            or row["market_available"] is not expected_market_available
            or row["market_feature_row_sha256"]
            != canonical_sha256(market_feature_body)
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source market feature availability or row checksum changed"
            )
        if type(row["extraction_evidence_authenticated"]) is not bool:
            raise SecFilingGemmaLearnerPredictionError(
                "Source extraction authentication flag is not an exact boolean"
            )
        expected_reasons: list[str] = []
        if not expected_market_available:
            expected_reasons.append("missing_required_market_history")
        if not row["extraction_evidence_authenticated"]:
            expected_reasons.append(
                "missing_or_unauthenticated_extraction_evidence"
            )
        if reasons != expected_reasons:
            raise SecFilingGemmaLearnerPredictionError(
                "Source feature unavailable reasons do not match causal evidence"
            )

        semantic_names = _expect_list(
            row["semantic_feature_names"], "source semantic feature names"
        )
        ablation_names = _expect_list(
            row["ablation_feature_names"], "source ablation feature names"
        )
        if (
            semantic_names != feature_names
            or ablation_names != feature_names
            or row["semantic_feature_schema_sha256"] != feature_schema_hash
            or row["ablation_feature_schema_sha256"] != feature_schema_hash
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source prediction feature schema changed"
            )
        semantic_values = row["semantic_feature_values_hex"]
        ablation_values = row["ablation_feature_values_hex"]
        if row["prediction_available"]:
            semantic_values = _expect_list(semantic_values, "source semantic vector")
            ablation_values = _expect_list(ablation_values, "source ablation vector")
            expected_semantic_values = [
                *(price_features[name] for name in PRICE_FEATURE_COLUMNS),
                *(
                    sentiment_features[name]
                    for name in MARKET_SENTIMENT_FEATURE_COLUMNS
                ),
                *(calendar_features[name] for name in FILING_CALENDAR_FEATURE_COLUMNS),
                *(
                    semantic_features[name]
                    for name in SEMANTIC_AGGREGATE_FEATURE_COLUMNS
                ),
            ]
            expected_ablation_values = [
                *(price_features[name] for name in PRICE_FEATURE_COLUMNS),
                *(
                    sentiment_features[name]
                    for name in MARKET_SENTIMENT_FEATURE_COLUMNS
                ),
                *(calendar_features[name] for name in FILING_CALENDAR_FEATURE_COLUMNS),
                *(
                    float(0.0).hex()
                    for _ in SEMANTIC_AGGREGATE_FEATURE_COLUMNS
                ),
            ]
            if len(semantic_values) != len(feature_names) or len(ablation_values) != len(
                feature_names
            ) or semantic_values != expected_semantic_values or ablation_values != expected_ablation_values:
                raise SecFilingGemmaLearnerPredictionError(
                    "Source prediction feature vector or ablation invariant changed"
                )
            for variant, values in (
                ("semantic", semantic_values),
                ("ablation", ablation_values),
            ):
                for index, item in enumerate(values):
                    _float_hex(item, f"source {variant} feature {index}")
            semantic_values_hash: str | None = canonical_sha256(semantic_values)
            ablation_values_hash: str | None = canonical_sha256(ablation_values)
        else:
            if semantic_values is not None or ablation_values is not None:
                raise SecFilingGemmaLearnerPredictionError(
                    "Unavailable source event exposed a prediction vector"
                )
            semantic_values_hash = None
            ablation_values_hash = None

        fold = _maybe_fold_for_session(session)
        if fold is None:
            continue
        if session in prediction_sessions:
            raise SecFilingGemmaLearnerPredictionError(
                "Frozen development OOF prediction events duplicate a decision session"
            )
        prediction_sessions.add(session)
        _fold_ordinal, fold_id, _first, _last = fold
        event_binding = {
            "accession_number": accession,
            "form": form,
            "stage": "development",
            "decision_session": session,
            "extraction_identity_sha256": bindings["extraction_identity_sha256"],
            "market_prefix_chain_identity_sha256": bindings[
                "market_prefix_chain_identity_sha256"
            ],
            "market_feature_row_sha256": row["market_feature_row_sha256"],
            "fold_id": fold_id,
        }
        body = {
            "schema_version": (
                DEVELOPMENT_OOF_PREDICTION_FEATURE_INPUT_ROW_SCHEMA_VERSION
            ),
            "prediction_ordinal": len(compact_rows) + 1,
            "source_event_ordinal": source_ordinal,
            "source_feature_row_sha256": row_hash,
            "event_binding": event_binding,
            "event_binding_sha256": canonical_sha256(event_binding),
            "source_unavailable_reasons": copy.deepcopy(reasons),
            "source_unavailable_reasons_sha256": canonical_sha256(reasons),
            "prediction_available": row["prediction_available"],
            "unavailable_reason": mapped_reason,
            "semantic_feature_schema_sha256": feature_schema_hash,
            "semantic_feature_values_hex": copy.deepcopy(semantic_values),
            "semantic_feature_values_sha256": semantic_values_hash,
            "ablation_feature_schema_sha256": feature_schema_hash,
            "ablation_feature_values_hex": copy.deepcopy(ablation_values),
            "ablation_feature_values_sha256": ablation_values_hash,
        }
        compact_rows.append(
            {
                **body,
                "prediction_feature_input_sha256": canonical_sha256(body),
            }
        )

    if not compact_rows:
        raise SecFilingGemmaLearnerPredictionError(
            "Source feature batch has no frozen development OOF prediction events"
        )
    seen_folds = {row["event_binding"]["fold_id"] for row in compact_rows}
    if seen_folds != set(_AUTHORIZED_FOLD_IDS):
        raise SecFilingGemmaLearnerPredictionError(
            "Source feature batch does not cover all five frozen OOF folds"
        )
    body = {
        "schema_version": DEVELOPMENT_OOF_PREDICTION_FEATURE_BATCH_SCHEMA_VERSION,
        "artifact_stage": "development",
        "source_feature_batch_sha256": source["feature_batch_sha256"],
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "candidate_sha256": source["candidate_sha256"],
        "corpus_universe_sha256": source["corpus_universe_sha256"],
        "source_event_count": event_count,
        "prediction_event_count": len(compact_rows),
        "prediction_fold_count": len(_AUTHORIZED_FOLD_IDS),
        "prediction_fold_ids": list(_AUTHORIZED_FOLD_IDS),
        "feature_names": feature_names,
        "feature_schema_sha256": feature_schema_hash,
        "prediction_order_rule": _PREDICTION_ORDER_RULE,
        "prediction_feature_rows": compact_rows,
        "prediction_feature_rows_sha256": canonical_sha256(compact_rows),
        "labels_included": False,
        "outcomes_included": False,
        "training_membership_included": False,
        "learner_states_included": False,
        "prediction_components_included": False,
        "candidate_selection_authorized": False,
        "threshold_action_authorized": False,
        "policy_transition_authorized": False,
        "scoring_authorized": False,
        "sealing_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {
        **body,
        "prediction_feature_batch_sha256": canonical_sha256(body),
    }


def _validated_prediction_feature_batch(raw: Any) -> dict[str, Any]:
    batch = _expect_dict(
        _detached(raw, "prediction feature batch"), "prediction feature batch"
    )
    _expect_keys(batch, _FEATURE_BATCH_KEYS, "prediction feature batch")
    _self_hash(
        batch, "prediction_feature_batch_sha256", "prediction feature batch"
    )
    rows = _expect_list(batch["prediction_feature_rows"], "prediction feature rows")
    names = _expect_list(batch["feature_names"], "prediction feature names")
    count = _strict_int(
        batch["prediction_event_count"], "prediction event count", minimum=1
    )
    if (
        batch["schema_version"]
        != DEVELOPMENT_OOF_PREDICTION_FEATURE_BATCH_SCHEMA_VERSION
        or batch["artifact_stage"] != "development"
        or count != len(rows)
        or _strict_int(batch["source_event_count"], "source event count", minimum=count)
        < count
        or _strict_int(batch["prediction_fold_count"], "prediction fold count", minimum=1)
        != len(_AUTHORIZED_FOLD_IDS)
        or batch["prediction_fold_ids"] != list(_AUTHORIZED_FOLD_IDS)
        or names != list(SEMANTIC_FEATURE_COLUMNS)
        or batch["feature_schema_sha256"] != canonical_sha256(names)
        or batch["prediction_order_rule"] != _PREDICTION_ORDER_RULE
        or batch["prediction_feature_rows_sha256"] != canonical_sha256(rows)
        or batch["contract_sha256"] != canonical_sha256(build_contract_manifest())
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Prediction feature batch identity, order, or schema changed"
        )
    for field in (
        "source_feature_batch_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "feature_schema_sha256",
        "prediction_feature_rows_sha256",
    ):
        _sha256(batch[field], f"prediction feature batch.{field}")
    false_flags = (
        "labels_included",
        "outcomes_included",
        "training_membership_included",
        "learner_states_included",
        "prediction_components_included",
        "candidate_selection_authorized",
        "threshold_action_authorized",
        "policy_transition_authorized",
        "scoring_authorized",
        "sealing_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    )
    if any(type(batch[field]) is not bool or batch[field] is not False for field in false_flags):
        raise SecFilingGemmaLearnerPredictionError(
            "Prediction feature batch crossed its non-authorizing boundary"
        )

    previous_key: tuple[str, str] | None = None
    prior_source_ordinal = 0
    seen_folds: set[str] = set()
    decision_sessions: set[str] = set()
    for ordinal, raw_row in enumerate(rows, start=1):
        row = _expect_dict(raw_row, f"prediction feature row {ordinal}")
        _expect_keys(row, _FEATURE_INPUT_ROW_KEYS, f"prediction feature row {ordinal}")
        _self_hash(
            row,
            "prediction_feature_input_sha256",
            f"prediction feature row {ordinal}",
        )
        source_ordinal = _strict_int(
            row["source_event_ordinal"], "source event ordinal", minimum=1
        )
        event = _expect_dict(row["event_binding"], "prediction event binding")
        _expect_keys(event, _EVENT_BINDING_KEYS, "prediction event binding")
        fold_ordinal, fold_id, _first, _last = _fold_for_session(
            event["decision_session"]
        )
        del fold_ordinal
        current_key = (event["decision_session"], event["accession_number"])
        reasons = _expect_list(
            row["source_unavailable_reasons"], "prediction unavailable reasons"
        )
        if (
            row["schema_version"]
            != DEVELOPMENT_OOF_PREDICTION_FEATURE_INPUT_ROW_SCHEMA_VERSION
            or _strict_int(row["prediction_ordinal"], "prediction ordinal", minimum=1)
            != ordinal
            or source_ordinal <= prior_source_ordinal
            or event["stage"] != "development"
            or event["fold_id"] != fold_id
            or event["form"] not in {"10-K", "10-Q"}
            or type(event["accession_number"]) is not str
            or not event["accession_number"]
            or row["event_binding_sha256"] != canonical_sha256(event)
            or row["source_unavailable_reasons_sha256"] != canonical_sha256(reasons)
            or any(type(item) is not str or item not in FEATURE_UNAVAILABLE_REASONS for item in reasons)
            or len(reasons) != len(set(reasons))
            or type(row["prediction_available"]) is not bool
            or row["prediction_available"] is not (not reasons)
            or row["unavailable_reason"] != _mapped_unavailable_reason(reasons)
            or row["semantic_feature_schema_sha256"] != batch["feature_schema_sha256"]
            or row["ablation_feature_schema_sha256"] != batch["feature_schema_sha256"]
            or event["decision_session"] in decision_sessions
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Prediction feature row identity or availability changed"
            )
        if previous_key is not None and current_key <= previous_key:
            raise SecFilingGemmaLearnerPredictionError(
                "Prediction feature rows are duplicated or reordered"
            )
        for field in (
            "source_feature_row_sha256",
            "event_binding_sha256",
            "source_unavailable_reasons_sha256",
            "semantic_feature_schema_sha256",
            "ablation_feature_schema_sha256",
        ):
            _sha256(row[field], f"prediction feature row.{field}")
        for field in (
            "extraction_identity_sha256",
            "market_prefix_chain_identity_sha256",
            "market_feature_row_sha256",
        ):
            _sha256(event[field], f"prediction event binding.{field}")
        semantic = row["semantic_feature_values_hex"]
        ablation = row["ablation_feature_values_hex"]
        if row["prediction_available"]:
            semantic = _expect_list(semantic, "prediction semantic vector")
            ablation = _expect_list(ablation, "prediction ablation vector")
            if len(semantic) != len(names) or len(ablation) != len(names):
                raise SecFilingGemmaLearnerPredictionError(
                    "Prediction feature vector width changed"
                )
            nonsemantic_count = len(names) - len(SEMANTIC_AGGREGATE_FEATURE_COLUMNS)
            if (
                ablation[:nonsemantic_count] != semantic[:nonsemantic_count]
                or ablation[nonsemantic_count:]
                != [
                    float(0.0).hex()
                    for _ in SEMANTIC_AGGREGATE_FEATURE_COLUMNS
                ]
            ):
                raise SecFilingGemmaLearnerPredictionError(
                    "Prediction feature ablation invariant changed"
                )
            for variant, vector in (("semantic", semantic), ("ablation", ablation)):
                for index, item in enumerate(vector):
                    _float_hex(item, f"prediction {variant} feature {index}")
            if (
                row["semantic_feature_values_sha256"] != canonical_sha256(semantic)
                or row["ablation_feature_values_sha256"] != canonical_sha256(ablation)
            ):
                raise SecFilingGemmaLearnerPredictionError(
                    "Prediction feature vector checksum changed"
                )
            _sha256(
                row["semantic_feature_values_sha256"],
                "prediction semantic feature-values hash",
            )
            _sha256(
                row["ablation_feature_values_sha256"],
                "prediction ablation feature-values hash",
            )
        elif (
            semantic is not None
            or ablation is not None
            or row["semantic_feature_values_sha256"] is not None
            or row["ablation_feature_values_sha256"] is not None
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Unavailable prediction feature row exposed vectors or hashes"
            )
        previous_key = current_key
        prior_source_ordinal = source_ordinal
        decision_sessions.add(event["decision_session"])
        seen_folds.add(fold_id)
    if seen_folds != set(_AUTHORIZED_FOLD_IDS):
        raise SecFilingGemmaLearnerPredictionError(
            "Prediction feature rows do not cover exactly the five folds"
        )
    return batch


def derive_development_oof_prediction_fold_model_bundle(
    source_learner_fit_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Project exactly five immutable semantic/ablation OOF state pairs."""

    source = _expect_dict(
        _detached(source_learner_fit_batch, "source learner-fit batch"),
        "source learner-fit batch",
    )
    _expect_keys(source, _SOURCE_FIT_BATCH_KEYS, "source learner-fit batch")
    _self_hash(source, "learner_fit_batch_sha256", "source learner-fit batch")
    views = _expect_list(source["learner_fit_views"], "source learner-fit views")
    records = _expect_list(
        source["learner_fit_records"], "source learner-fit records"
    )
    source_view_ids = _expect_list(
        source["source_training_view_ids"], "source training-view ids"
    )
    authorized_ids = _expect_list(
        source["authorized_training_view_ids"], "authorized training-view ids"
    )
    variant_ids = _expect_list(source["model_variant_ids"], "model variant ids")
    expected_false = (
        "source_training_membership_batch_included",
        "training_membership_rows_included",
        "training_feature_matrices_included",
        "training_target_vectors_included",
        "deferred_training_view_state_included",
        "prediction_included",
        "prediction_authorized",
        "candidate_selection_authorized",
        "threshold_action_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    )
    expected_true = ("learner_states_included", "compact_fit_audit_included")
    if (
        source["schema_version"]
        != OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION
        or source["artifact_stage"] != "development"
        or source["contract_sha256"] != canonical_sha256(build_contract_manifest())
        or source_view_ids != [*_AUTHORIZED_FOLD_IDS, "intermediate_frozen_through_2018"]
        or authorized_ids != list(_AUTHORIZED_FOLD_IDS)
        or variant_ids != list(_VARIANT_IDS)
        or _strict_int(source["source_training_view_count"], "source view count", minimum=1)
        != 6
        or _strict_int(source["authorized_training_view_count"], "authorized view count", minimum=1)
        != 5
        or _strict_int(source["deferred_training_view_count"], "deferred view count", minimum=1)
        != 1
        or _strict_int(source["model_variant_count"], "variant count", minimum=1) != 2
        or _strict_int(source["learner_fit_count"], "fit count", minimum=1) != 10
        or _strict_int(source["learner_state_count"], "state count", minimum=1) != 10
        or source["learner_model_type"] != LEARNER_MODEL_TYPE
        or _strict_int(source["learner_state_schema_version"], "state schema", minimum=1)
        != LEARNER_STATE_SCHEMA_VERSION
        or len(views) != 5
        or len(records) != 10
        or source["learner_fit_views_sha256"] != canonical_sha256(views)
        or source["learner_fit_records_sha256"] != canonical_sha256(records)
        or source["learner_states_sha256"]
        != canonical_sha256([record.get("learner_state") for record in records])
        or any(type(source[field]) is not bool or source[field] is not False for field in expected_false)
        or any(type(source[field]) is not bool or source[field] is not True for field in expected_true)
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Source learner-fit batch identity or capability boundary changed"
        )
    for field in (
        "learner_fit_batch_sha256",
        "development_root_scope_sha256",
        "development_oof_learner_fit_plan_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "learner_config_sha256",
        "learner_fit_views_sha256",
        "learner_fit_records_sha256",
        "learner_states_sha256",
    ):
        _sha256(source[field], f"source learner-fit batch.{field}")
    _iso_date(source["development_cutoff_session"], "development cutoff")

    records_by_hash: dict[str, dict[str, Any]] = {}
    for ordinal, raw_record in enumerate(records, start=1):
        record = _expect_dict(raw_record, f"source learner-fit record {ordinal}")
        _expect_keys(
            record, _SOURCE_FIT_RECORD_KEYS, f"source learner-fit record {ordinal}"
        )
        record_hash = _self_hash(
            record,
            "learner_fit_record_sha256",
            f"source learner-fit record {ordinal}",
        )
        expected_fold = _AUTHORIZED_FOLD_IDS[(ordinal - 1) // 2]
        expected_variant = _VARIANT_IDS[(ordinal - 1) % 2]
        if (
            record["schema_version"]
            != DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION
            or _strict_int(record["fit_ordinal"], "fit ordinal", minimum=1) != ordinal
            or _strict_int(
                record["source_training_view_ordinal"],
                "fit source-view ordinal",
                minimum=1,
            )
            != (ordinal - 1) // 2 + 1
            or record["training_view_id"] != expected_fold
            or record["head_variant"] != expected_variant
            or record_hash in records_by_hash
            or record["training_input_identity_sha256"]
            != canonical_sha256(record["training_input_identity"])
            or record["fit_metadata_sha256"] != canonical_sha256(record["fit_metadata"])
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source learner-fit record order or identity changed"
            )
        state = _expect_dict(record["learner_state"], "source learner state")
        state_hash = _sha256(record["learner_state_sha256"], "learner state hash")
        try:
            restored = SecFilingGemmaTwoHeadLearner.from_state(state)
        except (TypeError, SecFilingGemmaLearnerError) as exc:
            raise SecFilingGemmaLearnerPredictionError(
                "Source learner state failed canonical replay"
            ) from exc
        metadata = _expect_dict(state["fit_metadata"], "learner fit metadata")
        if (
            restored.to_state() != state
            or state["state_sha256"] != state_hash
            or state["parameters_sha256"] != record["learner_parameters_sha256"]
            or metadata.get("fold_id") != expected_fold
            or metadata.get("head_variant") != expected_variant
            or metadata.get("candidate_sha256") != source["candidate_sha256"]
            or state["model_type"] != LEARNER_MODEL_TYPE
            or state["state_schema_version"] != LEARNER_STATE_SCHEMA_VERSION
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source learner state crossed its fit binding"
            )
        records_by_hash[record_hash] = record

    feature_names = list(SEMANTIC_FEATURE_COLUMNS)
    feature_schema_hash = canonical_sha256(feature_names)
    fold_models: list[dict[str, Any]] = []
    for ordinal, (raw_view, fold_spec) in enumerate(
        zip(views, DEVELOPMENT_FOLD_SPECS, strict=True), start=1
    ):
        fold_id, _cutoff, first, last = fold_spec
        view = _expect_dict(raw_view, f"source learner-fit view {ordinal}")
        _expect_keys(view, _SOURCE_FIT_VIEW_KEYS, f"source learner-fit view {ordinal}")
        view_hash = _self_hash(
            view, "learner_fit_view_sha256", f"source learner-fit view {ordinal}"
        )
        context = _expect_dict(
            view["prediction_fold_context"], "source prediction fold context"
        )
        _expect_keys(context, _PREDICTION_FOLD_CONTEXT_KEYS, "source prediction fold context")
        semantic_record_hash = _sha256(
            view["semantic_fit_record_sha256"], "semantic fit-record hash"
        )
        ablation_record_hash = _sha256(
            view["ablation_fit_record_sha256"], "ablation fit-record hash"
        )
        semantic_record = records_by_hash.get(semantic_record_hash)
        ablation_record = records_by_hash.get(ablation_record_hash)
        if semantic_record is None or ablation_record is None:
            raise SecFilingGemmaLearnerPredictionError(
                "Source learner-fit view references an unknown fit record"
            )
        semantic_state = _expect_dict(
            semantic_record["learner_state"], "semantic learner state"
        )
        ablation_state = _expect_dict(
            ablation_record["learner_state"], "ablation learner state"
        )
        semantic_metadata = _expect_dict(
            semantic_state["fit_metadata"], "semantic learner fit metadata"
        )
        ablation_metadata = _expect_dict(
            ablation_state["fit_metadata"], "ablation learner fit metadata"
        )
        train_count = _strict_int(
            context["training_set_count"], "fold training count", minimum=2
        )
        positives = _strict_int(
            context["training_positive_count"], "fold positive count", minimum=1
        )
        maximum_maturity = _iso_date(
            context["training_set_max_label_maturity_session"],
            "fold maximum maturity",
        )
        if (
            view["schema_version"] != DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION
            or _strict_int(
                view["source_training_view_ordinal"], "source view ordinal", minimum=1
            )
            != ordinal
            or view["training_view_id"] != fold_id
            or semantic_record["training_view_id"] != fold_id
            or semantic_record["head_variant"] != "semantic"
            or ablation_record["training_view_id"] != fold_id
            or ablation_record["head_variant"] != "ablation"
            or semantic_record["source_training_view_sha256"]
            != view["source_training_view_sha256"]
            or ablation_record["source_training_view_sha256"]
            != view["source_training_view_sha256"]
            or view["prediction_fold_context_sha256"] != canonical_sha256(context)
            or context["semantic_fold_state_sha256"]
            != view["semantic_learner_state_sha256"]
            or context["ablation_fold_state_sha256"]
            != view["ablation_learner_state_sha256"]
            or semantic_state["state_sha256"] != view["semantic_learner_state_sha256"]
            or ablation_state["state_sha256"] != view["ablation_learner_state_sha256"]
            or semantic_state["feature_names"] != feature_names
            or ablation_state["feature_names"] != feature_names
            or semantic_metadata.get("train_label_maturity_through")
            != fold_spec[1]
            or ablation_metadata.get("train_label_maturity_through")
            != fold_spec[1]
            or semantic_metadata.get("maximum_training_label_maturity_session")
            != maximum_maturity
            or ablation_metadata.get("maximum_training_label_maturity_session")
            != maximum_maturity
            or semantic_metadata.get("training_row_count") != train_count
            or ablation_metadata.get("training_row_count") != train_count
            or semantic_metadata.get("training_set_sha256")
            != context["training_set_membership_sha256"]
            or ablation_metadata.get("training_set_sha256")
            != context["training_set_membership_sha256"]
            or semantic_metadata.get("feature_schema_sha256")
            != feature_schema_hash
            or ablation_metadata.get("feature_schema_sha256")
            != feature_schema_hash
            or semantic_state.get("training_positive_count") != positives
            or ablation_state.get("training_positive_count") != positives
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source learner-fit view crossed its state or fold context"
            )
        for field in _PREDICTION_FOLD_CONTEXT_KEYS - {
            "fold_train_cutoff_session",
            "training_set_count",
            "training_positive_count",
            "training_set_max_label_maturity_session",
        }:
            _sha256(context[field], f"prediction fold context.{field}")
        if (
            positives >= train_count
            or context["fold_train_cutoff_session"] != fold_spec[1]
            or maximum_maturity > fold_spec[1]
            or maximum_maturity >= first
            or context["semantic_fold_state_sha256"]
            == context["ablation_fold_state_sha256"]
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Source prediction fold context crossed chronology or class support"
            )
        body = {
            "schema_version": DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_SCHEMA_VERSION,
            "fold_ordinal": ordinal,
            "fold_id": fold_id,
            "prediction_window_first_date": first,
            "prediction_window_last_date": last,
            "source_learner_fit_view_sha256": view_hash,
            "prediction_fold_context": copy.deepcopy(context),
            "prediction_fold_context_sha256": canonical_sha256(context),
            "semantic_fit_record_sha256": semantic_record_hash,
            "semantic_learner_state": copy.deepcopy(semantic_state),
            "semantic_learner_state_sha256": semantic_state["state_sha256"],
            "ablation_fit_record_sha256": ablation_record_hash,
            "ablation_learner_state": copy.deepcopy(ablation_state),
            "ablation_learner_state_sha256": ablation_state["state_sha256"],
            "feature_names": feature_names,
            "feature_schema_sha256": feature_schema_hash,
        }
        fold_models.append(
            {**body, "prediction_fold_model_sha256": canonical_sha256(body)}
        )

    body = {
        "schema_version": DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_BUNDLE_SCHEMA_VERSION,
        "artifact_stage": "development",
        "source_learner_fit_batch_sha256": source["learner_fit_batch_sha256"],
        "contract_sha256": source["contract_sha256"],
        "candidate_sha256": source["candidate_sha256"],
        "corpus_universe_sha256": source["corpus_universe_sha256"],
        "calendar_sessions_sha256": source["calendar_sessions_sha256"],
        "development_cutoff_session": source["development_cutoff_session"],
        "fold_model_count": len(fold_models),
        "fold_ids": list(_AUTHORIZED_FOLD_IDS),
        "model_variant_count": len(_VARIANT_IDS),
        "model_variant_ids": list(_VARIANT_IDS),
        "learner_model_type": LEARNER_MODEL_TYPE,
        "learner_state_schema_version": LEARNER_STATE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "feature_schema_sha256": feature_schema_hash,
        "fold_models": fold_models,
        "fold_models_sha256": canonical_sha256(fold_models),
        "deferred_training_view_state_included": False,
        "training_membership_rows_included": False,
        "training_feature_matrices_included": False,
        "training_target_vectors_included": False,
        "prediction_components_included": False,
        "candidate_selection_authorized": False,
        "threshold_action_authorized": False,
        "policy_transition_authorized": False,
        "scoring_authorized": False,
        "sealing_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {
        **body,
        "prediction_fold_model_bundle_sha256": canonical_sha256(body),
    }


def _validated_prediction_fold_model_bundle(raw: Any) -> dict[str, Any]:
    bundle = _expect_dict(
        _detached(raw, "prediction fold-model bundle"),
        "prediction fold-model bundle",
    )
    _expect_keys(bundle, _FOLD_MODEL_BUNDLE_KEYS, "prediction fold-model bundle")
    _self_hash(
        bundle,
        "prediction_fold_model_bundle_sha256",
        "prediction fold-model bundle",
    )
    models = _expect_list(bundle["fold_models"], "prediction fold models")
    names = _expect_list(bundle["feature_names"], "prediction model feature names")
    if (
        bundle["schema_version"]
        != DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_BUNDLE_SCHEMA_VERSION
        or bundle["artifact_stage"] != "development"
        or bundle["contract_sha256"] != canonical_sha256(build_contract_manifest())
        or _strict_int(bundle["fold_model_count"], "fold-model count", minimum=1)
        != len(_AUTHORIZED_FOLD_IDS)
        or len(models) != len(_AUTHORIZED_FOLD_IDS)
        or bundle["fold_ids"] != list(_AUTHORIZED_FOLD_IDS)
        or _strict_int(bundle["model_variant_count"], "variant count", minimum=1)
        != len(_VARIANT_IDS)
        or bundle["model_variant_ids"] != list(_VARIANT_IDS)
        or bundle["learner_model_type"] != LEARNER_MODEL_TYPE
        or _strict_int(bundle["learner_state_schema_version"], "state schema", minimum=1)
        != LEARNER_STATE_SCHEMA_VERSION
        or names != list(SEMANTIC_FEATURE_COLUMNS)
        or bundle["feature_schema_sha256"] != canonical_sha256(names)
        or bundle["fold_models_sha256"] != canonical_sha256(models)
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Prediction fold-model bundle identity or schema changed"
        )
    for field in (
        "source_learner_fit_batch_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "feature_schema_sha256",
        "fold_models_sha256",
    ):
        _sha256(bundle[field], f"prediction fold-model bundle.{field}")
    _iso_date(bundle["development_cutoff_session"], "development cutoff")
    false_flags = (
        "deferred_training_view_state_included",
        "training_membership_rows_included",
        "training_feature_matrices_included",
        "training_target_vectors_included",
        "prediction_components_included",
        "candidate_selection_authorized",
        "threshold_action_authorized",
        "policy_transition_authorized",
        "scoring_authorized",
        "sealing_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    )
    if any(type(bundle[field]) is not bool or bundle[field] is not False for field in false_flags):
        raise SecFilingGemmaLearnerPredictionError(
            "Prediction fold-model bundle crossed its non-authorizing boundary"
        )

    for ordinal, (raw_model, fold_spec) in enumerate(
        zip(models, DEVELOPMENT_FOLD_SPECS, strict=True), start=1
    ):
        fold_id, cutoff, first, last = fold_spec
        model = _expect_dict(raw_model, f"prediction fold model {ordinal}")
        _expect_keys(model, _FOLD_MODEL_KEYS, f"prediction fold model {ordinal}")
        _self_hash(
            model,
            "prediction_fold_model_sha256",
            f"prediction fold model {ordinal}",
        )
        context = _expect_dict(
            model["prediction_fold_context"], "prediction fold context"
        )
        _expect_keys(context, _PREDICTION_FOLD_CONTEXT_KEYS, "prediction fold context")
        semantic_state = _expect_dict(
            model["semantic_learner_state"], "semantic learner state"
        )
        ablation_state = _expect_dict(
            model["ablation_learner_state"], "ablation learner state"
        )
        try:
            semantic = SecFilingGemmaTwoHeadLearner.from_state(semantic_state)
            ablation = SecFilingGemmaTwoHeadLearner.from_state(ablation_state)
        except (TypeError, SecFilingGemmaLearnerError) as exc:
            raise SecFilingGemmaLearnerPredictionError(
                "Prediction fold model contains a noncanonical learner state"
            ) from exc
        semantic_metadata = _expect_dict(
            semantic_state["fit_metadata"], "semantic fit metadata"
        )
        ablation_metadata = _expect_dict(
            ablation_state["fit_metadata"], "ablation fit metadata"
        )
        train_count = _strict_int(
            context["training_set_count"], "fold training count", minimum=2
        )
        positive_count = _strict_int(
            context["training_positive_count"], "fold positive count", minimum=1
        )
        maximum_maturity = _iso_date(
            context["training_set_max_label_maturity_session"],
            "fold maximum maturity",
        )
        if (
            model["schema_version"] != DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_SCHEMA_VERSION
            or _strict_int(model["fold_ordinal"], "fold ordinal", minimum=1)
            != ordinal
            or model["fold_id"] != fold_id
            or model["prediction_window_first_date"] != first
            or model["prediction_window_last_date"] != last
            or model["feature_names"] != names
            or model["feature_schema_sha256"] != bundle["feature_schema_sha256"]
            or model["prediction_fold_context_sha256"] != canonical_sha256(context)
            or context["fold_train_cutoff_session"] != cutoff
            or context["semantic_fold_state_sha256"]
            != model["semantic_learner_state_sha256"]
            or context["ablation_fold_state_sha256"]
            != model["ablation_learner_state_sha256"]
            or semantic_state["state_sha256"]
            != model["semantic_learner_state_sha256"]
            or ablation_state["state_sha256"]
            != model["ablation_learner_state_sha256"]
            or semantic.to_state() != semantic_state
            or ablation.to_state() != ablation_state
            or semantic_state["feature_names"] != names
            or ablation_state["feature_names"] != names
            or semantic_metadata.get("fold_id") != fold_id
            or semantic_metadata.get("head_variant") != "semantic"
            or semantic_metadata.get("candidate_sha256") != bundle["candidate_sha256"]
            or ablation_metadata.get("fold_id") != fold_id
            or ablation_metadata.get("head_variant") != "ablation"
            or ablation_metadata.get("candidate_sha256") != bundle["candidate_sha256"]
            or semantic_metadata.get("train_label_maturity_through") != cutoff
            or ablation_metadata.get("train_label_maturity_through") != cutoff
            or semantic_metadata.get("maximum_training_label_maturity_session")
            != maximum_maturity
            or ablation_metadata.get("maximum_training_label_maturity_session")
            != maximum_maturity
            or semantic_metadata.get("training_row_count") != train_count
            or ablation_metadata.get("training_row_count") != train_count
            or semantic_metadata.get("training_set_sha256")
            != context["training_set_membership_sha256"]
            or ablation_metadata.get("training_set_sha256")
            != context["training_set_membership_sha256"]
            or semantic_metadata.get("feature_schema_sha256")
            != model["feature_schema_sha256"]
            or ablation_metadata.get("feature_schema_sha256")
            != model["feature_schema_sha256"]
            or semantic_state.get("training_positive_count") != positive_count
            or ablation_state.get("training_positive_count") != positive_count
            or model["semantic_learner_state_sha256"]
            == model["ablation_learner_state_sha256"]
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Prediction fold model crossed its exact state or chronology binding"
            )
        for field in (
            "source_learner_fit_view_sha256",
            "prediction_fold_context_sha256",
            "semantic_fit_record_sha256",
            "semantic_learner_state_sha256",
            "ablation_fit_record_sha256",
            "ablation_learner_state_sha256",
            "feature_schema_sha256",
        ):
            _sha256(model[field], f"prediction fold model.{field}")
        if positive_count >= train_count or maximum_maturity > cutoff or maximum_maturity >= first:
            raise SecFilingGemmaLearnerPredictionError(
                "Prediction fold model crossed its causal training boundary"
            )
        for field in _PREDICTION_FOLD_CONTEXT_KEYS - {
            "fold_train_cutoff_session",
            "training_set_count",
            "training_positive_count",
            "training_set_max_label_maturity_session",
        }:
            _sha256(context[field], f"prediction fold context.{field}")
    return bundle


def derive_development_oof_prediction_input_specs(
    prediction_feature_batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    batch = _validated_prediction_feature_batch(prediction_feature_batch)
    specs: list[dict[str, Any]] = []
    for row in batch["prediction_feature_rows"]:
        fold_ordinal, fold_id, _first, _last = _fold_for_session(
            row["event_binding"]["decision_session"]
        )
        body = {
            "prediction_ordinal": row["prediction_ordinal"],
            "source_event_ordinal": row["source_event_ordinal"],
            "fold_ordinal": fold_ordinal,
            "fold_id": fold_id,
            "decision_session": row["event_binding"]["decision_session"],
            "accession_number": row["event_binding"]["accession_number"],
            "source_feature_row_sha256": row["source_feature_row_sha256"],
            "event_binding_sha256": row["event_binding_sha256"],
            "prediction_available": row["prediction_available"],
            "unavailable_reason": row["unavailable_reason"],
            "semantic_feature_schema_sha256": row[
                "semantic_feature_schema_sha256"
            ],
            "semantic_feature_values_sha256": row[
                "semantic_feature_values_sha256"
            ],
            "ablation_feature_schema_sha256": row[
                "ablation_feature_schema_sha256"
            ],
            "ablation_feature_values_sha256": row[
                "ablation_feature_values_sha256"
            ],
            "prediction_feature_input_sha256": row[
                "prediction_feature_input_sha256"
            ],
        }
        specs.append({**body, "prediction_input_spec_sha256": canonical_sha256(body)})
    return specs


def derive_development_oof_prediction_fold_model_specs(
    prediction_fold_model_bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    bundle = _validated_prediction_fold_model_bundle(prediction_fold_model_bundle)
    specs: list[dict[str, Any]] = []
    for model in bundle["fold_models"]:
        body = {
            "fold_ordinal": model["fold_ordinal"],
            "fold_id": model["fold_id"],
            "prediction_window_first_date": model["prediction_window_first_date"],
            "prediction_window_last_date": model["prediction_window_last_date"],
            "source_learner_fit_view_sha256": model[
                "source_learner_fit_view_sha256"
            ],
            "prediction_fold_context_sha256": model[
                "prediction_fold_context_sha256"
            ],
            "semantic_fit_record_sha256": model["semantic_fit_record_sha256"],
            "semantic_learner_state_sha256": model[
                "semantic_learner_state_sha256"
            ],
            "ablation_fit_record_sha256": model["ablation_fit_record_sha256"],
            "ablation_learner_state_sha256": model[
                "ablation_learner_state_sha256"
            ],
            "feature_schema_sha256": model["feature_schema_sha256"],
            "prediction_fold_model_sha256": model[
                "prediction_fold_model_sha256"
            ],
        }
        specs.append(
            {
                **body,
                "prediction_fold_model_spec_sha256": canonical_sha256(body),
            }
        )
    return specs


def _validated_plan_and_inputs(
    *,
    development_oof_prediction_plan: Mapping[str, Any],
    expected_development_oof_prediction_plan_sha256: str,
    prediction_feature_batch: Mapping[str, Any],
    expected_prediction_feature_batch_sha256: str,
    prediction_fold_model_bundle: Mapping[str, Any],
    expected_prediction_fold_model_bundle_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_plan_hash = _sha256(
        expected_development_oof_prediction_plan_sha256,
        "expected development OOF prediction plan hash",
    )
    plan = _expect_dict(
        _detached(
            development_oof_prediction_plan,
            "development OOF prediction plan",
        ),
        "development OOF prediction plan",
    )
    try:
        observed_plan_hash = validate_development_oof_prediction_plan(
            plan,
            expected_development_oof_prediction_plan_sha256=expected_plan_hash,
        )
    except (SecFilingGemmaStageAuthorizationError, TypeError, ValueError) as exc:
        raise SecFilingGemmaLearnerPredictionError(
            "Development OOF numeric prediction was not exactly authorized"
        ) from exc
    if not hmac.compare_digest(observed_plan_hash, expected_plan_hash):
        raise SecFilingGemmaLearnerPredictionError(
            "Development OOF prediction authority changed"
        )

    feature_batch = _validated_prediction_feature_batch(prediction_feature_batch)
    expected_feature_hash = _sha256(
        expected_prediction_feature_batch_sha256,
        "expected prediction feature-batch hash",
    )
    if not hmac.compare_digest(
        feature_batch["prediction_feature_batch_sha256"],
        expected_feature_hash,
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Prediction feature batch is not externally pinned"
        )

    fold_bundle = _validated_prediction_fold_model_bundle(
        prediction_fold_model_bundle
    )
    expected_fold_bundle_hash = _sha256(
        expected_prediction_fold_model_bundle_sha256,
        "expected prediction fold-model bundle hash",
    )
    if not hmac.compare_digest(
        fold_bundle["prediction_fold_model_bundle_sha256"],
        expected_fold_bundle_hash,
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Prediction fold-model bundle is not externally pinned"
        )

    input_specs = derive_development_oof_prediction_input_specs(feature_batch)
    fold_specs = derive_development_oof_prediction_fold_model_specs(fold_bundle)
    plan_input_specs = _expect_list(
        plan.get("prediction_input_specs"),
        "authorized prediction input specs",
    )
    plan_fold_specs = _expect_list(
        plan.get("prediction_fold_model_specs"),
        "authorized prediction fold-model specs",
    )
    for ordinal, raw_spec in enumerate(plan_input_specs, start=1):
        spec = _expect_dict(raw_spec, f"authorized prediction input spec {ordinal}")
        _expect_keys(spec, _INPUT_SPEC_KEYS, f"authorized prediction input spec {ordinal}")
        _self_hash(
            spec,
            "prediction_input_spec_sha256",
            f"authorized prediction input spec {ordinal}",
        )
    for ordinal, raw_spec in enumerate(plan_fold_specs, start=1):
        spec = _expect_dict(
            raw_spec,
            f"authorized prediction fold-model spec {ordinal}",
        )
        _expect_keys(
            spec,
            _FOLD_MODEL_SPEC_KEYS,
            f"authorized prediction fold-model spec {ordinal}",
        )
        _self_hash(
            spec,
            "prediction_fold_model_spec_sha256",
            f"authorized prediction fold-model spec {ordinal}",
        )

    available_count = sum(
        1 for row in feature_batch["prediction_feature_rows"]
        if row["prediction_available"]
    )
    unavailable_count = feature_batch["prediction_event_count"] - available_count
    plan_fold_ids = _expect_list(
        plan.get("authorized_fold_ids"), "authorized prediction fold ids"
    )
    plan_variant_ids = _expect_list(
        plan.get("model_variant_ids"), "authorized prediction model variants"
    )
    cross_bound_hash_fields = {
        "contract_sha256": feature_batch["contract_sha256"],
        "candidate_sha256": feature_batch["candidate_sha256"],
        "corpus_universe_sha256": feature_batch["corpus_universe_sha256"],
        "source_feature_batch_sha256": feature_batch[
            "source_feature_batch_sha256"
        ],
        "prediction_feature_batch_sha256": feature_batch[
            "prediction_feature_batch_sha256"
        ],
        "source_development_oof_learner_fit_batch_sha256": fold_bundle[
            "source_learner_fit_batch_sha256"
        ],
        "prediction_fold_model_bundle_sha256": fold_bundle[
            "prediction_fold_model_bundle_sha256"
        ],
        "calendar_sessions_sha256": fold_bundle["calendar_sessions_sha256"],
        "feature_schema_sha256": feature_batch["feature_schema_sha256"],
    }
    for field, expected in cross_bound_hash_fields.items():
        observed = _sha256(plan.get(field), f"development OOF prediction {field}")
        if not hmac.compare_digest(observed, expected):
            raise SecFilingGemmaLearnerPredictionError(
                f"Development OOF prediction plan does not bind {field}"
            )
    for field in ("contract_sha256", "candidate_sha256", "corpus_universe_sha256"):
        if fold_bundle[field] != feature_batch[field]:
            raise SecFilingGemmaLearnerPredictionError(
                "Prediction feature and fold-model identities diverged"
            )

    development_cutoff = _iso_date(
        plan.get("development_cutoff_session"),
        "authorized development cutoff",
    )
    integer_bindings = {
        "source_event_count": feature_batch["source_event_count"],
        "authorized_fold_count": len(_AUTHORIZED_FOLD_IDS),
        "prediction_fold_model_count": len(_AUTHORIZED_FOLD_IDS),
        "model_variant_count": len(_VARIANT_IDS),
        "learner_state_count": 2 * len(_AUTHORIZED_FOLD_IDS),
        "learner_state_schema_version": LEARNER_STATE_SCHEMA_VERSION,
        "prediction_input_count": feature_batch["prediction_event_count"],
        "available_prediction_input_count": available_count,
        "unavailable_prediction_input_count": unavailable_count,
        "maximum_prediction_calls": 2 * available_count,
        "maximum_prediction_seconds": _MAXIMUM_PREDICTION_SECONDS,
    }
    for field, expected in integer_bindings.items():
        if _strict_int(
            plan.get(field), f"development OOF prediction {field}"
        ) != expected:
            raise SecFilingGemmaLearnerPredictionError(
                "Development OOF prediction count or runtime authority changed"
            )
    if (
        plan_input_specs != input_specs
        or plan.get("prediction_input_specs_sha256")
        != canonical_sha256(input_specs)
        or plan_fold_specs != fold_specs
        or plan.get("prediction_fold_model_specs_sha256")
        != canonical_sha256(fold_specs)
        or plan_fold_ids != list(_AUTHORIZED_FOLD_IDS)
        or plan_variant_ids != list(_VARIANT_IDS)
        or plan.get("learner_model_type") != LEARNER_MODEL_TYPE
        or development_cutoff != fold_bundle["development_cutoff_session"]
        or fold_bundle["feature_schema_sha256"]
        != feature_batch["feature_schema_sha256"]
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Development OOF prediction specs, model identity, or chronology changed"
        )
    for field in (
        "prediction_input_specs_sha256",
        "prediction_fold_model_specs_sha256",
        "development_root_scope_sha256",
    ):
        _sha256(plan.get(field), f"development OOF prediction {field}")
    return plan, feature_batch, fold_bundle


def _prediction_chain_genesis(
    *,
    development_root_scope_sha256: str,
    contract_sha256: str,
    candidate_sha256: str,
    corpus_universe_sha256: str,
    calendar_sessions_sha256: str,
    development_cutoff_session: str,
) -> str:
    return canonical_sha256(
        {
            "chain_domain": _RAW_PREDICTION_CHAIN_DOMAIN,
            "development_root_scope_sha256": development_root_scope_sha256,
            "contract_sha256": contract_sha256,
            "candidate_sha256": candidate_sha256,
            "corpus_universe_sha256": corpus_universe_sha256,
            "calendar_sessions_sha256": calendar_sessions_sha256,
            "development_cutoff_session": development_cutoff_session,
        }
    )


def _predict_variant_once(
    *,
    learner_state: dict[str, Any],
    feature_values_hex: list[str],
    variant: str,
) -> tuple[str, str]:
    before_state = copy.deepcopy(learner_state)
    feature_values = [
        _float_hex(value, f"{variant} prediction feature {index}")
        for index, value in enumerate(feature_values_hex)
    ]
    try:
        learner = SecFilingGemmaTwoHeadLearner.from_state(before_state)
        if learner.to_state() != before_state:
            raise SecFilingGemmaLearnerPredictionError(
                f"{variant} learner changed during state reconstruction"
            )
        edge_lower = learner.config.edge_clip_lower
        edge_upper = learner.config.edge_clip_upper
        result = learner.predict_components([feature_values])
        after_state = learner.to_state()
    except SecFilingGemmaLearnerPredictionError:
        raise
    except (SecFilingGemmaLearnerError, TypeError, ValueError, OverflowError) as exc:
        raise SecFilingGemmaLearnerPredictionError(
            f"{variant} numeric prediction failed"
        ) from exc
    if after_state != before_state:
        raise SecFilingGemmaLearnerPredictionError(
            f"{variant} learner mutated during numeric prediction"
        )
    result = _expect_dict(result, f"{variant} prediction components")
    _expect_keys(result, _PREDICTION_COMPONENT_KEYS, f"{variant} prediction components")
    probability_values = result["cash_win_probability_10bps"]
    edge_values = result["expected_active_log_edge_10bps"]
    if (
        type(probability_values) is not np.ndarray
        or probability_values.shape != (1,)
        or not np.issubdtype(probability_values.dtype, np.floating)
        or type(edge_values) is not np.ndarray
        or edge_values.shape != (1,)
        or not np.issubdtype(edge_values.dtype, np.floating)
    ):
        raise SecFilingGemmaLearnerPredictionError(
            f"{variant} prediction components must be exact one-value float arrays"
        )
    probability = float(probability_values[0])
    edge = float(edge_values[0])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise SecFilingGemmaLearnerPredictionError(
            f"{variant} cash probability is invalid"
        )
    if not math.isfinite(edge) or not edge_lower <= edge <= edge_upper:
        raise SecFilingGemmaLearnerPredictionError(
            f"{variant} expected edge is outside the fitted learner bounds"
        )
    return probability.hex(), edge.hex()


def _predict_one_event(
    feature_input: dict[str, Any],
    fold_model: dict[str, Any],
    *,
    parent_raw_prediction_row_sha256: str,
    development_root_scope_sha256: str,
    contract_sha256: str,
    candidate_sha256: str,
    corpus_universe_sha256: str,
    calendar_sessions_sha256: str,
    development_cutoff_session: str,
) -> tuple[dict[str, Any], int]:
    """Run arithmetic for one current event without any future row or state."""

    for value, location in (
        (parent_raw_prediction_row_sha256, "parent raw-prediction row hash"),
        (development_root_scope_sha256, "development root-scope hash"),
        (contract_sha256, "prediction contract hash"),
        (candidate_sha256, "prediction candidate hash"),
        (corpus_universe_sha256, "prediction corpus-universe hash"),
        (calendar_sessions_sha256, "prediction calendar hash"),
    ):
        _sha256(value, location)
    _iso_date(development_cutoff_session, "prediction development cutoff")
    if feature_input["event_binding"]["fold_id"] != fold_model["fold_id"]:
        raise SecFilingGemmaLearnerPredictionError(
            "Current prediction event was paired with the wrong fold state"
        )

    if feature_input["prediction_available"]:
        semantic_probability, semantic_edge = _predict_variant_once(
            learner_state=fold_model["semantic_learner_state"],
            feature_values_hex=feature_input["semantic_feature_values_hex"],
            variant="semantic",
        )
        ablation_probability, ablation_edge = _predict_variant_once(
            learner_state=fold_model["ablation_learner_state"],
            feature_values_hex=feature_input["ablation_feature_values_hex"],
            variant="ablation",
        )
        status = _AVAILABLE_PREDICTION_STATUS
        unavailable_reason = None
        call_count = 2
    else:
        semantic_probability = None
        semantic_edge = None
        ablation_probability = None
        ablation_edge = None
        status = _UNAVAILABLE_PREDICTION_STATUS
        unavailable_reason = feature_input["unavailable_reason"]
        call_count = 0

    body = {
        "schema_version": DEVELOPMENT_OOF_RAW_PREDICTION_ROW_SCHEMA_VERSION,
        "prediction_ordinal": feature_input["prediction_ordinal"],
        "parent_raw_prediction_row_sha256": parent_raw_prediction_row_sha256,
        "source_prediction_feature_input_sha256": feature_input[
            "prediction_feature_input_sha256"
        ],
        "source_feature_row_sha256": feature_input["source_feature_row_sha256"],
        "event_binding": copy.deepcopy(feature_input["event_binding"]),
        "event_binding_sha256": feature_input["event_binding_sha256"],
        "fold_id": fold_model["fold_id"],
        "prediction_fold_context": copy.deepcopy(
            fold_model["prediction_fold_context"]
        ),
        "prediction_fold_context_sha256": fold_model[
            "prediction_fold_context_sha256"
        ],
        "semantic_learner_state_sha256": fold_model[
            "semantic_learner_state_sha256"
        ],
        "ablation_learner_state_sha256": fold_model[
            "ablation_learner_state_sha256"
        ],
        "prediction_status": status,
        "unavailable_reason": unavailable_reason,
        "semantic_cash_probability_hex": semantic_probability,
        "semantic_expected_edge_hex": semantic_edge,
        "ablation_cash_probability_hex": ablation_probability,
        "ablation_expected_edge_hex": ablation_edge,
    }
    return {**body, "raw_prediction_row_sha256": canonical_sha256(body)}, call_count


def build_owned_development_oof_prediction_batch(
    *,
    development_oof_prediction_plan: Mapping[str, Any],
    expected_development_oof_prediction_plan_sha256: str,
    prediction_feature_batch: Mapping[str, Any],
    expected_prediction_feature_batch_sha256: str,
    prediction_fold_model_bundle: Mapping[str, Any],
    expected_prediction_fold_model_bundle_sha256: str,
) -> dict[str, Any]:
    """Emit only deterministic numeric OOF components from frozen states."""

    plan, feature_batch, fold_bundle = _validated_plan_and_inputs(
        development_oof_prediction_plan=development_oof_prediction_plan,
        expected_development_oof_prediction_plan_sha256=(
            expected_development_oof_prediction_plan_sha256
        ),
        prediction_feature_batch=prediction_feature_batch,
        expected_prediction_feature_batch_sha256=(
            expected_prediction_feature_batch_sha256
        ),
        prediction_fold_model_bundle=prediction_fold_model_bundle,
        expected_prediction_fold_model_bundle_sha256=(
            expected_prediction_fold_model_bundle_sha256
        ),
    )
    fold_models = {
        model["fold_id"]: model for model in fold_bundle["fold_models"]
    }
    parent_hash = _prediction_chain_genesis(
        development_root_scope_sha256=plan["development_root_scope_sha256"],
        contract_sha256=feature_batch["contract_sha256"],
        candidate_sha256=feature_batch["candidate_sha256"],
        corpus_universe_sha256=feature_batch["corpus_universe_sha256"],
        calendar_sessions_sha256=fold_bundle["calendar_sessions_sha256"],
        development_cutoff_session=fold_bundle["development_cutoff_session"],
    )
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    prediction_call_count = 0
    for feature_input in feature_batch["prediction_feature_rows"]:
        fold_id = feature_input["event_binding"]["fold_id"]
        model = fold_models.get(fold_id)
        if model is None:
            raise SecFilingGemmaLearnerPredictionError(
                "Prediction event has no authorized fold-model pair"
            )
        row, calls = _predict_one_event(
            feature_input,
            model,
            parent_raw_prediction_row_sha256=parent_hash,
            development_root_scope_sha256=plan[
                "development_root_scope_sha256"
            ],
            contract_sha256=feature_batch["contract_sha256"],
            candidate_sha256=feature_batch["candidate_sha256"],
            corpus_universe_sha256=feature_batch["corpus_universe_sha256"],
            calendar_sessions_sha256=fold_bundle["calendar_sessions_sha256"],
            development_cutoff_session=fold_bundle[
                "development_cutoff_session"
            ],
        )
        rows.append(row)
        prediction_call_count += calls
        parent_hash = row["raw_prediction_row_sha256"]
        if time.monotonic() - started > _MAXIMUM_PREDICTION_SECONDS:
            raise SecFilingGemmaLearnerPredictionError(
                "Development OOF numeric predictions exceeded the frozen 60-second cap"
            )

    available_count = sum(
        row["prediction_status"] == _AVAILABLE_PREDICTION_STATUS for row in rows
    )
    unavailable_count = len(rows) - available_count
    if (
        prediction_call_count != 2 * available_count
        or prediction_call_count != plan["maximum_prediction_calls"]
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Development OOF prediction call count changed"
        )
    body = {
        "schema_version": OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION,
        "artifact_stage": "development",
        "development_root_scope_sha256": plan[
            "development_root_scope_sha256"
        ],
        "development_oof_prediction_plan_sha256": plan[
            "development_oof_prediction_plan_sha256"
        ],
        "source_feature_batch_sha256": feature_batch[
            "source_feature_batch_sha256"
        ],
        "prediction_feature_batch_sha256": feature_batch[
            "prediction_feature_batch_sha256"
        ],
        "source_learner_fit_batch_sha256": fold_bundle[
            "source_learner_fit_batch_sha256"
        ],
        "prediction_fold_model_bundle_sha256": fold_bundle[
            "prediction_fold_model_bundle_sha256"
        ],
        "contract_sha256": feature_batch["contract_sha256"],
        "candidate_sha256": feature_batch["candidate_sha256"],
        "corpus_universe_sha256": feature_batch["corpus_universe_sha256"],
        "calendar_sessions_sha256": fold_bundle["calendar_sessions_sha256"],
        "development_cutoff_session": fold_bundle[
            "development_cutoff_session"
        ],
        "prediction_event_count": len(rows),
        "available_prediction_count": available_count,
        "unavailable_prediction_count": unavailable_count,
        "prediction_call_count": prediction_call_count,
        "prediction_fold_count": len(_AUTHORIZED_FOLD_IDS),
        "prediction_fold_ids": list(_AUTHORIZED_FOLD_IDS),
        "model_variant_count": len(_VARIANT_IDS),
        "model_variant_ids": list(_VARIANT_IDS),
        "learner_model_type": LEARNER_MODEL_TYPE,
        "learner_state_schema_version": LEARNER_STATE_SCHEMA_VERSION,
        "feature_schema_sha256": feature_batch["feature_schema_sha256"],
        "maximum_prediction_calls": plan["maximum_prediction_calls"],
        "maximum_prediction_seconds": _MAXIMUM_PREDICTION_SECONDS,
        "prediction_order_rule": _PREDICTION_ORDER_RULE,
        "raw_prediction_rows": rows,
        "raw_prediction_rows_sha256": canonical_sha256(rows),
        "raw_prediction_tip_sha256": parent_hash,
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
    }
    return {**body, "prediction_batch_sha256": canonical_sha256(body)}


def _validated_prediction_batch_structure(raw: Any) -> dict[str, Any]:
    batch = _expect_dict(
        _detached(raw, "owned development OOF prediction batch"),
        "owned development OOF prediction batch",
    )
    _expect_keys(
        batch,
        _PREDICTION_BATCH_KEYS,
        "owned development OOF prediction batch",
    )
    _self_hash(
        batch,
        "prediction_batch_sha256",
        "owned development OOF prediction batch",
    )
    rows = _expect_list(batch["raw_prediction_rows"], "raw prediction rows")
    fold_ids = _expect_list(batch["prediction_fold_ids"], "prediction fold ids")
    variant_ids = _expect_list(batch["model_variant_ids"], "prediction variants")
    event_count = _strict_int(
        batch["prediction_event_count"], "prediction event count", minimum=1
    )
    available_count = _strict_int(
        batch["available_prediction_count"],
        "available prediction count",
    )
    unavailable_count = _strict_int(
        batch["unavailable_prediction_count"],
        "unavailable prediction count",
    )
    call_count = _strict_int(
        batch["prediction_call_count"], "prediction call count"
    )
    maximum_calls = _strict_int(
        batch["maximum_prediction_calls"], "maximum prediction calls"
    )
    maximum_seconds = _strict_int(
        batch["maximum_prediction_seconds"],
        "maximum prediction seconds",
        minimum=1,
    )
    if (
        batch["schema_version"]
        != OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION
        or batch["artifact_stage"] != "development"
        or event_count != len(rows)
        or event_count != available_count + unavailable_count
        or call_count != 2 * available_count
        or maximum_calls != call_count
        or maximum_seconds != _MAXIMUM_PREDICTION_SECONDS
        or _strict_int(
            batch["prediction_fold_count"], "prediction fold count", minimum=1
        )
        != len(_AUTHORIZED_FOLD_IDS)
        or fold_ids != list(_AUTHORIZED_FOLD_IDS)
        or _strict_int(
            batch["model_variant_count"], "prediction variant count", minimum=1
        )
        != len(_VARIANT_IDS)
        or variant_ids != list(_VARIANT_IDS)
        or batch["learner_model_type"] != LEARNER_MODEL_TYPE
        or _strict_int(
            batch["learner_state_schema_version"],
            "prediction learner-state schema",
            minimum=1,
        )
        != LEARNER_STATE_SCHEMA_VERSION
        or batch["prediction_order_rule"] != _PREDICTION_ORDER_RULE
        or batch["raw_prediction_rows_sha256"] != canonical_sha256(rows)
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Owned prediction batch identity, count, or order changed"
        )
    for field in (
        "development_root_scope_sha256",
        "development_oof_prediction_plan_sha256",
        "source_feature_batch_sha256",
        "prediction_feature_batch_sha256",
        "source_learner_fit_batch_sha256",
        "prediction_fold_model_bundle_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "feature_schema_sha256",
        "raw_prediction_rows_sha256",
        "raw_prediction_tip_sha256",
    ):
        _sha256(batch[field], f"owned prediction batch.{field}")
    _iso_date(batch["development_cutoff_session"], "prediction development cutoff")
    false_flags = (
        "source_feature_rows_included",
        "learner_states_included",
        "training_membership_rows_included",
        "training_feature_matrices_included",
        "training_target_vectors_included",
        "labels_included",
        "outcomes_included",
        "post_2018_data_included",
        "candidate_selection_authorized",
        "threshold_action_authorized",
        "policy_transition_authorized",
        "scoring_authorized",
        "sealing_authorized",
        "label_release_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    )
    if (
        any(
            type(batch[field]) is not bool or batch[field] is not False
            for field in false_flags
        )
        or type(batch["prediction_components_included"]) is not bool
        or batch["prediction_components_included"] is not True
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Owned prediction batch crossed its numeric-only capability boundary"
        )

    expected_parent = _prediction_chain_genesis(
        development_root_scope_sha256=batch[
            "development_root_scope_sha256"
        ],
        contract_sha256=batch["contract_sha256"],
        candidate_sha256=batch["candidate_sha256"],
        corpus_universe_sha256=batch["corpus_universe_sha256"],
        calendar_sessions_sha256=batch["calendar_sessions_sha256"],
        development_cutoff_session=batch["development_cutoff_session"],
    )
    seen_accessions: set[str] = set()
    seen_sessions: set[str] = set()
    seen_folds: set[str] = set()
    observed_available = 0
    observed_unavailable = 0
    for ordinal, raw_row in enumerate(rows, start=1):
        row = _expect_dict(raw_row, f"raw prediction row {ordinal}")
        _expect_keys(row, _RAW_PREDICTION_ROW_KEYS, f"raw prediction row {ordinal}")
        row_hash = _self_hash(
            row,
            "raw_prediction_row_sha256",
            f"raw prediction row {ordinal}",
        )
        event = _expect_dict(row["event_binding"], "raw prediction event binding")
        _expect_keys(event, _EVENT_BINDING_KEYS, "raw prediction event binding")
        context = _expect_dict(
            row["prediction_fold_context"], "raw prediction fold context"
        )
        _expect_keys(context, _PREDICTION_FOLD_CONTEXT_KEYS, "raw prediction fold context")
        _fold_ordinal, expected_fold_id, _first, _last = _fold_for_session(
            event["decision_session"]
        )
        accession = event["accession_number"]
        session = event["decision_session"]
        if (
            row["schema_version"]
            != DEVELOPMENT_OOF_RAW_PREDICTION_ROW_SCHEMA_VERSION
            or _strict_int(row["prediction_ordinal"], "raw prediction ordinal", minimum=1)
            != ordinal
            or row["parent_raw_prediction_row_sha256"] != expected_parent
            or event["stage"] != "development"
            or event["fold_id"] != expected_fold_id
            or row["fold_id"] != expected_fold_id
            or event["form"] not in {"10-K", "10-Q"}
            or type(accession) is not str
            or not accession
            or accession in seen_accessions
            or session in seen_sessions
            or row["event_binding_sha256"] != canonical_sha256(event)
            or row["prediction_fold_context_sha256"] != canonical_sha256(context)
            or context["semantic_fold_state_sha256"]
            != row["semantic_learner_state_sha256"]
            or context["ablation_fold_state_sha256"]
            != row["ablation_learner_state_sha256"]
            or row["semantic_learner_state_sha256"]
            == row["ablation_learner_state_sha256"]
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Raw prediction row crossed its event, fold, or parent binding"
            )
        for field in (
            "parent_raw_prediction_row_sha256",
            "source_prediction_feature_input_sha256",
            "source_feature_row_sha256",
            "event_binding_sha256",
            "prediction_fold_context_sha256",
            "semantic_learner_state_sha256",
            "ablation_learner_state_sha256",
        ):
            _sha256(row[field], f"raw prediction row.{field}")
        for field in (
            "extraction_identity_sha256",
            "market_prefix_chain_identity_sha256",
            "market_feature_row_sha256",
        ):
            _sha256(event[field], f"raw prediction event.{field}")
        for field in _PREDICTION_FOLD_CONTEXT_KEYS - {
            "fold_train_cutoff_session",
            "training_set_count",
            "training_positive_count",
            "training_set_max_label_maturity_session",
        }:
            _sha256(context[field], f"raw prediction context.{field}")
        train_count = _strict_int(
            context["training_set_count"], "raw prediction training count", minimum=2
        )
        positive_count = _strict_int(
            context["training_positive_count"],
            "raw prediction positive count",
            minimum=1,
        )
        cutoff = _iso_date(
            context["fold_train_cutoff_session"], "raw prediction fold cutoff"
        )
        maximum_maturity = _iso_date(
            context["training_set_max_label_maturity_session"],
            "raw prediction maximum label maturity",
        )
        if (
            positive_count >= train_count
            or maximum_maturity > cutoff
            or cutoff >= session
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Raw prediction row crossed its causal fold chronology"
            )
        component_fields = (
            "semantic_cash_probability_hex",
            "semantic_expected_edge_hex",
            "ablation_cash_probability_hex",
            "ablation_expected_edge_hex",
        )
        if row["prediction_status"] == _AVAILABLE_PREDICTION_STATUS:
            if row["unavailable_reason"] is not None:
                raise SecFilingGemmaLearnerPredictionError(
                    "Available raw prediction has an unavailable reason"
                )
            _float_hex(row[component_fields[0]], component_fields[0], probability=True)
            _float_hex(row[component_fields[1]], component_fields[1])
            _float_hex(row[component_fields[2]], component_fields[2], probability=True)
            _float_hex(row[component_fields[3]], component_fields[3])
            observed_available += 1
        elif row["prediction_status"] == _UNAVAILABLE_PREDICTION_STATUS:
            if (
                row["unavailable_reason"]
                not in {
                    "missing_required_market_features",
                    "missing_required_extraction_features",
                    "missing_required_market_and_extraction_features",
                }
                or any(row[field] is not None for field in component_fields)
            ):
                raise SecFilingGemmaLearnerPredictionError(
                    "Unavailable raw prediction exposed numeric components"
                )
            observed_unavailable += 1
        else:
            raise SecFilingGemmaLearnerPredictionError(
                "Raw prediction status changed"
            )
        seen_accessions.add(accession)
        seen_sessions.add(session)
        seen_folds.add(expected_fold_id)
        expected_parent = row_hash
    if (
        observed_available != available_count
        or observed_unavailable != unavailable_count
        or seen_folds != set(_AUTHORIZED_FOLD_IDS)
        or batch["raw_prediction_tip_sha256"] != expected_parent
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Raw prediction population, fold coverage, or chain tip changed"
        )
    return batch


def validate_owned_development_oof_prediction_batch_structure(
    batch: Mapping[str, Any],
    *,
    expected_prediction_batch_sha256: str,
) -> str:
    """Validate and externally pin a raw OOF batch without predicting again.

    This deliberately narrower validator is for downstream deterministic
    consumers such as threshold/policy replay.  It checks the complete exact
    JSON shape, row chain, chronology, statuses, canonical numeric components,
    capability flags, and caller-supplied batch pin, but it never reconstructs
    a learner or invokes a prediction boundary.
    """

    value = _validated_prediction_batch_structure(batch)
    observed = value["prediction_batch_sha256"]
    expected = _sha256(
        expected_prediction_batch_sha256,
        "expected owned development OOF prediction-batch hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaLearnerPredictionError(
            "Development OOF prediction batch is not externally pinned"
        )
    return observed


def validate_owned_development_oof_prediction_batch(
    batch: Mapping[str, Any],
    *,
    development_oof_prediction_plan: Mapping[str, Any],
    expected_development_oof_prediction_plan_sha256: str,
    prediction_feature_batch: Mapping[str, Any],
    expected_prediction_feature_batch_sha256: str,
    prediction_fold_model_bundle: Mapping[str, Any],
    expected_prediction_fold_model_bundle_sha256: str,
    expected_prediction_batch_sha256: str,
) -> str:
    """Validate the pinned batch without consuming another prediction call."""

    value = _validated_prediction_batch_structure(batch)
    observed = value["prediction_batch_sha256"]
    expected = _sha256(
        expected_prediction_batch_sha256,
        "expected owned development OOF prediction-batch hash",
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaLearnerPredictionError(
            "Development OOF prediction batch is not externally pinned"
        )
    plan, feature_batch, fold_bundle = _validated_plan_and_inputs(
        development_oof_prediction_plan=development_oof_prediction_plan,
        expected_development_oof_prediction_plan_sha256=(
            expected_development_oof_prediction_plan_sha256
        ),
        prediction_feature_batch=prediction_feature_batch,
        expected_prediction_feature_batch_sha256=(
            expected_prediction_feature_batch_sha256
        ),
        prediction_fold_model_bundle=prediction_fold_model_bundle,
        expected_prediction_fold_model_bundle_sha256=(
            expected_prediction_fold_model_bundle_sha256
        ),
    )
    expected_top_level = {
        "development_root_scope_sha256": plan[
            "development_root_scope_sha256"
        ],
        "development_oof_prediction_plan_sha256": plan[
            "development_oof_prediction_plan_sha256"
        ],
        "source_feature_batch_sha256": feature_batch[
            "source_feature_batch_sha256"
        ],
        "prediction_feature_batch_sha256": feature_batch[
            "prediction_feature_batch_sha256"
        ],
        "source_learner_fit_batch_sha256": fold_bundle[
            "source_learner_fit_batch_sha256"
        ],
        "prediction_fold_model_bundle_sha256": fold_bundle[
            "prediction_fold_model_bundle_sha256"
        ],
        "contract_sha256": feature_batch["contract_sha256"],
        "candidate_sha256": feature_batch["candidate_sha256"],
        "corpus_universe_sha256": feature_batch["corpus_universe_sha256"],
        "calendar_sessions_sha256": fold_bundle["calendar_sessions_sha256"],
        "development_cutoff_session": fold_bundle[
            "development_cutoff_session"
        ],
        "prediction_event_count": feature_batch["prediction_event_count"],
        "prediction_fold_count": feature_batch["prediction_fold_count"],
        "prediction_fold_ids": feature_batch["prediction_fold_ids"],
        "model_variant_count": fold_bundle["model_variant_count"],
        "model_variant_ids": fold_bundle["model_variant_ids"],
        "learner_model_type": fold_bundle["learner_model_type"],
        "learner_state_schema_version": fold_bundle[
            "learner_state_schema_version"
        ],
        "feature_schema_sha256": feature_batch["feature_schema_sha256"],
        "maximum_prediction_calls": plan["maximum_prediction_calls"],
        "maximum_prediction_seconds": plan["maximum_prediction_seconds"],
    }
    if any(value[field] != expected for field, expected in expected_top_level.items()):
        raise SecFilingGemmaLearnerPredictionError(
            "Development OOF prediction batch crossed its authorized source binding"
        )
    model_by_fold = {
        model["fold_id"]: model for model in fold_bundle["fold_models"]
    }
    feature_rows = feature_batch["prediction_feature_rows"]
    output_rows = value["raw_prediction_rows"]
    if len(output_rows) != len(feature_rows):
        raise SecFilingGemmaLearnerPredictionError(
            "Raw prediction row count differs from the compact feature population"
        )
    expected_available = 0
    for output_row, feature_row in zip(
        output_rows,
        feature_rows,
        strict=True,
    ):
        model = model_by_fold[feature_row["event_binding"]["fold_id"]]
        expected_row_bindings = {
            "prediction_ordinal": feature_row["prediction_ordinal"],
            "source_prediction_feature_input_sha256": feature_row[
                "prediction_feature_input_sha256"
            ],
            "source_feature_row_sha256": feature_row[
                "source_feature_row_sha256"
            ],
            "event_binding": feature_row["event_binding"],
            "event_binding_sha256": feature_row["event_binding_sha256"],
            "fold_id": model["fold_id"],
            "prediction_fold_context": model["prediction_fold_context"],
            "prediction_fold_context_sha256": model[
                "prediction_fold_context_sha256"
            ],
            "semantic_learner_state_sha256": model[
                "semantic_learner_state_sha256"
            ],
            "ablation_learner_state_sha256": model[
                "ablation_learner_state_sha256"
            ],
            "unavailable_reason": feature_row["unavailable_reason"],
        }
        if any(
            output_row[field] != expected
            for field, expected in expected_row_bindings.items()
        ):
            raise SecFilingGemmaLearnerPredictionError(
                "Raw prediction row crossed its current feature or fold-state binding"
            )
        expected_status = (
            _AVAILABLE_PREDICTION_STATUS
            if feature_row["prediction_available"]
            else _UNAVAILABLE_PREDICTION_STATUS
        )
        if output_row["prediction_status"] != expected_status:
            raise SecFilingGemmaLearnerPredictionError(
                "Raw prediction availability differs from its compact feature input"
            )
        if feature_row["prediction_available"]:
            semantic_edge = _float_hex(
                output_row["semantic_expected_edge_hex"],
                "semantic expected edge",
            )
            ablation_edge = _float_hex(
                output_row["ablation_expected_edge_hex"],
                "ablation expected edge",
            )
            semantic_config = _expect_dict(
                model["semantic_learner_state"]["config"],
                "semantic learner config",
            )
            ablation_config = _expect_dict(
                model["ablation_learner_state"]["config"],
                "ablation learner config",
            )
            semantic_lower = _float_hex(
                semantic_config.get("edge_clip_lower"),
                "semantic edge lower bound",
            )
            semantic_upper = _float_hex(
                semantic_config.get("edge_clip_upper"),
                "semantic edge upper bound",
            )
            ablation_lower = _float_hex(
                ablation_config.get("edge_clip_lower"),
                "ablation edge lower bound",
            )
            ablation_upper = _float_hex(
                ablation_config.get("edge_clip_upper"),
                "ablation edge upper bound",
            )
            if (
                not semantic_lower <= semantic_edge <= semantic_upper
                or not ablation_lower <= ablation_edge <= ablation_upper
            ):
                raise SecFilingGemmaLearnerPredictionError(
                    "Raw expected edge exceeds its exact learner-state clip bounds"
                )
        expected_available += int(feature_row["prediction_available"])
    if (
        value["available_prediction_count"] != expected_available
        or value["unavailable_prediction_count"]
        != feature_batch["prediction_event_count"] - expected_available
        or value["prediction_call_count"] != 2 * expected_available
    ):
        raise SecFilingGemmaLearnerPredictionError(
            "Prediction output counts differ from the authorized input population"
        )
    return observed


__all__ = [
    "DEVELOPMENT_OOF_PREDICTION_FEATURE_BATCH_SCHEMA_VERSION",
    "DEVELOPMENT_OOF_PREDICTION_FEATURE_INPUT_ROW_SCHEMA_VERSION",
    "DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_BUNDLE_SCHEMA_VERSION",
    "DEVELOPMENT_OOF_PREDICTION_FOLD_MODEL_SCHEMA_VERSION",
    "DEVELOPMENT_OOF_RAW_PREDICTION_ROW_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_OOF_PREDICTION_BATCH_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION",
    "SecFilingGemmaLearnerPredictionError",
    "build_owned_development_oof_prediction_batch",
    "derive_development_oof_prediction_feature_batch",
    "derive_development_oof_prediction_fold_model_bundle",
    "derive_development_oof_prediction_fold_model_specs",
    "derive_development_oof_prediction_input_specs",
    "validate_owned_development_oof_prediction_batch_structure",
    "validate_owned_development_oof_prediction_batch",
]
