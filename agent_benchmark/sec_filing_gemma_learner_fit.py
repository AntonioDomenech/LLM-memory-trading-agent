"""Owned deterministic development OOF fits for the SEC/Gemma experiment.

This boundary accepts only the already-owned training-membership artifact and
the exact authorization plan derived from it.  It fits the five development
OOF views, in frozen order, for the semantic and ablation variants.  The
frozen-through-2018 refit is deliberately not performed here.  No function in
this module reads files, calls a model or network, predicts, selects a
candidate, changes a threshold, or mutates a trading ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from datetime import date
import hmac
import math
import re
import time
from typing import Any, Final

from .sec_filing_gemma_contract import (
    SecFilingGemmaContractError,
    build_contract_manifest,
    canonical_sha256,
)
from .sec_filing_gemma_learner import (
    MODEL_TYPE as LEARNER_MODEL_TYPE,
    STATE_SCHEMA_VERSION as LEARNER_STATE_SCHEMA_VERSION,
    SecFilingGemmaFitMetadata,
    SecFilingGemmaLearnerConfig,
    SecFilingGemmaLearnerError,
    SecFilingGemmaTwoHeadLearner,
    beta_1_1_climatology,
)
from .sec_filing_gemma_stage_authorization import (
    DEVELOPMENT_OOF_LEARNER_FIT_PLAN_SCHEMA_VERSION,
    SecFilingGemmaStageAuthorizationError,
    validate_development_oof_learner_fit_plan,
)
from .sec_filing_gemma_training_membership import (
    DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION,
)


OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-oof-learner-fit-projection-v1"
)
OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-oof-learner-fit-batch-v1"
)
DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-learner-fit-view-v1"
)
DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-oof-learner-fit-record-v1"
)

_AUTHORIZED_VIEW_IDS: Final[tuple[str, ...]] = (
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
)
_DEFERRED_VIEW_ID: Final[str] = "intermediate_frozen_through_2018"
_VARIANT_IDS: Final[tuple[str, ...]] = ("semantic", "ablation")
_FIT_ORDER_RULE: Final[str] = (
    "view_ordinal_ascending_then_semantic_then_ablation_exactly_once"
)
_MAXIMUM_FIT_SECONDS: Final[int] = 60
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

_VIEW_SPEC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "view_ordinal",
        "training_view_id",
        "view_kind",
        "training_source_stage",
        "prediction_stage",
        "train_label_maturity_through",
        "prediction_window_first_date",
        "prediction_window_last_date",
        "state_updates_inside_prediction_window",
    }
)
_LEARNER_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
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
    }
)
_PREDICTION_FOLD_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        *_LEARNER_CONTEXT_KEYS,
        "semantic_fold_state_sha256",
        "ablation_fold_state_sha256",
    }
)
_FIT_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "candidate_sha256",
        "head_variant",
        "fold_id",
        "train_label_maturity_through",
        "training_set_sha256",
        "training_row_count",
        "maximum_training_label_maturity_session",
        "feature_schema_sha256",
    }
)
_TRAINING_MEMBER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "training_row_ordinal",
        "source_event_ordinal",
        "accession_number",
        "form",
        "decision_session",
        "label_maturity_session",
        "training_event_identity_sha256",
    }
)
_TRAINING_VIEW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        *_VIEW_SPEC_KEYS,
        "training_set_count",
        "training_positive_count",
        "training_set_max_label_maturity_session",
        "training_set_membership",
        "training_set_membership_sha256",
        "semantic_training_features_hex",
        "semantic_training_feature_matrix_sha256",
        "ablation_training_features_hex",
        "ablation_training_feature_matrix_sha256",
        "training_binary_targets",
        "training_binary_target_sha256",
        "training_edge_targets_hex",
        "training_edge_target_sha256",
        "learner_input_context",
        "learner_input_context_sha256",
        "semantic_fit_metadata_template",
        "semantic_fit_metadata_template_sha256",
        "ablation_fit_metadata_template",
        "ablation_fit_metadata_template_sha256",
        "training_view_sha256",
    }
)
_MEMBERSHIP_BATCH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "development_root_scope_sha256",
        "training_membership_assembly_plan_sha256",
        "source_feature_assembly_plan_sha256",
        "source_label_assembly_plan_sha256",
        "source_feature_batch_sha256",
        "source_label_batch_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "development_cutoff_session",
        "event_count",
        "matured_label_count",
        "unmatured_event_count",
        "training_view_count",
        "training_view_ids",
        "membership_view_specs_sha256",
        "model_variant_specs_sha256",
        "feature_names",
        "feature_schema_sha256",
        "target_cost_bps",
        "binary_target_field",
        "binary_target_encoding",
        "edge_target_field",
        "binary_comparison_tolerance_hex",
        "membership_maturity_rule",
        "feature_eligibility_rule",
        "membership_order_rule",
        "shared_variant_support_rule",
        "event_audit_rows",
        "event_audit_rows_sha256",
        "training_views",
        "training_views_sha256",
        "development_labels_included",
        "development_outcomes_included",
        "training_membership_included",
        "learner_input_matrices_included",
        "learner_targets_included",
        "compact_adjusted_open_paths_included",
        "full_market_rows_included",
        "post_cutoff_market_data_included",
        "outcome_based_membership_filtering_permitted",
        "row_rebalancing_permitted",
        "learner_fit_authorized",
        "prediction_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
        "training_membership_batch_sha256",
    }
)
_FIT_INPUT_SPEC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "fit_ordinal",
        "source_training_view_ordinal",
        "training_view_id",
        "source_training_view_sha256",
        "head_variant",
        "training_row_count",
        "training_positive_count",
        "training_set_membership_sha256",
        "training_feature_matrix_sha256",
        "training_binary_target_sha256",
        "training_edge_target_sha256",
        "learner_input_context_sha256",
        "fit_metadata_template_sha256",
        "feature_schema_sha256",
        "train_label_maturity_through",
        "maximum_training_label_maturity_session",
        "fit_input_spec_sha256",
    }
)
_TRAINING_INPUT_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "development_oof_learner_fit_plan_sha256",
        "source_training_membership_batch_sha256",
        *_FIT_INPUT_SPEC_KEYS,
        "learner_config_sha256",
    }
)
_FIT_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "fit_ordinal",
        "source_training_view_ordinal",
        "training_view_id",
        "head_variant",
        "source_training_view_sha256",
        "training_input_identity",
        "training_input_identity_sha256",
        "fit_metadata",
        "fit_metadata_sha256",
        "learner_state",
        "learner_state_sha256",
        "learner_parameters_sha256",
        "learner_fit_record_sha256",
    }
)
_FIT_VIEW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "source_training_view_ordinal",
        "training_view_id",
        "source_training_view_sha256",
        "training_row_count",
        "training_positive_count",
        "learner_input_context",
        "learner_input_context_sha256",
        "semantic_fit_record_sha256",
        "ablation_fit_record_sha256",
        "semantic_learner_state_sha256",
        "ablation_learner_state_sha256",
        "prediction_fold_context",
        "prediction_fold_context_sha256",
        "beta_1_1_climatology_hex",
        "learner_fit_view_sha256",
    }
)


class SecFilingGemmaLearnerFitError(SecFilingGemmaContractError):
    """Raised when an owned fit crosses its frozen input or authority boundary."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise SecFilingGemmaLearnerFitError(
            f"{location} must be an exact built-in string-keyed dict"
        )
    return value


def _list(value: Any, location: str) -> list[Any]:
    if type(value) is not list:
        raise SecFilingGemmaLearnerFitError(
            f"{location} must be an exact built-in list"
        )
    return value


def _expect_keys(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    if set(value) != set(expected):
        raise SecFilingGemmaLearnerFitError(
            f"{location} keys changed"
        )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaLearnerFitError(
            f"{location} must be a bare lowercase SHA-256"
        )
    return value


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecFilingGemmaLearnerFitError(
            f"{location} must be an exact integer of at least {minimum}"
        )
    return value


def _iso_date(value: Any, location: str) -> str:
    if type(value) is not str:
        raise SecFilingGemmaLearnerFitError(f"{location} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise SecFilingGemmaLearnerFitError(
            f"{location} must be an ISO date"
        ) from None
    if parsed.isoformat() != value:
        raise SecFilingGemmaLearnerFitError(f"{location} must be canonical")
    return value


def _canonical_float_hex(value: Any, location: str) -> float:
    if type(value) is not str:
        raise SecFilingGemmaLearnerFitError(
            f"{location} must be a canonical hexadecimal float"
        )
    try:
        number = float.fromhex(value)
    except ValueError:
        raise SecFilingGemmaLearnerFitError(
            f"{location} must be a canonical hexadecimal float"
        ) from None
    if not math.isfinite(number) or number.hex() != value:
        raise SecFilingGemmaLearnerFitError(
            f"{location} must be a canonical finite hexadecimal float"
        )
    return number


def _self_hash(value: Mapping[str, Any], field: str, location: str) -> str:
    observed = _sha256(value.get(field), f"{location}.{field}")
    body = {key: value[key] for key in value if key != field}
    if not hmac.compare_digest(canonical_sha256(body), observed):
        raise SecFilingGemmaLearnerFitError(f"{location} checksum changed")
    return observed


def _detached(value: Any, location: str, *, depth: int = 0) -> Any:
    """Copy only exact built-in finite JSON without invoking caller hooks."""

    if depth > 64:
        raise SecFilingGemmaLearnerFitError(f"{location} nesting is too deep")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SecFilingGemmaLearnerFitError(
                f"{location} contains a non-finite float"
            )
        return value
    if type(value) is list:
        return [
            _detached(item, f"{location}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise SecFilingGemmaLearnerFitError(
                f"{location} contains a non-string key"
            )
        return {
            key: _detached(item, f"{location}.{key}", depth=depth + 1)
            for key, item in value.items()
        }
    raise SecFilingGemmaLearnerFitError(
        f"{location} contains a non-exact JSON value; exact built-in values are required"
    )


def _validated_metadata(
    raw: Any,
    *,
    view: Mapping[str, Any],
    variant: str,
    candidate_sha256: str,
    feature_schema_sha256: str,
) -> dict[str, Any]:
    metadata = _mapping(raw, f"{variant} fit metadata")
    _expect_keys(metadata, _FIT_METADATA_KEYS, f"{variant} fit metadata")
    try:
        validated = SecFilingGemmaFitMetadata.coerce(metadata)
    except SecFilingGemmaLearnerError as exc:
        raise SecFilingGemmaLearnerFitError(
            f"{variant} fit metadata is invalid"
        ) from exc
    if (
        validated.candidate_sha256 != candidate_sha256
        or validated.head_variant != variant
        or validated.fold_id != view["training_view_id"]
        or validated.train_label_maturity_through
        != view["train_label_maturity_through"]
        or validated.training_set_sha256
        != view["training_set_membership_sha256"]
        or validated.training_row_count != view["training_set_count"]
        or validated.maximum_training_label_maturity_session
        != view["training_set_max_label_maturity_session"]
        or validated.feature_schema_sha256 != feature_schema_sha256
    ):
        raise SecFilingGemmaLearnerFitError(
            f"{variant} fit metadata crossed its exact training input"
        )
    return copy.deepcopy(metadata)


def _validated_training_view(
    raw: Any,
    *,
    expected_ordinal: int,
    expected_view_id: str,
    feature_names: list[str],
    feature_schema_sha256: str,
    candidate_sha256: str,
) -> dict[str, Any]:
    view = _mapping(raw, f"training view {expected_ordinal}")
    _expect_keys(view, _TRAINING_VIEW_KEYS, f"training view {expected_ordinal}")
    _self_hash(view, "training_view_sha256", f"training view {expected_ordinal}")
    row_count = _strict_int(
        view["training_set_count"], "training row count", minimum=2
    )
    positive_count = _strict_int(
        view["training_positive_count"], "training positive count", minimum=1
    )
    maximum_maturity = _iso_date(
        view["training_set_max_label_maturity_session"],
        "training maximum maturity",
    )
    cutoff = _iso_date(
        view["train_label_maturity_through"], "training view cutoff"
    )
    prediction_first = _iso_date(
        view["prediction_window_first_date"], "prediction window first date"
    )
    _iso_date(view["prediction_window_last_date"], "prediction window last date")
    if (
        view["schema_version"] != DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION
        or _strict_int(view["view_ordinal"], "training view ordinal", minimum=1)
        != expected_ordinal
        or view["training_view_id"] != expected_view_id
        or type(view["state_updates_inside_prediction_window"]) is not bool
        or view["state_updates_inside_prediction_window"] is not False
        or positive_count >= row_count
        or maximum_maturity > cutoff
        or maximum_maturity >= prediction_first
    ):
        raise SecFilingGemmaLearnerFitError(
            "Training view crossed its frozen chronology or class boundary"
        )

    members = _list(view["training_set_membership"], "training membership")
    if len(members) != row_count:
        raise SecFilingGemmaLearnerFitError("Training membership count changed")
    for ordinal, member_raw in enumerate(members, start=1):
        member = _mapping(member_raw, f"training member {ordinal}")
        _expect_keys(member, _TRAINING_MEMBER_KEYS, f"training member {ordinal}")
        if (
            _strict_int(
                member["training_row_ordinal"], "training row ordinal", minimum=1
            )
            != ordinal
            or type(member["accession_number"]) is not str
            or type(member["form"]) is not str
            or not member["accession_number"]
            or not member["form"]
        ):
            raise SecFilingGemmaLearnerFitError("Training member order changed")
        _strict_int(member["source_event_ordinal"], "source event ordinal", minimum=1)
        _iso_date(member["decision_session"], "training decision session")
        member_maturity = _iso_date(
            member["label_maturity_session"], "training label maturity"
        )
        if member_maturity > cutoff or member_maturity >= prediction_first:
            raise SecFilingGemmaLearnerFitError(
                "Training member maturity crossed its view"
            )
        _sha256(member["training_event_identity_sha256"], "training event identity")
    if canonical_sha256(members) != _sha256(
        view["training_set_membership_sha256"], "training membership hash"
    ):
        raise SecFilingGemmaLearnerFitError("Training membership hash changed")

    binary = _list(view["training_binary_targets"], "binary targets")
    edge_hex = _list(view["training_edge_targets_hex"], "edge targets")
    if (
        len(binary) != row_count
        or any(type(item) is not int or item not in (0, 1) for item in binary)
        or sum(binary) != positive_count
        or len(edge_hex) != row_count
    ):
        raise SecFilingGemmaLearnerFitError("Training targets changed")
    for index, item in enumerate(edge_hex):
        _canonical_float_hex(item, f"edge target {index}")
    if (
        canonical_sha256(binary)
        != _sha256(view["training_binary_target_sha256"], "binary target hash")
        or canonical_sha256(edge_hex)
        != _sha256(view["training_edge_target_sha256"], "edge target hash")
    ):
        raise SecFilingGemmaLearnerFitError("Training target hash changed")

    dimension = len(feature_names)
    matrices: list[list[list[str]]] = []
    for variant in _VARIANT_IDS:
        matrix_field = f"{variant}_training_features_hex"
        matrix_hash_field = f"{variant}_training_feature_matrix_sha256"
        matrix = _list(view[matrix_field], f"{variant} training matrix")
        if len(matrix) != row_count:
            raise SecFilingGemmaLearnerFitError(
                f"{variant} training matrix row count changed"
            )
        for row_index, row_raw in enumerate(matrix):
            row = _list(row_raw, f"{variant} training row {row_index}")
            if len(row) != dimension:
                raise SecFilingGemmaLearnerFitError(
                    f"{variant} training feature dimension changed"
                )
            for column_index, item in enumerate(row):
                _canonical_float_hex(
                    item, f"{variant} feature {row_index}:{column_index}"
                )
        if canonical_sha256(matrix) != _sha256(
            view[matrix_hash_field], f"{variant} matrix hash"
        ):
            raise SecFilingGemmaLearnerFitError(
                f"{variant} training matrix hash changed"
            )
        matrices.append(matrix)
    if matrices[0] == matrices[1] or hmac.compare_digest(
        view["semantic_training_feature_matrix_sha256"],
        view["ablation_training_feature_matrix_sha256"],
    ):
        raise SecFilingGemmaLearnerFitError(
            "Semantic and ablation training matrices must differ"
        )

    context = _mapping(view["learner_input_context"], "learner input context")
    _expect_keys(context, _LEARNER_CONTEXT_KEYS, "learner input context")
    if (
        context
        != {
            "fold_train_cutoff_session": cutoff,
            "training_set_count": row_count,
            "training_positive_count": positive_count,
            "training_set_membership_sha256": view[
                "training_set_membership_sha256"
            ],
            "semantic_training_feature_matrix_sha256": view[
                "semantic_training_feature_matrix_sha256"
            ],
            "ablation_training_feature_matrix_sha256": view[
                "ablation_training_feature_matrix_sha256"
            ],
            "training_binary_target_sha256": view[
                "training_binary_target_sha256"
            ],
            "training_edge_target_sha256": view[
                "training_edge_target_sha256"
            ],
            "training_set_max_label_maturity_session": maximum_maturity,
        }
        or canonical_sha256(context)
        != _sha256(view["learner_input_context_sha256"], "learner context hash")
    ):
        raise SecFilingGemmaLearnerFitError("Learner input context changed")

    for variant in _VARIANT_IDS:
        metadata_field = f"{variant}_fit_metadata_template"
        metadata_hash_field = f"{variant}_fit_metadata_template_sha256"
        metadata = _validated_metadata(
            view[metadata_field],
            view=view,
            variant=variant,
            candidate_sha256=candidate_sha256,
            feature_schema_sha256=feature_schema_sha256,
        )
        if canonical_sha256(metadata) != _sha256(
            view[metadata_hash_field], f"{variant} metadata hash"
        ):
            raise SecFilingGemmaLearnerFitError(
                f"{variant} fit metadata hash changed"
            )
    return copy.deepcopy(view)


def _validated_membership_batch(raw: Any) -> dict[str, Any]:
    batch = _mapping(
        _detached(raw, "source training-membership batch"),
        "source training-membership batch",
    )
    _expect_keys(batch, _MEMBERSHIP_BATCH_KEYS, "source training-membership batch")
    _self_hash(
        batch,
        "training_membership_batch_sha256",
        "source training-membership batch",
    )
    feature_names = _list(batch["feature_names"], "membership feature names")
    if (
        not feature_names
        or any(type(item) is not str or not item for item in feature_names)
        or len(set(feature_names)) != len(feature_names)
        or canonical_sha256(feature_names)
        != _sha256(batch["feature_schema_sha256"], "feature schema hash")
    ):
        raise SecFilingGemmaLearnerFitError("Membership feature schema changed")
    expected_ids = [*_AUTHORIZED_VIEW_IDS, _DEFERRED_VIEW_ID]
    view_ids = _list(batch["training_view_ids"], "training view ids")
    views = _list(batch["training_views"], "training views")
    event_audits = _list(batch["event_audit_rows"], "event audit rows")
    if (
        batch["schema_version"]
        != OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION
        or _strict_int(batch["training_view_count"], "training view count", minimum=1)
        != 6
        or view_ids != expected_ids
        or len(views) != 6
        or canonical_sha256(views)
        != _sha256(batch["training_views_sha256"], "training views hash")
        or canonical_sha256(event_audits)
        != _sha256(batch["event_audit_rows_sha256"], "event audit rows hash")
    ):
        raise SecFilingGemmaLearnerFitError(
            "Membership batch views or audit ancestry changed"
        )
    for field in (
        "development_root_scope_sha256",
        "training_membership_assembly_plan_sha256",
        "source_feature_assembly_plan_sha256",
        "source_label_assembly_plan_sha256",
        "source_feature_batch_sha256",
        "source_label_batch_sha256",
        "contract_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "calendar_sessions_sha256",
        "membership_view_specs_sha256",
        "model_variant_specs_sha256",
    ):
        _sha256(batch[field], f"membership batch {field}")
    _iso_date(batch["development_cutoff_session"], "development cutoff")
    for field in ("event_count", "matured_label_count", "unmatured_event_count"):
        _strict_int(batch[field], f"membership {field}")
    if (
        batch["matured_label_count"] + batch["unmatured_event_count"]
        != batch["event_count"]
        or batch["contract_sha256"] != canonical_sha256(build_contract_manifest())
        or batch["binary_target_field"] != "cash_beats_long_10bps"
        or batch["binary_target_encoding"] != "false_to_0_true_to_1"
        or batch["edge_target_field"] != "cash_active_log_edge_10bps_hex"
    ):
        raise SecFilingGemmaLearnerFitError("Membership batch semantics changed")
    expected_flags = {
        "development_labels_included": True,
        "development_outcomes_included": True,
        "training_membership_included": True,
        "learner_input_matrices_included": True,
        "learner_targets_included": True,
        "compact_adjusted_open_paths_included": False,
        "full_market_rows_included": False,
        "post_cutoff_market_data_included": False,
        "outcome_based_membership_filtering_permitted": False,
        "row_rebalancing_permitted": False,
        "learner_fit_authorized": False,
        "prediction_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    if any(
        type(batch[field]) is not bool or batch[field] is not expected
        for field, expected in expected_flags.items()
    ):
        raise SecFilingGemmaLearnerFitError(
            "Membership batch capability boundary changed"
        )
    validated_views = [
        _validated_training_view(
            view,
            expected_ordinal=index,
            expected_view_id=expected_ids[index - 1],
            feature_names=feature_names,
            feature_schema_sha256=batch["feature_schema_sha256"],
            candidate_sha256=batch["candidate_sha256"],
        )
        for index, view in enumerate(views, start=1)
    ]
    detached = copy.deepcopy(batch)
    detached["training_views"] = validated_views
    return detached


def derive_development_oof_learner_fit_input_specs(
    source_training_membership_batch: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Derive exactly ten compact fit identities from five fixed views."""

    batch = _validated_membership_batch(source_training_membership_batch)
    result: list[dict[str, Any]] = []
    for view in batch["training_views"][:5]:
        for variant in _VARIANT_IDS:
            metadata = view[f"{variant}_fit_metadata_template"]
            body = {
                "fit_ordinal": len(result) + 1,
                "source_training_view_ordinal": view["view_ordinal"],
                "training_view_id": view["training_view_id"],
                "source_training_view_sha256": view["training_view_sha256"],
                "head_variant": variant,
                "training_row_count": view["training_set_count"],
                "training_positive_count": view["training_positive_count"],
                "training_set_membership_sha256": view[
                    "training_set_membership_sha256"
                ],
                "training_feature_matrix_sha256": view[
                    f"{variant}_training_feature_matrix_sha256"
                ],
                "training_binary_target_sha256": view[
                    "training_binary_target_sha256"
                ],
                "training_edge_target_sha256": view[
                    "training_edge_target_sha256"
                ],
                "learner_input_context_sha256": view[
                    "learner_input_context_sha256"
                ],
                "fit_metadata_template_sha256": view[
                    f"{variant}_fit_metadata_template_sha256"
                ],
                "feature_schema_sha256": metadata["feature_schema_sha256"],
                "train_label_maturity_through": view[
                    "train_label_maturity_through"
                ],
                "maximum_training_label_maturity_session": view[
                    "training_set_max_label_maturity_session"
                ],
            }
            result.append(
                {**body, "fit_input_spec_sha256": canonical_sha256(body)}
            )
    return result


def _validated_plan_and_batch(
    *,
    development_oof_learner_fit_plan: Any,
    expected_development_oof_learner_fit_plan_sha256: str,
    source_training_membership_batch: Any,
    expected_source_training_membership_batch_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = _mapping(
        _detached(
            development_oof_learner_fit_plan,
            "development OOF learner-fit plan",
        ),
        "development OOF learner-fit plan",
    )
    expected_plan_hash = _sha256(
        expected_development_oof_learner_fit_plan_sha256,
        "expected learner-fit plan hash",
    )
    try:
        validate_development_oof_learner_fit_plan(
            plan,
            expected_development_oof_learner_fit_plan_sha256=expected_plan_hash,
        )
    except SecFilingGemmaStageAuthorizationError as exc:
        raise SecFilingGemmaLearnerFitError(
            "Development OOF learner-fit plan failed exact validation"
        ) from exc
    batch = _validated_membership_batch(source_training_membership_batch)
    expected_batch_hash = _sha256(
        expected_source_training_membership_batch_sha256,
        "expected source training-membership batch hash",
    )
    if not hmac.compare_digest(
        batch["training_membership_batch_sha256"], expected_batch_hash
    ):
        raise SecFilingGemmaLearnerFitError(
            "Source training-membership batch is not externally pinned"
        )
    specs = derive_development_oof_learner_fit_input_specs(batch)
    for spec in specs:
        _expect_keys(spec, _FIT_INPUT_SPEC_KEYS, "learner fit-input spec")
        _self_hash(spec, "fit_input_spec_sha256", "learner fit-input spec")
    source_plan = _mapping(
        plan["source_training_membership_assembly_plan"],
        "source training-membership plan",
    )
    if (
        plan["schema_version"] != DEVELOPMENT_OOF_LEARNER_FIT_PLAN_SCHEMA_VERSION
        or plan["development_oof_learner_fit_plan_sha256"] != expected_plan_hash
        or plan["source_training_membership_batch_sha256"] != expected_batch_hash
        or plan["learner_fit_input_specs"] != specs
        or plan["learner_fit_input_specs_sha256"] != canonical_sha256(specs)
        or plan["authorized_training_view_ids"] != list(_AUTHORIZED_VIEW_IDS)
        or plan["source_training_view_ids"]
        != [*_AUTHORIZED_VIEW_IDS, _DEFERRED_VIEW_ID]
        or plan["model_variant_ids"] != list(_VARIANT_IDS)
        or plan["fit_order_rule"] != _FIT_ORDER_RULE
        or plan["maximum_fit_seconds"] != _MAXIMUM_FIT_SECONDS
        or plan["learner_model_type"] != LEARNER_MODEL_TYPE
        or plan["learner_state_schema_version"] != LEARNER_STATE_SCHEMA_VERSION
        or plan["development_root_scope_sha256"]
        != batch["development_root_scope_sha256"]
        or plan["candidate_sha256"] != batch["candidate_sha256"]
        or plan["corpus_universe_sha256"] != batch["corpus_universe_sha256"]
        or plan["calendar_sessions_sha256"]
        != batch["calendar_sessions_sha256"]
        or plan["development_cutoff_session"]
        != batch["development_cutoff_session"]
        or plan["source_training_membership_assembly_plan_sha256"]
        != batch["training_membership_assembly_plan_sha256"]
        or source_plan.get("training_membership_assembly_plan_sha256")
        != batch["training_membership_assembly_plan_sha256"]
        or plan["source_training_view_specs_sha256"]
        != batch["membership_view_specs_sha256"]
        or source_plan.get("model_variant_specs_sha256")
        != batch["model_variant_specs_sha256"]
    ):
        raise SecFilingGemmaLearnerFitError(
            "Learner-fit plan crossed its exact membership ancestry"
        )
    return copy.deepcopy(plan), batch, specs


def _fit_record(
    *,
    plan: Mapping[str, Any],
    batch: Mapping[str, Any],
    view: Mapping[str, Any],
    spec: Mapping[str, Any],
    variant: str,
) -> dict[str, Any]:
    matrix_hex = view[f"{variant}_training_features_hex"]
    matrix = [
        [_canonical_float_hex(item, "training feature") for item in row]
        for row in matrix_hex
    ]
    binary = list(view["training_binary_targets"])
    edge = [
        _canonical_float_hex(item, "training edge")
        for item in view["training_edge_targets_hex"]
    ]
    metadata = copy.deepcopy(view[f"{variant}_fit_metadata_template"])
    try:
        config = SecFilingGemmaLearnerConfig(**copy.deepcopy(plan["learner_config"]))
        config.validate()
        learner = SecFilingGemmaTwoHeadLearner(config)
        learner.fit(
            matrix,
            binary,
            edge,
            feature_names=batch["feature_names"],
            fit_metadata=metadata,
        )
        state = learner.to_state()
        SecFilingGemmaTwoHeadLearner.from_state(state)
    except (TypeError, SecFilingGemmaLearnerError) as exc:
        raise SecFilingGemmaLearnerFitError(
            f"Frozen {view['training_view_id']} {variant} fit failed"
        ) from exc
    input_identity = {
        "development_oof_learner_fit_plan_sha256": plan[
            "development_oof_learner_fit_plan_sha256"
        ],
        "source_training_membership_batch_sha256": batch[
            "training_membership_batch_sha256"
        ],
        **copy.deepcopy(dict(spec)),
        "learner_config_sha256": plan["learner_config_sha256"],
    }
    _expect_keys(
        input_identity, _TRAINING_INPUT_IDENTITY_KEYS, "training input identity"
    )
    body = {
        "schema_version": DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION,
        "fit_ordinal": spec["fit_ordinal"],
        "source_training_view_ordinal": spec[
            "source_training_view_ordinal"
        ],
        "training_view_id": spec["training_view_id"],
        "head_variant": variant,
        "source_training_view_sha256": spec["source_training_view_sha256"],
        "training_input_identity": input_identity,
        "training_input_identity_sha256": canonical_sha256(input_identity),
        "fit_metadata": metadata,
        "fit_metadata_sha256": canonical_sha256(metadata),
        "learner_state": state,
        "learner_state_sha256": state["state_sha256"],
        "learner_parameters_sha256": state["parameters_sha256"],
    }
    return {**body, "learner_fit_record_sha256": canonical_sha256(body)}


def _fit_view(
    *,
    view: Mapping[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    semantic, ablation = records
    prediction_context = {
        **copy.deepcopy(view["learner_input_context"]),
        "semantic_fold_state_sha256": semantic["learner_state_sha256"],
        "ablation_fold_state_sha256": ablation["learner_state_sha256"],
    }
    _expect_keys(
        prediction_context,
        _PREDICTION_FOLD_CONTEXT_KEYS,
        "prediction fold context",
    )
    body = {
        "schema_version": DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION,
        "source_training_view_ordinal": view["view_ordinal"],
        "training_view_id": view["training_view_id"],
        "source_training_view_sha256": view["training_view_sha256"],
        "training_row_count": view["training_set_count"],
        "training_positive_count": view["training_positive_count"],
        "learner_input_context": copy.deepcopy(view["learner_input_context"]),
        "learner_input_context_sha256": view["learner_input_context_sha256"],
        "semantic_fit_record_sha256": semantic["learner_fit_record_sha256"],
        "ablation_fit_record_sha256": ablation["learner_fit_record_sha256"],
        "semantic_learner_state_sha256": semantic["learner_state_sha256"],
        "ablation_learner_state_sha256": ablation["learner_state_sha256"],
        "prediction_fold_context": prediction_context,
        "prediction_fold_context_sha256": canonical_sha256(prediction_context),
        "beta_1_1_climatology_hex": float(
            beta_1_1_climatology(
                view["training_positive_count"], view["training_set_count"]
            )
        ).hex(),
    }
    return {**body, "learner_fit_view_sha256": canonical_sha256(body)}


def build_owned_development_oof_learner_fit_batch(
    *,
    development_oof_learner_fit_plan: Mapping[str, Any],
    expected_development_oof_learner_fit_plan_sha256: str,
    source_training_membership_batch: Mapping[str, Any],
    expected_source_training_membership_batch_sha256: str,
) -> dict[str, Any]:
    """Fit exactly ten fresh learners and emit no training rows or targets."""

    plan, membership, specs = _validated_plan_and_batch(
        development_oof_learner_fit_plan=development_oof_learner_fit_plan,
        expected_development_oof_learner_fit_plan_sha256=(
            expected_development_oof_learner_fit_plan_sha256
        ),
        source_training_membership_batch=source_training_membership_batch,
        expected_source_training_membership_batch_sha256=(
            expected_source_training_membership_batch_sha256
        ),
    )
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for view_index, view in enumerate(membership["training_views"][:5]):
        pair: list[dict[str, Any]] = []
        for variant_index, variant in enumerate(_VARIANT_IDS):
            spec = specs[2 * view_index + variant_index]
            record = _fit_record(
                plan=plan,
                batch=membership,
                view=view,
                spec=spec,
                variant=variant,
            )
            pair.append(record)
            records.append(record)
            if time.monotonic() - started > _MAXIMUM_FIT_SECONDS:
                raise SecFilingGemmaLearnerFitError(
                    "Development OOF learner fits exceeded the frozen 60-second cap"
                )
        views.append(_fit_view(view=view, records=pair))
    if len(records) != 10 or len(views) != 5:
        raise SecFilingGemmaLearnerFitError("Learner fit count changed")
    learner_states_sha256 = canonical_sha256(
        [record["learner_state"] for record in records]
    )
    body = {
        "schema_version": OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION,
        "artifact_stage": "development",
        "development_root_scope_sha256": plan["development_root_scope_sha256"],
        "development_oof_learner_fit_plan_sha256": plan[
            "development_oof_learner_fit_plan_sha256"
        ],
        "source_training_membership_assembly_plan_sha256": plan[
            "source_training_membership_assembly_plan_sha256"
        ],
        "source_training_membership_projection_sha256": plan[
            "source_training_membership_projection_sha256"
        ],
        "source_training_membership_batch_sha256": membership[
            "training_membership_batch_sha256"
        ],
        "contract_sha256": plan["contract_sha256"],
        "candidate_sha256": plan["candidate_sha256"],
        "corpus_universe_sha256": plan["corpus_universe_sha256"],
        "calendar_sessions_sha256": plan["calendar_sessions_sha256"],
        "development_cutoff_session": plan["development_cutoff_session"],
        "source_training_view_count": plan["source_training_view_count"],
        "source_training_view_ids": copy.deepcopy(plan["source_training_view_ids"]),
        "authorized_training_view_count": 5,
        "authorized_training_view_ids": list(_AUTHORIZED_VIEW_IDS),
        "deferred_training_view_count": 1,
        "deferred_training_views": copy.deepcopy(plan["deferred_training_views"]),
        "deferred_training_views_sha256": plan[
            "deferred_training_views_sha256"
        ],
        "model_variant_count": 2,
        "model_variant_ids": list(_VARIANT_IDS),
        "learner_fit_count": 10,
        "learner_state_count": 10,
        "learner_model_type": LEARNER_MODEL_TYPE,
        "learner_state_schema_version": LEARNER_STATE_SCHEMA_VERSION,
        "learner_config_sha256": plan["learner_config_sha256"],
        "fit_order_rule": _FIT_ORDER_RULE,
        "maximum_fit_seconds": _MAXIMUM_FIT_SECONDS,
        "learner_fit_views": views,
        "learner_fit_views_sha256": canonical_sha256(views),
        "learner_fit_records": records,
        "learner_fit_records_sha256": canonical_sha256(records),
        "learner_states_sha256": learner_states_sha256,
        "source_training_membership_batch_included": False,
        "training_membership_rows_included": False,
        "training_feature_matrices_included": False,
        "training_target_vectors_included": False,
        "deferred_training_view_state_included": False,
        "learner_states_included": True,
        "compact_fit_audit_included": True,
        "prediction_included": False,
        "prediction_authorized": False,
        "candidate_selection_authorized": False,
        "threshold_action_authorized": False,
        "holdout_access_authorized": False,
        "ledger_mutation_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {**body, "learner_fit_batch_sha256": canonical_sha256(body)}


def validate_owned_development_oof_learner_fit_batch(
    batch: Mapping[str, Any],
    *,
    development_oof_learner_fit_plan: Mapping[str, Any],
    expected_development_oof_learner_fit_plan_sha256: str,
    source_training_membership_batch: Mapping[str, Any],
    expected_source_training_membership_batch_sha256: str,
    expected_learner_fit_batch_sha256: str,
) -> str:
    """Refit all ten states and require exact deterministic replay."""

    value = _mapping(
        _detached(batch, "owned development OOF learner-fit batch"),
        "owned development OOF learner-fit batch",
    )
    observed = _self_hash(
        value, "learner_fit_batch_sha256", "owned development OOF learner-fit batch"
    )
    expected = _sha256(
        expected_learner_fit_batch_sha256, "expected learner-fit batch hash"
    )
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaLearnerFitError(
            "Development OOF learner-fit batch is not externally pinned"
        )
    rebuilt = build_owned_development_oof_learner_fit_batch(
        development_oof_learner_fit_plan=development_oof_learner_fit_plan,
        expected_development_oof_learner_fit_plan_sha256=(
            expected_development_oof_learner_fit_plan_sha256
        ),
        source_training_membership_batch=source_training_membership_batch,
        expected_source_training_membership_batch_sha256=(
            expected_source_training_membership_batch_sha256
        ),
    )
    if value != rebuilt:
        raise SecFilingGemmaLearnerFitError(
            "Development OOF learner-fit batch differs from deterministic refit"
        )
    for record in value["learner_fit_records"]:
        _expect_keys(record, _FIT_RECORD_KEYS, "learner fit record")
        _self_hash(record, "learner_fit_record_sha256", "learner fit record")
    for view in value["learner_fit_views"]:
        _expect_keys(view, _FIT_VIEW_KEYS, "learner fit view")
        _self_hash(view, "learner_fit_view_sha256", "learner fit view")
    return observed


__all__ = [
    "DEVELOPMENT_OOF_LEARNER_FIT_RECORD_SCHEMA_VERSION",
    "DEVELOPMENT_OOF_LEARNER_FIT_VIEW_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_OOF_LEARNER_FIT_BATCH_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION",
    "SecFilingGemmaLearnerFitError",
    "build_owned_development_oof_learner_fit_batch",
    "derive_development_oof_learner_fit_input_specs",
    "validate_owned_development_oof_learner_fit_batch",
]
