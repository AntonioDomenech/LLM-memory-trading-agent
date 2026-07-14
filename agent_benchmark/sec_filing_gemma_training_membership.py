"""Pure, outcome-blind training membership for the SEC/Gemma experiment.

This module joins the already owned development feature and label batches.  It
does not read files, call a model, fit a learner, make a prediction, or mutate a
ledger.  Membership depends only on frozen chronology and causal feature
availability.  Label values are copied into separate target vectors only after
the ordered membership has been fixed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from datetime import date
import hmac
import math
import re
from typing import Any, Final

from .sec_filing_gemma_contract import (
    ACTIVE_EDGE_TOLERANCE,
    BRIER_TARGET_COST_BPS,
    SecFilingGemmaContractError,
    build_contract_manifest,
    canonical_sha256,
)
from .sec_filing_gemma_features import (
    ABLATION_FEATURE_COLUMNS,
    SEMANTIC_FEATURE_COLUMNS,
    validate_owned_development_feature_batch,
    validate_owned_development_label_batch,
)
from .sec_filing_gemma_learner import SecFilingGemmaFitMetadata


DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-training-membership-assembly-plan-v1"
)
OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-training-membership-projection-v1"
)
OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-owned-development-training-membership-batch-v1"
)
DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-development-training-view-v1"
)

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
_VIEW_DECISION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "training_view_id",
        "label_maturity_eligible",
        "feature_fit_eligible",
        "included",
        "exclusion_reasons",
    }
)
_EVENT_AUDIT_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_ordinal",
        "accession_number",
        "form",
        "decision_session",
        "label_maturity_session",
        "matured_by_development_cutoff",
        "feature_row_sha256",
        "label_evidence_sha256",
        "extraction_identity_sha256",
        "market_prefix_chain_identity_sha256",
        "market_feature_row_sha256",
        "semantic_feature_schema_sha256",
        "semantic_feature_values_sha256",
        "ablation_feature_schema_sha256",
        "ablation_feature_values_sha256",
        "training_event_identity_sha256",
        "prediction_available",
        "fit_eligible",
        "unavailable_reasons",
        "training_view_decisions",
        "training_view_decisions_sha256",
        "event_audit_row_sha256",
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
_LEARNER_INPUT_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
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
_BATCH_KEYS: Final[frozenset[str]] = frozenset(
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

_LABEL_NOT_MATURE_REASON: Final[str] = "label_not_mature_by_view_cutoff"
_FEATURE_NOT_ELIGIBLE_REASON: Final[str] = "feature_not_fit_eligible"
_EXCLUSION_REASONS: Final[frozenset[str]] = frozenset(
    {_LABEL_NOT_MATURE_REASON, _FEATURE_NOT_ELIGIBLE_REASON}
)


class SecFilingGemmaTrainingMembershipError(SecFilingGemmaContractError):
    """Raised when training support is noncanonical, leaky, or unbound."""


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(
        type(key) is str for key in value
    ):
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be a string-keyed mapping"
        )
    return value


def _expect_keys(
    value: Mapping[str, Any], expected: frozenset[str], location: str
) -> None:
    observed = set(value)
    if observed != set(expected):
        raise SecFilingGemmaTrainingMembershipError(
            f"Invalid {location} keys; missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _optional_sha256(value: Any, location: str) -> str | None:
    return None if value is None else _sha256(value, location)


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be an exact integer of at least {minimum}"
        )
    return value


def _iso_date(value: Any, location: str) -> str:
    if type(value) is not str:
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be a canonical ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be a canonical ISO date"
        ) from None
    if parsed.isoformat() != value:
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be a canonical ISO date"
        )
    return value


def _canonical_float_hex(value: Any, location: str) -> str:
    if type(value) is not str:
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be canonical finite float.hex text"
        )
    try:
        decoded = float.fromhex(value)
    except ValueError:
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be canonical finite float.hex text"
        ) from None
    if not math.isfinite(decoded) or decoded.hex() != value:
        raise SecFilingGemmaTrainingMembershipError(
            f"{location} must be canonical finite float.hex text"
        )
    return value


def _validated_membership_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = _expect_mapping(plan, "development training membership assembly plan")
    try:
        from .sec_filing_gemma_stage_authorization import (
            validate_development_training_membership_assembly_plan,
        )

        observed = validate_development_training_membership_assembly_plan(
            value,
            expected_training_membership_assembly_plan_sha256=value.get(
                "training_membership_assembly_plan_sha256"
            ),
        )
    except Exception:
        raise SecFilingGemmaTrainingMembershipError(
            "Development training membership plan failed exact authorization replay"
        ) from None
    if (
        value.get("schema_version")
        != DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION
        or observed != value.get("training_membership_assembly_plan_sha256")
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Development training membership plan identity changed"
        )
    return copy.deepcopy(dict(value))


def _validated_sources(
    *,
    plan: Mapping[str, Any],
    source_feature_batch: Mapping[str, Any],
    source_label_batch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    label_plan = _expect_mapping(
        plan.get("source_label_assembly_plan"),
        "membership source label assembly plan",
    )
    feature_plan = _expect_mapping(
        label_plan.get("source_feature_assembly_plan"),
        "membership source feature assembly plan",
    )
    feature_batch = _expect_mapping(
        source_feature_batch, "membership source feature batch"
    )
    label_batch = _expect_mapping(
        source_label_batch, "membership source label batch"
    )
    try:
        validate_owned_development_feature_batch(
            feature_batch,
            feature_assembly_plan=feature_plan,
            expected_feature_assembly_plan_sha256=plan[
                "source_feature_assembly_plan_sha256"
            ],
            expected_feature_batch_sha256=feature_batch.get(
                "feature_batch_sha256"
            ),
        )
        validate_owned_development_label_batch(
            label_batch,
            label_assembly_plan=label_plan,
            expected_label_assembly_plan_sha256=plan[
                "source_label_assembly_plan_sha256"
            ],
            source_feature_batch=feature_batch,
            expected_source_feature_batch_sha256=feature_batch.get(
                "feature_batch_sha256"
            ),
            expected_label_batch_sha256=label_batch.get("label_batch_sha256"),
        )
    except Exception:
        raise SecFilingGemmaTrainingMembershipError(
            "Development training membership source batches failed exact replay"
        ) from None

    cross_bindings = (
        feature_batch.get("development_root_scope_sha256")
        == plan.get("development_root_scope_sha256")
        == label_batch.get("development_root_scope_sha256")
        and feature_batch.get("feature_assembly_plan_sha256")
        == plan.get("source_feature_assembly_plan_sha256")
        == label_batch.get("source_feature_assembly_plan_sha256")
        and label_batch.get("label_assembly_plan_sha256")
        == plan.get("source_label_assembly_plan_sha256")
        and label_batch.get("source_feature_batch_sha256")
        == feature_batch.get("feature_batch_sha256")
        and feature_batch.get("candidate_sha256")
        == plan.get("candidate_sha256")
        == label_batch.get("candidate_sha256")
        and feature_batch.get("corpus_universe_sha256")
        == plan.get("corpus_universe_sha256")
        == label_batch.get("corpus_universe_sha256")
        and feature_batch.get("event_count") == plan.get("event_count")
        == label_batch.get("event_count")
        and label_batch.get("matured_label_count")
        == plan.get("matured_event_count")
        and label_batch.get("unmatured_event_count")
        == plan.get("unmatured_event_count")
    )
    if not cross_bindings:
        raise SecFilingGemmaTrainingMembershipError(
            "Development training membership source batches crossed their plan"
        )
    return (
        copy.deepcopy(dict(feature_plan)),
        copy.deepcopy(dict(label_plan)),
        copy.deepcopy(dict(feature_batch)),
        copy.deepcopy(dict(label_batch)),
    )


def _label_rows_by_event(
    label_batch: Mapping[str, Any],
) -> list[dict[str, Any] | None]:
    audits = label_batch.get("maturity_audit_rows")
    labels = label_batch.get("label_evidence_rows")
    if type(audits) is not list or type(labels) is not list:
        raise SecFilingGemmaTrainingMembershipError(
            "Membership source label rows are not exact lists"
        )
    result: list[dict[str, Any] | None] = []
    label_index = 0
    for ordinal, audit in enumerate(audits, start=1):
        audit_value = _expect_mapping(audit, f"membership label audit {ordinal}")
        matured = audit_value.get("matured_by_development_cutoff")
        if type(matured) is not bool:
            raise SecFilingGemmaTrainingMembershipError(
                "Membership label maturity flag must be an exact boolean"
            )
        if matured:
            if label_index >= len(labels):
                raise SecFilingGemmaTrainingMembershipError(
                    "Membership source omitted a matured development label"
                )
            label = _expect_mapping(
                labels[label_index], f"membership label evidence {label_index + 1}"
            )
            if (
                label.get("accession_number") != audit_value.get("accession_number")
                or label.get("decision_session")
                != audit_value.get("decision_session")
                or label.get("label_maturity_session")
                != audit_value.get("label_maturity_session")
                or label.get("feature_row_sha256")
                != audit_value.get("feature_row_sha256")
                or label.get("label_evidence_sha256")
                != audit_value.get("label_evidence_sha256")
            ):
                raise SecFilingGemmaTrainingMembershipError(
                    "Membership label evidence crossed its chronological audit row"
                )
            result.append(copy.deepcopy(dict(label)))
            label_index += 1
        else:
            if audit_value.get("label_evidence_sha256") is not None:
                raise SecFilingGemmaTrainingMembershipError(
                    "Chronologically unmatured membership event exposed a label"
                )
            result.append(None)
    if label_index != len(labels):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership source contains an unclaimed development label"
        )
    return result


def _view_specs(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = plan.get("membership_view_specs")
    if type(raw) is not list or not raw:
        raise SecFilingGemmaTrainingMembershipError(
            "Membership plan has no exact training views"
        )
    specs: list[dict[str, Any]] = []
    for ordinal, raw_spec in enumerate(raw, start=1):
        spec = _expect_mapping(raw_spec, f"membership view spec {ordinal}")
        _expect_keys(spec, _VIEW_SPEC_KEYS, f"membership view spec {ordinal}")
        if (
            _strict_int(spec["view_ordinal"], "membership view ordinal", minimum=1)
            != ordinal
            or type(spec["training_view_id"]) is not str
            or not spec["training_view_id"]
            or type(spec["view_kind"]) is not str
            or not spec["view_kind"]
            or type(spec["training_source_stage"]) is not str
            or type(spec["prediction_stage"]) is not str
            or type(spec["state_updates_inside_prediction_window"]) is not bool
        ):
            raise SecFilingGemmaTrainingMembershipError(
                "Membership view identity or chronology type changed"
            )
        cutoff = _iso_date(
            spec["train_label_maturity_through"],
            "membership view train-label cutoff",
        )
        first = _iso_date(
            spec["prediction_window_first_date"],
            "membership view prediction first date",
        )
        last = _iso_date(
            spec["prediction_window_last_date"],
            "membership view prediction last date",
        )
        if not (cutoff < first <= last):
            raise SecFilingGemmaTrainingMembershipError(
                "Membership view chronology is not strictly causal"
            )
        specs.append(copy.deepcopy(dict(spec)))
    if len({spec["training_view_id"] for spec in specs}) != len(specs):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership training view identifiers are duplicated"
        )
    if (
        plan.get("membership_view_count") != len(specs)
        or plan.get("membership_view_specs_sha256") != canonical_sha256(specs)
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership view count or checksum changed"
        )
    return specs


def _event_identity_and_audit_base(
    *,
    ordinal: int,
    feature_row: Mapping[str, Any],
    label_audit: Mapping[str, Any],
    label_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    feature = _expect_mapping(feature_row, f"membership feature row {ordinal}")
    audit = _expect_mapping(label_audit, f"membership label audit {ordinal}")
    bindings = _expect_mapping(
        feature.get("bindings"), f"membership feature bindings {ordinal}"
    )
    if (
        audit.get("event_ordinal") != ordinal
        or feature.get("accession_number") != audit.get("accession_number")
        or feature.get("decision_session") != audit.get("decision_session")
        or feature.get("feature_row_sha256") != audit.get("feature_row_sha256")
        or type(audit.get("matured_by_development_cutoff")) is not bool
        or type(feature.get("prediction_available")) is not bool
        or type(feature.get("fit_eligible")) is not bool
        or feature.get("fit_eligible") is not feature.get("prediction_available")
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership event crossed its feature or maturity audit identity"
        )
    reasons = feature.get("unavailable_reasons")
    if type(reasons) is not list or any(type(item) is not str for item in reasons):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership feature unavailable reasons are not an exact list"
        )
    feature_hash = _sha256(feature.get("feature_row_sha256"), "feature_row_sha256")
    label_hash = _optional_sha256(
        audit.get("label_evidence_sha256"), "label_evidence_sha256"
    )
    if (label_row is None) is not (label_hash is None):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership label value and audit hash availability differ"
        )
    identity_fields = {
        "accession_number": feature.get("accession_number"),
        "form": feature.get("form"),
        "decision_session": feature.get("decision_session"),
        "label_maturity_session": audit.get("label_maturity_session"),
        "extraction_identity_sha256": _sha256(
            bindings.get("extraction_identity_sha256"),
            "extraction_identity_sha256",
        ),
        "market_prefix_chain_identity_sha256": _sha256(
            bindings.get("market_prefix_chain_identity_sha256"),
            "market_prefix_chain_identity_sha256",
        ),
        "market_feature_row_sha256": _sha256(
            feature.get("market_feature_row_sha256"),
            "market_feature_row_sha256",
        ),
        "semantic_feature_schema_sha256": _sha256(
            feature.get("semantic_feature_schema_sha256"),
            "semantic_feature_schema_sha256",
        ),
        "semantic_feature_values_sha256": canonical_sha256(
            feature.get("semantic_feature_values_hex")
        ),
        "ablation_feature_schema_sha256": _sha256(
            feature.get("ablation_feature_schema_sha256"),
            "ablation_feature_schema_sha256",
        ),
        "ablation_feature_values_sha256": canonical_sha256(
            feature.get("ablation_feature_values_hex")
        ),
    }
    if (
        type(identity_fields["accession_number"]) is not str
        or type(identity_fields["form"]) is not str
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership event accession or form is invalid"
        )
    _iso_date(identity_fields["decision_session"], "membership decision session")
    _iso_date(
        identity_fields["label_maturity_session"],
        "membership label maturity session",
    )
    training_event_hash = canonical_sha256(identity_fields)
    return {
        "event_ordinal": ordinal,
        **identity_fields,
        "matured_by_development_cutoff": audit[
            "matured_by_development_cutoff"
        ],
        "feature_row_sha256": feature_hash,
        "label_evidence_sha256": label_hash,
        "training_event_identity_sha256": training_event_hash,
        "prediction_available": feature["prediction_available"],
        "fit_eligible": feature["fit_eligible"],
        "unavailable_reasons": copy.deepcopy(reasons),
    }


def _view_decision(
    *, audit_base: Mapping[str, Any], view_spec: Mapping[str, Any]
) -> dict[str, Any]:
    maturity = _iso_date(
        audit_base["label_maturity_session"], "membership label maturity"
    )
    cutoff = _iso_date(
        view_spec["train_label_maturity_through"], "membership train cutoff"
    )
    first = _iso_date(
        view_spec["prediction_window_first_date"], "membership prediction first"
    )
    label_eligible = bool(
        audit_base["matured_by_development_cutoff"]
        and audit_base["label_evidence_sha256"] is not None
        and maturity <= cutoff
        and maturity < first
    )
    feature_eligible = bool(
        audit_base["fit_eligible"] is True
        and audit_base["prediction_available"] is True
    )
    reasons: list[str] = []
    if not label_eligible:
        reasons.append(_LABEL_NOT_MATURE_REASON)
    if not feature_eligible:
        reasons.append(_FEATURE_NOT_ELIGIBLE_REASON)
    return {
        "training_view_id": view_spec["training_view_id"],
        "label_maturity_eligible": label_eligible,
        "feature_fit_eligible": feature_eligible,
        "included": not reasons,
        "exclusion_reasons": reasons,
    }


def _training_member(
    *, training_row_ordinal: int, audit_row: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "training_row_ordinal": training_row_ordinal,
        "source_event_ordinal": audit_row["event_ordinal"],
        "accession_number": audit_row["accession_number"],
        "form": audit_row["form"],
        "decision_session": audit_row["decision_session"],
        "label_maturity_session": audit_row["label_maturity_session"],
        "training_event_identity_sha256": audit_row[
            "training_event_identity_sha256"
        ],
    }


def _fit_metadata_template(
    *,
    candidate_sha256: str,
    variant: str,
    view_spec: Mapping[str, Any],
    membership_sha256: str,
    row_count: int,
    maximum_maturity: str,
    feature_schema_sha256: str,
) -> dict[str, Any]:
    value = {
        "candidate_sha256": candidate_sha256,
        "head_variant": variant,
        "fold_id": view_spec["training_view_id"],
        "train_label_maturity_through": view_spec[
            "train_label_maturity_through"
        ],
        "training_set_sha256": membership_sha256,
        "training_row_count": row_count,
        "maximum_training_label_maturity_session": maximum_maturity,
        "feature_schema_sha256": feature_schema_sha256,
    }
    try:
        SecFilingGemmaFitMetadata.coerce(value)
    except Exception:
        raise SecFilingGemmaTrainingMembershipError(
            "Membership learner fit metadata template is invalid"
        ) from None
    return value


def _build_training_view(
    *,
    view_spec: Mapping[str, Any],
    event_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any] | None],
    candidate_sha256: str,
    semantic_schema_sha256: str,
    ablation_schema_sha256: str,
    minimum_training_row_count: int,
) -> dict[str, Any]:
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    view_id = view_spec["training_view_id"]
    for audit, feature, label in zip(
        event_rows, feature_rows, label_rows, strict=True
    ):
        decision = next(
            item
            for item in audit["training_view_decisions"]
            if item["training_view_id"] == view_id
        )
        if decision["included"]:
            if label is None:
                raise SecFilingGemmaTrainingMembershipError(
                    "An included training member has no matured target"
                )
            selected.append((audit, feature, label))

    if len(selected) < minimum_training_row_count:
        raise SecFilingGemmaTrainingMembershipError(
            f"Training view {view_id} has fewer than the frozen minimum rows"
        )
    members = [
        _training_member(training_row_ordinal=index, audit_row=audit)
        for index, (audit, _feature, _label) in enumerate(selected, start=1)
    ]
    semantic_matrix: list[list[str]] = []
    ablation_matrix: list[list[str]] = []
    binary_targets: list[int] = []
    edge_targets: list[str] = []
    for row_index, (_audit, feature, label) in enumerate(selected, start=1):
        semantic = feature.get("semantic_feature_values_hex")
        ablation = feature.get("ablation_feature_values_hex")
        if type(semantic) is not list or type(ablation) is not list:
            raise SecFilingGemmaTrainingMembershipError(
                "A fit-eligible training member has no exact feature vectors"
            )
        if (
            len(semantic) != len(SEMANTIC_FEATURE_COLUMNS)
            or len(ablation) != len(ABLATION_FEATURE_COLUMNS)
        ):
            raise SecFilingGemmaTrainingMembershipError(
                "Training feature matrix width changed"
            )
        semantic_matrix.append(
            [
                _canonical_float_hex(item, f"semantic matrix row {row_index}")
                for item in semantic
            ]
        )
        ablation_matrix.append(
            [
                _canonical_float_hex(item, f"ablation matrix row {row_index}")
                for item in ablation
            ]
        )
        binary = label.get("cash_beats_long_10bps")
        if type(binary) is not bool:
            raise SecFilingGemmaTrainingMembershipError(
                "Training binary target must be an exact boolean source value"
            )
        binary_targets.append(1 if binary else 0)
        edge_targets.append(
            _canonical_float_hex(
                label.get("cash_active_log_edge_10bps_hex"),
                f"training edge target row {row_index}",
            )
        )

    positive_count = sum(binary_targets)
    if positive_count <= 0 or positive_count >= len(binary_targets):
        raise SecFilingGemmaTrainingMembershipError(
            f"Training view {view_id} does not contain both binary classes"
        )
    semantic_hash = canonical_sha256(semantic_matrix)
    ablation_hash = canonical_sha256(ablation_matrix)
    if hmac.compare_digest(semantic_hash, ablation_hash):
        raise SecFilingGemmaTrainingMembershipError(
            f"Training view {view_id} semantic and ablation matrices are identical"
        )
    membership_hash = canonical_sha256(members)
    binary_hash = canonical_sha256(binary_targets)
    edge_hash = canonical_sha256(edge_targets)
    maximum_maturity = max(
        member["label_maturity_session"] for member in members
    )
    if not (
        maximum_maturity <= view_spec["train_label_maturity_through"]
        and maximum_maturity < view_spec["prediction_window_first_date"]
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Training view maximum maturity crossed its causal boundary"
        )
    context = {
        "fold_train_cutoff_session": view_spec["train_label_maturity_through"],
        "training_set_count": len(members),
        "training_positive_count": positive_count,
        "training_set_membership_sha256": membership_hash,
        "semantic_training_feature_matrix_sha256": semantic_hash,
        "ablation_training_feature_matrix_sha256": ablation_hash,
        "training_binary_target_sha256": binary_hash,
        "training_edge_target_sha256": edge_hash,
        "training_set_max_label_maturity_session": maximum_maturity,
    }
    semantic_metadata = _fit_metadata_template(
        candidate_sha256=candidate_sha256,
        variant="semantic",
        view_spec=view_spec,
        membership_sha256=membership_hash,
        row_count=len(members),
        maximum_maturity=maximum_maturity,
        feature_schema_sha256=semantic_schema_sha256,
    )
    ablation_metadata = _fit_metadata_template(
        candidate_sha256=candidate_sha256,
        variant="ablation",
        view_spec=view_spec,
        membership_sha256=membership_hash,
        row_count=len(members),
        maximum_maturity=maximum_maturity,
        feature_schema_sha256=ablation_schema_sha256,
    )
    body = {
        "schema_version": DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION,
        **copy.deepcopy(dict(view_spec)),
        "training_set_count": len(members),
        "training_positive_count": positive_count,
        "training_set_max_label_maturity_session": maximum_maturity,
        "training_set_membership": members,
        "training_set_membership_sha256": membership_hash,
        "semantic_training_features_hex": semantic_matrix,
        "semantic_training_feature_matrix_sha256": semantic_hash,
        "ablation_training_features_hex": ablation_matrix,
        "ablation_training_feature_matrix_sha256": ablation_hash,
        "training_binary_targets": binary_targets,
        "training_binary_target_sha256": binary_hash,
        "training_edge_targets_hex": edge_targets,
        "training_edge_target_sha256": edge_hash,
        "learner_input_context": context,
        "learner_input_context_sha256": canonical_sha256(context),
        "semantic_fit_metadata_template": semantic_metadata,
        "semantic_fit_metadata_template_sha256": canonical_sha256(
            semantic_metadata
        ),
        "ablation_fit_metadata_template": ablation_metadata,
        "ablation_fit_metadata_template_sha256": canonical_sha256(
            ablation_metadata
        ),
    }
    return {**body, "training_view_sha256": canonical_sha256(body)}


def build_owned_development_training_membership_batch(
    *,
    training_membership_assembly_plan: Mapping[str, Any],
    source_feature_batch: Mapping[str, Any],
    source_label_batch: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive all six fixed training views without fitting or target filtering."""

    plan = _validated_membership_plan(training_membership_assembly_plan)
    _feature_plan, label_plan, feature_batch, label_batch = _validated_sources(
        plan=plan,
        source_feature_batch=source_feature_batch,
        source_label_batch=source_label_batch,
    )
    specs = _view_specs(plan)
    feature_rows = feature_batch.get("feature_rows")
    label_audits = label_batch.get("maturity_audit_rows")
    if (
        type(feature_rows) is not list
        or type(label_audits) is not list
        or len(feature_rows) != plan["event_count"]
        or len(label_audits) != plan["event_count"]
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership source event counts changed"
        )
    label_rows = _label_rows_by_event(label_batch)

    event_audits: list[dict[str, Any]] = []
    for ordinal, (feature, label_audit, label_row) in enumerate(
        zip(feature_rows, label_audits, label_rows, strict=True), start=1
    ):
        base = _event_identity_and_audit_base(
            ordinal=ordinal,
            feature_row=feature,
            label_audit=label_audit,
            label_row=label_row,
        )
        decisions = [
            _view_decision(audit_base=base, view_spec=spec) for spec in specs
        ]
        body = {
            **base,
            "training_view_decisions": decisions,
            "training_view_decisions_sha256": canonical_sha256(decisions),
        }
        event_audits.append(
            {**body, "event_audit_row_sha256": canonical_sha256(body)}
        )

    semantic_schema_sha256 = canonical_sha256(list(SEMANTIC_FEATURE_COLUMNS))
    ablation_schema_sha256 = canonical_sha256(list(ABLATION_FEATURE_COLUMNS))
    if list(SEMANTIC_FEATURE_COLUMNS) != list(ABLATION_FEATURE_COLUMNS):
        raise SecFilingGemmaTrainingMembershipError(
            "Semantic and ablation feature names no longer share one schema"
        )
    views = [
        _build_training_view(
            view_spec=spec,
            event_rows=event_audits,
            feature_rows=feature_rows,
            label_rows=label_rows,
            candidate_sha256=feature_batch["candidate_sha256"],
            semantic_schema_sha256=semantic_schema_sha256,
            ablation_schema_sha256=ablation_schema_sha256,
            minimum_training_row_count=plan["minimum_training_row_count"],
        )
        for spec in specs
    ]
    body = {
        "schema_version": (
            OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION
        ),
        "development_root_scope_sha256": plan[
            "development_root_scope_sha256"
        ],
        "training_membership_assembly_plan_sha256": plan[
            "training_membership_assembly_plan_sha256"
        ],
        "source_feature_assembly_plan_sha256": plan[
            "source_feature_assembly_plan_sha256"
        ],
        "source_label_assembly_plan_sha256": plan[
            "source_label_assembly_plan_sha256"
        ],
        "source_feature_batch_sha256": feature_batch["feature_batch_sha256"],
        "source_label_batch_sha256": label_batch["label_batch_sha256"],
        "contract_sha256": canonical_sha256(build_contract_manifest()),
        "candidate_sha256": plan["candidate_sha256"],
        "corpus_universe_sha256": plan["corpus_universe_sha256"],
        "calendar_sessions_sha256": plan["calendar_sessions_sha256"],
        "development_cutoff_session": plan["development_cutoff_session"],
        "event_count": plan["event_count"],
        "matured_label_count": plan["matured_event_count"],
        "unmatured_event_count": plan["unmatured_event_count"],
        "training_view_count": plan["membership_view_count"],
        "training_view_ids": [spec["training_view_id"] for spec in specs],
        "membership_view_specs_sha256": plan["membership_view_specs_sha256"],
        "model_variant_specs_sha256": plan["model_variant_specs_sha256"],
        "feature_names": list(SEMANTIC_FEATURE_COLUMNS),
        "feature_schema_sha256": semantic_schema_sha256,
        "target_cost_bps": plan["target_cost_bps"],
        "binary_target_field": plan["binary_target_field"],
        "binary_target_encoding": plan["binary_target_encoding"],
        "edge_target_field": plan["edge_target_field"],
        "binary_comparison_tolerance_hex": float(
            ACTIVE_EDGE_TOLERANCE
        ).hex(),
        "membership_maturity_rule": plan["membership_maturity_rule"],
        "feature_eligibility_rule": plan["feature_eligibility_rule"],
        "membership_order_rule": plan["membership_order_rule"],
        "shared_variant_support_rule": plan["shared_variant_support_rule"],
        "event_audit_rows": event_audits,
        "event_audit_rows_sha256": canonical_sha256(event_audits),
        "training_views": views,
        "training_views_sha256": canonical_sha256(views),
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
    if (
        plan["target_cost_bps"] != BRIER_TARGET_COST_BPS
        or plan["binary_target_field"] != "cash_beats_long_10bps"
        or plan["edge_target_field"] != "cash_active_log_edge_10bps_hex"
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership target semantics changed"
        )
    return {
        **body,
        "training_membership_batch_sha256": canonical_sha256(body),
    }


def validate_owned_development_training_membership_batch(
    batch: Mapping[str, Any],
    *,
    training_membership_assembly_plan: Mapping[str, Any],
    expected_training_membership_assembly_plan_sha256: str,
    source_feature_batch: Mapping[str, Any],
    expected_source_feature_batch_sha256: str,
    source_label_batch: Mapping[str, Any],
    expected_source_label_batch_sha256: str,
    expected_training_membership_batch_sha256: str,
) -> str:
    """Rebuild the exact membership artifact and require all external pins."""

    value = _expect_mapping(batch, "owned development training membership batch")
    _expect_keys(value, _BATCH_KEYS, "owned development training membership batch")
    observed = _sha256(
        value["training_membership_batch_sha256"],
        "training_membership_batch_sha256",
    )
    supplied_body = {
        key: value[key]
        for key in value
        if key != "training_membership_batch_sha256"
    }
    if canonical_sha256(supplied_body) != observed:
        raise SecFilingGemmaTrainingMembershipError(
            "Owned development training membership batch checksum changed"
        )
    plan = _validated_membership_plan(training_membership_assembly_plan)
    if plan["training_membership_assembly_plan_sha256"] != _sha256(
        expected_training_membership_assembly_plan_sha256,
        "expected_training_membership_assembly_plan_sha256",
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Training membership assembly plan is not externally pinned"
        )
    feature_batch = _expect_mapping(
        source_feature_batch, "externally pinned membership feature batch"
    )
    label_batch = _expect_mapping(
        source_label_batch, "externally pinned membership label batch"
    )
    if feature_batch.get("feature_batch_sha256") != _sha256(
        expected_source_feature_batch_sha256,
        "expected_source_feature_batch_sha256",
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership source feature batch is not externally pinned"
        )
    if label_batch.get("label_batch_sha256") != _sha256(
        expected_source_label_batch_sha256,
        "expected_source_label_batch_sha256",
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Membership source label batch is not externally pinned"
        )
    rebuilt = build_owned_development_training_membership_batch(
        training_membership_assembly_plan=plan,
        source_feature_batch=feature_batch,
        source_label_batch=label_batch,
    )
    if dict(value) != rebuilt:
        raise SecFilingGemmaTrainingMembershipError(
            "Owned development training membership batch differs from exact replay"
        )
    if observed != _sha256(
        expected_training_membership_batch_sha256,
        "expected_training_membership_batch_sha256",
    ):
        raise SecFilingGemmaTrainingMembershipError(
            "Training membership batch is not externally pinned"
        )
    return observed


__all__ = [
    "DEVELOPMENT_TRAINING_MEMBERSHIP_ASSEMBLY_PLAN_SCHEMA_VERSION",
    "DEVELOPMENT_TRAINING_VIEW_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_BATCH_SCHEMA_VERSION",
    "OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION",
    "SecFilingGemmaTrainingMembershipError",
    "build_owned_development_training_membership_batch",
    "validate_owned_development_training_membership_batch",
]
