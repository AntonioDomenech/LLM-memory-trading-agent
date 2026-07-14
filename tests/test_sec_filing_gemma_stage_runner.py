from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

import agent_benchmark.sec_filing_gemma_corpus as corpus_module
import agent_benchmark.sec_filing_gemma_stage_runner as runner_module
from agent_benchmark.sec_audit_transport import ResponseAudit
from agent_benchmark.sec_filing_gemma_contract import CONTRACT_VERSION, canonical_sha256
from agent_benchmark.sec_filing_gemma_reveal_store import (
    DEVELOPMENT_SEC_ROOT_COMPLETE_MARKER_SCHEMA_VERSION,
    SEC_BATCH_COMPLETE_MARKER_FILENAME,
    SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION,
    SEC_STAGE_COMPONENT_DIRECTORY_NAME,
    SecFilingGemmaRevealStore,
)
from agent_benchmark.sec_filing_gemma_corpus import SecCorpusBudget
from agent_benchmark.sec_filing_gemma_stage_access import (
    DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES,
)
from agent_benchmark.sec_filing_gemma_stage_authorization import (
    OWNED_SEC_RAW_BATCH_MAX_BYTES,
    SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
)
from agent_benchmark.sec_filing_gemma_stage_runner import (
    SecFilingGemmaStageRunnerError,
    run_authorized_sec_stage,
    run_owned_development_feature_batch,
    run_owned_development_label_batch,
    run_owned_development_model_batch,
    run_owned_development_oof_learner_fit_batch,
    run_owned_development_oof_prediction_batch,
    run_owned_development_policy_replay_batch,
    run_owned_development_sec_root,
    run_owned_development_training_membership_batch,
    run_owned_stage_model_batch,
)
from agent_benchmark.sec_filing_gemma_learner_fit import (
    OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_learner_prediction import (
    OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_policy_replay import (
    OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_training_membership import (
    OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_market_evidence import (
    market_session_calendar_sha256,
)
from agent_benchmark.sec_point_in_time import content_sha256, validate_sec_user_agent
from agent_benchmark.sec_session_calendar import EXPECTED_MARKET_HISTORY_SESSIONS
from tests import test_sec_filing_gemma_policy_replay as policy_replay_scaffold
from tests import (
    test_sec_filing_gemma_stage_authorization as authorization_scaffold,
)


USER_AGENT = "Private Owner owner-contact@real-domain-for-tests.dev"
USER_AGENT_SHA256 = validate_sec_user_agent(USER_AGENT).sha256
REQUEST_SHA256 = "1" * 64
CLAIM_SHA256 = "2" * 64
NAMESPACE = "ns"
ACCESSION = "0000320193-24-000001"
OFFICIAL_URL = (
    "https://www.sec.gov/Archives/edgar/data/320193/"
    "000032019324000001/apple-2024.htm"
)
RAW_DOCUMENT = b"<html><body><p>Exact filing bytes &amp; evidence.</p></body></html>"
DEVELOPMENT_CANDIDATE_SHA256 = "5" * 64
DEVELOPMENT_UNIVERSE_SHA256 = "6" * 64


def _claim() -> dict[str, Any]:
    return {
        "request_sha256": REQUEST_SHA256,
        "claim_sha256": CLAIM_SHA256,
        "output_namespace": NAMESPACE,
        "sec_user_agent_sha256": USER_AGENT_SHA256,
        "execution_source_hashes": {},
    }


def _stage_model_claim(
    *, request_sha256: str = "a" * 64
) -> dict[str, Any]:
    event_plan = [
        {
            "event_ordinal": 1,
            "accession_number": "0000320193-20-000001",
            "form": "10-K",
            "availability_session": "2020-01-02",
            "sec_document_ordinal": 1,
        }
    ]
    limits = {
        "model_call_count": 1,
        "maximum_model_seconds": 3_600,
        "redirects": 0,
        "retries": 0,
        "pull_attempts": 0,
        "repair_attempts": 0,
        "streaming": False,
        "thinking": False,
    }
    sources = {"preprocessor": "e" * 64}
    body = {
        "request_sha256": request_sha256,
        "candidate_sha256": "b" * 64,
        "authorized_stage": "intermediate",
        "output_namespace": "model-stage-test",
        "corpus_universe_sha256": "f" * 64,
        "stage_sec_reader_receipt_sha256": "c" * 64,
        "carry_in_reader_receipt_sha256": "d" * 64,
        "event_count": 1,
        "event_plan": event_plan,
        "event_plan_sha256": canonical_sha256(event_plan),
        "identity_lexicon_sha256": canonical_sha256(
            list(runner_module.CANONICAL_IDENTITY_LEXICON)
        ),
        "execution_source_hashes": sources,
        "execution_source_hashes_sha256": canonical_sha256(sources),
        "model_name": "gemma4:12b",
        "model_digest": "1" * 64,
        "runtime_fingerprint_sha256": "2" * 64,
        "model_transport_sha256": "3" * 64,
        "model_runtime_limits": limits,
        "model_runtime_limits_sha256": canonical_sha256(limits),
        "model_component_id": "owned_stage_gemma_model_batch",
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def _development_model_claim(
    *, scope_sha256: str = "9" * 64
) -> dict[str, Any]:
    event_plan = [
        {
            "event_ordinal": 1,
            "accession_number": "0000320193-00-000001",
            "form": "10-K",
            "availability_session": "2000-01-03",
            "sec_document_ordinal": 1,
        }
    ]
    limits = {
        "model_call_count": 1,
        "maximum_model_seconds": 3_600,
        "redirects": 0,
        "retries": 0,
        "pull_attempts": 0,
        "repair_attempts": 0,
        "streaming": False,
        "thinking": False,
    }
    sources = {"preprocessor": "e" * 64}
    body = {
        "development_root_scope_sha256": scope_sha256,
        "candidate_sha256": "b" * 64,
        "authorized_stage": "development",
        "output_namespace": "model-development-test",
        "corpus_universe_sha256": "f" * 64,
        "development_sec_reader_receipt_sha256": "c" * 64,
        "event_count": 1,
        "event_plan": event_plan,
        "event_plan_sha256": canonical_sha256(event_plan),
        "identity_lexicon_sha256": canonical_sha256(
            list(runner_module.CANONICAL_IDENTITY_LEXICON)
        ),
        "execution_source_hashes": sources,
        "execution_source_hashes_sha256": canonical_sha256(sources),
        "model_name": "gemma4:12b",
        "model_digest": "1" * 64,
        "runtime_fingerprint_sha256": "2" * 64,
        "model_transport_sha256": "3" * 64,
        "model_runtime_limits": limits,
        "model_runtime_limits_sha256": canonical_sha256(limits),
        "model_component_id": "owned_stage_gemma_model_batch",
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def _development_feature_plan(
    *,
    scope_sha256: str = "9" * 64,
    event_plan: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    events = event_plan or [
        {
            "event_ordinal": 1,
            "accession_number": "0000320193-02-000001",
            "form": "10-Q",
            "availability_session": "2002-02-01",
            "sec_document_ordinal": 1,
        }
    ]
    body = {
        "schema_version": "aapl-sec-gemma-development-feature-assembly-plan-v1",
        "contract_version": CONTRACT_VERSION,
        "plan_kind": "request_free_development_feature_assembly",
        "artifact_stage": "development",
        "development_root_scope_sha256": scope_sha256,
        "development_content_root_plan_sha256": "1" * 64,
        "candidate_sha256": "2" * 64,
        "candidate_design_sha256": "3" * 64,
        "corpus_universe_sha256": "4" * 64,
        "development_cutoff_session": "2018-12-31",
        "start_consumed_request_count": 0,
        "development_sec_execution_claim_sha256": "5" * 64,
        "development_sec_reader_receipt_sha256": "6" * 64,
        "development_market_execution_claim_sha256": "7" * 64,
        "development_market_reader_receipt_sha256": "8" * 64,
        "development_market_acquisition_receipt_sha256": "a" * 64,
        "development_market_acquisition_bundle_sha256": "b" * 64,
        "development_market_acquisition_validation_sha256": "c" * 64,
        "development_market_source_manifest_sha256": "d" * 64,
        "development_market_stage_manifest_sha256": "e" * 64,
        "development_market_source_reconciliation_sha256": "f" * 64,
        "development_market_byte_index_sha256": hashlib.sha256(
            b"market-byte-index"
        ).hexdigest(),
        "development_model_execution_claim_sha256": hashlib.sha256(
            b"model-claim"
        ).hexdigest(),
        "development_model_reader_receipt_sha256": hashlib.sha256(
            b"model-reader"
        ).hexdigest(),
        "event_count": len(events),
        "event_plan": events,
        "event_plan_sha256": canonical_sha256(events),
        "execution_source_hashes_sha256": hashlib.sha256(
            b"feature-sources"
        ).hexdigest(),
        "canonical_market_rows_required": True,
        "raw_market_output_permitted": False,
        "normalized_filing_text_output_permitted": False,
        "model_transport_envelope_output_permitted": False,
        "feature_rows_output_permitted": True,
        "outcome_access_permitted": False,
        "label_access_permitted": False,
        "training_membership_access_permitted": False,
        "learner_fit_permitted": False,
        "prediction_access_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
    }
    return {**body, "feature_assembly_plan_sha256": canonical_sha256(body)}


def _rehash_development_feature_plan(plan: dict[str, Any]) -> dict[str, Any]:
    plan["event_plan_sha256"] = canonical_sha256(plan["event_plan"])
    body = {
        key: plan[key] for key in plan if key != "feature_assembly_plan_sha256"
    }
    plan["feature_assembly_plan_sha256"] = canonical_sha256(body)
    return plan


def _development_feature_projection(plan: dict[str, Any]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for event_plan_item in plan["event_plan"]:
        decision_session = event_plan_item["availability_session"]
        decision_index = EXPECTED_MARKET_HISTORY_SESSIONS.index(decision_session)
        sessions = EXPECTED_MARKET_HISTORY_SESSIONS[
            decision_index - 252 : decision_index + 1
        ]
        prefix = {
            "artifact_stage": "development",
            "decision_event_id": event_plan_item["accession_number"],
            "decision_session": decision_session,
            "market_cutoff_session": decision_session,
            "market_stage_manifest_sha256": plan[
                "development_market_stage_manifest_sha256"
            ],
            "source_manifest_sha256": plan[
                "development_market_source_manifest_sha256"
            ],
            "lookback_row_count": 253,
            "lookback_rows": [{"session": session} for session in sessions],
        }
        events.append(
            {
                "event_ordinal": event_plan_item["event_ordinal"],
                "event_plan_item": event_plan_item,
                "market_prefix": prefix,
                "market_prefix_proof": {
                    "market_prefix_proof_sha256": hashlib.sha256(
                        f"market-proof-{event_plan_item['event_ordinal']}".encode()
                    ).hexdigest()
                },
                "universe_event_proof": {
                    "universe_event_proof_sha256": hashlib.sha256(
                        f"universe-proof-{event_plan_item['event_ordinal']}".encode()
                    ).hexdigest()
                },
                "extraction_event_proof": {
                    "extraction_event_proof_sha256": hashlib.sha256(
                        f"extraction-proof-{event_plan_item['event_ordinal']}".encode()
                    ).hexdigest()
                },
            }
        )
    body = {
        "schema_version": "aapl-sec-gemma-owned-development-feature-inputs-v1",
        "feature_assembly_plan": plan,
        "events": events,
    }
    return {**body, "feature_inputs_sha256": canonical_sha256(body)}


def _rehash_development_feature_projection(
    projection: dict[str, Any],
) -> dict[str, Any]:
    body = {
        key: projection[key]
        for key in projection
        if key != "feature_inputs_sha256"
    }
    projection["feature_inputs_sha256"] = canonical_sha256(body)
    return projection


def _fake_feature_batch(
    plan: dict[str, Any], feature_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    body = {
        "schema_version": "aapl-sec-gemma-owned-development-feature-batch-v1",
        "development_root_scope_sha256": plan[
            "development_root_scope_sha256"
        ],
        "feature_assembly_plan_sha256": plan["feature_assembly_plan_sha256"],
        "candidate_sha256": plan["candidate_sha256"],
        "corpus_universe_sha256": plan["corpus_universe_sha256"],
        "development_sec_reader_receipt_sha256": plan[
            "development_sec_reader_receipt_sha256"
        ],
        "development_market_reader_receipt_sha256": plan[
            "development_market_reader_receipt_sha256"
        ],
        "development_model_reader_receipt_sha256": plan[
            "development_model_reader_receipt_sha256"
        ],
        "event_count": plan["event_count"],
        "event_plan_sha256": plan["event_plan_sha256"],
        "feature_row_schema_version": "aapl-sec-gemma-feature-row-v1",
        "feature_row_sha256s": [row["feature_row_sha256"] for row in feature_rows],
        "feature_rows_sha256": canonical_sha256(feature_rows),
        "feature_rows": feature_rows,
        "labels_included": False,
        "outcomes_included": False,
        "post_decision_market_rows_included": False,
        "training_membership_included": False,
        "learner_fit_authorized": False,
        "stage_promotion_authorized": False,
        "production_authorized": False,
    }
    return {**body, "feature_batch_sha256": canonical_sha256(body)}


def _development_label_plan(feature_plan: dict[str, Any]) -> dict[str, Any]:
    maturity_plan: list[dict[str, Any]] = []
    for event in feature_plan["event_plan"]:
        decision_index = EXPECTED_MARKET_HISTORY_SESSIONS.index(
            event["availability_session"]
        )
        maturity = EXPECTED_MARKET_HISTORY_SESSIONS[decision_index + 21]
        maturity_plan.append(
            {
                "event_ordinal": event["event_ordinal"],
                "accession_number": event["accession_number"],
                "form": event["form"],
                "decision_session": event["availability_session"],
                "sec_document_ordinal": event["sec_document_ordinal"],
                "label_maturity_session": maturity,
                "matured_by_development_cutoff": maturity <= "2018-12-31",
            }
        )
    matured = sum(item["matured_by_development_cutoff"] for item in maturity_plan)
    body = {
        "schema_version": "aapl-sec-gemma-development-label-assembly-plan-v1",
        "contract_version": CONTRACT_VERSION,
        "plan_kind": "request_free_development_label_assembly",
        "artifact_stage": "development",
        "development_root_scope_sha256": feature_plan[
            "development_root_scope_sha256"
        ],
        "start_consumed_request_count": 0,
        "source_feature_assembly_plan": copy.deepcopy(feature_plan),
        "source_feature_assembly_plan_sha256": feature_plan[
            "feature_assembly_plan_sha256"
        ],
        "calendar_sessions_sha256": market_session_calendar_sha256(
            EXPECTED_MARKET_HISTORY_SESSIONS
        ),
        "development_cutoff_session": "2018-12-31",
        "label_horizon_sessions": 20,
        "label_entry_session_offset": 1,
        "label_maturity_session_offset": 21,
        "maturity_rule": "t_plus_21_session_lte_development_cutoff_inclusive",
        "event_count": len(maturity_plan),
        "maturity_plan": maturity_plan,
        "maturity_plan_sha256": canonical_sha256(maturity_plan),
        "matured_event_count": matured,
        "unmatured_event_count": len(maturity_plan) - matured,
        "canonical_market_rows_required": True,
        "development_outcome_derivation_permitted": True,
        "development_label_rows_output_permitted": True,
        "post_cutoff_market_access_permitted": False,
        "raw_market_output_permitted": False,
        "normalized_filing_text_output_permitted": False,
        "model_transport_envelope_output_permitted": False,
        "training_membership_access_permitted": False,
        "learner_fit_permitted": False,
        "prediction_access_permitted": False,
        "holdout_access_permitted": False,
        "ledger_mutation_permitted": False,
        "stage_promotion_permitted": False,
        "production_permitted": False,
    }
    return {**body, "label_assembly_plan_sha256": canonical_sha256(body)}


def _development_label_projection(
    plan: dict[str, Any],
    source_feature_batch: dict[str, Any],
) -> dict[str, Any]:
    audits: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    feature_rows = source_feature_batch["feature_rows"]
    for maturity_item, feature_row in zip(
        plan["maturity_plan"], feature_rows, strict=True
    ):
        label_hash = (
            hashlib.sha256(
                f"label-{maturity_item['event_ordinal']}".encode()
            ).hexdigest()
            if maturity_item["matured_by_development_cutoff"]
            else None
        )
        audits.append(
            {
                "event_ordinal": maturity_item["event_ordinal"],
                "accession_number": maturity_item["accession_number"],
                "decision_session": maturity_item["decision_session"],
                "feature_row_sha256": feature_row["feature_row_sha256"],
                "label_maturity_session": maturity_item[
                    "label_maturity_session"
                ],
                "matured_by_development_cutoff": maturity_item[
                    "matured_by_development_cutoff"
                ],
                "label_evidence_sha256": label_hash,
            }
        )
        if label_hash is not None:
            labels.append(
                {
                    "accession_number": maturity_item["accession_number"],
                    "label_evidence_sha256": label_hash,
                }
            )
    body = {
        "schema_version": "aapl-sec-gemma-owned-development-label-projection-v1",
        "label_assembly_plan": plan,
        "source_feature_batch": source_feature_batch,
        "maturity_audit_rows": audits,
        "label_evidence_rows": labels,
    }
    return {**body, "label_projection_sha256": canonical_sha256(body)}


def _rehash_development_label_projection(
    projection: dict[str, Any],
) -> dict[str, Any]:
    body = {
        key: projection[key]
        for key in projection
        if key != "label_projection_sha256"
    }
    projection["label_projection_sha256"] = canonical_sha256(body)
    return projection


def _development_training_membership_projection(
    *, scope_sha256: str = "9" * 64
) -> dict[str, Any]:
    feature_batch = {
        "development_root_scope_sha256": scope_sha256,
        "feature_assembly_plan_sha256": "1" * 64,
        "feature_batch_sha256": "2" * 64,
        "candidate_sha256": "3" * 64,
        "corpus_universe_sha256": "4" * 64,
        "event_count": 4,
        "private_market_rows": "FEATURE-SOURCE-MUST-NOT-ESCAPE",
    }
    label_batch = {
        "development_root_scope_sha256": scope_sha256,
        "source_feature_assembly_plan_sha256": "1" * 64,
        "source_feature_batch_sha256": "2" * 64,
        "label_assembly_plan_sha256": "5" * 64,
        "label_batch_sha256": "6" * 64,
        "candidate_sha256": "3" * 64,
        "corpus_universe_sha256": "4" * 64,
        "event_count": 4,
        "matured_label_count": 4,
        "unmatured_event_count": 0,
        "private_adjusted_open_path": "LABEL-SOURCE-MUST-NOT-ESCAPE",
    }
    plan_body = {
        "development_root_scope_sha256": scope_sha256,
        "source_label_assembly_plan": {
            "label_assembly_plan_sha256": "5" * 64,
        },
        "source_feature_assembly_plan_sha256": "1" * 64,
        "source_label_assembly_plan_sha256": "5" * 64,
        "candidate_sha256": "3" * 64,
        "corpus_universe_sha256": "4" * 64,
        "event_count": 4,
        "matured_event_count": 4,
        "unmatured_event_count": 0,
    }
    plan = {
        **plan_body,
        "training_membership_assembly_plan_sha256": canonical_sha256(plan_body),
    }
    body = {
        "schema_version": (
            OWNED_DEVELOPMENT_TRAINING_MEMBERSHIP_PROJECTION_SCHEMA_VERSION
        ),
        "training_membership_assembly_plan": plan,
        "source_feature_batch": feature_batch,
        "source_label_batch": label_batch,
    }
    return {**body, "membership_projection_sha256": canonical_sha256(body)}


def _rehash_development_training_membership_projection(
    projection: dict[str, Any],
) -> dict[str, Any]:
    body = {
        key: projection[key]
        for key in projection
        if key != "membership_projection_sha256"
    }
    projection["membership_projection_sha256"] = canonical_sha256(body)
    return projection


def _development_oof_learner_fit_projection(
    *, scope_sha256: str = "9" * 64
) -> dict[str, Any]:
    membership = {
        "development_root_scope_sha256": scope_sha256,
        "training_membership_assembly_plan_sha256": "1" * 64,
        "training_membership_batch_sha256": "2" * 64,
        "candidate_sha256": "3" * 64,
        "corpus_universe_sha256": "4" * 64,
        "calendar_sessions_sha256": "5" * 64,
        "development_cutoff_session": "2018-12-31",
        "private_training_rows": "MEMBERSHIP-SOURCE-MUST-NOT-ESCAPE",
    }
    plan = {
        "development_oof_learner_fit_plan_sha256": "6" * 64,
        "development_root_scope_sha256": scope_sha256,
        "source_training_membership_batch_sha256": "2" * 64,
        "source_training_membership_assembly_plan_sha256": "1" * 64,
        "candidate_sha256": "3" * 64,
        "corpus_universe_sha256": "4" * 64,
        "calendar_sessions_sha256": "5" * 64,
        "development_cutoff_session": "2018-12-31",
    }
    body = {
        "schema_version": (
            OWNED_DEVELOPMENT_OOF_LEARNER_FIT_PROJECTION_SCHEMA_VERSION
        ),
        "development_oof_learner_fit_plan": plan,
        "source_training_membership_batch": membership,
    }
    return {
        **body,
        "learner_fit_projection_sha256": canonical_sha256(body),
    }


def _rehash_development_oof_learner_fit_projection(
    projection: dict[str, Any],
) -> dict[str, Any]:
    body = {
        key: projection[key]
        for key in projection
        if key != "learner_fit_projection_sha256"
    }
    projection["learner_fit_projection_sha256"] = canonical_sha256(body)
    return projection


def _development_oof_prediction_projection(
    *, scope_sha256: str = "9" * 64
) -> dict[str, Any]:
    fold_specs = [{"fold_id": "FOLD-SPECS"}]
    input_specs = [{"prediction_ordinal": "INPUT-SPECS"}]
    plan = {
        "development_oof_prediction_plan_sha256": "6" * 64,
        "development_root_scope_sha256": scope_sha256,
        "source_development_oof_learner_fit_batch_sha256": "5" * 64,
        "source_feature_batch_sha256": "4" * 64,
        "prediction_fold_model_bundle_sha256": "7" * 64,
        "prediction_feature_batch_sha256": "8" * 64,
        "prediction_fold_model_count": 5,
        "prediction_fold_model_specs": fold_specs,
        "prediction_fold_model_specs_sha256": canonical_sha256(fold_specs),
        "prediction_input_count": 3,
        "prediction_input_specs": input_specs,
        "prediction_input_specs_sha256": canonical_sha256(input_specs),
        "contract_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "corpus_universe_sha256": "c" * 64,
        "calendar_sessions_sha256": "d" * 64,
        "development_cutoff_session": "2018-12-31",
    }
    fold_models = {
        "source_learner_fit_batch_sha256": "5" * 64,
        "prediction_fold_model_bundle_sha256": "7" * 64,
        "fold_model_count": 5,
        "contract_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "corpus_universe_sha256": "c" * 64,
        "calendar_sessions_sha256": "d" * 64,
        "development_cutoff_session": "2018-12-31",
        "private_training_state": "ONLY-TEN-COMPACT-STATES-MAY-ESCAPE",
    }
    features = {
        "source_feature_batch_sha256": "4" * 64,
        "prediction_feature_batch_sha256": "8" * 64,
        "prediction_event_count": 3,
        "contract_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "corpus_universe_sha256": "c" * 64,
        "private_future_features": "NO-POST-2018-FEATURES-MAY-ESCAPE",
    }
    body = {
        "schema_version": (
            OWNED_DEVELOPMENT_OOF_PREDICTION_PROJECTION_SCHEMA_VERSION
        ),
        "development_oof_prediction_plan": plan,
        "prediction_fold_model_bundle": fold_models,
        "prediction_feature_batch": features,
    }
    return {**body, "prediction_projection_sha256": canonical_sha256(body)}


def _rehash_development_oof_prediction_projection(
    projection: dict[str, Any],
) -> dict[str, Any]:
    body = {
        key: projection[key]
        for key in projection
        if key != "prediction_projection_sha256"
    }
    projection["prediction_projection_sha256"] = canonical_sha256(body)
    return projection


def _development_policy_replay_projection(
    *, scope_sha256: str = "9" * 64
) -> dict[str, Any]:
    input_specs = [{"input_ordinal": "POLICY-INPUT-SPECS"}]
    source = {
        "schema_version": "test-owned-oof-prediction-batch-v1",
        "development_root_scope_sha256": scope_sha256,
        "development_oof_prediction_plan_sha256": "1" * 64,
        "prediction_batch_sha256": "2" * 64,
        "raw_prediction_rows_sha256": "3" * 64,
        "raw_prediction_tip_sha256": "4" * 64,
        "prediction_event_count": 1,
        "raw_prediction_rows": [{"raw_prediction_row_sha256": "4" * 64}],
        "contract_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "corpus_universe_sha256": "c" * 64,
        "calendar_sessions_sha256": "d" * 64,
        "development_cutoff_session": "2018-12-31",
        "model_variant_count": 2,
        "model_variant_ids": ["semantic", "ablation"],
    }
    plan = {
        "development_policy_replay_plan_sha256": "6" * 64,
        "development_root_scope_sha256": scope_sha256,
        "source_development_oof_prediction_plan_sha256": "1" * 64,
        "source_development_oof_prediction_projection_sha256": "7" * 64,
        "source_development_oof_prediction_batch_sha256": "2" * 64,
        "source_raw_prediction_rows_sha256": "3" * 64,
        "source_raw_prediction_tip_sha256": "4" * 64,
        "source_raw_prediction_row_count": 1,
        "policy_replay_input_count": 1,
        "policy_replay_input_specs": input_specs,
        "policy_replay_input_specs_sha256": canonical_sha256(input_specs),
        "contract_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "corpus_universe_sha256": "c" * 64,
        "calendar_sessions_sha256": "d" * 64,
        "development_cutoff_session": "2018-12-31",
        "candidate_count": 4,
        "candidate_ids": ["p50_e0", "p55_e0", "p50_e25", "p55_e25"],
        "model_variant_count": 2,
        "model_variant_ids": ["semantic", "ablation"],
        "candidate_gate_comparison_rule": (
            "probability_gte_and_expected_edge_gte"
        ),
        "cash_episode_rule": (
            "accepted_after_close_fill_t_plus_1_exit_t_plus_21_fixed_20_session_"
            "cash_episode_never_extend_scheduled_or_active_episode"
        ),
        "unavailable_prediction_rule": (
            "unavailable_prediction_starts_no_new_cash_episode_"
            "existing_episode_keeps_original_exit"
        ),
        "policy_replay_order_rule": (
            "source_raw_prediction_ordinal_ascending_exactly_once"
        ),
    }
    body = {
        "schema_version": (
            OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION
        ),
        "development_policy_replay_plan": plan,
        "source_development_oof_prediction_batch": source,
    }
    return {
        **body,
        "policy_replay_projection_sha256": canonical_sha256(body),
    }


def _rehash_development_policy_replay_projection(
    projection: dict[str, Any],
) -> dict[str, Any]:
    body = {
        key: projection[key]
        for key in projection
        if key != "policy_replay_projection_sha256"
    }
    projection["policy_replay_projection_sha256"] = canonical_sha256(body)
    return projection


def _component_plan() -> dict[str, Any]:
    return {
        "documents": [
            {"accession_number": ACCESSION, "official_url": OFFICIAL_URL}
        ],
        "max_requests": 1,
        "authorized_max_sec_response_bytes": 100_000,
        "owned_sec_raw_batch_max_bytes": OWNED_SEC_RAW_BATCH_MAX_BYTES,
        "max_bytes": 100_000,
        "max_seconds": 10.0,
    }


def _development_root_plan() -> dict[str, Any]:
    universe = {
        "universe_sha256": DEVELOPMENT_UNIVERSE_SHA256,
        "records": [
            {
                "accession_number": ACCESSION,
                "artifact_stage": "development",
                "form": "10-K",
                "availability_session": "2018-11-05",
                "primary_document": "apple-2024.htm",
            }
        ],
    }
    root_scope = {
        "artifact_stage": "development",
        "candidate_sha256": DEVELOPMENT_CANDIDATE_SHA256,
        "candidate_design_sha256": "7" * 64,
        "attempt_id": "attempt-001",
        "corpus_universe_sha256": DEVELOPMENT_UNIVERSE_SHA256,
        "corpus_universe_semantic_sha256": "8" * 64,
        "document_count": 1,
        "output_namespace": "development-root-attempt-001",
        "component_id": DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    }
    body = {
        "development_root_scope_sha256": canonical_sha256(root_scope),
        "root_scope": root_scope,
        "corpus_universe_manifest": universe,
        "sec_access_plan": {
            "artifact_stage": "development",
            "document_count": 1,
            "documents": [
                {"accession_number": ACCESSION, "official_url": OFFICIAL_URL}
            ],
        },
        "budgets": {
            "max_sec_requests": 1,
            "max_raw_batch_bytes": DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES,
            "max_sec_acquisition_seconds": 10.0,
        },
        "output": {
            "namespace": "development-root-attempt-001",
            "component_id": DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
        },
    }
    return {
        **body,
        "development_content_root_plan_sha256": canonical_sha256(body),
    }


def _development_root_claim(
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exact_plan = plan or _development_root_plan()
    return {
        "development_root_scope_sha256": exact_plan[
            "development_root_scope_sha256"
        ],
        "development_content_root_plan_sha256": exact_plan[
            "development_content_root_plan_sha256"
        ],
        "development_content_root_plan": exact_plan,
        "claim_sha256": CLAIM_SHA256,
        "output_namespace": "development-root-attempt-001",
        "candidate_sha256": DEVELOPMENT_CANDIDATE_SHA256,
        "sec_user_agent_sha256": USER_AGENT_SHA256,
        "execution_source_hashes": {},
    }


def _development_root_component_plan(
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "documents": [
            {"accession_number": ACCESSION, "official_url": OFFICIAL_URL}
        ],
        "max_requests": 1,
        "max_bytes": DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES,
        "max_seconds": 10.0,
        "corpus_universe_sha256": DEVELOPMENT_UNIVERSE_SHA256,
        "corpus_universe_manifest": plan["corpus_universe_manifest"],
        "development_content_root_plan_sha256": plan[
            "development_content_root_plan_sha256"
        ],
    }


def _new_store(tmp_path: Path) -> SecFilingGemmaRevealStore:
    repository = tmp_path / "repository"
    repository.mkdir()
    directory = tmp_path / "store"
    directory.mkdir()
    store = object.__new__(SecFilingGemmaRevealStore)
    store._repository_root = repository
    store._store_directory = directory
    store._lock_timeout_seconds = 1.0
    return store


class FakeTransport:
    def __init__(
        self,
        *,
        payload: bytes = RAW_DOCUMENT,
        fail: bool = False,
        max_requests: int = 1,
        max_bytes: int = 100_000,
        max_seconds: float = 10.0,
    ) -> None:
        self.payload = payload
        self.fail = fail
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds
        self.calls: list[str] = []
        self.user_agent_audit = validate_sec_user_agent(USER_AGENT)

    def acquisition_security_state(self) -> dict[str, Any]:
        return {
            "trust_env": False,
            "proxies": False,
            "follow_redirects": False,
            "max_retries": 0,
            "max_redirects": 0,
            "allow_cache_reads": False,
            "allow_cache_writes": False,
            "streaming_body": True,
            "content_length_preflight": True,
            "incremental_byte_budget": True,
            "transport_max_requests": self.max_requests,
            "transport_max_bytes": self.max_bytes,
            "transport_max_seconds": self.max_seconds,
        }

    def fetch(self, url: str) -> tuple[bytes, ResponseAudit]:
        self.calls.append(url)
        if self.fail:
            raise RuntimeError(USER_AGENT)
        audit = ResponseAudit(
            url=url,
            status_code=200,
            content_type="text/html; charset=iso-8859-1",
            size_bytes=len(self.payload),
            content_sha256=content_sha256(self.payload),
            cache_hit=False,
            user_agent_sha256=self.user_agent_audit.sha256,
            network_requests=1,
            retries=0,
            redirects=0,
        )
        return self.payload, replace(audit)


def _install_created_claim_store(
    store: SecFilingGemmaRevealStore,
    *,
    tip: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    records: list[str] = []
    aborts: list[str] = []
    claim = _claim()

    def claim_execution(
        *,
        request_sha256: str,
        sec_user_agent_sha256: str,
    ) -> dict[str, Any]:
        assert request_sha256 == REQUEST_SHA256
        assert sec_user_agent_sha256 == USER_AGENT_SHA256
        return {
            "claim": claim,
            "created": True,
            "reader_receipt": None,
            "abort": None,
        }

    def load_tip() -> dict[str, Any]:
        return tip or {
            "authorization_bundles": {REQUEST_SHA256: {}},
            "stage_sec_execution_claims": {REQUEST_SHA256: claim},
            "stage_sec_reader_receipts": {},
            "stage_sec_execution_aborts": {},
        }

    def record_output(*, request_sha256: str) -> dict[str, Any]:
        records.append(request_sha256)
        return {
            "claim_sha256": CLAIM_SHA256,
            "receipt_sha256": "3" * 64,
        }

    def abort_execution(*, request_sha256: str, reason: str) -> dict[str, Any]:
        assert request_sha256 == REQUEST_SHA256
        aborts.append(reason)
        return {"reason": reason}

    store.claim_authorized_sec_stage_execution = claim_execution
    store.load_current_tip_anchor = load_tip
    store._record_authorized_sec_stage_reader_output = record_output
    store._revalidate_authorized_sec_execution_sources = lambda _claim: None
    store.abort_authorized_sec_stage_execution = abort_execution
    return records, aborts


def _install_created_development_root_store(
    store: SecFilingGemmaRevealStore,
    *,
    plan: dict[str, Any],
) -> tuple[list[str], list[str]]:
    records: list[str] = []
    aborts: list[str] = []
    claim = _development_root_claim(plan)
    root_scope_sha256 = plan["development_root_scope_sha256"]

    def claim_execution(
        *,
        development_content_root_plan: dict[str, Any],
        sec_user_agent_sha256: str,
    ) -> dict[str, Any]:
        assert development_content_root_plan == plan
        assert sec_user_agent_sha256 == USER_AGENT_SHA256
        return {
            "claim": claim,
            "created": True,
            "reader_receipt": None,
            "abort": None,
        }

    def record_output(*, development_root_scope_sha256: str) -> dict[str, Any]:
        assert development_root_scope_sha256 == root_scope_sha256
        records.append(development_root_scope_sha256)
        return {
            "development_root_scope_sha256": root_scope_sha256,
            "claim_sha256": CLAIM_SHA256,
            "receipt_sha256": "3" * 64,
        }

    def abort_execution(
        *,
        development_root_scope_sha256: str,
        reason: str,
    ) -> dict[str, Any]:
        assert development_root_scope_sha256 == root_scope_sha256
        aborts.append(reason)
        return {"reason": reason}

    store.claim_owned_development_sec_root_execution = claim_execution
    store._record_owned_development_sec_root_reader_output = record_output
    store.abort_owned_development_sec_root_execution = abort_execution
    store._revalidate_authorized_sec_execution_sources = lambda _claim: None
    return records, aborts


def _install_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "_load_component_plan",
        lambda _store, *, request_sha256, claim: _component_plan(),
    )


def _install_development_root_plan(
    monkeypatch: pytest.MonkeyPatch,
    plan: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        runner_module,
        "_load_development_root_component_plan",
        lambda _store, *, development_root_scope_sha256, claim: (
            _development_root_component_plan(plan)
        ),
    )


def _install_transport_factory(
    monkeypatch: pytest.MonkeyPatch,
    transport: FakeTransport,
    calls: list[str],
) -> None:
    @contextmanager
    def factory(**kwargs: Any) -> Iterator[FakeTransport]:
        calls.append("factory")
        assert kwargs["user_agent"] == USER_AGENT
        assert kwargs["transport_budget"].max_requests == 1
        assert kwargs["transport_budget"].max_bytes == 100_000
        assert kwargs["transport_budget"].max_seconds == 10.0
        assert kwargs["cache_directory"].name == ".disabled-sec-cache"
        yield transport

    monkeypatch.setattr(runner_module, "_owned_transport_factory", factory)


def _install_development_root_transport_factory(
    monkeypatch: pytest.MonkeyPatch,
    transport: FakeTransport,
    calls: list[str],
) -> None:
    @contextmanager
    def factory(**kwargs: Any) -> Iterator[FakeTransport]:
        calls.append("factory")
        assert kwargs["user_agent"] == USER_AGENT
        assert kwargs["transport_budget"].max_requests == 1
        assert (
            kwargs["transport_budget"].max_bytes
            == DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES
        )
        assert kwargs["transport_budget"].max_seconds == 10.0
        assert kwargs["cache_directory"].name == ".disabled-sec-cache"
        yield transport

    monkeypatch.setattr(runner_module, "_owned_transport_factory", factory)


def test_public_runner_signature_exposes_no_effect_authority() -> None:
    signature = inspect.signature(run_authorized_sec_stage)
    assert tuple(signature.parameters) == (
        "reveal_store",
        "request_sha256",
        "user_agent",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    forbidden = {
        "stage",
        "candidate",
        "grant",
        "url",
        "path",
        "digest",
        "bytes",
        "transport",
        "budget",
    }
    assert forbidden.isdisjoint(signature.parameters)
    assert runner_module.__all__ == [
        "SecFilingGemmaStageRunnerError",
        "run_authorized_sec_stage",
        "run_owned_development_feature_batch",
        "run_owned_development_label_batch",
        "run_owned_development_oof_learner_fit_batch",
        "run_owned_development_oof_prediction_batch",
        "run_owned_development_policy_replay_batch",
        "run_owned_development_training_membership_batch",
        "run_owned_development_market_batch",
        "run_owned_development_model_batch",
        "run_owned_development_sec_root",
        "run_owned_stage_model_batch",
    ]


def test_public_development_root_runner_signature_has_no_direct_effect_authority() -> None:
    signature = inspect.signature(run_owned_development_sec_root)
    assert tuple(signature.parameters) == (
        "reveal_store",
        "development_content_root_plan",
        "user_agent",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert {
        "request_sha256",
        "stage",
        "candidate",
        "url",
        "path",
        "digest",
        "bytes",
        "transport",
        "budget",
    }.isdisjoint(signature.parameters)


def test_public_model_runner_signatures_are_hash_only_and_keyword_only() -> None:
    development = inspect.signature(run_owned_development_model_batch)
    stage = inspect.signature(run_owned_stage_model_batch)
    assert tuple(development.parameters) == (
        "reveal_store",
        "development_root_scope_sha256",
    )
    assert tuple(stage.parameters) == ("reveal_store", "request_sha256")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for signature in (development, stage)
        for parameter in signature.parameters.values()
    )
    forbidden = {
        "candidate",
        "stage",
        "path",
        "text",
        "payload",
        "model",
        "transport",
        "session",
    }
    assert forbidden.isdisjoint(development.parameters)
    assert forbidden.isdisjoint(stage.parameters)


def test_public_feature_runner_signature_is_scope_only_and_keyword_only() -> None:
    signature = inspect.signature(run_owned_development_feature_batch)
    assert tuple(signature.parameters) == (
        "reveal_store",
        "development_root_scope_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert {
        "request_sha256",
        "stage",
        "candidate",
        "event",
        "rows",
        "snapshot",
        "label",
        "outcome",
        "holdout",
        "path",
        "bytes",
        "payload",
        "model",
        "transport",
        "session",
    }.isdisjoint(signature.parameters)


def test_public_label_runner_signature_is_scope_only_and_keyword_only() -> None:
    signature = inspect.signature(run_owned_development_label_batch)
    assert tuple(signature.parameters) == (
        "reveal_store",
        "development_root_scope_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert {
        "request_sha256",
        "stage",
        "candidate",
        "event",
        "rows",
        "snapshot",
        "feature_batch",
        "label",
        "outcome",
        "holdout",
        "path",
        "bytes",
        "payload",
        "model",
        "transport",
        "session",
    }.isdisjoint(signature.parameters)


def test_feature_runner_consumes_only_compact_causal_projection_and_is_non_authorizing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_benchmark.sec_filing_gemma_features as features_module
    import agent_benchmark.sec_filing_gemma_learner as learner_module
    import agent_benchmark.sec_filing_gemma_prediction_evidence as prediction_module
    import agent_benchmark.sec_filing_gemma_stage_verifier as verifier_module

    store = _new_store(tmp_path)
    plan = _development_feature_plan()
    projection = _development_feature_projection(plan)
    loader_calls: list[str] = []

    def load_projection(
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        loader_calls.append(development_root_scope_sha256)
        return copy.deepcopy(projection)

    store._load_owned_development_feature_inputs = load_projection
    feature_calls: list[dict[str, Any]] = []

    def build_feature(**kwargs: Any) -> dict[str, Any]:
        feature_calls.append(kwargs)
        prefix = kwargs["market_prefix"]
        assert set(kwargs) == {
            "market_prefix",
            "market_prefix_proof",
            "expected_market_prefix_proof_sha256",
            "universe_event_proof",
            "expected_universe_event_proof_sha256",
            "extraction_event_proof",
            "expected_extraction_event_proof_sha256",
        }
        assert len(prefix["lookback_rows"]) == 253
        assert prefix["lookback_rows"][-1]["session"] == prefix["decision_session"]
        assert all(
            row["session"] <= prefix["decision_session"]
            for row in prefix["lookback_rows"]
        )
        assert "stage_manifest" not in kwargs
        assert "future_market_rows" not in kwargs
        body = {
            "accession_number": prefix["decision_event_id"],
            "decision_session": prefix["decision_session"],
            "fit_eligible": True,
        }
        return {**body, "feature_row_sha256": canonical_sha256(body)}

    monkeypatch.setattr(runner_module, "build_sec_filing_gemma_feature_row", build_feature)
    monkeypatch.setattr(
        runner_module,
        "build_owned_development_feature_batch",
        lambda *, feature_assembly_plan, feature_rows: _fake_feature_batch(
            feature_assembly_plan, feature_rows
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "validate_owned_development_feature_batch",
        lambda batch, **_kwargs: batch["feature_batch_sha256"],
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("feature-only runner entered a forbidden path")

    for name in (
        "_acquire_owned_development_market_evidence",
        "probe_owned_ollama_runtime",
        "call_ollama_extractor_attempt",
        "_owned_transport_factory",
    ):
        monkeypatch.setattr(runner_module, name, forbidden)
    monkeypatch.setattr(
        features_module, "build_twenty_session_label_evidence", forbidden
    )
    monkeypatch.setattr(
        features_module, "validate_twenty_session_label_evidence", forbidden
    )
    monkeypatch.setattr(learner_module, "SecFilingGemmaTwoHeadLearner", forbidden)
    monkeypatch.setattr(prediction_module, "validate_prediction_prefix", forbidden)
    monkeypatch.setattr(
        verifier_module, "validate_market_snapshot_stage_replay", forbidden
    )
    monkeypatch.setattr(store, "_load_owned_development_model_event_inputs", forbidden)
    monkeypatch.setattr(store, "_load_authorized_model_stage_event_inputs", forbidden)
    assert {
        "build_twenty_session_label_evidence",
        "validate_twenty_session_label_evidence",
        "SecFilingGemmaTwoHeadLearner",
        "validate_prediction_prefix",
        "validate_market_snapshot_stage_replay",
        "validate_raw_scores_gates_and_ranking",
        "validate_sec_gemma_no_leverage_proof",
        "authoritative_prerequisite_validator",
    }.isdisjoint(runner_module.__dict__)

    result = run_owned_development_feature_batch(
        reveal_store=store,
        development_root_scope_sha256=plan["development_root_scope_sha256"],
    )
    assert loader_calls == [plan["development_root_scope_sha256"]]
    assert len(feature_calls) == 1
    assert result == _fake_feature_batch(plan, result["feature_rows"])
    assert result["feature_rows"][0]["fit_eligible"] is True
    for field in (
        "labels_included",
        "outcomes_included",
        "post_decision_market_rows_included",
        "training_membership_included",
        "learner_fit_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    ):
        assert result[field] is False

    forbidden_public_keys = {
        "lookback_rows",
        "observations",
        "raw_response_bytes_by_symbol",
        "artifact_bytes_by_symbol",
        "window_bytes_by_symbol",
        "current_normalized_text",
        "prior_same_form_normalized_text",
        "model_payload",
        "request_bytes_base64",
        "response_bytes_base64",
        "model_attempt_receipt",
        "label_evidence_sha256",
        "future_market_rows",
        "learner_state",
        "prediction_rows",
        "consumption_ledger",
    }
    observed_keys: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            observed_keys.update(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            assert not isinstance(value, bytes)

    walk(result)
    assert forbidden_public_keys.isdisjoint(observed_keys)


def test_feature_runner_rejects_extra_cross_root_reordered_and_malformed_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    scope = "9" * 64

    def forbidden_builder(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("invalid compact input reached the feature builder")

    monkeypatch.setattr(
        runner_module, "build_sec_filing_gemma_feature_row", forbidden_builder
    )

    def rejected(projection: dict[str, Any], match: str) -> None:
        store._load_owned_development_feature_inputs = (
            lambda **_kwargs: copy.deepcopy(projection)
        )
        with pytest.raises(SecFilingGemmaStageRunnerError, match=match):
            run_owned_development_feature_batch(
                reveal_store=store,
                development_root_scope_sha256=scope,
            )

    extra = _development_feature_projection(_development_feature_plan())
    extra["label_evidence"] = {}
    rejected(extra, "exact store projection")

    cross_root_plan = _development_feature_plan(scope_sha256="0" * 64)
    cross_root = _development_feature_projection(cross_root_plan)
    rejected(cross_root, "crossed their root scope")

    first = _development_feature_plan()["event_plan"][0]
    first_position = EXPECTED_MARKET_HISTORY_SESSIONS.index(
        first["availability_session"]
    )
    second = {
        "event_ordinal": 2,
        "accession_number": "0000320193-02-000002",
        "form": "10-K",
        "availability_session": EXPECTED_MARKET_HISTORY_SESSIONS[
            first_position + 20
        ],
        "sec_document_ordinal": 2,
    }
    ordered_plan = _development_feature_plan(event_plan=[first, second])
    reordered = _development_feature_projection(ordered_plan)
    reordered["events"].reverse()
    _rehash_development_feature_projection(reordered)
    rejected(reordered, "reordered or cross-event")

    malformed = _development_feature_projection(_development_feature_plan())
    del malformed["events"][0]["extraction_event_proof"][
        "extraction_event_proof_sha256"
    ]
    _rehash_development_feature_projection(malformed)
    rejected(malformed, "proof pin is unavailable")

    post_decision = _development_feature_projection(_development_feature_plan())
    decision = post_decision["events"][0]["event_plan_item"][
        "availability_session"
    ]
    decision_position = EXPECTED_MARKET_HISTORY_SESSIONS.index(decision)
    post_decision["events"][0]["market_prefix"]["lookback_rows"][-1][
        "session"
    ] = EXPECTED_MARKET_HISTORY_SESSIONS[decision_position + 1]
    _rehash_development_feature_projection(post_decision)
    rejected(post_decision, "post-decision row")

    tampered = _development_feature_projection(_development_feature_plan())
    tampered["feature_inputs_sha256"] = "0" * 64
    rejected(tampered, "checksum changed")


def test_label_runner_consumes_only_owned_compact_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    feature_plan = _development_feature_plan()
    feature_rows = [
        {
            "accession_number": feature_plan["event_plan"][0][
                "accession_number"
            ],
            "decision_session": feature_plan["event_plan"][0][
                "availability_session"
            ],
            "feature_row_sha256": hashlib.sha256(b"feature-row").hexdigest(),
        }
    ]
    source_batch = _fake_feature_batch(feature_plan, feature_rows)
    label_plan = _development_label_plan(feature_plan)
    projection = _development_label_projection(label_plan, source_batch)
    loader_calls: list[str] = []

    def load_projection(
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        loader_calls.append(development_root_scope_sha256)
        return copy.deepcopy(projection)

    store._load_owned_development_label_projection = load_projection
    build_calls: list[dict[str, Any]] = []

    def build_batch(**kwargs: Any) -> dict[str, Any]:
        build_calls.append(kwargs)
        assert set(kwargs) == {
            "label_assembly_plan",
            "source_feature_batch",
            "maturity_audit_rows",
            "label_evidence_rows",
        }
        body = {
            "development_root_scope_sha256": kwargs["label_assembly_plan"][
                "development_root_scope_sha256"
            ],
            "matured_label_count": len(kwargs["label_evidence_rows"]),
            "training_membership_included": False,
            "learner_fit_authorized": False,
            "prediction_authorized": False,
            "holdout_access_authorized": False,
            "ledger_mutation_authorized": False,
            "stage_promotion_authorized": False,
            "production_authorized": False,
        }
        return {**body, "label_batch_sha256": canonical_sha256(body)}

    monkeypatch.setattr(
        runner_module, "build_owned_development_label_batch", build_batch
    )
    monkeypatch.setattr(
        runner_module,
        "validate_owned_development_label_batch",
        lambda batch, **_kwargs: batch["label_batch_sha256"],
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("label runner entered a forbidden path")

    for name in (
        "_acquire_owned_development_market_evidence",
        "probe_owned_ollama_runtime",
        "call_ollama_extractor_attempt",
        "_owned_transport_factory",
        "build_sec_filing_gemma_feature_row",
    ):
        monkeypatch.setattr(runner_module, name, forbidden)

    result = run_owned_development_label_batch(
        reveal_store=store,
        development_root_scope_sha256=feature_plan[
            "development_root_scope_sha256"
        ],
    )
    assert loader_calls == [feature_plan["development_root_scope_sha256"]]
    assert len(build_calls) == 1
    assert result["matured_label_count"] == 1
    for field in (
        "training_membership_included",
        "learner_fit_authorized",
        "prediction_authorized",
        "holdout_access_authorized",
        "ledger_mutation_authorized",
        "stage_promotion_authorized",
        "production_authorized",
    ):
        assert result[field] is False


def test_label_runner_rejects_extra_cross_root_bad_counts_and_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    scope = "9" * 64
    feature_plan = _development_feature_plan(scope_sha256=scope)
    feature_rows = [
        {
            "accession_number": feature_plan["event_plan"][0][
                "accession_number"
            ],
            "decision_session": feature_plan["event_plan"][0][
                "availability_session"
            ],
            "feature_row_sha256": hashlib.sha256(b"feature-row").hexdigest(),
        }
    ]
    source_batch = _fake_feature_batch(feature_plan, feature_rows)
    plan = _development_label_plan(feature_plan)

    def forbidden_builder(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("invalid label projection reached the batch builder")

    monkeypatch.setattr(
        runner_module,
        "build_owned_development_label_batch",
        forbidden_builder,
    )

    def rejected(projection: dict[str, Any], match: str) -> None:
        store._load_owned_development_label_projection = (
            lambda **_kwargs: copy.deepcopy(projection)
        )
        with pytest.raises(SecFilingGemmaStageRunnerError, match=match):
            run_owned_development_label_batch(
                reveal_store=store,
                development_root_scope_sha256=scope,
            )

    extra = _development_label_projection(plan, source_batch)
    extra["training_membership"] = []
    rejected(extra, "not exact")

    other_feature_plan = _development_feature_plan(scope_sha256="0" * 64)
    other_source_batch = _fake_feature_batch(other_feature_plan, feature_rows)
    cross_root = _development_label_projection(
        _development_label_plan(other_feature_plan), other_source_batch
    )
    rejected(cross_root, "crossed its root scope")

    bad_count = _development_label_projection(plan, source_batch)
    bad_count["maturity_audit_rows"] = []
    _rehash_development_label_projection(bad_count)
    rejected(bad_count, "row counts changed")

    bool_count_source = copy.deepcopy(source_batch)
    bool_count_source["event_count"] = True
    source_body = {
        key: bool_count_source[key]
        for key in bool_count_source
        if key != "feature_batch_sha256"
    }
    bool_count_source["feature_batch_sha256"] = canonical_sha256(source_body)
    bool_count = _development_label_projection(plan, bool_count_source)
    rejected(bool_count, "source feature batch crossed its plan")

    tampered = _development_label_projection(plan, source_batch)
    tampered["label_projection_sha256"] = "0" * 64
    rejected(tampered, "checksum changed")


def test_training_membership_runner_returns_only_derived_public_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(
        run_owned_development_training_membership_batch
    )
    assert tuple(signature.parameters) == (
        "reveal_store",
        "development_root_scope_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    store = _new_store(tmp_path)
    scope = "9" * 64
    projection = _development_training_membership_projection(scope_sha256=scope)
    loader_calls: list[str] = []
    build_calls: list[dict[str, Any]] = []
    validation_calls: list[dict[str, Any]] = []

    def load_projection(*, development_root_scope_sha256: str) -> dict[str, Any]:
        loader_calls.append(development_root_scope_sha256)
        return copy.deepcopy(projection)

    store._load_owned_development_training_membership_projection = load_projection

    def validate_plan(
        plan: dict[str, Any],
        *,
        expected_training_membership_assembly_plan_sha256: str,
    ) -> str:
        assert expected_training_membership_assembly_plan_sha256 == plan[
            "training_membership_assembly_plan_sha256"
        ]
        return expected_training_membership_assembly_plan_sha256

    public_batch = {
        "schema_version": "aapl-sec-gemma-owned-development-training-membership-batch-v1",
        "training_view_count": 6,
        "training_views": [{"training_view_id": "fold_1"}],
        "compact_adjusted_open_paths_included": False,
        "full_market_rows_included": False,
        "learner_fit_authorized": False,
        "prediction_authorized": False,
        "training_membership_batch_sha256": "7" * 64,
    }

    def build_batch(**kwargs: Any) -> dict[str, Any]:
        build_calls.append(copy.deepcopy(kwargs))
        return copy.deepcopy(public_batch)

    def validate_batch(batch: dict[str, Any], **kwargs: Any) -> str:
        assert batch == public_batch
        validation_calls.append(copy.deepcopy(kwargs))
        return batch["training_membership_batch_sha256"]

    monkeypatch.setattr(
        runner_module,
        "validate_development_training_membership_assembly_plan",
        validate_plan,
    )
    monkeypatch.setattr(
        runner_module,
        "build_owned_development_training_membership_batch",
        build_batch,
    )
    monkeypatch.setattr(
        runner_module,
        "validate_owned_development_training_membership_batch",
        validate_batch,
    )

    result = run_owned_development_training_membership_batch(
        reveal_store=store,
        development_root_scope_sha256=scope,
    )
    assert result == public_batch
    assert result is not public_batch
    assert loader_calls == [scope]
    assert len(build_calls) == 1
    assert set(build_calls[0]) == {
        "training_membership_assembly_plan",
        "source_feature_batch",
        "source_label_batch",
    }
    assert len(validation_calls) == 1
    assert set(validation_calls[0]) == {
        "training_membership_assembly_plan",
        "expected_training_membership_assembly_plan_sha256",
        "source_feature_batch",
        "expected_source_feature_batch_sha256",
        "source_label_batch",
        "expected_source_label_batch_sha256",
        "expected_training_membership_batch_sha256",
    }
    encoded = json.dumps(result, sort_keys=True)
    assert "FEATURE-SOURCE-MUST-NOT-ESCAPE" not in encoded
    assert "LABEL-SOURCE-MUST-NOT-ESCAPE" not in encoded
    assert ".fit(" not in inspect.getsource(
        run_owned_development_training_membership_batch
    )
    assert ".predict" not in inspect.getsource(
        run_owned_development_training_membership_batch
    )


def test_training_membership_runner_rejects_projection_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    scope = "9" * 64

    def validate_plan(
        plan: dict[str, Any],
        *,
        expected_training_membership_assembly_plan_sha256: str,
    ) -> str:
        del plan
        return expected_training_membership_assembly_plan_sha256

    def forbidden_builder(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("invalid membership projection reached the builder")

    monkeypatch.setattr(
        runner_module,
        "validate_development_training_membership_assembly_plan",
        validate_plan,
    )
    monkeypatch.setattr(
        runner_module,
        "build_owned_development_training_membership_batch",
        forbidden_builder,
    )

    def rejected(projection: dict[str, Any], match: str) -> None:
        store._load_owned_development_training_membership_projection = (
            lambda **_kwargs: copy.deepcopy(projection)
        )
        with pytest.raises(SecFilingGemmaStageRunnerError, match=match):
            run_owned_development_training_membership_batch(
                reveal_store=store,
                development_root_scope_sha256=scope,
            )

    extra = _development_training_membership_projection(scope_sha256=scope)
    extra["raw_market_rows"] = []
    rejected(extra, "not exact")

    cross_root = _development_training_membership_projection(
        scope_sha256="0" * 64
    )
    rejected(cross_root, "crossed its root scope")

    bool_count = _development_training_membership_projection(scope_sha256=scope)
    bool_count["source_feature_batch"]["event_count"] = True
    _rehash_development_training_membership_projection(bool_count)
    rejected(bool_count, "sources crossed their plan")

    checksum = _development_training_membership_projection(scope_sha256=scope)
    checksum["membership_projection_sha256"] = "0" * 64
    rejected(checksum, "checksum changed")


def test_oof_learner_fit_runner_uses_only_one_owned_projection_and_returns_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(run_owned_development_oof_learner_fit_batch)
    assert tuple(signature.parameters) == (
        "reveal_store",
        "development_root_scope_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    store = _new_store(tmp_path)
    scope = "9" * 64
    projection = _development_oof_learner_fit_projection(scope_sha256=scope)
    loader_calls: list[str] = []
    build_calls: list[dict[str, Any]] = []
    validation_calls: list[dict[str, Any]] = []

    def load_projection(*, development_root_scope_sha256: str) -> dict[str, Any]:
        loader_calls.append(development_root_scope_sha256)
        return copy.deepcopy(projection)

    store._load_owned_development_oof_learner_fit_projection = load_projection

    def forbidden_loader(**_kwargs: Any) -> None:
        raise AssertionError("The public fit runner requested another store source")

    store._load_owned_development_training_membership_projection = forbidden_loader
    store._load_owned_development_feature_inputs = forbidden_loader
    store._load_owned_development_label_projection = forbidden_loader

    def validate_plan(
        plan: dict[str, Any],
        *,
        expected_development_oof_learner_fit_plan_sha256: str,
    ) -> str:
        assert plan is not projection["development_oof_learner_fit_plan"]
        assert expected_development_oof_learner_fit_plan_sha256 == "6" * 64
        return expected_development_oof_learner_fit_plan_sha256

    public_batch = {
        "schema_version": (
            "aapl-sec-gemma-owned-development-oof-learner-fit-batch-v1"
        ),
        "learner_fit_count": 10,
        "learner_state_count": 10,
        "deferred_training_view_state_included": False,
        "prediction_included": False,
        "prediction_authorized": False,
        "learner_fit_batch_sha256": "7" * 64,
    }

    def build_batch(**kwargs: Any) -> dict[str, Any]:
        build_calls.append(copy.deepcopy(kwargs))
        return copy.deepcopy(public_batch)

    def validate_batch(batch: dict[str, Any], **kwargs: Any) -> str:
        assert batch == public_batch
        validation_calls.append(copy.deepcopy(kwargs))
        return batch["learner_fit_batch_sha256"]

    monkeypatch.setattr(
        runner_module,
        "validate_development_oof_learner_fit_plan",
        validate_plan,
    )
    monkeypatch.setattr(
        runner_module,
        "build_owned_development_oof_learner_fit_batch",
        build_batch,
    )
    monkeypatch.setattr(
        runner_module,
        "validate_owned_development_oof_learner_fit_batch",
        validate_batch,
    )

    result = run_owned_development_oof_learner_fit_batch(
        reveal_store=store,
        development_root_scope_sha256=scope,
    )
    assert result == public_batch
    assert result is not public_batch
    assert loader_calls == [scope]
    assert len(build_calls) == 1
    assert set(build_calls[0]) == {
        "development_oof_learner_fit_plan",
        "expected_development_oof_learner_fit_plan_sha256",
        "source_training_membership_batch",
        "expected_source_training_membership_batch_sha256",
    }
    assert len(validation_calls) == 1
    assert set(validation_calls[0]) == {
        "development_oof_learner_fit_plan",
        "expected_development_oof_learner_fit_plan_sha256",
        "source_training_membership_batch",
        "expected_source_training_membership_batch_sha256",
        "expected_learner_fit_batch_sha256",
    }
    assert "MEMBERSHIP-SOURCE-MUST-NOT-ESCAPE" not in json.dumps(
        result, sort_keys=True
    )
    source = inspect.getsource(run_owned_development_oof_learner_fit_batch)
    assert ".predict" not in source


def test_oof_learner_fit_runner_rejects_projection_before_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    scope = "9" * 64

    def validate_plan(
        _plan: dict[str, Any],
        *,
        expected_development_oof_learner_fit_plan_sha256: str,
    ) -> str:
        return expected_development_oof_learner_fit_plan_sha256

    def forbidden_builder(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Invalid fit projection reached the fitter")

    monkeypatch.setattr(
        runner_module,
        "validate_development_oof_learner_fit_plan",
        validate_plan,
    )
    monkeypatch.setattr(
        runner_module,
        "build_owned_development_oof_learner_fit_batch",
        forbidden_builder,
    )

    def rejected(projection: dict[str, Any], match: str) -> None:
        store._load_owned_development_oof_learner_fit_projection = (
            lambda **_kwargs: copy.deepcopy(projection)
        )
        with pytest.raises(SecFilingGemmaStageRunnerError, match=match):
            run_owned_development_oof_learner_fit_batch(
                reveal_store=store,
                development_root_scope_sha256=scope,
            )

    extra = _development_oof_learner_fit_projection(scope_sha256=scope)
    extra["source_feature_batch"] = {}
    rejected(extra, "not exact")

    cross_root = _development_oof_learner_fit_projection(
        scope_sha256="0" * 64
    )
    rejected(cross_root, "crossed its ancestry")

    crossed_batch = _development_oof_learner_fit_projection(
        scope_sha256=scope
    )
    crossed_batch["source_training_membership_batch"][
        "training_membership_batch_sha256"
    ] = "8" * 64
    _rehash_development_oof_learner_fit_projection(crossed_batch)
    rejected(crossed_batch, "crossed its ancestry")

    checksum = _development_oof_learner_fit_projection(scope_sha256=scope)
    checksum["learner_fit_projection_sha256"] = "0" * 64
    rejected(checksum, "checksum changed")


def test_oof_prediction_runner_uses_one_compact_projection_and_returns_raw_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(run_owned_development_oof_prediction_batch)
    assert tuple(signature.parameters) == (
        "reveal_store",
        "development_root_scope_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    store = _new_store(tmp_path)
    scope = "9" * 64
    projection = _development_oof_prediction_projection(scope_sha256=scope)
    loader_calls: list[str] = []
    build_calls: list[dict[str, Any]] = []
    validation_calls: list[dict[str, Any]] = []

    def load_projection(*, development_root_scope_sha256: str) -> dict[str, Any]:
        loader_calls.append(development_root_scope_sha256)
        return copy.deepcopy(projection)

    store._load_owned_development_oof_prediction_projection = load_projection

    def forbidden_loader(**_kwargs: Any) -> None:
        raise AssertionError("The public prediction runner requested another source")

    store._load_owned_development_oof_learner_fit_projection = forbidden_loader
    store._load_owned_development_training_membership_projection = forbidden_loader
    store._load_owned_development_feature_inputs = forbidden_loader
    store._load_owned_development_label_projection = forbidden_loader

    def validate_plan(
        plan: dict[str, Any],
        *,
        expected_development_oof_prediction_plan_sha256: str,
    ) -> str:
        assert plan is not projection["development_oof_prediction_plan"]
        assert expected_development_oof_prediction_plan_sha256 == "6" * 64
        return expected_development_oof_prediction_plan_sha256

    monkeypatch.setattr(
        runner_module, "validate_development_oof_prediction_plan", validate_plan
    )
    monkeypatch.setattr(
        runner_module,
        "derive_development_oof_prediction_fold_model_specs",
        lambda _bundle: [{"fold_id": "FOLD-SPECS"}],
    )
    monkeypatch.setattr(
        runner_module,
        "derive_development_oof_prediction_input_specs",
        lambda _features: [{"prediction_ordinal": "INPUT-SPECS"}],
    )

    public_batch = {
        "schema_version": "aapl-sec-gemma-owned-development-oof-prediction-batch-v1",
        "prediction_count": 3,
        "available_prediction_count": 2,
        "unavailable_prediction_count": 1,
        "candidate_selection_authorized": False,
        "threshold_action_authorized": False,
        "prediction_batch_sha256": "f" * 64,
    }

    def build_batch(**kwargs: Any) -> dict[str, Any]:
        build_calls.append(copy.deepcopy(kwargs))
        return copy.deepcopy(public_batch)

    def validate_batch(batch: dict[str, Any], **kwargs: Any) -> str:
        assert batch == public_batch
        validation_calls.append(copy.deepcopy(kwargs))
        return batch["prediction_batch_sha256"]

    monkeypatch.setattr(
        runner_module, "build_owned_development_oof_prediction_batch", build_batch
    )
    monkeypatch.setattr(
        runner_module,
        "validate_owned_development_oof_prediction_batch",
        validate_batch,
    )

    result = run_owned_development_oof_prediction_batch(
        reveal_store=store,
        development_root_scope_sha256=scope,
    )
    assert result == public_batch
    assert result is not public_batch
    assert loader_calls == [scope]
    assert len(build_calls) == 1
    assert set(build_calls[0]) == {
        "development_oof_prediction_plan",
        "expected_development_oof_prediction_plan_sha256",
        "prediction_feature_batch",
        "expected_prediction_feature_batch_sha256",
        "prediction_fold_model_bundle",
        "expected_prediction_fold_model_bundle_sha256",
    }
    assert len(validation_calls) == 1
    assert set(validation_calls[0]) == {
        "development_oof_prediction_plan",
        "expected_development_oof_prediction_plan_sha256",
        "prediction_feature_batch",
        "expected_prediction_feature_batch_sha256",
        "prediction_fold_model_bundle",
        "expected_prediction_fold_model_bundle_sha256",
        "expected_prediction_batch_sha256",
    }
    serialized = json.dumps(result, sort_keys=True)
    assert "ONLY-TEN-COMPACT-STATES-MAY-ESCAPE" not in serialized
    assert "NO-POST-2018-FEATURES-MAY-ESCAPE" not in serialized
    source = inspect.getsource(run_owned_development_oof_prediction_batch)
    assert ".fit" not in source
    assert ".predict" not in source


def test_oof_prediction_runner_rejects_crossed_projection_before_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    scope = "9" * 64

    def validate_plan(
        _plan: dict[str, Any],
        *,
        expected_development_oof_prediction_plan_sha256: str,
    ) -> str:
        return expected_development_oof_prediction_plan_sha256

    monkeypatch.setattr(
        runner_module, "validate_development_oof_prediction_plan", validate_plan
    )
    monkeypatch.setattr(
        runner_module,
        "derive_development_oof_prediction_fold_model_specs",
        lambda _bundle: [{"fold_id": "FOLD-SPECS"}],
    )
    monkeypatch.setattr(
        runner_module,
        "derive_development_oof_prediction_input_specs",
        lambda _features: [{"prediction_ordinal": "INPUT-SPECS"}],
    )

    def forbidden_builder(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Invalid projection reached prediction arithmetic")

    monkeypatch.setattr(
        runner_module,
        "build_owned_development_oof_prediction_batch",
        forbidden_builder,
    )

    def rejected(projection: dict[str, Any], match: str) -> None:
        store._load_owned_development_oof_prediction_projection = (
            lambda **_kwargs: copy.deepcopy(projection)
        )
        with pytest.raises(SecFilingGemmaStageRunnerError, match=match):
            run_owned_development_oof_prediction_batch(
                reveal_store=store,
                development_root_scope_sha256=scope,
            )

    extra = _development_oof_prediction_projection(scope_sha256=scope)
    extra["source_training_membership_batch"] = {}
    rejected(extra, "not exact")

    cross_root = _development_oof_prediction_projection(scope_sha256="0" * 64)
    rejected(cross_root, "crossed its ancestry")

    crossed_bundle = _development_oof_prediction_projection(scope_sha256=scope)
    crossed_bundle["prediction_fold_model_bundle"][
        "source_learner_fit_batch_sha256"
    ] = "0" * 64
    _rehash_development_oof_prediction_projection(crossed_bundle)
    rejected(crossed_bundle, "crossed its ancestry")

    checksum = _development_oof_prediction_projection(scope_sha256=scope)
    checksum["prediction_projection_sha256"] = "0" * 64
    rejected(checksum, "checksum changed")


def test_policy_replay_runner_uses_one_exact_projection_and_rebuild_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = inspect.signature(run_owned_development_policy_replay_batch)
    assert tuple(signature.parameters) == (
        "reveal_store",
        "development_root_scope_sha256",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    store = _new_store(tmp_path)
    scope = "9" * 64
    projection = _development_policy_replay_projection(scope_sha256=scope)
    assert set(projection) == {
        "schema_version",
        "development_policy_replay_plan",
        "source_development_oof_prediction_batch",
        "policy_replay_projection_sha256",
    }
    assert (
        projection["schema_version"]
        == OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION
    )
    assert projection["policy_replay_projection_sha256"] == canonical_sha256(
        {
            key: projection[key]
            for key in projection
            if key != "policy_replay_projection_sha256"
        }
    )
    loader_calls: list[str] = []

    def load_projection(
        owned_store: SecFilingGemmaRevealStore,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        assert owned_store is store
        loader_calls.append(development_root_scope_sha256)
        if len(loader_calls) != 1:
            raise AssertionError("Policy replay projection was loaded more than once")
        return copy.deepcopy(projection)

    monkeypatch.setattr(
        SecFilingGemmaRevealStore,
        "_load_owned_development_policy_replay_projection",
        load_projection,
    )

    def forbidden_loader(_owned_store: SecFilingGemmaRevealStore, **_kwargs: Any) -> None:
        raise AssertionError("Policy replay runner requested another store projection")

    for loader_name in (
        "_load_owned_development_oof_prediction_projection",
        "_load_owned_development_oof_learner_fit_projection",
        "_load_owned_development_training_membership_projection",
        "_load_owned_development_feature_inputs",
        "_load_owned_development_label_projection",
    ):
        monkeypatch.setattr(
            SecFilingGemmaRevealStore,
            loader_name,
            forbidden_loader,
        )

    plan_calls: list[dict[str, Any]] = []
    source_calls: list[dict[str, Any]] = []
    derived_calls: list[dict[str, Any]] = []

    def validate_plan(
        plan: dict[str, Any],
        *,
        expected_development_policy_replay_plan_sha256: str,
    ) -> str:
        plan_calls.append(copy.deepcopy(plan))
        assert expected_development_policy_replay_plan_sha256 == "6" * 64
        return expected_development_policy_replay_plan_sha256

    def validate_source(
        source: dict[str, Any],
        *,
        expected_prediction_batch_sha256: str,
    ) -> str:
        source_calls.append(copy.deepcopy(source))
        assert expected_prediction_batch_sha256 == "2" * 64
        return expected_prediction_batch_sha256

    def derive_specs(
        source: dict[str, Any],
        *,
        expected_source_prediction_batch_sha256: str,
    ) -> list[dict[str, Any]]:
        derived_calls.append(copy.deepcopy(source))
        assert expected_source_prediction_batch_sha256 == "2" * 64
        return copy.deepcopy(
            projection["development_policy_replay_plan"][
                "policy_replay_input_specs"
            ]
        )

    monkeypatch.setattr(
        runner_module, "validate_development_policy_replay_plan", validate_plan
    )
    monkeypatch.setattr(
        runner_module,
        "validate_owned_development_oof_prediction_batch_structure",
        validate_source,
    )
    monkeypatch.setattr(
        runner_module, "derive_development_policy_replay_input_specs", derive_specs
    )

    built_batch = {
        "schema_version": "test-owned-development-policy-replay-batch-v1",
        "policy_replay_batch_sha256": "f" * 64,
        "labels_included": False,
        "outcomes_included": False,
        "market_prices_included": False,
        "model_transport_authorized": False,
        "network_access_authorized": False,
        "learner_fit_authorized": False,
        "numeric_prediction_authorized": False,
        "threshold_comparison_rule": (
            "probability_gte_and_expected_edge_gte"
        ),
        "cash_episode_rule": (
            "accepted_after_close_fill_t_plus_1_exit_t_plus_21_fixed_20_session_"
            "cash_episode_never_extend_scheduled_or_active_episode"
        ),
        "unavailable_prediction_rule": (
            "unavailable_prediction_starts_no_new_cash_episode_"
            "existing_episode_keeps_original_exit"
        ),
        "policy_replay_order_rule": (
            "source_raw_prediction_ordinal_ascending_exactly_once"
        ),
    }
    build_calls: list[dict[str, Any]] = []
    rebuild_validation_calls: list[dict[str, Any]] = []

    def build_batch(**kwargs: Any) -> dict[str, Any]:
        build_calls.append(copy.deepcopy(kwargs))
        return built_batch

    def validate_batch(batch: dict[str, Any], **kwargs: Any) -> str:
        assert batch is built_batch
        rebuild_validation_calls.append(copy.deepcopy(kwargs))
        return batch["policy_replay_batch_sha256"]

    monkeypatch.setattr(
        runner_module, "build_owned_development_policy_replay_batch", build_batch
    )
    monkeypatch.setattr(
        runner_module,
        "validate_owned_development_policy_replay_batch",
        validate_batch,
    )

    result = run_owned_development_policy_replay_batch(
        reveal_store=store,
        development_root_scope_sha256=scope,
    )
    assert loader_calls == [scope]
    assert len(plan_calls) == len(source_calls) == len(derived_calls) == 1
    assert len(build_calls) == len(rebuild_validation_calls) == 1
    assert set(build_calls[0]) == {
        "source_prediction_batch",
        "expected_source_prediction_batch_sha256",
        "expected_development_policy_replay_plan_sha256",
        "expected_source_prediction_projection_sha256",
    }
    assert build_calls[0]["expected_source_prediction_batch_sha256"] == "2" * 64
    assert (
        build_calls[0]["expected_development_policy_replay_plan_sha256"]
        == "6" * 64
    )
    assert (
        build_calls[0]["expected_source_prediction_projection_sha256"]
        == "7" * 64
    )
    assert set(rebuild_validation_calls[0]) == {
        "source_prediction_batch",
        "expected_source_prediction_batch_sha256",
        "expected_development_policy_replay_plan_sha256",
        "expected_source_prediction_projection_sha256",
        "expected_policy_replay_batch_sha256",
    }
    assert result == built_batch
    assert result is not built_batch
    result["mutated_by_caller"] = True
    assert "mutated_by_caller" not in built_batch
    assert "mutated_by_caller" not in projection

    serialized_projection = json.dumps(projection, sort_keys=True)
    for forbidden in (
        "labels",
        "outcomes",
        "market_prices",
        "learner_states",
        "training_feature_matrices",
        "adjusted_open",
        "adjusted_close",
    ):
        assert forbidden not in serialized_projection
    source = inspect.getsource(run_owned_development_policy_replay_batch)
    for forbidden in (
        ".fit(",
        ".predict(",
        "call_ollama",
        "network",
        "transport",
        "label",
        "outcome",
        "market",
    ):
        assert forbidden not in source

    store._load_owned_development_policy_replay_projection = (
        lambda **_kwargs: copy.deepcopy(projection)
    )
    with pytest.raises(TypeError, match="must not shadow"):
        run_owned_development_policy_replay_batch(
            reveal_store=store,
            development_root_scope_sha256=scope,
        )


def test_policy_replay_runner_composes_real_plan_batch_builder_and_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production policy schema across the runner boundary."""

    store = _new_store(tmp_path)
    source = policy_replay_scaffold._raw_batch()
    input_specs = runner_module.derive_development_policy_replay_input_specs(
        source,
        expected_source_prediction_batch_sha256=source[
            "prediction_batch_sha256"
        ],
    )
    _state, _tip, _prediction_plan, _specs, plan = (
        authorization_scaffold._development_policy_replay_plan_fixture()
    )
    plan.update(
        {
            "development_root_scope_sha256": source[
                "development_root_scope_sha256"
            ],
            "source_development_oof_prediction_plan_sha256": source[
                "development_oof_prediction_plan_sha256"
            ],
            "source_development_oof_prediction_batch_sha256": source[
                "prediction_batch_sha256"
            ],
            "source_raw_prediction_rows_sha256": source[
                "raw_prediction_rows_sha256"
            ],
            "source_raw_prediction_tip_sha256": source[
                "raw_prediction_tip_sha256"
            ],
            "source_raw_prediction_row_count": source[
                "prediction_event_count"
            ],
            "policy_replay_input_count": len(input_specs),
            "policy_replay_input_specs": input_specs,
            "policy_replay_input_specs_sha256": canonical_sha256(input_specs),
            "contract_sha256": source["contract_sha256"],
            "candidate_sha256": source["candidate_sha256"],
            "corpus_universe_sha256": source["corpus_universe_sha256"],
            "calendar_sessions_sha256": source["calendar_sessions_sha256"],
            "development_cutoff_session": source[
                "development_cutoff_session"
            ],
            "model_variant_count": source["model_variant_count"],
            "model_variant_ids": source["model_variant_ids"],
        }
    )
    plan_body = {
        key: value
        for key, value in plan.items()
        if key != "development_policy_replay_plan_sha256"
    }
    plan["development_policy_replay_plan_sha256"] = canonical_sha256(
        plan_body
    )
    assert runner_module.validate_development_policy_replay_plan(
        plan,
        expected_development_policy_replay_plan_sha256=plan[
            "development_policy_replay_plan_sha256"
        ],
    ) == plan["development_policy_replay_plan_sha256"]

    projection_body = {
        "schema_version": (
            OWNED_DEVELOPMENT_POLICY_REPLAY_PROJECTION_SCHEMA_VERSION
        ),
        "development_policy_replay_plan": plan,
        "source_development_oof_prediction_batch": source,
    }
    projection = {
        **projection_body,
        "policy_replay_projection_sha256": canonical_sha256(projection_body),
    }

    def load_projection(
        owned_store: SecFilingGemmaRevealStore,
        *,
        development_root_scope_sha256: str,
    ) -> dict[str, Any]:
        assert owned_store is store
        assert development_root_scope_sha256 == source[
            "development_root_scope_sha256"
        ]
        return copy.deepcopy(projection)

    monkeypatch.setattr(
        SecFilingGemmaRevealStore,
        "_load_owned_development_policy_replay_projection",
        load_projection,
    )

    result = run_owned_development_policy_replay_batch(
        reveal_store=store,
        development_root_scope_sha256=source[
            "development_root_scope_sha256"
        ],
    )
    assert result["unavailable_prediction_rule"] == plan[
        "unavailable_prediction_rule"
    ]
    assert result["policy_replay_order_rule"] == plan[
        "policy_replay_order_rule"
    ]
    assert runner_module.validate_owned_development_policy_replay_batch(
        result,
        source_prediction_batch=source,
        expected_source_prediction_batch_sha256=source[
            "prediction_batch_sha256"
        ],
        expected_development_policy_replay_plan_sha256=plan[
            "development_policy_replay_plan_sha256"
        ],
        expected_source_prediction_projection_sha256=plan[
            "source_development_oof_prediction_projection_sha256"
        ],
        expected_policy_replay_batch_sha256=result[
            "policy_replay_batch_sha256"
        ],
    ) == result["policy_replay_batch_sha256"]


def test_policy_replay_runner_rejects_every_crossing_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    scope = "9" * 64
    input_specs = [{"input_ordinal": "POLICY-INPUT-SPECS"}]

    def validate_plan(
        _plan: dict[str, Any],
        *,
        expected_development_policy_replay_plan_sha256: str,
    ) -> str:
        return expected_development_policy_replay_plan_sha256

    def validate_source(
        _source: dict[str, Any],
        *,
        expected_prediction_batch_sha256: str,
    ) -> str:
        return expected_prediction_batch_sha256

    monkeypatch.setattr(
        runner_module, "validate_development_policy_replay_plan", validate_plan
    )
    monkeypatch.setattr(
        runner_module,
        "validate_owned_development_oof_prediction_batch_structure",
        validate_source,
    )
    monkeypatch.setattr(
        runner_module,
        "derive_development_policy_replay_input_specs",
        lambda _source, **_kwargs: copy.deepcopy(input_specs),
    )

    def forbidden_builder(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("Crossed policy projection reached replay arithmetic")

    monkeypatch.setattr(
        runner_module,
        "build_owned_development_policy_replay_batch",
        forbidden_builder,
    )

    def rejected(projection: dict[str, Any], match: str) -> None:
        calls = 0

        def load_once(
            owned_store: SecFilingGemmaRevealStore,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            nonlocal calls
            assert owned_store is store
            calls += 1
            if calls > 1:
                raise AssertionError("Invalid projection triggered a duplicate load")
            return copy.deepcopy(projection)

        monkeypatch.setattr(
            SecFilingGemmaRevealStore,
            "_load_owned_development_policy_replay_projection",
            load_once,
        )
        with pytest.raises(SecFilingGemmaStageRunnerError, match=match):
            run_owned_development_policy_replay_batch(
                reveal_store=store,
                development_root_scope_sha256=scope,
            )
        assert calls == 1

    extra = _development_policy_replay_projection(scope_sha256=scope)
    extra["source_feature_batch"] = {}
    rejected(extra, "not exact")

    schema = _development_policy_replay_projection(scope_sha256=scope)
    schema["schema_version"] = "changed-schema"
    _rehash_development_policy_replay_projection(schema)
    rejected(schema, "schema changed")

    mutations: list[tuple[str, str, Any]] = [
        ("plan", "development_root_scope_sha256", "0" * 64),
        ("source", "development_root_scope_sha256", "0" * 64),
        ("source", "development_oof_prediction_plan_sha256", "0" * 64),
        ("source", "prediction_batch_sha256", "0" * 64),
        ("source", "raw_prediction_rows_sha256", "0" * 64),
        ("source", "raw_prediction_tip_sha256", "0" * 64),
        ("source", "prediction_event_count", 2),
        ("plan", "contract_sha256", "0" * 64),
        ("plan", "candidate_sha256", "0" * 64),
        ("plan", "corpus_universe_sha256", "0" * 64),
        ("plan", "calendar_sessions_sha256", "0" * 64),
        ("plan", "development_cutoff_session", "2017-12-29"),
        ("source", "model_variant_count", 3),
        ("source", "model_variant_ids", ["semantic", "changed"]),
    ]
    for container, field, replacement in mutations:
        crossed = _development_policy_replay_projection(scope_sha256=scope)
        key = (
            "development_policy_replay_plan"
            if container == "plan"
            else "source_development_oof_prediction_batch"
        )
        crossed[key][field] = replacement
        _rehash_development_policy_replay_projection(crossed)
        rejected(crossed, "crossed its ancestry")

    crossed_count = _development_policy_replay_projection(scope_sha256=scope)
    crossed_count["development_policy_replay_plan"][
        "source_raw_prediction_row_count"
    ] = 2
    _rehash_development_policy_replay_projection(crossed_count)
    rejected(crossed_count, "crossed its ancestry")

    crossed_specs = _development_policy_replay_projection(scope_sha256=scope)
    crossed_specs["development_policy_replay_plan"][
        "policy_replay_input_specs"
    ] = [{"input_ordinal": "CHANGED"}]
    crossed_specs["development_policy_replay_plan"][
        "policy_replay_input_specs_sha256"
    ] = canonical_sha256(
        crossed_specs["development_policy_replay_plan"][
            "policy_replay_input_specs"
        ]
    )
    _rehash_development_policy_replay_projection(crossed_specs)
    rejected(crossed_specs, "crossed its ancestry")

    checksum = _development_policy_replay_projection(scope_sha256=scope)
    checksum["policy_replay_projection_sha256"] = "0" * 64
    rejected(checksum, "checksum changed")


def test_runner_persists_exact_raw_normalized_and_canonical_batch_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    records, aborts = _install_created_claim_store(store)
    _install_plan(monkeypatch)
    transport = FakeTransport()
    factory_calls: list[str] = []
    _install_transport_factory(monkeypatch, transport, factory_calls)

    result = run_authorized_sec_stage(
        reveal_store=store,
        request_sha256=REQUEST_SHA256,
        user_agent=USER_AGENT,
    )

    assert result["claim"] == _claim()
    assert result["reader_receipt"]["claim_sha256"] == CLAIM_SHA256
    assert records == [REQUEST_SHA256]
    assert aborts == []
    assert factory_calls == ["factory"]
    assert transport.calls == [OFFICIAL_URL]

    directory = (
        store.store_directory
        / "stage_outputs"
        / CLAIM_SHA256
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    observed = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert observed["document-0001.raw"] == RAW_DOCUMENT
    assert b"Exact filing bytes & evidence." in observed[
        "document-0001.normalized.txt"
    ]
    receipts = json.loads(observed["request-receipts.json"])
    manifest = json.loads(observed["byte-manifest.json"])
    assert receipts[0]["requested_url"] == OFFICIAL_URL
    assert manifest["documents"][0]["raw_document_sha256"] == hashlib.sha256(
        RAW_DOCUMENT
    ).hexdigest()
    marker_bytes = observed[SEC_BATCH_COMPLETE_MARKER_FILENAME]
    marker = json.loads(marker_bytes)
    marker_body = {
        key: value for key, value in marker.items() if key != "marker_sha256"
    }
    assert marker["schema_version"] == SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION
    assert marker["request_sha256"] == REQUEST_SHA256
    assert marker["claim_sha256"] == CLAIM_SHA256
    assert marker["component_id"] == SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
    assert marker["marker_sha256"] == canonical_sha256(marker_body)
    assert marker_bytes == runner_module._canonical_marker_bytes(marker)
    indexed_names = [item["relative_path"] for item in marker["byte_index"]]
    assert indexed_names == [
        "document-0001.raw",
        "document-0001.normalized.txt",
        "request-receipts.json",
        "byte-manifest.json",
    ]
    for item in marker["byte_index"]:
        payload = observed[item["relative_path"]]
        assert item["byte_count"] == len(payload)
        assert item["sha256"] == hashlib.sha256(payload).hexdigest()
    assert USER_AGENT.encode() not in b"".join(observed.values())
    assert b"owner-contact@real-domain-for-tests.dev" not in b"".join(
        observed.values()
    )


def test_development_root_runner_persists_complete_universe_and_content_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    plan = _development_root_plan()
    records, aborts = _install_created_development_root_store(store, plan=plan)
    _install_development_root_plan(monkeypatch, plan)
    transport = FakeTransport(
        max_bytes=DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES
    )
    factory_calls: list[str] = []
    _install_development_root_transport_factory(
        monkeypatch,
        transport,
        factory_calls,
    )

    result = run_owned_development_sec_root(
        reveal_store=store,
        development_content_root_plan=plan,
        user_agent=USER_AGENT,
    )

    root_scope_sha256 = plan["development_root_scope_sha256"]
    assert result["claim"] == _development_root_claim(plan)
    assert result["reader_receipt"]["claim_sha256"] == CLAIM_SHA256
    assert records == [root_scope_sha256]
    assert aborts == []
    assert factory_calls == ["factory"]
    assert transport.calls == [OFFICIAL_URL]

    directory = (
        store.store_directory
        / "stage_outputs"
        / CLAIM_SHA256
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    observed = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert observed["document-0001.raw"] == RAW_DOCUMENT
    assert json.loads(observed["corpus-universe.json"]) == plan[
        "corpus_universe_manifest"
    ]
    assert observed["corpus-universe.json"] == runner_module._canonical_marker_bytes(
        plan["corpus_universe_manifest"]
    )
    content_manifest = json.loads(observed["development-content-manifest.json"])
    assert content_manifest["artifact_stage"] == "development"
    assert content_manifest["corpus_universe_sha256"] == DEVELOPMENT_UNIVERSE_SHA256
    assert content_manifest["document_count"] == 1
    assert content_manifest["documents"][0]["accession_number"] == ACCESSION
    assert content_manifest["documents"][0][
        "primary_document_sha256"
    ] == hashlib.sha256(RAW_DOCUMENT).hexdigest()
    assert observed[
        "development-content-manifest.json"
    ] == runner_module._canonical_marker_bytes(content_manifest)

    marker_bytes = observed[SEC_BATCH_COMPLETE_MARKER_FILENAME]
    marker = json.loads(marker_bytes)
    marker_body = {
        key: value for key, value in marker.items() if key != "marker_sha256"
    }
    assert set(marker) == {
        "schema_version",
        "development_root_scope_sha256",
        "claim_sha256",
        "candidate_sha256",
        "corpus_universe_sha256",
        "development_content_root_plan_sha256",
        "component_id",
        "development_content_manifest_sha256",
        "byte_index",
        "byte_index_sha256",
        "marker_sha256",
    }
    assert (
        marker["schema_version"]
        == DEVELOPMENT_SEC_ROOT_COMPLETE_MARKER_SCHEMA_VERSION
    )
    assert marker["development_root_scope_sha256"] == root_scope_sha256
    assert marker["claim_sha256"] == CLAIM_SHA256
    assert marker["candidate_sha256"] == DEVELOPMENT_CANDIDATE_SHA256
    assert marker["corpus_universe_sha256"] == DEVELOPMENT_UNIVERSE_SHA256
    assert marker["development_content_root_plan_sha256"] == plan[
        "development_content_root_plan_sha256"
    ]
    assert marker["component_id"] == DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
    assert marker["development_content_manifest_sha256"] == content_manifest[
        "content_manifest_sha256"
    ]
    assert marker["marker_sha256"] == canonical_sha256(marker_body)
    assert marker_bytes == runner_module._canonical_marker_bytes(marker)
    assert [item["relative_path"] for item in marker["byte_index"]] == [
        "document-0001.raw",
        "document-0001.normalized.txt",
        "request-receipts.json",
        "byte-manifest.json",
        "corpus-universe.json",
        "development-content-manifest.json",
    ]
    assert USER_AGENT.encode() not in b"".join(observed.values())


def test_completed_development_root_retry_replays_with_zero_network_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    plan = _development_root_plan()
    claim = _development_root_claim(plan)
    root_scope_sha256 = plan["development_root_scope_sha256"]
    receipt = {
        "development_root_scope_sha256": root_scope_sha256,
        "claim_sha256": CLAIM_SHA256,
        "receipt_sha256": "3" * 64,
    }
    replayed: list[str] = []
    store.claim_owned_development_sec_root_execution = lambda **_kwargs: {
        "claim": claim,
        "created": False,
        "reader_receipt": receipt,
        "abort": None,
    }
    store._record_owned_development_sec_root_reader_output = (
        lambda *, development_root_scope_sha256: (
            replayed.append(development_root_scope_sha256) or receipt
        )
    )
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed development root must perform zero I/O")
        ),
    )

    result = run_owned_development_sec_root(
        reveal_store=store,
        development_content_root_plan=plan,
        user_agent=USER_AGENT,
    )

    assert result == {"claim": claim, "reader_receipt": receipt}
    assert replayed == [root_scope_sha256]


def test_recovered_development_root_without_marker_aborts_with_zero_network_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    plan = _development_root_plan()
    claim = _development_root_claim(plan)
    root_scope_sha256 = plan["development_root_scope_sha256"]
    aborts: list[str] = []
    store.claim_owned_development_sec_root_execution = lambda **_kwargs: {
        "claim": claim,
        "created": False,
        "reader_receipt": None,
        "abort": None,
    }
    store._record_owned_development_sec_root_reader_output = (
        lambda *, development_root_scope_sha256: (_ for _ in ()).throw(
            FileNotFoundError(SEC_BATCH_COMPLETE_MARKER_FILENAME)
        )
    )
    store.abort_owned_development_sec_root_execution = (
        lambda *, development_root_scope_sha256, reason: aborts.append(reason)
    )
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("indeterminate development root must perform zero I/O")
        ),
    )

    with pytest.raises(SecFilingGemmaStageRunnerError, match="cannot be retried"):
        run_owned_development_sec_root(
            reveal_store=store,
            development_content_root_plan=plan,
            user_agent=USER_AGENT,
        )

    assert aborts == ["claim_recovered_without_terminal_receipt"]
    assert root_scope_sha256 == claim["development_root_scope_sha256"]


def test_development_root_crash_after_marker_recovers_with_zero_second_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    plan = _development_root_plan()
    _records, aborts = _install_created_development_root_store(store, plan=plan)
    _install_development_root_plan(monkeypatch, plan)
    transport = FakeTransport(
        max_bytes=DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES
    )
    factory_calls: list[str] = []
    _install_development_root_transport_factory(
        monkeypatch,
        transport,
        factory_calls,
    )
    root_scope_sha256 = plan["development_root_scope_sha256"]
    record_calls: list[str] = []

    def crash_then_record(
        *, development_root_scope_sha256: str
    ) -> dict[str, Any]:
        record_calls.append(development_root_scope_sha256)
        if len(record_calls) == 1:
            raise SystemExit("simulated development root crash after marker")
        marker = (
            store.store_directory
            / "stage_outputs"
            / CLAIM_SHA256
            / SEC_STAGE_COMPONENT_DIRECTORY_NAME
            / SEC_BATCH_COMPLETE_MARKER_FILENAME
        )
        assert marker.is_file()
        return {
            "development_root_scope_sha256": root_scope_sha256,
            "claim_sha256": CLAIM_SHA256,
            "receipt_sha256": "3" * 64,
        }

    store._record_owned_development_sec_root_reader_output = crash_then_record
    with pytest.raises(SystemExit, match="simulated development root crash"):
        run_owned_development_sec_root(
            reveal_store=store,
            development_content_root_plan=plan,
            user_agent=USER_AGENT,
        )
    assert transport.calls == [OFFICIAL_URL]
    assert factory_calls == ["factory"]
    assert aborts == []

    store.claim_owned_development_sec_root_execution = lambda **_kwargs: {
        "claim": _development_root_claim(plan),
        "created": False,
        "reader_receipt": None,
        "abort": None,
    }
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed development root must perform zero reader I/O")
        ),
    )
    result = run_owned_development_sec_root(
        reveal_store=store,
        development_content_root_plan=plan,
        user_agent=USER_AGENT,
    )
    assert result["reader_receipt"]["claim_sha256"] == CLAIM_SHA256
    assert record_calls == [root_scope_sha256, root_scope_sha256]
    assert transport.calls == [OFFICIAL_URL]
    assert factory_calls == ["factory"]
    assert aborts == []


def test_development_root_finalizer_error_leaves_sealed_marker_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    plan = _development_root_plan()
    _records, aborts = _install_created_development_root_store(store, plan=plan)
    _install_development_root_plan(monkeypatch, plan)
    transport = FakeTransport(
        max_bytes=DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES
    )
    factory_calls: list[str] = []
    _install_development_root_transport_factory(
        monkeypatch,
        transport,
        factory_calls,
    )
    root_scope_sha256 = plan["development_root_scope_sha256"]
    record_calls: list[str] = []

    def fail_then_record(
        *, development_root_scope_sha256: str
    ) -> dict[str, Any]:
        record_calls.append(development_root_scope_sha256)
        if len(record_calls) == 1:
            raise PermissionError("simulated transient receipt failure")
        return {
            "development_root_scope_sha256": root_scope_sha256,
            "claim_sha256": CLAIM_SHA256,
            "receipt_sha256": "3" * 64,
        }

    store._record_owned_development_sec_root_reader_output = fail_then_record
    with pytest.raises(
        SecFilingGemmaStageRunnerError,
        match="sealed marker remains recoverable",
    ):
        run_owned_development_sec_root(
            reveal_store=store,
            development_content_root_plan=plan,
            user_agent=USER_AGENT,
        )
    marker = (
        store.store_directory
        / "stage_outputs"
        / CLAIM_SHA256
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
        / SEC_BATCH_COMPLETE_MARKER_FILENAME
    )
    assert marker.is_file()
    assert aborts == []
    assert transport.calls == [OFFICIAL_URL]

    store.claim_owned_development_sec_root_execution = lambda **_kwargs: {
        "claim": _development_root_claim(plan),
        "created": False,
        "reader_receipt": None,
        "abort": None,
    }
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed development root must not refetch")
        ),
    )
    result = run_owned_development_sec_root(
        reveal_store=store,
        development_content_root_plan=plan,
        user_agent=USER_AGENT,
    )
    assert result["reader_receipt"]["claim_sha256"] == CLAIM_SHA256
    assert record_calls == [root_scope_sha256, root_scope_sha256]
    assert factory_calls == ["factory"]
    assert aborts == []


def test_development_root_lock_contention_cannot_claim_or_abort(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    plan = _development_root_plan()
    claim_calls: list[str] = []

    class UnavailableExecutionLock:
        def __enter__(self) -> None:
            raise TimeoutError("simulated concurrent owner")

        def __exit__(self, *_args: Any) -> None:
            return None

    store._owned_development_sec_root_execution_lock = (
        lambda **_kwargs: UnavailableExecutionLock()
    )
    store.claim_owned_development_sec_root_execution = lambda **_kwargs: (
        claim_calls.append("claim")
    )

    with pytest.raises(
        SecFilingGemmaStageRunnerError,
        match="already owned or cannot be locked",
    ):
        run_owned_development_sec_root(
            reveal_store=store,
            development_content_root_plan=plan,
            user_agent=USER_AGENT,
        )
    assert claim_calls == []


def test_completed_retry_replays_durable_receipt_with_zero_network_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _claim()
    receipt = {"claim_sha256": CLAIM_SHA256, "receipt_sha256": "3" * 64}
    aborts: list[str] = []
    replayed: list[str] = []
    store.claim_authorized_sec_stage_execution = lambda **_kwargs: {
        "claim": claim,
        "created": False,
        "reader_receipt": receipt,
        "abort": None,
    }
    store.abort_authorized_sec_stage_execution = (
        lambda *, request_sha256, reason: aborts.append(reason)
    )
    store._record_authorized_sec_stage_reader_output = lambda *, request_sha256: (
        replayed.append(request_sha256) or receipt
    )

    def forbidden_factory(**_kwargs: Any) -> Any:
        raise AssertionError("transport factory must not run on completed retry")

    monkeypatch.setattr(runner_module, "_owned_transport_factory", forbidden_factory)
    result = run_authorized_sec_stage(
        reveal_store=store,
        request_sha256=REQUEST_SHA256,
        user_agent=USER_AGENT,
    )
    assert result == {"claim": claim, "reader_receipt": receipt}
    assert replayed == [REQUEST_SHA256]
    assert aborts == []


def test_completed_retry_rejects_receipt_for_another_claim_without_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    store.claim_authorized_sec_stage_execution = lambda **_kwargs: {
        "claim": _claim(),
        "created": False,
        "reader_receipt": {
            "claim_sha256": "9" * 64,
            "receipt_sha256": "3" * 64,
        },
        "abort": None,
    }
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("crossed completed receipt must perform zero I/O")
        ),
    )
    with pytest.raises(SecFilingGemmaStageRunnerError, match="crossed"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )


def test_source_substitution_after_claim_blocks_transport_and_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    _records, aborts = _install_created_claim_store(store)
    _install_plan(monkeypatch)
    store._revalidate_authorized_sec_execution_sources = lambda _claim: (
        (_ for _ in ()).throw(RuntimeError("source changed after claim"))
    )

    def forbidden_factory(**_kwargs: Any) -> Any:
        raise AssertionError("transport factory must not run after source substitution")

    monkeypatch.setattr(runner_module, "_owned_transport_factory", forbidden_factory)
    with pytest.raises(SecFilingGemmaStageRunnerError, match="will not be retried"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )

    assert aborts == ["durable_output_verification_failed"]


def test_recovered_active_claim_aborts_and_performs_zero_reader_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    aborts: list[str] = []
    store.claim_authorized_sec_stage_execution = lambda **_kwargs: {
        "claim": _claim(),
        "created": False,
        "reader_receipt": None,
        "abort": None,
    }
    store.abort_authorized_sec_stage_execution = (
        lambda *, request_sha256, reason: aborts.append(reason)
    )
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("indeterminate claim must perform zero I/O")
        ),
    )

    with pytest.raises(SecFilingGemmaStageRunnerError, match="cannot be retried"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )
    assert aborts == ["claim_recovered_without_terminal_receipt"]


def test_crash_after_complete_marker_recovers_receipt_with_zero_second_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    _records, aborts = _install_created_claim_store(store)
    _install_plan(monkeypatch)
    transport = FakeTransport()
    factory_calls: list[str] = []
    _install_transport_factory(monkeypatch, transport, factory_calls)
    record_calls: list[str] = []

    def crash_then_record(*, request_sha256: str) -> dict[str, Any]:
        record_calls.append(request_sha256)
        if len(record_calls) == 1:
            raise SystemExit("simulated crash after marker before receipt CAS")
        marker = (
            store.store_directory
            / "stage_outputs"
            / CLAIM_SHA256
            / SEC_STAGE_COMPONENT_DIRECTORY_NAME
            / SEC_BATCH_COMPLETE_MARKER_FILENAME
        )
        assert marker.is_file()
        return {"claim_sha256": CLAIM_SHA256, "receipt_sha256": "3" * 64}

    store._record_authorized_sec_stage_reader_output = crash_then_record
    with pytest.raises(SystemExit, match="simulated crash"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )
    assert transport.calls == [OFFICIAL_URL]
    assert factory_calls == ["factory"]
    assert aborts == []

    store.claim_authorized_sec_stage_execution = lambda **_kwargs: {
        "claim": _claim(),
        "created": False,
        "reader_receipt": None,
        "abort": None,
    }
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("sealed marker recovery must perform zero reader I/O")
        ),
    )
    result = run_authorized_sec_stage(
        reveal_store=store,
        request_sha256=REQUEST_SHA256,
        user_agent=USER_AGENT,
    )
    assert result["reader_receipt"]["claim_sha256"] == CLAIM_SHA256
    assert record_calls == [REQUEST_SHA256, REQUEST_SHA256]
    assert transport.calls == [OFFICIAL_URL]
    assert factory_calls == ["factory"]
    assert aborts == []


def test_private_production_factory_uses_typed_strict_transport_budgets(
    tmp_path: Path,
) -> None:
    outer = SecCorpusBudget(
        clock=lambda: 0.0,
        max_requests=1,
        max_bytes=100_000,
        max_seconds=10.0,
    )
    inner = SecCorpusBudget(
        clock=lambda: 0.0,
        max_requests=1,
        max_bytes=100_000,
        max_seconds=10.0,
    )
    with runner_module._owned_transport_factory(
        user_agent=USER_AGENT,
        transport_budget=inner,
        cache_directory=tmp_path / "unused-disabled-cache",
    ) as transport:
        observed_hash, security = corpus_module._prepare_transport(
            transport,
            user_agent=USER_AGENT,
            budget=outer,
        )
    assert observed_hash == validate_sec_user_agent(USER_AGENT).sha256
    assert security["transport_max_requests"] == 1
    assert security["transport_max_bytes"] == 100_000
    assert security["transport_max_seconds"] == 10.0
    assert security["trust_env"] is False
    assert security["proxies"] is False
    assert security["max_retries"] == 0
    assert security["max_redirects"] == 0
    assert not (tmp_path / "unused-disabled-cache").exists()


def test_invalid_persisted_grant_aborts_before_transport_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    _records, aborts = _install_created_claim_store(store)
    factory_calls: list[str] = []

    def forbidden_factory(**_kwargs: Any) -> Any:
        factory_calls.append("called")
        raise AssertionError("invalid grant must not construct a transport")

    monkeypatch.setattr(runner_module, "_owned_transport_factory", forbidden_factory)
    with pytest.raises(SecFilingGemmaStageRunnerError, match="will not be retried"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )
    assert factory_calls == []
    assert aborts == ["durable_output_verification_failed"]


def test_existing_output_extra_aborts_before_transport_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    _records, aborts = _install_created_claim_store(store)
    _install_plan(monkeypatch)
    directory = (
        store.store_directory
        / "stage_outputs"
        / CLAIM_SHA256
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    directory.mkdir(parents=True)
    (directory / "unexpected.bin").write_bytes(b"collision")
    factory_calls: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: factory_calls.append("called"),
    )

    with pytest.raises(SecFilingGemmaStageRunnerError, match="will not be retried"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )
    assert factory_calls == []
    assert aborts == ["durable_output_verification_failed"]
    assert (directory / "unexpected.bin").read_bytes() == b"collision"


def test_effect_failure_is_redacted_and_terminally_aborted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    records, aborts = _install_created_claim_store(store)
    _install_plan(monkeypatch)
    transport = FakeTransport(fail=True)
    factory_calls: list[str] = []
    _install_transport_factory(monkeypatch, transport, factory_calls)

    with pytest.raises(SecFilingGemmaStageRunnerError) as caught:
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )
    assert USER_AGENT not in str(caught.value)
    assert "owner-contact@real-domain-for-tests.dev" not in str(caught.value)
    assert factory_calls == ["factory"]
    assert transport.calls == [OFFICIAL_URL]
    assert records == []
    assert aborts == ["external_effect_failed_or_completion_unknown"]


def test_unsafe_claim_namespace_aborts_without_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    records, aborts = _install_created_claim_store(store)
    _install_plan(monkeypatch)
    unsafe = _claim()
    unsafe["output_namespace"] = "../escape"
    store.claim_authorized_sec_stage_execution = lambda **_kwargs: {
        "claim": unsafe,
        "created": True,
        "reader_receipt": None,
        "abort": None,
    }
    factory_calls: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: factory_calls.append("called"),
    )

    with pytest.raises(SecFilingGemmaStageRunnerError):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )
    assert factory_calls == []
    assert records == []
    assert aborts == ["durable_output_verification_failed"]
    assert not (store.store_directory.parent / "escape").exists()


def test_noncanonical_private_contact_is_rejected_before_claim_or_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim_calls: list[str] = []
    factory_calls: list[str] = []
    store.claim_authorized_sec_stage_execution = lambda **kwargs: claim_calls.append(
        kwargs["request_sha256"]
    )
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: factory_calls.append("called"),
    )

    with pytest.raises(
        SecFilingGemmaStageRunnerError,
        match="validated before claiming",
    ) as caught:
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=f" {USER_AGENT}",
        )

    assert claim_calls == []
    assert factory_calls == []
    assert USER_AGENT not in str(caught.value)
    assert not (store.store_directory / "stage_outputs").exists()


def test_precreated_empty_component_directory_aborts_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    _records, aborts = _install_created_claim_store(store)
    _install_plan(monkeypatch)
    directory = (
        store.store_directory
        / "stage_outputs"
        / CLAIM_SHA256
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    directory.mkdir(parents=True)
    factory_calls: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: factory_calls.append("called"),
    )

    with pytest.raises(SecFilingGemmaStageRunnerError, match="will not be retried"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )

    assert factory_calls == []
    assert aborts == ["durable_output_verification_failed"]
    assert directory.is_dir()
    assert list(directory.iterdir()) == []


def test_forged_overwide_component_plan_aborts_before_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    _records, aborts = _install_created_claim_store(store)
    overwide = _component_plan()
    overwide["max_bytes"] = OWNED_SEC_RAW_BATCH_MAX_BYTES + 1
    monkeypatch.setattr(
        runner_module,
        "_load_component_plan",
        lambda _store, *, request_sha256, claim: overwide,
    )
    factory_calls: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_owned_transport_factory",
        lambda **_kwargs: factory_calls.append("called"),
    )

    with pytest.raises(SecFilingGemmaStageRunnerError, match="will not be retried"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )

    assert factory_calls == []
    assert aborts == ["durable_output_verification_failed"]
    assert not (store.store_directory / "stage_outputs").exists()


def test_actual_document_bytes_reject_self_consistent_forged_receipts_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    records, aborts = _install_created_claim_store(store)
    _install_plan(monkeypatch)
    plan = _component_plan()
    budget = SecCorpusBudget(
        clock=lambda: 0.0,
        max_requests=1,
        max_bytes=100_000,
        max_seconds=10.0,
    )
    genuine = corpus_module._acquire_authenticated_stage_access_document_batch(
        authenticated_document_plan=plan["documents"],
        transport=FakeTransport(),
        user_agent=USER_AGENT,
        budget=budget,
    )
    receipts = json.loads(genuine.request_receipts_json)
    receipts[0]["content_sha256"] = f"sha256:{'f' * 64}"
    receipt_body = {
        key: value
        for key, value in receipts[0].items()
        if key != "request_receipt_sha256"
    }
    receipts[0]["request_receipt_sha256"] = canonical_sha256(receipt_body)
    manifest = json.loads(genuine.byte_manifest_json)
    manifest["documents"][0]["raw_document_sha256"] = "f" * 64
    manifest["documents"][0]["request_receipt_sha256"] = receipts[0][
        "request_receipt_sha256"
    ]
    manifest["request_receipts_sha256"] = canonical_sha256(receipts)
    manifest_body = {
        key: value
        for key, value in manifest.items()
        if key != "byte_manifest_sha256"
    }
    manifest["byte_manifest_sha256"] = canonical_sha256(manifest_body)

    def canonical_bytes(value: Any) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    forged = corpus_module._AuthenticatedStageAccessDocumentBatch(
        documents=genuine.documents,
        request_receipts=tuple(
            corpus_module._deep_freeze(receipt) for receipt in receipts
        ),
        byte_manifest=corpus_module._deep_freeze(manifest),
        request_receipts_json=canonical_bytes(receipts),
        byte_manifest_json=canonical_bytes(manifest),
    )
    monkeypatch.setattr(
        runner_module,
        "_acquire_authenticated_stage_access_document_batch",
        lambda **_kwargs: forged,
    )
    factory_calls: list[str] = []
    _install_transport_factory(
        monkeypatch,
        FakeTransport(),
        factory_calls,
    )

    with pytest.raises(SecFilingGemmaStageRunnerError, match="will not be retried"):
        run_authorized_sec_stage(
            reveal_store=store,
            request_sha256=REQUEST_SHA256,
            user_agent=USER_AGENT,
        )

    component_directory = (
        store.store_directory
        / "stage_outputs"
        / CLAIM_SHA256
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    assert factory_calls == ["factory"]
    assert records == []
    assert aborts == ["durable_output_verification_failed"]
    assert component_directory.is_dir()
    assert list(component_directory.iterdir()) == []


def test_fresh_stage_model_batch_uses_event_plan_and_fsyncs_intent_before_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    request_hash = "a" * 64
    candidate_hash = "b" * 64
    sec_receipt_hash = "c" * 64
    carry_receipt_hash = "d" * 64
    preprocessor_hash = "e" * 64
    current_accessions = (
        "0000320193-20-000002",
        "0000320193-20-000003",
    )
    prior_accessions = (
        "0000320193-19-000001",
        "0000320193-19-000002",
    )
    event_plan = [
        {
            "event_ordinal": 1,
            "accession_number": current_accessions[0],
            "form": "10-K",
            "availability_session": "2020-01-02",
            "sec_document_ordinal": 2,
        },
        {
            "event_ordinal": 2,
            "accession_number": current_accessions[1],
            "form": "10-Q",
            "availability_session": "2020-01-03",
            "sec_document_ordinal": 1,
        },
    ]
    limits = {
        "model_call_count": 2,
        "maximum_model_seconds": 3_600,
        "redirects": 0,
        "retries": 0,
        "pull_attempts": 0,
        "repair_attempts": 0,
        "streaming": False,
        "thinking": False,
    }
    source_hashes = {"preprocessor": preprocessor_hash}
    claim_body = {
        "request_sha256": request_hash,
        "candidate_sha256": candidate_hash,
        "authorized_stage": "intermediate",
        "output_namespace": "model-stage-test",
        "corpus_universe_sha256": "f" * 64,
        "stage_sec_reader_receipt_sha256": sec_receipt_hash,
        "carry_in_reader_receipt_sha256": carry_receipt_hash,
        "event_count": 2,
        "event_plan": event_plan,
        "event_plan_sha256": canonical_sha256(event_plan),
        "identity_lexicon_sha256": canonical_sha256(
            list(runner_module.CANONICAL_IDENTITY_LEXICON)
        ),
        "execution_source_hashes": source_hashes,
        "execution_source_hashes_sha256": canonical_sha256(source_hashes),
        "model_name": "gemma4:12b",
        "model_digest": "1" * 64,
        "runtime_fingerprint_sha256": "2" * 64,
        "model_transport_sha256": "3" * 64,
        "model_runtime_limits": limits,
        "model_runtime_limits_sha256": canonical_sha256(limits),
        "model_component_id": "owned_stage_gemma_model_batch",
    }
    claim = {**claim_body, "claim_sha256": canonical_sha256(claim_body)}

    current_payloads = (b"Current annual filing text.", b"Current quarter filing text.")
    prior_payloads = (b"Prior annual filing text.", b"Prior quarter filing text.")
    events: list[dict[str, Any]] = []
    universe_records: list[dict[str, Any]] = []
    development_documents: list[dict[str, Any]] = []
    intermediate_documents: list[dict[str, Any]] = []
    for index, event in enumerate(event_plan):
        current_payload = current_payloads[index]
        prior_payload = prior_payloads[index]
        current_hash = hashlib.sha256(current_payload).hexdigest()
        prior_hash = hashlib.sha256(prior_payload).hexdigest()
        prior_availability = f"2019-01-0{index + 2}"
        universe_records.extend(
            (
                {
                    "accession_number": prior_accessions[index],
                    "form": event["form"],
                    "availability_session": prior_availability,
                    "artifact_stage": "development",
                },
                {
                    "accession_number": event["accession_number"],
                    "form": event["form"],
                    "availability_session": event["availability_session"],
                    "artifact_stage": "intermediate",
                },
            )
        )
        development_documents.append(
            {
                "accession_number": prior_accessions[index],
                "normalized_text_sha256": prior_hash,
            }
        )
        intermediate_documents.append(
            {
                "accession_number": event["accession_number"],
                "normalized_text_sha256": current_hash,
            }
        )
        events.append(
            {
                "event": event,
                "current_normalized_text": current_payload,
                "current_normalized_source": {
                    "relative_path": (
                        f"document-{event['sec_document_ordinal']:04d}.normalized.txt"
                    ),
                    "byte_count": len(current_payload),
                    "sha256": current_hash,
                },
                "prior_same_form_normalized_text": prior_payload,
                "prior_same_form_normalized_source": {
                    "relative_path": f"carry-in-{index + 1:04d}.normalized.txt",
                    "byte_count": len(prior_payload),
                    "sha256": prior_hash,
                },
                "prior_accession_number": prior_accessions[index],
                "prior_availability_session": prior_availability,
                "prior_provenance_kind": "development_root_carry_in",
                "sec_reader_receipt_sha256": sec_receipt_hash,
                "carry_in_reader_receipt_sha256": carry_receipt_hash,
            }
        )
    loaded = {
        "claim": claim,
        "scope_kind": "stage_request",
        "scope_sha256": request_hash,
        "candidate_manifest": {"candidate_sha256": candidate_hash},
        "corpus_universe_manifest": {
            "universe_sha256": claim["corpus_universe_sha256"],
            "records": universe_records,
        },
        "content_manifests_by_stage": {
            "development": {
                "content_manifest_sha256": "4" * 64,
                "documents": development_documents,
            },
            "intermediate": {
                "content_manifest_sha256": "5" * 64,
                "documents": intermediate_documents,
            },
        },
        "session_dates": ["2019-01-02", "2019-01-03", "2020-01-02", "2020-01-03"],
        "sec_reader_receipt_sha256": sec_receipt_hash,
        "carry_in_reader_receipt_sha256": carry_receipt_hash,
        "events": events,
    }

    class ModelLock:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: Any) -> None:
            return None

    store._owned_model_execution_lock = lambda: ModelLock()
    store.claim_authorized_model_stage_execution = lambda **_kwargs: {
        "claim": claim,
        "created": True,
        "reader_receipt": None,
        "abort": None,
    }
    store._load_authorized_model_stage_event_inputs = lambda **_kwargs: loaded
    source_revalidations: list[str] = []
    store._revalidate_authorized_model_execution_sources = (
        lambda observed: source_revalidations.append(observed["claim_sha256"])
    )
    aborts: list[str] = []
    store.abort_authorized_model_stage_execution = (
        lambda *, request_sha256, reason: aborts.append(reason)
    )

    probe_counter = 0

    class FakeIdentity:
        evidence_sha256 = "6" * 64

    class FakeProbe:
        def __init__(self, ordinal: int) -> None:
            self.ordinal = ordinal

        def to_manifest(self) -> dict[str, Any]:
            body = {"probe_ordinal": self.ordinal}
            return {**body, "receipt_sha256": canonical_sha256(body)}

        def pinned_runtime_identity(self) -> FakeIdentity:
            return FakeIdentity()

        def runtime_evidence(self) -> dict[str, Any]:
            return {"runtime": "same"}

    def fake_probe(**_kwargs: Any) -> FakeProbe:
        nonlocal probe_counter
        probe_counter += 1
        return FakeProbe(probe_counter)

    monkeypatch.setattr(runner_module, "probe_owned_ollama_runtime", fake_probe)
    monkeypatch.setattr(
        runner_module,
        "validate_ollama_runtime_probe_receipt",
        lambda manifest, **_kwargs: FakeProbe(manifest["probe_ordinal"]),
    )

    preprocessing_counter = 0

    def fake_preprocess(**_kwargs: Any) -> dict[str, Any]:
        nonlocal preprocessing_counter
        preprocessing_counter += 1
        sentences = [{"id": "C0001", "text": f"Safe filing {preprocessing_counter}"}]
        payload_hash = hashlib.sha256(
            f"payload-{preprocessing_counter}".encode()
        ).hexdigest()
        return {
            "sentences": sentences,
            "model_payload": {"safe": preprocessing_counter},
            "model_payload_sha256": payload_hash,
            "preprocessed_event_sha256": hashlib.sha256(
                f"event-{preprocessing_counter}".encode()
            ).hexdigest(),
            "redaction_report": {"safe": True},
        }

    monkeypatch.setattr(runner_module, "preprocess_filing_event", fake_preprocess)
    monkeypatch.setattr(
        runner_module,
        "build_owned_preprocessing_receipt",
        lambda **kwargs: {
            "receipt_sha256": hashlib.sha256(
                f"pre-{kwargs['event_ordinal']}".encode()
            ).hexdigest()
        },
    )
    monkeypatch.setattr(
        runner_module,
        "build_redacted_input_manifest",
        lambda **kwargs: {
            "redacted_input_manifest_sha256": hashlib.sha256(
                f"redacted-{kwargs['accession_number']}".encode()
            ).hexdigest()
        },
    )
    validated_accessions: list[str] = []

    def fake_validate(request: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        validated_accessions.append(request["current_accession_number"])
        assert set(kwargs["content_manifests_by_stage"]) == {
            "development",
            "intermediate",
        }
        assert kwargs["expected_preprocessed_event_sha256"]
        assert kwargs["expected_owned_preprocessing_receipt_sha256"]
        assert kwargs["expected_sec_reader_receipt_sha256"] == sec_receipt_hash
        assert kwargs["expected_carry_in_reader_receipt_sha256"] == carry_receipt_hash
        return {
            "candidate_sha256": candidate_hash,
            "sentence_ids": ("C0001",),
            "model_payload": request["model_payload"],
            "model_payload_sha256": request["model_payload_sha256"],
        }

    monkeypatch.setattr(runner_module, "validate_extractor_request", fake_validate)

    call_counter = 0

    class FakeAttempt:
        def __init__(self, ordinal: int) -> None:
            self.ordinal = ordinal
            self.elapsed_nanoseconds = 1

        def to_manifest(self) -> dict[str, Any]:
            body = {"attempt_ordinal": self.ordinal, "elapsed_nanoseconds": 1}
            return {**body, "receipt_sha256": canonical_sha256(body)}

    def fake_call(_validated: dict[str, Any], **kwargs: Any) -> FakeAttempt:
        nonlocal call_counter
        call_counter += 1
        intent = (
            store.store_directory
            / "stage_outputs"
            / claim["claim_sha256"]
            / "model_extraction"
            / "events"
            / f"{call_counter:06d}"
            / "call_intent.json"
        )
        assert intent.is_file()
        assert json.loads(intent.read_text(encoding="utf-8"))[
            "external_effect_started"
        ] is False
        assert "transport" not in kwargs
        return FakeAttempt(call_counter)

    monkeypatch.setattr(runner_module, "call_ollama_extractor_attempt", fake_call)
    monkeypatch.setattr(
        runner_module,
        "validate_ollama_model_attempt_receipt",
        lambda manifest, **_kwargs: FakeAttempt(manifest["attempt_ordinal"]),
    )
    monkeypatch.setattr(
        runner_module,
        "build_runtime_identity_guard",
        lambda **_kwargs: {"runtime_guard_sha256": "7" * 64},
    )

    receipt = {"claim_sha256": claim["claim_sha256"], "receipt_sha256": "8" * 64}

    def record(*, request_sha256: str) -> dict[str, Any]:
        assert request_sha256 == request_hash
        marker = (
            store.store_directory
            / "stage_outputs"
            / claim["claim_sha256"]
            / "model_extraction"
            / "complete.json"
        )
        assert marker.is_file()
        return receipt

    store._record_authorized_model_stage_reader_output = record
    result = run_owned_stage_model_batch(
        reveal_store=store,
        request_sha256=request_hash,
    )

    assert result == {"claim": claim, "reader_receipt": receipt}
    assert validated_accessions == list(current_accessions)
    assert call_counter == 2
    assert probe_counter == 2
    assert len(source_revalidations) >= 6
    assert aborts == []


def test_completed_model_retry_replays_with_zero_probe_or_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _stage_model_claim()
    receipt = {"claim_sha256": claim["claim_sha256"], "receipt_sha256": "8" * 64}

    @contextmanager
    def model_lock() -> Iterator[None]:
        yield

    store._owned_model_execution_lock = model_lock
    store.claim_authorized_model_stage_execution = lambda **_kwargs: {
        "claim": claim,
        "created": False,
        "reader_receipt": receipt,
        "abort": None,
    }
    replayed: list[str] = []
    store._record_authorized_model_stage_reader_output = (
        lambda *, request_sha256: replayed.append(request_sha256) or receipt
    )
    monkeypatch.setattr(
        runner_module,
        "probe_owned_ollama_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed retry must not probe")
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "call_ollama_extractor_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed retry must not call the model")
        ),
    )

    assert run_owned_stage_model_batch(
        reveal_store=store,
        request_sha256=claim["request_sha256"],
    ) == {"claim": claim, "reader_receipt": receipt}
    assert replayed == [claim["request_sha256"]]


def test_completed_development_model_retry_uses_root_lifecycle_with_zero_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _development_model_claim()
    receipt = {"claim_sha256": claim["claim_sha256"], "receipt_sha256": "8" * 64}

    @contextmanager
    def model_lock() -> Iterator[None]:
        yield

    store._owned_model_execution_lock = model_lock
    store.claim_owned_development_model_execution = lambda **_kwargs: {
        "claim": claim,
        "created": False,
        "reader_receipt": receipt,
        "abort": None,
    }
    replayed: list[str] = []
    store._record_owned_development_model_reader_output = (
        lambda *, development_root_scope_sha256: replayed.append(
            development_root_scope_sha256
        )
        or receipt
    )
    monkeypatch.setattr(
        runner_module,
        "probe_owned_ollama_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed development retry must not probe")
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "call_ollama_extractor_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed development retry must not call the model")
        ),
    )

    assert run_owned_development_model_batch(
        reveal_store=store,
        development_root_scope_sha256=claim[
            "development_root_scope_sha256"
        ],
    ) == {"claim": claim, "reader_receipt": receipt}
    assert replayed == [claim["development_root_scope_sha256"]]


def test_recovered_incomplete_model_claim_aborts_with_zero_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _stage_model_claim()

    @contextmanager
    def model_lock() -> Iterator[None]:
        yield

    store._owned_model_execution_lock = model_lock
    store.claim_authorized_model_stage_execution = lambda **_kwargs: {
        "claim": claim,
        "created": False,
        "reader_receipt": None,
        "abort": None,
    }
    store._record_authorized_model_stage_reader_output = (
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("no complete marker"))
    )
    aborts: list[str] = []
    store.abort_authorized_model_stage_execution = (
        lambda *, request_sha256, reason: aborts.append(reason)
    )
    monkeypatch.setattr(
        runner_module,
        "probe_owned_ollama_runtime",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovered incomplete claim must not probe")
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "call_ollama_extractor_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recovered incomplete claim must not call the model")
        ),
    )

    with pytest.raises(SecFilingGemmaStageRunnerError, match="cannot be retried"):
        run_owned_stage_model_batch(
            reveal_store=store,
            request_sha256=claim["request_sha256"],
        )
    assert aborts == ["claim_recovered_without_terminal_receipt"]


def test_global_model_lock_contention_precedes_claim(
    tmp_path: Path,
) -> None:
    store = _new_store(tmp_path)
    claim_calls: list[str] = []

    class UnavailableModelLock:
        def __enter__(self) -> None:
            raise TimeoutError("simulated owner")

        def __exit__(self, *_args: Any) -> None:
            return None

    store._owned_model_execution_lock = lambda: UnavailableModelLock()
    store.claim_authorized_model_stage_execution = (
        lambda **_kwargs: claim_calls.append("claim")
    )
    with pytest.raises(SecFilingGemmaStageRunnerError, match="globally owned"):
        run_owned_stage_model_batch(
            reveal_store=store,
            request_sha256="a" * 64,
        )
    assert claim_calls == []
