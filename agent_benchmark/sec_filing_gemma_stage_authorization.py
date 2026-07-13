"""Pure consumed-stage authorization grants for downstream SEC/Gemma APIs.

The effectful reveal store remains the only component allowed to consume a
request.  This module can derive a detached pin from an already authenticated
post-consumption store snapshot, mint a grant for that snapshot's exact ledger
tip, and validate the grant later against a separately supplied snapshot/pin
plus the independently loaded monotonic current-tip anchor.

Pin derivation is not external authentication.  A downstream caller must load
the current-tip anchor independently of the bundle; an old snapshot and its old
bundled pin are deliberately rejected after any newer store transition.
No function here performs filesystem, network, model, market, SEC, or outcome
I/O, and the compact grant contains no market values, labels, scores, or returns.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_contract import (
    CONTRACT_VERSION,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    canonical_sha256,
    validate_candidate_manifest,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    REVEAL_REQUEST_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    STAGE_ACCESS_MANIFEST_SCHEMA_VERSION,
)
from agent_benchmark.sec_filing_gemma_stage_verifier import (
    STAGE_EVIDENCE_SCHEMA_VERSION,
    detach_untrusted_stage_json,
)
from agent_benchmark.sec_filing_gemma_source_identity import (
    CANONICAL_SOURCE_ROLE_PATHS,
)


STORE_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-reveal-store-v1"
CONSUMPTION_LEDGER_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-request-ledger-v1"
)
CONSUMPTION_ENTRY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-request-entry-v1"
)
SEMANTIC_PREREQUISITE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-semantic-prerequisite-validation-v1"
)
CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-store-pin-v1"
)
CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-authorization-grant-v1"
)
CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-authorization-bundle-v1"
)
CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-output-receipt-v2"
)
STAGE_SEC_EXECUTION_CLAIM_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-sec-execution-claim-v2"
)
STAGE_SEC_READER_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-sec-reader-receipt-v2"
)
STAGE_SEC_EXECUTION_ABORT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-sec-execution-abort-v1"
)
REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-reveal-store-current-tip-anchor-v4"
)
TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-trusted-stage-content-pin-v2"
)
TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-trusted-stage-content-authentication-v1"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_OUTPUT_NAMESPACE_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_AAPL_ACCESSION_RE = re.compile(r"0000320193-[0-9]{2}-[0-9]{6}\Z")
STAGE_RUNNER_REPOSITORY_PATH: Final[str] = (
    "agent_benchmark/sec_filing_gemma_stage_runner.py"
)
SEC_CORPUS_REPOSITORY_PATH: Final[str] = (
    "agent_benchmark/sec_filing_gemma_corpus.py"
)
SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID: Final[str] = "sec_stage_document_batch"
STAGE_EVIDENCE_OUTPUT_COMPONENT_ID: Final[str] = "owned_stage_evidence_document"
STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH: Final[str] = "stage_evidence.json"
# A Latin-1 source byte can expand to at most two UTF-8 bytes.  Keeping the
# complete raw batch at 64 MiB therefore guarantees that each normalized file
# remains inside the reveal store's fixed 128 MiB per-file ceiling, while the
# raw and normalized batch plus canonical JSON stays far below its total cap.
OWNED_SEC_RAW_BATCH_MAX_BYTES: Final[int] = 64 * 1024 * 1024
SEC_EXECUTION_RESOLVED_SOURCE_PATHS: Final[tuple[tuple[str, str], ...]] = tuple(
    (role, path)
    for role, path in CANONICAL_SOURCE_ROLE_PATHS.items()
    if path is not None
)
_STAGE_PREREQUISITES: Final[dict[str, str]] = {
    "intermediate": "development",
    "final": "intermediate",
}
_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "anchor",
        "latest_registry",
        "latest_registry_pin",
        "consumption_ledger",
        "state_sha256",
    }
)
_LEDGER_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "entries", "chain", "ledger_sha256"}
)
_LEDGER_CHAIN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "genesis_tip_sha256",
        "tip_sha256",
        "consumed_request_count",
        "actual_final_touch_count",
        "historical_final_reveal_count_lower_bound",
        "repository_final_touch_count_lower_bound",
    }
)
_ENTRY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "sequence",
        "request_sha256",
        "request",
        "stage_access_manifest",
        "stage",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "prerequisite_validation",
        "prior_tip_sha256",
        "final_touch_delta",
        "cumulative_actual_final_touch_count",
        "entry_sha256",
    }
)
_REQUEST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "registry_sha256",
        "registry_tip_sha256",
        "registered_entry_count",
        "historical_final_reveal_count_lower_bound",
        "stage",
        "stage_access_manifest_sha256",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_scope",
        "authorizes_outcome_access",
        "effectful_atomic_single_use_consumption_required",
        "cross_attempt_comparison_permitted",
        "cross_attempt_winner_selection_permitted",
        "globally_pristine_claim",
        "request_sha256",
    }
)
_VALIDATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "validation_kind",
        "validator_id",
        "validator_source_sha256",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_sha256",
        "requested_stage",
        "stage_access_manifest_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "semantic_checks",
        "semantic_receipt",
        "semantic_receipt_sha256",
        "semantic_validation_completed",
        "authorizes_outcome_access",
        "result_sha256",
    }
)
_STORE_PIN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "store_state_sha256",
        "store_snapshot_bytes_sha256",
        "store_snapshot_byte_count",
        "store_anchor_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
        "consumed_request_count",
        "store_pin_sha256",
    }
)
_GRANT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "authorization_kind",
        "consumption_entry_sha256",
        "consumption_entry_sequence",
        "request_sha256",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "prerequisite_stage",
        "stage",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "output_namespace",
        "store_state_sha256",
        "store_snapshot_bytes_sha256",
        "store_snapshot_byte_count",
        "store_pin_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
        "consumed_stage_access_authorized",
        "authorization_scope",
        "outcomes_included",
        "market_values_included",
        "cross_stage_access_permitted",
        "grant_reuse_across_store_state_tips_permitted",
        "authorization_grant_sha256",
    }
)
_BUNDLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "authenticated_store_snapshot",
        "store_state_pin",
        "authorization_grant",
        "bundle_sha256",
    }
)
_STAGE_OUTPUT_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "consumption_entry_sha256",
        "consumption_entry_sequence",
        "request_sha256",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "input_prerequisite_stage",
        "output_stage",
        "input_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "output_namespace",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "sec_execution_claim_sha256",
        "sec_reader_receipt_sha256",
        "grant_store_state_sha256",
        "grant_consumption_ledger_sha256",
        "grant_consumption_ledger_tip_sha256",
        "output_kind",
        "output_stage_evidence_schema_version",
        "output_stage_evidence_component_id",
        "output_stage_evidence_relative_path",
        "output_stage_evidence_sha256",
        "output_stage_evidence_document_sha256",
        "output_stage_evidence_complete_marker_sha256",
        "output_stage_evidence_canonical_byte_count",
        "output_stage_evidence_prerequisite_stage",
        "output_parent_stage_evidence_sha256",
        "output_candidate_sha256",
        "output_stage_evidence_recomputed_by_store",
        "fresh_stage_evidence_provenance_claimed",
        "cross_stage_output_permitted",
        "grant_reuse_for_different_output_permitted",
        "output_receipt_sha256",
    }
)
_STAGE_SEC_EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "claim_kind",
        "request_sha256",
        "consumption_entry_sha256",
        "consumption_entry_sequence",
        "attempt_id",
        "candidate_sha256",
        "registry_entry_sha256",
        "input_prerequisite_stage",
        "authorized_stage",
        "input_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "output_namespace",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "grant_store_state_sha256",
        "grant_consumption_ledger_sha256",
        "grant_consumption_ledger_tip_sha256",
        "start_current_tip_anchor_sha256",
        "runner_repository_path",
        "runner_source_sha256",
        "sec_corpus_repository_path",
        "sec_corpus_source_sha256",
        "execution_source_hashes",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "sec_user_agent_sha256",
        "sec_component_id",
        "sec_component_plan_sha256",
        "effect_may_be_repeated_after_indeterminate_crash",
        "claim_sha256",
    }
)
_STAGE_SEC_READER_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "receipt_kind",
        "request_sha256",
        "claim_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "sec_component_id",
        "sec_component_plan_sha256",
        "runner_source_sha256",
        "sec_corpus_source_sha256",
        "execution_source_hashes_sha256",
        "execution_source_role_count",
        "sec_user_agent_sha256",
        "byte_index",
        "byte_index_sha256",
        "byte_count_total",
        "complete_marker_sha256",
        "fresh_network_provenance_claimed",
        "reader_output_recomputed_by_store",
        "receipt_sha256",
    }
)
_STAGE_SEC_EXECUTION_ABORT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "abort_kind",
        "request_sha256",
        "claim_sha256",
        "authorized_stage",
        "candidate_sha256",
        "output_namespace",
        "reason",
        "external_effect_retry_permitted",
        "abort_sha256",
    }
)
_CURRENT_TIP_ANCHOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "revision",
        "previous_tip_anchor_sha256",
        "state_sha256",
        "state_snapshot_bytes_sha256",
        "state_snapshot_byte_count",
        "registry_sha256",
        "registry_tip_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
        "consumed_request_count",
        "trusted_stage_content_pins",
        "authorization_bundles",
        "consumed_stage_output_receipts",
        "stage_sec_execution_claims",
        "stage_sec_reader_receipts",
        "stage_sec_execution_aborts",
        "tip_anchor_sha256",
    }
)
_TRUSTED_STAGE_CONTENT_PIN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "prerequisite_stage",
        "requested_stage",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "content_manifest_sha256",
        "stage_artifact_sha256",
        "external_seal_receipt_sha256",
        "trusted_store_state_sha256",
        "pin_sha256",
    }
)
_TRUSTED_STAGE_CONTENT_AUTHENTICATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "authentication_kind",
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "prerequisite_stage",
        "requested_stage",
        "trusted_stage_content_pin_sha256",
        "trusted_store_state_sha256",
        "trusted_current_tip_anchor_sha256",
        "trusted_current_tip_revision",
        "trusted_stage_content_pins_sha256",
        "authentication_receipt_sha256",
    }
)


class SecFilingGemmaStageAuthorizationError(ValueError):
    """A consumed-stage grant or its externally supplied state proof is invalid."""


def _plain(value: Any, location: str) -> Any:
    """Return one detached exact-JSON value under fixed allocation bounds."""

    try:
        return detach_untrusted_stage_json(value, location)
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must contain exact built-in values within the "
            "bounded exact-JSON authorization limits"
        ) from exc


def _mapping(value: Any, location: str) -> dict[str, Any]:
    # Authorization-critical inputs must never execute caller-controlled
    # ``Mapping`` methods while they are being detached.  Exact built-in
    # containers have no overridable iteration/item hooks.
    if type(value) is not dict:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be an exact built-in dict"
        )
    detached = _plain(value, location)
    if type(detached) is not dict:
        raise SecFilingGemmaStageAuthorizationError(f"{location} must be an object")
    return detached


def _expect_keys(value: Mapping[str, Any], expected: frozenset[str], location: str) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} keys changed; missing={missing}, extra={extra}"
        )


def _sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _tagged_sha256(value: Any, location: str) -> str:
    if type(value) is not str or _TAGGED_SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be a tagged lowercase SHA-256 digest"
        )
    return value


def _safe_id(value: Any, location: str) -> str:
    if type(value) is not str or _SAFE_ID_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageAuthorizationError(f"{location} is invalid")
    return value


def _strict_int(value: Any, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} must be an integer at least {minimum}"
        )
    return value


def _self_hash(value: Mapping[str, Any], field: str, location: str) -> str:
    observed = _sha256(value.get(field), f"{location} {field}")
    body = {key: value[key] for key in value if key != field}
    expected = canonical_sha256(body)
    if not hmac.compare_digest(observed, expected):
        raise SecFilingGemmaStageAuthorizationError(
            f"{location} self-hash is inconsistent"
        )
    return observed


def _encoded_store_snapshot(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "Authenticated store snapshot cannot be encoded exactly"
        ) from exc


def _consumption_genesis(anchor: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema_version": "aapl-sec-gemma-consumed-request-genesis-v1",
            "contract_version": CONTRACT_VERSION,
            "store_anchor_sha256": canonical_sha256(anchor),
        }
    )


def _validated_entry(
    raw: Any, *, expected_sequence: int, expected_prior_tip_sha256: str
) -> dict[str, Any]:
    entry = _mapping(raw, f"consumption entry {expected_sequence}")
    _expect_keys(entry, _ENTRY_KEYS, f"consumption entry {expected_sequence}")
    if (
        entry["schema_version"] != CONSUMPTION_ENTRY_SCHEMA_VERSION
        or _strict_int(entry["sequence"], "consumption entry sequence", minimum=1)
        != expected_sequence
        or entry["prior_tip_sha256"] != expected_prior_tip_sha256
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumption entry sequence, schema, or prior tip changed"
        )
    stage = entry["stage"]
    if stage not in _STAGE_PREREQUISITES:
        raise SecFilingGemmaStageAuthorizationError("Consumption entry stage is invalid")
    _safe_id(entry["attempt_id"], "consumption attempt id")
    _sha256(entry["request_sha256"], "consumption request hash")
    _sha256(entry["candidate_sha256"], "consumption candidate hash")
    _sha256(entry["registry_entry_sha256"], "consumption registry entry hash")
    _self_hash(entry, "entry_sha256", f"consumption entry {expected_sequence}")
    return entry


def _validated_ledger(
    raw: Any, *, anchor: Mapping[str, Any]
) -> dict[str, Any]:
    ledger = _mapping(raw, "consumption ledger")
    _expect_keys(ledger, _LEDGER_KEYS, "consumption ledger")
    if ledger["schema_version"] != CONSUMPTION_LEDGER_SCHEMA_VERSION:
        raise SecFilingGemmaStageAuthorizationError("Consumption ledger schema changed")
    entries_raw = ledger["entries"]
    if type(entries_raw) is not list:
        raise SecFilingGemmaStageAuthorizationError("Consumption entries must be a list")
    chain = _mapping(ledger["chain"], "consumption ledger chain")
    _expect_keys(chain, _LEDGER_CHAIN_KEYS, "consumption ledger chain")
    genesis = _consumption_genesis(anchor)
    if chain["genesis_tip_sha256"] != genesis:
        raise SecFilingGemmaStageAuthorizationError("Consumption genesis changed")
    current_tip = genesis
    entries: list[dict[str, Any]] = []
    seen_requests: set[str] = set()
    seen_attempt_stages: set[tuple[str, str]] = set()
    intermediate_candidates: set[tuple[str, str, str]] = set()
    final_count = 0
    for sequence, raw_entry in enumerate(entries_raw, start=1):
        entry = _validated_entry(
            raw_entry,
            expected_sequence=sequence,
            expected_prior_tip_sha256=current_tip,
        )
        request_hash = entry["request_sha256"]
        attempt_stage = (entry["attempt_id"], entry["stage"])
        if request_hash in seen_requests or attempt_stage in seen_attempt_stages:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumption request or candidate stage was replayed"
            )
        candidate_key = (
            entry["attempt_id"],
            entry["candidate_sha256"],
            entry["registry_entry_sha256"],
        )
        if entry["stage"] == "final" and candidate_key not in intermediate_candidates:
            raise SecFilingGemmaStageAuthorizationError(
                "Final consumption lacks its exact intermediate predecessor"
            )
        expected_delta = 1 if entry["stage"] == "final" else 0
        if entry["final_touch_delta"] != expected_delta:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumption final-touch delta changed"
            )
        final_count += expected_delta
        if entry["cumulative_actual_final_touch_count"] != final_count:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumption cumulative final-touch count changed"
            )
        entries.append(entry)
        seen_requests.add(request_hash)
        seen_attempt_stages.add(attempt_stage)
        if entry["stage"] == "intermediate":
            intermediate_candidates.add(candidate_key)
        current_tip = entry["entry_sha256"]
    count = _strict_int(chain["consumed_request_count"], "consumed request count")
    actual_final = _strict_int(chain["actual_final_touch_count"], "actual final count")
    historical_final = _strict_int(
        chain["historical_final_reveal_count_lower_bound"],
        "historical final lower bound",
    )
    repository_final = _strict_int(
        chain["repository_final_touch_count_lower_bound"],
        "repository final lower bound",
    )
    if (
        count != len(entries)
        or actual_final != final_count
        or repository_final != historical_final + final_count
        or chain["tip_sha256"] != current_tip
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumption ledger count, tip, or final-touch totals changed"
        )
    _self_hash(ledger, "ledger_sha256", "consumption ledger")
    return ledger


def _validated_store_snapshot(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _mapping(raw, "authenticated reveal-store snapshot")
    _expect_keys(state, _STATE_KEYS, "authenticated reveal-store snapshot")
    if (
        state["schema_version"] != STORE_SCHEMA_VERSION
        or state["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Reveal-store snapshot schema or contract changed"
        )
    anchor = _mapping(state["anchor"], "store anchor")
    registry = _mapping(state["latest_registry"], "latest registry")
    registry_pin = _mapping(state["latest_registry_pin"], "latest registry pin")
    registry_hash = _sha256(registry.get("registry_sha256"), "latest registry hash")
    if registry_pin.get("registry_sha256") != registry_hash:
        raise SecFilingGemmaStageAuthorizationError(
            "Latest registry pin belongs to another registry"
        )
    registry_tip = _sha256(registry_pin.get("tip_sha256"), "latest registry tip")
    registry_chain = _mapping(registry.get("chain"), "latest registry chain")
    if (
        registry_chain.get("tip_sha256") != registry_tip
        or registry_chain.get("registered_entry_count")
        != registry_pin.get("registered_entry_count")
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Latest registry chain and pin disagree"
        )
    ledger = _validated_ledger(state["consumption_ledger"], anchor=anchor)
    _self_hash(state, "state_sha256", "authenticated reveal-store snapshot")
    return state, ledger


def derive_consumed_stage_store_state_pin(
    authenticated_store_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a pin that must be persisted through a separate trust path."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    registry = state["latest_registry"]
    registry_pin = state["latest_registry_pin"]
    chain = ledger["chain"]
    snapshot_bytes = _encoded_store_snapshot(state)
    body = {
        "schema_version": CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "store_state_sha256": state["state_sha256"],
        "store_snapshot_bytes_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "store_snapshot_byte_count": len(snapshot_bytes),
        "store_anchor_sha256": canonical_sha256(state["anchor"]),
        "registry_sha256": registry["registry_sha256"],
        "registry_tip_sha256": registry_pin["tip_sha256"],
        "consumption_ledger_sha256": ledger["ledger_sha256"],
        "consumption_ledger_tip_sha256": chain["tip_sha256"],
        "consumed_request_count": chain["consumed_request_count"],
    }
    return {**body, "store_pin_sha256": canonical_sha256(body)}


def validate_consumed_stage_store_state_pin(
    authenticated_store_snapshot: Mapping[str, Any],
    external_store_state_pin: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an authenticated snapshot against a separately supplied pin."""

    observed = _mapping(external_store_state_pin, "external store-state pin")
    _expect_keys(observed, _STORE_PIN_KEYS, "external store-state pin")
    _self_hash(observed, "store_pin_sha256", "external store-state pin")
    expected = derive_consumed_stage_store_state_pin(authenticated_store_snapshot)
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "External store-state pin does not authenticate this exact snapshot"
        )
    return observed


def _validated_trusted_stage_content_pin(
    raw: Any,
    *,
    expected_request_sha256: str | None = None,
) -> dict[str, Any]:
    pin = _mapping(raw, "trusted stage-content pin")
    _expect_keys(
        pin,
        _TRUSTED_STAGE_CONTENT_PIN_KEYS,
        "trusted stage-content pin",
    )
    if (
        pin["schema_version"] != TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION
        or pin["contract_version"] != CONTRACT_VERSION
        or _STAGE_PREREQUISITES.get(pin["requested_stage"])
        != pin["prerequisite_stage"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted stage-content pin schema, contract, or stage transition changed"
        )
    _safe_id(pin["attempt_id"], "trusted content attempt id")
    for field in (
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "content_manifest_sha256",
        "stage_artifact_sha256",
        "external_seal_receipt_sha256",
        "trusted_store_state_sha256",
    ):
        _sha256(pin[field], f"trusted content {field}")
    pin_hash = _self_hash(pin, "pin_sha256", "trusted stage-content pin")
    if (
        expected_request_sha256 is not None
        and pin["request_sha256"]
        != _sha256(expected_request_sha256, "expected trusted-content request hash")
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted stage-content pin is stored under another request"
        )
    pin["pin_sha256"] = pin_hash
    return pin


def _validated_trusted_stage_content_pins(
    raw: Any,
) -> dict[str, dict[str, Any]]:
    pins = _mapping(raw, "current-tip trusted stage-content pins")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_pin in pins.items():
        request_hash = _sha256(
            request_sha256,
            "current-tip trusted stage-content pin key",
        )
        validated[request_hash] = _validated_trusted_stage_content_pin(
            raw_pin,
            expected_request_sha256=request_hash,
        )
    return validated


def _validated_trusted_content_request_and_access(
    reveal_request: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request = _mapping(reveal_request, "trusted-content reveal request")
    _expect_keys(request, _REQUEST_KEYS, "trusted-content reveal request")
    request_hash = _self_hash(
        request,
        "request_sha256",
        "trusted-content reveal request",
    )
    requested_stage = request["stage"]
    prerequisite_stage = request["prerequisite_stage"]
    if (
        request["schema_version"] != REVEAL_REQUEST_SCHEMA_VERSION
        or request["contract_version"] != CONTRACT_VERSION
        or _STAGE_PREREQUISITES.get(requested_stage) != prerequisite_stage
        or request["authorizes_outcome_access"] is not False
        or request["effectful_atomic_single_use_consumption_required"] is not True
        or request["cross_attempt_comparison_permitted"] is not False
        or request["cross_attempt_winner_selection_permitted"] is not False
        or request["globally_pristine_claim"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted-content request changed its frozen non-authorizing semantics"
        )
    _safe_id(request["attempt_id"], "trusted-content request attempt id")
    for field in (
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
    ):
        _sha256(request[field], f"trusted-content request {field}")

    access = _mapping(
        stage_access_manifest,
        "trusted-content stage-access manifest",
    )
    access_hash = _self_hash(
        access,
        "stage_access_manifest_sha256",
        "trusted-content stage-access manifest",
    )
    transition = _mapping(
        access.get("transition"),
        "trusted-content stage-access transition",
    )
    candidate = _mapping(
        access.get("candidate"),
        "trusted-content stage-access candidate",
    )
    evidence_pin = _mapping(
        access.get("prerequisite_evidence_pin"),
        "trusted-content prerequisite evidence pin",
    )
    _expect_keys(
        evidence_pin,
        frozenset(
            {
                "stage",
                "content_manifest_sha256",
                "stage_artifact_sha256",
                "external_seal_receipt_sha256",
            }
        ),
        "trusted-content prerequisite evidence pin",
    )
    if (
        access.get("schema_version") != STAGE_ACCESS_MANIFEST_SCHEMA_VERSION
        or access.get("contract_version") != CONTRACT_VERSION
        or access_hash != request["stage_access_manifest_sha256"]
        or transition.get("prerequisite_stage") != prerequisite_stage
        or transition.get("requested_stage") != requested_stage
        or transition.get("single_use_consumption_required") is not True
        or transition.get("stage_reuse_permitted") is not False
        or candidate.get("attempt_id") != request["attempt_id"]
        or candidate.get("candidate_sha256") != request["candidate_sha256"]
        or candidate.get("candidate_design_sha256")
        != request["candidate_design_sha256"]
        or evidence_pin.get("stage") != prerequisite_stage
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted-content stage access crossed its request or stage boundary"
        )
    for field in (
        "content_manifest_sha256",
        "stage_artifact_sha256",
        "external_seal_receipt_sha256",
    ):
        _sha256(evidence_pin[field], f"trusted-content prerequisite {field}")
    request["request_sha256"] = request_hash
    access["stage_access_manifest_sha256"] = access_hash
    return request, access, evidence_pin


def derive_reveal_store_trusted_stage_content_pin(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    reveal_request: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact pre-consumption content pin for one validated request."""

    state, _ledger = _validated_store_snapshot(authenticated_store_snapshot)
    request, access, evidence_pin = _validated_trusted_content_request_and_access(
        reveal_request,
        stage_access_manifest,
    )
    body = {
        "schema_version": TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "request_sha256": request["request_sha256"],
        "prerequisite_stage_evidence_sha256": request[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": access[
            "stage_access_manifest_sha256"
        ],
        "prerequisite_stage": request["prerequisite_stage"],
        "requested_stage": request["stage"],
        "attempt_id": request["attempt_id"],
        "candidate_sha256": request["candidate_sha256"],
        "candidate_design_sha256": request["candidate_design_sha256"],
        "registry_entry_sha256": request["registry_entry_sha256"],
        "content_manifest_sha256": evidence_pin["content_manifest_sha256"],
        "stage_artifact_sha256": evidence_pin["stage_artifact_sha256"],
        "external_seal_receipt_sha256": evidence_pin[
            "external_seal_receipt_sha256"
        ],
        "trusted_store_state_sha256": state["state_sha256"],
    }
    return {**body, "pin_sha256": canonical_sha256(body)}


def authenticate_reveal_store_trusted_stage_content_pin(
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    *,
    reveal_request: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate exact pin membership through the independently loaded tip."""

    state, _ledger = _validated_store_snapshot(authenticated_store_snapshot)
    current_tip = validate_reveal_store_current_tip_anchor(
        state,
        independent_current_tip_anchor,
    )
    expected_pin = derive_reveal_store_trusted_stage_content_pin(
        state,
        reveal_request=reveal_request,
        stage_access_manifest=stage_access_manifest,
    )
    request_hash = expected_pin["request_sha256"]
    persisted = current_tip["trusted_stage_content_pins"].get(request_hash)
    if persisted != expected_pin:
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted stage-content pin is not exactly persisted at the current tip"
        )
    body = {
        "schema_version": TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "authentication_kind": (
            "reveal_store_current_tip_persisted_trusted_stage_content_pin"
        ),
        "request_sha256": request_hash,
        "prerequisite_stage_evidence_sha256": expected_pin[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": expected_pin[
            "stage_access_manifest_sha256"
        ],
        "prerequisite_stage": expected_pin["prerequisite_stage"],
        "requested_stage": expected_pin["requested_stage"],
        "trusted_stage_content_pin_sha256": expected_pin["pin_sha256"],
        "trusted_store_state_sha256": state["state_sha256"],
        "trusted_current_tip_anchor_sha256": current_tip[
            "tip_anchor_sha256"
        ],
        "trusted_current_tip_revision": current_tip["revision"],
        "trusted_stage_content_pins_sha256": canonical_sha256(
            current_tip["trusted_stage_content_pins"]
        ),
    }
    return {
        **body,
        "authentication_receipt_sha256": canonical_sha256(body),
    }


def validate_trusted_stage_content_authentication_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(receipt, "trusted stage-content authentication receipt")
    _expect_keys(
        value,
        _TRUSTED_STAGE_CONTENT_AUTHENTICATION_KEYS,
        "trusted stage-content authentication receipt",
    )
    if (
        value["schema_version"]
        != TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["authentication_kind"]
        != "reveal_store_current_tip_persisted_trusted_stage_content_pin"
        or _STAGE_PREREQUISITES.get(value["requested_stage"])
        != value["prerequisite_stage"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Trusted stage-content authentication semantics changed"
        )
    for field in (
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "trusted_stage_content_pin_sha256",
        "trusted_store_state_sha256",
        "trusted_current_tip_anchor_sha256",
        "trusted_stage_content_pins_sha256",
    ):
        _sha256(value[field], f"trusted content authentication {field}")
    _strict_int(
        value["trusted_current_tip_revision"],
        "trusted content current-tip revision",
    )
    _self_hash(
        value,
        "authentication_receipt_sha256",
        "trusted stage-content authentication receipt",
    )
    return value


def _validated_authorization_bundles(raw: Any) -> dict[str, dict[str, Any]]:
    bundles = _mapping(raw, "current-tip authorization bundles")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_bundle in bundles.items():
        request_hash = _sha256(
            request_sha256, "current-tip authorization-bundle key"
        )
        bundle = _mapping(
            raw_bundle,
            f"current-tip authorization bundle {request_hash}",
        )
        _expect_keys(
            bundle,
            _BUNDLE_KEYS,
            f"current-tip authorization bundle {request_hash}",
        )
        if (
            bundle["schema_version"]
            != CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Current-tip authorization bundle schema changed"
            )
        _self_hash(
            bundle,
            "bundle_sha256",
            f"current-tip authorization bundle {request_hash}",
        )
        snapshot = _mapping(
            bundle["authenticated_store_snapshot"],
            "current-tip bundled store snapshot",
        )
        pin = _mapping(
            bundle["store_state_pin"],
            "current-tip bundled store-state pin",
        )
        grant = _mapping(
            bundle["authorization_grant"],
            "current-tip bundled authorization grant",
        )
        if grant.get("request_sha256") != request_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Current-tip authorization bundle is stored under another request"
            )
        # Validate all detached component self-hashes, then reconstruct the
        # exact grant from the bundled snapshot and pin.
        _validated_store_snapshot(snapshot)
        validate_consumed_stage_store_state_pin(snapshot, pin)
        _expect_keys(grant, _GRANT_KEYS, "current-tip bundled authorization grant")
        _self_hash(
            grant,
            "authorization_grant_sha256",
            "current-tip bundled authorization grant",
        )
        expected_grant = build_consumed_stage_authorization_grant(
            authenticated_store_snapshot=snapshot,
            external_store_state_pin=pin,
            expected_new_consumption_entry_sha256=_sha256(
                grant.get("consumption_entry_sha256"),
                "current-tip bundled consumption entry hash",
            ),
        )
        if grant != expected_grant:
            raise SecFilingGemmaStageAuthorizationError(
                "Current-tip authorization bundle grant is not exact"
            )
        validated[request_hash] = bundle
    return validated


def _validated_consumed_stage_output_receipts(
    raw: Any,
    *,
    authorization_bundles: Mapping[str, Any],
    sec_execution_claims: Mapping[str, Any],
    sec_reader_receipts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip consumed-stage output receipts")
    bundles = _mapping(
        authorization_bundles,
        "current-tip authorization bundles for output receipts",
    )
    claims = _mapping(
        sec_execution_claims,
        "current-tip SEC claims for output receipts",
    )
    readers = _mapping(
        sec_reader_receipts,
        "current-tip SEC reader receipts for output receipts",
    )
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_receipt in receipts.items():
        request_hash = _sha256(
            request_sha256,
            "current-tip consumed-stage output-receipt key",
        )
        receipt = _mapping(
            raw_receipt,
            f"current-tip consumed-stage output receipt {request_hash}",
        )
        _expect_keys(
            receipt,
            _STAGE_OUTPUT_RECEIPT_KEYS,
            f"current-tip consumed-stage output receipt {request_hash}",
        )
        if (
            receipt["schema_version"]
            != CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "first_output_for_exact_consumed_stage_grant"
            or receipt["output_kind"] != "next_stage_evidence"
            or receipt["output_stage_evidence_schema_version"]
            != STAGE_EVIDENCE_SCHEMA_VERSION
            or receipt["output_stage_evidence_component_id"]
            != STAGE_EVIDENCE_OUTPUT_COMPONENT_ID
            or receipt["output_stage_evidence_relative_path"]
            != STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH
            or receipt["output_stage_evidence_recomputed_by_store"] is not True
            or receipt["fresh_stage_evidence_provenance_claimed"] is not False
            or receipt["cross_stage_output_permitted"] is not False
            or receipt["grant_reuse_for_different_output_permitted"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt semantics changed"
            )
        _self_hash(
            receipt,
            "output_receipt_sha256",
            f"current-tip consumed-stage output receipt {request_hash}",
        )
        for field in (
            "consumption_entry_sha256",
            "request_sha256",
            "candidate_sha256",
            "registry_entry_sha256",
            "input_stage_evidence_sha256",
            "stage_access_manifest_sha256",
            "authorization_bundle_sha256",
            "authorization_grant_sha256",
            "sec_execution_claim_sha256",
            "sec_reader_receipt_sha256",
            "grant_store_state_sha256",
            "grant_consumption_ledger_sha256",
            "grant_consumption_ledger_tip_sha256",
            "output_stage_evidence_sha256",
            "output_stage_evidence_document_sha256",
            "output_stage_evidence_complete_marker_sha256",
            "output_parent_stage_evidence_sha256",
            "output_candidate_sha256",
        ):
            _sha256(receipt[field], f"consumed-stage output receipt {field}")
        _strict_int(
            receipt["consumption_entry_sequence"],
            "consumed-stage output receipt entry sequence",
            minimum=1,
        )
        _strict_int(
            receipt["output_stage_evidence_canonical_byte_count"],
            "consumed-stage output receipt evidence byte count",
            minimum=2,
        )
        for field in (
            "attempt_id",
            "input_prerequisite_stage",
            "output_stage",
            "output_stage_evidence_schema_version",
            "output_stage_evidence_prerequisite_stage",
        ):
            _safe_id(receipt[field], f"consumed-stage output receipt {field}")
        namespace = receipt["output_namespace"]
        if (
            type(namespace) is not str
            or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt namespace is invalid"
            )
        if receipt["request_sha256"] != request_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt is stored under another request"
            )
        bundle = bundles.get(request_hash)
        if type(bundle) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt lacks its persisted authorization bundle"
            )
        grant = _mapping(
            bundle.get("authorization_grant"),
            "consumed-stage output receipt authorization grant",
        )
        sec_claim = claims.get(request_hash)
        sec_reader = readers.get(request_hash)
        if type(sec_claim) is not dict or type(sec_reader) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt lacks its SEC execution ancestry"
            )
        expected_bindings = {
            "consumption_entry_sha256": grant.get("consumption_entry_sha256"),
            "consumption_entry_sequence": grant.get("consumption_entry_sequence"),
            "request_sha256": grant.get("request_sha256"),
            "attempt_id": grant.get("attempt_id"),
            "candidate_sha256": grant.get("candidate_sha256"),
            "registry_entry_sha256": grant.get("registry_entry_sha256"),
            "input_prerequisite_stage": grant.get("prerequisite_stage"),
            "output_stage": grant.get("stage"),
            "input_stage_evidence_sha256": grant.get(
                "prerequisite_stage_evidence_sha256"
            ),
            "stage_access_manifest_sha256": grant.get(
                "stage_access_manifest_sha256"
            ),
            "output_namespace": grant.get("output_namespace"),
            "authorization_bundle_sha256": bundle.get("bundle_sha256"),
            "authorization_grant_sha256": grant.get(
                "authorization_grant_sha256"
            ),
            "sec_execution_claim_sha256": sec_claim.get("claim_sha256"),
            "sec_reader_receipt_sha256": sec_reader.get("receipt_sha256"),
            "grant_store_state_sha256": grant.get("store_state_sha256"),
            "grant_consumption_ledger_sha256": grant.get(
                "consumption_ledger_sha256"
            ),
            "grant_consumption_ledger_tip_sha256": grant.get(
                "consumption_ledger_tip_sha256"
            ),
            "output_stage_evidence_prerequisite_stage": grant.get("stage"),
            "output_parent_stage_evidence_sha256": grant.get(
                "prerequisite_stage_evidence_sha256"
            ),
            "output_candidate_sha256": grant.get("candidate_sha256"),
        }
        if any(receipt[field] != expected for field, expected in expected_bindings.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt crossed its grant or output boundary"
            )
        validated[request_hash] = receipt
    return validated


def _validated_sec_byte_index(raw: Any) -> list[dict[str, Any]]:
    if type(raw) is not list or not raw:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC reader byte index must be a non-empty exact list"
        )
    validated: list[dict[str, Any]] = []
    logical_ids: set[str] = set()
    relative_paths: set[str] = set()
    for ordinal, raw_item in enumerate(raw, start=1):
        item = _mapping(raw_item, f"SEC reader byte index item {ordinal}")
        _expect_keys(
            item,
            {"ordinal", "logical_id", "relative_path", "byte_count", "sha256"},
            f"SEC reader byte index item {ordinal}",
        )
        if _strict_int(item["ordinal"], "SEC byte ordinal", minimum=1) != ordinal:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader byte index ordinals are not contiguous"
            )
        logical_id = _safe_id(item["logical_id"], "SEC byte logical id")
        path = item["relative_path"]
        if (
            type(path) is not str
            or not path
            or len(path) > 240
            or "\\" in path
            or path.startswith("/")
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", part) is None
                for part in path.split("/")
            )
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader byte index contains an unsafe relative path"
            )
        if logical_id in logical_ids or path.casefold() in relative_paths:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader byte index contains a duplicate or case-colliding item"
            )
        logical_ids.add(logical_id)
        relative_paths.add(path.casefold())
        validated.append(
            {
                "ordinal": ordinal,
                "logical_id": logical_id,
                "relative_path": path,
                "byte_count": _strict_int(
                    item["byte_count"], "SEC byte count", minimum=1
                ),
                "sha256": _sha256(item["sha256"], "SEC byte SHA-256"),
            }
        )
    return validated


def _validated_stage_sec_execution_claims(
    raw: Any,
    *,
    authorization_bundles: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    claims = _mapping(raw, "current-tip SEC execution claims")
    bundles = _mapping(
        authorization_bundles,
        "current-tip authorization bundles for SEC execution claims",
    )
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_claim in claims.items():
        request_hash = _sha256(request_sha256, "SEC execution claim map key")
        claim = _mapping(raw_claim, f"SEC execution claim {request_hash}")
        _expect_keys(
            claim,
            _STAGE_SEC_EXECUTION_CLAIM_KEYS,
            f"SEC execution claim {request_hash}",
        )
        if (
            claim["schema_version"] != STAGE_SEC_EXECUTION_CLAIM_SCHEMA_VERSION
            or claim["contract_version"] != CONTRACT_VERSION
            or claim["claim_kind"] != "owned_sec_stage_document_batch"
            or claim["runner_repository_path"] != STAGE_RUNNER_REPOSITORY_PATH
            or claim["sec_corpus_repository_path"] != SEC_CORPUS_REPOSITORY_PATH
            or claim["sec_component_id"]
            != SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
            or claim["effect_may_be_repeated_after_indeterminate_crash"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim semantics changed"
            )
        _self_hash(claim, "claim_sha256", "SEC execution claim")
        raw_execution_sources = _mapping(
            claim["execution_source_hashes"],
            "SEC execution claim source hashes",
        )
        expected_source_roles = {
            role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
        }
        if set(raw_execution_sources) != expected_source_roles:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim does not bind the complete resolved source set"
            )
        execution_sources = {
            role: _sha256(
                raw_execution_sources[role],
                f"SEC execution claim source hash {role}",
            )
            for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
        }
        if (
            claim["execution_source_hashes"] != execution_sources
            or claim["execution_source_hashes_sha256"]
            != canonical_sha256(execution_sources)
            or claim["execution_source_role_count"] != len(execution_sources)
            or claim["runner_source_sha256"] != execution_sources["runner"]
            or claim["sec_corpus_source_sha256"]
            != execution_sources["sec_corpus_selector"]
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim source closure is inconsistent"
            )
        for field in (
            "request_sha256",
            "consumption_entry_sha256",
            "candidate_sha256",
            "registry_entry_sha256",
            "input_stage_evidence_sha256",
            "stage_access_manifest_sha256",
            "authorization_bundle_sha256",
            "authorization_grant_sha256",
            "grant_store_state_sha256",
            "grant_consumption_ledger_sha256",
            "grant_consumption_ledger_tip_sha256",
            "start_current_tip_anchor_sha256",
            "runner_source_sha256",
            "sec_corpus_source_sha256",
            "execution_source_hashes_sha256",
            "sec_component_plan_sha256",
        ):
            _sha256(claim[field], f"SEC execution claim {field}")
        _tagged_sha256(
            claim["sec_user_agent_sha256"],
            "SEC execution claim sec_user_agent_sha256",
        )
        _strict_int(
            claim["consumption_entry_sequence"],
            "SEC execution claim entry sequence",
            minimum=1,
        )
        _strict_int(
            claim["execution_source_role_count"],
            "SEC execution claim source-role count",
            minimum=1,
        )
        for field in (
            "attempt_id",
            "input_prerequisite_stage",
            "authorized_stage",
        ):
            _safe_id(claim[field], f"SEC execution claim {field}")
        namespace = claim["output_namespace"]
        if (
            type(namespace) is not str
            or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim namespace is invalid"
            )
        if claim["request_sha256"] != request_hash:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim is stored under another request"
            )
        bundle = bundles.get(request_hash)
        if type(bundle) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim lacks its persisted grant bundle"
            )
        grant = _mapping(
            bundle.get("authorization_grant"),
            "SEC execution claim grant",
        )
        expected = {
            "request_sha256": grant.get("request_sha256"),
            "consumption_entry_sha256": grant.get("consumption_entry_sha256"),
            "consumption_entry_sequence": grant.get("consumption_entry_sequence"),
            "attempt_id": grant.get("attempt_id"),
            "candidate_sha256": grant.get("candidate_sha256"),
            "registry_entry_sha256": grant.get("registry_entry_sha256"),
            "input_prerequisite_stage": grant.get("prerequisite_stage"),
            "authorized_stage": grant.get("stage"),
            "input_stage_evidence_sha256": grant.get(
                "prerequisite_stage_evidence_sha256"
            ),
            "stage_access_manifest_sha256": grant.get(
                "stage_access_manifest_sha256"
            ),
            "output_namespace": grant.get("output_namespace"),
            "authorization_bundle_sha256": bundle.get("bundle_sha256"),
            "authorization_grant_sha256": grant.get(
                "authorization_grant_sha256"
            ),
            "grant_store_state_sha256": grant.get("store_state_sha256"),
            "grant_consumption_ledger_sha256": grant.get(
                "consumption_ledger_sha256"
            ),
            "grant_consumption_ledger_tip_sha256": grant.get(
                "consumption_ledger_tip_sha256"
            ),
        }
        if any(claim[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim crossed its persisted grant"
            )
        validated[request_hash] = claim
    return validated


def _validated_stage_sec_reader_receipts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    receipts = _mapping(raw, "current-tip SEC reader receipts")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_receipt in receipts.items():
        request_hash = _sha256(request_sha256, "SEC reader receipt map key")
        receipt = _mapping(raw_receipt, f"SEC reader receipt {request_hash}")
        _expect_keys(
            receipt,
            _STAGE_SEC_READER_RECEIPT_KEYS,
            f"SEC reader receipt {request_hash}",
        )
        if (
            receipt["schema_version"] != STAGE_SEC_READER_RECEIPT_SCHEMA_VERSION
            or receipt["contract_version"] != CONTRACT_VERSION
            or receipt["receipt_kind"]
            != "store_rehashed_owned_sec_stage_document_batch"
            or receipt["sec_component_id"]
            != SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID
            or receipt["fresh_network_provenance_claimed"] is not False
            or receipt["reader_output_recomputed_by_store"] is not True
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader receipt semantics changed"
            )
        _self_hash(receipt, "receipt_sha256", "SEC reader receipt")
        byte_index = _validated_sec_byte_index(receipt["byte_index"])
        if (
            receipt["byte_index"] != byte_index
            or receipt["byte_index_sha256"] != canonical_sha256(byte_index)
            or receipt["byte_count_total"]
            != sum(item["byte_count"] for item in byte_index)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader receipt byte index is inconsistent"
            )
        _strict_int(
            receipt["byte_count_total"],
            "SEC reader receipt total byte count",
            minimum=1,
        )
        _strict_int(
            receipt["execution_source_role_count"],
            "SEC reader receipt source-role count",
            minimum=1,
        )
        for field in (
            "request_sha256",
            "claim_sha256",
            "candidate_sha256",
            "authorization_bundle_sha256",
            "authorization_grant_sha256",
            "sec_component_plan_sha256",
            "runner_source_sha256",
            "sec_corpus_source_sha256",
            "execution_source_hashes_sha256",
            "byte_index_sha256",
            "complete_marker_sha256",
        ):
            _sha256(receipt[field], f"SEC reader receipt {field}")
        _tagged_sha256(
            receipt["sec_user_agent_sha256"],
            "SEC reader receipt sec_user_agent_sha256",
        )
        claim = claims.get(request_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader receipt lacks its execution claim"
            )
        expected = {
            "request_sha256": request_hash,
            "claim_sha256": claim.get("claim_sha256"),
            "authorized_stage": claim.get("authorized_stage"),
            "candidate_sha256": claim.get("candidate_sha256"),
            "output_namespace": claim.get("output_namespace"),
            "authorization_bundle_sha256": claim.get(
                "authorization_bundle_sha256"
            ),
            "authorization_grant_sha256": claim.get(
                "authorization_grant_sha256"
            ),
            "sec_component_id": claim.get("sec_component_id"),
            "sec_component_plan_sha256": claim.get(
                "sec_component_plan_sha256"
            ),
            "runner_source_sha256": claim.get("runner_source_sha256"),
            "sec_corpus_source_sha256": claim.get("sec_corpus_source_sha256"),
            "execution_source_hashes_sha256": claim.get(
                "execution_source_hashes_sha256"
            ),
            "execution_source_role_count": claim.get(
                "execution_source_role_count"
            ),
            "sec_user_agent_sha256": claim.get("sec_user_agent_sha256"),
        }
        if any(receipt[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC reader receipt crossed its execution claim"
            )
        validated[request_hash] = receipt
    return validated


def _validated_stage_sec_execution_aborts(
    raw: Any,
    *,
    claims: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    aborts = _mapping(raw, "current-tip SEC execution aborts")
    validated: dict[str, dict[str, Any]] = {}
    for request_sha256, raw_abort in aborts.items():
        request_hash = _sha256(request_sha256, "SEC execution abort map key")
        abort = _mapping(raw_abort, f"SEC execution abort {request_hash}")
        _expect_keys(
            abort,
            _STAGE_SEC_EXECUTION_ABORT_KEYS,
            f"SEC execution abort {request_hash}",
        )
        if (
            abort["schema_version"] != STAGE_SEC_EXECUTION_ABORT_SCHEMA_VERSION
            or abort["contract_version"] != CONTRACT_VERSION
            or abort["abort_kind"] != "indeterminate_owned_sec_stage_execution"
            or abort["reason"]
            not in {
                "claim_recovered_without_terminal_receipt",
                "external_effect_failed_or_completion_unknown",
                "durable_output_verification_failed",
            }
            or abort["external_effect_retry_permitted"] is not False
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution abort semantics changed"
            )
        _self_hash(abort, "abort_sha256", "SEC execution abort")
        for field in (
            "request_sha256",
            "claim_sha256",
            "candidate_sha256",
        ):
            _sha256(abort[field], f"SEC execution abort {field}")
        claim = claims.get(request_hash)
        if type(claim) is not dict:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution abort lacks its execution claim"
            )
        expected = {
            "request_sha256": request_hash,
            "claim_sha256": claim.get("claim_sha256"),
            "authorized_stage": claim.get("authorized_stage"),
            "candidate_sha256": claim.get("candidate_sha256"),
            "output_namespace": claim.get("output_namespace"),
        }
        if any(abort[field] != value for field, value in expected.items()):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution abort crossed its claim"
            )
        validated[request_hash] = abort
    return validated


def validate_reveal_store_current_tip_anchor_structure(
    current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a detached stable tip anchor without choosing a store state.

    This structural form is used by the effectful store while resolving an
    interrupted two-file transaction.  It does *not* authenticate a snapshot;
    callers authorizing downstream access must use
    :func:`validate_reveal_store_current_tip_anchor` instead.
    """

    anchor = _mapping(current_tip_anchor, "independent current-tip anchor")
    _expect_keys(anchor, _CURRENT_TIP_ANCHOR_KEYS, "independent current-tip anchor")
    if (
        anchor["schema_version"]
        != REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION
        or anchor["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Independent current-tip anchor schema or contract changed"
        )
    revision = _strict_int(anchor["revision"], "current-tip revision")
    previous = anchor["previous_tip_anchor_sha256"]
    if revision == 0:
        if previous is not None:
            raise SecFilingGemmaStageAuthorizationError(
                "Genesis current-tip anchor cannot have a predecessor"
            )
    else:
        _sha256(previous, "previous current-tip anchor hash")
    for field in (
        "state_sha256",
        "state_snapshot_bytes_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
    ):
        _sha256(anchor[field], f"current-tip {field}")
    _strict_int(
        anchor["state_snapshot_byte_count"],
        "current-tip state snapshot byte count",
        minimum=1,
    )
    _strict_int(
        anchor["consumed_request_count"],
        "current-tip consumed request count",
    )
    anchor["trusted_stage_content_pins"] = _validated_trusted_stage_content_pins(
        anchor["trusted_stage_content_pins"]
    )
    anchor["authorization_bundles"] = _validated_authorization_bundles(
        anchor["authorization_bundles"]
    )
    anchor["stage_sec_execution_claims"] = _validated_stage_sec_execution_claims(
        anchor["stage_sec_execution_claims"],
        authorization_bundles=anchor["authorization_bundles"],
    )
    anchor["stage_sec_reader_receipts"] = _validated_stage_sec_reader_receipts(
        anchor["stage_sec_reader_receipts"],
        claims=anchor["stage_sec_execution_claims"],
    )
    anchor["stage_sec_execution_aborts"] = _validated_stage_sec_execution_aborts(
        anchor["stage_sec_execution_aborts"],
        claims=anchor["stage_sec_execution_claims"],
    )
    anchor["consumed_stage_output_receipts"] = (
        _validated_consumed_stage_output_receipts(
            anchor["consumed_stage_output_receipts"],
            authorization_bundles=anchor["authorization_bundles"],
            sec_execution_claims=anchor["stage_sec_execution_claims"],
            sec_reader_receipts=anchor["stage_sec_reader_receipts"],
        )
    )
    if set(anchor["stage_sec_reader_receipts"]) & set(
        anchor["stage_sec_execution_aborts"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution cannot be both completed and aborted"
        )
    active_claims = set(anchor["stage_sec_execution_claims"]) - set(
        anchor["stage_sec_reader_receipts"]
    ) - set(anchor["stage_sec_execution_aborts"])
    if len(active_claims) > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "At most one SEC execution claim may be active"
        )
    _self_hash(anchor, "tip_anchor_sha256", "independent current-tip anchor")
    return anchor


def build_reveal_store_current_tip_anchor(
    authenticated_store_snapshot: Mapping[str, Any],
    *,
    revision: int,
    previous_tip_anchor_sha256: str | None,
    authorization_bundles: Mapping[str, Any],
    trusted_stage_content_pins: Mapping[str, Any] | None = None,
    consumed_stage_output_receipts: Mapping[str, Any] | None = None,
    stage_sec_execution_claims: Mapping[str, Any] | None = None,
    stage_sec_reader_receipts: Mapping[str, Any] | None = None,
    stage_sec_execution_aborts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the separately persisted CAS anchor for one exact store state."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    current_revision = _strict_int(revision, "current-tip revision")
    if current_revision == 0:
        if previous_tip_anchor_sha256 is not None:
            raise SecFilingGemmaStageAuthorizationError(
                "Genesis current-tip anchor cannot have a predecessor"
            )
    else:
        _sha256(
            previous_tip_anchor_sha256,
            "previous current-tip anchor hash",
        )
    bundles = _validated_authorization_bundles(authorization_bundles)
    pins = _validated_trusted_stage_content_pins(
        {} if trusted_stage_content_pins is None else trusted_stage_content_pins
    )
    sec_claims = _validated_stage_sec_execution_claims(
        {} if stage_sec_execution_claims is None else stage_sec_execution_claims,
        authorization_bundles=bundles,
    )
    sec_receipts = _validated_stage_sec_reader_receipts(
        {} if stage_sec_reader_receipts is None else stage_sec_reader_receipts,
        claims=sec_claims,
    )
    sec_aborts = _validated_stage_sec_execution_aborts(
        {} if stage_sec_execution_aborts is None else stage_sec_execution_aborts,
        claims=sec_claims,
    )
    if set(sec_receipts) & set(sec_aborts):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution cannot be both completed and aborted"
        )
    if len(set(sec_claims) - set(sec_receipts) - set(sec_aborts)) > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "At most one SEC execution claim may be active"
        )
    output_receipts = _validated_consumed_stage_output_receipts(
        (
            {}
            if consumed_stage_output_receipts is None
            else consumed_stage_output_receipts
        ),
        authorization_bundles=bundles,
        sec_execution_claims=sec_claims,
        sec_reader_receipts=sec_receipts,
    )
    state_bytes = _encoded_store_snapshot(state)
    body = {
        "schema_version": REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "revision": current_revision,
        "previous_tip_anchor_sha256": previous_tip_anchor_sha256,
        "state_sha256": state["state_sha256"],
        "state_snapshot_bytes_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "state_snapshot_byte_count": len(state_bytes),
        "registry_sha256": state["latest_registry"]["registry_sha256"],
        "registry_tip_sha256": state["latest_registry_pin"]["tip_sha256"],
        "consumption_ledger_sha256": ledger["ledger_sha256"],
        "consumption_ledger_tip_sha256": ledger["chain"]["tip_sha256"],
        "consumed_request_count": ledger["chain"]["consumed_request_count"],
        "trusted_stage_content_pins": pins,
        "authorization_bundles": bundles,
        "consumed_stage_output_receipts": output_receipts,
        "stage_sec_execution_claims": sec_claims,
        "stage_sec_reader_receipts": sec_receipts,
        "stage_sec_execution_aborts": sec_aborts,
    }
    return {**body, "tip_anchor_sha256": canonical_sha256(body)}


def validate_reveal_store_current_tip_anchor_transition(
    prior_tip_anchor: Mapping[str, Any],
    next_tip_anchor: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one monotonic, bundle-preserving CAS-anchor transition."""

    prior = validate_reveal_store_current_tip_anchor_structure(prior_tip_anchor)
    next_anchor = validate_reveal_store_current_tip_anchor_structure(next_tip_anchor)
    if (
        next_anchor["revision"] != prior["revision"] + 1
        or next_anchor["previous_tip_anchor_sha256"]
        != prior["tip_anchor_sha256"]
        or next_anchor["consumed_request_count"]
        - prior["consumed_request_count"]
        not in {0, 1}
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip anchor transition is stale, non-monotonic, or forked"
        )
    prior_bundles = prior["authorization_bundles"]
    next_bundles = next_anchor["authorization_bundles"]
    if any(next_bundles.get(key) != value for key, value in prior_bundles.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip anchor transition removed or changed a persisted bundle"
        )
    bundle_delta = len(next_bundles) - len(prior_bundles)
    consumption_delta = (
        next_anchor["consumed_request_count"]
        - prior["consumed_request_count"]
    )
    if bundle_delta not in {0, 1} or bundle_delta > consumption_delta:
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one authorization bundle"
        )
    prior_pins = prior["trusted_stage_content_pins"]
    next_pins = next_anchor["trusted_stage_content_pins"]
    if any(next_pins.get(key) != value for key, value in prior_pins.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip anchor transition removed or changed a trusted content pin"
        )
    pin_delta = len(next_pins) - len(prior_pins)
    if pin_delta not in {0, 1}:
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one trusted content pin"
        )
    if pin_delta:
        if consumption_delta != 0 or bundle_delta != 0:
            raise SecFilingGemmaStageAuthorizationError(
                "Trusted content pin append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "Trusted content pin append changed the authenticated store state"
            )
        new_request_hash = next(iter(set(next_pins) - set(prior_pins)))
        if next_pins[new_request_hash]["trusted_store_state_sha256"] != prior[
            "state_sha256"
        ]:
            raise SecFilingGemmaStageAuthorizationError(
                "Trusted content pin does not bind the pre-consumption state"
            )
    elif consumption_delta == 1 and set(next_pins) != set(prior_pins):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumption transition changed trusted content pin membership"
        )
    prior_outputs = prior["consumed_stage_output_receipts"]
    next_outputs = next_anchor["consumed_stage_output_receipts"]
    if any(next_outputs.get(key) != value for key, value in prior_outputs.items()):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip anchor transition removed or changed a consumed-stage output receipt"
        )
    output_delta = len(next_outputs) - len(prior_outputs)
    if output_delta not in {0, 1}:
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one consumed-stage output receipt"
        )
    if output_delta:
        if consumption_delta != 0 or bundle_delta != 0 or pin_delta != 0:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt append changed the authenticated store state"
            )
        new_request_hash = next(iter(set(next_outputs) - set(prior_outputs)))
        if new_request_hash not in prior_bundles:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt was appended before its grant bundle"
            )
        new_receipt = next_outputs[new_request_hash]
        current_grant_bindings = {
            "grant_store_state_sha256": prior["state_sha256"],
            "grant_consumption_ledger_sha256": prior[
                "consumption_ledger_sha256"
            ],
            "grant_consumption_ledger_tip_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sequence": prior["consumed_request_count"],
        }
        if any(
            new_receipt[field] != expected
            for field, expected in current_grant_bindings.items()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed-stage output receipt does not bind the exact current grant tip"
            )
    elif set(next_outputs) != set(prior_outputs):
        raise SecFilingGemmaStageAuthorizationError(
            "Non-output transition changed consumed-stage output receipt membership"
        )

    prior_sec_claims = prior["stage_sec_execution_claims"]
    next_sec_claims = next_anchor["stage_sec_execution_claims"]
    prior_sec_receipts = prior["stage_sec_reader_receipts"]
    next_sec_receipts = next_anchor["stage_sec_reader_receipts"]
    prior_sec_aborts = prior["stage_sec_execution_aborts"]
    next_sec_aborts = next_anchor["stage_sec_execution_aborts"]
    for prior_map, next_map, label in (
        (prior_sec_claims, next_sec_claims, "SEC execution claim"),
        (prior_sec_receipts, next_sec_receipts, "SEC reader receipt"),
        (prior_sec_aborts, next_sec_aborts, "SEC execution abort"),
    ):
        if any(next_map.get(key) != value for key, value in prior_map.items()):
            raise SecFilingGemmaStageAuthorizationError(
                f"Current-tip transition removed or changed a persisted {label}"
            )
    claim_delta = len(next_sec_claims) - len(prior_sec_claims)
    reader_delta = len(next_sec_receipts) - len(prior_sec_receipts)
    abort_delta = len(next_sec_aborts) - len(prior_sec_aborts)
    if any(delta not in {0, 1} for delta in (claim_delta, reader_delta, abort_delta)):
        raise SecFilingGemmaStageAuthorizationError(
            "Current-tip transition may append at most one SEC execution artifact"
        )
    sec_delta_count = claim_delta + reader_delta + abort_delta
    if sec_delta_count > 1:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC claim, reader receipt, and abort require separate transitions"
        )
    prior_active = set(prior_sec_claims) - set(prior_sec_receipts) - set(
        prior_sec_aborts
    )
    if prior_active:
        active_request = next(iter(prior_active))
        terminal_request: str | None = None
        if reader_delta:
            terminal_request = next(
                iter(set(next_sec_receipts) - set(prior_sec_receipts))
            )
        elif abort_delta:
            terminal_request = next(iter(set(next_sec_aborts) - set(prior_sec_aborts)))
        if (
            claim_delta
            or sec_delta_count != 1
            or terminal_request != active_request
            or consumption_delta
            or bundle_delta
            or pin_delta
            or output_delta
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Active SEC execution claim blocks every transition except its exact terminal receipt"
            )
    if sec_delta_count:
        if consumption_delta or bundle_delta or pin_delta or output_delta:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution artifact append must be a dedicated tip-only transition"
            )
        immutable_state_fields = (
            "state_sha256",
            "state_snapshot_bytes_sha256",
            "state_snapshot_byte_count",
            "registry_sha256",
            "registry_tip_sha256",
            "consumption_ledger_sha256",
            "consumption_ledger_tip_sha256",
            "consumed_request_count",
        )
        if any(next_anchor[field] != prior[field] for field in immutable_state_fields):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution artifact append changed authenticated store state"
            )
    if claim_delta:
        request_hash = next(iter(set(next_sec_claims) - set(prior_sec_claims)))
        claim = next_sec_claims[request_hash]
        current_bindings = {
            "start_current_tip_anchor_sha256": prior["tip_anchor_sha256"],
            "grant_store_state_sha256": prior["state_sha256"],
            "grant_consumption_ledger_sha256": prior[
                "consumption_ledger_sha256"
            ],
            "grant_consumption_ledger_tip_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sha256": prior[
                "consumption_ledger_tip_sha256"
            ],
            "consumption_entry_sequence": prior["consumed_request_count"],
        }
        if request_hash not in prior_bundles or any(
            claim[field] != expected
            for field, expected in current_bindings.items()
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim does not bind the exact current grant tip"
            )
        expected_claim = build_stage_sec_execution_claim(
            prior_bundles[request_hash],
            independent_current_tip_anchor=prior,
            execution_source_hashes=claim["execution_source_hashes"],
            sec_user_agent_sha256=claim["sec_user_agent_sha256"],
        )
        if claim != expected_claim:
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution claim differs from the exact authenticated component plan"
            )
    if not sec_delta_count:
        if (
            set(next_sec_claims) != set(prior_sec_claims)
            or set(next_sec_receipts) != set(prior_sec_receipts)
            or set(next_sec_aborts) != set(prior_sec_aborts)
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "Non-SEC transition changed SEC execution membership"
            )
    return prior, next_anchor


def validate_reveal_store_current_tip_anchor(
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate a snapshot against the independently loaded latest tip."""

    state, _ledger = _validated_store_snapshot(authenticated_store_snapshot)
    observed = validate_reveal_store_current_tip_anchor_structure(
        independent_current_tip_anchor
    )
    expected = build_reveal_store_current_tip_anchor(
        state,
        revision=observed["revision"],
        previous_tip_anchor_sha256=observed["previous_tip_anchor_sha256"],
        authorization_bundles=observed["authorization_bundles"],
        trusted_stage_content_pins=observed["trusted_stage_content_pins"],
        consumed_stage_output_receipts=observed[
            "consumed_stage_output_receipts"
        ],
        stage_sec_execution_claims=observed["stage_sec_execution_claims"],
        stage_sec_reader_receipts=observed["stage_sec_reader_receipts"],
        stage_sec_execution_aborts=observed["stage_sec_execution_aborts"],
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Independent current-tip anchor does not authenticate this exact latest state"
        )
    return observed


def _validated_latest_consumption(
    state: Mapping[str, Any], *, expected_entry_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ledger = state["consumption_ledger"]
    entries = ledger["entries"]
    expected_entry = _sha256(expected_entry_sha256, "expected consumption entry hash")
    if not entries:
        raise SecFilingGemmaStageAuthorizationError(
            "No consumed stage exists from which to mint a grant"
        )
    entry = entries[-1]
    if (
        entry["entry_sha256"] != expected_entry
        or ledger["chain"]["tip_sha256"] != expected_entry
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Grant entry is not the exact newly appended consumption-ledger tip"
        )
    request = _mapping(entry["request"], "consumed reveal request")
    _expect_keys(request, _REQUEST_KEYS, "consumed reveal request")
    request_hash = _self_hash(request, "request_sha256", "consumed reveal request")
    stage = request["stage"]
    prerequisite = request["prerequisite_stage"]
    if (
        request["schema_version"] != REVEAL_REQUEST_SCHEMA_VERSION
        or request["contract_version"] != CONTRACT_VERSION
        or _STAGE_PREREQUISITES.get(stage) != prerequisite
        or request["authorizes_outcome_access"] is not False
        or request["effectful_atomic_single_use_consumption_required"] is not True
        or request["cross_attempt_comparison_permitted"] is not False
        or request["cross_attempt_winner_selection_permitted"] is not False
        or request["globally_pristine_claim"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed request changed its frozen non-authorizing semantics"
        )
    for field in (
        "registry_sha256",
        "registry_tip_sha256",
        "stage_access_manifest_sha256",
        "prerequisite_stage_evidence_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
    ):
        _sha256(request[field], f"consumed request {field}")
    _safe_id(request["attempt_id"], "consumed request attempt id")
    if (
        entry["request_sha256"] != request_hash
        or entry["stage"] != stage
        or entry["attempt_id"] != request["attempt_id"]
        or entry["candidate_sha256"] != request["candidate_sha256"]
        or entry["registry_entry_sha256"] != request["registry_entry_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumption entry is not bound to its exact request"
        )

    access = _mapping(entry["stage_access_manifest"], "consumed stage-access manifest")
    access_hash = _self_hash(
        access,
        "stage_access_manifest_sha256",
        "consumed stage-access manifest",
    )
    transition = _mapping(access.get("transition"), "stage-access transition")
    candidate = _mapping(access.get("candidate"), "stage-access candidate")
    output = _mapping(access.get("output"), "stage-access output")
    scope = _mapping(access.get("scope"), "stage-access scope")
    _expect_keys(
        output,
        frozenset(
            {
                "namespace",
                "write_mode",
                "existing_namespace_reuse_permitted",
                "cross_stage_write_permitted",
            }
        ),
        "stage-access output",
    )
    expected_namespace = f"aapl-sec-gemma-{request['attempt_id']}-{stage}"
    namespace = output["namespace"]
    if (
        access.get("schema_version") != STAGE_ACCESS_MANIFEST_SCHEMA_VERSION
        or access.get("contract_version") != CONTRACT_VERSION
        or access_hash != request["stage_access_manifest_sha256"]
        or transition.get("prerequisite_stage") != prerequisite
        or transition.get("requested_stage") != stage
        or transition.get("single_use_consumption_required") is not True
        or transition.get("stage_reuse_permitted") is not False
        or candidate.get("candidate_sha256") != request["candidate_sha256"]
        or candidate.get("candidate_design_sha256")
        != request["candidate_design_sha256"]
        or candidate.get("attempt_id") != request["attempt_id"]
        or not isinstance(namespace, str)
        or _OUTPUT_NAMESPACE_RE.fullmatch(namespace) is None
        or namespace != expected_namespace
        or output["write_mode"] != "create_new_exclusive"
        or output["existing_namespace_reuse_permitted"] is not False
        or output["cross_stage_write_permitted"] is not False
        or scope.get("authorized_stage") != stage
        or scope.get("future_stage_access_permitted") is not False
        or scope.get("outcome_access_before_atomic_request_consumption_permitted")
        is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed stage-access manifest crossed an identity or namespace boundary"
        )

    validation = _mapping(
        entry["prerequisite_validation"], "consumed prerequisite validation"
    )
    _expect_keys(validation, _VALIDATION_KEYS, "consumed prerequisite validation")
    _self_hash(validation, "result_sha256", "consumed prerequisite validation")
    semantic_receipt = _mapping(
        validation["semantic_receipt"], "consumed semantic receipt"
    )
    checks = validation["semantic_checks"]
    if (
        validation["schema_version"] != SEMANTIC_PREREQUISITE_SCHEMA_VERSION
        or validation["validation_kind"]
        != "independent_semantic_prerequisite_replay"
        or validation["semantic_validation_completed"] is not True
        or validation["authorizes_outcome_access"] is not False
        or checks != list(REQUIRED_STAGE_VERIFIER_CHECKS)
        or not semantic_receipt
        or canonical_sha256(semantic_receipt)
        != validation["semantic_receipt_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed prerequisite validation is not exact semantic evidence"
        )
    for request_field, validation_field in (
        ("prerequisite_stage", "prerequisite_stage"),
        (
            "prerequisite_stage_evidence_sha256",
            "prerequisite_stage_evidence_sha256",
        ),
        ("attempt_id", "attempt_id"),
        ("candidate_sha256", "candidate_sha256"),
        ("candidate_design_sha256", "candidate_design_sha256"),
        ("registry_entry_sha256", "registry_entry_sha256"),
        ("request_sha256", "request_sha256"),
        ("stage_access_manifest_sha256", "stage_access_manifest_sha256"),
        ("registry_sha256", "registry_sha256"),
        ("registry_tip_sha256", "registry_tip_sha256"),
    ):
        if request[request_field] != validation[validation_field]:
            raise SecFilingGemmaStageAuthorizationError(
                "Consumed prerequisite validation lost its request binding"
            )
    if validation["requested_stage"] != stage:
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed prerequisite validation crossed a stage"
        )
    return entry, request, access


def build_consumed_stage_authorization_grant(
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    external_store_state_pin: Mapping[str, Any],
    expected_new_consumption_entry_sha256: str,
) -> dict[str, Any]:
    """Mint a compact grant for the exact authenticated consumption-ledger tip."""

    state, ledger = _validated_store_snapshot(authenticated_store_snapshot)
    pin = validate_consumed_stage_store_state_pin(state, external_store_state_pin)
    entry, request, access = _validated_latest_consumption(
        state,
        expected_entry_sha256=expected_new_consumption_entry_sha256,
    )
    body = {
        "schema_version": CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION,
        "authorization_kind": "exact_consumed_stage_output_namespace_access",
        "consumption_entry_sha256": entry["entry_sha256"],
        "consumption_entry_sequence": entry["sequence"],
        "request_sha256": request["request_sha256"],
        "attempt_id": request["attempt_id"],
        "candidate_sha256": request["candidate_sha256"],
        "registry_entry_sha256": request["registry_entry_sha256"],
        "prerequisite_stage": request["prerequisite_stage"],
        "stage": request["stage"],
        "prerequisite_stage_evidence_sha256": request[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": request["stage_access_manifest_sha256"],
        "output_namespace": access["output"]["namespace"],
        "store_state_sha256": state["state_sha256"],
        "store_snapshot_bytes_sha256": pin["store_snapshot_bytes_sha256"],
        "store_snapshot_byte_count": pin["store_snapshot_byte_count"],
        "store_pin_sha256": pin["store_pin_sha256"],
        "consumption_ledger_sha256": ledger["ledger_sha256"],
        "consumption_ledger_tip_sha256": ledger["chain"]["tip_sha256"],
        "consumed_stage_access_authorized": True,
        "authorization_scope": "exact_consumed_stage_output_namespace_only",
        "outcomes_included": False,
        "market_values_included": False,
        "cross_stage_access_permitted": False,
        "grant_reuse_across_store_state_tips_permitted": False,
    }
    return {**body, "authorization_grant_sha256": canonical_sha256(body)}


def validate_consumed_stage_authorization_grant(
    grant: Mapping[str, Any],
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    external_store_state_pin: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    expected_consumption_entry_sha256: str,
    expected_request_sha256: str,
    expected_candidate_sha256: str,
    expected_stage: str,
    expected_prerequisite_stage_evidence_sha256: str,
    expected_stage_access_manifest_sha256: str,
    expected_output_namespace: str,
) -> str:
    """Require an exact *current-tip* persisted grant and return its hash.

    The bundled state pin is deliberately insufficient as a trust root: an
    old snapshot and its old pin still agree with one another.  Every
    downstream authorization therefore also needs the independently loaded
    latest CAS anchor, and the exact reconstructed bundle must be present in
    that anchor.
    """

    observed = _mapping(grant, "consumed-stage authorization grant")
    _expect_keys(observed, _GRANT_KEYS, "consumed-stage authorization grant")
    grant_hash = _self_hash(
        observed,
        "authorization_grant_sha256",
        "consumed-stage authorization grant",
    )
    expected = build_consumed_stage_authorization_grant(
        authenticated_store_snapshot=authenticated_store_snapshot,
        external_store_state_pin=external_store_state_pin,
        expected_new_consumption_entry_sha256=expected_consumption_entry_sha256,
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Authorization grant differs from the authenticated consumption entry"
        )
    current_anchor = validate_reveal_store_current_tip_anchor(
        authenticated_store_snapshot,
        independent_current_tip_anchor,
    )
    bundle_body = {
        "schema_version": CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION,
        "authenticated_store_snapshot": _mapping(
            authenticated_store_snapshot,
            "authenticated reveal-store snapshot",
        ),
        "store_state_pin": _mapping(
            external_store_state_pin,
            "external store-state pin",
        ),
        "authorization_grant": observed,
    }
    expected_bundle = {
        **bundle_body,
        "bundle_sha256": canonical_sha256(bundle_body),
    }
    if (
        current_anchor["authorization_bundles"].get(observed["request_sha256"])
        != expected_bundle
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Authorization grant bundle is not persisted at the independent current tip"
        )
    bindings = {
        "consumption_entry_sha256": _sha256(
            expected_consumption_entry_sha256, "expected consumption entry hash"
        ),
        "request_sha256": _sha256(expected_request_sha256, "expected request hash"),
        "candidate_sha256": _sha256(
            expected_candidate_sha256, "expected candidate hash"
        ),
        "stage": expected_stage,
        "prerequisite_stage_evidence_sha256": _sha256(
            expected_prerequisite_stage_evidence_sha256,
            "expected prerequisite-stage evidence hash",
        ),
        "stage_access_manifest_sha256": _sha256(
            expected_stage_access_manifest_sha256,
            "expected stage-access manifest hash",
        ),
        "output_namespace": expected_output_namespace,
    }
    if type(expected_stage) is not str or expected_stage not in _STAGE_PREREQUISITES:
        raise SecFilingGemmaStageAuthorizationError("Expected grant stage is invalid")
    if (
        type(expected_output_namespace) is not str
        or _OUTPUT_NAMESPACE_RE.fullmatch(expected_output_namespace) is None
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Expected output namespace is invalid"
        )
    for key, expected_value in bindings.items():
        if observed[key] != expected_value:
            raise SecFilingGemmaStageAuthorizationError(
                f"Authorization grant is not bound to expected {key}"
            )
    if (
        observed["consumed_stage_access_authorized"] is not True
        or observed["outcomes_included"] is not False
        or observed["market_values_included"] is not False
        or observed["cross_stage_access_permitted"] is not False
        or observed["grant_reuse_across_store_state_tips_permitted"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Authorization grant changed its narrow no-outcome scope"
        )
    return grant_hash


def _sec_component_plan_from_bundle(
    authorization_bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_bundle = _mapping(authorization_bundle, "SEC execution authorization bundle")
    raw_grant = _mapping(
        raw_bundle.get("authorization_grant"),
        "SEC execution authorization grant",
    )
    request_hash = _sha256(
        raw_grant.get("request_sha256"),
        "SEC execution authorization request hash",
    )
    bundle = _validated_authorization_bundles({request_hash: raw_bundle})[
        request_hash
    ]
    grant = bundle["authorization_grant"]
    snapshot = bundle["authenticated_store_snapshot"]
    entry, _request, _access = _validated_latest_consumption(
        snapshot,
        expected_entry_sha256=grant["consumption_entry_sha256"],
    )
    access = _mapping(entry["stage_access_manifest"], "SEC execution stage access")
    access_body = {
        key: access[key]
        for key in access
        if key != "stage_access_manifest_sha256"
    }
    if (
        access.get("stage_access_manifest_sha256")
        != grant["stage_access_manifest_sha256"]
        or canonical_sha256(access_body) != grant["stage_access_manifest_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution stage-access manifest lost its self-hash"
        )
    sec_plan = _mapping(access.get("sec_access_plan"), "SEC execution access plan")
    _expect_keys(
        sec_plan,
        {
            "selection_policy",
            "method",
            "network_scope",
            "redirects_permitted",
            "retries_permitted",
            "cache_substitution_permitted",
            "document_count",
            "accessions_sha256",
            "official_urls_sha256",
            "documents",
        },
        "SEC execution access plan",
    )
    if (
        sec_plan["selection_policy"]
        != "all_and_only_requested_stage_universe_primary_documents"
        or sec_plan["method"] != "GET"
        or sec_plan["network_scope"] != "official_sec_https_only"
        or sec_plan["redirects_permitted"] is not False
        or sec_plan["retries_permitted"] is not False
        or sec_plan["cache_substitution_permitted"] is not False
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution access plan permits an unsafe transport"
        )
    documents = sec_plan["documents"]
    if type(documents) is not list or not documents:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution access plan has no exact document list"
        )
    normalized_documents: list[dict[str, str]] = []
    for index, raw_document in enumerate(documents):
        document = _mapping(raw_document, f"SEC execution document {index}")
        _expect_keys(
            document,
            {"accession_number", "official_url"},
            f"SEC execution document {index}",
        )
        accession = _safe_id(
            document["accession_number"], f"SEC execution accession {index}"
        )
        url = document["official_url"]
        expected_prefix = (
            "https://www.sec.gov/Archives/edgar/data/320193/"
            f"{accession.replace('-', '')}/"
        )
        if (
            _AAPL_ACCESSION_RE.fullmatch(accession) is None
            or type(url) is not str
            or not url.startswith(expected_prefix)
            or len(url) <= len(expected_prefix)
            or any(token in url for token in ("?", "#", "\\", ".."))
        ):
            raise SecFilingGemmaStageAuthorizationError(
                "SEC execution document URL is outside the fixed Apple archive"
            )
        normalized_documents.append(
            {"accession_number": accession, "official_url": url}
        )
    accessions = [document["accession_number"] for document in normalized_documents]
    official_urls = [document["official_url"] for document in normalized_documents]
    if (
        sec_plan["documents"] != normalized_documents
        or sec_plan["document_count"] != len(normalized_documents)
        or accessions != sorted(accessions)
        or official_urls != sorted(official_urls)
        or len(accessions) != len(set(accessions))
        or len(official_urls) != len(set(official_urls))
        or sec_plan["accessions_sha256"] != canonical_sha256(accessions)
        or sec_plan["official_urls_sha256"] != canonical_sha256(official_urls)
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution document plan is not canonical"
        )
    budgets = _mapping(access.get("budgets"), "SEC execution budgets")
    authorized_max_response_bytes = _strict_int(
        budgets.get("max_sec_response_bytes"),
        "SEC execution byte budget",
        minimum=1,
    )
    component_plan = {
        "authorized_stage": grant["stage"],
        "sec_access_plan": sec_plan,
        "max_sec_requests": _strict_int(
            budgets.get("max_sec_requests"),
            "SEC execution request budget",
            minimum=1,
        ),
        "authorized_max_sec_response_bytes": authorized_max_response_bytes,
        "owned_sec_raw_batch_max_bytes": OWNED_SEC_RAW_BATCH_MAX_BYTES,
        "max_sec_response_bytes": min(
            authorized_max_response_bytes,
            OWNED_SEC_RAW_BATCH_MAX_BYTES,
        ),
        "max_sec_acquisition_seconds": budgets.get(
            "max_sec_acquisition_seconds"
        ),
    }
    if component_plan["max_sec_requests"] != len(normalized_documents):
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution request budget differs from its document plan"
        )
    return bundle, grant, component_plan


def build_stage_sec_execution_claim(
    authorization_bundle: Mapping[str, Any],
    *,
    independent_current_tip_anchor: Mapping[str, Any],
    execution_source_hashes: Mapping[str, Any],
    sec_user_agent_sha256: str,
) -> dict[str, Any]:
    """Claim the exact latest grant before the owned SEC reader may run."""

    bundle, grant, component_plan = _sec_component_plan_from_bundle(
        authorization_bundle
    )
    registry = _mapping(
        bundle["authenticated_store_snapshot"].get("latest_registry"),
        "SEC execution candidate registry",
    )
    raw_entries = registry.get("entries")
    if type(raw_entries) is not list:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution candidate registry has no exact entries"
        )
    candidate_matches: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        registry_entry = _mapping(
            raw_entry,
            f"SEC execution candidate registry entry {index}",
        )
        candidate = registry_entry.get("candidate_manifest")
        if (
            registry_entry.get("entry_sha256") == grant["registry_entry_sha256"]
            and type(candidate) is dict
            and candidate.get("candidate_sha256") == grant["candidate_sha256"]
        ):
            candidate_matches.append(candidate)
    if len(candidate_matches) != 1:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution grant lacks exactly one registry-pinned candidate"
        )
    candidate = candidate_matches[0]
    try:
        validated_candidate_hash = validate_candidate_manifest(
            candidate,
            expected_candidate_sha256=grant["candidate_sha256"],
        )
    except Exception as exc:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution candidate source pins are not canonical"
        ) from exc
    if validated_candidate_hash != grant["candidate_sha256"]:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution candidate identity changed"
        )
    candidate_sources = _mapping(
        _mapping(candidate.get("bindings"), "SEC execution candidate bindings").get(
            "source_hashes"
        ),
        "SEC execution candidate source hashes",
    )
    raw_execution_sources = _mapping(
        execution_source_hashes,
        "owned SEC execution source hashes",
    )
    expected_source_roles = {
        role for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    }
    if set(raw_execution_sources) != expected_source_roles:
        raise SecFilingGemmaStageAuthorizationError(
            "Owned SEC execution source closure is incomplete"
        )
    execution_sources = {
        role: _sha256(
            raw_execution_sources[role],
            f"owned SEC execution source hash {role}",
        )
        for role, _path in SEC_EXECUTION_RESOLVED_SOURCE_PATHS
    }
    if any(
        candidate_sources.get(role) != source_hash
        for role, source_hash in execution_sources.items()
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Owned SEC execution bytes differ from the registered candidate"
        )
    runner_source_hash = execution_sources["runner"]
    corpus_source_hash = execution_sources["sec_corpus_selector"]
    user_agent_hash = _tagged_sha256(
        sec_user_agent_sha256,
        "owned SEC execution User-Agent hash",
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        bundle["authenticated_store_snapshot"],
        independent_current_tip_anchor,
    )
    request_hash = grant["request_sha256"]
    if current_tip["authorization_bundles"].get(request_hash) != bundle:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution grant bundle is not exact at the current tip"
        )
    validate_consumed_stage_authorization_grant(
        grant,
        authenticated_store_snapshot=bundle["authenticated_store_snapshot"],
        external_store_state_pin=bundle["store_state_pin"],
        independent_current_tip_anchor=current_tip,
        expected_consumption_entry_sha256=grant["consumption_entry_sha256"],
        expected_request_sha256=request_hash,
        expected_candidate_sha256=grant["candidate_sha256"],
        expected_stage=grant["stage"],
        expected_prerequisite_stage_evidence_sha256=grant[
            "prerequisite_stage_evidence_sha256"
        ],
        expected_stage_access_manifest_sha256=grant[
            "stage_access_manifest_sha256"
        ],
        expected_output_namespace=grant["output_namespace"],
    )
    body = {
        "schema_version": STAGE_SEC_EXECUTION_CLAIM_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "claim_kind": "owned_sec_stage_document_batch",
        "request_sha256": request_hash,
        "consumption_entry_sha256": grant["consumption_entry_sha256"],
        "consumption_entry_sequence": grant["consumption_entry_sequence"],
        "attempt_id": grant["attempt_id"],
        "candidate_sha256": grant["candidate_sha256"],
        "registry_entry_sha256": grant["registry_entry_sha256"],
        "input_prerequisite_stage": grant["prerequisite_stage"],
        "authorized_stage": grant["stage"],
        "input_stage_evidence_sha256": grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": grant["stage_access_manifest_sha256"],
        "output_namespace": grant["output_namespace"],
        "authorization_bundle_sha256": bundle["bundle_sha256"],
        "authorization_grant_sha256": grant["authorization_grant_sha256"],
        "grant_store_state_sha256": grant["store_state_sha256"],
        "grant_consumption_ledger_sha256": grant[
            "consumption_ledger_sha256"
        ],
        "grant_consumption_ledger_tip_sha256": grant[
            "consumption_ledger_tip_sha256"
        ],
        "start_current_tip_anchor_sha256": current_tip["tip_anchor_sha256"],
        "runner_repository_path": STAGE_RUNNER_REPOSITORY_PATH,
        "runner_source_sha256": runner_source_hash,
        "sec_corpus_repository_path": SEC_CORPUS_REPOSITORY_PATH,
        "sec_corpus_source_sha256": corpus_source_hash,
        "execution_source_hashes": execution_sources,
        "execution_source_hashes_sha256": canonical_sha256(execution_sources),
        "execution_source_role_count": len(execution_sources),
        "sec_user_agent_sha256": user_agent_hash,
        "sec_component_id": SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
        "sec_component_plan_sha256": canonical_sha256(component_plan),
        "effect_may_be_repeated_after_indeterminate_crash": False,
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def build_stage_sec_reader_receipt(
    claim: Mapping[str, Any],
    *,
    byte_index: list[dict[str, Any]],
    complete_marker_sha256: str,
) -> dict[str, Any]:
    """Bind store-rehashed durable SEC bytes to one execution claim."""

    claim_value = _mapping(claim, "SEC reader receipt claim")
    _expect_keys(
        claim_value,
        _STAGE_SEC_EXECUTION_CLAIM_KEYS,
        "SEC reader receipt claim",
    )
    _self_hash(claim_value, "claim_sha256", "SEC reader receipt claim")
    index = _validated_sec_byte_index(byte_index)
    body = {
        "schema_version": STAGE_SEC_READER_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "store_rehashed_owned_sec_stage_document_batch",
        "request_sha256": claim_value["request_sha256"],
        "claim_sha256": claim_value["claim_sha256"],
        "authorized_stage": claim_value["authorized_stage"],
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "authorization_bundle_sha256": claim_value[
            "authorization_bundle_sha256"
        ],
        "authorization_grant_sha256": claim_value[
            "authorization_grant_sha256"
        ],
        "sec_component_id": claim_value["sec_component_id"],
        "sec_component_plan_sha256": claim_value[
            "sec_component_plan_sha256"
        ],
        "runner_source_sha256": claim_value["runner_source_sha256"],
        "sec_corpus_source_sha256": claim_value["sec_corpus_source_sha256"],
        "execution_source_hashes_sha256": claim_value[
            "execution_source_hashes_sha256"
        ],
        "execution_source_role_count": claim_value[
            "execution_source_role_count"
        ],
        "sec_user_agent_sha256": claim_value["sec_user_agent_sha256"],
        "byte_index": index,
        "byte_index_sha256": canonical_sha256(index),
        "byte_count_total": sum(item["byte_count"] for item in index),
        "complete_marker_sha256": _sha256(
            complete_marker_sha256, "SEC complete marker hash"
        ),
        "fresh_network_provenance_claimed": False,
        "reader_output_recomputed_by_store": True,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def build_stage_sec_execution_abort(
    claim: Mapping[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Terminally refuse any retry after a possibly executed SEC effect."""

    claim_value = _mapping(claim, "SEC execution abort claim")
    _expect_keys(
        claim_value,
        _STAGE_SEC_EXECUTION_CLAIM_KEYS,
        "SEC execution abort claim",
    )
    _self_hash(claim_value, "claim_sha256", "SEC execution abort claim")
    if reason not in {
        "claim_recovered_without_terminal_receipt",
        "external_effect_failed_or_completion_unknown",
        "durable_output_verification_failed",
    }:
        raise SecFilingGemmaStageAuthorizationError(
            "SEC execution abort reason is not canonical"
        )
    body = {
        "schema_version": STAGE_SEC_EXECUTION_ABORT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "abort_kind": "indeterminate_owned_sec_stage_execution",
        "request_sha256": claim_value["request_sha256"],
        "claim_sha256": claim_value["claim_sha256"],
        "authorized_stage": claim_value["authorized_stage"],
        "candidate_sha256": claim_value["candidate_sha256"],
        "output_namespace": claim_value["output_namespace"],
        "reason": reason,
        "external_effect_retry_permitted": False,
    }
    return {**body, "abort_sha256": canonical_sha256(body)}


def build_consumed_stage_output_receipt(
    authorization_bundle: Mapping[str, Any],
    *,
    stage_sec_execution_claim: Mapping[str, Any],
    stage_sec_reader_receipt: Mapping[str, Any],
    output_stage_evidence_schema_version: str,
    output_stage_evidence_sha256: str,
    output_stage_evidence_document_sha256: str,
    output_stage_evidence_complete_marker_sha256: str,
    output_stage_evidence_canonical_byte_count: int,
    output_stage_evidence_prerequisite_stage: str,
    output_parent_stage_evidence_sha256: str,
    output_candidate_sha256: str,
) -> dict[str, Any]:
    """Bind the first recorded next-stage evidence document to one consumed grant.

    This pure builder does not authorize I/O.  The effectful reveal store must
    first validate the bundle at its independently loaded current tip and then
    persist the resulting receipt with an append-only CAS transition.  The
    receipt binds store-recomputed durable evidence to the exact SEC claim and
    reader receipt, but deliberately does not claim fresh end-to-end evidence
    provenance until the remaining owned components exist.
    """

    raw_bundle = _mapping(authorization_bundle, "stage-output authorization bundle")
    raw_grant = _mapping(
        raw_bundle.get("authorization_grant"),
        "stage-output authorization grant",
    )
    request_hash = _sha256(
        raw_grant.get("request_sha256"),
        "stage-output authorization request hash",
    )
    bundle = _validated_authorization_bundles({request_hash: raw_bundle})[
        request_hash
    ]
    grant = bundle["authorization_grant"]
    claim = _validated_stage_sec_execution_claims(
        {request_hash: stage_sec_execution_claim},
        authorization_bundles={request_hash: bundle},
    )[request_hash]
    reader = _validated_stage_sec_reader_receipts(
        {request_hash: stage_sec_reader_receipt},
        claims={request_hash: claim},
    )[request_hash]
    schema_version = _safe_id(
        output_stage_evidence_schema_version,
        "output stage-evidence schema version",
    )
    if schema_version != STAGE_EVIDENCE_SCHEMA_VERSION:
        raise SecFilingGemmaStageAuthorizationError(
            "Output stage evidence is not the exact durable v3 schema"
        )
    evidence_hash = _sha256(
        output_stage_evidence_sha256,
        "output stage-evidence hash",
    )
    document_hash = _sha256(
        output_stage_evidence_document_sha256,
        "output stage-evidence document hash",
    )
    marker_hash = _sha256(
        output_stage_evidence_complete_marker_sha256,
        "output stage-evidence complete-marker hash",
    )
    sec_claim_hash = claim["claim_sha256"]
    sec_reader_hash = reader["receipt_sha256"]
    byte_count = _strict_int(
        output_stage_evidence_canonical_byte_count,
        "output stage-evidence canonical byte count",
        minimum=2,
    )
    output_prerequisite = _safe_id(
        output_stage_evidence_prerequisite_stage,
        "output stage-evidence prerequisite stage",
    )
    output_parent_hash = _sha256(
        output_parent_stage_evidence_sha256,
        "output parent stage-evidence hash",
    )
    output_candidate_hash = _sha256(
        output_candidate_sha256,
        "output candidate hash",
    )
    if (
        output_prerequisite != grant["stage"]
        or output_parent_hash != grant["prerequisite_stage_evidence_sha256"]
        or output_candidate_hash != grant["candidate_sha256"]
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage-output evidence crossed its consumed grant boundary"
        )
    body = {
        "schema_version": CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "receipt_kind": "first_output_for_exact_consumed_stage_grant",
        "consumption_entry_sha256": grant["consumption_entry_sha256"],
        "consumption_entry_sequence": grant["consumption_entry_sequence"],
        "request_sha256": grant["request_sha256"],
        "attempt_id": grant["attempt_id"],
        "candidate_sha256": grant["candidate_sha256"],
        "registry_entry_sha256": grant["registry_entry_sha256"],
        "input_prerequisite_stage": grant["prerequisite_stage"],
        "output_stage": grant["stage"],
        "input_stage_evidence_sha256": grant[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": grant["stage_access_manifest_sha256"],
        "output_namespace": grant["output_namespace"],
        "authorization_bundle_sha256": bundle["bundle_sha256"],
        "authorization_grant_sha256": grant["authorization_grant_sha256"],
        "sec_execution_claim_sha256": sec_claim_hash,
        "sec_reader_receipt_sha256": sec_reader_hash,
        "grant_store_state_sha256": grant["store_state_sha256"],
        "grant_consumption_ledger_sha256": grant[
            "consumption_ledger_sha256"
        ],
        "grant_consumption_ledger_tip_sha256": grant[
            "consumption_ledger_tip_sha256"
        ],
        "output_kind": "next_stage_evidence",
        "output_stage_evidence_schema_version": schema_version,
        "output_stage_evidence_component_id": STAGE_EVIDENCE_OUTPUT_COMPONENT_ID,
        "output_stage_evidence_relative_path": STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH,
        "output_stage_evidence_sha256": evidence_hash,
        "output_stage_evidence_document_sha256": document_hash,
        "output_stage_evidence_complete_marker_sha256": marker_hash,
        "output_stage_evidence_canonical_byte_count": byte_count,
        "output_stage_evidence_prerequisite_stage": output_prerequisite,
        "output_parent_stage_evidence_sha256": output_parent_hash,
        "output_candidate_sha256": output_candidate_hash,
        "output_stage_evidence_recomputed_by_store": True,
        "fresh_stage_evidence_provenance_claimed": False,
        "cross_stage_output_permitted": False,
        "grant_reuse_for_different_output_permitted": False,
    }
    return {**body, "output_receipt_sha256": canonical_sha256(body)}


def validate_consumed_stage_output_receipt(
    receipt: Mapping[str, Any],
    *,
    authenticated_store_snapshot: Mapping[str, Any],
    independent_current_tip_anchor: Mapping[str, Any],
    authorization_bundle: Mapping[str, Any],
    output_stage_evidence_schema_version: str,
    output_stage_evidence_sha256: str,
    output_stage_evidence_document_sha256: str,
    output_stage_evidence_complete_marker_sha256: str,
    output_stage_evidence_canonical_byte_count: int,
    output_stage_evidence_prerequisite_stage: str,
    output_parent_stage_evidence_sha256: str,
    output_candidate_sha256: str,
) -> str:
    """Require exact current-tip membership for one first-output receipt."""

    bundle = _mapping(authorization_bundle, "stage-output authorization bundle")
    grant = _mapping(
        bundle.get("authorization_grant"),
        "stage-output authorization grant",
    )
    current_tip = validate_reveal_store_current_tip_anchor(
        authenticated_store_snapshot,
        independent_current_tip_anchor,
    )
    request_hash = _sha256(
        grant.get("request_sha256"),
        "stage-output authorization request hash",
    )
    sec_claim = current_tip["stage_sec_execution_claims"].get(request_hash)
    sec_reader = current_tip["stage_sec_reader_receipts"].get(request_hash)
    if (
        type(sec_claim) is not dict
        or type(sec_reader) is not dict
        or sec_reader.get("claim_sha256") != sec_claim.get("claim_sha256")
    ):
        raise SecFilingGemmaStageAuthorizationError(
            "Stage-output receipt lacks its exact SEC claim and reader ancestry"
        )
    if current_tip["authorization_bundles"].get(request_hash) != bundle:
        raise SecFilingGemmaStageAuthorizationError(
            "Stage-output authorization bundle is not persisted at the current tip"
        )
    validate_consumed_stage_authorization_grant(
        grant,
        authenticated_store_snapshot=authenticated_store_snapshot,
        external_store_state_pin=bundle["store_state_pin"],
        independent_current_tip_anchor=current_tip,
        expected_consumption_entry_sha256=grant["consumption_entry_sha256"],
        expected_request_sha256=request_hash,
        expected_candidate_sha256=output_candidate_sha256,
        expected_stage=output_stage_evidence_prerequisite_stage,
        expected_prerequisite_stage_evidence_sha256=(
            output_parent_stage_evidence_sha256
        ),
        expected_stage_access_manifest_sha256=grant[
            "stage_access_manifest_sha256"
        ],
        expected_output_namespace=grant["output_namespace"],
    )
    expected = build_consumed_stage_output_receipt(
        bundle,
        stage_sec_execution_claim=sec_claim,
        stage_sec_reader_receipt=sec_reader,
        output_stage_evidence_schema_version=output_stage_evidence_schema_version,
        output_stage_evidence_sha256=output_stage_evidence_sha256,
        output_stage_evidence_document_sha256=(
            output_stage_evidence_document_sha256
        ),
        output_stage_evidence_complete_marker_sha256=(
            output_stage_evidence_complete_marker_sha256
        ),
        output_stage_evidence_canonical_byte_count=(
            output_stage_evidence_canonical_byte_count
        ),
        output_stage_evidence_prerequisite_stage=(
            output_stage_evidence_prerequisite_stage
        ),
        output_parent_stage_evidence_sha256=output_parent_stage_evidence_sha256,
        output_candidate_sha256=output_candidate_sha256,
    )
    observed = _mapping(receipt, "consumed-stage output receipt")
    _expect_keys(observed, _STAGE_OUTPUT_RECEIPT_KEYS, "consumed-stage output receipt")
    _self_hash(
        observed,
        "output_receipt_sha256",
        "consumed-stage output receipt",
    )
    if observed != expected:
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed-stage output receipt differs from the exact grant or evidence"
        )
    if current_tip["consumed_stage_output_receipts"].get(request_hash) != observed:
        raise SecFilingGemmaStageAuthorizationError(
            "Consumed-stage output receipt is not persisted at the current tip"
        )
    return observed["output_receipt_sha256"]


__all__ = [
    "CONSUMED_STAGE_AUTHORIZATION_BUNDLE_SCHEMA_VERSION",
    "CONSUMED_STAGE_AUTHORIZATION_GRANT_SCHEMA_VERSION",
    "CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION",
    "CONSUMED_STAGE_STORE_PIN_SCHEMA_VERSION",
    "REVEAL_STORE_CURRENT_TIP_ANCHOR_SCHEMA_VERSION",
    "OWNED_SEC_RAW_BATCH_MAX_BYTES",
    "SEC_EXECUTION_RESOLVED_SOURCE_PATHS",
    "SEC_CORPUS_REPOSITORY_PATH",
    "SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID",
    "STAGE_EVIDENCE_OUTPUT_COMPONENT_ID",
    "STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH",
    "STAGE_RUNNER_REPOSITORY_PATH",
    "STAGE_SEC_EXECUTION_ABORT_SCHEMA_VERSION",
    "STAGE_SEC_EXECUTION_CLAIM_SCHEMA_VERSION",
    "STAGE_SEC_READER_RECEIPT_SCHEMA_VERSION",
    "TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION",
    "TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION",
    "SecFilingGemmaStageAuthorizationError",
    "authenticate_reveal_store_trusted_stage_content_pin",
    "build_consumed_stage_authorization_grant",
    "build_consumed_stage_output_receipt",
    "build_stage_sec_execution_abort",
    "build_stage_sec_execution_claim",
    "build_stage_sec_reader_receipt",
    "build_reveal_store_current_tip_anchor",
    "derive_consumed_stage_store_state_pin",
    "derive_reveal_store_trusted_stage_content_pin",
    "validate_consumed_stage_authorization_grant",
    "validate_consumed_stage_output_receipt",
    "validate_consumed_stage_store_state_pin",
    "validate_reveal_store_current_tip_anchor",
    "validate_reveal_store_current_tip_anchor_structure",
    "validate_reveal_store_current_tip_anchor_transition",
    "validate_trusted_stage_content_authentication_receipt",
]
