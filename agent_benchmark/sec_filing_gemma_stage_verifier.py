"""Deliberately non-authorizing SEC/Gemma stage-evidence audit.

This module is the narrow bridge between the evidence components and the
effectful reveal store.  It replays the parts that can currently be proved
from detached bytes and records the remaining trust gaps explicitly.  It does
*not* return ``SemanticPrerequisiteValidation`` and therefore cannot unlock a
holdout stage.  The fail-closed boundary is intentional: several required
production attestations do not yet have authoritative component APIs.

The runtime source-identity check reads bounded local source files that were
already imported by the process. No function performs network, model, market,
SEC, or outcome I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
import copy
import hashlib
import hmac
import json
import math
import re
from typing import Any, Final

from agent_benchmark.sec_filing_gemma_artifact_sealer import (
    validate_prediction_artifact_seal_receipt,
)
from agent_benchmark.sec_filing_gemma_contract import (
    CANDIDATE_IDS,
    CONTRACT_VERSION,
    MAX_FIT_SECONDS,
    MAX_MODEL_SECONDS,
    MAX_RUNTIME_SECONDS,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    REQUIRED_SOURCE_HASHES,
    REQUIRED_STAGE_VERIFIER_CHECKS,
    STAGE_MODEL_CALL_CAPS,
    STAGE_ORDER,
    SecFilingGemmaContractError,
    canonical_sha256,
    validate_calendar_source_evidence_manifest,
    validate_candidate_manifest,
    validate_contract_manifest,
    validate_corpus_universe_manifest,
    validate_stage_content_manifest,
)
from agent_benchmark.sec_filing_gemma_corpus import (
    DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION,
    DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION,
    validate_detached_catalog_replay,
    validate_detached_stage_content_replay,
)
from agent_benchmark.sec_filing_gemma_learner import (
    SecFilingGemmaTwoHeadLearner,
)
from agent_benchmark.sec_filing_gemma_market_source_bytes import (
    validate_market_source_bytes_against_stage,
)
from agent_benchmark.sec_filing_gemma_no_leverage import (
    validate_sec_gemma_no_leverage_proof,
)
from agent_benchmark.sec_filing_gemma_ollama import (
    validate_ollama_model_attempt_receipt,
    validate_runtime_identity_guard,
)
from agent_benchmark.sec_filing_gemma_prediction_evidence import (
    AVAILABLE_PREDICTION_STATUS,
    validate_prediction_prefix,
)
from agent_benchmark.sec_filing_gemma_reveal_registry import (
    build_single_candidate_reveal_request,
    candidate_design_sha256,
    validate_reveal_registry,
)
from agent_benchmark.sec_filing_gemma_scoring import (
    validate_development_ranking_receipt,
    validate_score_receipt,
    validate_stage_gate_receipt,
)
import agent_benchmark.sec_filing_gemma_source_identity as source_identity_module
from agent_benchmark.sec_filing_gemma_source_identity import (
    CANONICAL_SOURCE_ROLE_PATHS,
    UNRESOLVED_SOURCE_ROLES,
    audit_candidate_source_identity,
)
from agent_benchmark.sec_filing_gemma_stage_access import (
    validate_stage_access_manifest,
)
from agent_benchmark.sec_session_calendar import EXPECTED_SESSIONS


STAGE_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-evidence-audit-v3"
)
STAGE_EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "prerequisite_stage",
        "parent_stage_evidence_sha256",
        "parent_stage_lineage",
        "contract_manifest",
        "candidate_manifest",
        "source_bytes_base64_by_role",
        "calendar_evidence_manifest",
        "calendar_source_bytes_base64_by_name",
        "corpus_universe_manifest",
        "catalog_replay",
        "content_replays_by_stage",
        "prerequisite_content_manifest",
        "model_batches_by_stage",
        "market_replays_by_stage",
        "prediction_replay",
        "learner_replays",
        "score_replay",
        "stage_runtime_receipt",
        "reveal_registry",
        "registry_external_pin",
        "stage_evidence_sha256",
    }
)
STAGE_AUDIT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-evidence-audit-receipt-v6"
)
STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-stage-runtime-receipt-v1"
)
AUTHORITATIVE_VALIDATOR_ID: Final[str] = (
    "aapl-sec-gemma-authoritative-stage-verifier-v1"
)
OWNED_HARDENED_TRANSPORT_MODE: Final[str] = (
    "owned_hardened_loopback_session_requires_stage_attestation"
)
TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-trusted-stage-content-pin-v2"
)
TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-trusted-stage-content-authentication-v1"
)
AUTHENTICATED_STORE_VERIFIER_CONTEXT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-authenticated-store-verifier-context-v1"
)
PARENT_CONSUMPTION_BINDING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-parent-consumption-binding-v2"
)
CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-consumed-stage-output-receipt-v2"
)
STAGE_EVIDENCE_OUTPUT_COMPONENT_ID: Final[str] = "owned_stage_evidence_document"
STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH: Final[str] = "stage_evidence.json"
_EXPECTED_UNRESOLVED_SOURCE_ROLES: Final[tuple[str, ...]] = (
    "ledger",
)

# These are parser/allocation ceilings, not acquisition budgets.  The SEC
# contract has a 1.5-GiB aggregate transport ceiling, but no single Apple
# filing, source module, market CSV, or canonical receipt is permitted to make
# the verifier allocate anything close to that amount in one operation.
MAX_BASE64_ITEM_BYTES: Final[int] = 64 * 1024 * 1024
MAX_SOURCE_MODULE_BYTES: Final[int] = 16 * 1024 * 1024
MAX_SOURCE_MODULE_TOTAL_BYTES: Final[int] = 128 * 1024 * 1024
MAX_SMALL_RECEIPT_BYTES: Final[int] = 1 * 1024 * 1024
MAX_PREDICTION_ARTIFACT_BYTES: Final[int] = 64 * 1024 * 1024
MAX_PREFLIGHT_NESTING_DEPTH: Final[int] = 32
MAX_PREFLIGHT_CONTAINER_ITEMS: Final[int] = 20_000
MAX_PREFLIGHT_TOTAL_ELEMENTS: Final[int] = 1_000_000
MAX_PREFLIGHT_BASE64_ITEMS: Final[int] = 4_096
MAX_PREFLIGHT_BASE64_DECODED_BYTES: Final[int] = 192 * 1024 * 1024
MAX_PREFLIGHT_STRING_CHARACTERS: Final[int] = 4 * 1024 * 1024
MAX_PREFLIGHT_TEXT_CHARACTERS_TOTAL: Final[int] = 64 * 1024 * 1024
MAX_PREFLIGHT_KEY_CHARACTERS: Final[int] = 512
MAX_PREFLIGHT_INTEGER_BITS: Final[int] = 256
MAX_PREFLIGHT_ESTIMATED_JSON_BYTES: Final[int] = 256 * 1024 * 1024
MAX_STAGE_CONTENT_BYTES: Final[int] = 128 * 1024 * 1024
MAX_CATALOG_SOURCE_ITEMS: Final[int] = 64
MAX_CATALOG_SOURCE_BYTES: Final[int] = 128 * 1024 * 1024

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BASE64_ALPHABET: Final[frozenset[str]] = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
)
_PREREQUISITE_TRANSITIONS: Final[dict[str, str]] = {
    "development": "intermediate",
    "intermediate": "final",
}
_EXPECTED_REVEAL_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "request_sha256",
        "stage",
        "stage_access_manifest_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "trusted_stage_content_pin_sha256",
        "trusted_stage_content_authentication_receipt_sha256",
        "parent_consumption_binding_sha256",
        "authenticated_store_context_sha256",
    }
)
_PARENT_BINDING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "contract_version",
        "binding_kind",
        "child_request",
        "parent_consumption",
        "authenticated_preconsumption_tip",
        "parent_consumption_binding_sha256",
    }
)
_PARENT_BINDING_CHILD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "request_sha256",
        "stage",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
    }
)
_PARENT_BINDING_PARENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "entry_sha256",
        "sequence",
        "request_sha256",
        "stage",
        "prerequisite_stage",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "expected_context_sha256",
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "prerequisite_validation_result_sha256",
        "semantic_receipt_sha256",
        "audit_receipt",
        "audit_receipt_sha256",
        "trusted_stage_content_pin",
        "trusted_stage_content_pin_sha256",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "store_pin_sha256",
        "consumed_stage_output_receipt",
        "consumed_stage_output_receipt_sha256",
    }
)
_PARENT_BINDING_TIP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "store_state_sha256",
        "state_snapshot_bytes_sha256",
        "state_snapshot_byte_count",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
        "consumed_request_count",
        "current_tip_anchor_sha256",
        "current_tip_revision",
        "consumed_stage_output_receipts",
        "consumed_stage_output_receipts_sha256",
        "stage_sec_execution_claims",
        "stage_sec_execution_claims_sha256",
        "stage_sec_reader_receipts",
        "stage_sec_reader_receipts_sha256",
    }
)
_CONSUMED_STAGE_OUTPUT_RECEIPT_KEYS: Final[frozenset[str]] = frozenset(
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
_BLOCKING_GAPS: Final[dict[str, str]] = {
    "artifact_seal_cas": (
        "only the latest seal transition is replayed; the envelope does not yet "
        "chain every artifact and receipt forward from deterministic genesis"
    ),
    "artifact_replay": (
        "detached SEC catalogue and document bytes replay within the current "
        "envelope; the request-free development path now also owns preprocessing, "
        "extraction, causal market prefixes, feature proofs, and the mature-label "
        "projection plus fixed training membership, but the detached envelope does "
        "not yet chain those projections, the prelabel ledger, or the full seal "
        "chain; parent lineage "
        "is bound to the prior consumed entry and grant, but that does not replace "
        "the missing end-to-end artifact derivation proofs"
    ),
    "calendar_source_semantics": (
        "official calendar bytes are hash-bound but no pure parser proves the "
        "frozen session semantics from those bytes"
    ),
    "chronology": (
        "development event bindings and feature rows are now derived chronologically "
        "from replayed filing, model, and causal market evidence, and development "
        "labels are derived only when mature by the development cutoff; all six "
        "fixed training views admit only labels matured by their respective frozen "
        "cutoffs and are assembled through an owned reader without target-based "
        "membership filtering, but learner fit and prediction are not yet owned"
    ),
    "market_source_byte_reconciliation": (
        "the fixed provider responses now replay through a store-attested market "
        "reader into canonical snapshots, and the request-free development feature "
        "assembler loads only reader-bound causal prefixes; this detached stage "
        "verifier still accepts caller-supplied market snapshots, and the owned "
        "development label projection now extends into owned fixed training "
        "membership, but not into learner or later-stage assembly"
    ),
    "model_attempt_replay": (
        "exact attempt bytes replay, but the independently expected payload is "
        "not yet derived from authoritative preprocessing of the replayed filing bytes"
    ),
    "prediction_replay": (
        "policy-prefix transitions and learner arithmetic replay; the request-free "
        "development path now owns event bindings and training matrices, but this "
        "prediction replay does not yet consume them through that owned boundary"
    ),
    "prerequisite_evidence_identity": (
        "the directly invoked verifier replays raw catalogue and content bytes but "
        "still omits the owned development-label and training-membership projections, "
        "prelabel-ledger, and full seal-chain proofs; the request-free feature and "
        "label projections bind their upstream preprocessing, model extraction, "
        "causal market evidence, and mature outcomes, while the downstream membership "
        "projection binds both sources; learner and later downstream readers are not "
        "yet grant-bound"
    ),
    "runtime_budget": (
        "stage-specific measurements are internally reconciled but the owned "
        "transport and monotonic runtime are not independently attested"
    ),
    "source_identity": (
        "resolved source roles replay against frozen paths, but ledger remains "
        "unresolved; "
        "current files at loaded module paths do not yet attest the source bytes "
        "that created the executing code objects"
    ),
    "stage_access_identity": (
        "the owned runner now claims the exact current grant before the SEC batch and "
        "the store rehashes its durable actual bytes; model, carry-in, and development "
        "market reads plus request-free development feature and mature-label assembly "
        "plus fixed training-membership assembly are owned and cross-bound, but "
        "learner, prediction, ledger, and stage-evidence readers are not yet forced "
        "through the same "
        "authenticated lineage"
    ),
    "zero_cost": (
        "loopback and zero-cost receipt fields replay, but independent network "
        "transport attestation is still absent"
    ),
}
_INTERMEDIATE_STAGE_IDENTITY_GAP: Final[str] = (
    "the parent evidence and audit receipt are replayed exactly, including earlier "
    "content, but the development winner and parent learner output state are not "
    "yet bound to the intermediate learner input state"
)
_LEARNER_FIT_METADATA_KEYS: Final[frozenset[str]] = frozenset(
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
_LEARNER_FOLD_CONTEXT_KEYS: Final[frozenset[str]] = frozenset(
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


class SecFilingGemmaStageVerifierError(ValueError):
    """Detached stage evidence is invalid or cannot authorize a reveal."""


class SecFilingGemmaStageVerifierBlocked(SecFilingGemmaStageVerifierError):
    """All replayable evidence passed, but authoritative promotion is blocked."""


def _sha256(value: Any, location: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SecFilingGemmaStageVerifierError(
            f"{location} must be a lowercase SHA-256 digest"
        )
    return value


def _expect_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    location: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise SecFilingGemmaStageVerifierError(
            f"{location} keys changed; missing={missing}, extra={extra}"
        )


def _is_plain_mapping(value: Any, *, top_level: bool) -> bool:
    del top_level
    return type(value) is dict


def _base64_decoded_length_preflight(
    value: Any,
    location: str,
    *,
    maximum: int = MAX_BASE64_ITEM_BYTES,
) -> int:
    """Validate Base64 shape/size without encoding, decoding, or copying it."""

    if type(value) is not str or not value:
        raise SecFilingGemmaStageVerifierError(
            f"{location} must be nonempty canonical Base64"
        )
    encoded_length = len(value)
    maximum_encoded_length = 4 * ((maximum + 2) // 3)
    if encoded_length > maximum_encoded_length or encoded_length % 4:
        raise SecFilingGemmaStageVerifierError(
            f"{location} encoded length exceeds its allocation ceiling or is invalid"
        )
    # Inspect padding by index.  ``rstrip`` and a core slice would each copy a
    # caller-controlled multi-megabyte string during the no-copy preflight.
    padding = 0
    if value[-1] == "=":
        padding = 1
        if encoded_length >= 2 and value[-2] == "=":
            padding = 2
    core_length = encoded_length - padding
    for index in range(core_length):
        if value[index] not in _BASE64_ALPHABET:
            raise SecFilingGemmaStageVerifierError(
                f"{location} is not strict Base64"
            )
    for index in range(core_length, encoded_length):
        if value[index] != "=":  # pragma: no cover - defensive by construction
            raise SecFilingGemmaStageVerifierError(
                f"{location} has invalid Base64 padding"
            )
    decoded_length = (encoded_length // 4) * 3 - padding
    if decoded_length < 1 or decoded_length > maximum:
        raise SecFilingGemmaStageVerifierError(
            f"{location} decoded length exceeds its allocation ceiling"
        )
    return decoded_length


def _json_string_encoded_upper_bound(value: str) -> int:
    """Bound ``json.dumps(..., ensure_ascii=True)`` without serializing."""

    encoded = 2  # surrounding quotes
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"}:
            encoded += 2
        elif codepoint < 0x20:
            encoded += 6
        elif codepoint < 0x80:
            encoded += 1
        elif codepoint <= 0xFFFF:
            encoded += 6
        else:
            encoded += 12
    return encoded


def _walk_plain_json(
    value: Any,
    location: str,
    *,
    detach: bool,
) -> tuple[Any, dict[str, int]]:
    """Bound and optionally detach one untrusted finite-JSON envelope.

    The copy mode applies every bound during the same traversal that builds
    the snapshot.  A mutation after a prior no-copy pass therefore cannot add
    an unchecked deep or oversized value.  Only exact built-in containers are
    accepted, so custom iteration hooks never run.
    """

    totals = {
        "elements": 0,
        "base64_items": 0,
        "base64_decoded_bytes": 0,
        "text_characters": 0,
        "estimated_json_bytes": 0,
    }

    def add_json_bytes(count: int) -> None:
        totals["estimated_json_bytes"] += count
        if (
            totals["estimated_json_bytes"]
            > MAX_PREFLIGHT_ESTIMATED_JSON_BYTES
        ):
            raise SecFilingGemmaStageVerifierError(
                f"{location} exceeds the estimated canonical-JSON byte ceiling"
            )

    def add_text_characters(count: int) -> None:
        totals["text_characters"] += count
        if totals["text_characters"] > MAX_PREFLIGHT_TEXT_CHARACTERS_TOTAL:
            raise SecFilingGemmaStageVerifierError(
                f"{location} exceeds the aggregate text-character ceiling"
            )

    def walk(
        node: Any,
        node_location: str,
        *,
        depth: int,
        base64_values: bool,
        top_level: bool,
    ) -> Any:
        if depth > MAX_PREFLIGHT_NESTING_DEPTH:
            raise SecFilingGemmaStageVerifierError(
                f"{location} exceeds the maximum nesting depth"
            )
        totals["elements"] += 1
        if totals["elements"] > MAX_PREFLIGHT_TOTAL_ELEMENTS:
            raise SecFilingGemmaStageVerifierError(
                f"{location} exceeds the maximum element count"
            )
        if node is None:
            add_json_bytes(4)
            return None
        if type(node) is bool:
            add_json_bytes(4 if node else 5)
            return node
        if type(node) is int:
            if node.bit_length() > MAX_PREFLIGHT_INTEGER_BITS:
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} exceeds the maximum integer size"
                )
            add_json_bytes(len(str(node)))
            return node
        if type(node) is float:
            if not math.isfinite(node):
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} is non-finite"
                )
            add_json_bytes(32)
            return node
        if type(node) is str:
            if not base64_values and len(node) > MAX_PREFLIGHT_STRING_CHARACTERS:
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} exceeds the maximum string length"
                )
            if base64_values:
                decoded_length = _base64_decoded_length_preflight(
                    node,
                    node_location,
                    maximum=MAX_PREDICTION_ARTIFACT_BYTES,
                )
                totals["base64_items"] += 1
                totals["base64_decoded_bytes"] += decoded_length
                if totals["base64_items"] > MAX_PREFLIGHT_BASE64_ITEMS:
                    raise SecFilingGemmaStageVerifierError(
                        f"{location} exceeds the maximum Base64 item count"
                    )
                if (
                    totals["base64_decoded_bytes"]
                    > MAX_PREFLIGHT_BASE64_DECODED_BYTES
                ):
                    raise SecFilingGemmaStageVerifierError(
                        f"{location} exceeds the aggregate Base64 byte ceiling"
                    )
                add_json_bytes(len(node) + 2)
            else:
                add_text_characters(len(node))
                add_json_bytes(_json_string_encoded_upper_bound(node))
            return node
        if type(node) is list:
            initial_length = len(node)
            if initial_length > MAX_PREFLIGHT_CONTAINER_ITEMS:
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} exceeds the maximum item count"
                )
            add_json_bytes(2 + max(0, initial_length - 1))
            result: list[Any] | None = [] if detach else None
            try:
                for index, child in enumerate(node):
                    copied = walk(
                        child,
                        f"{node_location}[{index}]",
                        depth=depth + 1,
                        base64_values=False,
                        top_level=False,
                    )
                    if result is not None:
                        result.append(copied)
            except (IndexError, RuntimeError) as exc:
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} changed during bounded traversal"
                ) from exc
            if len(node) != initial_length or (
                result is not None and len(result) != initial_length
            ):
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} changed during bounded traversal"
                )
            return result
        if _is_plain_mapping(node, top_level=top_level):
            initial_length = len(node)
            if initial_length > MAX_PREFLIGHT_CONTAINER_ITEMS:
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} exceeds the maximum item count"
                )
            add_json_bytes(2 + max(0, initial_length - 1) + initial_length)
            result_dict: dict[str, Any] | None = {} if detach else None
            observed = 0
            try:
                for key, child in node.items():
                    observed += 1
                    if type(key) is not str:
                        raise SecFilingGemmaStageVerifierError(
                            f"{node_location} contains a non-string key"
                        )
                    if len(key) > MAX_PREFLIGHT_KEY_CHARACTERS:
                        raise SecFilingGemmaStageVerifierError(
                            f"{node_location} contains an oversized key"
                        )
                    add_text_characters(len(key))
                    add_json_bytes(_json_string_encoded_upper_bound(key))
                    child_is_base64 = key.endswith("_base64") or (
                        base64_values and type(child) is str
                    )
                    child_base64_mapping = key.endswith(
                        ("_base64_by_role", "_base64_by_name", "_base64_by_symbol")
                    )
                    copied = walk(
                        child,
                        f"{node_location}.{key}",
                        depth=depth + 1,
                        base64_values=(child_is_base64 or child_base64_mapping),
                        top_level=False,
                    )
                    if result_dict is not None:
                        result_dict[key] = copied
            except RuntimeError as exc:
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} changed during bounded traversal"
                ) from exc
            if len(node) != initial_length or observed != initial_length:
                raise SecFilingGemmaStageVerifierError(
                    f"{node_location} changed during bounded traversal"
                )
            return result_dict
        raise SecFilingGemmaStageVerifierError(
            f"{node_location} must contain detached plain JSON values"
        )

    snapshot = walk(
        value,
        location,
        depth=0,
        base64_values=False,
        top_level=True,
    )
    return snapshot, totals


def _preflight_plain_json(value: Any, location: str) -> dict[str, int]:
    """Bound an untrusted envelope without copying or serializing it."""

    _snapshot, totals = _walk_plain_json(value, location, detach=False)
    return totals


def _bounded_plain_json_copy(value: Any, location: str) -> tuple[Any, dict[str, int]]:
    """Detach an untrusted envelope while independently reapplying all bounds."""

    return _walk_plain_json(value, location, detach=True)


def preflight_untrusted_stage_json(value: Any, location: str) -> dict[str, int]:
    """Public no-copy bound for effectful callers before defensive copying.

    The reveal store invokes this immediately after authenticating its own
    state and before traversing a caller-owned request, candidate, access
    manifest, or evidence envelope.  This preserves the verifier's allocation
    limits across the effectful boundary instead of applying them only after a
    potentially enormous defensive copy has already been made.
    """

    if type(location) is not str or not location or len(location) > 128:
        raise SecFilingGemmaStageVerifierError(
            "Preflight location must be a short exact built-in string"
        )
    return _preflight_plain_json(value, location)


def detach_untrusted_stage_json(value: Any, location: str) -> Any:
    """Return a bounded detached snapshot after a fresh copy-time traversal."""

    if type(location) is not str or not location or len(location) > 128:
        raise SecFilingGemmaStageVerifierError(
            "Detach location must be a short exact built-in string"
        )
    snapshot, _totals = _bounded_plain_json_copy(value, location)
    return snapshot


def _preflight_base64_mapping_values(
    value: Any,
    location: str,
    *,
    maximum_items: int,
    maximum_item_bytes: int,
    maximum_total_bytes: int,
) -> int:
    """Apply context-specific Base64 bounds without constructing a snapshot."""

    _preflight_plain_json(value, location)
    if not _is_plain_mapping(value, top_level=True):
        raise SecFilingGemmaStageVerifierError(f"{location} must be a mapping")
    if not 1 <= len(value) <= maximum_items:
        raise SecFilingGemmaStageVerifierError(
            f"{location} exceeds its fixed item-count ceiling"
        )
    total = 0
    for key, encoded in value.items():
        if type(key) is not str:
            raise SecFilingGemmaStageVerifierError(
                f"{location} contains a non-string key"
            )
        total += _base64_decoded_length_preflight(
            encoded,
            f"{location}.{key}",
            maximum=maximum_item_bytes,
        )
        if total > maximum_total_bytes:
            raise SecFilingGemmaStageVerifierError(
                f"{location} exceeds its aggregate byte ceiling"
            )
    return total


def _preflight_catalog_payloads(value: Any) -> None:
    _preflight_plain_json(value, "detached catalogue evidence")
    if not _is_plain_mapping(value, top_level=True):
        raise SecFilingGemmaStageVerifierError(
            "detached catalogue evidence must be a mapping"
        )
    sources = value.get("source_payloads")
    if type(sources) is not list or not 1 <= len(sources) <= MAX_CATALOG_SOURCE_ITEMS:
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue source payload count exceeds its fixed ceiling"
        )
    total = 0
    for index, item in enumerate(sources):
        if (
            type(item) is not dict
            or len(item) != 2
            or "name" not in item
            or "payload_base64" not in item
        ):
            raise SecFilingGemmaStageVerifierError(
                f"catalogue sources[{index}] shape changed"
            )
        total += _base64_decoded_length_preflight(
            item["payload_base64"],
            f"catalogue sources[{index}].payload_base64",
            maximum=MAX_BASE64_ITEM_BYTES,
        )
        if total > MAX_CATALOG_SOURCE_BYTES:
            raise SecFilingGemmaStageVerifierError(
                "Detached catalogue source payloads exceed their aggregate byte ceiling"
            )


def _preflight_stage_documents(
    value: Any,
    *,
    expected_stage: str,
    maximum_items: int,
    maximum_total_bytes: int,
) -> None:
    location = f"detached {expected_stage} content evidence"
    _preflight_plain_json(value, location)
    if not _is_plain_mapping(value, top_level=True):
        raise SecFilingGemmaStageVerifierError(f"{location} must be a mapping")
    documents = value.get("document_payloads")
    if type(documents) is not list or not 1 <= len(documents) <= maximum_items:
        raise SecFilingGemmaStageVerifierError(
            "Detached stage document count exceeds its fixed ceiling"
        )
    total = 0
    for index, item in enumerate(documents):
        if (
            type(item) is not dict
            or len(item) != 2
            or "accession_number" not in item
            or "payload_base64" not in item
        ):
            raise SecFilingGemmaStageVerifierError(
                f"{location}.document_payloads[{index}] shape changed"
            )
        total += _base64_decoded_length_preflight(
            item["payload_base64"],
            f"{location}.document_payloads[{index}].payload_base64",
            maximum=MAX_BASE64_ITEM_BYTES,
        )
        if total > maximum_total_bytes:
            raise SecFilingGemmaStageVerifierError(
                "Detached stage documents exceed the externally bounded aggregate bytes"
            )


def _mapping_snapshot(value: Any, location: str) -> dict[str, Any]:
    if not _is_plain_mapping(value, top_level=True):
        raise SecFilingGemmaStageVerifierError(f"{location} must be a mapping")
    try:
        detached, _totals = _bounded_plain_json_copy(value, location)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            f"{location} could not be detached safely"
        ) from exc
    if type(detached) is not dict:
        raise SecFilingGemmaStageVerifierError(f"{location} must be an object")
    return detached


def _decode_base64(
    value: Any,
    location: str,
    *,
    maximum: int = MAX_BASE64_ITEM_BYTES,
) -> bytes:
    expected_length = _base64_decoded_length_preflight(
        value, location, maximum=maximum
    )
    try:
        payload = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            f"{location} is not strict Base64"
        ) from exc
    if (
        len(payload) != expected_length
        or base64.b64encode(payload).decode("ascii") != value
    ):
        raise SecFilingGemmaStageVerifierError(
            f"{location} is empty, oversized, or noncanonical"
        )
    return payload


def _strict_json_bytes(payload: bytes, location: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        parsed = json.loads(text)
        _preflight_plain_json(parsed, location)
        canonical = json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise SecFilingGemmaStageVerifierError(
            f"{location} is not canonical JSON bytes"
        ) from exc
    if type(parsed) is not dict or canonical != payload:
        raise SecFilingGemmaStageVerifierError(
            f"{location} is not one canonical JSON object"
        )
    detached, _totals = _bounded_plain_json_copy(parsed, location)
    if type(detached) is not dict:  # pragma: no cover - checked above
        raise SecFilingGemmaStageVerifierError(
            f"{location} is not one canonical JSON object"
        )
    return detached


def validate_candidate_source_bytes(
    *,
    contract_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    source_bytes_base64_by_role: Mapping[str, str],
) -> dict[str, Any]:
    """Replay the frozen contract/candidate and every candidate source byte pin."""

    _preflight_base64_mapping_values(
        source_bytes_base64_by_role,
        "source bytes",
        maximum_items=len(REQUIRED_SOURCE_HASHES),
        maximum_item_bytes=MAX_SOURCE_MODULE_BYTES,
        maximum_total_bytes=MAX_SOURCE_MODULE_TOTAL_BYTES,
    )
    contract = _mapping_snapshot(contract_manifest, "contract manifest")
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    sources = _mapping_snapshot(source_bytes_base64_by_role, "source bytes")
    try:
        contract_hash = validate_contract_manifest(contract)
        candidate_hash = validate_candidate_manifest(
            candidate, expected_candidate_sha256=expected_candidate_sha256
        )
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Contract or candidate identity replay failed"
        ) from exc
    _expect_keys(sources, set(REQUIRED_SOURCE_HASHES), "source bytes")
    observed: dict[str, str] = {}
    for role in REQUIRED_SOURCE_HASHES:
        payload = _decode_base64(
            sources[role],
            f"source bytes {role}",
            maximum=MAX_SOURCE_MODULE_BYTES,
        )
        digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(
            digest, candidate["bindings"]["source_hashes"][role]
        ):
            raise SecFilingGemmaStageVerifierError(
                f"Source bytes for {role} differ from the candidate pin"
            )
        observed[role] = digest
    return {
        "contract_sha256": contract_hash,
        "candidate_sha256": candidate_hash,
        "source_hashes_sha256": canonical_sha256(observed),
        "stage_verifier_source_sha256": observed["stage_verifier"],
    }


def validate_candidate_source_role_audit(
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
    source_bytes_base64_by_role: Mapping[str, str],
) -> dict[str, Any]:
    """Audit resolved source roles against frozen paths without inventing owners."""

    _preflight_base64_mapping_values(
        source_bytes_base64_by_role,
        "source bytes",
        maximum_items=len(REQUIRED_SOURCE_HASHES),
        maximum_item_bytes=MAX_SOURCE_MODULE_BYTES,
        maximum_total_bytes=MAX_SOURCE_MODULE_TOTAL_BYTES,
    )
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    sources = _mapping_snapshot(source_bytes_base64_by_role, "source bytes")
    _expect_keys(sources, set(REQUIRED_SOURCE_HASHES), "source bytes")
    paths = dict(CANONICAL_SOURCE_ROLE_PATHS)
    payloads: dict[str, bytes | None] = {}
    for role in REQUIRED_SOURCE_HASHES:
        payload = _decode_base64(
            sources[role],
            f"source bytes {role}",
            maximum=MAX_SOURCE_MODULE_BYTES,
        )
        payloads[role] = None if paths[role] is None else payload
    try:
        raw_receipt = audit_candidate_source_identity(
            candidate_manifest=candidate,
            expected_candidate_sha256=expected_candidate_sha256,
            source_paths_by_role=paths,
            source_bytes_by_role=payloads,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Frozen source-role identity audit failed"
        ) from exc
    receipt = _mapping_snapshot(raw_receipt, "source identity receipt")
    if (
        receipt.get("candidate_sha256") != expected_candidate_sha256
        or receipt.get("unresolved_roles") != list(UNRESOLVED_SOURCE_ROLES)
        or receipt.get("unresolved_role_count") != len(UNRESOLVED_SOURCE_ROLES)
        or UNRESOLVED_SOURCE_ROLES != _EXPECTED_UNRESOLVED_SOURCE_ROLES
        or receipt.get("complete") is not False
        or receipt.get("authorizes") is not False
    ):
        raise SecFilingGemmaStageVerifierError(
            "Source-role receipt changed its incomplete non-authorizing semantics"
        )
    return receipt


def validate_candidate_runtime_source_audit(
    *,
    candidate_manifest: Mapping[str, Any],
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    """Audit current files at already-loaded module paths without caller input."""

    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    try:
        raw_receipt = source_identity_module.audit_runtime_candidate_source_identity(
            candidate_manifest=candidate,
            expected_candidate_sha256=expected_candidate_sha256,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Runtime source-module identity audit failed"
        ) from exc
    receipt = _mapping_snapshot(raw_receipt, "runtime source identity receipt")
    observed_receipt_hash = _sha256(
        receipt.get("source_identity_receipt_sha256"),
        "runtime source identity receipt hash",
    )
    body = {
        key: receipt[key]
        for key in receipt
        if key != "source_identity_receipt_sha256"
    }
    if not hmac.compare_digest(observed_receipt_hash, canonical_sha256(body)):
        raise SecFilingGemmaStageVerifierError(
            "Runtime source identity receipt is not canonical"
        )
    expected_tree = candidate.get("bindings", {}).get("source_tree_sha256")
    if (
        receipt.get("candidate_sha256") != expected_candidate_sha256
        or receipt.get("candidate_source_tree_sha256") != expected_tree
        or receipt.get("computed_source_tree_sha256") != expected_tree
        or receipt.get("required_source_roles") != list(REQUIRED_SOURCE_HASHES)
        or receipt.get("declared_static_local_import_closure_complete") is not True
        or receipt.get("runtime_source_files_verified") is not True
        or receipt.get("runtime_module_paths_verified") is not True
        or receipt.get("runtime_module_files_attested") is not True
        or receipt.get("runtime_executing_code_bytes_attested") is not False
        or receipt.get("caller_supplied_paths_or_bytes_accepted") is not False
        or receipt.get("complete") is not False
        or receipt.get("authorizes") is not False
    ):
        raise SecFilingGemmaStageVerifierError(
            "Runtime source identity receipt crossed its non-substitutable boundary"
        )
    return receipt


def _detached_replay_trust_boundary(
    receipt: Mapping[str, Any], location: str
) -> dict[str, bool]:
    boundary = {
        "authorizing": receipt.get("authorizing"),
        "fresh_network_provenance_verified": receipt.get(
            "fresh_network_provenance_verified"
        ),
        "network_receipt_claims_replayed_not_observed": receipt.get(
            "network_receipt_claims_replayed_not_observed"
        ),
    }
    expected = {
        "authorizing": False,
        "fresh_network_provenance_verified": False,
        "network_receipt_claims_replayed_not_observed": True,
    }
    if boundary != expected:
        raise SecFilingGemmaStageVerifierError(
            f"{location} changed its detached non-authorizing trust boundary"
        )
    return expected


def validate_detached_catalog_evidence(
    value: Mapping[str, Any],
    *,
    candidate_manifest: Mapping[str, Any],
    corpus_universe_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the exact detached SEC catalogue bytes into the bound universe."""

    _preflight_catalog_payloads(value)
    replay = _mapping_snapshot(value, "detached catalogue evidence")
    _expect_keys(
        replay,
        {
            "source_payloads",
            "request_receipts",
            "catalog_artifact",
            "expected_source_payload_sha256s",
            "expected_request_receipt_sha256s",
            "expected_request_receipts_sha256",
            "expected_catalog_artifact_sha256",
            "expected_corpus_universe_sha256",
            "expected_calendar_artifact_sha256",
        },
        "detached catalogue evidence",
    )
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    universe = _mapping_snapshot(corpus_universe_manifest, "corpus universe")
    sources = replay["source_payloads"]
    if type(sources) is not list or not sources:
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue source payloads must be a nonempty list"
        )
    if len(sources) > MAX_CATALOG_SOURCE_ITEMS:
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue source payload count exceeds its fixed ceiling"
        )
    aggregate_source_bytes = 0
    for index, item in enumerate(sources):
        if type(item) is not dict:
            raise SecFilingGemmaStageVerifierError(
                "Detached catalogue source item must be an object"
            )
        _expect_keys(
            item, {"name", "payload_base64"}, f"catalogue sources[{index}]"
        )
        aggregate_source_bytes += _base64_decoded_length_preflight(
            item["payload_base64"],
            f"catalogue sources[{index}].payload_base64",
            maximum=MAX_BASE64_ITEM_BYTES,
        )
        if aggregate_source_bytes > MAX_CATALOG_SOURCE_BYTES:
            raise SecFilingGemmaStageVerifierError(
                "Detached catalogue source payloads exceed their aggregate byte ceiling"
            )
    decoded_sources = [
        {
            "name": item["name"],
            "payload": _decode_base64(
                item["payload_base64"],
                f"catalogue sources[{index}].payload_base64",
                maximum=MAX_BASE64_ITEM_BYTES,
            ),
        }
        for index, item in enumerate(sources)
    ]
    candidate_bindings = candidate["bindings"]
    expected_catalog = _sha256(
        replay["expected_catalog_artifact_sha256"],
        "expected catalogue artifact hash",
    )
    expected_universe = _sha256(
        replay["expected_corpus_universe_sha256"],
        "expected corpus universe hash",
    )
    expected_calendar = _sha256(
        replay["expected_calendar_artifact_sha256"],
        "expected calendar artifact hash",
    )
    if (
        expected_catalog != candidate_bindings["sec_catalog_artifact_sha256"]
        or expected_catalog != universe.get("catalog_artifact_sha256")
        or expected_universe != candidate_bindings["corpus_universe_sha256"]
        or expected_universe != universe.get("universe_sha256")
        or expected_calendar
        != candidate_bindings["calendar_source_evidence_sha256"]
        or expected_calendar != universe.get("calendar_artifact_sha256")
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue external pins differ from candidate or universe"
        )
    catalog_artifact = replay["catalog_artifact"]
    if (
        type(catalog_artifact) is not dict
        or catalog_artifact.get("catalog_artifact_sha256") != expected_catalog
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue artifact differs from the candidate pin"
        )
    try:
        receipt = validate_detached_catalog_replay(
            source_payloads=decoded_sources,
            request_receipts=replay["request_receipts"],
            catalog_artifact=catalog_artifact,
            corpus_universe_manifest=universe,
            expected_source_payload_sha256s=replay[
                "expected_source_payload_sha256s"
            ],
            expected_request_receipt_sha256s=replay[
                "expected_request_receipt_sha256s"
            ],
            expected_request_receipts_sha256=replay[
                "expected_request_receipts_sha256"
            ],
            expected_catalog_artifact_sha256=expected_catalog,
            expected_corpus_universe_sha256=expected_universe,
            expected_calendar_artifact_sha256=expected_calendar,
            session_dates=list(EXPECTED_SESSIONS),
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue byte replay failed"
        ) from exc
    if (
        DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION
        == DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
        or receipt["schema_version"]
        != DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue replay receipt schema is missing or ambiguous"
        )
    if (
        receipt["catalog_artifact_sha256"] != expected_catalog
        or receipt["corpus_universe_sha256"] != expected_universe
        or receipt["request_receipts_sha256"]
        != replay["expected_request_receipts_sha256"]
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue replay receipt crossed an external binding"
        )
    trust_boundary = _detached_replay_trust_boundary(
        receipt, "Detached catalogue replay"
    )
    return {
        **trust_boundary,
        "schema_version": receipt["schema_version"],
        "catalog_artifact_sha256": receipt["catalog_artifact_sha256"],
        "corpus_universe_sha256": receipt["corpus_universe_sha256"],
        "eligible_records_sha256": receipt["eligible_records_sha256"],
        "request_receipts_sha256": receipt["request_receipts_sha256"],
        "source_payload_sha256s": dict(receipt["source_payload_sha256s"]),
        "request_receipt_sha256s": dict(receipt["request_receipt_sha256s"]),
        "replay_validation_sha256": receipt["replay_validation_sha256"],
    }


def validate_detached_stage_content_evidence(
    value: Mapping[str, Any],
    *,
    expected_stage: str,
    candidate_manifest: Mapping[str, Any],
    corpus_universe_manifest: Mapping[str, Any],
    externally_pinned_content_manifest_sha256: str,
    externally_pinned_stage_artifact_sha256: str,
    maximum_total_document_bytes: int,
) -> dict[str, Any]:
    """Replay one stage's exact primary bytes and deterministic normalization."""

    if expected_stage not in STAGE_ORDER:
        raise SecFilingGemmaStageVerifierError("Detached content expected stage is invalid")
    if (
        type(maximum_total_document_bytes) is not int
        or not 1 <= maximum_total_document_bytes <= MAX_SEC_BYTES
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached stage aggregate document-byte ceiling is invalid"
        )
    _preflight_stage_documents(
        value,
        expected_stage=expected_stage,
        maximum_items=STAGE_MODEL_CALL_CAPS[expected_stage],
        maximum_total_bytes=maximum_total_document_bytes,
    )
    replay = _mapping_snapshot(value, f"detached {expected_stage} content evidence")
    _expect_keys(
        replay,
        {
            "stage",
            "document_payloads",
            "request_receipts",
            "content_manifest",
            "stage_artifact",
            "expected_document_sha256s",
            "expected_normalized_text_sha256s",
            "expected_request_receipt_sha256s",
            "expected_request_receipts_sha256",
            "expected_content_manifest_sha256",
            "expected_stage_artifact_sha256",
        },
        f"detached {expected_stage} content evidence",
    )
    if replay["stage"] != expected_stage:
        raise SecFilingGemmaStageVerifierError(
            "Detached content evidence crossed a stage slot"
        )
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    universe = _mapping_snapshot(corpus_universe_manifest, "corpus universe")
    documents = replay["document_payloads"]
    if type(documents) is not list or not documents:
        raise SecFilingGemmaStageVerifierError(
            "Detached stage document payloads must be a nonempty list"
        )
    universe_records = universe.get("records")
    if type(universe_records) is not list:
        raise SecFilingGemmaStageVerifierError(
            "Detached content universe records are unavailable"
        )
    expected_record_count = sum(
        type(record) is dict and record.get("artifact_stage") == expected_stage
        for record in universe_records
    )
    if (
        not 1 <= len(documents) <= STAGE_MODEL_CALL_CAPS[expected_stage]
        or len(documents) != expected_record_count
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached stage document count differs from the bounded universe stage"
        )
    aggregate_document_bytes = 0
    for index, item in enumerate(documents):
        if type(item) is not dict:
            raise SecFilingGemmaStageVerifierError(
                "Detached stage document item must be an object"
            )
        _expect_keys(
            item,
            {"accession_number", "payload_base64"},
            f"detached {expected_stage} documents[{index}]",
        )
        aggregate_document_bytes += _base64_decoded_length_preflight(
            item["payload_base64"],
            f"detached {expected_stage} documents[{index}].payload_base64",
            maximum=MAX_BASE64_ITEM_BYTES,
        )
        if aggregate_document_bytes > maximum_total_document_bytes:
            raise SecFilingGemmaStageVerifierError(
                "Detached stage documents exceed the externally bounded aggregate bytes"
            )
    decoded_documents = [
        {
            "accession_number": item["accession_number"],
            "payload": _decode_base64(
                item["payload_base64"],
                f"detached {expected_stage} documents[{index}].payload_base64",
                maximum=MAX_BASE64_ITEM_BYTES,
            ),
        }
        for index, item in enumerate(documents)
    ]
    universe_hash = candidate["bindings"]["corpus_universe_sha256"]
    if universe.get("universe_sha256") != universe_hash:
        raise SecFilingGemmaStageVerifierError(
            "Detached content universe differs from the candidate"
        )
    content_manifest = replay["content_manifest"]
    stage_artifact = replay["stage_artifact"]
    expected_content = _sha256(
        replay["expected_content_manifest_sha256"],
        "expected content manifest hash",
    )
    expected_artifact = _sha256(
        replay["expected_stage_artifact_sha256"],
        "expected stage artifact hash",
    )
    pinned_content = _sha256(
        externally_pinned_content_manifest_sha256,
        "externally pinned content manifest hash",
    )
    pinned_artifact = _sha256(
        externally_pinned_stage_artifact_sha256,
        "externally pinned stage artifact hash",
    )
    if expected_content != pinned_content or expected_artifact != pinned_artifact:
        raise SecFilingGemmaStageVerifierError(
            "Detached content hashes differ from the pre-existing trusted pins"
        )
    if (
        type(content_manifest) is not dict
        or content_manifest.get("artifact_stage") != expected_stage
        or content_manifest.get("corpus_universe_sha256") != universe_hash
        or content_manifest.get("content_manifest_sha256") != expected_content
        or type(stage_artifact) is not dict
        or stage_artifact.get("artifact_stage") != expected_stage
        or stage_artifact.get("corpus_universe_sha256") != universe_hash
        or stage_artifact.get("content_manifest_sha256") != expected_content
        or stage_artifact.get("stage_artifact_sha256") != expected_artifact
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached content manifests crossed a stage, universe, or artifact binding"
        )
    try:
        receipt = validate_detached_stage_content_replay(
            authorized_stage=expected_stage,
            document_payloads=decoded_documents,
            request_receipts=replay["request_receipts"],
            content_manifest=content_manifest,
            stage_artifact=stage_artifact,
            corpus_universe_manifest=universe,
            expected_document_sha256s=replay["expected_document_sha256s"],
            expected_normalized_text_sha256s=replay[
                "expected_normalized_text_sha256s"
            ],
            expected_request_receipt_sha256s=replay[
                "expected_request_receipt_sha256s"
            ],
            expected_request_receipts_sha256=replay[
                "expected_request_receipts_sha256"
            ],
            expected_content_manifest_sha256=expected_content,
            expected_stage_artifact_sha256=expected_artifact,
            expected_corpus_universe_sha256=universe_hash,
            session_dates=list(EXPECTED_SESSIONS),
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            f"Detached {expected_stage} content byte replay failed"
        ) from exc
    if (
        DETACHED_CATALOG_REPLAY_RECEIPT_SCHEMA_VERSION
        == DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
        or receipt["schema_version"]
        != DETACHED_STAGE_CONTENT_REPLAY_RECEIPT_SCHEMA_VERSION
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached content replay receipt schema is missing or ambiguous"
        )
    if (
        receipt["artifact_stage"] != expected_stage
        or receipt["corpus_universe_sha256"] != universe_hash
        or receipt["content_manifest_sha256"] != expected_content
        or receipt["stage_artifact_sha256"] != expected_artifact
        or receipt["request_receipts_sha256"]
        != replay["expected_request_receipts_sha256"]
    ):
        raise SecFilingGemmaStageVerifierError(
            "Detached content replay receipt crossed an external binding"
        )
    trust_boundary = _detached_replay_trust_boundary(
        receipt, f"Detached {expected_stage} content replay"
    )
    return {
        **trust_boundary,
        "schema_version": receipt["schema_version"],
        "artifact_stage": receipt["artifact_stage"],
        "corpus_universe_sha256": receipt["corpus_universe_sha256"],
        "content_manifest_sha256": receipt["content_manifest_sha256"],
        "stage_artifact_sha256": receipt["stage_artifact_sha256"],
        "request_receipts_sha256": receipt["request_receipts_sha256"],
        "primary_document_sha256s": dict(receipt["primary_document_sha256s"]),
        "normalized_text_sha256s": dict(receipt["normalized_text_sha256s"]),
        "request_receipt_sha256s": dict(receipt["request_receipt_sha256s"]),
        "replay_validation_sha256": receipt["replay_validation_sha256"],
    }


def validate_calendar_and_universe_snapshot(
    *,
    calendar_evidence_manifest: Mapping[str, Any],
    calendar_source_bytes_base64_by_name: Mapping[str, str],
    corpus_universe_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact source-byte hashes and the complete frozen universe.

    This proves byte identity, not the still-missing semantic parse from the
    official NYSE pages/PDFs into the session sequence.
    """

    _preflight_base64_mapping_values(
        calendar_source_bytes_base64_by_name,
        "calendar source bytes",
        maximum_items=16,
        maximum_item_bytes=MAX_BASE64_ITEM_BYTES,
        maximum_total_bytes=256 * 1024 * 1024,
    )
    calendar = _mapping_snapshot(calendar_evidence_manifest, "calendar evidence")
    source_bytes = _mapping_snapshot(
        calendar_source_bytes_base64_by_name, "calendar source bytes"
    )
    universe = _mapping_snapshot(corpus_universe_manifest, "corpus universe")
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    records = calendar.get("source_records")
    if type(records) is not dict:
        raise SecFilingGemmaStageVerifierError("Calendar source records are invalid")
    _expect_keys(source_bytes, set(records), "calendar source bytes")
    for name, record in records.items():
        if type(record) is not dict:
            raise SecFilingGemmaStageVerifierError("Calendar source record is invalid")
        payload = _decode_base64(
            source_bytes[name], f"calendar source bytes {name}", maximum=64 * 1024 * 1024
        )
        if (
            hashlib.sha256(payload).hexdigest() != record.get("content_sha256")
            or len(payload) != record.get("byte_count")
        ):
            raise SecFilingGemmaStageVerifierError(
                f"Calendar source bytes for {name} do not match the manifest"
            )
    try:
        calendar_hash = validate_calendar_source_evidence_manifest(
            calendar,
            expected_calendar_source_evidence_sha256=candidate["bindings"][
                "calendar_source_evidence_sha256"
            ],
        )
        universe_hash = validate_corpus_universe_manifest(
            universe,
            session_dates=EXPECTED_SESSIONS,
            expected_universe_sha256=candidate["bindings"]["corpus_universe_sha256"],
            require_complete_coverage=True,
        )
    except (SecFilingGemmaContractError, TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Calendar or complete-universe replay failed"
        ) from exc
    for observed, expected, label in (
        (universe.get("calendar_artifact_sha256"), calendar_hash, "calendar artifact"),
        (
            universe.get("calendar_sessions_sha256"),
            candidate["bindings"]["calendar_sessions_sha256"],
            "calendar sessions",
        ),
        (
            universe.get("catalog_artifact_sha256"),
            candidate["bindings"]["sec_catalog_artifact_sha256"],
            "SEC catalog",
        ),
        (
            universe.get("universe_semantic_sha256"),
            candidate["bindings"]["corpus_universe_semantic_sha256"],
            "universe semantics",
        ),
    ):
        if observed != expected:
            raise SecFilingGemmaStageVerifierError(
                f"Corpus {label} differs from the candidate"
            )
    return {
        "calendar_source_evidence_sha256": calendar_hash,
        "corpus_universe_sha256": universe_hash,
        "calendar_source_bytes_sha256": canonical_sha256(
            {
                name: hashlib.sha256(
                    _decode_base64(source_bytes[name], f"calendar {name}")
                ).hexdigest()
                for name in sorted(source_bytes)
            }
        ),
    }


def validate_model_attempt_batch(
    batch: Mapping[str, Any],
    *,
    candidate_manifest: Mapping[str, Any],
    expected_accessions: Sequence[str],
) -> dict[str, Any]:
    """Replay exact request/response bytes and one before/after runtime guard."""

    _preflight_plain_json(batch, "model batch")
    if not _is_plain_mapping(batch, top_level=True):
        raise SecFilingGemmaStageVerifierError("model batch must be a mapping")
    preflight_stage = batch.get("stage")
    preflight_attempts = batch.get("attempts")
    if preflight_stage not in STAGE_ORDER:
        raise SecFilingGemmaStageVerifierError("Model batch stage is invalid")
    if (
        type(preflight_attempts) is not list
        or not 1 <= len(preflight_attempts) <= STAGE_MODEL_CALL_CAPS[preflight_stage]
    ):
        raise SecFilingGemmaStageVerifierError(
            "Model batch count exceeds the fixed stage cap"
        )
    value = _mapping_snapshot(batch, "model batch")
    _expect_keys(
        value,
        {
            "stage",
            "before_runtime_evidence",
            "after_runtime_evidence",
            "runtime_guard",
            "attempts",
        },
        "model batch",
    )
    stage = value["stage"]
    if stage not in STAGE_ORDER:
        raise SecFilingGemmaStageVerifierError("Model batch stage is invalid")
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    attempts = value["attempts"]
    if type(attempts) is not list or not attempts:
        raise SecFilingGemmaStageVerifierError("Model batch attempts must be nonempty")
    expected = list(expected_accessions)
    if len(attempts) != len(expected) or len(attempts) > STAGE_MODEL_CALL_CAPS[stage]:
        raise SecFilingGemmaStageVerifierError(
            "Model batch count differs from exact stage coverage or exceeds its cap"
        )
    before = value["before_runtime_evidence"]
    after = value["after_runtime_evidence"]
    if type(before) is not dict or type(after) is not dict:
        raise SecFilingGemmaStageVerifierError("Runtime probes must be detached objects")
    runtime_evidence_hash = canonical_sha256(before)
    receipt_hashes: list[str] = []
    observed_accessions: list[str] = []
    elapsed = 0
    for index, item in enumerate(attempts):
        if type(item) is not dict:
            raise SecFilingGemmaStageVerifierError("Model attempt item must be an object")
        _expect_keys(
            item,
            {
                "accession_number",
                "expected_model_payload",
                "expected_sentence_ids",
                "receipt",
            },
            f"model attempts[{index}]",
        )
        accession = item["accession_number"]
        if accession != expected[index]:
            raise SecFilingGemmaStageVerifierError(
                "Model attempt coverage is omitted, duplicated, or reordered"
            )
        payload = item["expected_model_payload"]
        sentence_ids = item["expected_sentence_ids"]
        if type(payload) is not dict or type(sentence_ids) is not list:
            raise SecFilingGemmaStageVerifierError(
                "Model attempt expected payload or sentence IDs are not detached"
            )
        try:
            receipt = validate_ollama_model_attempt_receipt(
                item["receipt"],
                expected_candidate_sha256=candidate["candidate_sha256"],
                expected_model_payload_sha256=canonical_sha256(payload),
                expected_sentence_ids=sentence_ids,
                expected_runtime_evidence_sha256=runtime_evidence_hash,
                expected_model_digest=candidate["model"]["digest"],
                expected_runtime_fingerprint_sha256=candidate["model"][
                    "runtime_fingerprint_sha256"
                ],
                expected_transport_mode=OWNED_HARDENED_TRANSPORT_MODE,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise SecFilingGemmaStageVerifierError(
                f"Model attempt {accession} did not replay from exact bytes"
            ) from exc
        if json.loads(receipt.request_bytes.decode("utf-8")) != payload:
            raise SecFilingGemmaStageVerifierError(
                "Model receipt request differs from the independent payload"
            )
        receipt_hashes.append(receipt.receipt_sha256)
        observed_accessions.append(accession)
        elapsed += receipt.elapsed_nanoseconds
    guard = value["runtime_guard"]
    if type(guard) is not dict:
        raise SecFilingGemmaStageVerifierError("Runtime guard must be detached")
    try:
        guard_hash = validate_runtime_identity_guard(
            guard,
            before_evidence=before,
            after_evidence=after,
            expected_model_digest=candidate["model"]["digest"],
            expected_runtime_fingerprint_sha256=candidate["model"][
                "runtime_fingerprint_sha256"
            ],
            stage=stage,
            model_call_receipt_sha256s=receipt_hashes,
            expected_runtime_guard_sha256=guard.get("runtime_guard_sha256"),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Before/after runtime guard or exact receipt order did not replay"
        ) from exc
    return {
        "stage": stage,
        "accessions_sha256": canonical_sha256(observed_accessions),
        "model_attempt_receipt_sha256s": receipt_hashes,
        "runtime_guard_sha256": guard_hash,
        "model_elapsed_nanoseconds": elapsed,
    }


def validate_market_snapshot_stage_replay(value: Mapping[str, Any]) -> dict[str, Any]:
    """Replay canonical snapshot bytes into one structural market stage."""

    _preflight_plain_json(value, "market replay")
    if not _is_plain_mapping(value, top_level=True):
        raise SecFilingGemmaStageVerifierError("market replay must be a mapping")
    _preflight_base64_mapping_values(
        value.get("artifact_bytes_base64_by_symbol"),
        "market artifact bytes",
        maximum_items=16,
        maximum_item_bytes=MAX_BASE64_ITEM_BYTES,
        maximum_total_bytes=256 * 1024 * 1024,
    )
    _preflight_base64_mapping_values(
        value.get("window_bytes_base64_by_symbol"),
        "market window bytes",
        maximum_items=16,
        maximum_item_bytes=MAX_BASE64_ITEM_BYTES,
        maximum_total_bytes=256 * 1024 * 1024,
    )
    replay = _mapping_snapshot(value, "market replay")
    _expect_keys(
        replay,
        {
            "stage",
            "source_manifest",
            "market_stage_manifest",
            "artifact_bytes_base64_by_symbol",
            "window_bytes_base64_by_symbol",
        },
        "market replay",
    )
    artifacts = {
        symbol: _decode_base64(encoded, f"market artifact {symbol}")
        for symbol, encoded in _mapping_snapshot(
            replay["artifact_bytes_base64_by_symbol"], "market artifact bytes"
        ).items()
    }
    windows = {
        symbol: _decode_base64(encoded, f"market window {symbol}")
        for symbol, encoded in _mapping_snapshot(
            replay["window_bytes_base64_by_symbol"], "market window bytes"
        ).items()
    }
    source = replay["source_manifest"]
    stage_manifest = replay["market_stage_manifest"]
    if type(source) is not dict or type(stage_manifest) is not dict:
        raise SecFilingGemmaStageVerifierError("Market manifests must be detached")
    try:
        return validate_market_source_bytes_against_stage(
            artifact_bytes_by_symbol=artifacts,
            window_bytes_by_symbol=windows,
            source_manifest=source,
            stage_manifest=stage_manifest,
            expected_artifact_stage=replay["stage"],
            expected_source_manifest_sha256=source["source_manifest_sha256"],
            expected_market_stage_manifest_sha256=stage_manifest[
                "market_stage_manifest_sha256"
            ],
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Market snapshot-to-stage replay failed"
        ) from exc


def validate_prediction_artifact_replay(
    value: Mapping[str, Any],
    *,
    candidate_sha256: str,
    corpus_universe_sha256: str,
) -> dict[str, Any]:
    """Validate exact seal CAS bytes and replay every prediction transition."""

    _preflight_plain_json(value, "prediction replay")
    if not _is_plain_mapping(value, top_level=True):
        raise SecFilingGemmaStageVerifierError("prediction replay must be a mapping")
    prediction_fields = (
        ("artifact_bytes_base64", MAX_PREDICTION_ARTIFACT_BYTES),
        ("seal_receipt_bytes_base64", MAX_SMALL_RECEIPT_BYTES),
        ("external_prior_pin_bytes_base64", MAX_SMALL_RECEIPT_BYTES),
        ("external_next_pin_bytes_base64", MAX_SMALL_RECEIPT_BYTES),
    )
    prediction_total = 0
    for field, maximum in prediction_fields:
        prediction_total += _base64_decoded_length_preflight(
            value.get(field), f"prediction replay.{field}", maximum=maximum
        )
    if prediction_total > MAX_PREDICTION_ARTIFACT_BYTES + 3 * MAX_SMALL_RECEIPT_BYTES:
        raise SecFilingGemmaStageVerifierError(
            "Prediction replay exceeds its aggregate byte ceiling"
        )
    replay = _mapping_snapshot(value, "prediction replay")
    _expect_keys(
        replay,
        {
            "artifact_bytes_base64",
            "seal_receipt_bytes_base64",
            "external_prior_pin_bytes_base64",
            "external_next_pin_bytes_base64",
            "event_bindings",
        },
        "prediction replay",
    )
    artifact = _decode_base64(
        replay["artifact_bytes_base64"],
        "prediction artifact",
        maximum=MAX_PREDICTION_ARTIFACT_BYTES,
    )
    seal_receipt = _decode_base64(
        replay["seal_receipt_bytes_base64"],
        "prediction seal receipt",
        maximum=MAX_SMALL_RECEIPT_BYTES,
    )
    prior_pin = _decode_base64(
        replay["external_prior_pin_bytes_base64"],
        "prediction prior pin",
        maximum=MAX_SMALL_RECEIPT_BYTES,
    )
    next_pin = _decode_base64(
        replay["external_next_pin_bytes_base64"],
        "prediction next pin",
        maximum=MAX_SMALL_RECEIPT_BYTES,
    )
    try:
        seal = validate_prediction_artifact_seal_receipt(
            artifact_bytes=artifact,
            receipt_bytes=seal_receipt,
            external_prior_pin_bytes=prior_pin,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Prediction artifact seal CAS replay failed"
        ) from exc
    if seal.next_external_pin_bytes != next_pin:
        raise SecFilingGemmaStageVerifierError(
            "Prediction seal does not match the externally pinned next state"
        )
    prefix = _strict_json_bytes(artifact, "prediction artifact")
    events = replay["event_bindings"]
    if type(events) is not list:
        raise SecFilingGemmaStageVerifierError("Prediction event bindings must be a list")
    try:
        summary = validate_prediction_prefix(
            prefix,
            session_dates=EXPECTED_SESSIONS,
            expected_calendar_sessions_sha256=prefix["calendar_sessions_sha256"],
            expected_candidate_sha256=candidate_sha256,
            expected_corpus_universe_sha256=corpus_universe_sha256,
            expected_event_bindings=events,
            expected_prediction_prefix_sha256=seal.prediction_prefix_sha256,
            expected_tip_sha256=seal.prediction_tip_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Prediction chronology, policy state, or prefix ancestry failed replay"
        ) from exc
    if seal.candidate_sha256 != candidate_sha256:
        raise SecFilingGemmaStageVerifierError("Prediction artifact belongs to another candidate")
    return {
        **summary,
        "artifact_sha256": seal.artifact_sha256,
        "seal_sha256": seal.seal_sha256,
        "seal_receipt_sha256": seal.receipt_sha256,
        "stage": seal.stage,
        "prefix": prefix,
    }


def _decode_matrix(value: Any, location: str) -> list[list[float]]:
    if type(value) is not list or not value:
        raise SecFilingGemmaStageVerifierError(f"{location} must be a nonempty matrix")
    result: list[list[float]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        if type(row) is not list or not row:
            raise SecFilingGemmaStageVerifierError(f"{location}[{row_index}] is invalid")
        decoded: list[float] = []
        for column_index, item in enumerate(row):
            if not isinstance(item, str):
                raise SecFilingGemmaStageVerifierError(f"{location} must use float.hex")
            try:
                number = float.fromhex(item)
            except ValueError as exc:
                raise SecFilingGemmaStageVerifierError(f"{location} has invalid float.hex") from exc
            if not math.isfinite(number) or number.hex() != item:
                raise SecFilingGemmaStageVerifierError(f"{location} has noncanonical float.hex")
            decoded.append(number)
        if width is None:
            width = len(decoded)
        elif len(decoded) != width:
            raise SecFilingGemmaStageVerifierError(f"{location} is ragged")
        result.append(decoded)
    return result


def validate_learner_refit_replays(
    replays: Sequence[Mapping[str, Any]],
    *,
    candidate_sha256: str,
    prediction_prefix: Mapping[str, Any],
) -> dict[str, Any]:
    """Refit every supplied fold/variant and replay its row predictions exactly."""

    if isinstance(replays, (str, bytes)) or not isinstance(replays, Sequence) or not replays:
        raise SecFilingGemmaStageVerifierError("Learner replays must be a nonempty sequence")
    prefix_rows = prediction_prefix.get("rows")
    if type(prefix_rows) is not list:
        raise SecFilingGemmaStageVerifierError("Prediction prefix rows are invalid")
    rows_by_hash = {row["prediction_row_sha256"]: row for row in prefix_rows}
    available = [
        row for row in prefix_rows if row.get("prediction_status") == AVAILABLE_PREDICTION_STATUS
    ]
    contexts_by_fold: dict[str, dict[str, Any]] = {}
    for row in available:
        fold_id = row.get("fold_id")
        context = row.get("fold_context")
        if not isinstance(fold_id, str) or type(context) is not dict:
            raise SecFilingGemmaStageVerifierError(
                "Available prediction row lacks a detached fold context"
            )
        _expect_keys(context, set(_LEARNER_FOLD_CONTEXT_KEYS), "prediction fold context")
        prior = contexts_by_fold.setdefault(fold_id, context)
        if prior != context:
            raise SecFilingGemmaStageVerifierError(
                "Available predictions disagree about their fixed fold context"
            )
    seen_cases: set[tuple[str, str]] = set()
    seen_fits: set[tuple[str, str]] = set()
    state_hashes: dict[str, dict[str, str]] = {}
    input_identity_hashes: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(replays):
        item = _mapping_snapshot(raw, f"learner replays[{index}]")
        _expect_keys(
            item,
            {
                "fold_id",
                "variant",
                "feature_names",
                "training_features_hex",
                "binary_targets",
                "edge_targets_hex",
                "fit_metadata",
                "expected_state",
                "prediction_cases",
            },
            f"learner replays[{index}]",
        )
        variant = item["variant"]
        if variant not in {"semantic", "ablation"}:
            raise SecFilingGemmaStageVerifierError("Learner variant is invalid")
        fit_key = (item["fold_id"], variant)
        if fit_key in seen_fits:
            raise SecFilingGemmaStageVerifierError(
                "Learner fold/variant refit is duplicated"
            )
        seen_fits.add(fit_key)
        features = _decode_matrix(item["training_features_hex"], "training features")
        edges_matrix = _decode_matrix(
            [[edge] for edge in item["edge_targets_hex"]], "edge targets"
        )
        edges = [row[0] for row in edges_matrix]
        targets = item["binary_targets"]
        names = item["feature_names"]
        metadata = item["fit_metadata"]
        if type(targets) is not list or type(names) is not list or type(metadata) is not dict:
            raise SecFilingGemmaStageVerifierError("Learner inputs are not detached")
        _expect_keys(metadata, set(_LEARNER_FIT_METADATA_KEYS), "learner fit metadata")
        if any(type(target) is not int or target not in {0, 1} for target in targets):
            raise SecFilingGemmaStageVerifierError(
                "Learner binary targets must be canonical integer zero or one"
            )
        if metadata.get("candidate_sha256") != candidate_sha256 or metadata.get(
            "head_variant"
        ) != variant or metadata.get("fold_id") != item["fold_id"]:
            raise SecFilingGemmaStageVerifierError("Learner metadata crossed a binding")
        context = contexts_by_fold.get(item["fold_id"])
        if context is None:
            raise SecFilingGemmaStageVerifierError(
                "Learner refit has no available prediction fold context"
            )
        feature_matrix_hash = canonical_sha256(item["training_features_hex"])
        binary_target_hash = canonical_sha256(targets)
        edge_target_hash = canonical_sha256(item["edge_targets_hex"])
        if metadata.get("training_set_sha256") != context.get(
            "training_set_membership_sha256"
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training membership differs from the prediction fold context"
            )
        if feature_matrix_hash != context.get(
            f"{variant}_training_feature_matrix_sha256"
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training feature matrix differs from the prediction fold context"
            )
        if (
            binary_target_hash != context.get("training_binary_target_sha256")
            or edge_target_hash != context.get("training_edge_target_sha256")
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training targets differ from the prediction fold context"
            )
        if (
            metadata.get("training_row_count") != len(features)
            or context.get("training_set_count") != len(features)
            or context.get("training_positive_count") != sum(targets)
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training counts differ from the prediction fold context"
            )
        if (
            metadata.get("train_label_maturity_through")
            != context.get("fold_train_cutoff_session")
            or metadata.get("maximum_training_label_maturity_session")
            != context.get("training_set_max_label_maturity_session")
        ):
            raise SecFilingGemmaStageVerifierError(
                "Learner training maturity differs from the prediction fold context"
            )
        input_identity = {
            "training_set_membership_sha256": metadata["training_set_sha256"],
            "training_feature_matrix_sha256": feature_matrix_hash,
            "training_binary_target_sha256": binary_target_hash,
            "training_edge_target_sha256": edge_target_hash,
            "feature_schema_sha256": metadata["feature_schema_sha256"],
            "training_row_count": len(features),
            "training_positive_count": sum(targets),
            "fold_train_cutoff_session": metadata["train_label_maturity_through"],
            "training_set_max_label_maturity_session": metadata[
                "maximum_training_label_maturity_session"
            ],
        }
        input_identity_hashes.setdefault(item["fold_id"], {})[variant] = canonical_sha256(
            input_identity
        )
        try:
            learner = SecFilingGemmaTwoHeadLearner().fit(
                features,
                targets,
                edges,
                feature_names=names,
                fit_metadata=metadata,
            )
        except (TypeError, ValueError) as exc:
            raise SecFilingGemmaStageVerifierError("Deterministic learner refit failed") from exc
        state = item["expected_state"]
        if type(state) is not dict or learner.to_state() != state:
            raise SecFilingGemmaStageVerifierError(
                "Persisted learner state differs from deterministic refit"
            )
        state_hashes.setdefault(item["fold_id"], {})[variant] = learner.model_sha256
        cases = item["prediction_cases"]
        if type(cases) is not list:
            raise SecFilingGemmaStageVerifierError("Learner prediction cases must be a list")
        for case in cases:
            if type(case) is not dict:
                raise SecFilingGemmaStageVerifierError("Learner prediction case is invalid")
            _expect_keys(case, {"prediction_row_sha256", "features_hex"}, "learner case")
            row_hash = case["prediction_row_sha256"]
            row = rows_by_hash.get(row_hash)
            if row is None or row.get("fold_id") != item["fold_id"]:
                raise SecFilingGemmaStageVerifierError("Learner case crossed a prediction fold")
            case_key = (row_hash, variant)
            if case_key in seen_cases:
                raise SecFilingGemmaStageVerifierError("Learner prediction case is duplicated")
            seen_cases.add(case_key)
            matrix = _decode_matrix([case["features_hex"]], "prediction features")
            output = learner.predict_components(matrix)
            probability = float(output["cash_win_probability_10bps"][0]).hex()
            edge = float(output["expected_active_log_edge_10bps"][0]).hex()
            if (
                row[f"{variant}_cash_probability_hex"] != probability
                or row[f"{variant}_expected_edge_hex"] != edge
            ):
                raise SecFilingGemmaStageVerifierError(
                    "Prediction row differs from deterministic learner output"
                )
    expected_cases = {
        (row["prediction_row_sha256"], variant)
        for row in available
        for variant in ("semantic", "ablation")
    }
    if seen_cases != expected_cases:
        raise SecFilingGemmaStageVerifierError(
            "Learner replays omit or add an available prediction"
        )
    expected_fits = {
        (row["fold_id"], variant)
        for row in available
        for variant in ("semantic", "ablation")
    }
    if seen_fits != expected_fits:
        raise SecFilingGemmaStageVerifierError(
            "Learner replays omit or add a fold/variant fit"
        )
    for row in available:
        fold_states = state_hashes.get(row["fold_id"], {})
        context = row["fold_context"]
        if (
            fold_states.get("semantic") != context["semantic_fold_state_sha256"]
            or fold_states.get("ablation") != context["ablation_fold_state_sha256"]
        ):
            raise SecFilingGemmaStageVerifierError(
                "Prediction fold context differs from deterministic learner states"
            )
    return {
        "learner_state_sha256s_by_fold": state_hashes,
        "learner_input_identity_sha256s_by_fold": input_identity_hashes,
        "prediction_case_count": len(seen_cases),
        "learner_replay_sha256": canonical_sha256(
            {
                "states": state_hashes,
                "inputs": input_identity_hashes,
                "cases": sorted(seen_cases),
            }
        ),
    }


def validate_raw_scores_gates_and_ranking(
    value: Mapping[str, Any],
    *,
    prediction_prefix: Mapping[str, Any],
    market_stage: Mapping[str, Any],
    prerequisite_stage: str,
) -> dict[str, Any]:
    """Replay every raw score before any gate or ranking receipt is inspected."""

    replay = _mapping_snapshot(value, "score replay")
    _expect_keys(
        replay,
        {
            "label_release_evidence",
            "evaluations",
            "ranking_receipt",
            "selected_candidate_id",
        },
        "score replay",
    )
    labels = replay["label_release_evidence"]
    evaluations = replay["evaluations"]
    selected = replay["selected_candidate_id"]
    if type(labels) is not dict or type(evaluations) is not list or selected not in CANDIDATE_IDS:
        raise SecFilingGemmaStageVerifierError("Score replay inputs are invalid")
    prefix_hash = prediction_prefix["prediction_prefix_sha256"]
    market_hash = market_stage["market_stage_manifest_sha256"]
    label_hash = labels.get("label_release_ledger_sha256")
    _sha256(label_hash, "label release ledger hash")
    raw_score_hashes: list[str] = []
    gate_receipts: list[dict[str, Any]] = []
    gate_hashes: list[str] = []
    evaluation_ids: list[str] = []
    no_leverage_hashes: list[str] = []
    for evaluation in evaluations:
        if type(evaluation) is not dict:
            raise SecFilingGemmaStageVerifierError("Score evaluation must be detached")
        _expect_keys(evaluation, {"candidate_id", "score_receipts", "gate_receipt"}, "score evaluation")
        candidate_id = evaluation["candidate_id"]
        if candidate_id not in CANDIDATE_IDS or candidate_id in evaluation_ids:
            raise SecFilingGemmaStageVerifierError("Score candidate is invalid or duplicated")
        receipts = evaluation["score_receipts"]
        if type(receipts) is not list:
            raise SecFilingGemmaStageVerifierError("Raw score receipts must be a list")
        candidate_hashes: list[str] = []
        for receipt in receipts:
            if type(receipt) is not dict:
                raise SecFilingGemmaStageVerifierError("Raw score receipt is invalid")
            config = receipt.get("configuration")
            if type(config) is not dict or config.get("selected_candidate_id") != candidate_id:
                raise SecFilingGemmaStageVerifierError("Raw score crossed a candidate binding")
            score_hash = _sha256(receipt.get("score_receipt_sha256"), "score receipt hash")
            try:
                validate_score_receipt(
                    receipt,
                    expected_score_receipt_sha256=score_hash,
                    prediction_prefix=prediction_prefix,
                    expected_prediction_prefix_sha256=prefix_hash,
                    label_release_evidence=labels,
                    expected_label_release_ledger_sha256=label_hash,
                    market_stage=market_stage,
                    expected_market_stage_manifest_sha256=market_hash,
                    selected_candidate_id=candidate_id,
                    selected_variant=config["selected_variant"],
                    cost_bps=config["cost_bps"],
                    stage=prerequisite_stage,
                    score_cutoff_session=config["score_cutoff_session"],
                    terminal_convention=config["terminal_convention"],
                )
                proof = validate_sec_gemma_no_leverage_proof(
                    receipt, expected_score_receipt_sha256=score_hash
                )
            except (TypeError, ValueError) as exc:
                raise SecFilingGemmaStageVerifierError(
                    "Raw score or independent no-leverage replay failed"
                ) from exc
            candidate_hashes.append(score_hash)
            raw_score_hashes.append(score_hash)
            no_leverage_hashes.append(proof["proof_sha256"])
        gate = evaluation["gate_receipt"]
        if type(gate) is not dict:
            raise SecFilingGemmaStageVerifierError("Gate receipt is invalid")
        gate_hash = _sha256(gate.get("gate_receipt_sha256"), "gate receipt hash")
        try:
            validate_stage_gate_receipt(
                gate,
                expected_gate_receipt_sha256=gate_hash,
                score_receipts=receipts,
                expected_score_receipt_sha256s=candidate_hashes,
                stage=prerequisite_stage,
            )
        except (TypeError, ValueError) as exc:
            raise SecFilingGemmaStageVerifierError(
                "Gate receipt differs from the already-replayed raw scores"
            ) from exc
        evaluation_ids.append(candidate_id)
        gate_receipts.append(gate)
        gate_hashes.append(gate_hash)
    ranking_hash: str | None = None
    if prerequisite_stage == "development":
        if evaluation_ids != list(CANDIDATE_IDS) or type(replay["ranking_receipt"]) is not dict:
            raise SecFilingGemmaStageVerifierError(
                "Development requires the complete frozen candidate grid and ranking"
            )
        ranking = replay["ranking_receipt"]
        ranking_hash = _sha256(ranking.get("ranking_receipt_sha256"), "ranking hash")
        try:
            validate_development_ranking_receipt(
                ranking,
                expected_ranking_receipt_sha256=ranking_hash,
                gate_receipts=gate_receipts,
                expected_gate_receipt_sha256s=gate_hashes,
            )
        except (TypeError, ValueError) as exc:
            raise SecFilingGemmaStageVerifierError(
                "Development ranking differs from the replayed gates"
            ) from exc
        if ranking.get("selected_candidate_id") != selected:
            raise SecFilingGemmaStageVerifierError("Selected candidate differs from ranking")
    else:
        if evaluation_ids != [selected] or replay["ranking_receipt"] is not None:
            raise SecFilingGemmaStageVerifierError(
                "Intermediate stage must evaluate only the frozen selected candidate"
            )
    selected_gate = gate_receipts[evaluation_ids.index(selected)]
    if selected_gate.get("passed") is not True:
        raise SecFilingGemmaStageVerifierError("Selected prerequisite gate did not pass")
    return {
        "selected_candidate_id": selected,
        "raw_score_receipt_sha256s": raw_score_hashes,
        "gate_receipt_sha256s": gate_hashes,
        "ranking_receipt_sha256": ranking_hash,
        "no_leverage_proof_sha256s": no_leverage_hashes,
    }


def validate_stage_runtime_receipt(
    receipt: Mapping[str, Any],
    *,
    prerequisite_stage: str,
    candidate_sha256: str,
    model_batch_summary: Mapping[str, Any],
) -> str:
    """Validate one stage-local diagnostic receipt; it is not an attestation."""

    value = _mapping_snapshot(receipt, "stage runtime receipt")
    _expect_keys(
        value,
        {
            "schema_version",
            "stage",
            "candidate_sha256",
            "phase_elapsed_seconds",
            "sec_request_count",
            "sec_response_bytes",
            "model_attempt_receipt_sha256s",
            "runtime_guard_sha256",
            "paid_api_calls",
            "estimated_cost_usd_hex",
            "pull_attempts",
            "retries",
            "repair_attempts",
            "runtime_receipt_sha256",
        },
        "stage runtime receipt",
    )
    body = {key: value[key] for key in value if key != "runtime_receipt_sha256"}
    receipt_hash = _sha256(value["runtime_receipt_sha256"], "runtime receipt hash")
    if canonical_sha256(body) != receipt_hash:
        raise SecFilingGemmaStageVerifierError("Runtime receipt is not canonical")
    if (
        value["schema_version"] != STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION
        or value["stage"] != prerequisite_stage
        or value["candidate_sha256"] != candidate_sha256
        or value["model_attempt_receipt_sha256s"]
        != model_batch_summary["model_attempt_receipt_sha256s"]
        or value["runtime_guard_sha256"] != model_batch_summary["runtime_guard_sha256"]
    ):
        raise SecFilingGemmaStageVerifierError("Runtime receipt crossed a stage binding")
    phases = value["phase_elapsed_seconds"]
    if type(phases) is not dict or set(phases) != {"sec", "model", "fit_simulation_sealing", "total"}:
        raise SecFilingGemmaStageVerifierError("Runtime phases are invalid")
    decoded: dict[str, float] = {}
    for name, encoded in phases.items():
        if not isinstance(encoded, str):
            raise SecFilingGemmaStageVerifierError("Runtime seconds must use float.hex")
        try:
            number = float.fromhex(encoded)
        except ValueError as exc:
            raise SecFilingGemmaStageVerifierError("Runtime seconds are invalid") from exc
        if not math.isfinite(number) or number < 0.0 or number.hex() != encoded:
            raise SecFilingGemmaStageVerifierError("Runtime seconds are noncanonical")
        decoded[name] = number
    if (
        decoded["sec"] > MAX_SEC_SECONDS
        or decoded["model"] > MAX_MODEL_SECONDS
        or decoded["fit_simulation_sealing"] > MAX_FIT_SECONDS
        or decoded["total"] > MAX_RUNTIME_SECONDS
        or decoded["total"] + 1e-9
        < decoded["sec"] + decoded["model"] + decoded["fit_simulation_sealing"]
    ):
        raise SecFilingGemmaStageVerifierError("Runtime budget was exceeded")
    model_elapsed_nanoseconds = model_batch_summary.get("model_elapsed_nanoseconds")
    if (
        type(model_elapsed_nanoseconds) is not int
        or model_elapsed_nanoseconds < 0
        or decoded["model"] + 1e-9 < model_elapsed_nanoseconds / 1_000_000_000.0
    ):
        raise SecFilingGemmaStageVerifierError(
            "Runtime model phase is shorter than exact attempt diagnostics"
        )
    if type(value["sec_request_count"]) is not int or not 0 <= value["sec_request_count"] <= MAX_SEC_REQUESTS:
        raise SecFilingGemmaStageVerifierError("SEC request count is invalid")
    if type(value["sec_response_bytes"]) is not int or not 0 <= value["sec_response_bytes"] <= MAX_SEC_BYTES:
        raise SecFilingGemmaStageVerifierError("SEC response bytes are invalid")
    for field in ("paid_api_calls", "pull_attempts", "retries", "repair_attempts"):
        if type(value[field]) is not int or value[field] != 0:
            raise SecFilingGemmaStageVerifierError(f"{field} must be exactly zero")
    if value["estimated_cost_usd_hex"] != 0.0.hex():
        raise SecFilingGemmaStageVerifierError("Estimated cost must be exactly zero")
    return receipt_hash


def validate_registry_request_and_stage_access(
    *,
    registry: Mapping[str, Any],
    registry_external_pin: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
    expected_context: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    corpus_universe_manifest: Mapping[str, Any],
    prerequisite_content_manifest: Mapping[str, Any],
    prerequisite_stage_artifact_sha256: str,
    prerequisite_external_seal_receipt_sha256: str,
    expected_stage_evidence_sha256: str,
) -> dict[str, str]:
    """Replay registry/request/access identities without opening requested bytes."""

    registry_value = _mapping_snapshot(registry, "reveal registry")
    pin = _mapping_snapshot(registry_external_pin, "registry external pin")
    access = _mapping_snapshot(stage_access_manifest, "stage access manifest")
    context = _mapping_snapshot(expected_context, "expected reveal context")
    candidate = _mapping_snapshot(candidate_manifest, "candidate manifest")
    universe = _mapping_snapshot(corpus_universe_manifest, "corpus universe")
    prerequisite_content = _mapping_snapshot(
        prerequisite_content_manifest, "prerequisite content manifest"
    )
    _expect_keys(
        context,
        set(_EXPECTED_REVEAL_CONTEXT_KEYS),
        "expected reveal context",
    )
    prerequisite = context["prerequisite_stage"]
    requested = context["stage"]
    if _PREREQUISITE_TRANSITIONS.get(prerequisite) != requested:
        raise SecFilingGemmaStageVerifierError("Reveal transition is invalid")
    if context["prerequisite_stage_evidence_sha256"] != expected_stage_evidence_sha256:
        raise SecFilingGemmaStageVerifierError("Context binds another stage evidence artifact")
    content_hash = _sha256(
        prerequisite_content.get("content_manifest_sha256"),
        "prerequisite content manifest hash",
    )
    if prerequisite_content.get("artifact_stage") != prerequisite:
        raise SecFilingGemmaStageVerifierError(
            "Prerequisite content manifest belongs to another stage"
        )
    try:
        validate_stage_content_manifest(
            prerequisite_content,
            universe_manifest=universe,
            expected_content_manifest_sha256=content_hash,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Prerequisite content manifest failed exact universe replay"
        ) from exc
    try:
        registry_hash = validate_reveal_registry(registry_value, external_pin=pin)
        request = build_single_candidate_reveal_request(
            registry_value,
            external_pin=pin,
            candidate_manifest=candidate,
            stage=requested,
            stage_access_manifest_sha256=context["stage_access_manifest_sha256"],
            prerequisite_stage_evidence_sha256=expected_stage_evidence_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Registry or reveal request identity failed replay"
        ) from exc
    if request["request_sha256"] != context["request_sha256"]:
        raise SecFilingGemmaStageVerifierError("Context request hash is stale or substituted")
    for key in (
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "stage_access_manifest_sha256",
    ):
        if request[key] != context[key]:
            raise SecFilingGemmaStageVerifierError(f"Context field {key} changed")
    market_access = access.get("market_access")
    budgets = access.get("budgets")
    output = access.get("output")
    sec_plan = access.get("sec_access_plan")
    registry_section = access.get("registry")
    if not all(type(item) is dict for item in (market_access, budgets, output, sec_plan, registry_section)):
        raise SecFilingGemmaStageVerifierError("Stage access sections are invalid")
    sources = market_access["sources"]
    if type(sources) is not list:
        raise SecFilingGemmaStageVerifierError("Stage market sources are invalid")
    artifact_hashes = {item["symbol"]: item["artifact_sha256"] for item in sources}
    window_hashes = {item["symbol"]: item["window_sha256"] for item in sources}
    try:
        access_hash = validate_stage_access_manifest(
            access,
            expected_stage_access_manifest_sha256=context[
                "stage_access_manifest_sha256"
            ],
            prerequisite_stage=prerequisite,
            requested_stage=requested,
            candidate_manifest=candidate,
            expected_candidate_sha256=context["candidate_sha256"],
            expected_candidate_design_sha256=context["candidate_design_sha256"],
            expected_attempt_id=context["attempt_id"],
            expected_stage_verifier_source_sha256=candidate["bindings"][
                "source_hashes"
            ]["stage_verifier"],
            reveal_request=request,
            expected_reveal_request_sha256=context["request_sha256"],
            registry_sha256=context["registry_sha256"],
            registry_tip_sha256=context["registry_tip_sha256"],
            registered_entry_count=registry_section["registered_entry_count"],
            registry_entry_sha256=context["registry_entry_sha256"],
            base_corpus_universe_sha256=candidate["bindings"][
                "corpus_universe_sha256"
            ],
            corpus_universe_manifest=universe,
            prerequisite_content_manifest=prerequisite_content,
            expected_prerequisite_content_manifest_sha256=content_hash,
            prerequisite_stage_artifact_sha256=(
                prerequisite_stage_artifact_sha256
            ),
            prerequisite_external_seal_receipt_sha256=(
                prerequisite_external_seal_receipt_sha256
            ),
            session_calendar_sha256=candidate["bindings"][
                "calendar_sessions_sha256"
            ],
            authorized_documents=sec_plan["documents"],
            market_source_manifest_sha256=market_access[
                "source_manifest_sha256"
            ],
            market_source_artifact_sha256s=artifact_hashes,
            market_source_window_sha256s=window_hashes,
            max_sec_response_bytes=budgets["max_sec_response_bytes"],
            output_namespace=output["namespace"],
        )
    except (TypeError, ValueError) as exc:
        raise SecFilingGemmaStageVerifierError(
            "Requested-stage access plan differs from exact non-authorizing replay"
        ) from exc
    return {
        "registry_sha256": registry_hash,
        "request_sha256": request["request_sha256"],
        "stage_access_manifest_sha256": access_hash,
    }


def validate_trusted_stage_content_pin(
    *,
    stage_access_manifest: Mapping[str, Any],
    trusted_stage_content_pin: Mapping[str, Any] | None,
    trusted_stage_content_authentication: Mapping[str, Any] | None,
    expected_context: Mapping[str, Any],
    expected_stage: str,
) -> dict[str, Any]:
    """Cross-bind stage access to the reveal-store-authenticated pin context."""

    access = _mapping_snapshot(stage_access_manifest, "stage access manifest")
    context = _mapping_snapshot(expected_context, "trusted content expected context")
    if trusted_stage_content_pin is None:
        raise SecFilingGemmaStageVerifierError(
            "A separately persisted trusted stage-content pin is required"
        )
    trusted = _mapping_snapshot(
        trusted_stage_content_pin, "trusted stage-content pin"
    )
    access_pin = access.get("prerequisite_evidence_pin")
    if type(access_pin) is not dict:
        raise SecFilingGemmaStageVerifierError(
            "Stage access lacks its pre-existing prerequisite evidence pin"
        )
    _expect_keys(
        access_pin,
        {
            "stage",
            "content_manifest_sha256",
            "stage_artifact_sha256",
            "external_seal_receipt_sha256",
        },
        "stage-access prerequisite evidence pin",
    )
    _expect_keys(
        trusted,
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
        },
        "trusted stage-content pin",
    )
    if (
        trusted["schema_version"] != TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION
        or trusted["contract_version"] != CONTRACT_VERSION
    ):
        raise SecFilingGemmaStageVerifierError(
            "Trusted stage-content pin schema or contract changed"
        )
    body = {key: trusted[key] for key in trusted if key != "pin_sha256"}
    expected_pin_hash = _sha256(trusted["pin_sha256"], "trusted content pin hash")
    if not hmac.compare_digest(expected_pin_hash, canonical_sha256(body)):
        raise SecFilingGemmaStageVerifierError(
            "Trusted stage-content pin is not canonical"
        )
    if (
        access_pin["stage"] != expected_stage
        or trusted["prerequisite_stage"] != expected_stage
        or trusted["requested_stage"] != context.get("stage")
    ):
        raise SecFilingGemmaStageVerifierError(
            "Trusted stage-content pin crossed a stage"
        )
    context_bindings = {
        "request_sha256": "request_sha256",
        "prerequisite_stage_evidence_sha256": (
            "prerequisite_stage_evidence_sha256"
        ),
        "stage_access_manifest_sha256": "stage_access_manifest_sha256",
        "prerequisite_stage": "prerequisite_stage",
        "attempt_id": "attempt_id",
        "candidate_sha256": "candidate_sha256",
        "candidate_design_sha256": "candidate_design_sha256",
        "registry_entry_sha256": "registry_entry_sha256",
    }
    for pin_key, context_key in context_bindings.items():
        if trusted[pin_key] != context.get(context_key):
            raise SecFilingGemmaStageVerifierError(
                f"Trusted stage-content pin differs from expected {context_key}"
            )
    if trusted["pin_sha256"] != context.get("trusted_stage_content_pin_sha256"):
        raise SecFilingGemmaStageVerifierError(
            "Trusted stage-content pin hash differs from the reveal-store context"
        )
    for key in (
        "content_manifest_sha256",
        "stage_artifact_sha256",
        "external_seal_receipt_sha256",
    ):
        access_value = _sha256(access_pin[key], f"stage-access {key}")
        trusted_value = _sha256(trusted[key], f"trusted content {key}")
        if not hmac.compare_digest(access_value, trusted_value):
            raise SecFilingGemmaStageVerifierError(
                f"Trusted stage-content {key} differs from the pre-existing stage-access pin"
            )
    store_state_hash = _sha256(
        trusted["trusted_store_state_sha256"], "trusted content store-state hash"
    )
    if trusted_stage_content_authentication is None:
        raise SecFilingGemmaStageVerifierError(
            "Reveal-store trusted content authentication receipt is required"
        )
    authentication = _mapping_snapshot(
        trusted_stage_content_authentication,
        "trusted stage-content authentication receipt",
    )
    _expect_keys(
        authentication,
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
        },
        "trusted stage-content authentication receipt",
    )
    if (
        authentication["schema_version"]
        != TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION
        or authentication["contract_version"] != trusted["contract_version"]
        or authentication["authentication_kind"]
        != "reveal_store_current_tip_persisted_trusted_stage_content_pin"
    ):
        raise SecFilingGemmaStageVerifierError(
            "Trusted stage-content authentication semantics changed"
        )
    authentication_body = {
        key: authentication[key]
        for key in authentication
        if key != "authentication_receipt_sha256"
    }
    authentication_hash = _sha256(
        authentication["authentication_receipt_sha256"],
        "trusted content authentication receipt hash",
    )
    if not hmac.compare_digest(authentication_hash, canonical_sha256(authentication_body)):
        raise SecFilingGemmaStageVerifierError(
            "Trusted stage-content authentication receipt is not canonical"
        )
    for field in (
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "prerequisite_stage",
        "requested_stage",
        "trusted_stage_content_pin_sha256",
        "trusted_store_state_sha256",
    ):
        expected_value = (
            trusted["pin_sha256"]
            if field == "trusted_stage_content_pin_sha256"
            else trusted[field]
        )
        if authentication[field] != expected_value:
            raise SecFilingGemmaStageVerifierError(
                f"Trusted content authentication differs from pin field {field}"
            )
    for field in (
        "trusted_current_tip_anchor_sha256",
        "trusted_stage_content_pins_sha256",
    ):
        _sha256(authentication[field], f"trusted content authentication {field}")
    if (
        type(authentication["trusted_current_tip_revision"]) is not int
        or authentication["trusted_current_tip_revision"] < 0
        or authentication_hash
        != context.get("trusted_stage_content_authentication_receipt_sha256")
    ):
        raise SecFilingGemmaStageVerifierError(
            "Trusted content authentication lost its current-tip or context binding"
        )
    return {
        "stage": expected_stage,
        "content_manifest_sha256": trusted["content_manifest_sha256"],
        "stage_artifact_sha256": trusted["stage_artifact_sha256"],
        "external_seal_receipt_sha256": trusted[
            "external_seal_receipt_sha256"
        ],
        "trusted_store_state_sha256": store_state_hash,
        "pin_sha256": expected_pin_hash,
        "authentication_receipt": authentication,
        "authentication_receipt_sha256": authentication_hash,
    }


def _stage_evidence_hash(evidence: Mapping[str, Any]) -> str:
    observed = _sha256(evidence.get("stage_evidence_sha256"), "stage evidence hash")
    body = {key: evidence[key] for key in evidence if key != "stage_evidence_sha256"}
    calculated = canonical_sha256(body)
    if not hmac.compare_digest(observed, calculated):
        raise SecFilingGemmaStageVerifierError("Stage evidence is not canonical")
    return observed


def validate_authenticated_store_context(
    authenticated_store_context: Mapping[str, Any] | None,
    *,
    stage_access_manifest: Mapping[str, Any],
    expected_context: Mapping[str, Any],
    expected_stage: str,
) -> dict[str, Any]:
    """Validate the compact context created inside the locked reveal store."""

    if authenticated_store_context is None:
        raise SecFilingGemmaStageVerifierError(
            "A reveal-store-authenticated verifier context is required"
        )
    store_context = _mapping_snapshot(
        authenticated_store_context,
        "authenticated store verifier context",
    )
    _expect_keys(
        store_context,
        {
            "schema_version",
            "contract_version",
            "context_kind",
            "trusted_stage_content_pin",
            "trusted_stage_content_authentication",
            "parent_consumption_binding",
            "authenticated_store_context_sha256",
        },
        "authenticated store verifier context",
    )
    if (
        store_context["schema_version"]
        != AUTHENTICATED_STORE_VERIFIER_CONTEXT_SCHEMA_VERSION
        or store_context["contract_version"] != CONTRACT_VERSION
        or store_context["context_kind"]
        != "reveal_store_authenticated_preconsumption_context"
    ):
        raise SecFilingGemmaStageVerifierError(
            "Authenticated store verifier context semantics changed"
        )
    body = {
        key: store_context[key]
        for key in store_context
        if key != "authenticated_store_context_sha256"
    }
    store_context_hash = _sha256(
        store_context["authenticated_store_context_sha256"],
        "authenticated store verifier context hash",
    )
    if not hmac.compare_digest(store_context_hash, canonical_sha256(body)):
        raise SecFilingGemmaStageVerifierError(
            "Authenticated store verifier context is not canonical"
        )
    expected = _mapping_snapshot(expected_context, "store verifier expected context")
    if store_context_hash != expected.get("authenticated_store_context_sha256"):
        raise SecFilingGemmaStageVerifierError(
            "Authenticated store context differs from the expected reveal-store context"
        )
    trusted_pin = validate_trusted_stage_content_pin(
        stage_access_manifest=stage_access_manifest,
        trusted_stage_content_pin=store_context["trusted_stage_content_pin"],
        trusted_stage_content_authentication=store_context[
            "trusted_stage_content_authentication"
        ],
        expected_context=expected,
        expected_stage=expected_stage,
    )
    parent_binding = store_context["parent_consumption_binding"]
    if expected_stage == "development":
        if parent_binding is not None or expected.get(
            "parent_consumption_binding_sha256"
        ) is not None:
            raise SecFilingGemmaStageVerifierError(
                "Development prerequisite cannot carry a parent-consumption binding"
            )
    elif expected_stage == "intermediate":
        if type(parent_binding) is not dict:
            raise SecFilingGemmaStageVerifierError(
                "Intermediate prerequisite requires a parent-consumption binding"
            )
        binding_hash = _sha256(
            parent_binding.get("parent_consumption_binding_sha256"),
            "parent-consumption binding hash",
        )
        binding_body = {
            key: parent_binding[key]
            for key in parent_binding
            if key != "parent_consumption_binding_sha256"
        }
        if (
            not hmac.compare_digest(binding_hash, canonical_sha256(binding_body))
            or binding_hash != expected.get("parent_consumption_binding_sha256")
        ):
            raise SecFilingGemmaStageVerifierError(
                "Parent-consumption binding is not canonical or expected"
            )
    else:  # pragma: no cover - guarded by stage contract before this helper
        raise SecFilingGemmaStageVerifierError(
            "Authenticated store context crossed a prerequisite stage"
        )
    return {
        "authenticated_store_context_sha256": store_context_hash,
        "trusted_stage_content_pin": trusted_pin,
        "parent_consumption_binding": parent_binding,
    }


def _validate_parent_consumed_stage_output_receipt(
    receipt: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    *,
    parent_consumption: Mapping[str, Any],
    authenticated_tip: Mapping[str, Any],
    current_stage_evidence: Mapping[str, Any],
    current_stage_evidence_sha256: str,
    current_candidate_sha256: str,
    declared_parent_stage_evidence_sha256: str,
) -> str:
    value = _mapping_snapshot(receipt, "parent consumed-stage output receipt")
    _expect_keys(
        value,
        set(_CONSUMED_STAGE_OUTPUT_RECEIPT_KEYS),
        "parent consumed-stage output receipt",
    )
    receipt_hash = _sha256(
        value["output_receipt_sha256"],
        "parent consumed-stage output receipt hash",
    )
    receipt_body = {
        key: value[key] for key in value if key != "output_receipt_sha256"
    }
    if (
        value["schema_version"] != CONSUMED_STAGE_OUTPUT_RECEIPT_SCHEMA_VERSION
        or value["contract_version"] != CONTRACT_VERSION
        or value["receipt_kind"]
        != "first_output_for_exact_consumed_stage_grant"
        or value["output_kind"] != "next_stage_evidence"
        or value["output_stage_evidence_schema_version"]
        != STAGE_EVIDENCE_SCHEMA_VERSION
        or value["output_stage_evidence_component_id"]
        != STAGE_EVIDENCE_OUTPUT_COMPONENT_ID
        or value["output_stage_evidence_relative_path"]
        != STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH
        or value["output_stage_evidence_recomputed_by_store"] is not True
        or value["fresh_stage_evidence_provenance_claimed"] is not False
        or value["cross_stage_output_permitted"] is not False
        or value["grant_reuse_for_different_output_permitted"] is not False
        or not hmac.compare_digest(receipt_hash, canonical_sha256(receipt_body))
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent consumed-stage output receipt is not canonical"
        )
    receipts = _mapping_snapshot(
        receipt_map,
        "authenticated consumed-stage output receipt map",
    )
    for request_hash in receipts:
        _sha256(request_hash, "authenticated output-receipt map request hash")
    if (
        canonical_sha256(receipts)
        != authenticated_tip["consumed_stage_output_receipts_sha256"]
        or receipts.get(parent_consumption["request_sha256"]) != value
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent output receipt is not an exact member of the authenticated tip"
        )
    sec_claims = _mapping_snapshot(
        authenticated_tip["stage_sec_execution_claims"],
        "authenticated parent SEC execution claims",
    )
    sec_readers = _mapping_snapshot(
        authenticated_tip["stage_sec_reader_receipts"],
        "authenticated parent SEC reader receipts",
    )
    if (
        canonical_sha256(sec_claims)
        != authenticated_tip["stage_sec_execution_claims_sha256"]
        or canonical_sha256(sec_readers)
        != authenticated_tip["stage_sec_reader_receipts_sha256"]
    ):
        raise SecFilingGemmaStageVerifierError(
            "Authenticated parent SEC ancestry maps are inconsistent"
        )
    parent_request_hash = parent_consumption["request_sha256"]
    sec_claim = _mapping_snapshot(
        sec_claims.get(parent_request_hash),
        "authenticated parent SEC execution claim",
    )
    sec_reader = _mapping_snapshot(
        sec_readers.get(parent_request_hash),
        "authenticated parent SEC reader receipt",
    )
    claim_hash = _sha256(
        sec_claim.get("claim_sha256"),
        "authenticated parent SEC execution claim hash",
    )
    reader_hash = _sha256(
        sec_reader.get("receipt_sha256"),
        "authenticated parent SEC reader receipt hash",
    )
    if (
        sec_claim.get("request_sha256") != parent_request_hash
        or sec_reader.get("request_sha256") != parent_request_hash
        or sec_reader.get("claim_sha256") != claim_hash
        or value["sec_execution_claim_sha256"] != claim_hash
        or value["sec_reader_receipt_sha256"] != reader_hash
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent output receipt crossed its SEC claim or reader ancestry"
        )
    evidence = _mapping_snapshot(
        current_stage_evidence,
        "current stage evidence for output receipt",
    )
    try:
        evidence_bytes = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - detached above
        raise SecFilingGemmaStageVerifierError(
            "Current stage evidence is not canonical finite JSON"
        ) from exc
    expected = {
        "consumption_entry_sha256": parent_consumption["entry_sha256"],
        "consumption_entry_sequence": parent_consumption["sequence"],
        "request_sha256": parent_consumption["request_sha256"],
        "attempt_id": parent_consumption["attempt_id"],
        "candidate_sha256": parent_consumption["candidate_sha256"],
        "registry_entry_sha256": parent_consumption["registry_entry_sha256"],
        "input_prerequisite_stage": parent_consumption["prerequisite_stage"],
        "output_stage": parent_consumption["stage"],
        "input_stage_evidence_sha256": parent_consumption[
            "prerequisite_stage_evidence_sha256"
        ],
        "stage_access_manifest_sha256": parent_consumption[
            "stage_access_manifest_sha256"
        ],
        "authorization_bundle_sha256": parent_consumption[
            "authorization_bundle_sha256"
        ],
        "authorization_grant_sha256": parent_consumption[
            "authorization_grant_sha256"
        ],
        "sec_execution_claim_sha256": claim_hash,
        "sec_reader_receipt_sha256": reader_hash,
        "grant_store_state_sha256": authenticated_tip["store_state_sha256"],
        "grant_consumption_ledger_sha256": authenticated_tip[
            "consumption_ledger_sha256"
        ],
        "grant_consumption_ledger_tip_sha256": authenticated_tip[
            "consumption_ledger_tip_sha256"
        ],
        "output_stage_evidence_schema_version": evidence["schema_version"],
        "output_stage_evidence_component_id": STAGE_EVIDENCE_OUTPUT_COMPONENT_ID,
        "output_stage_evidence_relative_path": STAGE_EVIDENCE_OUTPUT_RELATIVE_PATH,
        "output_stage_evidence_sha256": current_stage_evidence_sha256,
        "output_stage_evidence_document_sha256": hashlib.sha256(
            evidence_bytes
        ).hexdigest(),
        "output_stage_evidence_canonical_byte_count": len(evidence_bytes),
        "output_stage_evidence_prerequisite_stage": evidence[
            "prerequisite_stage"
        ],
        "output_parent_stage_evidence_sha256": (
            declared_parent_stage_evidence_sha256
        ),
        "output_candidate_sha256": current_candidate_sha256,
    }
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise SecFilingGemmaStageVerifierError(
                f"Parent output receipt crossed {field}"
            )
    if (
        value["output_namespace"]
        != f"aapl-sec-gemma-{parent_consumption['attempt_id']}-intermediate"
        or type(value["output_stage_evidence_canonical_byte_count"]) is not int
        or value["output_stage_evidence_canonical_byte_count"] < 2
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent output receipt namespace or byte count is invalid"
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
        _sha256(value[field], f"parent output receipt {field}")
    return receipt_hash


def validate_parent_stage_lineage(
    parent_stage_lineage: Mapping[str, Any] | None,
    *,
    prerequisite_stage: str,
    declared_parent_stage_evidence_sha256: str | None,
    current_candidate_sha256: str,
    current_expected_context: Mapping[str, Any],
    current_trusted_stage_content_pin: Mapping[str, Any],
    parent_consumption_binding: Mapping[str, Any] | None,
    current_stage_evidence: Mapping[str, Any] | None = None,
    current_stage_evidence_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Replay lineage bound to the exact consumed parent entry and grant."""

    if prerequisite_stage == "development":
        if (
            parent_stage_lineage is not None
            or declared_parent_stage_evidence_sha256 is not None
            or parent_consumption_binding is not None
        ):
            raise SecFilingGemmaStageVerifierError(
                "Development cannot claim a parent stage or parent consumption"
            )
        return None
    if prerequisite_stage != "intermediate":
        raise SecFilingGemmaStageVerifierError(
            "Only the intermediate prerequisite can carry parent lineage"
        )
    if parent_consumption_binding is None:
        raise SecFilingGemmaStageVerifierError(
            "Intermediate evidence requires its reveal-store parent-consumption binding"
        )
    binding = _mapping_snapshot(
        parent_consumption_binding,
        "parent-consumption binding",
    )
    _expect_keys(binding, set(_PARENT_BINDING_KEYS), "parent-consumption binding")
    if (
        binding["schema_version"] != PARENT_CONSUMPTION_BINDING_SCHEMA_VERSION
        or binding["contract_version"] != CONTRACT_VERSION
        or binding["binding_kind"]
        != "exact_prior_intermediate_consumption_and_grant"
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent-consumption binding semantics changed"
        )
    binding_hash = _sha256(
        binding["parent_consumption_binding_sha256"],
        "parent-consumption binding hash",
    )
    binding_body = {
        key: binding[key]
        for key in binding
        if key != "parent_consumption_binding_sha256"
    }
    current_context = _mapping_snapshot(
        current_expected_context,
        "current expected reveal context",
    )
    _expect_keys(
        current_context,
        set(_EXPECTED_REVEAL_CONTEXT_KEYS),
        "current expected reveal context",
    )
    if (
        not hmac.compare_digest(binding_hash, canonical_sha256(binding_body))
        or current_context["parent_consumption_binding_sha256"] != binding_hash
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent-consumption binding is not canonical or current-request bound"
        )

    child = _mapping_snapshot(binding["child_request"], "bound child request")
    parent = _mapping_snapshot(
        binding["parent_consumption"],
        "bound parent consumption",
    )
    tip = _mapping_snapshot(
        binding["authenticated_preconsumption_tip"],
        "bound authenticated preconsumption tip",
    )
    _expect_keys(child, set(_PARENT_BINDING_CHILD_KEYS), "bound child request")
    _expect_keys(parent, set(_PARENT_BINDING_PARENT_KEYS), "bound parent consumption")
    _expect_keys(tip, set(_PARENT_BINDING_TIP_KEYS), "bound preconsumption tip")
    expected_child = {
        key: current_context[key] for key in _PARENT_BINDING_CHILD_KEYS
    }
    if child != expected_child:
        raise SecFilingGemmaStageVerifierError(
            "Parent-consumption binding belongs to another final request"
        )
    if (
        child["stage"] != "final"
        or child["prerequisite_stage"] != "intermediate"
        or child["candidate_sha256"] != current_candidate_sha256
    ):
        raise SecFilingGemmaStageVerifierError(
            "Bound child request crossed its stage or candidate"
        )

    for field in (
        "store_state_sha256",
        "state_snapshot_bytes_sha256",
        "consumption_ledger_sha256",
        "consumption_ledger_tip_sha256",
        "current_tip_anchor_sha256",
        "consumed_stage_output_receipts_sha256",
    ):
        _sha256(tip[field], f"bound preconsumption tip {field}")
    if (
        type(tip["state_snapshot_byte_count"]) is not int
        or tip["state_snapshot_byte_count"] < 1
        or type(tip["consumed_request_count"]) is not int
        or tip["consumed_request_count"] < 1
        or type(tip["current_tip_revision"]) is not int
        or tip["current_tip_revision"] < 0
        or type(parent["sequence"]) is not int
        or parent["sequence"] < 1
        or parent["sequence"] != tip["consumed_request_count"]
    ):
        raise SecFilingGemmaStageVerifierError(
            "Bound parent sequence or preconsumption tip count is invalid"
        )
    current_pin = _mapping_snapshot(
        current_trusted_stage_content_pin,
        "current trusted stage-content summary",
    )
    current_authentication = _mapping_snapshot(
        current_pin.get("authentication_receipt"),
        "current trusted content authentication",
    )
    if (
        tip["store_state_sha256"] != current_pin.get("trusted_store_state_sha256")
        or tip["current_tip_anchor_sha256"]
        != current_authentication.get("trusted_current_tip_anchor_sha256")
        or tip["current_tip_revision"]
        != current_authentication.get("trusted_current_tip_revision")
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent binding differs from the authenticated current store tip"
        )

    for field in (
        "entry_sha256",
        "request_sha256",
        "prerequisite_stage_evidence_sha256",
        "stage_access_manifest_sha256",
        "expected_context_sha256",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
        "prerequisite_validation_result_sha256",
        "semantic_receipt_sha256",
        "audit_receipt_sha256",
        "trusted_stage_content_pin_sha256",
        "authorization_bundle_sha256",
        "authorization_grant_sha256",
        "store_pin_sha256",
        "consumed_stage_output_receipt_sha256",
    ):
        _sha256(parent[field], f"bound parent consumption {field}")
    # The reveal store derived the entry/bundle/grant/tip hashes from its locked
    # authenticated files before constructing this context. This verifier
    # cross-binds those store-origin identities to the recursive evidence; it
    # does not claim to have independently loaded the raw bundle or store files.
    if (
        parent["stage"] != "intermediate"
        or parent["prerequisite_stage"] != "development"
        or parent["entry_sha256"] != tip["consumption_ledger_tip_sha256"]
        or parent["candidate_sha256"] != current_candidate_sha256
    ):
        raise SecFilingGemmaStageVerifierError(
            "Bound parent is not the exact intermediate consumption-ledger tip"
        )
    for field in (
        "attempt_id",
        "candidate_sha256",
        "candidate_design_sha256",
        "registry_entry_sha256",
        "registry_sha256",
        "registry_tip_sha256",
    ):
        if parent[field] != child[field]:
            raise SecFilingGemmaStageVerifierError(
                f"Parent consumption crossed child identity {field}"
            )

    if current_stage_evidence is None or current_stage_evidence_sha256 is None:
        raise SecFilingGemmaStageVerifierError(
            "Intermediate evidence requires its exact consumed-stage output document"
        )
    current_evidence_hash = _sha256(
        current_stage_evidence_sha256,
        "current stage evidence hash for parent output receipt",
    )
    declared_hash = _sha256(
        declared_parent_stage_evidence_sha256,
        "parent stage evidence hash",
    )
    output_receipt_hash = _validate_parent_consumed_stage_output_receipt(
        parent["consumed_stage_output_receipt"],
        tip["consumed_stage_output_receipts"],
        parent_consumption=parent,
        authenticated_tip=tip,
        current_stage_evidence=current_stage_evidence,
        current_stage_evidence_sha256=current_evidence_hash,
        current_candidate_sha256=current_candidate_sha256,
        declared_parent_stage_evidence_sha256=declared_hash,
    )
    if output_receipt_hash != parent["consumed_stage_output_receipt_sha256"]:
        raise SecFilingGemmaStageVerifierError(
            "Parent consumed-stage output receipt hash changed"
        )

    if parent_stage_lineage is None:
        raise SecFilingGemmaStageVerifierError(
            "Intermediate evidence requires its complete parent lineage"
        )
    lineage = _mapping_snapshot(parent_stage_lineage, "parent stage lineage")
    _expect_keys(
        lineage,
        {
            "evidence",
            "stage_access_manifest",
            "expected_context",
        },
        "parent stage lineage",
    )
    parent_evidence = _mapping_snapshot(lineage["evidence"], "parent stage evidence")
    if parent_evidence.get("prerequisite_stage") != "development":
        raise SecFilingGemmaStageVerifierError(
            "Intermediate parent evidence must be the development prerequisite"
        )
    parent_hash = _stage_evidence_hash(parent_evidence)
    if not hmac.compare_digest(parent_hash, declared_hash):
        raise SecFilingGemmaStageVerifierError(
            "Parent evidence bytes differ from the declared parent hash"
        )
    if parent_hash != parent["prerequisite_stage_evidence_sha256"]:
        raise SecFilingGemmaStageVerifierError(
            "Parent evidence differs from the prior consumed request"
        )
    parent_candidate = parent_evidence.get("candidate_manifest")
    parent_candidate_hash = (
        parent_candidate.get("candidate_sha256")
        if type(parent_candidate) is dict
        else None
    )
    if parent_candidate_hash != current_candidate_sha256:
        raise SecFilingGemmaStageVerifierError(
            "Parent evidence belongs to another candidate"
        )

    parent_access = _mapping_snapshot(
        lineage["stage_access_manifest"],
        "parent stage-access manifest",
    )
    parent_context = _mapping_snapshot(
        lineage["expected_context"],
        "parent expected reveal context",
    )
    _expect_keys(
        parent_context,
        set(_EXPECTED_REVEAL_CONTEXT_KEYS),
        "parent expected reveal context",
    )
    parent_audit = _mapping_snapshot(
        parent["audit_receipt"],
        "stored parent stage audit receipt",
    )
    parent_pin = _mapping_snapshot(
        parent["trusted_stage_content_pin"],
        "stored parent trusted content pin",
    )
    parent_authentication = _mapping_snapshot(
        parent_audit.get("trusted_stage_content_authentication"),
        "stored parent trusted content authentication",
    )
    parent_store_context_body = {
        "schema_version": AUTHENTICATED_STORE_VERIFIER_CONTEXT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "context_kind": "reveal_store_authenticated_preconsumption_context",
        "trusted_stage_content_pin": parent_pin,
        "trusted_stage_content_authentication": parent_authentication,
        "parent_consumption_binding": None,
    }
    parent_store_context = {
        **parent_store_context_body,
        "authenticated_store_context_sha256": canonical_sha256(
            parent_store_context_body
        ),
    }
    expected_parent_context = {
        "prerequisite_stage": parent["prerequisite_stage"],
        "prerequisite_stage_evidence_sha256": parent[
            "prerequisite_stage_evidence_sha256"
        ],
        "attempt_id": parent["attempt_id"],
        "candidate_sha256": parent["candidate_sha256"],
        "candidate_design_sha256": parent["candidate_design_sha256"],
        "registry_entry_sha256": parent["registry_entry_sha256"],
        "request_sha256": parent["request_sha256"],
        "stage": parent["stage"],
        "stage_access_manifest_sha256": parent[
            "stage_access_manifest_sha256"
        ],
        "registry_sha256": parent["registry_sha256"],
        "registry_tip_sha256": parent["registry_tip_sha256"],
        "trusted_stage_content_pin_sha256": parent[
            "trusted_stage_content_pin_sha256"
        ],
        "trusted_stage_content_authentication_receipt_sha256": (
            parent_authentication.get("authentication_receipt_sha256")
        ),
        "parent_consumption_binding_sha256": None,
        "authenticated_store_context_sha256": parent_store_context[
            "authenticated_store_context_sha256"
        ],
    }
    if (
        parent_context != expected_parent_context
        or canonical_sha256(parent_context) != parent["expected_context_sha256"]
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent expected context differs from the prior consumed request"
        )
    parent_content_pin = validate_trusted_stage_content_pin(
        stage_access_manifest=parent_access,
        trusted_stage_content_pin=parent_pin,
        trusted_stage_content_authentication=parent_authentication,
        expected_context=parent_context,
        expected_stage="development",
    )
    if parent_content_pin["pin_sha256"] != parent[
        "trusted_stage_content_pin_sha256"
    ]:
        raise SecFilingGemmaStageVerifierError(
            "Parent trusted content pin differs from the consumed parent binding"
        )
    computed_receipt = audit_stage_evidence(
        parent_evidence,
        parent_access,
        parent_context,
        authenticated_store_context=parent_store_context,
    )
    if parent_audit != computed_receipt:
        raise SecFilingGemmaStageVerifierError(
            "Stored parent audit receipt differs from authoritative parent replay"
        )
    if (
        computed_receipt.get("stage_evidence_sha256") != parent_hash
        or computed_receipt.get("candidate_sha256") != current_candidate_sha256
        or computed_receipt.get("prerequisite_stage") != "development"
        or computed_receipt.get("requested_stage") != "intermediate"
        or computed_receipt.get("trusted_stage_content_pin_sha256")
        != parent["trusted_stage_content_pin_sha256"]
        or computed_receipt.get("parent_consumption_binding_sha256") is not None
        or computed_receipt.get("authorizes_outcome_access") is not False
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent audit receipt crossed an evidence, candidate, or stage binding"
        )
    if (
        computed_receipt["audit_receipt_sha256"]
        != parent["audit_receipt_sha256"]
        or canonical_sha256(computed_receipt)
        != parent["semantic_receipt_sha256"]
    ):
        raise SecFilingGemmaStageVerifierError(
            "Parent audit receipt differs from the prior semantic validation"
        )
    parent_content = parent_evidence.get("content_replays_by_stage")
    if type(parent_content) is not dict or set(parent_content) != {"development"}:
        raise SecFilingGemmaStageVerifierError(
            "Parent evidence lacks exact development content replay"
        )
    return {
        "parent_stage_evidence_sha256": parent_hash,
        "parent_audit_receipt_sha256": computed_receipt[
            "audit_receipt_sha256"
        ],
        "parent_trusted_stage_content_pin_sha256": parent_content_pin[
            "pin_sha256"
        ],
        "parent_consumption_binding_sha256": binding_hash,
        "parent_consumption_entry_sha256": parent["entry_sha256"],
        "parent_prerequisite_validation_result_sha256": parent[
            "prerequisite_validation_result_sha256"
        ],
        "parent_semantic_receipt_sha256": parent["semantic_receipt_sha256"],
        "parent_authorization_bundle_sha256": parent[
            "authorization_bundle_sha256"
        ],
        "parent_authorization_grant_sha256": parent[
            "authorization_grant_sha256"
        ],
        "parent_store_pin_sha256": parent["store_pin_sha256"],
        "parent_consumed_stage_output_receipt_sha256": output_receipt_hash,
        "parent_output_stage_evidence_document_sha256": parent[
            "consumed_stage_output_receipt"
        ]["output_stage_evidence_document_sha256"],
        "trusted_stage_content_pin": {
            "stage": "development",
            "content_manifest_sha256": parent_content_pin[
                "content_manifest_sha256"
            ],
            "stage_artifact_sha256": parent_content_pin[
                "stage_artifact_sha256"
            ],
            "external_seal_receipt_sha256": parent_content_pin[
                "external_seal_receipt_sha256"
            ],
            "trusted_store_state_sha256": parent_content_pin[
                "trusted_store_state_sha256"
            ],
            "pin_sha256": parent_content_pin["pin_sha256"],
        },
        "content_replays_by_stage": parent_content,
    }


def audit_stage_evidence(
    evidence: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
    expected_context: Mapping[str, Any],
    *,
    authenticated_store_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay all currently provable evidence and return a blocked audit receipt."""

    # Use one shared budget across all independent caller objects.  The first
    # traversal rejects an already-oversized bundle before any copy; the second
    # traversal rechecks the same bounds while detaching, closing mutation
    # windows between check and copy.
    caller_bundle = {
        "evidence": evidence,
        "stage_access_manifest": stage_access_manifest,
        "expected_context": expected_context,
    }
    _preflight_plain_json(caller_bundle, "stage audit caller bundle")
    detached_bundle, _bundle_totals = _bounded_plain_json_copy(
        caller_bundle, "stage audit caller bundle"
    )
    if type(detached_bundle) is not dict:  # pragma: no cover - fixed wrapper
        raise SecFilingGemmaStageVerifierError(
            "Stage audit caller bundle could not be detached"
        )
    value = _mapping_snapshot(detached_bundle["evidence"], "stage evidence")
    stage_access_manifest = detached_bundle["stage_access_manifest"]
    expected_context = detached_bundle["expected_context"]
    _preflight_plain_json(
        authenticated_store_context,
        "authenticated store verifier context",
    )
    authenticated_store_context, _trusted_totals = _bounded_plain_json_copy(
        authenticated_store_context,
        "authenticated store verifier context",
    )
    _expect_keys(
        value,
        STAGE_EVIDENCE_KEYS,
        "stage evidence",
    )
    if value["schema_version"] != STAGE_EVIDENCE_SCHEMA_VERSION:
        raise SecFilingGemmaStageVerifierError("Stage evidence schema changed")
    prerequisite = value["prerequisite_stage"]
    if prerequisite not in _PREREQUISITE_TRANSITIONS:
        raise SecFilingGemmaStageVerifierError("Only prerequisite stages are auditable")
    evidence_hash = _stage_evidence_hash(value)
    candidate = value["candidate_manifest"]
    candidate_hash = candidate.get("candidate_sha256") if type(candidate) is dict else None
    _sha256(candidate_hash, "candidate hash")
    authenticated_context = validate_authenticated_store_context(
        authenticated_store_context,
        stage_access_manifest=stage_access_manifest,
        expected_context=expected_context,
        expected_stage=prerequisite,
    )
    parent_lineage = validate_parent_stage_lineage(
        value["parent_stage_lineage"],
        prerequisite_stage=prerequisite,
        declared_parent_stage_evidence_sha256=value[
            "parent_stage_evidence_sha256"
        ],
        current_candidate_sha256=candidate_hash,
        current_expected_context=expected_context,
        current_trusted_stage_content_pin=authenticated_context[
            "trusted_stage_content_pin"
        ],
        parent_consumption_binding=authenticated_context[
            "parent_consumption_binding"
        ],
        current_stage_evidence=value,
        current_stage_evidence_sha256=evidence_hash,
    )
    current_content_pin = authenticated_context["trusted_stage_content_pin"]
    identities = validate_candidate_source_bytes(
        contract_manifest=value["contract_manifest"],
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate_hash,
        source_bytes_base64_by_role=value["source_bytes_base64_by_role"],
    )
    source_identity = validate_candidate_source_role_audit(
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate_hash,
        source_bytes_base64_by_role=value["source_bytes_base64_by_role"],
    )
    runtime_source_identity = validate_candidate_runtime_source_audit(
        candidate_manifest=candidate,
        expected_candidate_sha256=candidate_hash,
    )
    calendar = validate_calendar_and_universe_snapshot(
        calendar_evidence_manifest=value["calendar_evidence_manifest"],
        calendar_source_bytes_base64_by_name=value[
            "calendar_source_bytes_base64_by_name"
        ],
        corpus_universe_manifest=value["corpus_universe_manifest"],
        candidate_manifest=candidate,
    )
    catalog_replay = validate_detached_catalog_evidence(
        value["catalog_replay"],
        candidate_manifest=candidate,
        corpus_universe_manifest=value["corpus_universe_manifest"],
    )
    if catalog_replay["corpus_universe_sha256"] != calendar[
        "corpus_universe_sha256"
    ]:
        raise SecFilingGemmaStageVerifierError(
            "Detached catalogue and structural universe identities differ"
        )
    stages = STAGE_ORDER[: STAGE_ORDER.index(prerequisite) + 1]
    batches = value["model_batches_by_stage"]
    markets = value["market_replays_by_stage"]
    content_replays = value["content_replays_by_stage"]
    if type(batches) is not dict or set(batches) != set(stages):
        raise SecFilingGemmaStageVerifierError("Model batch stage coverage is incomplete")
    if type(markets) is not dict or set(markets) != set(stages):
        raise SecFilingGemmaStageVerifierError("Market replay stage coverage is incomplete")
    if type(content_replays) is not dict or set(content_replays) != set(stages):
        raise SecFilingGemmaStageVerifierError(
            "Detached content replay stage coverage is incomplete"
        )
    universe_records = value["corpus_universe_manifest"]["records"]
    model_summaries: dict[str, Any] = {}
    market_summaries: dict[str, Any] = {}
    content_summaries: dict[str, Any] = {}
    for stage in stages:
        if stage == prerequisite:
            stage_content_pin = current_content_pin
        else:
            if parent_lineage is None:
                raise SecFilingGemmaStageVerifierError(
                    "Earlier-stage content lacks replayed parent lineage"
                )
            parent_receipt_content = parent_lineage[
                "trusted_stage_content_pin"
            ]
            if parent_receipt_content.get("stage") != stage:
                raise SecFilingGemmaStageVerifierError(
                    "Parent trusted content pin crossed an earlier stage"
                )
            stage_content_pin = parent_receipt_content
        content_summaries[stage] = validate_detached_stage_content_evidence(
            content_replays[stage],
            expected_stage=stage,
            candidate_manifest=candidate,
            corpus_universe_manifest=value["corpus_universe_manifest"],
            externally_pinned_content_manifest_sha256=stage_content_pin[
                "content_manifest_sha256"
            ],
            externally_pinned_stage_artifact_sha256=stage_content_pin[
                "stage_artifact_sha256"
            ],
            maximum_total_document_bytes=MAX_STAGE_CONTENT_BYTES,
        )
        if content_summaries[stage]["artifact_stage"] != stage:
            raise SecFilingGemmaStageVerifierError(
                "Detached content replay crossed a stage slot"
            )
        expected_accessions = [
            record["accession_number"]
            for record in universe_records
            if record["artifact_stage"] == stage
        ]
        expected_accessions.sort(
            key=lambda accession: next(
                (
                    record["availability_session"],
                    record["accession_number"],
                )
                for record in universe_records
                if record["accession_number"] == accession
            )
        )
        model_summaries[stage] = validate_model_attempt_batch(
            batches[stage],
            candidate_manifest=candidate,
            expected_accessions=expected_accessions,
        )
        if model_summaries[stage].get("stage") != stage:
            raise SecFilingGemmaStageVerifierError(
                "Model batch replay crossed a stage slot"
            )
        market_summaries[stage] = validate_market_snapshot_stage_replay(markets[stage])
        if market_summaries[stage]["artifact_stage"] != stage:
            raise SecFilingGemmaStageVerifierError("Market replay crossed a stage")
    if parent_lineage is not None:
        parent_content_replays = parent_lineage["content_replays_by_stage"]
        for stage in stages[:-1]:
            if content_replays[stage] != parent_content_replays.get(stage):
                raise SecFilingGemmaStageVerifierError(
                    "Earlier-stage content differs from the replayed parent evidence"
                )
    prerequisite_content = value["prerequisite_content_manifest"]
    if (
        type(prerequisite_content) is not dict
        or prerequisite_content
        != content_replays[prerequisite]["content_manifest"]
        or prerequisite_content.get("content_manifest_sha256")
        != content_summaries[prerequisite]["content_manifest_sha256"]
    ):
        raise SecFilingGemmaStageVerifierError(
            "Prerequisite content manifest differs from detached byte replay"
        )
    prediction = validate_prediction_artifact_replay(
        value["prediction_replay"],
        candidate_sha256=candidate_hash,
        corpus_universe_sha256=calendar["corpus_universe_sha256"],
    )
    if prediction["stage"] != prerequisite:
        raise SecFilingGemmaStageVerifierError("Prediction artifact stage is not prerequisite")
    learner = validate_learner_refit_replays(
        value["learner_replays"],
        candidate_sha256=candidate_hash,
        prediction_prefix=prediction["prefix"],
    )
    prerequisite_market = markets[prerequisite]["market_stage_manifest"]
    scores = validate_raw_scores_gates_and_ranking(
        value["score_replay"],
        prediction_prefix=prediction["prefix"],
        market_stage=prerequisite_market,
        prerequisite_stage=prerequisite,
    )
    runtime_hash = validate_stage_runtime_receipt(
        value["stage_runtime_receipt"],
        prerequisite_stage=prerequisite,
        candidate_sha256=candidate_hash,
        model_batch_summary=model_summaries[prerequisite],
    )
    access = validate_registry_request_and_stage_access(
        registry=value["reveal_registry"],
        registry_external_pin=value["registry_external_pin"],
        stage_access_manifest=stage_access_manifest,
        expected_context=expected_context,
        candidate_manifest=candidate,
        corpus_universe_manifest=value["corpus_universe_manifest"],
        prerequisite_content_manifest=value["prerequisite_content_manifest"],
        prerequisite_stage_artifact_sha256=current_content_pin[
            "stage_artifact_sha256"
        ],
        prerequisite_external_seal_receipt_sha256=current_content_pin[
            "external_seal_receipt_sha256"
        ],
        expected_stage_evidence_sha256=evidence_hash,
    )

    replayed = {
        "candidate_identity",
        "gate_replay",
        "ledger_replay",
        "no_leverage",
        "registry_identity",
        "request_identity",
        "stage_access_manifest_replay",
        "stage_identity",
    }
    status = {
        check: (
            {"status": "blocked", "reason": _BLOCKING_GAPS[check]}
            if check in _BLOCKING_GAPS
            else {"status": "replayed" if check in replayed else "blocked", "reason": None}
        )
        for check in REQUIRED_STAGE_VERIFIER_CHECKS
    }
    if prerequisite == "intermediate":
        status["stage_identity"] = {
            "status": "blocked",
            "reason": _INTERMEDIATE_STAGE_IDENTITY_GAP,
        }
    # Checks not backed by one authoritative component remain blocked even if
    # their lower-level ingredients were individually inspected.
    for check in REQUIRED_STAGE_VERIFIER_CHECKS:
        if status[check]["status"] == "blocked" and status[check]["reason"] is None:
            status[check]["reason"] = (
                "no complete authoritative end-to-end component currently proves this check"
            )
    corpus_replay_trust_boundary = {
        "catalog": {
            "authorizing": catalog_replay["authorizing"],
            "fresh_network_provenance_verified": catalog_replay[
                "fresh_network_provenance_verified"
            ],
            "network_receipt_claims_replayed_not_observed": catalog_replay[
                "network_receipt_claims_replayed_not_observed"
            ],
        },
        "content_by_stage": {
            stage: {
                "authorizing": summary["authorizing"],
                "fresh_network_provenance_verified": summary[
                    "fresh_network_provenance_verified"
                ],
                "network_receipt_claims_replayed_not_observed": summary[
                    "network_receipt_claims_replayed_not_observed"
                ],
            }
            for stage, summary in content_summaries.items()
        },
    }
    corpus_replay_receipt_schema_versions = {
        "catalog": catalog_replay["schema_version"],
        "content_by_stage": {
            stage: summary["schema_version"]
            for stage, summary in content_summaries.items()
        },
    }
    source_runtime_trust_boundary = {
        "candidate_pinned_detached_source_bytes_verified": True,
        "current_files_at_loaded_module_paths_verified": True,
        "executing_validator_source_bytes_attested": False,
        "runtime_module_paths_verified": True,
        "caller_supplied_runtime_paths_or_bytes_accepted": False,
        "role_ownership_complete": runtime_source_identity["complete"],
        "identity_scope": "current_files_at_already_loaded_module_paths",
    }
    trusted_content_pin_boundary = {
        "stage_access_pin_cross_bound": True,
        "store_state_and_tip_authenticated_by_reveal_store": True,
        "pin_membership_authenticated_by_reveal_store": True,
        "pin_and_stage_access_cross_bound_by_verifier": True,
        "parent_consumption_cross_bound_by_verifier": parent_lineage is not None,
        "parent_first_output_receipt_cross_bound_by_verifier": (
            parent_lineage is not None
        ),
        "trusted_store_state_authenticated_by_verifier": False,
        "store_files_independently_loaded_by_verifier": False,
        "same_directory_is_external_trust_domain": False,
        "coordinated_state_and_tip_replacement_resistant": False,
        "authorizing": False,
    }
    component_hashes = {
        "candidate_and_sources": canonical_sha256(identities),
        "source_identity_audit": source_identity[
            "source_identity_receipt_sha256"
        ],
        "runtime_source_identity_audit": runtime_source_identity[
            "source_identity_receipt_sha256"
        ],
        "source_runtime_trust_boundary": canonical_sha256(
            source_runtime_trust_boundary
        ),
        "trusted_stage_content_pin": current_content_pin["pin_sha256"],
        "trusted_stage_content_authentication": current_content_pin[
            "authentication_receipt_sha256"
        ],
        "authenticated_store_context": authenticated_context[
            "authenticated_store_context_sha256"
        ],
        "trusted_content_pin_boundary": canonical_sha256(
            trusted_content_pin_boundary
        ),
        "parent_stage_lineage": canonical_sha256(parent_lineage),
        "parent_consumption_binding": (
            canonical_sha256(None)
            if parent_lineage is None
            else parent_lineage["parent_consumption_binding_sha256"]
        ),
        "parent_consumed_stage_output": (
            canonical_sha256(None)
            if parent_lineage is None
            else parent_lineage[
                "parent_consumed_stage_output_receipt_sha256"
            ]
        ),
        "calendar_and_universe": canonical_sha256(calendar),
        "catalog_replay": catalog_replay["replay_validation_sha256"],
        "catalog_request_receipts": catalog_replay[
            "request_receipts_sha256"
        ],
        "content_replays": canonical_sha256(
            {
                stage: summary["replay_validation_sha256"]
                for stage, summary in content_summaries.items()
            }
        ),
        "content_request_receipts": canonical_sha256(
            {
                stage: summary["request_receipts_sha256"]
                for stage, summary in content_summaries.items()
            }
        ),
        "corpus_replay_trust_boundary": canonical_sha256(
            corpus_replay_trust_boundary
        ),
        "corpus_replay_receipt_schemas": canonical_sha256(
            corpus_replay_receipt_schema_versions
        ),
        "model_batches": canonical_sha256(model_summaries),
        "market_replays": canonical_sha256(market_summaries),
        "prediction": canonical_sha256(
            {key: prediction[key] for key in prediction if key != "prefix"}
        ),
        "learner": canonical_sha256(learner),
        "scores": canonical_sha256(scores),
        "runtime": runtime_hash,
        "registry_request_access": canonical_sha256(access),
    }
    body = {
        "schema_version": STAGE_AUDIT_RECEIPT_SCHEMA_VERSION,
        "validator_id": AUTHORITATIVE_VALIDATOR_ID,
        "validator_source_sha256": identities["stage_verifier_source_sha256"],
        "validator_source_sha256_role": (
            "candidate_pin_reconciled_to_current_file_at_loaded_module_path"
        ),
        "source_runtime_trust_boundary": source_runtime_trust_boundary,
        "trusted_content_pin_boundary": trusted_content_pin_boundary,
        "prerequisite_stage": prerequisite,
        "requested_stage": _PREREQUISITE_TRANSITIONS[prerequisite],
        "stage_evidence_sha256": evidence_hash,
        "candidate_sha256": candidate_hash,
        "candidate_design_sha256": candidate_design_sha256(candidate),
        "selected_candidate_id": scores["selected_candidate_id"],
        "source_identity_receipt_sha256": source_identity[
            "source_identity_receipt_sha256"
        ],
        "runtime_source_identity_receipt_sha256": runtime_source_identity[
            "source_identity_receipt_sha256"
        ],
        "trusted_stage_content_pin_sha256": current_content_pin["pin_sha256"],
        "trusted_stage_content_authentication": current_content_pin[
            "authentication_receipt"
        ],
        "trusted_stage_content_authentication_receipt_sha256": current_content_pin[
            "authentication_receipt_sha256"
        ],
        "authenticated_store_context_sha256": authenticated_context[
            "authenticated_store_context_sha256"
        ],
        "parent_consumption_binding_sha256": expected_context.get(
            "parent_consumption_binding_sha256"
        ),
        "trusted_content_manifest_sha256": current_content_pin[
            "content_manifest_sha256"
        ],
        "trusted_stage_artifact_sha256": current_content_pin[
            "stage_artifact_sha256"
        ],
        "trusted_external_seal_receipt_sha256": current_content_pin[
            "external_seal_receipt_sha256"
        ],
        "trusted_store_state_sha256": current_content_pin[
            "trusted_store_state_sha256"
        ],
        "parent_stage_evidence_sha256": (
            None
            if parent_lineage is None
            else parent_lineage["parent_stage_evidence_sha256"]
        ),
        "parent_audit_receipt_sha256": (
            None
            if parent_lineage is None
            else parent_lineage["parent_audit_receipt_sha256"]
        ),
        "parent_consumption_entry_sha256": (
            None
            if parent_lineage is None
            else parent_lineage["parent_consumption_entry_sha256"]
        ),
        "parent_prerequisite_validation_result_sha256": (
            None
            if parent_lineage is None
            else parent_lineage[
                "parent_prerequisite_validation_result_sha256"
            ]
        ),
        "parent_semantic_receipt_sha256": (
            None
            if parent_lineage is None
            else parent_lineage["parent_semantic_receipt_sha256"]
        ),
        "parent_authorization_bundle_sha256": (
            None
            if parent_lineage is None
            else parent_lineage["parent_authorization_bundle_sha256"]
        ),
        "parent_authorization_grant_sha256": (
            None
            if parent_lineage is None
            else parent_lineage["parent_authorization_grant_sha256"]
        ),
        "parent_store_pin_sha256": (
            None
            if parent_lineage is None
            else parent_lineage["parent_store_pin_sha256"]
        ),
        "parent_consumed_stage_output_receipt_sha256": (
            None
            if parent_lineage is None
            else parent_lineage[
                "parent_consumed_stage_output_receipt_sha256"
            ]
        ),
        "parent_output_stage_evidence_document_sha256": (
            None
            if parent_lineage is None
            else parent_lineage[
                "parent_output_stage_evidence_document_sha256"
            ]
        ),
        "catalog_replay_receipt_sha256": catalog_replay[
            "replay_validation_sha256"
        ],
        "content_replay_receipt_sha256s_by_stage": {
            stage: summary["replay_validation_sha256"]
            for stage, summary in content_summaries.items()
        },
        "catalog_request_receipts_sha256": catalog_replay[
            "request_receipts_sha256"
        ],
        "content_request_receipts_sha256s_by_stage": {
            stage: summary["request_receipts_sha256"]
            for stage, summary in content_summaries.items()
        },
        "corpus_replay_trust_boundary": corpus_replay_trust_boundary,
        "corpus_replay_receipt_schema_versions": (
            corpus_replay_receipt_schema_versions
        ),
        "semantic_checks": list(REQUIRED_STAGE_VERIFIER_CHECKS),
        "check_status": status,
        "component_receipt_sha256s": component_hashes,
        "blocking_check_count": sum(
            item["status"] == "blocked" for item in status.values()
        ),
        "all_checks_completed": False,
        "authorizes_outcome_access": False,
    }
    return {**body, "audit_receipt_sha256": canonical_sha256(body)}


def authoritative_prerequisite_validator(
    evidence: Mapping[str, Any],
    stage_access_manifest: Mapping[str, Any],
    expected_context: Mapping[str, Any],
    *,
    authenticated_store_context: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed instead of manufacturing an authorizing semantic result."""

    receipt = audit_stage_evidence(
        evidence,
        stage_access_manifest,
        expected_context,
        authenticated_store_context=authenticated_store_context,
    )
    blocked = [
        check
        for check, result in receipt["check_status"].items()
        if result["status"] == "blocked"
    ]
    raise SecFilingGemmaStageVerifierBlocked(
        "Stage promotion remains disabled; unresolved authoritative checks: "
        + ", ".join(blocked)
    )


__all__ = [
    "AUTHENTICATED_STORE_VERIFIER_CONTEXT_SCHEMA_VERSION",
    "AUTHORITATIVE_VALIDATOR_ID",
    "MAX_STAGE_CONTENT_BYTES",
    "OWNED_HARDENED_TRANSPORT_MODE",
    "PARENT_CONSUMPTION_BINDING_SCHEMA_VERSION",
    "STAGE_AUDIT_RECEIPT_SCHEMA_VERSION",
    "STAGE_EVIDENCE_KEYS",
    "STAGE_EVIDENCE_SCHEMA_VERSION",
    "STAGE_RUNTIME_RECEIPT_SCHEMA_VERSION",
    "TRUSTED_STAGE_CONTENT_PIN_SCHEMA_VERSION",
    "TRUSTED_STAGE_CONTENT_AUTHENTICATION_SCHEMA_VERSION",
    "SecFilingGemmaStageVerifierBlocked",
    "SecFilingGemmaStageVerifierError",
    "audit_stage_evidence",
    "authoritative_prerequisite_validator",
    "detach_untrusted_stage_json",
    "preflight_untrusted_stage_json",
    "validate_calendar_and_universe_snapshot",
    "validate_authenticated_store_context",
    "validate_candidate_source_bytes",
    "validate_candidate_source_role_audit",
    "validate_candidate_runtime_source_audit",
    "validate_detached_catalog_evidence",
    "validate_detached_stage_content_evidence",
    "validate_learner_refit_replays",
    "validate_market_snapshot_stage_replay",
    "validate_model_attempt_batch",
    "validate_prediction_artifact_replay",
    "validate_raw_scores_gates_and_ranking",
    "validate_registry_request_and_stage_access",
    "validate_parent_stage_lineage",
    "validate_stage_runtime_receipt",
    "validate_trusted_stage_content_pin",
]
