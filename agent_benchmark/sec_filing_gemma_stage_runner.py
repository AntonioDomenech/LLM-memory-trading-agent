"""Owned one-shot SEC/market readers and local-model runners for exact claims.

These are the only production entry points that may turn authenticated store
state into SEC network effects.  A consumed-stage caller supplies no stage,
candidate, URL, digest, path, byte payload, transport, or budget.  The
request-free development entry accepts a complete self-hashed plan only as
evidence for independent store validation; its effect capability is then
derived from the atomically persisted claim.  A claim is never retried after
an indeterminate process exit.

Private SEC contact and raw Yahoo metadata exist only inside their owned call
stacks and fixed quarantine components.  Effectful public results expose only
the exact claim and store-recomputed reader receipt.  The request-free feature
checkpoint exposes only derived, self-hashed development feature rows and is
explicitly non-authorizing.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from types import MappingProxyType
from typing import Any, Final

from .sec_audit_transport import SecAuditTransport
from .sec_filing_gemma_contract import (
    CANONICAL_IDENTITY_LEXICON,
    EXTRACTOR_REQUEST_VERSION,
    PREPROCESSOR_VERSION,
    build_stage_content_manifest,
    build_redacted_input_manifest,
    canonical_sha256,
    validate_extractor_request,
)
from .sec_filing_gemma_corpus import (
    SecCorpusBudget,
    SecFilingGemmaCorpusError,
    StageDocumentBytes,
    _AUTHENTICATED_STAGE_ACCESS_BATCH_MANIFEST_SCHEMA_VERSION,
    _AuthenticatedStageAccessDocumentBatch,
    _acquire_authenticated_stage_access_document_batch,
    _detached_transport_claim,
    _reconcile_detached_transport_usage,
    _reconcile_authenticated_stage_access_receipt,
    _validate_persisted_authenticated_stage_access_batch,
)
from .sec_filing_gemma_features import (
    MARKET_LOOKBACK_ROW_COUNT,
    OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION,
    OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION,
    SecFilingGemmaFeatureError,
    build_owned_development_feature_batch,
    build_owned_development_label_batch,
    build_sec_filing_gemma_feature_row,
    validate_owned_development_feature_batch,
    validate_owned_development_label_batch,
)
from .sec_filing_gemma_reveal_store import (
    DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME,
    DEVELOPMENT_MARKET_COMPLETE_MARKER_SCHEMA_VERSION,
    DEVELOPMENT_MARKET_COMPONENT_ID,
    DEVELOPMENT_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION,
    DEVELOPMENT_SEC_ROOT_COMPLETE_MARKER_SCHEMA_VERSION,
    MARKET_SOURCE_COMPONENT_DIRECTORY_NAME,
    MAX_MARKET_BATCH_FILE_BYTES,
    MAX_MARKET_BATCH_FILES,
    MAX_MARKET_BATCH_TOTAL_BYTES,
    MAX_SEC_BATCH_FILE_BYTES,
    MAX_SEC_BATCH_FILES,
    MAX_SEC_BATCH_TOTAL_BYTES,
    MODEL_EXTRACTION_COMPLETE_MARKER_FILENAME,
    MODEL_EXTRACTION_COMPONENT_DIRECTORY_NAME,
    SEC_BATCH_COMPLETE_MARKER_FILENAME,
    SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION,
    SEC_STAGE_COMPONENT_DIRECTORY_NAME,
    STAGE_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION,
    STAGE_OUTPUTS_DIRECTORY_NAME,
    SecFilingGemmaRevealStore,
    _build_owned_model_call_intent,
    _fsync_directory,
    _secure_directory,
)
from .sec_filing_gemma_market_acquirer import (
    _acquire_owned_development_market_evidence,
    _bind_owned_market_transport_capability_to_claim,
    _unwrap_owned_development_market_acquisition,
    validate_development_market_acquisition_bundle,
)
from .sec_filing_gemma_market_evidence import MARKET_SYMBOLS
from .sec_filing_gemma_ollama import (
    build_runtime_identity_guard,
    call_ollama_extractor_attempt,
    probe_owned_ollama_runtime,
    validate_ollama_model_attempt_receipt,
    validate_ollama_runtime_probe_receipt,
)
from .sec_filing_gemma_preprocessor import (
    build_owned_preprocessing_receipt,
    preprocess_filing_event,
)
from .sec_filing_gemma_stage_access import (
    DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES,
)
from .sec_filing_gemma_stage_authorization import (
    OWNED_SEC_RAW_BATCH_MAX_BYTES,
    SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
    _sec_component_plan_from_bundle,
    validate_development_label_assembly_plan,
    validate_development_feature_assembly_plan,
)
from .sec_point_in_time import validate_sec_user_agent


class SecFilingGemmaStageRunnerError(RuntimeError):
    """The owned SEC effect could not finish with a durable exact receipt."""


_SHA256_LENGTH: Final[int] = 64
_OUTPUT_FILE_MODE: Final[int] = 0o600
_BINARY: Final[int] = getattr(os, "O_BINARY", 0)
_NOFOLLOW: Final[int] = getattr(os, "O_NOFOLLOW", 0)
_MAX_COMPONENT_PLAN_SECONDS: Final[float] = 720.0
_OWNED_MODEL_ATTEMPT_TRANSPORT_MODE: Final[str] = (
    "owned_hardened_loopback_session_requires_stage_attestation"
)
_OWNED_RUNTIME_PROBE_TRANSPORT_MODE: Final[str] = (
    "owned_hardened_loopback_runtime_probe_unattested"
)
_OWNED_DEVELOPMENT_FEATURE_INPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "feature_assembly_plan",
        "events",
        "feature_inputs_sha256",
    }
)
_OWNED_DEVELOPMENT_FEATURE_EVENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "event_ordinal",
        "event_plan_item",
        "market_prefix",
        "market_prefix_proof",
        "universe_event_proof",
        "extraction_event_proof",
    }
)
_OWNED_DEVELOPMENT_LABEL_PROJECTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "label_assembly_plan",
        "source_feature_batch",
        "maturity_audit_rows",
        "label_evidence_rows",
        "label_projection_sha256",
    }
)


def _is_bare_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_marker_bytes(value: Mapping[str, Any]) -> bytes:
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
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC completion evidence is not canonical JSON"
        ) from None


def _owned_frozen_json_snapshot(value: Any) -> Any:
    """Detach only the exact immutable containers emitted by the owned reader."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SecFilingGemmaStageRunnerError(
                "Owned SEC reader evidence contains a non-finite number"
            )
        return value
    if type(value) is tuple:
        return [_owned_frozen_json_snapshot(item) for item in value]
    if type(value) is MappingProxyType:
        if not all(type(key) is str for key in value):
            raise SecFilingGemmaStageRunnerError(
                "Owned SEC reader evidence contains a non-string key"
            )
        return {
            key: _owned_frozen_json_snapshot(item)
            for key, item in value.items()
        }
    raise SecFilingGemmaStageRunnerError(
        "Owned SEC reader evidence is not the exact frozen JSON shape"
    )


def _write_new_regular_file(path: Path, payload: bytes) -> None:
    """Create, flush, and close one immutable component file without replacing."""

    if type(payload) is not bytes or not payload:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC output payload must be non-empty exact bytes"
        )
    if len(payload) > MAX_SEC_BATCH_FILE_BYTES:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC output payload exceeds the per-file limit"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, _OUTPUT_FILE_MODE)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise SecFilingGemmaStageRunnerError(
                    "Owned SEC output file could not be completed"
                )
            remaining = remaining[written:]
        os.fsync(descriptor)
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_size != len(payload)
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned SEC output is not one exact regular file"
            )
    except SecFilingGemmaStageRunnerError:
        raise
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC output file could not be created exactly once"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _component_directory(
    reveal_store: SecFilingGemmaRevealStore,
    claim: Mapping[str, Any],
) -> Path:
    namespace = claim.get("output_namespace")
    claim_sha256 = claim.get("claim_sha256")
    if (
        type(namespace) is not str
        or not namespace
        or namespace in {".", ".."}
        or "/" in namespace
        or "\\" in namespace
        or not _is_bare_sha256(claim_sha256)
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC claim has no safe fixed output identity"
        )
    expected = (
        reveal_store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / claim_sha256
        / SEC_STAGE_COMPONENT_DIRECTORY_NAME
    )
    parent = expected.parent
    secured_parent = _secure_directory(
        parent,
        create=True,
        location="owned SEC stage claim directory",
    )
    if secured_parent != parent:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC stage claim directory identity changed"
        )
    try:
        expected.mkdir(exist_ok=False)
        _fsync_directory(parent)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC stage batch directory is not create-new"
        ) from None
    secured = _secure_directory(
        expected,
        create=False,
        location="owned SEC stage batch directory",
    )
    if secured != expected:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC stage batch directory identity changed"
        )
    return secured


def _load_component_plan(
    reveal_store: SecFilingGemmaRevealStore,
    *,
    request_sha256: str,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact effect capability only from authenticated store state."""

    tip = reveal_store.load_current_tip_anchor()
    if type(tip) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC current-tip evidence is not an exact object"
        )
    for map_name in (
        "authorization_bundles",
        "stage_sec_execution_claims",
        "stage_sec_reader_receipts",
        "stage_sec_execution_aborts",
    ):
        if type(tip.get(map_name)) is not dict:
            raise SecFilingGemmaStageRunnerError(
                "Owned SEC current-tip maps are unavailable"
            )
    if (
        tip["stage_sec_execution_claims"].get(request_sha256) != claim
        or request_sha256 in tip["stage_sec_reader_receipts"]
        or request_sha256 in tip["stage_sec_execution_aborts"]
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC claim is not the exact active current-tip claim"
        )
    bundle = tip["authorization_bundles"].get(request_sha256)
    if type(bundle) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC claim lacks its persisted grant bundle"
        )
    try:
        exact_bundle, grant, component_plan = _sec_component_plan_from_bundle(bundle)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC persisted grant cannot derive an exact component plan"
        ) from None
    if exact_bundle != bundle or type(grant) is not dict or type(component_plan) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC component plan changed while being authenticated"
        )
    expected_claim_bindings = {
        "request_sha256": request_sha256,
        "candidate_sha256": grant.get("candidate_sha256"),
        "authorized_stage": grant.get("stage"),
        "output_namespace": grant.get("output_namespace"),
        "authorization_bundle_sha256": bundle.get("bundle_sha256"),
        "authorization_grant_sha256": grant.get("authorization_grant_sha256"),
        "stage_access_manifest_sha256": grant.get("stage_access_manifest_sha256"),
        "sec_component_id": SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
        "sec_component_plan_sha256": canonical_sha256(component_plan),
    }
    if any(claim.get(key) != value for key, value in expected_claim_bindings.items()):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC component plan crossed its durable claim"
        )
    sec_plan = component_plan.get("sec_access_plan")
    documents = sec_plan.get("documents") if type(sec_plan) is dict else None
    max_requests = component_plan.get("max_sec_requests")
    authorized_max_bytes = component_plan.get(
        "authorized_max_sec_response_bytes"
    )
    owned_max_bytes = component_plan.get("owned_sec_raw_batch_max_bytes")
    max_bytes = component_plan.get("max_sec_response_bytes")
    max_seconds = component_plan.get("max_sec_acquisition_seconds")
    if (
        type(documents) is not list
        or not documents
        or type(max_requests) is not int
        or isinstance(max_requests, bool)
        or max_requests != len(documents)
        or type(max_bytes) is not int
        or isinstance(max_bytes, bool)
        or max_bytes < 1
        or type(authorized_max_bytes) is not int
        or isinstance(authorized_max_bytes, bool)
        or authorized_max_bytes < max_bytes
        or owned_max_bytes != OWNED_SEC_RAW_BATCH_MAX_BYTES
        or max_bytes
        != min(authorized_max_bytes, OWNED_SEC_RAW_BATCH_MAX_BYTES)
        or type(max_seconds) not in {int, float}
        or isinstance(max_seconds, bool)
        or not 0.0 < float(max_seconds) <= _MAX_COMPONENT_PLAN_SECONDS
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC component budgets are not exact and bounded"
        )
    return {
        "documents": [dict(document) for document in documents],
        "max_requests": max_requests,
        "max_bytes": max_bytes,
        "max_seconds": float(max_seconds),
    }


def _load_development_root_component_plan(
    reveal_store: SecFilingGemmaRevealStore,
    *,
    development_root_scope_sha256: str,
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the request-free development effect from its persisted claim."""

    tip = reveal_store.load_current_tip_anchor()
    if type(tip) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC current-tip evidence is not an exact object"
        )
    for map_name in (
        "development_sec_execution_claims",
        "development_sec_reader_receipts",
        "development_sec_execution_aborts",
    ):
        if type(tip.get(map_name)) is not dict:
            raise SecFilingGemmaStageRunnerError(
                "Owned development SEC current-tip maps are unavailable"
            )
    if (
        tip["development_sec_execution_claims"].get(
            development_root_scope_sha256
        )
        != claim
        or development_root_scope_sha256
        in tip["development_sec_reader_receipts"]
        or development_root_scope_sha256
        in tip["development_sec_execution_aborts"]
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC claim is not the exact active current-tip claim"
        )

    persisted_plan = claim.get("development_content_root_plan")
    if type(persisted_plan) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC claim lacks its exact persisted plan"
        )
    plan_sha256 = persisted_plan.get("development_content_root_plan_sha256")
    root_scope = persisted_plan.get("root_scope")
    sec_plan = persisted_plan.get("sec_access_plan")
    budgets = persisted_plan.get("budgets")
    output = persisted_plan.get("output")
    universe = persisted_plan.get("corpus_universe_manifest")
    if (
        not _is_bare_sha256(plan_sha256)
        or canonical_sha256(
            {
                key: value
                for key, value in persisted_plan.items()
                if key != "development_content_root_plan_sha256"
            }
        )
        != plan_sha256
        or type(root_scope) is not dict
        or canonical_sha256(root_scope) != development_root_scope_sha256
        or persisted_plan.get("development_root_scope_sha256")
        != development_root_scope_sha256
        or type(sec_plan) is not dict
        or type(budgets) is not dict
        or type(output) is not dict
        or type(universe) is not dict
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC persisted plan is not canonically bound"
        )

    documents = sec_plan.get("documents")
    max_requests = budgets.get("max_sec_requests")
    max_bytes = budgets.get("max_raw_batch_bytes")
    max_seconds = budgets.get("max_sec_acquisition_seconds")
    expected_claim_bindings = {
        "development_root_scope_sha256": development_root_scope_sha256,
        "development_content_root_plan_sha256": plan_sha256,
        "candidate_sha256": root_scope.get("candidate_sha256"),
        "candidate_design_sha256": root_scope.get("candidate_design_sha256"),
        "attempt_id": root_scope.get("attempt_id"),
        "corpus_universe_sha256": root_scope.get("corpus_universe_sha256"),
        "corpus_universe_semantic_sha256": root_scope.get(
            "corpus_universe_semantic_sha256"
        ),
        "output_namespace": output.get("namespace"),
        "sec_component_id": DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    }
    if any(
        claim.get(key) != value for key, value in expected_claim_bindings.items()
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC component plan crossed its durable claim"
        )
    if (
        root_scope.get("artifact_stage") != "development"
        or root_scope.get("component_id")
        != DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
        or root_scope.get("output_namespace") != output.get("namespace")
        or universe.get("universe_sha256")
        != root_scope.get("corpus_universe_sha256")
        or sec_plan.get("artifact_stage") != "development"
        or sec_plan.get("selection_policy")
        != "all_and_only_development_stage_universe_primary_documents"
        or sec_plan.get("method") != "GET"
        or sec_plan.get("network_scope") != "official_sec_https_only"
        or sec_plan.get("redirects_permitted") is not False
        or sec_plan.get("retries_permitted") is not False
        or sec_plan.get("cache_substitution_permitted") is not False
        or output.get("component_id") != DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID
        or output.get("write_mode") != "create_new_exclusive"
        or output.get("existing_namespace_reuse_permitted") is not False
        or type(documents) is not list
        or not documents
        or sec_plan.get("document_count") != len(documents)
        or root_scope.get("document_count") != len(documents)
        or type(max_requests) is not int
        or isinstance(max_requests, bool)
        or max_requests != len(documents)
        or max_bytes != DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES
        or max_bytes != OWNED_SEC_RAW_BATCH_MAX_BYTES
        or type(max_seconds) not in {int, float}
        or isinstance(max_seconds, bool)
        or not 0.0 < float(max_seconds) <= _MAX_COMPONENT_PLAN_SECONDS
        or budgets.get("max_redirects") != 0
        or budgets.get("max_retries") != 0
        or budgets.get("max_paid_api_calls") != 0
        or budgets.get("max_estimated_cost_usd") != 0.0
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC component scope or budgets are not exact"
        )
    if any(
        type(document) is not dict
        or set(document) != {"accession_number", "official_url"}
        or type(document["accession_number"]) is not str
        or not document["accession_number"]
        or type(document["official_url"]) is not str
        or not document["official_url"]
        for document in documents
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC document plan is not exact"
        )
    return {
        "documents": [dict(document) for document in documents],
        "max_requests": max_requests,
        "max_bytes": max_bytes,
        "max_seconds": float(max_seconds),
        "corpus_universe_sha256": root_scope["corpus_universe_sha256"],
        "corpus_universe_manifest": universe,
        "development_content_root_plan_sha256": plan_sha256,
    }


@contextmanager
def _owned_transport_factory(
    *,
    user_agent: str,
    transport_budget: SecCorpusBudget,
    cache_directory: Path,
) -> Iterator[SecAuditTransport]:
    """Construct the reviewed network transport without exposing it publicly."""

    import requests

    with requests.Session() as session:
        session.trust_env = False
        session.proxies.clear()
        yield SecAuditTransport(
            session=session,
            cache_dir=cache_directory,
            user_agent=user_agent,
            budget=transport_budget,
            clock=time.monotonic,
            sleep=time.sleep,
            max_retries=0,
            max_redirects=0,
            allow_cache_reads=False,
            allow_cache_writes=False,
        )


def _validated_batch_payloads(
    batch: _AuthenticatedStageAccessDocumentBatch,
    *,
    documents_plan: list[dict[str, str]],
    expected_user_agent_sha256: str,
    expected_max_requests: int,
    expected_max_bytes: int,
    expected_max_seconds: float,
) -> list[tuple[str, str, bytes]]:
    if type(batch) is not _AuthenticatedStageAccessDocumentBatch:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader returned an unexpected batch type"
        )
    if len(batch.documents) != len(documents_plan):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader omitted or added a document"
        )
    try:
        replayed_batch = _validate_persisted_authenticated_stage_access_batch(
            authenticated_document_plan=documents_plan,
            raw_documents=tuple(
                document.raw_primary_document for document in batch.documents
            ),
            normalized_documents=tuple(
                document.normalized_text for document in batch.documents
            ),
            request_receipts_json=batch.request_receipts_json,
            byte_manifest_json=batch.byte_manifest_json,
            expected_max_requests=expected_max_requests,
            expected_max_bytes=expected_max_bytes,
            expected_max_seconds=expected_max_seconds,
            expected_user_agent_sha256=expected_user_agent_sha256,
        )
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader persisted evidence does not replay"
        ) from None
    if (
        replayed_batch.documents != batch.documents
        or _owned_frozen_json_snapshot(replayed_batch.request_receipts)
        != _owned_frozen_json_snapshot(batch.request_receipts)
        or _owned_frozen_json_snapshot(replayed_batch.byte_manifest)
        != _owned_frozen_json_snapshot(batch.byte_manifest)
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader in-memory evidence differs from its persisted replay"
        )
    payloads: list[tuple[str, str, bytes]] = []
    for ordinal, (planned, document) in enumerate(
        zip(documents_plan, batch.documents), start=1
    ):
        if (
            type(document) is not StageDocumentBytes
            or document.accession_number != planned["accession_number"]
            or document.url != planned["official_url"]
            or type(document.raw_primary_document) is not bytes
            or not document.raw_primary_document
            or type(document.normalized_text) is not bytes
            or not document.normalized_text
            or hashlib.sha256(document.raw_primary_document).hexdigest()
            != document.primary_document_sha256
            or hashlib.sha256(document.normalized_text).hexdigest()
            != document.normalized_text_sha256
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned SEC reader bytes do not reconcile to the authenticated plan"
            )
        prefix = f"document-{ordinal:04d}"
        payloads.extend(
            (
                (
                    f"{prefix}-raw",
                    f"{prefix}.raw",
                    document.raw_primary_document,
                ),
                (
                    f"{prefix}-normalized",
                    f"{prefix}.normalized.txt",
                    document.normalized_text,
                ),
            )
        )
    raw_total = sum(len(document.raw_primary_document) for document in batch.documents)
    normalized_total = sum(len(document.normalized_text) for document in batch.documents)
    if (
        raw_total > OWNED_SEC_RAW_BATCH_MAX_BYTES
        or normalized_total > 2 * OWNED_SEC_RAW_BATCH_MAX_BYTES
        or any(
            len(document.normalized_text) > MAX_SEC_BATCH_FILE_BYTES
            for document in batch.documents
        )
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader bytes exceed their claim-bound durable ceilings"
        )
    try:
        receipts = json.loads(batch.request_receipts_json.decode("utf-8"))
        manifest = json.loads(batch.byte_manifest_json.decode("utf-8"))
        canonical_receipts = json.dumps(
            receipts,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        canonical_manifest = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader canonical evidence cannot be decoded"
        ) from None
    expected_rows: list[dict[str, Any]] = []
    reconciled_receipts: list[dict[str, Any]] = []
    if type(receipts) is list and type(manifest) is dict:
        user_agent_sha256 = manifest.get("user_agent_sha256")
        try:
            for ordinal, (planned, document, receipt) in enumerate(
                zip(documents_plan, batch.documents, receipts), start=1
            ):
                reconciled = _reconcile_authenticated_stage_access_receipt(
                    receipt,
                    sequence_number=ordinal,
                    official_url=planned["official_url"],
                    raw_document=document.raw_primary_document,
                    user_agent_sha256=user_agent_sha256,
                )
                if document.request_receipt_sha256 != reconciled[
                    "request_receipt_sha256"
                ]:
                    raise SecFilingGemmaCorpusError(
                        "Authenticated document lost its request receipt binding"
                    )
                reconciled_receipts.append(reconciled)
                expected_rows.append(
                    {
                        "sequence_number": ordinal,
                        "accession_number": planned["accession_number"],
                        "official_url": planned["official_url"],
                        "raw_document_sha256": hashlib.sha256(
                            document.raw_primary_document
                        ).hexdigest(),
                        "raw_document_bytes": len(document.raw_primary_document),
                        "normalized_document_sha256": hashlib.sha256(
                            document.normalized_text
                        ).hexdigest(),
                        "normalized_document_bytes": len(document.normalized_text),
                        "request_receipt_sha256": reconciled[
                            "request_receipt_sha256"
                        ],
                    }
                )
            transport_claim = _detached_transport_claim(
                manifest.get("transport_security")
            )
            if (
                transport_claim["transport_max_requests"]
                != expected_max_requests
                or transport_claim["transport_max_bytes"] != expected_max_bytes
                or transport_claim["transport_max_seconds"]
                != expected_max_seconds
            ):
                raise SecFilingGemmaCorpusError(
                    "Authenticated transport evidence crossed its claimed budgets"
                )
            _reconcile_detached_transport_usage(
                transport_claim,
                request_count=len(expected_rows),
                byte_count=sum(
                    len(document.raw_primary_document)
                    for document in batch.documents
                ),
            )
        except (KeyError, SecFilingGemmaCorpusError, TypeError, ValueError):
            reconciled_receipts = []
            expected_rows = []
    manifest_body = (
        {key: manifest[key] for key in manifest if key != "byte_manifest_sha256"}
        if type(manifest) is dict
        else {}
    )
    batch_receipts = [dict(receipt) for receipt in batch.request_receipts]
    if (
        type(receipts) is not list
        or len(receipts) != len(documents_plan)
        or type(manifest) is not dict
        or batch.request_receipts_json != canonical_receipts
        or batch.byte_manifest_json != canonical_manifest
        or receipts != reconciled_receipts
        or batch_receipts != reconciled_receipts
        or manifest.get("schema_version")
        != _AUTHENTICATED_STAGE_ACCESS_BATCH_MANIFEST_SCHEMA_VERSION
        or manifest.get("document_count") != len(expected_rows)
        or manifest.get("user_agent_sha256") != expected_user_agent_sha256
        or manifest.get("documents") != expected_rows
        or manifest.get("request_receipts_sha256")
        != canonical_sha256(reconciled_receipts)
        or manifest.get("acquisition_request_count") != len(expected_rows)
        or manifest.get("acquisition_bytes")
        != sum(len(document.raw_primary_document) for document in batch.documents)
        or manifest.get("outer_budget_role")
        != "post_transport_reconciliation_not_streaming_protection"
        or manifest.get("selection_policy")
        != "exact_ordered_authenticated_stage_access_document_plan"
        or manifest.get("caller_stage_candidate_or_digest_authority_accepted")
        is not False
        or manifest.get("byte_manifest_sha256") != canonical_sha256(manifest_body)
        or canonical_sha256(_owned_frozen_json_snapshot(batch.byte_manifest))
        != canonical_sha256(manifest)
        or canonical_sha256(receipts) != batch.request_receipts_sha256
        or canonical_sha256(manifest) != hashlib.sha256(canonical_manifest).hexdigest()
        or manifest.get("byte_manifest_sha256") != batch.byte_manifest_sha256
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader canonical evidence does not reconcile"
        )
    payloads.extend(
        (
            (
                "request-receipts-json",
                "request-receipts.json",
                batch.request_receipts_json,
            ),
            (
                "byte-manifest-json",
                "byte-manifest.json",
                batch.byte_manifest_json,
            ),
        )
    )
    if len(payloads) + 1 > MAX_SEC_BATCH_FILES:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader output contains too many files"
        )
    total = sum(len(payload) for _logical_id, _name, payload in payloads)
    if (
        any(
            len(payload) > MAX_SEC_BATCH_FILE_BYTES
            for _logical_id, _name, payload in payloads
        )
        or total > MAX_SEC_BATCH_TOTAL_BYTES
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader output exceeds its durable byte limits"
        )
    names = [name.casefold() for _logical_id, name, _payload in payloads]
    if len(names) != len(set(names)):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC reader output contains a case-colliding file"
        )
    return payloads


def _persist_batch(
    component_directory: Path,
    *,
    request_sha256: str,
    claim: Mapping[str, Any],
    payloads: list[tuple[str, str, bytes]],
) -> None:
    byte_index: list[dict[str, Any]] = []
    for ordinal, (logical_id, name, payload) in enumerate(payloads, start=1):
        if (
            not name
            or name in {".", "..", SEC_BATCH_COMPLETE_MARKER_FILENAME}
            or "/" in name
            or "\\" in name
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned SEC output filename is not flat and safe"
            )
        _write_new_regular_file(component_directory / name, payload)
        byte_index.append(
            {
                "ordinal": ordinal,
                "logical_id": logical_id,
                "relative_path": name,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    marker_body = {
        "schema_version": SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION,
        "request_sha256": request_sha256,
        "claim_sha256": claim["claim_sha256"],
        "component_id": SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
        "byte_index": byte_index,
        "byte_index_sha256": canonical_sha256(byte_index),
    }
    marker = {**marker_body, "marker_sha256": canonical_sha256(marker_body)}
    _write_new_regular_file(
        component_directory / SEC_BATCH_COMPLETE_MARKER_FILENAME,
        _canonical_marker_bytes(marker),
    )
    _fsync_directory(component_directory)
    observed = [entry.name for entry in component_directory.iterdir()]
    expected = [name for _logical_id, name, _payload in payloads] + [
        SEC_BATCH_COMPLETE_MARKER_FILENAME
    ]
    if (
        set(observed) != set(expected)
        or len(observed) != len(expected)
        or len({name.casefold() for name in observed}) != len(observed)
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC stage batch has missing, extra, or colliding files"
        )


def _development_root_payloads(
    batch: _AuthenticatedStageAccessDocumentBatch,
    *,
    component_plan: Mapping[str, Any],
    batch_payloads: list[tuple[str, str, bytes]],
) -> tuple[list[tuple[str, str, bytes]], dict[str, Any]]:
    """Add the exact complete-universe and development content identities."""

    universe = component_plan.get("corpus_universe_manifest")
    universe_sha256 = component_plan.get("corpus_universe_sha256")
    if (
        type(universe) is not dict
        or not _is_bare_sha256(universe_sha256)
        or universe.get("universe_sha256") != universe_sha256
        or type(batch) is not _AuthenticatedStageAccessDocumentBatch
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC universe evidence is not exact"
        )
    try:
        content_manifest = build_stage_content_manifest(
            artifact_stage="development",
            corpus_universe_sha256=universe_sha256,
            documents=[
                {
                    "accession_number": document.accession_number,
                    "primary_document_sha256": document.primary_document_sha256,
                    "normalized_text_sha256": document.normalized_text_sha256,
                    "primary_document_bytes": len(
                        document.raw_primary_document
                    ),
                    "normalized_text_bytes": len(document.normalized_text),
                }
                for document in batch.documents
            ],
            universe_manifest=universe,
        )
        universe_bytes = _canonical_marker_bytes(universe)
        content_manifest_bytes = _canonical_marker_bytes(content_manifest)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC content identity cannot be constructed"
        ) from None
    payloads = list(batch_payloads)
    payloads.extend(
        (
            (
                "corpus-universe-json",
                "corpus-universe.json",
                universe_bytes,
            ),
            (
                "development-content-manifest-json",
                "development-content-manifest.json",
                content_manifest_bytes,
            ),
        )
    )
    if len(payloads) + 1 > MAX_SEC_BATCH_FILES:
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC output contains too many files"
        )
    total = sum(len(payload) for _logical_id, _name, payload in payloads)
    names = [name.casefold() for _logical_id, name, _payload in payloads]
    if (
        any(
            len(payload) > MAX_SEC_BATCH_FILE_BYTES
            for _logical_id, _name, payload in payloads
        )
        or total > MAX_SEC_BATCH_TOTAL_BYTES
        or len(names) != len(set(names))
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC output exceeds its durable limits"
        )
    return payloads, content_manifest


def _persist_development_root_batch(
    component_directory: Path,
    *,
    claim: Mapping[str, Any],
    component_plan: Mapping[str, Any],
    content_manifest: Mapping[str, Any],
    payloads: list[tuple[str, str, bytes]],
) -> None:
    byte_index: list[dict[str, Any]] = []
    for ordinal, (logical_id, name, payload) in enumerate(payloads, start=1):
        if (
            not name
            or name in {".", "..", SEC_BATCH_COMPLETE_MARKER_FILENAME}
            or "/" in name
            or "\\" in name
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned development SEC output filename is not flat and safe"
            )
        _write_new_regular_file(component_directory / name, payload)
        byte_index.append(
            {
                "ordinal": ordinal,
                "logical_id": logical_id,
                "relative_path": name,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    marker_body = {
        "schema_version": DEVELOPMENT_SEC_ROOT_COMPLETE_MARKER_SCHEMA_VERSION,
        "development_root_scope_sha256": claim[
            "development_root_scope_sha256"
        ],
        "claim_sha256": claim["claim_sha256"],
        "candidate_sha256": claim["candidate_sha256"],
        "corpus_universe_sha256": component_plan[
            "corpus_universe_sha256"
        ],
        "development_content_root_plan_sha256": component_plan[
            "development_content_root_plan_sha256"
        ],
        "component_id": DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
        "development_content_manifest_sha256": content_manifest[
            "content_manifest_sha256"
        ],
        "byte_index": byte_index,
        "byte_index_sha256": canonical_sha256(byte_index),
    }
    marker = {**marker_body, "marker_sha256": canonical_sha256(marker_body)}
    _write_new_regular_file(
        component_directory / SEC_BATCH_COMPLETE_MARKER_FILENAME,
        _canonical_marker_bytes(marker),
    )
    _fsync_directory(component_directory)
    observed = [entry.name for entry in component_directory.iterdir()]
    expected = [name for _logical_id, name, _payload in payloads] + [
        SEC_BATCH_COMPLETE_MARKER_FILENAME
    ]
    if (
        set(observed) != set(expected)
        or len(observed) != len(expected)
        or len({name.casefold() for name in observed}) != len(observed)
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC batch has missing, extra, or colliding files"
        )


def _attempt_abort(
    reveal_store: SecFilingGemmaRevealStore,
    *,
    request_sha256: str,
    reason: str,
) -> None:
    try:
        reveal_store.abort_authorized_sec_stage_execution(
            request_sha256=request_sha256,
            reason=reason,
        )
    except Exception:
        # The primary failure remains fixed and redacted.  A later invocation
        # will still refuse to repeat the already-durable claim.
        return


def _attempt_development_root_abort(
    reveal_store: SecFilingGemmaRevealStore,
    *,
    development_root_scope_sha256: str,
    reason: str,
) -> None:
    try:
        reveal_store.abort_owned_development_sec_root_execution(
            development_root_scope_sha256=development_root_scope_sha256,
            reason=reason,
        )
    except Exception:
        # The fixed local claim remains non-repeatable even if terminal state
        # persistence itself cannot be completed by this process.
        return


def run_authorized_sec_stage(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    request_sha256: str,
    user_agent: str,
) -> dict[str, Any]:
    """Execute exactly one persisted SEC grant and return its claim and receipt."""

    if type(reveal_store) is not SecFilingGemmaRevealStore:
        raise TypeError("reveal_store must be the owned reveal-store implementation")
    try:
        user_agent_audit = validate_sec_user_agent(user_agent)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Private SEC contact must be validated before claiming a grant"
        ) from None
    claim_result = reveal_store.claim_authorized_sec_stage_execution(
        request_sha256=request_sha256,
        sec_user_agent_sha256=user_agent_audit.sha256,
    )
    if type(claim_result) is not dict or set(claim_result) != {
        "claim",
        "created",
        "reader_receipt",
        "abort",
    }:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC claim result is not exact"
        )
    claim = claim_result["claim"]
    if type(claim) is not dict or claim.get("request_sha256") != request_sha256:
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC claim crossed its request"
        )
    if claim_result["created"] is False:
        receipt = claim_result["reader_receipt"]
        abort = claim_result["abort"]
        if type(receipt) is dict and abort is None:
            if receipt.get("claim_sha256") != claim.get("claim_sha256"):
                raise SecFilingGemmaStageRunnerError(
                    "Completed SEC receipt crossed its durable claim"
                )
            try:
                replayed = reveal_store._record_authorized_sec_stage_reader_output(
                    request_sha256=request_sha256
                )
            except Exception:
                raise SecFilingGemmaStageRunnerError(
                    "Completed SEC receipt cannot replay its durable output"
                ) from None
            if replayed != receipt:
                raise SecFilingGemmaStageRunnerError(
                    "Completed SEC receipt differs from replayed durable output"
                )
            return {"claim": claim, "reader_receipt": replayed}
        if receipt is None and abort is None:
            # The prior process may have durably sealed complete.json and then
            # exited before the store receipt CAS.  Rehashing that fixed local
            # directory performs no SEC reader I/O and is the only safe
            # recovery.  If no exact marker exists, terminally abort instead of
            # repeating an effect whose completion is unknown.
            try:
                recovered = reveal_store._record_authorized_sec_stage_reader_output(
                    request_sha256=request_sha256
                )
                if (
                    type(recovered) is not dict
                    or recovered.get("claim_sha256") != claim.get("claim_sha256")
                ):
                    raise SecFilingGemmaStageRunnerError(
                        "Recovered SEC receipt crossed its durable claim"
                    )
            except Exception:
                _attempt_abort(
                    reveal_store,
                    request_sha256=request_sha256,
                    reason="claim_recovered_without_terminal_receipt",
                )
            else:
                return {"claim": claim, "reader_receipt": recovered}
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC claim is terminal or indeterminate and cannot be retried"
        )
    if claim_result["created"] is not True or any(
        claim_result[key] is not None for key in ("reader_receipt", "abort")
    ):
        raise SecFilingGemmaStageRunnerError(
            "New owned SEC claim has an impossible terminal state"
        )

    phase = "validation"
    try:
        component_plan = _load_component_plan(
            reveal_store,
            request_sha256=request_sha256,
            claim=claim,
        )
        if (
            type(component_plan.get("documents")) is not list
            or component_plan.get("max_requests")
            != len(component_plan["documents"])
            or type(component_plan.get("max_bytes")) is not int
            or isinstance(component_plan.get("max_bytes"), bool)
            or not 1
            <= component_plan["max_bytes"]
            <= OWNED_SEC_RAW_BATCH_MAX_BYTES
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned SEC execution plan exceeds its durable raw-byte ceiling"
            )
        component_directory = _component_directory(reveal_store, claim)
        outer_budget = SecCorpusBudget(
            clock=time.monotonic,
            max_requests=component_plan["max_requests"],
            max_bytes=component_plan["max_bytes"],
            max_seconds=component_plan["max_seconds"],
        )
        transport_budget = SecCorpusBudget(
            clock=time.monotonic,
            max_requests=component_plan["max_requests"],
            max_bytes=component_plan["max_bytes"],
            max_seconds=component_plan["max_seconds"],
        )
        reveal_store._revalidate_authorized_sec_execution_sources(claim)
        phase = "external_effect"
        with _owned_transport_factory(
            user_agent=user_agent,
            transport_budget=transport_budget,
            cache_directory=reveal_store.store_directory / ".disabled-sec-cache",
        ) as transport:
            batch = _acquire_authenticated_stage_access_document_batch(
                authenticated_document_plan=component_plan["documents"],
                transport=transport,
                user_agent=user_agent,
                budget=outer_budget,
            )
        phase = "durable_output"
        payloads = _validated_batch_payloads(
            batch,
            documents_plan=component_plan["documents"],
            expected_user_agent_sha256=user_agent_audit.sha256,
            expected_max_requests=component_plan["max_requests"],
            expected_max_bytes=component_plan["max_bytes"],
            expected_max_seconds=component_plan["max_seconds"],
        )
        _persist_batch(
            component_directory,
            request_sha256=request_sha256,
            claim=claim,
            payloads=payloads,
        )
        receipt = reveal_store._record_authorized_sec_stage_reader_output(
            request_sha256=request_sha256
        )
        if type(receipt) is not dict or receipt.get("claim_sha256") != claim.get(
            "claim_sha256"
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned SEC store receipt does not bind the execution claim"
            )
        return {"claim": claim, "reader_receipt": receipt}
    except Exception:
        reason = (
            "external_effect_failed_or_completion_unknown"
            if phase == "external_effect"
            else "durable_output_verification_failed"
        )
        _attempt_abort(
            reveal_store,
            request_sha256=request_sha256,
            reason=reason,
        )
        raise SecFilingGemmaStageRunnerError(
            "Owned SEC stage execution failed and will not be retried"
        ) from None


def run_owned_development_sec_root(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    development_content_root_plan: Mapping[str, Any],
    user_agent: str,
) -> dict[str, Any]:
    """Serialize and acquire one complete request-free development corpus."""

    if type(reveal_store) is not SecFilingGemmaRevealStore:
        raise TypeError("reveal_store must be the owned reveal-store implementation")
    if type(development_content_root_plan) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Development SEC root plan must be an exact JSON object"
        )
    try:
        validate_sec_user_agent(user_agent)
        root_scope = development_content_root_plan["root_scope"]
        scope_hash = development_content_root_plan[
            "development_root_scope_sha256"
        ]
        plan_hash = development_content_root_plan[
            "development_content_root_plan_sha256"
        ]
        if (
            type(root_scope) is not dict
            or not _is_bare_sha256(scope_hash)
            or not _is_bare_sha256(plan_hash)
            or canonical_sha256(root_scope) != scope_hash
            or canonical_sha256(
                {
                    key: value
                    for key, value in development_content_root_plan.items()
                    if key != "development_content_root_plan_sha256"
                }
            )
            != plan_hash
        ):
            raise ValueError("development root identity mismatch")
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Development SEC root identity must be canonical before execution locking"
        ) from None
    try:
        execution_lock = (
            reveal_store._owned_development_sec_root_execution_lock(
                development_root_scope_sha256=scope_hash
            )
        )
        execution_lock.__enter__()
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Development SEC root execution is already owned or cannot be locked"
        ) from None
    try:
        return _run_owned_development_sec_root_locked(
            reveal_store=reveal_store,
            development_content_root_plan=development_content_root_plan,
            user_agent=user_agent,
        )
    finally:
        execution_lock.__exit__(None, None, None)


def _run_owned_development_sec_root_locked(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    development_content_root_plan: Mapping[str, Any],
    user_agent: str,
) -> dict[str, Any]:
    """Acquire all-and-only development filings under one request-free claim."""

    if type(reveal_store) is not SecFilingGemmaRevealStore:
        raise TypeError("reveal_store must be the owned reveal-store implementation")
    if type(development_content_root_plan) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Development SEC root plan must be an exact JSON object"
        )
    try:
        user_agent_audit = validate_sec_user_agent(user_agent)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Private SEC contact must be validated before claiming a development root"
        ) from None

    claim_result = reveal_store.claim_owned_development_sec_root_execution(
        development_content_root_plan=development_content_root_plan,
        sec_user_agent_sha256=user_agent_audit.sha256,
    )
    if type(claim_result) is not dict or set(claim_result) != {
        "claim",
        "created",
        "reader_receipt",
        "abort",
    }:
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC claim result is not exact"
        )
    claim = claim_result["claim"]
    submitted_plan = (
        claim.get("development_content_root_plan")
        if type(claim) is dict
        else None
    )
    development_root_scope_sha256 = (
        submitted_plan.get("development_root_scope_sha256")
        if type(submitted_plan) is dict
        else None
    )
    plan_sha256 = (
        submitted_plan.get("development_content_root_plan_sha256")
        if type(submitted_plan) is dict
        else None
    )
    submitted_root_scope = (
        submitted_plan.get("root_scope")
        if type(submitted_plan) is dict
        else None
    )
    if (
        type(claim) is not dict
        or type(submitted_plan) is not dict
        or submitted_plan != development_content_root_plan
        or not _is_bare_sha256(development_root_scope_sha256)
        or not _is_bare_sha256(plan_sha256)
        or type(submitted_root_scope) is not dict
        or canonical_sha256(submitted_root_scope)
        != development_root_scope_sha256
        or canonical_sha256(
            {
                key: value
                for key, value in submitted_plan.items()
                if key != "development_content_root_plan_sha256"
            }
        )
        != plan_sha256
        or claim.get("development_root_scope_sha256")
        != development_root_scope_sha256
        or claim.get("development_content_root_plan_sha256") != plan_sha256
        or claim.get("development_content_root_plan") != submitted_plan
        or claim.get("sec_user_agent_sha256") != user_agent_audit.sha256
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC claim crossed its submitted root"
        )

    if claim_result["created"] is False:
        receipt = claim_result["reader_receipt"]
        abort = claim_result["abort"]
        if type(receipt) is dict and abort is None:
            if (
                receipt.get("claim_sha256") != claim.get("claim_sha256")
                or (
                    receipt.get("development_root_scope_sha256")
                    != development_root_scope_sha256
                )
            ):
                raise SecFilingGemmaStageRunnerError(
                    "Completed development SEC receipt crossed its durable claim"
                )
            try:
                replayed = (
                    reveal_store._record_owned_development_sec_root_reader_output(
                        development_root_scope_sha256=(
                            development_root_scope_sha256
                        )
                    )
                )
            except Exception:
                raise SecFilingGemmaStageRunnerError(
                    "Completed development SEC receipt cannot replay its durable output"
                ) from None
            if replayed != receipt:
                raise SecFilingGemmaStageRunnerError(
                    "Completed development SEC receipt differs from durable replay"
                )
            return {"claim": claim, "reader_receipt": replayed}
        if receipt is None and abort is None:
            try:
                recovered = (
                    reveal_store._record_owned_development_sec_root_reader_output(
                        development_root_scope_sha256=(
                            development_root_scope_sha256
                        )
                    )
                )
                if (
                    type(recovered) is not dict
                    or recovered.get("claim_sha256")
                    != claim.get("claim_sha256")
                    or recovered.get("development_root_scope_sha256")
                    != development_root_scope_sha256
                ):
                    raise SecFilingGemmaStageRunnerError(
                        "Recovered development SEC receipt crossed its claim"
                    )
            except Exception:
                _attempt_development_root_abort(
                    reveal_store,
                    development_root_scope_sha256=(
                        development_root_scope_sha256
                    ),
                    reason="claim_recovered_without_terminal_receipt",
                )
            else:
                return {"claim": claim, "reader_receipt": recovered}
        raise SecFilingGemmaStageRunnerError(
            "Owned development SEC claim is terminal or indeterminate and cannot be retried"
        )
    if claim_result["created"] is not True or any(
        claim_result[key] is not None for key in ("reader_receipt", "abort")
    ):
        raise SecFilingGemmaStageRunnerError(
            "New owned development SEC claim has an impossible terminal state"
        )

    phase = "validation"
    marker_sealed = False
    try:
        component_plan = _load_development_root_component_plan(
            reveal_store,
            development_root_scope_sha256=development_root_scope_sha256,
            claim=claim,
        )
        if (
            type(component_plan.get("documents")) is not list
            or component_plan.get("max_requests")
            != len(component_plan["documents"])
            or component_plan.get("max_bytes")
            != DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned development SEC plan exceeds its durable raw-byte ceiling"
            )
        component_directory = _component_directory(reveal_store, claim)
        outer_budget = SecCorpusBudget(
            clock=time.monotonic,
            max_requests=component_plan["max_requests"],
            max_bytes=component_plan["max_bytes"],
            max_seconds=component_plan["max_seconds"],
        )
        transport_budget = SecCorpusBudget(
            clock=time.monotonic,
            max_requests=component_plan["max_requests"],
            max_bytes=component_plan["max_bytes"],
            max_seconds=component_plan["max_seconds"],
        )
        reveal_store._revalidate_authorized_sec_execution_sources(claim)
        phase = "external_effect"
        with _owned_transport_factory(
            user_agent=user_agent,
            transport_budget=transport_budget,
            cache_directory=reveal_store.store_directory / ".disabled-sec-cache",
        ) as transport:
            batch = _acquire_authenticated_stage_access_document_batch(
                authenticated_document_plan=component_plan["documents"],
                transport=transport,
                user_agent=user_agent,
                budget=outer_budget,
            )
        phase = "durable_output"
        batch_payloads = _validated_batch_payloads(
            batch,
            documents_plan=component_plan["documents"],
            expected_user_agent_sha256=user_agent_audit.sha256,
            expected_max_requests=component_plan["max_requests"],
            expected_max_bytes=component_plan["max_bytes"],
            expected_max_seconds=component_plan["max_seconds"],
        )
        payloads, content_manifest = _development_root_payloads(
            batch,
            component_plan=component_plan,
            batch_payloads=batch_payloads,
        )
        _persist_development_root_batch(
            component_directory,
            claim=claim,
            component_plan=component_plan,
            content_manifest=content_manifest,
            payloads=payloads,
        )
        marker_sealed = True
        receipt = reveal_store._record_owned_development_sec_root_reader_output(
            development_root_scope_sha256=development_root_scope_sha256
        )
        if (
            type(receipt) is not dict
            or receipt.get("claim_sha256") != claim.get("claim_sha256")
            or (
                receipt.get("development_root_scope_sha256")
                != development_root_scope_sha256
            )
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned development SEC store receipt does not bind the claim"
            )
        return {"claim": claim, "reader_receipt": receipt}
    except Exception:
        reason = (
            "external_effect_failed_or_completion_unknown"
            if phase == "external_effect"
            else "durable_output_verification_failed"
        )
        if not marker_sealed:
            _attempt_development_root_abort(
                reveal_store,
                development_root_scope_sha256=development_root_scope_sha256,
                reason=reason,
            )
        raise SecFilingGemmaStageRunnerError(
            (
                "Owned development SEC root receipt finalization failed; "
                "the sealed marker remains recoverable without network I/O"
                if marker_sealed
                else "Owned development SEC root execution failed and will not be retried"
            )
        ) from None


def _model_component_directory(
    reveal_store: SecFilingGemmaRevealStore,
    claim: Mapping[str, Any],
) -> Path:
    """Create the fixed model component hierarchy exactly once."""

    claim_sha256 = claim.get("claim_sha256")
    event_count = claim.get("event_count")
    if (
        not _is_bare_sha256(claim_sha256)
        or type(event_count) is not int
        or not 1 <= event_count <= 9999
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned model claim has no safe fixed output identity"
        )
    claim_directory = (
        reveal_store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / claim_sha256
    )
    secured_claim_directory = _secure_directory(
        claim_directory,
        create=True,
        location="owned model claim directory",
    )
    if secured_claim_directory != claim_directory:
        raise SecFilingGemmaStageRunnerError(
            "Owned model claim directory identity changed"
        )
    component_directory = claim_directory / MODEL_EXTRACTION_COMPONENT_DIRECTORY_NAME
    try:
        component_directory.mkdir(exist_ok=False)
        _fsync_directory(claim_directory)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned model component directory is not create-new"
        ) from None
    secured_component = _secure_directory(
        component_directory,
        create=False,
        location="owned model component directory",
    )
    events_directory = component_directory / "events"
    try:
        events_directory.mkdir(exist_ok=False)
        _fsync_directory(component_directory)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned model events directory is not create-new"
        ) from None
    secured_events = _secure_directory(
        events_directory,
        create=False,
        location="owned model events directory",
    )
    for event_ordinal in range(1, event_count + 1):
        event_directory = events_directory / f"{event_ordinal:06d}"
        try:
            event_directory.mkdir(exist_ok=False)
            _fsync_directory(events_directory)
        except Exception:
            raise SecFilingGemmaStageRunnerError(
                "Owned model event directory is not create-new"
            ) from None
        if _secure_directory(
            event_directory,
            create=False,
            location=f"owned model event directory {event_ordinal}",
        ) != event_directory:
            raise SecFilingGemmaStageRunnerError(
                "Owned model event directory identity changed"
            )
    if secured_component != component_directory or secured_events != events_directory:
        raise SecFilingGemmaStageRunnerError(
            "Owned model component hierarchy identity changed"
        )
    return component_directory


def _model_json_payload(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Owned model durable artifact must be an exact JSON object"
        )
    return _canonical_marker_bytes(value)


def _persist_model_json(
    component_directory: Path,
    *,
    relative_path: str,
    logical_id: str,
    value: Mapping[str, Any],
    byte_index: list[dict[str, Any]],
) -> bytes:
    """Create and fsync one fixed model artifact, then extend its byte index."""

    parts = relative_path.split("/")
    if (
        not parts
        or len(parts) > 3
        or any(
            not part
            or part in {".", ".."}
            or "\\" in part
            for part in parts
        )
        or relative_path == MODEL_EXTRACTION_COMPLETE_MARKER_FILENAME
        or type(logical_id) is not str
        or not logical_id
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned model artifact path is unsafe"
        )
    parent = component_directory.joinpath(*parts[:-1])
    secured_parent = _secure_directory(
        parent,
        create=False,
        location=f"owned model artifact parent {logical_id}",
    )
    path = parent / parts[-1]
    payload = _model_json_payload(value)
    _write_new_regular_file(path, payload)
    _fsync_directory(secured_parent)
    byte_index.append(
        {
            "ordinal": len(byte_index) + 1,
            "logical_id": logical_id,
            "relative_path": relative_path,
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return payload


def _validated_model_claim(
    claim: Any,
    *,
    lifecycle_kind: str,
    lifecycle_sha256: str,
) -> dict[str, Any]:
    """Detach and validate the execution-critical model claim subset."""

    if type(claim) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Owned model claim is not an exact object"
        )
    if not _is_bare_sha256(lifecycle_sha256):
        raise SecFilingGemmaStageRunnerError(
            "Owned model lifecycle hash is not canonical"
        )
    lifecycle_field = (
        "development_root_scope_sha256"
        if lifecycle_kind == "development_root"
        else "request_sha256"
    )
    expected_stage = "development" if lifecycle_kind == "development_root" else None
    stage = claim.get("authorized_stage")
    event_plan = claim.get("event_plan")
    event_count = claim.get("event_count")
    execution_sources = claim.get("execution_source_hashes")
    limits = claim.get("model_runtime_limits")
    if (
        claim.get(lifecycle_field) != lifecycle_sha256
        or (expected_stage is not None and stage != expected_stage)
        or (
            lifecycle_kind == "stage_request"
            and stage not in {"intermediate", "final"}
        )
        or not _is_bare_sha256(claim.get("claim_sha256"))
        or canonical_sha256(
            {key: value for key, value in claim.items() if key != "claim_sha256"}
        )
        != claim.get("claim_sha256")
        or not _is_bare_sha256(claim.get("candidate_sha256"))
        or not _is_bare_sha256(claim.get("model_digest"))
        or not _is_bare_sha256(claim.get("runtime_fingerprint_sha256"))
        or not _is_bare_sha256(claim.get("execution_source_hashes_sha256"))
        or type(execution_sources) is not dict
        or canonical_sha256(execution_sources)
        != claim.get("execution_source_hashes_sha256")
        or not _is_bare_sha256(execution_sources.get("preprocessor"))
        or type(event_plan) is not list
        or type(event_count) is not int
        or not 1 <= event_count <= 9999
        or len(event_plan) != event_count
        or claim.get("event_plan_sha256") != canonical_sha256(event_plan)
        or type(limits) is not dict
        or limits.get("model_call_count") != event_count
        or any(
            limits.get(field) != 0
            for field in ("redirects", "retries", "pull_attempts", "repair_attempts")
        )
        or limits.get("streaming") is not False
        or limits.get("thinking") is not False
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned model claim is not canonically bound and bounded"
        )
    expected_lexicon_hash = canonical_sha256(list(CANONICAL_IDENTITY_LEXICON))
    if claim.get("identity_lexicon_sha256") != expected_lexicon_hash:
        raise SecFilingGemmaStageRunnerError(
            "Owned model claim changed the canonical identity lexicon"
        )
    exact_plan: list[dict[str, Any]] = []
    expected_keys = {
        "event_ordinal",
        "accession_number",
        "form",
        "availability_session",
        "sec_document_ordinal",
    }
    for ordinal, raw_event in enumerate(event_plan, start=1):
        if type(raw_event) is not dict or set(raw_event) != expected_keys:
            raise SecFilingGemmaStageRunnerError(
                "Owned model event plan is not exact"
            )
        event = dict(raw_event)
        if (
            event["event_ordinal"] != ordinal
            or type(event["sec_document_ordinal"]) is not int
            or not 1 <= event["sec_document_ordinal"] <= event_count
            or type(event["accession_number"]) is not str
            or type(event["availability_session"]) is not str
            or event["form"] not in {"10-K", "10-Q"}
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned model event plan identity is invalid"
            )
        exact_plan.append(event)
    if (
        exact_plan
        != sorted(
            exact_plan,
            key=lambda event: (
                event["availability_session"],
                event["accession_number"],
            ),
        )
        or {event["sec_document_ordinal"] for event in exact_plan}
        != set(range(1, event_count + 1))
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned model events are not in authoritative chronological order"
        )
    detached = json.loads(
        json.dumps(
            claim,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    if type(detached) is not dict:
        raise SecFilingGemmaStageRunnerError("Owned model claim could not detach")
    return detached


def _normalized_text_from_owned_bytes(
    payload: Any,
    descriptor: Any,
    *,
    location: str,
) -> str:
    if type(payload) is not bytes or not payload or type(descriptor) is not dict:
        raise SecFilingGemmaStageRunnerError(
            f"Owned model {location} normalized source is unavailable"
        )
    if set(descriptor) != {"relative_path", "byte_count", "sha256"}:
        raise SecFilingGemmaStageRunnerError(
            f"Owned model {location} source descriptor is not exact"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise SecFilingGemmaStageRunnerError(
            f"Owned model {location} normalized bytes are not UTF-8"
        ) from None
    if (
        text.encode("utf-8") != payload
        or descriptor["byte_count"] != len(payload)
        or descriptor["sha256"] != hashlib.sha256(payload).hexdigest()
    ):
        raise SecFilingGemmaStageRunnerError(
            f"Owned model {location} normalized bytes crossed their descriptor"
        )
    return text


def _validated_model_event_inputs(
    loaded: Any,
    *,
    claim: Mapping[str, Any],
    lifecycle_kind: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[str],
]:
    """Require store-selected current/prior bytes in exact claim chronology."""

    if type(loaded) is not dict or set(loaded) != {
        "claim",
        "scope_kind",
        "scope_sha256",
        "candidate_manifest",
        "corpus_universe_manifest",
        "content_manifests_by_stage",
        "session_dates",
        "sec_reader_receipt_sha256",
        "carry_in_reader_receipt_sha256",
        "events",
    }:
        raise SecFilingGemmaStageRunnerError(
            "Owned model event-input bundle is not exact"
        )
    if loaded["claim"] != claim:
        raise SecFilingGemmaStageRunnerError(
            "Owned model event inputs crossed the durable claim"
        )
    candidate = loaded["candidate_manifest"]
    universe = loaded["corpus_universe_manifest"]
    content_by_stage = loaded["content_manifests_by_stage"]
    session_dates = loaded["session_dates"]
    events = loaded["events"]
    if (
        type(candidate) is not dict
        or candidate.get("candidate_sha256") != claim["candidate_sha256"]
        or type(universe) is not dict
        or universe.get("universe_sha256") != claim["corpus_universe_sha256"]
        or type(content_by_stage) is not dict
        or not all(type(value) is dict for value in content_by_stage.values())
        or type(session_dates) is not list
        or not session_dates
        or not all(type(value) is str for value in session_dates)
        or type(events) is not list
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned model event-input evidence is unavailable"
        )
    if len(events) != claim["event_count"]:
        raise SecFilingGemmaStageRunnerError(
            "Owned model event-input count differs from the claim"
        )
    expected_sec_receipt = claim[
        "development_sec_reader_receipt_sha256"
        if lifecycle_kind == "development_root"
        else "stage_sec_reader_receipt_sha256"
    ]
    expected_carry_receipt = (
        None
        if lifecycle_kind == "development_root"
        else claim["carry_in_reader_receipt_sha256"]
    )
    lifecycle_sha256 = claim[
        "development_root_scope_sha256"
        if lifecycle_kind == "development_root"
        else "request_sha256"
    ]
    if (
        loaded["scope_kind"]
        != (
            "development_root"
            if lifecycle_kind == "development_root"
            else "stage_request"
        )
        or loaded["scope_sha256"] != lifecycle_sha256
        or loaded["sec_reader_receipt_sha256"] != expected_sec_receipt
        or loaded["carry_in_reader_receipt_sha256"] != expected_carry_receipt
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned model event-input bundle crossed its lifecycle ancestry"
        )
    allowed_stages = {
        "development": {"development"},
        "intermediate": {"development", "intermediate"},
        "final": {"development", "intermediate", "final"},
    }[claim["authorized_stage"]]
    if set(content_by_stage) != allowed_stages:
        raise SecFilingGemmaStageRunnerError(
            "Owned model content manifests do not cover exactly the allowed stages"
        )
    universe_records = universe.get("records")
    if type(universe_records) is not list:
        raise SecFilingGemmaStageRunnerError(
            "Owned model universe has no exact event records"
        )
    content_hash_by_accession: dict[str, str] = {}
    for manifest in content_by_stage.values():
        documents = manifest.get("documents")
        if type(documents) is not list:
            raise SecFilingGemmaStageRunnerError(
                "Owned model content manifest has no exact documents"
            )
        for document in documents:
            if (
                type(document) is not dict
                or type(document.get("accession_number")) is not str
                or not _is_bare_sha256(document.get("normalized_text_sha256"))
            ):
                raise SecFilingGemmaStageRunnerError(
                    "Owned model content identity is invalid"
                )
            content_hash_by_accession[document["accession_number"]] = document[
                "normalized_text_sha256"
            ]
    event_keys = {
        "event",
        "current_normalized_text",
        "current_normalized_source",
        "prior_same_form_normalized_text",
        "prior_same_form_normalized_source",
        "prior_accession_number",
        "prior_availability_session",
        "prior_provenance_kind",
        "sec_reader_receipt_sha256",
        "carry_in_reader_receipt_sha256",
    }
    last_same_form: dict[str, dict[str, Any]] = {}
    validated: list[dict[str, Any]] = []
    for event_plan, raw_event in zip(claim["event_plan"], events):
        if type(raw_event) is not dict or set(raw_event) != event_keys:
            raise SecFilingGemmaStageRunnerError(
                "Owned model event input is not an exact store result"
            )
        if raw_event["event"] != event_plan:
            raise SecFilingGemmaStageRunnerError(
                "Owned model event input crossed the chronological event plan"
            )
        if (
            raw_event["sec_reader_receipt_sha256"] != expected_sec_receipt
            or raw_event["carry_in_reader_receipt_sha256"]
            != expected_carry_receipt
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned model event input crossed its SEC or carry reader ancestry"
            )
        current_text = _normalized_text_from_owned_bytes(
            raw_event["current_normalized_text"],
            raw_event["current_normalized_source"],
            location="current",
        )
        expected_current_name = (
            f"document-{event_plan['sec_document_ordinal']:04d}.normalized.txt"
        )
        if raw_event["current_normalized_source"]["relative_path"] != expected_current_name:
            raise SecFilingGemmaStageRunnerError(
                "Owned model current source differs from its SEC document ordinal"
            )
        if (
            raw_event["current_normalized_source"]["sha256"]
            != content_hash_by_accession.get(event_plan["accession_number"])
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned model current bytes differ from stage content evidence"
            )
        prior_payload = raw_event["prior_same_form_normalized_text"]
        prior_source = raw_event["prior_same_form_normalized_source"]
        prior_kind = raw_event["prior_provenance_kind"]
        if prior_payload is None:
            if prior_source is not None or prior_kind is not None:
                raise SecFilingGemmaStageRunnerError(
                    "Owned model absent prior has inconsistent provenance"
                )
            prior_text = None
        else:
            prior_text = _normalized_text_from_owned_bytes(
                prior_payload,
                prior_source,
                location="prior",
            )
        preceding_records = [
            record
            for record in universe_records
            if type(record) is dict
            and record.get("form") == event_plan["form"]
            and (
                record.get("availability_session"),
                record.get("accession_number"),
            )
            < (
                event_plan["availability_session"],
                event_plan["accession_number"],
            )
        ]
        preceding_records.sort(
            key=lambda record: (
                record["availability_session"],
                record["accession_number"],
            )
        )
        expected_prior_record = preceding_records[-1] if preceding_records else None
        expected_prior_accession = (
            None
            if expected_prior_record is None
            else expected_prior_record["accession_number"]
        )
        expected_prior_availability = (
            None
            if expected_prior_record is None
            else expected_prior_record["availability_session"]
        )
        if (
            raw_event["prior_accession_number"] != expected_prior_accession
            or raw_event["prior_availability_session"]
            != expected_prior_availability
            or (
                prior_source is not None
                and prior_source["sha256"]
                != content_hash_by_accession.get(expected_prior_accession)
            )
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned model prior is not the universe's immediate same-form filing"
            )
        previous = last_same_form.get(event_plan["form"])
        if previous is not None:
            if (
                prior_kind != "same_scope_document"
                or prior_payload != previous["current_normalized_text"]
                or prior_source != previous["current_normalized_source"]
                or raw_event["prior_accession_number"]
                != previous["event"]["accession_number"]
                or raw_event["prior_availability_session"]
                != previous["event"]["availability_session"]
            ):
                raise SecFilingGemmaStageRunnerError(
                    "Owned model prior is not the immediate same-scope same-form filing"
                )
        elif lifecycle_kind == "development_root":
            if prior_payload is not None or expected_prior_record is not None:
                raise SecFilingGemmaStageRunnerError(
                    "Development model first same-form event cannot have a prior"
                )
        else:
            expected_kind = (
                "development_root_carry_in"
                if claim["authorized_stage"] == "intermediate"
                else "stage_carry_in"
            )
            if prior_payload is None or prior_kind != expected_kind:
                raise SecFilingGemmaStageRunnerError(
                    "Owned model first same-form prior is not stage-correct carry-in"
                )
        detached_event = dict(event_plan)
        detached_event["current_normalized_text"] = current_text
        detached_event["current_normalized_source"] = dict(
            raw_event["current_normalized_source"]
        )
        detached_event["prior_normalized_text"] = prior_text
        detached_event["prior_normalized_source"] = (
            None if prior_source is None else dict(prior_source)
        )
        detached_event["prior_accession_number"] = raw_event[
            "prior_accession_number"
        ]
        detached_event["prior_availability_session"] = raw_event[
            "prior_availability_session"
        ]
        detached_event["prior_provenance_kind"] = prior_kind
        detached_event["sec_reader_receipt_sha256"] = raw_event[
            "sec_reader_receipt_sha256"
        ]
        detached_event["carry_in_reader_receipt_sha256"] = raw_event[
            "carry_in_reader_receipt_sha256"
        ]
        validated.append(detached_event)
        last_same_form[event_plan["form"]] = raw_event
    return validated, candidate, universe, content_by_stage, list(session_dates)


def _model_call_intent(
    *,
    lifecycle_kind: str,
    lifecycle_sha256: str,
    claim: Mapping[str, Any],
    event: Mapping[str, Any],
    preprocessed_event: Mapping[str, Any],
    preprocessing_receipt: Mapping[str, Any],
    redacted_manifest: Mapping[str, Any],
    before_probe_receipt_sha256: str,
    before_runtime_evidence_sha256: str,
) -> dict[str, Any]:
    del lifecycle_kind, lifecycle_sha256
    event_identity = {
        key: event[key]
        for key in (
            "event_ordinal",
            "accession_number",
            "form",
            "availability_session",
            "sec_document_ordinal",
        )
    }
    return _build_owned_model_call_intent(
        claim=claim,
        event=event_identity,
        preprocessed_event_sha256=preprocessed_event[
            "preprocessed_event_sha256"
        ],
        owned_preprocessing_receipt_sha256=preprocessing_receipt[
            "receipt_sha256"
        ],
        redacted_input_manifest_sha256=redacted_manifest[
            "redacted_input_manifest_sha256"
        ],
        model_payload_sha256=preprocessed_event["model_payload_sha256"],
        runtime_probe_receipt_sha256=before_probe_receipt_sha256,
        runtime_evidence_sha256=before_runtime_evidence_sha256,
    )


def _validated_owned_runtime_probe(claim: Mapping[str, Any]) -> Any:
    receipt = probe_owned_ollama_runtime(
        expected_model_digest=claim["model_digest"],
        expected_runtime_fingerprint_sha256=claim[
            "runtime_fingerprint_sha256"
        ],
    )
    manifest = receipt.to_manifest()
    replayed = validate_ollama_runtime_probe_receipt(
        manifest,
        expected_model_digest=claim["model_digest"],
        expected_runtime_fingerprint_sha256=claim[
            "runtime_fingerprint_sha256"
        ],
        expected_transport_mode=_OWNED_RUNTIME_PROBE_TRANSPORT_MODE,
        expected_probe_receipt_sha256=manifest["receipt_sha256"],
    )
    if replayed.to_manifest() != manifest:
        raise SecFilingGemmaStageRunnerError(
            "Owned runtime probe changed during immediate replay"
        )
    return replayed


def _validated_extractor_request_from_owned_inputs(
    *,
    claim: Mapping[str, Any],
    event: Mapping[str, Any],
    preprocessed_event: Mapping[str, Any],
    preprocessing_receipt: Mapping[str, Any],
    redacted_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
    universe_manifest: Mapping[str, Any],
    content_manifests_by_stage: Mapping[str, Mapping[str, Any]],
    session_dates: list[str],
) -> dict[str, Any]:
    """Build the full metadata envelope, then cross the contract boundary."""

    request = {
        "request_version": EXTRACTOR_REQUEST_VERSION,
        "preprocessor_version": PREPROCESSOR_VERSION,
        "corpus_universe_sha256": claim["corpus_universe_sha256"],
        "identity_lexicon_sha256": claim["identity_lexicon_sha256"],
        "redacted_input_manifest_sha256": redacted_manifest[
            "redacted_input_manifest_sha256"
        ],
        "stage": claim["authorized_stage"],
        "current_accession_number": event["accession_number"],
        "current_form": event["form"],
        "current_availability_session": event["availability_session"],
        "current_filing_sha256": event["current_normalized_source"]["sha256"],
        "prior_accession_number": event["prior_accession_number"],
        "prior_availability_session": event["prior_availability_session"],
        "prior_same_form_filing_sha256": (
            None
            if event["prior_normalized_source"] is None
            else event["prior_normalized_source"]["sha256"]
        ),
        "model_payload": preprocessed_event["model_payload"],
        "model_payload_sha256": preprocessed_event["model_payload_sha256"],
        "redaction_report": preprocessed_event["redaction_report"],
    }
    expected_content_hashes = {
        stage: manifest["content_manifest_sha256"]
        for stage, manifest in content_manifests_by_stage.items()
    }
    validated = validate_extractor_request(
        request,
        candidate_manifest=candidate_manifest,
        expected_candidate_sha256=claim["candidate_sha256"],
        universe_manifest=universe_manifest,
        content_manifests_by_stage=content_manifests_by_stage,
        expected_content_manifest_sha256s=expected_content_hashes,
        session_dates=session_dates,
        forbidden_identity_terms=CANONICAL_IDENTITY_LEXICON,
        redacted_input_manifest=redacted_manifest,
        expected_redacted_input_manifest_sha256=redacted_manifest[
            "redacted_input_manifest_sha256"
        ],
        expected_preprocessed_event_sha256=preprocessed_event[
            "preprocessed_event_sha256"
        ],
        expected_owned_preprocessing_receipt_sha256=preprocessing_receipt[
            "receipt_sha256"
        ],
        expected_sec_reader_receipt_sha256=event[
            "sec_reader_receipt_sha256"
        ],
        expected_carry_in_reader_receipt_sha256=event[
            "carry_in_reader_receipt_sha256"
        ],
    )
    if type(validated) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Contract validator did not return an exact safe model request"
        )
    return validated


def _persist_model_complete_marker(
    component_directory: Path,
    *,
    lifecycle_kind: str,
    lifecycle_sha256: str,
    claim: Mapping[str, Any],
    byte_index: list[dict[str, Any]],
) -> None:
    schema_version = (
        DEVELOPMENT_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
        if lifecycle_kind == "development_root"
        else STAGE_MODEL_EXTRACTION_COMPLETE_MARKER_SCHEMA_VERSION
    )
    marker_body = {
        "schema_version": schema_version,
        "lifecycle_kind": (
            "development_root_scope"
            if lifecycle_kind == "development_root"
            else "stage_request"
        ),
        "lifecycle_sha256": lifecycle_sha256,
        "claim_sha256": claim["claim_sha256"],
        "candidate_sha256": claim["candidate_sha256"],
        "authorized_stage": claim["authorized_stage"],
        "output_namespace": claim["output_namespace"],
        "model_component_id": claim["model_component_id"],
        "model_name": claim["model_name"],
        "model_digest": claim["model_digest"],
        "runtime_fingerprint_sha256": claim["runtime_fingerprint_sha256"],
        "event_count": claim["event_count"],
        "event_plan_sha256": claim["event_plan_sha256"],
        "identity_lexicon_sha256": claim["identity_lexicon_sha256"],
        "execution_source_hashes_sha256": claim[
            "execution_source_hashes_sha256"
        ],
        "model_transport_sha256": claim["model_transport_sha256"],
        "model_runtime_limits_sha256": claim[
            "model_runtime_limits_sha256"
        ],
        "byte_index": byte_index,
        "byte_index_sha256": canonical_sha256(byte_index),
        "byte_count_total": sum(item["byte_count"] for item in byte_index),
    }
    marker = {**marker_body, "marker_sha256": canonical_sha256(marker_body)}
    _write_new_regular_file(
        component_directory / MODEL_EXTRACTION_COMPLETE_MARKER_FILENAME,
        _model_json_payload(marker),
    )
    _fsync_directory(component_directory)


def _execute_owned_model_batch(
    reveal_store: SecFilingGemmaRevealStore,
    *,
    lifecycle_kind: str,
    lifecycle_sha256: str,
    claim: Mapping[str, Any],
    loaded_inputs: Mapping[str, Any],
) -> None:
    (
        events,
        candidate,
        universe,
        content_by_stage,
        session_dates,
    ) = _validated_model_event_inputs(
        loaded_inputs,
        claim=claim,
        lifecycle_kind=lifecycle_kind,
    )
    component_directory = _model_component_directory(reveal_store, claim)
    byte_index: list[dict[str, Any]] = []

    reveal_store._revalidate_authorized_model_execution_sources(claim)
    before_probe = _validated_owned_runtime_probe(claim)
    before_manifest = before_probe.to_manifest()
    _persist_model_json(
        component_directory,
        relative_path="pre_runtime_probe.json",
        logical_id="pre-runtime-probe",
        value=before_manifest,
        byte_index=byte_index,
    )
    runtime_identity = before_probe.pinned_runtime_identity()
    attempt_receipt_sha256s: list[str] = []
    elapsed_nanoseconds = 0
    for event in events:
        ordinal = event["event_ordinal"]
        prefix = f"events/{ordinal:06d}"
        logical_prefix = f"event-{ordinal:06d}"
        preprocessed = preprocess_filing_event(
            current_normalized_text=event["current_normalized_text"],
            prior_same_form_normalized_text=event["prior_normalized_text"],
            identity_lexicon=CANONICAL_IDENTITY_LEXICON,
        )
        scope_kind = (
            "development_root"
            if lifecycle_kind == "development_root"
            else "stage_request"
        )
        preprocessing_receipt = build_owned_preprocessing_receipt(
            scope_kind=scope_kind,
            scope_sha256=lifecycle_sha256,
            candidate_sha256=claim["candidate_sha256"],
            model_execution_claim_sha256=claim["claim_sha256"],
            sec_reader_receipt_sha256=event["sec_reader_receipt_sha256"],
            carry_in_reader_receipt_sha256=event[
                "carry_in_reader_receipt_sha256"
            ],
            stage=claim["authorized_stage"],
            event_ordinal=ordinal,
            accession_number=event["accession_number"],
            form=event["form"],
            current_normalized_source=event["current_normalized_source"],
            prior_same_form_normalized_source=event["prior_normalized_source"],
            prior_provenance_kind=event["prior_provenance_kind"],
            preprocessor_source_sha256=claim["execution_source_hashes"][
                "preprocessor"
            ],
            preprocessed_event=preprocessed,
            current_normalized_text=event["current_normalized_text"],
            prior_same_form_normalized_text=event["prior_normalized_text"],
        )
        redacted_manifest = build_redacted_input_manifest(
            artifact_stage=claim["authorized_stage"],
            accession_number=event["accession_number"],
            corpus_universe_sha256=claim["corpus_universe_sha256"],
            model_payload_sha256=preprocessed["model_payload_sha256"],
            preprocessed_event_sha256=preprocessed["preprocessed_event_sha256"],
            owned_preprocessing_receipt_sha256=preprocessing_receipt[
                "receipt_sha256"
            ],
            sec_reader_receipt_sha256=event["sec_reader_receipt_sha256"],
            carry_in_reader_receipt_sha256=event[
                "carry_in_reader_receipt_sha256"
            ],
            universe_manifest=universe,
            stage_content_manifest=content_by_stage[claim["authorized_stage"]],
        )
        for logical_suffix, filename, artifact in (
            ("preprocessed-event", "preprocessed_event.json", preprocessed),
            (
                "preprocessing-receipt",
                "preprocessing_receipt.json",
                preprocessing_receipt,
            ),
            (
                "redacted-input-manifest",
                "redacted_input_manifest.json",
                redacted_manifest,
            ),
        ):
            _persist_model_json(
                component_directory,
                relative_path=f"{prefix}/{filename}",
                logical_id=f"{logical_prefix}-{logical_suffix}",
                value=artifact,
                byte_index=byte_index,
            )
        intent = _model_call_intent(
            lifecycle_kind=lifecycle_kind,
            lifecycle_sha256=lifecycle_sha256,
            claim=claim,
            event=event,
            preprocessed_event=preprocessed,
            preprocessing_receipt=preprocessing_receipt,
            redacted_manifest=redacted_manifest,
            before_probe_receipt_sha256=before_manifest["receipt_sha256"],
            before_runtime_evidence_sha256=runtime_identity.evidence_sha256,
        )
        _persist_model_json(
            component_directory,
            relative_path=f"{prefix}/call_intent.json",
            logical_id=f"{logical_prefix}-call-intent",
            value=intent,
            byte_index=byte_index,
        )
        reveal_store._revalidate_authorized_model_execution_sources(claim)
        validated_request = _validated_extractor_request_from_owned_inputs(
            claim=claim,
            event=event,
            preprocessed_event=preprocessed,
            preprocessing_receipt=preprocessing_receipt,
            redacted_manifest=redacted_manifest,
            candidate_manifest=candidate,
            universe_manifest=universe,
            content_manifests_by_stage=content_by_stage,
            session_dates=session_dates,
        )
        attempt = call_ollama_extractor_attempt(
            validated_request,
            expected_candidate_sha256=claim["candidate_sha256"],
            expected_model_payload_sha256=preprocessed[
                "model_payload_sha256"
            ],
            expected_runtime_evidence_sha256=runtime_identity.evidence_sha256,
            expected_model_digest=claim["model_digest"],
            expected_runtime_fingerprint_sha256=claim[
                "runtime_fingerprint_sha256"
            ],
            runtime_identity=runtime_identity,
        )
        attempt_manifest = attempt.to_manifest()
        replayed_attempt = validate_ollama_model_attempt_receipt(
            attempt_manifest,
            expected_candidate_sha256=claim["candidate_sha256"],
            expected_model_payload_sha256=preprocessed[
                "model_payload_sha256"
            ],
            expected_sentence_ids=validated_request["sentence_ids"],
            expected_runtime_evidence_sha256=runtime_identity.evidence_sha256,
            expected_model_digest=claim["model_digest"],
            expected_runtime_fingerprint_sha256=claim[
                "runtime_fingerprint_sha256"
            ],
            expected_transport_mode=_OWNED_MODEL_ATTEMPT_TRANSPORT_MODE,
        )
        if replayed_attempt.to_manifest() != attempt_manifest:
            raise SecFilingGemmaStageRunnerError(
                "Owned model attempt changed during immediate replay"
            )
        elapsed_nanoseconds += replayed_attempt.elapsed_nanoseconds
        maximum_seconds = claim["model_runtime_limits"].get(
            "maximum_model_seconds"
        )
        if (
            type(maximum_seconds) not in {int, float}
            or isinstance(maximum_seconds, bool)
            or maximum_seconds <= 0
            or elapsed_nanoseconds > int(float(maximum_seconds) * 1_000_000_000)
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned model batch exceeded its fixed diagnostic time ceiling"
            )
        _persist_model_json(
            component_directory,
            relative_path=f"{prefix}/model_attempt_receipt.json",
            logical_id=f"{logical_prefix}-model-attempt-receipt",
            value=attempt_manifest,
            byte_index=byte_index,
        )
        attempt_receipt_sha256s.append(attempt_manifest["receipt_sha256"])

    reveal_store._revalidate_authorized_model_execution_sources(claim)
    after_probe = _validated_owned_runtime_probe(claim)
    after_manifest = after_probe.to_manifest()
    _persist_model_json(
        component_directory,
        relative_path="post_runtime_probe.json",
        logical_id="post-runtime-probe",
        value=after_manifest,
        byte_index=byte_index,
    )
    runtime_guard = build_runtime_identity_guard(
        before_evidence=before_probe.runtime_evidence(),
        after_evidence=after_probe.runtime_evidence(),
        expected_model_digest=claim["model_digest"],
        expected_runtime_fingerprint_sha256=claim[
            "runtime_fingerprint_sha256"
        ],
        stage=claim["authorized_stage"],
        model_call_receipt_sha256s=attempt_receipt_sha256s,
    )
    _persist_model_json(
        component_directory,
        relative_path="runtime_guard.json",
        logical_id="runtime-guard",
        value=runtime_guard,
        byte_index=byte_index,
    )
    reveal_store._revalidate_authorized_model_execution_sources(claim)
    _persist_model_complete_marker(
        component_directory,
        lifecycle_kind=lifecycle_kind,
        lifecycle_sha256=lifecycle_sha256,
        claim=claim,
        byte_index=byte_index,
    )


def _attempt_model_abort(
    reveal_store: SecFilingGemmaRevealStore,
    *,
    lifecycle_kind: str,
    lifecycle_sha256: str,
    reason: str,
) -> None:
    try:
        if lifecycle_kind == "development_root":
            reveal_store.abort_owned_development_model_execution(
                development_root_scope_sha256=lifecycle_sha256,
                reason=reason,
            )
        else:
            reveal_store.abort_authorized_model_stage_execution(
                request_sha256=lifecycle_sha256,
                reason=reason,
            )
    except Exception:
        return


def _run_owned_model_batch_locked(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    lifecycle_kind: str,
    lifecycle_sha256: str,
) -> dict[str, Any]:
    if lifecycle_kind == "development_root":
        claim_result = reveal_store.claim_owned_development_model_execution(
            development_root_scope_sha256=lifecycle_sha256
        )
        record = reveal_store._record_owned_development_model_reader_output
        loader = reveal_store._load_owned_development_model_event_inputs
        record_kwargs = {"development_root_scope_sha256": lifecycle_sha256}
    else:
        claim_result = reveal_store.claim_authorized_model_stage_execution(
            request_sha256=lifecycle_sha256
        )
        record = reveal_store._record_authorized_model_stage_reader_output
        loader = reveal_store._load_authorized_model_stage_event_inputs
        record_kwargs = {"request_sha256": lifecycle_sha256}
    if type(claim_result) is not dict or set(claim_result) != {
        "claim",
        "created",
        "reader_receipt",
        "abort",
    }:
        raise SecFilingGemmaStageRunnerError(
            "Owned model claim result is not exact"
        )
    claim = _validated_model_claim(
        claim_result["claim"],
        lifecycle_kind=lifecycle_kind,
        lifecycle_sha256=lifecycle_sha256,
    )
    if claim_result["created"] is False:
        receipt = claim_result["reader_receipt"]
        abort = claim_result["abort"]
        if type(receipt) is dict and abort is None:
            if receipt.get("claim_sha256") != claim["claim_sha256"]:
                raise SecFilingGemmaStageRunnerError(
                    "Completed model receipt crossed its durable claim"
                )
            try:
                replayed = record(**record_kwargs)
            except Exception:
                raise SecFilingGemmaStageRunnerError(
                    "Completed model receipt cannot replay its durable output"
                ) from None
            if replayed != receipt:
                raise SecFilingGemmaStageRunnerError(
                    "Completed model receipt differs from durable replay"
                )
            return {"claim": claim, "reader_receipt": replayed}
        if receipt is None and abort is None:
            try:
                recovered = record(**record_kwargs)
                if (
                    type(recovered) is not dict
                    or recovered.get("claim_sha256") != claim["claim_sha256"]
                ):
                    raise SecFilingGemmaStageRunnerError(
                        "Recovered model receipt crossed its durable claim"
                    )
            except Exception:
                _attempt_model_abort(
                    reveal_store,
                    lifecycle_kind=lifecycle_kind,
                    lifecycle_sha256=lifecycle_sha256,
                    reason="claim_recovered_without_terminal_receipt",
                )
            else:
                return {"claim": claim, "reader_receipt": recovered}
        raise SecFilingGemmaStageRunnerError(
            "Owned model claim is terminal or indeterminate and cannot be retried"
        )
    if claim_result["created"] is not True or any(
        claim_result[key] is not None for key in ("reader_receipt", "abort")
    ):
        raise SecFilingGemmaStageRunnerError(
            "New owned model claim has an impossible terminal state"
        )

    marker_sealed = False
    external_effect_started = False
    try:
        reveal_store._revalidate_authorized_model_execution_sources(claim)
        loaded_inputs = loader(**record_kwargs)
        external_effect_started = True
        _execute_owned_model_batch(
            reveal_store,
            lifecycle_kind=lifecycle_kind,
            lifecycle_sha256=lifecycle_sha256,
            claim=claim,
            loaded_inputs=loaded_inputs,
        )
        marker_sealed = True
        receipt = record(**record_kwargs)
        if (
            type(receipt) is not dict
            or receipt.get("claim_sha256") != claim["claim_sha256"]
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned model store receipt does not bind the execution claim"
            )
        return {"claim": claim, "reader_receipt": receipt}
    except Exception:
        if not marker_sealed:
            _attempt_model_abort(
                reveal_store,
                lifecycle_kind=lifecycle_kind,
                lifecycle_sha256=lifecycle_sha256,
                reason=(
                    "external_effect_failed_or_completion_unknown"
                    if external_effect_started
                    else "durable_output_verification_failed"
                ),
            )
        raise SecFilingGemmaStageRunnerError(
            (
                "Owned model receipt finalization failed; the sealed marker "
                "remains recoverable with zero model calls"
                if marker_sealed
                else "Owned model execution failed and will not be retried"
            )
        ) from None


def _run_owned_model_batch(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    lifecycle_kind: str,
    lifecycle_sha256: str,
) -> dict[str, Any]:
    if type(reveal_store) is not SecFilingGemmaRevealStore:
        raise TypeError("reveal_store must be the owned reveal-store implementation")
    if not _is_bare_sha256(lifecycle_sha256):
        raise SecFilingGemmaStageRunnerError(
            "Owned model lifecycle hash must be a bare lowercase SHA-256"
        )
    try:
        execution_lock = reveal_store._owned_model_execution_lock()
        execution_lock.__enter__()
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned model execution is already globally owned or cannot be locked"
        ) from None
    try:
        return _run_owned_model_batch_locked(
            reveal_store=reveal_store,
            lifecycle_kind=lifecycle_kind,
            lifecycle_sha256=lifecycle_sha256,
        )
    finally:
        execution_lock.__exit__(None, None, None)


def run_owned_development_model_batch(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    """Run or zero-call recover the one development-root Gemma batch."""

    return _run_owned_model_batch(
        reveal_store=reveal_store,
        lifecycle_kind="development_root",
        lifecycle_sha256=development_root_scope_sha256,
    )


def run_owned_stage_model_batch(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    request_sha256: str,
) -> dict[str, Any]:
    """Run or zero-call recover one authorized intermediate/final Gemma batch."""

    return _run_owned_model_batch(
        reveal_store=reveal_store,
        lifecycle_kind="stage_request",
        lifecycle_sha256=request_sha256,
    )


_MARKET_RAW_FILENAMES: Final[dict[str, str]] = {
    symbol: f"raw-response-{symbol}.json" for symbol in MARKET_SYMBOLS
}
_MARKET_ARTIFACT_FILENAMES: Final[dict[str, str]] = {
    symbol: f"artifact-{symbol}.json" for symbol in MARKET_SYMBOLS
}
_MARKET_WINDOW_FILENAMES: Final[dict[str, str]] = {
    symbol: f"window-{symbol}.json" for symbol in MARKET_SYMBOLS
}


def _validated_market_claim(
    claim: Any,
    *,
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    if type(claim) is not dict or not _is_bare_sha256(
        development_root_scope_sha256
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned market claim is not an exact development claim"
        )
    claim_sha256 = claim.get("claim_sha256")
    plan = claim.get("market_acquisition_plan")
    plan_sha256 = claim.get("market_acquisition_plan_sha256")
    if (
        not _is_bare_sha256(claim_sha256)
        or canonical_sha256(
            {key: value for key, value in claim.items() if key != "claim_sha256"}
        )
        != claim_sha256
        or claim.get("development_root_scope_sha256")
        != development_root_scope_sha256
        or claim.get("authorized_stage") != "development"
        or claim.get("market_component_id") != DEVELOPMENT_MARKET_COMPONENT_ID
        or type(plan) is not dict
        or not _is_bare_sha256(plan_sha256)
        or plan.get("acquisition_plan_sha256") != plan_sha256
        or canonical_sha256(
            {
                key: value
                for key, value in plan.items()
                if key != "acquisition_plan_sha256"
            }
        )
        != plan_sha256
        or claim.get("market_symbols") != list(MARKET_SYMBOLS)
        or claim.get("fixed_request_count") != len(MARKET_SYMBOLS)
        or claim.get("owned_market_execution_required") is not True
        or claim.get("market_access_permitted") is not True
        or claim.get("external_network_access_permitted") is not True
        or claim.get("caller_supplied_path_permitted") is not False
        or claim.get("caller_supplied_bytes_permitted") is not False
        or claim.get("paid_api_access_permitted") is not False
        or claim.get("outcome_access_permitted") is not False
        or claim.get("future_stage_access_permitted") is not False
        or claim.get("effect_may_be_repeated_after_indeterminate_crash") is not False
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned market claim crossed its fixed authority"
        )
    return claim


def _validated_market_reader_receipt(
    receipt: Any,
    *,
    claim: Mapping[str, Any],
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    if type(receipt) is not dict:
        raise SecFilingGemmaStageRunnerError(
            "Owned market reader receipt is not an exact object"
        )
    receipt_sha256 = receipt.get("receipt_sha256")
    if (
        not _is_bare_sha256(receipt_sha256)
        or canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        != receipt_sha256
        or receipt.get("claim_sha256") != claim["claim_sha256"]
        or receipt.get("development_root_scope_sha256")
        != development_root_scope_sha256
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned market reader receipt crossed its durable claim"
        )
    return receipt


def _market_component_directory(
    reveal_store: SecFilingGemmaRevealStore,
    claim: Mapping[str, Any],
) -> Path:
    """Create the fixed claim-owned market quarantine exactly once."""

    claim_sha256 = claim.get("claim_sha256")
    if not _is_bare_sha256(claim_sha256):
        raise SecFilingGemmaStageRunnerError(
            "Owned market claim has no safe output identity"
        )
    claim_directory = (
        reveal_store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / claim_sha256
    )
    if _secure_directory(
        claim_directory,
        create=True,
        location="owned market claim directory",
    ) != claim_directory:
        raise SecFilingGemmaStageRunnerError(
            "Owned market claim directory identity changed"
        )
    component_directory = claim_directory / MARKET_SOURCE_COMPONENT_DIRECTORY_NAME
    try:
        component_directory.mkdir(exist_ok=False)
        _fsync_directory(claim_directory)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned market component directory is not create-new"
        ) from None
    if _secure_directory(
        component_directory,
        create=False,
        location="owned market component directory",
    ) != component_directory:
        raise SecFilingGemmaStageRunnerError(
            "Owned market component directory identity changed"
        )
    return component_directory


def _market_payloads(bundle: Mapping[str, Any]) -> list[tuple[str, str, bytes]]:
    raw = bundle["raw_response_bytes_by_symbol"]
    artifacts = bundle["artifact_bytes_by_symbol"]
    windows = bundle["window_bytes_by_symbol"]
    payloads: list[tuple[str, str, bytes]] = []
    for symbol in MARKET_SYMBOLS:
        payloads.append(
            (
                f"raw-response-{symbol}",
                _MARKET_RAW_FILENAMES[symbol],
                raw[symbol],
            )
        )
    for symbol in MARKET_SYMBOLS:
        payloads.append(
            (
                f"artifact-{symbol}",
                _MARKET_ARTIFACT_FILENAMES[symbol],
                artifacts[symbol],
            )
        )
    for symbol in MARKET_SYMBOLS:
        payloads.append(
            (
                f"window-{symbol}",
                _MARKET_WINDOW_FILENAMES[symbol],
                windows[symbol],
            )
        )
    payloads.extend(
        (
            (
                "source-manifest",
                "source-manifest.json",
                _canonical_marker_bytes(bundle["source_manifest"]),
            ),
            (
                "stage-manifest",
                "stage-manifest.json",
                _canonical_marker_bytes(bundle["stage_manifest"]),
            ),
            (
                "reconciliation-receipt",
                "reconciliation-receipt.json",
                _canonical_marker_bytes(bundle["reconciliation_receipt"]),
            ),
            (
                "acquisition-receipt",
                "acquisition-receipt.json",
                _canonical_marker_bytes(bundle["acquisition_receipt"]),
            ),
        )
    )
    names = [name.casefold() for _logical_id, name, _payload in payloads]
    total_bytes = sum(len(payload) for _logical_id, _name, payload in payloads)
    if (
        len(payloads) + 1 != MAX_MARKET_BATCH_FILES
        or len(names) != len(set(names))
        or any(
            type(payload) is not bytes
            or not payload
            or len(payload) > MAX_MARKET_BATCH_FILE_BYTES
            for _logical_id, _name, payload in payloads
        )
        or total_bytes > MAX_MARKET_BATCH_TOTAL_BYTES
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned market evidence exceeds its fixed flat layout"
        )
    return payloads


def _persist_market_batch(
    component_directory: Path,
    *,
    claim: Mapping[str, Any],
    bundle: Mapping[str, Any],
    acquisition_validation_sha256: str,
    marker_state: list[bool],
) -> None:
    if type(marker_state) is not list or marker_state != [False]:
        raise SecFilingGemmaStageRunnerError(
            "Owned market marker state must begin unsealed"
        )
    payloads = _market_payloads(bundle)
    byte_index: list[dict[str, Any]] = []
    for ordinal, (logical_id, name, payload) in enumerate(payloads, start=1):
        if (
            not name
            or name in {".", "..", DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME}
            or "/" in name
            or "\\" in name
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned market artifact filename is unsafe"
            )
        _write_new_regular_file(component_directory / name, payload)
        byte_index.append(
            {
                "ordinal": ordinal,
                "logical_id": logical_id,
                "relative_path": name,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    marker_body = {
        "schema_version": DEVELOPMENT_MARKET_COMPLETE_MARKER_SCHEMA_VERSION,
        "development_root_scope_sha256": claim[
            "development_root_scope_sha256"
        ],
        "claim_sha256": claim["claim_sha256"],
        "market_acquisition_plan_sha256": claim[
            "market_acquisition_plan_sha256"
        ],
        "acquisition_receipt_sha256": bundle[
            "acquisition_receipt_sha256"
        ],
        "acquisition_bundle_sha256": bundle["bundle_sha256"],
        "acquisition_validation_sha256": acquisition_validation_sha256,
        "source_manifest_sha256": bundle["source_manifest_sha256"],
        "market_stage_manifest_sha256": bundle[
            "market_stage_manifest_sha256"
        ],
        "source_reconciliation_sha256": bundle[
            "source_reconciliation_sha256"
        ],
        "market_component_id": DEVELOPMENT_MARKET_COMPONENT_ID,
        "byte_index": byte_index,
        "byte_index_sha256": canonical_sha256(byte_index),
        "byte_count_total": sum(item["byte_count"] for item in byte_index),
    }
    marker = {**marker_body, "marker_sha256": canonical_sha256(marker_body)}
    marker_payload = _canonical_marker_bytes(marker)
    if (
        len(marker_payload) > MAX_MARKET_BATCH_FILE_BYTES
        or marker_body["byte_count_total"] + len(marker_payload)
        > MAX_MARKET_BATCH_TOTAL_BYTES
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned market completion marker exceeds its fixed byte limits"
        )
    _write_new_regular_file(
        component_directory / DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME,
        marker_payload,
    )
    _fsync_directory(component_directory)
    marker_state[0] = True
    observed = [entry.name for entry in component_directory.iterdir()]
    expected = [name for _logical_id, name, _payload in payloads] + [
        DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME
    ]
    if (
        set(observed) != set(expected)
        or len(observed) != len(expected)
        or len({name.casefold() for name in observed}) != len(observed)
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned market evidence has missing, extra, or colliding files"
        )


def _attempt_market_abort(
    reveal_store: SecFilingGemmaRevealStore,
    *,
    development_root_scope_sha256: str,
    reason: str,
) -> None:
    try:
        reveal_store.abort_owned_development_market_execution(
            development_root_scope_sha256=development_root_scope_sha256,
            reason=reason,
        )
    except Exception:
        return


def _run_owned_development_market_batch_locked(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    claim_result = reveal_store.claim_owned_development_market_execution(
        development_root_scope_sha256=development_root_scope_sha256
    )
    if type(claim_result) is not dict or set(claim_result) != {
        "claim",
        "created",
        "reader_receipt",
        "abort",
    }:
        raise SecFilingGemmaStageRunnerError(
            "Owned market claim result is not exact"
        )
    claim = _validated_market_claim(
        claim_result["claim"],
        development_root_scope_sha256=development_root_scope_sha256,
    )
    record_kwargs = {
        "development_root_scope_sha256": development_root_scope_sha256
    }
    if claim_result["created"] is False:
        receipt = claim_result["reader_receipt"]
        abort = claim_result["abort"]
        if type(receipt) is dict and abort is None:
            validated_receipt = _validated_market_reader_receipt(
                receipt,
                claim=claim,
                development_root_scope_sha256=development_root_scope_sha256,
            )
            try:
                replayed = (
                    reveal_store._record_owned_development_market_reader_output(
                        **record_kwargs
                    )
                )
            except Exception:
                raise SecFilingGemmaStageRunnerError(
                    "Completed market receipt cannot replay durable output"
                ) from None
            replayed = _validated_market_reader_receipt(
                replayed,
                claim=claim,
                development_root_scope_sha256=development_root_scope_sha256,
            )
            if replayed != validated_receipt:
                raise SecFilingGemmaStageRunnerError(
                    "Completed market receipt differs from durable replay"
                )
            return {"claim": claim, "reader_receipt": replayed}
        if receipt is None and abort is None:
            _attempt_market_abort(
                reveal_store,
                development_root_scope_sha256=development_root_scope_sha256,
                reason="claim_recovered_without_terminal_receipt",
            )
        raise SecFilingGemmaStageRunnerError(
            "Owned market claim is terminal or indeterminate and cannot be retried"
        )
    if claim_result["created"] is not True or any(
        claim_result[key] is not None for key in ("reader_receipt", "abort")
    ):
        raise SecFilingGemmaStageRunnerError(
            "New owned market claim has an impossible terminal state"
        )

    external_effect_started = False
    try:
        component_directory = _market_component_directory(reveal_store, claim)
        reveal_store._revalidate_authorized_market_execution_sources(claim)
        external_effect_started = True
        owned_acquisition = _acquire_owned_development_market_evidence()
        bundle, owned_transport_capability = (
            _unwrap_owned_development_market_acquisition(owned_acquisition)
        )
        if (
            type(bundle) is not dict
            or bundle.get("acquisition_plan_sha256")
            != claim["market_acquisition_plan_sha256"]
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned market bundle crossed its claimed plan"
            )
        _bind_owned_market_transport_capability_to_claim(
            owned_transport_capability,
            development_root_scope_sha256=development_root_scope_sha256,
            claim_sha256=claim["claim_sha256"],
        )
        validation = validate_development_market_acquisition_bundle(
            bundle,
            expected_acquisition_plan_sha256=claim[
                "market_acquisition_plan_sha256"
            ],
            expected_acquisition_receipt_sha256=bundle[
                "acquisition_receipt_sha256"
            ],
            expected_bundle_sha256=bundle["bundle_sha256"],
        )
        validation_sha256 = validation.get("validation_sha256")
        if not _is_bare_sha256(validation_sha256):
            raise SecFilingGemmaStageRunnerError(
                "Owned market bundle validation is not exact"
        )
        reveal_store._revalidate_authorized_market_execution_sources(claim)
        marker_state = [False]
        _persist_market_batch(
            component_directory,
            claim=claim,
            bundle=bundle,
            acquisition_validation_sha256=validation_sha256,
            marker_state=marker_state,
        )
        receipt = reveal_store._record_owned_development_market_reader_output(
            owned_transport_capability=owned_transport_capability,
            **record_kwargs
        )
        receipt = _validated_market_reader_receipt(
            receipt,
            claim=claim,
            development_root_scope_sha256=development_root_scope_sha256,
        )
        return {"claim": claim, "reader_receipt": receipt}
    except Exception:
        _attempt_market_abort(
            reveal_store,
            development_root_scope_sha256=development_root_scope_sha256,
            reason=(
                "external_effect_failed_or_completion_unknown"
                if external_effect_started
                else "durable_output_verification_failed"
            ),
        )
        raise SecFilingGemmaStageRunnerError(
            "Owned market execution failed and will not be retried"
        ) from None


def run_owned_development_market_batch(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    """Run once, or replay only an already-committed market reader receipt."""

    if type(reveal_store) is not SecFilingGemmaRevealStore:
        raise TypeError("reveal_store must be the owned reveal-store implementation")
    if not _is_bare_sha256(development_root_scope_sha256):
        raise SecFilingGemmaStageRunnerError(
            "Development market scope must be a bare lowercase SHA-256"
        )
    try:
        execution_lock = reveal_store._owned_market_execution_lock()
        execution_lock.__enter__()
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned market execution is already globally owned or cannot be locked"
        ) from None
    try:
        return _run_owned_development_market_batch_locked(
            reveal_store=reveal_store,
            development_root_scope_sha256=development_root_scope_sha256,
        )
    finally:
        execution_lock.__exit__(None, None, None)


def _validated_owned_development_feature_inputs(
    loaded: Any,
    *,
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    """Require the exact store-owned, causal feature projection."""

    if type(loaded) is not dict or set(loaded) != set(
        _OWNED_DEVELOPMENT_FEATURE_INPUT_KEYS
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature inputs are not an exact store projection"
        )
    if loaded["schema_version"] != OWNED_DEVELOPMENT_FEATURE_INPUTS_SCHEMA_VERSION:
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature-input schema changed"
        )
    plan = loaded["feature_assembly_plan"]
    if type(plan) is not dict or not _is_bare_sha256(
        plan.get("feature_assembly_plan_sha256")
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature assembly plan is unavailable"
        )
    try:
        validated_plan_hash = validate_development_feature_assembly_plan(
            plan,
            expected_feature_assembly_plan_sha256=plan[
                "feature_assembly_plan_sha256"
            ],
        )
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature assembly plan failed exact validation"
        ) from None
    if (
        validated_plan_hash != plan["feature_assembly_plan_sha256"]
        or plan.get("development_root_scope_sha256")
        != development_root_scope_sha256
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature inputs crossed their root scope"
        )
    events = loaded["events"]
    event_plan = plan.get("event_plan")
    if (
        type(events) is not list
        or type(event_plan) is not list
        or len(events) != plan.get("event_count")
        or len(events) != len(event_plan)
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature-input event count changed"
        )
    for ordinal, (event, planned) in enumerate(
        zip(events, event_plan, strict=True), start=1
    ):
        if (
            type(event) is not dict
            or set(event) != set(_OWNED_DEVELOPMENT_FEATURE_EVENT_KEYS)
            or isinstance(event["event_ordinal"], bool)
            or type(event["event_ordinal"]) is not int
            or event["event_ordinal"] != ordinal
            or type(event["event_plan_item"]) is not dict
            or event["event_plan_item"] != planned
            or planned.get("event_ordinal") != ordinal
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned development feature inputs are reordered or cross-event"
            )
        prefix = event["market_prefix"]
        prefix_proof = event["market_prefix_proof"]
        universe_proof = event["universe_event_proof"]
        extraction_proof = event["extraction_event_proof"]
        if any(
            type(value) is not dict
            for value in (
                prefix,
                prefix_proof,
                universe_proof,
                extraction_proof,
            )
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned development feature event proofs are not exact mappings"
            )
        rows = prefix.get("lookback_rows")
        decision_session = planned.get("availability_session")
        if (
            prefix.get("artifact_stage") != "development"
            or prefix.get("decision_event_id") != planned.get("accession_number")
            or prefix.get("decision_session") != decision_session
            or prefix.get("market_cutoff_session") != decision_session
            or prefix.get("market_stage_manifest_sha256")
            != plan.get("development_market_stage_manifest_sha256")
            or prefix.get("source_manifest_sha256")
            != plan.get("development_market_source_manifest_sha256")
            or prefix.get("lookback_row_count") != MARKET_LOOKBACK_ROW_COUNT
            or type(rows) is not list
            or len(rows) != MARKET_LOOKBACK_ROW_COUNT
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned development feature event did not isolate the exact market prefix"
            )
        row_sessions = [
            row.get("session") if type(row) is dict else None for row in rows
        ]
        if (
            any(type(session) is not str for session in row_sessions)
            or row_sessions != sorted(row_sessions)
            or len(row_sessions) != len(set(row_sessions))
            or row_sessions[-1] != decision_session
            or any(session > decision_session for session in row_sessions)
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned development feature prefix contains a post-decision row"
            )
        proof_hash_fields = (
            (prefix_proof, "market_prefix_proof_sha256"),
            (universe_proof, "universe_event_proof_sha256"),
            (extraction_proof, "extraction_event_proof_sha256"),
        )
        if any(
            not _is_bare_sha256(proof.get(field))
            for proof, field in proof_hash_fields
        ):
            raise SecFilingGemmaStageRunnerError(
                "Owned development feature proof pin is unavailable"
            )
    feature_inputs_hash = loaded["feature_inputs_sha256"]
    if not _is_bare_sha256(feature_inputs_hash):
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature-input checksum is invalid"
        )
    body = {
        key: loaded[key] for key in loaded if key != "feature_inputs_sha256"
    }
    try:
        calculated_inputs_hash = canonical_sha256(body)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature inputs are not canonical JSON"
        ) from None
    if calculated_inputs_hash != feature_inputs_hash:
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature-input checksum changed"
        )
    return copy.deepcopy(loaded)


def run_owned_development_feature_batch(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    """Build the non-authorizing development feature batch from one owned projection."""

    if type(reveal_store) is not SecFilingGemmaRevealStore:
        raise TypeError("reveal_store must be the owned reveal-store implementation")
    if not _is_bare_sha256(development_root_scope_sha256):
        raise SecFilingGemmaStageRunnerError(
            "Development feature scope must be a bare lowercase SHA-256"
        )
    try:
        loaded = reveal_store._load_owned_development_feature_inputs(
            development_root_scope_sha256=development_root_scope_sha256
        )
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature inputs could not be loaded"
        ) from None
    inputs = _validated_owned_development_feature_inputs(
        loaded,
        development_root_scope_sha256=development_root_scope_sha256,
    )
    plan = inputs["feature_assembly_plan"]
    feature_rows: list[dict[str, Any]] = []
    try:
        for event in inputs["events"]:
            prefix_proof = event["market_prefix_proof"]
            universe_proof = event["universe_event_proof"]
            extraction_proof = event["extraction_event_proof"]
            feature_rows.append(
                build_sec_filing_gemma_feature_row(
                    market_prefix=event["market_prefix"],
                    market_prefix_proof=prefix_proof,
                    expected_market_prefix_proof_sha256=prefix_proof[
                        "market_prefix_proof_sha256"
                    ],
                    universe_event_proof=universe_proof,
                    expected_universe_event_proof_sha256=universe_proof[
                        "universe_event_proof_sha256"
                    ],
                    extraction_event_proof=extraction_proof,
                    expected_extraction_event_proof_sha256=extraction_proof[
                        "extraction_event_proof_sha256"
                    ],
                )
            )
        batch = build_owned_development_feature_batch(
            feature_assembly_plan=plan,
            feature_rows=feature_rows,
        )
        validate_owned_development_feature_batch(
            batch,
            feature_assembly_plan=plan,
            expected_feature_assembly_plan_sha256=plan[
                "feature_assembly_plan_sha256"
            ],
            expected_feature_batch_sha256=batch["feature_batch_sha256"],
        )
    except (KeyError, TypeError, ValueError, SecFilingGemmaFeatureError):
        raise SecFilingGemmaStageRunnerError(
            "Owned development feature batch failed causal replay"
        ) from None
    return batch


def _validated_owned_development_label_projection(
    loaded: Any,
    *,
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    """Require the exact store-owned development label projection."""

    if type(loaded) is not dict or set(loaded) != set(
        _OWNED_DEVELOPMENT_LABEL_PROJECTION_KEYS
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development label projection is not exact"
        )
    if (
        loaded["schema_version"]
        != OWNED_DEVELOPMENT_LABEL_PROJECTION_SCHEMA_VERSION
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development label projection schema changed"
        )
    plan = loaded["label_assembly_plan"]
    if type(plan) is not dict or not _is_bare_sha256(
        plan.get("label_assembly_plan_sha256")
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development label assembly plan is unavailable"
        )
    try:
        validated_plan_hash = validate_development_label_assembly_plan(
            plan,
            expected_label_assembly_plan_sha256=plan[
                "label_assembly_plan_sha256"
            ],
        )
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned development label assembly plan failed exact validation"
        ) from None
    if (
        validated_plan_hash != plan["label_assembly_plan_sha256"]
        or plan.get("development_root_scope_sha256")
        != development_root_scope_sha256
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development label projection crossed its root scope"
        )
    source_batch = loaded["source_feature_batch"]
    if (
        type(source_batch) is not dict
        or type(plan.get("event_count")) is not int
        or type(plan.get("matured_event_count")) is not int
        or type(plan.get("unmatured_event_count")) is not int
        or type(source_batch.get("event_count")) is not int
        or not _is_bare_sha256(source_batch.get("feature_batch_sha256"))
        or source_batch.get("development_root_scope_sha256")
        != development_root_scope_sha256
        or source_batch.get("feature_assembly_plan_sha256")
        != plan.get("source_feature_assembly_plan_sha256")
        or source_batch.get("event_count") != plan.get("event_count")
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development label source feature batch crossed its plan"
        )
    audits = loaded["maturity_audit_rows"]
    labels = loaded["label_evidence_rows"]
    if (
        type(audits) is not list
        or len(audits) != plan.get("event_count")
        or type(labels) is not list
        or len(labels) != plan.get("matured_event_count")
    ):
        raise SecFilingGemmaStageRunnerError(
            "Owned development label projection row counts changed"
        )
    projection_hash = loaded["label_projection_sha256"]
    if not _is_bare_sha256(projection_hash):
        raise SecFilingGemmaStageRunnerError(
            "Owned development label projection checksum is invalid"
        )
    body = {
        key: loaded[key]
        for key in loaded
        if key != "label_projection_sha256"
    }
    try:
        calculated = canonical_sha256(body)
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned development label projection is not canonical JSON"
        ) from None
    if calculated != projection_hash:
        raise SecFilingGemmaStageRunnerError(
            "Owned development label projection checksum changed"
        )
    return copy.deepcopy(loaded)


def run_owned_development_label_batch(
    *,
    reveal_store: SecFilingGemmaRevealStore,
    development_root_scope_sha256: str,
) -> dict[str, Any]:
    """Build compact development labels from one store-owned projection."""

    if type(reveal_store) is not SecFilingGemmaRevealStore:
        raise TypeError("reveal_store must be the owned reveal-store implementation")
    if not _is_bare_sha256(development_root_scope_sha256):
        raise SecFilingGemmaStageRunnerError(
            "Development label scope must be a bare lowercase SHA-256"
        )
    try:
        loaded = reveal_store._load_owned_development_label_projection(
            development_root_scope_sha256=development_root_scope_sha256
        )
    except Exception:
        raise SecFilingGemmaStageRunnerError(
            "Owned development label projection could not be loaded"
        ) from None
    projection = _validated_owned_development_label_projection(
        loaded,
        development_root_scope_sha256=development_root_scope_sha256,
    )
    plan = projection["label_assembly_plan"]
    source_batch = projection["source_feature_batch"]
    try:
        batch = build_owned_development_label_batch(
            label_assembly_plan=plan,
            source_feature_batch=source_batch,
            maturity_audit_rows=projection["maturity_audit_rows"],
            label_evidence_rows=projection["label_evidence_rows"],
        )
        validate_owned_development_label_batch(
            batch,
            label_assembly_plan=plan,
            expected_label_assembly_plan_sha256=plan[
                "label_assembly_plan_sha256"
            ],
            source_feature_batch=source_batch,
            expected_source_feature_batch_sha256=source_batch[
                "feature_batch_sha256"
            ],
            expected_label_batch_sha256=batch["label_batch_sha256"],
        )
    except (KeyError, TypeError, ValueError, SecFilingGemmaFeatureError):
        raise SecFilingGemmaStageRunnerError(
            "Owned development label batch failed compact causal replay"
        ) from None
    return batch


__all__ = [
    "SecFilingGemmaStageRunnerError",
    "run_authorized_sec_stage",
    "run_owned_development_feature_batch",
    "run_owned_development_label_batch",
    "run_owned_development_market_batch",
    "run_owned_development_model_batch",
    "run_owned_development_sec_root",
    "run_owned_stage_model_batch",
]
