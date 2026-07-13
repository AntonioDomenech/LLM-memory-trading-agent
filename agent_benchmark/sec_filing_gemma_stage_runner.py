"""Owned, one-shot SEC readers for exact stage and development-root claims.

These are the only production entry points that may turn authenticated store
state into SEC network effects.  A consumed-stage caller supplies no stage,
candidate, URL, digest, path, byte payload, transport, or budget.  The
request-free development entry accepts a complete self-hashed plan only as
evidence for independent store validation; its effect capability is then
derived from the atomically persisted claim.  A claim is never retried after
an indeterminate process exit.

The private SEC contact exists only in the public call stack and the private
transport factory.  Durable output contains its validated hash, never its raw
value.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
    build_stage_content_manifest,
    canonical_sha256,
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
from .sec_filing_gemma_reveal_store import (
    DEVELOPMENT_SEC_ROOT_COMPLETE_MARKER_SCHEMA_VERSION,
    MAX_SEC_BATCH_FILE_BYTES,
    MAX_SEC_BATCH_FILES,
    MAX_SEC_BATCH_TOTAL_BYTES,
    SEC_BATCH_COMPLETE_MARKER_FILENAME,
    SEC_BATCH_COMPLETE_MARKER_SCHEMA_VERSION,
    SEC_STAGE_COMPONENT_DIRECTORY_NAME,
    STAGE_OUTPUTS_DIRECTORY_NAME,
    SecFilingGemmaRevealStore,
    _fsync_directory,
    _secure_directory,
)
from .sec_filing_gemma_stage_access import (
    DEVELOPMENT_CONTENT_ROOT_COMPONENT_ID,
    DEVELOPMENT_CONTENT_ROOT_RAW_BATCH_CAP_BYTES,
)
from .sec_filing_gemma_stage_authorization import (
    OWNED_SEC_RAW_BATCH_MAX_BYTES,
    SEC_STAGE_DOCUMENT_BATCH_COMPONENT_ID,
    _sec_component_plan_from_bundle,
)
from .sec_point_in_time import validate_sec_user_agent


class SecFilingGemmaStageRunnerError(RuntimeError):
    """The owned SEC effect could not finish with a durable exact receipt."""


_SHA256_LENGTH: Final[int] = 64
_OUTPUT_FILE_MODE: Final[int] = 0o600
_BINARY: Final[int] = getattr(os, "O_BINARY", 0)
_NOFOLLOW: Final[int] = getattr(os, "O_NOFOLLOW", 0)
_MAX_COMPONENT_PLAN_SECONDS: Final[float] = 720.0


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


__all__ = [
    "SecFilingGemmaStageRunnerError",
    "run_authorized_sec_stage",
    "run_owned_development_sec_root",
]
