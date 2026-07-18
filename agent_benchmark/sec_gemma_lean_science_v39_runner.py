"""Development-only v3.9 SEC-to-science execution worker.

The module is inert on import.  Its pure builders prepare the authenticated
75-row model plan, segment-aware Gemma provenance, frozen feature rows, and the
existing deterministic evaluator.  Network and private-store orchestration is
entered only through :func:`run_development` or the command-line ``development``
command after the separately pushed v3.9 preflight has authorized it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import ssl
import stat
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, HTTPRedirectHandler, ProxyHandler, Request, build_opener

from . import sec_gemma_lean_science_v39_contract as contract


RUNNER_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-lean-science-v3-9-runner-v1"
MODEL_PLAN_SCHEMA_VERSION: Final[str] = "aapl-sec-gemma-lean-science-v3-9-model-plan-v1"
REQUEST_COMMITMENTS_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-request-commitments-v1"
)
RUNTIME_GUARD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-runtime-segment-guard-v1"
)
RUNTIME_AGGREGATE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-runtime-aggregate-v1"
)
LATENCY_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-latency-receipt-v1"
)
SEMANTIC_EVENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-semantic-event-v1"
)
SEMANTIC_ROW_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-semantic-row-v1"
)
SEMANTIC_BATCH_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-semantic-batch-v1"
)
DETERMINISTIC_PAYLOAD_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-deterministic-payload-v1"
)
TERMINAL_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-development-result-v1"
)
PRIVATE_TERMINAL_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-private-terminal-material-v1"
)
PAUSE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-development-pause-v1"
)
CONTINUATION_DOCUMENT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-lean-science-v3-9-continuation-document-v1"
)

DEVELOPMENT: Final[str] = "development"
PILOT_COUNT: Final[int] = 5
TOTAL_CALL_COUNT: Final[int] = 75
REMAINING_CALL_COUNT: Final[int] = 70
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class V39RunnerError(RuntimeError):
    """A fixed public-safe v3.9 runner failure."""

    def __init__(self, code: str) -> None:
        if (
            type(code) is not str
            or not code
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in code)
        ):
            code = "runner_invalid_error_code"
        self.code = code
        super().__init__(code)

    def __repr__(self) -> str:
        return f"V39RunnerError(code={self.code!r})"


def _fail(code: str) -> None:
    raise V39RunnerError(code)


def _sha(value: Any, code: str = "runner_hash_invalid") -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(code)
    return value


def _mapping(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return dict(value)


def _plain(value: Any) -> Any:
    """Detach mappings/tuples through the contract's strict JSON boundary."""

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: thaw(child) for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [thaw(child) for child in item]
        return item

    try:
        return json.loads(contract.canonical_json_bytes(thaw(value)).decode("utf-8"))
    except Exception:
        _fail("runner_value_not_canonical")


@dataclass(frozen=True)
class PreparedModelCall:
    canonical_ordinal: int
    execution_ordinal: int
    accession_number: str
    acceptance_datetime: str
    request: Mapping[str, Any]
    proof: Mapping[str, Any]
    request_bytes: bytes
    sentence_ids: tuple[str, ...]
    request_byte_count: int
    pilot: bool


@dataclass(frozen=True)
class ModelPlan:
    model_slice: Mapping[str, Any]
    universe: Mapping[str, Any]
    canonical_calls: tuple[PreparedModelCall, ...]
    execution_calls: tuple[PreparedModelCall, ...]
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class DevelopmentDependencies:
    """Injected production boundary for an offline-testable one-shot worker."""

    authenticate_execution: Callable[[Path, bool], Mapping[str, Any]]
    load_private_contact: Callable[[Path], str]
    authenticate_source: Callable[[Path, str], Any]
    build_projection: Callable[[Any], Any]
    open_store: Callable[[Path, Mapping[str, Any]], Any]
    inspect_invocation_parent: (
        Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None
    ) = None
    publication_recovery_active: Callable[[], bool] | None = None
    yahoo_fetch: Callable[[str], tuple[bytes, Mapping[str, Any]]] | None = None
    runtime_probe: Callable[[str], tuple[bytes, Mapping[str, Any]]] | None = None
    generation: Callable[[bytes], tuple[bytes, Mapping[str, Any], int]] | None = None
    publish_pause: Callable[[Path, Mapping[str, Any]], None] | None = None
    publish_result: Callable[[Path, Mapping[str, Any]], None] | None = None


def _projection_attr(projection: Any, name: str) -> Any:
    if not hasattr(projection, name):
        _fail("bridge_projection_invalid")
    return getattr(projection, name)


def _legacy_universe(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        from .sec_filing_gemma_contract import build_corpus_universe_manifest
        from .sec_session_calendar import EXPECTED_SESSIONS

        return build_corpus_universe_manifest(
            catalog_artifact_sha256="0" * 64,
            calendar_artifact_sha256="1" * 64,
            catalog_total_record_count=len(records),
            catalog_eligible_record_count=len(records),
            session_dates=EXPECTED_SESSIONS,
            records=[dict(item) for item in records],
        )
    except Exception:
        _fail("legacy_universe_invalid")


def build_model_plan(projection: Any) -> ModelPlan:
    """Build the exact 75 requests/proofs and fixed pilot execution order."""

    if _projection_attr(projection, "stage") != DEVELOPMENT:
        _fail("bridge_projection_stage_invalid")
    primary_documents = [
        dict(item) for item in _projection_attr(projection, "primary_documents")
    ]
    records = [dict(item) for item in _projection_attr(projection, "records")]
    event_order = [dict(item) for item in _projection_attr(projection, "event_order")]
    if not (
        len(primary_documents)
        == len(records)
        == len(event_order)
        == TOTAL_CALL_COUNT
    ):
        _fail("bridge_projection_count_invalid")
    try:
        from .sec_gemma_online_risk_overlay_acquisition import (
            _build_model_requests,
            _build_model_slice,
            _build_universe_event_proofs,
            build_acquisition_plan,
        )
        from .sec_gemma_online_risk_overlay_production import (
            _validate_blinded_model_request,
            _validate_universe_proof,
        )

        universe = _legacy_universe(records)
        requests = _build_model_requests(
            primary_documents,
            carry_in_documents=[],
            stage=DEVELOPMENT,
        )
        proofs = _build_universe_event_proofs(
            stage=DEVELOPMENT,
            universe=universe,
            current_documents=primary_documents,
            predecessor_bundles=[],
        )
        old_plan = build_acquisition_plan(DEVELOPMENT)
        plan = copy.deepcopy(old_plan)
        plan["attempt_id"] = contract.DEVELOPMENT_ATTEMPT_ID
        model_slice = _build_model_slice(
            plan=plan,
            model_requests=requests,
            universe_event_proofs=proofs,
        )
    except V39RunnerError:
        raise
    except Exception:
        _fail("model_plan_construction_failed")

    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, (request, raw_proof) in enumerate(
        zip(requests, proofs, strict=True), start=1
    ):
        try:
            proof = _validate_universe_proof(raw_proof)
            request_bytes, sentence_ids, _payload = _validate_blinded_model_request(
                request, proof
            )
            current = proof["current_record"]
            accession = request["accession_number"]
            acceptance = current["acceptance_datetime"]
        except Exception:
            _fail("model_plan_request_invalid")
        if (
            type(accession) is not str
            or accession in seen
            or type(acceptance) is not str
        ):
            _fail("model_plan_request_invalid")
        seen.add(accession)
        prepared.append(
            {
                "canonical_ordinal": ordinal,
                "accession_number": accession,
                "acceptance_datetime": acceptance,
                "request": request,
                "proof": proof,
                "request_bytes": request_bytes,
                "sentence_ids": sentence_ids,
                "request_byte_count": len(request_bytes),
            }
        )
    chronological = sorted(
        prepared,
        key=lambda item: (
            item["request"]["availability_session"],
            item["acceptance_datetime"],
            item["accession_number"],
        ),
    )
    if [item["accession_number"] for item in chronological] != [
        item["accession_number"] for item in prepared
    ]:
        _fail("model_plan_event_order_invalid")
    pilots = sorted(
        prepared,
        key=lambda item: (-item["request_byte_count"], item["accession_number"]),
    )[:PILOT_COUNT]
    pilot_ids = {item["accession_number"] for item in pilots}
    execution = pilots + [
        item for item in chronological if item["accession_number"] not in pilot_ids
    ]
    execution_ordinal = {
        item["accession_number"]: ordinal
        for ordinal, item in enumerate(execution, start=1)
    }
    canonical_calls = tuple(
        PreparedModelCall(
            canonical_ordinal=item["canonical_ordinal"],
            execution_ordinal=execution_ordinal[item["accession_number"]],
            accession_number=item["accession_number"],
            acceptance_datetime=item["acceptance_datetime"],
            request=MappingProxyType(dict(item["request"])),
            proof=MappingProxyType(dict(item["proof"])),
            request_bytes=item["request_bytes"],
            sentence_ids=tuple(item["sentence_ids"]),
            request_byte_count=item["request_byte_count"],
            pilot=item["accession_number"] in pilot_ids,
        )
        for item in prepared
    )
    by_accession = {item.accession_number: item for item in canonical_calls}
    execution_calls = tuple(by_accession[item["accession_number"]] for item in execution)
    model_request_index = [
        {
            "canonical_ordinal": item.canonical_ordinal,
            "request_sha256": item.request["request_sha256"],
            "request_byte_count": item.request_byte_count,
            "preprocessed_event_sha256": item.request["preprocessed_event_sha256"],
        }
        for item in canonical_calls
    ]
    execution_index = [
        {
            "execution_ordinal": item.execution_ordinal,
            "request_sha256": item.request["request_sha256"],
            "request_byte_count": item.request_byte_count,
            "pilot": item.pilot,
        }
        for item in execution_calls
    ]
    order_commitment_index = [
        {
            "request_sha256": item.request["request_sha256"],
            "request_byte_count": item.request_byte_count,
        }
        for item in execution_calls
    ]
    body = {
        "schema_version": MODEL_PLAN_SCHEMA_VERSION,
        "stage": DEVELOPMENT,
        "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
        "document_count": TOTAL_CALL_COUNT,
        "model_slice_sha256": model_slice["model_slice_sha256"],
        "universe_sha256": universe["universe_sha256"],
        "canonical_request_index_sha256": contract.canonical_sha256(model_request_index),
        "execution_index_sha256": contract.canonical_sha256(execution_index),
        "pilot_order_sha256": contract.canonical_sha256(
            order_commitment_index[:PILOT_COUNT]
        ),
        "remaining_order_sha256": contract.canonical_sha256(
            order_commitment_index[PILOT_COUNT:]
        ),
        "model_call_count": TOTAL_CALL_COUNT,
        "pilot_count": PILOT_COUNT,
        "remaining_count": REMAINING_CALL_COUNT,
    }
    manifest = {**body, "model_plan_sha256": contract.canonical_sha256(body)}
    return ModelPlan(
        MappingProxyType(model_slice),
        MappingProxyType(universe),
        canonical_calls,
        execution_calls,
        MappingProxyType(manifest),
    )


def build_preflight_request_commitments(projection: Any) -> dict[str, Any]:
    """Return the strict aggregate-only projection/request commitments."""

    plan = build_model_plan(projection)
    bridge_manifest = _mapping(_projection_attr(projection, "manifest"), "bridge_manifest_invalid")
    source_order = [dict(item) for item in _projection_attr(projection, "source_order")]
    prior_links = [dict(item) for item in _projection_attr(projection, "prior_links")]
    requests = [item.request for item in plan.canonical_calls]
    execution = [
        {
            "request_sha256": item.request["request_sha256"],
            "request_byte_count": item.request_byte_count,
        }
        for item in plan.execution_calls
    ]
    value = {
        "schema_version": REQUEST_COMMITMENTS_SCHEMA_VERSION,
        "stage": DEVELOPMENT,
        "document_count": TOTAL_CALL_COUNT,
        "record_count": TOTAL_CALL_COUNT,
        "event_count": TOTAL_CALL_COUNT,
        "request_count": TOTAL_CALL_COUNT,
        "pilot_count": PILOT_COUNT,
        "remaining_count": REMAINING_CALL_COUNT,
        "documents_sha256": bridge_manifest["documents_sha256"],
        "records_sha256": bridge_manifest["records_sha256"],
        "events_sha256": bridge_manifest["event_order_sha256"],
        "preprocessed_events_sha256": contract.canonical_sha256(
            [item["preprocessed_event_sha256"] for item in requests]
        ),
        "canonical_requests_sha256": contract.canonical_sha256(
            [
                {
                    "request_sha256": item["request_sha256"],
                    "request_byte_count": len(item["request_bytes"]),
                }
                for item in requests
            ]
        ),
        "pilot_order_sha256": contract.canonical_sha256(execution[:PILOT_COUNT]),
        "remaining_order_sha256": contract.canonical_sha256(execution[PILOT_COUNT:]),
        "source_order_sha256": bridge_manifest["source_order_sha256"],
        "prior_links_sha256": bridge_manifest["prior_links_sha256"],
        "set_parity": True,
        "sequence_parity": True,
        "prior_link_parity": True,
        "confirmation_or_final_opened": False,
        "contains_private_rows": False,
    }
    return value


def build_attempt_authority(
    *,
    execution_authority: Mapping[str, Any],
    projection: Any,
    commitments: Mapping[str, Any],
    plan: ModelPlan,
) -> dict[str, Any]:
    """Validate the stable nine-key F authority against the private rebuild."""

    execution = _mapping(execution_authority, "execution_authority_invalid")
    bridge = _mapping(_projection_attr(projection, "manifest"), "bridge_manifest_invalid")
    frozen = dict(commitments)
    if set(execution) != {
        "plan",
        "attempt",
        "implementation",
        "preflight",
        "source",
        "science",
        "effect_budget",
        "request_order",
        "pilot_order",
    }:
        _fail("execution_authority_invalid")
    implementation = _mapping(execution["implementation"], "execution_authority_invalid")
    preflight = _mapping(execution["preflight"], "execution_authority_invalid")
    source = _mapping(execution["source"], "execution_authority_invalid")
    science = _mapping(execution["science"], "execution_authority_invalid")
    request_order = _mapping(execution["request_order"], "execution_authority_invalid")
    pilot_order = _mapping(execution["pilot_order"], "execution_authority_invalid")
    expected_request = {
        "count": TOTAL_CALL_COUNT,
        "canonical_requests_sha256": frozen.get("canonical_requests_sha256"),
        "remaining_order_sha256": frozen.get("remaining_order_sha256"),
    }
    authority_hashes = (
        implementation.get("production_source_inventory_sha256"),
        implementation.get("test_source_inventory_sha256"),
        preflight.get("public_artifact_sha256"),
        preflight.get("public_artifact_literal_sha256"),
        preflight.get("private_manifest_sha256"),
        preflight.get("private_manifest_literal_sha256"),
    )
    if (
        set(implementation)
        != {
            "commit",
            "tree",
            "production_source_inventory_sha256",
            "test_source_inventory_sha256",
        }
        or set(preflight)
        != {
            "commit",
            "tree",
            "public_artifact_sha256",
            "public_artifact_literal_sha256",
            "private_manifest_sha256",
            "private_manifest_literal_sha256",
        }
        or set(source)
        != {
            "base_commit",
            "base_tree",
            "terminal_internal_sha256",
            "inventory_sha256",
            "bridge_sha256",
        }
        or execution["plan"] != contract.CONTRACT_MANIFEST_SHA256
        or execution["attempt"] != contract.DEVELOPMENT_ATTEMPT_ID
        or source.get("base_commit") != contract.BASE_COMMIT
        or source.get("base_tree") != contract.BASE_TREE
        or source.get("terminal_internal_sha256")
        != contract.V38_TERMINAL_INTERNAL_SHA256
        or source.get("inventory_sha256") != contract.V38_INVENTORY_SHA256
        or source.get("bridge_sha256") != bridge.get("bridge_sha256")
        or science != {"projection_sha256": contract.SCIENCE_PROJECTION_SHA256}
        or request_order != expected_request
        or pilot_order
        != {
            "count": PILOT_COUNT,
            "pilot_order_sha256": frozen.get("pilot_order_sha256"),
        }
        or execution["effect_budget"] != contract.build_effect_budgets()
        or any(
            type(value) is not str or _SHA256_RE.fullmatch(value) is None
            for value in authority_hashes
        )
    ):
        _fail("execution_projection_mismatch")
    if (
        type(implementation.get("commit")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", implementation["commit"]) is None
        or type(implementation.get("tree")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", implementation["tree"]) is None
        or type(preflight.get("commit")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", preflight["commit"]) is None
        or type(preflight.get("tree")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", preflight["tree"]) is None
        or plan.manifest.get("remaining_order_sha256")
        != frozen.get("remaining_order_sha256")
    ):
        _fail("execution_authority_invalid")
    # The store performs its own exact-key and canonical-JSON validation.  This
    # round-trip catches bytes or proxy objects before any attempt is consumed.
    try:
        return json.loads(contract.canonical_json_bytes(execution).decode("utf-8"))
    except Exception:
        _fail("attempt_authority_invalid")


_RUNTIME_STABLE_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "model_name",
    "manifest_sha256",
    "config_sha256",
    "ordered_layer_sha256s",
    "ordered_layer_content_sha256s",
    "version_response_sha256",
    "show_response_semantic_sha256",
    "show_semantic_excluded_keys",
    "model_info_sha256",
    "active_from_blob_sha256s",
    "runtime_fingerprint_material",
    "runtime_fingerprint_sha256",
    "manifest_config_layers_replayed",
    "all_layer_contents_hashed",
    "exact_two_active_from_blobs_verified",
)


def build_runtime_segment_guard(
    *,
    segment_id: str,
    store_segment_id: str | None = None,
    pre_runtime_receipt: Mapping[str, Any],
    post_runtime_receipt: Mapping[str, Any],
    ordered_generation_response_event_sha256s: Sequence[str],
    generation_count: int,
) -> dict[str, Any]:
    """Bind one uninterrupted model segment to v2.2 semantic identity."""

    if segment_id not in {"full", "pilot", "continuation"}:
        _fail("runtime_segment_invalid")
    physical_segment = segment_id if store_segment_id is None else store_segment_id
    if physical_segment not in {"initial", "continuation", "full", "pilot"}:
        _fail("runtime_segment_invalid")
    if segment_id == "continuation" and physical_segment != "continuation":
        _fail("runtime_segment_invalid")
    if segment_id in {"full", "pilot"} and physical_segment not in {
        "initial",
        segment_id,
    }:
        _fail("runtime_segment_invalid")
    pre = _mapping(pre_runtime_receipt, "runtime_receipt_invalid")
    post = _mapping(post_runtime_receipt, "runtime_receipt_invalid")
    for field in _RUNTIME_STABLE_FIELDS:
        if field not in pre or pre.get(field) != post.get(field):
            _fail("runtime_semantic_identity_changed")
    if (
        pre.get("model_name") != contract.MODEL_NAME
        or pre.get("runtime_fingerprint_sha256") != contract.RUNTIME_FINGERPRINT_SHA256
        or pre.get("version_response_sha256")
        != contract.RUNTIME_VERSION_RESPONSE_SHA256
        or pre.get("show_response_semantic_sha256")
        != contract.RUNTIME_SHOW_SEMANTIC_SHA256
        or tuple(pre.get("show_semantic_excluded_keys", ()))
        != tuple(contract.RUNTIME_SHOW_SEMANTIC_EXCLUDED_KEYS)
        or pre.get("model_info_sha256") != contract.RUNTIME_MODEL_INFO_SHA256
        or pre.get("manifest_sha256") != contract.MODEL_MANIFEST_SHA256
        or pre.get("config_sha256") != contract.MODEL_CONFIG_DIGEST
        or tuple(pre.get("ordered_layer_content_sha256s", ()))
        != tuple(contract.MODEL_LAYER_DIGESTS)
        or len(tuple(pre.get("active_from_blob_sha256s", ())))
        != contract.MODEL_ACTIVE_FROM_BLOB_COUNT
    ):
        _fail("runtime_candidate_pin_mismatch")
    expected_count = {"full": TOTAL_CALL_COUNT, "pilot": PILOT_COUNT, "continuation": REMAINING_CALL_COUNT}[segment_id]
    events = list(ordered_generation_response_event_sha256s)
    if generation_count != expected_count or len(events) != expected_count:
        _fail("runtime_segment_generation_count_invalid")
    for item in events:
        _sha(item, "runtime_segment_event_hash_invalid")
    body = {
        "schema_version": RUNTIME_GUARD_SCHEMA_VERSION,
        "segment_id": segment_id,
        "store_segment_id": physical_segment,
        "generation_count": generation_count,
        "pre_runtime_receipt_sha256": _sha(pre.get("runtime_receipt_sha256")),
        "post_runtime_receipt_sha256": _sha(post.get("runtime_receipt_sha256")),
        "stable_runtime_identity_sha256": contract.canonical_sha256(
            {field: pre[field] for field in _RUNTIME_STABLE_FIELDS}
        ),
        "ordered_generation_response_event_sha256s": events,
        "ordered_generation_response_events_sha256": contract.canonical_sha256(events),
        "raw_show_hash_is_diagnostic_only": True,
        "modified_at_is_excluded_only": True,
        "identity_http_request_count": 4,
        "retry_count": 0,
    }
    return {**body, "segment_guard_sha256": contract.canonical_sha256(body)}


def build_runtime_aggregate_guard(
    *,
    segment_guards: Sequence[Mapping[str, Any]],
    execution_request_sha256s: Sequence[str],
) -> dict[str, Any]:
    guards = [dict(item) for item in segment_guards]
    expected_ids = ["full"] if len(guards) == 1 else ["pilot", "continuation"]
    if [item.get("segment_id") for item in guards] != expected_ids:
        _fail("runtime_aggregate_segment_order_invalid")
    for item in guards:
        supplied = item.get("segment_guard_sha256")
        if supplied != contract.canonical_sha256(
            {key: value for key, value in item.items() if key != "segment_guard_sha256"}
        ):
            _fail("runtime_segment_guard_invalid")
    requests = list(execution_request_sha256s)
    if len(requests) != TOTAL_CALL_COUNT:
        _fail("runtime_aggregate_request_count_invalid")
    for item in requests:
        _sha(item)
    body = {
        "schema_version": RUNTIME_AGGREGATE_SCHEMA_VERSION,
        "segment_guard_sha256s": [item["segment_guard_sha256"] for item in guards],
        "segment_ids": expected_ids,
        "store_segment_ids": [item["store_segment_id"] for item in guards],
        "execution_request_sha256s": requests,
        "execution_order_sha256": contract.canonical_sha256(requests),
        "generation_count": TOTAL_CALL_COUNT,
        "identity_http_request_count": 4 if len(guards) == 1 else 8,
        "one_guard_does_not_span_pause": len(guards) == 2,
        "retry_count": 0,
    }
    return {**body, "runtime_aggregate_sha256": contract.canonical_sha256(body)}


def _strict_output_mapping(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=lambda _x: (_ for _ in ()).throw(ValueError()))
    except Exception:
        _fail("extractor_output_json_invalid")
    if type(value) is not dict:
        _fail("extractor_output_json_invalid")
    return value


def parse_sealed_generation_response(
    *, response_bytes: bytes, supplied_sentence_ids: tuple[str, ...]
) -> dict[str, Any]:
    """Parse one already-sealed response only after its segment guard passes."""

    if type(response_bytes) is not bytes or not response_bytes:
        _fail("generation_response_invalid")
    try:
        from .sec_filing_gemma_ollama import (
            _extract_ollama_attempt_output_bytes,
            _validate_extractor_output_bytes,
        )

        output_bytes, normal_completion = _extract_ollama_attempt_output_bytes(
            response_bytes
        )
    except Exception:
        _fail("generation_response_envelope_invalid")
    output_sha = hashlib.sha256(output_bytes).hexdigest()
    canonical_output_sha: str | None = None
    validated_output: dict[str, Any] | None = None
    status = "invalid"
    if normal_completion:
        try:
            validated_sha, canonical_output_sha = _validate_extractor_output_bytes(
                output_bytes, supplied_sentence_ids=supplied_sentence_ids
            )
            candidate = _strict_output_mapping(output_bytes)
        except Exception:
            pass
        else:
            if validated_sha != output_sha:
                _fail("extractor_output_hash_invalid")
            validated_output = candidate
            status = "valid"
    return {
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "extractor_output_sha256": output_sha,
        "extractor_output_canonical_sha256": canonical_output_sha,
        "normal_completion": bool(normal_completion),
        "extraction_status": status,
        "validated_output": validated_output,
    }


def build_latency_receipt(
    *, plan: ModelPlan, durations_ns_by_request_sha256: Mapping[str, int]
) -> dict[str, Any]:
    durations = dict(durations_ns_by_request_sha256)
    pilots = plan.execution_calls[:PILOT_COUNT]
    pilot_rows: list[dict[str, Any]] = []
    pilot_values: list[int] = []
    for item in pilots:
        digest = item.request["request_sha256"]
        duration = durations.get(digest)
        if type(duration) is not int or type(duration) is bool or duration <= 0:
            _fail("pilot_duration_invalid")
        pilot_values.append(duration)
        pilot_rows.append(
            {
                "execution_ordinal": item.execution_ordinal,
                "request_sha256": digest,
                "request_byte_count": item.request_byte_count,
                "duration_ns": duration,
            }
        )
    projected = contract.projected_pilot_ns(pilot_values)
    body = {
        "schema_version": LATENCY_RECEIPT_SCHEMA_VERSION,
        "selection": "five_largest_request_bytes_desc_accession_asc",
        "remaining_order": "availability_acceptance_accession_ascending",
        "pilot_rows": pilot_rows,
        "pilot_order_sha256": contract.canonical_sha256(pilot_rows),
        "pilot_count": PILOT_COUNT,
        "remaining_count": REMAINING_CALL_COUNT,
        "formula": "sum(pilot_duration_ns)+70*max(pilot_duration_ns)",
        "projected_ns": projected,
        "threshold_ns": contract.PILOT_PROJECTED_THRESHOLD_NS,
        "pause_required": projected > contract.PILOT_PROJECTED_THRESHOLD_NS,
        "timing_interval": (
            "monotonic_ns_after_durable_intent_before_transport_through_"
            "bounded_body_framing_and_close_before_persistence_or_semantic_parse"
        ),
    }
    return {**body, "latency_receipt_sha256": contract.canonical_sha256(body)}


def build_semantic_payload(
    *,
    plan: ModelPlan,
    sealed_calls_by_request_sha256: Mapping[str, Mapping[str, Any]],
    segment_guard_by_request_sha256: Mapping[str, Mapping[str, Any]],
    runtime_aggregate: Mapping[str, Any],
    latency_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Open sealed response bytes and build segment-aware semantic rows."""

    sealed = dict(sealed_calls_by_request_sha256)
    guard_by_request = dict(segment_guard_by_request_sha256)
    aggregate = dict(runtime_aggregate)
    latency = dict(latency_receipt)
    if aggregate.get("runtime_aggregate_sha256") != contract.canonical_sha256(
        {key: value for key, value in aggregate.items() if key != "runtime_aggregate_sha256"}
    ):
        _fail("runtime_aggregate_invalid")
    if latency.get("latency_receipt_sha256") != contract.canonical_sha256(
        {key: value for key, value in latency.items() if key != "latency_receipt_sha256"}
    ):
        _fail("latency_receipt_invalid")
    rows: list[dict[str, Any]] = []
    event_receipt_hashes: list[str] = []
    for item in plan.canonical_calls:
        request_sha = item.request["request_sha256"]
        call = _mapping(sealed.get(request_sha), "sealed_generation_missing")
        guard = _mapping(guard_by_request.get(request_sha), "segment_guard_missing")
        if guard.get("segment_guard_sha256") != contract.canonical_sha256(
            {key: value for key, value in guard.items() if key != "segment_guard_sha256"}
        ):
            _fail("segment_guard_invalid")
        response_bytes = call.get("response_bytes")
        if type(response_bytes) is not bytes:
            _fail("sealed_generation_missing")
        parsed = parse_sealed_generation_response(
            response_bytes=response_bytes,
            supplied_sentence_ids=item.sentence_ids,
        )
        duration_ns = call.get("duration_ns")
        response_event_sha = _sha(call.get("response_event_sha256"))
        if type(duration_ns) is not int or type(duration_ns) is bool or duration_ns <= 0:
            _fail("generation_duration_invalid")
        receipt_body = {
            "schema_version": SEMANTIC_EVENT_SCHEMA_VERSION,
            "stage": DEVELOPMENT,
            "canonical_ordinal": item.canonical_ordinal,
            "execution_ordinal": item.execution_ordinal,
            "request_sha256": request_sha,
            "request_byte_count": item.request_byte_count,
            "universe_event_proof_sha256": item.proof["universe_event_proof_sha256"],
            "segment_id": guard["segment_id"],
            "store_segment_id": guard["store_segment_id"],
            "segment_guard_sha256": guard["segment_guard_sha256"],
            "runtime_aggregate_sha256": aggregate["runtime_aggregate_sha256"],
            "response_event_sha256": response_event_sha,
            "response_sha256": parsed["response_sha256"],
            "extractor_output_sha256": parsed["extractor_output_sha256"],
            "extractor_output_canonical_sha256": parsed[
                "extractor_output_canonical_sha256"
            ],
            "normal_completion": parsed["normal_completion"],
            "extraction_status": parsed["extraction_status"],
            "duration_ns": duration_ns,
            "retry_count": 0,
            "repair_count": 0,
            "fallback_count": 0,
        }
        receipt = {
            **receipt_body,
            "semantic_event_receipt_sha256": contract.canonical_sha256(receipt_body),
        }
        event_receipt_hashes.append(receipt["semantic_event_receipt_sha256"])
        row_body = {
            "schema_version": SEMANTIC_ROW_SCHEMA_VERSION,
            "stage": DEVELOPMENT,
            "ordinal": item.canonical_ordinal,
            "accession_number": item.accession_number,
            "form": item.request["form"],
            "decision_session": item.request["availability_session"],
            "preprocessed_event_sha256": item.request["preprocessed_event_sha256"],
            "request_sha256": request_sha,
            "model_slice_sha256": plan.model_slice["model_slice_sha256"],
            "universe_event_proof_sha256": item.proof["universe_event_proof_sha256"],
            "extraction_status": parsed["extraction_status"],
            "extraction_authenticated": True,
            "document_quality": (
                None
                if parsed["validated_output"] is None
                else parsed["validated_output"]["document_quality"]
            ),
            "validated_output": parsed["validated_output"],
            "semantic_event_receipt": receipt,
            "semantic_event_receipt_sha256": receipt[
                "semantic_event_receipt_sha256"
            ],
            "latency_preflight_receipt": latency,
            "latency_preflight_receipt_sha256": latency["latency_receipt_sha256"],
            "runtime_segment_guard_sha256": guard["segment_guard_sha256"],
            "runtime_aggregate_sha256": aggregate["runtime_aggregate_sha256"],
        }
        rows.append(
            {
                **row_body,
                "semantic_extraction_row_sha256": contract.canonical_sha256(row_body),
            }
        )
    batch_body = {
        "schema_version": SEMANTIC_BATCH_SCHEMA_VERSION,
        "stage": DEVELOPMENT,
        "model_slice_sha256": plan.model_slice["model_slice_sha256"],
        "runtime_aggregate_sha256": aggregate["runtime_aggregate_sha256"],
        "latency_receipt_sha256": latency["latency_receipt_sha256"],
        "semantic_extraction_row_sha256s": [
            item["semantic_extraction_row_sha256"] for item in rows
        ],
        "semantic_event_receipt_sha256s": event_receipt_hashes,
        "model_call_count": len(rows),
        "retry_count": 0,
        "repair_count": 0,
        "fallback_count": 0,
        "model_pull_count": 0,
        "paid_api_call_count": 0,
    }
    batch_hash = contract.canonical_sha256(batch_body)
    return {
        "semantic_extraction_rows": rows,
        "semantic_extraction_row_sha256s": [
            item["semantic_extraction_row_sha256"] for item in rows
        ],
        "semantic_batch_receipt_sha256": batch_hash,
    }


def build_deterministic_payload(
    *, stage_slice: Mapping[str, Any], semantic_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Mint frozen feature rows while retaining v3.9 segment provenance."""

    try:
        from .sec_gemma_online_risk_overlay_features import (
            _mint_feature_row_from_source_bound_components,
        )
        from .sec_gemma_online_risk_overlay_production import (
            _decision_market_lookback_rows,
            _stage_slice_market_material,
            _validate_blinded_model_request,
            _validate_universe_proof,
            _validated_stage_slice,
        )

        fixed = _validated_stage_slice(stage_slice)
    except Exception:
        _fail("stage_slice_invalid")
    if fixed.get("stage") != DEVELOPMENT or fixed.get("last_value_session") != "2018-12-31":
        _fail("stage_slice_invalid")
    semantic = _mapping(semantic_payload, "semantic_payload_invalid")
    if set(semantic) != {
        "semantic_extraction_rows",
        "semantic_extraction_row_sha256s",
        "semantic_batch_receipt_sha256",
    }:
        _fail("semantic_payload_invalid")
    rows = list(semantic["semantic_extraction_rows"])
    row_hashes = list(semantic["semantic_extraction_row_sha256s"])
    requests = fixed["model_requests"]
    proofs = fixed["universe_event_proofs"]
    if not (
        len(rows) == len(row_hashes) == len(requests) == len(proofs) == TOTAL_CALL_COUNT
    ):
        _fail("deterministic_batch_count_invalid")
    try:
        market_material = _stage_slice_market_material(fixed)
    except Exception:
        _fail("market_material_invalid")
    feature_rows: list[dict[str, Any]] = []
    proof_hashes: list[str] = []
    request_hashes: list[str] = []
    for ordinal, (request, raw_proof, row, row_hash) in enumerate(
        zip(requests, proofs, rows, row_hashes, strict=True), start=1
    ):
        try:
            proof = _validate_universe_proof(raw_proof)
            _validate_blinded_model_request(request, proof)
        except Exception:
            _fail("deterministic_source_binding_invalid")
        if (
            type(row) is not dict
            or row.get("semantic_extraction_row_sha256") != row_hash
            or contract.canonical_sha256(
                {key: value for key, value in row.items() if key != "semantic_extraction_row_sha256"}
            )
            != row_hash
            or row.get("ordinal") != ordinal
            or row.get("request_sha256") != request["request_sha256"]
            or row.get("universe_event_proof_sha256")
            != proof["universe_event_proof_sha256"]
        ):
            _fail("semantic_row_binding_invalid")
        current = proof["current_record"]
        try:
            lookback = _decision_market_lookback_rows(
                market_material, current["availability_session"]
            )
            bindings = {
                "stage_slice_sha256": fixed["slice_sha256"],
                "model_slice_sha256": row["model_slice_sha256"],
                "request_sha256": request["request_sha256"],
                "preprocessed_event_sha256": request["preprocessed_event_sha256"],
                "universe_event_proof_sha256": proof["universe_event_proof_sha256"],
                "current_record_sha256": proof["current_record_sha256"],
                "current_filing_sha256": proof["current_filing_sha256"],
                "prior_same_form_filing_sha256": proof[
                    "prior_same_form_filing_sha256"
                ],
                "semantic_extraction_row_sha256": row_hash,
                "semantic_event_receipt_sha256": row[
                    "semantic_event_receipt_sha256"
                ],
                "latency_preflight_receipt_sha256": row[
                    "latency_preflight_receipt_sha256"
                ],
                "semantic_batch_receipt_sha256": semantic[
                    "semantic_batch_receipt_sha256"
                ],
                "runtime_segment_guard_sha256": row[
                    "runtime_segment_guard_sha256"
                ],
                "runtime_aggregate_sha256": row["runtime_aggregate_sha256"],
                "market_source_commitments_sha256": market_material[
                    "source_commitments"
                ]["source_commitments_sha256"],
            }
            minted = _mint_feature_row_from_source_bound_components(
                accession_number=current["accession_number"],
                form=current["form"],
                decision_session=current["availability_session"],
                acceptance_datetime=current["acceptance_datetime"],
                artifact_stage=current["artifact_stage"],
                market_lookback_rows=lookback,
                extraction_status=row["extraction_status"],
                extraction_authenticated=row["extraction_authenticated"],
                document_quality=row["document_quality"],
                validated_output=row["validated_output"],
                has_prior_same_form=proof["prior_same_form_record"] is not None,
                upstream_bindings=bindings,
                market_unavailable_authenticated=True,
            )
        except Exception:
            _fail("feature_mint_failed")
        feature_rows.append(minted)
        proof_hashes.append(proof["universe_event_proof_sha256"])
        request_hashes.append(request["request_sha256"])
    source_body = {
        **market_material["source_commitments"],
        "stage": DEVELOPMENT,
        "stage_slice_sha256": fixed["slice_sha256"],
        "model_slice_sha256": rows[0]["model_slice_sha256"],
        "runtime_aggregate_sha256": rows[0]["runtime_aggregate_sha256"],
        "semantic_batch_receipt_sha256": semantic[
            "semantic_batch_receipt_sha256"
        ],
        "model_request_sha256s": request_hashes,
        "universe_event_proof_sha256s": proof_hashes,
        "feature_row_sha256s": [row["feature_row_sha256"] for row in feature_rows],
    }
    source_commitments = {
        **source_body,
        "v39_source_commitments_sha256": contract.canonical_sha256(source_body),
    }
    market_rows = market_material["market_rows"]
    baseline_signals = market_material["baseline_signals"]
    return {
        "market_rows": market_rows,
        "market_rows_sha256": contract.canonical_sha256(market_rows),
        "baseline_signals": baseline_signals,
        "baseline_signals_sha256": contract.canonical_sha256(baseline_signals),
        "feature_rows": feature_rows,
        "feature_row_sha256s": [row["feature_row_sha256"] for row in feature_rows],
        "semantic_batch_receipt_sha256": semantic[
            "semantic_batch_receipt_sha256"
        ],
        "source_commitments": source_commitments,
    }


def evaluate_deterministic_science(
    *, semantic_payload: Mapping[str, Any], deterministic_payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Run and independently rebuild every frozen replay, metric, gate, and proof."""

    try:
        from .sec_gemma_online_risk_overlay_runner import (
            DefaultDeterministicStageEvaluator,
            _stage_input_bundle,
        )

        outputs = {
            "gemma": {"payload": dict(semantic_payload)},
            "deterministic": {"payload": dict(deterministic_payload)},
        }
        input_bundle = _stage_input_bundle(stage=DEVELOPMENT, outputs=outputs)
        evaluator = DefaultDeterministicStageEvaluator()
        evaluation = dict(evaluator.evaluate(stage=DEVELOPMENT, input_bundle=input_bundle))
        replayed = dict(
            evaluator.validate(
                evaluation,
                stage=DEVELOPMENT,
                input_bundle=input_bundle,
            )
        )
    except Exception:
        _fail("deterministic_science_failed")
    if replayed != evaluation:
        _fail("deterministic_science_replay_mismatch")
    return {
        "stage_input_bundle": input_bundle,
        "evaluation": evaluation,
    }


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        _fail("yahoo_redirect_forbidden")


def _yahoo_request_material(url: str) -> dict[str, Any]:
    if url not in contract.YAHOO_URLS:
        _fail("yahoo_url_invalid")
    return {
        "method": "GET",
        "url": url,
        "headers": {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-store",
            "User-Agent": contract.YAHOO_USER_AGENT,
        },
        "body_sha256": None,
    }


def yahoo_request_sha256(url: str) -> str:
    return contract.canonical_sha256(_yahoo_request_material(url))


def _yahoo_content_length(headers: Any) -> int | None:
    transfer = headers.get("Transfer-Encoding")
    if transfer not in {None, "", "identity"}:
        _fail("yahoo_transfer_encoding_invalid")
    values = headers.get_all("Content-Length") or []
    if not values:
        return None
    if len(values) != 1:
        _fail("yahoo_content_length_invalid")
    raw = values[0].strip()
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", raw) is None:
        _fail("yahoo_content_length_invalid")
    length = int(raw)
    if not 1 <= length <= contract.YAHOO_MAX_RESPONSE_BYTES:
        _fail("yahoo_content_length_invalid")
    return length


def fetch_yahoo_once(url: str, *, deadline_monotonic: float) -> tuple[bytes, dict[str, Any]]:
    """Perform one exact no-retry Yahoo request after its durable intent."""

    if url not in contract.YAHOO_URLS:
        _fail("yahoo_url_invalid")
    now = time.monotonic()
    timeout = min(
        contract.YAHOO_REQUEST_TIMEOUT_SECONDS,
        float(deadline_monotonic) - now,
    )
    if not math.isfinite(timeout) or timeout <= 0.0:
        _fail("yahoo_deadline_reached")
    opener = build_opener(
        ProxyHandler({}),
        _RejectRedirect(),
        HTTPSHandler(context=ssl.create_default_context()),
    )
    material = _yahoo_request_material(url)
    request = Request(url, method="GET", headers=material["headers"])
    try:
        with opener.open(request, timeout=timeout) as response:
            headers = response.headers
            declared = _yahoo_content_length(headers)
            read1 = getattr(response, "read1", None)
            if not callable(read1):
                _fail("yahoo_bounded_read_unavailable")
            body = bytearray()
            absolute = min(
                float(deadline_monotonic),
                now + contract.YAHOO_REQUEST_TIMEOUT_SECONDS,
            )
            while True:
                if time.monotonic() >= absolute:
                    _fail("yahoo_deadline_reached")
                read_size = min(64 * 1024, contract.YAHOO_MAX_RESPONSE_BYTES + 1 - len(body))
                chunk = read1(read_size)
                if type(chunk) is not bytes or len(chunk) > read_size:
                    _fail("yahoo_body_invalid")
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > contract.YAHOO_MAX_RESPONSE_BYTES:
                    _fail("yahoo_body_too_large")
                if declared is not None and len(body) > declared:
                    _fail("yahoo_content_length_invalid")
            payload = bytes(body)
            if time.monotonic() > absolute:
                _fail("yahoo_deadline_reached")
            final_url = response.geturl()
            status_code = response.getcode()
            content_type = headers.get_content_type()
            charset = headers.get_content_charset()
            content_encoding = headers.get("Content-Encoding")
    except V39RunnerError:
        raise
    except HTTPError:
        _fail("yahoo_http_failed")
    except (URLError, OSError, TimeoutError):
        _fail("yahoo_transport_failed")
    if (
        not payload
        or (declared is not None and declared != len(payload))
        or final_url != url
        or status_code != 200
        or content_type != "application/json"
        or charset not in {None, "utf-8", "UTF-8"}
        or content_encoding not in {None, "identity"}
    ):
        _fail("yahoo_response_invalid")
    metadata = {
        "request_url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
        "final_url_sha256": hashlib.sha256(final_url.encode("utf-8")).hexdigest(),
        "status_code": 200,
        "content_type": "application/json",
        "charset": charset,
        "content_encoding": content_encoding,
        "declared_content_length": declared,
        "network_requests": 1,
        "retries": 0,
        "redirects": 0,
    }
    return payload, metadata


def _v39_market_plan() -> dict[str, Any]:
    try:
        from .sec_gemma_online_risk_overlay_acquisition import build_acquisition_plan

        plan = build_acquisition_plan(DEVELOPMENT)
    except Exception:
        _fail("market_plan_invalid")
    expected_urls = list(contract.YAHOO_URLS)
    if [item["url"] for item in plan["market"]["requests"]] != expected_urls:
        _fail("market_plan_invalid")
    value = copy.deepcopy(plan)
    value["attempt_id"] = contract.DEVELOPMENT_ATTEMPT_ID
    return value


def build_stage_slice_from_yahoo(
    *, plan: ModelPlan, raw_by_symbol: Mapping[str, bytes]
) -> dict[str, Any]:
    """Open Yahoo values only after all six raw responses are sealed."""

    try:
        from .sec_gemma_online_risk_overlay_acquisition import (
            _build_stage_slice,
            _market_commitments,
        )

        market_plan = _v39_market_plan()
        _commitments, rows = _market_commitments(dict(raw_by_symbol), market_plan)
        return _build_stage_slice(
            plan=market_plan,
            rows_by_symbol=rows,
            model_requests=[dict(item.request) for item in plan.canonical_calls],
            universe_event_proofs=[dict(item.proof) for item in plan.canonical_calls],
        )
    except V39RunnerError:
        raise
    except Exception:
        _fail("market_batch_validation_failed")


def _runtime_probe_request(
    session: Any, *, kind: str
) -> tuple[bytes, dict[str, Any]]:
    try:
        from .sec_filing_gemma_ollama import (
            OLLAMA_SHOW_ENDPOINT,
            OLLAMA_VERSION_ENDPOINT,
            _PROBE_GET_HEADERS,
            _REQUEST_HEADERS,
            _perform_runtime_probe_request,
            _runtime_probe_show_request_bytes,
        )

        if kind == "version":
            body, http = _perform_runtime_probe_request(
                session,
                method="GET",
                endpoint=OLLAMA_VERSION_ENDPOINT,
                headers=_PROBE_GET_HEADERS,
                request_bytes=None,
            )
        elif kind == "show":
            body, http = _perform_runtime_probe_request(
                session,
                method="POST",
                endpoint=OLLAMA_SHOW_ENDPOINT,
                headers=_REQUEST_HEADERS,
                request_bytes=_runtime_probe_show_request_bytes(),
            )
        else:
            _fail("runtime_probe_kind_invalid")
    except V39RunnerError:
        raise
    except Exception:
        _fail("runtime_probe_failed")
    return body, _plain(http)


def runtime_probe_request_sha256(kind: str) -> str:
    """Hash the exact probe envelope; pre and post use identical requests."""

    try:
        from .sec_filing_gemma_ollama import (
            OLLAMA_SHOW_ENDPOINT,
            OLLAMA_VERSION_ENDPOINT,
            _PROBE_GET_HEADERS,
            _REQUEST_HEADERS,
            _runtime_probe_show_request_bytes,
        )

        if kind == "version":
            body = None
            method = "GET"
            endpoint = OLLAMA_VERSION_ENDPOINT
            headers = dict(_PROBE_GET_HEADERS)
        elif kind == "show":
            show_bytes = _runtime_probe_show_request_bytes()
            body = hashlib.sha256(show_bytes).hexdigest()
            method = "POST"
            endpoint = OLLAMA_SHOW_ENDPOINT
            headers = dict(_REQUEST_HEADERS)
        else:
            _fail("runtime_probe_kind_invalid")
    except V39RunnerError:
        raise
    except Exception:
        _fail("runtime_probe_commitment_failed")
    return contract.canonical_sha256(
        {
            "method": method,
            "endpoint": endpoint,
            "headers": headers,
            "body_sha256": body,
        }
    )


def verify_runtime_probe_pair(version_bytes: bytes, show_bytes: bytes) -> dict[str, Any]:
    try:
        from .sec_gemma_online_risk_overlay_runtime import verify_installed_pinned_runtime

        receipt = verify_installed_pinned_runtime(
            version_response_bytes=version_bytes,
            show_response_bytes=show_bytes,
        )
    except Exception:
        _fail("runtime_identity_rejected")
    return dict(receipt)


def _raw_generation_request(
    session: Any, *, request_bytes: bytes
) -> tuple[bytes, dict[str, Any], int]:
    """Seal one opaque response; semantic parsing is deliberately absent."""

    try:
        from .sec_filing_gemma_ollama import (
            CONNECT_TIMEOUT_SECONDS,
            OLLAMA_ENDPOINT,
            READ_TIMEOUT_SECONDS,
            _REQUEST_HEADERS,
            _read_bounded_response,
            _validate_http_envelope,
        )
    except Exception:
        _fail("ollama_transport_unavailable")
    response: Any | None = None
    start_ns = time.monotonic_ns()
    try:
        response = session.request(
            "POST",
            OLLAMA_ENDPOINT,
            headers=dict(_REQUEST_HEADERS),
            data=request_bytes,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            allow_redirects=False,
            stream=True,
        )
        response_bytes = _read_bounded_response(response)
        http = _validate_http_envelope(response, response_bytes)
    except Exception:
        _fail("ollama_generation_failed")
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
    end_ns = time.monotonic_ns()
    duration = end_ns - start_ns
    if duration <= 0:
        _fail("ollama_generation_clock_invalid")
    return response_bytes, _plain(http), duration


def _checkpoint_state(*, phase: str, request_sha256: str, body_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "aapl-sec-gemma-lean-science-v3-9-checkpoint-state-v1",
        "phase": phase,
        "request_sha256": _sha(request_sha256),
        "body_sha256": _sha(body_sha256),
    }


def _commit_effect(
    store: Any,
    *,
    intent: Any,
    body: bytes,
    metadata: Mapping[str, Any],
    phase: str,
    duration_ns: int | None = None,
) -> Any:
    request_sha256 = _sha(getattr(intent, "request_sha256", None))
    response, _checkpoint = store.commit_response_and_checkpoint(
        request_intent_event_sha256=intent.event_sha256,
        body=body,
        metadata=dict(metadata),
        duration_ns=duration_ns,
        checkpoint_state=_checkpoint_state(
            phase=phase,
            request_sha256=request_sha256,
            body_sha256=hashlib.sha256(body).hexdigest(),
        ),
    )
    return response


def execute_yahoo_effects(
    store: Any,
    *,
    fetcher: Callable[[str], tuple[bytes, Mapping[str, Any]]] | None = None,
) -> None:
    """Execute only missing Yahoo units from an authenticated safe boundary."""

    snapshot = store.snapshot
    if snapshot.status not in {"active"}:
        _fail("yahoo_store_state_invalid")
    start = snapshot.yahoo_response_count
    if not 0 <= start <= contract.YAHOO_REQUEST_COUNT:
        _fail("yahoo_store_state_invalid")
    deadline = time.monotonic() + float(contract.YAHOO_MAX_BATCH_SECONDS)
    for index in range(start, contract.YAHOO_REQUEST_COUNT):
        url = contract.YAHOO_URLS[index]
        request_sha = yahoo_request_sha256(url)
        intent = store.begin_request(
            request_id=f"yahoo-{index + 1:02d}",
            effect_kind="yahoo",
            request_sha256=request_sha,
            segment_id=None,
            model_phase="none",
        )
        if fetcher is None:
            body, metadata = fetch_yahoo_once(url, deadline_monotonic=deadline)
        else:
            body, metadata = fetcher(url)
        if (
            type(body) is not bytes
            or not 1 <= len(body) <= contract.YAHOO_MAX_RESPONSE_BYTES
        ):
            _fail("yahoo_response_bytes_invalid")
        if (
            store.snapshot.yahoo_body_bytes + len(body)
            > contract.YAHOO_MAX_TOTAL_RESPONSE_BYTES
        ):
            _fail("yahoo_batch_bytes_exceeded")
        metadata_value = dict(metadata)
        metadata_value["symbol"] = contract.YAHOO_SYMBOL_ORDER[index]
        _commit_effect(
            store,
            intent=intent,
            body=body,
            metadata=metadata_value,
            phase="none",
        )


def execute_model_segment(
    store: Any,
    *,
    calls: Sequence[PreparedModelCall],
    segment_id: str,
    probe_request: Callable[[str], tuple[bytes, Mapping[str, Any]]] | None = None,
    generation_request: Callable[[bytes], tuple[bytes, Mapping[str, Any], int]] | None = None,
) -> dict[str, Any]:
    """Execute one uninterrupted pre/generation/post guarded model segment.

    The first physical segment uses ``segment_id='initial'`` because its
    scientific role (``full`` or ``pilot``) is unknowable until the fifth
    measured response is durably sealed.  The returned guard records both the
    physical identifier and the role selected solely from those durations.
    """

    if segment_id not in {"initial", "continuation"}:
        _fail("model_segment_invalid")
    expected_calls = TOTAL_CALL_COUNT if segment_id == "initial" else REMAINING_CALL_COUNT
    if len(calls) != expected_calls:
        _fail("model_segment_call_count_invalid")
    try:
        from .sec_filing_gemma_ollama import build_hardened_loopback_session

        session = build_hardened_loopback_session() if (probe_request is None or generation_request is None) else None
    except Exception:
        _fail("ollama_session_unavailable")

    def probe(kind: str) -> tuple[bytes, Mapping[str, Any]]:
        return _runtime_probe_request(session, kind=kind) if probe_request is None else probe_request(kind)

    def generate(request_bytes: bytes) -> tuple[bytes, Mapping[str, Any], int]:
        return _raw_generation_request(session, request_bytes=request_bytes) if generation_request is None else generation_request(request_bytes)

    def validate_probe_body(body: bytes) -> None:
        if type(body) is not bytes or not (
            1 <= len(body) <= contract.MODEL_RESPONSE_MAX_BYTES
        ):
            _fail("runtime_probe_response_bytes_invalid")

    response_events: list[str] = []
    duration_by_request: dict[str, int] = {}
    try:
        version_intent = store.begin_request(
            request_id=f"{segment_id}-pre-version",
            effect_kind="identity",
            request_sha256=runtime_probe_request_sha256("version"),
            segment_id=segment_id,
            model_phase="pre_probe_start",
        )
        version, version_meta = probe("version")
        validate_probe_body(version)
        _commit_effect(
            store,
            intent=version_intent,
            body=version,
            metadata=version_meta,
            phase="pre_probe_start",
        )
        show_intent = store.begin_request(
            request_id=f"{segment_id}-pre-show",
            effect_kind="identity",
            request_sha256=runtime_probe_request_sha256("show"),
            segment_id=segment_id,
            model_phase="pre_probe",
        )
        show, show_meta = probe("show")
        validate_probe_body(show)
        _commit_effect(
            store,
            intent=show_intent,
            body=show,
            metadata=show_meta,
            phase="pre_probe",
        )
        # Identity bytes are opened only after the complete two-response
        # pre-probe is sealed marker-last.
        sealed_pre = store.committed_responses(
            effect_kind="identity", segment_id=segment_id
        )
        if len(sealed_pre) != 2:
            _fail("runtime_pre_probe_seal_invalid")
        pre_receipt = verify_runtime_probe_pair(
            sealed_pre[0].body, sealed_pre[1].body
        )
        for item in calls:
            generation_intent = store.begin_request(
                request_id=f"generation-{item.execution_ordinal:03d}",
                effect_kind="gemma",
                request_sha256=item.request["request_sha256"],
                segment_id=segment_id,
                model_phase="generation",
            )
            response_bytes, http, duration_ns = generate(item.request_bytes)
            if type(response_bytes) is not bytes or not (
                1 <= len(response_bytes) <= contract.MODEL_RESPONSE_MAX_BYTES
            ):
                _fail("model_response_bytes_invalid")
            response = _commit_effect(
                store,
                intent=generation_intent,
                body=response_bytes,
                metadata=http,
                duration_ns=duration_ns,
                phase="generation",
            )
            response_events.append(response.event_sha256)
            duration_by_request[item.request["request_sha256"]] = duration_ns
            if (
                segment_id == "initial"
                and len(response_events) == PILOT_COUNT
                and store.snapshot.pause_required
            ):
                break
        guard_role = (
            "continuation"
            if segment_id == "continuation"
            else "pilot"
            if len(response_events) == PILOT_COUNT
            else "full"
        )
        post_version_intent = store.begin_request(
            request_id=f"{segment_id}-post-version",
            effect_kind="identity",
            request_sha256=runtime_probe_request_sha256("version"),
            segment_id=segment_id,
            model_phase="post_probe",
        )
        post_version, post_version_meta = probe("version")
        validate_probe_body(post_version)
        _commit_effect(
            store,
            intent=post_version_intent,
            body=post_version,
            metadata=post_version_meta,
            phase="post_probe",
        )
        post_show_intent = store.begin_request(
            request_id=f"{segment_id}-post-show",
            effect_kind="identity",
            request_sha256=runtime_probe_request_sha256("show"),
            segment_id=segment_id,
            model_phase="post_probe_close",
        )
        post_show, post_show_meta = probe("show")
        validate_probe_body(post_show)
        _commit_effect(
            store,
            intent=post_show_intent,
            body=post_show,
            metadata=post_show_meta,
            phase="post_probe_close",
        )
        # Generation bodies remain opaque.  Only the sealed post-probe is
        # opened here to establish the candidate-authoritative guard.
        sealed_segment = store.committed_responses(
            effect_kind="identity", segment_id=segment_id
        )
        if len(sealed_segment) != 4:
            _fail("runtime_post_probe_seal_invalid")
        post_receipt = verify_runtime_probe_pair(
            sealed_segment[2].body, sealed_segment[3].body
        )
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
    guard = build_runtime_segment_guard(
        segment_id=guard_role,
        store_segment_id=segment_id,
        pre_runtime_receipt=pre_receipt,
        post_runtime_receipt=post_receipt,
        ordered_generation_response_event_sha256s=response_events,
        generation_count=len(response_events),
    )
    return {
        "store_segment_id": segment_id,
        "guard_role": guard_role,
        "runtime_guard": guard,
        "duration_ns_by_request_sha256": duration_by_request,
        "pause_required": guard_role == "pilot",
    }


def rebuild_closed_segment_guard(
    store: Any, *, store_segment_id: str, guard_role: str
) -> dict[str, Any]:
    """Rebuild one guard from marker-last evidence, without model bodies."""

    if store_segment_id not in {"initial", "continuation"}:
        _fail("model_segment_invalid")
    if guard_role not in {"full", "pilot", "continuation"}:
        _fail("runtime_segment_invalid")
    identities = store.committed_responses(
        effect_kind="identity", segment_id=store_segment_id
    )
    generations = store.committed_response_receipts(
        effect_kind="gemma", segment_id=store_segment_id
    )
    if len(identities) != 4:
        _fail("runtime_post_probe_seal_invalid")
    pre = verify_runtime_probe_pair(identities[0].body, identities[1].body)
    post = verify_runtime_probe_pair(identities[2].body, identities[3].body)
    return build_runtime_segment_guard(
        segment_id=guard_role,
        store_segment_id=store_segment_id,
        pre_runtime_receipt=pre,
        post_runtime_receipt=post,
        ordered_generation_response_event_sha256s=[
            item.event_sha256 for item in generations
        ],
        generation_count=len(generations),
    )


def rebuild_runtime_evidence(
    store: Any, *, plan: ModelPlan
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Rebuild segment/aggregate/timing provenance from the private journal."""

    summaries = list(store.snapshot.segment_summaries)
    if [item.generation_count for item in summaries] == [TOTAL_CALL_COUNT]:
        layout = [(summaries[0].segment_id, "full")]
    elif [item.generation_count for item in summaries] == [PILOT_COUNT, REMAINING_CALL_COUNT]:
        layout = [
            (summaries[0].segment_id, "pilot"),
            (summaries[1].segment_id, "continuation"),
        ]
    else:
        _fail("runtime_segment_layout_invalid")
    guards = [
        rebuild_closed_segment_guard(
            store, store_segment_id=physical, guard_role=role
        )
        for physical, role in layout
    ]
    aggregate = build_runtime_aggregate_guard(
        segment_guards=guards,
        execution_request_sha256s=[
            item.request["request_sha256"] for item in plan.execution_calls
        ],
    )
    all_generation_receipts: list[Any] = []
    guard_by_request: dict[str, dict[str, Any]] = {}
    durations: dict[str, int] = {}
    for (physical, _role), guard in zip(layout, guards, strict=True):
        receipts = list(
            store.committed_response_receipts(
                effect_kind="gemma", segment_id=physical
            )
        )
        all_generation_receipts.extend(receipts)
        for receipt in receipts:
            request_sha = receipt.intent.request_sha256
            if request_sha in guard_by_request or receipt.duration_ns is None:
                _fail("runtime_generation_receipt_invalid")
            guard_by_request[request_sha] = guard
            durations[request_sha] = receipt.duration_ns
    expected_order = [
        item.request["request_sha256"] for item in plan.execution_calls
    ]
    if [item.intent.request_sha256 for item in all_generation_receipts] != expected_order:
        _fail("runtime_generation_order_invalid")
    latency = build_latency_receipt(
        plan=plan, durations_ns_by_request_sha256=durations
    )
    return guards, aggregate, latency, guard_by_request


def build_effect_report(snapshot: Any) -> dict[str, Any]:
    """Expose honest attempted/completed counts without private request data."""

    attempted = {
        "yahoo_requests": int(snapshot.yahoo_intent_count),
        "ollama_identity_http_requests": int(snapshot.identity_intent_count),
        "ollama_chat_generations": int(snapshot.gemma_intent_count),
    }
    completed = {
        "sec_requests": 0,
        "experiment_family_sec_requests": contract.EXPERIMENT_FAMILY_SEC_REQUEST_COUNT,
        "yahoo_requests": int(snapshot.yahoo_response_count),
        "ollama_identity_http_requests": int(snapshot.identity_response_count),
        "ollama_chat_generations": int(snapshot.gemma_response_count),
        "retries": 0,
        "repairs": 0,
        "pulls": 0,
        "fallbacks": 0,
        "paid_calls": 0,
        "confirmation_final_data_opens": 0,
        "broker_effects": 0,
        "real_money_effects": 0,
    }
    return {
        "attempted_external_requests": attempted,
        "completed_effect_counts": completed,
        "request_intent_count": int(snapshot.request_intent_count),
        "response_count": int(snapshot.response_count),
        "checkpoint_count": int(snapshot.checkpoint_count),
        "yahoo_body_bytes": int(snapshot.yahoo_body_bytes),
        "no_retry_repair_pull_fallback_paid_or_trading_effect": True,
    }


def build_public_effect_report(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove private transport-size metadata from a public effect receipt."""

    item = _mapping(value, "effect_report_invalid")
    expected = {
        "attempted_external_requests",
        "completed_effect_counts",
        "request_intent_count",
        "response_count",
        "checkpoint_count",
        "yahoo_body_bytes",
        "no_retry_repair_pull_fallback_paid_or_trading_effect",
    }
    if set(item) != expected:
        _fail("effect_report_invalid")
    plain = _plain(item)
    return {
        key: child
        for key, child in plain.items()
        if key != "yahoo_body_bytes"
    }


def build_pilot_pause_effect_report(snapshot: Any) -> dict[str, Any]:
    """Rebuild the immutable effect boundary captured by the pilot pause."""

    if (
        int(snapshot.yahoo_intent_count) < contract.YAHOO_REQUEST_COUNT
        or int(snapshot.yahoo_response_count) < contract.YAHOO_REQUEST_COUNT
        or int(snapshot.identity_intent_count) < 4
        or int(snapshot.identity_response_count) < 4
        or int(snapshot.gemma_intent_count) < PILOT_COUNT
        or int(snapshot.gemma_response_count) < PILOT_COUNT
        or int(snapshot.yahoo_body_bytes) <= 0
        or snapshot.paused is not True
    ):
        _fail("pilot_pause_effect_boundary_invalid")
    request_count = contract.YAHOO_REQUEST_COUNT + 4 + PILOT_COUNT
    completed = {
        "sec_requests": 0,
        "experiment_family_sec_requests": contract.EXPERIMENT_FAMILY_SEC_REQUEST_COUNT,
        "yahoo_requests": contract.YAHOO_REQUEST_COUNT,
        "ollama_identity_http_requests": 4,
        "ollama_chat_generations": PILOT_COUNT,
        "retries": 0,
        "repairs": 0,
        "pulls": 0,
        "fallbacks": 0,
        "paid_calls": 0,
        "confirmation_final_data_opens": 0,
        "broker_effects": 0,
        "real_money_effects": 0,
    }
    return {
        "attempted_external_requests": {
            "yahoo_requests": contract.YAHOO_REQUEST_COUNT,
            "ollama_identity_http_requests": 4,
            "ollama_chat_generations": PILOT_COUNT,
        },
        "completed_effect_counts": completed,
        "request_intent_count": request_count,
        "response_count": request_count,
        "checkpoint_count": request_count + 1,
        "yahoo_body_bytes": int(snapshot.yahoo_body_bytes),
        "no_retry_repair_pull_fallback_paid_or_trading_effect": True,
    }


def _redact_action_report(value: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(value)
    return {
        key: copy.deepcopy(child)
        for key, child in item.items()
        if key != "differences"
    }


def _redact_attribution(value: Mapping[str, Any], *, row_field: str) -> dict[str, Any]:
    item = dict(value)
    return {
        key: copy.deepcopy(child)
        for key, child in item.items()
        if key != row_field
    }


def build_public_science_summary(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Project aggregate diagnostics only; no accession or daily source row."""

    value = _mapping(evaluation, "deterministic_evaluation_invalid")
    if value.get("deterministic_evaluation_sha256") != contract.canonical_sha256(
        {
            key: child
            for key, child in value.items()
            if key != "deterministic_evaluation_sha256"
        }
    ):
        _fail("deterministic_evaluation_invalid")
    metrics = _mapping(value.get("stage_metrics"), "stage_metrics_invalid")
    gate = _mapping(value.get("gate_report"), "gate_report_invalid")
    if (
        metrics.get("stage_metrics_sha256")
        != value.get("stage_metrics_sha256")
        or gate.get("gate_report_sha256") != value.get("gate_report_sha256")
        or gate.get("stage_metrics_sha256") != metrics.get("stage_metrics_sha256")
    ):
        _fail("deterministic_evaluation_invalid")
    coverage = _mapping(metrics.get("coverage_metrics"), "coverage_metrics_invalid")
    brier = _mapping(metrics.get("brier_metrics"), "brier_metrics_invalid")
    action_fields = (
        "semantic_vs_no_filing_meaning_action_differences",
        "semantic_vs_no_gemma_channel_action_differences",
        "online_vs_frozen_action_differences",
    )
    attribution_fields = (
        ("semantic_vs_no_filing_meaning_xor_attribution_10bps", "intervals"),
        ("semantic_vs_no_gemma_channel_xor_attribution_10bps", "intervals"),
        ("semantic_overlay_episode_attribution_10bps", "episodes"),
    )
    proofs = _mapping(value.get("no_leverage_proofs"), "no_leverage_proofs_invalid")
    if value.get("no_leverage_proofs_sha256") != contract.canonical_sha256(proofs):
        _fail("no_leverage_proofs_invalid")
    summary = {
        "stage_input_bundle_sha256": value["stage_input_bundle_sha256"],
        "deterministic_evaluation_sha256": value["deterministic_evaluation_sha256"],
        "stage_metrics_sha256": metrics["stage_metrics_sha256"],
        "gate_report": copy.deepcopy(gate),
        "window_metrics": copy.deepcopy(metrics["window_metrics"]),
        "window_metrics_sha256": metrics["window_metrics_sha256"],
        "frozen_window_metrics": copy.deepcopy(metrics["frozen_window_metrics"]),
        "frozen_window_metrics_sha256": metrics["frozen_window_metrics_sha256"],
        "calendar_diagnostics": copy.deepcopy(metrics["calendar_diagnostics"]),
        "development_block_diagnostics": copy.deepcopy(
            metrics["development_block_diagnostics"]
        ),
        "action_difference_summaries": {
            field: _redact_action_report(_mapping(metrics[field], "action_report_invalid"))
            for field in action_fields
        },
        "attribution_summaries": {
            field: _redact_attribution(
                _mapping(metrics[field], "attribution_invalid"), row_field=row_field
            )
            for field, row_field in attribution_fields
        },
        "coverage": {
            key: copy.deepcopy(child)
            for key, child in coverage.items()
            if key != "included_feature_row_sha256s"
        },
        "brier": {
            key: copy.deepcopy(child)
            for key, child in brier.items()
            if key != "rows"
        },
        "no_leverage_proofs": copy.deepcopy(proofs),
        "no_leverage_proofs_sha256": value["no_leverage_proofs_sha256"],
    }
    encoded = contract.canonical_json_bytes(summary)
    if re.search(rb"[0-9]{10}-[0-9]{2}-[0-9]{6}", encoded):
        _fail("public_science_summary_private_identity")
    return summary


def build_private_terminal_material(
    *,
    authority: Mapping[str, Any],
    projection: Any,
    plan: ModelPlan,
    runtime_guards: Sequence[Mapping[str, Any]],
    runtime_aggregate: Mapping[str, Any],
    latency_receipt: Mapping[str, Any],
    stage_slice: Mapping[str, Any],
    semantic_payload: Mapping[str, Any],
    deterministic_payload: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    effect_report: Mapping[str, Any],
    invocation_parent: str,
    invocation_parent_kind: str,
) -> dict[str, Any]:
    """Build compact replay material; raw transport bodies stay in the store."""

    route = _terminal_route(
        terminal_status="completed", parent_kind=invocation_parent_kind
    )
    if type(invocation_parent) is not str or re.fullmatch(r"[0-9a-f]{40}", invocation_parent) is None:
        _fail("invocation_parent_invalid")
    science_summary = build_public_science_summary(evaluation)
    body = {
        "schema_version": PRIVATE_TERMINAL_SCHEMA_VERSION,
        "stage": DEVELOPMENT,
        "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
        "invocation_parent": invocation_parent,
        "invocation_parent_kind": invocation_parent_kind,
        "route": route,
        "authority_sha256": contract.canonical_sha256(authority),
        "bridge_manifest": copy.deepcopy(
            _mapping(_projection_attr(projection, "manifest"), "bridge_manifest_invalid")
        ),
        "model_plan_manifest": copy.deepcopy(dict(plan.manifest)),
        "runtime_segment_guards": [copy.deepcopy(dict(item)) for item in runtime_guards],
        "runtime_aggregate": copy.deepcopy(dict(runtime_aggregate)),
        "latency_receipt": copy.deepcopy(dict(latency_receipt)),
        "stage_slice_sha256": _sha(stage_slice.get("slice_sha256")),
        "semantic_payload": copy.deepcopy(dict(semantic_payload)),
        "semantic_payload_sha256": contract.canonical_sha256(semantic_payload),
        "deterministic_payload": copy.deepcopy(dict(deterministic_payload)),
        "deterministic_payload_sha256": contract.canonical_sha256(deterministic_payload),
        "science_summary": science_summary,
        "science_summary_sha256": contract.canonical_sha256(science_summary),
        "effect_report": copy.deepcopy(dict(effect_report)),
        "market_values_opened": True,
        "model_responses_opened": True,
        "confirmation_and_final_opened": False,
        "raw_sec_yahoo_or_gemma_response_copied": False,
    }
    return {**body, "private_terminal_material_sha256": contract.canonical_sha256(body)}


def build_public_pause_artifact(
    *,
    authority: Mapping[str, Any],
    plan: ModelPlan,
    pilot_guard: Mapping[str, Any],
    latency_receipt: Mapping[str, Any],
    effect_report: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": PAUSE_SCHEMA_VERSION,
        "status": "paused_for_justification",
        "stage": DEVELOPMENT,
        "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
        "branch": contract.BRANCH_NAME,
        "preflight_commit": authority["preflight"]["commit"],
        "implementation_commit": authority["implementation"]["commit"],
        "source_bridge_sha256": authority["source"]["bridge_sha256"],
        "science_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
        "model_plan_sha256": plan.manifest["model_plan_sha256"],
        "pilot_count": PILOT_COUNT,
        "pilot_durations_ns": [
            row["duration_ns"] for row in latency_receipt["pilot_rows"]
        ],
        "formula": latency_receipt["formula"],
        "projected_ns": latency_receipt["projected_ns"],
        "threshold_ns": latency_receipt["threshold_ns"],
        "strictly_greater_pause": True,
        "pilot_guard_sha256": pilot_guard["segment_guard_sha256"],
        "latency_receipt_sha256": latency_receipt["latency_receipt_sha256"],
        "remaining_order_sha256": plan.manifest["remaining_order_sha256"],
        "effect_report": build_public_effect_report(effect_report),
        "model_responses_opened": False,
        "market_values_opened": False,
        "sixth_generation_attempted": False,
        "continuation_requires_fresh_explicit_permission": True,
        "confirmation_and_final_opened": False,
        "privacy_passed": True,
    }
    return {**body, "pause_artifact_sha256": contract.canonical_sha256(body)}


def build_failure_terminal_material(
    *,
    authority: Mapping[str, Any],
    terminal_code: str,
    terminal_status: str,
    effect_report: Mapping[str, Any],
    snapshot: Any,
    invocation_parent: str,
    invocation_parent_kind: str,
    market_values_opened: bool,
    model_responses_opened: bool,
    source_authenticated: bool,
) -> dict[str, Any]:
    if terminal_status not in {"rejected", "indeterminate"}:
        _fail("terminal_status_invalid")
    if (
        type(invocation_parent) is not str
        or re.fullmatch(r"[0-9a-f]{40}", invocation_parent) is None
        or type(market_values_opened) is not bool
        or type(model_responses_opened) is not bool
        or type(source_authenticated) is not bool
    ):
        _fail("terminal_context_invalid")
    route = _terminal_route(
        terminal_status=terminal_status, parent_kind=invocation_parent_kind
    )
    body = {
        "schema_version": PRIVATE_TERMINAL_SCHEMA_VERSION,
        "stage": DEVELOPMENT,
        "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
        "invocation_parent": invocation_parent,
        "invocation_parent_kind": invocation_parent_kind,
        "route": route,
        "authority_sha256": contract.canonical_sha256(authority),
        "terminal_status": terminal_status,
        "terminal_code": terminal_code,
        "effect_report": copy.deepcopy(dict(effect_report)),
        "journal_head_before_terminal_sha256": snapshot.journal_head_sha256,
        "closed_segment_generation_counts": [
            item.generation_count for item in snapshot.segment_summaries
        ],
        "market_values_opened": market_values_opened,
        "model_responses_opened": model_responses_opened,
        "source_authenticated": source_authenticated,
        "confirmation_and_final_opened": False,
        "redacted_error_only": True,
    }
    return {**body, "private_terminal_material_sha256": contract.canonical_sha256(body)}


def _terminal_route(*, terminal_status: str, parent_kind: str) -> str:
    if parent_kind == "continuation":
        return "paused_resumed"
    if parent_kind != "preflight":
        _fail("terminal_parent_kind_invalid")
    return "indeterminate_before_pause" if terminal_status == "indeterminate" else "normal"


def build_public_terminal_artifact(
    *,
    authority: Mapping[str, Any],
    terminal_receipt: Any,
) -> dict[str, Any]:
    evidence = _mapping(
        getattr(terminal_receipt, "evidence", None), "terminal_evidence_invalid"
    )
    status = getattr(terminal_receipt, "status", None)
    terminal_code = getattr(terminal_receipt, "terminal_code", None)
    if status not in {"completed", "rejected", "indeterminate"}:
        _fail("terminal_evidence_invalid")
    science_summary = evidence.get("science_summary")
    gate_passed: bool | None = None
    failed_checks: list[str] = []
    if science_summary is not None:
        summary = _mapping(science_summary, "terminal_science_summary_invalid")
        gate = _mapping(summary.get("gate_report"), "terminal_science_summary_invalid")
        gate_passed = gate.get("passed")
        failed_checks = list(gate.get("failed_checks", ()))
        if type(gate_passed) is not bool or any(type(item) is not str for item in failed_checks):
            _fail("terminal_science_summary_invalid")
    outcome = (
        "passed"
        if status == "completed" and gate_passed is True
        else "failed_gate"
        if status == "completed"
        else status
    )
    invocation_parent = evidence.get("invocation_parent")
    invocation_parent_kind = evidence.get("invocation_parent_kind")
    route = evidence.get("route")
    expected_route = _terminal_route(
        terminal_status=status, parent_kind=invocation_parent_kind
    )
    if (
        type(invocation_parent) is not str
        or re.fullmatch(r"[0-9a-f]{40}", invocation_parent) is None
        or route != expected_route
    ):
        _fail("terminal_invocation_parent_invalid")
    market_values_opened = evidence.get("market_values_opened")
    model_responses_opened = evidence.get("model_responses_opened")
    if type(market_values_opened) is not bool or type(model_responses_opened) is not bool:
        _fail("terminal_open_state_invalid")
    body = {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "status": outcome,
        "terminal_store_status": status,
        "terminal_code": terminal_code,
        "stage": DEVELOPMENT,
        "route": route,
        "invocation_parent": invocation_parent,
        "invocation_parent_kind": invocation_parent_kind,
        "attempt_id": contract.DEVELOPMENT_ATTEMPT_ID,
        "branch": contract.BRANCH_NAME,
        "implementation_commit": authority["implementation"]["commit"],
        "implementation_tree": authority["implementation"]["tree"],
        "preflight_commit": authority["preflight"]["commit"],
        "preflight_tree": authority["preflight"]["tree"],
        "source_authority": copy.deepcopy(dict(authority["source"])),
        "science_projection_sha256": contract.SCIENCE_PROJECTION_SHA256,
        "contract_manifest_sha256": contract.CONTRACT_MANIFEST_SHA256,
        "private_terminal_evidence_sha256": terminal_receipt.evidence_sha256,
        "private_terminal_evidence_bytes": terminal_receipt.evidence_bytes,
        "journal_head_before_terminal_sha256": (
            terminal_receipt.journal_head_before_terminal_sha256
        ),
        "terminal_event_sha256": terminal_receipt.event_sha256,
        "effect_report": build_public_effect_report(evidence["effect_report"]),
        "market_values_opened": market_values_opened,
        "model_responses_opened": model_responses_opened,
        "science_summary": _plain(science_summary) if science_summary is not None else None,
        "development_gate_passed": gate_passed,
        "failed_gate_count": len(failed_checks),
        "failed_gate_names": failed_checks,
        "chronology": {
            "development_only": True,
            "development_end": contract.DEVELOPMENT_CORPUS_END,
            "confirmation_2019_2023_opened": False,
            "final_2024_plus_opened": False,
        },
        "privacy": {
            "readable_sec_contact_published": False,
            "accessions_urls_filenames_offsets_or_bodies_published": False,
            "canonical_model_requests_published": False,
            "raw_sec_yahoo_or_gemma_responses_published": False,
            "realized_transport_metadata_published": False,
            "redacted_errors_only": True,
            "privacy_passed": True,
        },
        "evidence_limitations": [
            "retrospective_development_evidence_only",
            "confirmation_and_final_remain_closed",
            "no_real_money_execution_authorized",
            "pushed_result_gate_required_before_any_later_stage",
        ],
        "real_money_authorized": False,
    }
    return {**body, "development_result_sha256": contract.canonical_sha256(body)}


def assert_public_artifact_private_free(
    value: Mapping[str, Any], *, forbidden_tokens: Sequence[bytes | str]
) -> None:
    payload = contract.canonical_json_bytes(value)
    folded_payload = payload.lower()
    text_tokens: list[str] = []
    for token in forbidden_tokens:
        encoded = token.encode("utf-8") if type(token) is str else token
        if type(encoded) is not bytes or not encoded or not encoded.isascii():
            _fail("public_artifact_privacy_failed")
        variants = {encoded}
        try:
            text = encoded.decode("utf-8", errors="strict")
        except UnicodeError:
            text = ""
        if text:
            text_tokens.append(text.casefold())
            variants.add(json.dumps(text, ensure_ascii=True)[1:-1].encode("utf-8"))
            variants.add(json.dumps(text, ensure_ascii=False)[1:-1].encode("utf-8"))
        if any(item.lower() in folded_payload for item in variants):
            _fail("public_artifact_privacy_failed")

    def inspect_strings(item: Any) -> None:
        if type(item) is str:
            folded = item.casefold()
            if any(token in folded for token in text_tokens):
                _fail("public_artifact_privacy_failed")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                inspect_strings(key)
                inspect_strings(child)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                inspect_strings(child)

    inspect_strings(value)
    if re.search(rb"[0-9]{10}-[0-9]{2}-[0-9]{6}", payload):
        _fail("public_artifact_privacy_failed")
    forbidden_keys = (
        b'"accession_number"',
        b'"request_bytes"',
        b'"response_bytes"',
        b'"body_base64"',
        b'"selected_filename"',
        b'"official_url"',
    )
    if any(item in payload for item in forbidden_keys):
        _fail("public_artifact_privacy_failed")


def assert_private_namespace_private_free(
    private_root: Path,
    *,
    forbidden_tokens: Sequence[bytes | str],
    locked_payloads: Mapping[str, bytes] | None = None,
) -> None:
    """Stream-scan every serialized private file without exposing its content."""

    variants: set[bytes] = set()
    for token in forbidden_tokens:
        encoded = token.encode("utf-8") if type(token) is str else token
        if type(encoded) is not bytes or not encoded or not encoded.isascii():
            _fail("private_namespace_privacy_failed")
        variants.add(encoded.lower())
        try:
            text = encoded.decode("utf-8", errors="strict")
        except UnicodeError:
            continue
        variants.add(
            json.dumps(text, ensure_ascii=True)[1:-1].encode("utf-8").lower()
        )
        variants.add(
            json.dumps(text, ensure_ascii=False)[1:-1].encode("utf-8").lower()
        )
    if not variants:
        _fail("private_namespace_privacy_failed")
    maximum = max(len(item) for item in variants)
    reparse_attribute = 0x400
    locked: dict[str, bytes] = {}
    if locked_payloads is not None:
        if not isinstance(locked_payloads, Mapping):
            _fail("private_namespace_privacy_failed")
        for relative, payload in locked_payloads.items():
            if (
                type(relative) is not str
                or not relative
                or "\\" in relative
                or relative.startswith("/")
                or ".." in relative.split("/")
                or type(payload) is not bytes
                or not payload
            ):
                _fail("private_namespace_privacy_failed")
            locked[relative] = payload

    def is_reparse(details: os.stat_result) -> bool:
        return bool(getattr(details, "st_file_attributes", 0) & reparse_attribute)

    def scan_bytes(payload: bytes) -> None:
        folded = payload.lower()
        if any(token in folded for token in variants):
            _fail("private_namespace_privacy_failed")

    def scan_file(path: Path, details: os.stat_result) -> None:
        try:
            relative = path.relative_to(private_root).as_posix()
        except ValueError:
            _fail("private_namespace_privacy_failed")
        if relative in locked:
            payload = locked.pop(relative)
            if details.st_size != len(payload):
                _fail("private_namespace_privacy_failed")
            scan_bytes(payload)
            return
        overlap = b""
        try:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    window = (overlap + chunk).lower()
                    if any(token in window for token in variants):
                        _fail("private_namespace_privacy_failed")
                    overlap = window[-(maximum - 1) :] if maximum > 1 else b""
        except V39RunnerError:
            raise
        except OSError:
            _fail("private_namespace_privacy_failed")

    def descend(directory: Path) -> None:
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            _fail("private_namespace_privacy_failed")
        for entry in entries:
            path = Path(entry.path)
            try:
                details = entry.stat(follow_symlinks=False)
            except OSError:
                _fail("private_namespace_privacy_failed")
            if entry.is_symlink() or is_reparse(details):
                _fail("private_namespace_privacy_failed")
            if stat.S_ISDIR(details.st_mode):
                descend(path)
            elif stat.S_ISREG(details.st_mode) and details.st_nlink in {0, 1}:
                scan_file(path, details)
            else:
                _fail("private_namespace_privacy_failed")

    try:
        root_details = private_root.lstat()
    except OSError:
        _fail("private_namespace_privacy_failed")
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or stat.S_ISLNK(root_details.st_mode)
        or is_reparse(root_details)
    ):
        _fail("private_namespace_privacy_failed")
    descend(private_root)
    if locked:
        _fail("private_namespace_privacy_failed")


def _assert_complete_v39_private_namespace_private_free(
    store: Any, *, forbidden_tokens: Sequence[bytes | str]
) -> None:
    """Scan both the preflight and development children, including the lock."""

    try:
        store_root = Path(store.root)
        private_root = store_root.parent
        child = store_root.relative_to(private_root).as_posix()
        if child != Path(contract.PRIVATE_DEVELOPMENT_NAMESPACE).name:
            _fail("private_namespace_privacy_failed")
        locked_payloads = {
            f"{child}/{relative}": payload
            for relative, payload in store.privacy_scan_locked_payloads().items()
        }
    except V39RunnerError:
        raise
    except Exception:
        _fail("private_namespace_privacy_failed")
    assert_private_namespace_private_free(
        private_root,
        forbidden_tokens=forbidden_tokens,
        locked_payloads=locked_payloads,
    )


def _fsync_public_directory(path: Path) -> None:
    try:
        from .sec_gemma_lean_science_v39_journal import fsync_directory

        fsync_directory(path)
    except Exception:
        _fail("public_artifact_directory_sync_failed")


def _assert_exact_regular_file(path: Path, payload: bytes, *, code: str) -> None:
    try:
        details = path.lstat()
        observed = path.read_bytes()
    except OSError:
        _fail(code)
    if (
        not stat.S_ISREG(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or getattr(details, "st_file_attributes", 0) & 0x400
        or details.st_nlink not in {0, 1}
        or details.st_size != len(payload)
        or observed != payload
    ):
        _fail(code)


def _path_entry_exists(path: Path, *, code: str) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        _fail(code)
    return True


def _write_new_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = contract.canonical_json_bytes(value)
    pending = path.with_name(path.name + ".v39-pending")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if _path_entry_exists(path, code="public_artifact_unwritable"):
            if _path_entry_exists(
                pending, code="public_artifact_recovery_state_invalid"
            ):
                _fail("public_artifact_recovery_state_invalid")
            _assert_exact_regular_file(
                path, payload, code="public_artifact_already_exists"
            )
            _fsync_public_directory(path.parent)
            return
        try:
            with pending.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            _assert_exact_regular_file(
                pending, payload, code="public_artifact_recovery_state_invalid"
            )
        _assert_exact_regular_file(
            pending, payload, code="public_artifact_recovery_state_invalid"
        )
        _fsync_public_directory(path.parent)
        if _path_entry_exists(path, code="public_artifact_recovery_state_invalid"):
            _fail("public_artifact_recovery_state_invalid")
        os.replace(pending, path)
        _fsync_public_directory(path.parent)
        _assert_exact_regular_file(path, payload, code="public_artifact_unwritable")
        if _path_entry_exists(
            pending, code="public_artifact_recovery_state_invalid"
        ):
            _fail("public_artifact_recovery_state_invalid")
    except OSError:
        _fail("public_artifact_unwritable")


def build_comparison_update(existing: bytes, result: Mapping[str, Any]) -> bytes:
    try:
        text = existing.decode("utf-8", errors="strict")
    except UnicodeError:
        _fail("comparison_invalid")
    marker = "| SEC/Gemma lean evidence v3.8 |"
    lines = text.splitlines()
    outcome = result["status"]
    failed = result["failed_gate_count"]
    if outcome == "passed":
        development = "All 22 frozen development gates passed under exact replay"
        decision = (
            "Development passed; confirmation remains closed until the pushed-result gate passes"
        )
    elif outcome == "failed_gate":
        development = f"Frozen development replay failed {failed} of 22 gates"
        decision = "Rejected on development evidence; later data remains unopened"
    else:
        development = f"Attempt ended {outcome} with redacted code {result['terminal_code']}"
        decision = "Attempt rejected or indeterminate; later data remains unopened"
    row = (
        "| SEC/Gemma lean science v3.9 | Authenticated v3.8 source bridge, local "
        "Gemma filing semantics and the frozen causal long/cash evaluator | "
        f"{development} | No | {decision} |"
    )
    existing_v39 = [line for line in lines if line.startswith("| SEC/Gemma lean science v3.9 |")]
    if existing_v39:
        if existing_v39 != [row]:
            _fail("comparison_existing_result_invalid")
        return existing
    positions = [index for index, line in enumerate(lines) if line.startswith(marker)]
    if len(positions) != 1:
        _fail("comparison_marker_invalid")
    lines.insert(positions[0] + 1, row)
    return ("\n".join(lines) + "\n").encode("utf-8")


def publish_pause_artifact(repo_root: Path, value: Mapping[str, Any]) -> None:
    _write_new_canonical_json(repo_root / Path(contract.PAUSE_ARTIFACT_PATH), value)


def publish_result_artifacts(repo_root: Path, value: Mapping[str, Any]) -> None:
    result_path = repo_root / Path(contract.RESULT_ARTIFACT_PATH)
    comparison_path = repo_root / Path(contract.COMPARISON_PATH)
    try:
        comparison_base = _git_bytes(
            repo_root, "show", f"HEAD:{contract.COMPARISON_PATH}"
        )
        comparison_current = comparison_path.read_bytes()
    except OSError:
        _fail("comparison_unreadable")
    comparison_after = build_comparison_update(comparison_base, value)
    if comparison_current not in {comparison_base, comparison_after}:
        _fail("comparison_recovery_state_invalid")
    _write_new_canonical_json(result_path, value)
    pending = comparison_path.with_name(comparison_path.name + ".v39-pending")
    if comparison_current == comparison_after:
        if _path_entry_exists(pending, code="comparison_recovery_state_invalid"):
            try:
                _assert_exact_regular_file(
                    pending,
                    comparison_after,
                    code="comparison_recovery_state_invalid",
                )
                os.replace(pending, comparison_path)
            except V39RunnerError:
                raise
            except OSError:
                _fail("comparison_unwritable")
        _fsync_public_directory(comparison_path.parent)
        return
    try:
        try:
            with pending.open("xb") as handle:
                handle.write(comparison_after)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            _assert_exact_regular_file(
                pending,
                comparison_after,
                code="comparison_recovery_state_invalid",
            )
        _assert_exact_regular_file(
            pending,
            comparison_after,
            code="comparison_recovery_state_invalid",
        )
        _fsync_public_directory(comparison_path.parent)
        os.replace(pending, comparison_path)
        _fsync_public_directory(comparison_path.parent)
    except V39RunnerError:
        raise
    except OSError:
        _fail("comparison_unwritable")


def _git_text(root: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError):
        _fail("git_authentication_failed")
    if completed.returncode != 0:
        _fail("git_authentication_failed")
    try:
        return completed.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeError:
        _fail("git_authentication_failed")


def _git_bytes(root: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError):
        _fail("git_authentication_failed")
    if completed.returncode != 0:
        _fail("git_authentication_failed")
    return completed.stdout


def _git_changed_paths(root: Path, parent: str, child: str) -> dict[str, str]:
    text = _git_text(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        parent,
        child,
    )
    result: dict[str, str] = {}
    for line in text.splitlines():
        try:
            state, path = line.split("\t", 1)
        except ValueError:
            _fail("git_delta_invalid")
        if path in result:
            _fail("git_delta_invalid")
        result[path] = state
    return result


def _git_single_parent(root: Path, commit: str) -> str:
    row = _git_text(root, "rev-list", "--parents", "-n", "1", commit)
    fields = row.split()
    if (
        len(fields) != 2
        or fields[0] != commit
        or any(re.fullmatch(r"[0-9a-f]{40}", item) is None for item in fields)
    ):
        _fail("git_single_parent_invalid")
    return fields[1]


def _normalize_invocation_parent(
    value: Mapping[str, Any], *, authority: Mapping[str, Any]
) -> dict[str, str]:
    item = _mapping(value, "invocation_parent_invalid")
    if set(item) != {"invocation_parent", "invocation_parent_kind"}:
        _fail("invocation_parent_invalid")
    commit = item["invocation_parent"]
    kind = item["invocation_parent_kind"]
    if (
        type(commit) is not str
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or kind not in {"preflight", "pause", "continuation"}
        or kind == "preflight"
        and commit != authority["preflight"]["commit"]
    ):
        _fail("invocation_parent_invalid")
    return {"invocation_parent": commit, "invocation_parent_kind": kind}


def _assert_invocation_store_binding(
    snapshot: Any, *, invocation_parent: str, invocation_parent_kind: str
) -> None:
    """Reject an S/C private-state mismatch before any external effect."""

    if re.fullmatch(r"[0-9a-f]{40}", invocation_parent) is None:
        _fail("invocation_parent_invalid")
    layout = [item.generation_count for item in snapshot.segment_summaries]
    if invocation_parent_kind == "preflight":
        if snapshot.continuation_authorized or len(layout) > 1:
            _fail("preflight_invocation_store_mismatch")
        return
    if invocation_parent_kind == "pause":
        if (
            snapshot.status != "paused"
            or snapshot.paused is not True
            or snapshot.continuation_authorized
            or layout != [PILOT_COUNT]
        ):
            _fail("pause_invocation_store_mismatch")
        return
    if invocation_parent_kind == "continuation":
        if (
            snapshot.paused is not True
            or layout not in ([PILOT_COUNT], [PILOT_COUNT, REMAINING_CALL_COUNT])
            or layout == [PILOT_COUNT, REMAINING_CALL_COUNT]
            and not snapshot.continuation_authorized
        ):
            _fail("continuation_invocation_store_mismatch")
        if (
            snapshot.continuation_authorized
            and snapshot.continuation_commit != invocation_parent
        ):
            _fail("continuation_restart_binding_invalid")
        return
    _fail("invocation_parent_invalid")


def inspect_production_invocation_parent(
    repo_root: Path, authority: Mapping[str, Any]
) -> dict[str, str]:
    """Identify the already-authenticated F, S, or C invocation parent."""

    head = _git_text(repo_root, "rev-parse", "HEAD")
    if head == authority["preflight"]["commit"]:
        kind = "preflight"
    else:
        parent = _git_single_parent(repo_root, head)
        delta = _git_changed_paths(repo_root, parent, head)
        if delta == {contract.PAUSE_ARTIFACT_PATH: "A"}:
            kind = "pause"
        elif delta == {contract.CONTINUATION_PREREGISTRATION_PATH: "A"}:
            kind = "continuation"
        else:
            _fail("invocation_parent_invalid")
    return _normalize_invocation_parent(
        {"invocation_parent": head, "invocation_parent_kind": kind},
        authority=authority,
    )


def build_continuation_preregistration(
    pause_artifact: Mapping[str, Any]
) -> bytes:
    """Build the only allowed timing/count-only continuation document."""

    pause = _mapping(pause_artifact, "continuation_pause_invalid")
    durations = pause.get("pilot_durations_ns")
    effect = pause.get("effect_report")
    if (
        pause.get("schema_version") != PAUSE_SCHEMA_VERSION
        or pause.get("status") != "paused_for_justification"
        or not isinstance(durations, Sequence)
        or isinstance(durations, (str, bytes, bytearray))
        or len(durations) != PILOT_COUNT
        or any(type(item) is not int or item <= 0 for item in durations)
        or pause.get("pilot_count") != PILOT_COUNT
        or pause.get("formula")
        != "sum(pilot_duration_ns)+70*max(pilot_duration_ns)"
        or pause.get("projected_ns") != contract.projected_pilot_ns(durations)
        or pause.get("threshold_ns") != contract.PILOT_PROJECTED_THRESHOLD_NS
        or pause.get("strictly_greater_pause") is not True
        or pause.get("projected_ns") <= pause.get("threshold_ns")
        or not isinstance(effect, Mapping)
    ):
        _fail("continuation_pause_invalid")
    for field in (
        "pause_artifact_sha256",
        "pilot_guard_sha256",
        "latency_receipt_sha256",
        "remaining_order_sha256",
    ):
        _sha(pause.get(field), "continuation_pause_invalid")
    effect_json = contract.canonical_json_bytes(dict(effect)).decode("utf-8")
    duration_json = json.dumps(list(durations), separators=(",", ":"))
    remaining_ns = REMAINING_CALL_COUNT * max(durations)
    text = (
        "# AAPL SEC/Gemma lean science v3.9 continuation preregistration\n\n"
        f"Schema: `{CONTINUATION_DOCUMENT_SCHEMA_VERSION}`\n\n"
        "The authenticated five-generation pilot crossed the frozen twelve-hour "
        "projection threshold. This document freezes only the timing justification "
        "and the already-preregistered remaining local compute.\n\n"
        "## Authenticated pilot evidence\n\n"
        f"- Pilot generations: `{PILOT_COUNT}`\n"
        f"- Pilot durations in nanoseconds: `{duration_json}`\n"
        f"- Projection formula: `{pause['formula']}`\n"
        f"- Projected total nanoseconds: `{pause['projected_ns']}`\n"
        f"- Frozen threshold nanoseconds: `{pause['threshold_ns']}`\n"
        f"- Pilot guard SHA-256: `{pause['pilot_guard_sha256']}`\n"
        f"- Latency receipt SHA-256: `{pause['latency_receipt_sha256']}`\n"
        f"- Pause artifact SHA-256: `{pause['pause_artifact_sha256']}`\n"
        f"- Public effect counts: `{effect_json}`\n\n"
        "## Frozen remaining compute\n\n"
        f"- Physical model segments: `{PILOT_COUNT}` then `{REMAINING_CALL_COUNT}` generations\n"
        f"- Remaining order SHA-256: `{pause['remaining_order_sha256']}`\n"
        f"- Expected remaining generations: `{REMAINING_CALL_COUNT}`\n"
        "- Expected additional identity HTTP requests: `4`\n"
        f"- Projected remaining compute nanoseconds: `{remaining_ns}`\n\n"
        "## Unchanged safety rules\n\n"
        "- Development data only; confirmation and final data remain closed.\n"
        "- No SEC request, retry, repair, pull, fallback, paid call, broker effect, or real-money effect.\n"
        "- Long AAPL or cash only; no shorting, leverage, borrowing, or negative cash.\n"
        "- No science, model, prompt, request order, source, or gate change.\n"
        "- Fresh explicit user permission is still required after this document and the pause artifact are pushed and authenticated.\n"
    )
    return text.encode("ascii", errors="strict")


def authenticate_pushed_continuation(
    repo_root: Path,
    *,
    pause_artifact: Mapping[str, Any],
    forbidden_tokens: Sequence[bytes | str],
) -> dict[str, Any]:
    """Authenticate exact F -> S -> C before the remaining 70 calls."""

    head = _git_text(repo_root, "rev-parse", "HEAD")
    pause = _git_single_parent(repo_root, head)
    preflight = _git_single_parent(repo_root, pause)
    ancestry = {
        "branch": _git_text(repo_root, "branch", "--show-current"),
        "commit": head,
        "tree": _git_text(repo_root, "rev-parse", "HEAD^{tree}"),
        "parent": pause,
        "pause_commit": pause,
        "preflight_commit": preflight,
        "local_head": head,
        "remote_head": _git_text(
            repo_root, "rev-parse", f"refs/remotes/origin/{contract.BRANCH_NAME}"
        ),
        "changed_paths_from_pause": _git_changed_paths(repo_root, pause, head),
        "changed_paths_from_preflight": _git_changed_paths(
            repo_root, preflight, head
        ),
        "clean_worktree": _git_text(
            repo_root, "status", "--porcelain=v1", "--untracked-files=all"
        )
        == "",
        "earlier_blobs_equal_preflight": True,
    }
    try:
        contract.validate_continuation_ancestry(ancestry)
    except Exception:
        _fail("continuation_ancestry_invalid")
    path = repo_root / Path(contract.CONTINUATION_PREREGISTRATION_PATH)
    try:
        payload = path.read_bytes()
    except OSError:
        _fail("continuation_document_invalid")
    expected = build_continuation_preregistration(pause_artifact)
    committed_payload = _git_bytes(
        repo_root, "show", f"HEAD:{contract.CONTINUATION_PREREGISTRATION_PATH}"
    )
    committed_pause = _git_bytes(
        repo_root, "show", f"{pause}:{contract.PAUSE_ARTIFACT_PATH}"
    )
    if (
        payload != expected
        or committed_payload != expected
        or committed_pause != contract.canonical_json_bytes(pause_artifact)
    ):
        _fail("continuation_document_invalid")
    assert_public_artifact_private_free(
        {"continuation_document": expected.decode("ascii")},
        forbidden_tokens=forbidden_tokens,
    )
    return {
        "continuation_commit": head,
        "pause_commit": pause,
        "preflight_commit": preflight,
        "continuation_document_sha256": hashlib.sha256(committed_payload).hexdigest(),
    }


def build_production_development_dependencies() -> DevelopmentDependencies:
    """Bind the real read-only gates, store, and transports lazily."""

    try:
        from .sec_gemma_lean_science_v39_bridge import (
            authenticate_v38_private_root,
            build_streaming_science_projection,
        )
        from .sec_gemma_lean_science_v39_preflight import (
            V38_PRIVATE_ROOT,
            authenticate_execution_preflight,
            load_execution_authority,
            load_private_contact,
            load_publication_recovery_authority,
        )
        from .sec_gemma_lean_science_v39_store import AttemptStore
    except Exception:
        _fail("development_dependency_unavailable")

    recovery = {"active": False}

    def authenticate(root: Path, descendant: bool) -> Mapping[str, Any]:
        try:
            if not descendant:
                authenticate_execution_preflight(root)
            return load_execution_authority(root)
        except Exception:
            authority = load_publication_recovery_authority(root)
            recovery["active"] = True
            return authority

    def source(root: Path, contact: str) -> Any:
        return authenticate_v38_private_root(
            root,
            root / V38_PRIVATE_ROOT,
            readable_contact=contact,
        )

    def open_store(path: Path, authority: Mapping[str, Any]) -> Any:
        return (
            AttemptStore.open(path, authority=authority)
            if path.exists()
            else AttemptStore.create(path, authority=authority)
        )

    return DevelopmentDependencies(
        authenticate_execution=authenticate,
        load_private_contact=load_private_contact,
        authenticate_source=source,
        build_projection=build_streaming_science_projection,
        open_store=open_store,
        inspect_invocation_parent=inspect_production_invocation_parent,
        publication_recovery_active=lambda: recovery["active"],
        publish_pause=publish_pause_artifact,
        publish_result=publish_result_artifacts,
    )


def _recover_checkpoint(store: Any) -> None:
    """Recover only the exact checkpoint state derived by the sealed store."""

    try:
        state = store.derive_pending_checkpoint_state()
    except Exception as exc:
        code = getattr(exc, "code", "checkpoint_recovery_invalid")
        _fail(code if type(code) is str else "checkpoint_recovery_invalid")
    store.recover_pending_checkpoint(lambda _context: state)


def _pilot_evidence(store: Any, *, plan: ModelPlan) -> tuple[dict[str, Any], dict[str, Any]]:
    guard = rebuild_closed_segment_guard(
        store, store_segment_id="initial", guard_role="pilot"
    )
    receipts = store.committed_response_receipts(
        effect_kind="gemma", segment_id="initial"
    )
    durations = {
        item.intent.request_sha256: item.duration_ns for item in receipts
    }
    latency = build_latency_receipt(
        plan=plan, durations_ns_by_request_sha256=durations
    )
    if latency["pause_required"] is not True:
        _fail("pilot_pause_evidence_invalid")
    return guard, latency


def _load_and_validate_pause_artifact(
    repo_root: Path,
    *,
    expected_guard: Mapping[str, Any],
    expected_latency: Mapping[str, Any],
    plan: ModelPlan,
    store: Any,
    authority: Mapping[str, Any],
    dependencies: DevelopmentDependencies,
    forbidden_tokens: Sequence[bytes | str],
) -> dict[str, Any]:
    effect_report = build_pilot_pause_effect_report(store.snapshot)
    contract.validate_effect_counts(
        effect_report["completed_effect_counts"], route="pilot_pause"
    )
    expected = build_public_pause_artifact(
        authority=authority,
        plan=plan,
        pilot_guard=expected_guard,
        latency_receipt=expected_latency,
        effect_report=effect_report,
    )
    _assert_complete_v39_private_namespace_private_free(
        store,
        forbidden_tokens=forbidden_tokens,
    )
    assert_public_artifact_private_free(expected, forbidden_tokens=forbidden_tokens)
    path = repo_root / Path(contract.PAUSE_ARTIFACT_PATH)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except FileNotFoundError:
        publisher = dependencies.publish_pause or publish_pause_artifact
        publisher(repo_root, expected)
        return expected
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("pause_artifact_invalid")
    if (
        type(value) is not dict
        or contract.canonical_json_bytes(value) != raw
        or value != expected
    ):
        _fail("pause_artifact_invalid")
    return expected


def _open_stage_slice(store: Any, *, plan: ModelPlan) -> dict[str, Any]:
    responses = list(store.committed_responses(effect_kind="yahoo"))
    if len(responses) != contract.YAHOO_REQUEST_COUNT:
        _fail("yahoo_batch_unsealed")
    raw_by_symbol: dict[str, bytes] = {}
    for index, response in enumerate(responses):
        symbol = contract.YAHOO_SYMBOL_ORDER[index]
        if (
            response.intent.request_id != f"yahoo-{index + 1:02d}"
            or response.intent.request_sha256
            != yahoo_request_sha256(contract.YAHOO_URLS[index])
            or response.metadata.get("symbol") != symbol
            or response.body_bytes != len(response.body)
            or not 1 <= response.body_bytes <= contract.YAHOO_MAX_RESPONSE_BYTES
            or symbol in raw_by_symbol
        ):
            _fail("yahoo_batch_binding_invalid")
        raw_by_symbol[symbol] = response.body
    return build_stage_slice_from_yahoo(plan=plan, raw_by_symbol=raw_by_symbol)


def _open_generation_batch(
    store: Any, *, plan: ModelPlan
) -> dict[str, dict[str, Any]]:
    responses = list(store.committed_responses(effect_kind="gemma"))
    expected = [item.request["request_sha256"] for item in plan.execution_calls]
    if [item.intent.request_sha256 for item in responses] != expected:
        _fail("generation_batch_order_invalid")
    result: dict[str, dict[str, Any]] = {}
    for response in responses:
        request_sha = response.intent.request_sha256
        if (
            request_sha in result
            or response.duration_ns is None
            or response.body_bytes != len(response.body)
            or not 1
            <= response.body_bytes
            <= contract.MODEL_RESPONSE_MAX_BYTES
        ):
            _fail("generation_batch_binding_invalid")
        result[request_sha] = {
            "response_bytes": response.body,
            "duration_ns": response.duration_ns,
            "response_event_sha256": response.event_sha256,
        }
    return result


def _publish_terminal_from_store(
    repo_root: Path,
    *,
    store: Any,
    authority: Mapping[str, Any],
    invocation_parent: Mapping[str, str],
    dependencies: DevelopmentDependencies,
    forbidden_tokens: Sequence[bytes | str],
) -> dict[str, Any]:
    snapshot = store.snapshot
    if invocation_parent["invocation_parent_kind"] == "continuation":
        if (
            snapshot.continuation_authorized is not True
            or snapshot.continuation_commit
            != invocation_parent["invocation_parent"]
        ):
            _fail("terminal_continuation_binding_invalid")
    elif getattr(snapshot, "continuation_commit", None) is not None:
        _fail("terminal_continuation_binding_invalid")
    receipt = store.committed_terminal_evidence()
    if (
        receipt.evidence.get("invocation_parent")
        != invocation_parent["invocation_parent"]
        or receipt.evidence.get("invocation_parent_kind")
        != invocation_parent["invocation_parent_kind"]
    ):
        _fail("terminal_invocation_parent_mismatch")
    _assert_complete_v39_private_namespace_private_free(
        store,
        forbidden_tokens=forbidden_tokens,
    )
    public = build_public_terminal_artifact(
        authority=authority, terminal_receipt=receipt
    )
    assert_public_artifact_private_free(public, forbidden_tokens=forbidden_tokens)
    publisher = dependencies.publish_result or publish_result_artifacts
    publisher(repo_root, public)
    return public


def _terminalize_failure(
    repo_root: Path,
    *,
    store: Any,
    authority: Mapping[str, Any],
    dependencies: DevelopmentDependencies,
    forbidden_tokens: Sequence[bytes | str],
    code: str,
    invocation_parent: Mapping[str, Any],
    market_values_opened: bool,
    model_responses_opened: bool,
    source_authenticated: bool,
) -> dict[str, Any]:
    snapshot = store.snapshot
    indeterminate = bool(
        snapshot.active_request_id
        or snapshot.open_model_segment_id
        or snapshot.pending_checkpoint_event_sha256
        and not snapshot.recoverable_checkpoint
        or snapshot.status == "indeterminate"
    )
    status = "indeterminate" if indeterminate else "rejected"
    material = build_failure_terminal_material(
        authority=authority,
        terminal_code=code,
        terminal_status=status,
        effect_report=build_effect_report(snapshot),
        snapshot=snapshot,
        invocation_parent=invocation_parent["invocation_parent"],
        invocation_parent_kind=invocation_parent["invocation_parent_kind"],
        market_values_opened=market_values_opened,
        model_responses_opened=model_responses_opened,
        source_authenticated=source_authenticated,
    )
    store.seal_terminal(status=status, terminal_code=code, evidence=material)
    return _publish_terminal_from_store(
        repo_root,
        store=store,
        authority=authority,
        invocation_parent=invocation_parent,
        dependencies=dependencies,
        forbidden_tokens=forbidden_tokens,
    )


def status(repo_root: Path) -> dict[str, Any]:
    """Read only the safe v3.9 attempt summary; never open private payload bodies."""

    try:
        from .sec_gemma_lean_science_v39_store import audit_attempt
        from .sec_gemma_lean_science_v39_preflight import (
            load_execution_authority,
            load_publication_recovery_authority,
        )

        root = repo_root.resolve(strict=True)
        try:
            authority = load_execution_authority(root)
        except Exception:
            authority = load_publication_recovery_authority(root)
        attempt_root = root / contract.PRIVATE_DEVELOPMENT_NAMESPACE
        if not _path_entry_exists(attempt_root, code="status_attempt_invalid"):
            preflight = _mapping(
                authority.get("preflight"), "status_authority_invalid"
            )
            if (
                _git_text(root, "rev-parse", "HEAD")
                != preflight.get("commit")
            ):
                _fail("status_attempt_missing")
            return {"schema_version": RUNNER_SCHEMA_VERSION, "status": "not_started"}
        snapshot = audit_attempt(
            attempt_root,
            authority=authority,
        )
    except V39RunnerError:
        raise
    except Exception:
        _fail("status_replay_failed")
    return {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": snapshot.status,
        "terminal_code": snapshot.terminal_code,
        "event_count": snapshot.event_count,
        "yahoo_response_count": snapshot.yahoo_response_count,
        "identity_response_count": snapshot.identity_response_count,
        "gemma_response_count": snapshot.gemma_response_count,
        "paused": snapshot.paused,
        "continuation_authorized": snapshot.continuation_authorized,
    }


def run_development(
    repo_root: Path,
    *,
    dependencies: DevelopmentDependencies | None = None,
    continuation_permission_sha256: str | None = None,
) -> dict[str, Any]:
    """Run or safely resume the one-shot development attempt."""

    try:
        root = repo_root.resolve(strict=True)
    except (OSError, RuntimeError):
        _fail("repository_root_invalid")
    deps = dependencies or build_production_development_dependencies()
    if type(deps) is not DevelopmentDependencies:
        _fail("development_dependencies_invalid")
    if continuation_permission_sha256 is not None:
        _sha(continuation_permission_sha256, "continuation_permission_invalid")
    descendant = (
        (root / Path(contract.PAUSE_ARTIFACT_PATH)).exists()
        or (
            root / Path(contract.PAUSE_ARTIFACT_PATH + ".v39-pending")
        ).exists()
        or (root / Path(contract.RESULT_ARTIFACT_PATH)).exists()
        or (
            root / Path(contract.RESULT_ARTIFACT_PATH + ".v39-pending")
        ).exists()
        or (root / Path(contract.CONTINUATION_PREREGISTRATION_PATH)).exists()
    )
    store: Any | None = None
    authority: dict[str, Any] | None = None
    invocation_parent: dict[str, str] | None = None
    contact: str | None = None
    projection: Any | None = None
    plan: ModelPlan | None = None
    authenticated_continuation: dict[str, Any] | None = None
    authenticated_continuation_pause: dict[str, Any] | None = None
    try:
        raw_authority = deps.authenticate_execution(root, descendant)
        try:
            authority = json.loads(
                contract.canonical_json_bytes(raw_authority).decode("utf-8")
            )
        except Exception:
            _fail("execution_authority_invalid")
        parent_inspector = deps.inspect_invocation_parent
        invocation_parent = _normalize_invocation_parent(
            parent_inspector(root, authority)
            if parent_inspector is not None
            else {
                "invocation_parent": authority["preflight"]["commit"],
                "invocation_parent_kind": "preflight",
            },
            authority=authority,
        )
        contact = deps.load_private_contact(root)
        if type(contact) is not str or not contact:
            _fail("private_contact_invalid")
        # The frozen order authenticates and rebuilds the complete source
        # authority before the one-shot attempt directory can be created.
        source = deps.authenticate_source(root, contact)
        projection = deps.build_projection(source)
        commitments = build_preflight_request_commitments(projection)
        plan = build_model_plan(projection)
        authority = build_attempt_authority(
            execution_authority=authority,
            projection=projection,
            commitments=commitments,
            plan=plan,
        )
    except V39RunnerError:
        raise
    except Exception as exc:
        code = getattr(exc, "code", "development_authority_failed")
        _fail(code if type(code) is str else "development_authority_failed")

    v38_private = str(
        (root / "data/aapl_sec_gemma_lean_evidence_v3_8").resolve(strict=False)
    )
    v39_private = str((root / Path(contract.PRIVATE_NAMESPACE)).resolve(strict=False))
    forbidden_tokens: tuple[bytes | str, ...] = (
        contact,
        str(root),
        str(root).replace("\\", "/"),
        v38_private,
        v38_private.replace("\\", "/"),
        v39_private,
        v39_private.replace("\\", "/"),
    )
    attempt_root = root / Path(contract.PRIVATE_DEVELOPMENT_NAMESPACE)

    def terminalize_current_failure(code: str) -> dict[str, Any] | None:
        if store is None or authority is None or invocation_parent is None:
            return None
        snapshot = store.snapshot
        try:
            store.committed_terminal_evidence()
        except Exception:
            terminal_sealed = False
        else:
            terminal_sealed = True
        if terminal_sealed or snapshot.status in {"completed", "rejected"}:
            return None
        if snapshot.status == "checkpoint_recovery":
            if plan is None or not snapshot.recoverable_checkpoint:
                return None
            _recover_checkpoint(store)
            snapshot = store.snapshot
        if (
            snapshot.paused
            and not snapshot.continuation_authorized
            or invocation_parent["invocation_parent_kind"] == "pause"
        ):
            return None
        return _terminalize_failure(
            root,
            store=store,
            authority=authority,
            dependencies=deps,
            forbidden_tokens=forbidden_tokens,
            code=code,
            invocation_parent=invocation_parent,
            market_values_opened=snapshot.market_values_opened,
            model_responses_opened=snapshot.model_responses_opened,
            source_authenticated=True,
        )

    try:
        if (
            invocation_parent["invocation_parent_kind"] != "preflight"
            and not _path_entry_exists(
                attempt_root, code="invocation_parent_store_invalid"
            )
        ):
            _fail("invocation_parent_store_missing")
        candidate_store = deps.open_store(attempt_root, authority)
        try:
            snapshot = candidate_store.snapshot
            _assert_invocation_store_binding(
                snapshot,
                invocation_parent=invocation_parent["invocation_parent"],
                invocation_parent_kind=invocation_parent[
                    "invocation_parent_kind"
                ],
            )
        except Exception:
            try:
                candidate_store.close()
            except Exception:
                pass
            raise
        store = candidate_store
        if (
            deps.publication_recovery_active is not None
            and deps.publication_recovery_active()
        ):
            try:
                store.committed_terminal_evidence()
            except Exception:
                if (
                    snapshot.status != "paused"
                    or snapshot.continuation_authorized
                ):
                    _fail("publication_recovery_state_invalid")
                pilot_guard, latency = _pilot_evidence(store, plan=plan)
                return _load_and_validate_pause_artifact(
                    root,
                    expected_guard=pilot_guard,
                    expected_latency=latency,
                    plan=plan,
                    store=store,
                    authority=authority,
                    dependencies=deps,
                    forbidden_tokens=forbidden_tokens,
                )
            return _publish_terminal_from_store(
                root,
                store=store,
                authority=authority,
                invocation_parent=invocation_parent,
                dependencies=deps,
                forbidden_tokens=forbidden_tokens,
            )
        if invocation_parent["invocation_parent_kind"] == "continuation":
            pilot_guard, pilot_latency = _pilot_evidence(store, plan=plan)
            authenticated_continuation_pause = _load_and_validate_pause_artifact(
                root,
                expected_guard=pilot_guard,
                expected_latency=pilot_latency,
                plan=plan,
                store=store,
                authority=authority,
                dependencies=deps,
                forbidden_tokens=forbidden_tokens,
            )
            authenticated_continuation = authenticate_pushed_continuation(
                root,
                pause_artifact=authenticated_continuation_pause,
                forbidden_tokens=forbidden_tokens,
            )
            if (
                authenticated_continuation["preflight_commit"]
                != authority["preflight"]["commit"]
                or authenticated_continuation["continuation_commit"]
                != invocation_parent["invocation_parent"]
            ):
                _fail("continuation_preflight_mismatch")
            if store.snapshot.continuation_authorized:
                if (
                    store.snapshot.continuation_sha256
                    != authenticated_continuation["continuation_document_sha256"]
                    or store.snapshot.continuation_commit
                    != authenticated_continuation["continuation_commit"]
                    or continuation_permission_sha256 is not None
                    and store.snapshot.continuation_permission_sha256
                    != continuation_permission_sha256
                ):
                    _fail("continuation_restart_binding_invalid")
        if snapshot.status in {"completed", "rejected"}:
            return _publish_terminal_from_store(
                root,
                store=store,
                authority=authority,
                invocation_parent=invocation_parent,
                dependencies=deps,
                forbidden_tokens=forbidden_tokens,
            )
        if snapshot.status == "indeterminate":
            try:
                store.committed_terminal_evidence()
            except Exception:
                pass
            else:
                return _publish_terminal_from_store(
                    root,
                    store=store,
                    authority=authority,
                    invocation_parent=invocation_parent,
                    dependencies=deps,
                    forbidden_tokens=forbidden_tokens,
                )

        if True:
            snapshot = store.snapshot
            if snapshot.status in {"completed", "rejected"}:
                return _publish_terminal_from_store(
                    root,
                    store=store,
                    authority=authority,
                    invocation_parent=invocation_parent,
                    dependencies=deps,
                    forbidden_tokens=forbidden_tokens,
                )
            if snapshot.status == "indeterminate":
                try:
                    store.committed_terminal_evidence()
                except Exception:
                    if invocation_parent["invocation_parent_kind"] == "pause":
                        _fail("pause_invocation_terminal_forbidden")
                    return _terminalize_failure(
                        root,
                        store=store,
                        authority=authority,
                        dependencies=deps,
                        forbidden_tokens=forbidden_tokens,
                        code=snapshot.terminal_code or "attempt_indeterminate",
                        invocation_parent=invocation_parent,
                        market_values_opened=snapshot.market_values_opened,
                        model_responses_opened=snapshot.model_responses_opened,
                        source_authenticated=True,
                    )
                return _publish_terminal_from_store(
                    root,
                    store=store,
                    authority=authority,
                    invocation_parent=invocation_parent,
                    dependencies=deps,
                    forbidden_tokens=forbidden_tokens,
                )
            if snapshot.status == "checkpoint_recovery":
                _recover_checkpoint(store)
                snapshot = store.snapshot
            if snapshot.yahoo_response_count < contract.YAHOO_REQUEST_COUNT:
                execute_yahoo_effects(store, fetcher=deps.yahoo_fetch)
            if store.snapshot.yahoo_response_count != contract.YAHOO_REQUEST_COUNT:
                _fail("yahoo_batch_incomplete")

            summaries = list(store.snapshot.segment_summaries)
            if not summaries:
                execute_model_segment(
                    store,
                    calls=plan.execution_calls,
                    segment_id="initial",
                    probe_request=deps.runtime_probe,
                    generation_request=deps.generation,
                )
                summaries = list(store.snapshot.segment_summaries)

            if [item.generation_count for item in summaries] == [PILOT_COUNT]:
                pilot_guard, latency = _pilot_evidence(store, plan=plan)
                pause_committed_now = False
                if not store.snapshot.paused and not store.snapshot.continuation_authorized:
                    store.commit_planned_pause()
                    pause_committed_now = True
                parent_kind = invocation_parent["invocation_parent_kind"]
                pause = (
                    authenticated_continuation_pause
                    if parent_kind == "continuation"
                    and authenticated_continuation_pause is not None
                    else _load_and_validate_pause_artifact(
                        root,
                        expected_guard=pilot_guard,
                        expected_latency=latency,
                        plan=plan,
                        store=store,
                        authority=authority,
                        dependencies=deps,
                        forbidden_tokens=forbidden_tokens,
                    )
                )
                if pause_committed_now or parent_kind in {"preflight", "pause"}:
                    # This invocation began before an authenticated C existed.
                    # A supplied stale token can never authorize call six.
                    return pause
                if parent_kind != "continuation":
                    _fail("continuation_parent_invalid")
                if (
                    not store.snapshot.continuation_authorized
                    and continuation_permission_sha256 is None
                ):
                    return pause
                if authenticated_continuation is None:
                    _fail("continuation_parent_invalid")
                if store.snapshot.continuation_authorized:
                    if (
                        store.snapshot.continuation_sha256
                        != authenticated_continuation[
                            "continuation_document_sha256"
                        ]
                        or store.snapshot.continuation_commit
                        != authenticated_continuation["continuation_commit"]
                    ):
                        _fail("continuation_restart_binding_invalid")
                else:
                    store.authorize_continuation(
                        continuation_sha256=authenticated_continuation[
                            "continuation_document_sha256"
                        ],
                        permission_sha256=continuation_permission_sha256,
                        continuation_commit=authenticated_continuation[
                            "continuation_commit"
                        ],
                    )
                    if (
                        store.snapshot.continuation_sha256
                        != authenticated_continuation[
                            "continuation_document_sha256"
                        ]
                        or store.snapshot.continuation_commit
                        != authenticated_continuation["continuation_commit"]
                    ):
                        _fail("continuation_restart_binding_invalid")
                execute_model_segment(
                    store,
                    calls=plan.execution_calls[PILOT_COUNT:],
                    segment_id="continuation",
                    probe_request=deps.runtime_probe,
                    generation_request=deps.generation,
                )

            summaries = list(store.snapshot.segment_summaries)
            layout = [item.generation_count for item in summaries]
            if layout not in ([TOTAL_CALL_COUNT], [PILOT_COUNT, REMAINING_CALL_COUNT]):
                _fail("completed_segment_layout_invalid")
            expected_parent_kind = (
                "preflight" if layout == [TOTAL_CALL_COUNT] else "continuation"
            )
            if invocation_parent["invocation_parent_kind"] != expected_parent_kind:
                _fail("completed_invocation_parent_invalid")
            guards, aggregate, latency, guard_by_request = rebuild_runtime_evidence(
                store, plan=plan
            )
            effect_report = build_effect_report(store.snapshot)
            route = "normal_complete" if layout == [TOTAL_CALL_COUNT] else "paused_resumed_complete"
            contract.validate_effect_counts(
                effect_report["completed_effect_counts"], route=route
            )
            attempted = effect_report["attempted_external_requests"]
            completed = effect_report["completed_effect_counts"]
            if (
                attempted["yahoo_requests"] != completed["yahoo_requests"]
                or attempted["ollama_identity_http_requests"]
                != completed["ollama_identity_http_requests"]
                or attempted["ollama_chat_generations"]
                != completed["ollama_chat_generations"]
            ):
                _fail("successful_effect_intent_count_invalid")

            # Only now are market values and model bodies opened.
            if not store.snapshot.market_values_opened:
                store.mark_market_values_opened()
            stage_slice = _open_stage_slice(store, plan=plan)
            if not store.snapshot.model_responses_opened:
                store.mark_model_responses_opened()
            sealed = _open_generation_batch(store, plan=plan)
            semantic = build_semantic_payload(
                plan=plan,
                sealed_calls_by_request_sha256=sealed,
                segment_guard_by_request_sha256=guard_by_request,
                runtime_aggregate=aggregate,
                latency_receipt=latency,
            )
            deterministic = build_deterministic_payload(
                stage_slice=stage_slice, semantic_payload=semantic
            )
            evaluated = evaluate_deterministic_science(
                semantic_payload=semantic,
                deterministic_payload=deterministic,
            )
            evaluation = _mapping(
                evaluated.get("evaluation"), "deterministic_evaluation_invalid"
            )
            material = build_private_terminal_material(
                authority=authority,
                projection=projection,
                plan=plan,
                runtime_guards=guards,
                runtime_aggregate=aggregate,
                latency_receipt=latency,
                stage_slice=stage_slice,
                semantic_payload=semantic,
                deterministic_payload=deterministic,
                evaluation=evaluation,
                effect_report=effect_report,
                invocation_parent=invocation_parent["invocation_parent"],
                invocation_parent_kind=invocation_parent["invocation_parent_kind"],
            )
            gate_passed = evaluation["gate_report"]["passed"]
            terminal_code = (
                "development_pass" if gate_passed else "development_failed_gate"
            )
            store.seal_terminal(
                status="completed",
                terminal_code=terminal_code,
                evidence=material,
            )
            return _publish_terminal_from_store(
                root,
                store=store,
                authority=authority,
                invocation_parent=invocation_parent,
                dependencies=deps,
                forbidden_tokens=forbidden_tokens,
            )
    except V39RunnerError as exc:
        try:
            terminal = terminalize_current_failure(exc.code)
            if terminal is not None:
                return terminal
        except Exception:
            pass
        raise
    except Exception as exc:
        code = getattr(exc, "code", "development_worker_failed")
        safe_code = code if type(code) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code) else "development_worker_failed"
        try:
            terminal = terminalize_current_failure(safe_code)
            if terminal is not None:
                return terminal
        except V39RunnerError:
            raise
        except Exception:
            pass
        _fail(safe_code)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
    _fail("development_worker_failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AAPL SEC/Gemma v3.9 science runner")
    parser.add_argument("command", choices=("development", "status"))
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--continuation-permission-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        root = Path(arguments.repo_root)
        value = (
            status(root)
            if arguments.command == "status"
            else run_development(
                root,
                continuation_permission_sha256=(
                    arguments.continuation_permission_sha256
                ),
            )
        )
        print(contract.canonical_json_bytes(value).decode("utf-8"))
        return 0
    except V39RunnerError as exc:
        print(contract.canonical_json_bytes({"status": "rejected", "code": exc.code}).decode("utf-8"))
        return 2


__all__ = [
    "CONTINUATION_DOCUMENT_SCHEMA_VERSION",
    "LATENCY_RECEIPT_SCHEMA_VERSION",
    "ModelPlan",
    "PreparedModelCall",
    "RUNNER_SCHEMA_VERSION",
    "V39RunnerError",
    "assert_private_namespace_private_free",
    "assert_public_artifact_private_free",
    "authenticate_pushed_continuation",
    "build_continuation_preregistration",
    "build_deterministic_payload",
    "build_effect_report",
    "build_latency_receipt",
    "build_model_plan",
    "build_preflight_request_commitments",
    "build_private_terminal_material",
    "build_public_effect_report",
    "build_public_terminal_artifact",
    "build_runtime_aggregate_guard",
    "build_runtime_segment_guard",
    "build_semantic_payload",
    "evaluate_deterministic_science",
    "inspect_production_invocation_parent",
    "main",
    "parse_sealed_generation_response",
    "run_development",
    "status",
]


if __name__ == "__main__":
    raise SystemExit(main())
