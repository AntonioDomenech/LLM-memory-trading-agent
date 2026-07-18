from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_benchmark import sec_gemma_lean_science_v39_contract as contract
from agent_benchmark import sec_gemma_lean_science_v39_runner as runner


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _assert_code(code: str, function: Any, /, *args: Any, **kwargs: Any) -> None:
    with pytest.raises(runner.V39RunnerError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code


def _with_hash(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: contract.canonical_sha256(body)}


def _projection() -> SimpleNamespace:
    documents = [{"accession_number": f"A{i:03}"} for i in range(1, 76)]
    records = [{"accession_number": f"A{i:03}"} for i in range(1, 76)]
    events = [
        {
            "accession_number": f"A{i:03}",
            "availability_session": f"2018-01-{i:03}",
        }
        for i in range(1, 76)
    ]
    source_order = [dict(item) for item in events]
    prior_links = [
        {"accession_number": f"A{i:03}", "prior_accession_number": None}
        for i in range(1, 76)
    ]
    manifest = {
        "documents_sha256": _digest("documents"),
        "records_sha256": _digest("records"),
        "event_order_sha256": _digest("events"),
        "source_order_sha256": _digest("source-order"),
        "prior_links_sha256": _digest("prior-links"),
    }
    return SimpleNamespace(
        stage="development",
        primary_documents=documents,
        records=records,
        event_order=events,
        source_order=source_order,
        prior_links=prior_links,
        manifest=manifest,
    )


def _synthetic_requests_and_proofs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for i in range(1, 76):
        accession = f"A{i:03}"
        size = 100 + i
        if i in {73, 74, 75}:
            size = 1_000
        elif i == 72:
            size = 999
        elif i == 71:
            size = 998
        request_bytes = accession.encode("ascii") + b"x" * (size - len(accession))
        request = {
            "accession_number": accession,
            "form": "10-K" if i % 2 else "10-Q",
            "availability_session": f"2018-01-{i:03}",
            "preprocessed_event_sha256": _digest(f"preprocessed:{accession}"),
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "request_bytes": request_bytes,
        }
        proof = {
            "current_record": {
                "accession_number": accession,
                "acceptance_datetime": f"2018-01-{i:03}T10:00:00Z",
            },
            "universe_event_proof_sha256": _digest(f"proof:{accession}"),
        }
        requests.append(request)
        proofs.append(proof)
    return requests, proofs


def _install_model_plan_fakes(
    monkeypatch: pytest.MonkeyPatch, *, reverse: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from agent_benchmark import sec_gemma_online_risk_overlay_acquisition as acquisition
    from agent_benchmark import sec_gemma_online_risk_overlay_production as production

    requests, proofs = _synthetic_requests_and_proofs()
    if reverse:
        requests.reverse()
        proofs.reverse()

    monkeypatch.setattr(
        acquisition,
        "_build_model_requests",
        lambda _documents, *, carry_in_documents, stage: copy.deepcopy(requests),
    )
    monkeypatch.setattr(
        acquisition,
        "_build_universe_event_proofs",
        lambda *, stage, universe, current_documents, predecessor_bundles: copy.deepcopy(
            proofs
        ),
    )
    monkeypatch.setattr(
        acquisition,
        "build_acquisition_plan",
        lambda stage: {"stage": stage, "attempt_id": "old-attempt"},
    )
    monkeypatch.setattr(
        acquisition,
        "_build_model_slice",
        lambda *, plan, model_requests, universe_event_proofs: {
            "model_slice_sha256": _digest("model-slice")
        },
    )
    monkeypatch.setattr(production, "_validate_universe_proof", lambda value: dict(value))
    monkeypatch.setattr(
        production,
        "_validate_blinded_model_request",
        lambda request, proof: (
            request["request_bytes"],
            (f"S-{request['accession_number']}",),
            {"synthetic": True},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_legacy_universe",
        lambda records: {"universe_sha256": _digest("universe")},
    )
    return requests, proofs


def _model_plan() -> runner.ModelPlan:
    calls: list[runner.PreparedModelCall] = []
    for i in range(1, 76):
        accession = f"A{i:03}"
        request_bytes = f"request:{accession}".encode("ascii")
        request = {
            "accession_number": accession,
            "form": "10-K" if i % 2 else "10-Q",
            "availability_session": f"2018-01-{i:03}",
            "preprocessed_event_sha256": _digest(f"preprocessed:{accession}"),
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "request_bytes": request_bytes,
        }
        proof = {
            "universe_event_proof_sha256": _digest(f"proof:{accession}"),
        }
        calls.append(
            runner.PreparedModelCall(
                canonical_ordinal=i,
                execution_ordinal=i,
                accession_number=accession,
                acceptance_datetime=f"2018-01-{i:03}T10:00:00Z",
                request=request,
                proof=proof,
                request_bytes=request_bytes,
                sentence_ids=(f"S-{accession}",),
                request_byte_count=len(request_bytes),
                pilot=i <= 5,
            )
        )
    manifest_body = {
        "schema_version": runner.MODEL_PLAN_SCHEMA_VERSION,
        "document_count": 75,
    }
    return runner.ModelPlan(
        model_slice={"model_slice_sha256": _digest("model-slice")},
        universe={"universe_sha256": _digest("universe")},
        canonical_calls=tuple(calls),
        execution_calls=tuple(calls),
        manifest=_with_hash(manifest_body, "model_plan_sha256"),
    )


def _runtime_receipt(seed: str, *, raw_show_seed: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "ollama-runtime-probe-v2-2",
        "model_name": "gemma4:12b",
        "manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "config_sha256": contract.MODEL_CONFIG_DIGEST,
        "ordered_layer_sha256s": list(contract.MODEL_LAYER_DIGESTS),
        "ordered_layer_content_sha256s": list(contract.MODEL_LAYER_DIGESTS),
        "version_response_sha256": contract.RUNTIME_VERSION_RESPONSE_SHA256,
        "show_response_semantic_sha256": contract.RUNTIME_SHOW_SEMANTIC_SHA256,
        "show_semantic_excluded_keys": ["modified_at"],
        "model_info_sha256": contract.RUNTIME_MODEL_INFO_SHA256,
        "active_from_blob_sha256s": [_digest("from:1"), _digest("from:2")],
        "runtime_fingerprint_material": {"model": "gemma4:12b", "version": "v2.2"},
        "runtime_fingerprint_sha256": contract.RUNTIME_FINGERPRINT_SHA256,
        "manifest_config_layers_replayed": True,
        "all_layer_contents_hashed": True,
        "exact_two_active_from_blobs_verified": True,
        "runtime_receipt_sha256": _digest(f"receipt:{seed}"),
        "raw_show_response_sha256": _digest(raw_show_seed or f"raw:{seed}"),
        "modified_at": f"diagnostic-{seed}",
    }


def _guard(
    segment_id: str, events: list[str], *, seed: str = "segment"
) -> dict[str, Any]:
    return runner.build_runtime_segment_guard(
        segment_id=segment_id,
        pre_runtime_receipt=_runtime_receipt(f"{seed}:pre"),
        post_runtime_receipt=_runtime_receipt(f"{seed}:post"),
        ordered_generation_response_event_sha256s=events,
        generation_count=len(events),
    )


def _neutral_semantic_rows() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    row_hashes: list[str] = []
    for i in range(1, 76):
        accession = f"A{i:03}"
        body = {
            "schema_version": runner.SEMANTIC_ROW_SCHEMA_VERSION,
            "stage": "development",
            "ordinal": i,
            "accession_number": accession,
            "form": "10-K" if i % 2 else "10-Q",
            "decision_session": f"2018-01-{i:03}",
            "preprocessed_event_sha256": _digest(f"preprocessed:{accession}"),
            "request_sha256": _digest(f"request:{accession}"),
            "model_slice_sha256": _digest("model-slice"),
            "universe_event_proof_sha256": _digest(f"proof:{accession}"),
            "extraction_status": "invalid",
            "extraction_authenticated": True,
            "document_quality": None,
            "validated_output": None,
            "semantic_event_receipt": {"synthetic": True},
            "semantic_event_receipt_sha256": _digest(f"event-receipt:{accession}"),
            "latency_preflight_receipt": {"synthetic": True},
            "latency_preflight_receipt_sha256": _digest("latency"),
            "runtime_segment_guard_sha256": _digest("runtime-guard"),
            "runtime_aggregate_sha256": _digest("runtime-aggregate"),
        }
        row = _with_hash(body, "semantic_extraction_row_sha256")
        rows.append(row)
        row_hashes.append(row["semantic_extraction_row_sha256"])
    return {
        "semantic_extraction_rows": rows,
        "semantic_extraction_row_sha256s": row_hashes,
        "semantic_batch_receipt_sha256": _digest("semantic-batch"),
    }


class _RecordingEffectStore:
    """In-memory effect boundary that records intent/callback/commit order."""

    def __init__(
        self,
        *,
        yahoo_response_count: int = 0,
    ) -> None:
        self.yahoo_response_count = yahoo_response_count
        self.timeline: list[tuple[str, str]] = []
        self.identity_reads: list[tuple[Any, ...]] = []
        self.committed: list[Any] = []
        self.generation_durations: list[int] = []
        self.active_intent: Any | None = None
        self._intent_by_sha: dict[str, Any] = {}

    @property
    def snapshot(self) -> SimpleNamespace:
        pause_required = (
            len(self.generation_durations) >= 5
            and contract.pilot_pause_required(self.generation_durations[:5])
        )
        return SimpleNamespace(
            status="active",
            yahoo_response_count=self.yahoo_response_count,
            yahoo_body_bytes=sum(
                len(item.body)
                for item in self.committed
                if item.intent.effect_kind == "yahoo"
            ),
            pause_required=pause_required,
        )

    def begin_request(
        self,
        *,
        request_id: str,
        effect_kind: str,
        request_sha256: str,
        segment_id: str | None,
        model_phase: str,
    ) -> Any:
        assert self.active_intent is None
        intent = SimpleNamespace(
            event_sha256=_digest(f"intent:{request_id}"),
            request_id=request_id,
            effect_kind=effect_kind,
            request_sha256=request_sha256,
            segment_id=segment_id,
            model_phase=model_phase,
        )
        self.active_intent = intent
        self._intent_by_sha[intent.event_sha256] = intent
        self.timeline.append(("begin", request_id))
        return intent

    def callback_started(self) -> str:
        assert self.active_intent is not None
        request_id = self.active_intent.request_id
        self.timeline.append(("callback", request_id))
        return request_id

    def commit_response_and_checkpoint(
        self,
        *,
        request_intent_event_sha256: str,
        body: bytes,
        metadata: dict[str, Any],
        duration_ns: int | None,
        checkpoint_state: dict[str, Any],
    ) -> tuple[Any, Any]:
        intent = self._intent_by_sha[request_intent_event_sha256]
        assert intent is self.active_intent
        assert checkpoint_state["request_sha256"] == intent.request_sha256
        assert checkpoint_state["body_sha256"] == hashlib.sha256(body).hexdigest()
        # Force a distinct immutable bytes object.  Runtime verification must
        # use this sealed copy returned by committed_responses, not callback
        # locals retained by the runner.
        sealed_body = bytes(bytearray(body))
        response = SimpleNamespace(
            event_sha256=_digest(f"response:{intent.request_id}"),
            body=sealed_body,
            metadata=copy.deepcopy(metadata),
            intent=intent,
        )
        self.committed.append(response)
        if intent.effect_kind == "gemma":
            assert type(duration_ns) is int and duration_ns > 0
            self.generation_durations.append(duration_ns)
        else:
            assert duration_ns is None
        if intent.effect_kind == "yahoo":
            self.yahoo_response_count += 1
        self.timeline.append(("commit", intent.request_id))
        self.active_intent = None
        return response, SimpleNamespace(event_sha256=_digest(f"checkpoint:{intent.request_id}"))

    def committed_responses(
        self, *, effect_kind: str, segment_id: str | None = None
    ) -> tuple[Any, ...]:
        selected = tuple(
            item
            for item in self.committed
            if item.intent.effect_kind == effect_kind
            and (segment_id is None or item.intent.segment_id == segment_id)
        )
        self.identity_reads.append(selected)
        return selected


def _assert_effect_triples(store: _RecordingEffectStore) -> None:
    assert len(store.timeline) % 3 == 0
    for offset in range(0, len(store.timeline), 3):
        begin, callback, commit = store.timeline[offset : offset + 3]
        assert begin[0] == "begin"
        assert callback == ("callback", begin[1])
        assert commit == ("commit", begin[1])


def test_model_plan_fixes_canonical_pilot_and_remaining_commitments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests, _proofs = _install_model_plan_fakes(monkeypatch)
    projection = _projection()

    plan = runner.build_model_plan(projection)

    canonical = [f"A{i:03}" for i in range(1, 76)]
    pilots = ["A073", "A074", "A075", "A072", "A071"]
    remaining = [item for item in canonical if item not in set(pilots)]
    assert [item.accession_number for item in plan.canonical_calls] == canonical
    assert [item.accession_number for item in plan.execution_calls[:5]] == pilots
    assert [item.accession_number for item in plan.execution_calls[5:]] == remaining
    assert [item.pilot for item in plan.execution_calls[:5]] == [True] * 5
    assert all(not item.pilot for item in plan.execution_calls[5:])
    assert plan.manifest["model_plan_sha256"] == contract.canonical_sha256(
        {key: value for key, value in plan.manifest.items() if key != "model_plan_sha256"}
    )

    commitments = runner.build_preflight_request_commitments(projection)
    execution = [
        {
            "request_sha256": item.request["request_sha256"],
            "request_byte_count": item.request_byte_count,
        }
        for item in plan.execution_calls
    ]
    assert commitments["pilot_order_sha256"] == contract.canonical_sha256(execution[:5])
    assert commitments["remaining_order_sha256"] == contract.canonical_sha256(execution[5:])
    assert commitments["canonical_requests_sha256"] == contract.canonical_sha256(
        [
            {
                "request_sha256": item["request_sha256"],
                "request_byte_count": len(item["request_bytes"]),
            }
            for item in requests
        ]
    )
    assert commitments["confirmation_or_final_opened"] is False
    assert commitments["contains_private_rows"] is False


def test_model_plan_rejects_noncanonical_science_event_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_model_plan_fakes(monkeypatch, reverse=True)
    _assert_code("model_plan_event_order_invalid", runner.build_model_plan, _projection())


def test_attempt_authority_binds_execution_source_science_and_exact_order() -> None:
    projection = _projection()
    projection.manifest["bridge_sha256"] = _digest("bridge")
    commitments = {
        "canonical_requests_sha256": _digest("canonical-requests"),
        "pilot_order_sha256": _digest("pilot-order"),
        "remaining_order_sha256": _digest("remaining-order"),
    }
    base_plan = _model_plan()
    plan = runner.ModelPlan(
        model_slice=base_plan.model_slice,
        universe=base_plan.universe,
        canonical_calls=base_plan.canonical_calls,
        execution_calls=base_plan.execution_calls,
        manifest={
            **base_plan.manifest,
            "remaining_order_sha256": commitments["remaining_order_sha256"],
        },
    )
    execution = {
        "plan": contract.CONTRACT_MANIFEST_SHA256,
        "attempt": contract.DEVELOPMENT_ATTEMPT_ID,
        "implementation": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "production_source_inventory_sha256": _digest("production-sources"),
            "test_source_inventory_sha256": _digest("test-sources"),
        },
        "preflight": {
            "commit": "3" * 40,
            "tree": "4" * 40,
            "public_artifact_sha256": _digest("public-artifact"),
            "public_artifact_literal_sha256": _digest("public-artifact-literal"),
            "private_manifest_sha256": _digest("private-manifest"),
            "private_manifest_literal_sha256": _digest("private-manifest-literal"),
        },
        "source": {
            "base_commit": contract.BASE_COMMIT,
            "base_tree": contract.BASE_TREE,
            "terminal_internal_sha256": contract.V38_TERMINAL_INTERNAL_SHA256,
            "inventory_sha256": contract.V38_INVENTORY_SHA256,
            "bridge_sha256": projection.manifest["bridge_sha256"],
        },
        "science": {"projection_sha256": contract.SCIENCE_PROJECTION_SHA256},
        "effect_budget": contract.build_effect_budgets(),
        "request_order": {
            "count": 75,
            "canonical_requests_sha256": commitments["canonical_requests_sha256"],
            "remaining_order_sha256": commitments["remaining_order_sha256"],
        },
        "pilot_order": {
            "count": 5,
            "pilot_order_sha256": commitments["pilot_order_sha256"],
        },
    }

    authority = runner.build_attempt_authority(
        execution_authority=execution,
        projection=projection,
        commitments=commitments,
        plan=plan,
    )

    assert authority == execution
    assert authority is not execution

    changed = copy.deepcopy(execution)
    changed["request_order"]["remaining_order_sha256"] = _digest("different")
    _assert_code(
        "execution_projection_mismatch",
        runner.build_attempt_authority,
        execution_authority=changed,
        projection=projection,
        commitments=commitments,
        plan=plan,
    )


def test_runtime_guard_uses_v22_semantics_but_raw_show_is_diagnostic() -> None:
    pre = _runtime_receipt("pre", raw_show_seed="raw-before")
    post = _runtime_receipt("post", raw_show_seed="raw-after")
    post["modified_at"] = "different-diagnostic-time"
    events = [_digest(f"response:{i}") for i in range(75)]

    guard = runner.build_runtime_segment_guard(
        segment_id="full",
        pre_runtime_receipt=pre,
        post_runtime_receipt=post,
        ordered_generation_response_event_sha256s=events,
        generation_count=75,
    )

    assert guard["identity_http_request_count"] == 4
    assert guard["raw_show_hash_is_diagnostic_only"] is True
    assert guard["modified_at_is_excluded_only"] is True
    assert guard["ordered_generation_response_event_sha256s"] == events

    changed = copy.deepcopy(post)
    changed["version_response_sha256"] = _digest("changed-version")
    _assert_code(
        "runtime_semantic_identity_changed",
        runner.build_runtime_segment_guard,
        segment_id="full",
        pre_runtime_receipt=pre,
        post_runtime_receipt=changed,
        ordered_generation_response_event_sha256s=events,
        generation_count=75,
    )


def test_runtime_aggregate_counts_four_normally_and_eight_after_pause() -> None:
    requests = [_digest(f"request:{i}") for i in range(75)]
    events = [_digest(f"response:{i}") for i in range(75)]
    full = _guard("full", events, seed="full")
    normal = runner.build_runtime_aggregate_guard(
        segment_guards=[full], execution_request_sha256s=requests
    )
    assert normal["segment_ids"] == ["full"]
    assert normal["identity_http_request_count"] == 4
    assert normal["one_guard_does_not_span_pause"] is False

    pilot = _guard("pilot", events[:5], seed="pilot")
    continuation = _guard("continuation", events[5:], seed="continuation")
    paused = runner.build_runtime_aggregate_guard(
        segment_guards=[pilot, continuation], execution_request_sha256s=requests
    )
    assert paused["segment_ids"] == ["pilot", "continuation"]
    assert paused["identity_http_request_count"] == 8
    assert paused["one_guard_does_not_span_pause"] is True
    _assert_code(
        "runtime_aggregate_segment_order_invalid",
        runner.build_runtime_aggregate_guard,
        segment_guards=[continuation, pilot],
        execution_request_sha256s=requests,
    )


def test_latency_pause_boundary_is_strict_and_uses_only_five_pilots() -> None:
    plan = _model_plan()
    exact_duration = 576_000_000_000
    exact = {
        item.request["request_sha256"]: exact_duration
        for item in plan.execution_calls[:5]
    }
    exact_receipt = runner.build_latency_receipt(
        plan=plan, durations_ns_by_request_sha256=exact
    )
    assert exact_receipt["projected_ns"] == contract.PILOT_PROJECTED_THRESHOLD_NS
    assert exact_receipt["pause_required"] is False

    above = dict(exact)
    above[plan.execution_calls[0].request["request_sha256"]] += 1
    above_receipt = runner.build_latency_receipt(
        plan=plan, durations_ns_by_request_sha256=above
    )
    assert above_receipt["projected_ns"] == contract.PILOT_PROJECTED_THRESHOLD_NS + 71
    assert above_receipt["pause_required"] is True
    assert above_receipt["formula"] == (
        "sum(pilot_duration_ns)+70*max(pilot_duration_ns)"
    )

    invalid = dict(exact)
    invalid[plan.execution_calls[0].request["request_sha256"]] = True
    _assert_code(
        "pilot_duration_invalid",
        runner.build_latency_receipt,
        plan=plan,
        durations_ns_by_request_sha256=invalid,
    )


def test_yahoo_effects_are_intent_then_callback_then_commit_without_retry() -> None:
    store = _RecordingEffectStore(yahoo_response_count=2)
    fetched: list[str] = []

    def fetch(url: str) -> tuple[bytes, dict[str, Any]]:
        store.callback_started()
        fetched.append(url)
        return f'{{"url_ordinal":{len(fetched)}}}'.encode("ascii"), {
            "network_requests": 1,
            "retries": 0,
            "redirects": 0,
        }

    runner.execute_yahoo_effects(store, fetcher=fetch)

    assert fetched == list(contract.YAHOO_URLS[2:])
    assert store.yahoo_response_count == 6
    assert [item.intent.request_id for item in store.committed] == [
        "yahoo-03",
        "yahoo-04",
        "yahoo-05",
        "yahoo-06",
    ]
    assert [item.metadata["symbol"] for item in store.committed] == list(
        contract.YAHOO_SYMBOL_ORDER[2:]
    )
    _assert_effect_triples(store)

    failed = _RecordingEffectStore()
    attempts = 0

    def fail_once(_url: str) -> tuple[bytes, dict[str, Any]]:
        nonlocal attempts
        failed.callback_started()
        attempts += 1
        raise RuntimeError("synthetic transport failure")

    with pytest.raises(RuntimeError, match="synthetic transport failure"):
        runner.execute_yahoo_effects(failed, fetcher=fail_once)
    assert attempts == 1
    assert failed.timeline == [
        ("begin", "yahoo-01"),
        ("callback", "yahoo-01"),
    ]
    assert failed.active_intent is not None
    assert failed.committed == []


def test_injected_yahoo_response_cap_accepts_exact_and_rejects_above_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "YAHOO_MAX_RESPONSE_BYTES", 4)
    monkeypatch.setattr(contract, "YAHOO_MAX_TOTAL_RESPONSE_BYTES", 24)
    exact = _RecordingEffectStore()

    def exact_fetch(_url: str) -> tuple[bytes, dict[str, Any]]:
        exact.callback_started()
        return b"xxxx", {"network_requests": 1, "retries": 0, "redirects": 0}

    runner.execute_yahoo_effects(exact, fetcher=exact_fetch)
    assert exact.yahoo_response_count == contract.YAHOO_REQUEST_COUNT
    assert len(exact.committed) == contract.YAHOO_REQUEST_COUNT
    assert {len(item.body) for item in exact.committed} == {4}

    oversized = _RecordingEffectStore()

    def oversized_fetch(_url: str) -> tuple[bytes, dict[str, Any]]:
        oversized.callback_started()
        return b"xxxxx", {"network_requests": 1, "retries": 0, "redirects": 0}

    _assert_code(
        "yahoo_response_bytes_invalid",
        runner.execute_yahoo_effects,
        oversized,
        fetcher=oversized_fetch,
    )
    assert oversized.timeline == [
        ("begin", "yahoo-01"),
        ("callback", "yahoo-01"),
    ]
    assert oversized.active_intent is not None
    assert oversized.committed == []


def _execute_recorded_model_segment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: tuple[runner.PreparedModelCall, ...],
    segment_id: str,
    durations: list[int],
) -> tuple[dict[str, Any], _RecordingEffectStore, list[tuple[bytes, bytes]]]:
    store = _RecordingEffectStore(yahoo_response_count=6)
    duration_iterator = iter(durations)
    callback_bodies: list[bytes] = []
    verified: list[tuple[bytes, bytes]] = []

    def probe(kind: str) -> tuple[bytes, dict[str, Any]]:
        request_id = store.callback_started()
        body = f"probe:{request_id}:{kind}".encode("ascii")
        callback_bodies.append(body)
        return body, {"kind": kind, "synthetic": True}

    def generate(request_bytes: bytes) -> tuple[bytes, dict[str, Any], int]:
        request_id = store.callback_started()
        body = b"sealed-opaque:" + request_id.encode("ascii") + b":" + request_bytes
        callback_bodies.append(body)
        return body, {"synthetic": True}, next(duration_iterator)

    def verify(version_bytes: bytes, show_bytes: bytes) -> dict[str, Any]:
        verified.append((version_bytes, show_bytes))
        return _runtime_receipt(f"verified:{len(verified)}")

    monkeypatch.setattr(runner, "verify_runtime_probe_pair", verify)
    monkeypatch.setattr(
        runner,
        "parse_sealed_generation_response",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("model bodies must remain opaque during segment execution")
        ),
    )
    result = runner.execute_model_segment(
        store,
        calls=calls,
        segment_id=segment_id,
        probe_request=probe,
        generation_request=generate,
    )

    _assert_effect_triples(store)
    identity = [item for item in store.committed if item.intent.effect_kind == "identity"]
    assert len(identity) == 4
    assert verified == [
        (identity[0].body, identity[1].body),
        (identity[2].body, identity[3].body),
    ]
    assert verified[0][0] is identity[0].body
    assert verified[0][1] is identity[1].body
    assert verified[1][0] is identity[2].body
    assert verified[1][1] is identity[3].body
    for sealed in identity:
        assert all(sealed.body is not callback for callback in callback_bodies)

    by_id = {item.intent.request_id: item.intent for item in identity}
    assert by_id[f"{segment_id}-pre-version"].request_sha256 == by_id[
        f"{segment_id}-post-version"
    ].request_sha256
    assert by_id[f"{segment_id}-pre-show"].request_sha256 == by_id[
        f"{segment_id}-post-show"
    ].request_sha256
    return result, store, verified


def test_injected_model_response_cap_accepts_exact_and_rejects_above_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _model_plan()
    exact_body_bytes = len(b"sealed-opaque:generation-001:request:A001")
    monkeypatch.setattr(contract, "MODEL_RESPONSE_MAX_BYTES", exact_body_bytes)
    completed, exact, _verified = _execute_recorded_model_segment(
        monkeypatch,
        calls=plan.execution_calls,
        segment_id="initial",
        durations=[1] * 75,
    )
    assert completed["runtime_guard"]["generation_count"] == 75
    assert {
        len(item.body)
        for item in exact.committed
        if item.intent.effect_kind == "gemma"
    } == {exact_body_bytes}

    monkeypatch.setattr(contract, "MODEL_RESPONSE_MAX_BYTES", exact_body_bytes - 1)
    oversized = _RecordingEffectStore(yahoo_response_count=6)

    def probe(kind: str) -> tuple[bytes, dict[str, Any]]:
        oversized.callback_started()
        return f"probe:{kind}".encode("ascii"), {"kind": kind}

    def generate(_request_bytes: bytes) -> tuple[bytes, dict[str, Any], int]:
        oversized.callback_started()
        return b"x" * exact_body_bytes, {"synthetic": True}, 1

    monkeypatch.setattr(
        runner,
        "verify_runtime_probe_pair",
        lambda version_bytes, show_bytes: _runtime_receipt("cap-check"),
    )
    _assert_code(
        "model_response_bytes_invalid",
        runner.execute_model_segment,
        oversized,
        calls=plan.execution_calls,
        segment_id="initial",
        probe_request=probe,
        generation_request=generate,
    )
    assert [
        item for item in oversized.committed if item.intent.effect_kind == "gemma"
    ] == []
    assert oversized.active_intent is not None
    assert oversized.active_intent.effect_kind == "gemma"

    monkeypatch.setattr(contract, "MODEL_RESPONSE_MAX_BYTES", 4)
    probe_oversized = _RecordingEffectStore(yahoo_response_count=6)

    def oversized_probe(_kind: str) -> tuple[bytes, dict[str, Any]]:
        probe_oversized.callback_started()
        return b"xxxxx", {"synthetic": True}

    _assert_code(
        "runtime_probe_response_bytes_invalid",
        runner.execute_model_segment,
        probe_oversized,
        calls=plan.execution_calls,
        segment_id="initial",
        probe_request=oversized_probe,
        generation_request=lambda _request: (_ for _ in ()).throw(
            AssertionError("generation must remain unreachable")
        ),
    )
    assert probe_oversized.committed == []
    assert probe_oversized.active_intent is not None
    assert probe_oversized.active_intent.effect_kind == "identity"


def test_replay_openers_enforce_exact_yahoo_and_model_response_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "YAHOO_MAX_RESPONSE_BYTES", 4)
    monkeypatch.setattr(contract, "MODEL_RESPONSE_MAX_BYTES", 4)
    plan = _model_plan()

    def yahoo_responses(size: int) -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(
                intent=SimpleNamespace(
                    request_id=f"yahoo-{index + 1:02d}",
                    request_sha256=runner.yahoo_request_sha256(
                        contract.YAHOO_URLS[index]
                    ),
                ),
                metadata={"symbol": symbol},
                body=b"x" * size,
                body_bytes=size,
            )
            for index, symbol in enumerate(contract.YAHOO_SYMBOL_ORDER)
        )

    class ReplayStore:
        def __init__(
            self,
            *,
            yahoo: tuple[SimpleNamespace, ...] = (),
            gemma: tuple[SimpleNamespace, ...] = (),
        ) -> None:
            self.yahoo = yahoo
            self.gemma = gemma

        def committed_responses(
            self, *, effect_kind: str, segment_id: str | None = None
        ) -> tuple[SimpleNamespace, ...]:
            assert segment_id is None
            return self.yahoo if effect_kind == "yahoo" else self.gemma

    monkeypatch.setattr(
        runner,
        "build_stage_slice_from_yahoo",
        lambda *, plan, raw_by_symbol: {
            "slice_sha256": _digest("cap-stage-slice"),
            "body_sizes": sorted(len(item) for item in raw_by_symbol.values()),
        },
    )
    exact_market = runner._open_stage_slice(
        ReplayStore(yahoo=yahoo_responses(4)), plan=plan
    )
    assert exact_market["body_sizes"] == [4] * contract.YAHOO_REQUEST_COUNT
    _assert_code(
        "yahoo_batch_binding_invalid",
        runner._open_stage_slice,
        ReplayStore(yahoo=yahoo_responses(5)),
        plan=plan,
    )

    def gemma_responses(size: int) -> tuple[SimpleNamespace, ...]:
        return tuple(
            SimpleNamespace(
                intent=SimpleNamespace(
                    request_sha256=item.request["request_sha256"]
                ),
                body=b"x" * size,
                body_bytes=size,
                duration_ns=1,
                event_sha256=_digest(f"replayed-generation:{index}"),
            )
            for index, item in enumerate(plan.execution_calls, start=1)
        )

    exact_model = runner._open_generation_batch(
        ReplayStore(gemma=gemma_responses(4)), plan=plan
    )
    assert len(exact_model) == 75
    assert {len(item["response_bytes"]) for item in exact_model.values()} == {4}
    _assert_code(
        "generation_batch_binding_invalid",
        runner._open_generation_batch,
        ReplayStore(gemma=gemma_responses(5)),
        plan=plan,
    )


def test_initial_model_segment_continues_uninterrupted_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _model_plan()
    exact_duration = 576_000_000_000

    result, store, _verified = _execute_recorded_model_segment(
        monkeypatch,
        calls=plan.execution_calls,
        segment_id="initial",
        durations=[exact_duration] * 75,
    )

    assert result["store_segment_id"] == "initial"
    assert result["guard_role"] == "full"
    assert result["pause_required"] is False
    assert result["runtime_guard"]["segment_id"] == "full"
    assert result["runtime_guard"]["store_segment_id"] == "initial"
    assert result["runtime_guard"]["generation_count"] == 75
    assert len(result["duration_ns_by_request_sha256"]) == 75
    assert len(store.generation_durations) == 75

    callbacks = [request_id for action, request_id in store.timeline if action == "callback"]
    assert callbacks[:2] == ["initial-pre-version", "initial-pre-show"]
    assert callbacks[2:7] == [f"generation-{i:03d}" for i in range(1, 6)]
    # At the exact threshold the sixth generation follows inside the same
    # physical segment; no post-probe is inserted after pilot five.
    assert callbacks[7] == "generation-006"
    assert callbacks[-2:] == ["initial-post-version", "initial-post-show"]


def test_above_boundary_closes_pilot_then_continuation_has_own_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _model_plan()
    exact_duration = 576_000_000_000
    pilot_durations = [exact_duration + 1, *([exact_duration] * 4)]

    pilot, pilot_store, _verified = _execute_recorded_model_segment(
        monkeypatch,
        calls=plan.execution_calls,
        segment_id="initial",
        durations=pilot_durations,
    )

    assert pilot["store_segment_id"] == "initial"
    assert pilot["guard_role"] == "pilot"
    assert pilot["pause_required"] is True
    assert pilot["runtime_guard"]["segment_id"] == "pilot"
    assert pilot["runtime_guard"]["store_segment_id"] == "initial"
    assert pilot["runtime_guard"]["generation_count"] == 5
    assert len(pilot_store.generation_durations) == 5
    pilot_callbacks = [
        request_id for action, request_id in pilot_store.timeline if action == "callback"
    ]
    assert pilot_callbacks == [
        "initial-pre-version",
        "initial-pre-show",
        *[f"generation-{i:03d}" for i in range(1, 6)],
        "initial-post-version",
        "initial-post-show",
    ]
    latency = runner.build_latency_receipt(
        plan=plan,
        durations_ns_by_request_sha256=pilot["duration_ns_by_request_sha256"],
    )
    assert latency["pause_required"] is True
    assert latency["projected_ns"] == contract.PILOT_PROJECTED_THRESHOLD_NS + 71

    continuation, continuation_store, _verified = _execute_recorded_model_segment(
        monkeypatch,
        calls=plan.execution_calls[5:],
        segment_id="continuation",
        durations=[1] * 70,
    )
    assert continuation["guard_role"] == "continuation"
    assert continuation["runtime_guard"]["store_segment_id"] == "continuation"
    assert continuation["runtime_guard"]["generation_count"] == 70
    assert len(continuation_store.generation_durations) == 70

    aggregate = runner.build_runtime_aggregate_guard(
        segment_guards=[pilot["runtime_guard"], continuation["runtime_guard"]],
        execution_request_sha256s=[
            item.request["request_sha256"] for item in plan.execution_calls
        ],
    )
    assert aggregate["segment_ids"] == ["pilot", "continuation"]
    assert aggregate["store_segment_ids"] == ["initial", "continuation"]
    assert aggregate["identity_http_request_count"] == 8
    assert aggregate["one_guard_does_not_span_pause"] is True


def test_abnormal_done_reason_is_authenticated_neutral_not_envelope_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_benchmark import sec_filing_gemma_ollama as ollama

    monkeypatch.setattr(
        ollama,
        "_validate_extractor_output_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("abnormal completion must not be schema-validated")
        ),
    )
    output = '{"synthetic":"bounded but not semantically opened"}'
    envelope = {
        "model": "gemma4:12b",
        "created_at": "2026-07-11T10:11:12.123456789Z",
        "message": {"role": "assistant", "content": output},
        "done": True,
        "done_reason": "length",
        "total_duration": 100,
        "load_duration": 10,
        "prompt_eval_count": 20,
        "prompt_eval_duration": 30,
        "eval_count": 40,
        "eval_duration": 50,
    }
    response_bytes = json.dumps(
        envelope, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    parsed = runner.parse_sealed_generation_response(
        response_bytes=response_bytes, supplied_sentence_ids=("S1",)
    )

    assert parsed == {
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "extractor_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "extractor_output_canonical_sha256": None,
        "normal_completion": False,
        "extraction_status": "invalid",
        "validated_output": None,
    }


def test_semantic_rows_hash_all_segment_and_response_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _model_plan()
    response_events = [_digest(f"response-event:{i}") for i in range(1, 76)]
    guard = _guard("full", response_events, seed="semantic")
    aggregate = runner.build_runtime_aggregate_guard(
        segment_guards=[guard],
        execution_request_sha256s=[
            item.request["request_sha256"] for item in plan.execution_calls
        ],
    )
    latency = runner.build_latency_receipt(
        plan=plan,
        durations_ns_by_request_sha256={
            item.request["request_sha256"]: 1 for item in plan.execution_calls[:5]
        },
    )
    sealed: dict[str, dict[str, Any]] = {}
    guards: dict[str, dict[str, Any]] = {}
    for item, event_sha in zip(plan.canonical_calls, response_events, strict=True):
        sealed[item.request["request_sha256"]] = {
            "response_bytes": f"sealed:{item.accession_number}".encode("ascii"),
            "duration_ns": item.canonical_ordinal,
            "response_event_sha256": event_sha,
        }
        guards[item.request["request_sha256"]] = guard

    monkeypatch.setattr(
        runner,
        "parse_sealed_generation_response",
        lambda *, response_bytes, supplied_sentence_ids: {
            "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
            "extractor_output_sha256": _digest("neutral-output"),
            "extractor_output_canonical_sha256": None,
            "normal_completion": False,
            "extraction_status": "invalid",
            "validated_output": None,
        },
    )

    payload = runner.build_semantic_payload(
        plan=plan,
        sealed_calls_by_request_sha256=sealed,
        segment_guard_by_request_sha256=guards,
        runtime_aggregate=aggregate,
        latency_receipt=latency,
    )

    assert len(payload["semantic_extraction_rows"]) == 75
    for item, row in zip(plan.canonical_calls, payload["semantic_extraction_rows"], strict=True):
        event_receipt = row["semantic_event_receipt"]
        assert event_receipt["semantic_event_receipt_sha256"] == contract.canonical_sha256(
            {
                key: value
                for key, value in event_receipt.items()
                if key != "semantic_event_receipt_sha256"
            }
        )
        assert event_receipt["segment_guard_sha256"] == guard["segment_guard_sha256"]
        assert event_receipt["runtime_aggregate_sha256"] == aggregate[
            "runtime_aggregate_sha256"
        ]
        assert row["request_sha256"] == item.request["request_sha256"]
        assert row["document_quality"] is None
        assert row["validated_output"] is None
        assert row["semantic_extraction_row_sha256"] == contract.canonical_sha256(
            {
                key: value
                for key, value in row.items()
                if key != "semantic_extraction_row_sha256"
            }
        )

    first = payload["semantic_extraction_rows"][0]
    changed = {
        key: value
        for key, value in first.items()
        if key != "semantic_extraction_row_sha256"
    }
    changed["runtime_segment_guard_sha256"] = _digest("different-guard")
    assert contract.canonical_sha256(changed) != first["semantic_extraction_row_sha256"]


def _deterministic_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = _neutral_semantic_rows()
    requests: list[dict[str, Any]] = []
    proofs: list[dict[str, Any]] = []
    for i in range(1, 76):
        accession = f"A{i:03}"
        requests.append(
            {
                "accession_number": accession,
                "request_sha256": _digest(f"request:{accession}"),
                "preprocessed_event_sha256": _digest(f"preprocessed:{accession}"),
            }
        )
        proofs.append(
            {
                "current_record": {
                    "accession_number": accession,
                    "form": "10-K" if i % 2 else "10-Q",
                    "availability_session": f"2018-01-{i:03}",
                    "acceptance_datetime": f"2018-01-{i:03}T10:00:00Z",
                    "artifact_stage": "development",
                },
                "universe_event_proof_sha256": _digest(f"proof:{accession}"),
                "current_record_sha256": _digest(f"record:{accession}"),
                "current_filing_sha256": _digest(f"filing:{accession}"),
                "prior_same_form_filing_sha256": None,
                "prior_same_form_record": None,
            }
        )
    stage_slice = {
        "stage": "development",
        "last_value_session": "2018-12-31",
        "slice_sha256": _digest("stage-slice"),
        "model_requests": requests,
        "universe_event_proofs": proofs,
    }
    return stage_slice, semantic


def _install_deterministic_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_benchmark import sec_gemma_online_risk_overlay_features as features
    from agent_benchmark import sec_gemma_online_risk_overlay_production as production

    market_material = {
        "source_commitments": {
            "schema_version": "synthetic-market-source-v1",
            "source_commitments_sha256": _digest("market-source"),
        },
        "market_rows": [{"session": "2018-12-31", "close": "synthetic"}],
        "baseline_signals": [{"session": "2018-12-31", "signal": 0}],
    }
    monkeypatch.setattr(production, "_validated_stage_slice", lambda value: copy.deepcopy(value))
    monkeypatch.setattr(
        production, "_stage_slice_market_material", lambda fixed: copy.deepcopy(market_material)
    )
    monkeypatch.setattr(production, "_validate_universe_proof", lambda value: dict(value))
    monkeypatch.setattr(
        production, "_validate_blinded_model_request", lambda request, proof: (b"x", (), {})
    )
    monkeypatch.setattr(
        production,
        "_decision_market_lookback_rows",
        lambda material, session: [{"decision_session": session}],
    )

    def mint(**kwargs: Any) -> dict[str, Any]:
        body = {
            "schema_version": "synthetic-feature-row-v1",
            "accession_number": kwargs["accession_number"],
            "decision_session": kwargs["decision_session"],
            "document_quality": kwargs["document_quality"],
            "upstream_bindings": kwargs["upstream_bindings"],
        }
        return _with_hash(body, "feature_row_sha256")

    monkeypatch.setattr(features, "_mint_feature_row_from_source_bound_components", mint)


def test_deterministic_payload_is_reproducible_with_pure_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_fakes(monkeypatch)
    stage_slice, semantic = _deterministic_inputs()

    first = runner.build_deterministic_payload(
        stage_slice=stage_slice, semantic_payload=semantic
    )
    second = runner.build_deterministic_payload(
        stage_slice=copy.deepcopy(stage_slice), semantic_payload=copy.deepcopy(semantic)
    )

    assert first == second
    assert len(first["feature_rows"]) == 75
    assert first["feature_row_sha256s"] == [
        row["feature_row_sha256"] for row in first["feature_rows"]
    ]
    for source_row, feature_row in zip(
        semantic["semantic_extraction_rows"], first["feature_rows"], strict=True
    ):
        assert feature_row["upstream_bindings"]["semantic_extraction_row_sha256"] == (
            source_row["semantic_extraction_row_sha256"]
        )
        assert feature_row["feature_row_sha256"] == contract.canonical_sha256(
            {
                key: value
                for key, value in feature_row.items()
                if key != "feature_row_sha256"
            }
        )
    source = first["source_commitments"]
    assert source["v39_source_commitments_sha256"] == contract.canonical_sha256(
        {key: value for key, value in source.items() if key != "v39_source_commitments_sha256"}
    )
    assert first["market_rows_sha256"] == contract.canonical_sha256(first["market_rows"])
    assert first["baseline_signals_sha256"] == contract.canonical_sha256(
        first["baseline_signals"]
    )


def test_deterministic_evaluator_replays_exactly_and_rejects_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_benchmark import sec_gemma_online_risk_overlay_runner as legacy_runner

    expected_bundle = {"stage": "development", "bundle_sha256": _digest("bundle")}
    monkeypatch.setattr(
        legacy_runner,
        "_stage_input_bundle",
        lambda *, stage, outputs: {**expected_bundle, "output_names": sorted(outputs)},
    )

    class ExactEvaluator:
        def evaluate(self, *, stage: str, input_bundle: dict[str, Any]) -> dict[str, Any]:
            assert stage == "development"
            assert input_bundle["output_names"] == ["deterministic", "gemma"]
            return {"schema_version": "synthetic-evaluation-v1", "score": 1}

        def validate(
            self,
            evaluation: dict[str, Any],
            *,
            stage: str,
            input_bundle: dict[str, Any],
        ) -> dict[str, Any]:
            return dict(evaluation)

    monkeypatch.setattr(legacy_runner, "DefaultDeterministicStageEvaluator", ExactEvaluator)
    result = runner.evaluate_deterministic_science(
        semantic_payload={"semantic": "synthetic"},
        deterministic_payload={"deterministic": "synthetic"},
    )
    assert result["evaluation"]["score"] == 1
    assert result["stage_input_bundle"]["bundle_sha256"] == _digest("bundle")

    class MismatchedEvaluator(ExactEvaluator):
        def validate(
            self,
            evaluation: dict[str, Any],
            *,
            stage: str,
            input_bundle: dict[str, Any],
        ) -> dict[str, Any]:
            return {**evaluation, "score": 2}

    monkeypatch.setattr(
        legacy_runner, "DefaultDeterministicStageEvaluator", MismatchedEvaluator
    )
    _assert_code(
        "deterministic_science_replay_mismatch",
        runner.evaluate_deterministic_science,
        semantic_payload={"semantic": "synthetic"},
        deterministic_payload={"deterministic": "synthetic"},
    )


def test_public_privacy_scanner_detects_json_escaped_windows_path() -> None:
    private_windows_path = r"C:\Users\example\private\v3_9"
    artifact = {"diagnostic": private_windows_path}

    assert private_windows_path.encode("utf-8") not in contract.canonical_json_bytes(
        artifact
    )
    assert (
        private_windows_path.replace("\\", "\\\\").encode("utf-8")
        in contract.canonical_json_bytes(artifact)
    )
    _assert_code(
        "public_artifact_privacy_failed",
        runner.assert_public_artifact_private_free,
        artifact,
        forbidden_tokens=(private_windows_path,),
    )


def test_privacy_scanners_fail_closed_for_non_ascii_tokens(tmp_path: Path) -> None:
    _assert_code(
        "public_artifact_privacy_failed",
        runner.assert_public_artifact_private_free,
        {"diagnostic": "safe"},
        forbidden_tokens=("private-\u00c4",),
    )

    private_root = tmp_path / "private"
    private_root.mkdir()
    (private_root / "safe.json").write_bytes(b'{"diagnostic":"safe"}\n')
    _assert_code(
        "private_namespace_privacy_failed",
        runner.assert_private_namespace_private_free,
        private_root,
        forbidden_tokens=("private-\u00c4",),
    )


def _install_fast_orchestration_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep runner integration tests real while avoiding quadratic disk replay.

    The journal and store test modules separately exercise every durable write,
    fsync, no-follow read, and full-replay failure path.  These runner tests need
    the production state machine and real restart artifacts, but they do not
    need to re-read hundreds of immutable files after every single transition.

    Each store instance therefore performs one normal full audit when opened.
    During that open session, appends write the same canonical journal/content
    bytes and refresh through the production replay/state validators using an
    in-memory view of those already-written bytes.  Opening the store again --
    including every explicit ``audit_attempt`` assertion below -- starts with a
    normal disk-backed production audit.  This keeps restart and final-audit
    coverage while changing only redundant test I/O.
    """

    from agent_benchmark import sec_gemma_lean_science_v39_journal as journal_module
    from agent_benchmark import sec_gemma_lean_science_v39_store as store_module

    original_refresh = store_module.AttemptStore._refresh
    original_write_content = store_module.AttemptStore._write_content
    content_cache: dict[Path, dict[str, tuple[dict[str, Any], bytes]]] = {}

    def cache_key(directory: Path) -> Path:
        return directory.resolve()

    def fast_write_content(
        self: Any,
        directory: Path,
        digest: str,
        encoded: bytes,
        *,
        allow_existing_exact: bool,
        conflict_code: str,
    ) -> Path:
        path = original_write_content(
            self,
            directory,
            digest,
            encoded,
            allow_existing_exact=allow_existing_exact,
            conflict_code=conflict_code,
        )
        value = json.loads(encoded.decode("utf-8", errors="strict"))
        assert type(value) is dict
        content_cache.setdefault(cache_key(directory), {})[digest] = (
            value,
            encoded,
        )
        return path

    def fast_journal_append(
        self: Any, event_type: str, payload: Mapping[str, Any]
    ) -> Any:
        if (
            type(event_type) is not str
            or journal_module._EVENT_TYPE_RE.fullmatch(event_type) is None
            or not isinstance(payload, Mapping)
        ):
            raise journal_module.V39JournalError("journal_event_invalid")
        normalized = dict(payload)
        journal_module.canonical_json_bytes(normalized)
        events = getattr(self, "_v39_test_events", None)
        if events is None:
            events = list(self.replay().events)
            self._v39_test_events = events
        sequence = len(events) + 1
        previous = (
            events[-1].event_sha256
            if events
            else journal_module.ZERO_SHA256
        )
        unsigned = {
            "schema_version": journal_module.JOURNAL_SCHEMA_VERSION,
            "sequence": sequence,
            "event_type": event_type,
            "authority_sha256": self.authority_sha256,
            "previous_event_sha256": previous,
            "payload": normalized,
        }
        event_sha256 = journal_module.canonical_sha256(unsigned)
        encoded = journal_module.canonical_json_bytes(
            {**unsigned, "event_sha256": event_sha256}
        )
        if len(encoded) > journal_module.MAX_EVENT_BYTES:
            raise journal_module.V39JournalError("journal_event_too_large")
        final = self.root / f"{sequence:08d}-{event_sha256}.json"
        try:
            with final.open("xb") as handle:
                handle.write(encoded)
        except OSError:
            raise journal_module.V39JournalError(
                "journal_append_durability_failed"
            ) from None
        event = journal_module.JournalEvent(
            sequence,
            event_type,
            self.authority_sha256,
            previous,
            normalized,
            event_sha256,
            final,
        )
        events.append(event)
        return event

    def fast_refresh(self: Any) -> None:
        if not getattr(self, "_v39_test_fast_refresh_ready", False):
            original_refresh(self)
            self._v39_test_fast_refresh_ready = True
            return

        self._require_open()
        journal = self._journal
        if journal is None:
            raise store_module.V39StoreError("store_not_initialized")
        events = getattr(journal, "_v39_test_events", None)
        if events is None:
            events = list(journal.replay().events)
            journal._v39_test_events = events
        replay = journal_module.JournalReplay(
            self.authority_sha256, tuple(events)
        )
        machine = store_module._replay_state(replay, self.authority)

        original_scan_content = store_module._scan_content

        def cached_scan_content(
            directory: Path, *, code: str
        ) -> dict[str, tuple[dict[str, Any], bytes]]:
            del code
            return dict(content_cache.get(cache_key(directory), {}))

        store_module._scan_content = cached_scan_content
        try:
            responses, checkpoints, candidates = store_module._audit_contents(
                machine,
                self.root / store_module.PAYLOAD_DIRECTORY,
                self.root / store_module.CHECKPOINT_DIRECTORY,
            )
            terminal_evidence = store_module._audit_terminal_evidence(
                machine, self.root / store_module.TERMINAL_DIRECTORY
            )
        finally:
            store_module._scan_content = original_scan_content

        self._machine = machine
        self._responses = responses
        self._checkpoints = checkpoints
        self._checkpoint_candidates = candidates
        self._terminal_evidence = terminal_evidence

    monkeypatch.setattr(
        store_module.AttemptStore, "_write_content", fast_write_content
    )
    monkeypatch.setattr(journal_module.HashChainJournal, "append", fast_journal_append)
    monkeypatch.setattr(store_module.AttemptStore, "_refresh", fast_refresh)


def _development_test_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    generation_durations: list[int],
    fail_first_yahoo: bool = False,
) -> tuple[
    runner.DevelopmentDependencies,
    dict[str, Any],
    dict[str, Any],
    runner.ModelPlan,
]:
    from agent_benchmark import sec_gemma_lean_science_v39_journal as journal_module
    from agent_benchmark import sec_gemma_lean_science_v39_store as store_module
    from agent_benchmark.sec_gemma_lean_science_v39_store import AttemptStore

    _install_fast_orchestration_store(monkeypatch)

    # These orchestration tests exercise the real immutable store and replay
    # rules, while the dedicated journal/store modules separately prove the
    # physical fsync calls and their failure paths.  Avoiding repeated device
    # flushes here keeps the complete repository preflight suite bounded
    # without weakening any production durability operation.
    monkeypatch.setattr(journal_module.os, "fsync", lambda _descriptor: None)
    monkeypatch.setattr(journal_module, "fsync_directory", lambda _path: None)
    monkeypatch.setattr(store_module, "fsync_directory", lambda _path: None)

    projection = _projection()
    projection.manifest["bridge_sha256"] = _digest("orchestration-bridge")
    commitments = {
        "canonical_requests_sha256": _digest("orchestration-canonical-requests"),
        "pilot_order_sha256": _digest("orchestration-pilot-order"),
        "remaining_order_sha256": _digest("orchestration-remaining-order"),
    }
    base_plan = _model_plan()
    manifest_body = {
        key: value
        for key, value in base_plan.manifest.items()
        if key != "model_plan_sha256"
    }
    manifest_body.update(
        {
            "pilot_order_sha256": commitments["pilot_order_sha256"],
            "remaining_order_sha256": commitments["remaining_order_sha256"],
        }
    )
    plan = runner.ModelPlan(
        model_slice=base_plan.model_slice,
        universe=base_plan.universe,
        canonical_calls=base_plan.canonical_calls,
        execution_calls=base_plan.execution_calls,
        manifest=_with_hash(manifest_body, "model_plan_sha256"),
    )
    authority = {
        "plan": contract.CONTRACT_MANIFEST_SHA256,
        "attempt": contract.DEVELOPMENT_ATTEMPT_ID,
        "implementation": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "production_source_inventory_sha256": _digest("production-source-inventory"),
            "test_source_inventory_sha256": _digest("test-source-inventory"),
        },
        "preflight": {
            "commit": "3" * 40,
            "tree": "4" * 40,
            "public_artifact_sha256": _digest("preflight-public"),
            "public_artifact_literal_sha256": _digest("preflight-public-literal"),
            "private_manifest_sha256": _digest("preflight-private"),
            "private_manifest_literal_sha256": _digest("preflight-private-literal"),
        },
        "source": {
            "base_commit": contract.BASE_COMMIT,
            "base_tree": contract.BASE_TREE,
            "terminal_internal_sha256": contract.V38_TERMINAL_INTERNAL_SHA256,
            "inventory_sha256": contract.V38_INVENTORY_SHA256,
            "bridge_sha256": projection.manifest["bridge_sha256"],
        },
        "science": {"projection_sha256": contract.SCIENCE_PROJECTION_SHA256},
        "effect_budget": contract.build_effect_budgets(),
        "request_order": {
            "count": 75,
            "canonical_requests_sha256": commitments["canonical_requests_sha256"],
            "remaining_order_sha256": commitments["remaining_order_sha256"],
        },
        "pilot_order": {
            "count": 5,
            "pilot_order_sha256": commitments["pilot_order_sha256"],
        },
    }
    trace: dict[str, Any] = {
        "authenticate_descendant": [],
        "invocation_parent_kind": "preflight",
        "invocation_parents": [],
        "contacts": [],
        "yahoo": [],
        "probes": [],
        "generations": [],
        "pause_publications": [],
        "result_publications": [],
        "stores": [],
    }
    invocation_commits = {
        "preflight": authority["preflight"]["commit"],
        "pause": "5" * 40,
        "continuation": "6" * 40,
    }
    duration_iterator = iter(generation_durations)

    monkeypatch.setattr(
        runner,
        "build_preflight_request_commitments",
        lambda value: copy.deepcopy(commitments),
    )
    monkeypatch.setattr(runner, "build_model_plan", lambda value: plan)
    monkeypatch.setattr(
        runner,
        "verify_runtime_probe_pair",
        lambda version_bytes, show_bytes: _runtime_receipt(
            f"orchestration-runtime:{hashlib.sha256(version_bytes + show_bytes).hexdigest()}"
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_stage_slice_from_yahoo",
        lambda *, plan, raw_by_symbol: {
            "stage": "development",
            "slice_sha256": _digest("synthetic-stage-slice"),
            "symbol_count": len(raw_by_symbol),
        },
    )
    monkeypatch.setattr(
        runner,
        "build_semantic_payload",
        lambda **kwargs: {"schema_version": "synthetic-semantic-v1"},
    )
    monkeypatch.setattr(
        runner,
        "build_deterministic_payload",
        lambda **kwargs: {"schema_version": "synthetic-deterministic-v1"},
    )
    monkeypatch.setattr(
        runner,
        "evaluate_deterministic_science",
        lambda **kwargs: {"evaluation": {"gate_report": {"passed": True}}},
    )
    def private_terminal_material(**kwargs: Any) -> dict[str, Any]:
        parent_kind = kwargs["invocation_parent_kind"]
        body = {
            "schema_version": "synthetic-private-terminal-v1",
            "all_inputs_bound": True,
            "invocation_parent": kwargs["invocation_parent"],
            "invocation_parent_kind": parent_kind,
            "route": "normal" if parent_kind == "preflight" else "paused_resumed",
            "science_summary": {
                "gate_report": {"passed": True, "failed_checks": []},
            },
            "effect_report": copy.deepcopy(kwargs["effect_report"]),
            "market_values_opened": True,
            "model_responses_opened": True,
        }
        return {
            **body,
            "private_terminal_material_sha256": contract.canonical_sha256(body),
        }

    monkeypatch.setattr(
        runner, "build_private_terminal_material", private_terminal_material
    )

    def authenticate_execution(root: Path, descendant: bool) -> dict[str, Any]:
        assert root.is_absolute()
        trace["authenticate_descendant"].append(descendant)
        return copy.deepcopy(authority)

    def load_contact(root: Path) -> str:
        trace["contacts"].append(root)
        return "synthetic-private-contact@example.invalid"

    def authenticate_source(root: Path, contact: str) -> object:
        assert contact == "synthetic-private-contact@example.invalid"
        return object()

    def open_store(path: Path, supplied_authority: dict[str, Any]) -> Any:
        assert supplied_authority == authority
        path.parent.mkdir(parents=True, exist_ok=True)
        store = (
            AttemptStore.open(path, authority=supplied_authority)
            if path.exists()
            else AttemptStore.create(path, authority=supplied_authority)
        )
        trace["stores"].append(store)
        return store

    def inspect_invocation_parent(
        root: Path, supplied_authority: dict[str, Any]
    ) -> dict[str, str]:
        assert root.is_absolute()
        assert supplied_authority == authority
        kind = trace["invocation_parent_kind"]
        value = {
            "invocation_parent": invocation_commits[kind],
            "invocation_parent_kind": kind,
        }
        trace["invocation_parents"].append(copy.deepcopy(value))
        return value

    def yahoo_fetch(url: str) -> tuple[bytes, dict[str, Any]]:
        trace["yahoo"].append(url)
        if fail_first_yahoo:
            raise RuntimeError("synthetic private transport detail")
        return b'{"synthetic":"sealed yahoo"}', {
            "network_requests": 1,
            "retries": 0,
            "redirects": 0,
        }

    def runtime_probe(kind: str) -> tuple[bytes, dict[str, Any]]:
        trace["probes"].append(kind)
        ordinal = len(trace["probes"])
        return f"synthetic-probe:{kind}:{ordinal}".encode("ascii"), {
            "kind": kind,
            "synthetic": True,
        }

    def generation(request_bytes: bytes) -> tuple[bytes, dict[str, Any], int]:
        trace["generations"].append(hashlib.sha256(request_bytes).hexdigest())
        return b'{"synthetic":"sealed gemma"}', {"synthetic": True}, next(
            duration_iterator
        )

    def publish_pause(root: Path, value: dict[str, Any]) -> None:
        trace["pause_publications"].append(copy.deepcopy(value))
        path = root / Path(contract.PAUSE_ARTIFACT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contract.canonical_json_bytes(value))

    def publish_result(root: Path, value: dict[str, Any]) -> None:
        trace["result_publications"].append(copy.deepcopy(value))

    dependencies = runner.DevelopmentDependencies(
        authenticate_execution=authenticate_execution,
        load_private_contact=load_contact,
        authenticate_source=authenticate_source,
        build_projection=lambda source: projection,
        open_store=open_store,
        inspect_invocation_parent=inspect_invocation_parent,
        yahoo_fetch=yahoo_fetch,
        runtime_probe=runtime_probe,
        generation=generation,
        publish_pause=publish_pause,
        publish_result=publish_result,
    )
    return dependencies, trace, authority, plan


def test_development_dependencies_complete_normal_route_without_external_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_benchmark.sec_gemma_lean_science_v39_store import audit_attempt

    dependencies, trace, authority, _plan = _development_test_dependencies(
        monkeypatch, generation_durations=[1] * 75
    )

    result = runner.run_development(tmp_path, dependencies=dependencies)

    assert result["status"] == "passed"
    assert result["terminal_store_status"] == "completed"
    assert result["terminal_code"] == "development_pass"
    assert trace["authenticate_descendant"] == [False]
    assert trace["yahoo"] == list(contract.YAHOO_URLS)
    assert trace["probes"] == ["version", "show", "version", "show"]
    assert len(trace["generations"]) == 75
    assert trace["pause_publications"] == []
    assert trace["result_publications"] == [result]
    assert "yahoo_body_bytes" not in result["effect_report"]
    assert b"yahoo_body_bytes" not in contract.canonical_json_bytes(result)
    assert result["market_values_opened"] is True
    assert result["model_responses_opened"] is True
    snapshot = audit_attempt(
        tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE,
        authority=authority,
    )
    assert snapshot.status == "completed"
    assert snapshot.terminal_code == "development_pass"
    assert snapshot.yahoo_response_count == 6
    assert snapshot.identity_response_count == 4
    assert snapshot.gemma_response_count == 75


def test_development_dependencies_pause_after_five_without_opening_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_benchmark.sec_gemma_lean_science_v39_store import audit_attempt

    exact = 576_000_000_000
    dependencies, trace, authority, _plan = _development_test_dependencies(
        monkeypatch,
        generation_durations=[exact + 1, *([exact] * 4)],
    )
    monkeypatch.setattr(
        runner,
        "build_stage_slice_from_yahoo",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("market values must stay sealed at the pause")
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_semantic_payload",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("model responses must stay sealed at the pause")
        ),
    )

    result = runner.run_development(tmp_path, dependencies=dependencies)

    assert result["status"] == "paused_for_justification"
    assert result["projected_ns"] == contract.PILOT_PROJECTED_THRESHOLD_NS + 71
    assert result["sixth_generation_attempted"] is False
    assert "yahoo_body_bytes" not in result["effect_report"]
    assert "yahoo_body_bytes" not in contract.canonical_json_bytes(result).decode(
        "utf-8"
    )
    assert len(trace["generations"]) == 5
    assert trace["probes"] == ["version", "show", "version", "show"]
    assert trace["pause_publications"] == [result]
    assert trace["result_publications"] == []
    snapshot = audit_attempt(
        tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE,
        authority=authority,
    )
    assert snapshot.status == "paused"
    assert snapshot.yahoo_response_count == 6
    assert snapshot.identity_response_count == 4
    assert snapshot.gemma_response_count == 5


def test_durable_pause_requires_exact_artifact_and_republishes_only_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exact = 576_000_000_000
    dependencies, trace, _authority, _plan = _development_test_dependencies(
        monkeypatch,
        generation_durations=[exact + 1, *([exact] * 4)],
    )

    first = runner.run_development(tmp_path, dependencies=dependencies)
    pause_path = tmp_path / Path(contract.PAUSE_ARTIFACT_PATH)
    frozen_bytes = contract.canonical_json_bytes(first)
    assert pause_path.read_bytes() == frozen_bytes
    assert trace["pause_publications"] == [first]

    trace["invocation_parent_kind"] = "pause"
    second = runner.run_development(tmp_path, dependencies=dependencies)
    assert second == first
    assert trace["pause_publications"] == [first]

    tampered = {**first, "privacy_passed": False}
    pause_path.write_bytes(contract.canonical_json_bytes(tampered))
    _assert_code(
        "pause_artifact_invalid",
        runner.run_development,
        tmp_path,
        dependencies=dependencies,
    )
    assert trace["pause_publications"] == [first]

    pause_path.unlink()
    trace["invocation_parent_kind"] = "preflight"
    recovered = runner.run_development(tmp_path, dependencies=dependencies)
    assert recovered == first
    assert pause_path.read_bytes() == frozen_bytes
    assert trace["pause_publications"] == [first, first]


def _expected_continuation_document(pause: dict[str, Any]) -> bytes:
    durations = pause["pilot_durations_ns"]
    duration_json = json.dumps(durations, separators=(",", ":"))
    effect_json = contract.canonical_json_bytes(pause["effect_report"]).decode("utf-8")
    remaining_ns = runner.REMAINING_CALL_COUNT * max(durations)
    return (
        "# AAPL SEC/Gemma lean science v3.9 continuation preregistration\n\n"
        f"Schema: `{runner.CONTINUATION_DOCUMENT_SCHEMA_VERSION}`\n\n"
        "The authenticated five-generation pilot crossed the frozen twelve-hour "
        "projection threshold. This document freezes only the timing justification "
        "and the already-preregistered remaining local compute.\n\n"
        "## Authenticated pilot evidence\n\n"
        f"- Pilot generations: `{runner.PILOT_COUNT}`\n"
        f"- Pilot durations in nanoseconds: `{duration_json}`\n"
        f"- Projection formula: `{pause['formula']}`\n"
        f"- Projected total nanoseconds: `{pause['projected_ns']}`\n"
        f"- Frozen threshold nanoseconds: `{pause['threshold_ns']}`\n"
        f"- Pilot guard SHA-256: `{pause['pilot_guard_sha256']}`\n"
        f"- Latency receipt SHA-256: `{pause['latency_receipt_sha256']}`\n"
        f"- Pause artifact SHA-256: `{pause['pause_artifact_sha256']}`\n"
        f"- Public effect counts: `{effect_json}`\n\n"
        "## Frozen remaining compute\n\n"
        f"- Physical model segments: `{runner.PILOT_COUNT}` then "
        f"`{runner.REMAINING_CALL_COUNT}` generations\n"
        f"- Remaining order SHA-256: `{pause['remaining_order_sha256']}`\n"
        f"- Expected remaining generations: `{runner.REMAINING_CALL_COUNT}`\n"
        "- Expected additional identity HTTP requests: `4`\n"
        f"- Projected remaining compute nanoseconds: `{remaining_ns}`\n\n"
        "## Unchanged safety rules\n\n"
        "- Development data only; confirmation and final data remain closed.\n"
        "- No SEC request, retry, repair, pull, fallback, paid call, broker effect, "
        "or real-money effect.\n"
        "- Long AAPL or cash only; no shorting, leverage, borrowing, or negative cash.\n"
        "- No science, model, prompt, request order, source, or gate change.\n"
        "- Fresh explicit user permission is still required after this document and "
        "the pause artifact are pushed and authenticated.\n"
    ).encode("ascii")


def test_continuation_document_and_git_blobs_are_bound_to_exact_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    exact = 576_000_000_000
    dependencies, _trace, authority, _plan = _development_test_dependencies(
        monkeypatch,
        generation_durations=[exact + 1, *([exact] * 4)],
    )
    pause = runner.run_development(tmp_path, dependencies=dependencies)
    expected = _expected_continuation_document(pause)
    assert runner.build_continuation_preregistration(pause) == expected

    continuation_path = tmp_path / Path(contract.CONTINUATION_PREREGISTRATION_PATH)
    continuation_path.parent.mkdir(parents=True, exist_ok=True)
    continuation_path.write_bytes(expected)
    head = "6" * 40
    pause_commit = "5" * 40
    preflight_commit = authority["preflight"]["commit"]
    tree = "7" * 40

    def git_text(_root: Path, *arguments: str) -> str:
        values = {
            ("rev-parse", "HEAD"): head,
            ("rev-list", "--parents", "-n", "1", head): (
                f"{head} {pause_commit}"
            ),
            ("rev-list", "--parents", "-n", "1", pause_commit): (
                f"{pause_commit} {preflight_commit}"
            ),
            ("branch", "--show-current"): contract.BRANCH_NAME,
            ("rev-parse", "HEAD^{tree}"): tree,
            (
                "rev-parse",
                f"refs/remotes/origin/{contract.BRANCH_NAME}",
            ): head,
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
        }
        return values[arguments]

    def changed_paths(_root: Path, base: str, tip: str) -> dict[str, str]:
        assert tip == head
        return (
            {contract.CONTINUATION_PREREGISTRATION_PATH: "A"}
            if base == pause_commit
            else {
                contract.PAUSE_ARTIFACT_PATH: "A",
                contract.CONTINUATION_PREREGISTRATION_PATH: "A",
            }
        )

    committed_pause = contract.canonical_json_bytes(pause)

    def git_bytes(_root: Path, *arguments: str) -> bytes:
        assert arguments[0] == "show"
        return expected if arguments[1].startswith("HEAD:") else committed_pause

    monkeypatch.setattr(runner, "_git_text", git_text)
    monkeypatch.setattr(runner, "_git_changed_paths", changed_paths)
    monkeypatch.setattr(runner, "_git_bytes", git_bytes)
    receipt = runner.authenticate_pushed_continuation(
        tmp_path,
        pause_artifact=pause,
        forbidden_tokens=("never-present-private-token",),
    )
    assert receipt == {
        "continuation_commit": head,
        "pause_commit": pause_commit,
        "preflight_commit": preflight_commit,
        "continuation_document_sha256": hashlib.sha256(expected).hexdigest(),
    }

    def merge_git_text(_root: Path, *arguments: str) -> str:
        if arguments == ("rev-list", "--parents", "-n", "1", head):
            return f"{head} {pause_commit} {'8' * 40}"
        return git_text(_root, *arguments)

    monkeypatch.setattr(runner, "_git_text", merge_git_text)
    _assert_code(
        "git_single_parent_invalid",
        runner.authenticate_pushed_continuation,
        tmp_path,
        pause_artifact=pause,
        forbidden_tokens=("never-present-private-token",),
    )
    monkeypatch.setattr(runner, "_git_text", git_text)

    continuation_path.write_bytes(expected + b"\n")
    _assert_code(
        "continuation_document_invalid",
        runner.authenticate_pushed_continuation,
        tmp_path,
        pause_artifact=pause,
        forbidden_tokens=("never-present-private-token",),
    )
    continuation_path.write_bytes(expected)
    monkeypatch.setattr(
        runner,
        "_git_bytes",
        lambda _root, *arguments: (
            expected
            if arguments[1].startswith("HEAD:")
            else committed_pause + b"x"
        ),
    )
    _assert_code(
        "continuation_document_invalid",
        runner.authenticate_pushed_continuation,
        tmp_path,
        pause_artifact=pause,
        forbidden_tokens=("never-present-private-token",),
    )
    monkeypatch.setattr(
        runner,
        "_git_bytes",
        lambda _root, *arguments: (
            expected + b"x"
            if arguments[1].startswith("HEAD:")
            else committed_pause
        ),
    )
    _assert_code(
        "continuation_document_invalid",
        runner.authenticate_pushed_continuation,
        tmp_path,
        pause_artifact=pause,
        forbidden_tokens=("never-present-private-token",),
    )


def test_full_synthetic_pause_restart_binds_exact_continuation_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_benchmark.sec_gemma_lean_science_v39_store import audit_attempt

    exact = 576_000_000_000
    dependencies, trace, authority, _plan = _development_test_dependencies(
        monkeypatch,
        generation_durations=[exact + 1, *([exact] * 4), *([1] * 70)],
    )
    pause = runner.run_development(tmp_path, dependencies=dependencies)
    assert pause["status"] == "paused_for_justification"
    assert len(trace["generations"]) == 5

    continuation_document = runner.build_continuation_preregistration(pause)
    continuation_sha256 = hashlib.sha256(continuation_document).hexdigest()
    continuation_path = tmp_path / Path(contract.CONTINUATION_PREREGISTRATION_PATH)
    continuation_path.parent.mkdir(parents=True, exist_ok=True)
    continuation_path.write_bytes(continuation_document)
    continuation_commit = "6" * 40
    trace["invocation_parent_kind"] = "continuation"
    continuation_authentications: list[dict[str, Any]] = []

    def authenticate_continuation(
        root: Path,
        *,
        pause_artifact: dict[str, Any],
        forbidden_tokens: tuple[bytes | str, ...],
    ) -> dict[str, str]:
        assert root == tmp_path.resolve()
        assert pause_artifact == pause
        assert forbidden_tokens
        receipt = {
            "continuation_commit": continuation_commit,
            "pause_commit": "5" * 40,
            "preflight_commit": authority["preflight"]["commit"],
            "continuation_document_sha256": continuation_sha256,
        }
        continuation_authentications.append(copy.deepcopy(receipt))
        return receipt

    monkeypatch.setattr(
        runner, "authenticate_pushed_continuation", authenticate_continuation
    )
    original_execute = runner.execute_model_segment
    interrupted = {"once": False}

    def interrupt_after_authorization(store: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs["segment_id"] == "continuation" and not interrupted["once"]:
            interrupted["once"] = True
            raise KeyboardInterrupt("synthetic process interruption")
        return original_execute(store, **kwargs)

    monkeypatch.setattr(runner, "execute_model_segment", interrupt_after_authorization)
    permission_sha256 = _digest("fresh-explicit-continuation-permission")
    with pytest.raises(KeyboardInterrupt, match="synthetic process interruption"):
        runner.run_development(
            tmp_path,
            dependencies=dependencies,
            continuation_permission_sha256=permission_sha256,
        )

    interrupted_snapshot = audit_attempt(
        tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE,
        authority=authority,
    )
    assert interrupted_snapshot.status == "active"
    assert interrupted_snapshot.paused is True
    assert interrupted_snapshot.continuation_authorized is True
    assert interrupted_snapshot.continuation_sha256 == continuation_sha256
    assert interrupted_snapshot.continuation_commit == continuation_commit
    assert interrupted_snapshot.continuation_permission_sha256 == permission_sha256
    assert interrupted_snapshot.identity_response_count == 4
    assert interrupted_snapshot.gemma_response_count == 5

    # C2 presents byte-for-byte identical continuation text, but it is not the
    # exact C1 commit durably authorized above.  Reject it before authentication,
    # transport, or terminal mutation; then prove the exact C1 can still resume.
    second_continuation_commit = "7" * 40

    def inspect_second_continuation(
        root: Path, supplied_authority: dict[str, Any]
    ) -> dict[str, str]:
        assert root == tmp_path.resolve()
        assert supplied_authority == authority
        value = {
            "invocation_parent": second_continuation_commit,
            "invocation_parent_kind": "continuation",
        }
        trace["invocation_parents"].append(copy.deepcopy(value))
        return value

    effects_before_c2 = (
        len(trace["yahoo"]),
        len(trace["probes"]),
        len(trace["generations"]),
        len(continuation_authentications),
    )
    _assert_code(
        "continuation_restart_binding_invalid",
        runner.run_development,
        tmp_path,
        dependencies=replace(
            dependencies, inspect_invocation_parent=inspect_second_continuation
        ),
    )
    assert (
        len(trace["yahoo"]),
        len(trace["probes"]),
        len(trace["generations"]),
        len(continuation_authentications),
    ) == effects_before_c2
    after_c2 = audit_attempt(
        tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE,
        authority=authority,
    )
    assert after_c2 == interrupted_snapshot
    assert not (tmp_path / contract.RESULT_ARTIFACT_PATH).exists()
    assert not (tmp_path / f"{contract.RESULT_ARTIFACT_PATH}.v39-pending").exists()
    assert not (tmp_path / f"{contract.COMPARISON_PATH}.v39-pending").exists()

    result = runner.run_development(tmp_path, dependencies=dependencies)
    assert result["status"] == "passed"
    assert result["terminal_store_status"] == "completed"
    assert result["route"] == "paused_resumed"
    assert result["invocation_parent"] == continuation_commit
    assert result["invocation_parent_kind"] == "continuation"
    assert result["market_values_opened"] is True
    assert result["model_responses_opened"] is True
    assert "yahoo_body_bytes" not in result["effect_report"]
    assert b"yahoo_body_bytes" not in contract.canonical_json_bytes(result)
    assert trace["authenticate_descendant"] == [False, True, True, True]
    assert [item["invocation_parent_kind"] for item in trace["invocation_parents"]] == [
        "preflight",
        "continuation",
        "continuation",
        "continuation",
    ]
    assert [item["invocation_parent"] for item in trace["invocation_parents"]] == [
        authority["preflight"]["commit"],
        continuation_commit,
        second_continuation_commit,
        continuation_commit,
    ]
    assert len(continuation_authentications) == 2
    assert {
        item["continuation_document_sha256"]
        for item in continuation_authentications
    } == {continuation_sha256}
    assert trace["yahoo"] == list(contract.YAHOO_URLS)
    assert trace["probes"] == [
        "version",
        "show",
        "version",
        "show",
        "version",
        "show",
        "version",
        "show",
    ]
    assert len(trace["generations"]) == 75

    completed_snapshot = audit_attempt(
        tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE,
        authority=authority,
    )
    assert completed_snapshot.identity_response_count == 8
    assert completed_snapshot.gemma_response_count == 75
    assert completed_snapshot.continuation_sha256 == continuation_sha256
    assert completed_snapshot.continuation_commit == continuation_commit


def test_paused_resumed_failure_reports_route_and_opened_state_honestly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_benchmark.sec_gemma_lean_science_v39_store import audit_attempt

    exact = 576_000_000_000
    dependencies, trace, authority, _plan = _development_test_dependencies(
        monkeypatch,
        generation_durations=[exact + 1, *([exact] * 4), *([1] * 70)],
    )
    pause = runner.run_development(tmp_path, dependencies=dependencies)
    continuation_document = runner.build_continuation_preregistration(pause)
    continuation_sha256 = hashlib.sha256(continuation_document).hexdigest()
    continuation_path = tmp_path / Path(contract.CONTINUATION_PREREGISTRATION_PATH)
    continuation_path.parent.mkdir(parents=True, exist_ok=True)
    continuation_path.write_bytes(continuation_document)
    continuation_commit = "6" * 40
    trace["invocation_parent_kind"] = "continuation"
    continuation_authentications = 0

    def authenticate_failure_continuation(
        root: Path,
        *,
        pause_artifact: dict[str, Any],
        forbidden_tokens: tuple[bytes | str, ...],
    ) -> dict[str, str]:
        nonlocal continuation_authentications
        continuation_authentications += 1
        assert root == tmp_path.resolve()
        assert pause_artifact == pause
        assert forbidden_tokens
        return {
            "continuation_commit": continuation_commit,
            "pause_commit": "5" * 40,
            "preflight_commit": authority["preflight"]["commit"],
            "continuation_document_sha256": continuation_sha256,
        }

    monkeypatch.setattr(
        runner,
        "authenticate_pushed_continuation",
        authenticate_failure_continuation,
    )
    opener_calls = 0

    def fail_after_durable_open_markers(store: Any, *, plan: runner.ModelPlan) -> Any:
        nonlocal opener_calls
        opener_calls += 1
        assert store.snapshot.market_values_opened is True
        assert store.snapshot.model_responses_opened is True
        if opener_calls == 1:
            raise KeyboardInterrupt("synthetic crash after durable open markers")
        raise RuntimeError("synthetic private post-restart failure")

    monkeypatch.setattr(
        runner, "_open_generation_batch", fail_after_durable_open_markers
    )
    permission_sha256 = _digest("failure-route-permission")
    with pytest.raises(
        KeyboardInterrupt, match="synthetic crash after durable open markers"
    ):
        runner.run_development(
            tmp_path,
            dependencies=dependencies,
            continuation_permission_sha256=permission_sha256,
        )

    interrupted = audit_attempt(
        tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE,
        authority=authority,
    )
    assert interrupted.status == "active"
    assert interrupted.market_values_opened is True
    assert interrupted.model_responses_opened is True
    assert interrupted.identity_response_count == 8
    assert interrupted.gemma_response_count == 75

    result = runner.run_development(tmp_path, dependencies=dependencies)
    assert result["status"] == "rejected"
    assert result["terminal_store_status"] == "rejected"
    assert result["terminal_code"] == "development_worker_failed"
    assert result["route"] == "paused_resumed"
    assert result["invocation_parent"] == continuation_commit
    assert result["invocation_parent_kind"] == "continuation"
    assert result["market_values_opened"] is True
    assert result["model_responses_opened"] is True
    assert "yahoo_body_bytes" not in result["effect_report"]
    assert b"yahoo_body_bytes" not in contract.canonical_json_bytes(result)
    assert "synthetic private post-restart failure" not in json.dumps(result)
    assert opener_calls == 2
    assert continuation_authentications == 2
    assert len(trace["generations"]) == 75
    assert trace["probes"] == [
        "version",
        "show",
        "version",
        "show",
        "version",
        "show",
        "version",
        "show",
    ]
    snapshot = audit_attempt(
        tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE,
        authority=authority,
    )
    assert snapshot.status == "rejected"
    assert snapshot.market_values_opened is True
    assert snapshot.model_responses_opened is True
    assert snapshot.identity_response_count == 8
    assert snapshot.gemma_response_count == 75


def test_development_dependencies_terminalize_first_transport_failure_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_benchmark.sec_gemma_lean_science_v39_store import audit_attempt

    dependencies, trace, authority, _plan = _development_test_dependencies(
        monkeypatch,
        generation_durations=[],
        fail_first_yahoo=True,
    )

    result = runner.run_development(tmp_path, dependencies=dependencies)

    assert result["status"] == "indeterminate"
    assert result["terminal_code"] == "development_worker_failed"
    assert "synthetic private transport detail" not in json.dumps(result)
    assert trace["yahoo"] == [contract.YAHOO_URLS[0]]
    assert trace["probes"] == []
    assert trace["generations"] == []
    assert trace["result_publications"] == [result]
    snapshot = audit_attempt(
        tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE,
        authority=authority,
    )
    assert snapshot.status == "indeterminate"
    assert snapshot.terminal_code == "development_worker_failed"
    assert snapshot.yahoo_intent_count == 1
    assert snapshot.yahoo_response_count == 0


@pytest.mark.parametrize("failure_boundary", ["authenticate_source", "build_projection"])
def test_source_failure_precedes_attempt_creation_and_all_external_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_boundary: str
) -> None:
    dependencies, trace, _authority, _plan = _development_test_dependencies(
        monkeypatch,
        generation_durations=[],
    )

    def fail_source(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("synthetic private source failure")

    blocked = replace(
        dependencies,
        **{
            failure_boundary: fail_source,
        },
    )
    _assert_code(
        "development_authority_failed",
        runner.run_development,
        tmp_path,
        dependencies=blocked,
    )
    assert not (tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE).exists()
    assert trace["stores"] == []
    assert trace["yahoo"] == []
    assert trace["probes"] == []
    assert trace["generations"] == []
    assert trace["pause_publications"] == []
    assert trace["result_publications"] == []


@pytest.mark.parametrize(
    ("parent_kind", "expected_code"),
    [
        ("pause", "invocation_parent_store_missing"),
        ("continuation", "invocation_parent_store_missing"),
    ],
)
def test_descendant_parent_cannot_create_a_missing_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parent_kind: str,
    expected_code: str,
) -> None:
    dependencies, trace, _authority, _plan = _development_test_dependencies(
        monkeypatch, generation_durations=[]
    )
    trace["invocation_parent_kind"] = parent_kind

    _assert_code(
        expected_code,
        runner.run_development,
        tmp_path,
        dependencies=dependencies,
    )

    assert not (tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE).exists()
    assert trace["yahoo"] == []
    assert trace["probes"] == []
    assert trace["generations"] == []
    assert trace["pause_publications"] == []
    assert trace["result_publications"] == []


@pytest.mark.parametrize(
    ("parent_kind", "expected_code"),
    [
        ("pause", "pause_invocation_store_mismatch"),
        ("continuation", "continuation_invocation_store_mismatch"),
    ],
)
def test_descendant_parent_rejects_unmatched_active_store_before_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    parent_kind: str,
    expected_code: str,
) -> None:
    dependencies, trace, authority, _plan = _development_test_dependencies(
        monkeypatch, generation_durations=[]
    )
    attempt_root = tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE
    with dependencies.open_store(attempt_root, authority):
        pass
    trace["invocation_parent_kind"] = parent_kind

    _assert_code(
        expected_code,
        runner.run_development,
        tmp_path,
        dependencies=dependencies,
    )

    assert trace["yahoo"] == []
    assert trace["probes"] == []
    assert trace["generations"] == []
    assert trace["pause_publications"] == []
    assert trace["result_publications"] == []


def test_status_reads_only_safe_snapshot_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent_benchmark import sec_gemma_lean_science_v39_preflight as preflight
    from agent_benchmark import sec_gemma_lean_science_v39_store as store

    authority = {"preflight": {"commit": "a" * 40}}
    seen: dict[str, Any] = {}
    monkeypatch.setattr(preflight, "load_execution_authority", lambda root: authority)
    monkeypatch.setattr(
        preflight,
        "load_publication_recovery_authority",
        lambda root: (_ for _ in ()).throw(AssertionError("fallback not expected")),
    )

    def audit(path: Path, *, authority: object) -> SimpleNamespace:
        seen["path"] = path
        seen["authority"] = authority
        return SimpleNamespace(
            status="paused",
            terminal_code=None,
            event_count=19,
            yahoo_response_count=6,
            identity_response_count=4,
            gemma_response_count=5,
            paused=True,
            continuation_authorized=False,
        )

    monkeypatch.setattr(store, "audit_attempt", audit)
    attempt_root = tmp_path / contract.PRIVATE_DEVELOPMENT_NAMESPACE
    attempt_root.mkdir(parents=True)
    value = runner.status(tmp_path)
    assert value == {
        "schema_version": runner.RUNNER_SCHEMA_VERSION,
        "status": "paused",
        "terminal_code": None,
        "event_count": 19,
        "yahoo_response_count": 6,
        "identity_response_count": 4,
        "gemma_response_count": 5,
        "paused": True,
        "continuation_authorized": False,
    }
    assert seen["path"] == tmp_path.resolve() / contract.PRIVATE_DEVELOPMENT_NAMESPACE
    assert seen["authority"] is authority

    missing_root = tmp_path / "missing-attempt"
    missing_root.mkdir()
    monkeypatch.setattr(runner, "_git_text", lambda *_args: "a" * 40)
    assert runner.status(missing_root) == {
        "schema_version": runner.RUNNER_SCHEMA_VERSION,
        "status": "not_started",
    }
    monkeypatch.setattr(runner, "_git_text", lambda *_args: "b" * 40)
    _assert_code("status_attempt_missing", runner.status, missing_root)
    monkeypatch.setattr(
        preflight,
        "load_execution_authority",
        lambda root: (_ for _ in ()).throw(ValueError("private detail")),
    )
    monkeypatch.setattr(
        preflight,
        "load_publication_recovery_authority",
        lambda root: (_ for _ in ()).throw(ValueError("private detail")),
    )
    _assert_code("status_replay_failed", runner.status, tmp_path)


def test_cli_emits_canonical_safe_result_or_fixed_rejection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    success = {"schema_version": runner.RUNNER_SCHEMA_VERSION, "status": "not_started"}
    monkeypatch.setattr(runner, "status", lambda root: success)
    assert runner.main(["status", "--repo-root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == contract.canonical_json_bytes(success).decode("utf-8") + "\n"

    monkeypatch.setattr(
        runner,
        "status",
        lambda root: (_ for _ in ()).throw(runner.V39RunnerError("fixed_rejection")),
    )
    assert runner.main(["status", "--repo-root", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "rejected",
        "code": "fixed_rejection",
    }


def test_development_entrypoint_rejects_invalid_dependency_bundle_before_effect(
    tmp_path: Path,
) -> None:
    _assert_code(
        "development_dependencies_invalid",
        runner.run_development,
        tmp_path,
        dependencies=object(),
    )
