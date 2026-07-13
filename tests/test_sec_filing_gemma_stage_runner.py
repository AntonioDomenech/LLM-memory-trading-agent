from __future__ import annotations

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
from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
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
    run_owned_development_sec_root,
)
from agent_benchmark.sec_point_in_time import content_sha256, validate_sec_user_agent


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
        "run_owned_development_sec_root",
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
