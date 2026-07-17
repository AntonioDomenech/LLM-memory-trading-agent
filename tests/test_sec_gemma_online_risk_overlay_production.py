from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import math
import os
import subprocess
import sys
import time
from types import MethodType

import pandas as pd
import pytest

import agent_benchmark.sec_gemma_online_risk_overlay_production as production
requests = production.requests
from agent_benchmark.sec_audit_transport import SecAuditTransport
from agent_benchmark.sec_filing_gemma_contract import (
    build_extractor_model_payload,
)
from agent_benchmark.sec_filing_gemma_ollama import OLLAMA_ENDPOINT
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    DEVELOPMENT_ACQUISITION_ID,
    MAX_DETERMINISTIC_SECONDS,
    PUBLICATION_NORMAL_OPERATION_SHA256,
    PUBLICATION_WORKER_OWNERSHIP_FIELDS,
    PUBLICATION_WORKER_QUIESCENCE_FIELDS,
    canonical_json_bytes,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_production import (
    SecGemmaOnlineRiskOverlayProductionError,
    VerifiedProductionAuthority,
    VerifiedProductionAuthorities,
    _ADAPTER_SENTINEL,
    _AUTHORITY_SENTINEL,
    _AUTHORITIES_SENTINEL,
    _AcquisitionExecutionLedger,
    _EXECUTOR_SENTINEL,
    _PrivateSecUserAgent,
    _ProductionAcquisitionAdapter,
    _ProductionPublicationRuntime,
    _ProductionPhaseExecutor,
    _PUBLICATION_RUNTIME_SENTINEL,
    _acquisition_worker_payload,
    _decision_market_lookback_rows,
    _deterministic_phase_payload,
    _gemma_phase_payload,
    _run_acquisition_subprocess,
    _run_deterministic_evaluation_subprocess,
    _run_phase_subprocess,
    _runtime_identity_payload,
    _stage_slice_market_material,
    create_production_sec_transport,
    is_verified_production_authority,
)


def _market_stage_slice(count: int = 300) -> dict[str, object]:
    sessions = [
        value.strftime("%Y-%m-%d")
        for value in pd.bdate_range("2000-01-03", periods=count)
    ]
    rows: dict[str, list[dict[str, object]]] = {}
    for symbol_index, symbol in enumerate(
        ("AAPL", "SPY", "QQQ", "IWM", "VIX", "TNX"),
        start=1,
    ):
        values: list[dict[str, object]] = []
        for position, session in enumerate(sessions):
            close = 20.0 + symbol_index + math.exp(position / 1_000)
            values.append(
                {
                    "session": session,
                    "available": True,
                    "values": {
                        "open": (close * 0.999).hex(),
                        "high": (close * 1.01).hex(),
                        "low": (close * 0.99).hex(),
                        "close": close.hex(),
                        "volume": float(1_000_000 + position).hex(),
                        "adjusted_close": (close * 0.95).hex(),
                    },
                }
            )
        rows[symbol] = values
    return {
        "last_value_session": sessions[-1],
        "slice_sha256": "a" * 64,
        "market_rows": rows,
    }


def _test_authority(tmp_path: Path) -> VerifiedProductionAuthority:
    return VerifiedProductionAuthority(
        payload={
            "head_commit": "0" * 40,
            "authority_sha256": "1" * 64,
        },
        repo_root=tmp_path.resolve(),
        _sentinel=_AUTHORITY_SENTINEL,
    )


def _acquisition_worker_request(
    tmp_path: Path,
) -> tuple[dict[str, object], object]:
    from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
        DEVELOPMENT_ACQUISITION_ID,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
        ProductionAcquisitionVault,
        _VAULT_CONSTRUCTOR_SENTINEL,
    )

    store_id = "2" * 64
    vault = ProductionAcquisitionVault(
        database_path=tmp_path / "worker-vault.sqlite3",
        production_authority=True,
        bound_store_instance_id=store_id,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )
    request: dict[str, object] = {
        "command": "development_acquisition",
        "stage": "development",
        "deadline_monotonic": time.monotonic() + 10.0,
        "authority": {
            "payload": {
                "head_commit": "0" * 40,
                "authority_sha256": "1" * 64,
            },
            "repo_root": str(tmp_path.resolve()),
        },
        "vault": {
            "database_path": str(vault._database_path),
            "bound_store_instance_id": store_id,
            "vault_id": vault.vault_id,
        },
        "capability": {
            "attempt_id": DEVELOPMENT_ACQUISITION_ID,
            "allowed_effects": [
                "official_sec_network",
                "market_network",
            ],
            "receipt": {
                "table": "attempts",
                "identity": DEVELOPMENT_ACQUISITION_ID,
                "attempt_id": DEVELOPMENT_ACQUISITION_ID,
                "payload_sha256": "3" * 64,
                "journal_sequence": 1,
                "journal_entry_sha256": "4" * 64,
            },
            "store_instance_id": store_id,
            "store_nonce": "5" * 64,
            "transition_sha256": "6" * 64,
            "store_database_path": str(
                tmp_path / "worker-store.sqlite3"
            ),
            "implementation_manifest": {},
        },
        "sec_user_agent": "Researcher research@openai.com",
        "predecessors": [],
    }
    return request, vault


def test_production_authority_is_exact_and_cannot_be_publicly_forged(
    tmp_path: Path,
) -> None:
    authority = _test_authority(tmp_path)
    assert is_verified_production_authority(authority)

    with pytest.raises(SecGemmaOnlineRiskOverlayProductionError):
        VerifiedProductionAuthority(
            payload={
                "head_commit": "0" * 40,
                "authority_sha256": "1" * 64,
            },
            repo_root=tmp_path.resolve(),
            _sentinel=object(),
        )


def test_exact_reviewed_sec_transport_is_constructed_without_cache(
    tmp_path: Path,
) -> None:
    authority = _test_authority(tmp_path)
    now = time.monotonic()
    parent_deadline = now + 100.0
    transport = create_production_sec_transport(
        authority,
        user_agent="Researcher research@openai.com",
        deadline_monotonic=parent_deadline,
    )

    assert type(transport) is SecAuditTransport
    assert type(transport._session) is requests.Session
    assert transport._session.trust_env is False
    assert transport._session.auth is None
    assert len(transport._session.cookies) == 0
    assert len(transport._session.headers) == 0
    assert transport._session.proxies == {}
    assert (
        transport._session.get_adapter("https://").max_retries.total
        == 0
    )
    state = transport.safe_state()
    assert state["allow_cache_reads"] is False
    assert state["allow_cache_writes"] is False
    assert state["max_retries"] == 0
    assert state["max_redirects"] == 0
    assert transport._parent_deadline_monotonic == parent_deadline
    assert (
        transport._budget._started_at
        + transport._budget.max_seconds
        <= parent_deadline
    )
    transport._session.close()


def test_sec_transport_rejects_foreign_authority_and_deadline(
    tmp_path: Path,
) -> None:
    authority = _test_authority(tmp_path)
    with pytest.raises(SecGemmaOnlineRiskOverlayProductionError):
        create_production_sec_transport(
            object(),  # type: ignore[arg-type]
            user_agent="Researcher research@openai.com",
            deadline_monotonic=time.monotonic() + 100.0,
        )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="clock authorities",
    ):
        create_production_sec_transport(
            authority,
            user_agent="Researcher research@openai.com",
            deadline_monotonic=time.monotonic() + 100.0,
            clock=lambda: time.monotonic(),
        )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="outside the frozen budget",
    ):
        create_production_sec_transport(
            authority,
            user_agent="Researcher research@openai.com",
            deadline_monotonic=time.monotonic() + 1_000.0,
        )


def test_stage_slice_projects_exact_replay_baseline_and_lookback() -> None:
    stage_slice = _market_stage_slice()
    material = _stage_slice_market_material(stage_slice)

    assert len(material["market_rows"]) == 300
    assert len(material["baseline_signals"]) == 300
    assert material["market_rows"][0]["session"] == "2000-01-03"
    assert material["source_commitments"]["stage_slice_sha256"] == "a" * 64

    decision = material["ordered_sessions"][-1]
    lookback = _decision_market_lookback_rows(material, decision)
    assert len(lookback) == 253
    assert lookback[-1]["session"] == decision
    assert set(lookback[-1]["observations"]) == {
        "AAPL",
        "QQQ",
        "SPY",
        "IWM",
        "VIX",
    }


def test_stage_slice_rejects_unavailable_aapl() -> None:
    stage_slice = _market_stage_slice()
    stage_slice["market_rows"]["AAPL"][10] = {
        "session": stage_slice["market_rows"]["AAPL"][10]["session"],
        "available": False,
        "values": None,
    }
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="AAPL stage prefix",
    ):
        _stage_slice_market_material(stage_slice)


class _FakeLoopbackResponse:
    def __init__(self, url: str, body: bytes) -> None:
        self.status_code = 200
        self.url = url
        self.history: list[object] = []
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        self._body = body
        self.closed = False

    def iter_content(self, *, chunk_size: int):
        assert chunk_size > 0
        yield self._body

    def close(self) -> None:
        self.closed = True


class _FakeLoopbackTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def request(
        self,
        method: str,
        url: str,
        **kwargs: object,
    ) -> _FakeLoopbackResponse:
        self.calls.append((method, url, dict(kwargs)))
        return _FakeLoopbackResponse(url, b"{}")


def _blinded_request_and_proof() -> tuple[dict[str, object], dict[str, object]]:
    sentences = [{"id": "C0001", "text": "Revenue was stable."}]
    payload = build_extractor_model_payload(sentences)
    request_bytes = canonical_json_bytes(payload)
    proof_hash = "b" * 64
    proof: dict[str, object] = {
        "universe_event_proof_sha256": proof_hash,
        "current_record": {
            "accession_number": "0000320193-18-000145",
            "form": "10-K",
            "availability_session": "2023-12-29",
            "acceptance_datetime": "2023-12-28T16:00:00Z",
            "artifact_stage": "intermediate",
        },
        "current_record_sha256": "c" * 64,
        "current_filing_sha256": "d" * 64,
        "prior_same_form_record": None,
        "prior_same_form_filing_sha256": None,
    }
    request: dict[str, object] = {
        "schema_version": "test-blinded-model-request-v1",
        "accession_number": "0000320193-18-000145",
        "form": "10-K",
        "availability_session": "2023-12-29",
        "preprocessed_event_sha256": "e" * 64,
        "supplied_sentence_ids": ["C0001"],
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "request_bytes": request_bytes,
    }
    return request, proof


def _valid_call_result(
    _request_bytes: bytes,
    _sentence_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "response_sha256": "1" * 64,
        "extractor_output_sha256": "2" * 64,
        "extractor_output_canonical_sha256": "3" * 64,
        "normal_completion": True,
        "extraction_status": "valid",
        "validated_output": {"document_quality": "usable"},
        "http_status": 200,
        "response_url": OLLAMA_ENDPOINT,
        "response_content_type": "application/json",
        "response_history_count": 0,
        "elapsed_seconds_hex": (1.0).hex(),
    }


def test_local_preflight_runtime_probe_never_calls_generation() -> None:
    transport = _FakeLoopbackTransport()

    payload = _runtime_identity_payload(transport=transport)

    assert bytes.fromhex(payload["version_response_hex"]) == b"{}"
    assert bytes.fromhex(payload["show_response_hex"]) == b"{}"
    assert [call[0] for call in transport.calls] == ["GET", "POST"]
    assert [call[1].rsplit("/", 1)[-1] for call in transport.calls] == [
        "version",
        "show",
    ]
    assert all(call[1] != OLLAMA_ENDPOINT for call in transport.calls)


def test_gemma_payload_embeds_event_receipts_and_batch_binding() -> None:
    request, proof = _blinded_request_and_proof()
    model_slice = {
        "stage": "confirmation",
        "attempt_id": "test-attempt",
        "model_requests": [request],
        "universe_event_proofs": [proof],
        "model_slice_sha256": "a" * 64,
    }
    runtime = {
        "version_response_hex": b"{}".hex(),
        "show_response_hex": b"{}".hex(),
    }

    payload = _gemma_phase_payload(
        model_slice,
        runtime,
        request_call=_valid_call_result,
        universe_validator=lambda value: copy.deepcopy(value),
        slice_validator=lambda value: copy.deepcopy(value),
    )

    assert len(payload["semantic_extraction_rows"]) == 1
    row = payload["semantic_extraction_rows"][0]
    receipt = row["semantic_event_receipt"]
    assert receipt["retry_count"] == 0
    assert receipt["fallback_count"] == 0
    assert (
        receipt["semantic_event_receipt_sha256"]
        == row["semantic_event_receipt_sha256"]
    )
    assert (
        canonical_sha256(
            {
                key: value
                for key, value in row.items()
                if key != "semantic_extraction_row_sha256"
            }
        )
        == row["semantic_extraction_row_sha256"]
    )
    assert (
        row["latency_preflight_receipt"]["selection"]
        == "not_applicable"
    )
    assert (
        row["latency_preflight_receipt"][
            "calls_are_unique_and_not_duplicated"
        ]
        is True
    )
    assert len(payload["semantic_batch_receipt_sha256"]) == 64


def _development_event(
    index: int,
    *,
    text_size: int,
    availability: str,
    acceptance: str,
) -> tuple[dict[str, object], dict[str, object]]:
    accession = f"0000320193-{index:02d}-{index:06d}"
    payload = build_extractor_model_payload(
        [{"id": "C0001", "text": "x" * text_size}]
    )
    request_bytes = canonical_json_bytes(payload)
    proof: dict[str, object] = {
        "universe_event_proof_sha256": (
            f"{index:064x}"[-64:]
        ),
        "current_record": {
            "accession_number": accession,
            "form": "10-Q",
            "availability_session": availability,
            "acceptance_datetime": acceptance,
            "artifact_stage": "development",
        },
        "current_record_sha256": f"{index + 20:064x}"[-64:],
        "current_filing_sha256": f"{index + 40:064x}"[-64:],
        "prior_same_form_record": None,
        "prior_same_form_filing_sha256": None,
    }
    request: dict[str, object] = {
        "schema_version": "test-blinded-model-request-v1",
        "accession_number": accession,
        "form": "10-Q",
        "availability_session": availability,
        "preprocessed_event_sha256": f"{index + 60:064x}"[-64:],
        "supplied_sentence_ids": ["C0001"],
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "request_bytes": request_bytes,
    }
    return request, proof


def test_development_latency_preflight_calls_largest_five_first_once() -> None:
    events = [
        _development_event(
            index,
            text_size=index * 10,
            availability=(
                "2001-01-03"
                if index == 2
                else "2001-01-04"
                if index == 1
                else f"200{index}-01-03"
            ),
            acceptance=f"200{index}-01-02T16:00:00Z",
        )
        for index in range(1, 8)
    ]
    requests_batch = [item[0] for item in events]
    proofs = [item[1] for item in events]
    accession_by_hash = {
        item["request_sha256"]: item["accession_number"]
        for item in requests_batch
    }
    observed: list[str] = []

    def call(
        request_bytes: bytes,
        sentence_ids: tuple[str, ...],
    ) -> dict[str, object]:
        assert sentence_ids == ("C0001",)
        observed.append(
            accession_by_hash[hashlib.sha256(request_bytes).hexdigest()]
        )
        return _valid_call_result(request_bytes, sentence_ids)

    payload = _gemma_phase_payload(
        {
            "stage": "development",
            "attempt_id": "development-test",
            "model_requests": requests_batch,
            "universe_event_proofs": proofs,
            "model_slice_sha256": "a" * 64,
        },
        {
            "version_response_hex": b"{}".hex(),
            "show_response_hex": b"{}".hex(),
        },
        request_call=call,
        universe_validator=lambda value: copy.deepcopy(value),
        slice_validator=lambda value: copy.deepcopy(value),
    )

    expected_preflight = [
        requests_batch[index - 1]["accession_number"]
        for index in (7, 6, 5, 4, 3)
    ]
    expected_remaining = [
        requests_batch[index - 1]["accession_number"]
        for index in (2, 1)
    ]
    assert observed == expected_preflight + expected_remaining
    assert len(observed) == len(set(observed)) == 7
    latency = payload["semantic_extraction_rows"][0][
        "latency_preflight_receipt"
    ]
    assert latency["preflight_accession_numbers"] == expected_preflight
    assert float.fromhex(latency["projected_total_seconds_hex"]) == 7.0
    assert latency["projection_passed"] is True
    assert [
        row["accession_number"]
        for row in payload["semantic_extraction_rows"]
    ] == [item["accession_number"] for item in requests_batch]


def test_development_latency_projection_fails_before_remaining_calls() -> None:
    events = [
        _development_event(
            index,
            text_size=index * 10,
            availability=f"200{index}-01-03",
            acceptance=f"200{index}-01-02T16:00:00Z",
        )
        for index in range(1, 7)
    ]
    calls = [0]

    def slow_call(
        request_bytes: bytes,
        sentence_ids: tuple[str, ...],
    ) -> dict[str, object]:
        calls[0] += 1
        result = _valid_call_result(request_bytes, sentence_ids)
        result["elapsed_seconds_hex"] = (500.0).hex()
        return result

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="latency projection",
    ):
        _gemma_phase_payload(
            {
                "stage": "development",
                "attempt_id": "development-test",
                "model_requests": [item[0] for item in events],
                "universe_event_proofs": [item[1] for item in events],
                "model_slice_sha256": "a" * 64,
            },
            {
                "version_response_hex": b"{}".hex(),
                "show_response_hex": b"{}".hex(),
            },
            request_call=slow_call,
            universe_validator=lambda value: copy.deepcopy(value),
            slice_validator=lambda value: copy.deepcopy(value),
        )
    assert calls[0] == 5


def test_development_latency_ties_break_by_accession_ascending() -> None:
    events = [
        _development_event(
            index,
            text_size=100,
            availability=f"200{index}-01-03",
            acceptance=f"200{index}-01-02T16:00:00Z",
        )
        for index in range(5, 0, -1)
    ]
    calls = [0]

    def call(
        request_bytes: bytes,
        sentence_ids: tuple[str, ...],
    ) -> dict[str, object]:
        calls[0] += 1
        return _valid_call_result(request_bytes, sentence_ids)

    payload = _gemma_phase_payload(
        {
            "stage": "development",
            "attempt_id": "development-test",
            "model_requests": [item[0] for item in events],
            "universe_event_proofs": [item[1] for item in events],
            "model_slice_sha256": "a" * 64,
        },
        {
            "version_response_hex": b"{}".hex(),
            "show_response_hex": b"{}".hex(),
        },
        request_call=call,
        universe_validator=lambda value: copy.deepcopy(value),
        slice_validator=lambda value: copy.deepcopy(value),
    )
    preflight = payload["semantic_extraction_rows"][0][
        "latency_preflight_receipt"
    ]["preflight_accession_numbers"]
    assert calls[0] == 5
    assert preflight == sorted(preflight)


def test_deterministic_payload_binds_semantics_market_and_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, proof = _blinded_request_and_proof()
    runtime = {
        "version_response_hex": b"{}".hex(),
        "show_response_hex": b"{}".hex(),
    }
    gemma = _gemma_phase_payload(
        {
            "stage": "confirmation",
            "attempt_id": "test-attempt",
            "model_requests": [request],
            "universe_event_proofs": [proof],
            "model_slice_sha256": "a" * 64,
        },
        runtime,
        request_call=_valid_call_result,
        universe_validator=lambda value: copy.deepcopy(value),
        slice_validator=lambda value: copy.deepcopy(value),
    )
    market_commitments = {
        "source_commitments_sha256": "4" * 64,
        "stage_slice_sha256": "5" * 64,
    }
    market_material = {
        "market_rows": [
            {"session": "2000-01-03"},
            {"session": "2023-12-29"},
        ],
        "baseline_signals": [
            {"session": "2000-01-03"},
            {"session": "2023-12-29"},
        ],
        "ordered_sessions": ["2023-12-29"],
        "feature_market_rows_by_session": {
            "2023-12-29": {
                "session": "2023-12-29",
                "observations": {},
            }
        },
        "source_commitments": market_commitments,
    }
    monkeypatch.setattr(
        "agent_benchmark.sec_gemma_online_risk_overlay_production._stage_slice_market_material",
        lambda _value: copy.deepcopy(market_material),
    )

    def fake_minter(**kwargs: object) -> dict[str, object]:
        body = {
            "accession_number": kwargs["accession_number"],
            "upstream_bindings": kwargs["upstream_bindings"],
        }
        return {
            **body,
            "feature_row_sha256": canonical_sha256(body),
        }

    result = _deterministic_phase_payload(
        {
            "schema_version": "test-stage-slice-v1",
            "stage": "confirmation",
            "attempt_id": "test-attempt",
            "attempt_kind": "development",
            "last_value_session": "2023-12-29",
            "market_rows": {},
            "model_requests": [request],
            "universe_event_proofs": [proof],
            "slice_sha256": "5" * 64,
        },
        runtime,
        gemma,
        universe_validator=lambda value: copy.deepcopy(value),
        slice_validator=lambda value: copy.deepcopy(value),
        feature_minter=fake_minter,
    )

    assert result["market_rows"][0]["session"] == "2000-01-03"
    assert len(result["feature_rows"]) == 1
    assert (
        result["semantic_batch_receipt_sha256"]
        == gemma["semantic_batch_receipt_sha256"]
    )
    assert (
        result["source_commitments"][
            "semantic_batch_receipt_sha256"
        ]
        == gemma["semantic_batch_receipt_sha256"]
    )


def test_phase_subprocess_timeout_terminates_worker() -> None:
    class Process:
        def __init__(
            self,
            command: tuple[str, ...],
            **_kwargs: object,
        ) -> None:
            self.command = command
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def communicate(
            self,
            *,
            input: bytes,
            timeout: float,
        ) -> tuple[bytes, None]:
            del input
            raise subprocess.TimeoutExpired(self.command, timeout)

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    created: list[Process] = []

    def popen(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> Process:
        process = Process(command, **kwargs)
        created.append(process)
        return process

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="hard deadline",
    ):
        _run_phase_subprocess(
            {
                "command": "local_preflight",
                "phase": "runtime_identity",
                "prior_phase_outputs": {},
                "released_slice": None,
            },
            deadline_monotonic=1.0,
            clock=lambda: 0.0,
            popen_factory=popen,
        )
    assert created[0].terminated is True


def test_deterministic_evaluation_stall_is_killed_within_480_seconds() -> None:
    class Process:
        def __init__(
            self,
            command: tuple[str, ...],
            **_kwargs: object,
        ) -> None:
            self.command = command
            self.returncode: int | None = None
            self.terminated = False
            self.timeout: float | None = None

        def poll(self) -> int | None:
            return self.returncode

        def communicate(
            self,
            *,
            input: bytes,
            timeout: float,
        ) -> tuple[bytes, None]:
            del input
            self.timeout = timeout
            raise subprocess.TimeoutExpired(self.command, timeout)

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

    created: list[Process] = []

    def popen(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> Process:
        process = Process(command, **kwargs)
        created.append(process)
        return process

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="hard deadline",
    ):
        _run_deterministic_evaluation_subprocess(
            stage="confirmation",
            input_bundle={"stage": "confirmation"},
            deadline_monotonic=float(MAX_DETERMINISTIC_SECONDS),
            clock=lambda: 0.0,
            popen_factory=popen,
        )

    process = created[0]
    assert process.timeout is not None
    assert process.timeout <= MAX_DETERMINISTIC_SECONDS
    assert process.terminated is True


def test_deterministic_stdio_worker_real_spawn_smoke() -> None:
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="failed closed",
    ):
        _run_deterministic_evaluation_subprocess(
            stage="confirmation",
            input_bundle={"stage": "confirmation"},
            deadline_monotonic=time.monotonic() + 20.0,
        )


def test_acquisition_subprocess_timeout_hard_terminates_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC_PRIVATE_SENTINEL", "must-not-reach-worker")

    class Process:
        def __init__(
            self,
            command: tuple[str, ...],
            **kwargs: object,
        ) -> None:
            self.command = command
            self.kwargs = kwargs
            self.returncode: int | None = None
            self.terminated = False
            self.killed = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self, timeout: float) -> int:
            del timeout
            assert self.returncode is not None
            return self.returncode

        def communicate(
            self,
            *,
            input: bytes,
            timeout: float,
        ) -> tuple[bytes, None]:
            assert b"private@example.com" in input
            raise subprocess.TimeoutExpired(self.command, timeout)

    created: list[Process] = []

    def popen(
        command: tuple[str, ...],
        **kwargs: object,
    ) -> Process:
        process = Process(command, **kwargs)
        created.append(process)
        return process

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="hard deadline",
    ):
        _run_acquisition_subprocess(
            {
                "command": "development_acquisition",
                "stage": "development",
                "sec_user_agent": "Private private@example.com",
            },
            deadline_monotonic=1.0,
            clock=lambda: 0.0,
            popen_factory=popen,
        )
    process = created[0]
    assert process.terminated is True
    assert all("private@example.com" not in item for item in process.command)
    assert process.command[1:4] == ("-I", "-S", "-c")
    assert process.command[7] == (
        "agent_benchmark.sec_gemma_online_risk_overlay_production"
    )
    assert process.command[8] == "--acquisition-worker"
    assert process.kwargs["env"] == {
        "SYSTEMROOT": str(Path(production.os.environ["SYSTEMROOT"])),
        "WINDIR": str(Path(production.os.environ["WINDIR"])),
    }
    assert "SEC_PRIVATE_SENTINEL" not in process.kwargs["env"]
    assert process.kwargs["cwd"] == str(Path(production.__file__).parents[1])
    assert process.kwargs["stderr"] is subprocess.DEVNULL


def test_worker_cli_rejects_ambient_start_before_reading_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_called = False

    def forbidden_read() -> dict[str, object]:
        nonlocal read_called
        read_called = True
        return {}

    monkeypatch.setattr(
        production,
        "_read_stdio_worker_request",
        forbidden_read,
    )

    assert production._production_worker_cli(["--acquisition-worker"]) == 1
    assert read_called is False


def test_real_worker_bootstrap_enters_cli_in_isolated_no_site_mode() -> None:
    root = Path(production.__file__).resolve().parents[1]
    bootstrap_material = production.canonical_json_bytes(
        production._worker_bootstrap_material()
    ).decode("ascii")
    command = (
        str(Path(production.sys.executable).resolve()),
        "-I",
        "-S",
        "-c",
        production._WORKER_BOOTSTRAP,
        str(root),
        bootstrap_material,
        production._WORKER_MODULE,
        "--phase-worker",
    )

    result = subprocess.run(
        command,
        cwd=root,
        env=production._sanitized_worker_environment(),
        input=b"{}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert result.returncode == 1
    assert result.stdout == b""
    assert result.stderr == b""


def test_worker_bootstrap_allows_only_bound_external_import_roots() -> None:
    root = Path(production.__file__).resolve().parents[1]
    material = production._worker_bootstrap_material()
    assert material["allowed_external_import_roots"] == [
        "certifi",
        "charset_normalizer",
        "idna",
        "numpy",
        "requests",
        "tzdata",
        "urllib3",
    ]
    probe = production._WORKER_BOOTSTRAP.replace(
        (
            "runpy.run_module("
            "module_name, run_name=\"__main__\", alter_sys=True)"
        ),
        """
import numpy
import tzdata
try:
    import yfinance
except ModuleNotFoundError as exc:
    if exc.name != "yfinance":
        fail()
else:
    fail()
if tuple(allowed_external_import_roots) != tuple(
    sys._sec_gemma_worker_external_import_roots
):
    fail()
raise SystemExit(0)
""",
    )
    assert probe != production._WORKER_BOOTSTRAP
    command = (
        str(Path(production.sys.executable).resolve()),
        "-I",
        "-S",
        "-c",
        probe,
        str(root),
        production.canonical_json_bytes(material).decode("ascii"),
        production._WORKER_MODULE,
        "--phase-worker",
    )

    result = subprocess.run(
        command,
        cwd=root,
        env=production._sanitized_worker_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""


def test_acquisition_worker_rejects_stage_and_expired_deadline_before_effects(
    tmp_path: Path,
) -> None:
    request, _vault = _acquisition_worker_request(tmp_path)
    crossed = copy.deepcopy(request)
    crossed["stage"] = "confirmation"
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="crossed its stage",
    ):
        _acquisition_worker_payload(crossed)

    expired = copy.deepcopy(request)
    expired["deadline_monotonic"] = time.monotonic() - 1.0
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="outside the frozen cap",
    ):
        _acquisition_worker_payload(expired)


def test_spawned_acquisition_worker_fails_closed_before_network(
    tmp_path: Path,
) -> None:
    request, _vault = _acquisition_worker_request(tmp_path)
    request["stage"] = "confirmation"

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="failed closed",
    ):
        _run_acquisition_subprocess(
            request,
            deadline_monotonic=time.monotonic() + 10.0,
        )


def test_acquisition_worker_rejects_foreign_capability_store_and_predecessor(
    tmp_path: Path,
) -> None:
    request, _vault = _acquisition_worker_request(tmp_path)

    fabricated = copy.deepcopy(request)
    fabricated_capability = fabricated["capability"]
    assert isinstance(fabricated_capability, dict)
    fabricated_receipt = fabricated_capability["receipt"]
    assert isinstance(fabricated_receipt, dict)
    fabricated_receipt["attempt_id"] = "foreign-attempt"
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="effects are invalid",
    ):
        _acquisition_worker_payload(fabricated)

    foreign_store = copy.deepcopy(request)
    foreign_capability = foreign_store["capability"]
    assert isinstance(foreign_capability, dict)
    foreign_capability["store_instance_id"] = "7" * 64
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="store and vault authorities differ",
    ):
        _acquisition_worker_payload(foreign_store)

    foreign_predecessor = copy.deepcopy(request)
    foreign_predecessor["predecessors"] = [{}]
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="predecessor chain is foreign or incomplete",
    ):
        _acquisition_worker_payload(foreign_predecessor)


def test_parent_revalidates_consumed_capability_after_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
        DEVELOPMENT_ACQUISITION_ID,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_store import (
        EffectCapability,
        SecGemmaOnlineRiskOverlayStore,
        StoreRecordReceipt,
        _CAPABILITY_SENTINEL,
    )
    from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
        ProductionAcquisitionVault,
        _VAULT_CONSTRUCTOR_SENTINEL,
    )

    store_id = "8" * 64
    store = object.__new__(SecGemmaOnlineRiskOverlayStore)
    store._store_instance_id = store_id
    store._path = tmp_path / "parent-store.sqlite3"
    store._implementation_manifest = {}
    calls: list[str] = []

    def authorize_effect(
        _store: object,
        _capability: object,
        effect: str,
    ) -> None:
        calls.append(effect)
        if len(calls) > 2:
            raise RuntimeError("capability was revoked")

    store.authorize_effect = MethodType(authorize_effect, store)
    vault = ProductionAcquisitionVault(
        database_path=tmp_path / "parent-vault.sqlite3",
        production_authority=True,
        bound_store_instance_id=store_id,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )
    receipt = StoreRecordReceipt(
        table="attempts",
        identity=DEVELOPMENT_ACQUISITION_ID,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        payload_sha256="9" * 64,
        journal_sequence=1,
        journal_entry_sha256="a" * 64,
    )
    capability = EffectCapability(
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        allowed_effects=(
            "official_sec_network",
            "market_network",
        ),
        receipt=receipt,
        store_instance_id=store_id,
        store_nonce="b" * 64,
        transition_sha256="c" * 64,
        _sentinel=_CAPABILITY_SENTINEL,
    )
    ledger = _AcquisitionExecutionLedger()
    adapter = _ProductionAcquisitionAdapter(
        authority=_test_authority(tmp_path),
        repo_root=tmp_path.resolve(),
        store=store,
        vault=vault,
        private_identity=_PrivateSecUserAgent(
            "Researcher research@openai.com"
        ),
        ledger=ledger,
        clock=time.monotonic,
        sleeper=time.sleep,
        _sentinel=_ADAPTER_SENTINEL,
    )
    monkeypatch.setattr(
        production,
        "_run_acquisition_subprocess",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="changed while its worker ran",
    ):
        adapter.acquire(
            command="development_acquisition",
            store=store,
            capability=capability,
            deadline_monotonic=time.monotonic() + 10.0,
        )

    assert calls == [
        "official_sec_network",
        "market_network",
        "official_sec_network",
    ]
    assert ledger.retained_stages() == ()


def test_production_executor_wraps_only_subprocess_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _ProductionPhaseExecutor(
        ledger=_AcquisitionExecutionLedger(),
        clock=time.monotonic,
        _sentinel=_EXECUTOR_SENTINEL,
    )
    monkeypatch.setattr(
        "agent_benchmark.sec_gemma_online_risk_overlay_production._run_phase_subprocess",
        lambda *_args, **_kwargs: {
            "counters": {
                "sec_request_count": 0,
                "market_request_count": 0,
                "model_call_count": 0,
                "retry_count": 0,
                "fallback_count": 0,
                "model_pull_count": 0,
                "paid_api_call_count": 0,
            },
            "payload": {
                "version_response_hex": b"{}".hex(),
                "show_response_hex": b"{}".hex(),
            },
        },
    )

    output = executor.execute(
        command="local_preflight",
        phase="runtime_identity",
        permit=None,
        deadline_monotonic=time.monotonic() + 10.0,
        prior_phase_outputs={},
        acquisition_execution=None,
    )

    assert output["command"] == "local_preflight"
    assert output["phase"] == "runtime_identity"
    assert output["counters"]["model_call_count"] == 0


def test_production_executor_hard_bounds_full_deterministic_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _ProductionPhaseExecutor(
        ledger=_AcquisitionExecutionLedger(),
        clock=time.monotonic,
        _sentinel=_EXECUTOR_SENTINEL,
    )
    expected = {"deterministic_evaluation_sha256": "a" * 64}
    observed: dict[str, object] = {}

    def fake_worker(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return copy.deepcopy(expected)

    monkeypatch.setattr(
        production,
        "_run_deterministic_evaluation_subprocess",
        fake_worker,
    )
    deadline = time.monotonic() + 10.0

    result = executor.evaluate_deterministic(
        stage="confirmation",
        input_bundle={"stage": "confirmation"},
        deadline_monotonic=deadline,
    )

    assert result == expected
    assert observed["stage"] == "confirmation"
    assert observed["deadline_monotonic"] == deadline
    assert observed["clock"] is time.monotonic


def test_private_identity_and_bundle_never_accept_generic_authorities(
    tmp_path: Path,
) -> None:
    identity = _PrivateSecUserAgent(
        "Researcher research@openai.com"
    )
    assert "research@openai.com" not in repr(identity)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="exact identity-equal components",
    ):
        VerifiedProductionAuthorities(
            repo_root=tmp_path.resolve(),
            implementation_manifest={},
            store=object(),
            authority=_test_authority(tmp_path),
            private_identity=identity,
            vault=object(),
            phase_executor=object(),  # type: ignore[arg-type]
            acquisition_adapter=object(),  # type: ignore[arg-type]
            report_publisher=object(),
            publication_runtime=object(),  # type: ignore[arg-type]
            final_registry_authority=object(),
            _sentinel=_AUTHORITIES_SENTINEL,
        )


def test_restart_readiness_fails_before_consumption_without_opaque_state() -> None:
    from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
        DEVELOPMENT_ACQUISITION_ID,
    )

    class Store:
        def attempt_history(self, attempt_id: str) -> list[dict[str, str]]:
            if attempt_id == DEVELOPMENT_ACQUISITION_ID:
                return [{"status": "terminal_pass"}]
            return []

    ledger = _AcquisitionExecutionLedger()
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="absent, restarted",
    ):
        ledger.assert_ready_for_command("development", store=Store())
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="already exists durably",
    ):
        ledger.assert_ready_for_command(
            "development_acquisition",
            store=Store(),
        )


class _FakePublicationKernel:
    def __init__(self, *, job_name: str, mutex_name: str) -> None:
        self.job_name = job_name
        self.mutex_name = mutex_name
        self.executions: list[dict[str, object]] = []
        self.quiesce_count = 0
        self.abort_count = 0

    def execute(self, **kwargs: object):
        from agent_benchmark.sec_gemma_online_risk_overlay_publisher import (
            PublicationProcessExecution,
        )

        self.executions.append(copy.deepcopy(kwargs))
        return PublicationProcessExecution(
            process_exit_status="exited",
            process_exit_code=0,
            stdout=b"ok\n",
            stderr=b"",
        )

    def quiesce(self) -> tuple[int, int]:
        self.quiesce_count += 1
        return 0, 0

    def close_uncommitted(self) -> None:
        self.abort_count += 1


def _test_publication_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    _ProductionPublicationRuntime,
    list[_FakePublicationKernel],
    list[dict[str, object]],
]:
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher

    repo = tmp_path.resolve()
    (repo / ".git" / "objects").mkdir(parents=True)
    child_environment = {"exact": "child"}
    monkeypatch.setattr(
        publisher,
        "build_exact_child_environment",
        lambda **kwargs: dict(child_environment),
    )
    monkeypatch.setattr(
        production,
        "_publication_owner_process_identity",
        lambda: (4321, "0x1234"),
    )
    kernels: list[_FakePublicationKernel] = []
    pin_checks: list[dict[str, object]] = []

    def kernel_factory(**kwargs: str) -> _FakePublicationKernel:
        kernel = _FakePublicationKernel(**kwargs)
        kernels.append(kernel)
        return kernel

    def pin_verifier(value: object, **kwargs: object) -> None:
        assert kwargs == {
            "expected_dependency_closure_sha256": "9" * 64
        }
        pin_checks.append(copy.deepcopy(dict(value)))  # type: ignore[arg-type]

    class Store:
        material: dict[str, object] | None = None
        intent: dict[str, object] | None = None
        transport: dict[str, object] | None = None
        authorization: dict[str, object] | None = None

        @staticmethod
        def _authority(material: dict[str, object] | None) -> object:
            assert material is not None
            authority = type("Authority", (), {})()
            authority.material = copy.deepcopy(material)
            return authority

        def publication_worker_ownership_authority(
            self,
            attempt_id: str,
            ownership_sha256: str,
        ) -> object:
            assert self.material is not None
            assert self.material["attempt_id"] == attempt_id
            assert (
                self.material["worker_ownership_sha256"]
                == ownership_sha256
            )

            return self._authority(self.material)

        def publication_intent_authority(
            self, attempt_id: str
        ) -> object:
            assert self.intent is not None
            assert self.intent["attempt_id"] == attempt_id
            return self._authority(self.intent)

        def transport_manifest_authority(
            self,
            attempt_id: str,
            manifest_sha256: str,
        ) -> object:
            assert self.transport is not None
            assert self.transport["attempt_id"] == attempt_id
            assert (
                self.transport[
                    "isolated_transport_git_directory_manifest_sha256"
                ]
                == manifest_sha256
            )
            return self._authority(self.transport)

        def pre_push_authorization_authority(
            self,
            attempt_id: str,
            authorization_sha256: str,
        ) -> object:
            assert self.authorization is not None
            assert self.authorization["attempt_id"] == attempt_id
            assert (
                self.authorization["pre_push_authorization_sha256"]
                == authorization_sha256
            )
            return self._authority(self.authorization)

    runtime = _ProductionPublicationRuntime(
        repo_root=repo,
        implementation_manifest={
            "implementation_manifest_sha256": "a" * 64,
            "implementation_commit": "b" * 40,
        },
        store=Store(),
        executable_pins={
            "git_executable_path": (
                "C:/Program Files/Git/mingw64/bin/git.exe"
            ),
        },
        dependency_closure_sha256="9" * 64,
        host_environment_values={"host": "value"},
        pin_verifier=pin_verifier,
        kernel_factory=kernel_factory,
        restart_verifier=lambda owner: {
            "prior_owner_process_dead": True,
            "owner_mutex_unowned": True,
            "job_object_active_process_count": 0,
            "recorded_git_ssh_processes_alive_count": 0,
        },
        token_bytes=lambda size: b"x" * size,
        _sentinel=_PUBLICATION_RUNTIME_SENTINEL,
    )
    return runtime, kernels, pin_checks


def _run_test_publication_command(
    runtime: _ProductionPublicationRuntime,
    handle: object,
    *,
    push: bool = False,
    source_object: str = "f" * 40,
    cwd_name: str = "transport.git",
):
    import agent_benchmark.sec_gemma_online_risk_overlay_publisher as publisher

    context = handle._context
    ownership = handle.ownership_material
    tag_ref = (
        "refs/tags/sec-gemma-online-risk-overlay-v2-2/"
        f"attempts/{DEVELOPMENT_ACQUISITION_ID}/terminal"
    )
    intent = {
        "attempt_id": DEVELOPMENT_ACQUISITION_ID,
        "publication_intent_sha256": context[
            "publication_intent_sha256"
        ],
        "tag_ref": tag_ref,
        "expected_tag_object_sha1": "f" * 40,
    }
    transport_hash = "7" * 64
    transport = {
        "attempt_id": DEVELOPMENT_ACQUISITION_ID,
        "publication_intent_sha256": context[
            "publication_intent_sha256"
        ],
        "operation_kind": context["operation_kind"],
        "operation_sha256": context["operation_sha256"],
        "isolated_transport_git_directory_manifest_sha256": transport_hash,
    }
    runtime._store.intent = intent
    runtime._store.transport = transport
    authorization_hash: str | None = None
    if push:
        authorization_hash = "8" * 64
        runtime._store.authorization = {
            "attempt_id": DEVELOPMENT_ACQUISITION_ID,
            "publication_intent_sha256": context[
                "publication_intent_sha256"
            ],
            "authorization_operation_kind": context["operation_kind"],
            "authorization_operation_sha256": context["operation_sha256"],
            "worker_ownership_sha256": ownership[
                "worker_ownership_sha256"
            ],
            "tag_ref": tag_ref,
            "expected_tag_object_sha1": "f" * 40,
            "authorization_status": "push_authorized_once",
            "push_command_limit": 1,
            "pre_push_authorization_sha256": authorization_hash,
        }
    argv = (
        (
            "C:/Program Files/Git/mingw64/bin/git.exe",
            "push",
            "--porcelain",
            "--no-verify",
            publisher.ALLOWED_REMOTE_URL,
            f"{source_object}:{tag_ref}",
        )
        if push
        else (
            "C:/Program Files/Git/mingw64/bin/git.exe",
            "ls-remote",
            "--tags",
            publisher.ALLOWED_REMOTE_URL,
            tag_ref,
            f"{tag_ref}^{{}}",
        )
    )
    return publisher._execution_from_executor(
        runtime,
        argv=argv,
        cwd=handle.transport_root / cwd_name,
        environment={"exact": "child"},
        timeout_seconds=10.0,
        profile_kind="push" if push else "readback",
        transport_manifest_sha256=transport_hash,
        publication_intent_sha256=context[
            "publication_intent_sha256"
        ],
        operation_kind=context["operation_kind"],
        operation_sha256=context["operation_sha256"],
        worker_ownership_sha256=ownership["worker_ownership_sha256"],
        pre_push_authorization_sha256=authorization_hash,
        expected_tag_object_sha1="f" * 40,
        tag_ref=tag_ref,
    )


def test_publication_runtime_binds_owner_executes_and_quiesces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, kernels, pin_checks = _test_publication_runtime(
        tmp_path,
        monkeypatch,
    )
    implementation = {
        "implementation_manifest_sha256": "a" * 64,
        "implementation_commit": "b" * 40,
    }
    handle = runtime.begin(
        implementation_manifest=implementation,
        store_instance_id="c" * 64,
        store_session_nonce_sha256="d" * 64,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        publication_intent_sha256="e" * 64,
        operation_kind="normal_publication",
        operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
    )

    ownership = handle.ownership_material
    assert tuple(ownership) == PUBLICATION_WORKER_OWNERSHIP_FIELDS
    assert ownership["contract_version"] == CONTRACT_VERSION
    assert ownership["contract_sha256"] == CONTRACT_SHA256
    assert ownership["kill_on_parent_exit"] is True
    assert ownership["child_assignment_before_resume_required"] is True
    assert ownership["worker_ownership_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in ownership.items()
            if key != "worker_ownership_sha256"
        }
    )
    assert handle.source_object_directory == (
        tmp_path.resolve() / ".git" / "objects"
    )
    runtime._store.material = copy.deepcopy(ownership)
    result = _run_test_publication_command(
        runtime,
        handle,
    )

    assert result.process_exit_status == "exited"
    assert len(pin_checks) == 2
    assert len(kernels) == 1
    assert len(kernels[0].executions) == 1
    quiescence = handle.quiesce(
        worker_ownership_sha256=ownership[
            "worker_ownership_sha256"
        ]
    )
    assert tuple(quiescence) == PUBLICATION_WORKER_QUIESCENCE_FIELDS
    assert quiescence["verification_mode"] == (
        "same_session_clean_release"
    )
    assert quiescence["job_object_active_process_count"] == 0
    assert kernels[0].quiesce_count == 1
    assert runtime._active_handle is None


def test_publication_runtime_aborts_uncommitted_owner_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, kernels, _ = _test_publication_runtime(
        tmp_path,
        monkeypatch,
    )
    handle = runtime.begin(
        implementation_manifest={
            "implementation_manifest_sha256": "a" * 64,
            "implementation_commit": "b" * 40,
        },
        store_instance_id="c" * 64,
        store_session_nonce_sha256="d" * 64,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        publication_intent_sha256="e" * 64,
        operation_kind="normal_publication",
        operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
    )

    handle.abort_uncommitted()

    assert kernels[0].abort_count == 1
    assert kernels[0].executions == []
    assert runtime._active_handle is None


def test_publication_runtime_forbids_child_before_durable_owner_and_raw_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, kernels, _ = _test_publication_runtime(
        tmp_path,
        monkeypatch,
    )
    handle = runtime.begin(
        implementation_manifest={
            "implementation_manifest_sha256": "a" * 64,
            "implementation_commit": "b" * 40,
        },
        store_instance_id="c" * 64,
        store_session_nonce_sha256="d" * 64,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        publication_intent_sha256="e" * 64,
        operation_kind="normal_publication",
        operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="lacks exact durable command authorities",
    ):
        _run_test_publication_command(runtime, handle)
    runtime._store.material = handle.ownership_material
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="Raw publication process execution is forbidden",
    ):
        runtime(
            argv=(
                "C:/Program Files/Git/mingw64/bin/git.exe",
                "--version",
            ),
            cwd=handle.transport_root / "transport.git",
            env={"exact": "child"},
            timeout_seconds=10.0,
        )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="request changed",
    ):
        _run_test_publication_command(
            runtime,
            handle,
            push=True,
            source_object="a" * 40,
        )
    assert kernels[0].executions == []
    handle.abort_uncommitted()


def test_publication_runtime_consumes_exact_durable_push_authorization_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, kernels, _ = _test_publication_runtime(
        tmp_path,
        monkeypatch,
    )
    handle = runtime.begin(
        implementation_manifest={
            "implementation_manifest_sha256": "a" * 64,
            "implementation_commit": "b" * 40,
        },
        store_instance_id="c" * 64,
        store_session_nonce_sha256="d" * 64,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        publication_intent_sha256="e" * 64,
        operation_kind="normal_publication",
        operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
    )
    runtime._store.material = handle.ownership_material

    result = _run_test_publication_command(
        runtime,
        handle,
        push=True,
    )

    assert result.process_exit_status == "exited"
    assert len(kernels[0].executions) == 1
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="changed or was reused",
    ):
        _run_test_publication_command(
            runtime,
            handle,
            push=True,
        )
    assert len(kernels[0].executions) == 1
    handle.abort_uncommitted()


def test_publication_kernel_guards_cleanup_for_arbitrary_base_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Kernel32:
        terminate_calls = 0

        def TerminateJobObject(
            self,
            job: object,
            exit_code: int,
        ) -> bool:
            del job, exit_code
            self.terminate_calls += 1
            raise RuntimeError("cleanup job failure")

    class Process:
        pid = 123
        returncode = None

        def __init__(self) -> None:
            self.communicate_calls = 0
            self.kill_calls = 0
            self.close_calls = 0

        def communicate(self, timeout: float):
            del timeout
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise SystemExit("original failure")
            raise RuntimeError("cleanup communicate failure")

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.kill_calls += 1
            raise RuntimeError("cleanup kill failure")

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("cleanup close failure")

    kernel32 = Kernel32()
    process = Process()
    kernel = object.__new__(production._WindowsPublicationKernel)
    kernel._kernel32 = kernel32
    kernel._job = object()
    kernel._mutex = object()
    kernel._closed = False
    kernel._terminated = False
    kernel._recorded_processes = []
    monkeypatch.setattr(
        production,
        "_create_atomic_publication_process",
        lambda **kwargs: process,
    )
    monkeypatch.setattr(
        production._WindowsPublicationKernel,
        "_process_creation_filetime_hex",
        lambda self, child: "0x1",
    )

    with pytest.raises(SystemExit, match="original failure"):
        kernel.execute(
            argv=("C:/real/git.exe", "--version"),
            cwd=Path("C:/transport.git"),
            env={"exact": "child"},
            timeout_seconds=1.0,
        )

    assert kernel32.terminate_calls == 1
    assert process.communicate_calls == 2
    assert process.kill_calls == 1
    assert process.close_calls == 1
    assert kernel._terminated is True


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_publication_kernel_atomically_starts_child_inside_exact_job(
    tmp_path: Path,
) -> None:
    token = hashlib.sha256(
        f"membership:{time.time_ns()}".encode("ascii")
    ).hexdigest()
    job_name = f"Local\\CodexPublicationJob-{token}"
    mutex_name = f"Local\\CodexPublicationMutex-{token}"
    kernel = production._WindowsPublicationKernel(
        job_name=job_name,
        mutex_name=mutex_name,
    )
    script = (
        "import ctypes;"
        "from ctypes import wintypes;"
        "k=ctypes.WinDLL('kernel32',use_last_error=True);"
        "k.OpenJobObjectW.argtypes=[wintypes.DWORD,wintypes.BOOL,"
        "wintypes.LPCWSTR];"
        "k.OpenJobObjectW.restype=wintypes.HANDLE;"
        "k.GetCurrentProcess.restype=wintypes.HANDLE;"
        "k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,"
        "ctypes.POINTER(wintypes.BOOL)];"
        "k.IsProcessInJob.restype=wintypes.BOOL;"
        f"j=k.OpenJobObjectW(4,False,{job_name!r});"
        "inside=wintypes.BOOL();"
        "ok=bool(j) and bool(k.IsProcessInJob("
        "k.GetCurrentProcess(),j,ctypes.byref(inside)));"
        "print(int(ok and inside.value),flush=True)"
    )
    try:
        result = kernel.execute(
            argv=(sys.executable, "-c", script),
            cwd=tmp_path.resolve(),
            env=dict(os.environ),
            timeout_seconds=10.0,
        )
        assert result.process_exit_status == "exited"
        assert result.process_exit_code == 0
        assert result.stdout.strip() == b"1"
        assert kernel._active_process_count() == 0
        assert kernel.quiesce() == (0, 0)
    finally:
        if not kernel._closed:
            kernel.close_uncommitted()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_publication_kernel_timeout_kills_descendant_tree(
    tmp_path: Path,
) -> None:
    token = hashlib.sha256(
        f"timeout:{time.time_ns()}".encode("ascii")
    ).hexdigest()
    kernel = production._WindowsPublicationKernel(
        job_name=f"Local\\CodexPublicationJob-{token}",
        mutex_name=f"Local\\CodexPublicationMutex-{token}",
    )
    pid_file = (tmp_path / "descendant.pid").resolve()
    child_script = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;"
        "time.sleep(60)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    try:
        result = kernel.execute(
            argv=(sys.executable, "-c", child_script),
            cwd=tmp_path.resolve(),
            env=dict(os.environ),
            timeout_seconds=1.0,
        )
        assert result.process_exit_status == "deadline"
        assert pid_file.is_file()
        descendant_pid = int(pid_file.read_text(encoding="utf-8"))
        kernel32 = production.ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )
        kernel32.OpenProcess.argtypes = [
            production.wintypes.DWORD,
            production.wintypes.BOOL,
            production.wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = production.wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [
            production.wintypes.HANDLE,
            production.wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = production.wintypes.DWORD
        kernel32.CloseHandle.argtypes = [production.wintypes.HANDLE]
        kernel32.CloseHandle.restype = production.wintypes.BOOL
        descendant = kernel32.OpenProcess(
            0x00100000,
            False,
            descendant_pid,
        )
        if descendant:
            try:
                assert kernel32.WaitForSingleObject(descendant, 0) == (
                    production._WindowsPublicationKernel._WAIT_OBJECT_0
                )
            finally:
                kernel32.CloseHandle(descendant)
        assert kernel._active_process_count() == 0
        assert kernel.quiesce() == (0, 0)
    finally:
        if not kernel._closed:
            kernel.close_uncommitted()


def test_publication_runtime_builds_restart_only_quiescence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _, _ = _test_publication_runtime(tmp_path, monkeypatch)
    handle = runtime.begin(
        implementation_manifest={
            "implementation_manifest_sha256": "a" * 64,
            "implementation_commit": "b" * 40,
        },
        store_instance_id="c" * 64,
        store_session_nonce_sha256="d" * 64,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        publication_intent_sha256="e" * 64,
        operation_kind="normal_publication",
        operation_sha256=PUBLICATION_NORMAL_OPERATION_SHA256,
    )
    ownership = handle.ownership_material
    handle.abort_uncommitted()

    material = runtime.restarted_quiescence_material(
        ownership=ownership,
        current_store_session_nonce_sha256="f" * 64,
    )

    assert tuple(material) == PUBLICATION_WORKER_QUIESCENCE_FIELDS
    assert material["verification_mode"] == (
        "post_restart_prior_owner_dead"
    )
    assert material["store_session_nonce_sha256"] == "f" * 64
    assert material["worker_ownership_sha256"] == ownership[
        "worker_ownership_sha256"
    ]
    assert material["worker_quiescence_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in material.items()
            if key != "worker_quiescence_sha256"
        }
    )


def test_adapter_rehydrates_pending_report_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_benchmark.sec_gemma_online_risk_overlay_acquisition as acquisition

    report_payload = {
        "validation_sha256": "a" * 64,
    }

    class Report:
        def as_dict(self) -> dict[str, str]:
            return copy.deepcopy(report_payload)

    report = Report()

    class Execution:
        verified_report = report

    class Ledger:
        def __init__(self) -> None:
            self._executions: dict[str, object] = {}
            self.recorded: list[tuple[str, object]] = []

        def predecessors(self, stage: str) -> tuple[object, ...]:
            assert stage == "development"
            return ()

        def record(self, stage: str, execution: object) -> None:
            self._executions[stage] = execution
            self.recorded.append((stage, execution))

    class Store:
        def pending_acquisition_phase_evidence(
            self,
            attempt_id: str,
        ) -> dict[str, object]:
            assert attempt_id == DEVELOPMENT_ACQUISITION_ID
            return {
                "verified_acquisition_report": copy.deepcopy(
                    report_payload
                )
            }

    ledger = Ledger()
    adapter = object.__new__(_ProductionAcquisitionAdapter)
    adapter._store = Store()
    adapter._vault = object()
    adapter._ledger = ledger
    calls: list[dict[str, object]] = []

    def fake_rehydrate(**kwargs: object) -> Execution:
        calls.append(copy.deepcopy(kwargs))
        return Execution()

    monkeypatch.setattr(
        acquisition,
        "_rehydrate_one_production_acquisition",
        fake_rehydrate,
    )
    monkeypatch.setattr(
        acquisition,
        "is_verified_acquisition_report",
        lambda value: value is report,
    )

    observed = adapter.rehydrate_pending_acquisition_report(
        attempt_id=DEVELOPMENT_ACQUISITION_ID
    )

    assert observed is report
    assert len(calls) == 1
    assert len(ledger.recorded) == 1
    assert ledger.recorded[0][0] == "development"
    assert ledger.recorded[0][1] is ledger._executions["development"]


def _pending_publication_recovery_output() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": (
            "aapl-sec-gemma-online-risk-overlay-v2-2-"
            "publication-pending-result-v1"
        ),
        "contract_version": CONTRACT_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "command": "development_acquisition",
        "attempt_id": DEVELOPMENT_ACQUISITION_ID,
        "result_status": "publication_pending",
        "publication_intent_sha256": "a" * 64,
        "publication_intent_store_receipt_sha256": "b" * 64,
        "semantic_result_released": False,
        "next_stage_authority_blocked": True,
        "publication_recovery_required": True,
        "external_cost_usd": 0,
    }
    return {
        **body,
        "publication_pending_result_sha256": canonical_sha256(body),
    }


def test_public_recovery_clock_is_sampled_at_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []
    expected = _pending_publication_recovery_output()

    def fake_entry(**kwargs: object) -> dict[str, object]:
        captured.append(kwargs)
        return copy.deepcopy(expected)

    monkeypatch.setattr(production.time, "monotonic", lambda: 42.5)
    monkeypatch.setattr(
        production,
        "_recover_production_publication_from_entry",
        fake_entry,
    )

    observed = production.recover_production_publication(
        repo_root=Path(production.__file__).resolve().parents[1],
        implementation_manifest={},
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        sec_user_agent="Private private@example.com",
    )

    assert observed == expected
    assert len(captured) == 1
    assert captured[0]["invocation_started_at"] == 42.5


def test_supervised_recovery_authority_is_source_bound_and_one_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorities = object.__new__(VerifiedProductionAuthorities)
    authorities._sentinel = _AUTHORITIES_SENTINEL
    root = Path(production.__file__).resolve().parents[1]
    manifest = {"implementation_manifest_sha256": "a" * 64}
    start = 100.0
    deadline = start + 300.0
    monkeypatch.setattr(
        production,
        "_publication_owner_process_identity",
        lambda: (222, "0x333"),
    )
    monkeypatch.setattr(
        production,
        "_current_process_is_in_recovery_job",
        lambda name: name.endswith("c" * 64),
    )
    monkeypatch.setattr(
        production,
        "_windows_prior_process_is_dead",
        lambda process_id, creation: False,
    )
    monkeypatch.setattr(production.time, "monotonic", lambda: 101.0)
    authority = (
        production.VerifiedSupervisedPublicationRecoveryWorkerAuthority(
            repo_root=root,
            implementation_manifest=manifest,
            production_authorities=authorities,
            attempt_id=DEVELOPMENT_ACQUISITION_ID,
            invocation_nonce_sha256="c" * 64,
            job_name=(
                "Local\\CodexPublicationRecoveryJob-" + "c" * 64
            ),
            supervisor_started_at=start,
            supervisor_deadline=deadline,
            supervisor_process_id=111,
            supervisor_process_creation_filetime_hex="0x222",
            worker_process_id=222,
            worker_process_creation_filetime_hex="0x333",
            _sentinel=production._RECOVERY_WORKER_AUTHORITY_SENTINEL,
        )
    )

    assert (
        authority.authorize_runner_recovery(
            repo_root=root,
            implementation_manifest=manifest,
            production_authorities=authorities,
            attempt_id=DEVELOPMENT_ACQUISITION_ID,
            worker_entry_monotonic=101.0,
            worker_deadline_monotonic=401.0,
        )
        == (start, deadline)
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="expired, reused",
    ):
        authority.authorize_runner_recovery(
            repo_root=root,
            implementation_manifest=manifest,
            production_authorities=authorities,
            attempt_id=DEVELOPMENT_ACQUISITION_ID,
            worker_entry_monotonic=101.0,
            worker_deadline_monotonic=401.0,
        )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="cannot be forged",
    ):
        production.VerifiedSupervisedPublicationRecoveryWorkerAuthority(
            repo_root=root,
            implementation_manifest=manifest,
            production_authorities=authorities,
            attempt_id=DEVELOPMENT_ACQUISITION_ID,
            invocation_nonce_sha256="c" * 64,
            job_name=(
                "Local\\CodexPublicationRecoveryJob-" + "c" * 64
            ),
            supervisor_started_at=start,
            supervisor_deadline=deadline,
            supervisor_process_id=111,
            supervisor_process_creation_filetime_hex="0x222",
            worker_process_id=222,
            worker_process_creation_filetime_hex="0x333",
            _sentinel=object(),
        )


def test_recovery_supervisor_uses_private_stdin_and_accepts_only_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = time.monotonic()
    deadline = start + 300.0
    root = Path(production.__file__).resolve().parents[1]
    request = production._build_publication_recovery_worker_request(
        repo_root=root,
        implementation_manifest={},
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        sec_user_agent="Private private@example.com",
        supervisor_started_at=start,
        supervisor_deadline=deadline,
    )
    pending = _pending_publication_recovery_output()
    encoded_output = canonical_json_bytes(
        {"status": "ok", "payload": pending}
    )
    calls: list[dict[str, object]] = []

    class FakeSupervisor:
        def __init__(self, *, job_name: str) -> None:
            self.job_name = job_name
            self.closed = False

        def run(self, **kwargs: object) -> bytes:
            calls.append(kwargs)
            return encoded_output

        def close(self) -> None:
            self.closed = True

    created: list[FakeSupervisor] = []

    def factory(*, job_name: str) -> FakeSupervisor:
        supervisor = FakeSupervisor(job_name=job_name)
        created.append(supervisor)
        return supervisor

    observed = production._supervise_publication_recovery_worker(
        request,
        supervisor_deadline=deadline,
        supervisor_factory=factory,
    )

    assert observed == pending
    assert len(calls) == 1
    assert created[0].closed is True
    assert calls[0]["deadline_monotonic"] == deadline
    assert calls[0]["argv"][-1] == "--publication-recovery-worker"
    assert b"private@example.com" in calls[0]["input_bytes"]
    assert "private@example.com" not in repr(calls[0]["argv"])
    assert "private@example.com" not in repr(calls[0]["env"])

    changed = copy.deepcopy(pending)
    changed["semantic_result_released"] = True
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="invalid pending result",
    ):
        production._decode_publication_recovery_worker_output(
            canonical_json_bytes(
                {"status": "ok", "payload": changed}
            ),
            attempt_id=DEVELOPMENT_ACQUISITION_ID,
        )


def test_recovery_request_and_output_require_exact_canonical_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {"value": "private"}
    noncanonical_request = json.dumps(
        request,
        indent=2,
    ).encode("utf-8")
    monkeypatch.setattr(
        production.sys,
        "stdin",
        type(
            "PrivateStdin",
            (),
            {"buffer": io.BytesIO(noncanonical_request)},
        )(),
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="not canonical JSON",
    ):
        production._read_stdio_worker_request(
            require_canonical=True
        )

    pending = _pending_publication_recovery_output()
    noncanonical_output = json.dumps(
        {"status": "ok", "payload": pending},
        indent=2,
    ).encode("utf-8")
    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="not canonical JSON",
    ):
        production._decode_publication_recovery_worker_output(
            noncanonical_output,
            attempt_id=DEVELOPMENT_ACQUISITION_ID,
        )


def test_recovery_decoder_accepts_canonical_envelope_just_over_16_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_payload = {"padding": "x" * (16 * 1024 * 1024)}
    encoded = canonical_json_bytes(
        {"status": "ok", "payload": worker_payload}
    )
    assert 16 * 1024 * 1024 < len(encoded)
    assert len(encoded) < production._STDIO_WORKER_MAX_RESPONSE_BYTES
    expected = {"accepted": True}
    observed_validation: list[tuple[dict[str, object], str]] = []

    def validate(
        payload: object,
        *,
        attempt_id: str,
    ) -> dict[str, bool]:
        assert type(payload) is dict
        observed_validation.append((payload, attempt_id))
        return expected

    monkeypatch.setattr(
        production,
        "_validate_publication_recovery_output",
        validate,
    )

    assert production._decode_publication_recovery_worker_output(
        encoded,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
    ) is expected
    assert observed_validation == [
        (worker_payload, DEVELOPMENT_ACQUISITION_ID)
    ]


def test_recovery_decoder_uses_named_worker_response_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = canonical_json_bytes(
        {"status": "ok", "payload": {"small": True}}
    )
    monkeypatch.setattr(
        production,
        "_STDIO_WORKER_MAX_RESPONSE_BYTES",
        len(encoded) - 1,
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="absent or oversized",
    ):
        production._decode_publication_recovery_worker_output(
            encoded,
            attempt_id=DEVELOPMENT_ACQUISITION_ID,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows recovery environment")
def test_recovery_private_request_restores_exact_git_host_environment(
    tmp_path: Path,
) -> None:
    host_environment = (
        production._publication_recovery_host_environment()
    )
    sanitized = {
        "SYSTEMROOT": host_environment["SystemRoot"],
        "WINDIR": host_environment["WINDIR"],
    }
    script = (
        "import json,sys;"
        "from pathlib import Path;"
        "from agent_benchmark.sec_gemma_online_risk_overlay_production "
        "import _install_publication_recovery_host_environment as install;"
        "install(json.loads(sys.stdin.buffer.read().decode('utf-8')));"
        "print(Path.home(),flush=True)"
    )

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=Path(production.__file__).resolve().parents[1],
        env=sanitized,
        input=canonical_json_bytes(host_environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(
        completed.stdout.decode("utf-8").strip()
    ).resolve(strict=True) == Path(
        host_environment["USERPROFILE"]
    ).resolve(strict=True)


def test_recovery_supervisor_timeout_terminates_outer_job_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Kernel32:
        def __init__(self) -> None:
            self.terminate_calls: list[tuple[object, int]] = []

        def TerminateJobObject(
            self,
            job: object,
            exit_code: int,
        ) -> bool:
            self.terminate_calls.append((job, exit_code))
            return True

        def CloseHandle(self, job: object) -> bool:
            del job
            return True

    class Process:
        returncode = None

        def __init__(self) -> None:
            self.closed = False

        def communicate(self, timeout: float) -> tuple[bytes, bytes]:
            raise subprocess.TimeoutExpired("recovery-child", timeout)

        def close(self) -> None:
            self.closed = True

    kernel32 = Kernel32()
    process = Process()
    supervisor = object.__new__(
        production._WindowsPublicationRecoverySupervisor
    )
    supervisor._kernel32 = kernel32
    supervisor._job = object()
    supervisor._job_name = (
        "Local\\CodexPublicationRecoveryJob-" + "d" * 64
    )
    supervisor._closed = False
    supervisor._terminated = False
    calls: list[dict[str, object]] = []

    def create(**kwargs: object) -> Process:
        calls.append(kwargs)
        return process

    monkeypatch.setattr(
        production,
        "_create_atomic_publication_process",
        create,
    )
    active_counts = [1, 0, 0]
    monkeypatch.setattr(
        production._WindowsPublicationRecoverySupervisor,
        "_active_process_count",
        lambda self: active_counts.pop(0),
    )
    monkeypatch.setattr(
        production,
        "_windows_job_process_ids",
        lambda **kwargs: (),
    )
    active_counts.extend((0, 0, 0))
    monkeypatch.setattr(production.time, "monotonic", lambda: 1.0)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="300-second deadline",
    ):
        supervisor.run(
            argv=("C:/Python/python.exe", "-c", "pass"),
            cwd=Path("C:/repo"),
            env={},
            input_bytes=b'{"private":"request"}',
            deadline_monotonic=301.0,
        )

    assert calls[0]["input_bytes"] == b'{"private":"request"}'
    assert kernel32.terminate_calls == [(supervisor._job, 124)]
    assert supervisor._terminated is True
    assert process.closed is True
    supervisor.close()
    assert supervisor._closed is True
    assert active_counts == []


def test_recovery_cleanup_failure_is_authoritative_over_worker_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Kernel32:
        def TerminateJobObject(
            self,
            job: object,
            exit_code: int,
        ) -> bool:
            del job, exit_code
            return True

    class Process:
        returncode = None

        def communicate(self, timeout: float) -> tuple[bytes, bytes]:
            raise subprocess.TimeoutExpired("recovery-child", timeout)

        def close(self) -> None:
            return None

    supervisor = object.__new__(
        production._WindowsPublicationRecoverySupervisor
    )
    supervisor._kernel32 = Kernel32()
    supervisor._job = object()
    supervisor._job_name = (
        "Local\\CodexPublicationRecoveryJob-" + "e" * 64
    )
    supervisor._cleanup_deadline = None
    supervisor._closed = False
    supervisor._terminated = False
    monkeypatch.setattr(
        production,
        "_create_atomic_publication_process",
        lambda **kwargs: Process(),
    )
    monkeypatch.setattr(
        production._WindowsPublicationRecoverySupervisor,
        "_active_process_count",
        lambda self: 1,
    )
    observed_times = iter((1.0, 1.0, 301.0))
    monkeypatch.setattr(
        production.time,
        "monotonic",
        lambda: next(observed_times),
    )

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="did not become quiescent",
    ) as failure:
        supervisor.run(
            argv=("C:/Python/python.exe", "-c", "pass"),
            cwd=Path("C:/repo"),
            env={},
            input_bytes=b'{"private":"request"}',
            deadline_monotonic=301.0,
        )

    assert failure.value.__cause__ is not None
    assert "300-second deadline" in str(failure.value.__cause__)


def test_recovery_zero_active_must_be_observed_before_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = object.__new__(
        production._WindowsPublicationRecoverySupervisor
    )
    supervisor._job = object()
    monkeypatch.setattr(
        production._WindowsPublicationRecoverySupervisor,
        "_active_process_count",
        lambda self: 0,
    )
    monkeypatch.setattr(production.time, "monotonic", lambda: 301.0)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayProductionError,
        match="did not become quiescent",
    ):
        supervisor._wait_for_zero_active(deadline_monotonic=301.0)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_atomic_recovery_child_reads_exact_private_stdin(
    tmp_path: Path,
) -> None:
    token = hashlib.sha256(
        f"recovery-stdin:{time.time_ns()}".encode("ascii")
    ).hexdigest()
    supervisor = production._WindowsPublicationRecoverySupervisor(
        job_name=(
            "Local\\CodexPublicationRecoveryJob-" + token
        )
    )
    payload = b'{"private":"recovery request"}'
    script = (
        "import hashlib,sys;"
        "data=sys.stdin.buffer.read();"
        "print(hashlib.sha256(data).hexdigest(),flush=True)"
    )
    process = None
    try:
        process = production._create_atomic_publication_process(
            kernel32=supervisor._kernel32,
            job_handle=supervisor._job,
            argv=(sys.executable, "-c", script),
            cwd=tmp_path.resolve(),
            env=dict(os.environ),
            input_bytes=payload,
        )
        stdout, stderr = process.communicate(timeout=10.0)
        assert process.returncode == 0
        assert stderr == b""
        assert stdout.strip() == hashlib.sha256(payload).hexdigest().encode(
            "ascii"
        )
        accounting_deadline = time.monotonic() + 1.0
        while (
            supervisor._active_process_count()
            and time.monotonic() < accounting_deadline
        ):
            time.sleep(0.01)
        assert supervisor._active_process_count() == 0
    finally:
        if process is not None:
            process.close()
        supervisor.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_recovery_supervisor_bounds_stdout_while_child_is_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token = hashlib.sha256(
        f"recovery-output-cap:{time.time_ns()}".encode("ascii")
    ).hexdigest()
    supervisor = production._WindowsPublicationRecoverySupervisor(
        job_name=(
            "Local\\CodexPublicationRecoveryJob-" + token
        )
    )
    monkeypatch.setattr(
        production,
        "_STDIO_WORKER_MAX_RESPONSE_BYTES",
        64 * 1024,
    )
    script = (
        "import sys,time;"
        "sys.stdout.buffer.write(b'x'*(128*1024));"
        "sys.stdout.buffer.flush();"
        "time.sleep(60)"
    )
    closed = False
    try:
        with pytest.raises(
            SecGemmaOnlineRiskOverlayProductionError,
            match="stdout exceeded its fixed byte cap",
        ):
            supervisor.run(
                argv=(sys.executable, "-c", script),
                cwd=tmp_path.resolve(),
                env=dict(os.environ),
                input_bytes=b'{"private":"request"}',
                deadline_monotonic=time.monotonic() + 10.0,
            )
        assert supervisor._active_process_count() == 0
        supervisor.close()
        closed = True
    finally:
        if not closed:
            supervisor.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_recovery_supervisor_real_timeout_kills_descendant_tree(
    tmp_path: Path,
) -> None:
    token = hashlib.sha256(
        f"recovery-timeout:{time.time_ns()}".encode("ascii")
    ).hexdigest()
    supervisor = production._WindowsPublicationRecoverySupervisor(
        job_name=(
            "Local\\CodexPublicationRecoveryJob-" + token
        )
    )
    pid_file = (tmp_path / "recovery-descendant.pid").resolve()
    script = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;"
        "time.sleep(60)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )
    closed = False
    try:
        with pytest.raises(
            SecGemmaOnlineRiskOverlayProductionError,
            match="300-second deadline",
        ):
            supervisor.run(
                argv=(sys.executable, "-c", script),
                cwd=tmp_path.resolve(),
                env=dict(os.environ),
                input_bytes=b'{"private":"request"}',
                deadline_monotonic=time.monotonic() + 6.0,
            )
        assert supervisor._active_process_count() == 0
        supervisor.close()
        closed = True
        assert pid_file.is_file()
        descendant_pid = int(pid_file.read_text(encoding="utf-8"))
        kernel32 = production.ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )
        kernel32.OpenProcess.argtypes = [
            production.wintypes.DWORD,
            production.wintypes.BOOL,
            production.wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = production.wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [
            production.wintypes.HANDLE,
            production.wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = production.wintypes.DWORD
        kernel32.CloseHandle.argtypes = [production.wintypes.HANDLE]
        kernel32.CloseHandle.restype = production.wintypes.BOOL
        descendant = kernel32.OpenProcess(
            0x00100000,
            False,
            descendant_pid,
        )
        if descendant:
            try:
                wait_deadline = time.monotonic() + 2.0
                wait = kernel32.WaitForSingleObject(descendant, 0)
                while (
                    wait
                    != production._WindowsPublicationKernel._WAIT_OBJECT_0
                    and time.monotonic() < wait_deadline
                ):
                    time.sleep(0.01)
                    wait = kernel32.WaitForSingleObject(descendant, 0)
                assert wait == (
                    production._WindowsPublicationKernel._WAIT_OBJECT_0
                )
            finally:
                kernel32.CloseHandle(descendant)
    finally:
        if not closed:
            supervisor.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows nested Job contract")
def test_recovery_outer_job_kills_nested_publication_job_tree(
    tmp_path: Path,
) -> None:
    token = hashlib.sha256(
        f"nested-recovery:{time.time_ns()}".encode("ascii")
    ).hexdigest()
    supervisor = production._WindowsPublicationRecoverySupervisor(
        job_name=(
            "Local\\CodexPublicationRecoveryJob-" + token
        )
    )
    pid_file = (tmp_path / "nested-publication.pid").resolve()
    worker_script = (
        "import hashlib,os,sys,time;"
        "from pathlib import Path;"
        "from agent_benchmark.sec_gemma_online_risk_overlay_production "
        "import _WindowsPublicationKernel;"
        "token=hashlib.sha256(str(time.time_ns()).encode()).hexdigest();"
        "kernel=_WindowsPublicationKernel("
        "job_name='Local\\\\NestedPublicationJob-'+token,"
        "mutex_name='Local\\\\NestedPublicationMutex-'+token);"
        f"child=\"import os,time;from pathlib import Path;"
        f"Path({pid_file.as_posix()!r}).write_text(str(os.getpid()));"
        "time.sleep(60)\";"
        "kernel.execute(argv=(sys.executable,'-c',child),"
        "cwd=Path.cwd(),env=dict(os.environ),timeout_seconds=60.0)"
    )
    try:
        with pytest.raises(
            SecGemmaOnlineRiskOverlayProductionError,
            match="300-second deadline",
        ):
            supervisor.run(
                argv=(sys.executable, "-c", worker_script),
                cwd=Path(production.__file__).resolve().parents[1],
                env=dict(os.environ),
                input_bytes=b'{"private":"request"}',
                deadline_monotonic=time.monotonic() + 6.0,
            )
        supervisor.close()
        assert pid_file.is_file()
        nested_pid = int(pid_file.read_text(encoding="utf-8"))
        kernel32 = production.ctypes.WinDLL(
            "kernel32",
            use_last_error=True,
        )
        kernel32.OpenProcess.argtypes = [
            production.wintypes.DWORD,
            production.wintypes.BOOL,
            production.wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = production.wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [
            production.wintypes.HANDLE,
            production.wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = production.wintypes.DWORD
        kernel32.CloseHandle.argtypes = [production.wintypes.HANDLE]
        kernel32.CloseHandle.restype = production.wintypes.BOOL
        nested = kernel32.OpenProcess(
            0x00100000,
            False,
            nested_pid,
        )
        if nested:
            try:
                assert kernel32.WaitForSingleObject(nested, 0) == (
                    production._WindowsPublicationKernel._WAIT_OBJECT_0
                )
            finally:
                kernel32.CloseHandle(nested)
    finally:
        if not supervisor._closed:
            supervisor.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_final_registry_atomic_runner_round_trips_private_input(
    tmp_path: Path,
) -> None:
    runner = production._WindowsFinalRegistrySubprocessRunner()
    payload = b"private-final-registry-input"
    script = (
        "import hashlib,sys;"
        "data=sys.stdin.buffer.read();"
        "print(hashlib.sha256(data).hexdigest(),flush=True)"
    )

    completed = runner(
        (sys.executable, "-c", script),
        cwd=tmp_path.resolve(),
        env=dict(os.environ),
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10.0,
    )

    assert completed.returncode == 0
    assert completed.stderr == b""
    assert completed.stdout.strip() == hashlib.sha256(payload).hexdigest().encode(
        "ascii"
    )


def test_job_process_id_query_retries_successful_truncated_list() -> None:
    class Kernel32:
        calls = 0

        def QueryInformationJobObject(
            self,
            job: object,
            information_class: int,
            buffer: object,
            buffer_size: int,
            returned: object,
        ) -> bool:
            del job
            assert information_class == (
                production._WINDOWS_JOB_OBJECT_BASIC_PROCESS_ID_LIST
            )
            self.calls += 1
            listed_ids = (101,) if self.calls == 1 else (101, 202)
            header = production.ctypes.cast(
                buffer,
                production.ctypes.POINTER(
                    production._JobObjectBasicProcessIdList
                ),
            ).contents
            header.NumberOfAssignedProcesses = 2
            header.NumberOfProcessIdsInList = len(listed_ids)
            ids = (
                production.ctypes.c_size_t * len(listed_ids)
            ).from_address(
                production.ctypes.addressof(buffer)
                + production._JobObjectBasicProcessIdList.ProcessIdList.offset
            )
            for index, process_id in enumerate(listed_ids):
                ids[index] = process_id
            returned._obj.value = min(
                buffer_size,
                production._JobObjectBasicProcessIdList.ProcessIdList.offset
                + (
                    len(listed_ids)
                    * production.ctypes.sizeof(production.ctypes.c_size_t)
                ),
            )
            return True

    kernel32 = Kernel32()

    assert production._windows_job_process_ids(
        kernel32=kernel32,
        job=object(),
    ) == (101, 202)
    assert kernel32.calls == 2


def test_final_registry_waits_for_late_child_process_object_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Kernel32:
        def __init__(self) -> None:
            self.terminate_calls: list[tuple[object, int]] = []
            self.open_calls: list[int] = []
            self.wait_calls: list[int] = []
            self.closed_handles: list[int] = []
            self.late_child_waits = 0

        def OpenProcess(
            self,
            access: int,
            inherit: bool,
            process_id: int,
        ) -> int:
            assert access == (
                production._WINDOWS_SYNCHRONIZE
                | production._WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION
            )
            assert inherit is False
            self.open_calls.append(process_id)
            return process_id + 10_000

        @staticmethod
        def IsProcessInJob(
            handle: int,
            job: object,
            inside: object,
        ) -> bool:
            del handle, job
            inside._obj.value = True
            return True

        def TerminateJobObject(
            self,
            job: object,
            exit_code: int,
        ) -> bool:
            self.terminate_calls.append((job, exit_code))
            return True

        def WaitForSingleObject(
            self,
            handle: int,
            milliseconds: int,
        ) -> int:
            assert milliseconds == 0
            self.wait_calls.append(handle)
            if handle == 10_202:
                self.late_child_waits += 1
                if self.late_child_waits == 1:
                    return production._WindowsPublicationKernel._WAIT_TIMEOUT
            return production._WindowsPublicationKernel._WAIT_OBJECT_0

        def CloseHandle(self, handle: int) -> bool:
            self.closed_handles.append(handle)
            return True

    kernel32 = Kernel32()
    runner = object.__new__(
        production._WindowsFinalRegistrySubprocessRunner
    )
    runner._kernel32 = kernel32
    job = object()
    snapshots = iter(
        (
            (101,),
            (101, 202),
            (101, 202),
            (101, 202),
            (101, 202),
        )
    )

    def process_ids(**kwargs: object) -> tuple[int, ...]:
        assert kwargs == {"kernel32": kernel32, "job": job}
        return next(snapshots)

    active_counts = iter((0, 0, 0, 0))
    monkeypatch.setattr(
        production,
        "_windows_job_process_ids",
        process_ids,
    )
    monkeypatch.setattr(
        production._WindowsFinalRegistrySubprocessRunner,
        "_active_process_count",
        lambda self, observed_job: next(active_counts),
    )
    monkeypatch.setattr(production.time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(production.time, "sleep", lambda seconds: None)

    runner._terminate_and_wait(
        job,
        exit_code=124,
        deadline_monotonic=10.0,
    )

    assert kernel32.terminate_calls == [(job, 124)]
    assert kernel32.open_calls == [101, 202]
    assert kernel32.late_child_waits == 2
    assert kernel32.closed_handles == [10_101, 10_202]


def test_final_registry_closes_process_witness_on_termination_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Kernel32:
        closed_handles: list[int] = []

        @staticmethod
        def OpenProcess(
            access: int,
            inherit: bool,
            process_id: int,
        ) -> int:
            del access, inherit
            return process_id + 20_000

        @staticmethod
        def IsProcessInJob(
            handle: int,
            job: object,
            inside: object,
        ) -> bool:
            del handle, job
            inside._obj.value = True
            return True

        @staticmethod
        def TerminateJobObject(job: object, exit_code: int) -> bool:
            del job, exit_code
            return False

        def CloseHandle(self, handle: int) -> bool:
            self.closed_handles.append(handle)
            return True

    kernel32 = Kernel32()
    runner = object.__new__(
        production._WindowsFinalRegistrySubprocessRunner
    )
    runner._kernel32 = kernel32
    monkeypatch.setattr(
        production,
        "_windows_job_process_ids",
        lambda **kwargs: (303,),
    )
    monkeypatch.setattr(production.time, "monotonic", lambda: 1.0)

    with pytest.raises(
        OSError,
        match="could not be terminated",
    ):
        runner._terminate_and_wait(
            object(),
            exit_code=125,
            deadline_monotonic=10.0,
        )

    assert kernel32.closed_handles == [20_303]


def test_final_registry_zero_active_must_be_observed_before_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Kernel32:
        @staticmethod
        def TerminateJobObject(job: object, exit_code: int) -> bool:
            del job, exit_code
            return True

    runner = object.__new__(
        production._WindowsFinalRegistrySubprocessRunner
    )
    runner._kernel32 = Kernel32()
    monkeypatch.setattr(
        production._WindowsFinalRegistrySubprocessRunner,
        "_active_process_count",
        lambda self, job: 0,
    )
    monkeypatch.setattr(production.time, "monotonic", lambda: 10.0)

    with pytest.raises(
        OSError,
        match="did not become quiescent",
    ):
        runner._terminate_and_wait(
            object(),
            exit_code=125,
            deadline_monotonic=10.0,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_final_registry_atomic_runner_bounds_stdout_while_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = production._WindowsFinalRegistrySubprocessRunner()
    monkeypatch.setattr(
        production,
        "_FINAL_REGISTRY_MAX_COMMAND_IO_BYTES",
        64 * 1024,
    )
    script = (
        "import sys,time;"
        "sys.stdout.buffer.write(b'x'*(128*1024));"
        "sys.stdout.buffer.flush();"
        "time.sleep(60)"
    )

    with pytest.raises(
        OSError,
        match="output is malformed or oversized",
    ) as failure:
        runner(
            (sys.executable, "-c", script),
            cwd=tmp_path.resolve(),
            env=dict(os.environ),
            input=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
        )
    assert isinstance(
        failure.value.__cause__,
        SecGemmaOnlineRiskOverlayProductionError,
    )
    assert "stdout exceeded its fixed byte cap" in str(
        failure.value.__cause__
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
@pytest.mark.parametrize("_repetition", range(5))
def test_final_registry_atomic_runner_timeout_kills_descendant_tree(
    tmp_path: Path,
    _repetition: int,
) -> None:
    runner = production._WindowsFinalRegistrySubprocessRunner()
    pid_file = (
        tmp_path / f"final-registry-descendant-{_repetition}.pid"
    ).resolve()
    script = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time;"
        "time.sleep(60)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));"
        "time.sleep(60)"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        runner(
            (sys.executable, "-c", script),
            cwd=tmp_path.resolve(),
            env=dict(os.environ),
            input=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=6.0,
        )

    assert pid_file.is_file()
    descendant_pid = int(pid_file.read_text(encoding="utf-8"))
    kernel32 = production.ctypes.WinDLL(
        "kernel32",
        use_last_error=True,
    )
    kernel32.OpenProcess.argtypes = [
        production.wintypes.DWORD,
        production.wintypes.BOOL,
        production.wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = production.wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [
        production.wintypes.HANDLE,
        production.wintypes.DWORD,
    ]
    kernel32.WaitForSingleObject.restype = production.wintypes.DWORD
    kernel32.CloseHandle.argtypes = [production.wintypes.HANDLE]
    kernel32.CloseHandle.restype = production.wintypes.BOOL
    descendant = kernel32.OpenProcess(
        0x00100000,
        False,
        descendant_pid,
    )
    if descendant:
        try:
            assert kernel32.WaitForSingleObject(descendant, 0) == (
                production._WindowsPublicationKernel._WAIT_OBJECT_0
            )
        finally:
            kernel32.CloseHandle(descendant)
