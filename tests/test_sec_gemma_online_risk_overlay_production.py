from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import math
import subprocess
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
    MAX_DETERMINISTIC_SECONDS,
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
    _ProductionPhaseExecutor,
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
