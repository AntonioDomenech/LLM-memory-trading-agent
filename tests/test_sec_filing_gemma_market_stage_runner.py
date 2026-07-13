from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

import agent_benchmark.sec_filing_gemma_stage_runner as runner
from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_market_evidence import MARKET_SYMBOLS
from agent_benchmark.sec_filing_gemma_reveal_store import (
    DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME,
    DEVELOPMENT_MARKET_COMPLETE_MARKER_SCHEMA_VERSION,
    DEVELOPMENT_MARKET_COMPONENT_ID,
    MARKET_SOURCE_COMPONENT_DIRECTORY_NAME,
    STAGE_OUTPUTS_DIRECTORY_NAME,
    SecFilingGemmaRevealStore,
)


SCOPE_SHA256 = "1" * 64
PLAN_SHA256 = "2" * 64
VALIDATION_SHA256 = "3" * 64


def _plan() -> dict[str, Any]:
    body = {"schema_version": "synthetic-market-plan-v1", "fixed": True}
    return {**body, "acquisition_plan_sha256": canonical_sha256(body)}


def _claim() -> dict[str, Any]:
    plan = _plan()
    body = {
        "development_root_scope_sha256": SCOPE_SHA256,
        "authorized_stage": "development",
        "market_component_id": DEVELOPMENT_MARKET_COMPONENT_ID,
        "market_acquisition_plan": plan,
        "market_acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "market_symbols": list(MARKET_SYMBOLS),
        "fixed_request_count": len(MARKET_SYMBOLS),
        "owned_market_execution_required": True,
        "market_access_permitted": True,
        "external_network_access_permitted": True,
        "caller_supplied_path_permitted": False,
        "caller_supplied_bytes_permitted": False,
        "paid_api_access_permitted": False,
        "outcome_access_permitted": False,
        "future_stage_access_permitted": False,
        "effect_may_be_repeated_after_indeterminate_crash": False,
    }
    return {**body, "claim_sha256": canonical_sha256(body)}


def _receipt(claim: dict[str, Any]) -> dict[str, Any]:
    body = {
        "development_root_scope_sha256": SCOPE_SHA256,
        "claim_sha256": claim["claim_sha256"],
        "receipt_kind": "synthetic-market-reader",
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _bundle(claim: dict[str, Any]) -> dict[str, Any]:
    raw = {
        symbol: json.dumps({"raw": symbol}).encode("utf-8")
        for symbol in MARKET_SYMBOLS
    }
    artifacts = {
        symbol: json.dumps({"artifact": symbol}).encode("utf-8")
        for symbol in MARKET_SYMBOLS
    }
    windows = {
        symbol: json.dumps({"window": symbol}).encode("utf-8")
        for symbol in MARKET_SYMBOLS
    }
    acquisition_receipt_sha256 = "4" * 64
    return {
        "acquisition_plan_sha256": claim["market_acquisition_plan_sha256"],
        "acquisition_receipt_sha256": acquisition_receipt_sha256,
        "bundle_sha256": "5" * 64,
        "source_manifest_sha256": "6" * 64,
        "market_stage_manifest_sha256": "7" * 64,
        "source_reconciliation_sha256": "8" * 64,
        "raw_response_bytes_by_symbol": raw,
        "artifact_bytes_by_symbol": artifacts,
        "window_bytes_by_symbol": windows,
        "source_manifest": {"kind": "source-manifest"},
        "stage_manifest": {"kind": "stage-manifest"},
        "reconciliation_receipt": {"kind": "reconciliation-receipt"},
        "acquisition_receipt": {
            "kind": "acquisition-receipt",
            "acquisition_receipt_sha256": acquisition_receipt_sha256,
        },
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


def _install_lock(store: SecFilingGemmaRevealStore, events: list[str]) -> None:
    @contextmanager
    def market_lock() -> Iterator[None]:
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")

    store._owned_market_execution_lock = market_lock


def _install_claim(
    store: SecFilingGemmaRevealStore,
    *,
    claim: dict[str, Any],
    created: bool,
    reader_receipt: dict[str, Any] | None = None,
    abort: dict[str, Any] | None = None,
) -> None:
    def claim_execution(*, development_root_scope_sha256: str) -> dict[str, Any]:
        assert development_root_scope_sha256 == SCOPE_SHA256
        return {
            "claim": claim,
            "created": created,
            "reader_receipt": reader_receipt,
            "abort": abort,
        }

    store.claim_owned_development_market_execution = claim_execution


def _install_successful_effect(
    monkeypatch: pytest.MonkeyPatch,
    *,
    claim: dict[str, Any],
    calls: list[str],
) -> tuple[dict[str, Any], object]:
    bundle = _bundle(claim)
    owned_acquisition = object()
    owned_transport_capability = object()

    def acquire() -> object:
        calls.append("acquire")
        return owned_acquisition

    def unwrap(value: object) -> tuple[dict[str, Any], object]:
        assert value is owned_acquisition
        return bundle, owned_transport_capability

    def bind(
        value: object,
        *,
        development_root_scope_sha256: str,
        claim_sha256: str,
    ) -> None:
        assert value is owned_transport_capability
        assert development_root_scope_sha256 == SCOPE_SHA256
        assert claim_sha256 == claim["claim_sha256"]

    def validate(value: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append("validate")
        assert value is bundle
        assert kwargs == {
            "expected_acquisition_plan_sha256": claim[
                "market_acquisition_plan_sha256"
            ],
            "expected_acquisition_receipt_sha256": bundle[
                "acquisition_receipt_sha256"
            ],
            "expected_bundle_sha256": bundle["bundle_sha256"],
        }
        return {"validation_sha256": VALIDATION_SHA256}

    monkeypatch.setattr(
        runner,
        "_acquire_owned_development_market_evidence",
        acquire,
    )
    monkeypatch.setattr(
        runner,
        "_unwrap_owned_development_market_acquisition",
        unwrap,
    )
    monkeypatch.setattr(
        runner,
        "_bind_owned_market_transport_capability_to_claim",
        bind,
    )
    monkeypatch.setattr(
        runner,
        "validate_development_market_acquisition_bundle",
        validate,
    )
    return bundle, owned_transport_capability


def _component_directory(
    store: SecFilingGemmaRevealStore,
    claim: dict[str, Any],
) -> Path:
    return (
        store.store_directory
        / STAGE_OUTPUTS_DIRECTORY_NAME
        / claim["claim_sha256"]
        / MARKET_SOURCE_COMPONENT_DIRECTORY_NAME
    )


def test_owned_market_success_persists_exact_flat_layout_marker_last_and_returns_no_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _claim()
    receipt = _receipt(claim)
    events: list[str] = []
    _install_lock(store, events)
    _install_claim(store, claim=claim, created=True)
    effect_calls: list[str] = []
    _bundle_value, capability = _install_successful_effect(
        monkeypatch,
        claim=claim,
        calls=effect_calls,
    )
    revalidations: list[str] = []
    store._revalidate_authorized_market_execution_sources = (
        lambda value: revalidations.append(value["claim_sha256"])
    )
    store.abort_owned_development_market_execution = (
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not abort"))
    )
    recorded: list[tuple[str, object]] = []

    def record(
        *,
        development_root_scope_sha256: str,
        owned_transport_capability: object,
    ) -> dict[str, Any]:
        recorded.append(
            (development_root_scope_sha256, owned_transport_capability)
        )
        return receipt

    store._record_owned_development_market_reader_output = record
    writes: list[str] = []
    original_writer = runner._write_new_regular_file

    def tracked_write(path: Path, payload: bytes) -> None:
        writes.append(path.name)
        original_writer(path, payload)

    monkeypatch.setattr(runner, "_write_new_regular_file", tracked_write)

    result = runner.run_owned_development_market_batch(
        reveal_store=store,
        development_root_scope_sha256=SCOPE_SHA256,
    )

    assert result == {"claim": claim, "reader_receipt": receipt}
    assert set(result) == {"claim", "reader_receipt"}
    assert "capability" not in repr(result).lower()
    assert effect_calls == ["acquire", "validate"]
    assert len(revalidations) == 2
    assert recorded == [(SCOPE_SHA256, capability)]
    assert events == ["lock-enter", "lock-exit"]
    assert len(writes) == 23
    assert writes[-1] == DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME
    expected_names = {
        *(f"raw-response-{symbol}.json" for symbol in MARKET_SYMBOLS),
        *(f"artifact-{symbol}.json" for symbol in MARKET_SYMBOLS),
        *(f"window-{symbol}.json" for symbol in MARKET_SYMBOLS),
        "source-manifest.json",
        "stage-manifest.json",
        "reconciliation-receipt.json",
        "acquisition-receipt.json",
        DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME,
    }
    component = _component_directory(store, claim)
    assert {path.name for path in component.iterdir()} == expected_names
    marker = json.loads(
        (component / DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert marker["schema_version"] == (
        DEVELOPMENT_MARKET_COMPLETE_MARKER_SCHEMA_VERSION
    )
    assert marker["acquisition_validation_sha256"] == VALIDATION_SHA256
    assert marker["byte_index"][-1]["relative_path"] == "acquisition-receipt.json"
    assert len(marker["byte_index"]) == 22


def test_completed_market_retry_replays_reader_with_zero_network_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _claim()
    receipt = _receipt(claim)
    events: list[str] = []
    _install_lock(store, events)
    _install_claim(
        store,
        claim=claim,
        created=False,
        reader_receipt=receipt,
    )
    replayed: list[str] = []
    store._record_owned_development_market_reader_output = (
        lambda *, development_root_scope_sha256: replayed.append(
            development_root_scope_sha256
        )
        or receipt
    )
    monkeypatch.setattr(
        runner,
        "_acquire_owned_development_market_evidence",
        lambda: (_ for _ in ()).throw(AssertionError("must not acquire")),
    )

    assert runner.run_owned_development_market_batch(
        reveal_store=store,
        development_root_scope_sha256=SCOPE_SHA256,
    ) == {"claim": claim, "reader_receipt": receipt}
    assert replayed == [SCOPE_SHA256]
    assert events == ["lock-enter", "lock-exit"]


def test_active_orphan_market_claim_aborts_and_never_retries_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _claim()
    _install_lock(store, [])
    _install_claim(store, claim=claim, created=False)
    record_calls: list[object] = []
    store._record_owned_development_market_reader_output = (
        lambda **_kwargs: record_calls.append(object())
    )
    aborts: list[str] = []
    store.abort_owned_development_market_execution = (
        lambda *, development_root_scope_sha256, reason: aborts.append(reason)
    )
    monkeypatch.setattr(
        runner,
        "_acquire_owned_development_market_evidence",
        lambda: (_ for _ in ()).throw(AssertionError("must not acquire")),
    )

    with pytest.raises(runner.SecFilingGemmaStageRunnerError, match="cannot be retried"):
        runner.run_owned_development_market_batch(
            reveal_store=store,
            development_root_scope_sha256=SCOPE_SHA256,
        )

    assert aborts == ["claim_recovered_without_terminal_receipt"]
    assert record_calls == []


def test_precreated_empty_market_component_fails_before_effect_and_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _claim()
    _install_lock(store, [])
    _install_claim(store, claim=claim, created=True)
    component = _component_directory(store, claim)
    component.mkdir(parents=True)
    store._revalidate_authorized_market_execution_sources = lambda _claim: None
    aborts: list[str] = []
    store.abort_owned_development_market_execution = (
        lambda *, development_root_scope_sha256, reason: aborts.append(reason)
    )
    monkeypatch.setattr(
        runner,
        "_acquire_owned_development_market_evidence",
        lambda: (_ for _ in ()).throw(AssertionError("must not acquire")),
    )

    with pytest.raises(runner.SecFilingGemmaStageRunnerError, match="will not be retried"):
        runner.run_owned_development_market_batch(
            reveal_store=store,
            development_root_scope_sha256=SCOPE_SHA256,
        )

    assert aborts == ["durable_output_verification_failed"]
    assert not any(component.iterdir())


def test_failure_after_market_marker_is_terminally_aborted_and_never_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _new_store(tmp_path)
    claim = _claim()
    _install_lock(store, [])
    state: dict[str, Any] = {"created": True, "abort": None}

    def claim_execution(*, development_root_scope_sha256: str) -> dict[str, Any]:
        return {
            "claim": claim,
            "created": state["created"],
            "reader_receipt": None,
            "abort": state["abort"],
        }

    store.claim_owned_development_market_execution = claim_execution
    effect_calls: list[str] = []
    _bundle_value, capability = _install_successful_effect(
        monkeypatch,
        claim=claim,
        calls=effect_calls,
    )
    store._revalidate_authorized_market_execution_sources = lambda _claim: None
    aborts: list[str] = []
    def abort_execution(
        *,
        development_root_scope_sha256: str,
        reason: str,
    ) -> None:
        assert development_root_scope_sha256 == SCOPE_SHA256
        aborts.append(reason)
        state["created"] = False
        state["abort"] = {"terminal": True}

    store.abort_owned_development_market_execution = abort_execution
    records = {"count": 0}

    def record(
        *,
        development_root_scope_sha256: str,
        owned_transport_capability: object,
    ) -> dict[str, Any]:
        assert development_root_scope_sha256 == SCOPE_SHA256
        assert owned_transport_capability is capability
        records["count"] += 1
        assert (
            _component_directory(store, claim)
            / DEVELOPMENT_MARKET_COMPLETE_MARKER_FILENAME
        ).is_file()
        raise RuntimeError("simulated crash after marker")

    store._record_owned_development_market_reader_output = record

    with pytest.raises(
        runner.SecFilingGemmaStageRunnerError,
        match="will not be retried",
    ):
        runner.run_owned_development_market_batch(
            reveal_store=store,
            development_root_scope_sha256=SCOPE_SHA256,
        )
    assert aborts == ["external_effect_failed_or_completion_unknown"]
    assert effect_calls == ["acquire", "validate"]
    assert records["count"] == 1

    with pytest.raises(
        runner.SecFilingGemmaStageRunnerError,
        match="cannot be retried",
    ):
        runner.run_owned_development_market_batch(
            reveal_store=store,
            development_root_scope_sha256=SCOPE_SHA256,
        )
    assert effect_calls == ["acquire", "validate"]
    assert records["count"] == 1
    assert aborts == ["external_effect_failed_or_completion_unknown"]


def test_owned_market_public_signature_accepts_only_store_and_scope() -> None:
    signature = inspect.signature(runner.run_owned_development_market_batch)
    assert list(signature.parameters) == [
        "reveal_store",
        "development_root_scope_sha256",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert "run_owned_development_market_batch" in runner.__all__
