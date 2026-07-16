from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
from types import MethodType
from typing import Any

import pytest

import agent_benchmark.sec_gemma_online_risk_overlay_acquisition as acquisition
import agent_benchmark.sec_gemma_online_risk_overlay_vault as vault_module
from agent_benchmark.sec_audit_transport import ResponseAudit
from agent_benchmark.sec_filing_gemma_corpus import MAIN_SUBMISSIONS_URL
from agent_benchmark.sec_gemma_online_risk_overlay_acquisition import (
    ACQUISITION_VALIDATION_VERIFIER_ID,
    CONFIRMATION,
    DEVELOPMENT,
    FINAL,
    STAGES,
    AcquisitionExecutionResult,
    SecGemmaOnlineRiskOverlayAcquisitionError,
    SecGemmaOnlineRiskOverlayAcquisitionIndeterminate,
    VerifiedAcquisitionReport,
    acquire_stage_to_quarantine,
    build_acquisition_plan,
    is_verified_acquisition_report,
    rehydrate_contiguous_production_acquisitions,
    validate_acquisition_bundle,
    validate_acquisition_plan,
)
from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    CONFIRMATION_SCORING,
    DEVELOPMENT_ACQUISITION,
    FINAL_SCORING,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONFIRMATION_ATTEMPT_ID,
    DEVELOPMENT_ACQUISITION_ID,
    FINAL_ATTEMPT_ID,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_SECONDS,
    canonical_sha256,
)
from agent_benchmark.sec_gemma_online_risk_overlay_store import (
    EffectCapability,
    SecGemmaOnlineRiskOverlayStore,
    SecGemmaOnlineRiskOverlayStoreError,
    StoreRecordReceipt,
    _CAPABILITY_SENTINEL,
)
from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
    ProductionAcquisitionVault,
    TestAcquisitionVault,
    _VAULT_CONSTRUCTOR_SENTINEL,
    _seal_quarantine,
    open_test_acquisition_vault,
)
from agent_benchmark.sec_point_in_time import (
    content_sha256,
    validate_sec_user_agent,
)
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS as REAL_MARKET_SESSIONS,
)


USER_AGENT = "Private Researcher contact@real-domain-for-tests.dev"
HISTORY_NAME = "CIK0000320193-submissions-999.json"
HISTORY_URL = f"https://data.sec.gov/submissions/{HISTORY_NAME}"


def _test_market_calendar() -> tuple[str, ...]:
    development = [
        session
        for session in REAL_MARKET_SESSIONS
        if session <= "2018-12-31"
    ]
    confirmation = [
        session
        for session in REAL_MARKET_SESSIONS
        if "2019-01-01" <= session <= "2023-12-29"
    ]
    final = [
        session
        for session in REAL_MARKET_SESSIONS
        if "2024-01-01" <= session <= "2026-07-10"
    ]
    required_inceptions = {
        "1998-01-02",
        "1999-03-10",
        "2000-05-26",
    }
    return tuple(
        sorted(
            {
                *required_inceptions,
                *development[-300:],
                *confirmation[-30:],
                *final[-20:],
            }
        )
    )


TEST_MARKET_SESSIONS = _test_market_calendar()


@pytest.fixture(autouse=True)
def _short_test_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        acquisition,
        "EXPECTED_MARKET_HISTORY_SESSIONS",
        TEST_MARKET_SESSIONS,
    )


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _row(
    serial: int,
    *,
    year: int,
    month_day: str,
    form: str,
) -> dict[str, Any]:
    filing_date = f"{year:04d}-{month_day}"
    return {
        "accessionNumber": f"0000320193-{year % 100:02d}-{serial:06d}",
        "acceptanceDateTime": filing_date.replace("-", "") + "120000",
        "form": form,
        "primaryDocument": f"apple-{year}-{serial}.htm",
        "items": "",
        "filingDate": filing_date,
        "reportDate": f"{year:04d}-01-31",
        "isXBRL": 1,
        "dateOfFilingDateChange": "",
    }


def _columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys = (
        "accessionNumber",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
        "items",
        "filingDate",
        "reportDate",
        "isXBRL",
        "dateOfFilingDateChange",
    )
    return {key: [row[key] for row in rows] for key in keys}


def _all_catalog_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    serial = 1
    schedule = (
        ("02-01", "10-K"),
        ("05-01", "10-Q"),
        ("08-01", "10-Q"),
        ("11-01", "10-Q"),
    )
    for year in range(2000, 2019):
        for month_day, form in schedule:
            rows.append(
                _row(serial, year=year, month_day=month_day, form=form)
            )
            serial += 1
    for year in range(2019, 2024):
        count = 3 if year == 2023 else 4
        for month_day, form in schedule[:count]:
            rows.append(
                _row(serial, year=year, month_day=month_day, form=form)
            )
            serial += 1
    for year, count in ((2024, 4), (2025, 4), (2026, 2)):
        for month_day, form in schedule[:count]:
            rows.append(
                _row(serial, year=year, month_day=month_day, form=form)
            )
            serial += 1
    return rows


def _primary_url(row: dict[str, Any]) -> str:
    accession = row["accessionNumber"]
    return (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        f"{accession.replace('-', '')}/{row['primaryDocument']}"
    )


def _sec_payloads(stage: str) -> dict[str, bytes]:
    rows = _all_catalog_rows()
    old = _row(999_999, year=1995, month_day="01-03", form="8-K")
    main = {
        "cik": 320193,
        "filings": {
            "recent": _columns(rows),
            "files": [
                {
                    "name": HISTORY_NAME,
                    "filingCount": 1,
                    "filingFrom": old["filingDate"],
                    "filingTo": old["filingDate"],
                }
            ],
        },
    }
    payloads = {
        MAIN_SUBMISSIONS_URL: _json_bytes(main),
        HISTORY_URL: _json_bytes(_columns([old])),
    }
    windows = {
        DEVELOPMENT: (2000, 2018),
        CONFIRMATION: (2019, 2023),
        FINAL: (2024, 2026),
    }
    first, last = windows[stage]
    for row in rows:
        year = int(row["filingDate"][:4])
        if first <= year <= last:
            payloads[_primary_url(row)] = (
                b"<html><body>"
                b"Customer demand weakened and supply constraints increased. "
                b"Gross margins declined while operating uncertainty increased. "
                b"Management expects commercial conditions to remain uncertain."
                b"</body></html>"
            )
    return payloads


class FakeSecTransport:
    def __init__(
        self,
        stage: str,
        *,
        fail_with: str | None = None,
    ) -> None:
        self.payloads = _sec_payloads(stage)
        self.user_agent_audit = validate_sec_user_agent(USER_AGENT)
        self.calls: list[str] = []
        self.fail_with = fail_with

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
            "transport_max_requests": MAX_SEC_REQUESTS,
            "transport_max_bytes": MAX_SEC_BYTES,
            "transport_max_seconds": float(MAX_SEC_SECONDS),
        }

    def fetch(self, url: str) -> tuple[bytes, ResponseAudit]:
        self.calls.append(url)
        if self.fail_with is not None:
            raise RuntimeError(self.fail_with)
        payload = self.payloads[url]
        return payload, ResponseAudit(
            url=url,
            status_code=200,
            content_type=(
                "application/json; charset=utf-8"
                if url.startswith("https://data.sec.gov/")
                else "text/html; charset=iso-8859-1"
            ),
            size_bytes=len(payload),
            content_sha256=content_sha256(payload),
            cache_hit=False,
            user_agent_sha256=self.user_agent_audit.sha256,
            network_requests=1,
            retries=0,
            redirects=0,
        )


def _market_sessions(symbol: str, stage: str) -> tuple[str, ...]:
    cutoff = {
        DEVELOPMENT: "2018-12-31",
        CONFIRMATION: "2023-12-29",
        FINAL: "2026-07-10",
    }[stage]
    sessions = tuple(
        session for session in TEST_MARKET_SESSIONS if session <= cutoff
    )
    if symbol == "AAPL":
        return sessions
    first = acquisition.CONTEXT_FIRST_ACCEPTED_SESSION[symbol]
    return tuple(session for session in sessions if session >= first)


def _market_body(
    symbol: str,
    stage: str,
    *,
    truncate_to: int | None = None,
) -> bytes:
    request = next(
        item
        for item in build_acquisition_plan(stage)["market"]["requests"]
        if item["symbol"] == symbol
    )
    sessions = _market_sessions(symbol, stage)
    if truncate_to is not None:
        sessions = sessions[-truncate_to:]
    timestamps = [
        int(
            datetime.fromisoformat(
                f"{session}T18:00:00+00:00"
            ).timestamp()
        )
        for session in sessions
    ]
    count = len(sessions)
    values = {
        "open": [100.0 + index / 100_000.0 for index in range(count)],
        "high": [102.0 + index / 100_000.0 for index in range(count)],
        "low": [99.0 + index / 100_000.0 for index in range(count)],
        "close": [101.0 + index / 100_000.0 for index in range(count)],
        "volume": [1_000.0 + index for index in range(count)],
        "adjclose": [101.0 + index / 100_000.0 for index in range(count)],
    }
    return _json_bytes(
        {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": request["provider_symbol"],
                            "exchangeTimezoneName": request[
                                "provider_timezone"
                            ],
                            "dataGranularity": "1d",
                            "regularMarketPrice": 999_999.0,
                        },
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    key: values[key]
                                    for key in (
                                        "open",
                                        "high",
                                        "low",
                                        "close",
                                        "volume",
                                    )
                                }
                            ],
                            "adjclose": [
                                {"adjclose": values["adjclose"]}
                            ],
                        },
                    }
                ],
                "error": None,
            }
        }
    )


class FakeMarketTransport:
    def __init__(
        self,
        stage: str,
        *,
        overrides: dict[str, Any] | None = None,
        truncate_aapl_to: int | None = None,
    ) -> None:
        self.stage = stage
        self.calls: list[str] = []
        self.overrides = dict(overrides or {})
        self.truncate_aapl_to = truncate_aapl_to

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
            "transport_max_requests": 6,
            "transport_max_bytes": 128 * 1024 * 1024,
            "transport_max_seconds": 210.0,
        }

    def fetch(self, url: str) -> dict[str, Any]:
        self.calls.append(url)
        request = next(
            item
            for item in build_acquisition_plan(self.stage)["market"][
                "requests"
            ]
            if item["url"] == url
        )
        body = _market_body(
            request["symbol"],
            self.stage,
            truncate_to=(
                self.truncate_aapl_to
                if request["symbol"] == "AAPL"
                else None
            ),
        )
        value = {
            "request_url": url,
            "final_url": url,
            "status_code": 200,
            "content_type": "application/json",
            "charset": "utf-8",
            "content_encoding": None,
            "declared_content_length": len(body),
            "body": body,
            "network_requests": 1,
            "retries": 0,
            "redirects": 0,
        }
        value.update(self.overrides)
        return value


def _capability(
    stage: str,
    *,
    effects: tuple[str, ...] | None = None,
) -> EffectCapability:
    plan = build_acquisition_plan(stage)
    return EffectCapability(
        attempt_id=plan["attempt_id"],
        allowed_effects=effects
        or (
            "official_sec_network",
            "market_network",
            "deterministic_private_quarantine",
        ),
        receipt=StoreRecordReceipt(
            table="attempts",
            identity="attempt:1",
            attempt_id=plan["attempt_id"],
            payload_sha256="0" * 64,
            journal_sequence=1,
            journal_entry_sha256="1" * 64,
        ),
        store_instance_id="2" * 64,
        store_nonce="3" * 64,
        transition_sha256="4" * 64,
        _sentinel=_CAPABILITY_SENTINEL,
    )


class FakeStore:
    def __init__(self, capability: EffectCapability) -> None:
        self.capability = capability
        self.authorizations: list[str] = []

    def authorize_effect(
        self, capability: EffectCapability, effect: str
    ) -> None:
        self.authorizations.append(effect)
        if capability is not self.capability:
            raise RuntimeError("foreign capability")
        if effect not in capability.allowed_effects:
            raise RuntimeError("forbidden effect")


def _exact_recovery_store(
    capability: EffectCapability,
) -> SecGemmaOnlineRiskOverlayStore:
    store = object.__new__(SecGemmaOnlineRiskOverlayStore)
    store._store_instance_id = "2" * 64

    def authorize_effect(
        self: SecGemmaOnlineRiskOverlayStore,
        observed: EffectCapability,
        effect: str,
    ) -> None:
        assert self is store
        if observed is not capability or effect not in observed.allowed_effects:
            raise RuntimeError("foreign")

    store.authorize_effect = MethodType(  # type: ignore[method-assign]
        authorize_effect,
        store,
    )
    return store


def _development_recovery_material(
    *,
    bundle_sha256: str,
    manifest_sha256: str,
    private_index_sha256: str,
) -> dict[str, Any]:
    plan = build_acquisition_plan(DEVELOPMENT)
    checks = {
        name: f"{ordinal:064x}"
        for ordinal, name in enumerate(
            acquisition._VALIDATION_CHECK_NAMES,
            start=1,
        )
    }
    report_body = {
        "schema_version": acquisition.ACQUISITION_VALIDATION_SCHEMA_VERSION,
        "verifier_id": acquisition.ACQUISITION_VALIDATION_VERIFIER_ID,
        "verdict": "pass",
        "stage": DEVELOPMENT,
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": manifest_sha256,
        "private_index_sha256": private_index_sha256,
        "predecessor_chain_bundle_sha256s": [],
        "checks": checks,
        "check_set_sha256": canonical_sha256(checks),
    }
    report = {
        **report_body,
        "validation_sha256": canonical_sha256(report_body),
    }
    elapsed = (0.25).hex()
    summary_body = {
        "schema_version": (
            acquisition.ACQUISITION_PUBLIC_SUMMARY_SCHEMA_VERSION
        ),
        "stage": DEVELOPMENT,
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "predecessor_bundle_sha256": None,
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": manifest_sha256,
        "private_index_sha256": private_index_sha256,
        "sec_catalog_source_count": 2,
        "sec_primary_document_count": 72,
        "market_response_count": 6,
        "model_request_count": 72,
        "model_slice_sha256": "d" * 64,
        "stage_slice_sha256": "e" * 64,
        "market_elapsed_seconds_hex": elapsed,
        "total_raw_byte_count": 300,
        "complete_batch": True,
        "quarantine_only": True,
        "production_authority": True,
    }
    summary = {
        **summary_body,
        "public_summary_sha256": canonical_sha256(summary_body),
    }
    accounting_body = {
        "schema_version": (
            acquisition.ACQUISITION_REQUEST_ACCOUNTING_SCHEMA_VERSION
        ),
        "stage": DEVELOPMENT,
        "attempt_id": plan["attempt_id"],
        "sec_request_count": 74,
        "market_request_count": 6,
        "sec_bytes": 100,
        "market_bytes": 200,
        "network_request_count": 80,
        "retry_count": 0,
        "redirect_count": 0,
        "market_elapsed_seconds_hex": elapsed,
    }
    accounting = {
        **accounting_body,
        "accounting_sha256": canonical_sha256(accounting_body),
    }
    return {
        "verified_acquisition_report": report,
        "public_summary": summary,
        "request_accounting": accounting,
    }


def _sealed_production_recovery_fixture(
    tmp_path: Path,
) -> tuple[
    ProductionAcquisitionVault,
    SecGemmaOnlineRiskOverlayStore,
    Any,
    dict[str, Any],
]:
    capability = _capability(DEVELOPMENT)
    store = _exact_recovery_store(capability)
    path = tmp_path / "production-quarantine.sqlite3"
    vault = ProductionAcquisitionVault(
        database_path=path,
        production_authority=True,
        bound_store_instance_id="2" * 64,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )
    original = _seal_quarantine(
        vault,
        store=store,
        capability=capability,
        stage=DEVELOPMENT,
        attempt_id=DEVELOPMENT_ACQUISITION_ID,
        bundle_sha256="a" * 64,
        manifest_sha256="b" * 64,
        private_index_sha256="c" * 64,
        predecessor_handles=(),
        quarantine={
            "private_quarantine": {
                "secret_bytes": b"must-not-be-decoded-during-recovery"
            }
        },
    )
    reopened = ProductionAcquisitionVault(
        database_path=path,
        production_authority=True,
        bound_store_instance_id="2" * 64,
        _sentinel=_VAULT_CONSTRUCTOR_SENTINEL,
    )
    material = _development_recovery_material(
        bundle_sha256=original.bundle_sha256,
        manifest_sha256=original.manifest_sha256,
        private_index_sha256=original.private_index_sha256,
    )
    return reopened, store, original, material


def _acquire(
    stage: str,
    *,
    vault: TestAcquisitionVault,
    predecessors: tuple[AcquisitionExecutionResult, ...] = (),
    market: FakeMarketTransport | None = None,
) -> tuple[
    AcquisitionExecutionResult,
    FakeSecTransport,
    FakeMarketTransport,
    FakeStore,
]:
    capability = _capability(stage)
    store = FakeStore(capability)
    sec = FakeSecTransport(stage)
    fixed_market = market or FakeMarketTransport(stage)
    clock = Clock()
    execution = acquire_stage_to_quarantine(
        plan=build_acquisition_plan(stage),
        store=store,
        capability=capability,
        vault=vault,
        sec_user_agent=USER_AGENT,
        sec_transport=sec,
        market_transport=fixed_market,
        predecessor_executions=predecessors,
        allow_test_authorities=True,
        clock=clock,
        sleeper=clock.sleep,
    )
    return execution, sec, fixed_market, store


def test_real_frozen_aapl_session_counts_are_exact() -> None:
    assert sum(x <= "2018-12-31" for x in REAL_MARKET_SESSIONS) == 5283
    assert sum(x <= "2023-12-29" for x in REAL_MARKET_SESSIONS) == 6541
    assert sum(x <= "2026-07-09" for x in REAL_MARKET_SESSIONS) == 7172
    assert sum(x <= "2026-07-10" for x in REAL_MARKET_SESSIONS) == 7173


def test_plans_are_v2_1_exact_stage_bound_and_prefix_linked() -> None:
    plans = {stage: build_acquisition_plan(stage) for stage in STAGES}
    assert [plans[stage]["attempt_kind"] for stage in STAGES] == [
        DEVELOPMENT_ACQUISITION,
        CONFIRMATION_SCORING,
        FINAL_SCORING,
    ]
    assert [plans[stage]["attempt_id"] for stage in STAGES] == [
        DEVELOPMENT_ACQUISITION_ID,
        CONFIRMATION_ATTEMPT_ID,
        FINAL_ATTEMPT_ID,
    ]
    assert plans[CONFIRMATION]["predecessor_plan_sha256"] == plans[
        DEVELOPMENT
    ]["acquisition_plan_sha256"]
    assert plans[FINAL]["predecessor_plan_sha256"] == plans[
        CONFIRMATION
    ]["acquisition_plan_sha256"]
    for stage, plan in plans.items():
        assert "v2-1" in plan["schema_version"]
        assert validate_acquisition_plan(plan, expected_stage=stage) == plan
        assert len(plan["market"]["requests"]) == 6


def test_production_recovery_rehydrates_fresh_opaque_execution_and_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reopened, store, original, material = (
        _sealed_production_recovery_fixture(tmp_path)
    )

    def terminal_material(
        self: SecGemmaOnlineRiskOverlayStore,
        attempt_id: str,
    ) -> dict[str, Any]:
        assert self is store
        if attempt_id == DEVELOPMENT_ACQUISITION_ID:
            return json.loads(json.dumps(material))
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition recovery attempt did not terminal-pass"
        )

    monkeypatch.setattr(
        SecGemmaOnlineRiskOverlayStore,
        "terminal_acquisition_phase_evidence",
        terminal_material,
    )

    def forbidden_decode(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("recovery decoded opaque quarantine bytes")

    monkeypatch.setattr(vault_module, "_decode_opaque", forbidden_decode)
    recovered = rehydrate_contiguous_production_acquisitions(
        vault=reopened,
        store=store,
    )

    assert len(recovered) == 1
    execution = recovered[0]
    assert type(execution) is AcquisitionExecutionResult
    assert execution.production_authority is True
    assert execution.vault_handle is not original
    assert execution.public_summary() == material["public_summary"]
    assert (
        execution.terminal_sealing_material()
        == material["verified_acquisition_report"]
    )
    assert "must-not-be-decoded" not in repr(execution)

    from agent_benchmark.sec_gemma_online_risk_overlay_production import (
        _rehydrate_production_acquisition_ledger,
    )

    ledger = _rehydrate_production_acquisition_ledger(
        vault=reopened,
        store=store,
    )
    assert ledger.retained_stages() == (DEVELOPMENT,)

    class DurableHistory:
        def attempt_history(self, attempt_id: str) -> list[dict[str, str]]:
            if attempt_id == DEVELOPMENT_ACQUISITION_ID:
                return [{"status": "terminal_pass"}]
            return []

    assert (
        ledger.assert_ready_for_command(
            "development",
            store=DurableHistory(),
        )
        is None
    )


def test_production_recovery_rejects_sealed_stage_without_durable_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reopened, store, _original, _material = (
        _sealed_production_recovery_fixture(tmp_path)
    )

    def no_terminal_pass(
        self: SecGemmaOnlineRiskOverlayStore,
        _attempt_id: str,
    ) -> dict[str, Any]:
        assert self is store
        raise SecGemmaOnlineRiskOverlayStoreError(
            "Acquisition recovery attempt did not terminal-pass"
        )

    monkeypatch.setattr(
        SecGemmaOnlineRiskOverlayStore,
        "terminal_acquisition_phase_evidence",
        no_terminal_pass,
    )
    with pytest.raises(
        SecGemmaOnlineRiskOverlayAcquisitionError,
        match="passes and sealed vault stages differ",
    ):
        rehydrate_contiguous_production_acquisitions(
            vault=reopened,
            store=store,
        )


def test_success_is_opaque_detached_and_immutable(tmp_path: Path) -> None:
    vault = open_test_acquisition_vault(tmp_path)
    execution, sec, market, store = _acquire(
        DEVELOPMENT, vault=vault
    )

    assert type(execution) is AcquisitionExecutionResult
    assert not hasattr(execution, "bundle")
    assert execution.production_authority is False
    assert sec.calls
    assert market.calls
    assert store.authorizations[:2] == [
        "official_sec_network",
        "market_network",
    ]
    summary = execution.public_summary()
    accounting = execution.request_accounting()
    report = execution.verified_report
    assert summary["stage"] == DEVELOPMENT
    assert summary["sec_primary_document_count"] == 76
    assert summary["market_response_count"] == 6
    assert summary["model_request_count"] == 76
    assert summary["quarantine_only"] is True
    assert accounting["market_request_count"] == 6
    assert report["verifier_id"] == ACQUISITION_VALIDATION_VERIFIER_ID
    assert is_verified_acquisition_report(report)
    assert type(report) is VerifiedAcquisitionReport
    assert USER_AGENT not in repr(execution)
    assert "999999" not in json.dumps(summary)
    with pytest.raises(AttributeError):
        report._canonical_bytes = b"forged"  # type: ignore[misc]
    database_path = object.__getattribute__(vault, "_database_path")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE quarantine_entries SET payload=? WHERE entry_id=?",
            (
                b"changed",
                object.__getattribute__(
                    execution.vault_handle, "_entry_id"
                ),
            ),
        )
        connection.commit()
    assert not is_verified_acquisition_report(report)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayAcquisitionError,
        match="changed after validation",
    ):
        execution.terminal_sealing_material()


def test_arbitrary_mapping_cannot_enter_validator(tmp_path: Path) -> None:
    vault = open_test_acquisition_vault(tmp_path)
    capability = _capability(DEVELOPMENT)
    store = FakeStore(capability)
    with pytest.raises(
        SecGemmaOnlineRiskOverlayAcquisitionError,
        match="opaque vault handle",
    ):
        validate_acquisition_bundle(
            {"bundle_sha256": "0" * 64},  # type: ignore[arg-type]
            vault=vault,
            store=store,
            capability=capability,
            plan=build_acquisition_plan(DEVELOPMENT),
        )


def test_production_boundary_rejects_fakes_before_calls(
    tmp_path: Path,
) -> None:
    capability = _capability(DEVELOPMENT)
    store = FakeStore(capability)
    sec = FakeSecTransport(DEVELOPMENT)
    market = FakeMarketTransport(DEVELOPMENT)
    vault = open_test_acquisition_vault(tmp_path)

    with pytest.raises(
        SecGemmaOnlineRiskOverlayAcquisitionError,
        match="Production acquisition requires exact reviewed authorities",
    ):
        acquire_stage_to_quarantine(
            plan=build_acquisition_plan(DEVELOPMENT),
            store=store,
            capability=capability,
            vault=vault,
            sec_user_agent=USER_AGENT,
            sec_transport=sec,
            market_transport=market,
        )
    assert sec.calls == []
    assert market.calls == []
    assert store.authorizations == []


def test_short_parent_deadline_stops_before_second_sec_or_any_yahoo_call(
    tmp_path: Path,
) -> None:
    capability = _capability(DEVELOPMENT)
    store = FakeStore(capability)
    sec = FakeSecTransport(DEVELOPMENT)
    market = FakeMarketTransport(DEVELOPMENT)
    vault = open_test_acquisition_vault(tmp_path)
    clock = Clock()

    with pytest.raises(SecGemmaOnlineRiskOverlayAcquisitionIndeterminate):
        acquire_stage_to_quarantine(
            plan=build_acquisition_plan(DEVELOPMENT),
            store=store,
            capability=capability,
            vault=vault,
            sec_user_agent=USER_AGENT,
            sec_transport=sec,
            market_transport=market,
            allow_test_authorities=True,
            deadline_monotonic=0.25,
            clock=clock,
            sleeper=clock.sleep,
        )

    assert len(sec.calls) == 1
    assert market.calls == []
    assert clock.value == 0.0


def test_owned_yahoo_subcap_starts_at_market_phase_and_honors_parent() -> None:
    plan = build_acquisition_plan(DEVELOPMENT)
    short_parent = time.monotonic() + 5.0
    short = acquisition.create_production_market_transport(
        plan,
        deadline_monotonic=short_parent,
    )
    assert short._deadline is None
    short_phase_start = time.monotonic()
    short_deadline = min(short_parent, short_phase_start + 210.0)
    short._begin_batch(deadline_monotonic=short_deadline)
    assert short._deadline == short_parent
    assert short.acquisition_security_state()[
        "transport_max_seconds"
    ] == 210.0

    long_parent = time.monotonic() + 300.0
    long = acquisition.create_production_market_transport(
        plan,
        deadline_monotonic=long_parent,
    )
    assert long._deadline is None
    long_phase_start = time.monotonic()
    market_cap = long_phase_start + 210.0
    long._begin_batch(deadline_monotonic=market_cap)
    assert long._deadline == market_cap
    assert long._deadline < long_parent


def test_truncated_253_row_tail_is_terminal(tmp_path: Path) -> None:
    vault = open_test_acquisition_vault(tmp_path)
    market = FakeMarketTransport(
        DEVELOPMENT, truncate_aapl_to=253
    )
    with pytest.raises(SecGemmaOnlineRiskOverlayAcquisitionIndeterminate):
        _acquire(DEVELOPMENT, vault=vault, market=market)
    assert len(market.calls) == 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"final_url": "https://example.invalid/redirect"},
        {"retries": 1},
        {"redirects": 1},
        {"network_requests": 2},
        {"network_requests": True},
    ],
)
def test_redirect_retry_or_hidden_request_is_terminal(
    tmp_path: Path,
    overrides: dict[str, Any],
) -> None:
    vault = open_test_acquisition_vault(tmp_path)
    market = FakeMarketTransport(DEVELOPMENT, overrides=overrides)
    with pytest.raises(SecGemmaOnlineRiskOverlayAcquisitionIndeterminate):
        _acquire(DEVELOPMENT, vault=vault, market=market)
    assert len(market.calls) == 1


def test_exact_predecessor_chain_and_safe_final_cutoff(
    tmp_path: Path,
) -> None:
    vault = open_test_acquisition_vault(tmp_path)
    development, *_ = _acquire(DEVELOPMENT, vault=vault)
    confirmation, *_ = _acquire(
        CONFIRMATION,
        vault=vault,
        predecessors=(development,),
    )
    final, *_ = _acquire(
        FINAL,
        vault=vault,
        predecessors=(development, confirmation),
    )

    assert confirmation.public_summary()[
        "predecessor_bundle_sha256"
    ] == development.vault_handle.bundle_sha256
    assert final.public_summary()[
        "predecessor_bundle_sha256"
    ] == confirmation.vault_handle.bundle_sha256

    slice_capability = _capability(
        FINAL,
        effects=("canonical_market_value_read",),
    )
    slice_store = FakeStore(slice_capability)
    safe_slice = final.stage_slice(
        store=slice_store,
        capability=slice_capability,
    )
    assert safe_slice["last_value_session"] == "2026-07-09"
    assert safe_slice["market_rows"]["AAPL"][-1]["session"] == "2026-07-09"
    assert all(
        row["session"] != "2026-07-10"
        for rows in safe_slice["market_rows"].values()
        for row in rows
    )
    model_capability = _capability(
        FINAL,
        effects=("gemma_batch",),
    )
    model_store = FakeStore(model_capability)
    model_slice = final.model_slice(
        store=model_store,
        capability=model_capability,
    )
    assert set(model_slice) == {
        "stage",
        "attempt_id",
        "model_requests",
        "universe_event_proofs",
        "model_slice_sha256",
    }
    assert "market_rows" not in model_slice
    assert len(model_slice["model_requests"]) == len(
        model_slice["universe_event_proofs"]
    )
    assert model_slice["model_requests"][0]["supplied_sentence_ids"]


def test_identity_echo_and_broken_pacing_fail_without_leak(
    tmp_path: Path,
) -> None:
    vault = open_test_acquisition_vault(tmp_path / "echo")
    capability = _capability(DEVELOPMENT)
    store = FakeStore(capability)
    sec = FakeSecTransport(DEVELOPMENT, fail_with=USER_AGENT)
    market = FakeMarketTransport(DEVELOPMENT)
    clock = Clock()
    with pytest.raises(
        SecGemmaOnlineRiskOverlayAcquisitionIndeterminate
    ) as raised:
        acquire_stage_to_quarantine(
            plan=build_acquisition_plan(DEVELOPMENT),
            store=store,
            capability=capability,
            vault=vault,
            sec_user_agent=USER_AGENT,
            sec_transport=sec,
            market_transport=market,
            allow_test_authorities=True,
            clock=clock,
            sleeper=clock.sleep,
        )
    assert USER_AGENT not in str(raised.value)
    assert market.calls == []

    second_vault = open_test_acquisition_vault(tmp_path / "pacing")
    second_capability = _capability(DEVELOPMENT)
    second_store = FakeStore(second_capability)
    second_sec = FakeSecTransport(DEVELOPMENT)
    second_market = FakeMarketTransport(DEVELOPMENT)
    frozen = Clock()
    with pytest.raises(SecGemmaOnlineRiskOverlayAcquisitionIndeterminate):
        acquire_stage_to_quarantine(
            plan=build_acquisition_plan(DEVELOPMENT),
            store=second_store,
            capability=second_capability,
            vault=second_vault,
            sec_user_agent=USER_AGENT,
            sec_transport=second_sec,
            market_transport=second_market,
            allow_test_authorities=True,
            clock=frozen,
            sleeper=lambda _seconds: None,
        )
    assert len(second_sec.calls) == 1
    assert second_market.calls == []
