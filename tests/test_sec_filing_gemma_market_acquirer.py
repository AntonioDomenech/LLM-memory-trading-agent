from __future__ import annotations

from datetime import datetime, timezone
import copy
from dataclasses import replace
import hashlib
import inspect
import json
import pickle
from unittest.mock import patch
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

import pytest

import agent_benchmark.sec_filing_gemma_market_acquirer as acquirer
from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_market_evidence import MARKET_SYMBOLS
from agent_benchmark.sec_session_calendar import EXPECTED_MARKET_HISTORY_SESSIONS


DEV_SESSIONS = tuple(
    value
    for value in EXPECTED_MARKET_HISTORY_SESSIONS
    if acquirer.DEVELOPMENT_REQUEST_START
    <= value
    <= acquirer.DEVELOPMENT_LAST_SESSION
)


def _coverage_sessions(symbol: str) -> tuple[str, ...]:
    if symbol == "AAPL":
        return DEV_SESSIONS
    first = acquirer.CONTEXT_FIRST_ACCEPTED_SESSION[symbol]
    return tuple(session for session in DEV_SESSIONS if session >= first)


def _timestamp(session: str, symbol: str) -> int:
    observed = datetime.fromisoformat(f"{session}T09:30:00").replace(
        tzinfo=ZoneInfo(acquirer.YAHOO_PROVIDER_TIMEZONES[symbol])
    )
    return int(observed.timestamp())


def _response_object(
    symbol: str,
    sessions: tuple[str, ...],
    *,
    absent_indices: frozenset[int] = frozenset(),
) -> dict:
    count = len(sessions)
    values = {
        "open": [100.0 + index / 100_000.0 for index in range(count)],
        "high": [102.0 + index / 100_000.0 for index in range(count)],
        "low": [99.0 + index / 100_000.0 for index in range(count)],
        "close": [101.0 + index / 100_000.0 for index in range(count)],
        "volume": [1_000.0 + index for index in range(count)],
        "adjclose": [101.0 + index / 100_000.0 for index in range(count)],
    }
    for index in absent_indices:
        for array in values.values():
            array[index] = None
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": acquirer.YAHOO_PROVIDER_SYMBOLS[symbol],
                        "exchangeTimezoneName": acquirer.YAHOO_PROVIDER_TIMEZONES[
                            symbol
                        ],
                        "dataGranularity": acquirer.YAHOO_INTERVAL,
                        # Provider meta is deliberately not normalized into rows.
                        "regularMarketTime": 4_102_444_800,
                    },
                    "timestamp": [_timestamp(session, symbol) for session in sessions],
                    "indicators": {
                        "quote": [
                            {
                                "open": values["open"],
                                "high": values["high"],
                                "low": values["low"],
                                "close": values["close"],
                                "volume": values["volume"],
                            }
                        ],
                        "adjclose": [{"adjclose": values["adjclose"]}],
                    },
                }
            ],
            "error": None,
        }
    }


def _bytes(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _small_context_response(symbol: str = "SPY") -> dict:
    return _response_object(symbol, ("2018-12-27", "2018-12-28"))


class _FakeTransport:
    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.urls: list[str] = []

    def fetch(self, url: str) -> acquirer._TransportResponse:
        self.urls.append(url)
        provider_symbol = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        symbol = next(
            key
            for key, value in acquirer.YAHOO_PROVIDER_SYMBOLS.items()
            if value == provider_symbol
        )
        return acquirer._TransportResponse(
            request_url=url,
            final_url=url,
            status_code=200,
            content_type="application/json",
            charset="utf-8",
            content_encoding=None,
            declared_content_length=len(self.bodies[symbol]),
            body=self.bodies[symbol],
        )


def _development_response_bodies() -> dict[str, bytes]:
    tnx_sessions = _coverage_sessions("TNX")
    tnx_absent_session = acquirer.CONTEXT_ALLOWED_MISSING_SESSIONS["TNX"][0]
    return {
        "AAPL": _bytes(_response_object("AAPL", DEV_SESSIONS)),
        **{
            symbol: _bytes(
                _response_object(
                    symbol,
                    _coverage_sessions(symbol),
                    absent_indices=(
                        frozenset({tnx_sessions.index(tnx_absent_session)})
                        if symbol == "TNX"
                        else frozenset()
                    ),
                )
            )
            for symbol in MARKET_SYMBOLS
            if symbol != "AAPL"
        },
    }


def _owned_acquisition_without_network(
    bodies: dict[str, bytes] | None = None,
) -> object:
    exact_bodies = bodies if bodies is not None else _development_response_bodies()

    def fetch(
        self: acquirer._OwnedYahooChartTransport,
        url: str,
    ) -> acquirer._TransportResponse:
        if (
            self.request_count >= len(MARKET_SYMBOLS)
            or url != acquirer._canonical_url(MARKET_SYMBOLS[self.request_count])
        ):
            raise AssertionError("owned transport received a non-canonical test URL")
        provider_symbol = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        symbol = next(
            key
            for key, value in acquirer.YAHOO_PROVIDER_SYMBOLS.items()
            if value == provider_symbol
        )
        body = exact_bodies[symbol]
        self.request_count += 1
        self.response_bytes += len(body)
        return acquirer._TransportResponse(
            request_url=url,
            final_url=url,
            status_code=200,
            content_type="application/json",
            charset="utf-8",
            content_encoding=None,
            declared_content_length=len(body),
            body=body,
        )

    with patch.object(acquirer._OwnedYahooChartTransport, "fetch", fetch):
        return acquirer._acquire_owned_development_market_evidence()


@pytest.fixture(scope="module")
def acquisition_bundle() -> dict:
    bodies = _development_response_bodies()
    return acquirer._acquire_development_market_evidence_with_transport(
        _FakeTransport(bodies)
    )


def test_frozen_plan_binds_exact_urls_epochs_aliases_sources_and_zero_cost() -> None:
    plan = acquirer.build_development_market_acquisition_plan()

    assert acquirer.DEVELOPMENT_PERIOD1_UTC == 883_612_800
    assert acquirer.DEVELOPMENT_PERIOD2_UTC == 1_546_300_800
    assert acquirer.YAHOO_PROVIDER_SYMBOLS == {
        "AAPL": "AAPL",
        "SPY": "SPY",
        "QQQ": "QQQ",
        "IWM": "IWM",
        "VIX": "^VIX",
        "TNX": "^TNX",
    }
    assert acquirer.YAHOO_PROVIDER_TIMEZONES == {
        "AAPL": "America/New_York",
        "SPY": "America/New_York",
        "QQQ": "America/New_York",
        "IWM": "America/New_York",
        "VIX": "America/Chicago",
        "TNX": "America/Chicago",
    }
    assert [item["symbol"] for item in plan["requests"]] == list(MARKET_SYMBOLS)
    assert plan["requests"][-1]["canonical_url"] == (
        "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?"
        "period1=883612800&period2=1546300800&interval=1d&"
        "includePrePost=false&includeAdjustedClose=true&events=div%2Csplits"
    )
    authority = plan["transport_authority"]
    assert authority["fixed_request_count"] == 6
    assert authority["aggregate_response_byte_ceiling"] == 128 * 1024 * 1024
    assert authority["monotonic_batch_deadline_seconds"] == 210
    assert authority["estimated_cost_usd"] == "0.00"
    assert authority["paid_api_calls_permitted"] == 0
    assert not any(
        authority[key]
        for key in (
            "fallback_permitted",
            "retry_permitted",
            "redirect_permitted",
            "proxy_permitted",
            "cookie_permitted",
            "authentication_permitted",
            "cache_read_permitted",
            "cache_write_permitted",
            "compression_permitted",
        )
    )
    assert set(plan["source_code_sha256s"]) == {
        "agent_benchmark/sec_filing_gemma_market_acquirer.py",
        "agent_benchmark/sec_filing_gemma_market_evidence.py",
        "agent_benchmark/sec_filing_gemma_market_source_bytes.py",
        "agent_benchmark/sec_filing_gemma_contract.py",
        "agent_benchmark/sec_session_calendar.py",
    }
    assert all(
        len(value) == 64 for value in plan["source_code_sha256s"].values()
    )
    body = {key: value for key, value in plan.items() if key != "acquisition_plan_sha256"}
    assert plan["acquisition_plan_sha256"] == canonical_sha256(body)
    assert len(
        inspect.signature(
            acquirer._acquire_owned_development_market_evidence
        ).parameters
    ) == 0
    assert "_acquire_owned_development_market_evidence" not in acquirer.__all__


def test_offline_acquisition_normalizes_and_reconciles_exact_development_bundle(
    acquisition_bundle: dict,
) -> None:
    bundle = acquisition_bundle

    assert len(DEV_SESSIONS) == 5_283
    assert set(bundle["raw_response_bytes_by_symbol"]) == set(MARKET_SYMBOLS)
    assert bundle["stage_manifest"]["row_count"] == 5_283
    assert bundle["stage_manifest"]["first_session"] == DEV_SESSIONS[0]
    assert bundle["stage_manifest"]["last_session"] == DEV_SESSIONS[-1]
    assert bundle["reconciliation_receipt"]["reconciled_row_counts"] == {
        "AAPL": 5_283,
        "SPY": len(_coverage_sessions("SPY")),
        "QQQ": len(_coverage_sessions("QQQ")),
        "IWM": len(_coverage_sessions("IWM")),
        "VIX": len(_coverage_sessions("VIX")),
        "TNX": len(_coverage_sessions("TNX")) - 1,
    }
    receipt = bundle["acquisition_receipt"]
    assert receipt["acquisition_plan_sha256"] == bundle["acquisition_plan_sha256"]
    assert receipt["raw_response_count"] == 6
    assert receipt["raw_response_total_bytes"] == sum(
        map(len, bundle["raw_response_bytes_by_symbol"].values())
    )
    assert receipt["transport_policy"]["trusted_production_transport"] is False
    assert receipt["transport_policy"]["authorizes_production_use"] is False
    assert receipt["transport_policy"]["fresh_network_provenance_claimed"] is False
    assert receipt["normalization_contract"]["append_compatibility_claimed"] is False
    stats = {item["symbol"]: item for item in receipt["requests"]}
    assert stats["AAPL"]["accepted_row_count"] == 5_283
    assert stats["VIX"]["accepted_row_count"] == 5_283
    assert stats["VIX"]["explicit_absent_row_count"] == 0
    assert stats["VIX"]["implicit_missing_session_count"] == 0
    assert stats["TNX"]["explicit_absent_row_count"] == 1
    source_tnx = next(
        item for item in bundle["source_manifest"]["sources"] if item["symbol"] == "TNX"
    )
    assert len(source_tnx["available_sessions"]) == len(_coverage_sessions("TNX")) - 1
    assert source_tnx["artifact_sha256"] == hashlib.sha256(
        bundle["artifact_bytes_by_symbol"]["TNX"]
    ).hexdigest()
    assert source_tnx["artifact_sha256"] != hashlib.sha256(
        bundle["raw_response_bytes_by_symbol"]["TNX"]
    ).hexdigest()
    absent_session = acquirer.CONTEXT_ALLOWED_MISSING_SESSIONS["TNX"][0]
    absent_row = next(
        row for row in bundle["stage_manifest"]["rows"]
        if row["session"] == absent_session
    )["observations"]["TNX"]
    assert absent_row["available"] is False
    assert all(value is None for key, value in absent_row.items() if key.endswith("_hex"))


def test_exact_owned_transport_issues_opaque_claim_bound_one_shot_capability() -> None:
    owned_result = _owned_acquisition_without_network()

    assert type(owned_result) is not dict
    bundle, capability = acquirer._unwrap_owned_development_market_acquisition(
        owned_result
    )
    assert bundle["acquisition_receipt"]["transport_policy"][
        "trusted_production_transport"
    ] is True
    assert not hasattr(capability, "__dict__")
    assert repr(capability) == "<owned Yahoo transport capability>"
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(capability)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(capability)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(capability)
    with pytest.raises(acquirer.MarketAcquisitionError, match="already unwrapped"):
        acquirer._unwrap_owned_development_market_acquisition(owned_result)

    validation = acquirer.validate_development_market_acquisition_bundle(
        bundle,
        expected_acquisition_plan_sha256=bundle["acquisition_plan_sha256"],
        expected_acquisition_receipt_sha256=bundle[
            "acquisition_receipt_sha256"
        ],
        expected_bundle_sha256=bundle["bundle_sha256"],
    )
    scope_hash = "a" * 64
    claim_hash = "b" * 64
    acquirer._bind_owned_market_transport_capability_to_claim(
        capability,
        development_root_scope_sha256=scope_hash,
        claim_sha256=claim_hash,
    )
    consume_kwargs = {
        "development_root_scope_sha256": scope_hash,
        "claim_sha256": claim_hash,
        "acquisition_plan_sha256": bundle["acquisition_plan_sha256"],
        "acquisition_receipt_sha256": bundle["acquisition_receipt_sha256"],
        "bundle_sha256": bundle["bundle_sha256"],
        "validation_sha256": validation["validation_sha256"],
        "source_manifest_sha256": bundle["source_manifest_sha256"],
        "market_stage_manifest_sha256": bundle[
            "market_stage_manifest_sha256"
        ],
        "source_reconciliation_sha256": bundle[
            "source_reconciliation_sha256"
        ],
    }
    with pytest.raises(acquirer.MarketAcquisitionError, match="binding changed"):
        acquirer._consume_owned_market_transport_capability(
            capability,
            **{**consume_kwargs, "bundle_sha256": "c" * 64},
        )
    with pytest.raises(acquirer.MarketAcquisitionError, match="already consumed"):
        acquirer._consume_owned_market_transport_capability(
            capability,
            **consume_kwargs,
        )


def test_injected_bundle_cannot_cross_owned_transport_capability_boundary(
    acquisition_bundle: dict,
) -> None:
    with pytest.raises(
        acquirer.MarketAcquisitionError,
        match="did not come from the owned Yahoo transport boundary",
    ):
        acquirer._unwrap_owned_development_market_acquisition(
            acquisition_bundle
        )


def test_pure_validator_enforces_aggregate_raw_response_cap(
    acquisition_bundle: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_total = sum(
        len(payload)
        for payload in acquisition_bundle["raw_response_bytes_by_symbol"].values()
    )
    monkeypatch.setattr(
        acquirer,
        "YAHOO_MAX_TOTAL_RESPONSE_BYTES",
        raw_total - 1,
    )

    with pytest.raises(
        acquirer.MarketAcquisitionError,
        match="aggregate raw-response byte ceiling",
    ):
        acquirer.validate_development_market_acquisition_bundle(
            acquisition_bundle,
            expected_acquisition_plan_sha256=acquisition_bundle[
                "acquisition_plan_sha256"
            ],
            expected_acquisition_receipt_sha256=acquisition_bundle[
                "acquisition_receipt_sha256"
            ],
            expected_bundle_sha256=acquisition_bundle["bundle_sha256"],
        )


def test_context_parser_allows_all_null_rows_and_sparse_pre_inception_history() -> None:
    payload = _response_object(
        "QQQ",
        ("1999-03-10", "1999-03-11"),
        absent_indices=frozenset({0}),
    )

    parsed = acquirer._parse_yahoo_chart_response(
        symbol="QQQ",
        raw_bytes=_bytes(payload),
    )

    assert parsed.explicit_absent_sessions == ("1999-03-10",)
    assert list(parsed.rows_by_session) == ["1999-03-11"]


def test_context_coverage_rejects_truncation_and_unexplained_gaps(
    acquisition_bundle: dict,
) -> None:
    raw = acquisition_bundle["raw_response_bytes_by_symbol"]
    for symbol in ("SPY", "QQQ", "IWM", "VIX", "TNX"):
        parsed = acquirer._parse_yahoo_chart_response(
            symbol=symbol,
            raw_bytes=raw[symbol],
        )
        acquirer._validate_symbol_coverage(symbol, parsed)
        assert next(iter(parsed.rows_by_session)) == (
            acquirer.CONTEXT_FIRST_ACCEPTED_SESSION[symbol]
        )
        assert next(reversed(parsed.rows_by_session)) == acquirer.DEVELOPMENT_LAST_SESSION

    spy = acquirer._parse_yahoo_chart_response(symbol="SPY", raw_bytes=raw["SPY"])
    truncated_rows = dict(spy.rows_by_session)
    truncated_rows.pop(DEV_SESSIONS[len(DEV_SESSIONS) // 2])
    with pytest.raises(acquirer.MarketAcquisitionError, match="unexplained"):
        acquirer._validate_symbol_coverage(
            "SPY",
            replace(spy, rows_by_session=truncated_rows),
        )

    wrong_first_rows = dict(spy.rows_by_session)
    wrong_first_rows.pop(DEV_SESSIONS[0])
    with pytest.raises(acquirer.MarketAcquisitionError, match="first or final"):
        acquirer._validate_symbol_coverage(
            "SPY",
            replace(spy, rows_by_session=wrong_first_rows),
        )


def test_tnx_only_allows_frozen_optional_bond_holiday_gaps(
    acquisition_bundle: dict,
) -> None:
    raw = acquisition_bundle["raw_response_bytes_by_symbol"]["TNX"]
    parsed = acquirer._parse_yahoo_chart_response(symbol="TNX", raw_bytes=raw)
    acquirer._validate_symbol_coverage("TNX", parsed)
    permitted = set(acquirer.CONTEXT_ALLOWED_MISSING_SESSIONS["TNX"])
    assert permitted
    assert set(parsed.explicit_absent_sessions).issubset(permitted)

    ordinary_session = next(
        session
        for session in _coverage_sessions("TNX")[1:-1]
        if session not in permitted and session in parsed.rows_by_session
    )
    changed_rows = dict(parsed.rows_by_session)
    changed_rows.pop(ordinary_session)
    with pytest.raises(acquirer.MarketAcquisitionError, match="unexplained"):
        acquirer._validate_symbol_coverage(
            "TNX",
            replace(parsed, rows_by_session=changed_rows),
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value["chart"]["result"].append(
                value["chart"]["result"][0]
            ),
            "exactly one object",
        ),
        (
            lambda value: value["chart"]["result"][0]["meta"].__setitem__(
                "symbol", "QQQ"
            ),
            "provider symbol",
        ),
        (
            lambda value: value["chart"]["result"][0]["meta"].__setitem__(
                "exchangeTimezoneName", "UTC"
            ),
            "provider timezone",
        ),
        (
            lambda value: value["chart"]["result"][0]["meta"].__setitem__(
                "dataGranularity", "1h"
            ),
            "provider interval",
        ),
        (
            lambda value: value["chart"]["result"][0]["indicators"]["quote"][0][
                "open"
            ].pop(),
            "misaligned",
        ),
        (
            lambda value: value["chart"]["result"][0]["timestamp"].__setitem__(
                1, value["chart"]["result"][0]["timestamp"][0]
            ),
            "duplicated or reordered",
        ),
        (
            lambda value: value["chart"]["result"][0]["indicators"]["quote"][0][
                "open"
            ].__setitem__(0, None),
            "partial",
        ),
        (
            lambda value: value["chart"]["result"][0]["indicators"]["quote"][0][
                "volume"
            ].__setitem__(0, -1),
            "volume cannot be negative",
        ),
        (
            lambda value: value["chart"]["result"][0]["indicators"]["quote"][0][
                "high"
            ].__setitem__(0, 1),
            "high is inconsistent",
        ),
    ],
)
def test_parser_rejects_provider_shape_identity_partial_and_value_failures(
    mutate,
    match: str,
) -> None:
    value = _small_context_response()
    mutate(value)

    with pytest.raises(acquirer.MarketAcquisitionError, match=match):
        acquirer._parse_yahoo_chart_response(symbol="SPY", raw_bytes=_bytes(value))


def test_parser_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    with pytest.raises(acquirer.MarketAcquisitionError, match="duplicate JSON key"):
        acquirer._parse_yahoo_chart_response(
            symbol="SPY",
            raw_bytes=b'{"chart":{},"chart":{}}',
        )

    ordinary = _bytes(_small_context_response())
    nonfinite = ordinary.replace(b"100.0", b"1e999", 1)
    with pytest.raises(acquirer.MarketAcquisitionError, match="non-finite"):
        acquirer._parse_yahoo_chart_response(symbol="SPY", raw_bytes=nonfinite)


def test_parser_rejects_future_rows_and_incomplete_aapl_coverage() -> None:
    future = _small_context_response()
    future["chart"]["result"][0]["timestamp"][1] = int(
        datetime.fromisoformat("2019-01-02T09:30:00")
        .replace(tzinfo=ZoneInfo(acquirer.YAHOO_PROVIDER_TIMEZONES["SPY"]))
        .timestamp()
    )
    with pytest.raises(acquirer.MarketAcquisitionError, match="request window"):
        acquirer._parse_yahoo_chart_response(symbol="SPY", raw_bytes=_bytes(future))

    with pytest.raises(acquirer.MarketAcquisitionError, match="every frozen"):
        acquirer._parse_yahoo_chart_response(
            symbol="AAPL",
            raw_bytes=_bytes(_response_object("AAPL", (DEV_SESSIONS[0],))),
        )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"final_url": "https://example.invalid/redirect"}, "redirected or changed"),
        ({"status_code": 429}, "not HTTP 200"),
        ({"content_type": "text/html"}, "not JSON"),
        ({"content_encoding": "gzip"}, "compressed"),
        ({"declared_content_length": 0}, "Content-Length"),
    ],
)
def test_transport_response_rejects_redirect_status_type_and_compression(
    change: dict,
    match: str,
) -> None:
    url = acquirer._canonical_url("SPY")
    values = {
        "request_url": url,
        "final_url": url,
        "status_code": 200,
        "content_type": "application/json",
        "charset": "utf-8",
        "content_encoding": None,
        "declared_content_length": len(_bytes(_small_context_response())),
        "body": _bytes(_small_context_response()),
    }
    values.update(change)

    with pytest.raises(acquirer.MarketAcquisitionError, match=match):
        acquirer._validate_transport_response(
            acquirer._TransportResponse(**values),
            expected_url=url,
            symbol="SPY",
        )


class _FakeDeadlineClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _FakeDeadlineSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)


class _FakeDeadlineRaw:
    def __init__(self, sock: _FakeDeadlineSocket) -> None:
        self._sock = sock


class _FakeDeadlineFilePointer:
    def __init__(self, sock: _FakeDeadlineSocket) -> None:
        self.raw = _FakeDeadlineRaw(sock)


class _FakeDeadlineHeaders:
    def __init__(
        self,
        content_length: int | None = None,
        *,
        content_length_values: list[str] | None = None,
        transfer_encoding_values: list[str] | None = None,
    ) -> None:
        self._values = {
            "Content-Length": (
                list(content_length_values)
                if content_length_values is not None
                else ([] if content_length is None else [str(content_length)])
            ),
            "Transfer-Encoding": list(transfer_encoding_values or []),
        }

    def get(self, name: str) -> str | None:
        values = self._values.get(name, [])
        return values[0] if values else None

    def get_all(self, name: str, default=None):
        values = self._values.get(name, [])
        return list(values) if values else default

    def get_content_type(self) -> str:
        return "application/json"

    def get_content_charset(self) -> str:
        return "utf-8"


class _FakeDeadlineHTTPResponse:
    def __init__(
        self,
        *,
        url: str,
        clock: _FakeDeadlineClock,
        steps: list[tuple[float, bytes | BaseException]],
        content_length: int | None = None,
        content_length_values: list[str] | None = None,
        transfer_encoding_values: list[str] | None = None,
    ) -> None:
        self._url = url
        self._clock = clock
        self._steps = list(steps)
        self.socket = _FakeDeadlineSocket()
        self.fp = _FakeDeadlineFilePointer(self.socket)
        self.headers = _FakeDeadlineHeaders(
            content_length,
            content_length_values=content_length_values,
            transfer_encoding_values=transfer_encoding_values,
        )
        self.read_sizes: list[int] = []

    def __enter__(self) -> _FakeDeadlineHTTPResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        raise AssertionError("owned transport used an unbounded response read")

    def read1(self, size: int) -> bytes:
        self.read_sizes.append(size)
        if not self._steps:
            raise AssertionError("owned transport read beyond the fake response")
        advance, result = self._steps.pop(0)
        self._clock.now += advance
        if isinstance(result, BaseException):
            raise result
        return result

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return 200


class _FakeDeadlineOpener:
    def __init__(self, response: _FakeDeadlineHTTPResponse) -> None:
        self._response = response
        self.timeouts: list[float] = []

    def open(self, request, *, timeout: float) -> _FakeDeadlineHTTPResponse:
        self.timeouts.append(timeout)
        return self._response


def _deadline_transport(
    *,
    monkeypatch: pytest.MonkeyPatch,
    clock: _FakeDeadlineClock,
    response: _FakeDeadlineHTTPResponse,
    deadline: float,
) -> tuple[acquirer._OwnedYahooChartTransport, _FakeDeadlineOpener]:
    transport = acquirer._OwnedYahooChartTransport()
    opener = _FakeDeadlineOpener(response)
    transport._opener = opener
    transport._deadline = deadline
    monkeypatch.setattr(acquirer.time, "monotonic", clock)
    return transport, opener


def test_owned_transport_reads_only_bounded_chunks_with_remaining_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = acquirer._canonical_url("AAPL")
    clock = _FakeDeadlineClock(100.0)
    response = _FakeDeadlineHTTPResponse(
        url=url,
        clock=clock,
        steps=[(0.2, b"ab"), (0.2, b"cd"), (0.2, b"")],
    )
    transport, opener = _deadline_transport(
        monkeypatch=monkeypatch,
        clock=clock,
        response=response,
        deadline=105.0,
    )

    fetched = transport.fetch(url)

    assert fetched.body == b"abcd"
    assert opener.timeouts == [pytest.approx(5.0)]
    assert response.read_sizes == [
        acquirer._YAHOO_BODY_READ_CHUNK_BYTES,
        acquirer._YAHOO_BODY_READ_CHUNK_BYTES,
        acquirer._YAHOO_BODY_READ_CHUNK_BYTES,
    ]
    assert response.socket.timeouts == pytest.approx([5.0, 4.8, 4.6])
    assert all(
        later < earlier
        for earlier, later in zip(
            response.socket.timeouts,
            response.socket.timeouts[1:],
        )
    )
    assert transport.request_count == 1
    assert transport.response_bytes == 4


def test_owned_transport_rejects_chunked_body_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = acquirer._canonical_url("AAPL")
    clock = _FakeDeadlineClock(100.0)
    response = _FakeDeadlineHTTPResponse(
        url=url,
        clock=clock,
        steps=[(0.1, b"ab"), (0.1, b"cd"), (0.1, b"")],
        transfer_encoding_values=["chunked"],
    )
    transport, _ = _deadline_transport(
        monkeypatch=monkeypatch,
        clock=clock,
        response=response,
        deadline=105.0,
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match="transfer coding"):
        transport.fetch(url)

    assert response.read_sizes == []
    assert transport.response_bytes == 0


@pytest.mark.parametrize(
    ("header_changes", "match"),
    (
        (
            {
                "content_length_values": ["2"],
                "transfer_encoding_values": ["chunked"],
            },
            "conflicting body-framing",
        ),
        ({"content_length_values": ["2", "2"]}, "duplicate body-framing"),
        (
            {"transfer_encoding_values": ["chunked", "chunked"]},
            "duplicate body-framing",
        ),
        ({"transfer_encoding_values": ["gzip"]}, "unsupported transfer coding"),
    ),
)
def test_owned_transport_rejects_ambiguous_or_unsupported_body_framing(
    monkeypatch: pytest.MonkeyPatch,
    header_changes: dict[str, list[str]],
    match: str,
) -> None:
    url = acquirer._canonical_url("AAPL")
    clock = _FakeDeadlineClock(100.0)
    response = _FakeDeadlineHTTPResponse(
        url=url,
        clock=clock,
        steps=[(0.0, b"")],
        **header_changes,
    )
    transport, _ = _deadline_transport(
        monkeypatch=monkeypatch,
        clock=clock,
        response=response,
        deadline=105.0,
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match=match):
        transport.fetch(url)

    assert response.read_sizes == []
    assert transport.response_bytes == 0


def test_owned_transport_does_not_accept_content_length_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = acquirer._canonical_url("AAPL")
    clock = _FakeDeadlineClock(100.0)
    response = _FakeDeadlineHTTPResponse(
        url=url,
        clock=clock,
        steps=[(0.1, b"ab"), (0.1, b"cd")],
        content_length=2,
    )
    transport, _ = _deadline_transport(
        monkeypatch=monkeypatch,
        clock=clock,
        response=response,
        deadline=105.0,
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match="Content-Length"):
        transport.fetch(url)

    assert response.read_sizes == [
        acquirer._YAHOO_BODY_READ_CHUNK_BYTES,
        acquirer._YAHOO_BODY_READ_CHUNK_BYTES,
    ]
    assert transport.response_bytes == 0


def test_owned_transport_slow_drip_cannot_cross_absolute_body_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = acquirer._canonical_url("AAPL")
    clock = _FakeDeadlineClock(100.0)
    response = _FakeDeadlineHTTPResponse(
        url=url,
        clock=clock,
        steps=[(0.3, b"a"), (0.3, b"b"), (0.3, b"c"), (0.0, b"")],
    )
    transport, _ = _deadline_transport(
        monkeypatch=monkeypatch,
        clock=clock,
        response=response,
        deadline=100.75,
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match="body crossed"):
        transport.fetch(url)

    assert len(response.read_sizes) == 3
    assert response.socket.timeouts == pytest.approx([0.75, 0.45, 0.15])
    assert all(timeout > 0 for timeout in response.socket.timeouts)
    assert transport.request_count == 1
    assert transport.response_bytes == 0


def test_owned_transport_body_timeout_fails_closed_without_accounting_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = acquirer._canonical_url("AAPL")
    clock = _FakeDeadlineClock(100.0)
    response = _FakeDeadlineHTTPResponse(
        url=url,
        clock=clock,
        steps=[(0.1, TimeoutError("simulated socket timeout"))],
    )
    transport, _ = _deadline_transport(
        monkeypatch=monkeypatch,
        clock=clock,
        response=response,
        deadline=105.0,
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match="body crossed"):
        transport.fetch(url)

    assert response.read_sizes == [acquirer._YAHOO_BODY_READ_CHUNK_BYTES]
    assert transport.response_bytes == 0


def test_owned_transport_max_plus_one_body_fails_at_fixed_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(acquirer, "YAHOO_MAX_RESPONSE_BYTES", 3)
    monkeypatch.setattr(acquirer, "_YAHOO_BODY_READ_CHUNK_BYTES", 2)
    url = acquirer._canonical_url("AAPL")
    clock = _FakeDeadlineClock(100.0)
    response = _FakeDeadlineHTTPResponse(
        url=url,
        clock=clock,
        steps=[(0.1, b"ab"), (0.1, b"cd")],
    )
    transport, _ = _deadline_transport(
        monkeypatch=monkeypatch,
        clock=clock,
        response=response,
        deadline=105.0,
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match="byte ceiling"):
        transport.fetch(url)

    assert response.read_sizes == [2, 2]
    assert transport.response_bytes == 0


@pytest.mark.parametrize("missing", ("read1", "socket"))
def test_owned_transport_rejects_response_without_bounded_read_shape(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    url = acquirer._canonical_url("AAPL")
    clock = _FakeDeadlineClock(100.0)
    response = _FakeDeadlineHTTPResponse(
        url=url,
        clock=clock,
        steps=[(0.0, b"")],
    )
    if missing == "read1":
        response.read1 = None
        match = "bounded single-read"
    else:
        response.fp.raw._sock = None
        match = "socket cannot be deadline-bounded"
    transport, _ = _deadline_transport(
        monkeypatch=monkeypatch,
        clock=clock,
        response=response,
        deadline=105.0,
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match=match):
        transport.fetch(url)

    assert transport.response_bytes == 0


def test_owned_transport_rejects_reordered_url_and_expired_batch_without_network() -> None:
    reordered = acquirer._OwnedYahooChartTransport()
    with pytest.raises(acquirer.MarketAcquisitionError, match="non-canonical"):
        reordered.fetch(acquirer._canonical_url("SPY"))

    expired = acquirer._OwnedYahooChartTransport()
    expired._deadline = -1.0
    with pytest.raises(acquirer.MarketAcquisitionError, match="deadline"):
        expired.fetch(acquirer._canonical_url("AAPL"))


def _validate_bundle(bundle: dict) -> dict:
    return acquirer.validate_development_market_acquisition_bundle(
        bundle,
        expected_acquisition_plan_sha256=bundle["acquisition_plan_sha256"],
        expected_acquisition_receipt_sha256=bundle[
            "acquisition_receipt_sha256"
        ],
        expected_bundle_sha256=bundle["bundle_sha256"],
    )


def test_pure_validator_replays_provider_bytes_to_stage_without_source_io(
    acquisition_bundle: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        acquirer,
        "_source_sha256s",
        lambda: (_ for _ in ()).throw(AssertionError("validator performed source I/O")),
    )

    validation = _validate_bundle(acquisition_bundle)

    assert validation["provider_response_to_stage_replayed"] is True
    assert validation["exact_raw_bytes_replayed"] is True
    assert validation["production_authorized"] is False
    assert len(validation["validation_sha256"]) == 64


@pytest.mark.parametrize(
    "target",
    ("raw", "request_receipt", "snapshot", "source_manifest", "stage_manifest"),
)
def test_validator_rejects_independent_evidence_mutations(
    acquisition_bundle: dict,
    target: str,
) -> None:
    changed = copy.deepcopy(acquisition_bundle)
    if target == "raw":
        changed["raw_response_bytes_by_symbol"]["SPY"] += b" "
    elif target == "request_receipt":
        changed["acquisition_receipt"]["requests"][1]["accepted_row_count"] += 1
    elif target == "snapshot":
        changed["artifact_bytes_by_symbol"]["SPY"] += b" "
    elif target == "source_manifest":
        changed["source_manifest"]["source_family"] = "changed"
    else:
        changed["stage_manifest"]["row_count"] += 1

    with pytest.raises(acquirer.MarketAcquisitionError):
        acquirer.validate_development_market_acquisition_bundle(
            changed,
            expected_acquisition_plan_sha256=acquisition_bundle[
                "acquisition_plan_sha256"
            ],
            expected_acquisition_receipt_sha256=acquisition_bundle[
                "acquisition_receipt_sha256"
            ],
            expected_bundle_sha256=acquisition_bundle["bundle_sha256"],
        )


def test_validator_rejects_coherently_rehashed_raw_price_without_new_snapshots(
    acquisition_bundle: dict,
) -> None:
    changed = copy.deepcopy(acquisition_bundle)
    raw_object = json.loads(changed["raw_response_bytes_by_symbol"]["SPY"])
    result = raw_object["chart"]["result"][0]
    result["indicators"]["quote"][0]["close"][-1] = 100.5
    result["indicators"]["adjclose"][0]["adjclose"][-1] = 100.5
    changed_raw = _bytes(raw_object)
    changed["raw_response_bytes_by_symbol"]["SPY"] = changed_raw
    parsed = acquirer._parse_yahoo_chart_response(symbol="SPY", raw_bytes=changed_raw)
    old_request = changed["acquisition_receipt"]["requests"][1]
    changed["acquisition_receipt"]["requests"][1] = acquirer._request_receipt(
        symbol="SPY",
        raw_bytes=changed_raw,
        parsed=parsed,
        charset=old_request["charset"],
        content_encoding=old_request["content_encoding"],
        declared_content_length=len(changed_raw),
    )
    raw_set = [
        {
            "symbol": symbol,
            "sha256": hashlib.sha256(
                changed["raw_response_bytes_by_symbol"][symbol]
            ).hexdigest(),
            "bytes": len(changed["raw_response_bytes_by_symbol"][symbol]),
        }
        for symbol in MARKET_SYMBOLS
    ]
    receipt = changed["acquisition_receipt"]
    receipt["raw_response_total_bytes"] = sum(item["bytes"] for item in raw_set)
    receipt["raw_response_set_sha256"] = canonical_sha256(raw_set)
    receipt_body = {
        key: value for key, value in receipt.items() if key != "acquisition_receipt_sha256"
    }
    receipt["acquisition_receipt_sha256"] = canonical_sha256(receipt_body)
    changed["acquisition_receipt_sha256"] = receipt[
        "acquisition_receipt_sha256"
    ]
    changed["raw_response_sha256s"]["SPY"] = hashlib.sha256(changed_raw).hexdigest()
    compact_keys = (
        "schema_version",
        "artifact_stage",
        "acquisition_receipt_sha256",
        "acquisition_plan_sha256",
        "source_manifest_sha256",
        "market_stage_manifest_sha256",
        "source_reconciliation_sha256",
        "raw_response_sha256s",
        "artifact_sha256s",
        "window_sha256s",
    )
    changed["bundle_sha256"] = canonical_sha256(
        {key: changed[key] for key in compact_keys}
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match="snapshot bytes"):
        acquirer.validate_development_market_acquisition_bundle(
            changed,
            expected_acquisition_plan_sha256=changed["acquisition_plan_sha256"],
            expected_acquisition_receipt_sha256=changed[
                "acquisition_receipt_sha256"
            ],
            expected_bundle_sha256=changed["bundle_sha256"],
        )


def test_validator_rejects_switching_mapping_containers(
    acquisition_bundle: dict,
) -> None:
    class SwitchingDict(dict):
        pass

    with pytest.raises(acquirer.MarketAcquisitionError, match="plain dict"):
        acquirer.validate_development_market_acquisition_bundle(
            SwitchingDict(acquisition_bundle),
            expected_acquisition_plan_sha256=acquisition_bundle[
                "acquisition_plan_sha256"
            ],
            expected_acquisition_receipt_sha256=acquisition_bundle[
                "acquisition_receipt_sha256"
            ],
            expected_bundle_sha256=acquisition_bundle["bundle_sha256"],
        )

    changed = dict(acquisition_bundle)
    changed["artifact_bytes_by_symbol"] = SwitchingDict(
        acquisition_bundle["artifact_bytes_by_symbol"]
    )
    with pytest.raises(acquirer.MarketAcquisitionError, match="six symbols"):
        _validate_bundle(changed)


def test_source_hash_change_changes_plan_and_toctou_fails_before_normalization(
    acquisition_bundle: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_plan = acquirer.build_development_market_acquisition_plan()
    changed_sources = dict(original_plan["source_code_sha256s"])
    changed_sources["agent_benchmark/sec_filing_gemma_market_acquirer.py"] = "0" * 64
    changed_plan = acquirer._build_development_market_acquisition_plan(changed_sources)
    assert changed_plan["acquisition_plan_sha256"] != original_plan[
        "acquisition_plan_sha256"
    ]
    plans = iter((original_plan, changed_plan))
    monkeypatch.setattr(
        acquirer,
        "build_development_market_acquisition_plan",
        lambda: next(plans),
    )

    with pytest.raises(acquirer.MarketAcquisitionError, match="changed across"):
        acquirer._acquire_development_market_evidence_with_transport(
            _FakeTransport(acquisition_bundle["raw_response_bytes_by_symbol"])
        )


def test_index_timezones_are_frozen_to_chicago_and_utc_date_shift_fails() -> None:
    valid = _response_object("VIX", ("2018-12-28",))
    parsed = acquirer._parse_yahoo_chart_response(
        symbol="VIX",
        raw_bytes=_bytes(valid),
    )
    assert list(parsed.rows_by_session) == ["2018-12-28"]

    wrong_zone = copy.deepcopy(valid)
    wrong_zone["chart"]["result"][0]["meta"][
        "exchangeTimezoneName"
    ] = "America/New_York"
    with pytest.raises(acquirer.MarketAcquisitionError, match="provider timezone"):
        acquirer._parse_yahoo_chart_response(
            symbol="VIX",
            raw_bytes=_bytes(wrong_zone),
        )

    date_shift = copy.deepcopy(valid)
    date_shift["chart"]["result"][0]["timestamp"][0] = int(
        datetime(2018, 12, 28, 0, 30, tzinfo=timezone.utc).timestamp()
    )
    with pytest.raises(acquirer.MarketAcquisitionError, match="changes date"):
        acquirer._parse_yahoo_chart_response(
            symbol="VIX",
            raw_bytes=_bytes(date_shift),
        )
