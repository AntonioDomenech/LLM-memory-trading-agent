"""Owned, fixed-provider market acquisition for the SEC/Gemma development stage.

The owned effect entry point in this module has no caller-controlled provider,
URL, query, symbol, date, row, or transport input.  It performs exactly one
Yahoo Finance Chart-v8 request for each frozen market symbol, preserves the
exact UTF-8 response bytes for immediate store quarantine, and deterministically
produces the canonical market snapshots and manifests used by the pure
evidence modules.  It is intentionally private: raw Yahoo metadata can include
current quote fields even when the requested historical window ends in 2018.

This component is deliberately development-only.  Yahoo adjusted-close
history may be back-adjusted later, so this receipt freezes one historical
prefix; it does not claim that a future download will be byte-compatible with
that prefix.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import ssl
from threading import Lock
import time
from types import ModuleType
from typing import Any, Final, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)
from weakref import WeakKeyDictionary
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import agent_benchmark.sec_filing_gemma_contract as _contract_source
import agent_benchmark.sec_filing_gemma_market_evidence as _market_evidence_source
import agent_benchmark.sec_filing_gemma_market_source_bytes as _source_bytes_source
import agent_benchmark.sec_session_calendar as _calendar_source
from agent_benchmark.sec_filing_gemma_contract import canonical_sha256
from agent_benchmark.sec_filing_gemma_market_evidence import (
    MARKET_FIELDS,
    MARKET_SYMBOLS,
    PRICE_FIELDS,
    build_market_source_manifest,
    build_market_stage_manifest,
    encode_float_hex,
)
from agent_benchmark.sec_filing_gemma_market_source_bytes import (
    build_market_source_snapshot_bytes,
    validate_market_source_bytes_against_stage,
)
from agent_benchmark.sec_session_calendar import EXPECTED_MARKET_HISTORY_SESSIONS


MARKET_ACQUISITION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-yahoo-development-acquisition-v1"
)
MARKET_ACQUISITION_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-yahoo-development-acquisition-plan-v1"
)
MARKET_ACQUISITION_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-yahoo-development-bundle-v1"
)
MARKET_ACQUISITION_VALIDATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-yahoo-development-validation-v1"
)
YAHOO_CHART_PROVIDER_FAMILY: Final[str] = (
    "yahoo-finance-chart-v8-public-unauthenticated"
)
YAHOO_CHART_ENDPOINT: Final[str] = (
    "https://query1.finance.yahoo.com/v8/finance/chart"
)
YAHOO_PROVIDER_SYMBOLS: Final[dict[str, str]] = {
    "AAPL": "AAPL",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "VIX": "^VIX",
    "TNX": "^TNX",
}
YAHOO_PROVIDER_TIMEZONES: Final[dict[str, str]] = {
    "AAPL": "America/New_York",
    "SPY": "America/New_York",
    "QQQ": "America/New_York",
    "IWM": "America/New_York",
    "VIX": "America/Chicago",
    "TNX": "America/Chicago",
}
DEVELOPMENT_REQUEST_START: Final[str] = "1998-01-01"
DEVELOPMENT_REQUEST_END_EXCLUSIVE: Final[str] = "2019-01-01"
DEVELOPMENT_LAST_SESSION: Final[str] = "2018-12-31"
DEVELOPMENT_PERIOD1_UTC: Final[int] = 883_612_800
DEVELOPMENT_PERIOD2_UTC: Final[int] = 1_546_300_800
YAHOO_INTERVAL: Final[str] = "1d"
YAHOO_REQUEST_TIMEOUT_SECONDS: Final[int] = 30
YAHOO_MAX_RESPONSE_BYTES: Final[int] = 64 * 1024 * 1024
YAHOO_MAX_TOTAL_RESPONSE_BYTES: Final[int] = 128 * 1024 * 1024
YAHOO_MAX_BATCH_SECONDS: Final[int] = 210
YAHOO_REQUEST_COUNT: Final[int] = len(MARKET_SYMBOLS)
_YAHOO_BODY_READ_CHUNK_BYTES: Final[int] = 64 * 1024
YAHOO_USER_AGENT: Final[str] = (
    "LLM-memory-trading-agent/1.0 market-evidence (no-auth; one-shot)"
)

_QUERY_ITEMS: Final[tuple[tuple[str, str], ...]] = (
    ("period1", str(DEVELOPMENT_PERIOD1_UTC)),
    ("period2", str(DEVELOPMENT_PERIOD2_UTC)),
    ("interval", YAHOO_INTERVAL),
    ("includePrePost", "false"),
    ("includeAdjustedClose", "true"),
    ("events", "div,splits"),
)


CONTEXT_FIRST_ACCEPTED_SESSION: Final[dict[str, str]] = {
    "SPY": "1998-01-02",
    "QQQ": "1999-03-10",
    "IWM": "2000-05-26",
    "VIX": "1998-01-02",
    "TNX": "1998-01-02",
}
CONTEXT_ALLOWED_MISSING_SESSIONS: Final[dict[str, tuple[str, ...]]] = {
    "SPY": (),
    "QQQ": (),
    "IWM": (),
    "VIX": (),
    "TNX": (
        "1998-10-12",
        "1998-11-11",
        "1999-10-11",
        "1999-11-11",
        "2003-11-11",
        "2005-10-10",
        "2005-11-11",
        "2006-10-09",
        "2010-10-11",
        "2016-11-11",
    ),
}
_RAW_RESPONSE_KEYS = {"chart"}
_CHART_KEYS = {"result", "error"}
_REQUIRED_RESULT_KEYS = {"meta", "timestamp", "indicators"}
_REQUIRED_META_KEYS = {"symbol", "exchangeTimezoneName", "dataGranularity"}
_INDICATOR_KEYS = {"quote", "adjclose"}
_QUOTE_KEYS = {"open", "high", "low", "close", "volume"}
_ADJCLOSE_KEYS = {"adjclose"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_PATHS: Final[tuple[str, ...]] = (
    "agent_benchmark/sec_filing_gemma_market_acquirer.py",
    "agent_benchmark/sec_filing_gemma_market_evidence.py",
    "agent_benchmark/sec_filing_gemma_market_source_bytes.py",
    "agent_benchmark/sec_filing_gemma_contract.py",
    "agent_benchmark/sec_session_calendar.py",
)


class MarketAcquisitionError(RuntimeError):
    """Raised when provider bytes or acquisition provenance fail closed."""


@dataclass(frozen=True)
class _TransportResponse:
    """Exact response material returned by the private transport boundary."""

    request_url: str
    final_url: str
    status_code: int
    content_type: str | None
    charset: str | None
    content_encoding: str | None
    declared_content_length: int | None
    body: bytes


class _TransportLike(Protocol):
    def fetch(self, url: str) -> _TransportResponse: ...


@dataclass(frozen=True)
class _ParsedSymbol:
    rows_by_session: dict[str, dict[str, float]]
    explicit_absent_sessions: tuple[str, ...]
    timestamp_count: int
    first_provider_session: str | None
    last_provider_session: str | None
    provider_timestamps: tuple[int, ...]


class _RejectRedirectHandler(HTTPRedirectHandler):
    """Reject every redirect rather than inheriting urllib redirect behavior."""

    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise MarketAcquisitionError("Yahoo market transport rejected a redirect")


def _exact_response_header_values(headers: Any, name: str) -> list[str]:
    """Return one HTTP header's exact occurrences or fail closed."""

    get_all = getattr(headers, "get_all", None)
    if not callable(get_all):
        raise MarketAcquisitionError(
            "Yahoo market response headers lack exact occurrence access"
        )
    values = get_all(name, [])
    if type(values) is not list or any(type(value) is not str for value in values):
        raise MarketAcquisitionError(
            f"Yahoo market {name} headers are not exact text occurrences"
        )
    return values


def _response_body_framing(headers: Any) -> int | None:
    """Reject ambiguous response framing before any provider body is read."""

    content_lengths = _exact_response_header_values(headers, "Content-Length")
    transfer_encodings = _exact_response_header_values(headers, "Transfer-Encoding")
    if len(content_lengths) > 1 or len(transfer_encodings) > 1:
        raise MarketAcquisitionError(
            "Yahoo market response has duplicate body-framing headers"
        )
    if content_lengths and transfer_encodings:
        raise MarketAcquisitionError(
            "Yahoo market response has conflicting body-framing headers"
        )
    if transfer_encodings:
        raise MarketAcquisitionError(
            "Yahoo market response uses an unsupported transfer coding"
        )
    if not content_lengths:
        return None
    content_length_text = content_lengths[0]
    if not content_length_text.isascii() or not content_length_text.isdigit():
        raise MarketAcquisitionError(
            "Yahoo market Content-Length is not a canonical integer"
        )
    declared_content_length = int(content_length_text)
    if declared_content_length > YAHOO_MAX_RESPONSE_BYTES:
        raise MarketAcquisitionError(
            "Yahoo market response exceeded the fixed byte ceiling"
        )
    return declared_content_length


def _read_owned_yahoo_response_body(
    response: Any,
    *,
    absolute_deadline: float,
    declared_content_length: int | None,
) -> bytes:
    """Read one owned HTTP response without a slow-drip deadline bypass."""

    read1 = getattr(response, "read1", None)
    if not callable(read1):
        raise MarketAcquisitionError(
            "Yahoo market response lacks bounded single-read support"
        )
    body = bytearray()
    while True:
        remaining = absolute_deadline - time.monotonic()
        if remaining <= 0:
            raise MarketAcquisitionError(
                "Yahoo market response body crossed its deadline"
            )
        fp = getattr(response, "fp", None)
        if fp is None:
            if body:
                return bytes(body)
            raise MarketAcquisitionError(
                "Yahoo market response socket cannot be deadline-bounded"
            )
        raw = getattr(fp, "raw", None)
        sock = getattr(raw, "_sock", None)
        settimeout = getattr(sock, "settimeout", None)
        if not callable(settimeout):
            raise MarketAcquisitionError(
                "Yahoo market response socket cannot be deadline-bounded"
            )
        try:
            settimeout(remaining)
        except (OSError, TypeError, ValueError) as exc:
            raise MarketAcquisitionError(
                "Yahoo market response socket cannot be deadline-bounded"
            ) from exc
        read_size = min(
            _YAHOO_BODY_READ_CHUNK_BYTES,
            YAHOO_MAX_RESPONSE_BYTES + 1 - len(body),
        )
        try:
            chunk = read1(read_size)
        except TimeoutError as exc:
            raise MarketAcquisitionError(
                "Yahoo market response body crossed its deadline"
            ) from exc
        if time.monotonic() >= absolute_deadline:
            raise MarketAcquisitionError(
                "Yahoo market response body crossed its deadline"
            )
        if type(chunk) is not bytes or len(chunk) > read_size:
            raise MarketAcquisitionError(
                "Yahoo market response returned an invalid bounded chunk"
            )
        if not chunk:
            return bytes(body)
        body.extend(chunk)
        if len(body) > YAHOO_MAX_RESPONSE_BYTES:
            raise MarketAcquisitionError(
                "Yahoo market response exceeded the fixed byte ceiling"
            )
        if (
            declared_content_length is not None
            and len(body) > declared_content_length
        ):
            raise MarketAcquisitionError(
                "Yahoo market Content-Length does not match exact bytes"
            )


class _OwnedYahooChartTransport:
    """One-shot HTTPS transport with no proxy, cookie, auth, cache, or retry."""

    def __init__(self) -> None:
        context = ssl.create_default_context()
        self._opener = build_opener(
            ProxyHandler({}),
            _RejectRedirectHandler(),
            HTTPSHandler(context=context),
        )
        self.request_count = 0
        self.response_bytes = 0
        self._deadline = time.monotonic() + YAHOO_MAX_BATCH_SECONDS

    def fetch(self, url: str) -> _TransportResponse:
        request_started = time.monotonic()
        if request_started >= self._deadline:
            raise MarketAcquisitionError("Yahoo market batch crossed its deadline")
        if self.request_count >= YAHOO_REQUEST_COUNT or url != _canonical_url(
            MARKET_SYMBOLS[self.request_count]
        ):
            raise MarketAcquisitionError(
                "Yahoo transport received a non-canonical or reordered URL"
            )
        self.request_count += 1
        request = Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Cache-Control": "no-store",
                "User-Agent": YAHOO_USER_AGENT,
            },
        )
        absolute_deadline = min(
            self._deadline,
            request_started + YAHOO_REQUEST_TIMEOUT_SECONDS,
        )
        open_timeout = absolute_deadline - time.monotonic()
        if open_timeout <= 0:
            raise MarketAcquisitionError("Yahoo market batch crossed its deadline")
        try:
            with self._opener.open(
                request,
                timeout=open_timeout,
            ) as response:
                headers = response.headers
                declared_content_length = _response_body_framing(headers)
                body = _read_owned_yahoo_response_body(
                    response,
                    absolute_deadline=absolute_deadline,
                    declared_content_length=declared_content_length,
                )
                if declared_content_length is not None:
                    if declared_content_length != len(body):
                        raise MarketAcquisitionError(
                            "Yahoo market Content-Length does not match exact bytes"
                        )
                if self.response_bytes + len(body) > YAHOO_MAX_TOTAL_RESPONSE_BYTES:
                    raise MarketAcquisitionError(
                        "Yahoo market batch exceeded the aggregate byte ceiling"
                    )
                self.response_bytes += len(body)
                if time.monotonic() > self._deadline:
                    raise MarketAcquisitionError(
                        "Yahoo market batch crossed its deadline"
                    )
                return _TransportResponse(
                    request_url=url,
                    final_url=response.geturl(),
                    status_code=response.getcode(),
                    content_type=headers.get_content_type(),
                    charset=headers.get_content_charset(),
                    content_encoding=headers.get("Content-Encoding"),
                    declared_content_length=declared_content_length,
                    body=body,
                )
        except MarketAcquisitionError:
            raise
        except HTTPError as exc:
            raise MarketAcquisitionError(
                f"Yahoo market request failed with HTTP status {exc.code}"
            ) from exc
        except (URLError, OSError, TimeoutError) as exc:
            raise MarketAcquisitionError("Yahoo market request failed") from exc


def _reject_duplicate_pairs(location: str):
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MarketAcquisitionError(
                    f"{location} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    return hook


def _finite_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise MarketAcquisitionError("Yahoo response contains a non-finite number")
    return value


def _strict_json_bytes(payload: bytes, location: str) -> Mapping[str, Any]:
    if type(payload) is not bytes:
        raise TypeError(f"{location} must be exact bytes")
    if not payload or len(payload) > YAHOO_MAX_RESPONSE_BYTES:
        raise MarketAcquisitionError(f"{location} is empty or oversized")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs(location),
            parse_float=_finite_json_float,
            parse_constant=lambda token: (_ for _ in ()).throw(
                MarketAcquisitionError(
                    f"{location} contains a non-finite JSON constant: {token}"
                )
            ),
        )
    except MarketAcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MarketAcquisitionError(f"{location} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise MarketAcquisitionError(f"{location} must contain one JSON object")
    return value


def _expect_exact_keys(value: Any, keys: set[str], location: str) -> Mapping[str, Any]:
    if type(value) is not dict or set(value) != keys or not all(
        type(key) is str for key in value
    ):
        raise MarketAcquisitionError(f"{location} has unexpected or missing keys")
    return value


def _expect_required_keys(
    value: Any,
    keys: set[str],
    location: str,
) -> Mapping[str, Any]:
    if type(value) is not dict or not keys.issubset(value) or not all(
        type(key) is str for key in value
    ):
        raise MarketAcquisitionError(f"{location} is missing required keys")
    return value


def _expect_one_object(value: Any, location: str) -> Mapping[str, Any]:
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        raise MarketAcquisitionError(f"{location} must contain exactly one object")
    return value[0]


def _as_finite_float(value: Any, location: str) -> float:
    if isinstance(value, bool) or type(value) not in {int, float}:
        raise MarketAcquisitionError(f"{location} must be a JSON number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise MarketAcquisitionError(f"{location} is outside binary64") from exc
    if not math.isfinite(normalized):
        raise MarketAcquisitionError(f"{location} must be finite")
    return normalized


def _validate_ohlcv(values: Mapping[str, float], location: str) -> None:
    if any(values[name] <= 0.0 for name in PRICE_FIELDS):
        raise MarketAcquisitionError(f"{location} prices must be positive")
    if values["volume"] < 0.0:
        raise MarketAcquisitionError(f"{location} volume cannot be negative")
    if values["high"] < max(values["open"], values["low"], values["close"]):
        raise MarketAcquisitionError(f"{location} high is inconsistent with OHLC")
    if values["low"] > min(values["open"], values["high"], values["close"]):
        raise MarketAcquisitionError(f"{location} low is inconsistent with OHLC")


def _provider_session(
    timestamp: int,
    *,
    expected_timezone: str,
    location: str,
) -> str:
    if isinstance(timestamp, bool) or type(timestamp) is not int:
        raise MarketAcquisitionError(f"{location} must be an integer Unix timestamp")
    if timestamp < DEVELOPMENT_PERIOD1_UTC or timestamp >= DEVELOPMENT_PERIOD2_UTC:
        raise MarketAcquisitionError(f"{location} is outside the frozen request window")
    try:
        exchange_zone = ZoneInfo(expected_timezone)
    except ZoneInfoNotFoundError as exc:
        raise MarketAcquisitionError(
            f"The frozen {expected_timezone} timezone database is unavailable"
        ) from exc
    try:
        utc_observed = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        observed = utc_observed.astimezone(exchange_zone)
    except (OverflowError, OSError, ValueError) as exc:
        raise MarketAcquisitionError(f"{location} is not a valid Unix timestamp") from exc
    if observed.date() != utc_observed.date():
        raise MarketAcquisitionError(
            f"{location} changes date between UTC and the frozen exchange timezone"
        )
    return observed.date().isoformat()


def _parse_yahoo_chart_response(
    *,
    symbol: str,
    raw_bytes: bytes,
) -> _ParsedSymbol:
    """Parse one exact Yahoo response without accepting caller-derived rows."""

    if symbol not in MARKET_SYMBOLS:
        raise MarketAcquisitionError("Market symbol is not frozen")
    provider_symbol = YAHOO_PROVIDER_SYMBOLS[symbol]
    provider_timezone = YAHOO_PROVIDER_TIMEZONES[symbol]
    root = _expect_exact_keys(
        _strict_json_bytes(raw_bytes, f"{symbol} raw response"),
        _RAW_RESPONSE_KEYS,
        f"{symbol} raw response",
    )
    chart = _expect_exact_keys(root["chart"], _CHART_KEYS, f"{symbol}.chart")
    if chart["error"] is not None:
        raise MarketAcquisitionError(f"{symbol} provider returned a chart error")
    result = _expect_one_object(chart["result"], f"{symbol}.chart.result")
    result = _expect_required_keys(result, _REQUIRED_RESULT_KEYS, f"{symbol}.result")
    meta = _expect_required_keys(result["meta"], _REQUIRED_META_KEYS, f"{symbol}.meta")
    if meta["symbol"] != provider_symbol:
        raise MarketAcquisitionError(f"{symbol} provider symbol changed")
    if meta["exchangeTimezoneName"] != provider_timezone:
        raise MarketAcquisitionError(f"{symbol} provider timezone changed")
    if meta["dataGranularity"] != YAHOO_INTERVAL:
        raise MarketAcquisitionError(f"{symbol} provider interval changed")

    timestamps = result["timestamp"]
    if type(timestamps) is not list:
        raise MarketAcquisitionError(f"{symbol}.timestamp must be an array")
    indicators = _expect_exact_keys(
        result["indicators"], _INDICATOR_KEYS, f"{symbol}.indicators"
    )
    quote_values = _expect_exact_keys(
        _expect_one_object(indicators["quote"], f"{symbol}.indicators.quote"),
        _QUOTE_KEYS,
        f"{symbol}.quote",
    )
    adjusted_values = _expect_exact_keys(
        _expect_one_object(
            indicators["adjclose"], f"{symbol}.indicators.adjclose"
        ),
        _ADJCLOSE_KEYS,
        f"{symbol}.adjclose",
    )
    arrays: dict[str, list[Any]] = {
        "open": quote_values["open"],
        "high": quote_values["high"],
        "low": quote_values["low"],
        "close": quote_values["close"],
        "volume": quote_values["volume"],
        "adjusted_close": adjusted_values["adjclose"],
    }
    if any(type(values) is not list for values in arrays.values()) or any(
        len(values) != len(timestamps) for values in arrays.values()
    ):
        raise MarketAcquisitionError(
            f"{symbol} timestamps and indicator arrays are misaligned"
        )

    frozen_sessions = tuple(
        session
        for session in EXPECTED_MARKET_HISTORY_SESSIONS
        if DEVELOPMENT_REQUEST_START <= session <= DEVELOPMENT_LAST_SESSION
    )
    frozen_set = set(frozen_sessions)
    rows: dict[str, dict[str, float]] = {}
    explicit_absent: list[str] = []
    previous_timestamp: int | None = None
    previous_session: str | None = None
    provider_sessions: list[str] = []
    for index, timestamp in enumerate(timestamps):
        if isinstance(timestamp, bool) or type(timestamp) is not int:
            raise MarketAcquisitionError(
                f"{symbol}.timestamp[{index}] must be an integer"
            )
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise MarketAcquisitionError(
                f"{symbol} timestamps are duplicated or reordered"
            )
        previous_timestamp = timestamp
        session = _provider_session(
            timestamp,
            expected_timezone=provider_timezone,
            location=f"{symbol}.timestamp[{index}]",
        )
        if session not in frozen_set:
            raise MarketAcquisitionError(
                f"{symbol} provider row is not a frozen development session"
            )
        if previous_session is not None and session <= previous_session:
            raise MarketAcquisitionError(
                f"{symbol} sessions are duplicated or reordered"
            )
        previous_session = session
        provider_sessions.append(session)
        values = {name: arrays[name][index] for name in MARKET_FIELDS}
        null_count = sum(value is None for value in values.values())
        if null_count == len(MARKET_FIELDS):
            explicit_absent.append(session)
            continue
        if null_count:
            raise MarketAcquisitionError(
                f"{symbol} row {session} is partial rather than explicitly absent"
            )
        normalized = {
            name: _as_finite_float(values[name], f"{symbol}.{session}.{name}")
            for name in MARKET_FIELDS
        }
        _validate_ohlcv(normalized, f"{symbol}.{session}")
        rows[session] = normalized

    if symbol == "AAPL":
        if explicit_absent or tuple(rows) != frozen_sessions:
            raise MarketAcquisitionError(
                "AAPL provider rows must cover every frozen development session"
            )
    return _ParsedSymbol(
        rows_by_session=rows,
        explicit_absent_sessions=tuple(explicit_absent),
        timestamp_count=len(timestamps),
        first_provider_session=provider_sessions[0] if provider_sessions else None,
        last_provider_session=provider_sessions[-1] if provider_sessions else None,
        provider_timestamps=tuple(timestamps),
    )


def _validate_symbol_coverage(symbol: str, parsed: _ParsedSymbol) -> None:
    """Reject truncated or silently sparse post-inception context history."""

    if symbol == "AAPL":
        return
    first_expected = CONTEXT_FIRST_ACCEPTED_SESSION[symbol]
    accepted = tuple(parsed.rows_by_session)
    if (
        not accepted
        or accepted[0] != first_expected
        or accepted[-1] != DEVELOPMENT_LAST_SESSION
    ):
        raise MarketAcquisitionError(
            f"{symbol} context coverage has the wrong first or final accepted session"
        )
    required_sessions = {
        session
        for session in EXPECTED_MARKET_HISTORY_SESSIONS
        if first_expected <= session <= DEVELOPMENT_LAST_SESSION
    }
    missing = required_sessions - set(accepted)
    permitted_missing = set(CONTEXT_ALLOWED_MISSING_SESSIONS[symbol])
    if not missing.issubset(permitted_missing):
        raise MarketAcquisitionError(
            f"{symbol} context coverage contains an unexplained post-inception gap"
        )
    explicit_post_inception = {
        session
        for session in parsed.explicit_absent_sessions
        if session >= first_expected
    }
    if not explicit_post_inception.issubset(permitted_missing):
        raise MarketAcquisitionError(
            f"{symbol} explicitly absent post-inception rows are not allowlisted"
        )


def _canonical_url(symbol: str) -> str:
    provider_symbol = YAHOO_PROVIDER_SYMBOLS[symbol]
    path_symbol = quote(provider_symbol, safe="")
    return f"{YAHOO_CHART_ENDPOINT}/{path_symbol}?{urlencode(_QUERY_ITEMS)}"


def _validate_transport_response(
    response: Any,
    *,
    expected_url: str,
    symbol: str,
) -> bytes:
    if type(response) is not _TransportResponse:
        raise MarketAcquisitionError(
            "Yahoo transport returned an unreviewed response container"
        )
    if response.request_url != expected_url or response.final_url != expected_url:
        raise MarketAcquisitionError(f"{symbol} market request was redirected or changed")
    if isinstance(response.status_code, bool) or response.status_code != 200:
        raise MarketAcquisitionError(f"{symbol} market response was not HTTP 200")
    if response.content_type != "application/json":
        raise MarketAcquisitionError(f"{symbol} market response is not JSON")
    if response.charset not in {None, "utf-8", "UTF-8"}:
        raise MarketAcquisitionError(f"{symbol} market response charset changed")
    if response.content_encoding not in {None, "identity"}:
        raise MarketAcquisitionError(f"{symbol} compressed market response is forbidden")
    if type(response.body) is not bytes:
        raise MarketAcquisitionError(f"{symbol} market response body is not exact bytes")
    if not response.body or len(response.body) > YAHOO_MAX_RESPONSE_BYTES:
        raise MarketAcquisitionError(f"{symbol} market response is empty or oversized")
    if (
        response.declared_content_length is not None
        and (
            isinstance(response.declared_content_length, bool)
            or type(response.declared_content_length) is not int
            or response.declared_content_length != len(response.body)
        )
    ):
        raise MarketAcquisitionError(
            f"{symbol} market Content-Length does not match exact bytes"
        )
    return response.body


def _module_sha256(module: ModuleType, expected_name: str) -> str:
    path_value = getattr(module, "__file__", None)
    if not isinstance(path_value, str):
        raise MarketAcquisitionError(f"{expected_name} source path is unavailable")
    path = Path(path_value).resolve()
    if path.suffix != ".py" or not path.is_file():
        raise MarketAcquisitionError(f"{expected_name} source is not an exact Python file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256s() -> dict[str, str]:
    own_path = Path(__file__).resolve()
    result = {
        "agent_benchmark/sec_filing_gemma_market_acquirer.py": hashlib.sha256(
            own_path.read_bytes()
        ).hexdigest(),
        "agent_benchmark/sec_filing_gemma_market_evidence.py": _module_sha256(
            _market_evidence_source, "market evidence"
        ),
        "agent_benchmark/sec_filing_gemma_market_source_bytes.py": _module_sha256(
            _source_bytes_source, "market source bytes"
        ),
        "agent_benchmark/sec_filing_gemma_contract.py": _module_sha256(
            _contract_source, "SEC/Gemma contract"
        ),
        "agent_benchmark/sec_session_calendar.py": _module_sha256(
            _calendar_source, "market session calendar"
        ),
    }
    if any(_SHA256_RE.fullmatch(value) is None for value in result.values()):
        raise MarketAcquisitionError("Market acquisition source hash is invalid")
    return result


def _build_development_market_acquisition_plan(
    source_code_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    if type(source_code_sha256s) is not dict or set(source_code_sha256s) != set(
        _SOURCE_PATHS
    ) or any(
        type(value) is not str or _SHA256_RE.fullmatch(value) is None
        for value in source_code_sha256s.values()
    ):
        raise MarketAcquisitionError(
            "Acquisition plan source hashes are incomplete or invalid"
        )
    body = {
        "schema_version": MARKET_ACQUISITION_PLAN_SCHEMA_VERSION,
        "provider_family": YAHOO_CHART_PROVIDER_FAMILY,
        "artifact_stage": "development",
        "request_window": {
            "start": DEVELOPMENT_REQUEST_START,
            "end_inclusive": DEVELOPMENT_LAST_SESSION,
            "end_exclusive": DEVELOPMENT_REQUEST_END_EXCLUSIVE,
            "period1_utc": DEVELOPMENT_PERIOD1_UTC,
            "period2_utc": DEVELOPMENT_PERIOD2_UTC,
            "timestamp_session_timezones": dict(YAHOO_PROVIDER_TIMEZONES),
        },
        "requests": [
            {
                "symbol": symbol,
                "provider_symbol": YAHOO_PROVIDER_SYMBOLS[symbol],
                "method": "GET",
                "canonical_url": _canonical_url(symbol),
                "canonical_url_sha256": hashlib.sha256(
                    _canonical_url(symbol).encode("ascii")
                ).hexdigest(),
                "ordered_query": [
                    {"name": name, "value": value} for name, value in _QUERY_ITEMS
                ],
                "expected_exchange_timezone": YAHOO_PROVIDER_TIMEZONES[symbol],
            }
            for symbol in MARKET_SYMBOLS
        ],
        "transport_authority": {
            "mode": "owned_hardened_yahoo_chart_v8_https",
            "fixed_request_count": YAHOO_REQUEST_COUNT,
            "one_request_per_symbol": True,
            "fallback_permitted": False,
            "retry_permitted": False,
            "redirect_permitted": False,
            "proxy_permitted": False,
            "cookie_permitted": False,
            "authentication_permitted": False,
            "cache_read_permitted": False,
            "cache_write_permitted": False,
            "compression_permitted": False,
            "paid_api_calls_permitted": 0,
            "estimated_cost_usd": "0.00",
            "response_byte_ceiling_per_symbol": YAHOO_MAX_RESPONSE_BYTES,
            "aggregate_response_byte_ceiling": YAHOO_MAX_TOTAL_RESPONSE_BYTES,
            "timeout_seconds_per_symbol": YAHOO_REQUEST_TIMEOUT_SECONDS,
            "monotonic_batch_deadline_seconds": YAHOO_MAX_BATCH_SECONDS,
        },
        "normalization_authority": {
            "raw_bytes": "exact_strict_utf8_json",
            "json_duplicate_keys_permitted": False,
            "provider_interval": YAHOO_INTERVAL,
            "provider_timezones": dict(YAHOO_PROVIDER_TIMEZONES),
            "numeric_encoding": "python_binary64_then_exact_float_hex",
            "missing_context_encoding": "available_false_all_fields_null",
            "aapl_coverage": "every_frozen_development_session",
            "future_rows_permitted": False,
            "append_compatibility_claimed": False,
            "context_coverage_policy": {
                "first_accepted_sessions": dict(CONTEXT_FIRST_ACCEPTED_SESSION),
                "final_accepted_session": DEVELOPMENT_LAST_SESSION,
                "post_inception_missing_sessions_must_be_allowlisted": True,
                "allowed_missing_sessions": {
                    symbol: list(CONTEXT_ALLOWED_MISSING_SESSIONS[symbol])
                    for symbol in MARKET_SYMBOLS
                    if symbol != "AAPL"
                },
            },
        },
        "source_code_sha256s": dict(source_code_sha256s),
    }
    return {**body, "acquisition_plan_sha256": canonical_sha256(body)}


def build_development_market_acquisition_plan() -> dict[str, Any]:
    """Return the complete pre-effect authority for the owned acquisition.

    A durable store can bind this hash before any network access.  The plan is
    generated without network I/O and hashes the exact reviewed source bytes
    that implement transport, parsing, normalization, calendar, and evidence.
    """

    return _build_development_market_acquisition_plan(_source_sha256s())


def _snapshot_rows(parsed: _ParsedSymbol) -> list[dict[str, str]]:
    return [
        {
            "session": session,
            **{
                f"{field}_hex": encode_float_hex(
                    values[field], f"snapshot.{session}.{field}"
                )
                for field in MARKET_FIELDS
            },
        }
        for session, values in parsed.rows_by_session.items()
    ]


def _request_receipt(
    *,
    symbol: str,
    raw_bytes: bytes,
    parsed: _ParsedSymbol,
    charset: str | None,
    content_encoding: str | None,
    declared_content_length: int | None,
) -> dict[str, Any]:
    development_session_count = sum(
        DEVELOPMENT_REQUEST_START <= session <= DEVELOPMENT_LAST_SESSION
        for session in EXPECTED_MARKET_HISTORY_SESSIONS
    )
    url = _canonical_url(symbol)
    return {
        "symbol": symbol,
        "provider_symbol": YAHOO_PROVIDER_SYMBOLS[symbol],
        "expected_exchange_timezone": YAHOO_PROVIDER_TIMEZONES[symbol],
        "method": "GET",
        "canonical_url": url,
        "canonical_url_sha256": hashlib.sha256(url.encode("ascii")).hexdigest(),
        "ordered_query": [
            {"name": name, "value": value} for name, value in _QUERY_ITEMS
        ],
        "http_status": 200,
        "content_type": "application/json",
        "charset": charset,
        "content_encoding": content_encoding,
        "declared_content_length": declared_content_length,
        "raw_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "raw_response_bytes": len(raw_bytes),
        "provider_timestamp_count": parsed.timestamp_count,
        "provider_timestamps_sha256": canonical_sha256(
            list(parsed.provider_timestamps)
        ),
        "accepted_row_count": len(parsed.rows_by_session),
        "accepted_sessions_sha256": canonical_sha256(
            list(parsed.rows_by_session)
        ),
        "explicit_absent_row_count": len(parsed.explicit_absent_sessions),
        "implicit_missing_session_count": (
            development_session_count
            - len(parsed.rows_by_session)
            - len(parsed.explicit_absent_sessions)
        ),
        "first_provider_session": parsed.first_provider_session,
        "last_provider_session": parsed.last_provider_session,
        "explicit_absent_sessions_sha256": canonical_sha256(
            list(parsed.explicit_absent_sessions)
        ),
        "normalized_snapshot_rows_sha256": canonical_sha256(_snapshot_rows(parsed)),
    }


def _receipt_normalization_contract() -> dict[str, Any]:
    return {
        "raw_bytes": "exact_strict_utf8_json",
        "json_duplicate_keys_permitted": False,
        "numeric_encoding": "python_binary64_then_exact_float_hex",
        "missing_context_encoding": "available_false_all_fields_null",
        "aapl_coverage": "every_frozen_development_session",
        "future_rows_permitted": False,
        "append_compatibility_claimed": False,
        "context_coverage_policy": {
            "first_accepted_sessions": dict(CONTEXT_FIRST_ACCEPTED_SESSION),
            "final_accepted_session": DEVELOPMENT_LAST_SESSION,
            "post_inception_missing_sessions_must_be_allowlisted": True,
            "allowed_missing_sessions": {
                symbol: list(CONTEXT_ALLOWED_MISSING_SESSIONS[symbol])
                for symbol in MARKET_SYMBOLS
                if symbol != "AAPL"
            },
        },
        "adjusted_close_mutability_warning": (
            "frozen_prefix_only_future_yahoo_download_may_be_back_adjusted"
        ),
    }


def _receipt_transport_policy(*, trusted_transport: bool) -> dict[str, Any]:
    return {
        "mode": (
            "owned_hardened_yahoo_chart_v8_https"
            if trusted_transport
            else "untrusted_injected_test_transport"
        ),
        "trusted_production_transport": trusted_transport,
        "authorizes_production_use": False,
        "fresh_network_provenance_claimed": False,
        "network_occurrence_is_only_locally_observed": True,
        "fixed_request_count": YAHOO_REQUEST_COUNT,
        "actual_request_count": YAHOO_REQUEST_COUNT,
        "one_request_per_symbol": True,
        "fallback_permitted": False,
        "fallback_count": 0,
        "retry_permitted": False,
        "retry_count": 0,
        "redirect_permitted": False,
        "redirect_count": 0,
        "proxy_permitted": False,
        "cookie_permitted": False,
        "authentication_permitted": False,
        "cache_read_permitted": False,
        "cache_write_permitted": False,
        "compression_permitted": False,
        "paid_api_calls": 0,
        "estimated_cost_usd": "0.00",
        "response_byte_ceiling_per_symbol": YAHOO_MAX_RESPONSE_BYTES,
        "aggregate_response_byte_ceiling": YAHOO_MAX_TOTAL_RESPONSE_BYTES,
        "timeout_seconds_per_symbol": YAHOO_REQUEST_TIMEOUT_SECONDS,
        "monotonic_batch_deadline_seconds": YAHOO_MAX_BATCH_SECONDS,
    }


def _stage_rows(
    parsed_by_symbol: Mapping[str, _ParsedSymbol],
) -> list[dict[str, Any]]:
    sessions = tuple(
        session
        for session in EXPECTED_MARKET_HISTORY_SESSIONS
        if DEVELOPMENT_REQUEST_START <= session <= DEVELOPMENT_LAST_SESSION
    )
    result: list[dict[str, Any]] = []
    for session in sessions:
        observations: dict[str, dict[str, Any]] = {}
        for symbol in MARKET_SYMBOLS:
            values = parsed_by_symbol[symbol].rows_by_session.get(session)
            observations[symbol] = (
                {
                    "available": True,
                    **{field: values[field] for field in MARKET_FIELDS},
                }
                if values is not None
                else {
                    "available": False,
                    **{field: None for field in MARKET_FIELDS},
                }
            )
        result.append({"session": session, "observations": observations})
    return result


def _pinned_sha256(value: Any, location: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise MarketAcquisitionError(f"{location} must be a lowercase SHA-256")
    return value


_OWNED_TRANSPORT_CAPABILITY_BINDING_KEYS: Final[tuple[str, ...]] = (
    "acquisition_plan_sha256",
    "acquisition_receipt_sha256",
    "bundle_sha256",
    "validation_sha256",
    "source_manifest_sha256",
    "market_stage_manifest_sha256",
    "source_reconciliation_sha256",
)


def _owned_transport_capability_binding(
    *,
    acquisition_plan_sha256: str,
    acquisition_receipt_sha256: str,
    bundle_sha256: str,
    validation_sha256: str,
    source_manifest_sha256: str,
    market_stage_manifest_sha256: str,
    source_reconciliation_sha256: str,
) -> tuple[tuple[str, str], ...]:
    values = {
        "acquisition_plan_sha256": acquisition_plan_sha256,
        "acquisition_receipt_sha256": acquisition_receipt_sha256,
        "bundle_sha256": bundle_sha256,
        "validation_sha256": validation_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "market_stage_manifest_sha256": market_stage_manifest_sha256,
        "source_reconciliation_sha256": source_reconciliation_sha256,
    }
    return tuple(
        (
            key,
            _pinned_sha256(
                values[key],
                f"owned Yahoo transport capability {key}",
            ),
        )
        for key in _OWNED_TRANSPORT_CAPABILITY_BINDING_KEYS
    )


def _build_owned_transport_capability_boundary():
    """Create a process-local, one-use capability with no serializable form."""

    issuer = object()
    missing = object()
    capability_bindings: WeakKeyDictionary[
        object, tuple[tuple[str, str], ...]
    ] = WeakKeyDictionary()
    acquisition_results: WeakKeyDictionary[
        object, tuple[dict[str, Any], object]
    ] = WeakKeyDictionary()
    registry_lock = Lock()

    class OwnedMarketTransportCapability:
        __slots__ = ("__weakref__",)

        def __init__(self, token: object) -> None:
            if token is not issuer:
                raise TypeError("Owned market transport capabilities are not constructible")

        def __copy__(self) -> None:
            raise TypeError("Owned market transport capabilities cannot be copied")

        def __deepcopy__(self, _memo: Any) -> None:
            raise TypeError("Owned market transport capabilities cannot be copied")

        def __reduce__(self) -> None:
            raise TypeError("Owned market transport capabilities cannot be serialized")

        def __repr__(self) -> str:
            return "<owned Yahoo transport capability>"

    class OwnedDevelopmentMarketAcquisition:
        __slots__ = ("__weakref__",)

        def __init__(self, token: object) -> None:
            if token is not issuer:
                raise TypeError("Owned market acquisition results are not constructible")

        def __copy__(self) -> None:
            raise TypeError("Owned market acquisition results cannot be copied")

        def __deepcopy__(self, _memo: Any) -> None:
            raise TypeError("Owned market acquisition results cannot be copied")

        def __reduce__(self) -> None:
            raise TypeError("Owned market acquisition results cannot be serialized")

        def __repr__(self) -> str:
            return "<owned Yahoo development market acquisition>"

    def issue(
        bundle: dict[str, Any],
        binding: tuple[tuple[str, str], ...],
    ) -> OwnedDevelopmentMarketAcquisition:
        capability = OwnedMarketTransportCapability(issuer)
        result = OwnedDevelopmentMarketAcquisition(issuer)
        with registry_lock:
            capability_bindings[capability] = binding
            acquisition_results[result] = (bundle, capability)
        return result

    def acquire_owned() -> object:
        """Run the exact owned Yahoo effect and issue its process-local authority."""

        transport = _OwnedYahooChartTransport()
        bundle = _acquire_development_market_evidence_with_transport(
            transport,
            _owned_production_transport=True,
        )
        if (
            type(transport) is not _OwnedYahooChartTransport
            or type(bundle) is not dict
            or transport.request_count != YAHOO_REQUEST_COUNT
            or transport.response_bytes
            != bundle["acquisition_receipt"]["raw_response_total_bytes"]
        ):
            raise MarketAcquisitionError(
                "Owned Yahoo transport accounting did not reconcile"
            )
        if bundle["acquisition_receipt"].get("transport_policy") != (
            _receipt_transport_policy(trusted_transport=True)
        ):
            raise MarketAcquisitionError(
                "Owned Yahoo transport did not produce its exact trusted policy"
            )
        validation = validate_development_market_acquisition_bundle(
            bundle,
            expected_acquisition_plan_sha256=bundle[
                "acquisition_plan_sha256"
            ],
            expected_acquisition_receipt_sha256=bundle[
                "acquisition_receipt_sha256"
            ],
            expected_bundle_sha256=bundle["bundle_sha256"],
        )
        binding = _owned_transport_capability_binding(
            acquisition_plan_sha256=bundle["acquisition_plan_sha256"],
            acquisition_receipt_sha256=bundle["acquisition_receipt_sha256"],
            bundle_sha256=bundle["bundle_sha256"],
            validation_sha256=validation["validation_sha256"],
            source_manifest_sha256=bundle["source_manifest_sha256"],
            market_stage_manifest_sha256=bundle[
                "market_stage_manifest_sha256"
            ],
            source_reconciliation_sha256=bundle[
                "source_reconciliation_sha256"
            ],
        )
        return issue(bundle, binding)

    def unwrap(
        value: object,
    ) -> tuple[dict[str, Any], OwnedMarketTransportCapability]:
        if type(value) is not OwnedDevelopmentMarketAcquisition:
            raise MarketAcquisitionError(
                "Market evidence did not come from the owned Yahoo transport boundary"
            )
        with registry_lock:
            owned_result = acquisition_results.pop(value, missing)
        if owned_result is missing:
            raise MarketAcquisitionError(
                "Owned Yahoo acquisition result was already unwrapped"
            )
        return owned_result

    def bind_to_claim(
        value: object,
        *,
        development_root_scope_sha256: str,
        claim_sha256: str,
    ) -> None:
        if type(value) is not OwnedMarketTransportCapability:
            raise MarketAcquisitionError(
                "Only an owned Yahoo transport capability can bind to a claim"
            )
        scope_hash = _pinned_sha256(
            development_root_scope_sha256,
            "owned Yahoo transport capability development scope",
        )
        claim_hash = _pinned_sha256(
            claim_sha256,
            "owned Yahoo transport capability claim",
        )
        with registry_lock:
            observed_binding = capability_bindings.pop(value, missing)
            if observed_binding is missing:
                raise MarketAcquisitionError(
                    "Owned Yahoo transport capability is invalid or already bound"
                )
            if tuple(key for key, _value in observed_binding) != (
                _OWNED_TRANSPORT_CAPABILITY_BINDING_KEYS
            ):
                raise MarketAcquisitionError(
                    "Owned Yahoo transport capability was already claim-bound"
                )
            capability_bindings[value] = (
                *observed_binding,
                ("development_root_scope_sha256", scope_hash),
                ("claim_sha256", claim_hash),
            )

    def consume(
        value: object,
        binding: tuple[tuple[str, str], ...],
        *,
        development_root_scope_sha256: str,
        claim_sha256: str,
    ) -> None:
        if type(value) is not OwnedMarketTransportCapability:
            raise MarketAcquisitionError(
                "A new market receipt requires the owned Yahoo transport capability"
            )
        expected_binding = (
            *binding,
            (
                "development_root_scope_sha256",
                _pinned_sha256(
                    development_root_scope_sha256,
                    "owned Yahoo transport capability development scope",
                ),
            ),
            (
                "claim_sha256",
                _pinned_sha256(
                    claim_sha256,
                    "owned Yahoo transport capability claim",
                ),
            ),
        )
        with registry_lock:
            observed_binding = capability_bindings.pop(value, missing)
        if observed_binding is missing:
            raise MarketAcquisitionError(
                "Owned Yahoo transport capability is invalid or already consumed"
            )
        if len(observed_binding) != len(expected_binding) or any(
            observed_key != expected_key
            or not hmac.compare_digest(observed_hash, expected_hash)
            for (observed_key, observed_hash), (expected_key, expected_hash) in zip(
                observed_binding,
                expected_binding,
                strict=True,
            )
        ):
            raise MarketAcquisitionError(
                "Owned Yahoo transport capability binding changed"
            )

    return acquire_owned, unwrap, bind_to_claim, consume


(
    _acquire_owned_development_market_evidence,
    _unwrap_owned_development_market_acquisition,
    _bind_owned_market_transport_capability_to_claim,
    _consume_owned_market_transport_capability_binding,
) = _build_owned_transport_capability_boundary()


def _consume_owned_market_transport_capability(
    capability: object,
    *,
    development_root_scope_sha256: str,
    claim_sha256: str,
    acquisition_plan_sha256: str,
    acquisition_receipt_sha256: str,
    bundle_sha256: str,
    validation_sha256: str,
    source_manifest_sha256: str,
    market_stage_manifest_sha256: str,
    source_reconciliation_sha256: str,
) -> None:
    """Consume one exact owned-transport authority after durable store replay."""

    binding = _owned_transport_capability_binding(
        acquisition_plan_sha256=acquisition_plan_sha256,
        acquisition_receipt_sha256=acquisition_receipt_sha256,
        bundle_sha256=bundle_sha256,
        validation_sha256=validation_sha256,
        source_manifest_sha256=source_manifest_sha256,
        market_stage_manifest_sha256=market_stage_manifest_sha256,
        source_reconciliation_sha256=source_reconciliation_sha256,
    )
    _consume_owned_market_transport_capability_binding(
        capability,
        binding,
        development_root_scope_sha256=development_root_scope_sha256,
        claim_sha256=claim_sha256,
    )


def _plain_json(value: Any, location: str) -> Any:
    if type(value) is dict:
        result: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str or key in result:
                raise MarketAcquisitionError(
                    f"{location} must have unique built-in string keys"
                )
            result[key] = _plain_json(child, f"{location}.{key}")
        return result
    if type(value) is list:
        return [
            _plain_json(child, f"{location}[{index}]")
            for index, child in enumerate(value)
        ]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise MarketAcquisitionError(f"{location} must be detached finite JSON")


def _exact_bytes_by_symbol(value: Any, location: str) -> dict[str, bytes]:
    if type(value) is not dict or set(value) != set(MARKET_SYMBOLS):
        raise MarketAcquisitionError(f"{location} must contain exactly six symbols")
    result: dict[str, bytes] = {}
    for symbol in MARKET_SYMBOLS:
        payload = value[symbol]
        if type(payload) is not bytes or not payload:
            raise MarketAcquisitionError(f"{location}.{symbol} must be exact bytes")
        result[symbol] = payload
    return result


def validate_development_market_acquisition_bundle(
    bundle: Mapping[str, Any],
    *,
    expected_acquisition_plan_sha256: str,
    expected_acquisition_receipt_sha256: str,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Replay raw Yahoo bytes through every normalization and evidence layer.

    The caller must independently pin the pre-effect plan, the post-effect
    acquisition receipt, and the compact bundle identity.  Exact raw responses
    remain verifier inputs and are never returned from this function.
    """

    expected_plan_hash = _pinned_sha256(
        expected_acquisition_plan_sha256,
        "expected_acquisition_plan_sha256",
    )
    expected_receipt_hash = _pinned_sha256(
        expected_acquisition_receipt_sha256,
        "expected_acquisition_receipt_sha256",
    )
    expected_bundle_hash = _pinned_sha256(
        expected_bundle_sha256,
        "expected_bundle_sha256",
    )
    if type(bundle) is not dict:
        raise MarketAcquisitionError("Acquisition bundle must be a detached plain dict")
    expected_bundle_keys = {
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
        "bundle_sha256",
        "raw_response_bytes_by_symbol",
        "artifact_bytes_by_symbol",
        "window_bytes_by_symbol",
        "source_manifest",
        "stage_manifest",
        "reconciliation_receipt",
        "acquisition_receipt",
    }
    if set(bundle) != expected_bundle_keys:
        raise MarketAcquisitionError("Acquisition bundle keys changed")
    if (
        bundle["schema_version"] != MARKET_ACQUISITION_BUNDLE_SCHEMA_VERSION
        or bundle["artifact_stage"] != "development"
    ):
        raise MarketAcquisitionError("Acquisition bundle schema or stage changed")

    raw_by_symbol = _exact_bytes_by_symbol(
        bundle["raw_response_bytes_by_symbol"], "raw response bytes"
    )
    if sum(len(raw_by_symbol[symbol]) for symbol in MARKET_SYMBOLS) > (
        YAHOO_MAX_TOTAL_RESPONSE_BYTES
    ):
        raise MarketAcquisitionError(
            "Acquisition bundle exceeded the aggregate raw-response byte ceiling"
        )

    receipt = _plain_json(bundle["acquisition_receipt"], "acquisition receipt")
    receipt_sources = receipt.get("source_code_sha256s")
    if type(receipt_sources) is not dict or any(
        type(key) is not str
        or not isinstance(value, str)
        or _SHA256_RE.fullmatch(value) is None
        for key, value in receipt_sources.items()
    ):
        raise MarketAcquisitionError("Acquisition receipt source hashes are invalid")
    pinned_plan = _build_development_market_acquisition_plan(receipt_sources)
    if not hmac.compare_digest(
        pinned_plan["acquisition_plan_sha256"], expected_plan_hash
    ) or bundle["acquisition_plan_sha256"] != expected_plan_hash:
        raise MarketAcquisitionError(
            "Acquisition plan or its live reviewed source identity changed"
        )

    artifact_inputs = _exact_bytes_by_symbol(
        bundle["artifact_bytes_by_symbol"], "artifact snapshot bytes"
    )
    window_inputs = _exact_bytes_by_symbol(
        bundle["window_bytes_by_symbol"], "window snapshot bytes"
    )
    source_input = _plain_json(bundle["source_manifest"], "source manifest")
    stage_input = _plain_json(bundle["stage_manifest"], "stage manifest")
    reconciliation_input = _plain_json(
        bundle["reconciliation_receipt"], "source reconciliation receipt"
    )
    compact_identity_inputs = {
        "raw_response_sha256s": _plain_json(
            bundle["raw_response_sha256s"], "raw response hash map"
        ),
        "artifact_sha256s": _plain_json(
            bundle["artifact_sha256s"], "artifact hash map"
        ),
        "window_sha256s": _plain_json(
            bundle["window_sha256s"], "window hash map"
        ),
    }

    receipt_hash = _pinned_sha256(
        receipt.get("acquisition_receipt_sha256"),
        "acquisition_receipt_sha256",
    )
    receipt_body = {
        key: value for key, value in receipt.items() if key != "acquisition_receipt_sha256"
    }
    if (
        not hmac.compare_digest(receipt_hash, canonical_sha256(receipt_body))
        or not hmac.compare_digest(receipt_hash, expected_receipt_hash)
        or bundle["acquisition_receipt_sha256"] != expected_receipt_hash
    ):
        raise MarketAcquisitionError("Acquisition receipt is not externally pinned")
    if receipt.get("acquisition_plan_sha256") != expected_plan_hash:
        raise MarketAcquisitionError("Acquisition receipt plan binding changed")

    parsed_by_symbol: dict[str, _ParsedSymbol] = {}
    expected_requests: list[dict[str, Any]] = []
    observed_requests = receipt.get("requests")
    if type(observed_requests) is not list or len(observed_requests) != len(
        MARKET_SYMBOLS
    ):
        raise MarketAcquisitionError("Acquisition request receipts are incomplete")
    for index, symbol in enumerate(MARKET_SYMBOLS):
        item = observed_requests[index]
        if type(item) is not dict:
            raise MarketAcquisitionError("Acquisition request receipt is not an object")
        charset = item.get("charset")
        content_encoding = item.get("content_encoding")
        declared_content_length = item.get("declared_content_length")
        if charset not in {None, "utf-8", "UTF-8"}:
            raise MarketAcquisitionError(f"{symbol} receipt charset changed")
        if content_encoding not in {None, "identity"}:
            raise MarketAcquisitionError(f"{symbol} receipt compression changed")
        if declared_content_length is not None and (
            isinstance(declared_content_length, bool)
            or type(declared_content_length) is not int
            or declared_content_length != len(raw_by_symbol[symbol])
        ):
            raise MarketAcquisitionError(
                f"{symbol} receipt Content-Length changed"
            )
        parsed = _parse_yahoo_chart_response(
            symbol=symbol,
            raw_bytes=raw_by_symbol[symbol],
        )
        _validate_symbol_coverage(symbol, parsed)
        parsed_by_symbol[symbol] = parsed
        expected_requests.append(
            _request_receipt(
                symbol=symbol,
                raw_bytes=raw_by_symbol[symbol],
                parsed=parsed,
                charset=charset,
                content_encoding=content_encoding,
                declared_content_length=declared_content_length,
            )
        )
    if observed_requests != expected_requests:
        raise MarketAcquisitionError(
            "Request receipt does not replay from exact provider bytes"
        )

    expected_artifacts = {
        symbol: build_market_source_snapshot_bytes(
            symbol=symbol,
            rows=_snapshot_rows(parsed_by_symbol[symbol]),
        )
        for symbol in MARKET_SYMBOLS
    }
    if any(
        artifact_inputs[symbol] != expected_artifacts[symbol]
        or window_inputs[symbol] != expected_artifacts[symbol]
        for symbol in MARKET_SYMBOLS
    ):
        raise MarketAcquisitionError(
            "Canonical snapshot bytes do not replay from provider bytes"
        )
    source_artifacts = {
        symbol: {
            "artifact_sha256": hashlib.sha256(expected_artifacts[symbol]).hexdigest(),
            "artifact_bytes": len(expected_artifacts[symbol]),
            "window_sha256": hashlib.sha256(expected_artifacts[symbol]).hexdigest(),
            "window_bytes": len(expected_artifacts[symbol]),
            "available_sessions": list(parsed_by_symbol[symbol].rows_by_session),
        }
        for symbol in MARKET_SYMBOLS
    }
    expected_source = build_market_source_manifest(
        artifact_stage="development",
        source_artifacts=source_artifacts,
    )
    if source_input != expected_source:
        raise MarketAcquisitionError(
            "Source manifest does not replay from canonical snapshot bytes"
        )
    expected_stage = build_market_stage_manifest(
        artifact_stage="development",
        source_manifest=expected_source,
        expected_source_manifest_sha256=expected_source["source_manifest_sha256"],
        rows=_stage_rows(parsed_by_symbol),
    )
    if stage_input != expected_stage:
        raise MarketAcquisitionError(
            "Stage manifest does not replay from exact provider observations"
        )
    expected_reconciliation = validate_market_source_bytes_against_stage(
        artifact_bytes_by_symbol=expected_artifacts,
        window_bytes_by_symbol=expected_artifacts,
        source_manifest=expected_source,
        stage_manifest=expected_stage,
        expected_artifact_stage="development",
        expected_source_manifest_sha256=expected_source["source_manifest_sha256"],
        expected_market_stage_manifest_sha256=expected_stage[
            "market_stage_manifest_sha256"
        ],
    )
    if reconciliation_input != expected_reconciliation:
        raise MarketAcquisitionError("Source reconciliation receipt changed")

    raw_set = [
        {
            "symbol": symbol,
            "sha256": hashlib.sha256(raw_by_symbol[symbol]).hexdigest(),
            "bytes": len(raw_by_symbol[symbol]),
        }
        for symbol in MARKET_SYMBOLS
    ]
    snapshot_receipts = {
        symbol: {
            "artifact_sha256": hashlib.sha256(expected_artifacts[symbol]).hexdigest(),
            "artifact_bytes": len(expected_artifacts[symbol]),
            "window_sha256": hashlib.sha256(expected_artifacts[symbol]).hexdigest(),
            "window_bytes": len(expected_artifacts[symbol]),
        }
        for symbol in MARKET_SYMBOLS
    }
    policy = receipt.get("transport_policy")
    if type(policy) is not dict or type(policy.get("trusted_production_transport")) is not bool:
        raise MarketAcquisitionError("Acquisition transport policy is invalid")
    expected_policy = _receipt_transport_policy(
        trusted_transport=policy["trusted_production_transport"]
    )
    expected_receipt_body = {
        "schema_version": MARKET_ACQUISITION_SCHEMA_VERSION,
        "provider_family": YAHOO_CHART_PROVIDER_FAMILY,
        "artifact_stage": "development",
        "acquisition_plan_sha256": expected_plan_hash,
        "request_window": pinned_plan["request_window"],
        "requests": expected_requests,
        "raw_response_count": len(raw_set),
        "raw_response_total_bytes": sum(item["bytes"] for item in raw_set),
        "raw_response_set_sha256": canonical_sha256(raw_set),
        "snapshot_artifacts": snapshot_receipts,
        "source_code_sha256s": pinned_plan["source_code_sha256s"],
        "source_manifest_sha256": expected_source["source_manifest_sha256"],
        "market_stage_manifest_sha256": expected_stage[
            "market_stage_manifest_sha256"
        ],
        "source_reconciliation_sha256": expected_reconciliation[
            "reconciliation_sha256"
        ],
        "normalization_contract": _receipt_normalization_contract(),
        "transport_policy": expected_policy,
    }
    if receipt_body != expected_receipt_body:
        raise MarketAcquisitionError(
            "Acquisition receipt fields do not replay from exact evidence"
        )

    expected_bundle_body = {
        "schema_version": MARKET_ACQUISITION_BUNDLE_SCHEMA_VERSION,
        "artifact_stage": "development",
        "acquisition_receipt_sha256": expected_receipt_hash,
        "acquisition_plan_sha256": expected_plan_hash,
        "source_manifest_sha256": expected_source["source_manifest_sha256"],
        "market_stage_manifest_sha256": expected_stage[
            "market_stage_manifest_sha256"
        ],
        "source_reconciliation_sha256": expected_reconciliation[
            "reconciliation_sha256"
        ],
        "raw_response_sha256s": {
            symbol: hashlib.sha256(raw_by_symbol[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        },
        "artifact_sha256s": {
            symbol: hashlib.sha256(expected_artifacts[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        },
        "window_sha256s": {
            symbol: hashlib.sha256(expected_artifacts[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        },
    }
    observed_bundle_body = {
        key: (
            compact_identity_inputs[key]
            if key in compact_identity_inputs
            else bundle[key]
        )
        for key in expected_bundle_body
    }
    observed_compact_hash = _pinned_sha256(bundle["bundle_sha256"], "bundle_sha256")
    if (
        observed_bundle_body != expected_bundle_body
        or not hmac.compare_digest(
            observed_compact_hash, canonical_sha256(expected_bundle_body)
        )
        or not hmac.compare_digest(observed_compact_hash, expected_bundle_hash)
    ):
        raise MarketAcquisitionError("Acquisition bundle is not externally pinned")

    validation_body = {
        "schema_version": MARKET_ACQUISITION_VALIDATION_SCHEMA_VERSION,
        "artifact_stage": "development",
        "acquisition_plan_sha256": expected_plan_hash,
        "acquisition_receipt_sha256": expected_receipt_hash,
        "bundle_sha256": expected_bundle_hash,
        "source_manifest_sha256": expected_source["source_manifest_sha256"],
        "market_stage_manifest_sha256": expected_stage[
            "market_stage_manifest_sha256"
        ],
        "source_reconciliation_sha256": expected_reconciliation[
            "reconciliation_sha256"
        ],
        "provider_response_to_stage_replayed": True,
        "exact_raw_bytes_replayed": True,
        "production_authorized": False,
    }
    return {
        **validation_body,
        "validation_sha256": canonical_sha256(validation_body),
    }


def _acquire_development_market_evidence_with_transport(
    transport: _TransportLike,
    *,
    _owned_production_transport: bool = False,
) -> dict[str, Any]:
    """Private injection seam for deterministic, network-free tests only."""

    if not hasattr(transport, "fetch"):
        raise TypeError("Market transport must provide fetch()")
    acquisition_plan = build_development_market_acquisition_plan()
    trusted_transport = (
        _owned_production_transport and type(transport) is _OwnedYahooChartTransport
    )
    raw_by_symbol: dict[str, bytes] = {}
    parsed_by_symbol: dict[str, _ParsedSymbol] = {}
    request_receipts: list[dict[str, Any]] = []
    for symbol in MARKET_SYMBOLS:
        url = _canonical_url(symbol)
        response = transport.fetch(url)
        raw_bytes = _validate_transport_response(
            response,
            expected_url=url,
            symbol=symbol,
        )
        parsed = _parse_yahoo_chart_response(symbol=symbol, raw_bytes=raw_bytes)
        _validate_symbol_coverage(symbol, parsed)
        raw_by_symbol[symbol] = raw_bytes
        parsed_by_symbol[symbol] = parsed
        request_receipts.append(
            _request_receipt(
                symbol=symbol,
                raw_bytes=raw_bytes,
                parsed=parsed,
                charset=response.charset,
                content_encoding=response.content_encoding,
                declared_content_length=response.declared_content_length,
            )
        )

    post_effect_plan = build_development_market_acquisition_plan()
    if post_effect_plan != acquisition_plan:
        raise MarketAcquisitionError(
            "Acquisition source identity changed across the network effect"
        )

    artifact_bytes: dict[str, bytes] = {}
    for symbol in MARKET_SYMBOLS:
        artifact_bytes[symbol] = build_market_source_snapshot_bytes(
            symbol=symbol,
            rows=_snapshot_rows(parsed_by_symbol[symbol]),
        )
    window_bytes = dict(artifact_bytes)
    source_artifacts = {
        symbol: {
            "artifact_sha256": hashlib.sha256(artifact_bytes[symbol]).hexdigest(),
            "artifact_bytes": len(artifact_bytes[symbol]),
            "window_sha256": hashlib.sha256(window_bytes[symbol]).hexdigest(),
            "window_bytes": len(window_bytes[symbol]),
            "available_sessions": list(parsed_by_symbol[symbol].rows_by_session),
        }
        for symbol in MARKET_SYMBOLS
    }
    source_manifest = build_market_source_manifest(
        artifact_stage="development",
        source_artifacts=source_artifacts,
    )
    stage_manifest = build_market_stage_manifest(
        artifact_stage="development",
        source_manifest=source_manifest,
        expected_source_manifest_sha256=source_manifest["source_manifest_sha256"],
        rows=_stage_rows(parsed_by_symbol),
    )
    reconciliation = validate_market_source_bytes_against_stage(
        artifact_bytes_by_symbol=artifact_bytes,
        window_bytes_by_symbol=window_bytes,
        source_manifest=source_manifest,
        stage_manifest=stage_manifest,
        expected_artifact_stage="development",
        expected_source_manifest_sha256=source_manifest["source_manifest_sha256"],
        expected_market_stage_manifest_sha256=stage_manifest[
            "market_stage_manifest_sha256"
        ],
    )
    snapshot_receipts = {
        symbol: {
            "artifact_sha256": hashlib.sha256(artifact_bytes[symbol]).hexdigest(),
            "artifact_bytes": len(artifact_bytes[symbol]),
            "window_sha256": hashlib.sha256(window_bytes[symbol]).hexdigest(),
            "window_bytes": len(window_bytes[symbol]),
        }
        for symbol in MARKET_SYMBOLS
    }
    raw_total = sum(len(raw_by_symbol[symbol]) for symbol in MARKET_SYMBOLS)
    if raw_total > YAHOO_MAX_TOTAL_RESPONSE_BYTES:
        raise MarketAcquisitionError(
            "Yahoo market batch exceeded the aggregate byte ceiling"
        )
    raw_set = [
        {
            "symbol": symbol,
            "sha256": hashlib.sha256(raw_by_symbol[symbol]).hexdigest(),
            "bytes": len(raw_by_symbol[symbol]),
        }
        for symbol in MARKET_SYMBOLS
    ]
    receipt_body = {
        "schema_version": MARKET_ACQUISITION_SCHEMA_VERSION,
        "provider_family": YAHOO_CHART_PROVIDER_FAMILY,
        "artifact_stage": "development",
        "acquisition_plan_sha256": acquisition_plan["acquisition_plan_sha256"],
        "request_window": {
            "start": DEVELOPMENT_REQUEST_START,
            "end_inclusive": DEVELOPMENT_LAST_SESSION,
            "end_exclusive": DEVELOPMENT_REQUEST_END_EXCLUSIVE,
            "period1_utc": DEVELOPMENT_PERIOD1_UTC,
            "period2_utc": DEVELOPMENT_PERIOD2_UTC,
            "timestamp_session_timezones": dict(YAHOO_PROVIDER_TIMEZONES),
        },
        "requests": request_receipts,
        "raw_response_count": len(raw_set),
        "raw_response_total_bytes": raw_total,
        "raw_response_set_sha256": canonical_sha256(raw_set),
        "snapshot_artifacts": snapshot_receipts,
        "source_code_sha256s": acquisition_plan["source_code_sha256s"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "market_stage_manifest_sha256": stage_manifest[
            "market_stage_manifest_sha256"
        ],
        "source_reconciliation_sha256": reconciliation["reconciliation_sha256"],
        "normalization_contract": _receipt_normalization_contract(),
        "transport_policy": _receipt_transport_policy(
            trusted_transport=trusted_transport
        ),
    }
    acquisition_receipt = {
        **receipt_body,
        "acquisition_receipt_sha256": canonical_sha256(receipt_body),
    }
    bundle_body = {
        "schema_version": MARKET_ACQUISITION_BUNDLE_SCHEMA_VERSION,
        "artifact_stage": "development",
        "acquisition_receipt_sha256": acquisition_receipt[
            "acquisition_receipt_sha256"
        ],
        "acquisition_plan_sha256": acquisition_plan["acquisition_plan_sha256"],
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "market_stage_manifest_sha256": stage_manifest[
            "market_stage_manifest_sha256"
        ],
        "source_reconciliation_sha256": reconciliation["reconciliation_sha256"],
        "raw_response_sha256s": {
            symbol: hashlib.sha256(raw_by_symbol[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        },
        "artifact_sha256s": {
            symbol: hashlib.sha256(artifact_bytes[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        },
        "window_sha256s": {
            symbol: hashlib.sha256(window_bytes[symbol]).hexdigest()
            for symbol in MARKET_SYMBOLS
        },
    }
    result = {
        **bundle_body,
        "bundle_sha256": canonical_sha256(bundle_body),
        "raw_response_bytes_by_symbol": raw_by_symbol,
        "artifact_bytes_by_symbol": artifact_bytes,
        "window_bytes_by_symbol": window_bytes,
        "source_manifest": source_manifest,
        "stage_manifest": stage_manifest,
        "reconciliation_receipt": reconciliation,
        "acquisition_receipt": acquisition_receipt,
    }
    if build_development_market_acquisition_plan() != acquisition_plan:
        raise MarketAcquisitionError(
            "Acquisition source identity changed before bundle return"
        )
    return result


__all__ = [
    "DEVELOPMENT_LAST_SESSION",
    "DEVELOPMENT_PERIOD1_UTC",
    "DEVELOPMENT_PERIOD2_UTC",
    "DEVELOPMENT_REQUEST_END_EXCLUSIVE",
    "DEVELOPMENT_REQUEST_START",
    "CONTEXT_ALLOWED_MISSING_SESSIONS",
    "CONTEXT_FIRST_ACCEPTED_SESSION",
    "MARKET_ACQUISITION_BUNDLE_SCHEMA_VERSION",
    "MARKET_ACQUISITION_PLAN_SCHEMA_VERSION",
    "MARKET_ACQUISITION_SCHEMA_VERSION",
    "MARKET_ACQUISITION_VALIDATION_SCHEMA_VERSION",
    "MarketAcquisitionError",
    "YAHOO_CHART_ENDPOINT",
    "YAHOO_CHART_PROVIDER_FAMILY",
    "YAHOO_PROVIDER_TIMEZONES",
    "YAHOO_PROVIDER_SYMBOLS",
    "build_development_market_acquisition_plan",
    "validate_development_market_acquisition_bundle",
]
