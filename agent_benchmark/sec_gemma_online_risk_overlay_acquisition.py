"""One-shot, stage-bound acquisition for the SEC/Gemma online overlay.

Plan construction and plan validation are pure.  The effectful entry point
accepts only a store-issued :class:`EffectCapability`, authorizes both network
effects against the issuing store before the first request, and returns a
private quarantine bundle.  The bundle contains exact source bytes and
deterministic redacted model-request bytes, but never decoded market values,
model output, labels, actions, returns, or scores.

The official-SEC catalogue/parser and filing preprocessor are inherited from
the pinned v1 implementation.  Yahoo Chart-v8 bytes are parsed only inside the
quarantine boundary so later stages can prove bit-exact historical-prefix
continuity.  Decoded values are never returned.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import ssl
import time
from typing import Any, Final, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)
from zoneinfo import ZoneInfo

from agent_benchmark.sec_gemma_online_risk_overlay_source_verifier import (
    load_allowed_requests,
)

requests = load_allowed_requests()

from agent_benchmark.sec_audit_transport import SecAuditTransport
from agent_benchmark.sec_filing_content import normalize_filing_text
from agent_benchmark.sec_filing_gemma_contract import (
    build_corpus_universe_manifest,
    build_contract_manifest as build_sec_contract_manifest,
    build_stage_content_manifest,
)
from agent_benchmark.sec_filing_gemma_corpus import (
    MAIN_SUBMISSIONS_URL,
    REQUEST_RECEIPT_SCHEMA_VERSION as SEC_REQUEST_RECEIPT_SCHEMA_VERSION,
    STAGE_ARTIFACT_SCHEMA_VERSION,
    SecCorpusBudget,
    _acquire_authenticated_stage_access_document_batch,
    acquire_official_sec_catalog,
    validate_detached_catalog_replay,
    validate_detached_stage_content_replay,
)
from agent_benchmark.sec_filing_gemma_market_acquirer import (
    CONTEXT_ALLOWED_MISSING_SESSIONS,
    CONTEXT_FIRST_ACCEPTED_SESSION,
)
from agent_benchmark.sec_gemma_online_risk_overlay_features import (
    build_validated_universe_event_proof,
    validate_universe_event_proof,
)
from agent_benchmark.sec_filing_gemma_preprocessor import (
    CANONICAL_IDENTITY_LEXICON,
    preprocess_filing_event,
)
from agent_benchmark.sec_gemma_online_risk_overlay_attempt import (
    CONFIRMATION_SCORING,
    DEVELOPMENT_ACQUISITION,
    FINAL_SCORING,
)
from agent_benchmark.sec_gemma_online_risk_overlay_contract import (
    CONFIRMATION_ATTEMPT_ID,
    CONTRACT_SHA256,
    DEVELOPMENT_ACQUISITION_ID,
    FINAL_ATTEMPT_ID,
    MAX_MARKET_REQUESTS_PER_STAGE,
    MAX_MARKET_SECONDS,
    MAX_SEC_BYTES,
    MAX_SEC_REQUESTS,
    MAX_SEC_REQUESTS_PER_SECOND,
    MAX_SEC_SECONDS,
    canonical_json_bytes,
    canonical_sha256,
    build_contract_manifest,
)
from agent_benchmark.sec_gemma_online_risk_overlay_store import EffectCapability
from agent_benchmark.sec_gemma_online_risk_overlay_vault import (
    ProductionAcquisitionVault,
    SecGemmaOnlineRiskOverlayVaultError,
    TestAcquisitionVault,
    VaultHandle,
    _assert_vault_handle_current,
    _load_current_production_handle_for_recovery,
    _production_recovery_sealed_stages,
    _read_quarantine_for_model_slice,
    _read_quarantine_for_replay,
    _read_quarantine_for_stage_slice,
    _seal_quarantine,
)
from agent_benchmark.sec_point_in_time import validate_sec_user_agent
from agent_benchmark.sec_session_calendar import (
    EXPECTED_MARKET_HISTORY_SESSIONS,
    EXPECTED_SESSIONS,
)


ACQUISITION_PLAN_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-acquisition-plan-v1"
)
ACQUISITION_BUNDLE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-acquisition-bundle-v1"
)
ACQUISITION_MANIFEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-acquisition-manifest-v1"
)
ACQUISITION_VALIDATION_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-acquisition-validation-v1"
)
ACQUISITION_VALIDATION_VERIFIER_ID: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-acquisition-validator-v1"
)
ACQUISITION_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-request-receipt-v1"
)
BLINDED_MODEL_REQUEST_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-blinded-model-request-v1"
)
STAGE_SLICE_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-stage-slice-v1"
)
ACQUISITION_PUBLIC_SUMMARY_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-public-summary-v1"
)
ACQUISITION_REQUEST_ACCOUNTING_SCHEMA_VERSION: Final[str] = (
    "aapl-sec-gemma-online-risk-overlay-v2-2-request-accounting-v1"
)

DEVELOPMENT: Final[str] = "development"
CONFIRMATION: Final[str] = "confirmation"
FINAL: Final[str] = "final"
STAGES: Final[tuple[str, ...]] = (DEVELOPMENT, CONFIRMATION, FINAL)

YAHOO_ENDPOINT: Final[str] = (
    "https://query1.finance.yahoo.com/v8/finance/chart"
)
YAHOO_USER_AGENT: Final[str] = (
    "LLM-memory-trading-agent/1.0 market-evidence (no-auth; one-shot)"
)
YAHOO_SYMBOLS: Final[tuple[str, ...]] = (
    "AAPL",
    "SPY",
    "QQQ",
    "IWM",
    "VIX",
    "TNX",
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
MAX_MARKET_RESPONSE_BYTES: Final[int] = 64 * 1024 * 1024
MAX_MARKET_BATCH_BYTES: Final[int] = 128 * 1024 * 1024
MARKET_REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0
_YAHOO_BODY_READ_CHUNK_BYTES: Final[int] = 64 * 1024

_STAGE_SPECS: Final[dict[str, dict[str, Any]]] = {
    DEVELOPMENT: {
        "attempt_id": DEVELOPMENT_ACQUISITION_ID,
        "attempt_kind": DEVELOPMENT_ACQUISITION,
        "sec_stage": "development",
        "sec_start": "2000-01-01",
        "sec_end": "2018-12-31",
        "minimum_filings": 72,
        "maximum_model_requests": 80,
        "period1_utc": 883_612_800,
        "period2_utc": 1_546_300_800,
        "request_start": "1998-01-01",
        "request_end_exclusive": "2019-01-01",
        "last_transport_session": "2018-12-31",
        "last_value_session": "2018-12-31",
        "predecessor": None,
    },
    CONFIRMATION: {
        "attempt_id": CONFIRMATION_ATTEMPT_ID,
        "attempt_kind": CONFIRMATION_SCORING,
        "sec_stage": "intermediate",
        "sec_start": "2019-01-01",
        "sec_end": "2023-12-31",
        "minimum_filings": 19,
        "maximum_model_requests": 20,
        "period1_utc": 883_612_800,
        "period2_utc": 1_704_067_200,
        "request_start": "1998-01-01",
        "request_end_exclusive": "2024-01-01",
        "last_transport_session": "2023-12-29",
        "last_value_session": "2023-12-29",
        "predecessor": DEVELOPMENT,
    },
    FINAL: {
        "attempt_id": FINAL_ATTEMPT_ID,
        "attempt_kind": FINAL_SCORING,
        "sec_stage": "final",
        "sec_start": "2024-01-01",
        "sec_end": "2026-07-09",
        "minimum_filings": 1,
        "maximum_model_requests": 12,
        "period1_utc": 883_612_800,
        "period2_utc": 1_783_728_000,
        "request_start": "1998-01-01",
        "request_end_exclusive": "2026-07-11",
        "last_transport_session": "2026-07-10",
        "last_value_session": "2026-07-09",
        "predecessor": CONFIRMATION,
    },
}

_MARKET_RESPONSE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "request_url",
        "final_url",
        "status_code",
        "content_type",
        "charset",
        "content_encoding",
        "declared_content_length",
        "body",
        "network_requests",
        "retries",
        "redirects",
    }
)
_TRANSPORT_SECURITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "trust_env",
        "proxies",
        "follow_redirects",
        "max_retries",
        "max_redirects",
        "allow_cache_reads",
        "allow_cache_writes",
        "streaming_body",
        "content_length_preflight",
        "incremental_byte_budget",
        "transport_max_requests",
        "transport_max_bytes",
        "transport_max_seconds",
    }
)
_STRICT_TRANSPORT_FLAGS: Final[dict[str, bool | int]] = {
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
}
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"(?:sha256:)?[0-9a-f]{64}\Z")
_ACCESSION_RE: Final[re.Pattern[str]] = re.compile(
    r"[0-9]{10}-[0-9]{2}-[0-9]{6}\Z"
)
_PRIMARY_URL_RE: Final[re.Pattern[str]] = re.compile(
    r"https://www\.sec\.gov/Archives/edgar/data/320193/"
    r"(?P<accession>[0-9]{18})/"
    r"(?P<filename>[A-Za-z0-9][A-Za-z0-9._~-]{0,255})\Z"
)
_VERIFIED_REPORT_SENTINEL = object()
_EXECUTION_RESULT_SENTINEL = object()


class SecGemmaOnlineRiskOverlayAcquisitionError(RuntimeError):
    """A pure plan, private identity, capability, or quarantine is invalid."""


class SecGemmaOnlineRiskOverlayAcquisitionIndeterminate(
    SecGemmaOnlineRiskOverlayAcquisitionError
):
    """An effectful batch started but did not produce one complete result."""


class VerifiedAcquisitionReport(Mapping[str, Any]):
    """Opaque canonical report issued only after detached exact-byte replay."""

    __slots__ = (
        "_canonical_bytes",
        "_handle",
        "_locked",
        "_sentinel",
        "_vault",
    )

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        vault: ProductionAcquisitionVault | TestAcquisitionVault,
        handle: VaultHandle,
        _sentinel: object,
    ) -> None:
        if _sentinel is not _VERIFIED_REPORT_SENTINEL:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Verified acquisition reports can only be issued by the validator"
            )
        detached = _plain_json(
            dict(payload), "verified acquisition report"
        )
        object.__setattr__(self, "_locked", False)
        object.__setattr__(
            self, "_canonical_bytes", canonical_json_bytes(detached)
        )
        object.__setattr__(self, "_vault", vault)
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_sentinel", _sentinel)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("VerifiedAcquisitionReport is immutable")
        object.__setattr__(self, name, value)

    def _assert_current(self) -> None:
        try:
            _assert_vault_handle_current(self._vault, self._handle)
        except SecGemmaOnlineRiskOverlayVaultError:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Verified acquisition report is stale after vault mutation"
            ) from None

    def as_dict(self) -> dict[str, Any]:
        self._assert_current()
        return json.loads(self._canonical_bytes.decode("utf-8"))

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self):
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def __repr__(self) -> str:
        payload = self.as_dict()
        return (
            "VerifiedAcquisitionReport("
            f"stage={payload['stage']!r}, "
            f"validation_sha256={payload['validation_sha256']!r})"
        )


class AcquisitionExecutionResult:
    """Opaque durable acquisition result with no public raw bundle access."""

    __slots__ = (
        "_accounting_bytes",
        "_handle",
        "_locked",
        "_model_slice_sha256",
        "_public_summary_bytes",
        "_sentinel",
        "_stage_slice_sha256",
        "_vault",
        "_verified_report",
    )

    def __init__(
        self,
        *,
        vault: ProductionAcquisitionVault | TestAcquisitionVault,
        handle: VaultHandle,
        verified_report: VerifiedAcquisitionReport,
        public_summary: Mapping[str, Any],
        request_accounting: Mapping[str, Any],
        model_slice_sha256: str,
        stage_slice_sha256: str,
        _sentinel: object,
    ) -> None:
        if (
            _sentinel is not _EXECUTION_RESULT_SENTINEL
            or not is_verified_acquisition_report(verified_report)
            or verified_report["bundle_sha256"]
            != handle.bundle_sha256
            or verified_report["manifest_sha256"]
            != handle.manifest_sha256
            or verified_report["private_index_sha256"]
            != handle.private_index_sha256
            or type(public_summary) is not dict
            or type(request_accounting) is not dict
            or type(stage_slice_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", stage_slice_sha256) is None
            or type(model_slice_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", model_slice_sha256) is None
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Execution result requires one matching verified acquisition report"
            )
        object.__setattr__(self, "_locked", False)
        object.__setattr__(self, "_vault", vault)
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_verified_report", verified_report)
        object.__setattr__(
            self,
            "_public_summary_bytes",
            canonical_json_bytes(
                _plain_json(
                    dict(public_summary),
                    "acquisition public summary",
                )
            ),
        )
        object.__setattr__(
            self,
            "_accounting_bytes",
            canonical_json_bytes(
                _plain_json(
                    dict(request_accounting),
                    "acquisition request accounting",
                )
            ),
        )
        object.__setattr__(
            self, "_stage_slice_sha256", stage_slice_sha256
        )
        object.__setattr__(
            self, "_model_slice_sha256", model_slice_sha256
        )
        object.__setattr__(self, "_sentinel", _sentinel)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_locked", False):
            raise AttributeError("AcquisitionExecutionResult is immutable")
        object.__setattr__(self, name, value)

    def _assert_intact(self) -> None:
        try:
            _assert_vault_handle_current(self._vault, self._handle)
        except SecGemmaOnlineRiskOverlayVaultError:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Execution quarantine changed after validation"
            ) from None
        if not is_verified_acquisition_report(self._verified_report):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Execution validator report is stale or corrupt"
            )

    @property
    def verified_report(self) -> VerifiedAcquisitionReport:
        self._assert_intact()
        return self._verified_report

    @property
    def vault_handle(self) -> VaultHandle:
        self._assert_intact()
        return self._handle

    @property
    def production_authority(self) -> bool:
        self._assert_intact()
        return self._handle.production_authority

    def public_summary(self) -> dict[str, Any]:
        self._assert_intact()
        return json.loads(self._public_summary_bytes.decode("utf-8"))

    def request_accounting(self) -> dict[str, Any]:
        self._assert_intact()
        return json.loads(self._accounting_bytes.decode("utf-8"))

    def terminal_sealing_material(self) -> dict[str, Any]:
        self._assert_intact()
        return self._verified_report.as_dict()

    def stage_slice(
        self,
        *,
        store: Any,
        capability: EffectCapability,
    ) -> dict[str, Any]:
        """Release only the canonical stage-cutoff slice after scored access."""

        self._assert_intact()
        try:
            quarantine = _read_quarantine_for_stage_slice(
                self._vault,
                self._handle,
                store=store,
                capability=capability,
            )
        except SecGemmaOnlineRiskOverlayVaultError:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Stage slice access is unauthorized or stale"
            ) from None
        private = quarantine.get("private_quarantine")
        value = (
            None
            if type(private) is not dict
            else private.get("stage_slice")
        )
        if type(value) is not dict:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Stage slice is absent from the durable quarantine"
            )
        if (
            value.get("slice_sha256")
            != canonical_sha256(_stage_slice_index(value))
            or value["slice_sha256"] != self._stage_slice_sha256
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Stage slice changed after validation"
            )
        return copy.deepcopy(value)

    def model_slice(
        self,
        *,
        store: Any,
        capability: EffectCapability,
    ) -> dict[str, Any]:
        """Release blinded requests/proofs to Gemma without market values."""

        self._assert_intact()
        try:
            quarantine = _read_quarantine_for_model_slice(
                self._vault,
                self._handle,
                store=store,
                capability=capability,
            )
        except SecGemmaOnlineRiskOverlayVaultError:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Model slice access is unauthorized or stale"
            ) from None
        private = quarantine.get("private_quarantine")
        value = (
            None if type(private) is not dict else private.get("model_slice")
        )
        if (
            type(value) is not dict
            or value.get("model_slice_sha256")
            != canonical_sha256(_model_slice_index(value))
            or value["model_slice_sha256"]
            != self._model_slice_sha256
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Model slice changed after validation"
            )
        return copy.deepcopy(value)


class MarketTransport(Protocol):
    """Injected one-request transport used by the offline-testable adapter."""

    def fetch(self, url: str) -> Mapping[str, Any]: ...

    def acquisition_security_state(self) -> Mapping[str, Any]: ...


class _RejectYahooRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Yahoo redirect is forbidden"
        )


def _owned_content_length(headers: Any) -> int | None:
    transfer_encoding = headers.get("Transfer-Encoding")
    if transfer_encoding not in {None, "", "identity"}:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Yahoo transfer encoding is forbidden"
        )
    values = headers.get_all("Content-Length") or []
    if not values:
        return None
    if len(values) != 1:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Yahoo Content-Length is ambiguous"
        )
    value = values[0].strip()
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Yahoo Content-Length is invalid"
        )
    result = int(value)
    if result < 1 or result > MAX_MARKET_RESPONSE_BYTES:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Yahoo Content-Length exceeds the fixed byte ceiling"
        )
    return result


def _read_owned_yahoo_body(
    response: Any,
    *,
    deadline: float,
    declared_content_length: int | None,
) -> bytes:
    read1 = getattr(response, "read1", None)
    if not callable(read1):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Yahoo response lacks bounded streaming reads"
        )
    body = bytearray()
    while True:
        if time.monotonic() >= deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo response body crossed its deadline"
            )
        read_size = min(
            _YAHOO_BODY_READ_CHUNK_BYTES,
            MAX_MARKET_RESPONSE_BYTES + 1 - len(body),
        )
        try:
            chunk = read1(read_size)
        except TimeoutError:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo response body crossed its deadline"
            ) from None
        if type(chunk) is not bytes or len(chunk) > read_size:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo response returned an invalid bounded chunk"
            )
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > MAX_MARKET_RESPONSE_BYTES:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo response exceeded the fixed byte ceiling"
            )
        if (
            declared_content_length is not None
            and len(body) > declared_content_length
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo Content-Length does not match exact bytes"
            )
    result = bytes(body)
    if (
        not result
        or (
            declared_content_length is not None
            and declared_content_length != len(result)
        )
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Yahoo response body is empty or truncated"
        )
    return result


class _OwnedYahooMarketTransport:
    """Exact six-request Yahoo authority with owned counters and limits."""

    __slots__ = (
        "_deadline",
        "_expected_urls",
        "_opener",
        "_parent_deadline",
        "_request_count",
        "_response_bytes",
    )

    def __init__(
        self,
        expected_urls: tuple[str, ...],
        *,
        parent_deadline_monotonic: float,
    ) -> None:
        if (
            type(expected_urls) is not tuple
            or len(expected_urls) != MAX_MARKET_REQUESTS_PER_STAGE
            or not all(type(item) is str for item in expected_urls)
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Owned Yahoo transport requires six exact canonical URLs"
            )
        try:
            now = float(time.monotonic())
            parent_deadline = float(parent_deadline_monotonic)
        except Exception:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Owned Yahoo parent deadline clock failed"
            ) from None
        if (
            not math.isfinite(now)
            or not math.isfinite(parent_deadline)
            or not 0.0 < parent_deadline - now <= float(MAX_SEC_SECONDS)
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Owned Yahoo parent deadline is outside the frozen acquisition cap"
            )
        context = ssl.create_default_context()
        self._opener = build_opener(
            ProxyHandler({}),
            _RejectYahooRedirectHandler(),
            HTTPSHandler(context=context),
        )
        self._expected_urls = expected_urls
        self._request_count = 0
        self._response_bytes = 0
        self._parent_deadline = parent_deadline
        self._deadline: float | None = None

    def _begin_batch(self, *, deadline_monotonic: float) -> None:
        try:
            now = float(time.monotonic())
            deadline = float(deadline_monotonic)
        except Exception:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo batch deadline clock failed"
            ) from None
        if (
            self._deadline is not None
            or self._request_count != 0
            or self._response_bytes != 0
            or not math.isfinite(now)
            or not math.isfinite(deadline)
            or deadline > self._parent_deadline
            or not 0.0 < deadline - now <= float(MAX_MARKET_SECONDS)
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo batch deadline is stale, repeated, or outside its exact cap"
            )
        self._deadline = deadline

    def acquisition_security_state(self) -> dict[str, Any]:
        return {
            **_STRICT_TRANSPORT_FLAGS,
            "transport_max_requests": MAX_MARKET_REQUESTS_PER_STAGE,
            "transport_max_bytes": MAX_MARKET_BATCH_BYTES,
            "transport_max_seconds": float(MAX_MARKET_SECONDS),
        }

    def fetch(self, url: str) -> dict[str, Any]:
        started = time.monotonic()
        deadline = self._deadline
        if (
            deadline is None
            or started >= deadline
            or self._request_count >= len(self._expected_urls)
            or url != self._expected_urls[self._request_count]
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Owned Yahoo transport received a late, reordered, or foreign URL"
            )
        self._request_count += 1
        absolute_deadline = min(
            deadline, started + MARKET_REQUEST_TIMEOUT_SECONDS
        )
        timeout = absolute_deadline - time.monotonic()
        if timeout <= 0.0:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo request crossed its deadline"
            )
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
        try:
            with self._opener.open(request, timeout=timeout) as response:
                headers = response.headers
                declared = _owned_content_length(headers)
                body = _read_owned_yahoo_body(
                    response,
                    deadline=absolute_deadline,
                    declared_content_length=declared,
                )
                self._response_bytes += len(body)
                if (
                    self._response_bytes > MAX_MARKET_BATCH_BYTES
                    or time.monotonic() > deadline
                ):
                    raise SecGemmaOnlineRiskOverlayAcquisitionError(
                        "Yahoo batch exceeded its fixed byte or time ceiling"
                    )
                return {
                    "request_url": url,
                    "final_url": response.geturl(),
                    "status_code": response.getcode(),
                    "content_type": headers.get_content_type(),
                    "charset": headers.get_content_charset(),
                    "content_encoding": headers.get("Content-Encoding"),
                    "declared_content_length": declared,
                    "body": body,
                    "network_requests": 1,
                    "retries": 0,
                    "redirects": 0,
                }
        except SecGemmaOnlineRiskOverlayAcquisitionError:
            raise
        except HTTPError as exc:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"Yahoo request failed with HTTP status {exc.code}"
            ) from None
        except (URLError, OSError, TimeoutError):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Yahoo request failed"
            ) from None


def _strict_stage(value: Any) -> str:
    if type(value) is not str or value not in STAGES:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition stage must be development, confirmation, or final"
        )
    return value


def _sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Quarantine payload must be exact bytes"
        )
    return hashlib.sha256(value).hexdigest()


def _plain_json(value: Any, location: str, *, depth: int = 0) -> Any:
    if depth > 64:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{location} exceeds the JSON nesting limit"
        )
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"{location} contains a non-finite value"
            )
        return value
    if type(value) is list:
        return [
            _plain_json(item, f"{location}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"{location} keys must be exact strings"
            )
        return {
            key: _plain_json(item, f"{location}.{key}", depth=depth + 1)
            for key, item in value.items()
        }
    raise SecGemmaOnlineRiskOverlayAcquisitionError(
        f"{location} must contain only detached plain JSON"
    )


def _market_url(symbol: str, spec: Mapping[str, Any]) -> str:
    provider = quote(YAHOO_PROVIDER_SYMBOLS[symbol], safe="")
    query = urlencode(
        (
            ("period1", str(spec["period1_utc"])),
            ("period2", str(spec["period2_utc"])),
            ("interval", "1d"),
            ("includePrePost", "false"),
            ("includeAdjustedClose", "true"),
            ("events", "div,splits"),
        )
    )
    return f"{YAHOO_ENDPOINT}/{provider}?{query}"


def build_acquisition_plan(stage: str) -> dict[str, Any]:
    """Build the exact stage plan without I/O, clocks, values, or model access."""

    fixed_stage = _strict_stage(stage)
    spec = _STAGE_SPECS[fixed_stage]
    predecessor = spec["predecessor"]
    predecessor_plan_sha256 = (
        None
        if predecessor is None
        else build_acquisition_plan(predecessor)["acquisition_plan_sha256"]
    )
    market_requests = [
        {
            "sequence_number": index,
            "symbol": symbol,
            "provider_symbol": YAHOO_PROVIDER_SYMBOLS[symbol],
            "provider_timezone": YAHOO_PROVIDER_TIMEZONES[symbol],
            "url": _market_url(symbol, spec),
        }
        for index, symbol in enumerate(YAHOO_SYMBOLS, start=1)
    ]
    body = {
        "schema_version": ACQUISITION_PLAN_SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": fixed_stage,
        "attempt_id": spec["attempt_id"],
        "attempt_kind": spec["attempt_kind"],
        "predecessor_stage": predecessor,
        "predecessor_plan_sha256": predecessor_plan_sha256,
        "sec": {
            "issuer_cik": "0000320193",
            "catalog_root_url": MAIN_SUBMISSIONS_URL,
            "catalog_policy": (
                "exact current Apple Submissions plus every referenced historical "
                "Submissions file"
            ),
            "primary_document_policy": (
                "all and only eligible stage primary documents derived from the "
                "authenticated catalogue"
            ),
            "official_hosts": ["data.sec.gov", "www.sec.gov"],
            "forms": ["10-K", "10-Q"],
            "amendments": False,
            "stage_name": spec["sec_stage"],
            "availability_start": spec["sec_start"],
            "availability_end": spec["sec_end"],
            "minimum_filings": spec["minimum_filings"],
            "maximum_model_requests": spec["maximum_model_requests"],
            "request_cap": MAX_SEC_REQUESTS,
            "byte_cap": MAX_SEC_BYTES,
            "deadline_seconds": MAX_SEC_SECONDS,
            "requests_per_second_cap": MAX_SEC_REQUESTS_PER_SECOND,
            "retries": 0,
            "redirects": 0,
        },
        "market": {
            "provider_family": (
                "yahoo-finance-chart-v8-public-unauthenticated"
            ),
            "fixed_user_agent": YAHOO_USER_AGENT,
            "request_start": spec["request_start"],
            "request_end_exclusive": spec["request_end_exclusive"],
            "period1_utc": spec["period1_utc"],
            "period2_utc": spec["period2_utc"],
            "last_transport_session": spec["last_transport_session"],
            "last_value_session": spec["last_value_session"],
            "requests": market_requests,
            "request_count": MAX_MARKET_REQUESTS_PER_STAGE,
            "response_byte_cap": MAX_MARKET_RESPONSE_BYTES,
            "batch_byte_cap": MAX_MARKET_BATCH_BYTES,
            "deadline_seconds": MAX_MARKET_SECONDS,
            "retries": 0,
            "redirects": 0,
            "prefix_rule": (
                "the complete predecessor value-session prefix must reproduce "
                "identical sessions, explicit absences, and binary64 values"
            ),
        },
        "output_boundary": {
            "private_quarantine_only": True,
            "exact_raw_bytes": True,
            "request_receipts": True,
            "hashes_and_counts": True,
            "deterministic_blinded_model_requests": True,
            "decoded_market_values": False,
            "model_output": False,
            "semantic_extractions": False,
            "labels": False,
            "actions": False,
            "returns": False,
            "scores": False,
        },
    }
    return {**body, "acquisition_plan_sha256": canonical_sha256(body)}


def validate_acquisition_plan(
    value: Mapping[str, Any],
    *,
    expected_stage: str | None = None,
) -> dict[str, Any]:
    """Detached, effect-free validation of one exact plan."""

    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition plan must be an exact built-in mapping"
        )
    stage = value.get("stage")
    if expected_stage is not None and stage != _strict_stage(expected_stage):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition plan names the wrong stage"
        )
    expected = build_acquisition_plan(_strict_stage(stage))
    if value != expected:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition plan differs from the frozen deterministic plan"
        )
    return copy.deepcopy(expected)


def create_production_market_transport(
    plan: Mapping[str, Any],
    *,
    deadline_monotonic: float,
) -> MarketTransport:
    """Construct the only market transport accepted by production acquisition."""

    fixed = validate_acquisition_plan(plan)
    return _OwnedYahooMarketTransport(
        tuple(item["url"] for item in fixed["market"]["requests"]),
        parent_deadline_monotonic=deadline_monotonic,
    )


def _checked_now(clock: Callable[[], float]) -> float:
    if not callable(clock):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition clock must be callable"
        )
    try:
        value = float(clock())
    except Exception:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition clock failed"
        ) from None
    if not math.isfinite(value):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition clock must be finite"
        )
    return value


class _PacedSecTransport:
    """Delegate exact SEC requests while enforcing the two-per-second ceiling."""

    def __init__(
        self,
        inner: Any,
        *,
        clock: Callable[[], float],
        sleeper: Callable[[float], None],
        deadline: float,
    ) -> None:
        self._inner = inner
        self._clock = clock
        self._sleeper = sleeper
        self._deadline = deadline
        self._last_request_at: float | None = None

    def user_agent_audit(self) -> Any:
        value = getattr(self._inner, "user_agent_audit", None)
        return value() if callable(value) else value

    def acquisition_security_state(self) -> Any:
        provider = getattr(self._inner, "acquisition_security_state", None)
        if callable(provider):
            return provider()
        if type(self._inner) is SecAuditTransport:
            safe_provider = getattr(self._inner, "safe_state", None)
            safe = safe_provider() if callable(safe_provider) else None
            session = getattr(self._inner, "_session", None)
            budget = (
                None
                if type(safe) is not dict
                else safe.get("budget")
            )
            if (
                type(safe) is not dict
                or type(budget) is not dict
                or type(session) is not requests.Session
            ):
                raise SecGemmaOnlineRiskOverlayAcquisitionError(
                    "SEC transport lacks security evidence"
                )
            return {
                **_STRICT_TRANSPORT_FLAGS,
                "trust_env": session.trust_env,
                "proxies": bool(session.proxies),
                "transport_max_requests": budget["max_requests"],
                "transport_max_bytes": budget["max_bytes"],
                "transport_max_seconds": budget["max_seconds"],
            }
        else:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC transport lacks security evidence"
            )

    def fetch(self, url: str) -> Any:
        now = _checked_now(self._clock)
        if now >= self._deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC acquisition deadline was reached"
            )
        if self._last_request_at is not None:
            earliest = self._last_request_at + (
                1.0 / MAX_SEC_REQUESTS_PER_SECOND
            )
            if now < earliest:
                if earliest >= self._deadline:
                    raise SecGemmaOnlineRiskOverlayAcquisitionError(
                        "SEC pacing would cross the acquisition deadline"
                    )
                if not callable(self._sleeper):
                    raise SecGemmaOnlineRiskOverlayAcquisitionError(
                        "SEC pacing sleeper must be callable"
                    )
                try:
                    self._sleeper(earliest - now)
                except Exception:
                    raise SecGemmaOnlineRiskOverlayAcquisitionError(
                        "SEC pacing failed"
                    ) from None
                now = _checked_now(self._clock)
                if now < earliest:
                    raise SecGemmaOnlineRiskOverlayAcquisitionError(
                        "SEC transport exceeded the request-rate ceiling"
                    )
        if now >= self._deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC acquisition deadline was reached"
            )
        if type(self._inner) is SecAuditTransport:
            remaining = self._deadline - now
            current_timeout = getattr(self._inner, "_timeout", None)
            if (
                type(current_timeout) not in {int, float}
                or not math.isfinite(float(current_timeout))
                or float(current_timeout) <= 0.0
                or remaining <= 0.0
            ):
                raise SecGemmaOnlineRiskOverlayAcquisitionError(
                    "SEC transport timeout authority is invalid"
                )
            self._inner._timeout = min(float(current_timeout), remaining)
        self._last_request_at = now
        result = self._inner.fetch(url)
        if _checked_now(self._clock) > self._deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC acquisition deadline was exceeded"
            )
        return result


def _official_primary_url(record: Mapping[str, Any]) -> str:
    accession = record["accession_number"]
    filename = record["primary_document"]
    if (
        type(accession) is not str
        or _ACCESSION_RE.fullmatch(accession) is None
        or type(filename) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._~-]{0,255}", filename)
        is None
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Authenticated SEC record cannot derive one official primary URL"
        )
    return (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        f"{accession.replace('-', '')}/{quote(filename, safe='._~-')}"
    )


def _build_detached_stage_evidence(
    *,
    stage: str,
    universe: dict[str, Any],
    selected: list[dict[str, Any]],
    document_batch: Any,
) -> dict[str, Any]:
    """Build replay artifacts from the already-authenticated exact fetch batch."""

    authorized_stage = _STAGE_SPECS[stage]["sec_stage"]
    try:
        authenticated_receipts = json.loads(
            document_batch.request_receipts_json.decode("utf-8")
        )
        byte_manifest = json.loads(
            document_batch.byte_manifest_json.decode("utf-8")
        )
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Authenticated SEC document evidence is not canonical"
        ) from None
    if (
        type(authenticated_receipts) is not list
        or len(authenticated_receipts) != len(selected)
        or type(byte_manifest) is not dict
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Authenticated SEC document evidence is partial"
        )

    replay_receipts: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for sequence, (record, document, original) in enumerate(
        zip(
            selected,
            document_batch.documents,
            authenticated_receipts,
            strict=True,
        ),
        start=1,
    ):
        if (
            type(original) is not dict
            or original.get("schema_version")
            != SEC_REQUEST_RECEIPT_SCHEMA_VERSION
            or original.get("sequence_number") != sequence
            or original.get("purpose")
            != "authenticated_stage_access_primary_document"
            or original.get("requested_url") != document.url
            or original.get("final_url") != document.url
            or original.get("content_sha256")
            != f"sha256:{document.primary_document_sha256}"
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Authenticated SEC request receipt changed before replay"
            )
        replay_body = {
            key: value
            for key, value in original.items()
            if key != "request_receipt_sha256"
        }
        replay_body["purpose"] = f"{authorized_stage}_primary_document"
        replay_receipt = {
            **replay_body,
            "request_receipt_sha256": canonical_sha256(replay_body),
        }
        replay_receipts.append(replay_receipt)
        normalized = document.normalized_text
        try:
            normalized_result = normalize_filing_text(
                document.raw_primary_document.decode("latin-1")
            )
        except Exception:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC primary document could not be replay-normalized"
            ) from None
        if (
            normalized_result.text.encode("utf-8") != normalized
            or normalized_result.sha256
            != f"sha256:{document.normalized_text_sha256}"
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC normalized bytes changed before detached replay"
            )
        manifest_rows.append(
            {
                "accession_number": record["accession_number"],
                "primary_document_sha256": (
                    document.primary_document_sha256
                ),
                "normalized_text_sha256": (
                    document.normalized_text_sha256
                ),
                "primary_document_bytes": len(
                    document.raw_primary_document
                ),
                "normalized_text_bytes": len(normalized),
            }
        )
        artifact_rows.append(
            {
                "accession_number": record["accession_number"],
                "availability_session": record["availability_session"],
                "primary_document": record["primary_document"],
                "url": document.url,
                "primary_document_sha256": (
                    document.primary_document_sha256
                ),
                "primary_document_bytes": len(
                    document.raw_primary_document
                ),
                "normalized_text_sha256": (
                    document.normalized_text_sha256
                ),
                "normalized_text_bytes": len(normalized),
                "normalized_character_count": (
                    normalized_result.character_count
                ),
                "normalized_text_usable": normalized_result.usable,
                "request_receipt_sha256": replay_receipt[
                    "request_receipt_sha256"
                ],
            }
        )

    content_manifest = build_stage_content_manifest(
        artifact_stage=authorized_stage,
        corpus_universe_sha256=universe["universe_sha256"],
        documents=manifest_rows,
        universe_manifest=universe,
    )
    receipt_set_hash = canonical_sha256(replay_receipts)
    artifact_body = {
        "schema_version": STAGE_ARTIFACT_SCHEMA_VERSION,
        "contract_sha256": canonical_sha256(
            build_sec_contract_manifest()
        ),
        "artifact_stage": authorized_stage,
        "corpus_universe_sha256": universe["universe_sha256"],
        "catalog_artifact_sha256": universe[
            "catalog_artifact_sha256"
        ],
        "content_manifest_sha256": content_manifest[
            "content_manifest_sha256"
        ],
        "document_count": len(artifact_rows),
        "documents": artifact_rows,
        "request_receipts": replay_receipts,
        "request_receipts_sha256": receipt_set_hash,
        "acquisition_request_count": len(artifact_rows),
        "acquisition_bytes": sum(
            len(document.raw_primary_document)
            for document in document_batch.documents
        ),
        "transport_security": byte_manifest["transport_security"],
        "outer_budget_role": (
            "post_transport_reconciliation_not_streaming_protection"
        ),
        "user_agent_sha256": byte_manifest["user_agent_sha256"],
        "selection_policy": (
            "all_and_only_authorized_stage_universe_primary_documents"
        ),
        "arbitrary_urls_or_accessions_accepted": False,
        "sampling_dropping_cache_or_substitution_allowed": False,
        "evidence_boundary": {
            "metadata": (
                "exact_current_plus_referenced_official_sec_submissions_bytes"
            ),
            "document": "exact_official_sec_primary_document_bytes",
            "legacy_24_slot_audit_is_exhaustive_catalog_proof": False,
            "full_predictive_corpus_master_index_sgml_or_index_reconciled": False,
            "sgml_or_master_index_reconciliation_claimed_for_2025_2026": False,
        },
        "contains_outcomes_market_data_or_model_output": False,
    }
    stage_artifact = {
        **artifact_body,
        "stage_artifact_sha256": canonical_sha256(artifact_body),
    }
    return {
        "authenticated_stage_request_receipts": authenticated_receipts,
        "stage_request_receipts": replay_receipts,
        "stage_content_manifest": content_manifest,
        "stage_artifact": stage_artifact,
    }


def _receipt(
    *,
    sequence_number: int,
    channel: str,
    purpose: str,
    requested_url: str,
    final_url: str,
    status_code: int,
    content_type: str,
    byte_count: int,
    body_sha256: str,
    network_requests: int,
    retries: int,
    redirects: int,
    identity_sha256: str | None,
) -> dict[str, Any]:
    body = {
        "schema_version": ACQUISITION_RECEIPT_SCHEMA_VERSION,
        "sequence_number": sequence_number,
        "channel": channel,
        "purpose": purpose,
        "requested_url": requested_url,
        "final_url": final_url,
        "status_code": status_code,
        "content_type": content_type,
        "byte_count": byte_count,
        "body_sha256": body_sha256,
        "network_requests": network_requests,
        "retries": retries,
        "redirects": redirects,
        "identity_sha256": identity_sha256,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _sec_receipt(
    *,
    sequence_number: int,
    purpose: str,
    raw_receipt: Mapping[str, Any],
    payload: bytes,
) -> dict[str, Any]:
    value = _plain_json(dict(raw_receipt), "SEC request receipt")
    required = {
        "requested_url",
        "final_url",
        "status_code",
        "content_type",
        "size_bytes",
        "content_sha256",
        "cache_hit",
        "user_agent_sha256",
        "network_requests",
        "retries",
        "redirects",
    }
    if not required.issubset(value):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC request receipt is incomplete"
        )
    url = value["requested_url"]
    content_hash = value["content_sha256"]
    if (
        type(url) is not str
        or value["final_url"] != url
        or type(value["content_type"]) is not str
        or value["status_code"] != 200
        or value["size_bytes"] != len(payload)
        or content_hash != f"sha256:{_sha256_bytes(payload)}"
        or value["cache_hit"] is not False
        or value["network_requests"] != 1
        or value["retries"] != 0
        or value["redirects"] != 0
        or type(value["user_agent_sha256"]) is not str
        or _SHA256_RE.fullmatch(value["user_agent_sha256"]) is None
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC request receipt does not reconcile to exact bytes"
        )
    return _receipt(
        sequence_number=sequence_number,
        channel="sec",
        purpose=purpose,
        requested_url=url,
        final_url=url,
        status_code=200,
        content_type=value["content_type"],
        byte_count=len(payload),
        body_sha256=_sha256_bytes(payload),
        network_requests=1,
        retries=0,
        redirects=0,
        identity_sha256=value["user_agent_sha256"],
    )


def _market_security_state(transport: Any) -> dict[str, Any]:
    provider = getattr(transport, "acquisition_security_state", None)
    if not callable(provider):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market transport lacks exact security evidence"
        )
    try:
        state = _plain_json(
            dict(provider()), "market transport security evidence"
        )
    except SecGemmaOnlineRiskOverlayAcquisitionError:
        raise
    except Exception:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market transport security evidence could not be read"
        ) from None
    if set(state) != set(_TRANSPORT_SECURITY_KEYS):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market transport security schema is not exact"
        )
    for key, expected in _STRICT_TRANSPORT_FLAGS.items():
        if type(state[key]) is not type(expected) or state[key] != expected:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Market transport permits a forbidden retry, redirect, proxy, "
                "cache, or unbounded body"
            )
    if (
        type(state["transport_max_requests"]) is not int
        or state["transport_max_requests"]
        != MAX_MARKET_REQUESTS_PER_STAGE
        or type(state["transport_max_bytes"]) is not int
        or not 1 <= state["transport_max_bytes"] <= MAX_MARKET_BATCH_BYTES
        or type(state["transport_max_seconds"]) not in {int, float}
        or not 0.0
        < float(state["transport_max_seconds"])
        <= MAX_MARKET_SECONDS
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market transport limits exceed the frozen caps"
        )
    return state


def _reject_duplicate_pairs(location: str):
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SecGemmaOnlineRiskOverlayAcquisitionError(
                    f"{location} contains a duplicate JSON key"
                )
            result[key] = value
        return result

    return hook


def _strict_json_bytes(payload: bytes, location: str) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{location} must be nonempty exact bytes"
        )
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs(location),
            parse_constant=lambda token: (_ for _ in ()).throw(
                SecGemmaOnlineRiskOverlayAcquisitionError(
                    f"{location} contains non-finite JSON token {token}"
                )
            ),
        )
    except SecGemmaOnlineRiskOverlayAcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{location} is not strict UTF-8 JSON"
        ) from None
    if type(value) is not dict:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{location} must contain one JSON object"
        )
    return value


def _number_hex(value: Any, location: str) -> str:
    if type(value) not in {int, float}:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{location} must be one JSON number"
        )
    observed = float(value)
    if not math.isfinite(observed):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{location} must be finite"
        )
    return observed.hex()


def _market_rows(
    *,
    symbol: str,
    payload: bytes,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    value = _strict_json_bytes(payload, f"{symbol} Yahoo response")
    if set(value) != {"chart"} or type(value["chart"]) is not dict:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{symbol} Yahoo response root changed"
        )
    chart = value["chart"]
    if (
        set(chart) != {"result", "error"}
        or chart["error"] is not None
        or type(chart["result"]) is not list
        or len(chart["result"]) != 1
        or type(chart["result"][0]) is not dict
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{symbol} Yahoo chart result is incomplete"
        )
    result = chart["result"][0]
    if not {"meta", "timestamp", "indicators"}.issubset(result):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{symbol} Yahoo chart result lacks required fields"
        )
    meta = result["meta"]
    timestamps = result["timestamp"]
    indicators = result["indicators"]
    if (
        type(meta) is not dict
        or meta.get("symbol") != YAHOO_PROVIDER_SYMBOLS[symbol]
        or meta.get("exchangeTimezoneName")
        != YAHOO_PROVIDER_TIMEZONES[symbol]
        or meta.get("dataGranularity") != "1d"
        or type(timestamps) is not list
        or type(indicators) is not dict
        or set(indicators) != {"quote", "adjclose"}
        or type(indicators["quote"]) is not list
        or len(indicators["quote"]) != 1
        or type(indicators["quote"][0]) is not dict
        or type(indicators["adjclose"]) is not list
        or len(indicators["adjclose"]) != 1
        or type(indicators["adjclose"][0]) is not dict
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{symbol} Yahoo metadata or indicators changed"
        )
    quote_row = indicators["quote"][0]
    adjusted_row = indicators["adjclose"][0]
    fields = ("open", "high", "low", "close", "volume")
    if set(quote_row) != set(fields) or set(adjusted_row) != {"adjclose"}:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{symbol} Yahoo indicator fields changed"
        )
    arrays = {field: quote_row[field] for field in fields}
    arrays["adjusted_close"] = adjusted_row["adjclose"]
    if any(
        type(items) is not list or len(items) != len(timestamps)
        for items in arrays.values()
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            f"{symbol} Yahoo arrays are partial or misaligned"
        )

    period1 = plan["market"]["period1_utc"]
    period2 = plan["market"]["period2_utc"]
    expected_zone = ZoneInfo(YAHOO_PROVIDER_TIMEZONES[symbol])
    rows: list[dict[str, Any]] = []
    prior_session: str | None = None
    for index, timestamp in enumerate(timestamps):
        if (
            type(timestamp) is not int
            or type(timestamp) is bool
            or not period1 <= timestamp < period2
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"{symbol} Yahoo timestamp is outside the frozen request"
            )
        utc_value = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        local_value = utc_value.astimezone(expected_zone)
        session = local_value.date().isoformat()
        if utc_value.date() != local_value.date():
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"{symbol} Yahoo timestamp crosses its expected session date"
            )
        if prior_session is not None and session <= prior_session:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"{symbol} Yahoo sessions are duplicated or reordered"
            )
        prior_session = session
        raw_values = {field: arrays[field][index] for field in arrays}
        null_count = sum(item is None for item in raw_values.values())
        if null_count not in {0, len(raw_values)}:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"{symbol} Yahoo row has a partial observation"
            )
        if null_count:
            row = {"session": session, "available": False, "values": None}
        else:
            numeric = {
                field: float.fromhex(_number_hex(item, f"{symbol}.{field}"))
                for field, item in raw_values.items()
            }
            if (
                any(
                    numeric[field] <= 0.0
                    for field in (
                        "open",
                        "high",
                        "low",
                        "close",
                        "adjusted_close",
                    )
                )
                or numeric["volume"] < 0.0
                or numeric["high"]
                < max(numeric["open"], numeric["low"], numeric["close"])
                or numeric["low"]
                > min(numeric["open"], numeric["high"], numeric["close"])
            ):
                raise SecGemmaOnlineRiskOverlayAcquisitionError(
                    f"{symbol} Yahoo OHLCV row is invalid"
                )
            row = {
                "session": session,
                "available": True,
                "values": {
                    field: numeric[field].hex() for field in raw_values
                },
            }
        rows.append(row)
    cutoff = plan["market"]["last_transport_session"]
    if symbol == "AAPL":
        required = [
            session
            for session in EXPECTED_MARKET_HISTORY_SESSIONS
            if session <= cutoff
        ]
        if (
            [row["session"] for row in rows] != required
            or any(not row["available"] for row in rows)
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "AAPL Yahoo rows do not cover every frozen stage session"
            )
    else:
        first = CONTEXT_FIRST_ACCEPTED_SESSION[symbol]
        required = {
            session
            for session in EXPECTED_MARKET_HISTORY_SESSIONS
            if first <= session <= cutoff
        }
        permitted_missing = {
            session
            for session in CONTEXT_ALLOWED_MISSING_SESSIONS[symbol]
            if session <= cutoff
        }
        observed = {row["session"] for row in rows}
        available = {
            row["session"] for row in rows if row["available"]
        }
        explicit_absent = observed - available
        if (
            not rows
            or rows[0]["session"] != first
            or rows[-1]["session"] != cutoff
            or not observed.issubset(required)
            or not (required - available).issubset(permitted_missing)
            or not explicit_absent.issubset(permitted_missing)
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"{symbol} Yahoo context coverage is truncated or unexplained"
            )
    return rows


def _market_commitments(
    raw_by_symbol: Mapping[str, bytes],
    plan: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    if type(raw_by_symbol) is not dict or set(raw_by_symbol) != set(
        YAHOO_SYMBOLS
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market quarantine must contain exactly the six frozen symbols"
        )
    cutoff = plan["market"]["last_value_session"]
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    commitments: dict[str, str] = {}
    for symbol in YAHOO_SYMBOLS:
        rows = _market_rows(
            symbol=symbol,
            payload=raw_by_symbol[symbol],
            plan=plan,
        )
        prefix = [row for row in rows if row["session"] <= cutoff]
        if not prefix:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                f"{symbol} Yahoo value prefix is empty"
            )
        rows_by_symbol[symbol] = rows
        commitments[symbol] = canonical_sha256(prefix)
    return commitments, rows_by_symbol


def _validate_primary_document(item: Mapping[str, Any], stage: str) -> None:
    expected_keys = {
        "accession_number",
        "form",
        "availability_session",
        "acceptance_datetime",
        "official_url",
        "body",
    }
    if type(item) is not dict or set(item) != expected_keys:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC primary-document quarantine schema changed"
        )
    accession = item["accession_number"]
    form = item["form"]
    availability = item["availability_session"]
    acceptance = item["acceptance_datetime"]
    official_url = item["official_url"]
    match = (
        _PRIMARY_URL_RE.fullmatch(official_url)
        if type(official_url) is str
        else None
    )
    spec = _STAGE_SPECS[stage]
    if (
        type(accession) is not str
        or _ACCESSION_RE.fullmatch(accession) is None
        or form not in {"10-K", "10-Q"}
        or type(availability) is not str
        or not spec["sec_start"] <= availability <= spec["sec_end"]
        or type(acceptance) is not str
        or re.fullmatch(r"[0-9]{14}", acceptance) is None
        or match is None
        or match.group("accession") != accession.replace("-", "")
        or type(item["body"]) is not bytes
        or not item["body"]
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC primary document is not one exact official non-amendment 10-K/10-Q"
        )


def _normalize_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    if type(documents) is not list:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC primary documents must be one exact list"
        )
    normalized: list[dict[str, Any]] = []
    prior_key: tuple[str, str, str] | None = None
    seen: set[str] = set()
    for raw in documents:
        _validate_primary_document(raw, stage)
        key = (
            raw["availability_session"],
            raw["acceptance_datetime"],
            raw["accession_number"],
        )
        if prior_key is not None and key <= prior_key:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC primary documents are duplicated or reordered"
            )
        prior_key = key
        if raw["accession_number"] in seen:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC primary accession is duplicated"
            )
        seen.add(raw["accession_number"])
        try:
            text = normalize_filing_text(
                raw["body"].decode("latin-1")
            ).text
        except Exception:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC primary document cannot be deterministically normalized"
            ) from None
        if not text:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC primary document has no normalized visible text"
            )
        normalized.append({**raw, "normalized_text": text})
    spec = _STAGE_SPECS[stage]
    if not (
        spec["minimum_filings"]
        <= len(normalized)
        <= spec["maximum_model_requests"]
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC stage filing count is outside the frozen complete-batch bounds"
        )
    return normalized


def _carry_in_by_form(
    carry_in_documents: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> dict[str, str]:
    expected_count = 0 if stage == DEVELOPMENT else 2
    if (
        type(carry_in_documents) is not list
        or len(carry_in_documents) != expected_count
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC carry-in quarantine is incomplete"
        )
    result: dict[str, str] = {}
    for item in carry_in_documents:
        predecessor = _STAGE_SPECS[stage]["predecessor"]
        if predecessor is None:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Development cannot contain carry-in documents"
            )
        _validate_primary_document(item, predecessor)
        form = item["form"]
        if form in result:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC carry-in duplicates a filing form"
            )
        try:
            result[form] = normalize_filing_text(
                item["body"].decode("latin-1")
            ).text
        except Exception:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC carry-in cannot be deterministically normalized"
            ) from None
    if stage != DEVELOPMENT and set(result) != {"10-K", "10-Q"}:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC carry-in must bind the latest prior 10-K and 10-Q"
        )
    return result


def _blinded_request_bytes(model_payload: Mapping[str, Any]) -> bytes:
    # These are the exact canonical bytes later passed to the pinned Ollama
    # client.  The payload already contains the frozen system prompt and the
    # redacted sentence-only user message.
    return canonical_json_bytes(
        _plain_json(dict(model_payload), "model payload")
    )


def _build_model_requests(
    documents: Sequence[Mapping[str, Any]],
    *,
    carry_in_documents: Sequence[Mapping[str, Any]],
    stage: str,
) -> list[dict[str, Any]]:
    normalized = _normalize_documents(list(documents), stage=stage)
    prior_by_form = _carry_in_by_form(
        list(carry_in_documents), stage=stage
    )
    requests: list[dict[str, Any]] = []
    for item in normalized:
        prior = prior_by_form.get(item["form"])
        try:
            preprocessed = preprocess_filing_event(
                current_normalized_text=item["normalized_text"],
                prior_same_form_normalized_text=prior,
                identity_lexicon=CANONICAL_IDENTITY_LEXICON,
            )
        except Exception:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Blinded filing request preprocessing failed"
            ) from None
        request_bytes = _blinded_request_bytes(preprocessed["model_payload"])
        requests.append(
            {
                "schema_version": BLINDED_MODEL_REQUEST_SCHEMA_VERSION,
                "accession_number": item["accession_number"],
                "form": item["form"],
                "availability_session": item["availability_session"],
                "preprocessed_event_sha256": preprocessed[
                    "preprocessed_event_sha256"
                ],
                "supplied_sentence_ids": [
                    sentence["id"]
                    for sentence in preprocessed["sentences"]
                ],
                "request_sha256": _sha256_bytes(request_bytes),
                "request_bytes": request_bytes,
            }
        )
        prior_by_form[item["form"]] = item["normalized_text"]
    return requests


def _content_manifest_from_quarantined_documents(
    *,
    artifact_stage: str,
    universe: Mapping[str, Any],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    records = [
        record
        for record in universe["records"]
        if record["artifact_stage"] == artifact_stage
    ]
    records.sort(
        key=lambda record: (
            record["availability_session"],
            record["accession_number"],
        )
    )
    by_accession = {
        document["accession_number"]: document for document in documents
    }
    if [record["accession_number"] for record in records] != list(
        by_accession
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Quarantined stage documents differ from the replayed universe"
        )
    rows: list[dict[str, Any]] = []
    for record in records:
        document = by_accession[record["accession_number"]]
        normalized = normalize_filing_text(
            document["body"].decode("latin-1")
        ).text.encode("utf-8")
        rows.append(
            {
                "accession_number": record["accession_number"],
                "primary_document_sha256": _sha256_bytes(
                    document["body"]
                ),
                "normalized_text_sha256": _sha256_bytes(normalized),
                "primary_document_bytes": len(document["body"]),
                "normalized_text_bytes": len(normalized),
            }
        )
    return build_stage_content_manifest(
        artifact_stage=artifact_stage,
        corpus_universe_sha256=universe["universe_sha256"],
        documents=rows,
        universe_manifest=universe,
    )


def _build_universe_event_proofs(
    *,
    stage: str,
    universe: dict[str, Any],
    current_documents: list[dict[str, Any]],
    predecessor_bundles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents_by_artifact_stage: dict[str, list[dict[str, Any]]] = {}
    for bundle in predecessor_bundles:
        predecessor_stage = bundle["public_manifest"]["stage"]
        artifact_stage = _STAGE_SPECS[predecessor_stage]["sec_stage"]
        documents_by_artifact_stage[artifact_stage] = bundle[
            "private_quarantine"
        ]["sec_primary_documents"]
    current_artifact_stage = _STAGE_SPECS[stage]["sec_stage"]
    documents_by_artifact_stage[current_artifact_stage] = current_documents
    manifests = {
        artifact_stage: _content_manifest_from_quarantined_documents(
            artifact_stage=artifact_stage,
            universe=universe,
            documents=documents,
        )
        for artifact_stage, documents in documents_by_artifact_stage.items()
    }
    records = list(universe["records"])
    proofs: list[dict[str, Any]] = []
    for document in current_documents:
        current = next(
            record
            for record in records
            if record["accession_number"] == document["accession_number"]
        )
        prior_candidates = [
            record
            for record in records
            if record["form"] == current["form"]
            and (
                record["availability_session"],
                record["accession_number"],
            )
            < (
                current["availability_session"],
                current["accession_number"],
            )
        ]
        prior = (
            None
            if not prior_candidates
            else max(
                prior_candidates,
                key=lambda record: (
                    record["availability_session"],
                    record["accession_number"],
                ),
            )
        )
        needed = {current["artifact_stage"]}
        if prior is not None:
            needed.add(prior["artifact_stage"])
        selected_manifests = {
            artifact_stage: manifests[artifact_stage]
            for artifact_stage in needed
        }
        proofs.append(
            build_validated_universe_event_proof(
                universe_manifest=universe,
                expected_corpus_universe_sha256=universe[
                    "universe_sha256"
                ],
                current_accession_number=document["accession_number"],
                content_manifests_by_stage=selected_manifests,
                expected_content_manifest_sha256s={
                    artifact_stage: manifest[
                        "content_manifest_sha256"
                    ]
                    for artifact_stage, manifest in selected_manifests.items()
                },
                session_dates=EXPECTED_SESSIONS,
            )
        )
    return proofs


def _model_slice_index(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "stage",
        "attempt_id",
        "model_requests",
        "universe_event_proofs",
        "model_slice_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Model slice schema changed"
        )
    requests = value["model_requests"]
    proofs = value["universe_event_proofs"]
    if (
        type(requests) is not list
        or type(proofs) is not list
        or len(requests) != len(proofs)
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Model slice requests or proofs changed"
        )
    request_index: list[dict[str, Any]] = []
    for request, proof in zip(requests, proofs, strict=True):
        if (
            type(request) is not dict
            or type(request.get("request_bytes")) is not bytes
            or request.get("request_sha256")
            != _sha256_bytes(request["request_bytes"])
            or type(proof) is not dict
            or proof.get("current_record", {}).get("accession_number")
            != request.get("accession_number")
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Model slice request/proof binding changed"
            )
        request_index.append(
            {
                key: request[key]
                for key in (
                    "schema_version",
                    "accession_number",
                    "form",
                    "availability_session",
                    "preprocessed_event_sha256",
                    "supplied_sentence_ids",
                    "request_sha256",
                )
            }
        )
    return {
        "stage": value["stage"],
        "attempt_id": value["attempt_id"],
        "model_request_index": request_index,
        "universe_event_proofs": proofs,
    }


def _build_model_slice(
    *,
    plan: Mapping[str, Any],
    model_requests: list[dict[str, Any]],
    universe_event_proofs: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {
        "stage": plan["stage"],
        "attempt_id": plan["attempt_id"],
        "model_requests": copy.deepcopy(model_requests),
        "universe_event_proofs": copy.deepcopy(
            universe_event_proofs
        ),
    }
    indexed = {**body, "model_slice_sha256": "0" * 64}
    return {
        **body,
        "model_slice_sha256": canonical_sha256(
            _model_slice_index(indexed)
        ),
    }


def _stage_slice_index(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "stage",
        "attempt_id",
        "attempt_kind",
        "last_value_session",
        "market_rows",
        "model_requests",
        "universe_event_proofs",
        "slice_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Stage slice schema changed"
        )
    requests = value["model_requests"]
    proofs = value["universe_event_proofs"]
    if (
        type(requests) is not list
        or type(proofs) is not list
        or len(proofs) != len(requests)
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Stage slice model requests or universe proofs changed"
        )
    request_index: list[dict[str, Any]] = []
    for request in requests:
        if (
            type(request) is not dict
            or type(request.get("request_bytes")) is not bytes
            or request.get("request_sha256")
            != _sha256_bytes(request["request_bytes"])
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Stage slice contains a changed model request"
            )
        request_index.append(
            {
                key: request[key]
                for key in (
                    "schema_version",
                    "accession_number",
                    "form",
                    "availability_session",
                    "preprocessed_event_sha256",
                    "supplied_sentence_ids",
                    "request_sha256",
                )
            }
        )
    for request, proof in zip(requests, proofs, strict=True):
        if (
            type(proof) is not dict
            or proof.get("current_record", {}).get("accession_number")
            != request["accession_number"]
            or type(proof.get("universe_event_proof_sha256")) is not str
            or re.fullmatch(
                r"[0-9a-f]{64}",
                proof["universe_event_proof_sha256"],
            )
            is None
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Stage slice universe proof is changed or reordered"
            )
    return {
        "schema_version": value["schema_version"],
        "stage": value["stage"],
        "attempt_id": value["attempt_id"],
        "attempt_kind": value["attempt_kind"],
        "last_value_session": value["last_value_session"],
        "market_rows": value["market_rows"],
        "model_request_index": request_index,
        "universe_event_proofs": value["universe_event_proofs"],
    }


def _build_stage_slice(
    *,
    plan: Mapping[str, Any],
    rows_by_symbol: Mapping[str, list[dict[str, Any]]],
    model_requests: list[dict[str, Any]],
    universe_event_proofs: list[dict[str, Any]],
) -> dict[str, Any]:
    cutoff = plan["market"]["last_value_session"]
    body = {
        "schema_version": STAGE_SLICE_SCHEMA_VERSION,
        "stage": plan["stage"],
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "last_value_session": cutoff,
        "market_rows": {
            symbol: [
                copy.deepcopy(row)
                for row in rows_by_symbol[symbol]
                if row["session"] <= cutoff
            ]
            for symbol in YAHOO_SYMBOLS
        },
        "model_requests": copy.deepcopy(model_requests),
        "universe_event_proofs": copy.deepcopy(
            universe_event_proofs
        ),
    }
    indexed = {**body, "slice_sha256": "0" * 64}
    return {
        **body,
        "slice_sha256": canonical_sha256(_stage_slice_index(indexed)),
    }


def _private_index(private: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sec_replay_evidence_sha256": canonical_sha256(
            private["sec_replay_evidence"]
        ),
        "sec_catalog_sources": [
            {
                "name": item["name"],
                "url": item["url"],
                "byte_count": len(item["body"]),
                "sha256": _sha256_bytes(item["body"]),
            }
            for item in private["sec_catalog_sources"]
        ],
        "sec_primary_documents": [
            {
                "accession_number": item["accession_number"],
                "form": item["form"],
                "availability_session": item["availability_session"],
                "acceptance_datetime": item["acceptance_datetime"],
                "official_url": item["official_url"],
                "byte_count": len(item["body"]),
                "sha256": _sha256_bytes(item["body"]),
            }
            for item in private["sec_primary_documents"]
        ],
        "carry_in_documents": [
            {
                "accession_number": item["accession_number"],
                "form": item["form"],
                "availability_session": item["availability_session"],
                "acceptance_datetime": item["acceptance_datetime"],
                "official_url": item["official_url"],
                "byte_count": len(item["body"]),
                "sha256": _sha256_bytes(item["body"]),
            }
            for item in private["carry_in_documents"]
        ],
        "market_responses": [
            {
                "symbol": item["symbol"],
                "url": item["url"],
                "byte_count": len(item["body"]),
                "sha256": _sha256_bytes(item["body"]),
            }
            for item in private["market_responses"]
        ],
        "request_receipts_sha256": canonical_sha256(
            private["request_receipts"]
        ),
        "model_requests": [
            {
                key: (
                    _sha256_bytes(item["request_bytes"])
                    if key == "request_bytes_sha256"
                    else item[key]
                )
                for key in (
                    "schema_version",
                    "accession_number",
                    "form",
                    "availability_session",
                    "preprocessed_event_sha256",
                    "supplied_sentence_ids",
                    "request_sha256",
                    "request_bytes_sha256",
                )
            }
            for item in (
                {
                    **request,
                    "request_bytes_sha256": request["request_bytes"],
                }
                for request in private["model_requests"]
            )
        ],
        "model_slice_sha256": private["model_slice"][
            "model_slice_sha256"
        ],
        "stage_slice_sha256": private["stage_slice"]["slice_sha256"],
    }


def _public_manifest(
    *,
    plan: Mapping[str, Any],
    identity_sha256: str,
    predecessor_bundle_sha256: str | None,
    private: Mapping[str, Any],
    market_commitments: Mapping[str, str],
    prefix_continuity_verified: bool,
) -> dict[str, Any]:
    index = _private_index(private)
    body = {
        "schema_version": ACQUISITION_MANIFEST_SCHEMA_VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "stage": plan["stage"],
        "attempt_id": plan["attempt_id"],
        "attempt_kind": plan["attempt_kind"],
        "acquisition_plan_sha256": plan["acquisition_plan_sha256"],
        "predecessor_bundle_sha256": predecessor_bundle_sha256,
        "sec_identity_sha256": identity_sha256,
        "sec_catalog_source_count": len(private["sec_catalog_sources"]),
        "sec_primary_document_count": len(
            private["sec_primary_documents"]
        ),
        "market_response_count": len(private["market_responses"]),
        "model_request_count": len(private["model_requests"]),
        "total_raw_byte_count": sum(
            row["byte_count"]
            for key in (
                "sec_catalog_sources",
                "sec_primary_documents",
                "market_responses",
            )
            for row in index[key]
        ),
        "private_index_sha256": canonical_sha256(index),
        "market_prefix_commitment_sha256s": dict(market_commitments),
        "prefix_continuity_verified": prefix_continuity_verified,
        "complete_batch": True,
        "quarantine_only": True,
        "decoded_market_values_returned": False,
        "model_output_returned": False,
        "semantic_extractions_returned": False,
        "labels_actions_returns_or_scores_returned": False,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _bundle(
    *,
    plan: Mapping[str, Any],
    identity_sha256: str,
    predecessor_bundle_sha256: str | None,
    private: Mapping[str, Any],
    market_commitments: Mapping[str, str],
    prefix_continuity_verified: bool,
) -> dict[str, Any]:
    manifest = _public_manifest(
        plan=plan,
        identity_sha256=identity_sha256,
        predecessor_bundle_sha256=predecessor_bundle_sha256,
        private=private,
        market_commitments=market_commitments,
        prefix_continuity_verified=prefix_continuity_verified,
    )
    body = {
        "schema_version": ACQUISITION_BUNDLE_SCHEMA_VERSION,
        "public_manifest": manifest,
        "private_index_sha256": manifest["private_index_sha256"],
    }
    return {
        **body,
        "private_quarantine": private,
        "bundle_sha256": canonical_sha256(body),
    }


def _bundle_raw_market(bundle: Mapping[str, Any]) -> dict[str, bytes]:
    private = bundle["private_quarantine"]
    return {
        item["symbol"]: item["body"]
        for item in private["market_responses"]
    }


def _last_documents_by_form(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents = bundle["private_quarantine"]["sec_primary_documents"]
    result: dict[str, dict[str, Any]] = {}
    for item in documents:
        result[item["form"]] = copy.deepcopy(item)
    if set(result) != {"10-K", "10-Q"}:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Predecessor lacks a latest 10-K or 10-Q carry-in"
        )
    return [result["10-K"], result["10-Q"]]


def _validate_receipt(value: Mapping[str, Any]) -> None:
    keys = {
        "schema_version",
        "sequence_number",
        "channel",
        "purpose",
        "requested_url",
        "final_url",
        "status_code",
        "content_type",
        "byte_count",
        "body_sha256",
        "network_requests",
        "retries",
        "redirects",
        "identity_sha256",
        "receipt_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition request receipt schema changed"
        )
    body = {key: value[key] for key in keys if key != "receipt_sha256"}
    if (
        value["schema_version"] != ACQUISITION_RECEIPT_SCHEMA_VERSION
        or type(value["sequence_number"]) is not int
        or value["sequence_number"] < 1
        or value["channel"] not in {"sec", "market"}
        or type(value["purpose"]) is not str
        or type(value["requested_url"]) is not str
        or value["final_url"] != value["requested_url"]
        or type(value["status_code"]) is not int
        or value["status_code"] != 200
        or type(value["content_type"]) is not str
        or type(value["byte_count"]) is not int
        or value["byte_count"] < 1
        or type(value["body_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value["body_sha256"]) is None
        or type(value["network_requests"]) is not int
        or value["network_requests"] != 1
        or type(value["retries"]) is not int
        or value["retries"] != 0
        or type(value["redirects"]) is not int
        or value["redirects"] != 0
        or (
            value["identity_sha256"] is not None
            and (
                type(value["identity_sha256"]) is not str
                or _SHA256_RE.fullmatch(value["identity_sha256"]) is None
            )
        )
        or value["receipt_sha256"] != canonical_sha256(body)
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition request receipt is not canonical"
        )


def _validate_private_schema(private: Mapping[str, Any], stage: str) -> None:
    if type(private) is not dict or set(private) != {
        "sec_replay_evidence",
        "sec_catalog_sources",
        "sec_primary_documents",
        "carry_in_documents",
        "market_responses",
        "request_receipts",
        "model_requests",
        "model_slice",
        "stage_slice",
    }:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Private quarantine schema changed"
        )
    list_keys = {
        "sec_catalog_sources",
        "sec_primary_documents",
        "carry_in_documents",
        "market_responses",
        "request_receipts",
        "model_requests",
    }
    if any(type(private[key]) is not list for key in list_keys):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Private quarantine collections must be exact lists"
        )
    if (
        type(private["sec_replay_evidence"]) is not dict
        or type(private["model_slice"]) is not dict
        or type(private["stage_slice"]) is not dict
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Private replay evidence or stage slice changed"
        )
    source_names: set[str] = set()
    for item in private["sec_catalog_sources"]:
        if (
            type(item) is not dict
            or set(item) != {"name", "url", "body"}
            or type(item["name"]) is not str
            or item["name"] in source_names
            or type(item["url"]) is not str
            or urlsplit(item["url"]).scheme != "https"
            or urlsplit(item["url"]).hostname != "data.sec.gov"
            or type(item["body"]) is not bytes
            or not item["body"]
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC catalogue-source quarantine is invalid"
            )
        source_names.add(item["name"])
    _normalize_documents(private["sec_primary_documents"], stage=stage)
    _carry_in_by_form(private["carry_in_documents"], stage=stage)
    if len(private["market_responses"]) != len(YAHOO_SYMBOLS):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market quarantine is partial"
        )
    for expected_symbol, item in zip(
        YAHOO_SYMBOLS, private["market_responses"], strict=True
    ):
        expected_request = build_acquisition_plan(stage)["market"][
            "requests"
        ][YAHOO_SYMBOLS.index(expected_symbol)]
        if (
            type(item) is not dict
            or set(item) != {"symbol", "url", "body"}
            or item["symbol"] != expected_symbol
            or item["url"] != expected_request["url"]
            or type(item["body"]) is not bytes
            or not item["body"]
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Market response quarantine is invalid or reordered"
            )
    for expected_sequence, receipt in enumerate(
        private["request_receipts"], start=1
    ):
        _validate_receipt(receipt)
        if receipt["sequence_number"] != expected_sequence:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Acquisition receipts are duplicated or reordered"
            )
    receipt_targets = [
        ("sec", item["url"], item["body"])
        for item in private["sec_catalog_sources"]
    ]
    receipt_targets.extend(
        ("sec", item["official_url"], item["body"])
        for item in private["sec_primary_documents"]
    )
    receipt_targets.extend(
        ("market", item["url"], item["body"])
        for item in private["market_responses"]
    )
    if len(private["request_receipts"]) != len(receipt_targets):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition receipts omit or add an exact provider response"
        )
    for receipt, (channel, url, payload) in zip(
        private["request_receipts"], receipt_targets, strict=True
    ):
        if (
            receipt["channel"] != channel
            or receipt["requested_url"] != url
            or receipt["final_url"] != url
            or receipt["byte_count"] != len(payload)
            or receipt["body_sha256"] != _sha256_bytes(payload)
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Acquisition receipt does not reconcile to its exact response bytes"
            )
    expected_requests = _build_model_requests(
        private["sec_primary_documents"],
        carry_in_documents=private["carry_in_documents"],
        stage=stage,
    )
    if private["model_requests"] != expected_requests:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Blinded model requests do not replay from exact SEC bytes"
        )
    if (
        private["model_slice"].get("model_slice_sha256")
        != canonical_sha256(_model_slice_index(private["model_slice"]))
        or private["model_slice"]["model_requests"] != expected_requests
        or private["model_slice"]["universe_event_proofs"]
        != private["stage_slice"]["universe_event_proofs"]
        or
        private["stage_slice"].get("slice_sha256")
        != canonical_sha256(_stage_slice_index(private["stage_slice"]))
        or private["stage_slice"]["model_requests"] != expected_requests
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Canonical stage slice does not replay from exact SEC bytes"
        )


def _validate_sec_detached_replay(
    private: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, str]:
    evidence = private["sec_replay_evidence"]
    expected_keys = {
        "catalog_request_receipts",
        "catalog_artifact",
        "corpus_universe_manifest",
        "authenticated_stage_request_receipts",
        "stage_request_receipts",
        "stage_content_manifest",
        "stage_artifact",
    }
    if type(evidence) is not dict or set(evidence) != expected_keys:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC detached replay evidence schema changed"
        )
    sources = private["sec_catalog_sources"]
    catalog_receipts = evidence["catalog_request_receipts"]
    catalog_artifact = evidence["catalog_artifact"]
    universe = evidence["corpus_universe_manifest"]
    if (
        type(catalog_receipts) is not list
        or type(catalog_artifact) is not dict
        or type(universe) is not dict
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC catalogue replay evidence is not exact plain JSON"
        )
    try:
        catalog_replay = validate_detached_catalog_replay(
            source_payloads=[
                {"name": source["name"], "payload": source["body"]}
                for source in sources
            ],
            request_receipts=catalog_receipts,
            catalog_artifact=catalog_artifact,
            corpus_universe_manifest=universe,
            expected_source_payload_sha256s={
                source["name"]: _sha256_bytes(source["body"])
                for source in sources
            },
            expected_request_receipt_sha256s={
                source["name"]: receipt["request_receipt_sha256"]
                for source, receipt in zip(
                    sources, catalog_receipts, strict=True
                )
            },
            expected_request_receipts_sha256=catalog_artifact[
                "request_receipts_sha256"
            ],
            expected_catalog_artifact_sha256=catalog_artifact[
                "catalog_artifact_sha256"
            ],
            expected_corpus_universe_sha256=universe["universe_sha256"],
            expected_calendar_artifact_sha256=build_contract_manifest()[
                "data"
            ]["market"]["calendar"]["calendar_dates_sha256"],
            session_dates=list(EXPECTED_SESSIONS),
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC catalogue bytes failed detached exact replay"
        ) from None

    authorized_stage = _STAGE_SPECS[stage]["sec_stage"]
    selected = [
        record
        for record in universe["records"]
        if record["artifact_stage"] == authorized_stage
    ]
    selected.sort(
        key=lambda record: (
            record["availability_session"],
            record["accession_number"],
        )
    )
    documents = private["sec_primary_documents"]
    if len(selected) != len(documents):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC stage documents differ from the replayed universe"
        )
    for record, document in zip(selected, documents, strict=True):
        expected = {
            "accession_number": record["accession_number"],
            "form": record["form"],
            "availability_session": record["availability_session"],
            "acceptance_datetime": record["acceptance_datetime"],
            "official_url": _official_primary_url(record),
        }
        if any(document[key] != value for key, value in expected.items()):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC filing metadata or URL differs from detached replay"
            )

    authenticated_receipts = evidence[
        "authenticated_stage_request_receipts"
    ]
    stage_receipts = evidence["stage_request_receipts"]
    if (
        type(authenticated_receipts) is not list
        or type(stage_receipts) is not list
        or len(authenticated_receipts) != len(documents)
        or len(stage_receipts) != len(documents)
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC stage replay receipts are partial"
        )
    for original, replay in zip(
        authenticated_receipts, stage_receipts, strict=True
    ):
        if type(original) is not dict or type(replay) is not dict:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC stage replay receipt is not exact JSON"
            )
        original_body = {
            key: value
            for key, value in original.items()
            if key not in {"purpose", "request_receipt_sha256"}
        }
        replay_body = {
            key: value
            for key, value in replay.items()
            if key not in {"purpose", "request_receipt_sha256"}
        }
        if (
            original.get("purpose")
            != "authenticated_stage_access_primary_document"
            or replay.get("purpose")
            != f"{authorized_stage}_primary_document"
            or original_body != replay_body
            or original.get("request_receipt_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in original.items()
                    if key != "request_receipt_sha256"
                }
            )
            or replay.get("request_receipt_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in replay.items()
                    if key != "request_receipt_sha256"
                }
            )
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC replay receipt is not a purpose-only derivation of the "
                "authenticated exact fetch receipt"
            )

    content_manifest = evidence["stage_content_manifest"]
    stage_artifact = evidence["stage_artifact"]
    if type(content_manifest) is not dict or type(stage_artifact) is not dict:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC stage replay artifacts are not exact JSON"
        )
    try:
        stage_replay = validate_detached_stage_content_replay(
            authorized_stage=authorized_stage,
            document_payloads=[
                {
                    "accession_number": document["accession_number"],
                    "payload": document["body"],
                }
                for document in documents
            ],
            request_receipts=stage_receipts,
            content_manifest=content_manifest,
            stage_artifact=stage_artifact,
            corpus_universe_manifest=universe,
            expected_document_sha256s={
                document["accession_number"]: _sha256_bytes(
                    document["body"]
                )
                for document in documents
            },
            expected_normalized_text_sha256s={
                document["accession_number"]: hashlib.sha256(
                    normalize_filing_text(
                        document["body"].decode("latin-1")
                    ).text.encode("utf-8")
                ).hexdigest()
                for document in documents
            },
            expected_request_receipt_sha256s={
                document["accession_number"]: receipt[
                    "request_receipt_sha256"
                ]
                for document, receipt in zip(
                    documents, stage_receipts, strict=True
                )
            },
            expected_request_receipts_sha256=stage_artifact[
                "request_receipts_sha256"
            ],
            expected_content_manifest_sha256=content_manifest[
                "content_manifest_sha256"
            ],
            expected_stage_artifact_sha256=stage_artifact[
                "stage_artifact_sha256"
            ],
            expected_corpus_universe_sha256=universe["universe_sha256"],
            session_dates=list(EXPECTED_SESSIONS),
        )
    except Exception:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "SEC primary-document bytes failed detached exact replay"
        ) from None
    return {
        "catalog_replay_validation_sha256": catalog_replay[
            "replay_validation_sha256"
        ],
        "stage_replay_validation_sha256": stage_replay[
            "replay_validation_sha256"
        ],
    }


def _validate_one_bundle(
    bundle: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    predecessor: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if type(bundle) is not dict or set(bundle) != {
        "schema_version",
        "public_manifest",
        "private_index_sha256",
        "private_quarantine",
        "bundle_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition bundle schema changed"
        )
    if bundle["schema_version"] != ACQUISITION_BUNDLE_SCHEMA_VERSION:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition bundle version changed"
        )
    stage = plan["stage"]
    private = bundle["private_quarantine"]
    _validate_private_schema(private, stage)
    replay_receipts = _validate_sec_detached_replay(private, stage=stage)
    universe_hash = private["sec_replay_evidence"][
        "corpus_universe_manifest"
    ]["universe_sha256"]
    for request, proof in zip(
        private["model_requests"],
        private["stage_slice"]["universe_event_proofs"],
        strict=True,
    ):
        try:
            validated_proof = validate_universe_event_proof(
                proof,
                expected_universe_event_proof_sha256=proof[
                    "universe_event_proof_sha256"
                ],
            )
        except Exception:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Stage-slice universe event proof is not canonical"
            ) from None
        if (
            validated_proof["corpus_universe_sha256"] != universe_hash
            or validated_proof["current_record"]["accession_number"]
            != request["accession_number"]
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Stage-slice universe event proof is bound to another corpus event"
            )
    raw_market = _bundle_raw_market(bundle)
    commitments, current_rows = _market_commitments(raw_market, plan)
    expected_slice_rows = {
        symbol: [
            copy.deepcopy(row)
            for row in current_rows[symbol]
            if row["session"] <= plan["market"]["last_value_session"]
        ]
        for symbol in YAHOO_SYMBOLS
    }
    if private["stage_slice"]["market_rows"] != expected_slice_rows:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Canonical stage slice market rows differ from exact provider bytes"
        )

    expected_predecessor = _STAGE_SPECS[stage]["predecessor"]
    if expected_predecessor is None:
        if predecessor is not None:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Development bundle cannot bind a predecessor"
            )
        predecessor_hash = None
        continuity = True
    else:
        if predecessor is None:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Later-stage bundle lacks its exact predecessor"
            )
        predecessor_hash = predecessor["bundle_sha256"]
        predecessor_plan = build_acquisition_plan(expected_predecessor)
        predecessor_rows = _market_commitments(
            _bundle_raw_market(predecessor), predecessor_plan
        )[1]
        predecessor_cutoff = predecessor_plan["market"][
            "last_value_session"
        ]
        for symbol in YAHOO_SYMBOLS:
            expected_prefix = [
                row
                for row in predecessor_rows[symbol]
                if row["session"] <= predecessor_cutoff
            ]
            observed_prefix = [
                row
                for row in current_rows[symbol]
                if row["session"] <= predecessor_cutoff
            ]
            if observed_prefix != expected_prefix:
                raise SecGemmaOnlineRiskOverlayAcquisitionError(
                    "Market history changed inside the sealed predecessor prefix"
                )
        expected_carry = _last_documents_by_form(predecessor)
        if private["carry_in_documents"] != expected_carry:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "SEC carry-in does not equal the predecessor's latest same-form bytes"
            )
        continuity = True

    identity_hashes = {
        receipt["identity_sha256"]
        for receipt in private["request_receipts"]
        if receipt["channel"] == "sec"
    }
    if (
        len(identity_hashes) != 1
        or None in identity_hashes
        or any(
            receipt["identity_sha256"] is not None
            for receipt in private["request_receipts"]
            if receipt["channel"] == "market"
        )
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Request receipts do not bind one private SEC identity"
        )
    identity_hash = next(iter(identity_hashes))
    expected_manifest = _public_manifest(
        plan=plan,
        identity_sha256=identity_hash,
        predecessor_bundle_sha256=predecessor_hash,
        private=private,
        market_commitments=commitments,
        prefix_continuity_verified=continuity,
    )
    index_hash = canonical_sha256(_private_index(private))
    body = {
        "schema_version": ACQUISITION_BUNDLE_SCHEMA_VERSION,
        "public_manifest": expected_manifest,
        "private_index_sha256": index_hash,
    }
    if (
        bundle["public_manifest"] != expected_manifest
        or bundle["private_index_sha256"] != index_hash
        or bundle["bundle_sha256"] != canonical_sha256(body)
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition bundle hashes or public summary changed"
        )
    return {
        "stage": stage,
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_sha256": expected_manifest["manifest_sha256"],
        "private_index_sha256": index_hash,
        **replay_receipts,
    }


_VALIDATION_CHECK_NAMES: Final[tuple[str, ...]] = (
    "exact_raw_bytes_replayed_sha256",
    "request_receipts_reconciled_sha256",
    "stage_and_attempt_scope_bound_sha256",
    "private_identity_digest_only_sha256",
    "market_prefix_continuity_replayed_sha256",
    "blinded_model_requests_replayed_sha256",
    "request_byte_retry_redirect_caps_reconciled_sha256",
)


def _validation_checks(
    *,
    bundle: Mapping[str, Any],
    plan: Mapping[str, Any],
    predecessor_bundles: Sequence[Mapping[str, Any]],
    replay_result: Mapping[str, Any],
) -> dict[str, str]:
    private = bundle["private_quarantine"]
    manifest = bundle["public_manifest"]
    index = _private_index(private)
    receipts = private["request_receipts"]
    sec_receipts = [
        receipt for receipt in receipts if receipt["channel"] == "sec"
    ]
    market_receipts = [
        receipt for receipt in receipts if receipt["channel"] == "market"
    ]
    return {
        "exact_raw_bytes_replayed_sha256": canonical_sha256(
            {
                "private_index_sha256": canonical_sha256(index),
                "sec_catalog_source_hashes": [
                    item["sha256"] for item in index["sec_catalog_sources"]
                ],
                "sec_primary_document_hashes": [
                    item["sha256"]
                    for item in index["sec_primary_documents"]
                ],
                "market_response_hashes": [
                    item["sha256"] for item in index["market_responses"]
                ],
                "sec_replay_evidence_sha256": index[
                    "sec_replay_evidence_sha256"
                ],
                "catalog_replay_validation_sha256": replay_result[
                    "catalog_replay_validation_sha256"
                ],
                "stage_replay_validation_sha256": replay_result[
                    "stage_replay_validation_sha256"
                ],
            }
        ),
        "request_receipts_reconciled_sha256": canonical_sha256(
            {
                "request_receipts_sha256": index[
                    "request_receipts_sha256"
                ],
                "sequence_numbers": [
                    item["sequence_number"] for item in receipts
                ],
                "receipt_sha256s": [
                    item["receipt_sha256"] for item in receipts
                ],
                "body_sha256s": [
                    item["body_sha256"] for item in receipts
                ],
            }
        ),
        "stage_and_attempt_scope_bound_sha256": canonical_sha256(
            {
                "stage": plan["stage"],
                "attempt_id": plan["attempt_id"],
                "attempt_kind": plan["attempt_kind"],
                "acquisition_plan_sha256": plan[
                    "acquisition_plan_sha256"
                ],
                "sec_documents": [
                    {
                        "accession_number": item["accession_number"],
                        "form": item["form"],
                        "availability_session": item[
                            "availability_session"
                        ],
                    }
                    for item in index["sec_primary_documents"]
                ],
            }
        ),
        "private_identity_digest_only_sha256": canonical_sha256(
            {
                "sec_identity_sha256": manifest["sec_identity_sha256"],
                "sec_receipt_identity_sha256s": [
                    item["identity_sha256"] for item in sec_receipts
                ],
                "market_identity_fields": [
                    item["identity_sha256"] for item in market_receipts
                ],
                "raw_identity_returned": False,
            }
        ),
        "market_prefix_continuity_replayed_sha256": canonical_sha256(
            {
                "market_prefix_commitment_sha256s": manifest[
                    "market_prefix_commitment_sha256s"
                ],
                "predecessor_chain_bundle_sha256s": [
                    item["bundle_sha256"] for item in predecessor_bundles
                ],
                "predecessor_bundle_sha256": manifest[
                    "predecessor_bundle_sha256"
                ],
            }
        ),
        "blinded_model_requests_replayed_sha256": canonical_sha256(
            {
                "model_requests": index["model_requests"],
                "model_request_count": manifest["model_request_count"],
                "model_output_returned": False,
                "semantic_extractions_returned": False,
            }
        ),
        "request_byte_retry_redirect_caps_reconciled_sha256": canonical_sha256(
            {
                "sec_request_count": len(sec_receipts),
                "market_request_count": len(market_receipts),
                "sec_bytes": sum(
                    item["byte_count"] for item in sec_receipts
                ),
                "market_bytes": sum(
                    item["byte_count"] for item in market_receipts
                ),
                "all_network_requests": [
                    item["network_requests"] for item in receipts
                ],
                "all_retries": [item["retries"] for item in receipts],
                "all_redirects": [
                    item["redirects"] for item in receipts
                ],
                "sec_request_cap": plan["sec"]["request_cap"],
                "sec_byte_cap": plan["sec"]["byte_cap"],
                "market_request_cap": plan["market"]["request_count"],
                "market_batch_byte_cap": plan["market"]["batch_byte_cap"],
            }
        ),
    }


def _validate_verified_report_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "verifier_id",
        "verdict",
        "stage",
        "attempt_id",
        "attempt_kind",
        "acquisition_plan_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "private_index_sha256",
        "predecessor_chain_bundle_sha256s",
        "checks",
        "check_set_sha256",
        "validation_sha256",
    }:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Verified acquisition report schema changed"
        )
    checks = value["checks"]
    stage = value.get("stage")
    digest_fields = (
        "acquisition_plan_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "private_index_sha256",
        "check_set_sha256",
        "validation_sha256",
    )
    predecessor_hashes = value.get(
        "predecessor_chain_bundle_sha256s"
    )
    if (
        value["schema_version"] != ACQUISITION_VALIDATION_SCHEMA_VERSION
        or value["verifier_id"] != ACQUISITION_VALIDATION_VERIFIER_ID
        or value["verdict"] != "pass"
        or type(stage) is not str
        or stage not in STAGES
        or value["attempt_id"]
        != _STAGE_SPECS[stage]["attempt_id"]
        or value["attempt_kind"]
        != _STAGE_SPECS[stage]["attempt_kind"]
        or value["acquisition_plan_sha256"]
        != build_acquisition_plan(stage)["acquisition_plan_sha256"]
        or any(
            type(value[name]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value[name]) is None
            for name in digest_fields
        )
        or type(predecessor_hashes) is not list
        or len(predecessor_hashes) != STAGES.index(stage)
        or any(
            type(item) is not str
            or re.fullmatch(r"[0-9a-f]{64}", item) is None
            for item in predecessor_hashes
        )
        or type(checks) is not dict
        or set(checks) != set(_VALIDATION_CHECK_NAMES)
        or any(
            type(checks[name]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", checks[name]) is None
            for name in _VALIDATION_CHECK_NAMES
        )
        or value["check_set_sha256"] != canonical_sha256(checks)
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Verified acquisition report checks are not canonical"
        )
    body = {
        key: value[key] for key in value if key != "validation_sha256"
    }
    if value["validation_sha256"] != canonical_sha256(body):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Verified acquisition report self-hash changed"
        )
    return copy.deepcopy(value)


def is_verified_acquisition_report(value: Any) -> bool:
    """Return true only for an intact report issued by this validator."""

    if (
        type(value) is not VerifiedAcquisitionReport
        or getattr(value, "_sentinel", None) is not _VERIFIED_REPORT_SENTINEL
    ):
        return False
    try:
        _validate_verified_report_payload(value.as_dict())
    except SecGemmaOnlineRiskOverlayAcquisitionError:
        return False
    return True


def _canonical_recovery_elapsed(value: Any) -> str:
    if type(value) is not str:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered market elapsed evidence is noncanonical"
        )
    try:
        elapsed = float.fromhex(value)
    except ValueError:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered market elapsed evidence is noncanonical"
        ) from None
    if (
        not math.isfinite(elapsed)
        or elapsed < 0.0
        or elapsed > float(MAX_MARKET_SECONDS)
        or elapsed.hex() != value
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered market elapsed evidence exceeds its frozen cap"
        )
    return value


def _validate_recovery_public_material(
    value: Mapping[str, Any],
    *,
    stage: str,
    predecessor_bundle_sha256s: tuple[str, ...],
    handle: VaultHandle,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(value) is not dict or set(value) != {
        "verified_acquisition_report",
        "public_summary",
        "request_accounting",
    }:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered acquisition evidence schema changed"
        )
    report = _validate_verified_report_payload(
        value["verified_acquisition_report"]
    )
    plan = build_acquisition_plan(stage)
    expected_predecessor = (
        None
        if not predecessor_bundle_sha256s
        else predecessor_bundle_sha256s[-1]
    )
    if (
        report["stage"] != stage
        or report["attempt_id"] != plan["attempt_id"]
        or report["attempt_kind"] != plan["attempt_kind"]
        or report["acquisition_plan_sha256"]
        != plan["acquisition_plan_sha256"]
        or report["predecessor_chain_bundle_sha256s"]
        != list(predecessor_bundle_sha256s)
        or report["bundle_sha256"] != handle.bundle_sha256
        or report["manifest_sha256"] != handle.manifest_sha256
        or report["private_index_sha256"]
        != handle.private_index_sha256
        or handle.stage != stage
        or handle.attempt_id != plan["attempt_id"]
        or handle.production_authority is not True
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered acquisition report crossed its production vault handle"
        )

    summary_value = value["public_summary"]
    summary_fields = {
        "schema_version",
        "stage",
        "attempt_id",
        "attempt_kind",
        "acquisition_plan_sha256",
        "predecessor_bundle_sha256",
        "bundle_sha256",
        "manifest_sha256",
        "private_index_sha256",
        "sec_catalog_source_count",
        "sec_primary_document_count",
        "market_response_count",
        "model_request_count",
        "model_slice_sha256",
        "stage_slice_sha256",
        "market_elapsed_seconds_hex",
        "total_raw_byte_count",
        "complete_batch",
        "quarantine_only",
        "production_authority",
        "public_summary_sha256",
    }
    if type(summary_value) is not dict or set(summary_value) != summary_fields:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered acquisition public summary schema changed"
        )
    summary = copy.deepcopy(summary_value)
    summary_body = {
        key: child
        for key, child in summary.items()
        if key != "public_summary_sha256"
    }
    count_fields = (
        "sec_catalog_source_count",
        "sec_primary_document_count",
        "market_response_count",
        "model_request_count",
        "total_raw_byte_count",
    )
    spec = _STAGE_SPECS[stage]
    if (
        summary["public_summary_sha256"]
        != canonical_sha256(summary_body)
        or summary["schema_version"]
        != ACQUISITION_PUBLIC_SUMMARY_SCHEMA_VERSION
        or summary["stage"] != stage
        or summary["attempt_id"] != plan["attempt_id"]
        or summary["attempt_kind"] != plan["attempt_kind"]
        or summary["acquisition_plan_sha256"]
        != report["acquisition_plan_sha256"]
        or summary["predecessor_bundle_sha256"]
        != expected_predecessor
        or summary["bundle_sha256"] != report["bundle_sha256"]
        or summary["manifest_sha256"] != report["manifest_sha256"]
        or summary["private_index_sha256"]
        != report["private_index_sha256"]
        or any(
            type(summary[field]) is not int or summary[field] < 0
            for field in count_fields
        )
        or summary["sec_catalog_source_count"] < 1
        or summary["sec_primary_document_count"]
        != summary["model_request_count"]
        or not (
            spec["minimum_filings"]
            <= summary["model_request_count"]
            <= spec["maximum_model_requests"]
        )
        or summary["market_response_count"]
        != MAX_MARKET_REQUESTS_PER_STAGE
        or summary["complete_batch"] is not True
        or summary["quarantine_only"] is not True
        or summary["production_authority"] is not True
        or type(summary["model_slice_sha256"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}",
            summary["model_slice_sha256"],
        )
        is None
        or type(summary["stage_slice_sha256"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}",
            summary["stage_slice_sha256"],
        )
        is None
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered acquisition public summary is incomplete or crossed"
        )
    elapsed_hex = _canonical_recovery_elapsed(
        summary["market_elapsed_seconds_hex"]
    )

    accounting_value = value["request_accounting"]
    accounting_fields = {
        "schema_version",
        "stage",
        "attempt_id",
        "sec_request_count",
        "market_request_count",
        "sec_bytes",
        "market_bytes",
        "network_request_count",
        "retry_count",
        "redirect_count",
        "market_elapsed_seconds_hex",
        "accounting_sha256",
    }
    if (
        type(accounting_value) is not dict
        or set(accounting_value) != accounting_fields
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered acquisition accounting schema changed"
        )
    accounting = copy.deepcopy(accounting_value)
    accounting_body = {
        key: child
        for key, child in accounting.items()
        if key != "accounting_sha256"
    }
    integer_fields = (
        "sec_request_count",
        "market_request_count",
        "sec_bytes",
        "market_bytes",
        "network_request_count",
        "retry_count",
        "redirect_count",
    )
    if (
        accounting["accounting_sha256"]
        != canonical_sha256(accounting_body)
        or accounting["schema_version"]
        != ACQUISITION_REQUEST_ACCOUNTING_SCHEMA_VERSION
        or accounting["stage"] != stage
        or accounting["attempt_id"] != plan["attempt_id"]
        or any(
            type(accounting[field]) is not int
            or accounting[field] < 0
            for field in integer_fields
        )
        or not 1
        <= accounting["sec_request_count"]
        <= MAX_SEC_REQUESTS
        or accounting["sec_request_count"]
        != summary["sec_catalog_source_count"]
        + summary["sec_primary_document_count"]
        or accounting["market_request_count"]
        != MAX_MARKET_REQUESTS_PER_STAGE
        or accounting["sec_bytes"] > MAX_SEC_BYTES
        or accounting["market_bytes"] > MAX_MARKET_BATCH_BYTES
        or accounting["network_request_count"]
        != accounting["sec_request_count"]
        + accounting["market_request_count"]
        or accounting["retry_count"] != 0
        or accounting["redirect_count"] != 0
        or accounting["market_elapsed_seconds_hex"] != elapsed_hex
        or summary["total_raw_byte_count"]
        != accounting["sec_bytes"] + accounting["market_bytes"]
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Recovered acquisition accounting is incomplete or crossed"
        )
    _canonical_recovery_elapsed(
        accounting["market_elapsed_seconds_hex"]
    )
    return report, summary, accounting


def _rehydrate_one_production_acquisition(
    *,
    vault: ProductionAcquisitionVault,
    store: Any,
    stage: str,
    public_material: Mapping[str, Any],
    predecessor_executions: tuple[AcquisitionExecutionResult, ...],
) -> AcquisitionExecutionResult:
    from agent_benchmark.sec_gemma_online_risk_overlay_store import (
        SecGemmaOnlineRiskOverlayStore,
    )

    if (
        type(vault) is not ProductionAcquisitionVault
        or type(store) is not SecGemmaOnlineRiskOverlayStore
        or stage not in STAGES
        or type(predecessor_executions) is not tuple
        or len(predecessor_executions) != STAGES.index(stage)
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Production acquisition recovery received foreign authorities"
        )
    predecessor_handles: list[VaultHandle] = []
    predecessor_bundles: list[str] = []
    for expected_stage, execution in zip(
        STAGES[: STAGES.index(stage)],
        predecessor_executions,
        strict=True,
    ):
        if (
            type(execution) is not AcquisitionExecutionResult
            or execution.production_authority is not True
            or getattr(execution, "_vault", None) is not vault
            or execution.public_summary().get("stage") != expected_stage
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Production acquisition recovery predecessor is foreign"
            )
        predecessor_handles.append(execution.vault_handle)
        predecessor_bundles.append(
            execution.vault_handle.bundle_sha256
        )
    plan = build_acquisition_plan(stage)
    try:
        handle = _load_current_production_handle_for_recovery(
            vault,
            store=store,
            stage=stage,
            attempt_id=plan["attempt_id"],
            predecessor_handles=tuple(predecessor_handles),
        )
    except SecGemmaOnlineRiskOverlayVaultError:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Production acquisition recovery vault evidence is corrupt"
        ) from None
    if handle is None:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Production acquisition recovery lacks its sealed quarantine"
        )
    report_payload, summary, accounting = (
        _validate_recovery_public_material(
            public_material,
            stage=stage,
            predecessor_bundle_sha256s=tuple(predecessor_bundles),
            handle=handle,
        )
    )
    verified_report = VerifiedAcquisitionReport(
        report_payload,
        vault=vault,
        handle=handle,
        _sentinel=_VERIFIED_REPORT_SENTINEL,
    )
    return AcquisitionExecutionResult(
        vault=vault,
        handle=handle,
        verified_report=verified_report,
        public_summary=summary,
        request_accounting=accounting,
        model_slice_sha256=summary["model_slice_sha256"],
        stage_slice_sha256=summary["stage_slice_sha256"],
        _sentinel=_EXECUTION_RESULT_SENTINEL,
    )


def rehydrate_contiguous_production_acquisitions(
    *,
    vault: ProductionAcquisitionVault,
    store: Any,
) -> tuple[AcquisitionExecutionResult, ...]:
    """Rebuild only terminal-passed opaque acquisitions after a restart."""

    from agent_benchmark.sec_gemma_online_risk_overlay_store import (
        SecGemmaOnlineRiskOverlayStore,
        SecGemmaOnlineRiskOverlayStoreError,
    )

    if (
        type(vault) is not ProductionAcquisitionVault
        or vault.production_authority is not True
        or type(store) is not SecGemmaOnlineRiskOverlayStore
        or getattr(store, "_store_instance_id", None)
        != getattr(vault, "_bound_store_instance_id", None)
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Production acquisition recovery requires the exact reopened store"
        )
    materials: list[dict[str, Any] | None] = []
    missing = False
    for stage in STAGES:
        attempt_id = _STAGE_SPECS[stage]["attempt_id"]
        try:
            material = (
                SecGemmaOnlineRiskOverlayStore
                .terminal_acquisition_phase_evidence(
                    store,
                    attempt_id,
                )
            )
        except SecGemmaOnlineRiskOverlayStoreError as exc:
            if str(exc) != (
                "Acquisition recovery attempt did not terminal-pass"
            ):
                raise SecGemmaOnlineRiskOverlayAcquisitionError(
                    "Durable acquisition recovery evidence failed validation"
                ) from None
            material = None
        if material is None:
            missing = True
        elif missing:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Durable acquisition recovery stages are gapped"
            )
        materials.append(material)
    durable_stages = tuple(
        stage
        for stage, material in zip(STAGES, materials, strict=True)
        if material is not None
    )
    try:
        sealed_stages = _production_recovery_sealed_stages(
            vault,
            store=store,
        )
    except SecGemmaOnlineRiskOverlayVaultError:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Production acquisition recovery vault index is corrupt"
        ) from None
    if sealed_stages != durable_stages:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Durable acquisition passes and sealed vault stages differ"
        )
    recovered: list[AcquisitionExecutionResult] = []
    for stage, material in zip(STAGES, materials, strict=True):
        if material is None:
            break
        execution = _rehydrate_one_production_acquisition(
            vault=vault,
            store=store,
            stage=stage,
            public_material=material,
            predecessor_executions=tuple(recovered),
        )
        recovered.append(execution)
    return tuple(recovered)


def validate_acquisition_bundle(
    handle: VaultHandle,
    *,
    vault: ProductionAcquisitionVault | TestAcquisitionVault,
    store: Any,
    capability: EffectCapability,
    plan: Mapping[str, Any],
    predecessor_handles: Sequence[VaultHandle] = (),
) -> VerifiedAcquisitionReport:
    """Replay current sealed bytes internally without exposing a raw bundle."""

    fixed_plan = validate_acquisition_plan(plan)
    if type(handle) is not VaultHandle:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition validation requires one opaque vault handle"
        )
    if type(predecessor_handles) not in {tuple, list}:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Predecessor handles must be one exact ordered sequence"
        )
    expected_prior_stages = list(STAGES[: STAGES.index(fixed_plan["stage"])])
    if len(predecessor_handles) != len(expected_prior_stages):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition predecessor chain is incomplete or excessive"
        )
    try:
        bundle = _read_quarantine_for_replay(
            vault,
            handle,
            store=store,
            capability=capability,
        )
        predecessor_bundles = [
            _read_quarantine_for_replay(
                vault,
                predecessor,
                store=store,
                capability=capability,
            )
            for predecessor in predecessor_handles
        ]
    except SecGemmaOnlineRiskOverlayVaultError:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition vault replay is unauthorized, stale, or corrupt"
        ) from None
    prior: Mapping[str, Any] | None = None
    for expected_stage, predecessor_handle, predecessor in zip(
        expected_prior_stages,
        predecessor_handles,
        predecessor_bundles,
        strict=True,
    ):
        if predecessor_handle.stage != expected_stage:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Acquisition predecessor handles are reordered"
            )
        prior_plan = build_acquisition_plan(expected_stage)
        _validate_one_bundle(
            predecessor, plan=prior_plan, predecessor=prior
        )
        prior = predecessor
    if (
        handle.stage != fixed_plan["stage"]
        or handle.attempt_id != fixed_plan["attempt_id"]
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition vault handle names the wrong stage or attempt"
        )
    result = _validate_one_bundle(bundle, plan=fixed_plan, predecessor=prior)
    if (
        result["bundle_sha256"] != handle.bundle_sha256
        or result["manifest_sha256"] != handle.manifest_sha256
        or result["private_index_sha256"]
        != handle.private_index_sha256
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition replay hashes differ from the durable vault seal"
        )
    checks = _validation_checks(
        bundle=bundle,
        plan=fixed_plan,
        predecessor_bundles=predecessor_bundles,
        replay_result=result,
    )
    body = {
        "schema_version": ACQUISITION_VALIDATION_SCHEMA_VERSION,
        "verifier_id": ACQUISITION_VALIDATION_VERIFIER_ID,
        "verdict": "pass",
        "stage": result["stage"],
        "attempt_id": fixed_plan["attempt_id"],
        "attempt_kind": fixed_plan["attempt_kind"],
        "acquisition_plan_sha256": fixed_plan["acquisition_plan_sha256"],
        "bundle_sha256": result["bundle_sha256"],
        "manifest_sha256": result["manifest_sha256"],
        "private_index_sha256": result["private_index_sha256"],
        "predecessor_chain_bundle_sha256s": [
            item.bundle_sha256 for item in predecessor_handles
        ],
        "checks": checks,
        "check_set_sha256": canonical_sha256(checks),
    }
    payload = {**body, "validation_sha256": canonical_sha256(body)}
    _validate_verified_report_payload(payload)
    try:
        _assert_vault_handle_current(vault, handle)
    except SecGemmaOnlineRiskOverlayVaultError:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition vault changed during detached replay"
        ) from None
    return VerifiedAcquisitionReport(
        payload,
        vault=vault,
        handle=handle,
        _sentinel=_VERIFIED_REPORT_SENTINEL,
    )


def _authorize_before_effects(
    *,
    store: Any,
    capability: EffectCapability,
    plan: Mapping[str, Any],
) -> None:
    if (
        not isinstance(capability, EffectCapability)
        or capability.attempt_id != plan["attempt_id"]
        or "official_sec_network" not in capability.allowed_effects
        or "market_network" not in capability.allowed_effects
        or not callable(getattr(store, "authorize_effect", None))
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition capability is absent, wrong-stage, or not store-bound"
        )
    try:
        store.authorize_effect(capability, "official_sec_network")
        store.authorize_effect(capability, "market_network")
    except Exception:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition capability is stale, foreign, unconsumed, or unauthorized"
        ) from None


def _validate_private_identity(value: Any) -> tuple[str, bytes]:
    if type(value) is not str:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Private SEC identity is missing or invalid"
        )
    try:
        audit = validate_sec_user_agent(value)
        encoded = value.encode("utf-8")
    except Exception:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Private SEC identity is missing or invalid"
        ) from None
    return audit.sha256, encoded


def _acquire_market(
    *,
    transport: MarketTransport,
    plan: Mapping[str, Any],
    clock: Callable[[], float],
    combined_deadline: float,
    security_state: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if security_state is None:
        _market_security_state(transport)
    start = _checked_now(clock)
    market_deadline = min(
        combined_deadline, start + float(plan["market"]["deadline_seconds"])
    )
    if start >= market_deadline:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market acquisition deadline was reached"
        )
    if type(transport) is _OwnedYahooMarketTransport:
        transport._begin_batch(deadline_monotonic=market_deadline)
    responses: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    total_bytes = 0
    for request in plan["market"]["requests"]:
        if _checked_now(clock) >= market_deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Market acquisition deadline was reached"
            )
        raw = transport.fetch(request["url"])
        if type(raw) is not dict or set(raw) != set(_MARKET_RESPONSE_KEYS):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Market transport response schema changed"
            )
        body = raw["body"]
        if (
            raw["request_url"] != request["url"]
            or raw["final_url"] != request["url"]
            or type(raw["status_code"]) is not int
            or raw["status_code"] != 200
            or raw["content_type"] != "application/json"
            or raw["charset"] not in {None, "utf-8", "UTF-8"}
            or raw["content_encoding"] not in {None, "identity"}
            or type(body) is not bytes
            or not body
            or len(body) > MAX_MARKET_RESPONSE_BYTES
            or (
                raw["declared_content_length"] is not None
                and type(raw["declared_content_length"]) is not int
            )
            or raw["declared_content_length"] not in {None, len(body)}
            or type(raw["network_requests"]) is not int
            or raw["network_requests"] != 1
            or type(raw["retries"]) is not int
            or raw["retries"] != 0
            or type(raw["redirects"]) is not int
            or raw["redirects"] != 0
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Market response redirected, retried, exceeded a cap, or changed"
            )
        total_bytes += len(body)
        if total_bytes > MAX_MARKET_BATCH_BYTES:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Market batch exceeded the byte ceiling"
            )
        if _checked_now(clock) > market_deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Market acquisition deadline was exceeded"
            )
        responses.append(
            {
                "symbol": request["symbol"],
                "url": request["url"],
                "body": body,
            }
        )
        receipts.append(
            _receipt(
                sequence_number=0,
                channel="market",
                purpose=f"{plan['stage']}_{request['symbol']}_chart_v8",
                requested_url=request["url"],
                final_url=request["url"],
                status_code=200,
                content_type="application/json",
                byte_count=len(body),
                body_sha256=_sha256_bytes(body),
                network_requests=1,
                retries=0,
                redirects=0,
                identity_sha256=None,
            )
        )
    if len(responses) != MAX_MARKET_REQUESTS_PER_STAGE:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market batch is partial"
        )
    elapsed = _checked_now(clock) - start
    if (
        not math.isfinite(elapsed)
        or elapsed < 0.0
        or elapsed > float(MAX_MARKET_SECONDS)
        or elapsed > float(plan["market"]["deadline_seconds"])
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market acquisition elapsed time exceeded its exact cap"
        )
    return responses, receipts, elapsed.hex()


def _execution_public_summary(
    bundle: Mapping[str, Any],
    *,
    production_authority: bool,
    market_elapsed_seconds_hex: str,
) -> dict[str, Any]:
    manifest = bundle["public_manifest"]
    private = bundle["private_quarantine"]
    try:
        market_elapsed = float.fromhex(market_elapsed_seconds_hex)
    except (TypeError, ValueError):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market elapsed evidence is noncanonical"
        ) from None
    if (
        not math.isfinite(market_elapsed)
        or market_elapsed < 0.0
        or market_elapsed > float(MAX_MARKET_SECONDS)
        or market_elapsed.hex() != market_elapsed_seconds_hex
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Market elapsed evidence is outside the frozen cap"
        )
    body = {
        "schema_version": ACQUISITION_PUBLIC_SUMMARY_SCHEMA_VERSION,
        "stage": manifest["stage"],
        "attempt_id": manifest["attempt_id"],
        "attempt_kind": manifest["attempt_kind"],
        "acquisition_plan_sha256": manifest[
            "acquisition_plan_sha256"
        ],
        "predecessor_bundle_sha256": manifest[
            "predecessor_bundle_sha256"
        ],
        "bundle_sha256": bundle["bundle_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "private_index_sha256": manifest["private_index_sha256"],
        "sec_catalog_source_count": manifest[
            "sec_catalog_source_count"
        ],
        "sec_primary_document_count": manifest[
            "sec_primary_document_count"
        ],
        "market_response_count": manifest["market_response_count"],
        "model_request_count": manifest["model_request_count"],
        "model_slice_sha256": private["model_slice"][
            "model_slice_sha256"
        ],
        "stage_slice_sha256": private["stage_slice"]["slice_sha256"],
        "market_elapsed_seconds_hex": market_elapsed_seconds_hex,
        "total_raw_byte_count": manifest["total_raw_byte_count"],
        "complete_batch": manifest["complete_batch"],
        "quarantine_only": manifest["quarantine_only"],
        "production_authority": production_authority,
    }
    return {**body, "public_summary_sha256": canonical_sha256(body)}


def _execution_request_accounting(
    bundle: Mapping[str, Any],
    *,
    market_elapsed_seconds_hex: str,
) -> dict[str, Any]:
    manifest = bundle["public_manifest"]
    receipts = bundle["private_quarantine"]["request_receipts"]
    sec_receipts = [
        receipt for receipt in receipts if receipt["channel"] == "sec"
    ]
    market_receipts = [
        receipt for receipt in receipts if receipt["channel"] == "market"
    ]
    body = {
        "schema_version": ACQUISITION_REQUEST_ACCOUNTING_SCHEMA_VERSION,
        "stage": manifest["stage"],
        "attempt_id": manifest["attempt_id"],
        "sec_request_count": len(sec_receipts),
        "market_request_count": len(market_receipts),
        "sec_bytes": sum(item["byte_count"] for item in sec_receipts),
        "market_bytes": sum(
            item["byte_count"] for item in market_receipts
        ),
        "network_request_count": sum(
            item["network_requests"] for item in receipts
        ),
        "retry_count": sum(item["retries"] for item in receipts),
        "redirect_count": sum(item["redirects"] for item in receipts),
        "market_elapsed_seconds_hex": market_elapsed_seconds_hex,
    }
    return {**body, "accounting_sha256": canonical_sha256(body)}


def acquire_stage_to_quarantine(
    *,
    plan: Mapping[str, Any],
    store: Any,
    capability: EffectCapability,
    vault: ProductionAcquisitionVault | TestAcquisitionVault,
    sec_user_agent: str,
    sec_transport: Any,
    market_transport: MarketTransport,
    predecessor_executions: Sequence[AcquisitionExecutionResult] = (),
    allow_test_authorities: bool = False,
    deadline_monotonic: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> AcquisitionExecutionResult:
    """Run one complete acquisition after the store has consumed its attempt.

    Production accepts only the exact reviewed SEC transport, the owned Yahoo
    transport, and the production vault.  Explicit tests use the visibly
    non-production vault and can inject deterministic fakes.  Raw provider
    bytes are sealed before this function returns and never become an object
    attribute or public mapping.
    """

    fixed_plan = validate_acquisition_plan(plan)
    if type(allow_test_authorities) is not bool:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Test-authority mode must be an exact boolean"
        )
    acquisition_started = _checked_now(clock)
    if deadline_monotonic is None:
        if not allow_test_authorities:
            if (
                type(vault) is not ProductionAcquisitionVault
                or type(sec_transport) is not SecAuditTransport
                or type(market_transport)
                is not _OwnedYahooMarketTransport
            ):
                raise SecGemmaOnlineRiskOverlayAcquisitionError(
                    "Production acquisition requires exact reviewed authorities"
                )
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Production acquisition requires its runner parent deadline"
            )
        parent_deadline = acquisition_started + float(MAX_SEC_SECONDS)
    else:
        try:
            parent_deadline = float(deadline_monotonic)
        except Exception:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Acquisition parent deadline is invalid"
            ) from None
        if (
            not math.isfinite(parent_deadline)
            or not 0.0
            < parent_deadline - acquisition_started
            <= float(MAX_SEC_SECONDS)
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Acquisition parent deadline is outside the frozen combined cap"
            )
    combined_deadline = min(
        parent_deadline,
        acquisition_started + float(MAX_SEC_SECONDS),
    )
    if allow_test_authorities:
        if type(vault) is not TestAcquisitionVault:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Test acquisition requires the explicit non-production vault"
            )
    elif (
        type(vault) is not ProductionAcquisitionVault
        or type(sec_transport) is not SecAuditTransport
        or type(getattr(sec_transport, "_session", None))
        is not requests.Session
        or getattr(sec_transport, "_clock", None) is not time.monotonic
        or getattr(sec_transport, "_sleep", None) is not time.sleep
        or getattr(getattr(sec_transport, "_budget", None), "clock", None)
        is not time.monotonic
        or type(market_transport) is not _OwnedYahooMarketTransport
        or getattr(sec_transport, "_parent_deadline_monotonic", None)
        != parent_deadline
        or getattr(market_transport, "_parent_deadline", None)
        != parent_deadline
        or clock is not time.monotonic
        or sleeper is not time.sleep
    ):
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Production acquisition requires exact reviewed authorities"
        )
    identity_sha256, raw_identity_bytes = _validate_private_identity(
        sec_user_agent
    )
    if type(predecessor_executions) not in {tuple, list}:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Predecessor executions must be one exact ordered sequence"
        )
    expected_prior_count = STAGES.index(fixed_plan["stage"])
    if len(predecessor_executions) != expected_prior_count:
        raise SecGemmaOnlineRiskOverlayAcquisitionError(
            "Acquisition predecessor chain is incomplete or excessive"
        )
    predecessor_handles: list[VaultHandle] = []
    for expected_stage, execution in zip(
        STAGES[:expected_prior_count],
        predecessor_executions,
        strict=True,
    ):
        if (
            type(execution) is not AcquisitionExecutionResult
            or execution.vault_handle.stage != expected_stage
            or execution.production_authority
            is not vault.production_authority
            or getattr(execution, "_vault", None) is not vault
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Predecessor execution is foreign, reordered, or test-only"
            )
        predecessor_handles.append(execution.vault_handle)
    _authorize_before_effects(
        store=store, capability=capability, plan=fixed_plan
    )

    if predecessor_handles:
        predecessor_plan = build_acquisition_plan(
            STAGES[expected_prior_count - 1]
        )
        validate_acquisition_bundle(
            predecessor_handles[-1],
            vault=vault,
            store=store,
            capability=capability,
            plan=predecessor_plan,
            predecessor_handles=predecessor_handles[:-1],
        )

    try:
        if _checked_now(clock) >= combined_deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Combined acquisition parent deadline was reached"
            )
        predecessor_bundles = [
            _read_quarantine_for_replay(
                vault,
                handle,
                store=store,
                capability=capability,
            )
            for handle in predecessor_handles
        ]
        if _checked_now(clock) >= combined_deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Combined acquisition parent deadline was reached"
            )
        market_security = _market_security_state(market_transport)
        effective_sec_transport = _PacedSecTransport(
            sec_transport,
            clock=clock,
            sleeper=sleeper,
            deadline=combined_deadline,
        )
        budget = SecCorpusBudget(
            clock=clock,
            max_requests=MAX_SEC_REQUESTS,
            max_bytes=MAX_SEC_BYTES,
            max_seconds=float(MAX_SEC_SECONDS),
        )
        catalog = acquire_official_sec_catalog(
            transport=effective_sec_transport,
            user_agent=sec_user_agent,
            budget=budget,
            session_dates=EXPECTED_SESSIONS,
        )
        universe = build_corpus_universe_manifest(
            catalog_artifact_sha256=catalog.catalog_artifact_sha256,
            calendar_artifact_sha256=build_contract_manifest()["data"][
                "market"
            ]["calendar"]["calendar_dates_sha256"],
            catalog_total_record_count=catalog.catalog_total_record_count,
            catalog_eligible_record_count=catalog.catalog_eligible_record_count,
            session_dates=EXPECTED_SESSIONS,
            records=[dict(record) for record in catalog.universe_records],
        )
        selected = [
            record
            for record in universe["records"]
            if record["artifact_stage"]
            == _STAGE_SPECS[fixed_plan["stage"]]["sec_stage"]
        ]
        selected.sort(
            key=lambda item: (
                item["availability_session"],
                item["accession_number"],
            )
        )
        spec = _STAGE_SPECS[fixed_plan["stage"]]
        if not (
            spec["minimum_filings"]
            <= len(selected)
            <= spec["maximum_model_requests"]
        ):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Official SEC stage universe is incomplete or exceeds its model cap"
            )
        document_plan = [
            {
                "accession_number": item["accession_number"],
                "official_url": _official_primary_url(item),
            }
            for item in selected
        ]
        document_batch = _acquire_authenticated_stage_access_document_batch(
            authenticated_document_plan=document_plan,
            transport=effective_sec_transport,
            user_agent=sec_user_agent,
            budget=budget,
        )
        if len(document_batch.documents) != len(selected):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Official SEC document batch is partial"
            )
        stage_evidence = _build_detached_stage_evidence(
            stage=fixed_plan["stage"],
            universe=universe,
            selected=selected,
            document_batch=document_batch,
        )

        (
            market_responses,
            market_receipts,
            market_elapsed_seconds_hex,
        ) = _acquire_market(
            transport=market_transport,
            plan=fixed_plan,
            clock=clock,
            combined_deadline=combined_deadline,
            security_state=market_security,
        )
        catalog_sources = [
            {"name": item.name, "url": item.url, "body": item.payload}
            for item in catalog.sources
        ]
        documents = [
            {
                "accession_number": record["accession_number"],
                "form": record["form"],
                "availability_session": record["availability_session"],
                "acceptance_datetime": record["acceptance_datetime"],
                "official_url": document.url,
                "body": document.raw_primary_document,
            }
            for record, document in zip(
                selected, document_batch.documents, strict=True
            )
        ]
        carry_in = (
            []
            if not predecessor_bundles
            else _last_documents_by_form(predecessor_bundles[-1])
        )
        model_requests = _build_model_requests(
            documents,
            carry_in_documents=carry_in,
            stage=fixed_plan["stage"],
        )

        sec_receipts: list[dict[str, Any]] = []
        for source, receipt in zip(
            catalog.sources, catalog.request_receipts, strict=True
        ):
            sec_receipts.append(
                _sec_receipt(
                    sequence_number=0,
                    purpose=(
                        "apple_main_submissions"
                        if source.url == MAIN_SUBMISSIONS_URL
                        else "apple_historical_submissions"
                    ),
                    raw_receipt=receipt,
                    payload=source.payload,
                )
            )
        for document, receipt in zip(
            document_batch.documents,
            document_batch.request_receipts,
            strict=True,
        ):
            sec_receipts.append(
                _sec_receipt(
                    sequence_number=0,
                    purpose=(
                        f"{fixed_plan['stage']}_official_primary_document"
                    ),
                    raw_receipt=receipt,
                    payload=document.raw_primary_document,
                )
            )
        all_receipts = sec_receipts + market_receipts
        all_receipts = [
            {
                **{
                    key: value
                    for key, value in receipt.items()
                    if key not in {"sequence_number", "receipt_sha256"}
                },
                "sequence_number": index,
            }
            for index, receipt in enumerate(all_receipts, start=1)
        ]
        all_receipts = [
            {
                **receipt,
                "receipt_sha256": canonical_sha256(receipt),
            }
            for receipt in all_receipts
        ]
        market_commitments, current_rows = _market_commitments(
            {item["symbol"]: item["body"] for item in market_responses},
            fixed_plan,
        )
        universe_event_proofs = _build_universe_event_proofs(
            stage=fixed_plan["stage"],
            universe=universe,
            current_documents=documents,
            predecessor_bundles=predecessor_bundles,
        )
        model_slice = _build_model_slice(
            plan=fixed_plan,
            model_requests=model_requests,
            universe_event_proofs=universe_event_proofs,
        )
        stage_slice = _build_stage_slice(
            plan=fixed_plan,
            rows_by_symbol=current_rows,
            model_requests=model_requests,
            universe_event_proofs=universe_event_proofs,
        )
        try:
            catalog_receipts = json.loads(
                catalog.request_receipts_json.decode("utf-8")
            )
            catalog_artifact = json.loads(
                catalog.artifact_json.decode("utf-8")
            )
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Official SEC catalogue artifacts are not canonical"
            ) from None
        private = {
            "sec_replay_evidence": {
                "catalog_request_receipts": catalog_receipts,
                "catalog_artifact": catalog_artifact,
                "corpus_universe_manifest": universe,
                **stage_evidence,
            },
            "sec_catalog_sources": catalog_sources,
            "sec_primary_documents": documents,
            "carry_in_documents": carry_in,
            "market_responses": market_responses,
            "request_receipts": all_receipts,
            "model_requests": model_requests,
            "model_slice": model_slice,
            "stage_slice": stage_slice,
        }
        for item in (
            *(row["body"] for row in catalog_sources),
            *(row["body"] for row in documents),
            *(row["body"] for row in market_responses),
            *(row["request_bytes"] for row in model_requests),
        ):
            if raw_identity_bytes in item:
                raise SecGemmaOnlineRiskOverlayAcquisitionError(
                    "A provider echoed private SEC identity material"
                )

        if predecessor_bundles:
            predecessor = predecessor_bundles[-1]
            predecessor_plan = build_acquisition_plan(
                fixed_plan["predecessor_stage"]
            )
            predecessor_rows = _market_commitments(
                _bundle_raw_market(predecessor), predecessor_plan
            )[1]
            cutoff = predecessor_plan["market"]["last_value_session"]
            for symbol in YAHOO_SYMBOLS:
                if [
                    row
                    for row in current_rows[symbol]
                    if row["session"] <= cutoff
                ] != [
                    row
                    for row in predecessor_rows[symbol]
                    if row["session"] <= cutoff
                ]:
                    raise SecGemmaOnlineRiskOverlayAcquisitionError(
                        "Market prefix changed after its predecessor was sealed"
                    )
        result = _bundle(
            plan=fixed_plan,
            identity_sha256=identity_sha256,
            predecessor_bundle_sha256=(
                None
                if not predecessor_bundles
                else predecessor_handles[-1].bundle_sha256
            ),
            private=private,
            market_commitments=market_commitments,
            prefix_continuity_verified=True,
        )
        handle = _seal_quarantine(
            vault,
            store=store,
            capability=capability,
            stage=fixed_plan["stage"],
            attempt_id=fixed_plan["attempt_id"],
            bundle_sha256=result["bundle_sha256"],
            manifest_sha256=result["public_manifest"]["manifest_sha256"],
            private_index_sha256=result["private_index_sha256"],
            predecessor_handles=tuple(predecessor_handles),
            quarantine=result,
        )
        verified_report = validate_acquisition_bundle(
            handle,
            vault=vault,
            store=store,
            capability=capability,
            plan=fixed_plan,
            predecessor_handles=predecessor_handles,
        )
        if _checked_now(clock) > combined_deadline:
            raise SecGemmaOnlineRiskOverlayAcquisitionError(
                "Combined SEC and market acquisition deadline was exceeded"
            )
        return AcquisitionExecutionResult(
            vault=vault,
            handle=handle,
            verified_report=verified_report,
            public_summary=_execution_public_summary(
                result,
                production_authority=vault.production_authority,
                market_elapsed_seconds_hex=(
                    market_elapsed_seconds_hex
                ),
            ),
            request_accounting=_execution_request_accounting(
                result,
                market_elapsed_seconds_hex=(
                    market_elapsed_seconds_hex
                ),
            ),
            model_slice_sha256=model_slice["model_slice_sha256"],
            stage_slice_sha256=stage_slice["slice_sha256"],
            _sentinel=_EXECUTION_RESULT_SENTINEL,
        )
    except SecGemmaOnlineRiskOverlayAcquisitionIndeterminate:
        raise
    except Exception:
        raise SecGemmaOnlineRiskOverlayAcquisitionIndeterminate(
            "Acquisition batch was incomplete or indeterminate"
        ) from None


__all__ = [
    "ACQUISITION_BUNDLE_SCHEMA_VERSION",
    "ACQUISITION_MANIFEST_SCHEMA_VERSION",
    "ACQUISITION_PLAN_SCHEMA_VERSION",
    "ACQUISITION_PUBLIC_SUMMARY_SCHEMA_VERSION",
    "ACQUISITION_RECEIPT_SCHEMA_VERSION",
    "ACQUISITION_REQUEST_ACCOUNTING_SCHEMA_VERSION",
    "ACQUISITION_VALIDATION_SCHEMA_VERSION",
    "ACQUISITION_VALIDATION_VERIFIER_ID",
    "BLINDED_MODEL_REQUEST_SCHEMA_VERSION",
    "CONFIRMATION",
    "DEVELOPMENT",
    "FINAL",
    "STAGE_SLICE_SCHEMA_VERSION",
    "STAGES",
    "AcquisitionExecutionResult",
    "SecGemmaOnlineRiskOverlayAcquisitionError",
    "SecGemmaOnlineRiskOverlayAcquisitionIndeterminate",
    "VerifiedAcquisitionReport",
    "acquire_stage_to_quarantine",
    "build_acquisition_plan",
    "create_production_market_transport",
    "is_verified_acquisition_report",
    "rehydrate_contiguous_production_acquisitions",
    "validate_acquisition_bundle",
    "validate_acquisition_plan",
]
